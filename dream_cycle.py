"""
Dream Cycle: promote episodic memory → semantic memory.

Reads brain/raw/ files older than 24h, uses Claude to categorize and summarize
each entry, then writes the result to the appropriate brain/ output directory:
  brain/me/        — personal identity, habits, experiences
  brain/knowledge/ — concepts, learning, research
  brain/work/      — projects, tools, tasks, goals
  brain/media/     — videos, podcasts, articles consumed

Skips the brain/raw/skool/ subtree (handled by notion_sync).

Usage:
  python dream_cycle.py               # promote all eligible raw files
  python dream_cycle.py --dry-run     # show what would be promoted
  python dream_cycle.py --age-hours 0 # promote everything regardless of age

Launchd plist: ~/Library/LaunchAgents/com.llmbrains.dreamcycle.plist
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
RAW_DIR = SCRIPT_DIR / "brain" / "raw"
STATE_FILE = SCRIPT_DIR / "dream_cycle_state.json"
CREDS_FILE = Path.home() / ".credentials" / "api-keys.env"

BRAIN_DIRS = {
    "me":        SCRIPT_DIR / "brain" / "me",
    "knowledge": SCRIPT_DIR / "brain" / "knowledge",
    "work":      SCRIPT_DIR / "brain" / "work",
    "media":     SCRIPT_DIR / "brain" / "media",
}

# Skool subtree is managed by notion_sync — skip it
SKIP_SUBTREES = {"skool"}

# ── which brain answers ────────────────────────────────────────────────────────
# DEFAULT IS LITELLM ON mac2, NOT THE ANTHROPIC API. Every promotion from
# 2026-03 to 2026-08-29 answered "Your credit balance is too low to access the
# Anthropic API", so this script has been promoting 0 entries a night, silently,
# for months (register MK-26). The credit is Markus's to top up; routing through
# the proxy the fleet already runs costs nothing and works today.
#
# LiteLLM speaks the Anthropic wire format at /v1/messages, so the anthropic SDK
# below needs no change beyond a base_url — the reply still arrives as
# content[0].text. Verified live on 2026-08-29 against free-gpt-oss-120b:
#   POST /v1/messages -> {"content":[{"type":"text","text":"{\"ok\":true}"}],
#                         "stop_reason":"end_turn"}
#
# Set DREAM_CYCLE_DIRECT=1 to go straight to api.anthropic.com again once that
# account has credit.
LITELLM_BASE_URL = "http://100.88.99.116:4000"

# A FREE model, deliberately. The fleet's stated mission is free tiers, and the
# whole point of this change is to stop depending on a balance. Measured on
# 2026-08-29: all eight free-* aliases in mac2's ~/litellm/config.yaml answer as
# the SAME backend, openai/gpt-oss-120b served by Groq — the openrouter :free
# routes those aliases name are not what actually replies. So naming a different
# free alias here changes nothing today; it is written as an env override
# because that collapse is a config bug on mac2, not a decision.
MODEL = os.environ.get("DREAM_CYCLE_MODEL", "free-gpt-oss-120b")
DIRECT_MODEL = "claude-haiku-4-5-20251001"

# 1024 was enough for a non-reasoning haiku. gpt-oss-120b is a REASONING model
# and its reasoning tokens are billed against this same budget while never
# appearing in content — at max_tokens 40 a live call spent 38 of them thinking
# and returned an EMPTY string with finish_reason "length". An empty string
# reaches json.loads() below and raises, which would have swapped "no credit"
# for "invalid JSON" and still promoted 0. Give the answer real room.
MAX_TOKENS = 4096


# ── credentials ────────────────────────────────────────────────────────────────
def read_cred(name: str) -> str:
    """Env first, then ~/.credentials/api-keys.env. Never a literal in this file."""
    val = os.environ.get(name, "")
    if val:
        return val
    if CREDS_FILE.exists():
        for line in CREDS_FILE.read_text().splitlines():
            if line.startswith(f"{name}="):
                return line.split("=", 1)[1].strip()
    return ""


def build_client():
    """Return (client, model). LiteLLM unless DREAM_CYCLE_DIRECT=1 asks otherwise."""
    from anthropic import Anthropic

    if os.environ.get("DREAM_CYCLE_DIRECT") == "1":
        key = read_cred("ANTHROPIC_API_KEY")
        if not key:
            print("ERROR: DREAM_CYCLE_DIRECT=1 but ANTHROPIC_API_KEY is not set.")
            sys.exit(1)
        print(f"Model: {DIRECT_MODEL} (direct to api.anthropic.com)")
        return Anthropic(api_key=key), DIRECT_MODEL

    key = read_cred("LITELLM_MASTER_KEY")
    if not key:
        print("ERROR: LITELLM_MASTER_KEY not set. Add it to ~/.credentials/api-keys.env,")
        print("       or set DREAM_CYCLE_DIRECT=1 to use the Anthropic API instead.")
        sys.exit(1)
    base = read_cred("LITELLM_BASE_URL") or LITELLM_BASE_URL
    print(f"Model: {MODEL} via LiteLLM at {base}")
    return Anthropic(api_key=key, base_url=base), MODEL


# ── state ──────────────────────────────────────────────────────────────────────
def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {"promoted": {}}


def save_state(state: dict):
    STATE_FILE.write_text(json.dumps(state, indent=2))


# ── file helpers ───────────────────────────────────────────────────────────────
def parse_header(filepath: Path) -> tuple[dict, str]:
    text = filepath.read_text(encoding="utf-8", errors="ignore")
    parts = text.split("\n---\n", 1)
    if len(parts) < 2:
        return {}, text
    meta = {}
    for line in parts[0].strip().splitlines():
        if ": " in line:
            k, v = line.split(": ", 1)
            meta[k.strip()] = v.strip()
    return meta, parts[1].strip()


def slugify(text: str) -> str:
    s = re.sub(r"[^\w\s-]", "", text.lower())
    s = re.sub(r"[\s_]+", "-", s)
    return re.sub(r"-+", "-", s).strip("-")[:60] or "entry"


def collect_raw_files(age_hours: float) -> list[Path]:
    cutoff = datetime.now(timezone.utc) - timedelta(hours=age_hours)
    files = []
    for p in sorted(RAW_DIR.rglob("*.md")):
        # Skip skool subtree
        rel = p.relative_to(RAW_DIR)
        if rel.parts and rel.parts[0] in SKIP_SUBTREES:
            continue
        # Age check (use file mtime as fallback)
        try:
            mtime = datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc)
        except OSError:
            continue
        if mtime < cutoff:
            files.append(p)
    return files


# ── Claude categorization ──────────────────────────────────────────────────────
_SYSTEM = """\
You are a personal knowledge assistant. Given a raw episodic memory entry, you:
1. Assign a category: exactly one of: me, knowledge, work, media
   - me: personal experiences, habits, reflections, identity
   - knowledge: concepts learned, research, ideas, techniques
   - work: projects, tools, tasks, goals, professional activity
   - media: videos, podcasts, articles, books consumed
2. Write a concise semantic summary (3-8 sentences) that distills the key insight
   or information worth retaining long-term. Use plain prose, no bullet points.
3. Extract a short title (max 8 words).

Respond ONLY with valid JSON:
{"category": "...", "title": "...", "summary": "..."}
"""


def promote_with_claude(client, model: str, content: str, meta: dict) -> dict | None:
    source = meta.get("SOURCE", "")
    source_type = meta.get("Type", "")
    context = f"Source: {source}\nType: {source_type}\n\n" if source else ""
    user_content = context + content[:4000]  # stay within haiku context

    try:
        msg = client.messages.create(
            model=model,
            max_tokens=MAX_TOKENS,
            system=_SYSTEM,
            messages=[{"role": "user", "content": user_content}],
        )
        # A reasoning model can return an EMPTY content list, or one text block
        # holding "", when the whole token budget went to reasoning. Say that
        # plainly instead of dying inside json.loads on an empty string — a
        # truncated answer and a broken prompt are different problems and the
        # log has to tell them apart.
        blocks = [b for b in (msg.content or []) if getattr(b, "type", "") == "text"]
        raw = blocks[0].text.strip() if blocks else ""
        if not raw:
            reason = getattr(msg, "stop_reason", "?")
            print(f"    WARN: empty answer (stop_reason={reason}); "
                  f"raise MAX_TOKENS if this says max_tokens")
            return None
        # Strip markdown code fences if present
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.MULTILINE).strip()
        result = json.loads(raw)
        if result.get("category") not in BRAIN_DIRS:
            result["category"] = "knowledge"
        return result
    except Exception as e:
        print(f"    WARN: Claude error: {e}")
        return None


# ── write promoted entry ───────────────────────────────────────────────────────
def write_promoted(category: str, title: str, summary: str,
                   source_file: Path, meta: dict) -> Path:
    now = datetime.now(timezone.utc)
    ts = now.strftime("%Y-%m-%d-%H-%M")
    slug = slugify(title)
    out_dir = BRAIN_DIRS[category]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{ts}-{slug}.md"

    header_lines = [
        f"SOURCE: dream_cycle",
        f"Category: {category}",
        f"Title: {title}",
        f"OriginalFile: {source_file.name}",
        f"Promoted: {now.isoformat()}",
    ]
    if meta.get("SOURCE"):
        header_lines.append(f"OriginalSource: {meta['SOURCE']}")

    out_path.write_text(
        "\n".join(header_lines) + "\n---\n" + summary + "\n",
        encoding="utf-8"
    )
    return out_path


# ── main loop ──────────────────────────────────────────────────────────────────
def run(age_hours: float, dry_run: bool, force: bool):
    client, model = build_client()

    state = load_state()
    promoted_set = set(state["promoted"].keys())

    files = collect_raw_files(age_hours)
    eligible = [f for f in files if force or str(f) not in promoted_set]

    print(f"Raw files older than {age_hours}h: {len(files)}")
    print(f"Eligible for promotion: {len(eligible)}")
    if not eligible:
        print("Nothing to promote.")
        return

    promoted = 0
    skipped = 0

    for raw_file in eligible:
        meta, content = parse_header(raw_file)
        if not content.strip():
            skipped += 1
            continue

        if dry_run:
            print(f"  [dry-run] Would promote: {raw_file.name}")
            continue

        print(f"  Promoting: {raw_file.name[:70]}")
        result = promote_with_claude(client, model, content, meta)
        if not result:
            skipped += 1
            continue

        category = result["category"]
        title = result.get("title", raw_file.stem[:60])
        summary = result.get("summary", content[:500])

        out_path = write_promoted(category, title, summary, raw_file, meta)
        state["promoted"][str(raw_file)] = {
            "category": category,
            "title": title,
            "output": str(out_path),
            "promoted_at": datetime.now(timezone.utc).isoformat(),
        }
        save_state(state)
        promoted += 1
        print(f"    → {category}: {title[:60]}")
        time.sleep(0.3)  # avoid haiku rate limits

    print(f"\nDone. Promoted: {promoted}  Skipped: {skipped}")


def main():
    parser = argparse.ArgumentParser(description="Promote episodic → semantic memory")
    parser.add_argument("--age-hours", type=float, default=24.0,
                        help="Minimum age in hours to be eligible (default: 24)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be promoted without making changes")
    parser.add_argument("--force", action="store_true",
                        help="Re-promote already-promoted files")
    args = parser.parse_args()
    run(age_hours=args.age_hours, dry_run=args.dry_run, force=args.force)


if __name__ == "__main__":
    main()
