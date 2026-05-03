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

MODEL = "claude-haiku-4-5-20251001"
MAX_TOKENS = 1024


# ── credentials ────────────────────────────────────────────────────────────────
def load_anthropic_key() -> str:
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if key:
        return key
    if CREDS_FILE.exists():
        for line in CREDS_FILE.read_text().splitlines():
            if line.startswith("ANTHROPIC_API_KEY="):
                return line.split("=", 1)[1].strip()
    print("ERROR: ANTHROPIC_API_KEY not set. Add it to ~/.credentials/api-keys.env")
    sys.exit(1)


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


def promote_with_claude(client, content: str, meta: dict) -> dict | None:
    source = meta.get("SOURCE", "")
    source_type = meta.get("Type", "")
    context = f"Source: {source}\nType: {source_type}\n\n" if source else ""
    user_content = context + content[:4000]  # stay within haiku context

    try:
        msg = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=_SYSTEM,
            messages=[{"role": "user", "content": user_content}],
        )
        raw = msg.content[0].text.strip()
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
    from anthropic import Anthropic

    api_key = load_anthropic_key()
    client = Anthropic(api_key=api_key)

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
        result = promote_with_claude(client, content, meta)
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
