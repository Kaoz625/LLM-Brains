"""
NotebookLM sync: pushes scraped Skool content into per-community notebooks.

Usage:
  python notebooklm_sync.py          # sync all communities
  python notebooklm_sync.py --community aiautomationsbyjack
  python notebooklm_sync.py --dry-run  # show what would be synced

Requires: pip install "notebooklm-py[browser]"
One-time login: python ~/.claude/skills/notebooklm-skill/scripts/nlm.py login

Sources uploaded per community:
  - Each lesson/post .md file as an individual text source
  - PDF/DOCX/EPUB/CSV files via add-source --file
  - YouTube video URLs via add-source --url (NotebookLM processes natively)
  - .vtt/.srt subtitle files as transcript text sources
  - Video .info.json metadata as text sources (non-YouTube)
"""

import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
COMMUNITIES_FILE = SCRIPT_DIR / "skool_communities.json"
STATE_FILE = SCRIPT_DIR / "skool_sync_state.json"
RAW_DIR = SCRIPT_DIR / "brain" / "raw"
SKOOL_DIR = RAW_DIR / "skool"
NLM_SCRIPT = Path.home() / ".claude" / "skills" / "notebooklm-skill" / "scripts" / "nlm.py"

UPLOADABLE_EXTS = {".pdf", ".docx", ".epub", ".csv", ".txt"}
YOUTUBE_PATTERN = re.compile(r'https?://(?:www\.)?(?:youtube\.com/watch\?v=|youtu\.be/)[\w-]+')


def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {}


def save_state(state: dict):
    STATE_FILE.write_text(json.dumps(state, indent=2))


def run_nlm(args: list[str]) -> tuple[int, str]:
    cmd = [sys.executable, str(NLM_SCRIPT)] + args
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode, (result.stdout + result.stderr).strip()


def get_or_create_notebook(name: str, cs: dict, dry_run: bool) -> str | None:
    nb_id = cs.get("notebooklm_notebook_id")
    if nb_id:
        return nb_id

    if dry_run:
        print(f"  [dry-run] Would create notebook: {name}")
        return "dry-run-id"

    print(f"  Creating notebook: {name}")
    code, out = run_nlm(["create", name])
    if code != 0:
        print(f"  ERROR creating notebook: {out}")
        return None

    match = re.search(r'([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})', out)
    if not match:
        print(f"  ERROR: Could not parse notebook ID from: {out}")
        return None

    nb_id = match.group(0)
    cs["notebooklm_notebook_id"] = nb_id
    print(f"  Notebook created: {nb_id}")
    return nb_id


def parse_header(filepath: Path) -> tuple[dict, str]:
    """Split scraped .md file into (metadata dict, content string)."""
    text = filepath.read_text(encoding="utf-8")
    parts = text.split("\n---\n", 1)
    if len(parts) < 2:
        return {}, text
    meta = {}
    for line in parts[0].strip().splitlines():
        if ": " in line:
            k, v = line.split(": ", 1)
            meta[k.strip()] = v.strip()
    return meta, parts[1].strip()


def source_title(meta: dict, filepath: Path, community_name: str) -> str:
    """Derive a readable title for a NotebookLM source."""
    lesson = meta.get("Lesson") or meta.get("Title") or meta.get("PostID")
    course = meta.get("Course")
    if lesson and course:
        return f"{community_name} — {course} — {lesson}"[:100]
    if lesson:
        return f"{community_name} — {lesson}"[:100]
    return f"{community_name} — {filepath.stem}"[:100]


def upload_text_source(nb_id: str, title: str, content: str, dry_run: bool) -> bool:
    if dry_run:
        print(f"  [dry-run] text: {title[:80]} ({len(content):,} chars)")
        return True
    print(f"  Uploading text: {title[:80]}")
    code, out = run_nlm(["add-source", "--notebook-id", nb_id, "--title", title, "--text", content])
    if code != 0:
        print(f"  ERROR: {out}")
        return False
    return True


def upload_file_source(nb_id: str, filepath: Path, label: str, dry_run: bool) -> bool:
    if dry_run:
        print(f"  [dry-run] file: {label}")
        return True
    print(f"  Uploading file: {label}")
    code, out = run_nlm(["add-source", "--notebook-id", nb_id, "--file", str(filepath)])
    if code != 0:
        print(f"  ERROR: {out}")
        return False
    return True


def upload_url_source(nb_id: str, url: str, dry_run: bool) -> bool:
    if dry_run:
        print(f"  [dry-run] url: {url[:80]}")
        return True
    print(f"  Uploading URL: {url[:80]}")
    code, out = run_nlm(["add-source", "--notebook-id", nb_id, "--url", url])
    if code != 0:
        print(f"  ERROR: {out}")
        return False
    return True


def sync_community(slug: str, community_name: str, state: dict, dry_run: bool):
    print(f"\n{'='*60}")
    print(f"Syncing: {community_name} ({slug})")
    print(f"{'='*60}")

    cs = state.setdefault(slug, {"classroom": {}, "posts": {}, "last_sync": None})
    nb_id = get_or_create_notebook(community_name, cs, dry_run)
    if not nb_id:
        return

    synced = cs.setdefault("nlm_sources", {})
    new_count = 0
    community_dir = SKOOL_DIR / slug

    if not community_dir.exists():
        print(f"  No scraped content found at {community_dir}")
        return

    # ── 1. Lesson + post .md files ─────────────────────────────────────────────
    for md_file in sorted(community_dir.rglob("*.md")):
        key = f"md:{md_file.relative_to(SKOOL_DIR)}"
        if key in synced:
            continue

        meta, content = parse_header(md_file)
        if not content or len(content) < 50:
            continue

        title = source_title(meta, md_file, community_name)
        if upload_text_source(nb_id, title, content, dry_run):
            synced[key] = {"uploaded": datetime.now(timezone.utc).isoformat()}
            if not dry_run:
                save_state(state)
            new_count += 1

    # ── 2. Downloadable files (PDF, DOCX, EPUB, CSV) ───────────────────────────
    for f in sorted(community_dir.rglob("files/*")):
        if f.suffix.lower() not in UPLOADABLE_EXTS:
            continue
        key = f"file:{f.relative_to(SKOOL_DIR)}"
        if key in synced:
            continue

        label = f"{community_name} — {f.parent.parent.name} — {f.name}"
        if upload_file_source(nb_id, f, label, dry_run):
            synced[key] = {"uploaded": datetime.now(timezone.utc).isoformat()}
            if not dry_run:
                save_state(state)
            new_count += 1

    # ── 3. YouTube URLs from video info.json ───────────────────────────────────
    for info_file in sorted(community_dir.rglob("videos/**/*.info.json")):
        try:
            info = json.loads(info_file.read_text())
        except Exception:
            continue

        webpage_url = info.get("webpage_url", "")
        if YOUTUBE_PATTERN.match(webpage_url):
            key = f"yt:{webpage_url}"
            if key not in synced:
                if upload_url_source(nb_id, webpage_url, dry_run):
                    synced[key] = {"uploaded": datetime.now(timezone.utc).isoformat()}
                    if not dry_run:
                        save_state(state)
                    new_count += 1
        else:
            # Non-YouTube: upload title + description as text
            key = f"video-info:{info_file.relative_to(SKOOL_DIR)}"
            if key in synced:
                continue
            title = info.get("title", info_file.stem)
            desc = info.get("description", "")
            text = f"# {title}\n\n{desc}" if desc else f"# {title}"
            label = f"{community_name} — Video: {title[:60]}"
            if upload_text_source(nb_id, label, text, dry_run):
                synced[key] = {"uploaded": datetime.now(timezone.utc).isoformat()}
                if not dry_run:
                    save_state(state)
                new_count += 1

    # ── 4. VTT/SRT subtitle files (transcripts) ────────────────────────────────
    for vtt_file in sorted(community_dir.rglob("videos/**/*.vtt") ):
        key = f"transcript:{vtt_file.relative_to(SKOOL_DIR)}"
        if key in synced:
            continue

        content = vtt_file.read_text(encoding="utf-8", errors="ignore")
        # Strip VTT header/timestamps, keep spoken text
        lines = []
        for line in content.splitlines():
            if line.startswith("WEBVTT") or re.match(r'^\d{2}:\d{2}', line) or line == "":
                continue
            lines.append(line)
        transcript_text = "\n".join(lines).strip()
        if len(transcript_text) < 100:
            continue

        title = f"{community_name} — Transcript: {vtt_file.stem[:60]}"
        if upload_text_source(nb_id, title, transcript_text, dry_run):
            synced[key] = {"uploaded": datetime.now(timezone.utc).isoformat()}
            if not dry_run:
                save_state(state)
            new_count += 1

    # Also check for .srt files
    for srt_file in sorted(community_dir.rglob("videos/**/*.srt")):
        key = f"transcript:{srt_file.relative_to(SKOOL_DIR)}"
        if key in synced:
            continue

        content = srt_file.read_text(encoding="utf-8", errors="ignore")
        lines = [l for l in content.splitlines()
                 if not re.match(r'^\d+$', l.strip()) and not re.match(r'^\d{2}:\d{2}', l) and l.strip()]
        transcript_text = "\n".join(lines).strip()
        if len(transcript_text) < 100:
            continue

        title = f"{community_name} — Transcript: {srt_file.stem[:60]}"
        if upload_text_source(nb_id, title, transcript_text, dry_run):
            synced[key] = {"uploaded": datetime.now(timezone.utc).isoformat()}
            if not dry_run:
                save_state(state)
            new_count += 1

    if new_count == 0:
        print(f"  All content already synced")
    else:
        action = "Would sync" if dry_run else "Synced"
        print(f"  {action} {new_count} new source(s)")


def main():
    if not NLM_SCRIPT.exists():
        print(f"ERROR: nlm.py not found at {NLM_SCRIPT}")
        sys.exit(1)

    import argparse
    parser = argparse.ArgumentParser(description="Sync scraped Skool content to NotebookLM")
    parser.add_argument("--community", help="Sync only this community slug")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    communities = json.loads(COMMUNITIES_FILE.read_text())["communities"]
    if args.community:
        communities = [c for c in communities if c["slug"] == args.community]
        if not communities:
            print(f"ERROR: Community '{args.community}' not found")
            sys.exit(1)

    state = load_state()
    for community in communities:
        sync_community(community["slug"], community["name"], state, args.dry_run)
    save_state(state)
    print("\nDone.")


if __name__ == "__main__":
    main()
