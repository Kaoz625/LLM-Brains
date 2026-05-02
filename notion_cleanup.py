"""
Notion cleanup: archive duplicate post pages for the 3 dirty communities,
then reset their synced_posts state so notion_sync.py can re-sync cleanly.

Usage:
  python notion_cleanup.py --dry-run           # show what would be deleted
  python notion_cleanup.py                     # archive pages + clear state
  python notion_cleanup.py --community aiautomationsbyjack  # single community

After running this, run:
  python notion_sync.py
"""

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
STATE_FILE = SCRIPT_DIR / "skool_sync_state.json"
CREDS_FILE = Path.home() / ".credentials" / "api-keys.env"

# Communities with duplicate pages — the 3 dirty ones
DIRTY_COMMUNITIES = {
    "aiautomationsbyjack",
    "ai-seo-with-julian-goldie",
    "ai-seo-mastermind-group",
}


def load_token() -> str:
    import os
    token = os.environ.get("NOTION_API_KEY", "")
    if token:
        return token
    if CREDS_FILE.exists():
        for line in CREDS_FILE.read_text().splitlines():
            if line.startswith("NOTION_API_KEY="):
                return line.split("=", 1)[1].strip()
    print("ERROR: NOTION_API_KEY not set.")
    sys.exit(1)


def archive_page(client, page_id: str, dry_run: bool) -> bool:
    if dry_run:
        return True
    try:
        client.pages.update(page_id=page_id, archived=True)
        return True
    except Exception as e:
        print(f"  WARN: Could not archive {page_id}: {e}")
        return False


def cleanup_community(client, slug: str, state: dict, dry_run: bool):
    notion_cs = state.get(slug, {}).get("notion", {})
    if not notion_cs:
        print(f"  {slug}: no Notion state found, skipping")
        return

    synced_posts = notion_cs.get("synced_posts", {})
    month_pages = notion_cs.get("month_pages", {})

    total = len(synced_posts)
    print(f"\n  {slug}: {total} synced post pages to archive")

    archived = 0
    failed = 0
    for key, page_id in synced_posts.items():
        if not isinstance(page_id, str) or page_id.startswith("dry-run"):
            continue
        if archive_page(client, page_id, dry_run):
            archived += 1
            if archived % 50 == 0:
                print(f"    Archived {archived}/{total}...")
        else:
            failed += 1

    # Archive month bucket pages too
    for month_key, mp_id in month_pages.items():
        if isinstance(mp_id, str) and not mp_id.startswith("dry-run"):
            archive_page(client, mp_id, dry_run)

    # Archive the Posts page itself so it's recreated fresh
    posts_page_id = notion_cs.get("notion_posts_page_id")
    if posts_page_id and not dry_run:
        archive_page(client, posts_page_id, dry_run)

    action = "Would archive" if dry_run else "Archived"
    print(f"  {slug}: {action} {archived} pages ({failed} failed)")

    if not dry_run:
        # Clear posts state — notion_sync.py will recreate cleanly with PostID dedup
        notion_cs.pop("synced_posts", None)
        notion_cs.pop("month_pages", None)
        notion_cs.pop("notion_posts_page_id", None)
        print(f"  {slug}: state cleared — ready for clean re-sync")


def main():
    parser = argparse.ArgumentParser(description="Archive duplicate Notion post pages")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be deleted without making changes")
    parser.add_argument("--community",
                        help="Clean only this community slug (default: all 3 dirty ones)")
    args = parser.parse_args()

    if not STATE_FILE.exists():
        print("ERROR: skool_sync_state.json not found. Run notion_sync.py first.")
        sys.exit(1)

    state = json.loads(STATE_FILE.read_text())

    targets = {args.community} if args.community else DIRTY_COMMUNITIES
    # Verify targets are actually dirty
    for slug in list(targets):
        synced = state.get(slug, {}).get("notion", {}).get("synced_posts", {})
        print(f"  {slug}: {len(synced)} synced pages in state")

    print(f"\n{'DRY RUN — ' if args.dry_run else ''}Cleaning {len(targets)} communities")
    print("  This will archive all post pages and reset synced state.")
    if not args.dry_run:
        confirm = input("  Type 'yes' to continue: ").strip().lower()
        if confirm != "yes":
            print("  Aborted.")
            sys.exit(0)

    from notion_client import Client
    client = Client(auth=load_token())

    for slug in targets:
        cleanup_community(client, slug, state, args.dry_run)

    if not args.dry_run:
        STATE_FILE.write_text(json.dumps(state, indent=2))
        print("\nState saved. Now run:")
        print("  python notion_sync.py")
        print("to re-sync with correct PostID dedup keys.")
    else:
        print("\nDry run complete. Run without --dry-run to archive pages.")


if __name__ == "__main__":
    main()
