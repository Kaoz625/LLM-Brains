#!/usr/bin/env python3
"""
Orchestrates full Skool scrape + Notion sync for all communities.
Designed to run daily via launchd.

Usage:
  python skool_auto_sync.py              # all communities, incremental
  python skool_auto_sync.py --dry-run
  python skool_auto_sync.py --community aiautomationsbyjack
  python skool_auto_sync.py --full-history   # re-scrape all post history
"""

import argparse
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
LOG_FILE = Path.home() / ".skool" / "sync.log"
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

PYTHON = sys.executable


def log(msg: str):
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    line = f"[{ts}] {msg}"
    print(line)
    with LOG_FILE.open("a") as f:
        f.write(line + "\n")


def run(cmd: list[str], dry_run: bool) -> int:
    log(f"RUN: {' '.join(cmd)}")
    if dry_run:
        log("  (dry-run, skipped)")
        return 0
    result = subprocess.run(cmd, cwd=SCRIPT_DIR)
    log(f"  exit code: {result.returncode}")
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="Daily Skool scrape + Notion sync")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--community", help="Limit to one community slug")
    parser.add_argument("--full-history", action="store_true",
                        help="Scrape full post history (slow, use for first run)")
    args = parser.parse_args()

    log("=" * 60)
    log("Skool auto-sync starting")

    comm_args = ["--community", args.community] if args.community else []

    # Step 1: Classroom for all communities (incremental, no video/files)
    log("Step 1: Scraping classrooms…")
    rc = run(
        [PYTHON, "skool_scraper.py", "--classroom-only", "--no-video", "--no-files"] + comm_args,
        args.dry_run,
    )
    if rc != 0:
        log("WARNING: classroom scrape exited non-zero, continuing…")

    time.sleep(2)

    # Step 2: Posts (incremental by default; full history on request)
    log("Step 2: Scraping posts…")
    post_args = ["--posts-only", "--max-pages", "700" if args.full_history else "50"]
    if args.full_history:
        post_args.append("--full-history")
    rc = run([PYTHON, "skool_scraper.py"] + post_args + comm_args, args.dry_run)
    if rc != 0:
        log("WARNING: posts scrape exited non-zero, continuing…")

    time.sleep(2)

    # Step 3: Notion sync
    log("Step 3: Syncing to Notion…")
    notion_args = ["--dry-run"] if args.dry_run else []
    if args.community:
        notion_args += ["--community", args.community]
    rc = run([PYTHON, "notion_sync.py"] + notion_args, False)
    if rc != 0:
        log("ERROR: Notion sync failed")

    log("Auto-sync complete")
    log("=" * 60)


if __name__ == "__main__":
    main()
