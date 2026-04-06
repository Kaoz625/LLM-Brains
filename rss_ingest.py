#!/usr/bin/env python3
"""
rss_ingest.py — Continuous Feed Ingestion
==========================================
Pulls RSS/Atom feeds, YouTube channels, and web pages into brain/raw/
so compile.py can process them automatically.

Keeps your brain permanently up-to-date with:
  - AI research (arXiv, HuggingFace, Karpathy, etc.)
  - Personal feeds (Obsidian sync, calendar, custom sources)
  - Any domain-specific RSS you care about

Usage:
    python rss_ingest.py              # one-shot pull all feeds
    python rss_ingest.py --watch 30m  # pull every 30 minutes (default: 1h)
    python rss_ingest.py --add-feed "https://example.com/rss" "Label"
    python rss_ingest.py --list-feeds

Config file: brain/rss_feeds.json
"""

import json
import time
import hashlib
import logging
import argparse
import sys
import re
from pathlib import Path
from datetime import datetime
from urllib.request import urlopen, Request
from urllib.parse import urlparse

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

ROOT       = Path(__file__).parent / "brain"
RAW        = ROOT / "raw"
FEEDS_FILE = ROOT / "rss_feeds.json"
SEEN_FILE  = ROOT / ".rss_seen"
LOG_FILE   = ROOT / "rss_ingest.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE) if LOG_FILE.parent.exists() else logging.NullHandler(),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("rss")

# ---------------------------------------------------------------------------
# Default feed list — edit or add via --add-feed
# ---------------------------------------------------------------------------

DEFAULT_FEEDS = [
    # AI Research
    {"url": "https://arxiv.org/rss/cs.AI",   "label": "arXiv-AI",       "category": "knowledge"},
    {"url": "https://arxiv.org/rss/cs.CL",   "label": "arXiv-NLP",      "category": "knowledge"},
    {"url": "https://arxiv.org/rss/cs.LG",   "label": "arXiv-ML",       "category": "knowledge"},

    # AI Labs & Blogs
    {"url": "https://huggingface.co/blog/feed.xml",
     "label": "HuggingFace",   "category": "knowledge"},
    {"url": "https://openai.com/blog/rss/",  "label": "OpenAI-Blog",    "category": "knowledge"},
    {"url": "https://www.anthropic.com/rss", "label": "Anthropic-Blog", "category": "knowledge"},

    # Community
    {"url": "https://www.reddit.com/r/LocalLLaMA/.rss",
     "label": "LocalLLaMA",    "category": "knowledge"},
    {"url": "https://news.ycombinator.com/rss",
     "label": "HackerNews",    "category": "knowledge"},

    # GitHub releases (add specific repos you follow)
    # {"url": "https://github.com/karpathy/nanoGPT/releases.atom",
    #  "label": "karpathy-nanoGPT", "category": "knowledge"},
]

# ---------------------------------------------------------------------------
# Feed management
# ---------------------------------------------------------------------------

def load_feeds() -> list:
    if FEEDS_FILE.exists():
        return json.loads(FEEDS_FILE.read_text())
    return DEFAULT_FEEDS.copy()


def save_feeds(feeds: list):
    ROOT.mkdir(parents=True, exist_ok=True)
    FEEDS_FILE.write_text(json.dumps(feeds, indent=2))


def load_seen() -> set:
    if SEEN_FILE.exists():
        return set(json.loads(SEEN_FILE.read_text()))
    return set()


def save_seen(seen: set):
    SEEN_FILE.write_text(json.dumps(list(seen)))


def url_hash(url: str) -> str:
    return hashlib.sha256(url.encode()).hexdigest()[:16]

# ---------------------------------------------------------------------------
# RSS / Atom parsing (no external dependency required — pure stdlib)
# ---------------------------------------------------------------------------

def parse_feed(url: str) -> list:
    """Parse RSS or Atom feed, return list of {title, link, summary, published}."""
    try:
        req = Request(url, headers={"User-Agent": "LLM-Brains/1.0 RSS Reader"})
        xml = urlopen(req, timeout=15).read().decode("utf-8", errors="replace")
    except Exception as e:
        log.error("Failed to fetch feed %s: %s", url, e)
        return []

    items = []

    # Try feedparser first (much better parsing)
    try:
        import feedparser  # pip install feedparser
        feed = feedparser.parse(xml)
        for entry in feed.entries:
            items.append({
                "title":     getattr(entry, "title", "Untitled"),
                "link":      getattr(entry, "link", ""),
                "summary":   getattr(entry, "summary", ""),
                "published": getattr(entry, "published", ""),
            })
        return items
    except ImportError:
        pass  # fall through to manual parsing

    # Manual XML parsing fallback
    # RSS items
    for item in re.findall(r"<item>(.*?)</item>", xml, re.DOTALL):
        title   = re.search(r"<title[^>]*><!\[CDATA\[(.*?)\]\]>", item, re.DOTALL)
        title   = title.group(1) if title else re.search(r"<title[^>]*>(.*?)</title>", item)
        link    = re.search(r"<link>(.*?)</link>", item)
        desc    = re.search(r"<description[^>]*><!\[CDATA\[(.*?)\]\]>", item, re.DOTALL)
        pubdate = re.search(r"<pubDate>(.*?)</pubDate>", item)
        items.append({
            "title":     (title.group(1) if hasattr(title, 'group') else str(title)).strip(),
            "link":      link.group(1).strip() if link else "",
            "summary":   desc.group(1).strip()[:500] if desc else "",
            "published": pubdate.group(1).strip() if pubdate else "",
        })

    # Atom entries (if no RSS items found)
    if not items:
        for entry in re.findall(r"<entry>(.*?)</entry>", xml, re.DOTALL):
            title   = re.search(r"<title[^>]*>(.*?)</title>", entry, re.DOTALL)
            link    = re.search(r'<link[^>]+href=["\']([^"\']+)["\']', entry)
            summary = re.search(r"<summary[^>]*>(.*?)</summary>", entry, re.DOTALL)
            pubdate = re.search(r"<published>(.*?)</published>", entry)
            items.append({
                "title":     title.group(1).strip() if title else "Untitled",
                "link":      link.group(1).strip() if link else "",
                "summary":   re.sub(r"<[^>]+>", " ", summary.group(1))[:500] if summary else "",
                "published": pubdate.group(1).strip() if pubdate else "",
            })

    return items

# ---------------------------------------------------------------------------
# Write to raw/
# ---------------------------------------------------------------------------

def write_item_to_raw(item: dict, label: str, category: str):
    """Write a feed item as a .txt file in brain/raw/ for compile.py to process."""
    RAW.mkdir(parents=True, exist_ok=True)
    title_slug = re.sub(r"[^a-z0-9]+", "-", item["title"].lower())[:50]
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename   = f"rss_{label}_{timestamp}_{title_slug}.txt"
    filepath   = RAW / filename

    content = f"""Source: {item['link']}
Feed: {label}
Category: {category}
Title: {item['title']}
Published: {item.get('published', 'unknown')}
Date Ingested: {datetime.now().isoformat()}

{item.get('summary', '')}

URL: {item['link']}
"""
    filepath.write_text(content)
    log.info("Wrote: %s", filename)


def write_full_article(url: str, label: str, category: str) -> bool:
    """Optionally fetch the full article content (not just summary)."""
    try:
        req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        html = urlopen(req, timeout=15).read().decode("utf-8", errors="replace")
        # Strip HTML
        text = re.sub(r"<script[^>]*>.*?</script>", " ", html, flags=re.DOTALL)
        text = re.sub(r"<style[^>]*>.*?</style>", " ", text, flags=re.DOTALL)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) < 200:
            return False
        title_slug = re.sub(r"[^a-z0-9]+", "-", urlparse(url).path.strip("/"))[:40]
        timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename   = f"article_{label}_{timestamp}_{title_slug}.txt"
        (RAW / filename).write_text(f"Source: {url}\n\n{text[:8000]}")
        log.info("Full article: %s", filename)
        return True
    except Exception as e:
        log.debug("Could not fetch full article %s: %s", url, e)
        return False

# ---------------------------------------------------------------------------
# Main ingestion loop
# ---------------------------------------------------------------------------

def ingest_all(full_articles: bool = False) -> dict:
    """Pull all feeds and write new items to raw/. Returns stats."""
    feeds = load_feeds()
    seen  = load_seen()
    stats = {"new": 0, "skipped": 0, "feeds": 0, "errors": 0}

    for feed in feeds:
        url      = feed["url"]
        label    = feed.get("label", urlparse(url).netloc)
        category = feed.get("category", "knowledge")
        log.info("Fetching feed: %s", label)

        items = parse_feed(url)
        if not items:
            stats["errors"] += 1
            continue

        stats["feeds"] += 1
        for item in items:
            item_id = url_hash(item.get("link", item["title"]))
            if item_id in seen:
                stats["skipped"] += 1
                continue

            write_item_to_raw(item, label, category)
            if full_articles and item.get("link"):
                write_full_article(item["link"], label, category)

            seen.add(item_id)
            stats["new"] += 1

    save_seen(seen)
    return stats


def watch_feeds(interval_minutes: int = 60, full_articles: bool = False):
    """Continuously pull feeds on a timer."""
    log.info("Watching feeds every %d minutes... (Ctrl+C to stop)", interval_minutes)
    while True:
        try:
            stats = ingest_all(full_articles)
            log.info(
                "Ingest complete: %d new, %d skipped, %d feeds, %d errors",
                stats["new"], stats["skipped"], stats["feeds"], stats["errors"]
            )
            time.sleep(interval_minutes * 60)
        except KeyboardInterrupt:
            log.info("Feed watcher stopped.")
            break

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_interval(s: str) -> int:
    """Parse interval string like '30m', '2h', '1h30m' into minutes."""
    total = 0
    for match in re.finditer(r"(\d+)([hm])", s.lower()):
        val, unit = int(match.group(1)), match.group(2)
        total += val * 60 if unit == "h" else val
    return total or 60  # default 60 minutes


def main():
    parser = argparse.ArgumentParser(
        description="RSS/feed ingester — pulls new content into brain/raw/"
    )
    parser.add_argument("--watch", nargs="?", const="1h", metavar="INTERVAL",
                        help="Watch mode, e.g. --watch 30m (default: 1h)")
    parser.add_argument("--full-articles", action="store_true",
                        help="Fetch full article content, not just summaries")
    parser.add_argument("--add-feed", nargs=2, metavar=("URL", "LABEL"),
                        help="Add a new RSS feed")
    parser.add_argument("--list-feeds", action="store_true",
                        help="List all configured feeds")
    parser.add_argument("--remove-feed", metavar="LABEL",
                        help="Remove a feed by label")
    args = parser.parse_args()

    ROOT.mkdir(parents=True, exist_ok=True)

    if args.add_feed:
        url, label = args.add_feed
        feeds = load_feeds()
        feeds.append({"url": url, "label": label, "category": "knowledge"})
        save_feeds(feeds)
        print(f"Added feed: {label} ({url})")
        return

    if args.list_feeds:
        feeds = load_feeds()
        print(f"\nConfigured feeds ({len(feeds)}):")
        for f in feeds:
            print(f"  [{f.get('category','?')}] {f.get('label','?')} — {f['url']}")
        return

    if args.remove_feed:
        feeds = [f for f in load_feeds() if f.get("label") != args.remove_feed]
        save_feeds(feeds)
        print(f"Removed feed: {args.remove_feed}")
        return

    if args.watch is not None:
        interval = parse_interval(args.watch)
        watch_feeds(interval, args.full_articles)
    else:
        stats = ingest_all(args.full_articles)
        print(f"\n{'='*40}")
        print(f"  RSS Ingest Summary")
        print(f"{'='*40}")
        print(f"  Feeds pulled    : {stats['feeds']}")
        print(f"  New items       : {stats['new']}")
        print(f"  Already seen    : {stats['skipped']}")
        print(f"  Feed errors     : {stats['errors']}")
        print(f"{'='*40}")
        if stats["new"] > 0:
            print(f"\n  {stats['new']} new items in brain/raw/ — run compile.py to process them")


if __name__ == "__main__":
    main()
