#!/usr/bin/env python3
"""
rss_ingest.py — Continuous Feed Ingestion
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

rss_ingest.py — RSS/Atom feed ingestion for AI research papers, blogs, and news.

Fetches new items since last run, stores article text to brain/raw/ for compile.py.

Usage:
    python rss_ingest.py           # one-shot
    python rss_ingest.py --watch   # hourly polling
    python rss_ingest.py --add-feed "https://example.com/feed"
    python rss_ingest.py --list    # list configured feeds
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import feedparser
import requests
from dotenv import load_dotenv

load_dotenv()

BRAIN_DIR = Path(os.getenv("BRAIN_DIR", "./brain"))
RAW_DIR = BRAIN_DIR / "raw"
RSS_STATE_FILE = BRAIN_DIR / "rss_state.json"

DEFAULT_FEEDS = [
    {
        "name": "arXiv AI",
        "url": "http://arxiv.org/rss/cs.AI",
        "category": "ai_research",
    },
    {
        "name": "arXiv ML",
        "url": "http://arxiv.org/rss/cs.LG",
        "category": "ai_research",
    },
    {
        "name": "HuggingFace Papers",
        "url": "https://huggingface.co/blog/feed.xml",
        "category": "ai_research",
    },
    {
        "name": "Anthropic Blog",
        "url": "https://www.anthropic.com/blog/rss",
        "category": "ai_research",
    },
    {
        "name": "OpenAI Blog",
        "url": "https://openai.com/blog/rss",
        "category": "ai_research",
    },
    {
        "name": "HackerNews AI",
        "url": "https://hnrss.org/newest?q=AI+LLM+machine+learning&count=20",
        "category": "tech_news",
    },
    {
        "name": "Karpathy Blog",
        "url": "http://karpathy.github.io/feed.xml",
        "category": "ai_research",
    },
    {
        "name": "LessWrong",
        "url": "https://www.lesswrong.com/feed.xml?view=frontpage",
        "category": "philosophy",
    },
    {
        "name": "Papers With Code",
        "url": "https://paperswithcode.com/latest.rss",
        "category": "ai_research",
    },
    {
        "name": "The Gradient",
        "url": "https://thegradient.pub/rss/",
        "category": "ai_research",
    },
]


# ---------------------------------------------------------------------------
# State management
# ---------------------------------------------------------------------------

def load_state() -> dict:
    """Load RSS state (last fetch times, known GUIDs)."""
    if RSS_STATE_FILE.exists():
        try:
            return json.loads(RSS_STATE_FILE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    return {"feeds": {}, "custom_feeds": [], "last_run": None}


def save_state(state: dict):
    RSS_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    RSS_STATE_FILE.write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")


# ---------------------------------------------------------------------------
# Feed fetching
# ---------------------------------------------------------------------------

def fetch_feed(feed_url: str, timeout: int = 30) -> Optional[feedparser.FeedParserDict]:
    """Fetch and parse an RSS/Atom feed."""
    try:
        # Use requests for better error handling, then parse
        headers = {
            "User-Agent": "LLM-Brains RSS Reader/1.0 (personal knowledge manager)"
        }
        resp = requests.get(feed_url, headers=headers, timeout=timeout)
        resp.raise_for_status()
        parsed = feedparser.parse(resp.content)
        return parsed
    except requests.exceptions.RequestException as e:
        print(f"    Network error fetching {feed_url}: {e}", file=sys.stderr)
    except Exception as e:
        print(f"    Parse error for {feed_url}: {e}", file=sys.stderr)
    return None


def get_entry_id(entry: dict) -> str:
    """Get a unique identifier for a feed entry."""
    return entry.get("id") or entry.get("link") or entry.get("title", "")


def get_entry_date(entry: dict) -> Optional[datetime]:
    """Extract publication date from feed entry."""
    for field in ("published_parsed", "updated_parsed", "created_parsed"):
        val = entry.get(field)
        if val:
            try:
                import calendar
                ts = calendar.timegm(val)
                return datetime.fromtimestamp(ts, tz=timezone.utc)
            except Exception:
                pass
    return None


def extract_article_text(entry: dict) -> str:
    """Extract readable text from a feed entry."""
    parts = []

    title = entry.get("title", "")
    if title:
        parts.append(f"# {title}")

    link = entry.get("link", "")
    if link:
        parts.append(f"URL: {link}")

    date = get_entry_date(entry)
    if date:
        parts.append(f"Date: {date.strftime('%Y-%m-%d')}")

    authors = entry.get("authors", [])
    if authors:
        author_names = ", ".join(a.get("name", "") for a in authors)
        parts.append(f"Authors: {author_names}")

    # Try full content first, then summary
    content = ""
    if entry.get("content"):
        for c in entry["content"]:
            if c.get("value"):
                raw = c["value"]
                # Strip HTML tags
                content = re.sub(r"<[^>]+>", " ", raw)
                content = re.sub(r"\s+", " ", content).strip()
                break

    if not content and entry.get("summary"):
        content = re.sub(r"<[^>]+>", " ", entry["summary"])
        content = re.sub(r"\s+", " ", content).strip()

    if content:
        parts.append(f"\n{content}")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Processing new entries
# ---------------------------------------------------------------------------

def process_new_entries(feed_info: dict, entries: list[dict],
                         known_ids: set, state: dict) -> list[str]:
    """Process new (unseen) feed entries and write to brain/raw/."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    saved_files = []
    feed_name = feed_info["name"]
    category = feed_info.get("category", "knowledge")

    for entry in entries:
        entry_id = get_entry_id(entry)
        if not entry_id or entry_id in known_ids:
            continue

        title = entry.get("title", "Untitled")
        link = entry.get("link", "")
        date = get_entry_date(entry)
        date_str = date.strftime("%Y%m%d") if date else datetime.now().strftime("%Y%m%d")

        # Extract article text
        article_text = extract_article_text(entry)
        if len(article_text.strip()) < 50:
            continue  # Skip nearly empty entries

        # Add metadata header
        full_text = (
            f"---\n"
            f"source_feed: {feed_name}\n"
            f"category: {category}\n"
            f"url: {link}\n"
            f"fetched: {datetime.now().isoformat()}\n"
            f"---\n\n"
            f"{article_text}\n"
        )

        # Write to raw directory
        safe_title = re.sub(r"[^\w\s-]", "", title)[:60].strip().replace(" ", "_")
        filename = f"{date_str}_{feed_name.replace(' ', '_')}_{safe_title}.txt"
        out_path = RAW_DIR / filename

        out_path.write_text(full_text, encoding="utf-8")
        saved_files.append(str(out_path))
        known_ids.add(entry_id)

    return saved_files


# ---------------------------------------------------------------------------
# Main ingest logic
# ---------------------------------------------------------------------------

def run_once(verbose: bool = True) -> dict:
    """Fetch all configured feeds once. Returns stats."""
    state = load_state()
    all_feeds = DEFAULT_FEEDS + state.get("custom_feeds", [])
    stats = {
        "feeds_checked": 0,
        "new_entries": 0,
        "saved_files": 0,
        "errors": 0,
    }

    for feed_info in all_feeds:
        feed_url = feed_info["url"]
        feed_name = feed_info["name"]
        feed_state = state["feeds"].setdefault(feed_url, {"known_ids": [], "last_fetch": None})
        known_ids = set(feed_state.get("known_ids", []))

        if verbose:
            print(f"  Fetching: {feed_name}")

        parsed = fetch_feed(feed_url)
        if parsed is None:
            stats["errors"] += 1
            continue

        stats["feeds_checked"] += 1
        entries = parsed.get("entries", [])

        # Filter to new entries
        new_entries = []
        for entry in entries:
            entry_id = get_entry_id(entry)
            if entry_id and entry_id not in known_ids:
                new_entries.append(entry)

        if new_entries:
            if verbose:
                print(f"    {len(new_entries)} new items")
            saved = process_new_entries(feed_info, new_entries, known_ids, state)
            stats["new_entries"] += len(new_entries)
            stats["saved_files"] += len(saved)

            # Update state
            feed_state["known_ids"] = list(known_ids)
            feed_state["last_fetch"] = datetime.now().isoformat()
        elif verbose:
            print(f"    No new items")

    state["last_run"] = datetime.now().isoformat()
    save_state(state)
    return stats


def watch_mode(interval_hours: int = 1, verbose: bool = True):
    """Poll feeds at regular intervals."""
    print(f"\nRSS watch mode — polling every {interval_hours}h")
    print("Press Ctrl+C to stop.\n")
    interval_secs = interval_hours * 3600

    while True:
        print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M')}] Fetching feeds...")
        stats = run_once(verbose=verbose)
        print(f"  -> {stats['feeds_checked']} feeds, {stats['new_entries']} new items, "
              f"{stats['saved_files']} files saved")
        if stats["new_entries"] > 0:
            print(f"  -> Run 'python compile.py' to process new articles")
        try:
            time.sleep(interval_secs)
        except KeyboardInterrupt:
            print("\nWatch mode stopped.")
            break


def add_custom_feed(url: str, name: Optional[str] = None, category: str = "knowledge"):
    """Add a custom feed to the configuration."""
    state = load_state()
    custom_feeds = state.setdefault("custom_feeds", [])

    # Check for duplicates
    existing_urls = [f["url"] for f in custom_feeds]
    if url in existing_urls:
        print(f"Feed already configured: {url}")
        return

    if not name:
        name = urlparse(url).netloc

    # Test the feed
    print(f"Testing feed: {url}")
    parsed = fetch_feed(url)
    if parsed is None:
        print(f"Could not fetch feed: {url}")
        return

    feed_title = parsed.feed.get("title", name)
    entry_count = len(parsed.entries)
    print(f"  OK: '{feed_title}' with {entry_count} items")

    custom_feeds.append({"name": name, "url": url, "category": category})
    save_state(state)
    print(f"  Added: {name} ({url})")


def list_feeds():
    """List all configured feeds."""
    state = load_state()
    all_feeds = DEFAULT_FEEDS + state.get("custom_feeds", [])
    print(f"\nConfigured feeds ({len(all_feeds)} total):\n{'='*60}")

    for feed in all_feeds:
        feed_state = state.get("feeds", {}).get(feed["url"], {})
        last_fetch = feed_state.get("last_fetch", "never")[:10] if feed_state.get("last_fetch") else "never"
        known_count = len(feed_state.get("known_ids", []))
        custom = " [custom]" if feed in state.get("custom_feeds", []) else ""
        print(f"  [{feed.get('category', '?')}] {feed['name']}{custom}")
        print(f"    Last fetch: {last_fetch} | Known items: {known_count}")
        print(f"    URL: {feed['url'][:70]}")
    print()


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
def main():
    parser = argparse.ArgumentParser(description="RSS/Atom feed ingestion")
    parser.add_argument("--watch", action="store_true", help="Hourly watch mode")
    parser.add_argument("--interval", type=int, default=1,
                        help="Watch interval in hours (default: 1)")
    parser.add_argument("--add-feed", type=str, metavar="URL",
                        help="Add a custom feed URL")
    parser.add_argument("--feed-name", type=str, help="Name for custom feed")
    parser.add_argument("--feed-category", type=str, default="knowledge",
                        help="Category for custom feed")
    parser.add_argument("--list", action="store_true", help="List configured feeds")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")
    args = parser.parse_args()

    if args.list:
        list_feeds()
        return

    if args.add_feed:
        add_custom_feed(args.add_feed, name=args.feed_name, category=args.feed_category)
        return

    if args.watch:
        watch_mode(interval_hours=args.interval, verbose=not args.quiet)
    else:
        print("Fetching RSS feeds (one-shot)...")
        stats = run_once(verbose=not args.quiet)
        print(f"\nDone:")
        print(f"  Feeds checked:  {stats['feeds_checked']}")
        print(f"  New entries:    {stats['new_entries']}")
        print(f"  Files saved:    {stats['saved_files']}")
        print(f"  Errors:         {stats['errors']}")
        if stats["saved_files"] > 0:
            print(f"\nRun 'python compile.py' to process {stats['saved_files']} new articles.")


if __name__ == "__main__":
    main()
