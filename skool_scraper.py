"""
Skool multi-community scraper.

Usage:
  python skool_scraper.py --login          # one-time: open browser, log in, save cookies
  python skool_scraper.py                  # scrape all communities, save to brain/raw/skool/
  python skool_scraper.py --community aiautomationsbyjack  # single community
  python skool_scraper.py --posts-only     # skip classroom, only posts/comments
  python skool_scraper.py --classroom-only # skip posts
  python skool_scraper.py --no-video       # skip video downloading
  python skool_scraper.py --no-files       # skip file downloading

Output structure:
  brain/raw/skool/{slug}/classroom/{course-slug}/lesson-{N:02d}-{lesson-slug}.md
  brain/raw/skool/{slug}/classroom/{course-slug}/videos/{lesson-slug}/
  brain/raw/skool/{slug}/classroom/{course-slug}/files/
  brain/raw/skool/{slug}/posts/{YYYY-MM-DD}-{post-id}-{title-slug}.md

Runs standalone — no Claude involvement after setup.
"""

import argparse
import asyncio
import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests as req_lib

# ── paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
COMMUNITIES_FILE = SCRIPT_DIR / "skool_communities.json"
STATE_FILE = SCRIPT_DIR / "skool_sync_state.json"
RAW_DIR = SCRIPT_DIR / "brain" / "raw"
SKOOL_DIR = RAW_DIR / "skool"
AUTH_FILE = Path.home() / ".skool" / "auth.json"
COOKIES_FILE = Path.home() / ".skool" / "cookies.txt"
BASE_URL = "https://www.skool.com"

# File extensions to download from lesson pages
DOWNLOADABLE_EXTS = {".pdf", ".zip", ".docx", ".xlsx", ".pptx", ".mp3", ".csv", ".txt", ".epub"}

SKOOL_DIR.mkdir(parents=True, exist_ok=True)
AUTH_FILE.parent.mkdir(parents=True, exist_ok=True)


# ── helpers ────────────────────────────────────────────────────────────────────
def slugify(text: str) -> str:
    s = re.sub(r'[^\w\s-]', '', text.lower())
    s = re.sub(r'[\s_]+', '-', s)
    s = re.sub(r'-+', '-', s).strip('-')
    return s[:60] or 'untitled'


def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {}


def save_state(state: dict):
    STATE_FILE.write_text(json.dumps(state, indent=2))


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def community_state(state: dict, slug: str) -> dict:
    return state.setdefault(slug, {"classroom": {}, "posts": {}, "last_sync": None})


# ── cookie helpers ─────────────────────────────────────────────────────────────
def build_cookies_file() -> Path:
    """Convert Playwright auth.json → Netscape cookies.txt for yt-dlp."""
    auth = json.loads(AUTH_FILE.read_text())
    lines = ["# Netscape HTTP Cookie File\n"]
    for c in auth.get("cookies", []):
        domain = c.get("domain", "")
        flag = "TRUE" if domain.startswith(".") else "FALSE"
        path = c.get("path", "/")
        secure = "TRUE" if c.get("secure") else "FALSE"
        expires = str(int(c.get("expires", 0) or 0))
        name = c.get("name", "")
        value = c.get("value", "")
        lines.append(f"{domain}\t{flag}\t{path}\t{secure}\t{expires}\t{name}\t{value}\n")
    COOKIES_FILE.write_text("".join(lines))
    return COOKIES_FILE


def get_cookies_dict() -> dict:
    """Return cookies as a plain dict for requests."""
    auth = json.loads(AUTH_FILE.read_text())
    return {c["name"]: c["value"] for c in auth.get("cookies", []) if "name" in c and "value" in c}


# ── file output ────────────────────────────────────────────────────────────────
def make_header(slug: str, community_name: str, content_type: str, metadata: dict) -> str:
    lines = [
        f"SOURCE: skool-{slug}",
        f"Community: {community_name}",
        f"Type: {content_type}",
    ]
    for k, v in metadata.items():
        if v:
            lines.append(f"{k}: {v}")
    lines.append(f"Scraped: {datetime.now(timezone.utc).isoformat()}")
    return "\n".join(lines)


def write_individual_file(filepath: Path, header: str, content: str):
    filepath.parent.mkdir(parents=True, exist_ok=True)
    filepath.write_text(header + "\n---\n" + content.strip() + "\n", encoding="utf-8")


def extract_section(content: str, lesson_url: str) -> str:
    """Extract the sidebar section name a lesson belongs to.

    Skool lesson pages embed the full course sidebar. Section headers appear as
    plain text lines (no markdown link syntax) immediately above their lessons.
    """
    if not lesson_url:
        return ""
    lines = content.splitlines()
    current_section = ""
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        # Skip markdown images, headers, and percentage progress lines
        if stripped.startswith("!") or stripped.startswith("#") or re.match(r'^\d+%$', stripped):
            continue
        # Markdown link line — check if it contains our lesson URL
        if stripped.startswith("["):
            if lesson_url in stripped:
                return current_section
        else:
            # Plain text line = potential section header (not a nav link)
            if not re.match(r'^https?://', stripped) and len(stripped) < 100:
                current_section = stripped
    return ""


# ── video downloader ──────────────────────────────────────────────────────────
async def download_lesson_videos(page_url: str, output_dir: Path, cookies_file: Path):
    """Download any videos on the page using yt-dlp (handles Wistia, Vimeo, YouTube)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "yt-dlp",
        "--cookies", str(cookies_file),
        "--output", str(output_dir / "%(title)s.%(ext)s"),
        "--format", "bestvideo[height<=1080]+bestaudio/best[height<=1080]/best",
        "--merge-output-format", "mp4",
        "--no-playlist",
        "--ignore-errors",
        "--quiet",
        "--no-warnings",
        "--write-info-json",
        "--write-thumbnail",
        "--write-auto-subs",
        "--sub-langs", "en",
        "--convert-subs", "vtt",
        page_url,
    ]
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    if proc.returncode == 0 and any(output_dir.glob("*.mp4")):
        print(f"  [video] Downloaded to {output_dir.relative_to(SCRIPT_DIR)}")
    # Non-zero return is normal when no video is present — silently ignored


# ── file downloader ───────────────────────────────────────────────────────────
async def download_files_from_markdown(md: str, output_dir: Path, cookies: dict):
    """Download PDF/ZIP/DOCX etc. found in crawl4ai markdown content."""
    urls = re.findall(r'https?://[^\s)"\'<>]+', md)
    for href in urls:
        suffix = Path(href.split("?")[0]).suffix.lower()
        if suffix not in DOWNLOADABLE_EXTS:
            continue
        filename = Path(href.split("/")[-1].split("?")[0])
        if not filename.suffix:
            continue
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / filename
        if out_path.exists():
            continue
        try:
            r = req_lib.get(href, cookies=cookies, stream=True, timeout=60)
            if r.status_code == 200:
                out_path.write_bytes(r.content)
                print(f"  [file] Downloaded: {filename.name}")
        except Exception as e:
            print(f"  [file] Failed {href}: {e}")


# ── community discovery ────────────────────────────────────────────────────────
_NON_COMMUNITY_PATHS = {
    'discover', 'profile', 'settings', 'search', 'notifications',
    'explore', 'games', 'home', 'login', 'signup', 'about',
    'privacy', 'terms', 'blog', 'help', 'careers', 'contact',
    'classroom', 'members', 'leaderboards', 'calendar',
}


async def discover_communities() -> list[dict]:
    """Discover all Skool communities the user is a member of via sidebar links."""
    from playwright.async_api import async_playwright

    if not AUTH_FILE.exists():
        print(f"ERROR: No auth file at {AUTH_FILE}. Run --login first.")
        return []

    print("Discovering communities from Skool sidebar...")
    found_slugs = []

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        ctx = await browser.new_context(storage_state=str(AUTH_FILE))
        page = await ctx.new_page()

        await page.goto(BASE_URL, wait_until="domcontentloaded", timeout=30_000)
        await asyncio.sleep(3)

        # Extract all root-level hrefs (e.g. /aiautomationsbyjack)
        hrefs = await page.eval_on_selector_all(
            'a[href]',
            'els => els.map(el => el.getAttribute("href"))'
        )

        seen = set()
        for href in hrefs:
            if not href or not href.startswith('/'):
                continue
            slug = href.strip('/').split('/')[0].split('?')[0]
            if not slug or slug in _NON_COMMUNITY_PATHS:
                continue
            if slug.startswith('@') or slug.startswith('_') or slug.startswith('-'):
                continue
            if not re.match(r'^[a-z0-9][a-z0-9-]*$', slug):
                continue
            if slug not in seen:
                seen.add(slug)
                found_slugs.append(slug)

        print(f"  Found {len(found_slugs)} candidate community slug(s): {', '.join(found_slugs)}")

        # Verify each slug and detect classroom
        result = []
        for slug in found_slugs:
            try:
                await page.goto(f"{BASE_URL}/{slug}", wait_until="domcontentloaded", timeout=20_000)
                await asyncio.sleep(1)

                # Get community name from page title or h1
                name = slug
                for sel in ['h1', '[class*="group-name"]', '[class*="community-name"]',
                            '[class*="GroupName"]', 'title']:
                    el = await page.query_selector(sel)
                    if el:
                        raw = (await el.inner_text()).strip()
                        if raw and raw.lower() not in ('skool', ''):
                            name = raw.split('\n')[0].strip()
                            break

                has_classroom = bool(
                    await page.query_selector(f'a[href="/{slug}/classroom"]')
                )
                result.append({"slug": slug, "name": name, "has_classroom": has_classroom})
                print(f"  Verified: {name} ({slug}) — classroom: {has_classroom}")
            except Exception as e:
                print(f"  Skipping {slug}: {e}")

        await ctx.close()
        await browser.close()

    return result


def run_discover():
    """Discover communities and merge new ones into skool_communities.json."""
    new_entries = asyncio.run(discover_communities())
    if not new_entries:
        print("No communities found.")
        return

    existing = json.loads(COMMUNITIES_FILE.read_text())
    existing_slugs = {c["slug"] for c in existing["communities"]}

    added = 0
    for entry in new_entries:
        if entry["slug"] not in existing_slugs:
            existing["communities"].append(entry)
            added += 1
            print(f"  Added: {entry['name']} ({entry['slug']})")

    COMMUNITIES_FILE.write_text(json.dumps(existing, indent=2))
    print(f"\nDiscovery complete: {added} new community/communities added ({len(existing['communities'])} total).")


# ── login ──────────────────────────────────────────────────────────────────────
def login():
    """Open browser for user to log into Skool, save auth to ~/.skool/auth.json."""
    from playwright.sync_api import sync_playwright

    print("Opening browser — log into skool.com, then close the browser window.")
    print(f"Cookies will be saved to {AUTH_FILE}\n")

    with sync_playwright() as p:
        # channel="chrome" avoids Google login blocks that affect bundled Chromium
        browser = p.chromium.launch(channel="chrome", headless=False)
        ctx = browser.new_context()
        page = ctx.new_page()
        page.goto(f"{BASE_URL}/login")
        page.wait_for_url(lambda url: "/login" not in url, timeout=300_000)
        print("Login detected — saving cookies...")
        ctx.storage_state(path=str(AUTH_FILE))
        browser.close()

    print(f"Auth saved to {AUTH_FILE}")


# ── crawl4ai fetch ─────────────────────────────────────────────────────────────
async def fetch_markdown(url: str, crawler, wait_for: str = None, js_code: str = None) -> str:
    from crawl4ai import CrawlerRunConfig

    cfg = CrawlerRunConfig(only_text=False, word_count_threshold=10)
    if wait_for:
        cfg.wait_for = wait_for
    if js_code:
        cfg.js_code = js_code

    result = await crawler.arun(url=url, config=cfg)
    if result.success:
        return result.markdown or ""
    print(f"  [warn] Failed to fetch {url}: {result.error_message}")
    return ""


# ── classroom scraper (crawl4ai primary, Playwright click fallback) ───────────
_CLASSROOM_NAV = {'classroom'}  # the index page itself — not a course

def _extract_course_urls(md: str, slug: str) -> list[str]:
    """Extract course URLs from classroom index markdown."""
    pattern = r'https?://www\.skool\.com/' + re.escape(slug) + r'/classroom/([a-z0-9][a-z0-9-]*)'
    found = re.findall(pattern, md)
    seen = set()
    result = []
    for course_id in found:
        url = f"{BASE_URL}/{slug}/classroom/{course_id}"
        if url not in seen:
            seen.add(url)
            result.append(url)
    return result


def _extract_lesson_urls(md: str, slug: str, course_url: str) -> list[str]:
    """Extract lesson URLs from a course page markdown."""
    base = course_url.rstrip('/')
    # Lessons are deeper paths: /classroom/{course}/{lesson}
    pattern = r'https?://www\.skool\.com/' + re.escape(slug) + r'/classroom/[a-z0-9][a-z0-9-]*/([a-z0-9][a-z0-9-]*)'
    found = re.findall(pattern, md)
    # Also grab the full URL matches
    full_pattern = r'(https?://www\.skool\.com/' + re.escape(slug) + r'/classroom/[a-z0-9][a-z0-9-]*/[a-z0-9][a-z0-9-]*)'
    full_urls = re.findall(full_pattern, md)
    seen = {base}
    result = []
    for url in full_urls:
        url = url.split('?')[0].rstrip('/')
        if url not in seen:
            seen.add(url)
            result.append(url)
    return result


def _collect_lessons_from_next_data(children: list, slug: str, course_name: str) -> list[tuple[str, str]]:
    """Recursively extract (url, title) tuples from __NEXT_DATA__ course children tree.

    Lesson URLs use query-param format: /classroom/{course_name}?md={module_uuid}
    The 'name' field is a short hash used internally; the 'id' field is the UUID for URLs.
    """
    items = []
    for child in children:
        c = child.get('course', {})
        if (c.get('unitType') == 'module'
                and c.get('metadata', {}).get('hasAccess')
                and c.get('id')):
            url = f"{BASE_URL}/{slug}/classroom/{course_name}?md={c['id']}"
            title = c.get('metadata', {}).get('title', c.get('name', c['id']))
            items.append((url, title))
        if child.get('children'):
            items.extend(_collect_lessons_from_next_data(child['children'], slug, course_name))
    return items


async def scrape_classroom(slug: str, community_name: str, crawler, pw_context, state: dict,
                           slug_dir: Path, cookies_file: Path, cookies: dict,
                           dl_video: bool, dl_files: bool):
    """
    Scrape Skool classroom.
    Primary: crawl4ai regex (catches plain HTML communities).
    Fallback: Playwright extracts course+lesson tree from Next.js __NEXT_DATA__,
              then crawl4ai fetches lesson content.
    """
    print(f"  [classroom] {slug}")
    classroom_url = f"{BASE_URL}/{slug}/classroom"

    # next_lesson_map: course_url → ordered list of lesson URLs (from __NEXT_DATA__)
    # next_course_titles: course_url → display title
    next_lesson_map: dict[str, list[str]] = {}
    next_course_titles: dict[str, str] = {}

    # JS to trigger lazy-loaded course cards on Skool classroom pages
    _CLASSROOM_JS = (
        "window.scrollTo(0, document.body.scrollHeight); "
        "await new Promise(r => setTimeout(r, 2000)); "
        "window.scrollTo(0, 0);"
    )

    # ── Primary: crawl4ai (stealth + auth) ───────────────────────────────────
    classroom_md = await fetch_markdown(classroom_url, crawler, js_code=_CLASSROOM_JS)
    course_urls = _extract_course_urls(classroom_md, slug) if classroom_md else []

    # ── Fallback: extract from Next.js __NEXT_DATA__ via Playwright ───────────
    if not course_urls and pw_context:
        print(f"  [classroom] crawl4ai found no courses — reading __NEXT_DATA__ via Playwright")
        page = await pw_context.new_page()
        try:
            await page.goto(classroom_url, wait_until="load", timeout=60_000)
            await asyncio.sleep(3)

            all_courses_raw = await page.evaluate(
                '() => JSON.stringify(window.__NEXT_DATA__?.props?.pageProps?.allCourses || [])'
            )
            all_courses = json.loads(all_courses_raw)

            # Include all courses regardless of hasAccess — some communities omit this flag
            for course in all_courses:
                meta = course.get('metadata', {})
                course_name = course.get('name', '')
                course_title = meta.get('title', course_name)
                if not course_name:
                    continue

                course_url = f"{BASE_URL}/{slug}/classroom/{course_name}"
                course_urls.append(course_url)
                next_course_titles[course_url] = course_title

                # Navigate to course page to extract lesson tree
                await page.goto(course_url, wait_until="load", timeout=60_000)
                await asyncio.sleep(2)
                course_data_raw = await page.evaluate(
                    '() => JSON.stringify(window.__NEXT_DATA__?.props?.pageProps?.course || null)'
                )
                course_data = json.loads(course_data_raw)
                lessons = []
                if course_data and course_data.get('children'):
                    lessons = _collect_lessons_from_next_data(course_data['children'], slug, course_name)
                next_lesson_map[course_url] = lessons
                print(f"  [classroom] Found: {course_title[:50]} ({len(lessons)} lessons)")

            # If __NEXT_DATA__.allCourses was empty, extract course card links from DOM
            if not course_urls:
                print(f"  [classroom] allCourses empty — scraping course links from DOM")
                dom_links = await page.evaluate(f'''() => {{
                    const links = Array.from(document.querySelectorAll('a[href]'));
                    return links
                        .map(a => a.href)
                        .filter(h => h.includes('/classroom/') && !h.endsWith('/classroom'));
                }}''')
                for href in dict.fromkeys(dom_links):  # dedup preserving order
                    href = href.split('?')[0].rstrip('/')
                    if f"/{slug}/classroom/" in href:
                        parts = href.split(f"/{slug}/classroom/")
                        if len(parts) == 2 and parts[1] and '/' not in parts[1]:
                            course_url = href
                            if course_url not in course_urls:
                                course_urls.append(course_url)
                                print(f"  [classroom] DOM course: {course_url}")
        finally:
            await page.close()

    if not course_urls:
        h = content_hash(classroom_md)
        key = "classroom/index"
        if state["classroom"].get(key, {}).get("hash") != h and classroom_md:
            out = slug_dir / "classroom" / "index.md"
            write_individual_file(out, make_header(slug, community_name, "classroom", {"Page": "Index"}), classroom_md)
            state["classroom"][key] = {"hash": h, "scraped": datetime.now(timezone.utc).isoformat()}
            print(f"  [classroom] Saved index (no course links found)")
        return

    print(f"  [classroom] Scraping {len(course_urls)} courses")

    for course_url in course_urls:
        course_key = course_url.replace(f"{BASE_URL}/{slug}/", "")

        # Use __NEXT_DATA__ title if available; else derive from crawl4ai markdown heading
        if course_url in next_course_titles:
            course_title = next_course_titles[course_url]
            course_md = ""  # skip crawl4ai fetch for course index page
        else:
            course_md = await fetch_markdown(course_url, crawler)
            if not course_md:
                continue
            title_match = re.search(r'^#{1,3}\s+(.+)$', course_md, re.MULTILINE)
            course_title = title_match.group(1).strip() if title_match else course_key.split('/')[-1]

        course_slug_fs = slugify(course_title)
        course_dir = slug_dir / "classroom" / course_slug_fs

        # Use __NEXT_DATA__ lesson list (url, title tuples) or fallback to crawl4ai regex
        raw_lessons = next_lesson_map.get(course_url)
        if raw_lessons is None:
            # crawl4ai path: plain (url,) — wrap to match tuple format
            raw_lessons = [
                (u, u.split('/')[-1])
                for u in _extract_lesson_urls(course_md, slug, course_url)
            ]

        if not raw_lessons:
            if course_md:
                h = content_hash(course_md)
                if state["classroom"].get(course_key, {}).get("hash") != h:
                    out = course_dir / "lesson-01-index.md"
                    write_individual_file(out, make_header(slug, community_name, "classroom", {
                        "URL": course_url, "Course": course_title
                    }), course_md)
                    state["classroom"][course_key] = {"hash": h, "title": course_title,
                                                      "scraped": datetime.now(timezone.utc).isoformat()}
                    print(f"  [classroom] Course (single page): {course_title[:60]}")
                    if dl_video:
                        await download_lesson_videos(course_url, course_dir / "videos" / "index", cookies_file)
                    if dl_files:
                        await download_files_from_markdown(course_md, course_dir / "files", cookies)
            continue

        # JS for ?md={uuid} lesson pages — waits for lesson body to render
        _LESSON_JS = (
            "await new Promise(r => setTimeout(r, 2500)); "
            "window.scrollTo(0, document.body.scrollHeight);"
        )

        for lesson_idx, (lesson_url, next_data_title) in enumerate(raw_lessons, start=1):
            lesson_key = lesson_url.replace(f"{BASE_URL}/{slug}/", "")
            already_done = state["classroom"].get(lesson_key, {})

            # Use js_code only for query-param lesson URLs that require JS rendering
            _needs_js = "?md=" in lesson_url
            lesson_md = await fetch_markdown(lesson_url, crawler,
                                             js_code=_LESSON_JS if _needs_js else None)
            if not lesson_md:
                continue

            lesson_title_m = re.search(r'^#{1,3}\s+(.+)$', lesson_md, re.MULTILINE)
            lesson_title = (lesson_title_m.group(1).strip() if lesson_title_m
                            else next_data_title or lesson_key.split('/')[-1])
            lesson_slug_fs = slugify(lesson_title)
            lesson_file = course_dir / f"lesson-{lesson_idx:02d}-{lesson_slug_fs}.md"

            if not already_done.get("hash") and len(lesson_md) >= 50:
                h = content_hash(lesson_md)
                section_name = extract_section(lesson_md, lesson_url)
                write_individual_file(lesson_file, make_header(slug, community_name, "classroom", {
                    "URL": lesson_url, "Course": course_title,
                    "Section": section_name, "Lesson": lesson_title
                }), lesson_md)
                state["classroom"][lesson_key] = {
                    "hash": h, "course": course_title, "title": lesson_title,
                    "file": str(lesson_file.relative_to(SCRIPT_DIR)),
                    "scraped": datetime.now(timezone.utc).isoformat(),
                }
                print(f"  [classroom] Lesson {lesson_idx:02d}: {lesson_title[:60]}")

            video_dir = course_dir / "videos" / lesson_slug_fs
            if dl_video and not already_done.get("video_done"):
                await download_lesson_videos(lesson_url, video_dir, cookies_file)
                state["classroom"].setdefault(lesson_key, {})["video_done"] = True

            if dl_files and not already_done.get("files_done"):
                await download_files_from_markdown(lesson_md, course_dir / "files", cookies)
                state["classroom"].setdefault(lesson_key, {})["files_done"] = True

            await asyncio.sleep(0.5)


# ── posts scraper ──────────────────────────────────────────────────────────────
async def scrape_posts(slug: str, community_name: str, crawler, state: dict,
                       slug_dir: Path, max_pages: int = 10, full_history: bool = False):
    print(f"  [posts] {slug}")
    posts_dir = slug_dir / "posts"
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    for page_num in range(max_pages):
        feed_url = (
            f"{BASE_URL}/{slug}?c=&s=newest&fl=&p={page_num + 1}"
            if page_num > 0
            else f"{BASE_URL}/{slug}?c=&s=newest&fl="
        )
        feed_md = await fetch_markdown(feed_url, crawler)

        if not feed_md or len(feed_md) < 100:
            print(f"  [posts] Page {page_num + 1}: empty or end of feed")
            break

        # Skool post URLs: skool.com/{slug}/{post-slug}
        _SKOOL_NAV = {'calendar', 'about', 'classroom', 'members', 'leaderboards'}
        raw_matches = re.findall(
            r'https?://www\.skool\.com/' + re.escape(slug) + r'/([a-z0-9][a-z0-9-]{4,})(?:\?[^\s)"\']*)?',
            feed_md
        )
        post_links = []
        seen_posts = set()
        for post_slug_match in raw_matches:
            post_slug_only = post_slug_match.split('?')[0]
            if post_slug_only in _SKOOL_NAV or post_slug_only.startswith('-') or post_slug_only.startswith('@'):
                continue
            url = f"{BASE_URL}/{slug}/{post_slug_only}"
            if url not in seen_posts:
                seen_posts.add(url)
                post_links.append(url)

        if not post_links:
            h = content_hash(feed_md)
            key = f"feed/page-{page_num + 1}"
            if state["posts"].get(key, {}).get("hash") != h:
                out = posts_dir / f"{today}-feed-page-{page_num + 1:02d}.md"
                write_individual_file(out, make_header(slug, community_name, "post", {"Page": str(page_num + 1)}), feed_md)
                state["posts"][key] = {"hash": h, "scraped": datetime.now(timezone.utc).isoformat()}
            break

        new_posts = 0
        for post_url in post_links:
            post_id = post_url.rstrip('/').split('/')[-1].split('?')[0]
            if state["posts"].get(post_id, {}).get("hash"):
                continue

            post_md = await fetch_markdown(post_url, crawler)
            if not post_md:
                continue

            h = content_hash(post_md)
            title_match = re.search(r'^#{1,3}\s+(.+)$', post_md, re.MULTILINE)
            if title_match:
                title = title_match.group(1).strip()
            else:
                # Extract first meaningful line from post body (after nav/date line)
                _date_pat = re.compile(r'^\d+[dwmhsy]')
                _plines = post_md.splitlines()
                _body_start = 0
                for _i, _ln in enumerate(_plines):
                    _s = _ln.strip()
                    if _date_pat.match(_s) and ('•' in _s or re.match(r'^\d+[dwmhsy]$', _s)):
                        _body_start = _i + 1
                        break
                title = post_id  # fallback
                for _ln in _plines[_body_start:_body_start + 30]:
                    _s = _ln.strip()
                    if (_s and len(_s) > 5
                            and not _s.startswith('[')
                            and not _s.startswith('!')
                            and not _s.startswith('http')
                            and not _s.startswith('#')
                            and not _date_pat.match(_s)):
                        title = _s[:80]
                        break
            post_slug_fs = slugify(title)

            out = posts_dir / f"{today}-{post_id}-{post_slug_fs}.md"
            write_individual_file(out, make_header(slug, community_name, "post", {
                "PostID": post_id, "URL": post_url, "Title": title,
            }), post_md)
            state["posts"][post_id] = {
                "hash": h, "title": title,
                "file": str(out.relative_to(SCRIPT_DIR)),
                "scraped": datetime.now(timezone.utc).isoformat(),
            }
            new_posts += 1
            print(f"  [posts] Saved: {title[:60]}")
            await asyncio.sleep(0.5)

        if new_posts == 0:
            if full_history:
                print(f"  [posts] Page {page_num + 1}: all cached, continuing for full history")
            else:
                print(f"  [posts] Page {page_num + 1}: all posts already cached, stopping")
                break

        print(f"  [posts] Page {page_num + 1}: {new_posts} new posts")
        await asyncio.sleep(1)


# ── main scrape ────────────────────────────────────────────────────────────────
async def scrape(args):
    if not AUTH_FILE.exists():
        print(f"ERROR: No auth file at {AUTH_FILE}")
        print("Run:  python skool_scraper.py --login")
        sys.exit(1)

    from crawl4ai import AsyncWebCrawler, BrowserConfig
    from playwright.async_api import async_playwright

    communities = json.loads(COMMUNITIES_FILE.read_text())["communities"]
    if args.community:
        communities = [c for c in communities if c["slug"] == args.community]
        if not communities:
            print(f"ERROR: Community '{args.community}' not in skool_communities.json")
            sys.exit(1)

    dl_video = not args.no_video
    dl_files = not args.no_files

    # Build cookie files once
    cookies_file = build_cookies_file()
    cookies = get_cookies_dict()

    state = load_state()

    crawl4ai_cfg = BrowserConfig(
        headless=True,
        storage_state=str(AUTH_FILE),
        enable_stealth=True,
    )

    async with async_playwright() as pw:
        # Headless scraping uses bundled Chromium with stored cookies — no channel needed
        pw_browser = await pw.chromium.launch(headless=True)
        pw_context = await pw_browser.new_context(storage_state=str(AUTH_FILE))

        async with AsyncWebCrawler(config=crawl4ai_cfg) as crawler:
            for community in communities:
                slug = community["slug"]
                name = community["name"]
                print(f"\n{'='*60}")
                print(f"Scraping: {name} ({slug})")
                print(f"{'='*60}")

                cs = community_state(state, slug)
                slug_dir = SKOOL_DIR / slug

                if not args.posts_only and community.get("has_classroom"):
                    await scrape_classroom(slug, name, crawler, pw_context, cs, slug_dir,
                                           cookies_file, cookies, dl_video, dl_files)

                if not args.classroom_only:
                    await scrape_posts(slug, name, crawler, cs, slug_dir,
                                       max_pages=args.max_pages,
                                       full_history=args.full_history)

                cs["last_sync"] = datetime.now(timezone.utc).isoformat()
                save_state(state)

        await pw_context.close()
        await pw_browser.close()

    print(f"\nDone. Files written to {SKOOL_DIR}")
    print(f"State saved to {STATE_FILE}")


# ── entry point ────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Skool multi-community scraper")
    parser.add_argument("--login", action="store_true")
    parser.add_argument("--discover", action="store_true", help="Auto-detect all joined communities and update skool_communities.json")
    parser.add_argument("--community")
    parser.add_argument("--posts-only", action="store_true")
    parser.add_argument("--classroom-only", action="store_true")
    parser.add_argument("--max-pages", type=int, default=10)
    parser.add_argument("--full-history", action="store_true",
                        help="Scrape all historical posts — don't stop when a page is fully cached")
    parser.add_argument("--no-video", action="store_true", help="Skip video downloading")
    parser.add_argument("--no-files", action="store_true", help="Skip file downloading")
    args = parser.parse_args()

    if args.login:
        login()
    elif args.discover:
        run_discover()
    else:
        asyncio.run(scrape(args))


if __name__ == "__main__":
    main()
