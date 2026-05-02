"""
Notion sync: mirrors scraped Skool content into a clean Notion workspace.

Usage:
  python notion_sync.py              # sync all communities
  python notion_sync.py --community aiautomationsbyjack
  python notion_sync.py --dry-run    # show what would be created

Requires: pip install notion-client
Token loaded from ~/.credentials/api-keys.env (NOTION_API_KEY=...)

Notion structure created:
  📚 Skool Communities  (root page, created once)
  └── 🎓 {Community Name}
      ├── 📖 Classroom
      │   └── {Course Name}
      │       └── {Lesson Title}  (page with full content)
      └── 💬 Posts  (database: Title | Date | URL)
"""

import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
COMMUNITIES_FILE = SCRIPT_DIR / "skool_communities.json"
STATE_FILE = SCRIPT_DIR / "skool_sync_state.json"
SKOOL_DIR = SCRIPT_DIR / "brain" / "raw" / "skool"
CREDS_FILE = Path.home() / ".credentials" / "api-keys.env"

NOTION_BLOCK_LIMIT = 100   # Notion API max children per append
MAX_RICH_TEXT = 2000       # Notion rich_text content limit per element


# ── auth ───────────────────────────────────────────────────────────────────────
def load_token() -> str:
    token = os.environ.get("NOTION_API_KEY", "")
    if token:
        return token
    if CREDS_FILE.exists():
        for line in CREDS_FILE.read_text().splitlines():
            if line.startswith("NOTION_API_KEY="):
                return line.split("=", 1)[1].strip()
    print("ERROR: NOTION_API_KEY not set. Add it to ~/.credentials/api-keys.env")
    sys.exit(1)


def get_client():
    from notion_client import Client
    return Client(auth=load_token())


# ── state ──────────────────────────────────────────────────────────────────────
def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {}


def save_state(state: dict):
    STATE_FILE.write_text(json.dumps(state, indent=2))


# ── notion helpers ─────────────────────────────────────────────────────────────
def _utf16_slice(text: str, max_units: int) -> tuple[str, str]:
    """Split text so the first part fits within max_units UTF-16 code units.

    Notion's API counts string lengths the same way JavaScript does — characters
    above U+FFFF (emoji, some CJK) each count as 2 units (surrogate pair).
    Python's len() counts code points, so a 2000-char string with 11 emoji
    appears as 2011 units to Notion and fails validation.
    """
    units = 0
    for i, c in enumerate(text):
        cost = 2 if ord(c) > 0xFFFF else 1
        if units + cost > max_units:
            return text[:i], text[i:]
        units += cost
    return text, ""


def _rich(text: str) -> list[dict]:
    """Split text into ≤2000 UTF-16-unit rich_text elements."""
    chunks = []
    while text:
        chunk, text = _utf16_slice(text, MAX_RICH_TEXT)
        if not chunk:
            chunk, text = text[:1], text[1:]  # always advance at least one char
        chunks.append({"type": "text", "text": {"content": chunk}})
    return chunks or [{"type": "text", "text": {"content": ""}}]


def _heading(level: int, text: str) -> dict:
    key = f"heading_{level}"
    chunk, _ = _utf16_slice(text, MAX_RICH_TEXT)
    return {"object": "block", "type": key, key: {"rich_text": [{"type": "text", "text": {"content": chunk}}]}}


def _paragraph(text: str) -> list[dict]:
    """Return one or more paragraph blocks (split at 2000 UTF-16 units each)."""
    blocks = []
    while text:
        chunk, text = _utf16_slice(text, MAX_RICH_TEXT)
        if not chunk:
            chunk, text = text[:1], text[1:]
        blocks.append({"object": "block", "type": "paragraph",
                       "paragraph": {"rich_text": [{"type": "text", "text": {"content": chunk}}]}})
    return blocks


def _divider() -> dict:
    return {"object": "block", "type": "divider", "divider": {}}


def extract_post_body(content: str, include_comments: bool = True) -> str:
    """Extract post title+body+comments from full-page crawl4ai markdown.

    Skool pages are scraped as full-page markdown (nav + profile + post + comments).
    We find the date line (e.g. "10d • [💎 Gems](...)") for the post start.
    Comments follow the Like/reaction section and are included when include_comments=True.
    """
    lines = content.splitlines()
    date_pat = re.compile(r'^\d+[dwmhsy]')

    # Find post body start (line after the date/category line)
    start_idx = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if date_pat.match(stripped) and '•' in stripped:
            start_idx = i + 1
            break

    # Find the "Like" reaction line (end of post body, start of reactions)
    like_idx = len(lines)
    for i in range(start_idx, len(lines)):
        if lines[i].strip() == 'Like':
            like_idx = i
            break

    body = '\n'.join(lines[start_idx:like_idx]).strip()
    body = re.sub(r'\s*\.\.\. See more\s*$', '', body, flags=re.IGNORECASE).strip()

    if include_comments and like_idx < len(lines):
        # Skip the Like / count / "N comments" lines, then extract comment blocks
        # Comment pattern: commenter name line → date line → comment text → Like
        comment_start = like_idx
        for j in range(like_idx, min(like_idx + 5, len(lines))):
            if re.match(r'^\d+ comments?$', lines[j].strip(), re.IGNORECASE):
                comment_start = j + 1
                break

        comment_lines = lines[comment_start:]
        # Strip trailing navigation boilerplate (drag-and-drop aria text)
        stop_pats = re.compile(r'To pick up a draggable|Previous|^Next$|^\d+-\d+ of \d+', re.IGNORECASE)
        clean_comments = []
        for ln in comment_lines:
            if stop_pats.search(ln):
                break
            clean_comments.append(ln)

        comments_text = '\n'.join(clean_comments).strip()
        if comments_text:
            body = body + '\n\n---\n\n💬 **Comments**\n\n' + comments_text

    return body if body else content


def markdown_to_blocks(md: str) -> list[dict]:
    """Convert markdown text to Notion blocks (headings + paragraphs)."""
    blocks = []
    for para in re.split(r'\n{2,}', md.strip()):
        para = para.strip()
        if not para:
            continue
        if para.startswith("### "):
            blocks.append(_heading(3, para[4:].strip()))
        elif para.startswith("## "):
            blocks.append(_heading(2, para[3:].strip()))
        elif para.startswith("# "):
            blocks.append(_heading(1, para[2:].strip()))
        else:
            blocks.extend(_paragraph(para))
    return blocks


def slugify(text: str) -> str:
    s = re.sub(r'[^\w\s-]', '', text.lower())
    s = re.sub(r'[\s_]+', '-', s)
    return re.sub(r'-+', '-', s).strip('-')[:60] or 'untitled'


def _callout(text: str, emoji: str = "💬") -> dict:
    chunk, _ = _utf16_slice(text, MAX_RICH_TEXT)
    return {
        "object": "block", "type": "callout",
        "callout": {
            "rich_text": [{"type": "text", "text": {"content": chunk}}],
            "icon": {"type": "emoji", "emoji": emoji},
            "color": "gray_background",
        }
    }


def clean_lesson_content(content: str, lesson_url: str = "") -> str:
    """Strip Skool page navigation/sidebar from lesson markdown, leaving just the lesson body.

    crawl4ai returns the full page: nav links, profile image, course sidebar with all lesson
    links, then the actual lesson content. We strip everything up to and including the sidebar.
    """
    lines = content.splitlines()
    # Navigation markers that appear before the actual lesson content
    nav_patterns = [
        re.compile(r'^\[Community\]\('),
        re.compile(r'^\[Classroom\]\('),
        re.compile(r'^\[Calendar\]\('),
        re.compile(r'^\[Members\]\('),
        re.compile(r'^\[Leaderboards\]\('),
        re.compile(r'^\[About\]\('),
    ]
    # The sidebar ends when we stop seeing markdown links that are lesson URLs
    lesson_link_pat = re.compile(r'^\[.+\]\(https://www\.skool\.com/.+/classroom/')

    last_sidebar_idx = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        # Nav bar line (multiple links concatenated)
        if any(p.search(stripped) for p in nav_patterns):
            last_sidebar_idx = i
        # Sidebar lesson link
        if lesson_link_pat.match(stripped):
            last_sidebar_idx = i
        # Profile image lines
        if stripped.startswith('![') and 'skool.com' in stripped:
            last_sidebar_idx = i

    # Content starts after the last sidebar element
    body_lines = lines[last_sidebar_idx + 1:]

    # Strip leading empty lines and the progress indicator (e.g. "0%")
    while body_lines and (not body_lines[0].strip() or re.match(r'^\d+%$', body_lines[0].strip())):
        body_lines.pop(0)

    # Strip trailing boilerplate ("To pick up a draggable...")
    stop_pat = re.compile(r'To pick up a draggable|^Previous\d*Next$', re.IGNORECASE)
    clean = []
    for ln in body_lines:
        if stop_pat.search(ln.strip()):
            break
        clean.append(ln)

    return '\n'.join(clean).strip() or content.strip()


def extract_post_structured(content: str) -> dict:
    """Parse Skool full-page markdown into structured fields: author, date, body, comments."""
    lines = content.splitlines()
    date_pat = re.compile(r'^\d+[dwmhsy]')

    author = ""
    date_str = ""
    start_idx = 0

    _nav_pat = re.compile(
        r'^\[(?:Community|Classroom|Calendar|Members|Leaderboards|About)\]'
        r'|^!\[.*\]\(https://www\.skool\.com'
        r'|^https://www\.skool\.com'
    )

    for i, line in enumerate(lines):
        stripped = line.strip()
        # Date line pattern: "10d • [category](url)" or just "10d"
        if date_pat.match(stripped) and ('•' in stripped or re.match(r'^\d+[dwmhsy]$', stripped)):
            date_str = stripped.split('•')[0].strip()
            # Author is usually the non-empty line just before the date line
            for j in range(i - 1, max(i - 5, -1), -1):
                candidate = lines[j].strip()
                if candidate and not candidate.startswith('!') and not candidate.startswith('[') \
                        and not candidate.startswith('http') and len(candidate) < 80:
                    author = candidate
                    break
            start_idx = i + 1
            break

    # If no date line found, strip nav lines from the top so body doesn't include sidebar
    if start_idx == 0:
        for i, line in enumerate(lines):
            if not _nav_pat.search(line.strip()):
                start_idx = i
                break

    # Find "Like" reaction (end of body)
    like_idx = len(lines)
    for i in range(start_idx, len(lines)):
        if lines[i].strip() == 'Like':
            like_idx = i
            break

    body = '\n'.join(lines[start_idx:like_idx]).strip()
    body = re.sub(r'\s*\.\.\. See more\s*$', '', body, flags=re.IGNORECASE).strip()

    # Extract comments
    comment_start = like_idx
    for j in range(like_idx, min(like_idx + 5, len(lines))):
        if re.match(r'^\d+ comments?$', lines[j].strip(), re.IGNORECASE):
            comment_start = j + 1
            break

    stop_pats = re.compile(r'To pick up a draggable|^Previous$|^Next$|^\d+-\d+ of \d+', re.IGNORECASE)
    raw_comment_lines = []
    for ln in lines[comment_start:]:
        if stop_pats.search(ln.strip()):
            break
        raw_comment_lines.append(ln)

    # Parse individual comment blocks: name → date → body → Like
    comments = []
    c_lines = [l.strip() for l in raw_comment_lines if l.strip()]
    i = 0
    while i < len(c_lines):
        # Detect commenter name (short, no links, followed by a date-ish line)
        if (i + 1 < len(c_lines) and not c_lines[i].startswith('[')
                and not c_lines[i].startswith('!')
                and len(c_lines[i]) < 80
                and date_pat.match(c_lines[i + 1])):
            c_author = c_lines[i]
            c_date = c_lines[i + 1]
            i += 2
            c_body_lines = []
            while i < len(c_lines) and c_lines[i] != 'Like':
                c_body_lines.append(c_lines[i])
                i += 1
            if i < len(c_lines) and c_lines[i] == 'Like':
                i += 1  # skip Like
            comments.append({"author": c_author, "date": c_date, "body": '\n'.join(c_body_lines)})
        else:
            i += 1

    return {"author": author, "date": date_str, "body": body, "comments": comments}


def append_blocks_batched(client, page_id: str, blocks: list[dict]):
    """Append blocks in batches of NOTION_BLOCK_LIMIT."""
    for i in range(0, len(blocks), NOTION_BLOCK_LIMIT):
        client.blocks.children.append(block_id=page_id, children=blocks[i:i + NOTION_BLOCK_LIMIT])


def find_child_page(client, parent_id: str, title: str) -> str | None:
    """Search for an existing child page with given title."""
    try:
        results = client.blocks.children.list(block_id=parent_id)
        for block in results.get("results", []):
            if block.get("type") == "child_page":
                if block["child_page"].get("title", "").strip() == title.strip():
                    return block["id"]
    except Exception:
        pass
    return None


def get_or_create_page(client, parent_id: str, title: str, icon: str = "", dry_run: bool = False) -> str | None:
    """Return page ID, creating it under parent if needed."""
    existing = find_child_page(client, parent_id, title)
    if existing:
        return existing

    if dry_run:
        print(f"    [dry-run] Would create page: {title}")
        return "dry-run-id"

    props = {"title": [{"type": "text", "text": {"content": title}}]}
    kwargs = {"parent": {"page_id": parent_id}, "properties": props}
    if icon:
        kwargs["icon"] = {"type": "emoji", "emoji": icon}

    try:
        page = client.pages.create(**kwargs)
        return page["id"]
    except Exception as e:
        print(f"    ERROR creating page '{title}': {e}")
        return None


def get_or_create_root_page(client, state: dict, dry_run: bool) -> str | None:
    """Get or create the top-level 'Skool Communities' workspace page."""
    page_id = state.get("notion_root_page_id")
    if page_id:
        return page_id

    if dry_run:
        print("  [dry-run] Would create root page: 📚 Skool Communities")
        return "dry-run-root"

    # Search for existing page first
    try:
        results = client.search(query="Skool Communities", filter={"property": "object", "value": "page"})
        for r in results.get("results", []):
            title_prop = r.get("properties", {}).get("title", {})
            title_parts = title_prop.get("title", [])
            existing_title = "".join(p.get("plain_text", "") for p in title_parts)
            if "Skool Communities" in existing_title:
                page_id = r["id"]
                state["notion_root_page_id"] = page_id
                save_state(state)
                print(f"  Found existing root page: {page_id}")
                return page_id
    except Exception:
        pass

    # Internal integrations can't create workspace-level pages — need a parent page ID
    print()
    print("  ┌─ ACTION REQUIRED ───────────────────────────────────────────┐")
    print("  │  In Notion:                                                  │")
    print("  │  1. Create a new page named 'Skool Communities'             │")
    print("  │  2. Click ••• → Connections → add this integration          │")
    print("  │  3. Copy the page ID from the URL:                          │")
    print("  │     notion.so/{workspace}/[PAGE_ID_HERE]?v=...              │")
    print("  │  4. Run:  python notion_sync.py --root-page-id PAGE_ID      │")
    print("  └──────────────────────────────────────────────────────────────┘")
    print()
    return None


def get_or_create_posts_page(client, community_page_id: str,
                              state_cs: dict, dry_run: bool) -> str | None:
    """Get or create a Posts page (simple page, not database) under the community page."""
    page_id = state_cs.get("notion_posts_page_id")
    if page_id:
        return page_id

    if dry_run:
        print(f"    [dry-run] Would create Posts page")
        return "dry-run-posts"

    page_id = get_or_create_page(client, community_page_id, "💬 Posts", icon="💬", dry_run=False)
    if page_id:
        state_cs["notion_posts_page_id"] = page_id
    return page_id


# ── file parsing ───────────────────────────────────────────────────────────────
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


# ── sync logic ─────────────────────────────────────────────────────────────────
def sync_community(client, slug: str, community_name: str, root_page_id: str,
                   state: dict, dry_run: bool):
    print(f"\n{'='*60}")
    print(f"  Notion sync: {community_name} ({slug})")
    print(f"{'='*60}")

    cs = state.setdefault(slug, {"classroom": {}, "posts": {}, "last_sync": None})
    notion_cs = cs.setdefault("notion", {})
    community_dir = SKOOL_DIR / slug

    if not community_dir.exists():
        print(f"  No scraped content at {community_dir}")
        return

    # Community root page
    community_page_id = notion_cs.get("page_id") or get_or_create_page(
        client, root_page_id, f"🎓 {community_name}", dry_run=dry_run
    )
    if not community_page_id:
        return
    if not dry_run:
        notion_cs["page_id"] = community_page_id
        save_state(state)

    # Classroom section
    classroom_dir = community_dir / "classroom"
    if classroom_dir.exists():
        classroom_page_id = notion_cs.get("classroom_page_id") or get_or_create_page(
            client, community_page_id, "📖 Classroom", icon="📖", dry_run=dry_run
        )
        if classroom_page_id and not dry_run:
            notion_cs["classroom_page_id"] = classroom_page_id
            save_state(state)

        lesson_pages = notion_cs.setdefault("lesson_pages", {})
        new_lessons = 0

        for course_dir in sorted(d for d in classroom_dir.iterdir() if d.is_dir()):
            # Use Course: metadata (with emoji) from first lesson instead of folder slug
            first_lesson_file = next(iter(sorted(course_dir.glob("*.md"))), None)
            if first_lesson_file:
                first_meta, _ = parse_header(first_lesson_file)
                course_name = first_meta.get("Course") or course_dir.name.replace("-", " ").title()
            else:
                course_name = course_dir.name.replace("-", " ").title()

            course_page_id = lesson_pages.get(f"course:{course_dir.name}") or get_or_create_page(
                client, classroom_page_id, course_name, dry_run=dry_run
            )
            if course_page_id and not dry_run:
                lesson_pages[f"course:{course_dir.name}"] = course_page_id
                save_state(state)

            # Group lessons by Section metadata
            section_buckets: dict[str, list] = {}
            for lesson_file in sorted(course_dir.glob("*.md")):
                key = f"lesson:{lesson_file.relative_to(community_dir)}"
                if key in lesson_pages:
                    continue
                meta, content = parse_header(lesson_file)
                section = meta.get("Section", "")
                section_buckets.setdefault(section, []).append((lesson_file, meta, content, key))

            for section_name, section_lessons in section_buckets.items():
                # Determine parent: section page (if named) or course page directly
                if section_name:
                    sec_key = f"section:{course_dir.name}:{slugify(section_name)}"
                    section_page_id = lesson_pages.get(sec_key) or get_or_create_page(
                        client, course_page_id, section_name, dry_run=dry_run
                    )
                    if section_page_id and not dry_run:
                        lesson_pages[sec_key] = section_page_id
                        save_state(state)
                    parent_id = section_page_id or course_page_id
                else:
                    parent_id = course_page_id

                for lesson_file, meta, content, key in section_lessons:
                    lesson_title = meta.get("Lesson") or lesson_file.stem.replace("-", " ").title()
                    url = meta.get("URL", "")

                    if dry_run:
                        print(f"    [dry-run] lesson page: {lesson_title[:60]}")
                        lesson_pages[key] = "dry-run-page"
                        new_lessons += 1
                        continue

                    lesson_page_id = get_or_create_page(client, parent_id, lesson_title, dry_run=False)
                    if not lesson_page_id:
                        continue

                    blocks = []
                    if url:
                        blocks.append({"object": "block", "type": "bookmark", "bookmark": {"url": url}})
                        blocks.append(_divider())
                    clean_content = clean_lesson_content(content, url)
                    blocks.extend(markdown_to_blocks(clean_content))

                    files_dir = course_dir / "files"
                    if files_dir.exists():
                        file_list = [f.name for f in files_dir.iterdir() if f.is_file()]
                        if file_list:
                            blocks.append(_divider())
                            blocks.append(_heading(3, "📎 Files"))
                            for fname in file_list:
                                blocks.extend(_paragraph(f"• {fname}"))

                    try:
                        append_blocks_batched(client, lesson_page_id, blocks)
                    except Exception as e:
                        print(f"    ERROR appending blocks to '{lesson_title}': {e}")

                    lesson_pages[key] = lesson_page_id
                    save_state(state)
                    new_lessons += 1
                    print(f"    Created lesson: {lesson_title[:60]}")

        if new_lessons == 0 and not dry_run:
            print(f"  Classroom: all lessons already in Notion")
        elif new_lessons:
            action = "Would create" if dry_run else "Created"
            print(f"  {action} {new_lessons} lesson page(s)")

    # Posts — monthly buckets under "💬 Posts", each post rendered Skool-style
    posts_dir = community_dir / "posts"
    if posts_dir.exists():
        posts_page_id = get_or_create_posts_page(client, community_page_id, notion_cs, dry_run)
        if not posts_page_id:
            return
        if not dry_run:
            save_state(state)

        synced_posts = notion_cs.setdefault("synced_posts", {})
        month_pages = notion_cs.setdefault("month_pages", {})
        new_posts = 0

        for post_file in sorted(posts_dir.glob("*.md"), reverse=True):
            meta, raw_content = parse_header(post_file)
            # Dedup by PostID (stable across re-scrapes), not filename (changes with date prefix)
            post_id_key = meta.get("PostID") or post_file.stem
            key = post_id_key
            if key in synced_posts:
                continue

            parsed = extract_post_structured(raw_content)
            title = meta.get("Title") or post_id_key
            post_url = meta.get("URL", "")
            scraped = meta.get("Scraped", "")

            # Derive YYYY-MM bucket from Scraped timestamp or filename
            month_key = ""
            if scraped and len(scraped) >= 7:
                month_key = scraped[:7]
            elif len(key) >= 7 and key[:7].replace('-', '').isdigit():
                month_key = key[:7]

            if dry_run:
                print(f"    [dry-run] post ({month_key}): {title[:60]}")
                synced_posts[key] = "dry-run"
                new_posts += 1
                continue

            # Get or create month bucket page
            if month_key:
                if month_key not in month_pages:
                    mp_id = get_or_create_page(client, posts_page_id, month_key, dry_run=False)
                    if mp_id:
                        month_pages[month_key] = mp_id
                        save_state(state)
                post_parent_id = month_pages.get(month_key, posts_page_id)
            else:
                post_parent_id = posts_page_id

            for _attempt in range(4):
                try:
                    page = client.pages.create(
                        parent={"type": "page_id", "page_id": post_parent_id},
                        properties={"title": [{"text": {"content": title[:100]}}]},
                    )
                    time.sleep(0.35)  # ~3 req/s — Notion rate limit
                    break
                except Exception as _e:
                    if "rate" in str(_e).lower() and _attempt < 3:
                        time.sleep(15 * (_attempt + 1))
                        continue
                    print(f"    ERROR creating post '{title}': {_e}")
                    page = None
                    break
            if not page:
                continue
            try:
                post_page_id = page["id"]

                # Build Skool-style blocks
                blocks = []

                # Header: author • date • link
                header_parts = []
                if parsed["author"]:
                    header_parts.append(f"👤 {parsed['author']}")
                if parsed["date"]:
                    header_parts.append(f"📅 {parsed['date']}")
                if post_url:
                    header_parts.append(f"🔗 {post_url}")
                if header_parts:
                    blocks.extend(_paragraph("  •  ".join(header_parts)))
                blocks.append(_divider())

                # Post body
                if parsed["body"]:
                    blocks.extend(markdown_to_blocks(parsed["body"]))

                # Comments
                if parsed["comments"]:
                    blocks.append(_divider())
                    blocks.append(_heading(3, f"💬 Comments ({len(parsed['comments'])})"))
                    for comment in parsed["comments"]:
                        c_header = f"{comment['author']}  •  {comment['date']}" if comment.get("author") else ""
                        if c_header:
                            blocks.extend(_paragraph(f"**{comment['author']}**  •  {comment['date']}"))
                        if comment.get("body"):
                            blocks.append(_callout(comment["body"][:2000], "💬"))

                if blocks:
                    try:
                        append_blocks_batched(client, post_page_id, blocks)
                    except Exception as e:
                        print(f"    WARN: Failed to append blocks for '{title[:60]}': {e}")

                synced_posts[key] = post_page_id
                save_state(state)
                new_posts += 1
            except Exception as e:
                print(f"    ERROR creating post '{title}': {e}")

        if new_posts == 0 and not dry_run:
            print(f"  Posts: all already in Notion")
        elif new_posts:
            action = "Would create" if dry_run else "Created"
            print(f"  {action} {new_posts} post page(s)")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Sync scraped Skool content to Notion")
    parser.add_argument("--community", help="Sync only this community slug")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--categorize", action="store_true",
                        help="(stub) AI-categorize posts by topic — no-op, reserved for future Claude API pass")
    parser.add_argument("--rebuild-posts", action="store_true",
                        help="Clear synced_posts state so all posts are re-created with current formatting")
    parser.add_argument("--rebuild-classroom", action="store_true",
                        help="Archive old course pages in Notion and re-sync all lessons with current formatting")
    parser.add_argument("--root-page-id", help="Notion page ID to use as root (saves to state for future runs)")
    args = parser.parse_args()

    communities = json.loads(COMMUNITIES_FILE.read_text())["communities"]
    if args.community:
        communities = [c for c in communities if c["slug"] == args.community]
        if not communities:
            print(f"ERROR: Community '{args.community}' not found")
            sys.exit(1)

    client = get_client()
    state = load_state()

    # Allow passing root page ID directly on first run
    if args.root_page_id and not state.get("notion_root_page_id"):
        state["notion_root_page_id"] = args.root_page_id
        save_state(state)
        print(f"  Root page ID saved: {args.root_page_id}")

    root_page_id = get_or_create_root_page(client, state, args.dry_run)
    if not root_page_id:
        print("ERROR: Could not get/create root Notion page.")
        sys.exit(1)

    if args.rebuild_posts:
        for community in communities:
            slug = community["slug"]
            notion_cs = state.get(slug, {}).get("notion", {})
            notion_cs.pop("synced_posts", None)
            notion_cs.pop("month_pages", None)
            notion_cs.pop("notion_posts_page_id", None)
            print(f"  [rebuild] Cleared posts state for {slug} — will re-sync with current formatting")
        save_state(state)

    if args.rebuild_classroom:
        for community in communities:
            slug = community["slug"]
            notion_cs = state.get(slug, {}).get("notion", {})
            lesson_pages = notion_cs.get("lesson_pages", {})
            # Archive all existing course pages so fresh ones are created
            archived = 0
            for key, page_id in list(lesson_pages.items()):
                if not isinstance(page_id, str) or page_id.startswith("dry-run"):
                    continue
                try:
                    client.pages.update(page_id=page_id, archived=True)
                    archived += 1
                except Exception:
                    pass
            notion_cs.pop("lesson_pages", None)
            notion_cs.pop("classroom_page_id", None)
            print(f"  [rebuild] Archived {archived} classroom pages for {slug} — will re-sync")
        save_state(state)

    for community in communities:
        sync_community(client, community["slug"], community["name"],
                       root_page_id, state, args.dry_run)

    save_state(state)
    print("\nDone.")


if __name__ == "__main__":
    main()
