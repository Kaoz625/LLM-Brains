#!/usr/bin/env python3
"""
wiki_compiler.py — Karpathy-style wiki compiler.

Reads raw staging files and compiles them into structured wiki articles.
Supports: KEEP, UPDATE, MERGE, SUPERSEDE, ARCHIVE operations on existing entries.

Usage:
    python wiki_compiler.py --compile brain/raw/
    python wiki_compiler.py --rebuild           # full wiki rebuild
    python wiki_compiler.py --list              # list all wiki entries
"""

import argparse
import json
import os
import re
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import anthropic
from dotenv import load_dotenv

load_dotenv()

BRAIN_DIR = Path(os.getenv("BRAIN_DIR", "./brain"))
WIKI_DIR = BRAIN_DIR / "knowledge" / "wiki"
WIKI_DB = BRAIN_DIR / "wiki.db"

WIKI_SCHEMA = """
CREATE TABLE IF NOT EXISTS wiki_entries (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    title           TEXT UNIQUE NOT NULL,
    slug            TEXT UNIQUE NOT NULL,
    summary         TEXT,
    content         TEXT NOT NULL,
    key_concepts    TEXT,   -- JSON array
    cross_links     TEXT,   -- JSON array of [[wikilinks]]
    source_citations TEXT,  -- JSON array
    status          TEXT DEFAULT 'active',  -- active|archived|superseded
    conflicts       TEXT,   -- JSON array of conflict descriptions
    created_at      TEXT DEFAULT (datetime('now')),
    last_updated    TEXT DEFAULT (datetime('now'))
);

CREATE VIRTUAL TABLE IF NOT EXISTS wiki_fts USING fts5(
    title,
    summary,
    content,
    key_concepts,
    content='wiki_entries',
    content_rowid='id'
);

CREATE TRIGGER IF NOT EXISTS wiki_ai AFTER INSERT ON wiki_entries BEGIN
    INSERT INTO wiki_fts(rowid, title, summary, content, key_concepts)
    VALUES (new.id, new.title, new.summary, new.content, new.key_concepts);
END;

CREATE TRIGGER IF NOT EXISTS wiki_ad AFTER DELETE ON wiki_entries BEGIN
    INSERT INTO wiki_fts(wiki_fts, rowid, title, summary, content, key_concepts)
    VALUES ('delete', old.id, old.title, old.summary, old.content, old.key_concepts);
END;

CREATE TRIGGER IF NOT EXISTS wiki_au AFTER UPDATE ON wiki_entries BEGIN
    INSERT INTO wiki_fts(wiki_fts, rowid, title, summary, content, key_concepts)
    VALUES ('delete', old.id, old.title, old.summary, old.content, old.key_concepts);
    INSERT INTO wiki_fts(rowid, title, summary, content, key_concepts)
    VALUES (new.id, new.title, new.summary, new.content, new.key_concepts);
END;
"""


def get_db() -> sqlite3.Connection:
    WIKI_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(WIKI_DB))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(WIKI_SCHEMA)
    conn.commit()
    return conn


def slugify(title: str) -> str:
    slug = re.sub(r"[^\w\s-]", "", title.lower())
    slug = re.sub(r"[\s_]+", "-", slug).strip("-")
    return slug[:80]


# ---------------------------------------------------------------------------
# Claude integration
# ---------------------------------------------------------------------------

def get_client() -> anthropic.Anthropic:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("ANTHROPIC_API_KEY not set", file=sys.stderr)
        sys.exit(1)
    return anthropic.Anthropic(api_key=api_key)


def decide_operation(client: anthropic.Anthropic, new_content: str,
                     existing_entry: Optional[dict]) -> dict:
    """
    Ask Claude what to do with new content vs existing wiki entry.
    Returns: {operation: KEEP|UPDATE|MERGE|SUPERSEDE|ARCHIVE, reasoning, merged_content}
    """
    if not existing_entry:
        return {"operation": "CREATE", "reasoning": "No existing entry", "merged_content": None}

    prompt = f"""You are a wiki knowledge compiler. Decide how to handle new information vs an existing wiki entry.

EXISTING ENTRY (title: {existing_entry['title']}):
{existing_entry['content'][:3000]}

NEW INFORMATION:
{new_content[:3000]}

Decide which operation to apply:
- KEEP: New info is already covered; no change needed
- UPDATE: New info adds small details/corrections to existing entry
- MERGE: Both have valuable unique info; combine them
- SUPERSEDE: New info is more complete/accurate; replace old
- ARCHIVE: Existing entry is outdated; mark as archived, create new

Return JSON:
{{
  "operation": "KEEP|UPDATE|MERGE|SUPERSEDE|ARCHIVE",
  "reasoning": "brief explanation",
  "conflicts": ["conflict description if any"],
  "merged_content": "full merged markdown article if operation is MERGE or UPDATE (null otherwise)"
}}

If merged_content is provided, include [[wikilinks]] for key concepts."""

    try:
        response = client.messages.create(
            model="claude-opus-4-5",
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}]
        )
        text = response.content[0].text
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            return json.loads(match.group())
    except Exception as e:
        print(f"Claude operation decision error: {e}", file=sys.stderr)
    return {"operation": "MERGE", "reasoning": "Fallback", "merged_content": None}


def compile_to_wiki(client: anthropic.Anthropic, content: str, source: str) -> dict:
    """Compile raw content into a structured wiki article."""
    prompt = f"""You are a personal wiki compiler in the style of Andrej Karpathy's notes.
Transform this raw content into a clean, structured wiki article.

Source: {source}
Content:
{content[:6000]}

Return JSON:
{{
  "title": "clear, specific title",
  "summary": "2-3 sentence executive summary",
  "key_concepts": ["concept1", "concept2", "concept3"],
  "cross_links": ["[[related concept]]", "[[person name]]"],
  "source_citations": ["source1", "source2"],
  "content": "full structured markdown article with:\\n- ## sections\\n- [[wikilinks]] for concepts\\n- bullet points for facts\\n- code blocks if applicable\\n- tables if data-heavy",
  "tags": ["tag1", "tag2"]
}}

Make the article densely informative but readable. Every important noun should be a [[wikilink]]."""

    try:
        response = client.messages.create(
            model="claude-opus-4-5",
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}]
        )
        text = response.content[0].text
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            return json.loads(match.group())
    except Exception as e:
        print(f"Claude compile error: {e}", file=sys.stderr)

    return {
        "title": Path(source).stem,
        "summary": content[:200],
        "key_concepts": [],
        "cross_links": [],
        "source_citations": [source],
        "content": content,
        "tags": []
    }


# ---------------------------------------------------------------------------
# Wiki entry management
# ---------------------------------------------------------------------------

def save_wiki_entry(entry: dict, operation: str = "CREATE") -> Path:
    """Save a wiki entry to both SQLite and markdown file."""
    conn = get_db()
    title = entry["title"]
    slug = slugify(title)
    now = datetime.now().isoformat()

    # Add conflict markers to content if any
    conflicts = entry.get("conflicts", [])
    content = entry.get("content", "")
    if conflicts:
        conflict_block = "\n\n## Conflicts\n" + "\n".join(
            f"- [CONFLICT: {c}]" for c in conflicts
        )
        content += conflict_block

    conn.execute("""
        INSERT INTO wiki_entries
            (title, slug, summary, content, key_concepts, cross_links,
             source_citations, status, conflicts, last_updated)
        VALUES (?, ?, ?, ?, ?, ?, ?, 'active', ?, ?)
        ON CONFLICT(title) DO UPDATE SET
            summary=excluded.summary,
            content=excluded.content,
            key_concepts=excluded.key_concepts,
            cross_links=excluded.cross_links,
            source_citations=excluded.source_citations,
            conflicts=excluded.conflicts,
            last_updated=excluded.last_updated
    """, (
        title, slug,
        entry.get("summary", ""),
        content,
        json.dumps(entry.get("key_concepts", [])),
        json.dumps(entry.get("cross_links", [])),
        json.dumps(entry.get("source_citations", [])),
        json.dumps(conflicts),
        now,
    ))
    conn.commit()
    conn.close()

    # Write markdown file
    WIKI_DIR.mkdir(parents=True, exist_ok=True)
    md_path = WIKI_DIR / f"{slug}.md"
    tags = entry.get("tags", [])
    cross_links_str = " ".join(entry.get("cross_links", []))

    md_content = f"""---
title: {title}
slug: {slug}
summary: "{entry.get('summary', '').replace('"', "'")}"
key_concepts: {json.dumps(entry.get('key_concepts', []))}
cross_links: {json.dumps(entry.get('cross_links', []))}
sources: {json.dumps(entry.get('source_citations', []))}
tags: {json.dumps(tags)}
operation: {operation}
last_updated: {now}
---

# {title}

> {entry.get('summary', '')}

**Links:** {cross_links_str}

---

{content}

---
*Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')} | Operation: {operation}*
"""
    md_path.write_text(md_content, encoding="utf-8")
    return md_path


def get_existing_entry(title: str) -> Optional[dict]:
    """Look up an existing wiki entry by title or similar title."""
    conn = get_db()
    # Exact match first
    row = conn.execute(
        "SELECT * FROM wiki_entries WHERE title=? AND status='active'", (title,)
    ).fetchone()
    if row:
        conn.close()
        return dict(row)
    # FTS fallback
    try:
        rows = conn.execute("""
            SELECT we.* FROM wiki_fts
            JOIN wiki_entries we ON wiki_fts.rowid = we.id
            WHERE wiki_fts MATCH ? AND we.status='active'
            LIMIT 1
        """, (title,)).fetchall()
        if rows:
            conn.close()
            return dict(rows[0])
    except Exception:
        pass
    conn.close()
    return None


def archive_entry(title: str):
    """Mark a wiki entry as archived."""
    conn = get_db()
    conn.execute(
        "UPDATE wiki_entries SET status='archived', last_updated=datetime('now') WHERE title=?",
        (title,)
    )
    conn.commit()
    conn.close()

    # Rename the markdown file
    slug = slugify(title)
    md_path = WIKI_DIR / f"{slug}.md"
    if md_path.exists():
        archived_path = WIKI_DIR / f"_archived_{slug}.md"
        md_path.rename(archived_path)


# ---------------------------------------------------------------------------
# Compilation pipeline
# ---------------------------------------------------------------------------

def compile_file(path: Path, client: anthropic.Anthropic) -> dict:
    """Compile a single file into the wiki."""
    content = path.read_text(encoding="utf-8", errors="replace")
    print(f"  Compiling: {path.name}")

    # Generate wiki article
    entry_data = compile_to_wiki(client, content, path.name)
    title = entry_data.get("title", path.stem)

    # Check for existing entry
    existing = get_existing_entry(title)

    # Decide operation
    op_result = decide_operation(client, content, existing)
    operation = op_result.get("operation", "CREATE")
    print(f"    -> Operation: {operation} for '{title}'")

    # Handle conflicts
    if op_result.get("conflicts"):
        entry_data["conflicts"] = op_result["conflicts"]

    if operation == "KEEP":
        return {"operation": "KEEP", "title": title, "path": None}

    elif operation == "ARCHIVE" and existing:
        archive_entry(title)
        md_path = save_wiki_entry(entry_data, operation="SUPERSEDE")
        return {"operation": operation, "title": title, "path": str(md_path)}

    elif operation in ("UPDATE", "MERGE") and op_result.get("merged_content"):
        entry_data["content"] = op_result["merged_content"]
        md_path = save_wiki_entry(entry_data, operation=operation)
        return {"operation": operation, "title": title, "path": str(md_path)}

    else:
        # CREATE or SUPERSEDE
        md_path = save_wiki_entry(entry_data, operation=operation)
        return {"operation": operation, "title": title, "path": str(md_path)}


def compile_directory(directory: str | Path, client: anthropic.Anthropic) -> list[dict]:
    """Compile all files in a directory into the wiki."""
    directory = Path(directory)
    results = []
    files = list(directory.glob("*.md")) + list(directory.glob("*.txt"))
    print(f"Found {len(files)} files to compile in {directory}")

    for path in sorted(files):
        if path.name.startswith("."):
            continue
        try:
            result = compile_file(path, client)
            results.append(result)
        except Exception as e:
            print(f"  Error compiling {path.name}: {e}", file=sys.stderr)
            results.append({"operation": "ERROR", "title": path.stem, "error": str(e)})

    return results


def list_wiki_entries() -> list[dict]:
    """List all active wiki entries."""
    conn = get_db()
    rows = conn.execute("""
        SELECT title, slug, summary, status, last_updated
        FROM wiki_entries
        WHERE status='active'
        ORDER BY last_updated DESC
    """).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def search_wiki(query: str, limit: int = 10) -> list[dict]:
    """Full-text search over wiki entries."""
    conn = get_db()
    try:
        rows = conn.execute("""
            SELECT we.title, we.slug, we.summary, we.status,
                   bm25(wiki_fts) AS score
            FROM wiki_fts
            JOIN wiki_entries we ON wiki_fts.rowid = we.id
            WHERE wiki_fts MATCH ? AND we.status='active'
            ORDER BY score
            LIMIT ?
        """, (query, limit)).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception as e:
        conn.close()
        print(f"Wiki search error: {e}", file=sys.stderr)
        return []


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Karpathy-style wiki compiler")
    parser.add_argument("--compile", type=str, metavar="DIR",
                        help="Compile all files in directory into wiki")
    parser.add_argument("--file", type=str, help="Compile a single file")
    parser.add_argument("--rebuild", action="store_true",
                        help="Rebuild entire wiki from knowledge/ directory")
    parser.add_argument("--list", action="store_true", help="List all wiki entries")
    parser.add_argument("--search", type=str, help="Search wiki entries")
    args = parser.parse_args()

    if args.list:
        entries = list_wiki_entries()
        print(f"\nWiki entries ({len(entries)} active):\n{'='*50}")
        for e in entries:
            print(f"  [{e['status']}] {e['title']}")
            print(f"         {e['summary'][:80]}...")
            print(f"         Updated: {e['last_updated'][:10]}\n")
        return

    if args.search:
        results = search_wiki(args.search)
        print(f"\nSearch results for '{args.search}':\n{'='*50}")
        for r in results:
            print(f"  {r['title']}")
            print(f"  {r['summary'][:100]}...\n")
        return

    client = get_client()

    if args.file:
        result = compile_file(Path(args.file), client)
        print(f"\nResult: {result}")

    elif args.compile:
        results = compile_directory(args.compile, client)
        ops = {}
        for r in results:
            ops[r["operation"]] = ops.get(r["operation"], 0) + 1
        print(f"\nWiki compilation complete:")
        for op, count in ops.items():
            print(f"  {op}: {count}")

    elif args.rebuild:
        knowledge_dir = BRAIN_DIR / "knowledge"
        results = compile_directory(knowledge_dir, client)
        print(f"\nFull rebuild complete: {len(results)} entries processed")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
