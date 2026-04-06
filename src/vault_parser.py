"""
vault_parser.py
---------------
Parse an Obsidian vault directory.

For every .md file it extracts:
  - YAML frontmatter (via python-frontmatter)
  - Note body (markdown text)
  - [[wikilinks]] referenced in the body
  - #tags from the body and frontmatter
  - Last-modified timestamp (file mtime)

Returns a list of Note dataclasses ready for the database layer.
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import frontmatter  # python-frontmatter


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Note:
    path: str                    # relative path inside the vault
    title: str                   # file stem (filename without .md)
    content: str                 # full markdown body (after frontmatter)
    tags: List[str]              # combined frontmatter + inline #tags
    backlinks: List[str]         # [[wikilinks]] found in body
    modified_at: int             # Unix timestamp (seconds)
    frontmatter: dict = field(default_factory=dict)  # raw frontmatter dict


# ---------------------------------------------------------------------------
# Regex helpers
# ---------------------------------------------------------------------------

# Matches [[Link]], [[Link|Alias]], [[Link#Section|Alias]]
_WIKILINK_RE = re.compile(r"\[\[([^\]|#]+)(?:[#|][^\]]*)?\]\]")

# Matches #tag (not inside code blocks or URLs)
_INLINE_TAG_RE = re.compile(r"(?<!\S)#([A-Za-z_][A-Za-z0-9_/-]*)")


def _extract_wikilinks(text: str) -> List[str]:
    """Return a deduplicated list of wikilink targets from *text*."""
    return list(dict.fromkeys(_WIKILINK_RE.findall(text)))


def _extract_inline_tags(text: str) -> List[str]:
    """Return inline #tags found in *text*."""
    return list(dict.fromkeys(_INLINE_TAG_RE.findall(text)))


def _normalize_tags(fm: dict, body: str) -> List[str]:
    """Merge frontmatter tags with inline body tags into one deduplicated list."""
    fm_tags: list = []
    raw = fm.get("tags") or fm.get("tag") or []
    if isinstance(raw, str):
        # Handle comma-separated or space-separated tag strings
        fm_tags = [t.strip().lstrip("#") for t in re.split(r"[,\s]+", raw) if t.strip()]
    elif isinstance(raw, list):
        fm_tags = [str(t).strip().lstrip("#") for t in raw if t]

    inline_tags = _extract_inline_tags(body)
    seen: dict = {}
    for t in fm_tags + inline_tags:
        seen[t] = None
    return list(seen.keys())


# ---------------------------------------------------------------------------
# Main parser
# ---------------------------------------------------------------------------

def parse_vault(vault_path: str, verbose: bool = False) -> List[Note]:
    """
    Walk *vault_path* and parse every .md file.

    Parameters
    ----------
    vault_path:
        Absolute or relative path to the root of the Obsidian vault.
    verbose:
        If True, print each file as it is parsed.

    Returns
    -------
    A list of :class:`Note` objects.
    """
    vault_root = Path(vault_path).expanduser().resolve()
    if not vault_root.is_dir():
        raise NotADirectoryError(f"Vault path does not exist or is not a directory: {vault_root}")

    notes: List[Note] = []

    for md_file in vault_root.rglob("*.md"):
        # Skip hidden folders (e.g. .obsidian, .trash)
        if any(part.startswith(".") for part in md_file.relative_to(vault_root).parts):
            continue

        try:
            raw_bytes = md_file.read_bytes()
            post = frontmatter.loads(raw_bytes.decode("utf-8", errors="replace"))
        except Exception as exc:  # noqa: BLE001
            if verbose:
                print(f"  [skip] {md_file}: {exc}")
            continue

        rel_path = str(md_file.relative_to(vault_root))
        title = md_file.stem
        body: str = post.content or ""
        fm_dict: dict = dict(post.metadata)

        # Prefer explicit 'title' frontmatter key if present
        if "title" in fm_dict and fm_dict["title"]:
            title = str(fm_dict["title"])

        tags = _normalize_tags(fm_dict, body)
        backlinks = _extract_wikilinks(body)
        mtime = int(md_file.stat().st_mtime)

        notes.append(
            Note(
                path=rel_path,
                title=title,
                content=body,
                tags=tags,
                backlinks=backlinks,
                modified_at=mtime,
                frontmatter=fm_dict,
            )
        )

        if verbose:
            print(f"  parsed: {rel_path}  ({len(body)} chars, {len(tags)} tags, {len(backlinks)} links)")

    return notes
src/vault_parser.py — VaultParser class.

Parses Obsidian-style markdown vaults, extracts frontmatter, wikilinks,
tags, backlinks, and builds a graph of the vault's structure.
"""

import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional


class VaultParser:
    """
    Parses Obsidian-style markdown vaults.

    Features:
    - Frontmatter YAML parsing
    - [[wikilink]] extraction (including [[link|alias]] format)
    - #hashtag extraction
    - Backlink graph construction
    - Orphan note detection
    - Dead link detection
    """

    FRONTMATTER_RE = re.compile(r'^---\n(.*?)\n---\n', re.DOTALL)
    WIKILINK_RE = re.compile(r'\[\[([^\]|]+)(?:\|([^\]]+))?\]\]')
    HASHTAG_RE = re.compile(r'(?<!\w)#([a-zA-Z][a-zA-Z0-9_/-]+)')
    HEADING_RE = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)
    CODEBLOCK_RE = re.compile(r'```.*?```', re.DOTALL)
    INLINE_CODE_RE = re.compile(r'`[^`]+`')
    EMBED_RE = re.compile(r'!\[\[([^\]]+)\]\]')

    def __init__(self, vault_path: Optional[Path] = None):
        self.vault_path = Path(vault_path) if vault_path else None
        self._notes: dict[str, dict] = {}  # slug -> note data
        self._backlinks: dict[str, set[str]] = defaultdict(set)  # target -> sources
        self._parsed = False

    # ------------------------------------------------------------------
    # Single file parsing
    # ------------------------------------------------------------------

    def parse_file(self, path: Path) -> dict:
        """
        Parse a single markdown file.

        Returns dict with:
          - path, title, slug
          - frontmatter (dict)
          - content (body without frontmatter)
          - wikilinks (list of link targets)
          - hashtags (list of tags)
          - headings (list of heading dicts)
          - embeds (list of embedded file names)
          - word_count
        """
        try:
            raw = path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            return {"path": str(path), "error": str(e)}

        # Parse frontmatter
        frontmatter = {}
        content = raw
        fm_match = self.FRONTMATTER_RE.match(raw)
        if fm_match:
            frontmatter = self._parse_yaml_simple(fm_match.group(1))
            content = raw[fm_match.end():]

        # Remove code blocks before extracting links (avoid false positives)
        text_no_code = self.CODEBLOCK_RE.sub("", content)
        text_no_code = self.INLINE_CODE_RE.sub("", text_no_code)

        # Extract wikilinks
        wikilinks = []
        for match in self.WIKILINK_RE.finditer(text_no_code):
            target = match.group(1).strip()
            alias = match.group(2)
            # Remove heading anchors (e.g., [[Page#Section]] -> Page)
            target = target.split("#")[0].strip()
            if target:
                wikilinks.append({"target": target, "alias": alias})

        # Extract embeds
        embeds = [m.group(1).strip() for m in self.EMBED_RE.finditer(text_no_code)]

        # Extract hashtags (not in frontmatter)
        hashtags = list(set(self.HASHTAG_RE.findall(text_no_code)))

        # Extract headings
        headings = []
        for m in self.HEADING_RE.finditer(content):
            headings.append({
                "level": len(m.group(1)),
                "text": m.group(2).strip(),
            })

        # Determine title
        title = frontmatter.get("title", "")
        if not title and headings:
            title = headings[0]["text"]
        if not title:
            title = path.stem.replace("-", " ").replace("_", " ").title()

        slug = self._slugify(path.stem)

        return {
            "path": str(path),
            "title": str(title),
            "slug": slug,
            "frontmatter": frontmatter,
            "content": content,
            "wikilinks": wikilinks,
            "link_targets": [w["target"] for w in wikilinks],
            "hashtags": hashtags,
            "headings": headings,
            "embeds": embeds,
            "word_count": len(content.split()),
            "char_count": len(content),
            "last_modified": datetime.fromtimestamp(
                path.stat().st_mtime
            ).isoformat() if path.exists() else "",
        }

    # ------------------------------------------------------------------
    # Vault-level parsing
    # ------------------------------------------------------------------

    def parse_vault(self, vault_path: Optional[Path] = None) -> dict:
        """
        Parse an entire vault directory.

        Returns dict with:
          - notes: dict of slug -> note data
          - backlinks: dict of note_title -> list of notes linking to it
          - orphans: list of note slugs with no incoming links
          - dead_links: list of (source, target) tuples for broken links
          - stats: vault statistics
        """
        vp = vault_path or self.vault_path
        if vp is None:
            raise ValueError("No vault path provided")
        vp = Path(vp)

        notes = {}
        for md_file in vp.rglob("*.md"):
            if any(part.startswith(".") for part in md_file.parts):
                continue  # Skip hidden directories
            note = self.parse_file(md_file)
            slug = note.get("slug", md_file.stem)
            notes[slug] = note

        self._notes = notes
        self._build_backlinks()
        self._parsed = True

        orphans = self._find_orphans()
        dead_links = self._find_dead_links()

        return {
            "notes": notes,
            "backlinks": {k: list(v) for k, v in self._backlinks.items()},
            "orphans": orphans,
            "dead_links": dead_links,
            "stats": self._compute_stats(notes, orphans, dead_links),
        }

    def _build_backlinks(self):
        """Build reverse-link graph from parsed notes."""
        self._backlinks = defaultdict(set)
        for slug, note in self._notes.items():
            for target in note.get("link_targets", []):
                target_slug = self._slugify(target)
                self._backlinks[target_slug].add(slug)

    def _find_orphans(self) -> list[str]:
        """Find notes with no incoming links."""
        linked_slugs = set(self._backlinks.keys())
        return [
            slug for slug in self._notes
            if slug not in linked_slugs
        ]

    def _find_dead_links(self) -> list[dict]:
        """Find links pointing to non-existent notes."""
        dead = []
        known_slugs = set(self._notes.keys())
        # Also include known file stems directly
        known_titles = {
            note["title"].lower() for note in self._notes.values()
        }

        for slug, note in self._notes.items():
            for target in note.get("link_targets", []):
                target_slug = self._slugify(target)
                target_lower = target.lower()
                if (target_slug not in known_slugs and
                        target_lower not in known_titles):
                    dead.append({
                        "source": note.get("title", slug),
                        "source_path": note.get("path", ""),
                        "target": target,
                        "target_slug": target_slug,
                    })
        return dead

    def _compute_stats(self, notes: dict, orphans: list, dead_links: list) -> dict:
        """Compute vault statistics."""
        total_words = sum(n.get("word_count", 0) for n in notes.values())
        total_links = sum(len(n.get("link_targets", [])) for n in notes.values())
        all_tags: set[str] = set()
        for n in notes.values():
            all_tags.update(n.get("hashtags", []))
            tags = n.get("frontmatter", {}).get("tags", [])
            if isinstance(tags, list):
                all_tags.update(str(t) for t in tags)

        return {
            "total_notes": len(notes),
            "total_words": total_words,
            "total_links": total_links,
            "unique_tags": len(all_tags),
            "orphaned_notes": len(orphans),
            "dead_links": len(dead_links),
            "avg_words_per_note": total_words // max(len(notes), 1),
            "avg_links_per_note": total_links / max(len(notes), 1),
        }

    # ------------------------------------------------------------------
    # Query methods (require vault to be parsed first)
    # ------------------------------------------------------------------

    def get_backlinks(self, note_title: str) -> list[dict]:
        """Get all notes that link to the given note."""
        if not self._parsed:
            raise RuntimeError("Call parse_vault() first")
        slug = self._slugify(note_title)
        source_slugs = self._backlinks.get(slug, set())
        return [
            self._notes[s] for s in source_slugs
            if s in self._notes
        ]

    def get_note(self, title_or_slug: str) -> Optional[dict]:
        """Get a note by title or slug."""
        slug = self._slugify(title_or_slug)
        return self._notes.get(slug)

    def get_linked_notes(self, note_title: str) -> list[dict]:
        """Get all notes linked from the given note."""
        note = self.get_note(note_title)
        if not note:
            return []
        result = []
        for target in note.get("link_targets", []):
            linked = self.get_note(target)
            if linked:
                result.append(linked)
        return result

    def get_tag_notes(self, tag: str) -> list[dict]:
        """Get all notes with a specific hashtag or frontmatter tag."""
        if not self._parsed:
            return []
        tag_lower = tag.lower().lstrip("#")
        matches = []
        for note in self._notes.values():
            tags = note.get("hashtags", [])
            fm_tags = note.get("frontmatter", {}).get("tags", [])
            all_tags = [t.lower() for t in tags]
            if isinstance(fm_tags, list):
                all_tags.extend(str(t).lower() for t in fm_tags)
            if tag_lower in all_tags:
                matches.append(note)
        return matches

    def search_content(self, query: str) -> list[dict]:
        """Simple text search across note contents."""
        q = query.lower()
        results = []
        for note in self._notes.values():
            content = note.get("content", "").lower()
            title = note.get("title", "").lower()
            if q in content or q in title:
                score = content.count(q) + title.count(q) * 3
                results.append({**note, "search_score": score})
        results.sort(key=lambda x: x["search_score"], reverse=True)
        return results

    def get_most_linked(self, top_k: int = 10) -> list[dict]:
        """Get the most-linked notes (highest in-degree)."""
        if not self._parsed:
            return []
        scored = []
        for slug, backlinks in self._backlinks.items():
            note = self._notes.get(slug)
            if note:
                scored.append({**note, "backlink_count": len(backlinks)})
        scored.sort(key=lambda x: x["backlink_count"], reverse=True)
        return scored[:top_k]

    def export_graph(self) -> dict:
        """Export the link graph as nodes + edges for visualization."""
        if not self._parsed:
            return {"nodes": [], "edges": []}

        nodes = [
            {
                "id": slug,
                "title": note.get("title", slug),
                "route": note.get("frontmatter", {}).get("route", ""),
                "word_count": note.get("word_count", 0),
                "backlink_count": len(self._backlinks.get(slug, set())),
            }
            for slug, note in self._notes.items()
        ]

        edges = []
        for slug, note in self._notes.items():
            for target in note.get("link_targets", []):
                target_slug = self._slugify(target)
                if target_slug in self._notes:
                    edges.append({"source": slug, "target": target_slug})

        return {"nodes": nodes, "edges": edges}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _slugify(text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r"[^\w\s-]", "", text)
        text = re.sub(r"[\s_]+", "-", text).strip("-")
        return text

    @staticmethod
    def _parse_yaml_simple(yaml_str: str) -> dict:
        """Simple YAML parser for frontmatter (handles key: value pairs)."""
        result = {}
        current_key = None
        current_list = None

        for line in yaml_str.splitlines():
            # List item
            if line.startswith("  - ") or line.startswith("- "):
                item = line.lstrip("- ").strip()
                if current_key and current_list is not None:
                    current_list.append(item)
                continue

            # Key: value
            if ":" in line and not line.startswith(" "):
                key, _, value = line.partition(":")
                key = key.strip()
                value = value.strip()

                if not value:
                    # Start of a block sequence
                    current_key = key
                    current_list = []
                    result[key] = current_list
                else:
                    current_key = None
                    current_list = None
                    # Strip quotes
                    value = value.strip('"\'')
                    # Parse booleans
                    if value.lower() == "true":
                        result[key] = True
                    elif value.lower() == "false":
                        result[key] = False
                    else:
                        # Inline list [a, b, c]
                        if value.startswith("[") and value.endswith("]"):
                            items = [v.strip().strip("'\"")
                                     for v in value[1:-1].split(",")]
                            result[key] = [i for i in items if i]
                        else:
                            result[key] = value

        return result

    def __repr__(self) -> str:
        if self._parsed:
            return (f"VaultParser({len(self._notes)} notes, "
                    f"{sum(len(v) for v in self._backlinks.values())} backlinks)")
        return f"VaultParser(path={self.vault_path}, not parsed)"


# ---------------------------------------------------------------------------
# Standalone usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) < 2:
        print("Usage: python vault_parser.py <vault_path> [--stats] [--graph]")
        sys.exit(1)

    vault_path = Path(sys.argv[1])
    parser = VaultParser(vault_path)
    result = parser.parse_vault()

    if "--graph" in sys.argv:
        print(json.dumps(parser.export_graph(), indent=2))
    elif "--stats" in sys.argv:
        print(json.dumps(result["stats"], indent=2))
    else:
        stats = result["stats"]
        print(f"\nVault: {vault_path}")
        print(f"Notes:       {stats['total_notes']}")
        print(f"Total words: {stats['total_words']:,}")
        print(f"Total links: {stats['total_links']:,}")
        print(f"Tags:        {stats['unique_tags']}")
        print(f"Orphans:     {stats['orphaned_notes']}")
        print(f"Dead links:  {stats['dead_links']}")

        if result["dead_links"]:
            print("\nDead links (first 10):")
            for dl in result["dead_links"][:10]:
                print(f"  [[{dl['target']}]] in {dl['source']}")
