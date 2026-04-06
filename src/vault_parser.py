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
