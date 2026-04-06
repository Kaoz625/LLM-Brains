#!/usr/bin/env python3
"""
sqlite_rag.py — SQLite + Vector RAG Layer
Indexes your compiled brain/ files into a SQLite database with:
  - Layer 1: FTS5 full-text search (keyword matching)
  - Layer 2: sqlite-vec vector embeddings (semantic similarity)
  - Layer 3: Concept link graph (relationship reasoning)

Then provides a hybrid search interface that merges both layers
and reranks by combined relevance — the same way human memory works.

Usage:
    python sqlite_rag.py --build         # index all brain/ files
    python sqlite_rag.py --query "What do I know about transformers?"
    python sqlite_rag.py --query "What has been going on with my family?"
    python sqlite_rag.py --stats         # database statistics
    python sqlite_rag.py --export-mcp    # export as MCP server config

The .db file is your portable second brain — back it up, sync it, share it.
"""

import os
import re
import sys
import json
import sqlite3
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional

log = logging.getLogger("sqlite_rag")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT    = Path(__file__).parent / "brain"
DB_PATH = Path(__file__).parent / "brain.db"
sqlite_rag.py — Hybrid SQLite RAG search system.

Combines FTS5 full-text search with vector embeddings for semantic search.

Usage:
    python sqlite_rag.py --index brain/
    python sqlite_rag.py --search "query"
    python sqlite_rag.py --search "query" --limit 20
"""

import argparse
import json
import math
import os
import sqlite3
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from dotenv import load_dotenv

load_dotenv()

BRAIN_DIR = Path(os.getenv("BRAIN_DIR", "./brain"))
DB_PATH = BRAIN_DIR / "memory.db"


# ---------------------------------------------------------------------------
# Database setup
# ---------------------------------------------------------------------------

SCHEMA = """
-- ============================================================
-- Layer 1: Full-text search + metadata
-- ============================================================
SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS notes (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    path        TEXT UNIQUE NOT NULL,
    title       TEXT,
    content     TEXT,
    tags        TEXT DEFAULT '[]',
    backlinks   TEXT DEFAULT '[]',
    source_url  TEXT,
    source_type TEXT DEFAULT 'unknown',
    category    TEXT DEFAULT 'knowledge',
    word_count  INTEGER DEFAULT 0,
    created_at  INTEGER DEFAULT (strftime('%s', 'now')),
    modified_at INTEGER DEFAULT (strftime('%s', 'now'))
);

CREATE VIRTUAL TABLE IF NOT EXISTS notes_fts USING fts5(
    title, content, tags,
    content     TEXT NOT NULL,
    route       TEXT,
    tags        TEXT,
    created_at  TEXT DEFAULT (datetime('now')),
    updated_at  TEXT DEFAULT (datetime('now'))
);

CREATE VIRTUAL TABLE IF NOT EXISTS notes_fts USING fts5(
    title,
    content,
    tags,
    content='notes',
    content_rowid='id'
);

CREATE TABLE IF NOT EXISTS embeddings (
    note_id     INTEGER PRIMARY KEY REFERENCES notes(id) ON DELETE CASCADE,
    model       TEXT NOT NULL,
    vector_json TEXT NOT NULL,
    dim         INTEGER NOT NULL,
    created_at  TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS metadata (
    key   TEXT PRIMARY KEY,
    value TEXT
);

CREATE TRIGGER IF NOT EXISTS notes_ai AFTER INSERT ON notes BEGIN
    INSERT INTO notes_fts(rowid, title, content, tags)
    VALUES (new.id, new.title, new.content, new.tags);
END;

CREATE TRIGGER IF NOT EXISTS notes_ad AFTER DELETE ON notes BEGIN
    INSERT INTO notes_fts(notes_fts, rowid, title, content, tags)
    VALUES ('delete', old.id, old.title, old.content, old.tags);
END;

CREATE TRIGGER IF NOT EXISTS notes_au AFTER UPDATE ON notes BEGIN
    INSERT INTO notes_fts(notes_fts, rowid, title, content, tags)
    VALUES ('delete', old.id, old.title, old.content, old.tags);
    INSERT INTO notes_fts(rowid, title, content, tags)
    VALUES (new.id, new.title, new.content, new.tags);
END;

-- ============================================================
-- Layer 3: Concept relationship graph
-- ============================================================
CREATE TABLE IF NOT EXISTS concept_links (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    from_path   TEXT NOT NULL,
    to_path     TEXT NOT NULL,
    link_type   TEXT DEFAULT 'related',
    strength    REAL DEFAULT 0.5,
    UNIQUE(from_path, to_path, link_type)
);

-- ============================================================
-- Ingestion tracking
-- ============================================================
CREATE TABLE IF NOT EXISTS index_meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);
"""

def get_db(db_path: Path = DB_PATH) -> sqlite3.Connection:
    db = sqlite3.connect(str(db_path))
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=NORMAL")
    db.executescript(SCHEMA)
    db.commit()
    return db

# ---------------------------------------------------------------------------
# Embedding support
# ---------------------------------------------------------------------------

def get_embedder():
    """Return an embedding function. Tries local Ollama first, then OpenAI."""
    # Try Ollama (local, free, private)
    try:
        import urllib.request, json as _json
        def ollama_embed(text: str) -> list:
            payload = json.dumps({"model": "nomic-embed-text", "prompt": text[:2048]}).encode()
            req = urllib.request.Request(
                "http://localhost:11434/api/embeddings",
                data=payload,
                headers={"Content-Type": "application/json"}
            )
            resp = urllib.request.urlopen(req, timeout=10).read()
            return _json.loads(resp)["embedding"]
        # Test it
        ollama_embed("test")
        log.info("Using Ollama nomic-embed-text for embeddings (local)")
        return ollama_embed
    except Exception:
        pass

    # Try OpenAI
    try:
        import openai
        client = openai.OpenAI()
        def openai_embed(text: str) -> list:
            resp = client.embeddings.create(
                model="text-embedding-3-small",
                input=text[:8000]
            )
            return resp.data[0].embedding
        log.info("Using OpenAI text-embedding-3-small for embeddings")
        return openai_embed
    except Exception:
        pass

    log.warning("No embedding backend found — semantic search disabled. "
                "Install Ollama (ollama pull nomic-embed-text) or set OPENAI_API_KEY")
    return None

# ---------------------------------------------------------------------------
# Parse markdown files
# ---------------------------------------------------------------------------

def parse_markdown(path: Path) -> dict:
    """Extract metadata and content from a markdown file."""
    text = path.read_text(errors="replace")

    # Extract YAML frontmatter
    meta = {}
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) >= 3:
            try:
                import yaml
                meta = yaml.safe_load(parts[1]) or {}
                text = parts[2]
            except ImportError:
                # Manual YAML parse for simple key: value pairs
                for line in parts[1].splitlines():
                    if ":" in line:
                        k, _, v = line.partition(":")
                        meta[k.strip()] = v.strip()
                text = parts[2]

    # Extract title (first H1 or filename)
    title_match = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
    title = title_match.group(1) if title_match else path.stem.replace("-", " ").title()

    # Extract [[wikilinks]] as backlinks
    backlinks = re.findall(r"\[\[([^\]]+)\]\]", text)

    # Extract #tags
    tags = re.findall(r"(?<!\w)#([a-z][a-z0-9_-]*)", text.lower())

    # Extract source URL if present
    source_url = meta.get("source", meta.get("url", ""))
    url_match = re.search(r"^(?:Source|URL):\s*(https?://\S+)", text, re.MULTILINE | re.IGNORECASE)
    if url_match and not source_url:
        source_url = url_match.group(1)

    # Determine category from path
    rel = str(path.relative_to(ROOT))
    if rel.startswith("me/"):
        category = "personal"
    elif rel.startswith("work/"):
        category = "work"
    elif rel.startswith("knowledge/"):
        category = "knowledge"
    elif rel.startswith("media/"):
        category = "media"
    elif rel.startswith("flagged/"):
        category = "flagged"
    else:
        category = "general"

    return {
        "path":        rel,
        "title":       meta.get("title", title),
        "content":     text.strip(),
        "tags":        json.dumps(list(set(tags))),
        "backlinks":   json.dumps(list(set(backlinks))),
        "source_url":  source_url,
        "source_type": meta.get("source_type", "unknown"),
        "category":    category,
        "word_count":  len(text.split()),
        "modified_at": int(path.stat().st_mtime),
    }

# ---------------------------------------------------------------------------
# Index builder
# ---------------------------------------------------------------------------

def build_index(db: sqlite3.Connection, embedder=None) -> dict:
    """Walk brain/ and index all .md files into SQLite."""
    stats = {"indexed": 0, "updated": 0, "skipped": 0, "errors": 0}
    embed_store = {}

    # Check if sqlite-vec is available
    has_vec = False
    try:
        import sqlite_vec  # pip install sqlite-vec
        sqlite_vec.load(db)
        db.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS note_embeddings USING vec0(
                path TEXT,
                embedding FLOAT[768]
            )
        """)
        db.commit()
        has_vec = True
        log.info("sqlite-vec loaded — vector search enabled")
    except ImportError:
        log.warning("sqlite-vec not installed — skipping vector index. "
                    "Run: pip install sqlite-vec")
    except Exception as e:
        log.warning("sqlite-vec setup failed: %s", e)

    md_files = [f for f in ROOT.rglob("*.md") if "raw" not in f.parts]
    log.info("Indexing %d markdown files...", len(md_files))

    for path in md_files:
        try:
            parsed = parse_markdown(path)
            rel    = parsed["path"]

            # Check if already indexed and unchanged
            row = db.execute(
                "SELECT modified_at FROM notes WHERE path = ?", [rel]
            ).fetchone()

            if row and row["modified_at"] == parsed["modified_at"]:
                stats["skipped"] += 1
                continue

            if row:
                db.execute("""
                    UPDATE notes SET title=?, content=?, tags=?, backlinks=?,
                    source_url=?, source_type=?, category=?, word_count=?,
                    modified_at=? WHERE path=?
                """, [
                    parsed["title"], parsed["content"], parsed["tags"],
                    parsed["backlinks"], parsed["source_url"],
                    parsed["source_type"], parsed["category"],
                    parsed["word_count"], parsed["modified_at"], rel
                ])
                stats["updated"] += 1
            else:
                db.execute("""
                    INSERT INTO notes (path, title, content, tags, backlinks,
                    source_url, source_type, category, word_count, modified_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, [
                    rel, parsed["title"], parsed["content"], parsed["tags"],
                    parsed["backlinks"], parsed["source_url"],
                    parsed["source_type"], parsed["category"],
                    parsed["word_count"], parsed["modified_at"]
                ])
                stats["indexed"] += 1

            # Generate embedding
            if has_vec and embedder:
                text_for_embed = f"{parsed['title']}\n{parsed['content'][:1024]}"
                try:
                    vec = embedder(text_for_embed)
                    # sqlite-vec expects bytes for float arrays
                    import struct
                    vec_bytes = struct.pack(f"{len(vec)}f", *vec)
                    db.execute(
                        "DELETE FROM note_embeddings WHERE path = ?", [rel]
                    )
                    db.execute(
                        "INSERT INTO note_embeddings (path, embedding) VALUES (?, ?)",
                        [rel, vec_bytes]
                    )
                except Exception as e:
                    log.debug("Embedding failed for %s: %s", rel, e)

            # Extract and store concept links
            backlinks = json.loads(parsed["backlinks"])
            for link in backlinks:
                db.execute("""
                    INSERT OR IGNORE INTO concept_links (from_path, to_path, link_type)
                    VALUES (?, ?, 'wikilink')
                """, [rel, link])

        except Exception as e:
            log.error("Failed to index %s: %s", path, e)
            stats["errors"] += 1

    db.commit()
    db.execute(
        "INSERT OR REPLACE INTO index_meta (key, value) VALUES ('last_build', ?)",
        [datetime.now().isoformat()]
    )
    db.commit()
    return stats
"""


def get_connection(db_path: Path = DB_PATH) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db(db_path: Path = DB_PATH):
    conn = get_connection(db_path)
    conn.executescript(SCHEMA_SQL)
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Embedding engine
# ---------------------------------------------------------------------------

class EmbeddingEngine:
    """Generates embeddings using OpenAI or falls back to TF-IDF."""

    MODEL = "text-embedding-3-small"
    DIM_OPENAI = 1536

    def __init__(self):
        self.openai_client = None
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            try:
                import openai
                self.openai_client = openai.OpenAI(api_key=api_key)
            except ImportError:
                pass
        self._tfidf_vocab: dict[str, int] = {}

    def embed(self, text: str) -> np.ndarray:
        if self.openai_client:
            return self._embed_openai(text)
        return self._embed_tfidf(text)

    def _embed_openai(self, text: str) -> np.ndarray:
        text = text[:8192]
        try:
            response = self.openai_client.embeddings.create(
                model=self.MODEL,
                input=text,
            )
            vec = np.array(response.data[0].embedding, dtype=np.float32)
            return vec / (np.linalg.norm(vec) + 1e-10)
        except Exception as e:
            print(f"[EmbeddingEngine] OpenAI error: {e}, falling back to TF-IDF", file=sys.stderr)
            return self._embed_tfidf(text)

    def _tokenize(self, text: str) -> list[str]:
        import re
        return re.findall(r"\b[a-z]{2,}\b", text.lower())

    def _embed_tfidf(self, text: str) -> np.ndarray:
        tokens = self._tokenize(text)
        if not tokens:
            return np.zeros(64, dtype=np.float32)
        # Build/use vocabulary of top 512 terms
        if not self._tfidf_vocab:
            # default: hash-based projection
            dim = 512
            vec = np.zeros(dim, dtype=np.float32)
            for t in tokens:
                idx = hash(t) % dim
                vec[idx] += 1.0
            norm = np.linalg.norm(vec)
            return vec / (norm + 1e-10)
        dim = len(self._tfidf_vocab)
        vec = np.zeros(dim, dtype=np.float32)
        for t in tokens:
            if t in self._tfidf_vocab:
                vec[self._tfidf_vocab[t]] += 1.0
        norm = np.linalg.norm(vec)
        return vec / (norm + 1e-10)

    @property
    def model_name(self) -> str:
        return self.MODEL if self.openai_client else "tfidf-512"


_engine: Optional[EmbeddingEngine] = None


def get_engine() -> EmbeddingEngine:
    global _engine
    if _engine is None:
        _engine = EmbeddingEngine()
    return _engine


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------

def index_file(path: str | Path, content: str, title: str = "", route: str = "",
               tags: str = "", db_path: Path = DB_PATH):
    """Index a file's content into the database."""
    path = str(path)
    init_db(db_path)
    conn = get_connection(db_path)

    # Upsert note
    conn.execute("""
        INSERT INTO notes (path, title, content, route, tags, updated_at)
        VALUES (?, ?, ?, ?, ?, datetime('now'))
        ON CONFLICT(path) DO UPDATE SET
            title=excluded.title,
            content=excluded.content,
            route=excluded.route,
            tags=excluded.tags,
            updated_at=excluded.updated_at
    """, (path, title or Path(path).stem, content, route, tags))
    conn.commit()

    note_id = conn.execute("SELECT id FROM notes WHERE path=?", (path,)).fetchone()["id"]

    # Generate and store embedding
    engine = get_engine()
    vec = engine.embed(content[:4096])
    vec_json = json.dumps(vec.tolist())

    conn.execute("""
        INSERT INTO embeddings (note_id, model, vector_json, dim)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(note_id) DO UPDATE SET
            model=excluded.model,
            vector_json=excluded.vector_json,
            dim=excluded.dim,
            created_at=datetime('now')
    """, (note_id, engine.model_name, vec_json, len(vec)))
    conn.commit()
    conn.close()
    return note_id


def index_directory(directory: str | Path, db_path: Path = DB_PATH):
    """Recursively index all markdown files in a directory."""
    directory = Path(directory)
    indexed = 0
    for md_file in directory.rglob("*.md"):
        if ".gitkeep" in md_file.name:
            continue
        try:
            content = md_file.read_text(encoding="utf-8", errors="replace")
            # Try to parse frontmatter title
            title = md_file.stem
            route = ""
            tags = ""
            try:
                import frontmatter
                post = frontmatter.loads(content)
                title = post.get("title", md_file.stem)
                route = post.get("route", "")
                tag_list = post.get("tags", [])
                tags = ", ".join(tag_list) if isinstance(tag_list, list) else str(tag_list)
                content = post.content
            except Exception:
                pass
            index_file(md_file, content, title=str(title), route=str(route),
                       tags=str(tags), db_path=db_path)
            indexed += 1
            print(f"  Indexed: {md_file.relative_to(directory)}")
        except Exception as e:
            print(f"  Error indexing {md_file}: {e}", file=sys.stderr)
    print(f"\nIndexed {indexed} files into {db_path}")


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def keyword_search(db: sqlite3.Connection, query: str, limit: int = 20) -> list:
    """FTS5 keyword search."""
    try:
        rows = db.execute("""
            SELECT n.path, n.title, n.category,
                   snippet(notes_fts, 1, '[', ']', '...', 32) AS snippet,
                   rank AS score
            FROM notes_fts
            JOIN notes n ON notes_fts.rowid = n.id
            WHERE notes_fts MATCH ?
            ORDER BY rank
            LIMIT ?
        """, [query, limit]).fetchall()
        return [dict(r) for r in rows]
    except Exception as e:
        log.debug("Keyword search error: %s", e)
        return []


def semantic_search(db: sqlite3.Connection, query: str,
                    embedder, limit: int = 20) -> list:
    """Vector similarity search via sqlite-vec."""
    try:
        import sqlite_vec, struct
        vec = embedder(query)
        vec_bytes = struct.pack(f"{len(vec)}f", *vec)
        rows = db.execute("""
            SELECT e.path, n.title, n.category, n.content,
                   e.distance AS score
            FROM note_embeddings e
            JOIN notes n ON e.path = n.path
            WHERE e.embedding MATCH ?
            ORDER BY e.distance
            LIMIT ?
        """, [vec_bytes, limit]).fetchall()
        return [dict(r) for r in rows]
    except Exception as e:
        log.debug("Semantic search error: %s", e)
        return []


def hybrid_search(db: sqlite3.Connection, query: str,
                  embedder=None, top_k: int = 10) -> list:
    """Merge keyword + semantic results, rerank by combined score."""
    kw_results = keyword_search(db, query, limit=20)
    sem_results = semantic_search(db, query, embedder, limit=20) if embedder else []

    # Normalize and merge
    scored = {}
    for i, r in enumerate(kw_results):
        path = r["path"]
        scored[path] = scored.get(path, {**r, "score": 0.0})
        scored[path]["score"] += (1.0 - i / len(kw_results)) * 0.5  # keyword weight

    for i, r in enumerate(sem_results):
        path = r["path"]
        if path not in scored:
            scored[path] = {**r, "score": 0.0}
        # Lower distance = better; normalize to 0-1
        sem_score = max(0.0, 1.0 - r.get("score", 1.0))
        scored[path]["score"] += sem_score * 0.5  # semantic weight

    results = sorted(scored.values(), key=lambda x: x["score"], reverse=True)
    return results[:top_k]


def get_related(db: sqlite3.Connection, path: str, limit: int = 10) -> list:
    """Find conceptually related notes via the link graph."""
    rows = db.execute("""
        SELECT cl.to_path, cl.link_type, cl.strength,
               n.title, n.category
        FROM concept_links cl
        JOIN notes n ON cl.to_path = n.path
        WHERE cl.from_path = ?
        ORDER BY cl.strength DESC
        LIMIT ?
    """, [path, limit]).fetchall()
    return [dict(r) for r in rows]

# ---------------------------------------------------------------------------
# Query interface
# ---------------------------------------------------------------------------

def run_query(query: str, db: sqlite3.Connection, embedder=None):
    """Run a hybrid search and print results with context."""
    print(f"\nSearching: '{query}'\n{'='*60}")
    results = hybrid_search(db, query, embedder, top_k=10)

    if not results:
        print("No results found.")
        return

    for i, r in enumerate(results, 1):
        print(f"\n[{i}] {r.get('title', r['path'])}")
        print(f"    Path:     brain/{r['path']}")
        print(f"    Category: {r.get('category', '?')}")
        print(f"    Score:    {r.get('score', 0):.3f}")
        snippet = r.get("snippet") or r.get("content", "")[:200]
        if snippet:
            print(f"    Preview:  {snippet[:200]}...")

    # Find related concepts for top result
    top_path = results[0]["path"]
    related = get_related(db, top_path, limit=5)
    if related:
        print(f"\nRelated concepts from [{results[0].get('title', top_path)}]:")
        for r in related:
            print(f"  -> [[{r['to_path']}]] ({r['link_type']})")


def print_stats(db: sqlite3.Connection):
    """Print database statistics."""
    total     = db.execute("SELECT COUNT(*) FROM notes").fetchone()[0]
    by_cat    = db.execute(
        "SELECT category, COUNT(*) n, SUM(word_count) w FROM notes GROUP BY category"
    ).fetchall()
    links     = db.execute("SELECT COUNT(*) FROM concept_links").fetchone()[0]
    last_build = db.execute(
        "SELECT value FROM index_meta WHERE key='last_build'"
    ).fetchone()

    print(f"\n{'='*50}")
    print(f"  Brain Database Statistics")
    print(f"{'='*50}")
    print(f"  Total notes     : {total}")
    print(f"  Concept links   : {links}")
    print(f"  Last indexed    : {last_build[0] if last_build else 'never'}")
    print(f"\n  By Category:")
    for row in by_cat:
        print(f"    {row[0]:<16} {row[1]:>5} notes   {row[2]:>8,} words")
    print(f"{'='*50}")

# ---------------------------------------------------------------------------
# MCP server config export
# ---------------------------------------------------------------------------

def export_mcp_config(db_path: Path = DB_PATH):
    """Export an MCP server config so Claude Desktop can query the brain."""
    config = {
        "mcpServers": {
            "llm-brain": {
                "command": "python",
                "args": [str(Path(__file__).resolve()), "--mcp-server"],
                "env": {
                    "BRAIN_DB": str(db_path.resolve())
                }
            }
        }
    }
    config_path = Path(__file__).parent / "mcp_config.json"
    config_path.write_text(json.dumps(config, indent=2))
    print(f"MCP config written to: {config_path}")
    print("\nAdd this to your Claude Desktop config to query your brain directly:")
    print(json.dumps(config, indent=2))
def fts_search(query: str, limit: int = 20, db_path: Path = DB_PATH) -> list[dict]:
    """Full-text search using FTS5."""
    init_db(db_path)
    conn = get_connection(db_path)
    try:
        rows = conn.execute("""
            SELECT n.id, n.path, n.title, n.content, n.route, n.tags,
                   bm25(notes_fts) AS score
            FROM notes_fts
            JOIN notes n ON notes_fts.rowid = n.id
            WHERE notes_fts MATCH ?
            ORDER BY score
            LIMIT ?
        """, (query, limit)).fetchall()
        return [dict(r) for r in rows]
    except Exception as e:
        print(f"FTS error: {e}", file=sys.stderr)
        return []
    finally:
        conn.close()


def vector_search(query: str, limit: int = 20, db_path: Path = DB_PATH) -> list[dict]:
    """Semantic search using cosine similarity over stored embeddings."""
    init_db(db_path)
    engine = get_engine()
    query_vec = engine.embed(query)

    conn = get_connection(db_path)
    rows = conn.execute("""
        SELECT n.id, n.path, n.title, n.content, n.route, n.tags,
               e.vector_json
        FROM embeddings e
        JOIN notes n ON e.note_id = n.id
    """).fetchall()
    conn.close()

    scored = []
    for row in rows:
        try:
            doc_vec = np.array(json.loads(row["vector_json"]), dtype=np.float32)
            if doc_vec.shape != query_vec.shape:
                # Resize if shapes differ (tfidf vs openai mismatch)
                min_dim = min(len(doc_vec), len(query_vec))
                doc_vec = doc_vec[:min_dim]
                qv = query_vec[:min_dim]
            else:
                qv = query_vec
            sim = float(np.dot(doc_vec, qv) / (np.linalg.norm(doc_vec) * np.linalg.norm(qv) + 1e-10))
            scored.append({**dict(row), "score": sim})
        except Exception:
            continue

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:limit]


def hybrid_search(query: str, limit: int = 10, db_path: Path = DB_PATH,
                  fts_weight: float = 0.4, vec_weight: float = 0.6) -> list[dict]:
    """
    Hybrid search combining FTS5 keyword search + vector similarity.
    Normalizes scores and merges with weighted combination (RRF-style).
    """
    fts_results = fts_search(query, limit=limit * 2, db_path=db_path)
    vec_results = vector_search(query, limit=limit * 2, db_path=db_path)

    # Build score maps
    fts_scores: dict[int, float] = {}
    for rank, r in enumerate(fts_results):
        # BM25 from FTS5 is negative; higher = better match when negated
        fts_scores[r["id"]] = 1.0 / (rank + 1)  # Reciprocal rank

    vec_scores: dict[int, float] = {}
    for rank, r in enumerate(vec_results):
        vec_scores[r["id"]] = 1.0 / (rank + 1)

    # Combine all unique doc IDs
    all_ids = set(fts_scores.keys()) | set(vec_scores.keys())

    combined: list[dict] = []
    # Build an id→row lookup
    id_to_row: dict[int, dict] = {}
    for r in fts_results + vec_results:
        id_to_row[r["id"]] = r

    for doc_id in all_ids:
        fts_s = fts_scores.get(doc_id, 0.0)
        vec_s = vec_scores.get(doc_id, 0.0)
        hybrid_score = fts_weight * fts_s + vec_weight * vec_s
        row = id_to_row[doc_id].copy()
        row["hybrid_score"] = hybrid_score
        row["fts_score"] = fts_s
        row["vec_score"] = vec_s
        combined.append(row)

    combined.sort(key=lambda x: x["hybrid_score"], reverse=True)
    return combined[:limit]


def search(query: str, limit: int = 10, db_path: Path = DB_PATH) -> list[dict]:
    """Top-level search function: tries hybrid, falls back gracefully."""
    return hybrid_search(query, limit=limit, db_path=db_path)


def format_results(results: list[dict], query: str) -> str:
    if not results:
        return f"No results found for: {query}"
    lines = [f"Search results for: '{query}'\n{'='*50}"]
    for i, r in enumerate(results, 1):
        score = r.get("hybrid_score", r.get("score", 0))
        title = r.get("title", Path(r.get("path", "?")).stem)
        path = r.get("path", "?")
        route = r.get("route", "")
        snippet = r.get("content", "")[:200].replace("\n", " ")
        lines.append(
            f"\n{i}. [{title}] ({route})\n"
            f"   Score: {score:.4f}\n"
            f"   Path: {path}\n"
            f"   {snippet}..."
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="SQLite RAG layer for the LLM Brain"
    )
    parser.add_argument("--build", action="store_true",
                        help="Index all brain/ files into SQLite")
    parser.add_argument("--query", "-q", type=str,
                        help="Search your brain")
    parser.add_argument("--stats", action="store_true",
                        help="Show database statistics")
    parser.add_argument("--export-mcp", action="store_true",
                        help="Export MCP server config for Claude Desktop")
    parser.add_argument("--db", type=str, default=str(DB_PATH),
                        help=f"Database path (default: {DB_PATH})")
    args = parser.parse_args()

    db_path = Path(args.db)
    db = get_db(db_path)

    if args.build:
        embedder = get_embedder()
        log.info("Building index at %s...", db_path)
        stats = build_index(db, embedder)
        print(f"\n{'='*50}")
        print(f"  Index Build Summary")
        print(f"{'='*50}")
        print(f"  New files indexed : {stats['indexed']}")
        print(f"  Files updated     : {stats['updated']}")
        print(f"  Unchanged (skip)  : {stats['skipped']}")
        print(f"  Errors            : {stats['errors']}")
        print(f"{'='*50}")
        print(f"\n  Database: {db_path}")

    elif args.query:
        embedder = get_embedder()
        run_query(args.query, db, embedder)

    elif args.stats:
        print_stats(db)

    elif args.export_mcp:
        export_mcp_config(db_path)

    else:
    parser = argparse.ArgumentParser(description="LLM-Brains hybrid search")
    parser.add_argument("--search", "-s", type=str, help="Search query")
    parser.add_argument("--index", "-i", type=str, help="Directory to index")
    parser.add_argument("--limit", "-l", type=int, default=10, help="Max results")
    parser.add_argument("--db", type=str, default=str(DB_PATH), help="Database path")
    parser.add_argument("--fts-only", action="store_true", help="Use FTS only")
    parser.add_argument("--vec-only", action="store_true", help="Use vector only")
    args = parser.parse_args()

    db_path = Path(args.db)

    if args.index:
        print(f"Indexing {args.index} into {db_path}...")
        index_directory(args.index, db_path=db_path)

    if args.search:
        if args.fts_only:
            results = fts_search(args.search, limit=args.limit, db_path=db_path)
        elif args.vec_only:
            results = vector_search(args.search, limit=args.limit, db_path=db_path)
        else:
            results = hybrid_search(args.search, limit=args.limit, db_path=db_path)
        print(format_results(results, args.search))

    if not args.index and not args.search:
        parser.print_help()


if __name__ == "__main__":
    main()
