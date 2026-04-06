#!/usr/bin/env python3
"""
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

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS notes (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    path        TEXT UNIQUE NOT NULL,
    title       TEXT,
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
