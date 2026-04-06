"""
db_manager.py
-------------
Manages the SQLite database that backs the LLM-Brains knowledge base.

Schema
------
notes           – Full-text content, tags, backlinks, metadata.
notes_fts       – FTS5 virtual table mirroring notes for keyword search.
note_embeddings – sqlite-vec virtual table for semantic (vector) search.
wiki_articles   – LLM-compiled concept articles keyed by slug.
wiki_fts        – FTS5 virtual table mirroring wiki_articles.

The sqlite-vec extension is loaded at connection time when available.
If the extension is absent the database still works for FTS5 searches;
vector operations raise a clear RuntimeError.
"""

from __future__ import annotations

import json
import sqlite3
import struct
from pathlib import Path
from typing import List, Optional, Tuple

# Try to import sqlite_vec; it is optional at import time.
try:
    import sqlite_vec  # type: ignore

    _SQLITE_VEC_AVAILABLE = True
except ImportError:
    _SQLITE_VEC_AVAILABLE = False


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

EMBEDDING_DIM = 768  # nomic-embed-text / text-embedding-3-small truncated to 768


def _serialize_vector(vec: List[float]) -> bytes:
    """Pack a list of floats into a little-endian bytes blob for sqlite-vec."""
    return struct.pack(f"{len(vec)}f", *vec)


def _deserialize_vector(blob: bytes) -> List[float]:
    n = len(blob) // 4
    return list(struct.unpack(f"{n}f", blob))


# ---------------------------------------------------------------------------
# Database manager
# ---------------------------------------------------------------------------

class DBManager:
    """
    Context-manager-friendly wrapper around the SQLite knowledge-base.

    Usage::

        with DBManager("vault_memory.db") as db:
            db.upsert_note(note, embedding=[...])
            results = db.fts_search("transformer attention")
    """

    def __init__(self, db_path: str, embedding_dim: int = EMBEDDING_DIM):
        self.db_path = str(Path(db_path).expanduser().resolve())
        self.embedding_dim = embedding_dim
        self._conn: Optional[sqlite3.Connection] = None

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "DBManager":
        self.open()
        return self

    def __exit__(self, *_) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def open(self) -> None:
        """Open (or create) the database and load extensions."""
        self._conn = sqlite3.connect(self.db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")

        if _SQLITE_VEC_AVAILABLE:
            self._conn.enable_load_extension(True)
            sqlite_vec.load(self._conn)
            self._conn.enable_load_extension(False)

        self._create_schema()

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError("Database is not open. Call open() or use as a context manager.")
        return self._conn

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _create_schema(self) -> None:
        c = self.conn

        # Core notes table
        c.executescript("""
            CREATE TABLE IF NOT EXISTS notes (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                path        TEXT    UNIQUE NOT NULL,
                title       TEXT    NOT NULL,
                content     TEXT    NOT NULL DEFAULT '',
                tags        TEXT    NOT NULL DEFAULT '[]',
                backlinks   TEXT    NOT NULL DEFAULT '[]',
                modified_at INTEGER NOT NULL DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS wiki_articles (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                slug        TEXT    UNIQUE NOT NULL,
                title       TEXT    NOT NULL,
                content     TEXT    NOT NULL DEFAULT '',
                source_paths TEXT   NOT NULL DEFAULT '[]',
                created_at  INTEGER NOT NULL DEFAULT 0,
                updated_at  INTEGER NOT NULL DEFAULT 0
            );
        """)

        # FTS5 tables – drop and recreate only when columns change is not
        # needed here; CREATE IF NOT EXISTS is idempotent.
        c.executescript("""
            CREATE VIRTUAL TABLE IF NOT EXISTS notes_fts
                USING fts5(
                    title,
                    content,
                    tags,
                    content='notes',
                    content_rowid='id'
                );

            CREATE VIRTUAL TABLE IF NOT EXISTS wiki_fts
                USING fts5(
                    title,
                    content,
                    content='wiki_articles',
                    content_rowid='id'
                );
        """)

        # FTS5 triggers to keep the index in sync
        c.executescript("""
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

            CREATE TRIGGER IF NOT EXISTS wiki_ai AFTER INSERT ON wiki_articles BEGIN
                INSERT INTO wiki_fts(rowid, title, content)
                VALUES (new.id, new.title, new.content);
            END;

            CREATE TRIGGER IF NOT EXISTS wiki_ad AFTER DELETE ON wiki_articles BEGIN
                INSERT INTO wiki_fts(wiki_fts, rowid, title, content)
                VALUES ('delete', old.id, old.title, old.content);
            END;

            CREATE TRIGGER IF NOT EXISTS wiki_au AFTER UPDATE ON wiki_articles BEGIN
                INSERT INTO wiki_fts(wiki_fts, rowid, title, content)
                VALUES ('delete', old.id, old.title, old.content);
                INSERT INTO wiki_fts(rowid, title, content)
                VALUES (new.id, new.title, new.content);
            END;
        """)

        # Vector table (requires sqlite-vec)
        if _SQLITE_VEC_AVAILABLE:
            c.execute(f"""
                CREATE VIRTUAL TABLE IF NOT EXISTS note_embeddings
                USING vec0(
                    note_id INTEGER PRIMARY KEY,
                    embedding FLOAT[{self.embedding_dim}]
                )
            """)
            c.execute(f"""
                CREATE VIRTUAL TABLE IF NOT EXISTS wiki_embeddings
                USING vec0(
                    article_id INTEGER PRIMARY KEY,
                    embedding FLOAT[{self.embedding_dim}]
                )
            """)

        c.commit()

    # ------------------------------------------------------------------
    # Notes CRUD
    # ------------------------------------------------------------------

    def upsert_note(
        self,
        path: str,
        title: str,
        content: str,
        tags: List[str],
        backlinks: List[str],
        modified_at: int,
        embedding: Optional[List[float]] = None,
    ) -> int:
        """Insert or replace a note. Returns the row id."""
        tags_json = json.dumps(tags)
        backlinks_json = json.dumps(backlinks)

        cur = self.conn.execute(
            """
            INSERT INTO notes (path, title, content, tags, backlinks, modified_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(path) DO UPDATE SET
                title       = excluded.title,
                content     = excluded.content,
                tags        = excluded.tags,
                backlinks   = excluded.backlinks,
                modified_at = excluded.modified_at
            RETURNING id
            """,
            (path, title, content, tags_json, backlinks_json, modified_at),
        )
        row = cur.fetchone()
        note_id: int = row["id"]

        if embedding is not None:
            self._upsert_note_embedding(note_id, embedding)

        self.conn.commit()
        return note_id

    def _upsert_note_embedding(self, note_id: int, embedding: List[float]) -> None:
        if not _SQLITE_VEC_AVAILABLE:
            raise RuntimeError(
                "sqlite-vec extension is not installed. "
                "Run: pip install sqlite-vec"
            )
        blob = _serialize_vector(embedding)
        self.conn.execute(
            """
            INSERT INTO note_embeddings (note_id, embedding)
            VALUES (?, ?)
            ON CONFLICT(note_id) DO UPDATE SET embedding = excluded.embedding
            """,
            (note_id, blob),
        )

    def get_note_by_path(self, path: str) -> Optional[sqlite3.Row]:
        cur = self.conn.execute("SELECT * FROM notes WHERE path = ?", (path,))
        return cur.fetchone()

    def get_all_notes(self) -> List[sqlite3.Row]:
        cur = self.conn.execute("SELECT * FROM notes ORDER BY modified_at DESC")
        return cur.fetchall()

    # ------------------------------------------------------------------
    # Wiki articles CRUD
    # ------------------------------------------------------------------

    def upsert_wiki_article(
        self,
        slug: str,
        title: str,
        content: str,
        source_paths: List[str],
        embedding: Optional[List[float]] = None,
    ) -> int:
        import time as _time

        now = int(_time.time())
        source_json = json.dumps(source_paths)

        cur = self.conn.execute(
            """
            INSERT INTO wiki_articles (slug, title, content, source_paths, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(slug) DO UPDATE SET
                title        = excluded.title,
                content      = excluded.content,
                source_paths = excluded.source_paths,
                updated_at   = excluded.updated_at
            RETURNING id
            """,
            (slug, title, content, source_json, now, now),
        )
        row = cur.fetchone()
        article_id: int = row["id"]

        if embedding is not None and _SQLITE_VEC_AVAILABLE:
            blob = _serialize_vector(embedding)
            self.conn.execute(
                """
                INSERT INTO wiki_embeddings (article_id, embedding)
                VALUES (?, ?)
                ON CONFLICT(article_id) DO UPDATE SET embedding = excluded.embedding
                """,
                (article_id, blob),
            )

        self.conn.commit()
        return article_id

    def get_wiki_article(self, slug: str) -> Optional[sqlite3.Row]:
        cur = self.conn.execute("SELECT * FROM wiki_articles WHERE slug = ?", (slug,))
        return cur.fetchone()

    def get_all_wiki_articles(self) -> List[sqlite3.Row]:
        cur = self.conn.execute("SELECT * FROM wiki_articles ORDER BY updated_at DESC")
        return cur.fetchall()

    # ------------------------------------------------------------------
    # Search helpers (used by search.py)
    # ------------------------------------------------------------------

    def fts_search_notes(self, query: str, limit: int = 20) -> List[sqlite3.Row]:
        """FTS5 full-text search over notes."""
        cur = self.conn.execute(
            """
            SELECT n.*, bm25(notes_fts) AS score
            FROM notes_fts
            JOIN notes n ON notes_fts.rowid = n.id
            WHERE notes_fts MATCH ?
            ORDER BY score
            LIMIT ?
            """,
            (query, limit),
        )
        return cur.fetchall()

    def fts_search_wiki(self, query: str, limit: int = 10) -> List[sqlite3.Row]:
        """FTS5 full-text search over wiki articles."""
        cur = self.conn.execute(
            """
            SELECT w.*, bm25(wiki_fts) AS score
            FROM wiki_fts
            JOIN wiki_articles w ON wiki_fts.rowid = w.id
            WHERE wiki_fts MATCH ?
            ORDER BY score
            LIMIT ?
            """,
            (query, limit),
        )
        return cur.fetchall()

    def vector_search_notes(
        self, query_embedding: List[float], limit: int = 20
    ) -> List[Tuple[int, float]]:
        """
        Approximate nearest-neighbour search over note_embeddings.
        Returns list of (note_id, distance) tuples sorted by distance ascending.
        """
        if not _SQLITE_VEC_AVAILABLE:
            raise RuntimeError("sqlite-vec is required for vector search.")
        blob = _serialize_vector(query_embedding)
        cur = self.conn.execute(
            """
            SELECT note_id, distance
            FROM note_embeddings
            WHERE embedding MATCH ?
              AND k = ?
            ORDER BY distance
            """,
            (blob, limit),
        )
        return [(row[0], row[1]) for row in cur.fetchall()]

    def vector_search_wiki(
        self, query_embedding: List[float], limit: int = 10
    ) -> List[Tuple[int, float]]:
        """ANN search over wiki_embeddings."""
        if not _SQLITE_VEC_AVAILABLE:
            raise RuntimeError("sqlite-vec is required for vector search.")
        blob = _serialize_vector(query_embedding)
        cur = self.conn.execute(
            """
            SELECT article_id, distance
            FROM wiki_embeddings
            WHERE embedding MATCH ?
              AND k = ?
            ORDER BY distance
            """,
            (blob, limit),
        )
        return [(row[0], row[1]) for row in cur.fetchall()]

    def get_notes_by_ids(self, ids: List[int]) -> List[sqlite3.Row]:
        if not ids:
            return []
        placeholders = ",".join("?" * len(ids))
        cur = self.conn.execute(
            f"SELECT * FROM notes WHERE id IN ({placeholders})", ids
        )
        return cur.fetchall()

    def get_wiki_by_ids(self, ids: List[int]) -> List[sqlite3.Row]:
        if not ids:
            return []
        placeholders = ",".join("?" * len(ids))
        cur = self.conn.execute(
            f"SELECT * FROM wiki_articles WHERE id IN ({placeholders})", ids
        )
        return cur.fetchall()

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        note_count = self.conn.execute("SELECT COUNT(*) FROM notes").fetchone()[0]
        wiki_count = self.conn.execute("SELECT COUNT(*) FROM wiki_articles").fetchone()[0]
        has_vec = _SQLITE_VEC_AVAILABLE
        emb_count = 0
        if has_vec:
            try:
                emb_count = self.conn.execute(
                    "SELECT COUNT(*) FROM note_embeddings"
                ).fetchone()[0]
            except sqlite3.OperationalError:
                emb_count = 0
        return {
            "notes": note_count,
            "wiki_articles": wiki_count,
            "embeddings": emb_count,
            "sqlite_vec": has_vec,
            "db_path": self.db_path,
        }
