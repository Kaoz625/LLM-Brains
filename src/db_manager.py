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
src/db_manager.py — DatabaseManager class for all SQLite operations.

Handles schema creation, migrations, and all CRUD operations for brain/memory.db.
"""

import json
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

SCHEMA_VERSION = 3

MIGRATIONS = {
    1: """
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
            title, content, tags,
            content='notes', content_rowid='id'
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
    """,
    2: """
        CREATE TABLE IF NOT EXISTS embeddings (
            note_id     INTEGER PRIMARY KEY REFERENCES notes(id) ON DELETE CASCADE,
            model       TEXT NOT NULL,
            vector_json TEXT NOT NULL,
            dim         INTEGER NOT NULL,
            created_at  TEXT DEFAULT (datetime('now'))
        );
    """,
    3: """
        CREATE TABLE IF NOT EXISTS metadata (
            key   TEXT PRIMARY KEY,
            value TEXT
        );
        CREATE TABLE IF NOT EXISTS tags (
            id      INTEGER PRIMARY KEY AUTOINCREMENT,
            name    TEXT UNIQUE NOT NULL
        );
        CREATE TABLE IF NOT EXISTS note_tags (
            note_id INTEGER REFERENCES notes(id) ON DELETE CASCADE,
            tag_id  INTEGER REFERENCES tags(id) ON DELETE CASCADE,
            PRIMARY KEY (note_id, tag_id)
        );
    """,
}


class DatabaseManager:
    """Manages the brain SQLite database with migrations and CRUD operations."""

    def __init__(self, db_path: Optional[Path] = None):
        if db_path is None:
            brain_dir = Path(os.getenv("BRAIN_DIR", "./brain"))
            db_path = brain_dir / "memory.db"
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def connect(self) -> sqlite3.Connection:
        """Get a database connection with row factory set."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA cache_size=-64000")  # 64MB cache
        return conn

    def _get_schema_version(self, conn: sqlite3.Connection) -> int:
        try:
            row = conn.execute(
                "SELECT value FROM metadata WHERE key='schema_version'"
            ).fetchone()
            return int(row["value"]) if row else 0
        except Exception:
            return 0

    def _set_schema_version(self, conn: sqlite3.Connection, version: int):
        conn.execute("""
            INSERT INTO metadata (key, value) VALUES ('schema_version', ?)
            ON CONFLICT(key) DO UPDATE SET value=excluded.value
        """, (str(version),))

    def _init_db(self):
        """Initialize database and run any pending migrations."""
        conn = self.connect()

        # Ensure metadata table exists for version tracking
        conn.execute("""
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY, value TEXT
            )
        """)
        conn.commit()

        current_version = self._get_schema_version(conn)

        for version in sorted(MIGRATIONS.keys()):
            if version > current_version:
                conn.executescript(MIGRATIONS[version])
                self._set_schema_version(conn, version)
                conn.commit()

        conn.close()

    # ------------------------------------------------------------------
    # Notes CRUD
    # ------------------------------------------------------------------

    def upsert_note(self, path: str, content: str, title: str = "",
                    route: str = "", tags: str = "") -> int:
        """Insert or update a note. Returns note ID."""
        conn = self.connect()
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
        conn.close()
        return note_id

    def get_note(self, note_id: int) -> Optional[dict]:
        conn = self.connect()
        row = conn.execute("SELECT * FROM notes WHERE id=?", (note_id,)).fetchone()
        conn.close()
        return dict(row) if row else None

    def get_note_by_path(self, path: str) -> Optional[dict]:
        conn = self.connect()
        row = conn.execute("SELECT * FROM notes WHERE path=?", (path,)).fetchone()
        conn.close()
        return dict(row) if row else None

    def delete_note(self, note_id: int):
        conn = self.connect()
        conn.execute("DELETE FROM notes WHERE id=?", (note_id,))
        conn.commit()
        conn.close()

    def list_notes(self, route: Optional[str] = None,
                   limit: int = 100, offset: int = 0) -> list[dict]:
        conn = self.connect()
        if route:
            rows = conn.execute(
                "SELECT * FROM notes WHERE route=? ORDER BY updated_at DESC LIMIT ? OFFSET ?",
                (route, limit, offset)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM notes ORDER BY updated_at DESC LIMIT ? OFFSET ?",
                (limit, offset)
            ).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def count_notes(self) -> int:
        conn = self.connect()
        count = conn.execute("SELECT COUNT(*) FROM notes").fetchone()[0]
        conn.close()
        return count

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------

    def upsert_embedding(self, note_id: int, model: str,
                          vector: list[float]) -> None:
        """Store or update an embedding vector for a note."""
        conn = self.connect()
        conn.execute("""
            INSERT INTO embeddings (note_id, model, vector_json, dim, created_at)
            VALUES (?, ?, ?, ?, datetime('now'))
            ON CONFLICT(note_id) DO UPDATE SET
                model=excluded.model,
                vector_json=excluded.vector_json,
                dim=excluded.dim,
                created_at=excluded.created_at
        """, (note_id, model, json.dumps(vector), len(vector)))
        conn.commit()
        conn.close()

    def get_embedding(self, note_id: int) -> Optional[dict]:
        conn = self.connect()
        row = conn.execute(
            "SELECT * FROM embeddings WHERE note_id=?", (note_id,)
        ).fetchone()
        conn.close()
        return dict(row) if row else None

    def get_all_embeddings(self) -> list[dict]:
        conn = self.connect()
        rows = conn.execute("""
            SELECT e.note_id, e.model, e.vector_json, e.dim,
                   n.path, n.title, n.content, n.route
            FROM embeddings e
            JOIN notes n ON e.note_id = n.id
        """).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # FTS Search
    # ------------------------------------------------------------------

    def fts_search(self, query: str, limit: int = 20) -> list[dict]:
        """Full-text search using FTS5. Returns rows with BM25 score."""
        conn = self.connect()
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
            conn.close()
            return [dict(r) for r in rows]
        except Exception as e:
            conn.close()
            return []

    # ------------------------------------------------------------------
    # Tags
    # ------------------------------------------------------------------

    def add_tags_to_note(self, note_id: int, tags: list[str]):
        conn = self.connect()
        for tag in tags:
            tag = tag.strip().lower()
            if not tag:
                continue
            conn.execute("INSERT OR IGNORE INTO tags (name) VALUES (?)", (tag,))
            tag_id = conn.execute("SELECT id FROM tags WHERE name=?", (tag,)).fetchone()["id"]
            conn.execute(
                "INSERT OR IGNORE INTO note_tags (note_id, tag_id) VALUES (?, ?)",
                (note_id, tag_id)
            )
        conn.commit()
        conn.close()

    def get_tags(self, note_id: int) -> list[str]:
        conn = self.connect()
        rows = conn.execute("""
            SELECT t.name FROM tags t
            JOIN note_tags nt ON t.id = nt.tag_id
            WHERE nt.note_id = ?
        """, (note_id,)).fetchall()
        conn.close()
        return [r["name"] for r in rows]

    def list_all_tags(self) -> list[dict]:
        conn = self.connect()
        rows = conn.execute("""
            SELECT t.name, COUNT(nt.note_id) AS count
            FROM tags t
            LEFT JOIN note_tags nt ON t.id = nt.tag_id
            GROUP BY t.name ORDER BY count DESC
        """).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def set_meta(self, key: str, value: Any):
        conn = self.connect()
        conn.execute("""
            INSERT INTO metadata (key, value) VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE SET value=excluded.value
        """, (key, json.dumps(value) if not isinstance(value, str) else value))
        conn.commit()
        conn.close()

    def get_meta(self, key: str, default: Any = None) -> Any:
        conn = self.connect()
        row = conn.execute("SELECT value FROM metadata WHERE key=?", (key,)).fetchone()
        conn.close()
        if row is None:
            return default
        try:
            return json.loads(row["value"])
        except (json.JSONDecodeError, TypeError):
            return row["value"]

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
    def get_stats(self) -> dict:
        conn = self.connect()
        note_count = conn.execute("SELECT COUNT(*) FROM notes").fetchone()[0]
        emb_count = conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
        tag_count = conn.execute("SELECT COUNT(*) FROM tags").fetchone()[0]
        route_counts = dict(conn.execute(
            "SELECT route, COUNT(*) FROM notes GROUP BY route"
        ).fetchall())
        conn.close()
        return {
            "notes": note_count,
            "embeddings": emb_count,
            "tags": tag_count,
            "routes": route_counts,
            "schema_version": SCHEMA_VERSION,
            "db_path": str(self.db_path),
        }

    def __repr__(self) -> str:
        stats = self.get_stats()
        return f"DatabaseManager({self.db_path}, {stats['notes']} notes)"
