"""
pipeline.py
-----------
Orchestrates the full LLM-Brains ingestion pipeline:

  1. Parse the Obsidian vault  (vault_parser)
  2. Generate embeddings       (embeddings)
  3. Upsert notes into SQLite  (db_manager)
  4. Optionally compile wiki articles (compiler)

Run via ``main.py`` or import :func:`run_pipeline` directly.
"""

from __future__ import annotations

import os
import time
from typing import Optional

from tqdm import tqdm  # type: ignore

from .db_manager import DBManager
from .embeddings import embed_note
from .vault_parser import Note, parse_vault


def run_pipeline(
    vault_path: str,
    db_path: str = "vault_memory.db",
    embed: bool = True,
    embedding_backend: Optional[str] = None,
    compile_wiki: bool = False,
    force_reindex: bool = False,
    verbose: bool = False,
) -> dict:
    """
    Run the full ingestion pipeline from vault to SQLite knowledge base.

    Parameters
    ----------
    vault_path:
        Path to the root of the Obsidian vault directory.
    db_path:
        Output SQLite database file path.
    embed:
        Whether to generate and store embeddings for semantic search.
    embedding_backend:
        ``"openai"`` or ``"ollama"`` (see :mod:`src.embeddings`).
    compile_wiki:
        If True, run the LLM compiler after ingestion to produce wiki articles.
    force_reindex:
        Re-embed notes even if they haven't changed since the last run.
    verbose:
        Print progress messages.

    Returns
    -------
    A stats dict with keys: ``notes_parsed``, ``notes_upserted``,
    ``notes_embedded``, ``wiki_articles``, ``elapsed_seconds``.
    """
    start = time.time()

    # ------------------------------------------------------------------
    # Step 1: Parse vault
    # ------------------------------------------------------------------
    if verbose:
        print(f"Parsing vault: {vault_path}")

    notes = parse_vault(vault_path, verbose=verbose)

    if verbose:
        print(f"  Found {len(notes)} notes.")

    # ------------------------------------------------------------------
    # Step 2: Ingest into database
    # ------------------------------------------------------------------
    notes_upserted = 0
    notes_embedded = 0

    with DBManager(db_path) as db:
        for note in tqdm(notes, desc="Ingesting notes", disable=not verbose):
            # Check if note needs re-embedding
            existing = db.get_note_by_path(note.path)
            needs_embed = embed and (
                force_reindex
                or existing is None
                or existing["modified_at"] < note.modified_at
            )

            embedding = None
            if needs_embed:
                try:
                    embedding = embed_note(
                        note.content,
                        title=note.title,
                        backend=embedding_backend,
                    )
                    notes_embedded += 1
                except Exception as exc:  # noqa: BLE001
                    if verbose:
                        print(f"    [embed error] {note.path}: {exc}")

            db.upsert_note(
                path=note.path,
                title=note.title,
                content=note.content,
                tags=note.tags,
                backlinks=note.backlinks,
                modified_at=note.modified_at,
                embedding=embedding,
            )
            notes_upserted += 1

        # ------------------------------------------------------------------
        # Step 3 (optional): Compile wiki articles
        # ------------------------------------------------------------------
        wiki_count = 0
        if compile_wiki:
            if verbose:
                print("Running LLM compiler …")
            from .compiler import run_compiler

            wiki_count = run_compiler(
                db,
                embed=embed,
                embedding_backend=embedding_backend,
                verbose=verbose,
            )

        stats = db.stats()

    elapsed = round(time.time() - start, 2)

    return {
        "notes_parsed": len(notes),
        "notes_upserted": notes_upserted,
        "notes_embedded": notes_embedded,
        "wiki_articles": wiki_count,
        "elapsed_seconds": elapsed,
        **stats,
    }
