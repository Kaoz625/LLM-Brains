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
src/pipeline.py — Pipeline class orchestrating the full ingest pipeline.

Stages:
  1. Ingest    — detect and extract content from raw files
  2. Compile   — Claude-powered content compilation into structured entries
  3. Index     — store in SQLite with FTS5 + vector embeddings
  4. Generate  — optional studio output generation (podcast, slides, etc.)

Usage:
    from src.pipeline import Pipeline
    p = Pipeline()
    results = p.run()
"""

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class PipelineResult:
    """Holds results from a pipeline run."""

    def __init__(self):
        self.ingested: list[dict] = []
        self.indexed: list[int] = []
        self.generated: list[dict] = []
        self.errors: list[dict] = []
        self.started_at = datetime.now()
        self.finished_at: Optional[datetime] = None

    def add_ingested(self, data: dict):
        self.ingested.append(data)

    def add_error(self, source: str, error: str):
        self.errors.append({"source": source, "error": error, "time": datetime.now().isoformat()})

    def finish(self):
        self.finished_at = datetime.now()

    @property
    def duration_secs(self) -> float:
        if self.finished_at:
            return (self.finished_at - self.started_at).total_seconds()
        return 0.0

    def summary(self) -> dict:
        return {
            "ingested": len(self.ingested),
            "indexed": len(self.indexed),
            "generated": len(self.generated),
            "errors": len(self.errors),
            "duration_secs": round(self.duration_secs, 2),
            "started_at": self.started_at.isoformat(),
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
        }

    def __repr__(self) -> str:
        s = self.summary()
        return (f"PipelineResult(ingested={s['ingested']}, indexed={s['indexed']}, "
                f"errors={s['errors']}, duration={s['duration_secs']}s)")


class Pipeline:
    """
    Full LLM-Brains ingest → compile → index → generate pipeline.

    Wires together Compiler, DatabaseManager, EmbeddingEngine, and optionally
    the StudioGenerator.
    """

    def __init__(
        self,
        brain_dir: Optional[Path] = None,
        api_key: Optional[str] = None,
        model: str = "claude-opus-4-5",
        enable_studio: bool = False,
        studio_formats: Optional[list[str]] = None,
        enable_wiki: bool = True,
        enable_fragments: bool = True,
    ):
        self.brain_dir = Path(brain_dir or os.getenv("BRAIN_DIR", "./brain"))
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY", "")
        self.model = model
        self.enable_studio = enable_studio
        self.studio_formats = studio_formats or ["flashcards", "report"]
        self.enable_wiki = enable_wiki
        self.enable_fragments = enable_fragments

        # Lazy-initialized components
        self._compiler = None
        self._db = None
        self._embedding_engine = None
        self._search_engine = None
        self._fragment_manager = None
        self._studio_generator = None

    # ------------------------------------------------------------------
    # Component accessors (lazy init)
    # ------------------------------------------------------------------

    @property
    def compiler(self):
        if self._compiler is None:
            from src.compiler import Compiler
            self._compiler = Compiler(
                brain_dir=self.brain_dir,
                api_key=self.api_key,
                model=self.model,
            )
        return self._compiler

    @property
    def db(self):
        if self._db is None:
            from src.db_manager import DatabaseManager
            self._db = DatabaseManager(self.brain_dir / "memory.db")
        return self._db

    @property
    def embedding_engine(self):
        if self._embedding_engine is None:
            from src.embeddings import EmbeddingEngine
            self._embedding_engine = EmbeddingEngine(db_manager=self.db)
        return self._embedding_engine

    @property
    def search_engine(self):
        if self._search_engine is None:
            from src.search import SearchEngine
            self._search_engine = SearchEngine(
                db_manager=self.db,
                embedding_engine=self.embedding_engine,
            )
        return self._search_engine

    @property
    def fragment_manager(self):
        if self._fragment_manager is None and self.enable_fragments:
            from fragment_manager import FragmentManager
            self._fragment_manager = FragmentManager(brain_dir=self.brain_dir)
        return self._fragment_manager

    # ------------------------------------------------------------------
    # Pipeline stages
    # ------------------------------------------------------------------

    def stage_ingest(self, source_dir: Optional[Path] = None) -> list[dict]:
        """Stage 1: Ingest all files from raw directory."""
        raw_dir = source_dir or (self.brain_dir / "raw")
        logger.info(f"[Pipeline] Stage 1: Ingest from {raw_dir}")
        self.compiler.ensure_dirs()
        return self.compiler.process_directory(raw_dir)

    def stage_index(self, entries: list[dict]) -> list[int]:
        """Stage 2: Index compiled entries into SQLite."""
        logger.info(f"[Pipeline] Stage 2: Index {len(entries)} entries")
        indexed_ids = []

        for entry in entries:
            if entry.get("skipped"):
                continue
            dest_path = entry.get("dest_path")
            if not dest_path:
                continue
            try:
                path = Path(dest_path)
                content = path.read_text(encoding="utf-8", errors="replace")
                note_id = self.db.upsert_note(
                    path=dest_path,
                    content=content,
                    title=entry.get("title", path.stem),
                    route=entry.get("route", ""),
                    tags=", ".join(entry.get("tags", [])),
                )
                # Generate and store embedding
                self.embedding_engine.index_text(note_id, content[:4096])
                indexed_ids.append(note_id)
            except Exception as e:
                logger.error(f"Index error for {dest_path}: {e}")

        logger.info(f"[Pipeline] Indexed {len(indexed_ids)} entries")
        return indexed_ids

    def stage_fragment_ingest(self, entries: list[dict]):
        """Stage 3: Route entries into fragment agents."""
        if not self.enable_fragments or self.fragment_manager is None:
            return
        logger.info(f"[Pipeline] Stage 3: Fragment ingest for {len(entries)} entries")
        for entry in entries:
            if entry.get("skipped"):
                continue
            content = entry.get("markdown", "")
            title = entry.get("title", "")
            dest = entry.get("dest_path", "")
            if content:
                try:
                    self.fragment_manager.ingest_entry(
                        content=content, title=title, source_path=dest
                    )
                except Exception as e:
                    logger.warning(f"Fragment ingest error: {e}")

    def stage_wiki(self, entries: list[dict]):
        """Stage 4: Compile entries into wiki."""
        if not self.enable_wiki:
            return
        logger.info(f"[Pipeline] Stage 4: Wiki compilation for {len(entries)} entries")
        if not self.api_key:
            logger.warning("No API key — skipping wiki compilation")
            return
        try:
            import anthropic as _anthropic
            from wiki_compiler import compile_to_wiki, save_wiki_entry
            client = _anthropic.Anthropic(api_key=self.api_key)
            for entry in entries:
                if entry.get("skipped"):
                    continue
                dest = entry.get("dest_path")
                if not dest or not Path(dest).exists():
                    continue
                try:
                    content = Path(dest).read_text(encoding="utf-8", errors="replace")
                    wiki_data = compile_to_wiki(client, content, entry.get("title", ""))
                    save_wiki_entry(wiki_data, operation="CREATE")
                except Exception as e:
                    logger.warning(f"Wiki compile error for {dest}: {e}")
        except ImportError as e:
            logger.warning(f"Wiki compiler unavailable: {e}")

    def stage_studio(self, entries: list[dict]) -> list[dict]:
        """Stage 5: Generate studio outputs for compiled entries."""
        if not self.enable_studio:
            return []
        logger.info(f"[Pipeline] Stage 5: Studio generation")
        results = []
        try:
            import anthropic as _anthropic
            from studio_generator import generate_for_entry
            client = _anthropic.Anthropic(api_key=self.api_key)
            wiki_dir = self.brain_dir / "knowledge" / "wiki"
            for md_file in sorted(wiki_dir.glob("*.md"))[:5]:  # Limit to 5 per run
                try:
                    result = generate_for_entry(md_file, client, formats=self.studio_formats)
                    results.append(result)
                except Exception as e:
                    logger.warning(f"Studio generation error for {md_file.name}: {e}")
        except Exception as e:
            logger.warning(f"Studio stage error: {e}")
        return results

    def stage_index_all(self):
        """Index all existing markdown files in brain/ into SQLite."""
        logger.info("[Pipeline] Indexing all existing brain files...")
        indexed = 0
        for md_file in self.brain_dir.rglob("*.md"):
            if any(part.startswith(".") for part in md_file.parts):
                continue
            try:
                content = md_file.read_text(encoding="utf-8", errors="replace")
                title = md_file.stem
                route = ""
                try:
                    import frontmatter
                    post = frontmatter.loads(content)
                    title = str(post.get("title", md_file.stem))
                    route = str(post.get("route", ""))
                except Exception:
                    pass
                note_id = self.db.upsert_note(
                    path=str(md_file), content=content,
                    title=title, route=route,
                )
                self.embedding_engine.index_text(note_id, content[:4096])
                indexed += 1
            except Exception as e:
                logger.warning(f"Could not index {md_file}: {e}")
        logger.info(f"[Pipeline] Indexed {indexed} existing files")
        return indexed

    # ------------------------------------------------------------------
    # Full pipeline run
    # ------------------------------------------------------------------

    def run(self, source_dir: Optional[Path] = None,
             stages: Optional[list[str]] = None) -> PipelineResult:
        """
        Run the full pipeline.

        Args:
            source_dir: Directory to ingest from (default: brain/raw)
            stages: List of stage names to run, or None for all
                    Options: ingest, index, fragments, wiki, studio

        Returns:
            PipelineResult with stats
        """
        result = PipelineResult()
        all_stages = stages or ["ingest", "index", "fragments", "wiki", "studio"]

        logger.info(f"[Pipeline] Starting run. Stages: {all_stages}")

        try:
            # Stage 1: Ingest
            if "ingest" in all_stages:
                entries = self.stage_ingest(source_dir)
                for e in entries:
                    if not e.get("skipped"):
                        result.add_ingested(e)
            else:
                entries = []

            # Stage 2: Index
            if "index" in all_stages and result.ingested:
                ids = self.stage_index(result.ingested)
                result.indexed = ids

            # Stage 3: Fragment ingest
            if "fragments" in all_stages and result.ingested:
                self.stage_fragment_ingest(result.ingested)

            # Stage 4: Wiki
            if "wiki" in all_stages and result.ingested:
                self.stage_wiki(result.ingested)

            # Stage 5: Studio
            if "studio" in all_stages and self.enable_studio:
                gen_results = self.stage_studio(result.ingested)
                result.generated = gen_results

        except Exception as e:
            logger.error(f"[Pipeline] Fatal error: {e}")
            result.add_error("pipeline", str(e))

        result.finish()
        logger.info(f"[Pipeline] Complete: {result.summary()}")
        return result

    def rebuild_index(self) -> int:
        """Rebuild the entire search index from existing brain files."""
        return self.stage_index_all()

    def search(self, query: str, limit: int = 10) -> list[dict]:
        """Run a hybrid search query."""
        return self.search_engine.hybrid_search(query, limit=limit)

    def get_stats(self) -> dict:
        """Return pipeline and database statistics."""
        return {
            "brain_dir": str(self.brain_dir),
            "model": self.model,
            "enable_studio": self.enable_studio,
            "enable_wiki": self.enable_wiki,
            "enable_fragments": self.enable_fragments,
            **self.db.get_stats(),
        }

    def __repr__(self) -> str:
        return (f"Pipeline(brain={self.brain_dir}, model={self.model}, "
                f"studio={self.enable_studio})")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import sys

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="LLM-Brains pipeline")
    parser.add_argument("--run", action="store_true", help="Run full pipeline")
    parser.add_argument("--rebuild-index", action="store_true",
                        help="Rebuild search index from all brain files")
    parser.add_argument("--search", type=str, help="Search query")
    parser.add_argument("--stages", type=str, default="ingest,index,fragments,wiki",
                        help="Comma-separated stages to run")
    parser.add_argument("--studio", action="store_true", help="Enable studio generation")
    parser.add_argument("--stats", action="store_true", help="Show pipeline stats")
    args = parser.parse_args()

    pipeline = Pipeline(enable_studio=args.studio)

    if args.stats:
        import json
        print(json.dumps(pipeline.get_stats(), indent=2))
    elif args.rebuild_index:
        n = pipeline.rebuild_index()
        print(f"Rebuilt index: {n} files indexed")
    elif args.run:
        stages = [s.strip() for s in args.stages.split(",")]
        result = pipeline.run(stages=stages)
        print(f"\nPipeline complete: {result.summary()}")
    elif args.search:
        results = pipeline.search(args.search)
        print(pipeline.search_engine.format_results(results, args.search))
    else:
        parser.print_help()
