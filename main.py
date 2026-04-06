"""
main.py
-------
Command-line interface for LLM-Brains.

Commands
--------
  ingest   Parse an Obsidian vault and populate the SQLite knowledge base.
  search   Run a hybrid keyword + vector search query.
  compile  Run the LLM compiler to generate wiki articles from clustered notes.
  stats    Print database statistics.

Example usage::

    # Ingest a vault (keyword-search only, no embedding API needed)
    python main.py ingest ~/Documents/MyVault --no-embed

    # Ingest with OpenAI embeddings
    OPENAI_API_KEY=sk-... python main.py ingest ~/Documents/MyVault

    # Ingest with local Ollama embeddings
    EMBEDDING_BACKEND=ollama python main.py ingest ~/Documents/MyVault

    # Search
    python main.py search "transformer attention mechanisms"

    # Compile wiki articles (requires LLM_BACKEND to be set)
    LLM_BACKEND=openai OPENAI_API_KEY=sk-... python main.py compile

    # Database stats
    python main.py stats
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import click
from dotenv import load_dotenv

load_dotenv()  # load .env file if present

DEFAULT_DB = "vault_memory.db"


@click.group()
def cli() -> None:
    """LLM-Brains: Obsidian vault → SQLite + RAG knowledge base."""


# ---------------------------------------------------------------------------
# ingest
# ---------------------------------------------------------------------------

@cli.command()
@click.argument("vault_path", type=click.Path(exists=True, file_okay=False))
@click.option("--db", default=DEFAULT_DB, show_default=True, help="SQLite database file path.")
@click.option(
    "--embed/--no-embed",
    default=True,
    show_default=True,
    help="Generate and store embeddings for semantic search.",
)
@click.option(
    "--backend",
    default=None,
    envvar="EMBEDDING_BACKEND",
    help="Embedding backend: 'openai' (default) or 'ollama'.",
)
@click.option(
    "--compile/--no-compile",
    "compile_wiki",
    default=False,
    show_default=True,
    help="Run the LLM compiler after ingestion to generate wiki articles.",
)
@click.option("--force", is_flag=True, help="Re-embed all notes, even unchanged ones.")
@click.option("-v", "--verbose", is_flag=True, help="Print detailed progress.")
def ingest(
    vault_path: str,
    db: str,
    embed: bool,
    backend: str,
    compile_wiki: bool,
    force: bool,
    verbose: bool,
) -> None:
    """Parse VAULT_PATH and ingest into the SQLite knowledge base."""
    from src.pipeline import run_pipeline

    click.echo(f"Ingesting vault: {vault_path}")
    click.echo(f"Database: {db}")
    if embed:
        click.echo(f"Embeddings: {backend or os.environ.get('EMBEDDING_BACKEND', 'openai')}")

    stats = run_pipeline(
        vault_path=vault_path,
        db_path=db,
        embed=embed,
        embedding_backend=backend,
        compile_wiki=compile_wiki,
        force_reindex=force,
        verbose=verbose,
    )

    click.echo("\n✅ Ingestion complete:")
    click.echo(f"   Notes parsed   : {stats['notes_parsed']}")
    click.echo(f"   Notes upserted : {stats['notes_upserted']}")
    click.echo(f"   Embeddings     : {stats['notes_embedded']}")
    if compile_wiki:
        click.echo(f"   Wiki articles  : {stats['wiki_articles']}")
    click.echo(f"   Elapsed        : {stats['elapsed_seconds']}s")
    click.echo(f"   Database       : {stats['db_path']}")


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------

@cli.command()
@click.argument("query")
@click.option("--db", default=DEFAULT_DB, show_default=True, help="SQLite database file path.")
@click.option("-k", "--top-k", default=10, show_default=True, help="Number of results to return.")
@click.option(
    "--mode",
    type=click.Choice(["hybrid", "keyword", "vector"], case_sensitive=False),
    default="hybrid",
    show_default=True,
    help="Search mode.",
)
@click.option(
    "--backend",
    default=None,
    envvar="EMBEDDING_BACKEND",
    help="Embedding backend for vector search.",
)
@click.option(
    "--notes-only",
    is_flag=True,
    help="Search raw notes only (skip wiki articles).",
)
@click.option(
    "--wiki-only",
    is_flag=True,
    help="Search wiki articles only (skip raw notes).",
)
def search(
    query: str,
    db: str,
    top_k: int,
    mode: str,
    backend: str,
    notes_only: bool,
    wiki_only: bool,
) -> None:
    """Search the knowledge base with QUERY."""
    from src.db_manager import DBManager
    from src.search import hybrid_search, keyword_search

    if not Path(db).exists():
        click.echo(f"Database not found: {db}. Run 'ingest' first.", err=True)
        sys.exit(1)

    with DBManager(db) as db_mgr:
        if mode == "keyword":
            results = keyword_search(db_mgr, query, top_k=top_k)
        elif mode == "vector":
            results = hybrid_search(
                db_mgr,
                query,
                top_k=top_k,
                fts_weight=0.0,
                vec_weight=1.0,
                embedding_backend=backend,
                search_wiki=not notes_only,
                search_notes=not wiki_only,
            )
        else:
            results = hybrid_search(
                db_mgr,
                query,
                top_k=top_k,
                embedding_backend=backend,
                search_wiki=not notes_only,
                search_notes=not wiki_only,
            )

    if not results:
        click.echo("No results found.")
        return

    click.echo(f"\n🔍 Top {len(results)} results for: \"{query}\"\n")
    for i, r in enumerate(results, start=1):
        source_label = "📝 Note" if r["source"] == "note" else "📖 Wiki"
        location = r.get("path") or r.get("slug", "")
        click.echo(f"{i:2}. {source_label}  {r['title']}")
        click.echo(f"     {location}")
        # Print a content preview
        preview = r["content"].strip().replace("\n", " ")[:160]
        click.echo(f"     {preview}…" if len(r["content"]) > 160 else f"     {preview}")
        click.echo()


# ---------------------------------------------------------------------------
# compile
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--db", default=DEFAULT_DB, show_default=True, help="SQLite database file path.")
@click.option(
    "--embed/--no-embed",
    default=True,
    show_default=True,
    help="Embed compiled wiki articles.",
)
@click.option(
    "--backend",
    default=None,
    envvar="EMBEDDING_BACKEND",
    help="Embedding backend.",
)
@click.option("-v", "--verbose", is_flag=True)
def compile(db: str, embed: bool, backend: str, verbose: bool) -> None:
    """Run the LLM compiler to synthesise wiki articles from note clusters."""
    from src.compiler import run_compiler
    from src.db_manager import DBManager

    if not Path(db).exists():
        click.echo(f"Database not found: {db}. Run 'ingest' first.", err=True)
        sys.exit(1)

    with DBManager(db) as db_mgr:
        count = run_compiler(
            db_mgr,
            embed=embed,
            embedding_backend=backend,
            verbose=verbose,
        )

    click.echo(f"✅ Compiled {count} wiki articles into {db}")


# ---------------------------------------------------------------------------
# stats
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--db", default=DEFAULT_DB, show_default=True, help="SQLite database file path.")
def stats(db: str) -> None:
    """Print database statistics."""
    from src.db_manager import DBManager

    if not Path(db).exists():
        click.echo(f"Database not found: {db}", err=True)
        sys.exit(1)

    with DBManager(db) as db_mgr:
        s = db_mgr.stats()

    click.echo(f"\n📊 LLM-Brains Database Stats")
    click.echo(f"   Path          : {s['db_path']}")
    click.echo(f"   Notes         : {s['notes']}")
    click.echo(f"   Wiki articles : {s['wiki_articles']}")
    click.echo(f"   Embeddings    : {s['embeddings']}")
    click.echo(f"   sqlite-vec    : {'✅ available' if s['sqlite_vec'] else '❌ not installed'}")


if __name__ == "__main__":
    cli()
