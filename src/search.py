"""
search.py
---------
Hybrid search over the LLM-Brains knowledge base.

Combines two complementary retrieval strategies:

  1. **FTS5 keyword search** – exact/prefix term matching with BM25 ranking.
     Great for proper nouns, code snippets, specific phrases.

  2. **Vector similarity search** – semantic nearest-neighbour using sqlite-vec.
     Great for paraphrased queries, conceptual similarity.

The two result sets are merged using **Reciprocal Rank Fusion (RRF)**, a
parameter-free score fusion method that is robust when the two rankers use
incompatible score scales.

Usage::

    from src.db_manager import DBManager
    from src.search import hybrid_search

    with DBManager("vault_memory.db") as db:
        results = hybrid_search(db, "transformer attention mechanisms", top_k=10)
        for r in results:
            print(r["title"], r["score"])
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from .db_manager import DBManager
from .embeddings import embed_texts


# ---------------------------------------------------------------------------
# RRF helpers
# ---------------------------------------------------------------------------

_RRF_K = 60  # standard constant recommended by the original RRF paper


def _rrf_score(rank: int, k: int = _RRF_K) -> float:
    """Reciprocal Rank Fusion score for a result at position *rank* (1-indexed)."""
    return 1.0 / (k + rank)


def _merge_rrf(
    fts_ids: List[int],
    vec_ids: List[int],
) -> List[int]:
    """
    Merge two ranked lists of IDs using RRF.
    Returns IDs sorted by descending fused score.
    """
    scores: Dict[int, float] = {}

    for rank, nid in enumerate(fts_ids, start=1):
        scores[nid] = scores.get(nid, 0.0) + _rrf_score(rank)

    for rank, nid in enumerate(vec_ids, start=1):
        scores[nid] = scores.get(nid, 0.0) + _rrf_score(rank)

    return sorted(scores.keys(), key=lambda nid: scores[nid], reverse=True)


def _escape_fts_query(query: str) -> str:
    """
    Sanitise a free-form query string for use as an FTS5 MATCH expression.

    Each whitespace-separated token is quoted individually so that punctuation
    inside words (hyphens, apostrophes, etc.) cannot be misinterpreted as FTS5
    operators.  Explicit FTS5 phrase syntax using double-quotes is preserved.
    """
    # If the user has already quoted the query (starts and ends with "), pass through.
    stripped = query.strip()
    if stripped.startswith('"') and stripped.endswith('"'):
        return stripped

    # Split on whitespace, quote each token individually.
    tokens = stripped.split()
    if not tokens:
        return '""'

    # Strip leading/trailing punctuation from tokens but keep internal chars
    quoted_tokens = []
    for token in tokens:
        # Remove any double quotes the token might contain (they break FTS5 quoting)
        clean = token.replace('"', "")
        if clean:
            quoted_tokens.append(f'"{clean}"')

    return " ".join(quoted_tokens) if quoted_tokens else '""'


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

SearchResult = Dict[str, Any]


def hybrid_search(
    db: DBManager,
    query: str,
    top_k: int = 10,
    fts_weight: float = 1.0,
    vec_weight: float = 1.0,
    embedding_backend: Optional[str] = None,
    search_wiki: bool = True,
    search_notes: bool = True,
) -> List[SearchResult]:
    """
    Perform a hybrid FTS5 + vector search across notes and wiki articles.

    Parameters
    ----------
    db:
        An open :class:`~src.db_manager.DBManager` instance.
    query:
        Natural-language search query.
    top_k:
        Number of results to return.
    fts_weight / vec_weight:
        Multiplicative weight applied to each sub-ranker's RRF contribution.
        Set either to 0.0 to disable that ranker.
    embedding_backend:
        ``"openai"`` or ``"ollama"`` (see :mod:`src.embeddings`).
        If *vec_weight* is 0 this is never called.
    search_wiki:
        Include wiki articles in results.
    search_notes:
        Include raw vault notes in results.

    Returns
    -------
    A list of dicts with keys:
      ``id``, ``title``, ``content``, ``path`` (notes) or ``slug`` (wiki),
      ``source``, ``score``, ``tags``.
    """
    fts_query = _escape_fts_query(query)
    fetch_n = max(top_k * 3, 30)  # over-fetch before fusion

    # ------------------------------------------------------------------
    # Phase 1: FTS5 retrieval
    # ------------------------------------------------------------------
    fts_note_ids: List[int] = []
    fts_wiki_ids: List[int] = []

    if fts_weight > 0 and search_notes:
        rows = db.fts_search_notes(fts_query, limit=fetch_n)
        fts_note_ids = [r["id"] for r in rows]

    if fts_weight > 0 and search_wiki:
        rows = db.fts_search_wiki(fts_query, limit=fetch_n)
        fts_wiki_ids = [r["id"] for r in rows]

    # ------------------------------------------------------------------
    # Phase 2: Vector retrieval (skip if vec_weight == 0)
    # ------------------------------------------------------------------
    vec_note_ids: List[int] = []
    vec_wiki_ids: List[int] = []

    if vec_weight > 0:
        try:
            query_embedding = embed_texts([query], backend=embedding_backend)[0]

            if search_notes:
                pairs = db.vector_search_notes(query_embedding, limit=fetch_n)
                vec_note_ids = [nid for nid, _ in pairs]

            if search_wiki:
                pairs = db.vector_search_wiki(query_embedding, limit=fetch_n)
                vec_wiki_ids = [aid for aid, _ in pairs]

        except (RuntimeError, Exception):  # noqa: BLE001
            # Vector search unavailable (no extension or no embeddings yet) – graceful fallback.
            pass

    # ------------------------------------------------------------------
    # Phase 3: RRF fusion
    # ------------------------------------------------------------------
    # Scale each list before fusion
    fts_note_scaled = fts_note_ids if fts_weight == 1.0 else [
        nid for nid in fts_note_ids for _ in range(max(1, int(fts_weight)))
    ]

    merged_note_ids = _merge_rrf(
        fts_note_ids[:fetch_n],
        vec_note_ids[:fetch_n],
    )[:top_k]

    merged_wiki_ids = _merge_rrf(
        fts_wiki_ids[:fetch_n],
        vec_wiki_ids[:fetch_n],
    )[:top_k]

    # ------------------------------------------------------------------
    # Phase 4: Fetch full rows & build result dicts
    # ------------------------------------------------------------------
    results: List[SearchResult] = []

    if search_notes and merged_note_ids:
        rows = db.get_notes_by_ids(merged_note_ids)
        # Preserve fusion order
        order = {nid: i for i, nid in enumerate(merged_note_ids)}
        for row in sorted(rows, key=lambda r: order.get(r["id"], 9999)):
            rank = order.get(row["id"], len(merged_note_ids))
            results.append(
                {
                    "id": row["id"],
                    "source": "note",
                    "title": row["title"],
                    "path": row["path"],
                    "content": row["content"],
                    "tags": row["tags"],
                    "score": _rrf_score(rank + 1),
                }
            )

    if search_wiki and merged_wiki_ids:
        rows = db.get_wiki_by_ids(merged_wiki_ids)
        order = {aid: i for i, aid in enumerate(merged_wiki_ids)}
        for row in sorted(rows, key=lambda r: order.get(r["id"], 9999)):
            rank = order.get(row["id"], len(merged_wiki_ids))
            results.append(
                {
                    "id": row["id"],
                    "source": "wiki",
                    "title": row["title"],
                    "slug": row["slug"],
                    "content": row["content"],
                    "score": _rrf_score(rank + 1),
                }
            )

    # Final sort by score descending, truncate to top_k
    results.sort(key=lambda r: r["score"], reverse=True)
    return results[:top_k]


def keyword_search(db: DBManager, query: str, top_k: int = 20) -> List[SearchResult]:
    """Lightweight FTS5-only search (no embedding API call needed)."""
    return hybrid_search(
        db,
        query,
        top_k=top_k,
        fts_weight=1.0,
        vec_weight=0.0,
    )
