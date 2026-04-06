"""
src/search.py — SearchEngine class.

Hybrid FTS5 keyword search + vector semantic search with result merging and reranking.
"""

import math
import os
from pathlib import Path
from typing import Optional

import numpy as np


class SearchEngine:
    """
    Hybrid search combining FTS5 full-text search and vector similarity.

    Uses Reciprocal Rank Fusion (RRF) for score combination.
    Supports result reranking and filtering.
    """

    RRF_K = 60  # RRF constant (standard value)

    def __init__(self, db_manager=None, embedding_engine=None):
        """
        Args:
            db_manager: DatabaseManager instance
            embedding_engine: EmbeddingEngine instance
        """
        self.db = db_manager
        self.emb = embedding_engine

        # Lazy-load if not provided
        if self.db is None:
            from src.db_manager import DatabaseManager
            self.db = DatabaseManager()
        if self.emb is None:
            from src.embeddings import EmbeddingEngine
            self.emb = EmbeddingEngine(db_manager=self.db)

    # ------------------------------------------------------------------
    # Individual search methods
    # ------------------------------------------------------------------

    def keyword_search(self, query: str, limit: int = 20,
                        route: Optional[str] = None) -> list[dict]:
        """FTS5 full-text search. Returns results with BM25 scores."""
        results = self.db.fts_search(query, limit=limit)
        if route:
            results = [r for r in results if r.get("route") == route]
        # Normalize BM25 scores (FTS5 returns negative, lower = better)
        if results:
            scores = [abs(r.get("score", 0)) for r in results]
            max_score = max(scores) or 1.0
            for r in results:
                r["fts_score"] = abs(r.get("score", 0)) / max_score
        return results

    def semantic_search(self, query: str, limit: int = 20,
                         route: Optional[str] = None) -> list[dict]:
        """Vector similarity search. Returns results with cosine similarity scores."""
        results = self.emb.search_similar(query, top_k=limit)
        if route:
            results = [r for r in results if r.get("route") == route]
        for r in results:
            r["vec_score"] = r.pop("score", 0.0)
        return results

    # ------------------------------------------------------------------
    # Hybrid search with RRF
    # ------------------------------------------------------------------

    def hybrid_search(self, query: str, limit: int = 10,
                       route: Optional[str] = None,
                       fts_weight: float = 0.3,
                       vec_weight: float = 0.7) -> list[dict]:
        """
        Hybrid search combining FTS5 + vector search using Reciprocal Rank Fusion.

        RRF score = sum(w_i / (k + rank_i)) for each retrieval system i.

        Args:
            query: Search query
            limit: Maximum number of results
            route: Filter by route (me/work/knowledge/media)
            fts_weight: Weight for FTS5 results
            vec_weight: Weight for vector results

        Returns:
            List of result dicts sorted by hybrid score
        """
        k = self.RRF_K

        # Get results from both systems
        fts_results = self.keyword_search(query, limit=limit * 2, route=route)
        vec_results = self.semantic_search(query, limit=limit * 2, route=route)

        # Build RRF score maps
        fts_rrf: dict[int, float] = {}
        for rank, r in enumerate(fts_results):
            doc_id = r.get("id", hash(r.get("path", "")))
            fts_rrf[doc_id] = fts_weight / (k + rank + 1)

        vec_rrf: dict[int, float] = {}
        for rank, r in enumerate(vec_results):
            doc_id = r.get("id", hash(r.get("path", "")))
            vec_rrf[doc_id] = vec_weight / (k + rank + 1)

        # Merge all unique docs
        all_ids = set(fts_rrf.keys()) | set(vec_rrf.keys())
        id_to_row: dict[int, dict] = {}
        for r in fts_results + vec_results:
            doc_id = r.get("id", hash(r.get("path", "")))
            if doc_id not in id_to_row:
                id_to_row[doc_id] = r

        combined = []
        for doc_id in all_ids:
            row = id_to_row[doc_id].copy()
            rrf_score = fts_rrf.get(doc_id, 0.0) + vec_rrf.get(doc_id, 0.0)
            row["hybrid_score"] = rrf_score
            row["fts_rrf"] = fts_rrf.get(doc_id, 0.0)
            row["vec_rrf"] = vec_rrf.get(doc_id, 0.0)
            combined.append(row)

        combined.sort(key=lambda x: x["hybrid_score"], reverse=True)
        return combined[:limit]

    # ------------------------------------------------------------------
    # Reranking
    # ------------------------------------------------------------------

    def rerank(self, query: str, results: list[dict],
               method: str = "mmr", diversity: float = 0.3) -> list[dict]:
        """
        Rerank results for diversity and relevance.

        Methods:
          - "mmr": Maximum Marginal Relevance (balances relevance + diversity)
          - "score": Pure score-based (no reranking)

        Args:
            query: Original query
            results: Initial ranked results
            method: Reranking method
            diversity: MMR diversity parameter (0=pure relevance, 1=pure diversity)
        """
        if method == "score" or len(results) <= 1:
            return results

        if method == "mmr":
            return self._mmr_rerank(query, results, diversity=diversity)

        return results

    def _mmr_rerank(self, query: str, results: list[dict],
                     diversity: float = 0.3) -> list[dict]:
        """Maximum Marginal Relevance reranking."""
        if not results:
            return []

        query_vec = self.emb.embed(query)

        # Get embedding vectors for results
        vecs = []
        for r in results:
            content = r.get("content", "") or r.get("title", "")
            vecs.append(self.emb.embed(content[:512]))

        selected = []
        remaining = list(range(len(results)))
        selected_indices = []

        while remaining and len(selected) < len(results):
            if not selected_indices:
                # First: pick highest relevance score
                best_idx = max(remaining, key=lambda i: results[i].get("hybrid_score", 0))
            else:
                # MMR: balance relevance and diversity
                best_idx = None
                best_mmr = float("-inf")
                for i in remaining:
                    rel = self.emb.cosine_similarity(query_vec, vecs[i])
                    # Max similarity to already-selected docs
                    max_sim = max(
                        self.emb.cosine_similarity(vecs[i], vecs[j])
                        for j in selected_indices
                    ) if selected_indices else 0.0
                    mmr = (1 - diversity) * rel - diversity * max_sim
                    if mmr > best_mmr:
                        best_mmr = mmr
                        best_idx = i

            if best_idx is None:
                break

            selected.append(results[best_idx])
            selected_indices.append(best_idx)
            remaining.remove(best_idx)

        return selected

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def suggest_similar(self, note_id: int, limit: int = 5) -> list[dict]:
        """Find notes similar to a given note (by ID)."""
        note = self.db.get_note(note_id)
        if not note:
            return []
        return self.semantic_search(note.get("content", "")[:500], limit=limit + 1)

    def search_by_tags(self, tags: list[str], limit: int = 20) -> list[dict]:
        """Search for notes with specific tags."""
        tag_query = " ".join(tags)
        return self.keyword_search(tag_query, limit=limit)

    def format_results(self, results: list[dict], query: str,
                        snippet_length: int = 200) -> str:
        """Format search results as readable text."""
        if not results:
            return f"No results found for: {query}"

        lines = [f"Results for '{query}' ({len(results)} found):", "=" * 60]
        for i, r in enumerate(results, 1):
            score = r.get("hybrid_score", r.get("vec_score", r.get("fts_score", 0)))
            title = r.get("title", Path(r.get("path", "?")).stem)
            route = r.get("route", "?")
            snippet = r.get("content", "")[:snippet_length].replace("\n", " ")
            lines.append(
                f"\n{i}. [{route}] {title} (score: {score:.4f})\n"
                f"   {snippet}..."
            )
        return "\n".join(lines)

    def get_stats(self) -> dict:
        """Return search engine statistics."""
        db_stats = self.db.get_stats()
        return {
            "engine": "hybrid_fts5_vector",
            "rrf_k": self.RRF_K,
            "embedding_model": self.emb.model_name,
            "embedding_dim": self.emb.dim,
            **db_stats,
        }

    def __repr__(self) -> str:
        stats = self.get_stats()
        return (f"SearchEngine(notes={stats['notes']}, "
                f"embeddings={stats['embeddings']}, "
                f"model={stats['embedding_model']})")


# ---------------------------------------------------------------------------
# Standalone usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    se = SearchEngine()
    print(f"SearchEngine: {se}")
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        results = se.hybrid_search(query)
        print(se.format_results(results, query))
