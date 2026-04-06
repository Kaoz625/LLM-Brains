"""
src/embeddings.py — EmbeddingEngine class.

Generates embeddings using OpenAI text-embedding-3-small (or TF-IDF fallback),
stores/retrieves from SQLite, and performs cosine similarity search.
"""

import json
import math
import os
import re
from pathlib import Path
from typing import Optional

import numpy as np


class EmbeddingEngine:
    """
    Generates and manages text embeddings.

    Primary: OpenAI text-embedding-3-small (1536-dim)
    Fallback: Hash-based TF-IDF projection (512-dim)
    """

    OPENAI_MODEL = "text-embedding-3-small"
    OPENAI_DIM = 1536
    TFIDF_DIM = 512
    MAX_TEXT_LENGTH = 8192

    def __init__(self, db_manager=None):
        self.db_manager = db_manager
        self._openai_client = None
        self._tfidf_idf: dict[str, float] = {}
        self._vocab: dict[str, int] = {}
        self._doc_count = 0
        self._model_name = None
        self._init_openai()

    def _init_openai(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return
        try:
            import openai
            self._openai_client = openai.OpenAI(api_key=api_key)
            self._model_name = self.OPENAI_MODEL
        except ImportError:
            pass

    @property
    def model_name(self) -> str:
        return self._model_name or f"tfidf-{self.TFIDF_DIM}"

    @property
    def dim(self) -> int:
        return self.OPENAI_DIM if self._openai_client else self.TFIDF_DIM

    # ------------------------------------------------------------------
    # Embedding generation
    # ------------------------------------------------------------------

    def embed(self, text: str) -> np.ndarray:
        """Generate embedding for text. Returns normalized numpy array."""
        text = text[:self.MAX_TEXT_LENGTH].strip()
        if not text:
            return np.zeros(self.dim, dtype=np.float32)

        if self._openai_client:
            return self._embed_openai(text)
        return self._embed_tfidf(text)

    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        """Generate embeddings for multiple texts efficiently."""
        if self._openai_client:
            return self._embed_openai_batch(texts)
        return [self._embed_tfidf(t) for t in texts]

    def _embed_openai(self, text: str) -> np.ndarray:
        try:
            response = self._openai_client.embeddings.create(
                model=self.OPENAI_MODEL,
                input=text,
            )
            vec = np.array(response.data[0].embedding, dtype=np.float32)
            return self._normalize(vec)
        except Exception as e:
            import sys
            print(f"[EmbeddingEngine] OpenAI error: {e}", file=sys.stderr)
            return self._embed_tfidf(text)

    def _embed_openai_batch(self, texts: list[str]) -> list[np.ndarray]:
        if not texts:
            return []
        try:
            response = self._openai_client.embeddings.create(
                model=self.OPENAI_MODEL,
                input=[t[:self.MAX_TEXT_LENGTH] for t in texts],
            )
            return [
                self._normalize(np.array(d.embedding, dtype=np.float32))
                for d in response.data
            ]
        except Exception as e:
            import sys
            print(f"[EmbeddingEngine] Batch OpenAI error: {e}", file=sys.stderr)
            return [self._embed_tfidf(t) for t in texts]

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r"\b[a-zA-Z][a-z]{2,}\b", text.lower())

    def _embed_tfidf(self, text: str) -> np.ndarray:
        """Hash-based TF-IDF projection into fixed-dim vector space."""
        tokens = self._tokenize(text)
        if not tokens:
            return np.zeros(self.TFIDF_DIM, dtype=np.float32)

        # Term frequency
        tf: dict[str, int] = {}
        for t in tokens:
            tf[t] = tf.get(t, 0) + 1

        vec = np.zeros(self.TFIDF_DIM, dtype=np.float32)
        for term, count in tf.items():
            # Hash into fixed-dim bucket
            bucket = hash(term) % self.TFIDF_DIM
            idf = self._tfidf_idf.get(term, 1.0)
            tfidf = (count / len(tokens)) * idf
            vec[bucket] += tfidf

        return self._normalize(vec)

    def update_idf(self, corpus: list[str]):
        """Update IDF weights from a corpus of documents."""
        n_docs = len(corpus)
        doc_freq: dict[str, int] = {}
        for doc in corpus:
            terms = set(self._tokenize(doc))
            for t in terms:
                doc_freq[t] = doc_freq.get(t, 0) + 1
        self._tfidf_idf = {
            t: math.log(n_docs / (df + 1)) + 1
            for t, df in doc_freq.items()
        }
        self._doc_count = n_docs

    @staticmethod
    def _normalize(vec: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vec)
        if norm < 1e-10:
            return vec
        return vec / norm

    # ------------------------------------------------------------------
    # Similarity
    # ------------------------------------------------------------------

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        if a.shape != b.shape:
            min_dim = min(len(a), len(b))
            a, b = a[:min_dim], b[:min_dim]
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a < 1e-10 or norm_b < 1e-10:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def find_similar(self, query_vec: np.ndarray,
                     doc_vecs: list[tuple[Any, np.ndarray]],
                     top_k: int = 10) -> list[tuple[Any, float]]:
        """
        Find top-k most similar documents to query vector.

        Args:
            query_vec: Query embedding
            doc_vecs: List of (doc_id, vector) tuples
            top_k: Number of results

        Returns:
            List of (doc_id, similarity_score) sorted by score descending
        """
        if not doc_vecs:
            return []

        scored = []
        for doc_id, doc_vec in doc_vecs:
            sim = self.cosine_similarity(query_vec, doc_vec)
            scored.append((doc_id, sim))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    # ------------------------------------------------------------------
    # Database integration
    # ------------------------------------------------------------------

    def index_text(self, note_id: int, text: str) -> np.ndarray:
        """Generate embedding for text and store in database."""
        vec = self.embed(text)
        if self.db_manager:
            self.db_manager.upsert_embedding(note_id, self.model_name, vec.tolist())
        return vec

    def search_similar(self, query: str, top_k: int = 10) -> list[dict]:
        """
        Search for notes similar to query using stored embeddings.

        Returns list of dicts with note fields + similarity score.
        """
        if self.db_manager is None:
            return []

        query_vec = self.embed(query)
        all_embeddings = self.db_manager.get_all_embeddings()

        doc_vecs = []
        row_map = {}
        for row in all_embeddings:
            try:
                vec = np.array(json.loads(row["vector_json"]), dtype=np.float32)
                doc_vecs.append((row["note_id"], vec))
                row_map[row["note_id"]] = row
            except Exception:
                continue

        scored = self.find_similar(query_vec, doc_vecs, top_k=top_k)

        results = []
        for note_id, score in scored:
            row = row_map.get(note_id, {})
            results.append({
                "id": note_id,
                "score": score,
                "path": row.get("path", ""),
                "title": row.get("title", ""),
                "content": row.get("content", ""),
                "route": row.get("route", ""),
            })
        return results

    def rebuild_index(self, notes: list[dict]) -> int:
        """Rebuild embeddings index for a list of notes."""
        indexed = 0
        batch_size = 10
        for i in range(0, len(notes), batch_size):
            batch = notes[i:i + batch_size]
            texts = [n.get("content", "")[:self.MAX_TEXT_LENGTH] for n in batch]
            vecs = self.embed_batch(texts)
            for note, vec in zip(batch, vecs):
                if self.db_manager:
                    self.db_manager.upsert_embedding(
                        note["id"], self.model_name, vec.tolist()
                    )
                indexed += 1
        return indexed

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    @staticmethod
    def vec_to_json(vec: np.ndarray) -> str:
        return json.dumps(vec.tolist())

    @staticmethod
    def json_to_vec(json_str: str) -> np.ndarray:
        return np.array(json.loads(json_str), dtype=np.float32)

    def __repr__(self) -> str:
        return f"EmbeddingEngine(model={self.model_name}, dim={self.dim})"


# ---------------------------------------------------------------------------
# Standalone usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    engine = EmbeddingEngine()
    print(f"Engine: {engine}")

    if len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
        vec = engine.embed(text)
        print(f"Text: {text[:50]}...")
        print(f"Vector shape: {vec.shape}")
        print(f"Norm: {np.linalg.norm(vec):.4f}")
        print(f"First 5 dims: {vec[:5]}")
