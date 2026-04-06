"""
embeddings.py
-------------
Generate dense vector embeddings for note content.

Supports two backends selected via the EMBEDDING_BACKEND env var:

  openai  – Uses OpenAI's text-embedding-3-small (default, 768 dims truncated).
             Requires OPENAI_API_KEY.

  ollama  – Uses nomic-embed-text via a local Ollama server (free, offline).
             Requires Ollama running at OLLAMA_BASE_URL (default: http://localhost:11434).

Chunking
--------
Long notes are split into overlapping chunks of ~512 tokens so that the
embedding covers the full content.  Each chunk is embedded individually and
the note's final embedding is the mean-pooled average of its chunks.
"""

from __future__ import annotations

import os
import time
from typing import List, Optional

# tiktoken is used for approximate token counting (OpenAI tokeniser).
# The encoding data is downloaded on first use; if offline or unavailable
# we fall back to a character-based approximation.
_tiktoken_enc = None


def _token_count(text: str) -> int:
    """Return an approximate token count for *text*."""
    global _tiktoken_enc
    if _tiktoken_enc is None:
        try:
            import tiktoken  # type: ignore

            _tiktoken_enc = tiktoken.get_encoding("cl100k_base")
        except Exception:  # noqa: BLE001
            # Offline / not installed – use character-based fallback.
            _tiktoken_enc = False  # type: ignore[assignment]

    if _tiktoken_enc:
        return len(_tiktoken_enc.encode(text))
    # Rough approximation: 1 token ≈ 4 characters
    return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CHUNK_SIZE = 512       # tokens per chunk
CHUNK_OVERLAP = 64     # token overlap between consecutive chunks
EMBEDDING_DIM = 768    # target dimensionality


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Split *text* into overlapping chunks of at most *chunk_size* tokens.

    Returns a list of strings.  If the text is short enough it returns
    a single-element list.
    """
    if not text.strip():
        return [""]

    words = text.split()
    chunks: List[str] = []
    start = 0

    while start < len(words):
        # Greedily take words until we hit the token budget
        end = start
        token_budget = chunk_size
        while end < len(words) and token_budget > 0:
            token_budget -= _token_count(words[end])
            end += 1

        chunk = " ".join(words[start:end])
        chunks.append(chunk)

        if end >= len(words):
            break

        # Step forward but keep some overlap
        overlap_tokens = 0
        step = end
        while step > start and overlap_tokens < overlap:
            step -= 1
            overlap_tokens += _token_count(words[step])

        start = max(step, start + 1)  # avoid infinite loop

    return chunks if chunks else [""]


# ---------------------------------------------------------------------------
# OpenAI backend
# ---------------------------------------------------------------------------

def _embed_openai(texts: List[str], model: str = "text-embedding-3-small") -> List[List[float]]:
    """Batch embed *texts* using the OpenAI Embeddings API."""
    import openai  # lazy import

    client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    response = client.embeddings.create(
        input=texts,
        model=model,
        dimensions=EMBEDDING_DIM,
    )
    return [item.embedding for item in response.data]


# ---------------------------------------------------------------------------
# Ollama backend
# ---------------------------------------------------------------------------

def _embed_ollama(
    texts: List[str],
    model: str = "nomic-embed-text",
    base_url: str = "http://localhost:11434",
) -> List[List[float]]:
    """
    Embed *texts* using Ollama's local embedding endpoint.

    Ollama does not support batch embedding in a single request (as of v0.2),
    so we call the API once per text.
    """
    import urllib.request
    import json

    results: List[List[float]] = []
    url = f"{base_url.rstrip('/')}/api/embeddings"

    for text in texts:
        payload = json.dumps({"model": model, "prompt": text}).encode()
        req = urllib.request.Request(
            url, data=payload, headers={"Content-Type": "application/json"}, method="POST"
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read())
        results.append(data["embedding"])

    return results


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def embed_texts(
    texts: List[str],
    backend: Optional[str] = None,
    batch_size: int = 64,
    retry_delay: float = 2.0,
    max_retries: int = 3,
) -> List[List[float]]:
    """
    Return a list of embedding vectors, one per entry in *texts*.

    Parameters
    ----------
    texts:
        The strings to embed.
    backend:
        ``"openai"`` or ``"ollama"``.  Defaults to the
        ``EMBEDDING_BACKEND`` env var, or ``"openai"`` if unset.
    batch_size:
        How many texts to send per API call (OpenAI only).
    retry_delay / max_retries:
        Simple retry logic for transient API errors.
    """
    if not texts:
        return []

    backend = (backend or os.environ.get("EMBEDDING_BACKEND", "openai")).lower()

    all_embeddings: List[List[float]] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]

        for attempt in range(max_retries):
            try:
                if backend == "openai":
                    embs = _embed_openai(batch)
                elif backend == "ollama":
                    ollama_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
                    embs = _embed_ollama(batch, base_url=ollama_url)
                else:
                    raise ValueError(
                        f"Unknown embedding backend '{backend}'. "
                        "Set EMBEDDING_BACKEND to 'openai' or 'ollama'."
                    )
                all_embeddings.extend(embs)
                break
            except Exception as exc:  # noqa: BLE001
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (attempt + 1))
                else:
                    raise RuntimeError(
                        f"Embedding failed after {max_retries} attempts: {exc}"
                    ) from exc

    return all_embeddings


def embed_note(content: str, title: str = "", backend: Optional[str] = None) -> List[float]:
    """
    Embed a single note (possibly long) by chunking, embedding each chunk,
    and returning the mean-pooled vector.

    Parameters
    ----------
    content:
        The note body text.
    title:
        Prepended to the first chunk to give the model context.
    backend:
        Embedding backend (see :func:`embed_texts`).
    """
    text = f"{title}\n\n{content}".strip() if title else content
    chunks = chunk_text(text)
    embeddings = embed_texts(chunks, backend=backend)

    if not embeddings:
        return [0.0] * EMBEDDING_DIM

    # Mean pool
    dim = len(embeddings[0])
    mean_vec = [sum(emb[d] for emb in embeddings) / len(embeddings) for d in range(dim)]
    return mean_vec
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
