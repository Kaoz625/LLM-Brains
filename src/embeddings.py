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
