"""
compiler.py
-----------
The Karpathy "Compiler" Layer.

Takes clusters of related raw vault notes and uses an LLM to synthesize
them into structured wiki-style concept articles.  These compiled articles
are stored back in the database alongside the raw notes.

The compilation process:
  1. Cluster notes by shared tags / wikilinks (see :func:`cluster_notes`).
  2. For each cluster, call the LLM with a structured prompt that asks it
     to produce a concise, well-linked concept article.
  3. Upsert the resulting article into ``wiki_articles`` (and optionally
     embed it for vector search).

Health checks (periodic LLM passes) can also flag contradictions, stale
information, or missing cross-links.

Supported LLM backends (set via LLM_BACKEND env var):
  openai  – GPT-4o or GPT-3.5-turbo (OPENAI_API_KEY required)
  ollama  – Local model via Ollama (OLLAMA_BASE_URL, OLLAMA_MODEL)
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import time
from typing import Dict, List, Optional, Tuple

from .db_manager import DBManager
from .embeddings import embed_note


# ---------------------------------------------------------------------------
# Note clustering
# ---------------------------------------------------------------------------

def cluster_notes(
    notes: List[sqlite3.Row],
    min_cluster_size: int = 2,
    max_cluster_size: int = 12,
) -> List[List[sqlite3.Row]]:
    """
    Group notes by shared tags.

    Each note may appear in multiple clusters if it shares tags with different
    groups.  Clusters smaller than *min_cluster_size* are dropped.

    Returns a list of clusters, each cluster being a list of note rows.
    """
    tag_map: Dict[str, List[sqlite3.Row]] = {}

    for note in notes:
        tags_raw = note["tags"] if isinstance(note["tags"], str) else "[]"
        try:
            tags: List[str] = json.loads(tags_raw)
        except (json.JSONDecodeError, TypeError):
            tags = []

        for tag in tags:
            tag_map.setdefault(tag, []).append(note)

    clusters: List[List[sqlite3.Row]] = []
    seen: set = set()

    for tag, members in tag_map.items():
        if len(members) < min_cluster_size:
            continue

        key = frozenset(r["id"] for r in members)
        if key in seen:
            continue
        seen.add(key)

        clusters.append(members[:max_cluster_size])

    return clusters


def _slug_from_title(title: str) -> str:
    """Convert a title to a URL-safe slug."""
    slug = title.lower().strip()
    slug = re.sub(r"[^\w\s-]", "", slug)
    slug = re.sub(r"[\s_-]+", "-", slug)
    return slug.strip("-")[:80]


# ---------------------------------------------------------------------------
# LLM backends
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are a knowledge base compiler. Your job is to synthesize
a collection of related notes into a single, well-structured concept article.

Guidelines:
- Write in clear, concise prose.
- Use Markdown headers (##, ###) to organize sections.
- Include a brief summary paragraph at the top.
- Use [[wikilinks]] to reference concepts that deserve their own article.
- Resolve any contradictions by noting both perspectives.
- End with a "## See Also" section listing related topics.
- Do NOT hallucinate facts not present in the source notes.
"""

_USER_PROMPT_TEMPLATE = """Please compile the following {n} notes about the topic "{topic}" into
a single structured concept article. Use all relevant information from the notes.

--- SOURCE NOTES ---
{notes_text}
--- END NOTES ---

Write the compiled article now:"""


def _call_openai(
    system: str,
    user: str,
    model: str = "gpt-4o-mini",
) -> str:
    import openai  # lazy import

    client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.3,
        max_tokens=2048,
    )
    return response.choices[0].message.content or ""


def _call_ollama(
    system: str,
    user: str,
    model: Optional[str] = None,
    base_url: str = "http://localhost:11434",
) -> str:
    import urllib.request

    model = model or os.environ.get("OLLAMA_MODEL", "llama3")
    payload = json.dumps(
        {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
        }
    ).encode()

    url = f"{base_url.rstrip('/')}/api/chat"
    req = urllib.request.Request(
        url, data=payload, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read())
    return data.get("message", {}).get("content", "")


def _call_llm(system: str, user: str) -> str:
    backend = os.environ.get("LLM_BACKEND", "openai").lower()
    if backend == "openai":
        model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        return _call_openai(system, user, model=model)
    elif backend == "ollama":
        base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        model = os.environ.get("OLLAMA_MODEL", "llama3")
        return _call_ollama(system, user, model=model, base_url=base_url)
    else:
        raise ValueError(
            f"Unknown LLM_BACKEND '{backend}'. Set to 'openai' or 'ollama'."
        )


# ---------------------------------------------------------------------------
# Compilation
# ---------------------------------------------------------------------------

def compile_cluster(
    cluster: List[sqlite3.Row],
    topic: str,
) -> Tuple[str, str]:
    """
    Compile a cluster of notes into a wiki article.

    Returns ``(title, content)`` where *title* is the article heading
    and *content* is the full Markdown body.
    """
    notes_text = "\n\n---\n\n".join(
        f"## {r['title']}\n\n{r['content']}" for r in cluster
    )

    user_prompt = _USER_PROMPT_TEMPLATE.format(
        n=len(cluster),
        topic=topic,
        notes_text=notes_text[:12_000],  # truncate to avoid context overflow
    )

    content = _call_llm(_SYSTEM_PROMPT, user_prompt)
    return topic, content


def run_compiler(
    db: DBManager,
    min_cluster_size: int = 2,
    max_cluster_size: int = 10,
    embed: bool = True,
    embedding_backend: Optional[str] = None,
    verbose: bool = False,
) -> int:
    """
    Run the full compilation pipeline:

      1. Load all notes from the database.
      2. Cluster them by shared tags.
      3. Compile each cluster → wiki article.
      4. Upsert articles (and optional embeddings) into the database.

    Returns the number of articles upserted.
    """
    notes = db.get_all_notes()
    if not notes:
        if verbose:
            print("No notes found in database. Ingest your vault first.")
        return 0

    clusters = cluster_notes(notes, min_cluster_size=min_cluster_size, max_cluster_size=max_cluster_size)
    if not clusters:
        if verbose:
            print("No clusters found (not enough notes share tags).")
        return 0

    count = 0
    for cluster in clusters:
        # Use the most common tag as the topic label
        all_tags: List[str] = []
        for note in cluster:
            try:
                all_tags.extend(json.loads(note["tags"]))
            except (json.JSONDecodeError, TypeError):
                pass

        topic = all_tags[0] if all_tags else "general"

        if verbose:
            print(f"  compiling cluster '{topic}' ({len(cluster)} notes) …")

        try:
            title, content = compile_cluster(cluster, topic)
        except Exception as exc:  # noqa: BLE001
            if verbose:
                print(f"    [error] {exc}")
            continue

        slug = _slug_from_title(title)
        source_paths = [r["path"] for r in cluster]

        embedding: Optional[List[float]] = None
        if embed:
            try:
                embedding = embed_note(content, title=title, backend=embedding_backend)
            except Exception:  # noqa: BLE001
                pass

        db.upsert_wiki_article(
            slug=slug,
            title=title,
            content=content,
            source_paths=source_paths,
            embedding=embedding,
        )
        count += 1

    if verbose:
        print(f"Compiled {count} wiki articles.")

    return count


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

_HEALTH_SYSTEM = """You are a knowledge base auditor. Analyse the following wiki article
and identify: (1) factual contradictions, (2) stale or potentially outdated claims,
(3) missing cross-links to important related concepts.
Be concise. Output a JSON object with keys: contradictions, stale_claims, missing_links.
"""


def health_check_article(article_content: str) -> dict:
    """
    Ask the LLM to audit a single wiki article for quality issues.

    Returns a dict with keys ``contradictions``, ``stale_claims``, ``missing_links``.
    """
    try:
        raw = _call_llm(_HEALTH_SYSTEM, article_content[:8_000])
        # Extract JSON block if wrapped in markdown fences
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        return {"raw_response": raw}
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}
