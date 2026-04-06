"""
query_example.py
----------------
End-to-end example that:

  1. Creates a tiny in-memory Obsidian-style vault in /tmp/example_vault
  2. Ingests it into an SQLite knowledge base (keyword-search only, no API key)
  3. Runs FTS5 keyword searches
  4. Shows how you would add vector search once an embedding API is configured

Run with::

    python examples/query_example.py
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

# Make sure src/ is importable when running from the project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.db_manager import DBManager
from src.pipeline import run_pipeline
from src.search import keyword_search


# ---------------------------------------------------------------------------
# 1. Build a tiny sample vault
# ---------------------------------------------------------------------------

SAMPLE_NOTES = {
    "AI/transformers.md": """---
title: Transformer Architecture
tags: [ai, deep-learning, nlp]
---
# Transformer Architecture

The Transformer is a neural network architecture introduced in the paper
"Attention Is All You Need" (Vaswani et al., 2017).

It relies entirely on [[self-attention]] mechanisms, dispensing with recurrence.

Key components:
- Multi-head self-attention
- Position-wise feed-forward networks
- Positional encodings

## Benefits
- Parallelisable training (unlike RNNs)
- Long-range dependency modelling
- Scales well with data and compute

[[llm-memory]] is an important application area for transformers.
""",
    "AI/llm-memory.md": """---
title: LLM Memory Systems
tags: [ai, llm, memory, rag]
---
# LLM Memory Systems

Large Language Models have a fixed context window and no persistent memory by default.
Several techniques extend their effective memory:

1. **RAG (Retrieval-Augmented Generation)** – retrieve relevant documents from an external
   store at inference time.  [[sqlite-rag]] is a lightweight implementation.

2. **Vector databases** – store dense embeddings and retrieve by semantic similarity.

3. **Knowledge graphs** – explicit structured representations of facts and relationships.

Karpathy's insight: compile raw notes into a structured wiki once, so the LLM
doesn't re-derive knowledge from scratch on every query. #karpathy #memory
""",
    "Tools/sqlite-rag.md": """---
title: SQLite RAG Setup
tags: [tools, sqlite, rag, memory]
---
# SQLite RAG Setup

SQLite is a surprisingly capable vector store when combined with the
`sqlite-vec` extension.

## Schema
```sql
CREATE VIRTUAL TABLE note_embeddings USING vec0(
  note_id INTEGER PRIMARY KEY,
  embedding FLOAT[768]
);
```

## Benefits
- Single portable `.db` file
- No external service required
- Works offline
- Version-controllable with Git LFS

Use [[llm-memory]] patterns with this for a personal knowledge base.
""",
    "Tools/obsidian-tips.md": """---
title: Obsidian Tips
tags: [tools, obsidian, note-taking]
---
# Obsidian Tips

Obsidian stores notes as plain Markdown files – perfect for version control.

- Use `[[wikilinks]]` to connect ideas
- Front-matter YAML for metadata
- The Obsidian Web Clipper browser extension adds web pages straight to your vault

The Obsidian CEO (Steph Ango) recommends keeping a separate vault for
AI-generated content to avoid contaminating your personal notes. #obsidian
""",
}


def build_sample_vault(base_dir: str) -> str:
    vault = Path(base_dir) / "example_vault"
    for rel_path, content in SAMPLE_NOTES.items():
        note_file = vault / rel_path
        note_file.parent.mkdir(parents=True, exist_ok=True)
        note_file.write_text(content, encoding="utf-8")
    print(f"Created sample vault with {len(SAMPLE_NOTES)} notes at: {vault}")
    return str(vault)


# ---------------------------------------------------------------------------
# 2. Ingest
# ---------------------------------------------------------------------------

def ingest_vault(vault_path: str, db_path: str) -> None:
    print(f"\nIngesting vault into: {db_path}")
    stats = run_pipeline(
        vault_path=vault_path,
        db_path=db_path,
        embed=False,           # no API key required for this example
        compile_wiki=False,
        verbose=True,
    )
    print(f"\nStats: {stats}")


# ---------------------------------------------------------------------------
# 3. Search
# ---------------------------------------------------------------------------

def run_searches(db_path: str) -> None:
    queries = [
        "transformer attention",
        "sqlite vector embeddings",
        "obsidian note-taking vault",
        "RAG memory retrieval",
    ]

    with DBManager(db_path) as db:
        for query in queries:
            results = keyword_search(db, query, top_k=3)
            print(f"\n🔍 Query: '{query}'")
            if not results:
                print("  (no results)")
            for i, r in enumerate(results, 1):
                preview = r["content"].strip().replace("\n", " ")[:100]
                print(f"  {i}. [{r['source']}] {r['title']}")
                print(f"     {preview}…")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="llm_brains_example_") as tmpdir:
        vault_path = build_sample_vault(tmpdir)
        db_path = os.path.join(tmpdir, "example_vault.db")

        ingest_vault(vault_path, db_path)
        run_searches(db_path)

    print("\n✅ Example complete. (Temp files cleaned up.)")
