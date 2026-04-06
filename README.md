# LLM-Brains

> **Research goal:** Push AI from Generative to General by giving LLMs the same thing that
> makes humans intelligent — persistent, structured, ever-growing memory with pattern recognition
> and intuition built on top.

This project is based on the insight that the gap between current AI and human-level intelligence
isn't model size — it's **memory architecture**. Humans don't re-derive everything from scratch
each time they think. They have episodic memories, compiled semantic knowledge, pattern baselines,
and a "6th sense" that fires when something deviates from normal. This project builds all of that.

---

## Core Insight (Karpathy, April 2026)

Traditional RAG is stateless — every query burns tokens re-reading raw documents and re-deriving
the same answers. Karpathy's paradigm shift: **use the LLM as a compiler, not a search engine.**
Ingest raw content once, compile it into a structured wiki, and query that. Knowledge compounds.

This project extends that idea with: wearable video ingestion, personal life data, RSS continuous
feeds, SQLite + vector search, an intuition engine, and a concept link graph.

---

## The Memory Neuroscience Connection

> "It's not that the memory is gone — the pathway to it is broken.
>  Build more pathways and you can reach the same memory from any direction."

The `concept_links` table in SQLite is literally implementing this. Every `[[wikilink]]` in a
compiled note is an alternate retrieval pathway. The more cross-links, the more durable the
memory. This mirrors how the brain consolidates memories during sleep — finding new connections
between the day's experiences.

---

## Files

| File | Purpose |
|------|---------|
| `compile.py` | Main brain compiler — processes raw/ → brain/ |
| `rss_ingest.py` | Pulls RSS feeds, YouTube, web into raw/ |
| `sqlite_rag.py` | Indexes brain/ into SQLite with FTS5 + vector search |
| `wearable_ingest.py` | Processes Meta glasses / phone video into episodic memories |
| `ARCHITECTURE.md` | Full research breakdown, intuition engine theory, improvement ideas |
| `requirements.txt` | Python dependencies |
| `brain/` | Your personal knowledge structure (gitignored by default) |
# LLM-Brains 🧠

> *Compile your Obsidian vault into a portable SQLite knowledge base that any LLM can query — offline, private, and lightning-fast.*

Inspired by Andrej Karpathy's April 2026 insight on LLM memory architecture, LLM-Brains converts your raw Obsidian notes into a two-layer retrieval system: **FTS5 keyword search** + **sqlite-vec vector search**, all inside a single `.db` file. An optional LLM "compiler" layer synthesises your raw notes into structured wiki articles — so the model doesn't have to re-derive knowledge from scratch on every query.

---

## Why This Matters

LLMs have a fixed context window and no persistent memory by default. The breakthrough insight is:

1. **Collect** raw knowledge (your Obsidian vault)
2. **Compress** it without losing information (chunking + embeddings)
3. **Convert** it into the LLM's native language (structured wiki articles)
4. **Chain** many SQLite files by category — RAG finds what's relevant instantly

This is directly aligned with what leading AI memory companies (Mem, Notion AI, Rewind, etc.) are converging on, and with Karpathy's own research wiki workflow.

---

## Architecture

```
Obsidian Vault (.md files)
         │
         ▼
   vault_parser.py          ← parse frontmatter YAML, wikilinks, #tags
         │
         ▼
   embeddings.py            ← chunk text → OpenAI / Ollama embeddings
         │
         ▼
   db_manager.py ──────────────────────────────────────────────────┐
   ┌─────────────────────────────────────────────────────────────┐ │
   │  SQLite: vault_memory.db                                    │ │
   │                                                             │ │
   │  notes          ← full content, tags, backlinks, mtime      │ │
   │  notes_fts      ← FTS5 virtual table (keyword search)       │ │
   │  note_embeddings← sqlite-vec virtual table (vector search)  │ │
   │                                                             │ │
   │  wiki_articles  ← LLM-compiled concept articles             │ │
   │  wiki_fts       ← FTS5 on wiki articles                     │ │
   │  wiki_embeddings← vectors on wiki articles                  │ │
   └─────────────────────────────────────────────────────────────┘ │
         │                                                          │
         ▼                                                          │
   compiler.py              ← cluster notes → LLM → wiki articles ─┘
         │
         ▼
   search.py                ← hybrid FTS5 + vector + RRF fusion
```

### Memory Architecture (aligned with AI research)

| Memory Type | This Project |
|---|---|
| **Episodic** (specific, dated) | Raw vault notes |
| **Semantic** (distilled facts) | LLM-compiled wiki articles |
| **Retrieval** (at query time) | Hybrid FTS5 + vector search |

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt
brew install ffmpeg  # for video processing

# 2. Set your API key
export ANTHROPIC_API_KEY=your-key-here

# 3. Drop anything into brain/raw/
cp ~/Downloads/some_article.pdf brain/raw/
cp ~/voice_memo.m4a brain/raw/
echo "https://youtube.com/watch?v=VIDEO_ID" > brain/raw/video.url

# 4. Run the compiler
python compile.py

# 5. Build the search index
python sqlite_rag.py --build

# 6. Query your brain
python sqlite_rag.py --query "What do I know about transformers?"
python sqlite_rag.py --query "What was going on with my family last month?"
```

---

## Meta Glasses / Wearable Video

```bash
# Watch your Meta glasses sync folder — auto-process new recordings
python wearable_ingest.py --watch ~/Documents/MetaGlasses/ --interval 60

# Or process a single video manually
python wearable_ingest.py --video ~/Downloads/glasses_recording.mp4
```

Each video is processed as:
1. Keyframes extracted every N seconds
2. Each frame analyzed by Claude vision (people, place, objects, emotions)
3. Audio transcribed by Whisper
4. Synthesized into a rich episodic memory with 10-15+ cross-links
5. Added to `brain/me/experiences/` and `brain/me/timeline.md`

---

## Continuous RSS Feed

```bash
# Pull all configured AI research feeds now
python rss_ingest.py

# Watch feeds and pull every 30 minutes
python rss_ingest.py --watch 30m

# Add your own feed
python rss_ingest.py --add-feed "https://example.com/rss" "MyFeed"
```

Default feeds include arXiv AI/ML/NLP, HuggingFace blog, OpenAI blog, Anthropic blog,
LocalLLaMA, and HackerNews.

---

## The Intuition Engine

```bash
python compile.py --intuition
```

Scans your recent brain entries for:
- Patterns that deviate from your baseline
- High-importance items that need attention
- Hidden connections between recent events
- Contradictions with prior beliefs
- Emerging themes across multiple entries

Flags land in `brain/flagged/` with a timestamp. This is the "Spidey Sense" layer.

---

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full research breakdown including:
- The full pipeline diagram
- SQLite schema with FTS5 + vector layers
- Human intuition mapping to AI systems
- The neuroscience connection (pathways, not data loss)
- 7 concrete improvement ideas (temporal decay, dream mode, contradiction detection, etc.)
- The General AI thesis — why this bridges Generative → General AI

---

## Roadmap

- [ ] MCP server — query your brain directly from Claude Desktop
- [ ] Obsidian vault sync — bidirectional Obsidian ↔ brain/
- [ ] Mobile app — drop content from phone
- [ ] "Dream mode" nightly consolidation cron job
- [ ] Contradiction detection and belief revision tracking
- [ ] Personal growth tracking (how your views evolve over time)
- [ ] Multi-device SQLite sync (branch per device, merge strategy)
### 1. Install dependencies

```bash
pip install -r requirements.txt
```

For vector search, also install the `sqlite-vec` native extension (automatically pulled by `pip install sqlite-vec`).

### 2. Configure your API key (or use Ollama for free/offline)

```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

**Or use Ollama (100% free, offline):**

```bash
ollama pull nomic-embed-text   # embeddings
ollama pull llama3             # LLM compiler
```

Then set in `.env`:

```
EMBEDDING_BACKEND=ollama
LLM_BACKEND=ollama
```

### 3. Ingest your vault

```bash
# With embeddings (recommended)
python main.py ingest ~/Documents/MyVault

# Keyword-search only (no API key needed)
python main.py ingest ~/Documents/MyVault --no-embed

# Ingest + compile wiki articles
python main.py ingest ~/Documents/MyVault --compile
```

### 4. Search

```bash
# Hybrid keyword + semantic search (default)
python main.py search "transformer attention mechanisms"

# Keyword only (fast, no API call)
python main.py search "RAG memory" --mode keyword

# Top 5 results from wiki articles only
python main.py search "memory architecture" --wiki-only -k 5
```

### 5. Compile wiki articles separately

```bash
python main.py compile
```

### 6. Database stats

```bash
python main.py stats
```

---

## CLI Reference

```
Usage: python main.py [COMMAND] [OPTIONS]

Commands:
  ingest   Parse an Obsidian vault and populate the SQLite knowledge base.
  search   Run a hybrid keyword + vector search query.
  compile  Run the LLM compiler to generate wiki articles from note clusters.
  stats    Print database statistics.

ingest options:
  VAULT_PATH          Path to your Obsidian vault directory
  --db PATH           Output database file (default: vault_memory.db)
  --embed/--no-embed  Enable/disable embedding generation (default: on)
  --backend TEXT      Embedding backend: openai | ollama
  --compile           Also run LLM compiler after ingestion
  --force             Re-embed all notes, even unchanged ones
  -v, --verbose       Show detailed progress

search options:
  QUERY               Search query
  --db PATH           Database file (default: vault_memory.db)
  -k, --top-k INT     Number of results (default: 10)
  --mode TEXT         hybrid | keyword | vector (default: hybrid)
  --backend TEXT      Embedding backend for vector search
  --notes-only        Search raw notes only
  --wiki-only         Search wiki articles only
```

---

## Source Modules

| File | Purpose |
|---|---|
| `src/vault_parser.py` | Walk `.md` files, extract YAML frontmatter, wikilinks, #tags |
| `src/db_manager.py` | SQLite schema: FTS5 tables + sqlite-vec vector tables + CRUD |
| `src/embeddings.py` | Chunk text, call OpenAI / Ollama, mean-pool chunk embeddings |
| `src/search.py` | Hybrid FTS5 + vector search with Reciprocal Rank Fusion |
| `src/compiler.py` | Cluster notes by tags, LLM-synthesise wiki articles |
| `src/pipeline.py` | Orchestrate parse → embed → upsert → (compile) |
| `main.py` | Click CLI wrapping the pipeline |
| `examples/query_example.py` | Self-contained demo with a tiny sample vault |

---

## Try the Example

No API key required:

```bash
python examples/query_example.py
```

This builds a tiny 4-note vault in a temp directory, ingests it, and runs keyword searches.

---

## Embedding Backends

### OpenAI (default)
- Model: `text-embedding-3-small` (768 dimensions)
- Requires: `OPENAI_API_KEY`
- Cost: ~$0.02 per 1M tokens

### Ollama (free, offline)
- Model: `nomic-embed-text` (768 dimensions)
- Requires: [Ollama](https://ollama.ai) running locally
- Cost: free

---

## Extending to Multiple SQLite Chains

The problem statement envisions **chains of SQLite files by category**. You can achieve this by running separate ingests for different vault sub-folders:

```bash
python main.py ingest ~/Vault/AI      --db ai_knowledge.db
python main.py ingest ~/Vault/Finance --db finance_knowledge.db
python main.py ingest ~/Vault/Health  --db health_knowledge.db
```

Each `.db` file is independent and portable. A category router can decide which database(s) to query based on the question topic.

---

## MCP Integration (Claude Desktop)

To expose your knowledge base to Claude Desktop as a tool:

1. Install [gnosis-mcp](https://github.com/gnosis/gnosis-mcp) or write a custom MCP server that calls `src/search.py`.
2. Point it at your `vault_memory.db`.
3. Claude can then call `search_knowledge_base("your query")` directly during conversations.

---

## Relevant Projects & Further Reading

- [Andrej Karpathy's research](https://github.com/karpathy) — LLM architecture & memory insights
- [NousResearch Hermes Agent](https://github.com/nousresearch/hermes-agent) — Agent framework
- [sqlite-vec](https://github.com/asg017/sqlite-vec) — Vector search in SQLite
- [Obsidian Web Clipper](https://obsidian.md/clipper) — Add web pages to your vault
- [Ollama](https://ollama.ai) — Run LLMs and embedding models locally

---

## License

MIT
