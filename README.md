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
