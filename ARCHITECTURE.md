# LLM-Brains Architecture

> A personal second-brain and AI memory system combining Karpathy's LLM Wiki paradigm,
> SQLite RAG hybrid search, and a hive-mind fragment agent architecture.

---

## Core Thesis

**1-bit fragment models + Karpathy wiki + SQLite RAG = efficient hive mind**

The central insight: you don't need a large monolithic model to maintain a rich personal knowledge
base. Instead, distribute knowledge across 20 domain-specialist "fragment" agents, each maintaining
their own wiki (plain markdown files) and SQLite store. A thin orchestrator routes queries to
relevant fragments and synthesizes answers. The wiki files serve as persistent long-term memory —
the same role Karpathy's LLM wiki serves for knowledge management.

This system is:
- **Cheap**: Most queries run against local SQLite; Claude API only for compilation + synthesis
- **Persistent**: Everything stored in plain text + SQLite; survives model changes
- **Inspectable**: All knowledge is readable markdown; no black-box embeddings required
- **Extensible**: New fragment domains, new data sources, new output formats are plug-and-play

---

## Memory Pathway Repair Theory

Inspired by neuroscience: the brain forms redundant memory pathways through repeated association.
When one pathway degrades, others compensate.

In LLM-Brains, we implement this as:
1. **Redundant wikilinks**: Every key concept appears in multiple fragment wikis with `[[links]]`
2. **Cross-fragment indexing**: The SQLite FTS5 + vector search spans all fragments
3. **Studio outputs**: Flashcards, podcasts, and quizzes reinforce knowledge through different modalities
4. **Contradiction detection**: `cross_fragment_lint.py` actively finds and repairs inconsistencies

When a fragment's wiki is sparse or stale, the hybrid search finds the concept in other fragments.
The more a concept is cross-linked, the more "pathways" lead to it.

---

## System Architecture Diagram

```
                        ┌─────────────────────────────────┐
                        │         orchestrator.py          │
                        │    (Claude tool-calling loop)    │
                        └──────────────┬──────────────────┘
                                       │
               ┌───────────────────────┼───────────────────────┐
               │                       │                       │
        ┌──────▼──────┐        ┌───────▼──────┐      ┌────────▼──────┐
        │  search_wiki │        │query_fragment│      │  recall_memory │
        │ sqlite_rag.py│        │fragment_mgr  │      │  me/ directory │
        └──────┬──────┘        └───────┬──────┘      └───────────────┘
               │                       │
        ┌──────▼──────┐    ┌──────────▼──────────────────────────────┐
        │  memory.db   │    │          FragmentManager                │
        │ FTS5+vectors │    │   (ThreadPoolExecutor, 20 domains)      │
        └─────────────┘    └──┬─────────────────────────────────┬───┘
                              │                                 │
                     ┌────────▼────────┐              ┌────────▼────────┐
                     │ Fragment: ai_ml  │   ...×20...  │Fragment: people │
                     │ wiki/ + .db      │              │ wiki/ + .db     │
                     └─────────────────┘              └─────────────────┘

                              INPUT PIPELINE
                              ──────────────
    brain/raw/  ──►  compile.py  ──►  wiki_compiler.py  ──►  brain/knowledge/wiki/
        │                │                    │
        │          extract text          KEEP/UPDATE/
     .pdf .mp4     route content         MERGE/SUPERSEDE
     .jpg .wav     [[wikilinks]]         ARCHIVE ops
     .txt .md
        │
    rss_ingest.py ──► brain/raw/   (automatic hourly feed ingestion)
    wearable_ingest.py ──► brain/me/experiences/
    life_data_ingest.py ──► brain/me/{health,timeline,relationships}
    media_store.py ──► brain/media/ + media.db

                              OUTPUT PIPELINE
                              ───────────────
    brain/knowledge/wiki/  ──►  studio_generator.py  ──►  brain/studio/{slug}/
                                                              ├── podcast.md
                                                              ├── slides.md
                                                              ├── mindmap.md
                                                              ├── report.md
                                                              ├── flashcards.json
                                                              ├── quiz.json
                                                              ├── infographic.svg
                                                              └── datatable.csv

                              LINT + MAINTENANCE
                              ──────────────────
    cross_fragment_lint.py  ──►  lint_report.md
        ├── Contradiction detection (SPO triples)
        ├── Stale entry flagging (>30 days)
        ├── Orphan link detection
        └── Duplicate detection (>80% similarity)
```

---

## Karpathy LLM Wiki Paradigm vs Traditional RAG

| Traditional RAG | Karpathy Wiki + LLM-Brains |
|----------------|---------------------------|
| Retrieve chunks → inject as context | Maintain curated wiki articles → compile over time |
| Quality degrades with noise | Quality improves with each compilation |
| No persistent knowledge | Persistent markdown wiki survives model updates |
| Context window limited | Wiki can be arbitrarily large |
| Retrieval is ad-hoc | Knowledge is structured and cross-linked |
| No contradiction detection | Active lint + conflict markers |
| Duplicate information common | MERGE/SUPERSEDE operations prevent duplication |

The key insight from Karpathy's approach: **the compiler is the intelligence**, not the retriever.
By having Claude curate and merge knowledge (rather than just retrieve it), the wiki improves
over time instead of accumulating noise.

---

## Fragment Architecture

Each of the 20 fragment agents is domain-specialized:

| Domain | Covers |
|--------|--------|
| `geography` | Places, maps, travel, cities, countries |
| `people` | Public figures, historical persons |
| `science` | Physics, chemistry, biology, math |
| `technology` | Software, hardware, engineering |
| `history` | Historical events, timelines |
| `philosophy` | Ethics, logic, world-views |
| `health` | Medicine, fitness, nutrition |
| `creative` | Art, music, design, writing |
| `business` | Economics, finance, startups |
| `personal_memory` | Personal experiences, episodic memory |
| `ai_ml` | ML, neural networks, LLMs, AI research |
| `code` | Programming, algorithms, code snippets |
| `media` | Books, films, podcasts consumed |
| `relationships` | Personal relationships, social dynamics |
| `events` | Calendar, appointments, milestones |
| `concepts` | Abstract ideas, mental models |
| `emotions` | Emotional experiences, mood patterns |
| `skills` | Learned abilities, certifications |
| `projects` | Active/completed projects, goals |
| `misc` | Uncategorized notes |

**Fragment "1-bit awareness"**: Fragments are designed to work with minimal model parameters
by offloading as much as possible to their wiki files. The wiki is the "long-term weight" —
the model is just the reasoner that reads it.

**Parallel query execution**: `FragmentManager.route()` queries relevant fragments in parallel
using `ThreadPoolExecutor`, then synthesizes a combined answer with Claude.

---

## BitNet b1.58 + Tool-Calling Research Findings

BitNet b1.58 (1-bit quantization where weights are -1, 0, +1) enables:
- **8-10x memory reduction** compared to FP16
- **Near full-precision accuracy** at 3B+ parameter scale
- **CPU-feasible inference** for smaller fragment models

In the LLM-Brains context:
- Fragment agents could run locally as BitNet models for the "1-bit aware" design
- The wiki serves as the "extended context" that compensates for reduced model capacity
- Tool-calling (as implemented in `orchestrator.py`) allows a small model to access
  rich external knowledge without storing it in weights
- The orchestrator can run as a larger remote model (Claude) while fragments run locally

**Key insight**: A 1-bit 3B model + rich wiki ≈ a 13B model with no wiki,
at a fraction of the compute cost.

---

## SQLite as Universal Storage Layer

All data in LLM-Brains lives in SQLite:

| Database | Contents |
|----------|----------|
| `brain/memory.db` | All notes with FTS5 + embeddings |
| `brain/wiki.db` | Wiki entries with FTS5 + metadata |
| `brain/media.db` | Media items with transcripts + summaries |
| `brain/fragments/{domain}/{domain}.db` | Per-fragment entry stores with FTS5 |

**Why SQLite:**
- Zero-dependency, zero-config, zero-server
- FTS5 is a production-grade full-text search engine
- Works on any platform (including iOS/Android via SQLite)
- Entire brain fits in a single directory
- ACID guarantees without infrastructure
- Easy backup: `cp brain/memory.db backup/`

**Hybrid search architecture** (`sqlite_rag.py`):
```
Query → FTS5 keyword search → BM25 scores → RRF fusion ─┐
      └→ Vector similarity search → cosine scores ───────┘→ Reranked results
```

---

## Studio Output Generation

`studio_generator.py` implements NotebookLM-style multi-format output.
For every wiki entry, it generates 8 output types in parallel:

1. **Podcast script** — Host A/B dialogue for audio consumption
2. **Marp slides** — Presentation deck for visual learning
3. **Mermaid mindmap** — Visual concept map
4. **Research report** — Academic-style analysis with executive summary
5. **Flashcards** — 12-15 Q&A pairs for spaced repetition (JSON)
6. **Quiz** — 8-10 multiple choice questions (JSON)
7. **Infographic** — SVG-spec visual summary
8. **Data table** — CSV of quantitative facts

**Memory pathway reinforcement**: Each format engages different learning modalities.
Podcast → auditory. Slides → visual. Flashcards → active recall. Quiz → testing effect.
Multiple formats = multiple redundant memory pathways.

---

## Data Source Map

```
PERSONAL DATA SOURCES
─────────────────────
Wearable cameras (Meta Glasses)   → wearable_ingest.py → brain/me/experiences/
Apple Health XML                  → life_data_ingest.py → brain/me/health.md
Google Location History JSON      → life_data_ingest.py → brain/me/locations.md
iMessage exports                  → life_data_ingest.py → brain/me/relationships.md
Calendar (.ics)                   → life_data_ingest.py → brain/me/timeline.md
Contacts (.vcf)                   → life_data_ingest.py → brain/knowledge/people/
Browser history                   → life_data_ingest.py → brain/me/interests.md
Spotify history JSON              → life_data_ingest.py → brain/me/music.md
Email (.mbox/.eml)                → life_data_ingest.py → brain/me/relationships.md
GPS tracks (.gpx)                 → life_data_ingest.py → brain/me/locations.md

EXTERNAL KNOWLEDGE SOURCES
──────────────────────────
RSS feeds (arXiv, HF, blogs)      → rss_ingest.py → brain/raw/ → compile.py
YouTube / video URLs              → media_store.py → brain/media/
Twitter / Vimeo / etc.            → media_store.py → brain/media/
PDF documents                     → compile.py → brain/knowledge/
Images / photos                   → compile.py (Claude vision) → brain/me/ or brain/knowledge/
Audio files                       → compile.py (Whisper) → brain/knowledge/
Plain text / markdown notes       → compile.py → routed automatically
```

---

## Cross-Fragment Lint System

`cross_fragment_lint.py` runs automated consistency checks:

### Check 1: Contradiction Detection
Uses Claude to extract Subject-Predicate-Object triples from each wiki entry.
Flags cases where `subject::predicate` maps to different objects across fragments.

Example conflict:
```
[ERROR] Conflicting claim: 'GPT-4' was released by 'OpenAI' vs 'Microsoft'
  Files: ai_ml/gpt-4.md, technology/gpt-4.md
```

### Check 2: Stale Entries
Flags wiki files not modified in >30 days (configurable). Old knowledge may
be superseded by newer findings.

### Check 3: Orphan Links
Finds `[[wikilinks]]` in wiki files that don't correspond to any known entry.
Indicates missing knowledge or renamed concepts.

### Check 4: Duplicate Detection
Uses Jaccard similarity to find entries with >80% token overlap across fragments.
Suggests merging to reduce redundancy.

### Auto-fix mode
`--fix` flag automatically:
- Archives older duplicate (keeps newer)
- Removes dead `[[links]]` (replaces with plain text)

---

## How to Run Everything

### Initial Setup
```bash
cp .env.example .env
# Edit .env with your API keys
pip install -r requirements.txt
```

### Ingest RSS Feeds
```bash
python rss_ingest.py          # one-shot fetch
python rss_ingest.py --watch  # hourly auto-fetch
```

### Compile Raw Notes
```bash
# Drop files in brain/raw/ then:
python compile.py             # one-shot process
python compile.py --watch     # continuous watch mode
```

### Search Your Brain
```bash
python sqlite_rag.py --index brain/
python sqlite_rag.py --search "transformer attention mechanism"
```

### Interactive Orchestrator
```bash
python orchestrator.py                    # REPL
python orchestrator.py --query "What do I know about diffusion models?"
```

### Wiki Compiler
```bash
python wiki_compiler.py --compile brain/raw/
python wiki_compiler.py --list
python wiki_compiler.py --search "neural networks"
```

### Fragment Manager
```bash
python fragment_manager.py --status
python fragment_manager.py --query "What is attention mechanism?"
python fragment_manager.py --lint
```

### Generate Studio Outputs
```bash
python studio_generator.py --all
python studio_generator.py --entry brain/knowledge/wiki/my-topic.md
python studio_generator.py --entry file.md --formats podcast,flashcards,quiz
```

### Ingest Personal Life Data
```bash
python life_data_ingest.py --apple-health export.xml
python life_data_ingest.py --calendar calendar.ics
python life_data_ingest.py --all ~/Downloads/GoogleTakeout/
```

### Wearable Camera Ingest
```bash
python wearable_ingest.py --watch ~/Documents/MetaGlasses/ --interval 60
python wearable_ingest.py --file video.mp4
```

### Download and Index Media
```bash
python media_store.py --url "https://youtube.com/watch?v=..."
python media_store.py --recall "transformers architecture"
```

### Run Lint
```bash
python cross_fragment_lint.py               # check only
python cross_fragment_lint.py --fix         # auto-fix
python cross_fragment_lint.py --output report.md
```

### Full Pipeline (src module)
```python
from src.pipeline import Pipeline

p = Pipeline(enable_studio=True, studio_formats=["podcast", "flashcards"])
result = p.run()
print(result.summary())
```

---

## Directory Structure

```
LLM-Brains/
├── compile.py              Main file compiler + watcher
├── sqlite_rag.py           Hybrid FTS5 + vector search
├── wiki_compiler.py        Karpathy-style wiki compiler
├── fragment_manager.py     20-domain hive-mind fragment system
├── cross_fragment_lint.py  Contradiction + consistency checker
├── orchestrator.py         Claude tool-calling orchestrator REPL
├── wearable_ingest.py      Meta Glasses / wearable camera ingest
├── media_store.py          yt-dlp + Whisper + vision media indexer
├── rss_ingest.py           RSS/Atom feed fetcher
├── studio_generator.py     NotebookLM-style multi-format generator
├── life_data_ingest.py     Personal data export ingestor
├── requirements.txt
├── .env.example
├── ARCHITECTURE.md         (this file)
│
├── src/                    Reusable library modules
│   ├── __init__.py
│   ├── compiler.py         Compiler class (core compile logic)
│   ├── db_manager.py       DatabaseManager (SQLite CRUD + migrations)
│   ├── embeddings.py       EmbeddingEngine (OpenAI / TF-IDF fallback)
│   ├── search.py           SearchEngine (hybrid FTS5 + vector + MMR rerank)
│   ├── vault_parser.py     VaultParser (Obsidian vault parsing)
│   └── pipeline.py         Pipeline (full ingest→compile→index→generate)
│
└── brain/                  Your personal knowledge base
    ├── raw/                Drop files here for processing
    │   └── processed/      Processed files archived here
    ├── me/                 Personal life data
    │   ├── identity.md
    │   ├── timeline.md
    │   ├── health.md
    │   ├── relationships.md
    │   └── experiences/    Wearable camera episodic memories
    ├── knowledge/          Research and learned knowledge
    │   └── wiki/           Compiled wiki articles
    ├── work/               Professional activities
    ├── media/              Downloaded and indexed media
    ├── studio/             Generated outputs (podcast, slides, etc.)
    ├── fragments/          Domain fragment agent stores
    │   └── {domain}/
    │       ├── wiki/       Domain-specific wiki files
    │       └── {domain}.db Fragment SQLite store
    ├── memory.db           Master search index (FTS5 + embeddings)
    ├── wiki.db             Wiki entries database
    ├── media.db            Media items database
    ├── index.md            Master index (auto-updated)
    ├── rss_state.json      RSS fetch state
    └── compile.log         Compilation log
```

---

## Philosophy

> "The best knowledge management system is one you actually use."

LLM-Brains is designed around four principles:

1. **Plain text first**: All knowledge lives in markdown files. No vendor lock-in.
2. **Incremental improvement**: Each ingest makes the wiki better, not noisier.
3. **Redundant pathways**: Cross-links + multiple output formats = robust recall.
4. **Cheap at rest, smart at compile**: SQLite is free to query; Claude is only invoked for compilation.

The system is intentionally over-engineered for a personal tool — because the complexity
is in the _structure_, not the _code_, and structure is what makes knowledge retrievable
decades from now.

---

*LLM-Brains — built for lifelong knowledge accumulation.*
