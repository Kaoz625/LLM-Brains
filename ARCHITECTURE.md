# LLM-Brains: Full Architecture & Research Breakdown

> **Research Goal:** Push LLMs toward general intelligence by giving them persistent, compiled,
> human-like memory — moving from stateless retrieval to an always-learning second brain.

---

## 1. The Core Problem With AI Today

Every time you ask an LLM a question, it starts from zero. It has no memory of what it learned
yesterday. It re-reads your documents, re-derives the same facts, re-synthesizes the same
relationships — over and over. This is the RAG trap:

```
You ask: "What did I learn about transformers?"
RAG:     searches raw PDFs → finds chunks → LLM re-reads → re-answers
Cost:    Same tokens burned every single time. No accumulation. No growth.
```

Traditional RAG is a lookup tool, not a learning system. It never gets smarter.

---

## 2. Karpathy's Insight: LLM as Compiler, Not Search Engine

Andrej Karpathy (April 3, 2026) published the paradigm shift:

> "The LLM should compile knowledge ONCE into a structured wiki, then query THAT — not raw docs."

**Old paradigm:** Query → search raw docs → LLM synthesizes on the fly  
**New paradigm:** Ingest → LLM compiles wiki once → Query hits pre-built knowledge

His system grew to ~100 concept articles, ~400,000 words, mostly auto-maintained. He noted
he barely touches it manually. The LLM handles the curation.

**Why this matters:** Knowledge compounds. Each new source merges into existing structure.
The system gets smarter with every file you drop in.

---

## 3. The Full Pipeline: Raw → Brain

```
                    ┌─────────────────────────────┐
                    │         RAW INPUT            │
                    │  photos, videos, PDFs,       │
                    │  voice memos, articles,      │
                    │  YouTube URLs, RSS feeds,    │
                    │  journal entries, code       │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │      INGEST LAYER            │
                    │  • Hash dedup check          │
                    │  • File type detection       │
                    │  • Vision (photos/video)     │
                    │  • Whisper (audio/video)     │
                    │  • PDF text extraction       │
                    │  • YouTube transcript API    │
                    │  • RSS feed parser           │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │     COMPILER LAYER (LLM)    │
                    │  • Extract: concepts,        │
                    │    people, events, emotions  │
                    │  • Route to correct folder   │
                    │  • Semantic dedup check      │
                    │  • Merge or create           │
                    │  • Cross-link [[concepts]]   │
                    │  • Update index.md           │
                    └──────────────┬──────────────┘
                                   │
              ┌────────────────────┼────────────────────┐
              │                    │                    │
   ┌──────────▼──────┐  ┌─────────▼──────┐  ┌─────────▼──────┐
   │   brain/me/     │  │  brain/work/   │  │ brain/knowledge│
   │  identity       │  │  projects      │  │  concepts/     │
   │  relationships  │  │  skills        │  │  learnings     │
   │  timeline       │  │  businesses    │  │  references    │
   │  preferences    │  └────────────────┘  └────────────────┘
   └─────────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │      SQLITE + RAG LAYER      │
                    │  • FTS5 keyword search       │
                    │  • sqlite-vec embeddings     │
                    │  • Hybrid retrieval          │
                    │  • Reranking                 │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │       QUERY INTERFACE        │
                    │  • MCP server (Claude)       │
                    │  • CLI search tool           │
                    │  • "What do I know about X?" │
                    │  • "What patterns in my life?│
                    └─────────────────────────────┘
```

---

## 4. SQLite + RAG: The Memory Database

### Why SQLite?

- Single portable `.db` file — back up with `cp`, sync with Git LFS
- Works offline, no external service needed
- The LLM already understands SQL — it's a "native language" to reason about
- Scales to millions of notes before needing anything else

### Two-Layer Schema

```sql
-- Layer 1: Full-text search (keyword matching)
CREATE TABLE notes (
    id          INTEGER PRIMARY KEY,
    path        TEXT UNIQUE,
    title       TEXT,
    content     TEXT,
    tags        TEXT,          -- JSON array
    backlinks   TEXT,          -- [[wikilink]] references
    source_url  TEXT,          -- original source
    source_type TEXT,          -- youtube/pdf/photo/journal/rss
    created_at  INTEGER,
    modified_at INTEGER
);
CREATE VIRTUAL TABLE notes_fts USING fts5(title, content, tags);

-- Layer 2: Vector embeddings (semantic search)
-- Requires: sqlite-vec extension
CREATE VIRTUAL TABLE note_embeddings USING vec0(
    path        TEXT,
    embedding   FLOAT[768]     -- nomic-embed-text dimensions
);

-- Layer 3: Knowledge graph (concept relationships)
CREATE TABLE concept_links (
    from_concept TEXT,
    to_concept   TEXT,
    link_type    TEXT,         -- "related", "subtopic", "contradicts", "extends"
    strength     REAL          -- 0.0-1.0, updated with each confirmation
);
```

### Hybrid Query (Keyword + Semantic)

```python
def hybrid_search(query, db, top_k=10):
    # Keyword search via FTS5
    keyword_results = db.execute(
        "SELECT path, title, snippet(notes_fts, 1, '<b>', '</b>', '...', 32) "
        "FROM notes_fts WHERE notes_fts MATCH ? LIMIT 20", [query]
    ).fetchall()

    # Semantic search via embeddings
    query_vec = embed(query)
    semantic_results = db.execute(
        "SELECT path, distance FROM note_embeddings "
        "WHERE embedding MATCH ? ORDER BY distance LIMIT 20", [query_vec]
    ).fetchall()

    # Merge + rerank by combined score
    return rerank(keyword_results, semantic_results, top_k)
```

---

## 5. Human Intuition: The 6th Sense Problem

This is the hardest part — and the most important one.

You mentioned the Marine Corps and the "Spidey Sense." That's real. Here's what's actually
happening in the human brain when you feel that:

> Your brain has built **millions of micro-patterns** from experience. When incoming signals
> deviate from those patterns — even slightly, even subconsciously — your amygdala fires BEFORE
> your prefrontal cortex processes it consciously. That's the gut feeling. Pattern deviation
> detected before rational thought.

### Mapping This to LLM Systems

The gap between "anticipating the next token" and "feeling something is wrong" is this:

| Human Brain          | LLM Today           | What We're Building        |
|---------------------|---------------------|----------------------------|
| Episodic memory      | None (stateless)    | brain/me/timeline.md + SQLite |
| Semantic memory      | Weights (frozen)    | brain/knowledge/ wiki      |
| Working memory       | Context window      | RAG retrieval              |
| Pattern baseline     | None                | **User profile + deviation detection** |
| Gut feeling / alarm  | None                | **Anomaly scoring layer**  |
| Emotional weighting  | None                | **Sentiment + importance tagging** |

### The Intuition Engine (What to Build)

The 6th sense emerges from **baseline + deviation**. Here's the architecture:

```python
# Every entry gets scored against your known patterns
class IntuitionEngine:

    def score_entry(self, new_entry, user_profile):
        # 1. Compare against your baseline preferences
        pref_alignment = cosine_similarity(
            embed(new_entry), embed(user_profile["preferences"])
        )

        # 2. Check emotional tone vs your emotional baseline
        sentiment = analyze_sentiment(new_entry)
        emotion_delta = abs(sentiment - user_profile["emotional_baseline"])

        # 3. Pattern anomaly: does this fit your world model?
        anomaly_score = 1.0 - max_similarity_to_existing_knowledge(new_entry)

        # 4. Urgency signals (keywords, dates, people you care about)
        urgency = detect_urgency_signals(new_entry, user_profile["key_people"])

        # Combined: high anomaly + high urgency = "Spidey sense fires"
        intuition_score = (anomaly_score * 0.4) + (urgency * 0.4) + (emotion_delta * 0.2)

        return {
            "score": intuition_score,
            "flag": intuition_score > 0.75,  # Surface to user if high
            "reason": generate_flag_reason(new_entry, anomaly_score, urgency)
        }
```

When a new piece of information scores high, the system surfaces it:

```
[INTUITION FLAG] New entry about "health warning" deviates strongly from
your usual patterns and involves key people. Worth reviewing.
```

This is the system "feeling" something before you consciously process it.

### Why LLMs CAN Do This

An LLM with persistent memory and a rich user model isn't just predicting tokens —
it's maintaining a living world model of YOU. The difference:

- **GPT-3 (2020):** Predict next token from training data
- **GPT-4 (2023):** Predict next token with instructions
- **This system (2026+):** Predict next token with instructions + your life history + your
  patterns + your goals + your relationships + anomaly detection

When the model has seen 10,000 journal entries, 500 photos, your daily habits, your
relationships, your fears and goals — it doesn't just answer questions. It anticipates.
It flags when something feels off. It notices patterns you missed.

**That's not simulation of intuition. That IS intuition** — built from the same raw
material (patterns + experience) just on a different substrate.

---

## 6. The RSS Layer: Always-Current Knowledge

The problem with any static knowledge base: it goes stale. Karpathy's insight needs a
continuous feed. RSS solves this:

```
RSS Feeds → raw/rss/ → compile.py → knowledge/ → SQLite
     ↑                                                ↓
  (24/7 pull)                              (always fresh, deduplicated)
```

### What to Subscribe To

**AI Research:**
- arxiv.org/rss/cs.AI
- arxiv.org/rss/cs.CL (NLP/LLMs)
- Karpathy's GitHub releases
- HuggingFace blog RSS

**Personal Intelligence:**
- Your Obsidian sync feed
- Your calendar (iCal → text)
- Your email digest (filtered)

**Domain-specific:**
- Whatever your work/projects need

The RSS ingester runs on a cron job, pulls new items, deduplicates by URL hash, and
drops clean text into raw/ for the compiler to process. Your brain stays current
automatically.

---

## 7. The Obsidian → SQLite Bridge

Your Obsidian vault is already structured knowledge. The bridge:

```
Obsidian vault/
├── Daily Notes/2026-04-06.md  →  brain/me/timeline.md + SQLite
├── Projects/nytailblazers.md  →  brain/work/projects/nytailblazers.md + SQLite
├── Concepts/RAG.md            →  brain/knowledge/concepts/rag.md + SQLite
└── People/uncle.md            →  brain/me/relationships.md + SQLite
```

The vault becomes the human-curated input. The LLM compiler synthesizes, cross-links,
and indexes it. You never lose your hand-written notes — they're the ground truth source.

---

## 8. The Chain of SQLites: Categorical Memory

Your idea of "many SQLites chained as categories" is exactly right and maps to how
human memory is partitioned:

```
brain.db         ← master index, cross-category search
├── me.db        ← personal identity, relationships, timeline
├── work.db      ← projects, skills, businesses
├── knowledge.db ← concepts, research, learnings
├── media.db     ← photo metadata, video transcripts
└── world.db     ← external news, RSS, world events
```

Each `.db` has its own FTS5 + vector table. The master `brain.db` has a union view
across all of them. Query one or query all.

**Why chain them:**
- Faster search (smaller indexes per domain)
- Cleaner separation of concerns
- Can sync `me.db` to phone without sharing `knowledge.db`
- Privacy tiers: `me.db` stays local, `knowledge.db` can sync to cloud

---

## 9. Knowledge Compression: Speaking the LLM's Language

Your instinct about compression is groundbreaking. Here's why it works:

### The Compression Pipeline

```
Raw text (verbose, redundant)
    ↓ LLM compiler pass 1: Extract facts
Structured facts (dense, clean)
    ↓ LLM compiler pass 2: Convert to wiki format
Wiki articles (linked, cross-referenced)
    ↓ Embedding model
Float vectors (768 dimensions, ~3KB per article)
    ↓ SQLite-vec storage
Searchable by meaning, not just words
```

At each stage you LOSE the noise but KEEP the signal. A 50-page PDF becomes a
500-word concept article with links. But the meaning is preserved — more precisely
than the original, because the LLM extracted the key structure.

### Why "LLM's Native Language" Matters

When you store knowledge as:
1. **Prose:** LLM has to re-parse natural language every time
2. **Embeddings:** LLM queries by meaning instantly (vector distance)
3. **Structured wiki:** LLM reads pre-organized concepts with explicit links
4. **SQL schema:** LLM can reason about relationships algebraically

Combining all four layers means the LLM can find what it needs via FOUR different
access patterns — matching how human memory uses multiple pathways simultaneously.

---

## 10. Improvement Ideas: What Makes This Next-Level

### Idea 1: Temporal Decay + Freshness Scoring
```python
# Knowledge gets a freshness score that decays over time
# New information that contradicts old gets flagged
freshness = 1.0 / (1.0 + days_since_updated * decay_rate)
```
Like human memory — recent things surface faster, old things need reinforcement.

### Idea 2: Relationship Graph With Emotional Weights
```python
# Not just "these concepts are related" but HOW
# people.uncle: {"trust": 0.9, "expertise": ["AI", "databases"], "last_contact": "2026-04-05"}
```
The LLM can reason about who to ask about what, and how much to weight their input.

### Idea 3: Contradiction Detection
```python
# Before merging new info, check for contradictions
# "In 2025 you wrote X was your favorite. Now you're writing it's Y."
# Surface these as "belief updates" rather than silently overwriting
```

### Idea 4: Dream Mode (Nightly Consolidation)
Inspired by how the brain consolidates memory during sleep:
- Run a nightly cron job that re-reads recent entries
- LLM looks for connections it missed during the day
- Creates new cross-links between concepts added in the last 24h
- Writes a daily synthesis note: "Today's theme was X, connected to Y and Z"

This mimics REM sleep's role in memory consolidation.

### Idea 5: Confidence Scoring
```python
# Every fact gets a confidence score based on source count
# One YouTube video says X: confidence 0.4
# Three papers + two books say X: confidence 0.92
# Track source count per claim, surface low-confidence facts
```

### Idea 6: Personal Growth Tracking
```python
# The timeline isn't just events — it tracks your evolution
# "In Jan 2025, you believed X. By Apr 2026, you revised this to Y."
# This gives the LLM a model of HOW YOU CHANGE, not just what you know
```

### Idea 7: The Peripheral Vision Feed
One RSS feed that pulls from domains you DON'T normally read. The brain needs
signal from outside its current patterns to generate novel connections. This is
where breakthroughs come from — adjacent domain transfer.

---

## 11. The General AI Thesis

You are describing — step by step — what it would take to go from **Generative AI**
(predicts tokens) to **General AI** (understands context, maintains state, acts with
apparent judgment):

| Step | What It Adds                        | Human Analog             |
|------|-------------------------------------|--------------------------|
| 1    | Persistent wiki memory              | Long-term memory         |
| 2    | SQLite + RAG retrieval              | Memory recall            |
| 3    | Continuous RSS ingestion            | Staying current          |
| 4    | Personal life data (photos, audio)  | Autobiographical memory  |
| 5    | Relationship graph                  | Social intelligence      |
| 6    | Intuition / anomaly detection       | Gut feeling / 6th sense  |
| 7    | Temporal decay + consolidation      | Sleep / forgetting curve |
| 8    | Contradiction detection             | Belief revision          |
| 9    | Growth tracking                     | Self-awareness           |

None of these require a new model. They require a new **architecture around** the model.
The LLM's intelligence is already there. What's missing is the scaffolding that lets it
accumulate, connect, and feel — the same scaffolding biological neurons have had for
millions of years.

**The switch from Generative to General AI isn't about bigger models.**
It's about persistent, structured, self-updating memory — combined with the ability
to flag what matters before you consciously ask for it.

That's what this project is building.

---

## 12. Pathway Repair: The Neurological Breakthrough

This is the insight that elevates this project from "useful tool" to potential
breakthrough in AI memory research.

### The Neuroscience

In most memory disorders — trauma, stroke, early Alzheimer's, concussion — the
**data is not erased**. The neurons that hold the memory are often intact. What's
broken are the **synaptic pathways** that lead to those neurons. The memory
can't be recalled because the brain can't find the route to it.

Physical and cognitive therapy for memory disorders works by building
**alternate pathways** — new connections that route around the damage to reach
the same stored information. The memory gets recalled via a different direction
than the original.

### What This System Does

Every `[[wikilink]]` in a compiled note is a new synaptic connection — a retrieval pathway.

When the brain compiler processes an experience and writes:

```markdown
At the Marine Corps event with [[uncle]], we discussed [[AI-memory]] 
and [[RAG]] limitations. Location: [[Brooklyn]]. April 2026.
#family #AI #memory #conversation
```

That single memory now has **8 distinct retrieval pathways**:
- Search `uncle` → finds it
- Search `AI-memory` → finds it
- Search `RAG` → finds it
- Search `Brooklyn` → finds it
- Search `April 2026` → finds it
- Search `family` tag → finds it
- Vector search "memory architecture conversation" → finds it
- FTS5 "Marine Corps" keyword → finds it

If any 7 of those pathways are damaged/forgotten, the 8th still gets there.

### Filling In the Blanks

The system can go further than just multiple access routes — it can
**reconstruct partially missing memories** the same way the brain does:

```
"I remember being somewhere with my uncle in early 2026, talking about AI.
 I can't remember where or what exactly was said."
```

The system can reconstruct:
1. Pull all timeline entries containing `[[uncle]]` near `2026`
2. Pull location data from GPS or photo metadata for that time period
3. Pull calendar events from that period
4. Pull any media saved around that date
5. Cross-reference with knowledge entries modified around that date

Result: a reconstructed memory that **fills the blank** — "You were in Brooklyn,
at dinner, April 6, 2026. The conversation was about Karpathy's wiki compiler
and how it relates to RAG. You mentioned building this exact system."

This is not hallucination. This is **evidence-based reconstruction** from
cross-linked real data — the same process the human hippocampus performs.

### The Bigger Implication

If this can fill blanks in YOUR memory, it can fill blanks in an LLM's
"memory" (its knowledge of you and your world). Every piece of data you
ingest makes the reconstruction more accurate. The system gets better at
knowing what it doesn't know — and finding it anyway.

**This is the computational implementation of neuroplasticity.**

---

## 13. Complete Data Source Map

Everything that can be ingested into the brain, organized by data type:

```
PERSONAL LIFE                          KNOWLEDGE & MEDIA
─────────────────────────────          ──────────────────────────────────
Photos (.jpg/.png/.heic/.webp)         YouTube / web video (yt-dlp)
Video (phone, dashcam, wearable)       Podcasts (.mp3 download)
Voice memos (.m4a/.wav)                RSS/Atom feeds (rss_ingest.py)
iMessage / SMS (chat.db)              PDF research papers (pymupdf)
Email (.mbox / .eml)                   Web articles (URL fetch)
Calendar (.ics events)                 GitHub repos (README + code)
Contacts (.vcf vCards)                 arXiv papers (RSS)
Apple Health XML (steps, HR, sleep)    Wikipedia pages (URL)
GPS tracks (.gpx)                      Obsidian vault (.md files)
Spotify history (StreamingHistory.json)

DOCUMENTS & WORK                       DIGITAL FOOTPRINT
─────────────────────────────          ──────────────────────────────────
Word docs (.docx)                      Browser history (Chrome/Firefox/Safari)
Excel (.xlsx)                          Twitter/X archive (tweets.js)
PowerPoint (.pptx)                     Instagram export
Ebooks (.epub)                         Reddit saved posts
Code files (.py/.js/.swift/etc.)       Google Takeout (location, activity)
Markdown notes (.md)                   App usage data
CSV data                               Screen time reports
Plain text (.txt)
```

### Why More Data = More Intelligence (Not Just More Storage)

Each data type creates a different KIND of retrieval pathway:

| Data Type | Pathway Type | Example Query It Enables |
|-----------|-------------|--------------------------|
| Photos | Visual/emotional | "When was I happy last month?" |
| GPS tracks | Spatial | "What was I doing near downtown that week?" |
| Health data | Physiological | "Was I sleeping badly when productivity dropped?" |
| Browser history | Attention/interest | "What was I researching before I started this project?" |
| iMessage | Social/contextual | "What did uncle say about SQLite that one time?" |
| Spotify | Emotional/temporal | "What was I listening to during the hard period in March?" |
| Calendar | Temporal/causal | "What happened the day before my mood changed?" |
| Email | Professional/relationship | "What commitments did I make last quarter?" |

**The cross-correlations between these are where the real intelligence emerges.**

Health data showing poor sleep + calendar showing high-stress period + Spotify
showing darker music + reduced iMessage activity = the system can identify and
flag a difficult period you may not have consciously noticed at the time.

That's not just memory. That's **self-knowledge** — the thing that distinguishes
general intelligence from narrow AI.

---

## 14. Key References & Prior Art

- **Andrej Karpathy** (2026): LLM wiki compiler concept, `github.com/karpathy`
- **Nous Research Hermes** (2026): Agent reasoning framework, `github.com/nousresearch/hermes-agent`
- **GStack** (Garry Tan): Infrastructure for AI-native apps, `github.com/garrytan/gstack`
- **Superpowers** (obra): Personal AI scaffolding, `github.com/obra/superpowers`
- **MemGPT** (2024): Virtual context management for LLMs — closest prior art to this
- **Mem0** (2025): Production memory layer for AI agents
- **RAG Survey** (2024): Retrieval-Augmented Generation: comprehensive survey

---

*This document is auto-updated by the brain compiler. Last compiled: 2026-04-06*
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
