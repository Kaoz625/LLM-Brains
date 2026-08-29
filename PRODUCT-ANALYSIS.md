# LLM-Brains — Product Analysis

## What Makes It Unique

LLM-Brains implements Karpathy's "LLM as compiler" paradigm: ingest raw content once, compile it into a structured wiki, query the wiki — not the raw docs. This is fundamentally different from RAG.

**Key differentiator:** Knowledge compounds. Each new source merges into existing structure. The system gets smarter with every file dropped in. Traditional RAG re-derives the same facts on every query; LLM-Brains accumulates them permanently.

**Unique capabilities found in codebase:**
- `dream_cycle.py` — autonomous background consolidation (LLM reflects on its own knowledge while idle)
- `fragment_manager.py` — hive-mind fragment agents (specialized sub-wikis per topic)
- `cross_fragment_lint.py` — cross-wiki consistency checking
- `life_data_ingest.py` — wearable/life-data ingestion (Meta glasses video → episodic memory)
- `mcp_server.py` — exposed as MCP tool (Claude Code can query it directly)
- `notebooklm_sync.py` — exports to NotebookLM for audio overviews
- `notion_cleanup.py` — Notion integration

No competitor has all of these. Most have 1-2.

---

## Competitive Landscape

| Product | Model | What they do | What LLM-Brains does better |
|---------|-------|-------------|----------------------------|
| **Mem.ai** | B2C SaaS, $8-20/mo | AI note-taking, vector search over notes | LLM compiles knowledge (not just stores); wearable ingestion; open/self-hosted |
| **Rewind.ai** | B2C SaaS, $20/mo | Records your screen + audio, searchable | LLM-Brains compiles understanding, not just records events; multi-source |
| **NotebookLM** | Free (Google) | RAG over uploaded docs, audio overviews | LLM-Brains is always-learning, multi-source, self-hosted; NotebookLM is per-project |
| **Obsidian + Copilot** | Plugin ecosystem | Chat with your vault | LLM-Brains compiles the vault into a wiki first; not just Q&A over raw notes |
| **MemGPT / Letta** | Open source | Agent with tiered memory (in-context, archival) | LLM-Brains has richer ingest pipeline; domain-specific wiki structure |
| **Cognee** | Open source | Knowledge graph from documents | LLM-Brains has wearable/video/audio pipeline; dream cycle; MCP server |
| **Khoj** | Open source | Personal AI with memory | Lacks dream cycle, fragment agents, cross-wiki linting |

**LLM-Brains' actual moat:**
1. Wearable video → episodic memory pipeline (unique)
2. Dream cycle — autonomous reflection (unique)
3. Fragment agents — hive-mind sub-wikis (unique)
4. MCP server — direct Claude Code integration (unique)
5. Fully self-hosted, no subscription, no data leaving your machine

---

## Top 5 Improvements for Production-Readiness

### 1. Web UI / Dashboard
**Gap:** No UI — CLI only. Real product needs a UI.
**Build:** Lightweight Flask/FastAPI + HTMX dashboard showing:
- Brain stats (articles, word count, last compiled)
- Recent additions (timeline feed)
- Search interface
- Dream cycle status
- Fragment health

Estimated: 3-5 days. Highest ROI for making it feel like a product.

### 2. One-Command Install + Auto-Ingest
**Gap:** Setup requires reading CLAUDE.md and running multiple commands.
**Build:** Single `llm-brains install` command that:
- Creates brain/ structure
- Sets up launchd agents (ingest + compile + dream)
- Configures API key once
- Drops a watched folder in ~/Desktop for easy adds

### 3. Multi-Model Support (local-first)
**Gap:** Requires OpenAI API key for embeddings + compilation.
**Build:** Add Ollama backend option for both compilation and embeddings. This makes it:
- Fully offline-capable
- Zero ongoing cost
- Privacy-complete (nothing leaves the machine)

Use `nomic-embed-text` via Ollama for embeddings (384-dim, fast on Intel Mac GPU).

### 4. Sharing / Export Layer
**Gap:** Knowledge is locked to one machine.
**Build:**
- `llm-brains export --format obsidian|notion|html|pdf`
- Read-only web share link (static HTML export)
- NotebookLM sync already exists — expose as scheduled task

### 5. Collaboration Mode
**Gap:** Single-user only.
**Build:** Multi-user brain (family, team) via Supabase backend sync:
- Each user has own `brain/me/` partition
- Shared `brain/knowledge/` and `brain/work/` merge across users
- Conflict resolution via LLM (same as semantic dedup)

---

## Monetization Angles

### B2C Personal — Strongest near-term
- "Your AI second brain" — target knowledge workers, researchers, students
- Self-hosted = privacy story (huge differentiator vs Rewind/Mem.ai)
- Pricing: free (self-hosted) + $10/mo hosted version with managed compilation
- Wearable pipeline is the hook — nothing else does Meta glasses → searchable memory

### B2B Knowledge Management — Medium term
- Teams using Obsidian/Notion → LLM-Brains as the intelligence layer on top
- Integrates with existing tools (already has Notion sync)
- Target: research orgs, law firms, consulting teams
- Pricing: $50-200/mo per team seat

### Developer Tool / API — Long term
- The MCP server already exists — position as "memory layer for AI agents"
- Agents that need persistent, compiling memory plug into LLM-Brains via MCP
- Pricing: API calls at $0.001/query after free tier

### Fastest path to revenue: B2C hosted
- Ship the web UI
- One-click deploy to Coolify (self-hosted) or managed cloud
- Launch on Product Hunt with the wearable angle as the hook

---

## Patterns/Tools to Incorporate

| Tool | What it adds | Priority |
|------|-------------|----------|
| **PageIndex** | Zero-vector RAG for PDF/structured docs — plug into query layer | High (already in tools/) |
| **Cognee patterns** | Graph-based knowledge relationships on top of existing wiki | Medium |
| **Letta/MemGPT** | Tiered memory model for very large knowledge bases | Low |
| **Nomic Embed** | Local embeddings via Ollama (nomic-embed-text) | High |
| **Prefect** | Durable scheduled ingest jobs (already installed) | High |
| **CrewAI** | Multi-agent compilation (multiple fragment agents in parallel) | Medium |

---

## Summary

LLM-Brains is more sophisticated than any open-source competitor. The wearable pipeline, dream cycle, and fragment system are genuinely novel. The gap is product polish — no UI, complex setup, single-user.

**Recommended next steps:**
1. Web dashboard (Flask + HTMX, 3 days)
2. One-command install script
3. Ollama backend for zero-cost offline mode
4. Product Hunt launch with wearable angle as hook
