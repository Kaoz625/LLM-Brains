# LLM-Brains + knowledge-pipeline Merge Plan

## Current State (2026-05-06)

Two separate RAG/knowledge systems running independently:

| System | Location | Stack | Purpose |
|--------|----------|-------|---------|
| LLM-Brains | `Projects/tools/LLM-Brains/` | Python, SQLite FTS5 + vectors, Notion | Skool community scraping → brain wiki → queryable DB |
| knowledge-pipeline | `Projects/tools/knowledge-pipeline/` | Python, Supabase, CLI | Generic AI content pipeline with pgvector |

## Why Merge

- Duplicate embedding + search logic
- Two separate DBs for what is logically one knowledge base
- knowledge-pipeline has better CLI and Supabase integration; LLM-Brains has richer ingest sources

## Merge Strategy: LLM-Brains as Primary

Keep LLM-Brains as the canonical system. Absorb knowledge-pipeline's strengths.

### Phase 1 — Extract from knowledge-pipeline (1 day)
1. Copy `knowledge-pipeline/pipeline/` ingest modules → `LLM-Brains/src/pipeline/`
2. Copy `knowledge-pipeline/cli.py` → review for features not in `LLM-Brains/main.py`
3. Copy `setup_supabase.sql` → `LLM-Brains/db/supabase_schema.sql` (keep as optional backend)

### Phase 2 — Add Supabase as optional vector backend (2 days)
- `LLM-Brains/sqlite_rag.py` currently uses local SQLite only
- Add a `--backend supabase` flag that routes vector writes to Supabase pgvector
- Default remains SQLite (offline, no cost)
- Supabase used when `SUPABASE_URL` + `SUPABASE_KEY` env vars are set

### Phase 3 — Unify CLI (1 day)
- Merge knowledge-pipeline's CLI commands into `LLM-Brains/main.py`
- Commands to add: `pipeline run`, `pipeline status`, `pipeline schedule`

### Phase 4 — Deprecate knowledge-pipeline (after Phase 3 verified)
- Archive `knowledge-pipeline/` → `Projects/tools/_archived/knowledge-pipeline-YYYYMMDD/`
- Update team hub status

## Files to NOT merge
- `knowledge-pipeline/AIAgency/` — separate agent logic, keep separate
- `knowledge-pipeline/Desktop_Docs/` — may have unique ingest sources

## Blockers
- Need to verify Supabase schema is compatible with LLM-Brains' embedding dimensions (3072-dim gemini-embedding-001)
- knowledge-pipeline requirements.txt may have conflicts with LLM-Brains

## Next step
`diff knowledge-pipeline/requirements.txt LLM-Brains/requirements.txt` to check for conflicts, then start Phase 1.
