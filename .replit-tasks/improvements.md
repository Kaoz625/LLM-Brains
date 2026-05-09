# Replit Agent Task: LLM-Brains

## Goal
Upgrade LLM-Brains from a collection of standalone sync scripts into a cohesive AI research and community management platform — cleaning up the ai-seo Skool community data, integrating dream_cycle.py as a scheduled insight engine, and adding a SQLite-backed reranker for surfacing the best content.

## Tasks
1. **Audit existing scripts**: read all .py files (dream_cycle.py, notion_sync.py, notion_cleanup.py, skool_scraper.py, skool_auto_sync.py, notebooklm_sync.py, wearable_ingest.py, run_sync.sh); document what each does, what APIs it calls, and what env vars it needs; add a `SCRIPTS.md` summary file
2. **ai-seo community cleanup** (`community_cleanup.py`): reads from `skool_communities.json` (already present); identifies posts/members in the ai-seo community that are: spam (no replies, external link only), inactive (last post >90 days), or duplicate topics; outputs a `cleanup_report.json` with lists of post_ids to archive and member_ids to review; does NOT auto-delete — generates a human-review report only
3. **dream_cycle.py integration**: read the existing `dream_cycle.py` — it likely generates creative AI insights or reflections; create a scheduler wrapper `dream_scheduler.py` that:
   - Runs dream_cycle.py on a configurable cron schedule (default: 3am daily)
   - Saves each run's output to `logs/dream_YYYY-MM-DD.json`
   - Syncs the latest dream output to a Notion page (reuse notion_sync.py logic)
   - Sends a Telegram summary if `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` are set
4. **SQLite reranker** (`reranker.py`): 
   - Schema: `CREATE TABLE content (id TEXT PRIMARY KEY, source TEXT, title TEXT, body TEXT, url TEXT, score REAL, created_at TEXT, tags TEXT)`
   - `ingest(items: list[dict])` → inserts/updates items from Skool posts, Notion pages, or dream outputs
   - `rerank(query: str, top_k=10)` → uses TF-IDF cosine similarity (scikit-learn) to score all items against the query; returns top_k sorted by score
   - `get_top_content(days=7, min_score=0.3)` → returns highest-scored content from the last N days
   - Expose as CLI: `python reranker.py --query "AI SEO tips" --top 5`
5. **Unified runner** (`run_all.py`): single entry point that runs the full pipeline in order: skool_scraper → community_cleanup → notion_sync → reranker ingest → dream_scheduler check; accepts `--dry-run` flag that skips all writes; logs each step with timestamp to `logs/run_YYYY-MM-DD.log`
6. **Environment setup**: create `.env.example` documenting all required env vars: `NOTION_API_KEY`, `NOTION_DATABASE_ID`, `SKOOL_SESSION_COOKIE`, `ANTHROPIC_API_KEY`, `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`; update `run_sync.sh` to source `.env` before running
7. **Update index.html**: the existing index.html appears to be a web dashboard placeholder; add a simple status page showing: last sync timestamp, number of Skool posts indexed, number of Notion pages synced, latest dream_cycle.py output (last 3 lines), reranker DB size; read data from `logs/status.json` which run_all.py writes on each run
8. **Dependencies**: create/update `requirements.txt` with: `anthropic`, `notion-client`, `scikit-learn`, `python-dotenv`, `schedule`, `requests`; pin to stable versions
9. **Launchd plist**: create `com.nyctailblazers.llmbrains.plist` for macOS launchd to run `run_all.py` daily at 3am; document in README

## Tech Stack
- Python 3.10+ (existing)
- SQLite (stdlib — no ORM needed)
- scikit-learn for TF-IDF reranking
- Anthropic SDK (dream_cycle.py likely uses it)
- Notion API (notion-client)
- schedule library for dream_scheduler
- python-dotenv

## Deploy Target
Coolify (backend Python service) or local macOS via launchd. Scripts run headless. index.html status page → Cloudflare Pages. Never Vercel.

## Done When
- [ ] `SCRIPTS.md` documents all existing scripts and their env var requirements
- [ ] `community_cleanup.py` generates `cleanup_report.json` without auto-deleting anything
- [ ] `dream_scheduler.py` runs dream_cycle.py and saves output to logs/ + syncs to Notion
- [ ] `reranker.py` CLI: `python reranker.py --query "..." --top 5` returns ranked results
- [ ] `run_all.py --dry-run` completes without errors and logs each step
- [ ] `.env.example` lists all required environment variables
- [ ] `requirements.txt` includes all dependencies with versions
- [ ] `index.html` status page reads from `logs/status.json` and displays sync stats
- [ ] `com.nyctailblazers.llmbrains.plist` launchd file present and documented in README
