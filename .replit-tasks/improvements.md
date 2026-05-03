# LLM-Brains — Replit Import Notes

## What This Project Is
Skool multi-community scraper + Notion sync pipeline + local AI memory system.

## Files Added This Session
- `dream_cycle.py` — nightly episodic→semantic memory promotion via Claude Haiku
- `wearable_ingest.py` — Meta glasses video→text transcription via whisper-cpp (local, free)
- `notion_cleanup.py` — updated with correct community slugs + --fast + --yes flags
- `brain/` output dirs: me/, knowledge/, work/, media/ (created by dream_cycle at runtime)
- `logs/` dir for launchd job output

## Pending (requires Markus action)
- Run `python3 notion_cleanup.py --fast --yes` to archive 28K duplicate Notion pages
- Notion sync will auto-correct dedup on next run after cleanup

## Known Issues
- ai-seo-with-julian-goldie-1553 and ai-automation-society have 0 raw posts (may need re-scrape)
- whisper-cli only has test model; real model auto-downloaded on first `wearable_ingest.py` run

## Improvements Roadmap
- [ ] Cross-encoder reranker on top of FTS5 results in sqlite_rag.py
- [ ] Classroom JS-extraction fix (crawl4ai js_code for course card links)
- [ ] Dream cycle: batch multiple files per Claude call to reduce API costs
