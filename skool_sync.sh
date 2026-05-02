#!/bin/zsh
# Skool daily sync — runs via launchd at 3 AM
# Logs to ~/Library/Logs/skool-sync.log

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_FILE="$HOME/Library/Logs/skool-sync.log"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

log "=== Skool sync started ==="

# Start crawl4ai if needed (used internally by AsyncWebCrawler)
if ! curl -s --connect-timeout 1 http://localhost:11235/health &>/dev/null; then
  log "Starting crawl4ai service..."
  bash "$HOME/.bootstrap/start-crawl4ai.sh" >> "$LOG_FILE" 2>&1 || true
  sleep 3
fi

# Run scraper
log "Running skool_scraper.py..."
python3 "$SCRIPT_DIR/skool_scraper.py" >> "$LOG_FILE" 2>&1
SCRAPE_EXIT=$?

if [ $SCRAPE_EXIT -eq 0 ]; then
  log "Scraper finished successfully"
else
  log "Scraper exited with code $SCRAPE_EXIT (may be auth/access issue — cached content preserved)"
fi

# Run NotebookLM sync
log "Running notebooklm_sync.py..."
python3 "$SCRIPT_DIR/notebooklm_sync.py" >> "$LOG_FILE" 2>&1
NLM_EXIT=$?

if [ $NLM_EXIT -eq 0 ]; then
  log "NotebookLM sync finished successfully"
else
  log "NotebookLM sync exited with code $NLM_EXIT"
fi

# Run Notion sync
log "Running notion_sync.py..."
python3 "$SCRIPT_DIR/notion_sync.py" >> "$LOG_FILE" 2>&1
NOTION_EXIT=$?

if [ $NOTION_EXIT -eq 0 ]; then
  log "Notion sync finished successfully"
else
  log "Notion sync exited with code $NOTION_EXIT"
fi

log "=== Skool sync complete ==="
