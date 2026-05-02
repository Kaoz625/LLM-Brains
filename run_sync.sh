#!/bin/bash
# Auto-restart notion_sync.py until all communities are fully synced.
# Waits 5 minutes between attempts to let Notion rate limits clear.
#
# Usage:
#   bash run_sync.sh            # sync all communities
#   bash run_sync.sh --community slug  # single community

cd "$(dirname "$0")"
ARGS="$@"
ATTEMPT=0

while true; do
  ATTEMPT=$((ATTEMPT + 1))
  echo "$(date '+%Y-%m-%d %H:%M:%S') — Starting sync attempt $ATTEMPT..."
  python3 -u notion_sync.py $ARGS
  EXIT=$?
  echo "$(date '+%Y-%m-%d %H:%M:%S') — Sync exited (code $EXIT)"

  # Check if all communities are done
  DONE=$(python3 -c "
import json
from pathlib import Path
state = json.loads(open('skool_sync_state.json').read())
slugs = [d.name for d in (Path('brain/raw/skool')).iterdir() if d.is_dir() and d.name != '.DS_Store']
remaining = 0
for slug in slugs:
    files = list((Path('brain/raw/skool') / slug / 'posts').glob('*.md'))
    if not files: continue
    sp = len(state.get(slug,{}).get('notion',{}).get('synced_posts',{}))
    if sp < len(files):
        remaining += len(files) - sp
print(remaining)
" 2>/dev/null)

  if [ "$DONE" = "0" ]; then
    echo "$(date '+%Y-%m-%d %H:%M:%S') — All communities synced! Done."
    break
  fi

  echo "$(date '+%Y-%m-%d %H:%M:%S') — $DONE posts remaining. Waiting 5 minutes before retry..."
  sleep 300
done
