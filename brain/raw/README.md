# DROP ANYTHING HERE

This folder is your input bin. The compiler automatically processes everything.

## What you can drop here:
- `.txt` / `.md` — notes, journal entries, articles
- `.pdf` — research papers, documents, books
- `.jpg` / `.png` / `.webp` — photos (vision AI describes them)
- `.mp3` / `.m4a` / `.wav` — voice memos, recordings (Whisper transcribes)
- `.mp4` / `.mov` — videos (keyframes analyzed + audio transcribed)
- `.url` — a file containing a URL (article or YouTube link gets fetched)

## Then run:
```bash
python compile.py
```

Or for continuous mode (watches for new files):
```bash
python compile.py --watch
```

Files move to `raw/processed/` after being compiled — originals are never deleted.
