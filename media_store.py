#!/usr/bin/env python3
"""
media_store.py — Local Media Download & Visual Memory Index
============================================================
Downloads video/audio from URLs and stores them locally with a rich
visual index so the brain can recall them in explicit detail.

The key insight: a downloaded video isn't just a file — it's a collection
of VISUAL MEMORIES with timestamps. This script builds that index:

  video.mp4
    ├── keyframe_0010.jpg  → "You at your desk, MacBook open, Karpathy video playing"
    ├── keyframe_0020.jpg  → "Terminal window, Python code, compile.py visible"
    ├── keyframe_0030.jpg  → "Browser, YouTube, LLM memory architecture search"
    └── transcript.txt     → full spoken word transcript with timestamps

The SQLite index means you can ask:
  "Show me everything I watched about RAG in the last month"
  "What YouTube videos did I save about memory architecture?"
  "Recall what Karpathy said about wiki compilation"

And the system can answer with: exact timestamps, frame descriptions,
spoken quotes, and the full stored media path to replay.

Pathway repair: if you forgot you watched a video, the visual index
creates 20+ retrieval pathways to that memory — by person, topic,
location, object, timestamp, spoken word. ANY of those paths finds it.

Sources supported:
  - YouTube (yt-dlp)
  - Any URL yt-dlp supports (Twitter/X, Instagram, TikTok, Vimeo, etc.)
  - Direct .mp4 / .mov / .mp3 / .m4a URLs
  - Local files (copy + index)

Usage:
    python media_store.py --url "https://youtube.com/watch?v=..."
    python media_store.py --url URL --no-video     # audio + transcript only
    python media_store.py --index                  # (re)index all stored media
    python media_store.py --search "karpathy rag"  # search stored media
    python media_store.py --recall path/to/video   # full recall of a stored video
"""

import os
import re
import sys
import json
import base64
import struct
import sqlite3
import hashlib
import logging
import argparse
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from datetime import datetime
from typing import Optional, List

log = logging.getLogger("media_store")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT        = Path(__file__).parent / "brain"
MEDIA_STORE = ROOT / "media" / "store"   # downloaded video/audio lives here
FRAMES_DIR  = ROOT / "media" / "frames"  # extracted keyframes
MEDIA_DB    = Path(__file__).parent / "media.db"

KEYFRAME_INTERVAL = 15   # seconds between keyframes
MAX_KEYFRAMES     = 40   # cap per video to limit API cost

# ---------------------------------------------------------------------------
# Media SQLite schema
# ---------------------------------------------------------------------------

MEDIA_SCHEMA = """
CREATE TABLE IF NOT EXISTS media (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    url          TEXT,
    url_hash     TEXT UNIQUE,
    title        TEXT,
    channel      TEXT,
    duration_sec REAL,
    file_path    TEXT,        -- local path to downloaded file
    transcript   TEXT,        -- full spoken word transcript
    summary      TEXT,        -- LLM-generated summary
    tags         TEXT,        -- JSON array
    indexed_at   INTEGER DEFAULT (strftime('%s','now'))
);

CREATE VIRTUAL TABLE IF NOT EXISTS media_fts USING fts5(
    title, transcript, summary, tags,
    content='media', content_rowid='id'
);

CREATE TABLE IF NOT EXISTS media_frames (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    media_id    INTEGER REFERENCES media(id),
    timestamp_s REAL,          -- seconds into video
    frame_path  TEXT,          -- local path to keyframe image
    description TEXT,          -- Claude vision description
    people      TEXT,          -- JSON array of people visible
    location    TEXT,
    objects     TEXT,          -- JSON array
    tags        TEXT           -- JSON array for fast lookup
);

CREATE INDEX IF NOT EXISTS idx_frames_media ON media_frames(media_id);
"""

def get_media_db() -> sqlite3.Connection:
    db = sqlite3.connect(str(MEDIA_DB))
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA journal_mode=WAL")
    db.executescript(MEDIA_SCHEMA)
    db.commit()
    return db

def url_hash(url: str) -> str:
    return hashlib.sha256(url.encode()).hexdigest()[:20]

# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_media(url: str, audio_only: bool = False) -> Optional[dict]:
    """
    Download video/audio via yt-dlp. Returns metadata dict.
    Stores file in brain/media/store/
    """
    MEDIA_STORE.mkdir(parents=True, exist_ok=True)

    try:
        # Get metadata first (no download)
        result = subprocess.run([
            "yt-dlp", "--dump-json", "--no-playlist", url
        ], capture_output=True, text=True, timeout=30)

        if result.returncode != 0:
            log.error("yt-dlp metadata failed: %s", result.stderr[:200])
            return None

        meta = json.loads(result.stdout)
    except FileNotFoundError:
        log.error("yt-dlp not found. Install: pip install yt-dlp  OR  brew install yt-dlp")
        return None
    except Exception as e:
        log.error("Metadata fetch failed: %s", e)
        return None

    title    = meta.get("title", "untitled")
    channel  = meta.get("uploader", meta.get("channel", "unknown"))
    duration = meta.get("duration", 0)
    vid_id   = meta.get("id", url_hash(url))

    # Safe filename
    safe_title = re.sub(r"[^\w\s-]", "", title)[:60].strip()
    timestamp  = datetime.now().strftime("%Y%m%d")
    ext        = "mp3" if audio_only else "mp4"
    filename   = f"{timestamp}_{vid_id}_{safe_title}.{ext}"
    out_path   = MEDIA_STORE / filename

    if out_path.exists():
        log.info("Already downloaded: %s", filename)
        return {
            "title": title, "channel": channel, "duration": duration,
            "file_path": str(out_path), "vid_id": vid_id, "meta": meta
        }

    log.info("Downloading: %s (%.0fs)", title, duration or 0)

    if audio_only:
        cmd = [
            "yt-dlp", "-x", "--audio-format", "mp3",
            "-o", str(out_path), "--no-playlist", url
        ]
    else:
        cmd = [
            "yt-dlp", "-f", "bestvideo[height<=720]+bestaudio/best[height<=720]",
            "--merge-output-format", "mp4",
            "-o", str(out_path), "--no-playlist", url
        ]

    result = subprocess.run(cmd, capture_output=True, timeout=600)
    if result.returncode != 0:
        log.error("Download failed: %s", result.stderr[:300])
        return None

    log.info("Downloaded: %s", out_path.name)
    return {
        "title": title, "channel": channel, "duration": duration,
        "file_path": str(out_path), "vid_id": vid_id, "meta": meta
    }

# ---------------------------------------------------------------------------
# Keyframe extraction + visual indexing
# ---------------------------------------------------------------------------

def extract_and_analyze_frames(
    video_path: Path,
    media_id: int,
    db: sqlite3.Connection,
    client
) -> List[dict]:
    """Extract keyframes from video and analyze each with Claude vision."""

    try:
        result = subprocess.run([
            "ffprobe", "-v", "quiet", "-print_format", "json",
            "-show_format", str(video_path)
        ], capture_output=True, text=True)
        duration = float(json.loads(result.stdout)["format"].get("duration", 0))
    except Exception:
        duration = 0

    # Adaptive interval
    interval = KEYFRAME_INTERVAL
    if duration > 0 and duration / interval > MAX_KEYFRAMES:
        interval = int(duration / MAX_KEYFRAMES)

    frames_out = FRAMES_DIR / str(media_id)
    frames_out.mkdir(parents=True, exist_ok=True)

    pattern = str(frames_out / "f_%05d.jpg")
    subprocess.run([
        "ffmpeg", "-y", "-i", str(video_path),
        "-vf", f"fps=1/{interval}", "-q:v", "3", pattern
    ], capture_output=True)

    frame_files = sorted(frames_out.glob("f_*.jpg"))
    log.info("Extracted %d keyframes", len(frame_files))

    analyses = []
    prior = ""
    for i, frame_path in enumerate(frame_files):
        ts = i * interval
        img_b64 = base64.standard_b64encode(frame_path.read_bytes()).decode()

        try:
            resp = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=400,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image", "source": {
                            "type": "base64", "media_type": "image/jpeg",
                            "data": img_b64
                        }},
                        {"type": "text", "text": f"""Analyze this video frame at {ts}s.
{f'Prior context: {prior}' if prior else ''}

Return JSON:
{{"description":"what is happening in 1-2 sentences",
  "people":["visible people with descriptions"],
  "location":"where this is",
  "objects":["notable items on screen or in scene"],
  "text_visible":"any text, code, UI, signs visible",
  "tags":["5-8 keyword tags"]}}"""}
                    ]
                }]
            )
            raw = resp.content[0].text.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1].lstrip("json").strip()
            frame_data = json.loads(raw)
        except Exception as e:
            log.debug("Frame %d analysis failed: %s", i, e)
            frame_data = {"description": "", "people": [], "tags": []}

        frame_data["timestamp_s"] = ts
        frame_data["frame_path"]  = str(frame_path)
        analyses.append(frame_data)

        db.execute("""
            INSERT INTO media_frames
            (media_id, timestamp_s, frame_path, description, people, location, objects, tags)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            media_id, ts, str(frame_path),
            frame_data.get("description", ""),
            json.dumps(frame_data.get("people", [])),
            frame_data.get("location", ""),
            json.dumps(frame_data.get("objects", [])),
            json.dumps(frame_data.get("tags", []))
        ])
        db.commit()

        if frame_data.get("description"):
            prior = frame_data["description"]
        time.sleep(0.15)

    return analyses

# ---------------------------------------------------------------------------
# Transcript
# ---------------------------------------------------------------------------

def transcribe(file_path: Path) -> str:
    """Transcribe audio/video via Whisper."""
    # For video, extract audio first
    suffix = file_path.suffix.lower()
    if suffix in {".mp4", ".mov", ".mkv", ".avi", ".webm"}:
        audio_tmp = file_path.with_suffix(".tmp.mp3")
        subprocess.run([
            "ffmpeg", "-y", "-i", str(file_path),
            "-q:a", "0", "-map", "a", str(audio_tmp)
        ], capture_output=True)
        result = _run_whisper(audio_tmp)
        audio_tmp.unlink(missing_ok=True)
        return result
    return _run_whisper(file_path)


def _run_whisper(path: Path) -> str:
    try:
        from faster_whisper import WhisperModel
        model = WhisperModel("base", device="cpu", compute_type="int8")
        segments, _ = model.transcribe(str(path))
        return " ".join(s.text for s in segments)
    except ImportError:
        pass
    try:
        import whisper
        return whisper.load_model("base").transcribe(str(path))["text"]
    except ImportError:
        log.warning("No Whisper available. pip install faster-whisper")
        return ""

# ---------------------------------------------------------------------------
# LLM summary with explicit detail
# ---------------------------------------------------------------------------

def generate_summary(title: str, channel: str, transcript: str,
                     frame_analyses: list, client) -> str:
    """
    Generate an extremely detailed summary — the 'explicit recall' the user wants.
    Covers: what was said, what was shown, visual details, key quotes, timestamps.
    """
    frames_text = "\n".join(
        f"[{a['timestamp_s']:.0f}s] {a.get('description','')} | "
        f"Visible: {a.get('text_visible','none')} | Tags: {a.get('tags',[])}"
        for a in frame_analyses[:25]
    )

    prompt = f"""Create an EXPLICIT DETAIL memory entry for this media.

TITLE: {title}
CHANNEL: {channel}

FRAME-BY-FRAME VISUAL RECORD:
{frames_text}

TRANSCRIPT (first 3000 chars):
{transcript[:3000]}

Write a comprehensive memory entry that includes:
1. **What this is** — title, source, why it's relevant
2. **Visual walkthrough** — what was shown at key timestamps, screen content,
   code visible, diagrams, people, demonstrations
3. **Key ideas and quotes** — exact or near-exact quotes from the transcript,
   with timestamps where possible
4. **New concepts introduced** — with definitions
5. **[[Cross-links]]** — at least 15 wikilinks to concepts, people, tools mentioned
6. **What to remember** — the 5 most important takeaways
7. **#tags** for retrieval

The goal: someone reading this memory entry should be able to describe the
video in explicit detail WITHOUT watching it. Every visual detail matters."""

    try:
        resp = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=3000,
            messages=[{"role": "user", "content": prompt}]
        )
        return resp.content[0].text
    except Exception as e:
        log.error("Summary generation failed: %s", e)
        return f"# {title}\n\nSource: {channel}\n\n{transcript[:1000]}"

# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def ingest_url(url: str, audio_only: bool = False,
               no_frames: bool = False) -> Optional[Path]:
    """Full pipeline: download → transcribe → index frames → summarize → store."""
    try:
        import anthropic
        client = anthropic.Anthropic()
    except ImportError:
        log.error("pip install anthropic")
        return None

    db = get_media_db()
    uhash = url_hash(url)

    # Check if already ingested
    existing = db.execute(
        "SELECT id, file_path FROM media WHERE url_hash = ?", [uhash]
    ).fetchone()
    if existing:
        log.info("Already in media store (id=%d)", existing["id"])
        return Path(existing["file_path"])

    # 1. Download
    media_info = download_media(url, audio_only)
    if not media_info:
        return None

    file_path = Path(media_info["file_path"])

    # 2. Transcribe
    log.info("Transcribing...")
    transcript = transcribe(file_path)
    log.info("Transcript: %d words", len(transcript.split()))

    # 3. Insert into DB (get ID first)
    db.execute("""
        INSERT INTO media (url, url_hash, title, channel, duration_sec, file_path, transcript)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, [url, uhash, media_info["title"], media_info["channel"],
          media_info["duration"], str(file_path), transcript])
    db.commit()
    media_id = db.execute(
        "SELECT id FROM media WHERE url_hash = ?", [uhash]
    ).fetchone()["id"]

    # 4. Extract and analyze keyframes (video only)
    frame_analyses = []
    if not audio_only and not no_frames and file_path.suffix == ".mp4":
        log.info("Analyzing keyframes...")
        frame_analyses = extract_and_analyze_frames(file_path, media_id, db, client)

    # 5. Generate explicit detail summary
    log.info("Generating detailed memory summary...")
    summary = generate_summary(
        media_info["title"], media_info["channel"],
        transcript, frame_analyses, client
    )

    # 6. Update DB with summary
    db.execute("UPDATE media SET summary = ? WHERE id = ?", [summary, media_id])
    db.commit()

    # 7. Write memory file to brain/knowledge or brain/me
    MEDIA_STORE.parent.mkdir(parents=True, exist_ok=True)
    memories_dir = ROOT / "media" / "memories"
    memories_dir.mkdir(parents=True, exist_ok=True)

    safe_title = re.sub(r"[^\w\s-]", "", media_info["title"])[:50].strip()
    mem_path = memories_dir / f"{datetime.now().strftime('%Y%m%d')}_{safe_title}.md"
    mem_path.write_text(
        f"---\nsource: {url}\nchannel: {media_info['channel']}\n"
        f"duration: {media_info['duration']:.0f}s\n"
        f"local_file: {file_path}\ndate: {datetime.now().strftime('%Y-%m-%d')}\n---\n\n"
        + summary
    )

    # 8. Also drop a pointer in raw/ for compile.py to weave into knowledge base
    (ROOT / "raw" / f"media_memory_{media_id}.txt").write_text(
        f"Source: {url}\nTitle: {media_info['title']}\nChannel: {media_info['channel']}\n\n"
        + summary
    )

    log.info("Media ingested. Memory at: %s", mem_path)
    return file_path


def search_media(query: str) -> list:
    """FTS5 search across all stored media."""
    db = get_media_db()
    try:
        rows = db.execute("""
            SELECT m.id, m.title, m.channel, m.url,
                   snippet(media_fts, 1, '[', ']', '...', 30) AS snippet
            FROM media_fts
            JOIN media m ON media_fts.rowid = m.id
            WHERE media_fts MATCH ?
            ORDER BY rank LIMIT 10
        """, [query]).fetchall()
        return [dict(r) for r in rows]
    except Exception as e:
        log.error("Search failed: %s", e)
        return []


def recall_media(identifier: str) -> str:
    """
    Return full explicit recall of a stored media item.
    identifier can be: URL, title fragment, or media ID.
    """
    db = get_media_db()
    row = None

    if identifier.isdigit():
        row = db.execute("SELECT * FROM media WHERE id = ?", [int(identifier)]).fetchone()
    elif identifier.startswith("http"):
        row = db.execute(
            "SELECT * FROM media WHERE url_hash = ?", [url_hash(identifier)]
        ).fetchone()
    else:
        row = db.execute(
            "SELECT * FROM media WHERE title LIKE ?", [f"%{identifier}%"]
        ).fetchone()

    if not row:
        return f"No media found matching: {identifier}"

    frames = db.execute(
        "SELECT timestamp_s, description, text_visible, people, tags "
        "FROM media_frames WHERE media_id = ? ORDER BY timestamp_s",
        [row["id"]]
    ).fetchall()

    output = [
        f"# {row['title']}",
        f"**Channel:** {row['channel']}  |  **Duration:** {row['duration_sec']:.0f}s",
        f"**URL:** {row['url']}",
        f"**Local:** {row['file_path']}",
        f"\n## Summary\n{row['summary'] or '(not yet generated)'}",
        f"\n## Frame-by-Frame Visual Record ({len(frames)} keyframes)"
    ]
    for f in frames:
        output.append(
            f"**[{f['timestamp_s']:.0f}s]** {f['description']}"
            + (f"  *(text: {f['text_visible']})*" if f['text_visible'] else "")
        )

    return "\n".join(output)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Media store — download, index, and recall video/audio"
    )
    parser.add_argument("--url", type=str,
                        help="Download and index a URL (YouTube, etc.)")
    parser.add_argument("--no-video", action="store_true",
                        help="Audio + transcript only (no video download)")
    parser.add_argument("--no-frames", action="store_true",
                        help="Skip frame-by-frame visual analysis")
    parser.add_argument("--search", type=str,
                        help="Search stored media by keyword")
    parser.add_argument("--recall", type=str, metavar="ID_OR_TITLE",
                        help="Full explicit recall of a stored media item")
    parser.add_argument("--list", action="store_true",
                        help="List all stored media")
    args = parser.parse_args()

    ROOT.mkdir(parents=True, exist_ok=True)
    MEDIA_STORE.mkdir(parents=True, exist_ok=True)

    if args.url:
        result = ingest_url(args.url, audio_only=args.no_video,
                            no_frames=args.no_frames)
        if result:
            print(f"\nStored: {result}")
            print("Run 'python compile.py' to weave into knowledge base.")

    elif args.search:
        results = search_media(args.search)
        if not results:
            print("No results.")
        for r in results:
            print(f"\n[{r['id']}] {r['title']}")
            print(f"     {r['channel']} | {r['url']}")
            print(f"     {r['snippet']}")

    elif args.recall:
        print(recall_media(args.recall))

    elif args.list:
        db = get_media_db()
        rows = db.execute(
            "SELECT id, title, channel, duration_sec, indexed_at FROM media ORDER BY id DESC"
        ).fetchall()
        print(f"\nStored media ({len(rows)} items):")
        for r in rows:
            ts = datetime.fromtimestamp(r["indexed_at"]).strftime("%Y-%m-%d")
            print(f"  [{r['id']}] {r['title'][:55]:<55} {r['channel'][:20]:<20} {ts}")

    else:
        parser.print_help()
        print("\nExamples:")
        print("  python media_store.py --url 'https://youtube.com/watch?v=...'")
        print("  python media_store.py --search 'karpathy memory wiki'")
        print("  python media_store.py --recall 'karpathy'")


if __name__ == "__main__":
    main()
