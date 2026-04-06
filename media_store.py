#!/usr/bin/env python3
"""
media_store.py — Downloads and indexes any URL (YouTube, Twitter, Vimeo, etc.)

Extracts keyframes, transcribes audio with Whisper, generates Claude vision summary.
Stores in brain/media/ with full metadata.

Usage:
    python media_store.py --url "https://youtube.com/watch?v=..."
    python media_store.py --recall "search term"
    python media_store.py --list
"""

import argparse
import base64
import hashlib
import json
import os
import re
import sqlite3
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional

import anthropic
from dotenv import load_dotenv

load_dotenv()

BRAIN_DIR = Path(os.getenv("BRAIN_DIR", "./brain"))
MEDIA_DIR = BRAIN_DIR / "media"
MEDIA_DB = BRAIN_DIR / "media.db"

MEDIA_DB_SCHEMA = """
CREATE TABLE IF NOT EXISTS media_items (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    url             TEXT UNIQUE NOT NULL,
    url_hash        TEXT UNIQUE NOT NULL,
    title           TEXT,
    author          TEXT,
    platform        TEXT,
    duration_secs   REAL,
    description     TEXT,
    transcript      TEXT,
    visual_summary  TEXT,
    key_topics      TEXT,    -- JSON array
    wikilinks       TEXT,    -- JSON array
    tags            TEXT,    -- JSON array
    file_path       TEXT,
    thumbnail_path  TEXT,
    processed_at    TEXT DEFAULT (datetime('now'))
);

CREATE VIRTUAL TABLE IF NOT EXISTS media_fts USING fts5(
    title,
    description,
    transcript,
    visual_summary,
    key_topics,
    content='media_items',
    content_rowid='id'
);

CREATE TRIGGER IF NOT EXISTS media_ai AFTER INSERT ON media_items BEGIN
    INSERT INTO media_fts(rowid, title, description, transcript, visual_summary, key_topics)
    VALUES (new.id, new.title, new.description, new.transcript, new.visual_summary, new.key_topics);
END;
CREATE TRIGGER IF NOT EXISTS media_ad AFTER DELETE ON media_items BEGIN
    INSERT INTO media_fts(media_fts, rowid, title, description, transcript, visual_summary, key_topics)
    VALUES ('delete', old.id, old.title, old.description, old.transcript, old.visual_summary, old.key_topics);
END;
"""


def get_db() -> sqlite3.Connection:
    MEDIA_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(MEDIA_DB))
    conn.row_factory = sqlite3.Row
    conn.executescript(MEDIA_DB_SCHEMA)
    conn.commit()
    return conn


def url_hash(url: str) -> str:
    return hashlib.sha256(url.encode()).hexdigest()[:16]


def get_client() -> anthropic.Anthropic:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("ANTHROPIC_API_KEY not set", file=sys.stderr)
        sys.exit(1)
    return anthropic.Anthropic(api_key=api_key)


# ---------------------------------------------------------------------------
# yt-dlp download
# ---------------------------------------------------------------------------

def download_with_ytdlp(url: str, output_dir: Path) -> dict:
    """Download media using yt-dlp and return metadata."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_template = str(output_dir / "%(id)s.%(ext)s")

    # Get metadata first
    meta_result = subprocess.run(
        ["yt-dlp", "--dump-json", "--no-download", url],
        capture_output=True, text=True, timeout=60
    )

    metadata = {}
    if meta_result.returncode == 0:
        try:
            metadata = json.loads(meta_result.stdout)
        except json.JSONDecodeError:
            pass

    # Download video (best quality, max 1080p for efficiency)
    dl_result = subprocess.run(
        [
            "yt-dlp",
            "-f", "bestvideo[height<=720]+bestaudio/best[height<=720]/best",
            "--merge-output-format", "mp4",
            "-o", output_template,
            "--write-thumbnail",
            "--no-playlist",
            url
        ],
        capture_output=True, text=True, timeout=600
    )

    if dl_result.returncode != 0:
        print(f"  yt-dlp error: {dl_result.stderr[:300]}", file=sys.stderr)

    return metadata


# ---------------------------------------------------------------------------
# Keyframe extraction
# ---------------------------------------------------------------------------

def extract_keyframes(video_path: Path, max_frames: int = 8) -> list[Path]:
    """Extract evenly-spaced keyframes from video."""
    frames_dir = video_path.parent / f"frames_{video_path.stem}"
    frames_dir.mkdir(exist_ok=True)

    try:
        # Get duration
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "json", str(video_path)],
            capture_output=True, text=True, timeout=30
        )
        duration = 0.0
        try:
            duration = float(json.loads(probe.stdout)["format"]["duration"])
        except Exception:
            pass

        interval = max(5, int(duration / max_frames)) if duration > 0 else 30

        subprocess.run(
            ["ffmpeg", "-y", "-i", str(video_path),
             "-vf", f"fps=1/{interval},scale=1024:-1",
             "-q:v", "3",
             str(frames_dir / "frame_%04d.jpg")],
            capture_output=True, timeout=300
        )
        return sorted(frames_dir.glob("frame_*.jpg"))[:max_frames]
    except Exception as e:
        print(f"  Frame extraction failed: {e}", file=sys.stderr)
        return []


# ---------------------------------------------------------------------------
# Transcription
# ---------------------------------------------------------------------------

def transcribe_audio(video_path: Path) -> str:
    """Transcribe audio track from video using Whisper."""
    audio_path = video_path.with_suffix(".tmp_audio.wav")
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(video_path), "-vn", "-acodec", "pcm_s16le",
             "-ar", "16000", "-ac", "1", str(audio_path)],
            capture_output=True, timeout=300
        )
        if not audio_path.exists():
            return ""

        try:
            from faster_whisper import WhisperModel
            model = WhisperModel("base", device="cpu", compute_type="int8")
            segments, _ = model.transcribe(str(audio_path))
            return " ".join(s.text.strip() for s in segments)
        except ImportError:
            return "[faster-whisper not available]"
        except Exception as e:
            return f"[Transcription error: {e}]"
    finally:
        if audio_path.exists():
            audio_path.unlink()


# ---------------------------------------------------------------------------
# Claude analysis
# ---------------------------------------------------------------------------

def analyze_frames(client: anthropic.Anthropic, frames: list[Path],
                   title: str = "") -> str:
    """Analyze video keyframes with Claude vision."""
    if not frames:
        return ""

    # Sample up to 6 frames
    sampled = frames[:6]
    content_blocks = []

    for frame in sampled:
        try:
            with open(frame, "rb") as f:
                img_b64 = base64.standard_b64encode(f.read()).decode()
            content_blocks.append({
                "type": "image",
                "source": {"type": "base64", "media_type": "image/jpeg", "data": img_b64}
            })
        except Exception:
            pass

    content_blocks.append({
        "type": "text",
        "text": (
            f"These are keyframes from a video titled: '{title}'\n"
            "Provide a detailed visual analysis:\n"
            "1. What is happening in each frame?\n"
            "2. Visual style, production quality\n"
            "3. Key people, objects, or demonstrations shown\n"
            "4. Overall narrative arc visible from frames\n"
            "5. Any text, code, or diagrams visible"
        )
    })

    try:
        response = client.messages.create(
            model="claude-opus-4-5",
            max_tokens=1024,
            messages=[{"role": "user", "content": content_blocks}]
        )
        return response.content[0].text
    except Exception as e:
        return f"[Vision analysis failed: {e}]"


def generate_media_summary(client: anthropic.Anthropic, metadata: dict,
                            transcript: str, visual_analysis: str) -> dict:
    """Generate comprehensive media summary using Claude."""
    title = metadata.get("title", "Unknown")
    description = metadata.get("description", "")[:500]
    uploader = metadata.get("uploader", metadata.get("channel", "Unknown"))
    duration = metadata.get("duration", 0)

    prompt = f"""Analyze this media content and create a comprehensive knowledge entry.

Title: {title}
Creator: {uploader}
Duration: {duration}s
Description: {description}

Transcript (excerpt):
{transcript[:3000]}

Visual analysis:
{visual_analysis[:2000]}

Return JSON:
{{
  "title": "clean title",
  "author": "creator name",
  "platform": "YouTube|Twitter|Vimeo|etc",
  "summary": "3-5 sentence summary of content and key insights",
  "key_topics": ["topic1", "topic2", "topic3"],
  "key_takeaways": ["takeaway1", "takeaway2"],
  "wikilinks": ["[[concept]]", "[[person]]"],
  "tags": ["tag1", "tag2"],
  "quality_rating": "educational|entertainment|reference|news",
  "markdown": "full structured markdown entry with sections and [[wikilinks]]"
}}"""

    try:
        response = client.messages.create(
            model="claude-opus-4-5",
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}]
        )
        text = response.content[0].text
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            return json.loads(match.group())
    except Exception as e:
        print(f"  Summary generation error: {e}", file=sys.stderr)

    return {
        "title": title,
        "author": uploader,
        "platform": "unknown",
        "summary": description[:300],
        "key_topics": [],
        "key_takeaways": [],
        "wikilinks": [],
        "tags": [],
        "quality_rating": "unknown",
        "markdown": f"# {title}\n\n{description}\n\n## Transcript\n{transcript[:2000]}"
    }


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

def save_media_entry(url: str, metadata: dict, summary: dict,
                     transcript: str, visual_summary: str,
                     file_path: str = "", thumbnail_path: str = "") -> Path:
    """Save media entry to database and markdown file."""
    conn = get_db()
    h = url_hash(url)

    conn.execute("""
        INSERT INTO media_items
            (url, url_hash, title, author, platform, duration_secs,
             description, transcript, visual_summary, key_topics,
             wikilinks, tags, file_path, thumbnail_path)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(url) DO UPDATE SET
            title=excluded.title, transcript=excluded.transcript,
            visual_summary=excluded.visual_summary,
            key_topics=excluded.key_topics, tags=excluded.tags,
            processed_at=datetime('now')
    """, (
        url, h,
        summary.get("title", metadata.get("title", "Unknown")),
        summary.get("author", ""),
        summary.get("platform", ""),
        metadata.get("duration", 0),
        metadata.get("description", "")[:1000],
        transcript[:10000],
        visual_summary[:3000],
        json.dumps(summary.get("key_topics", [])),
        json.dumps(summary.get("wikilinks", [])),
        json.dumps(summary.get("tags", [])),
        file_path,
        thumbnail_path,
    ))
    conn.commit()
    conn.close()

    # Write markdown
    MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    title = summary.get("title", "media")
    safe = re.sub(r"[^\w\s-]", "", title.lower()).strip().replace(" ", "_")[:60]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    md_path = MEDIA_DIR / f"{timestamp}_{safe}.md"

    wikilinks_str = " ".join(summary.get("wikilinks", []))
    topics_str = ", ".join(summary.get("key_topics", []))

    content = f"""---
title: {summary.get('title', title)}
url: {url}
author: {summary.get('author', '')}
platform: {summary.get('platform', '')}
duration: {metadata.get('duration', 0)}s
tags: {json.dumps(summary.get('tags', []))}
key_topics: {json.dumps(summary.get('key_topics', []))}
quality: {summary.get('quality_rating', '')}
processed_at: {datetime.now().isoformat()}
---

# {summary.get('title', title)}

**Author:** {summary.get('author', '')} | **Platform:** {summary.get('platform', '')}
**Duration:** {metadata.get('duration', 0)}s
**Topics:** {topics_str}
**Links:** {wikilinks_str}

---

## Summary
{summary.get('summary', '')}

## Key Takeaways
{chr(10).join('- ' + t for t in summary.get('key_takeaways', []))}

---

{summary.get('markdown', '')}

---

## Transcript Excerpt
{transcript[:2000]}...

---
*URL: {url}*
*Processed: {datetime.now().strftime('%Y-%m-%d %H:%M')}*
"""
    md_path.write_text(content, encoding="utf-8")
    return md_path


# ---------------------------------------------------------------------------
# Search / recall
# ---------------------------------------------------------------------------

def search_media(query: str, limit: int = 10) -> list[dict]:
    """Search media store by query."""
    conn = get_db()
    try:
        rows = conn.execute("""
            SELECT m.id, m.title, m.url, m.author, m.platform,
                   m.summary, m.key_topics, m.processed_at,
                   bm25(media_fts) AS score
            FROM media_fts
            JOIN media_items m ON media_fts.rowid = m.id
            WHERE media_fts MATCH ?
            ORDER BY score
            LIMIT ?
        """, (query, limit)).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception as e:
        conn.close()
        # Fallback
        conn2 = get_db()
        rows = conn2.execute("""
            SELECT id, title, url, author, platform, description as summary,
                   key_topics, processed_at
            FROM media_items
            WHERE title LIKE ? OR description LIKE ? OR transcript LIKE ?
            LIMIT ?
        """, (f"%{query}%",) * 3 + (limit,)).fetchall()
        conn2.close()
        return [dict(r) for r in rows]


def list_media(limit: int = 20) -> list[dict]:
    conn = get_db()
    rows = conn.execute("""
        SELECT id, title, url, author, platform, duration_secs, processed_at
        FROM media_items ORDER BY processed_at DESC LIMIT ?
    """, (limit,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Main download + process pipeline
# ---------------------------------------------------------------------------

def process_url(url: str, client: anthropic.Anthropic) -> Optional[Path]:
    """Full pipeline: download → extract → transcribe → analyze → store."""
    print(f"\nProcessing URL: {url}")

    # Check if already processed
    conn = get_db()
    existing = conn.execute(
        "SELECT id, title FROM media_items WHERE url=?", (url,)
    ).fetchone()
    conn.close()
    if existing:
        print(f"  Already processed: {existing['title']}")
        return None

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Download
        print("  Downloading...")
        metadata = {}
        try:
            metadata = download_with_ytdlp(url, tmp_path)
        except FileNotFoundError:
            print("  yt-dlp not found — attempting transcript-only fallback", file=sys.stderr)
            # Try YouTube transcript API fallback
            try:
                from youtube_transcript_api import YouTubeTranscriptApi
                vid_match = re.search(r"(?:v=|youtu\.be/)([a-zA-Z0-9_-]{11})", url)
                if vid_match:
                    vid_id = vid_match.group(1)
                    entries = YouTubeTranscriptApi.get_transcript(vid_id)
                    transcript = " ".join(e["text"] for e in entries)
                    summary = generate_media_summary(client, {"title": url}, transcript, "")
                    md_path = save_media_entry(url, {}, summary, transcript, "")
                    print(f"  -> Saved (transcript only): {md_path.name}")
                    return md_path
            except Exception as e2:
                print(f"  Transcript fallback failed: {e2}", file=sys.stderr)
            return None
        except subprocess.TimeoutExpired:
            print("  Download timed out", file=sys.stderr)
            return None

        # Find downloaded video file
        video_files = list(tmp_path.glob("*.mp4")) + list(tmp_path.glob("*.mkv")) + \
                      list(tmp_path.glob("*.webm"))
        thumbnail_files = list(tmp_path.glob("*.jpg")) + list(tmp_path.glob("*.webp")) + \
                          list(tmp_path.glob("*.png"))

        transcript = ""
        visual_summary = ""
        frames = []

        if video_files:
            video_file = video_files[0]
            print(f"  Extracting keyframes from {video_file.name}...")
            frames = extract_keyframes(video_file, max_frames=8)

            print("  Transcribing audio...")
            transcript = transcribe_audio(video_file)

            if frames:
                print(f"  Analyzing {len(frames)} frames...")
                visual_summary = analyze_frames(
                    client, frames, title=metadata.get("title", url)
                )
        else:
            # No video file — try YouTube transcript API
            print("  No video downloaded, trying transcript API...")
            try:
                from youtube_transcript_api import YouTubeTranscriptApi
                vid_match = re.search(r"(?:v=|youtu\.be/)([a-zA-Z0-9_-]{11})", url)
                if vid_match:
                    entries = YouTubeTranscriptApi.get_transcript(vid_match.group(1))
                    transcript = " ".join(e["text"] for e in entries)
            except Exception:
                pass

        # Generate summary
        print("  Generating summary...")
        summary = generate_media_summary(client, metadata, transcript, visual_summary)

        # Copy thumbnail to media dir
        thumb_dest = ""
        if thumbnail_files:
            MEDIA_DIR.mkdir(parents=True, exist_ok=True)
            thumb_dest = str(MEDIA_DIR / f"thumb_{url_hash(url)}{thumbnail_files[0].suffix}")
            import shutil
            shutil.copy2(str(thumbnail_files[0]), thumb_dest)

        # Save
        md_path = save_media_entry(
            url, metadata, summary, transcript, visual_summary,
            file_path="", thumbnail_path=thumb_dest
        )
        print(f"  -> Saved: {md_path.name}")
        return md_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Media store: download and index URLs")
    parser.add_argument("--url", type=str, help="URL to download and index")
    parser.add_argument("--recall", type=str, help="Search for media by topic")
    parser.add_argument("--list", action="store_true", help="List all stored media")
    parser.add_argument("--limit", type=int, default=10, help="Max results")
    args = parser.parse_args()

    if args.recall:
        results = search_media(args.recall, limit=args.limit)
        if not results:
            print(f"No media found for: {args.recall}")
        else:
            print(f"\nMedia results for '{args.recall}':\n{'='*50}")
            for r in results:
                print(f"\n  [{r.get('platform', '?')}] {r.get('title', 'Untitled')}")
                print(f"  By: {r.get('author', '?')} | {r.get('processed_at', '')[:10]}")
                print(f"  {r.get('summary', '')[:150]}...")
                print(f"  URL: {r.get('url', '')}")
        return

    if args.list:
        items = list_media(limit=args.limit)
        print(f"\nStored media ({len(items)} items):\n{'='*50}")
        for item in items:
            duration = int(item.get("duration_secs") or 0)
            print(f"  [{item.get('platform', '?')}] {item.get('title', 'Untitled')}")
            print(f"  {item.get('processed_at', '')[:10]} | {duration}s | {item.get('url', '')[:60]}")
        return

    if args.url:
        client = get_client()
        process_url(args.url, client)
        return

    parser.print_help()


if __name__ == "__main__":
    main()
