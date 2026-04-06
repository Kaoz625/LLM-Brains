#!/usr/bin/env python3
"""
compile.py — Main brain compiler.
Watches brain/raw/ and processes any file dropped there.

Usage:
    python compile.py           # one-shot
    python compile.py --watch   # continuous watch mode
"""

import argparse
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import anthropic
from dotenv import load_dotenv

load_dotenv()

BRAIN_DIR = Path(os.getenv("BRAIN_DIR", "./brain"))
RAW_DIR = BRAIN_DIR / "raw"
PROCESSED_DIR = RAW_DIR / "processed"
HASHES_FILE = BRAIN_DIR / ".processed_hashes"
LOG_FILE = BRAIN_DIR / "compile.log"
INDEX_FILE = BRAIN_DIR / "index.md"

SUPPORTED_TEXT = {".txt", ".md"}
SUPPORTED_PDF = {".pdf"}
SUPPORTED_IMAGE = {".jpg", ".jpeg", ".png", ".webp"}
SUPPORTED_AUDIO = {".mp3", ".wav", ".m4a"}
SUPPORTED_VIDEO = {".mp4", ".mov"}

# Route keywords for classification
ROUTE_KEYWORDS = {
    "me": ["i ", "my ", "me ", "i'm", "i've", "personal", "diary", "journal", "health", "sleep", "exercise"],
    "work": ["work", "job", "project", "meeting", "client", "business", "deadline", "task", "team", "sprint"],
    "knowledge": ["research", "paper", "study", "theory", "concept", "science", "technology", "history", "philosophy"],
    "media": ["video", "podcast", "movie", "film", "music", "book", "article", "youtube", "watch", "listen"],
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(str(LOG_FILE), mode="a"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


def ensure_dirs():
    for d in [RAW_DIR, PROCESSED_DIR, BRAIN_DIR / "me", BRAIN_DIR / "work",
              BRAIN_DIR / "knowledge" / "wiki", BRAIN_DIR / "media", BRAIN_DIR / "fragments"]:
        d.mkdir(parents=True, exist_ok=True)


def load_hashes() -> set:
    if HASHES_FILE.exists():
        return set(HASHES_FILE.read_text().splitlines())
    return set()


def save_hash(h: str):
    with open(HASHES_FILE, "a") as f:
        f.write(h + "\n")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def extract_youtube_urls(text: str) -> list[str]:
    pattern = r"(https?://(?:www\.)?(?:youtube\.com/watch\?v=|youtu\.be/)[\w\-_?=&]+)"
    return re.findall(pattern, text)


def transcribe_audio(audio_path: Path) -> str:
    """Transcribe audio file using faster-whisper."""
    try:
        from faster_whisper import WhisperModel
        model = WhisperModel("base", device="cpu", compute_type="int8")
        segments, _ = model.transcribe(str(audio_path))
        return " ".join(seg.text for seg in segments).strip()
    except ImportError:
        logger.warning("faster-whisper not available, skipping transcription")
        return f"[Audio file: {audio_path.name} — transcription unavailable]"
    except Exception as e:
        logger.error(f"Transcription error for {audio_path}: {e}")
        return f"[Transcription failed: {e}]"


def extract_audio_from_video(video_path: Path) -> Optional[Path]:
    """Extract audio track from video using ffmpeg."""
    audio_path = video_path.with_suffix(".extracted.wav")
    try:
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", str(video_path), "-vn", "-acodec", "pcm_s16le",
             "-ar", "16000", "-ac", "1", str(audio_path)],
            capture_output=True, text=True, timeout=300
        )
        if result.returncode == 0 and audio_path.exists():
            return audio_path
        logger.error(f"ffmpeg failed: {result.stderr}")
        return None
    except FileNotFoundError:
        logger.warning("ffmpeg not found, cannot extract audio from video")
        return None
    except subprocess.TimeoutExpired:
        logger.error("ffmpeg timed out")
        return None


def fetch_youtube_transcript(url: str) -> str:
    """Fetch YouTube transcript using youtube-transcript-api."""
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        import re as _re
        match = _re.search(r"(?:v=|youtu\.be/)([a-zA-Z0-9_-]{11})", url)
        if not match:
            return f"[Could not parse YouTube URL: {url}]"
        video_id = match.group(1)
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return " ".join(entry["text"] for entry in transcript)
    except Exception as e:
        logger.warning(f"Could not get YouTube transcript for {url}: {e}")
        # Fall back to yt-dlp
        try:
            result = subprocess.run(
                ["yt-dlp", "--skip-download", "--write-auto-sub", "--sub-format", "vtt",
                 "--output", "/tmp/yt_%(id)s", url],
                capture_output=True, text=True, timeout=60
            )
            return f"[YouTube URL: {url} — transcript unavailable: {e}]"
        except Exception:
            return f"[YouTube URL: {url} — transcript unavailable]"


def analyze_image_with_claude(client: anthropic.Anthropic, image_path: Path) -> str:
    """Use Claude vision to analyze an image."""
    import base64
    suffix = image_path.suffix.lower()
    media_type_map = {".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                      ".png": "image/png", ".webp": "image/webp"}
    media_type = media_type_map.get(suffix, "image/jpeg")
    with open(image_path, "rb") as f:
        image_data = base64.standard_b64encode(f.read()).decode("utf-8")
    try:
        response = client.messages.create(
            model="claude-opus-4-5",
            max_tokens=1024,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": image_data}},
                    {"type": "text", "text": (
                        "Describe this image in detail. Extract any text visible. "
                        "Identify people, places, objects, activities, dates, or events. "
                        "Format as structured notes suitable for a personal knowledge base."
                    )}
                ]
            }]
        )
        return response.content[0].text
    except Exception as e:
        logger.error(f"Claude vision error for {image_path}: {e}")
        return f"[Image: {image_path.name} — vision analysis failed: {e}]"


def extract_pdf_text(pdf_path: Path) -> str:
    """Extract text from PDF using pymupdf."""
    try:
        import fitz
        doc = fitz.open(str(pdf_path))
        pages = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            pages.append(f"[Page {page_num+1}]\n{page.get_text()}")
        doc.close()
        return "\n\n".join(pages)
    except ImportError:
        logger.warning("pymupdf not available")
        return f"[PDF: {pdf_path.name} — text extraction unavailable]"
    except Exception as e:
        logger.error(f"PDF extraction error for {pdf_path}: {e}")
        return f"[PDF extraction failed: {e}]"


def route_content(content: str) -> str:
    """Simple keyword-based routing fallback."""
    content_lower = content.lower()
    scores = {domain: 0 for domain in ROUTE_KEYWORDS}
    for domain, keywords in ROUTE_KEYWORDS.items():
        for kw in keywords:
            scores[domain] += content_lower.count(kw)
    return max(scores, key=scores.get)


def compile_with_claude(client: anthropic.Anthropic, content: str, source_name: str) -> dict:
    """
    Use Claude to extract structure from raw content.
    Returns: {title, summary, concepts, events, people, dates, route, wikilinks, markdown}
    """
    prompt = f"""You are a personal knowledge base compiler. Process this raw content and extract structured information.

Source: {source_name}
Content:
{content[:8000]}

Return a JSON object with:
{{
  "title": "concise title",
  "summary": "2-3 sentence summary",
  "concepts": ["concept1", "concept2"],
  "events": ["event1 (date if known)"],
  "people": ["person1", "person2"],
  "dates": ["YYYY-MM-DD or description"],
  "route": "me|work|knowledge|media",
  "wikilinks": ["[[concept1]]", "[[concept2]]"],
  "markdown": "full markdown article with [[wikilinks]] for key concepts",
  "tags": ["tag1", "tag2"]
}}

Guidelines for routing:
- "me": personal experiences, health, diary, emotions
- "work": professional activities, projects, meetings
- "knowledge": research, learning, facts, theories
- "media": articles read, videos watched, books, podcasts

Use [[double bracket]] syntax for important concepts in the markdown field."""

    try:
        response = client.messages.create(
            model="claude-opus-4-5",
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}]
        )
        text = response.content[0].text
        # Extract JSON from response
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        return {"title": source_name, "summary": content[:200], "route": route_content(content),
                "markdown": content, "wikilinks": [], "concepts": [], "events": [], "people": [],
                "dates": [], "tags": []}
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error from Claude response: {e}")
        return {"title": source_name, "summary": content[:200], "route": route_content(content),
                "markdown": content, "wikilinks": [], "concepts": [], "events": [], "people": [],
                "dates": [], "tags": []}
    except Exception as e:
        logger.error(f"Claude API error: {e}")
        return {"title": source_name, "summary": content[:200], "route": route_content(content),
                "markdown": content, "wikilinks": [], "concepts": [], "events": [], "people": [],
                "dates": [], "tags": []}


def write_entry(data: dict, source_file: Path) -> Path:
    """Write compiled entry to the appropriate brain directory."""
    route = data.get("route", "knowledge")
    title = data.get("title", source_file.stem)
    safe_title = re.sub(r'[^\w\s-]', '', title).strip().replace(' ', '_')[:80]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{safe_title}.md"

    route_map = {
        "me": BRAIN_DIR / "me",
        "work": BRAIN_DIR / "work",
        "knowledge": BRAIN_DIR / "knowledge",
        "media": BRAIN_DIR / "media",
    }
    dest_dir = route_map.get(route, BRAIN_DIR / "knowledge")
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / filename

    wikilinks = " ".join(data.get("wikilinks", []))
    tags = ", ".join(data.get("tags", []))
    content = f"""---
title: {title}
source: {source_file.name}
date: {datetime.now().isoformat()}
route: {route}
tags: [{tags}]
concepts: {json.dumps(data.get('concepts', []))}
people: {json.dumps(data.get('people', []))}
events: {json.dumps(data.get('events', []))}
dates: {json.dumps(data.get('dates', []))}
---

# {title}

**Summary:** {data.get('summary', '')}

**Links:** {wikilinks}

---

{data.get('markdown', '')}

---
*Compiled: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Source: {source_file.name}*
"""
    dest_path.write_text(content, encoding="utf-8")
    return dest_path


def update_index(entries: list[dict]):
    """Update brain/index.md with newly added entries."""
    today = datetime.now().strftime("%Y-%m-%d")
    existing = INDEX_FILE.read_text(encoding="utf-8") if INDEX_FILE.exists() else "# Brain Index\n\n"

    new_lines = [f"\n## Entries added {today}\n"]
    for e in entries:
        route = e.get("route", "knowledge")
        title = e.get("title", "Untitled")
        dest = e.get("dest_path", "")
        rel = Path(dest).relative_to(BRAIN_DIR) if dest else title
        new_lines.append(f"- [{title}]({rel}) `{route}`")

    INDEX_FILE.write_text(existing + "\n".join(new_lines) + "\n", encoding="utf-8")


def process_file(path: Path, client: anthropic.Anthropic, known_hashes: set) -> Optional[dict]:
    """Process a single file. Returns compiled data dict or None if skipped."""
    suffix = path.suffix.lower()
    logger.info(f"Processing: {path.name}")

    # Deduplication
    file_hash = sha256_file(path)
    if file_hash in known_hashes:
        logger.info(f"  Skipping duplicate: {path.name}")
        return {"skipped": True, "reason": "duplicate"}

    content = ""

    if suffix in SUPPORTED_TEXT:
        raw = path.read_text(encoding="utf-8", errors="replace")
        # Check for YouTube URLs
        yt_urls = extract_youtube_urls(raw)
        if yt_urls:
            transcripts = []
            for url in yt_urls:
                logger.info(f"  Fetching YouTube transcript: {url}")
                transcripts.append(f"[YouTube: {url}]\n{fetch_youtube_transcript(url)}")
            content = raw + "\n\n" + "\n\n".join(transcripts)
        else:
            content = raw

    elif suffix in SUPPORTED_PDF:
        content = extract_pdf_text(path)

    elif suffix in SUPPORTED_IMAGE:
        content = analyze_image_with_claude(client, path)

    elif suffix in SUPPORTED_AUDIO:
        content = transcribe_audio(path)

    elif suffix in SUPPORTED_VIDEO:
        audio_path = extract_audio_from_video(path)
        if audio_path:
            content = f"[Video: {path.name}]\n" + transcribe_audio(audio_path)
            audio_path.unlink(missing_ok=True)
        else:
            content = f"[Video: {path.name} — audio extraction unavailable]"
    else:
        logger.warning(f"  Unsupported file type: {suffix}")
        return None

    if not content.strip():
        logger.warning(f"  Empty content after extraction: {path.name}")
        return None

    # Compile with Claude
    data = compile_with_claude(client, content, path.name)
    dest_path = write_entry(data, path)
    data["dest_path"] = str(dest_path)

    # Store hash
    save_hash(file_hash)
    known_hashes.add(file_hash)

    # Move to processed
    dest_processed = PROCESSED_DIR / path.name
    if dest_processed.exists():
        dest_processed = PROCESSED_DIR / f"{path.stem}_{int(time.time())}{path.suffix}"
    shutil.move(str(path), str(dest_processed))

    logger.info(f"  -> {dest_path.relative_to(BRAIN_DIR)} [{data.get('route')}]")
    return data


def run_once() -> dict:
    """Process all files currently in brain/raw/. Returns stats."""
    ensure_dirs()
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        logger.error("ANTHROPIC_API_KEY not set")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)
    known_hashes = load_hashes()

    files = [f for f in RAW_DIR.iterdir()
             if f.is_file() and f.suffix.lower() in
             (SUPPORTED_TEXT | SUPPORTED_PDF | SUPPORTED_IMAGE | SUPPORTED_AUDIO | SUPPORTED_VIDEO)]

    stats = {"processed": 0, "merged": 0, "duplicates": 0, "new_entries": 0, "errors": 0}
    new_entries = []

    for path in sorted(files):
        try:
            result = process_file(path, client, known_hashes)
            if result is None:
                stats["errors"] += 1
            elif result.get("skipped"):
                stats["duplicates"] += 1
            else:
                stats["processed"] += 1
                stats["new_entries"] += 1
                new_entries.append(result)
        except Exception as e:
            logger.error(f"Error processing {path.name}: {e}")
            stats["errors"] += 1

    if new_entries:
        update_index(new_entries)

    return stats


def watch_mode(interval: int = 5):
    """Continuously watch brain/raw/ for new files."""
    logger.info(f"Watch mode active — polling every {interval}s")
    try:
        from watchdog.observers import Observer
        from watchdog.events import FileSystemEventHandler

        class Handler(FileSystemEventHandler):
            def __init__(self):
                self.pending = set()

            def on_created(self, event):
                if not event.is_directory:
                    self.pending.add(event.src_path)

            def on_modified(self, event):
                if not event.is_directory:
                    self.pending.add(event.src_path)

        handler = Handler()
        observer = Observer()
        observer.schedule(handler, str(RAW_DIR), recursive=False)
        observer.start()
        logger.info("Watchdog observer started")

        ensure_dirs()
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            logger.error("ANTHROPIC_API_KEY not set")
            sys.exit(1)
        client = anthropic.Anthropic(api_key=api_key)
        known_hashes = load_hashes()

        try:
            while True:
                time.sleep(interval)
                if handler.pending:
                    paths = list(handler.pending)
                    handler.pending.clear()
                    for p in paths:
                        path = Path(p)
                        if path.exists() and path.suffix.lower() in (
                            SUPPORTED_TEXT | SUPPORTED_PDF | SUPPORTED_IMAGE | SUPPORTED_AUDIO | SUPPORTED_VIDEO
                        ):
                            try:
                                process_file(path, client, known_hashes)
                            except Exception as e:
                                logger.error(f"Watch error for {path}: {e}")
        except KeyboardInterrupt:
            observer.stop()
        observer.join()

    except ImportError:
        logger.warning("watchdog not available, falling back to polling")
        while True:
            run_once()
            time.sleep(interval)


def print_summary(stats: dict):
    print("\n" + "=" * 50)
    print("COMPILE SUMMARY")
    print("=" * 50)
    print(f"  Files processed:   {stats['processed']}")
    print(f"  Merged entries:    {stats['merged']}")
    print(f"  Duplicates skipped:{stats['duplicates']}")
    print(f"  New entries:       {stats['new_entries']}")
    print(f"  Errors:            {stats['errors']}")
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(description="LLM-Brains compiler")
    parser.add_argument("--watch", action="store_true", help="Continuous watch mode")
    parser.add_argument("--interval", type=int, default=5, help="Watch poll interval in seconds")
    args = parser.parse_args()

    ensure_dirs()

    if args.watch:
        watch_mode(args.interval)
    else:
        stats = run_once()
        print_summary(stats)


if __name__ == "__main__":
    main()
