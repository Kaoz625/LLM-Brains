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

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

from dotenv import load_dotenv

load_dotenv()

# LLM backend config: "anthropic" (default) or "ollama"
LLM_BACKEND = os.getenv("LLM_BACKEND", "anthropic").lower()
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma4:e4b")
OLLAMA_VISION_MODEL = os.getenv("OLLAMA_VISION_MODEL", "vision-moondream:latest")

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
                    {"type": "image", "source": {"type": "base64",
                                                  "media_type": media_type,
                                                  "data": img_b64}},
                    {"type": "text", "text": (
                        "Describe this image in detail. Include: people visible (names if known), "
                        "location/setting, objects, approximate date if guessable, mood/atmosphere, "
                        "and any text visible. Format as structured notes."
                    )}
                ]
            }]
        )
        return f"[IMAGE: {path.name}]\n{response.content[0].text}"
    except Exception as e:
        log.error("Image description failed for %s: %s", path.name, e)
        return None


def fetch_url_content(url: str) -> Optional[str]:
    """Fetch article text from a URL, with YouTube transcript support."""
    # YouTube
    if "youtube.com" in url or "youtu.be" in url:
        return fetch_youtube_transcript(url)

    # General web page
    try:
        import urllib.request
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        html = urllib.request.urlopen(req, timeout=15).read().decode("utf-8", errors="replace")
        # Strip HTML tags (simple approach)
        import re
        text = re.sub(r"<[^>]+>", " ", html)
        text = re.sub(r"\s+", " ", text).strip()
        return f"Source: {url}\n\n{text[:8000]}"  # cap at 8K chars
    except Exception as e:
        log.error("URL fetch failed for %s: %s", url, e)
        return None


def fetch_youtube_transcript(url: str) -> Optional[str]:
    """Fetch YouTube video transcript via youtube-transcript-api."""
    try:
        from youtube_transcript_api import YouTubeTranscriptApi  # pip install youtube-transcript-api
        import re
        # Extract video ID
        match = re.search(r"(?:v=|youtu\.be/)([A-Za-z0-9_-]{11})", url)
        if not match:
            log.error("Could not extract YouTube video ID from %s", url)
            return None
        video_id = match.group(1)
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        text = " ".join(t["text"] for t in transcript)
        return f"Source: {url}\nVideo ID: {video_id}\n\n{text}"
    except ImportError:
        log.warning("youtube-transcript-api not installed — skipping YouTube URL")
        return None
    except Exception as e:
        log.warning("YouTube transcript unavailable for %s (%s) — trying yt-dlp+whisper", url, e)
        return fetch_youtube_via_whisper(url)


def fetch_youtube_via_whisper(url: str) -> Optional[str]:
    """Fallback: download YouTube audio and transcribe with Whisper."""
    try:
        import subprocess, tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = Path(tmpdir) / "audio.mp3"
            subprocess.run(
                ["yt-dlp", "-x", "--audio-format", "mp3",
                 "-o", str(audio_path), url],
                capture_output=True, check=True
            )
            return transcribe_audio(audio_path)
    except FileNotFoundError:
        log.warning("yt-dlp not found — cannot download YouTube audio")
        return None
    except Exception as e:
        log.error("yt-dlp extraction failed: %s", e)
        return None

# ---------------------------------------------------------------------------
# LLM Compiler
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a personal knowledge compiler for a second-brain system.

Your job: read raw content and route/compile it into the correct folder structure.

FOLDER STRUCTURE:
- brain/me/          → personal identity, relationships, life events, preferences, emotions
- brain/work/        → projects, skills, businesses, professional knowledge
- brain/knowledge/   → general concepts, research, learnings, references, technical knowledge
- brain/media/       → transcribed media, image descriptions

RULES:
1. Extract ALL meaningful information — concepts, people, events, emotions, decisions
2. Route each piece to the MOST SPECIFIC folder and file
3. Use [[concept-name]] syntax to cross-link related ideas
4. Before writing anything, check: does this already exist? If yes, MERGE, don't duplicate
5. Write in clear, dense prose — no fluff, every sentence adds information
6. Add the source as a citation at the bottom of any new/updated file
7. Update brain/index.md with any new files created
8. Flag anything unusual or high-importance with [INTUITION FLAG] prefix

OUTPUT FORMAT (JSON):
{
  "files_to_write": [
    {
      "path": "brain/knowledge/concepts/topic-name.md",
      "action": "create|update|merge",
      "content": "full markdown content",
      "summary": "one-line description of what was added"
    }
  ],
  "index_updates": ["list of new file paths to add to index.md"],
  "intuition_flags": ["any high-importance items to surface to user"],
  "duplicate_detected": false,
  "duplicate_reason": ""
}
"""


def _call_ollama(
    system: str,
    user: str,
    model: Optional[str] = None,
    base_url: str = OLLAMA_BASE_URL,
) -> str:
    """Call Ollama local model. Returns response text."""
    import urllib.request

    model = model or OLLAMA_MODEL
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})

    payload = json.dumps(
        {"model": model, "messages": messages, "stream": False}
    ).encode()

    url = f"{base_url.rstrip('/')}/api/chat"
    req = urllib.request.Request(
        url, data=payload, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        data = json.loads(resp.read())
    return data.get("message", {}).get("content", "")


def compile_with_llm(raw_text: str, source_path: Path,
                     existing_files: dict) -> Optional[dict]:
    """Send raw content to LLM for compilation. Returns structured output."""
    # Build context: what already exists (titles only, to save tokens)
    existing_summary = "\n".join(
        f"- {path}" for path in list(existing_files.keys())[:50]
    )

    user_message = f"""SOURCE FILE: {source_path.name}
DATE: {datetime.now().strftime('%Y-%m-%d')}

EXISTING BRAIN FILES (titles):
{existing_summary}

RAW CONTENT TO COMPILE:
{raw_text[:6000]}

Compile this content into the brain structure. Check existing files for overlap.
Return valid JSON only."""

    try:
        if LLM_BACKEND == "ollama":
            raw_json = _call_ollama(SYSTEM_PROMPT, user_message).strip()
        elif HAS_ANTHROPIC:
            client = anthropic.Anthropic()
            response = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=4096,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_message}]
            )
            raw_json = response.content[0].text.strip()
        else:
            log.error("No LLM backend available. Install anthropic SDK or set LLM_BACKEND=ollama")
            return None
        # Strip markdown code fences if present
        if raw_json.startswith("```"):
            raw_json = raw_json.split("```")[1]
            if raw_json.startswith("json"):
                raw_json = raw_json[4:]
        return json.loads(raw_json)
    except json.JSONDecodeError as e:
        log.error("LLM returned invalid JSON: %s", e)
        return None
    except Exception as e:
        log.error("LLM compilation failed: %s", e)
        return None


def get_existing_files() -> dict:
    """Return dict of {relative_path: first_200_chars} for all brain files."""
    files = {}
    for path in ROOT.rglob("*.md"):
        if path == INDEX or "raw" in path.parts:
            continue
        try:
            files[str(path.relative_to(ROOT))] = path.read_text()[:200]
        except Exception:
            pass
    return files


def write_compiled_output(result: dict) -> int:
    """Write files from LLM output. Returns number of files written."""
    if not result or "files_to_write" not in result:
        return 0

    written = 0
    for file_spec in result.get("files_to_write", []):
        path = ROOT.parent / file_spec["path"]  # paths are relative to project root
        path.parent.mkdir(parents=True, exist_ok=True)

        action = file_spec.get("action", "create")
        content = file_spec.get("content", "")

        if action == "merge" and path.exists():
            existing = path.read_text()
            # Append new unique content rather than overwrite
            content = existing + "\n\n---\n*Updated: " + \
                      datetime.now().strftime('%Y-%m-%d') + "*\n\n" + content

        path.write_text(content)
        log.info("[%s] %s — %s", action.upper(), file_spec["path"],
                 file_spec.get("summary", ""))
        written += 1

    # Handle intuition flags
    for flag in result.get("intuition_flags", []):
        flag_path = FLAGGED / f"flag_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        flag_path.write_text(f"# Intuition Flag\n\n{flag}\n\n*Flagged: {datetime.now().isoformat()}*\n")
        log.warning("[INTUITION] %s", flag)

    return written


def update_index(new_paths: list):
    """Append new file paths to brain/index.md."""
    if not new_paths:
        return
    current = INDEX.read_text()
    additions = "\n".join(f"- [[{p}]]" for p in new_paths)
    INDEX.write_text(current + f"\n{additions}\n")

# ---------------------------------------------------------------------------
# Main processing loop
# ---------------------------------------------------------------------------

def process_file(path: Path, hashes: dict) -> dict:
    """Process a single raw file. Returns stats dict."""
    stats = {"processed": 0, "merged": 0, "skipped": 0, "flagged": 0}

    if already_processed(path, hashes):
        log.info("SKIP (already processed): %s", path.name)
        stats["skipped"] += 1
        return stats

    log.info("PROCESSING: %s", path.name)

    # Handle .url files (contains a URL as text)
    if path.suffix.lower() == ".url" or (
        path.suffix.lower() == ".txt" and
        path.read_text().strip().startswith("http")
    ):
        url = path.read_text().strip().splitlines()[0]
        raw_text = fetch_url_content(url)
    else:
        raw_text = extract_text_from_file(path)

    if not raw_text or not raw_text.strip():
        log.warning("No content extracted from %s", path.name)
        stats["skipped"] += 1
        mark_processed(path, hashes)
        return stats

    existing_files = get_existing_files()
    result = compile_with_llm(raw_text, path, existing_files)

    if result is None:
        log.error("Compilation failed for %s", path.name)
        return stats

    if result.get("duplicate_detected"):
        log.info("DUPLICATE: %s — %s", path.name, result.get("duplicate_reason", ""))
        stats["skipped"] += 1
    else:
        n_written = write_compiled_output(result)
        stats["processed"] += n_written
        stats["merged"] += sum(
            1 for f in result.get("files_to_write", []) if f.get("action") == "merge"
        )
        stats["flagged"] += len(result.get("intuition_flags", []))
        update_index(result.get("index_updates", []))

    # Move to processed/
    dest = PROCESSED / path.name
    shutil.move(str(path), str(dest))
    mark_processed(path, hashes)
    log.info("Moved %s → raw/processed/", path.name)

    return stats


def run_one_shot() -> dict:
    """Process all files currently in raw/."""
    ensure_dirs()
    hashes = load_hashes()
    total = {"processed": 0, "merged": 0, "skipped": 0, "flagged": 0}

    raw_files = [
        f for f in RAW.iterdir()
        if f.is_file() and not f.name.startswith(".")
    ]

    if not raw_files:
        log.info("No files in raw/ to process.")
        return total

    log.info("Found %d file(s) in raw/", len(raw_files))

    for path in sorted(raw_files):
        stats = process_file(path, hashes)
        for k in total:
            total[k] += stats[k]

    return total


def run_watch():
    """Continuously watch raw/ for new files."""
    log.info("Watching brain/raw/ for new files... (Ctrl+C to stop)")
    ensure_dirs()
    hashes = load_hashes()
    seen = set()

    while True:
        try:
            for path in RAW.iterdir():
                if path.is_file() and not path.name.startswith(".") and path not in seen:
                    seen.add(path)
                    process_file(path, hashes)
            time.sleep(5)
        except KeyboardInterrupt:
            log.info("Watch mode stopped.")
            break


# ---------------------------------------------------------------------------
# Intuition Engine
# ---------------------------------------------------------------------------

def run_intuition_scan():
    """
    Scan recent brain entries for anomalies vs. the user's baseline patterns.
    This is the 'Spidey Sense' layer — surfaces things that feel off or important.
    """
    if LLM_BACKEND != "ollama" and not HAS_ANTHROPIC:
        log.error("No LLM backend available. Install anthropic SDK or set LLM_BACKEND=ollama")
        return

    # Read recent files (last 7 days worth of changes)
    recent_files = []
    cutoff = time.time() - (7 * 86400)
    for path in ROOT.rglob("*.md"):
        if path.stat().st_mtime > cutoff and "raw" not in path.parts:
            recent_files.append((path, path.read_text()[:500]))

    if not recent_files:
        log.info("No recent files for intuition scan.")
        return

    recent_summary = "\n\n".join(
        f"FILE: {p.relative_to(ROOT)}\n{content}"
        for p, content in recent_files[:20]
    )

    prompt = f"""Review these recent brain entries and look for:
1. Patterns that are unusual or break from normal behavior
2. High-importance items that might need attention
3. Connections between seemingly unrelated entries
4. Anything that "feels off" — contradictions, gaps, anomalies
5. Emerging themes across multiple entries

RECENT ENTRIES:
{recent_summary}

Return a list of observations. Mark urgent ones with [URGENT]. Be like a trusted advisor
who reads between the lines."""

    try:
        if LLM_BACKEND == "ollama":
            intuition_note = _call_ollama("", prompt)
        else:
            client = anthropic.Anthropic()
            response = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}]
            )
            intuition_note = response.content[0].text
        scan_path = FLAGGED / f"intuition_scan_{datetime.now().strftime('%Y%m%d')}.md"
        scan_path.write_text(
            f"# Intuition Scan — {datetime.now().strftime('%Y-%m-%d')}\n\n{intuition_note}\n"
        )
        log.info("Intuition scan written to %s", scan_path)
        print("\n[INTUITION SCAN]\n" + intuition_note)
    except Exception as e:
        log.error("Intuition scan failed: %s", e)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="LLM Brain Compiler — compile raw/ into your personal knowledge brain"
    )
    parser.add_argument("--watch", action="store_true",
                        help="Continuously watch raw/ for new files")
    parser.add_argument("--rebuild", action="store_true",
                        help="Clear processed hashes and reprocess everything")
    parser.add_argument("--intuition", action="store_true",
                        help="Run intuition scan on recent entries")
    parser.add_argument("--brain-dir", type=str,
                        help="Override brain directory path")
    args = parser.parse_args()

    if args.brain_dir:
        global ROOT, RAW, PROCESSED, ME, WORK, KNOWLEDGE, MEDIA, INDEX, LOG_FILE, HASH_DB, FLAGGED
        ROOT      = Path(args.brain_dir)
        RAW       = ROOT / "raw"
        PROCESSED = RAW / "processed"
        ME        = ROOT / "me"
        WORK      = ROOT / "work"
        KNOWLEDGE = ROOT / "knowledge" / "concepts"
        MEDIA     = ROOT / "media" / "transcripts"
        INDEX     = ROOT / "index.md"
        LOG_FILE  = ROOT / "compile.log"
        HASH_DB   = ROOT / ".processed_hashes"
        FLAGGED   = ROOT / "flagged"

    if args.rebuild:
        log.info("Rebuild mode: clearing processed hashes")
        HASH_DB.write_text("{}")

    if args.intuition:
        run_intuition_scan()
        return

    if args.watch:
        run_watch()
    else:
        stats = run_one_shot()
        print(f"\n{'='*50}")
        print(f"  Brain Compiler Summary")
        print(f"{'='*50}")
        print(f"  Files processed : {stats['processed']}")
        print(f"  Entries merged  : {stats['merged']}")
        print(f"  Duplicates skip : {stats['skipped']}")
        print(f"  Intuition flags : {stats['flagged']}")
        print(f"{'='*50}")
        if stats["flagged"] > 0:
            print(f"\n  Check brain/flagged/ for {stats['flagged']} flagged item(s)!")


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
        if LLM_BACKEND == "ollama":
            text = _call_ollama("", prompt)
        elif HAS_ANTHROPIC:
            response = client.messages.create(
                model="claude-opus-4-5",
                max_tokens=2048,
                messages=[{"role": "user", "content": prompt}]
            )
            text = response.content[0].text
        else:
            log.error("No LLM backend available")
            return {"title": source_name, "summary": content[:200], "route": route_content(content),
                    "markdown": content, "wikilinks": [], "concepts": [], "events": [], "people": [],
                    "dates": [], "tags": []}
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
