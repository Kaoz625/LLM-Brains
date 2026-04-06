#!/usr/bin/env python3
"""
compile.py — LLM Brain Compiler
================================
Drop anything into brain/raw/ and this script:
  1. Detects file type (photo, audio, video, PDF, text, URL)
  2. Extracts content (vision / Whisper / text extraction)
  3. Uses an LLM to route, compile, and merge into your brain/ structure
  4. Deduplicates via hash + semantic similarity
  5. Cross-links concepts using [[wikilink]] syntax
  6. Updates brain/index.md master map

Usage:
    python compile.py            # one-shot: process everything in raw/
    python compile.py --watch    # continuous: watch raw/ for new files
    python compile.py --rebuild  # recompile everything from scratch
"""

import os
import sys
import json
import time
import hashlib
import logging
import argparse
import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional

# ---------------------------------------------------------------------------
# Optional / lazy imports — only error if the relevant path is actually used
# ---------------------------------------------------------------------------
try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

try:
    import frontmatter
    HAS_FRONTMATTER = True
except ImportError:
    HAS_FRONTMATTER = False

# ---------------------------------------------------------------------------
# Directory layout
# ---------------------------------------------------------------------------

ROOT       = Path(__file__).parent / "brain"
RAW        = ROOT / "raw"
PROCESSED  = RAW / "processed"
ME         = ROOT / "me"
WORK       = ROOT / "work"
KNOWLEDGE  = ROOT / "knowledge" / "concepts"
MEDIA      = ROOT / "media" / "transcripts"
INDEX      = ROOT / "index.md"
LOG_FILE   = ROOT / "compile.log"
HASH_DB    = ROOT / ".processed_hashes"

# Intuition flags land here
FLAGGED    = ROOT / "flagged"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("compile")

# ---------------------------------------------------------------------------
# Ensure directory structure exists
# ---------------------------------------------------------------------------

def ensure_dirs():
    for d in [RAW, PROCESSED, ME, WORK, KNOWLEDGE, MEDIA, FLAGGED,
              ROOT / "work" / "projects"]:
        d.mkdir(parents=True, exist_ok=True)
    if not INDEX.exists():
        INDEX.write_text("# Brain Index\n\nAuto-maintained by compile.py\n\n")
    if not HASH_DB.exists():
        HASH_DB.write_text("{}")

# ---------------------------------------------------------------------------
# Hash-based dedup
# ---------------------------------------------------------------------------

def load_hashes() -> dict:
    try:
        return json.loads(HASH_DB.read_text())
    except Exception:
        return {}

def save_hashes(h: dict):
    HASH_DB.write_text(json.dumps(h, indent=2))

def file_hash(path: Path) -> str:
    sha = hashlib.sha256()
    sha.update(path.read_bytes())
    return sha.hexdigest()

def already_processed(path: Path, hashes: dict) -> bool:
    h = file_hash(path)
    return hashes.get(str(path.name)) == h

def mark_processed(path: Path, hashes: dict):
    hashes[str(path.name)] = file_hash(path)
    save_hashes(hashes)

# ---------------------------------------------------------------------------
# Content extraction by file type
# ---------------------------------------------------------------------------

def extract_text_from_file(path: Path) -> Optional[str]:
    """Return raw text content from any supported file type."""
    suffix = path.suffix.lower()

    # --- Plain text / markdown ---
    if suffix in {".txt", ".md", ".rst", ".csv", ".json"}:
        return path.read_text(errors="replace")

    # --- PDF ---
    if suffix == ".pdf":
        try:
            import pymupdf  # pip install pymupdf
            doc = pymupdf.open(str(path))
            return "\n".join(page.get_text() for page in doc)
        except ImportError:
            log.warning("pymupdf not installed — skipping PDF %s", path.name)
            return None
        except Exception as e:
            log.error("PDF extraction failed for %s: %s", path.name, e)
            return None

    # --- Audio (mp3, wav, m4a, ogg) ---
    if suffix in {".mp3", ".wav", ".m4a", ".ogg", ".flac"}:
        return transcribe_audio(path)

    # --- Video (mp4, mov, mkv) ---
    if suffix in {".mp4", ".mov", ".mkv", ".avi", ".webm"}:
        audio_path = path.with_suffix(".mp3")
        try:
            import subprocess
            subprocess.run(
                ["ffmpeg", "-y", "-i", str(path), "-q:a", "0", "-map", "a", str(audio_path)],
                capture_output=True, check=True
            )
            text = transcribe_audio(audio_path)
            audio_path.unlink(missing_ok=True)
            return text
        except FileNotFoundError:
            log.warning("ffmpeg not found — skipping video %s", path.name)
            return None
        except Exception as e:
            log.error("Video extraction failed for %s: %s", path.name, e)
            return None

    # --- Image (jpg, png, webp, heic) ---
    if suffix in {".jpg", ".jpeg", ".png", ".webp", ".gif", ".heic"}:
        return describe_image(path)

    # --- URL file (a .url or .txt file containing a URL) ---
    if suffix == ".url":
        content = path.read_text().strip()
        if content.startswith("http"):
            return fetch_url_content(content)

    log.warning("Unsupported file type: %s", path.suffix)
    return None


def transcribe_audio(path: Path) -> Optional[str]:
    """Transcribe audio using faster-whisper (local, free)."""
    try:
        from faster_whisper import WhisperModel  # pip install faster-whisper
        model = WhisperModel("base", device="cpu", compute_type="int8")
        segments, _ = model.transcribe(str(path))
        return " ".join(seg.text for seg in segments)
    except ImportError:
        log.warning("faster-whisper not installed — trying openai-whisper fallback")
    except Exception as e:
        log.error("Whisper transcription failed for %s: %s", path.name, e)
        return None

    try:
        import whisper  # pip install openai-whisper
        model = whisper.load_model("base")
        result = model.transcribe(str(path))
        return result["text"]
    except ImportError:
        log.warning("openai-whisper not installed — skipping audio %s", path.name)
        return None


def describe_image(path: Path) -> Optional[str]:
    """Use Claude's vision to describe an image."""
    if not HAS_ANTHROPIC:
        log.warning("anthropic not installed — skipping image %s", path.name)
        return None
    try:
        import base64
        client = anthropic.Anthropic()
        img_b64 = base64.standard_b64encode(path.read_bytes()).decode("utf-8")
        suffix = path.suffix.lower().lstrip(".")
        media_type_map = {"jpg": "image/jpeg", "jpeg": "image/jpeg",
                          "png": "image/png", "webp": "image/webp", "gif": "image/gif"}
        media_type = media_type_map.get(suffix, "image/jpeg")
        response = client.messages.create(
            model="claude-sonnet-4-6",
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


def compile_with_llm(raw_text: str, source_path: Path,
                     existing_files: dict) -> Optional[dict]:
    """Send raw content to Claude for compilation. Returns structured output."""
    if not HAS_ANTHROPIC:
        log.error("anthropic SDK not installed. Run: pip install anthropic")
        return None

    client = anthropic.Anthropic()

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
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}]
        )
        raw_json = response.content[0].text.strip()
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
    if not HAS_ANTHROPIC:
        log.error("anthropic SDK required for intuition scan")
        return

    client = anthropic.Anthropic()

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


if __name__ == "__main__":
    main()
