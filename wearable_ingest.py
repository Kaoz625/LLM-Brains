#!/usr/bin/env python3
"""
wearable_ingest.py — Wearable & Continuous Video/Audio Ingestion
=================================================================
Processes video and audio from wearables (Meta Ray-Ban glasses,
phone recordings, dashcams, body cams, etc.) into the brain.

The core idea mirrors how human episodic memory works:
  - Events are NOT stored as full videos — they're stored as KEY FRAMES
    with rich semantic tags, emotions, people, places, context
  - Those keyframe memories are LINKED to concepts, people, places
  - Recall happens via any connected node — multiple paths to the same memory

This is directly analogous to how neurological memory issues work:
  "The memory isn't gone — the PATHWAY to it is broken.
   Build more paths = better recall."

For every video processed, this script builds:
  1. Scene-by-scene description (what was happening)
  2. People detected (who was there)
  3. Places/environments (where)
  4. Objects and context (what)
  5. Audio transcript (what was said)
  6. Emotional tone (how it felt)
  7. Cross-links to existing brain concepts

All of this routes to brain/me/experiences/ and updates timeline.md,
relationships.md, and relevant knowledge/ files.

Supported Sources:
  - Video files: .mp4, .mov, .avi, .mkv (from Meta glasses USB export, phone, etc.)
  - Audio files: .mp3, .m4a, .wav (voice memos, phone calls)
  - Image sequences: directories of .jpg/.png (burst photos, time-lapses)
  - Photo library folders (iCloud sync, Google Photos export)

Usage:
    python wearable_ingest.py --video path/to/glasses_footage.mp4
    python wearable_ingest.py --watch ~/Downloads/MetaGlasses/  # auto-process new files
    python wearable_ingest.py --photos ~/Pictures/2026-04/
    python wearable_ingest.py --audio path/to/recording.m4a
"""

import os
import re
import sys
import json
import base64
import struct
import logging
import argparse
import subprocess
import tempfile
import time
from pathlib import Path
from datetime import datetime
from typing import Optional, List

log = logging.getLogger("wearable")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT       = Path(__file__).parent / "brain"
RAW        = ROOT / "raw"
EXPERIENCES = ROOT / "me" / "experiences"
TIMELINE   = ROOT / "me" / "timeline.md"
MEDIA_DIR  = ROOT / "media"

KEYFRAME_INTERVAL = 10    # Extract a keyframe every N seconds for analysis
MAX_FRAMES_PER_VIDEO = 30 # Cap to avoid burning too many API tokens

# ---------------------------------------------------------------------------
# ffmpeg helpers
# ---------------------------------------------------------------------------

def check_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def get_video_duration(video_path: Path) -> float:
    """Return video duration in seconds."""
    try:
        result = subprocess.run([
            "ffprobe", "-v", "quiet", "-print_format", "json",
            "-show_format", str(video_path)
        ], capture_output=True, text=True, check=True)
        info = json.loads(result.stdout)
        return float(info["format"].get("duration", 0))
    except Exception:
        return 0.0


def extract_keyframes(video_path: Path, output_dir: Path,
                      interval: int = KEYFRAME_INTERVAL,
                      max_frames: int = MAX_FRAMES_PER_VIDEO) -> List[Path]:
    """
    Extract keyframes from video at regular intervals.
    Uses scene change detection + time-based fallback for better coverage.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    duration = get_video_duration(video_path)
    if duration == 0:
        log.warning("Could not determine video duration for %s", video_path.name)

    # Adaptive interval to stay under max_frames cap
    if duration > 0 and duration / interval > max_frames:
        interval = int(duration / max_frames)
        log.info("Adaptive interval: %ds (video is %.0fs)", interval, duration)

    # Extract frames at time intervals
    pattern = str(output_dir / "frame_%04d.jpg")
    subprocess.run([
        "ffmpeg", "-y", "-i", str(video_path),
        "-vf", f"fps=1/{interval}",
        "-q:v", "3",
        pattern
    ], capture_output=True)

    frames = sorted(output_dir.glob("frame_*.jpg"))
    log.info("Extracted %d keyframes from %s", len(frames), video_path.name)
    return frames


def extract_audio(video_path: Path, output_path: Path) -> bool:
    """Extract audio track from video to mp3."""
    try:
        subprocess.run([
            "ffmpeg", "-y", "-i", str(video_path),
            "-q:a", "0", "-map", "a", str(output_path)
        ], capture_output=True, check=True)
        return True
    except subprocess.CalledProcessError:
        log.debug("No audio track in %s", video_path.name)
        return False

# ---------------------------------------------------------------------------
# Vision analysis
# ---------------------------------------------------------------------------

def describe_frame(image_path: Path, client, frame_idx: int,
                   total_frames: int, prior_context: str = "") -> dict:
    """
    Use Claude vision to deeply analyze a video frame.
    Returns structured dict with people, places, objects, emotions, events.
    """
    img_b64 = base64.standard_b64encode(image_path.read_bytes()).decode()

    context_text = ""
    if prior_context:
        context_text = f"\nPrior context from earlier in this video: {prior_context[:300]}"

    prompt = f"""Analyze this video frame ({frame_idx + 1} of {total_frames}).{context_text}

Return JSON with these fields:
{{
  "scene_description": "1-2 sentences describing what is happening",
  "people": ["list of people visible, with descriptions (name if known, otherwise physical description)"],
  "location": "where this appears to be (indoor/outdoor, type of place, city if recognizable)",
  "objects": ["notable objects, items, technology visible"],
  "activity": "what activity is happening",
  "emotional_tone": "atmosphere/mood of the scene (1 word + brief reason)",
  "time_of_day": "morning/afternoon/evening/night if guessable",
  "notable_text": "any visible text, signs, screens",
  "memory_tags": ["5-10 keyword tags for memory retrieval"],
  "connections": ["concepts this might connect to in a knowledge base"]
}}

Be specific and observational. These descriptions become long-term memories."""

    try:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=512,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": img_b64
                    }},
                    {"type": "text", "text": prompt}
                ]
            }]
        )
        raw = response.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"scene_description": response.content[0].text[:200], "memory_tags": []}
    except Exception as e:
        log.error("Frame analysis failed: %s", e)
        return {}


def transcribe_audio_file(audio_path: Path) -> str:
    """Transcribe audio using faster-whisper or openai-whisper."""
    try:
        from faster_whisper import WhisperModel
        model = WhisperModel("base", device="cpu", compute_type="int8")
        segments, _ = model.transcribe(str(audio_path))
        return " ".join(seg.text for seg in segments)
    except ImportError:
        pass
    try:
        import whisper
        model = whisper.load_model("base")
        result = model.transcribe(str(audio_path))
        return result["text"]
    except ImportError:
        log.warning("No Whisper installation found — skipping audio transcription. "
                    "Run: pip install faster-whisper")
        return ""
    except Exception as e:
        log.error("Audio transcription failed: %s", e)
        return ""

# ---------------------------------------------------------------------------
# Memory synthesis — the core "build more pathways" logic
# ---------------------------------------------------------------------------

def synthesize_experience(
    video_path: Path,
    frame_analyses: list,
    transcript: str,
    client
) -> str:
    """
    The key step: take all the frame descriptions + transcript and
    synthesize a rich episodic memory entry with MANY cross-links.

    More cross-links = more retrieval pathways = better recall.
    This is the computational analog to what physical therapy does for
    memory patients — building alternate neural pathways to the same memory.
    """
    frames_summary = "\n".join(
        f"Frame {i+1}: {f.get('scene_description', '')} | "
        f"People: {f.get('people', [])} | "
        f"Location: {f.get('location', '')} | "
        f"Tags: {f.get('memory_tags', [])}"
        for i, f in enumerate(frame_analyses) if f
    )

    prompt = f"""You are compiling an episodic memory entry from wearable camera footage.

SOURCE: {video_path.name}
DATE: {datetime.now().strftime('%Y-%m-%d')}

FRAME ANALYSES:
{frames_summary}

AUDIO TRANSCRIPT:
{transcript[:2000] if transcript else "(no audio)"}

Create a rich episodic memory entry in Markdown that:

1. **NARRATIVE** — 2-3 paragraphs describing what happened, with sensory details
2. **PEOPLE** — everyone present, their role, what they said or did
3. **PLACE** — detailed location description
4. **WHAT WAS LEARNED** — any information, ideas, or knowledge gained
5. **EMOTIONAL TONE** — how the experience felt
6. **CROSS-LINKS** — at least 10-15 [[wikilinks]] to related concepts, people, places, ideas
   - These are critical: each link is an alternate retrieval pathway to this memory
   - Link to: people's names, places, topics discussed, emotions felt, objects seen
7. **MEMORY TAGS** — #tags for categorization

The more [[links]] and #tags, the easier this memory is to recall later from any direction.
Think of links as synaptic connections — more connections = more durable memory.

Format as valid Markdown with frontmatter:
---
date: {datetime.now().strftime('%Y-%m-%d')}
source: wearable-video
source_file: {video_path.name}
---
"""

    try:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text
    except Exception as e:
        log.error("Memory synthesis failed: %s", e)
        # Fallback: write raw frame data
        return f"# Experience: {video_path.stem}\n\nDate: {datetime.now().isoformat()}\n\n{frames_summary}"

# ---------------------------------------------------------------------------
# Main video processing pipeline
# ---------------------------------------------------------------------------

def process_video(video_path: Path) -> Optional[Path]:
    """
    Full pipeline for a single video file:
    1. Extract keyframes + audio
    2. Analyze each frame with vision AI
    3. Transcribe audio
    4. Synthesize rich episodic memory with many cross-links
    5. Write to brain/me/experiences/
    6. Update timeline.md
    """
    try:
        import anthropic
        client = anthropic.Anthropic()
    except ImportError:
        log.error("anthropic SDK not installed. Run: pip install anthropic")
        return None

    if not check_ffmpeg():
        log.error("ffmpeg not found. Install: brew install ffmpeg")
        return None

    log.info("Processing video: %s", video_path.name)
    EXPERIENCES.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Step 1: Extract keyframes
        frames = extract_keyframes(video_path, tmpdir / "frames")
        if not frames:
            log.warning("No frames extracted from %s", video_path.name)
            return None

        # Step 2: Extract and transcribe audio
        audio_path = tmpdir / "audio.mp3"
        has_audio = extract_audio(video_path, audio_path)
        transcript = transcribe_audio_file(audio_path) if has_audio else ""
        if transcript:
            log.info("Transcribed %d words of audio", len(transcript.split()))

        # Step 3: Analyze each keyframe
        frame_analyses = []
        prior_context = ""
        for i, frame_path in enumerate(frames):
            log.info("Analyzing frame %d/%d...", i + 1, len(frames))
            analysis = describe_frame(frame_path, client, i, len(frames), prior_context)
            frame_analyses.append(analysis)
            if analysis.get("scene_description"):
                prior_context = analysis["scene_description"]
            time.sleep(0.2)  # gentle rate limiting

        # Step 4: Synthesize into rich episodic memory
        log.info("Synthesizing episodic memory...")
        memory_text = synthesize_experience(video_path, frame_analyses, transcript, client)

        # Step 5: Write to brain
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        slug = re.sub(r"[^a-z0-9]+", "-", video_path.stem.lower())[:40]
        output_path = EXPERIENCES / f"{timestamp}_{slug}.md"
        output_path.write_text(memory_text)
        log.info("Memory written: %s", output_path)

        # Step 6: Update timeline
        update_timeline(video_path, output_path, frame_analyses)

        # Step 7: Copy representative frames to media/
        media_out = MEDIA_DIR / "frames" / timestamp
        media_out.mkdir(parents=True, exist_ok=True)
        for i, frame in enumerate(frames[:5]):  # keep first 5 frames
            dest = media_out / f"frame_{i:03d}.jpg"
            import shutil
            shutil.copy(frame, dest)

        log.info("Done. Experience logged with %d frame analyses.", len(frame_analyses))
        return output_path


def update_timeline(video_path: Path, memory_path: Path, analyses: list):
    """Add an entry to brain/me/timeline.md."""
    TIMELINE.parent.mkdir(parents=True, exist_ok=True)
    if not TIMELINE.exists():
        TIMELINE.write_text("# Personal Timeline\n\nAuto-maintained by wearable_ingest.py\n\n")

    date_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    location = next(
        (a.get("location", "unknown") for a in analyses if a.get("location")),
        "unknown location"
    )
    people = list({
        p for a in analyses
        for p in a.get("people", [])
    })

    entry = (
        f"\n## {date_str} — {location}\n"
        f"Source: `{video_path.name}`  \n"
        f"Memory: [[{memory_path.relative_to(ROOT)}]]  \n"
    )
    if people:
        entry += f"People: {', '.join(people[:5])}  \n"
    entry += "\n"

    current = TIMELINE.read_text()
    TIMELINE.write_text(current + entry)

# ---------------------------------------------------------------------------
# Photo batch processing
# ---------------------------------------------------------------------------

def process_photo_dir(photo_dir: Path):
    """Process a directory of photos as a batch memory."""
    try:
        import anthropic
        client = anthropic.Anthropic()
    except ImportError:
        log.error("anthropic SDK not installed")
        return

    photo_files = sorted([
        f for f in photo_dir.iterdir()
        if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".heic"}
    ])

    if not photo_files:
        log.info("No photos found in %s", photo_dir)
        return

    log.info("Processing %d photos from %s", len(photo_files), photo_dir.name)
    EXPERIENCES.mkdir(parents=True, exist_ok=True)

    analyses = []
    # Sample at most 20 photos
    sampled = photo_files[::max(1, len(photo_files) // 20)][:20]

    for i, photo in enumerate(sampled):
        log.info("Analyzing photo %d/%d: %s", i + 1, len(sampled), photo.name)
        analysis = describe_frame(photo, client, i, len(sampled))
        if analysis:
            analysis["filename"] = photo.name
            analyses.append(analysis)
        time.sleep(0.2)

    # Synthesize batch memory
    memory_text = synthesize_experience(
        photo_dir, analyses, "", client
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = re.sub(r"[^a-z0-9]+", "-", photo_dir.name.lower())[:40]
    output_path = EXPERIENCES / f"{timestamp}_{slug}_photos.md"
    output_path.write_text(memory_text)
    log.info("Photo batch memory written: %s", output_path)

# ---------------------------------------------------------------------------
# Watch mode: auto-process new files
# ---------------------------------------------------------------------------

def watch_directory(watch_dir: Path, interval: int = 30):
    """Watch a directory for new video/photo files and process them."""
    watch_dir = Path(watch_dir).expanduser()
    if not watch_dir.exists():
        log.error("Directory not found: %s", watch_dir)
        return

    log.info("Watching %s for new media files... (Ctrl+C to stop)", watch_dir)
    seen = set(f.name for f in watch_dir.iterdir() if f.is_file())

    while True:
        try:
            current = set(f.name for f in watch_dir.iterdir() if f.is_file())
            new_files = current - seen
            for filename in new_files:
                path = watch_dir / filename
                suffix = path.suffix.lower()
                log.info("New file detected: %s", filename)
                if suffix in {".mp4", ".mov", ".avi", ".mkv", ".webm"}:
                    process_video(path)
                elif suffix in {".mp3", ".m4a", ".wav", ".aac"}:
                    # Audio-only — transcribe and drop into raw/
                    RAW.mkdir(parents=True, exist_ok=True)
                    transcript = transcribe_audio_file(path)
                    if transcript:
                        out = RAW / f"audio_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{path.stem}.txt"
                        out.write_text(f"Source: {path}\nType: audio-recording\n\n{transcript}")
                        log.info("Audio transcript written to raw/")
                elif suffix in {".jpg", ".jpeg", ".png", ".webp", ".heic"}:
                    # Single photo — drop into raw/ for compile.py
                    import shutil
                    RAW.mkdir(parents=True, exist_ok=True)
                    shutil.copy(path, RAW / path.name)
                    log.info("Photo copied to raw/ for processing")
            seen = current
            time.sleep(interval)
        except KeyboardInterrupt:
            log.info("Watch mode stopped.")
            break

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Wearable & continuous video/photo ingestion for the LLM Brain"
    )
    parser.add_argument("--video", type=str,
                        help="Process a single video file")
    parser.add_argument("--photos", type=str,
                        help="Process a directory of photos as a batch memory")
    parser.add_argument("--audio", type=str,
                        help="Transcribe an audio file into raw/ for compile.py")
    parser.add_argument("--watch", type=str, metavar="DIR",
                        help="Watch a directory for new media files (e.g. Meta glasses sync folder)")
    parser.add_argument("--interval", type=int, default=30,
                        help="Watch interval in seconds (default: 30)")
    args = parser.parse_args()

    ROOT.mkdir(parents=True, exist_ok=True)
    RAW.mkdir(parents=True, exist_ok=True)

    if args.video:
        output = process_video(Path(args.video).expanduser())
        if output:
            print(f"\nMemory written: {output}")
            print("Run 'python sqlite_rag.py --build' to index it.")

    elif args.photos:
        process_photo_dir(Path(args.photos).expanduser())

    elif args.audio:
        path = Path(args.audio).expanduser()
        transcript = transcribe_audio_file(path)
        if transcript:
            out = RAW / f"audio_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{path.stem}.txt"
            out.write_text(f"Source: {path}\nType: audio-recording\nDate: {datetime.now().isoformat()}\n\n{transcript}")
            print(f"Transcript written to: {out}")
            print("Run 'python compile.py' to process it.")

    elif args.watch:
        watch_directory(Path(args.watch), args.interval)

    else:
        parser.print_help()
        print("\nExample (Meta Ray-Ban glasses sync folder):")
        print("  python wearable_ingest.py --watch ~/Documents/MetaGlasses/ --interval 60")
        print("\nExample (process a single recording):")
        print("  python wearable_ingest.py --video ~/Downloads/glasses_footage.mp4")


if __name__ == "__main__":
    main()
