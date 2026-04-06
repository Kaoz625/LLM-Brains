#!/usr/bin/env python3
"""
wearable_ingest.py — Ingests wearable camera footage (Meta Glasses, phone video, etc.)

Watches a sync folder for new video/photo files, extracts keyframes, transcribes
audio with Whisper, and synthesizes episodic memories with [[wikilinks]].

Usage:
    python wearable_ingest.py --watch ~/Documents/MetaGlasses/ --interval 60
    python wearable_ingest.py --file video.mp4
"""

import argparse
import base64
import json
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
EXPERIENCES_DIR = BRAIN_DIR / "me" / "experiences"
TIMELINE_FILE = BRAIN_DIR / "me" / "timeline.md"

SUPPORTED_VIDEO = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}
SUPPORTED_IMAGE = {".jpg", ".jpeg", ".png", ".webp", ".heic"}


def ensure_dirs():
    EXPERIENCES_DIR.mkdir(parents=True, exist_ok=True)


def get_client() -> anthropic.Anthropic:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("ANTHROPIC_API_KEY not set", file=sys.stderr)
        sys.exit(1)
    return anthropic.Anthropic(api_key=api_key)


# ---------------------------------------------------------------------------
# Frame extraction
# ---------------------------------------------------------------------------

def extract_keyframes(video_path: Path, interval_secs: int = 10,
                      output_dir: Optional[Path] = None) -> list[Path]:
    """Extract keyframes from video at adaptive interval using ffmpeg."""
    if output_dir is None:
        output_dir = video_path.parent / f"frames_{video_path.stem}"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_pattern = str(output_dir / "frame_%04d.jpg")
    try:
        result = subprocess.run(
            [
                "ffmpeg", "-y", "-i", str(video_path),
                "-vf", f"fps=1/{interval_secs},scale=1280:-1",
                "-q:v", "3",
                output_pattern
            ],
            capture_output=True, text=True, timeout=600
        )
        if result.returncode != 0:
            print(f"  ffmpeg error: {result.stderr[:200]}", file=sys.stderr)
            return []
        frames = sorted(output_dir.glob("frame_*.jpg"))
        print(f"  Extracted {len(frames)} keyframes (1 per {interval_secs}s)")
        return frames
    except FileNotFoundError:
        print("  ffmpeg not found — skipping frame extraction", file=sys.stderr)
        return []
    except subprocess.TimeoutExpired:
        print("  ffmpeg timed out", file=sys.stderr)
        return []


def get_video_duration(video_path: Path) -> float:
    """Get video duration in seconds using ffprobe."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "json", str(video_path)],
            capture_output=True, text=True, timeout=30
        )
        data = json.loads(result.stdout)
        return float(data["format"]["duration"])
    except Exception:
        return 0.0


def adaptive_interval(duration_secs: float) -> int:
    """Choose keyframe interval based on video length."""
    if duration_secs <= 60:
        return 5
    elif duration_secs <= 300:
        return 10
    elif duration_secs <= 1800:
        return 30
    else:
        return 60


# ---------------------------------------------------------------------------
# Audio transcription
# ---------------------------------------------------------------------------

def transcribe_video_audio(video_path: Path) -> str:
    """Extract and transcribe audio from video."""
    audio_path = video_path.with_suffix(".tmp.wav")
    try:
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", str(video_path), "-vn", "-acodec", "pcm_s16le",
             "-ar", "16000", "-ac", "1", str(audio_path)],
            capture_output=True, text=True, timeout=300
        )
        if result.returncode != 0 or not audio_path.exists():
            return "[Audio extraction failed]"

        try:
            from faster_whisper import WhisperModel
            model = WhisperModel("base", device="cpu", compute_type="int8")
            segments, info = model.transcribe(str(audio_path))
            transcript = " ".join(seg.text.strip() for seg in segments)
            print(f"  Transcribed: {len(transcript)} chars")
            return transcript
        except ImportError:
            return "[faster-whisper not available]"
        except Exception as e:
            return f"[Transcription error: {e}]"
    finally:
        if audio_path.exists():
            audio_path.unlink()


# ---------------------------------------------------------------------------
# Vision analysis
# ---------------------------------------------------------------------------

def analyze_frames_with_claude(client: anthropic.Anthropic,
                                frames: list[Path], max_frames: int = 10) -> list[dict]:
    """Analyze keyframes using Claude vision, sampling evenly."""
    if not frames:
        return []

    # Sample evenly if too many frames
    if len(frames) > max_frames:
        step = len(frames) // max_frames
        sampled = frames[::step][:max_frames]
    else:
        sampled = frames

    frame_analyses = []
    for i, frame_path in enumerate(sampled):
        try:
            with open(frame_path, "rb") as f:
                img_data = base64.standard_b64encode(f.read()).decode("utf-8")

            response = client.messages.create(
                model="claude-opus-4-5",
                max_tokens=512,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {"type": "base64", "media_type": "image/jpeg", "data": img_data}
                        },
                        {
                            "type": "text",
                            "text": (
                                "Describe what you see in this video frame. "
                                "Include: location/setting, people, objects, activities, "
                                "time of day, mood/atmosphere. Be specific and factual."
                            )
                        }
                    ]
                }]
            )
            analysis = response.content[0].text
            frame_analyses.append({
                "frame": frame_path.name,
                "index": i,
                "analysis": analysis
            })
        except Exception as e:
            frame_analyses.append({
                "frame": frame_path.name,
                "index": i,
                "analysis": f"[Analysis failed: {e}]"
            })

    return frame_analyses


def analyze_single_image(client: anthropic.Anthropic, image_path: Path) -> str:
    """Analyze a single image file."""
    suffix = image_path.suffix.lower()
    media_map = {".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                 ".png": "image/png", ".webp": "image/webp"}
    media_type = media_map.get(suffix, "image/jpeg")

    try:
        with open(image_path, "rb") as f:
            img_data = base64.standard_b64encode(f.read()).decode("utf-8")

        response = client.messages.create(
            model="claude-opus-4-5",
            max_tokens=1024,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image",
                     "source": {"type": "base64", "media_type": media_type, "data": img_data}},
                    {"type": "text", "text": (
                        "Analyze this photo in detail. Describe: location, people present, "
                        "activities, objects, time of day, season, mood, and any text visible. "
                        "Format as structured notes for a personal memory archive."
                    )}
                ]
            }]
        )
        return response.content[0].text
    except Exception as e:
        return f"[Image analysis failed: {e}]"


# ---------------------------------------------------------------------------
# Memory synthesis
# ---------------------------------------------------------------------------

def synthesize_episodic_memory(client: anthropic.Anthropic,
                                frame_analyses: list[dict],
                                transcript: str,
                                source_file: str) -> dict:
    """Synthesize frame analyses + transcript into episodic memory entry."""
    frames_summary = "\n".join(
        f"Frame {a['index']}: {a['analysis'][:200]}" for a in frame_analyses
    )

    prompt = f"""You are compiling an episodic memory from wearable camera footage.

Source: {source_file}
Date: {datetime.now().strftime('%Y-%m-%d')}

Visual frames ({len(frame_analyses)} keyframes):
{frames_summary[:3000]}

Audio transcript:
{transcript[:2000]}

Synthesize this into a rich episodic memory entry with:
1. A clear narrative title
2. When/where this happened
3. Who was involved
4. What was happening
5. The emotional tone and significance
6. At least 10 [[wikilinks]] for people, places, concepts, activities

Return JSON:
{{
  "title": "Descriptive episode title",
  "date": "YYYY-MM-DD or estimated",
  "location": "location if determinable",
  "people": ["person1", "person2"],
  "summary": "2-3 paragraph narrative description",
  "wikilinks": ["[[person]]", "[[place]]", "[[activity]]"],
  "tags": ["tag1", "tag2"],
  "emotional_tone": "description of mood/feeling",
  "significance": "why this moment matters",
  "markdown": "full formatted memory entry with [[wikilinks]]"
}}"""

    try:
        response = client.messages.create(
            model="claude-opus-4-5",
            max_tokens=3000,
            messages=[{"role": "user", "content": prompt}]
        )
        text = response.content[0].text
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            return json.loads(match.group())
    except Exception as e:
        print(f"  Memory synthesis error: {e}", file=sys.stderr)

    return {
        "title": f"Experience: {Path(source_file).stem}",
        "date": datetime.now().strftime("%Y-%m-%d"),
        "location": "Unknown",
        "people": [],
        "summary": f"Recorded experience from {source_file}",
        "wikilinks": [],
        "tags": ["experience"],
        "emotional_tone": "neutral",
        "significance": "Recorded via wearable camera",
        "markdown": f"# Experience: {source_file}\n\nAudio: {transcript[:500]}\n\nFrames analyzed: {len(frame_analyses)}"
    }


def save_experience(memory: dict, source_path: Path) -> Path:
    """Save episodic memory to experiences directory."""
    EXPERIENCES_DIR.mkdir(parents=True, exist_ok=True)
    date_str = memory.get("date", datetime.now().strftime("%Y-%m-%d")).replace("/", "-")[:10]
    safe_title = re.sub(r"[^\w\s-]", "", memory.get("title", "experience"))
    safe_title = safe_title.strip().replace(" ", "_")[:60]
    filename = f"{date_str}_{safe_title}.md"
    out_path = EXPERIENCES_DIR / filename

    wikilinks_str = " ".join(memory.get("wikilinks", []))
    tags_str = json.dumps(memory.get("tags", []))

    content = f"""---
title: {memory.get('title', 'Experience')}
date: {memory.get('date', date_str)}
location: {memory.get('location', 'Unknown')}
people: {json.dumps(memory.get('people', []))}
tags: {tags_str}
source: {source_path.name}
emotional_tone: {memory.get('emotional_tone', '')}
created_at: {datetime.now().isoformat()}
---

# {memory.get('title', 'Experience')}

**Date:** {memory.get('date', date_str)}
**Location:** {memory.get('location', 'Unknown')}
**People:** {', '.join(memory.get('people', []))}
**Tone:** {memory.get('emotional_tone', '')}

**Links:** {wikilinks_str}

---

{memory.get('markdown', memory.get('summary', ''))}

---
**Significance:** {memory.get('significance', '')}

*Captured: {source_path.name} | Processed: {datetime.now().strftime('%Y-%m-%d %H:%M')}*
"""
    out_path.write_text(content, encoding="utf-8")
    return out_path


def update_timeline(memory: dict, experience_path: Path):
    """Append experience to personal timeline."""
    if not TIMELINE_FILE.exists():
        TIMELINE_FILE.write_text("# Personal Timeline\n\n", encoding="utf-8")

    existing = TIMELINE_FILE.read_text(encoding="utf-8")
    date = memory.get("date", datetime.now().strftime("%Y-%m-%d"))
    title = memory.get("title", "Experience")
    location = memory.get("location", "")
    people = ", ".join(memory.get("people", []))
    wikilinks = " ".join(memory.get("wikilinks", [])[:5])

    rel_path = experience_path.relative_to(BRAIN_DIR)
    entry = (
        f"\n### {date} — [{title}]({rel_path})\n"
        f"**Location:** {location} | **People:** {people}\n"
        f"{memory.get('summary', '')[:200]}\n"
        f"*{wikilinks}*\n"
    )
    TIMELINE_FILE.write_text(existing + entry, encoding="utf-8")


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------

def process_video(video_path: Path, client: anthropic.Anthropic) -> Optional[dict]:
    """Process a single video file into an episodic memory."""
    print(f"\nProcessing video: {video_path.name}")
    ensure_dirs()

    # Get duration and choose interval
    duration = get_video_duration(video_path)
    interval = adaptive_interval(duration)
    print(f"  Duration: {duration:.0f}s, keyframe interval: {interval}s")

    # Extract keyframes
    frames_dir = video_path.parent / f"_frames_{video_path.stem}"
    frames = extract_keyframes(video_path, interval_secs=interval, output_dir=frames_dir)

    # Transcribe audio
    print("  Transcribing audio...")
    transcript = transcribe_video_audio(video_path)

    # Analyze frames
    print(f"  Analyzing {len(frames)} frames with Claude vision...")
    frame_analyses = analyze_frames_with_claude(client, frames, max_frames=12)

    # Synthesize memory
    print("  Synthesizing episodic memory...")
    memory = synthesize_episodic_memory(client, frame_analyses, transcript, video_path.name)

    # Save
    exp_path = save_experience(memory, video_path)
    update_timeline(memory, exp_path)

    # Cleanup temp frames
    if frames_dir.exists():
        shutil.rmtree(str(frames_dir), ignore_errors=True)

    print(f"  -> Saved: {exp_path.relative_to(BRAIN_DIR)}")
    return memory


def process_image(image_path: Path, client: anthropic.Anthropic) -> Optional[dict]:
    """Process a single image file."""
    print(f"\nProcessing image: {image_path.name}")
    ensure_dirs()

    analysis = analyze_single_image(client, image_path)
    memory = {
        "title": f"Photo: {image_path.stem}",
        "date": datetime.now().strftime("%Y-%m-%d"),
        "location": "Unknown",
        "people": [],
        "summary": analysis[:500],
        "wikilinks": re.findall(r'\[\[([^\]]+)\]\]', analysis),
        "tags": ["photo", "visual"],
        "emotional_tone": "neutral",
        "significance": "Wearable camera capture",
        "markdown": analysis,
    }
    exp_path = save_experience(memory, image_path)
    update_timeline(memory, exp_path)
    print(f"  -> Saved: {exp_path.relative_to(BRAIN_DIR)}")
    return memory


def watch_directory(watch_dir: Path, interval: int = 60, client: Optional[anthropic.Anthropic] = None):
    """Watch a directory for new video/photo files and process them."""
    if client is None:
        client = get_client()

    processed_files: set[str] = set()
    print(f"\nWatching {watch_dir} every {interval}s...")
    print("Press Ctrl+C to stop.\n")

    while True:
        try:
            if not watch_dir.exists():
                time.sleep(interval)
                continue

            for path in sorted(watch_dir.iterdir()):
                if not path.is_file():
                    continue
                if str(path) in processed_files:
                    continue
                suffix = path.suffix.lower()
                if suffix in SUPPORTED_VIDEO:
                    try:
                        process_video(path, client)
                        processed_files.add(str(path))
                    except Exception as e:
                        print(f"  Error processing {path.name}: {e}", file=sys.stderr)
                elif suffix in SUPPORTED_IMAGE:
                    try:
                        process_image(path, client)
                        processed_files.add(str(path))
                    except Exception as e:
                        print(f"  Error processing {path.name}: {e}", file=sys.stderr)

            time.sleep(interval)
        except KeyboardInterrupt:
            print("\nWatch mode stopped.")
            break


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Wearable camera ingest")
    parser.add_argument("--watch", type=str, metavar="DIR",
                        help="Directory to watch for new files")
    parser.add_argument("--file", type=str, help="Process a single file")
    parser.add_argument("--interval", type=int, default=60,
                        help="Watch poll interval in seconds")
    args = parser.parse_args()

    client = get_client()

    if args.file:
        p = Path(args.file)
        if not p.exists():
            print(f"File not found: {args.file}", file=sys.stderr)
            sys.exit(1)
        if p.suffix.lower() in SUPPORTED_VIDEO:
            process_video(p, client)
        elif p.suffix.lower() in SUPPORTED_IMAGE:
            process_image(p, client)
        else:
            print(f"Unsupported file type: {p.suffix}", file=sys.stderr)
            sys.exit(1)
    elif args.watch:
        watch_directory(Path(args.watch), interval=args.interval, client=client)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
