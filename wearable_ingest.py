"""
Wearable ingest: transcribe Meta glasses video/audio → episodic memory entries.

Usage:
  python wearable_ingest.py video.mp4          # transcribe one file
  python wearable_ingest.py ~/Videos/glasses/  # process entire directory
  python wearable_ingest.py --watch ~/Videos/glasses/  # watch dir for new files

Output:
  brain/raw/YYYY-MM-DD-HH-MM-wearable-{slug}.md

Requires:
  - whisper-cpp: brew install whisper-cpp
  - ffmpeg: brew install ffmpeg
  - model auto-downloaded to ~/.whisper/models/ on first run
"""

import argparse
import hashlib
import json
import re
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
RAW_DIR = SCRIPT_DIR / "brain" / "raw"
STATE_FILE = SCRIPT_DIR / "wearable_state.json"

WHISPER_CLI = "/usr/local/Cellar/whisper-cpp/1.8.4/bin/whisper-cli"
MODELS_DIR = Path.home() / ".whisper" / "models"
MODEL_NAME = "ggml-base.en.bin"
MODEL_URL = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin"

SUPPORTED_AUDIO = {".mp3", ".wav", ".ogg", ".flac", ".m4a", ".aac"}
SUPPORTED_VIDEO = {".mp4", ".mov", ".mkv", ".avi", ".webm"}


def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {"processed": {}}


def save_state(state: dict):
    STATE_FILE.write_text(json.dumps(state, indent=2))


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def slugify(text: str) -> str:
    s = re.sub(r"[^\w\s-]", "", text.lower())
    s = re.sub(r"[\s_]+", "-", s)
    return re.sub(r"-+", "-", s).strip("-")[:50] or "wearable"


def ensure_model() -> Path:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / MODEL_NAME
    if model_path.exists():
        return model_path
    print(f"Downloading whisper model {MODEL_NAME} (~150MB)...")
    try:
        subprocess.run(
            ["curl", "-L", "-o", str(model_path), MODEL_URL],
            check=True, capture_output=False
        )
        print(f"Model saved to {model_path}")
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Could not download model: {e}")
        print(f"Manual download: {MODEL_URL}")
        print(f"Save to: {model_path}")
        sys.exit(1)
    return model_path


def to_wav(input_path: Path) -> Path | None:
    """Convert video/audio to 16kHz mono WAV for whisper. Returns temp WAV path."""
    suffix = input_path.suffix.lower()
    if suffix == ".wav":
        return input_path  # whisper handles WAV natively

    out = Path(tempfile.mktemp(suffix=".wav"))
    try:
        subprocess.run(
            ["ffmpeg", "-i", str(input_path),
             "-ar", "16000", "-ac", "1", "-f", "wav", str(out),
             "-y", "-loglevel", "error"],
            check=True
        )
        return out
    except FileNotFoundError:
        print("ERROR: ffmpeg not found. Install: brew install ffmpeg")
        return None
    except subprocess.CalledProcessError as e:
        print(f"ERROR: ffmpeg failed for {input_path.name}: {e}")
        return None


def transcribe(audio_path: Path, model_path: Path) -> str:
    """Run whisper-cli and return transcript text."""
    with tempfile.TemporaryDirectory() as tmp:
        out_base = Path(tmp) / "transcript"
        cmd = [
            WHISPER_CLI,
            "-m", str(model_path),
            "-f", str(audio_path),
            "--language", "en",
            "--output-txt",
            "--output-file", str(out_base),
            "--no-prints",    # suppress progress bars
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        except subprocess.TimeoutExpired:
            print(f"  WARN: whisper timed out on {audio_path.name}")
            return ""

        txt_file = Path(str(out_base) + ".txt")
        if txt_file.exists():
            return txt_file.read_text().strip()

        # Fallback: parse stdout (some whisper-cpp versions print to stdout)
        if result.stdout.strip():
            lines = [ln for ln in result.stdout.splitlines()
                     if not ln.startswith("[") and ln.strip()]
            return "\n".join(lines).strip()

        if result.returncode != 0:
            print(f"  WARN: whisper error: {result.stderr[:200]}")
        return ""


def write_memory(source_file: Path, transcript: str, duration_s: float = 0) -> Path:
    now = datetime.now(timezone.utc)
    timestamp = now.strftime("%Y-%m-%d-%H-%M")
    slug = slugify(source_file.stem)
    out_path = RAW_DIR / f"{timestamp}-wearable-{slug}.md"
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    header = "\n".join([
        f"SOURCE: wearable",
        f"Type: episodic",
        f"OriginalFile: {source_file.name}",
        f"Scraped: {now.isoformat()}",
        f"DurationSeconds: {int(duration_s)}" if duration_s else "",
    ])
    header = "\n".join(ln for ln in header.splitlines() if ln)

    out_path.write_text(f"{header}\n---\n{transcript}\n", encoding="utf-8")
    print(f"  Saved: {out_path.relative_to(SCRIPT_DIR)}")
    return out_path


def get_duration(path: Path) -> float:
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
             "-of", "csv=p=0", str(path)],
            capture_output=True, text=True
        )
        return float(r.stdout.strip() or 0)
    except Exception:
        return 0.0


def process_file(path: Path, model_path: Path, state: dict, force: bool = False) -> bool:
    h = file_hash(path)
    if not force and state["processed"].get(str(path)) == h:
        print(f"  Already processed: {path.name}")
        return False

    print(f"Processing: {path.name}")
    suffix = path.suffix.lower()

    if suffix in SUPPORTED_VIDEO:
        duration = get_duration(path)
        wav = to_wav(path)
        if not wav:
            return False
        is_temp = wav != path
    elif suffix in SUPPORTED_AUDIO:
        duration = get_duration(path)
        wav = to_wav(path) if suffix != ".wav" else path
        is_temp = wav != path
    else:
        print(f"  Skipped (unsupported format): {path.name}")
        return False

    try:
        transcript = transcribe(wav, model_path)
    finally:
        if is_temp and wav and wav.exists():
            wav.unlink()

    if not transcript:
        print(f"  WARN: Empty transcript for {path.name}")
        return False

    write_memory(path, transcript, duration)
    state["processed"][str(path)] = h
    save_state(state)
    return True


def watch_dir(directory: Path, model_path: Path, state: dict, interval: int = 30):
    print(f"Watching {directory} every {interval}s... (Ctrl+C to stop)")
    while True:
        for p in sorted(directory.iterdir()):
            if p.suffix.lower() in SUPPORTED_VIDEO | SUPPORTED_AUDIO:
                process_file(p, model_path, state)
        time.sleep(interval)


def main():
    parser = argparse.ArgumentParser(description="Wearable video/audio → episodic memory")
    parser.add_argument("paths", nargs="*", help="Files or directories to process")
    parser.add_argument("--watch", metavar="DIR", help="Watch directory for new files")
    parser.add_argument("--force", action="store_true", help="Re-process already-done files")
    parser.add_argument("--model", help=f"Whisper model path (default: auto-download {MODEL_NAME})")
    args = parser.parse_args()

    if not args.paths and not args.watch:
        parser.print_help()
        sys.exit(0)

    model_path = Path(args.model) if args.model else ensure_model()
    if not model_path.exists():
        print(f"ERROR: Model not found: {model_path}")
        sys.exit(1)

    state = load_state()

    if args.watch:
        watch_dir(Path(args.watch).expanduser(), model_path, state)
        return

    for raw in args.paths:
        p = Path(raw).expanduser()
        if p.is_dir():
            files = sorted(
                f for f in p.iterdir()
                if f.suffix.lower() in SUPPORTED_VIDEO | SUPPORTED_AUDIO
            )
            print(f"Found {len(files)} media file(s) in {p}")
            for f in files:
                process_file(f, model_path, state, force=args.force)
        elif p.is_file():
            process_file(p, model_path, state, force=args.force)
        else:
            print(f"WARN: Not found: {p}")

    print("\nDone.")


if __name__ == "__main__":
    main()
