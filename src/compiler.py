"""
src/compiler.py — Core compilation logic as a reusable Compiler class.

Extracted from compile.py for library use. Handles file ingestion,
content extraction, Claude routing, and wiki entry writing.
"""

import base64
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import anthropic

logger = logging.getLogger(__name__)

SUPPORTED_TEXT = frozenset({".txt", ".md"})
SUPPORTED_PDF = frozenset({".pdf"})
SUPPORTED_IMAGE = frozenset({".jpg", ".jpeg", ".png", ".webp"})
SUPPORTED_AUDIO = frozenset({".mp3", ".wav", ".m4a"})
SUPPORTED_VIDEO = frozenset({".mp4", ".mov"})
ALL_SUPPORTED = SUPPORTED_TEXT | SUPPORTED_PDF | SUPPORTED_IMAGE | SUPPORTED_AUDIO | SUPPORTED_VIDEO

ROUTE_KEYWORDS = {
    "me": ["i ", "my ", "me ", "i'm", "personal", "diary", "journal", "health"],
    "work": ["work", "job", "project", "meeting", "client", "business", "deadline"],
    "knowledge": ["research", "paper", "study", "theory", "concept", "science"],
    "media": ["video", "podcast", "movie", "music", "book", "article", "youtube"],
}


class Compiler:
    """
    Core compiler: ingests files, extracts content, compiles with Claude.

    Can be embedded in other pipelines or used standalone.
    """

    def __init__(self, brain_dir: Optional[Path] = None,
                 api_key: Optional[str] = None,
                 model: str = "claude-opus-4-5"):
        self.brain_dir = Path(brain_dir or os.getenv("BRAIN_DIR", "./brain"))
        self.raw_dir = self.brain_dir / "raw"
        self.processed_dir = self.raw_dir / "processed"
        self.hashes_file = self.brain_dir / ".processed_hashes"
        self.index_file = self.brain_dir / "index.md"
        self.model = model

        _key = api_key or os.getenv("ANTHROPIC_API_KEY", "")
        self.client = anthropic.Anthropic(api_key=_key) if _key else None

        self._known_hashes: Optional[set] = None
        self._stats = {"processed": 0, "duplicates": 0, "errors": 0, "new_entries": 0}

    def ensure_dirs(self):
        """Create required brain directory structure."""
        for d in [
            self.raw_dir, self.processed_dir,
            self.brain_dir / "me",
            self.brain_dir / "work",
            self.brain_dir / "knowledge" / "wiki",
            self.brain_dir / "media",
            self.brain_dir / "fragments",
            self.brain_dir / "studio",
        ]:
            d.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Deduplication
    # ------------------------------------------------------------------

    @property
    def known_hashes(self) -> set:
        if self._known_hashes is None:
            self._known_hashes = self._load_hashes()
        return self._known_hashes

    def _load_hashes(self) -> set:
        if self.hashes_file.exists():
            return set(self.hashes_file.read_text().splitlines())
        return set()

    def _save_hash(self, h: str):
        with open(self.hashes_file, "a") as f:
            f.write(h + "\n")
        self.known_hashes.add(h)

    @staticmethod
    def sha256_file(path: Path) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    @staticmethod
    def sha256_text(text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()

    def is_duplicate(self, path: Path) -> bool:
        return self.sha256_file(path) in self.known_hashes

    # ------------------------------------------------------------------
    # Content extraction
    # ------------------------------------------------------------------

    def extract_text(self, path: Path) -> str:
        """Extract text content from any supported file type."""
        suffix = path.suffix.lower()

        if suffix in SUPPORTED_TEXT:
            raw = path.read_text(encoding="utf-8", errors="replace")
            yt_urls = self._extract_youtube_urls(raw)
            if yt_urls:
                transcripts = [self._fetch_youtube_transcript(u) for u in yt_urls]
                return raw + "\n\n" + "\n\n".join(transcripts)
            return raw

        elif suffix in SUPPORTED_PDF:
            return self._extract_pdf(path)

        elif suffix in SUPPORTED_IMAGE:
            return self._analyze_image(path)

        elif suffix in SUPPORTED_AUDIO:
            return self._transcribe_audio(path)

        elif suffix in SUPPORTED_VIDEO:
            audio = self._extract_audio(path)
            if audio:
                transcript = self._transcribe_audio(audio)
                audio.unlink(missing_ok=True)
                return f"[Video: {path.name}]\n{transcript}"
            return f"[Video: {path.name} — audio extraction failed]"

        return ""

    def _extract_youtube_urls(self, text: str) -> list[str]:
        pattern = r"(https?://(?:www\.)?(?:youtube\.com/watch\?v=|youtu\.be/)[\w\-_?=&]+)"
        return re.findall(pattern, text)

    def _fetch_youtube_transcript(self, url: str) -> str:
        try:
            from youtube_transcript_api import YouTubeTranscriptApi
            match = re.search(r"(?:v=|youtu\.be/)([a-zA-Z0-9_-]{11})", url)
            if not match:
                return f"[YouTube: {url}]"
            transcript = YouTubeTranscriptApi.get_transcript(match.group(1))
            return f"[YouTube: {url}]\n" + " ".join(e["text"] for e in transcript)
        except Exception as e:
            return f"[YouTube: {url} — transcript unavailable: {e}]"

    def _extract_pdf(self, path: Path) -> str:
        try:
            import fitz
            doc = fitz.open(str(path))
            pages = [f"[Page {i+1}]\n{page.get_text()}" for i, page in enumerate(doc)]
            doc.close()
            return "\n\n".join(pages)
        except ImportError:
            return f"[PDF: {path.name} — pymupdf not available]"
        except Exception as e:
            return f"[PDF error: {e}]"

    def _analyze_image(self, path: Path) -> str:
        if self.client is None:
            return f"[Image: {path.name} — no API key]"
        suffix = path.suffix.lower()
        media_map = {".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                     ".png": "image/png", ".webp": "image/webp"}
        media_type = media_map.get(suffix, "image/jpeg")
        with open(path, "rb") as f:
            img_data = base64.standard_b64encode(f.read()).decode()
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                messages=[{"role": "user", "content": [
                    {"type": "image", "source": {"type": "base64",
                                                  "media_type": media_type, "data": img_data}},
                    {"type": "text", "text": "Describe this image in detail for a personal knowledge base. Extract any visible text, identify people, places, objects, and events."}
                ]}]
            )
            return response.content[0].text
        except Exception as e:
            return f"[Image analysis failed: {e}]"

    def _transcribe_audio(self, path: Path) -> str:
        try:
            from faster_whisper import WhisperModel
            model = WhisperModel("base", device="cpu", compute_type="int8")
            segments, _ = model.transcribe(str(path))
            return " ".join(s.text for s in segments).strip()
        except ImportError:
            return f"[Audio: {path.name} — faster-whisper not available]"
        except Exception as e:
            return f"[Transcription error: {e}]"

    def _extract_audio(self, video_path: Path) -> Optional[Path]:
        audio_path = video_path.with_suffix(".tmp_compiler.wav")
        try:
            result = subprocess.run(
                ["ffmpeg", "-y", "-i", str(video_path), "-vn",
                 "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", str(audio_path)],
                capture_output=True, timeout=300
            )
            return audio_path if result.returncode == 0 and audio_path.exists() else None
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Claude compilation
    # ------------------------------------------------------------------

    def compile(self, content: str, source_name: str) -> dict:
        """
        Compile raw content into structured entry using Claude.

        Returns dict: {title, summary, concepts, events, people, dates,
                       route, wikilinks, markdown, tags}
        """
        if self.client is None:
            return self._fallback_compile(content, source_name)

        prompt = f"""You are a personal knowledge base compiler. Process this content and return structured JSON.

Source: {source_name}
Content:
{content[:8000]}

Return JSON:
{{
  "title": "concise title",
  "summary": "2-3 sentence summary",
  "concepts": ["concept1", "concept2"],
  "events": ["event1 (date if known)"],
  "people": ["person1"],
  "dates": ["YYYY-MM-DD"],
  "route": "me|work|knowledge|media",
  "wikilinks": ["[[concept]]"],
  "markdown": "full article with [[wikilinks]]",
  "tags": ["tag1"]
}}

Route: me=personal/health/diary, work=professional/projects, knowledge=research/facts, media=videos/books/articles"""

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=2048,
                messages=[{"role": "user", "content": prompt}]
            )
            text = response.content[0].text
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                return json.loads(match.group())
        except Exception as e:
            logger.error(f"Claude compile error: {e}")

        return self._fallback_compile(content, source_name)

    def _fallback_compile(self, content: str, source_name: str) -> dict:
        """Keyword-based fallback when Claude is unavailable."""
        content_lower = content.lower()
        scores = {d: sum(content_lower.count(kw) for kw in kws)
                  for d, kws in ROUTE_KEYWORDS.items()}
        route = max(scores, key=scores.get)
        return {
            "title": source_name,
            "summary": content[:200],
            "concepts": [],
            "events": [],
            "people": [],
            "dates": [],
            "route": route,
            "wikilinks": [],
            "markdown": content,
            "tags": [],
        }

    # ------------------------------------------------------------------
    # Entry writing
    # ------------------------------------------------------------------

    def write_entry(self, data: dict, source_file: Path) -> Path:
        """Write compiled entry to the appropriate brain directory."""
        route = data.get("route", "knowledge")
        title = data.get("title", source_file.stem)
        safe_title = re.sub(r"[^\w\s-]", "", title).strip().replace(" ", "_")[:80]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{safe_title}.md"

        route_map = {
            "me": self.brain_dir / "me",
            "work": self.brain_dir / "work",
            "knowledge": self.brain_dir / "knowledge",
            "media": self.brain_dir / "media",
        }
        dest_dir = route_map.get(route, self.brain_dir / "knowledge")
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = dest_dir / filename

        wikilinks_str = " ".join(data.get("wikilinks", []))
        tags_str = ", ".join(data.get("tags", []))

        content = f"""---
title: {title}
source: {source_file.name}
date: {datetime.now().isoformat()}
route: {route}
tags: [{tags_str}]
concepts: {json.dumps(data.get('concepts', []))}
people: {json.dumps(data.get('people', []))}
---

# {title}

**Summary:** {data.get('summary', '')}

**Links:** {wikilinks_str}

---

{data.get('markdown', '')}

---
*Compiled: {datetime.now().strftime('%Y-%m-%d %H:%M')} | Source: {source_file.name}*
"""
        dest_path.write_text(content, encoding="utf-8")
        return dest_path

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def process_file(self, path: Path, move_to_processed: bool = True) -> Optional[dict]:
        """
        Full pipeline: extract → compile → write entry → move file.

        Returns compiled data dict, or None if skipped/error.
        """
        suffix = path.suffix.lower()
        if suffix not in ALL_SUPPORTED:
            return None

        if self.is_duplicate(path):
            logger.info(f"Duplicate skipped: {path.name}")
            self._stats["duplicates"] += 1
            return {"skipped": True, "reason": "duplicate"}

        content = self.extract_text(path)
        if not content.strip():
            logger.warning(f"Empty content: {path.name}")
            return None

        data = self.compile(content, path.name)
        dest_path = self.write_entry(data, path)
        data["dest_path"] = str(dest_path)

        file_hash = self.sha256_file(path)
        self._save_hash(file_hash)

        if move_to_processed:
            self.processed_dir.mkdir(parents=True, exist_ok=True)
            dest = self.processed_dir / path.name
            if dest.exists():
                dest = self.processed_dir / f"{path.stem}_{int(time.time())}{path.suffix}"
            shutil.move(str(path), str(dest))

        self._stats["processed"] += 1
        self._stats["new_entries"] += 1
        logger.info(f"Compiled: {path.name} -> {dest_path.relative_to(self.brain_dir)}")
        return data

    def process_directory(self, directory: Path) -> list[dict]:
        """Process all supported files in a directory."""
        self.ensure_dirs()
        files = [f for f in directory.iterdir()
                 if f.is_file() and f.suffix.lower() in ALL_SUPPORTED]
        results = []
        for path in sorted(files):
            try:
                result = self.process_file(path)
                if result:
                    results.append(result)
            except Exception as e:
                logger.error(f"Error processing {path.name}: {e}")
                self._stats["errors"] += 1
        return results

    @property
    def stats(self) -> dict:
        return dict(self._stats)

    def __repr__(self) -> str:
        return (f"Compiler(brain={self.brain_dir}, model={self.model}, "
                f"stats={self._stats})")
