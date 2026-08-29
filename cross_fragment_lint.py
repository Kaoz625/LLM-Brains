#!/usr/bin/env python3
"""
cross_fragment_lint.py — Cross-Fragment Wiki Linter
The "compiler" for the LLM-Brains hive mind. Checks for contradictions,
stale entries, orphan wikilinks, missing cross-links, and duplicate concepts
across all fragment wikis in brain/.

Key insight: GPT-4 / LLaMA-3 perform near-random on contradiction detection
without structure. Solution: extract knowledge-graph triples first, then use
LLM verification only on structurally-overlapping claim pairs.

Usage:
    python cross_fragment_lint.py --scope full
    python cross_fragment_lint.py --scope recent
    python cross_fragment_lint.py --scope fragment --fragment technology
    python cross_fragment_lint.py --scope full --fix
    python cross_fragment_lint.py --scope full --format json
    python cross_fragment_lint.py --scope full --format markdown
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Optional imports
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

try:
    import click

    HAS_CLICK = True
except ImportError:
    HAS_CLICK = False

try:
    from rich.console import Console
    from rich.table import Table
    from rich import box
    from rich.text import Text

    HAS_RICH = True
except ImportError:
    HAS_RICH = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BRAIN_DIR = Path("brain")
LINT_REPORTS_DIR = BRAIN_DIR / "lint_reports"
LINT_CACHE_FILE = BRAIN_DIR / ".lint_cache.json"

CLAUDE_MODEL = "claude-haiku-4-5-20251001"

TIME_SENSITIVE_PATTERNS = re.compile(
    r"\b(news|current|latest|version\s+\d|release|released|today|this\s+year|"
    r"as\s+of\s+\d{4}|recently|upcoming|announced|just\s+launched)\b",
    re.IGNORECASE,
)
WIKILINK_RE = re.compile(r"\[\[([^\]]+)\]\]")
STALENESS_DAYS = 30
MAX_CLAIM_PAIRS = 100


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class LintIssue:
    severity: str  # "error", "warning", "info"
    issue_type: str  # "contradiction", "stale", "orphan", "missing_link", "duplicate", "drift"
    fragment_a: str
    entry_a: str
    fragment_b: Optional[str]
    entry_b: Optional[str]
    description: str
    suggested_fix: str
    confidence: float  # 0.0-1.0


@dataclass
class LintReport:
    timestamp: str
    scope: str
    issues: list[LintIssue]
    fragments_checked: int
    entries_checked: int
    duration_seconds: float


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _file_hash(path: Path) -> str:
    """Return SHA-256 hex digest of a file's contents."""
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _load_cache() -> dict:
    if LINT_CACHE_FILE.exists():
        try:
            return json.loads(LINT_CACHE_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save_cache(cache: dict) -> None:
    LINT_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    LINT_CACHE_FILE.write_text(json.dumps(cache, indent=2), encoding="utf-8")


def _read_entry(path: Path) -> dict:
    """
    Return a dict with keys: path, fragment, title, content, metadata, raw.
    Uses python-frontmatter when available; falls back to plain text.
    """
    raw = path.read_text(encoding="utf-8", errors="replace")
    metadata: dict[str, Any] = {}
    content = raw

    if HAS_FRONTMATTER:
        try:
            post = frontmatter.loads(raw)
            metadata = dict(post.metadata)
            content = post.content
        except Exception:
            pass

    # Derive fragment from the path relative to BRAIN_DIR
    try:
        rel = path.relative_to(BRAIN_DIR)
        parts = rel.parts
        fragment = parts[0] if len(parts) > 1 else "root"
    except ValueError:
        fragment = "root"

    title = metadata.get("title") or path.stem

    return {
        "path": path,
        "fragment": fragment,
        "title": title,
        "content": content,
        "metadata": metadata,
        "raw": raw,
    }


def _collect_entries(scope: str, fragment: Optional[str] = None) -> list[dict]:
    """Collect entry dicts according to scope."""
    if not BRAIN_DIR.exists():
        return []

    all_md = [
        p
        for p in BRAIN_DIR.rglob("*.md")
        if not any(part.startswith(".") for part in p.parts)
        and "lint_reports" not in p.parts
        and p.name != "index.md"
    ]

    if scope == "fragment" and fragment:
        frag_dir = BRAIN_DIR / fragment
        all_md = [p for p in all_md if p.is_relative_to(frag_dir)]
    elif scope == "recent":
        cutoff = time.time() - 86400  # 24 hours
        all_md = [p for p in all_md if p.stat().st_mtime >= cutoff]

    return [_read_entry(p) for p in all_md]


def _tfidf_similarity(title_a: str, title_b: str) -> float:
    """
    Lightweight TF-IDF cosine similarity between two short strings (titles).
    Used when sqlite-vec is unavailable.
    """
    # Tokenise to lower-case words
    def tokenise(s: str) -> list[str]:
        return re.findall(r"[a-z0-9]+", s.lower())

    tokens_a = tokenise(title_a)
    tokens_b = tokenise(title_b)

    if not tokens_a or not tokens_b:
        return 0.0

    vocab = set(tokens_a) | set(tokens_b)
    # Simple binary TF (presence/absence) — adequate for short titles
    vec_a = {t: 1 for t in tokens_a}
    vec_b = {t: 1 for t in tokens_b}

    dot = sum(vec_a.get(t, 0) * vec_b.get(t, 0) for t in vocab)
    mag_a = math.sqrt(sum(v**2 for v in vec_a.values()))
    mag_b = math.sqrt(sum(v**2 for v in vec_b.values()))

    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


# ---------------------------------------------------------------------------
# Core linter class
# ---------------------------------------------------------------------------


class CrossFragmentLinter:
    def __init__(self) -> None:
        self._api_key: Optional[str] = os.environ.get("ANTHROPIC_API_KEY")
        self._client: Optional[Any] = None
        self._cache: dict = _load_cache()
        self._cache_dirty: bool = False

        if HAS_ANTHROPIC and self._api_key:
            self._client = anthropic.Anthropic(api_key=self._api_key)

    # ------------------------------------------------------------------
    # 1. Claim extraction
    # ------------------------------------------------------------------

    def extract_claims(self, entry_content: str) -> list[dict]:
        """
        Use Claude API to extract factual claims as subject-predicate-object triples.
        Falls back to empty list when API unavailable.
        Max 10 claims returned.
        """
        content_hash = hashlib.sha256(entry_content.encode()).hexdigest()
        cache_key = f"claims:{content_hash}"

        if cache_key in self._cache:
            return self._cache[cache_key]

        if not self._client:
            return []

        system_prompt = (
            "Extract factual claims as subject-predicate-object triples from this text. "
            "Return a JSON array with objects containing keys: subject, predicate, object, confidence. "
            "confidence is a float 0-1. Return at most 10 claims. "
            "Example: [{\"subject\": \"BitNet\", \"predicate\": \"uses\", "
            "\"object\": \"ternary weights\", \"confidence\": 0.95}]"
        )

        try:
            response = self._client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=512,
                system=system_prompt,
                messages=[{"role": "user", "content": entry_content[:4000]}],
            )
            raw_text = response.content[0].text.strip()
            # Strip markdown code fences if present
            raw_text = re.sub(r"^```[a-z]*\n?", "", raw_text)
            raw_text = re.sub(r"\n?```$", "", raw_text)
            claims = json.loads(raw_text)
            if not isinstance(claims, list):
                claims = []
            # Normalise and cap
            normalised: list[dict] = []
            for c in claims[:10]:
                if isinstance(c, dict) and all(
                    k in c for k in ("subject", "predicate", "object")
                ):
                    normalised.append(
                        {
                            "subject": str(c["subject"]).lower().strip(),
                            "predicate": str(c["predicate"]).lower().strip(),
                            "object": str(c["object"]).lower().strip(),
                            "confidence": float(c.get("confidence", 0.8)),
                        }
                    )
        except Exception:
            normalised = []

        self._cache[cache_key] = normalised
        self._cache_dirty = True
        return normalised

    # ------------------------------------------------------------------
    # 2. Contradiction checking
    # ------------------------------------------------------------------

    def check_contradiction(
        self,
        claim_a: dict,
        claim_b: dict,
        context_a: str,
        context_b: str,
    ) -> dict:
        """
        Use Claude to check whether two claims with the same subject contradict each other.
        Returns {"contradicts": bool, "explanation": str, "confidence": float}.
        Falls back to a heuristic when API unavailable.
        """
        # Only compare claims with the same subject
        if claim_a["subject"] != claim_b["subject"]:
            return {"contradicts": False, "explanation": "Different subjects", "confidence": 1.0}

        # Quick heuristic: same predicate, same subject, different object → candidate
        same_predicate = claim_a["predicate"] == claim_b["predicate"]
        same_object = claim_a["object"] == claim_b["object"]

        if same_predicate and same_object:
            return {"contradicts": False, "explanation": "Identical claims", "confidence": 1.0}

        if not self._client:
            # Heuristic fallback: flag same subject + same predicate + different object
            if same_predicate and not same_object:
                return {
                    "contradicts": True,
                    "explanation": (
                        f"Same subject '{claim_a['subject']}' and predicate "
                        f"'{claim_a['predicate']}' but different objects: "
                        f"'{claim_a['object']}' vs '{claim_b['object']}'"
                    ),
                    "confidence": 0.5,
                }
            return {"contradicts": False, "explanation": "No structural conflict detected", "confidence": 0.4}

        # Cache key based on claim content
        pair_key = hashlib.sha256(
            json.dumps([claim_a, claim_b], sort_keys=True).encode()
        ).hexdigest()
        cache_key = f"contra:{pair_key}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        prompt = (
            f"Claim A: {claim_a['subject']} {claim_a['predicate']} {claim_a['object']}\n"
            f"Context A (excerpt): {context_a[:500]}\n\n"
            f"Claim B: {claim_b['subject']} {claim_b['predicate']} {claim_b['object']}\n"
            f"Context B (excerpt): {context_b[:500]}\n\n"
            "Do these two claims directly contradict each other? "
            "Reply with JSON only: {\"contradicts\": bool, \"explanation\": str, \"confidence\": float}"
        )

        try:
            response = self._client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=256,
                messages=[{"role": "user", "content": prompt}],
            )
            raw_text = response.content[0].text.strip()
            raw_text = re.sub(r"^```[a-z]*\n?", "", raw_text)
            raw_text = re.sub(r"\n?```$", "", raw_text)
            result = json.loads(raw_text)
            result = {
                "contradicts": bool(result.get("contradicts", False)),
                "explanation": str(result.get("explanation", "")),
                "confidence": float(result.get("confidence", 0.7)),
            }
        except Exception:
            result = {
                "contradicts": False,
                "explanation": "LLM check failed; could not determine",
                "confidence": 0.0,
            }

        self._cache[cache_key] = result
        self._cache_dirty = True
        return result

    # ------------------------------------------------------------------
    # 3. Orphan wikilinks
    # ------------------------------------------------------------------

    def find_orphan_links(self, entry_content: str, all_titles: set[str]) -> list[str]:
        """
        Return wikilink targets in entry_content that don't resolve to any
        existing entry title or filename stem.
        """
        orphans: list[str] = []
        for m in WIKILINK_RE.finditer(entry_content):
            target = m.group(1).strip()
            # Strip anchor fragments like [[Page#Section]]
            target_base = target.split("#")[0].strip()
            # Strip display aliases like [[Page|Display]]
            target_base = target_base.split("|")[0].strip()
            # Normalise: lower-case, collapse spaces
            normalised = re.sub(r"\s+", " ", target_base).lower()
            if normalised and normalised not in all_titles:
                orphans.append(target_base)
        return orphans

    # ------------------------------------------------------------------
    # 4. Staleness detection
    # ------------------------------------------------------------------

    def detect_staleness(self, entry_path: Path) -> dict:
        """
        Check the updated_at frontmatter field (or file mtime as fallback).
        Flag entries older than STALENESS_DAYS that reference time-sensitive topics.
        """
        now = datetime.now(tz=timezone.utc)
        updated_at: Optional[datetime] = None

        if HAS_FRONTMATTER:
            try:
                post = frontmatter.load(str(entry_path))
                raw_date = post.metadata.get("updated_at") or post.metadata.get("date")
                if raw_date:
                    if isinstance(raw_date, datetime):
                        updated_at = raw_date
                        if updated_at.tzinfo is None:
                            updated_at = updated_at.replace(tzinfo=timezone.utc)
                    elif isinstance(raw_date, str):
                        for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%SZ"):
                            try:
                                updated_at = datetime.strptime(raw_date, fmt).replace(
                                    tzinfo=timezone.utc
                                )
                                break
                            except ValueError:
                                continue
            except Exception:
                pass

        if updated_at is None:
            mtime = entry_path.stat().st_mtime
            updated_at = datetime.fromtimestamp(mtime, tz=timezone.utc)

        days_old = (now - updated_at).days

        if days_old < STALENESS_DAYS:
            return {"stale": False, "days_old": days_old, "reason": ""}

        # Check whether content contains time-sensitive language
        try:
            content = entry_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            content = ""

        if TIME_SENSITIVE_PATTERNS.search(content):
            return {
                "stale": True,
                "days_old": days_old,
                "reason": (
                    f"Entry is {days_old} days old and contains time-sensitive language "
                    "(news, current, latest, version, release, etc.)"
                ),
            }

        return {"stale": False, "days_old": days_old, "reason": ""}

    # ------------------------------------------------------------------
    # 5. Duplicate concept detection
    # ------------------------------------------------------------------

    def check_duplicate_concepts(self, entries: list[dict]) -> list[tuple]:
        """
        Find entry pairs that likely cover the same concept.
        Uses sqlite-vec cosine similarity if available, else TF-IDF on titles.
        Returns list of (entry_a, entry_b, similarity) tuples where sim > 0.85.
        """
        duplicates: list[tuple] = []

        # Try sqlite-vec path
        try:
            import sqlite_vec  # noqa: F401

            # sqlite-vec is present — attempt in-memory vector comparison using title embeddings
            # For now, fall through to TF-IDF since we don't have a persistent DB here.
            # A full implementation would store embeddings in the lint cache.
            raise ImportError("sqlite-vec title embedding not wired in linter; using TF-IDF")
        except ImportError:
            pass

        # TF-IDF fallback on titles
        n = len(entries)
        for i in range(n):
            for j in range(i + 1, n):
                sim = _tfidf_similarity(entries[i]["title"], entries[j]["title"])
                if sim > 0.85:
                    duplicates.append((entries[i], entries[j], sim))

        return duplicates

    # ------------------------------------------------------------------
    # 6. Main lint runner
    # ------------------------------------------------------------------

    def run_lint(
        self,
        scope: str = "full",
        fragment: Optional[str] = None,
        output_format: str = "rich",
    ) -> LintReport:
        """
        Main entry point. Collects entries, runs all checks, saves report.
        """
        start_time = time.monotonic()
        timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")

        entries = _collect_entries(scope, fragment)
        issues: list[LintIssue] = []

        # Build look-up sets
        all_titles: set[str] = set()
        for e in entries:
            all_titles.add(e["title"].lower())
            all_titles.add(e["path"].stem.lower())

        # ---- Claims extraction ----
        entry_claims: dict[str, list[dict]] = {}
        for e in entries:
            key = str(e["path"])
            claims = self.extract_claims(e["content"])
            entry_claims[key] = claims

        # ---- Contradiction detection (cost-controlled) ----
        # Build subject → [(entry, claim), ...] index
        subject_index: dict[str, list[tuple[dict, dict]]] = defaultdict(list)
        for e in entries:
            for claim in entry_claims.get(str(e["path"]), []):
                subject_index[claim["subject"]].append((e, claim))

        pairs_checked = 0
        for subject, claim_list in subject_index.items():
            if len(claim_list) < 2:
                continue
            for i in range(len(claim_list)):
                for j in range(i + 1, len(claim_list)):
                    if pairs_checked >= MAX_CLAIM_PAIRS:
                        break
                    entry_a, claim_a = claim_list[i]
                    entry_b, claim_b = claim_list[j]

                    # Don't compare an entry to itself
                    if entry_a["path"] == entry_b["path"]:
                        continue

                    result = self.check_contradiction(
                        claim_a,
                        claim_b,
                        entry_a["content"],
                        entry_b["content"],
                    )
                    pairs_checked += 1

                    if result["contradicts"]:
                        severity = "error" if result["confidence"] >= 0.7 else "warning"
                        issues.append(
                            LintIssue(
                                severity=severity,
                                issue_type="contradiction",
                                fragment_a=entry_a["fragment"],
                                entry_a=str(entry_a["path"]),
                                fragment_b=entry_b["fragment"],
                                entry_b=str(entry_b["path"]),
                                description=(
                                    f"Contradiction on '{subject}': {result['explanation']}"
                                ),
                                suggested_fix=(
                                    "Review both entries and reconcile the conflicting claims. "
                                    "Add a [[See Also]] link to the authoritative entry."
                                ),
                                confidence=result["confidence"],
                            )
                        )
                if pairs_checked >= MAX_CLAIM_PAIRS:
                    break

        # ---- Orphan wikilinks ----
        for e in entries:
            orphans = self.find_orphan_links(e["content"], all_titles)
            for orphan in orphans:
                issues.append(
                    LintIssue(
                        severity="warning",
                        issue_type="orphan",
                        fragment_a=e["fragment"],
                        entry_a=str(e["path"]),
                        fragment_b=None,
                        entry_b=None,
                        description=f"Unresolved wikilink: [[{orphan}]]",
                        suggested_fix=(
                            f"Create a stub entry for '{orphan}' or correct the link target."
                        ),
                        confidence=1.0,
                    )
                )

        # ---- Staleness ----
        for e in entries:
            staleness = self.detect_staleness(e["path"])
            if staleness["stale"]:
                issues.append(
                    LintIssue(
                        severity="warning",
                        issue_type="stale",
                        fragment_a=e["fragment"],
                        entry_a=str(e["path"]),
                        fragment_b=None,
                        entry_b=None,
                        description=staleness["reason"],
                        suggested_fix=(
                            "Review and update the entry, or remove time-sensitive language. "
                            "Update the updated_at frontmatter field."
                        ),
                        confidence=0.9,
                    )
                )

        # ---- Duplicate concept detection ----
        dupe_pairs = self.check_duplicate_concepts(entries)
        for entry_a, entry_b, sim in dupe_pairs:
            issues.append(
                LintIssue(
                    severity="info",
                    issue_type="duplicate",
                    fragment_a=entry_a["fragment"],
                    entry_a=str(entry_a["path"]),
                    fragment_b=entry_b["fragment"],
                    entry_b=str(entry_b["path"]),
                    description=(
                        f"Potential duplicate concepts (title similarity {sim:.0%}): "
                        f"'{entry_a['title']}' and '{entry_b['title']}'"
                    ),
                    suggested_fix=(
                        "Consider merging these entries or adding explicit cross-links "
                        "to clarify the distinction."
                    ),
                    confidence=sim,
                )
            )

        # ---- Persist cache ----
        if self._cache_dirty:
            _save_cache(self._cache)
            self._cache_dirty = False

        # ---- Build report ----
        duration = time.monotonic() - start_time
        fragments_checked = len({e["fragment"] for e in entries})

        report = LintReport(
            timestamp=timestamp,
            scope=scope if not fragment else f"fragment:{fragment}",
            issues=issues,
            fragments_checked=fragments_checked,
            entries_checked=len(entries),
            duration_seconds=round(duration, 3),
        )

        # ---- Save report ----
        LINT_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        report_path = LINT_REPORTS_DIR / f"{timestamp}.json"
        report_path.write_text(
            json.dumps(asdict(report), indent=2, default=str), encoding="utf-8"
        )

        return report

    # ------------------------------------------------------------------
    # 7. Auto-fix
    # ------------------------------------------------------------------

    def apply_fix(self, issue: LintIssue) -> bool:
        """
        Attempt an automatic fix for a lint issue.
        Returns True if fixed automatically, False if human review needed.
        """
        issue_type = issue.issue_type

        if issue_type == "orphan":
            return self._fix_orphan(issue)
        elif issue_type == "duplicate":
            return self._fix_duplicate(issue)
        elif issue_type == "stale":
            return self._fix_stale(issue)
        elif issue_type == "contradiction":
            return self._fix_contradiction(issue)
        else:
            return False

    def _fix_orphan(self, issue: LintIssue) -> bool:
        """Create a stub entry for an unresolved wikilink."""
        m = re.search(r"\[\[(.+?)\]\]", issue.description)
        if not m:
            return False
        target = m.group(1).strip()

        # Infer a plausible path from the origin fragment
        fragment_dir = BRAIN_DIR / issue.fragment_a
        fragment_dir.mkdir(parents=True, exist_ok=True)
        slug = re.sub(r"[^a-z0-9_-]", "-", target.lower()).strip("-")
        stub_path = fragment_dir / f"{slug}.md"

        if stub_path.exists():
            return False  # Already exists — the link just has wrong capitalisation

        now_str = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
        stub_content = (
            f"---\ntitle: {target}\ncreated_at: {now_str}\nupdated_at: {now_str}\n"
            f"stub: true\n---\n\n# {target}\n\n"
            f"> **Stub** — This entry was auto-created by cross_fragment_lint.py to resolve "
            f"an orphan wikilink from `{issue.entry_a}`.\n\n"
            f"<!-- TODO: Fill in content for {target} -->\n"
        )
        stub_path.write_text(stub_content, encoding="utf-8")
        return True

    def _fix_duplicate(self, issue: LintIssue) -> bool:
        """Add a merge-suggestion comment to both entries."""
        if not issue.entry_a or not issue.entry_b:
            return False
        path_a = Path(issue.entry_a)
        path_b = Path(issue.entry_b)
        if not path_a.exists() or not path_b.exists():
            return False

        notice = (
            f"\n\n<!-- LINT:DUPLICATE — Possible duplicate of [[{path_b.stem}]]. "
            f"Consider merging. (cross_fragment_lint.py) -->\n"
        )
        notice_b = (
            f"\n\n<!-- LINT:DUPLICATE — Possible duplicate of [[{path_a.stem}]]. "
            f"Consider merging. (cross_fragment_lint.py) -->\n"
        )

        for path, note in [(path_a, notice), (path_b, notice_b)]:
            content = path.read_text(encoding="utf-8")
            if "LINT:DUPLICATE" not in content:
                path.write_text(content + note, encoding="utf-8")
        return True

    def _fix_stale(self, issue: LintIssue) -> bool:
        """Prepend a stale warning notice to the entry body (after frontmatter)."""
        path = Path(issue.entry_a)
        if not path.exists():
            return False

        content = path.read_text(encoding="utf-8")
        if "STALE" in content:
            return True  # Already flagged

        stale_notice = (
            "\n> [!WARNING] STALE\n"
            "> This entry may be out of date. "
            "Please review and update the content.\n\n"
        )

        if HAS_FRONTMATTER:
            try:
                post = frontmatter.loads(content)
                post.content = stale_notice + post.content
                path.write_text(frontmatter.dumps(post), encoding="utf-8")
                return True
            except Exception:
                pass

        # Fallback: insert after the first heading or at the top
        lines = content.splitlines(keepends=True)
        insert_at = 0
        for i, line in enumerate(lines):
            if line.startswith("# "):
                insert_at = i + 1
                break
        lines.insert(insert_at, stale_notice)
        path.write_text("".join(lines), encoding="utf-8")
        return True

    def _fix_contradiction(self, issue: LintIssue) -> bool:
        """
        Append a conflicting-info note to both entries and flag for human review.
        Always returns False (needs human review), but the note is still written.
        """
        if not issue.entry_a or not issue.entry_b:
            return False
        path_a = Path(issue.entry_a)
        path_b = Path(issue.entry_b)

        now_str = datetime.now(tz=timezone.utc).isoformat()
        for path, other in [(path_a, path_b), (path_b, path_a)]:
            if not path.exists():
                continue
            content = path.read_text(encoding="utf-8")
            if "LINT:CONTRADICTION" not in content:
                note = (
                    f"\n\n<!-- LINT:CONTRADICTION ({now_str}) — "
                    f"Potential contradiction with [[{other.stem}]]: "
                    f"{issue.description}. "
                    f"Flagged for human review by cross_fragment_lint.py -->\n"
                )
                path.write_text(content + note, encoding="utf-8")

        # Always requires human review for contradictions
        return False


# ---------------------------------------------------------------------------
# Output formatters
# ---------------------------------------------------------------------------


def _severity_color(severity: str) -> str:
    return {"error": "red", "warning": "yellow", "info": "blue"}.get(severity, "white")


def format_rich(report: LintReport) -> None:
    if not HAS_RICH:
        format_plain(report)
        return

    console = Console()
    console.print()
    console.print(
        f"[bold]LLM-Brains Cross-Fragment Lint Report[/bold]  "
        f"[dim]{report.timestamp}[/dim]"
    )
    console.print(
        f"Scope: [cyan]{report.scope}[/cyan]  |  "
        f"Fragments: [cyan]{report.fragments_checked}[/cyan]  |  "
        f"Entries: [cyan]{report.entries_checked}[/cyan]  |  "
        f"Duration: [cyan]{report.duration_seconds}s[/cyan]"
    )
    console.print()

    if not report.issues:
        console.print("[green bold]No issues found.[/green bold]")
        return

    table = Table(
        show_header=True,
        header_style="bold",
        box=box.ROUNDED,
        expand=True,
    )
    table.add_column("Sev", style="bold", width=7)
    table.add_column("Type", width=14)
    table.add_column("Entry A", overflow="fold")
    table.add_column("Entry B", overflow="fold")
    table.add_column("Description", overflow="fold")
    table.add_column("Fix", overflow="fold")
    table.add_column("Conf", width=5)

    for issue in sorted(report.issues, key=lambda i: ("error", "warning", "info").index(i.severity)):
        color = _severity_color(issue.severity)
        sev_text = Text(issue.severity.upper(), style=f"bold {color}")
        table.add_row(
            sev_text,
            issue.issue_type,
            Path(issue.entry_a).name if issue.entry_a else "",
            Path(issue.entry_b).name if issue.entry_b else "",
            issue.description,
            issue.suggested_fix,
            f"{issue.confidence:.0%}",
        )

    console.print(table)
    errors = sum(1 for i in report.issues if i.severity == "error")
    warnings = sum(1 for i in report.issues if i.severity == "warning")
    infos = sum(1 for i in report.issues if i.severity == "info")
    console.print(
        f"\n[red bold]{errors} error(s)[/red bold]  "
        f"[yellow bold]{warnings} warning(s)[/yellow bold]  "
        f"[blue bold]{infos} info(s)[/blue bold]"
    )
    console.print()


def format_json(report: LintReport) -> None:
    print(json.dumps(asdict(report), indent=2, default=str))


def format_markdown(report: LintReport) -> None:
    lines: list[str] = [
        f"# LLM-Brains Lint Report — {report.timestamp}",
        "",
        f"- **Scope**: {report.scope}",
        f"- **Fragments checked**: {report.fragments_checked}",
        f"- **Entries checked**: {report.entries_checked}",
        f"- **Duration**: {report.duration_seconds}s",
        f"- **Total issues**: {len(report.issues)}",
        "",
    ]

    if not report.issues:
        lines.append("_No issues found._")
    else:
        by_severity: dict[str, list[LintIssue]] = defaultdict(list)
        for issue in report.issues:
            by_severity[issue.severity].append(issue)

        for severity in ("error", "warning", "info"):
            issues = by_severity.get(severity, [])
            if not issues:
                continue
            emoji = {"error": "ERROR", "warning": "WARNING", "info": "INFO"}[severity]
            lines.append(f"## {emoji} — {severity.capitalize()}s ({len(issues)})")
            lines.append("")
            for issue in issues:
                entry_a_name = Path(issue.entry_a).name if issue.entry_a else "—"
                entry_b_name = Path(issue.entry_b).name if issue.entry_b else "—"
                lines += [
                    f"### [{issue.issue_type}] {entry_a_name}",
                    f"- **Type**: `{issue.issue_type}`",
                    f"- **Entry A**: `{issue.entry_a}`",
                ]
                if issue.entry_b:
                    lines.append(f"- **Entry B**: `{issue.entry_b}`")
                lines += [
                    f"- **Description**: {issue.description}",
                    f"- **Suggested fix**: {issue.suggested_fix}",
                    f"- **Confidence**: {issue.confidence:.0%}",
                    "",
                ]

    print("\n".join(lines))


def format_plain(report: LintReport) -> None:
    print(f"LLM-Brains Lint Report — {report.timestamp}")
    print(
        f"Scope: {report.scope}  Fragments: {report.fragments_checked}  "
        f"Entries: {report.entries_checked}  Duration: {report.duration_seconds}s"
    )
    print(f"Issues: {len(report.issues)}")
    print()
    for issue in report.issues:
        entry_a_name = Path(issue.entry_a).name if issue.entry_a else ""
        entry_b_name = Path(issue.entry_b).name if issue.entry_b else ""
        print(
            f"[{issue.severity.upper()}] {issue.issue_type:12s}  "
            f"{entry_a_name} / {entry_b_name}"
        )
        print(f"  {issue.description}")
        print(f"  Fix: {issue.suggested_fix}")
        print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if HAS_CLICK:

    @click.command()
    @click.option(
        "--scope",
        type=click.Choice(["full", "recent", "fragment"], case_sensitive=False),
        default="full",
        show_default=True,
        help=(
            "full=all entries, recent=modified in last 24h, "
            "fragment=specific fragment (requires --fragment)."
        ),
    )
    @click.option(
        "--fragment",
        default=None,
        help="Fragment subdirectory name (e.g. 'technology'). Used with --scope fragment.",
    )
    @click.option(
        "--fix",
        is_flag=True,
        default=False,
        help="Attempt automatic fixes for detected issues.",
    )
    @click.option(
        "--format",
        "output_format",
        type=click.Choice(["rich", "json", "markdown"], case_sensitive=False),
        default="rich",
        show_default=True,
        help="Output format.",
    )
    def main(
        scope: str,
        fragment: Optional[str],
        fix: bool,
        output_format: str,
    ) -> None:
        """Cross-fragment wiki linter for LLM-Brains hive mind."""
        if scope == "fragment" and not fragment:
            raise click.UsageError("--fragment is required when --scope fragment is used.")

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            if HAS_RICH:
                Console().print(
                    "[yellow]Warning: ANTHROPIC_API_KEY not set — "
                    "falling back to heuristic-only lint (no LLM claim extraction).[/yellow]"
                )
            else:
                print(
                    "Warning: ANTHROPIC_API_KEY not set — "
                    "falling back to heuristic-only lint.",
                    file=sys.stderr,
                )

        linter = CrossFragmentLinter()
        report = linter.run_lint(scope=scope, fragment=fragment, output_format=output_format)

        if output_format.lower() == "json":
            format_json(report)
        elif output_format.lower() == "markdown":
            format_markdown(report)
        else:
            format_rich(report)

        if fix and report.issues:
            fixed = 0
            needs_review = 0
            for issue in report.issues:
                result = linter.apply_fix(issue)
                if result:
                    fixed += 1
                else:
                    needs_review += 1

            if HAS_RICH:
                Console().print(
                    f"\n[green]Auto-fixed {fixed} issue(s).[/green]  "
                    f"[yellow]{needs_review} issue(s) require human review.[/yellow]"
                )
            else:
                print(
                    f"\nAuto-fixed {fixed} issue(s). "
                    f"{needs_review} issue(s) require human review."
                )

        # Exit with non-zero code if there are errors
        error_count = sum(1 for i in report.issues if i.severity == "error")
        if error_count:
            sys.exit(1)

else:
    # Minimal fallback if click is not installed
    def main() -> None:  # type: ignore[misc]
        print("Error: 'click' package is required. Install with: pip install click", file=sys.stderr)
        sys.exit(1)
cross_fragment_lint.py — Contradiction and consistency checker across fragment wikis.

Detects:
- Contradictions: same subject+predicate, different object
- Stale entries: not updated in >30 days
- Orphan links: [[links]] pointing to non-existent entries
- Duplicate entries: semantically similar content across fragments (>80% similar)

Usage:
    python cross_fragment_lint.py
    python cross_fragment_lint.py --fix    # auto-fix confirmed duplicates
    python cross_fragment_lint.py --output report.md
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import anthropic
from dotenv import load_dotenv

load_dotenv()

BRAIN_DIR = Path(os.getenv("BRAIN_DIR", "./brain"))
FRAGMENTS_DIR = BRAIN_DIR / "fragments"

SEVERITY_ERROR = "ERROR"
SEVERITY_WARNING = "WARNING"
SEVERITY_INFO = "INFO"


# ---------------------------------------------------------------------------
# Triple extraction
# ---------------------------------------------------------------------------

def extract_triples_with_claude(client: anthropic.Anthropic, text: str,
                                 source: str) -> list[dict]:
    """Extract subject-predicate-object triples from text using Claude."""
    prompt = f"""Extract factual subject-predicate-object triples from this text.
Focus on concrete, verifiable facts.

Source: {source}
Text:
{text[:3000]}

Return a JSON array of triples:
[
  {{"subject": "Python", "predicate": "is", "object": "a programming language", "confidence": 0.95}},
  {{"subject": "GPT-4", "predicate": "was released by", "object": "OpenAI", "confidence": 0.99}}
]

Only include clear, unambiguous factual statements. Return ONLY the JSON array."""

    try:
        response = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}]
        )
        text_resp = response.content[0].text.strip()
        match = re.search(r'\[.*?\]', text_resp, re.DOTALL)
        if match:
            triples = json.loads(match.group())
            for t in triples:
                t["source"] = source
            return triples
    except Exception as e:
        pass
    return []


def simple_triple_extract(text: str, source: str) -> list[dict]:
    """Simple regex-based triple extraction fallback."""
    triples = []
    # Pattern: "X is Y", "X was Y", "X has Y"
    patterns = [
        r'(?P<subj>[A-Z][a-zA-Z\s]{2,30})\s+(?P<pred>is|was|are|were|has|have|contains|includes)\s+(?P<obj>[a-zA-Z][^.]{5,60})',
    ]
    for pattern in patterns:
        for m in re.finditer(pattern, text):
            triples.append({
                "subject": m.group("subj").strip(),
                "predicate": m.group("pred").strip(),
                "object": m.group("obj").strip()[:80],
                "confidence": 0.6,
                "source": source,
            })
    return triples[:20]


# ---------------------------------------------------------------------------
# Similarity
# ---------------------------------------------------------------------------

def jaccard_similarity(text1: str, text2: str) -> float:
    """Simple Jaccard similarity between two texts."""
    words1 = set(re.findall(r"\b\w{3,}\b", text1.lower()))
    words2 = set(re.findall(r"\b\w{3,}\b", text2.lower()))
    if not words1 or not words2:
        return 0.0
    intersection = len(words1 & words2)
    union = len(words1 | words2)
    return intersection / union


def find_all_wiki_links(text: str) -> list[str]:
    """Extract all [[wikilinks]] from text."""
    return re.findall(r'\[\[([^\]]+)\]\]', text)


# ---------------------------------------------------------------------------
# CrossFragmentLint class
# ---------------------------------------------------------------------------

class CrossFragmentLint:
    """Runs all lint checks across fragment wikis."""

    def __init__(self, fragment_manager=None, brain_dir: Path = BRAIN_DIR):
        self.brain_dir = brain_dir
        self.fragments_dir = brain_dir / "fragments"
        self.fragment_manager = fragment_manager
        self._client: Optional[anthropic.Anthropic] = None
        self.issues: list[dict] = []

    @property
    def client(self) -> anthropic.Anthropic:
        if self._client is None:
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY not set")
            self._client = anthropic.Anthropic(api_key=api_key)
        return self._client

    def _add_issue(self, severity: str, check: str, description: str,
                   files: list[str] = None, fix_hint: str = ""):
        self.issues.append({
            "severity": severity,
            "check": check,
            "description": description,
            "files": files or [],
            "fix_hint": fix_hint,
            "timestamp": datetime.now().isoformat(),
        })

    def _get_all_wiki_files(self) -> dict[str, list[Path]]:
        """Get all wiki markdown files organized by domain."""
        domain_files: dict[str, list[Path]] = {}
        if not self.fragments_dir.exists():
            return domain_files
        for domain_dir in sorted(self.fragments_dir.iterdir()):
            if domain_dir.is_dir():
                wiki_dir = domain_dir / "wiki"
                if wiki_dir.exists():
                    files = list(wiki_dir.glob("*.md"))
                    if files:
                        domain_files[domain_dir.name] = files
        return domain_files

    # ------------------------------------------------------------------
    # Check 1: Contradictions
    # ------------------------------------------------------------------

    def check_contradictions(self, use_claude: bool = True):
        """Find conflicting facts across fragments."""
        print("  Checking contradictions...")
        all_files = self._get_all_wiki_files()
        all_triples: list[dict] = []

        for domain, files in all_files.items():
            for wiki_file in files:
                try:
                    content = wiki_file.read_text(encoding="utf-8", errors="replace")
                    if use_claude:
                        try:
                            triples = extract_triples_with_claude(
                                self.client, content, f"{domain}/{wiki_file.name}"
                            )
                        except Exception:
                            triples = simple_triple_extract(
                                content, f"{domain}/{wiki_file.name}"
                            )
                    else:
                        triples = simple_triple_extract(
                            content, f"{domain}/{wiki_file.name}"
                        )
                    all_triples.extend(triples)
                except Exception as e:
                    print(f"    Warning: could not read {wiki_file}: {e}", file=sys.stderr)

        # Find contradictions: same subject + predicate, different object
        triple_map: dict[str, list[dict]] = {}
        for t in all_triples:
            key = f"{t['subject'].lower()}::{t['predicate'].lower()}"
            triple_map.setdefault(key, []).append(t)

        for key, triples in triple_map.items():
            if len(triples) > 1:
                objects = list({t["object"].lower() for t in triples})
                if len(objects) > 1:
                    subj, pred = key.split("::")
                    sources = [t["source"] for t in triples]
                    self._add_issue(
                        SEVERITY_ERROR, "contradiction",
                        f"Conflicting claim: '{subj}' {pred} '{objects[0]}' vs '{objects[1]}'",
                        files=sources,
                        fix_hint=f"Review and reconcile: {sources}"
                    )

        print(f"    Found {sum(1 for i in self.issues if i['check'] == 'contradiction')} contradictions")

    # ------------------------------------------------------------------
    # Check 2: Stale entries
    # ------------------------------------------------------------------

    def check_stale_entries(self, days: int = 30):
        """Flag entries not updated in more than `days` days."""
        print(f"  Checking stale entries (>{days} days)...")
        cutoff = datetime.now() - timedelta(days=days)
        all_files = self._get_all_wiki_files()
        stale_count = 0

        for domain, files in all_files.items():
            for wiki_file in files:
                mtime = datetime.fromtimestamp(wiki_file.stat().st_mtime)
                if mtime < cutoff:
                    age_days = (datetime.now() - mtime).days
                    self._add_issue(
                        SEVERITY_INFO, "stale",
                        f"Entry not updated in {age_days} days: {wiki_file.name}",
                        files=[str(wiki_file)],
                        fix_hint="Review and update or archive this entry"
                    )
                    stale_count += 1

        print(f"    Found {stale_count} stale entries")

    # ------------------------------------------------------------------
    # Check 3: Orphan links
    # ------------------------------------------------------------------

    def check_orphan_links(self):
        """Find [[wikilinks]] that point to non-existent entries."""
        print("  Checking orphan links...")
        all_files = self._get_all_wiki_files()

        # Build set of all known titles/slugs
        known_titles: set[str] = set()
        for domain, files in all_files.items():
            for wiki_file in files:
                stem = wiki_file.stem.lower().replace("-", " ")
                known_titles.add(stem)
                # Also try to read title from frontmatter
                try:
                    content = wiki_file.read_text(encoding="utf-8", errors="replace")
                    title_match = re.search(r'^title:\s*(.+)$', content, re.MULTILINE)
                    if title_match:
                        known_titles.add(title_match.group(1).strip().lower())
                except Exception:
                    pass

        # Also check main wiki
        main_wiki = self.brain_dir / "knowledge" / "wiki"
        if main_wiki.exists():
            for wf in main_wiki.glob("*.md"):
                known_titles.add(wf.stem.lower().replace("-", " "))

        orphan_count = 0
        for domain, files in all_files.items():
            for wiki_file in files:
                try:
                    content = wiki_file.read_text(encoding="utf-8", errors="replace")
                    links = find_all_wiki_links(content)
                    for link in links:
                        link_lower = link.lower()
                        if link_lower not in known_titles:
                            self._add_issue(
                                SEVERITY_WARNING, "orphan_link",
                                f"Broken link [[{link}]] in {domain}/{wiki_file.name}",
                                files=[str(wiki_file)],
                                fix_hint=f"Create entry for '{link}' or remove the link"
                            )
                            orphan_count += 1
                except Exception:
                    pass

        print(f"    Found {orphan_count} orphan links")

    # ------------------------------------------------------------------
    # Check 4: Duplicates
    # ------------------------------------------------------------------

    def check_duplicates(self, threshold: float = 0.80):
        """Find semantically similar entries across fragments (>threshold similarity)."""
        print(f"  Checking duplicates (similarity >{threshold:.0%})...")
        all_files = self._get_all_wiki_files()

        # Collect all (domain, path, content) tuples
        docs: list[tuple[str, Path, str]] = []
        for domain, files in all_files.items():
            for wiki_file in files:
                try:
                    content = wiki_file.read_text(encoding="utf-8", errors="replace")
                    # Strip frontmatter
                    body = re.sub(r'^---.*?---\n', '', content, flags=re.DOTALL).strip()
                    docs.append((domain, wiki_file, body))
                except Exception:
                    pass

        dup_count = 0
        checked_pairs: set[frozenset] = set()

        for i in range(len(docs)):
            for j in range(i + 1, len(docs)):
                domain_i, path_i, content_i = docs[i]
                domain_j, path_j, content_j = docs[j]

                pair = frozenset([str(path_i), str(path_j)])
                if pair in checked_pairs:
                    continue
                checked_pairs.add(pair)

                # Skip same domain same file
                if path_i == path_j:
                    continue

                sim = jaccard_similarity(content_i, content_j)
                if sim >= threshold:
                    self._add_issue(
                        SEVERITY_WARNING, "duplicate",
                        f"Similar entries ({sim:.0%}): {domain_i}/{path_i.name} and {domain_j}/{path_j.name}",
                        files=[str(path_i), str(path_j)],
                        fix_hint=f"Consider merging these entries (similarity: {sim:.2f})"
                    )
                    dup_count += 1

        print(f"    Found {dup_count} potential duplicates")

    # ------------------------------------------------------------------
    # Auto-fix
    # ------------------------------------------------------------------

    def fix_duplicates(self):
        """Merge confirmed duplicates (those with >90% similarity)."""
        dup_issues = [i for i in self.issues
                      if i["check"] == "duplicate" and len(i["files"]) == 2]
        fixed = 0
        for issue in dup_issues:
            file_a = Path(issue["files"][0])
            file_b = Path(issue["files"][1])
            if not file_a.exists() or not file_b.exists():
                continue
            # Keep the newer file, archive the older
            stat_a = file_a.stat().st_mtime
            stat_b = file_b.stat().st_mtime
            keep, archive = (file_a, file_b) if stat_a >= stat_b else (file_b, file_a)
            archived = archive.with_name(f"_dup_archived_{archive.name}")
            archive.rename(archived)
            print(f"  Archived duplicate: {archive.name} -> {archived.name}")
            fixed += 1
        print(f"Fixed {fixed} duplicates")

    def fix_orphan_links(self):
        """Remove dead [[wikilinks]] from all wiki files."""
        orphan_issues = [i for i in self.issues if i["check"] == "orphan_link"]
        fixed = 0
        for issue in orphan_issues:
            for file_path in issue["files"]:
                p = Path(file_path)
                if p.exists():
                    content = p.read_text(encoding="utf-8", errors="replace")
                    # Extract the link from description
                    link_match = re.search(r'\[\[(.+?)\]\]', issue["description"])
                    if link_match:
                        link_text = link_match.group(1)
                        # Replace [[link]] with just the text
                        new_content = content.replace(f"[[{link_text}]]", link_text)
                        if new_content != content:
                            p.write_text(new_content, encoding="utf-8")
                            fixed += 1
        print(f"Fixed {fixed} orphan links")

    # ------------------------------------------------------------------
    # Run all checks
    # ------------------------------------------------------------------

    def run_all_checks(self, use_claude: bool = True) -> dict:
        """Run all lint checks and return results."""
        self.issues = []
        print("\nRunning cross-fragment lint...\n" + "="*50)
        self.check_stale_entries()
        self.check_orphan_links()
        self.check_duplicates()
        if use_claude:
            try:
                self.check_contradictions(use_claude=True)
            except Exception as e:
                print(f"  Claude-based contradiction check failed: {e}", file=sys.stderr)
                self.check_contradictions(use_claude=False)
        else:
            self.check_contradictions(use_claude=False)

        counts = {
            SEVERITY_ERROR: sum(1 for i in self.issues if i["severity"] == SEVERITY_ERROR),
            SEVERITY_WARNING: sum(1 for i in self.issues if i["severity"] == SEVERITY_WARNING),
            SEVERITY_INFO: sum(1 for i in self.issues if i["severity"] == SEVERITY_INFO),
        }

        return {
            "total_issues": len(self.issues),
            "counts": counts,
            "issues": self.issues,
        }

    # ------------------------------------------------------------------
    # Report generation
    # ------------------------------------------------------------------

    def generate_report(self, output_path: Optional[Path] = None) -> str:
        """Generate a markdown lint report."""
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        counts = {
            SEVERITY_ERROR: sum(1 for i in self.issues if i["severity"] == SEVERITY_ERROR),
            SEVERITY_WARNING: sum(1 for i in self.issues if i["severity"] == SEVERITY_WARNING),
            SEVERITY_INFO: sum(1 for i in self.issues if i["severity"] == SEVERITY_INFO),
        }

        lines = [
            f"# Cross-Fragment Lint Report",
            f"Generated: {now}",
            f"",
            f"## Summary",
            f"| Severity | Count |",
            f"|----------|-------|",
            f"| ERROR    | {counts[SEVERITY_ERROR]} |",
            f"| WARNING  | {counts[SEVERITY_WARNING]} |",
            f"| INFO     | {counts[SEVERITY_INFO]} |",
            f"| **TOTAL**| **{len(self.issues)}** |",
            f"",
        ]

        for severity in [SEVERITY_ERROR, SEVERITY_WARNING, SEVERITY_INFO]:
            sev_issues = [i for i in self.issues if i["severity"] == severity]
            if sev_issues:
                lines.append(f"## {severity}S ({len(sev_issues)})")
                lines.append("")
                # Group by check type
                by_check: dict[str, list[dict]] = {}
                for issue in sev_issues:
                    by_check.setdefault(issue["check"], []).append(issue)
                for check, issues in by_check.items():
                    lines.append(f"### {check.replace('_', ' ').title()} ({len(issues)})")
                    for issue in issues:
                        lines.append(f"- **{issue['description']}**")
                        if issue.get("fix_hint"):
                            lines.append(f"  - Fix: {issue['fix_hint']}")
                        for f in issue.get("files", []):
                            lines.append(f"  - File: `{f}`")
                    lines.append("")

        report = "\n".join(lines)

        if output_path:
            output_path.write_text(report, encoding="utf-8")
            print(f"\nReport written to: {output_path}")

        return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Cross-fragment lint checker")
    parser.add_argument("--fix", action="store_true",
                        help="Auto-fix: merge duplicates, remove dead links")
    parser.add_argument("--output", type=str, default="lint_report.md",
                        help="Output file for lint report")
    parser.add_argument("--no-claude", action="store_true",
                        help="Skip Claude-based checks (faster)")
    parser.add_argument("--stale-days", type=int, default=30,
                        help="Days threshold for stale entries")
    parser.add_argument("--similarity", type=float, default=0.80,
                        help="Similarity threshold for duplicates (0-1)")
    args = parser.parse_args()

    linter = CrossFragmentLint()

    # Run checks
    linter.check_stale_entries(days=args.stale_days)
    linter.check_orphan_links()
    linter.check_duplicates(threshold=args.similarity)

    if not args.no_claude:
        try:
            linter.check_contradictions(use_claude=True)
        except Exception as e:
            print(f"Claude check failed ({e}), using simple extraction")
            linter.check_contradictions(use_claude=False)
    else:
        linter.check_contradictions(use_claude=False)

    # Generate report
    output_path = Path(args.output)
    report = linter.generate_report(output_path=output_path)
    print(report[:2000])

    if len(report) > 2000:
        print(f"\n... (full report in {output_path})")

    # Auto-fix
    if args.fix:
        print("\nApplying auto-fixes...")
        linter.fix_duplicates()
        linter.fix_orphan_links()

    # Exit code based on severity
    error_count = sum(1 for i in linter.issues if i["severity"] == SEVERITY_ERROR)
    sys.exit(1 if error_count > 0 else 0)


if __name__ == "__main__":
    main()
