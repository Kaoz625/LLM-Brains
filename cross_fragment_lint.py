#!/usr/bin/env python3
"""
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
