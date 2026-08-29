#!/usr/bin/env python3
"""
fragment_manager.py — Hive-mind fragment agent system.

Manages 20 domain-specialized fragment agents, each with their own wiki
and SQLite store. Fragments are queried in parallel and results synthesized.

Usage:
    from fragment_manager import FragmentManager
    fm = FragmentManager()
    results = fm.route("What do I know about transformer architectures?")
"""

import json
import os
import re
import sqlite3
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Optional

import anthropic
from dotenv import load_dotenv

load_dotenv()

BRAIN_DIR = Path(os.getenv("BRAIN_DIR", "./brain"))
FRAGMENTS_DIR = BRAIN_DIR / "fragments"

DOMAINS = [
    "geography", "people", "science", "technology", "history",
    "philosophy", "health", "creative", "business", "personal_memory",
    "ai_ml", "code", "media", "relationships", "events",
    "concepts", "emotions", "skills", "projects", "misc",
]

DOMAIN_DESCRIPTIONS = {
    "geography":       "Physical places, maps, travel routes, locations, cities, countries",
    "people":          "Public figures, historical persons, celebrities, experts",
    "science":         "Physics, chemistry, biology, mathematics, formal sciences",
    "technology":      "Software, hardware, tools, engineering, systems design",
    "history":         "Historical events, timelines, civilizations, eras",
    "philosophy":      "Ethics, epistemology, logic, metaphysics, world-views",
    "health":          "Medicine, fitness, nutrition, mental health, biology",
    "creative":        "Art, music, writing, design, games, imagination",
    "business":        "Economics, finance, startups, management, markets",
    "personal_memory": "Personal experiences, episodic memories, autobiographical notes",
    "ai_ml":           "Machine learning, neural networks, LLMs, AI research",
    "code":            "Programming languages, algorithms, code snippets, repos",
    "media":           "Books, films, podcasts, articles, videos consumed",
    "relationships":   "Personal relationships, social dynamics, communication",
    "events":          "Upcoming/past events, meetings, appointments, milestones",
    "concepts":        "Abstract ideas, mental models, frameworks, terminology",
    "emotions":        "Emotional experiences, mood patterns, psychological states",
    "skills":          "Learned abilities, expertise, competencies, certifications",
    "projects":        "Active/completed projects, goals, plans, roadmaps",
    "misc":            "Uncategorized facts and notes that don't fit elsewhere",
}

FRAGMENT_SCHEMA = """
CREATE TABLE IF NOT EXISTS entries (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    title       TEXT NOT NULL,
    content     TEXT NOT NULL,
    tags        TEXT,
    source_path TEXT,
    created_at  TEXT DEFAULT (datetime('now')),
    updated_at  TEXT DEFAULT (datetime('now'))
);

CREATE VIRTUAL TABLE IF NOT EXISTS entries_fts USING fts5(
    title,
    content,
    tags,
    content='entries',
    content_rowid='id'
);

CREATE TRIGGER IF NOT EXISTS ent_ai AFTER INSERT ON entries BEGIN
    INSERT INTO entries_fts(rowid, title, content, tags)
    VALUES (new.id, new.title, new.content, new.tags);
END;
CREATE TRIGGER IF NOT EXISTS ent_ad AFTER DELETE ON entries BEGIN
    INSERT INTO entries_fts(entries_fts, rowid, title, content, tags)
    VALUES ('delete', old.id, old.title, old.content, old.tags);
END;
CREATE TRIGGER IF NOT EXISTS ent_au AFTER UPDATE ON entries BEGIN
    INSERT INTO entries_fts(entries_fts, rowid, title, content, tags)
    VALUES ('delete', old.id, old.title, old.content, old.tags);
    INSERT INTO entries_fts(rowid, title, content, tags)
    VALUES (new.id, new.title, new.content, new.tags);
END;
"""


# ---------------------------------------------------------------------------
# Fragment class
# ---------------------------------------------------------------------------

class Fragment:
    """A single domain-specialist fragment agent."""

    def __init__(self, domain: str, brain_dir: Path = BRAIN_DIR):
        self.domain = domain
        self.description = DOMAIN_DESCRIPTIONS.get(domain, domain)
        self.dir = brain_dir / "fragments" / domain
        self.wiki_dir = self.dir / "wiki"
        self.db_path = self.dir / f"{domain}.db"
        self._ensure_dirs()
        self._init_db()

    def _ensure_dirs(self):
        self.wiki_dir.mkdir(parents=True, exist_ok=True)

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_db(self):
        conn = self._get_conn()
        conn.executescript(FRAGMENT_SCHEMA)
        conn.commit()
        conn.close()

    def add_entry(self, title: str, content: str, tags: str = "",
                  source_path: str = "") -> int:
        """Add or update an entry in this fragment's store."""
        conn = self._get_conn()
        conn.execute("""
            INSERT INTO entries (title, content, tags, source_path)
            VALUES (?, ?, ?, ?)
        """, (title, content, tags, source_path))
        conn.commit()
        row_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.close()

        # Also write wiki file
        slug = re.sub(r"[^\w-]", "-", title.lower())[:60]
        wiki_file = self.wiki_dir / f"{slug}.md"
        wiki_file.write_text(
            f"---\ntitle: {title}\ndomain: {self.domain}\n"
            f"tags: {tags}\nupdated: {datetime.now().isoformat()}\n---\n\n"
            f"# {title}\n\n{content}\n",
            encoding="utf-8"
        )
        return row_id

    def query(self, query_text: str, limit: int = 5) -> list[dict]:
        """Search this fragment's knowledge base."""
        conn = self._get_conn()
        results = []
        # FTS search
        try:
            rows = conn.execute("""
                SELECT e.id, e.title, e.content, e.tags, e.updated_at,
                       bm25(entries_fts) AS score
                FROM entries_fts
                JOIN entries e ON entries_fts.rowid = e.id
                WHERE entries_fts MATCH ?
                ORDER BY score
                LIMIT ?
            """, (query_text, limit)).fetchall()
            results = [dict(r) for r in rows]
        except Exception:
            # Fallback: LIKE search
            rows = conn.execute("""
                SELECT id, title, content, tags, updated_at
                FROM entries
                WHERE content LIKE ? OR title LIKE ?
                LIMIT ?
            """, (f"%{query_text}%", f"%{query_text}%", limit)).fetchall()
            results = [dict(r) for r in rows]

        # Also scan wiki files for any that FTS might miss
        wiki_hits = self._scan_wiki(query_text, limit=3)
        conn.close()

        # Merge, deduplicating by title
        seen_titles = {r["title"] for r in results}
        for h in wiki_hits:
            if h["title"] not in seen_titles:
                results.append(h)
                seen_titles.add(h["title"])

        return results[:limit]

    def _scan_wiki(self, query: str, limit: int = 3) -> list[dict]:
        """Scan wiki markdown files for query matches."""
        query_lower = query.lower()
        hits = []
        for md_file in sorted(self.wiki_dir.glob("*.md")):
            try:
                content = md_file.read_text(encoding="utf-8", errors="replace")
                if query_lower in content.lower():
                    hits.append({
                        "title": md_file.stem.replace("-", " "),
                        "content": content[:500],
                        "tags": "",
                        "updated_at": datetime.fromtimestamp(
                            md_file.stat().st_mtime
                        ).isoformat(),
                        "score": content.lower().count(query_lower),
                    })
            except Exception:
                continue
        hits.sort(key=lambda x: x.get("score", 0), reverse=True)
        return hits[:limit]

    def list_entries(self) -> list[dict]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT id, title, tags, updated_at FROM entries ORDER BY updated_at DESC"
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def get_wiki_summary(self) -> str:
        """Return a brief summary of this fragment's knowledge."""
        entries = self.list_entries()
        wiki_files = list(self.wiki_dir.glob("*.md"))
        return (
            f"Domain: {self.domain}\n"
            f"Description: {self.description}\n"
            f"DB entries: {len(entries)}\n"
            f"Wiki files: {len(wiki_files)}\n"
            f"Recent: {', '.join(e['title'] for e in entries[:5])}"
        )

    def __repr__(self) -> str:
        return f"Fragment({self.domain})"


# ---------------------------------------------------------------------------
# Fragment Manager
# ---------------------------------------------------------------------------

class FragmentManager:
    """Manages all 20 fragment agents and routes queries to relevant ones."""

    def __init__(self, brain_dir: Path = BRAIN_DIR, max_workers: int = 8):
        self.brain_dir = brain_dir
        self.max_workers = max_workers
        self.fragments: dict[str, Fragment] = {
            d: Fragment(d, brain_dir) for d in DOMAINS
        }
        self._client: Optional[anthropic.Anthropic] = None

    @property
    def client(self) -> anthropic.Anthropic:
        if self._client is None:
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY not set")
            self._client = anthropic.Anthropic(api_key=api_key)
        return self._client

    def identify_relevant_fragments(self, query: str, top_k: int = 4) -> list[str]:
        """Use Claude to identify which fragment domains are most relevant."""
        domain_list = "\n".join(
            f"- {d}: {DOMAIN_DESCRIPTIONS[d]}" for d in DOMAINS
        )
        prompt = f"""Given this query, identify the top {top_k} most relevant knowledge domains.

Query: {query}

Available domains:
{domain_list}

Return a JSON array of domain names (from the list above) that are most relevant, ordered by relevance.
Example: ["ai_ml", "technology", "concepts", "science"]
Return ONLY the JSON array, nothing else."""

        try:
            response = self.client.messages.create(
                model="claude-haiku-4-5",
                max_tokens=256,
                messages=[{"role": "user", "content": prompt}]
            )
            text = response.content[0].text.strip()
            match = re.search(r'\[.*?\]', text, re.DOTALL)
            if match:
                domains = json.loads(match.group())
                # Validate
                return [d for d in domains if d in self.fragments][:top_k]
        except Exception as e:
            print(f"Fragment routing error: {e}", file=sys.stderr)

        # Fallback: keyword matching
        query_lower = query.lower()
        scores = {}
        for domain, desc in DOMAIN_DESCRIPTIONS.items():
            score = sum(1 for w in query_lower.split() if w in desc.lower())
            scores[domain] = score
        top = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [d for d, _ in top[:top_k]]

    def route(self, query: str, top_k: int = 4) -> dict:
        """
        Route a query to the most relevant fragments and query them in parallel.
        Returns synthesized results.
        """
        relevant_domains = self.identify_relevant_fragments(query, top_k=top_k)
        print(f"  Routing to fragments: {relevant_domains}")

        fragment_results: dict[str, list[dict]] = {}

        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(relevant_domains))) as ex:
            future_to_domain = {
                ex.submit(self.fragments[d].query, query): d
                for d in relevant_domains
                if d in self.fragments
            }
            for future in as_completed(future_to_domain):
                domain = future_to_domain[future]
                try:
                    fragment_results[domain] = future.result()
                except Exception as e:
                    fragment_results[domain] = [{"error": str(e)}]

        return {
            "query": query,
            "domains_queried": relevant_domains,
            "results": fragment_results,
            "total_hits": sum(len(v) for v in fragment_results.values()),
        }

    def route_and_synthesize(self, query: str, top_k: int = 4) -> str:
        """Route query, get results, then synthesize a cohesive answer with Claude."""
        raw = self.route(query, top_k=top_k)

        # Build context from fragment results
        context_parts = []
        for domain, results in raw["results"].items():
            if results:
                context_parts.append(f"\n## {domain.upper()} FRAGMENT\n")
                for r in results[:3]:
                    if "error" not in r:
                        context_parts.append(
                            f"**{r.get('title', 'Untitled')}**\n"
                            f"{r.get('content', '')[:400]}\n"
                        )

        if not context_parts:
            return f"No relevant information found for: {query}"

        context = "\n".join(context_parts)
        prompt = f"""Using the following knowledge fragments, answer this query:

Query: {query}

Fragment Context:
{context[:6000]}

Synthesize a clear, comprehensive answer. Use [[wikilinks]] for key concepts.
If information is incomplete or contradictory, note it explicitly."""

        try:
            response = self.client.messages.create(
                model="claude-haiku-4-5",
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        except Exception as e:
            return f"Synthesis error: {e}\n\nRaw results:\n{json.dumps(raw, indent=2)}"

    def ingest_entry(self, content: str, title: str = "", source_path: str = ""):
        """Ingest a new entry, routing it to the correct fragment(s)."""
        # Identify relevant domains for this content
        relevant = self.identify_relevant_fragments(content[:500], top_k=2)
        for domain in relevant:
            self.fragments[domain].add_entry(
                title=title or "Untitled",
                content=content,
                source_path=source_path,
            )
            print(f"  Ingested into fragment: {domain}")

    def lint(self) -> dict:
        """Check for contradictions across fragment wikis."""
        # Delegate to cross_fragment_lint module
        try:
            from cross_fragment_lint import CrossFragmentLint
            linter = CrossFragmentLint(self)
            return linter.run_all_checks()
        except ImportError:
            return self._basic_lint()

    def _basic_lint(self) -> dict:
        """Basic lint: check for duplicate titles across fragments."""
        all_titles: dict[str, list[str]] = {}
        for domain, frag in self.fragments.items():
            for entry in frag.list_entries():
                t = entry["title"].lower()
                all_titles.setdefault(t, []).append(domain)

        duplicates = {t: domains for t, domains in all_titles.items() if len(domains) > 1}
        return {
            "duplicates": duplicates,
            "duplicate_count": len(duplicates),
            "total_entries": sum(len(f.list_entries()) for f in self.fragments.values()),
        }

    def status(self) -> dict:
        """Return status of all fragments."""
        return {
            domain: {
                "entries": len(frag.list_entries()),
                "wiki_files": len(list(frag.wiki_dir.glob("*.md"))),
            }
            for domain, frag in self.fragments.items()
        }

    def __repr__(self) -> str:
        total = sum(len(f.list_entries()) for f in self.fragments.values())
        return f"FragmentManager({len(self.fragments)} domains, {total} total entries)"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Fragment manager CLI")
    parser.add_argument("--query", "-q", type=str, help="Query all fragments")
    parser.add_argument("--status", action="store_true", help="Show fragment status")
    parser.add_argument("--lint", action="store_true", help="Run cross-fragment lint")
    parser.add_argument("--ingest", type=str, help="Ingest a file into fragments")
    parser.add_argument("--domain", type=str, help="Specific domain to query")
    args = parser.parse_args()

    fm = FragmentManager()

    if args.status:
        status = fm.status()
        print("\nFragment Status:\n" + "="*50)
        for domain, info in status.items():
            print(f"  {domain:20s}: {info['entries']:4d} entries, "
                  f"{info['wiki_files']:3d} wiki files")
        total = sum(i["entries"] for i in status.values())
        print(f"\n  TOTAL: {total} entries across {len(status)} fragments")

    elif args.lint:
        report = fm.lint()
        print(f"\nLint Report:\n{json.dumps(report, indent=2)}")

    elif args.query:
        if args.domain and args.domain in fm.fragments:
            results = fm.fragments[args.domain].query(args.query)
            print(f"\nResults from {args.domain}:")
            for r in results:
                print(f"  [{r.get('title')}] {r.get('content', '')[:200]}")
        else:
            answer = fm.route_and_synthesize(args.query)
            print(f"\nAnswer:\n{answer}")

    elif args.ingest:
        path = Path(args.ingest)
        content = path.read_text(encoding="utf-8", errors="replace")
        fm.ingest_entry(content, title=path.stem, source_path=str(path))
        print(f"Ingested: {path.name}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
