"""
orchestrator.py — The Main Brain

The Orchestrator is the top-level reasoning agent that:
  1. Accepts user queries
  2. Routes to the correct fragment(s) via tool calls
  3. Executes ALL external tool calls (fragments only REQUEST, never execute)
  4. Synthesizes final answers from fragment responses
  5. Triggers cross-fragment lint passes on schedule
  6. Files new knowledge back into the wiki

Architecture note: This is the ONLY layer that executes tool calls.
Fragments emit structured requests; the orchestrator fulfills them.
This is the key design principle — fragments stay lean and deterministic.

Tool-calling model recommendation:
  - Best open-source: xLAM-2 8b (Salesforce) — #1 on Berkeley BFCL leaderboard
  - Best lightweight: Hammer 2.1 7B (MadeAgents) — designed for on-device
  - Best 1-bit: BitNet b1.58 2B4T fine-tuned for function calling (2.3s/req on CPU)
  - Default here: claude-sonnet-4-6 via Anthropic API
"""

import json
import logging
import os
import sqlite3
#!/usr/bin/env python3
"""
orchestrator.py — Main brain orchestrator agent with tool-calling loop.

Full Claude tool-calling loop. Routes queries to fragments, synthesizes answers,
and files results back into the wiki.

Usage:
    python orchestrator.py                    # interactive REPL
    python orchestrator.py --query "question" # single-shot
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import anthropic
import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

BRAIN_DIR = Path("brain")
WIKI_DB = BRAIN_DIR / "brain.db"
ORCHESTRATOR_LOG = Path("orchestrator.log")

# ─────────────────────────────────────────────────────────────
# Tool definitions — the orchestrator's full tool palette
# These are exposed to the LLM as callable functions
# ─────────────────────────────────────────────────────────────

TOOLS = [
    {
        "name": "search_brain",
        "description": (
            "Search the compiled brain wiki using hybrid FTS5 keyword + semantic similarity. "
            "Use this to retrieve pre-compiled knowledge before answering."
from dotenv import load_dotenv

load_dotenv()

BRAIN_DIR = Path(os.getenv("BRAIN_DIR", "./brain"))

# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "name": "search_wiki",
        "description": (
            "Search the personal wiki knowledge base using full-text + semantic search. "
            "Returns relevant wiki articles and their content."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "fragment": {
                    "type": "string",
                    "description": "Optional: restrict search to a specific fragment domain (e.g. 'people', 'science', 'geography')",
                },
                "limit": {"type": "integer", "description": "Max results to return", "default": 5},
                "query": {"type": "string", "description": "The search query"},
                "limit": {"type": "integer", "description": "Max results (default 5)", "default": 5},
            },
            "required": ["query"],
        },
    },
    {
        "name": "query_fragment",
        "description": (
            "Route a sub-question to a specific knowledge fragment (domain expert). "
            "The fragment reads its compiled wiki and returns a structured knowledge response. "
            "Use when the question maps clearly to one domain."
            "Query a specific domain fragment agent. "
            "Domains: geography, people, science, technology, history, philosophy, "
            "health, creative, business, personal_memory, ai_ml, code, media, "
            "relationships, events, concepts, emotions, skills, projects, misc."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "fragment_name": {
                    "type": "string",
                    "description": "Fragment domain: people, science, technology, history, geography, health, media, work, philosophy",
                },
                "question": {"type": "string", "description": "The sub-question to route to this fragment"},
                "context": {"type": "string", "description": "Any relevant context from other fragments"},
            },
            "required": ["fragment_name", "question"],
        },
    },
    {
        "name": "read_wiki_entry",
        "description": "Read a specific compiled wiki entry by title or path.",
        "input_schema": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Title or partial title of the wiki entry"},
                "fragment": {"type": "string", "description": "Optional: fragment domain to scope search"},
            },
            "required": ["title"],
        },
    },
    {
        "name": "write_wiki_entry",
        "description": (
            "File new knowledge into the wiki after answering a query. "
            "This is how the brain grows smarter — every answer that produces new synthesis "
            "gets filed back as a new or updated wiki entry."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Title of the wiki entry"},
                "content": {"type": "string", "description": "Markdown content with [[wikilinks]]"},
                "fragment": {"type": "string", "description": "Fragment domain this belongs to"},
                "operation": {
                    "type": "string",
                    "enum": ["CREATE", "UPDATE", "MERGE", "SUPERSEDE"],
                    "description": "How to handle this write relative to existing content",
                },
            },
            "required": ["title", "content", "fragment", "operation"],
        },
    },
    {
        "name": "run_lint_pass",
        "description": (
            "Run a cross-fragment lint pass to detect contradictions, stale entries, "
            "and missing cross-links across all fragment wikis. "
            "Like a compiler checking for errors across files. "
            "Run periodically or after major knowledge updates."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "scope": {
                    "type": "string",
                    "enum": ["full", "recent", "fragment"],
                    "description": "full=all wikis, recent=last 24h changes, fragment=one domain",
                },
                "fragment": {"type": "string", "description": "Required if scope=fragment"},
            },
            "required": ["scope"],
        },
    },
    {
        "name": "get_brain_stats",
        "description": "Get statistics about the brain: total entries, fragment sizes, last lint pass, growth over time.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
]

# ─────────────────────────────────────────────────────────────
# Tool implementations
# ─────────────────────────────────────────────────────────────

def _get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(WIKI_DB)
    conn.row_factory = sqlite3.Row
    return conn


def tool_search_brain(query: str, fragment: str | None = None, limit: int = 5) -> dict:
    """Hybrid FTS5 keyword search against compiled wiki."""
    if not WIKI_DB.exists():
        return {"results": [], "message": "Brain database not yet built. Run: python sqlite_rag.py --index"}

    try:
        conn = _get_db()
        where_clause = "WHERE notes_fts MATCH ?"
        params: list[Any] = [query]

        if fragment:
            where_clause += " AND n.fragment = ?"
            params.append(fragment)

        rows = conn.execute(
            f"""
            SELECT n.path, n.title, n.fragment,
                   snippet(notes_fts, 1, '[', ']', '...', 20) AS snippet,
                   rank
            FROM notes_fts
            JOIN notes n ON notes_fts.rowid = n.id
            {where_clause}
            ORDER BY rank
            LIMIT ?
            """,
            params + [limit],
        ).fetchall()
        conn.close()

        return {
            "results": [
                {"path": r["path"], "title": r["title"], "fragment": r["fragment"], "snippet": r["snippet"]}
                for r in rows
            ],
            "count": len(rows),
        }
    except Exception as e:
        return {"error": str(e), "results": []}


def tool_query_fragment(fragment_name: str, question: str, context: str = "") -> dict:
    """Route a question to a fragment's compiled wiki domain."""
    fragment_dir = BRAIN_DIR / fragment_name
    wiki_index = fragment_dir / "index.md"

    if not fragment_dir.exists():
        return {
            "fragment": fragment_name,
            "answer": f"Fragment '{fragment_name}' does not yet exist. It will be created as content is ingested.",
            "entries": [],
        }

    # Gather relevant wiki entries from this fragment
    entries = list(fragment_dir.glob("**/*.md"))
    relevant_content = []

    for entry in entries[:10]:  # cap at 10 entries for context
        try:
            text = entry.read_text(encoding="utf-8")
            if any(word.lower() in text.lower() for word in question.split()):
                relevant_content.append({"file": str(entry.relative_to(BRAIN_DIR)), "content": text[:2000]})
        except Exception:
            continue

    return {
        "fragment": fragment_name,
        "question": question,
        "relevant_entries": relevant_content,
        "entry_count": len(entries),
        "message": f"Found {len(relevant_content)} relevant entries in {fragment_name} fragment.",
    }


def tool_read_wiki_entry(title: str, fragment: str | None = None) -> dict:
    """Read a specific wiki entry by title."""
    search_dirs = [BRAIN_DIR / fragment] if fragment else [BRAIN_DIR]

    for search_dir in search_dirs:
        if not search_dir.exists():
            continue
        for md_file in search_dir.rglob("*.md"):
            content = md_file.read_text(encoding="utf-8")
            first_line = content.split("\n")[0].lstrip("#").strip()
            if title.lower() in first_line.lower() or title.lower() in md_file.stem.lower():
                return {
                    "path": str(md_file.relative_to(BRAIN_DIR)),
                    "title": first_line,
                    "content": content,
                    "found": True,
                }

    return {"found": False, "message": f"No wiki entry found matching '{title}'"}


def tool_write_wiki_entry(title: str, content: str, fragment: str, operation: str) -> dict:
    """Write or update a wiki entry, filing new knowledge back into the brain."""
    fragment_dir = BRAIN_DIR / fragment
    fragment_dir.mkdir(parents=True, exist_ok=True)

    safe_title = title.lower().replace(" ", "-").replace("/", "-")
    entry_path = fragment_dir / f"{safe_title}.md"

    timestamp = datetime.now().isoformat()

    if operation == "CREATE" or not entry_path.exists():
        full_content = f"# {title}\n\n_Created by orchestrator: {timestamp}_\n\n{content}"
        entry_path.write_text(full_content, encoding="utf-8")
        action = "created"
    elif operation == "UPDATE":
        existing = entry_path.read_text(encoding="utf-8")
        full_content = existing + f"\n\n---\n_Updated: {timestamp}_\n\n{content}"
        entry_path.write_text(full_content, encoding="utf-8")
        action = "updated"
    elif operation == "MERGE":
        existing = entry_path.read_text(encoding="utf-8")
        full_content = existing + f"\n\n### Merged knowledge ({timestamp})\n\n{content}"
        entry_path.write_text(full_content, encoding="utf-8")
        action = "merged"
    elif operation == "SUPERSEDE":
        full_content = f"# {title}\n\n_Superseded: {timestamp}_\n\n{content}"
        entry_path.write_text(full_content, encoding="utf-8")
        action = "superseded"
    else:
        action = "no-op"

    log.info(f"Wiki entry {action}: {entry_path}")
    return {"path": str(entry_path), "action": action, "title": title}


def tool_run_lint_pass(scope: str, fragment: str | None = None) -> dict:
    """Trigger a cross-fragment lint pass."""
    try:
        import subprocess
        cmd = ["python", "cross_fragment_lint.py", f"--scope={scope}"]
        if fragment:
            cmd.append(f"--fragment={fragment}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        return {
            "scope": scope,
            "stdout": result.stdout[-3000:] if result.stdout else "",
            "stderr": result.stderr[-1000:] if result.stderr else "",
            "returncode": result.returncode,
        }
    except FileNotFoundError:
        return {"error": "cross_fragment_lint.py not found. Run from LLM-Brains root directory."}
    except Exception as e:
        return {"error": str(e)}


def tool_get_brain_stats() -> dict:
    """Get brain statistics."""
    stats: dict[str, Any] = {"fragments": {}, "total_entries": 0, "total_size_kb": 0}

    for item in BRAIN_DIR.iterdir():
        if item.is_dir() and not item.name.startswith(".") and item.name != "raw":
            md_files = list(item.rglob("*.md"))
            size_kb = sum(f.stat().st_size for f in md_files) // 1024
            stats["fragments"][item.name] = {"entries": len(md_files), "size_kb": size_kb}
            stats["total_entries"] += len(md_files)
            stats["total_size_kb"] += size_kb

    if WIKI_DB.exists():
        stats["database_size_kb"] = WIKI_DB.stat().st_size // 1024

    return stats


# ─────────────────────────────────────────────────────────────
# Tool dispatcher
# ─────────────────────────────────────────────────────────────

TOOL_DISPATCH = {
    "search_brain": lambda i: tool_search_brain(**i),
    "query_fragment": lambda i: tool_query_fragment(**i),
    "read_wiki_entry": lambda i: tool_read_wiki_entry(**i),
    "write_wiki_entry": lambda i: tool_write_wiki_entry(**i),
    "run_lint_pass": lambda i: tool_run_lint_pass(**i),
    "get_brain_stats": lambda i: tool_get_brain_stats(),
}


def execute_tool(tool_name: str, tool_input: dict) -> str:
    """Execute a tool call and return JSON string result."""
    if tool_name not in TOOL_DISPATCH:
        return json.dumps({"error": f"Unknown tool: {tool_name}"})
    try:
        result = TOOL_DISPATCH[tool_name](tool_input)
        return json.dumps(result, default=str)
    except Exception as e:
        log.error(f"Tool {tool_name} failed: {e}")
        return json.dumps({"error": str(e)})


# ─────────────────────────────────────────────────────────────
# Orchestrator loop
# ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are the Orchestrator — the main brain of a personal second-brain system.

Your role:
1. Answer questions by searching and reading from the compiled wiki fragments
2. Route sub-questions to the appropriate domain fragment
3. Synthesize answers from multiple fragment responses
4. File new synthesized knowledge back into the wiki (it should grow smarter with every query)
5. Periodically trigger lint passes to maintain consistency across fragments

Fragment domains available: people, science, technology, history, geography, health, media, work, philosophy, knowledge, me

Design principle: You are the ONLY layer that executes tool calls. Fragments only request.
This keeps fragments lean and deterministic — you handle all I/O.

When answering:
- Always search the brain first before relying on your own training
- If you synthesize new knowledge that isn't yet in the wiki, file it back with write_wiki_entry
- Use [[wikilink]] syntax when referencing related concepts
- Be explicit about what came from the wiki vs. your own reasoning

Cross-fragment lint: Run run_lint_pass periodically (after 10+ queries or when you detect drift)."""


def run_query(query: str, verbose: bool = False) -> str:
    """Run a query through the orchestrator with full tool-calling loop."""
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    messages = [{"role": "user", "content": query}]

    with ORCHESTRATOR_LOG.open("a") as log_file:
        log_file.write(f"\n{'='*60}\n[{datetime.now().isoformat()}] QUERY: {query}\n")

    iteration = 0
    max_iterations = 10  # prevent runaway loops

    while iteration < max_iterations:
        iteration += 1

        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )

        if verbose:
            console.print(f"[dim]Iteration {iteration}: stop_reason={response.stop_reason}[/dim]")

        # Collect text and tool uses from this response
        tool_uses = []
        text_blocks = []

        for block in response.content:
            if block.type == "text":
                text_blocks.append(block.text)
            elif block.type == "tool_use":
                tool_uses.append(block)

        # If no tool calls, we're done
        if response.stop_reason == "end_turn" or not tool_uses:
            final_answer = "\n".join(text_blocks)
            with ORCHESTRATOR_LOG.open("a") as log_file:
                log_file.write(f"ANSWER: {final_answer[:500]}...\n")
            return final_answer

        # Execute tool calls
        tool_results = []
        for tool_use in tool_uses:
            if verbose:
                console.print(f"[yellow]→ Tool: {tool_use.name}({json.dumps(tool_use.input, indent=2)[:200]})[/yellow]")

            result_str = execute_tool(tool_use.name, tool_use.input)

            if verbose:
                console.print(f"[green]← Result: {result_str[:300]}[/green]")

            with ORCHESTRATOR_LOG.open("a") as log_file:
                log_file.write(f"  TOOL: {tool_use.name} → {result_str[:200]}\n")

            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tool_use.id,
                "content": result_str,
            })

        # Add assistant response + tool results to messages
        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": tool_results})

    return "Max iterations reached. Partial results may be incomplete."


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

@click.group()
def cli():
    """LLM-Brains Orchestrator — The Main Brain"""


@cli.command()
@click.argument("query")
@click.option("--verbose", "-v", is_flag=True, help="Show tool calls and results")
def ask(query: str, verbose: bool):
    """Ask a question — the orchestrator searches wiki, routes to fragments, synthesizes answer."""
    console.print(Panel(f"[bold]Query:[/bold] {query}", style="blue"))

    answer = run_query(query, verbose=verbose)
    console.print(Panel(answer, title="[bold green]Answer[/bold green]", style="green"))


@cli.command()
def stats():
    """Show brain statistics."""
    data = tool_get_brain_stats()
    table = Table(title="Brain Statistics")
    table.add_column("Fragment", style="cyan")
    table.add_column("Entries", justify="right")
    table.add_column("Size (KB)", justify="right")

    for name, info in data.get("fragments", {}).items():
        table.add_row(name, str(info["entries"]), str(info["size_kb"]))

    table.add_row("[bold]TOTAL[/bold]", str(data["total_entries"]), str(data["total_size_kb"]), style="bold")
    console.print(table)
    if "database_size_kb" in data:
        console.print(f"[dim]SQLite DB: {data['database_size_kb']} KB[/dim]")


@cli.command()
@click.option("--scope", default="recent", type=click.Choice(["full", "recent", "fragment"]))
@click.option("--fragment", default=None)
def lint(scope: str, fragment: str | None):
    """Run a cross-fragment lint pass to detect contradictions."""
    console.print(f"[yellow]Running lint pass: scope={scope}...[/yellow]")
    result = tool_run_lint_pass(scope, fragment)
    if result.get("stdout"):
        console.print(result["stdout"])
    if result.get("error"):
        console.print(f"[red]Error: {result['error']}[/red]")


@cli.command()
def chat():
    """Interactive chat mode with the orchestrator."""
    console.print(Panel("[bold]LLM-Brains Orchestrator[/bold]\nType 'exit' to quit, 'stats' for brain stats, 'lint' to run lint pass.", style="blue"))

    while True:
        try:
            query = console.input("[bold cyan]You:[/bold cyan] ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not query:
            continue
        if query.lower() == "exit":
            break
        if query.lower() == "stats":
            stats.callback()
            continue
        if query.lower() == "lint":
            result = tool_run_lint_pass("recent")
            console.print(result.get("stdout", "Lint complete."))
            continue

        answer = run_query(query, verbose=False)
        console.print(f"\n[bold green]Brain:[/bold green] {answer}\n")


if __name__ == "__main__":
    cli()
                "domain": {"type": "string", "description": "The domain fragment to query"},
                "query": {"type": "string", "description": "The query text"},
            },
            "required": ["domain", "query"],
        },
    },
    {
        "name": "ingest_file",
        "description": "Ingest a file into the brain (process and store in wiki/fragments).",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path to ingest"},
            },
            "required": ["path"],
        },
    },
    {
        "name": "run_lint",
        "description": "Run the cross-fragment lint checker to find contradictions and issues.",
        "input_schema": {
            "type": "object",
            "properties": {
                "fix": {"type": "boolean", "description": "Auto-fix issues if True", "default": False},
            },
        },
    },
    {
        "name": "update_wiki",
        "description": "Create or update a wiki entry with new information.",
        "input_schema": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Wiki entry title"},
                "content": {"type": "string", "description": "Markdown content for the entry"},
                "domain": {"type": "string", "description": "Domain/category for routing"},
            },
            "required": ["title", "content"],
        },
    },
    {
        "name": "search_media",
        "description": "Search the media store (videos, podcasts, articles) by topic.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What to search for"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "recall_memory",
        "description": (
            "Recall episodic personal memories by description, date range, or people involved. "
            "Searches the personal_memory fragment and me/ directory."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "description": {"type": "string", "description": "What to recall"},
                "date_from": {"type": "string", "description": "Start date YYYY-MM-DD (optional)"},
                "date_to": {"type": "string", "description": "End date YYYY-MM-DD (optional)"},
                "people": {"type": "array", "items": {"type": "string"},
                           "description": "People involved (optional)"},
            },
            "required": ["description"],
        },
    },
]


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def tool_search_wiki(query: str, limit: int = 5) -> str:
    try:
        from sqlite_rag import hybrid_search
        results = hybrid_search(query, limit=limit)
        if not results:
            return f"No wiki entries found for: {query}"
        parts = [f"Found {len(results)} results for '{query}':\n"]
        for r in results:
            parts.append(
                f"**{r.get('title', 'Untitled')}** [{r.get('route', '')}]\n"
                f"{r.get('content', '')[:300]}...\n"
                f"Path: {r.get('path', '')}\n"
            )
        return "\n".join(parts)
    except Exception as e:
        return f"Search error: {e}"


def tool_query_fragment(domain: str, query: str) -> str:
    try:
        from fragment_manager import Fragment, BRAIN_DIR as FM_BRAIN_DIR
        frag = Fragment(domain, FM_BRAIN_DIR)
        results = frag.query(query)
        if not results:
            return f"No results in {domain} fragment for: {query}"
        parts = [f"Results from {domain} fragment:\n"]
        for r in results:
            parts.append(
                f"**{r.get('title', 'Untitled')}**\n"
                f"{r.get('content', '')[:300]}...\n"
            )
        return "\n".join(parts)
    except Exception as e:
        return f"Fragment query error: {e}"


def tool_ingest_file(path: str) -> str:
    p = Path(path)
    if not p.exists():
        return f"File not found: {path}"
    try:
        # Copy to raw directory for compile.py to pick up
        import shutil
        raw_dir = BRAIN_DIR / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        dest = raw_dir / p.name
        shutil.copy2(str(p), str(dest))
        return f"File queued for processing: {p.name} -> {dest}"
    except Exception as e:
        return f"Ingest error: {e}"


def tool_run_lint(fix: bool = False) -> str:
    try:
        from cross_fragment_lint import CrossFragmentLint
        linter = CrossFragmentLint()
        results = linter.run_all_checks()
        report = linter.generate_report()
        if fix:
            linter.fix_duplicates()
            linter.fix_orphan_links()
            return f"Lint complete + fixed. {results['total_issues']} issues found.\n\n{report[:1000]}"
        return f"Lint complete. {results['total_issues']} issues found.\n{results['counts']}\n\n{report[:1000]}"
    except Exception as e:
        return f"Lint error: {e}"


def tool_update_wiki(title: str, content: str, domain: str = "knowledge") -> str:
    try:
        from wiki_compiler import save_wiki_entry, compile_to_wiki, get_client
        api_key = os.getenv("ANTHROPIC_API_KEY")
        client = anthropic.Anthropic(api_key=api_key) if api_key else None

        entry_data = {
            "title": title,
            "summary": content[:200],
            "content": content,
            "key_concepts": [],
            "cross_links": [],
            "source_citations": ["orchestrator"],
            "tags": [domain],
        }

        if client:
            compiled = compile_to_wiki(client, content, f"orchestrator:{title}")
            compiled["title"] = title
            entry_data = compiled

        path = save_wiki_entry(entry_data, operation="UPDATE")
        return f"Wiki entry saved: {title} -> {path}"
    except Exception as e:
        return f"Wiki update error: {e}"


def tool_search_media(query: str) -> str:
    try:
        media_dir = BRAIN_DIR / "media"
        if not media_dir.exists():
            return "Media directory not found"
        results = []
        query_lower = query.lower()
        for md_file in sorted(media_dir.glob("*.md")):
            try:
                content = md_file.read_text(encoding="utf-8", errors="replace")
                if query_lower in content.lower():
                    results.append(f"- **{md_file.stem}**: {content[:200]}...")
            except Exception:
                pass
        if not results:
            return f"No media found for: {query}"
        return f"Media results for '{query}':\n" + "\n".join(results[:5])
    except Exception as e:
        return f"Media search error: {e}"


def tool_recall_memory(description: str, date_from: str = "", date_to: str = "",
                       people: list = None) -> str:
    try:
        me_dir = BRAIN_DIR / "me"
        results = []
        query_lower = description.lower()
        people = people or []

        search_dirs = [me_dir]
        try:
            from fragment_manager import Fragment, BRAIN_DIR as FM_BRAIN_DIR
            pm_frag = Fragment("personal_memory", FM_BRAIN_DIR)
            frag_results = pm_frag.query(description)
            for r in frag_results:
                results.append(f"[personal_memory] **{r.get('title')}**: {r.get('content', '')[:200]}")
        except Exception:
            pass

        for md_file in sorted(me_dir.rglob("*.md")):
            try:
                content = md_file.read_text(encoding="utf-8", errors="replace")
                match_score = content.lower().count(query_lower)
                people_score = sum(content.lower().count(p.lower()) for p in people)
                if match_score > 0 or people_score > 0:
                    results.append(
                        f"[me/{md_file.parent.name}] **{md_file.stem}**: {content[:200]}..."
                    )
            except Exception:
                pass

        if not results:
            return f"No memories found for: {description}"
        return f"Memory recall for '{description}':\n" + "\n\n".join(results[:5])
    except Exception as e:
        return f"Memory recall error: {e}"


# ---------------------------------------------------------------------------
# Tool dispatcher
# ---------------------------------------------------------------------------

def dispatch_tool(tool_name: str, tool_input: dict) -> str:
    """Execute a tool and return string result."""
    if tool_name == "search_wiki":
        return tool_search_wiki(**tool_input)
    elif tool_name == "query_fragment":
        return tool_query_fragment(**tool_input)
    elif tool_name == "ingest_file":
        return tool_ingest_file(**tool_input)
    elif tool_name == "run_lint":
        return tool_run_lint(**tool_input)
    elif tool_name == "update_wiki":
        return tool_update_wiki(**tool_input)
    elif tool_name == "search_media":
        return tool_search_media(**tool_input)
    elif tool_name == "recall_memory":
        return tool_recall_memory(**tool_input)
    else:
        return f"Unknown tool: {tool_name}"


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are the LLM-Brains orchestrator — a personal AI assistant with access to a comprehensive second-brain knowledge system.

You have tools to:
- Search the wiki knowledge base
- Query domain-specialist fragment agents (20 domains: geography, people, science, technology, history, philosophy, health, creative, business, personal_memory, ai_ml, code, media, relationships, events, concepts, emotions, skills, projects, misc)
- Ingest new files
- Run consistency checks
- Update wiki entries
- Search media library
- Recall personal memories

Guidelines:
- Always search relevant fragments before answering questions about knowledge
- Use recall_memory for personal/episodic questions
- File interesting synthesized answers back into wiki
- Use [[wikilinks]] in your responses for key concepts
- Be concise but thorough; cite your sources from the fragments
- Acknowledge when information is missing or uncertain"""


def run_agent_loop(client: anthropic.Anthropic, user_message: str,
                   conversation_history: list) -> tuple[str, list]:
    """Run the full tool-calling agent loop for one user message."""
    conversation_history.append({"role": "user", "content": user_message})

    while True:
        response = client.messages.create(
            model="claude-opus-4-5",
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=conversation_history,
        )

        # Collect text and tool use blocks
        assistant_content = []
        tool_calls = []

        for block in response.content:
            assistant_content.append(block)
            if block.type == "tool_use":
                tool_calls.append(block)

        conversation_history.append({"role": "assistant", "content": assistant_content})

        if response.stop_reason == "end_turn" or not tool_calls:
            # Extract final text response
            final_text = ""
            for block in response.content:
                if hasattr(block, "text"):
                    final_text += block.text
            return final_text, conversation_history

        # Execute all tool calls
        tool_results = []
        for tool_call in tool_calls:
            print(f"  [Tool] {tool_call.name}({json.dumps(tool_call.input)[:80]}...)")
            result = dispatch_tool(tool_call.name, tool_call.input)
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tool_call.id,
                "content": result,
            })

        conversation_history.append({"role": "user", "content": tool_results})


# ---------------------------------------------------------------------------
# REPL
# ---------------------------------------------------------------------------

def run_repl(client: anthropic.Anthropic):
    """Interactive REPL mode."""
    print("\nLLM-Brains Orchestrator")
    print("="*50)
    print("Type your query, or 'quit' to exit.")
    print("Commands: /lint, /status, /help\n")

    conversation_history = []

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye.")
            break

        if user_input == "/lint":
            result = tool_run_lint()
            print(f"\nBrain: {result}\n")
            continue

        if user_input == "/status":
            try:
                from fragment_manager import FragmentManager
                fm = FragmentManager()
                status = fm.status()
                print("\nFragment Status:")
                for domain, info in status.items():
                    if info["entries"] > 0 or info["wiki_files"] > 0:
                        print(f"  {domain:20s}: {info['entries']} entries")
            except Exception as e:
                print(f"Status error: {e}")
            continue

        if user_input == "/help":
            print("\nCommands:")
            print("  /lint   - Run cross-fragment lint")
            print("  /status - Show fragment status")
            print("  /help   - This help message")
            print("  quit    - Exit\n")
            continue

        try:
            print("Brain: ", end="", flush=True)
            response, conversation_history = run_agent_loop(
                client, user_input, conversation_history
            )
            print(response)
            print()
        except Exception as e:
            print(f"\n[Error] {e}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="LLM-Brains orchestrator")
    parser.add_argument("--query", "-q", type=str, help="Single-shot query")
    parser.add_argument("--model", type=str, default="claude-opus-4-5",
                        help="Claude model to use")
    args = parser.parse_args()

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("ANTHROPIC_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)

    if args.query:
        response, _ = run_agent_loop(client, args.query, [])
        print(response)
    else:
        run_repl(client)


if __name__ == "__main__":
    main()
