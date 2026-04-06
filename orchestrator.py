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
