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
                "query": {"type": "string", "description": "The search query"},
                "limit": {"type": "integer", "description": "Max results (default 5)", "default": 5},
            },
            "required": ["query"],
        },
    },
    {
        "name": "query_fragment",
        "description": (
            "Query a specific domain fragment agent. "
            "Domains: geography, people, science, technology, history, philosophy, "
            "health, creative, business, personal_memory, ai_ml, code, media, "
            "relationships, events, concepts, emotions, skills, projects, misc."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
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
