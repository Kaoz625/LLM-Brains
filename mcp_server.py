#!/usr/bin/env python3
"""
LLM-Brains MCP Server — Exposes brain search to all Claude agents.

Provides two tools:
  - brain_search: Search the indexed brain knowledge base
  - brain_stats:  Show database statistics

Usage (stdio MCP protocol):
    python3 ~/LLM-Brains/mcp_server.py

Add to ~/.claude/settings.json mcpServers to use with Claude Code.
"""

import json
import os
import sqlite3
import sys
from pathlib import Path

BRAIN_DIR = Path(os.getenv("BRAIN_DIR", str(Path.home() / "LLM-Brains" / "brain")))
DB_PATH = BRAIN_DIR / "brain.db"


def search_brain(query: str, limit: int = 10) -> list[dict]:
    """Search the brain using FTS5 full-text search."""
    if not DB_PATH.exists():
        return [{"error": f"Brain database not built yet. Run: cd ~/LLM-Brains && python3 main.py ingest ~/path/to/vault"}]

    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row

        # Try FTS5 search first
        try:
            rows = conn.execute(
                "SELECT title, content, path, rank FROM brain_fts WHERE brain_fts MATCH ? ORDER BY rank LIMIT ?",
                (query, limit)
            ).fetchall()
        except sqlite3.OperationalError:
            # Fallback: LIKE search if FTS5 table not available
            rows = conn.execute(
                "SELECT title, content, path FROM brain_chunks WHERE content LIKE ? LIMIT ?",
                (f"%{query}%", limit)
            ).fetchall()

        results = []
        for row in rows:
            d = dict(row)
            # Truncate long content for readability
            if "content" in d and len(d["content"]) > 500:
                d["content"] = d["content"][:500] + "..."
            results.append(d)

        conn.close()
        return results if results else [{"message": f"No results found for: {query}"}]

    except Exception as e:
        return [{"error": str(e)}]


def get_stats() -> dict:
    """Get brain database statistics."""
    if not DB_PATH.exists():
        return {"status": "not_built", "db_path": str(DB_PATH),
                "setup": "Run: cd ~/LLM-Brains && python3 main.py ingest ~/path/to/vault"}

    try:
        conn = sqlite3.connect(DB_PATH)
        stats = {"db_path": str(DB_PATH), "db_size_mb": round(DB_PATH.stat().st_size / 1e6, 2)}

        for table in ["brain_chunks", "brain_fts", "documents"]:
            try:
                count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                stats[f"{table}_count"] = count
            except sqlite3.OperationalError:
                pass

        conn.close()
        return stats
    except Exception as e:
        return {"error": str(e)}


# ── MCP JSON-RPC protocol (stdio) ────────────────────────────────────────────

TOOLS = [
    {
        "name": "brain_search",
        "description": "Search the LLM-Brains knowledge base — your indexed Obsidian vault, compiled articles, episodic memories, and dream cycle lessons.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "limit": {"type": "integer", "description": "Max results (default 10)", "default": 10},
            },
            "required": ["query"],
        },
    },
    {
        "name": "brain_stats",
        "description": "Get statistics about the LLM-Brains knowledge base (size, document count, etc.)",
        "inputSchema": {"type": "object", "properties": {}},
    },
]


def handle_request(req: dict) -> dict:
    method = req.get("method", "")
    req_id = req.get("id")

    if method == "initialize":
        return {
            "jsonrpc": "2.0", "id": req_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "llm-brains", "version": "1.0.0"},
            },
        }

    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": req_id, "result": {"tools": TOOLS}}

    if method == "tools/call":
        name = req.get("params", {}).get("name")
        args = req.get("params", {}).get("arguments", {})

        if name == "brain_search":
            results = search_brain(args.get("query", ""), args.get("limit", 10))
            text = json.dumps(results, indent=2)
        elif name == "brain_stats":
            text = json.dumps(get_stats(), indent=2)
        else:
            text = json.dumps({"error": f"Unknown tool: {name}"})

        return {
            "jsonrpc": "2.0", "id": req_id,
            "result": {"content": [{"type": "text", "text": text}]},
        }

    return {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32601, "message": f"Method not found: {method}"}}


def main():
    while True:
        try:
            line = sys.stdin.readline()
            if not line:
                break
            req = json.loads(line.strip())
            resp = handle_request(req)
            sys.stdout.write(json.dumps(resp) + "\n")
            sys.stdout.flush()
        except json.JSONDecodeError:
            pass
        except KeyboardInterrupt:
            break
        except Exception as e:
            sys.stderr.write(f"Error: {e}\n")


if __name__ == "__main__":
    main()
