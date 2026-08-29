#!/usr/bin/env python3
"""
studio_generator.py — NotebookLM-style output generator.

For every wiki entry, auto-generates:
  - Podcast script (Host A + Host B dialogue)
  - Marp slide deck (markdown slides)
  - Mind map (Mermaid mindmap syntax)
  - Research report with executive summary
  - Flashcards (10-15 Q&A pairs as JSON)
  - Quiz (8-10 multiple choice as JSON)
  - Infographic description (detailed SVG spec)
  - Data table (CSV of quantitative facts)

Usage:
    python studio_generator.py --entry brain/knowledge/wiki/my-topic.md
    python studio_generator.py --all              # generate for all wiki entries
    python studio_generator.py --entry file.md --formats podcast,slides,flashcards
"""

import argparse
import csv
import io
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Optional

import anthropic
from dotenv import load_dotenv

load_dotenv()

BRAIN_DIR = Path(os.getenv("BRAIN_DIR", "./brain"))
STUDIO_DIR = BRAIN_DIR / "studio"
WIKI_DIR = BRAIN_DIR / "knowledge" / "wiki"

ALL_FORMATS = ["podcast", "slides", "mindmap", "report", "flashcards", "quiz", "infographic", "datatable"]


def get_client() -> anthropic.Anthropic:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("ANTHROPIC_API_KEY not set", file=sys.stderr)
        sys.exit(1)
    return anthropic.Anthropic(api_key=api_key)


def slugify(title: str) -> str:
    slug = re.sub(r"[^\w\s-]", "", title.lower())
    return re.sub(r"[\s_]+", "-", slug).strip("-")[:60]


def call_claude(client: anthropic.Anthropic, prompt: str,
                max_tokens: int = 2048) -> str:
    """Call Claude and return text response."""
    try:
        response = client.messages.create(
            model="claude-opus-4-5",
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text
    except Exception as e:
        return f"[Generation failed: {e}]"


# ---------------------------------------------------------------------------
# Individual format generators
# ---------------------------------------------------------------------------

def generate_podcast(client: anthropic.Anthropic, title: str, content: str) -> str:
    prompt = f"""Create an engaging podcast script about this topic. Format as a natural conversation.

Topic: {title}
Content:
{content[:4000]}

Write a 5-7 minute podcast script with two hosts:
- **Host A (Alex)**: The curious generalist, asks questions
- **Host B (Blake)**: The domain expert, explains clearly

Format:
```
[INTRO MUSIC]
ALEX: ...
BLAKE: ...
```

Make it engaging, educational, and natural-sounding. Include analogies, examples, and occasional humor.
End with key takeaways and a call to curiosity."""

    return call_claude(client, prompt, max_tokens=3000)


def generate_slides(client: anthropic.Anthropic, title: str, content: str) -> str:
    prompt = f"""Create a Marp slide deck for this topic.

Topic: {title}
Content:
{content[:4000]}

Generate a complete Marp markdown presentation (10-15 slides):
- Use Marp frontmatter: `marp: true`, `theme: default`
- Start with title slide
- Include: overview, key concepts (each as slide), examples, diagrams (described in text), summary
- Use `---` between slides
- Add speaker notes with `<!-- Note: ... -->`
- Include code blocks, tables, or bullet points as appropriate
- End with "Further Reading" and "Key Takeaways" slides

Output ONLY the Marp markdown, starting with the frontmatter."""

    return call_claude(client, prompt, max_tokens=3000)


def generate_mindmap(client: anthropic.Anthropic, title: str, content: str) -> str:
    prompt = f"""Create a Mermaid mindmap for this topic.

Topic: {title}
Content:
{content[:3000]}

Generate a complete Mermaid mindmap with:
- Central node: the main topic
- 5-8 main branches for key concepts/themes
- 2-4 sub-nodes per branch
- Use Mermaid mindmap syntax exactly

Example format:
```
mindmap
  root((Topic Name))
    Branch 1
      Sub 1a
      Sub 1b
    Branch 2
      Sub 2a
```

Output ONLY the Mermaid code block."""

    return call_claude(client, prompt, max_tokens=1500)


def generate_report(client: anthropic.Anthropic, title: str, content: str) -> str:
    prompt = f"""Write a comprehensive research report about this topic.

Topic: {title}
Content:
{content[:5000]}

Structure:
## Executive Summary
(3-5 sentences, key findings, implications)

## Introduction
(Context, why this matters)

## Key Findings
(Main points, organized by theme)

## Analysis
(Deep dive, connections, implications)

## Applications
(How this knowledge can be applied)

## Open Questions
(What we still don't know)

## Conclusion

## References & Further Reading

Use [[wikilinks]] for key concepts. Be thorough and analytical."""

    return call_claude(client, prompt, max_tokens=4000)


def generate_flashcards(client: anthropic.Anthropic, title: str, content: str) -> str:
    prompt = f"""Create 12-15 flashcards for this topic.

Topic: {title}
Content:
{content[:4000]}

Return a JSON array of flashcard objects:
[
  {{
    "id": 1,
    "front": "Question or term",
    "back": "Answer or definition",
    "difficulty": "easy|medium|hard",
    "tags": ["tag1"]
  }}
]

Cover:
- Key definitions
- Important facts
- Causal relationships
- Examples and applications
- Common misconceptions

Return ONLY the JSON array."""

    result = call_claude(client, prompt, max_tokens=2000)
    # Validate JSON
    try:
        match = re.search(r'\[.*\]', result, re.DOTALL)
        if match:
            json.loads(match.group())  # Validate
            return match.group()
    except Exception:
        pass
    return result


def generate_quiz(client: anthropic.Anthropic, title: str, content: str) -> str:
    prompt = f"""Create a 8-10 question multiple choice quiz about this topic.

Topic: {title}
Content:
{content[:4000]}

Return a JSON array:
[
  {{
    "id": 1,
    "question": "Question text?",
    "options": ["A) option", "B) option", "C) option", "D) option"],
    "correct": "A",
    "explanation": "Why A is correct",
    "difficulty": "easy|medium|hard"
  }}
]

Include questions at various difficulty levels. Return ONLY the JSON array."""

    result = call_claude(client, prompt, max_tokens=2000)
    try:
        match = re.search(r'\[.*\]', result, re.DOTALL)
        if match:
            json.loads(match.group())
            return match.group()
    except Exception:
        pass
    return result


def generate_infographic(client: anthropic.Anthropic, title: str, content: str) -> str:
    prompt = f"""Design a detailed infographic specification for this topic.

Topic: {title}
Content:
{content[:3000]}

Create a comprehensive SVG infographic specification including:

1. **Layout**: Overall structure (width, height, sections)
2. **Title Section**: Typography, color scheme, tagline
3. **Main Visualization**: The central graphic (chart, diagram, or illustration)
4. **Data Panels**: Key statistics, numbers, percentages (with exact values)
5. **Icon Set**: 6-8 icons to represent key concepts
6. **Color Palette**: Hex codes for 5 key colors
7. **Typography**: Font choices, sizes, weights
8. **Caption/Footer**: Sources, notes

Then provide the actual SVG code (simplified) for a 800x600 infographic.
Use `<text>`, `<rect>`, `<circle>`, `<path>` elements.
Make it visually rich and information-dense."""

    return call_claude(client, prompt, max_tokens=3000)


def generate_datatable(client: anthropic.Anthropic, title: str, content: str) -> str:
    prompt = f"""Extract all quantitative and structured data from this content into a CSV table.

Topic: {title}
Content:
{content[:4000]}

Create a CSV with:
- First row: headers
- Each row: one data point, fact, or comparison
- Include: dates, numbers, percentages, comparisons, rankings, measurements

If the topic is conceptual with no direct numbers:
- Create a comparison table of key concepts/properties
- Rate attributes on a 1-10 scale
- List relationships (A → B → C format)

Return ONLY valid CSV content (no markdown code blocks)."""

    result = call_claude(client, prompt, max_tokens=1500)
    # Clean up any markdown code blocks
    result = re.sub(r"```csv\n?", "", result)
    result = re.sub(r"```\n?", "", result)
    return result.strip()


# ---------------------------------------------------------------------------
# Output writer
# ---------------------------------------------------------------------------

def write_output(output_dir: Path, format_name: str, content: str, slug: str):
    """Write generated content to appropriate file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    ext_map = {
        "podcast": ".md",
        "slides": ".md",
        "mindmap": ".md",
        "report": ".md",
        "flashcards": ".json",
        "quiz": ".json",
        "infographic": ".svg",
        "datatable": ".csv",
    }
    ext = ext_map.get(format_name, ".txt")

    # For JSON formats, try to pretty-print
    if ext == ".json":
        try:
            match = re.search(r'\[.*\]', content, re.DOTALL)
            if match:
                data = json.loads(match.group())
                content = json.dumps(data, indent=2)
        except Exception:
            pass

    # For SVG, wrap in proper SVG if not already
    if ext == ".svg" and not content.strip().startswith("<svg"):
        # Extract SVG if embedded in text
        svg_match = re.search(r'<svg.*?</svg>', content, re.DOTALL | re.IGNORECASE)
        if svg_match:
            content = svg_match.group()
        else:
            # Wrap content as SVG with embedded text
            content = (
                '<svg xmlns="http://www.w3.org/2000/svg" width="800" height="600">'
                f'<text x="10" y="20" font-size="12">{slug}</text>'
                f'<!-- Infographic spec:\n{content[:2000]}\n-->'
                '</svg>'
            )

    # Add metadata header for markdown files
    if ext == ".md":
        header = f"---\ntitle: {format_name.title()} — {slug}\ngenerated: {datetime.now().isoformat()}\n---\n\n"
        content = header + content

    out_path = output_dir / f"{format_name}{ext}"
    out_path.write_text(content, encoding="utf-8")
    return out_path


# ---------------------------------------------------------------------------
# Main studio pipeline
# ---------------------------------------------------------------------------

GENERATOR_MAP = {
    "podcast": generate_podcast,
    "slides": generate_slides,
    "mindmap": generate_mindmap,
    "report": generate_report,
    "flashcards": generate_flashcards,
    "quiz": generate_quiz,
    "infographic": generate_infographic,
    "datatable": generate_datatable,
}


def generate_for_entry(entry_path: Path, client: anthropic.Anthropic,
                        formats: list[str] = None) -> dict:
    """Generate all studio outputs for a single wiki entry."""
    if formats is None:
        formats = ALL_FORMATS

    # Read entry
    content = entry_path.read_text(encoding="utf-8", errors="replace")

    # Extract title
    title = entry_path.stem.replace("-", " ").title()
    title_match = re.search(r'^title:\s*(.+)$', content, re.MULTILINE)
    if title_match:
        title = title_match.group(1).strip()

    # Strip frontmatter
    body = re.sub(r'^---.*?---\n', '', content, flags=re.DOTALL).strip()

    slug = slugify(title)
    output_dir = STUDIO_DIR / slug
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nGenerating studio outputs for: {title}")
    print(f"  Formats: {', '.join(formats)}")

    results = {}

    # Run in parallel with ThreadPoolExecutor
    def generate_one(fmt: str) -> tuple[str, str]:
        gen_fn = GENERATOR_MAP.get(fmt)
        if gen_fn is None:
            return fmt, f"[Unknown format: {fmt}]"
        print(f"  Generating {fmt}...")
        content_out = gen_fn(client, title, body)
        out_path = write_output(output_dir, fmt, content_out, slug)
        return fmt, str(out_path)

    max_workers = min(len(formats), 4)  # Limit parallelism to avoid rate limits
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(generate_one, fmt): fmt for fmt in formats}
        for future in as_completed(futures):
            try:
                fmt, out_path = future.result()
                results[fmt] = out_path
                print(f"  ✓ {fmt}: {Path(out_path).name}")
            except Exception as e:
                fmt = futures[future]
                print(f"  ✗ {fmt}: {e}", file=sys.stderr)
                results[fmt] = f"ERROR: {e}"

    # Write index file for this entry's studio outputs
    index_content = f"""---
title: Studio — {title}
source: {entry_path}
generated: {datetime.now().isoformat()}
---

# Studio: {title}

Generated outputs:

| Format | File |
|--------|------|
"""
    for fmt, path in results.items():
        if not path.startswith("ERROR"):
            fname = Path(path).name
            index_content += f"| {fmt} | [{fname}]({fname}) |\n"

    (output_dir / "index.md").write_text(index_content, encoding="utf-8")
    print(f"\n  -> Studio dir: {output_dir.relative_to(BRAIN_DIR)}")

    return {"title": title, "slug": slug, "outputs": results}


def generate_for_all(client: anthropic.Anthropic,
                      formats: list[str] = None) -> list[dict]:
    """Generate studio outputs for all wiki entries."""
    if not WIKI_DIR.exists():
        print(f"Wiki directory not found: {WIKI_DIR}", file=sys.stderr)
        return []

    wiki_files = [f for f in WIKI_DIR.glob("*.md")
                  if not f.name.startswith("_") and f.name != ".gitkeep"]

    if not wiki_files:
        print("No wiki entries found")
        return []

    print(f"Generating studio outputs for {len(wiki_files)} wiki entries...")
    results = []

    for wiki_file in sorted(wiki_files):
        try:
            result = generate_for_entry(wiki_file, client, formats=formats)
            results.append(result)
        except Exception as e:
            print(f"Error processing {wiki_file.name}: {e}", file=sys.stderr)

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Studio output generator")
    parser.add_argument("--entry", type=str, help="Wiki entry file to process")
    parser.add_argument("--all", action="store_true",
                        help="Generate for all wiki entries")
    parser.add_argument("--formats", type=str,
                        default=",".join(ALL_FORMATS),
                        help=f"Comma-separated formats: {','.join(ALL_FORMATS)}")
    parser.add_argument("--list-formats", action="store_true",
                        help="List available formats")
    args = parser.parse_args()

    if args.list_formats:
        print("Available formats:")
        for fmt in ALL_FORMATS:
            print(f"  - {fmt}")
        return

    formats = [f.strip() for f in args.formats.split(",") if f.strip() in ALL_FORMATS]
    if not formats:
        print(f"No valid formats specified. Available: {', '.join(ALL_FORMATS)}")
        sys.exit(1)

    client = get_client()

    if args.entry:
        entry_path = Path(args.entry)
        if not entry_path.exists():
            print(f"File not found: {args.entry}", file=sys.stderr)
            sys.exit(1)
        result = generate_for_entry(entry_path, client, formats=formats)
        print(f"\nGenerated {len(result['outputs'])} outputs for '{result['title']}'")

    elif args.all:
        results = generate_for_all(client, formats=formats)
        print(f"\nGenerated studio outputs for {len(results)} entries")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
