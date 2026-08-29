# LLM-Brains — Claude Code Rules

## Branch Policy
- **ALWAYS use `main`**. Never create feature branches. One branch, one truth.
- All commits go directly to `main` and are pushed immediately.
- Never create a pull request unless the user explicitly asks for one.

## Commit Style
- Commit all new/changed files together, not piecemeal.
- Use clear, one-line commit messages describing what was added/changed.

## Project Structure
- All personal data lives under `brain/` — never move or restructure this.
- `brain/raw/` is the drop zone — anything dropped here gets compiled automatically.
- `brain/me/`, `brain/knowledge/`, `brain/work/`, `brain/media/`, `brain/studio/` are outputs — never manually edit these.
- `brain/fragments/` holds the hive-mind fragment agent wikis.

## Development Rules
- Keep it simple. Don't create extra files, helpers, or abstractions unless truly needed.
- Don't rebuild things that already exist — check before creating.
- No duplicate work, no redundant agents doing the same task.
