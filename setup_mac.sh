#!/bin/bash
# LLM-Brains Mac Setup
# Run once: bash setup_mac.sh

set -e

echo "=== LLM-Brains Setup ==="

# 1. Check Python
if ! command -v python3 &>/dev/null; then
  echo "Installing Python via Homebrew..."
  brew install python
fi
echo "✓ Python $(python3 --version)"

# 2. Check pip alias
if ! command -v pip &>/dev/null; then
  echo 'alias pip=pip3' >> ~/.zshrc
  echo 'alias python=python3' >> ~/.zshrc
fi

# 3. Install dependencies
echo "Installing Python packages..."
pip3 install --quiet \
  anthropic openai python-frontmatter pymupdf faster-whisper \
  yt-dlp youtube-transcript-api sqlite-vec numpy requests \
  Pillow watchdog python-dotenv pydantic scipy feedparser --no-deps
pip3 install --quiet feedparser --no-deps

# 4. Check ffmpeg (needed for video/audio)
if ! command -v ffmpeg &>/dev/null; then
  echo "Installing ffmpeg..."
  brew install ffmpeg
fi
echo "✓ ffmpeg"

# 5. Create brain folder in Syncthing
BRAIN_DIR="/Users/$USER/Syncthing/marksbrain"
mkdir -p \
  "$BRAIN_DIR/raw/processed" \
  "$BRAIN_DIR/me" \
  "$BRAIN_DIR/work" \
  "$BRAIN_DIR/knowledge/wiki" \
  "$BRAIN_DIR/media" \
  "$BRAIN_DIR/studio" \
  "$BRAIN_DIR/fragments"
echo "✓ Brain folder ready at $BRAIN_DIR"

# 6. Create .env if it doesn't exist
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
if [ ! -f "$SCRIPT_DIR/.env" ]; then
  cp "$SCRIPT_DIR/.env.example" "$SCRIPT_DIR/.env"
  echo ""
  echo "⚠️  Add your Anthropic API key to .env:"
  echo "    $SCRIPT_DIR/.env"
  echo "    → set ANTHROPIC_API_KEY=sk-ant-..."
else
  echo "✓ .env already exists"
fi

echo ""
echo "=== Done! ==="
echo ""
echo "To start adding memories:"
echo "  1. Drop any file into ~/Syncthing/marksbrain/raw/"
echo "  2. Run: python3 compile.py"
echo ""
echo "To watch continuously (auto-processes as you drop files):"
echo "  python3 compile.py --watch"
