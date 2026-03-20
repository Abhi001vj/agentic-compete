#!/bin/bash
# Setup script for AgenticCompete + Colab MCP
# Run this once to set up your environment.

set -e

echo "╔═══════════════════════════════════════════╗"
echo "║   AgenticCompete Setup                    ║"
echo "╚═══════════════════════════════════════════╝"

# 1. Check prerequisites
echo ""
echo "=== Checking prerequisites ==="

# Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found. Install from https://python.org"
    exit 1
fi
echo "✅ Python $(python3 --version 2>&1 | awk '{print $2}')"

# Git
if ! command -v git &> /dev/null; then
    echo "❌ Git not found. Install from https://git-scm.com"
    exit 1
fi
echo "✅ Git $(git --version | awk '{print $3}')"

# uv (Python package manager)
if ! command -v uv &> /dev/null; then
    echo "📦 Installing uv..."
    pip install uv
fi
echo "✅ uv installed"

# 2. Install AgenticCompete
echo ""
echo "=== Installing AgenticCompete ==="
pip install -e ".[dev]"
echo "✅ AgenticCompete installed"

# 3. Configure Kaggle API
echo ""
echo "=== Kaggle API ==="
if [ -f ~/.kaggle/kaggle.json ]; then
    echo "✅ Kaggle credentials found at ~/.kaggle/kaggle.json"
elif [ -n "$KAGGLE_USERNAME" ] && [ -n "$KAGGLE_KEY" ]; then
    echo "✅ Kaggle credentials found in environment"
else
    echo "⚠️  Kaggle credentials not found."
    echo "   Option 1: Place kaggle.json in ~/.kaggle/"
    echo "   Option 2: Set KAGGLE_USERNAME and KAGGLE_KEY env vars"
    echo "   Get your API key from: https://www.kaggle.com/settings"
fi

# 4. Configure Colab MCP
echo ""
echo "=== Colab MCP Server ==="
echo "The Colab MCP server connects your agent to Google Colab."
echo ""

# Determine MCP config location
MCP_CONFIG=""
if [ -d ~/.claude ]; then
    MCP_CONFIG=~/.claude/mcp.json
    echo "Detected Claude Code environment"
elif [ -d ~/.cursor ]; then
    MCP_CONFIG=~/.cursor/mcp.json
    echo "Detected Cursor environment"
else
    MCP_CONFIG=~/.config/mcp/mcp.json
    echo "Using default MCP config location"
fi

# Create MCP config if it doesn't exist
if [ ! -f "$MCP_CONFIG" ]; then
    mkdir -p "$(dirname "$MCP_CONFIG")"
    cat > "$MCP_CONFIG" << 'EOF'
{
  "mcpServers": {
    "colab-mcp": {
      "command": "uvx",
      "args": ["git+https://github.com/googlecolab/colab-mcp"],
      "timeout": 30000
    }
  }
}
EOF
    echo "✅ Created MCP config at $MCP_CONFIG"
else
    echo "✅ MCP config exists at $MCP_CONFIG"
    echo "   Make sure colab-mcp is configured. See config/mcp_config.json for reference."
fi

# 5. Verify
echo ""
echo "=== Verification ==="
python3 -c "
import anthropic
print('✅ Anthropic SDK available')
" 2>/dev/null || echo "⚠️  Anthropic SDK not found (pip install anthropic)"

python3 -c "
import langgraph
print('✅ LangGraph available')
" 2>/dev/null || echo "⚠️  LangGraph not found (pip install langgraph)"

echo ""
echo "═══════════════════════════════════════════"
echo "Setup complete! Next steps:"
echo ""
echo "  1. Open a Google Colab notebook in your browser"
echo "  2. Run: python scripts/run_competition.py --competition titanic --metric accuracy"
echo ""
echo "  The agent will automatically connect to your open Colab notebook"
echo "  and start building, executing, and iterating on models."
echo "═══════════════════════════════════════════"
