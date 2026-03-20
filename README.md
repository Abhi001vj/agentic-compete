# AgenticCompete 🏆

**An Agentic Framework for Kaggle Competitions — Powered by Claude + Google Colab MCP**

> Evolving AutoCompete (Thakur & Krohn-Grimberghe, ICML 2015) from a static pipeline into a modern
> LLM-driven agent that progressively tackles Kaggle competitions using Google Colab as remote compute.

---

## Philosophy

The original AutoCompete followed a fixed pipeline: split → identify types → transform → select model → tune.
AgenticCompete replaces that rigidity with **LLM reasoning at every decision point**, mirroring how a
top Kaggle competitor actually works:

1. **Understand** the competition (metric, data, domain)
2. **Explore** deeply (EDA, distributions, correlations, missing patterns)
3. **Plan** a progressive strategy (baseline → medium → complex)
4. **Execute** with discipline (run baseline first, extract maximum before escalating)
5. **Reflect** on results and adapt the plan
6. **Iterate** until diminishing returns, then escalate model complexity

The agent uses **Google Colab MCP** (released March 17, 2026) for GPU compute, meaning your
local machine only runs the orchestrator while all heavy lifting happens in Colab.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    AgenticCompete Orchestrator                   │
│                      (LangGraph + Claude)                       │
├─────────────┬──────────────┬──────────────┬────────────────────┤
│  EDA Agent  │ Feature Eng  │ Model Agent  │ Ensemble Agent     │
│             │    Agent     │              │                    │
└──────┬──────┴──────┬───────┴──────┬───────┴────────┬───────────┘
       │             │              │                │
       ▼             ▼              ▼                ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Colab Execution Layer                        │
│              (Google Colab MCP — GPU/TPU Runtime)                │
│                                                                  │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────────┐  │
│  │ EDA      │ │ Feature  │ │ Training │ │ Inference &      │  │
│  │ Notebook │ │ Notebook │ │ Notebook │ │ Submission NB    │  │
│  └──────────┘ └──────────┘ └──────────┘ └──────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
       │             │              │                │
       ▼             ▼              ▼                ▼
┌─────────────────────────────────────────────────────────────────┐
│                        Kaggle API Layer                          │
│         (Download data, Submit predictions, Check LB)           │
└─────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
agentic-compete/
├── README.md                    # This file
├── pyproject.toml               # Project dependencies
├── config/
│   ├── settings.yaml            # Global settings
│   ├── mcp_config.json          # Colab MCP server config
│   └── model_registry.yaml      # Model tiers & hyperparameter spaces
├── core/
│   ├── __init__.py
│   ├── orchestrator.py          # Main LangGraph orchestrator (state machine)
│   ├── state.py                 # AgenticCompete state definition
│   ├── decision_engine.py       # LLM-powered decision making
│   └── memory.py                # Competition memory / knowledge base
├── agents/
│   ├── __init__.py
│   ├── competition_analyzer.py  # Understand competition rules & metric
│   ├── eda_agent.py             # Exploratory data analysis
│   ├── feature_agent.py         # Feature engineering
│   ├── model_agent.py           # Model selection & training
│   ├── tuning_agent.py          # Hyperparameter optimization
│   ├── ensemble_agent.py        # Ensembling & stacking
│   └── submission_agent.py      # Generate & submit predictions
├── colab/
│   ├── __init__.py
│   ├── session_manager.py       # Manage Colab MCP connection
│   ├── notebook_builder.py      # Programmatically build notebooks
│   ├── code_templates.py        # Reusable code snippets for Colab cells
│   └── output_parser.py         # Parse execution results from Colab
├── kaggle/
│   ├── __init__.py
│   ├── competition_client.py    # Kaggle API wrapper
│   └── submission_tracker.py    # Track scores over time
├── strategies/
│   ├── __init__.py
│   ├── progressive.py           # Progressive complexity strategy
│   ├── baseline_strategy.py     # Baseline model strategy
│   └── escalation_rules.py      # When to escalate model complexity
├── prompts/
│   ├── system_prompt.md         # Master system prompt for the agent
│   ├── eda_prompt.md            # EDA analysis prompt
│   ├── feature_eng_prompt.md    # Feature engineering prompt
│   ├── model_selection_prompt.md # Model selection prompt
│   └── reflection_prompt.md     # Post-training reflection prompt
├── scripts/
│   ├── run_competition.py       # CLI entry point
│   ├── setup_colab_mcp.sh       # Setup script for Colab MCP
│   └── init_competition.py      # Initialize a new competition workspace
└── docs/
    ├── architecture.md          # Detailed architecture doc
    ├── autocompete_analysis.md  # Analysis of original AutoCompete paper
    └── colab_mcp_guide.md       # Guide to Colab MCP capabilities
```

---

## Core Workflow (State Machine)

The orchestrator is a LangGraph state machine with these states:

```
INIT → ANALYZE_COMPETITION → DOWNLOAD_DATA → EDA
  → PLAN_STRATEGY → BASELINE_MODELS → EVALUATE_BASELINE
  → [DECISION: good enough?]
      ├─ YES → EXTRACT_MAX_BASELINE → FEATURE_ENGINEERING → RE_EVALUATE
      └─ NO  → MEDIUM_MODELS → EVALUATE_MEDIUM
                  → [DECISION: good enough?]
                      ├─ YES → EXTRACT_MAX_MEDIUM → FEATURE_ENGINEERING → RE_EVALUATE  
                      └─ NO  → COMPLEX_MODELS → EVALUATE_COMPLEX
                                  → EXTRACT_MAX_COMPLEX
  → ENSEMBLE → FINAL_SUBMISSION → DONE
```

---

## Colab MCP Integration

The official Google Colab MCP server (released March 17, 2026) provides:

- **Add code/markdown cells** to live Colab notebooks
- **Execute cells** and retrieve outputs/tracebacks
- **Organize/rearrange** cells programmatically
- **Full notebook lifecycle** control from any MCP-compatible agent

### Setup

```bash
# Install uv (if not already)
pip install uv

# MCP config (for Claude Code / any MCP client)
# Add to ~/.claude/mcp.json or equivalent:
{
  "mcpServers": {
    "colab-mcp": {
      "command": "uvx",
      "args": ["git+https://github.com/googlecolab/colab-mcp"],
      "timeout": 30000
    }
  }
}
```

The agent connects to whichever Colab notebook you have open in the browser.

---

## Model Tiers

| Tier | Models | When to Use |
|------|--------|-------------|
| **Baseline** | LogisticRegression, Ridge, RandomForest, XGBoost (default params) | Always start here |
| **Medium** | Tuned XGBoost, LightGBM, CatBoost, MLP | Baseline < top 50% |
| **Complex** | Stacking, Neural Nets, TabNet, AutoGluon, Custom architectures | Medium plateaus |
| **Ensemble** | Weighted avg, Stacking, Blending | Final push |

---

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/youruser/agentic-compete.git
cd agentic-compete
pip install -e .

# 2. Configure Kaggle API
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_api_key

# 3. Configure Colab MCP (see setup above)

# 4. Run on a competition
python scripts/run_competition.py \
  --competition "titanic" \
  --metric "accuracy" \
  --max-hours 4
```
