# AgenticCompete — Architecture & Design Document

## 1. From AutoCompete to AgenticCompete

### 1.1 AutoCompete (2015) — What It Did

AutoCompete (Thakur & Krohn-Grimberghe, ICML 2015 AutoML Workshop) was a **static pipeline**
for automated machine learning. Its core flow was:

```
Dataset → Splitter → Type Identifier → {Numerical, Categorical, Text} Transformers
  → Stacker → {PCA, LDA, QDA, SVD, Original} Decompositions
  → Feature Selector → ML Model Selector → Hyperparameter Selector → Output
```

**Key components:**
- **ML Model Selector**: Tried a fixed set of models (RF, GBM, LR, Ridge, NB, SVM, KNN)
- **Hyperparameter Selector**: Grid search or random search with pre-defined spaces
- **Evaluator**: Used validation set (train/val split) to pick the best pipeline
- **Stacker**: Combined different feature types into unified representation

**Strengths:** Fast baseline results, codified expert knowledge from 100+ competitions.
**Limitations:** Fixed pipeline, no adaptive strategy, no feature engineering, no ensemble
optimization, no error recovery, no reasoning about *why* a model works or fails.

### 1.2 AgenticCompete (2026) — What We're Building

AgenticCompete replaces AutoCompete's static pipeline with an **LLM-driven agent** that:

| AutoCompete (2015) | AgenticCompete (2026) |
|--------------------|-----------------------|
| Fixed pipeline order | Adaptive state machine with conditional branching |
| Pre-defined model set | LLM selects models based on data profile |
| Grid/random search | Optuna with LLM-generated search spaces |
| No feature engineering | LLM generates feature ideas from EDA insights |
| No error recovery | Auto-retry with LLM-based code fixing |
| Local compute only | Google Colab MCP for GPU/TPU access |
| No cross-competition learning | Memory system stores learnings |
| Single best model | Progressive tier system + ensemble |

---

## 2. Google Colab MCP — The Compute Bridge

### 2.1 What Is Colab MCP?

Released March 17, 2026 by Google, the **official Colab MCP server** (`github.com/googlecolab/colab-mcp`)
gives any MCP-compatible agent programmatic control over a live Colab notebook.

### 2.2 Capabilities

The MCP server exposes these tools:

| Tool | Description |
|------|-------------|
| `add_code_cell` | Inject a Python code cell at a given index |
| `add_text_cell` | Inject a markdown cell at a given index |
| `execute_cell` | Run a cell and get stdout/stderr/outputs |
| `move_cell` | Rearrange cells |
| `delete_cell` | Remove a cell |

The agent can:
- **Create notebooks from scratch** (cell by cell)
- **Write and execute Python** in real-time
- **Read outputs** including text, tracebacks, and (indirectly) images
- **Iterate on failures** — read error, fix code, re-execute
- **Use GPU/TPU** — whatever runtime is active in the browser

### 2.3 Setup

```json
// ~/.claude/mcp.json or equivalent
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

**Prerequisite**: A Colab notebook must be open in your browser. The MCP server
connects to the active notebook tab.

### 2.4 How AgenticCompete Uses It

```
Agent (local)                    Colab (cloud)
─────────────                    ─────────────
1. Generate EDA code      →     add_code_cell + execute
2. Parse output           ←     JSON/text output
3. Reason about results          (LLM decision)
4. Generate training code →     add_code_cell + execute
5. Parse metrics          ←     JSON output
6. Decide next step              (LLM decision)
7. ...repeat...           →     ...
```

The key insight: the agent never runs heavy compute locally. It only **generates code**
and **parses results**. All GPU-intensive work happens in Colab.

---

## 3. System Architecture

### 3.1 LangGraph State Machine

The orchestrator is a **LangGraph** compiled graph. Each node is a function that:
1. Reads from `CompetitionState`
2. Calls the relevant agent
3. Returns state updates

```
                        ┌──────────┐
                        │ ANALYZE  │
                        └────┬─────┘
                             ▼
                        ┌──────────┐
                        │ DOWNLOAD │
                        └────┬─────┘
                             ▼
                        ┌──────────┐
                        │   EDA    │
                        └────┬─────┘
                             ▼
                        ┌──────────┐
                        │   PLAN   │
                        └────┬─────┘
                             ▼
                        ┌──────────┐
                   ┌───▶│ BASELINE │
                   │    └────┬─────┘
                   │         ▼
                   │    ┌──────────┐
                   │    │ EVALUATE │◄────────────────────────────┐
                   │    └────┬─────┘                             │
                   │         │                                    │
                   │    ┌────┴──────────────────┐                │
                   │    ▼              ▼         ▼                │
                   │ extract_max   escalate   submit              │
                   │    │              │                          │
                   │    ▼              ▼                          │
                   │ FEATURE_ENG   MEDIUM ──▶ EVALUATE ──┐       │
                   │    │                         │       │       │
                   │    └─────────────────────────┘   escalate    │
                   │                                    │         │
                   │                              COMPLEX ──▶ EVALUATE
                   │                                    │
                   │                                    ▼
                   │                               ENSEMBLE
                   │                                    │
                   │                                    ▼
                   └──────────────────────────────── SUBMIT
```

### 3.2 State Object

`CompetitionState` is the single source of truth. Key fields:

```python
@dataclass
class CompetitionState:
    # Identity
    competition_slug: str
    evaluation_metric: str
    metric_direction: str  # "maximize" / "minimize"

    # Current position
    phase: Phase           # Which node we're at
    current_tier: ModelTier  # BASELINE / MEDIUM / COMPLEX / ENSEMBLE
    iteration: int

    # Data understanding
    data_profile: DataProfile   # Populated by EDA agent

    # Strategy
    strategy_plan: str
    baseline_models_planned: list[str]
    medium_models_planned: list[str]
    complex_models_planned: list[str]

    # Results
    model_results: list[ModelResult]  # All model runs
    best_score: float
    best_model_id: str
    feature_sets: list[FeatureSet]

    # Reasoning trace
    reasoning_log: list[dict]  # Full audit trail
```

### 3.3 Agent Hierarchy

```
Orchestrator (LangGraph)
├── CompetitionAnalyzer    — Understands the competition
├── EDAAgent               — Runs EDA in Colab, builds DataProfile
├── DecisionEngine         — LLM-powered strategic decisions
├── ModelAgent             — Trains models at each tier
├── FeatureAgent           — Engineers features from EDA insights
├── EnsembleAgent          — Creates optimal ensembles
└── SubmissionAgent        — Generates and submits predictions
```

Each agent:
- **Generates Python code** (possibly via LLM)
- **Sends it to Colab** via `ColabSession`
- **Parses results** via `OutputParser`
- **Returns structured data** to update state

---

## 4. Progressive Strategy System

### 4.1 The Kaggle Competitor's Workflow

Top Kagglers follow this pattern:

1. **Quick EDA** → Understand the data landscape
2. **Baseline** → Establish a floor (LR, default XGB)
3. **Is baseline competitive?**
   - YES → Extract max via feature eng + tuning
   - NO → Try medium models
4. **Medium models** (tuned GBMs)
5. **Still not competitive?** → Complex models (Optuna, stacking)
6. **Ensemble** top diverse models
7. **Submit** best ensemble

### 4.2 Escalation Rules

```python
# The agent escalates when:
1. Patience exhausted (3 consecutive models without improvement)
2. Feature engineering attempted (up to 3 rounds)
3. Current tier's models all trained

# The agent DOESN'T escalate when:
1. Still improving at current tier
2. Feature engineering not yet tried
3. Time budget allows more exploration
```

### 4.3 Model Tiers

**Tier 1 — Baseline** (1-5 min per model)
- LogisticRegression / RidgeClassifier
- RandomForest (n=200, depth=10)
- XGBoost (default: n=200, lr=0.1, depth=6)

**Tier 2 — Medium** (5-30 min per model)
- LightGBM (n=500, lr=0.05, depth=7, tuned regularization)
- CatBoost (iterations=500, depth=7)
- XGBoost (tuned subsample, colsample, reg)

**Tier 3 — Complex** (30+ min)
- Optuna-tuned XGBoost/LightGBM (100 trials)
- Stacking ensemble (top 3-5 models as base, LR meta-learner)

**Tier 4 — Ensemble**
- Optuna-weighted averaging of top diverse models
- Blending with held-out fold predictions

---

## 5. Key Design Decisions

### 5.1 Why LangGraph?

- **State machine semantics**: Natural fit for the progressive workflow
- **Conditional edges**: Decision points route to different paths
- **Built-in persistence**: Can checkpoint and resume
- **Composable**: Each node is independently testable

### 5.2 Why Claude as the Brain?

- **Code generation**: Generates Python for EDA, feature eng, model code
- **Strategic reasoning**: Decides when to escalate, what features to engineer
- **Error fixing**: Reads tracebacks and generates fixes
- **Search space design**: Creates Optuna search spaces tailored to the data

### 5.3 Why Colab MCP (not local execution)?

- **GPU access**: Free T4 GPU, paid A100
- **Pre-installed libraries**: sklearn, XGBoost, etc.
- **Isolated environment**: No risk to local machine
- **Visual notebook**: Human can inspect the notebook anytime
- **Reproducible**: Notebook serves as documentation

### 5.4 Why Progressive (not try-everything)?

- **Time-efficient**: Don't waste time on complex models if baseline is good
- **Resource-efficient**: GBM tuning is cheap; neural net training is expensive
- **Kaggle-proven**: This is how top competitors actually work
- **Interpretable**: Clear reasoning trail at each decision

---

## 6. Implementation Phases

### Phase 1 — MVP (Week 1-2)
- [ ] Core state machine (orchestrator + state)
- [ ] Colab session manager (connect to MCP, execute cells, parse output)
- [ ] EDA agent (generate EDA code, parse into DataProfile)
- [ ] Baseline model agent (train LR, RF, XGB with default params)
- [ ] Simple decision engine (escalate after N failures)
- [ ] CLI entry point

### Phase 2 — Intelligence (Week 3-4)
- [ ] LLM-powered strategy planning
- [ ] Feature engineering agent
- [ ] Optuna tuning integration
- [ ] Medium model tier
- [ ] Auto-retry with code fixing

### Phase 3 — Polish (Week 5-6)
- [ ] Ensemble agent with weight optimization
- [ ] Kaggle submission integration
- [ ] Competition memory (cross-competition learning)
- [ ] Comprehensive logging and reporting
- [ ] Tests for all core components

### Phase 4 — Advanced (Week 7+)
- [ ] Multi-notebook support (EDA in one, training in another)
- [ ] Parallel model training
- [ ] Custom model injection (user-provided model code)
- [ ] Competition-specific strategies (time series, NLP, vision)
- [ ] Dashboard UI for monitoring runs

---

## 7. File-by-File Reference

| File | Lines | Purpose |
|------|-------|---------|
| `core/state.py` | ~180 | State definition, ModelResult, DataProfile |
| `core/orchestrator.py` | ~280 | LangGraph state machine, all nodes |
| `core/decision_engine.py` | ~200 | LLM-powered strategic decisions |
| `core/memory.py` | ~120 | Cross-competition learning |
| `agents/eda_agent.py` | ~160 | EDA execution and DataProfile building |
| `agents/model_agent.py` | ~280 | Model training across all tiers |
| `agents/feature_agent.py` | ~110 | LLM-driven feature engineering |
| `agents/ensemble_agent.py` | ~100 | Ensemble creation |
| `agents/submission_agent.py` | ~100 | Prediction + Kaggle submission |
| `agents/competition_analyzer.py` | ~80 | Competition metadata extraction |
| `colab/session_manager.py` | ~230 | Colab MCP connection and cell execution |
| `colab/code_templates.py` | ~380 | Reusable code snippets for Colab |
| `colab/notebook_builder.py` | ~160 | Programmatic notebook construction |
| `colab/output_parser.py` | ~170 | Structured parsing of cell outputs |
| `kaggle/competition_client.py` | ~70 | Kaggle API wrapper |
| `strategies/escalation_rules.py` | ~100 | When to escalate model complexity |
| `config/model_registry.yaml` | ~130 | Model definitions + Optuna spaces |
| `config/settings.yaml` | ~35 | Global configuration |
| `scripts/run_competition.py` | ~110 | CLI entry point |
| `tests/test_core.py` | ~130 | Unit tests |

**Total: ~3,200 lines across 20 files**
