"""
AgenticCompete State Definition
================================
Central state that flows through the entire LangGraph orchestrator.
Every agent reads from and writes to this state.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional


class Phase(str, Enum):
    """Current phase in the competition pipeline."""
    INIT = "init"
    ANALYZE = "analyze_competition"
    DOWNLOAD = "download_data"
    EDA = "eda"
    PLAN = "plan_strategy"
    BASELINE = "baseline_models"
    EVALUATE_BASELINE = "evaluate_baseline"
    FEATURE_ENG = "feature_engineering"
    MEDIUM_MODELS = "medium_models"
    EVALUATE_MEDIUM = "evaluate_medium"
    COMPLEX_MODELS = "complex_models"
    EVALUATE_COMPLEX = "evaluate_complex"
    TUNING = "hyperparameter_tuning"
    ENSEMBLE = "ensemble"
    SUBMISSION = "final_submission"
    DONE = "done"
    ERROR = "error"


class ModelTier(str, Enum):
    BASELINE = "baseline"
    MEDIUM = "medium"
    COMPLEX = "complex"
    ENSEMBLE = "ensemble"


@dataclass
class DataProfile:
    """Profile of the competition dataset discovered during EDA."""
    n_rows: int = 0
    n_cols: int = 0
    n_train: int = 0
    n_test: int = 0
    target_column: str = ""
    task_type: str = ""  # "binary_classification", "multiclass", "regression"
    numeric_cols: list[str] = field(default_factory=list)
    categorical_cols: list[str] = field(default_factory=list)
    text_cols: list[str] = field(default_factory=list)
    datetime_cols: list[str] = field(default_factory=list)
    missing_pct: dict[str, float] = field(default_factory=dict)
    target_distribution: dict[str, Any] = field(default_factory=dict)
    feature_importances: dict[str, float] = field(default_factory=dict)
    correlations: dict[str, float] = field(default_factory=dict)
    key_insights: list[str] = field(default_factory=list)
    eda_plots: list[str] = field(default_factory=list)  # Paths to saved plots


@dataclass
class ModelResult:
    """Result from training a single model."""
    model_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    model_name: str = ""
    tier: ModelTier = ModelTier.BASELINE
    cv_score: float = 0.0
    cv_std: float = 0.0
    lb_score: Optional[float] = None  # Leaderboard score if submitted
    training_time_sec: float = 0.0
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    feature_set: str = "original"  # Which feature set was used
    notes: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class FeatureSet:
    """A named collection of engineered features."""
    name: str = ""
    description: str = ""
    n_features: int = 0
    feature_names: list[str] = field(default_factory=list)
    creation_code: str = ""  # The Python code that creates these features
    impact_vs_baseline: Optional[float] = None


@dataclass
class CompetitionState:
    """
    Master state for the AgenticCompete orchestrator.
    This is what flows through every node in the LangGraph.
    """
    # --- Competition metadata ---
    competition_slug: str = ""
    competition_name: str = ""
    competition_description: str = ""
    evaluation_metric: str = ""  # "auc", "rmse", "accuracy", "f1", etc.
    metric_direction: str = "maximize"  # "maximize" or "minimize"
    deadline: Optional[str] = None
    max_runtime_hours: float = 8.0

    # --- Current state ---
    phase: Phase = Phase.INIT
    current_tier: ModelTier = ModelTier.BASELINE
    iteration: int = 0
    started_at: str = field(default_factory=lambda: datetime.now().isoformat())

    # --- Data profile (populated by EDA agent) ---
    data_profile: DataProfile = field(default_factory=DataProfile)

    # --- Strategy (populated by planning phase) ---
    strategy_plan: str = ""  # Natural language plan from LLM
    baseline_models_planned: list[str] = field(default_factory=list)
    medium_models_planned: list[str] = field(default_factory=list)
    complex_models_planned: list[str] = field(default_factory=list)

    # --- Results tracking ---
    model_results: list[ModelResult] = field(default_factory=list)
    best_score: float = 0.0
    best_model_id: str = ""
    best_lb_score: Optional[float] = None

    # --- Feature engineering ---
    feature_sets: list[FeatureSet] = field(default_factory=list)
    current_feature_set: str = "original"

    # --- Escalation tracking ---
    baseline_exhausted: bool = False
    medium_exhausted: bool = False
    complex_exhausted: bool = False
    improvement_history: list[dict[str, Any]] = field(default_factory=list)

    # --- Colab session ---
    colab_notebook_id: Optional[str] = None
    colab_cells_executed: int = 0

    # --- Agent reasoning log ---
    reasoning_log: list[dict[str, str]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    # --- Kaggle submission tracking ---
    submissions: list[dict[str, Any]] = field(default_factory=list)

    def add_result(self, result: ModelResult) -> None:
        """Add a model result and update best tracking."""
        self.model_results.append(result)
        is_better = (
            (self.metric_direction == "maximize" and result.cv_score > self.best_score)
            or (self.metric_direction == "minimize" and result.cv_score < self.best_score)
            or self.best_score == 0.0
        )
        if is_better:
            self.best_score = result.cv_score
            self.best_model_id = result.model_id
            self.improvement_history.append({
                "iteration": self.iteration,
                "model": result.model_name,
                "score": result.cv_score,
                "tier": result.tier.value,
                "timestamp": result.timestamp,
            })

    def log_reasoning(self, agent: str, reasoning: str) -> None:
        """Log agent reasoning for traceability."""
        self.reasoning_log.append({
            "agent": agent,
            "reasoning": reasoning,
            "phase": self.phase.value,
            "timestamp": datetime.now().isoformat(),
        })

    def should_escalate(self, patience: int = 3) -> bool:
        """
        Determine if we should escalate to the next model tier.
        Returns True if the last `patience` models showed no improvement.
        """
        if len(self.model_results) < patience:
            return False
        recent = self.model_results[-patience:]
        recent_best = max(r.cv_score for r in recent) if self.metric_direction == "maximize" else min(r.cv_score for r in recent)
        return abs(recent_best - self.best_score) < 1e-6

    def get_results_summary(self) -> str:
        """Get a formatted summary of all model results."""
        if not self.model_results:
            return "No models trained yet."
        lines = ["Model Results Summary:", "=" * 60]
        for r in sorted(self.model_results, key=lambda x: x.cv_score,
                        reverse=(self.metric_direction == "maximize")):
            marker = " ★" if r.model_id == self.best_model_id else ""
            lines.append(
                f"  [{r.tier.value:>8}] {r.model_name:<25} "
                f"CV: {r.cv_score:.5f} ± {r.cv_std:.5f} "
                f"({r.training_time_sec:.0f}s){marker}"
            )
        lines.append(f"\nBest: {self.best_score:.5f}")
        return "\n".join(lines)
