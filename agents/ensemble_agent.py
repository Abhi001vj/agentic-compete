"""
Ensemble Agent
===============
Creates model ensembles from top performers.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from anthropic import Anthropic

from colab.session_manager import ColabSession
from core.state import CompetitionState, ModelResult, ModelTier

logger = logging.getLogger(__name__)


class EnsembleAgent:
    """Creates ensembles from best models."""

    def __init__(self, model: str = "claude-sonnet-4-20250514"):
        self.client = Anthropic()
        self.model = model

    def create_ensemble(self, state: CompetitionState) -> ModelResult:
        """Create a weighted ensemble from top models."""
        import asyncio
        return asyncio.run(self._create_async(state))

    async def _create_async(self, state: CompetitionState) -> ModelResult:
        session = ColabSession()

        # Get top N diverse models
        top_models = self._select_diverse_models(state, n=5)

        # Generate ensemble code via LLM
        code = self._generate_ensemble_code(top_models, state)
        result = await session.run_training_cell(code, "Ensemble")

        if result.success:
            try:
                parsed = json.loads(
                    next(l for l in reversed(result.output.split("\n"))
                         if l.strip().startswith("{"))
                )
                return ModelResult(
                    model_name="Weighted_Ensemble",
                    tier=ModelTier.ENSEMBLE,
                    cv_score=parsed.get("cv_mean", state.best_score),
                    cv_std=parsed.get("cv_std", 0),
                    training_time_sec=parsed.get("training_time", 0),
                    notes=f"Ensemble of {len(top_models)} models",
                )
            except (json.JSONDecodeError, StopIteration):
                pass

        return ModelResult(
            model_name="Ensemble_Failed",
            tier=ModelTier.ENSEMBLE,
            cv_score=state.best_score,
            notes="Ensemble creation failed, using best single model",
        )

    def _select_diverse_models(
        self, state: CompetitionState, n: int = 5
    ) -> list[ModelResult]:
        """Select top N models, preferring diversity in model type."""
        sorted_results = sorted(
            state.model_results,
            key=lambda r: r.cv_score,
            reverse=(state.metric_direction == "maximize"),
        )
        selected = []
        seen_types = set()
        for r in sorted_results:
            base_name = r.model_name.split("_")[0]
            if base_name not in seen_types or len(selected) < n:
                selected.append(r)
                seen_types.add(base_name)
            if len(selected) >= n:
                break
        return selected

    def _generate_ensemble_code(
        self, models: list[ModelResult], state: CompetitionState
    ) -> str:
        prompt = f"""Generate Python code for an optimized ensemble of these models:
{json.dumps([{{"name": m.model_name, "score": m.cv_score, "params": m.hyperparameters}} for m in models], indent=2, default=str)}

Task: {state.data_profile.task_type}
Metric: {state.evaluation_metric}

Use weighted averaging with Optuna to find optimal weights.
Load data from /content/data/X_train_processed.pkl and y_train.pkl.
Print JSON: {{"cv_mean": X, "cv_std": X, "weights": [...], "training_time": X}}
Return ONLY Python code."""

        response = self.client.messages.create(
            model=self.model, max_tokens=3000,
            messages=[{"role": "user", "content": prompt}],
        )
        code = response.content[0].text.strip()
        if code.startswith("```"):
            code = code.split("\n", 1)[1].rsplit("```", 1)[0]
        return code
