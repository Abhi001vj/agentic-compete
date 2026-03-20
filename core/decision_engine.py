"""
Decision Engine
================
Uses Claude to make strategic decisions at key points in the pipeline.
This is the "brain" that decides whether to escalate, optimize, or submit.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from anthropic import Anthropic

from core.state import CompetitionState, ModelTier

logger = logging.getLogger(__name__)


class DecisionEngine:
    """LLM-powered decision engine for strategic choices."""

    def __init__(self, model: str = "claude-sonnet-4-20250514"):
        self.client = Anthropic()
        self.model = model

    def create_strategy(self, state: CompetitionState) -> dict[str, Any]:
        """
        Create a progressive modeling strategy based on EDA findings.
        Returns a structured plan with model selections for each tier.
        """
        prompt = f"""You are an expert Kaggle competitor creating a modeling strategy.

## Competition
- Name: {state.competition_slug}
- Metric: {state.evaluation_metric} ({state.metric_direction})
- Description: {state.competition_description[:500]}

## Data Profile
- Rows: {state.data_profile.n_rows}, Columns: {state.data_profile.n_cols}
- Task: {state.data_profile.task_type}
- Target: {state.data_profile.target_column}
- Target distribution: {json.dumps(state.data_profile.target_distribution, default=str)}
- Numeric columns: {len(state.data_profile.numeric_cols)}
- Categorical columns: {len(state.data_profile.categorical_cols)}
- Text columns: {len(state.data_profile.text_cols)}
- Missing values: {json.dumps(dict(list(state.data_profile.missing_pct.items())[:10]), default=str)}
- Key insights: {state.data_profile.key_insights}

## Your Task
Create a 3-tier progressive strategy. Return ONLY valid JSON:

{{
  "reasoning": "Your analysis of why this strategy is appropriate (2-3 sentences)",
  "strategy_text": "Human-readable strategy summary",
  "baseline_models": ["model1", "model2"],
  "medium_models": ["model1", "model2"],
  "complex_models": ["model1", "model2"],
  "feature_engineering_ideas": ["idea1", "idea2"],
  "preprocessing_notes": "Any special preprocessing needed",
  "cv_strategy": "Recommended cross-validation approach"
}}

Available baseline models: LogisticRegression, RidgeClassifier, RandomForest, XGBoost_default
Available medium models: LightGBM, CatBoost, XGBoost_tuned
Available complex models: XGBoost_optuna, LightGBM_optuna, Stacking

Pick models appropriate for the data size and type. For small datasets, simpler is better.
For large datasets with many features, GBMs are usually strongest."""

        response = self.client.messages.create(
            model=self.model,
            max_tokens=1500,
            messages=[{"role": "user", "content": prompt}],
        )

        try:
            text = response.content[0].text.strip()
            if "```" in text:
                text = text.split("```")[1].split("```")[0]
                if text.startswith("json"):
                    text = text[4:]
            return json.loads(text)
        except (json.JSONDecodeError, IndexError):
            # Sensible defaults
            is_clf = "classif" in state.data_profile.task_type
            return {
                "reasoning": "Defaulting to standard progressive strategy.",
                "strategy_text": "Baseline GBMs → Tuned GBMs → Optuna + Stacking",
                "baseline_models": ["RandomForest", "XGBoost_default"] if is_clf else ["RandomForest_Reg", "XGBoost_Reg_default"],
                "medium_models": ["LightGBM", "CatBoost"] if is_clf else ["LightGBM_Reg", "CatBoost_Reg"],
                "complex_models": ["XGBoost_optuna", "Stacking"],
            }

    def evaluate_tier(
        self, state: CompetitionState, tier: ModelTier
    ) -> dict[str, Any]:
        """
        Evaluate current tier results and decide next action.

        Returns:
            {"action": "extract_max" | "escalate" | "submit", "reasoning": "..."}
        """
        results_summary = state.get_results_summary()
        tier_results = [r for r in state.model_results if r.tier == tier]

        prompt = f"""You are an expert Kaggle competitor evaluating model results.

## Current Situation
- Competition: {state.competition_slug}
- Metric: {state.evaluation_metric} ({state.metric_direction})
- Current tier: {tier.value}
- Iterations completed: {state.iteration}
- Time used: (check runtime)

## Results So Far
{results_summary}

## Current Tier Results ({tier.value})
{json.dumps([{{"name": r.model_name, "cv": r.cv_score, "std": r.cv_std}} for r in tier_results], indent=2)}

## Improvement History
{json.dumps(state.improvement_history, indent=2, default=str)}

## Decision Required
Choose ONE action and explain why:

1. **extract_max**: The {tier.value} models show promise. Invest in feature engineering 
   and hyperparameter tuning at this tier before moving on.
   - Choose this if: scores are reasonable, there's likely room for improvement with 
     better features or tuning, AND we haven't already done extensive feature engineering.

2. **escalate**: The {tier.value} models have plateaued. Move to the next complexity tier.
   - Choose this if: models are underperforming, multiple attempts haven't improved scores, 
     OR the data characteristics suggest more powerful models are needed.

3. **submit**: Current results are strong enough for a competitive submission.
   - Choose this if: we're at the complex tier with good results, or time is running out.

{"Feature engineering has already been done " + str(len(state.feature_sets)) + " time(s)." if state.feature_sets else "No feature engineering done yet."}

Return ONLY valid JSON:
{{"action": "extract_max|escalate|submit", "reasoning": "Your reasoning"}}"""

        response = self.client.messages.create(
            model=self.model,
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}],
        )

        try:
            text = response.content[0].text.strip()
            if "```" in text:
                text = text.split("```")[1].split("```")[0]
                if text.startswith("json"):
                    text = text[4:]
            result = json.loads(text)
            assert result["action"] in ("extract_max", "escalate", "submit")
            return result
        except (json.JSONDecodeError, KeyError, AssertionError):
            # Default: if we have feature sets, escalate; otherwise extract_max
            if state.feature_sets:
                return {
                    "action": "escalate",
                    "reasoning": "Already tried feature engineering. Escalating to next tier."
                }
            return {
                "action": "extract_max",
                "reasoning": "Haven't tried feature engineering yet. Extracting max from current tier."
            }

    def select_features_to_engineer(self, state: CompetitionState) -> dict[str, Any]:
        """Use LLM to determine which features to engineer."""
        prompt = f"""Based on EDA results for {state.competition_slug}:

Top features by importance:
{json.dumps(dict(list(state.data_profile.feature_importances.items())[:15]), indent=2)}

Top correlations:
{json.dumps(dict(list(state.data_profile.correlations.items())[:15]), indent=2)}

Current columns: {state.data_profile.numeric_cols[:20]}

Suggest specific feature engineering operations. Return JSON:
{{
  "interactions": [["col1", "col2"], ["col3", "col4"]],
  "aggregations": [{{"group_by": "col", "agg_cols": ["col1"], "agg_funcs": ["mean", "std"]}}],
  "transformations": [{{"col": "col1", "transform": "log1p"}}],
  "reasoning": "Why these features matter"
}}"""

        response = self.client.messages.create(
            model=self.model,
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}],
        )

        try:
            text = response.content[0].text.strip()
            if "```" in text:
                text = text.split("```")[1].split("```")[0]
                if text.startswith("json"):
                    text = text[4:]
            return json.loads(text)
        except (json.JSONDecodeError, IndexError):
            return {
                "interactions": [],
                "aggregations": [],
                "transformations": [],
                "reasoning": "Failed to generate feature ideas.",
            }
