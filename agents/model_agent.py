"""
Model Agent
============
Trains models at different complexity tiers via Colab MCP.
Follows the progressive strategy: baseline → medium → complex.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from anthropic import Anthropic

from colab.session_manager import ColabSession
from colab.code_templates import TRAINING_TEMPLATES
from core.state import CompetitionState, ModelResult, ModelTier

logger = logging.getLogger(__name__)


# Model registry: tier → list of (name, imports, init, scoring)
MODEL_REGISTRY: dict[ModelTier, list[dict[str, str]]] = {
    ModelTier.BASELINE: [
        {
            "name": "LogisticRegression",
            "imports": "from sklearn.linear_model import LogisticRegression",
            "init": "LogisticRegression(max_iter=1000, random_state=42)",
            "task": "classification",
        },
        {
            "name": "RidgeClassifier",
            "imports": "from sklearn.linear_model import RidgeClassifier",
            "init": "RidgeClassifier(random_state=42)",
            "task": "classification",
        },
        {
            "name": "RandomForest",
            "imports": "from sklearn.ensemble import RandomForestClassifier",
            "init": "RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)",
            "task": "classification",
        },
        {
            "name": "XGBoost_default",
            "imports": "from xgboost import XGBClassifier",
            "init": "XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1, verbosity=0)",
            "task": "classification",
        },
        {
            "name": "Ridge_Regression",
            "imports": "from sklearn.linear_model import Ridge",
            "init": "Ridge(alpha=1.0, random_state=42)",
            "task": "regression",
        },
        {
            "name": "RandomForest_Reg",
            "imports": "from sklearn.ensemble import RandomForestRegressor",
            "init": "RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)",
            "task": "regression",
        },
        {
            "name": "XGBoost_Reg_default",
            "imports": "from xgboost import XGBRegressor",
            "init": "XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1, verbosity=0)",
            "task": "regression",
        },
    ],
    ModelTier.MEDIUM: [
        {
            "name": "LightGBM",
            "imports": "from lightgbm import LGBMClassifier",
            "init": "LGBMClassifier(n_estimators=500, learning_rate=0.05, max_depth=7, num_leaves=63, subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=-1, n_jobs=-1)",
            "task": "classification",
        },
        {
            "name": "CatBoost",
            "imports": "from catboost import CatBoostClassifier",
            "init": "CatBoostClassifier(iterations=500, learning_rate=0.05, depth=7, random_seed=42, verbose=0)",
            "task": "classification",
        },
        {
            "name": "XGBoost_tuned",
            "imports": "from xgboost import XGBClassifier",
            "init": "XGBClassifier(n_estimators=500, max_depth=7, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0, random_state=42, n_jobs=-1, verbosity=0)",
            "task": "classification",
        },
        {
            "name": "LightGBM_Reg",
            "imports": "from lightgbm import LGBMRegressor",
            "init": "LGBMRegressor(n_estimators=500, learning_rate=0.05, max_depth=7, num_leaves=63, subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=-1, n_jobs=-1)",
            "task": "regression",
        },
        {
            "name": "CatBoost_Reg",
            "imports": "from catboost import CatBoostRegressor",
            "init": "CatBoostRegressor(iterations=500, learning_rate=0.05, depth=7, random_seed=42, verbose=0)",
            "task": "regression",
        },
    ],
    ModelTier.COMPLEX: [
        {
            "name": "XGBoost_optuna",
            "type": "optuna",
            "task": "classification",
        },
        {
            "name": "LightGBM_optuna",
            "type": "optuna",
            "task": "classification",
        },
        {
            "name": "Stacking",
            "type": "stacking",
            "task": "both",
        },
    ],
}

# Metric mapping
METRIC_TO_SCORING = {
    "auc": "roc_auc",
    "roc_auc": "roc_auc",
    "accuracy": "accuracy",
    "f1": "f1",
    "f1_weighted": "f1_weighted",
    "log_loss": "neg_log_loss",
    "rmse": "neg_root_mean_squared_error",
    "mae": "neg_mean_absolute_error",
    "r2": "r2",
    "mse": "neg_mean_squared_error",
}


class ModelAgent:
    """Agent that trains models at different complexity tiers."""

    def __init__(self, model: str = "claude-sonnet-4-20250514"):
        self.client = Anthropic()
        self.model = model

    def train_tier(
        self,
        state: CompetitionState,
        tier: ModelTier,
        models: list[str] | None = None,
        feature_set: str = "original",
    ) -> list[ModelResult]:
        """Train all models in a given tier."""
        import asyncio
        return asyncio.run(self._train_tier_async(state, tier, models, feature_set))

    async def _train_tier_async(
        self,
        state: CompetitionState,
        tier: ModelTier,
        models: list[str] | None,
        feature_set: str,
    ) -> list[ModelResult]:
        session = ColabSession()
        results = []

        # Determine task type and scoring
        task = "classification" if "classif" in state.data_profile.task_type else "regression"
        scoring = METRIC_TO_SCORING.get(state.evaluation_metric, "roc_auc")

        # Get models for this tier, filtered by task
        tier_models = MODEL_REGISTRY.get(tier, [])
        tier_models = [
            m for m in tier_models
            if m["task"] == task or m.get("task") == "both"
        ]

        # Filter by requested model names if specified
        if models:
            tier_models = [
                m for m in tier_models
                if m["name"] in models or any(n.lower() in m["name"].lower() for n in models)
            ]

        for model_config in tier_models:
            if model_config.get("type") == "optuna":
                result = await self._train_optuna(session, state, model_config, scoring, tier)
            elif model_config.get("type") == "stacking":
                result = await self._train_stacking(session, state, scoring, tier)
            else:
                result = await self._train_standard(
                    session, state, model_config, scoring, tier, feature_set
                )
            if result:
                results.append(result)

        return results

    async def _train_standard(
        self,
        session: ColabSession,
        state: CompetitionState,
        model_config: dict,
        scoring: str,
        tier: ModelTier,
        feature_set: str,
    ) -> ModelResult | None:
        """Train a standard sklearn-compatible model."""
        template_key = (
            "baseline_classification"
            if "classif" in state.data_profile.task_type
            else "baseline_regression"
        )
        code = TRAINING_TEMPLATES[template_key].format(
            model_imports=model_config["imports"],
            model_init=model_config["init"],
            model_name=model_config["name"],
            scoring=scoring,
            metric_func="roc_auc_score",
        )

        result = await session.run_training_cell(code, model_config["name"])

        if result.success:
            try:
                parsed = json.loads(
                    next(l for l in reversed(result.output.split("\n"))
                         if l.strip().startswith("{"))
                )
                return ModelResult(
                    model_name=parsed["model_name"],
                    tier=tier,
                    cv_score=parsed["cv_mean"],
                    cv_std=parsed["cv_std"],
                    training_time_sec=parsed["training_time"],
                    feature_set=feature_set,
                )
            except (json.JSONDecodeError, StopIteration) as e:
                logger.warning(f"Failed to parse result for {model_config['name']}: {e}")

        return None

    async def _train_optuna(
        self,
        session: ColabSession,
        state: CompetitionState,
        model_config: dict,
        scoring: str,
        tier: ModelTier,
    ) -> ModelResult | None:
        """Train with Optuna hyperparameter tuning."""
        # Use LLM to generate the Optuna search space based on model and data
        search_space = self._generate_optuna_space(model_config["name"], state)

        code = TRAINING_TEMPLATES["optuna_tuning"].format(
            optuna_search_space=search_space["search_space"],
            model_init_with_params=search_space["model_init"],
            scoring=scoring,
            direction="maximize" if state.metric_direction == "maximize" else "minimize",
            n_trials=50,
            timeout=600,  # 10 minutes
        )

        result = await session.run_training_cell(code, f"{model_config['name']}_optuna")

        if result.success:
            try:
                parsed = json.loads(
                    next(l for l in reversed(result.output.split("\n"))
                         if l.strip().startswith("{"))
                )
                return ModelResult(
                    model_name=f"{model_config['name']}_optuna",
                    tier=tier,
                    cv_score=parsed["best_score"],
                    hyperparameters=parsed["best_params"],
                    training_time_sec=parsed["tuning_time"],
                    notes=f"Optuna: {parsed['n_trials_completed']} trials",
                )
            except (json.JSONDecodeError, StopIteration):
                pass
        return None

    async def _train_stacking(
        self,
        session: ColabSession,
        state: CompetitionState,
        scoring: str,
        tier: ModelTier,
    ) -> ModelResult | None:
        """Create a stacking ensemble from top models."""
        # Get top 3 model results
        top_models = sorted(
            state.model_results,
            key=lambda r: r.cv_score,
            reverse=(state.metric_direction == "maximize"),
        )[:3]

        # Generate stacking code via LLM
        code = self._generate_stacking_code(top_models, state, scoring)
        result = await session.run_training_cell(code, "Stacking_Ensemble")

        if result.success:
            try:
                parsed = json.loads(
                    next(l for l in reversed(result.output.split("\n"))
                         if l.strip().startswith("{"))
                )
                return ModelResult(
                    model_name="Stacking_Ensemble",
                    tier=tier,
                    cv_score=parsed["cv_mean"],
                    cv_std=parsed.get("cv_std", 0),
                    training_time_sec=parsed.get("training_time", 0),
                    notes=f"Stacking of: {[m.model_name for m in top_models]}",
                )
            except (json.JSONDecodeError, StopIteration):
                pass
        return None

    def _generate_optuna_space(self, model_name: str, state: CompetitionState) -> dict:
        """Use Claude to generate an Optuna search space for the given model."""
        prompt = f"""Generate an Optuna search space for {model_name} for a {'classification' if 'classif' in state.data_profile.task_type else 'regression'} task.

Dataset: {state.data_profile.n_rows} rows, {state.data_profile.n_cols} columns.
Metric: {state.evaluation_metric}

Return JSON with two keys:
- "search_space": Python code for Optuna trial parameter suggestions
- "model_init": Python code to create the model with trial params

Example format:
{{"search_space": "n_estimators = trial.suggest_int('n_estimators', 100, 1000)\\nlearning_rate = trial.suggest_float('learning_rate', 0.01, 0.3, log=True)",
"model_init": "XGBClassifier(n_estimators=n_estimators, learning_rate=learning_rate, random_state=42)"}}"""

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
            # Fallback to a sensible default
            return {
                "search_space": (
                    "n_estimators = trial.suggest_int('n_estimators', 100, 1000)\n"
                    "learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3, log=True)\n"
                    "max_depth = trial.suggest_int('max_depth', 3, 10)"
                ),
                "model_init": "XGBClassifier(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, random_state=42, n_jobs=-1, verbosity=0)",
            }

    def _generate_stacking_code(
        self,
        top_models: list[ModelResult],
        state: CompetitionState,
        scoring: str,
    ) -> str:
        """Generate stacking ensemble code."""
        # Use Claude to generate appropriate stacking code
        prompt = f"""Generate Python code for a sklearn StackingClassifier/Regressor that combines these models:
{[m.model_name for m in top_models]}

Task: {'classification' if 'classif' in state.data_profile.task_type else 'regression'}
Scoring: {scoring}

The code should:
1. Import all necessary libraries
2. Create base estimators
3. Create StackingClassifier/Regressor with LogisticRegression/Ridge as meta-learner
4. Run 5-fold CV
5. Print JSON with cv_mean, cv_std, training_time

Use data from: X = pd.read_pickle("/content/data/X_train_processed.pkl")
y = pd.read_pickle("/content/data/y_train.pkl")

Return ONLY the Python code, no markdown."""

        response = self.client.messages.create(
            model=self.model,
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}],
        )
        code = response.content[0].text.strip()
        if code.startswith("```"):
            code = code.split("\n", 1)[1].rsplit("```", 1)[0]
        return code
