"""
Submission Agent
=================
Generates predictions and submits to Kaggle.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from colab.session_manager import ColabSession
from colab.code_templates import TRAINING_TEMPLATES
from core.state import CompetitionState

logger = logging.getLogger(__name__)


class SubmissionAgent:
    """Generates predictions and submits to Kaggle."""

    def submit(self, state: CompetitionState) -> dict[str, Any]:
        import asyncio
        return asyncio.run(self._submit_async(state))

    async def _submit_async(self, state: CompetitionState) -> dict[str, Any]:
        session = ColabSession()

        # Find the best model's details
        best = next(
            (r for r in state.model_results if r.model_id == state.best_model_id),
            state.model_results[0] if state.model_results else None,
        )

        if not best:
            return {"error": "No models available for submission"}

        # Generate submission
        code = f'''
import pandas as pd
import numpy as np
import json

# Retrain best model on full data and generate predictions
# Model: {best.model_name}
# CV Score: {best.cv_score}

X_train = pd.read_pickle("/content/data/X_train_processed.pkl")
y_train = pd.read_pickle("/content/data/y_train.pkl")
X_test = pd.read_pickle("/content/data/X_test_processed.pkl")

# Import and create model (will be filled by the orchestrator)
{self._get_model_code(best)}

model.fit(X_train, y_train)
preds = model.predict(X_test)

# Create submission file
import glob
sub_files = glob.glob("/content/data/*sample*sub*") + glob.glob("/content/data/*submission*")
if sub_files:
    sub = pd.read_csv(sub_files[0])
    sub.iloc[:, -1] = preds
    sub.to_csv("/content/submissions/submission.csv", index=False)
    print(f"Submission created: {{sub.shape}}")
    print(sub.head())

# Submit to Kaggle
import subprocess
try:
    result = subprocess.run(
        ["kaggle", "competitions", "submit",
         "-c", "{state.competition_slug}",
         "-f", "/content/submissions/submission.csv",
         "-m", "AgenticCompete: {best.model_name} CV={best.cv_score:.5f}"],
        capture_output=True, text=True
    )
    print(f"Kaggle response: {{result.stdout}}")
    print(json.dumps({{"status": "submitted", "model": "{best.model_name}", "cv_score": {best.cv_score}}}))
except Exception as e:
    print(json.dumps({{"status": "error", "error": str(e)}}))
'''

        result = await session.execute_code(code, timeout=600)

        return {
            "model": best.model_name,
            "cv_score": best.cv_score,
            "submitted": result.success,
            "output": result.output[:500],
        }

    def _get_model_code(self, model_result) -> str:
        """Generate the import + init code for the best model."""
        name = model_result.model_name.lower()
        params = model_result.hyperparameters

        if "xgb" in name:
            params_str = ", ".join(f"{k}={repr(v)}" for k, v in params.items()) if params else "n_estimators=500, random_state=42"
            return f"from xgboost import XGBClassifier, XGBRegressor\nmodel = XGBClassifier({params_str}, verbosity=0, n_jobs=-1)"
        elif "lgb" in name or "lightgbm" in name:
            params_str = ", ".join(f"{k}={repr(v)}" for k, v in params.items()) if params else "n_estimators=500, random_state=42"
            return f"from lightgbm import LGBMClassifier\nmodel = LGBMClassifier({params_str}, verbosity=-1, n_jobs=-1)"
        elif "catboost" in name:
            params_str = ", ".join(f"{k}={repr(v)}" for k, v in params.items()) if params else "iterations=500, random_seed=42"
            return f"from catboost import CatBoostClassifier\nmodel = CatBoostClassifier({params_str}, verbose=0)"
        elif "random" in name:
            return "from sklearn.ensemble import RandomForestClassifier\nmodel = RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1)"
        elif "logistic" in name:
            return "from sklearn.linear_model import LogisticRegression\nmodel = LogisticRegression(max_iter=1000, random_state=42)"
        else:
            return "from xgboost import XGBClassifier\nmodel = XGBClassifier(n_estimators=500, random_state=42, n_jobs=-1, verbosity=0)"
