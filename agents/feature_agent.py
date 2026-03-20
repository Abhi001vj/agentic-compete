"""
Feature Engineering Agent
==========================
Generates and executes feature engineering code in Colab,
guided by EDA insights and LLM reasoning.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from anthropic import Anthropic

from colab.session_manager import ColabSession
from core.state import CompetitionState, FeatureSet

logger = logging.getLogger(__name__)


class FeatureAgent:
    """Agent that engineers features based on data profile and model feedback."""

    def __init__(self, model: str = "claude-sonnet-4-20250514"):
        self.client = Anthropic()
        self.model = model

    def engineer_features(self, state: CompetitionState) -> FeatureSet:
        """Generate and execute feature engineering in Colab."""
        import asyncio
        return asyncio.run(self._engineer_async(state))

    async def _engineer_async(self, state: CompetitionState) -> FeatureSet:
        session = ColabSession()

        # Ask Claude to generate feature engineering code
        code = self._generate_feature_code(state)

        result = await session.execute_code_with_retry(
            code,
            max_retries=3,
            fix_callback=self._fix_code,
        )

        # Parse the feature set info from output
        n_features = state.data_profile.n_cols  # default
        if result.success:
            try:
                for line in reversed(result.output.split("\n")):
                    if line.strip().startswith("{"):
                        info = json.loads(line.strip())
                        n_features = info.get("n_features", n_features)
                        break
            except json.JSONDecodeError:
                pass

        feature_set_name = f"engineered_v{len(state.feature_sets) + 1}"

        return FeatureSet(
            name=feature_set_name,
            description=f"Auto-engineered features based on EDA insights",
            n_features=n_features,
            creation_code=code,
        )

    def _generate_feature_code(self, state: CompetitionState) -> str:
        """Use Claude to generate feature engineering code."""
        prompt = f"""Generate Python feature engineering code for a Kaggle competition.

## Data Profile
- Task: {state.data_profile.task_type}
- Target: {state.data_profile.target_column}
- Numeric cols ({len(state.data_profile.numeric_cols)}): {state.data_profile.numeric_cols[:20]}
- Categorical cols ({len(state.data_profile.categorical_cols)}): {state.data_profile.categorical_cols[:20]}
- Missing: {json.dumps(dict(list(state.data_profile.missing_pct.items())[:10]), default=str)}
- Top features: {json.dumps(dict(list(state.data_profile.feature_importances.items())[:10]), default=str)}
- Key insights: {state.data_profile.key_insights}

## Previous feature sets
{[fs.name + ': ' + fs.description for fs in state.feature_sets]}

## Requirements
1. Load data from /content/data/X_train_processed.pkl and X_test_processed.pkl
2. Create meaningful feature interactions, aggregations, transformations
3. Handle edge cases (division by zero, inf values)
4. Save enhanced data to /content/data/X_train_enhanced.pkl and X_test_enhanced.pkl
5. Print JSON on last line: {{"n_features": N, "new_features": ["col1", "col2"]}}

Return ONLY Python code, no markdown fences."""

        response = self.client.messages.create(
            model=self.model,
            max_tokens=3000,
            messages=[{"role": "user", "content": prompt}],
        )

        code = response.content[0].text.strip()
        if code.startswith("```"):
            code = code.split("\n", 1)[1].rsplit("```", 1)[0]
        return code

    async def _fix_code(self, code: str, error: str) -> str:
        """Use Claude to fix broken feature engineering code."""
        prompt = f"""Fix this Python code that failed with an error.

## Code:
```python
{code}
```

## Error:
{error[:1000]}

Return ONLY the fixed Python code, no markdown fences."""

        response = self.client.messages.create(
            model=self.model,
            max_tokens=3000,
            messages=[{"role": "user", "content": prompt}],
        )

        fixed = response.content[0].text.strip()
        if fixed.startswith("```"):
            fixed = fixed.split("\n", 1)[1].rsplit("```", 1)[0]
        return fixed
