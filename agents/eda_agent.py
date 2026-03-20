"""
EDA Agent
==========
Performs comprehensive Exploratory Data Analysis by generating Python code,
executing it in Colab via MCP, and parsing results into a DataProfile.

This agent follows a structured EDA workflow:
1. Load & inspect data (shape, dtypes, head/tail)
2. Missing value analysis
3. Target distribution analysis
4. Numerical feature statistics & distributions
5. Categorical feature analysis
6. Correlation analysis
7. Feature importance (quick RF-based)
8. Key insights summary
"""

from __future__ import annotations

import json
import logging
from typing import Any

from anthropic import Anthropic

from colab.session_manager import ColabSession
from colab.code_templates import EDA_TEMPLATES
from core.state import CompetitionState, DataProfile

logger = logging.getLogger(__name__)


class EDAAgent:
    """Agent that runs EDA in Colab and builds a DataProfile."""

    def __init__(self, model: str = "claude-sonnet-4-20250514"):
        self.client = Anthropic()
        self.model = model

    def run(self, state: CompetitionState) -> DataProfile:
        """Execute full EDA pipeline and return a DataProfile."""
        import asyncio
        return asyncio.run(self._run_async(state))

    async def _run_async(self, state: CompetitionState) -> DataProfile:
        session = ColabSession()
        profile = DataProfile()

        # Step 1: Load data and get basic info
        result = await session.execute_code(
            EDA_TEMPLATES["load_and_inspect"],
            add_markdown_header="1. Data Loading & Inspection",
        )
        if result.success:
            info = self._parse_data_info(result.output)
            profile.n_rows = info.get("n_rows", 0)
            profile.n_cols = info.get("n_cols", 0)
            profile.n_train = info.get("n_train", 0)
            profile.n_test = info.get("n_test", 0)
            profile.numeric_cols = info.get("numeric_cols", [])
            profile.categorical_cols = info.get("categorical_cols", [])
            profile.text_cols = info.get("text_cols", [])

        # Step 2: Missing values
        result = await session.execute_code(
            EDA_TEMPLATES["missing_values"],
            add_markdown_header="2. Missing Value Analysis",
        )
        if result.success:
            profile.missing_pct = self._parse_json_output(result.output)

        # Step 3: Target distribution
        target_code = EDA_TEMPLATES["target_distribution"].format(
            target=state.data_profile.target_column or self._infer_target(result.output)
        )
        result = await session.execute_code(
            target_code,
            add_markdown_header="3. Target Variable Analysis",
        )
        if result.success:
            target_info = self._parse_json_output(result.output)
            profile.target_distribution = target_info
            profile.target_column = target_info.get("column", "")
            task_type = target_info.get("task_type", "")
            profile.task_type = task_type or state.data_profile.task_type

        # Step 4: Numerical distributions
        result = await session.execute_code(
            EDA_TEMPLATES["numerical_distributions"],
            add_markdown_header="4. Numerical Feature Distributions",
        )

        # Step 5: Categorical analysis
        result = await session.execute_code(
            EDA_TEMPLATES["categorical_analysis"],
            add_markdown_header="5. Categorical Feature Analysis",
        )

        # Step 6: Correlation matrix
        result = await session.execute_code(
            EDA_TEMPLATES["correlations"],
            add_markdown_header="6. Feature Correlations",
        )
        if result.success:
            profile.correlations = self._parse_json_output(result.output)

        # Step 7: Quick feature importance
        result = await session.execute_code(
            EDA_TEMPLATES["feature_importance"],
            add_markdown_header="7. Feature Importance (Quick RF)",
        )
        if result.success:
            profile.feature_importances = self._parse_json_output(result.output)

        # Step 8: LLM-generated insights
        profile.key_insights = self._generate_insights(profile, state)

        return profile

    def _generate_insights(
        self, profile: DataProfile, state: CompetitionState
    ) -> list[str]:
        """Use Claude to generate key insights from the EDA results."""
        prompt = f"""Based on this EDA summary for Kaggle competition '{state.competition_slug}':

- Rows: {profile.n_rows}, Columns: {profile.n_cols}
- Task: {profile.task_type}
- Target: {profile.target_column}
- Target distribution: {json.dumps(profile.target_distribution, default=str)}
- Numeric cols ({len(profile.numeric_cols)}): {profile.numeric_cols[:20]}
- Categorical cols ({len(profile.categorical_cols)}): {profile.categorical_cols[:20]}
- Missing values: {json.dumps(dict(list(profile.missing_pct.items())[:10]), default=str)}
- Top correlations with target: {json.dumps(dict(list(profile.correlations.items())[:10]), default=str)}
- Top feature importances: {json.dumps(dict(list(profile.feature_importances.items())[:10]), default=str)}

Generate 5-8 key insights that would guide the modeling strategy. Focus on:
1. Data quality issues
2. Class imbalance
3. High-importance features
4. Potential feature interactions
5. Recommended preprocessing steps
6. Model selection implications

Return ONLY a JSON array of strings."""

        response = self.client.messages.create(
            model=self.model,
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}],
        )
        try:
            text = response.content[0].text.strip()
            # Strip markdown code fences if present
            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("```", 1)[0]
            return json.loads(text)
        except (json.JSONDecodeError, IndexError):
            return [response.content[0].text]

    def _parse_data_info(self, output: str) -> dict[str, Any]:
        """Parse the output of the load_and_inspect template."""
        try:
            for line in output.strip().split("\n"):
                if line.strip().startswith("{"):
                    return json.loads(line.strip())
        except json.JSONDecodeError:
            pass
        return {}

    def _parse_json_output(self, output: str) -> dict:
        """Parse JSON from cell output."""
        try:
            for line in reversed(output.strip().split("\n")):
                line = line.strip()
                if line.startswith("{"):
                    return json.loads(line)
        except json.JSONDecodeError:
            pass
        return {}

    def _infer_target(self, output: str) -> str:
        """Infer target column name from data inspection output."""
        # Common Kaggle target column names
        common_targets = [
            "target", "label", "y", "class", "outcome", "is_", "survived",
            "salePrice", "price", "revenue", "default", "fraud",
        ]
        # Simple heuristic — will be overridden by LLM analysis
        return "target"
