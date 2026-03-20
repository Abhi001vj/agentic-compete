"""
Competition Analyzer
=====================
Analyzes a Kaggle competition to extract metadata:
metric, task type, rules, special constraints.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from anthropic import Anthropic

logger = logging.getLogger(__name__)


class CompetitionAnalyzer:
    """Analyzes competition metadata using Kaggle API + LLM reasoning."""

    def __init__(self, model: str = "claude-sonnet-4-20250514"):
        self.client = Anthropic()
        self.model = model

    def analyze(self, competition_slug: str, description: str = "") -> dict[str, Any]:
        """
        Analyze a competition and return structured metadata.
        Uses Kaggle API for data, Claude for reasoning.
        """
        # Try to get info from Kaggle API
        comp_info = self._fetch_kaggle_info(competition_slug)

        # Use LLM to analyze and structure the information
        prompt = f"""Analyze this Kaggle competition and extract key information.

Competition slug: {competition_slug}
Description: {description or comp_info.get('description', 'Not available')}
Kaggle API info: {json.dumps(comp_info, default=str)}

Return ONLY valid JSON:
{{
  "name": "Full competition name",
  "description": "Brief description of the task",
  "metric": "evaluation metric (e.g., auc, accuracy, rmse, f1)",
  "metric_direction": "maximize or minimize",
  "task_type": "binary_classification, multiclass_classification, or regression",
  "data_format": "tabular, image, text, or mixed",
  "special_notes": "Any special rules or constraints",
  "reasoning": "Your analysis"
}}"""

        response = self.client.messages.create(
            model=self.model,
            max_tokens=800,
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
                "name": competition_slug,
                "description": description,
                "metric": "accuracy",
                "metric_direction": "maximize",
                "task_type": "binary_classification",
                "reasoning": "Failed to parse competition info, using defaults.",
            }

    def _fetch_kaggle_info(self, competition_slug: str) -> dict:
        """Fetch competition info from Kaggle API."""
        try:
            import kaggle
            api = kaggle.KaggleApi()
            api.authenticate()
            competitions = api.competitions_list(search=competition_slug)
            for comp in competitions:
                if comp.ref.endswith(competition_slug):
                    return {
                        "title": comp.title,
                        "description": comp.description,
                        "evaluationMetric": comp.evaluationMetric,
                        "deadline": str(comp.deadline),
                        "category": comp.category,
                        "tags": [t.name for t in comp.tags] if comp.tags else [],
                    }
        except Exception as e:
            logger.warning(f"Failed to fetch Kaggle info: {e}")
        return {}
