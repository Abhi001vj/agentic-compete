"""
Competition Memory
===================
Persists learnings from past competitions to inform future strategy.
Similar to AutoCompete's "knowledge from 100+ competitions" but stored as JSON.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

MEMORY_FILE = Path("~/.agentic-compete/memory.json").expanduser()


class CompetitionMemory:
    """
    Stores and retrieves learnings from past competitions.

    Memory entries include:
    - What worked: best model + features for similar data profiles
    - What didn't: model failures, poor strategies
    - Heuristics: data size → model recommendations
    """

    def __init__(self, path: Path = MEMORY_FILE):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.entries: list[dict[str, Any]] = self._load()

    def _load(self) -> list[dict]:
        if self.path.exists():
            try:
                return json.loads(self.path.read_text())
            except (json.JSONDecodeError, IOError):
                return []
        return []

    def _save(self) -> None:
        self.path.write_text(json.dumps(self.entries, indent=2, default=str))

    def record_competition(
        self,
        competition_slug: str,
        data_profile: dict,
        best_model: str,
        best_score: float,
        best_features: list[str],
        strategy_notes: str,
        all_results: list[dict],
    ) -> None:
        """Record results from a completed competition."""
        entry = {
            "competition": competition_slug,
            "data_profile": {
                "n_rows": data_profile.get("n_rows", 0),
                "n_cols": data_profile.get("n_cols", 0),
                "task_type": data_profile.get("task_type", ""),
                "has_text": bool(data_profile.get("text_cols")),
                "has_categoricals": bool(data_profile.get("categorical_cols")),
                "missing_rate": sum(data_profile.get("missing_pct", {}).values()) / max(len(data_profile.get("missing_pct", {})), 1),
            },
            "best_model": best_model,
            "best_score": best_score,
            "best_features": best_features[:20],
            "strategy_notes": strategy_notes,
            "top_results": sorted(all_results, key=lambda r: r.get("cv_score", 0), reverse=True)[:10],
        }
        self.entries.append(entry)
        self._save()
        logger.info(f"Recorded memory for {competition_slug}")

    def get_similar_competitions(self, data_profile: dict, top_k: int = 3) -> list[dict]:
        """Find past competitions with similar data profiles."""
        if not self.entries:
            return []

        def similarity(entry: dict) -> float:
            ep = entry.get("data_profile", {})
            score = 0.0
            # Task type match
            if ep.get("task_type") == data_profile.get("task_type"):
                score += 3.0
            # Size similarity (log scale)
            import math
            size_ratio = abs(math.log(max(ep.get("n_rows", 1), 1)) - math.log(max(data_profile.get("n_rows", 1), 1)))
            score += max(0, 3.0 - size_ratio)
            # Feature count similarity
            feat_ratio = abs(math.log(max(ep.get("n_cols", 1), 1)) - math.log(max(data_profile.get("n_cols", 1), 1)))
            score += max(0, 2.0 - feat_ratio)
            # Text presence
            if ep.get("has_text") == data_profile.get("has_text"):
                score += 1.0
            return score

        ranked = sorted(self.entries, key=similarity, reverse=True)
        return ranked[:top_k]

    def get_strategy_hints(self, data_profile: dict) -> str:
        """Get strategy hints based on similar past competitions."""
        similar = self.get_similar_competitions(data_profile)
        if not similar:
            return "No past competition data available. Using default strategy."

        hints = []
        for comp in similar:
            hints.append(
                f"- {comp['competition']}: Best model was {comp['best_model']} "
                f"(score: {comp['best_score']:.5f}). "
                f"Notes: {comp.get('strategy_notes', 'N/A')}"
            )
        return "Based on similar past competitions:\n" + "\n".join(hints)
