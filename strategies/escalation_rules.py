"""
Escalation Rules
==================
Defines when the agent should move from one tier to the next.
Uses both rule-based checks and LLM reasoning.
"""

from __future__ import annotations

from core.state import CompetitionState, ModelTier


class EscalationRules:
    """Rules for escalating between model tiers."""

    DEFAULT_PATIENCE = 3
    MIN_IMPROVEMENT_PCT = 0.001  # 0.1%
    MAX_FEATURE_ENG_ROUNDS = 3

    @classmethod
    def should_escalate(
        cls,
        state: CompetitionState,
        patience: int | None = None,
    ) -> bool:
        """Check if the current tier has been exhausted."""
        patience = patience or cls.DEFAULT_PATIENCE

        current_tier_results = [
            r for r in state.model_results if r.tier == state.current_tier
        ]

        if len(current_tier_results) < 2:
            return False

        # Check if recent models show no improvement
        recent = current_tier_results[-patience:]
        if len(recent) < patience:
            return False

        if state.metric_direction == "maximize":
            best_recent = max(r.cv_score for r in recent)
            overall_best = state.best_score
            improving = (best_recent - overall_best) > cls.MIN_IMPROVEMENT_PCT * abs(overall_best)
        else:
            best_recent = min(r.cv_score for r in recent)
            overall_best = state.best_score
            improving = (overall_best - best_recent) > cls.MIN_IMPROVEMENT_PCT * abs(overall_best)

        return not improving

    @classmethod
    def should_do_feature_engineering(cls, state: CompetitionState) -> bool:
        """Check if feature engineering should be attempted."""
        current_tier_fe = [
            fs for fs in state.feature_sets
        ]
        return len(current_tier_fe) < cls.MAX_FEATURE_ENG_ROUNDS

    @classmethod
    def get_next_tier(cls, current: ModelTier) -> ModelTier | None:
        """Get the next tier in the progression."""
        progression = [ModelTier.BASELINE, ModelTier.MEDIUM, ModelTier.COMPLEX, ModelTier.ENSEMBLE]
        idx = progression.index(current)
        if idx < len(progression) - 1:
            return progression[idx + 1]
        return None

    @classmethod
    def should_submit_early(cls, state: CompetitionState) -> bool:
        """Check if we should submit now (time pressure or good enough)."""
        import time
        from datetime import datetime

        # Time-based: if we've used > 80% of allocated time
        started = datetime.fromisoformat(state.started_at)
        elapsed_hours = (datetime.now() - started).total_seconds() / 3600
        if elapsed_hours > state.max_runtime_hours * 0.8:
            return True

        # Score-based: if we're at complex tier with good results
        if state.current_tier == ModelTier.COMPLEX and len(state.model_results) > 10:
            return True

        return False

    @classmethod
    def recommend_action(cls, state: CompetitionState) -> str:
        """
        High-level recommendation based on current state.
        Returns: "continue" | "escalate" | "feature_engineer" | "ensemble" | "submit"
        """
        if cls.should_submit_early(state):
            return "submit"

        if state.current_tier == ModelTier.ENSEMBLE:
            return "submit"

        if cls.should_escalate(state):
            next_tier = cls.get_next_tier(state.current_tier)
            if next_tier == ModelTier.ENSEMBLE:
                return "ensemble"
            return "escalate"

        if cls.should_do_feature_engineering(state):
            return "feature_engineer"

        return "continue"
