"""
Tests for AgenticCompete core components.
"""

import json
import pytest

from core.state import CompetitionState, DataProfile, ModelResult, ModelTier, Phase


class TestCompetitionState:
    """Test the state management."""

    def test_initial_state(self):
        state = CompetitionState(competition_slug="titanic")
        assert state.phase == Phase.INIT
        assert state.current_tier == ModelTier.BASELINE
        assert state.best_score == 0.0
        assert len(state.model_results) == 0

    def test_add_result_tracks_best(self):
        state = CompetitionState(
            competition_slug="titanic",
            metric_direction="maximize",
        )

        r1 = ModelResult(model_name="LR", tier=ModelTier.BASELINE, cv_score=0.78)
        state.add_result(r1)
        assert state.best_score == 0.78
        assert state.best_model_id == r1.model_id

        r2 = ModelResult(model_name="RF", tier=ModelTier.BASELINE, cv_score=0.85)
        state.add_result(r2)
        assert state.best_score == 0.85
        assert state.best_model_id == r2.model_id

        # Worse model shouldn't update best
        r3 = ModelResult(model_name="SVM", tier=ModelTier.BASELINE, cv_score=0.80)
        state.add_result(r3)
        assert state.best_score == 0.85
        assert state.best_model_id == r2.model_id

    def test_add_result_minimize(self):
        state = CompetitionState(
            competition_slug="house-prices",
            metric_direction="minimize",
        )

        r1 = ModelResult(model_name="RF", tier=ModelTier.BASELINE, cv_score=0.15)
        state.add_result(r1)
        assert state.best_score == 0.15

        r2 = ModelResult(model_name="XGB", tier=ModelTier.BASELINE, cv_score=0.12)
        state.add_result(r2)
        assert state.best_score == 0.12
        assert state.best_model_id == r2.model_id

    def test_should_escalate(self):
        state = CompetitionState(metric_direction="maximize")
        state.best_score = 0.85

        # Add 3 models that don't improve
        for i in range(3):
            state.model_results.append(
                ModelResult(model_name=f"model_{i}", cv_score=0.85)
            )

        assert state.should_escalate(patience=3) is True

    def test_should_not_escalate_with_improvement(self):
        state = CompetitionState(metric_direction="maximize")
        state.best_score = 0.87

        state.model_results = [
            ModelResult(model_name="m1", cv_score=0.85),
            ModelResult(model_name="m2", cv_score=0.86),
            ModelResult(model_name="m3", cv_score=0.87),
        ]

        assert state.should_escalate(patience=3) is False

    def test_results_summary(self):
        state = CompetitionState(
            competition_slug="titanic",
            metric_direction="maximize",
        )
        r = ModelResult(model_name="XGBoost", tier=ModelTier.BASELINE, cv_score=0.85, cv_std=0.02)
        state.add_result(r)

        summary = state.get_results_summary()
        assert "XGBoost" in summary
        assert "0.85" in summary

    def test_reasoning_log(self):
        state = CompetitionState()
        state.log_reasoning("eda_agent", "Found 20% missing values in feature X")
        assert len(state.reasoning_log) == 1
        assert state.reasoning_log[0]["agent"] == "eda_agent"


class TestDataProfile:
    def test_default_profile(self):
        profile = DataProfile()
        assert profile.n_rows == 0
        assert profile.numeric_cols == []
        assert profile.key_insights == []


class TestEscalationRules:
    def test_next_tier(self):
        from strategies.escalation_rules import EscalationRules

        assert EscalationRules.get_next_tier(ModelTier.BASELINE) == ModelTier.MEDIUM
        assert EscalationRules.get_next_tier(ModelTier.MEDIUM) == ModelTier.COMPLEX
        assert EscalationRules.get_next_tier(ModelTier.COMPLEX) == ModelTier.ENSEMBLE
        assert EscalationRules.get_next_tier(ModelTier.ENSEMBLE) is None


class TestOutputParser:
    def test_extract_json(self):
        from colab.output_parser import OutputParser

        output = 'Some text\n{"cv_mean": 0.85, "cv_std": 0.02}\n'
        result = OutputParser.parse(output)
        assert result.json_data is not None
        assert result.json_data["cv_mean"] == 0.85

    def test_classify_oom_error(self):
        from colab.output_parser import OutputParser

        output = "Traceback...\nRuntimeError: CUDA out of memory"
        result = OutputParser.parse(output)
        assert result.error_type == "oom"

    def test_classify_import_error(self):
        from colab.output_parser import OutputParser

        output = "Traceback...\nModuleNotFoundError: No module named 'xgboost'"
        result = OutputParser.parse(output)
        assert result.error_type == "import"

    def test_detect_data_ready(self):
        from colab.output_parser import OutputParser

        output = "DATA_READY: ['train.csv', 'test.csv']"
        files = OutputParser.extract_data_ready_signal(output)
        assert files is not None

    def test_metrics_from_text(self):
        from colab.output_parser import OutputParser

        output = "Accuracy: 0.856\nAUC: 0.912"
        result = OutputParser.parse(output)
        assert "accuracy" in result.metrics
        assert abs(result.metrics["accuracy"] - 0.856) < 1e-6
