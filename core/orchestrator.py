"""
AgenticCompete Orchestrator
============================
LangGraph state machine that drives the entire competition workflow.
Each node is an agent that reads/writes to CompetitionState.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Literal

from langgraph.graph import END, StateGraph

from core.state import CompetitionState, ModelTier, Phase

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Node functions — each takes state, calls the relevant agent, returns updates
# ---------------------------------------------------------------------------

def analyze_competition(state: CompetitionState) -> dict[str, Any]:
    """
    Node: Understand the competition.
    - Parse competition page / description
    - Identify metric, task type, data format
    - Note any special rules or constraints
    """
    from agents.competition_analyzer import CompetitionAnalyzer

    analyzer = CompetitionAnalyzer()
    analysis = analyzer.analyze(state.competition_slug, state.competition_description)

    state.log_reasoning("competition_analyzer", analysis["reasoning"])
    return {
        "competition_name": analysis["name"],
        "evaluation_metric": analysis["metric"],
        "metric_direction": analysis["metric_direction"],
        "competition_description": analysis["description"],
        "phase": Phase.DOWNLOAD,
    }


def download_data(state: CompetitionState) -> dict[str, Any]:
    """
    Node: Download competition data via Kaggle API and upload to Colab.
    """
    from kaggle.competition_client import KaggleClient

    client = KaggleClient()
    client.download_competition_data(state.competition_slug)

    # Upload to Colab Drive via MCP
    from colab.session_manager import ColabSession

    session = ColabSession()
    session.upload_data(state.competition_slug)

    state.log_reasoning("download", f"Downloaded data for {state.competition_slug}")
    return {"phase": Phase.EDA}


def run_eda(state: CompetitionState) -> dict[str, Any]:
    """
    Node: Perform comprehensive EDA in Colab.
    The EDA agent generates code, sends it to Colab via MCP,
    parses results, and builds a DataProfile.
    """
    from agents.eda_agent import EDAAgent

    agent = EDAAgent()
    profile = agent.run(state)

    state.log_reasoning("eda_agent", f"EDA complete. Key insights: {profile.key_insights}")
    return {
        "data_profile": profile,
        "phase": Phase.PLAN,
    }


def plan_strategy(state: CompetitionState) -> dict[str, Any]:
    """
    Node: Create a progressive modeling strategy based on EDA findings.
    The LLM analyzes the data profile and creates a tiered plan.
    """
    from core.decision_engine import DecisionEngine

    engine = DecisionEngine()
    plan = engine.create_strategy(state)

    state.log_reasoning("strategy_planner", plan["reasoning"])
    return {
        "strategy_plan": plan["strategy_text"],
        "baseline_models_planned": plan["baseline_models"],
        "medium_models_planned": plan["medium_models"],
        "complex_models_planned": plan["complex_models"],
        "phase": Phase.BASELINE,
        "current_tier": ModelTier.BASELINE,
    }


def run_baseline_models(state: CompetitionState) -> dict[str, Any]:
    """
    Node: Train baseline models with minimal tuning.
    Quick-and-dirty models to establish a performance floor.
    """
    from agents.model_agent import ModelAgent

    agent = ModelAgent()
    results = agent.train_tier(
        state=state,
        tier=ModelTier.BASELINE,
        models=state.baseline_models_planned,
    )

    for r in results:
        state.add_result(r)

    state.log_reasoning(
        "model_agent",
        f"Baseline complete. Best: {state.best_score:.5f}"
    )
    return {
        "phase": Phase.EVALUATE_BASELINE,
        "iteration": state.iteration + 1,
    }


def evaluate_and_decide_baseline(state: CompetitionState) -> dict[str, Any]:
    """
    Node: Evaluate baseline results and decide next step.
    """
    from core.decision_engine import DecisionEngine

    engine = DecisionEngine()
    decision = engine.evaluate_tier(state, ModelTier.BASELINE)

    state.log_reasoning("decision_engine", decision["reasoning"])

    if decision["action"] == "extract_max":
        # Baseline is promising — squeeze more out via feature eng + tuning
        return {"phase": Phase.FEATURE_ENG}
    elif decision["action"] == "escalate":
        # Baseline insufficient — try medium models
        return {"phase": Phase.MEDIUM_MODELS, "current_tier": ModelTier.MEDIUM}
    else:
        # Already good enough for submission
        return {"phase": Phase.SUBMISSION}


def run_feature_engineering(state: CompetitionState) -> dict[str, Any]:
    """
    Node: Engineer features based on EDA insights and current tier results.
    """
    from agents.feature_agent import FeatureAgent

    agent = FeatureAgent()
    feature_set = agent.engineer_features(state)

    state.feature_sets.append(feature_set)
    state.current_feature_set = feature_set.name

    state.log_reasoning(
        "feature_agent",
        f"Created feature set '{feature_set.name}' with {feature_set.n_features} features"
    )

    # Re-run current tier's best model with new features
    from agents.model_agent import ModelAgent
    model_agent = ModelAgent()
    best_model_name = next(
        (r.model_name for r in state.model_results if r.model_id == state.best_model_id),
        state.baseline_models_planned[0]
    )
    results = model_agent.train_tier(
        state=state,
        tier=state.current_tier,
        models=[best_model_name],
        feature_set=feature_set.name,
    )
    for r in results:
        state.add_result(r)

    return {"iteration": state.iteration + 1}


def run_medium_models(state: CompetitionState) -> dict[str, Any]:
    """
    Node: Train medium-complexity models (tuned GBMs, etc.)
    """
    from agents.model_agent import ModelAgent

    agent = ModelAgent()
    results = agent.train_tier(
        state=state,
        tier=ModelTier.MEDIUM,
        models=state.medium_models_planned,
    )
    for r in results:
        state.add_result(r)

    state.log_reasoning(
        "model_agent",
        f"Medium models complete. Best: {state.best_score:.5f}"
    )
    return {"phase": Phase.EVALUATE_MEDIUM, "iteration": state.iteration + 1}


def evaluate_and_decide_medium(state: CompetitionState) -> dict[str, Any]:
    from core.decision_engine import DecisionEngine

    engine = DecisionEngine()
    decision = engine.evaluate_tier(state, ModelTier.MEDIUM)
    state.log_reasoning("decision_engine", decision["reasoning"])

    if decision["action"] == "extract_max":
        return {"phase": Phase.FEATURE_ENG}
    elif decision["action"] == "escalate":
        return {"phase": Phase.COMPLEX_MODELS, "current_tier": ModelTier.COMPLEX}
    else:
        return {"phase": Phase.ENSEMBLE}


def run_complex_models(state: CompetitionState) -> dict[str, Any]:
    from agents.model_agent import ModelAgent

    agent = ModelAgent()
    results = agent.train_tier(
        state=state,
        tier=ModelTier.COMPLEX,
        models=state.complex_models_planned,
    )
    for r in results:
        state.add_result(r)

    state.log_reasoning(
        "model_agent",
        f"Complex models complete. Best: {state.best_score:.5f}"
    )
    return {"phase": Phase.ENSEMBLE, "iteration": state.iteration + 1}


def run_ensemble(state: CompetitionState) -> dict[str, Any]:
    """
    Node: Create ensemble from top models.
    """
    from agents.ensemble_agent import EnsembleAgent

    agent = EnsembleAgent()
    result = agent.create_ensemble(state)

    state.add_result(result)
    state.log_reasoning(
        "ensemble_agent",
        f"Ensemble score: {result.cv_score:.5f}"
    )
    return {"phase": Phase.SUBMISSION, "iteration": state.iteration + 1}


def submit(state: CompetitionState) -> dict[str, Any]:
    """
    Node: Generate predictions and submit to Kaggle.
    """
    from agents.submission_agent import SubmissionAgent

    agent = SubmissionAgent()
    submission = agent.submit(state)

    state.submissions.append(submission)
    state.log_reasoning(
        "submission_agent",
        f"Submitted. LB score: {submission.get('lb_score', 'pending')}"
    )
    return {"phase": Phase.DONE}


# ---------------------------------------------------------------------------
# Routing functions
# ---------------------------------------------------------------------------

def route_after_baseline(state: CompetitionState) -> Literal["feature_eng", "medium", "submit"]:
    if state.phase == Phase.FEATURE_ENG:
        return "feature_eng"
    elif state.phase == Phase.MEDIUM_MODELS:
        return "medium"
    else:
        return "submit"


def route_after_medium(state: CompetitionState) -> Literal["feature_eng", "complex", "ensemble"]:
    if state.phase == Phase.FEATURE_ENG:
        return "feature_eng"
    elif state.phase == Phase.COMPLEX_MODELS:
        return "complex"
    else:
        return "ensemble"


def route_after_feature_eng(state: CompetitionState) -> Literal["baseline_eval", "medium_eval", "ensemble"]:
    """After feature engineering, re-evaluate the current tier."""
    if state.current_tier == ModelTier.BASELINE:
        return "baseline_eval"
    elif state.current_tier == ModelTier.MEDIUM:
        return "medium_eval"
    else:
        return "ensemble"


# ---------------------------------------------------------------------------
# Build the graph
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    """
    Build the full AgenticCompete LangGraph workflow.
    """
    graph = StateGraph(CompetitionState)

    # Add all nodes
    graph.add_node("analyze", analyze_competition)
    graph.add_node("download", download_data)
    graph.add_node("eda", run_eda)
    graph.add_node("plan", plan_strategy)
    graph.add_node("baseline", run_baseline_models)
    graph.add_node("baseline_eval", evaluate_and_decide_baseline)
    graph.add_node("feature_eng", run_feature_engineering)
    graph.add_node("medium", run_medium_models)
    graph.add_node("medium_eval", evaluate_and_decide_medium)
    graph.add_node("complex", run_complex_models)
    graph.add_node("ensemble", run_ensemble)
    graph.add_node("submit", submit)

    # Linear edges
    graph.set_entry_point("analyze")
    graph.add_edge("analyze", "download")
    graph.add_edge("download", "eda")
    graph.add_edge("eda", "plan")
    graph.add_edge("plan", "baseline")
    graph.add_edge("baseline", "baseline_eval")

    # Conditional edges after baseline evaluation
    graph.add_conditional_edges(
        "baseline_eval",
        route_after_baseline,
        {"feature_eng": "feature_eng", "medium": "medium", "submit": "submit"},
    )

    # After feature engineering, re-evaluate
    graph.add_conditional_edges(
        "feature_eng",
        route_after_feature_eng,
        {"baseline_eval": "baseline_eval", "medium_eval": "medium_eval", "ensemble": "ensemble"},
    )

    # Medium model flow
    graph.add_edge("medium", "medium_eval")
    graph.add_conditional_edges(
        "medium_eval",
        route_after_medium,
        {"feature_eng": "feature_eng", "complex": "complex", "ensemble": "ensemble"},
    )

    # Complex → ensemble → submit
    graph.add_edge("complex", "ensemble")
    graph.add_edge("ensemble", "submit")
    graph.add_edge("submit", END)

    return graph.compile()


def run_competition(
    competition_slug: str,
    description: str = "",
    metric: str = "",
    max_hours: float = 8.0,
) -> CompetitionState:
    """
    Main entry point: run the full AgenticCompete pipeline.
    """
    initial_state = CompetitionState(
        competition_slug=competition_slug,
        competition_description=description,
        evaluation_metric=metric,
        max_runtime_hours=max_hours,
    )

    graph = build_graph()
    final_state = graph.invoke(initial_state)

    logger.info("Competition run complete!")
    logger.info(final_state.get_results_summary())

    return final_state
