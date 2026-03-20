#!/usr/bin/env python3
"""
AgenticCompete CLI
===================
Command-line interface to run AgenticCompete on a Kaggle competition.

Usage:
    python scripts/run_competition.py --competition titanic --metric accuracy --max-hours 4
    python scripts/run_competition.py --competition house-prices-advanced-regression-techniques --metric rmse --max-hours 6
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.orchestrator import run_competition
from core.state import CompetitionState


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def main():
    parser = argparse.ArgumentParser(
        description="AgenticCompete — Agentic Kaggle Competition Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run on Titanic (classification)
  python run_competition.py --competition titanic --metric accuracy

  # Run on House Prices (regression)  
  python run_competition.py --competition house-prices-advanced-regression-techniques --metric rmse

  # Run with time limit
  python run_competition.py --competition titanic --metric accuracy --max-hours 2

  # Dry run (plan only, no execution)
  python run_competition.py --competition titanic --metric accuracy --dry-run
        """,
    )

    parser.add_argument(
        "--competition", "-c",
        required=True,
        help="Kaggle competition slug (e.g., 'titanic')",
    )
    parser.add_argument(
        "--metric", "-m",
        default="",
        help="Evaluation metric (auc, accuracy, rmse, f1, etc.)",
    )
    parser.add_argument(
        "--max-hours",
        type=float,
        default=8.0,
        help="Maximum runtime in hours (default: 8)",
    )
    parser.add_argument(
        "--description", "-d",
        default="",
        help="Optional competition description for the agent",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only analyze and plan, don't execute",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="./output",
        help="Directory for output files (default: ./output)",
    )

    args = parser.parse_args()
    setup_logging(args.verbose)
    logger = logging.getLogger("agenticcompete")

    # Print banner
    print("""
╔═══════════════════════════════════════════════════════╗
║            🏆 AgenticCompete v0.1.0 🏆               ║
║     Agentic Kaggle Competition Framework              ║
║     Powered by Claude + Google Colab MCP              ║
╚═══════════════════════════════════════════════════════╝
    """)

    logger.info(f"Competition: {args.competition}")
    logger.info(f"Metric: {args.metric or 'auto-detect'}")
    logger.info(f"Max runtime: {args.max_hours} hours")
    logger.info(f"Mode: {'DRY RUN' if args.dry_run else 'FULL RUN'}")

    try:
        final_state = run_competition(
            competition_slug=args.competition,
            description=args.description,
            metric=args.metric,
            max_hours=args.max_hours,
        )

        # Save results
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = output_dir / f"{args.competition}_{timestamp}_results.json"

        results = {
            "competition": args.competition,
            "metric": final_state.evaluation_metric,
            "best_score": final_state.best_score,
            "best_model": final_state.best_model_id,
            "n_models_trained": len(final_state.model_results),
            "n_submissions": len(final_state.submissions),
            "results_summary": final_state.get_results_summary(),
            "reasoning_log": final_state.reasoning_log,
            "improvement_history": final_state.improvement_history,
        }

        with open(results_file, "w") as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"\nResults saved to: {results_file}")
        print("\n" + final_state.get_results_summary())

    except KeyboardInterrupt:
        logger.info("\nInterrupted by user. Saving partial results...")
    except Exception as e:
        logger.error(f"Competition run failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
