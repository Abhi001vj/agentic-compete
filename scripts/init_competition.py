#!/usr/bin/env python3
"""
Initialize a new competition workspace.
Creates the necessary directory structure and config.

Usage:
    python scripts/init_competition.py titanic --metric accuracy
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    parser = argparse.ArgumentParser(description="Initialize a new competition workspace")
    parser.add_argument("competition", help="Kaggle competition slug")
    parser.add_argument("--metric", default="", help="Evaluation metric")
    parser.add_argument("--workspace", default="./competitions", help="Base workspace dir")
    args = parser.parse_args()

    workspace = Path(args.workspace) / args.competition
    workspace.mkdir(parents=True, exist_ok=True)

    # Create subdirectories
    for subdir in ["data", "models", "submissions", "plots", "notebooks", "logs"]:
        (workspace / subdir).mkdir(exist_ok=True)

    # Create competition config
    config = {
        "competition_slug": args.competition,
        "metric": args.metric,
        "created_at": datetime.now().isoformat(),
        "status": "initialized",
        "notes": "",
    }

    config_path = workspace / "competition.json"
    config_path.write_text(json.dumps(config, indent=2))

    print(f"""
╔═══════════════════════════════════════════╗
║   Competition Workspace Initialized       ║
╚═══════════════════════════════════════════╝

  Competition: {args.competition}
  Metric: {args.metric or 'auto-detect'}
  Workspace: {workspace}

  Structure:
    {workspace}/
    ├── data/          # Downloaded competition data
    ├── models/        # Saved model artifacts
    ├── submissions/   # Generated submission files
    ├── plots/         # EDA and analysis plots
    ├── notebooks/     # Generated Colab notebooks
    ├── logs/          # Agent reasoning logs
    └── competition.json

  Next: python scripts/run_competition.py -c {args.competition} -m {args.metric or 'auto'}
""")


if __name__ == "__main__":
    main()
