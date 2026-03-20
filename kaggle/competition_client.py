"""
Kaggle Competition Client
==========================
Wrapper around the Kaggle API for data download, submission, and LB tracking.
"""

from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DATA_DIR = Path("./data")


class KaggleClient:
    """Simplified Kaggle API wrapper."""

    def __init__(self):
        self._verify_credentials()

    def _verify_credentials(self) -> None:
        username = os.environ.get("KAGGLE_USERNAME", "")
        key = os.environ.get("KAGGLE_KEY", "")
        if not username or not key:
            kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
            if not kaggle_json.exists():
                logger.warning(
                    "Kaggle credentials not found. Set KAGGLE_USERNAME and KAGGLE_KEY "
                    "environment variables, or place kaggle.json in ~/.kaggle/"
                )

    def download_competition_data(self, competition_slug: str, dest: str | Path = DATA_DIR) -> Path:
        """Download competition data."""
        dest = Path(dest)
        dest.mkdir(parents=True, exist_ok=True)

        cmd = ["kaggle", "competitions", "download", "-c", competition_slug, "-p", str(dest)]
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            logger.error(f"Download failed: {result.stderr}")
            raise RuntimeError(f"Failed to download data: {result.stderr}")

        # Unzip
        import zipfile, glob
        for zf in glob.glob(str(dest / "*.zip")):
            with zipfile.ZipFile(zf, "r") as z:
                z.extractall(dest)
            os.remove(zf)

        logger.info(f"Data downloaded to {dest}: {os.listdir(dest)}")
        return dest

    def submit(self, competition_slug: str, file_path: str, message: str = "") -> dict[str, Any]:
        """Submit predictions to Kaggle."""
        cmd = [
            "kaggle", "competitions", "submit",
            "-c", competition_slug,
            "-f", file_path,
            "-m", message or "AgenticCompete submission",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }

    def get_leaderboard(self, competition_slug: str) -> list[dict]:
        """Get current leaderboard."""
        try:
            import kaggle
            api = kaggle.KaggleApi()
            api.authenticate()
            lb = api.competition_leaderboard_view(competition_slug)
            return [{"rank": e.rank, "team": e.teamName, "score": e.score} for e in lb[:20]]
        except Exception as e:
            logger.warning(f"Failed to fetch leaderboard: {e}")
            return []
