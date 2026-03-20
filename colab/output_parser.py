"""
Colab Output Parser
====================
Extracts structured data from Colab cell execution outputs.
Handles JSON extraction, plot detection, error classification, and metric parsing.
"""

from __future__ import annotations

import json
import re
import logging
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class ParsedOutput:
    """Structured representation of a cell output."""
    json_data: Optional[dict[str, Any]] = None
    metrics: dict[str, float] = None
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    has_plot: bool = False
    plot_paths: list[str] = None
    raw_text: str = ""

    def __post_init__(self):
        if self.metrics is None:
            self.metrics = {}
        if self.plot_paths is None:
            self.plot_paths = []


class OutputParser:
    """Parses Colab cell outputs into structured data."""

    # Common error patterns
    ERROR_PATTERNS = {
        "oom": re.compile(r"(OutOfMemoryError|CUDA out of memory|ResourceExhausted)", re.I),
        "import": re.compile(r"(ModuleNotFoundError|ImportError|No module named)", re.I),
        "type": re.compile(r"(TypeError|ValueError|KeyError|IndexError)", re.I),
        "timeout": re.compile(r"(TimeoutError|DeadlineExceeded|timed out)", re.I),
        "data": re.compile(r"(FileNotFoundError|ParserError|EmptyDataError)", re.I),
        "sklearn": re.compile(r"(NotFittedError|ConvergenceWarning)", re.I),
    }

    # Metric extraction patterns
    METRIC_PATTERNS = {
        "accuracy": re.compile(r"accuracy[:\s]+([0-9]+\.?[0-9]*)", re.I),
        "auc": re.compile(r"(?:auc|roc_auc)[:\s]+([0-9]+\.?[0-9]*)", re.I),
        "f1": re.compile(r"f1[_\s]?(?:score)?[:\s]+([0-9]+\.?[0-9]*)", re.I),
        "rmse": re.compile(r"rmse[:\s]+([0-9]+\.?[0-9]*)", re.I),
        "mae": re.compile(r"mae[:\s]+([0-9]+\.?[0-9]*)", re.I),
        "log_loss": re.compile(r"log[_\s]?loss[:\s]+([0-9]+\.?[0-9]*)", re.I),
        "cv_mean": re.compile(r"cv[_\s]?mean[:\s]+([0-9]+\.?[0-9]*)", re.I),
        "cv_std": re.compile(r"cv[_\s]?std[:\s]+([0-9]+\.?[0-9]*)", re.I),
    }

    @classmethod
    def parse(cls, output: str) -> ParsedOutput:
        """Parse raw cell output into structured form."""
        result = ParsedOutput(raw_text=output)

        # Extract JSON
        result.json_data = cls._extract_json(output)

        # Extract metrics from JSON or raw text
        if result.json_data:
            result.metrics = cls._extract_metrics_from_json(result.json_data)
        if not result.metrics:
            result.metrics = cls._extract_metrics_from_text(output)

        # Detect errors
        error = cls._classify_error(output)
        if error:
            result.error_type, result.error_message = error

        # Detect plots
        result.has_plot = bool(re.search(r"savefig|\.png|\.jpg|plot saved", output, re.I))
        result.plot_paths = re.findall(r'/content/plots/[\w.-]+\.(?:png|jpg|svg)', output)

        return result

    @classmethod
    def _extract_json(cls, output: str) -> Optional[dict]:
        """Find and parse the last JSON object in the output."""
        for line in reversed(output.strip().split("\n")):
            line = line.strip()
            if line.startswith("{"):
                try:
                    return json.loads(line)
                except json.JSONDecodeError:
                    continue
            # Also try to find JSON in lines with prefix
            json_match = re.search(r'(\{[^{}]+\})', line)
            if json_match:
                try:
                    return json.loads(json_match.group(1))
                except json.JSONDecodeError:
                    continue
        return None

    @classmethod
    def _extract_metrics_from_json(cls, data: dict) -> dict[str, float]:
        """Extract numeric metrics from a JSON dict."""
        metrics = {}
        metric_keys = [
            "cv_mean", "cv_std", "accuracy", "auc", "roc_auc", "f1",
            "rmse", "mae", "log_loss", "r2", "best_score", "score",
            "training_time", "cv_scores",
        ]
        for key in metric_keys:
            if key in data:
                val = data[key]
                if isinstance(val, (int, float)):
                    metrics[key] = float(val)
                elif isinstance(val, list) and val:
                    metrics[f"{key}_mean"] = float(sum(val) / len(val))
        return metrics

    @classmethod
    def _extract_metrics_from_text(cls, output: str) -> dict[str, float]:
        """Extract metrics from raw text using regex patterns."""
        metrics = {}
        for name, pattern in cls.METRIC_PATTERNS.items():
            match = pattern.search(output)
            if match:
                try:
                    metrics[name] = float(match.group(1))
                except ValueError:
                    continue
        return metrics

    @classmethod
    def _classify_error(cls, output: str) -> Optional[tuple[str, str]]:
        """Classify error type from output text."""
        # Check for Python traceback
        if "Traceback" not in output and "Error" not in output:
            return None

        for error_type, pattern in cls.ERROR_PATTERNS.items():
            match = pattern.search(output)
            if match:
                # Extract error message (last line of traceback)
                lines = output.strip().split("\n")
                error_lines = [l for l in lines if "Error" in l or "Exception" in l]
                message = error_lines[-1] if error_lines else match.group(0)
                return (error_type, message)

        # Generic error
        lines = output.strip().split("\n")
        error_lines = [l for l in lines if "Error" in l]
        if error_lines:
            return ("unknown", error_lines[-1])

        return None

    @classmethod
    def extract_data_ready_signal(cls, output: str) -> Optional[list[str]]:
        """Check if data download completed and return file list."""
        match = re.search(r"DATA_READY:\s*\[(.+)\]", output)
        if match:
            try:
                files = json.loads(f"[{match.group(1)}]")
                return files
            except json.JSONDecodeError:
                return match.group(1).split(", ")
        return None

    @classmethod
    def is_setup_complete(cls, output: str) -> bool:
        """Check if environment setup completed."""
        return "SETUP_COMPLETE" in output

    @classmethod
    def is_submission_ready(cls, output: str) -> bool:
        """Check if submission file was created."""
        return "SUBMISSION_READY" in output
