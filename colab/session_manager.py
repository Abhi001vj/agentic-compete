"""
Colab Session Manager
======================
Manages the connection to Google Colab via the official Colab MCP server.
Handles cell creation, execution, output parsing, and error recovery.

The official MCP server (github.com/googlecolab/colab-mcp) provides:
- add_code_cell: Add a Python code cell
- add_text_cell: Add a markdown cell
- execute_cell: Run a cell and get output
- move_cell: Rearrange cells
- delete_cell: Remove a cell

This module wraps those primitives into higher-level operations
suited for ML workflows.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class CellResult:
    """Result from executing a Colab cell."""
    cell_index: int
    success: bool
    output: str = ""
    error: str = ""
    execution_time_sec: float = 0.0
    output_images: list[str] = field(default_factory=list)  # base64 encoded
    output_data: dict[str, Any] = field(default_factory=dict)  # parsed structured output


class ColabSession:
    """
    High-level interface to a Colab notebook via MCP.

    Usage:
        session = ColabSession()
        result = session.execute_code("import pandas as pd; print('ok')")
        if result.success:
            print(result.output)
    """

    def __init__(self, mcp_client: Any = None, timeout: int = 300):
        """
        Args:
            mcp_client: An MCP client connected to the colab-mcp server.
                        If None, will attempt to create one from config.
            timeout: Default cell execution timeout in seconds.
        """
        self.mcp_client = mcp_client
        self.timeout = timeout
        self.cell_count = 0
        self._setup_complete = False

    async def _call_mcp(self, tool_name: str, params: dict) -> dict:
        """
        Call a Colab MCP tool. This is the low-level MCP interface.

        In production, this routes through the actual MCP client.
        For development/testing, this can be mocked.
        """
        if self.mcp_client is None:
            raise RuntimeError(
                "No MCP client configured. Ensure colab-mcp server is running "
                "and accessible. See README for setup instructions."
            )
        result = await self.mcp_client.call_tool(tool_name, params)
        return result

    # -------------------------------------------------------------------
    # High-level operations
    # -------------------------------------------------------------------

    async def setup_environment(self, competition_slug: str) -> CellResult:
        """
        Set up the Colab environment for a competition.
        Installs packages, mounts Drive, configures Kaggle API.
        """
        setup_code = f'''
# === AgenticCompete Environment Setup ===
import subprocess, os, sys

# Install core packages
packages = [
    "kaggle", "xgboost", "lightgbm", "catboost",
    "optuna", "scikit-learn", "pandas", "numpy",
    "matplotlib", "seaborn", "plotly",
    "shap", "eli5", "category_encoders",
]
for pkg in packages:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

# Configure Kaggle
os.environ["KAGGLE_USERNAME"] = os.getenv("KAGGLE_USERNAME", "")
os.environ["KAGGLE_KEY"] = os.getenv("KAGGLE_KEY", "")

# Download competition data
os.makedirs("/content/data", exist_ok=True)
os.makedirs("/content/models", exist_ok=True)
os.makedirs("/content/submissions", exist_ok=True)
os.makedirs("/content/plots", exist_ok=True)

try:
    subprocess.check_call([
        "kaggle", "competitions", "download", "-c", "{competition_slug}",
        "-p", "/content/data"
    ])
    # Unzip if needed
    import zipfile, glob
    for zf in glob.glob("/content/data/*.zip"):
        with zipfile.ZipFile(zf, 'r') as z:
            z.extractall("/content/data")
        os.remove(zf)
    print("DATA_READY: " + str(os.listdir("/content/data")))
except Exception as e:
    print(f"DATA_ERROR: {{e}}")

print("SETUP_COMPLETE")
'''
        result = await self.execute_code(setup_code, timeout=600)
        self._setup_complete = result.success
        return result

    async def execute_code(
        self,
        code: str,
        timeout: Optional[int] = None,
        add_markdown_header: Optional[str] = None,
    ) -> CellResult:
        """
        Add a code cell, execute it, and return parsed results.

        Args:
            code: Python code to execute
            timeout: Execution timeout in seconds
            add_markdown_header: If provided, add a markdown cell before code
        """
        timeout = timeout or self.timeout
        start = time.time()

        try:
            # Optionally add a markdown header cell
            if add_markdown_header:
                await self._call_mcp("add_text_cell", {
                    "cellIndex": self.cell_count,
                    "content": f"## {add_markdown_header}",
                })
                self.cell_count += 1

            # Add the code cell
            await self._call_mcp("add_code_cell", {
                "cellIndex": self.cell_count,
                "content": code,
            })
            code_cell_idx = self.cell_count
            self.cell_count += 1

            # Execute the cell
            exec_result = await self._call_mcp("execute_cell", {
                "cellIndex": code_cell_idx,
            })

            elapsed = time.time() - start

            # Parse the output
            output = self._parse_output(exec_result)
            error = self._parse_error(exec_result)

            return CellResult(
                cell_index=code_cell_idx,
                success=not bool(error),
                output=output,
                error=error,
                execution_time_sec=elapsed,
            )

        except Exception as e:
            elapsed = time.time() - start
            logger.error(f"Cell execution failed: {e}")
            return CellResult(
                cell_index=self.cell_count,
                success=False,
                error=str(e),
                execution_time_sec=elapsed,
            )

    async def execute_code_with_retry(
        self,
        code: str,
        max_retries: int = 3,
        fix_callback: Any = None,
    ) -> CellResult:
        """
        Execute code with automatic retry and optional LLM-based error fixing.

        Args:
            code: Python code to execute
            max_retries: Maximum retry attempts
            fix_callback: Async function(code, error) -> fixed_code
                          Typically calls the LLM to fix the bug
        """
        current_code = code
        for attempt in range(max_retries):
            result = await self.execute_code(current_code)
            if result.success:
                return result

            logger.warning(
                f"Attempt {attempt + 1}/{max_retries} failed: {result.error[:200]}"
            )

            if fix_callback and attempt < max_retries - 1:
                current_code = await fix_callback(current_code, result.error)

        return result  # Return last failed result

    async def execute_and_capture_json(self, code: str) -> dict[str, Any]:
        """
        Execute code that prints a JSON result on its last line.
        Commonly used for capturing metrics, data profiles, etc.
        """
        # Wrap code to ensure JSON output
        wrapped = code + "\n\nimport json as _json\n"
        result = await self.execute_code(wrapped)
        if not result.success:
            return {"error": result.error}
        try:
            # Find the last JSON-like line in output
            for line in reversed(result.output.strip().split("\n")):
                line = line.strip()
                if line.startswith("{") or line.startswith("["):
                    return json.loads(line)
            return {"raw_output": result.output}
        except json.JSONDecodeError:
            return {"raw_output": result.output}

    async def run_training_cell(
        self,
        training_code: str,
        model_name: str,
    ) -> CellResult:
        """
        Execute a training cell with proper markdown documentation.
        """
        return await self.execute_code(
            training_code,
            timeout=1800,  # 30 min for training
            add_markdown_header=f"Training: {model_name}",
        )

    async def save_and_download_file(self, remote_path: str, local_path: str) -> bool:
        """
        Download a file from Colab to local filesystem.
        Uses Colab's file download mechanism.
        """
        download_code = f'''
from google.colab import files
files.download("{remote_path}")
print("DOWNLOAD_READY")
'''
        result = await self.execute_code(download_code)
        return result.success

    # -------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------

    def _parse_output(self, exec_result: dict) -> str:
        """Extract text output from MCP execution result."""
        if isinstance(exec_result, dict):
            # Handle various MCP response formats
            if "content" in exec_result:
                contents = exec_result["content"]
                if isinstance(contents, list):
                    return "\n".join(
                        c.get("text", "") for c in contents
                        if c.get("type") == "text"
                    )
                return str(contents)
            return str(exec_result)
        return str(exec_result)

    def _parse_error(self, exec_result: dict) -> str:
        """Extract error information from MCP execution result."""
        if isinstance(exec_result, dict):
            if "error" in exec_result:
                return str(exec_result["error"])
            output = self._parse_output(exec_result)
            # Check for Python tracebacks in output
            if "Traceback" in output or "Error:" in output:
                # Extract traceback portion
                lines = output.split("\n")
                error_lines = []
                in_traceback = False
                for line in lines:
                    if "Traceback" in line:
                        in_traceback = True
                    if in_traceback:
                        error_lines.append(line)
                return "\n".join(error_lines) if error_lines else ""
        return ""

    def upload_data(self, competition_slug: str) -> None:
        """Synchronous wrapper for setup_environment."""
        asyncio.run(self.setup_environment(competition_slug))
