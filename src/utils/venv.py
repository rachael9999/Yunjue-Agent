# Copyright (c) 2026 Yunjue Tech
# SPDX-License-Identifier: Apache-2.0
import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

# Path to isolated virtual environment for dynamic tools
ISOLATED_VENV_PATH = Path(".dynamic_tools_venv")

# Python binary file in the isolated virtual environment
ISOLATED_PYTHON_PATH = ISOLATED_VENV_PATH / "bin" / "python"


def ensure_isolated_venv_exists() -> None:
    """Ensure the isolated virtual environment exists, creating it if necessary."""
    if not ISOLATED_VENV_PATH.exists():
        logger.info(f"Creating isolated virtual environment at {ISOLATED_VENV_PATH}")
        subprocess.run(["uv", "venv", str(ISOLATED_VENV_PATH)], check=True)


# Automatically ensure the isolated virtual environment exists when this module is imported
ensure_isolated_venv_exists()
