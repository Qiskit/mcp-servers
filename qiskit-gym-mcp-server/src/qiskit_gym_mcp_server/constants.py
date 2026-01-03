# This code is part of Qiskit.
#
# (C) Copyright IBM 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Configuration constants and defaults for the qiskit-gym MCP server."""

import logging
import os
from pathlib import Path

from dotenv import load_dotenv


# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)

# ============================================================================
# Model Storage Configuration
# ============================================================================

# Directory for storing trained models
QISKIT_GYM_MODEL_DIR = os.getenv(
    "QISKIT_GYM_MODEL_DIR", str(Path.home() / ".qiskit-gym" / "models")
)

# Directory for TensorBoard logs
QISKIT_GYM_TENSORBOARD_DIR = os.getenv(
    "QISKIT_GYM_TENSORBOARD_DIR", str(Path.home() / ".qiskit-gym" / "runs")
)

# ============================================================================
# Training Limits
# ============================================================================

# Maximum training iterations (optional limit)
# Default is 0 (no limit) - set via environment variable to enforce a limit
# Example: QISKIT_GYM_MAX_ITERATIONS=10000 to limit training runs
QISKIT_GYM_MAX_ITERATIONS = int(os.getenv("QISKIT_GYM_MAX_ITERATIONS", "0"))

# Maximum number of qubits for environments
QISKIT_GYM_MAX_QUBITS = int(os.getenv("QISKIT_GYM_MAX_QUBITS", "15"))

# ============================================================================
# Synthesis Limits
# ============================================================================

# Maximum search attempts for synthesis
QISKIT_GYM_MAX_SEARCHES = int(os.getenv("QISKIT_GYM_MAX_SEARCHES", "10000"))

# Synthesis timeout in seconds
QISKIT_GYM_SYNTHESIS_TIMEOUT = int(os.getenv("QISKIT_GYM_SYNTHESIS_TIMEOUT", "300"))

# ============================================================================
# Debug Configuration
# ============================================================================

# Debug level for logging
QISKIT_GYM_DEBUG_LEVEL = os.getenv("QISKIT_GYM_DEBUG_LEVEL", "INFO")

# ============================================================================
# Validation
# ============================================================================


def validate_configuration() -> bool:
    """Validate configuration values and log any issues.

    Returns:
        True if all configuration is valid, False otherwise.
    """
    valid = True

    # Validate model directory
    model_dir = Path(QISKIT_GYM_MODEL_DIR)
    if not model_dir.exists():
        try:
            model_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created model directory: {model_dir}")
        except Exception as e:
            logger.warning(f"Could not create model directory {model_dir}: {e}")

    # Validate TensorBoard directory
    tb_dir = Path(QISKIT_GYM_TENSORBOARD_DIR)
    if not tb_dir.exists():
        try:
            tb_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created TensorBoard directory: {tb_dir}")
        except Exception as e:
            logger.warning(f"Could not create TensorBoard directory {tb_dir}: {e}")

    # Validate numeric limits
    if QISKIT_GYM_MAX_ITERATIONS < 0:
        logger.error("QISKIT_GYM_MAX_ITERATIONS must be >= 0 (0 means no limit)")
        valid = False

    if QISKIT_GYM_MAX_QUBITS < 2:
        logger.error("QISKIT_GYM_MAX_QUBITS must be at least 2")
        valid = False

    if QISKIT_GYM_MAX_SEARCHES < 1:
        logger.error("QISKIT_GYM_MAX_SEARCHES must be at least 1")
        valid = False

    return valid


# Validate on import
if not validate_configuration():
    logger.warning("Configuration validation failed. Some features may not work correctly.")
