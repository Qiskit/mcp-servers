# This code is part of Qiskit.
#
# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import logging
import os


logger = logging.getLogger(__name__)


def _get_env_float(name: str, default: float) -> float:
    """
    Get environment variable as float with fallback to default.

    Args:
        name: Environment variable name
        default: Default value if not set or invalid

    Returns:
        Float value from environment or default
    """
    try:
        value = os.getenv(name)
        if value is None:
            return default
        return float(value)
    except (ValueError, TypeError):
        logger.warning(f"Invalid {name} value: {os.getenv(name)}, using default {default}")
        return default


# Qiskit documentation base URL (configurable via environment variable)
QISKIT_DOCS_BASE = os.getenv("QISKIT_DOCS_BASE", "https://quantum.cloud.ibm.com/docs/")
BASE_URL = os.getenv("QISKIT_SEARCH_BASE_URL", "https://quantum.cloud.ibm.com/")

# Error code registry
ERROR_CODE_CATEGORIES = {
    "1XXX": "Validation, transpilation, backend availability, authorization, and job management",
    "2XXX": "Backend configuration, booking, and data retrieval",
    "3XXX": "Job handling, authentication, and analytics",
    "4XXX": "Session management and job limits",
    "5XXX": "Job timeout and cancellation",
    "6XXX": "Shot limits, compiler input, and control system",
    "7XXX": "Instruction and basis gate compatibility",
    "8XXX": "Pulse and channel configuration",
    "9XXX": "Hardware loading and internal errors",
}

# HTTP timeout configuration (in seconds)
HTTP_TIMEOUT = _get_env_float("QISKIT_HTTP_TIMEOUT", 10.0)

# Cache configuration
CACHE_TTL = _get_env_float("QISKIT_DOCS_CACHE_TTL", 3600.0)  # 1 hour default
CACHE_MAXSIZE = int(_get_env_float("QISKIT_DOCS_CACHE_MAXSIZE", 128.0))
CACHE_DIR = os.getenv("QISKIT_DOCS_CACHE_DIR", "")  # empty = disk cache disabled
OFFLINE_MODE = os.getenv("QISKIT_DOCS_OFFLINE", "").lower() in ("1", "true", "yes")

# Qiskit modules and their documentation paths
AVAILABLE_MODULES = [
    # Circuit construction
    "circuit",
    # Quantum information
    "quantum_info",
    # Transpilation
    "transpiler",
    "synthesis",
    "dagcircuit",
    "passmanager",
    "converters",
    "compiler",
    # Primitives and providers
    "primitives",
    "providers",
    # Results and visualization
    "result",
    "visualization",
    # Serialization
    "qasm2",
    "qasm3",
    "qpy",
    # Utilities
    "utils",
    "exceptions",
]

AVAILABLE_ADDONS = [
    "aqc-tensor",
    "cutting",
    "mpf",
    "obp",
    "sqd",
    "utils",
]

AVAILABLE_GUIDES = [
    # Getting started
    "quick-start",
    # Circuit building
    "construct-circuits",
    # Transpilation
    "transpile",
    "transpiler-stages",
    "transpile-with-pass-managers",
    "defaults-and-configuration-options",
    "circuit-transpilation-settings",
    "qiskit-transpiler-service",
    # Error mitigation and suppression
    "error-mitigation-and-suppression-techniques",
    "configure-error-mitigation",
    "configure-error-suppression",
    # Execution
    "primitives",
    "execution-modes",
    "runtime-options-overview",
    "directed-execution-model",
    # Dynamic circuits
    "dynamic-circuits",
    # Post-processing addons
    "qiskit-addons-sqd",
    # Qiskit Functions - circuit functions
    "functions",
    "ibm-circuit-function",
    "algorithmiq-tem",
    "qedma-qesem",
    "q-ctrl-performance-management",
    # Qiskit Functions - application functions
    "colibritd-pde",
    "global-data-quantum-optimizer",
    "qunova-chemistry",
    "kipu-optimization",
    "q-ctrl-optimization-solver",
    "multiverse-computing-singularity",
    # Security and support
    "secure-data",
    "support",
]

SEARCH_PATH = "endpoints-docs-learning/api/search"
