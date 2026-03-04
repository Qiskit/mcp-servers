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


# Qiskit documentation bases (configurable via environment variables)
QISKIT_DOCS_BASE = os.getenv("QISKIT_DOCS_BASE", "https://quantum.cloud.ibm.com/docs/")
QISKIT_SDK_DOCS = os.getenv("QISKIT_SDK_DOCS", "https://quantum.cloud.ibm.com/docs/")
BASE_URL = os.getenv("QISKIT_SEARCH_BASE_URL", "https://quantum.cloud.ibm.com/")

# Error code registry path
ERROR_CODES_PATH = "errors"

# HTTP timeout configuration (in seconds)
HTTP_TIMEOUT = _get_env_float("QISKIT_HTTP_TIMEOUT", 10.0)

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
