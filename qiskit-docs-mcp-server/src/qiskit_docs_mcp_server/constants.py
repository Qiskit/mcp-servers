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
CACHE_TTL = _get_env_float("QISKIT_DOCS_CACHE_TTL", 3600.0)
SEARCH_CACHE_TTL = _get_env_float("QISKIT_SEARCH_CACHE_TTL", 300.0)  # 5 min default

# Qiskit modules and their documentation paths
AVAILABLE_MODULES = {
    # Circuit construction
    "circuit": "Quantum circuit construction and manipulation (QuantumCircuit, gates, registers)",
    # Quantum information
    "quantum_info": "Quantum information utilities (states, operators, channels, measures)",
    # Transpilation
    "transpiler": "Circuit transpilation and optimization for target hardware",
    "synthesis": "Circuit synthesis algorithms (unitary, Clifford, linear functions)",
    "dagcircuit": "Directed acyclic graph (DAG) representation of quantum circuits",
    "passmanager": "Transpiler pass manager framework for custom transpilation pipelines",
    "converters": "Circuit format converters and interoperability utilities",
    "compiler": "High-level compilation routines (transpile shortcut)",
    # Primitives and providers
    "primitives": "Sampler and Estimator primitives for quantum execution",
    "providers": "Backend providers and job management interfaces",
    # Results and visualization
    "result": "Quantum job result handling and analysis",
    "visualization": "Circuit and result visualization tools",
    # Serialization
    "qasm2": "OpenQASM 2.0 parsing and generation",
    "qasm3": "OpenQASM 3.0 parsing and generation",
    "qpy": "Qiskit Python serialization format (QPY) for circuit persistence",
    # Utilities
    "utils": "General utility functions and helpers",
    "exceptions": "Qiskit exception classes and error hierarchy",
}

AVAILABLE_ADDONS = {
    "aqc-tensor": "Approximate Quantum Compiler with tensor network techniques",
    "cutting": "Circuit cutting to run large circuits on smaller devices",
    "mpf": "Multi-product formulas for Hamiltonian simulation",
    "obp": "Operator backpropagation for expectation value estimation",
    "sqd": "Sample-based Quantum Diagonalization for chemistry and optimization",
    "utils": "Shared utilities for Qiskit addon packages",
}

AVAILABLE_GUIDES = {
    # Getting started
    "quick-start": "Get started with Qiskit — create and run your first circuit",
    # Circuit building
    "construct-circuits": "Build and manipulate quantum circuits",
    # Transpilation
    "transpile": "Transpile circuits for target backends",
    "transpiler-stages": "Understand the six stages of the transpiler pipeline",
    "transpile-with-pass-managers": "Use custom pass managers for transpilation",
    "defaults-and-configuration-options": "Transpiler defaults and configuration options",
    "circuit-transpilation-settings": "Circuit-level transpilation settings",
    "qiskit-transpiler-service": "Use the Qiskit Transpiler cloud service",
    # Error mitigation and suppression
    "error-mitigation-and-suppression-techniques": "Overview of error mitigation and suppression techniques",
    "configure-error-mitigation": "Configure error mitigation for Qiskit primitives",
    "configure-error-suppression": "Configure error suppression techniques",
    # Execution
    "primitives": "Use Sampler and Estimator primitives for quantum execution",
    "execution-modes": "Job, session, and batch execution modes",
    "runtime-options-overview": "Overview of Qiskit Runtime configuration options",
    "directed-execution-model": "Use the directed execution model",
    # Dynamic circuits
    "dynamic-circuits": "Mid-circuit measurements and classical control flow",
    # Post-processing addons
    "qiskit-addons-sqd": "Use Sample-based Quantum Diagonalization (SQD)",
    # Qiskit Functions - circuit functions
    "functions": "Overview of Qiskit Functions",
    "ibm-circuit-function": "IBM Circuit Function for optimized execution",
    "algorithmiq-tem": "Algorithmiq Tensor Error Mitigation (TEM)",
    "qedma-qesem": "Qedma Quantum Error Suppression and Error Mitigation (QESEM)",
    "q-ctrl-performance-management": "Q-CTRL Performance Management for optimized circuits",
    # Qiskit Functions - application functions
    "colibritd-pde": "ColibrITD PDE solver function",
    "global-data-quantum-optimizer": "Global Data Quantum Optimizer function",
    "qunova-chemistry": "Qunova Chemistry solver function",
    "kipu-optimization": "Kipu Optimization solver function",
    "q-ctrl-optimization-solver": "Q-CTRL Optimization Solver function",
    "multiverse-computing-singularity": "Multiverse Computing Singularity function",
    # Security and support
    "secure-data": "Data security and privacy on IBM Quantum",
    "support": "Getting support and help with Qiskit and IBM Quantum",
}

SEARCH_PATH = "endpoints-docs-learning/api/search"
