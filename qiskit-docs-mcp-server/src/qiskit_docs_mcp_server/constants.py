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

# Sitemap URL for dynamic page discovery
SITEMAP_URL = os.getenv(
    "QISKIT_DOCS_SITEMAP_URL",
    "https://quantum.cloud.ibm.com/docs/sitemap-0.xml",
)

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

# ---------------------------------------------------------------------------
# Fallback lists — used when sitemap discovery is unavailable
# ---------------------------------------------------------------------------

# Qiskit SDK modules and their documentation paths
AVAILABLE_MODULES: list[str] = [
    # Circuit construction
    "circuit",
    "circuit_annotation",
    "circuit_classical",
    "circuit_library",
    "circuit_random",
    "circuit_singleton",
    # Quantum information
    "quantum_info",
    # Transpilation
    "transpiler",
    "transpiler_passes",
    "transpiler_plugins",
    "transpiler_preset",
    "transpiler_synthesis_plugins",
    "synthesis",
    "dagcircuit",
    "passmanager",
    "converters",
    "compiler",
    # Primitives and providers
    "primitives",
    "providers",
    "providers_basic_provider",
    "providers_fake_provider",
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

AVAILABLE_ADDONS: list[str] = [
    "aqc-tensor",
    "cutting",
    "mpf",
    "obp",
    "sqd",
    "utils",
]

# Additional API packages beyond the core SDK and addons
AVAILABLE_API_PACKAGES: list[str] = [
    "qiskit-ibm-runtime",
    "qiskit-ibm-transpiler",
    "qiskit-c",
    "qiskit-runtime-rest",
    "quantum-system-rest",
    "functions",
]

AVAILABLE_GUIDES: list[str] = [
    "DAG-representation",
    "access-groups",
    "access-instances-platform-apis",
    "add-job-tags",
    "addons",
    "ai-transpiler-passes",
    "algorithmiq-tem",
    "allocation-limits",
    "bit-ordering",
    "build-noise-models",
    "c-extension-for-python",
    "calibration-jobs",
    "choose-execution-mode",
    "circuit-library",
    "circuit-transpilation-settings",
    "classical-feedforward-and-control-flow",
    "cloud-account-structure",
    "cloud-setup",
    "cloud-setup-invited",
    "cloud-setup-rest-api",
    "cloud-setup-untrusted",
    "code-of-conduct",
    "colibritd-pde",
    "common-parameters",
    "composer",
    "compute-services",
    "configure-error-mitigation",
    "configure-error-suppression",
    "configure-qiskit-local",
    "considerations-set-up-runtime",
    "construct-circuits",
    "context-based-restrictions",
    "create-a-provider",
    "create-transpiler-plugin",
    "custom-backend",
    "custom-roles",
    "custom-transpiler-pass",
    "debug-qiskit-runtime-jobs",
    "debugging-tools",
    "defaults-and-configuration-options",
    "directed-execution-model",
    "dynamical-decoupling-pass-manager",
    "error-mitigation-and-suppression-techniques",
    "error-mitigation-overview",
    "estimate-job-run-time",
    "execute-dynamic-circuits",
    "execution-modes",
    "execution-modes-faq",
    "execution-modes-rest-api",
    "fair-share-scheduler",
    "faq",
    "fractional-gates",
    "function-template-chemistry-workflow",
    "function-template-hamiltonian-simulation",
    "functions",
    "get-started-with-primitives",
    "global-data-quantum-optimizer",
    "ha-dr",
    "hello-world",
    "ibm-circuit-function",
    "initialize-account",
    "install-c-api",
    "install-qiskit",
    "install-qiskit-runtime",
    "install-qiskit-runtime-source",
    "install-qiskit-source",
    "instances",
    "interoperate-qiskit-qasm2",
    "interoperate-qiskit-qasm3",
    "intro-to-patterns",
    "introduction-to-qasm",
    "invite-and-manage-users",
    "job-limits",
    "kipu-optimization",
    "latest-updates",
    "local-simulators",
    "local-testing-mode",
    "logging",
    "manage-appid",
    "manage-cloud-users",
    "manage-cost",
    "max-execution-time",
    "measure-qubits",
    "metapackage-migration",
    "minimize-time",
    "monitor-job",
    "multiverse-computing-singularity",
    "noise-learning",
    "observability-quantum-system",
    "observability-runtime-rest",
    "online-lab-environments",
    "open-source",
    "operator-class",
    "operators-overview",
    "plans-overview",
    "plot-quantum-states",
    "primitive-input-output",
    "primitives",
    "primitives-examples",
    "primitives-rest-api",
    "processor-types",
    "pulse-migration",
    "q-ctrl-optimization-solver",
    "q-ctrl-performance-management",
    "qasm-feature-table",
    "qedma-qesem",
    "qiskit-1.0",
    "qiskit-1.0-features",
    "qiskit-1.0-installation",
    "qiskit-2.0",
    "qiskit-addons-sqd",
    "qiskit-addons-sqd-get-started",
    "qiskit-backendv1-to-v2",
    "qiskit-code-assistant",
    "qiskit-code-assistant-jupyterlab",
    "qiskit-code-assistant-local",
    "qiskit-code-assistant-openai-api",
    "qiskit-code-assistant-vscode",
    "qiskit-function-templates",
    "qiskit-mcp-servers",
    "qiskit-runtime-circuit-timing",
    "qiskit-runtime-primitives",
    "qiskit-sdk-version-strategy",
    "qiskit-transpiler-service",
    "qpu-information",
    "qrmi",
    "quick-start",
    "quickstart-steps-org",
    "qunova-chemistry",
    "repetition-rate-execution",
    "represent-quantum-computers",
    "responsibilities",
    "retired-qpus",
    "run-jobs-batch",
    "run-jobs-session",
    "runtime-options-overview",
    "save-circuits",
    "save-credentials",
    "save-jobs",
    "secure-data",
    "serverless",
    "serverless-first-program",
    "serverless-manage-resources",
    "serverless-port-code",
    "serverless-run-first-workload",
    "set-optimization",
    "simulate-stabilizer-circuits",
    "simulate-with-qiskit-aer",
    "simulate-with-qiskit-sdk-primitives",
    "slurm-hpc-ux",
    "slurm-plugin",
    "specify-observables-pauli",
    "specify-runtime-options",
    "stretch",
    "support",
    "synthesize-unitary-operators",
    "tools-intro",
    "transpile",
    "transpile-with-pass-managers",
    "transpiler-plugins",
    "transpiler-stages",
    "upgrade-from-open",
    "v2-primitives",
    "view-cost",
    "virtual-private-endpoints",
    "visualize-circuit-timing",
    "visualize-circuits",
    "visualize-results",
]

AVAILABLE_TUTORIALS: list[str] = [
    "advanced-techniques-for-qaoa",
    "ai-transpiler-introduction",
    "approximate-quantum-compilation-for-time-evolution",
    "chsh-inequality",
    "colibritd-pde",
    "combine-error-mitigation-techniques",
    "compilation-methods-for-hamiltonian-simulation-circuits",
    "dc-hex-ising",
    "depth-reduction-with-circuit-cutting",
    "edc-cut-bell-pair-benchmarking",
    "error-mitigation-with-qiskit-functions",
    "fractional-gates",
    "ghz-spacetime-codes",
    "global-data-quantum-optimizer",
    "grovers-algorithm",
    "krylov-quantum-diagonalization",
    "long-range-entanglement",
    "multi-product-formula",
    "nishimori-phase-transition",
    "operator-back-propagation",
    "pauli-correlation-encoding-for-qaoa",
    "periodic-boundary-conditions-with-circuit-cutting",
    "probabilistic-error-amplification",
    "projected-quantum-kernels",
    "qedma-2d-ising-with-qesem",
    "quantum-approximate-optimization-algorithm",
    "quantum-kernel-training",
    "quantum-phase-estimation-qctrl",
    "qunova-hivqe",
    "readout-error-mitigation-sampler",
    "real-time-benchmarking-for-qubit-selection",
    "repetition-codes",
    "sample-based-krylov-quantum-diagonalization",
    "sample-based-quantum-diagonalization",
    "shors-algorithm",
    "simulate-kicked-ising-tem",
    "sml-classification",
    "solve-higher-order-binary-optimization-problems-with-q-ctrls-optimization-solver",
    "solve-market-split-problem-with-iskay-quantum-optimizer",
    "spin-chain-vqe",
    "transpilation-optimizations-with-sabre",
    "transverse-field-ising-model",
    "wire-cutting",
]

SEARCH_PATH = "endpoints-docs-learning/api/search"
