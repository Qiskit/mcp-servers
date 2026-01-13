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

"""Core IBM Runtime functions for the MCP server."""

import contextlib
import logging
import os
from typing import Any, Literal

from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
from qiskit_ibm_runtime.fake_provider import FakeProviderForBackendV2
from qiskit_ibm_runtime.options import SamplerOptions
from qiskit_mcp_server.circuit_serialization import CircuitFormat, load_circuit

from qiskit_ibm_runtime_mcp_server.utils import with_sync


# Type alias for dynamical decoupling sequence types
DDSequenceType = Literal["XX", "XpXm", "XY4"]

# Configure logging
logger = logging.getLogger(__name__)


def get_instance_from_env() -> str | None:
    """
    Get IBM Quantum instance from MCP server environment variable.

    This is an MCP server-specific environment variable (not a standard Qiskit SDK variable).
    Setting an instance avoids the slow instance lookup during service initialization.

    The instance should be a Cloud Resource Name (CRN) or service name for IBM Quantum Platform.

    Returns:
        Instance string if found in environment, None otherwise
    """
    instance = os.getenv("QISKIT_IBM_RUNTIME_MCP_INSTANCE")
    if instance and instance.strip():
        return instance.strip()
    return None


def least_busy(backends: list[Any]) -> Any | None:
    """Find the least busy backend from a list of backends."""
    if not backends:
        return None

    operational_backends = []
    for b in backends:
        try:
            if hasattr(b, "status"):
                status = b.status()
                if status.operational:
                    operational_backends.append((b, status.pending_jobs))
        except Exception as e:
            logger.warning(
                f"Skipping backend {getattr(b, 'name', 'unknown')} in least_busy: {e}"
            )
            continue

    if not operational_backends:
        return None

    # Sort by pending jobs and return the backend with fewest pending jobs
    operational_backends.sort(key=lambda x: x[1])
    return operational_backends[0][0]


def get_token_from_env() -> str | None:
    """
    Get IBM Quantum token from environment variables.

    Returns:
        Token string if found in environment, None otherwise
    """
    token = os.getenv("QISKIT_IBM_TOKEN")
    if (
        token
        and token.strip()
        and token.strip() not in ["<PASSWORD>", "<TOKEN>", "YOUR_TOKEN_HERE"]
    ):
        return token.strip()
    return None


# Global service instance
service: QiskitRuntimeService | None = None


def _build_adjacency_list(
    edges: list[list[int]], num_qubits: int
) -> dict[str, list[int]]:
    """Build adjacency list from edges."""
    adjacency: dict[str, list[int]] = {str(i): [] for i in range(num_qubits)}
    for edge in edges:
        adjacency[str(edge[0])].append(edge[1])
    return adjacency


def _process_coupling_map(
    edges: list[list[int]],
    num_qubits: int,
    backend_name: str,
    source: str | None = None,
) -> dict[str, Any]:
    """Process coupling map data into standardized response format.

    Args:
        edges: List of [control, target] qubit connection pairs
        num_qubits: Total number of qubits
        backend_name: Name of the backend
        source: Optional source identifier (e.g., "fake_backend")

    Returns:
        Standardized coupling map response dict
    """
    # Check if bidirectional (both [i,j] and [j,i] exist for all edges)
    edge_set = {(e[0], e[1]) for e in edges}
    bidirectional = len(edges) > 0 and all((e[1], e[0]) in edge_set for e in edges)

    # Build adjacency list
    adjacency_list = _build_adjacency_list(edges, num_qubits)

    result: dict[str, Any] = {
        "status": "success",
        "backend_name": backend_name,
        "num_qubits": num_qubits,
        "num_edges": len(edges),
        "edges": edges,
        "bidirectional": bidirectional,
        "adjacency_list": adjacency_list,
    }

    if source:
        result["source"] = source

    return result


def _create_runtime_service(channel: str, instance: str | None) -> QiskitRuntimeService:
    """
    Create a QiskitRuntimeService instance with the given channel and optional instance.

    Args:
        channel: Service channel ('ibm_quantum_platform')
        instance: IBM Quantum instance (CRN or service name), or None

    Returns:
        QiskitRuntimeService: New service instance
    """
    if instance:
        logger.info(f"Initializing with instance: {instance}")
        return QiskitRuntimeService(channel=channel, instance=instance)
    else:
        logger.info(
            "No instance specified - service will search all instances (slower). "
            "Set QISKIT_IBM_RUNTIME_MCP_INSTANCE for faster startup."
        )
        return QiskitRuntimeService(channel=channel)


def initialize_service(
    token: str | None = None,
    channel: str = "ibm_quantum_platform",
    instance: str | None = None,
) -> QiskitRuntimeService:
    """
    Initialize the Qiskit IBM Runtime service.

    Args:
        token: IBM Quantum API token (optional if saved)
        channel: Service channel ('ibm_quantum_platform')
        instance: IBM Quantum instance (e.g., 'ibm-q/open/main'). If provided,
                 significantly speeds up initialization by skipping instance lookup.

    Returns:
        QiskitRuntimeService: Initialized service instance
    """
    global service

    # Return existing service if already initialized (singleton pattern)
    if service is not None and token is None:
        return service

    # Check for instance in environment if not explicitly provided
    if instance is None:
        instance = get_instance_from_env()

    try:
        # First, try to initialize from saved credentials (unless a new token is explicitly provided)
        if not token:
            try:
                service = _create_runtime_service(channel, instance)
                logger.info(
                    f"Successfully initialized IBM Runtime service from saved credentials on channel: {channel}"
                )
                return service
            except Exception as e:
                logger.info(f"No saved credentials found or invalid: {e}")
                raise ValueError(
                    "No IBM Quantum token provided and no saved credentials available"
                ) from e

        # If a token is provided, validate it's not a placeholder before saving
        if token and token.strip():
            # Check for common placeholder patterns
            if token.strip() in ["<PASSWORD>", "<TOKEN>", "YOUR_TOKEN_HERE", "xxx"]:
                raise ValueError(
                    f"Invalid token: '{token.strip()}' appears to be a placeholder value"
                )

            # Save account with provided token
            try:
                QiskitRuntimeService.save_account(
                    channel=channel, token=token.strip(), overwrite=True
                )
                logger.info(f"Saved IBM Quantum account for channel: {channel}")
            except Exception as e:
                logger.error(f"Failed to save account: {e}")
                raise ValueError("Invalid token or channel") from e

            # Initialize service with the new token
            try:
                service = _create_runtime_service(channel, instance)
                logger.info(
                    f"Successfully initialized IBM Runtime service on channel: {channel}"
                )
                return service
            except Exception as e:
                logger.error(f"Failed to initialize IBM Runtime service: {e}")
                raise

    except Exception as e:
        if not isinstance(e, ValueError):
            logger.error(f"Failed to initialize IBM Runtime service: {e}")
        raise


@with_sync
async def setup_ibm_quantum_account(
    token: str | None = None, channel: str = "ibm_quantum_platform"
) -> dict[str, Any]:
    """
    Set up IBM Quantum account with credentials.

    Args:
        token: IBM Quantum API token (optional - will try environment or saved credentials)
        channel: Service channel ('ibm_quantum_platform')

    Returns:
        Setup status and information
    """
    # Try to get token from environment if not provided
    if not token or not token.strip():
        env_token = get_token_from_env()
        if env_token:
            logger.info("Using token from QISKIT_IBM_TOKEN environment variable")
            token = env_token
        else:
            # Try to use saved credentials
            logger.info("No token provided, attempting to use saved credentials")
            token = None

    if channel not in ["ibm_quantum_platform"]:
        return {
            "status": "error",
            "message": "Channel must be 'ibm_quantum_platform'",
        }

    try:
        service_instance = initialize_service(token.strip() if token else None, channel)

        # Get backend count for response
        try:
            backends = service_instance.backends()
            backend_count = len(backends)
        except Exception:
            backend_count = 0

        return {
            "status": "success",
            "message": f"IBM Quantum account set up successfully for channel: {channel}",
            "channel": service_instance._channel,
            "available_backends": backend_count,
        }
    except Exception as e:
        logger.error(f"Failed to set up IBM Quantum account: {e}")
        return {"status": "error", "message": f"Failed to set up account: {e!s}"}


@with_sync
async def list_backends() -> dict[str, Any]:
    """
    List available IBM Quantum backends.

    Returns:
        List of backends with their properties
    """
    global service

    try:
        if service is None:
            service = initialize_service()

        backends = service.backends()
        backend_list = []

        for backend in backends:
            backend_name = getattr(backend, "name", "unknown")
            num_qubits = getattr(backend, "num_qubits", 0)
            simulator = getattr(backend, "simulator", False)

            # Try to get status (this is where API errors can occur)
            try:
                status = backend.status()
                backend_info = {
                    "name": backend_name,
                    "num_qubits": num_qubits,
                    "simulator": simulator,
                    "operational": status.operational,
                    "pending_jobs": status.pending_jobs,
                    "status_msg": status.status_msg,
                }
            except Exception as status_err:
                logger.warning(
                    f"Failed to get status for backend {backend_name}: {status_err}"
                )
                backend_info = {
                    "name": backend_name,
                    "num_qubits": num_qubits,
                    "simulator": simulator,
                    "operational": False,
                    "pending_jobs": 0,
                    "status_msg": "Status unavailable",
                }

            backend_list.append(backend_info)

        return {
            "status": "success",
            "backends": backend_list,
            "total_backends": len(backend_list),
        }

    except Exception as e:
        logger.error(f"Failed to list backends: {e}")
        return {"status": "error", "message": f"Failed to list backends: {e!s}"}


@with_sync
async def least_busy_backend() -> dict[str, Any]:
    """
    Find the least busy operational backend.

    Returns:
        Information about the least busy backend
    """
    global service

    try:
        if service is None:
            service = initialize_service()

        # Don't filter by operational=True here since that filter might trigger
        # API calls for problematic backends. Let least_busy() handle the filtering.
        backends = service.backends(simulator=False)

        if not backends:
            return {
                "status": "error",
                "message": "No quantum backends available",
            }

        backend = least_busy(backends)
        if backend is None:
            return {
                "status": "error",
                "message": "Could not find a suitable operational backend. "
                "All backends may be offline or under maintenance.",
            }

        try:
            status = backend.status()
            return {
                "status": "success",
                "backend_name": backend.name,
                "num_qubits": getattr(backend, "num_qubits", 0),
                "pending_jobs": status.pending_jobs,
                "operational": status.operational,
                "status_msg": status.status_msg,
            }
        except Exception as status_err:
            logger.warning(
                f"Could not get final status for {backend.name}: {status_err}"
            )
            return {
                "status": "success",
                "backend_name": backend.name,
                "num_qubits": getattr(backend, "num_qubits", 0),
                "pending_jobs": 0,
                "operational": True,
                "status_msg": "Status refresh failed but backend was operational",
            }

    except Exception as e:
        logger.error(f"Failed to find least busy backend: {e}")
        return {
            "status": "error",
            "message": f"Failed to find least busy backend: {e!s}",
        }


@with_sync
async def get_backend_properties(backend_name: str) -> dict[str, Any]:
    """
    Get detailed properties of a specific backend.

    Args:
        backend_name: Name of the backend

    Returns:
        Backend properties and capabilities
    """
    global service

    try:
        if service is None:
            service = initialize_service()

        backend = service.backend(backend_name)
        status = backend.status()

        # Get configuration
        processor_type = None
        backend_version = None
        basis_gates: list[str] = []
        coupling_map: list[list[int]] = []
        max_shots = 0
        max_experiments = 0
        try:
            config = backend.configuration()
            basis_gates = getattr(config, "basis_gates", []) or []
            coupling_map = getattr(config, "coupling_map", []) or []
            max_shots = getattr(config, "max_shots", 0)
            max_experiments = getattr(config, "max_experiments", 0)
            backend_version = getattr(config, "backend_version", None)
            processor_type = getattr(config, "processor_type", None)
            # processor_type may be a dict with 'family' and 'revision' keys
            if isinstance(processor_type, dict):
                family = processor_type.get("family", "")
                revision = processor_type.get("revision", "")
                processor_type = f"{family} r{revision}" if revision else family
        except Exception:
            pass  # nosec B110 - Intentionally ignoring config errors; defaults are acceptable

        return {
            "status": "success",
            "backend_name": backend.name,
            "num_qubits": getattr(backend, "num_qubits", 0),
            "simulator": getattr(backend, "simulator", False),
            "operational": status.operational,
            "pending_jobs": status.pending_jobs,
            "status_msg": status.status_msg,
            "processor_type": processor_type,
            "backend_version": backend_version,
            "basis_gates": basis_gates,
            "coupling_map": coupling_map,
            "max_shots": max_shots,
            "max_experiments": max_experiments,
        }

    except Exception as e:
        logger.error(f"Failed to get backend properties: {e}")
        return {
            "status": "error",
            "message": f"Failed to get backend properties: {e!s}",
        }


def _get_fake_backend(backend_name: str) -> Any:
    """
    Get a fake backend by name from the FakeProviderForBackendV2.

    Args:
        backend_name: Name of the fake backend (e.g., 'fake_brisbane')

    Returns:
        Fake backend instance

    Raises:
        ValueError: If the fake backend is not found
    """
    provider = FakeProviderForBackendV2()
    backends = provider.backends()

    for backend in backends:
        if backend.name == backend_name:
            return backend

    # Only build available list for error message
    available = sorted(b.name for b in backends)[:10]
    raise ValueError(
        f"Fake backend '{backend_name}' not found. "
        f"Available fake backends: {', '.join(available)}..."
    )


@with_sync
async def get_coupling_map(backend_name: str) -> dict[str, Any]:
    """
    Get the coupling map (qubit connectivity) for a specific backend.

    Supports both real IBM Quantum backends and fake backends (no credentials needed).
    Use 'fake_' prefix for fake backends (e.g., 'fake_brisbane', 'fake_sherbrooke').

    Args:
        backend_name: Name of the backend. Use 'fake_' prefix for fake backends.

    Returns:
        Coupling map details including edges and adjacency list
    """
    try:
        # Check if this is a fake backend request
        if backend_name.startswith("fake_"):
            return _get_fake_backend_coupling_map(backend_name)

        # Real backend - requires credentials
        global service
        if service is None:
            service = initialize_service()

        backend = service.backend(backend_name)
        num_qubits = getattr(backend, "num_qubits", 0)

        # Get configuration
        edges: list[list[int]] = []
        try:
            config = backend.configuration()
            coupling_map_raw = getattr(config, "coupling_map", []) or []
            edges = [[int(e[0]), int(e[1])] for e in coupling_map_raw]
        except Exception:
            pass  # nosec B110 - Config errors are acceptable; defaults used

        return _process_coupling_map(edges, num_qubits, backend.name)

    except Exception as e:
        logger.error(f"Failed to get coupling map: {e}")
        return {
            "status": "error",
            "message": f"Failed to get coupling map: {e!s}",
        }


def _get_fake_backend_coupling_map(backend_name: str) -> dict[str, Any]:
    """
    Get coupling map from a fake backend (no credentials needed).

    Args:
        backend_name: Name of the fake backend (e.g., 'fake_brisbane')

    Returns:
        Coupling map details
    """
    backend = _get_fake_backend(backend_name)
    coupling_map = backend.coupling_map

    if coupling_map is None:
        return {
            "status": "error",
            "message": f"Backend '{backend_name}' has no coupling map (fully connected)",
        }

    edges = [[int(e[0]), int(e[1])] for e in coupling_map.get_edges()]
    num_qubits = coupling_map.size()

    return _process_coupling_map(edges, num_qubits, backend.name, source="fake_backend")


def _get_qubit_calibration_data(
    properties: Any, qubit: int, faulty_qubits: list[int]
) -> dict[str, Any]:
    """Extract calibration data for a single qubit."""
    qubit_info: dict[str, Any] = {
        "qubit": qubit,
        "t1_us": None,
        "t2_us": None,
        "frequency_ghz": None,
        "readout_error": None,
        "prob_meas0_prep1": None,
        "prob_meas1_prep0": None,
        "operational": qubit not in faulty_qubits,
    }

    # Get T1 time (in microseconds)
    with contextlib.suppress(Exception):
        t1 = properties.t1(qubit)
        if t1 is not None:
            qubit_info["t1_us"] = round(t1 * 1e6, 2) if t1 < 1 else round(t1, 2)

    # Get T2 time (in microseconds)
    with contextlib.suppress(Exception):
        t2 = properties.t2(qubit)
        if t2 is not None:
            qubit_info["t2_us"] = round(t2 * 1e6, 2) if t2 < 1 else round(t2, 2)

    # Get qubit frequency (in GHz)
    with contextlib.suppress(Exception):
        freq = properties.frequency(qubit)
        if freq is not None:
            qubit_info["frequency_ghz"] = round(freq / 1e9, 6)

    # Get readout error
    with contextlib.suppress(Exception):
        readout_err = properties.readout_error(qubit)
        if readout_err is not None:
            qubit_info["readout_error"] = round(readout_err, 6)

    # Get measurement preparation errors if available
    with contextlib.suppress(Exception):
        prob_meas0_prep1 = properties.prob_meas0_prep1(qubit)
        if prob_meas0_prep1 is not None:
            qubit_info["prob_meas0_prep1"] = round(prob_meas0_prep1, 6)

    with contextlib.suppress(Exception):
        prob_meas1_prep0 = properties.prob_meas1_prep0(qubit)
        if prob_meas1_prep0 is not None:
            qubit_info["prob_meas1_prep0"] = round(prob_meas1_prep0, 6)

    return qubit_info


def _get_gate_errors(
    properties: Any, qubit_indices: list[int], coupling_map: list[list[int]]
) -> list[dict[str, Any]]:
    """Extract gate error data for common gates."""
    gate_errors: list[dict[str, Any]] = []
    single_qubit_gates = ["x", "sx", "rz"]
    two_qubit_gates = ["cx", "ecr", "cz"]

    # Single-qubit gates
    for gate in single_qubit_gates:
        for qubit in qubit_indices[:5]:
            with contextlib.suppress(Exception):
                error = properties.gate_error(gate, [qubit])
                if error is not None:
                    gate_errors.append(
                        {"gate": gate, "qubits": [qubit], "error": round(error, 6)}
                    )

    # Two-qubit gates
    for gate in two_qubit_gates:
        for edge in coupling_map[:5]:
            with contextlib.suppress(Exception):
                error = properties.gate_error(gate, edge)
                if error is not None:
                    gate_errors.append(
                        {"gate": gate, "qubits": edge, "error": round(error, 6)}
                    )

    return gate_errors


# Type alias for chain scoring metrics
ScoringMetric = Literal["two_qubit_error", "readout_error", "combined"]

# Type alias for QV subgraph scoring metrics
QVScoringMetric = Literal["qv_optimized", "connectivity", "gate_error"]


def _find_all_linear_chains(
    adjacency_list: dict[str, list[int]],
    chain_length: int,
    faulty_qubits: set[int],
) -> list[list[int]]:
    """Find all valid linear chains of specified length using DFS.

    Args:
        adjacency_list: Mapping from qubit index (as string) to list of connected qubits
        chain_length: Number of qubits in each chain
        faulty_qubits: Set of qubit indices to exclude from chains

    Returns:
        List of chains, where each chain is a list of qubit indices
    """
    chains: list[list[int]] = []
    num_qubits = len(adjacency_list)

    def dfs(current: int, path: list[int], visited: set[int]) -> None:
        if len(path) == chain_length:
            chains.append(path.copy())
            return

        for neighbor in adjacency_list.get(str(current), []):
            if neighbor not in visited and neighbor not in faulty_qubits:
                visited.add(neighbor)
                path.append(neighbor)
                dfs(neighbor, path, visited)
                path.pop()
                visited.remove(neighbor)

    # Start DFS from each non-faulty qubit
    for start in range(num_qubits):
        if start not in faulty_qubits:
            dfs(start, [start], {start})

    return chains


def _score_chain(
    chain: list[int],
    qubit_calibration: dict[int, dict[str, Any]],
    gate_errors: dict[tuple[int, int], float],
    metric: ScoringMetric,
) -> float:
    """Score a chain based on selected metric. Lower is better.

    Args:
        chain: List of qubit indices forming the chain
        qubit_calibration: Per-qubit calibration data (t1_us, t2_us, readout_error)
        gate_errors: Two-qubit gate errors keyed by (control, target) tuple
        metric: Scoring metric to use

    Returns:
        Score value (lower is better)
    """
    if metric == "two_qubit_error":
        # Sum of two-qubit gate errors between consecutive qubits
        total = 0.0
        for i in range(len(chain) - 1):
            edge = (chain[i], chain[i + 1])
            reverse_edge = (chain[i + 1], chain[i])
            # Look up gate error for this edge; try both directions since coupling
            # maps may store edges in either order. Fall back to 1% if not found.
            total += gate_errors.get(edge, gate_errors.get(reverse_edge, 0.01))
        return total

    elif metric == "readout_error":
        # Sum of readout errors for all qubits
        return float(
            sum(qubit_calibration.get(q, {}).get("readout_error", 0.01) for q in chain)
        )

    # Combined metric: gate_errors + readout + inverse coherence
    gate_score = 0.0
    for i in range(len(chain) - 1):
        edge = (chain[i], chain[i + 1])
        reverse_edge = (chain[i + 1], chain[i])
        gate_score += gate_errors.get(edge, gate_errors.get(reverse_edge, 0.01))

    readout_score = float(
        sum(qubit_calibration.get(q, {}).get("readout_error", 0.01) for q in chain)
    )

    # Lower T1/T2 is worse, so use inverse (capped to avoid division issues)
    coherence_score = float(
        sum(1.0 / max(qubit_calibration.get(q, {}).get("t1_us", 100), 1) for q in chain)
    )

    return gate_score + readout_score + 0.01 * coherence_score


def _build_qubit_calibration(
    properties: Any, num_qubits: int, faulty_qubits: set[int]
) -> dict[int, dict[str, Any]]:
    """Build qubit calibration dictionary from backend properties.

    Args:
        properties: Backend properties object
        num_qubits: Total number of qubits
        faulty_qubits: Set of faulty qubit indices to skip

    Returns:
        Dictionary mapping qubit index to calibration data
    """
    qubit_calibration: dict[int, dict[str, Any]] = {}
    for q in range(num_qubits):
        if q in faulty_qubits:
            continue
        qubit_data: dict[str, Any] = {}
        with contextlib.suppress(Exception):
            t1 = properties.t1(q)
            if t1 is not None:
                qubit_data["t1_us"] = t1 * 1e6 if t1 < 1 else t1
        with contextlib.suppress(Exception):
            t2 = properties.t2(q)
            if t2 is not None:
                qubit_data["t2_us"] = t2 * 1e6 if t2 < 1 else t2
        with contextlib.suppress(Exception):
            readout_err = properties.readout_error(q)
            if readout_err is not None:
                qubit_data["readout_error"] = readout_err
        qubit_calibration[q] = qubit_data
    return qubit_calibration


def _build_gate_errors(
    properties: Any, edges: list[list[int]]
) -> dict[tuple[int, int], float]:
    """Build gate error dictionary from backend properties.

    Args:
        properties: Backend properties object
        edges: List of coupling map edges

    Returns:
        Dictionary mapping (control, target) tuples to gate error rates
    """
    gate_errors: dict[tuple[int, int], float] = {}
    for edge in edges:
        for gate in ["cx", "ecr", "cz"]:
            with contextlib.suppress(Exception):
                error = properties.gate_error(gate, edge)
                if error is not None:
                    gate_errors[(edge[0], edge[1])] = error
                    break
    return gate_errors


def _build_chain_result(
    rank: int,
    chain: list[int],
    score: float,
    qubit_calibration: dict[int, dict[str, Any]],
    gate_errors: dict[tuple[int, int], float],
) -> dict[str, Any]:
    """Build result dictionary for a single chain.

    Args:
        rank: Ranking position (1 = best)
        chain: List of qubit indices in the chain
        score: Chain score
        qubit_calibration: Per-qubit calibration data
        gate_errors: Two-qubit gate errors

    Returns:
        Dictionary with chain metrics
    """
    return {
        "rank": rank,
        "qubits": chain,
        "score": round(score, 6),
        "qubit_details": [
            {
                "qubit": q,
                "t1_us": round(qubit_calibration.get(q, {}).get("t1_us") or 0, 2),
                "t2_us": round(qubit_calibration.get(q, {}).get("t2_us") or 0, 2),
                "readout_error": round(
                    qubit_calibration.get(q, {}).get("readout_error") or 0, 6
                ),
            }
            for q in chain
        ],
        "edge_errors": [
            {
                "edge": [chain[i], chain[i + 1]],
                "error": round(
                    gate_errors.get(
                        (chain[i], chain[i + 1]),
                        gate_errors.get((chain[i + 1], chain[i]), 0),
                    ),
                    6,
                ),
            }
            for i in range(len(chain) - 1)
        ],
    }


def _find_connected_subgraphs(
    adjacency_list: dict[str, list[int]],
    subgraph_size: int,
    faulty_qubits: set[int],
    qubit_calibration: dict[int, dict[str, Any]] | None = None,
) -> list[set[int]]:
    """Find connected subgraphs of specified size using DFS exploration.

    Unlike linear chains, these subgraphs can have any internal connectivity structure
    as long as all qubits are connected. This is essential for Quantum Volume which
    benefits from densely connected qubit sets.

    Args:
        adjacency_list: Mapping from qubit index (as string) to list of connected qubits
        subgraph_size: Number of qubits in each subgraph
        faulty_qubits: Set of qubit indices to exclude
        qubit_calibration: Optional calibration data to prioritize better qubits

    Returns:
        List of subgraphs, where each subgraph is a set of qubit indices
    """
    subgraphs: list[set[int]] = []
    seen: set[frozenset[int]] = set()
    num_qubits = len(adjacency_list)

    def get_neighbors(qubit: int) -> list[int]:
        """Get non-faulty neighbors of a qubit."""
        return [n for n in adjacency_list.get(str(qubit), []) if n not in faulty_qubits]

    def dfs_expand(current_set: set[int], frontier: set[int]) -> None:
        """Expand current set by adding connected qubits."""
        if len(current_set) == subgraph_size:
            frozen = frozenset(current_set)
            if frozen not in seen:
                seen.add(frozen)
                subgraphs.append(current_set.copy())
            return

        # Sort frontier by qubit quality if calibration available
        if qubit_calibration:
            sorted_frontier = sorted(
                frontier,
                key=lambda q: qubit_calibration.get(q, {}).get("readout_error", 1.0),
            )
        else:
            sorted_frontier = sorted(frontier)

        # Try adding each frontier qubit
        for next_qubit in sorted_frontier:
            new_set = current_set | {next_qubit}
            # New frontier: neighbors of next_qubit not already in set
            new_frontier = frontier.copy()
            new_frontier.discard(next_qubit)
            for neighbor in get_neighbors(next_qubit):
                if neighbor not in new_set:
                    new_frontier.add(neighbor)

            dfs_expand(new_set, new_frontier)

    # Get starting qubits, sorted by quality if calibration available
    valid_qubits = [q for q in range(num_qubits) if q not in faulty_qubits]
    if qubit_calibration:
        # Sort by readout error (lower is better)
        valid_qubits.sort(
            key=lambda q: qubit_calibration.get(q, {}).get("readout_error", 1.0)
        )

    # Start from each valid qubit (prioritizing better qubits)
    for start in valid_qubits:
        initial_frontier = set(get_neighbors(start))
        dfs_expand({start}, initial_frontier)

    return subgraphs


def _count_internal_edges(
    subgraph: set[int],
    adjacency_list: dict[str, list[int]],
) -> int:
    """Count the number of edges within a subgraph.

    For Quantum Volume, more internal edges mean more direct connectivity
    and fewer SWAP gates needed.

    Args:
        subgraph: Set of qubit indices
        adjacency_list: Full adjacency list of the backend

    Returns:
        Number of edges connecting qubits within the subgraph
    """
    edge_count = 0
    for qubit in subgraph:
        for neighbor in adjacency_list.get(str(qubit), []):
            if neighbor in subgraph and neighbor > qubit:  # Count each edge once
                edge_count += 1
    return edge_count


def _compute_average_path_length(
    subgraph: set[int],
    adjacency_list: dict[str, list[int]],
) -> float:
    """Compute average shortest path length between all qubit pairs in subgraph.

    Uses BFS to find shortest paths. Lower average path length means qubits
    can interact with fewer SWAP operations, which is beneficial for QV.

    Args:
        subgraph: Set of qubit indices
        adjacency_list: Full adjacency list of the backend

    Returns:
        Average shortest path length (1.0 for fully connected, higher for sparse)
    """
    if len(subgraph) < 2:
        return 0.0

    qubits = list(subgraph)
    total_distance = 0
    num_pairs = 0

    for i, source in enumerate(qubits):
        # BFS from source
        distances: dict[int, int] = {source: 0}
        queue = [source]
        head = 0

        while head < len(queue):
            current = queue[head]
            head += 1

            for neighbor in adjacency_list.get(str(current), []):
                if neighbor in subgraph and neighbor not in distances:
                    distances[neighbor] = distances[current] + 1
                    queue.append(neighbor)

        # Sum distances to qubits we haven't counted yet (avoid double counting)
        for j in range(i + 1, len(qubits)):
            target = qubits[j]
            if target in distances:
                total_distance += distances[target]
                num_pairs += 1

    return total_distance / num_pairs if num_pairs > 0 else float("inf")


def _score_qv_subgraph(
    subgraph: set[int],
    adjacency_list: dict[str, list[int]],
    qubit_calibration: dict[int, dict[str, Any]],
    gate_errors: dict[tuple[int, int], float],
    metric: QVScoringMetric,
) -> float:
    """Score a subgraph for Quantum Volume. Lower is better.

    Args:
        subgraph: Set of qubit indices
        adjacency_list: Full adjacency list of the backend
        qubit_calibration: Per-qubit calibration data
        gate_errors: Two-qubit gate errors
        metric: Scoring metric to use

    Returns:
        Score value (lower is better)
    """
    n = len(subgraph)
    max_edges = n * (n - 1) // 2  # Complete graph edges

    # Compute common metrics
    internal_edges = _count_internal_edges(subgraph, adjacency_list)
    avg_path_length = _compute_average_path_length(subgraph, adjacency_list)

    if metric == "connectivity":
        # Maximize connectivity: score = (max_edges - internal_edges) + avg_path_length
        # Lower score = more edges and shorter paths
        connectivity_penalty = max_edges - internal_edges
        return connectivity_penalty + avg_path_length

    elif metric == "gate_error":
        # Minimize total gate error on internal edges
        total_error = 0.0
        for qubit in subgraph:
            for neighbor in adjacency_list.get(str(qubit), []):
                if neighbor in subgraph and neighbor > qubit:
                    edge = (qubit, neighbor)
                    reverse_edge = (neighbor, qubit)
                    total_error += gate_errors.get(
                        edge, gate_errors.get(reverse_edge, 0.01)
                    )
        return total_error

    # metric == "qv_optimized"
    # Balanced: connectivity + gate errors + readout errors
    # Weight connectivity heavily since it's crucial for QV
    connectivity_score = (max_edges - internal_edges) * 0.5 + avg_path_length * 0.3

    # Gate errors on internal edges
    gate_error_sum = 0.0
    for qubit in subgraph:
        for neighbor in adjacency_list.get(str(qubit), []):
            if neighbor in subgraph and neighbor > qubit:
                edge = (qubit, neighbor)
                reverse_edge = (neighbor, qubit)
                gate_error_sum += gate_errors.get(
                    edge, gate_errors.get(reverse_edge, 0.01)
                )

    # Readout errors
    readout_sum = float(
        sum(qubit_calibration.get(q, {}).get("readout_error", 0.01) for q in subgraph)
    )

    # Coherence factor (inverse of T1 average, penalize low coherence)
    coherence_penalty = float(
        sum(
            1.0 / max(qubit_calibration.get(q, {}).get("t1_us", 100), 1)
            for q in subgraph
        )
    )

    return (
        connectivity_score
        + gate_error_sum * 10
        + readout_sum
        + coherence_penalty * 0.01
    )


def _build_qv_subgraph_result(
    rank: int,
    subgraph: set[int],
    score: float,
    adjacency_list: dict[str, list[int]],
    qubit_calibration: dict[int, dict[str, Any]],
    gate_errors: dict[tuple[int, int], float],
) -> dict[str, Any]:
    """Build result dictionary for a QV subgraph.

    Args:
        rank: Ranking position (1 = best)
        subgraph: Set of qubit indices
        score: Subgraph score
        adjacency_list: Full adjacency list
        qubit_calibration: Per-qubit calibration data
        gate_errors: Two-qubit gate errors

    Returns:
        Dictionary with subgraph metrics
    """
    qubits = sorted(subgraph)
    internal_edges = _count_internal_edges(subgraph, adjacency_list)
    avg_path_length = _compute_average_path_length(subgraph, adjacency_list)
    max_edges = len(subgraph) * (len(subgraph) - 1) // 2

    # Find actual internal edge pairs
    edge_list = []
    for qubit in subgraph:
        for neighbor in adjacency_list.get(str(qubit), []):
            if neighbor in subgraph and neighbor > qubit:
                edge = (qubit, neighbor)
                reverse_edge = (neighbor, qubit)
                error = gate_errors.get(edge, gate_errors.get(reverse_edge, 0))
                edge_list.append(
                    {
                        "edge": [qubit, neighbor],
                        "error": round(error, 6),
                    }
                )

    return {
        "rank": rank,
        "qubits": qubits,
        "score": round(score, 6),
        "internal_edges": internal_edges,
        "max_possible_edges": max_edges,
        "connectivity_ratio": round(internal_edges / max_edges, 3)
        if max_edges > 0
        else 0,
        "average_path_length": round(avg_path_length, 3),
        "qubit_details": [
            {
                "qubit": q,
                "t1_us": round(qubit_calibration.get(q, {}).get("t1_us") or 0, 2),
                "t2_us": round(qubit_calibration.get(q, {}).get("t2_us") or 0, 2),
                "readout_error": round(
                    qubit_calibration.get(q, {}).get("readout_error") or 0, 6
                ),
            }
            for q in qubits
        ],
        "edge_errors": edge_list,
    }


@with_sync
async def get_backend_calibration(
    backend_name: str, qubit_indices: list[int] | None = None
) -> dict[str, Any]:
    """
    Get calibration data for a specific backend including T1, T2, and error rates.

    Args:
        backend_name: Name of the backend
        qubit_indices: Optional list of qubit indices to get data for.
                      If None, returns data for all qubits (limited to first 10 for brevity).

    Returns:
        Calibration data including T1, T2 times and error rates
    """
    global service

    try:
        if service is None:
            service = initialize_service()

        backend = service.backend(backend_name)
        num_qubits = getattr(backend, "num_qubits", 0)

        # Get coupling map from configuration (needed for gate errors)
        coupling_map: list[list[int]] = []
        with contextlib.suppress(Exception):
            config = backend.configuration()
            coupling_map = getattr(config, "coupling_map", []) or []

        # Get backend properties (calibration data)
        try:
            properties = backend.properties()
        except Exception as e:
            logger.warning(f"Could not get properties for {backend_name}: {e}")
            return {
                "status": "error",
                "message": f"Calibration data not available for {backend_name}. "
                "This may be a simulator or the backend doesn't provide calibration data.",
            }

        if properties is None:
            return {
                "status": "error",
                "message": f"No calibration data available for {backend_name}. "
                "This is likely a simulator backend.",
            }

        # Get faulty qubits and gates (important for avoiding failed jobs)
        faulty_qubits: list[int] = []
        faulty_gates: list[dict[str, Any]] = []
        with contextlib.suppress(Exception):
            faulty_qubits = list(properties.faulty_qubits())

        with contextlib.suppress(Exception):
            faulty_gates_raw = properties.faulty_gates()
            for gate in faulty_gates_raw:
                with contextlib.suppress(Exception):
                    faulty_gates.append(
                        {"gate": gate.gate, "qubits": list(gate.qubits)}
                    )

        # Determine which qubits to report on
        if qubit_indices is None:
            qubit_indices = list(range(min(10, num_qubits)))
        else:
            qubit_indices = [q for q in qubit_indices if 0 <= q < num_qubits]

        # Collect qubit calibration data
        qubit_data: list[dict[str, Any]] = []
        for qubit in qubit_indices:
            try:
                qubit_data.append(
                    _get_qubit_calibration_data(properties, qubit, faulty_qubits)
                )
            except Exception as qe:
                logger.warning(f"Failed to get calibration for qubit {qubit}: {qe}")
                qubit_data.append({"qubit": qubit, "error": str(qe)})

        # Collect gate error data
        gate_errors = _get_gate_errors(properties, qubit_indices, coupling_map)

        # Get last calibration time if available
        last_update = None
        with contextlib.suppress(Exception):
            last_update = str(properties.last_update_date)

        return {
            "status": "success",
            "backend_name": backend_name,
            "num_qubits": num_qubits,
            "last_calibration": last_update,
            "faulty_qubits": faulty_qubits,
            "faulty_gates": faulty_gates,
            "qubit_calibration": qubit_data,
            "gate_errors": gate_errors,
            "note": "T1/T2 in microseconds, frequency in GHz, errors are probabilities (0-1). "
            f"Showing data for qubits {qubit_indices}. "
            "Check faulty_qubits/faulty_gates before submitting jobs.",
        }

    except Exception as e:
        logger.error(f"Failed to get backend calibration: {e}")
        return {
            "status": "error",
            "message": f"Failed to get backend calibration: {e!s}",
        }


@with_sync
async def list_my_jobs(limit: int = 10) -> dict[str, Any]:
    """
    List user's recent jobs.

    Args:
        limit: Maximum number of jobs to retrieve

    Returns:
        List of jobs with their information
    """
    global service

    try:
        if service is None:
            service = initialize_service()

        jobs = service.jobs(limit=limit)
        job_list = []

        for job in jobs:
            try:
                job_info = {
                    "job_id": job.job_id(),
                    "status": job.status(),
                    "creation_date": getattr(job, "creation_date", "Unknown"),
                    "backend": job.backend().name if job.backend() else "Unknown",
                    "tags": getattr(job, "tags", []),
                    "error_message": job.error_message()
                    if hasattr(job, "error_message")
                    else None,
                }
                job_list.append(job_info)
            except Exception as je:
                logger.warning(f"Failed to get info for job: {je}")
                continue

        return {"status": "success", "jobs": job_list, "total_jobs": len(job_list)}

    except Exception as e:
        logger.error(f"Failed to list jobs: {e}")
        return {"status": "error", "message": f"Failed to list jobs: {e!s}"}


@with_sync
async def get_job_status(job_id: str) -> dict[str, Any]:
    """
    Get status of a specific job.

    Args:
        job_id: ID of the job

    Returns:
        Job status information
    """
    global service

    try:
        if service is None:
            return {
                "status": "error",
                "message": "Failed to get job status: service not initialized",
            }

        job = service.job(job_id)

        job_info = {
            "status": "success",
            "job_id": job.job_id(),
            "job_status": job.status(),
            "creation_date": getattr(job, "creation_date", "Unknown"),
            "backend": job.backend().name if job.backend() else "Unknown",
            "tags": getattr(job, "tags", []),
            "error_message": job.error_message()
            if hasattr(job, "error_message")
            else None,
        }

        return job_info

    except Exception as e:
        logger.error(f"Failed to get job status: {e}")
        return {"status": "error", "message": f"Failed to get job status: {e!s}"}


@with_sync
async def get_job_results(job_id: str) -> dict[str, Any]:
    """
    Get measurement results from a completed quantum job.

    Retrieves the measurement outcomes (counts) from a job that has finished
    execution. The job must be in DONE status to retrieve results.

    Args:
        job_id: ID of the completed job

    Returns:
        Dictionary containing:
        - status: "success", "pending", or "error"
        - job_id: The job ID
        - job_status: Current status of the job
        - counts: Dictionary of measurement outcomes and their counts (if completed)
        - shots: Total number of shots executed (if completed)
        - backend: Name of the backend used
        - execution_time: Time taken to execute (if available)
        - message: Status message or error description
    """
    global service

    try:
        if service is None:
            return {
                "status": "error",
                "message": "Failed to get job results: service not initialized",
            }

        job = service.job(job_id)
        job_status = job.status()
        backend_name = job.backend().name if job.backend() else "Unknown"

        # Check if job is still running
        # Official status values: INITIALIZING, QUEUED, RUNNING, CANCELLED, DONE, ERROR
        if job_status in ["INITIALIZING", "QUEUED", "RUNNING"]:
            return {
                "status": "pending",
                "job_id": job_id,
                "job_status": job_status,
                "backend": backend_name,
                "message": f"Job is still {job_status.lower()}. Please check again later.",
            }

        # Check if job failed or was cancelled
        if job_status in ["CANCELLED", "ERROR"]:
            error_msg = job.error_message() if hasattr(job, "error_message") else None
            return {
                "status": "error",
                "job_id": job_id,
                "job_status": job_status,
                "backend": backend_name,
                "message": f"Job {job_status.lower()}"
                + (f": {error_msg}" if error_msg else ""),
            }

        # Job is DONE - retrieve results
        result = job.result()

        # Extract counts from the result
        # SamplerV2 results have data attributes for each classical register
        counts = {}
        if result and len(result) > 0:
            pub_result = result[0]
            # Try common classical register names
            data = pub_result.data
            for attr_name in ["meas", "c", "cr", "result"]:
                if hasattr(data, attr_name):
                    creg_data = getattr(data, attr_name)
                    if hasattr(creg_data, "get_counts"):
                        counts = creg_data.get_counts()
                        break
            # If no common name found, try to get any BitArray attribute
            if not counts:
                for attr_name in dir(data):
                    if not attr_name.startswith("_"):
                        attr = getattr(data, attr_name)
                        if hasattr(attr, "get_counts"):
                            counts = attr.get_counts()
                            break

        # Calculate total shots from counts
        total_shots = sum(counts.values()) if counts else 0

        # Get execution time if available
        execution_time = None
        if hasattr(job, "metrics") and job.metrics():
            metrics = job.metrics()
            if "usage" in metrics:
                execution_time = metrics["usage"].get("quantum_seconds")

        return {
            "status": "success",
            "job_id": job_id,
            "job_status": job_status,
            "backend": backend_name,
            "counts": counts,
            "shots": total_shots,
            "execution_time": execution_time,
            "message": "Results retrieved successfully",
        }

    except Exception as e:
        logger.error(f"Failed to get job results: {e}")
        return {"status": "error", "message": f"Failed to get job results: {e!s}"}


@with_sync
async def cancel_job(job_id: str) -> dict[str, Any]:
    """
    Cancel a specific job.

    Args:
        job_id: ID of the job to cancel

    Returns:
        Cancellation status
    """
    global service

    try:
        if service is None:
            return {
                "status": "error",
                "message": "Failed to cancel job: service not initialized",
            }

        job = service.job(job_id)
        job.cancel()

        return {
            "status": "success",
            "job_id": job_id,
            "message": "Job cancellation requested",
        }
    except Exception as e:
        logger.error(f"Failed to cancel job: {e}")
        return {"status": "error", "message": f"Failed to cancel job: {e!s}"}


@with_sync
async def get_service_status() -> str:
    """
    Get current IBM Quantum service status.

    Returns:
        Service connection status and basic information
    """
    global service

    try:
        if service is None:
            service = initialize_service()

        # Test connectivity by listing backends
        backends = service.backends()
        backend_count = len(backends)

        status_info = {
            "connected": True,
            "channel": service._channel,
            "available_backends": backend_count,
            "service": "IBM Quantum",
        }

        return f"IBM Quantum Service Status: {status_info}"

    except Exception as e:
        logger.error(f"Failed to check service status: {e}")
        status_info = {"connected": False, "error": str(e), "service": "IBM Quantum"}
        return f"IBM Quantum Service Status: {status_info}"


def _clamp(value: int, min_val: int, max_val: int) -> int:
    """Clamp value to range [min_val, max_val]."""
    return max(min_val, min(value, max_val))


@with_sync
async def find_optimal_qubit_chains(
    backend_name: str,
    chain_length: int = 5,
    num_results: int = 5,
    metric: ScoringMetric = "two_qubit_error",
) -> dict[str, Any]:
    """
    Find optimal linear qubit chains for experiments based on connectivity and calibration.

    Uses coupling map connectivity to find all valid linear chains of the specified length,
    then scores each chain based on calibration data (gate errors, readout errors, coherence
    times). Returns the top N chains ranked by the selected metric.

    Args:
        backend_name: Name of the backend (e.g., 'ibm_brisbane')
        chain_length: Number of qubits in each chain (default: 5, range: 2-20)
        num_results: Number of top chains to return (default: 5, max: 20)
        metric: Scoring metric to optimize:
            - "two_qubit_error": Minimize sum of CX/ECR gate errors (default)
            - "readout_error": Minimize sum of measurement errors
            - "combined": Weighted combination of gate errors, readout, and coherence

    Returns:
        Ranked chains with detailed metrics including:
        - qubits: Ordered list of qubit indices in the chain
        - score: Total score (lower is better)
        - qubit_details: T1, T2, readout_error for each qubit
        - edge_errors: Two-qubit gate error for each connection
    """
    global service

    try:
        # Validate inputs
        chain_length = _clamp(chain_length, 2, 20)
        num_results = _clamp(num_results, 1, 20)

        # Check for fake backends early - they don't have calibration data
        if backend_name.startswith("fake_"):
            return {
                "status": "error",
                "message": f"Fake backends like '{backend_name}' are not supported. "
                "This tool requires real-time calibration data from IBM Quantum backends. "
                "Use get_coupling_map_tool for connectivity information on fake backends.",
            }

        # Initialize service if needed
        if service is None:
            service = initialize_service()

        # Get coupling map
        coupling_result = await get_coupling_map(backend_name)
        if coupling_result["status"] == "error":
            return coupling_result

        adjacency_list = coupling_result["adjacency_list"]
        backend_num_qubits = coupling_result["num_qubits"]

        # Validate requested size against backend capacity
        if chain_length > backend_num_qubits:
            return {
                "status": "error",
                "message": f"Requested chain_length={chain_length}, but {backend_name} only has "
                f"{backend_num_qubits} qubits. Please reduce chain_length.",
            }

        # Get backend properties
        backend = service.backend(backend_name)
        properties = _get_backend_properties_for_chains(backend, backend_name)
        if isinstance(properties, dict):
            return properties  # Error response

        # Extract faulty qubits
        faulty_qubits: set[int] = set()
        with contextlib.suppress(Exception):
            faulty_qubits = set(properties.faulty_qubits())

        # Build calibration data
        qubit_calibration = _build_qubit_calibration(
            properties, backend_num_qubits, faulty_qubits
        )
        gate_errors = _build_gate_errors(properties, coupling_result["edges"])

        # Find all chains
        chains = _find_all_linear_chains(adjacency_list, chain_length, faulty_qubits)

        if not chains:
            return {
                "status": "error",
                "message": f"No valid chains of length {chain_length} found on {backend_name}. "
                f"The backend has {backend_num_qubits} qubits and {len(faulty_qubits)} faulty qubits.",
            }

        # Score, rank, and build results
        scored_chains = [
            (c, _score_chain(c, qubit_calibration, gate_errors, metric)) for c in chains
        ]
        scored_chains.sort(key=lambda x: x[1])
        top_chains = scored_chains[:num_results]

        results = [
            _build_chain_result(rank, chain, score, qubit_calibration, gate_errors)
            for rank, (chain, score) in enumerate(top_chains, 1)
        ]

        return {
            "status": "success",
            "backend_name": backend_name,
            "chain_length": chain_length,
            "metric": metric,
            "total_chains_found": len(chains),
            "faulty_qubits": list(faulty_qubits),
            "chains": results,
        }

    except Exception as e:
        logger.error(f"Failed to find optimal chains: {e}")
        return {
            "status": "error",
            "message": f"Failed to find optimal chains: {e!s}",
        }


def _get_backend_properties_for_chains(
    backend: Any, backend_name: str
) -> Any | dict[str, Any]:
    """Get backend properties or return error dict if unavailable.

    Args:
        backend: Backend object
        backend_name: Name of the backend for error messages

    Returns:
        Properties object if available, or error dict
    """
    try:
        properties = backend.properties()
    except Exception as e:
        logger.warning(f"Could not get properties for {backend_name}: {e}")
        return {
            "status": "error",
            "message": f"Calibration data not available for {backend_name}. "
            "This tool requires calibration data to score chains.",
        }

    if properties is None:
        return {
            "status": "error",
            "message": f"No calibration data available for {backend_name}. "
            "This is likely a simulator backend.",
        }

    return properties


@with_sync
async def find_optimal_qv_qubits(
    backend_name: str,
    num_qubits: int = 5,
    num_results: int = 5,
    metric: QVScoringMetric = "qv_optimized",
) -> dict[str, Any]:
    """
    Find optimal qubit subgraphs for Quantum Volume experiments.

    Unlike linear chains, Quantum Volume benefits from densely connected qubit sets
    where any qubit can interact with any other with minimal SWAP operations.
    This tool finds connected subgraphs and ranks them by connectivity and
    calibration quality.

    Args:
        backend_name: Name of the backend (e.g., 'ibm_brisbane')
        num_qubits: Number of qubits in the subgraph (default: 5, range: 2-10)
        num_results: Number of top subgraphs to return (default: 5, max: 20)
        metric: Scoring metric to optimize:
            - "qv_optimized": Balanced scoring for QV (connectivity + errors + coherence)
            - "connectivity": Maximize internal edges and minimize path lengths
            - "gate_error": Minimize total two-qubit gate errors on internal edges

    Returns:
        Ranked subgraphs with detailed metrics including:
        - qubits: List of qubit indices in the subgraph (sorted)
        - score: Total score (lower is better)
        - internal_edges: Number of edges within the subgraph
        - connectivity_ratio: internal_edges / max_possible_edges
        - average_path_length: Mean shortest path between qubit pairs
        - qubit_details: T1, T2, readout_error for each qubit
        - edge_errors: Two-qubit gate error for each internal edge
    """
    global service

    try:
        # Validate inputs - limit num_qubits to 10 for performance
        num_qubits = _clamp(num_qubits, 2, 10)
        num_results = _clamp(num_results, 1, 20)

        # Check for fake backends early
        if backend_name.startswith("fake_"):
            return {
                "status": "error",
                "message": f"Fake backends like '{backend_name}' are not supported. "
                "This tool requires real-time calibration data from IBM Quantum backends.",
            }

        # Initialize service if needed
        if service is None:
            service = initialize_service()

        # Get coupling map
        coupling_result = await get_coupling_map(backend_name)
        if coupling_result["status"] == "error":
            return coupling_result

        adjacency_list = coupling_result["adjacency_list"]
        backend_num_qubits = coupling_result["num_qubits"]

        # Validate requested size against backend capacity
        if num_qubits > backend_num_qubits:
            return {
                "status": "error",
                "message": f"Requested {num_qubits} qubits, but {backend_name} only has "
                f"{backend_num_qubits} qubits. Please reduce num_qubits.",
            }

        # Get backend properties
        backend = service.backend(backend_name)
        properties = _get_backend_properties_for_chains(backend, backend_name)
        if isinstance(properties, dict):
            return properties  # Error response

        # Extract faulty qubits
        faulty_qubits: set[int] = set()
        with contextlib.suppress(Exception):
            faulty_qubits = set(properties.faulty_qubits())

        # Build calibration data
        qubit_calibration = _build_qubit_calibration(
            properties, backend_num_qubits, faulty_qubits
        )
        gate_errors = _build_gate_errors(properties, coupling_result["edges"])

        # Find all connected subgraphs (prioritizing qubits with better calibration)
        subgraphs = _find_connected_subgraphs(
            adjacency_list, num_qubits, faulty_qubits, qubit_calibration
        )

        if not subgraphs:
            return {
                "status": "error",
                "message": f"No valid connected subgraphs of size {num_qubits} found on {backend_name}. "
                f"The backend has {backend_num_qubits} qubits and {len(faulty_qubits)} faulty qubits.",
            }

        # Score and rank subgraphs
        scored_subgraphs = [
            (
                sg,
                _score_qv_subgraph(
                    sg, adjacency_list, qubit_calibration, gate_errors, metric
                ),
            )
            for sg in subgraphs
        ]
        scored_subgraphs.sort(key=lambda x: x[1])
        top_subgraphs = scored_subgraphs[:num_results]

        # Build detailed results
        results = [
            _build_qv_subgraph_result(
                rank, sg, score, adjacency_list, qubit_calibration, gate_errors
            )
            for rank, (sg, score) in enumerate(top_subgraphs, 1)
        ]

        return {
            "status": "success",
            "backend_name": backend_name,
            "num_qubits": num_qubits,
            "metric": metric,
            "total_subgraphs_found": len(subgraphs),
            "faulty_qubits": list(faulty_qubits),
            "subgraphs": results,
        }

    except Exception as e:
        logger.error(f"Failed to find optimal QV qubits: {e}")
        return {
            "status": "error",
            "message": f"Failed to find optimal QV qubits: {e!s}",
        }


def _get_sampler_backend(
    svc: QiskitRuntimeService, backend_name: str | None
) -> tuple[Any | None, str | None]:
    """Get the backend for sampler execution.

    Returns:
        Tuple of (backend, error_message). If successful, error_message is None.
    """
    if backend_name:
        try:
            return svc.backend(backend_name), None
        except Exception as e:
            return None, f"Failed to get backend '{backend_name}': {e!s}"

    # Find least busy backend
    backends = svc.backends(simulator=False)
    backend = least_busy(backends)
    if backend is None:
        return (
            None,
            "No operational backend available. Please specify a backend_name or try again later.",
        )
    return backend, None


@with_sync
async def run_sampler(
    circuit: str,
    backend_name: str | None = None,
    shots: int = 4096,
    circuit_format: CircuitFormat = "auto",
    dynamical_decoupling: bool = True,
    dd_sequence: DDSequenceType = "XY4",
    twirling: bool = True,
    measure_twirling: bool = True,
) -> dict[str, Any]:
    """
    Run a quantum circuit using the Qiskit Runtime SamplerV2 primitive.

    The Sampler primitive returns measurement outcome samples from circuit execution.
    This is useful for algorithms that need to sample from probability distributions,
    such as variational algorithms, quantum machine learning, and quantum simulation.

    Error Mitigation:
        This function includes built-in error mitigation techniques enabled by default:
        - Dynamical Decoupling (DD): Suppresses decoherence during idle periods
        - Twirling: Randomizes errors to improve measurement accuracy

    Args:
        circuit: The quantum circuit to execute. Accepts:
                - OpenQASM 3.0 string (recommended)
                - OpenQASM 2.0 string (legacy, auto-detected)
                - Base64-encoded QPY binary (for tool chaining)
                The circuit must include measurement operations to produce results.
        backend_name: Name of the IBM Quantum backend to use (e.g., 'ibm_brisbane').
                     If not provided, uses the least busy operational backend.
        shots: Number of measurement shots (repetitions) per circuit. Default is 4096.
               Maximum depends on the backend (typically 8192 or higher).
        circuit_format: Format of the circuit input. Options:
                       - "auto" (default): Automatically detect format
                       - "qasm3": OpenQASM 3.0/2.0 text format
                       - "qpy": Base64-encoded QPY binary format
        dynamical_decoupling: Enable dynamical decoupling to suppress decoherence
                             during idle periods in the circuit. Default is True.
        dd_sequence: Type of dynamical decoupling sequence to use. Options:
                    - "XX": Basic X-X sequence
                    - "XpXm": X+/X- sequence with better noise suppression
                    - "XY4": Most robust 4-pulse sequence (default, recommended)
        twirling: Enable Pauli twirling on 2-qubit gates to convert coherent
                 errors into stochastic noise. Default is True.
        measure_twirling: Enable twirling on measurement operations for improved
                         readout error mitigation. Default is True.

    Returns:
        Dictionary containing:
        - status: "success" or "error"
        - job_id: The ID of the submitted job (can be used to check status later)
        - backend: Name of the backend used
        - shots: Number of shots executed
        - execution_mode: "job" (direct execution)
        - error_mitigation: Summary of enabled error mitigation techniques
        - message: Status message indicating job was submitted
        - note: Information about how to retrieve results

    Note:
        This function submits the job and returns immediately. For long-running jobs,
        use get_job_status_tool to check completion, then retrieve results separately.
        Results include measurement outcomes as bitstrings with their counts.
    """
    global service

    try:
        if service is None:
            service = initialize_service()

        # Load the circuit using the shared serialization module
        load_result = load_circuit(circuit, circuit_format)
        if load_result["status"] == "error":
            return {"status": "error", "message": load_result["message"]}
        qc = load_result["circuit"]

        # Get the backend
        backend, backend_error = _get_sampler_backend(service, backend_name)
        if backend_error:
            return {"status": "error", "message": backend_error}
        assert backend is not None  # Type narrowing for mypy  # nosec B101

        # Validate shots
        if shots < 1:
            return {"status": "error", "message": "shots must be at least 1"}

        # Configure error mitigation options
        options = SamplerOptions()

        # Dynamical Decoupling - suppresses decoherence during idle periods
        options.dynamical_decoupling.enable = dynamical_decoupling
        if dynamical_decoupling:
            options.dynamical_decoupling.sequence_type = dd_sequence

        # Twirling - randomizes errors to convert coherent errors to stochastic noise
        options.twirling.enable_gates = twirling
        options.twirling.enable_measure = measure_twirling

        # Build error mitigation summary for response
        error_mitigation: dict[str, Any] = {
            "dynamical_decoupling": {
                "enabled": dynamical_decoupling,
                "sequence": dd_sequence if dynamical_decoupling else None,
            },
            "twirling": {
                "gates_enabled": twirling,
                "measure_enabled": measure_twirling,
            },
        }

        # Create SamplerV2 with options and run
        sampler = SamplerV2(mode=backend, options=options)
        job = sampler.run([qc], shots=shots)

        return {
            "status": "success",
            "job_id": job.job_id(),
            "backend": backend.name,
            "shots": shots,
            "execution_mode": "job",
            "error_mitigation": error_mitigation,
            "message": f"Sampler job submitted successfully to {backend.name}",
            "note": "Use get_job_status_tool with the job_id to check completion. "
            "Results will contain measurement bitstrings and their counts.",
        }

    except Exception as e:
        logger.error(f"Failed to run sampler: {e}")
        return {"status": "error", "message": f"Failed to run sampler: {e!s}"}


def get_bell_state_circuit() -> dict[str, Any]:
    """Get a Bell state (maximally entangled 2-qubit) circuit in QASM3 format.

    The Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2 is created by applying a Hadamard gate
    to the first qubit followed by a CNOT gate. This is the simplest demonstration
    of quantum entanglement.

    Returns:
        Dictionary containing:
        - circuit: QASM3 string ready to use with run_sampler_tool
        - description: Explanation of the circuit
        - expected_results: What measurement outcomes to expect
        - num_qubits: Number of qubits used
    """
    qasm3_circuit = """OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;
bit[2] c;

// Create Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2
h q[0];        // Put first qubit in superposition
cx q[0], q[1]; // Entangle with second qubit

c = measure q;
"""
    return {
        "circuit": qasm3_circuit,
        "name": "Bell State",
        "description": "Creates the Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2, "
        "demonstrating quantum entanglement between two qubits.",
        "expected_results": "Approximately 50% '00' and 50% '11' outcomes. "
        "Never '01' or '10' due to entanglement.",
        "num_qubits": 2,
        "usage": "Pass the 'circuit' field directly to run_sampler_tool",
    }


def get_ghz_state_circuit(num_qubits: int = 3) -> dict[str, Any]:
    """Get a GHZ (Greenberger-Horne-Zeilinger) state circuit in QASM3 format.

    The GHZ state is a maximally entangled state of N qubits:
    |GHZ⟩ = (|00...0⟩ + |11...1⟩)/√2

    This generalizes the Bell state to more qubits and is useful for
    demonstrating multi-qubit entanglement.

    Args:
        num_qubits: Number of qubits for the GHZ state (2-10). Default is 3.

    Returns:
        Dictionary containing:
        - circuit: QASM3 string ready to use with run_sampler_tool
        - description: Explanation of the circuit
        - expected_results: What measurement outcomes to expect
        - num_qubits: Number of qubits used
    """
    # Validate num_qubits
    if num_qubits < 2:
        num_qubits = 2
    elif num_qubits > 10:
        num_qubits = 10

    # Build the circuit
    lines = [
        "OPENQASM 3.0;",
        'include "stdgates.inc";',
        f"qubit[{num_qubits}] q;",
        f"bit[{num_qubits}] c;",
        "",
        f"// Create {num_qubits}-qubit GHZ state",
        "h q[0];  // Put first qubit in superposition",
    ]

    # Add CNOT cascade
    lines.extend(
        f"cx q[{i}], q[{i + 1}];  // Entangle qubit {i} with {i + 1}"
        for i in range(num_qubits - 1)
    )

    lines.extend(["", "c = measure q;", ""])

    qasm3_circuit = "\n".join(lines)

    all_zeros = "0" * num_qubits
    all_ones = "1" * num_qubits

    return {
        "circuit": qasm3_circuit,
        "name": f"{num_qubits}-qubit GHZ State",
        "description": f"Creates the {num_qubits}-qubit GHZ state "
        f"|GHZ⟩ = (|{all_zeros}⟩ + |{all_ones}⟩)/√2, "
        "demonstrating multi-qubit entanglement.",
        "expected_results": f"Approximately 50% '{all_zeros}' and 50% '{all_ones}' outcomes. "
        "No other bitstrings should appear due to entanglement.",
        "num_qubits": num_qubits,
        "usage": "Pass the 'circuit' field directly to run_sampler_tool",
    }


def get_quantum_random_circuit() -> dict[str, Any]:
    """Get a simple quantum random number generator circuit in QASM3 format.

    Creates true random bits using quantum superposition. Each qubit is put into
    an equal superposition using a Hadamard gate, then measured. The outcome is
    fundamentally random according to quantum mechanics.

    Returns:
        Dictionary containing:
        - circuit: QASM3 string ready to use with run_sampler_tool
        - description: Explanation of the circuit
        - expected_results: What measurement outcomes to expect
        - num_qubits: Number of qubits used
    """
    qasm3_circuit = """OPENQASM 3.0;
include "stdgates.inc";
qubit[4] q;
bit[4] c;

// Quantum random number generator
// Each qubit produces a truly random bit
h q[0];
h q[1];
h q[2];
h q[3];

c = measure q;
"""
    return {
        "circuit": qasm3_circuit,
        "name": "Quantum Random Number Generator",
        "description": "Generates 4 truly random bits using quantum superposition. "
        "Each Hadamard gate creates a 50/50 superposition that collapses randomly upon measurement.",
        "expected_results": "All 16 possible 4-bit outcomes (0000 to 1111) with roughly equal probability. "
        "Each outcome should appear about 6.25% of the time.",
        "num_qubits": 4,
        "usage": "Pass the 'circuit' field directly to run_sampler_tool. "
        "Use multiple shots to generate many random numbers.",
    }


def get_superposition_circuit() -> dict[str, Any]:
    """Get a simple single-qubit superposition circuit in QASM3 format.

    The simplest possible quantum circuit: puts one qubit in superposition
    using a Hadamard gate. Perfect for testing and learning.

    Returns:
        Dictionary containing:
        - circuit: QASM3 string ready to use with run_sampler_tool
        - description: Explanation of the circuit
        - expected_results: What measurement outcomes to expect
        - num_qubits: Number of qubits used
    """
    qasm3_circuit = """OPENQASM 3.0;
include "stdgates.inc";
qubit[1] q;
bit[1] c;

// Simple superposition: |0⟩ -> (|0⟩ + |1⟩)/√2
h q[0];

c = measure q;
"""
    return {
        "circuit": qasm3_circuit,
        "name": "Single Qubit Superposition",
        "description": "The simplest quantum circuit: applies a Hadamard gate to create "
        "an equal superposition (|0⟩ + |1⟩)/√2.",
        "expected_results": "Approximately 50% '0' and 50% '1' outcomes.",
        "num_qubits": 1,
        "usage": "Pass the 'circuit' field directly to run_sampler_tool. "
        "This is the simplest possible quantum experiment.",
    }


# Assisted by watsonx Code Assistant
