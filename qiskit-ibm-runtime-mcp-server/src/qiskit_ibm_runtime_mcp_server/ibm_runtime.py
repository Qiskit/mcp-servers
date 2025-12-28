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
from qiskit_ibm_runtime.options import SamplerOptions
from qiskit_mcp_server.circuit_serialization import CircuitFormat, load_circuit

from qiskit_ibm_runtime_mcp_server.utils import with_sync


# Type alias for dynamical decoupling sequence types
DDSequenceType = Literal["XX", "XpXm", "XY4"]


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
            logger.warning(f"Skipping backend {getattr(b, 'name', 'unknown')} in least_busy: {e}")
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


logger = logging.getLogger(__name__)

# Global service instance
service: QiskitRuntimeService | None = None


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
                logger.info(f"Successfully initialized IBM Runtime service on channel: {channel}")
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
                logger.warning(f"Failed to get status for backend {backend_name}: {status_err}")
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
            logger.warning(f"Could not get final status for {backend.name}: {status_err}")
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
                    gate_errors.append({"gate": gate, "qubits": [qubit], "error": round(error, 6)})

    # Two-qubit gates
    for gate in two_qubit_gates:
        for edge in coupling_map[:5]:
            with contextlib.suppress(Exception):
                error = properties.gate_error(gate, edge)
                if error is not None:
                    gate_errors.append({"gate": gate, "qubits": edge, "error": round(error, 6)})

    return gate_errors


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
                    faulty_gates.append({"gate": gate.gate, "qubits": list(gate.qubits)})

        # Determine which qubits to report on
        if qubit_indices is None:
            qubit_indices = list(range(min(10, num_qubits)))
        else:
            qubit_indices = [q for q in qubit_indices if 0 <= q < num_qubits]

        # Collect qubit calibration data
        qubit_data: list[dict[str, Any]] = []
        for qubit in qubit_indices:
            try:
                qubit_data.append(_get_qubit_calibration_data(properties, qubit, faulty_qubits))
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
                    "error_message": job.error_message() if hasattr(job, "error_message") else None,
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
            "error_message": job.error_message() if hasattr(job, "error_message") else None,
        }

        return job_info

    except Exception as e:
        logger.error(f"Failed to get job status: {e}")
        return {"status": "error", "message": f"Failed to get job status: {e!s}"}


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
        f"cx q[{i}], q[{i + 1}];  // Entangle qubit {i} with {i + 1}" for i in range(num_qubits - 1)
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
