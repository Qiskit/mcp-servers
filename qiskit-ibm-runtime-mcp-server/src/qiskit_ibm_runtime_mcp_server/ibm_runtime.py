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
from typing import Any

from qiskit_ibm_runtime import QiskitRuntimeService

from qiskit_ibm_runtime_mcp_server.utils import with_sync


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
                    "error_message": (
                        job.error_message() if hasattr(job, "error_message") else None
                    ),
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
            "error_message": (job.error_message() if hasattr(job, "error_message") else None),
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


@with_sync
async def delete_saved_account(account_name: str = "") -> dict[str, Any]:
    """
    Delete a saved IBM Quantum account from disk.

    **WARNING: DESTRUCTIVE OPERATION**
    This permanently removes saved credentials on ~/.qiskit/qiskit-ibm.json.
    The operation CANNOT be undone. Deleted credentials must be re-entered to restore access.

    Args:
        account_name: Name of the saved account to delete. If empty string (default),
                     deletes the default account. Account names are typically in the
                     format 'ibm_quantum_platform' or custom names set during save.

    Returns:
        Dictionary containing deletion status:
        - On success: {"status": "success", "deleted": True, "message": "Account successfully deleted"}
        - On failure: {"status": "error", "deleted": False, "error": "Account name not found or could not be deleted"}
        - On error: {"status": "error", "deleted": False, "error": error_message}

    """
    global service

    try:
        if service is None:
            service = initialize_service()
        is_deleted = service.delete_account(name=account_name)

        if is_deleted:
            return {
                "status": "success",
                "deleted": True,
                "message": "Account successfully deleted",
            }
        else:
            return {
                "status": "error",
                "deleted": False,
                "error": "Account name not found or could not be deleted",
            }

    except Exception as e:
        logger.error(f"Failed to delete account: {e}")
        return {"status": "error", "deleted": False, "error": str(e)}


@with_sync
async def list_saved_account() -> dict[str, Any]:
    """
    List all IBM Quantum accounts saved on disk.

    Retrieves account information from the local configuration file without requiring authentication.
    Useful for checking which accounts are available before initializing
    the service.

    Returns:
        Dictionary containing account list status:
        - On success with accounts: {"status": "success", "accounts": [list of account dicts]}
          Each account dict contains: name, channel, and other metadata
        - On success with no accounts: {"status": "success", "accounts": "No accounts found"}
        - On error: {"status": "error", "error": error_message}
    """
    global service

    try:
        if service is None:
            service = initialize_service()
        accounts_list = service.saved_accounts()
        if len(accounts_list) > 0:
            return {"status": "success", "accounts": accounts_list}
        else:
            return {"status": "success", "accounts": "No accounts found"}
    except Exception as e:
        logger.error(f"Failed to collect accounts: {e}")
        return {"status": "error", "error": str(e)}


@with_sync
async def active_account_info() -> dict[str, Any]:
    """
    Get information about the IBM Quantum account currently active in this session.

    Returns details about the account being used by the current QiskitRuntimeService
    instance, including channel, instance, and authentication status. This is the
    account that will be used for all quantum operations (backend access, job submission).

    Returns:
        Dictionary containing active account information:
        - On success: {"status": "success", "account_info": account_info}
        account_info including:
          * "channel": Service channel (e.g., 'ibm_quantum')
          * "url"
          * "token"
          * "verify"
          * "private_endpoint"
        - On error: {"status": "error" "error": error_message}
    """
    global service

    try:
        if service is None:
            service = initialize_service()
        account_info = service.active_account()
        account_data = {"status": "success", "account_info": account_info}
        return account_data
    except Exception as e:
        logger.error(f"Failed to collect active account: {e}")
        return {"status": "error", "error": str(e)}


@with_sync
async def active_instance_info() -> dict[str, Any]:
    """
    Get the Cloud Resource Name (CRN) of the currently active IBM Quantum instance.

    Returns the instance identifier being used for the current session. The instance
    determines which quantum backends and resources are accessible. This is particularly
    important for users with access to multiple instances (e.g., different organizations
    or service plans).

    Returns:
        String containing the instance CRN (Cloud Resource Name), or error dict:
        - On success: Instance CRN string (e.g., "crn:v1:bluemix:public:quantum-computing:...")
        - On error: {"status": "error" "error": error_message}
    """
    global service

    try:
        if service is None:
            service = initialize_service()
        intance_details = {
            "status": "success",
            "instance_crn": service.active_instance(),
        }
        return intance_details
    except Exception as e:
        logger.error(f"Failed to collect instance crn: {e}")
        return {"status": "error", "error": str(e)}


@with_sync
async def available_instances() -> dict[str, Any]:
    """
    List all IBM Quantum instances available to the active account.

    Retrieves information about all instances (organizations, projects, or service plans)
    that the authenticated user has access to. Each instance provides access to different
    quantum backends and may have different quotas and permissions.

    Returns:
        Dictionary containing instance list or error:
        - On success: {"status": "success", "instances": [list of instance dicts]}
          Each instance dict contains:
          * "crn": Cloud Resource Name (unique instance identifier)
          * "plan": Service plan type (e.g., "open", "premium")
          * "name": Human-readable instance name
          * "tags"
          * "pricing_type"
        - On error: {"status": "error", "error": error_message}
    """
    global service

    try:
        if service is None:
            service = initialize_service()
        instances = service.instances()
        instance_data = {
            "status": "success",
            "instances": list(instances),
            "total_instances": len(instances),
        }
        return instance_data
    except Exception as e:
        logger.error(f"Failed to collect available instances for the active account: {e}")
        return {"status": "error", "error": str(e)}


@with_sync
async def usage_info() -> dict[str, Any]:
    """
    Get usage information and statistics for the currently active IBM Quantum instance.

    Retrieves detailed usage metrics including job counts, quantum runtime consumption,
    and quota information for the active instance. This helps track resource utilization
    and monitor remaining quotas.

    Returns:
        Dictionary containing usage statistics, or error dict:
        - On success:  {"status": "success" "usage": usage_info}
        usage_info with fields such as:
          * instance_id
          * plan_id
          * usage_consumed_seconds
          * usage_period
          * usage_limit_seconds
          * usage_limit_reached
          * usage_remaining_seconds
        - On error: {"status": "error" "error": error_message}
    """
    global service

    try:
        if service is None:
            service = initialize_service()
        usage = service.usage()
        usage_info = {"status": "success", "usage": usage}
        return usage_info
    except Exception as e:
        logger.error(f"Failed to collect usage information: {e}")
        return {"status": "error", "error": str(e)}


# Assisted by watsonx Code Assistant
