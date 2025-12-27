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

#!/usr/bin/env python3
"""
Qiskit IBM Runtime MCP Server

A Model Context Protocol server that provides access to IBM Quantum services
through Qiskit IBM Runtime, enabling AI assistants to interact with quantum
computing resources.

Dependencies:
- fastmcp
- qiskit-ibm-runtime
- qiskit
- python-dotenv
"""

import logging
from typing import Any

from fastmcp import FastMCP

from qiskit_ibm_runtime_mcp_server.ibm_runtime import (
    active_account_info,
    active_instance_info,
    available_instances,
    cancel_job,
    delete_saved_account,
    get_backend_calibration,
    get_backend_properties,
    get_job_status,
    get_service_status,
    least_busy_backend,
    list_backends,
    list_my_jobs,
    list_saved_account,
    setup_ibm_quantum_account,
    usage_info,
)


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize MCP server
mcp = FastMCP("Qiskit IBM Runtime")


# Tools
@mcp.tool()
async def setup_ibm_quantum_account_tool(
    token: str = "", channel: str = "ibm_quantum_platform"
) -> dict[str, Any]:
    """Set up IBM Quantum account with credentials.

    If token is not provided, will attempt to use QISKIT_IBM_TOKEN environment variable
    or saved credentials from ~/.qiskit/qiskit-ibm.json
    """
    return await setup_ibm_quantum_account(token if token else None, channel)


@mcp.tool()
async def list_backends_tool() -> dict[str, Any]:
    """List available IBM Quantum backends."""
    return await list_backends()


@mcp.tool()
async def least_busy_backend_tool() -> dict[str, Any]:
    """Find the least busy operational backend."""
    return await least_busy_backend()


@mcp.tool()
async def get_backend_properties_tool(backend_name: str) -> dict[str, Any]:
    """Get detailed properties of a specific backend.

    Args:
        backend_name: Name of the backend (e.g., 'ibm_brisbane')

    Returns:
        Backend properties including:
        - num_qubits: Number of qubits on the backend
        - simulator: Whether this is a simulator backend
        - operational: Current operational status
        - pending_jobs: Number of jobs in the queue
        - processor_type: Processor family (e.g., 'Eagle r3', 'Heron')
        - backend_version: Backend software version
        - basis_gates: Native gates supported (e.g., ['cx', 'id', 'rz', 'sx', 'x'])
        - coupling_map: Qubit connectivity as list of [control, target] pairs
        - max_shots: Maximum shots per circuit execution
        - max_experiments: Maximum circuits per job

    Note:
        For time-varying calibration data (T1, T2, gate errors, faulty qubits),
        use get_backend_calibration_tool instead.
    """
    return await get_backend_properties(backend_name)


@mcp.tool()
async def get_backend_calibration_tool(
    backend_name: str, qubit_indices: list[int] | None = None
) -> dict[str, Any]:
    """Get calibration data for a backend including T1, T2 times and error rates.

    Args:
        backend_name: Name of the backend (e.g., 'ibm_brisbane')
        qubit_indices: Optional list of specific qubit indices to get data for.
                      If not provided, returns data for the first 10 qubits.

    Returns:
        Calibration data including:
        - T1 and T2 coherence times (in microseconds)
        - Qubit frequency (in GHz)
        - Readout errors for each qubit
        - Gate errors for common gates (x, sx, cx, etc.)
        - faulty_qubits: List of non-operational qubit indices
        - faulty_gates: List of non-operational gates with affected qubits
        - Last calibration timestamp

    Note:
        For static backend info (processor_type, backend_version, quantum_volume),
        use get_backend_properties_tool instead.
    """
    return await get_backend_calibration(backend_name, qubit_indices)


@mcp.tool()
async def list_my_jobs_tool(limit: int = 10) -> dict[str, Any]:
    """List user's recent jobs."""
    return await list_my_jobs(limit)


@mcp.tool()
async def get_job_status_tool(job_id: str) -> dict[str, Any]:
    """Get status of a specific job."""
    return await get_job_status(job_id)


@mcp.tool()
async def cancel_job_tool(job_id: str) -> dict[str, Any]:
    """Cancel a specific job."""
    return await cancel_job(job_id)


@mcp.tool()
async def delete_saved_account_tool(account_name: str) -> dict[str, Any]:
    """Delete a saved IBM Quantum account from disk.

    WARNING: This permanently removes credentials from ~/.qiskit/qiskit-ibm.json.
    The operation cannot be undone. Use list_saved_account_tool() first to verify
    the account name before deletion.

    Args:
        account_name: Name of the saved account to delete (e.g., 'ibm_quantum_platform')
    """
    return await delete_saved_account(account_name)


@mcp.tool()
async def list_saved_account_tool() -> dict[str, Any]:
    """List all IBM Quantum accounts saved on disk.

    Returns account information from ~/.qiskit/qiskit-ibm.json including account names
    and channels. Useful for checking available accounts before initializing the service
    or before deleting an account.
    """
    return await list_saved_account()


@mcp.tool()
async def active_account_info_tool() -> dict[str, Any]:
    """Get information about the currently active IBM Quantum account.

    Returns details about the account being used in the current session, including
    channel, instance, and name. This is the account used for all quantum operations.
    """
    return await active_account_info()


@mcp.tool()
async def active_instance_info_tool() -> dict[str, Any]:
    """Get the Cloud Resource Name (CRN) of the currently active instance.

    Returns the instance identifier determining which quantum backends and resources
    are accessible. Important for users with access to multiple instances.
    """
    return await active_instance_info()


@mcp.tool()
async def available_instances_tool() -> dict[str, Any]:
    """List all IBM Quantum instances available to the active account.

    Returns information about all instances (organizations, projects, or service plans)
    the user has access to, including CRN, plan type, and name. Each instance provides
    access to different quantum backends with different quotas.
    """
    return await available_instances()


@mcp.tool()
async def usage_info_tool() -> dict[str, Any]:
    """Get usage statistics and quota information for the active instance.

    Returns detailed metrics including job counts, quantum runtime consumption,
    quota limits, and billing period information. Useful for monitoring resource
    utilization and planning job submissions.
    """
    return await usage_info()


# Resources
@mcp.resource("ibm://status", mime_type="text/plain")
async def get_service_status_resource() -> str:
    """Get current IBM Quantum service status."""
    return await get_service_status()


def main() -> None:
    """Run the server."""
    mcp.run()


if __name__ == "__main__":
    main()


# Assisted by watsonx Code Assistant
