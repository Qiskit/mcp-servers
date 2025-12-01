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

"""Synchronous wrappers for async IBM Runtime functions.

This module provides synchronous versions of the async functions for use with
frameworks that don't support async operations (like DSPy).
"""

import asyncio
from collections.abc import Coroutine
from typing import Any, TypeVar

from qiskit_ibm_runtime_mcp_server.ibm_runtime import (
    cancel_job,
    get_backend_properties,
    get_job_status,
    get_service_status,
    least_busy_backend,
    list_backends,
    list_my_jobs,
    setup_ibm_quantum_account,
)


# Apply nest_asyncio to allow running async code in environments with existing event loops
try:
    import nest_asyncio

    nest_asyncio.apply()
except ImportError:
    pass


T = TypeVar("T")


def _run_async(coro: Coroutine[Any, Any, T]) -> T:
    """Helper to run async functions synchronously.

    This handles both cases:
    - Running in a Jupyter notebook or other environment with an existing event loop
    - Running in a standard Python script without an event loop
    """
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # We're in a running loop (e.g., Jupyter), use run_until_complete
            # This works because nest_asyncio allows nested loops
            return loop.run_until_complete(coro)
        else:
            return loop.run_until_complete(coro)
    except RuntimeError:
        # No event loop exists, create one
        return asyncio.run(coro)


def setup_ibm_quantum_account_sync(
    token: str = "",
    channel: str = "ibm_quantum_platform",  # nosec B107
) -> dict[str, Any]:
    """Set up IBM Quantum account with credentials.

    Synchronous version of setup_ibm_quantum_account.

    Args:
        token: IBM Quantum API token (optional - will try environment or saved credentials)
        channel: Service channel ('ibm_quantum_platform')

    Returns:
        Setup status and information
    """
    return _run_async(setup_ibm_quantum_account(token if token else None, channel))


def list_backends_sync() -> dict[str, Any]:
    """List available IBM Quantum backends.

    Synchronous version of list_backends.

    Returns:
        List of backends with their properties
    """
    return _run_async(list_backends())


def least_busy_backend_sync() -> dict[str, Any]:
    """Find the least busy operational backend.

    Synchronous version of least_busy_backend.

    Returns:
        Information about the least busy backend
    """
    return _run_async(least_busy_backend())


def get_backend_properties_sync(backend_name: str) -> dict[str, Any]:
    """Get detailed properties of a specific backend.

    Synchronous version of get_backend_properties.

    Args:
        backend_name: Name of the backend

    Returns:
        Backend properties and capabilities
    """
    return _run_async(get_backend_properties(backend_name))


def list_my_jobs_sync(limit: int = 10) -> dict[str, Any]:
    """List user's recent jobs.

    Synchronous version of list_my_jobs.

    Args:
        limit: Maximum number of jobs to retrieve

    Returns:
        List of jobs with their information
    """
    return _run_async(list_my_jobs(limit))


def get_job_status_sync(job_id: str) -> dict[str, Any]:
    """Get status of a specific job.

    Synchronous version of get_job_status.

    Args:
        job_id: ID of the job

    Returns:
        Job status information
    """
    return _run_async(get_job_status(job_id))


def cancel_job_sync(job_id: str) -> dict[str, Any]:
    """Cancel a specific job.

    Synchronous version of cancel_job.

    Args:
        job_id: ID of the job to cancel

    Returns:
        Cancellation status
    """
    return _run_async(cancel_job(job_id))


def get_service_status_sync() -> str:
    """Get current IBM Quantum service status.

    Synchronous version of get_service_status.

    Returns:
        Service connection status and basic information
    """
    return _run_async(get_service_status())


# Assisted by watsonx Code Assistant
