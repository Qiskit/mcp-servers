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
import asyncio
import logging
import os
from collections.abc import Callable, Coroutine
from functools import wraps
from typing import Any, TypeVar

from qiskit.qasm3 import loads  # type: ignore[import-untyped]

from qiskit_ibm_transpiler_mcp_server.qiskit_runtime_service_provider import (
    QiskitRuntimeServiceProvider,
)


logger = logging.getLogger(__name__)

# Apply nest_asyncio to allow running async code in environments with existing event loops
try:
    import nest_asyncio

    nest_asyncio.apply()
except ImportError:
    pass


F = TypeVar("F", bound=Callable[..., Any])
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


def with_sync(func: F) -> F:
    """Decorator that adds a `.sync` attribute to async functions for synchronous execution.

    Usage:
        @with_sync
        async def my_async_function(arg: str) -> Dict[str, Any]:
            ...

        # Async call
        result = await my_async_function("hello")

        # Sync call
        result = my_async_function.sync("hello")
    """

    @wraps(func)
    def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
        return _run_async(func(*args, **kwargs))

    func.sync = sync_wrapper  # type: ignore[attr-defined]
    return func


def load_qasm_circuit(qasm_string: str) -> dict[str, Any]:
    """
    Load Qiskit QuantumCircuit from QASM 3.0 string.
    Args:
        qasm_string: QASM 3.0 string describing input circuit
    """
    try:
        circuit = loads(qasm_string)
        return {"status": "success", "circuit": circuit}
    except Exception as e:
        logger.error(f"Error in loading QuantumCircuit object from QASM 3.0 string: {e}")
        return {
            "status": "error",
            "message": "QASM 3.0 string not valid. Cannot be loaded as QuantumCircuit.",
        }


async def get_backend_service(backend_name: str) -> dict[str, Any]:
    """
    Get the required backend.
    Args:
        backend_name: name of the backend to retrieve
    """
    try:
        # instantiate QiskitRuntimeService through Singleton provider
        service = QiskitRuntimeServiceProvider().get()
        backend = service.backend(backend_name)

        if not backend:
            return {
                "status": "error",
                "message": f"No backend {backend_name} available",
            }

        return {"status": "success", "backend": backend}
    except Exception as e:
        logger.error(f"Failed to find backend {backend_name}: {e}")
        return {
            "status": "error",
            "message": f"Failed to find backend {backend_name}: {e!s}",
        }


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


@with_sync
async def setup_ibm_quantum_account(
    token: str | None = None, channel: str = "ibm_quantum_platform"
) -> dict[str, Any]:
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
        # instantiate QiskitRuntimeService through Singleton provider
        service = QiskitRuntimeServiceProvider().get(
            token=token.strip() if token else None, channel=channel
        )
        backends = service.backends()
        return {
            "status": "success",
            "message": "IBM Quantum account set up successfully",
            "available_backends": len(backends),
        }
    except Exception as e:
        logger.error(f"Failed to set up IBM Quantum account: {e}")
        return {
            "status": "error",
            "message": f"Failed to set up IBM Quantum account: {e!s}",
        }
