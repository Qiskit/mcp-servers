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
import logging
import os
from typing import Any

# Import shared utilities from qiskit-mcp-server
from qiskit_mcp_server import with_sync
from qiskit_mcp_server.circuit_serialization import (
    CircuitFormat,
    detect_circuit_format,
    dump_circuit,
    dump_qasm_circuit,
    dump_qpy_circuit,
    load_circuit,
    load_qasm_circuit,
    load_qpy_circuit,
)

from qiskit_ibm_transpiler_mcp_server.qiskit_runtime_service_provider import (
    QiskitRuntimeServiceProvider,
)


logger = logging.getLogger(__name__)

# Placeholder tokens that should be rejected during validation
INVALID_PLACEHOLDER_TOKENS = frozenset(["<PASSWORD>", "<TOKEN>", "YOUR_TOKEN_HERE", "xxx"])

# Re-export shared utilities for backwards compatibility
__all__ = [
    "CircuitFormat",
    "detect_circuit_format",
    "dump_circuit",
    "dump_qasm_circuit",
    "dump_qpy_circuit",
    "get_backend_service",
    "get_token_from_env",
    "load_circuit",
    "load_qasm_circuit",
    "load_qpy_circuit",
    "setup_ibm_quantum_account",
    "with_sync",
]


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
    if token and token.strip():
        stripped = token.strip()
        # Reject tokens that are all the same character (e.g., "xxxx", "0000")
        # as these are likely placeholder values
        if len(set(stripped)) == 1 or stripped in INVALID_PLACEHOLDER_TOKENS:
            return None
        return stripped
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
