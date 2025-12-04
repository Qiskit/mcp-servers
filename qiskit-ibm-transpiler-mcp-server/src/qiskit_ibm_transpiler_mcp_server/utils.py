from qiskit.qasm3 import loads  # type: ignore[import-untyped]
from typing import Any


from qiskit_ibm_transpiler_mcp_server.qiskit_runtime_service_provider import (
    QiskitRuntimeServiceProvider,
)

from typing import Optional
import os
import logging

logger = logging.getLogger(__name__)


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
        logger.error(
            f"Error in loading QuantumCircuit object from QASM 3.0 string: {e}"
        )
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
            "message": f"Failed to find backend {backend_name}: {str(e)}",
        }


def get_token_from_env() -> Optional[str]:
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


async def setup_ibm_quantum_account(
    token: Optional[str] = None, channel: str = "ibm_quantum_platform"
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
            "message": f"Failed to set up IBM Quantum account: {str(e)}",
        }
