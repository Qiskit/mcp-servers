"""Synchronous wrappers for async IBM Transpiler functions.

This module provides synchronous versions of the async functions for use with
frameworks that don't support async operations (like DSPy).
"""

import asyncio  # type: ignore[import-untyped]
from typing import Any, Literal
from qiskit_ibm_transpiler_mcp_server.qta import (
    ai_routing,
    ai_clifford_synthesis,
    ai_linear_function_synthesis,
    ai_permutation_synthesis,
    ai_pauli_network_synthesis,
)

from qiskit_ibm_transpiler_mcp_server.utils import setup_ibm_quantum_account

import logging

logger = logging.getLogger(__name__)

# Apply nest_asyncio to allow running async code in environments with existing event loops
try:
    import nest_asyncio  # type: ignore[import-not-found]

    nest_asyncio.apply()
except ImportError:
    pass


def _run_async(coro):
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
    token: str = "", channel: str = "ibm_quantum_platform"
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


def ai_routing_sync(
    circuit_qasm: str,
    backend_name: str,
    optimization_level: int = 1,
    layout_mode: str = "optimize",
    optimization_preferences: Literal[
        "n_cnots", "n_gates", "cnot_layers", "layers", "noise"
    ]
    | list[Literal["n_cnots", "n_gates", "cnot_layers", "layers", "noise"]]
    | None = None,
    local_mode: bool = True,
) -> dict[str, Any]:
    """
    Route input quantum circuit. It inserts SWAP operations on a circuit to make two-qubits operations compatible with a given coupling map that restricts the pair of qubits on which operations can be applied.
    It should be used as an initial step before any other AI synthesis routine.
    Synchronous version of ai_routing.

    Args:
        circuit_qasm: quantum circuit as QASM string to be synthesized.
        backend_name: Qiskit Runtime Service backend name on which to map the input circuit synthesis
        optimization_level: The potential optimization level to apply during the transpilation process. Valid values are [1,2,3], where 1 is the least optimization (and fastest), and 3 the most optimization (and most time-intensive)
        layout_mode: specifies how to handle the layout selection. It can assume the following values:
            - keep: This respects the layout set by the previous transpiler passes. Typically used when the circuit must be run on specific qubits of the device. It often produces worse results because it has less room for optimization.
            - improve: It is useful when you have a good initial guess for the layout
            - optimize: This is the default mode. It works best for general circuits where you might not have good layout guesses. This mode ignores previous layout selections.
        optimization_preferences: indicates what you want to reduce through optimization: number of cnot gates (n_cnots), number of gates (n_gates), number of cnots layers (cnot_layers), number of layers (layers), and/or noise (noise)
        local_mode: determines where the AIRouting pass runs. If False, AIRouting runs remotely through the Qiskit Transpiler Service. If True, the package tries to run the pass in your local environment with a fallback to cloud mode if the required dependencies are not found
    """

    return _run_async(ai_routing(**locals()))


def ai_clifford_synthesis_sync(
    circuit_qasm: str,
    backend_name: str,
    replace_only_if_better: bool = True,
    local_mode: bool = True,
) -> dict[str, Any]:
    """
    Synthesis for Clifford circuits (blocks of H, S, and CX gates) from the given QASM string.. Currently, up to nine qubit blocks.

    Synchronous version of ai_clifford_synthesis.
    Args:
        circuit_qasm: quantum circuit as QASM string to be synthesized.
        backend_name: Qiskit Runtime Service backend name on which to map the input circuit synthesis
        replace_only_if_better: By default, the synthesis will replace the original sub-circuit only if the synthesized sub-circuit improves the original (currently only checking CNOT count), but this can be forced to always replace the circuit by setting replace_only_if_better=False
        local_mode: determines where the AI Clifford synthesis runs. If False, AI Clifford synthesis runs remotely through the Qiskit Transpiler Service. If True, the package tries to run the pass in your local environment with a fallback to cloud mode if the required dependencies are not found
    """
    return _run_async(ai_clifford_synthesis(**locals()))


def ai_linear_function_synthesis_sync(
    circuit_qasm: str,
    backend_name: str,
    replace_only_if_better: bool = True,
    local_mode: bool = True,
) -> dict[str, Any]:
    """
    Synthesis for Linear Function circuits (blocks of CX and SWAP gates). Currently, up to nine qubit blocks.

    Synchronous version of ai_linear_function_synthesis.
    Args:
        circuit_qasm: quantum circuit as QASM string to be synthesized.
        backend_name: Qiskit Runtime Service backend name on which to map the input circuit synthesis
        replace_only_if_better: By default, the synthesis will replace the original sub-circuit only if the synthesized sub-circuit improves the original (currently only checking CNOT count), but this can be forced to always replace the circuit by setting replace_only_if_better=False
        local_mode: determines where the Linear Function synthesis pass runs. If False, Linear Function synthesis runs remotely through the Qiskit Transpiler Service. If True, the package tries to run the pass in your local environment with a fallback to cloud mode if the required dependencies are not found
    """
    return _run_async(ai_linear_function_synthesis(**locals()))


def ai_permutation_synthesis_sync(
    circuit_qasm: str,
    backend_name: str,
    replace_only_if_better: bool = True,
    local_mode: bool = True,
) -> dict[str, Any]:
    """
    Synthesis for Permutation circuits (blocks of SWAP gates). Currently available for 65, 33, and 27 qubit blocks.

    Synchronous version of ai_permutation_synthesis.
    Args:
        circuit_qasm: quantum circuit as QASM string to be synthesized.
        backend_name: Qiskit Runtime Service backend name on which to map the input circuit synthesis
        replace_only_if_better: By default, the synthesis will replace the original sub-circuit only if the synthesized sub-circuit improves the original (currently only checking CNOT count), but this can be forced to always replace the circuit by setting replace_only_if_better=False
        local_mode: determines where the AI Permutation synthesis pass runs. If False, AI Permutation synthesis runs remotely through the Qiskit Transpiler Service. If True, the package tries to run the pass in your local environment with a fallback to cloud mode if the required dependencies are not found
    """
    return _run_async(ai_permutation_synthesis(**locals()))


def ai_pauli_network_synthesis_sync(
    circuit_qasm: str,
    backend_name: str,
    replace_only_if_better: bool = True,
    local_mode: bool = True,
) -> dict[str, Any]:
    """
    Synthesis for Pauli Network circuits (blocks of H, S, SX, CX, RX, RY and RZ gates). Currently up to six qubit blocks.

    Synchronous version of ai_pauli_network_synthesis.
    Args:
        circuit_qasm: quantum circuit as QASM string to be synthesized.
        backend_name: Qiskit Runtime Service backend name on which to map the input circuit synthesis
        replace_only_if_better: By default, the synthesis will replace the original sub-circuit only if the synthesized sub-circuit improves the original (currently only checking CNOT count), but this can be forced to always replace the circuit by setting replace_only_if_better=False
        local_mode: determines where the AI Pauli Network synthesis pass runs. If False, AI Pauli Network synthesis runs remotely through the Qiskit Transpiler Service. If True, the package tries to run the pass in your local environment with a fallback to cloud mode if the required dependencies are not found
    """
    return _run_async(ai_pauli_network_synthesis(**locals()))
