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

"""Circuit synthesis functions for qiskit-gym MCP server.

This module provides functions to synthesize optimal quantum circuits
using trained RL models for:
- Permutation synthesis (SWAP routing)
- Linear function synthesis (CNOT optimization)
- Clifford synthesis
"""

import base64
import io
import logging
from typing import Any

import numpy as np
from qiskit import QuantumCircuit, qpy

from qiskit_gym_mcp_server.constants import QISKIT_GYM_MAX_SEARCHES
from qiskit_gym_mcp_server.state import GymStateProvider
from qiskit_gym_mcp_server.utils import with_sync


logger = logging.getLogger(__name__)


def _circuit_to_qpy(circuit: QuantumCircuit) -> str:
    """Convert a QuantumCircuit to base64-encoded QPY format.

    Args:
        circuit: Qiskit QuantumCircuit

    Returns:
        Base64-encoded QPY string
    """
    buffer = io.BytesIO()
    qpy.dump(circuit, buffer)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _get_circuit_metrics(circuit: QuantumCircuit) -> dict[str, Any]:
    """Extract metrics from a quantum circuit.

    Args:
        circuit: Qiskit QuantumCircuit

    Returns:
        Dict with circuit metrics
    """
    # Count gates by type
    gate_counts: dict[str, int] = {}
    for instruction in circuit.data:
        gate_name = instruction.operation.name
        gate_counts[gate_name] = gate_counts.get(gate_name, 0) + 1

    # Count two-qubit gates
    two_qubit_gates = sum(
        count
        for name, count in gate_counts.items()
        if name.lower() in ["cx", "cz", "swap", "cnot", "iswap", "ecr"]
    )

    return {
        "num_qubits": circuit.num_qubits,
        "depth": circuit.depth(),
        "size": circuit.size(),
        "gate_counts": gate_counts,
        "two_qubit_gates": two_qubit_gates,
    }


def _validate_model(model_id: str, expected_type: str) -> tuple[Any, dict[str, Any] | None]:
    """Validate a model exists and is of the expected type.

    Args:
        model_id: Model ID
        expected_type: Expected environment type

    Returns:
        Tuple of (model, error_dict) where error_dict is None on success
    """
    state = GymStateProvider()
    model = state.get_model(model_id)

    if model is None:
        return None, {
            "status": "error",
            "message": f"Model '{model_id}' not found. Use list_loaded_models to see available models.",
        }

    if model.env_type != expected_type:
        return None, {
            "status": "error",
            "message": f"Model '{model_id}' is for {model.env_type}, not {expected_type}",
        }

    return model, None


# ============================================================================
# Permutation Synthesis
# ============================================================================


@with_sync
async def synthesize_permutation(
    model_id: str,
    permutation: list[int],
    num_searches: int = 1000,
    deterministic: bool = False,
) -> dict[str, Any]:
    """Synthesize an optimal quantum circuit for a qubit permutation.

    Uses a trained PermutationGym model to find an optimal SWAP gate sequence
    that implements the desired qubit permutation on the coupling map.

    Args:
        model_id: ID of a loaded PermutationGym model
        permutation: Target permutation as list of qubit indices.
            Example: [2, 0, 1] means qubit 0 -> position 2, qubit 1 -> position 0, etc.
        num_searches: Number of search attempts. Higher = better results but slower.
        deterministic: If True, use deterministic action selection

    Returns:
        Dict with synthesized circuit and metrics
    """
    try:
        # Validate model
        model, error = _validate_model(model_id, "permutation")
        if error:
            return error

        # Validate permutation
        perm_array = np.array(permutation)
        expected_size = model.num_qubits

        if len(perm_array) != expected_size:
            return {
                "status": "error",
                "message": f"Permutation has {len(perm_array)} elements but model expects {expected_size}",
            }

        if set(perm_array) != set(range(expected_size)):
            return {
                "status": "error",
                "message": f"Permutation must contain each index 0 to {expected_size - 1} exactly once",
            }

        # Validate num_searches
        if num_searches > QISKIT_GYM_MAX_SEARCHES:
            return {
                "status": "error",
                "message": f"num_searches ({num_searches}) exceeds maximum ({QISKIT_GYM_MAX_SEARCHES})",
            }

        # Synthesize circuit
        logger.info(f"Synthesizing permutation with {num_searches} searches")
        circuit = model.rls_instance.synth(
            perm_array,
            num_searches=num_searches,
            deterministic=deterministic,
        )

        # Handle case where synthesis fails to find a solution
        if circuit is None:
            return {
                "status": "error",
                "message": "Synthesis failed to find a solution. Try increasing num_searches or training the model longer.",
            }

        # Get metrics
        metrics = _get_circuit_metrics(circuit)

        return {
            "status": "success",
            "circuit_qpy": _circuit_to_qpy(circuit),
            "permutation": permutation,
            "metrics": metrics,
            "num_searches": num_searches,
            "note": "Use convert_qpy_to_qasm3 to view circuit in human-readable format",
        }

    except Exception as e:
        logger.error(f"Permutation synthesis failed: {e}")
        return {"status": "error", "message": str(e)}


# ============================================================================
# Linear Function Synthesis
# ============================================================================


@with_sync
async def synthesize_linear_function(
    model_id: str,
    linear_function: list[list[int]],
    num_searches: int = 1000,
    deterministic: bool = False,
) -> dict[str, Any]:
    """Synthesize an optimal quantum circuit for a linear Boolean function.

    Uses a trained LinearFunctionGym model to find an optimal CNOT circuit
    that implements the given linear function.

    Args:
        model_id: ID of a loaded LinearFunctionGym model
        linear_function: Matrix representation of the linear function.
            An NxN binary matrix where entry [i][j] indicates if output i
            depends on input j (XOR relationship).
        num_searches: Number of search attempts
        deterministic: If True, use deterministic action selection

    Returns:
        Dict with synthesized circuit and metrics
    """
    try:
        # Validate model
        model, error = _validate_model(model_id, "linear_function")
        if error:
            return error

        # Convert to numpy array
        lf_array = np.array(linear_function, dtype=np.int32)

        # Validate matrix
        if lf_array.ndim != 2:
            return {
                "status": "error",
                "message": "linear_function must be a 2D matrix",
            }

        if lf_array.shape[0] != lf_array.shape[1]:
            return {
                "status": "error",
                "message": f"linear_function must be square, got {lf_array.shape}",
            }

        if lf_array.shape[0] != model.num_qubits:
            return {
                "status": "error",
                "message": f"Matrix is {lf_array.shape[0]}x{lf_array.shape[0]} but model expects {model.num_qubits}x{model.num_qubits}",
            }

        # Check it's binary
        if not np.all((lf_array == 0) | (lf_array == 1)):
            return {
                "status": "error",
                "message": "linear_function must contain only 0s and 1s",
            }

        # Validate num_searches
        if num_searches > QISKIT_GYM_MAX_SEARCHES:
            return {
                "status": "error",
                "message": f"num_searches ({num_searches}) exceeds maximum ({QISKIT_GYM_MAX_SEARCHES})",
            }

        # Synthesize circuit
        # Convert numpy array to LinearFunction object (required by qiskit-gym synth)
        from qiskit.circuit.library import LinearFunction

        linear_func = LinearFunction(lf_array)
        logger.info(f"Synthesizing linear function with {num_searches} searches")
        circuit = model.rls_instance.synth(
            linear_func,
            num_searches=num_searches,
            deterministic=deterministic,
        )

        # Handle case where synthesis fails to find a solution
        if circuit is None:
            return {
                "status": "error",
                "message": "Synthesis failed to find a solution. Try increasing num_searches or training the model longer.",
            }

        # Get metrics
        metrics = _get_circuit_metrics(circuit)

        return {
            "status": "success",
            "circuit_qpy": _circuit_to_qpy(circuit),
            "linear_function_shape": [int(x) for x in lf_array.shape],
            "metrics": metrics,
            "num_searches": num_searches,
            "note": "Use convert_qpy_to_qasm3 to view circuit in human-readable format",
        }

    except Exception as e:
        logger.error(f"Linear function synthesis failed: {e}")
        return {"status": "error", "message": str(e)}


# ============================================================================
# Clifford Synthesis
# ============================================================================


@with_sync
async def synthesize_clifford(
    model_id: str,
    clifford_tableau: list[list[int]] | dict[str, Any],
    num_searches: int = 1000,
    deterministic: bool = False,
) -> dict[str, Any]:
    """Synthesize an optimal quantum circuit for a Clifford operation.

    Uses a trained CliffordGym model to find an optimal circuit
    implementing the given Clifford element.

    Args:
        model_id: ID of a loaded CliffordGym model
        clifford_tableau: Clifford tableau representation. Can be:
            - A (2N+1) x (2N) binary matrix (standard tableau format)
            - A dict with "destab" and "stab" matrices
        num_searches: Number of search attempts
        deterministic: If True, use deterministic action selection

    Returns:
        Dict with synthesized circuit and metrics
    """
    try:
        from qiskit.quantum_info import Clifford

        # Validate model
        model, error = _validate_model(model_id, "clifford")
        if error:
            return error

        # Parse clifford tableau
        if isinstance(clifford_tableau, dict):
            # Dict format with destab and stab
            destab = np.array(clifford_tableau.get("destab", []))
            stab = np.array(clifford_tableau.get("stab", []))
            if destab.size == 0 or stab.size == 0:
                return {
                    "status": "error",
                    "message": "clifford_tableau dict must have 'destab' and 'stab' keys",
                }
            # Reconstruct tableau (simplified - may need adjustment for phases)
            tableau = np.vstack([destab, stab])
        else:
            tableau = np.array(clifford_tableau, dtype=np.int32)

        # Validate dimensions
        if tableau.ndim != 2:
            return {
                "status": "error",
                "message": "clifford_tableau must be a 2D matrix",
            }

        # Expected shape is (2N, 2N+1) or (2N+1, 2N) depending on format
        n_qubits = model.num_qubits
        expected_rows = 2 * n_qubits
        expected_cols = 2 * n_qubits + 1

        # Try to create Clifford object
        try:
            clifford = Clifford(tableau)
            if clifford.num_qubits != n_qubits:
                return {
                    "status": "error",
                    "message": f"Clifford has {clifford.num_qubits} qubits but model expects {n_qubits}",
                }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Invalid Clifford tableau: {e}. Expected ({expected_rows}, {expected_cols}) binary matrix.",
            }

        # Validate num_searches
        if num_searches > QISKIT_GYM_MAX_SEARCHES:
            return {
                "status": "error",
                "message": f"num_searches ({num_searches}) exceeds maximum ({QISKIT_GYM_MAX_SEARCHES})",
            }

        # Synthesize circuit
        logger.info(f"Synthesizing Clifford with {num_searches} searches")
        circuit = model.rls_instance.synth(
            clifford,
            num_searches=num_searches,
            deterministic=deterministic,
        )

        # Handle case where synthesis fails to find a solution
        if circuit is None:
            return {
                "status": "error",
                "message": "Synthesis failed to find a solution. Try increasing num_searches or training the model longer.",
            }

        # Get metrics
        metrics = _get_circuit_metrics(circuit)

        return {
            "status": "success",
            "circuit_qpy": _circuit_to_qpy(circuit),
            "num_qubits": n_qubits,
            "metrics": metrics,
            "num_searches": num_searches,
            "note": "Use convert_qpy_to_qasm3 to view circuit in human-readable format",
        }

    except Exception as e:
        logger.error(f"Clifford synthesis failed: {e}")
        return {"status": "error", "message": str(e)}


# ============================================================================
# Random Target Generation (for testing/demos)
# ============================================================================


@with_sync
async def generate_random_permutation(num_qubits: int) -> dict[str, Any]:
    """Generate a random permutation for testing.

    Args:
        num_qubits: Number of qubits

    Returns:
        Dict with random permutation
    """
    permutation = np.random.permutation(num_qubits).tolist()
    return {
        "status": "success",
        "permutation": permutation,
        "num_qubits": num_qubits,
    }


@with_sync
async def generate_random_linear_function(num_qubits: int) -> dict[str, Any]:
    """Generate a random invertible linear function for testing.

    Args:
        num_qubits: Number of qubits

    Returns:
        Dict with random linear function matrix
    """
    # Generate random invertible binary matrix (mod 2)
    while True:
        matrix = np.random.randint(0, 2, size=(num_qubits, num_qubits))
        # Check invertibility (det != 0 mod 2)
        det = int(round(np.linalg.det(matrix))) % 2
        if det != 0:
            break

    return {
        "status": "success",
        "linear_function": matrix.tolist(),
        "shape": [num_qubits, num_qubits],
    }


@with_sync
async def generate_random_clifford(num_qubits: int) -> dict[str, Any]:
    """Generate a random Clifford element for testing.

    Args:
        num_qubits: Number of qubits

    Returns:
        Dict with random Clifford tableau
    """
    try:
        from qiskit.quantum_info import random_clifford

        clifford = random_clifford(num_qubits)
        tableau = clifford.tableau.astype(int).tolist()

        return {
            "status": "success",
            "clifford_tableau": tableau,
            "num_qubits": num_qubits,
        }

    except Exception as e:
        logger.error(f"Failed to generate random Clifford: {e}")
        return {"status": "error", "message": str(e)}
