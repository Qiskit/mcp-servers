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
from typing import Any

from qiskit import QuantumCircuit
from qiskit.qasm3 import loads


def validate_synthesis_result(result: dict) -> None:
    """Helper to validate synthesis result structure and values.

    Validates that the result contains:
    - status: "success"
    - circuit_qpy: non-empty base64-encoded QPY string
    - original_circuit: metrics dict with num_qubits, depth, size, two_qubit_gates
    - optimized_circuit: metrics dict with num_qubits, depth, size, two_qubit_gates
    - improvements: dict with depth_reduction and two_qubit_gate_reduction

    Also verifies that improvement calculations are correct.
    """
    assert result["status"] == "success"
    assert isinstance(result["circuit_qpy"], str)
    assert len(result["circuit_qpy"]) > 0  # Non-empty QPY

    # Validate original_circuit metrics
    orig = result["original_circuit"]
    assert isinstance(orig, dict)
    assert "num_qubits" in orig and isinstance(orig["num_qubits"], int)
    assert "depth" in orig and isinstance(orig["depth"], int)
    assert "size" in orig and isinstance(orig["size"], int)
    assert "two_qubit_gates" in orig and isinstance(orig["two_qubit_gates"], int)
    assert orig["num_qubits"] > 0

    # Validate optimized_circuit metrics
    opt = result["optimized_circuit"]
    assert isinstance(opt, dict)
    assert "num_qubits" in opt and isinstance(opt["num_qubits"], int)
    assert "depth" in opt and isinstance(opt["depth"], int)
    assert "size" in opt and isinstance(opt["size"], int)
    assert "two_qubit_gates" in opt and isinstance(opt["two_qubit_gates"], int)
    # Qubit count should be preserved
    assert opt["num_qubits"] == orig["num_qubits"]

    # Validate improvements
    imp = result["improvements"]
    assert isinstance(imp, dict)
    assert "depth_reduction" in imp and isinstance(imp["depth_reduction"], int)
    assert "two_qubit_gate_reduction" in imp and isinstance(imp["two_qubit_gate_reduction"], int)
    # Verify improvement calculation is correct
    assert imp["depth_reduction"] == orig["depth"] - opt["depth"]
    assert imp["two_qubit_gate_reduction"] == orig["two_qubit_gates"] - opt["two_qubit_gates"]


def return_2q_count_and_depth(circuit: QuantumCircuit) -> dict[str, Any]:
    circuit_without_swaps = circuit.decompose("swap")
    return {
        "2q_gates": circuit_without_swaps.num_nonlocal_gates(),
        "2q_depth": circuit_without_swaps.depth(lambda op: len(op.qubits) >= 2),
    }


def calculate_2q_count_and_depth_improvement(
    circuit1_qasm: str, circuit2_qasm: str
) -> dict[str, Any]:
    """Compute 2 qubit gate count and depth improvement"""
    circuit1 = loads(circuit1_qasm)
    circuit2 = loads(circuit2_qasm)
    # Calculate improvement
    circuit1_gates = return_2q_count_and_depth(circuit1).get("2q_gates")
    circuit2_gates = return_2q_count_and_depth(circuit2).get("2q_gates")

    if circuit1_gates == 0:
        improvement_2q_gates = 0.0
    else:
        improvement_2q_gates = ((circuit1_gates - circuit2_gates) / circuit1_gates) * 100

    circuit1_depth = return_2q_count_and_depth(circuit1).get("2q_depth")
    circuit2_depth = return_2q_count_and_depth(circuit2).get("2q_depth")

    if circuit1_depth == 0:
        improvement_2q_depth = 0.0
    else:
        improvement_2q_depth = ((circuit1_depth - circuit2_depth) / circuit1_depth) * 100

    return {
        "improvement_2q_gates": improvement_2q_gates,
        "improvement_2q_depth": improvement_2q_depth,
    }
