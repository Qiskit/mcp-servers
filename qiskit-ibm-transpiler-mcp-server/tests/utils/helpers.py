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
