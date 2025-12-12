from qiskit import QuantumCircuit
from typing import Any
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
    circuit2_gates = return_2q_count_and_depth(circuit2).get("2q_depth")
    improvement_2q_gates = ((circuit1_gates - circuit2_gates) / circuit1_gates) * 100

    circuit1_depth = return_2q_count_and_depth(circuit1).get("2q_depth")
    circuit2_depth = return_2q_count_and_depth(circuit2).get("2q_depth")
    improvement_sq_depth = ((circuit1_depth - circuit2_depth) / circuit1_depth) * 100

    return {
        "improvement_2q_gates": improvement_2q_gates,
        "improvement_2q_depth": improvement_sq_depth,
    }
