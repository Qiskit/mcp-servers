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

"""Test configuration and fixtures for Qiskit MCP Server tests."""

import pytest
from qiskit import QuantumCircuit
from qiskit_mcp_server.circuit_serialization import dump_qpy_circuit


@pytest.fixture
def simple_circuit_qasm() -> str:
    """A simple 2-qubit circuit in OpenQASM 2.0 format."""
    return """OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
h q[0];
cx q[0], q[1];
measure q -> c;
"""


@pytest.fixture
def bell_state_qasm() -> str:
    """Bell state preparation circuit."""
    return """OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
h q[0];
cx q[0], q[1];
measure q[0] -> c[0];
measure q[1] -> c[1];
"""


@pytest.fixture
def ghz_state_qasm() -> str:
    """GHZ state preparation circuit (3 qubits)."""
    return """OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
creg c[3];
h q[0];
cx q[0], q[1];
cx q[1], q[2];
measure q -> c;
"""


@pytest.fixture
def complex_circuit_qasm() -> str:
    """A more complex circuit with various gates."""
    return """OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
creg c[4];
h q[0];
h q[1];
h q[2];
h q[3];
cx q[0], q[1];
cx q[1], q[2];
cx q[2], q[3];
rz(0.5) q[0];
rz(0.5) q[1];
rz(0.5) q[2];
rz(0.5) q[3];
cx q[3], q[2];
cx q[2], q[1];
cx q[1], q[0];
h q[0];
h q[1];
h q[2];
h q[3];
measure q -> c;
"""


@pytest.fixture
def parameterized_circuit_qasm() -> str:
    """Circuit with rotation gates."""
    return """OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
rx(1.5707963267948966) q[0];
ry(0.7853981633974483) q[1];
cx q[0], q[1];
rz(3.141592653589793) q[1];
measure q -> c;
"""


@pytest.fixture
def invalid_qasm() -> str:
    """Invalid QASM string for error testing."""
    return "this is not valid qasm"


@pytest.fixture
def sample_coupling_map() -> list[list[int]]:
    """Sample coupling map for a linear 5-qubit device."""
    return [
        [0, 1],
        [1, 0],
        [1, 2],
        [2, 1],
        [2, 3],
        [3, 2],
        [3, 4],
        [4, 3],
    ]


@pytest.fixture
def sample_basis_gates() -> list[str]:
    """Sample basis gates for IBM backends."""
    return ["id", "rz", "sx", "x", "cx"]


@pytest.fixture
def simple_circuit_qpy() -> str:
    """A simple 2-qubit circuit in base64-encoded QPY format."""
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])
    return dump_qpy_circuit(qc)
