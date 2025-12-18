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
"""Tests for circuit serialization utilities."""

import pytest
from qiskit import QuantumCircuit
from qiskit_mcp_server.circuit_serialization import (
    detect_circuit_format,
    dump_circuit,
    dump_qasm_circuit,
    dump_qpy_circuit,
    load_circuit,
    load_qasm_circuit,
    load_qpy_circuit,
    qpy_to_qasm3,
)


@pytest.fixture
def simple_circuit():
    """Create a simple quantum circuit for testing."""
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    return qc


@pytest.fixture
def valid_qasm3():
    """Valid QASM 3.0 string."""
    return """OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;
h q[0];
cx q[0], q[1];
"""


@pytest.fixture
def invalid_qasm3():
    """Invalid QASM 3.0 string."""
    return "INVALID QASM STRING { this is not valid }"


class TestLoadQasmCircuit:
    """Tests for load_qasm_circuit function."""

    def test_load_valid_qasm(self, valid_qasm3):
        """Test loading a valid QASM 3.0 string."""
        result = load_qasm_circuit(valid_qasm3)

        assert result["status"] == "success"
        assert isinstance(result["circuit"], QuantumCircuit)
        assert result["circuit"].num_qubits == 2

    def test_load_invalid_qasm(self, invalid_qasm3):
        """Test loading an invalid QASM string."""
        result = load_qasm_circuit(invalid_qasm3)

        assert result["status"] == "error"
        assert "message" in result
        assert "QASM string not valid" in result["message"]

    def test_load_qasm2_fallback(self):
        """Test that QASM2 strings are loaded via fallback."""
        qasm2_string = """OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
h q[0];
cx q[0],q[1];
"""
        result = load_qasm_circuit(qasm2_string)

        assert result["status"] == "success"
        assert isinstance(result["circuit"], QuantumCircuit)
        assert result["circuit"].num_qubits == 2


class TestLoadQpyCircuit:
    """Tests for load_qpy_circuit function."""

    def test_load_valid_qpy(self, simple_circuit):
        """Test loading a valid QPY string."""
        # First dump the circuit to QPY
        qpy_str = dump_qpy_circuit(simple_circuit)

        # Then load it back
        result = load_qpy_circuit(qpy_str)

        assert result["status"] == "success"
        assert isinstance(result["circuit"], QuantumCircuit)
        assert result["circuit"].num_qubits == simple_circuit.num_qubits

    def test_load_invalid_qpy(self):
        """Test loading an invalid QPY string."""
        result = load_qpy_circuit("not-valid-base64!")

        assert result["status"] == "error"
        assert "message" in result
        assert "Invalid QPY data" in result["message"]

    def test_load_invalid_base64_qpy(self):
        """Test loading valid base64 but invalid QPY data."""
        import base64

        invalid_data = base64.b64encode(b"not qpy data").decode("utf-8")
        result = load_qpy_circuit(invalid_data)

        assert result["status"] == "error"
        assert "Invalid QPY data" in result["message"]


class TestDumpQasmCircuit:
    """Tests for dump_qasm_circuit function."""

    def test_dump_circuit_to_qasm(self, simple_circuit):
        """Test dumping a circuit to QASM 3.0."""
        qasm_str = dump_qasm_circuit(simple_circuit)

        assert isinstance(qasm_str, str)
        assert "OPENQASM" in qasm_str
        # Should contain gate operations
        assert "h " in qasm_str.lower() or "h(" in qasm_str.lower()


class TestDumpQpyCircuit:
    """Tests for dump_qpy_circuit function."""

    def test_dump_circuit_to_qpy(self, simple_circuit):
        """Test dumping a circuit to QPY."""
        qpy_str = dump_qpy_circuit(simple_circuit)

        assert isinstance(qpy_str, str)
        # QPY base64 strings are typically alphanumeric with + / =
        import base64

        # Should be valid base64
        decoded = base64.b64decode(qpy_str)
        assert len(decoded) > 0


class TestLoadCircuit:
    """Tests for unified load_circuit function."""

    def test_load_circuit_qasm3_default(self, valid_qasm3):
        """Test load_circuit defaults to QASM3 format."""
        result = load_circuit(valid_qasm3)

        assert result["status"] == "success"
        assert isinstance(result["circuit"], QuantumCircuit)

    def test_load_circuit_qasm3_explicit(self, valid_qasm3):
        """Test load_circuit with explicit QASM3 format."""
        result = load_circuit(valid_qasm3, circuit_format="qasm3")

        assert result["status"] == "success"
        assert isinstance(result["circuit"], QuantumCircuit)

    def test_load_circuit_qpy(self, simple_circuit):
        """Test load_circuit with QPY format."""
        qpy_str = dump_qpy_circuit(simple_circuit)
        result = load_circuit(qpy_str, circuit_format="qpy")

        assert result["status"] == "success"
        assert isinstance(result["circuit"], QuantumCircuit)


class TestDumpCircuit:
    """Tests for unified dump_circuit function."""

    def test_dump_circuit_qasm3_default(self, simple_circuit):
        """Test dump_circuit defaults to QASM3 format."""
        result = dump_circuit(simple_circuit)

        assert isinstance(result, str)
        assert "OPENQASM" in result

    def test_dump_circuit_qasm3_explicit(self, simple_circuit):
        """Test dump_circuit with explicit QASM3 format."""
        result = dump_circuit(simple_circuit, circuit_format="qasm3")

        assert isinstance(result, str)
        assert "OPENQASM" in result

    def test_dump_circuit_qpy(self, simple_circuit):
        """Test dump_circuit with QPY format."""
        result = dump_circuit(simple_circuit, circuit_format="qpy")

        assert isinstance(result, str)
        # QPY is base64, won't contain OPENQASM
        assert "OPENQASM" not in result


class TestRoundTrip:
    """Tests for round-trip serialization/deserialization."""

    def test_qasm3_round_trip(self, simple_circuit):
        """Test QASM3 round-trip preserves circuit structure."""
        # Dump to QASM3
        qasm_str = dump_circuit(simple_circuit, circuit_format="qasm3")

        # Load back
        result = load_circuit(qasm_str, circuit_format="qasm3")

        assert result["status"] == "success"
        loaded_circuit = result["circuit"]

        # Check structure is preserved
        assert loaded_circuit.num_qubits == simple_circuit.num_qubits
        assert loaded_circuit.depth() == simple_circuit.depth()

    def test_qpy_round_trip(self, simple_circuit):
        """Test QPY round-trip preserves circuit exactly."""
        # Dump to QPY
        qpy_str = dump_circuit(simple_circuit, circuit_format="qpy")

        # Load back
        result = load_circuit(qpy_str, circuit_format="qpy")

        assert result["status"] == "success"
        loaded_circuit = result["circuit"]

        # Check structure is preserved
        assert loaded_circuit.num_qubits == simple_circuit.num_qubits
        assert loaded_circuit.depth() == simple_circuit.depth()
        # QPY should preserve gate count exactly
        assert len(loaded_circuit.data) == len(simple_circuit.data)

    def test_qpy_preserves_parameters(self):
        """Test QPY preserves parameterized gates with exact values."""
        import math

        qc = QuantumCircuit(1)
        qc.rx(math.pi / 7, 0)  # Use an angle that might lose precision in QASM3

        # Round-trip through QPY
        qpy_str = dump_circuit(qc, circuit_format="qpy")
        result = load_circuit(qpy_str, circuit_format="qpy")

        loaded_circuit = result["circuit"]
        original_angle = qc.data[0].operation.params[0]
        loaded_angle = loaded_circuit.data[0].operation.params[0]

        # QPY should preserve exact floating point value
        assert original_angle == loaded_angle

    def test_qpy_preserves_circuit_name(self):
        """Test QPY preserves circuit metadata like name."""
        qc = QuantumCircuit(2, name="my_test_circuit")
        qc.h(0)

        # Round-trip through QPY
        qpy_str = dump_circuit(qc, circuit_format="qpy")
        result = load_circuit(qpy_str, circuit_format="qpy")

        loaded_circuit = result["circuit"]
        assert loaded_circuit.name == "my_test_circuit"


class TestQpyToQasm3:
    """Tests for qpy_to_qasm3 conversion utility."""

    def test_qpy_to_qasm3_success(self, simple_circuit):
        """Test successful QPY to QASM3 conversion."""
        # First create QPY string
        qpy_str = dump_qpy_circuit(simple_circuit)

        # Convert to QASM3
        result = qpy_to_qasm3(qpy_str)

        assert result["status"] == "success"
        assert "qasm3" in result
        assert isinstance(result["qasm3"], str)
        assert "OPENQASM" in result["qasm3"]

    def test_qpy_to_qasm3_invalid_qpy(self):
        """Test qpy_to_qasm3 with invalid QPY data."""
        result = qpy_to_qasm3("not-valid-base64!")

        assert result["status"] == "error"
        assert "message" in result
        assert "Invalid QPY data" in result["message"]

    def test_qpy_to_qasm3_invalid_base64(self):
        """Test qpy_to_qasm3 with valid base64 but invalid QPY."""
        import base64

        invalid_data = base64.b64encode(b"not qpy data").decode("utf-8")
        result = qpy_to_qasm3(invalid_data)

        assert result["status"] == "error"
        assert "message" in result

    def test_qpy_to_qasm3_preserves_structure(self, simple_circuit):
        """Test that qpy_to_qasm3 produces valid QASM3 that can be re-loaded."""
        # Create QPY
        qpy_str = dump_qpy_circuit(simple_circuit)

        # Convert to QASM3
        result = qpy_to_qasm3(qpy_str)
        assert result["status"] == "success"

        # Load the QASM3 back
        load_result = load_qasm_circuit(result["qasm3"])
        assert load_result["status"] == "success"

        # Verify structure is preserved
        loaded_circuit = load_result["circuit"]
        assert loaded_circuit.num_qubits == simple_circuit.num_qubits
        assert loaded_circuit.depth() == simple_circuit.depth()


class TestQasm3ToQpy:
    """Tests for qasm3_to_qpy conversion utility."""

    def test_qasm3_to_qpy_success(self, valid_qasm3):
        """Test successful QASM3 to QPY conversion."""
        from qiskit_mcp_server.circuit_serialization import qasm3_to_qpy

        result = qasm3_to_qpy(valid_qasm3)

        assert result["status"] == "success"
        assert "circuit_qpy" in result
        assert isinstance(result["circuit_qpy"], str)
        # Verify it's valid base64 that can be loaded back
        load_result = load_qpy_circuit(result["circuit_qpy"])
        assert load_result["status"] == "success"

    def test_qasm3_to_qpy_invalid_qasm(self, invalid_qasm3):
        """Test qasm3_to_qpy with invalid QASM data."""
        from qiskit_mcp_server.circuit_serialization import qasm3_to_qpy

        result = qasm3_to_qpy(invalid_qasm3)

        assert result["status"] == "error"
        assert "message" in result

    def test_qasm3_to_qpy_preserves_structure(self, valid_qasm3):
        """Test that qasm3_to_qpy produces QPY that preserves circuit structure."""
        from qiskit_mcp_server.circuit_serialization import qasm3_to_qpy

        # Convert QASM3 to QPY
        result = qasm3_to_qpy(valid_qasm3)
        assert result["status"] == "success"

        # Load the QPY back
        load_result = load_qpy_circuit(result["circuit_qpy"])
        assert load_result["status"] == "success"

        # Original circuit from QASM3
        original_result = load_qasm_circuit(valid_qasm3)
        original_circuit = original_result["circuit"]

        # Verify structure is preserved
        loaded_circuit = load_result["circuit"]
        assert loaded_circuit.num_qubits == original_circuit.num_qubits
        assert loaded_circuit.depth() == original_circuit.depth()

    def test_qasm2_to_qpy_fallback(self):
        """Test that QASM2 input works via fallback."""
        from qiskit_mcp_server.circuit_serialization import qasm3_to_qpy

        qasm2_string = """OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
h q[0];
cx q[0],q[1];
"""
        result = qasm3_to_qpy(qasm2_string)

        assert result["status"] == "success"
        assert "circuit_qpy" in result

        # Verify the loaded circuit
        load_result = load_qpy_circuit(result["circuit_qpy"])
        assert load_result["status"] == "success"
        assert load_result["circuit"].num_qubits == 2


class TestDetectCircuitFormat:
    """Tests for auto-detection of circuit format."""

    def test_detect_qasm3_with_openqasm_header(self, valid_qasm3):
        """Test detection of QASM3 format from OPENQASM header."""
        detected = detect_circuit_format(valid_qasm3)
        assert detected == "qasm3"

    def test_detect_qasm3_uppercase(self):
        """Test detection works with uppercase OPENQASM."""
        qasm = "OPENQASM 3.0; qubit q;"
        detected = detect_circuit_format(qasm)
        assert detected == "qasm3"

    def test_detect_qasm3_lowercase(self):
        """Test detection works with lowercase openqasm."""
        qasm = "openqasm 3.0; qubit q;"
        detected = detect_circuit_format(qasm)
        assert detected == "qasm3"

    def test_detect_qasm_with_qubit_keyword(self):
        """Test detection of QASM from qubit keyword."""
        qasm = "qubit[2] q; h q[0];"
        detected = detect_circuit_format(qasm)
        assert detected == "qasm3"

    def test_detect_qasm_with_qreg_keyword(self):
        """Test detection of QASM from qreg keyword (QASM2 style)."""
        qasm = "qreg q[2]; creg c[2];"
        detected = detect_circuit_format(qasm)
        assert detected == "qasm3"

    def test_detect_qpy_format(self, simple_circuit):
        """Test detection of QPY format from base64-encoded data."""
        qpy_str = dump_qpy_circuit(simple_circuit)
        detected = detect_circuit_format(qpy_str)
        assert detected == "qpy"

    def test_detect_with_whitespace(self, valid_qasm3):
        """Test detection handles leading/trailing whitespace."""
        qasm_with_whitespace = f"  \n  {valid_qasm3}  \n  "
        detected = detect_circuit_format(qasm_with_whitespace)
        assert detected == "qasm3"


class TestAutoDetectionLoadCircuit:
    """Tests for load_circuit with auto-detection."""

    def test_load_circuit_auto_detect_qasm3(self, valid_qasm3):
        """Test load_circuit auto-detects QASM3 format."""
        result = load_circuit(valid_qasm3)  # No circuit_format specified

        assert result["status"] == "success"
        assert result.get("detected_format") == "qasm3"
        assert isinstance(result["circuit"], QuantumCircuit)

    def test_load_circuit_auto_detect_qpy(self, simple_circuit):
        """Test load_circuit auto-detects QPY format."""
        qpy_str = dump_qpy_circuit(simple_circuit)
        result = load_circuit(qpy_str)  # No circuit_format specified

        assert result["status"] == "success"
        assert result.get("detected_format") == "qpy"
        assert isinstance(result["circuit"], QuantumCircuit)

    def test_load_circuit_explicit_auto(self, valid_qasm3):
        """Test load_circuit with explicit 'auto' format."""
        result = load_circuit(valid_qasm3, circuit_format="auto")

        assert result["status"] == "success"
        assert result.get("detected_format") == "qasm3"

    def test_load_circuit_explicit_format_no_detected_field(self, valid_qasm3):
        """Test that explicit format doesn't add detected_format field."""
        result = load_circuit(valid_qasm3, circuit_format="qasm3")

        assert result["status"] == "success"
        # detected_format should not be present when format is explicit
        assert "detected_format" not in result

    def test_auto_detect_preserves_circuit_data(self, simple_circuit):
        """Test that auto-detection correctly loads circuit data."""
        # Test with QPY
        qpy_str = dump_qpy_circuit(simple_circuit)
        result = load_circuit(qpy_str)

        assert result["status"] == "success"
        loaded = result["circuit"]
        assert loaded.num_qubits == simple_circuit.num_qubits
        assert loaded.depth() == simple_circuit.depth()
