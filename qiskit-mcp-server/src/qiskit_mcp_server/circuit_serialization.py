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
"""Circuit serialization utilities for QPY and QASM3 formats.

This module provides functions to load and dump quantum circuits in both
QASM 3.0 (text) and QPY (binary, base64-encoded) formats.

QPY Format:
    QPY is Qiskit's native binary serialization format that preserves the full
    circuit object including metadata, custom gates, exact numerical parameters,
    and layout information. QPY data is base64-encoded for JSON transport.

QASM3 Format:
    OpenQASM 3.0 is a text-based standard format for quantum circuits. It is
    human-readable and interoperable with other quantum computing frameworks.

Example:
    >>> from qiskit import QuantumCircuit
    >>> from qiskit_mcp_server import load_circuit, dump_circuit
    >>>
    >>> # Load a QASM3 circuit
    >>> qasm = '''OPENQASM 3.0;
    ... include "stdgates.inc";
    ... qubit[2] q;
    ... h q[0];
    ... cx q[0], q[1];
    ... '''
    >>> result = load_circuit(qasm, circuit_format="qasm3")
    >>> circuit = result["circuit"]
    >>>
    >>> # Dump as QPY (base64-encoded)
    >>> qpy_str = dump_circuit(circuit, circuit_format="qpy")
"""

import base64
import io
import logging
from typing import Any, Literal

from qiskit import QuantumCircuit, qpy
from qiskit.qasm3 import dumps as qasm3_dumps
from qiskit.qasm3 import loads as qasm3_loads


logger = logging.getLogger(__name__)

CircuitFormat = Literal["qasm3", "qpy"]


def load_qasm_circuit(qasm_string: str) -> dict[str, Any]:
    """Load a quantum circuit from a QASM 3.0 string.

    Args:
        qasm_string: A valid OpenQASM 3.0 string describing the circuit.

    Returns:
        A dictionary with:
        - status: "success" or "error"
        - circuit: The loaded QuantumCircuit (if successful)
        - message: Error message (if failed)

    Example:
        >>> qasm = 'OPENQASM 3.0; include "stdgates.inc"; qubit[1] q; h q[0];'
        >>> result = load_qasm_circuit(qasm)
        >>> result["status"]
        'success'
    """
    try:
        circuit = qasm3_loads(qasm_string)
        return {"status": "success", "circuit": circuit}
    except Exception as e:
        logger.error(f"Error loading QASM 3.0: {e}")
        return {
            "status": "error",
            "message": "QASM 3.0 string not valid. Cannot be loaded as QuantumCircuit.",
        }


def load_qpy_circuit(qpy_b64: str) -> dict[str, Any]:
    """Load a quantum circuit from a base64-encoded QPY string.

    Args:
        qpy_b64: A base64-encoded string containing QPY binary data.

    Returns:
        A dictionary with:
        - status: "success" or "error"
        - circuit: The loaded QuantumCircuit (if successful)
        - message: Error message (if failed)

    Example:
        >>> # qpy_str obtained from dump_qpy_circuit()
        >>> result = load_qpy_circuit(qpy_str)
        >>> result["status"]
        'success'
    """
    try:
        buffer = io.BytesIO(base64.b64decode(qpy_b64))
        circuits = qpy.load(buffer)
        return {"status": "success", "circuit": circuits[0]}
    except Exception as e:
        logger.error(f"Error loading QPY: {e}")
        return {
            "status": "error",
            "message": f"Invalid QPY data: {e}",
        }


def dump_qasm_circuit(circuit: QuantumCircuit) -> str:
    """Serialize a quantum circuit to a QASM 3.0 string.

    Args:
        circuit: The QuantumCircuit to serialize.

    Returns:
        An OpenQASM 3.0 string representation of the circuit.

    Example:
        >>> from qiskit import QuantumCircuit
        >>> qc = QuantumCircuit(2)
        >>> qc.h(0)
        >>> qc.cx(0, 1)
        >>> qasm_str = dump_qasm_circuit(qc)
    """
    return str(qasm3_dumps(circuit))


def dump_qpy_circuit(circuit: QuantumCircuit) -> str:
    """Serialize a quantum circuit to a base64-encoded QPY string.

    QPY format preserves all circuit metadata, custom gates, exact numerical
    parameters, and layout information that may be lost in QASM3 conversion.

    Args:
        circuit: The QuantumCircuit to serialize.

    Returns:
        A base64-encoded string containing the QPY binary data.

    Example:
        >>> from qiskit import QuantumCircuit
        >>> qc = QuantumCircuit(2)
        >>> qc.h(0)
        >>> qc.cx(0, 1)
        >>> qpy_str = dump_qpy_circuit(qc)
    """
    buffer = io.BytesIO()
    qpy.dump(circuit, buffer)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def load_circuit(
    circuit_data: str, circuit_format: CircuitFormat = "qasm3"
) -> dict[str, Any]:
    """Load a quantum circuit from either QASM3 or QPY format.

    This is a unified interface for loading circuits in different formats.

    Args:
        circuit_data: The circuit data as a string. For QASM3, this is the
            OpenQASM 3.0 text. For QPY, this is a base64-encoded binary string.
        circuit_format: The format of the input data. Either "qasm3" (default)
            or "qpy".

    Returns:
        A dictionary with:
        - status: "success" or "error"
        - circuit: The loaded QuantumCircuit (if successful)
        - message: Error message (if failed)

    Example:
        >>> # Load QASM3
        >>> result = load_circuit(qasm_string, circuit_format="qasm3")
        >>> # Load QPY
        >>> result = load_circuit(qpy_b64_string, circuit_format="qpy")
    """
    if circuit_format == "qpy":
        return load_qpy_circuit(circuit_data)
    return load_qasm_circuit(circuit_data)


def dump_circuit(circuit: QuantumCircuit, circuit_format: CircuitFormat = "qasm3") -> str:
    """Serialize a quantum circuit to either QASM3 or QPY format.

    This is a unified interface for serializing circuits to different formats.

    Args:
        circuit: The QuantumCircuit to serialize.
        circuit_format: The target format. Either "qasm3" (default) or "qpy".

    Returns:
        The serialized circuit as a string. For QASM3, this is OpenQASM 3.0 text.
        For QPY, this is a base64-encoded binary string.

    Example:
        >>> # Dump as QASM3
        >>> qasm_str = dump_circuit(circuit, circuit_format="qasm3")
        >>> # Dump as QPY
        >>> qpy_str = dump_circuit(circuit, circuit_format="qpy")
    """
    if circuit_format == "qpy":
        return dump_qpy_circuit(circuit)
    return dump_qasm_circuit(circuit)
