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
import binascii
import io
import logging
from typing import Any, Literal

from qiskit import QuantumCircuit, qasm2, qpy
from qiskit.qasm3 import dumps as qasm3_dumps
from qiskit.qasm3 import loads as qasm3_loads


logger = logging.getLogger(__name__)

CircuitFormat = Literal["qasm3", "qpy", "auto"]

# QPY magic number (first 6 bytes after base64 decoding)
# QPY files start with "QISKIT" in ASCII
QPY_MAGIC = b"QISKIT"


def detect_circuit_format(circuit_data: str) -> CircuitFormat:
    """Detect whether circuit data is QASM3 or QPY format.

    This function examines the circuit data to determine its format:
    - QASM3/QASM2: Text starting with "OPENQASM" or containing typical QASM markers
    - QPY: Base64-encoded binary data that decodes to QPY magic number

    Args:
        circuit_data: The circuit data string to analyze.

    Returns:
        "qasm3" if the data appears to be QASM (including QASM2),
        "qpy" if the data appears to be base64-encoded QPY.

    Example:
        >>> format_type = detect_circuit_format("OPENQASM 3.0; ...")
        >>> format_type
        'qasm3'
    """
    # Strip whitespace for detection
    stripped = circuit_data.strip()

    # Check for QASM indicators (both QASM2 and QASM3)
    if stripped.upper().startswith("OPENQASM"):
        return "qasm3"

    # Check for other common QASM indicators
    if any(marker in stripped for marker in ["qubit", "qreg", "include"]):
        return "qasm3"

    # Try to decode as base64 and check for QPY magic
    try:
        decoded = base64.b64decode(stripped)
        if decoded.startswith(QPY_MAGIC):
            return "qpy"
    except (ValueError, binascii.Error):
        # Not valid base64, likely QASM - this is expected for QASM input
        logger.debug("Input is not valid base64, treating as QASM")

    # Default to QASM3 if we can't determine
    logger.debug("Could not definitively detect format, defaulting to qasm3")
    return "qasm3"


def load_qasm_circuit(qasm_string: str) -> dict[str, Any]:
    """Load a quantum circuit from a QASM 3.0 or QASM 2.0 string.

    Attempts to parse as QASM3 first, then falls back to QASM2 for
    backwards compatibility.

    Args:
        qasm_string: A valid OpenQASM 3.0 or 2.0 string describing the circuit.

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
    # Try QASM3 first
    try:
        circuit = qasm3_loads(qasm_string)
        return {"status": "success", "circuit": circuit}
    except Exception as qasm3_error:
        logger.debug(f"QASM3 parsing failed: {qasm3_error}, trying QASM2")

    # Fall back to QASM2
    try:
        circuit = qasm2.loads(qasm_string)
        return {"status": "success", "circuit": circuit}
    except Exception as qasm2_error:
        logger.error(f"Both QASM3 and QASM2 parsing failed: {qasm2_error}")
        return {
            "status": "error",
            "message": "QASM string not valid. Cannot be loaded as QuantumCircuit (tried both QASM3 and QASM2).",
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


def load_circuit(circuit_data: str, circuit_format: CircuitFormat = "auto") -> dict[str, Any]:
    """Load a quantum circuit from either QASM3 or QPY format.

    This is a unified interface for loading circuits in different formats.

    Args:
        circuit_data: The circuit data as a string. For QASM3, this is the
            OpenQASM 3.0 text. For QPY, this is a base64-encoded binary string.
        circuit_format: The format of the input data. Options:
            - "auto" (default): Automatically detect format from the data
            - "qasm3": Treat as QASM3 (also accepts QASM2 as fallback)
            - "qpy": Treat as base64-encoded QPY binary

    Returns:
        A dictionary with:
        - status: "success" or "error"
        - circuit: The loaded QuantumCircuit (if successful)
        - message: Error message (if failed)
        - detected_format: The format that was used (when auto-detecting)

    Example:
        >>> # Auto-detect format (recommended)
        >>> result = load_circuit(circuit_string)
        >>> # Explicit QASM3
        >>> result = load_circuit(qasm_string, circuit_format="qasm3")
        >>> # Explicit QPY
        >>> result = load_circuit(qpy_b64_string, circuit_format="qpy")
    """
    # Auto-detect format if not specified
    actual_format = circuit_format
    if circuit_format == "auto":
        actual_format = detect_circuit_format(circuit_data)
        logger.debug(f"Auto-detected circuit format: {actual_format}")

    if actual_format == "qpy":
        result = load_qpy_circuit(circuit_data)
    else:
        result = load_qasm_circuit(circuit_data)

    # Add detected format to result for transparency
    if circuit_format == "auto" and result["status"] == "success":
        result["detected_format"] = actual_format

    return result


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


def qpy_to_qasm3(qpy_b64: str) -> dict[str, Any]:
    """Convert a base64-encoded QPY circuit to human-readable QASM3 format.

    This is a convenience function for viewing QPY circuit output from MCP tools
    in a human-readable format.

    Args:
        qpy_b64: A base64-encoded string containing QPY binary data.

    Returns:
        A dictionary with:
        - status: "success" or "error"
        - qasm3: The QASM 3.0 string representation (if successful)
        - message: Error message (if failed)

    Example:
        >>> from qiskit_mcp_server import qpy_to_qasm3
        >>> # After getting QPY output from transpile_circuit
        >>> result = qpy_to_qasm3(transpiled_qpy)
        >>> if result["status"] == "success":
        ...     print(result["qasm3"])
    """
    load_result = load_qpy_circuit(qpy_b64)
    if load_result["status"] == "error":
        return {
            "status": "error",
            "message": load_result["message"],
        }

    circuit = load_result["circuit"]
    try:
        qasm3_str = dump_qasm_circuit(circuit)
        return {
            "status": "success",
            "qasm3": qasm3_str,
        }
    except Exception as e:
        logger.error(f"Error converting to QASM3: {e}")
        return {
            "status": "error",
            "message": f"Failed to convert circuit to QASM3: {e}",
        }


def qasm3_to_qpy(qasm_string: str) -> dict[str, Any]:
    """Convert a QASM3 (or QASM2) circuit to base64-encoded QPY format.

    This is a convenience function for converting human-readable QASM circuits
    to QPY format for use with MCP tools that accept QPY input.

    Args:
        qasm_string: A valid OpenQASM 3.0 or 2.0 string describing the circuit.

    Returns:
        A dictionary with:
        - status: "success" or "error"
        - circuit_qpy: The base64-encoded QPY string (if successful)
        - message: Error message (if failed)

    Example:
        >>> from qiskit_mcp_server import qasm3_to_qpy
        >>> qasm = 'OPENQASM 3.0; include "stdgates.inc"; qubit[2] q; h q[0];'
        >>> result = qasm3_to_qpy(qasm)
        >>> if result["status"] == "success":
        ...     qpy_str = result["circuit_qpy"]
    """
    load_result = load_qasm_circuit(qasm_string)
    if load_result["status"] == "error":
        return {
            "status": "error",
            "message": load_result["message"],
        }

    circuit = load_result["circuit"]
    try:
        qpy_str = dump_qpy_circuit(circuit)
        return {
            "status": "success",
            "circuit_qpy": qpy_str,
        }
    except Exception as e:
        logger.error(f"Error converting to QPY: {e}")
        return {
            "status": "error",
            "message": f"Failed to convert circuit to QPY: {e}",
        }
