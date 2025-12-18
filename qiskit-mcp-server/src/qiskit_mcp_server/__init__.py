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
"""Qiskit MCP Server package.

This package provides:
- MCP server for quantum circuit transpilation
- Shared utilities for circuit serialization (QPY/QASM3)
- Async/sync helpers via the `with_sync` decorator
"""

from qiskit_mcp_server import server
from qiskit_mcp_server.circuit_serialization import (
    CircuitFormat,
    detect_circuit_format,
    dump_circuit,
    dump_qasm_circuit,
    dump_qpy_circuit,
    load_circuit,
    load_qasm_circuit,
    load_qpy_circuit,
    qasm3_to_qpy,
    qpy_to_qasm3,
)
from qiskit_mcp_server.utils import with_sync


def main() -> None:
    """Main entry point for the package."""
    server.mcp.run(transport="stdio")


__all__ = [
    "CircuitFormat",
    "detect_circuit_format",
    "dump_circuit",
    "dump_qasm_circuit",
    "dump_qpy_circuit",
    "load_circuit",
    "load_qasm_circuit",
    "load_qpy_circuit",
    "main",
    "qasm3_to_qpy",
    "qpy_to_qasm3",
    "server",
    "with_sync",
]
