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

#!/usr/bin/env python3
"""
Qiskit MCP Server

A Model Context Protocol server that provides Qiskit quantum computing
capabilities, enabling AI assistants to work with quantum circuits,
transpilation, and other Qiskit features.

Dependencies:
- fastmcp
- qiskit
- python-dotenv
"""

import logging
from typing import Any

from fastmcp import FastMCP

from qiskit_mcp_server.circuit_serialization import CircuitFormat, qasm3_to_qpy, qpy_to_qasm3
from qiskit_mcp_server.transpiler import (
    analyze_circuit,
    compare_optimization_levels,
    get_available_basis_gates,
    get_available_topologies,
    get_transpiler_info,
    transpile_circuit,
)


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize MCP server
mcp = FastMCP("Qiskit")


# Tools - Only action-oriented tools, metadata is via resources
@mcp.tool()
async def transpile_circuit_tool(
    circuit: str,
    optimization_level: int = 2,
    basis_gates: list[str] | str | None = None,
    coupling_map: list[list[int]] | str | None = None,
    initial_layout: list[int] | None = None,
    seed_transpiler: int | None = None,
    circuit_format: CircuitFormat = "qasm3",
) -> dict[str, Any]:
    """Transpile a quantum circuit using Qiskit's preset pass managers.

    Takes a quantum circuit and transpiles it to match target hardware
    constraints while optimizing for depth and gate count.

    IMPORTANT: Optimization level 3 can be very slow for large circuits (100+ qubits
    or 1000+ gates). Consider using level 2 for faster results with good quality.

    Args:
        circuit: Quantum circuit as QASM3 string, base64-encoded QPY, or QASM2 string.
            Maximum supported: 100 qubits, 10000 gates.
            For QASM2, set circuit_format="qasm3" (it will auto-detect and parse QASM2).
        optimization_level: Optimization level (0-3):
            - 0: No optimization, just maps to basis gates (fastest)
            - 1: Light optimization (default mapping, simple optimizations)
            - 2: Medium optimization (noise-adaptive layout) [default, recommended]
            - 3: Heavy optimization (best results, can be very slow for large circuits)
        basis_gates: Target basis gates. Can be:
            - A list of gate names (e.g., ["cx", "id", "rz", "sx", "x"])
            - A preset name: "ibm_default", "ibm_eagle", "ibm_heron",
              "generic_clifford_t", "ion_trap", "superconducting"
            - None for no basis gate restriction
        coupling_map: Qubit connectivity. Can be:
            - A list of [control, target] pairs (e.g., [[0, 1], [1, 2]])
            - A topology name: "linear", "ring", "grid", "full"
            - None for all-to-all connectivity
        initial_layout: Optional initial qubit layout as list of physical qubit indices.
            Length must match the number of qubits in the circuit.
        seed_transpiler: Random seed for reproducibility
        circuit_format: Format of the input circuit ("qasm3" or "qpy"). Defaults to "qasm3".
            When "qasm3" is specified, QASM2 is also accepted as a fallback.

    Returns:
        Dictionary with original and transpiled circuit info, and optimization metrics
    """
    return await transpile_circuit(
        circuit=circuit,
        optimization_level=optimization_level,
        basis_gates=basis_gates,
        coupling_map=coupling_map,
        initial_layout=initial_layout,
        seed_transpiler=seed_transpiler,
        circuit_format=circuit_format,
    )


@mcp.tool()
async def analyze_circuit_tool(
    circuit: str,
    circuit_format: CircuitFormat = "qasm3",
) -> dict[str, Any]:
    """Analyze a quantum circuit without transpiling it.

    Provides detailed information about circuit structure, gate counts,
    and metrics useful for understanding circuit complexity.

    Args:
        circuit: Quantum circuit as QASM3 string, base64-encoded QPY, or QASM2 string.
        circuit_format: Format of the input circuit ("qasm3" or "qpy"). Defaults to "qasm3".
            When "qasm3" is specified, QASM2 is also accepted as a fallback.

    Returns:
        Dictionary with circuit analysis including gate counts, depth, and categorization
    """
    return await analyze_circuit(circuit, circuit_format=circuit_format)


@mcp.tool()
async def compare_optimization_levels_tool(
    circuit: str,
    circuit_format: CircuitFormat = "qasm3",
) -> dict[str, Any]:
    """Compare transpilation results across all optimization levels (0-3).

    Useful for understanding the trade-off between compilation time
    and circuit quality for a specific circuit.

    WARNING: This runs transpilation 4 times. For large circuits, this can be slow.

    Args:
        circuit: Quantum circuit as QASM3 string, base64-encoded QPY, or QASM2 string.
        circuit_format: Format of the input circuit ("qasm3" or "qpy"). Defaults to "qasm3".
            When "qasm3" is specified, QASM2 is also accepted as a fallback.

    Returns:
        Dictionary comparing depth, size, and gate counts across all levels
    """
    return await compare_optimization_levels(circuit, circuit_format=circuit_format)


@mcp.tool()
async def convert_qpy_to_qasm3_tool(
    circuit_qpy: str,
) -> dict[str, Any]:
    """Convert a QPY circuit to human-readable QASM3 format.

    Use this tool to view the contents of a QPY circuit output from other tools
    (like transpile_circuit) in a human-readable OpenQASM 3.0 format.

    Args:
        circuit_qpy: Base64-encoded QPY circuit string (from transpile_circuit output)

    Returns:
        Dict with 'status' and 'qasm3' (the human-readable circuit string).
    """
    return qpy_to_qasm3(circuit_qpy)


@mcp.tool()
async def convert_qasm3_to_qpy_tool(
    circuit_qasm: str,
) -> dict[str, Any]:
    """Convert a QASM3 (or QASM2) circuit to base64-encoded QPY format.

    Use this tool to convert human-readable QASM circuits to QPY format,
    which preserves full circuit fidelity (exact parameters, metadata, custom gates).
    The QPY output can then be used with other tools that accept QPY input.

    Args:
        circuit_qasm: OpenQASM 3.0 or 2.0 circuit string

    Returns:
        Dict with 'status' and 'circuit_qpy' (base64-encoded QPY string).
    """
    return qasm3_to_qpy(circuit_qasm)


# Resources - Static metadata accessible without tool calls
@mcp.resource("qiskit://transpiler/info", mime_type="application/json")
async def transpiler_info_resource() -> dict[str, Any]:
    """Get Qiskit transpiler information and capabilities.

    Returns comprehensive documentation about how transpilation works,
    the six transpiler stages, optimization levels, and usage recommendations.
    """
    return await get_transpiler_info()


@mcp.resource("qiskit://transpiler/basis-gates", mime_type="application/json")
async def basis_gates_resource() -> dict[str, Any]:
    """Get available preset basis gate sets.

    Returns information about predefined basis gate sets that can be
    used with the transpile_circuit tool, including IBM Eagle, Heron,
    ion trap, and other common gate sets.
    """
    return await get_available_basis_gates()


@mcp.resource("qiskit://transpiler/topologies", mime_type="application/json")
async def topologies_resource() -> dict[str, Any]:
    """Get available coupling map topologies.

    Returns information about predefined qubit connectivity topologies
    (linear, ring, grid, full) that can be used with the transpile_circuit tool.
    """
    return await get_available_topologies()


def main() -> None:
    """Run the server."""
    mcp.run()


if __name__ == "__main__":
    main()
