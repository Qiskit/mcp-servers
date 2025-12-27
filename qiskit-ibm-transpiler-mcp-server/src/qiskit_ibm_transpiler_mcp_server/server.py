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
import logging
from typing import Any, Literal

from fastmcp import FastMCP

from qiskit_ibm_transpiler_mcp_server.qta import (
    ai_clifford_synthesis,
    ai_linear_function_synthesis,
    ai_pauli_network_synthesis,
    ai_permutation_synthesis,
    ai_routing,
    hybrid_ai_transpile,
)
from qiskit_ibm_transpiler_mcp_server.utils import CircuitFormat, setup_ibm_quantum_account


logger = logging.getLogger(__name__)

# Initialize FastMCP server
mcp = FastMCP("Qiskit IBM Transpiler")

logger.info("Qiskit IBM Transpiler MCP Server initialized")

##################################################
## MCP Tools
## - https://modelcontextprotocol.io/docs/concepts/tools
##################################################


# Tools
@mcp.tool()
async def setup_ibm_quantum_account_tool(
    token: str = "", channel: str = "ibm_quantum_platform"
) -> dict[str, Any]:
    """Set up IBM Quantum account with credentials. Call this before using other tools.

    Args:
        token: IBM Quantum API token. If empty, uses QISKIT_IBM_TOKEN env var or saved credentials from ~/.qiskit/qiskit-ibm.json
        channel: Service channel, must be 'ibm_quantum_platform'

    Returns:
        Dict with 'status' ('success' or 'error'), 'message', and 'available_backends' count on success.
    """
    return await setup_ibm_quantum_account(token if token else None, channel)


@mcp.tool()
async def ai_routing_tool(
    circuit: str,
    backend_name: str,
    optimization_level: Literal[1, 2, 3] = 1,
    layout_mode: Literal["keep", "improve", "optimize"] = "optimize",
    optimization_preferences: Literal["n_cnots", "n_gates", "cnot_layers", "layers", "noise"]
    | list[Literal["n_cnots", "n_gates", "cnot_layers", "layers", "noise"]]
    | None = None,
    local_mode: bool = True,
    circuit_format: CircuitFormat = "qasm3",
) -> dict[str, Any]:
    """Route a quantum circuit by inserting SWAP operations for backend compatibility. Use this FIRST before other synthesis tools.

    Args:
        circuit: Input quantum circuit as QASM 3.0 string or base64-encoded QPY
        backend_name: Target IBM Quantum backend (e.g., 'ibm_torino', 'ibm_fez')
        optimization_level: 1 (fastest, least optimization) to 3 (slowest, most optimization)
        layout_mode: 'keep' (respect existing layout), 'improve' (refine initial guess), 'optimize' (best for general circuits)
        optimization_preferences: What to minimize - 'n_cnots', 'n_gates', 'cnot_layers', 'layers', or 'noise'. Can be a list.
        local_mode: True runs locally (recommended), False uses remote Qiskit Transpiler Service
        circuit_format: Format of the input circuit - 'qasm3' (default) or 'qpy' (base64-encoded QPY for full circuit fidelity)

    Returns:
        Dict with:
        - status: 'success' or 'error'
        - circuit_qpy: Base64-encoded QPY format (for chaining with other tools)
        - original_circuit: Metrics dict (num_qubits, depth, size, two_qubit_gates)
        - optimized_circuit: Metrics dict for the optimized circuit
        - improvements: Dict with depth_reduction and two_qubit_gate_reduction
    """
    return await ai_routing(
        circuit=circuit,
        backend_name=backend_name,
        optimization_level=optimization_level,
        layout_mode=layout_mode,
        optimization_preferences=optimization_preferences,
        local_mode=local_mode,
        circuit_format=circuit_format,
    )


@mcp.tool()
async def ai_linear_function_synthesis_tool(
    circuit: str,
    backend_name: str,
    replace_only_if_better: bool = True,
    local_mode: bool = True,
    circuit_format: CircuitFormat = "qasm3",
) -> dict[str, Any]:
    """AI-powered synthesis for Linear Function circuits (CX and SWAP gate blocks, up to 9 qubits).

    Args:
        circuit: Input quantum circuit as QASM 3.0 string or base64-encoded QPY
        backend_name: Target IBM Quantum backend (e.g., 'ibm_torino', 'ibm_fez')
        replace_only_if_better: If True, only replaces sub-circuits when synthesis improves CNOT count
        local_mode: True runs locally (recommended), False uses remote Qiskit Transpiler Service
        circuit_format: Format of the input circuit - 'qasm3' (default) or 'qpy' (base64-encoded QPY for full circuit fidelity)

    Returns:
        Dict with:
        - status: 'success' or 'error'
        - circuit_qpy: Base64-encoded QPY format (for chaining with other tools)
        - original_circuit: Metrics dict (num_qubits, depth, size, two_qubit_gates)
        - optimized_circuit: Metrics dict for the optimized circuit
        - improvements: Dict with depth_reduction and two_qubit_gate_reduction
    """
    return await ai_linear_function_synthesis(
        circuit=circuit,
        backend_name=backend_name,
        replace_only_if_better=replace_only_if_better,
        local_mode=local_mode,
        circuit_format=circuit_format,
    )


@mcp.tool()
async def ai_clifford_synthesis_tool(
    circuit: str,
    backend_name: str,
    replace_only_if_better: bool = True,
    local_mode: bool = True,
    circuit_format: CircuitFormat = "qasm3",
) -> dict[str, Any]:
    """AI-powered synthesis for Clifford circuits (H, S, and CX gate blocks, up to 9 qubits).

    Args:
        circuit: Input quantum circuit as QASM 3.0 string or base64-encoded QPY
        backend_name: Target IBM Quantum backend (e.g., 'ibm_torino', 'ibm_fez')
        replace_only_if_better: If True, only replaces sub-circuits when synthesis improves CNOT count
        local_mode: True runs locally (recommended), False uses remote Qiskit Transpiler Service
        circuit_format: Format of the input circuit - 'qasm3' (default) or 'qpy' (base64-encoded QPY for full circuit fidelity)

    Returns:
        Dict with:
        - status: 'success' or 'error'
        - circuit_qpy: Base64-encoded QPY format (for chaining with other tools)
        - original_circuit: Metrics dict (num_qubits, depth, size, two_qubit_gates)
        - optimized_circuit: Metrics dict for the optimized circuit
        - improvements: Dict with depth_reduction and two_qubit_gate_reduction
    """
    return await ai_clifford_synthesis(
        circuit=circuit,
        backend_name=backend_name,
        replace_only_if_better=replace_only_if_better,
        local_mode=local_mode,
        circuit_format=circuit_format,
    )


@mcp.tool()
async def ai_permutation_synthesis_tool(
    circuit: str,
    backend_name: str,
    replace_only_if_better: bool = True,
    local_mode: bool = True,
    circuit_format: CircuitFormat = "qasm3",
) -> dict[str, Any]:
    """AI-powered synthesis for Permutation circuits (SWAP gate blocks, supports 27, 33, and 65 qubit blocks).

    Args:
        circuit: Input quantum circuit as QASM 3.0 string or base64-encoded QPY
        backend_name: Target IBM Quantum backend (e.g., 'ibm_torino', 'ibm_fez')
        replace_only_if_better: If True, only replaces sub-circuits when synthesis improves CNOT count
        local_mode: True runs locally (recommended), False uses remote Qiskit Transpiler Service
        circuit_format: Format of the input circuit - 'qasm3' (default) or 'qpy' (base64-encoded QPY for full circuit fidelity)

    Returns:
        Dict with:
        - status: 'success' or 'error'
        - circuit_qpy: Base64-encoded QPY format (for chaining with other tools)
        - original_circuit: Metrics dict (num_qubits, depth, size, two_qubit_gates)
        - optimized_circuit: Metrics dict for the optimized circuit
        - improvements: Dict with depth_reduction and two_qubit_gate_reduction
    """
    return await ai_permutation_synthesis(
        circuit=circuit,
        backend_name=backend_name,
        replace_only_if_better=replace_only_if_better,
        local_mode=local_mode,
        circuit_format=circuit_format,
    )


@mcp.tool()
async def ai_pauli_network_synthesis_tool(
    circuit: str,
    backend_name: str,
    replace_only_if_better: bool = True,
    local_mode: bool = True,
    circuit_format: CircuitFormat = "qasm3",
) -> dict[str, Any]:
    """AI-powered synthesis for Pauli Network circuits (H, S, SX, CX, RX, RY, RZ gate blocks, up to 6 qubits).

    Args:
        circuit: Input quantum circuit as QASM 3.0 string or base64-encoded QPY
        backend_name: Target IBM Quantum backend (e.g., 'ibm_torino', 'ibm_fez')
        replace_only_if_better: If True, only replaces sub-circuits when synthesis improves CNOT count
        local_mode: True runs locally (recommended), False uses remote Qiskit Transpiler Service
        circuit_format: Format of the input circuit - 'qasm3' (default) or 'qpy' (base64-encoded QPY for full circuit fidelity)

    Returns:
        Dict with:
        - status: 'success' or 'error'
        - circuit_qpy: Base64-encoded QPY format (for chaining with other tools)
        - original_circuit: Metrics dict (num_qubits, depth, size, two_qubit_gates)
        - optimized_circuit: Metrics dict for the optimized circuit
        - improvements: Dict with depth_reduction and two_qubit_gate_reduction
    """
    return await ai_pauli_network_synthesis(
        circuit=circuit,
        backend_name=backend_name,
        replace_only_if_better=replace_only_if_better,
        local_mode=local_mode,
        circuit_format=circuit_format,
    )


@mcp.tool()
async def hybrid_ai_transpile_tool(
    circuit: str,
    backend_name: str,
    ai_optimization_level: Literal[1, 2, 3] = 3,
    optimization_level: Literal[1, 2, 3] = 3,
    ai_layout_mode: Literal["keep", "improve", "optimize"] = "optimize",
    circuit_format: CircuitFormat = "qasm3",
) -> dict[str, Any]:
    """Transpile a circuit using a hybrid pass manager combining Qiskit heuristics with AI-powered passes.

    This provides end-to-end transpilation that leverages both classical heuristic optimization
    and AI-based optimization for routing and synthesis in a single unified pipeline.

    Args:
        circuit: Input quantum circuit as QASM 3.0 string or base64-encoded QPY
        backend_name: Target IBM Quantum backend (e.g., 'ibm_torino', 'ibm_fez')
        ai_optimization_level: Optimization level (1-3) for AI components. Higher = better results but slower.
        optimization_level: Optimization level (1-3) for heuristic components.
        ai_layout_mode: Layout selection strategy:
            - 'keep': Respect existing layout (for specific qubit requirements)
            - 'improve': Use prior layout as starting point
            - 'optimize': Best for general circuits (default)
        circuit_format: Format of the input circuit - 'qasm3' (default) or 'qpy' (base64-encoded QPY for full circuit fidelity)

    Returns:
        Dict with:
        - status: 'success' or 'error'
        - circuit_qpy: Base64-encoded QPY format (for chaining with other tools)
        - original_circuit: Metrics dict (num_qubits, depth, size, two_qubit_gates)
        - optimized_circuit: Metrics dict for the optimized circuit
        - improvements: Dict with depth_reduction and two_qubit_gate_reduction
    """
    return await hybrid_ai_transpile(
        circuit=circuit,
        backend_name=backend_name,
        ai_optimization_level=ai_optimization_level,
        optimization_level=optimization_level,
        ai_layout_mode=ai_layout_mode,
        circuit_format=circuit_format,
    )


def main() -> None:
    """Run the server."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
