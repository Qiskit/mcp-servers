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
    coupling_map: list[list[int]] | None = None,
    circuit_format: CircuitFormat = "qasm3",
) -> dict[str, Any]:
    """Route a quantum circuit by inserting SWAP operations for backend compatibility. Use this FIRST before other synthesis tools.

    Args:
        circuit: Input quantum circuit as QASM 3.0 string or base64-encoded QPY
        backend_name: Target IBM Quantum backend (e.g., 'ibm_boston', 'ibm_fez')
        optimization_level: 1 (fastest, least optimization) to 3 (slowest, most optimization)
        layout_mode: 'keep' (respect existing layout), 'improve' (refine initial guess), 'optimize' (best for general circuits)
        optimization_preferences: What to minimize - 'n_cnots', 'n_gates', 'cnot_layers', 'layers', or 'noise'. Can be a list.
        local_mode: True runs locally (recommended), False uses remote Qiskit Transpiler Service
        coupling_map: Optional list of qubit pairs representing the backend topology.
            If provided, overrides the backend's coupling map. Useful for targeting a
            specific subset of qubits.
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
        coupling_map=coupling_map,
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
        backend_name: Target IBM Quantum backend (e.g., 'ibm_boston', 'ibm_fez')
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
        backend_name: Target IBM Quantum backend (e.g., 'ibm_boston', 'ibm_fez')
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
        backend_name: Target IBM Quantum backend (e.g., 'ibm_boston', 'ibm_fez')
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
        backend_name: Target IBM Quantum backend (e.g., 'ibm_boston', 'ibm_fez')
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
    initial_layout: list[int] | None = None,
    coupling_map: list[list[int]] | None = None,
    circuit_format: CircuitFormat = "qasm3",
) -> dict[str, Any]:
    """Transpile a circuit using a hybrid pass manager combining Qiskit heuristics with AI-powered passes.

    This provides end-to-end transpilation that leverages both classical heuristic optimization
    and AI-based optimization for routing and synthesis in a single unified pipeline.

    Args:
        circuit: Input quantum circuit as QASM 3.0 string or base64-encoded QPY
        backend_name: Target IBM Quantum backend (e.g., 'ibm_boston', 'ibm_fez')
        ai_optimization_level: Optimization level (1-3) for AI components. Higher = better results but slower.
        optimization_level: Optimization level (1-3) for heuristic components.
        ai_layout_mode: Layout selection strategy:
            - 'keep': Respect existing layout (for specific qubit requirements)
            - 'improve': Use prior layout as starting point
            - 'optimize': Best for general circuits (default)
            Note: If initial_layout is provided with 'optimize', it automatically converts
            to 'improve' to leverage the user-provided layout.
        initial_layout: Optional list of physical qubit indices specifying where to place
            virtual qubits. For example, [0, 1, 5, 6, 7] maps virtual qubit 0 to physical
            qubit 0, virtual qubit 1 to physical qubit 1, etc.
        coupling_map: Optional list of qubit pairs representing the backend topology.
            If provided, overrides the backend's coupling map. Useful for targeting a
            specific subset of qubits.
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
        initial_layout=initial_layout,
        coupling_map=coupling_map,
        circuit_format=circuit_format,
    )


##################################################
## MCP Prompts
## - https://modelcontextprotocol.io/docs/concepts/prompts
##################################################


@mcp.prompt()
def transpile_circuit(circuit: str, backend_name: str) -> str:
    """Transpile a quantum circuit for an IBM backend using AI-powered routing and synthesis."""
    return (
        f"Transpile the circuit for backend '{backend_name}': "
        "1) Call setup_ibm_quantum_account_tool to authenticate, "
        f"2) Call ai_routing_tool with the circuit and backend_name='{backend_name}' "
        "to route the circuit for the target topology, "
        "3) Optionally apply ai_clifford_synthesis_tool or "
        "ai_linear_function_synthesis_tool on the routed circuit_qpy to further optimize, "
        "4) Report the improvements from the metrics in each response."
    )


@mcp.prompt()
def optimize_circuit(circuit: str, backend_name: str) -> str:
    """Run end-to-end AI-powered transpilation on a circuit for an IBM backend."""
    return (
        f"Run full AI-powered transpilation on the circuit for backend '{backend_name}': "
        "1) Call setup_ibm_quantum_account_tool if not already authenticated, "
        f"2) Call hybrid_ai_transpile_tool with the circuit and backend_name='{backend_name}' "
        "to perform routing and synthesis in one pass, "
        "3) Report the depth and gate count improvements from the response metrics."
    )


@mcp.prompt()
def explain_synthesis_type(synthesis_type: str) -> str:
    """Explain a specific AI synthesis pass type and when to use it."""
    return (
        "Read the qiskit-ibm-transpiler://synthesis-types resource to find information "
        f"about the '{synthesis_type}' synthesis type, then explain what kinds of circuits "
        "it applies to, its qubit limits, and when to use it in a transpilation pipeline."
    )


##################################################
## MCP Resources
## - https://modelcontextprotocol.io/docs/concepts/resources
##################################################


@mcp.resource("qiskit-ibm-transpiler://info", mime_type="application/json")
def transpiler_info_resource() -> dict[str, Any]:
    """Get information about the Qiskit IBM Transpiler server capabilities."""
    return {
        "status": "success",
        "server": "Qiskit IBM Transpiler",
        "description": (
            "AI-powered quantum circuit transpilation using IBM's cloud AI passes "
            "for routing and synthesis on IBM Quantum backends."
        ),
        "workflow": [
            "1. setup_ibm_quantum_account_tool - authenticate with IBM Quantum",
            "2. ai_routing_tool - route circuit for target backend",
            "3. Optionally apply synthesis tools on the routed circuit_qpy",
            "4. hybrid_ai_transpile_tool - alternative end-to-end transpilation",
        ],
    }


@mcp.resource("qiskit-ibm-transpiler://synthesis-types", mime_type="application/json")
def synthesis_types_resource() -> dict[str, Any]:
    """Get documentation for the AI synthesis pass types supported by this server."""
    return {
        "status": "success",
        "synthesis_types": {
            "linear_function": {
                "tool": "ai_linear_function_synthesis_tool",
                "description": "AI synthesis for linear function circuits (CX and SWAP gates)",
                "max_qubits": 9,
                "input_gates": ["cx", "swap"],
            },
            "clifford": {
                "tool": "ai_clifford_synthesis_tool",
                "description": "AI synthesis for Clifford circuits (H, S, CX gates)",
                "max_qubits": 9,
                "input_gates": ["h", "s", "cx"],
            },
            "permutation": {
                "tool": "ai_permutation_synthesis_tool",
                "description": "AI synthesis for permutation circuits (SWAP gates)",
                "supported_qubit_counts": [27, 33, 65],
                "input_gates": ["swap"],
            },
            "pauli_network": {
                "tool": "ai_pauli_network_synthesis_tool",
                "description": "AI synthesis for Pauli network circuits",
                "max_qubits": 6,
                "input_gates": ["h", "s", "sx", "cx", "rx", "ry", "rz"],
            },
        },
    }


def main() -> None:
    """Run the server."""
    mcp.run(transport="stdio", show_banner=False)


if __name__ == "__main__":
    main()
