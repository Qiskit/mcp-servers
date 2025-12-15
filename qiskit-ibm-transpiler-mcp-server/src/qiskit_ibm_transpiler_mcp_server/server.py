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
)
from qiskit_ibm_transpiler_mcp_server.utils import setup_ibm_quantum_account


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
    token: str | None = None, channel: str = "ibm_quantum_platform"
) -> dict[str, Any]:
    """Set up IBM Quantum account with credentials.

    If token is not provided, will attempt to use QISKIT_IBM_TOKEN environment variable
    or saved credentials from ~/.qiskit/qiskit-ibm.json
    """
    return await setup_ibm_quantum_account(token, channel)


@mcp.tool()
async def ai_routing_tool(
    circuit_qasm: str,
    backend_name: str,
    optimization_level: int = 1,
    layout_mode: str = "optimize",
    optimization_preferences: Literal["n_cnots", "n_gates", "cnot_layers", "layers", "noise"]
    | list[Literal["n_cnots", "n_gates", "cnot_layers", "layers", "noise"]]
    | None = None,
    local_mode: bool = True,
) -> dict[str, Any]:
    """
    This tool acts both as a layout stage and a routing stage. It inserts SWAP operations on a circuit to make two-qubits operations compatible with a given coupling map that restricts the pair of qubits on which operations can be applied.
    It should be used as an initial step before any other AI synthesis routine.
    It returns the routed quantum circuit as QASM 3.0 string.
    """

    return await ai_routing(
        circuit_qasm=circuit_qasm,
        backend_name=backend_name,
        optimization_level=optimization_level,
        layout_mode=layout_mode,
        optimization_preferences=optimization_preferences,
        local_mode=local_mode,
    )


@mcp.tool()
async def ai_linear_function_synthesis_tool(
    circuit_qasm: str,
    backend_name: str,
    replace_only_if_better: bool = True,
    local_mode: bool = True,
) -> dict[str, Any]:
    """
    Synthesis for Linear Function circuits (blocks of CX and SWAP gates). Currently, up to nine qubit blocks.
    """
    return await ai_linear_function_synthesis(
        circuit_qasm=circuit_qasm,
        backend_name=backend_name,
        replace_only_if_better=replace_only_if_better,
        local_mode=local_mode,
    )


@mcp.tool()
async def ai_clifford_synthesis_tool(
    circuit_qasm: str,
    backend_name: str,
    replace_only_if_better: bool = True,
    local_mode: bool = True,
) -> dict[str, Any]:
    """
    Synthesis for Clifford circuits (blocks of H, S, and CX gates). Currently, up to nine qubit blocks.
    """

    return await ai_clifford_synthesis(
        circuit_qasm=circuit_qasm,
        backend_name=backend_name,
        replace_only_if_better=replace_only_if_better,
        local_mode=local_mode,
    )


@mcp.tool()
async def ai_permutation_synthesis_tool(
    circuit_qasm: str,
    backend_name: str,
    replace_only_if_better: bool = True,
    local_mode: bool = True,
) -> dict[str, Any]:
    """
    Synthesis for Permutation circuits (blocks of SWAP gates). Currently available for 65, 33, and 27 qubit blocks.
    """
    return await ai_permutation_synthesis(
        circuit_qasm=circuit_qasm,
        backend_name=backend_name,
        replace_only_if_better=replace_only_if_better,
        local_mode=local_mode,
    )


@mcp.tool()
async def ai_pauli_network_synthesis_tool(
    circuit_qasm: str,
    backend_name: str,
    replace_only_if_better: bool = True,
    local_mode: bool = True,
) -> dict[str, Any]:
    """
    Synthesis for Pauli Network circuits (blocks of H, S, SX, CX, RX, RY and RZ gates). Currently, up to six qubit blocks.
    """
    return await ai_pauli_network_synthesis(
        circuit_qasm=circuit_qasm,
        backend_name=backend_name,
        replace_only_if_better=replace_only_if_better,
        local_mode=local_mode,
    )


def main() -> None:
    """Run the server."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
