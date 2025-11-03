from fastmcp import FastMCP
from typing import Literal, Dict, Any

from qiskit_ibm_transpiler_mcp_server.ibm_runtime import (
    setup_ibm_quantum_account,
    list_backends,
    least_busy_backend,
    get_backend_properties,
    list_my_jobs,
    get_job_status,
    cancel_job,
)

from qiskit_ibm_transpiler_mcp_server.qta import (
    ai_routing,
    ai_linear_function_synthesis,
    ai_clifford_synthesis,
    ai_pauli_network_synthesis,
    ai_permutation_synthesis,
)
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastMCP server
mcp = FastMCP("Qiskit Transpiler Assistant")

logger.info("Qiskit Transpiler Assistant MCP Server initialized")

##################################################
## MCP Tools
## - https://modelcontextprotocol.io/docs/concepts/tools
##################################################


# Tools
@mcp.tool()
async def setup_ibm_quantum_account_tool(
    token: str = "", channel: str = "ibm_quantum_platform"
):
    """Set up IBM Quantum account with credentials.

    If token is not provided, will attempt to use IBM_QUANTUM_TOKEN environment variable
    or saved credentials from ~/.qiskit/qiskit-ibm.json
    """
    return await setup_ibm_quantum_account(token if token else None, channel)


@mcp.tool()
async def list_backends_tool():
    """List available IBM Quantum backends."""
    return await list_backends()


@mcp.tool()
async def get_backend_properties_tool(backend_name: str):
    """Get detailed properties of a specific backend."""
    return await get_backend_properties(backend_name)


@mcp.tool()
async def list_my_jobs_tool(limit: int = 10):
    """List user's recent jobs."""
    return await list_my_jobs(limit)


@mcp.tool()
async def get_job_status_tool(job_id: str):
    """Get status of a specific job."""
    return await get_job_status(job_id)


@mcp.tool()
async def cancel_job_tool(job_id: str):
    """Cancel a specific job."""
    return await cancel_job(job_id)


@mcp.tool()
async def least_busy_backend_tool():
    """Find the least busy operational backend."""
    return await least_busy_backend()


@mcp.tool()
async def ai_routing_tool(
    circuit_qasm: str,
    backend_name: str,
    optimization_level: int = 1,
    layout_mode: str = "optimize",
    optimization_preferences: Literal[
        "n_cnots", "n_gates", "cnot_layers", "layers", "noise"
    ]
    | list[Literal["n_cnots", "n_gates", "cnot_layers", "layers", "noise"]]
    | None = None,
    local_mode: bool = True,
) -> Dict[str, Any]:
    """
    This tool acts both as a layout stage and a routing stage. It inserts SWAP operations on a circuit to make two-qubits operations compatible with a given coupling map that restricts the pair of qubits on which operations can be applied.
    It should be used as an initial step before any other AI synthesis routine.
    It returns the routed quantum circuit as QASM3.0 string.
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
) -> Dict[str, Any]:
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
) -> Dict[str, Any]:
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
) -> Dict[str, Any]:
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
) -> Dict[str, Any]:
    """
    Synthesis for Pauli Network circuits (blocks of H, S, SX, CX, RX, RY and RZ gates). Currently, up to six qubit blocks.
    """
    return await ai_pauli_network_synthesis(
        circuit_qasm=circuit_qasm,
        backend_name=backend_name,
        replace_only_if_better=replace_only_if_better,
        local_mode=local_mode,
    )


def main():
    """Run the server."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
