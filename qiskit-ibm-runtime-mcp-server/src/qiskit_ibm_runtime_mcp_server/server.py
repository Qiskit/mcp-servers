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
Qiskit IBM Runtime MCP Server

A Model Context Protocol server that provides access to IBM Quantum services
through Qiskit IBM Runtime, enabling AI assistants to interact with quantum
computing resources.

Dependencies:
- fastmcp
- qiskit-ibm-runtime
- qiskit
- python-dotenv
"""

import logging
from typing import Any

from fastmcp import FastMCP
from qiskit_mcp_server.circuit_serialization import CircuitFormat

from qiskit_ibm_runtime_mcp_server.ibm_runtime import (
    DDSequenceType,
    cancel_job,
    get_backend_calibration,
    get_backend_properties,
    get_bell_state_circuit,
    get_ghz_state_circuit,
    get_job_status,
    get_quantum_random_circuit,
    get_service_status,
    get_superposition_circuit,
    least_busy_backend,
    list_backends,
    list_my_jobs,
    run_sampler,
    setup_ibm_quantum_account,
)


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize MCP server
mcp = FastMCP("Qiskit IBM Runtime")


# Tools
@mcp.tool()
async def setup_ibm_quantum_account_tool(
    token: str = "", channel: str = "ibm_quantum_platform"
) -> dict[str, Any]:
    """Set up IBM Quantum account with credentials.

    If token is not provided, will attempt to use QISKIT_IBM_TOKEN environment variable
    or saved credentials from ~/.qiskit/qiskit-ibm.json
    """
    return await setup_ibm_quantum_account(token if token else None, channel)


@mcp.tool()
async def list_backends_tool() -> dict[str, Any]:
    """List available IBM Quantum backends."""
    return await list_backends()


@mcp.tool()
async def least_busy_backend_tool() -> dict[str, Any]:
    """Find the least busy operational backend."""
    return await least_busy_backend()


@mcp.tool()
async def get_backend_properties_tool(backend_name: str) -> dict[str, Any]:
    """Get detailed properties of a specific backend.

    Args:
        backend_name: Name of the backend (e.g., 'ibm_brisbane')

    Returns:
        Backend properties including:
        - num_qubits: Number of qubits on the backend
        - simulator: Whether this is a simulator backend
        - operational: Current operational status
        - pending_jobs: Number of jobs in the queue
        - processor_type: Processor family (e.g., 'Eagle r3', 'Heron')
        - backend_version: Backend software version
        - basis_gates: Native gates supported (e.g., ['cx', 'id', 'rz', 'sx', 'x'])
        - coupling_map: Qubit connectivity as list of [control, target] pairs
        - max_shots: Maximum shots per circuit execution
        - max_experiments: Maximum circuits per job

    Note:
        For time-varying calibration data (T1, T2, gate errors, faulty qubits),
        use get_backend_calibration_tool instead.
    """
    return await get_backend_properties(backend_name)


@mcp.tool()
async def get_backend_calibration_tool(
    backend_name: str, qubit_indices: list[int] | None = None
) -> dict[str, Any]:
    """Get calibration data for a backend including T1, T2 times and error rates.

    Args:
        backend_name: Name of the backend (e.g., 'ibm_brisbane')
        qubit_indices: Optional list of specific qubit indices to get data for.
                      If not provided, returns data for the first 10 qubits.

    Returns:
        Calibration data including:
        - T1 and T2 coherence times (in microseconds)
        - Qubit frequency (in GHz)
        - Readout errors for each qubit
        - Gate errors for common gates (x, sx, cx, etc.)
        - faulty_qubits: List of non-operational qubit indices
        - faulty_gates: List of non-operational gates with affected qubits
        - Last calibration timestamp

    Note:
        For static backend info (processor_type, backend_version, quantum_volume),
        use get_backend_properties_tool instead.
    """
    return await get_backend_calibration(backend_name, qubit_indices)


@mcp.tool()
async def list_my_jobs_tool(limit: int = 10) -> dict[str, Any]:
    """List user's recent jobs."""
    return await list_my_jobs(limit)


@mcp.tool()
async def get_job_status_tool(job_id: str) -> dict[str, Any]:
    """Get status of a specific job."""
    return await get_job_status(job_id)


@mcp.tool()
async def cancel_job_tool(job_id: str) -> dict[str, Any]:
    """Cancel a specific job."""
    return await cancel_job(job_id)


@mcp.tool()
async def run_sampler_tool(
    circuit: str,
    backend_name: str | None = None,
    shots: int = 4096,
    circuit_format: CircuitFormat = "auto",
    dynamical_decoupling: bool = True,
    dd_sequence: DDSequenceType = "XY4",
    twirling: bool = True,
    measure_twirling: bool = True,
) -> dict[str, Any]:
    """Run a quantum circuit using the Qiskit Runtime SamplerV2 primitive.

    The Sampler primitive executes quantum circuits and returns measurement outcome
    samples. This is the primary way to run quantum circuits on IBM Quantum hardware.

    Error Mitigation (enabled by default):
        - Dynamical Decoupling: Suppresses decoherence during idle periods
        - Twirling: Randomizes errors into stochastic noise for better results

    Args:
        circuit: The quantum circuit to execute. Accepts multiple formats:
                - OpenQASM 3.0 string (recommended):
                  ```
                  OPENQASM 3.0;
                  include "stdgates.inc";
                  qubit[2] q;
                  bit[2] c;
                  h q[0];
                  cx q[0], q[1];
                  c = measure q;
                  ```
                - OpenQASM 2.0 string (legacy, auto-detected)
                - Base64-encoded QPY binary (for tool chaining with transpiler output)
                Must include measurement operations to produce results.
        backend_name: Name of the IBM Quantum backend (e.g., 'ibm_brisbane').
                     If not provided, uses the least busy operational backend.
        shots: Number of measurement shots (repetitions). Default is 4096.
               Higher values give more statistical accuracy.
        circuit_format: Format of the circuit input. Options:
                       - "auto" (default): Automatically detect format
                       - "qasm3": OpenQASM 3.0/2.0 text format
                       - "qpy": Base64-encoded QPY binary format
        dynamical_decoupling: Enable dynamical decoupling to suppress decoherence
                             during idle periods. Default is True (recommended).
        dd_sequence: Dynamical decoupling pulse sequence. Options:
                    - "XX": Basic X-X sequence
                    - "XpXm": X+/X- sequence
                    - "XY4": 4-pulse XY sequence (default, most robust)
        twirling: Enable Pauli twirling on 2-qubit gates to convert coherent
                 errors into stochastic noise. Default is True (recommended).
        measure_twirling: Enable twirling on measurements for readout error
                         mitigation. Default is True (recommended).

    Returns:
        Job submission status including:
        - job_id: Use with get_job_status_tool to check completion
        - backend: The backend where the circuit will run
        - shots: Number of shots requested
        - error_mitigation: Summary of enabled techniques

    Note:
        Jobs run asynchronously. Use get_job_status_tool to monitor progress.
        Results contain measurement bitstrings and their occurrence counts.
    """
    return await run_sampler(
        circuit,
        backend_name,
        shots,
        circuit_format,
        dynamical_decoupling,
        dd_sequence,
        twirling,
        measure_twirling,
    )


# Resources
@mcp.resource("ibm://status", mime_type="text/plain")
async def get_service_status_resource() -> str:
    """Get current IBM Quantum service status."""
    return await get_service_status()


# Example Circuit Resources - Pre-built circuits for easy LLM usage
@mcp.resource("circuits://bell-state", mime_type="application/json")
def get_bell_state_resource() -> dict[str, Any]:
    """Get a ready-to-run Bell state (quantum entanglement) circuit.

    Returns a 2-qubit circuit that creates the Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2.
    This is the simplest demonstration of quantum entanglement.

    The returned circuit field can be passed directly to run_sampler_tool.
    Expected results: ~50% '00' and ~50% '11', never '01' or '10'.
    """
    return get_bell_state_circuit()


@mcp.resource("circuits://ghz-state", mime_type="application/json")
def get_ghz_state_resource() -> dict[str, Any]:
    """Get a ready-to-run 3-qubit GHZ state (multi-qubit entanglement) circuit.

    Returns a circuit that creates the GHZ state |GHZ⟩ = (|000⟩ + |111⟩)/√2.
    This generalizes the Bell state to demonstrate 3-qubit entanglement.

    The returned circuit field can be passed directly to run_sampler_tool.
    Expected results: ~50% '000' and ~50% '111', no other outcomes.
    """
    return get_ghz_state_circuit(3)


@mcp.resource("circuits://random", mime_type="application/json")
def get_random_circuit_resource() -> dict[str, Any]:
    """Get a ready-to-run quantum random number generator circuit.

    Returns a 4-qubit circuit that generates truly random bits using quantum
    superposition. Each qubit is put in superposition and measured.

    The returned circuit field can be passed directly to run_sampler_tool.
    Expected results: All 16 outcomes (0000-1111) with ~6.25% probability each.
    """
    return get_quantum_random_circuit()


@mcp.resource("circuits://superposition", mime_type="application/json")
def get_superposition_resource() -> dict[str, Any]:
    """Get the simplest possible quantum circuit - single qubit superposition.

    Returns a 1-qubit circuit that demonstrates quantum superposition by
    applying a Hadamard gate to create (|0⟩ + |1⟩)/√2.

    The returned circuit field can be passed directly to run_sampler_tool.
    Expected results: ~50% '0' and ~50% '1'.
    """
    return get_superposition_circuit()


def main() -> None:
    """Run the server."""
    mcp.run()


if __name__ == "__main__":
    main()


# Assisted by watsonx Code Assistant
