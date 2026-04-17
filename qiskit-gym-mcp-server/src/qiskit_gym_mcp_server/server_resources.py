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

"""MCP Resource definitions for Qiskit Gym server.

This module contains all @mcp.resource() decorated functions that expose
qiskit-gym information via the Model Context Protocol.
"""

from typing import Any

from qiskit_gym_mcp_server.app import mcp
from qiskit_gym_mcp_server.constants import (
    QISKIT_GYM_MAX_ITERATIONS,
    QISKIT_GYM_MAX_QUBITS,
    QISKIT_GYM_MAX_SEARCHES,
    QISKIT_GYM_MODEL_DIR,
    QISKIT_GYM_TENSORBOARD_DIR,
)
from qiskit_gym_mcp_server.coupling_maps import (
    HARDWARE_PRESETS,
    get_coupling_map_presets,
)
from qiskit_gym_mcp_server.gym_core import get_environment_info
from qiskit_gym_mcp_server.models import get_model_info, list_loaded_models
from qiskit_gym_mcp_server.training import (
    get_available_algorithms,
    get_available_policies,
    get_training_status,
    list_training_sessions,
)


# ============================================================================
# Resources
# ============================================================================


@mcp.resource("qiskit-gym://presets/coupling-maps", mime_type="application/json")
async def coupling_map_presets_resource() -> dict[str, Any]:
    """Available hardware coupling map presets.

    Returns preset names, descriptions, and specifications for IBM Quantum
    hardware topologies (Heron, Nighthawk) and common grid/linear configurations.
    """
    return await get_coupling_map_presets()


@mcp.resource("qiskit-gym://algorithms", mime_type="application/json")
async def algorithms_resource() -> dict[str, Any]:
    """Available RL algorithms for training.

    Returns information about PPO and AlphaZero algorithms including
    descriptions and recommended use cases.
    """
    return await get_available_algorithms()


@mcp.resource("qiskit-gym://policies", mime_type="application/json")
async def policies_resource() -> dict[str, Any]:
    """Available policy network architectures.

    Returns information about Basic and Conv1D policy networks.
    """
    return await get_available_policies()


@mcp.resource("qiskit-gym://environments", mime_type="application/json")
async def environments_info_resource() -> dict[str, Any]:
    """Environment type documentation.

    Returns information about available environment types (Permutation,
    LinearFunction, Clifford) and their use cases.
    """
    return {
        "status": "success",
        "environment_types": {
            "permutation": {
                "name": "PermutationGym",
                "description": "Learn optimal qubit routing using SWAP gates",
                "use_case": "Routing qubits on constrained coupling maps",
                "input": "Target qubit permutation",
                "output": "SWAP gate sequence",
                "create_tool": "create_permutation_env_tool",
            },
            "linear_function": {
                "name": "LinearFunctionGym",
                "description": "Learn optimal CNOT synthesis for linear functions",
                "use_case": "Decomposing linear Boolean functions",
                "input": "Binary matrix representing linear function",
                "output": "CNOT circuit",
                "create_tool": "create_linear_function_env_tool",
            },
            "clifford": {
                "name": "CliffordGym",
                "description": "Learn optimal Clifford circuit synthesis",
                "use_case": "Synthesizing Clifford group elements",
                "input": "Clifford tableau",
                "output": "Clifford circuit with custom gateset",
                "create_tool": "create_clifford_env_tool",
            },
        },
    }


@mcp.resource("qiskit-gym://training/sessions", mime_type="application/json")
async def training_sessions_resource() -> dict[str, Any]:
    """Active training sessions.

    Returns list of all training sessions with their status and progress.
    """
    return await list_training_sessions()


@mcp.resource("qiskit-gym://models", mime_type="application/json")
async def models_resource() -> dict[str, Any]:
    """Available loaded models.

    Returns list of models currently loaded in memory and ready for synthesis.
    """
    return await list_loaded_models()


@mcp.resource("qiskit-gym://server/config", mime_type="application/json")
async def server_config_resource() -> dict[str, Any]:
    """Server configuration.

    Returns current server configuration including limits and directories.
    """
    return {
        "status": "success",
        "configuration": {
            "model_directory": QISKIT_GYM_MODEL_DIR,
            "tensorboard_directory": QISKIT_GYM_TENSORBOARD_DIR,
            "max_iterations": QISKIT_GYM_MAX_ITERATIONS,
            "max_qubits": QISKIT_GYM_MAX_QUBITS,
            "max_searches": QISKIT_GYM_MAX_SEARCHES,
        },
        "hardware_presets": list(HARDWARE_PRESETS.keys()),
    }


@mcp.resource("qiskit-gym://workflows", mime_type="application/json")
async def workflows_resource() -> dict[str, Any]:
    """Common workflows for using qiskit-gym MCP server.

    READ THIS FIRST to understand how to use this server effectively.
    Returns step-by-step workflows for common tasks.
    """
    return {
        "status": "success",
        "overview": (
            "This server trains RL models to synthesize optimal quantum circuits. "
            "Workflow: 1) Create environment → 2) Train → 3) Save → 4) Synthesize."
        ),
        "quick_start": {
            "description": "Fastest way to train and use a model",
            "steps": [
                "1. create_clifford_env_tool(num_qubits=4, preset='grid_2x2') → env_id",
                "2. start_training_tool(env_id, algorithm='ppo', num_iterations=100) → model_id",
                "3. generate_random_clifford_tool(num_qubits=4) → clifford_tableau",
                "4. synthesize_clifford_tool(model_id, clifford_tableau) → circuit_qpy",
                "5. convert_qpy_to_qasm3_tool(circuit_qpy) → human-readable circuit",
            ],
        },
        "workflows": {
            "train_on_hardware_subtopologies": {
                "description": "Train models for IBM Nighthawk or Heron hardware",
                "steps": [
                    "1. extract_subtopologies_tool(preset='ibm_nighthawk', num_qubits=6)",
                    "2. For each subtopology: create_*_env_tool(coupling_map=edges)",
                    "3. start_training_tool(env_id) for each",
                    "4. save_model_tool(session_id, model_name='nighthawk_6q_v1')",
                ],
            },
            "load_and_synthesize": {
                "description": "Use a previously saved model",
                "steps": [
                    "1. list_saved_models_tool() → see available models",
                    "2. load_model_tool(model_name) → model_id",
                    "3. synthesize_*_tool(model_id, input) → circuit",
                ],
            },
        },
        "environment_types": {
            "permutation": {
                "purpose": "Qubit routing with SWAP gates",
                "create": "create_permutation_env_tool",
                "synthesize": "synthesize_permutation_tool",
                "input": "List[int] - target qubit positions, e.g. [2, 0, 1]",
            },
            "linear_function": {
                "purpose": "CNOT synthesis for linear functions",
                "create": "create_linear_function_env_tool",
                "synthesize": "synthesize_linear_function_tool",
                "input": "NxN binary matrix",
            },
            "clifford": {
                "purpose": "General Clifford circuit synthesis",
                "create": "create_clifford_env_tool",
                "synthesize": "synthesize_clifford_tool",
                "input": "Clifford tableau (use generate_random_clifford_tool for testing)",
            },
        },
        "tips": [
            "Start with small presets (grid_2x2, linear_3) to verify workflow",
            "PPO is faster; AlphaZero gives better results on hard problems",
            "100-500 iterations usually sufficient for ≤6 qubits",
            "Always save models to avoid losing trained weights",
            "Increase num_searches (up to 10000) for better synthesis results",
        ],
    }


##################################################
## MCP Prompts
## - https://modelcontextprotocol.io/docs/concepts/prompts
##################################################


@mcp.prompt()
def train_synthesis_model(env_type: str, num_qubits: str) -> str:
    """Train a reinforcement learning model for quantum circuit synthesis."""
    return (
        f"Train an RL model for {env_type} synthesis on {num_qubits} qubits: "
        f"1) Call create_{env_type}_env_tool with appropriate parameters for "
        f"{num_qubits} qubits to create a training environment, "
        "2) Call start_training_tool with the returned env_id and algorithm='ppo', "
        "3) Call get_training_status_tool with the session_id to monitor progress, "
        "4) When training completes, call save_model_tool to persist the trained model."
    )


@mcp.prompt()
def synthesize_circuit(circuit_type: str) -> str:
    """Synthesize a quantum circuit using a trained RL model."""
    return (
        f"Synthesize a {circuit_type} circuit using a trained model: "
        "1) Call list_saved_models_tool to see available models, "
        f"2) Call load_model_tool with a suitable model_name for {circuit_type} synthesis, "
        f"3) Generate a test input using generate_random_{circuit_type}_tool, "
        f"4) Call synthesize_{circuit_type}_tool with the model_id and the generated input, "
        "5) Call convert_qpy_to_qasm3_tool to view the resulting circuit."
    )


@mcp.prompt()
def explore_hardware_topology(backend_preset: str) -> str:
    """Explore hardware topology and extract subtopologies for training."""
    return (
        f"Explore the '{backend_preset}' hardware topology: "
        "1) Read the qiskit-gym://presets/coupling-maps resource to see available presets, "
        f"2) Call extract_subtopologies_tool with preset='{backend_preset}' "
        "to find connected subtopologies, "
        "3) Call list_subtopology_shapes_tool to summarize the shapes found, "
        "4) Create environments for each unique subtopology using create_permutation_env_tool."
    )


##################################################
## MCP Resource Templates
## - https://modelcontextprotocol.io/docs/concepts/resources#resource-templates
##################################################


@mcp.resource("qiskit-gym://environments/{env_id}", mime_type="application/json")
async def environment_info_resource(env_id: str) -> dict[str, Any]:
    """Get detailed information about a specific gym environment."""
    return await get_environment_info(env_id)


@mcp.resource("qiskit-gym://models/{model_name}", mime_type="application/json")
async def model_info_resource(model_name: str) -> dict[str, Any]:
    """Get information about a specific trained model."""
    return await get_model_info(model_name=model_name)


@mcp.resource("qiskit-gym://training/{session_id}", mime_type="application/json")
async def training_status_resource(session_id: str) -> dict[str, Any]:
    """Get the status and metrics of a specific training session."""
    return await get_training_status(session_id)
