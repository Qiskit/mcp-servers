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

"""MCP Tool definitions for Qiskit Gym server.

This module contains all @mcp.tool() decorated functions that expose
qiskit-gym functionality via the Model Context Protocol.
"""

import atexit
import subprocess  # nosec B404 - subprocess needed for TensorBoard process management
import time
from typing import Any, Literal

from qiskit_mcp_server.circuit_serialization import qasm3_to_qpy, qpy_to_qasm3

from qiskit_gym_mcp_server.app import mcp
from qiskit_gym_mcp_server.constants import QISKIT_GYM_TENSORBOARD_DIR
from qiskit_gym_mcp_server.coupling_maps import (
    create_custom_coupling_map,
    extract_subtopologies,
    get_fake_backend_coupling_map,
    list_available_fake_backends,
    list_subtopology_shapes,
)
from qiskit_gym_mcp_server.gym_core import (
    create_clifford_environment,
    create_linear_function_environment,
    create_permutation_environment,
    delete_environment,
    get_environment_info,
    list_environments,
)
from qiskit_gym_mcp_server.models import (
    delete_model,
    get_model_info,
    list_loaded_models,
    list_saved_models,
    load_model,
    save_model,
)
from qiskit_gym_mcp_server.synthesis import (
    generate_random_clifford,
    generate_random_linear_function,
    generate_random_permutation,
    synthesize_clifford,
    synthesize_linear_function,
    synthesize_permutation,
)
from qiskit_gym_mcp_server.training import (
    batch_train_environments,
    get_tensorboard_metrics,
    get_training_metrics,
    get_training_status,
    list_tensorboard_experiments,
    list_training_sessions,
    start_training,
    stop_training,
    wait_for_training,
)


# ============================================================================
# Environment Management Tools
# ============================================================================


@mcp.tool()
async def create_permutation_env_tool(
    coupling_map: list[list[int]] | None = None,
    preset: str | None = None,
) -> dict[str, Any]:
    """Create a PermutationGym environment for learning qubit routing with SWAP gates.

    Use this to create an RL environment that learns to implement arbitrary qubit
    permutations using minimal SWAP gates on constrained coupling maps.

    Args:
        coupling_map: Custom coupling map as list of [qubit1, qubit2] edges.
            Example: [[0,1], [1,2], [2,3]] for a linear chain.
        preset: Hardware preset name (use instead of coupling_map).
            Options: "ibm_heron_r1", "ibm_heron_r2", "ibm_nighthawk",
                    "grid_3x3", "grid_5x5", "linear_5", "linear_10"

    Returns:
        Dict with env_id, env_type, num_qubits, action_space_size on success.

    Note:
        Either coupling_map OR preset must be provided, not both.
    """
    return await create_permutation_environment(
        coupling_map=coupling_map,
        preset=preset,
    )


@mcp.tool()
async def create_linear_function_env_tool(
    coupling_map: list[list[int]] | None = None,
    preset: str | None = None,
    basis_gates: list[str] | None = None,
) -> dict[str, Any]:
    """Create a LinearFunctionGym environment for learning CNOT synthesis.

    The environment learns to decompose linear Boolean functions into efficient
    quantum circuits using CNOT gates.

    Args:
        coupling_map: Custom coupling map as list of [qubit1, qubit2] edges.
        preset: Hardware preset name (e.g., "ibm_nighthawk", "grid_3x3")
        basis_gates: Optional list of basis gates (default: ["cx"])

    Returns:
        Dict with env_id, environment info on success.
    """
    return await create_linear_function_environment(
        coupling_map=coupling_map,
        preset=preset,
        basis_gates=basis_gates,
    )


@mcp.tool()
async def create_clifford_env_tool(
    num_qubits: int,
    coupling_map: list[list[int]] | None = None,
    preset: str | None = None,
    gateset: list[str | dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Create a CliffordGym environment for learning Clifford circuit synthesis.

    The environment learns to synthesize optimal implementations of Clifford
    group elements using a customizable gate set.

    Args:
        num_qubits: Number of qubits for the environment
        coupling_map: Custom coupling map (optional)
        preset: Hardware preset name (optional)
        gateset: Custom gate set. Can be:
            - List of gate names: ["H", "S", "CX"] (applied to all qubits/edges)
            - List of dicts: [{"gate": "H", "qubits": [0]}, {"gate": "CX", "qubits": [0, 1]}]

    Returns:
        Dict with env_id, environment info on success.
    """
    return await create_clifford_environment(
        num_qubits=num_qubits,
        coupling_map=coupling_map,
        preset=preset,
        gateset=gateset,
    )


@mcp.tool()
async def list_environments_tool() -> dict[str, Any]:
    """List all active RL environments.

    Returns:
        Dict with list of environments and their info.
    """
    return await list_environments()


@mcp.tool()
async def get_environment_info_tool(env_id: str) -> dict[str, Any]:
    """Get detailed information about a specific environment.

    Args:
        env_id: Environment ID

    Returns:
        Dict with environment details.
    """
    return await get_environment_info(env_id)


@mcp.tool()
async def delete_environment_tool(env_id: str) -> dict[str, Any]:
    """Delete an environment.

    Args:
        env_id: Environment ID to delete

    Returns:
        Dict with deletion status.
    """
    return await delete_environment(env_id)


# ============================================================================
# Training Tools
# ============================================================================


@mcp.tool()
async def start_training_tool(
    env_id: str,
    algorithm: Literal["ppo", "alphazero"] = "ppo",
    policy: Literal["basic", "conv1d"] = "basic",
    num_iterations: int = 100,
    tensorboard_experiment: str | None = None,
    background: bool = False,
) -> dict[str, Any]:
    """Start training an RL agent on a created environment.

    This initiates training that learns to synthesize optimal circuits.

    Args:
        env_id: Environment ID from create_*_env_tool
        algorithm: RL algorithm to use:
            - "ppo": Proximal Policy Optimization (recommended, faster)
            - "alphazero": AlphaZero-style MCTS (better for complex problems)
        policy: Neural network policy architecture:
            - "basic": Simple feedforward network (faster, good for small problems)
            - "conv1d": 1D convolutional network (better for larger problems)
        num_iterations: Number of training iterations. Default: 100
        tensorboard_experiment: Name for TensorBoard logging (optional)
        background: If True, run training in background and return immediately.
            Use get_training_status_tool to poll progress, or wait_for_training_tool
            to block until done. Default: False (synchronous).

    Returns:
        If background=False: Dict with session_id, model_id, training metrics.
        If background=True: Dict with session_id for polling. Use
            wait_for_training_tool or get_training_status_tool to check progress.

    Note:
        For long training runs (>100 iterations), set background=True to avoid
        timeouts, then use wait_for_training_tool to get results.
    """
    return await start_training(
        env_id=env_id,
        algorithm=algorithm,
        policy=policy,
        num_iterations=num_iterations,
        tensorboard_experiment=tensorboard_experiment,
        background=background,
    )


@mcp.tool()
async def batch_train_environments_tool(
    env_ids: list[str],
    algorithm: Literal["ppo", "alphazero"] = "ppo",
    policy: Literal["basic", "conv1d"] = "basic",
    num_iterations: int = 100,
    tensorboard_prefix: str | None = None,
    background: bool = False,
) -> dict[str, Any]:
    """Train multiple environments in sequence.

    Useful for training models across multiple topologies or subtopologies
    extracted from hardware.

    Args:
        env_ids: List of environment IDs to train
        algorithm: RL algorithm to use
        policy: Neural network policy architecture
        num_iterations: Number of iterations per environment
        tensorboard_prefix: Prefix for TensorBoard experiment names
        background: If True, start all training in background and return immediately.
            Use get_training_status_tool or wait_for_training_tool to monitor.

    Returns:
        If background=False: Dict with results for each environment.
        If background=True: Dict with session_ids for polling progress.

    Note:
        For batch training with many environments, set background=True to avoid
        timeouts. Training sessions run in parallel background threads.
    """
    return await batch_train_environments(
        env_ids=env_ids,
        algorithm=algorithm,
        policy=policy,
        num_iterations=num_iterations,
        tensorboard_prefix=tensorboard_prefix,
        background=background,
    )


@mcp.tool()
async def get_training_status_tool(session_id: str) -> dict[str, Any]:
    """Get the status and metrics of a training session.

    Args:
        session_id: Training session ID

    Returns:
        Dict with session status, progress, and metrics.
    """
    return await get_training_status(session_id)


@mcp.tool()
async def get_training_metrics_tool(session_id: str) -> dict[str, Any]:
    """Get detailed training metrics from TensorBoard logs.

    Returns the progression of difficulty, success rate, and reward
    throughout training. Use this after training completes to understand
    how well the model trained and what difficulty level it reached.

    Args:
        session_id: Training session ID

    Returns:
        Dict with:
        - metrics: Full progression data (difficulty, success, reward by step)
        - final_difficulty: The highest difficulty level reached
        - final_success: The final success rate achieved
        - final_success_percent: Success rate as percentage string

    Example:
        After training completes with session_id, call this to see:
        - Did it reach high difficulty levels? (good generalization)
        - Is success rate near 100%? (reliable synthesis)
    """
    return await get_training_metrics(session_id)


@mcp.tool()
async def wait_for_training_tool(
    session_id: str,
    timeout: int = 600,
) -> dict[str, Any]:
    """Wait for a background training session to complete.

    Blocks until training completes, fails, or times out. Use this after
    starting training with background=True.

    Args:
        session_id: Training session ID to wait for
        timeout: Maximum time to wait in seconds (default: 600 = 10 minutes)

    Returns:
        Dict with final training status. If completed, includes model_id
        for synthesis.

    Example workflow:
        1. start_training_tool(env_id, background=True) -> session_id
        2. wait_for_training_tool(session_id) -> model_id
        3. synthesize_*_tool(model_id, ...) -> circuit
    """
    return await wait_for_training(session_id, timeout)


@mcp.tool()
async def stop_training_tool(session_id: str) -> dict[str, Any]:
    """Stop a training session.

    Args:
        session_id: Training session ID to stop

    Returns:
        Dict with stop status.
    """
    return await stop_training(session_id)


@mcp.tool()
async def list_training_sessions_tool() -> dict[str, Any]:
    """List all training sessions.

    Returns:
        Dict with list of training sessions.
    """
    return await list_training_sessions()


@mcp.tool()
async def list_tensorboard_experiments_tool() -> dict[str, Any]:
    """List available TensorBoard experiments from past training runs.

    Returns a list of experiment names that can be used with
    get_tensorboard_metrics_tool to view historical training metrics.

    Returns:
        Dict with list of experiment names (newest first).
    """
    return await list_tensorboard_experiments()


@mcp.tool()
async def get_tensorboard_metrics_tool(
    experiment_name: str | None = None,
    tensorboard_path: str | None = None,
) -> dict[str, Any]:
    """Get training metrics from TensorBoard logs for historical runs.

    Use this to read metrics from past training runs that are no longer
    in the active session list. Use list_tensorboard_experiments_tool to
    see available experiments.

    Args:
        experiment_name: Name of the TensorBoard experiment
            (e.g., "linear_function_train_0001_abc123").
        tensorboard_path: Direct path to TensorBoard logs (alternative to name).

    Returns:
        Dict with metrics progression (difficulty, success, reward by step)
        and final values.

    Example:
        1. list_tensorboard_experiments_tool() -> see available experiments
        2. get_tensorboard_metrics_tool(experiment_name="linear_function_...") -> metrics
    """
    return await get_tensorboard_metrics(
        experiment_name=experiment_name,
        tensorboard_path=tensorboard_path,
    )


# ============================================================================
# TensorBoard Process Management
# ============================================================================


class _TensorBoardState:
    """Manages the TensorBoard process state."""

    def __init__(self) -> None:
        self.process: subprocess.Popen[bytes] | None = None
        self.port: int | None = None

    def cleanup(self) -> None:
        """Cleanup function to terminate TensorBoard on exit."""
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()


# Module-level state instance
_tb_state = _TensorBoardState()

# Register cleanup function to handle orphaned processes
atexit.register(_tb_state.cleanup)


@mcp.tool()
async def start_tensorboard_tool(port: int = 6006) -> dict[str, Any]:
    """Start TensorBoard to visualize training metrics.

    Launches TensorBoard as a background process using the configured
    QISKIT_GYM_TENSORBOARD_DIR as the log directory.

    Args:
        port: The port to run TensorBoard on (default: 6006)

    Returns:
        Dict with status and TensorBoard URL on success.

    Note:
        Use stop_tensorboard_tool to stop the TensorBoard process when done.
    """
    if _tb_state.process is not None and _tb_state.process.poll() is None:
        return {
            "status": "already_running",
            "message": f"TensorBoard is already running on port {_tb_state.port}. "
            "Use stop_tensorboard_tool to stop it first.",
            "url": f"http://localhost:{_tb_state.port}",
        }

    try:
        _tb_state.process = subprocess.Popen(  # nosec B603 B607
            ["tensorboard", "--logdir", QISKIT_GYM_TENSORBOARD_DIR, "--port", str(port)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Wait briefly and verify TensorBoard actually started
        time.sleep(1)
        if _tb_state.process.poll() is not None:
            # Process exited - read stderr for error message
            stderr_output = ""
            if _tb_state.process.stderr:
                stderr_output = _tb_state.process.stderr.read().decode()
            _tb_state.process = None
            _tb_state.port = None
            return {
                "status": "error",
                "error": f"TensorBoard failed to start: {stderr_output or 'unknown error'}",
            }

        _tb_state.port = port
        return {
            "status": "success",
            "message": "TensorBoard started successfully",
            "url": f"http://localhost:{port}",
            "logdir": QISKIT_GYM_TENSORBOARD_DIR,
        }
    except FileNotFoundError:
        return {
            "status": "error",
            "error": "TensorBoard is not installed. Install it with: pip install tensorboard",
        }
    except Exception as e:
        return {
            "status": "error",
            "error": f"Failed to start TensorBoard: {e}",
        }


@mcp.tool()
async def stop_tensorboard_tool() -> dict[str, Any]:
    """Stop the running TensorBoard process.

    Terminates the TensorBoard process that was started with start_tensorboard_tool.

    Returns:
        Dict with status message indicating whether TensorBoard was stopped.
    """
    if _tb_state.process is None:
        return {
            "status": "not_running",
            "message": "TensorBoard is not running.",
        }

    if _tb_state.process.poll() is not None:
        _tb_state.process = None
        _tb_state.port = None
        return {
            "status": "not_running",
            "message": "TensorBoard is not running (process already terminated).",
        }

    _tb_state.process.terminate()
    try:
        _tb_state.process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        _tb_state.process.kill()
        _tb_state.process.wait()

    _tb_state.process = None
    _tb_state.port = None
    return {
        "status": "success",
        "message": "TensorBoard stopped successfully.",
    }


@mcp.tool()
async def get_tensorboard_status_tool() -> dict[str, Any]:
    """Check the status of the TensorBoard process.

    Returns whether TensorBoard is running and on which port.

    Returns:
        Dict with running status, port, and URL if running.
    """
    if _tb_state.process is None:
        return {
            "status": "not_running",
            "running": False,
            "message": "TensorBoard is not running.",
        }

    if _tb_state.process.poll() is not None:
        # Process has terminated
        _tb_state.process = None
        _tb_state.port = None
        return {
            "status": "not_running",
            "running": False,
            "message": "TensorBoard is not running (process terminated).",
        }

    return {
        "status": "running",
        "running": True,
        "port": _tb_state.port,
        "url": f"http://localhost:{_tb_state.port}",
        "logdir": QISKIT_GYM_TENSORBOARD_DIR,
        "message": f"TensorBoard is running on port {_tb_state.port}.",
    }


# ============================================================================
# Synthesis Tools
# ============================================================================


@mcp.tool()
async def synthesize_permutation_tool(
    model_id: str,
    permutation: list[int],
    num_searches: int = 1000,
    deterministic: bool = False,
) -> dict[str, Any]:
    """Synthesize an optimal quantum circuit for a qubit permutation.

    Uses a trained PermutationGym model to find an optimal SWAP gate sequence
    that implements the desired qubit permutation on the coupling map.

    Args:
        model_id: ID of a loaded PermutationGym model
        permutation: Target permutation as list of qubit indices.
            Example: [2, 0, 1] means qubit 0 -> position 2, qubit 1 -> position 0
        num_searches: Number of search attempts. Higher = better results. Max: 10000
        deterministic: If True, use deterministic action selection

    Returns:
        Dict with circuit_qpy (base64-encoded), depth, gate counts, etc.
    """
    return await synthesize_permutation(
        model_id=model_id,
        permutation=permutation,
        num_searches=num_searches,
        deterministic=deterministic,
    )


@mcp.tool()
async def synthesize_linear_function_tool(
    model_id: str,
    linear_function: list[list[int]],
    num_searches: int = 1000,
    deterministic: bool = False,
) -> dict[str, Any]:
    """Synthesize an optimal quantum circuit for a linear Boolean function.

    Uses a trained LinearFunctionGym model to find an optimal CNOT circuit.

    Args:
        model_id: ID of a loaded LinearFunctionGym model
        linear_function: NxN binary matrix representing the linear function.
            Entry [i][j]=1 means output i depends on input j (XOR).
        num_searches: Number of search attempts. Max: 10000
        deterministic: If True, use deterministic action selection

    Returns:
        Dict with circuit_qpy (base64-encoded), metrics on success.
    """
    return await synthesize_linear_function(
        model_id=model_id,
        linear_function=linear_function,
        num_searches=num_searches,
        deterministic=deterministic,
    )


@mcp.tool()
async def synthesize_clifford_tool(
    model_id: str,
    clifford_tableau: list[list[int]] | dict[str, Any],
    num_searches: int = 1000,
    deterministic: bool = False,
) -> dict[str, Any]:
    """Synthesize an optimal quantum circuit for a Clifford operation.

    Uses a trained CliffordGym model to find an optimal Clifford implementation.

    Args:
        model_id: ID of a loaded CliffordGym model
        clifford_tableau: Clifford tableau in standard (2N+1 x 2N) format,
            or dict with "destab" and "stab" matrices.
        num_searches: Number of search attempts. Max: 10000
        deterministic: If True, use deterministic action selection

    Returns:
        Dict with circuit_qpy (base64-encoded), metrics on success.
    """
    return await synthesize_clifford(
        model_id=model_id,
        clifford_tableau=clifford_tableau,
        num_searches=num_searches,
        deterministic=deterministic,
    )


# ============================================================================
# Model Management Tools
# ============================================================================


@mcp.tool()
async def save_model_tool(
    session_id: str | None = None,
    model_id: str | None = None,
    model_name: str | None = None,
) -> dict[str, Any]:
    """Save a trained model to disk.

    Save by session_id (from just-completed training) or model_id (loaded model).

    Args:
        session_id: Training session ID (from start_training result)
        model_id: Model ID (alternative to session_id)
        model_name: Name to save as (defaults to auto-generated)

    Returns:
        Dict with save status and file paths.
    """
    return await save_model(
        session_id=session_id,
        model_id=model_id,
        model_name=model_name,
    )


@mcp.tool()
async def load_model_tool(model_name: str) -> dict[str, Any]:
    """Load a saved model from disk.

    Args:
        model_name: Name of the model to load

    Returns:
        Dict with model_id and model info.
    """
    return await load_model(model_name)


@mcp.tool()
async def list_saved_models_tool() -> dict[str, Any]:
    """List all models saved to disk.

    Returns:
        Dict with list of saved models.
    """
    return await list_saved_models()


@mcp.tool()
async def list_loaded_models_tool(filter_type: str | None = None) -> dict[str, Any]:
    """List all models currently loaded in memory.

    Args:
        filter_type: Optional filter by env_type or model name prefix

    Returns:
        Dict with list of loaded models.
    """
    return await list_loaded_models(filter_type)


@mcp.tool()
async def delete_model_tool(model_name: str, delete_files: bool = False) -> dict[str, Any]:
    """Delete a model.

    Args:
        model_name: Name of the model to delete
        delete_files: If True, also delete saved files from disk

    Returns:
        Dict with deletion status.
    """
    return await delete_model(model_name, delete_files)


@mcp.tool()
async def get_model_info_tool(
    model_id: str | None = None,
    model_name: str | None = None,
) -> dict[str, Any]:
    """Get detailed information about a model.

    Args:
        model_id: Model ID (for loaded models)
        model_name: Model name (can also check saved models on disk)

    Returns:
        Dict with model details.
    """
    return await get_model_info(model_id=model_id, model_name=model_name)


# ============================================================================
# Coupling Map Tools
# ============================================================================


@mcp.tool()
async def create_coupling_map_tool(
    edges: list[list[int]] | None = None,
    topology: str | None = None,
    num_qubits: int | None = None,
    rows: int | None = None,
    cols: int | None = None,
    bidirectional: bool = True,
) -> dict[str, Any]:
    """Create a custom coupling map.

    Args:
        edges: Custom edges as [[q1, q2], ...] (mutually exclusive with topology)
        topology: Topology type ("grid", "line") (mutually exclusive with edges)
        num_qubits: Number of qubits (for line topology)
        rows: Number of rows (for grid topology)
        cols: Number of columns (for grid topology)
        bidirectional: Whether edges are bidirectional (default: True)

    Returns:
        Dict with coupling map info and edges.
    """
    return await create_custom_coupling_map(
        edges=edges,
        topology=topology,
        num_qubits=num_qubits,
        rows=rows,
        cols=cols,
        bidirectional=bidirectional,
    )


@mcp.tool()
async def extract_subtopologies_tool(
    preset: str | None = None,
    edges: list[list[int]] | None = None,
    num_qubits: int = 4,
    max_subtopologies: int = 50,
) -> dict[str, Any]:
    """Extract connected subtopologies of N qubits from a hardware preset.

    Use this to find all unique connected subgraphs of a specified size from a
    larger coupling map. Essential for training RL models on subtopologies of
    real quantum hardware like IBM Nighthawk.

    Args:
        preset: Hardware preset name (e.g., "ibm_nighthawk", "ibm_heron_r1")
        edges: Custom coupling map edges (alternative to preset)
        num_qubits: Number of qubits for subtopologies (default: 4)
        max_subtopologies: Maximum number of subtopologies to return (default: 50)

    Returns:
        Dict with list of subtopologies, each containing edges and metadata.
    """
    return await extract_subtopologies(
        preset=preset,
        edges=edges,
        num_qubits=num_qubits,
        max_subtopologies=max_subtopologies,
    )


@mcp.tool()
async def list_subtopology_shapes_tool(
    preset: str,
    num_qubits: int,
) -> dict[str, Any]:
    """List available subtopology shapes for a given hardware and qubit count.

    Summarizes the types of subtopologies available (line, grid, star, etc.)
    without returning all edges.

    Args:
        preset: Hardware preset name
        num_qubits: Number of qubits for subtopologies

    Returns:
        Dict with shape counts and example subtopologies.
    """
    return await list_subtopology_shapes(preset, num_qubits)


@mcp.tool()
async def get_fake_backend_coupling_map_tool(backend_name: str) -> dict[str, Any]:
    """Get the exact coupling map from a fake IBM backend (no credentials needed).

    Use this to get exact IBM Quantum hardware topologies without needing
    IBM Quantum credentials. This is the recommended way to get accurate
    topologies for offline development.

    Args:
        backend_name: Backend name (e.g., "ibm_fez", "ibm_brisbane", "ibm_torino",
            "ibm_sherbrooke"). Use list_available_fake_backends to see all options.

    Returns:
        Dict with exact coupling map edges that can be used with create_*_env tools.

    Example:
        1. get_fake_backend_coupling_map_tool("ibm_fez") -> edges
        2. create_clifford_env_tool(num_qubits=..., coupling_map=edges)
    """
    return await get_fake_backend_coupling_map(backend_name)


@mcp.tool()
async def list_available_fake_backends_tool() -> dict[str, Any]:
    """List all available fake backends for offline topology access.

    Returns a list of IBM Quantum backends that have fake versions available
    in qiskit-ibm-runtime. These provide exact coupling maps without needing
    IBM Quantum credentials.

    Returns:
        Dict with list of backends, their qubit counts, and usage instructions.
    """
    return await list_available_fake_backends()


# ============================================================================
# Utility Tools
# ============================================================================


@mcp.tool()
async def generate_random_permutation_tool(num_qubits: int) -> dict[str, Any]:
    """Generate a random permutation for testing synthesis.

    Args:
        num_qubits: Number of qubits

    Returns:
        Dict with random permutation.
    """
    return await generate_random_permutation(num_qubits)


@mcp.tool()
async def generate_random_linear_function_tool(num_qubits: int) -> dict[str, Any]:
    """Generate a random invertible linear function for testing.

    Args:
        num_qubits: Number of qubits

    Returns:
        Dict with random linear function matrix.
    """
    return await generate_random_linear_function(num_qubits)


@mcp.tool()
async def generate_random_clifford_tool(num_qubits: int) -> dict[str, Any]:
    """Generate a random Clifford element for testing.

    Args:
        num_qubits: Number of qubits

    Returns:
        Dict with random Clifford tableau.
    """
    return await generate_random_clifford(num_qubits)


@mcp.tool()
async def convert_qpy_to_qasm3_tool(circuit_qpy: str) -> dict[str, Any]:
    """Convert a QPY circuit to human-readable QASM3 format.

    Use this tool to view the contents of a QPY circuit output from synthesis
    tools (like synthesize_permutation, synthesize_linear_function, synthesize_clifford)
    in a human-readable OpenQASM 3.0 format.

    Args:
        circuit_qpy: Base64-encoded QPY circuit string (from synthesis output)

    Returns:
        Dict with 'status' and 'qasm3' (the human-readable circuit string).
    """
    return qpy_to_qasm3(circuit_qpy)


@mcp.tool()
async def convert_qasm3_to_qpy_tool(circuit_qasm: str) -> dict[str, Any]:
    """Convert a QASM3 (or QASM2) circuit to base64-encoded QPY format.

    Use this tool to convert human-readable QASM circuits to QPY format,
    which preserves full circuit fidelity. The QPY output can be used with
    synthesis tools that may require circuit input.

    Args:
        circuit_qasm: OpenQASM 3.0 or 2.0 circuit string

    Returns:
        Dict with 'status' and 'circuit_qpy' (base64-encoded QPY string).
    """
    return qasm3_to_qpy(circuit_qasm)
