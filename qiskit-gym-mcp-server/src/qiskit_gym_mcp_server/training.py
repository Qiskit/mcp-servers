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

"""Training session management for qiskit-gym MCP server.

This module provides functions to:
- Start RL training sessions with configurable algorithms and policies
- Monitor training progress and metrics
- Stop running training sessions
- Batch train across multiple environments/topologies
- Background (async) training with polling
"""

import logging
import threading
import time
from pathlib import Path
from typing import Any, Literal

from qiskit_gym_mcp_server.constants import (
    QISKIT_GYM_MAX_ITERATIONS,
    QISKIT_GYM_TENSORBOARD_DIR,
)
from qiskit_gym_mcp_server.state import GymStateProvider
from qiskit_gym_mcp_server.utils import with_sync


logger = logging.getLogger(__name__)


def _get_rl_config(algorithm: str) -> Any:
    """Get the configuration class for an RL algorithm.

    Args:
        algorithm: Algorithm name ("ppo" or "alphazero")

    Returns:
        Config class instance
    """
    from qiskit_gym.rl import AlphaZeroConfig, PPOConfig

    if algorithm == "ppo":
        return PPOConfig()
    elif algorithm == "alphazero":
        return AlphaZeroConfig()
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}. Supported: ppo, alphazero")


def _get_policy_config(policy: str) -> Any:
    """Get the policy configuration class.

    Args:
        policy: Policy name ("basic" or "conv1d")

    Returns:
        Policy config class instance
    """
    from qiskit_gym.rl import BasicPolicyConfig, Conv1dPolicyConfig

    if policy == "basic":
        return BasicPolicyConfig()
    elif policy == "conv1d":
        return Conv1dPolicyConfig()
    else:
        raise ValueError(f"Unknown policy: {policy}. Supported: basic, conv1d")


def _ensure_tensorboard_dir(experiment_name: str | None) -> str | None:
    """Ensure TensorBoard directory exists and return path.

    Args:
        experiment_name: Name for the experiment

    Returns:
        Path to TensorBoard log directory, or None if not enabled
    """
    if experiment_name is None:
        return None

    tb_dir = Path(QISKIT_GYM_TENSORBOARD_DIR).expanduser()
    tb_dir.mkdir(parents=True, exist_ok=True)

    experiment_dir = tb_dir / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    return str(experiment_dir)


# ============================================================================
# Training Functions
# ============================================================================


def _suppress_training_logs():
    """Suppress verbose INFO training logs from twisterl (uses loguru).

    This only suppresses INFO-level terminal output. WARNING and ERROR
    logs are still shown. Training metrics are still logged to TensorBoard.
    """
    import sys

    try:
        from loguru import logger as loguru_logger

        # Remove default handler and add one with WARNING level minimum
        # This preserves WARNING and ERROR logs for debugging
        loguru_logger.remove()
        loguru_logger.add(sys.stderr, level="WARNING")
    except ImportError:
        pass  # loguru not available, logs will still appear
    except ValueError:
        pass  # Handler already removed or other configuration issue


def _run_training_in_background(
    session_id: str,
    env_id: str,
    env_type: str,
    env_coupling_map_edges: list[list[int]],
    env_num_qubits: int,
    gym_instance: Any,
    algorithm: str,
    policy: str,
    num_iterations: int,
    tb_path: str | None,
) -> None:
    """Run training in a background thread.

    This function is called from a thread and updates the state as it progresses.
    """
    try:
        from qiskit_gym.rl import RLSynthesis

        state = GymStateProvider()

        # Get configs
        rl_config = _get_rl_config(algorithm)
        policy_config = _get_policy_config(policy)

        # Create RLSynthesis instance
        rls = RLSynthesis(gym_instance, rl_config, policy_config)

        # Store RLS instance in session
        state.set_training_rls_instance(session_id, rls)
        state.set_training_status(session_id, "running")

        # Suppress verbose twisterl logs (metrics still go to TensorBoard)
        _suppress_training_logs()

        # Run training
        logger.info(f"Background training started for session {session_id}")
        if tb_path:
            rls.learn(num_iterations=num_iterations, tb_path=tb_path)
        else:
            rls.learn(num_iterations=num_iterations)

        # Training completed
        state.set_training_status(session_id, "completed")
        state.update_training_progress(session_id, num_iterations)

        # Register as a model
        model_id = state.register_model(
            model_name=f"{env_type}_{session_id}",
            env_type=env_type,
            coupling_map_edges=env_coupling_map_edges,
            num_qubits=env_num_qubits,
            rls_instance=rls,
            from_session_id=session_id,
        )

        # Store model_id in session for retrieval
        state.set_training_model_id(session_id, model_id)

        logger.info(f"Background training completed for session {session_id}, model_id={model_id}")

    except Exception as e:
        logger.error(f"Background training failed for session {session_id}: {e}")
        # state is already available from the try block
        GymStateProvider().set_training_status(session_id, "error", str(e))


@with_sync
async def start_training(
    env_id: str,
    algorithm: Literal["ppo", "alphazero"] = "ppo",
    policy: Literal["basic", "conv1d"] = "basic",
    num_iterations: int = 100,
    tensorboard_experiment: str | None = None,
    background: bool = False,
) -> dict[str, Any]:
    """Start training an RL agent on an environment.

    This initiates a training session that learns to synthesize optimal circuits.

    Args:
        env_id: Environment ID from create_*_env tools
        algorithm: RL algorithm to use:
            - "ppo": Proximal Policy Optimization (recommended for most cases)
            - "alphazero": AlphaZero-style MCTS (better for complex problems, slower)
        policy: Neural network policy architecture:
            - "basic": Simple feedforward network (faster, good for small problems)
            - "conv1d": 1D convolutional network (better for larger problems)
        num_iterations: Number of training iterations (default: 100)
        tensorboard_experiment: Name for TensorBoard experiment logging (optional)
        background: If True, run training in background and return immediately.
            Use get_training_status to poll for completion, or wait_for_training
            to block until done. (default: False)

    Returns:
        Dict with session_id. If background=False, also includes final status
        and training metrics. If background=True, returns immediately with
        session_id for polling.
    """
    try:
        # Validate iteration count
        if num_iterations > QISKIT_GYM_MAX_ITERATIONS:
            return {
                "status": "error",
                "message": f"num_iterations ({num_iterations}) exceeds maximum ({QISKIT_GYM_MAX_ITERATIONS})",
            }

        if num_iterations < 1:
            return {
                "status": "error",
                "message": "num_iterations must be at least 1",
            }

        # Get environment
        state = GymStateProvider()
        env = state.get_environment(env_id)
        if env is None:
            return {
                "status": "error",
                "message": f"Environment '{env_id}' not found. Use list_environments to see available.",
            }

        # Create training session first to get session_id
        session_id = state.create_training_session(
            env_id=env_id,
            algorithm=algorithm,
            policy=policy,
            total_iterations=num_iterations,
            tensorboard_path=None,  # Will be set below
        )

        # Auto-generate TensorBoard experiment name if not provided
        tb_experiment = tensorboard_experiment
        if tb_experiment is None:
            # Include env_type for easier identification: e.g., "linear_function_train_0001_abc123"
            tb_experiment = f"{env.env_type}_{session_id}"

        # Set up TensorBoard path
        tb_path = _ensure_tensorboard_dir(tb_experiment)

        # Update session with TensorBoard path
        session = state.get_training_session(session_id)
        if session:
            session.tensorboard_path = tb_path

        if background:
            # Run training in background thread
            thread = threading.Thread(
                target=_run_training_in_background,
                args=(
                    session_id,
                    env_id,
                    env.env_type,
                    env.coupling_map_edges,
                    env.num_qubits,
                    env.gym_instance,
                    algorithm,
                    policy,
                    num_iterations,
                    tb_path,
                ),
                daemon=True,
            )
            thread.start()

            return {
                "status": "started",
                "session_id": session_id,
                "env_id": env_id,
                "algorithm": algorithm,
                "policy": policy,
                "num_iterations": num_iterations,
                "background": True,
                "tensorboard_path": tb_path,
                "message": "Training started in background",
                "next_steps": [
                    f"Use get_training_status('{session_id}') to check progress",
                    f"Use wait_for_training('{session_id}') to wait for completion",
                ],
            }

        # Synchronous training (original behavior)
        from qiskit_gym.rl import RLSynthesis

        # Get configs
        rl_config = _get_rl_config(algorithm)
        policy_config = _get_policy_config(policy)

        # Create RLSynthesis instance
        rls = RLSynthesis(env.gym_instance, rl_config, policy_config)

        # Store RLS instance in session
        state.set_training_rls_instance(session_id, rls)
        state.set_training_status(session_id, "running")

        # Suppress verbose twisterl logs (metrics still go to TensorBoard)
        _suppress_training_logs()

        # Run training
        logger.info(f"Starting training session {session_id} with {num_iterations} iterations")
        try:
            if tb_path:
                rls.learn(num_iterations=num_iterations, tb_path=tb_path)
            else:
                rls.learn(num_iterations=num_iterations)

            # Training completed
            state.set_training_status(session_id, "completed")
            state.update_training_progress(session_id, num_iterations)

            # Register as a model
            model_id = state.register_model(
                model_name=f"{env.env_type}_{session_id}",
                env_type=env.env_type,
                coupling_map_edges=env.coupling_map_edges,
                num_qubits=env.num_qubits,
                rls_instance=rls,
                from_session_id=session_id,
            )

            # Store model_id in session for retrieval via wait_for_training
            state.set_training_model_id(session_id, model_id)

            return {
                "status": "success",
                "session_id": session_id,
                "model_id": model_id,
                "env_id": env_id,
                "algorithm": algorithm,
                "policy": policy,
                "iterations_completed": num_iterations,
                "tensorboard_path": tb_path,
                "message": "Training completed successfully",
                "next_steps": [
                    f"Use save_model with session_id='{session_id}' to persist the model",
                    f"Use synthesize_{env.env_type} with model_id='{model_id}' to generate circuits",
                ],
            }

        except Exception as train_error:
            state.set_training_status(session_id, "error", str(train_error))
            raise

    except ImportError as e:
        logger.error(f"qiskit-gym not installed: {e}")
        return {
            "status": "error",
            "message": "qiskit-gym package not installed. Install with: pip install qiskit-gym",
        }
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return {"status": "error", "message": str(e)}


@with_sync
async def get_training_status(session_id: str) -> dict[str, Any]:
    """Get the status and metrics of a training session.

    Args:
        session_id: Training session ID

    Returns:
        Dict with session status, progress, and metrics
    """
    state = GymStateProvider()
    session = state.get_training_session(session_id)

    if session is None:
        return {
            "status": "error",
            "message": f"Training session '{session_id}' not found",
        }

    result = {
        "status": "success",
        "session_id": session.session_id,
        "env_id": session.env_id,
        "algorithm": session.algorithm,
        "policy": session.policy,
        "training_status": session.status,
        "progress": session.progress,
        "total_iterations": session.total_iterations,
        "progress_percent": round(100 * session.progress / session.total_iterations, 1)
        if session.total_iterations > 0
        else 0,
        "metrics": session.metrics,
        "tensorboard_path": session.tensorboard_path,
        "error_message": session.error_message,
    }

    # Include model_id if training completed
    if session.status == "completed" and hasattr(session, "model_id") and session.model_id:
        result["model_id"] = session.model_id

    return result


@with_sync
async def wait_for_training(
    session_id: str,
    timeout: int = 600,
    poll_interval: float = 2.0,
) -> dict[str, Any]:
    """Wait for a background training session to complete.

    Blocks until the training session completes, fails, or times out.
    Use this after starting training with background=True.

    Args:
        session_id: Training session ID to wait for
        timeout: Maximum time to wait in seconds (default: 600 = 10 minutes)
        poll_interval: How often to check status in seconds (default: 2.0)

    Returns:
        Dict with final training status. If completed successfully, includes
        model_id for synthesis.
    """
    state = GymStateProvider()
    start_time = time.time()

    while True:
        session = state.get_training_session(session_id)

        if session is None:
            return {
                "status": "error",
                "message": f"Training session '{session_id}' not found",
            }

        # Check if training finished
        if session.status in ("completed", "error", "stopped"):
            result = {
                "status": "success" if session.status == "completed" else "error",
                "session_id": session_id,
                "training_status": session.status,
                "progress": session.progress,
                "total_iterations": session.total_iterations,
                "error_message": session.error_message,
            }

            if session.status == "completed":
                # Include model_id
                if hasattr(session, "model_id") and session.model_id:
                    result["model_id"] = session.model_id
                result["message"] = "Training completed successfully"
                result["next_steps"] = [
                    f"Use save_model('{session_id}') to persist the model",
                    "Use synthesize_* with model_id to generate circuits",
                ]
            elif session.status == "error":
                result["message"] = f"Training failed: {session.error_message}"
            else:
                result["message"] = "Training was stopped"

            return result

        # Check timeout
        elapsed = time.time() - start_time
        if elapsed >= timeout:
            return {
                "status": "timeout",
                "session_id": session_id,
                "training_status": session.status,
                "progress": session.progress,
                "total_iterations": session.total_iterations,
                "elapsed_seconds": round(elapsed, 1),
                "message": f"Timeout after {timeout} seconds. Training still running. "
                "Use get_training_status to check progress later.",
            }

        # Wait before next poll
        time.sleep(poll_interval)


@with_sync
async def stop_training(session_id: str) -> dict[str, Any]:
    """Stop a running training session.

    Note: This marks the session as stopped but cannot interrupt
    an in-progress training iteration.

    Args:
        session_id: Training session ID to stop

    Returns:
        Dict with stop status
    """
    state = GymStateProvider()
    session = state.get_training_session(session_id)

    if session is None:
        return {
            "status": "error",
            "message": f"Training session '{session_id}' not found",
        }

    if session.status == "completed":
        return {
            "status": "error",
            "message": f"Training session '{session_id}' already completed",
        }

    if session.status == "stopped":
        return {
            "status": "error",
            "message": f"Training session '{session_id}' already stopped",
        }

    state.set_training_status(session_id, "stopped")

    return {
        "status": "success",
        "session_id": session_id,
        "message": "Training session marked as stopped",
        "progress_at_stop": session.progress,
    }


@with_sync
async def list_training_sessions() -> dict[str, Any]:
    """List all training sessions.

    Returns:
        Dict with list of training sessions
    """
    state = GymStateProvider()
    sessions = state.list_training_sessions()

    return {
        "status": "success",
        "sessions": sessions,
        "total": len(sessions),
    }


# ============================================================================
# Batch Training
# ============================================================================


@with_sync
async def batch_train_environments(
    env_ids: list[str],
    algorithm: Literal["ppo", "alphazero"] = "ppo",
    policy: Literal["basic", "conv1d"] = "basic",
    num_iterations: int = 100,
    tensorboard_prefix: str | None = None,
    background: bool = False,
) -> dict[str, Any]:
    """Train multiple environments in sequence.

    Useful for training models across multiple topologies or subtopologies.

    Args:
        env_ids: List of environment IDs to train
        algorithm: RL algorithm to use
        policy: Neural network policy architecture
        num_iterations: Number of iterations per environment
        tensorboard_prefix: Prefix for TensorBoard experiment names
        background: If True, start all training in background and return immediately.
            Use get_training_status or wait_for_training to monitor each session.

    Returns:
        If background=False: Dict with results for each environment.
        If background=True: Dict with session IDs for each environment (returns immediately).
    """
    results: list[dict[str, Any]] = []
    successful = 0
    failed = 0
    session_ids: list[str] = []

    for i, env_id in enumerate(env_ids):
        logger.info(f"Batch training {i + 1}/{len(env_ids)}: {env_id}")

        # Generate TensorBoard experiment name
        tb_name = None
        if tensorboard_prefix:
            tb_name = f"{tensorboard_prefix}_{env_id}"

        # Train this environment
        result = await start_training(
            env_id=env_id,
            algorithm=algorithm,
            policy=policy,
            num_iterations=num_iterations,
            tensorboard_experiment=tb_name,
            background=background,
        )

        result["env_id"] = env_id
        results.append(result)

        if background:
            # Track session IDs for background mode
            if "session_id" in result:
                session_ids.append(result["session_id"])
        elif result["status"] == "success":
            successful += 1
        else:
            failed += 1

    if background:
        return {
            "status": "started",
            "message": f"Started {len(session_ids)} training sessions in background",
            "total_environments": len(env_ids),
            "session_ids": session_ids,
            "results": results,
            "next_steps": [
                "Use get_training_status_tool(session_id) to check progress of each session",
                "Use wait_for_training_tool(session_id) to wait for each session to complete",
                "Use list_training_sessions_tool() to see all sessions",
            ],
        }

    return {
        "status": "success" if failed == 0 else "partial",
        "total_environments": len(env_ids),
        "successful": successful,
        "failed": failed,
        "results": results,
    }


# ============================================================================
# Training Configuration
# ============================================================================


@with_sync
async def get_available_algorithms() -> dict[str, Any]:
    """Get information about available RL algorithms.

    Returns:
        Dict with algorithm descriptions and recommendations
    """
    return {
        "status": "success",
        "algorithms": {
            "ppo": {
                "name": "Proximal Policy Optimization",
                "description": "Stable, sample-efficient policy gradient method",
                "recommended_for": "Most use cases, especially small to medium problems",
                "training_speed": "Fast",
                "sample_efficiency": "Good",
            },
            "alphazero": {
                "name": "AlphaZero",
                "description": "MCTS-based algorithm with neural network guidance",
                "recommended_for": "Complex problems requiring strategic planning",
                "training_speed": "Slower",
                "sample_efficiency": "Better for complex problems",
            },
        },
        "default": "ppo",
    }


@with_sync
async def get_available_policies() -> dict[str, Any]:
    """Get information about available policy network architectures.

    Returns:
        Dict with policy descriptions and recommendations
    """
    return {
        "status": "success",
        "policies": {
            "basic": {
                "name": "Basic Policy",
                "description": "Simple feedforward neural network",
                "recommended_for": "Small problems (< 8 qubits), faster training",
                "architecture": "MLP with 2-3 hidden layers",
            },
            "conv1d": {
                "name": "Conv1D Policy",
                "description": "1D convolutional neural network",
                "recommended_for": "Larger problems, when spatial structure matters",
                "architecture": "Conv1D layers followed by dense layers",
            },
        },
        "default": "basic",
    }
