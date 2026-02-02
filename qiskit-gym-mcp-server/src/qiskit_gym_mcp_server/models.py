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

"""Model persistence for qiskit-gym MCP server.

This module provides functions to:
- Save trained models to disk
- Load models from disk
- List saved models
- Delete saved models
- Get model metadata
"""

import json
import logging
import os
from pathlib import Path
from typing import Any

from qiskit_gym_mcp_server.constants import QISKIT_GYM_MODEL_DIR
from qiskit_gym_mcp_server.state import GymStateProvider
from qiskit_gym_mcp_server.utils import with_sync


logger = logging.getLogger(__name__)


def _get_model_dir() -> Path:
    """Get the model storage directory, creating if needed."""
    model_dir = Path(QISKIT_GYM_MODEL_DIR).expanduser()
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir


def _get_model_paths(model_name: str) -> tuple[Path, Path]:
    """Get paths for model config and weights files.

    Args:
        model_name: Name of the model

    Returns:
        Tuple of (config_path, model_path)
    """
    model_dir = _get_model_dir()
    safe_name = model_name.replace("/", "_").replace("\\", "_")
    config_path = model_dir / f"{safe_name}_config.json"
    model_path = model_dir / f"{safe_name}_model.pt"
    return config_path, model_path


def _load_model_metadata(model_name: str) -> dict[str, Any] | None:
    """Load model metadata from config file.

    Args:
        model_name: Name of the model

    Returns:
        Model metadata dict or None if not found
    """
    config_path, _ = _get_model_paths(model_name)
    if not config_path.exists():
        return None

    try:
        with open(config_path) as f:
            data: dict[str, Any] = json.load(f)
            return data
    except Exception as e:
        logger.error(f"Failed to load model metadata: {e}")
        return None


# ============================================================================
# Model Persistence Functions
# ============================================================================


@with_sync
async def save_model(
    session_id: str | None = None,
    model_id: str | None = None,
    model_name: str | None = None,
) -> dict[str, Any]:
    """Save a trained model to disk.

    You can save either by session_id (from a just-completed training) or
    by model_id (from a loaded model).

    Args:
        session_id: Training session ID (from start_training result)
        model_id: Model ID (alternative to session_id)
        model_name: Name to save the model as (defaults to auto-generated)

    Returns:
        Dict with save status and file paths
    """
    try:
        state = GymStateProvider()

        # Get RLS instance and metadata
        rls_instance = None
        env_type = None
        coupling_map_edges = None
        num_qubits = None

        if session_id is not None:
            session = state.get_training_session(session_id)
            if session is None:
                return {
                    "status": "error",
                    "message": f"Training session '{session_id}' not found",
                }
            if session.rls_instance is None:
                return {
                    "status": "error",
                    "message": f"No trained model in session '{session_id}'",
                }
            rls_instance = session.rls_instance

            # Get env info
            env = state.get_environment(session.env_id)
            if env:
                env_type = env.env_type
                coupling_map_edges = env.coupling_map_edges
                num_qubits = env.num_qubits

            # Generate default name
            if model_name is None:
                model_name = f"model_{session.env_id}_{session_id}"

        elif model_id is not None:
            model = state.get_model(model_id)
            if model is None:
                return {
                    "status": "error",
                    "message": f"Model '{model_id}' not found",
                }
            rls_instance = model.rls_instance
            env_type = model.env_type
            coupling_map_edges = model.coupling_map_edges
            num_qubits = model.num_qubits

            if model_name is None:
                model_name = model.model_name

        else:
            return {
                "status": "error",
                "message": "Must provide either session_id or model_id",
            }

        # Get file paths
        config_path, model_path = _get_model_paths(model_name)

        # Save model using qiskit-gym's save method
        rls_instance.save(str(config_path), str(model_path))

        # Add additional metadata to config
        with open(config_path) as f:
            config_data = json.load(f)

        config_data["mcp_metadata"] = {
            "model_name": model_name,
            "env_type": env_type,
            "num_qubits": num_qubits,
            "coupling_map_edges": coupling_map_edges,
        }

        with open(config_path, "w") as f:
            json.dump(config_data, f, indent=2)

        logger.info(f"Saved model '{model_name}' to {model_path}")

        return {
            "status": "success",
            "model_name": model_name,
            "config_path": str(config_path),
            "model_path": str(model_path),
            "env_type": env_type,
            "num_qubits": num_qubits,
            "message": f"Model saved successfully. Load with load_model(model_name='{model_name}')",
        }

    except Exception as e:
        logger.error(f"Failed to save model: {e}")
        return {"status": "error", "message": str(e)}


@with_sync
async def load_model(model_name: str) -> dict[str, Any]:
    """Load a saved model from disk.

    Args:
        model_name: Name of the model to load

    Returns:
        Dict with model_id and model info
    """
    try:
        from qiskit_gym.rl import RLSynthesis

        config_path, model_path = _get_model_paths(model_name)

        # Check files exist
        if not config_path.exists():
            return {
                "status": "error",
                "message": f"Model config not found: {config_path}. Use list_saved_models to see available models.",
            }
        if not model_path.exists():
            return {
                "status": "error",
                "message": f"Model weights not found: {model_path}",
            }

        # Load metadata first
        with open(config_path) as f:
            config_data = json.load(f)

        mcp_metadata = config_data.get("mcp_metadata", {})
        env_type = mcp_metadata.get("env_type", "unknown")
        num_qubits = mcp_metadata.get("num_qubits", 0)
        coupling_map_edges = mcp_metadata.get("coupling_map_edges", [])

        # Load model using qiskit-gym
        rls = RLSynthesis.from_config_json(str(config_path), str(model_path))

        # Register in state
        state = GymStateProvider()
        model_id = state.register_model(
            model_name=model_name,
            env_type=env_type,
            coupling_map_edges=coupling_map_edges,
            num_qubits=num_qubits,
            rls_instance=rls,
            model_path=str(model_path),
        )

        logger.info(f"Loaded model '{model_name}' as {model_id}")

        return {
            "status": "success",
            "model_id": model_id,
            "model_name": model_name,
            "env_type": env_type,
            "num_qubits": num_qubits,
            "message": f"Model loaded. Use synthesize_{env_type} with model_id='{model_id}'",
        }

    except ImportError as e:
        logger.error(f"qiskit-gym not installed: {e}")
        return {
            "status": "error",
            "message": "qiskit-gym package not installed",
        }
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return {"status": "error", "message": str(e)}


@with_sync
async def list_saved_models() -> dict[str, Any]:
    """List all models saved to disk.

    Returns:
        Dict with list of saved models
    """
    try:
        model_dir = _get_model_dir()
        models = []

        # Find all config files
        for config_file in model_dir.glob("*_config.json"):
            model_name = config_file.stem.replace("_config", "")
            model_path = config_file.parent / f"{model_name}_model.pt"

            # Load metadata
            try:
                with open(config_file) as f:
                    config_data = json.load(f)
                mcp_metadata = config_data.get("mcp_metadata", {})
            except Exception:
                mcp_metadata = {}

            models.append(
                {
                    "model_name": model_name,
                    "env_type": mcp_metadata.get("env_type", "unknown"),
                    "num_qubits": mcp_metadata.get("num_qubits"),
                    "has_weights": model_path.exists(),
                    "config_path": str(config_file),
                }
            )

        return {
            "status": "success",
            "models": models,
            "total": len(models),
            "model_dir": str(model_dir),
        }

    except Exception as e:
        logger.error(f"Failed to list saved models: {e}")
        return {"status": "error", "message": str(e)}


@with_sync
async def list_loaded_models(filter_type: str | None = None) -> dict[str, Any]:
    """List all models currently loaded in memory.

    Args:
        filter_type: Optional filter by env_type or model name prefix

    Returns:
        Dict with list of loaded models
    """
    state = GymStateProvider()
    models = state.list_models(filter_type)

    return {
        "status": "success",
        "models": models,
        "total": len(models),
    }


@with_sync
async def delete_model(model_name: str, delete_files: bool = False) -> dict[str, Any]:
    """Delete a model.

    Args:
        model_name: Name of the model to delete
        delete_files: If True, also delete saved files from disk

    Returns:
        Dict with deletion status
    """
    try:
        state = GymStateProvider()

        # Find and remove from loaded models
        model = state.get_model_by_name(model_name)
        if model:
            state.delete_model(model.model_id)

        # Delete files if requested
        files_deleted = []
        if delete_files:
            config_path, model_path = _get_model_paths(model_name)
            if config_path.exists():
                os.remove(config_path)
                files_deleted.append(str(config_path))
            if model_path.exists():
                os.remove(model_path)
                files_deleted.append(str(model_path))

        return {
            "status": "success",
            "model_name": model_name,
            "removed_from_memory": model is not None,
            "files_deleted": files_deleted,
        }

    except Exception as e:
        logger.error(f"Failed to delete model: {e}")
        return {"status": "error", "message": str(e)}


@with_sync
async def get_model_info(
    model_id: str | None = None,
    model_name: str | None = None,
) -> dict[str, Any]:
    """Get detailed information about a model.

    Args:
        model_id: Model ID (for loaded models)
        model_name: Model name (can also check saved models)

    Returns:
        Dict with model details
    """
    try:
        state = GymStateProvider()

        # Try to find loaded model
        model = None
        if model_id:
            model = state.get_model(model_id)
        elif model_name:
            model = state.get_model_by_name(model_name)

        if model:
            return {
                "status": "success",
                "source": "loaded",
                "model_id": model.model_id,
                "model_name": model.model_name,
                "env_type": model.env_type,
                "num_qubits": model.num_qubits,
                "coupling_map_edges": len(model.coupling_map_edges),
                "saved_to_disk": model.model_path is not None,
                "model_path": model.model_path,
                "from_session_id": model.from_session_id,
            }

        # Check for saved model on disk
        if model_name:
            metadata = _load_model_metadata(model_name)
            if metadata:
                mcp_metadata = metadata.get("mcp_metadata", {})
                config_path, model_path = _get_model_paths(model_name)
                return {
                    "status": "success",
                    "source": "disk",
                    "model_name": model_name,
                    "env_type": mcp_metadata.get("env_type", "unknown"),
                    "num_qubits": mcp_metadata.get("num_qubits"),
                    "loaded": False,
                    "config_path": str(config_path),
                    "model_path": str(model_path) if model_path.exists() else None,
                    "note": "Use load_model to load this model into memory",
                }

        return {
            "status": "error",
            "message": "Model not found. Provide model_id for loaded models or model_name for saved models.",
        }

    except Exception as e:
        logger.error(f"Failed to get model info: {e}")
        return {"status": "error", "message": str(e)}
