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

"""State management for the qiskit-gym MCP server.

This module provides a singleton state manager that tracks:
- Active RL environments (PermutationGym, LinearFunctionGym, CliffordGym)
- Training sessions with their progress and metrics
- Loaded/trained models ready for synthesis
"""

import logging
import uuid
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Literal


logger = logging.getLogger(__name__)


@dataclass
class Environment:
    """Represents an active RL environment."""

    env_id: str
    env_type: Literal["permutation", "linear_function", "clifford"]
    config: dict[str, Any]
    gym_instance: Any  # Actual gym object (PermutationGym, etc.)
    coupling_map_edges: list[list[int]]
    num_qubits: int


@dataclass
class TrainingSession:
    """Represents an active or completed training session."""

    session_id: str
    env_id: str
    algorithm: Literal["ppo", "alphazero"]
    policy: Literal["basic", "conv1d"]
    status: Literal["pending", "running", "completed", "stopped", "error"]
    progress: int  # iterations completed
    total_iterations: int
    metrics: dict[str, Any] = field(default_factory=dict)
    tensorboard_path: str | None = None
    rls_instance: Any = None  # RLSynthesis object
    error_message: str | None = None
    model_id: str | None = None  # Set when training completes


@dataclass
class LoadedModel:
    """Represents a loaded or trained model ready for synthesis."""

    model_id: str
    model_name: str
    model_path: str | None  # None if not yet saved
    env_type: Literal["permutation", "linear_function", "clifford"]
    coupling_map_edges: list[list[int]]
    num_qubits: int
    rls_instance: Any  # RLSynthesis object
    from_session_id: str | None = None  # If created from training session


class GymStateProvider:
    """Singleton state manager for qiskit-gym MCP server.

    Tracks environments, training sessions, and loaded models across
    MCP tool calls within a session. Thread-safe for concurrent access
    from background training threads.

    Usage:
        state = GymStateProvider()
        env_id = state.register_environment(env, "permutation", config)
        env = state.get_environment(env_id)
    """

    _instance: "GymStateProvider | None" = None
    _lock: Lock = Lock()  # Class-level lock for singleton creation

    def __new__(cls) -> "GymStateProvider":
        """Create or return the singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialize()
        return cls._instance

    def _initialize(self) -> None:
        """Initialize instance state (called once on first creation)."""
        self._state_lock: Lock = Lock()  # Instance-level lock for state mutations
        self._environments: dict[str, Environment] = {}
        self._training_sessions: dict[str, TrainingSession] = {}
        self._loaded_models: dict[str, LoadedModel] = {}
        self._env_counter: int = 0
        self._session_counter: int = 0
        self._model_counter: int = 0
        logger.info("GymStateProvider initialized")

    # =========================================================================
    # Environment Management
    # =========================================================================

    def register_environment(
        self,
        gym_instance: Any,
        env_type: Literal["permutation", "linear_function", "clifford"],
        config: dict[str, Any],
        coupling_map_edges: list[list[int]],
        num_qubits: int,
    ) -> str:
        """Register a new environment and return its ID."""
        with self._state_lock:
            self._env_counter += 1
            env_id = f"env_{env_type[:4]}_{self._env_counter:04d}"

            environment = Environment(
                env_id=env_id,
                env_type=env_type,
                config=config,
                gym_instance=gym_instance,
                coupling_map_edges=coupling_map_edges,
                num_qubits=num_qubits,
            )
            self._environments[env_id] = environment
            logger.info(f"Registered environment: {env_id} ({env_type}, {num_qubits} qubits)")
            return env_id

    def get_environment(self, env_id: str) -> Environment | None:
        """Get an environment by ID."""
        return self._environments.get(env_id)

    def delete_environment(self, env_id: str) -> bool:
        """Delete an environment by ID. Returns True if deleted."""
        with self._state_lock:
            if env_id in self._environments:
                del self._environments[env_id]
                logger.info(f"Deleted environment: {env_id}")
                return True
            return False

    def list_environments(self) -> list[dict[str, Any]]:
        """List all active environments."""
        return [
            {
                "env_id": env.env_id,
                "env_type": env.env_type,
                "num_qubits": env.num_qubits,
                "coupling_map_edges": len(env.coupling_map_edges),
            }
            for env in self._environments.values()
        ]

    # =========================================================================
    # Training Session Management
    # =========================================================================

    def create_training_session(
        self,
        env_id: str,
        algorithm: Literal["ppo", "alphazero"],
        policy: Literal["basic", "conv1d"],
        total_iterations: int,
        tensorboard_path: str | None = None,
    ) -> str:
        """Create a new training session and return its ID."""
        with self._state_lock:
            self._session_counter += 1
            session_id = f"train_{self._session_counter:04d}_{uuid.uuid4().hex[:6]}"

            session = TrainingSession(
                session_id=session_id,
                env_id=env_id,
                algorithm=algorithm,
                policy=policy,
                status="pending",
                progress=0,
                total_iterations=total_iterations,
                tensorboard_path=tensorboard_path,
            )
            self._training_sessions[session_id] = session
            logger.info(f"Created training session: {session_id}")
            return session_id

    def get_training_session(self, session_id: str) -> TrainingSession | None:
        """Get a training session by ID."""
        return self._training_sessions.get(session_id)

    def update_training_progress(
        self,
        session_id: str,
        progress: int,
        metrics: dict[str, Any] | None = None,
    ) -> None:
        """Update training session progress."""
        with self._state_lock:
            session = self._training_sessions.get(session_id)
            if session:
                session.progress = progress
                if metrics:
                    session.metrics.update(metrics)

    def set_training_status(
        self,
        session_id: str,
        status: Literal["pending", "running", "completed", "stopped", "error"],
        error_message: str | None = None,
    ) -> None:
        """Update training session status."""
        with self._state_lock:
            session = self._training_sessions.get(session_id)
            if session:
                session.status = status
                if error_message:
                    session.error_message = error_message
                logger.info(f"Training session {session_id} status: {status}")

    def set_training_rls_instance(self, session_id: str, rls_instance: Any) -> None:
        """Set the RLSynthesis instance for a training session."""
        with self._state_lock:
            session = self._training_sessions.get(session_id)
            if session:
                session.rls_instance = rls_instance

    def set_training_model_id(self, session_id: str, model_id: str) -> None:
        """Set the model_id for a completed training session."""
        with self._state_lock:
            session = self._training_sessions.get(session_id)
            if session:
                session.model_id = model_id

    def list_training_sessions(self) -> list[dict[str, Any]]:
        """List all training sessions."""
        return [
            {
                "session_id": s.session_id,
                "env_id": s.env_id,
                "algorithm": s.algorithm,
                "status": s.status,
                "progress": s.progress,
                "total_iterations": s.total_iterations,
            }
            for s in self._training_sessions.values()
        ]

    # =========================================================================
    # Model Management
    # =========================================================================

    def register_model(
        self,
        model_name: str,
        env_type: Literal["permutation", "linear_function", "clifford"],
        coupling_map_edges: list[list[int]],
        num_qubits: int,
        rls_instance: Any,
        model_path: str | None = None,
        from_session_id: str | None = None,
    ) -> str:
        """Register a loaded or trained model and return its ID."""
        with self._state_lock:
            self._model_counter += 1
            model_id = f"model_{env_type[:4]}_{self._model_counter:04d}"

            model = LoadedModel(
                model_id=model_id,
                model_name=model_name,
                model_path=model_path,
                env_type=env_type,
                coupling_map_edges=coupling_map_edges,
                num_qubits=num_qubits,
                rls_instance=rls_instance,
                from_session_id=from_session_id,
            )
            self._loaded_models[model_id] = model
            logger.info(f"Registered model: {model_id} ({model_name})")
            return model_id

    def get_model(self, model_id: str) -> LoadedModel | None:
        """Get a model by ID."""
        return self._loaded_models.get(model_id)

    def get_model_by_name(self, model_name: str) -> LoadedModel | None:
        """Get a model by name."""
        for model in self._loaded_models.values():
            if model.model_name == model_name:
                return model
        return None

    def delete_model(self, model_id: str) -> bool:
        """Delete a model by ID. Returns True if deleted."""
        with self._state_lock:
            if model_id in self._loaded_models:
                del self._loaded_models[model_id]
                logger.info(f"Deleted model: {model_id}")
                return True
            return False

    def list_models(self, filter_type: str | None = None) -> list[dict[str, Any]]:
        """List all loaded models, optionally filtered by type or name prefix."""
        models = []
        for m in self._loaded_models.values():
            if filter_type and filter_type not in m.model_name and filter_type != m.env_type:
                continue
            models.append(
                {
                    "model_id": m.model_id,
                    "model_name": m.model_name,
                    "env_type": m.env_type,
                    "num_qubits": m.num_qubits,
                    "saved": m.model_path is not None,
                }
            )
        return models

    # =========================================================================
    # Reset (for testing)
    # =========================================================================

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton instance (primarily for testing)."""
        with cls._lock:
            if cls._instance is not None:
                with cls._instance._state_lock:
                    cls._instance._environments.clear()
                    cls._instance._training_sessions.clear()
                    cls._instance._loaded_models.clear()
                    cls._instance._env_counter = 0
                    cls._instance._session_counter = 0
                    cls._instance._model_counter = 0
                    logger.info("GymStateProvider reset")
