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

"""Tests for state.py singleton state manager."""

from unittest.mock import MagicMock

from qiskit_gym_mcp_server.state import GymStateProvider


class TestGymStateProviderSingleton:
    """Tests for singleton behavior."""

    def test_singleton_same_instance(self):
        """Test that multiple calls return same instance."""
        state1 = GymStateProvider()
        state2 = GymStateProvider()
        assert state1 is state2

    def test_reset_clears_state(self):
        """Test that reset clears all state."""
        state = GymStateProvider()

        # Add some data
        state.register_environment(
            gym_instance=MagicMock(),
            env_type="permutation",
            config={},
            coupling_map_edges=[[0, 1]],
            num_qubits=2,
        )

        assert len(state.list_environments()) == 1

        # Reset
        GymStateProvider.reset()

        assert len(state.list_environments()) == 0


class TestEnvironmentManagement:
    """Tests for environment registration and retrieval."""

    def test_register_environment(self):
        """Test registering an environment."""
        state = GymStateProvider()
        mock_gym = MagicMock()

        env_id = state.register_environment(
            gym_instance=mock_gym,
            env_type="permutation",
            config={"preset": "linear_5"},
            coupling_map_edges=[[0, 1], [1, 2]],
            num_qubits=3,
        )

        assert env_id.startswith("env_perm_")

    def test_get_environment(self):
        """Test retrieving an environment."""
        state = GymStateProvider()
        mock_gym = MagicMock()

        env_id = state.register_environment(
            gym_instance=mock_gym,
            env_type="clifford",
            config={},
            coupling_map_edges=[[0, 1]],
            num_qubits=2,
        )

        env = state.get_environment(env_id)
        assert env is not None
        assert env.env_type == "clifford"
        assert env.gym_instance is mock_gym

    def test_get_nonexistent_environment(self):
        """Test getting nonexistent environment returns None."""
        state = GymStateProvider()
        assert state.get_environment("nonexistent") is None

    def test_delete_environment(self):
        """Test deleting an environment."""
        state = GymStateProvider()
        mock_gym = MagicMock()

        env_id = state.register_environment(
            gym_instance=mock_gym,
            env_type="permutation",
            config={},
            coupling_map_edges=[],
            num_qubits=2,
        )

        assert state.delete_environment(env_id) is True
        assert state.get_environment(env_id) is None

    def test_delete_nonexistent_environment(self):
        """Test deleting nonexistent environment returns False."""
        state = GymStateProvider()
        assert state.delete_environment("nonexistent") is False

    def test_list_environments(self):
        """Test listing environments."""
        state = GymStateProvider()

        state.register_environment(
            gym_instance=MagicMock(),
            env_type="permutation",
            config={},
            coupling_map_edges=[[0, 1]],
            num_qubits=2,
        )
        state.register_environment(
            gym_instance=MagicMock(),
            env_type="clifford",
            config={},
            coupling_map_edges=[[0, 1], [1, 2]],
            num_qubits=3,
        )

        envs = state.list_environments()
        assert len(envs) == 2
        assert any(e["env_type"] == "permutation" for e in envs)
        assert any(e["env_type"] == "clifford" for e in envs)


class TestTrainingSessionManagement:
    """Tests for training session management."""

    def test_create_training_session(self):
        """Test creating a training session."""
        state = GymStateProvider()

        session_id = state.create_training_session(
            env_id="env_test_001",
            algorithm="ppo",
            policy="basic",
            total_iterations=100,
        )

        assert session_id.startswith("train_")

    def test_get_training_session(self):
        """Test retrieving a training session."""
        state = GymStateProvider()

        session_id = state.create_training_session(
            env_id="env_test_001",
            algorithm="alphazero",
            policy="conv1d",
            total_iterations=50,
        )

        session = state.get_training_session(session_id)
        assert session is not None
        assert session.algorithm == "alphazero"
        assert session.policy == "conv1d"
        assert session.status == "pending"

    def test_update_training_progress(self):
        """Test updating training progress."""
        state = GymStateProvider()

        session_id = state.create_training_session(
            env_id="env_test_001",
            algorithm="ppo",
            policy="basic",
            total_iterations=100,
        )

        state.update_training_progress(session_id, 50, {"loss": 0.5})

        session = state.get_training_session(session_id)
        assert session.progress == 50
        assert session.metrics["loss"] == 0.5

    def test_set_training_status(self):
        """Test setting training status."""
        state = GymStateProvider()

        session_id = state.create_training_session(
            env_id="env_test_001",
            algorithm="ppo",
            policy="basic",
            total_iterations=100,
        )

        state.set_training_status(session_id, "running")
        assert state.get_training_session(session_id).status == "running"

        state.set_training_status(session_id, "error", "Something went wrong")
        session = state.get_training_session(session_id)
        assert session.status == "error"
        assert session.error_message == "Something went wrong"


class TestModelManagement:
    """Tests for model management."""

    def test_register_model(self):
        """Test registering a model."""
        state = GymStateProvider()
        mock_rls = MagicMock()

        model_id = state.register_model(
            model_name="test_model",
            env_type="permutation",
            coupling_map_edges=[[0, 1]],
            num_qubits=2,
            rls_instance=mock_rls,
        )

        assert model_id.startswith("model_perm_")

    def test_get_model(self):
        """Test retrieving a model by ID."""
        state = GymStateProvider()
        mock_rls = MagicMock()

        model_id = state.register_model(
            model_name="test_model",
            env_type="clifford",
            coupling_map_edges=[[0, 1]],
            num_qubits=2,
            rls_instance=mock_rls,
        )

        model = state.get_model(model_id)
        assert model is not None
        assert model.model_name == "test_model"
        assert model.env_type == "clifford"

    def test_get_model_by_name(self):
        """Test retrieving a model by name."""
        state = GymStateProvider()
        mock_rls = MagicMock()

        state.register_model(
            model_name="unique_model_name",
            env_type="linear_function",
            coupling_map_edges=[[0, 1]],
            num_qubits=2,
            rls_instance=mock_rls,
        )

        model = state.get_model_by_name("unique_model_name")
        assert model is not None
        assert model.env_type == "linear_function"

    def test_list_models_with_filter(self):
        """Test listing models with filter."""
        state = GymStateProvider()

        state.register_model(
            model_name="perm_model_1",
            env_type="permutation",
            coupling_map_edges=[[0, 1]],
            num_qubits=2,
            rls_instance=MagicMock(),
        )
        state.register_model(
            model_name="cliff_model_1",
            env_type="clifford",
            coupling_map_edges=[[0, 1]],
            num_qubits=2,
            rls_instance=MagicMock(),
        )

        # Filter by type
        perm_models = state.list_models(filter_type="permutation")
        assert len(perm_models) == 1
        assert perm_models[0]["env_type"] == "permutation"

        # Filter by name
        cliff_models = state.list_models(filter_type="cliff")
        assert len(cliff_models) == 1
