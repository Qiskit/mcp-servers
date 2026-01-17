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

"""Test fixtures and configuration for qiskit-gym MCP server tests."""

from unittest.mock import MagicMock

import pytest


@pytest.fixture(autouse=True)
def reset_state():
    """Reset the singleton state provider between tests."""
    from qiskit_gym_mcp_server.state import GymStateProvider

    GymStateProvider.reset()
    yield
    GymStateProvider.reset()


@pytest.fixture
def sample_coupling_map_linear():
    """Sample linear coupling map edges for testing."""
    return [[0, 1], [1, 0], [1, 2], [2, 1], [2, 3], [3, 2], [3, 4], [4, 3]]


@pytest.fixture
def sample_coupling_map_grid():
    """Sample 2x3 grid coupling map edges for testing."""
    # Grid layout:
    # 0 - 1 - 2
    # |   |   |
    # 3 - 4 - 5
    return [
        [0, 1],
        [1, 0],
        [1, 2],
        [2, 1],  # Top row
        [3, 4],
        [4, 3],
        [4, 5],
        [5, 4],  # Bottom row
        [0, 3],
        [3, 0],
        [1, 4],
        [4, 1],
        [2, 5],
        [5, 2],  # Vertical
    ]


@pytest.fixture
def sample_permutation():
    """Sample permutation for synthesis testing."""
    return [2, 0, 1, 4, 3]


@pytest.fixture
def sample_linear_function():
    """Sample invertible linear function matrix for testing."""
    return [
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 1],
    ]


@pytest.fixture
def mock_permutation_gym(mocker):
    """Mock PermutationGym class for testing."""
    mock_class = mocker.patch("qiskit_gym.envs.PermutationGym")
    mock_instance = MagicMock()
    mock_instance.action_space = MagicMock()
    mock_instance.action_space.n = 8
    mock_instance.observation_space = MagicMock()
    mock_instance.observation_space.shape = (5,)
    mock_class.from_coupling_map.return_value = mock_instance
    return mock_class


@pytest.fixture
def mock_linear_function_gym(mocker):
    """Mock LinearFunctionGym class for testing."""
    mock_class = mocker.patch("qiskit_gym.envs.LinearFunctionGym")
    mock_instance = MagicMock()
    mock_instance.action_space = MagicMock()
    mock_instance.action_space.n = 12
    mock_instance.observation_space = MagicMock()
    mock_instance.observation_space.shape = (9,)
    mock_class.from_coupling_map.return_value = mock_instance
    return mock_class


@pytest.fixture
def mock_clifford_gym(mocker):
    """Mock CliffordGym class for testing."""
    mock_class = mocker.patch("qiskit_gym.envs.CliffordGym")
    mock_instance = MagicMock()
    mock_instance.action_space = MagicMock()
    mock_instance.action_space.n = 16
    mock_instance.observation_space = MagicMock()
    mock_instance.observation_space.shape = (12,)
    mock_class.return_value = mock_instance
    return mock_class


@pytest.fixture
def mock_rls_synthesis(mocker):
    """Mock RLSynthesis class for training tests."""
    mock_class = mocker.patch("qiskit_gym.rl.RLSynthesis")
    mock_instance = MagicMock()
    mock_instance.learn = MagicMock()
    mock_instance.save = MagicMock()

    # Create a mock quantum circuit for synth output
    from qiskit import QuantumCircuit

    mock_circuit = QuantumCircuit(3)
    mock_circuit.swap(0, 1)
    mock_circuit.swap(1, 2)
    mock_instance.synth.return_value = mock_circuit

    mock_class.return_value = mock_instance
    mock_class.from_config_json.return_value = mock_instance
    return mock_class


@pytest.fixture
def mock_ppo_config(mocker):
    """Mock PPOConfig class."""
    mock_class = mocker.patch("qiskit_gym.rl.PPOConfig")
    mock_class.return_value = MagicMock()
    return mock_class


@pytest.fixture
def mock_alphazero_config(mocker):
    """Mock AlphaZeroConfig class."""
    mock_class = mocker.patch("qiskit_gym.rl.AlphaZeroConfig")
    mock_class.return_value = MagicMock()
    return mock_class


@pytest.fixture
def mock_basic_policy_config(mocker):
    """Mock BasicPolicyConfig class."""
    mock_class = mocker.patch("qiskit_gym.rl.BasicPolicyConfig")
    mock_class.return_value = MagicMock()
    return mock_class


@pytest.fixture
def mock_conv1d_policy_config(mocker):
    """Mock Conv1dPolicyConfig class."""
    mock_class = mocker.patch("qiskit_gym.rl.Conv1dPolicyConfig")
    mock_class.return_value = MagicMock()
    return mock_class


@pytest.fixture
def mock_quantum_circuit():
    """Create a simple mock quantum circuit."""
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(3)
    qc.swap(0, 1)
    qc.swap(1, 2)
    return qc


@pytest.fixture
def temp_model_dir(tmp_path):
    """Create a temporary model directory for testing."""
    import os

    model_dir = tmp_path / "models"
    model_dir.mkdir()

    # Temporarily override the model directory constant
    original_dir = os.environ.get("QISKIT_GYM_MODEL_DIR")
    os.environ["QISKIT_GYM_MODEL_DIR"] = str(model_dir)

    yield model_dir

    # Restore original
    if original_dir:
        os.environ["QISKIT_GYM_MODEL_DIR"] = original_dir
    else:
        os.environ.pop("QISKIT_GYM_MODEL_DIR", None)
