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

"""Tests for training.py."""

import pytest

from qiskit_gym_mcp_server.gym_core import create_permutation_environment
from qiskit_gym_mcp_server.training import (
    batch_train_environments,
    get_available_algorithms,
    get_available_policies,
    get_training_status,
    list_training_sessions,
    start_training,
    stop_training,
    wait_for_training,
)


class TestStartTraining:
    """Tests for training session creation and execution."""

    @pytest.mark.asyncio
    async def test_start_training_success(
        self,
        mock_permutation_gym,
        mock_rls_synthesis,
        mock_ppo_config,
        mock_basic_policy_config,
    ):
        """Test starting training successfully."""
        # Create environment first
        env_result = await create_permutation_environment(preset="linear_5")
        env_id = env_result["env_id"]

        # Start training
        result = await start_training(
            env_id=env_id,
            algorithm="ppo",
            policy="basic",
            num_iterations=10,
        )

        assert result["status"] == "success"
        assert "session_id" in result
        assert "model_id" in result
        assert result["iterations_completed"] == 10
        mock_rls_synthesis.return_value.learn.assert_called_once()

    @pytest.mark.asyncio
    async def test_start_training_invalid_env(
        self,
        mock_rls_synthesis,
        mock_ppo_config,
        mock_basic_policy_config,
    ):
        """Test error when environment not found."""
        result = await start_training(
            env_id="nonexistent_env",
            algorithm="ppo",
            num_iterations=10,
        )
        assert result["status"] == "error"
        assert "not found" in result["message"]

    @pytest.mark.asyncio
    async def test_start_training_exceeds_max_iterations(
        self,
        mock_permutation_gym,
        mock_rls_synthesis,
    ):
        """Test error when iterations exceed maximum."""
        env_result = await create_permutation_environment(preset="linear_5")
        env_id = env_result["env_id"]

        result = await start_training(
            env_id=env_id,
            num_iterations=999999999,  # Way over limit
        )
        assert result["status"] == "error"
        assert "exceeds maximum" in result["message"]


class TestTrainingStatus:
    """Tests for training status retrieval."""

    @pytest.mark.asyncio
    async def test_get_training_status(
        self,
        mock_permutation_gym,
        mock_rls_synthesis,
        mock_ppo_config,
        mock_basic_policy_config,
    ):
        """Test getting training status."""
        env_result = await create_permutation_environment(preset="linear_5")
        env_id = env_result["env_id"]

        train_result = await start_training(
            env_id=env_id,
            num_iterations=10,
        )
        session_id = train_result["session_id"]

        status_result = await get_training_status(session_id)
        assert status_result["status"] == "success"
        assert status_result["training_status"] == "completed"
        assert status_result["progress"] == 10

    @pytest.mark.asyncio
    async def test_get_training_status_not_found(self):
        """Test error when session not found."""
        result = await get_training_status("nonexistent_session")
        assert result["status"] == "error"
        assert "not found" in result["message"]


class TestStopTraining:
    """Tests for stopping training sessions."""

    @pytest.mark.asyncio
    async def test_stop_training_not_found(self):
        """Test error when stopping nonexistent session."""
        result = await stop_training("nonexistent_session")
        assert result["status"] == "error"
        assert "not found" in result["message"]


class TestListTrainingSessions:
    """Tests for listing training sessions."""

    @pytest.mark.asyncio
    async def test_list_empty(self):
        """Test listing when no sessions exist."""
        result = await list_training_sessions()
        assert result["status"] == "success"
        assert result["total"] == 0

    @pytest.mark.asyncio
    async def test_list_with_sessions(
        self,
        mock_permutation_gym,
        mock_rls_synthesis,
        mock_ppo_config,
        mock_basic_policy_config,
    ):
        """Test listing after creating sessions."""
        env_result = await create_permutation_environment(preset="linear_5")
        env_id = env_result["env_id"]

        await start_training(env_id=env_id, num_iterations=5)

        result = await list_training_sessions()
        assert result["status"] == "success"
        assert result["total"] == 1


class TestBatchTraining:
    """Tests for batch training."""

    @pytest.mark.asyncio
    async def test_batch_train_multiple_envs(
        self,
        mock_permutation_gym,
        mock_rls_synthesis,
        mock_ppo_config,
        mock_basic_policy_config,
    ):
        """Test batch training multiple environments."""
        # Create multiple environments
        env1_result = await create_permutation_environment(preset="linear_5")
        env2_result = await create_permutation_environment(preset="grid_3x3")

        env_ids = [env1_result["env_id"], env2_result["env_id"]]

        result = await batch_train_environments(
            env_ids=env_ids,
            num_iterations=5,
        )

        assert result["status"] == "success"
        assert result["total_environments"] == 2
        assert result["successful"] == 2
        assert result["failed"] == 0

    @pytest.mark.asyncio
    async def test_batch_train_background(
        self,
        mock_permutation_gym,
        mock_rls_synthesis,
        mock_ppo_config,
        mock_basic_policy_config,
    ):
        """Test batch training with background=True returns immediately."""
        # Create multiple environments
        env1_result = await create_permutation_environment(preset="linear_5")
        env2_result = await create_permutation_environment(preset="grid_3x3")

        env_ids = [env1_result["env_id"], env2_result["env_id"]]

        result = await batch_train_environments(
            env_ids=env_ids,
            num_iterations=5,
            background=True,
        )

        # Should return "started" status with session_ids
        assert result["status"] == "started"
        assert result["total_environments"] == 2
        assert "session_ids" in result
        assert len(result["session_ids"]) == 2
        assert "next_steps" in result
        assert "results" in result


class TestAlgorithmsAndPolicies:
    """Tests for algorithm and policy information."""

    @pytest.mark.asyncio
    async def test_get_available_algorithms(self):
        """Test getting available algorithms."""
        result = await get_available_algorithms()
        assert result["status"] == "success"
        assert "ppo" in result["algorithms"]
        assert "alphazero" in result["algorithms"]

    @pytest.mark.asyncio
    async def test_get_available_policies(self):
        """Test getting available policies."""
        result = await get_available_policies()
        assert result["status"] == "success"
        assert "basic" in result["policies"]
        assert "conv1d" in result["policies"]


class TestBackgroundTraining:
    """Tests for background/async training functionality."""

    @pytest.mark.asyncio
    async def test_start_training_background(
        self,
        mock_permutation_gym,
        mock_rls_synthesis,
        mock_ppo_config,
        mock_basic_policy_config,
    ):
        """Test starting training in background mode returns immediately."""
        # Create environment first
        env_result = await create_permutation_environment(preset="linear_5")
        env_id = env_result["env_id"]

        # Start training in background
        result = await start_training(
            env_id=env_id,
            algorithm="ppo",
            policy="basic",
            num_iterations=10,
            background=True,
        )

        # Should return immediately with "started" status
        assert result["status"] == "started"
        assert "session_id" in result
        assert result["background"] is True
        assert result["env_id"] == env_id
        assert result["algorithm"] == "ppo"
        assert result["policy"] == "basic"
        assert "next_steps" in result

    @pytest.mark.asyncio
    async def test_wait_for_training_not_found(self):
        """Test error when waiting for nonexistent session."""
        result = await wait_for_training("nonexistent_session", timeout=1)
        assert result["status"] == "error"
        assert "not found" in result["message"]

    @pytest.mark.asyncio
    async def test_wait_for_training_completed(
        self,
        mock_permutation_gym,
        mock_rls_synthesis,
        mock_ppo_config,
        mock_basic_policy_config,
    ):
        """Test waiting for a completed training session."""
        # Create environment and start synchronous training (which completes immediately)
        env_result = await create_permutation_environment(preset="linear_5")
        env_id = env_result["env_id"]

        # Start training synchronously (completes immediately with mocks)
        train_result = await start_training(
            env_id=env_id,
            num_iterations=5,
            background=False,
        )
        session_id = train_result["session_id"]

        # Wait should return immediately since training is already completed
        result = await wait_for_training(session_id, timeout=1)
        assert result["status"] == "success"
        assert result["training_status"] == "completed"
        assert "model_id" in result

    @pytest.mark.asyncio
    async def test_wait_for_training_timeout(
        self,
        mock_permutation_gym,
        mock_rls_synthesis,
        mock_ppo_config,
        mock_basic_policy_config,
        mocker,
    ):
        """Test timeout when waiting for training that doesn't complete."""
        # Create environment
        env_result = await create_permutation_environment(preset="linear_5")
        env_id = env_result["env_id"]

        # Make the background training never complete by making learn() block
        def slow_learn(*args, **kwargs):
            import time

            time.sleep(10)  # This will be interrupted by timeout

        mock_rls_synthesis.return_value.learn.side_effect = slow_learn

        # Start training in background
        result = await start_training(
            env_id=env_id,
            num_iterations=10,
            background=True,
        )
        session_id = result["session_id"]

        # Wait with very short timeout - should timeout
        wait_result = await wait_for_training(
            session_id, timeout=0.1, poll_interval=0.05
        )

        assert wait_result["status"] == "timeout"
        assert wait_result["session_id"] == session_id
        assert "elapsed_seconds" in wait_result
