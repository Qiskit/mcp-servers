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

"""Tests for gym_core.py."""

import pytest

from qiskit_gym_mcp_server.gym_core import (
    create_clifford_environment,
    create_linear_function_environment,
    create_permutation_environment,
    delete_environment,
    get_environment_info,
    list_environments,
)


class TestPermutationEnvironment:
    """Tests for PermutationGym environment creation."""

    @pytest.mark.asyncio
    async def test_create_with_preset(self, mock_permutation_gym):
        """Test creating permutation environment with preset."""
        result = await create_permutation_environment(preset="grid_3x3")
        assert result["status"] == "success"
        assert result["env_type"] == "permutation"
        assert "env_id" in result
        mock_permutation_gym.from_coupling_map.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_with_custom_edges(self, mock_permutation_gym, sample_coupling_map_linear):
        """Test creating permutation environment with custom edges."""
        result = await create_permutation_environment(coupling_map=sample_coupling_map_linear)
        assert result["status"] == "success"
        assert result["env_type"] == "permutation"

    @pytest.mark.asyncio
    async def test_create_error_both_params(self, mock_permutation_gym, sample_coupling_map_linear):
        """Test error when both coupling_map and preset provided."""
        result = await create_permutation_environment(
            coupling_map=sample_coupling_map_linear,
            preset="grid_3x3",
        )
        assert result["status"] == "error"
        assert "not both" in result["message"]

    @pytest.mark.asyncio
    async def test_create_error_no_params(self, mock_permutation_gym):
        """Test error when neither coupling_map nor preset provided."""
        result = await create_permutation_environment()
        assert result["status"] == "error"


class TestLinearFunctionEnvironment:
    """Tests for LinearFunctionGym environment creation."""

    @pytest.mark.asyncio
    async def test_create_with_preset(self, mock_linear_function_gym):
        """Test creating linear function environment with preset."""
        result = await create_linear_function_environment(preset="grid_3x3")
        assert result["status"] == "success"
        assert result["env_type"] == "linear_function"

    @pytest.mark.asyncio
    async def test_create_with_basis_gates(self, mock_linear_function_gym):
        """Test creating with custom basis gates."""
        result = await create_linear_function_environment(
            preset="linear_5",
            basis_gates=["cx", "swap"],
        )
        assert result["status"] == "success"


class TestCliffordEnvironment:
    """Tests for CliffordGym environment creation."""

    @pytest.mark.asyncio
    async def test_create_basic(self, mock_clifford_gym):
        """Test creating Clifford environment."""
        result = await create_clifford_environment(num_qubits=4)
        assert result["status"] == "success"
        assert result["env_type"] == "clifford"
        assert result["num_qubits"] == 4

    @pytest.mark.asyncio
    async def test_create_with_preset(self, mock_clifford_gym):
        """Test creating with coupling map preset."""
        result = await create_clifford_environment(num_qubits=9, preset="grid_3x3")
        assert result["status"] == "success"

    @pytest.mark.asyncio
    async def test_create_with_gateset(self, mock_clifford_gym):
        """Test creating with custom gateset."""
        result = await create_clifford_environment(
            num_qubits=3,
            gateset=["H", "S", "CX"],
        )
        assert result["status"] == "success"

    @pytest.mark.asyncio
    async def test_create_qubit_mismatch(self, mock_clifford_gym):
        """Test error when num_qubits doesn't match preset."""
        result = await create_clifford_environment(num_qubits=5, preset="grid_3x3")
        assert result["status"] == "error"
        assert "qubits" in result["message"]


class TestEnvironmentManagement:
    """Tests for environment listing and deletion."""

    @pytest.mark.asyncio
    async def test_list_environments_empty(self):
        """Test listing environments when none exist."""
        result = await list_environments()
        assert result["status"] == "success"
        assert result["total"] == 0

    @pytest.mark.asyncio
    async def test_list_environments_with_envs(self, mock_permutation_gym):
        """Test listing environments after creating some."""
        await create_permutation_environment(preset="linear_5")
        await create_permutation_environment(preset="grid_3x3")

        result = await list_environments()
        assert result["status"] == "success"
        assert result["total"] == 2

    @pytest.mark.asyncio
    async def test_get_environment_info(self, mock_permutation_gym):
        """Test getting environment info."""
        create_result = await create_permutation_environment(preset="linear_5")
        env_id = create_result["env_id"]

        result = await get_environment_info(env_id)
        assert result["status"] == "success"
        assert result["env_id"] == env_id
        assert result["env_type"] == "permutation"

    @pytest.mark.asyncio
    async def test_get_environment_info_not_found(self):
        """Test error when environment not found."""
        result = await get_environment_info("nonexistent_env")
        assert result["status"] == "error"
        assert "not found" in result["message"]

    @pytest.mark.asyncio
    async def test_delete_environment(self, mock_permutation_gym):
        """Test deleting an environment."""
        create_result = await create_permutation_environment(preset="linear_5")
        env_id = create_result["env_id"]

        delete_result = await delete_environment(env_id)
        assert delete_result["status"] == "success"

        # Verify it's deleted
        list_result = await list_environments()
        assert list_result["total"] == 0

    @pytest.mark.asyncio
    async def test_delete_environment_not_found(self):
        """Test error when deleting nonexistent environment."""
        result = await delete_environment("nonexistent_env")
        assert result["status"] == "error"
