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

"""Tests for synthesis.py."""

from unittest.mock import MagicMock

import pytest
from qiskit import QuantumCircuit

from qiskit_gym_mcp_server.state import GymStateProvider
from qiskit_gym_mcp_server.synthesis import (
    generate_random_clifford,
    generate_random_linear_function,
    generate_random_permutation,
    synthesize_clifford,
    synthesize_linear_function,
    synthesize_permutation,
)


class TestSynthesizePermutation:
    """Tests for permutation synthesis."""

    @pytest.fixture
    def setup_permutation_model(self):
        """Set up a mock permutation model."""
        state = GymStateProvider()

        # Create mock circuit for synth result
        mock_circuit = QuantumCircuit(5)
        mock_circuit.swap(0, 1)
        mock_circuit.swap(1, 2)

        mock_rls = MagicMock()
        mock_rls.synth.return_value = mock_circuit

        model_id = state.register_model(
            model_name="test_perm_model",
            env_type="permutation",
            coupling_map_edges=[[0, 1], [1, 2], [2, 3], [3, 4]],
            num_qubits=5,
            rls_instance=mock_rls,
        )
        return model_id, mock_rls

    @pytest.mark.asyncio
    async def test_synthesize_permutation_success(self, setup_permutation_model):
        """Test successful permutation synthesis."""
        model_id, mock_rls = setup_permutation_model

        result = await synthesize_permutation(
            model_id=model_id,
            permutation=[2, 0, 1, 4, 3],
            num_searches=100,
        )

        assert result["status"] == "success"
        assert "circuit_qpy" in result
        assert "metrics" in result
        mock_rls.synth.assert_called_once()

    @pytest.mark.asyncio
    async def test_synthesize_permutation_invalid_model(self):
        """Test error with invalid model."""
        result = await synthesize_permutation(
            model_id="nonexistent",
            permutation=[1, 0, 2],
            num_searches=100,
        )
        assert result["status"] == "error"
        assert "not found" in result["message"]

    @pytest.mark.asyncio
    async def test_synthesize_permutation_wrong_model_type(self):
        """Test error when using wrong model type."""
        state = GymStateProvider()

        model_id = state.register_model(
            model_name="clifford_model",
            env_type="clifford",  # Wrong type
            coupling_map_edges=[[0, 1]],
            num_qubits=3,
            rls_instance=MagicMock(),
        )

        result = await synthesize_permutation(
            model_id=model_id,
            permutation=[1, 0, 2],
            num_searches=100,
        )
        assert result["status"] == "error"
        assert "not permutation" in result["message"]

    @pytest.mark.asyncio
    async def test_synthesize_permutation_wrong_size(self, setup_permutation_model):
        """Test error when permutation size doesn't match model."""
        model_id, _ = setup_permutation_model

        result = await synthesize_permutation(
            model_id=model_id,
            permutation=[1, 0, 2],  # Only 3 elements, model has 5 qubits
            num_searches=100,
        )
        assert result["status"] == "error"
        assert "elements" in result["message"]

    @pytest.mark.asyncio
    async def test_synthesize_permutation_invalid_permutation(self, setup_permutation_model):
        """Test error when permutation is invalid."""
        model_id, _ = setup_permutation_model

        result = await synthesize_permutation(
            model_id=model_id,
            permutation=[0, 0, 1, 2, 3],  # Invalid: 0 appears twice
            num_searches=100,
        )
        assert result["status"] == "error"


class TestSynthesizeLinearFunction:
    """Tests for linear function synthesis."""

    @pytest.fixture
    def setup_linear_function_model(self):
        """Set up a mock linear function model."""
        state = GymStateProvider()

        mock_circuit = QuantumCircuit(3)
        mock_circuit.cx(0, 1)
        mock_circuit.cx(1, 2)

        mock_rls = MagicMock()
        mock_rls.synth.return_value = mock_circuit

        model_id = state.register_model(
            model_name="test_lf_model",
            env_type="linear_function",
            coupling_map_edges=[[0, 1], [1, 2]],
            num_qubits=3,
            rls_instance=mock_rls,
        )
        return model_id, mock_rls

    @pytest.mark.asyncio
    async def test_synthesize_linear_function_success(
        self, setup_linear_function_model, sample_linear_function
    ):
        """Test successful linear function synthesis."""
        model_id, mock_rls = setup_linear_function_model

        result = await synthesize_linear_function(
            model_id=model_id,
            linear_function=sample_linear_function,
            num_searches=100,
        )

        assert result["status"] == "success"
        assert "circuit_qpy" in result
        mock_rls.synth.assert_called_once()

    @pytest.mark.asyncio
    async def test_synthesize_linear_function_non_square(self, setup_linear_function_model):
        """Test error with non-square matrix."""
        model_id, _ = setup_linear_function_model

        result = await synthesize_linear_function(
            model_id=model_id,
            linear_function=[[1, 0], [0, 1], [1, 1]],  # 3x2 not square
            num_searches=100,
        )
        assert result["status"] == "error"
        assert "square" in result["message"]


class TestSynthesizeClifford:
    """Tests for Clifford synthesis."""

    @pytest.fixture
    def setup_clifford_model(self):
        """Set up a mock Clifford model."""
        state = GymStateProvider()

        mock_circuit = QuantumCircuit(2)
        mock_circuit.h(0)
        mock_circuit.cx(0, 1)

        mock_rls = MagicMock()
        mock_rls.synth.return_value = mock_circuit

        model_id = state.register_model(
            model_name="test_cliff_model",
            env_type="clifford",
            coupling_map_edges=[[0, 1]],
            num_qubits=2,
            rls_instance=mock_rls,
        )
        return model_id, mock_rls

    @pytest.mark.asyncio
    async def test_synthesize_clifford_invalid_model(self):
        """Test error with invalid model."""
        result = await synthesize_clifford(
            model_id="nonexistent",
            clifford_tableau=[[1, 0], [0, 1]],
            num_searches=100,
        )
        assert result["status"] == "error"


class TestRandomGeneration:
    """Tests for random target generation utilities."""

    @pytest.mark.asyncio
    async def test_generate_random_permutation(self):
        """Test random permutation generation."""
        result = await generate_random_permutation(num_qubits=5)
        assert result["status"] == "success"
        assert len(result["permutation"]) == 5
        assert set(result["permutation"]) == set(range(5))

    @pytest.mark.asyncio
    async def test_generate_random_linear_function(self):
        """Test random linear function generation."""
        result = await generate_random_linear_function(num_qubits=4)
        assert result["status"] == "success"
        assert result["shape"] == [4, 4]
        assert len(result["linear_function"]) == 4

    @pytest.mark.asyncio
    async def test_generate_random_clifford(self):
        """Test random Clifford generation."""
        result = await generate_random_clifford(num_qubits=2)
        assert result["status"] == "success"
        assert result["num_qubits"] == 2
        assert "clifford_tableau" in result
