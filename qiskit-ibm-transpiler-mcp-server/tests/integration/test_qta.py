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
"""Integration tests for IBM Qiskit Transpiler MCP Server functions."""

from pathlib import Path

import pytest
from qiskit_ibm_transpiler_mcp_server.qta import (
    ai_clifford_synthesis,
    ai_linear_function_synthesis,
    ai_pauli_network_synthesis,
    ai_permutation_synthesis,
    ai_routing,
)

from tests.utils.helpers import validate_synthesis_result


# Get the path to the tests directory
TESTS_DIR = Path(__file__).parent.parent


class TestAIRouting:
    """Test AIRouting tool."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_ai_routing_success(self, backend_name):
        """
        Successful test AI routing tool with existing backend, quantum circuit and PassManager
        """
        with open(TESTS_DIR / "qasm" / "correct_qasm_1") as f:
            qasm_str = f.read()

        result = await ai_routing(
            circuit=qasm_str,
            backend_name=backend_name,
        )
        assert result["status"] == "success"

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_ai_routing_failure_backend_name(
        self,
    ):
        """
        Failed test AI routing tool with existing backend, quantum circuit and PassManager. Here we simulate wrong backend name.
        """
        with open(TESTS_DIR / "qasm" / "correct_qasm_1") as f:
            qasm_str = f.read()
        backend_name = "ibm_fake"

        result = await ai_routing(
            circuit=qasm_str,
            backend_name=backend_name,
        )
        assert result["status"] == "error"
        assert "Failed to find backend ibm_fake" in result["message"]

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_ai_routing_empty_backend(self):
        """
        Failed test AI routing tool with empty backend.
        """
        with open(TESTS_DIR / "qasm" / "correct_qasm_1") as f:
            qasm_str = f.read()
        result = await ai_routing(circuit=qasm_str, backend_name="")
        assert result["status"] == "error"
        assert result["message"] == "backend is required and cannot be empty"

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_ai_routing_failure_wrong_qasm_str(self, backend_name):
        """
        Failed test AI routing tool with existing backend, quantum circuit and PassManager. Here we simulate wrong input QASM string.
        """
        with open(TESTS_DIR / "qasm" / "wrong_qasm_1") as f:
            qasm_str = f.read()

        result = await ai_routing(
            circuit=qasm_str,
            backend_name=backend_name,
        )
        assert result["status"] == "error"


class TestAICliffordSynthesis:
    """Test AI Clifford synthesis tool"""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_ai_clifford_synthesis_success(self, backend_name):
        """
        Successful test AI Clifford synthesis tool with existing backend, quantum circuit and PassManager.
        """

        with open(TESTS_DIR / "qasm" / "correct_qasm_1") as f:
            qasm_str = f.read()
        result = await ai_clifford_synthesis(circuit=qasm_str, backend_name=backend_name)
        validate_synthesis_result(result)

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_ai_clifford_synthesis_failure_backend_name(
        self,
    ):
        """
        Failed test AI Clifford synthesis tool with existing backend, quantum circuit and PassManager. Wrong backend name
        """
        with open(TESTS_DIR / "qasm" / "correct_qasm_1") as f:
            qasm_str = f.read()
        backend_name = "ibm_fake"
        result = await ai_clifford_synthesis(circuit=qasm_str, backend_name=backend_name)
        assert result["status"] == "error"
        assert "Failed to find backend ibm_fake" in result["message"]

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_ai_clifford_synthesis_empty_backend(self):
        """
        Failed test AI Clifford synthesis tool with empty backend.
        """
        with open(TESTS_DIR / "qasm" / "correct_qasm_1") as f:
            qasm_str = f.read()
        result = await ai_clifford_synthesis(circuit=qasm_str, backend_name="")
        assert result["status"] == "error"
        assert result["message"] == "backend is required and cannot be empty"

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_ai_clifford_synthesis_failure_wrong_qasm_str(self, backend_name):
        """
        Failed test AI Clifford synthesis tool with existing backend, quantum circuit and PassManager. Wrong QASM str
        """
        with open(TESTS_DIR / "qasm" / "wrong_qasm_1") as f:
            qasm_str = f.read()

        result = await ai_clifford_synthesis(circuit=qasm_str, backend_name=backend_name)
        assert result["status"] == "error"


class TestAILinearFunctionSynthesis:
    """Test AI Linear Function synthesis tool"""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_ai_linear_function_synthesis_success(self, backend_name):
        """
        Successful test AI Linear Function synthesis tool with existing backend, quantum circuit and PassManager.
        """

        with open(TESTS_DIR / "qasm" / "correct_qasm_1") as f:
            qasm_str = f.read()
        result = await ai_linear_function_synthesis(circuit=qasm_str, backend_name=backend_name)
        validate_synthesis_result(result)

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_ai_linear_function_synthesis_failure_backend_name(
        self,
    ):
        """
        Failed test AI Linear Function synthesis tool with existing backend, quantum circuit and PassManager. Wrong backend name
        """
        with open(TESTS_DIR / "qasm" / "correct_qasm_1") as f:
            qasm_str = f.read()
        backend_name = "ibm_fake"
        result = await ai_linear_function_synthesis(circuit=qasm_str, backend_name=backend_name)
        assert result["status"] == "error"
        assert "Failed to find backend ibm_fake" in result["message"]

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_ai_linear_function_synthesis_empty_backend(self):
        """
        Failed test AI Linear Function synthesis tool with empty backend.
        """
        with open(TESTS_DIR / "qasm" / "correct_qasm_1") as f:
            qasm_str = f.read()
        result = await ai_linear_function_synthesis(circuit=qasm_str, backend_name="")
        assert result["status"] == "error"
        assert result["message"] == "backend is required and cannot be empty"

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_ai_linear_function_synthesis_failure_wrong_qasm_str(self, backend_name):
        """
        Failed test AI Linear Function synthesis tool with existing backend, quantum circuit and PassManager. Wrong QASM str
        """
        with open(TESTS_DIR / "qasm" / "wrong_qasm_1") as f:
            qasm_str = f.read()

        result = await ai_linear_function_synthesis(circuit=qasm_str, backend_name=backend_name)
        assert result["status"] == "error"


class TestAIPermutationSynthesis:
    """Test AI Permutation synthesis tool"""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_ai_permutation_synthesis_success(self, backend_name):
        """
        Successful test AI Permutation synthesis tool with existing backend, quantum circuit and PassManager.
        """

        with open(TESTS_DIR / "qasm" / "correct_qasm_1") as f:
            qasm_str = f.read()

        result = await ai_permutation_synthesis(circuit=qasm_str, backend_name=backend_name)
        validate_synthesis_result(result)

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_ai_permutation_synthesis_failure_backend_name(
        self,
    ):
        """
        Failed test AI Permutation synthesis tool with existing backend, quantum circuit and PassManager. Wrong backend name
        """
        with open(TESTS_DIR / "qasm" / "correct_qasm_1") as f:
            qasm_str = f.read()
        backend_name = "ibm_fake"
        result = await ai_permutation_synthesis(circuit=qasm_str, backend_name=backend_name)
        assert result["status"] == "error"
        assert "Failed to find backend ibm_fake" in result["message"]

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_ai_permutation_synthesis_empty_backend(self):
        """
        Failed test AI Permutation synthesis tool with empty backend.
        """
        with open(TESTS_DIR / "qasm" / "correct_qasm_1") as f:
            qasm_str = f.read()
        result = await ai_permutation_synthesis(circuit=qasm_str, backend_name="")
        assert result["status"] == "error"
        assert result["message"] == "backend is required and cannot be empty"

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_ai_permutation_synthesis_failure_wrong_qasm_str(self, backend_name):
        """
        Failed test AI Permutation synthesis tool with existing backend, quantum circuit and PassManager. Wrong QASM str
        """
        with open(TESTS_DIR / "qasm" / "wrong_qasm_1") as f:
            qasm_str = f.read()

        result = await ai_permutation_synthesis(circuit=qasm_str, backend_name=backend_name)
        assert result["status"] == "error"


class TestAIPauliNetworkSynthesis:
    """Test AI Pauli Network synthesis tool"""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_ai_pauli_network_synthesis_success(self, backend_name):
        """
        Successful test AI Pauli Network synthesis tool with existing backend, quantum circuit and PassManager.
        """

        with open(TESTS_DIR / "qasm" / "correct_qasm_1") as f:
            qasm_str = f.read()

        result = await ai_pauli_network_synthesis(circuit=qasm_str, backend_name=backend_name)
        validate_synthesis_result(result)

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_ai_pauli_network_synthesis_failure_backend_name(
        self,
    ):
        """
        Failed test AI Pauli Network synthesis tool with existing backend, quantum circuit and PassManager. Wrong backend name
        """
        with open(TESTS_DIR / "qasm" / "correct_qasm_1") as f:
            qasm_str = f.read()
        backend_name = "ibm_fake"
        result = await ai_pauli_network_synthesis(circuit=qasm_str, backend_name=backend_name)
        assert result["status"] == "error"
        assert "Failed to find backend ibm_fake" in result["message"]

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_ai_pauli_network_synthesis_empty_backend(self):
        """
        Failed test AI Pauli Network synthesis tool with empty backend.
        """
        with open(TESTS_DIR / "qasm" / "correct_qasm_1") as f:
            qasm_str = f.read()
        result = await ai_pauli_network_synthesis(circuit=qasm_str, backend_name="")
        assert result["status"] == "error"
        assert result["message"] == "backend is required and cannot be empty"

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_ai_pauli_network_synthesis_failure_wrong_qasm_str(self, backend_name):
        """
        Failed test AI Pauli Network synthesis tool with existing backend, quantum circuit and PassManager. Wrong QASM str
        """
        with open(TESTS_DIR / "qasm" / "wrong_qasm_1") as f:
            qasm_str = f.read()

        result = await ai_pauli_network_synthesis(circuit=qasm_str, backend_name=backend_name)
        assert result["status"] == "error"
