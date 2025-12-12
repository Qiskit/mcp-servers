import pytest
from unittest.mock import MagicMock
from qiskit_ibm_transpiler_mcp_server.server import mcp

from qiskit_ibm_transpiler_mcp_server.utils import setup_ibm_quantum_account
from qiskit_ibm_transpiler_mcp_server.qiskit_runtime_service_provider import (
    QiskitRuntimeServiceProvider,
)


class TestMCPServerIntegration:
    """Test MCP server integration."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_server_initialization(self, mock_env_vars):
        """Test that server initializes correctly."""
        # Server should initialize without errors
        assert mcp is not None
        assert mcp.name == "Qiskit IBM Transpiler"

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_service_initialization_flow(
        self, mocker, mock_env_vars, mock_runtime_service
    ):
        """Test service initialization flow."""
        qiskit_runtime_sp = QiskitRuntimeServiceProvider()
        qiskit_runtime_service_mock = mocker.patch(
            "qiskit_ibm_transpiler_mcp_server.qiskit_runtime_service_provider.QiskitRuntimeService",
            return_value=mock_runtime_service,
        )

        service = qiskit_runtime_sp.get()

        assert service == mock_runtime_service
        qiskit_runtime_service_mock.assert_called_once()


class TestErrorHandlingIntegration:
    """Test error handling in integration scenarios."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_authentication_failure_recovery(self, mocker, mock_env_vars):
        """Test recovery from authentication failures."""

        # First call fails with authentication error
        qiskit_runtime_sp = QiskitRuntimeServiceProvider()
        qiskit_runtime_service_get_mock = mocker.patch.object(
            qiskit_runtime_sp,
            "get",
            side_effect=[
                ValueError("Invalid token"),
                MagicMock(),
            ],  # second call succeeds
        )

        # First attempt should fail
        result1 = await setup_ibm_quantum_account("invalid_token")
        assert result1["status"] == "error"

        # Reset the mock for second attempt
        qiskit_runtime_service_get_mock.side_effect = None
        qiskit_runtime_service_get_mock.return_value = MagicMock()

        # Second attempt should succeed
        result2 = await setup_ibm_quantum_account("valid_token")
        assert result2["status"] == "success"

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_network_connectivity_issues(
        self, mocker, mock_env_vars, mock_runtime_service
    ):
        """Test handling of network connectivity issues."""
        from qiskit_ibm_transpiler_mcp_server.utils import get_backend_service

        qiskit_runtime_sp = QiskitRuntimeServiceProvider()
        qiskit_runtime_service_get_mock = mocker.patch.object(
            qiskit_runtime_sp, "get", side_effect=[Exception("Network timeout")]
        )

        result = await get_backend_service("ibm_brisbane")

        assert result["status"] == "error"
        assert "Failed to find backend" in result["message"]
        qiskit_runtime_service_get_mock.assert_called_once()


class TestEndToEndScenarios:
    """Test end-to-end scenarios."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_complete_synthesis_pass(self):
        """Test complete backend exploration scenario."""
        from qiskit_ibm_transpiler_mcp_server.qta import (
            ai_routing,
            ai_clifford_synthesis,
        )
        from qiskit_ibm_transpiler_mcp_server.utils import setup_ibm_quantum_account

        # 1. Load valid QASM 3.0 quantum circuit
        with open("tests/qasm/correct_qasm_1", "r") as f:
            qasm_string = f.read()

        # 2. Setup account
        setup_result = await setup_ibm_quantum_account()
        assert setup_result["status"] == "success"

        # 3. AI Routing
        ai_routing_result = await ai_routing(
            circuit_qasm=qasm_string, backend_name="ibm_fez"
        )
        assert ai_routing_result["status"] == "success"
        assert isinstance(ai_routing_result["optimized_circuit_qasm"], str)

        # 4. AI Clifford synthesis
        routed_qasm_circuit = ai_routing_result["optimized_circuit_qasm"]
        ai_clifford_synthesis_result = await ai_clifford_synthesis(
            circuit_qasm=routed_qasm_circuit, backend_name="ibm_fez"
        )
        assert ai_clifford_synthesis_result["status"] == "success"
        assert isinstance(ai_clifford_synthesis_result["optimized_circuit_qasm"], str)
