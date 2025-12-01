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

"""Integration tests for Qiskit Code Assistant MCP Server."""

from unittest.mock import patch

import httpx
import pytest
import respx

from qiskit_code_assistant_mcp_server.server import mcp


# Use the test API base URL that matches mock_env_vars fixture
TEST_QCA_API_BASE = "https://test-qca-api.example.com"


class TestMCPServerIntegration:
    """Test MCP server integration."""

    @pytest.mark.asyncio
    async def test_server_initialization(self, mock_env_vars):
        """Test that server initializes correctly."""
        # Server should initialize without errors
        assert mcp is not None
        assert mcp.name == "Qiskit Code Assistant"

    @pytest.mark.asyncio
    async def test_configuration_validation(self, mock_env_vars):
        """Test that configuration validation runs on startup."""
        with patch(
            "qiskit_code_assistant_mcp_server.constants.validate_configuration"
        ) as mock_validate:
            mock_validate.return_value = True

            # Reimport server module to trigger validation
            import importlib

            import qiskit_code_assistant_mcp_server.server

            importlib.reload(qiskit_code_assistant_mcp_server.server)

            mock_validate.assert_called_once()


class TestResourceHandlers:
    """Test MCP resource handlers."""

    @pytest.mark.asyncio
    async def test_models_resource(self, mock_env_vars, mock_http_responses):
        """Test qca://models resource handler."""
        from qiskit_code_assistant_mcp_server.qca import qca_list_models

        result = await qca_list_models()

        assert result["status"] == "success"
        assert "models" in result
        assert len(result["models"]) > 0

    @pytest.mark.asyncio
    async def test_model_resource(self, mock_env_vars, mock_http_responses):
        """Test qca://model/{model_id} resource handler."""
        from qiskit_code_assistant_mcp_server.qca import qca_get_model

        result = await qca_get_model("mistral-small-3.2-24b-qiskit")

        assert result["status"] == "success"
        assert "model" in result
        assert result["model"]["id"] == "mistral-small-3.2-24b-qiskit"

    @pytest.mark.asyncio
    async def test_disclaimer_resource(self, mock_env_vars, mock_http_responses):
        """Test qca://disclaimer/{model_id} resource handler."""
        from qiskit_code_assistant_mcp_server.qca import qca_get_model_disclaimer

        result = await qca_get_model_disclaimer("mistral-small-3.2-24b-qiskit")

        assert result["status"] == "success"
        assert "disclaimer" in result

    @pytest.mark.asyncio
    async def test_status_resource(self, mock_env_vars, mock_http_responses):
        """Test qca://status resource handler."""
        from qiskit_code_assistant_mcp_server.qca import qca_get_service_status

        result = await qca_get_service_status()

        assert "Qiskit Code Assistant Service Status" in result


class TestToolHandlers:
    """Test MCP tool handlers."""

    @pytest.mark.asyncio
    async def test_completion_tool(self, mock_env_vars, mock_http_responses):
        """Test qca_get_completion tool."""
        from qiskit_code_assistant_mcp_server.qca import qca_get_completion

        result = await qca_get_completion("Create a quantum circuit")

        assert result["status"] == "success"
        assert "completion_id" in result
        assert "choices" in result

    @pytest.mark.asyncio
    async def test_rag_completion_tool(self, mock_env_vars, mock_http_responses):
        """Test qca_get_rag_completion tool."""
        from qiskit_code_assistant_mcp_server.qca import qca_get_rag_completion

        result = await qca_get_rag_completion("What is quantum entanglement?")

        assert result["status"] == "success"
        assert "completion_id" in result
        assert "choices" in result

    @pytest.mark.asyncio
    async def test_accept_disclaimer_tool(self, mock_env_vars, mock_http_responses):
        """Test qca_accept_model_disclaimer tool."""
        from qiskit_code_assistant_mcp_server.qca import qca_accept_model_disclaimer

        result = await qca_accept_model_disclaimer("mistral-small-3.2-24b-qiskit", "disclaimer_123")

        assert result["status"] == "success"
        assert "result" in result

    @pytest.mark.asyncio
    async def test_accept_completion_tool(self, mock_env_vars, mock_http_responses):
        """Test qca_accept_completion tool."""
        from qiskit_code_assistant_mcp_server.qca import qca_accept_completion

        result = await qca_accept_completion("completion_456")

        assert result["status"] == "success"
        assert "result" in result


class TestErrorHandling:
    """Test error handling in integration scenarios."""

    @pytest.mark.asyncio
    async def test_network_timeout_handling(self, mock_env_vars):
        """Test handling of network timeouts."""
        with respx.mock() as respx_mock:
            respx_mock.get(f"{TEST_QCA_API_BASE}/v1/models").mock(
                side_effect=httpx.TimeoutException("Request timeout")
            )

            from qiskit_code_assistant_mcp_server.qca import qca_list_models

            result = await qca_list_models()

            assert result["status"] == "error"
            assert "timeout" in result["message"].lower() or "failed" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_authentication_error_handling(self, mock_env_vars):
        """Test handling of authentication errors."""
        with respx.mock() as respx_mock:
            respx_mock.get(f"{TEST_QCA_API_BASE}/v1/models").mock(
                return_value=httpx.Response(401, json={"detail": "Invalid token"})
            )

            from qiskit_code_assistant_mcp_server.qca import qca_list_models

            result = await qca_list_models()

            assert result["status"] == "error"

    @pytest.mark.asyncio
    async def test_server_error_handling(self, mock_env_vars):
        """Test handling of server errors."""
        with respx.mock() as respx_mock:
            respx_mock.post(f"{TEST_QCA_API_BASE}/v1/completions").mock(
                return_value=httpx.Response(500, json={"detail": "Internal server error"})
            )

            from qiskit_code_assistant_mcp_server.qca import qca_get_completion

            result = await qca_get_completion("test prompt")

            assert result["status"] == "error"


class TestEndToEndScenarios:
    """Test end-to-end scenarios."""

    @pytest.mark.asyncio
    async def test_complete_workflow(self, mock_env_vars, mock_http_responses):
        """Test complete workflow: list models -> get completion -> accept."""
        from qiskit_code_assistant_mcp_server.qca import (
            qca_accept_completion,
            qca_get_completion,
            qca_list_models,
        )

        # 1. List models
        models_result = await qca_list_models()
        assert models_result["status"] == "success"

        # 2. Get completion
        completion_result = await qca_get_completion("Create a quantum circuit")
        assert completion_result["status"] == "success"
        completion_id = completion_result["completion_id"]

        # 3. Accept completion
        accept_result = await qca_accept_completion(completion_id)
        assert accept_result["status"] == "success"

    @pytest.mark.asyncio
    async def test_disclaimer_workflow(self, mock_env_vars, mock_http_responses):
        """Test disclaimer workflow: get disclaimer -> accept disclaimer."""
        from qiskit_code_assistant_mcp_server.qca import (
            qca_accept_model_disclaimer,
            qca_get_model_disclaimer,
        )

        # 1. Get disclaimer
        disclaimer_result = await qca_get_model_disclaimer("mistral-small-3.2-24b-qiskit")
        assert disclaimer_result["status"] == "success"
        disclaimer_id = disclaimer_result["disclaimer"]["id"]

        # 2. Accept disclaimer
        accept_result = await qca_accept_model_disclaimer(
            "mistral-small-3.2-24b-qiskit", disclaimer_id
        )
        assert accept_result["status"] == "success"


# Assisted by watsonx Code Assistant
