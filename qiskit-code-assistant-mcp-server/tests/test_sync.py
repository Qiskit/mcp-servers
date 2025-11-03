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

"""Unit tests for sync wrapper functions."""

from unittest.mock import patch

from qiskit_code_assistant_mcp_server.sync import (
    qca_list_models_sync,
    qca_get_model_sync,
    qca_get_completion_sync,
    qca_get_rag_completion_sync,
    qca_accept_completion_sync,
    qca_get_service_status_sync,
)


class TestQCAListModelsSync:
    """Test qca_list_models_sync function."""

    def test_list_models_sync_success(self, mock_env_vars):
        """Test successful models listing with sync wrapper."""
        mock_response = {
            "status": "success",
            "models": [
                {"id": "mistral-small-3.2-24b-qiskit", "name": "Granite Qiskit"},
                {"id": "granite-3.3-2b-qiskit", "name": "Granite Qiskit Small"},
            ],
        }

        with patch("qiskit_code_assistant_mcp_server.sync._run_async") as mock_run:
            mock_run.return_value = mock_response

            result = qca_list_models_sync()

            assert result["status"] == "success"
            assert "models" in result
            assert len(result["models"]) == 2

    def test_list_models_sync_error(self, mock_env_vars):
        """Test error handling in sync wrapper."""
        mock_response = {"status": "error", "message": "Authentication failed"}

        with patch("qiskit_code_assistant_mcp_server.sync._run_async") as mock_run:
            mock_run.return_value = mock_response

            result = qca_list_models_sync()

            assert result["status"] == "error"
            assert "Authentication failed" in result["message"]


class TestQCAGetModelSync:
    """Test qca_get_model_sync function."""

    def test_get_model_sync_success(self, mock_env_vars):
        """Test successful model retrieval with sync wrapper."""
        mock_response = {
            "status": "success",
            "model": {"id": "mistral-small-3.2-24b-qiskit", "name": "Granite Qiskit"},
        }

        with patch("qiskit_code_assistant_mcp_server.sync._run_async") as mock_run:
            mock_run.return_value = mock_response

            result = qca_get_model_sync("mistral-small-3.2-24b-qiskit")

            assert result["status"] == "success"
            assert result["model"]["id"] == "mistral-small-3.2-24b-qiskit"

    def test_get_model_sync_empty_id(self, mock_env_vars):
        """Test validation of empty model_id."""
        mock_response = {
            "status": "error",
            "message": "model_id is required and cannot be empty",
        }

        with patch("qiskit_code_assistant_mcp_server.sync._run_async") as mock_run:
            mock_run.return_value = mock_response

            result = qca_get_model_sync("")

            assert result["status"] == "error"


class TestQCAGetCompletionSync:
    """Test qca_get_completion_sync function."""

    def test_get_completion_sync_success(self, mock_env_vars):
        """Test successful code completion with sync wrapper."""
        mock_response = {
            "status": "success",
            "completion_id": "comp_123",
            "choices": [{"text": "from qiskit import QuantumCircuit"}],
        }

        with patch("qiskit_code_assistant_mcp_server.sync._run_async") as mock_run:
            mock_run.return_value = mock_response

            result = qca_get_completion_sync("Create a quantum circuit")

            assert result["status"] == "success"
            assert "completion_id" in result
            assert len(result["choices"]) > 0

    def test_get_completion_sync_empty_prompt(self, mock_env_vars):
        """Test validation of empty prompt."""
        mock_response = {
            "status": "error",
            "message": "prompt is required and cannot be empty",
        }

        with patch("qiskit_code_assistant_mcp_server.sync._run_async") as mock_run:
            mock_run.return_value = mock_response

            result = qca_get_completion_sync("")

            assert result["status"] == "error"


class TestQCAGetRagCompletionSync:
    """Test qca_get_rag_completion_sync function."""

    def test_get_rag_completion_sync_success(self, mock_env_vars):
        """Test successful RAG completion with sync wrapper."""
        mock_response = {
            "status": "success",
            "completion_id": "rag_123",
            "choices": [{"text": "Quantum entanglement is..."}],
        }

        with patch("qiskit_code_assistant_mcp_server.sync._run_async") as mock_run:
            mock_run.return_value = mock_response

            result = qca_get_rag_completion_sync("What is quantum entanglement?")

            assert result["status"] == "success"
            assert "choices" in result


class TestQCAAcceptCompletionSync:
    """Test qca_accept_completion_sync function."""

    def test_accept_completion_sync_success(self, mock_env_vars):
        """Test successful completion acceptance with sync wrapper."""
        mock_response = {"status": "success", "result": "accepted"}

        with patch("qiskit_code_assistant_mcp_server.sync._run_async") as mock_run:
            mock_run.return_value = mock_response

            result = qca_accept_completion_sync("comp_123")

            assert result["status"] == "success"
            assert result["result"] == "accepted"


class TestQCAGetServiceStatusSync:
    """Test qca_get_service_status_sync function."""

    def test_get_service_status_sync_success(self, mock_env_vars):
        """Test successful service status check with sync wrapper."""
        mock_response = "Qiskit Code Assistant Service Status: {'connected': True}"

        with patch("qiskit_code_assistant_mcp_server.sync._run_async") as mock_run:
            mock_run.return_value = mock_response

            result = qca_get_service_status_sync()

            assert "connected" in result


# Assisted by watsonx Code Assistant
