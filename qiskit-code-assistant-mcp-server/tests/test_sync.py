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

"""Unit tests for the with_sync decorator and .sync methods."""

from unittest.mock import patch

from qiskit_code_assistant_mcp_server.qca import (
    accept_completion,
    get_completion,
    get_model,
    get_rag_completion,
    get_service_status,
    list_models,
)


class TestWithSyncDecorator:
    """Test that async functions have .sync attribute."""

    def test_list_models_has_sync(self):
        """Test list_models has .sync attribute."""
        assert hasattr(list_models, "sync")
        assert callable(list_models.sync)

    def test_get_model_has_sync(self):
        """Test get_model has .sync attribute."""
        assert hasattr(get_model, "sync")
        assert callable(get_model.sync)

    def test_get_completion_has_sync(self):
        """Test get_completion has .sync attribute."""
        assert hasattr(get_completion, "sync")
        assert callable(get_completion.sync)

    def test_get_rag_completion_has_sync(self):
        """Test get_rag_completion has .sync attribute."""
        assert hasattr(get_rag_completion, "sync")
        assert callable(get_rag_completion.sync)

    def test_accept_completion_has_sync(self):
        """Test accept_completion has .sync attribute."""
        assert hasattr(accept_completion, "sync")
        assert callable(accept_completion.sync)

    def test_get_service_status_has_sync(self):
        """Test get_service_status has .sync attribute."""
        assert hasattr(get_service_status, "sync")
        assert callable(get_service_status.sync)


class TestSyncMethodExecution:
    """Test that .sync methods execute correctly."""

    def test_list_models_sync_success(self, mock_env_vars):
        """Test successful models listing with .sync method."""
        mock_response = {
            "status": "success",
            "models": [
                {"id": "mistral-small-3.2-24b-qiskit", "name": "Granite Qiskit"},
                {"id": "granite-3.3-2b-qiskit", "name": "Granite Qiskit Small"},
            ],
        }

        with patch("qiskit_code_assistant_mcp_server.utils._run_async") as mock_run:
            mock_run.return_value = mock_response

            result = list_models.sync()

            assert result["status"] == "success"
            assert "models" in result
            assert len(result["models"]) == 2

    def test_list_models_sync_error(self, mock_env_vars):
        """Test error handling in .sync method."""
        mock_response = {"status": "error", "message": "Authentication failed"}

        with patch("qiskit_code_assistant_mcp_server.utils._run_async") as mock_run:
            mock_run.return_value = mock_response

            result = list_models.sync()

            assert result["status"] == "error"
            assert "Authentication failed" in result["message"]

    def test_get_model_sync_success(self, mock_env_vars):
        """Test successful model retrieval with .sync method."""
        mock_response = {
            "status": "success",
            "model": {"id": "mistral-small-3.2-24b-qiskit", "name": "Granite Qiskit"},
        }

        with patch("qiskit_code_assistant_mcp_server.utils._run_async") as mock_run:
            mock_run.return_value = mock_response

            result = get_model.sync("mistral-small-3.2-24b-qiskit")

            assert result["status"] == "success"
            assert result["model"]["id"] == "mistral-small-3.2-24b-qiskit"

    def test_get_completion_sync_success(self, mock_env_vars):
        """Test successful code completion with .sync method."""
        mock_response = {
            "status": "success",
            "completion_id": "comp_123",
            "choices": [{"text": "from qiskit import QuantumCircuit"}],
        }

        with patch("qiskit_code_assistant_mcp_server.utils._run_async") as mock_run:
            mock_run.return_value = mock_response

            result = get_completion.sync("Create a quantum circuit")

            assert result["status"] == "success"
            assert "completion_id" in result
            assert len(result["choices"]) > 0

    def test_get_rag_completion_sync_success(self, mock_env_vars):
        """Test successful RAG completion with .sync method."""
        mock_response = {
            "status": "success",
            "completion_id": "rag_123",
            "choices": [{"text": "Quantum entanglement is..."}],
        }

        with patch("qiskit_code_assistant_mcp_server.utils._run_async") as mock_run:
            mock_run.return_value = mock_response

            result = get_rag_completion.sync("What is quantum entanglement?")

            assert result["status"] == "success"
            assert "choices" in result

    def test_accept_completion_sync_success(self, mock_env_vars):
        """Test successful completion acceptance with .sync method."""
        mock_response = {"status": "success", "result": "accepted"}

        with patch("qiskit_code_assistant_mcp_server.utils._run_async") as mock_run:
            mock_run.return_value = mock_response

            result = accept_completion.sync("comp_123")

            assert result["status"] == "success"
            assert result["result"] == "accepted"

    def test_get_service_status_sync_success(self, mock_env_vars):
        """Test successful service status check with .sync method."""
        mock_response = "Qiskit Code Assistant Service Status: {'connected': True}"

        with patch("qiskit_code_assistant_mcp_server.utils._run_async") as mock_run:
            mock_run.return_value = mock_response

            result = get_service_status.sync()

            assert "connected" in result
