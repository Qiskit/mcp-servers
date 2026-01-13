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

from qiskit_ibm_runtime_mcp_server.ibm_runtime import (
    cancel_job,
    find_optimal_qubit_chains,
    find_optimal_qv_qubits,
    get_backend_properties,
    get_job_status,
    get_service_status,
    least_busy_backend,
    list_backends,
    list_my_jobs,
    setup_ibm_quantum_account,
)


class TestWithSyncDecorator:
    """Test that async functions have .sync attribute."""

    def test_setup_ibm_quantum_account_has_sync(self):
        """Test setup_ibm_quantum_account has .sync attribute."""
        assert hasattr(setup_ibm_quantum_account, "sync")
        assert callable(setup_ibm_quantum_account.sync)

    def test_list_backends_has_sync(self):
        """Test list_backends has .sync attribute."""
        assert hasattr(list_backends, "sync")
        assert callable(list_backends.sync)

    def test_least_busy_backend_has_sync(self):
        """Test least_busy_backend has .sync attribute."""
        assert hasattr(least_busy_backend, "sync")
        assert callable(least_busy_backend.sync)

    def test_get_backend_properties_has_sync(self):
        """Test get_backend_properties has .sync attribute."""
        assert hasattr(get_backend_properties, "sync")
        assert callable(get_backend_properties.sync)

    def test_list_my_jobs_has_sync(self):
        """Test list_my_jobs has .sync attribute."""
        assert hasattr(list_my_jobs, "sync")
        assert callable(list_my_jobs.sync)

    def test_get_job_status_has_sync(self):
        """Test get_job_status has .sync attribute."""
        assert hasattr(get_job_status, "sync")
        assert callable(get_job_status.sync)

    def test_cancel_job_has_sync(self):
        """Test cancel_job has .sync attribute."""
        assert hasattr(cancel_job, "sync")
        assert callable(cancel_job.sync)

    def test_get_service_status_has_sync(self):
        """Test get_service_status has .sync attribute."""
        assert hasattr(get_service_status, "sync")
        assert callable(get_service_status.sync)

    def test_find_optimal_qubit_chains_has_sync(self):
        """Test find_optimal_qubit_chains has .sync attribute."""
        assert hasattr(find_optimal_qubit_chains, "sync")
        assert callable(find_optimal_qubit_chains.sync)

    def test_find_optimal_qv_qubits_has_sync(self):
        """Test find_optimal_qv_qubits has .sync attribute."""
        assert hasattr(find_optimal_qv_qubits, "sync")
        assert callable(find_optimal_qv_qubits.sync)


class TestSyncMethodExecution:
    """Test that .sync methods execute correctly."""

    def test_list_backends_sync_success(self):
        """Test successful backends listing with .sync method."""
        mock_response = {
            "status": "success",
            "backends": [
                {"name": "ibm_brisbane", "num_qubits": 127},
                {"name": "ibm_osaka", "num_qubits": 127},
            ],
            "total_backends": 2,
        }

        with patch("qiskit_ibm_runtime_mcp_server.utils._run_async") as mock_run:
            mock_run.return_value = mock_response

            result = list_backends.sync()

            assert result["status"] == "success"
            assert "backends" in result
            assert result["total_backends"] == 2

    def test_list_backends_sync_error(self):
        """Test error handling in .sync method."""
        mock_response = {"status": "error", "message": "Service not initialized"}

        with patch("qiskit_ibm_runtime_mcp_server.utils._run_async") as mock_run:
            mock_run.return_value = mock_response

            result = list_backends.sync()

            assert result["status"] == "error"
            assert "Service not initialized" in result["message"]

    def test_least_busy_backend_sync_success(self):
        """Test successful least busy backend with .sync method."""
        mock_response = {
            "status": "success",
            "backend_name": "ibm_brisbane",
            "num_qubits": 127,
            "pending_jobs": 5,
        }

        with patch("qiskit_ibm_runtime_mcp_server.utils._run_async") as mock_run:
            mock_run.return_value = mock_response

            result = least_busy_backend.sync()

            assert result["status"] == "success"
            assert result["backend_name"] == "ibm_brisbane"

    def test_get_backend_properties_sync_success(self):
        """Test successful backend properties with .sync method."""
        mock_response = {
            "status": "success",
            "backend_name": "ibm_brisbane",
            "num_qubits": 127,
            "operational": True,
        }

        with patch("qiskit_ibm_runtime_mcp_server.utils._run_async") as mock_run:
            mock_run.return_value = mock_response

            result = get_backend_properties.sync("ibm_brisbane")

            assert result["status"] == "success"
            assert result["backend_name"] == "ibm_brisbane"

    def test_list_my_jobs_sync_success(self):
        """Test successful job listing with .sync method."""
        mock_response = {
            "status": "success",
            "jobs": [{"job_id": "job_123", "status": "COMPLETED"}],
            "total_jobs": 1,
        }

        with patch("qiskit_ibm_runtime_mcp_server.utils._run_async") as mock_run:
            mock_run.return_value = mock_response

            result = list_my_jobs.sync(limit=5)

            assert result["status"] == "success"
            assert len(result["jobs"]) == 1

    def test_get_job_status_sync_success(self):
        """Test successful job status with .sync method."""
        mock_response = {
            "status": "success",
            "job_id": "job_123",
            "job_status": "COMPLETED",
        }

        with patch("qiskit_ibm_runtime_mcp_server.utils._run_async") as mock_run:
            mock_run.return_value = mock_response

            result = get_job_status.sync("job_123")

            assert result["status"] == "success"
            assert result["job_status"] == "COMPLETED"

    def test_cancel_job_sync_success(self):
        """Test successful job cancellation with .sync method."""
        mock_response = {
            "status": "success",
            "job_id": "job_123",
            "message": "Job cancellation requested",
        }

        with patch("qiskit_ibm_runtime_mcp_server.utils._run_async") as mock_run:
            mock_run.return_value = mock_response

            result = cancel_job.sync("job_123")

            assert result["status"] == "success"
            assert "cancellation" in result["message"]

    def test_get_service_status_sync_success(self):
        """Test successful service status with .sync method."""
        mock_response = "IBM Quantum Service Status: {'connected': True}"

        with patch("qiskit_ibm_runtime_mcp_server.utils._run_async") as mock_run:
            mock_run.return_value = mock_response

            result = get_service_status.sync()

            assert "connected" in result

    def test_setup_ibm_quantum_account_sync_success(self):
        """Test successful account setup with .sync method."""
        mock_response = {
            "status": "success",
            "message": "IBM Quantum account set up successfully",
            "channel": "ibm_quantum_platform",
            "available_backends": 10,
        }

        with patch("qiskit_ibm_runtime_mcp_server.utils._run_async") as mock_run:
            mock_run.return_value = mock_response

            result = setup_ibm_quantum_account.sync()

            assert result["status"] == "success"
            assert result["available_backends"] == 10

    def test_find_optimal_qubit_chains_sync_success(self):
        """Test successful qubit chain finding with .sync method."""
        mock_response = {
            "status": "success",
            "backend_name": "ibm_brisbane",
            "chain_length": 5,
            "metric": "two_qubit_error",
            "total_chains_found": 100,
            "faulty_qubits": [],
            "chains": [
                {
                    "rank": 1,
                    "qubits": [0, 1, 2, 3, 4],
                    "score": 0.05,
                    "qubit_details": [],
                    "edge_errors": [],
                }
            ],
        }

        with patch("qiskit_ibm_runtime_mcp_server.utils._run_async") as mock_run:
            mock_run.return_value = mock_response

            result = find_optimal_qubit_chains.sync("ibm_brisbane", chain_length=5)

            assert result["status"] == "success"
            assert result["chain_length"] == 5
            assert len(result["chains"]) == 1

    def test_find_optimal_qv_qubits_sync_success(self):
        """Test successful QV qubit finding with .sync method."""
        mock_response = {
            "status": "success",
            "backend_name": "ibm_brisbane",
            "num_qubits": 5,
            "metric": "qv_optimized",
            "total_subgraphs_found": 50,
            "faulty_qubits": [],
            "subgraphs": [
                {
                    "rank": 1,
                    "qubits": [0, 1, 2, 3, 4],
                    "score": 0.1,
                    "internal_edges": 6,
                    "connectivity_ratio": 0.6,
                    "average_path_length": 1.5,
                    "qubit_details": [],
                    "edge_errors": [],
                }
            ],
        }

        with patch("qiskit_ibm_runtime_mcp_server.utils._run_async") as mock_run:
            mock_run.return_value = mock_response

            result = find_optimal_qv_qubits.sync("ibm_brisbane", num_qubits=5)

            assert result["status"] == "success"
            assert result["num_qubits"] == 5
            assert len(result["subgraphs"]) == 1
            assert result["subgraphs"][0]["connectivity_ratio"] == 0.6
