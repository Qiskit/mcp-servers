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
    active_account_info,
    active_instance_info,
    available_instances,
    cancel_job,
    delete_saved_account,
    find_optimal_qubit_chains,
    find_optimal_qv_qubits,
    get_backend_properties,
    get_job_results,
    get_job_status,
    get_service_status,
    least_busy_backend,
    list_backends,
    list_my_jobs,
    list_saved_accounts,
    setup_ibm_quantum_account,
    usage_info,
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

    def test_get_job_results_has_sync(self):
        """Test get_job_results has .sync attribute."""
        assert hasattr(get_job_results, "sync")
        assert callable(get_job_results.sync)

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

    def test_delete_saved_account_has_sync(self):
        """Test delete_saved_account has .sync attribute."""
        assert hasattr(delete_saved_account, "sync")
        assert callable(delete_saved_account.sync)

    def test_list_saved_accounts_has_sync(self):
        """Test list_saved_accounts has .sync attribute."""
        assert hasattr(list_saved_accounts, "sync")
        assert callable(list_saved_accounts.sync)

    def test_active_account_info_has_sync(self):
        """Test active_account_info has .sync attribute."""
        assert hasattr(active_account_info, "sync")
        assert callable(active_account_info.sync)

    def test_active_instance_info_has_sync(self):
        """Test active_instance_info has .sync attribute."""
        assert hasattr(active_instance_info, "sync")
        assert callable(active_instance_info.sync)

    def test_available_instances_has_sync(self):
        """Test available_instances has .sync attribute."""
        assert hasattr(available_instances, "sync")
        assert callable(available_instances.sync)

    def test_usage_info_has_sync(self):
        """Test usage_info has .sync attribute."""
        assert hasattr(usage_info, "sync")
        assert callable(usage_info.sync)


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

    def test_get_job_results_sync_success(self):
        """Test successful job results retrieval with .sync method."""
        mock_response = {
            "status": "success",
            "job_id": "job_123",
            "job_status": "DONE",
            "counts": {"00": 2048, "11": 2048},
            "shots": 4096,
            "backend": "ibm_brisbane",
        }

        with patch("qiskit_ibm_runtime_mcp_server.utils._run_async") as mock_run:
            mock_run.return_value = mock_response

            result = get_job_results.sync("job_123")

            assert result["status"] == "success"
            assert result["counts"] == {"00": 2048, "11": 2048}
            assert result["shots"] == 4096

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

    def test_delete_saved_account_sync_success(self):
        """Test successful account deletion with .sync method."""
        mock_response = {
            "status": "success",
            "deleted": True,
            "message": "Account successfully deleted",
        }

        with patch("qiskit_ibm_runtime_mcp_server.utils._run_async") as mock_run:
            mock_run.return_value = mock_response

            result = delete_saved_account.sync("test_account")

            assert result["status"] == "success"
            assert result["deleted"] is True
            assert "successfully deleted" in result["message"]

    def test_delete_saved_account_sync_error(self):
        """Test error handling in delete_saved_account .sync method."""
        mock_response = {
            "status": "error",
            "deleted": False,
            "error": "Account name not found or could not be deleted",
        }

        with patch("qiskit_ibm_runtime_mcp_server.utils._run_async") as mock_run:
            mock_run.return_value = mock_response

            result = delete_saved_account.sync("nonexistent_account")

            assert result["status"] == "error"
            assert result["deleted"] is False
            assert "not found" in result["error"]

    def test_list_saved_accounts_sync_success(self):
        """Test successful saved accounts listing with .sync method."""
        mock_response = {
            "status": "success",
            "accounts": [
                {"name": "ibm_quantum_platform", "channel": "ibm_quantum"},
                {"name": "custom_account", "channel": "ibm_cloud"},
            ],
        }

        with patch("qiskit_ibm_runtime_mcp_server.utils._run_async") as mock_run:
            mock_run.return_value = mock_response

            result = list_saved_accounts.sync()

            assert result["status"] == "success"
            assert "accounts" in result
            assert len(result["accounts"]) == 2

    def test_list_saved_accounts_sync_no_accounts(self):
        """Test saved accounts listing with no accounts."""
        mock_response = {
            "status": "success",
            "accounts": [],
            "message": "No accounts found",
        }

        with patch("qiskit_ibm_runtime_mcp_server.utils._run_async") as mock_run:
            mock_run.return_value = mock_response

            result = list_saved_accounts.sync()

            assert result["status"] == "success"
            assert result["accounts"] == []
            assert "No accounts found" in result["message"]

    def test_active_account_info_sync_success(self):
        """Test successful active account info with .sync method."""
        mock_response = {
            "status": "success",
            "account_info": {
                "channel": "ibm_quantum",
                "url": "https://auth.quantum-computing.ibm.com/api",
                "token": "test_token",
                "verify": True,
                "private_endpoint": False,
            },
        }

        with patch("qiskit_ibm_runtime_mcp_server.utils._run_async") as mock_run:
            mock_run.return_value = mock_response

            result = active_account_info.sync()

            assert result["status"] == "success"
            assert "account_info" in result
            assert result["account_info"]["channel"] == "ibm_quantum"

    def test_active_instance_info_sync_success(self):
        """Test successful active instance info with .sync method."""
        mock_response = {
            "status": "success",
            "instance_crn": "crn:v1:bluemix:public:quantum-computing:us-east:a/123:456::",
        }

        with patch("qiskit_ibm_runtime_mcp_server.utils._run_async") as mock_run:
            mock_run.return_value = mock_response

            result = active_instance_info.sync()

            assert result["status"] == "success"
            assert "instance_crn" in result
            assert result["instance_crn"].startswith("crn:v1:bluemix")

    def test_available_instances_sync_success(self):
        """Test successful available instances listing with .sync method."""
        mock_response = {
            "status": "success",
            "instances": [
                {
                    "crn": "crn:v1:bluemix:public:quantum-computing:us-east:a/123:456::",
                    "plan": "open",
                    "name": "My Instance",
                    "tags": [],
                    "pricing_type": "free",
                },
                {
                    "crn": "crn:v1:bluemix:public:quantum-computing:us-east:a/123:789::",
                    "plan": "premium",
                    "name": "Premium Instance",
                    "tags": ["production"],
                    "pricing_type": "paid",
                },
            ],
            "total_instances": 2,
        }

        with patch("qiskit_ibm_runtime_mcp_server.utils._run_async") as mock_run:
            mock_run.return_value = mock_response

            result = available_instances.sync()

            assert result["status"] == "success"
            assert "instances" in result
            assert result["total_instances"] == 2
            assert len(result["instances"]) == 2

    def test_usage_info_sync_success(self):
        """Test successful usage info with .sync method."""
        mock_response = {
            "status": "success",
            "usage": {
                "instance_id": "crn:v1:bluemix:public:quantum-computing:us-east:a/123:456::",
                "plan_id": "open",
                "usage_consumed_seconds": 3600,
                "usage_period": "2025-01",
                "usage_limit_seconds": 36000,
                "usage_limit_reached": False,
                "usage_remaining_seconds": 32400,
            },
        }

        with patch("qiskit_ibm_runtime_mcp_server.utils._run_async") as mock_run:
            mock_run.return_value = mock_response

            result = usage_info.sync()

            assert result["status"] == "success"
            assert "usage" in result
            assert result["usage"]["usage_consumed_seconds"] == 3600
            assert result["usage"]["usage_limit_reached"] is False
