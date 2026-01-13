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

"""Integration tests for Qiskit IBM Runtime MCP Server."""

from unittest.mock import Mock, patch

import pytest

from qiskit_ibm_runtime_mcp_server.server import mcp


class TestMCPServerIntegration:
    """Test MCP server integration."""

    @pytest.mark.asyncio
    async def test_server_initialization(self, mock_env_vars):
        """Test that server initializes correctly."""
        # Server should initialize without errors
        assert mcp is not None
        assert mcp.name == "Qiskit IBM Runtime"

    @pytest.mark.asyncio
    async def test_service_initialization_flow(
        self, mock_env_vars, mock_runtime_service
    ):
        """Test service initialization flow."""
        from qiskit_ibm_runtime_mcp_server.ibm_runtime import initialize_service

        with patch(
            "qiskit_ibm_runtime_mcp_server.ibm_runtime.QiskitRuntimeService"
        ) as mock_qrs:
            mock_qrs.return_value = mock_runtime_service

            service = initialize_service()

            assert service == mock_runtime_service


class TestToolIntegration:
    """Test MCP tool integration."""

    @pytest.mark.asyncio
    async def test_setup_and_list_backends_workflow(
        self, mock_env_vars, mock_runtime_service
    ):
        """Test setup account -> list backends workflow."""
        from qiskit_ibm_runtime_mcp_server.ibm_runtime import (
            list_backends,
            setup_ibm_quantum_account,
        )

        with patch(
            "qiskit_ibm_runtime_mcp_server.ibm_runtime.initialize_service"
        ) as mock_init:
            mock_init.return_value = mock_runtime_service

            # 1. Setup account
            setup_result = await setup_ibm_quantum_account("test_token")
            assert setup_result["status"] == "success"

            # 2. List backends
            backends_result = await list_backends()
            assert backends_result["status"] == "success"
            assert len(backends_result["backends"]) > 0

    @pytest.mark.asyncio
    async def test_backend_analysis_workflow(self, mock_env_vars, mock_runtime_service):
        """Test backend analysis workflow: list -> least busy -> properties."""
        from qiskit_ibm_runtime_mcp_server.ibm_runtime import (
            get_backend_properties,
            least_busy_backend,
            list_backends,
        )

        with (
            patch(
                "qiskit_ibm_runtime_mcp_server.ibm_runtime.initialize_service"
            ) as mock_init,
            patch(
                "qiskit_ibm_runtime_mcp_server.ibm_runtime.least_busy"
            ) as mock_least_busy,
        ):
            mock_init.return_value = mock_runtime_service

            # Mock least busy backend
            mock_backend = Mock()
            mock_backend.name = "ibm_brisbane"
            mock_backend.num_qubits = 127
            mock_backend.status.return_value = Mock(
                operational=True, pending_jobs=2, status_msg="active"
            )
            mock_least_busy.return_value = mock_backend

            # 1. List all backends
            backends_result = await list_backends()
            assert backends_result["status"] == "success"

            # 2. Get least busy backend
            least_busy_result = await least_busy_backend()
            assert least_busy_result["status"] == "success"
            backend_name = least_busy_result["backend_name"]

            # 3. Get properties of the least busy backend
            properties_result = await get_backend_properties(backend_name)
            assert properties_result["status"] == "success"

    @pytest.mark.asyncio
    async def test_job_management_workflow(self, mock_env_vars, mock_runtime_service):
        """Test job management workflow: list jobs -> get status -> cancel."""
        from qiskit_ibm_runtime_mcp_server.ibm_runtime import (
            cancel_job,
            get_job_status,
            list_my_jobs,
        )

        with patch(
            "qiskit_ibm_runtime_mcp_server.ibm_runtime.service", mock_runtime_service
        ):
            # 1. List jobs
            jobs_result = await list_my_jobs(5)
            assert jobs_result["status"] == "success"

            if jobs_result["total_jobs"] > 0:
                job_id = jobs_result["jobs"][0]["job_id"]

                # 2. Get job status
                status_result = await get_job_status(job_id)
                assert status_result["status"] == "success"

                # 3. Cancel job (if not already completed)
                cancel_result = await cancel_job(job_id)
                assert cancel_result["status"] == "success"


class TestCouplingMapTool:
    """Test coupling map tool functionality."""

    @pytest.mark.asyncio
    async def test_get_coupling_map_success(self, mock_env_vars, mock_runtime_service):
        """Test getting coupling map for a backend."""
        from qiskit_ibm_runtime_mcp_server.ibm_runtime import get_coupling_map

        with patch(
            "qiskit_ibm_runtime_mcp_server.ibm_runtime.initialize_service"
        ) as mock_init:
            mock_init.return_value = mock_runtime_service

            # Mock backend with coupling map
            mock_backend = Mock()
            mock_backend.name = "ibm_brisbane"
            mock_backend.num_qubits = 5
            mock_config = Mock()
            mock_config.coupling_map = [
                [0, 1],
                [1, 0],
                [1, 2],
                [2, 1],
                [2, 3],
                [3, 2],
                [3, 4],
                [4, 3],
            ]
            mock_backend.configuration.return_value = mock_config
            mock_runtime_service.backend.return_value = mock_backend

            result = await get_coupling_map("ibm_brisbane")

            assert result["status"] == "success"
            assert result["backend_name"] == "ibm_brisbane"
            assert result["num_qubits"] == 5
            assert result["num_edges"] == 8
            assert len(result["edges"]) == 8
            assert result["bidirectional"] is True
            assert "adjacency_list" in result
            assert result["adjacency_list"]["0"] == [1]
            assert result["adjacency_list"]["1"] == [0, 2]

    @pytest.mark.asyncio
    async def test_get_coupling_map_unidirectional(
        self, mock_env_vars, mock_runtime_service
    ):
        """Test coupling map with unidirectional edges."""
        from qiskit_ibm_runtime_mcp_server.ibm_runtime import get_coupling_map

        with patch(
            "qiskit_ibm_runtime_mcp_server.ibm_runtime.initialize_service"
        ) as mock_init:
            mock_init.return_value = mock_runtime_service

            mock_backend = Mock()
            mock_backend.name = "test_backend"
            mock_backend.num_qubits = 3
            mock_config = Mock()
            mock_config.coupling_map = [[0, 1], [1, 2]]  # Only one direction
            mock_backend.configuration.return_value = mock_config
            mock_runtime_service.backend.return_value = mock_backend

            result = await get_coupling_map("test_backend")

            assert result["status"] == "success"
            assert result["bidirectional"] is False

    @pytest.mark.asyncio
    async def test_get_coupling_map_error(self, mock_env_vars, mock_runtime_service):
        """Test coupling map with backend error."""
        from qiskit_ibm_runtime_mcp_server.ibm_runtime import get_coupling_map

        with patch(
            "qiskit_ibm_runtime_mcp_server.ibm_runtime.initialize_service"
        ) as mock_init:
            mock_init.return_value = mock_runtime_service
            mock_runtime_service.backend.side_effect = Exception("Backend not found")

            result = await get_coupling_map("nonexistent_backend")

            assert result["status"] == "error"
            assert "Backend not found" in result["message"]

    @pytest.mark.asyncio
    async def test_get_coupling_map_fake_backend(self):
        """Test getting coupling map from a fake backend (no credentials needed)."""
        from qiskit_ibm_runtime_mcp_server.ibm_runtime import get_coupling_map

        # This test uses the real fake provider - no mocking needed
        result = await get_coupling_map("fake_sherbrooke")

        assert result["status"] == "success"
        assert result["backend_name"] == "fake_sherbrooke"
        assert result["num_qubits"] > 0
        assert result["num_edges"] > 0
        assert len(result["edges"]) > 0
        assert "bidirectional" in result
        assert "adjacency_list" in result
        assert result["source"] == "fake_backend"

    @pytest.mark.asyncio
    async def test_get_coupling_map_fake_backend_not_found(self):
        """Test fake backend not found error."""
        from qiskit_ibm_runtime_mcp_server.ibm_runtime import get_coupling_map

        result = await get_coupling_map("fake_nonexistent_backend")

        assert result["status"] == "error"
        assert "not found" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_get_coupling_map_fake_nighthawk(self):
        """Test getting coupling map from FakeNighthawk backend.

        FakeNighthawk uses a rectangular grid topology unlike other IBM processors
        which use heavy-hex topology. This test verifies the tool works with
        different processor architectures.

        Note: FakeNighthawk was added in qiskit-ibm-runtime 0.44.0.
        """
        from qiskit_ibm_runtime_mcp_server.ibm_runtime import get_coupling_map

        # Check if FakeNighthawk is available in the current version
        try:
            from qiskit_ibm_runtime.fake_provider import FakeProviderForBackendV2

            provider = FakeProviderForBackendV2()
            available = [b.name for b in provider.backends()]
            has_nighthawk = "fake_nighthawk" in available
        except ImportError:
            has_nighthawk = False

        if not has_nighthawk:
            pytest.skip(
                "FakeNighthawk not available (requires qiskit-ibm-runtime >= 0.44.0)"
            )

        result = await get_coupling_map("fake_nighthawk")

        assert result["status"] == "success"
        assert result["backend_name"] == "fake_nighthawk"
        assert result["num_qubits"] > 0
        assert result["num_edges"] > 0
        assert len(result["edges"]) > 0
        assert "bidirectional" in result
        assert "adjacency_list" in result
        assert result["source"] == "fake_backend"


class TestOptimalQubitChainsTool:
    """Test optimal qubit chains tool functionality."""

    @pytest.mark.asyncio
    async def test_find_optimal_chains_success(
        self, mock_env_vars, mock_runtime_service
    ):
        """Test finding optimal chains successfully."""
        from qiskit_ibm_runtime_mcp_server.ibm_runtime import find_optimal_qubit_chains

        with patch(
            "qiskit_ibm_runtime_mcp_server.ibm_runtime.initialize_service"
        ) as mock_init:
            mock_init.return_value = mock_runtime_service

            # Mock backend with coupling map
            mock_backend = Mock()
            mock_backend.name = "ibm_test"
            mock_backend.num_qubits = 5

            # Create a simple linear coupling map: 0-1-2-3-4
            mock_config = Mock()
            mock_config.coupling_map = [
                [0, 1],
                [1, 0],
                [1, 2],
                [2, 1],
                [2, 3],
                [3, 2],
                [3, 4],
                [4, 3],
            ]
            mock_backend.configuration.return_value = mock_config

            # Mock properties with calibration data
            mock_properties = Mock()
            mock_properties.faulty_qubits.return_value = []
            mock_properties.t1.return_value = 100e-6  # 100 microseconds
            mock_properties.t2.return_value = 50e-6  # 50 microseconds
            mock_properties.readout_error.return_value = 0.01
            mock_properties.gate_error.return_value = 0.005
            mock_backend.properties.return_value = mock_properties

            mock_runtime_service.backend.return_value = mock_backend

            result = await find_optimal_qubit_chains(
                "ibm_test", chain_length=3, num_results=3
            )

            assert result["status"] == "success"
            assert result["backend_name"] == "ibm_test"
            assert result["chain_length"] == 3
            assert result["metric"] == "two_qubit_error"
            assert result["total_chains_found"] > 0
            assert len(result["chains"]) <= 3
            assert result["chains"][0]["rank"] == 1
            assert len(result["chains"][0]["qubits"]) == 3
            assert "qubit_details" in result["chains"][0]
            assert "edge_errors" in result["chains"][0]

    @pytest.mark.asyncio
    async def test_find_optimal_chains_no_valid_chains(
        self, mock_env_vars, mock_runtime_service
    ):
        """Test when no valid chains exist."""
        from qiskit_ibm_runtime_mcp_server.ibm_runtime import find_optimal_qubit_chains

        with patch(
            "qiskit_ibm_runtime_mcp_server.ibm_runtime.initialize_service"
        ) as mock_init:
            mock_init.return_value = mock_runtime_service

            # Mock backend with disconnected qubits (no chains possible)
            mock_backend = Mock()
            mock_backend.name = "ibm_test"
            mock_backend.num_qubits = 3

            mock_config = Mock()
            mock_config.coupling_map = []  # No connections
            mock_backend.configuration.return_value = mock_config

            mock_properties = Mock()
            mock_properties.faulty_qubits.return_value = []
            mock_backend.properties.return_value = mock_properties

            mock_runtime_service.backend.return_value = mock_backend

            result = await find_optimal_qubit_chains("ibm_test", chain_length=3)

            assert result["status"] == "error"
            assert "No valid chains" in result["message"]

    @pytest.mark.asyncio
    async def test_find_optimal_chains_excludes_faulty_qubits(
        self, mock_env_vars, mock_runtime_service
    ):
        """Test that faulty qubits are excluded from chains."""
        from qiskit_ibm_runtime_mcp_server.ibm_runtime import find_optimal_qubit_chains

        with patch(
            "qiskit_ibm_runtime_mcp_server.ibm_runtime.initialize_service"
        ) as mock_init:
            mock_init.return_value = mock_runtime_service

            mock_backend = Mock()
            mock_backend.name = "ibm_test"
            mock_backend.num_qubits = 6

            # Branching topology allowing chains even with qubit 2 faulty:
            #   0 - 1 - 2 - 3
            #       |
            #       4 - 5
            mock_config = Mock()
            mock_config.coupling_map = [
                [0, 1],
                [1, 0],
                [1, 2],
                [2, 1],
                [2, 3],
                [3, 2],
                [1, 4],
                [4, 1],
                [4, 5],
                [5, 4],
            ]
            mock_backend.configuration.return_value = mock_config

            mock_properties = Mock()
            mock_properties.faulty_qubits.return_value = [2]  # Qubit 2 is faulty
            mock_properties.t1.return_value = 100e-6
            mock_properties.t2.return_value = 50e-6
            mock_properties.readout_error.return_value = 0.01
            mock_properties.gate_error.return_value = 0.005
            mock_backend.properties.return_value = mock_properties

            mock_runtime_service.backend.return_value = mock_backend

            result = await find_optimal_qubit_chains("ibm_test", chain_length=3)

            assert result["status"] == "success"
            assert 2 in result["faulty_qubits"]
            # Verify no chain contains the faulty qubit
            for chain in result["chains"]:
                assert 2 not in chain["qubits"]

    @pytest.mark.asyncio
    async def test_find_optimal_chains_different_metrics(
        self, mock_env_vars, mock_runtime_service
    ):
        """Test different scoring metrics."""
        from qiskit_ibm_runtime_mcp_server.ibm_runtime import find_optimal_qubit_chains

        with patch(
            "qiskit_ibm_runtime_mcp_server.ibm_runtime.initialize_service"
        ) as mock_init:
            mock_init.return_value = mock_runtime_service

            mock_backend = Mock()
            mock_backend.name = "ibm_test"
            mock_backend.num_qubits = 5

            mock_config = Mock()
            mock_config.coupling_map = [
                [0, 1],
                [1, 0],
                [1, 2],
                [2, 1],
                [2, 3],
                [3, 2],
                [3, 4],
                [4, 3],
            ]
            mock_backend.configuration.return_value = mock_config

            mock_properties = Mock()
            mock_properties.faulty_qubits.return_value = []
            mock_properties.t1.return_value = 100e-6
            mock_properties.t2.return_value = 50e-6
            mock_properties.readout_error.return_value = 0.01
            mock_properties.gate_error.return_value = 0.005
            mock_backend.properties.return_value = mock_properties

            mock_runtime_service.backend.return_value = mock_backend

            # Test each metric
            for metric in ["two_qubit_error", "readout_error", "combined"]:
                result = await find_optimal_qubit_chains(
                    "ibm_test", chain_length=3, metric=metric
                )

                assert result["status"] == "success"
                assert result["metric"] == metric
                assert len(result["chains"]) > 0

    @pytest.mark.asyncio
    async def test_find_optimal_chains_no_calibration_data(
        self, mock_env_vars, mock_runtime_service
    ):
        """Test error when backend has no calibration data."""
        from qiskit_ibm_runtime_mcp_server.ibm_runtime import find_optimal_qubit_chains

        with patch(
            "qiskit_ibm_runtime_mcp_server.ibm_runtime.initialize_service"
        ) as mock_init:
            mock_init.return_value = mock_runtime_service

            mock_backend = Mock()
            mock_backend.name = "ibm_test"
            mock_backend.num_qubits = 5

            mock_config = Mock()
            mock_config.coupling_map = [[0, 1], [1, 0]]
            mock_backend.configuration.return_value = mock_config
            mock_backend.properties.return_value = None  # No calibration data

            mock_runtime_service.backend.return_value = mock_backend

            result = await find_optimal_qubit_chains("ibm_test", chain_length=2)

            assert result["status"] == "error"
            assert "calibration" in result["message"].lower()


class TestOptimalQVQubitsTool:
    """Test optimal QV qubits tool functionality."""

    @pytest.mark.asyncio
    async def test_find_optimal_qv_qubits_success(
        self, mock_env_vars, mock_runtime_service
    ):
        """Test finding optimal QV qubits successfully."""
        from qiskit_ibm_runtime_mcp_server.ibm_runtime import find_optimal_qv_qubits

        with patch(
            "qiskit_ibm_runtime_mcp_server.ibm_runtime.initialize_service"
        ) as mock_init:
            mock_init.return_value = mock_runtime_service

            # Mock backend with a more connected coupling map (triangle + extra)
            mock_backend = Mock()
            mock_backend.name = "ibm_test"
            mock_backend.num_qubits = 5

            # Create a coupling map with good connectivity:
            # 0 - 1 - 2
            # |   |
            # 3 - 4
            mock_config = Mock()
            mock_config.coupling_map = [
                [0, 1],
                [1, 0],
                [1, 2],
                [2, 1],
                [0, 3],
                [3, 0],
                [1, 4],
                [4, 1],
                [3, 4],
                [4, 3],
            ]
            mock_backend.configuration.return_value = mock_config

            # Mock properties with calibration data
            mock_properties = Mock()
            mock_properties.faulty_qubits.return_value = []
            mock_properties.t1.return_value = 100e-6
            mock_properties.t2.return_value = 50e-6
            mock_properties.readout_error.return_value = 0.01
            mock_properties.gate_error.return_value = 0.005
            mock_backend.properties.return_value = mock_properties

            mock_runtime_service.backend.return_value = mock_backend

            result = await find_optimal_qv_qubits(
                "ibm_test", num_qubits=3, num_results=3
            )

            assert result["status"] == "success"
            assert result["backend_name"] == "ibm_test"
            assert result["num_qubits"] == 3
            assert result["metric"] == "qv_optimized"
            assert result["total_subgraphs_found"] > 0
            assert len(result["subgraphs"]) <= 3
            assert result["subgraphs"][0]["rank"] == 1
            assert len(result["subgraphs"][0]["qubits"]) == 3
            assert "internal_edges" in result["subgraphs"][0]
            assert "connectivity_ratio" in result["subgraphs"][0]
            assert "average_path_length" in result["subgraphs"][0]
            assert "qubit_details" in result["subgraphs"][0]
            assert "edge_errors" in result["subgraphs"][0]

    @pytest.mark.asyncio
    async def test_find_optimal_qv_qubits_no_valid_subgraphs(
        self, mock_env_vars, mock_runtime_service
    ):
        """Test when no valid subgraphs exist."""
        from qiskit_ibm_runtime_mcp_server.ibm_runtime import find_optimal_qv_qubits

        with patch(
            "qiskit_ibm_runtime_mcp_server.ibm_runtime.initialize_service"
        ) as mock_init:
            mock_init.return_value = mock_runtime_service

            # Mock backend with disconnected qubits
            mock_backend = Mock()
            mock_backend.name = "ibm_test"
            mock_backend.num_qubits = 3

            mock_config = Mock()
            mock_config.coupling_map = []  # No connections
            mock_backend.configuration.return_value = mock_config

            mock_properties = Mock()
            mock_properties.faulty_qubits.return_value = []
            # Mock calibration methods for sorting
            mock_properties.readout_error.return_value = 0.01
            mock_properties.t1.return_value = 100e-6
            mock_properties.t2.return_value = 50e-6
            mock_properties.frequency.return_value = 5.0e9
            mock_properties.gate_error.return_value = 0.01
            mock_backend.properties.return_value = mock_properties

            mock_runtime_service.backend.return_value = mock_backend

            result = await find_optimal_qv_qubits("ibm_test", num_qubits=3)

            assert result["status"] == "error"
            assert "No valid connected subgraphs" in result["message"]

    @pytest.mark.asyncio
    async def test_find_optimal_qv_qubits_excludes_faulty_qubits(
        self, mock_env_vars, mock_runtime_service
    ):
        """Test that faulty qubits are excluded from subgraphs."""
        from qiskit_ibm_runtime_mcp_server.ibm_runtime import find_optimal_qv_qubits

        with patch(
            "qiskit_ibm_runtime_mcp_server.ibm_runtime.initialize_service"
        ) as mock_init:
            mock_init.return_value = mock_runtime_service

            mock_backend = Mock()
            mock_backend.name = "ibm_test"
            mock_backend.num_qubits = 5

            # Well-connected topology
            mock_config = Mock()
            mock_config.coupling_map = [
                [0, 1],
                [1, 0],
                [1, 2],
                [2, 1],
                [0, 3],
                [3, 0],
                [1, 4],
                [4, 1],
                [3, 4],
                [4, 3],
            ]
            mock_backend.configuration.return_value = mock_config

            mock_properties = Mock()
            mock_properties.faulty_qubits.return_value = [1]  # Qubit 1 is faulty
            mock_properties.t1.return_value = 100e-6
            mock_properties.t2.return_value = 50e-6
            mock_properties.readout_error.return_value = 0.01
            mock_properties.gate_error.return_value = 0.005
            mock_backend.properties.return_value = mock_properties

            mock_runtime_service.backend.return_value = mock_backend

            result = await find_optimal_qv_qubits("ibm_test", num_qubits=3)

            assert result["status"] == "success"
            assert 1 in result["faulty_qubits"]
            # Verify no subgraph contains the faulty qubit
            for subgraph in result["subgraphs"]:
                assert 1 not in subgraph["qubits"]

    @pytest.mark.asyncio
    async def test_find_optimal_qv_qubits_different_metrics(
        self, mock_env_vars, mock_runtime_service
    ):
        """Test different scoring metrics."""
        from qiskit_ibm_runtime_mcp_server.ibm_runtime import find_optimal_qv_qubits

        with patch(
            "qiskit_ibm_runtime_mcp_server.ibm_runtime.initialize_service"
        ) as mock_init:
            mock_init.return_value = mock_runtime_service

            mock_backend = Mock()
            mock_backend.name = "ibm_test"
            mock_backend.num_qubits = 5

            mock_config = Mock()
            mock_config.coupling_map = [
                [0, 1],
                [1, 0],
                [1, 2],
                [2, 1],
                [0, 3],
                [3, 0],
                [1, 4],
                [4, 1],
                [3, 4],
                [4, 3],
            ]
            mock_backend.configuration.return_value = mock_config

            mock_properties = Mock()
            mock_properties.faulty_qubits.return_value = []
            mock_properties.t1.return_value = 100e-6
            mock_properties.t2.return_value = 50e-6
            mock_properties.readout_error.return_value = 0.01
            mock_properties.gate_error.return_value = 0.005
            mock_backend.properties.return_value = mock_properties

            mock_runtime_service.backend.return_value = mock_backend

            # Test each metric
            for metric in ["qv_optimized", "connectivity", "gate_error"]:
                result = await find_optimal_qv_qubits(
                    "ibm_test", num_qubits=3, metric=metric
                )

                assert result["status"] == "success"
                assert result["metric"] == metric
                assert len(result["subgraphs"]) > 0

    @pytest.mark.asyncio
    async def test_find_optimal_qv_qubits_connectivity_metrics(
        self, mock_env_vars, mock_runtime_service
    ):
        """Test that connectivity metrics are computed correctly."""
        from qiskit_ibm_runtime_mcp_server.ibm_runtime import find_optimal_qv_qubits

        with patch(
            "qiskit_ibm_runtime_mcp_server.ibm_runtime.initialize_service"
        ) as mock_init:
            mock_init.return_value = mock_runtime_service

            mock_backend = Mock()
            mock_backend.name = "ibm_test"
            mock_backend.num_qubits = 4

            # Create a fully connected 3-qubit subgraph (triangle: 0-1-2-0)
            # plus qubit 3 connected only to 0
            mock_config = Mock()
            mock_config.coupling_map = [
                [0, 1],
                [1, 0],
                [1, 2],
                [2, 1],
                [0, 2],
                [2, 0],  # Triangle complete
                [0, 3],
                [3, 0],  # Extra qubit
            ]
            mock_backend.configuration.return_value = mock_config

            mock_properties = Mock()
            mock_properties.faulty_qubits.return_value = []
            mock_properties.t1.return_value = 100e-6
            mock_properties.t2.return_value = 50e-6
            mock_properties.readout_error.return_value = 0.01
            mock_properties.gate_error.return_value = 0.005
            mock_backend.properties.return_value = mock_properties

            mock_runtime_service.backend.return_value = mock_backend

            result = await find_optimal_qv_qubits(
                "ibm_test", num_qubits=3, metric="connectivity"
            )

            assert result["status"] == "success"
            # The triangle {0, 1, 2} should be among the top results
            # It has max_possible_edges = 3, internal_edges = 3, connectivity_ratio = 1.0
            found_triangle = False
            for subgraph in result["subgraphs"]:
                if set(subgraph["qubits"]) == {0, 1, 2}:
                    found_triangle = True
                    assert subgraph["internal_edges"] == 3
                    assert subgraph["max_possible_edges"] == 3
                    assert subgraph["connectivity_ratio"] == 1.0
                    assert subgraph["average_path_length"] == 1.0
                    break
            assert found_triangle, "Triangle subgraph {0, 1, 2} should be found"

    @pytest.mark.asyncio
    async def test_find_optimal_qv_qubits_fake_backend_error(self, mock_env_vars):
        """Test that fake backends return an error."""
        from qiskit_ibm_runtime_mcp_server.ibm_runtime import find_optimal_qv_qubits

        result = await find_optimal_qv_qubits("fake_brisbane", num_qubits=3)

        assert result["status"] == "error"
        assert "not supported" in result["message"]


class TestResourceIntegration:
    """Test MCP resource integration."""

    @pytest.mark.asyncio
    async def test_service_status_resource(self, mock_env_vars, mock_runtime_service):
        """Test ibm_quantum://status resource."""
        from qiskit_ibm_runtime_mcp_server.ibm_runtime import get_service_status

        with patch(
            "qiskit_ibm_runtime_mcp_server.ibm_runtime.initialize_service"
        ) as mock_init:
            mock_init.return_value = mock_runtime_service

            result = await get_service_status()

            assert "IBM Quantum Service Status" in result
            assert "connected" in result.lower()


class TestErrorHandlingIntegration:
    """Test error handling in integration scenarios."""

    @pytest.mark.asyncio
    async def test_authentication_failure_recovery(self, mock_env_vars):
        """Test recovery from authentication failures."""
        from qiskit_ibm_runtime_mcp_server.ibm_runtime import setup_ibm_quantum_account

        # First call fails with authentication error
        with patch(
            "qiskit_ibm_runtime_mcp_server.ibm_runtime.initialize_service"
        ) as mock_init:
            mock_init.side_effect = [
                ValueError("Invalid token"),
                Mock(),  # Second call succeeds
            ]

            # First attempt should fail
            result1 = await setup_ibm_quantum_account("invalid_token")
            assert result1["status"] == "error"

            # Reset the mock for second attempt
            mock_init.side_effect = None
            mock_init.return_value = Mock()

            # Second attempt should succeed
            result2 = await setup_ibm_quantum_account("valid_token")
            assert result2["status"] == "success"

    @pytest.mark.asyncio
    async def test_service_unavailable_handling(self, mock_env_vars):
        """Test handling when quantum service is unavailable."""
        from qiskit_ibm_runtime_mcp_server.ibm_runtime import list_backends

        with patch(
            "qiskit_ibm_runtime_mcp_server.ibm_runtime.initialize_service"
        ) as mock_init:
            mock_init.side_effect = Exception("Service unavailable")

            result = await list_backends()

            assert result["status"] == "error"
            assert "Failed to list backends" in result["message"]

    @pytest.mark.asyncio
    async def test_network_connectivity_issues(
        self, mock_env_vars, mock_runtime_service
    ):
        """Test handling of network connectivity issues."""
        from qiskit_ibm_runtime_mcp_server.ibm_runtime import get_backend_properties

        with patch(
            "qiskit_ibm_runtime_mcp_server.ibm_runtime.initialize_service"
        ) as mock_init:
            mock_init.return_value = mock_runtime_service
            mock_runtime_service.backend.side_effect = Exception("Network timeout")

            result = await get_backend_properties("ibm_brisbane")

            assert result["status"] == "error"
            assert "Failed to get backend properties" in result["message"]


class TestEndToEndScenarios:
    """Test end-to-end scenarios."""

    @pytest.mark.asyncio
    async def test_complete_backend_exploration(
        self, mock_env_vars, mock_runtime_service
    ):
        """Test complete backend exploration scenario."""
        from qiskit_ibm_runtime_mcp_server.ibm_runtime import (
            get_backend_properties,
            get_service_status,
            least_busy_backend,
            list_backends,
            setup_ibm_quantum_account,
        )

        with (
            patch(
                "qiskit_ibm_runtime_mcp_server.ibm_runtime.initialize_service"
            ) as mock_init,
            patch(
                "qiskit_ibm_runtime_mcp_server.ibm_runtime.least_busy"
            ) as mock_least_busy,
        ):
            mock_init.return_value = mock_runtime_service

            # Mock least busy backend
            mock_backend = Mock()
            mock_backend.name = "ibm_brisbane"
            mock_backend.num_qubits = 127
            mock_backend.status.return_value = Mock(
                operational=True, pending_jobs=2, status_msg="active"
            )
            mock_least_busy.return_value = mock_backend

            # 1. Setup account
            setup_result = await setup_ibm_quantum_account("test_token")
            assert setup_result["status"] == "success"

            # 2. Check service status
            status_result = await get_service_status()
            assert "connected" in status_result.lower()

            # 3. List all backends
            backends_result = await list_backends()
            assert backends_result["status"] == "success"

            # 4. Find least busy backend
            least_busy_result = await least_busy_backend()
            assert least_busy_result["status"] == "success"

            # 5. Get detailed properties of recommended backend
            properties_result = await get_backend_properties(
                least_busy_result["backend_name"]
            )
            assert properties_result["status"] == "success"

    @pytest.mark.asyncio
    async def test_job_monitoring_scenario(self, mock_env_vars, mock_runtime_service):
        """Test job monitoring scenario."""
        from qiskit_ibm_runtime_mcp_server.ibm_runtime import (
            get_job_status,
            list_my_jobs,
        )

        with patch(
            "qiskit_ibm_runtime_mcp_server.ibm_runtime.service", mock_runtime_service
        ):
            # 1. List recent jobs
            jobs_result = await list_my_jobs(10)
            assert jobs_result["status"] == "success"

            # 2. Monitor each job's status
            for job in jobs_result["jobs"]:
                status_result = await get_job_status(job["job_id"])
                assert status_result["status"] == "success"
                assert status_result["job_id"] == job["job_id"]


# Assisted by watsonx Code Assistant
