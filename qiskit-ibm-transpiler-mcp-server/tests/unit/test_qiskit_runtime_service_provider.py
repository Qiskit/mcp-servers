from qiskit_ibm_transpiler_mcp_server.qiskit_runtime_service_provider import (
    QiskitRuntimeServiceProvider,
)

from unittest.mock import MagicMock
import pytest


class TestQiskitRuntimeServiceProvider:
    """Test Qiskit Runtime Service Provider Singleton class."""

    def test_single_instance(self):
        """Test two different initializations correspond to the same instance"""
        qsp1 = QiskitRuntimeServiceProvider()
        qsp2 = QiskitRuntimeServiceProvider()
        assert qsp1 is qsp2, "Singleton should always return the same instance"

    def test_get_success(self, mocker):
        """Test lazy initialization. Get is only called the first time."""
        dummy_service = MagicMock()
        initialize_service_mock = mocker.patch(
            "qiskit_ibm_transpiler_mcp_server.qiskit_runtime_service_provider.QiskitRuntimeServiceProvider._initialize_service",
            return_value=dummy_service,
        )

        qsp = QiskitRuntimeServiceProvider()
        qsp_service1 = qsp.get(token="dummy")
        qsp_service2 = qsp.get()

        assert qsp_service1 is qsp_service2
        initialize_service_mock.assert_called_once_with(
            token="dummy", channel="ibm_quantum_platform"
        )

    def test_singleton_token_usage(self, mocker):
        """Simulate initialize_service to verify the saved/passed token"""
        called = {}

        def dummy_initialize_service(token=None, channel="ibm_quantum_platform"):
            called["token"] = token
            called["channel"] = channel
            return "qiskit_runtime_service"

        initialize_service_mock = mocker.patch(
            "qiskit_ibm_transpiler_mcp_server.qiskit_runtime_service_provider.QiskitRuntimeServiceProvider._initialize_service",
            side_effect=dummy_initialize_service,
        )

        qsp1 = QiskitRuntimeServiceProvider()
        qsp1.get(token="dummy_token")
        qsp2 = QiskitRuntimeServiceProvider()
        qsp2.get()

        assert qsp2 is qsp1
        assert called["token"] == "dummy_token"
        initialize_service_mock.assert_called_once_with(
            token="dummy_token", channel="ibm_quantum_platform"
        )


class TestInitializeService:
    """Test service initialization function."""

    def test_initialize_service_existing_account(self, mocker, mock_runtime_service):
        """Test initialization with existing account."""
        qiskit_runtime_service_mock = mocker.patch(
            "qiskit_ibm_transpiler_mcp_server.qiskit_runtime_service_provider.QiskitRuntimeService"
        )

        qiskit_runtime_service_mock.return_value = mock_runtime_service

        qsp = QiskitRuntimeServiceProvider()
        service = qsp._initialize_service()

        assert service == mock_runtime_service
        qiskit_runtime_service_mock.assert_called_once_with(
            channel="ibm_quantum_platform"
        )

    def test_initialize_service_with_token(self, mocker, mock_runtime_service):
        """Test initialization with provided token."""
        qiskit_runtime_service_mock = mocker.patch(
            "qiskit_ibm_transpiler_mcp_server.qiskit_runtime_service_provider.QiskitRuntimeService"
        )
        qiskit_runtime_service_mock.return_value = mock_runtime_service

        qsp = QiskitRuntimeServiceProvider()
        service = qsp._initialize_service(
            token="test_token", channel="ibm_quantum_platform"
        )

        assert service == mock_runtime_service
        qiskit_runtime_service_mock.save_account.assert_called_once_with(
            channel="ibm_quantum_platform", token="test_token", overwrite=True
        )

    def test_initialize_service_with_env_token(
        self, mocker, mock_runtime_service, mock_env_vars
    ):
        """Test initialization with environment token."""
        qiskit_runtime_service_mock = mocker.patch(
            "qiskit_ibm_transpiler_mcp_server.qiskit_runtime_service_provider.QiskitRuntimeService"
        )
        qiskit_runtime_service_mock.return_value = mock_runtime_service

        qsp = QiskitRuntimeServiceProvider()
        service = qsp._initialize_service()

        assert service == mock_runtime_service

    def test_initialize_service_no_token_available(self, mocker):
        """Test initialization failure when no token is available."""
        qiskit_runtime_service_mock = mocker.patch(
            "qiskit_ibm_transpiler_mcp_server.qiskit_runtime_service_provider.QiskitRuntimeService"
        )
        qiskit_runtime_service_mock.side_effect = Exception("No account")

        qsp = QiskitRuntimeServiceProvider()

        with pytest.raises(ValueError) as exc_info:
            qsp._initialize_service()

        assert "No IBM Quantum token provided" in str(exc_info.value)

    def test_initialize_service_invalid_token(self, mocker):
        """Test initialization with invalid token."""
        qiskit_runtime_service_mock = mocker.patch(
            "qiskit_ibm_transpiler_mcp_server.qiskit_runtime_service_provider.QiskitRuntimeService"
        )
        qiskit_runtime_service_mock.side_effect = Exception("No account")
        qiskit_runtime_service_mock.save_account.side_effect = Exception(
            "Invalid token"
        )

        qsp = QiskitRuntimeServiceProvider()
        with pytest.raises(ValueError) as exc_info:
            qsp._initialize_service(token="invalid_token")

        assert "Invalid token or channel" in str(exc_info.value)

    def test_initialize_service_placeholder_token(self):
        """Test that placeholder tokens are rejected."""

        qsp = QiskitRuntimeServiceProvider()
        with pytest.raises(ValueError) as exc_info:
            qsp._initialize_service(token="<PASSWORD>")

        assert "appears to be a placeholder value" in str(exc_info.value)

    def test_initialize_service_prioritizes_saved_credentials(
        self, mocker, mock_runtime_service
    ):
        """Test that saved credentials are tried first when no token provided."""
        qiskit_runtime_service_mock = mocker.patch(
            "qiskit_ibm_transpiler_mcp_server.qiskit_runtime_service_provider.QiskitRuntimeService"
        )
        qiskit_runtime_service_mock.return_value = mock_runtime_service

        qsp = QiskitRuntimeServiceProvider()
        service = qsp._initialize_service()
        assert service == mock_runtime_service
        # Should NOT call save_account
        qiskit_runtime_service_mock.save_account.assert_not_called()
