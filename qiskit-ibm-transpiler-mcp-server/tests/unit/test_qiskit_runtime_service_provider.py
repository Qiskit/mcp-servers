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
from unittest.mock import MagicMock

import pytest
from qiskit_ibm_transpiler_mcp_server.qiskit_runtime_service_provider import (
    QiskitRuntimeServiceProvider,
)


class TestQiskitRuntimeServiceProvider:
    """Test Qiskit Runtime Service Provider Singleton class."""

    def test_single_instance(self):
        """Test two different initializations correspond to the same instance"""
        qsp1 = QiskitRuntimeServiceProvider()
        qsp2 = QiskitRuntimeServiceProvider()
        assert qsp1 is qsp2, "Singleton should always return the same instance"

    def test_service_caching_with_same_token(self, mocker):
        """Verify that get() caches the service when the same token is used."""
        dummy_service = MagicMock()
        initialize_service_mock = mocker.patch(
            "qiskit_ibm_transpiler_mcp_server.qiskit_runtime_service_provider.QiskitRuntimeServiceProvider._initialize_service",
            return_value=dummy_service,
        )

        provider = QiskitRuntimeServiceProvider()
        service1 = provider.get(token="dummy_token")
        service2 = provider.get(token="dummy_token")

        assert service1 is service2
        # _initialize_service called only once for same token
        initialize_service_mock.assert_called_once_with(
            token="dummy_token", channel="ibm_quantum_platform"
        )

    def test_service_reinit_on_token_change(self, mocker):
        """Verify that _service is reinitialized if token changes."""
        dummy_service1 = MagicMock()
        dummy_service2 = MagicMock()

        initialize_service_mock = mocker.patch(
            "qiskit_ibm_transpiler_mcp_server.qiskit_runtime_service_provider.QiskitRuntimeServiceProvider._initialize_service",
            side_effect=[dummy_service1, dummy_service2],
        )

        provider = QiskitRuntimeServiceProvider()
        service1 = provider.get(token="dummy_token_a")
        service2 = provider.get(token="dummy_token_b")

        assert service1 is dummy_service1
        assert service2 is dummy_service2
        assert initialize_service_mock.call_count == 2
        initialize_service_mock.assert_any_call(
            token="dummy_token_a", channel="ibm_quantum_platform"
        )
        initialize_service_mock.assert_any_call(
            token="dummy_token_b", channel="ibm_quantum_platform"
        )

    def test_service_reinit_on_token_none(self, mocker):
        """Verify that _service is reinitialized if token changes (to None)."""
        dummy_service_1 = MagicMock()
        dummy_service_2 = MagicMock()

        initialize_service_mock = mocker.patch(
            "qiskit_ibm_transpiler_mcp_server.qiskit_runtime_service_provider.QiskitRuntimeServiceProvider._initialize_service",
            side_effect=[dummy_service_1, dummy_service_2],
        )

        qsp = QiskitRuntimeServiceProvider()

        service_with_token = qsp.get(token="dummy")
        service_without_token = qsp.get()

        assert service_with_token is dummy_service_1
        assert service_without_token is dummy_service_2
        assert service_with_token is not service_without_token

        assert initialize_service_mock.call_count == 2
        initialize_service_mock.assert_any_call(token="dummy", channel="ibm_quantum_platform")
        initialize_service_mock.assert_any_call(token=None, channel="ibm_quantum_platform")

    def test_service_reinit_on_channel_change(self, mocker):
        """Verify that _service is reinitialized if channel changes."""
        dummy_service1 = MagicMock()
        dummy_service2 = MagicMock()

        initialize_service_mock = mocker.patch(
            "qiskit_ibm_transpiler_mcp_server.qiskit_runtime_service_provider.QiskitRuntimeServiceProvider._initialize_service",
            side_effect=[dummy_service1, dummy_service2],
        )

        provider = QiskitRuntimeServiceProvider()
        service1 = provider.get(token="dummy_token", channel="channel1")
        service2 = provider.get(token="dummy_token", channel="channel2")

        assert service1 is dummy_service1
        assert service2 is dummy_service2
        assert initialize_service_mock.call_count == 2
        initialize_service_mock.assert_any_call(token="dummy_token", channel="channel1")
        initialize_service_mock.assert_any_call(token="dummy_token", channel="channel2")

    def test_invalid_token_does_not_overwrite_service(self, mocker):
        """Verify that invalid token does not overwrite the valid one"""
        dummy_service = MagicMock()

        def side_effect(token=None, channel="ibm_quantum_platform"):
            if token == "<TOKEN>":
                raise ValueError("Invalid token")
            return dummy_service

        mocker.patch(
            "qiskit_ibm_transpiler_mcp_server.qiskit_runtime_service_provider.QiskitRuntimeServiceProvider._initialize_service",
            side_effect=side_effect,
        )

        qrs = QiskitRuntimeServiceProvider()

        service1 = qrs.get(token="valid_token")
        assert qrs._cached_token == "valid_token"

        with pytest.raises(ValueError):
            qrs.get(token="<TOKEN>")

        assert qrs._service is service1
        assert qrs._cached_token == "valid_token"


class TestInitializeService:
    """Test service initialization function."""

    def test_initialize_service_existing_account(self, mocker, mock_runtime_service):
        """Test initialization with existing account."""
        qiskit_runtime_service_mock = mocker.patch(
            "qiskit_ibm_transpiler_mcp_server.qiskit_runtime_service_provider.QiskitRuntimeService",
            autospec=True,
        )

        qiskit_runtime_service_mock.return_value = mock_runtime_service

        qsp = QiskitRuntimeServiceProvider()
        service = qsp._initialize_service()

        assert service == mock_runtime_service
        qiskit_runtime_service_mock.assert_called_once_with(channel="ibm_quantum_platform")
        qiskit_runtime_service_mock.save_account.assert_not_called()

    def test_initialize_service_with_token(self, mocker, mock_runtime_service):
        """Test initialization with provided token."""
        qiskit_runtime_service_mock = mocker.patch(
            "qiskit_ibm_transpiler_mcp_server.qiskit_runtime_service_provider.QiskitRuntimeService",
            autospec=True,
        )
        qiskit_runtime_service_mock.return_value = mock_runtime_service

        qsp = QiskitRuntimeServiceProvider()
        service = qsp._initialize_service(token="test_token", channel="ibm_quantum_platform")

        assert service == mock_runtime_service
        qiskit_runtime_service_mock.save_account.assert_called_once_with(
            channel="ibm_quantum_platform", token="test_token", overwrite=True
        )

    def test_initialize_service_with_env_token(self, mocker, mock_runtime_service, mock_env_vars):
        """Test initialization with environment token."""
        qiskit_runtime_service_mock = mocker.patch(
            "qiskit_ibm_transpiler_mcp_server.qiskit_runtime_service_provider.QiskitRuntimeService",
            autospec=True,
        )
        qiskit_runtime_service_mock.return_value = mock_runtime_service

        qsp = QiskitRuntimeServiceProvider()
        service = qsp._initialize_service()

        assert service == mock_runtime_service

    def test_initialize_service_no_token_available(self, mocker):
        """Test initialization failure when no token is available."""
        qiskit_runtime_service_mock = mocker.patch(
            "qiskit_ibm_transpiler_mcp_server.qiskit_runtime_service_provider.QiskitRuntimeService",
            autospec=True,
        )
        qiskit_runtime_service_mock.side_effect = Exception("No account")

        qsp = QiskitRuntimeServiceProvider()

        with pytest.raises(ValueError) as exc_info:
            qsp._initialize_service()

        assert "No IBM Quantum token provided" in str(exc_info.value)

    def test_initialize_service_invalid_token(self, mocker):
        """Test initialization with invalid token."""
        qiskit_runtime_service_mock = mocker.patch(
            "qiskit_ibm_transpiler_mcp_server.qiskit_runtime_service_provider.QiskitRuntimeService",
            autospec=True,
        )
        qiskit_runtime_service_mock.side_effect = Exception("No account")
        qiskit_runtime_service_mock.save_account.side_effect = Exception("Invalid token")

        qsp = QiskitRuntimeServiceProvider()
        with pytest.raises(Exception) as exc_info:
            qsp._initialize_service(token="invalid_token")

        assert "Invalid token" in str(exc_info.value)

    def test_initialize_token_whitespace(self, mocker, mock_runtime_service):
        """Test initialization with empty token."""
        qrs_mock = mocker.patch(
            "qiskit_ibm_transpiler_mcp_server.qiskit_runtime_service_provider.QiskitRuntimeService",
            autospec=True,
        )
        qrs_mock.return_value = mock_runtime_service

        qsp = QiskitRuntimeServiceProvider()
        service = qsp._initialize_service(token="   ")

        assert service == mock_runtime_service
        qrs_mock.save_account.assert_not_called()

    def test_initialize_service_placeholder_token(self):
        """Test that placeholder tokens are rejected."""

        qsp = QiskitRuntimeServiceProvider()
        with pytest.raises(ValueError) as exc_info:
            qsp._initialize_service(token="<PASSWORD>")

        assert "appears to be a placeholder value" in str(exc_info.value)

    def test_initialize_service_prioritizes_saved_credentials(self, mocker, mock_runtime_service):
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
