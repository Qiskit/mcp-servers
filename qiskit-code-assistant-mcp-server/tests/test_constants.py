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

"""Unit tests for constants and configuration."""

from unittest.mock import patch
import os

from qiskit_code_assistant_mcp_server.constants import validate_configuration


class TestValidateConfiguration:
    """Test configuration validation function."""

    def test_validate_configuration_success(self, mock_env_vars):
        """Test successful configuration validation."""
        with patch(
            "qiskit_code_assistant_mcp_server.constants.QCA_TOOL_API_BASE",
            "https://valid-api.example.com",
        ):
            with patch(
                "qiskit_code_assistant_mcp_server.constants.QCA_TOOL_MODEL_NAME",
                "valid-model",
            ):
                result = validate_configuration()
                assert result is True

    def test_validate_configuration_invalid_api_base(self):
        """Test validation with invalid API base URL."""
        with patch(
            "qiskit_code_assistant_mcp_server.constants.QCA_TOOL_API_BASE",
            "invalid-url",
        ):
            with patch(
                "qiskit_code_assistant_mcp_server.constants.QCA_TOOL_MODEL_NAME",
                "valid-model",
            ):
                result = validate_configuration()
                assert result is False

    def test_validate_configuration_empty_model_name(self):
        """Test validation with empty model name."""
        with patch(
            "qiskit_code_assistant_mcp_server.constants.QCA_TOOL_API_BASE",
            "https://valid-api.example.com",
        ):
            with patch(
                "qiskit_code_assistant_mcp_server.constants.QCA_TOOL_MODEL_NAME", ""
            ):
                result = validate_configuration()
                assert result is False

    def test_validate_configuration_http_api_base(self):
        """Test validation with HTTP API base (should be valid)."""
        with patch(
            "qiskit_code_assistant_mcp_server.constants.QCA_TOOL_API_BASE",
            "http://test-api.example.com",
        ):
            with patch(
                "qiskit_code_assistant_mcp_server.constants.QCA_TOOL_MODEL_NAME",
                "valid-model",
            ):
                result = validate_configuration()
                assert result is True


class TestEnvironmentVariableHandling:
    """Test environment variable processing."""

    def test_request_timeout_valid(self):
        """Test valid request timeout configuration."""
        with patch.dict(os.environ, {"QCA_REQUEST_TIMEOUT": "45.5"}):
            # Re-import to get updated value
            import importlib
            import qiskit_code_assistant_mcp_server.constants as constants_module

            importlib.reload(constants_module)

            assert constants_module.QCA_REQUEST_TIMEOUT == 45.5

    def test_request_timeout_invalid_negative(self):
        """Test invalid negative timeout (should use default)."""
        with patch.dict(os.environ, {"QCA_REQUEST_TIMEOUT": "-5.0"}):
            import importlib
            import qiskit_code_assistant_mcp_server.constants as constants_module

            importlib.reload(constants_module)

            assert constants_module.QCA_REQUEST_TIMEOUT == 30.0

    def test_request_timeout_invalid_too_large(self):
        """Test invalid too large timeout (should use default)."""
        with patch.dict(os.environ, {"QCA_REQUEST_TIMEOUT": "400.0"}):
            import importlib
            import qiskit_code_assistant_mcp_server.constants as constants_module

            importlib.reload(constants_module)

            assert constants_module.QCA_REQUEST_TIMEOUT == 30.0

    def test_debug_level_valid(self):
        """Test valid debug level configuration."""
        with patch.dict(os.environ, {"QCA_MCP_DEBUG_LEVEL": "WARNING"}):
            import importlib
            import qiskit_code_assistant_mcp_server.constants as constants_module

            importlib.reload(constants_module)

            assert constants_module.QCA_MCP_DEBUG_LEVEL == "WARNING"

    def test_debug_level_invalid(self):
        """Test invalid debug level (should use default)."""
        with patch.dict(os.environ, {"QCA_MCP_DEBUG_LEVEL": "INVALID_LEVEL"}):
            import importlib
            import qiskit_code_assistant_mcp_server.constants as constants_module

            importlib.reload(constants_module)

            assert constants_module.QCA_MCP_DEBUG_LEVEL == "INFO"

    def test_api_base_trailing_slash_removal(self):
        """Test that trailing slashes are removed from API base."""
        with patch.dict(os.environ, {"QCA_TOOL_API_BASE": "https://api.example.com/"}):
            import importlib
            import qiskit_code_assistant_mcp_server.constants as constants_module

            importlib.reload(constants_module)

            assert not constants_module.QCA_TOOL_API_BASE.endswith("/")

    def test_default_values(self):
        """Test that default values are used when env vars are not set."""
        env_vars_to_remove = [
            "QCA_TOOL_API_BASE",
            "QCA_TOOL_MODEL_NAME",
            "QCA_REQUEST_TIMEOUT",
            "QCA_MCP_DEBUG_LEVEL",
        ]

        with patch.dict(os.environ, {}, clear=True):
            # Remove any existing env vars
            for var in env_vars_to_remove:
                if var in os.environ:
                    del os.environ[var]

            import importlib
            import qiskit_code_assistant_mcp_server.constants as constants_module

            importlib.reload(constants_module)

            assert (
                constants_module.QCA_TOOL_API_BASE
                == "https://qiskit-code-assistant.quantum.ibm.com"
            )
            assert constants_module.QCA_TOOL_MODEL_NAME == "granite-3.3-8b-qiskit"
            assert constants_module.QCA_REQUEST_TIMEOUT == 30.0
            assert constants_module.QCA_MCP_DEBUG_LEVEL == "INFO"


class TestVersionHandling:
    """Test package version handling."""

    def test_version_detection_success(self):
        """Test successful version detection."""
        with patch("importlib.metadata.version") as mock_version:
            mock_version.return_value = "1.2.3"

            import importlib
            import qiskit_code_assistant_mcp_server.constants as constants_module

            importlib.reload(constants_module)

            assert (
                "qiskit-code-assistant-mcp-server/1.2.3"
                in constants_module.QCA_TOOL_X_CALLER
            )

    def test_version_detection_failure(self):
        """Test version detection failure (should use unknown)."""
        with patch("importlib.metadata.version") as mock_version:
            mock_version.side_effect = Exception("Package not found")

            import importlib
            import qiskit_code_assistant_mcp_server.constants as constants_module

            importlib.reload(constants_module)

            assert (
                "qiskit-code-assistant-mcp-server/unknown"
                in constants_module.QCA_TOOL_X_CALLER
            )


# Assisted by watsonx Code Assistant
