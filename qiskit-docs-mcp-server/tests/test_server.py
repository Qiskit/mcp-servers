# This code is part of Qiskit.
#
# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for MCP server registration and configuration."""

from qiskit_docs_mcp_server.server import mcp


class TestServerRegistration:
    """Test that tools and resources are registered on the MCP server."""

    def test_server_name(self):
        """Test the MCP server has the correct name."""
        assert mcp.name == "Qiskit Documentation"

    def test_tools_registered(self):
        """Test that all expected tools are registered."""
        tool_names = {tool.name for tool in mcp._tool_manager._tools.values()}
        expected_tools = {
            "get_sdk_module_docs_tool",
            "get_addon_docs_tool",
            "get_guide_tool",
            "search_docs_tool",
            "lookup_error_code_tool",
            "cache_status_tool",
        }
        assert expected_tools.issubset(tool_names), f"Missing tools: {expected_tools - tool_names}"

    def test_resources_registered(self):
        """Test that all expected resources are registered."""
        resource_uris = set(mcp._resource_manager._resources.keys())
        expected_resources = {
            "qiskit-docs://modules",
            "qiskit-docs://addons",
            "qiskit-docs://guides",
            "qiskit-docs://error-codes",
        }
        assert expected_resources.issubset(resource_uris), (
            f"Missing resources: {expected_resources - resource_uris}"
        )

    def test_tool_count(self):
        """Test the expected number of tools."""
        assert len(mcp._tool_manager._tools) == 6

    def test_resource_count(self):
        """Test the expected number of resources."""
        assert len(mcp._resource_manager._resources) == 4
