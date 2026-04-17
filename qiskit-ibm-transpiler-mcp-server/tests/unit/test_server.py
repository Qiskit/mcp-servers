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

from qiskit_ibm_transpiler_mcp_server.server import mcp


class TestServerRegistration:
    """Test that tools, resources, and prompts are registered on the MCP server."""

    def test_server_name(self):
        """Test the server name is correct."""
        assert mcp.name == "Qiskit IBM Transpiler"

    def test_tools_registered(self):
        """Test that all expected tools are registered."""
        tool_names = {tool.name for tool in mcp._tool_manager._tools.values()}
        expected_tools = {
            "setup_ibm_quantum_account_tool",
            "ai_routing_tool",
            "ai_linear_function_synthesis_tool",
            "ai_clifford_synthesis_tool",
            "ai_permutation_synthesis_tool",
            "ai_pauli_network_synthesis_tool",
            "hybrid_ai_transpile_tool",
        }
        assert expected_tools.issubset(tool_names), f"Missing tools: {expected_tools - tool_names}"

    def test_tool_count(self):
        """Test the expected number of tools."""
        assert len(mcp._tool_manager._tools) == 7

    def test_resources_registered(self):
        """Test that all expected resources are registered."""
        resource_uris = set(mcp._resource_manager._resources.keys())
        expected_resources = {
            "qiskit-ibm-transpiler://info",
            "qiskit-ibm-transpiler://synthesis-types",
        }
        assert expected_resources.issubset(resource_uris), (
            f"Missing resources: {expected_resources - resource_uris}"
        )

    def test_resource_count(self):
        """Test the expected number of resources."""
        assert len(mcp._resource_manager._resources) == 2

    def test_prompts_registered(self):
        """Test that all expected prompts are registered."""
        prompt_names = set(mcp._prompt_manager._prompts.keys())
        expected_prompts = {
            "transpile_circuit",
            "optimize_circuit",
            "explain_synthesis_type",
        }
        assert expected_prompts.issubset(prompt_names), (
            f"Missing prompts: {expected_prompts - prompt_names}"
        )

    def test_prompt_count(self):
        """Test the expected number of prompts."""
        assert len(mcp._prompt_manager._prompts) == 3
