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

from qiskit_gym_mcp_server.server import mcp


class TestServerRegistration:
    """Test that tools, resources, prompts, and templates are registered."""

    def test_server_name(self):
        """Test the server name is correct."""
        assert mcp.name == "Qiskit Gym"

    def test_resources_registered(self):
        """Test that all expected static resources are registered."""
        resource_uris = set(mcp._resource_manager._resources.keys())
        expected_resources = {
            "qiskit-gym://presets/coupling-maps",
            "qiskit-gym://algorithms",
            "qiskit-gym://policies",
            "qiskit-gym://environments",
            "qiskit-gym://training/sessions",
            "qiskit-gym://models",
            "qiskit-gym://server/config",
            "qiskit-gym://workflows",
        }
        assert expected_resources.issubset(resource_uris), (
            f"Missing resources: {expected_resources - resource_uris}"
        )

    def test_resource_count(self):
        """Test the expected number of static resources."""
        assert len(mcp._resource_manager._resources) == 8

    def test_prompts_registered(self):
        """Test that all expected prompts are registered."""
        prompt_names = set(mcp._prompt_manager._prompts.keys())
        expected_prompts = {
            "train_synthesis_model",
            "synthesize_circuit",
            "explore_hardware_topology",
        }
        assert expected_prompts.issubset(prompt_names), (
            f"Missing prompts: {expected_prompts - prompt_names}"
        )

    def test_prompt_count(self):
        """Test the expected number of prompts."""
        assert len(mcp._prompt_manager._prompts) == 3

    def test_resource_templates_registered(self):
        """Test that all expected resource templates are registered."""
        template_uris = set(mcp._resource_manager._templates.keys())
        expected_templates = {
            "qiskit-gym://environments/{env_id}",
            "qiskit-gym://models/{model_name}",
            "qiskit-gym://training/{session_id}",
        }
        assert expected_templates.issubset(template_uris), (
            f"Missing resource templates: {expected_templates - template_uris}"
        )

    def test_resource_template_count(self):
        """Test the expected number of resource templates."""
        assert len(mcp._resource_manager._templates) == 3
