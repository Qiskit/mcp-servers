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

"""FastMCP application instance.

This module creates the shared MCP server instance used by tools and resources.
Separating this allows tools and resources to be defined in their own modules
without circular imports.
"""

from fastmcp import FastMCP


# Initialize MCP server - shared by all tools and resources
mcp = FastMCP("Qiskit Gym")
