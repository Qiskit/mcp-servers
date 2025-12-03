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

#!/usr/bin/env python3
"""
Qiskit IBM Runtime MCP Server

A Model Context Protocol server that provides access to IBM Quantum services
through Qiskit IBM Runtime, enabling AI assistants to interact with quantum
computing resources.

Dependencies:
- fastmcp
- qiskit-ibm-runtime
- qiskit
- python-dotenv
"""

import logging
from typing import Any

from fastmcp import FastMCP

from qiskit_ibm_runtime_mcp_server.ibm_runtime import (
    cancel_job,
    get_backend_properties,
    get_job_status,
    get_service_status,
    least_busy_backend,
    list_backends,
    list_my_jobs,
    setup_ibm_quantum_account,
)


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize MCP server
mcp = FastMCP("Qiskit IBM Runtime")


# Tools
@mcp.tool()
async def setup_ibm_quantum_account_tool(
    token: str = "", channel: str = "ibm_quantum_platform"
) -> dict[str, Any]:
    """Set up IBM Quantum account with credentials.

    If token is not provided, will attempt to use QISKIT_IBM_TOKEN environment variable
    or saved credentials from ~/.qiskit/qiskit-ibm.json
    """
    return await setup_ibm_quantum_account(token if token else None, channel)


@mcp.tool()
async def list_backends_tool() -> dict[str, Any]:
    """List available IBM Quantum backends."""
    return await list_backends()


@mcp.tool()
async def least_busy_backend_tool() -> dict[str, Any]:
    """Find the least busy operational backend."""
    return await least_busy_backend()


@mcp.tool()
async def get_backend_properties_tool(backend_name: str) -> dict[str, Any]:
    """Get detailed properties of a specific backend."""
    return await get_backend_properties(backend_name)


@mcp.tool()
async def list_my_jobs_tool(limit: int = 10) -> dict[str, Any]:
    """List user's recent jobs."""
    return await list_my_jobs(limit)


@mcp.tool()
async def get_job_status_tool(job_id: str) -> dict[str, Any]:
    """Get status of a specific job."""
    return await get_job_status(job_id)


@mcp.tool()
async def cancel_job_tool(job_id: str) -> dict[str, Any]:
    """Cancel a specific job."""
    return await cancel_job(job_id)


# Resources
@mcp.resource("ibm://status", mime_type="text/plain")
async def get_service_status_resource() -> str:
    """Get current IBM Quantum service status."""
    return await get_service_status()


def main() -> None:
    """Run the server."""
    mcp.run()


if __name__ == "__main__":
    main()


# Assisted by watsonx Code Assistant
