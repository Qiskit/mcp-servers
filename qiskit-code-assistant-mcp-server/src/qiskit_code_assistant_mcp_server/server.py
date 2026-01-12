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

"""
Qiskit Code Assistant MCP Server

A Model Context Protocol server that provides access to IBM Qiskit Code Assistant
for intelligent quantum code completion and assistance.

Dependencies:
- fastmcp
- httpx
- python-dotenv
"""

import logging
from typing import Any

from fastmcp import FastMCP

from qiskit_code_assistant_mcp_server.constants import (
    QCA_MCP_DEBUG_LEVEL,
    validate_configuration,
)
from qiskit_code_assistant_mcp_server.qca import (
    accept_completion,
    accept_model_disclaimer,
    get_completion,
    get_model,
    get_model_disclaimer,
    get_rag_completion,
    get_service_status,
    list_models,
)
from qiskit_code_assistant_mcp_server.utils import close_http_client


# Configure logging
logging.basicConfig(level=getattr(logging, QCA_MCP_DEBUG_LEVEL, logging.INFO))
logger = logging.getLogger(__name__)

# Initialize FastMCP server
mcp = FastMCP("Qiskit Code Assistant")

logger.info("Qiskit Code Assistant MCP Server initialized")

# Validate configuration on startup
if not validate_configuration():
    logger.error("Configuration validation failed - server may not work correctly")
else:
    logger.info("Configuration validation passed")


##################################################
## MCP Resources
## - https://modelcontextprotocol.io/docs/concepts/resources
##################################################


@mcp.resource("qca://models", mime_type="application/json")
async def list_models_resource() -> dict[str, Any]:
    """List the available models from the Qiskit Code Assistant."""
    return await list_models()


@mcp.resource("qca://model/{model_id}", mime_type="application/json")
async def get_model_resource(model_id: str) -> dict[str, Any]:
    """Get the info for a specific model from the Qiskit Code Assistant.

    Args:
        model_id: The ID of the model to retrieve
    """
    return await get_model(model_id)


@mcp.resource("qca://disclaimer/{model_id}", mime_type="application/json")
async def get_model_disclaimer_resource(model_id: str) -> dict[str, Any]:
    """Get the disclaimer for a specific model from the Qiskit Code Assistant.

    Args:
        model_id: The ID of the model for which we want to retrieve the disclaimer
    """
    return await get_model_disclaimer(model_id)


@mcp.resource("qca://status", mime_type="text/plain")
async def get_service_status_resource() -> str:
    """Get current Qiskit Code Assistant service status."""
    return await get_service_status()


##################################################
## MCP Tools
## - https://modelcontextprotocol.io/docs/concepts/tools
##################################################


@mcp.tool()
async def accept_model_disclaimer_tool(model_id: str, disclaimer_id: str) -> dict[str, Any]:
    """Accept the legal disclaimer for a Qiskit Code Assistant model.

    Call this when prompted with a disclaimer requirement before using completions.
    The model_id and disclaimer_id are provided in the disclaimer prompt.

    Args:
        model_id: The model ID (e.g., 'mistral-small-3.2-24b-qiskit')
        disclaimer_id: The disclaimer ID from the disclaimer prompt

    Returns:
        Dict with 'status' ('success' or 'error') and 'result' or 'message'.
    """
    return await accept_model_disclaimer(model_id, disclaimer_id)


@mcp.tool()
async def get_completion_tool(prompt: str) -> dict[str, Any]:
    """Generate Qiskit Python code based on a natural language prompt.

    Use this tool when users want to CREATE CODE for quantum circuits, algorithms,
    or any Qiskit programming task. The tool returns ready-to-use Python code.

    When to use this tool:
    - "Write a Bell state circuit"
    - "Create a 3-qubit GHZ state"
    - "Generate VQE code for H2 molecule"
    - "Make a quantum random number generator"

    For conceptual questions (what/why/how explanations), use get_rag_completion_tool instead.

    Args:
        prompt: Natural language description of the code to generate.
                Be specific about circuit size, gates, and desired functionality.

    Returns:
        Dict with:
        - status: 'success' or 'error'
        - code: The generated Python/Qiskit code (DISPLAY THIS TO THE USER)
        - completion_id: ID for tracking (use with accept_completion_tool if user likes it)
        - message: Error description if status is 'error'

    Example: prompt="Create a Bell state circuit" returns code like:
        from qiskit import QuantumCircuit
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
    """
    return await get_completion(prompt)


@mcp.tool()
async def get_rag_completion_tool(prompt: str) -> dict[str, Any]:
    """Answer questions about Qiskit and quantum computing concepts using RAG.

    Use this tool when users ask CONCEPTUAL QUESTIONS that need explanations,
    not code. The tool retrieves relevant information from IBM's knowledge base.

    When to use this tool:
    - "What is quantum entanglement?"
    - "How does the Qiskit transpiler work?"
    - "Explain the difference between Sampler and Estimator"
    - "What are the best practices for error mitigation?"

    For code generation requests, use get_completion_tool instead.

    Args:
        prompt: A question about quantum computing or Qiskit concepts.

    Returns:
        Dict with:
        - status: 'success' or 'error'
        - answer: The explanatory text response (DISPLAY THIS TO THE USER)
        - completion_id: ID for tracking
        - message: Error description if status is 'error'
    """
    return await get_rag_completion(prompt)


@mcp.tool()
async def accept_completion_tool(completion_id: str) -> dict[str, Any]:
    """Mark a generated completion as accepted/useful for feedback tracking.

    Call this after successfully using code or answers from get_completion_tool
    or get_rag_completion_tool. This helps improve the Qiskit Code Assistant.

    Args:
        completion_id: The completion_id from a previous completion response

    Returns:
        Dict with 'status' ('success' or 'error') and 'result' or 'message'.
    """
    return await accept_completion(completion_id)


if __name__ == "__main__":
    import atexit

    logger.info("Starting Qiskit Code Assistant MCP Server")

    # Register cleanup function
    def cleanup() -> None:
        import asyncio

        try:
            asyncio.run(close_http_client())
            logger.info("HTTP client closed successfully")
        except Exception as e:
            logger.error(f"Error closing HTTP client: {e}")

    atexit.register(cleanup)

    try:
        mcp.run(transport="stdio")
    except KeyboardInterrupt:
        logger.info("Server interrupted, shutting down...")
    finally:
        cleanup()


# Assisted by watsonx Code Assistant
