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

import asyncio
import logging
from typing import Any

from qiskit_code_assistant_mcp_server.constants import (
    QCA_REQUEST_TIMEOUT,
    QCA_TOOL_API_BASE,
    QCA_TOOL_MODEL_NAME,
)
from qiskit_code_assistant_mcp_server.utils import (
    close_http_client,
    make_qca_request,
    with_sync,
)


logger = logging.getLogger(__name__)


@with_sync
async def list_models() -> dict[str, Any]:
    """List the available models from the Qiskit Code Assistant."""
    try:
        logger.info("Fetching available models from Qiskit Code Assistant")
        url = f"{QCA_TOOL_API_BASE}/v1/models"
        data = await make_qca_request(url, method="GET")

        if "error" in data:
            logger.error(f"Failed to list models: {data['error']}")
            return {"status": "error", "message": data["error"]}

        models = data.get("data", [])
        if len(models) <= 0:
            logger.warning("No models retrieved from Qiskit Code Assistant")
            return {"status": "error", "message": "No models retrieved."}
        else:
            logger.info(f"Retrieved {len(models)} models from Qiskit Code Assistant")
            return {"status": "success", "models": models}
    except Exception as e:
        logger.error(f"Exception in list_models: {e!s}")
        return {"status": "error", "message": f"Failed to list models: {e!s}"}


def _select_available_model() -> str:
    """
    Select an available model from the Qiskit Code Assistant service.

    This function checks if the configured default model is available.
    If not, it selects the first available model as a fallback.

    Returns:
        The model name to use for completions
    """
    try:
        # Run the async list_models function synchronously
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            models_result = loop.run_until_complete(list_models())
            # Close the HTTP client since it's attached to this loop which we are about to close
            loop.run_until_complete(close_http_client())
        finally:
            loop.close()

        if models_result.get("status") == "success":
            available_models = models_result.get("models", [])
            model_ids: list[str] = [
                model.get("id") for model in available_models if model.get("id")
            ]

            # Check if default model is available
            if QCA_TOOL_MODEL_NAME in model_ids:
                logger.info(f"Default model '{QCA_TOOL_MODEL_NAME}' is available")
                return QCA_TOOL_MODEL_NAME

            # Default model not available, use first available model
            if model_ids:
                selected_model = model_ids[0]
                logger.warning(
                    f"Default model '{QCA_TOOL_MODEL_NAME}' is not available. "
                    f"Using '{selected_model}' instead. "
                    f"Available models: {', '.join(model_ids)}"
                )
                return selected_model

            # No models available
            logger.error("No models available from Qiskit Code Assistant service")
        else:
            error_msg = models_result.get("message", "Unknown error")
            logger.error(f"Failed to fetch available models: {error_msg}")

    except Exception as e:
        logger.error(f"Exception while selecting available model: {e!s}")

    # Fallback to configured default if anything goes wrong
    logger.warning(
        f"Unable to verify model availability. Using configured default: {QCA_TOOL_MODEL_NAME}"
    )
    return QCA_TOOL_MODEL_NAME


# Select the model to use at module initialization
_SELECTED_MODEL_NAME = _select_available_model()
logger.info(f"Using model: {_SELECTED_MODEL_NAME}")


@with_sync
async def get_model(model_id: str) -> dict[str, Any]:
    """Get the info for a model from the Qiskit Code Assistant.

    Args:
        model_id: The ID of the model to retrieve
    """
    if not model_id or not model_id.strip():
        return {
            "status": "error",
            "message": "model_id is required and cannot be empty",
        }

    try:
        logger.info(f"Fetching model info for model_id: {model_id}")
        url = f"{QCA_TOOL_API_BASE}/v1/model/{model_id}"
        data = await make_qca_request(url, method="GET")

        if "error" in data:
            logger.error(f"Failed to get model {model_id}: {data['error']}")
            return {"status": "error", "message": data["error"]}
        elif "id" not in data:
            logger.warning(f"Model {model_id} not retrieved - missing ID in response")
            return {"status": "error", "message": "Model not retrieved."}
        else:
            logger.info(f"Successfully retrieved model {model_id}")
            return {"status": "success", "model": data}
    except Exception as e:
        logger.error(f"Exception in get_model for {model_id}: {e!s}")
        return {"status": "error", "message": f"Failed to get model: {e!s}"}


@with_sync
async def get_model_disclaimer(model_id: str) -> dict[str, Any]:
    """Get the disclaimer for a model from the Qiskit Code Assistant.

    Args:
        model_id: The ID of the model for which we want to retrieve the disclaimer
    """
    try:
        url = f"{QCA_TOOL_API_BASE}/v1/model/{model_id}/disclaimer"
        data = await make_qca_request(url, method="GET")

        if "error" in data:
            return {"status": "error", "message": data["error"]}
        elif "id" not in data:
            return {"status": "error", "message": "Model disclaimer not retrieved."}
        else:
            return {"status": "success", "disclaimer": data}
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to get model disclaimer: {e!s}",
        }


@with_sync
async def accept_model_disclaimer(model_id: str, disclaimer_id: str) -> dict[str, Any]:
    """
    Accept the disclaimer for an available model from the Qiskit Code Assistant.

    Args:
        model_id: The ID of the model for which we want to accept the disclaimer
        disclaimer_id: The ID of the disclaimer we want to accept

    Returns:
        Disclaimer acceptance status
    """
    try:
        url = f"{QCA_TOOL_API_BASE}/v1/model/{model_id}/disclaimer"
        body = {"disclaimer": disclaimer_id, "accepted": "true"}
        data = await make_qca_request(url, method="POST", body=body)

        if "error" in data:
            return {"status": "error", "message": data["error"]}
        elif "success" not in data:
            return {
                "status": "error",
                "message": "The response does not contain the acceptance result.",
            }
        else:
            return {"status": "success", "result": data}
    except Exception as e:
        return {"status": "error", "message": f"Failed to accept disclaimer: {e!s}"}


@with_sync
async def get_completion(prompt: str) -> dict[str, Any]:
    """
    Get code completion for writing, completing, and optimizing quantum code using Qiskit.

    This tool generates Qiskit Python code based on the given prompt. Use it when users
    want to create quantum circuits, algorithms, or any Qiskit-related code.

    Args:
        prompt: Natural language description of what code to generate. Be specific about:
                - What quantum circuit or algorithm to create
                - Number of qubits if relevant
                - Any specific gates or operations needed

    Returns:
        Dictionary with:
        - status: 'success' or 'error'
        - code: The generated Python/Qiskit code (primary output - display this to user)
        - completion_id: ID for accepting/tracking this completion
        - message: Error description if status is 'error'

    Example successful response:
        {
            "status": "success",
            "code": "from qiskit import QuantumCircuit\\n\\nqc = QuantumCircuit(2)\\nqc.h(0)\\nqc.cx(0, 1)",
            "completion_id": "cmpl_abc123"
        }
    """
    if not prompt or not prompt.strip():
        return {"status": "error", "message": "prompt is required and cannot be empty"}

    if len(prompt.strip()) > 10000:  # Reasonable limit
        return {
            "status": "error",
            "message": "prompt is too long (max 10000 characters)",
        }

    try:
        logger.info(f"Requesting code completion for prompt (length: {len(prompt)})")
        url = f"{QCA_TOOL_API_BASE}/v1/completions"
        body = {"model": _SELECTED_MODEL_NAME, "prompt": prompt.strip()}
        data = await make_qca_request(url, method="POST", body=body)

        if "error" in data:
            logger.error(f"Failed to get completion: {data['error']}")
            return {"status": "error", "message": data["error"]}

        choices = data.get("choices")
        completion_id = data.get("id")

        if not choices:
            logger.warning("No choices returned for completion request")
            return {"status": "error", "message": "No choices for this prompt."}
        else:
            # Extract the primary code from the first choice for easy LLM access
            primary_code = choices[0].get("text", "") if choices else ""
            logger.info(
                f"Successfully generated completion with {len(choices)} choices (ID: {completion_id})"
            )
            return {
                "status": "success",
                "code": primary_code,
                "completion_id": completion_id,
            }
    except Exception as e:
        logger.error(f"Exception in get_completion: {e!s}")
        return {"status": "error", "message": f"Failed to get completion: {e!s}"}


@with_sync
async def get_rag_completion(prompt: str) -> dict[str, Any]:
    """
    Get RAG (Retrieval-Augmented Generation) completion for answering questions about
    Qiskit and quantum computing concepts.

    This tool uses IBM's knowledge base to answer conceptual questions. Use it when users
    ask about quantum computing theory, Qiskit features, best practices, or need explanations
    rather than code generation.

    When to use this vs get_completion:
    - Use get_rag_completion for: "What is entanglement?", "How does the transpiler work?"
    - Use get_completion for: "Write a Bell state circuit", "Generate VQE code"

    Args:
        prompt: A question about Qiskit or quantum computing concepts.
                Examples: "What are the different types of quantum gates?"
                         "How do I optimize circuit depth?"

    Returns:
        Dictionary with:
        - status: 'success' or 'error'
        - answer: The explanatory text response (primary output - display this to user)
        - completion_id: ID for accepting/tracking this completion
        - message: Error description if status is 'error'

    Example successful response:
        {
            "status": "success",
            "answer": "Quantum entanglement is a phenomenon where two qubits...",
            "completion_id": "cmpl_xyz789"
        }
    """
    try:
        url = f"{QCA_TOOL_API_BASE}/v1/completions"
        body = {"model": _SELECTED_MODEL_NAME, "prompt": prompt, "mode": "rag"}
        data = await make_qca_request(url, method="POST", body=body)

        if "error" in data:
            return {"status": "error", "message": data["error"]}

        choices = data.get("choices")
        if not choices:
            return {"status": "error", "message": "No choices for this prompt."}
        else:
            # Extract the primary answer from the first choice for easy LLM access
            primary_answer = choices[0].get("text", "") if choices else ""
            return {
                "status": "success",
                "answer": primary_answer,
                "completion_id": data.get("id"),
            }
    except Exception as e:
        return {"status": "error", "message": f"Failed to get RAG completion: {e!s}"}


@with_sync
async def accept_completion(completion_id: str) -> dict[str, Any]:
    """
    Accept a suggestion generated by the Qiskit Code Assistant.

    Args:
        completion_id: The ID of the completion to accept

    Returns:
        Completion acceptance status
    """
    try:
        url = f"{QCA_TOOL_API_BASE}/v1/completion/acceptance"
        body = {"completion": completion_id, "accepted": "true"}
        data = await make_qca_request(url, method="POST", body=body)

        if "error" in data:
            return {"status": "error", "message": data["error"]}

        result = data.get("result")
        if not result:
            return {
                "status": "error",
                "message": "No result for this completion acceptance.",
            }
        else:
            return {"status": "success", "result": result}
    except Exception as e:
        return {"status": "error", "message": f"Failed to accept completion: {e!s}"}


@with_sync
async def get_service_status() -> str:
    """
    Get current Qiskit Code Assistant service status.

    Returns:
        Service connection status and basic information
    """
    try:
        logger.info("Checking Qiskit Code Assistant service status")
        # Try to get models to test connectivity
        models_result = await list_models()

        if models_result.get("status") == "success":
            model_count = len(models_result.get("models", []))
            status_info = {
                "connected": True,
                "api_base": QCA_TOOL_API_BASE,
                "model_name": QCA_TOOL_MODEL_NAME,
                "available_models": model_count,
                "timeout": QCA_REQUEST_TIMEOUT,
            }
            logger.info("Qiskit Code Assistant service is accessible")
        else:
            status_info = {
                "connected": False,
                "api_base": QCA_TOOL_API_BASE,
                "model_name": QCA_TOOL_MODEL_NAME,
                "error": models_result.get("message", "Unknown error"),
                "timeout": QCA_REQUEST_TIMEOUT,
            }
            logger.warning("Qiskit Code Assistant service is not accessible")

        return f"Qiskit Code Assistant Service Status: {status_info}"
    except Exception as e:
        logger.error(f"Failed to check service status: {e!s}")
        return f"Qiskit Code Assistant Service Status: Error - {e!s}"


# Assisted by watsonx Code Assistant
