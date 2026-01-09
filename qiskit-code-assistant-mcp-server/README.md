# qiskit-code-assistant-mcp-server

MCP server for Qiskit Code Assistant


## Components


### Tools

The server implements four tools:
- `accept_model_disclaimer_tool`: Accept the disclaimer for a given model
- `get_completion_tool`: Get completion for a given prompt
- `get_rag_completion_tool`: Get RAG completion for answering conceptual or descriptive questions about Qiskit or Quantum
- `accept_completion_tool`: Accept a given completion


## Prerequisites

- Python 3.10 or higher
- [uv](https://astral.sh/uv) package manager (recommended)
- IBM Quantum account and API token
- Access to Qiskit Code Assistant service

## Installation

### Install from PyPI

The easiest way to install is via pip:

```bash
pip install qiskit-code-assistant-mcp-server
```

### Install from Source

This project uses [uv](https://astral.sh/uv) for virtual environments and dependencies management. If you don't have `uv` installed, check out the instructions in <https://docs.astral.sh/uv/getting-started/installation/>

### Setting up the Project with uv

1. **Initialize or sync the project**:
   ```bash
   # This will create a virtual environment and install dependencies
   uv sync
   ```

2. **Configure environment variables**:
   ```bash
   # Copy the example environment file
   cp .env.example .env
   
   # Edit .env and add your IBM Quantum API token
   # Get your token from: https://cloud.quantum.ibm.com/
   ```

## Configuration

### Environment Variables

The server can be configured using environment variables in your `.env` file:

- `QISKIT_IBM_TOKEN` - Your IBM Quantum API token (required)
- `QCA_TOOL_API_BASE` - Qiskit Code Assistant API base URL (default: `https://qiskit-code-assistant.quantum.ibm.com`)
- `QCA_TOOL_MODEL_NAME` - Default model name (default: `mistral-small-3.2-24b-qiskit`)
- `QCA_REQUEST_TIMEOUT` - Request timeout in seconds (default: `30.0`)
- `QCA_MCP_DEBUG_LEVEL` - Logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL (default: `INFO`)

### Model Selection

The server includes an **automatic model availability guardrail** that:
- Checks available models from the Qiskit Code Assistant service at startup
- Uses the configured `QCA_TOOL_MODEL_NAME` if available
- Automatically falls back to the first available model if the default is unavailable
- Logs warnings when using a fallback model
- Gracefully handles API errors by using the configured default

This ensures the server can start and function even when the default model is temporarily unavailable.

## Quick Start

### Running the Server

```bash
uv run qiskit-code-assistant-mcp-server
```

The server will start and listen for MCP connections.

### Synchronous Usage

For frameworks that don't support async operations (DSPy, traditional scripts, etc.), all async functions have a `.sync` attribute for synchronous execution:

```python
from qiskit_code_assistant_mcp_server.qca import (
    get_completion,
    get_rag_completion,
    list_models
)

# Use .sync for synchronous execution
result = get_completion.sync("Write a quantum circuit for a Bell state")
print(result)

# Works in Jupyter notebooks (handles nested event loops automatically)
rag_result = get_rag_completion.sync("What is quantum entanglement?")
print(rag_result)

# List available models
models = list_models.sync()
print(models)
```

**Available functions (all support `.sync`):**
- `list_models()` - List available models
- `get_model(model_id)` - Get model info
- `get_completion(prompt)` - Get code completion
- `get_rag_completion(prompt)` - Get RAG-based completion
- `accept_completion(completion_id)` - Accept a completion
- `get_service_status()` - Get service status


### Testing and debugging the server

> _**Note**: to launch the MCP inspector you will need to have [`node` and `npm`](https://docs.npmjs.com/downloading-and-installing-node-js-and-npm)_

1. From a terminal, go into the cloned repo directory

1. Switch to the virtual environment

    ```sh
    source .venv/bin/activate
    ```

1. Run the MCP Inspector:

    ```sh
    npx @modelcontextprotocol/inspector uv run qiskit-code-assistant-mcp-server
    ```

1. Open your browser to the URL shown in the console message e.g.,

    ```
    MCP Inspector is up and running at http://localhost:5173
    ```

## Testing

This project includes comprehensive unit and integration tests.

### Running Tests

**Quick test run:**
```bash
./run_tests.sh
```

**Manual test commands:**
```bash
# Install test dependencies
uv sync --group dev --group test

# Run all tests
uv run pytest

# Run only unit tests
uv run pytest -m "not integration"

# Run only integration tests
uv run pytest -m "integration"

# Run tests with coverage
uv run pytest --cov=src --cov-report=html

# Run specific test file
uv run pytest tests/test_qca.py -v
```

### Test Structure

- `tests/test_qca.py` - Unit tests for QCA functions
- `tests/test_utils.py` - Unit tests for utility functions
- `tests/test_constants.py` - Unit tests for configuration
- `tests/test_sync.py` - Unit tests for synchronous execution
- `tests/test_integration.py` - Integration tests
- `tests/conftest.py` - Test fixtures and configuration

### Test Coverage

The test suite covers:
- ✅ All QCA API interactions
- ✅ Model selection and availability guardrail
- ✅ Error handling and validation
- ✅ HTTP client management
- ✅ Synchronous execution (`.sync` methods)
- ✅ Configuration validation
- ✅ Integration scenarios
- ✅ Resource and tool handlers

## Resources

- [Qiskit Code Assistant](https://docs.quantum.ibm.com/guides/qiskit-code-assistant)
- [Model Context Protocol (MCP)](https://modelcontextprotocol.io/introduction)
- [MCP Inspector](https://github.com/modelcontextprotocol/inspector)
