# qiskit-code-assistant-mcp-server

MCP server for Qiskit Code Assistant


## Components


### Tools

The server implements one tool:
- `qca_completion`: Get completion for a given prompt
  - Takes a "prompt" as a required string argument
  - Connects to a Qiskit Code Assistant service and returns a code completion based on the prompt


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

## Quick Start

### Running the Server

```bash
uv run qiskit-code-assistant-mcp-server
```

The server will start and listen for MCP connections.

### Using Sync Wrappers

For frameworks that don't support async operations (DSPy, traditional scripts, etc.), use the synchronous wrappers:

```python
from qiskit_code_assistant_mcp_server.sync import (
    qca_get_completion_sync,
    qca_get_rag_completion_sync,
    qca_list_models_sync
)

# Use synchronously without async/await
result = qca_get_completion_sync("Write a quantum circuit for a Bell state")
print(result)

# Works in Jupyter notebooks
rag_result = qca_get_rag_completion_sync("What is quantum entanglement?")
print(rag_result)
```

**Available sync functions:**
- `qca_list_models_sync()` - List available models
- `qca_get_model_sync(model_id)` - Get model info
- `qca_get_completion_sync(prompt)` - Get code completion
- `qca_get_rag_completion_sync(prompt)` - Get RAG-based completion
- `qca_accept_completion_sync(completion_id)` - Accept a completion
- `qca_get_service_status_sync()` - Get service status


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
- `tests/test_integration.py` - Integration tests
- `tests/conftest.py` - Test fixtures and configuration

### Test Coverage

The test suite covers:
- ✅ All QCA API interactions
- ✅ Error handling and validation
- ✅ HTTP client management
- ✅ Configuration validation
- ✅ Integration scenarios
- ✅ Resource and tool handlers

## Resources

- [Qiskit Code Assistant](https://docs.quantum.ibm.com/guides/qiskit-code-assistant)
- [Model Context Protocol (MCP)](https://modelcontextprotocol.io/introduction)
- [MCP Inspector](https://github.com/modelcontextprotocol/inspector)
