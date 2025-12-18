# Qiskit IBM Runtime MCP Server

A comprehensive Model Context Protocol (MCP) server that provides AI assistants with access to IBM Quantum computing services through Qiskit IBM Runtime. This server enables quantum circuit creation, execution, and management directly from AI conversations.

## Features

- **Quantum Backend Management**: List and inspect available quantum backends
- **Job Management**: Monitor, cancel, and retrieve job results
- **Account Management**: Easy setup and configuration of IBM Quantum accounts

## Prerequisites

- Python 3.10 or higher
- IBM Quantum account (free at [quantum.cloud.ibm.com](https://quantum.cloud.ibm.com))
- IBM Quantum API token

## Installation

### Install from PyPI

The easiest way to install is via pip:

```bash
pip install qiskit-ibm-runtime-mcp-server
```

### Install from Source

This project recommends using [uv](https://astral.sh/uv) for virtual environments and dependencies management. If you don't have `uv` installed, check out the instructions in <https://docs.astral.sh/uv/getting-started/installation/>

### Setting up the Project with uv

1. **Initialize or sync the project**:
   ```bash
   # This will create a virtual environment and install dependencies
   uv sync
   ```

2. **Get your IBM Quantum token** (if you don't have saved credentials):
   - Visit [IBM Quantum](https://quantum.cloud.ibm.com/)
   - Find your API key. From the [dashboard](https://quantum.cloud.ibm.com/), create your API key, then copy it to a secure location so you can use it for authentication. [More information](https://quantum.cloud.ibm.com/docs/en/guides/save-credentials)

3. **Configure your credentials** (choose one method):

   **Option A: Environment Variable (Recommended)**
   ```bash
   # Copy the example environment file
   cp .env.example .env

   # Edit .env and add your IBM Quantum API token
   export QISKIT_IBM_TOKEN="your_token_here"

   # Optional: Set instance for faster startup (skips instance lookup)
   export QISKIT_IBM_RUNTIME_MCP_INSTANCE="your-instance-crn"
   ```

   **Option B: Save Credentials Locally**
   ```python
   from qiskit_ibm_runtime import QiskitRuntimeService

   # Save your credentials (one-time setup)
   QiskitRuntimeService.save_account(
       channel="ibm_quantum_platform",
       token="your_token_here",
       overwrite=True
   )
   ```
   This stores your credentials in `~/.qiskit/qiskit-ibm.json`

   **Option C: Pass Token Directly**
   ```python
   # Provide token when setting up the account
   await setup_ibm_quantum_account(token="your_token_here")
   ```

   **Credential Resolution Priority:**
   The server looks for credentials in this order:
   1. Explicit token passed to `setup_ibm_quantum_account()`
   2. `QISKIT_IBM_TOKEN` environment variable
   3. Saved credentials in `~/.qiskit/qiskit-ibm.json`

   **Instance Configuration (Optional):**
   To speed up service initialization, you can specify your IBM Quantum instance:
   - Set `QISKIT_IBM_RUNTIME_MCP_INSTANCE` environment variable with your instance CRN
   - This skips the automatic instance lookup which can be slow
   - Find your instance CRN in [IBM Quantum Platform](https://quantum.cloud.ibm.com/instances)

   **Instance Priority:**
   - If you saved credentials with an instance (via `save_account(instance="...")`), the SDK uses it automatically
   - `QISKIT_IBM_RUNTIME_MCP_INSTANCE` **overrides** any instance saved in credentials
   - If neither is set, the SDK performs a slow lookup across all instances

   > **Note:** `QISKIT_IBM_RUNTIME_MCP_INSTANCE` is an MCP server-specific variable, not a standard Qiskit SDK environment variable.

## Quick Start

### Running the Server

```bash
uv run qiskit-ibm-runtime-mcp-server
```

The server will start and listen for MCP connections.

### Basic Usage Examples

#### Async Usage (MCP Server)

```python
# 1. Setup IBM Quantum Account (optional if credentials already configured)
# Will use saved credentials or environment variable if token not provided
await setup_ibm_quantum_account()  # Uses saved credentials/env var
# OR
await setup_ibm_quantum_account(token="your_token_here")  # Explicit token

# 2. List Available Backends (no setup needed if credentials are saved)
backends = await list_backends()
print(f"Available backends: {len(backends['backends'])}")

# 3. Get the least busy backend
backend = await least_busy_backend()
print(f"Least busy backend: {backend}")

# 4. Get backend's properties
backend_props = await get_backend_properties("backend_name")
print(f"Backend_name properties: {backend_props}")

# 5. List recent jobs
jobs = await list_my_jobs(10)
print(f"Last 10 jobs: {jobs}")

# 6. Get job status
job_status = await get_job_status("job_id")
print(f"Job status: {job_status}")

# 7. Cancel job
cancelled_job = await cancel_job("job_id")
print(f"Cancelled job: {cancelled_job}")
```

#### Sync Usage (DSPy, Scripts, Jupyter)

For frameworks that don't support async operations, all async functions have a `.sync` attribute:

```python
from qiskit_ibm_runtime_mcp_server.ibm_runtime import (
    setup_ibm_quantum_account,
    list_backends,
    least_busy_backend,
    get_backend_properties,
    list_my_jobs,
    get_job_status,
    cancel_job
)

# Optional: Setup account if not already configured
# Will automatically use QISKIT_IBM_TOKEN env var or saved credentials
setup_ibm_quantum_account.sync()  # No token needed if already configured

# Use .sync for synchronous execution - no setup needed if credentials saved
backends = list_backends.sync()
print(f"Available backends: {backends['total_backends']}")

# Get least busy backend
backend = least_busy_backend.sync()
print(f"Least busy: {backend['backend_name']}")

# Works in Jupyter notebooks (handles nested event loops automatically)
jobs = list_my_jobs.sync(limit=5)
print(f"Recent jobs: {len(jobs['jobs'])}")
```

**DSPy Integration Example:**

```python
import dspy
from dotenv import load_dotenv
from qiskit_ibm_runtime_mcp_server.ibm_runtime import (
    setup_ibm_quantum_account,
    list_backends,
    least_busy_backend,
    get_backend_properties
)

# Load environment variables (includes QISKIT_IBM_TOKEN)
load_dotenv()

# Use .sync versions for DSPy tools
agent = dspy.ReAct(
    YourSignature,
    tools=[
        setup_ibm_quantum_account.sync,  # Optional - only if you need to verify setup
        list_backends.sync,
        least_busy_backend.sync,
        get_backend_properties.sync
    ]
)

result = agent(user_request="What QPUs are available?")
```


## API Reference

### Tools

#### `setup_ibm_quantum_account(token: str = "", channel: str = "ibm_quantum_platform")`
Configure IBM Quantum account with API token.

**Parameters:**
- `token` (optional): IBM Quantum API token. If not provided, the function will:
  1. Check for `QISKIT_IBM_TOKEN` environment variable
  2. Use saved credentials from `~/.qiskit/qiskit-ibm.json`
- `channel`: Service channel (default: `"ibm_quantum_platform"`)

**Returns:** Setup status and account information

**Note:** If you already have saved credentials or have set the `QISKIT_IBM_TOKEN` environment variable, you can call this function without parameters or skip it entirely and use other functions directly.

#### `list_backends()`
Get list of available quantum backends.

**Returns:** Array of backend information including:
- Name, status, queue length
- Number of qubits, coupling map
- Simulator vs. hardware designation

### `least_busy_backend()`
Get the current least busy IBM Quantum backend available
**Returns:** The backend with the fewest number of pending jobs

#### `get_backend_properties(backend_name: str)`
Get detailed properties of specific backend.

**Returns:** Complete backend configuration including:
- Hardware specifications
- Gate set and coupling map
- Current operational status
- Queue information

#### `list_my_jobs(limit: int = 10)`
Get list of recent jobs from your account.

**Parameters:**
- `limit`: The N of jobs to retrieve

#### `get_job_status(job_id: str)`
Check status of submitted job.

**Parameters:**
- `job_id`: The ID of the job to get its status

**Returns:** Current job status, creation date, backend info

#### `cancel_job(job_id: str)`
Cancel a running or queued job.

**Parameters:**
- `job_id`: The ID of the job to cancel


### Resources

#### `ibm_quantum://status`
Get current service status and connection info.


## Security Considerations

- **Store IBM Quantum tokens securely**: Never commit tokens to version control
- **Use environment variables for production deployments**: Set `QISKIT_IBM_TOKEN` environment variable
- **Credential Priority**: The server automatically resolves credentials in this order:
  1. Explicit token parameter (highest priority)
  2. `QISKIT_IBM_TOKEN` environment variable
  3. Saved credentials in `~/.qiskit/qiskit-ibm.json` (lowest priority)
- **Token Validation**: The server rejects placeholder values like `<PASSWORD>`, `<TOKEN>`, etc., to prevent accidental credential corruption
- **Implement rate limiting for production use**: Monitor API request frequency
- **Monitor quantum resource consumption**: Track job submissions and backend usage

## Contributing

Contributions are welcome! Areas for improvement:

- Support for Primitives
- Support for error mitigation/correction/cancellation techniques
- Other qiskit-ibm-runtime features


### Other ways of testing and debugging the server

> _**Note**: to launch the MCP inspector you will need to have [`node` and `npm`](https://docs.npmjs.com/downloading-and-installing-node-js-and-npm)_

1. From a terminal, go into the cloned repo directory

1. Switch to the virtual environment

    ```sh
    source .venv/bin/activate
    ```

1. Run the MCP Inspector:

    ```sh
    npx @modelcontextprotocol/inspector uv run qiskit-ibm-runtime-mcp-server
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
uv run pytest tests/test_server.py -v
```

### Test Structure

- `tests/test_server.py` - Unit tests for server functions
- `tests/test_sync.py` - Unit tests for synchronous execution
- `tests/test_integration.py` - Integration tests
- `tests/conftest.py` - Test fixtures and configuration

### Test Coverage

The test suite covers:
- ✅ Service initialization and account setup
- ✅ Backend listing and analysis
- ✅ Job management and monitoring
- ✅ Synchronous execution (`.sync` methods)
- ✅ Error handling and validation
- ✅ Integration scenarios
- ✅ Resource and tool handlers

