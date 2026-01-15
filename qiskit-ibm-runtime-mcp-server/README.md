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

# 7. Get job results (when job is DONE)
results = await get_job_results("job_id")
print(f"Counts: {results['counts']}")

# 8. Cancel job
cancelled_job = await cancel_job("job_id")
print(f"Cancelled job: {cancelled_job}")
```

#### Sync Usage (Scripts, Jupyter)

For frameworks that don't support async operations, all async functions have a `.sync` attribute:

```python
from qiskit_ibm_runtime_mcp_server.ibm_runtime import (
    setup_ibm_quantum_account,
    list_backends,
    least_busy_backend,
    get_backend_properties,
    get_coupling_map,
    find_optimal_qubit_chains,
    find_optimal_qv_qubits,
    list_my_jobs,
    get_job_status,
    get_job_results,
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

# Find optimal qubit chains for linear experiments
chains = find_optimal_qubit_chains.sync(backend['backend_name'], chain_length=5)
print(f"Best chain: {chains['chains'][0]['qubits']}")

# Find optimal qubits for Quantum Volume experiments
qv_qubits = find_optimal_qv_qubits.sync(backend['backend_name'], num_qubits=5)
print(f"Best QV subgraph: {qv_qubits['subgraphs'][0]['qubits']}")

# Works in Jupyter notebooks (handles nested event loops automatically)
jobs = list_my_jobs.sync(limit=5)
print(f"Recent jobs: {len(jobs['jobs'])}")
```

**LangChain Integration Example:**

> **Note:** To run LangChain examples you will need to install the dependencies:
> ```bash
> pip install langchain langchain-mcp-adapters langchain-openai python-dotenv
> ```

```python
import asyncio
import os
from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# Load environment variables (QISKIT_IBM_TOKEN, OPENAI_API_KEY, etc.)
load_dotenv()

async def main():
    # Configure MCP client
    mcp_client = MultiServerMCPClient({
        "qiskit-ibm-runtime": {
            "transport": "stdio",
            "command": "qiskit-ibm-runtime-mcp-server",
            "args": [],
            "env": {
                "QISKIT_IBM_TOKEN": os.getenv("QISKIT_IBM_TOKEN", ""),
                "QISKIT_IBM_RUNTIME_MCP_INSTANCE": os.getenv("QISKIT_IBM_RUNTIME_MCP_INSTANCE", ""),
            },
        }
    })

    # Use persistent session for efficient tool calls
    async with mcp_client.session("qiskit-ibm-runtime") as session:
        tools = await load_mcp_tools(session)

        # Create agent with LLM
        llm = ChatOpenAI(model="gpt-5.2", temperature=0)
        agent = create_agent(llm, tools)

        # Run a query
        response = await agent.ainvoke("What QPUs are available and which one is least busy?")
        print(response)

asyncio.run(main())
```

For more LLM providers (Anthropic, Google, Ollama, Watsonx) and detailed examples including Jupyter notebooks, see the [examples/](examples/) directory.


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

#### `get_coupling_map(backend_name: str)`
Get the coupling map (qubit connectivity) for a backend with detailed analysis.

Supports both real backends (requires credentials) and fake backends (no credentials needed).
Use `fake_` prefix for offline testing (e.g., `fake_sherbrooke`, `fake_brisbane`).

**Parameters:**
- `backend_name`: Name of the backend (e.g., `ibm_brisbane` or `fake_sherbrooke`)

**Returns:** Connectivity information including:
- `edges`: List of [control, target] qubit connection pairs
- `adjacency_list`: Neighbor mapping for each qubit
- `bidirectional`: Whether all connections work in both directions
- `num_qubits`: Total qubit count

**Use cases:**
- Circuit optimization and qubit mapping
- SWAP gate minimization planning
- Offline testing with fake backends

#### `find_optimal_qubit_chains(backend_name, chain_length, num_results, metric)`
Find optimal linear qubit chains for quantum experiments based on connectivity and calibration data.

Algorithmically identifies the best qubit chains by combining coupling map connectivity
with real-time calibration data. Essential for experiments requiring linear qubit arrangements.

**Parameters:**
- `backend_name`: Name of the backend (e.g., `ibm_brisbane`)
- `chain_length`: Number of qubits in the chain (default: 5, range: 2-20)
- `num_results`: Number of top chains to return (default: 5, max: 20)
- `metric`: Scoring metric to optimize:
  - `two_qubit_error`: Minimize sum of CX/ECR gate errors (default)
  - `readout_error`: Minimize sum of measurement errors
  - `combined`: Weighted combination of gate errors, readout, and coherence

**Returns:** Ranked chains with detailed metrics:
- `qubits`: Ordered list of qubit indices in the chain
- `score`: Total score (lower is better)
- `qubit_details`: T1, T2, readout_error for each qubit
- `edge_errors`: Two-qubit gate error for each connection

**Use cases:**
- Select qubits for variational quantum algorithms (VQE, QAOA)
- Plan linear qubit layouts for error correction experiments
- Identify high-fidelity qubit paths for state transfer
- Optimize qubit selection for 1D physics simulations

#### `find_optimal_qv_qubits(backend_name, num_qubits, num_results, metric)`
Find optimal qubit subgraphs for Quantum Volume experiments.

Unlike linear chains, Quantum Volume benefits from densely connected qubit sets where
qubits can interact with minimal SWAP operations. This tool finds connected subgraphs
and ranks them by connectivity and calibration quality.

**Parameters:**
- `backend_name`: Name of the backend (e.g., `ibm_brisbane`)
- `num_qubits`: Number of qubits in the subgraph (default: 5, range: 2-10)
- `num_results`: Number of top subgraphs to return (default: 5, max: 20)
- `metric`: Scoring metric to optimize:
  - `qv_optimized`: Balanced scoring for QV (connectivity + errors + coherence) (default)
  - `connectivity`: Maximize internal edges and minimize path lengths
  - `gate_error`: Minimize total two-qubit gate errors on internal edges

**Returns:** Ranked subgraphs with detailed metrics:
- `qubits`: List of qubit indices in the subgraph (sorted)
- `score`: Total score (lower is better)
- `internal_edges`: Number of edges within the subgraph
- `connectivity_ratio`: internal_edges / max_possible_edges
- `average_path_length`: Mean shortest path between qubit pairs
- `qubit_details`: T1, T2, readout_error for each qubit
- `edge_errors`: Two-qubit gate error for each internal edge

**Use cases:**
- Select optimal qubits for Quantum Volume experiments
- Find densely connected regions for random circuit sampling
- Identify high-quality qubit clusters for variational algorithms
- Plan qubit allocation for algorithms requiring all-to-all connectivity

#### `list_my_jobs(limit: int = 10)`
Get list of recent jobs from your account.

**Parameters:**
- `limit`: The N of jobs to retrieve

#### `get_job_status(job_id: str)`
Check status of submitted job.

**Parameters:**
- `job_id`: The ID of the job to get its status

**Returns:** Current job status, creation date, backend info

**Job Status Values:**
- `INITIALIZING`: Job is being prepared
- `QUEUED`: Job is waiting in the queue
- `RUNNING`: Job is currently executing
- `DONE`: Job completed successfully
- `CANCELLED`: Job was cancelled
- `ERROR`: Job failed with an error

#### `get_job_results(job_id: str)`
Retrieve measurement results from a completed quantum job.

**Parameters:**
- `job_id`: The ID of the completed job

**Returns:** Dictionary containing:
- `status`: "success", "pending", or "error"
- `job_id`: The job ID
- `job_status`: Current status of the job
- `counts`: Dictionary of measurement outcomes and their counts (e.g., `{"00": 2048, "11": 2048}`)
- `shots`: Total number of shots executed
- `backend`: Name of the backend used
- `execution_time`: Quantum execution time in seconds (if available)
- `message`: Status message

**Example workflow:**
```python
# 1. Submit job
result = await run_sampler_tool(circuit, backend_name)
job_id = result["job_id"]

# 2. Check status (poll until DONE)
status = await get_job_status(job_id)
print(f"Status: {status['job_status']}")

# 3. When DONE, retrieve results
if status['job_status'] == 'DONE':
    results = await get_job_results(job_id)
    print(f"Counts: {results['counts']}")
```

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

