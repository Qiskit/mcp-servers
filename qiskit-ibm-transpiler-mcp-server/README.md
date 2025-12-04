# qiskit-ibm-transpiler-mcp-server

MCP server for Qiskit transpiler. It supports **AI routing**, 
**AI Clifford** synthesis, **AI Linear Function** synthesis, **AI Permutation** synthesis, and **AI Pauli Network** synthesis using QASM 3.0 as the input/output tools format.

## Features 

- **AI transpiler**: Perform machine learning-based optimizations in both routing and synthesis passes
- **QASM 3.0** support: Uses QASM 3.0 strings for easy JSON serialization between tools

## Prerequisites

- python>=3.11
- qiskit-ibm-transpiler>=0.14.3
- fastmcp>=2.12.4

## Installation

### Install from PyPI

The easiest way to install is via pip:

```bash
pip install qiskit-ibm-transpiler-mcp-server
```

### Install from Source

This project recommends using [uv](https://astral.sh/uv) for virtual environments and dependencies management. If you don't have `uv` installed, check out the instructions in <https://docs.astral.sh/uv/getting-started/installation/>

### Setting up the Project with uv

1. **Initialize or sync the project**:
   ```bash
   # This will create a virtual environment and install dependencies
   uv sync
   ```
   

## Quick Start

### Running the Server

```bash
uv run qiskit-ibm-transpiler-mcp-server
```

The server will start and listen for MCP connections.


### Sync Usage (DSPy, Scripts, Jupyter)

For frameworks that don't support async operations:

```python
from qiskit_ibm_transpiler_mcp_server.sync import (
    ai_routing_sync,
    ai_clifford_synthesis_sync,
    setup_ibm_quantum_account_sync
)

from dotenv import load_dotenv

load_dotenv()

# 1. Load Quantum Circuit to be synthesized as QASM 3.0 string
qasm_string = "your_qasm_circuit_here"

# 2. Setup IBM Quantum Account (optional if credentials already configured)
# Will use saved credentials or environment variable if token not provided
setup_ibm_quantum_account_sync()

# Works in Jupyter notebooks and DSPy agents
# 3. AI Clifford Synthesis

# 3.1 AI Routing [Optional]
routed_circuit = ai_routing_sync(circuit_qasm=qasm_string, backend_name="backend_name")
routed_qasm_string = routed_circuit['optimized_circuit_qasm']
print(f"QASM 3.0 routed circuit: {routed_qasm_string}")

# 3.2 AI Clifford Synthesis pass
clifford_synthesized_circuit = ai_clifford_synthesis_sync(circuit_qasm=routed_qasm_string, backend_name="backend_name")
print(f"Clifford synthesized quantum circuit: {clifford_synthesized_circuit['optimized_circuit_qasm']}")
```

**DSPy Integration Example:**
> _**Note**: to launch the DSPy script you will need to have [`dspy`](https://dspy.ai) installed_ 
```python
import dspy

from dotenv import load_dotenv
from qiskit_ibm_transpiler_mcp_server.sync import (
    ai_clifford_synthesis_sync,
    ai_routing_sync,
    ai_linear_function_synthesis_sync,
    ai_pauli_network_synthesis_sync,
    ai_permutation_synthesis_sync,
    setup_ibm_quantum_account_sync
)


# Load environment variables (includes QISKIT_IBM_TOKEN)
load_dotenv()

lm = dspy.LM("your_llm_model_here", api_base="http://localhost:11434", api_key="")
dspy.configure(lm=lm)

# The agent will automatically use saved credentials or environment variables
# No need to explicitly pass tokens to individual functions
agent = dspy.ReAct(
    "question -> answer",
    tools=[
        setup_ibm_quantum_account_sync,
        ai_routing_sync,
        ai_clifford_synthesis_sync,
        ai_linear_function_synthesis_sync,
        ai_pauli_network_synthesis_sync,
        ai_permutation_synthesis_sync
    ]
)

# load your Quantum Circuit to be synthesized as QASM 3.0 string
qasm_strig = "your_qasm_string_here"
result = agent(question=f"Can you synthesize the given QASM Clifford Quantum circuit using ibm_fez backend? This is the QASM string {qasm_strig}")
print(result)
```

### Other ways of testing and debugging the server
1. From a terminal, go into the cloned repo directory 
2. Switch to the virtual environment
    ```sh
    source .venv/bin/activate
    ```
   
3. Run the MCP Inspector:
    ```sh
    npx @modelcontextprotocol/inspector uv run qiskit-ibm-transpiler-mcp-server
    ```
   
4. Open your browser to the URL shown in the console message e.g.,
    ```
    MCP Inspector is up and running at http://localhost:6277
    ```
> _**Note**: to launch the MCP inspector you will need to have [`node` and `npm`](https://docs.npmjs.com/downloading-and-installing-node-js-and-npm)_
# API Reference
## Tools

### Setup IBM Account
Configure IBM Quantum account with API token.
#### `setup_ibm_quantum_account_tool(token: str = "", channel: str = "ibm_quantum_platform")`
**Parameters:**
- `token` (optional): IBM Quantum API token. If not provided, the function will:
  1. Check for `QISKIT_IBM_TOKEN` environment variable
  2. Use saved credentials from `~/.qiskit/qiskit-ibm.json`
- `channel`: Service channel (default: `"ibm_quantum_platform"`)

**Returns:** Setup status and account information

**Note:** If you already have saved credentials or have set the `QISKIT_IBM_TOKEN` environment variable, you can call this function without parameters or skip it entirely and use other functions directly.


### AI Routing
Route input quantum circuit. It inserts SWAP operations on a circuit to make two-qubits operations compatible with a given coupling map that restricts the pair of qubits on which operations can be applied. You may want to execute routing pass before any other AI transpiling synthesis pass. 
```
ai_routing(
   circuit_qasm: str, 
   backend_name: str, 
   optimization_level: int = 1,
   layout_mode: str = "optimize",
   optimization_preferences: Literal[
      "n_cnots", "n_gates", "cnot_layers", "layers", "noise"
      ] | list[Literal["n_cnots", "n_gates", "cnot_layers", "layers", "noise"]] | None = None,
   local_mode: bool = True
)
```
**Parameters:**
- `circuit_qasm`: quantum circuit as QASM 3.0 string to be synthesized
- `backend_name`: Qiskit Runtime Service backend name on which to map the input circuit synthesis
- `optimization_level` (optional): The potential optimization level to apply during the transpilation process. Valid values are [1,2,3], where 1 is the least optimization (and fastest), and 3 the most optimization (and most time-intensive)
- `layout_mode` (optional): Specifies how to handle the layout selection. It can assume the following values:
  - `"keep"`: This respects the layout set by the previous transpiler passes. Typically used when the circuit must be run on specific qubits of the device. It often produces worse results because it has less room for optimization
  - `"improve"`: It is useful when you have a good initial guess for the layout
  - `"optimize"`: This is the default mode. It works best for general circuits where you might not have good layout guesses. This mode ignores previous layout selections
- `optimization_preferences` (optional): Indicates what you want to reduce through optimization: number of cnot gates (n_cnots), number of gates (n_gates), number of cnots layers (cnot_layers), number of layers (layers), and/or noise (noise)
- `local_mode` (optional): determines where the AIRouting pass runs. If False, AIRouting runs remotely through the Qiskit Transpiler Service. If True, the package tries to run the pass in your local environment with a fallback to cloud mode if the required dependencies are not found

**Returns:** Optimization status and routed qasm circuit

**Note:** Currently, only the local mode execution is available

### AI Clifford synthesis 
Synthesis for Clifford circuits (blocks of H, S, and CX gates). Currently, up to nine qubit blocks.
```
ai_clifford_synthesis(
   circuit_qasm: str, 
   backend_name: str, 
   replace_only_if_better: bool = True,
   local_mode: bool = True
)
```
**Parameters:**
- `circuit_qasm`: quantum circuit as QASM 3.0 string to be synthesized
- `backend_name`: Qiskit Runtime Service backend name on which to map the input circuit synthesis
- `replace_only_if_better` (optional): By default, the synthesis will replace the original sub-circuit only if the synthesized sub-circuit improves the original (currently only checking CNOT count), but this can be forced to always replace the circuit by setting replace_only_if_better=False
- `local_mode` (optional): determines where the AI Clifford synthesis runs. If False, AI Clifford synthesis runs remotely through the Qiskit Transpiler Service. If True, the package tries to run the pass in your local environment with a fallback to cloud mode if the required dependencies are not found

**Returns:** Optimization status and synthesized qasm for Clifford circuits

**Note:** Currently, only the local mode execution is available

### AI Linear Function synthesis
Synthesis for Linear Function circuits (blocks of CX and SWAP gates). Currently, up to nine qubit blocks.
```
ai_linear_function_synthesis(
   circuit_qasm: str, 
   backend_name: str, 
   replace_only_if_better: bool = True,
   local_mode: bool = True
)
```

**Parameters:**
- `circuit_qasm`: quantum circuit as QASM 3.0 string to be synthesized
- `backend_name`: Qiskit Runtime Service backend name on which to map the input circuit synthesis
- `replace_only_if_better` (optional): By default, the synthesis will replace the original sub-circuit only if the synthesized sub-circuit improves the original (currently only checking CNOT count), but this can be forced to always replace the circuit by setting replace_only_if_better=False
- `local_mode` (optional): determines where the Linear Function synthesis pass runs. If False, Linear Function synthesis runs remotely through the Qiskit Transpiler Service. If True, the package tries to run the pass in your local environment with a fallback to cloud mode if the required dependencies are not found

**Returns:** Optimization status and synthesized qasm for Linear Function circuits

**Note:** Currently, only the local mode execution is available

### AI Permutation synthesis
Synthesis for Permutation circuits (blocks of SWAP gates). Currently available for 65, 33, and 27 qubit blocks.
```
ai_permutation_synthesis(
   circuit_qasm: str, 
   backend_name: str, 
   replace_only_if_better: bool = True,
   local_mode: bool = True
)
```

**Parameters:**
- `circuit_qasm`: quantum circuit as QASM 3.0 string to be synthesized
- `backend_name`: Qiskit Runtime Service backend name on which to map the input circuit synthesis
- `replace_only_if_better` (optional): By default, the synthesis will replace the original sub-circuit only if the synthesized sub-circuit improves the original (currently only checking CNOT count), but this can be forced to always replace the circuit by setting replace_only_if_better=False
- `local_mode` (optional): determines where the AI Permutation synthesis pass runs. If False, AI Permutation synthesis runs remotely through the Qiskit Transpiler Service. If True, the package tries to run the pass in your local environment with a fallback to cloud mode if the required dependencies are not found

**Returns:** Optimization status and synthesized qasm for Permutation circuits

**Note:** Currently, only the local mode execution is available

### AI Pauli Network synthesis
Synthesis for Pauli Network circuits (blocks of H, S, SX, CX, RX, RY and RZ gates). Currently, up to six qubit blocks.
```
ai_pauli_network_synthesis(
   circuit_qasm: str, 
   backend_name: str, 
   replace_only_if_better: bool = True,
   local_mode: bool = True
)
```

**Parameters:**
- `circuit_qasm`: quantum circuit as QASM 3.0 string to be synthesized
- `backend_name`: Qiskit Runtime Service backend name on which to map the input circuit synthesis
- `replace_only_if_better` (optional): By default, the synthesis will replace the original sub-circuit only if the synthesized sub-circuit improves the original (currently only checking CNOT count), but this can be forced to always replace the circuit by setting replace_only_if_better=False
- `local_mode` (optional): determines where the AI Pauli Network synthesis pass runs. If False, AI Pauli Network synthesis runs remotely through the Qiskit Transpiler Service. If True, the package tries to run the pass in your local environment with a fallback to cloud mode if the required dependencies are not found

**Returns:** Optimization status and synthesized qasm for Pauli Network circuits

**Note:** Currently, only the local mode execution is available


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

- `tests/integration/test_qta.py` - Integration tests for async server functions
- `tests/integration/test_sync.py` - Integration tests for sync server functions
- `tests/integration/test_mcp_server.py` - Generic integration tests 
- `tests/unit/test_qta.py` - Unit tests for async server functions
- `tests/unit/test_sync.py` - Unit tests for sync server functions
- `tests/unit/test_utils.py` - Unit tests for utils functions
- `tests/unit/test_qiskit_runtime_service_provider.py` - Unit tests for QiskitRuntimeServiceProvider Singleton
- `tests/conftest.py` - Test fixtures and configuration
- `tests/utils/helpers.py` - Helper functions to compute 2-qubits count and depth improvement for integration tests
- `tests/qasm/` - Three QASM 3.0 test cases in /qasm folder (2 valid, 1 malformed)

### Test Coverage

The test suite covers:

- ✅ Service initialization and account setup
- ✅ AI synthesis passes (mocked and real)
- ✅ Error handling and validation
- ✅ Integration scenarios (mocked and real)
- ✅ Tool handlers

## Areas for improvement

- Integrate Collections for complete AI synthesis pass