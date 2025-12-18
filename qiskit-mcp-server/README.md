# Qiskit MCP Server

A Model Context Protocol (MCP) server that provides quantum circuit transpilation capabilities using Qiskit's pass managers. This server enables AI assistants to optimize quantum circuits for various hardware targets.

## Features

- **Circuit Transpilation**: Transpile quantum circuits with configurable optimization levels (0-3)
- **Preset Basis Gates**: Support for IBM Eagle, IBM Heron, ion trap, and other basis gate sets
- **Topology Support**: Built-in support for linear, ring, grid, and custom coupling maps
- **Circuit Analysis**: Analyze circuit complexity without transpilation
- **Optimization Comparison**: Compare results across all optimization levels
- **Dual API**: Supports both async (MCP) and sync (DSPy, Jupyter, scripts) usage

## Prerequisites

- Python 3.10 or higher
- [uv](https://docs.astral.sh/uv/) package manager (recommended) or pip

## Installation

### From PyPI (when published)

```bash
pip install qiskit-mcp-server
```

### From Source

```bash
# Clone the repository
git clone https://github.com/Qiskit/mcp-servers.git
cd mcp-servers/qiskit-mcp-server

# Install with uv
uv sync

# Or with pip
pip install -e .
```

## Quick Start

### Running the MCP Server

```bash
# With uv
uv run qiskit-mcp-server

# Or directly
qiskit-mcp-server
```

### Claude Desktop Configuration

Add to your Claude Desktop configuration (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "qiskit": {
      "command": "uv",
      "args": [
        "--directory",
        "/path/to/qiskit-mcp-servers/qiskit-mcp-server",
        "run",
        "qiskit-mcp-server"
      ]
    }
  }
}
```

## Usage Examples

### Async Usage (MCP Server / FastAPI)

```python
from qiskit_mcp_server.transpiler import transpile_circuit

# Simple Bell state circuit (QASM2 - automatically detected)
qasm = """OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
h q[0];
cx q[0], q[1];
measure q -> c;
"""

# Transpile with default settings (optimization level 2, QASM3 format)
result = await transpile_circuit(qasm)

# Transpile for IBM Heron processor
result = await transpile_circuit(
    qasm,
    optimization_level=3,
    basis_gates="ibm_heron",
    coupling_map="linear"
)
```

### Using QPY Format

```python
from qiskit import QuantumCircuit
from qiskit_mcp_server import dump_qpy_circuit
from qiskit_mcp_server.transpiler import transpile_circuit

# Create a circuit programmatically
qc = QuantumCircuit(2, 2)
qc.h(0)
qc.cx(0, 1)
qc.measure([0, 1], [0, 1])

# Convert to QPY (preserves exact parameters and metadata)
qpy_circuit = dump_qpy_circuit(qc)

# Transpile using QPY format
result = await transpile_circuit(qpy_circuit, circuit_format="qpy")
# Result includes transpiled circuit in QPY format (for chaining)
transpiled_qpy = result["transpiled_circuit"]["circuit_qpy"]

# Chain to another operation using QPY
result2 = await transpile_circuit(transpiled_qpy, circuit_format="qpy", optimization_level=3)
```

### Sync Usage (DSPy, Jupyter, Scripts)

```python
from qiskit_mcp_server.transpiler import transpile_circuit, analyze_circuit

# All async functions have a .sync attribute
result = transpile_circuit.sync(qasm, optimization_level=2)

# Analyze circuit without transpiling
analysis = analyze_circuit.sync(qasm)
print(f"Circuit depth: {analysis['circuit_info']['depth']}")
print(f"Two-qubit gates: {analysis['gate_categories']['two_qubit_gates']}")
```

### Compare Optimization Levels

```python
from qiskit_mcp_server.transpiler import compare_optimization_levels

# Compare all optimization levels (0-3) for your circuit
comparison = compare_optimization_levels.sync(qasm)

for level in range(4):
    result = comparison['optimization_results'][f'level_{level}']
    print(f"Level {level}: depth={result['depth']}, size={result['size']}")
```

## API Reference

### Tools

| Tool | Description |
|------|-------------|
| `transpile_circuit_tool` | Transpile a circuit with configurable optimization |
| `analyze_circuit_tool` | Analyze circuit structure without transpiling |
| `compare_optimization_levels_tool` | Compare all optimization levels (0-3) |

### Resources

| Resource URI | Description |
|--------------|-------------|
| `qiskit://transpiler/info` | Transpiler capabilities and documentation |
| `qiskit://transpiler/basis-gates` | Available basis gate presets |
| `qiskit://transpiler/topologies` | Available coupling map topologies |

### Core Functions

#### `transpile_circuit(circuit, optimization_level=2, basis_gates=None, coupling_map=None, initial_layout=None, seed_transpiler=None, circuit_format="qasm3")`

Transpile a quantum circuit using Qiskit's preset pass managers.

**Parameters:**
- `circuit`: Quantum circuit as QASM3 string, base64-encoded QPY, or QASM2 string (max 100 qubits, 10,000 gates)
- `optimization_level`: 0-3 (default: 2)
  - 0: No optimization, only basis gate decomposition (fastest)
  - 1: Light optimization with default layout
  - 2: Medium optimization with noise-aware layout (recommended)
  - 3: Heavy optimization for best results (can be slow for large circuits)
- `basis_gates`: List of gate names or preset ("ibm_default", "ibm_heron", etc.)
- `coupling_map`: List of edges or topology name ("linear", "ring", "grid", "full")
- `initial_layout`: List of physical qubit indices (length must match circuit qubits)
- `seed_transpiler`: Random seed for reproducibility
- `circuit_format`: Format of the input circuit ("qasm3" or "qpy"). Defaults to "qasm3". When "qasm3" is specified, QASM2 is also accepted as a fallback.

**Returns:** Dictionary with original/transpiled circuit info and optimization metrics

**Note:** Level 3 optimization can be very slow for circuits with >20 qubits or >500 gates. Use level 2 for faster results with good quality.

#### `analyze_circuit(circuit, circuit_format="qasm3")`

Analyze circuit structure and complexity.

**Returns:** Dictionary with gate counts, depth, and categorization (single/two/multi-qubit gates)

#### `compare_optimization_levels(circuit, circuit_format="qasm3")`

Compare transpilation results across all optimization levels.

**Returns:** Dictionary comparing depth, size, and gates for levels 0-3

### Circuit Format Support

The server supports two circuit formats for **input**:

| Format | Description |
|--------|-------------|
| `qasm3` | OpenQASM 3.0 string (with QASM2 fallback). Human-readable text format. |
| `qpy` | Base64-encoded QPY binary format. Preserves exact parameters and metadata. |

**QPY output:** All tools return circuits in QPY format (base64-encoded) for precision when chaining tools/servers.

**When to use each format:**
- **QASM3** (input): Best for human-readable circuits and initial input
- **QPY** (input/output): Best for preserving exact numerical parameters when chaining tools/servers

### Converting QPY to Human-Readable QASM3

To view a QPY circuit output in human-readable format, use the `qpy_to_qasm3` utility:

```python
from qiskit_mcp_server import qpy_to_qasm3
from qiskit_mcp_server.transpiler import transpile_circuit

# Transpile a circuit (returns QPY format)
result = transpile_circuit.sync(qasm_circuit, optimization_level=2)
qpy_output = result["transpiled_circuit"]["circuit_qpy"]

# Convert to human-readable QASM3
conversion = qpy_to_qasm3(qpy_output)
if conversion["status"] == "success":
    print(conversion["qasm3"])
```

### Converting QASM3 to QPY

To convert a QASM circuit to QPY format (for full fidelity when chaining tools), use `qasm3_to_qpy`:

```python
from qiskit_mcp_server import qasm3_to_qpy

qasm_circuit = '''
OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;
h q[0];
cx q[0], q[1];
'''

# Convert to QPY format
result = qasm3_to_qpy(qasm_circuit)
if result["status"] == "success":
    qpy_string = result["circuit_qpy"]
    # Use qpy_string with tools that accept QPY input
```

### Available Basis Gate Sets

| Preset | Gates | Description |
|--------|-------|-------------|
| `ibm_eagle` | id, rz, sx, x, ecr, reset | IBM Eagle r3 (127 qubits, uses ECR) |
| `ibm_heron` | id, rz, sx, x, cz, reset | IBM Heron (133-156 qubits, uses CZ) |
| `ibm_legacy` | id, rz, sx, x, cx, reset | Older IBM systems (uses CX) |

You can also provide a custom list of gate names for other hardware targets.

### Available Topologies

| Topology | Description |
|----------|-------------|
| `linear` | Chain connectivity (qubit i ↔ i+1) |
| `ring` | Linear with wraparound |
| `grid` | 2D grid connectivity |
| `heavy_hex` | IBM heavy-hex topology (Eagle/Heron architecture) |
| `full` | All-to-all connectivity |

## Limits and Performance

### Circuit Size Limits

To ensure reliable performance, the server enforces the following limits:

| Limit | Default | Environment Variable |
|-------|---------|---------------------|
| Maximum qubits | 100 | `QISKIT_MCP_MAX_QUBITS` |
| Maximum gates | 10,000 | `QISKIT_MCP_MAX_GATES` |

Circuits exceeding these limits will return an error with a descriptive message.

You can override these limits via environment variables:

```bash
# Allow up to 200 qubits and 50,000 gates
export QISKIT_MCP_MAX_QUBITS=200
export QISKIT_MCP_MAX_GATES=50000
```

### Performance Recommendations

| Optimization Level | Use Case | Performance |
|-------------------|----------|-------------|
| 0 | Quick iterations, debugging | Fastest |
| 1 | Development, prototyping | Fast |
| 2 | Production use (recommended) | Balanced |
| 3 | Critical applications, small circuits | Slowest |

**Tips:**
- Use level 2 for most use cases (best balance of quality and speed)
- Use level 3 only when circuit quality is critical AND circuit is small (<20 qubits, <500 gates)
- Use level 0 or 1 for rapid prototyping and development
- The `compare_optimization_levels` tool helps identify the best level for your specific circuit

## Transpilation Stages

The Qiskit transpiler processes circuits through six stages:

1. **init**: Decompose multi-qubit gates to 1 and 2-qubit operations
2. **layout**: Map virtual qubits to physical qubits
3. **routing**: Insert SWAP gates for hardware connectivity
4. **translation**: Convert to target basis gates
5. **optimization**: Reduce gate count and circuit depth
6. **scheduling**: Add timing and delay instructions

## Testing

```bash
# Run all tests
uv run pytest tests/ -v

# Run with coverage
uv run pytest tests/ --cov=src --cov-report=term-missing

# Run specific test class
uv run pytest tests/test_transpiler.py::TestTranspileCircuit -v
```

## Development

```bash
# Install dev dependencies
uv sync --group dev --group test

# Run linting
uv run ruff check src tests
uv run ruff format --check src tests

# Run type checking
uv run mypy src

# Run all checks
./run_tests.sh
```

## Contributing

Contributions are welcome! Please see the [CONTRIBUTING.md](../CONTRIBUTING.md) guide in the root of the repository.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
