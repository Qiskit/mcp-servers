# Qiskit MCP Servers - Advanced Examples

This directory contains world-class examples demonstrating the full power of combining multiple Qiskit MCP servers with advanced AI agent frameworks.

## Quantum Volume Optimizer

**A Deep Agent Multi-Server Example**

The Quantum Volume Optimizer is a sophisticated multi-agent system that finds the optimal Quantum Volume (QV) configuration for any IBM Quantum backend. It showcases:

- **Deep Agents Framework**: Coordinator agent with specialized subagents
- **Multi-Server Orchestration**: Combines 3 MCP servers working together
- **Real-World Quantum Optimization**: Finds best qubit chains and transpilation strategies

### Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    QUANTUM VOLUME OPTIMIZER                         │
│                      (Coordinator Agent)                            │
│                                                                     │
│  Plans strategy, coordinates subagents, synthesizes final report   │
└─────────────────────────────────────────────────────────────────────┘
                                   │
          ┌────────────────────────┼────────────────────────┐
          │                        │                        │
          ▼                        ▼                        ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  BACKEND        │    │  QUBIT CHAIN    │    │  TRANSPILER     │
│  ANALYST        │    │  OPTIMIZER      │    │  BENCHMARKER    │
│                 │    │                 │    │                 │
│  - List backends│    │  - Analyze      │    │  - Compare      │
│  - Get properties│   │    connectivity │    │    opt levels   │
│  - Find least   │    │  - Find best    │    │  - AI vs local  │
│    busy         │    │    chains       │    │    transpilation│
└─────────────────┘    └─────────────────┘    └─────────────────┘
        │                      │                      │
        ▼                      ▼               ┌──────┴──────┐
┌─────────────────┐    ┌─────────────────┐    ▼             ▼
│ qiskit-ibm-     │    │ qiskit-mcp-     │  qiskit-mcp-  qiskit-ibm-
│ runtime-mcp     │    │ server          │  server       transpiler-mcp
│                 │    │                 │  (local)      (AI)
│ Backend info    │    │ Local circuit   │
│ & properties    │    │ transpilation   │
└─────────────────┘    └─────────────────┘
```

**Note:** The Transpiler Benchmarker uses tools from both `qiskit-mcp-server` (local transpilation) and `qiskit-ibm-transpiler-mcp-server` (AI-powered optimization) to compare approaches.

### What is Quantum Volume?

Quantum Volume (QV) is a single-number metric that captures the largest random circuit of equal width and depth that a quantum computer can successfully implement. A QV of 2^n means the device can reliably execute n-qubit circuits of depth n.

Key factors the optimizer analyzes:
- **Two-qubit gate fidelity** (most important for QV)
- **Qubit connectivity** (linear chains need SWAP gates)
- **Coherence times** (T1, T2)
- **Readout accuracy**

### Available Examples

| File | Description |
|------|-------------|
| `quantum_volume_optimizer.py` | Full-featured command-line deep agent |
| `quantum_volume_optimizer.ipynb` | Interactive Jupyter notebook version |

### Prerequisites

```bash
# Install Deep Agents and LangChain
pip install deepagents langchain langchain-mcp-adapters python-dotenv

# Install your LLM provider (Anthropic recommended for complex reasoning)
pip install langchain-anthropic

# Install all Qiskit MCP servers
pip install qiskit-mcp-servers

# Or install individually:
pip install qiskit-mcp-server
pip install qiskit-ibm-runtime-mcp-server
pip install qiskit-ibm-transpiler-mcp-server
```

### Environment Variables

```bash
# Required
export QISKIT_IBM_TOKEN="your-ibm-quantum-token"
export ANTHROPIC_API_KEY="your-anthropic-api-key"

# Optional (recommended for faster IBM Runtime startup)
export QISKIT_IBM_RUNTIME_MCP_INSTANCE="your-instance-name"
```

Or create a `.env` file in this directory.

### Running the Optimizer

**Command-line (full optimization):**

```bash
python quantum_volume_optimizer.py
```

**With options:**

```bash
# Use OpenAI instead of Anthropic
python quantum_volume_optimizer.py --provider openai

# Evaluate up to QV depth 8
python quantum_volume_optimizer.py --depth 8

# Interactive mode for follow-up questions
python quantum_volume_optimizer.py --interactive
```

**Jupyter Notebook:**

```bash
jupyter notebook quantum_volume_optimizer.ipynb
```

### Optimization Workflow

1. **Backend Discovery**: The Backend Analyst subagent discovers all available backends and identifies promising candidates based on qubit count, existing QV, and queue length.

2. **Qubit Chain Analysis**: The Qubit Chain Optimizer analyzes backend coupling maps to find optimal linear chains considering:
   - Two-qubit gate error rates
   - Direct connectivity (minimize SWAPs)
   - Coherence times
   - Readout fidelity

3. **Transpilation Comparison**: The Transpiler Benchmarker compares:
   - Local transpilation (optimization levels 0-3)
   - AI-powered routing
   - AI synthesis passes (Clifford, Linear Function)

4. **Final Recommendation**: The Coordinator synthesizes all findings into a comprehensive report with actionable recommendations.

### Sample Output

```
═══════════════════════════════════════════════════════════════════════
  QUANTUM VOLUME OPTIMIZATION REPORT
═══════════════════════════════════════════════════════════════════════

## Executive Summary

Recommended backend: ibm_brisbane
Optimal QV-5 configuration: qubits [12, 13, 14, 15, 16]
Best transpilation: AI Routing + Local Level 2
Expected achievable QV: 32 (2^5) with high confidence

## Backend Analysis

| Backend       | Qubits | Current QV | Queue | Status      |
|--------------|--------|------------|-------|-------------|
| ibm_brisbane | 127    | 128        | 12    | Operational |
| ibm_kyiv     | 127    | 64         | 45    | Operational |
| ibm_sherbrooke| 127   | 32         | 8     | Operational |

## Optimal Qubit Chains

### QV-5 (5 qubits)
1. [12, 13, 14, 15, 16] - Score: 0.9823
2. [45, 46, 47, 48, 49] - Score: 0.9756
3. [89, 90, 91, 92, 93] - Score: 0.9701

### QV-4 (4 qubits)
1. [12, 13, 14, 15] - Score: 0.9912
...

## Transpilation Comparison (QV-5 on optimal chain)

| Strategy        | Depth | 2Q Gates | Total Gates |
|----------------|-------|----------|-------------|
| Local Level 0  | 47    | 32       | 89          |
| Local Level 1  | 38    | 28       | 76          |
| Local Level 2  | 31    | 24       | 64          |
| Local Level 3  | 29    | 23       | 61          |
| AI Routing     | 28    | 22       | 58          |
| AI + Level 2   | 26    | 20       | 54          | ← Best

## Configuration

```python
from qiskit_ibm_runtime import QiskitRuntimeService

service = QiskitRuntimeService()
backend = service.backend("ibm_brisbane")

# Use these qubits for QV-5
optimal_qubits = [12, 13, 14, 15, 16]

# Transpilation settings
optimization_level = 2
use_ai_routing = True
```
```

### MCP Servers Used

| Server | Role |
|--------|------|
| `qiskit-ibm-runtime-mcp-server` | Backend discovery, properties, calibration data |
| `qiskit-mcp-server` | Local circuit transpilation and analysis |
| `qiskit-ibm-transpiler-mcp-server` | AI-powered routing and synthesis |

### Deep Agent Components

| Component | Role |
|-----------|------|
| **Coordinator** | Plans strategy, delegates tasks, synthesizes final report |
| **Backend Analyst** | Discovers and analyzes IBM Quantum backends |
| **Qubit Chain Optimizer** | Finds optimal qubit chains based on error rates |
| **Transpiler Benchmarker** | Compares local vs AI optimization strategies |

### Extending the Example

You can extend this example to:

1. **Add Code Assistant**: Include `qiskit-code-assistant-mcp-server` for code generation
2. **Custom Metrics**: Modify chain scoring to weight different error sources
3. **Job Execution**: Actually run QV experiments on the recommended configuration
4. **Historical Analysis**: Track QV performance over time

### Troubleshooting

**"MCP server not found"**
- Ensure all MCP servers are installed: `pip install qiskit-mcp-servers`

**"Authentication failed"**
- Verify `QISKIT_IBM_TOKEN` is correct and has access to backends

**"Slow startup"**
- Set `QISKIT_IBM_RUNTIME_MCP_INSTANCE` for faster IBM Runtime initialization

**"Agent timeout"**
- Complex optimization may take several minutes
- Use `--interactive` mode for step-by-step analysis

## Individual Server Examples

Each MCP server also has its own examples directory with simpler LangChain agent demos:

- [`qiskit-mcp-server/examples/`](../qiskit-mcp-server/examples/) - Local transpilation
- [`qiskit-ibm-runtime-mcp-server/examples/`](../qiskit-ibm-runtime-mcp-server/examples/) - IBM Quantum Runtime
- [`qiskit-ibm-transpiler-mcp-server/examples/`](../qiskit-ibm-transpiler-mcp-server/examples/) - AI transpilation
- [`qiskit-code-assistant-mcp-server/examples/`](../qiskit-code-assistant-mcp-server/examples/) - Code generation
