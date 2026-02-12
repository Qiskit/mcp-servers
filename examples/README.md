# Qiskit MCP Servers - Advanced Examples

This directory contains examples demonstrating the full power of combining multiple Qiskit MCP servers with advanced AI agent frameworks.

## Quantum Volume Finder

**A Deep Agent for Actual QV Measurement**

The Quantum Volume Finder is a multi-agent system that **finds the highest achievable Quantum Volume (QV)** for IBM Quantum backends through **actual hardware execution**.

Unlike simple analysis tools, this agent:
- **Runs experiments** on real quantum hardware
- **Reports actual results** (HOP values, job IDs)
- Uses **top-down search**: starts at max depth, works down until success
- Searches **ALL qubits** on the backend (not just first 10)

### What is Quantum Volume?

Quantum Volume (QV) 2^n is **achieved** when:
- Running n-qubit, depth-n random circuits
- Heavy Output Probability (HOP) > 2/3
- HOP = (shots resulting in heavy outputs) / (total shots)

### Two Modes

| Mode | Flag | Description |
|------|------|-------------|
| **Single-circuit** (default) | — | Quick test: 1 circuit per depth, HOP > 2/3 check |
| **Full protocol** | `--num-circuits 100` | Statistically rigorous: N circuits per depth, 97.5% CI test |

The single-circuit mode is a fast scout. For official QV certification, use `--num-circuits 100` (or more) which implements the full protocol from arXiv:1811.12926.

### Strategy: Top-Down Search

```
Start at depth 5 (QV-32)
    │
    ├─► Find optimal qubits (searches ALL qubits)
    ├─► Transpile circuit (hybrid_ai_transpile_tool)
    ├─► Submit to hardware → poll for completion
    ├─► Calculate HOP
    │
    ├─► HOP > 2/3? ──YES──► SUCCESS! QV-32 achieved
    │       │
    │      NO
    │       │
    ▼       ▼
Try depth 4 (QV-16)
    │
    ... repeat until success or depth 2
```

### Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    QUANTUM VOLUME FINDER                            │
│                      (Coordinator Agent)                            │
│                                                                     │
│  Implements top-down search: try max depth, work down until PASS   │
└─────────────────────────────────────────────────────────────────────┘
                                   │
       ┌───────────────────────────┼───────────────────────────┐
       │                           │                           │
       ▼                           ▼                           ▼
┌─────────────────┐ ┌─────────────────────┐ ┌─────────────────────┐
│  BACKEND        │ │ QUBIT CHAIN         │ │ QV EXPERIMENT       │
│  ANALYST        │ │ OPTIMIZER           │ │ RUNNER              │
│                 │ │                     │ │                     │
│ Get backend     │ │ Searches ALL qubits │ │ Transpile circuit   │
│ properties      │ │ on backend          │ │ Submit job          │
│                 │ │                     │ │ Poll for completion │
└─────────────────┘ └─────────────────────┘ └─────────────────────┘
```

### Available Examples

| File | Description |
|------|-------------|
| `quantum_volume_optimizer.py` | Command-line QV finder with iterative experiments |
| `quantum_volume_optimizer.ipynb` | Interactive Jupyter notebook version |

### Prerequisites

```bash
# Install Deep Agents and LangChain
pip install deepagents langchain langchain-mcp-adapters python-dotenv

# Install your LLM provider
pip install langchain-anthropic

# Install all Qiskit MCP servers
pip install qiskit-mcp-servers

# Required for --num-circuits statistical analysis
pip install scipy
```

### Environment Variables

```bash
# Required
export QISKIT_IBM_TOKEN="your-ibm-quantum-token"
export ANTHROPIC_API_KEY="your-anthropic-api-key"

# Optional (faster startup)
export QISKIT_IBM_RUNTIME_MCP_INSTANCE="your-instance-crn"
```

### Running the QV Finder

**Find highest QV (runs experiments by default):**

```bash
# Single-circuit quick test
python quantum_volume_optimizer.py --backend ibm_brisbane --depth 5

# Full QV protocol with 100 circuits per depth
python quantum_volume_optimizer.py --backend ibm_brisbane --depth 5 --num-circuits 100
```

**Command-line options:**

| Option | Description | Default |
|--------|-------------|---------|
| `--backend BACKEND` | Backend to test (required for experiments) | Auto-select |
| `--depth N` | Maximum QV depth to try (2-20) | 5 |
| `--num-circuits N` | Independent circuits per depth (1=quick, 100+=full protocol) | 1 |
| `--no-experiment` | Analysis only, no hardware execution | Runs experiments |
| `--quiet` | Disable verbose activity logging | Verbose on |
| `--interactive` | Interactive mode for follow-ups | Off |
| `--provider` | LLM: `anthropic`, `openai`, `google` | `anthropic` |
| `--model MODEL` | Model name (provider-specific) | Provider default |

**Examples:**

```bash
# Try up to QV-256 (depth 8)
python quantum_volume_optimizer.py --backend ibm_brisbane --depth 8

# Full protocol with 50 circuits (faster than 100, still statistically meaningful)
python quantum_volume_optimizer.py --backend ibm_brisbane --depth 5 --num-circuits 50

# Analysis only (no hardware)
python quantum_volume_optimizer.py --backend ibm_brisbane --no-experiment

# Interactive mode
python quantum_volume_optimizer.py --backend ibm_brisbane --interactive
```

### Expected Output

**Single-circuit mode** — the agent reports actual execution results:

```
## QV EXPERIMENT RESULTS

### Depth 5 (First Attempt)
- Backend: ibm_brisbane
- Qubits: [45, 46, 47, 52, 53]
- Job ID: d5jm8tivcahs73a0uf70
- Shots: 4096
- HOP: 0.600
- Result: FAIL (HOP <= 0.667)

### Depth 4 (Second Attempt)
- Backend: ibm_brisbane
- Qubits: [45, 46, 47, 52]
- Job ID: d5jm8tivcahs73a0uf71
- Shots: 4096
- HOP: 0.681
- Result: PASS (HOP > 0.667)

## CONCLUSION
Highest Achieved QV: 2^4 = 16
```

**Multi-circuit mode** (`--num-circuits 100`) — includes statistical analysis:

```
## QV EXPERIMENT RESULTS (Full Protocol)

### Depth 5 (100 circuits)
- Backend: ibm_brisbane
- Qubits: [45, 46, 47, 52, 53]
- Mean HOP: 0.612
- CI Lower Bound: 0.598
- QV Achieved: NO (CI lower bound <= 0.667)

### Depth 4 (100 circuits)
- Backend: ibm_brisbane
- Qubits: [45, 46, 47, 52]
- Mean HOP: 0.701
- CI Lower Bound: 0.689
- QV Achieved: YES (CI lower bound > 0.667)

## CONCLUSION
Highest Achieved QV: 2^4 = 16
Protocol: Full (100 circuits, 97.5% CI)
```

### Key MCP Tools

| Tool | Purpose |
|------|---------|
| `find_optimal_qv_qubits_tool` | Finds best qubit subgraphs (searches ALL qubits) |
| `hybrid_ai_transpile_tool` | AI-powered circuit transpilation (accepts `backend_name`) |
| `run_sampler_tool` | Submits transpiled circuit to hardware |
| `get_job_status_tool` | Polls until job is DONE |
| `get_job_results_tool` | Retrieves measurement counts |

### Local Tools & Helper Functions

| Tool/Function | Purpose |
|---------------|---------|
| `transpile_qv_circuit` | Generates QV circuit and transpiles via MCP (keeps large data out of LLM) |
| `submit_qv_job` | Submits transpiled circuit to hardware via MCP |
| `calculate_hop` | Fetches job results and computes Heavy Output Probability |
| `run_qv_depth_trial` | Full protocol batch tool: N circuits → transpile → submit → poll → HOP → CI test |
| `generate_qv_circuit_with_ideal_distribution()` | Creates QV circuit + ideal heavy outputs |
| `calculate_heavy_output_probability()` | Calculates HOP from counts and heavy outputs |
| `analyze_qv_experiment_results()` | Statistical CI test for multi-circuit QV validation |

### Key Improvements Over Basic Analysis

1. **Actual Execution**: Runs circuits on real hardware, not just recommendations
2. **Iterative Search**: Automatically tries lower depths if higher ones fail
3. **All Qubits**: `find_optimal_qv_qubits_tool` searches entire backend
4. **AI Transpilation**: Uses `hybrid_ai_transpile_tool` for optimized circuit mapping
5. **Full QV Protocol**: Optional `--num-circuits` for statistically rigorous certification
6. **Efficient Data Flow**: Large data (QASM, QPY, counts) stays in local tools, never through LLM

### Troubleshooting

**"Job stuck in QUEUED"**
- Quantum jobs can queue for minutes to hours
- Use least busy backends or wait

**"HOP always below threshold"**
- Try lower depth (--depth 3 or --depth 2)
- Hardware noise affects larger circuits more

**"MCP server not found"**
- Install servers: `pip install qiskit-mcp-servers`

## Individual Server Examples

Each MCP server has simpler examples in its own directory:

- [`qiskit-mcp-server/examples/`](../qiskit-mcp-server/examples/) - Local transpilation
- [`qiskit-ibm-runtime-mcp-server/examples/`](../qiskit-ibm-runtime-mcp-server/examples/) - IBM Quantum Runtime
- [`qiskit-ibm-transpiler-mcp-server/examples/`](../qiskit-ibm-transpiler-mcp-server/examples/) - AI transpilation
- [`qiskit-code-assistant-mcp-server/examples/`](../qiskit-code-assistant-mcp-server/examples/) - Code generation
