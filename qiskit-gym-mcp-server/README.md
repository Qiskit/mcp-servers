# Qiskit Gym MCP Server

[![MCP Registry](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fregistry.modelcontextprotocol.io%2Fv0.1%2Fservers%2Fio.github.Qiskit%252Fqiskit-gym-mcp-server%2Fversions%2Flatest&query=%24.server.version&label=MCP%20Registry&logo=modelcontextprotocol)](https://registry.modelcontextprotocol.io/?q=io.github.Qiskit%2Fqiskit-gym-mcp-server)

<!-- mcp-name: io.github.Qiskit/qiskit-gym-mcp-server -->

A Model Context Protocol (MCP) server that provides reinforcement learning-based quantum circuit synthesis capabilities using [qiskit-gym](https://github.com/AI4quantum/qiskit-gym).

## Features

- **Train RL Models**: Train reinforcement learning agents to synthesize optimal quantum circuits
- **Background Training**: Run long training sessions in background threads with polling support
- **Three Synthesis Types**:
  - **Permutation**: Qubit routing with minimal SWAP gates
  - **Linear Function**: CNOT synthesis for linear Boolean functions
  - **Clifford**: Optimal Clifford circuit synthesis with custom gate sets
- **Hardware Support**: Presets for IBM Heron, Nighthawk, and common grid/linear topologies
- **Exact IBM Topologies**: Access exact coupling maps from IBM Quantum fake backends (offline, no credentials needed)
- **Subtopology Extraction**: Extract connected subgraphs from hardware coupling maps for targeted training
- **Model Persistence**: Save, load, and manage trained models
- **TensorBoard Integration**: Monitor training progress with TensorBoard

## Installation

```bash
pip install qiskit-gym-mcp-server
```

Or install from source:

```bash
git clone https://github.com/Qiskit/mcp-servers
cd mcp-servers/qiskit-gym-mcp-server
pip install -e .
```

## Quick Start

### Claude Desktop Configuration

Add to your Claude Desktop config (`~/Library/Application Support/Claude/claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "qiskit-gym": {
      "command": "qiskit-gym-mcp-server"
    }
  }
}
```

### Example Workflow

```
User: "Train a model to synthesize Clifford circuits on a 3x3 grid topology"

AI Agent:
1. create_clifford_env_tool(num_qubits=9, preset="grid_3x3")  # -> env_id
2. start_training_tool(env_id, algorithm="ppo", num_iterations=200)  # -> session_id, model_id
3. save_model_tool(session_id, model_name="clifford_3x3_v1")  # -> saved to disk
```

```
User: "Train on all 6-qubit subtopologies from IBM Nighthawk"

AI Agent:
1. extract_subtopologies_tool(preset="ibm_nighthawk", num_qubits=6)  # -> list of subtopologies
2. For each subtopology:
   - create_clifford_env_tool(num_qubits=6, coupling_map=subtopology["edges"])
   - start_training_tool(env_id, algorithm="ppo", num_iterations=100)
   - save_model_tool(session_id, model_name=f"nighthawk_6q_{i}")
```

```
User: "Train a model using the exact topology of IBM Fez backend"

AI Agent:
1. get_fake_backend_coupling_map_tool(backend_name="fake_fez")  # -> exact 156-qubit Heron topology
2. extract_subtopologies_tool(edges=coupling_map["edges"], num_qubits=5)  # -> 5-qubit subtopologies
3. For each subtopology:
   - create_permutation_env_tool(num_qubits=5, coupling_map=subtopology["edges"])
   - start_training_tool(env_id, algorithm="ppo", num_iterations=100)
```

```
User: "Train a model in the background so I can do other things"

AI Agent:
1. create_clifford_env_tool(num_qubits=6, preset="grid_2x3")  # -> env_id
2. start_training_tool(env_id, algorithm="ppo", num_iterations=500, background=True)  # -> session_id (returns immediately)
3. (User can now ask other questions or the agent can do other work)
4. get_training_status_tool(session_id)  # -> check progress
5. wait_for_training_tool(session_id, timeout=600)  # -> blocks until complete, returns model_id
6. save_model_tool(session_id, model_name="clifford_6q_v1")
```

## Tools Reference

### Environment Management

| Tool | Description |
|------|-------------|
| `create_permutation_env_tool` | Create PermutationGym for SWAP routing |
| `create_linear_function_env_tool` | Create LinearFunctionGym for CNOT synthesis |
| `create_clifford_env_tool` | Create CliffordGym with custom gate sets |
| `list_environments_tool` | List active environments |
| `get_environment_info_tool` | Get environment details |
| `delete_environment_tool` | Remove an environment |

### Training

| Tool | Description |
|------|-------------|
| `start_training_tool` | Start RL training (PPO or AlphaZero), supports `background=True` |
| `wait_for_training_tool` | Wait for background training to complete |
| `batch_train_environments_tool` | Train multiple environments |
| `get_training_status_tool` | Get training progress and metrics |
| `stop_training_tool` | Stop a training session |
| `list_training_sessions_tool` | List all training sessions |

### Synthesis

| Tool | Description |
|------|-------------|
| `synthesize_permutation_tool` | Generate optimal SWAP circuit |
| `synthesize_linear_function_tool` | Generate optimal CNOT circuit |
| `synthesize_clifford_tool` | Generate optimal Clifford circuit |

### Model Management

| Tool | Description |
|------|-------------|
| `save_model_tool` | Save trained model to disk |
| `load_model_tool` | Load model from disk |
| `list_saved_models_tool` | List models on disk |
| `list_loaded_models_tool` | List models in memory |
| `delete_model_tool` | Delete a model |
| `get_model_info_tool` | Get model details |

### Coupling Maps

| Tool | Description |
|------|-------------|
| `create_coupling_map_tool` | Create custom coupling map |
| `extract_subtopologies_tool` | Extract N-qubit subtopologies from hardware |
| `list_subtopology_shapes_tool` | List subtopology shapes (line, grid, etc.) |
| `get_fake_backend_coupling_map_tool` | Get exact topology from fake IBM backend (no credentials needed) |
| `list_available_fake_backends_tool` | List all available fake backends for offline topology access |

### Utility Tools

| Tool | Description |
|------|-------------|
| `generate_random_permutation_tool` | Generate random permutation for testing |
| `generate_random_linear_function_tool` | Generate random linear function for testing |
| `generate_random_clifford_tool` | Generate random Clifford element for testing |
| `convert_qpy_to_qasm3_tool` | Convert QPY circuit to human-readable QASM3 |
| `convert_qasm3_to_qpy_tool` | Convert QASM3 circuit to QPY format |

## Hardware Presets

| Preset | Qubits | Topology | Description |
|--------|--------|----------|-------------|
| `ibm_heron_r1` | 133 | Heavy-hex | IBM Heron r1 processor |
| `ibm_heron_r2` | 156 | Heavy-hex | IBM Heron r2 processor |
| `ibm_nighthawk` | 120 | 10x12 grid | IBM Nighthawk (600% depth improvement vs Heron) |
| `grid_3x3` | 9 | Grid | 3x3 square grid |
| `grid_5x5` | 25 | Grid | 5x5 square grid |
| `linear_5` | 5 | Line | 5-qubit linear chain |
| `linear_10` | 10 | Line | 10-qubit linear chain |

## Resources

| URI | Description |
|-----|-------------|
| `qiskit-gym://workflows` | **Start here** - Step-by-step workflows and quick start guide |
| `qiskit-gym://presets/coupling-maps` | Available hardware presets |
| `qiskit-gym://algorithms` | PPO, AlphaZero documentation |
| `qiskit-gym://policies` | BasicPolicy, Conv1dPolicy docs |
| `qiskit-gym://environments` | Environment type documentation |
| `qiskit-gym://training/sessions` | Active training sessions |
| `qiskit-gym://models` | Loaded models |
| `qiskit-gym://server/config` | Server configuration |

## Configuration

Environment variables:

```bash
# Model storage directory (default: ~/.qiskit-gym/models)
QISKIT_GYM_MODEL_DIR=~/.qiskit-gym/models

# TensorBoard logs (default: ~/.qiskit-gym/runs)
QISKIT_GYM_TENSORBOARD_DIR=~/.qiskit-gym/runs

# Training limits (0 = no limit, default)
# QISKIT_GYM_MAX_ITERATIONS=10000  # Uncomment to set a limit
QISKIT_GYM_MAX_QUBITS=15
QISKIT_GYM_MAX_SEARCHES=10000
```

## Development

```bash
# Install with dev dependencies
pip install -e ".[all]"

# Run tests
./run_tests.sh

# Or manually
uv run pytest tests/ -v
uv run ruff check src tests
uv run mypy src
```

## Dependencies

- [qiskit-gym](https://github.com/AI4quantum/qiskit-gym) - RL environments for quantum circuit synthesis
- [FastMCP](https://github.com/jlowin/fastmcp) - MCP server framework
- [Qiskit](https://qiskit.org/) - Quantum computing framework
- [qiskit-ibm-runtime](https://github.com/Qiskit/qiskit-ibm-runtime) - IBM Quantum access and fake backends

## License

Apache 2.0 - See [LICENSE](../LICENSE) for details.

## Contributing

See [CONTRIBUTING.md](../CONTRIBUTING.md) for contribution guidelines.
