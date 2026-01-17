# Qiskit Gym MCP Server Examples

This directory contains examples demonstrating how to build AI agents that interact with the **qiskit-gym-mcp-server** for reinforcement learning-based quantum circuit synthesis.

## Available Examples

| File | Description |
|------|-------------|
| `langchain_agent.ipynb` | **Jupyter Notebook** - Interactive tutorial with step-by-step examples |
| `langchain_agent.py` | **Python Script** - Command-line agent with multiple LLM provider support |

## LangChain Agent Example

The examples show how to create an AI agent using LangChain that connects to the qiskit-gym-mcp-server via the Model Context Protocol (MCP).

### Quick Start with Jupyter Notebook

For an interactive experience, open the notebook:

```bash
jupyter notebook langchain_agent.ipynb
```

The notebook includes:
- Step-by-step setup instructions
- Multiple LLM provider options (just run the cell for your provider)
- Sample workflows for training RL models
- Interactive examples for circuit synthesis
- Background training with progress monitoring

### Features

The agent can:

- **Create RL Environments**: Set up training environments for Permutation (SWAP routing), LinearFunction (CNOT synthesis), and Clifford circuits
- **Train Models**: Run training with PPO or AlphaZero algorithms, supports background training
- **Monitor Training**: Poll for progress, wait for completion with timeout
- **Synthesize Circuits**: Use trained models to generate optimal quantum circuits
- **Manage Models**: Save, load, list, and delete trained models
- **Hardware Topologies**: Access IBM Heron, Nighthawk, and common grid/linear presets

### Supported LLM Providers

| Provider | Package | Default Model | API Key Required |
|----------|---------|---------------|------------------|
| OpenAI | `langchain-openai` | gpt-4o | Yes (`OPENAI_API_KEY`) |
| Anthropic | `langchain-anthropic` | claude-sonnet-4-20250514 | Yes (`ANTHROPIC_API_KEY`) |
| Google | `langchain-google-genai` | gemini-2.5-pro | Yes (`GOOGLE_API_KEY`) |
| Ollama | `langchain-ollama` | llama3.2 | No (runs locally) |
| Watsonx | `langchain-ibm` | ibm/granite-3-8b-instruct | Yes (`WATSONX_APIKEY`, `WATSONX_PROJECT_ID`) |

### Architecture

```
┌─────────────┐     MCP Protocol     ┌──────────────────────────┐
│  LangChain  │ ◄──────────────────► │  qiskit-gym-mcp-server   │
│    Agent    │                      │                          │
└─────────────┘                      │  ┌────────────────────┐  │
                                     │  │    qiskit-gym      │  │
                                     │  │  (RL Environments) │  │
                                     │  └────────────────────┘  │
                                     └──────────────────────────┘
```

### Prerequisites

1. **Python 3.10+**

2. **Install the MCP server:**

```bash
pip install qiskit-gym-mcp-server
```

3. **Install LangChain dependencies:**

```bash
# Core dependencies
pip install langchain langchain-mcp-adapters python-dotenv

# Install ONE of the following based on your preferred LLM provider:
pip install langchain-openai       # For OpenAI
pip install langchain-anthropic    # For Anthropic Claude
pip install langchain-google-genai # For Google Gemini
pip install langchain-ollama       # For local Ollama
pip install langchain-ibm          # For IBM Watsonx
```

4. **Set up environment variables:**

```bash
# LLM API key (depends on provider)
export OPENAI_API_KEY="your-openai-api-key"       # For OpenAI
export ANTHROPIC_API_KEY="your-anthropic-api-key" # For Anthropic
export GOOGLE_API_KEY="your-google-api-key"       # For Google
# No API key needed for Ollama (runs locally)

# For Watsonx
export WATSONX_APIKEY="your-watsonx-api-key"
export WATSONX_PROJECT_ID="your-project-id"
export WATSONX_URL="https://us-south.ml.cloud.ibm.com"  # Optional, this is the default
```

Or create a `.env` file:

```env
OPENAI_API_KEY=your-openai-api-key

# For Watsonx
WATSONX_APIKEY=your-watsonx-api-key
WATSONX_PROJECT_ID=your-project-id
```

**Note:** This server doesn't require IBM Quantum credentials - it uses local qiskit-gym for RL training.

### Running the Example

**Interactive mode with OpenAI (default):**

```bash
cd examples
python langchain_agent.py
```

**With Anthropic Claude:**

```bash
python langchain_agent.py --provider anthropic
```

**With Google Gemini:**

```bash
python langchain_agent.py --provider google
```

**With local Ollama (no API key needed):**

```bash
# First, make sure Ollama is running with a model pulled
# ollama pull llama3.2
python langchain_agent.py --provider ollama --model llama3.2
```

**With IBM Watsonx:**

```bash
python langchain_agent.py --provider watsonx
# Or with a specific model
python langchain_agent.py --provider watsonx --model ibm/granite-3-8b-instruct
```

**Single query mode:**

```bash
python langchain_agent.py --single
python langchain_agent.py --provider anthropic --single
```

**Custom model:**

```bash
python langchain_agent.py --provider openai --model gpt-4-turbo
python langchain_agent.py --provider anthropic --model claude-3-haiku-20240307
```

### Example Interactions

Once running, you can ask the agent questions like:

- "Create a permutation environment for a 5-qubit linear chain"
- "Train a model with PPO for 50 iterations"
- "Start training in the background with 200 iterations"
- "Check the training status"
- "Synthesize a random permutation using the trained model"
- "Save the trained model as 'my_permutation_model'"
- "List all available models"

### Example Workflows

**Basic Training Workflow (with background=True):**
```
User: "Train a model to route qubits on a linear 5-qubit topology"

Agent:
1. create_permutation_env_tool(preset="linear_5")
2. start_training_tool(env_id, algorithm="ppo", num_iterations=100, background=True)
   -> Returns immediately with session_id
3. wait_for_training_tool(session_id, timeout=600)
4. save_model_tool(session_id, model_name="linear_5_router")
```

**Batch Training Multiple Subtopologies:**
```
User: "Train linear function models for all 4-qubit subtopologies in IBM Nighthawk"

Agent:
1. list_subtopology_shapes_tool(preset="ibm_nighthawk", num_qubits=4)
   -> Returns list of unique 4-qubit topologies
2. For each shape, create_linear_function_env_tool(coupling_map=edges)
3. batch_train_environments_tool(env_ids, num_iterations=100, background=True)
   -> Returns immediately with all session_ids
4. get_training_status_tool(session_id) for each to check progress
5. save_model_tool for each completed session
```

**Synthesis Workflow:**
```
User: "Synthesize an optimal circuit for a random permutation"

Agent:
1. load_model_tool(model_name="linear_5_router")
2. generate_random_permutation_tool(num_qubits=5)
3. synthesize_permutation_tool(model_id, permutation)
   -> Returns optimal SWAP circuit
```

### Available MCP Tools

The agent has access to these tools provided by the MCP server:

**Environment Management:**

| Tool | Description |
|------|-------------|
| `create_permutation_env_tool` | Create PermutationGym for SWAP routing |
| `create_linear_function_env_tool` | Create LinearFunctionGym for CNOT synthesis |
| `create_clifford_env_tool` | Create CliffordGym with custom gate sets |
| `list_environments_tool` | List active environments |
| `get_environment_info_tool` | Get environment details |
| `delete_environment_tool` | Remove an environment |

**Training:**

| Tool | Description |
|------|-------------|
| `start_training_tool` | Start RL training (PPO or AlphaZero), supports `background=True` |
| `wait_for_training_tool` | Wait for background training to complete |
| `batch_train_environments_tool` | Train multiple environments |
| `get_training_status_tool` | Get training progress and metrics |
| `stop_training_tool` | Stop a training session |
| `list_training_sessions_tool` | List all training sessions |

**Synthesis:**

| Tool | Description |
|------|-------------|
| `synthesize_permutation_tool` | Generate optimal SWAP circuit |
| `synthesize_linear_function_tool` | Generate optimal CNOT circuit |
| `synthesize_clifford_tool` | Generate optimal Clifford circuit |

**Model Management:**

| Tool | Description |
|------|-------------|
| `save_model_tool` | Save trained model to disk |
| `load_model_tool` | Load model from disk |
| `list_saved_models_tool` | List models on disk |
| `list_loaded_models_tool` | List models in memory |
| `delete_model_tool` | Delete a model |
| `get_model_info_tool` | Get model details |

### Hardware Presets

| Preset | Qubits | Topology | Description |
|--------|--------|----------|-------------|
| `ibm_heron_r1` | 133 | Heavy-hex | IBM Heron r1 processor |
| `ibm_heron_r2` | 156 | Heavy-hex | IBM Heron r2 processor |
| `ibm_nighthawk` | 120 | 10x12 grid | IBM Nighthawk (better for grid algorithms) |
| `grid_3x3` | 9 | Grid | 3x3 square grid |
| `grid_5x5` | 25 | Grid | 5x5 square grid |
| `linear_5` | 5 | Line | 5-qubit linear chain |
| `linear_10` | 10 | Line | 10-qubit linear chain |

### RL Algorithms

| Algorithm | Description | Recommended For |
|-----------|-------------|-----------------|
| `ppo` | Proximal Policy Optimization | Most cases, fast training |
| `alphazero` | MCTS with neural network guidance | Complex problems, slower |

### Policy Networks

| Policy | Description | Recommended For |
|--------|-------------|-----------------|
| `basic` | Simple feedforward network | Small problems (< 8 qubits) |
| `conv1d` | 1D convolutional network | Larger problems |

### Using as a Library

You can import and use the agent in your own async code:

```python
import asyncio
from langchain_agent import (
    get_mcp_client,
    create_gym_agent_with_session,
    run_agent_query,
)

async def main():
    # Use persistent session for efficient tool calls
    mcp_client = get_mcp_client()
    async with mcp_client.session("qiskit-gym") as session:
        agent = await create_gym_agent_with_session(session, provider="openai")

        # Run queries
        response = await run_agent_query(
            agent,
            "Create a permutation environment for a 5-qubit linear chain and train a model"
        )
        print(response)

asyncio.run(main())
```

### Troubleshooting

**"MCP server not found"**
- Ensure `qiskit-gym-mcp-server` is installed and available in your PATH
- Try running `qiskit-gym-mcp-server` directly to verify installation

**"qiskit-gym not installed"**
- Install the qiskit-gym package: `pip install qiskit-gym`

**"Training slow"**
- Use fewer iterations for testing (10-50)
- Use `background=True` for long training sessions
- Consider using `basic` policy for smaller problems

**"Model not found"**
- Use `list_loaded_models_tool` to see models in memory
- Use `list_saved_models_tool` to see models on disk
- Make sure to save models with `save_model_tool` for persistence
