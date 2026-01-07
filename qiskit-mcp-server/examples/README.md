# Qiskit MCP Server Examples

This directory contains examples demonstrating how to build AI agents that interact with Qiskit's transpiler through the **qiskit-mcp-server**.

## Available Examples

| File | Description |
|------|-------------|
| [`langchain_agent.ipynb`](langchain_agent.ipynb) | **Jupyter Notebook** - Interactive tutorial with step-by-step examples |
| [`langchain_agent.py`](langchain_agent.py) | **Python Script** - Command-line agent with multiple LLM provider support |

## LangChain Agent Example

The examples show how to create an AI agent using LangChain that connects to the qiskit-mcp-server via the Model Context Protocol (MCP).

### Quick Start with Jupyter Notebook

For an interactive experience, open the notebook:

```bash
jupyter notebook langchain_agent.ipynb
```

The notebook includes:
- Step-by-step setup instructions
- Multiple LLM provider options (just run the cell for your provider)
- Sample QASM circuits for testing
- Interactive examples for transpilation and analysis
- A custom query cell for your own circuits

### Features

The agent can:

- Transpile quantum circuits with configurable optimization levels (0-3)
- Analyze circuit structure and complexity
- Compare optimization levels to find the best settings
- Target different hardware backends (IBM Eagle, Heron, etc.)
- Apply different topologies (linear, ring, grid, heavy_hex, full)

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
│  LangChain  │ ◄──────────────────► │    qiskit-mcp-server     │
│    Agent    │                      │                          │
└─────────────┘                      │  ┌────────────────────┐  │
                                     │  │  Qiskit Transpiler │  │
                                     │  └────────────────────┘  │
                                     └──────────────────────────┘
```

### Prerequisites

1. **Python 3.10+**

2. **Install the MCP server:**

```bash
pip install qiskit-mcp-server
```

3. **Install LangChain dependencies:**

```bash
# Core dependencies
pip install langchain langchain-mcp-adapters python-dotenv

# Install at least ONE of the following based on your preferred LLM provider(s):
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

**Note:** This server doesn't require IBM Quantum credentials - it uses local Qiskit transpilation.

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

- "Transpile my bell circuit for IBM Heron"
- "Analyze my ghz circuit"
- "Compare optimization levels for my qft circuit"
- "Transpile this circuit with linear topology"
- "What's the depth of this circuit after transpilation?"

The agent comes with sample circuits built-in:
- `bell`: 2-qubit Bell state circuit
- `ghz`: 4-qubit GHZ state circuit
- `qft`: 3-qubit QFT circuit

### Available MCP Tools

The agent has access to these tools provided by the MCP server:

| Tool | Description |
|------|-------------|
| `transpile_circuit_tool` | Transpile a circuit with configurable optimization |
| `analyze_circuit_tool` | Analyze circuit structure without transpiling |
| `compare_optimization_levels_tool` | Compare all optimization levels (0-3) |

### Basis Gate Presets

| Preset | Gates | Description |
|--------|-------|-------------|
| `ibm_eagle` | id, rz, sx, x, ecr, reset | IBM Eagle r3 (127 qubits, uses ECR) |
| `ibm_heron` | id, rz, sx, x, cz, reset | IBM Heron (133-156 qubits, uses CZ) |
| `ibm_legacy` | id, rz, sx, x, cx, reset | Older IBM systems (uses CX) |

### Available Topologies

| Topology | Description |
|----------|-------------|
| `linear` | Chain connectivity (qubit i ↔ i+1) |
| `ring` | Linear with wraparound |
| `grid` | 2D grid connectivity |
| `heavy_hex` | IBM heavy-hex topology |
| `full` | All-to-all connectivity |

### Optimization Levels

| Level | Description | Use Case |
|-------|-------------|----------|
| 0 | No optimization, only basis gate decomposition | Quick iterations, debugging |
| 1 | Light optimization with default layout | Development, prototyping |
| 2 | Medium optimization with noise-aware layout | Production use (recommended) |
| 3 | Heavy optimization for best results | Critical applications, small circuits |

### Using as a Library

You can import and use the agent in your own async code:

```python
import asyncio
from langchain_agent import (
    get_mcp_client,
    create_transpiler_agent_with_session,
    run_agent_query,
    SAMPLE_BELL_STATE,
)

async def main():
    # Use persistent session for efficient tool calls
    mcp_client = get_mcp_client()
    async with mcp_client.session("qiskit") as session:
        agent = await create_transpiler_agent_with_session(session, provider="openai")

        # Run queries
        query = f"Transpile this circuit for IBM Heron:\n{SAMPLE_BELL_STATE}"
        response = await run_agent_query(agent, query)
        print(response)

asyncio.run(main())
```

### Customizing the Agent

You can modify the system prompt or use a different LLM by creating your own agent setup:

```python
import asyncio
from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_openai import ChatOpenAI

async def create_custom_agent():
    # Configure MCP client
    mcp_client = MultiServerMCPClient({
        "qiskit": {
            "transport": "stdio",
            "command": "qiskit-mcp-server",
            "args": [],
            "env": {},
        }
    })

    # Use persistent session for efficient tool calls
    async with mcp_client.session("qiskit") as session:
        tools = await load_mcp_tools(session)

        # Custom system prompt
        system_prompt = "You are a quantum circuit optimization expert..."

        llm = ChatOpenAI(model="gpt-4o", temperature=0)
        agent = create_agent(llm, tools, system_prompt=system_prompt)

        # Use the agent within the session context
        # ... your agent logic here ...

asyncio.run(create_custom_agent())
```

### Troubleshooting

**"MCP server not found"**
- Ensure `qiskit-mcp-server` is installed and available in your PATH
- Try running `qiskit-mcp-server` directly to verify installation

**"Invalid QASM"**
- Ensure your QASM circuit is valid QASM 3.0 or QASM 2.0 syntax
- Include the `include "stdgates.inc";` line for standard gates

**"Transpilation slow"**
- Use optimization level 2 instead of 3 for larger circuits
- Level 3 can be very slow for circuits with >20 qubits or >500 gates
- Use the `compare_optimization_levels` tool to find the best level for your circuit
