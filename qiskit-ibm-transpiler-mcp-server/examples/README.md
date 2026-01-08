# Qiskit IBM Transpiler MCP Server Examples

This directory contains examples demonstrating how to build AI agents that interact with IBM Quantum's AI-powered transpiler through the **qiskit-ibm-transpiler-mcp-server**.

## Available Examples

| File | Description |
|------|-------------|
| [`langchain_agent.ipynb`](langchain_agent.ipynb) | **Jupyter Notebook** - Interactive tutorial with step-by-step examples |
| [`langchain_agent.py`](langchain_agent.py) | **Python Script** - Command-line agent with multiple LLM provider support |

## LangChain Agent Example

The examples show how to create an AI agent using LangChain that connects to the qiskit-ibm-transpiler-mcp-server via the Model Context Protocol (MCP).

### Quick Start with Jupyter Notebook

For an interactive experience, open the notebook:

```bash
jupyter notebook langchain_agent.ipynb
```

The notebook includes:
- Step-by-step setup instructions
- Multiple LLM provider options (just run the cell for your provider)
- Sample QASM circuits for testing
- Interactive examples for routing and synthesis
- A custom query cell for your own circuits

### Features

The agent can:

- Perform AI routing on quantum circuits
- Synthesize Clifford circuits (H, S, CX gates)
- Synthesize Linear Function circuits (CX, SWAP gates)
- Synthesize Permutation circuits (SWAP gates)
- Synthesize Pauli Network circuits (H, S, SX, CX, RX, RY, RZ gates)

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
┌─────────────┐     MCP Protocol     ┌───────────────────────────────────┐
│  LangChain  │ ◄──────────────────► │ qiskit-ibm-transpiler-mcp-server  │
│    Agent    │                      │                                   │
└─────────────┘                      │  ┌─────────────────────────────┐  │
                                     │  │   qiskit-ibm-transpiler     │  │
                                     │  └─────────────────────────────┘  │
                                     │               │                   │
                                     └───────────────│───────────────────┘
                                                     ▼
                                            ┌─────────────────┐
                                            │  IBM Quantum    │
                                            │  AI Transpiler  │
                                            └─────────────────┘
```

### Prerequisites

1. **Python 3.10+ and <3.14**

2. **Install the MCP server:**

```bash
pip install qiskit-ibm-transpiler-mcp-server
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
# IBM Quantum token (required)
export QISKIT_IBM_TOKEN="your-ibm-quantum-token"

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
QISKIT_IBM_TOKEN=your-ibm-quantum-token
OPENAI_API_KEY=your-openai-api-key

# For Watsonx
WATSONX_APIKEY=your-watsonx-api-key
WATSONX_PROJECT_ID=your-project-id
```

### Getting Your IBM Quantum Token

1. Create an account at [IBM Quantum](https://quantum.ibm.com/)
2. Navigate to your account settings
3. Copy your API token

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

- "Route my bell state circuit for ibm_brisbane"
- "Optimize my clifford circuit using AI synthesis"
- "Apply AI routing and then Clifford synthesis to this circuit"
- "What AI synthesis passes are available?"
- "Optimize this Linear Function circuit for ibm_fez"

The agent comes with sample circuits built-in:
- `clifford`: 3-qubit Clifford circuit (H, CX, S gates)
- `linear`: 3-qubit Linear Function circuit (CX, SWAP gates)
- `bell`: 2-qubit Bell state circuit

### Available MCP Tools

The agent has access to these tools provided by the MCP server:

| Tool | Description |
|------|-------------|
| `setup_ibm_quantum_account_tool` | Configure IBM Quantum account with API token |
| `ai_routing` | Route circuits for backend coupling maps (insert SWAPs) |
| `ai_clifford_synthesis` | Optimize Clifford circuits (H, S, CX gates) - up to 9 qubits |
| `ai_linear_function_synthesis` | Optimize Linear Function circuits (CX, SWAP gates) - up to 9 qubits |
| `ai_permutation_synthesis` | Optimize Permutation circuits (SWAP gates) - 27, 33, 65 qubits |
| `ai_pauli_network_synthesis` | Optimize Pauli Network circuits (H, S, SX, CX, RX, RY, RZ gates) - up to 6 qubits |

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
    async with mcp_client.session("qiskit-ibm-transpiler") as session:
        agent = await create_transpiler_agent_with_session(session, provider="openai")

        # Run queries
        query = f"Route this circuit for ibm_brisbane:\n{SAMPLE_BELL_STATE}"
        response = await run_agent_query(agent, query)
        print(response)

asyncio.run(main())
```

### Customizing the Agent

You can modify the system prompt or use a different LLM by creating your own agent setup:

```python
import asyncio
import os
from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_openai import ChatOpenAI

async def create_custom_agent():
    # Configure MCP client
    mcp_client = MultiServerMCPClient({
        "qiskit-ibm-transpiler": {
            "transport": "stdio",
            "command": "qiskit-ibm-transpiler-mcp-server",
            "args": [],
            "env": {
                "QISKIT_IBM_TOKEN": os.getenv("QISKIT_IBM_TOKEN", ""),
            },
        }
    })

    # Use persistent session for efficient tool calls
    async with mcp_client.session("qiskit-ibm-transpiler") as session:
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
- Ensure `qiskit-ibm-transpiler-mcp-server` is installed and available in your PATH
- Try running `qiskit-ibm-transpiler-mcp-server` directly to verify installation

**"Authentication failed"**
- Verify your `QISKIT_IBM_TOKEN` is correct
- Check that your IBM Quantum account is active

**"Connection timeout"**
- The MCP server may take a few seconds to start
- Check your network connection to IBM Quantum services

**"Circuit not optimizing"**
- Ensure your QASM circuit is valid QASM 3.0 syntax
- Check that the circuit contains gates supported by the synthesis pass
- Try a different backend (ibm_brisbane, ibm_fez, ibm_sherbrooke)
