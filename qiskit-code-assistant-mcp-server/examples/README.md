# Qiskit Code Assistant MCP Server Examples

This directory contains examples demonstrating how to build AI agents that interact with Qiskit Code Assistant through the **qiskit-code-assistant-mcp-server**.

## Available Examples

| File | Description |
|------|-------------|
| [`langchain_agent.ipynb`](langchain_agent.ipynb) | **Jupyter Notebook** - Interactive tutorial with step-by-step examples |
| [`langchain_agent.py`](langchain_agent.py) | **Python Script** - Command-line agent with multiple LLM provider support |

## LangChain Agent Example

The examples show how to create an AI agent using LangChain that connects to the qiskit-code-assistant-mcp-server via the Model Context Protocol (MCP).

### Quick Start with Jupyter Notebook

For an interactive experience, open the notebook:

```bash
jupyter notebook langchain_agent.ipynb
```

The notebook includes:
- Step-by-step setup instructions
- Multiple LLM provider options (just run the cell for your provider)
- Interactive examples for generating quantum code and asking questions
- A custom query cell for your own questions

### Features

The agent can:

- Generate quantum code completions using Qiskit Code Assistant
- Answer questions about Qiskit and quantum computing concepts (RAG mode)
- Accept model disclaimers when required
- Track completion acceptance for feedback

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
┌─────────────┐     MCP Protocol     ┌──────────────────────────────────┐
│  LangChain  │ ◄──────────────────► │ qiskit-code-assistant-mcp-server │
│    Agent    │                      │                                  │
└─────────────┘                      │  ┌────────────────────────────┐  │
                                     │  │  Qiskit Code Assistant API │  │
                                     │  └────────────────────────────┘  │
                                     │               │                  │
                                     └───────────────│──────────────────┘
                                                     ▼
                                            ┌─────────────────┐
                                            │  IBM Quantum    │
                                            │  Code Assistant │
                                            └─────────────────┘
```

### Prerequisites

1. **Python 3.10+**

2. **Install the MCP server:**

```bash
pip install qiskit-code-assistant-mcp-server
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

- "Write a quantum circuit that creates a Bell state"
- "What is quantum entanglement and why is it important?"
- "Generate code to set up a simple VQE algorithm"
- "How does Qiskit's transpiler work?"
- "Write a circuit with 3 qubits that applies a Toffoli gate"

### Available MCP Tools

The agent has access to these tools provided by the MCP server:

| Tool | Description |
|------|-------------|
| `get_completion_tool` | Generate Qiskit code completions based on prompts |
| `get_rag_completion_tool` | Answer questions about Qiskit and quantum computing using RAG |
| `accept_model_disclaimer_tool` | Accept the disclaimer for a model |
| `accept_completion_tool` | Accept/acknowledge a generated completion |

### Using as a Library

You can import and use the agent in your own async code:

```python
import asyncio
from langchain_agent import (
    get_mcp_client,
    create_quantum_agent_with_session,
    run_agent_query,
)

async def main():
    # Use persistent session for efficient tool calls
    mcp_client = get_mcp_client()
    async with mcp_client.session("qiskit-code-assistant") as session:
        agent = await create_quantum_agent_with_session(session, provider="openai")

        # Run queries
        response = await run_agent_query(agent, "Write a Bell state circuit")
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
        "qiskit-code-assistant": {
            "transport": "stdio",
            "command": "qiskit-code-assistant-mcp-server",
            "args": [],
            "env": {
                "QISKIT_IBM_TOKEN": os.getenv("QISKIT_IBM_TOKEN", ""),
            },
        }
    })

    # Use persistent session for efficient tool calls
    async with mcp_client.session("qiskit-code-assistant") as session:
        tools = await load_mcp_tools(session)

        # Custom system prompt
        system_prompt = "You are a quantum computing expert assistant..."

        llm = ChatOpenAI(model="gpt-4o", temperature=0)
        agent = create_agent(llm, tools, system_prompt=system_prompt)

        # Use the agent within the session context
        # ... your agent logic here ...

asyncio.run(create_custom_agent())
```

### Troubleshooting

**"MCP server not found"**
- Ensure `qiskit-code-assistant-mcp-server` is installed and available in your PATH
- Try running `qiskit-code-assistant-mcp-server` directly to verify installation

**"Authentication failed"**
- Verify your `QISKIT_IBM_TOKEN` is correct
- Check that your IBM Quantum account has access to Qiskit Code Assistant

**"Connection timeout"**
- The MCP server may take a few seconds to start
- Check your network connection to IBM Quantum services

**"Model disclaimer required"**
- The agent will automatically use `accept_model_disclaimer_tool` when needed
- You can also manually accept the disclaimer through the tool
