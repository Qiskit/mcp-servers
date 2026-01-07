# Qiskit IBM Runtime MCP Server Examples

This directory contains examples demonstrating how to build AI agents that interact with IBM Quantum services through the **qiskit-ibm-runtime-mcp-server**.

## Available Examples

| File | Description |
|------|-------------|
| [`langchain_agent.ipynb`](langchain_agent.ipynb) | **Jupyter Notebook** - Interactive tutorial with step-by-step examples |
| [`langchain_agent.py`](langchain_agent.py) | **Python Script** - Command-line agent with multiple LLM provider support |

## LangChain Agent Example

The examples show how to create an AI agent using LangChain that connects to the qiskit-ibm-runtime-mcp-server via the Model Context Protocol (MCP).

### Quick Start with Jupyter Notebook

For an interactive experience, open the notebook:

```bash
jupyter notebook langchain_agent.ipynb
```

The notebook includes:
- Step-by-step setup instructions
- Multiple LLM provider options (just run the cell for your provider)
- Interactive examples for listing backends, finding the least busy backend, and more
- A custom query cell for your own questions

### Features

The agent can:

- Set up IBM Quantum account credentials
- List available quantum backends
- Find the least busy backend
- Get detailed backend properties
- List and manage quantum jobs

### Supported LLM Providers

| Provider | Package | Default Model | API Key Required |
|----------|---------|---------------|------------------|
| OpenAI | `langchain-openai` | gpt-4o | Yes (`OPENAI_API_KEY`) |
| Anthropic | `langchain-anthropic` | claude-sonnet-4-20250514 | Yes (`ANTHROPIC_API_KEY`) |
| Google | `langchain-google-genai` | gemini-1.5-pro | Yes (`GOOGLE_API_KEY`) |
| Ollama | `langchain-ollama` | llama3.2 | No (runs locally) |
| Watsonx | `langchain-ibm` | ibm/granite-3-8b-instruct | Yes (`WATSONX_APIKEY`, `WATSONX_PROJECT_ID`) |
| OpenAI-compatible | `langchain-openai` | granite-qiskit | Yes (`OPENAI_COMPATIBLE_BASE_URL`) |
| OpenAI-completions | `langchain-openai` | granite-qiskit | Yes (`OPENAI_COMPATIBLE_BASE_URL`)* |

> **Note:** The `openai-completions` provider uses the legacy `/completions` endpoint which does not support native tool calling. Agent functionality may be limited.

### Architecture

```
┌─────────────┐     MCP Protocol     ┌──────────────────────────────────┐
│  LangChain  │ ◄──────────────────► │ qiskit-ibm-runtime-mcp-server    │
│    Agent    │                      │                                  │
└─────────────┘                      │  ┌────────────────────────────┐  │
                                     │  │   qiskit-ibm-runtime       │  │
                                     │  └────────────────────────────┘  │
                                     │               │                  │
                                     └───────────────│──────────────────┘
                                                     ▼
                                            ┌─────────────────┐
                                            │  IBM Quantum    │
                                            │    Cloud        │
                                            └─────────────────┘
```

### Prerequisites

1. **Python 3.10+**

2. **Install the MCP server:**

```bash
pip install qiskit-ibm-runtime-mcp-server
```

3. **Install LangChain dependencies:**

```bash
# Core dependencies
pip install langchain langchain-mcp-adapters python-dotenv

# Install at least ONE of the following based on your preferred LLM provider(s):
pip install langchain-openai       # For OpenAI, OpenAI-compatible, OpenAI-completions
pip install langchain-anthropic    # For Anthropic Claude
pip install langchain-google-genai # For Google Gemini
pip install langchain-ollama       # For local Ollama
pip install langchain-ibm          # For IBM Watsonx
```

4. **Set up environment variables:**

```bash
# IBM Quantum token (required for all providers)
export QISKIT_IBM_TOKEN="your-ibm-quantum-token"

# IBM Quantum instance (HIGHLY RECOMMENDED for faster startup)
# Without this, the service searches all instances which is slow (~10-30 seconds).
# With instance set, startup is ~2-3 seconds.
# Find your available instances by running:
#   python -c "from qiskit_ibm_runtime import QiskitRuntimeService; s = QiskitRuntimeService(); print([i['name'] for i in s.instances()])"
# Or at: https://quantum.ibm.com/ -> Administration -> Instances
export QISKIT_IBM_RUNTIME_MCP_INSTANCE="your-instance-name"

# LLM API key (depends on provider)
export OPENAI_API_KEY="your-openai-api-key"       # For OpenAI
export ANTHROPIC_API_KEY="your-anthropic-api-key" # For Anthropic
export GOOGLE_API_KEY="your-google-api-key"       # For Google
# No API key needed for Ollama (runs locally)

# For Watsonx
export WATSONX_APIKEY="your-watsonx-api-key"
export WATSONX_PROJECT_ID="your-project-id"
export WATSONX_URL="https://us-south.ml.cloud.ibm.com"  # Optional, this is the default

# For OpenAI-compatible or OpenAI-completions providers
export OPENAI_COMPATIBLE_BASE_URL="https://your-api-endpoint.com/v1"
export OPENAI_COMPATIBLE_API_KEY="your-api-key"  # Optional if not required
```

Or create a `.env` file:

```env
QISKIT_IBM_TOKEN=your-ibm-quantum-token
QISKIT_IBM_RUNTIME_MCP_INSTANCE=your-instance-name  # Recommended for faster startup
OPENAI_API_KEY=your-openai-api-key

# For Watsonx
WATSONX_APIKEY=your-watsonx-api-key
WATSONX_PROJECT_ID=your-project-id

# For OpenAI-compatible APIs
OPENAI_COMPATIBLE_BASE_URL=https://your-api-endpoint.com/v1
OPENAI_COMPATIBLE_API_KEY=your-api-key
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

**With OpenAI-compatible API (e.g., Qiskit Code Assistant, vLLM):**

```bash
# Uses /chat/completions endpoint (supports tool calling)
python langchain_agent.py --provider openai-compatible --model granite-qiskit
```

**With OpenAI-completions API (legacy /completions endpoint):**

```bash
# Note: Limited agent support - no native tool calling
python langchain_agent.py --provider openai-completions --model granite-qiskit
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

- "Set up my IBM Quantum account"
- "What quantum backends are available?"
- "Which backend has the least queue right now?"
- "Tell me about the ibm_brisbane backend"
- "Show me my recent quantum jobs"
- "What are the properties of the least busy backend?"

### Available MCP Tools

The agent has access to these tools provided by the MCP server:

| Tool | Description |
|------|-------------|
| `setup_ibm_quantum_account_tool` | Set up IBM Quantum account with credentials |
| `list_backends_tool` | List all available quantum backends |
| `least_busy_backend_tool` | Find the least busy operational backend |
| `get_backend_properties_tool` | Get detailed properties of a specific backend |
| `list_my_jobs_tool` | List user's recent quantum jobs |
| `get_job_status_tool` | Get status of a specific job |
| `cancel_job_tool` | Cancel a running or queued job |

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
    async with mcp_client.session("qiskit-ibm-runtime") as session:
        agent = await create_quantum_agent_with_session(session, provider="openai")

        # Run queries
        response = await run_agent_query(agent, "List all available quantum backends")
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
- Ensure `qiskit-ibm-runtime-mcp-server` is installed and available in your PATH
- Try running `qiskit-ibm-runtime-mcp-server` directly to verify installation

**"Authentication failed"**
- Verify your `QISKIT_IBM_TOKEN` is correct
- Check that your IBM Quantum account is active

**"Connection timeout"**
- The MCP server may take a few seconds to start
- Check your network connection to IBM Quantum services

**Slow startup (10-30+ seconds)**
- Set `QISKIT_IBM_RUNTIME_MCP_INSTANCE` environment variable to skip instance lookup
- Without an instance specified, the service searches all available instances
- Find your available instances by running:
  ```python
  from qiskit_ibm_runtime import QiskitRuntimeService
  service = QiskitRuntimeService()
  print([i['name'] for i in service.instances()])
  ```
- Or find them at https://quantum.ibm.com/ → Administration → Instances
- With instance set, startup typically takes 2-3 seconds
