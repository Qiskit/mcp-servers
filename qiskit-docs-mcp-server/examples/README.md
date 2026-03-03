# Qiskit Docs MCP Server Examples

This directory contains examples demonstrating how to build AI agents that query Qiskit documentation through the **qiskit-docs-mcp-server**.

## Available Examples

| File | Description |
|------|-------------|
| [`langchain_agent.ipynb`](langchain_agent.ipynb) | **Jupyter Notebook** - Interactive tutorial with step-by-step examples |
| [`langchain_agent.py`](langchain_agent.py) | **Python Script** - Command-line agent with multiple LLM provider support |

## LangChain Agent Example

The examples show how to create an AI agent using LangChain that connects to the qiskit-docs-mcp-server via the Model Context Protocol (MCP).

### Quick Start with Jupyter Notebook

For an interactive experience, open the notebook:

```bash
jupyter notebook langchain_agent.ipynb
```

The notebook includes:
- Step-by-step setup instructions
- Multiple LLM provider options (just run the cell for your provider)
- Interactive examples for searching docs, getting module documentation, and more
- A custom query cell for your own questions

### Features

The agent can:

- Search Qiskit documentation
- Get SDK module documentation (circuit, primitives, transpiler, etc.)
- Get guide documentation (optimization, error-mitigation, etc.)
- List available modules, addons, and guides

### Supported LLM Providers

| Provider | Package | Default Model | API Key Required |
|----------|---------|---------------|------------------|
| OpenAI | `langchain-openai` | gpt-4o | Yes (`OPENAI_API_KEY`) |
| Anthropic | `langchain-anthropic` | claude-sonnet-4-5-20250929 | Yes (`ANTHROPIC_API_KEY`) |
| Google | `langchain-google-genai` | gemini-3-pro-preview | Yes (`GOOGLE_API_KEY`) |
| Ollama | `langchain-ollama` | llama3.3 | No (runs locally) |
| Watsonx | `langchain-ibm` | ibm/granite-4-h-small | Yes (`WATSONX_APIKEY`, `WATSONX_PROJECT_ID`) |

### Architecture

```
┌─────────────┐     MCP Protocol     ┌──────────────────────────────────┐
│  LangChain  │ ◄──────────────────► │ qiskit-docs-mcp-server           │
│    Agent    │                      │                                  │
└─────────────┘                      │  ┌────────────────────────────┐  │
                                     │  │   Documentation Fetcher    │  │
                                     │  └────────────────────────────┘  │
                                     │               │                  │
                                     └───────────────│──────────────────┘
                                                     ▼
                                            ┌─────────────────┐
                                            │  Qiskit Docs    │
                                            │  (docs.quantum  │
                                            │   .ibm.com)     │
                                            └─────────────────┘
```

### Prerequisites

1. **Python 3.10+**

2. **Install the MCP server:**

```bash
pip install qiskit-docs-mcp-server
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
# ollama pull llama3.3
python langchain_agent.py --provider ollama --model llama3.3
```

**With IBM Watsonx:**

```bash
python langchain_agent.py --provider watsonx
# Or with a specific model
python langchain_agent.py --provider watsonx --model ibm/granite-4-h-small
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

- "Search for information about quantum circuits"
- "Get documentation for the circuit module"
- "Show me the optimization guide"
- "What modules are available in Qiskit?"
- "Tell me about error mitigation techniques"
- "Search for transpiler documentation"

### Available MCP Tools

The agent has access to these tools provided by the MCP server:

| Tool | Description |
|------|-------------|
| `get_sdk_module_docs` | Get documentation for a Qiskit SDK module (circuit, primitives, etc.) |
| `get_guide` | Get a Qiskit guide or best practice documentation |
| `search_docs` | Search Qiskit documentation for relevant content |

### Available MCP Resources

The agent can also access these resources:

| Resource URI | Description |
|--------------|-------------|
| `qiskit-docs://modules` | List of all Qiskit SDK modules |
| `qiskit-docs://addons` | List of all Qiskit addon modules and tutorials |
| `qiskit-docs://guides` | List of Qiskit guides and best practices |

### Using as a Library

You can import and use the agent in your own async code:

```python
import asyncio
from langchain_agent import (
    get_mcp_client,
    create_docs_agent_with_session,
    run_agent_query,
)

async def main():
    # Use persistent session for efficient tool calls
    mcp_client = get_mcp_client()
    async with mcp_client.session("qiskit-docs") as session:
        agent = await create_docs_agent_with_session(session, provider="openai")

        # Run queries
        response = await run_agent_query(agent, "Search for quantum circuit documentation")
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
        "qiskit-docs": {
            "transport": "stdio",
            "command": "qiskit-docs-mcp-server",
            "args": [],
        }
    })

    # Use persistent session for efficient tool calls
    async with mcp_client.session("qiskit-docs") as session:
        tools = await load_mcp_tools(session)

        # Custom system prompt
        system_prompt = "You are a quantum computing documentation expert..."

        llm = ChatOpenAI(model="gpt-4o", temperature=0)
        agent = create_agent(llm, tools, system_prompt=system_prompt)

        # Use the agent within the session context
        # ... your agent logic here ...

asyncio.run(create_custom_agent())
```

### Troubleshooting

**"MCP server not found"**
- Ensure `qiskit-docs-mcp-server` is installed and available in your PATH
- Try running `qiskit-docs-mcp-server` directly to verify installation

**"Connection timeout"**
- The MCP server may take a few seconds to start
- Check your network connection to docs.quantum.ibm.com

**"No results found"**
- Try different search terms or module names
- Check the available modules with the `qiskit-docs://modules` resource
