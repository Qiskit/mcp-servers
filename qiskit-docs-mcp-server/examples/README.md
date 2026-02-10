# Qiskit Docs MCP Server Examples

This directory contains examples of how to use the Qiskit Docs MCP server with different AI agent frameworks.

## LangChain Agent with Multiple Providers

The `langchain_agent.py` script demonstrates how to create an AI agent that can query Qiskit documentation using LangChain and various LLM providers.

### Prerequisites

Install the package with example dependencies:

```bash
uv pip install -e ".[examples]"
```

### Usage

Run the agent with your preferred provider:

```bash
# OpenAI (default)
export OPENAI_API_KEY="your-key"
python langchain_agent.py

# Anthropic
export ANTHROPIC_API_KEY="your-key"
python langchain_agent.py --provider anthropic

# Local Ollama (no key needed)
python langchain_agent.py --provider ollama --model llama3
```

See `python langchain_agent.py --help` for all options.
