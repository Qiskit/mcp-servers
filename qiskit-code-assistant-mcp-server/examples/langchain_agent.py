# This code is part of Qiskit.
#
# (C) Copyright IBM 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
LangChain Agent Example with Qiskit Code Assistant MCP Server

This example demonstrates how to create an AI agent using LangChain that
connects to the qiskit-code-assistant-mcp-server via the Model Context Protocol (MCP).

The agent can interact with Qiskit Code Assistant to:
- Generate quantum code completions
- Answer questions about Qiskit and quantum computing (RAG mode)
- Accept model disclaimers
- Track completion acceptance

Supported LLM Providers:
    - OpenAI (default): pip install langchain-openai
    - Anthropic: pip install langchain-anthropic
    - Ollama (local): pip install langchain-ollama
    - Google: pip install langchain-google-genai
    - Watsonx: pip install langchain-ibm

Requirements:
    pip install langchain langchain-mcp-adapters python-dotenv
    pip install <provider-package>  # See above for your chosen provider

Usage:
    # With OpenAI (default)
    export OPENAI_API_KEY="your-api-key"
    export QISKIT_IBM_TOKEN="your-ibm-quantum-token"
    python langchain_agent.py

    # With Anthropic
    export ANTHROPIC_API_KEY="your-api-key"
    export QISKIT_IBM_TOKEN="your-ibm-quantum-token"
    python langchain_agent.py --provider anthropic

    # With Ollama (local, no API key needed)
    export QISKIT_IBM_TOKEN="your-ibm-quantum-token"
    python langchain_agent.py --provider ollama --model llama3.2

    # Single query mode
    python langchain_agent.py --single
"""

import argparse
import asyncio
import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools


# Load environment variables
load_dotenv()

SYSTEM_PROMPT = """You are a helpful quantum computing coding assistant with access to Qiskit Code Assistant
through the MCP server.

You can help users:
- Generate quantum code using Qiskit (get_completion_tool)
- Answer questions about Qiskit and quantum computing concepts (get_rag_completion_tool)
- Accept model disclaimers when required (accept_model_disclaimer_tool)
- Track completion acceptance for feedback (accept_completion_tool)

When generating code:
- Use the get_completion_tool to generate Qiskit code based on user prompts
- Provide clear explanations of the generated code
- Suggest improvements or alternatives when appropriate

When answering questions:
- Use the get_rag_completion_tool for conceptual questions about quantum computing
- Provide accurate and educational explanations
- Reference Qiskit documentation when relevant

Always provide clear explanations about quantum computing concepts when relevant.
If an operation fails, explain the error and suggest possible solutions."""


def get_llm(provider: str, model: str | None = None) -> BaseChatModel:
    """
    Get an LLM instance for the specified provider.

    Args:
        provider: The LLM provider ('openai', 'anthropic', 'ollama', 'google', 'watsonx').
        model: Optional model name override.

    Returns:
        Configured LLM instance.

    Raises:
        ValueError: If provider is not supported or required package is missing.
    """
    if provider == "openai":
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            raise ValueError("Install langchain-openai: pip install langchain-openai")
        return ChatOpenAI(model=model or "gpt-4o", temperature=0)

    elif provider == "anthropic":
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError:
            raise ValueError("Install langchain-anthropic: pip install langchain-anthropic")
        return ChatAnthropic(model=model or "claude-sonnet-4-20250514", temperature=0)

    elif provider == "ollama":
        try:
            from langchain_ollama import ChatOllama
        except ImportError:
            raise ValueError("Install langchain-ollama: pip install langchain-ollama")
        return ChatOllama(model=model or "llama3.2", temperature=0)

    elif provider == "google":
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError:
            raise ValueError("Install langchain-google-genai: pip install langchain-google-genai")
        return ChatGoogleGenerativeAI(model=model or "gemini-2.5-pro", temperature=0)

    elif provider == "watsonx":
        try:
            from langchain_ibm import ChatWatsonx
        except ImportError:
            raise ValueError("Install langchain-ibm: pip install langchain-ibm")
        return ChatWatsonx(
            model_id=model or "ibm/granite-3-8b-instruct",
            url=os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com"),
            project_id=os.getenv("WATSONX_PROJECT_ID"),
            params={
                "temperature": 0,
                "max_tokens": 4096,
            },
        )

    else:
        raise ValueError(
            f"Unknown provider: {provider}. Supported: openai, anthropic, ollama, google, watsonx"
        )


def check_api_key(provider: str) -> bool:
    """Check if required API key/config is set for the provider."""
    key_map = {
        "openai": ["OPENAI_API_KEY"],
        "anthropic": ["ANTHROPIC_API_KEY"],
        "google": ["GOOGLE_API_KEY"],
        "ollama": [],  # No API key needed for local Ollama
        "watsonx": ["WATSONX_APIKEY", "WATSONX_PROJECT_ID"],
    }

    # Always need QISKIT_IBM_TOKEN for the MCP server
    required_keys = [*key_map.get(provider, []), "QISKIT_IBM_TOKEN"]
    missing_keys = [key for key in required_keys if not os.getenv(key)]

    if not missing_keys:
        return True

    print(f"Error: Missing required environment variables for {provider}:")
    for key in missing_keys:
        print(f"  - {key}")
    print("\nSet them with:")
    for key in missing_keys:
        print(f"  export {key}='your-value'")
    return False


def get_mcp_client() -> MultiServerMCPClient:
    """
    Create and return an MCP client configured for the Qiskit Code Assistant server.

    Returns:
        Configured MultiServerMCPClient instance.
    """
    return MultiServerMCPClient(
        {
            "qiskit-code-assistant": {
                "transport": "stdio",
                "command": "qiskit-code-assistant-mcp-server",
                "args": [],
                "env": {
                    "QISKIT_IBM_TOKEN": os.getenv("QISKIT_IBM_TOKEN", ""),
                },
            }
        }
    )


async def create_quantum_agent_with_session(
    session, provider: str = "openai", model: str | None = None
):
    """
    Create a LangChain agent using an existing MCP session.

    This uses a persistent session to avoid spawning a new server process
    for each tool call, significantly improving performance.

    Args:
        session: An active MCP ClientSession from MultiServerMCPClient.session()
        provider: The LLM provider.
        model: Optional model name override.

    Returns:
        Configured LangChain agent.
    """
    # Load tools from the existing session (reuses the same server process)
    tools = await load_mcp_tools(session)

    # Get the LLM for the specified provider
    llm = get_llm(provider, model)

    # Create an agent using LangChain's create_agent
    agent = create_agent(llm, tools, system_prompt=SYSTEM_PROMPT)

    return agent


async def run_agent_query(agent, query: str, history: list | None = None) -> tuple[str, list]:
    """
    Run a query through the agent with conversation history.

    Args:
        agent: The configured LangChain agent.
        query: The user's query.
        history: Optional list of previous messages for context.

    Returns:
        Tuple of (response_text, updated_history).
    """
    from langchain_core.messages import HumanMessage

    # Build messages with history
    messages = list(history) if history else []
    messages.append(HumanMessage(content=query))

    result = await agent.ainvoke({"messages": messages})
    result_messages = result.get("messages", [])

    if result_messages:
        response = result_messages[-1].content
        # Return the full conversation history from the agent
        return response, result_messages

    return "No response generated.", messages


async def interactive_session(provider: str, model: str | None):
    """Run an interactive session with the quantum coding agent."""
    if not check_api_key(provider):
        return

    print("Qiskit Code Assistant Agent with LangChain + MCP")
    print("=" * 50)
    print(f"Provider: {provider}" + (f" (model: {model})" if model else ""))
    print("This agent connects to the qiskit-code-assistant-mcp-server")
    print("to help you write and understand quantum code.")
    print("Type 'quit' to exit, 'clear' to reset conversation history.\n")

    # Example queries to demonstrate capabilities
    example_queries = [
        "Write a quantum circuit that creates a Bell state",
        "What is quantum entanglement?",
        "Generate code to run a VQE algorithm",
        "Explain how Qiskit's transpiler works",
        "Write a circuit with 3 qubits and apply a Toffoli gate",
    ]

    print("Example queries you can try:")
    for i, query in enumerate(example_queries, 1):
        print(f"  {i}. {query}")
    print()

    print("Connecting to MCP server...")

    # Use persistent session to avoid spawning new server for each tool call
    mcp_client = get_mcp_client()
    async with mcp_client.session("qiskit-code-assistant") as session:
        agent = await create_quantum_agent_with_session(session, provider, model)
        print("Connected! Ready to help with quantum code.\n")

        # Maintain conversation history for context
        history: list = []

        while True:
            try:
                user_input = input("You: ").strip()
                if not user_input:
                    continue
                if user_input.lower() in ["quit", "exit", "q"]:
                    print("Goodbye!")
                    break
                if user_input.lower() == "clear":
                    history = []
                    print("Conversation history cleared.\n")
                    continue

                response, history = await run_agent_query(agent, user_input, history)
                print(f"\nAssistant: {response}\n")

            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}\n")


async def single_query_example(provider: str, model: str | None):
    """Example of running a single query programmatically."""
    if not check_api_key(provider):
        return

    print("Running single query example...")
    print(f"Provider: {provider}" + (f" (model: {model})" if model else ""))
    print("-" * 40)

    # Use persistent session for the query
    mcp_client = get_mcp_client()
    async with mcp_client.session("qiskit-code-assistant") as session:
        agent = await create_quantum_agent_with_session(session, provider, model)

        # Run a sample query
        response, _ = await run_agent_query(
            agent, "Write a Qiskit circuit that creates a GHZ state with 3 qubits"
        )
        print(f"\nResponse:\n{response}")


def main():
    """Entry point for the example."""
    parser = argparse.ArgumentParser(
        description="LangChain Agent for Qiskit Code Assistant via MCP",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # OpenAI (default)
  python langchain_agent.py

  # Anthropic Claude
  python langchain_agent.py --provider anthropic

  # Local Ollama
  python langchain_agent.py --provider ollama --model llama3.2

  # Google Gemini
  python langchain_agent.py --provider google

  # IBM Watsonx
  python langchain_agent.py --provider watsonx

  # Single query mode
  python langchain_agent.py --single
        """,
    )
    parser.add_argument(
        "--provider",
        choices=["openai", "anthropic", "ollama", "google", "watsonx"],
        default="openai",
        help="LLM provider to use (default: openai)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name override (uses provider default if not specified)",
    )
    parser.add_argument(
        "--single",
        action="store_true",
        help="Run a single example query instead of interactive mode",
    )

    args = parser.parse_args()

    if args.single:
        asyncio.run(single_query_example(args.provider, args.model))
    else:
        asyncio.run(interactive_session(args.provider, args.model))


if __name__ == "__main__":
    main()
