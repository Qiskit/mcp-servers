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
LangChain Agent Example with Qiskit IBM Runtime MCP Server

This example demonstrates how to create an AI agent using LangChain that
connects to the qiskit-ibm-runtime-mcp-server via the Model Context Protocol (MCP).

The agent can interact with IBM Quantum services through the MCP server, which
provides tools for listing backends, finding the least busy backend, managing
jobs, and more.

Supported LLM Providers:
    - OpenAI (default): pip install langchain-openai
    - Anthropic: pip install langchain-anthropic
    - Ollama (local): pip install langchain-ollama
    - Google: pip install langchain-google-genai
    - Watsonx: pip install langchain-ibm
    - OpenAI-compatible (/chat/completions): pip install langchain-openai
    - OpenAI-completions (/completions, limited agent support): pip install langchain-openai

Requirements:
    pip install langchain langchain-mcp-adapters python-dotenv
    pip install <provider-package>  # See above for your chosen provider

Usage:
    # With OpenAI (default)
    export OPENAI_API_KEY="your-api-key"
    python langchain_agent.py

    # With Anthropic
    export ANTHROPIC_API_KEY="your-api-key"
    python langchain_agent.py --provider anthropic

    # With Ollama (local, no API key needed)
    python langchain_agent.py --provider ollama --model llama3.2

    # With Google
    export GOOGLE_API_KEY="your-api-key"
    python langchain_agent.py --provider google

    # With Watsonx
    export WATSONX_APIKEY="your-watsonx-api-key"
    export WATSONX_PROJECT_ID="your-project-id"
    python langchain_agent.py --provider watsonx

    # With OpenAI-compatible chat API (e.g., Qiskit Code Assistant /chat/completions)
    export OPENAI_COMPATIBLE_API_KEY="your-api-key"
    export OPENAI_COMPATIBLE_BASE_URL="https://your-api-endpoint.com/v1"
    python langchain_agent.py --provider openai-compatible --model your-model-name

    # With OpenAI-compatible completions API (/completions endpoint)
    # Note: Limited agent support - no native tool calling
    export OPENAI_COMPATIBLE_BASE_URL="https://your-api-endpoint.com/v1"
    python langchain_agent.py --provider openai-completions --model your-model-name
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

SYSTEM_PROMPT = """You are a helpful quantum computing assistant with access to IBM Quantum services
through the Qiskit IBM Runtime MCP server.

You can help users:
- Set up their IBM Quantum account (setup_ibm_quantum_account_tool)
- List available quantum backends (list_backends_tool)
- Find the least busy backend for running jobs (least_busy_backend_tool)
- Get detailed backend properties (get_backend_properties_tool)
- Get backend calibration data including T1, T2, error rates, and faulty qubits (get_backend_calibration_tool)
- List recent jobs (list_my_jobs_tool)
- Check job status (get_job_status_tool)
- Cancel jobs (cancel_job_tool)

Always provide clear explanations about quantum computing concepts when relevant.
When listing backends, highlight key properties like qubit count and operational status.
When showing calibration data, highlight faulty qubits/gates that users should avoid.
If an operation fails, explain the error and suggest possible solutions."""


def get_llm(provider: str, model: str | None = None) -> BaseChatModel:
    """
    Get an LLM instance for the specified provider.

    Args:
        provider: The LLM provider ('openai', 'anthropic', 'ollama', 'google').
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

    elif provider == "openai-compatible":
        # Supports OpenAI-compatible APIs like Qiskit Code Assistant, vLLM, etc.
        # Uses the /chat/completions endpoint which supports tool calling for agents.
        # For the legacy /completions endpoint, use "openai-completions" provider instead.
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            raise ValueError("Install langchain-openai: pip install langchain-openai")
        base_url = os.getenv("OPENAI_COMPATIBLE_BASE_URL")
        if not base_url:
            raise ValueError(
                "OPENAI_COMPATIBLE_BASE_URL environment variable is required "
                "for openai-compatible provider"
            )
        return ChatOpenAI(
            model=model or "granite-qiskit",
            base_url=base_url,
            api_key=os.getenv("OPENAI_COMPATIBLE_API_KEY", "not-needed"),
            temperature=0,
        )

    elif provider == "openai-completions":
        # Uses the legacy /completions endpoint (text completions, not chat).
        # Note: This provider has LIMITED agent capabilities since it doesn't
        # support native tool calling. It relies on ReAct-style prompting.
        # Best suited for simple queries or models that only support /completions.
        try:
            from langchain_openai import OpenAI
        except ImportError:
            raise ValueError("Install langchain-openai: pip install langchain-openai")
        base_url = os.getenv("OPENAI_COMPATIBLE_BASE_URL")
        if not base_url:
            raise ValueError(
                "OPENAI_COMPATIBLE_BASE_URL environment variable is required "
                "for openai-completions provider"
            )
        return OpenAI(
            model=model or "granite-qiskit",
            base_url=base_url,
            api_key=os.getenv("OPENAI_COMPATIBLE_API_KEY", "not-needed"),
            temperature=0,
        )

    else:
        raise ValueError(
            f"Unknown provider: {provider}. "
            f"Supported: openai, anthropic, ollama, google, watsonx, "
            f"openai-compatible, openai-completions"
        )


def check_api_key(provider: str) -> bool:
    """Check if required API key/config is set for the provider."""
    key_map = {
        "openai": ["OPENAI_API_KEY"],
        "anthropic": ["ANTHROPIC_API_KEY"],
        "google": ["GOOGLE_API_KEY"],
        "ollama": [],  # No API key needed for local Ollama
        "watsonx": ["WATSONX_APIKEY", "WATSONX_PROJECT_ID"],
        "openai-compatible": ["OPENAI_COMPATIBLE_BASE_URL"],
        "openai-completions": ["OPENAI_COMPATIBLE_BASE_URL"],
    }

    required_keys = key_map.get(provider, [])
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
    Create and return an MCP client configured for the Qiskit IBM Runtime server.

    Returns:
        Configured MultiServerMCPClient instance.
    """
    return MultiServerMCPClient(
        {
            "qiskit-ibm-runtime": {
                "transport": "stdio",
                "command": "qiskit-ibm-runtime-mcp-server",
                "args": [],
                "env": {
                    "QISKIT_IBM_TOKEN": os.getenv("QISKIT_IBM_TOKEN", ""),
                    "QISKIT_IBM_RUNTIME_MCP_INSTANCE": os.getenv(
                        "QISKIT_IBM_RUNTIME_MCP_INSTANCE", ""
                    ),
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
    if provider == "openai-completions":
        print(
            "WARNING: The 'openai-completions' provider uses the legacy /completions "
            "endpoint which does not support native tool calling. "
            "Agent functionality may be limited. Consider using 'openai-compatible' "
            "with a /chat/completions endpoint if available."
        )

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
    """Run an interactive session with the quantum agent."""
    if not check_api_key(provider):
        return

    print("Quantum Computing Agent with LangChain + MCP")
    print("=" * 45)
    print(f"Provider: {provider}" + (f" (model: {model})" if model else ""))
    print("This agent connects to the qiskit-ibm-runtime-mcp-server")
    print("to interact with IBM Quantum services.")
    print("Type 'quit' to exit, 'clear' to reset conversation history.\n")

    # Example queries to demonstrate capabilities
    example_queries = [
        "Set up my IBM Quantum account",
        "What quantum backends are available?",
        "Which backend has the least queue right now?",
        "Tell me about the ibm_brisbane backend",
        "Show me my recent quantum jobs",
    ]

    print("Example queries you can try:")
    for i, query in enumerate(example_queries, 1):
        print(f"  {i}. {query}")
    print()

    print("Connecting to MCP server...")

    # Use persistent session to avoid spawning new server for each tool call
    mcp_client = get_mcp_client()
    async with mcp_client.session("qiskit-ibm-runtime") as session:
        agent = await create_quantum_agent_with_session(session, provider, model)
        print("Connected! Ready to answer your questions.\n")

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
    async with mcp_client.session("qiskit-ibm-runtime") as session:
        agent = await create_quantum_agent_with_session(session, provider, model)

        # Run a sample query
        response, _ = await run_agent_query(
            agent, "List all available quantum backends and tell me which one is least busy"
        )
        print(f"\nResponse:\n{response}")


def main():
    """Entry point for the example."""
    parser = argparse.ArgumentParser(
        description="LangChain Agent for IBM Quantum via MCP",
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

  # OpenAI-compatible chat API (e.g., Qiskit Code Assistant /chat/completions)
  python langchain_agent.py --provider openai-compatible --model granite-qiskit

  # OpenAI-compatible completions API (/completions endpoint - limited agent support)
  python langchain_agent.py --provider openai-completions --model granite-qiskit

  # Single query mode
  python langchain_agent.py --single
        """,
    )
    parser.add_argument(
        "--provider",
        choices=[
            "openai",
            "anthropic",
            "ollama",
            "google",
            "watsonx",
            "openai-compatible",
            "openai-completions",
        ],
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
