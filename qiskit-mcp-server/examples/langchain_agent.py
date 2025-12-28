#!/usr/bin/env python3
"""
LangChain Agent Example for Qiskit MCP Server

This example demonstrates how to create an AI agent using LangChain that
connects to the qiskit-mcp-server via the Model Context Protocol (MCP).

The agent can interact with Qiskit's transpiler to:
- Transpile quantum circuits with configurable optimization levels
- Analyze circuit structure and complexity
- Compare optimization levels to find the best settings

Prerequisites:
    pip install langchain langchain-mcp-adapters python-dotenv
    pip install langchain-openai  # or your preferred LLM provider

Environment variables:
    OPENAI_API_KEY: Your OpenAI API key (or other provider's key)

Usage:
    python langchain_agent.py [--provider PROVIDER] [--model MODEL] [--single]

    Providers: openai, anthropic, google, ollama, watsonx
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from typing import Any

from dotenv import load_dotenv

# LangChain imports
from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools


# Load environment variables from .env file
load_dotenv()


# System prompt for the quantum transpiler agent
SYSTEM_PROMPT = """You are a helpful quantum computing assistant with access to Qiskit's quantum circuit
transpiler through the MCP server.

You can help users optimize their quantum circuits using:
- transpile_circuit_tool: Transpile circuits with configurable optimization levels (0-3)
- analyze_circuit_tool: Analyze circuit structure without transpiling
- compare_optimization_levels_tool: Compare all optimization levels (0-3)

When working with circuits:
1. Accept QASM 3.0 or QASM 2.0 circuit strings from users
2. Use the appropriate tool based on what the user needs
3. Explain the results clearly, including gate counts, depth, and optimizations

Available basis gate presets:
- ibm_eagle: IBM Eagle r3 processors (id, rz, sx, x, ecr, reset)
- ibm_heron: IBM Heron processors (id, rz, sx, x, cz, reset)
- ibm_legacy: Older IBM systems (id, rz, sx, x, cx, reset)

Available topologies:
- linear: Chain connectivity (qubit i <-> i+1)
- ring: Linear with wraparound
- grid: 2D grid connectivity
- heavy_hex: IBM heavy-hex topology
- full: All-to-all connectivity

Optimization levels:
- 0: No optimization, only basis gate decomposition (fastest)
- 1: Light optimization with default layout
- 2: Medium optimization with noise-aware layout (recommended)
- 3: Heavy optimization for best results (can be slow for large circuits)

Always explain the trade-offs between optimization levels and help users choose the best one.
"""


# Sample QASM circuits for demonstration
SAMPLE_BELL_STATE = """OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;
h q[0];
cx q[0], q[1];
"""

SAMPLE_GHZ_STATE = """OPENQASM 3.0;
include "stdgates.inc";
qubit[4] q;
h q[0];
cx q[0], q[1];
cx q[1], q[2];
cx q[2], q[3];
"""

SAMPLE_QFT_CIRCUIT = """OPENQASM 3.0;
include "stdgates.inc";
qubit[3] q;
h q[0];
cp(pi/2) q[1], q[0];
cp(pi/4) q[2], q[0];
h q[1];
cp(pi/2) q[2], q[1];
h q[2];
swap q[0], q[2];
"""


def get_llm(provider: str, model: str | None = None) -> BaseChatModel:
    """Get the appropriate LLM based on the provider.

    Args:
        provider: The LLM provider (openai, anthropic, google, ollama, watsonx)
        model: Optional model name override

    Returns:
        A LangChain chat model instance
    """
    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=model or "gpt-4o", temperature=0)

    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=model or "claude-sonnet-4-20250514", temperature=0)

    elif provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(model=model or "gemini-2.5-pro", temperature=0)

    elif provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(model=model or "llama3.2", temperature=0)

    elif provider == "watsonx":
        from langchain_ibm import ChatWatsonx

        return ChatWatsonx(
            model_id=model or "ibm/granite-3-8b-instruct",
            url=os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com"),
            project_id=os.getenv("WATSONX_PROJECT_ID"),
            params={"temperature": 0, "max_tokens": 4096},
        )

    else:
        raise ValueError(f"Unknown provider: {provider}")


def get_mcp_client() -> MultiServerMCPClient:
    """Create and return an MCP client configured for the Qiskit MCP server."""
    return MultiServerMCPClient(
        {
            "qiskit": {
                "transport": "stdio",
                "command": "qiskit-mcp-server",
                "args": [],
                "env": {},
            }
        }
    )


async def create_transpiler_agent_with_session(
    session: Any, provider: str = "openai", model: str | None = None
) -> Any:
    """Create a LangChain agent using an existing MCP session.

    Args:
        session: An active MCP session
        provider: The LLM provider to use
        model: Optional model name override

    Returns:
        A configured LangChain agent
    """
    # Load tools from the existing session
    tools = await load_mcp_tools(session)
    print(f"Loaded {len(tools)} tools from MCP server:")
    for tool in tools:
        print(f"  - {tool.name}")

    # Get the LLM
    llm = get_llm(provider, model)

    # Create the agent
    agent = create_agent(llm, tools, system_prompt=SYSTEM_PROMPT)

    return agent


async def run_agent_query(
    agent: Any, query: str, history: list[Any] | None = None
) -> tuple[str, list[Any]]:
    """Run a query through the agent with conversation history.

    Args:
        agent: The LangChain agent
        query: The user's question or request
        history: Optional list of previous messages for context

    Returns:
        Tuple of (response_text, updated_history)
    """
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


async def interactive_session(provider: str, model: str | None) -> None:
    """Run an interactive session with the agent using a persistent MCP connection.

    Args:
        provider: The LLM provider to use
        model: Optional model name override
    """
    print("\n" + "=" * 60)
    print("Qiskit Transpiler Agent - Interactive Mode")
    print("=" * 60)
    print(f"\nUsing LLM provider: {provider}")
    if model:
        print(f"Using model: {model}")
    print("\nStarting MCP server and creating agent...")
    print("(This may take a few seconds on first run)\n")

    mcp_client = get_mcp_client()

    # Use persistent session for efficient tool calls
    async with mcp_client.session("qiskit") as session:
        agent = await create_transpiler_agent_with_session(session, provider, model)

        print("\n" + "-" * 60)
        print("Agent ready! You can ask questions about quantum circuit transpilation.")
        print("\nSample circuits are available:")
        print("  - 'bell': 2-qubit Bell state circuit")
        print("  - 'ghz': 4-qubit GHZ state circuit")
        print("  - 'qft': 3-qubit QFT circuit")
        print("\nExample queries:")
        print("  - 'Transpile my bell circuit for IBM Heron'")
        print("  - 'Analyze my ghz circuit'")
        print("  - 'Compare optimization levels for my qft circuit'")
        print("\nType 'quit' to exit, 'clear' to reset conversation history.")
        print("-" * 60 + "\n")

        # Maintain conversation history for context
        history: list[Any] = []

        while True:
            try:
                query = input("You: ").strip()

                if not query:
                    continue

                if query.lower() in ["quit", "exit", "q"]:
                    print("Goodbye!")
                    break

                if query.lower() == "clear":
                    history = []
                    print("Conversation history cleared.\n")
                    continue

                # Replace sample circuit keywords with actual QASM
                if "bell" in query.lower() and "circuit" in query.lower():
                    query = query + f"\n\nHere is the Bell state circuit:\n{SAMPLE_BELL_STATE}"
                elif "ghz" in query.lower() and "circuit" in query.lower():
                    query = query + f"\n\nHere is the GHZ state circuit:\n{SAMPLE_GHZ_STATE}"
                elif "qft" in query.lower() and "circuit" in query.lower():
                    query = query + f"\n\nHere is the QFT circuit:\n{SAMPLE_QFT_CIRCUIT}"

                print("\nAssistant: ", end="", flush=True)
                response, history = await run_agent_query(agent, query, history)
                print(response)
                print()

            except KeyboardInterrupt:
                print("\n\nInterrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\nError: {e}")
                print("Please try again.\n")


async def single_query_mode(provider: str, model: str | None) -> None:
    """Run a single demonstration query.

    Args:
        provider: The LLM provider to use
        model: Optional model name override
    """
    print("\n" + "=" * 60)
    print("Qiskit Transpiler Agent - Single Query Mode")
    print("=" * 60)

    mcp_client = get_mcp_client()

    async with mcp_client.session("qiskit") as session:
        agent = await create_transpiler_agent_with_session(session, provider, model)

        query = f"""Please compare the optimization levels for this QFT circuit and tell me which
level provides the best trade-off between circuit quality and compilation time:

{SAMPLE_QFT_CIRCUIT}"""

        print(f"\nQuery: {query}\n")
        print("-" * 60)
        print("\nAssistant: ", end="", flush=True)
        response, _ = await run_agent_query(agent, query)
        print(response)


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="LangChain Agent for Qiskit MCP Server")
    parser.add_argument(
        "--provider",
        choices=["openai", "anthropic", "google", "ollama", "watsonx"],
        default="openai",
        help="LLM provider to use (default: openai)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name to use (provider-specific)",
    )
    parser.add_argument(
        "--single",
        action="store_true",
        help="Run a single query instead of interactive mode",
    )
    args = parser.parse_args()

    # Check for LLM API keys
    provider_env_vars = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "google": "GOOGLE_API_KEY",
        "watsonx": "WATSONX_APIKEY",
    }

    if args.provider in provider_env_vars:
        env_var = provider_env_vars[args.provider]
        if not os.getenv(env_var):
            print(f"Error: {env_var} not set for provider '{args.provider}'")
            sys.exit(1)

    # Run the appropriate mode
    if args.single:
        asyncio.run(single_query_mode(args.provider, args.model))
    else:
        asyncio.run(interactive_session(args.provider, args.model))


if __name__ == "__main__":
    main()
