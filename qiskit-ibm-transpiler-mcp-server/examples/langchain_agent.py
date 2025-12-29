#!/usr/bin/env python3
"""
LangChain Agent Example for Qiskit IBM Transpiler MCP Server

This example demonstrates how to create an AI agent using LangChain that
connects to the qiskit-ibm-transpiler-mcp-server via the Model Context Protocol (MCP).

The agent can interact with IBM Quantum's AI-powered transpiler to:
- Perform AI routing on quantum circuits
- Synthesize Clifford circuits
- Synthesize Linear Function circuits
- Synthesize Permutation circuits
- Synthesize Pauli Network circuits

Prerequisites:
    pip install langchain langchain-mcp-adapters python-dotenv
    pip install langchain-openai  # or your preferred LLM provider

Environment variables:
    QISKIT_IBM_TOKEN: Your IBM Quantum API token
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
SYSTEM_PROMPT = """You are a helpful quantum computing assistant with access to IBM Quantum's AI-powered
transpiler through the MCP server.

You can help users optimize their quantum circuits using:
- AI Routing (ai_routing tool): Insert SWAP operations to make circuits compatible with backend coupling maps
- AI Clifford Synthesis (ai_clifford_synthesis tool): Optimize Clifford circuits (H, S, CX gates)
- AI Linear Function Synthesis (ai_linear_function_synthesis tool): Optimize Linear Function circuits (CX, SWAP gates)
- AI Permutation Synthesis (ai_permutation_synthesis tool): Optimize Permutation circuits (SWAP gates)
- AI Pauli Network Synthesis (ai_pauli_network_synthesis tool): Optimize Pauli Network circuits (H, S, SX, CX, RX, RY, RZ gates)

When optimizing circuits:
1. First use ai_routing to route the circuit for the target backend
2. Then apply the appropriate synthesis pass based on the circuit type
3. Report the optimization improvements (depth reduction, gate count reduction)

The tools accept QASM 3.0 strings as input and return optimized circuits in QPY format.
Always explain the optimization results and improvements achieved.

Available IBM Quantum backends include: ibm_brisbane, ibm_kyiv, ibm_sherbrooke, ibm_fez, etc.
"""


# Sample QASM circuits for demonstration
SAMPLE_CLIFFORD_CIRCUIT = """OPENQASM 3.0;
include "stdgates.inc";
qubit[3] q;
h q[0];
cx q[0], q[1];
s q[1];
cx q[1], q[2];
h q[2];
"""

SAMPLE_LINEAR_FUNCTION_CIRCUIT = """OPENQASM 3.0;
include "stdgates.inc";
qubit[3] q;
cx q[0], q[1];
cx q[1], q[2];
swap q[0], q[2];
cx q[2], q[1];
"""

SAMPLE_BELL_STATE = """OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;
h q[0];
cx q[0], q[1];
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
    """Create and return an MCP client configured for the Qiskit IBM Transpiler server."""
    return MultiServerMCPClient(
        {
            "qiskit-ibm-transpiler": {
                "transport": "stdio",
                "command": "qiskit-ibm-transpiler-mcp-server",
                "args": [],
                "env": {
                    "QISKIT_IBM_TOKEN": os.getenv("QISKIT_IBM_TOKEN", ""),
                },
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
    print("Qiskit IBM Transpiler Agent - Interactive Mode")
    print("=" * 60)
    print(f"\nUsing LLM provider: {provider}")
    if model:
        print(f"Using model: {model}")
    print("\nStarting MCP server and creating agent...")
    print("(This may take a few seconds on first run)\n")

    mcp_client = get_mcp_client()

    # Use persistent session for efficient tool calls
    async with mcp_client.session("qiskit-ibm-transpiler") as session:
        agent = await create_transpiler_agent_with_session(session, provider, model)

        print("\n" + "-" * 60)
        print("Agent ready! You can ask questions about quantum circuit optimization.")
        print("\nSample circuits are available:")
        print("  - 'clifford': 3-qubit Clifford circuit (H, CX, S gates)")
        print("  - 'linear': 3-qubit Linear Function circuit (CX, SWAP gates)")
        print("  - 'bell': 2-qubit Bell state circuit")
        print("\nExample queries:")
        print("  - 'Route my bell state circuit for ibm_brisbane'")
        print("  - 'Optimize my clifford circuit using AI synthesis'")
        print("  - 'What backends are available for optimization?'")
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
                if "clifford" in query.lower() and "circuit" in query.lower():
                    query = query + f"\n\nHere is the Clifford circuit:\n{SAMPLE_CLIFFORD_CIRCUIT}"
                elif "linear" in query.lower() and "circuit" in query.lower():
                    query = (
                        query
                        + f"\n\nHere is the Linear Function circuit:\n{SAMPLE_LINEAR_FUNCTION_CIRCUIT}"
                    )
                elif "bell" in query.lower() and "circuit" in query.lower():
                    query = query + f"\n\nHere is the Bell state circuit:\n{SAMPLE_BELL_STATE}"

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
    print("Qiskit IBM Transpiler Agent - Single Query Mode")
    print("=" * 60)

    mcp_client = get_mcp_client()

    async with mcp_client.session("qiskit-ibm-transpiler") as session:
        agent = await create_transpiler_agent_with_session(session, provider, model)

        query = f"""Please route this Bell state circuit for the ibm_brisbane backend and tell me about
the optimization results:

{SAMPLE_BELL_STATE}"""

        print(f"\nQuery: {query}\n")
        print("-" * 60)
        print("\nAssistant: ", end="", flush=True)
        response, _ = await run_agent_query(agent, query)
        print(response)


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="LangChain Agent for Qiskit IBM Transpiler MCP Server"
    )
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

    # Verify required environment variables
    if not os.getenv("QISKIT_IBM_TOKEN"):
        print("Warning: QISKIT_IBM_TOKEN not set. Some features may not work.")
        print("Get your token from https://quantum.ibm.com/")

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
