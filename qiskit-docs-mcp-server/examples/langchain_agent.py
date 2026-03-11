# This code is part of Qiskit.
#
# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
LangChain Agent Example with Qiskit Docs MCP Server

This example demonstrates how to create an AI agent using LangChain that
connects to the qiskit-docs-mcp-server via the Model Context Protocol (MCP).

The agent can query Qiskit documentation through the MCP server, which
provides tools for searching docs, getting module documentation, and accessing guides.

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
    python langchain_agent.py

    # With Anthropic
    export ANTHROPIC_API_KEY="your-api-key"
    python langchain_agent.py --provider anthropic

    # With Ollama (local, no API key needed)
    python langchain_agent.py --provider ollama --model llama3.3

    # With Google
    export GOOGLE_API_KEY="your-api-key"
    python langchain_agent.py --provider google

    # With Watsonx
    export WATSONX_APIKEY="your-watsonx-api-key"
    export WATSONX_PROJECT_ID="your-project-id"
    python langchain_agent.py --provider watsonx
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

SYSTEM_PROMPT = """You are a helpful quantum computing documentation assistant with access to Qiskit documentation
through the Qiskit Docs MCP server.

You can help users:
- Search Qiskit documentation (search_docs)
- Get SDK module documentation for circuit, primitives, transpiler, quantum_info, result, visualization (get_sdk_module_docs)
- Get guide documentation for optimization, quantum-circuits, error-mitigation, dynamic-circuits, parametric-compilation, performance-tuning (get_guide)
- List available modules, addons, and guides (via resources)

Always provide clear explanations about quantum computing concepts when relevant.
When showing documentation, highlight key points and provide context.
If a search returns no results, suggest alternative search terms or related modules."""


def get_llm(provider: str = "openai", model: str | None = None) -> BaseChatModel:
    """
    Get the appropriate LLM based on the provider.

    Args:
        provider: LLM provider name
        model: Optional model name override

    Returns:
        Configured LLM instance
    """
    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=model or "gpt-4o", temperature=0)

    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=model or "claude-sonnet-4-5-20250929", temperature=0)

    elif provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(model=model or "llama3.3", temperature=0)

    elif provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(model=model or "gemini-3-pro-preview", temperature=0)

    elif provider == "watsonx":
        from langchain_ibm import ChatWatsonx

        return ChatWatsonx(
            model_id=model or "ibm/granite-4-h-small",
            url=os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com"),
            project_id=os.getenv("WATSONX_PROJECT_ID"),
            temperature=0,
        )

    else:
        raise ValueError(f"Unknown provider: {provider}")


def get_mcp_client() -> MultiServerMCPClient:
    """
    Create and configure the MCP client for qiskit-docs-mcp-server.

    Returns:
        Configured MultiServerMCPClient instance
    """
    return MultiServerMCPClient(
        {
            "qiskit-docs": {
                "transport": "stdio",
                "command": "qiskit-docs-mcp-server",
                "args": [],
            }
        }
    )


async def create_docs_agent_with_session(
    session, provider: str = "openai", model: str | None = None
):
    """
    Create a LangChain agent with MCP tools using an existing session.

    Args:
        session: Active MCP session
        provider: LLM provider name
        model: Optional model name override

    Returns:
        Configured agent
    """
    # Load MCP tools from the session
    tools = await load_mcp_tools(session)

    # Get LLM
    llm = get_llm(provider, model)

    # Create agent
    agent = create_agent(llm, tools, system_prompt=SYSTEM_PROMPT)

    return agent


async def run_agent_query(agent, query: str) -> str:
    """
    Run a query through the agent.

    Args:
        agent: Configured agent
        query: User query

    Returns:
        Agent response
    """
    response = await agent.ainvoke({"messages": [("user", query)]})
    return response["messages"][-1].content


async def interactive_mode(provider: str = "openai", model: str | None = None):
    """
    Run the agent in interactive mode.

    Args:
        provider: LLM provider name
        model: Optional model name override
    """
    print(f"\n🤖 Qiskit Docs Agent (Provider: {provider}, Model: {model or 'default'})")
    print("=" * 60)
    print("Ask questions about Qiskit documentation!")
    print("Type 'quit' or 'exit' to end the session.")

    # Example queries to demonstrate capabilities
    example_queries = [
        "What are the available SDK modules in Qiskit?",
        "Show me documentation about quantum circuits",
        "How do I create and run a quantum circuit?",
        "What is VQE and how do I implement it?",
        "Explain the Qiskit transpiler",
    ]

    print("\nExample queries you can try:")
    for i, query in enumerate(example_queries, 1):
        print(f"  {i}. {query}")

    print("=" * 60)

    mcp_client = get_mcp_client()

    # Use persistent session for efficient tool calls
    async with mcp_client.session("qiskit-docs") as session:
        agent = await create_docs_agent_with_session(session, provider, model)

        while True:
            try:
                query = input("\n💬 You: ").strip()

                if query.lower() in ["quit", "exit", "q"]:
                    print("\n👋 Goodbye!")
                    break

                if not query:
                    continue

                print("\n🤔 Agent: ", end="", flush=True)
                response = await run_agent_query(agent, query)
                print(response)

            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")


async def single_query_mode(query: str, provider: str = "openai", model: str | None = None):
    """
    Run a single query and exit.

    Args:
        query: User query
        provider: LLM provider name
        model: Optional model name override
    """
    print(f"\n🤖 Qiskit Docs Agent (Provider: {provider}, Model: {model or 'default'})")
    print("=" * 60)

    mcp_client = get_mcp_client()

    # Use persistent session for efficient tool calls
    async with mcp_client.session("qiskit-docs") as session:
        agent = await create_docs_agent_with_session(session, provider, model)

        print(f"\n💬 Query: {query}")
        print("\n🤔 Agent: ", end="", flush=True)
        response = await run_agent_query(agent, query)
        print(response)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="LangChain Agent with Qiskit Docs MCP Server")
    parser.add_argument(
        "--provider",
        choices=["openai", "anthropic", "ollama", "google", "watsonx"],
        default="openai",
        help="LLM provider to use (default: openai)",
    )
    parser.add_argument("--model", type=str, help="Model name (overrides provider default)")
    parser.add_argument(
        "--single",
        action="store_true",
        help="Run a single query and exit (interactive mode by default)",
    )
    parser.add_argument("--query", type=str, help="Query to run in single mode")

    args = parser.parse_args()

    if args.single:
        if not args.query:
            query = input("Enter your query: ").strip()
            if not query:
                print("No query provided. Exiting.")
                return
        else:
            query = args.query

        asyncio.run(single_query_mode(query, args.provider, args.model))
    else:
        asyncio.run(interactive_mode(args.provider, args.model))


if __name__ == "__main__":
    main()
