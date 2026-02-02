#!/usr/bin/env python3
"""
LangChain Agent Example for Qiskit Gym MCP Server

This example demonstrates how to create an AI agent using LangChain that
connects to the qiskit-gym-mcp-server via the Model Context Protocol (MCP).

The agent can interact with qiskit-gym to:
- Create RL environments for quantum circuit synthesis
- Train models using PPO or AlphaZero algorithms
- Synthesize optimal circuits using trained models
- Manage models (save, load, list, delete)

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
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# LangChain imports
from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools


# Load environment variables from .env file (relative to this script's location)
_script_dir = Path(__file__).parent
load_dotenv(_script_dir / ".env")


# System prompt for the qiskit-gym agent
SYSTEM_PROMPT = """You are a helpful quantum computing assistant with access to qiskit-gym's
reinforcement learning-based circuit synthesis through the MCP server.

You can help users train RL models and synthesize optimal quantum circuits.

## IMPORTANT: Three Problem Types (Choose Correctly!)

There are THREE distinct environment types. Pay attention to what the user asks for:

1. **Permutation** (create_permutation_env_tool):
   - For QUBIT ROUTING / SWAP gate synthesis
   - Input: A permutation like [2, 0, 1, 3] meaning qubit 0→2, 1→0, 2→1, 3→3
   - Output: Optimal SWAP circuit to achieve that permutation
   - Keywords: "permutation", "swap", "routing", "qubit mapping"

2. **LinearFunction** (create_linear_function_env_tool):
   - For CNOT circuit synthesis of LINEAR BOOLEAN FUNCTIONS
   - Input: An invertible binary matrix representing the linear function
   - Output: Optimal CNOT-only circuit
   - Keywords: "linear function", "CNOT", "cx", "linear reversible", "parity network"

3. **Clifford** (create_clifford_env_tool):
   - For CLIFFORD CIRCUIT synthesis (H, S, CNOT gates)
   - Input: A Clifford tableau
   - Output: Optimal Clifford circuit
   - Keywords: "clifford", "stabilizer", "H+S+CNOT"

**When the user says "linear", determine if they mean:**
- "linear topology" → refers to the COUPLING MAP shape (a line: 0-1-2-3)
- "linear function" → refers to the LinearFunction ENVIRONMENT TYPE

## Environment Creation
- create_permutation_env_tool: Create PermutationGym for SWAP routing
- create_linear_function_env_tool: Create LinearFunctionGym for CNOT synthesis
- create_clifford_env_tool: Create CliffordGym for Clifford circuit synthesis
- list_environments_tool: List active environments
- delete_environment_tool: Remove an environment

## Training
- start_training_tool: Start RL training (supports background=True for async training)
- batch_train_environments_tool: Train multiple environments (supports background=True)
- get_training_status_tool: Check training progress
- wait_for_training_tool: Wait for background training to complete
- stop_training_tool: Stop a running training session
- list_training_sessions_tool: List all training sessions

## Synthesis
- synthesize_permutation_tool: Generate optimal SWAP circuit for a permutation
- synthesize_linear_function_tool: Generate optimal CNOT circuit
- synthesize_clifford_tool: Generate optimal Clifford circuit

## Model Management
- save_model_tool: Save trained model to disk
- load_model_tool: Load model from disk
- list_saved_models_tool: List models on disk
- list_loaded_models_tool: List models in memory

## TensorBoard Visualization
- start_tensorboard_tool: Start TensorBoard to visualize training metrics
- stop_tensorboard_tool: Stop the running TensorBoard process
- get_tensorboard_status_tool: Check if TensorBoard is running and on which port

## Utility Tools
- generate_random_permutation_tool: Generate random permutation for testing
- generate_random_linear_function_tool: Generate random linear function
- generate_random_clifford_tool: Generate random Clifford element

## Hardware Presets
Available presets for coupling maps:
- linear_5, linear_10: Linear chain topologies
- grid_3x3, grid_5x5: Grid topologies
- ibm_heron_r1, ibm_heron_r2: IBM Heron heavy-hex topology
- ibm_nighthawk: IBM Nighthawk 10x12 grid

## RL Algorithms
- ppo: Proximal Policy Optimization (recommended, fast)
- alphazero: MCTS with neural networks (better for complex problems, slower)

## Policy Networks
- basic: Simple feedforward network (good for <8 qubits)
- conv1d: 1D convolutional network (better for larger problems)

## IMPORTANT: Background Training
**ALWAYS use background=True for training** to avoid connection timeouts:
- start_training_tool(..., background=True) - returns immediately with session_id
- batch_train_environments_tool(..., background=True) - returns immediately with session_ids
- Use get_training_status_tool(session_id) to check progress
- Use wait_for_training_tool(session_id) to block until complete

Only use synchronous training (background=False) for very short demos (< 10 iterations).

## Workflow Tips
1. **Always use background=True** for any real training
2. For batch training multiple environments, use batch_train_environments_tool with background=True
3. Poll progress with get_training_status_tool or wait with wait_for_training_tool
4. Use start_tensorboard_tool to visualize training metrics in real-time
5. Save models you want to keep with save_model_tool
6. Use list_training_sessions_tool to see all running/completed training

When a user asks to train a model:
1. Create an appropriate environment using create_*_env_tool
2. Optionally start TensorBoard with start_tensorboard_tool for visualization
3. Start training with start_training_tool(..., background=True)
4. Use get_training_status_tool to check progress periodically
5. When complete, save the model if needed with save_model_tool
6. Stop TensorBoard with stop_tensorboard_tool when done
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
    """Create and return an MCP client configured for the Qiskit Gym MCP server."""
    # Pass relevant environment variables to the MCP server subprocess
    server_env = {}
    for key in [
        "QISKIT_GYM_MODEL_DIR",
        "QISKIT_GYM_TENSORBOARD_DIR",
        "QISKIT_GYM_MAX_QUBITS",
        "QISKIT_GYM_MAX_ITERATIONS",
        "QISKIT_GYM_MAX_SEARCHES",
    ]:
        if key in os.environ:
            server_env[key] = os.environ[key]

    return MultiServerMCPClient(
        {
            "qiskit-gym": {
                "transport": "stdio",
                "command": "qiskit-gym-mcp-server",
                "args": [],
                "env": server_env,
            }
        }
    )


async def create_gym_agent_with_session(
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
    print("Qiskit Gym Agent - Interactive Mode")
    print("=" * 60)
    print(f"\nUsing LLM provider: {provider}")
    if model:
        print(f"Using model: {model}")
    print("\nStarting MCP server and creating agent...")
    print("(This may take a few seconds on first run)\n")

    mcp_client = get_mcp_client()

    # Use persistent session for efficient tool calls
    async with mcp_client.session("qiskit-gym") as session:
        agent = await create_gym_agent_with_session(session, provider, model)

        print("\n" + "-" * 60)
        print("Agent ready! You can train RL models and synthesize quantum circuits.")
        print("\nExample queries:")
        print("  - 'Create a permutation environment for a 5-qubit linear chain'")
        print("  - 'Train a model with PPO for 50 iterations'")
        print("  - 'Train in background with 200 iterations and check status'")
        print("  - 'Start TensorBoard to monitor training'")
        print("  - 'Generate a random permutation and synthesize a circuit'")
        print("  - 'Save the model as my_router'")
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

                print("\nAssistant: ", end="", flush=True)
                response, history = await run_agent_query(agent, query, history)
                print(response)
                print()

            except KeyboardInterrupt:
                print("\n\nInterrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\nError: {type(e).__name__}: {e}")
                # Show more details for debugging
                import traceback
                traceback.print_exc()
                print("Please try again.\n")


async def single_query_mode(provider: str, model: str | None) -> None:
    """Run a single demonstration query.

    Args:
        provider: The LLM provider to use
        model: Optional model name override
    """
    print("\n" + "=" * 60)
    print("Qiskit Gym Agent - Single Query Mode")
    print("=" * 60)

    mcp_client = get_mcp_client()

    async with mcp_client.session("qiskit-gym") as session:
        agent = await create_gym_agent_with_session(session, provider, model)

        query = """Please help me train an RL model for qubit routing:
1. Create a permutation environment for a 5-qubit linear chain topology
2. Start training with PPO for 20 iterations
3. Show me the training results

Use synchronous training since it's only 20 iterations."""

        print(f"\nQuery: {query}\n")
        print("-" * 60)
        print("\nAssistant: ", end="", flush=True)
        response, _ = await run_agent_query(agent, query)
        print(response)


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="LangChain Agent for Qiskit Gym MCP Server"
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
