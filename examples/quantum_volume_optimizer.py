#!/usr/bin/env python3
"""
Quantum Volume Optimizer - A Deep Agent Example

This world-class example demonstrates a sophisticated multi-agent system that finds
the optimal Quantum Volume (QV) configuration for any IBM Quantum backend.

The system uses LangChain's Deep Agents framework to coordinate multiple specialized
subagents that work together across three Qiskit MCP servers:
- qiskit-ibm-runtime-mcp-server: Backend discovery and properties
- qiskit-mcp-server: Local circuit transpilation and analysis
- qiskit-ibm-transpiler-mcp-server: AI-powered circuit optimization

## What is Quantum Volume?

Quantum Volume (QV) is a single-number metric that captures the largest random
circuit of equal width and depth that a quantum computer can successfully implement.
A QV of 2^n means the device can reliably execute n-qubit circuits of depth n.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    QUANTUM VOLUME OPTIMIZER                         │
│                      (Coordinator Agent)                            │
│                                                                     │
│  Plans strategy, coordinates subagents, synthesizes final report   │
└─────────────────────────────────────────────────────────────────────┘
                                   │
          ┌────────────────────────┼────────────────────────┐
          │                        │                        │
          ▼                        ▼                        ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  BACKEND        │    │  QUBIT CHAIN    │    │  TRANSPILER     │
│  ANALYST        │    │  OPTIMIZER      │    │  BENCHMARKER    │
│                 │    │                 │    │                 │
│  - List backends│    │  - Analyze      │    │  - Compare      │
│  - Get properties│   │    connectivity │    │    opt levels   │
│  - Find least   │    │  - Find best    │    │  - AI vs local  │
│    busy         │    │    chains       │    │    transpilation│
│  - Check status │    │  - Score by     │    │  - Generate QV  │
│                 │    │    error rates  │    │    circuits     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
        │                      │                      │
        ▼                      ▼               ┌──────┴──────┐
┌─────────────────┐    ┌─────────────────┐    ▼             ▼
│ qiskit-ibm-     │    │ qiskit-mcp-     │  qiskit-mcp-  qiskit-ibm-
│ runtime-mcp     │    │ server          │  server       transpiler-mcp
│                 │    │                 │  (local)      (AI)
│ Backend info    │    │ Local circuit   │
│ & properties    │    │ transpilation   │
└─────────────────┘    └─────────────────┘

## Optimization Strategy

1. **Backend Discovery**: Find all available backends and their properties
2. **Qubit Chain Analysis**: Identify optimal qubit chains based on:
   - Connectivity (coupling map)
   - Two-qubit gate error rates
   - Single-qubit gate error rates
   - T1/T2 coherence times
   - Readout error rates
3. **Transpilation Comparison**: For each candidate chain, compare:
   - Local transpilation (optimization levels 0-3)
   - AI routing + synthesis passes
4. **QV Circuit Generation**: Generate and optimize QV circuits
5. **Final Recommendation**: Synthesize findings into actionable report

## Prerequisites

    pip install deepagents langchain langchain-mcp-adapters python-dotenv
    pip install langchain-anthropic  # or your preferred LLM provider

## Environment Variables

    QISKIT_IBM_TOKEN: Your IBM Quantum API token
    ANTHROPIC_API_KEY: Your Anthropic API key (or other LLM provider)
    QISKIT_IBM_RUNTIME_MCP_INSTANCE: (Optional) IBM Quantum instance for faster startup

## Usage

    python quantum_volume_optimizer.py [--provider PROVIDER] [--depth DEPTH] [--backend BACKEND]

    --provider: LLM provider (anthropic, openai, google) - default: anthropic
    --depth: Maximum QV depth to evaluate (2-20) - default: 5
    --backend: Specific backend to optimize (e.g., ibm_brisbane, fake_sherbrooke)
    --interactive: Run in interactive mode for follow-up questions

## Examples

    # Auto-discover best backend, optimize for QV up to depth 5
    python quantum_volume_optimizer.py

    # Optimize specifically for ibm_brisbane, QV up to depth 12
    python quantum_volume_optimizer.py --backend ibm_brisbane --depth 12

    # Use fake backend for testing (no credentials needed)
    python quantum_volume_optimizer.py --backend fake_sherbrooke --depth 8

    # Interactive mode to ask follow-up questions
    python quantum_volume_optimizer.py --interactive --backend ibm_fez
"""

from __future__ import annotations

import argparse
import asyncio
import os
from typing import Any

# Deep Agents imports
from deepagents import create_deep_agent
from dotenv import load_dotenv

# LangChain MCP imports
from langchain_mcp_adapters.client import MultiServerMCPClient


# Load environment variables
load_dotenv()


# =============================================================================
# System Prompts for Coordinator and Subagents
# =============================================================================

COORDINATOR_SYSTEM_PROMPT = """You are the Quantum Volume Optimizer, a world-class quantum computing expert
coordinating a team of specialized agents to find the optimal Quantum Volume configuration
for IBM Quantum backends.

## Your Mission

Find the best possible Quantum Volume (QV) configuration by:
1. Discovering available backends and their properties
2. Analyzing qubit connectivity and error rates to find optimal chains
3. Comparing transpilation strategies (local vs AI-powered)
4. Generating optimized QV circuits
5. Producing a comprehensive recommendation report

## Quantum Volume Background

Quantum Volume (QV) measures a quantum computer's capability as 2^n, where n is the largest
circuit width/depth that can be executed with heavy output probability > 2/3.

Key factors affecting QV:
- Two-qubit gate fidelity (most important)
- Qubit connectivity (linear chains need SWAP gates)
- Coherence times (T1, T2)
- Readout accuracy

## Your Team

You have three specialized subagents:

1. **backend-analyst**: Expert in IBM Quantum backends
   - Lists available backends
   - Gets detailed backend properties
   - Finds least busy backends
   - Retrieves calibration data

2. **qubit-chain-optimizer**: Expert in qubit topology analysis
   - Uses algorithmic tools to find optimal qubit subsets
   - find_optimal_qv_qubits_tool: Finds densely connected subgraphs for QV
   - find_optimal_qubit_chains_tool: Finds optimal linear chains
   - Considers connectivity, error rates, and coherence times

3. **transpiler-benchmarker**: Expert in circuit optimization
   - Compares local transpilation levels (0-3)
   - Uses AI-powered routing and synthesis
   - Generates QV circuits
   - Analyzes circuit depth and gate counts

## Workflow

1. First, use the backend-analyst to discover backends and get properties
2. For the best backend(s), use qubit-chain-optimizer to find optimal chains
3. For top chains, use transpiler-benchmarker to compare optimization strategies
4. Synthesize all findings into a final recommendation

## Output Format

Your final report should include:
- **Executive Summary**: Best backend and configuration in 2-3 sentences
- **Backend Analysis**: Properties of evaluated backends
- **Optimal Qubit Chains**: Top 3 chains with scores
- **Transpilation Comparison**: Best optimization strategy
- **QV Recommendation**: Expected achievable QV with confidence
- **Detailed Configuration**: Exact qubits, optimization level, basis gates

Be thorough, data-driven, and provide actionable recommendations.
"""

BACKEND_ANALYST_PROMPT = """You are the Backend Analyst, an expert in IBM Quantum hardware.

Your role is to:
1. List all available quantum backends for the user's account
2. Get detailed properties for promising backends
3. Identify backends suitable for Quantum Volume experiments
4. Report on current queue status and availability

When analyzing backends, focus on:
- Number of qubits (need at least the target QV depth)
- Quantum volume already achieved
- Overall system status
- Queue length (prefer less busy systems)

Use the IBM Runtime MCP tools to gather this information. Report your findings
in a structured format that the coordinator can use for decision-making.
"""

QUBIT_CHAIN_OPTIMIZER_PROMPT = """You are the Qubit Chain Optimizer, an expert in quantum hardware topology.

Your role is to:
1. Analyze backend coupling maps to understand connectivity
2. Find optimal qubit subsets for Quantum Volume experiments
3. Score chains/subgraphs based on error rates and connectivity
4. Recommend the best qubit configurations for different QV depths

## Available Tools

You have access to specialized tools for qubit selection:

1. **get_coupling_map_tool**: Get the full coupling map (connectivity graph) for a backend
   - Returns edges, adjacency list, and topology info

2. **find_optimal_qv_qubits_tool**: Find optimal qubit SUBGRAPHS for QV experiments
   - Use this for QV! QV benefits from densely connected regions, not just linear chains
   - Metrics: "qv_optimized" (default), "connectivity", "gate_error"
   - Returns ranked subgraphs with connectivity ratios and error details

3. **find_optimal_qubit_chains_tool**: Find optimal LINEAR chains
   - Use this for algorithms requiring linear connectivity
   - Metrics: "two_qubit_error" (default), "readout_error", "combined"

4. **get_backend_calibration_tool**: Get detailed error rates for specific qubits

## Recommendations

- For QV experiments: Use find_optimal_qv_qubits_tool with metric="qv_optimized"
- The tool handles all the graph algorithms and scoring automatically
- Report the top 3-5 qubit subsets with their scores and connectivity ratios
"""

TRANSPILER_BENCHMARKER_PROMPT = """You are the Transpiler Benchmarker, an expert in quantum circuit optimization.

Your role is to:
1. Generate Quantum Volume circuits for specific qubit counts
2. Compare different transpilation strategies
3. Benchmark local vs AI-powered optimization
4. Find the configuration that minimizes circuit depth and two-qubit gates

Transpilation strategies to compare:
1. **Local Level 0**: No optimization (baseline)
2. **Local Level 1**: Light optimization
3. **Local Level 2**: Medium optimization (recommended default)
4. **Local Level 3**: Heavy optimization (best quality, slowest)
5. **AI Routing**: ML-based qubit routing
6. **AI Synthesis**: ML-based gate synthesis (Clifford, Linear Function)

For QV circuits, key metrics are:
- Final circuit depth (lower is better)
- Two-qubit gate count (lower is better)
- Total gate count

Use both the Qiskit MCP server (local transpilation) and the IBM Transpiler
MCP server (AI optimization) to compare approaches.

Report your findings with specific metrics for each strategy.
"""


# =============================================================================
# MCP Server Configuration
# =============================================================================


def get_mcp_servers_config() -> dict[str, dict[str, Any]]:
    """Get MCP server configuration for all Qiskit servers."""
    return {
        "qiskit-ibm-runtime": {
            "transport": "stdio",
            "command": "qiskit-ibm-runtime-mcp-server",
            "args": [],
            "env": {
                "QISKIT_IBM_TOKEN": os.getenv("QISKIT_IBM_TOKEN", ""),
                "QISKIT_IBM_RUNTIME_MCP_INSTANCE": os.getenv("QISKIT_IBM_RUNTIME_MCP_INSTANCE", ""),
            },
        },
        "qiskit": {
            "transport": "stdio",
            "command": "qiskit-mcp-server",
            "args": [],
            "env": {},
        },
        "qiskit-ibm-transpiler": {
            "transport": "stdio",
            "command": "qiskit-ibm-transpiler-mcp-server",
            "args": [],
            "env": {
                "QISKIT_IBM_TOKEN": os.getenv("QISKIT_IBM_TOKEN", ""),
            },
        },
    }


def get_llm(provider: str, model: str | None = None):
    """Get the appropriate LLM based on the provider."""
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=model or "claude-sonnet-4-20250514",
            temperature=0,
            max_tokens=8192,
        )
    elif provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=model or "gpt-4o", temperature=0)
    elif provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(model=model or "gemini-2.5-pro", temperature=0)
    else:
        raise ValueError(f"Unknown provider: {provider}")


# =============================================================================
# QV Circuit Generation Utilities
# =============================================================================


def generate_qv_qasm(num_qubits: int, depth: int | None = None, seed: int = 42) -> str:
    """Generate a true Quantum Volume circuit in QASM 3.0 format.

    Uses Qiskit's quantum_volume function which generates circuits with
    Haar-random SU(4) two-qubit gates - the proper QV protocol.

    Args:
        num_qubits: Number of qubits (QV width)
        depth: Circuit depth (defaults to num_qubits for standard QV)
        seed: Random seed for reproducibility

    Returns:
        QASM 3.0 string of the decomposed QV circuit
    """
    from qiskit.circuit.library import quantum_volume
    from qiskit.qasm3 import dumps

    # Create true QV circuit with Haar-random SU(4) gates
    qv_circuit = quantum_volume(num_qubits, depth=depth, seed=seed)

    # Decompose to basis gates and convert to QASM3
    decomposed = qv_circuit.decompose()
    return dumps(decomposed)


# =============================================================================
# Main Agent Creation
# =============================================================================


async def create_qv_optimizer_agent(
    provider: str = "anthropic",
    model: str | None = None,
    max_qv_depth: int = 5,
) -> tuple[Any, MultiServerMCPClient]:
    """Create the Quantum Volume Optimizer deep agent with all subagents.

    Args:
        provider: LLM provider to use
        model: Optional model name override
        max_qv_depth: Maximum QV depth to evaluate

    Returns:
        Tuple of (agent, mcp_client) for cleanup
    """
    print("\n" + "=" * 70)
    print("  QUANTUM VOLUME OPTIMIZER")
    print("  A Deep Agent Multi-MCP-Server Example")
    print("=" * 70)

    # Create MCP client with all servers
    mcp_config = get_mcp_servers_config()
    mcp_client = MultiServerMCPClient(mcp_config)

    print("\nInitializing MCP servers...")

    # Load tools from each server using get_tools() which creates tools that
    # manage their own sessions (new session per tool call)
    all_tools = []
    server_tools = {}

    for server_name in mcp_config.keys():
        try:
            print(f"  Connecting to {server_name}...", end=" ", flush=True)
            tools = await mcp_client.get_tools(server_name=server_name)
            server_tools[server_name] = tools
            all_tools.extend(tools)
            print(f"OK ({len(tools)} tools)")
        except Exception as e:
            print(f"FAILED: {e}")

    print(f"\nTotal tools loaded: {len(all_tools)}")

    # Get the LLM
    llm = get_llm(provider, model)
    print(f"Using LLM: {provider} ({model or 'default'})")

    # Define subagents
    backend_analyst = {
        "name": "backend-analyst",
        "description": "Expert in IBM Quantum backends. Use this agent to list backends, get properties, find least busy systems, and analyze hardware capabilities.",
        "system_prompt": BACKEND_ANALYST_PROMPT,
        "tools": server_tools.get("qiskit-ibm-runtime", []),
    }

    qubit_chain_optimizer = {
        "name": "qubit-chain-optimizer",
        "description": "Expert in qubit topology analysis. Use this agent to find optimal qubit subsets for QV experiments using algorithmic chain/subgraph finding tools.",
        "system_prompt": QUBIT_CHAIN_OPTIMIZER_PROMPT,
        "tools": server_tools.get("qiskit-ibm-runtime", []),  # Has the QV qubit finding tools
    }

    transpiler_benchmarker = {
        "name": "transpiler-benchmarker",
        "description": "Expert in circuit optimization. Use this agent to compare transpilation strategies and find the best optimization approach.",
        "system_prompt": TRANSPILER_BENCHMARKER_PROMPT,
        "tools": (server_tools.get("qiskit", []) + server_tools.get("qiskit-ibm-transpiler", [])),
    }

    # Create the coordinator agent
    print("\nCreating Quantum Volume Optimizer agent...")

    agent = create_deep_agent(
        model=llm,
        tools=all_tools,
        system_prompt=COORDINATOR_SYSTEM_PROMPT,
        subagents=[backend_analyst, qubit_chain_optimizer, transpiler_benchmarker],
    )

    print("Agent ready!\n")

    return agent, mcp_client


async def run_qv_optimization(
    provider: str = "anthropic",
    model: str | None = None,
    max_qv_depth: int = 5,
    backend: str | None = None,
) -> str:
    """Run the full Quantum Volume optimization workflow.

    Args:
        provider: LLM provider to use
        model: Optional model name override
        max_qv_depth: Maximum QV depth to evaluate (2-20)
        backend: Specific backend to optimize for (optional)

    Returns:
        The final optimization report
    """
    agent, mcp_client = await create_qv_optimizer_agent(provider, model, max_qv_depth)

    # Generate sample QV circuits for the agent to work with
    qv_circuits = {}
    for depth in range(2, min(max_qv_depth + 1, 6)):
        qv_circuits[depth] = generate_qv_qasm(depth)

    # Build backend-specific or discovery request
    if backend:
        backend_instruction = f"""
## Target Backend: {backend}

Optimize Quantum Volume configuration specifically for **{backend}**.
First verify the backend exists and get its properties, then proceed with optimization.
"""
    else:
        backend_instruction = """
## Backend Discovery

Find all available backends and identify the most promising ones for QV experiments
(consider qubit count, current QV, queue length). Focus on the top 2-3 backends.
"""

    # Tool guidance based on depth
    if max_qv_depth <= 10:
        tool_guidance = f"""
- Use **find_optimal_qv_qubits_tool** to find densely connected subgraphs (best for QV)
- The tool automatically scores by connectivity, error rates, and coherence
- Report the top 3 qubit subsets for each QV depth from 2 to {max_qv_depth}
"""
    else:
        tool_guidance = f"""
- For QV depth 2-10: Use **find_optimal_qv_qubits_tool** (dense subgraphs)
- For QV depth 11-{max_qv_depth}: Use **find_optimal_qubit_chains_tool** (linear chains)
  Note: Subgraph enumeration is expensive for >10 qubits, so we use optimized linear chains
- Report the top 3 qubit configurations for each QV depth
"""

    # Construct the optimization request
    request = f"""
I need you to find the optimal Quantum Volume configuration.
{backend_instruction}
## Objectives

1. **Verify Backend**: Get backend properties and calibration data

2. **Find Optimal Qubits**: Use algorithmic tools to find optimal qubit configurations
   for QV depths 2 through {max_qv_depth}:
{tool_guidance}
3. **Compare Transpilation**: For the best qubit configurations, compare:
   - Local transpilation (levels 0, 1, 2, 3)
   - AI-powered routing and synthesis
   - Focus on minimizing two-qubit gate count and circuit depth

4. **Generate Recommendation**: Produce a detailed report with:
   - Backend analysis with key metrics
   - Optimal qubit subset for each QV depth (with scores from the tools)
   - Best transpilation strategy
   - Expected achievable QV with confidence level

## Sample QV Circuits

Here are sample QV circuits you can use for transpilation comparison:

### QV-2 (2 qubits, depth 2):
```qasm
{qv_circuits.get(2, "N/A")}
```

### QV-3 (3 qubits, depth 3):
```qasm
{qv_circuits.get(3, "N/A")}
```

### QV-4 (4 qubits, depth 4):
```qasm
{qv_circuits.get(4, "N/A")}
```

Please proceed with the analysis and provide your comprehensive recommendation.
"""

    print("=" * 70)
    print("  STARTING QUANTUM VOLUME OPTIMIZATION")
    print("=" * 70)
    print(f"\nTarget backend: {backend or 'Auto-discover best'}")
    print(f"Target QV depth: up to {max_qv_depth}")
    print("\nThis may take several minutes as the agents analyze your backends...\n")
    print("-" * 70)

    # Run the agent
    result = await agent.ainvoke({"messages": [{"role": "user", "content": request}]})

    # Extract final response
    messages = result.get("messages", [])
    if messages:
        final_response = messages[-1].content
    else:
        final_response = "No response generated."

    print("\n" + "=" * 70)
    print("  OPTIMIZATION COMPLETE")
    print("=" * 70)

    return final_response


async def interactive_mode(provider: str, model: str | None, backend: str | None = None) -> None:
    """Run interactive mode where users can ask follow-up questions."""
    from langchain_core.messages import HumanMessage

    agent, mcp_client = await create_qv_optimizer_agent(provider, model, 20)

    print("\n" + "-" * 70)
    print("Interactive Mode - Ask questions about Quantum Volume optimization")
    if backend:
        print(f"Target backend: {backend}")
    print("Type 'quit' to exit, 'clear' to reset history, 'optimize' to run full optimization")
    print("-" * 70 + "\n")

    # Maintain conversation history for context
    history: list = []

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

            if query.lower() == "optimize":
                if backend:
                    query = f"""Run the full Quantum Volume optimization for {backend}:
                    1. Get {backend} properties and calibration
                    2. Find optimal qubit subsets using find_optimal_qv_qubits_tool
                    3. Compare transpilation strategies
                    4. Generate recommendation report"""
                else:
                    query = """Run the full Quantum Volume optimization workflow:
                    1. List available backends and pick the best ones
                    2. Find optimal qubit subsets using find_optimal_qv_qubits_tool
                    3. Compare transpilation strategies
                    4. Generate recommendation report"""

            # Build messages with history
            messages = list(history) if history else []
            messages.append(HumanMessage(content=query))

            print("\nThinking...\n")

            result = await agent.ainvoke({"messages": messages})

            result_messages = result.get("messages", [])
            if result_messages:
                response = result_messages[-1].content
                # Update history with full conversation from agent
                history = result_messages
                print(f"Assistant:\n{response}\n")

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Quantum Volume Optimizer - Find optimal QV configuration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-discover best backend
  python quantum_volume_optimizer.py

  # Optimize for a specific backend with higher QV depth
  python quantum_volume_optimizer.py --backend ibm_brisbane --depth 15

  # Test with fake backend (no credentials needed)
  python quantum_volume_optimizer.py --backend fake_sherbrooke --depth 8

  # Interactive mode for follow-up questions
  python quantum_volume_optimizer.py --interactive --backend ibm_fez
        """,
    )
    parser.add_argument(
        "--provider",
        choices=["anthropic", "openai", "google"],
        default="anthropic",
        help="LLM provider to use (default: anthropic)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name to use (provider-specific)",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=5,
        choices=range(2, 21),
        metavar="[2-20]",
        help="Maximum QV depth to evaluate (default: 5, max: 20)",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default=None,
        help="Specific backend to optimize for (e.g., 'ibm_brisbane', 'fake_sherbrooke')",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode for follow-up questions",
    )

    args = parser.parse_args()

    # Verify required environment variables
    if not os.getenv("QISKIT_IBM_TOKEN"):
        print("Error: QISKIT_IBM_TOKEN environment variable not set")
        print("Get your token from https://quantum.ibm.com/")
        return

    provider_env_vars = {
        "anthropic": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
        "google": "GOOGLE_API_KEY",
    }

    env_var = provider_env_vars[args.provider]
    if not os.getenv(env_var):
        print(f"Error: {env_var} environment variable not set")
        return

    # Run the appropriate mode
    if args.interactive:
        asyncio.run(interactive_mode(args.provider, args.model, args.backend))
    else:
        result = asyncio.run(
            run_qv_optimization(args.provider, args.model, args.depth, args.backend)
        )
        print(result)


if __name__ == "__main__":
    main()
