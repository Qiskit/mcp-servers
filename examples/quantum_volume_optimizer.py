#!/usr/bin/env python3
"""
Quantum Volume Finder - A Deep Agent Example

This example demonstrates a multi-agent system that finds the highest achievable
Quantum Volume (QV) for IBM Quantum backends through actual hardware execution.

Unlike simple analysis tools, this agent RUNS experiments and reports ACTUAL results.
It uses a top-down strategy: start at the highest requested depth and work down
until it finds a depth that passes the QV criteria (HOP > 2/3).

## What is Quantum Volume?

Quantum Volume (QV) 2^n is achieved when:
- Running n-qubit, depth-n random circuits
- Heavy Output Probability (HOP) > 2/3
- HOP = (shots resulting in heavy outputs) / (total shots)

## Two Modes: Quick Test vs Full Protocol

### Single-Circuit Mode (default)
For each depth, generates one random QV circuit, runs it once on hardware, and checks
if HOP > 2/3. This gives a quick signal but is NOT statistically rigorous. A lucky
circuit can yield a false positive, and a good backend can fail on one unlucky circuit.

### Full QV Protocol (--num-circuits N)
Implements the standard QV protocol (arXiv:1811.12926):
1. Generates N **independent** random QV circuits per depth (each with a different seed)
2. Runs each circuit on hardware via the `run_qv_depth_trial` batch tool
3. Computes individual HOP for each circuit
4. Applies a one-sided confidence interval test: the lower bound of the 97.5% CI of
   the mean HOP must exceed 2/3
5. Only then is QV officially "achieved" for that depth

Use `--num-circuits 100` (or more) for official QV certification.

## Strategy: Top-Down Search

1. Start at max_depth (e.g., 5)
2. Run QV circuit on hardware
3. Calculate HOP from measurement results
4. If HOP > 2/3: SUCCESS! QV 2^n achieved
5. If HOP <= 2/3: Try depth-1
6. Repeat until success or depth 2

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    QUANTUM VOLUME FINDER                            │
│                      (Coordinator Agent)                            │
└─────────────────────────────────────────────────────────────────────┘
                                   │
       ┌───────────────────────────┼───────────────────────────┐
       │                           │                           │
       ▼                           ▼                           ▼
┌─────────────────┐ ┌─────────────────────┐ ┌─────────────────────┐
│  BACKEND        │ │ QUBIT CHAIN         │ │ QV EXPERIMENT       │
│  ANALYST        │ │ OPTIMIZER           │ │ RUNNER              │
│                 │ │                     │ │                     │
│ Get backend     │ │ Searches ALL qubits │ │ Transpile circuit   │
│ properties      │ │ on backend          │ │ Submit job          │
│                 │ │                     │ │ Poll for completion │
└─────────────────┘ └─────────────────────┘ └─────────────────────┘
```

## Prerequisites

    pip install deepagents langchain langchain-mcp-adapters python-dotenv
    pip install langchain-anthropic  # or your preferred LLM provider
    pip install qiskit-mcp-servers

## Environment Variables

    QISKIT_IBM_TOKEN: Your IBM Quantum API token
    ANTHROPIC_API_KEY: Your Anthropic API key (or other LLM provider)

## Usage

    # Find highest QV for a backend (single-circuit quick test)
    python quantum_volume_optimizer.py --backend ibm_brisbane --depth 5

    # Full QV protocol with 100 circuits per depth (statistically rigorous)
    python quantum_volume_optimizer.py --backend ibm_brisbane --depth 5 --num-circuits 100

    # Analysis only (no hardware execution)
    python quantum_volume_optimizer.py --backend ibm_brisbane --depth 5 --no-experiment

    # Interactive mode
    python quantum_volume_optimizer.py --backend ibm_brisbane --interactive

## Output

The agent reports ACTUAL results:
- Each depth attempted
- Qubits used (from find_optimal_qv_qubits_tool)
- Job ID, HOP value
- PASS/FAIL for each depth
- Final achieved QV
"""

from __future__ import annotations

import argparse
import json
import asyncio
import os
import sys
from datetime import datetime
from typing import Any

# Deep Agents imports
from deepagents import create_deep_agent
from dotenv import load_dotenv

# LangChain imports for callbacks
from langchain_core.callbacks import BaseCallbackHandler

# LangChain MCP imports
from langchain_mcp_adapters.client import MultiServerMCPClient


# Load environment variables (override=True so .env takes precedence over shell env)
load_dotenv(override=True)


# =============================================================================
# Callback Handler for Agent Observability
# =============================================================================


class AgentActivityHandler(BaseCallbackHandler):
    """Callback handler that shows what the agent is doing during execution.

    This handler provides real-time visibility into:
    - Tool calls (which tool, what arguments)
    - Tool results (success/failure, key info)
    - LLM chain starts and completions
    - Agent actions and reasoning
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.indent_level = 0
        self.current_tool = None

    def _timestamp(self) -> str:
        return datetime.now().strftime("%H:%M:%S")

    def _print(self, msg: str, color: str = "") -> None:
        """Print with optional color and indentation."""
        indent = "  " * self.indent_level
        if color and sys.stdout.isatty():
            colors = {
                "blue": "\033[94m",
                "green": "\033[92m",
                "yellow": "\033[93m",
                "red": "\033[91m",
                "cyan": "\033[96m",
                "magenta": "\033[95m",
                "reset": "\033[0m",
            }
            print(f"{colors.get(color, '')}{indent}{msg}{colors['reset']}", flush=True)
        else:
            print(f"{indent}{msg}", flush=True)

    def on_tool_start(self, serialized: dict | None, input_str: str, **kwargs) -> None:
        """Called when a tool starts running."""
        tool_name = serialized.get("name", "unknown_tool") if serialized else "unknown_tool"
        self.current_tool = tool_name

        self._print(f"\n[{self._timestamp()}] 🔧 TOOL: {tool_name}", "cyan")
        self.indent_level += 1

        # Show input (truncated for readability)
        if self.verbose and input_str:
            input_preview = str(input_str)[:200]
            if len(str(input_str)) > 200:
                input_preview += "..."
            self._print(f"Input: {input_preview}", "blue")

    def on_tool_end(self, output: str, **kwargs) -> None:
        """Called when a tool finishes."""
        # Show output preview
        if self.verbose and output:
            output_preview = str(output)[:300]
            if len(str(output)) > 300:
                output_preview += "..."
            self._print(f"Output: {output_preview}", "green")

        self.indent_level = max(0, self.indent_level - 1)
        self._print(f"[{self._timestamp()}] ✓ {self.current_tool} complete", "green")

    def on_tool_error(self, error: Exception, **kwargs) -> None:
        """Called when a tool errors."""
        self.indent_level = max(0, self.indent_level - 1)
        self._print(f"[{self._timestamp()}] ✗ {self.current_tool} failed: {error}", "red")

    def on_chain_start(self, serialized: dict | None, inputs: dict, **kwargs) -> None:
        """Called when a chain starts."""
        if serialized is None:
            return

        chain_name = serialized.get("name") or serialized.get("id", ["unknown"])[-1]

        # Only show interesting chains (not internal ones)
        if chain_name in ["AgentExecutor", "RunnableSequence", "unknown"]:
            return

        self._print(f"\n[{self._timestamp()}] ⚡ Starting: {chain_name}", "magenta")
        self.indent_level += 1

    def on_chain_end(self, outputs: dict, **kwargs) -> None:
        """Called when a chain ends."""
        self.indent_level = max(0, self.indent_level - 1)

    def on_agent_action(self, action, **kwargs) -> None:
        """Called when the agent takes an action."""
        tool = getattr(action, "tool", "unknown")
        tool_input = getattr(action, "tool_input", {})

        self._print(f"\n[{self._timestamp()}] 🤖 Agent calling: {tool}", "yellow")

        if self.verbose and tool_input and isinstance(tool_input, dict):
            # Show key info from input
            for key, value in list(tool_input.items())[:3]:
                val_preview = str(value)[:100]
                if len(str(value)) > 100:
                    val_preview += "..."
                self._print(f"  {key}: {val_preview}", "blue")

    def on_agent_finish(self, finish, **kwargs) -> None:
        """Called when the agent finishes."""
        self._print(f"\n[{self._timestamp()}] ✅ Agent finished", "green")

    def on_llm_start(self, serialized: dict | None, prompts: list, **kwargs) -> None:
        """Called when LLM starts generating."""
        if self.verbose:
            if serialized:
                model = serialized.get("name") or serialized.get("id", ["LLM"])[-1]
            else:
                model = "LLM"
            self._print(f"[{self._timestamp()}] 💭 {model} thinking...", "blue")


# =============================================================================
# System Prompts for Coordinator and Subagents
# =============================================================================

COORDINATOR_SYSTEM_PROMPT = """
╔══════════════════════════════════════════════════════════════════════════════╗
║  MANDATORY: task() REQUIRES description PARAMETER - NEVER OMIT IT!           ║
║                                                                              ║
║  task(subagent_type="X", description="Y")  ← BOTH parameters REQUIRED        ║
╚══════════════════════════════════════════════════════════════════════════════╝

╔══════════════════════════════════════════════════════════════════════════════╗
║  YOU MUST ACTUALLY RUN EXPERIMENTS - NOT JUST ANALYZE OR DOCUMENT!           ║
║                                                                              ║
║  DO NOT write reports about "what you would do"                              ║
║  DO NOT say "ready for execution" - EXECUTE IT!                              ║
║  DO NOT create documentation files - SUBMIT JOBS!                            ║
╚══════════════════════════════════════════════════════════════════════════════╝

You are the Quantum Volume Finder. Your job is to EXECUTE experiments and report ACTUAL results.

## REQUIRED WORKFLOW (DO ALL STEPS)

1. Get backend info → **backend-analyst**
2. Find optimal qubits for the depth → **qubit-chain-optimizer**
3. Run experiment on hardware → **qv-experiment-runner** (transpiles, submits, polls, returns job_id)
4. Calculate HOP → call calculate_hop(job_id=<job_id>, depth=N)
5. If HOP above threshold: SUCCESS. If not: try depth-1 (repeat from step 2)

## Subagents — COPY THESE FORMATS EXACTLY

### backend-analyst
```
task(subagent_type="backend-analyst", description="Get ibm_boston properties")
```

### qubit-chain-optimizer
```
task(subagent_type="qubit-chain-optimizer", description="Find 5 optimal qubits for QV-5 on ibm_boston")
```

### qv-experiment-runner
Pass depth, backend, and initial_layout (qubits). The subagent transpiles the QV circuit,
submits it to hardware, polls for completion, and returns the job_id.
```
task(subagent_type="qv-experiment-runner", description="Run QV experiment: depth=5, backend_name=ibm_boston, initial_layout=[47, 57, 66, 67, 68]")
```

## HOP Calculation

After the experiment runner returns the job_id:
1. Call calculate_hop(job_id=<job_id from runner>, depth=N)
   This fetches the job results and looks up heavy outputs automatically.
2. Check if above_threshold is true

## Critical Rules

1. DO NOT make recommendations - RUN the experiments
2. DO NOT stop after one failure - try lower depths
3. DO NOT limit qubit search - use all qubits on the backend
4. ALWAYS report actual job ID, measurement counts, and HOP
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

QUBIT_CHAIN_OPTIMIZER_PROMPT = """You are the Qubit Chain Optimizer, finding optimal qubits for QV experiments.

## Primary Tool: find_optimal_qv_qubits_tool

This tool searches the ENTIRE backend (all 127+ qubits) to find the best subgraphs.
It does NOT limit to just the first 10 qubits - it analyzes ALL qubits.

### Usage
```
find_optimal_qv_qubits_tool(
    backend_name="ibm_brisbane",
    num_qubits=5,       # QV depth
    num_results=10,     # Get 10 candidates to try
    metric="qv_optimized"
)
```

### Important Parameters
- **num_qubits**: The QV depth (e.g., 5 for QV-32)
- **num_results**: Request at least 10 results to have fallback options
- **metric**: Use "qv_optimized" for best QV performance

### What It Returns
- Ranked list of qubit subsets across the ENTIRE backend
- Each result includes: qubits list, score, connectivity_ratio, edge_errors
- Lower score = better (combines gate errors, coherence, connectivity)

## Secondary Tools

- **find_optimal_qubit_chains_tool**: For linear connectivity experiments
- **get_coupling_map_tool**: To visualize backend topology
- **get_backend_calibration_tool**: For detailed error analysis

## Output

Return the top 10 qubit subsets with their scores. The coordinator will use
these to run QV experiments, potentially trying multiple if the first fails.
"""

COORDINATOR_MULTI_CIRCUIT_APPENDIX = """

## Multi-Circuit QV Mode (Full Protocol)

You have the run_qv_depth_trial tool for statistically rigorous QV testing.
This runs N independent random circuits per depth and performs the full statistical
confidence interval test (arXiv:1811.12926).

### Workflow Change
- Steps 1-2 are the same (backend analysis, find optimal qubits)
- Step 3: Instead of qv-experiment-runner, call:
  run_qv_depth_trial(depth=N, backend_name=<backend>, initial_layout=[q1, q2, ...], num_circuits=<N>, shots=4096)
  This tool handles everything internally: generates N random circuits, transpiles each,
  submits each to hardware, polls for completion, computes all HOPs, and runs the
  statistical CI test. It prints progress as it works.
- Step 4: Check qv_achieved in the result (replaces manual HOP check)
  - qv_achieved=true means the lower bound of the 97.5% CI exceeds 2/3 — QV is officially achieved
  - qv_achieved=false means it failed the statistical test — try depth-1
- Step 5: If not achieved, try depth-1 (find new optimal qubits first)

### DO NOT use qv-experiment-runner in multi-circuit mode — use run_qv_depth_trial instead.

### Reading Results
- qv_achieved: true/false (the CI test result)
- mean_hop: average HOP across all circuits
- ci_lower: lower bound of 97.5% confidence interval (must be > 2/3 for QV)
- num_successful: how many circuits completed successfully
- individual_hops: list of all HOP values
- message: human-readable summary
"""

QV_EXPERIMENT_RUNNER_PROMPT = """You run QV experiments on hardware.

Your task description contains: depth, backend_name, and initial_layout (qubits).

## WORKFLOW — Follow these steps IN ORDER:

### STEP 1: Transpile the QV circuit
```
transpile_qv_circuit(depth=<depth>, backend_name=<backend>, optimization_level=3, initial_layout=<qubits>)
```
This generates the QV circuit and transpiles it. The result shows transpilation metrics.

### STEP 2: Submit to hardware
```
submit_qv_job(depth=<depth>, backend_name=<backend>, shots=4096)
```
This submits the transpiled circuit. Returns a job_id.

### STEP 3: Wait for completion
Poll get_job_status_tool(job_id=<id>) every call until job_status is "DONE".

### STEP 4: Report back
Return ALL of: backend, depth, qubits, job_id, shots.
The coordinator will fetch results and calculate HOP using the job_id.
"""


# =============================================================================
# MCP Server Configuration
# =============================================================================


def get_mcp_servers_config() -> dict[str, dict[str, Any]]:
    """Get MCP server configuration for Qiskit servers.

    Note: We only include qiskit-ibm-runtime and qiskit-ibm-transpiler.
    The qiskit-mcp-server is intentionally excluded because its transpile_circuit_tool
    requires coupling_map/basis_gates parameters, while hybrid_ai_transpile_tool
    from qiskit-ibm-transpiler accepts backend_name directly (simpler for agents).
    """
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


def generate_qv_circuit_with_ideal_distribution(
    num_qubits: int,
    depth: int | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """Generate a Quantum Volume circuit and compute its ideal heavy output bitstrings.

    This function creates a random QV circuit using Qiskit's quantum_volume function
    and simulates it to determine which output bitstrings are "heavy" (have above-median
    probability in the ideal distribution).

    The heavy outputs are required for calculating the Heavy Output Probability (HOP)
    when the circuit is run on real hardware.

    Args:
        num_qubits: Number of qubits for the QV circuit (2-20 supported)
        depth: Depth of the QV circuit (number of SU(4) layers). If None, defaults
               to num_qubits (square QV circuit). Range: 1 to num_qubits.
        seed: Random seed for reproducible circuit generation. If None, uses
              a random seed.

    Returns:
        Dictionary containing:
        - status: "success" or "error"
        - circuit_qasm: The QV circuit in QASM3 format
        - num_qubits: Number of qubits
        - depth: Depth used
        - seed: The random seed used (for reproducibility)
        - heavy_outputs: List of bitstrings with above-median ideal probability
        - num_heavy_outputs: Number of heavy output bitstrings
        - ideal_probabilities: Dict of all bitstrings and their ideal probabilities
          (for analysis, included for circuits with <= 6 qubits)

    Note:
        This function performs classical simulation which scales as O(2^n).
        For num_qubits > 20, simulation becomes computationally expensive.
    """
    import logging

    import numpy as np
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import quantum_volume
    from qiskit.qasm3 import dumps
    from qiskit.quantum_info import Statevector

    logger = logging.getLogger(__name__)

    try:
        # Validate and set defaults
        if num_qubits < 2:
            num_qubits = 2
        elif num_qubits > 20:
            # Warn but allow - simulation will be slow
            logger.warning(
                f"QV with {num_qubits} qubits will be slow to simulate. "
                "Consider using <= 20 qubits."
            )

        if depth is None:
            depth = num_qubits
        elif depth < 1:
            depth = 1
        elif depth > num_qubits:
            depth = num_qubits

        # Generate random seed if not provided
        if seed is None:
            seed = np.random.randint(0, 2**31)

        # Generate QV circuit
        qv_circuit = quantum_volume(num_qubits, depth=depth, seed=seed)

        # Get the unitary circuit (without measurements) for simulation
        # We need to decompose to get standard gates
        qv_decomposed = qv_circuit.decompose()

        # Simulate to get ideal probabilities
        # Start with |0...0⟩ state
        statevector = Statevector.from_label("0" * num_qubits)
        final_state = statevector.evolve(qv_decomposed)

        # Get probabilities for all computational basis states
        probabilities = final_state.probabilities()

        # Create mapping from bitstring to probability
        ideal_probs = {}
        for i, prob in enumerate(probabilities):
            # Format as bitstring matching Qiskit convention (qubit 0 = rightmost)
            bitstring = format(i, f"0{num_qubits}b")
            ideal_probs[bitstring] = prob

        # Find median probability
        median_prob = float(np.median(probabilities))

        # Heavy outputs are those with above-median probability
        heavy_outputs = [bs for bs, prob in ideal_probs.items() if prob > median_prob]

        # Add measurements to circuit for execution
        qv_with_meas = QuantumCircuit(num_qubits, num_qubits)
        qv_with_meas.compose(qv_decomposed, inplace=True)
        qv_with_meas.measure(range(num_qubits), range(num_qubits))

        # Convert to QASM3
        qasm3_circuit = dumps(qv_with_meas)

        result = {
            "status": "success",
            "circuit_qasm": qasm3_circuit,
            "num_qubits": num_qubits,
            "depth": depth,
            "seed": seed,
            "heavy_outputs": heavy_outputs,
            "num_heavy_outputs": len(heavy_outputs),
            "median_probability": median_prob,
            "message": f"Generated QV-{num_qubits} circuit with {len(heavy_outputs)} heavy outputs",
        }

        # Include ideal probabilities for small circuits (useful for analysis)
        if num_qubits <= 6:
            result["ideal_probabilities"] = ideal_probs

        return result

    except ImportError as e:
        logger.error(f"Missing required package for QV generation: {e}")
        return {
            "status": "error",
            "message": f"Missing required package: {e!s}. Ensure qiskit is installed.",
        }
    except Exception as e:
        logger.error(f"Failed to generate QV circuit: {e}")
        return {"status": "error", "message": f"Failed to generate QV circuit: {e!s}"}


def calculate_heavy_output_probability(
    counts: dict[str, int],
    heavy_outputs: list[str],
) -> dict[str, Any]:
    """Calculate the Heavy Output Probability (HOP) for Quantum Volume validation.

    The Heavy Output Probability is a key metric in Quantum Volume experiments.
    It measures the fraction of measurement outcomes that correspond to "heavy"
    bitstrings - those with above-median probability in the ideal distribution.

    For a successful QV experiment:
    - Generate many random QV circuits
    - For each circuit, determine the ideal heavy outputs (via simulation)
    - Run on hardware and calculate HOP
    - If mean(HOP) > 2/3 with statistical confidence, QV is achieved

    Args:
        counts: Dictionary of measurement outcomes and their counts
                (e.g., {"00": 2048, "11": 2048} from get_job_results)
        heavy_outputs: List of bitstrings that are "heavy" (above-median probability
                      in ideal simulation). These must be computed from ideal
                      simulation of the specific QV circuit.

    Returns:
        Dictionary containing:
        - status: "success" or "error"
        - heavy_output_probability: Fraction of shots resulting in heavy outputs (0.0 to 1.0)
        - total_shots: Total number of measurement shots
        - heavy_counts: Number of shots that were heavy outputs
        - threshold: The 2/3 threshold for QV success
        - above_threshold: Boolean indicating if HOP > 2/3
        - message: Status message

    Example:
        # After running a QV circuit and getting results:
        result = calculate_heavy_output_probability(
            counts={"00": 1500, "01": 300, "10": 200, "11": 2000},
            heavy_outputs=["00", "11"]  # From ideal simulation
        )
        # result["heavy_output_probability"] = (1500 + 2000) / 4000 = 0.875
    """
    import logging

    logger = logging.getLogger(__name__)

    try:
        if not counts:
            return {
                "status": "error",
                "message": "No counts provided",
            }

        if not heavy_outputs:
            return {
                "status": "error",
                "message": "No heavy outputs provided. Heavy outputs must be computed "
                "from ideal simulation of the QV circuit.",
            }

        # Convert heavy_outputs to a set for O(1) lookup
        heavy_set = set(heavy_outputs)

        # Calculate total shots and heavy counts
        total_shots = sum(counts.values())
        heavy_counts = sum(count for bitstring, count in counts.items() if bitstring in heavy_set)

        # Calculate HOP
        hop = heavy_counts / total_shots if total_shots > 0 else 0.0

        # QV threshold is 2/3
        threshold = 2 / 3
        above_threshold = hop > threshold

        return {
            "status": "success",
            "heavy_output_probability": hop,
            "total_shots": total_shots,
            "heavy_counts": heavy_counts,
            "num_heavy_bitstrings": len(heavy_outputs),
            "threshold": threshold,
            "above_threshold": above_threshold,
            "message": f"HOP = {hop:.4f} ({'above' if above_threshold else 'below'} threshold of {threshold:.4f})",
        }

    except Exception as e:
        logger.error(f"Failed to calculate heavy output probability: {e}")
        return {"status": "error", "message": f"Failed to calculate HOP: {e!s}"}


def analyze_qv_experiment_results(
    hop_values: list[float],
    confidence_level: float = 0.975,
) -> dict[str, Any]:
    """Analyze results from multiple Quantum Volume circuit runs.

    Used by the `run_qv_depth_trial` batch tool in multi-circuit mode (--num-circuits N)
    to perform the statistical confidence interval test required by the full QV protocol.

    After running multiple QV circuits on hardware and calculating their individual
    Heavy Output Probabilities (HOP), this function performs statistical analysis
    to determine if the QV benchmark is achieved.

    QV Success Criteria (per the standard QV protocol, arXiv:1811.12926):
    - Mean HOP must be > 2/3
    - The lower bound of the confidence interval must be > 2/3 (with given confidence)

    Args:
        hop_values: List of Heavy Output Probability values from individual QV circuits.
                   Should have at least 100 values for statistical significance.
        confidence_level: Confidence level for the statistical test (default: 0.975
                         for one-sided 97.5% confidence, matching IBM's QV protocol)

    Returns:
        Dictionary containing:
        - status: "success" or "error"
        - qv_achieved: Boolean indicating if QV benchmark is passed
        - mean_hop: Mean Heavy Output Probability across all circuits
        - std_hop: Standard deviation of HOP values
        - confidence_interval: (lower, upper) bounds
        - num_circuits: Number of circuits analyzed
        - threshold: The 2/3 threshold
        - margin: How far the lower confidence bound is from threshold
        - message: Human-readable result summary
    """
    import logging

    import numpy as np
    from scipy import stats

    logger = logging.getLogger(__name__)

    try:
        if not hop_values:
            return {
                "status": "error",
                "message": "No HOP values provided",
            }

        hop_array = np.array(hop_values)
        n = len(hop_array)

        if n < 10:
            logger.warning(
                f"Only {n} HOP values provided. Recommend at least 100 for statistical significance."
            )

        # Calculate statistics
        mean_hop = float(np.mean(hop_array))
        std_hop = float(np.std(hop_array, ddof=1))  # Sample std dev
        sem = std_hop / np.sqrt(n)  # Standard error of mean

        # Calculate confidence interval using t-distribution
        t_critical = stats.t.ppf(confidence_level, df=n - 1)
        ci_lower = mean_hop - t_critical * sem
        ci_upper = mean_hop + t_critical * sem

        # QV threshold
        threshold = 2 / 3

        # QV is achieved if the lower bound of confidence interval > threshold
        qv_achieved = bool(ci_lower > threshold)
        margin = float(ci_lower - threshold)

        # Determine result message
        if qv_achieved:
            message = (
                f"QV ACHIEVED! Mean HOP = {mean_hop:.4f}, "
                f"CI lower bound = {ci_lower:.4f} > threshold {threshold:.4f}"
            )
        else:
            message = (
                f"QV NOT achieved. Mean HOP = {mean_hop:.4f}, "
                f"CI lower bound = {ci_lower:.4f} <= threshold {threshold:.4f}"
            )

        return {
            "status": "success",
            "qv_achieved": qv_achieved,
            "mean_hop": mean_hop,
            "std_hop": std_hop,
            "standard_error": sem,
            "confidence_interval": (ci_lower, ci_upper),
            "confidence_level": confidence_level,
            "num_circuits": n,
            "threshold": threshold,
            "margin": margin,
            "message": message,
        }

    except ImportError as e:
        logger.error(f"Missing required package for QV analysis: {e}")
        return {
            "status": "error",
            "message": f"Missing scipy for statistical analysis: {e!s}",
        }
    except Exception as e:
        logger.error(f"Failed to analyze QV results: {e}")
        return {"status": "error", "message": f"Failed to analyze QV results: {e!s}"}


# =============================================================================
# Main Agent Creation
# =============================================================================


async def create_qv_optimizer_agent(
    provider: str = "anthropic",
    model: str | None = None,
    max_qv_depth: int = 5,
    num_circuits: int = 1,
    qv_data: dict[int, dict] | None = None,
) -> tuple[Any, MultiServerMCPClient]:
    """Create the Quantum Volume Optimizer deep agent with all subagents.

    Args:
        provider: LLM provider to use
        model: Optional model name override
        max_qv_depth: Maximum QV depth to evaluate
        num_circuits: Number of independent QV circuits per depth (1=quick test, 100+=full protocol)
        qv_data: Shared cache for QV circuit data, keyed by depth (populated lazily)

    Returns:
        Tuple of (agent, mcp_client) for cleanup
    """
    if qv_data is None:
        qv_data = {}
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
    server_tools = {}

    for server_name in mcp_config.keys():
        try:
            print(f"  Connecting to {server_name}...", end=" ", flush=True)
            tools = await mcp_client.get_tools(server_name=server_name)
            server_tools[server_name] = tools
            print(f"OK ({len(tools)} tools)")
        except Exception as e:
            print(f"FAILED: {e}")

    print(f"\nTotal tools loaded: {sum(len(t) for t in server_tools.values())}")

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

    # Create local tools
    from langchain_core.tools import tool as langchain_tool

    # Find MCP tools for programmatic use (avoids passing large circuits through LLM)
    transpiler_tools = server_tools.get("qiskit-ibm-transpiler", [])
    runtime_tools = server_tools.get("qiskit-ibm-runtime", [])
    hybrid_transpile_mcp = next(
        (t for t in transpiler_tools if t.name == "hybrid_ai_transpile_tool"), None
    )
    run_sampler_mcp = next(
        (t for t in runtime_tools if t.name == "run_sampler_tool"), None
    )
    get_job_results_mcp = next(
        (t for t in runtime_tools if t.name == "get_job_results_tool"), None
    )
    get_job_status_mcp = next(
        (t for t in runtime_tools if t.name == "get_job_status_tool"), None
    )

    def _parse_mcp_result(result: Any) -> dict:
        """Parse the result from a programmatic MCP tool call into a dict."""
        # MCP ainvoke returns: list of content blocks, a JSON string, or a dict
        if isinstance(result, list):
            # Raw MCP content blocks: [{'type': 'text', 'text': '{"status":...}'}]
            text = result[0]["text"] if result else "{}"
            return json.loads(text)
        if isinstance(result, str):
            return json.loads(result)
        return result

    def _ensure_qv_data(depth: int) -> dict | None:
        """Generate QV circuit data for a depth if not already cached. Returns error dict or None."""
        if depth < 2 or depth > 20:
            return {"status": "error", "message": f"Depth must be between 2 and 20, got {depth}"}
        if depth not in qv_data:
            result = generate_qv_circuit_with_ideal_distribution(depth, seed=42 + depth)
            if result["status"] != "success":
                return result
            qv_data[depth] = result
        return None

    @langchain_tool
    async def transpile_qv_circuit(
        depth: int,
        backend_name: str,
        optimization_level: int = 3,
        initial_layout: list[int] | None = None,
    ) -> dict:
        """Transpile a QV circuit using the AI transpiler.

        Generates the QV circuit on demand and transpiles it via the MCP transpiler.
        The transpiled QPY is stored internally — call submit_qv_job next.

        Args:
            depth: The QV depth (e.g., 5 for QV-32, 11 for QV-2048)
            backend_name: IBM backend name (e.g., 'ibm_boston')
            optimization_level: Optimization level 1-3 (default: 3)
            initial_layout: Physical qubits to map virtual qubits to
        """
        error = _ensure_qv_data(depth)
        if error:
            return error

        if hybrid_transpile_mcp is None:
            return {"status": "error", "message": "hybrid_ai_transpile_tool not available"}

        circuit_qasm = qv_data[depth]["circuit_qasm"]

        # Call the MCP transpiler tool programmatically (avoids QASM through LLM)
        result = await hybrid_transpile_mcp.ainvoke({
            "circuit": circuit_qasm,
            "backend_name": backend_name,
            "optimization_level": optimization_level,
            "initial_layout": initial_layout,
        })

        result = _parse_mcp_result(result)

        if result.get("status") == "success":
            # Store QPY for submit_qv_job (avoids QPY through LLM)
            qv_data[depth]["circuit_qpy"] = result["circuit_qpy"]
            return {
                "status": "success",
                "depth": depth,
                "backend_name": backend_name,
                "original_circuit": result.get("original_circuit"),
                "optimized_circuit": result.get("optimized_circuit"),
                "improvements": result.get("improvements"),
                "message": "Transpiled successfully. Call submit_qv_job next.",
            }
        return result

    @langchain_tool
    async def submit_qv_job(
        depth: int,
        backend_name: str,
        shots: int = 4096,
    ) -> dict:
        """Submit a transpiled QV circuit to hardware via the sampler.

        Uses the QPY stored by transpile_qv_circuit — must be called after it.

        Args:
            depth: The QV depth (must have been transpiled first)
            backend_name: IBM backend name
            shots: Number of measurement shots (default: 4096)
        """
        if depth not in qv_data or "circuit_qpy" not in qv_data.get(depth, {}):
            return {
                "status": "error",
                "message": f"No transpiled circuit for depth {depth}. Call transpile_qv_circuit first.",
            }

        if run_sampler_mcp is None:
            return {"status": "error", "message": "run_sampler_tool not available"}

        circuit_qpy = qv_data[depth]["circuit_qpy"]

        # Call the MCP sampler tool programmatically (avoids QPY through LLM)
        result = await run_sampler_mcp.ainvoke({
            "circuit": circuit_qpy,
            "backend_name": backend_name,
            "shots": shots,
        })

        return _parse_mcp_result(result)

    @langchain_tool
    async def calculate_hop(job_id: str, depth: int) -> dict:
        """Calculate Heavy Output Probability (HOP) for QV validation.

        Fetches job results automatically — just pass the job_id and depth.

        Args:
            job_id: The job ID from submit_qv_job (e.g., "d668ng8qbmes739dsh90")
            depth: The QV depth — used to look up heavy outputs automatically

        Returns:
            Dictionary with heavy_output_probability, above_threshold, and message
        """
        error = _ensure_qv_data(depth)
        if error:
            return error

        if get_job_results_mcp is None:
            return {"status": "error", "message": "get_job_results_tool not available"}

        # Fetch counts via MCP (avoids large counts dict through LLM)
        result = await get_job_results_mcp.ainvoke({"job_id": job_id})
        result = _parse_mcp_result(result)
        if result.get("status") != "success":
            return result

        counts = result["counts"]
        heavy_outputs = qv_data[depth]["heavy_outputs"]
        return calculate_heavy_output_probability(counts, heavy_outputs)

    @langchain_tool
    async def run_qv_depth_trial(
        depth: int,
        backend_name: str,
        initial_layout: list[int],
        num_circuits: int = 100,
        shots: int = 4096,
    ) -> dict:
        """Run a full QV trial: N independent circuits at one depth with statistical analysis.

        This is the batch tool for the full QV protocol. It generates N random QV circuits
        (each with a different seed), transpiles, submits, polls, computes HOPs, and runs
        the statistical confidence interval test.

        Args:
            depth: The QV depth to test (e.g., 5 for QV-32)
            backend_name: IBM backend name (e.g., 'ibm_boston')
            initial_layout: Physical qubits to map virtual qubits to
            num_circuits: Number of independent random circuits (default: 100)
            shots: Number of measurement shots per circuit (default: 4096)

        Returns:
            Dictionary with qv_achieved, mean_hop, confidence interval, and per-circuit details
        """
        if hybrid_transpile_mcp is None or run_sampler_mcp is None:
            return {"status": "error", "message": "Required MCP tools not available"}
        if get_job_results_mcp is None or get_job_status_mcp is None:
            return {"status": "error", "message": "Required MCP tools not available"}

        print(f"\n[QV Trial] Starting full QV trial: depth={depth}, {num_circuits} circuits, "
              f"backend={backend_name}, qubits={initial_layout}")

        # Phase 1: Generate N circuits with different seeds
        circuits = []  # List of {circuit_qasm, heavy_outputs, seed}
        for i in range(num_circuits):
            seed = depth * 1000 + i
            result = generate_qv_circuit_with_ideal_distribution(depth, seed=seed)
            if result["status"] != "success":
                return {"status": "error", "message": f"Failed to generate circuit {i}: {result['message']}"}
            circuits.append({
                "circuit_qasm": result["circuit_qasm"],
                "heavy_outputs": result["heavy_outputs"],
                "seed": seed,
            })
        print(f"[QV Trial] Generated {num_circuits} circuits")

        # Phase 2: Transpile each circuit via MCP (sequential — stdio transport)
        for i, circ in enumerate(circuits):
            result = await hybrid_transpile_mcp.ainvoke({
                "circuit": circ["circuit_qasm"],
                "backend_name": backend_name,
                "optimization_level": 3,
                "initial_layout": initial_layout,
            })
            result = _parse_mcp_result(result)
            if result.get("status") != "success":
                return {"status": "error", "message": f"Failed to transpile circuit {i}: {result.get('message', 'unknown error')}"}
            circ["circuit_qpy"] = result["circuit_qpy"]
            if (i + 1) % 10 == 0 or i == num_circuits - 1:
                print(f"[QV Trial] Transpiled {i + 1}/{num_circuits} circuits")

        # Phase 3: Submit each circuit via MCP (sequential)
        job_ids = []
        for i, circ in enumerate(circuits):
            result = await run_sampler_mcp.ainvoke({
                "circuit": circ["circuit_qpy"],
                "backend_name": backend_name,
                "shots": shots,
            })
            result = _parse_mcp_result(result)
            if result.get("status") != "success":
                return {"status": "error", "message": f"Failed to submit circuit {i}: {result.get('message', 'unknown error')}"}
            job_ids.append(result["job_id"])
            if (i + 1) % 10 == 0 or i == num_circuits - 1:
                print(f"[QV Trial] Submitted {i + 1}/{num_circuits} jobs")

        # Phase 4: Poll all jobs until completion
        pending = set(range(num_circuits))
        failed_jobs = {}  # index -> error message
        while pending:
            await asyncio.sleep(10)
            still_pending = set()
            for idx in pending:
                result = await get_job_status_mcp.ainvoke({"job_id": job_ids[idx]})
                result = _parse_mcp_result(result)
                job_status = result.get("job_status", "UNKNOWN")
                if job_status == "DONE":
                    continue  # Done — will collect results in phase 5
                elif job_status in ("ERROR", "CANCELLED"):
                    failed_jobs[idx] = result.get("error_message", job_status)
                else:
                    still_pending.add(idx)
            pending = still_pending
            done_count = num_circuits - len(pending) - len(failed_jobs)
            print(f"[QV Trial] depth={depth}: {done_count}/{num_circuits} jobs done, "
                  f"{len(pending)} pending, {len(failed_jobs)} failed")

        # Phase 5: Get results and compute HOPs
        hop_values = []
        circuit_results = []
        for i in range(num_circuits):
            if i in failed_jobs:
                circuit_results.append({
                    "circuit_index": i, "seed": circuits[i]["seed"],
                    "job_id": job_ids[i], "status": "failed",
                    "error": failed_jobs[i],
                })
                continue

            result = await get_job_results_mcp.ainvoke({"job_id": job_ids[i]})
            result = _parse_mcp_result(result)
            if result.get("status") != "success":
                circuit_results.append({
                    "circuit_index": i, "seed": circuits[i]["seed"],
                    "job_id": job_ids[i], "status": "error",
                    "error": result.get("message", "Failed to get results"),
                })
                continue

            counts = result["counts"]
            hop_result = calculate_heavy_output_probability(counts, circuits[i]["heavy_outputs"])
            hop = hop_result.get("heavy_output_probability", 0.0)
            hop_values.append(hop)
            circuit_results.append({
                "circuit_index": i, "seed": circuits[i]["seed"],
                "job_id": job_ids[i], "status": "success",
                "hop": hop, "above_threshold": hop_result.get("above_threshold", False),
            })

        if not hop_values:
            return {"status": "error", "message": "All circuits failed — no HOP values collected"}

        # Phase 6: Statistical analysis
        analysis = analyze_qv_experiment_results(hop_values)

        print(f"[QV Trial] depth={depth}: mean_hop={analysis.get('mean_hop', 0):.4f}, "
              f"ci_lower={analysis.get('confidence_interval', (0, 0))[0]:.4f}, "
              f"qv_achieved={analysis.get('qv_achieved', False)}")

        return {
            "status": "success",
            "depth": depth,
            "num_circuits": num_circuits,
            "num_successful": len(hop_values),
            "num_failed": len(failed_jobs),
            "qv_achieved": analysis.get("qv_achieved", False),
            "mean_hop": analysis.get("mean_hop", 0.0),
            "std_hop": analysis.get("std_hop", 0.0),
            "ci_lower": analysis.get("confidence_interval", (0, 0))[0],
            "ci_upper": analysis.get("confidence_interval", (0, 0))[1],
            "confidence_level": analysis.get("confidence_level", 0.975),
            "threshold": 2 / 3,
            "individual_hops": hop_values,
            "message": analysis.get("message", ""),
        }

    # qv-experiment-runner: local tools + MCP tools for job monitoring
    runner_mcp_tools = [
        t for t in runtime_tools
        if t.name == "get_job_status_tool"
    ]
    qv_experiment_runner = {
        "name": "qv-experiment-runner",
        "description": "Expert in running QV experiments on hardware. Use this agent to transpile circuits, submit jobs, and retrieve results.",
        "system_prompt": QV_EXPERIMENT_RUNNER_PROMPT,
        "tools": runner_mcp_tools + [transpile_qv_circuit, submit_qv_job],
    }

    # Coordinator: runtime tools (minus execution/results) + calculate_hop
    # get_job_results_tool excluded because calculate_hop fetches results internally
    excluded_tools = {"run_sampler_tool", "run_estimator_tool", "get_job_results_tool"}
    coordinator_tools = [
        tool
        for tool in server_tools.get("qiskit-ibm-runtime", [])
        if tool.name not in excluded_tools
    ]
    coordinator_tools.append(calculate_hop)

    # Multi-circuit mode: add batch tool and extended prompt
    system_prompt = COORDINATOR_SYSTEM_PROMPT
    if num_circuits > 1:
        coordinator_tools.append(run_qv_depth_trial)
        system_prompt = COORDINATOR_SYSTEM_PROMPT + COORDINATOR_MULTI_CIRCUIT_APPENDIX

    print("\nCreating Quantum Volume Optimizer agent...")
    print(f"  Mode: {'multi-circuit (' + str(num_circuits) + ' circuits/depth)' if num_circuits > 1 else 'single-circuit'}")
    print(f"  Coordinator tools: {len(coordinator_tools)} (excluded: {excluded_tools})")
    print(f"  qv-experiment-runner tools: {len(runner_mcp_tools)} MCP + transpile_qv_circuit + submit_qv_job")

    agent = create_deep_agent(
        model=llm,
        tools=coordinator_tools,
        system_prompt=system_prompt,
        subagents=[backend_analyst, qubit_chain_optimizer, qv_experiment_runner],
    )

    print("Agent ready!\n")

    return agent, mcp_client


async def run_qv_optimization(
    provider: str = "anthropic",
    model: str | None = None,
    max_qv_depth: int = 5,
    num_circuits: int = 1,
    backend: str | None = None,
    verbose: bool = True,
    run_experiment: bool = True,
) -> str:
    """Run iterative Quantum Volume finding workflow.

    Args:
        provider: LLM provider to use
        model: Optional model name override
        max_qv_depth: Maximum QV depth to try (2-20)
        num_circuits: Number of independent QV circuits per depth (1=quick test, 100+=full protocol)
        backend: Specific backend to test (required for experiments)
        verbose: Show detailed activity logging (tool calls, LLM activity)
        run_experiment: Actually run QV circuits on hardware (default: True)

    Returns:
        The final QV experiment report with actual results
    """
    # QV circuits are generated lazily by transpile_qv_circuit when the agent requests them
    qv_data: dict[int, dict] = {}

    agent, mcp_client = await create_qv_optimizer_agent(
        provider, model, max_qv_depth, num_circuits, qv_data
    )

    # Build the request based on mode
    if run_experiment:
        if not backend:
            backend_section = """
## Step 1: Select Backend
Use backend-analyst to find the least busy backend with good calibration.
Pick ONE backend to run experiments on."""
        else:
            backend_section = f"""
## Step 1: Backend
Use backend: **{backend}**
Get its properties to confirm it's available."""

        if num_circuits > 1:
            # Multi-circuit mode: full QV protocol
            request = f"""
# FIND THE HIGHEST ACHIEVABLE QUANTUM VOLUME (Full Protocol — {num_circuits} circuits/depth)

Your task: Find the highest QV this backend can achieve using the full QV protocol
with {num_circuits} independent random circuits per depth and statistical CI testing.

{backend_section}

## Step 2: Find Optimal Qubits
Use qubit-chain-optimizer with find_optimal_qv_qubits_tool(num_qubits=N) for the depth you're testing.
Get 10 candidate qubit subsets. The tool searches ALL qubits on the backend.

## Step 3: Run Iterative QV Experiments (TOP-DOWN)

Supported depths: 2 through {max_qv_depth} (QV {2**2} through QV {2**max_qv_depth}).

Start from depth {max_qv_depth} and work DOWN:

### For each depth:
1. Find optimal qubits: task(subagent_type="qubit-chain-optimizer", description="Find N optimal qubits for QV-N on <backend>")
2. Run full QV trial:
   run_qv_depth_trial(depth=N, backend_name=<backend>, initial_layout=[q1, q2, ...], num_circuits={num_circuits}, shots=4096)
   This tool runs all {num_circuits} circuits programmatically and returns the statistical result.
3. Check qv_achieved in the result:
   - If qv_achieved is true → SUCCESS! QV 2^N is statistically achieved
   - If qv_achieved is false → try depth N-1

### IMPORTANT:
- Use run_qv_depth_trial (NOT qv-experiment-runner) for each depth
- The tool handles transpile, submit, poll, HOP, and CI test internally
- You MUST try lower depths if higher ones fail
- STOP when you find a passing depth or reach depth 2

## Expected Output

```
## QV EXPERIMENT RESULTS (Full Protocol)

### Depth N (First Attempt)
- Backend: <name>
- Qubits: [<from qubit optimizer>]
- Circuits: {num_circuits}
- Mean HOP: <value>
- CI Lower Bound: <value>
- QV Achieved: YES/NO
- Result: PASS/FAIL (statistical test)

### Depth N-1 (if needed)
...

## CONCLUSION
Highest Achieved QV: 2^M = <value>
Protocol: Full ({num_circuits} circuits, 97.5% CI)
```
"""
        else:
            # Single-circuit mode: quick test
            request = f"""
# FIND THE HIGHEST ACHIEVABLE QUANTUM VOLUME

Your task: Find the highest QV this backend can achieve by running experiments.

{backend_section}

## Step 2: Find Optimal Qubits
Use qubit-chain-optimizer with find_optimal_qv_qubits_tool(num_qubits=N) for the depth you're testing.
Get 10 candidate qubit subsets. The tool searches ALL qubits on the backend.

## Step 3: Run Iterative QV Experiments (TOP-DOWN)

Supported depths: 2 through {max_qv_depth} (QV {2**2} through QV {2**max_qv_depth}).

Start from depth {max_qv_depth} and work DOWN:

### For each depth:
1. Find optimal qubits: task(subagent_type="qubit-chain-optimizer", description="Find N optimal qubits for QV-N on <backend>")
2. Run experiment: task(subagent_type="qv-experiment-runner", description="Run QV experiment: depth=N, backend_name=<backend>, initial_layout=[q1, q2, ...]")
   The runner handles: transpile → submit → poll. Returns job_id.
3. Calculate HOP: calculate_hop(job_id=<job_id from runner>, depth=N)
4. If above_threshold is true → SUCCESS! QV 2^N achieved
5. If above_threshold is false → try depth N-1

### IMPORTANT:
- You MUST call qv-experiment-runner for EACH depth you test
- You MUST call calculate_hop to evaluate results
- You MUST try lower depths if higher ones fail
- STOP when you find a passing depth or reach depth 2

## Expected Output

```
## QV EXPERIMENT RESULTS

### Depth N (First Attempt)
- Backend: <name>
- Qubits: [<from qubit optimizer>]
- Job ID: <id>
- Shots: 4096
- HOP: <value from calculate_hop>
- Result: PASS/FAIL

### Depth N-1 (if needed)
...

## CONCLUSION
Highest Achieved QV: 2^M = <value>
```
"""
    else:
        # Analysis only mode (no experiments)
        request = f"""
# QUANTUM VOLUME ANALYSIS (No Experiments)

Analyze this backend for QV potential without running hardware experiments.

## Target Backend: {backend or "Find least busy"}

## Tasks

1. **Backend Analysis**: Get backend properties and calibration data

2. **Find Optimal Qubits**: Use find_optimal_qv_qubits_tool to find:
   - Best qubit subsets for depths 2 through {max_qv_depth}
   - Request num_results=10 to get multiple candidates
   - The tool searches ALL qubits on the backend

3. **Report**: List the top 10 qubit configurations for each depth with scores

Note: This is analysis only. Run without the --no-experiment flag to execute actual experiments on hardware.
"""

    print("=" * 70)
    print("  STARTING QUANTUM VOLUME FINDER")
    print("=" * 70)
    print(f"\nBackend: {backend or 'Auto-select least busy'}")
    print(f"Max QV depth to try: {max_qv_depth} (QV 2^{max_qv_depth} = {2**max_qv_depth})")
    if num_circuits > 1:
        print(f"Protocol: Full QV ({num_circuits} circuits/depth, 97.5% CI test)")
    else:
        print("Protocol: Single-circuit quick test")
    print(f"Mode: {'EXPERIMENT (will run on hardware)' if run_experiment else 'ANALYSIS ONLY'}")
    print("\nThis may take several minutes...")
    print("-" * 70)

    # Create callback handler for observability
    callback_handler = AgentActivityHandler(verbose=verbose)

    # Run the agent with callback
    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": request}]},
        config={"callbacks": [callback_handler]},
    )

    # Extract final response
    messages = result.get("messages", [])
    if messages:
        final_response = messages[-1].content
    else:
        final_response = "No response generated."

    print("\n" + "=" * 70)
    print("  QV FINDING COMPLETE")
    print("=" * 70)

    return final_response


async def interactive_mode(
    provider: str,
    model: str | None,
    backend: str | None = None,
    verbose: bool = True,
) -> None:
    """Run interactive mode where users can ask follow-up questions."""
    from langchain_core.messages import HumanMessage

    agent, mcp_client = await create_qv_optimizer_agent(provider, model, 20)

    # Create callback handler for observability
    callback_handler = AgentActivityHandler(verbose=verbose)

    print("\n" + "-" * 70)
    print("Interactive Mode - Ask questions about Quantum Volume optimization")
    if backend:
        print(f"Target backend: {backend}")
    print("Commands: 'quit', 'clear', 'optimize', 'verbose' (toggle activity log)")
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

            if query.lower() == "verbose":
                callback_handler.verbose = not callback_handler.verbose
                status = "ON" if callback_handler.verbose else "OFF"
                print(f"Verbose activity logging is now {status}\n")
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

            print("\nProcessing...\n")

            result = await agent.ainvoke(
                {"messages": messages},
                config={"callbacks": [callback_handler]},
            )

            result_messages = result.get("messages", [])
            if result_messages:
                response = result_messages[-1].content
                # Update history with full conversation from agent
                history = result_messages
                print(f"\nAssistant:\n{response}\n")

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Quantum Volume Finder - Find highest achievable QV through experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Find highest QV for ibm_brisbane (single-circuit quick test)
  python quantum_volume_optimizer.py --backend ibm_brisbane --depth 5

  # Full QV protocol with 100 circuits per depth (statistically rigorous)
  python quantum_volume_optimizer.py --backend ibm_brisbane --depth 5 --num-circuits 100

  # Try higher QV depth
  python quantum_volume_optimizer.py --backend ibm_brisbane --depth 8

  # Analysis only (no hardware execution)
  python quantum_volume_optimizer.py --backend ibm_brisbane --depth 5 --no-experiment

  # Interactive mode for follow-up experiments
  python quantum_volume_optimizer.py --interactive --backend ibm_brisbane
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
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable verbose activity logging (hide tool calls and LLM activity)",
    )
    parser.add_argument(
        "--no-experiment",
        action="store_true",
        help="Skip running QV circuits on hardware (default: runs experiment)",
    )
    parser.add_argument(
        "--num-circuits",
        type=int,
        default=1,
        metavar="N",
        help="Number of independent QV circuits per depth (1=quick test, 100+=full protocol)",
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
    verbose = not args.quiet
    if args.interactive:
        asyncio.run(interactive_mode(args.provider, args.model, args.backend, verbose))
    else:
        result = asyncio.run(
            run_qv_optimization(
                args.provider, args.model, args.depth, args.num_circuits,
                args.backend, verbose, not args.no_experiment,
            )
        )
        print(result)


if __name__ == "__main__":
    main()
