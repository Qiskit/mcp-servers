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
│                 │ │                     │ │ Poll & get counts   │
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

    # Find highest QV for a backend (default: runs experiments)
    python quantum_volume_optimizer.py --backend ibm_brisbane --depth 5

    # Analysis only (no hardware execution)
    python quantum_volume_optimizer.py --backend ibm_brisbane --depth 5 --no-experiment

    # Interactive mode
    python quantum_volume_optimizer.py --backend ibm_brisbane --interactive

## Output

The agent reports ACTUAL results:
- Each depth attempted
- Qubits used (from find_optimal_qv_qubits_tool)
- Job ID, measurement counts
- Calculated HOP
- PASS/FAIL for each depth
- Final achieved QV
"""

from __future__ import annotations

import argparse
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


# Load environment variables
load_dotenv()


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

COORDINATOR_SYSTEM_PROMPT = """You are the Quantum Volume Finder, an expert system that determines
the highest achievable Quantum Volume (QV) for IBM Quantum backends through actual execution.

## Your Mission

Find the ACTUAL achievable Quantum Volume by running experiments, NOT by making recommendations.
You must execute QV circuits on hardware and report real results.

## Quantum Volume Protocol

QV 2^n is ACHIEVED if Heavy Output Probability (HOP) > 2/3.
- HOP = (shots with heavy outputs) / (total shots)
- Heavy outputs are pre-computed from ideal simulation

## Strategy: Top-Down Search

Start from the HIGHEST requested depth and work DOWN:
1. Try depth N (the maximum requested)
2. Run QV circuit on hardware, get measurement counts
3. Calculate HOP from the counts
4. If HOP > 2/3: SUCCESS! QV 2^N is achieved. Stop here.
5. If HOP <= 2/3: FAILED. Try depth N-1.
6. Repeat until you find the highest depth that passes, or reach depth 2.

## CRITICAL: How to Call Your Subagents

You have access to a `task` tool to delegate work to specialized subagents.

**EXACT SYNTAX** - You MUST use this exact format:
```json
{
  "name": "<one of the subagent names below>",
  "description": "<detailed task description with ALL required data>"
}
```

**AVAILABLE SUBAGENTS** (use these exact names):

### 1. "backend-analyst"
Gets backend information. Example call:
```json
{
  "name": "backend-analyst",
  "description": "Get properties and status for ibm_boston backend"
}
```

### 2. "qubit-chain-optimizer"
Finds optimal qubit subsets. Example call:
```json
{
  "name": "qubit-chain-optimizer",
  "description": "Find 10 optimal qubit subsets for QV depth 5 on ibm_boston using find_optimal_qv_qubits_tool"
}
```

### 3. "qv-experiment-runner"
Runs QV experiments - MUST include the FULL circuit QASM in the description!
```json
{
  "name": "qv-experiment-runner",
  "description": "Run QV experiment on ibm_boston. CIRCUIT QASM: OPENQASM 3.0; include 'stdgates.inc'; qubit[5] q; ... [full QASM here] ... QUBITS: [47, 57, 66, 67, 68]. DEPTH: 5. Execute these steps: 1) transpile with hybrid_ai_transpile_tool, 2) submit with run_sampler_tool using the circuit_qpy from step 1, 3) poll get_job_status_tool until DONE, 4) get counts with get_job_results_tool, 5) return the measurement counts."
}
```

**COMMON MISTAKES TO AVOID:**
- Do NOT use `subagent_type` - the parameter is called `name`
- Do NOT use generic names like "general-purpose" - use the exact names above
- Do NOT forget the `description` field - it is REQUIRED
- Do NOT call `task` with empty or missing parameters

## Output Format

Your final report MUST include actual execution results:

```
## QV EXPERIMENT RESULTS

### Depth N (tried first)
- Qubits used: [list of qubits]
- Job ID: xxx
- Shots: 4096
- Measurement counts: {"00...": count, ...}
- HOP: calculated value
- Result: PASS/FAIL

### Depth N-1 (if N failed)
...

## CONCLUSION
Highest achieved QV: 2^M (where M is the highest passing depth)
```

## Critical Rules

1. DO NOT make recommendations - RUN the experiments
2. DO NOT stop after one failure - try lower depths
3. DO NOT limit qubit search - use all qubits on the backend
4. DO NOT use write_file - return results as text
5. ALWAYS report actual measurement counts from hardware
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

QV_EXPERIMENT_RUNNER_PROMPT = """You are the QV Experiment Runner. You transpile and execute QV circuits.

## CRITICAL: Read This Carefully

You will receive the circuit (QASM string), backend_name, and qubits from the coordinator.
You MUST use these exact values - do not search for files or make up circuits.

## Your Workflow - Execute ALL Steps In Order

### STEP 1: TRANSPILE THE CIRCUIT

Call `hybrid_ai_transpile_tool` with the QASM string from the task description:

**Tool call:**
```json
{
  "circuit": "OPENQASM 3.0; include 'stdgates.inc'; qubit[5] q; ...",
  "backend_name": "ibm_boston",
  "optimization_level": 3,
  "ai_layout_mode": "optimize"
}
```

**Response will contain:**
```json
{
  "status": "success",
  "circuit_qpy": "UUlTS0lUEA...very long base64 string...",
  "num_qubits": 5,
  "depth": 42
}
```

**IMPORTANT:** Copy the ENTIRE `circuit_qpy` value. It is a long base64 string. You need it for step 2.

### STEP 2: SUBMIT TO HARDWARE

Call `run_sampler_tool` with the `circuit_qpy` from step 1:

**Tool call:**
```json
{
  "circuit": "UUlTS0lUEA...paste the entire circuit_qpy value here...",
  "backend_name": "ibm_boston",
  "shots": 4096
}
```

**CRITICAL:** The `circuit` parameter MUST contain the `circuit_qpy` value from step 1's response.
Do NOT leave it empty. Do NOT use a placeholder. Paste the actual base64 string.

**Response will contain:**
```json
{
  "job_id": "d5jm8tivcahs73a0uf70",
  "status": "QUEUED"
}
```

### STEP 3: WAIT FOR COMPLETION

Poll `get_job_status_tool` until the job is DONE:

**Tool call:**
```json
{
  "job_id": "d5jm8tivcahs73a0uf70"
}
```

Keep calling until `job_status` is "DONE". May take 1-10 minutes.

### STEP 4: GET MEASUREMENT RESULTS

Call `get_job_results_tool`:

**Tool call:**
```json
{
  "job_id": "d5jm8tivcahs73a0uf70"
}
```

**Response will contain:**
```json
{
  "counts": {"00000": 892, "00001": 145, "00010": 234, ...}
}
```

### STEP 5: REPORT BACK

Return a summary with ALL this information:
```
EXPERIMENT RESULT:
- Backend: ibm_boston
- Depth: 5
- Qubits: [47, 57, 66, 67, 68]
- Job ID: d5jm8tivcahs73a0uf70
- Shots: 4096
- Counts: {"00000": 892, "00001": 145, ...}
```

## CRITICAL RULES - READ CAREFULLY

1. **Step 2 circuit parameter is REQUIRED**: You MUST pass the `circuit_qpy` from step 1.
   - If you call `run_sampler_tool` with an empty circuit, it will FAIL.
   - The circuit_qpy is a long base64 string starting with "UUFT..." - use the entire string.

2. **Complete ALL steps**: Do not stop after step 1 or step 2. Run all 5 steps.

3. **Return actual counts**: The coordinator needs the measurement counts to calculate HOP.
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
        num_qubits: Number of qubits for the QV circuit (2-10 recommended)
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
        For num_qubits > 10, simulation becomes computationally expensive.
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
        elif num_qubits > 10:
            # Warn but allow - simulation will be slow
            logger.warning(
                f"QV with {num_qubits} qubits will be slow to simulate. "
                "Consider using <= 10 qubits."
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
            # Format as bitstring (little-endian to match Qiskit convention)
            bitstring = format(i, f"0{num_qubits}b")[::-1]
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

    After running multiple QV circuits on hardware and calculating their individual
    Heavy Output Probabilities (HOP), this function performs statistical analysis
    to determine if the QV benchmark is achieved.

    QV Success Criteria:
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

    qv_experiment_runner = {
        "name": "qv-experiment-runner",
        "description": "Expert in running QV experiments on hardware. Use this agent to transpile circuits, submit jobs, retrieve results, calculate HOP, and validate QV achievement.",
        "system_prompt": QV_EXPERIMENT_RUNNER_PROMPT,
        "tools": (
            server_tools.get("qiskit-ibm-runtime", [])
            + server_tools.get("qiskit-ibm-transpiler", [])  # For ai_routing_tool
        ),
    }

    # Create the coordinator agent
    # IMPORTANT: Coordinator only gets runtime tools (not transpilation tools)
    # This forces it to delegate transpilation to qv-experiment-runner subagent
    # instead of trying to call hybrid_ai_transpile_tool directly without arguments
    coordinator_tools = server_tools.get("qiskit-ibm-runtime", [])
    print("\nCreating Quantum Volume Optimizer agent...")
    print(f"  Coordinator tools: {len(coordinator_tools)} (runtime only)")
    print("  qv-experiment-runner has transpilation tools")

    agent = create_deep_agent(
        model=llm,
        tools=coordinator_tools,
        system_prompt=COORDINATOR_SYSTEM_PROMPT,
        subagents=[backend_analyst, qubit_chain_optimizer, qv_experiment_runner],
    )

    print("Agent ready!\n")

    return agent, mcp_client


async def run_qv_optimization(
    provider: str = "anthropic",
    model: str | None = None,
    max_qv_depth: int = 5,
    backend: str | None = None,
    verbose: bool = True,
    run_experiment: bool = True,
) -> str:
    """Run iterative Quantum Volume finding workflow.

    Args:
        provider: LLM provider to use
        model: Optional model name override
        max_qv_depth: Maximum QV depth to try (2-20)
        backend: Specific backend to test (required for experiments)
        verbose: Show detailed activity logging (tool calls, LLM activity)
        run_experiment: Actually run QV circuits on hardware (default: True)

    Returns:
        The final QV experiment report with actual results
    """
    agent, mcp_client = await create_qv_optimizer_agent(provider, model, max_qv_depth)

    # Generate QV circuits with heavy outputs for each depth we'll try
    qv_data = {}
    print("\nGenerating QV circuits with heavy output computation...")
    for depth in range(max_qv_depth, 1, -1):  # From max down to 2
        print(f"  Generating QV-{depth} circuit...", end=" ", flush=True)
        result = generate_qv_circuit_with_ideal_distribution(depth, seed=42 + depth)
        if result["status"] == "success":
            qv_data[depth] = result
            print(f"OK ({result['num_heavy_outputs']} heavy outputs)")
        else:
            print(f"FAILED: {result.get('message', 'Unknown error')}")

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

        # Build QV circuit info for each depth
        qv_circuit_sections = []
        for depth in range(max_qv_depth, 1, -1):
            if depth in qv_data:
                data = qv_data[depth]
                heavy_list = data["heavy_outputs"][:20]  # Show first 20 heavy outputs
                heavy_str = ", ".join(f'"{h}"' for h in heavy_list)
                if len(data["heavy_outputs"]) > 20:
                    heavy_str += f", ... ({len(data['heavy_outputs'])} total)"
                qv_circuit_sections.append(f"""
### QV-{depth} Circuit (for QV 2^{depth} = {2**depth})
```qasm
{data["circuit_qasm"]}
```
**Heavy outputs** (for HOP calculation): [{heavy_str}]
**Num heavy outputs**: {data["num_heavy_outputs"]} out of {2**depth}
""")

        qv_circuits_text = "\n".join(qv_circuit_sections)

        request = f"""
# FIND THE HIGHEST ACHIEVABLE QUANTUM VOLUME

Your task: Find the highest QV this backend can achieve by running experiments.

{backend_section}

## Step 2: Find Optimal Qubits
Use qubit-chain-optimizer with find_optimal_qv_qubits_tool:
- Get 10 candidate qubit subsets for depth {max_qv_depth}
- The tool searches ALL qubits on the backend (not just first 10)
- Save the top candidates for experiments

## Step 3: Run Iterative QV Experiments (TOP-DOWN)

Start from depth {max_qv_depth} and work DOWN until you find a passing depth:

### For each depth (starting at {max_qv_depth}):
1. Transpile circuit using hybrid_ai_transpile_tool:
   hybrid_ai_transpile_tool(circuit=<the QASM from this depth>, backend_name="{backend}", optimization_level=3, ai_layout_mode="optimize")
2. Submit the transpiled circuit: run_sampler_tool with 4096 shots
3. Wait for job completion (poll get_job_status_tool until DONE)
4. Get results (get_job_results_tool)
5. Calculate HOP:
   - Count how many shots resulted in heavy outputs
   - HOP = heavy_count / total_shots
6. If HOP > 0.667: SUCCESS! Report this depth as achieved QV
7. If HOP <= 0.667: FAILED. Try depth-1 with its circuit and heavy outputs

### IMPORTANT:
- You MUST report actual measurement counts from hardware
- You MUST calculate HOP for each depth tried
- You MUST try lower depths if higher ones fail
- STOP when you find a passing depth or reach depth 2

## QV Circuits and Heavy Outputs

{qv_circuits_text}

## Expected Output Format

```
## QV EXPERIMENT RESULTS

### Depth {max_qv_depth} (First Attempt)
- Backend: <name>
- Qubits: [<list from find_optimal_qv_qubits_tool>]
- Job ID: <id>
- Shots: 4096
- Counts: {{<actual measurement counts>}}
- Heavy output count: <number of shots in heavy outputs>
- HOP: <calculated value>
- Result: PASS/FAIL (HOP > 0.667?)

### Depth {max_qv_depth - 1} (if needed)
...

## CONCLUSION
Highest Achieved QV: 2^N = <value>
Backend: <name>
Optimal Qubits: [<list>]
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

Note: This is analysis only. Use --no-experiment flag was not set to run actual experiments.
"""

    print("=" * 70)
    print("  STARTING QUANTUM VOLUME FINDER")
    print("=" * 70)
    print(f"\nBackend: {backend or 'Auto-select least busy'}")
    print(f"Max QV depth to try: {max_qv_depth} (QV 2^{max_qv_depth} = {2**max_qv_depth})")
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
  # Find highest QV for ibm_brisbane (runs experiments by default)
  python quantum_volume_optimizer.py --backend ibm_brisbane --depth 5

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
                args.provider, args.model, args.depth, args.backend, verbose, not args.no_experiment
            )
        )
        print(result)


if __name__ == "__main__":
    main()
