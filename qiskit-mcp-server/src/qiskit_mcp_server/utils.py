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

"""Utility functions for the Qiskit MCP server.

This module provides the `with_sync` decorator for creating dual async/sync APIs.

Synchronous Execution
---------------------
All async functions decorated with `@with_sync` can be called synchronously
via the `.sync` attribute:

    from qiskit_mcp_server.transpiler import transpile_circuit

    # Async usage (in async context)
    result = await transpile_circuit(qasm_str, optimization_level=2)

    # Sync usage (in sync context, Jupyter notebooks, DSPy, etc.)
    result = transpile_circuit.sync(qasm_str, optimization_level=2)

The sync wrapper handles event loop management automatically, including
nested event loops in Jupyter notebooks (via nest_asyncio).
"""

import asyncio
from collections.abc import Callable, Coroutine
from functools import wraps
from typing import Any, TypeVar


# Apply nest_asyncio to allow running async code in environments with existing event loops
try:
    import nest_asyncio

    nest_asyncio.apply()
except ImportError:
    pass


F = TypeVar("F", bound=Callable[..., Any])
T = TypeVar("T")


def _run_async(coro: Coroutine[Any, Any, T]) -> T:
    """Helper to run async functions synchronously.

    This handles both cases:
    - Running in a Jupyter notebook or other environment with an existing event loop
    - Running in a standard Python script without an event loop
    """
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # We're in a running loop (e.g., Jupyter), use run_until_complete
            # This works because nest_asyncio allows nested loops
            return loop.run_until_complete(coro)
        else:
            return loop.run_until_complete(coro)
    except RuntimeError:
        # No event loop exists, create one
        return asyncio.run(coro)


def with_sync(func: F) -> F:
    """Decorator that adds a `.sync` attribute to async functions for synchronous execution.

    Usage:
        @with_sync
        async def my_async_function(arg: str) -> Dict[str, Any]:
            ...

        # Async call
        result = await my_async_function("hello")

        # Sync call
        result = my_async_function.sync("hello")
    """

    @wraps(func)
    def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
        return _run_async(func(*args, **kwargs))

    func.sync = sync_wrapper  # type: ignore[attr-defined]
    return func
