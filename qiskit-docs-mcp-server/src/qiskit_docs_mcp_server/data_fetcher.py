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

import httpx
import logging
from functools import lru_cache
from typing import Optional

logger = logging.getLogger(__name__)

# Qiskit documentation bases
QISKIT_DOCS_BASE = "https://docs.quantum.ibm.com/"
QISKIT_SDK_DOCS = "https://docs.quantum.ibm.com/"
QISKIT_RUNTIME_DOCS = "https://docs.quantum.ibm.com/run/"
BASE_URL = "https://quantum.cloud.ibm.com/"

# Qiskit modules and their documentation paths
QISKIT_MODULES = {
    "circuit": "api/qiskit/circuit",
    "primitives": "api/qiskit/primitives",
    "transpiler": "api/qiskit/transpiler",
    "quantum_info": "api/qiskit/quantum_info",
    "result": "api/qiskit/result",
    "visualization": "api/qiskit/visualization",
}

QISKIT_ADDON_MODULES = {
    "addon-opt-mapper": "guides/qaoa-mapper",
    "addon-qpe": "guides/qpe",
    "addon-vqe": "guides/vqe",
}

SEARCH_PATH = "endpoints-docs-learning/api/search"

@lru_cache(maxsize=100)
def fetch_text(url: str) -> Optional[str]: 
    """
    Fetch text content from a URL using httpx.
    
    Args:
        url: The URL to fetch
        
    Returns:
        The text content of the page, or None if fetch fails
    """
    try:
        with httpx.Client(timeout=10.0) as client:
            response = client.get(url, follow_redirects=True)
            response.raise_for_status()
            return response.text
    except httpx.HTTPError as e:
        logger.error(f"Failed to fetch {url}: {e} because of a HTTP error.")
        return None
    except Exception as e:
        logger.error(f"Unexpected error fetching {url}: {e}")
        return None

def get_component_docs(component: str) -> Optional[str]:
    """
    Fetch documentation for a Qiskit SDK module.
    
    Args:
        component: Module name (e.g., 'circuit', 'primitives', 'transpiler')
        
    Returns:
        The documentation content or None if not found
    """
    if component not in QISKIT_MODULES:
        return None
    
    path = QISKIT_MODULES[component]
    url = f"{QISKIT_SDK_DOCS}{path}"
    logger.info(f"Fetching component docs for {component} from {url}")
    return fetch_text(url)

def get_guide_docs(style: str) -> Optional[str]:
    """
    Fetch documentation for a Qiskit guide or best practice.
    
    Args:
        style: Guide name (e.g., 'optimization', 'error-mitigation')
        
    Returns:
        The documentation content or None if not found
    """
    guide_paths = {
        "optimization": "guides/optimization",
        "quantum-circuits": "guides/circuits",
        "error-mitigation": "guides/error-mitigation",
        "dynamic-circuits": "guides/dynamic-circuits",
        "parametric-compilation": "guides/parametric-compilation",
        "performance-tuning": "guides/performance-tuning",
    }
    
    if style not in guide_paths:
        return None
    
    path = guide_paths[style]
    url = f"{QISKIT_DOCS_BASE}{path}"
    logger.info(f"Fetching style docs for {style} from {url}")
    return fetch_text(url)


def search_qiskit_docs(query: str, module: str = "documentation") -> list[dict]:
    """
    Search Qiskit documentation for relevant results.
    
    Args:
        query: Search query string
        module: Search module string

    Returns:
        List of relevant documentation entries with name and description
    """

    url = f"{BASE_URL}{SEARCH_PATH}?query={query}&module={module}"
    logger.info(f"Querying from {query} which gives {url} from {module}")
    
    return fetch_text_json(url)


def fetch_text_json(url: str) -> list[dict]: 
    """
    Fetch text content from a URL using httpx.
    
    Args:
        url: The URL to fetch
        
    Returns:
        The text content of the page, or None if fetch fails
    """
    try:
        with httpx.Client(timeout=10.0) as client:
            response = client.get(url, follow_redirects=True)
            response.raise_for_status()
            return response.json()
    except httpx.HTTPError as e:
        logger.error(f"Failed to fetch {url}: {e} because of a HTTP error.")
        return None
    except Exception as e:
        logger.error(f"Unexpected error fetching {url}: {e}")
        return None
