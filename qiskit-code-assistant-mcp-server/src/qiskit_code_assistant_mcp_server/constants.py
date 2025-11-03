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

import importlib.metadata
import logging
import os

from dotenv import load_dotenv


# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


##################################################
## CONSTANTS
##################################################

try:
    QCA_TOOL_X_CALLER = f"qiskit-code-assistant-mcp-server/{importlib.metadata.version('qiskit-code-assistant-mcp-server')}"
except Exception:
    QCA_TOOL_X_CALLER = "qiskit-code-assistant-mcp-server/unknown"
    logger.warning("Could not determine package version for X-Caller header")

QCA_REQUEST_TIMEOUT = float(os.getenv("QCA_REQUEST_TIMEOUT", "30.0"))

# Validate timeout
if QCA_REQUEST_TIMEOUT <= 0 or QCA_REQUEST_TIMEOUT > 300:
    logger.warning(f"Invalid request timeout {QCA_REQUEST_TIMEOUT}, using default 30.0")
    QCA_REQUEST_TIMEOUT = 30.0


##################################################
## ENVIRONMENT VARIABLES
##################################################

QCA_TOOL_API_BASE = os.getenv(
    "QCA_TOOL_API_BASE", "https://qiskit-code-assistant.quantum.ibm.com"
).rstrip("/")  # Remove trailing slash for consistent URLs

QCA_TOOL_MODEL_NAME = os.getenv("QCA_TOOL_MODEL_NAME", "mistral-small-3.2-24b-qiskit")

# Validate log level
QCA_MCP_DEBUG_LEVEL = os.getenv("QCA_MCP_DEBUG_LEVEL", "INFO").upper()
if QCA_MCP_DEBUG_LEVEL not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
    logger.warning(f"Invalid debug level {QCA_MCP_DEBUG_LEVEL}, using INFO")
    QCA_MCP_DEBUG_LEVEL = "INFO"


##################################################
## CONFIGURATION VALIDATION
##################################################


def validate_configuration() -> bool:
    """Validate the current configuration and log any issues."""
    valid = True

    if not QCA_TOOL_API_BASE.startswith(("http://", "https://")):
        logger.error(f"Invalid API base URL: {QCA_TOOL_API_BASE}")
        valid = False

    if not QCA_TOOL_MODEL_NAME:
        logger.error("Model name cannot be empty")
        valid = False

    return valid
