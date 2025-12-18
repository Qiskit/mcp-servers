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
import logging
import threading
from typing import Any, Optional

from qiskit_ibm_runtime import QiskitRuntimeService


# Placeholder tokens that should be rejected during validation
INVALID_PLACEHOLDER_TOKENS = frozenset(["<PASSWORD>", "<TOKEN>", "YOUR_TOKEN_HERE", "xxx"])


logger = logging.getLogger(__name__)


class QiskitRuntimeServiceProvider:
    """
    Singleton thread-safe provider with lazy initialization for QiskitRuntimeService
    """

    _instance: Optional["QiskitRuntimeServiceProvider"] = None
    _lock = threading.Lock()

    def __new__(
        cls: type["QiskitRuntimeServiceProvider"], *args: Any, **kwargs: Any
    ) -> "QiskitRuntimeServiceProvider":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:  # double check to ensure multiple threads not enter in
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if not hasattr(self, "_service"):
            self._service: QiskitRuntimeService | None = None
            self._service_lock = threading.Lock()  # lock for lazy init
            self._cached_token: str | None = None
            self._cached_channel: str = "ibm_quantum_platform"

    def get(
        self, token: str | None = None, channel: str = "ibm_quantum_platform"
    ) -> QiskitRuntimeService:
        """Lazy initialization for QiskitRuntimeService"""
        with self._service_lock:
            if (
                self._service is None
                or token != self._cached_token
                or channel != self._cached_channel
            ):
                self._service = self._initialize_service(token=token, channel=channel)
                self._cached_token = token
                self._cached_channel = channel
        return self._service

    @staticmethod
    def _initialize_service(
        token: str | None = None, channel: str = "ibm_quantum_platform"
    ) -> QiskitRuntimeService:
        """
        Initialize the Qiskit IBM Runtime service.

        Args:
            token: IBM Quantum API token (optional if saved)
            channel: Service channel ('ibm_quantum_platform')

        Returns:
            QiskitRuntimeService: Initialized service instance
        """
        if not token or not token.strip():
            try:
                service = QiskitRuntimeService(channel=channel)
                logger.info("Initialized IBM Runtime service from saved credentials")
                return service
            except Exception as e:
                raise ValueError(
                    "No IBM Quantum token provided and no saved credentials available"
                ) from e

        # If a token is provided, validate it's not a placeholder before saving.
        # Reject tokens that are all the same character (e.g., "xxxx", "0000")
        # as these are likely placeholder values.
        token = token.strip()
        if len(set(token)) == 1 or token in INVALID_PLACEHOLDER_TOKENS:
            raise ValueError(f"Invalid token: '{token}' appears to be a placeholder value")

        # Save account and initialize it with the provided token
        try:
            QiskitRuntimeService.save_account(channel=channel, token=token, overwrite=True)
            service = QiskitRuntimeService(channel=channel)
            logger.info("Successfully initialized IBM Runtime service using provided token")
            return service
        except Exception as e:
            logger.error(f"Failed to initialize IBM Runtime service: {e}")
            raise
