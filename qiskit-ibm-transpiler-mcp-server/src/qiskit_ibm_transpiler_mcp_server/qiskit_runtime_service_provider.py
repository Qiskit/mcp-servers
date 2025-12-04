from qiskit_ibm_runtime import QiskitRuntimeService  # type: ignore[import-untyped]
from typing import Optional
import threading

import logging

logger = logging.getLogger(__name__)


class QiskitRuntimeServiceProvider:
    """
    Singleton thread-safe provider with lazy initialization for QiskitRuntimeService
    """

    _instance: Optional["QiskitRuntimeServiceProvider"] = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if (
                    cls._instance is None
                ):  # double check to ensure multiple threads not enter in
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, "_service"):
            self._service: Optional[QiskitRuntimeService] = None
            self._service_lock = threading.Lock()  # lock for lazy init

    def get(
        self, token: Optional[str] = None, channel: str = "ibm_quantum_platform"
    ) -> QiskitRuntimeService:
        """Lazy initialization for QiskitRuntimeService"""
        if self._service is not None:
            return self._service
        with self._service_lock:
            if self._service is None:
                self._service = self._initialize_service(token=token, channel=channel)
        return self._service

    @staticmethod
    def _initialize_service(
        token: Optional[str] = None, channel: str = "ibm_quantum_platform"
    ) -> QiskitRuntimeService:
        """
        Initialize the Qiskit IBM Runtime service.

        Args:
            token: IBM Quantum API token (optional if saved)
            channel: Service channel ('ibm_quantum_platform')

        Returns:
            QiskitRuntimeService: Initialized service instance
        """
        try:
            # First, try to initialize from saved credentials (unless a new token is explicitly provided)
            if not token:
                try:
                    service = QiskitRuntimeService(channel=channel)
                    logger.info(
                        f"Successfully initialized IBM Runtime service from saved credentials on channel: {channel}"
                    )
                    return service
                except Exception as e:
                    logger.info(f"No saved credentials found or invalid: {e}")
                    raise ValueError(
                        "No IBM Quantum token provided and no saved credentials available"
                    ) from e

            # If a token is provided, validate it's not a placeholder before saving
            if token and token.strip():
                # Check for common placeholder patterns
                if token.strip() in ["<PASSWORD>", "<TOKEN>", "YOUR_TOKEN_HERE", "xxx"]:
                    raise ValueError(
                        f"Invalid token: '{token.strip()}' appears to be a placeholder value"
                    )

                # Save account with provided token
                try:
                    QiskitRuntimeService.save_account(
                        channel=channel, token=token.strip(), overwrite=True
                    )
                    logger.info(f"Saved IBM Quantum account for channel: {channel}")
                except Exception as e:
                    logger.error(f"Failed to save account: {e}")
                    raise ValueError("Invalid token or channel") from e

                # Initialize service with the new token
                try:
                    service = QiskitRuntimeService(channel=channel)
                    logger.info(
                        f"Successfully initialized IBM Runtime service on channel: {channel}"
                    )
                    return service
                except Exception as e:
                    logger.error(f"Failed to initialize IBM Runtime service: {e}")
                    raise

        except Exception as e:
            if not isinstance(e, ValueError):
                logger.error(f"Failed to initialize IBM Runtime service: {e}")
            raise
