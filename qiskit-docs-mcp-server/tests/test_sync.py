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

"""Tests for sync module."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
from qiskit_docs_mcp_server.sync import _build_sync_urls, sync_all_docs


class TestBuildSyncUrls:
    """Test _build_sync_urls function."""

    def test_returns_nonempty_list(self):
        """Test that sync URLs list is not empty."""
        urls = _build_sync_urls()
        assert len(urls) > 0

    def test_includes_modules(self):
        """Test that module URLs are included."""
        urls = _build_sync_urls()
        labels = [label for _, _, label in urls]
        assert any(label.startswith("module:") for label in labels)

    def test_includes_addons(self):
        """Test that addon URLs are included."""
        urls = _build_sync_urls()
        labels = [label for _, _, label in urls]
        assert any(label.startswith("addon:") for label in labels)

    def test_includes_guides(self):
        """Test that guide URLs are included."""
        urls = _build_sync_urls()
        labels = [label for _, _, label in urls]
        assert any(label.startswith("guide:") for label in labels)

    def test_includes_errors_page(self):
        """Test that the errors page is included."""
        urls = _build_sync_urls()
        labels = [label for _, _, label in urls]
        assert "errors" in labels

    def test_all_entries_are_tuples(self):
        """Test that all entries are (url, content_type, label) tuples."""
        urls = _build_sync_urls()
        for entry in urls:
            assert len(entry) == 3
            url, content_type, label = entry
            assert isinstance(url, str)
            assert content_type in ("text", "json")
            assert isinstance(label, str)


class TestSyncAllDocs:
    """Test sync_all_docs function."""

    @patch("qiskit_docs_mcp_server.sync.httpx.AsyncClient")
    async def test_sync_downloads_pages(self, mock_client_class, tmp_path: Path):
        """Test that sync downloads all pages."""
        mock_response = MagicMock()
        mock_response.text = "<html>doc content</html>"
        mock_response.raise_for_status = MagicMock()
        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        mock_client_class.return_value.__aenter__.return_value = mock_client

        result = await sync_all_docs(str(tmp_path), ttl=3600.0)
        assert result["total"] > 0
        assert result["success"] == result["total"]
        assert result["failed"] == 0

    @patch("qiskit_docs_mcp_server.sync.httpx.AsyncClient")
    async def test_sync_handles_failures(self, mock_client_class, tmp_path: Path):
        """Test that sync handles individual page failures."""
        mock_client = AsyncMock()
        mock_client.get.side_effect = httpx.HTTPError("Connection failed")
        mock_client_class.return_value.__aenter__.return_value = mock_client

        result = await sync_all_docs(str(tmp_path), ttl=3600.0)
        assert result["total"] > 0
        assert result["success"] == 0
        assert result["failed"] == result["total"]

    @patch("qiskit_docs_mcp_server.sync.httpx.AsyncClient")
    async def test_sync_writes_to_disk(self, mock_client_class, tmp_path: Path):
        """Test that sync writes cache files to disk."""
        mock_response = MagicMock()
        mock_response.text = "<html>content</html>"
        mock_response.raise_for_status = MagicMock()
        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        mock_client_class.return_value.__aenter__.return_value = mock_client

        await sync_all_docs(str(tmp_path), ttl=3600.0)
        json_files = list(tmp_path.glob("*.json"))
        assert len(json_files) > 0

    @patch("qiskit_docs_mcp_server.sync.httpx.AsyncClient")
    async def test_sync_progress_callback(self, mock_client_class, tmp_path: Path):
        """Test that progress callback is called."""
        mock_response = MagicMock()
        mock_response.text = "<html>content</html>"
        mock_response.raise_for_status = MagicMock()
        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        mock_client_class.return_value.__aenter__.return_value = mock_client

        calls: list[tuple[int, int, str]] = []

        def progress(current: int, total: int, label: str) -> None:
            calls.append((current, total, label))

        await sync_all_docs(str(tmp_path), ttl=3600.0, progress_callback=progress)
        assert len(calls) > 0
        # First call should be (1, total, ...)
        assert calls[0][0] == 1
        # Last call should be (total, total, ...)
        assert calls[-1][0] == calls[-1][1]

    @patch("qiskit_docs_mcp_server.sync.httpx.AsyncClient")
    async def test_sync_partial_failure(self, mock_client_class, tmp_path: Path):
        """Test sync with some pages failing."""
        call_count = 0

        async def mock_get(url, **kwargs):  # type: ignore[no-untyped-def]
            nonlocal call_count
            call_count += 1
            if call_count % 3 == 0:
                raise httpx.HTTPError("Intermittent failure")
            response = MagicMock()
            response.text = "<html>content</html>"
            response.raise_for_status = MagicMock()
            return response

        mock_client = AsyncMock()
        mock_client.get.side_effect = mock_get
        mock_client_class.return_value.__aenter__.return_value = mock_client

        result = await sync_all_docs(str(tmp_path), ttl=3600.0)
        assert result["success"] > 0
        assert result["failed"] > 0
        assert result["success"] + result["failed"] == result["total"]
