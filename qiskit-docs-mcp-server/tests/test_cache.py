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

"""Tests for cache module."""

from pathlib import Path

from qiskit_docs_mcp_server.cache import DiskCache, LayeredCache, TTLCache


class TestTTLCache:
    """Test TTLCache class."""

    def test_put_and_get(self):
        """Test basic put and get."""
        cache = TTLCache(maxsize=10, ttl=60.0)
        cache.put("key1", "value1", "text")
        entry = cache.get("key1")
        assert entry is not None
        assert entry.value == "value1"

    def test_get_nonexistent_returns_none(self):
        """Test get for missing key."""
        cache = TTLCache(maxsize=10, ttl=60.0)
        assert cache.get("nonexistent") is None

    def test_expired_entry_returns_none(self):
        """Test that expired entries are not returned."""
        cache = TTLCache(maxsize=10, ttl=0.0)  # Immediate expiry
        cache.put("key1", "value1", "text")
        # Entry expires immediately
        assert cache.get("key1") is None

    def test_lru_eviction_at_maxsize(self):
        """Test LRU eviction when cache is full."""
        cache = TTLCache(maxsize=2, ttl=60.0)
        cache.put("key1", "value1", "text")
        cache.put("key2", "value2", "text")
        cache.put("key3", "value3", "text")  # Should evict key1
        assert cache.get("key1") is None
        assert cache.get("key2") is not None
        assert cache.get("key3") is not None

    def test_get_moves_to_most_recent(self):
        """Test that get moves entry to end (most recent)."""
        cache = TTLCache(maxsize=2, ttl=60.0)
        cache.put("key1", "value1", "text")
        cache.put("key2", "value2", "text")
        # Access key1 to make it most recent
        cache.get("key1")
        # Adding key3 should evict key2 (least recently used)
        cache.put("key3", "value3", "text")
        assert cache.get("key1") is not None
        assert cache.get("key2") is None

    def test_clear(self):
        """Test clear removes all entries and resets stats."""
        cache = TTLCache(maxsize=10, ttl=60.0)
        cache.put("key1", "value1", "text")
        cache.get("key1")  # Hit
        cache.get("key2")  # Miss
        cache.clear()
        info = cache.info()
        assert info["size"] == 0
        assert info["hits"] == 0
        assert info["misses"] == 0
        # Verify entries were actually removed
        assert cache.get("key1") is None

    def test_info(self):
        """Test info reports correct statistics."""
        cache = TTLCache(maxsize=10, ttl=60.0)
        cache.put("key1", "value1", "text")
        cache.get("key1")  # Hit
        cache.get("nonexistent")  # Miss
        info = cache.info()
        assert info["hits"] == 1
        assert info["misses"] == 1
        assert info["size"] == 1
        assert info["maxsize"] == 10
        assert info["ttl"] == 60.0

    def test_put_overwrites_existing(self):
        """Test that putting same key overwrites."""
        cache = TTLCache(maxsize=10, ttl=60.0)
        cache.put("key1", "old_value", "text")
        cache.put("key1", "new_value", "text")
        entry = cache.get("key1")
        assert entry is not None
        assert entry.value == "new_value"

    def test_last_was_hit(self):
        """Test last_was_hit property."""
        cache = TTLCache(maxsize=10, ttl=60.0)
        cache.put("key1", "value1", "text")
        cache.get("key1")
        assert cache.last_was_hit is True
        cache.get("nonexistent")
        assert cache.last_was_hit is False

    def test_json_content_type(self):
        """Test storing JSON content."""
        cache = TTLCache(maxsize=10, ttl=60.0)
        data = [{"name": "test"}]
        cache.put("key1", data, "json")
        entry = cache.get("key1")
        assert entry is not None
        assert entry.value == data
        assert entry.content_type == "json"


class TestDiskCache:
    """Test DiskCache class."""

    def test_put_creates_file(self, tmp_path: Path):
        """Test that put creates a cache file."""
        cache = DiskCache(tmp_path, ttl=60.0)
        cache.put("https://example.com/test", "content", "text")
        files = list(tmp_path.glob("*.json"))
        assert len(files) == 1

    def test_get_reads_from_file(self, tmp_path: Path):
        """Test that get reads cached content."""
        cache = DiskCache(tmp_path, ttl=60.0)
        cache.put("https://example.com/test", "content", "text")
        result = cache.get("https://example.com/test")
        assert result is not None
        assert result.value == "content"
        assert result.content_type == "text"
        assert result.is_stale is False

    def test_get_nonexistent_returns_none(self, tmp_path: Path):
        """Test get for missing URL."""
        cache = DiskCache(tmp_path, ttl=60.0)
        assert cache.get("https://example.com/missing") is None

    def test_expired_entry_marked_stale(self, tmp_path: Path):
        """Test that expired entries are marked stale."""
        cache = DiskCache(tmp_path, ttl=0.0)  # Immediate expiry
        cache.put("https://example.com/stale", "old content", "text")
        result = cache.get("https://example.com/stale")
        assert result is not None
        assert result.is_stale is True
        assert result.value == "old content"

    def test_get_stale_returns_expired(self, tmp_path: Path):
        """Test get_stale returns expired entries."""
        cache = DiskCache(tmp_path, ttl=0.0)
        cache.put("https://example.com/stale2", "stale content", "text")
        result = cache.get_stale("https://example.com/stale2")
        assert result is not None
        assert result.value == "stale content"

    def test_clear_removes_files(self, tmp_path: Path):
        """Test clear removes all cache files."""
        cache = DiskCache(tmp_path, ttl=60.0)
        cache.put("https://example.com/1", "content1", "text")
        cache.put("https://example.com/2", "content2", "text")
        assert len(list(tmp_path.glob("*.json"))) == 2
        cache.clear()
        assert len(list(tmp_path.glob("*.json"))) == 0

    def test_info(self, tmp_path: Path):
        """Test info reports file count and size."""
        cache = DiskCache(tmp_path, ttl=60.0)
        cache.put("https://example.com/info", "content", "text")
        info = cache.info()
        assert info["file_count"] == 1
        assert info["total_size_bytes"] > 0
        assert info["directory"] == str(tmp_path)

    def test_corrupted_file_handled(self, tmp_path: Path):
        """Test that corrupted cache files are handled gracefully."""
        cache = DiskCache(tmp_path, ttl=60.0)
        # Write a corrupted file
        filename = cache._url_to_filename("https://example.com/corrupt")
        (tmp_path / filename).write_text("not valid json", encoding="utf-8")
        result = cache.get("https://example.com/corrupt")
        assert result is None
        # File should be removed
        assert not (tmp_path / filename).exists()

    def test_missing_directory_created(self, tmp_path: Path):
        """Test that cache directory is created if missing."""
        new_dir = tmp_path / "subdir" / "cache"
        cache = DiskCache(new_dir, ttl=60.0)
        cache.put("https://example.com/newdir", "content", "text")
        assert new_dir.exists()

    def test_json_content(self, tmp_path: Path):
        """Test storing and retrieving JSON content."""
        cache = DiskCache(tmp_path, ttl=60.0)
        data = [{"name": "test", "value": 42}]
        cache.put("https://example.com/json", data, "json")
        result = cache.get("https://example.com/json")
        assert result is not None
        assert result.value == data
        assert result.content_type == "json"


class TestLayeredCache:
    """Test LayeredCache class."""

    def test_memory_hit_skips_disk(self, tmp_path: Path):
        """Test that memory hit does not check disk."""
        memory = TTLCache(maxsize=10, ttl=60.0)
        disk = DiskCache(tmp_path, ttl=60.0)
        cache = LayeredCache(memory, disk)

        memory.put("key", "memory_value", "text")
        value, is_stale = cache.get("key")
        assert value == "memory_value"
        assert is_stale is False

    def test_memory_miss_checks_disk(self, tmp_path: Path):
        """Test that memory miss falls through to disk."""
        memory = TTLCache(maxsize=10, ttl=60.0)
        disk = DiskCache(tmp_path, ttl=60.0)
        cache = LayeredCache(memory, disk)

        disk.put("https://example.com/disk", "disk_value", "text")
        value, is_stale = cache.get("https://example.com/disk")
        assert value == "disk_value"
        assert is_stale is False

    def test_disk_hit_populates_memory(self, tmp_path: Path):
        """Test that fresh disk hits are promoted to memory."""
        memory = TTLCache(maxsize=10, ttl=60.0)
        disk = DiskCache(tmp_path, ttl=60.0)
        cache = LayeredCache(memory, disk)

        disk.put("https://example.com/promote", "promoted_value", "text")
        cache.get("https://example.com/promote")
        # Now memory should have it
        entry = memory.get("https://example.com/promote")
        assert entry is not None
        assert entry.value == "promoted_value"

    def test_both_miss_returns_none(self):
        """Test that missing from both returns (None, False)."""
        memory = TTLCache(maxsize=10, ttl=60.0)
        cache = LayeredCache(memory, None)
        value, is_stale = cache.get("nonexistent")
        assert value is None
        assert is_stale is False

    def test_stale_fallback_from_disk(self, tmp_path: Path):
        """Test stale fallback returns disk content."""
        memory = TTLCache(maxsize=10, ttl=60.0)
        disk = DiskCache(tmp_path, ttl=0.0)  # Immediate expiry
        cache = LayeredCache(memory, disk)

        disk.put("https://example.com/stale", "stale_value", "text")
        result = cache.get_stale_fallback("https://example.com/stale")
        assert result == "stale_value"

    def test_stale_fallback_no_disk_returns_none(self):
        """Test stale fallback with no disk cache."""
        memory = TTLCache(maxsize=10, ttl=60.0)
        cache = LayeredCache(memory, None)
        assert cache.get_stale_fallback("key") is None

    def test_put_writes_to_both(self, tmp_path: Path):
        """Test put writes to both memory and disk."""
        memory = TTLCache(maxsize=10, ttl=60.0)
        disk = DiskCache(tmp_path, ttl=60.0)
        cache = LayeredCache(memory, disk)

        cache.put("https://example.com/both", "value", "text")
        assert memory.get("https://example.com/both") is not None
        assert disk.get("https://example.com/both") is not None

    def test_clear_clears_both(self, tmp_path: Path):
        """Test clear clears both memory and disk."""
        memory = TTLCache(maxsize=10, ttl=60.0)
        disk = DiskCache(tmp_path, ttl=60.0)
        cache = LayeredCache(memory, disk)

        cache.put("https://example.com/clear", "value", "text")
        cache.clear()
        assert memory.get("https://example.com/clear") is None
        assert disk.get("https://example.com/clear") is None

    def test_disk_none_memory_only(self):
        """Test that LayeredCache works with disk=None (memory only)."""
        memory = TTLCache(maxsize=10, ttl=60.0)
        cache = LayeredCache(memory, None)

        cache.put("key", "value", "text")
        value, is_stale = cache.get("key")
        assert value == "value"
        assert is_stale is False

    def test_info_memory_only(self):
        """Test info with memory-only cache."""
        memory = TTLCache(maxsize=10, ttl=60.0)
        cache = LayeredCache(memory, None)
        info = cache.info()
        assert "memory" in info
        assert "disk" not in info

    def test_info_with_disk(self, tmp_path: Path):
        """Test info with disk cache enabled."""
        memory = TTLCache(maxsize=10, ttl=60.0)
        disk = DiskCache(tmp_path, ttl=60.0)
        cache = LayeredCache(memory, disk)
        info = cache.info()
        assert "memory" in info
        assert "disk" in info

    def test_stale_disk_entry_returned_with_flag(self, tmp_path: Path):
        """Test that stale disk entries are returned with is_stale=True."""
        memory = TTLCache(maxsize=10, ttl=60.0)
        disk = DiskCache(tmp_path, ttl=0.0)  # Immediate expiry
        cache = LayeredCache(memory, disk)

        disk.put("https://example.com/stale-flag", "stale", "text")
        # Manually write with old timestamp to ensure staleness
        value, is_stale = cache.get("https://example.com/stale-flag")
        assert value == "stale"
        assert is_stale is True
