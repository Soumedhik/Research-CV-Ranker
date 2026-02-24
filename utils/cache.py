"""
utils/cache.py
File-based caching for paper lookups and venue rankings.
Avoids redundant API calls across runs.
"""

import json
import hashlib
import time
from pathlib import Path
from typing import Optional

from rich.console import Console

console = Console()

# Default cache directory: .cache/ next to the project root
_DEFAULT_CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache"


def _normalize_key(text: str) -> str:
    """Normalize a string into a cache-safe key."""
    cleaned = text.strip().lower()
    # Use a short hash for filesystem safety
    h = hashlib.md5(cleaned.encode("utf-8")).hexdigest()[:12]
    # Also keep a slug for readability
    slug = "".join(c if c.isalnum() else "_" for c in cleaned)[:60]
    return f"{slug}_{h}"


class FileCache:
    """
    Simple JSON-file-backed cache. Each namespace gets its own file.
    Structure: { key: { "data": ..., "ts": unix_timestamp } }
    """

    def __init__(self, namespace: str, cache_dir: Optional[Path] = None,
                 ttl_seconds: int = 7 * 86400):
        self.namespace = namespace
        self.cache_dir = cache_dir or _DEFAULT_CACHE_DIR
        self.ttl = ttl_seconds  # default 7 days
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._file = self.cache_dir / f"{namespace}.json"
        self._data: dict = self._load()

    def _load(self) -> dict:
        if self._file.exists():
            try:
                with open(self._file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return {}
        return {}

    def _save(self):
        try:
            with open(self._file, "w", encoding="utf-8") as f:
                json.dump(self._data, f, indent=1, default=str)
        except IOError:
            pass  # non-critical

    def get(self, raw_key: str) -> Optional[dict]:
        """Retrieve cached data, or None if missing/expired."""
        key = _normalize_key(raw_key)
        entry = self._data.get(key)
        if entry is None:
            return None
        # Check expiry
        ts = entry.get("ts", 0)
        if time.time() - ts > self.ttl:
            del self._data[key]
            return None
        return entry.get("data")

    def put(self, raw_key: str, data: dict):
        """Store data in the cache."""
        key = _normalize_key(raw_key)
        self._data[key] = {"data": data, "ts": time.time()}

    def flush(self):
        """Write current cache state to disk."""
        self._save()

    def stats(self) -> dict:
        """Return cache statistics."""
        valid = sum(
            1 for v in self._data.values()
            if time.time() - v.get("ts", 0) <= self.ttl
        )
        return {
            "namespace": self.namespace,
            "total_entries": len(self._data),
            "valid_entries": valid,
            "file": str(self._file),
        }

    def clear(self):
        """Remove all entries."""
        self._data = {}
        self._save()


class PaperCache(FileCache):
    """Cache for Semantic Scholar / OpenAlex paper lookups, keyed by title."""

    def __init__(self, cache_dir: Optional[Path] = None):
        super().__init__("papers", cache_dir=cache_dir, ttl_seconds=30 * 86400)  # 30 days


class VenueCache(FileCache):
    """Cache for venue → rank mappings."""

    def __init__(self, cache_dir: Optional[Path] = None):
        super().__init__("venues", cache_dir=cache_dir, ttl_seconds=90 * 86400)  # 90 days
