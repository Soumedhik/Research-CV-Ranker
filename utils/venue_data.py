"""
utils/venue_data.py
Loads and indexes bundled venue-ranking datasets (CORE, Scimago, aliases).
Provides fast venue → rank lookup with normalization.
"""

import json
import re
from pathlib import Path
from typing import Optional

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Quartile → rank mapping for Scimago
_QUARTILE_TO_RANK = {"Q1": "A*", "Q2": "A", "Q3": "B", "Q4": "C"}


class VenueDataset:
    """
    Unified venue ranking dataset built from:
      1. CORE conference/journal rankings (A*, A, B, C)
      2. Scimago journal quartiles (Q1→A*, Q2→A, Q3→B, Q4→C)
      3. Alias dictionary for name normalization
    """

    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or _DATA_DIR
        # venue_name_lower → rank
        self._rank_index: dict[str, str] = {}
        # alias_lower → canonical_name
        self._alias_index: dict[str, str] = {}
        # canonical_name → rank (from indexed data)
        self._canonical_rank: dict[str, str] = {}

        self._load_core()
        self._load_scimago()
        self._load_aliases()

    def _load_core(self):
        """Load CORE conference rankings."""
        path = self.data_dir / "core_rankings.json"
        if not path.exists():
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for rank in ("A*", "A", "B", "C"):
                for venue in data.get(rank, []):
                    key = venue.strip().lower()
                    # Don't overwrite higher rank with lower
                    if key not in self._rank_index or _rank_priority(rank) < _rank_priority(self._rank_index[key]):
                        self._rank_index[key] = rank
                        self._canonical_rank[venue.strip()] = rank
        except (json.JSONDecodeError, IOError):
            pass

    def _load_scimago(self):
        """Load Scimago journal quartile rankings."""
        path = self.data_dir / "scimago_rankings.json"
        if not path.exists():
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for quartile, rank in _QUARTILE_TO_RANK.items():
                for journal in data.get(quartile, []):
                    key = journal.strip().lower()
                    if key not in self._rank_index or _rank_priority(rank) < _rank_priority(self._rank_index[key]):
                        self._rank_index[key] = rank
                        self._canonical_rank[journal.strip()] = rank
        except (json.JSONDecodeError, IOError):
            pass

    def _load_aliases(self):
        """Load venue alias → canonical name mapping."""
        path = self.data_dir / "venue_aliases.json"
        if not path.exists():
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for section in ("conferences", "journals"):
                for alias, canonical in data.get(section, {}).items():
                    self._alias_index[alias.strip().lower()] = canonical
        except (json.JSONDecodeError, IOError):
            pass

    def normalize(self, venue: str) -> str:
        """
        Normalize a venue name to its canonical form.
        Returns the canonical name if found, else the original cleaned up.
        """
        if not venue:
            return venue
        cleaned = venue.strip()
        lower = cleaned.lower()
        # Remove year suffixes like "NeurIPS 2024" → "neurips"
        lower_no_year = re.sub(r"\s*\d{4}\s*$", "", lower).strip()
        lower_no_year = re.sub(r"\s*'\d{2}\s*$", "", lower_no_year).strip()

        # 1. Exact alias match
        if lower_no_year in self._alias_index:
            return self._alias_index[lower_no_year]
        if lower in self._alias_index:
            return self._alias_index[lower]

        # 2. Substring alias match (for long venue strings)
        for alias_key, canonical in self._alias_index.items():
            if len(alias_key) >= 4 and alias_key in lower_no_year:
                return canonical

        # 3. Return original (title-cased if short, else as-is)
        return cleaned

    def lookup(self, venue: str) -> tuple[str, str]:
        """
        Look up a venue's rank.
        Returns (rank, source) where source is "core", "scimago", "alias", or "unknown".
        """
        if not venue:
            return "?", "unknown"

        # Normalize first
        canonical = self.normalize(venue)
        canonical_lower = canonical.lower()

        # Direct lookup
        if canonical_lower in self._rank_index:
            return self._rank_index[canonical_lower], "dataset"

        # Try original (un-normalized) with year stripped
        original_lower = venue.strip().lower()
        original_no_year = re.sub(r"\s*\d{4}\s*$", "", original_lower).strip()

        if original_no_year in self._rank_index:
            return self._rank_index[original_no_year], "dataset"

        # Fuzzy substring matching against dataset entries
        for indexed_key, rank in self._rank_index.items():
            if len(indexed_key) >= 4:
                if indexed_key in original_no_year or original_no_year in indexed_key:
                    return rank, "dataset_fuzzy"

        return "?", "unknown"

    @property
    def total_venues(self) -> int:
        return len(self._rank_index)

    @property
    def total_aliases(self) -> int:
        return len(self._alias_index)


def _rank_priority(rank: str) -> int:
    """Lower number = higher priority."""
    return {"A*": 0, "A": 1, "B": 2, "C": 3, "?": 4}.get(rank, 5)


# Module-level singleton (lazy)
_instance: Optional[VenueDataset] = None


def get_venue_dataset() -> VenueDataset:
    global _instance
    if _instance is None:
        _instance = VenueDataset()
    return _instance
