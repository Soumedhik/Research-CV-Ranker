"""
config.py — ResearchRank configuration
Edit this file or use environment variables to configure the tool.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field

# Load .env from project root (same directory as config.py)
from dotenv import load_dotenv
_env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(_env_path)


@dataclass
class Config:
    # ── Groq LLM ──────────────────────────────────────────────────────────────
    groq_api_key: str = field(default_factory=lambda: os.environ.get("GROQ_API_KEY", ""))
    # Scout-17B: 30K TPM / 500K TPD — best throughput on free tier
    groq_model: str = "meta-llama/llama-4-scout-17b-16e-instruct"
    # 8B instant: 14.4K RPD / 500K TPD — highest daily request quota
    groq_fallback_model: str = "llama-3.1-8b-instant"
    groq_max_tokens: int = 1500                      # Enough for structured JSON output
    groq_temperature: float = 0.1                     # Low temp for structured extraction
    groq_cv_truncate_chars: int = 10000               # Truncate CV text to save tokens
    groq_max_retries: int = 3                         # Max retries on rate-limit / transient errors

    # ── Semantic Scholar API (free, no key needed) ─────────────────────────────
    semantic_scholar_base: str = "https://api.semanticscholar.org/graph/v1"
    semantic_scholar_delay: float = 1.0  # seconds between requests (rate limit)

    # ── OpenAlex API (free, no key needed) ────────────────────────────────────
    openalex_base: str = "https://api.openalex.org"
    openalex_email: str = "researchrank@lab.local"  # polite pool

    # ── CORE ranking data (bundled) ────────────────────────────────────────────
    # Venue rank inference uses local heuristics + Scimago/CORE name matching
    use_bundled_venue_data: bool = True

    # ── Scoring defaults ───────────────────────────────────────────────────────
    default_weights: dict = field(default_factory=lambda: {
        "publications":  0.30,   # Paper count, venue rank, citations
        "research":      0.25,   # Research experience quality & duration
        "education":     0.20,   # Institution prestige, degree level
        "fit":           0.15,   # Alignment with job description (if provided)
        "trajectory":    0.10,   # Career progression, recency of activity
    })

    # ── Institution tiers (used for education scoring) ────────────────────────
    tier1_keywords: list = field(default_factory=lambda: [
        "mit", "stanford", "harvard", "berkeley", "cambridge", "oxford",
        "eth zurich", "caltech", "princeton", "cmu", "carnegie mellon",
        "toronto", "montreal", "nus", "tsinghua", "peking", "epfl",
        "imperial", "ucl", "edinburgh", "amsterdam", "tu delft",
    ])
    tier2_keywords: list = field(default_factory=lambda: [
        "yale", "columbia", "michigan", "ucla", "ucsd", "cornell",
        "illinois", "purdue", "maryland", "washington", "gatech",
        "georgia tech", "upenn", "nyu", "usc", "boston", "northeastern",
        "melbourne", "sydney", "auckland", "seoul national", "kaist",
        "iit", "bits", "inria", "max planck", "mpi",
    ])

    # ── Venue rank tiers ───────────────────────────────────────────────────────
    # Pattern-matched against venue names (case-insensitive)
    astar_venues: list = field(default_factory=lambda: [
        "neurips", "nips", "icml", "iclr", "cvpr", "iccv", "eccv",
        "acl", "emnlp", "naacl", "sigir", "kdd", "www", "stoc", "focs",
        "sosp", "osdi", "nsdi", "sigcomm", "pldi", "popl", "usenix security",
        "ieee s&p", "ccs", "ndss", "vldb", "sigmod", "icde",
        "nature", "science", "cell", "nejm", "lancet", "pnas",
    ])
    a_venues: list = field(default_factory=lambda: [
        "aaai", "ijcai", "uai", "aistats", "interspeech", "icassp",
        "wsdm", "recsys", "ecir", "coling", "eacl", "acl findings",
        "icse", "fse", "ase", "issta", "eurosys", "atc", "middleware",
        "tpds", "tocs", "tois", "tkde", "tnnls", "tpami", "jmlr",
        "machine learning", "artificial intelligence journal",
    ])
    b_venues: list = field(default_factory=lambda: [
        "icwsm", "cikm", "pakdd", "icdm", "ssdm", "bigdata",
        "icpr", "bmvc", "wacv", "accv",
        "peerj", "plos one", "ieee access", "frontiers",
    ])

    def validate(self):
        if not self.groq_api_key:
            raise ValueError(
                "GROQ_API_KEY not set.\n"
                "Get a free key at https://console.groq.com → set it:\n"
                "  export GROQ_API_KEY=gsk_..."
            )
