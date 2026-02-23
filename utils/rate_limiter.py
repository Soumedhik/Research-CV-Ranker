"""
utils/rate_limiter.py
Groq API rate limiter — tracks RPM, RPD, TPM, TPD per model.
Proactively waits before sending requests to avoid 429 errors.
"""

import random
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from rich.console import Console

console = Console()

# ── Free-tier rate limits per model (Groq, as of Feb 2026) ────────────────────
MODEL_LIMITS = {
    # Best for batch CV analysis: 30K TPM allows ~5 calls/min
    "meta-llama/llama-4-scout-17b-16e-instruct": {
        "rpm": 30, "rpd": 1000, "tpm": 30_000, "tpd": 500_000,
    },
    "meta-llama/llama-4-maverick-17b-128e-instruct": {
        "rpm": 30, "rpd": 1000, "tpm": 6_000, "tpd": 500_000,
    },
    # Highest RPD (14.4K), good fallback but low TPM
    "llama-3.1-8b-instant": {
        "rpm": 30, "rpd": 14_400, "tpm": 6_000, "tpd": 500_000,
    },
    # Original default — quality but tight limits
    "llama-3.3-70b-versatile": {
        "rpm": 30, "rpd": 1000, "tpm": 12_000, "tpd": 100_000,
    },
    # Legacy fallback
    "llama3-8b-8192": {
        "rpm": 30, "rpd": 14_400, "tpm": 6_000, "tpd": 500_000,
    },
    # Double RPM but low TPM
    "qwen/qwen3-32b": {
        "rpm": 60, "rpd": 1000, "tpm": 6_000, "tpd": 500_000,
    },
    "openai/gpt-oss-120b": {
        "rpm": 30, "rpd": 1000, "tpm": 8_000, "tpd": 200_000,
    },
    "openai/gpt-oss-20b": {
        "rpm": 30, "rpd": 1000, "tpm": 8_000, "tpd": 200_000,
    },
}

# Conservative fallback for unknown models
DEFAULT_LIMITS = {"rpm": 20, "rpd": 500, "tpm": 5_000, "tpd": 100_000}


def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~1 token per 4 characters for English."""
    return max(len(text) // 4, 1)


def get_model_limits(model: str) -> dict:
    """Retrieve rate limits for a model, with fallback."""
    return MODEL_LIMITS.get(model, DEFAULT_LIMITS)


def backoff_delay(attempt: int, base: float = 2.0, max_delay: float = 60.0) -> float:
    """Exponential backoff with jitter."""
    delay = min(base * (2 ** attempt) + random.uniform(0, 1), max_delay)
    return delay


@dataclass
class UsageStats:
    """Accumulated API usage statistics."""
    total_requests: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0
    rate_limit_waits: int = 0
    total_wait_seconds: float = 0.0
    retries: int = 0

    @property
    def avg_tokens_per_request(self) -> float:
        return self.total_tokens / max(self.total_requests, 1)


class GroqRateLimiter:
    """
    Proactive rate limiter for Groq free tier.

    Tracks requests and tokens in sliding 60-second windows (RPM/TPM)
    and cumulative daily totals (RPD/TPD). Automatically waits when
    approaching limits instead of letting the API return 429 errors.
    """

    SAFETY_MARGIN = 0.85  # use 85% of limits to avoid edge-case 429s

    def __init__(self, model: str, verbose: bool = False):
        self.model = model
        self.verbose = verbose
        self.limits = get_model_limits(model)
        self.stats = UsageStats()

        # Sliding window tracking
        self._request_times: deque = deque()        # timestamps (for RPM)
        self._token_entries: deque = deque()         # (timestamp, tokens) for TPM

    def update_model(self, model: str):
        """Switch to tracking a different model's limits."""
        self.model = model
        self.limits = get_model_limits(model)

    def wait_if_needed(self, estimated_input_tokens: int = 0):
        """
        Block until it is safe to send the next request.
        Checks RPM, TPM, RPD, TPD with safety margins.
        """
        estimated_total = estimated_input_tokens + 1200  # estimate completion output
        self._cleanup_windows()

        # ── Check daily limits (hard stop) ─────────────────────────────────
        safe_rpd = int(self.limits["rpd"] * self.SAFETY_MARGIN)
        if self.stats.total_requests >= safe_rpd:
            raise RuntimeError(
                f"Approaching daily request limit ({self.stats.total_requests}/{self.limits['rpd']} RPD). "
                f"Stopping to avoid lockout. Resume tomorrow or switch model."
            )

        safe_tpd = int(self.limits["tpd"] * self.SAFETY_MARGIN)
        if self.stats.total_tokens + estimated_total > safe_tpd:
            raise RuntimeError(
                f"Approaching daily token limit ({self.stats.total_tokens:,}/{self.limits['tpd']:,} TPD). "
                f"Stopping to avoid lockout."
            )

        # ── Check RPM — wait if at capacity ────────────────────────────────
        safe_rpm = int(self.limits["rpm"] * self.SAFETY_MARGIN)
        if len(self._request_times) >= safe_rpm:
            oldest = self._request_times[0]
            wait_time = (oldest + 60.0) - time.time() + 0.5
            if wait_time > 0:
                self._wait(wait_time, f"RPM {len(self._request_times)}/{self.limits['rpm']}")
                self._cleanup_windows()

        # ── Check TPM — wait if at capacity ────────────────────────────────
        safe_tpm = int(self.limits["tpm"] * self.SAFETY_MARGIN)
        current_tpm = sum(t for _, t in self._token_entries)
        if current_tpm + estimated_total > safe_tpm:
            if self._token_entries:
                oldest_time = self._token_entries[0][0]
                wait_time = (oldest_time + 60.0) - time.time() + 1.0
                if wait_time > 0:
                    self._wait(wait_time, f"TPM {current_tpm:,}/{self.limits['tpm']:,}")
                    self._cleanup_windows()

        # ── Minimum inter-request gap to spread load evenly ────────────────
        if self._request_times:
            last_request = self._request_times[-1]
            min_gap = 60.0 / max(safe_rpm, 1)
            elapsed = time.time() - last_request
            if elapsed < min_gap:
                time.sleep(min_gap - elapsed)

    def record_usage(self, prompt_tokens: int, completion_tokens: int):
        """Record actual token usage from an API response."""
        now = time.time()
        total = prompt_tokens + completion_tokens

        self._request_times.append(now)
        self._token_entries.append((now, total))

        self.stats.total_requests += 1
        self.stats.total_prompt_tokens += prompt_tokens
        self.stats.total_completion_tokens += completion_tokens
        self.stats.total_tokens += total

    def get_summary(self) -> dict:
        """Return usage stats for display."""
        limits = self.limits
        return {
            "model": self.model,
            "requests_made": self.stats.total_requests,
            "requests_limit_daily": limits["rpd"],
            "requests_remaining": limits["rpd"] - self.stats.total_requests,
            "tokens_used": self.stats.total_tokens,
            "tokens_limit_daily": limits["tpd"],
            "tokens_remaining": limits["tpd"] - self.stats.total_tokens,
            "prompt_tokens": self.stats.total_prompt_tokens,
            "completion_tokens": self.stats.total_completion_tokens,
            "avg_tokens_per_call": round(self.stats.avg_tokens_per_request),
            "rate_limit_waits": self.stats.rate_limit_waits,
            "wait_seconds": round(self.stats.total_wait_seconds, 1),
            "retries": self.stats.retries,
            "rpm": limits["rpm"],
            "tpm": limits["tpm"],
        }

    # ── Internal ───────────────────────────────────────────────────────────────

    def _cleanup_windows(self):
        """Remove entries older than 60 seconds from sliding windows."""
        cutoff = time.time() - 60.0
        while self._request_times and self._request_times[0] < cutoff:
            self._request_times.popleft()
        while self._token_entries and self._token_entries[0][0] < cutoff:
            self._token_entries.popleft()

    def _wait(self, seconds: float, reason: str):
        """Wait with optional logging."""
        self.stats.rate_limit_waits += 1
        self.stats.total_wait_seconds += seconds
        if self.verbose:
            console.print(f"  [dim yellow]\u23f3 Waiting {seconds:.1f}s ({reason})[/dim yellow]")
        time.sleep(seconds)
