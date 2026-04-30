from __future__ import annotations

import statistics
import time
from collections.abc import Callable
from typing import Any


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    index = (len(values) - 1) * q
    lower = int(index)
    upper = min(lower + 1, len(values) - 1)
    weight = index - lower
    return values[lower] * (1 - weight) + values[upper] * weight


def timed_call(fn: Callable[[], Any]) -> tuple[Any, float]:
    start = time.perf_counter()
    value = fn()
    return value, (time.perf_counter() - start) * 1000


def summarize_ms(samples: list[float]) -> dict[str, float]:
    return {
        "p50": percentile(samples, 0.50),
        "p99": percentile(samples, 0.99),
        "mean": statistics.fmean(samples) if samples else 0.0,
    }
