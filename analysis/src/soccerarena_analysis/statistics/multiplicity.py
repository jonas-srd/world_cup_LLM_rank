from __future__ import annotations

from collections.abc import Mapping


def holm_adjust(pvalues: Mapping[str, float]) -> dict[str, float]:
    """Holm step-down adjusted p-values with monotonicity enforcement."""
    items = sorted(pvalues.items(), key=lambda item: item[1])
    count = len(items)
    adjusted: dict[str, float] = {}
    running = 0.0
    for rank, (name, value) in enumerate(items):
        candidate = min(1.0, (count - rank) * float(value))
        running = max(running, candidate)
        adjusted[name] = running
    return adjusted
