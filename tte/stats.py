"""Paired comparison of two models scored on the same items."""

from __future__ import annotations

import math
from typing import Any


def mcnemar_exact(before: list[bool], after: list[bool]) -> dict[str, Any]:
    """Exact two-sided McNemar test on paired right/wrong outcomes.

    Only the discordant items carry information: `fixed` were wrong before and right after, `broken` the
    reverse. Under the null hypothesis (no difference) each discordant item is a fair coin flip, so the
    p-value is a two-sided binomial test of min(fixed, broken) out of fixed + broken at p = 0.5.
    """
    if len(before) != len(after):
        raise ValueError("paired outcomes must have the same length")
    fixed = sum((not b) and a for b, a in zip(before, after, strict=True))
    broken = sum(b and (not a) for b, a in zip(before, after, strict=True))
    n = fixed + broken
    k = min(fixed, broken)
    p = 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2**n)
    return {"n": len(before), "fixed": fixed, "broken": broken, "p_value": p}
