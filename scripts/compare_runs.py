"""Compare two results folders and fail if anything but wall-clock timings differs.

    uv run python scripts/compare_runs.py results/full /path/to/other/results/full

Timing keys (train_seconds, seconds, latency) are ignored, and so is the loss-curve PNG. Every other
JSON value and every JSONL line must be identical.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

TIMING_KEYS = {"train_seconds", "seconds", "latency"}


def strip(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: strip(v) for k, v in obj.items() if k not in TIMING_KEYS}
    if isinstance(obj, list):
        return [strip(v) for v in obj]
    return obj


def main(a: Path, b: Path) -> int:
    files = sorted(p.relative_to(a) for p in a.rglob("*") if p.suffix in {".json", ".jsonl"})
    other = sorted(p.relative_to(b) for p in b.rglob("*") if p.suffix in {".json", ".jsonl"})
    bad = [f"only in one folder: {f}" for f in sorted(set(files) ^ set(other))]
    for f in sorted(set(files) & set(other)):
        ta, tb = (a / f).read_text(encoding="utf-8"), (b / f).read_text(encoding="utf-8")
        same = strip(json.loads(ta)) == strip(json.loads(tb)) if f.suffix == ".json" else ta == tb
        if not same:
            bad.append(f"differs: {f}")
    for line in bad:
        print(line)
    print(f"{len(files)} files compared, {len(bad)} differences (timings ignored)")
    return 1 if bad else 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    sys.exit(main(Path(sys.argv[1]), Path(sys.argv[2])))
