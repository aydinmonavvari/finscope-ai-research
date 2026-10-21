"""Strict-JSON output helpers (PHASE-2 plan acceptance criterion A8).

Every uncertainty artifact is written through :func:`strict_json_dumps`, which
recursively converts numpy/pandas scalars, rejects NaN/Infinity outright
(value error, not silent null) and serializes with ``allow_nan=False`` so no
non-finite number can ever reach a committed artifact.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


def _clean(obj):
    """Recursively convert to JSON-safe builtins; raise on non-finite floats."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_clean(v) for v in obj]
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        value = float(obj)
        if not math.isfinite(value):
            raise ValueError(
                f"non-finite value {obj!r} cannot be written to a strict-JSON artifact"
            )
        return value
    if isinstance(obj, np.ndarray):
        return _clean(obj.tolist())
    if isinstance(obj, (pd.Series, pd.Index)):
        return _clean(obj.tolist())
    return obj  # str, int, None pass through


def strict_json_dumps(payload) -> str:
    """JSON text (indent 2) with NaN/Infinity forbidden — A8."""
    return json.dumps(_clean(payload), indent=2, allow_nan=False)


def write_strict_json(payload, path: str | Path) -> str:
    """Write a strict-JSON artifact and return the text written."""
    text = strict_json_dumps(payload)
    Path(path).write_text(text + "\n", encoding="utf-8")
    return text
