# io/zoom_lut_loader.py
"""
Zoom LUT loader & utilities.

What this module does
- Load a zoom → intrinsics LUT from JSON / YAML / CSV.
- Validate & normalize entries (floats; sorted; deduplicated).
- Interpolate (linear) fx, fy, cx, cy for an arbitrary zoom step.
- Save a normalized LUT back to disk (JSON).

Shapes
- LUT in memory: Dict[float, Tuple[fx, fy, cx, cy]]
- Interpolator returns: Dict[str, float] with keys "fx","fy","cx","cy"

Why this exists
The doc recommends calibrating intrinsics at 3–5 zoom positions and
interpolating K(steps) at runtime; skipping K updates after zoom causes
depth drift—so keep a LUT and query it whenever zoom changes.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, Iterable, List, Any, Optional
import csv
import json
import math
import os

try:
    import yaml  # type: ignore
    _HAVE_YAML = True
except Exception:
    _HAVE_YAML = False


__all__ = [
    "load_zoom_lut",
    "save_zoom_lut",
    "validate_zoom_lut",
    "interpolate_zoom_lut",
    "ZoomLUT",
]


# -----------------------
# Core dataclass (optional convenience)
# -----------------------

@dataclass(frozen=True)
class ZoomLUT:
    """Convenience wrapper around the normalized dict LUT."""
    table: Dict[float, Tuple[float, float, float, float]]

    def steps(self) -> List[float]:
        return sorted(self.table.keys())

    def as_list(self) -> List[Tuple[float, float, float, float, float]]:
        """[(steps, fx, fy, cx, cy), ...] sorted by steps."""
        out = []
        for s in self.steps():
            fx, fy, cx, cy = self.table[s]
            out.append((s, fx, fy, cx, cy))
        return out

    def interp(self, steps: float, clamp: bool = True) -> Dict[str, float]:
        return interpolate_zoom_lut(steps, self.table, clamp=clamp)


# -----------------------
# Public API
# -----------------------

def load_zoom_lut(path: str) -> Dict[float, Tuple[float, float, float, float]]:
    """
    Load a zoom LUT from JSON/YAML/CSV and return a normalized dict:
        {steps: (fx, fy, cx, cy)}
    - steps, fx, fy, cx, cy are floats
    - steps keys are sorted and unique
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"zoom LUT not found: {path}")

    ext = os.path.splitext(path)[1].lower()
    if ext in (".json",):
        lut = _load_json(path)
    elif ext in (".yml", ".yaml"):
        if not _HAVE_YAML:
            raise RuntimeError("PyYAML not installed; cannot read YAML LUT")
        lut = _load_yaml(path)
    elif ext in (".csv",):
        lut = _load_csv(path)
    else:
        # Try JSON first, fall back to YAML, then CSV
        try:
            lut = _load_json(path)
        except Exception:
            if _HAVE_YAML:
                try:
                    lut = _load_yaml(path)
                except Exception:
                    lut = _load_csv(path)
            else:
                lut = _load_csv(path)

    return validate_zoom_lut(lut)


def save_zoom_lut(path: str, lut: Dict[float, Tuple[float, float, float, float]]) -> None:
    """
    Save a normalized LUT to JSON as a list of objects:
      [{"steps": 1000, "fx":..., "fy":..., "cx":..., "cy":...}, ...]
    """
    norm = validate_zoom_lut(lut)
    rows = [{"steps": s, "fx": v[0], "fy": v[1], "cx": v[2], "cy": v[3]} for s, v in sorted(norm.items())]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)


def validate_zoom_lut(lut: Any) -> Dict[float, Tuple[float, float, float, float]]:
    """
    Normalize and validate a LUT-like object into:
       {steps(float): (fx, fy, cx, cy)}
    Requirements:
      - ≥ 2 unique entries
      - fx, fy > 0
      - steps strictly increasing after normalization
    Duplicates: last value wins (by steps).
    """
    if lut is None:
        raise ValueError("Empty LUT")

    # Accept dict-like or list-like inputs
    items: List[Tuple[float, Tuple[float, float, float, float]]] = []

    if isinstance(lut, dict):
        # Could be {"1000": {"fx":..}, ...} or {1000: (fx,fy,cx,cy)}
        for k, v in lut.items():
            s = _to_float(k, name="steps")
            fx, fy, cx, cy = _unpack_values(v)
            items.append((s, (fx, fy, cx, cy)))
    elif isinstance(lut, (list, tuple)):
        # Could be [{"steps":..., "fx":...}, ...] or [(steps, fx, fy, cx, cy), ...]
        for row in lut:
            if isinstance(row, dict):
                s = _to_float(row.get("steps"), name="steps")
                fx = _to_float(row.get("fx"), name="fx")
                fy = _to_float(row.get("fy"), name="fy")
                cx = _to_float(row.get("cx"), name="cx")
                cy = _to_float(row.get("cy"), name="cy")
                items.append((s, (fx, fy, cx, cy)))
            else:
                try:
                    s, fx, fy, cx, cy = row  # type: ignore
                except Exception as e:
                    raise ValueError(f"Invalid row format in LUT: {row!r}") from e
                items.append((_to_float(s, "steps"),
                              (_to_float(fx, "fx"), _to_float(fy, "fy"),
                               _to_float(cx, "cx"), _to_float(cy, "cy"))))
    else:
        raise ValueError(f"Unsupported LUT type: {type(lut)}")

    # Deduplicate: last one wins
    d: Dict[float, Tuple[float, float, float, float]] = {}
    for s, vals in items:
        d[s] = vals

    # Validate values
    for s, (fx, fy, cx, cy) in d.items():
        if not (math.isfinite(fx) and math.isfinite(fy) and math.isfinite(cx) and math.isfinite(cy)):
            raise ValueError(f"Non-finite values at steps={s}")
        if fx <= 0 or fy <= 0:
            raise ValueError(f"fx,fy must be > 0 at steps={s}")

    if len(d) < 2:
        raise ValueError("LUT must have at least 2 unique zoom positions")

    # Sort by steps
    sorted_items = sorted(d.items(), key=lambda kv: kv[0])
    # Ensure strictly increasing
    last_s = None
    for s, _ in sorted_items:
        if last_s is not None and s <= last_s:
            raise ValueError(f"steps must be strictly increasing (got {s} after {last_s})")
        last_s = s

    return dict(sorted_items)


def interpolate_zoom_lut(steps: float,
                         lut: Dict[float, Tuple[float, float, float, float]],
                         *,
                         clamp: bool = True) -> Dict[str, float]:
    """
    Interpolate (linear) fx, fy, cx, cy at a given zoom step.

    Args:
      steps: zoom motor steps (float)
      lut: normalized LUT {steps: (fx, fy, cx, cy)}
      clamp: clamp outside range to endpoints (True) or raise (False)

    Returns:
      {"fx": fx, "fy": fy, "cx": cx, "cy": cy}
    """
    if not lut or len(lut) < 2:
        raise ValueError("Need a LUT with at least 2 entries")

    xs = sorted(lut.keys())
    if steps <= xs[0]:
        if not clamp:
            raise ValueError(f"steps {steps} < LUT min {xs[0]}")
        s0 = xs[0]
        fx, fy, cx, cy = lut[s0]
        return {"fx": fx, "fy": fy, "cx": cx, "cy": cy}
    if steps >= xs[-1]:
        if not clamp:
            raise ValueError(f"steps {steps} > LUT max {xs[-1]}")
        s1 = xs[-1]
        fx, fy, cx, cy = lut[s1]
        return {"fx": fx, "fy": fy, "cx": cx, "cy": cy}

    # Find bracketing points
    lo = 0
    hi = len(xs) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if xs[mid] <= steps:
            lo = mid
        else:
            hi = mid
    s0, s1 = xs[lo], xs[hi]
    t = (steps - s0) / (s1 - s0)

    fx0, fy0, cx0, cy0 = lut[s0]
    fx1, fy1, cx1, cy1 = lut[s1]

    fx = fx0 * (1 - t) + fx1 * t
    fy = fy0 * (1 - t) + fy1 * t
    # In many lenses cx,cy are ~constant; still interpolate in case you modeled drift.
    cx = cx0 * (1 - t) + cx1 * t
    cy = cy0 * (1 - t) + cy1 * t

    return {"fx": fx, "fy": fy, "cx": cx, "cy": cy}


# -----------------------
# File readers
# -----------------------

def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def _load_yaml(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def _load_csv(path: str) -> Any:
    rows = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        # Expect headers: steps, fx, fy, cx, cy
        for rec in r:
            rows.append({
                "steps": _to_float(rec.get("steps"), "steps"),
                "fx": _to_float(rec.get("fx"), "fx"),
                "fy": _to_float(rec.get("fy"), "fy"),
                "cx": _to_float(rec.get("cx"), "cx"),
                "cy": _to_float(rec.get("cy"), "cy"),
            })
    return rows


# -----------------------
# Helpers
# -----------------------

def _to_float(x: Any, name: str) -> float:
    if x is None:
        raise ValueError(f"Missing value for {name}")
    try:
        return float(x)
    except Exception as e:
        raise ValueError(f"Invalid {name} value: {x!r}") from e

def _unpack_values(v: Any) -> Tuple[float, float, float, float]:
    if isinstance(v, dict):
        return (_to_float(v.get("fx"), "fx"),
                _to_float(v.get("fy"), "fy"),
                _to_float(v.get("cx"), "cx"),
                _to_float(v.get("cy"), "cy"))
    try:
        fx, fy, cx, cy = v  # type: ignore
    except Exception as e:
        raise ValueError(f"Invalid LUT value (need fx,fy,cx,cy): {v!r}") from e
    return float(fx), float(fy), float(cx), float(cy)


# -----------------------
# CLI smoke test
# -----------------------

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Zoom LUT loader")
    ap.add_argument("path", help="Path to LUT (json/yaml/csv)")
    ap.add_argument("--steps", type=float, default=None, help="Zoom steps to interpolate")
    ap.add_argument("--save", type=str, default=None, help="Save normalized JSON here")
    ns = ap.parse_args()

    lut = load_zoom_lut(ns.path)
    zlut = ZoomLUT(lut)

    print(f"Loaded {len(zlut.table)} entries")
    print("First 3 rows:", zlut.as_list()[:3])

    if ns.steps is not None:
        k = zlut.interp(ns.steps)
        print(f"Interpolated at steps={ns.steps}: fx={k['fx']:.2f} fy={k['fy']:.2f} cx={k['cx']:.2f} cy={k['cy']:.2f}")

    if ns.save:
        save_zoom_lut(ns.save, zlut.table)
        print(f"Saved normalized LUT → {ns.save}")

