"""Display LUTs for image panels."""

from __future__ import annotations

import numpy as np

LUT_NAMES = ("grey", "turbo", "viridis", "magma", "jet")


def _lerp_table(stops: list[tuple[float, tuple[int, int, int]]], n: int = 256) -> np.ndarray:
    xs = np.array([s[0] for s in stops], dtype=np.float64)
    cs = np.array([s[1] for s in stops], dtype=np.float64)
    t = np.linspace(0.0, 1.0, n)
    out = np.zeros((n, 3), dtype=np.uint8)
    for ch in range(3):
        out[:, ch] = np.clip(np.interp(t, xs, cs[:, ch]), 0, 255).astype(np.uint8)
    return out


def make_lut(name: str) -> np.ndarray:
    """Return (256, 3) uint8 LUT."""
    key = name.lower()
    if key in ("grey", "gray"):
        g = np.arange(256, dtype=np.uint8)
        return np.stack([g, g, g], axis=1)
    if key == "turbo":
        # Compact approximation of matplotlib turbo
        return _lerp_table(
            [
                (0.0, (48, 18, 59)),
                (0.25, (33, 144, 214)),
                (0.5, (94, 201, 98)),
                (0.75, (253, 180, 47)),
                (1.0, (122, 4, 3)),
            ]
        )
    if key == "viridis":
        return _lerp_table(
            [
                (0.0, (68, 1, 84)),
                (0.25, (59, 82, 139)),
                (0.5, (33, 145, 140)),
                (0.75, (94, 201, 98)),
                (1.0, (253, 231, 37)),
            ]
        )
    if key == "magma":
        return _lerp_table(
            [
                (0.0, (0, 0, 4)),
                (0.25, (81, 18, 124)),
                (0.5, (183, 55, 121)),
                (0.75, (251, 135, 97)),
                (1.0, (252, 253, 191)),
            ]
        )
    if key == "jet":
        return _lerp_table(
            [
                (0.0, (0, 0, 127)),
                (0.25, (0, 0, 255)),
                (0.5, (0, 255, 255)),
                (0.75, (255, 255, 0)),
                (1.0, (127, 0, 0)),
            ]
        )
    raise ValueError(f"Unknown LUT: {name}")


def apply_lut(
    image: np.ndarray, lut: np.ndarray, vmin: float, vmax: float
) -> np.ndarray:
    """Map 2D image through LUT → RGB uint8."""
    img = np.asarray(image, dtype=np.float64)
    if vmax <= vmin:
        vmax = vmin + 1.0
    norm = np.clip((img - vmin) / (vmax - vmin), 0.0, 1.0)
    idx = (norm * 255.0).astype(np.uint8)
    return lut[idx]
