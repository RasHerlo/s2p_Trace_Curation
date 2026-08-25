"""Display LUTs for image panels."""

from __future__ import annotations

import numpy as np

LUT_NAMES = ("grey", "turbo", "viridis", "magma", "jet")

# LED+Shutter / missing samples in the raster (all LUTs)
RASTER_NAN_RGB = (128, 128, 128)


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


def lut_with_revert(name: str, revert: bool) -> np.ndarray:
    """LUT for raster values in [0, 1].

    Grey defaults to inverted (0=white, 1=black). ``revert`` flips 0↔1 for
    any LUT, so grey+revert is conventional black-to-white.
    """
    lut = make_lut(name)
    invert = (name.lower() in ("grey", "gray")) != bool(revert)
    if invert:
        return lut[::-1].copy()
    return lut


def selected_row_lut(revert: bool) -> np.ndarray:
    """Highlight LUT: 0=red, 1=black (flipped when ``revert`` is on)."""
    t = np.linspace(0.0, 1.0, 256, dtype=np.float64)[:, None]
    red = np.array([255.0, 0.0, 0.0], dtype=np.float64)
    black = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    if not revert:
        colors = (1.0 - t) * red + t * black
    else:
        colors = (1.0 - t) * black + t * red
    return np.clip(colors, 0, 255).astype(np.uint8)


def colorize_raster(
    matrix: np.ndarray,
    lut: np.ndarray,
    *,
    highlight_row: int | None = None,
    highlight_lut: np.ndarray | None = None,
) -> np.ndarray:
    """Map (n_roi, nframes) unit values to RGB; NaNs → mid-grey."""
    img = np.asarray(matrix, dtype=np.float64)
    if img.ndim != 2:
        raise ValueError("raster matrix must be 2D")
    finite = np.isfinite(img)
    scaled = np.clip(np.nan_to_num(img, nan=0.0), 0.0, 1.0)
    idx = (scaled * 255.0).astype(np.uint8)
    rgb = lut[idx]
    if (
        highlight_row is not None
        and highlight_lut is not None
        and 0 <= int(highlight_row) < rgb.shape[0]
    ):
        r = int(highlight_row)
        rgb[r] = highlight_lut[idx[r]]
    rgb[~finite] = np.asarray(RASTER_NAN_RGB, dtype=np.uint8)
    return rgb


def apply_lut(
    image: np.ndarray, lut: np.ndarray, vmin: float, vmax: float
) -> np.ndarray:
    """Map 2D image through LUT → RGB uint8. Non-finite pixels → mid-grey."""
    img = np.asarray(image, dtype=np.float64)
    if vmax <= vmin:
        vmax = vmin + 1.0
    finite = np.isfinite(img)
    norm = np.clip((np.nan_to_num(img, nan=vmin) - vmin) / (vmax - vmin), 0.0, 1.0)
    idx = (norm * 255.0).astype(np.uint8)
    rgb = lut[idx]
    rgb[~finite] = np.asarray((128, 128, 128), dtype=np.uint8)
    return rgb
