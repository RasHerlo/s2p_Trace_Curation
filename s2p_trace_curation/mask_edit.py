"""Brush edits for F / Fneu masks (mutually exclusive, never empty)."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

MaskEditMode = Literal["add_f", "remove_f", "add_fneu", "remove_fneu"]

MODE_LABELS: dict[MaskEditMode, str] = {
    "add_f": "Add F-ROI pixels",
    "remove_f": "Remove F-ROI pixels",
    "add_fneu": "Add Fneu-ROI pixels",
    "remove_fneu": "Remove Fneu-ROI pixels",
}


class ExtractCancelled(Exception):
    """Raised when the user cancels a long-running re-extract."""


def brush_offsets(radius: int) -> tuple[np.ndarray, np.ndarray]:
    r = max(0, int(radius))
    ys, xs = [], []
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if dy * dy + dx * dx <= r * r:
                ys.append(dy)
                xs.append(dx)
    return np.asarray(ys, dtype=np.int64), np.asarray(xs, dtype=np.int64)


def brush_pixels(
    cy: int, cx: int, radius: int, Ly: int, Lx: int
) -> tuple[np.ndarray, np.ndarray]:
    dy, dx = brush_offsets(radius)
    y = cy + dy
    x = cx + dx
    m = (y >= 0) & (y < Ly) & (x >= 0) & (x < Lx)
    return y[m], x[m]


def _lam_for_new_pixels(row: dict[str, Any], n_new: int) -> np.ndarray:
    lam = np.asarray(row["roi"]["lam"], dtype=np.float32)
    if lam.size == 0:
        val = np.float32(1.0)
    else:
        val = np.float32(float(np.mean(lam)))
    return np.full(n_new, val, dtype=np.float32)


def _pack_roi(
    row: dict[str, Any],
    coords: list[tuple[int, int]],
    lams: list[float] | np.ndarray,
) -> None:
    if not coords:
        raise ValueError("F-ROI cannot be empty")
    y = np.asarray([c[0] for c in coords], dtype=np.int32)
    x = np.asarray([c[1] for c in coords], dtype=np.int32)
    lam = np.asarray(lams, dtype=np.float32)
    if lam.shape[0] != y.shape[0]:
        raise ValueError("lam length mismatch")
    row["roi"]["ypix"] = y
    row["roi"]["xpix"] = x
    row["roi"]["lam"] = lam


def _pack_neu(row: dict[str, Any], ipix: np.ndarray | list[int]) -> None:
    ipix = np.asarray(ipix, dtype=np.int32).reshape(-1)
    if ipix.size == 0:
        raise ValueError("Fneu-ROI cannot be empty")
    row["neuropil"]["ipix"] = ipix


def apply_brush(
    row: dict[str, Any],
    mode: MaskEditMode,
    cy: int,
    cx: int,
    radius: int,
    Ly: int,
    Lx: int,
) -> tuple[bool, str]:
    """
    Apply one brush stamp to the ROI row.
    F and Fneu are mutually exclusive. Neither may become empty.
    New F pixels get lam = mean(existing lam) (or 1.0 if none).
    Returns (changed, status_message).
    """
    by, bx = brush_pixels(cy, cx, radius, Ly, Lx)
    if by.size == 0:
        return False, ""

    ypix = np.asarray(row["roi"]["ypix"], dtype=np.int64)
    xpix = np.asarray(row["roi"]["xpix"], dtype=np.int64)
    lam = np.asarray(row["roi"]["lam"], dtype=np.float32)
    roi_map: dict[tuple[int, int], float] = {
        (int(y), int(x)): float(w) for y, x, w in zip(ypix, xpix, lam)
    }

    ipix = np.asarray(row["neuropil"]["ipix"], dtype=np.int64)
    neu_set: set[int] = set(int(i) for i in ipix.tolist())

    brush = [(int(y), int(x)) for y, x in zip(by.tolist(), bx.tolist())]
    brush_lin = [y * Lx + x for y, x in brush]

    if mode == "add_f":
        to_add = [p for p in brush if p not in roi_map]
        if not to_add:
            return False, ""
        steal = [y * Lx + x for y, x in to_add if (y * Lx + x) in neu_set]
        # Allow building F from empty when Fneu is still empty (new ROI).
        if neu_set and len(neu_set) - len(set(steal)) < 1:
            return False, "Cannot add to F: would empty Fneu-ROI"
        for y, x in to_add:
            neu_set.discard(y * Lx + x)
        new_lam = _lam_for_new_pixels(row, len(to_add))
        for (y, x), w in zip(to_add, new_lam.tolist()):
            roi_map[(y, x)] = float(w)
        _pack_roi(row, list(roi_map.keys()), [roi_map[k] for k in roi_map])
        if neu_set:
            _pack_neu(row, sorted(neu_set))
        else:
            row["neuropil"]["ipix"] = np.zeros(0, dtype=np.int32)
        return True, f"Added {len(to_add)} F pixel(s)"

    if mode == "remove_f":
        to_rm = [p for p in brush if p in roi_map]
        if not to_rm:
            return False, ""
        if len(roi_map) - len(to_rm) < 1:
            return False, "Cannot remove: F-ROI would be empty"
        for p in to_rm:
            del roi_map[p]
        _pack_roi(row, list(roi_map.keys()), [roi_map[k] for k in roi_map])
        return True, f"Removed {len(to_rm)} F pixel(s)"

    if mode == "add_fneu":
        to_add_lin = [i for i in brush_lin if i not in neu_set]
        if not to_add_lin:
            return False, ""
        steal_coords = [(i // Lx, i % Lx) for i in to_add_lin if (i // Lx, i % Lx) in roi_map]
        # Allow building Fneu from empty when F is still empty (new ROI).
        if roi_map and len(roi_map) - len(steal_coords) < 1:
            return False, "Cannot add to Fneu: would empty F-ROI"
        for y, x in steal_coords:
            del roi_map[(y, x)]
        for i in to_add_lin:
            neu_set.add(i)
        if roi_map:
            _pack_roi(row, list(roi_map.keys()), [roi_map[k] for k in roi_map])
        else:
            row["roi"]["ypix"] = np.zeros(0, dtype=np.int32)
            row["roi"]["xpix"] = np.zeros(0, dtype=np.int32)
            row["roi"]["lam"] = np.zeros(0, dtype=np.float32)
        _pack_neu(row, sorted(neu_set))
        return True, f"Added {len(to_add_lin)} Fneu pixel(s)"

    if mode == "remove_fneu":
        to_rm = [i for i in brush_lin if i in neu_set]
        if not to_rm:
            return False, ""
        if len(neu_set) - len(to_rm) < 1:
            return False, "Cannot remove: Fneu-ROI would be empty"
        for i in to_rm:
            neu_set.discard(i)
        _pack_neu(row, sorted(neu_set))
        return True, f"Removed {len(to_rm)} Fneu pixel(s)"

    return False, f"Unknown mode: {mode}"
