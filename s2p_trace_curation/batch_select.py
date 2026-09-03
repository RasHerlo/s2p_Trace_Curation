"""Batch ROI selection via closed lasso on the FOV."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from s2p_trace_curation.gui.overlays import OverlayFilter, iter_visible_rois


def point_in_polygon(y: float, x: float, poly_yx: np.ndarray) -> bool:
    """Ray-casting test. ``poly_yx`` is (N, 2) with columns (y, x)."""
    poly = np.asarray(poly_yx, dtype=np.float64)
    if poly.shape[0] < 3:
        return False
    # Close polygon if needed
    if not np.allclose(poly[0], poly[-1]):
        poly = np.vstack([poly, poly[0]])
    inside = False
    j = poly.shape[0] - 1
    for i in range(poly.shape[0]):
        yi, xi = poly[i, 0], poly[i, 1]
        yj, xj = poly[j, 0], poly[j, 1]
        if ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / (yj - yi + 1e-30) + xi
        ):
            inside = not inside
        j = i
    return inside


def roi_fraction_inside(ypix: np.ndarray, xpix: np.ndarray, poly_yx: np.ndarray) -> float:
    ypix = np.asarray(ypix, dtype=np.float64)
    xpix = np.asarray(xpix, dtype=np.float64)
    if ypix.size == 0:
        return 0.0
    n_in = 0
    for y, x in zip(ypix, xpix):
        if point_in_polygon(float(y), float(x), poly_yx):
            n_in += 1
    return n_in / float(ypix.size)


def rois_in_lasso(
    rois: list[dict[str, Any]],
    poly_yx: Sequence[tuple[float, float]] | np.ndarray,
    overlay_filter: OverlayFilter,
    *,
    min_fraction: float = 0.5,
    active_roi_id: int | None = None,
) -> list[int]:
    """
    Return roi_ids (visible under filter) with > ``min_fraction`` of pixels inside
    the closed lasso. Default threshold is strictly more than 50%.
    """
    poly = np.asarray(poly_yx, dtype=np.float64)
    if poly.ndim != 2 or poly.shape[0] < 3 or poly.shape[1] != 2:
        return []
    # Use > 0.5 as agreed ("more than 50%")
    thr = float(min_fraction)
    selected: list[int] = []
    for row in iter_visible_rois(rois, overlay_filter, active_roi_id):
        frac = roi_fraction_inside(row["roi"]["ypix"], row["roi"]["xpix"], poly)
        if frac > thr:
            selected.append(int(row["roi_id"]))
    selected.sort()
    return selected


def mean_traces_for_rois(
    rois: list[dict[str, Any]], roi_ids: list[int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Mean F, mean Fneu, and display compensation with x=1:
    mean_F - 1.0 * mean_Fneu.
    Does not modify per-ROI stored ``x`` values.
    """
    id_set = set(int(i) for i in roi_ids)
    Fs: list[np.ndarray] = []
    Fns: list[np.ndarray] = []
    for row in rois:
        if int(row["roi_id"]) not in id_set:
            continue
        Fs.append(np.asarray(row["roi"]["F"], dtype=np.float64))
        Fns.append(np.asarray(row["neuropil"]["Fneu"], dtype=np.float64))
    if not Fs:
        empty = np.zeros(0, dtype=np.float64)
        return empty, empty, empty
    F = np.mean(np.stack(Fs, axis=0), axis=0)
    Fneu = np.mean(np.stack(Fns, axis=0), axis=0)
    comp = F - 1.0 * Fneu
    return F, Fneu, comp
