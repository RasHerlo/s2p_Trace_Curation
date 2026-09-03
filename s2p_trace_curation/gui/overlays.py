"""ROI / neuropil overlay helpers for image panels."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

OverlayFilter = Literal["none", "current", "noncell", "cell", "both"]


def roi_area(row: dict[str, Any]) -> int:
    return int(len(row["roi"]["ypix"]))


def roi_passes_overlay(
    row: dict[str, Any],
    overlay_filter: OverlayFilter,
    active_roi_id: int | None = None,
) -> bool:
    if overlay_filter == "none":
        return False
    if overlay_filter == "current":
        return active_roi_id is not None and int(row["roi_id"]) == int(active_roi_id)
    iscell = bool(row.get("iscell", True))
    if overlay_filter == "cell" and not iscell:
        return False
    if overlay_filter == "noncell" and iscell:
        return False
    return True


def iter_visible_rois(
    rois: list[dict[str, Any]],
    overlay_filter: OverlayFilter,
    active_roi_id: int | None = None,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rois:
        if roi_passes_overlay(row, overlay_filter, active_roi_id):
            out.append(row)
    # Large first so smallest ends on top when painted sequentially
    out.sort(key=roi_area, reverse=True)
    return out


def build_fov_overlay(
    Ly: int,
    Lx: int,
    rois: list[dict[str, Any]],
    active_roi_id: int,
    overlay_filter: OverlayFilter,
    alpha: float = 0.35,
    batch_roi_ids: set[int] | None = None,
    cluster_rgb: dict[int, tuple[int, int, int]] | None = None,
) -> np.ndarray:
    """RGBA uint8 overlay; non-active red, active/batch cyan.

    ``cluster_rgb`` (roi_id → RGB) overrides those fills for clustered ROIs
    while keeping the same alpha.
    """
    overlay = np.zeros((Ly, Lx, 4), dtype=np.uint8)
    a = int(round(alpha * 255))
    batch = batch_roi_ids or set()
    clustered = cluster_rgb or {}
    for row in iter_visible_rois(rois, overlay_filter, active_roi_id):
        y = np.asarray(row["roi"]["ypix"], dtype=np.int64)
        x = np.asarray(row["roi"]["xpix"], dtype=np.int64)
        if y.size == 0:
            continue
        rid = int(row["roi_id"])
        highlight = rid in batch or (not batch and rid == int(active_roi_id))
        if rid in clustered:
            r, g, b = clustered[rid]
        elif highlight:
            r, g, b = 0, 255, 255
        else:
            r, g, b = 255, 0, 0
        overlay[y, x, 0] = r
        overlay[y, x, 1] = g
        overlay[y, x, 2] = b
        overlay[y, x, 3] = a
    return overlay


def rois_at_pixel(
    rois: list[dict[str, Any]],
    y: int,
    x: int,
    overlay_filter: OverlayFilter,
    active_roi_id: int | None = None,
) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    for row in iter_visible_rois(rois, overlay_filter, active_roi_id):
        ypix = np.asarray(row["roi"]["ypix"], dtype=np.int64)
        xpix = np.asarray(row["roi"]["xpix"], dtype=np.int64)
        if np.any((ypix == y) & (xpix == x)):
            hits.append(row)
    hits.sort(key=roi_area)  # smallest first
    return hits


def thick_outline_mask(
    Ly: int, Lx: int, ypix: np.ndarray, xpix: np.ndarray, thickness: int = 2
) -> tuple[np.ndarray, np.ndarray]:
    """Return (y, x) coordinates of a thick outline around the ROI mask."""
    mask = np.zeros((Ly, Lx), dtype=bool)
    ypix = np.asarray(ypix, dtype=np.int64)
    xpix = np.asarray(xpix, dtype=np.int64)
    if ypix.size == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    mask[ypix, xpix] = True
    # binary erosion via neighbor AND
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    eroded = (
        padded[0:-2, 1:-1]
        & padded[2:, 1:-1]
        & padded[1:-1, 0:-2]
        & padded[1:-1, 2:]
        & padded[1:-1, 1:-1]
    )
    edge = mask & ~eroded
    if thickness > 1:
        yy, xx = np.nonzero(edge)
        thick = edge.copy()
        for dy in range(-thickness + 1, thickness):
            for dx in range(-thickness + 1, thickness):
                if dy == 0 and dx == 0:
                    continue
                y2 = yy + dy
                x2 = xx + dx
                valid = (y2 >= 0) & (y2 < Ly) & (x2 >= 0) & (x2 < Lx)
                thick[y2[valid], x2[valid]] = True
        edge = thick & ~mask | edge  # keep ring around / on boundary
        # Prefer ring mostly on boundary pixels and just outside
        edge = thick.copy()
        # Remove deep interior
        edge &= ~eroded
    ys, xs = np.nonzero(edge)
    return ys.astype(np.int64), xs.astype(np.int64)


def compose_rgb_with_overlay(rgb: np.ndarray, overlay_rgba: np.ndarray) -> np.ndarray:
    """Alpha-blend overlay onto RGB uint8 image."""
    base = rgb.astype(np.float64)
    ov = overlay_rgba.astype(np.float64)
    a = ov[..., 3:4] / 255.0
    out = base * (1.0 - a) + ov[..., :3] * a
    return np.clip(out, 0, 255).astype(np.uint8)


def zoom_masks_rgba(
    frame_rgb: np.ndarray,
    y0: int,
    x0: int,
    side: int,
    row: dict[str, Any],
    Ly: int,
    Lx: int,
    roi_alpha: float = 0.35,
    neu_alpha: float = 0.35,
    show_roi: bool = True,
    show_neu: bool = True,
    roi_rgb: tuple[int, int, int] = (255, 0, 0),
) -> np.ndarray:
    """Crop RGB frame and blend neuropil (yellow-orange) + optional ROI fill."""
    crop = frame_rgb[y0 : y0 + side, x0 : x0 + side].copy()
    overlay = np.zeros((side, side, 4), dtype=np.uint8)

    # neuropil first (under)
    if show_neu:
        ipix = np.asarray(row["neuropil"]["ipix"], dtype=np.int64)
        if ipix.size:
            ny, nx = np.unravel_index(ipix, (Ly, Lx))
            cy = ny - y0
            cx = nx - x0
            m = (cy >= 0) & (cy < side) & (cx >= 0) & (cx < side)
            overlay[cy[m], cx[m], 0] = 255
            overlay[cy[m], cx[m], 1] = 180
            overlay[cy[m], cx[m], 2] = 0
            overlay[cy[m], cx[m], 3] = int(round(neu_alpha * 255))

    if show_roi:
        ypix = np.asarray(row["roi"]["ypix"], dtype=np.int64) - y0
        xpix = np.asarray(row["roi"]["xpix"], dtype=np.int64) - x0
        m = (ypix >= 0) & (ypix < side) & (xpix >= 0) & (xpix < side)
        overlay[ypix[m], xpix[m], 0] = roi_rgb[0]
        overlay[ypix[m], xpix[m], 1] = roi_rgb[1]
        overlay[ypix[m], xpix[m], 2] = roi_rgb[2]
        overlay[ypix[m], xpix[m], 3] = int(round(roi_alpha * 255))

    return compose_rgb_with_overlay(crop, overlay)
