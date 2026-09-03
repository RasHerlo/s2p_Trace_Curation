"""Normalized traces (tc_norm) and raster-row helpers."""

from __future__ import annotations

from typing import Any

import numpy as np

from s2p_trace_curation.annotations import (
    ensure_annotations,
    nan_mask_from_annotations,
)
from s2p_trace_curation.gui.overlays import OverlayFilter, roi_passes_overlay


def led_shutter_ann_ids(doc: dict[str, Any]) -> list[int]:
    return [
        int(ann["ann_id"])
        for ann in ensure_annotations(doc)
        if str(ann["property"]) == "LED+Shutter"
    ]


def led_shutter_nan_mask(doc: dict[str, Any], nframes: int) -> np.ndarray:
    anns = ensure_annotations(doc)
    return nan_mask_from_annotations(int(nframes), anns, led_shutter_ann_ids(doc))


def compute_tc_norm(trace_comp: np.ndarray, nan_mask: np.ndarray) -> np.ndarray:
    """Per-trace min–max to [0, 1]; LED+Shutter samples stay NaN; constants → 0."""
    t = np.asarray(trace_comp, dtype=np.float64).copy()
    mask = np.asarray(nan_mask, dtype=bool)
    if mask.shape[0] != t.shape[0]:
        raise ValueError("nan_mask length must match trace_comp")
    if mask.any():
        t[mask] = np.nan
    finite = np.isfinite(t)
    out = np.full(t.shape, np.nan, dtype=np.float64)
    if not finite.any():
        return out
    lo = float(np.min(t[finite]))
    hi = float(np.max(t[finite]))
    if hi <= lo:
        out[finite] = 0.0
        return out
    out[finite] = (t[finite] - lo) / (hi - lo)
    return out


def tc_norm_sig(doc: dict[str, Any]) -> dict[str, Any]:
    """Fingerprint of inputs used to build tc_norm (LED spans + trace_comp sums)."""
    led = sorted(
        [int(a["start_frame"]), int(a["end_frame"])]
        for a in ensure_annotations(doc)
        if str(a["property"]) == "LED+Shutter"
    )
    ids: list[int] = []
    sums: list[float] = []
    for row in doc["rois"]:
        ids.append(int(row["roi_id"]))
        tc = np.asarray(row["compensation"]["trace_comp"], dtype=np.float64)
        sums.append(float(np.nansum(tc)))
    return {"led": led, "ids": ids, "sums": sums}


def _sigs_equal(a: Any, b: Any) -> bool:
    if not isinstance(a, dict) or not isinstance(b, dict):
        return False
    led_a = [list(x) for x in a.get("led") or []]
    led_b = [list(x) for x in b.get("led") or []]
    if led_a != led_b:
        return False
    if list(a.get("ids") or []) != list(b.get("ids") or []):
        return False
    sa = np.asarray(a.get("sums") or [], dtype=np.float64)
    sb = np.asarray(b.get("sums") or [], dtype=np.float64)
    if sa.shape != sb.shape:
        return False
    return bool(np.allclose(sa, sb, rtol=0.0, atol=1e-6, equal_nan=True))


def rois_missing_tc_norm(doc: dict[str, Any]) -> list[dict[str, Any]]:
    return [row for row in doc["rois"] if row.get("tc_norm") is None]


def tc_norm_is_stale(doc: dict[str, Any]) -> bool:
    """True when stored tc_norm exists but no longer matches current inputs."""
    rois = doc.get("rois") or []
    if not rois:
        return False
    has = [row.get("tc_norm") is not None for row in rois]
    if not any(has):
        return False
    if not all(has):
        return True
    stored = (doc.get("meta") or {}).get("tc_norm_sig")
    return not _sigs_equal(stored, tc_norm_sig(doc))


def fill_missing_tc_norm(doc: dict[str, Any]) -> int:
    """Compute tc_norm only for ROIs that lack it. Returns how many were filled."""
    missing = rois_missing_tc_norm(doc)
    if not missing:
        return 0
    nframes = int(doc["meta"]["nframes"])
    mask = led_shutter_nan_mask(doc, nframes)
    for row in missing:
        row["tc_norm"] = compute_tc_norm(row["compensation"]["trace_comp"], mask)
    return len(missing)


def rebuild_all_tc_norm(doc: dict[str, Any]) -> None:
    """Recompute tc_norm for every ROI and store the input fingerprint."""
    nframes = int(doc["meta"]["nframes"])
    mask = led_shutter_nan_mask(doc, nframes)
    for row in doc["rois"]:
        row["tc_norm"] = compute_tc_norm(row["compensation"]["trace_comp"], mask)
    doc.setdefault("meta", {})["tc_norm_sig"] = tc_norm_sig(doc)


def rois_for_raster(
    rois: list[dict[str, Any]],
    overlay_filter: OverlayFilter,
    active_roi_id: int | None = None,
) -> list[dict[str, Any]]:
    """Visible ROIs for the raster, in pickle (doc / roi_id) order."""
    out: list[dict[str, Any]] = []
    for row in rois:
        if roi_passes_overlay(row, overlay_filter, active_roi_id):
            out.append(row)
    return out


def stack_tc_norm(rows: list[dict[str, Any]], nframes: int) -> np.ndarray:
    """Stack stored tc_norm rows as (n_roi, nframes). Missing → NaN row."""
    n = len(rows)
    out = np.full((n, int(nframes)), np.nan, dtype=np.float64)
    for i, row in enumerate(rows):
        tr = row.get("tc_norm")
        if tr is None:
            continue
        arr = np.asarray(tr, dtype=np.float64)
        n_copy = min(arr.shape[0], out.shape[1])
        out[i, :n_copy] = arr[:n_copy]
    return out
