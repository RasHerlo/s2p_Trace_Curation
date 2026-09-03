"""Background measurement ROIs — stored apart from cell ``rois``.

BG ROIs never enter the raster, HAC, heatmaps, or iscell/batch flows.
Traces are an unweighted mean over painted pixels. Savitzky–Golay and
bleach use the same session parameters as cell traces (LED+Shutter
excised) but are **not** min–max normalized, so amplitude stays usable
for motion thresholding.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from s2p_trace_curation.suite2p_io import BinaryStack, plane_dir, resolve_suite2p_dir
from s2p_trace_curation.trace_processing import (
    TAU_SHARED,
    apply_savgol,
    biexponential_decay,
    conservative_fit_params,
    ensure_trace_processing,
    fit_amplitudes_frozen_tau,
    fit_biexponential_params,
    illumination_keep,
    scatter_to_full,
)

BG_FIELD_F = "F"
BG_FIELD_SM = "F_sm"
BG_FIELD_SM_BC = "F_sm_bc"
BG_TRACE_FIELDS = (BG_FIELD_F, BG_FIELD_SM, BG_FIELD_SM_BC)
BG_TRACE_LABELS = {
    BG_FIELD_F: "BG-ROI",
    BG_FIELD_SM: "BG-ROI sm",
    BG_FIELD_SM_BC: "BG-ROI sm_bc",
}

# W1 / W3 fill — distinct from cell red / active cyan.
BG_ROI_RGB = (46, 204, 113)
BG_ROI_RGB_DRAFT = (88, 255, 160)


def ensure_bg_rois(doc: dict[str, Any]) -> list[dict[str, Any]]:
    """Guarantee ``doc['bg_rois']`` exists and return it."""
    rows = doc.get("bg_rois")
    if rows is None:
        rows = []
        doc["bg_rois"] = rows
    return rows


def next_bg_id(doc: dict[str, Any]) -> int:
    rows = ensure_bg_rois(doc)
    if not rows:
        return 0
    return max(int(r["bg_id"]) for r in rows) + 1


def empty_bg_paint_draft(nframes: int) -> dict[str, Any]:
    """ROI-shaped draft so Add Mask brush tools can paint a BG ROI."""
    t = int(nframes)
    z = np.zeros(t, dtype=np.float64)
    return {
        "roi_id": -1,
        "iscell": False,
        "iscell_prob": None,
        "roi": {
            "ypix": np.zeros(0, dtype=np.int32),
            "xpix": np.zeros(0, dtype=np.int32),
            "lam": np.zeros(0, dtype=np.float32),
            "F": z.copy(),
            "modified": True,
        },
        "neuropil": {
            "ipix": np.zeros(0, dtype=np.int32),
            "Fneu": z.copy(),
            "modified": True,
        },
        "compensation": {"x": 1.0, "fneu_offset": 0.0, "trace_comp": z.copy()},
    }


def bg_pixels(entry: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """``(ypix, xpix)`` from a saved BG ROI or a paint draft."""
    if "roi" in entry and "ypix" not in entry:
        return (
            np.asarray(entry["roi"]["ypix"], dtype=np.int32),
            np.asarray(entry["roi"]["xpix"], dtype=np.int32),
        )
    return (
        np.asarray(entry["ypix"], dtype=np.int32),
        np.asarray(entry["xpix"], dtype=np.int32),
    )


def bg_roi_from_draft(draft: dict[str, Any], bg_id: int) -> dict[str, Any]:
    ypix, xpix = bg_pixels(draft)
    f = np.asarray(draft["roi"]["F"], dtype=np.float64)
    return {
        "bg_id": int(bg_id),
        "ypix": ypix.astype(np.int32, copy=False),
        "xpix": xpix.astype(np.int32, copy=False),
        "F": f,
        "F_sm": draft.get(BG_FIELD_SM),
        "F_sm_bc": draft.get(BG_FIELD_SM_BC),
        "bleach": draft.get("bleach"),
        "modified": True,
    }


def append_bg_roi(doc: dict[str, Any], row: dict[str, Any]) -> None:
    ensure_bg_rois(doc).append(row)
    doc["bg_rois"].sort(key=lambda r: int(r["bg_id"]))


def get_bg_roi(doc: dict[str, Any], bg_id: int) -> dict[str, Any] | None:
    want = int(bg_id)
    for row in ensure_bg_rois(doc):
        if int(row["bg_id"]) == want:
            return row
    return None


def bg_roi_label(row: dict[str, Any]) -> str:
    n = int(len(row.get("ypix") if "ypix" in row else row["roi"]["ypix"]))
    bid = row.get("bg_id", "?")
    return f"BG {bid}  ({n} px)"


def bg_trace(row: dict[str, Any], field: str, nframes: int) -> np.ndarray:
    """One BG series, padded/truncated to ``nframes``. Missing → all-NaN."""
    if field == BG_FIELD_F:
        src = row.get("F")
        if src is None and "roi" in row:
            src = row["roi"].get("F")
    else:
        src = row.get(field)
    out = np.full(int(nframes), np.nan, dtype=np.float64)
    if src is None:
        return out
    arr = np.asarray(src, dtype=np.float64)
    n = min(arr.shape[0], out.shape[0])
    out[:n] = arr[:n]
    return out


def sum_bg_traces(
    rows: list[dict[str, Any]], field: str, nframes: int
) -> np.ndarray:
    """Element-wise sum of selected BG traces (NaN if every addend is NaN)."""
    if not rows:
        return np.full(int(nframes), np.nan, dtype=np.float64)
    stacked = np.vstack([bg_trace(r, field, nframes) for r in rows])
    with np.errstate(all="ignore"):
        return np.nansum(stacked, axis=0)


def compute_bg_sm(
    trace: np.ndarray, nan_mask: np.ndarray, sg_window: int, sg_poly: int
) -> np.ndarray:
    """Savitzky–Golay on illumination frames; LED spans stay NaN. No min–max."""
    t = np.asarray(trace, dtype=np.float64)
    keep = illumination_keep(t, nan_mask)
    if int(np.count_nonzero(keep)) < 3:
        return np.full(t.shape, np.nan, dtype=np.float64)
    return scatter_to_full(apply_savgol(t[keep], sg_window, sg_poly), keep, t.shape[0])


def compute_bg_sm_bc(
    sm: np.ndarray,
    nan_mask: np.ndarray,
    params: tuple[float, float, float, float, float],
) -> np.ndarray:
    """Subtract the bleach fit on illumination frames. No min–max."""
    arr = np.asarray(sm, dtype=np.float64)
    keep = illumination_keep(arr, nan_mask)
    y = arr[keep]
    if y.size == 0:
        return np.full(arr.shape, np.nan, dtype=np.float64)
    t = np.arange(y.size, dtype=np.float64)
    return scatter_to_full(y - biexponential_decay(t, *params), keep, arr.shape[0])


def _fit_bg_bleach(
    sm: np.ndarray,
    nan_mask: np.ndarray,
    *,
    conservative: bool,
    tau1: float | None,
    tau2: float | None,
) -> tuple[tuple[float, float, float, float, float], bool]:
    keep = illumination_keep(sm, nan_mask)
    y = sm[keep]
    if conservative or y.size < 6:
        return conservative_fit_params(y if y.size else sm), True
    params = None
    if tau1 is not None and tau2 is not None:
        params = fit_amplitudes_frozen_tau(y, tau1, tau2)
    if params is None:
        params = fit_biexponential_params(y)
    if params is None:
        return conservative_fit_params(y), True
    return params, False


def process_bg_trace(doc: dict[str, Any], f: np.ndarray) -> dict[str, Any]:
    """Return ``F_sm``, ``F_sm_bc``, and bleach metadata for one raw F."""
    from s2p_trace_curation.raster import led_shutter_nan_mask

    tp = ensure_trace_processing(doc)
    nframes = int(doc["meta"]["nframes"])
    mask = led_shutter_nan_mask(doc, nframes)
    sm = compute_bg_sm(f, mask, int(tp["sg_window"]), int(tp["sg_poly"]))
    enabled = bool(tp["bleach_enabled"])
    tau_mode = str(tp.get("tau_mode") or TAU_SHARED)
    tau1 = tau2 = None
    if enabled and tau_mode == TAU_SHARED:
        if tp.get("shared_tau1") is not None and tp.get("shared_tau2") is not None:
            tau1, tau2 = float(tp["shared_tau1"]), float(tp["shared_tau2"])
    params, cons = _fit_bg_bleach(
        sm,
        mask,
        conservative=not enabled,
        tau1=tau1,
        tau2=tau2,
    )
    return {
        BG_FIELD_SM: sm,
        BG_FIELD_SM_BC: compute_bg_sm_bc(sm, mask, params),
        "bleach": {"fit_params": list(params), "conservative": bool(cons)},
    }


def apply_processed_to_bg(doc: dict[str, Any], row: dict[str, Any]) -> None:
    """Fill ``F_sm`` / ``F_sm_bc`` on a saved BG ROI or a paint draft."""
    if "roi" in row and "F" not in row:
        f = np.asarray(row["roi"]["F"], dtype=np.float64)
    else:
        f = np.asarray(row["F"], dtype=np.float64)
    extra = process_bg_trace(doc, f)
    row[BG_FIELD_SM] = extra[BG_FIELD_SM]
    row[BG_FIELD_SM_BC] = extra[BG_FIELD_SM_BC]
    row["bleach"] = extra["bleach"]


def rebuild_all_bg_processed(doc: dict[str, Any]) -> None:
    """Re-run SG + bleach on every BG ROI. No-op when the list is empty."""
    for row in ensure_bg_rois(doc):
        apply_processed_to_bg(doc, row)


def reextract_bg_draft(
    draft: dict[str, Any],
    suite2p_dir: Path,
    *,
    progress: Any = None,
    should_cancel: Any = None,
) -> None:
    """Unweighted mean of the painted pixels → ``draft['roi']['F']``."""
    plane = plane_dir(resolve_suite2p_dir(suite2p_dir))
    ypix, xpix = bg_pixels(draft)
    with BinaryStack(plane) as stack:
        draft["roi"]["F"] = stack.extract_unweighted_trace(
            ypix,
            xpix,
            progress=progress,
            should_cancel=should_cancel,
        )
    draft["roi"]["modified"] = True
    z = np.zeros_like(draft["roi"]["F"])
    draft["neuropil"]["Fneu"] = z
    draft["compensation"]["trace_comp"] = np.asarray(
        draft["roi"]["F"], dtype=np.float64
    ).copy()


def build_bg_overlay(
    Ly: int,
    Lx: int,
    bg_rois: list[dict[str, Any]],
    *,
    draft: dict[str, Any] | None = None,
    alpha: float = 0.40,
) -> np.ndarray:
    """RGBA overlay of saved BG ROIs (and an in-progress draft)."""
    overlay = np.zeros((Ly, Lx, 4), dtype=np.uint8)
    a = int(round(alpha * 255))
    entries: list[tuple[dict[str, Any], tuple[int, int, int]]] = [
        (row, BG_ROI_RGB) for row in bg_rois
    ]
    if draft is not None:
        entries.append((draft, BG_ROI_RGB_DRAFT))
    for entry, rgb in entries:
        y, x = bg_pixels(entry)
        if y.size == 0:
            continue
        overlay[y, x, 0] = rgb[0]
        overlay[y, x, 1] = rgb[1]
        overlay[y, x, 2] = rgb[2]
        overlay[y, x, 3] = a
    return overlay
