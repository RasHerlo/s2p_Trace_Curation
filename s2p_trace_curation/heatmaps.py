"""Named FOV heatmaps computed from data.bin (independent of ROI traces).

A heatmap is defined by frame ranges plus a metric kind. The first kind is
``auc_ratio``: per pixel, the span-normalized AUC inside the ranges divided by
the span-normalized AUC outside them. LED+Shutter frames are excluded from both
sides. Ranges are frame intervals, so maps never go stale from ROI edits.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

from s2p_trace_curation.suite2p_io import BinaryStack, plane_dir, resolve_suite2p_dir

KIND_AUC_RATIO = "auc_ratio"
HEATMAP_KINDS = (KIND_AUC_RATIO,)
HEATMAP_KIND_LABELS = {
    KIND_AUC_RATIO: "AUC ratio (inside / outside)",
}

# Frames per streaming block, sized to ~32 MB of raw movie data.
BLOCK_BYTES = 32_000_000


class HeatmapCancelled(Exception):
    """User cancelled heatmap computation."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def default_heatmap_params() -> dict[str, Any]:
    return {"kind": KIND_AUC_RATIO, "ranges": []}


def kind_label(kind: str) -> str:
    return HEATMAP_KIND_LABELS.get(str(kind), str(kind))


def ensure_heatmaps(doc: dict[str, Any]) -> list[dict[str, Any]]:
    maps = doc.get("heatmaps")
    if maps is None:
        maps = []
        doc["heatmaps"] = maps
    return maps


def next_heatmap_id(doc: dict[str, Any]) -> str:
    n = 0
    for hm in ensure_heatmaps(doc):
        s = str(hm.get("id", ""))
        if s.startswith("h-"):
            try:
                n = max(n, int(s[2:]))
            except ValueError:
                pass
    return f"h-{n + 1:03d}"


def get_heatmap(doc: dict[str, Any], heatmap_id: str) -> dict[str, Any] | None:
    hid = str(heatmap_id)
    for hm in ensure_heatmaps(doc):
        if str(hm.get("id")) == hid:
            return hm
    return None


def heatmap_combo_label(hm: dict[str, Any]) -> str:
    name = str(hm.get("label") or hm.get("id") or "Untitled")
    return f"HeatMap: {name}"


def heatmap_combo_data(hm: dict[str, Any]) -> str:
    return f"heatmap:{hm['id']}"


def parse_heatmap_combo_data(data: Any) -> str | None:
    text = str(data or "")
    if text.startswith("heatmap:"):
        return text.split(":", 1)[1]
    return None


def normalize_ranges(
    ranges: Any, nframes: int | None = None
) -> list[list[int]]:
    """Sorted, merged, inclusive [start, end] pairs clipped to the movie."""
    out: list[list[int]] = []
    for item in ranges or []:
        try:
            a, b = int(item[0]), int(item[1])
        except (TypeError, ValueError, IndexError):
            continue
        if b < a:
            a, b = b, a
        if nframes is not None:
            last = max(int(nframes) - 1, 0)
            a = max(0, min(a, last))
            b = max(0, min(b, last))
        out.append([a, b])
    out.sort(key=lambda p: (p[0], p[1]))
    merged: list[list[int]] = []
    for a, b in out:
        if merged and a <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return merged


def normalize_heatmap_params(
    params: dict[str, Any] | None, nframes: int | None = None
) -> dict[str, Any]:
    raw = dict(params or {})
    kind = str(raw.get("kind") or KIND_AUC_RATIO)
    if kind not in HEATMAP_KINDS:
        kind = KIND_AUC_RATIO
    return {"kind": kind, "ranges": normalize_ranges(raw.get("ranges"), nframes)}


def format_ranges(ranges: list[list[int]]) -> str:
    if not ranges:
        return "no ranges"
    return ", ".join(f"{int(a)}–{int(b)}" for a, b in ranges)


def ranges_to_mask(ranges: list[list[int]], nframes: int) -> np.ndarray:
    mask = np.zeros(int(nframes), dtype=bool)
    for a, b in normalize_ranges(ranges, nframes):
        mask[int(a) : int(b) + 1] = True
    return mask


def trapezoid_weights(mask: np.ndarray) -> np.ndarray:
    """Per-frame trapezoid weights over each contiguous run of ``mask``.

    Interior samples weigh 1, run endpoints 0.5, so the weighted sum is the
    trapezoid AUC and the weight total is the integrated span. A lone sample
    keeps weight 1 so single-frame ranges stay usable.
    """
    m = np.asarray(mask, dtype=bool)
    w = np.zeros(m.shape[0], dtype=np.float64)
    if not m.any():
        return w
    padded = np.concatenate(([False], m, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    for start, stop in zip(edges[0::2], edges[1::2]):
        n = int(stop - start)
        if n == 1:
            w[start] = 1.0
            continue
        w[start:stop] = 1.0
        w[start] = 0.5
        w[stop - 1] = 0.5
    return w


def split_frame_weights(
    ranges: list[list[int]], nan_mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Trapezoid weights for frames inside vs outside the ranges.

    Frames masked by LED+Shutter are dropped from both sides.
    """
    excluded = np.asarray(nan_mask, dtype=bool)
    nframes = excluded.shape[0]
    inside = ranges_to_mask(ranges, nframes) & ~excluded
    outside = ~ranges_to_mask(ranges, nframes) & ~excluded
    return trapezoid_weights(inside), trapezoid_weights(outside)


def compute_heatmap_map(
    suite2p_dir: Path,
    params: dict[str, Any],
    nan_mask: np.ndarray,
    *,
    progress: Callable[[str, float], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> np.ndarray:
    """Per-pixel metric map from data.bin. One streaming pass over the movie."""
    plane = plane_dir(resolve_suite2p_dir(suite2p_dir))
    with BinaryStack(plane) as stack:
        nframes = int(stack.nframes)
        height = int(stack.Ly)
        width = int(stack.Lx)
        path = stack.path
        dtype = stack.dtype

    mask = np.asarray(nan_mask, dtype=bool)
    if mask.shape[0] != nframes:
        raise ValueError("nan_mask length must match nframes")

    params = normalize_heatmap_params(params, nframes)
    if params["kind"] != KIND_AUC_RATIO:
        raise ValueError(f"Unknown heatmap kind: {params['kind']}")
    if not params["ranges"]:
        raise ValueError("Set at least one range before computing")

    w_in, w_out = split_frame_weights(params["ranges"], mask)
    span_in = float(w_in.sum())
    span_out = float(w_out.sum())
    if span_in <= 0:
        raise ValueError("Ranges contain no usable frames (all shutter?)")
    if span_out <= 0:
        raise ValueError("No frames left outside the ranges to compare against")

    def report(stage: str, fraction: float) -> None:
        if progress is not None:
            progress(stage, float(fraction))
        if should_cancel is not None and should_cancel():
            raise HeatmapCancelled()

    n_pixels = height * width
    sum_in = np.zeros(n_pixels, dtype=np.float64)
    sum_out = np.zeros(n_pixels, dtype=np.float64)
    block = max(1, int(BLOCK_BYTES // max(n_pixels * dtype.itemsize, 1)))

    report("Reading movie", 0.0)
    mmap = np.memmap(path, dtype=dtype, mode="r", shape=(nframes, n_pixels))
    try:
        for t0 in range(0, nframes, block):
            t1 = min(t0 + block, nframes)
            wi = w_in[t0:t1]
            wo = w_out[t0:t1]
            if wi.any() or wo.any():
                frames = np.asarray(mmap[t0:t1], dtype=np.float64)
                if wi.any():
                    sum_in += wi @ frames
                if wo.any():
                    sum_out += wo @ frames
                del frames
            report("Reading movie", 0.02 + 0.96 * (t1 / nframes))
    finally:
        del mmap

    mean_in = sum_in / span_in
    mean_out = sum_out / span_out
    out = np.full(n_pixels, np.nan, dtype=np.float64)
    valid = np.isfinite(mean_out) & (mean_out > 0)
    out[valid] = mean_in[valid] / mean_out[valid]
    report("Heatmap ready", 1.0)
    return np.asarray(out, dtype=np.float32).reshape(height, width)


def make_heatmap(
    doc: dict[str, Any],
    *,
    label: str,
    params: dict[str, Any],
    heatmap_map: np.ndarray,
    heatmap_id: str | None = None,
) -> dict[str, Any]:
    now = _utc_now()
    nframes = int((doc.get("meta") or {}).get("nframes") or 0) or None
    return {
        "id": heatmap_id or next_heatmap_id(doc),
        "label": str(label).strip() or "Untitled",
        "params": deepcopy(normalize_heatmap_params(params, nframes)),
        "map": np.asarray(heatmap_map, dtype=np.float32),
        "created_utc": now,
        "updated_utc": now,
    }


def apply_heatmap_result(
    hm: dict[str, Any],
    *,
    label: str | None = None,
    params: dict[str, Any] | None = None,
    heatmap_map: np.ndarray | None = None,
    nframes: int | None = None,
) -> None:
    if label is not None:
        hm["label"] = str(label).strip() or "Untitled"
    if params is not None:
        hm["params"] = deepcopy(normalize_heatmap_params(params, nframes))
    if heatmap_map is not None:
        hm["map"] = np.asarray(heatmap_map, dtype=np.float32)
    hm["updated_utc"] = _utc_now()
