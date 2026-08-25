"""Named FOV heatmaps computed from data.bin (independent of ROI traces)."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

from s2p_trace_curation.suite2p_io import BinaryStack, plane_dir, resolve_suite2p_dir
from s2p_trace_curation.trace_processing import apply_savgol

try:
    from scipy.integrate import trapezoid
except ImportError:  # pragma: no cover
    from scipy.integrate import trapz as trapezoid

DEFAULT_EXTENSION = 50
DEFAULT_BASELINE_FRACTION = 0.2
DEFAULT_SG_WINDOW = 11
DEFAULT_SG_POLY = 2
PIXEL_CHUNK = 4096


class HeatmapCancelled(Exception):
    """User cancelled heatmap computation."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def default_heatmap_params(nframes: int | None = None) -> dict[str, Any]:
    extension = DEFAULT_EXTENSION
    baseline_len = max(1, int(round(DEFAULT_BASELINE_FRACTION * extension)))
    total_len = baseline_len + extension
    return {
        "sg_window": DEFAULT_SG_WINDOW,
        "sg_poly": DEFAULT_SG_POLY,
        "starts": [],
        "extension": extension,
        "area_left": baseline_len + 1,
        "area_right": total_len,
        "baseline_fraction": DEFAULT_BASELINE_FRACTION,
    }


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


def normalize_heatmap_params(params: dict[str, Any] | None) -> dict[str, Any]:
    raw = dict(params or {})
    base = default_heatmap_params()
    base.update(raw)
    starts = base.get("starts") or []
    if isinstance(starts, str):
        starts = parse_start_frames(starts, 10**9)
    base["starts"] = [int(s) for s in starts]
    base["sg_window"] = int(base["sg_window"])
    base["sg_poly"] = int(base["sg_poly"])
    base["extension"] = int(base["extension"])
    left = int(base["area_left"])
    right = int(base["area_right"])
    if right < left:
        left, right = right, left
    base["area_left"] = left
    base["area_right"] = right
    base["baseline_fraction"] = float(base.get("baseline_fraction") or DEFAULT_BASELINE_FRACTION)
    return base


def parse_start_frames(text: str, n_frames: int) -> list[int]:
    parts = [p.strip() for p in str(text).replace(";", ",").split(",") if p.strip()]
    if not parts:
        return []
    values = [max(0, min(int(float(p)), max(int(n_frames) - 1, 0))) for p in parts]
    return values


def format_start_frames(frames: list[int]) -> str:
    return ", ".join(str(int(f)) for f in frames)


def segment_geometry(
    extension: int, baseline_fraction: float = DEFAULT_BASELINE_FRACTION
) -> tuple[int, int, np.ndarray]:
    baseline_len = max(1, int(round(float(baseline_fraction) * int(extension))))
    total_len = baseline_len + int(extension)
    rel_x = np.arange(1, total_len + 1)
    return baseline_len, total_len, rel_x


def compute_area_from_mean_trace(
    rel_x: np.ndarray,
    mean_values: np.ndarray,
    f_left: int,
    f_right: int,
    baseline_level: float = 1.0,
) -> np.ndarray:
    """Integrate (mean - baseline) between Area L and R. NaNs are skipped."""
    if f_right < f_left:
        f_left, f_right = f_right, f_left
    squeeze = mean_values.ndim == 1
    if squeeze:
        mean_values = mean_values[:, np.newaxis]
    frames = np.asarray(rel_x, dtype=np.float64)
    values = np.asarray(mean_values, dtype=np.float64) - baseline_level
    n_pix = values.shape[1]
    areas = np.full(n_pix, np.nan, dtype=np.float64)
    overlap = (frames >= f_left) & (frames <= f_right)
    if not np.any(overlap):
        return areas[0] if squeeze else areas
    x = frames[overlap]
    y = values[overlap, :]
    for col in range(n_pix):
        yc = y[:, col]
        finite = np.isfinite(yc) & np.isfinite(x)
        if int(np.count_nonzero(finite)) < 2:
            continue
        areas[col] = float(trapezoid(yc[finite], x[finite]))
    return float(areas[0]) if squeeze else areas


def compute_heatmap_map(
    suite2p_dir: Path,
    params: dict[str, Any],
    nan_mask: np.ndarray,
    *,
    progress: Callable[[str, float], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> np.ndarray:
    """Per-pixel area map from data.bin. LED+Shutter frames are treated as NaN."""
    params = normalize_heatmap_params(params)
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

    starts = [s for s in params["starts"] if 0 <= int(s) < nframes]
    extension = int(params["extension"])
    valid_starts = [int(s) for s in starts if int(s) + extension <= nframes]
    if not valid_starts:
        raise ValueError("Need at least one start frame with room for the extension")

    mmap = np.memmap(path, dtype=dtype, mode="r", shape=(nframes, height * width))
    n_pixels = height * width
    keep = ~mask
    n_keep = int(np.count_nonzero(keep))
    if n_keep < 3:
        raise ValueError("Too few non-shutter frames to smooth")

    baseline_len, total_len, rel_x = segment_geometry(
        extension, float(params["baseline_fraction"])
    )
    window = int(params["sg_window"])
    poly = int(params["sg_poly"])
    f_left = int(params["area_left"])
    f_right = int(params["area_right"])

    def report(stage: str, fraction: float) -> None:
        if progress is not None:
            progress(stage, float(fraction))
        if should_cancel is not None and should_cancel():
            raise HeatmapCancelled()

    areas = np.full(n_pixels, np.nan, dtype=np.float64)
    report("Preparing stack", 0.0)

    for col_start in range(0, n_pixels, PIXEL_CHUNK):
        col_end = min(col_start + PIXEL_CHUNK, n_pixels)
        raw = np.asarray(mmap[:, col_start:col_end], dtype=np.float64)
        raw[~keep, :] = np.nan
        excised = raw[keep, :]
        smooth_exc = apply_savgol(excised, window, poly, axis=0)
        smooth = np.full_like(raw, np.nan)
        smooth[keep, :] = smooth_exc

        aligned_segments: list[np.ndarray] = []
        for start in valid_starts:
            seg_start = max(0, start - baseline_len)
            chunk = smooth[seg_start : start + extension, :]
            available_baseline = start - seg_start
            with np.errstate(all="ignore"):
                baseline_mean = np.nanmean(chunk[:available_baseline, :], axis=0)
            baseline_mean = np.where(
                ~np.isfinite(baseline_mean) | (baseline_mean == 0.0),
                1.0,
                baseline_mean,
            )
            aligned = np.full((total_len, chunk.shape[1]), np.nan, dtype=np.float64)
            offset = baseline_len - available_baseline
            aligned[offset : offset + chunk.shape[0], :] = chunk / baseline_mean[np.newaxis, :]
            aligned_segments.append(aligned)

        stacked = np.stack(aligned_segments, axis=0)
        with np.errstate(all="ignore"):
            mean_trace = np.nanmean(stacked, axis=0)
        areas[col_start:col_end] = compute_area_from_mean_trace(
            rel_x, mean_trace, f_left, f_right
        )
        report("Smoothing pixels", 0.05 + 0.90 * (col_end / n_pixels))
        del raw, excised, smooth_exc, smooth, aligned_segments, stacked, mean_trace

    report("Heatmap ready", 1.0)
    del mmap
    return np.asarray(areas, dtype=np.float32).reshape(height, width)


def make_heatmap(
    doc: dict[str, Any],
    *,
    label: str,
    params: dict[str, Any],
    heatmap_map: np.ndarray,
    heatmap_id: str | None = None,
) -> dict[str, Any]:
    now = _utc_now()
    return {
        "id": heatmap_id or next_heatmap_id(doc),
        "label": str(label).strip() or "Untitled",
        "params": deepcopy(normalize_heatmap_params(params)),
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
) -> None:
    if label is not None:
        hm["label"] = str(label).strip() or "Untitled"
    if params is not None:
        hm["params"] = deepcopy(normalize_heatmap_params(params))
    if heatmap_map is not None:
        hm["map"] = np.asarray(heatmap_map, dtype=np.float32)
    hm["updated_utc"] = _utc_now()
