"""Savitzky–Golay smoothing and bleach correction for stored traces.

SG and bleach run on illumination samples only: LED+Shutter frames are
excised, then scattered back as NaN. ``tc_norm_sm`` / ``tc_norm_sm_bc``
are min–max normalized like ``tc_norm``.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter

from s2p_trace_curation.raster import compute_tc_norm, led_shutter_nan_mask, tc_norm_sig

TRACE_FIELD_NORM = "tc_norm"
TRACE_FIELD_SM = "tc_norm_sm"
TRACE_FIELD_SM_BC = "tc_norm_sm_bc"
TRACE_FIELDS = (TRACE_FIELD_NORM, TRACE_FIELD_SM, TRACE_FIELD_SM_BC)
TRACE_FIELD_LABELS = {
    TRACE_FIELD_NORM: "tc_norm",
    TRACE_FIELD_SM: "tc_norm_sm",
    TRACE_FIELD_SM_BC: "tc_norm_sm_bc",
}

TAU_SHARED = "shared"
TAU_INDEPENDENT = "independent"

DEFAULT_SG_WINDOW = 11
DEFAULT_SG_POLY = 2


def default_trace_processing() -> dict[str, Any]:
    return {
        "sg_window": DEFAULT_SG_WINDOW,
        "sg_poly": DEFAULT_SG_POLY,
        "bleach_enabled": False,
        "tau_mode": TAU_SHARED,
        "shared_tau1": None,
        "shared_tau2": None,
    }


def ensure_trace_processing(doc: dict[str, Any]) -> dict[str, Any]:
    meta = doc.setdefault("meta", {})
    tp = meta.get("trace_processing")
    if not isinstance(tp, dict):
        tp = default_trace_processing()
        meta["trace_processing"] = tp
    else:
        base = default_trace_processing()
        base.update(tp)
        tp = base
        meta["trace_processing"] = tp
    field = str(meta.get("raster_trace_field") or "")
    if field not in TRACE_FIELDS:
        meta["raster_trace_field"] = default_raster_trace_field(doc)
    return tp


def default_raster_trace_field(doc: dict[str, Any]) -> str:
    rois = doc.get("rois") or []
    if any(r.get(TRACE_FIELD_SM_BC) is not None for r in rois):
        return TRACE_FIELD_SM_BC
    if any(r.get(TRACE_FIELD_SM) is not None for r in rois):
        return TRACE_FIELD_SM
    return TRACE_FIELD_NORM


def raster_trace_field(doc: dict[str, Any]) -> str:
    ensure_trace_processing(doc)
    field = str((doc.get("meta") or {}).get("raster_trace_field") or TRACE_FIELD_NORM)
    return field if field in TRACE_FIELDS else TRACE_FIELD_NORM


def set_raster_trace_field(doc: dict[str, Any], field: str) -> None:
    if field not in TRACE_FIELDS:
        raise ValueError(f"Unknown trace field: {field}")
    doc.setdefault("meta", {})["raster_trace_field"] = str(field)


def normalize_sg_params(window: int, poly: int, n_samples: int) -> tuple[int, int]:
    window = int(window)
    if window % 2 == 0:
        window += 1
    max_window = n_samples - 1 if n_samples % 2 == 0 else n_samples
    window = max(3, min(window, max_window))
    poly = min(int(poly), window - 1)
    poly = max(1, poly)
    return window, poly


def apply_savgol(trace: np.ndarray, window: int, polyorder: int, axis: int = 0) -> np.ndarray:
    """Savitzky–Golay along ``axis``. Returns a copy if the series is too short."""
    arr = np.asarray(trace, dtype=np.float64)
    length = int(arr.shape[axis])
    window, polyorder = normalize_sg_params(window, polyorder, length)
    if window < 3 or polyorder < 1 or length < 3:
        return arr.copy()
    return savgol_filter(arr, window_length=window, polyorder=polyorder, axis=axis)


def illumination_keep(trace: np.ndarray, nan_mask: np.ndarray) -> np.ndarray:
    t = np.asarray(trace, dtype=np.float64)
    mask = np.asarray(nan_mask, dtype=bool)
    if mask.shape[0] != t.shape[0]:
        raise ValueError("nan_mask length must match trace")
    return np.isfinite(t) & ~mask


def scatter_to_full(values: np.ndarray, keep: np.ndarray, nframes: int) -> np.ndarray:
    out = np.full(int(nframes), np.nan, dtype=np.float64)
    out[np.asarray(keep, dtype=bool)] = np.asarray(values, dtype=np.float64)
    return out


def biexponential_decay(
    t: np.ndarray, a1: float, tau1: float, a2: float, tau2: float, c: float
) -> np.ndarray:
    return a1 * np.exp(-t / tau1) + a2 * np.exp(-t / tau2) + c


def conservative_fit_params(signal: np.ndarray) -> tuple[float, float, float, float, float]:
    y = np.asarray(signal, dtype=np.float64)
    finite = y[np.isfinite(y)]
    c = float(np.mean(finite)) if finite.size else 0.0
    return (0.0, 1.0, 0.0, 1.0, c)


def parse_fit_params(value: Any) -> tuple[float, float, float, float, float] | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple)) and len(value) >= 5:
        try:
            return tuple(float(value[i]) for i in range(5))  # type: ignore[return-value]
        except (TypeError, ValueError):
            return None
    return None


def fit_biexponential_params(signal: np.ndarray) -> tuple[float, float, float, float, float] | None:
    values = np.asarray(signal, dtype=np.float64)
    if values.ndim != 1 or values.size < 6:
        return None
    if not np.all(np.isfinite(values)):
        return None
    t = np.arange(values.size, dtype=np.float64)
    tail = float(values[-1])
    head = float(values[0])
    amplitude = head - tail
    if abs(amplitude) < 1e-12:
        amplitude = float(np.ptp(values)) or 1.0
    span = max(float(values.size), 1.0)
    initial = [
        amplitude * 0.6,
        max(span / 10.0, 1.0),
        amplitude * 0.4,
        max(span / 2.0, 1.0),
        tail,
    ]
    lower = [-np.inf, 0.1, -np.inf, 0.1, -np.inf]
    upper = [np.inf, span * 20.0, np.inf, span * 20.0, np.inf]
    try:
        params, _ = curve_fit(
            biexponential_decay,
            t,
            values,
            p0=initial,
            bounds=(lower, upper),
            maxfev=50_000,
        )
    except (RuntimeError, ValueError, TypeError):
        return None
    fitted = biexponential_decay(t, *params)
    if not np.all(np.isfinite(fitted)):
        return None
    return tuple(float(v) for v in params)


def fit_amplitudes_frozen_tau(
    signal: np.ndarray, tau1: float, tau2: float
) -> tuple[float, float, float, float, float] | None:
    values = np.asarray(signal, dtype=np.float64)
    if values.ndim != 1 or values.size < 4:
        return None
    if not np.all(np.isfinite(values)):
        return None
    t1 = max(float(tau1), 0.1)
    t2 = max(float(tau2), 0.1)

    def _fn(t: np.ndarray, a1: float, a2: float, c: float) -> np.ndarray:
        return a1 * np.exp(-t / t1) + a2 * np.exp(-t / t2) + c

    t = np.arange(values.size, dtype=np.float64)
    tail = float(values[-1])
    head = float(values[0])
    amplitude = head - tail
    if abs(amplitude) < 1e-12:
        amplitude = float(np.ptp(values)) or 1.0
    try:
        popt, _ = curve_fit(
            _fn,
            t,
            values,
            p0=[amplitude * 0.6, amplitude * 0.4, tail],
            maxfev=50_000,
        )
    except (RuntimeError, ValueError, TypeError):
        return None
    return (float(popt[0]), t1, float(popt[1]), t2, float(popt[2]))


def _led_list(doc: dict[str, Any]) -> list[list[int]]:
    from s2p_trace_curation.annotations import ensure_annotations

    return sorted(
        [int(a["start_frame"]), int(a["end_frame"])]
        for a in ensure_annotations(doc)
        if str(a["property"]) == "LED+Shutter"
    )


def _comp_sums(doc: dict[str, Any]) -> tuple[list[int], list[float]]:
    ids: list[int] = []
    sums: list[float] = []
    for row in doc.get("rois") or []:
        ids.append(int(row["roi_id"]))
        tc = np.asarray(row["compensation"]["trace_comp"], dtype=np.float64)
        sums.append(float(np.nansum(tc)))
    return ids, sums


def _field_sums(doc: dict[str, Any], field: str) -> tuple[list[int], list[float | None]]:
    ids: list[int] = []
    sums: list[float | None] = []
    for row in doc.get("rois") or []:
        ids.append(int(row["roi_id"]))
        tr = row.get(field)
        if tr is None:
            sums.append(None)
        else:
            sums.append(float(np.nansum(np.asarray(tr, dtype=np.float64))))
    return ids, sums


def _sums_close(a: Any, b: Any) -> bool:
    sa = list(a or [])
    sb = list(b or [])
    if len(sa) != len(sb):
        return False
    for xa, xb in zip(sa, sb):
        if xa is None and xb is None:
            continue
        if xa is None or xb is None:
            return False
        if not np.isclose(float(xa), float(xb), rtol=0.0, atol=1e-6, equal_nan=True):
            return False
    return True


def tc_norm_sm_sig(doc: dict[str, Any]) -> dict[str, Any]:
    tp = ensure_trace_processing(doc)
    ids, sums = _comp_sums(doc)
    return {
        "led": _led_list(doc),
        "ids": ids,
        "sums": sums,
        "sg_window": int(tp["sg_window"]),
        "sg_poly": int(tp["sg_poly"]),
    }


def tc_norm_sm_bc_sig(doc: dict[str, Any]) -> dict[str, Any]:
    tp = ensure_trace_processing(doc)
    ids, sums = _field_sums(doc, TRACE_FIELD_SM)
    flags = [
        bool((row.get("bleach") or {}).get("conservative", True))
        for row in doc.get("rois") or []
    ]
    return {
        "led": _led_list(doc),
        "ids": ids,
        "sm_sums": sums,
        "bleach_enabled": bool(tp["bleach_enabled"]),
        "tau_mode": str(tp["tau_mode"]),
        "shared_tau1": tp.get("shared_tau1"),
        "shared_tau2": tp.get("shared_tau2"),
        "conservative": flags,
    }


def _sm_sigs_equal(a: Any, b: Any) -> bool:
    if not isinstance(a, dict) or not isinstance(b, dict):
        return False
    if [list(x) for x in a.get("led") or []] != [list(x) for x in b.get("led") or []]:
        return False
    if list(a.get("ids") or []) != list(b.get("ids") or []):
        return False
    if int(a.get("sg_window", -1)) != int(b.get("sg_window", -2)):
        return False
    if int(a.get("sg_poly", -1)) != int(b.get("sg_poly", -2)):
        return False
    sa = np.asarray(a.get("sums") or [], dtype=np.float64)
    sb = np.asarray(b.get("sums") or [], dtype=np.float64)
    if sa.shape != sb.shape:
        return False
    return bool(np.allclose(sa, sb, rtol=0.0, atol=1e-6, equal_nan=True))


def _bc_sigs_equal(a: Any, b: Any) -> bool:
    if not isinstance(a, dict) or not isinstance(b, dict):
        return False
    if [list(x) for x in a.get("led") or []] != [list(x) for x in b.get("led") or []]:
        return False
    if list(a.get("ids") or []) != list(b.get("ids") or []):
        return False
    if bool(a.get("bleach_enabled")) != bool(b.get("bleach_enabled")):
        return False
    if str(a.get("tau_mode")) != str(b.get("tau_mode")):
        return False
    for key in ("shared_tau1", "shared_tau2"):
        va, vb = a.get(key), b.get(key)
        if va is None and vb is None:
            continue
        if va is None or vb is None:
            return False
        if not np.isclose(float(va), float(vb), rtol=0.0, atol=1e-6):
            return False
    if list(a.get("conservative") or []) != list(b.get("conservative") or []):
        return False
    return _sums_close(a.get("sm_sums"), b.get("sm_sums"))


def rois_missing_field(doc: dict[str, Any], field: str) -> list[dict[str, Any]]:
    return [row for row in doc.get("rois") or [] if row.get(field) is None]


def tc_norm_sm_is_stale(doc: dict[str, Any]) -> bool:
    rois = doc.get("rois") or []
    if not rois:
        return False
    has = [row.get(TRACE_FIELD_SM) is not None for row in rois]
    if not any(has):
        return False
    if not all(has):
        return True
    stored = (doc.get("meta") or {}).get("tc_norm_sm_sig")
    return not _sm_sigs_equal(stored, tc_norm_sm_sig(doc))


def tc_norm_sm_bc_is_stale(doc: dict[str, Any]) -> bool:
    rois = doc.get("rois") or []
    if not rois:
        return False
    has = [row.get(TRACE_FIELD_SM_BC) is not None for row in rois]
    if not any(has):
        return False
    if not all(has):
        return True
    stored = (doc.get("meta") or {}).get("tc_norm_sm_bc_sig")
    return not _bc_sigs_equal(stored, tc_norm_sm_bc_sig(doc))


def trace_field_is_stale(doc: dict[str, Any], field: str) -> bool:
    from s2p_trace_curation.raster import tc_norm_is_stale

    if field == TRACE_FIELD_NORM:
        return tc_norm_is_stale(doc)
    if field == TRACE_FIELD_SM:
        return tc_norm_sm_is_stale(doc)
    if field == TRACE_FIELD_SM_BC:
        return tc_norm_sm_is_stale(doc) or tc_norm_sm_bc_is_stale(doc)
    return False


def field_has_any(doc: dict[str, Any], field: str) -> bool:
    return any(row.get(field) is not None for row in doc.get("rois") or [])


def compute_tc_norm_sm_from_comp(
    trace_comp: np.ndarray,
    nan_mask: np.ndarray,
    sg_window: int,
    sg_poly: int,
) -> np.ndarray:
    comp = np.asarray(trace_comp, dtype=np.float64)
    keep = illumination_keep(comp, nan_mask)
    if int(np.count_nonzero(keep)) < 3:
        return np.full(comp.shape, np.nan, dtype=np.float64)
    smooth_exc = apply_savgol(comp[keep], sg_window, sg_poly)
    smooth_full = scatter_to_full(smooth_exc, keep, comp.shape[0])
    return compute_tc_norm(smooth_full, nan_mask)


def rebuild_all_tc_norm_sm(doc: dict[str, Any]) -> None:
    tp = ensure_trace_processing(doc)
    nframes = int(doc["meta"]["nframes"])
    mask = led_shutter_nan_mask(doc, nframes)
    window = int(tp["sg_window"])
    poly = int(tp["sg_poly"])
    for row in doc["rois"]:
        row[TRACE_FIELD_SM] = compute_tc_norm_sm_from_comp(
            row["compensation"]["trace_comp"], mask, window, poly
        )
    doc.setdefault("meta", {})["tc_norm_sm_sig"] = tc_norm_sm_sig(doc)


def mean_illumination_trace(
    doc: dict[str, Any], field: str, nan_mask: np.ndarray, roi_ids: list[int] | None = None
) -> np.ndarray | None:
    by_id = {int(r["roi_id"]): r for r in doc.get("rois") or []}
    ids = roi_ids if roi_ids is not None else [int(r["roi_id"]) for r in doc["rois"]]
    acc: list[np.ndarray] = []
    for i in ids:
        row = by_id.get(int(i))
        if row is None:
            continue
        tr = row.get(field)
        if tr is None:
            continue
        acc.append(np.asarray(tr, dtype=np.float64))
    if not acc:
        return None
    stacked = np.vstack(acc)
    stacked[:, np.asarray(nan_mask, dtype=bool)] = np.nan
    with np.errstate(all="ignore"):
        mean = np.nanmean(stacked, axis=0)
    return np.asarray(mean, dtype=np.float64)


def estimate_shared_taus(doc: dict[str, Any], nan_mask: np.ndarray) -> tuple[float, float] | None:
    ids = [int(r["roi_id"]) for r in doc.get("rois") or [] if bool(r.get("iscell", True))]
    if len(ids) < 1:
        ids = [int(r["roi_id"]) for r in doc["rois"]]
    mean = mean_illumination_trace(doc, TRACE_FIELD_SM, nan_mask, ids)
    if mean is None:
        return None
    keep = illumination_keep(mean, nan_mask)
    if int(np.count_nonzero(keep)) < 6:
        return None
    params = fit_biexponential_params(mean[keep])
    if params is None:
        return None
    return float(params[1]), float(params[3])


def _fit_row_bleach(
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


def apply_bleach_to_sm(
    sm: np.ndarray,
    nan_mask: np.ndarray,
    params: tuple[float, float, float, float, float],
) -> np.ndarray:
    keep = illumination_keep(sm, nan_mask)
    y = sm[keep]
    if y.size == 0:
        return np.full(sm.shape, np.nan, dtype=np.float64)
    t = np.arange(y.size, dtype=np.float64)
    fitted = biexponential_decay(t, *params)
    corrected = scatter_to_full(y - fitted, keep, sm.shape[0])
    return compute_tc_norm(corrected, nan_mask)


def bleach_fit_curve(row: dict[str, Any], nan_mask: np.ndarray, nframes: int) -> np.ndarray | None:
    """Full-length biexponential overlay for the bleach subplot (NaN on shutter)."""
    sm = row.get(TRACE_FIELD_SM)
    params = parse_fit_params((row.get("bleach") or {}).get("fit_params"))
    if sm is None or params is None:
        return None
    sm = np.asarray(sm, dtype=np.float64)
    keep = illumination_keep(sm, nan_mask)
    n_keep = int(np.count_nonzero(keep))
    if n_keep == 0:
        return None
    t = np.arange(n_keep, dtype=np.float64)
    return scatter_to_full(biexponential_decay(t, *params), keep, nframes)


def rebuild_all_tc_norm_sm_bc(doc: dict[str, Any]) -> None:
    tp = ensure_trace_processing(doc)
    nframes = int(doc["meta"]["nframes"])
    mask = led_shutter_nan_mask(doc, nframes)
    enabled = bool(tp["bleach_enabled"])
    tau_mode = str(tp.get("tau_mode") or TAU_SHARED)
    tau1 = tau2 = None
    if not enabled:
        tp["shared_tau1"] = None
        tp["shared_tau2"] = None
    elif tau_mode == TAU_SHARED:
        stored1, stored2 = tp.get("shared_tau1"), tp.get("shared_tau2")
        if stored1 is not None and stored2 is not None:
            tau1, tau2 = float(stored1), float(stored2)
        else:
            shared = estimate_shared_taus(doc, mask)
            if shared is not None:
                tau1, tau2 = shared
                tp["shared_tau1"] = float(tau1)
                tp["shared_tau2"] = float(tau2)

    for row in doc["rois"]:
        sm = row.get(TRACE_FIELD_SM)
        if sm is None:
            row[TRACE_FIELD_SM_BC] = None
            row["bleach"] = {"fit_params": None, "conservative": True}
            continue
        sm_arr = np.asarray(sm, dtype=np.float64)
        use_cons = not enabled
        freeze = tau_mode == TAU_SHARED and tau1 is not None and tau2 is not None
        params, cons = _fit_row_bleach(
            sm_arr,
            mask,
            conservative=use_cons,
            tau1=tau1 if (enabled and freeze) else None,
            tau2=tau2 if (enabled and freeze) else None,
        )
        row["bleach"] = {"fit_params": list(params), "conservative": bool(cons)}
        row[TRACE_FIELD_SM_BC] = apply_bleach_to_sm(sm_arr, mask, params)
    doc.setdefault("meta", {})["tc_norm_sm_bc_sig"] = tc_norm_sm_bc_sig(doc)


def row_trace_field(row: dict[str, Any], field: str, nframes: int) -> np.ndarray:
    tr = row.get(field)
    out = np.full(int(nframes), np.nan, dtype=np.float64)
    if tr is None:
        return out
    arr = np.asarray(tr, dtype=np.float64)
    n_copy = min(arr.shape[0], out.shape[0])
    out[:n_copy] = arr[:n_copy]
    return out


def stack_trace_field(
    rows: list[dict[str, Any]], nframes: int, field: str
) -> np.ndarray:
    n = len(rows)
    out = np.full((n, int(nframes)), np.nan, dtype=np.float64)
    for i, row in enumerate(rows):
        out[i] = row_trace_field(row, field, nframes)
    return out
