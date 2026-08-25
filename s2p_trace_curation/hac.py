"""Hierarchical clustering on selected tc_norm traces."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import pdist, squareform

METRIC_RUZICKA = "ruzicka"
METRIC_EUCLIDEAN = "euclidean"
LINKAGE_AVERAGE = "average"
LINKAGE_WARD = "ward"

HAC_METRICS = (METRIC_RUZICKA, METRIC_EUCLIDEAN)
HAC_LINKAGES = (LINKAGE_AVERAGE, LINKAGE_WARD)

HAC_METRIC_LABELS = {
    METRIC_RUZICKA: "Ružička",
    METRIC_EUCLIDEAN: "Euclidean",
}
HAC_LINKAGE_LABELS = {
    LINKAGE_AVERAGE: "Average",
    LINKAGE_WARD: "Ward",
}

DEFAULT_HAC_PARAMS: dict[str, str] = {
    "metric": METRIC_RUZICKA,
    "linkage": LINKAGE_AVERAGE,
    "trace_field": "tc_norm_sm_bc",
}


def normalize_hac_params(params: dict[str, Any] | None) -> dict[str, str]:
    from s2p_trace_curation.trace_processing import TRACE_FIELD_NORM, TRACE_FIELDS

    raw = dict(params or {})
    metric = str(raw.get("metric") or METRIC_RUZICKA)
    linkage_name = str(raw.get("linkage") or LINKAGE_AVERAGE)
    field = raw.get("trace_field")
    if field is None or str(field) == "":
        field = TRACE_FIELD_NORM
    else:
        field = str(field)
    if field not in TRACE_FIELDS:
        field = TRACE_FIELD_NORM
    if metric not in HAC_METRICS:
        raise ValueError(f"Unknown HAC metric: {metric}")
    if linkage_name not in HAC_LINKAGES:
        raise ValueError(f"Unknown HAC linkage: {linkage_name}")
    if linkage_name == LINKAGE_WARD and metric != METRIC_EUCLIDEAN:
        raise ValueError("Ward linkage requires Euclidean distance")
    return {"metric": metric, "linkage": linkage_name, "trace_field": field}


def trace_stack_for_ids(
    doc: dict[str, Any], roi_ids: list[int], field: str = "tc_norm"
) -> np.ndarray:
    """Stack a stored trace field for roi_ids as (n_roi, nframes). Missing → NaN rows."""
    nframes = int(doc["meta"]["nframes"])
    by_id = {int(r["roi_id"]): r for r in doc["rois"]}
    out = np.full((len(roi_ids), nframes), np.nan, dtype=np.float64)
    for i, rid in enumerate(roi_ids):
        row = by_id.get(int(rid))
        if row is None:
            continue
        tr = row.get(field)
        if tr is None:
            continue
        arr = np.asarray(tr, dtype=np.float64)
        n_copy = min(arr.shape[0], nframes)
        out[i, :n_copy] = arr[:n_copy]
    return out


def tc_norm_stack_for_ids(doc: dict[str, Any], roi_ids: list[int]) -> np.ndarray:
    """Stack tc_norm for roi_ids as (n_roi, nframes). Missing → NaN rows."""
    return trace_stack_for_ids(doc, roi_ids, "tc_norm")


def drop_nan_frames(X: np.ndarray) -> np.ndarray:
    """Keep columns finite for every ROI (shared LED+Shutter NaNs)."""
    if X.ndim != 2 or X.shape[0] == 0:
        raise ValueError("Need a 2-D trace matrix")
    keep = np.isfinite(X).all(axis=0)
    if not bool(keep.any()):
        raise ValueError("No finite frames left to cluster (check tc_norm / LED+Shutter)")
    return X[:, keep]


def ruzicka_condensed(X: np.ndarray) -> np.ndarray:
    """Condensed Ružička distances: 1 - sum(min)/sum(max) per pair."""
    n = int(X.shape[0])
    if n < 2:
        return np.zeros(0, dtype=np.float64)
    out = np.empty(n * (n - 1) // 2, dtype=np.float64)
    k = 0
    for i in range(n - 1):
        a = X[i]
        rest = X[i + 1 :]
        mn = np.minimum(rest, a).sum(axis=1)
        mx = np.maximum(rest, a).sum(axis=1)
        sim = np.divide(mn, mx, out=np.ones_like(mn), where=mx > 0)
        n_rest = n - 1 - i
        out[k : k + n_rest] = 1.0 - sim
        k += n_rest
    return out


def run_hac(doc: dict[str, Any], params: dict[str, Any] | None = None) -> dict[str, Any]:
    """Cluster selected (iscell=True) tc_norm traces. Does not write the pickle.

    Returns roi_ids (pickle order), leaf ``order``, linkage Z, and a leaf-sorted
    square matrix for display (similarity for Ružička, distance for Euclidean).
    """
    from s2p_trace_curation.analyses import current_iscell_ids

    params = normalize_hac_params(params)
    roi_ids = current_iscell_ids(doc)
    if len(roi_ids) < 2:
        raise ValueError("HAC needs at least two selected (iscell=True) ROIs")
    X = drop_nan_frames(trace_stack_for_ids(doc, roi_ids, params["trace_field"]))
    if not np.isfinite(X).all():
        raise ValueError("Some selected traces still contain NaNs after dropping LED frames")

    if params["metric"] == METRIC_RUZICKA:
        condensed = ruzicka_condensed(X)
    else:
        condensed = pdist(X, metric="euclidean")

    Z = linkage(condensed, method=params["linkage"])
    leaves = [int(i) for i in leaves_list(Z)]
    order = [int(roi_ids[i]) for i in leaves]
    square = squareform(condensed)
    sorted_sq = square[np.ix_(leaves, leaves)]
    if params["metric"] == METRIC_RUZICKA:
        display = 1.0 - sorted_sq
        display_kind = "similarity"
    else:
        display = sorted_sq
        display_kind = "distance"
    return {
        "kind": "hac",
        "params": params,
        "roi_ids": [int(i) for i in roi_ids],
        "order": order,
        "Z": Z,
        "leaves": leaves,
        "matrix": np.asarray(display, dtype=np.float64),
        "display_kind": display_kind,
        "n_frames_used": int(X.shape[1]),
    }
