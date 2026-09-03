"""Hierarchical clustering on selected tc_norm traces."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.cluster.hierarchy import fcluster, leaves_list, linkage
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

DEFAULT_HAC_PARAMS: dict[str, Any] = {
    "metric": METRIC_RUZICKA,
    "linkage": LINKAGE_AVERAGE,
    "trace_field": "tc_norm_sm_bc",
}

# Shared with the similarity-matrix outlines, raster boxes, and FOV fills.
HAC_CLUSTER_COLORS: tuple[str, ...] = (
    "#00e5ff",
    "#7cff6b",
    "#ffd54a",
    "#ff6b6b",
    "#c77dff",
    "#4fc3f7",
    "#ff9e80",
    "#f48fb1",
)


def hac_cluster_hex(index: int) -> str:
    return HAC_CLUSTER_COLORS[int(index) % len(HAC_CLUSTER_COLORS)]


def hac_cluster_rgb(index: int) -> tuple[int, int, int]:
    h = hac_cluster_hex(index).lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def normalize_hac_params(params: dict[str, Any] | None) -> dict[str, Any]:
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
    out: dict[str, Any] = {
        "metric": metric,
        "linkage": linkage_name,
        "trace_field": field,
    }
    if raw.get("cut_threshold") is not None:
        try:
            cut = float(raw["cut_threshold"])
        except (TypeError, ValueError):
            cut = float("nan")
        if np.isfinite(cut) and cut >= 0.0:
            out["cut_threshold"] = cut
    return out


def max_linkage_distance(Z: np.ndarray) -> float:
    Z = np.asarray(Z, dtype=np.float64)
    if Z.ndim != 2 or Z.shape[0] == 0:
        return 0.0
    return float(np.max(Z[:, 2]))


def default_cut_threshold(Z: np.ndarray) -> float:
    """Same rule as scipy's dendrogram ``color_threshold``: 0.7 × max height."""
    return 0.7 * max_linkage_distance(Z)


def clusters_at_distance(
    Z: np.ndarray,
    roi_ids: list[int],
    leaves: list[int],
    threshold: float,
) -> list[list[int]]:
    """Cut the tree at ``threshold``; return clusters as roi_id lists in leaf order."""
    n = len(roi_ids)
    if n == 0:
        return []
    if n == 1:
        return [[int(roi_ids[0])]]
    labels = np.asarray(
        fcluster(Z, t=float(threshold), criterion="distance"),
        dtype=np.int64,
    )
    leaf_idx = np.asarray(leaves, dtype=np.int64)
    leaf_labels = labels[leaf_idx]
    order_ids = [int(roi_ids[int(i)]) for i in leaf_idx]
    clusters: list[list[int]] = []
    start = 0
    for i in range(1, len(leaf_labels) + 1):
        if i == len(leaf_labels) or int(leaf_labels[i]) != int(leaf_labels[start]):
            clusters.append(order_ids[start:i])
            start = i
    return clusters


def cluster_leaf_spans(clusters: list[list[int]]) -> list[tuple[int, int]]:
    """Inclusive leaf-index spans for each cluster along the seriated order."""
    spans: list[tuple[int, int]] = []
    i = 0
    for group in clusters:
        n = len(group)
        if n:
            spans.append((i, i + n - 1))
        i += n
    return spans


def apply_distance_cut(result: dict[str, Any], threshold: float) -> list[list[int]]:
    """Write ``clusters`` / ``cut_threshold`` onto a ``run_hac`` result dict."""
    Z = result["Z"]
    clusters = clusters_at_distance(
        Z,
        [int(i) for i in result["roi_ids"]],
        [int(i) for i in result["leaves"]],
        float(threshold),
    )
    params = dict(result.get("params") or {})
    params["cut_threshold"] = float(threshold)
    result["params"] = params
    result["cut_threshold"] = float(threshold)
    result["clusters"] = clusters
    return clusters


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


def run_hac(
    doc: dict[str, Any],
    params: dict[str, Any] | None = None,
    *,
    roi_ids: list[int] | None = None,
) -> dict[str, Any]:
    """Cluster selected traces. Does not write the pickle.

    ``roi_ids`` defaults to current ``iscell=True``. Pass a saved membership
    list to redraw that run. Returns leaf ``order``, linkage Z, a leaf-sorted
    square matrix (similarity for Ružička, distance for Euclidean), and
    ``clusters`` from a distance cut (``params["cut_threshold"]``, or 0.7 ×
    max merge height when omitted).
    """
    from s2p_trace_curation.analyses import current_iscell_ids

    params = normalize_hac_params(params)
    have = {int(r["roi_id"]) for r in doc["rois"]}
    if roi_ids is None:
        ids = current_iscell_ids(doc)
    else:
        ids = [int(i) for i in roi_ids if int(i) in have]
    if len(ids) < 2:
        raise ValueError("HAC needs at least two selected (iscell=True) ROIs")
    X = drop_nan_frames(trace_stack_for_ids(doc, ids, params["trace_field"]))
    if not np.isfinite(X).all():
        raise ValueError("Some selected traces still contain NaNs after dropping LED frames")

    if params["metric"] == METRIC_RUZICKA:
        condensed = ruzicka_condensed(X)
    else:
        condensed = pdist(X, metric="euclidean")

    Z = linkage(condensed, method=params["linkage"])
    leaves = [int(i) for i in leaves_list(Z)]
    order = [int(ids[i]) for i in leaves]
    square = squareform(condensed)
    sorted_sq = square[np.ix_(leaves, leaves)]
    if params["metric"] == METRIC_RUZICKA:
        display = 1.0 - sorted_sq
        display_kind = "similarity"
    else:
        display = sorted_sq
        display_kind = "distance"
    if "cut_threshold" in params:
        cut = float(params["cut_threshold"])
    else:
        cut = default_cut_threshold(Z)
    clusters = clusters_at_distance(Z, ids, leaves, cut)
    params = dict(params)
    params["cut_threshold"] = float(cut)
    return {
        "kind": "hac",
        "params": params,
        "roi_ids": [int(i) for i in ids],
        "order": order,
        "Z": Z,
        "leaves": leaves,
        "matrix": np.asarray(display, dtype=np.float64),
        "display_kind": display_kind,
        "n_frames_used": int(X.shape[1]),
        "cut_threshold": float(cut),
        "clusters": clusters,
    }
