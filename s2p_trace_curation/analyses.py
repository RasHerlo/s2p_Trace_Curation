"""Named analysis runs: params, member list, sort order, fingerprints.

Matrices are recomputed in the Analysis Tools window, not stored.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from s2p_trace_curation.annotations import annotation_ranges, ensure_annotations
from s2p_trace_curation.trace_processing import (
    TRACE_FIELD_NORM,
    TRACE_FIELDS,
    field_has_any,
    trace_field_is_stale,
)

PICKLE_SORT_ID = "pickle"
FIGURES_DIRNAME = "figures"

KIND_PLACEHOLDER = "placeholder"
KIND_HAC = "hac"
KIND_LABELS: dict[str, str] = {
    KIND_HAC: "HAC (hierarchical clustering)",
    KIND_PLACEHOLDER: "Placeholder (selected pickle order)",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def figures_dir(suite2p_dir: Path) -> Path:
    """Reserved snapshot folder: sibling of plane0/. Created lazily on first export."""
    return Path(suite2p_dir) / FIGURES_DIRNAME


def ensure_analyses(doc: dict[str, Any]) -> list[dict[str, Any]]:
    runs = doc.get("analyses")
    if runs is None:
        runs = []
        doc["analyses"] = runs
    meta = doc.setdefault("meta", {})
    if not meta.get("raster_sort"):
        meta["raster_sort"] = PICKLE_SORT_ID
    for run in runs:
        if "clusters" not in run:
            run["clusters"] = []
    return runs


def next_analysis_id(doc: dict[str, Any]) -> str:
    n = 0
    for run in ensure_analyses(doc):
        s = str(run.get("id", ""))
        if s.startswith("a-"):
            try:
                n = max(n, int(s[2:]))
            except ValueError:
                pass
    return f"a-{n + 1:03d}"


def kind_label(kind: str) -> str:
    return KIND_LABELS.get(str(kind), str(kind))


def current_iscell_ids(doc: dict[str, Any]) -> list[int]:
    return [int(r["roi_id"]) for r in doc["rois"] if bool(r.get("iscell", True))]


def get_analysis(doc: dict[str, Any], analysis_id: str) -> dict[str, Any] | None:
    aid = str(analysis_id)
    for run in ensure_analyses(doc):
        if str(run.get("id")) == aid:
            return run
    return None


def raster_sort_id(doc: dict[str, Any]) -> str:
    sid = str((doc.get("meta") or {}).get("raster_sort") or PICKLE_SORT_ID)
    if sid != PICKLE_SORT_ID and get_analysis(doc, sid) is None:
        return PICKLE_SORT_ID
    return sid


def set_raster_sort(doc: dict[str, Any], sort_id: str) -> None:
    doc.setdefault("meta", {})["raster_sort"] = str(sort_id)


def _led_spans(doc: dict[str, Any]) -> list[list[int]]:
    return sorted(
        r
        for a in ensure_annotations(doc)
        if str(a["property"]) == "LED+Shutter"
        for r in annotation_ranges(a)
    )


def analysis_input_sig(
    doc: dict[str, Any],
    roi_ids: list[int],
    kind: str,
    params: dict[str, Any],
) -> dict[str, Any]:
    by_id = {int(r["roi_id"]): r for r in doc["rois"]}
    field = str((params or {}).get("trace_field") or TRACE_FIELD_NORM)
    if field not in TRACE_FIELDS:
        field = TRACE_FIELD_NORM
    sums: list[float | None] = []
    for i in roi_ids:
        row = by_id.get(int(i))
        if row is None:
            sums.append(None)
            continue
        tr = row.get(field)
        if tr is None:
            sums.append(None)
        else:
            sums.append(float(np.nansum(np.asarray(tr, dtype=np.float64))))
    return {
        "kind": str(kind),
        "params": deepcopy(params),
        "ids": [int(i) for i in roi_ids],
        "trace_field": field,
        "tc_sums": sums,
        "led": _led_spans(doc),
    }


def _params_equal(a: Any, b: Any) -> bool:
    return deepcopy(a) == deepcopy(b)


def _sigs_equal(a: Any, b: Any) -> bool:
    if not isinstance(a, dict) or not isinstance(b, dict):
        return False
    if str(a.get("kind")) != str(b.get("kind")):
        return False
    field_a = str(a.get("trace_field") or TRACE_FIELD_NORM)
    field_b = str(b.get("trace_field") or TRACE_FIELD_NORM)
    if field_a != field_b:
        return False
    if not _params_equal(a.get("params") or {}, b.get("params") or {}):
        return False
    if [int(i) for i in (a.get("ids") or [])] != [int(i) for i in (b.get("ids") or [])]:
        return False
    if _led_list(a.get("led")) != _led_list(b.get("led")):
        return False
    sa = a.get("tc_sums") or []
    sb = b.get("tc_sums") or []
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


def _led_list(raw: Any) -> list[list[int]]:
    return [list(x) for x in (raw or [])]


def is_run_stale(doc: dict[str, Any], run: dict[str, Any]) -> bool:
    field = str((run.get("params") or {}).get("trace_field") or TRACE_FIELD_NORM)
    if field not in TRACE_FIELDS:
        field = TRACE_FIELD_NORM
    if not field_has_any(doc, field) or trace_field_is_stale(doc, field):
        return True
    stored_ids = [int(i) for i in (run.get("roi_ids") or [])]
    if stored_ids != current_iscell_ids(doc):
        return True
    current = analysis_input_sig(
        doc, stored_ids, str(run.get("kind", "")), run.get("params") or {}
    )
    return not _sigs_equal(run.get("input_sig"), current)


def refresh_stale_flags(doc: dict[str, Any]) -> int:
    """Recompute `stale` on each run. Returns how many are stale."""
    n = 0
    for run in ensure_analyses(doc):
        stale = is_run_stale(doc, run)
        run["stale"] = stale
        if stale:
            n += 1
    return n


def compute_run(
    doc: dict[str, Any], kind: str, params: dict[str, Any]
) -> tuple[list[int], list[int]]:
    """Return (roi_ids in pickle order, sort permutation). Matrices are not stored."""
    kind = str(kind)
    if kind == KIND_PLACEHOLDER:
        ids = current_iscell_ids(doc)
        return ids, list(ids)
    if kind == KIND_HAC:
        from s2p_trace_curation.hac import run_hac

        result = run_hac(doc, params)
        return result["roi_ids"], result["order"]
    raise ValueError(f"Unknown analysis kind: {kind}")


def normalize_clusters(raw: Any) -> list[list[int]]:
    """Coerce a run's ``clusters`` field to lists of roi_ids."""
    out: list[list[int]] = []
    if not raw:
        return out
    for group in raw:
        ids = [int(i) for i in (group or [])]
        if ids:
            out.append(ids)
    return out


def active_hac_clusters(doc: dict[str, Any]) -> list[list[int]]:
    """Clusters from the raster Sort run, if it is a saved HAC cut."""
    run = active_sort_run(doc)
    if run is None or str(run.get("kind")) != KIND_HAC:
        return []
    return normalize_clusters(run.get("clusters"))


def roi_cluster_index(clusters: list[list[int]]) -> dict[int, int]:
    """Map roi_id → 0-based cluster index (first membership wins)."""
    out: dict[int, int] = {}
    for i, group in enumerate(clusters):
        for rid in group:
            rid_i = int(rid)
            if rid_i not in out:
                out[rid_i] = i
    return out


def cluster_row_spans(
    row_ids: list[int], cluster_of: dict[int, int]
) -> list[tuple[int, int, int]]:
    """Contiguous (start, end, cluster_index) spans along a raster row order."""
    spans: list[tuple[int, int, int]] = []
    i = 0
    n = len(row_ids)
    while i < n:
        cid = cluster_of.get(int(row_ids[i]))
        if cid is None:
            i += 1
            continue
        j = i + 1
        while j < n and cluster_of.get(int(row_ids[j])) == cid:
            j += 1
        spans.append((i, j - 1, int(cid)))
        i = j
    return spans


def make_analysis_run(
    doc: dict[str, Any],
    *,
    label: str,
    kind: str,
    params: dict[str, Any],
    roi_ids: list[int],
    order: list[int],
    clusters: list[list[int]] | None = None,
    analysis_id: str | None = None,
) -> dict[str, Any]:
    now = _utc_now()
    params = deepcopy(params)
    roi_ids = [int(i) for i in roi_ids]
    order = [int(i) for i in order]
    kind = str(kind)
    return {
        "id": analysis_id or next_analysis_id(doc),
        "label": str(label).strip() or "Untitled",
        "kind": kind,
        "params": params,
        "roi_ids": roi_ids,
        "order": order,
        "clusters": normalize_clusters(clusters) if kind == KIND_HAC else [],
        "input_sig": analysis_input_sig(doc, roi_ids, kind, params),
        "stale": False,
        "created_utc": now,
        "updated_utc": now,
    }


def apply_run_result(
    run: dict[str, Any],
    doc: dict[str, Any],
    *,
    label: str | None = None,
    kind: str | None = None,
    params: dict[str, Any] | None = None,
    roi_ids: list[int],
    order: list[int],
    clusters: list[list[int]] | None = None,
) -> None:
    if label is not None:
        run["label"] = str(label).strip() or "Untitled"
    if kind is not None:
        run["kind"] = str(kind)
    if params is not None:
        run["params"] = deepcopy(params)
    run["roi_ids"] = [int(i) for i in roi_ids]
    run["order"] = [int(i) for i in order]
    if clusters is not None:
        run["clusters"] = normalize_clusters(clusters)
    elif str(run.get("kind") or "") != KIND_HAC:
        run["clusters"] = []
    run["input_sig"] = analysis_input_sig(
        doc, run["roi_ids"], str(run["kind"]), run.get("params") or {}
    )
    run["stale"] = False
    run["updated_utc"] = _utc_now()


def dropdown_label(run: dict[str, Any]) -> str:
    name = str(run.get("label") or run.get("id") or "Untitled")
    if run.get("stale"):
        return f"{name} (stale)"
    return name


def apply_raster_sort(
    rows: list[dict[str, Any]],
    run: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Reorder visible raster rows.

    Pickle (run is None): keep incoming order (caller already used pickle order).
    Saved run: still-selected members in `order`, then new selected cells in
    pickle order, then unselected (when present) in pickle order.
    """
    if run is None:
        return list(rows)
    by_id = {int(r["roi_id"]): r for r in rows}
    selected = [r for r in rows if bool(r.get("iscell", True))]
    unselected = [r for r in rows if not bool(r.get("iscell", True))]
    used: set[int] = set()
    out: list[dict[str, Any]] = []
    for i in (int(x) for x in (run.get("order") or [])):
        row = by_id.get(i)
        if row is None or not bool(row.get("iscell", True)):
            continue
        out.append(row)
        used.add(i)
    for r in selected:
        i = int(r["roi_id"])
        if i not in used:
            out.append(r)
            used.add(i)
    out.extend(unselected)
    return out


def active_sort_run(doc: dict[str, Any]) -> dict[str, Any] | None:
    sid = raster_sort_id(doc)
    if sid == PICKLE_SORT_ID:
        return None
    return get_analysis(doc, sid)

