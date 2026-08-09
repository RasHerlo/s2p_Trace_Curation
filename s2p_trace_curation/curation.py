"""Versioned trc_curation.pkl create / load / save / reset."""

from __future__ import annotations

import pickle
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from s2p_trace_curation import PICKLE_NAME, SCHEMA_VERSION
from s2p_trace_curation.suite2p_io import (
    PLANE_NAME,
    BinaryStack,
    load_iscell,
    load_ops,
    load_stat,
    load_traces,
    plane_dir,
    resolve_suite2p_dir,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def compute_trace_comp(F: np.ndarray, Fneu: np.ndarray, x: float) -> np.ndarray:
    return np.asarray(F, dtype=np.float64) - float(x) * np.asarray(Fneu, dtype=np.float64)


def pickle_path(suite2p_dir: Path) -> Path:
    return Path(suite2p_dir) / PICKLE_NAME


def create_curation_from_plane(suite2p_dir: Path) -> dict[str, Any]:
    suite2p_dir = resolve_suite2p_dir(suite2p_dir)
    plane = plane_dir(suite2p_dir)
    ops = load_ops(plane)
    stat = load_stat(plane)
    F, Fneu = load_traces(plane)
    n_roi = len(stat)
    if F.shape[0] != n_roi or Fneu.shape[0] != n_roi:
        raise ValueError(
            f"ROI count mismatch: stat={n_roi}, F={F.shape[0]}, Fneu={Fneu.shape[0]}"
        )
    iscell, iscell_prob = load_iscell(plane, n_roi)
    Ly = int(ops["Ly"])
    Lx = int(ops["Lx"])
    nframes = int(F.shape[1])

    rois: list[dict[str, Any]] = []
    for i in range(n_roi):
        s = stat[i]
        Fi = np.asarray(F[i], dtype=np.float64)
        Fni = np.asarray(Fneu[i], dtype=np.float64)
        x = 1.0
        rois.append(
            {
                "roi_id": int(i),
                "iscell": bool(iscell[i]),
                "iscell_prob": (
                    float(iscell_prob[i]) if iscell_prob is not None else None
                ),
                "roi": {
                    "ypix": np.asarray(s["ypix"], dtype=np.int32),
                    "xpix": np.asarray(s["xpix"], dtype=np.int32),
                    "lam": np.asarray(s["lam"], dtype=np.float32),
                    "F": Fi,
                    "modified": False,
                },
                "neuropil": {
                    "ipix": np.asarray(s["neuropil_mask"], dtype=np.int32),
                    "Fneu": Fni,
                    "modified": False,
                },
                "compensation": {
                    "x": x,
                    "trace_comp": compute_trace_comp(Fi, Fni, x),
                },
            }
        )

    doc: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "meta": {
            "plane": PLANE_NAME,
            "plane_relpath": PLANE_NAME,
            "created_utc": _utc_now(),
            "updated_utc": _utc_now(),
            "Ly": Ly,
            "Lx": Lx,
            "nframes": nframes,
            "fs": float(ops["fs"]) if "fs" in ops else None,
            "meanImg": np.asarray(ops.get("meanImg")),
            "meanImgE": np.asarray(ops.get("meanImgE")) if "meanImgE" in ops else None,
            "VCorr": np.asarray(ops.get("Vcorr", ops.get("VCorr")))
            if ("Vcorr" in ops or "VCorr" in ops)
            else None,
            "notes": "",
            "source_suite2p_abspath": str(suite2p_dir),
        },
        "rois": rois,
    }
    return doc


def save_curation(doc: dict[str, Any], suite2p_dir: Path) -> Path:
    suite2p_dir = Path(suite2p_dir)
    doc = deepcopy(doc)
    doc["meta"]["updated_utc"] = _utc_now()
    doc["schema_version"] = SCHEMA_VERSION
    out = pickle_path(suite2p_dir)
    with open(out, "wb") as f:
        pickle.dump(doc, f, protocol=pickle.HIGHEST_PROTOCOL)
    return out


def load_curation(path: Path) -> dict[str, Any]:
    path = Path(path)
    with open(path, "rb") as f:
        doc = pickle.load(f)
    if not isinstance(doc, dict) or "rois" not in doc or "meta" not in doc:
        raise ValueError(f"Invalid curation pickle: {path}")
    version = int(doc.get("schema_version", 0))
    if version != SCHEMA_VERSION:
        # Placeholder for future migrations
        if version > SCHEMA_VERSION:
            raise ValueError(
                f"Pickle schema_version {version} is newer than supported {SCHEMA_VERSION}"
            )
        doc["schema_version"] = SCHEMA_VERSION
    return doc


def open_suite2p_session(suite2p_dir: Path) -> tuple[Path, dict[str, Any], bool]:
    """
    Resolve suite2p folder. Load existing trc_curation.pkl or create it.
    Returns (suite2p_dir, doc, created_new).
    """
    suite2p_dir = resolve_suite2p_dir(suite2p_dir)
    pkl = pickle_path(suite2p_dir)
    if pkl.exists():
        return suite2p_dir, load_curation(pkl), False
    doc = create_curation_from_plane(suite2p_dir)
    save_curation(doc, suite2p_dir)
    return suite2p_dir, doc, True


def reset_roi_from_suite2p(
    doc: dict[str, Any], suite2p_dir: Path, roi_id: int
) -> dict[str, Any]:
    """Reset one ROI row from live plane0 suite2p files (Option A)."""
    suite2p_dir = resolve_suite2p_dir(suite2p_dir)
    plane = plane_dir(suite2p_dir)
    ops = load_ops(plane)
    stat = load_stat(plane)
    F, Fneu = load_traces(plane)
    iscell, iscell_prob = load_iscell(plane, len(stat))
    if roi_id < 0 or roi_id >= len(stat):
        raise IndexError(f"roi_id {roi_id} out of range")

    s = stat[roi_id]
    Fi = np.asarray(F[roi_id], dtype=np.float64)
    Fni = np.asarray(Fneu[roi_id], dtype=np.float64)
    x = 1.0
    row = {
        "roi_id": int(roi_id),
        "iscell": bool(iscell[roi_id]),
        "iscell_prob": float(iscell_prob[roi_id]) if iscell_prob is not None else None,
        "roi": {
            "ypix": np.asarray(s["ypix"], dtype=np.int32),
            "xpix": np.asarray(s["xpix"], dtype=np.int32),
            "lam": np.asarray(s["lam"], dtype=np.float32),
            "F": Fi,
            "modified": False,
        },
        "neuropil": {
            "ipix": np.asarray(s["neuropil_mask"], dtype=np.int32),
            "Fneu": Fni,
            "modified": False,
        },
        "compensation": {
            "x": x,
            "trace_comp": compute_trace_comp(Fi, Fni, x),
        },
    }

    # Keep meta images in sync if missing
    meta = doc["meta"]
    if meta.get("meanImg") is None and "meanImg" in ops:
        meta["meanImg"] = np.asarray(ops["meanImg"])

    for i, r in enumerate(doc["rois"]):
        if int(r["roi_id"]) == int(roi_id):
            doc["rois"][i] = row
            break
    else:
        doc["rois"].append(row)
        doc["rois"].sort(key=lambda r: int(r["roi_id"]))
    return doc


def set_compensation_x(row: dict[str, Any], x: float) -> None:
    row["compensation"]["x"] = float(x)
    row["compensation"]["trace_comp"] = compute_trace_comp(
        row["roi"]["F"], row["neuropil"]["Fneu"], x
    )


def reextract_after_mask_edit(
    row: dict[str, Any], suite2p_dir: Path, *, roi_changed: bool, neuropil_changed: bool
) -> None:
    """Recompute F and/or Fneu from data.bin after mask edits; refresh trace_comp."""
    plane = plane_dir(resolve_suite2p_dir(suite2p_dir))
    with BinaryStack(plane) as stack:
        if roi_changed:
            row["roi"]["F"] = stack.extract_roi_trace(
                row["roi"]["ypix"], row["roi"]["xpix"], row["roi"]["lam"]
            )
            row["roi"]["modified"] = True
        if neuropil_changed:
            row["neuropil"]["Fneu"] = stack.extract_neuropil_trace(row["neuropil"]["ipix"])
            row["neuropil"]["modified"] = True
    set_compensation_x(row, float(row["compensation"]["x"]))
