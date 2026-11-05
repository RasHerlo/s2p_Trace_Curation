"""Shared ROI sets (masters) reused across experiments recorded in one FOV.

A master (``*.roiset.pkl``) holds geometry and curation intent in its own
reference frame: masks, default ``iscell``, ``x``, baseline-offset on/off,
BG-ROI masks, and SG / bleach mode. Traces are never shared — applying a
master to an experiment shifts the masks into that experiment's frame and
re-extracts every trace from its own ``data.bin``.

Shift convention: ``(dy, dx)`` maps master pixels to experiment pixels,
``y_exp = y_master + dy``. Shifts are integers so masks stay exact.
"""

from __future__ import annotations

import os
import pickle
import re
import shutil
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

from s2p_trace_curation import PICKLE_NAME
from s2p_trace_curation.suite2p_io import (
    enhanced_mean_from_ops,
    enhanced_mean_image,
    extract_traces_batch,
    fov_images_from_ops,
    load_ops,
    load_stat,
    plane_dir,
    resolve_suite2p_dir,
)

ROISET_KIND = "trc_roiset"
ROISET_SCHEMA_VERSION = 1
ROISET_SUFFIX = ".roiset.pkl"
LOCAL_ID_BASE = 10000
ORIGINAL_PICKLE_NAME = "trc_curation.original.pkl"
BACKUP_PICKLE_NAME = "trc_curation.bak.pkl"
IMAGE_KEYS = ("meanImg", "meanImgE", "VCorr")
ORIGIN_MASTER = "master"
ORIGIN_OWN = "own"
# A ghost overlapping a working ROI less than this is drawn as a candidate.
CANDIDATE_IOU = 0.2

_TP_SHARED_KEYS = ("sg_window", "sg_poly", "bleach_enabled", "tau_mode")


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


# ------------------------------------------------------------------ paths
def _relpath(target: Path, start_dir: Path) -> str:
    """Relative path when possible (same drive), else absolute."""
    try:
        return os.path.relpath(Path(target).resolve(), Path(start_dir).resolve())
    except ValueError:
        return str(Path(target).resolve())


def _resolve(base_dir: Path, raw: str) -> Path:
    p = Path(raw)
    if not p.is_absolute():
        p = Path(base_dir) / p
    return p.resolve()


def _same_path(a: Path, b: Path) -> bool:
    return os.path.normcase(str(Path(a).resolve())) == os.path.normcase(str(Path(b).resolve()))


def _name_token(text: str, capitalize: bool = False) -> str:
    words = [w for w in re.split(r"[^0-9A-Za-z]+", text) if w]
    if capitalize:
        words = [w[:1].upper() + w[1:] for w in words]
    return "_".join(words)


def default_master_name(folder: Path, channel: str = "") -> str:
    """``<parent>_<folder>_<channel>``, e.g. ``F1_RV_pdf_New_Recordings_PFC_ChanA``.

    The channel suffix keeps the one-master-per-channel files apart in one folder.
    """
    p = Path(folder)
    tokens = [_name_token(p.parent.name), _name_token(p.name, capitalize=True), _name_token(channel)]
    return "_".join(t for t in tokens if t) or "roiset"


def default_channel(suite2p_dir: Path) -> str:
    parts = Path(suite2p_dir).resolve().parts
    return parts[-2] if len(parts) >= 2 else ""


# ------------------------------------------------------------- master I/O
def load_master(path: Path) -> dict[str, Any]:
    with open(path, "rb") as f:
        master = pickle.load(f)
    if not isinstance(master, dict) or master.get("kind") != ROISET_KIND:
        raise ValueError(f"Not an ROI-set master: {path}")
    version = int(master.get("schema_version", 0))
    if version > ROISET_SCHEMA_VERSION:
        raise ValueError(
            f"ROI-set schema_version {version} is newer than supported {ROISET_SCHEMA_VERSION}"
        )
    master.setdefault("rois", [])
    master.setdefault("bg_rois", [])
    master.setdefault("experiments", [])
    for entry in master["experiments"]:
        entry.setdefault("approved", False)
        entry.setdefault("tag", "")
        entry.setdefault("shift", [0, 0])
        entry.setdefault("score", None)
        entry.setdefault("image_key", "meanImg")
        entry.setdefault("synced_rev", None)
    return master


def save_master(master: dict[str, Any], path: Path) -> Path:
    """Atomic write; the previous file is kept as ``<name>.bak``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = deepcopy(master)
    out["meta"]["updated_utc"] = _utc_now()
    out["schema_version"] = ROISET_SCHEMA_VERSION
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
    if path.exists():
        shutil.copy2(path, path.with_name(path.name + ".bak"))
    os.replace(tmp, path)
    master["meta"]["updated_utc"] = out["meta"]["updated_utc"]
    return path


def update_master_on_disk(
    path: Path, fn: Callable[[dict[str, Any]], None]
) -> dict[str, Any]:
    """Re-read the master, apply ``fn``, save. Keeps concurrent registry edits."""
    master = load_master(path)
    fn(master)
    save_master(master, path)
    return master


# --------------------------------------------------------------- registry
def find_experiment(
    master: dict[str, Any], master_path: Path, suite2p_dir: Path
) -> dict[str, Any] | None:
    base = Path(master_path).parent
    for entry in master.get("experiments") or []:
        if _same_path(_resolve(base, str(entry["path"])), suite2p_dir):
            return entry
    return None


def upsert_experiment(
    master: dict[str, Any], master_path: Path, suite2p_dir: Path, **fields: Any
) -> dict[str, Any]:
    entry = find_experiment(master, master_path, suite2p_dir)
    if entry is None:
        entry = {
            "path": _relpath(suite2p_dir, Path(master_path).parent),
            "approved": False,
            "tag": "",
            "shift": [0, 0],
            "score": None,
            "image_key": "meanImg",
            "synced_rev": None,
            "added_utc": _utc_now(),
        }
        master["experiments"].append(entry)
        master["experiments"].sort(key=lambda e: str(e["path"]).casefold())
    entry.update(fields)
    return entry


def experiment_dir(master_path: Path, entry: dict[str, Any]) -> Path:
    return _resolve(Path(master_path).parent, str(entry["path"]))


def _preferred_suite2p(dirs: list[Path]) -> Path:
    """One suite2p folder per channel folder: curated, then merged, then name order."""
    def rank(p: Path) -> tuple[int, str]:
        if (p / PICKLE_NAME).exists():
            return (0, p.name)
        if p.name == "suite2p_merged":
            return (1, p.name)
        return (2, p.name)

    return sorted(dirs, key=rank)[0]


def scan_suite2p_folders(root: Path) -> list[Path]:
    """suite2p folders under ``root`` (one per parent folder; see ``_preferred_suite2p``)."""
    by_parent: dict[Path, list[Path]] = {}
    for dirpath, dirnames, _files in os.walk(root):
        here = Path(dirpath)
        if here.name.startswith("plane"):
            dirnames[:] = []
            continue
        if (here / "plane0" / "ops.npy").exists():
            by_parent.setdefault(here.parent, []).append(here)
            dirnames[:] = []
    picked = [_preferred_suite2p(v) for v in by_parent.values()]
    picked.sort(key=lambda p: str(p).casefold())
    return picked


# ----------------------------------------------------------- registration
def _highpass(img: np.ndarray, sigma: float = 8.0) -> np.ndarray:
    from scipy.ndimage import gaussian_filter

    a = np.nan_to_num(np.asarray(img, dtype=np.float64))
    return a - gaussian_filter(a, sigma)


def overlap_score(ref: np.ndarray, target: np.ndarray, dy: int, dx: int) -> float | None:
    """Pearson r of high-passed images on the overlap after shifting ref by (dy, dx)."""
    a = _highpass(ref)
    b = _highpass(target)
    Ly, Lx = a.shape
    ry = slice(max(0, -dy), min(Ly, Ly - dy))
    rx = slice(max(0, -dx), min(Lx, Lx - dx))
    ty = slice(max(0, dy), min(Ly, Ly + dy))
    tx = slice(max(0, dx), min(Lx, Lx + dx))
    pa = a[ry, rx].ravel()
    pb = b[ty, tx].ravel()
    if pa.size < 16:
        return None
    pa = pa - pa.mean()
    pb = pb - pb.mean()
    denom = float(np.sqrt((pa * pa).sum() * (pb * pb).sum()))
    if denom <= 0:
        return None
    return float((pa * pb).sum() / denom)


def estimate_shift(
    ref: np.ndarray, target: np.ndarray, max_shift: int | None = None
) -> tuple[int, int, float | None]:
    """Integer ``(dy, dx)`` with ``target(y + dy) ≈ ref(y)``, plus an overlap score."""
    ref = np.asarray(ref, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if ref.shape != target.shape:
        raise ValueError(f"Image shapes differ: {ref.shape} vs {target.shape}")
    Ly, Lx = ref.shape
    if max_shift is None:
        max_shift = min(Ly, Lx) // 4
    win = np.outer(np.hanning(Ly), np.hanning(Lx))

    def prep(img: np.ndarray) -> np.ndarray:
        a = _highpass(img)
        a = a - a.mean()
        s = a.std()
        if s > 0:
            a = a / s
        return a * win

    fr = np.fft.fft2(prep(ref))
    ft = np.fft.fft2(prep(target))
    cross = ft * np.conj(fr)
    cross /= np.abs(cross) + 1e-12
    corr = np.fft.ifft2(cross).real
    corr = np.fft.fftshift(corr)
    cy, cx = Ly // 2, Lx // 2
    yy, xx = np.ogrid[:Ly, :Lx]
    outside = (np.abs(yy - cy) > max_shift) | (np.abs(xx - cx) > max_shift)
    corr[outside] = -np.inf
    py, px = np.unravel_index(int(np.argmax(corr)), corr.shape)
    dy, dx = int(py - cy), int(px - cx)
    return dy, dx, overlap_score(ref, target, dy, dx)


def _full_fov_images_from_doc_meta(meta: dict[str, Any]) -> dict[str, np.ndarray | None]:
    return {k: (None if meta.get(k) is None else np.asarray(meta[k])) for k in IMAGE_KEYS}


def master_reference_image(master: dict[str, Any], key: str) -> np.ndarray | None:
    refs = (master.get("meta") or {}).get("ref_images") or {}
    img = refs.get(key)
    if img is None and key == "meanImgE" and refs.get("meanImg") is not None:
        meta = master["meta"]
        img = enhanced_mean_image(np.asarray(refs["meanImg"]), int(meta["Ly"]), int(meta["Lx"]))
        refs["meanImgE"] = img
    if img is None:
        img = refs.get("meanImg")
    return None if img is None else np.asarray(img)


def experiment_image(
    suite2p_dir: Path, key: str, doc: dict[str, Any] | None = None
) -> np.ndarray | None:
    """Registration image for an experiment (doc cache first, else ops.npy)."""
    if doc is not None and doc.get("meta", {}).get(key) is not None:
        return np.asarray(doc["meta"][key])
    ops = load_ops(plane_dir(resolve_suite2p_dir(suite2p_dir)))
    imgs = fov_images_from_ops(ops)
    img = imgs.get(key)
    if img is None and key == "meanImgE":
        img = enhanced_mean_from_ops(ops)
    if img is None:
        img = imgs.get("meanImg")
    return img


def register_experiment(
    master: dict[str, Any], suite2p_dir: Path, key: str = "meanImg", doc: dict[str, Any] | None = None
) -> tuple[int, int, float | None]:
    ref = master_reference_image(master, key)
    tgt = experiment_image(suite2p_dir, key, doc)
    if ref is None or tgt is None:
        raise ValueError("Missing registration image")
    return estimate_shift(ref, tgt)


# ------------------------------------------------------------- shifting
def shift_yx(
    ypix: np.ndarray, xpix: np.ndarray, dy: int, dx: int, Ly: int, Lx: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Shifted pixel coords plus the boolean keep-mask of pixels still in the FOV."""
    y = np.asarray(ypix, dtype=np.int64) + int(dy)
    x = np.asarray(xpix, dtype=np.int64) + int(dx)
    keep = (y >= 0) & (y < Ly) & (x >= 0) & (x < Lx)
    return y[keep].astype(np.int32), x[keep].astype(np.int32), keep


def shift_ipix(ipix: np.ndarray, dy: int, dx: int, Ly: int, Lx: int) -> np.ndarray:
    ipix = np.asarray(ipix, dtype=np.int64)
    if ipix.size == 0:
        return ipix.astype(np.int32)
    y, x = np.unravel_index(ipix, (Ly, Lx))
    ys, xs, _ = shift_yx(y, x, dy, dx, Ly, Lx)
    return np.ravel_multi_index((ys, xs), (Ly, Lx)).astype(np.int32)


def shifted_master_masks(
    master: dict[str, Any], dy: int, dx: int
) -> list[tuple[int, np.ndarray, np.ndarray]]:
    """``(uid, ypix, xpix)`` of master cell ROIs in experiment coords (preview)."""
    Ly, Lx = int(master["meta"]["Ly"]), int(master["meta"]["Lx"])
    out = []
    for m in master["rois"]:
        y, x, _ = shift_yx(m["ypix"], m["xpix"], dy, dx, Ly, Lx)
        out.append((int(m["uid"]), y, x))
    return out


# --------------------------------------------------------------- publish
def _remap_ids(ids: Any, uid_map: dict[int, int]) -> list[int]:
    return [uid_map[int(i)] for i in (ids or []) if int(i) in uid_map]


def _suite2p_overlap(row: dict[str, Any], stat: Any) -> np.ndarray | None:
    """suite2p's per-pixel ``overlap`` flags when the row still has the stat.npy mask.

    suite2p (``allow_overlap=False``) leaves shared pixels out of F; carrying the
    flags keeps re-extracted traces equal to F.npy. Edited masks get None.
    """
    if stat is None or bool(row["roi"].get("modified", False)):
        return None
    rid = int(row["roi_id"])
    if rid < 0 or rid >= len(stat):
        return None
    s = stat[rid]
    ov = s.get("overlap") if hasattr(s, "get") else None
    if ov is None:
        return None
    ypix = np.asarray(row["roi"]["ypix"])
    xpix = np.asarray(row["roi"]["xpix"])
    if not (np.array_equal(ypix, np.asarray(s["ypix"])) and np.array_equal(xpix, np.asarray(s["xpix"]))):
        return None
    ov = np.asarray(ov, dtype=bool)
    if not ov.any() or ov.all():
        return None
    return ov.copy()


def master_from_doc(
    doc: dict[str, Any],
    suite2p_dir: Path,
    master_path: Path,
    *,
    name: str,
    channel: str,
    tag: str = "",
) -> tuple[dict[str, Any], dict[int, int]]:
    """New master from the iscell=True ROIs of ``doc``. Returns ``(master, roi_id→uid)``."""
    from s2p_trace_curation.bg_rois import ensure_bg_rois
    from s2p_trace_curation.curation import baseline_offset_enabled, ensure_compensation
    from s2p_trace_curation.trace_processing import ensure_trace_processing

    meta = doc["meta"]
    cells = [r for r in doc["rois"] if bool(r.get("iscell", True))]
    if not cells:
        raise ValueError("No iscell=True ROIs to publish")
    low_ids = [int(r["roi_id"]) for r in doc["rois"] if int(r["roi_id"]) < LOCAL_ID_BASE]
    next_uid = (max(low_ids) + 1) if low_ids else 0
    uid_map: dict[int, int] = {}
    for r in cells:
        rid = int(r["roi_id"])
        if rid < LOCAL_ID_BASE:
            uid_map[rid] = rid
        else:
            uid_map[rid] = next_uid
            next_uid += 1
    if next_uid >= LOCAL_ID_BASE:
        raise ValueError(f"Master uids must stay below {LOCAL_ID_BASE}")

    src_rel = _relpath(suite2p_dir, Path(master_path).parent)
    stat = None
    try:
        stat = load_stat(plane_dir(resolve_suite2p_dir(suite2p_dir)))
    except (FileNotFoundError, OSError):
        pass
    rois = []
    for r in cells:
        comp = ensure_compensation(r)
        rois.append(
            {
                "uid": uid_map[int(r["roi_id"])],
                "rev": 1,
                "iscell": True,
                "ypix": np.asarray(r["roi"]["ypix"], dtype=np.int32).copy(),
                "xpix": np.asarray(r["roi"]["xpix"], dtype=np.int32).copy(),
                "lam": np.asarray(r["roi"]["lam"], dtype=np.float32).copy(),
                "neuropil_ipix": np.asarray(r["neuropil"]["ipix"], dtype=np.int32).copy(),
                "x": float(comp["x"]),
                "baseline_offset": bool(baseline_offset_enabled(r)),
                "overlap": _suite2p_overlap(r, stat),
                "source": {"experiment": src_rel, "roi_id": int(r["roi_id"])},
            }
        )
    rois.sort(key=lambda m: int(m["uid"]))
    bg_rows = []
    for b in ensure_bg_rois(doc):
        bg_rows.append(
            {
                "uid": int(b["bg_id"]),
                "rev": 1,
                "ypix": np.asarray(b["ypix"], dtype=np.int32).copy(),
                "xpix": np.asarray(b["xpix"], dtype=np.int32).copy(),
            }
        )
    tp = ensure_trace_processing(doc)
    now = _utc_now()
    master = {
        "kind": ROISET_KIND,
        "schema_version": ROISET_SCHEMA_VERSION,
        "meta": {
            "name": str(name),
            "channel": str(channel),
            "rev": 1,
            "created_utc": now,
            "updated_utc": now,
            "Ly": int(meta["Ly"]),
            "Lx": int(meta["Lx"]),
            "source": src_rel,
            "ref_images": _full_fov_images_from_doc_meta(meta),
            "trace_processing": {k: tp.get(k) for k in _TP_SHARED_KEYS},
            "next_uid": int(next_uid),
            "notes": "",
        },
        "rois": rois,
        "bg_rois": bg_rows,
        "experiments": [],
    }
    upsert_experiment(
        master,
        master_path,
        suite2p_dir,
        approved=True,
        tag=str(tag),
        shift=[0, 0],
        score=1.0,
        image_key="meanImg",
        synced_rev=1,
    )
    return master, uid_map


def backup_experiment_pickle(suite2p_dir: Path) -> None:
    """Keep the first pickle as the original (ghost source) and the latest as ``.bak``."""
    pkl = Path(suite2p_dir) / PICKLE_NAME
    if not pkl.exists():
        return
    original = Path(suite2p_dir) / ORIGINAL_PICKLE_NAME
    if not original.exists():
        shutil.copy2(pkl, original)
    shutil.copy2(pkl, Path(suite2p_dir) / BACKUP_PICKLE_NAME)


def _link_meta(
    master: dict[str, Any],
    master_path: Path,
    suite2p_dir: Path,
    dy: int,
    dx: int,
    score: float | None,
    image_key: str,
) -> dict[str, Any]:
    return {
        "path": _relpath(master_path, suite2p_dir),
        "abs_hint": str(Path(master_path).resolve()),
        "name": str(master["meta"]["name"]),
        "channel": str(master["meta"].get("channel") or ""),
        "rev": int(master["meta"]["rev"]),
        "shift": [int(dy), int(dx)],
        "score": None if score is None else float(score),
        "image_key": str(image_key),
        "applied_utc": _utc_now(),
    }


def link_source_doc(
    doc: dict[str, Any],
    suite2p_dir: Path,
    master: dict[str, Any],
    master_path: Path,
    uid_map: dict[int, int],
) -> None:
    """Make the publishing experiment a member (shift 0; traces kept, no re-extraction)."""
    from s2p_trace_curation.analyses import ensure_analyses

    rows = []
    for r in doc["rois"]:
        rid = int(r["roi_id"])
        if rid not in uid_map:
            continue
        uid = uid_map[rid]
        r["roi_id"] = uid
        r["link"] = {"uid": uid, "rev": 1, "origin": ORIGIN_MASTER}
        rows.append(r)
    rows.sort(key=lambda r: int(r["roi_id"]))
    doc["rois"] = rows
    for run in ensure_analyses(doc):
        run["roi_ids"] = _remap_ids(run.get("roi_ids"), uid_map)
        run["order"] = _remap_ids(run.get("order"), uid_map)
        run["clusters"] = [
            c for c in (_remap_ids(cl, uid_map) for cl in run.get("clusters") or []) if c
        ]
    doc["master_link"] = _link_meta(master, master_path, suite2p_dir, 0, 0, 1.0, "meanImg")


# ----------------------------------------------------------------- apply
def _build_master_rows(
    master: dict[str, Any], dy: int, dx: int, nframes: int, uids: set[int] | None = None
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, list[int]]]:
    """Shifted ROI / BG rows with empty traces, plus a clip report."""
    Ly, Lx = int(master["meta"]["Ly"]), int(master["meta"]["Lx"])
    z = np.zeros(int(nframes), dtype=np.float64)
    rows: list[dict[str, Any]] = []
    report: dict[str, list[int]] = {"clipped": [], "dropped": [], "bg_dropped": []}
    for m in master["rois"]:
        uid = int(m["uid"])
        if uids is not None and uid not in uids:
            continue
        y, x, keep = shift_yx(m["ypix"], m["xpix"], dy, dx, Ly, Lx)
        if y.size == 0:
            report["dropped"].append(uid)
            continue
        if not keep.all():
            report["clipped"].append(uid)
        lam = np.asarray(m["lam"], dtype=np.float32)[keep]
        ov = m.get("overlap")
        ov = None if ov is None else np.asarray(ov, dtype=bool)[keep]
        rows.append(
            {
                "roi_id": uid,
                "iscell": bool(m.get("iscell", True)),
                "iscell_prob": None,
                "roi": {"ypix": y, "xpix": x, "lam": lam, "F": z.copy(), "modified": False},
                "neuropil": {
                    "ipix": shift_ipix(m["neuropil_ipix"], dy, dx, Ly, Lx),
                    "Fneu": z.copy(),
                    "modified": False,
                },
                "compensation": {
                    "x": float(m.get("x", 1.0)),
                    "fneu_offset": 0.0,
                    "baseline_offset": bool(m.get("baseline_offset", False)),
                    "trace_comp": z.copy(),
                },
                "link": {"uid": uid, "rev": int(m.get("rev", 1)), "origin": ORIGIN_MASTER},
                "_overlap": ov,
            }
        )
    bg_rows: list[dict[str, Any]] = []
    if uids is None:
        for b in master.get("bg_rois") or []:
            y, x, _ = shift_yx(b["ypix"], b["xpix"], dy, dx, Ly, Lx)
            if y.size == 0:
                report["bg_dropped"].append(int(b["uid"]))
                continue
            bg_rows.append(
                {
                    "bg_id": int(b["uid"]),
                    "ypix": y,
                    "xpix": x,
                    "F": z.copy(),
                    "F_sm": None,
                    "F_sm_bc": None,
                    "bleach": None,
                    "modified": False,
                    "link": {"uid": int(b["uid"]), "rev": int(b.get("rev", 1)), "origin": ORIGIN_MASTER},
                }
            )
    return rows, bg_rows, report


def master_baseline_rows(
    master: dict[str, Any], dy: int, dx: int, uids: set[int] | None = None
) -> dict[int, dict[str, Any]]:
    """Master ROIs in this experiment's frame, by uid, with no real traces.

    Used to tell which parts of a linked ROI differ from the shared version.
    """
    rows, _bg, _rep = _build_master_rows(master, dy, dx, 1, uids=uids)
    return {int(r["roi_id"]): r for r in rows}


def _extract_rows(
    suite2p_dir: Path,
    rows: list[dict[str, Any]],
    bg_rows: list[dict[str, Any]],
    Ly: int,
    Lx: int,
    progress: Callable[[int, int], None] | None,
    should_cancel: Callable[[], bool] | None,
) -> None:
    sets: list[tuple[np.ndarray, np.ndarray | None]] = []
    for r in rows:
        ipix = np.ravel_multi_index(
            (np.asarray(r["roi"]["ypix"], dtype=np.int64), np.asarray(r["roi"]["xpix"], dtype=np.int64)),
            (Ly, Lx),
        )
        lam = np.asarray(r["roi"]["lam"], dtype=np.float64)
        ov = r.pop("_overlap", None)
        if ov is not None and ov.shape == ipix.shape and not ov.all():
            ipix, lam = ipix[~ov], lam[~ov]
        sets.append((ipix, lam))
        sets.append((r["neuropil"]["ipix"], None))
    for b in bg_rows:
        ipix = np.ravel_multi_index(
            (np.asarray(b["ypix"], dtype=np.int64), np.asarray(b["xpix"], dtype=np.int64)), (Ly, Lx)
        )
        sets.append((ipix, None))
    if not sets:
        return
    traces = extract_traces_batch(
        plane_dir(resolve_suite2p_dir(suite2p_dir)),
        sets,
        progress=progress,
        should_cancel=should_cancel,
    )
    k = 0
    for r in rows:
        r["roi"]["F"] = traces[k]
        r["neuropil"]["Fneu"] = traces[k + 1]
        k += 2
    for b in bg_rows:
        b["F"] = traces[k]
        k += 1


def _finish_rows(doc: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    """trace_comp, then the master's baseline-offset intent (needs this doc's LED spans)."""
    from s2p_trace_curation.curation import apply_baseline_fneu_offset, refresh_trace_comp
    from s2p_trace_curation.raster import led_shutter_nan_mask

    mask = led_shutter_nan_mask(doc, int(doc["meta"]["nframes"]))
    for r in rows:
        refresh_trace_comp(r)
        if r["compensation"].get("baseline_offset"):
            try:
                apply_baseline_fneu_offset(r, mask)
            except ValueError:
                r["compensation"]["baseline_offset"] = False


def apply_master(
    doc: dict[str, Any],
    suite2p_dir: Path,
    master: dict[str, Any],
    master_path: Path,
    *,
    dy: int,
    dx: int,
    score: float | None,
    image_key: str,
    progress: Callable[[int, int], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> dict[str, list[int]]:
    """Replace the working ROI set of ``doc`` with the master, shifted by ``(dy, dx)``.

    Annotations, heatmaps and meta images stay. Analyses are cleared (the roi_ids
    changed). BG ROIs are replaced when the master has any. ``doc`` is only
    modified once extraction has succeeded.
    """
    from s2p_trace_curation.analyses import PICKLE_SORT_ID
    from s2p_trace_curation.bg_rois import ensure_bg_rois
    from s2p_trace_curation.raster import rebuild_all_tc_norm
    from s2p_trace_curation.trace_processing import (
        ensure_trace_processing,
        rebuild_all_tc_norm_sm,
        rebuild_all_tc_norm_sm_bc,
    )

    meta = doc["meta"]
    Ly, Lx = int(meta["Ly"]), int(meta["Lx"])
    if (Ly, Lx) != (int(master["meta"]["Ly"]), int(master["meta"]["Lx"])):
        raise ValueError(
            f"FOV size differs: experiment {Ly}×{Lx}, master "
            f"{master['meta']['Ly']}×{master['meta']['Lx']}"
        )
    nframes = int(meta["nframes"])
    rows, bg_rows, report = _build_master_rows(master, dy, dx, nframes)
    if not rows:
        raise ValueError("No master ROI lands inside this FOV at that shift")
    _extract_rows(suite2p_dir, rows, bg_rows, Ly, Lx, progress, should_cancel)

    doc["rois"] = rows
    if master.get("bg_rois"):
        doc["bg_rois"] = bg_rows
    else:
        ensure_bg_rois(doc)
    doc["analyses"] = []
    meta["raster_sort"] = PICKLE_SORT_ID
    tp = ensure_trace_processing(doc)
    for k, v in ((master["meta"].get("trace_processing") or {})).items():
        if k in _TP_SHARED_KEYS and v is not None:
            tp[k] = v
    tp["shared_tau1"] = None
    tp["shared_tau2"] = None
    _finish_rows(doc, rows)
    rebuild_all_tc_norm(doc)
    rebuild_all_tc_norm_sm(doc)
    rebuild_all_tc_norm_sm_bc(doc)
    doc["master_link"] = _link_meta(master, master_path, suite2p_dir, dy, dx, score, image_key)
    return report


def rederive_row(
    doc: dict[str, Any], suite2p_dir: Path, master: dict[str, Any], uid: int
) -> dict[str, Any]:
    """Fresh copy of one master ROI for this experiment (Reset ROI on linked docs)."""
    from s2p_trace_curation.trace_processing import rebuild_row_traces

    link = doc.get("master_link") or {}
    dy, dx = (int(v) for v in (link.get("shift") or [0, 0]))
    meta = doc["meta"]
    rows, _bg, _rep = _build_master_rows(master, dy, dx, int(meta["nframes"]), uids={int(uid)})
    if not rows:
        raise KeyError(f"uid {uid} is not in the master (or falls outside this FOV)")
    _extract_rows(suite2p_dir, rows, [], int(meta["Ly"]), int(meta["Lx"]), None, None)
    _finish_rows(doc, rows)
    row = rows[0]
    rebuild_row_traces(doc, row)
    return row


# ---------------------------------------------------------- link status
def master_path_for_doc(doc: dict[str, Any], suite2p_dir: Path) -> Path | None:
    """Resolved master file of a linked doc, or None (unlinked or missing on disk)."""
    link = doc.get("master_link")
    if not link:
        return None
    p = _resolve(Path(suite2p_dir), str(link.get("path") or ""))
    if p.is_file():
        return p
    hint = link.get("abs_hint")
    if hint and Path(hint).is_file():
        return Path(hint)
    return None


def link_summary(doc: dict[str, Any]) -> str:
    link = doc.get("master_link")
    if not link:
        return "not linked"
    dy, dx = (int(v) for v in (link.get("shift") or [0, 0]))
    return f"{link.get('name', '?')} · rev {link.get('rev', '?')} · shift ({dy:+d}, {dx:+d})"


# ------------------------------------------------------------ local ids
def next_local_roi_id(doc: dict[str, Any]) -> int:
    ids = [int(r["roi_id"]) for r in doc.get("rois") or [] if int(r["roi_id"]) >= LOCAL_ID_BASE]
    return max(ids) + 1 if ids else LOCAL_ID_BASE


# ---------------------------------------------------------------- ghosts
def load_ghost_rows(suite2p_dir: Path) -> tuple[list[dict[str, Any]], str]:
    """Original ROIs of this experiment and where they came from."""
    from s2p_trace_curation.curation import create_curation_from_plane, load_curation

    original = Path(suite2p_dir) / ORIGINAL_PICKLE_NAME
    if original.exists():
        return load_curation(original)["rois"], ORIGINAL_PICKLE_NAME
    return create_curation_from_plane(Path(suite2p_dir))["rois"], "plane0/stat.npy"


def ghost_candidates(
    ghosts: list[dict[str, Any]], rois: list[dict[str, Any]], Ly: int, Lx: int
) -> np.ndarray:
    """Bool per ghost: max IoU against any working ROI is below ``CANDIDATE_IOU``."""
    labels = np.full((Ly, Lx), -1, dtype=np.int64)
    areas: list[int] = []
    for k, r in enumerate(rois):
        y = np.asarray(r["roi"]["ypix"], dtype=np.int64)
        x = np.asarray(r["roi"]["xpix"], dtype=np.int64)
        labels[y, x] = k
        areas.append(int(y.size))
    area_arr = np.asarray(areas, dtype=np.int64)
    out = np.ones(len(ghosts), dtype=bool)
    for i, g in enumerate(ghosts):
        y = np.asarray(g["roi"]["ypix"], dtype=np.int64)
        x = np.asarray(g["roi"]["xpix"], dtype=np.int64)
        if y.size == 0:
            continue
        hit = labels[y, x]
        hit = hit[hit >= 0]
        if hit.size == 0:
            continue
        ks, counts = np.unique(hit, return_counts=True)
        iou = counts / (y.size + area_arr[ks] - counts)
        out[i] = bool(iou.max() < CANDIDATE_IOU)
    return out


def import_ghost(doc: dict[str, Any], ghost: dict[str, Any]) -> dict[str, Any]:
    """Copy an original ROI into the working set as a local ROI (traces carried over)."""
    from s2p_trace_curation.curation import append_roi, ensure_compensation
    from s2p_trace_curation.trace_processing import rebuild_row_traces

    row = deepcopy(ghost)
    for key in ("tc_norm", "tc_norm_sm", "tc_norm_sm_bc", "bleach", "edited_utc"):
        row.pop(key, None)
    src_id = int(row["roi_id"])
    row["roi_id"] = next_local_roi_id(doc)
    row["iscell"] = True
    row["link"] = {"uid": None, "rev": None, "origin": ORIGIN_OWN, "source_roi_id": src_id}
    ensure_compensation(row)
    rebuild_row_traces(doc, row)
    append_roi(doc, row)
    return row
