"""Merge two suite2p folders that share an identical data.bin into one."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Literal

import numpy as np

from s2p_trace_curation.suite2p_io import (
    load_iscell,
    load_ops,
    load_stat,
    load_traces,
    open_data_bin,
    plane_dir,
    resolve_suite2p_dir,
)

BinCheckMode = Literal["sample", "full"]
SAMPLE_CHUNK = 1024 * 1024  # 1 MiB
FULL_CHUNK = 8 * 1024 * 1024


class MergeError(ValueError):
    """User-facing merge failure."""


@dataclass(frozen=True)
class BinCompareResult:
    identical: bool
    mode: BinCheckMode
    size_a: int
    size_b: int
    detail: str


def compare_data_bins(
    path_a: Path, path_b: Path, mode: BinCheckMode = "sample"
) -> BinCompareResult:
    """Compare two data.bin files. ``sample`` = size + first/mid/last 1 MiB SHA-256."""
    path_a = Path(path_a)
    path_b = Path(path_b)
    if not path_a.is_file():
        raise MergeError(f"Missing data.bin: {path_a}")
    if not path_b.is_file():
        raise MergeError(f"Missing data.bin: {path_b}")

    size_a = path_a.stat().st_size
    size_b = path_b.stat().st_size
    if size_a != size_b:
        return BinCompareResult(
            False,
            mode,
            size_a,
            size_b,
            f"Size mismatch: {size_a} vs {size_b} bytes",
        )

    if mode == "sample":
        da = _sample_digest(path_a, size_a)
        db = _sample_digest(path_b, size_b)
        ok = da == db
        return BinCompareResult(
            ok,
            mode,
            size_a,
            size_b,
            "Sample digest match (first/middle/last 1 MiB)"
            if ok
            else "Sample digest mismatch (first/middle/last 1 MiB)",
        )

    # full byte compare
    identical = True
    with open(path_a, "rb") as fa, open(path_b, "rb") as fb:
        while True:
            ca = fa.read(FULL_CHUNK)
            cb = fb.read(FULL_CHUNK)
            if ca != cb:
                identical = False
                break
            if not ca:
                break
    return BinCompareResult(
        identical,
        mode,
        size_a,
        size_b,
        "Full byte-identical" if identical else "Full byte mismatch",
    )


def _sample_digest(path: Path, size: int, chunk: int = SAMPLE_CHUNK) -> str:
    h = sha256()
    with open(path, "rb") as f:
        h.update(f.read(chunk))
        if size > 2 * chunk:
            f.seek(size // 2)
            h.update(f.read(chunk))
        if size > chunk:
            f.seek(max(0, size - chunk))
            h.update(f.read(chunk))
    return h.hexdigest()


def suggested_output_parent(dir_a: Path, dir_b: Path) -> Path | None:
    """If both suite2p dirs share a parent, return that parent."""
    try:
        a = resolve_suite2p_dir(dir_a)
        b = resolve_suite2p_dir(dir_b)
    except FileNotFoundError:
        return None
    if a.parent.resolve() == b.parent.resolve():
        return a.parent.resolve()
    return None


def merge_suite2p_folders(
    dir_a: Path,
    dir_b: Path,
    output_parent: Path,
    output_name: str = "suite2p_merged",
    *,
    bin_check: BinCheckMode = "sample",
    overwrite: bool = False,
) -> Path:
    """
    Concatenate ROI catalogs from A then B into ``output_parent/output_name``.

    Requires matching data.bin under the chosen ``bin_check`` mode.
    Does not copy input ``trc_curation.pkl`` files.
    Returns the new suite2p directory path.
    """
    a = resolve_suite2p_dir(dir_a)
    b = resolve_suite2p_dir(dir_b)
    name = (output_name or "suite2p_merged").strip()
    if not name or any(sep in name for sep in ("/", "\\")):
        raise MergeError(f"Invalid output folder name: {output_name!r}")

    out_root = Path(output_parent).resolve() / name
    if out_root.exists():
        if not overwrite:
            raise FileExistsError(str(out_root))
        shutil.rmtree(out_root)

    plane_a = plane_dir(a)
    plane_b = plane_dir(b)
    bin_a = open_data_bin(plane_a)
    bin_b = open_data_bin(plane_b)

    cmp = compare_data_bins(bin_a, bin_b, mode=bin_check)
    if not cmp.identical:
        raise MergeError(
            f"data.bin files are not identical under {bin_check!r} check.\n"
            f"{cmp.detail}\nA: {bin_a}\nB: {bin_b}"
        )

    ops_a = load_ops(plane_a)
    ops_b = load_ops(plane_b)
    for key in ("Ly", "Lx"):
        if int(ops_a[key]) != int(ops_b[key]):
            raise MergeError(
                f"ops[{key!r}] mismatch despite data.bin check: "
                f"{ops_a[key]} vs {ops_b[key]}"
            )

    stat_a = load_stat(plane_a)
    stat_b = load_stat(plane_b)
    F_a, Fneu_a = load_traces(plane_a)
    F_b, Fneu_b = load_traces(plane_b)
    if F_a.shape[1] != F_b.shape[1]:
        raise MergeError(
            f"Trace length (nframes) mismatch: {F_a.shape[1]} vs {F_b.shape[1]}"
        )
    n_a = len(stat_a)
    n_b = len(stat_b)
    if F_a.shape[0] != n_a or Fneu_a.shape[0] != n_a:
        raise MergeError(f"Folder A ROI count mismatch (stat vs F/Fneu)")
    if F_b.shape[0] != n_b or Fneu_b.shape[0] != n_b:
        raise MergeError(f"Folder B ROI count mismatch (stat vs F/Fneu)")

    iscell_a, prob_a = load_iscell(plane_a, n_a)
    iscell_b, prob_b = load_iscell(plane_b, n_b)

    stat_m = np.array(list(stat_a) + list(stat_b), dtype=object)
    F_m = np.concatenate([F_a, F_b], axis=0)
    Fneu_m = np.concatenate([Fneu_a, Fneu_b], axis=0)
    iscell_m = _stack_iscell(iscell_a, prob_a, iscell_b, prob_b)
    n_m = n_a + n_b
    T = int(F_m.shape[1])
    spks_m = np.zeros((n_m, T), dtype=np.float32)

    ops_m = dict(ops_a)
    ops_m["nframes"] = int(ops_a.get("nframes", T))
    if "nROIs" in ops_m or "nrois" in ops_m:
        ops_m["nROIs"] = n_m
    # provenance hints (non-suite2p-standard; harmless extras)
    ops_m["merged_from"] = [str(a), str(b)]
    ops_m["merge_roi_counts"] = [int(n_a), int(n_b)]

    out_plane = out_root / "plane0"
    out_plane.mkdir(parents=True, exist_ok=False)

    np.save(out_plane / "ops.npy", ops_m, allow_pickle=True)
    np.save(out_plane / "stat.npy", stat_m, allow_pickle=True)
    np.save(out_plane / "F.npy", F_m)
    np.save(out_plane / "Fneu.npy", Fneu_m)
    np.save(out_plane / "iscell.npy", iscell_m)
    np.save(out_plane / "spks.npy", spks_m)
    shutil.copy2(bin_a, out_plane / "data.bin")

    note = _merge_note_text(
        a=a,
        b=b,
        out_root=out_root,
        n_a=n_a,
        n_b=n_b,
        T=T,
        bin_check=bin_check,
        cmp=cmp,
        Ly=int(ops_a["Ly"]),
        Lx=int(ops_a["Lx"]),
    )
    (out_root / "merge_note.txt").write_text(note, encoding="utf-8")
    return out_root


def _stack_iscell(
    iscell_a: np.ndarray,
    prob_a: np.ndarray | None,
    iscell_b: np.ndarray,
    prob_b: np.ndarray | None,
) -> np.ndarray:
    if prob_a is None and prob_b is None:
        return np.concatenate([iscell_a.astype(np.float32), iscell_b.astype(np.float32)])
    pa = (
        prob_a.astype(np.float64)
        if prob_a is not None
        else np.full(len(iscell_a), np.nan, dtype=np.float64)
    )
    pb = (
        prob_b.astype(np.float64)
        if prob_b is not None
        else np.full(len(iscell_b), np.nan, dtype=np.float64)
    )
    col0 = np.concatenate([iscell_a.astype(np.float64), iscell_b.astype(np.float64)])
    col1 = np.concatenate([pa, pb])
    return np.column_stack([col0, col1])


def _merge_note_text(
    *,
    a: Path,
    b: Path,
    out_root: Path,
    n_a: int,
    n_b: int,
    T: int,
    bin_check: BinCheckMode,
    cmp: BinCompareResult,
    Ly: int,
    Lx: int,
) -> str:
    stamp = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    return (
        "s2p Trace Curation — suite2p folder merge\n"
        f"created_utc: {stamp}\n"
        f"output: {out_root}\n"
        "\n"
        "Sources (ROI order = listed order):\n"
        f"  1) {a}  — {n_a} ROIs -> merged indices 0..{n_a - 1}\n"
        f"  2) {b}  — {n_b} ROIs -> merged indices {n_a}..{n_a + n_b - 1}\n"
        "\n"
        f"FOV: Ly={Ly}, Lx={Lx}, nframes={T}\n"
        f"data.bin check: mode={bin_check}, size={cmp.size_a} bytes, {cmp.detail}\n"
        "data.bin: copied from source 1 (byte-identical to source 2 under check)\n"
        "ops/mean images: taken from source 1\n"
        "Overlapping ROIs: kept (both sources)\n"
        "trc_curation.pkl: not copied; created on first Open in the GUI\n"
        "spks.npy: zeros (n_roi × nframes)\n"
    )
