"""Read suite2p plane0 artifacts and data.bin frames."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np

PLANE_NAME = "plane0"


def resolve_suite2p_dir(path: Path) -> Path:
    """Normalize a user-chosen path to the suite2p directory (parent of plane0)."""
    path = Path(path).resolve()
    if path.name == PLANE_NAME and (path / "ops.npy").exists():
        return path.parent
    if (path / PLANE_NAME / "ops.npy").exists():
        return path
    if (path / "ops.npy").exists() and path.name.startswith("plane"):
        return path.parent
    raise FileNotFoundError(
        f"Could not find {PLANE_NAME}/ops.npy under or beside: {path}"
    )


def plane_dir(suite2p_dir: Path) -> Path:
    return Path(suite2p_dir) / PLANE_NAME


def load_ops(plane: Path) -> dict[str, Any]:
    ops = np.load(plane / "ops.npy", allow_pickle=True).item()
    if not isinstance(ops, dict):
        raise TypeError(f"ops.npy did not contain a dict: {plane / 'ops.npy'}")
    return ops


def load_stat(plane: Path) -> np.ndarray:
    return np.load(plane / "stat.npy", allow_pickle=True)


def load_traces(plane: Path) -> tuple[np.ndarray, np.ndarray]:
    F = np.load(plane / "F.npy")
    Fneu = np.load(plane / "Fneu.npy")
    return np.asarray(F, dtype=np.float64), np.asarray(Fneu, dtype=np.float64)


def load_iscell(plane: Path, n_roi: int) -> tuple[np.ndarray, np.ndarray | None]:
    path = plane / "iscell.npy"
    if not path.exists():
        return np.ones(n_roi, dtype=bool), None
    raw = np.load(path)
    if raw.ndim == 1:
        return raw.astype(bool), None
    iscell = raw[:, 0].astype(bool)
    prob = raw[:, 1].astype(np.float64) if raw.shape[1] > 1 else None
    return iscell, prob


def bin_dtype(ops: dict[str, Any]) -> np.dtype:
    dt = ops.get("dtype", "int16")
    if isinstance(dt, np.dtype):
        return dt
    return np.dtype(dt)


def open_data_bin(plane: Path) -> Path:
    path = plane / "data.bin"
    if not path.exists():
        raise FileNotFoundError(f"Missing data.bin: {path}")
    return path


class BinaryStack:
    """Memory-mapped / seek-based reader for suite2p data.bin."""

    def __init__(self, plane: Path, ops: dict[str, Any] | None = None):
        self.plane = Path(plane)
        self.path = open_data_bin(self.plane)
        self.ops = ops if ops is not None else load_ops(self.plane)
        self.Ly = int(self.ops["Ly"])
        self.Lx = int(self.ops["Lx"])
        self.dtype = bin_dtype(self.ops)
        self.frame_bytes = self.Ly * self.Lx * self.dtype.itemsize
        file_size = self.path.stat().st_size
        n_from_file = file_size // self.frame_bytes
        n_ops = int(self.ops.get("nframes", n_from_file))
        self.nframes = min(n_from_file, n_ops) if n_ops else n_from_file
        self._fh = open(self.path, "rb")

    def close(self) -> None:
        if getattr(self, "_fh", None) is not None:
            self._fh.close()
            self._fh = None

    def __enter__(self) -> BinaryStack:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def read_frame(self, index: int) -> np.ndarray:
        if index < 0 or index >= self.nframes:
            raise IndexError(f"frame {index} out of range [0, {self.nframes})")
        self._fh.seek(index * self.frame_bytes)
        flat = np.fromfile(self._fh, dtype=self.dtype, count=self.Ly * self.Lx)
        if flat.size != self.Ly * self.Lx:
            raise IOError(f"Short read for frame {index} from {self.path}")
        return flat.reshape(self.Ly, self.Lx)

    def extract_roi_trace(
        self,
        ypix: np.ndarray,
        xpix: np.ndarray,
        lam: np.ndarray,
        *,
        progress: Callable[[int, int], None] | None = None,
        should_cancel: Callable[[], bool] | None = None,
    ) -> np.ndarray:
        """Weighted mean fluorescence over frames (full pass — used after mask edits)."""
        ypix = np.asarray(ypix, dtype=np.int64)
        xpix = np.asarray(xpix, dtype=np.int64)
        lam = np.asarray(lam, dtype=np.float64)
        wsum = float(lam.sum())
        if wsum <= 0 or ypix.size == 0:
            return np.zeros(self.nframes, dtype=np.float64)
        out = np.empty(self.nframes, dtype=np.float64)
        for t in range(self.nframes):
            if should_cancel is not None and should_cancel():
                from s2p_trace_curation.mask_edit import ExtractCancelled

                raise ExtractCancelled()
            frame = self.read_frame(t)
            out[t] = float((frame[ypix, xpix] * lam).sum() / wsum)
            if progress is not None:
                progress(t + 1, self.nframes)
        return out

    def extract_unweighted_trace(
        self,
        ypix: np.ndarray,
        xpix: np.ndarray,
        *,
        progress: Callable[[int, int], None] | None = None,
        should_cancel: Callable[[], bool] | None = None,
        progress_offset: int = 0,
        progress_total: int | None = None,
    ) -> np.ndarray:
        """Unweighted mean over ``(ypix, xpix)`` — used for BG ROIs."""
        ypix = np.asarray(ypix, dtype=np.int64)
        xpix = np.asarray(xpix, dtype=np.int64)
        if ypix.size == 0:
            return np.zeros(self.nframes, dtype=np.float64)
        out = np.empty(self.nframes, dtype=np.float64)
        total = self.nframes if progress_total is None else int(progress_total)
        for t in range(self.nframes):
            if should_cancel is not None and should_cancel():
                from s2p_trace_curation.mask_edit import ExtractCancelled

                raise ExtractCancelled()
            frame = self.read_frame(t)
            out[t] = float(frame[ypix, xpix].mean())
            if progress is not None:
                progress(progress_offset + t + 1, total)
        return out

    def extract_neuropil_trace(
        self,
        ipix: np.ndarray,
        *,
        progress: Callable[[int, int], None] | None = None,
        should_cancel: Callable[[], bool] | None = None,
        progress_offset: int = 0,
        progress_total: int | None = None,
    ) -> np.ndarray:
        """Unweighted mean over neuropil pixels."""
        ipix = np.asarray(ipix, dtype=np.int64)
        if ipix.size == 0:
            return np.zeros(self.nframes, dtype=np.float64)
        y, x = np.unravel_index(ipix, (self.Ly, self.Lx))
        out = np.empty(self.nframes, dtype=np.float64)
        total = self.nframes if progress_total is None else int(progress_total)
        for t in range(self.nframes):
            if should_cancel is not None and should_cancel():
                from s2p_trace_curation.mask_edit import ExtractCancelled

                raise ExtractCancelled()
            frame = self.read_frame(t)
            out[t] = float(frame[y, x].mean())
            if progress is not None:
                progress(progress_offset + t + 1, total)
        return out


def combined_bbox(
    ypix: np.ndarray,
    xpix: np.ndarray,
    neuropil_ipix: np.ndarray,
    Ly: int,
    Lx: int,
) -> tuple[int, int, int, int]:
    """Return (y0, x0, y1, x1) inclusive-exclusive bbox of ROI ∪ neuropil."""
    ys = [np.asarray(ypix, dtype=np.int64)]
    xs = [np.asarray(xpix, dtype=np.int64)]
    if neuropil_ipix is not None and len(neuropil_ipix):
        ny, nx = np.unravel_index(np.asarray(neuropil_ipix, dtype=np.int64), (Ly, Lx))
        ys.append(ny)
        xs.append(nx)
    y_all = np.concatenate(ys) if ys else np.array([0], dtype=np.int64)
    x_all = np.concatenate(xs) if xs else np.array([0], dtype=np.int64)
    if y_all.size == 0:
        return 0, 0, 1, 1
    return int(y_all.min()), int(x_all.min()), int(y_all.max()) + 1, int(x_all.max()) + 1


def zoom_square_window(
    ypix: np.ndarray,
    xpix: np.ndarray,
    neuropil_ipix: np.ndarray,
    Ly: int,
    Lx: int,
    pad_factor: float = 1.5,
) -> tuple[int, int, int]:
    """
    Square crop centered on ROI∪neuropil.
    Side length = pad_factor * max(width, height) of combined bbox.
    Returns (y0, x0, side) clamped inside the FOV (center may shift near edges).
    """
    y0, x0, y1, x1 = combined_bbox(ypix, xpix, neuropil_ipix, Ly, Lx)
    height = max(1, y1 - y0)
    width = max(1, x1 - x0)
    side = int(np.ceil(pad_factor * max(width, height)))
    side = max(side, 1)
    side = min(side, Ly, Lx)
    cy = 0.5 * (y0 + y1 - 1)
    cx = 0.5 * (x0 + x1 - 1)
    top = int(round(cy - (side - 1) / 2.0))
    left = int(round(cx - (side - 1) / 2.0))
    top = max(0, min(top, Ly - side))
    left = max(0, min(left, Lx - side))
    return top, left, side


def zoom_square_at(
    cy: float,
    cx: float,
    side: int,
    Ly: int,
    Lx: int,
) -> tuple[int, int, int]:
    """Square crop of given side centered on (cy, cx), clamped to FOV."""
    side = int(max(1, min(side, Ly, Lx)))
    top = int(round(cy - (side - 1) / 2.0))
    left = int(round(cx - (side - 1) / 2.0))
    top = max(0, min(top, Ly - side))
    left = max(0, min(left, Lx - side))
    return top, left, side


def median_zoom_side(
    rois: list,
    Ly: int,
    Lx: int,
    pad_factor: float = 1.5,
    default: int = 64,
) -> int:
    """Median of per-ROI zoom square sides (ROI∪neuropil, same rule as W3)."""
    sides: list[int] = []
    for row in rois:
        try:
            _, _, side = zoom_square_window(
                row["roi"]["ypix"],
                row["roi"]["xpix"],
                row["neuropil"]["ipix"],
                Ly,
                Lx,
                pad_factor=pad_factor,
            )
            sides.append(int(side))
        except Exception:
            continue
    if not sides:
        return int(max(1, min(default, Ly, Lx)))
    return int(max(1, min(int(np.median(sides)), Ly, Lx)))


def embed_into_fov(
    img: np.ndarray,
    Ly: int,
    Lx: int,
    yrange: Any = None,
    xrange: Any = None,
) -> np.ndarray:
    """
    Place a possibly cropped suite2p map into a full (Ly, Lx) array.
    Uses ops yrange/xrange when the crop matches; otherwise centers the crop.
    """
    img = np.asarray(img)
    if img.ndim != 2:
        raise ValueError(f"Expected 2D image, got shape {img.shape}")
    if img.shape == (Ly, Lx):
        return img.astype(np.float32, copy=False)

    out = np.zeros((Ly, Lx), dtype=np.float32)
    if yrange is not None and xrange is not None:
        y0, y1 = int(yrange[0]), int(yrange[1])
        x0, x1 = int(xrange[0]), int(xrange[1])
        if img.shape == (y1 - y0, x1 - x0) and 0 <= y0 < y1 <= Ly and 0 <= x0 < x1 <= Lx:
            out[y0:y1, x0:x1] = img.astype(np.float32, copy=False)
            return out

    if img.shape[0] <= Ly and img.shape[1] <= Lx:
        y0 = (Ly - img.shape[0]) // 2
        x0 = (Lx - img.shape[1]) // 2
        out[y0 : y0 + img.shape[0], x0 : x0 + img.shape[1]] = img.astype(
            np.float32, copy=False
        )
        return out

    return np.asarray(img[:Ly, :Lx], dtype=np.float32)


def fov_images_from_ops(ops: dict[str, Any]) -> dict[str, np.ndarray | None]:
    """Extract meanImg / meanImgE / VCorr as full-FOV float32 arrays (or None)."""
    Ly = int(ops["Ly"])
    Lx = int(ops["Lx"])
    yrange = ops.get("yrange")
    xrange = ops.get("xrange")

    meanImg = None
    if ops.get("meanImg") is not None:
        meanImg = embed_into_fov(np.asarray(ops["meanImg"]), Ly, Lx, yrange, xrange)

    meanImgE = None
    if ops.get("meanImgE") is not None:
        meanImgE = embed_into_fov(np.asarray(ops["meanImgE"]), Ly, Lx, yrange, xrange)

    vcorr_raw = ops.get("Vcorr", ops.get("VCorr"))
    VCorr = None
    if vcorr_raw is not None:
        VCorr = embed_into_fov(np.asarray(vcorr_raw), Ly, Lx, yrange, xrange)

    return {"meanImg": meanImg, "meanImgE": meanImgE, "VCorr": VCorr}


def ops_cell_diameter(ops: dict[str, Any]) -> tuple[float, float]:
    """(diameter_px, aspect) for filter sizing; falls back to the suite2p default."""
    diameter = 0.0
    for key in ("spatscale_pix", "diameter"):
        raw = ops.get(key)
        if raw is None:
            continue
        arr = np.atleast_1d(np.asarray(raw, dtype=np.float64)).ravel()
        arr = arr[np.isfinite(arr) & (arr > 0)]
        if arr.size:
            diameter = float(arr[-1])
            break
    if diameter <= 0:
        diameter = 12.0

    aspect = 1.0
    raw_aspect = ops.get("aspect")
    try:
        val = float(raw_aspect)
        if np.isfinite(val) and val > 0:
            aspect = val
    except (TypeError, ValueError):
        pass
    return diameter, aspect


def enhanced_mean_image(
    mean_img: np.ndarray,
    Ly: int,
    Lx: int,
    yrange: Any = None,
    xrange: Any = None,
    diameter: float = 12.0,
    aspect: float = 1.0,
) -> np.ndarray:
    """
    Rebuild suite2p's meanImgE for ops.npy files that never stored one.

    Follows suite2p's recipe: median-filter meanImg at ~4x the cell diameter,
    subtract it, divide by the median-filtered absolute residual, then clip to
    the fixed -6..6 window and rescale to 0..1. Anatomical (Cellpose) runs skip
    this step, which is why the key can be missing.
    """
    from scipy.ndimage import median_filter

    img = np.asarray(mean_img, dtype=np.float32)
    if img.ndim != 2:
        raise ValueError(f"Expected 2D meanImg, got shape {img.shape}")

    def _kernel(extent: float, limit: int) -> int:
        size = int(4 * np.ceil(max(float(extent), 1.0)) + 1)
        size = min(size, int(limit))
        if size % 2 == 0:
            size -= 1
        return max(size, 3)

    size = (
        _kernel(diameter * aspect, img.shape[0]),
        _kernel(diameter, img.shape[1]),
    )
    residual = img - median_filter(img, size=size)
    scale = median_filter(np.abs(residual), size=size)
    norm = residual / (1e-10 + scale)

    y0, y1 = 0, img.shape[0]
    x0, x1 = 0, img.shape[1]
    if yrange is not None and xrange is not None:
        cy0, cy1 = int(yrange[0]), int(yrange[1])
        cx0, cx1 = int(xrange[0]), int(xrange[1])
        if 0 <= cy0 < cy1 <= img.shape[0] and 0 <= cx0 < cx1 <= img.shape[1]:
            y0, y1, x0, x1 = cy0, cy1, cx0, cx1

    inner = np.clip((norm[y0:y1, x0:x1] + 6.0) / 12.0, 0.0, 1.0)
    out = np.full(img.shape, float(inner.min()), dtype=np.float32)
    out[y0:y1, x0:x1] = inner
    return embed_into_fov(out, Ly, Lx, yrange, xrange)


def enhanced_mean_from_ops(ops: dict[str, Any]) -> np.ndarray | None:
    """meanImgE computed from ops' meanImg; None when there is no meanImg."""
    if ops.get("meanImg") is None:
        return None
    diameter, aspect = ops_cell_diameter(ops)
    return enhanced_mean_image(
        np.asarray(ops["meanImg"]),
        int(ops["Ly"]),
        int(ops["Lx"]),
        ops.get("yrange"),
        ops.get("xrange"),
        diameter=diameter,
        aspect=aspect,
    )
