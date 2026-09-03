"""PMT-noise ranges from a per_frame.csv 'removed_rms' column."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

RMS_COLUMN = "removed_rms"

# Row order is the fallback; an explicit frame column wins when one is present.
FRAME_COLUMNS: tuple[str, ...] = (
    "frame",
    "frame_index",
    "frame_idx",
    "frameno",
    "frame_no",
    "index",
)


@dataclass(frozen=True)
class PerFrameRms:
    """removed_rms laid out per movie frame, NaN where the csv had no value."""

    values: np.ndarray
    path: Path
    n_rows: int
    n_mapped: int
    frame_column: str | None
    out_of_range: int

    @property
    def finite(self) -> np.ndarray:
        return self.values[np.isfinite(self.values)]

    def summary(self) -> str:
        parts = [f"{self.path.name}", f"{self.n_rows} row(s)"]
        if self.frame_column:
            parts.append(f"frame column '{self.frame_column}'")
        else:
            parts.append("row order as frame index")
        parts.append(f"{self.n_mapped} frame(s) with a value")
        if self.out_of_range:
            parts.append(f"{self.out_of_range} row(s) outside the movie")
        return " — ".join(parts)


def _clean_header(name: Any) -> str:
    return str(name or "").replace("\ufeff", "").strip().lower()


def _to_float(raw: Any) -> float:
    text = str(raw or "").strip()
    if not text:
        return float("nan")
    try:
        return float(text)
    except ValueError:
        return float("nan")


def read_per_frame_rms(
    path: str | Path, nframes: int | None = None
) -> PerFrameRms:
    """Read the removed_rms column, indexed by movie frame.

    Raises ValueError with a user-facing message when the file has no header
    or no removed_rms column.
    """
    p = Path(path)
    with p.open("r", newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames:
            raise ValueError(f"{p.name} has no header row.")
        headers = {
            _clean_header(h): h for h in reader.fieldnames if h is not None
        }
        if RMS_COLUMN not in headers:
            found = ", ".join(str(h) for h in reader.fieldnames if h) or "(none)"
            raise ValueError(
                f"{p.name} has no '{RMS_COLUMN}' column.\n\nColumns found: {found}"
            )
        rms_key = headers[RMS_COLUMN]
        frame_key = next(
            (headers[name] for name in FRAME_COLUMNS if name in headers), None
        )
        frames: list[int] = []
        vals: list[float] = []
        for i, row in enumerate(reader):
            frame = i
            if frame_key is not None:
                parsed = _to_float(row.get(frame_key))
                frame = int(parsed) if np.isfinite(parsed) else i
            frames.append(frame)
            vals.append(_to_float(row.get(rms_key)))

    n_rows = len(vals)
    size = int(nframes) if nframes and nframes > 0 else n_rows
    values = np.full(max(size, 1), np.nan, dtype=np.float64)
    out_of_range = 0
    for frame, value in zip(frames, vals):
        if 0 <= frame < values.shape[0]:
            values[frame] = value
        else:
            out_of_range += 1
    return PerFrameRms(
        values=values,
        path=p,
        n_rows=n_rows,
        n_mapped=int(np.isfinite(values).sum()),
        frame_column=str(frame_key) if frame_key is not None else None,
        out_of_range=out_of_range,
    )


def mask_to_ranges(
    mask: np.ndarray, *, merge_gap: int = 0, min_frames: int = 1
) -> list[list[int]]:
    """Contiguous True runs as inclusive [start, end] pairs.

    merge_gap joins runs separated by at most that many False frames;
    min_frames drops runs shorter than that (applied after merging).
    """
    m = np.asarray(mask, dtype=bool)
    if m.ndim != 1 or not m.any():
        return []
    edges = np.diff(np.concatenate(([False], m, [False])).astype(np.int8))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1) - 1
    ranges = [[int(a), int(b)] for a, b in zip(starts, ends)]

    gap = max(int(merge_gap), 0)
    if gap:
        merged: list[list[int]] = []
        for a, b in ranges:
            if merged and a - merged[-1][1] - 1 <= gap:
                merged[-1][1] = max(merged[-1][1], b)
            else:
                merged.append([a, b])
        ranges = merged

    keep = max(int(min_frames), 1)
    if keep > 1:
        ranges = [r for r in ranges if r[1] - r[0] + 1 >= keep]
    return ranges


def ranges_above_threshold(
    values: np.ndarray,
    threshold: float,
    *,
    merge_gap: int = 0,
    min_frames: int = 1,
) -> list[list[int]]:
    """Frames whose removed_rms exceeds threshold, as inclusive ranges."""
    v = np.asarray(values, dtype=np.float64)
    mask = np.isfinite(v) & (v > float(threshold))
    return mask_to_ranges(mask, merge_gap=merge_gap, min_frames=min_frames)


def suggest_threshold(values: np.ndarray) -> float:
    """A robust starting point: median + 5 robust sigma, kept inside the data."""
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 0.0
    lo = float(v.min())
    hi = float(v.max())
    if hi <= lo:
        return hi
    median = float(np.median(v))
    mad = float(np.median(np.abs(v - median)))
    sigma = 1.4826 * mad
    if sigma <= 0:
        sigma = float(v.std())
    if sigma <= 0:
        return hi
    return float(min(max(median + 5.0 * sigma, lo), hi))
