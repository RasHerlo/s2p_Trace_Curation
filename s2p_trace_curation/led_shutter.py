"""LED+Shutter ranges from a defringe families.json shutter block."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# families.json is small. mask_recipe.json repeats the same shutter object
# near the start, then a per-frame recipe that is hundreds of MB.
_READ_CHUNK = 1 << 20
_KEY_TAIL = 32


@dataclass(frozen=True)
class ShutterFile:
    """Inclusive shutter intervals read from shutter.runs."""

    ranges: list[list[int]]
    path: Path
    n_runs: int
    method: str | None
    out_of_range: int

    def summary(self) -> str:
        parts = [f"{self.path.name}", f"{self.n_runs} run(s)"]
        if self.method:
            parts.append(f"method '{self.method}'")
        parts.append(f"{len(self.ranges)} interval(s) in the movie")
        if self.out_of_range:
            parts.append(f"{self.out_of_range} run(s) outside the movie")
        return " — ".join(parts)


def read_shutter_runs(
    path: str | Path, nframes: int | None = None
) -> ShutterFile:
    """Read shutter.runs start/stop pairs from families.json.

    stop is inclusive when it matches the run's frame list or n. A stop that
    is one past the last frame (n == stop - start) is treated as exclusive.
    Raises ValueError with a user-facing message when the file has no shutter
    runs.
    """
    p = Path(path)
    try:
        shutter = _read_shutter_object(p)
    except OSError as exc:
        raise ValueError(f"Could not read {p.name}: {exc}") from exc
    runs = shutter.get("runs")
    if not isinstance(runs, list) or not runs:
        frames = shutter.get("frames")
        if isinstance(frames, list) and frames:
            runs = [{"frames": frames}]
        else:
            raise ValueError(
                f"{p.name} has no shutter runs.\n\n"
                "Expected a families.json with a shutter.runs list of "
                "start/stop intervals."
            )
    raw: list[list[int]] = []
    for i, run in enumerate(runs):
        if not isinstance(run, dict):
            raise ValueError(f"{p.name} shutter run {i} is not an object.")
        try:
            spans = _run_bounds(run)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"{p.name} shutter run {i} has no usable start/stop."
            ) from exc
        raw.extend(spans)

    last = int(nframes) - 1 if nframes and int(nframes) > 0 else None
    kept: list[list[int]] = []
    out_of_range = 0
    for a, b in raw:
        if last is None:
            kept.append([a, b])
            continue
        if b < 0 or a > last:
            out_of_range += 1
            continue
        kept.append([max(a, 0), min(b, last)])
    method = shutter.get("method")
    return ShutterFile(
        ranges=kept,
        path=p,
        n_runs=len(runs),
        method=str(method) if method else None,
        out_of_range=out_of_range,
    )


def _run_bounds(run: dict[str, Any]) -> list[list[int]]:
    frames = run.get("frames")
    if isinstance(frames, list) and frames:
        return _contiguous(sorted(int(x) for x in frames))
    start = int(run["start"])
    stop = int(run["stop"])
    n = run.get("n")
    if n is not None and int(n) == stop - start and int(n) > 0:
        stop -= 1
    if stop < start:
        start, stop = stop, start
    return [[start, stop]]


def _contiguous(frames: list[int]) -> list[list[int]]:
    if not frames:
        raise ValueError("empty frame list")
    ranges: list[list[int]] = []
    start = prev = frames[0]
    for frame in frames[1:]:
        if frame <= prev + 1:
            prev = frame
            continue
        ranges.append([start, prev])
        start = prev = frame
    ranges.append([start, prev])
    return ranges


def _read_shutter_object(path: Path) -> dict[str, Any]:
    """Decode the top-level shutter object without reading a huge recipe."""
    decoder = json.JSONDecoder()
    buf = ""
    with path.open("r", encoding="utf-8") as fh:
        while True:
            chunk = fh.read(_READ_CHUNK)
            buf += chunk
            key = buf.find('"shutter"')
            if key < 0:
                if not chunk:
                    break
                if len(buf) > _KEY_TAIL:
                    buf = buf[-_KEY_TAIL:]
                continue
            colon = buf.find(":", key + len('"shutter"'))
            if colon < 0:
                if not chunk:
                    break
                continue
            try:
                obj, _end = decoder.raw_decode(buf[colon + 1 :].lstrip())
            except json.JSONDecodeError:
                if not chunk:
                    break
                continue
            if not isinstance(obj, dict):
                raise ValueError(f"{path.name} has a shutter value that is not an object.")
            return obj
    raise ValueError(
        f"{path.name} has no shutter object.\n\n"
        "Choose the families.json written next to per_frame.csv in the defringe folder."
    )
