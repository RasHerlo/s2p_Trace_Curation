"""Per-ROI edit status: what differs from the baseline, and component-wise reset.

An ROI has six components. Four are per ROI and can be reset one by one:
``selected`` (iscell), ``roi_mask``, ``neuropil_mask`` and ``fneu_comp``
(x, Fneu offset, baseline-offset flag). Two are experiment-wide and only shown:
``smoothing`` (SG window / order) and ``bleach`` (edited only with a non-default
τ mode, or on/off differing from the shared ROI set; "on" alone is not an edit).

"Edited" means *differs from the baseline*, computed on demand, so changing a
value back un-ticks it. The baseline is

* the suite2p files (``plane0``) for an experiment without a shared ROI set, and
* the shared ROI set (master) for ROIs of a linked experiment.

ROIs drawn or imported here have no baseline. ``row["edited_utc"]`` additionally
records when each component was last edited (from this feature on).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from s2p_trace_curation.curation import (
    apply_baseline_fneu_offset,
    baseline_offset_enabled,
    compensation_fneu_offset,
    ensure_compensation,
    load_curation,
    plane_baseline_row,
    refresh_trace_comp,
    set_baseline_offset_enabled,
)
from s2p_trace_curation.roisets import (
    ORIGIN_MASTER,
    load_master,
    master_baseline_rows,
    master_path_for_doc,
    rederive_row,
)
from s2p_trace_curation.suite2p_io import load_iscell, load_stat, plane_dir, resolve_suite2p_dir
from s2p_trace_curation.trace_processing import (
    DEFAULT_SG_POLY,
    DEFAULT_SG_WINDOW,
    TAU_SHARED,
    ensure_trace_processing,
    rebuild_row_traces,
)

COMPONENTS = ("selected", "roi_mask", "neuropil_mask", "fneu_comp", "smoothing", "bleach")
COMPONENT_LABELS = {
    "selected": "Selected",
    "roi_mask": "ROI-mask",
    "neuropil_mask": "Neuropil-mask",
    "fneu_comp": "Fneu-compensation (x, offset)",
    "smoothing": "Trace smoothing (SG window, order)",
    "bleach": "Bleach-correction",
}
# Per-ROI components: they count towards "edited" and can be reset one by one.
RESETTABLE = ("selected", "roi_mask", "neuropil_mask", "fneu_comp")
EXPERIMENT_WIDE_NOTE = (
    "Set for the whole experiment in Trace Processing, so it is not reset per ROI "
    "and not counted in the ROI's Edited status."
)

STATUS_EDITED = "edited"
STATUS_UNCHANGED = "unchanged"
STATUS_NEW = "new"  # drawn / imported here: nothing to compare with
STATUS_UNKNOWN = "unknown"  # linked, but the shared ROI set could not be loaded

KIND_SUITE2P = "suite2p"
KIND_SHARED = "shared"

_KEY_STRIDE = 1_000_003  # > any FOV side; makes (y, x) one sortable integer
_X_TOL = 1e-9


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


# ------------------------------------------------------------ timestamps
def touch_edit(row: dict[str, Any], *components: str) -> None:
    """Record that these components were just edited on ``row``."""
    stamps = row.setdefault("edited_utc", {})
    now = _utc_now()
    for key in components:
        stamps[key] = now


def clear_edit_times(row: dict[str, Any], components: Any) -> None:
    stamps = row.get("edited_utc")
    if not isinstance(stamps, dict):
        return
    for key in components:
        stamps.pop(key, None)
    if not stamps:
        row.pop("edited_utc", None)


# -------------------------------------------------------------- baseline
@dataclass
class Baseline:
    iscell: bool
    roi_key: np.ndarray  # sorted linear pixel keys of the ROI mask
    roi_lam: np.ndarray  # weights in the same order
    neu_key: np.ndarray  # sorted neuropil pixel indices
    x: float
    aligned: bool  # offset comes from baseline alignment (shared sets only)


def _canon_mask(ypix: Any, xpix: Any, lam: Any) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(ypix, dtype=np.int64)
    x = np.asarray(xpix, dtype=np.int64)
    key = y * _KEY_STRIDE + x
    order = np.argsort(key, kind="stable")
    return key[order], np.asarray(lam, dtype=np.float64)[order]


def _baseline_from_stat(s: Any, iscell: bool) -> Baseline:
    key, lam = _canon_mask(s["ypix"], s["xpix"], s["lam"])
    try:
        neu = np.sort(np.asarray(s["neuropil_mask"], dtype=np.int64))
    except KeyError:
        neu = np.zeros(0, dtype=np.int64)
    return Baseline(bool(iscell), key, lam, neu, 1.0, False)


def _baseline_from_row(r: dict[str, Any]) -> Baseline:
    key, lam = _canon_mask(r["roi"]["ypix"], r["roi"]["xpix"], r["roi"]["lam"])
    neu = np.sort(np.asarray(r["neuropil"]["ipix"], dtype=np.int64))
    comp = r["compensation"]
    return Baseline(
        bool(r.get("iscell", True)),
        key,
        lam,
        neu,
        float(comp.get("x", 1.0)),
        bool(comp.get("baseline_offset", False)),
    )


class EditBaselines:
    """Lazy, cached baselines of one open experiment."""

    def __init__(self) -> None:
        self._key: Any = None
        self._items: dict[int, Baseline] = {}
        self._stat: Any = None
        self._iscell: np.ndarray | None = None
        self.kind = KIND_SUITE2P
        self.error: str | None = None
        self.label = ""
        self.sg_default: tuple[int, int] = (DEFAULT_SG_WINDOW, DEFAULT_SG_POLY)
        # None = bleach on or off are both "default" (no shared ROI set to say which);
        # the τ mode is what makes an "on" bleach correction non-default.
        self.bleach_default: bool | None = None
        self.tau_default = TAU_SHARED

    def prepare(self, doc: dict[str, Any], suite2p_dir: Path, master: dict[str, Any] | None) -> None:
        link = doc.get("master_link")
        if link:
            mrev = None if master is None else master.get("meta", {}).get("rev")
            key = (
                str(suite2p_dir),
                "shared",
                link.get("applied_utc"),
                tuple(link.get("shift") or ()),
                id(master),
                mrev,
            )
        else:
            key = (str(suite2p_dir), "suite2p")
        if key == self._key:
            return
        self._key = key
        self._items = {}
        self._stat = None
        self._iscell = None
        self.error = None
        self.sg_default = (DEFAULT_SG_WINDOW, DEFAULT_SG_POLY)
        self.bleach_default = None
        self.tau_default = TAU_SHARED
        if link:
            self.kind = KIND_SHARED
            if master is None:
                self.error = "the shared ROI set could not be loaded"
                self.label = f"shared ROI set '{link.get('name', '?')}' (not available)"
                return
            dy, dx = (int(v) for v in (link.get("shift") or [0, 0]))
            for uid, r in master_baseline_rows(master, dy, dx).items():
                self._items[uid] = _baseline_from_row(r)
            tp = (master.get("meta") or {}).get("trace_processing") or {}
            try:
                self.sg_default = (
                    int(tp.get("sg_window") or DEFAULT_SG_WINDOW),
                    int(tp.get("sg_poly") or DEFAULT_SG_POLY),
                )
            except (TypeError, ValueError):
                pass
            self.bleach_default = bool(tp.get("bleach_enabled", False))
            self.tau_default = str(tp.get("tau_mode") or TAU_SHARED)
            self.label = (
                f"shared ROI set '{master['meta'].get('name', '?')}' "
                f"(rev {master['meta'].get('rev', '?')})"
            )
        else:
            self.kind = KIND_SUITE2P
            self.label = "the suite2p files (plane0)"
            try:
                plane = plane_dir(resolve_suite2p_dir(suite2p_dir))
                self._stat = load_stat(plane)
                self._iscell, _prob = load_iscell(plane, len(self._stat))
            except (OSError, ValueError, KeyError) as exc:
                self._stat = None
                self.error = f"the suite2p files could not be read ({exc})"

    def baseline_for(self, row: dict[str, Any]) -> Baseline | None:
        link = row.get("link") or {}
        if self.kind == KIND_SHARED:
            if link.get("origin") != ORIGIN_MASTER or link.get("uid") is None:
                return None
            return self._items.get(int(link["uid"]))
        if link.get("origin") not in (None, ORIGIN_MASTER) or self._stat is None:
            return None
        rid = int(row["roi_id"])
        if rid < 0 or rid >= len(self._stat):
            return None
        base = self._items.get(rid)
        if base is None:
            base = _baseline_from_stat(self._stat[rid], bool(self._iscell[rid]))  # type: ignore[index]
            self._items[rid] = base
        return base


# ---------------------------------------------------------------- result
@dataclass
class Component:
    key: str
    label: str
    edited: bool
    current: str
    default: str
    resettable: bool
    when: str | None = None


@dataclass
class RoiEdits:
    roi_id: int
    status: str
    baseline_label: str
    components: dict[str, Component] = field(default_factory=dict)

    @property
    def edited_keys(self) -> list[str]:
        return [k for k in RESETTABLE if k in self.components and self.components[k].edited]

    @property
    def has_baseline(self) -> bool:
        return self.status in (STATUS_EDITED, STATUS_UNCHANGED)


def _num(value: float, decimals: int) -> str:
    s = f"{float(value):.{decimals}f}".rstrip("0")
    return s + "0" if s.endswith(".") else s


def _mask_text(n: int, n0: int | None, edited: bool) -> str:
    if edited and n0 is not None:
        return f"{n} px (was {n0})" if n != n0 else f"{n} px (changed)"
    return f"{n} px"


def analyze_row(doc: dict[str, Any], row: dict[str, Any], baselines: EditBaselines) -> RoiEdits:
    """Edit state of one ROI row. ``baselines`` must be prepared for ``doc``."""
    base = baselines.baseline_for(row)
    comp = ensure_compensation(row)
    stamps = row.get("edited_utc") if isinstance(row.get("edited_utc"), dict) else {}

    ypix = row["roi"]["ypix"]
    n_px = int(np.asarray(ypix).size)
    n_neu = int(np.asarray(row["neuropil"]["ipix"]).size)
    x = float(comp["x"])
    off = compensation_fneu_offset(row)
    flag = baseline_offset_enabled(row)
    comp_text = f"[{_num(x, 3)}, {_num(off, 2)}]" + (" baseline-aligned" if flag else "")
    selected = bool(row.get("iscell", True))

    sel_ed = roi_ed = neu_ed = comp_ed = False
    sel_def = roi_def = neu_def = comp_def = "—"
    n_px0: int | None = None
    n_neu0: int | None = None
    if base is not None:
        sel_ed = selected != base.iscell
        sel_def = str(base.iscell)

        key, lam = _canon_mask(row["roi"]["ypix"], row["roi"]["xpix"], row["roi"]["lam"])
        roi_ed = not (
            key.shape == base.roi_key.shape
            and np.array_equal(key, base.roi_key)
            and np.allclose(lam, base.roi_lam, rtol=1e-5, atol=1e-8)
        )
        n_px0 = int(base.roi_key.size)
        roi_def = f"{n_px0} px"

        neu = np.sort(np.asarray(row["neuropil"]["ipix"], dtype=np.int64))
        neu_ed = not (neu.shape == base.neu_key.shape and np.array_equal(neu, base.neu_key))
        n_neu0 = int(base.neu_key.size)
        neu_def = f"{n_neu0} px"

        # Baseline alignment is a derived offset (also applied automatically by
        # "Automate initial steps"), so it is not an edit. What counts: a changed x,
        # or an offset that is not the aligned one (flag off, value not 0).
        comp_ed = abs(x - base.x) > _X_TOL or (not flag and abs(off) > _X_TOL)
        comp_def = f"[{_num(base.x, 3)}, 0.0 or baseline-aligned]"

    tp = ensure_trace_processing(doc)
    sg = (int(tp["sg_window"]), int(tp["sg_poly"]))
    sg_edited = sg != baselines.sg_default
    enabled = bool(tp.get("bleach_enabled", False))
    bl = row.get("bleach") or {}
    tau_mode = str(tp.get("tau_mode") or TAU_SHARED)
    # "On" with the default settings is not an edit (the automation turns it on).
    # Edited: on/off differs from a shared ROI set that says which, or bleach is on
    # with a non-default τ mode (independent fits instead of a shared τ).
    if baselines.bleach_default is not None and enabled != baselines.bleach_default:
        bleach_ed = True
    else:
        bleach_ed = enabled and tau_mode != baselines.tau_default
    if not enabled:
        bleach_text = "off"
    elif tau_mode != TAU_SHARED:
        bleach_text = f"on ({tau_mode} fits)"
    elif bool(bl.get("conservative", True)):
        bleach_text = "on (conservative fallback)"
    else:
        bleach_text = "on"
    if baselines.bleach_default is None:
        bleach_def = "off, or on with shared \u03c4" if baselines.tau_default == TAU_SHARED else (
            f"off, or on with {baselines.tau_default} fits"
        )
    else:
        bleach_def = (
            "off"
            if not baselines.bleach_default
            else ("on" if baselines.tau_default == TAU_SHARED else f"on ({baselines.tau_default} fits)")
        )

    def comp_item(key: str, edited: bool, current: str, default: str) -> Component:
        return Component(
            key=key,
            label=COMPONENT_LABELS[key],
            edited=bool(edited),
            current=current,
            default=default,
            resettable=key in RESETTABLE,
            when=(stamps.get(key) if edited else None),
        )

    components = {
        "selected": comp_item("selected", sel_ed, str(selected), sel_def),
        "roi_mask": comp_item("roi_mask", roi_ed, _mask_text(n_px, n_px0, roi_ed), roi_def),
        "neuropil_mask": comp_item(
            "neuropil_mask", neu_ed, _mask_text(n_neu, n_neu0, neu_ed), neu_def
        ),
        "fneu_comp": comp_item("fneu_comp", comp_ed, comp_text, comp_def),
        "smoothing": comp_item(
            "smoothing",
            sg_edited,
            f"[{sg[0]}, {sg[1]}]",
            f"[{baselines.sg_default[0]}, {baselines.sg_default[1]}]",
        ),
        "bleach": comp_item("bleach", bleach_ed, bleach_text, bleach_def),
    }
    if base is None:
        # A baseline that could not be loaded is different from "nothing to compare with".
        status = STATUS_UNKNOWN if baselines.error else STATUS_NEW
    else:
        status = (
            STATUS_EDITED
            if any(components[k].edited for k in RESETTABLE)
            else STATUS_UNCHANGED
        )
    return RoiEdits(
        roi_id=int(row["roi_id"]),
        status=status,
        baseline_label=baselines.label,
        components=components,
    )


def analyze_all(doc: dict[str, Any], baselines: EditBaselines) -> list[RoiEdits]:
    return [analyze_row(doc, row, baselines) for row in doc["rois"]]


# ------------------------------------------------- summary of a pickle file
@dataclass
class EditSummary:
    """How many ROIs of a saved experiment differ from their baseline."""

    edited: int = 0
    total: int = 0
    new: int = 0  # drawn / imported ROIs: nothing to compare with
    unknown: int = 0  # baseline could not be loaded
    note: str | None = None  # why the counts are incomplete or missing
    failed: bool = False

    @property
    def text(self) -> str:
        if self.failed:
            return "?"
        if self.total and self.unknown == self.total:
            return f"?/{self.total}"  # no baseline to compare with at all
        return f"{self.edited}/{self.total}"


def summarize_pickle(pkl: Path, master_cache: dict[str, Any] | None = None) -> EditSummary:
    """Edit count of one ``trc_curation.pkl``, without opening a GUI session.

    Uses the same baselines as the Edits windows (suite2p files next to the pickle,
    or the linked shared ROI set). ``master_cache`` (path -> master) saves reloading a
    shared ROI set that many experiments link to. Never raises: problems come back
    as ``failed`` / ``note``.
    """
    try:
        pkl = Path(pkl)
        doc = load_curation(pkl)
        suite2p_dir = pkl.parent
        try:
            suite2p_dir = resolve_suite2p_dir(suite2p_dir)
        except FileNotFoundError:
            pass  # no plane0 next to the pickle: the baseline reports it below
        master = None
        master_note = None
        if doc.get("master_link"):
            mpath = master_path_for_doc(doc, suite2p_dir)
            if mpath is None:
                master_note = "the shared ROI set was not found"
            else:
                key = str(mpath)
                if master_cache is not None and key in master_cache:
                    master = master_cache[key]
                else:
                    try:
                        master = load_master(mpath)
                    except Exception as exc:  # noqa: BLE001 - reported, not raised
                        master_note = f"the shared ROI set could not be loaded ({exc})"
                    if master_cache is not None and master is not None:
                        master_cache[key] = master
        baselines = EditBaselines()
        baselines.prepare(doc, suite2p_dir, master)
        results = analyze_all(doc, baselines)
    except Exception as exc:  # noqa: BLE001 - one bad file must not stop the listing
        return EditSummary(note=f"could not be read ({exc})", failed=True)
    summary = EditSummary(
        edited=sum(1 for r in results if r.status == STATUS_EDITED),
        total=len(results),
        new=sum(1 for r in results if r.status == STATUS_NEW),
        unknown=sum(1 for r in results if r.status == STATUS_UNKNOWN),
    )
    note = master_note or baselines.error
    if note:
        summary.note = f"Compared with {baselines.label}: {note}."
    return summary


# ----------------------------------------------------------------- reset
def reset_components(
    doc: dict[str, Any],
    suite2p_dir: Path,
    row: dict[str, Any],
    components: list[str],
    *,
    master: dict[str, Any] | None,
) -> list[str]:
    """Restore the chosen per-ROI components of ``row`` (in place) to the baseline.

    Mask resets bring back the baseline traces too: from ``F.npy`` / ``Fneu.npy``
    for suite2p, from this experiment's ``data.bin`` for a shared ROI. The ROI's
    derived traces are rebuilt afterwards. Returns the components that were reset.
    Nothing is changed when the baseline cannot be built.
    """
    comps = [c for c in RESETTABLE if c in set(components)]
    if not comps:
        return []
    rid = int(row["roi_id"])
    need_traces = bool({"roi_mask", "neuropil_mask"} & set(comps))
    link = doc.get("master_link")
    if link:
        rlink = row.get("link") or {}
        if rlink.get("origin") != ORIGIN_MASTER or rlink.get("uid") is None:
            raise ValueError(
                "This ROI was added in this experiment, so there is no shared version to reset to."
            )
        if master is None:
            raise ValueError("The shared ROI set is not available.")
        uid = int(rlink["uid"])
        if need_traces:
            base = rederive_row(doc, suite2p_dir, master, uid)
        else:
            dy, dx = (int(v) for v in (link.get("shift") or [0, 0]))
            found = master_baseline_rows(master, dy, dx, {uid})
            if uid not in found:
                raise KeyError(f"uid {uid} is not in the shared ROI set (or falls outside this FOV)")
            base = found[uid]
    else:
        if (row.get("link") or {}).get("origin") not in (None, ORIGIN_MASTER):
            raise ValueError("This ROI has no suite2p version to reset to.")
        base = plane_baseline_row(suite2p_dir, rid, with_traces=need_traces)

    from s2p_trace_curation.raster import led_shutter_nan_mask

    def align(r: dict[str, Any]) -> None:
        mask = led_shutter_nan_mask(doc, int(doc["meta"]["nframes"]))
        try:
            apply_baseline_fneu_offset(r, mask)
        except ValueError:
            set_baseline_offset_enabled(r, False)

    if "selected" in comps:
        row["iscell"] = bool(base["iscell"])
    if "roi_mask" in comps:
        row["roi"]["ypix"] = np.asarray(base["roi"]["ypix"], dtype=np.int32).copy()
        row["roi"]["xpix"] = np.asarray(base["roi"]["xpix"], dtype=np.int32).copy()
        row["roi"]["lam"] = np.asarray(base["roi"]["lam"], dtype=np.float32).copy()
        row["roi"]["F"] = np.asarray(base["roi"]["F"], dtype=np.float64).copy()
        row["roi"]["modified"] = False
    if "neuropil_mask" in comps:
        row["neuropil"]["ipix"] = np.asarray(base["neuropil"]["ipix"], dtype=np.int32).copy()
        row["neuropil"]["Fneu"] = np.asarray(base["neuropil"]["Fneu"], dtype=np.float64).copy()
        row["neuropil"]["modified"] = False

    comp = ensure_compensation(row)
    if "fneu_comp" in comps:
        # Back to the default x and no manual offset. Baseline alignment is a mode,
        # not an edit: keep it if the ROI has it, or if the shared set asks for it.
        want_align = bool(comp.get("baseline_offset", False)) or bool(
            base["compensation"].get("baseline_offset", False)
        )
        comp["x"] = float(base["compensation"]["x"])
        comp["fneu_offset"] = 0.0
        comp["baseline_offset"] = False
        refresh_trace_comp(row)
        if want_align:
            align(row)
    elif need_traces and bool(comp.get("baseline_offset", False)):
        refresh_trace_comp(row)
        align(row)  # the new F / Fneu moved the baseline; keep the alignment

    refresh_trace_comp(row)
    rebuild_row_traces(doc, row)
    clear_edit_times(row, comps)
    return comps
