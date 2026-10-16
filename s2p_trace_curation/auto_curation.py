"""Merge suite2p_temp + suite2p_anat and apply the default curation steps."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from s2p_trace_curation.annotations import (
    PROPERTY_LED_SHUTTER,
    ensure_annotations,
    make_annotation,
    next_ann_id,
)
from s2p_trace_curation.curation import (
    apply_baseline_fneu_offset,
    create_curation_from_plane,
    save_curation,
)
from s2p_trace_curation.led_shutter import ShutterFile, read_shutter_runs
from s2p_trace_curation.merge_suite2p import BinCheckMode, merge_suite2p_folders
from s2p_trace_curation.raster import led_shutter_nan_mask, rebuild_all_tc_norm
from s2p_trace_curation.suite2p_io import load_traces, plane_dir, resolve_suite2p_dir
from s2p_trace_curation.trace_processing import (
    TAU_INDEPENDENT,
    TAU_SHARED,
    TRACE_FIELD_SM_BC,
    ensure_trace_processing,
    estimate_shared_taus,
    rebuild_all_tc_norm_sm,
    rebuild_all_tc_norm_sm_bc,
    set_raster_trace_field,
)

TEMP_DIR_NAME = "suite2p_temp"
ANAT_DIR_NAME = "suite2p_anat"
MERGED_DIR_NAME = "suite2p_merged"
DEFRINGE_DIR_NAME = "defringe_v4"
FAMILIES_NAME = "families.json"


class AutomationError(Exception):
    """Parent folder is missing an input the automated merge needs."""


@dataclass
class AutomationPreview:
    parent: Path
    temp_dir: Path
    anat_dir: Path
    families: Path
    shutter: ShutterFile
    nframes: int


@dataclass
class AutomationResult:
    merged_dir: Path
    n_led: int
    n_offset: int
    tau: tuple[float, float] | None
    notes: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"Wrote merged suite2p folder:\n{self.merged_dir}",
            "",
            f"LED+Shutter: {self.n_led} interval(s) from {DEFRINGE_DIR_NAME}/{FAMILIES_NAME}",
            "Smoothing: Savitzky–Golay defaults",
        ]
        if self.tau is None:
            lines.append("Bleach: independent fits (shared τ could not be estimated)")
        else:
            lines.append(f"Bleach: shared τ ({self.tau[0]:.2f}, {self.tau[1]:.2f} frames)")
        lines.append(f"Baseline offset: {self.n_offset} ROI(s)")
        if self.notes:
            lines.append("")
            lines.extend(self.notes)
        lines.append("")
        lines.append("The merged folder will now open in the main window.")
        return "\n".join(lines)


def _require_suite2p(parent: Path, name: str) -> Path:
    path = parent / name
    if not path.is_dir():
        raise AutomationError(
            f"The output parent must contain {TEMP_DIR_NAME} and {ANAT_DIR_NAME}.\n\n"
            f"Missing {name}:\n{path}"
        )
    try:
        return resolve_suite2p_dir(path)
    except (FileNotFoundError, OSError) as exc:
        raise AutomationError(
            f"{name} is not a suite2p folder (no plane0/ops.npy).\n\n{exc}"
        ) from exc


def preview_automation(parent: Path) -> AutomationPreview:
    """Check the parent folder. Does not write anything."""
    root = Path(parent)
    if not root.is_dir():
        raise AutomationError(f"Output parent folder does not exist:\n{root}")
    temp_dir = _require_suite2p(root, TEMP_DIR_NAME)
    anat_dir = _require_suite2p(root, ANAT_DIR_NAME)
    families = root / DEFRINGE_DIR_NAME / FAMILIES_NAME
    if not families.is_file():
        raise AutomationError(
            f"No {FAMILIES_NAME} in {DEFRINGE_DIR_NAME}.\n\nExpected:\n{families}"
        )
    F, _Fneu = load_traces(plane_dir(temp_dir))
    nframes = int(F.shape[1])
    try:
        shutter = read_shutter_runs(families, nframes=nframes)
    except ValueError as exc:
        raise AutomationError(str(exc)) from exc
    if not shutter.ranges:
        detail = (
            f"{shutter.out_of_range} run(s) fell outside this movie ({nframes} frame(s))."
            if shutter.out_of_range
            else f"{families.name} listed no shutter frames."
        )
        raise AutomationError(f"No LED+Shutter ranges to annotate.\n\n{detail}")
    return AutomationPreview(
        parent=root,
        temp_dir=temp_dir,
        anat_dir=anat_dir,
        families=families,
        shutter=shutter,
        nframes=nframes,
    )


def run_automated_merge(
    parent: Path,
    *,
    bin_check: BinCheckMode = "sample",
    overwrite: bool = False,
) -> AutomationResult:
    """Merge temp then anat, then annotate, smooth, bleach, and align Fneu."""
    preview = preview_automation(parent)
    merged = merge_suite2p_folders(
        preview.temp_dir,
        preview.anat_dir,
        preview.parent,
        MERGED_DIR_NAME,
        bin_check=bin_check,
        overwrite=overwrite,
    )
    doc = create_curation_from_plane(merged)
    notes: list[str] = []
    shutter = preview.shutter
    if shutter.out_of_range:
        notes.append(
            f"{shutter.out_of_range} shutter run(s) fell outside the movie and were dropped."
        )
    anns = ensure_annotations(doc)
    label = f"{shutter.path.name} shutter"
    for start, end in shutter.ranges:
        anns.append(
            make_annotation(
                next_ann_id(doc),
                PROPERTY_LED_SHUTTER,
                int(start),
                int(end),
                label=label,
            )
        )

    nframes = int(doc["meta"]["nframes"])
    mask = led_shutter_nan_mask(doc, nframes)
    n_offset = 0
    skipped: list[str] = []
    for row in doc["rois"]:
        try:
            apply_baseline_fneu_offset(row, mask)
        except ValueError:
            skipped.append(str(int(row["roi_id"])))
            continue
        n_offset += 1
    if skipped:
        shown = ", ".join(skipped[:8])
        extra = f" (+{len(skipped) - 8} more)" if len(skipped) > 8 else ""
        notes.append(f"Baseline offset skipped for ROI(s) {shown}{extra}.")

    rebuild_all_tc_norm(doc)
    tp = ensure_trace_processing(doc)
    tp["bleach_enabled"] = True
    rebuild_all_tc_norm_sm(doc)
    shared = estimate_shared_taus(doc, mask)
    if shared is None:
        tp["tau_mode"] = TAU_INDEPENDENT
        tp["shared_tau1"] = None
        tp["shared_tau2"] = None
        notes.append("Could not fit a shared τ; each ROI was fit on its own.")
        tau = None
    else:
        tp["tau_mode"] = TAU_SHARED
        tp["shared_tau1"] = float(shared[0])
        tp["shared_tau2"] = float(shared[1])
        tau = (float(shared[0]), float(shared[1]))
    rebuild_all_tc_norm_sm_bc(doc)
    set_raster_trace_field(doc, TRACE_FIELD_SM_BC)
    save_curation(doc, merged)
    return AutomationResult(
        merged_dir=merged,
        n_led=len(shutter.ranges),
        n_offset=n_offset,
        tau=tau,
        notes=notes,
    )
