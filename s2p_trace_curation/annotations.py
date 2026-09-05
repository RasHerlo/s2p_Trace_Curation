"""Global time-range annotations for curation sessions."""

from __future__ import annotations

from typing import Any

import numpy as np

PROPERTY_LED_SHUTTER = "LED+Shutter"
PROPERTY_AIRPUFF = "AirPuff"
PROPERTY_PMT_NOISE = "PMT-noise"
PROPERTY_BG_MOTION = "BG-motion"

ANNOTATION_PROPERTIES: tuple[str, ...] = (
    PROPERTY_LED_SHUTTER,
    PROPERTY_AIRPUFF,
    PROPERTY_PMT_NOISE,
    PROPERTY_BG_MOTION,
)

# Display / export behavior keyed by property name.
PROPERTY_SPEC: dict[str, dict[str, Any]] = {
    PROPERTY_LED_SHUTTER: {
        "color": "#c0392b",
        "nan_display": True,  # when selected in GUI, NaN that range for display/Y-scale
        "description": "Shutter/LED artifact; NaN for display when selected",
    },
    PROPERTY_AIRPUFF: {
        "color": "#2980b9",
        "nan_display": False,
        "description": "Air-puff epoch marker for later quantification",
    },
    PROPERTY_PMT_NOISE: {
        "color": "#d35400",
        "nan_display": False,
        "description": "PMT noise epoch marker; spans only, no display NaNs",
    },
    PROPERTY_BG_MOTION: {
        "color": "#16a085",
        "nan_display": False,
        "description": "Background-evaluated motion marker; spans only, no display NaNs",
    },
}

DEFAULT_PROPERTY_SPEC: dict[str, Any] = {
    "color": "#8e44ad",
    "nan_display": False,
    "description": "Custom marker",
}

# Overlay boxes on traces: ~80% transparent so the curve stays readable.
SPAN_FILL_ALPHA = 51  # 20% opaque
SPAN_PEN_ALPHA = 64


def property_spec(name: str) -> dict[str, Any]:
    return PROPERTY_SPEC.get(str(name), DEFAULT_PROPERTY_SPEC)


def normalize_property_name(name: str) -> str:
    text = str(name).strip()
    if not text:
        raise ValueError("Annotation kind cannot be empty")
    lowered = text.lower()
    for known in ANNOTATION_PROPERTIES:
        if lowered == known.lower():
            return known
    return text


def is_led_shutter(name: str) -> bool:
    return str(name) == PROPERTY_LED_SHUTTER


def is_pmt_noise(name: str) -> bool:
    return str(name).strip().lower() == PROPERTY_PMT_NOISE.lower()


def is_bg_motion(name: str) -> bool:
    return str(name).strip().lower() == PROPERTY_BG_MOTION.lower()


def is_bundle_kind(name: str) -> bool:
    """Kinds that store every interval in one annotation (PMT-noise, BG-motion)."""
    return is_pmt_noise(name) or is_bg_motion(name)


def merge_ranges(
    ranges: Any, nframes: int | None = None
) -> list[list[int]]:
    """Sorted, clipped, inclusive [start, end] pairs; touching runs join."""
    out: list[list[int]] = []
    for item in ranges or []:
        try:
            a, b = int(item[0]), int(item[1])
        except (TypeError, ValueError, IndexError):
            continue
        if b < a:
            a, b = b, a
        if nframes is not None:
            last = max(int(nframes) - 1, 0)
            a = max(0, min(a, last))
            b = max(0, min(b, last))
        out.append([a, b])
    out.sort()
    merged: list[list[int]] = []
    for a, b in out:
        if merged and a <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return merged


def annotation_ranges(ann: dict[str, Any]) -> list[list[int]]:
    """Every interval in an annotation.

    One annotation can cover several disjoint intervals (a PMT-noise import
    is one feature made of many bursts). Annotations written before 'ranges'
    existed fall back to their single start/end span.
    """
    merged = merge_ranges(ann.get("ranges"))
    if merged:
        return merged
    try:
        s = int(ann["start_frame"])
        e = int(ann["end_frame"])
    except (KeyError, TypeError, ValueError):
        return []
    if e < s:
        s, e = e, s
    return [[s, e]]


def annotation_span(ann: dict[str, Any]) -> tuple[int, int]:
    """Outer [first start, last end] across every interval."""
    rs = annotation_ranges(ann)
    if not rs:
        return (0, 0)
    return (rs[0][0], max(b for _, b in rs))


def annotation_frame_count(ann: dict[str, Any]) -> int:
    return sum(b - a + 1 for a, b in annotation_ranges(ann))


def set_annotation_ranges(
    ann: dict[str, Any], ranges: Any, nframes: int | None = None
) -> list[list[int]]:
    """Replace an annotation's intervals, keeping start/end as the outer span."""
    merged = merge_ranges(ranges, nframes)
    if not merged:
        raise ValueError("An annotation needs at least one frame range")
    ann["ranges"] = merged
    ann["start_frame"] = merged[0][0]
    ann["end_frame"] = max(b for _, b in merged)
    return merged


def ensure_annotations(doc: dict[str, Any]) -> list[dict[str, Any]]:
    """Guarantee doc['annotations'] exists and return it."""
    anns = doc.get("annotations")
    if anns is None:
        anns = []
        doc["annotations"] = anns
    for ann in anns:
        if "label" not in ann or ann["label"] is None:
            ann["label"] = ""
        if not ann.get("ranges"):
            ann["ranges"] = annotation_ranges(ann)
    return anns


def annotation_kind(ann: dict[str, Any]) -> str:
    return str(ann.get("property") or "").strip() or "Untitled"


def annotation_list_text(ann: dict[str, Any]) -> str:
    rs = annotation_ranges(ann)
    kind = annotation_kind(ann)
    if len(rs) <= 1:
        s, e = rs[0] if rs else (0, 0)
        return f"{kind}  [{s}–{e}]"
    start, end = annotation_span(ann)
    frames = sum(b - a + 1 for a, b in rs)
    return f"{kind}  [{start}–{end}] {len(rs)} intervals, {frames} frames"


def next_ann_id(doc: dict[str, Any]) -> int:
    anns = ensure_annotations(doc)
    if not anns:
        return 0
    return max(int(a["ann_id"]) for a in anns) + 1


def get_annotation(doc: dict[str, Any], ann_id: int) -> dict[str, Any] | None:
    want = int(ann_id)
    for ann in ensure_annotations(doc):
        if int(ann["ann_id"]) == want:
            return ann
    return None


def make_annotation(
    ann_id: int,
    property_name: str,
    start_frame: int,
    end_frame: int,
    *,
    label: str = "",
    ranges: Any = None,
) -> dict[str, Any]:
    """One annotation; pass ranges to cover several disjoint intervals."""
    ann: dict[str, Any] = {
        "ann_id": int(ann_id),
        "property": normalize_property_name(property_name),
        "label": str(label),
    }
    set_annotation_ranges(ann, ranges if ranges else [[start_frame, end_frame]])
    return ann


def validate_annotation_frames(start: int, end: int, nframes: int) -> tuple[int, int]:
    if nframes <= 0:
        raise ValueError("nframes must be positive")
    s = int(np.clip(start, 0, nframes - 1))
    e = int(np.clip(end, 0, nframes - 1))
    if e < s:
        s, e = e, s
    return s, e


def nan_mask_from_annotations(
    nframes: int,
    annotations: list[dict[str, Any]],
    active_ann_ids: set[int] | list[int],
) -> np.ndarray:
    """
    Boolean mask True where display should be NaN.
    Only annotations that are active AND have nan_display property contribute.
    End frame is inclusive.
    """
    mask = np.zeros(int(nframes), dtype=bool)
    active = {int(i) for i in active_ann_ids}
    for ann in annotations:
        if int(ann["ann_id"]) not in active:
            continue
        spec = property_spec(str(ann["property"]))
        if not spec.get("nan_display", False):
            continue
        # Per interval, not the outer span: the quiet frames between two
        # noise bursts of one annotation must stay visible.
        for a, b in annotation_ranges(ann):
            s = max(0, min(int(a), nframes - 1))
            e = max(0, min(int(b), nframes - 1))
            if e >= s:
                mask[s : e + 1] = True
    return mask


def apply_nan_mask(trace: np.ndarray, nan_mask: np.ndarray) -> np.ndarray:
    """Return a float copy of trace with nan_mask positions set to NaN (stored data untouched)."""
    out = np.asarray(trace, dtype=np.float64).copy()
    if nan_mask.shape[0] != out.shape[0]:
        raise ValueError("nan_mask length must match trace")
    out[nan_mask] = np.nan
    return out
