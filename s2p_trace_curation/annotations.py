"""Global time-range annotations for curation sessions."""

from __future__ import annotations

from typing import Any

import numpy as np

PROPERTY_LED_SHUTTER = "LED+Shutter"
PROPERTY_AIRPUFF = "AirPuff"
PROPERTY_PMT_NOISE = "PMT-noise"

ANNOTATION_PROPERTIES: tuple[str, ...] = (
    PROPERTY_LED_SHUTTER,
    PROPERTY_AIRPUFF,
    PROPERTY_PMT_NOISE,
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
        "nan_display": True,
        "description": "PMT noise epoch; NaN for display when selected",
    },
}

DEFAULT_PROPERTY_SPEC: dict[str, Any] = {
    "color": "#8e44ad",
    "nan_display": False,
    "description": "Custom marker",
}


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


def ensure_annotations(doc: dict[str, Any]) -> list[dict[str, Any]]:
    """Guarantee doc['annotations'] exists and return it."""
    anns = doc.get("annotations")
    if anns is None:
        anns = []
        doc["annotations"] = anns
    for ann in anns:
        if "label" not in ann or ann["label"] is None:
            ann["label"] = ""
    return anns


def annotation_kind(ann: dict[str, Any]) -> str:
    return str(ann.get("property") or "").strip() or "Untitled"


def annotation_list_text(ann: dict[str, Any]) -> str:
    return f"{annotation_kind(ann)}  [{ann['start_frame']}–{ann['end_frame']}]"


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
) -> dict[str, Any]:
    kind = normalize_property_name(property_name)
    s = int(start_frame)
    e = int(end_frame)
    if e < s:
        s, e = e, s
    return {
        "ann_id": int(ann_id),
        "property": kind,
        "start_frame": s,
        "end_frame": e,  # inclusive
        "label": str(label),
    }


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
        s = int(ann["start_frame"])
        e = int(ann["end_frame"])
        s = max(0, min(s, nframes - 1))
        e = max(0, min(e, nframes - 1))
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
