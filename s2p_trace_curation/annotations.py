"""Global time-range annotations for curation sessions."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

AnnotationProperty = Literal["LED+Shutter", "AirPuff"]

ANNOTATION_PROPERTIES: tuple[AnnotationProperty, ...] = ("LED+Shutter", "AirPuff")

# Display / export behavior keyed by property name.
PROPERTY_SPEC: dict[str, dict[str, Any]] = {
    "LED+Shutter": {
        "color": "#c0392b",
        "nan_display": True,  # when selected in GUI, NaN that range for display/Y-scale
        "description": "Shutter/LED artifact; NaN for display when selected",
    },
    "AirPuff": {
        "color": "#2980b9",
        "nan_display": False,
        "description": "Air-puff epoch marker for later quantification",
    },
}


def ensure_annotations(doc: dict[str, Any]) -> list[dict[str, Any]]:
    """Guarantee doc['annotations'] exists and return it."""
    anns = doc.get("annotations")
    if anns is None:
        anns = []
        doc["annotations"] = anns
    return anns


def next_ann_id(doc: dict[str, Any]) -> int:
    anns = ensure_annotations(doc)
    if not anns:
        return 0
    return max(int(a["ann_id"]) for a in anns) + 1


def make_annotation(
    ann_id: int,
    property_name: str,
    start_frame: int,
    end_frame: int,
    *,
    label: str = "",
) -> dict[str, Any]:
    if property_name not in PROPERTY_SPEC:
        raise ValueError(f"Unknown annotation property: {property_name}")
    s = int(start_frame)
    e = int(end_frame)
    if e < s:
        s, e = e, s
    return {
        "ann_id": int(ann_id),
        "property": str(property_name),
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
        spec = PROPERTY_SPEC.get(str(ann["property"]), {})
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
