"""X-axis units combo: frames vs seconds, with fps in the seconds label."""

from __future__ import annotations

from typing import Any

from pyqtgraph.Qt import QtWidgets

UNITS_FRAMES = "frames"
UNITS_SECONDS = "seconds"

QComboBox = QtWidgets.QComboBox


def normalize_x_units(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text == UNITS_SECONDS or text.startswith("seconds"):
        return UNITS_SECONDS
    return UNITS_FRAMES


def seconds_units_label(fs: float | None) -> str:
    if fs is None:
        return UNITS_SECONDS
    return f"{UNITS_SECONDS} ({float(fs):.2f}fps)"


def fill_x_units_combo(
    combo: QComboBox,
    fs: float | None,
    selected: Any = None,
) -> None:
    """Two items: 'frames' and 'seconds (2.52fps)'. Stored data stay frames/seconds."""
    if selected is None:
        selected = combo.currentData()
        if selected is None:
            selected = combo.currentText()
    selected = normalize_x_units(selected)
    combo.blockSignals(True)
    try:
        combo.clear()
        combo.addItem(UNITS_FRAMES, UNITS_FRAMES)
        combo.addItem(seconds_units_label(fs), UNITS_SECONDS)
        idx = combo.findData(selected)
        combo.setCurrentIndex(idx if idx >= 0 else 0)
    finally:
        combo.blockSignals(False)


def combo_x_units(combo: QComboBox) -> str:
    data = combo.currentData()
    if data is not None:
        return normalize_x_units(data)
    return normalize_x_units(combo.currentText())
