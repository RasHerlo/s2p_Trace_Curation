"""Dialog: Merge two suite2p folders into suite2p_merged (or custom name)."""

from __future__ import annotations

from pathlib import Path

from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from s2p_trace_curation.merge_suite2p import (
    BinCheckMode,
    MergeError,
    merge_suite2p_folders,
    suggested_output_parent,
)
from s2p_trace_curation.user_settings import last_open_start_dir, load_settings, save_settings

QDialog = QtWidgets.QDialog
QDialogButtonBox = QtWidgets.QDialogButtonBox
QFileDialog = QtWidgets.QFileDialog
QFormLayout = QtWidgets.QFormLayout
QGroupBox = QtWidgets.QGroupBox
QHBoxLayout = QtWidgets.QHBoxLayout
QLabel = QtWidgets.QLabel
QLineEdit = QtWidgets.QLineEdit
QMessageBox = QtWidgets.QMessageBox
QPushButton = QtWidgets.QPushButton
QRadioButton = QtWidgets.QRadioButton
QVBoxLayout = QtWidgets.QVBoxLayout
Qt = QtCore.Qt
QCursor = QtGui.QCursor


class MergeSuite2pDialog(QDialog):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Merge s2p folders")
        self.setMinimumWidth(560)
        self.merged_dir: Path | None = None

        settings = load_settings()
        start = last_open_start_dir(settings)

        layout = QVBoxLayout(self)
        form = QFormLayout()

        self.edit_a = QLineEdit()
        self.edit_b = QLineEdit()
        self.edit_out_parent = QLineEdit()
        self.edit_out_name = QLineEdit("suite2p_merged")
        if start:
            self.edit_a.setPlaceholderText(start)
            self.edit_b.setPlaceholderText(start)
            self.edit_out_parent.setText(str(Path(start).parent if Path(start).name.startswith("suite2p") else start))

        form.addRow("Folder A (first ROIs)", self._browse_row(self.edit_a, "Select first suite2p folder"))
        form.addRow("Folder B (second ROIs)", self._browse_row(self.edit_b, "Select second suite2p folder"))
        form.addRow("Output parent folder", self._browse_row(self.edit_out_parent, "Select output parent folder", dir_only=True))
        form.addRow("Output folder name", self.edit_out_name)
        layout.addLayout(form)

        check_box = QGroupBox("data.bin identity check")
        check_layout = QVBoxLayout(check_box)
        self.radio_sample = QRadioButton("Sample match (fast: size + first/middle/last 1 MiB) — default")
        self.radio_full = QRadioButton("Full byte match (slow on large movies)")
        mode = settings.get("merge_bin_check", "sample")
        if mode == "full":
            self.radio_full.setChecked(True)
        else:
            self.radio_sample.setChecked(True)
        check_layout.addWidget(self.radio_sample)
        check_layout.addWidget(self.radio_full)
        check_layout.addWidget(
            QLabel("Merge requires identical data.bin under the selected check.")
        )
        layout.addWidget(check_box)

        self.lbl_hint = QLabel("")
        self.lbl_hint.setWordWrap(True)
        layout.addWidget(self.lbl_hint)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Merge")
        buttons.accepted.connect(self._on_merge)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.edit_a.textChanged.connect(self._maybe_suggest_parent)
        self.edit_b.textChanged.connect(self._maybe_suggest_parent)

    def _browse_row(self, edit: QLineEdit, title: str, dir_only: bool = True) -> QtWidgets.QWidget:
        row = QtWidgets.QWidget()
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(edit, stretch=1)
        btn = QPushButton("Browse…")

        def _browse() -> None:
            start = edit.text().strip() or last_open_start_dir() or ""
            path = QFileDialog.getExistingDirectory(self, title, start)
            if path:
                edit.setText(path)

        btn.clicked.connect(_browse)
        h.addWidget(btn)
        return row

    def _bin_mode(self) -> BinCheckMode:
        return "full" if self.radio_full.isChecked() else "sample"

    def _maybe_suggest_parent(self) -> None:
        a = self.edit_a.text().strip()
        b = self.edit_b.text().strip()
        if not a or not b:
            return
        parent = suggested_output_parent(Path(a), Path(b))
        if parent is not None:
            self.edit_out_parent.setText(str(parent))
            self.lbl_hint.setText(f"Suggested output parent (shared): {parent}")

    def _on_merge(self) -> None:
        a = self.edit_a.text().strip()
        b = self.edit_b.text().strip()
        out_parent = self.edit_out_parent.text().strip()
        name = self.edit_out_name.text().strip() or "suite2p_merged"
        if not a or not b or not out_parent:
            QMessageBox.warning(
                self, "Merge", "Please set Folder A, Folder B, and the output parent folder."
            )
            return

        out_root = Path(out_parent) / name
        overwrite = False
        if out_root.exists():
            ans = QMessageBox.warning(
                self,
                "Output exists",
                f"The folder already exists:\n\n{out_root}\n\n"
                "Overwrite it?\n\n"
                "Yes = delete and recreate\n"
                "No = cancel merge",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if ans != QMessageBox.StandardButton.Yes:
                return
            overwrite = True

        mode = self._bin_mode()
        save_settings({"merge_bin_check": mode})
        self.setEnabled(False)
        QtWidgets.QApplication.setOverrideCursor(QCursor(Qt.CursorShape.WaitCursor))
        try:
            merged = merge_suite2p_folders(
                Path(a),
                Path(b),
                Path(out_parent),
                name,
                bin_check=mode,
                overwrite=overwrite,
            )
        except FileExistsError as exc:
            QMessageBox.warning(self, "Merge", f"Output already exists:\n{exc}")
            return
        except (MergeError, FileNotFoundError, OSError, ValueError) as exc:
            QMessageBox.critical(self, "Merge failed", str(exc))
            return
        except Exception as exc:
            QMessageBox.critical(self, "Merge failed", str(exc))
            return
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
            self.setEnabled(True)

        self.merged_dir = merged
        QMessageBox.information(
            self,
            "Merge complete",
            f"Wrote merged suite2p folder:\n{merged}\n\n"
            f"See merge_note.txt beside plane0/.\n"
            "The merged folder will now open in the main window.",
        )
        self.accept()
