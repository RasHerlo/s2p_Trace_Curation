"""Non-modal ROI Set window: publish a shared ROI set, manage its experiments, apply it."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from s2p_trace_curation.curation import save_curation
from s2p_trace_curation.mask_edit import ExtractCancelled
from s2p_trace_curation.roisets import (
    IMAGE_KEYS,
    ROISET_SUFFIX,
    apply_master,
    backup_experiment_pickle,
    default_channel,
    default_master_name,
    experiment_dir,
    find_experiment,
    link_source_doc,
    load_master,
    master_from_doc,
    register_experiment,
    save_master,
    scan_suite2p_folders,
    shifted_master_masks,
    update_master_on_disk,
    upsert_experiment,
)
from s2p_trace_curation.suite2p_io import resolve_suite2p_dir
from s2p_trace_curation.user_settings import load_settings, save_settings

Qt = QtCore.Qt
QApplication = QtWidgets.QApplication
QCheckBox = QtWidgets.QCheckBox
QComboBox = QtWidgets.QComboBox
QDialog = QtWidgets.QDialog
QDialogButtonBox = QtWidgets.QDialogButtonBox
QFileDialog = QtWidgets.QFileDialog
QFormLayout = QtWidgets.QFormLayout
QGroupBox = QtWidgets.QGroupBox
QHBoxLayout = QtWidgets.QHBoxLayout
QHeaderView = QtWidgets.QHeaderView
QLabel = QtWidgets.QLabel
QLineEdit = QtWidgets.QLineEdit
QMessageBox = QtWidgets.QMessageBox
QProgressBar = QtWidgets.QProgressBar
QProgressDialog = QtWidgets.QProgressDialog
QPushButton = QtWidgets.QPushButton
QSpinBox = QtWidgets.QSpinBox
QTableWidget = QtWidgets.QTableWidget
QTableWidgetItem = QtWidgets.QTableWidgetItem
QVBoxLayout = QtWidgets.QVBoxLayout

# Same-FOV pairs scored 0.83–0.96 on the 240916_pl100 data; other FOVs < 0.05.
SAME_FOV_SCORE = 0.5

COL_APPROVED, COL_PATH, COL_TAG, COL_SHIFT, COL_SCORE, COL_REV = range(6)


def _default_master_folder(suite2p_dir: Path) -> Path:
    """``…/<dataset>/<group>/<exp>/DATA/<chan>/suite2p_x`` → ``<dataset>``."""
    p = Path(suite2p_dir).resolve()
    parts = p.parts
    if len(parts) >= 6 and parts[-3].upper() == "DATA":
        return p.parents[4]
    return p.parents[1] if len(parts) >= 3 else p.parent


class _PublishDialog(QDialog):
    def __init__(self, suite2p_dir: Path, parent: Any = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Publish shared ROI set")
        self.setMinimumWidth(560)
        form = QFormLayout(self)
        folder = _default_master_folder(suite2p_dir)
        channel = default_channel(suite2p_dir)
        self.ed_name = QLineEdit(default_master_name(folder, channel))
        self.ed_name.setToolTip("Follows the folder and channel until you edit it.")
        self._auto_name = self.ed_name.text()
        self.ed_channel = QLineEdit(channel)
        self.ed_tag = QLineEdit("")
        self.ed_tag.setPlaceholderText("Condition label for this experiment, e.g. LED 2s")
        self.ed_folder = QLineEdit(str(folder))
        btn_browse = QPushButton("Browse…")
        btn_browse.clicked.connect(self._browse)
        folder_row = QHBoxLayout()
        folder_row.addWidget(self.ed_folder, stretch=1)
        folder_row.addWidget(btn_browse)
        self.lbl_path = QLabel("")
        self.lbl_path.setWordWrap(True)
        form.addRow("Name", self.ed_name)
        form.addRow("Channel", self.ed_channel)
        form.addRow("Tag", self.ed_tag)
        form.addRow("Folder", folder_row)
        form.addRow("File", self.lbl_path)
        for ed in (self.ed_folder, self.ed_channel):
            ed.textChanged.connect(self._sync_auto_name)
        self.ed_name.textChanged.connect(self._sync_path)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Publish")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)
        self._sync_path()

    def _browse(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Folder for the shared ROI set", self.ed_folder.text())
        if path:
            self.ed_folder.setText(path)

    def output_path(self) -> Path:
        name = self.ed_name.text().strip() or "roiset"
        return Path(self.ed_folder.text().strip()) / f"{name}{ROISET_SUFFIX}"

    def _sync_auto_name(self) -> None:
        if self.ed_name.text() == self._auto_name:
            self._auto_name = default_master_name(
                Path(self.ed_folder.text().strip()), self.ed_channel.text().strip()
            )
            self.ed_name.setText(self._auto_name)
        self._sync_path()

    def _sync_path(self) -> None:
        self.lbl_path.setText(str(self.output_path()))


class RoiSetWindow(QDialog):
    def __init__(self, main: Any) -> None:
        super().__init__(main)
        self.main = main
        self.setWindowTitle("ROI Set")
        self.setModal(False)
        self.setMinimumSize(820, 680)
        self._filling = False
        self._cancel = False
        self._estimated_score: float | None = None

        layout = QVBoxLayout(self)

        grp_master = QGroupBox("Shared ROI set (master)")
        master_layout = QVBoxLayout(grp_master)
        self.lbl_master = QLabel("")
        self.lbl_master.setWordWrap(True)
        self.lbl_master.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        master_layout.addWidget(self.lbl_master)
        master_btns = QHBoxLayout()
        self.btn_publish = QPushButton("Publish this experiment…")
        self.btn_publish.setToolTip(
            "Create a shared ROI set from this experiment's iscell=True ROIs. "
            "This experiment becomes its first approved member."
        )
        self.btn_publish.clicked.connect(self._on_publish)
        self.btn_open_master = QPushButton("Open master…")
        self.btn_open_master.setToolTip(
            f"Pick a *{ROISET_SUFFIX} file to manage or apply here (does not change this experiment)"
        )
        self.btn_open_master.clicked.connect(self._on_open_master)
        master_btns.addWidget(self.btn_publish)
        master_btns.addWidget(self.btn_open_master)
        master_btns.addStretch(1)
        master_layout.addLayout(master_btns)
        layout.addWidget(grp_master)

        grp_exp = QGroupBox("Experiments")
        exp_layout = QVBoxLayout(grp_exp)
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(["Approved", "Experiment", "Tag", "Shift", "Score", "Synced rev"])
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(COL_PATH, QHeaderView.ResizeMode.Stretch)
        for col in (COL_APPROVED, COL_TAG, COL_SHIFT, COL_SCORE, COL_REV):
            header.setSectionResizeMode(col, QHeaderView.ResizeMode.ResizeToContents)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.itemChanged.connect(self._on_item_changed)
        self.table.itemDoubleClicked.connect(lambda _item: self._on_open_experiment())
        exp_layout.addWidget(self.table, stretch=1)
        hint = QLabel(
            f"Scan and Register only suggest: a score ≥ {SAME_FOV_SCORE} usually means the same FOV. "
            "Nothing is approved until you tick it. Double-click a row to open that experiment."
        )
        hint.setWordWrap(True)
        exp_layout.addWidget(hint)
        exp_btns = QHBoxLayout()
        self.btn_scan = QPushButton("Scan folder…")
        self.btn_scan.setToolTip(
            "Find suite2p folders under a folder, register each against the master, "
            "and add them unapproved"
        )
        self.btn_scan.clicked.connect(self._on_scan)
        self.chk_scan_channel = QCheckBox("only matching channel")
        self.chk_scan_channel.setChecked(True)
        self.chk_scan_channel.setToolTip("Keep only folders whose path contains the master's channel name")
        self.btn_add = QPushButton("Add folder…")
        self.btn_add.clicked.connect(self._on_add_folder)
        self.btn_register = QPushButton("Register selected")
        self.btn_register.setToolTip("Re-estimate shift and score for the selected rows")
        self.btn_register.clicked.connect(self._on_register_selected)
        self.btn_remove = QPushButton("Remove")
        self.btn_remove.setToolTip("Remove the selected rows from the registry (experiments are untouched)")
        self.btn_remove.clicked.connect(self._on_remove)
        self.btn_open_exp = QPushButton("Open")
        self.btn_open_exp.setToolTip("Open the selected experiment in the main window")
        self.btn_open_exp.clicked.connect(self._on_open_experiment)
        for w in (self.btn_scan, self.chk_scan_channel, self.btn_add, self.btn_register, self.btn_remove):
            exp_btns.addWidget(w)
        exp_btns.addStretch(1)
        exp_btns.addWidget(self.btn_open_exp)
        exp_layout.addLayout(exp_btns)
        layout.addWidget(grp_exp, stretch=1)

        grp_here = QGroupBox("This experiment")
        here_form = QFormLayout(grp_here)
        self.lbl_here = QLabel("")
        self.lbl_here.setWordWrap(True)
        here_form.addRow(self.lbl_here)
        self.cmb_image = QComboBox()
        for key in IMAGE_KEYS:
            self.cmb_image.addItem(key, key)
        self.cmb_image.setToolTip("Image registered against the master's reference image")
        self.btn_estimate = QPushButton("Estimate shift")
        self.btn_estimate.clicked.connect(self._on_estimate)
        img_row = QHBoxLayout()
        img_row.addWidget(self.cmb_image)
        img_row.addWidget(self.btn_estimate)
        img_row.addStretch(1)
        here_form.addRow("Image", img_row)
        self.spin_dy = QSpinBox()
        self.spin_dx = QSpinBox()
        for spin in (self.spin_dy, self.spin_dx):
            spin.setRange(-256, 256)
            spin.valueChanged.connect(self._on_shift_edited)
        self.spin_dy.setToolTip("Rows: experiment y = master y + dY")
        self.spin_dx.setToolTip("Columns: experiment x = master x + dX")
        self.lbl_score = QLabel("—")
        self.chk_preview = QCheckBox("Preview on W1")
        self.chk_preview.setToolTip("Outline the shared ROIs on W1 (magenta) at this shift")
        self.chk_preview.toggled.connect(self._sync_preview)
        shift_row = QHBoxLayout()
        shift_row.addWidget(QLabel("dY"))
        shift_row.addWidget(self.spin_dy)
        shift_row.addWidget(QLabel("dX"))
        shift_row.addWidget(self.spin_dx)
        shift_row.addWidget(QLabel("score"))
        shift_row.addWidget(self.lbl_score)
        shift_row.addStretch(1)
        shift_row.addWidget(self.chk_preview)
        here_form.addRow("Shift", shift_row)
        apply_row = QHBoxLayout()
        self.btn_apply = QPushButton("Apply shared set to this experiment")
        self.btn_apply.clicked.connect(self._on_apply)
        self.btn_cancel_job = QPushButton("Cancel")
        self.btn_cancel_job.setEnabled(False)
        self.btn_cancel_job.clicked.connect(self._on_cancel_job)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setVisible(False)
        apply_row.addWidget(self.btn_apply)
        apply_row.addWidget(self.progress, stretch=1)
        apply_row.addWidget(self.btn_cancel_job)
        here_form.addRow(apply_row)
        layout.addWidget(grp_here)

        close_row = QHBoxLayout()
        close_row.addStretch(1)
        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.close)
        close_row.addWidget(btn_close)
        layout.addLayout(close_row)

        self.refresh_from_main()

    # ------------------------------------------------------------ state
    def _master(self) -> dict[str, Any] | None:
        return self.main._master

    def _master_path(self) -> Path | None:
        return self.main._master_path

    def _suite2p_dir(self) -> Path | None:
        return self.main.suite2p_dir

    def closeEvent(self, event) -> None:  # noqa: N802
        self.main.set_master_preview(None)
        super().closeEvent(event)

    def refresh_from_main(self) -> None:
        master = self._master()
        path = self._master_path()
        if master is None:
            err = self.main._master_error
            self.lbl_master.setText(
                "No shared ROI set loaded." + (f" ({err})" if err else "")
                + " Publish this experiment, or open an existing one."
            )
        else:
            m = master["meta"]
            self.lbl_master.setText(
                f"<b>{m['name']}</b> — channel {m.get('channel') or '?'} · rev {m['rev']} · "
                f"{len(master['rois'])} ROIs · {len(master.get('bg_rois') or [])} BG ROIs<br>"
                f"{path}"
            )
        has_master = master is not None
        for w in (self.btn_scan, self.btn_add, self.btn_register, self.btn_remove,
                  self.btn_open_exp, self.chk_scan_channel, self.btn_estimate,
                  self.btn_apply, self.chk_preview, self.spin_dy, self.spin_dx, self.cmb_image):
            w.setEnabled(has_master)
        self.btn_publish.setEnabled(self.main.doc is not None)
        self._fill_table()
        self._load_shift_defaults()
        self._sync_here_label()
        self._sync_preview()

    def _fill_table(self) -> None:
        master = self._master()
        path = self._master_path()
        here = self._suite2p_dir()
        self._filling = True
        try:
            self.table.setRowCount(0)
            if master is None or path is None:
                return
            for entry in master.get("experiments") or []:
                r = self.table.rowCount()
                self.table.insertRow(r)
                full = experiment_dir(path, entry)
                is_here = here is not None and str(full).casefold() == str(Path(here).resolve()).casefold()
                chk = QTableWidgetItem("")
                chk.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
                chk.setCheckState(Qt.CheckState.Checked if entry.get("approved") else Qt.CheckState.Unchecked)
                chk.setData(Qt.ItemDataRole.UserRole, str(entry["path"]))
                self.table.setItem(r, COL_APPROVED, chk)
                label = str(entry["path"]) + ("   (this experiment)" if is_here else "")
                item = QTableWidgetItem(label)
                item.setToolTip(str(full) + ("" if full.exists() else "\n(folder not found)"))
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                if is_here:
                    font = item.font()
                    font.setBold(True)
                    item.setFont(font)
                if not full.exists():
                    item.setForeground(QtGui.QBrush(QtGui.QColor("#999999")))
                self.table.setItem(r, COL_PATH, item)
                self.table.setItem(r, COL_TAG, QTableWidgetItem(str(entry.get("tag") or "")))
                dy, dx = (int(v) for v in (entry.get("shift") or [0, 0]))
                self._set_ro(r, COL_SHIFT, f"({dy:+d}, {dx:+d})")
                score = entry.get("score")
                score_item = self._set_ro(r, COL_SCORE, "—" if score is None else f"{float(score):.3f}")
                if score is not None:
                    color = "#2e7d32" if float(score) >= SAME_FOV_SCORE else "#c62828"
                    score_item.setForeground(QtGui.QBrush(QtGui.QColor(color)))
                rev = entry.get("synced_rev")
                self._set_ro(r, COL_REV, "—" if rev is None else str(rev))
        finally:
            self._filling = False

    def _set_ro(self, row: int, col: int, text: str) -> QTableWidgetItem:
        item = QTableWidgetItem(text)
        item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        self.table.setItem(row, col, item)
        return item

    def _row_path_key(self, row: int) -> str | None:
        item = self.table.item(row, COL_APPROVED)
        return None if item is None else str(item.data(Qt.ItemDataRole.UserRole))

    def _selected_keys(self) -> list[str]:
        rows = sorted({i.row() for i in self.table.selectedIndexes()})
        return [k for k in (self._row_path_key(r) for r in rows) if k]

    def _write_registry(self, fn) -> bool:
        path = self._master_path()
        if path is None:
            return False
        try:
            master = update_master_on_disk(path, fn)
        except Exception as exc:
            QMessageBox.critical(self, "ROI Set", f"Could not update the master:\n{exc}")
            return False
        self.main.set_master(master, path)
        return True

    def _load_shift_defaults(self) -> None:
        """Shift spins from the registry entry, else from this experiment's link."""
        master = self._master()
        path = self._master_path()
        here = self._suite2p_dir()
        dy = dx = 0
        score = None
        key = "meanImg"
        if master is not None and path is not None and here is not None:
            entry = find_experiment(master, path, here)
            link = (self.main.doc or {}).get("master_link") or {}
            if entry is not None and entry.get("score") is not None:
                dy, dx = (int(v) for v in entry["shift"])
                score = entry.get("score")
                key = str(entry.get("image_key") or key)
            elif link:
                dy, dx = (int(v) for v in (link.get("shift") or [0, 0]))
                score = link.get("score")
                key = str(link.get("image_key") or key)
        self._filling = True
        self.spin_dy.setValue(dy)
        self.spin_dx.setValue(dx)
        idx = self.cmb_image.findData(key)
        self.cmb_image.setCurrentIndex(idx if idx >= 0 else 0)
        self._filling = False
        self._estimated_score = None if score is None else float(score)
        self.lbl_score.setText("—" if score is None else f"{float(score):.3f}")

    def _sync_here_label(self) -> None:
        master = self._master()
        path = self._master_path()
        here = self._suite2p_dir()
        doc = self.main.doc
        if doc is None or here is None:
            self.lbl_here.setText("No experiment open.")
            return
        link = doc.get("master_link")
        parts = []
        if link:
            parts.append(f"Linked to <b>{link.get('name')}</b> rev {link.get('rev')}, shift {tuple(link.get('shift') or [0, 0])}.")
        else:
            parts.append("Not linked to a shared ROI set.")
        if master is not None and path is not None:
            entry = find_experiment(master, path, here)
            if entry is None:
                parts.append("Not in this master's experiment list.")
            elif not entry.get("approved"):
                parts.append("Listed but not approved.")
            else:
                parts.append("Approved for this master.")
        self.lbl_here.setText(" ".join(parts))

    # --------------------------------------------------------- registry
    def _on_item_changed(self, item: QTableWidgetItem) -> None:
        if self._filling:
            return
        key = self._row_path_key(item.row())
        path = self._master_path()
        if key is None or path is None:
            return
        if item.column() == COL_APPROVED:
            approved = item.checkState() == Qt.CheckState.Checked

            def fn(m: dict[str, Any]) -> None:
                for e in m["experiments"]:
                    if str(e["path"]) == key:
                        e["approved"] = approved
        elif item.column() == COL_TAG:
            tag = item.text().strip()

            def fn(m: dict[str, Any]) -> None:
                for e in m["experiments"]:
                    if str(e["path"]) == key:
                        e["tag"] = tag
        else:
            return
        self._write_registry(fn)
        self._sync_here_label()

    def _register_dirs(self, dirs: list[Path], key: str, title: str) -> dict[str, tuple[int, int, float | None]]:
        """Shift + score per folder, with a cancellable progress dialog."""
        master = self._master()
        assert master is not None
        out: dict[str, tuple[int, int, float | None]] = {}
        dlg = QProgressDialog(title, "Cancel", 0, len(dirs), self)
        dlg.setWindowModality(Qt.WindowModality.WindowModal)
        dlg.setMinimumDuration(0)
        errors = []
        for i, d in enumerate(dirs):
            dlg.setValue(i)
            dlg.setLabelText(f"{title}\n{d}")
            QApplication.processEvents()
            if dlg.wasCanceled():
                break
            try:
                out[str(d)] = register_experiment(master, d, key)
            except Exception as exc:
                errors.append(f"{d}: {exc}")
        dlg.setValue(len(dirs))
        if errors:
            QMessageBox.warning(self, "Register", "Some folders could not be registered:\n" + "\n".join(errors[:10]))
        return out

    def _on_scan(self) -> None:
        master = self._master()
        path = self._master_path()
        if master is None or path is None:
            return
        root = QFileDialog.getExistingDirectory(self, "Scan folder for experiments", str(path.parent))
        if not root:
            return
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            found = scan_suite2p_folders(Path(root))
        finally:
            QApplication.restoreOverrideCursor()
        chan = str(master["meta"].get("channel") or "").casefold()
        if self.chk_scan_channel.isChecked() and chan:
            found = [p for p in found if any(part.casefold() == chan for part in p.parts)]
        new = [p for p in found if find_experiment(master, path, p) is None]
        if not new:
            QMessageBox.information(self, "Scan", f"No new suite2p folders under {root}.")
            return
        key = str(self.cmb_image.currentData() or "meanImg")
        results = self._register_dirs(new, key, "Registering against the master…")

        def fn(m: dict[str, Any]) -> None:
            for d in new:
                res = results.get(str(d))
                if res is None:
                    continue
                dy, dx, score = res
                upsert_experiment(m, path, d, shift=[dy, dx], score=score, image_key=key)

        if self._write_registry(fn):
            likely = sum(1 for r in results.values() if r[2] is not None and r[2] >= SAME_FOV_SCORE)
            self.refresh_from_main()
            QMessageBox.information(
                self,
                "Scan",
                f"Added {len(results)} folder(s), {likely} with score ≥ {SAME_FOV_SCORE}. "
                "Tick Approved for the ones that share this FOV.",
            )

    def _on_add_folder(self) -> None:
        master = self._master()
        path = self._master_path()
        if master is None or path is None:
            return
        raw = QFileDialog.getExistingDirectory(self, "Add suite2p folder", str(path.parent))
        if not raw:
            return
        try:
            d = resolve_suite2p_dir(Path(raw))
        except FileNotFoundError as exc:
            QMessageBox.warning(self, "Add folder", str(exc))
            return
        key = str(self.cmb_image.currentData() or "meanImg")
        res = self._register_dirs([d], key, "Registering against the master…").get(str(d))
        dy, dx, score = res if res is not None else (0, 0, None)
        self._write_registry(
            lambda m: upsert_experiment(m, path, d, shift=[dy, dx], score=score, image_key=key)
        )
        self.refresh_from_main()

    def _on_register_selected(self) -> None:
        path = self._master_path()
        keys = self._selected_keys()
        if path is None or not keys:
            return
        dirs = [experiment_dir(path, {"path": k}) for k in keys]
        key = str(self.cmb_image.currentData() or "meanImg")
        results = self._register_dirs(dirs, key, "Registering…")

        def fn(m: dict[str, Any]) -> None:
            for k, d in zip(keys, dirs):
                res = results.get(str(d))
                if res is None:
                    continue
                for e in m["experiments"]:
                    if str(e["path"]) == k:
                        e["shift"] = [res[0], res[1]]
                        e["score"] = res[2]
                        e["image_key"] = key

        self._write_registry(fn)
        self.refresh_from_main()

    def _on_remove(self) -> None:
        keys = self._selected_keys()
        if not keys:
            return
        reply = QMessageBox.question(
            self,
            "Remove",
            f"Remove {len(keys)} experiment(s) from this master's list? "
            "Their folders and pickles are not touched.",
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        def fn(m: dict[str, Any]) -> None:
            m["experiments"] = [e for e in m["experiments"] if str(e["path"]) not in keys]

        self._write_registry(fn)
        self.refresh_from_main()

    def _confirm_main_saved(self, purpose: str) -> bool:
        """Offer to save unsaved changes in the main window. False = abort."""
        if not self.main.dirty:
            return True
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Warning)
        box.setWindowTitle("Unsaved changes")
        box.setText(f"This experiment has unsaved changes. Save them before {purpose}?")
        btn_save = box.addButton("Save", QMessageBox.ButtonRole.AcceptRole)
        btn_discard = box.addButton("Discard", QMessageBox.ButtonRole.DestructiveRole)
        box.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)
        box.setDefaultButton(btn_save)
        box.exec()
        clicked = box.clickedButton()
        if clicked is btn_save:
            return bool(self.main.save_session())
        return clicked is btn_discard

    def _on_open_experiment(self) -> None:
        path = self._master_path()
        keys = self._selected_keys()
        if path is None or len(keys) != 1:
            return
        d = experiment_dir(path, {"path": keys[0]})
        if not d.exists():
            QMessageBox.warning(self, "Open", f"Folder not found:\n{d}")
            return
        if self.main._mask_edit_active:
            QMessageBox.information(self, "Open", "Finish or cancel the mask edit first.")
            return
        if not self._confirm_main_saved("opening another experiment"):
            return
        master, master_path = self._master(), path
        try:
            self.main._load_suite2p(d)
        except Exception as exc:
            QMessageBox.critical(self, "Open failed", str(exc))
            return
        if self.main._master is None:
            self.main.set_master(master, master_path)
        self.refresh_from_main()

    # ----------------------------------------------------------- master
    def _on_open_master(self) -> None:
        start = str(load_settings().get("last_roiset_dir") or "")
        if not start and self._suite2p_dir() is not None:
            start = str(_default_master_folder(self._suite2p_dir()))
        raw, _ = QFileDialog.getOpenFileName(
            self, "Open shared ROI set", start, f"ROI sets (*{ROISET_SUFFIX});;All files (*)"
        )
        if not raw:
            return
        try:
            master = load_master(Path(raw))
        except Exception as exc:
            QMessageBox.critical(self, "Open master", str(exc))
            return
        save_settings({"last_roiset_dir": str(Path(raw).parent)})
        self.main.set_master(master, Path(raw))
        self.refresh_from_main()

    def _on_publish(self) -> None:
        main = self.main
        doc = main.doc
        here = self._suite2p_dir()
        if doc is None or here is None:
            return
        if main._mask_edit_active:
            QMessageBox.information(self, "Publish", "Finish or cancel the mask edit first.")
            return
        dlg = _PublishDialog(here, self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        out = dlg.output_path()
        n_cells = sum(1 for r in doc["rois"] if bool(r.get("iscell", True)))
        n_rest = len(doc["rois"]) - n_cells
        if n_cells == 0:
            QMessageBox.warning(self, "Publish", "There are no iscell=True ROIs to publish.")
            return
        if out.exists():
            reply = QMessageBox.warning(
                self,
                "Publish",
                f"{out.name} already exists. Overwriting starts a new master at rev 1 "
                "with only this experiment approved; experiments linked to the old one "
                "keep their ROIs but should be re-applied. Overwrite?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return
        text = (
            f"Publish {n_cells} iscell=True ROIs and {len(doc.get('bg_rois') or [])} BG ROIs "
            f"as '{dlg.ed_name.text().strip()}'?\n\n"
            "This experiment becomes the first approved member. Its working set becomes "
            f"those {n_cells} ROIs"
            + (f"; the other {n_rest} stay visible under Originals" if n_rest else "")
            + ". The current pickle is kept as trc_curation.original.pkl (first time only) "
            "and trc_curation.bak.pkl."
        )
        if QMessageBox.question(self, "Publish", text) != QMessageBox.StandardButton.Yes:
            return
        if not self._confirm_main_saved("publishing"):
            return
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            master, uid_map = master_from_doc(
                doc,
                here,
                out,
                name=dlg.ed_name.text().strip() or out.stem,
                channel=dlg.ed_channel.text().strip(),
                tag=dlg.ed_tag.text().strip(),
            )
            save_master(master, out)
            backup_experiment_pickle(here)
            linked = deepcopy(doc)
            link_source_doc(linked, here, master, out, uid_map)
            save_curation(linked, here)
        except Exception as exc:
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(self, "Publish failed", str(exc))
            return
        QApplication.restoreOverrideCursor()
        save_settings({"last_roiset_dir": str(out.parent)})
        main.dirty = False
        main.reload_session()
        self.refresh_from_main()
        main.statusBar().showMessage(f"Published {out} — {len(master['rois'])} ROIs")

    # ------------------------------------------------------------ apply
    def _on_shift_edited(self) -> None:
        if self._filling:
            return
        self._estimated_score = None
        self.lbl_score.setText("— (edited)")
        self._sync_preview()

    def _sync_preview(self) -> None:
        master = self._master()
        if master is None or not self.chk_preview.isChecked() or not self.isVisible():
            self.main.set_master_preview(None)
            return
        self.main.set_master_preview(
            shifted_master_masks(master, int(self.spin_dy.value()), int(self.spin_dx.value()))
        )

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        self._sync_preview()

    def _on_estimate(self) -> None:
        master = self._master()
        here = self._suite2p_dir()
        if master is None or here is None:
            return
        key = str(self.cmb_image.currentData() or "meanImg")
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            dy, dx, score = register_experiment(master, here, key, self.main.doc)
        except Exception as exc:
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(self, "Estimate shift", str(exc))
            return
        QApplication.restoreOverrideCursor()
        self._filling = True
        self.spin_dy.setValue(dy)
        self.spin_dx.setValue(dx)
        self._filling = False
        self._estimated_score = score
        self.lbl_score.setText("—" if score is None else f"{score:.3f}")
        self._sync_preview()
        if score is not None and score < SAME_FOV_SCORE:
            QMessageBox.warning(
                self,
                "Estimate shift",
                f"Score {score:.3f} is low — this experiment may not share the master's FOV.",
            )

    def _on_cancel_job(self) -> None:
        self._cancel = True

    def _on_apply(self) -> None:
        main = self.main
        master = self._master()
        path = self._master_path()
        here = self._suite2p_dir()
        if master is None or path is None or here is None or main.doc is None:
            return
        if main._mask_edit_active:
            QMessageBox.information(self, "Apply", "Finish or cancel the mask edit first.")
            return
        dy, dx = int(self.spin_dy.value()), int(self.spin_dx.value())
        key = str(self.cmb_image.currentData() or "meanImg")
        score = self._estimated_score
        entry = find_experiment(master, path, here)
        n_analyses = len(main.doc.get("analyses") or [])
        lines = [
            f"Apply '{master['meta']['name']}' (rev {master['meta']['rev']}, "
            f"{len(master['rois'])} ROIs, {len(master.get('bg_rois') or [])} BG ROIs) "
            f"at shift ({dy:+d}, {dx:+d})?",
            "",
            "• The working ROI set is replaced; traces are extracted from this experiment's data.bin.",
            "• The current pickle is kept as trc_curation.original.pkl (first time only — "
            "shown under Originals) and trc_curation.bak.pkl.",
            "• Annotations and heatmaps are kept."
            + (f" {n_analyses} saved analyses are removed (they stay in the backup)." if n_analyses else ""),
        ]
        if entry is None or not entry.get("approved"):
            lines.append("• This experiment is added to the master's approved list.")
        if score is None:
            lines.append("\nThe shift has not been estimated (or was edited by hand).")
        elif score < SAME_FOV_SCORE:
            lines.append(f"\nWarning: registration score {score:.3f} is low — check the preview.")
        if QMessageBox.question(self, "Apply shared set", "\n".join(lines)) != QMessageBox.StandardButton.Yes:
            return
        if not self._confirm_main_saved("applying the shared set"):
            return

        work = deepcopy(main.doc)
        self._cancel = False
        self.progress.setValue(0)
        self.progress.setVisible(True)
        self.btn_cancel_job.setEnabled(True)
        self.btn_apply.setEnabled(False)

        def _progress(step: int, total: int) -> None:
            self.progress.setValue(int(100 * step / max(total, 1)))
            QApplication.processEvents()

        try:
            report = apply_master(
                work,
                here,
                master,
                path,
                dy=dy,
                dx=dx,
                score=score,
                image_key=key,
                progress=_progress,
                should_cancel=lambda: self._cancel,
            )
            backup_experiment_pickle(here)
            save_curation(work, here)
        except ExtractCancelled:
            main.statusBar().showMessage("Apply cancelled — nothing changed")
            return
        except Exception as exc:
            QMessageBox.critical(self, "Apply failed", str(exc))
            return
        finally:
            self.progress.setVisible(False)
            self.btn_cancel_job.setEnabled(False)
            self.btn_apply.setEnabled(True)

        rev = int(master["meta"]["rev"])
        self._write_registry(
            lambda m: upsert_experiment(
                m, path, here, approved=True, shift=[dy, dx], score=score, image_key=key, synced_rev=rev
            )
        )
        master_now, path_now = self._master(), self._master_path()
        main.dirty = False
        main.reload_session()
        if main._master is None:
            main.set_master(master_now, path_now)
        self.refresh_from_main()
        notes = []
        if report["clipped"]:
            notes.append(f"{len(report['clipped'])} ROI(s) clipped at the FOV edge: {report['clipped'][:20]}")
        if report["dropped"]:
            notes.append(f"{len(report['dropped'])} ROI(s) fell outside the FOV and were skipped: {report['dropped'][:20]}")
        if report["bg_dropped"]:
            notes.append(f"{len(report['bg_dropped'])} BG ROI(s) fell outside the FOV")
        msg = f"Applied {len(work['rois'])} ROIs at shift ({dy:+d}, {dx:+d})."
        if notes:
            QMessageBox.information(self, "Apply shared set", msg + "\n\n" + "\n".join(notes))
        main.statusBar().showMessage(msg)
