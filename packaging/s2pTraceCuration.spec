# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the Windows desktop app (onedir, no console)."""

from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

spec_dir = Path(SPECPATH)
root = spec_dir.parent

_PG_SKIP = ("pyqtgraph.opengl", "pyqtgraph.examples", "pyqtgraph.jupyter", "pyqtgraph.canvas")

hiddenimports = collect_submodules("s2p_trace_curation")
hiddenimports += collect_submodules(
    "pyqtgraph", filter=lambda name: not name.startswith(_PG_SKIP)
)

datas = collect_data_files("pyqtgraph", excludes=["**/examples/**", "**/opengl/**"])

excludes = [
    "PySide2",
    "PySide6",
    "PyQt6",
    "matplotlib",
    "tkinter",
    "IPython",
    "jupyter_rfb",
    "pytest",
    "OpenGL",
    "PyQt5.QtWebEngine",
    "PyQt5.QtWebEngineCore",
    "PyQt5.QtWebEngineWidgets",
    "PyQt5.QtQml",
    "PyQt5.QtQuick",
    "PyQt5.QtMultimedia",
    "PyQt5.QtBluetooth",
    "PyQt5.QtLocation",
    "PyQt5.QtPositioning",
    "PyQt5.Qt3DCore",
]

a = Analysis(
    [str(spec_dir / "launch.py")],
    pathex=[str(root)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="s2pTraceCuration",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    icon=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="s2pTraceCuration",
)
