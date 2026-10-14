# s2p Trace Curation

Inspection, compensation and editing of suite2p ROIs and traces.

## Download (Windows app)

No Python needed: grab the latest
[release](https://github.com/RasHerlo/s2p_Trace_Curation/releases/latest) and
either run `s2pTraceCuration-<version>-Setup.exe` or unzip the portable zip and
double-click `s2pTraceCuration.exe`. Windows may warn about an unrecognised app
on first launch (the exe is unsigned): choose **More info → Run anyway**.

Errors in the app are shown in a dialog and logged to
`%USERPROFILE%\.s2p_trace_curation\error.log`.

## Setup (from source)

Needs NumPy, pyqtgraph, and a Qt binding (**PyQt5** in this project’s venv).

```powershell
# from the repo root — prefer a non-Anaconda Python so Cursor accepts the env
# e.g. "C:\Program Files\Python310\python.exe" -m venv venv_s2p_tc
python -m venv venv_s2p_tc
.\venv_s2p_tc\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -e .
```

`venv_s2p_tc` is gitignored. Recreate it anytime with the commands above.

**Note:** If Cursor says the environment/module is not supported, the venv was likely created from Anaconda. Recreate it with a stock CPython install (above), then select  
`venv_s2p_tc\Scripts\python.exe` as the interpreter.

## Run

```powershell
.\venv_s2p_tc\Scripts\Activate.ps1
python -m s2p_trace_curation
```

Use **File → Open suite2p folder…** and select a `suite2p` directory (the folder that contains `plane0/`).

- If `trc_curation.pkl` already exists beside `plane0/`, it is loaded.
- Otherwise it is generated from `plane0` (`stat.npy`, `F.npy`, `Fneu.npy`, `iscell.npy`, `ops.npy`).

## Building a release

```powershell
powershell -ExecutionPolicy Bypass -File packaging\build_windows.ps1
```

This builds in its own `venv_build\` (the dev venv is untouched), runs a
`--selftest` on the built exe, and writes the portable zip (plus an installer
if Inno Setup 6 is installed) to `dist\release\`. To publish, bump
`__version__` in `s2p_trace_curation/__init__.py`, commit, and push a matching
tag (`git tag v1.0.0; git push origin v1.0.0`); GitHub Actions builds and
attaches both files to the release.

Design decisions and schema notes live in [`DESIGN_LOG.md`](DESIGN_LOG.md).

## v1 status

- Load / save versioned `trc_curation.pkl`
- FOV, movie, ROI zoom, traces, 4 analysis cursors + movie cursor
- Per-ROI `x` / Fneu offset, **Rebuild traces** (`trace_comp` / `tc_norm` /
  `tc_norm_sm` / `tc_norm_sm_bc` for the current ROI), and `iscell` toggle
- Trace Processing: Savitzky–Golay (`tc_norm_sm`) and bleach (`tc_norm_sm_bc`)
- Named heatmaps from `data.bin`: set frame ranges on the raster trace, map is
  AUC inside / AUC outside (Edit HeatMaps → Image dropdowns)
- Analysis Tools HAC: drag a dotted distance cut on the dendrogram to
  outline clusters on the similarity matrix; the cut and membership are
  saved with the run and can be edited later
- Raster **Add clusters** outlines same-cluster rows; FOV **Show Clusters**
  paints those ROIs in the same colours (uses the current Raster Sort run)
- Draggable divider above the trace panel; click a plot title (or a C1–C4 zoom)
  to expand it to full height, click again to show all four
- Annotation Tools: kind is a preset (LED+Shutter, AirPuff, PMT-noise,
  BG-motion) or a name you type; pick a trace type; frames or seconds; click
  a saved row to edit; draw ranges, then Save
- Mask Tools: Add BG ROI (unweighted background traces, stored apart from
  cells); Annotation Tools can threshold their sum as BG-motion (raw,
  smoothed, or bleach-corrected)
