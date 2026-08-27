# s2p Trace Curation

Inspection, compensation and editing of suite2p ROIs and traces.

## Setup

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

Design decisions and schema notes live in [`DESIGN_LOG.md`](DESIGN_LOG.md).

## v1 status

- Load / save versioned `trc_curation.pkl`
- FOV, movie, ROI zoom, traces, 4 analysis cursors + movie cursor
- Per-ROI `x` compensation and `iscell` toggle
- Trace Processing: Savitzky–Golay (`tc_norm_sm`) and bleach (`tc_norm_sm_bc`)
- Named heatmaps from `data.bin`: set frame ranges on the raster trace, map is
  AUC inside / AUC outside (Edit HeatMaps → Image dropdowns)
- Analysis Tools HAC can use `tc_norm` / `tc_norm_sm` / `tc_norm_sm_bc`
- Draggable divider above the trace panel; click a plot title (or a C1–C4 zoom)
  to expand it to full height, click again to show all four
