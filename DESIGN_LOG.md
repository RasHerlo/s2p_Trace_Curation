# s2p Trace Curation — Design Log

Living document for inspiration, plans, requirements, and agreements.
Update this file as decisions solidify. Open questions stay open until explicitly closed.

---

## Session 2026-08-09 — Initial requirements & data model

### Project intent
Build a data-processing GUI to inspect, compensate, and edit single ROIs/traces from suite2p outputs. Persistence is via a reloadable/updatable pickle file that tracks processing state across sessions.

### Input (suite2p plane folder)
Example path pattern:
`.../suite2p/plane0`

Relevant contents:
| Source | Fields / role |
|--------|----------------|
| `stat.npy` | Per-ROI: `ypix`, `xpix`, `lam`; `neuropil_mask` (linear `Ly * Lx` indices, unweighted) |
| `ops.npy` | `meanImg`, `meanImgE`, `VCorr`; geometry/timing for `data.bin` (`Ly`, `Lx`, `nframes`, …) |
| `data.bin` | Movie used for re-extraction after mask edits |
| `F.npy`, `Fneu.npy` | Trusted initial traces on first load |
| `iscell.npy` | Cell / non-cell classification; loaded and editable in GUI |

Extraction sketch:
```python
stat = np.load("plane0/stat.npy", allow_pickle=True)
ypix, xpix, lam = stat[i]["ypix"], stat[i]["xpix"], stat[i]["lam"]
neuropil_ipix = stat[i]["neuropil_mask"]  # unweighted
```

### Explicitly deferred
- Bleach-correction parameter design and GUI controls
- Merge / split ROIs
- Drawing completely new ROIs
- Multi-plane / multi-version comparisons

---

## Clarification: “versioned container”

**Not** a multi-file archive (no sidecar metadata folder, zip, etc. for v1).

**Yes:** a **single** `trc_curation.pkl` file whose top-level object is a Python `dict`:

```text
{
  "schema_version": 1,
  "meta": { ... },      # session / plane / cached ops fields
  "rois": [ ... ],      # list of per-ROI row dicts (or equivalent table)
}
```

Future schema versions bump `schema_version` and migrate on load. Still one portable file that travels with the `suite2p/` folder.

---

## Frozen schema — v1

**Filename:** `trc_curation.pkl`  
**Location:** inside `suite2p/`, alongside `plane0/`  
**Portability:** pickle and suite2p folder travel together; resolve `plane0` relative to the pickle’s directory (drive letter / absolute path may change).

### Top-level dict

| Key | Type | Description |
|-----|------|-------------|
| `schema_version` | `int` | Currently `1` |
| `meta` | `dict` | See below |
| `rois` | `list[dict]` | One entry per suite2p ROI (all ROIs loaded) |

### `meta` (session / plane cache)

| Field | Type | Description |
|-------|------|-------------|
| `plane` | `str` | `"plane0"` (v1 single-plane) |
| `plane_relpath` | `str` | Relative path from pickle dir → plane folder, e.g. `"plane0"` |
| `created_utc` | `str` | ISO timestamp at creation |
| `updated_utc` | `str` | ISO timestamp at last save |
| `Ly` | `int` | Cached from `ops` |
| `Lx` | `int` | Cached from `ops` |
| `nframes` | `int` | Cached from `ops` / traces |
| `fs` | `float` or `None` | Sampling rate if available in `ops` |
| `meanImg` | `np.ndarray` | Cached display image (optional but agreed as OK if straightforward) |
| `meanImgE` | `np.ndarray` | Cached enhanced mean |
| `VCorr` | `np.ndarray` | Cached correlation image |
| `notes` | `str` | Free-form session notes (optional empty) |

Path policy: do **not** rely on absolute suite2p paths for correctness. Optional `meta.source_suite2p_abspath` may be stored as a hint only; loader always prefers `dirname(pkl) / plane_relpath`.

### Per-ROI row (`rois[i]`)

#### Identity
| Field | Type | Description |
|-------|------|-------------|
| `roi_id` | `int` | suite2p ROI index `i` |
| `iscell` | `bool` or `0/1` | From `iscell.npy` (first column if 2-col); GUI-editable drop/re-select |
| `iscell_prob` | `float` or `None` | Second column of `iscell.npy` if present |

#### ROI cluster (`roi`)
| Field | Type | Description |
|-------|------|-------------|
| `ypix` | `np.ndarray` | Pixel rows |
| `xpix` | `np.ndarray` | Pixel cols |
| `lam` | `np.ndarray` | Weights (ROI only) |
| `F` | `np.ndarray` | Shape `(nframes,)` — initial from `F.npy`; re-extract from `data.bin` after mask edits |
| `modified` | `bool` | True if spatial and/or trace diverged from suite2p originals via GUI edits |

#### Neuropil cluster (`neuropil`)
| Field | Type | Description |
|-------|------|-------------|
| `ipix` | `np.ndarray` | Linear indices in `Ly * Lx` (suite2p `neuropil_mask`); **unweighted** |
| `Fneu` | `np.ndarray` | Shape `(nframes,)` — initial from `Fneu.npy`; re-extract after mask edits |
| `modified` | `bool` | Separate from ROI `modified` |

#### Compensation cluster (`compensation`)
| Field | Type | Description |
|-------|------|-------------|
| `x` | `float` | Per-ROI neuropil coefficient; **default `1.0`** |
| `trace_comp` | `np.ndarray` | `F - x * Fneu`; recomputed whenever `F`, `Fneu`, or `x` changes |

#### Bleach cluster (`bleach`) — placeholder for later
| Field | Type | Description |
|-------|------|-------------|
| *(TBD)* | | Parameters + corrected trace; empty / omitted in v1 |

### Recompute & reset rules (agreed)
1. **Initial load:** trust `F.npy` / `Fneu.npy`; copy spatial from `stat.npy`; `x = 1.0`; compute `trace_comp`.
2. **After mask or `x` change:** recompute affected traces (`F`/`Fneu` from `data.bin` when masks change; always refresh `trace_comp`).
3. **Reset row:** restore **all** row parameters (spatial + temporal + `x` + flags + `iscell` from suite2p files) for that `roi_id` from live `plane0` sources. Option A — no embedded original snapshot required.
4. **Neuropil:** never weighted.
5. **Edits in v1:** modify masks/traces of existing ROIs; toggle `iscell` to drop/re-select. No merge/split/new-ROI yet.
6. **Modification means (GUI):** paint/erase pixels, redraw neuropil (e.g. ring), and/or edit `lam` weights for the cell ROI.

### Conceptual row shape (illustrative)

```python
{
  "schema_version": 1,
  "meta": {
      "plane": "plane0",
      "plane_relpath": "plane0",
      "Ly": 512, "Lx": 512, "nframes": 10000, "fs": 30.0,
      "meanImg": ..., "meanImgE": ..., "VCorr": ...,
      "created_utc": "...", "updated_utc": "...",
  },
  "rois": [
      {
          "roi_id": 0,
          "iscell": True,
          "iscell_prob": 0.98,
          "roi": {
              "ypix": ..., "xpix": ..., "lam": ...,
              "F": ..., "modified": False,
          },
          "neuropil": {
              "ipix": ..., "Fneu": ..., "modified": False,
          },
          "compensation": {
              "x": 1.0, "trace_comp": ...,
          },
          # "bleach": {}  # later
      },
      # ...
  ],
}
```

---

## Agreements (locked 2026-08-09)

| ID | Agreement |
|----|-----------|
| A1 | Single-file versioned dict pickle: `trc_curation.pkl` next to `plane0/` |
| A2 | Paths: pickle ↔ suite2p folder travel together; resolve plane relative to pickle |
| A3 | Reset = reload that row from original suite2p files (Option A) |
| A4 | Neuropil pixels unweighted (`ipix` only) |
| A5 | `trace_comp` and (when masks change) `F`/`Fneu` recomputed on edit |
| A6 | Per-ROI `x`, default `1.0` |
| A7 | v1 input: single `plane0` only |
| A8 | Load all ROIs; keep/edit `iscell` (+ prob if present) |
| A9 | v1 edits: masks/traces + `iscell`; future: merge/split/new ROIs |
| A10 | Initial traces from suite2p `F.npy`/`Fneu.npy` |
| A11 | Cache `Ly`/`Lx`/`nframes`/(optional images) in `meta` when straightforward |
| A12 | Schema frozen as **v1** above for implementation |

---

## Open questions (schema — closed)

All prior Q1–Q10 closed per session answers; see Agreements.

---

## Session 2026-08-09 — Main GUI design (**frozen for v1 shell**)

### GUI goals (v1)
Inspect ROIs/traces bound to `trc_curation.pkl` with linked FOV, movie, zoom, and traces.
**Modify mask** is a placeholder (no editing tools yet).
Bleach subplot allocated but empty.
Toolkit: **pyqtgraph + Qt bindings** (PyQt5 preferred on Windows/conda; PySide6 also supported via pyqtgraph.Qt).

### Layout (frozen)

```text
+--------+------------------+------------------+------------------+
| LEFT   | W1 Full FOV      | W2 Movie stack   | W3 ROI zoom      |
| PANEL  | mean/E/VCorr     | data.bin frames  | square crop      |
| FOV    | ROI fills @70%   | ROI thick outline| ROI red@70%      |
| block  | non-sel = red    | own LUT + B/C    | neuropil orange  |
| Movie  | selected = cyan  | frame cursor C0  | [Modify mask]    |
| block  | click to select  | (distinct style) |   (placeholder)  |
| ROI #  |                  |                  |                  |
| iscell |                  |                  |                  |
| x=     |                  |                  |                  |
+--------+------------------+------------------+------------------+
| Trace 1: F + Fneu                                              |
| Trace 2: trace_comp (+ callouts: value + frame index)          |
| Trace 3: bleach (placeholder)                                  |
|   C1–C4 dotted analysis cursors + C0 movie cursor (distinct) |
+----------------------------------------------------------------+
| 4 zoom thumbnails @ C1–C4 frames (500 ms debounce)             |
+----------------------------------------------------------------+
| Menu / toolbar: Open suite2p | Open pkl | Save | Reset ROI     |
+----------------------------------------------------------------+
```

### Left control panel (frozen)

**FOV display block (W1)**
- Dropdown: image source — `meanImg` | `meanImgE` | `VCorr`
- Dropdown: LUT — `grey` (default), `turbo`, `viridis`, `magma`, `jet`
- Sliders: lower / upper display bounds (brightness–contrast)
- Overlay filter: **non-selected** (`iscell=False`) | **selected** (`iscell=True`) | **both**

**Movie display block (W2)** — boxed separately
- Own LUT dropdown (same set; default `grey`)
- Own lower / upper sliders
- Frame driven by dedicated movie cursor **C0** (not one of C1–C4)

**ROI / curation**
- ROI number box (`roi_id` / pickle row) + up/down arrows; type-in allowed
- **iscell** tick/checkbox for the current ROI (toggle cell vs non-cell; persists in pickle on save)
- Compensation **x** control → updates middle trace (`F - x * Fneu`); autoscale Y for now

### Window 1 — Full FOV (static)
- Backdrop from FOV block settings
- Visible ROIs per overlay filter; fills at ~70% opacity
- Non-selected (in the sense of not the active ROI): **red**
- Active ROI: **cyan** (different color entirely)
- Click → select; if overlap, **smallest ROI wins** (and is drawn on top)
- Only ROIs passing the iscell overlay filter are clickable/visible

### Window 2 — Full movie stack
- `data.bin` frame at **C0**
- Independent LUT + B/C (movie block)
- Active ROI: **thick red outline** only
- C0 has **distinct color/appearance** from C1–C4 (exact style TBD in last questions; e.g. solid white/yellow vs dotted analysis cursors)

### Window 3 — Square ROI zoom
- Square crop **centered** on ROI∪neuropil
- Side length = `1.5 × max(width, height)` of the combined ROI+neuropil bounding box  
  (widest dimension dominates; 50% padding beyond that span)
- Clamp crop to FOV edges when near border
- ROI fill red @ 70%; neuropil yellowish-orange @ 70%
- **"Modify mask"** button — placeholder only

### Trace panel
1. Upper: `F` + `Fneu`
2. Middle: `trace_comp`; callouts on **C1–C4** show **Y value + frame index** (following each line)
3. Lower: bleach placeholder
- Shared time axis; **autoscale Y** on ROI switch / `x` change (manual scale later)
- **C1–C4:** always visible; default at **20%, 40%, 60%, 80%** of trace length; dotted; draggable
- **C0:** movie frame cursor; distinct style; also shown on traces; drives W2 (and possibly W3 — see last Q)

### Cursor-linked zoom strip
- Four thumbnails = same crop geometry as W3, at frames **C1–C4**
- **500 ms** debounce after cursor movement before reading `data.bin`

### File actions (menu / toolbar)
- Open suite2p (create/load `trc_curation.pkl`)
- Open pkl
- Save
- Reset ROI (reload row from suite2p sources)

### GUI agreements (locked)

| ID | Agreement |
|----|-----------|
| G-A1 | Toolkit: pyqtgraph + PyQt5/PySide6 (via pyqtgraph.Qt) |
| G-A2 | LUT set: grey, turbo, viridis, magma, jet; grey default |
| G-A3 | W1/W2 have separate boxed LUT + lower/upper controls |
| G-A4 | Overlay filter: iscell false / true / both |
| G-A5 | Left-panel iscell checkbox for active ROI |
| G-A6 | Active ROI on W1 = cyan; others red @ 70% |
| G-A7 | Overlap: smallest ROI on top / wins click |
| G-A8 | W2 frame locked to distinct cursor C0 |
| G-A9 | Analysis cursors C1–C4 always on; defaults 20/40/60/80% |
| G-A10 | Middle-plot callouts: value + frame index |
| G-A11 | Zoom side = 1.5 × max bbox side of ROI∪neuropil; square; clamp to FOV |
| G-A12 | Thumbnail debounce 500 ms |
| G-A13 | File actions in menu/toolbar |
| G-A14 | Trace autoscale for now |
| G-A15 | Modify mask + bleach subplot = placeholders |

### Last questions — closed (2026-08-09)
| # | Decision |
|---|----------|
| L1 | W3 shows **movie frame at C0**, zoomed (same frame as W2) |
| L2 | C0 = solid bright yellow/white; C1–C4 = dotted muted colors (tune in UI) |
| L3 | Default FOV image = **meanImg** |
| L4 | Open = choose suite2p folder; load existing `trc_curation.pkl` if present, else generate from `plane0` |

### Future GUI features (parked)
- **Modify mask** tools (paint/erase, neuropil ring, lam weights)
- Bleach-correction panel + lower trace
- Manual Y-scale controls
- Merge / split / draw new ROIs
- Multi-plane or cross-recording comparison

---

## Inspiration / references
- suite2p native outputs and GUI curation patterns (`iscell`, neuropil coeff)
- Compensation: `trace_comp = F - x * Fneu`
- Portability: USB / cross-machine folder moves with co-located pickle
- Multi-cursor trace inspection + debounced frame thumbnails

---

## Implementation status
- **v1 shell in place** (2026-08-09):
  - `s2p_trace_curation/curation.py` — create/load/save `trc_curation.pkl`, reset ROI, set `x`
  - `s2p_trace_curation/suite2p_io.py` — ops/stat/traces/`data.bin` + zoom geometry
  - `s2p_trace_curation/gui/main_window.py` — FOV / movie / zoom / traces / cursors / thumbnails
- Placeholders: Modify mask, bleach subplot
- Run: `python -m s2p_trace_curation`
