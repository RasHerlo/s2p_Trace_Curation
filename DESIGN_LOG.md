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
- Bleach-correction panel + lower trace
- Manual Y-scale controls
- Merge / split / draw **new** ROIs within one plane (distinct from folder merge)
- Multi-plane or cross-recording comparison

---

## Session 2026-08-23 — Merge s2p folders (design)

### Intent (draft)
File menu → **Merge s2p folders**: dialog to pick two suite2p folders + output parent + name (default `suite2p_merged`). Write a normal suite2p tree (`plane0/` + files) plus `merge_note.txt` beside `plane0/`. Typical case: merge `_anat` / `_temp` (or temporal vs Cellpose) arms under the same Chan folder onto one shared movie.

Aligned with handoff note: intercalate ROI sets on the **same** registered movie; do not re-run suite2p registration here.

### UI (draft)
- Inputs: Folder A, Folder B (each = suite2p dir containing `plane0/`)
- Output parent: if A and B share a parent, suggest that parent; editable
- Output name: default `suite2p_merged`; editable
- Result path: `{output_parent}/{output_name}/`

### Open questions (merge)
| # | Topic | Status |
|---|--------|--------|
| M1 | Merge = concatenate ROI catalogs; require **identical** `data.bin` | **Agreed** |
| M2 | Canonical movie = copy from Folder A (identical under check) | **Agreed** |
| M3 | `Ly`/`Lx`/`nframes` must match | **Agreed** |
| M4 | ROI order = dialog pick order (A then B) | **Agreed** |
| M5 | Keep overlapping ROIs | **Agreed** |
| M6 | Provenance: `merge_note.txt` only for now | **Agreed** |
| M7 | No input pickles; clean merged folder; pickle on first Open | **Agreed** |
| M8 | If output exists: warn and ask (overwrite / abort) | **Agreed** |
| M9 | Auto-Open merged folder after success | **Agreed** |
| M10 | Write zero `spks.npy` for merged ROI count | **Agreed** |
| M11 | `data.bin` check toggle: **sample** (default) vs **full**; persist in settings | **Agreed** |

### Implementation (2026-08-23)
- `s2p_trace_curation/merge_suite2p.py` — compare + merge
- `s2p_trace_curation/gui/merge_dialog.py` — File → Merge s2p folders…

---

## Session 2026-08-23 — Batch iscell selection (design)

### Intent (draft)
Button **batch-select** beside the iscell control. Enter batch mode → freehand region on **W1** → ROIs in that region become the batch. iscell checkbox applies to **all** in the batch (default on). Traces show **mean F** and **mean Fneu**. **W3** disabled/greyed until returning to single-ROI mode.

### Open questions (batch)
| # | Topic | Status |
|---|--------|--------|
| B1 | Single↔Batch **slide toggle** | **Agreed** |
| B2 | Closed **lasso** on W1 | **Agreed** |
| B3 | Include if **>50%** of ROI pixels inside | **Agreed** |
| B4 | Only ROIs shown by W1 overlay filter | **Agreed** |
| B5 | Snap all batch ROIs to iscell **checked** on lasso | **Agreed** |
| B6 | Mean F, mean Fneu, mean display `trace_comp` with **x=1** (stored x unchanged) | **Agreed** |
| B7 | New lasso replaces previous batch | **Agreed** |
| B8 | Disable Modify mask + ROI # (+ x edit) in batch | **Agreed** |
| B9 | W3 greyed / disabled in batch | **Agreed** |

### Implementation
- `s2p_trace_curation/batch_select.py`
- Main window: mode slider, lasso on W1, batch iscell, mean traces

---

## Session 2026-08-23 — Add Mask / new ROI (design)

### Intent (draft)
**Mask tools → Add Mask** (under Modify Mask). Pick a point on **W1** with a small red cross cursor → **W3** shows a square zoom centered there (size from typical pickle zoom / ROI∪neuropil scale). Paint new F and Fneu masks from scratch (same brush modes as Modify Mask). Extra zoom in/out on W3. Finish with **Save Mask** (not Apply) → append ROI + extracted traces to `trc_curation.pkl`.

### Open questions (add mask)
| # | Topic | Status |
|---|--------|--------|
| N1 | Initial W3 square = **median** of existing zooms; then adjustable | **Agreed** |
| N2 | Zoom: **wheel + +/-** (and side spin) | **Agreed** |
| N3 | `roi_id` = max+1 | **Agreed** |
| N4 | Default `iscell=True`, `x=1.0` | **Agreed** |
| N5 | Forbid empty F / Fneu on Save | **Agreed** |
| N6 | Pickle-only (no suite2p file rewrite) | **Agreed** |
| N7 | Click again moves center **before painting**; Cancel discards | **Agreed** |
| N8 | Disable Add Mask in batch mode | **Agreed** |

### Implementation
- Mask tools: **Add Mask** → W1 red cross → W3 paint → **Save Mask**
- `empty_roi_draft` / `append_roi` in `curation.py`; brush allows empty-start builds

---

## Inspiration / references
- suite2p native outputs and GUI curation patterns (`iscell`, neuropil coeff)
- Compensation: `trace_comp = F - x * Fneu`
- Portability: USB / cross-machine folder moves with co-located pickle
- Multi-cursor trace inspection + debounced frame thumbnails
- Handoff: temporal vs Cellpose / anat vs temp ROI intercalation on shared `data.bin`

---

## Implementation status
- **v1 shell in place** (2026-08-09 + later remote updates):
  - curation I/O, suite2p_io, main GUI, mask edit, user settings
- **Merge s2p folders:** design only (2026-08-23)
- Run: `python -m s2p_trace_curation`
