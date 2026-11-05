# Shared ROI sets (master pickles) — phase 1, 2026-11-01

Reuse one curated ROI set across experiments recorded in the same FOV
(e.g. `D:\CollectedData\F1_RV_pdf\New recordings PFC`: 240916_pl100 LED 2s
ex02–04 and LED+AP ex01–03). Full design and agreements (RS-A1…A12):
`DESIGN_LOG.md`, session "2026-11-01 — Shared ROI sets across experiments".

## What changed

- New `s2p_trace_curation/roisets.py`: master file I/O and experiment
  registry, meanImg registration (phase correlation), mask shifting,
  publish / apply / re-derive, originals ("ghosts") and import.
- New `s2p_trace_curation/gui/roiset_dialog.py`: the **ROI Set** window.
- `gui/main_window.py`: FOV (W1) box gets the ROI-set status line, the
  **ROI Set…** button and the **Originals** row (show + Import). On linked
  experiments the **Edits** window resets components to the master version.
- `gui/overlays.py`: outline overlays for ghosts and the master preview.
- `suite2p_io.py`: `extract_traces_batch` (all ROIs in one pass over `data.bin`).
- `curation.py`: local ROI ids start at 10000 on linked experiments.
- Pickle schema 8 (`master_link` on the doc, `link` on each ROI). Older
  pickles still load; older app versions cannot open schema-8 files.
- Masters keep suite2p's per-pixel `overlap` flags for unedited ROIs, so
  re-extracted traces match `F.npy` (suite2p ran with `allow_overlap=False`).

## Files on disk

- `<name>.roiset.pkl` — the master, one per channel, in the folder you
  choose on Publish. Default folder is the dataset folder; default name is
  `<parent>_<folder>_<channel>`, e.g. `F1_RV_pdf_New_Recordings_PFC_ChanA`
  (editable; it follows folder/channel changes until you edit it).
  A `.bak` of the previous master is kept on every save.
- Per experiment, next to `trc_curation.pkl`:
  `trc_curation.original.pkl` (written once, before the first publish/apply;
  source of the Originals layer) and `trc_curation.bak.pkl` (every time).

## How to use

1. **Publish** — open a curated experiment, **ROI Set…** → *Publish this
   experiment…*. Check name, channel, tag (e.g. `LED 2s`) and folder.
   iscell=True ROIs + BG ROIs go into the master; the experiment is reduced
   to those ROIs and becomes the first approved member.
2. **Register experiments** — in the ROI Set window, *Scan folder…* (or
   *Add folder…*). Each found suite2p folder is registered against the
   master's meanImg and listed with shift and score. Same FOV scores
   ~0.8–0.96; other FOVs < 0.05. Tick **Approved** for the ones that belong;
   edit **Tag** freely. Both are saved into the master immediately.
3. **Apply to another experiment** — double-click its row (or *Open*), then
   in *This experiment*: *Estimate shift* → check *Preview on W1* (magenta
   outlines) → adjust dY/dX if needed → *Apply shared set*. Traces are
   extracted from this experiment's own movie; SG/bleach mode and per-ROI
   choices come from the master as defaults, fitted values stay local.
   Saved analyses are cleared; ROI ids equal master uids.
4. **Keep optimising locally** — masks, compensation, smoothing, bleach and
   iscell edits are local to the experiment. The **Edits** button in the
   ROI / curation box lists what differs from the master for the active ROI
   and resets the components you tick back to the master version; **View →
   Show Edits** gives the same for all ROIs (see DESIGN_LOG, 2026-11-05).
5. **Compare with originals** — **Originals** dropdown in the W1 box
   (iscell=1 / all). Yellow = not covered by any working ROI (IoU < 0.2),
   grey = covered. Click one to view it read-only in W3/traces; **Import**
   adds it as a local ROI (id ≥ 10000).

## Next steps

- **Phase 2 — Compare / Sync**: table matching each experiment's ROIs to
  master uids; per-ROI *push* (local edit → master, bumps rev), *pull*
  (master → local) and *keep local*; warn when the master rev is newer than
  the experiment's `synced_rev` (status line already shows this).
- Open questions: should Scan prefer `suite2p_temp` over `suite2p_anat` when
  there is no `suite2p_merged`/pickle (currently alphabetical → anat)?
- **Phase 3 — collected analysis** (parked): cross-experiment analysis on
  aligned ROIs; needs its own design for switching between collected and
  single-experiment work, probably a separate window.
