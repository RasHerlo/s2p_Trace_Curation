# Incoming from suite2p (read-only)

Authoritative handoff in the suite2p clone:

`C:\Users\rasmu\Projects\Repos\suite2p\lab\notes\HANDOFF_FOR_S2P_TRACE_CURATION.md`

Copy of that note follows. You load `suite2p_temp` / `suite2p_anat`
(parent of `plane0`) in `DATA/ChanA|B`. Leave `trc_curation.pkl` in that
Chan folder. You do **not** re-run registration or train Cellpose here.
Intercalating temporal vs Cellpose ROI sets is **this** repo’s job
(`DESIGN_LOG.md` currently defers multi-version comparisons).

---

# Handoff: suite2p repo → s2p_Trace_Curation

**Audience:** an agent or person working in
[s2p_Trace_Curation](https://github.com/RasHerlo/s2p_Trace_Curation)
(inspect, compensate, edit ROIs/traces; later **intercalate** temporal vs
Cellpose ROI sets).

**This file is the outbound note.** Authoritative copy:
`C:\Users\rasmu\Projects\Repos\suite2p\lab\notes\HANDOFF_FOR_S2P_TRACE_CURATION.md`

Mirror for the curation clone:
`C:\Users\rasmu\Projects\Repos\s2p_Trace_Curation\notes\INCOMING_FROM_SUITE2P.md`

Do **not** re-run motion correction or Cellpose training in the curation
repo. Do **not** extract paper traces until both ROI families look usable.
Do **not** turn on OASIS here (`spks.npy` may be zeros; that is expected).

Last updated: 2026-08-20.

---

## Split (do not collapse)

| Repo | Job |
|---|---|
| [suite2p](https://github.com/RasHerlo/suite2p) (this sender) | Register delivered stacks, detect ROIs, extract F/Fneu, write GUI-openable `suite2p_temp` / `suite2p_anat` next to the v22 TIFFs |
| [s2p_Trace_Curation](https://github.com/RasHerlo/s2p_Trace_Curation) | Load those folders; curate; later merge/intercalate **temporal** vs **Cellpose** ROI sets on the same movie |
| [derippling_PMT_noise](https://github.com/RasHerlo/derippling_PMT_noise) | Defringe (upstream of suite2p) |
| [figure_for_cAMP_Neu_paper](https://github.com/RasHerlo/figure_for_cAMP_Neu_paper) | Paper figures / catalog, not ROI merging |

Sandbox (data, not git):

`F:\bPACNewData2026\PreProcessing Optimization\Level3b copy`

Interpreter for suite2p products:
`C:\Users\rasmu\anaconda3\envs\suite2p\python.exe`

Curation GUI: File → Open suite2p folder… = the folder that **contains**
`plane0/` (not `plane0` itself). For the collected pipeline that is
`suite2p_temp` or `suite2p_anat`. See that repo’s README.

---

## What you receive (collected folder contract)

Defringe already leaves `ChanA/B_stk_defringed_v22.tif` in
`<session>/DATA/ChanA` and `DATA/ChanB`. This repo writes MC + both
detection arms **into those same Chan folders**. Do not overwrite the
v22 TIFFs. Prefer one pickle per Chan folder so intercalation stays
local:

```
<session>/DATA/ChanA/
  ChanA_stk_defringed_v22.tif   # defringe repo; leave it
  MeanImg.png, offsets.npz, ops.npy
  suite2p_temp/plane0/          # temporal ROIs + F/Fneu  (open this)
  suite2p_anat/plane0/          # anatomical ROIs + F/Fneu (open this)
  trc_curation.pkl              # yours; write it here, not inside suite2p_*
<session>/DATA/ChanB/
  same + independent_meanImg.png (fringe guide; not the processing mean)
```

Each arm is a complete suite2p folder (parent of `plane0/`):

```
suite2p_temp/   or   suite2p_anat/
  plane0/
    ops.npy
    stat.npy
    F.npy
    Fneu.npy
    iscell.npy
    spks.npy      # zeros; OASIS off; suite2p GUI still requires the file
    data.bin      # registered movie (hardlinked into suite2p_anat)
```

Required fields in `stat[i]`: `ypix`, `xpix`, `lam`; neuropil mask when
extraction ran (`neuropil_extract=True`). `ops` has `meanImg`, `Ly`, `Lx`,
`nframes`, `fs`, `um_per_px` (from Experiment.xml; lock FOV ~14.80 Hz,
0.935 µm/px).

Do not apply suite2p lab ellipticity filtering on these eval arms — you
see what detection produced.

---

## Current bakeoff to open (2026-08-20)

Temporal vs Cellpose `cyto3`, raw vs v2.1 vs v2.2, independent-B register
(older than the share-A default; still valid for looking at detection).

Layout:

```
seg_runs/<kind>_cell_<method>/ChanA|B/suite2p/
```

`<kind>` = `raw` | `v21` | `v22`  
`<method>` = `temporal` | `cyto3`

Example (v22 ChanA temporal):

`F:\bPACNewData2026\PreProcessing Optimization\Level3b copy\seg_runs\v22_cell_temporal\ChanA\suite2p`

Overview figure: `seg_runs/raw_vs_v21_vs_v22_eval/compare.png`

| | raw n ROI | v21 | v22 |
|---|---|---|---|
| temporal A | 229 | 109 | 113 |
| temporal B | 4 | 6 | 6 |
| cyto3 A | 502 | 469 | 490 |
| cyto3 B (stock soma model = wrong prior) | 28 | 14 | 22 |

**ChanA / ChanB are PMT letters, not cell types.**

| Rig | `Experiment.xml` `<Computer>` | Astro (G-Flamp, green) | Neuron (jRGECO/RCaMP, red) |
|---|---|---|---|
| Shinano | `THORLABS_30_016` | ChanB | ChanA |
| Musashi | `USER-PC` | ChanA | ChanB |

Level3b bakeoff folders are **Shinano**. 260616 in `AC_cAMP_Neu_Ca_C1_C2` is **Musashi**.

`temporal` and `cyto3` for the same kind×channel **share the same
`data.bin`** (hardlinked from `seg_runs/_bin/<kind>_cell/ChanX/`). That is
the intercalation geometry: two `stat`/`F` tables, one movie.

---

## Intercalation (your job; not built yet)

Intent: keep **both** detections. Do not pick a winner in suite2p.
Curation should load a **pair** (or a small bundle) and let a human
interleave / match / accept ROIs.

Suggested pair key (collected pipeline):

| Role | Path pattern |
|---|---|
| Functional | `<session>/DATA/ChanX/suite2p_temp` |
| Anatomical | `<session>/DATA/ChanX/suite2p_anat` |

Same FOV, same `ChanX`. Match ROIs by spatial overlap (IoU of
`ypix`/`xpix`), tag source `temporal` | `cellpose`, keep both F traces
when two masks claim the same soma. `DESIGN_LOG.md` currently defers
“multi-version comparisons”; that is this feature.

Sandbox bakeoff pairs still exist at
`seg_runs/<kind>_cell_<method>/ChanX/suite2p` (older; ChanB cyto3 there
used auto diameter). The locked AC anatomical check is
`mc_runs\260511\C1_RLV_LW_maybe\seg_cyto3_d9\<FOV>\ChanB\suite2p`.

Do **not** merge `stat.npy` in the suite2p repo. Write a curation schema
(e.g. `schema_version` ≥ 2) that can point at two `plane_relpath`s or two
suite2p dirs beside one pickle.

When share-align MC is the processing movie (default going forward), the
non-align PMT fringe **guide** (not the processing mean) lives at:

```
DATA/Chan{non-align}/independent_meanImg.png
```

Older sandbox example:

```
mc_runs/v21_cell_shareA/ChanB/independent_meanImg.png
mc_runs/v21_cell_shareA/ChanB/roi_guide_independent_vs_shareA.png
```

Use the independent mean when deciding that an ROI sits on residual
stripes. `shift_agreement.json` in the same folder; a
`SHIFT_AGREEMENT_WARNING.txt` appears only if independent B shifts
disagree with ChanA.

---

## Do not

- Do not treat G-Flamp **territories** as the anatomical target. Locked
  ChanB `cyto3` is compact **somata** (8.42 µm / 9 px at 0.935 µm/px).
  Temporal still over-segments (scale-1 flood). Keep both; curate.
- Do not extract Fig 1 traces from fringe-shaped or empty ChanB sets.
- Do not assume `fs=10` or `diameter=9` on a new FOV. Read `ops['fs']` and
  `ops['um_per_px']` (from Experiment.xml). Zoom / line-averaging retunes
  both arms.
- Do not share one `trc_curation.pkl` across unequal-length movies
  (SUPPORT 5340 vs full 5400).
- Do not overwrite `Chan*_stk_defringed_v22.tif`. Collected products
  belong in the same `DATA/ChanA|B` folders; pickle there too. Older
  sandbox bakeoffs still live under `mc_runs/` / `seg_runs/` — do not
  mix those trees with the collected contract.

---

## What suite2p will send next

1. Collected runner: `python lab/pipeline/run_collected.py --gui` →
   `DATA/Chan{A|B}/suite2p_temp` and `suite2p_anat` next to the v22 TIFFs.
2. Open those two folders; leave `trc_curation.pkl` in `DATA/ChanA|B`.
3. Custom astrocyte weights still optional (`CELLPOSE['models']['astrocyte']`);
   stock `cyto3` + physical diameter is the locked anatomical AC arm.
4. Processing movie is share-align on the neuron PMT; independent
   non-align mean is the fringe guide.

Older bakeoff pairs (`seg_runs/`, `mc_runs/.../seg_locked/`) stay for
comparison only. Do not start MC or Cellpose training from the curation
repo.
