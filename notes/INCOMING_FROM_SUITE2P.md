# Incoming from suite2p (read-only)

Authoritative handoff in the suite2p clone:

`C:\Users\rasmu\Projects\Repos\suite2p\lab\notes\HANDOFF_FOR_S2P_TRACE_CURATION.md`

Copy of that note follows. You load `suite2p/` folders (parent of `plane0`).
You do **not** re-run registration or train Cellpose here. Intercalating
temporal vs Cellpose ROI sets is **this** repo’s future job
(`DESIGN_LOG.md` currently defers multi-version comparisons).

---

# Handoff: suite2p repo → s2p_Trace_Curation

**Audience:** an agent or person working in
[s2p_Trace_Curation](https://github.com/RasHerlo/s2p_Trace_Curation)
(inspect, compensate, edit ROIs/traces; later **intercalate** temporal vs
Cellpose ROI sets).

Do **not** re-run motion correction or Cellpose training in the curation
repo. Do **not** extract paper traces until both ROI families look usable.
Do **not** turn on OASIS here (`spks.npy` may be zeros; that is expected).

Last updated: 2026-08-20.

---

## Split (do not collapse)

| Repo | Job |
|---|---|
| [suite2p](https://github.com/RasHerlo/suite2p) | Register delivered stacks, detect ROIs, extract F/Fneu, write GUI-openable `suite2p/plane0` |
| **This repo** | Load those folders; curate; later merge/intercalate **temporal** vs **Cellpose** ROI sets on the same movie |
| [derippling_PMT_noise](https://github.com/RasHerlo/derippling_PMT_noise) | Defringe (upstream of suite2p) |
| [figure_for_cAMP_Neu_paper](https://github.com/RasHerlo/figure_for_cAMP_Neu_paper) | Paper figures / catalog, not ROI merging |

Sandbox (data, not git):

`F:\bPACNewData2026\PreProcessing Optimization\Level3b copy`

Curation GUI: File → Open suite2p folder… = the folder that **contains**
`plane0/` (not `plane0` itself).

---

## What you receive (folder contract)

Each detection arm is a complete `suite2p/` directory:

```
.../suite2p/
  plane0/
    ops.npy
    stat.npy
    F.npy
    Fneu.npy
    iscell.npy
    spks.npy      # zeros; OASIS off; suite2p GUI still requires the file
    data.bin      # registered movie (keep; both GUIs need it)
  trc_curation.pkl   # yours to create; do not require it from suite2p
```

Required fields in `stat[i]`: `ypix`, `xpix`, `lam`; neuropil mask when
extraction ran. `ops` has `meanImg`, `Ly`, `Lx`, `nframes`, `fs`
(Level3b: **14.80 Hz**).

---

## Current bakeoff to open (2026-08-20)

```
seg_runs/<kind>_cell_<method>/ChanA|B/suite2p/
```

`<kind>` = `raw` | `v21` | `v22`  
`<method>` = `temporal` | `cyto3`

Example:

`F:\bPACNewData2026\PreProcessing Optimization\Level3b copy\seg_runs\v22_cell_temporal\ChanA\suite2p`

Figure: `seg_runs/raw_vs_v21_vs_v22_eval/compare.png`

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
`data.bin`**. Two `stat`/`F` tables, one movie — that is the intercalation
geometry.

---

## Intercalation (your job; not built yet)

Keep **both** detections. Load a **pair** and match by spatial overlap
(IoU of `ypix`/`xpix`). Tag source `temporal` | `cellpose`. Keep both F
traces when two masks claim the same soma.

| Role | Path pattern |
|---|---|
| Functional | `seg_runs/<kind>_cell_temporal/ChanX/suite2p` |
| Anatomical | `seg_runs/<kind>_cell_cyto3/ChanX/suite2p` (later: custom astrocyte weights on ChanB) |

Do **not** merge `stat.npy` in the suite2p repo. A curation schema that
can point at two suite2p dirs is the right place.

ChanB fringe guide (share-align processing vs independent non-align mean; Level3b is Shinano share-A):

```
mc_runs/v21_cell_shareA/ChanB/independent_meanImg.png
mc_runs/v21_cell_shareA/ChanB/roi_guide_independent_vs_shareA.png
```

---

## Do not

- Do not treat stock `cyto3` ChanB ROIs as astrocytes.
- Do not extract Fig 1 traces from fringe-shaped or empty ChanB sets.
- Do not assume `fs=10`. Level3b is 14.80 Hz.
- Do not overwrite sandbox `inputs/` or original session `DATA/`.
- Do not start MC or Cellpose training from this repo.

Ask suite2p if a new named `seg_runs/` tag is needed.
