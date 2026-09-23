# pyHole

pyHole is a lightweight, fully Python, cross-platform toolkit for rapid pore-profile analysis of macromolecular channels, with tight [ChimeraX](https://www.cgl.ucsf.edu/chimerax/) integration for interactive visualization.

Given a channel's entrance and exit (as residue selections) and a structure file, pyHole walks the pore, reporting at each point along the way:

- **Pore radius** — geometric clearance to the nearest atom surface
- **True cross-sectional area and volume** — rasterized directly from atom positions, not approximated from a single radius (see [Cross-sectional area vs. the circular approximation](#cross-sectional-area-vs-the-circular-approximation))
- **Hydrophobicity and electrostatics indices** — residue-composition-derived, per slice
- **A conductance proxy** — geometry-only, explicitly *not* a substitute for real electrophysiology (see caveat below)
- **Passability** — whether specific ion/water species (by their VdW radius) can physically pass through

pyHole is intentionally scoped as an **interactive, linear/near-straight pore profiler** for use inside ChimeraX — not a general-purpose channel-discovery tool. If you need tunnel/cavity discovery through branched or arbitrarily-shaped voids in a structure with no prior knowledge of the path, tools like [CAVER](https://www.caver.cz/) or [MOLE](https://mole.upol.cz/) (both of which also offer web interfaces) are a better fit. pyHole's niche is fast, local, desktop-based profiling once you already know roughly where your channel is.

## Table of contents

- [Installation](#installation)
- [Quick start](#quick-start)
- [Input structures: PDB and mmCIF](#input-structures-pdb-and-mmcif)
- [Selecting top and bottom](#selecting-top-and-bottom)
- [Full CLI reference](#full-cli-reference)
- [Output files](#output-files)
- [Cross-sectional area vs. the circular approximation](#cross-sectional-area-vs-the-circular-approximation)
- [True-surface rendering (density map)](#true-surface-rendering-density-map)
- [The conductance proxy — read this before citing a number](#the-conductance-proxy--read-this-before-citing-a-number)
- [Plotting profiles](#plotting-profiles)
- [ChimeraX plugin](#chimerax-plugin)
- [Known limitations](#known-limitations)
- [Citing pyHole](#citing-pyhole)
- [License](#license)

## Installation

Requires Python 3.9+.

```bash
git clone https://github.com/mlbaker-uth/pyHole.git
cd pyHole
pip install -r requirements.txt
```

That's it — `pyHole.py` and `pyhole_plotter.py` are standalone scripts, no package installation step needed. Verify everything imported correctly:

```bash
python3 pyHole.py --help
```

For the ChimeraX plugin, see [ChimeraX plugin](#chimerax-plugin) below — it has its own install step inside ChimeraX itself.

## Quick start

```bash
python3 pyHole.py my_structure.pdb \
    --top "A:120" --bottom "A:20" \
    --outprefix results/my_channel
```

This writes `results/my_channel.csv` (the full per-slice profile), `results/my_channel_summary.json` (headline numbers), `results/my_channel_centerline.pdb` and `results/my_channel_mesh.pdb` (for loading back into ChimeraX/PyMOL/etc.).

A more realistic invocation, ignoring waters/ions already resolved in the structure and using a curved centerline for a channel with gentle curvature:

```bash
python3 pyHole.py my_structure.cif \
    --top "A:120" --bottom "A:20" \
    --noHet --centerline curved \
    --probe 1.4 --occupancy electro \
    --outprefix results/my_channel
```

## Input structures: PDB and mmCIF

pyHole's parser is fixed-width legacy PDB internally, by design — but you can hand it **mmCIF, mmJSON, or anything else [gemmi](https://gemmi.readthedocs.io/) can read**, and it will transparently convert to a temporary legacy PDB before profiling. This happens automatically based on file extension; no flag needed.

There's one thing worth knowing: legacy PDB's fixed-width fields cap out at **99,999 atoms**, **4-digit residue numbers**, and **single-character chain IDs**. If your structure exceeds any of these (common for very large cryo-EM assemblies — ribosomes, viral capsids, etc.), pyHole will refuse the conversion with a clear error naming exactly which chains/residues are the problem, rather than silently truncating or corrupting numbering. In that case, subset the structure to the chain(s)/region around your pore first (most real pore-profiling targets — ion channels, GPCRs, aquaporins, transporters — are well within these limits on their own even when solved as part of a much larger assembly).

## Selecting top and bottom

`--top` and `--bottom` each take one or more residues, comma-separated; when more than one residue is given, their Cα positions are averaged to define that end's center point. Three formats are accepted, and can be mixed:

| Format | Example | Meaning |
|---|---|---|
| `<one-or-three-letter code><resnum>/<chain>` | `A55/B`, `ALA55/B` | Residue 55 in chain B (chain optional: `A55/` matches any chain) |
| `<chain>:<resnum>` or `<chain> <resnum>` | `B:55` | Residue 55 in chain B |
| `<resnum>` | `55` | Residue 55 in any chain |

Multiple residues for one end: `--top "A55/B,A56/B,A57/B"` averages all three Cα positions.

In the ChimeraX plugin, you don't need to type these by hand — select residues in the graphics window and click **Get atoms** next to the Top/Bottom field to fill it in automatically.

## Full CLI reference

```
python3 pyHole.py <pdb> --top <sel> --bottom <sel> [options]
```

**Required**

| Flag | Description |
|---|---|
| `pdb` (positional) | Legacy PDB, or mmCIF/other gemmi-readable format (auto-converted) |
| `--top` | Top-of-pore residue selection (see above) |
| `--bottom` | Bottom-of-pore residue selection |

**Structure loading**

| Flag | Default | Description |
|---|---|---|
| `--noH` | off | Exclude hydrogens |
| `--noHet` | off | Exclude HETATM records (waters, ions, ligands) — useful when a resolved ion/water sits inside the pore and would otherwise be treated as part of the wall, incorrectly zeroing the radius there |
| `--vdwjson PATH` | — | Custom per-element VdW radii (JSON, e.g. `{"NA": 1.1}`); keys are case-insensitive |

**Sampling**

| Flag | Default | Description |
|---|---|---|
| `--step` | `1.0` | Sampling interval along the pore axis (Å) |
| `--eps` | `0.25` | Contact tolerance (Å) for listing contributing residues at each slice |
| `--centerline` | `straight` | `straight`: fixed top-bottom axis. `curved`: local perturbation search around that axis for gentle bends (see limitations below for what this can't handle) |
| `--adaptive` | off | Adaptive re-sampling in `straight` mode where the radius changes quickly |
| `--slope_thresh` | `0.5` | Slope threshold (Å radius change / Å step) that triggers adaptive refinement |
| `--max_refine` | `3` | Max adaptive-refinement passes |
| `--curve_radius` | `2.0` | Search radius (Å) for `curved` mode's per-slice perturbation |
| `--curve_iters` | `3` | Refinement iterations per slice in `curved` mode |

**True area / volume** (see [Cross-sectional area vs. the circular approximation](#cross-sectional-area-vs-the-circular-approximation))

| Flag | Default | Description |
|---|---|---|
| `--no-true-area` | (area computed by default) | Skip rasterized area/volume, falling back to the circular approximation only (faster) |
| `--area-grid-step` | `0.25` | Rasterization grid spacing (Å) |
| `--area-min-radius` | `4.0` | Minimum half-extent (Å) of the per-slice search box |
| `--area-max-radius` | `20.0` | Maximum half-extent (Å) of the per-slice search box |
| `--area-radius-factor` | `3.0` | Search box half-extent = this × local clearance, clamped to `[min, max]` |

**True-surface density map**

| Flag | Default | Description |
|---|---|---|
| `--surface` | off | Also write a `.mrc` density map for isosurfacing the true pore lumen in ChimeraX |
| `--surface-voxel` | `0.5` | Density map voxel spacing (Å) |
| `--surface-margin` | `3.0` | Extra padding (Å) around the swept pore region |

**Physical/chemical properties**

| Flag | Default | Description |
|---|---|---|
| `--probe` | `0.0` | Probe radius (Å) subtracted for "accessible" volume/area/surface (e.g. `1.4` for water) |
| `--conductivity` | `1.5` | Bulk conductivity (S/m) for the geometric conductance proxy |
| `--occupancy` | `hydro` | Which metric colors the output mesh/centerline: `hydro`, `electro`, or `radii` |
| `--hydroscale` | `raw` | `raw` (Kyte-Doolittle units) or `01` (rescaled 0-1) |
| `--electroscale` | `raw` | `raw` (mean formal charge, roughly -1 to +1) or `01` |
| `--passable_json PATH` | built-in defaults | Custom species radii for passability reporting, e.g. `{"K": 1.38}`; built-in defaults cover water, Na⁺, K⁺, Ca²⁺ |

**Output**

| Flag | Default | Description |
|---|---|---|
| `--outprefix` | `holepy_out` | Path/prefix for all output files |
| `--rings` | `24` | Atoms per ring in the legacy tube mesh (`_mesh.pdb`) |
| `--interactive` | off | Prompt for `--top`/`--bottom` if not given on the command line |

## Output files

Given `--outprefix results/run1`, pyHole writes:

| File | Contents |
|---|---|
| `run1.csv` | Full per-slice profile. Columns: `s_A` (position along pore), `x,y,z`, `radius_A`, `hydro_index`, `electro_index`, `contributors` (residues within `--eps` of the tightest point), `tx,ty,tz` (local tangent), `area_A2`, `area_access_A2`, `area_half_extent_A`, `occ_value` (whichever metric `--occupancy` selected) |
| `run1_summary.json` | Headline numbers: min radius, pore length, both circular-approximation and true-rasterized volumes, conductance proxy, resistance breakdown, passability report per species, and the run's own parameters for reproducibility |
| `run1_centerline.pdb` | The pore centerline as a chain of pseudo-atoms, radius in the B-factor column, colorable metric in occupancy — load into ChimeraX/PyMOL directly |
| `run1_mesh.pdb` | A tube-of-rings mesh approximating the pore surface (legacy representation; see [True-surface rendering](#true-surface-rendering-density-map) for the more accurate alternative) |
| `run1_density.mrc` | *(only with `--surface`)* A real isosurfaceable density map of the pore lumen |

## Cross-sectional area vs. the circular approximation

By default (`--no-true-area` to disable), pyHole rasterizes the actual accessible cross-section at each slice — gridding the perpendicular plane and flood-filling from the centerline point, rather than assuming a circle of the single reported minimum-clearance radius. This matters more than it might sound: on an elliptical test pore, the circular approximation (π×r²) understated the true cross-sectional area by **56%**, since a single radius only ever sees the *short* axis of a non-circular lumen. The volume in `run1_summary.json` is reported both ways (`volume_geometric_A3`/`volume_accessible_A3` from the circular approximation, `volume_true_geometric_A3`/`volume_true_accessible_A3` from the rasterized area) so you can see the difference on your own structures.

The flood fill is anchored to the centerline point specifically so that an unrelated cavity or pocket that happens to intersect the same cutting plane elsewhere in the structure isn't counted as part of the pore.

## True-surface rendering (density map)

`--surface` writes a real 3D density map (`.mrc`, non-crystallographic MRC2014 format) instead of the legacy tube-of-spheres mesh. The density value at each voxel is clearance-to-nearest-atom minus your `--probe` radius, so **level 0 is exactly the probe-accessible pore boundary** — no threshold guessing required. Open it in ChimeraX:

```
open run1_density.mrc
volume #<N> level 0
```

The ChimeraX plugin does this automatically when its "Render true pore surface" option is checked.

## The conductance proxy — read this before citing a number

`G_nS` in the summary is a **purely geometric** proxy — bulk conductivity divided across the pore's resistance profile, computed from radius alone. It does **not** account for ion occupancy, selectivity-filter state, energetic barriers, or electrostatics, and can be badly misleading for channels where geometry and conductance are decoupled (a wide-open voltage-gated channel that happens to be inactivated conducts nothing, geometry notwithstanding). Treat it as a geometric index for comparing profiles to each other, not as a predicted physiological conductance.

## Plotting profiles

`pyhole_plotter.py` turns one or more pyHole runs into publication-style figures:

```bash
python3 pyhole_plotter.py results/run1 --out figures/run1_plot \
    --secondary area --species water --pdf
```

Key options: `--overlay` (multiple runs on one axes) or `--grid 2x2` (small-multiples panel), `--secondary {hydro,electro,occ,area}` for a second y-axis curve, `--species` to shade the passability region for a given ion/water, `--style_paper` for compact journal-style formatting, `--swap_axes` to plot radius on x and position on y (vertical profile, useful for membrane-spanning channels). Run `python3 pyhole_plotter.py --help` for the complete list.

## ChimeraX plugin

The plugin lives in `pyhole_chimerax_fix4/` and gives you the same functionality as an interactive ChimeraX tool: pick top/bottom residues by selecting them in the graphics window, run pyHole, and have the result automatically opened and colored by your chosen metric.

### Install (or update) via the ChimeraX toolshed system

All of the following commands are typed into **ChimeraX's own command line**, not your system shell.

If you're updating an existing install, remove it first:

```
toolshed uninstall pyHole
```

(`pyHole` is the bundle's short name — `ChimeraX-pyHole` from `bundle_info.xml` with the `ChimeraX-` prefix dropped.) If that throws an error instead of uninstalling (a known issue on some ChimeraX versions), fall back to:

```bash
ChimeraX -m pip uninstall ChimeraX-pyHole
```

Then build and install from the cloned repo:

```
devel clean /path/to/pyHole/pyhole_chimerax_fix4
devel install /path/to/pyHole/pyhole_chimerax_fix4
```

This resolves and installs the plugin's Python dependencies (`gemmi`, `mrcfile`) automatically from the `bundle_info.xml` manifest. If that fails (e.g. no network access from within ChimeraX), install them directly into ChimeraX's own Python first:

```bash
ChimeraX -m pip install gemmi mrcfile
```

Verify:

```
toolshed list installed
```

pyHole should now appear under **Tools → Structure Analysis → pyHole**.

### Using the plugin

1. Select a model from the dropdown at the top (or select atoms in the graphics window and click **Get atoms** next to Top/Bottom to fill in a selection).
2. Set Top plane / Bottom plane.
3. Adjust options as needed — Probe, Conductivity, Step, occupancy metric (hydrophobicity/electrostatics/radii), centerline mode (straight/curved), and optionally check **Render true pore surface (isosurface)**.
4. Click **Run pyHole**. The mesh and centerline are opened and colored automatically; if surface rendering was requested, the density map is opened and isosurfaced at level 0.

## Known limitations

- **Straight/curved centerlines only.** pyHole assumes the pore is a single, roughly linear-to-gently-curved path between the given top and bottom. It cannot follow sharp (~90°) bends, lateral fenestrations, branching pathways, or routes that double back relative to the top-bottom axis — a straight line assumption breaks down for those geometries. This is a deliberate scope decision, not an oversight: pyHole is positioned as an interactive linear pore profiler, and general tunnel/branch discovery is better served by CAVER or MOLE.
- **`--curve_radius`/`--curve_iters`** perform *local* perturbation around the straight axis and only handle gentle curvature, not sharp turns.
- **Single reported minimum radius per slice** still drives the conductance proxy and legacy mesh, even though true area/volume are rasterized — see the caveats above.
- **Large structures**: `--surface` and the true-area rasterization scale with search-box/grid size; for very large assemblies, coarsen `--area-grid-step`/`--surface-voxel` or increase `--area-max-radius`'s ceiling deliberately rather than leaving defaults on an unusually large pore.

## Citing pyHole

A citation will be added here once the associated manuscript is published — check back, or open an issue if you need a citable reference in the meantime.

## License

GPLv3 — see [LICENSE](LICENSE).
