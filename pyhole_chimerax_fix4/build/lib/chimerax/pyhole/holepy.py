"""HOLE-like pore profile calculation with straight/curved centerline options.

Standalone engine for the pyHole ChimeraX bundle. Deliberately self-contained
(no dependency on the cryomodel or root-level pyHole.py code) so the bundle
can be installed and run from the ChimeraX toolshed on its own. Accepts
legacy PDB directly, or any other gemmi-readable structure format (mmCIF,
mmJSON, ...), transparently converted to a temp legacy PDB before parsing.
"""
from pathlib import Path
import argparse
import sys
import re
import math
import json
import csv
import os
import tempfile
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy.spatial import cKDTree
from scipy.ndimage import label as _cc_label
import gemmi
import mrcfile

# --- Constants (in sync with CLI v8) ---
ONE_LETTER = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F', 'GLY': 'G',
    'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N',
    'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 'SER': 'S', 'THR': 'T', 'VAL': 'V',
    'TRP': 'W', 'TYR': 'Y', 'SEC': 'U', 'PYL': 'O',
}
VDW_DEFAULT = {
    'H': 1.20, 'C': 1.70, 'N': 1.55, 'O': 1.52, 'F': 1.47, 'P': 1.80,
    'S': 1.80, 'CL': 1.75, 'BR': 1.85, 'I': 1.98, 'NA': 2.27, 'MG': 1.73,
    'K': 2.75, 'CA': 2.31, 'ZN': 1.39, 'FE': 1.56, 'CU': 1.40, 'MN': 1.61,
}
KD_HYDRO = {
    'ILE': 4.5, 'VAL': 4.2, 'LEU': 3.8, 'PHE': 2.8, 'CYS': 2.5, 'MET': 1.9,
    'ALA': 1.8, 'GLY': -0.4, 'THR': -0.7, 'SER': -0.8, 'TRP': -0.9,
    'TYR': -1.3, 'PRO': -1.6, 'HIS': -3.2, 'GLU': -3.5, 'GLN': -3.5,
    'ASP': -3.5, 'ASN': -3.5, 'LYS': -3.9, 'ARG': -4.5,
}
KD_MIN = min(KD_HYDRO.values())
KD_MAX = max(KD_HYDRO.values())
CHARGE_MAP = {'ARG': +1.0, 'LYS': +1.0, 'HIS': +0.1, 'ASP': -1.0, 'GLU': -1.0}
ELEC_MIN, ELEC_MAX = -1.0, +1.0  # mean formal charge per slice maps into [-1, 1] before optional 01 remap
# Floor for reported mesh/centerline radius (Å); avoids zero/negative clearance breaking mesh / coloring.
MIN_PORE_RADIUS_A = 0.1


# --- mmCIF / any-gemmi-format -> legacy PDB (self-contained; see
#     cryomodel/pore/structure_io.py for the full rationale) ---
_MAX_ATOMS = 99_999
_MAX_RESI = 9_999
_MIN_RESI = -999
_LEGACY_PDB_SUFFIXES = {'.pdb', '.ent', '.pdb1'}


class StructureTooLargeForLegacyPDB(ValueError):
    """Raised when a structure can't be safely represented as legacy PDB."""


def ensure_legacy_pdb(path, tmp_dir: Optional[str] = None) -> Path:
    """Return a path to a legacy fixed-width PDB file equivalent to ``path``.

    holepy.py's parser assumes fixed-width legacy-PDB columns and does not
    understand hybrid-36 encoding, so structures that would overflow legacy
    PDB's fields (>99,999 atoms, >4-digit residue numbers, multi-character
    chain IDs) raise StructureTooLargeForLegacyPDB instead of silently
    corrupting atom/residue numbering. Subset the structure and retry.
    """
    path = Path(path)
    if path.suffix.lower() in _LEGACY_PDB_SUFFIXES:
        return path

    st = gemmi.read_structure(str(path))
    if len(st) == 0:
        raise ValueError(f"No models found in '{path.name}'.")

    single = gemmi.Structure()
    single.name = st.name
    single.cell = st.cell
    single.spacegroup_hm = st.spacegroup_hm
    single.add_model(st[0])
    single.setup_entities()

    n_atoms = 0
    bad_chains = set()
    bad_residues = []
    for chain in single[0]:
        if len(chain.name) > 1:
            bad_chains.add(chain.name)
        for res in chain:
            if res.seqid.num > _MAX_RESI or res.seqid.num < _MIN_RESI:
                bad_residues.append((chain.name, res.seqid.num))
            n_atoms += len(res)

    problems = []
    if n_atoms > _MAX_ATOMS:
        problems.append(f"{n_atoms} atoms (legacy PDB's serial field caps at {_MAX_ATOMS})")
    if bad_chains:
        problems.append(f"chain ID(s) longer than 1 character: {sorted(bad_chains)}")
    if bad_residues:
        sample = ', '.join(f"{c}:{n}" for c, n in bad_residues[:5])
        more = f" (+{len(bad_residues) - 5} more)" if len(bad_residues) > 5 else ""
        problems.append(f"{len(bad_residues)} residue number(s) outside PDB's 4-digit field, e.g. {sample}{more}")

    if problems:
        raise StructureTooLargeForLegacyPDB(
            f"'{path.name}' cannot be safely converted to legacy PDB for pyHole: " + "; ".join(problems) + ". "
            "Subset the structure (e.g. to the chain(s)/region around your pore) and retry."
        )

    fd, tmp_path = tempfile.mkstemp(suffix='.pdb', prefix='pyhole_cif2pdb_', dir=tmp_dir)
    os.close(fd)
    single.write_pdb(tmp_path)
    return Path(tmp_path)


def guess_element(atom_name, element_field):
    if element_field and element_field.strip():
        return element_field.strip().upper()
    an = atom_name.strip()
    if not an:
        return 'C'
    if an[0].isdigit() and len(an) >= 2:
        c = an[1]
        if len(an) >= 3 and an[2].isalpha():
            return (c + an[2]).upper()
        return c.upper()
    c = an[0]
    if len(an) >= 2 and an[1].isalpha() and an[1].islower():
        return (c + an[1]).upper()
    return c.upper()


def vdw_radius(element, custom):
    """Get van der Waals radius for element.

    ``custom`` must already have upper-cased keys -- callers should normalize
    a loaded JSON file with normalize_vdw_keys() before passing it in here,
    since this does a case-sensitive lookup after upper-casing ``element``.
    """
    e = element.upper()
    return custom.get(e, VDW_DEFAULT.get(e, 1.7))


def normalize_vdw_keys(raw: Dict[str, float]) -> Dict[str, float]:
    """Upper-case all keys in a custom VDW radii dict.

    Without this, a JSON file with lowercase element keys (e.g. {"na": 1.1})
    silently fails every lookup in vdw_radius() and falls back to the
    generic default -- the custom radii are ignored with no warning.
    """
    return {str(k).upper(): v for k, v in (raw or {}).items()}


class Atom:
    __slots__ = ('x', 'y', 'z', 'name', 'resname', 'chain', 'resi', 'icode', 'element', 'occ')

    def __init__(self, x, y, z, name, resname, chain, resi, icode, element, occ):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        self.name = name.strip()
        self.resname = resname
        self.chain = chain
        self.resi = int(resi)
        self.icode = icode  # PDB insertion code (single char, '' if none)
        self.element = element
        self.occ = float(occ)


def load_pdb_atoms(path, include_h=True, include_hetatm=True):
    atoms = []
    with open(path, 'r') as f:
        for line in f:
            if line[:6] not in ('ATOM  ', 'HETATM'):
                continue
            if (not include_hetatm) and line[:6] == 'HETATM':
                continue
            name = line[12:16]
            resname = line[17:20].strip()
            chain = line[21].strip() or 'A'
            resi_str = line[22:26].strip() or '0'
            # Column 27 (index 26) is the PDB insertion code. Residues that
            # differ only by insertion code (e.g. 55 and 55A) are distinct
            # residues -- dropping this silently merged them.
            icode = line[26].strip() if len(line) > 26 else ''
            try:
                resi = int(resi_str)
            except ValueError:
                continue
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            occ = float(line[54:60]) if line[54:60].strip() else 1.0
            elem_field = line[76:78] if len(line) >= 78 else ''
            elem = guess_element(name, elem_field)
            if (not include_h) and elem.upper() == 'H':
                continue
            atoms.append(Atom(x, y, z, name, resname, chain, resi, icode, elem, occ))
    return atoms


def parse_residue_tokens(s):
    tokens = []
    if not s:
        return tokens
    for t in s.split(','):
        t = t.strip()
        if not t:
            continue
        # ONELETTER...NUM/CHAIN or THREELETNUM/CHAIN (chain optional -> match any chain_id)
        m = re.match(r'^([A-Za-z]{1,3})(\d+)\s*/\s*([A-Za-z0-9]*)$', t)
        if m:
            chain = m.group(3).strip() or '*'
            tokens.append((chain, int(m.group(2))))
            continue
        m = re.match(r'^([A-Za-z0-9])\s*[:\s]\s*(\d+)$', t)
        if m:
            tokens.append((m.group(1), int(m.group(2))))
            continue
        m = re.match(r'^\s*(\d+)\s*$', t)
        if m:
            tokens.append(('*', int(m.group(1))))
            continue
        raise ValueError(f"Could not parse residue token: '{t}'.")
    return tokens


def ca_positions_for(atoms, sel):
    out = []
    for (ch, rnum) in sel:
        for a in atoms:
            if a.name == 'CA' and a.resi == rnum and (ch == '*' or a.chain == ch):
                out.append([a.x, a.y, a.z])
    return np.array(out, float) if out else np.zeros((0, 3), float)


def orthonormal_basis_from_axis(axis):
    u = axis / (np.linalg.norm(axis) + 1e-12)
    a = np.array([1.0, 0.0, 0.0]) if abs(u[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    v = np.cross(u, a)
    n = np.linalg.norm(v)
    if n < 1e-8:
        a = np.array([0.0, 0.0, 1.0])
        v = np.cross(u, a)
        n = np.linalg.norm(v)
    v /= (n + 1e-12)
    w = np.cross(u, v)
    w /= (np.linalg.norm(w) + 1e-12)
    return v, w


class SpatialIndex:
    """KD-tree wrapper for fast local clearance queries.

    Evaluating a slice used to scan every atom in the structure (O(N) per
    query point), which dominates runtime for large assemblies with many
    query points (dense/adaptive sampling, curved-centerline refinement).
    We instead query only atoms within a radius of the candidate point and
    grow that radius until we can prove no atom outside the search ball
    could produce a smaller clearance -- returns exactly the same value as
    the brute-force scan, just faster.
    """

    def __init__(self, atom_xyz: np.ndarray, atom_r: np.ndarray, initial_radius: float = 12.0):
        self.atom_xyz = atom_xyz
        self.atom_r = atom_r
        self.tree = cKDTree(atom_xyz) if len(atom_xyz) else None
        self.initial_radius = initial_radius
        self.max_atom_r = float(np.max(atom_r)) if len(atom_r) else 0.0

    def clearance_and_mask(self, c: np.ndarray, contact_eps: float = 0.0):
        if self.tree is None or len(self.atom_xyz) == 0:
            return float('inf'), np.zeros(0, dtype=bool)
        r = self.initial_radius
        n_atoms = len(self.atom_xyz)
        while True:
            idx = self.tree.query_ball_point(c, r=r)
            if idx:
                idx = np.asarray(idx)
                dv = self.atom_xyz[idx] - c
                d = np.sqrt((dv * dv).sum(axis=1)) - self.atom_r[idx]
                d_min = float(np.min(d))
                # Any atom outside this ball is at distance > r, so its
                # clearance is > r - max_atom_r. If our current best is
                # already <= that bound, no unseen atom can beat it.
                if d_min <= (r - self.max_atom_r) or len(idx) >= n_atoms:
                    full_d = np.full(n_atoms, np.inf)
                    full_d[idx] = d
                    mask = full_d <= (d_min + contact_eps)
                    return d_min, mask
            r *= 1.5

    def batch_knn_clearance(self, points: np.ndarray, k: int = 12) -> np.ndarray:
        """Vectorized clearance (dist - vdw_r) to the nearest atom for many
        points at once, via a k-nearest-neighbor search rather than the exact
        expanding-radius search used by clearance_and_mask().

        This is an approximation: it's only exact if the atom that actually
        minimizes clearance is among the k nearest by *raw* distance, which
        holds in practice because VDW radii vary within a bounded range
        (~1-2.8 A) -- an atom much farther away by raw distance essentially
        never wins on clearance. Used for area/volume rasterization and the
        density-map surface, where we evaluate thousands-to-millions of grid
        points and the expanding-radius search's per-point Python loop would
        be too slow; not used for the single-point clearance that determines
        the reported minimum pore radius (clearance_and_mask stays exact there).
        """
        if self.tree is None or len(self.atom_xyz) == 0:
            return np.full(len(points), np.inf)
        k_eff = min(k, len(self.atom_r))
        dists, idxs = self.tree.query(points, k=k_eff)
        if k_eff == 1:
            dists = dists[:, None]
            idxs = idxs[:, None]
        clearances = dists - self.atom_r[idxs]
        return clearances.min(axis=1)


def _clearance_at_point(atom_xyz, atom_r, c, index: Optional[SpatialIndex] = None):
    if index is not None:
        d_min, _ = index.clearance_and_mask(c)
        return d_min
    dv = atom_xyz - c
    d = np.sqrt((dv * dv).sum(axis=1)) - atom_r
    return float(np.min(d))


def _evaluate_slice(atom_xyz, atom_r, atom_meta, c, contact_eps, hydro_scale, electro_scale,
                     index: Optional[SpatialIndex] = None):
    if index is not None:
        r_raw, mask = index.clearance_and_mask(c, contact_eps)
    else:
        dv = atom_xyz - c
        d = np.sqrt((dv * dv).sum(axis=1)) - atom_r
        r_raw = float(np.min(d))
        mask = d <= (r_raw + contact_eps)

    seen = set()
    tags = []
    hyd = []
    chg = []
    for i, ok in enumerate(mask):
        if not ok:
            continue
        chain, resname, resi, icode = atom_meta[i]
        # Insertion code is part of residue identity: 55 and 55A must not
        # be collapsed into a single contributor tag.
        key = (chain, resi, icode, resname)
        if key in seen:
            continue
        seen.add(key)
        resi_label = f"{resi}{icode}" if icode else str(resi)
        tags.append(f"{ONE_LETTER.get(resname.upper(), '?')}{resi_label}/{chain}")
        hyd.append(KD_HYDRO.get(resname.upper(), 0.0))
        chg.append(CHARGE_MAP.get(resname.upper(), 0.0))

    hydro = float(np.mean(hyd)) if hyd else 0.0
    electro = float(np.mean(chg)) if chg else 0.0
    if hydro_scale == '01':
        hydro = (hydro - KD_MIN) / (KD_MAX - KD_MIN) if KD_MAX > KD_MIN else 0.0
    if electro_scale == '01':
        electro = (electro - ELEC_MIN) / (ELEC_MAX - ELEC_MIN) if ELEC_MAX > ELEC_MIN else 0.5

    rmin = max(r_raw, MIN_PORE_RADIUS_A)
    return rmin, ';'.join(tags[:50]), hydro, electro


def _trapz(y, x) -> float:
    """Trapezoidal integral of y(x). Hand-rolled instead of np.trapz/np.trapezoid
    since that function's name changed between numpy versions."""
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    if len(y) < 2:
        return 0.0
    return float(np.sum((y[1:] + y[:-1]) * 0.5 * (x[1:] - x[:-1])))


def slice_area(index: 'SpatialIndex', center: np.ndarray, v: np.ndarray, w: np.ndarray,
                probe: float = 0.0, seed_clearance: Optional[float] = None,
                min_half_extent: float = 4.0, max_half_extent: float = 20.0,
                radius_factor: float = 3.0, grid_step: float = 0.25, k: int = 12):
    """Rasterized true cross-sectional area of the pore in the plane through
    ``center`` spanned by orthonormal (v, w), instead of treating the slice
    as a circle of the single reported min-clearance radius.

    A single "radius" number assumes a circular cross-section; real pores are
    often lobed, eccentric, or locally widen off-axis, which a circular
    assumption hides entirely and which HOLE-style single-radius profiles
    can't represent. This grids the plane, marks each cell as accessible
    (clearance > probe) or not, and keeps only the connected component
    reachable from ``center`` by flood fill -- so an unrelated cavity or
    pocket that happens to intersect the same cutting plane elsewhere in the
    structure isn't counted as part of the pore.

    Returns (area_geometric_A2, area_accessible_A2, half_extent_used_A).
    """
    if seed_clearance is None:
        seed_clearance = float(index.batch_knn_clearance(center[None, :], k=k)[0])
    half_extent = float(np.clip(radius_factor * max(seed_clearance, 0.5), min_half_extent, max_half_extent))

    n = max(3, int(round(2 * half_extent / grid_step)) + 1)
    axis_vals = np.linspace(-half_extent, half_extent, n)
    gv, gw = np.meshgrid(axis_vals, axis_vals, indexing='ij')
    pts = center[None, None, :] + gv[..., None] * v[None, None, :] + gw[..., None] * w[None, None, :]
    clearance = index.batch_knn_clearance(pts.reshape(-1, 3), k=k).reshape(n, n)

    cell_area = grid_step * grid_step
    center_idx = (n // 2, n // 2)  # axis_vals is symmetric about 0 -> this cell is (0, 0)

    def flood_area(mask):
        if not mask[center_idx]:
            # Center point comes from the already-validated centerline, so
            # this shouldn't normally trigger; fail safe to 0 rather than
            # grabbing an unrelated component if it ever does.
            return 0.0
        labeled, _ = _cc_label(mask, structure=np.ones((3, 3)))
        comp_id = labeled[center_idx]
        return float(np.sum(labeled == comp_id)) * cell_area if comp_id else 0.0

    area_geom = flood_area(clearance > 0.0)
    area_access = flood_area(clearance > probe) if probe > 0 else area_geom
    return area_geom, area_access, half_extent


def build_density_grid(index: 'SpatialIndex', rows: List[Dict],
                        probe: float = 0.0, voxel_size: float = 0.5, margin: float = 3.0,
                        min_half_extent: float = 4.0, max_half_extent: float = 20.0,
                        radius_factor: float = 3.0, max_voxels: int = 40_000_000, k: int = 12):
    """Build a 3D scalar field (clearance-to-nearest-atom minus probe radius)
    over a padded box around the sampled centerline, for isosurfacing the
    true pore lumen (level=0 is exactly the probe-accessible boundary)
    instead of the tube-of-spheres mesh approximation.

    Returns (values[nx,ny,nz] as (z,y,x)-ordered for MRC, origin_xyz, voxel_size).
    """
    centers = np.array([[r['x'], r['y'], r['z']] for r in rows], dtype=float)
    # Per-slice half-extent estimate (same adaptive sizing as slice_area) sets
    # how far sideways from the centerline we need to cover.
    seed_clear = index.batch_knn_clearance(centers, k=k)
    half_extents = np.clip(radius_factor * np.maximum(seed_clear, 0.5), min_half_extent, max_half_extent)
    pad = float(np.max(half_extents)) + margin

    lo = centers.min(axis=0) - pad
    hi = centers.max(axis=0) + pad

    nx = max(2, int(np.ceil((hi[0] - lo[0]) / voxel_size)) + 1)
    ny = max(2, int(np.ceil((hi[1] - lo[1]) / voxel_size)) + 1)
    nz = max(2, int(np.ceil((hi[2] - lo[2]) / voxel_size)) + 1)
    n_total = nx * ny * nz
    if n_total > max_voxels:
        raise ValueError(
            f"Requested density grid is {nx}x{ny}x{nz} = {n_total:,} voxels, over the "
            f"{max_voxels:,} safety limit. Increase --surface-voxel (coarser spacing) or "
            f"reduce --surface-margin / --area-max-radius."
        )

    xs = lo[0] + voxel_size * np.arange(nx)
    ys = lo[1] + voxel_size * np.arange(ny)
    zs = lo[2] + voxel_size * np.arange(nz)
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing='ij')
    pts = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)

    clearance = index.batch_knn_clearance(pts, k=k)
    values = (clearance - probe).reshape(nx, ny, nz).astype(np.float32)
    values_zyx = np.transpose(values, (2, 1, 0))  # mrcfile wants (z, y, x)
    return values_zyx, lo, voxel_size


def write_density_map(path, origin: np.ndarray, voxel_size: float, values_zyx: np.ndarray):
    """Write a scalar field to a non-crystallographic MRC map (MRC2014 ORIGIN
    convention), so it opens already aligned with the source structure in
    ChimeraX/ChimeraX-compatible viewers. level=0 is the probe-accessible
    pore boundary by construction (see build_density_grid)."""
    with mrcfile.new(str(path), overwrite=True) as mrc:
        mrc.set_data(np.ascontiguousarray(values_zyx, dtype=np.float32))
        mrc.voxel_size = float(voxel_size)
        mrc.header.origin.x = float(origin[0])
        mrc.header.origin.y = float(origin[1])
        mrc.header.origin.z = float(origin[2])
        mrc.update_header_stats()


def profile_along_axis(atom_xyz, atom_r, c0, c1, step, eps, atom_meta,
                        adaptive=False, slope_thresh=0.5, max_refine=3,
                        hydro_scale='raw', electro_scale='raw', occupancy_metric='hydro',
                        true_area=True, probe=0.0, area_grid_step=0.25,
                        area_min_radius=4.0, area_max_radius=20.0, area_radius_factor=3.0):
    axis = c1 - c0
    L = np.linalg.norm(axis)
    if L < 1e-6:
        raise ValueError("Top and bottom centers are too close.")
    u = axis / L
    index = SpatialIndex(atom_xyz, atom_r)
    v_global, w_global = orthonormal_basis_from_axis(u)

    def ctr(s):
        return c0 + u * s

    svals = list(np.linspace(0.0, L, max(1, int(round(L / step)) + 1)))

    def eval_rows(vals):
        out = []
        for s in vals:
            c = ctr(s)
            rmin, tags, hyd, elec = _evaluate_slice(
                atom_xyz, atom_r, atom_meta, c, eps, hydro_scale, electro_scale, index=index)
            out.append({
                's_A': float(s), 'x': float(c[0]), 'y': float(c[1]), 'z': float(c[2]),
                'radius_A': float(rmin), 'hydro_index': float(hyd), 'electro_index': float(elec),
                'contributors': tags,
            })
        return out

    rows = eval_rows(svals)
    if adaptive:
        for _ in range(max_refine):
            rows.sort(key=lambda r: r['s_A'])
            ns = []
            for i in range(len(rows) - 1):
                s0, r0 = rows[i]['s_A'], rows[i]['radius_A']
                s1, r1 = rows[i + 1]['s_A'], rows[i + 1]['radius_A']
                ds = s1 - s0
                if ds <= step / 2:
                    continue
                slope = abs((r1 - r0) / ds) if ds > 1e-9 else 0.0
                if slope > slope_thresh:
                    ns.append(0.5 * (s0 + s1))
            if not ns:
                break
            rows += eval_rows(ns)
    rows.sort(key=lambda r: r['s_A'])
    if true_area:
        for r in rows:
            c = np.array([r['x'], r['y'], r['z']], float)
            area_geom, area_access, half_extent = slice_area(
                index, c, v_global, w_global, probe=probe,
                min_half_extent=area_min_radius, max_half_extent=area_max_radius,
                radius_factor=area_radius_factor, grid_step=area_grid_step)
            r['area_A2'] = float(area_geom)
            r['area_access_A2'] = float(area_access)
            r['area_half_extent_A'] = float(half_extent)
    for r in rows:
        if occupancy_metric == 'hydro':
            r['occ_value'] = float(r['hydro_index'])
        elif occupancy_metric == 'electro':
            r['occ_value'] = float(r['electro_index'])
        else:
            r['occ_value'] = float(r['radius_A'])
    return rows, u, L


def construct_centers_curved(atom_xyz, atom_r, c0, c1, step, curve_radius, curve_iters):
    axis = c1 - c0
    L = float(np.linalg.norm(axis))
    if L < 1e-6:
        raise ValueError("Top and bottom centers are too close.")
    u = axis / L
    v, w = orthonormal_basis_from_axis(u)
    index = SpatialIndex(atom_xyz, atom_r)
    svals = list(np.linspace(0.0, L, max(1, int(round(L / step)) + 1)))
    centers = []
    c_prev = c0.copy()
    for idx, s in enumerate(svals):
        if s <= 1e-9:
            c = c0.copy()
        elif abs(s - L) <= 1e-9:
            c = c1.copy()
        else:
            ds = svals[idx] - svals[idx - 1]
            c = c_prev + u * ds
            r = float(curve_radius)
            for _ in range(int(curve_iters)):
                best_c = c
                best_cl = _clearance_at_point(atom_xyz, atom_r, c, index=index)
                for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1), (1, -1), (-1, -1), (0, 0)]:
                    cand = c + (dx * r) * v + (dy * r) * w
                    cl = _clearance_at_point(atom_xyz, atom_r, cand, index=index)
                    if cl > best_cl:
                        best_cl = cl
                        best_c = cand
                c = best_c
                r *= 0.5
        centers.append(c)
        c_prev = c
    return centers, u, L


def sample_straight_centers(c0: np.ndarray, c1: np.ndarray, step: float) -> List[np.ndarray]:
    """Evenly-spaced points along the straight line from c0 to c1 (inclusive
    of both ends). Pulled out standalone so a straight segment can be one
    leg of a multi-waypoint path (see build_waypoint_centers)."""
    axis = c1 - c0
    L = float(np.linalg.norm(axis))
    if L < 1e-6:
        raise ValueError("Segment endpoints are too close together.")
    u = axis / L
    svals = np.linspace(0.0, L, max(1, int(round(L / step)) + 1))
    return [c0 + u * s for s in svals]


def build_waypoint_centers(atom_xyz, atom_r, waypoints: List[np.ndarray], step: float,
                            centerline: str, curve_radius: float, curve_iters: float) -> List[np.ndarray]:
    """Join 2+ waypoints (bottom, [mid ...], top) into one continuous list of
    centerline points, sampling each consecutive pair as either a straight
    segment or a curved (locally-refined) segment depending on `centerline`.

    This is the "midpoint" feature: an optional intermediate waypoint lets a
    user who already knows roughly where a channel bends route the profile
    through it. Joining is genuinely just concatenation -- arc length and
    local tangent direction are already computed generically over an
    arbitrary point sequence by profile_along_centers.

    Caveat found in testing: for a genuinely SHARP bend (~90 degrees), plain
    straight sub-segments can still cut across the inner corner of the turn
    regardless of exactly which midpoint residue is chosen near the bend --
    two straight chords meeting at a sharp angle inherently shortcut the
    corner, the same "racing line" effect seen with shortest-path search in
    explore-mode work. This resolved cleanly with centerline='curved' in
    testing (local per-segment refinement pulls the path off the chord), and
    didn't appear at all for a moderate ~45 degree bend even with straight
    sub-segments. For sharp bends, combine --mid with --centerline curved.
    """
    if len(waypoints) < 2:
        raise ValueError("build_waypoint_centers needs at least 2 waypoints.")
    centers: List[np.ndarray] = [waypoints[0]]
    for c0, c1 in zip(waypoints[:-1], waypoints[1:]):
        if centerline == 'curved':
            seg, _, _ = construct_centers_curved(atom_xyz, atom_r, c0, c1, step, curve_radius, curve_iters)
        else:
            seg = sample_straight_centers(c0, c1, step)
        centers.extend(seg[1:])  # skip seg[0]: duplicates the previous segment's endpoint
    return centers


def profile_along_centers(atom_xyz, atom_r, centers, eps, atom_meta, hydro_scale, electro_scale, occupancy_metric,
                           true_area=True, probe=0.0, area_grid_step=0.25,
                           area_min_radius=4.0, area_max_radius=20.0, area_radius_factor=3.0):
    rows = []
    s_acc = 0.0
    index = SpatialIndex(atom_xyz, atom_r)
    for i, c in enumerate(centers):
        if i > 0:
            s_acc += float(np.linalg.norm(centers[i] - centers[i - 1]))
        rmin, tags, hyd, elec = _evaluate_slice(
            atom_xyz, atom_r, atom_meta, c, eps, hydro_scale, electro_scale, index=index)
        rows.append({
            's_A': float(s_acc), 'x': float(c[0]), 'y': float(c[1]), 'z': float(c[2]),
            'radius_A': float(rmin), 'hydro_index': float(hyd), 'electro_index': float(elec),
            'contributors': tags,
        })
    for i in range(len(rows)):
        if i == 0:
            t = np.array([rows[1]['x'] - rows[0]['x'], rows[1]['y'] - rows[0]['y'], rows[1]['z'] - rows[0]['z']], float)
        elif i == len(rows) - 1:
            t = np.array([rows[i]['x'] - rows[i - 1]['x'], rows[i]['y'] - rows[i - 1]['y'], rows[i]['z'] - rows[i - 1]['z']], float)
        else:
            t = np.array([rows[i + 1]['x'] - rows[i - 1]['x'], rows[i + 1]['y'] - rows[i - 1]['y'], rows[i + 1]['z'] - rows[i - 1]['z']], float)
        n = np.linalg.norm(t)
        t = np.array([0.0, 0.0, 1.0]) if n < 1e-9 else (t / n)
        rows[i]['tx'] = float(t[0])
        rows[i]['ty'] = float(t[1])
        rows[i]['tz'] = float(t[2])
        if true_area:
            # Cross-section area is measured perpendicular to the *local*
            # centerline direction, not a fixed global axis -- important once
            # the path is genuinely curved.
            v_local, w_local = orthonormal_basis_from_axis(t)
            c = np.array([rows[i]['x'], rows[i]['y'], rows[i]['z']], float)
            area_geom, area_access, half_extent = slice_area(
                index, c, v_local, w_local, probe=probe,
                min_half_extent=area_min_radius, max_half_extent=area_max_radius,
                radius_factor=area_radius_factor, grid_step=area_grid_step)
            rows[i]['area_A2'] = float(area_geom)
            rows[i]['area_access_A2'] = float(area_access)
            rows[i]['area_half_extent_A'] = float(half_extent)
        if occupancy_metric == 'hydro':
            rows[i]['occ_value'] = float(rows[i]['hydro_index'])
        elif occupancy_metric == 'electro':
            rows[i]['occ_value'] = float(rows[i]['electro_index'])
        else:
            rows[i]['occ_value'] = float(rows[i]['radius_A'])
    return rows


def write_csv(path, rows):
    if not rows:
        return
    cols = list(rows[0].keys())
    if 'occ_value' not in cols:
        cols.append('occ_value')
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _format_pdb_atom_line(serial, name, resName, chainID, resSeq, x, y, z, occupancy, bfactor, element,
                           altLoc=' ', iCode=' '):
    return (
        f"ATOM  {serial:5d} {name:^4}{altLoc}{resName:>3} {chainID}{resSeq:>4}{iCode}   "
        f"{x:8.3f}{y:8.3f}{z:8.3f}{occupancy:6.3f}{bfactor:6.3f}          {element:>2}  \r\n"
    )


def write_mesh_pdb(path, rows, axis_u, rings=24):
    use_local = ('tx' in rows[0])
    if not use_local:
        u = axis_u / (np.linalg.norm(axis_u) + 1e-12)
        base_v, base_w = orthonormal_basis_from_axis(u)
    coords = []
    bvals = []
    occs = []
    for r in rows:
        c = np.array([r['x'], r['y'], r['z']], float)
        rad = max(MIN_PORE_RADIUS_A, float(r['radius_A']))
        occ = float(r.get('occ_value', 0.0))
        if use_local:
            u_loc = np.array([r['tx'], r['ty'], r['tz']], float)
            v, w = orthonormal_basis_from_axis(u_loc)
        else:
            v, w = base_v, base_w
        for k in range(rings):
            ang = 2 * np.pi * (k / rings)
            p = c + rad * (np.cos(ang) * v + np.sin(ang) * w)
            coords.append(p)
            bvals.append(rad)
            occs.append(occ)
    lines = []
    serial_start = 1
    for i, pnt in enumerate(coords, start=serial_start):
        x, y, z = pnt
        b = bvals[i - serial_start]
        occ = occs[i - serial_start]
        lines.append(_format_pdb_atom_line(i, 'C', 'ALA', 'M', 1, x, y, z, occ, b, 'C'))
    # ring & longitudinal CONECT like v8
    n = len(rows)
    R = rings

    def idx(step, k):
        return serial_start + step * R + k

    for step in range(n):
        for k in range(R):
            lines.append(f"CONECT{idx(step, k):5d}{idx(step, (k + 1) % R):5d}\r\n")
    for step in range(n - 1):
        for k in range(R):
            lines.append(f"CONECT{idx(step, k):5d}{idx(step + 1, k):5d}\r\n")
    with open(path, 'w', newline='') as f:
        f.writelines(lines)


def write_centerline_pdb(path, rows):
    lines = []
    serial = 1
    for i, r in enumerate(rows, start=1):
        x, y, z = r['x'], r['y'], r['z']
        b = r['radius_A']
        occ = float(r.get('occ_value', 0.0))
        lines.append(f"HETATM{serial:5d}  O  PORE Z{i:4d}    {x:8.3f}{y:8.3f}{z:8.3f}{occ:6.3f}{b:6.3f}          O \r\n")
        serial += 1
    with open(path, 'w', newline='') as f:
        f.writelines(lines)


# --- Extras (volumes, conductance, passability), matched to CLI v8 ---
def _trap_volume_A3(rows, probe_A):
    if len(rows) < 2:
        return 0.0
    vol = 0.0
    for i in range(len(rows) - 1):
        ds = float(rows[i + 1]['s_A'] - rows[i]['s_A'])
        r0 = max(0.0, float(rows[i]['radius_A']) - probe_A)
        r1 = max(0.0, float(rows[i + 1]['radius_A']) - probe_A)
        vol += 0.5 * (math.pi * r0 * r0 + math.pi * r1 * r1) * ds
    return vol


def _trap_true_volume_A3(rows, area_key: str) -> float:
    """Integrate a rasterized area column over s_A -- the true-cross-section
    counterpart to _trap_volume_A3's circular-radius approximation."""
    if len(rows) < 2 or area_key not in rows[0]:
        return 0.0
    s_vals = [r['s_A'] for r in rows]
    a_vals = [r[area_key] for r in rows]
    return _trapz(a_vals, s_vals)


def _conductance_proxy_S(rows, conductivity_S_per_m):
    kappa = max(1e-12, float(conductivity_S_per_m))
    rho = 1.0 / kappa
    R_pore = 0.0
    blocked = False
    for i in range(len(rows) - 1):
        ds_m = float(rows[i + 1]['s_A'] - rows[i]['s_A']) * 1e-10
        r0 = max(1e-6, float(rows[i]['radius_A'])) * 1e-10
        r1 = max(1e-6, float(rows[i + 1]['radius_A'])) * 1e-10
        if r0 <= 0.0 or r1 <= 0.0:
            blocked = True
        A0 = math.pi * r0 * r0
        A1 = math.pi * r1 * r1
        R_pore += 0.5 * ((1.0 / max(A0, 1e-30)) + (1.0 / max(A1, 1e-30))) * ds_m * rho
    rmin_A = max(1e-6, float(min([r['radius_A'] for r in rows])))
    rmin_m = rmin_A * 1e-10
    R_access = (1.0 / (2.0 * kappa * max(rmin_m, 1e-12)))
    if blocked:
        return 0.0, float('inf'), R_access
    R_total = R_pore + R_access
    return (1.0 / R_total), R_pore, R_access


def _passability_report(rows, pass_radii):
    pass_report = {}
    for sp, radA in pass_radii.items():
        radA = float(radA)
        spans = []
        start = None
        local_min = 1e9
        local_idx = None
        for i, row in enumerate(rows):
            rA = float(row['radius_A'])
            if rA < radA:
                if start is None:
                    start = row['s_A']
                    local_min = rA
                    local_idx = i
                elif rA < local_min:
                    local_min = rA
                    local_idx = i
            elif start is not None:
                end = rows[i]['s_A']
                contrib = rows[local_idx]['contributors'] if local_idx is not None else ''
                spans.append({
                    'start_s_A': float(start), 'end_s_A': float(end),
                    'min_radius_A': float(local_min), 'min_contributors': contrib,
                })
                start = None
                local_min = 1e9
                local_idx = None
        if start is not None:
            end = rows[-1]['s_A']
            contrib = rows[local_idx]['contributors'] if local_idx is not None else ''
            spans.append({
                'start_s_A': float(start), 'end_s_A': float(end),
                'min_radius_A': float(local_min), 'min_contributors': contrib,
            })
        pass_report[sp] = {'is_passable': len(spans) == 0, 'blocked_spans': spans}
    return pass_report


# --- CLI entry point ---
def main(args=None):
    p = argparse.ArgumentParser(description="HOLE-like pore profile with straight/curved centerline options.")
    p.add_argument("pdb", help="Legacy PDB, or mmCIF/other gemmi-readable format (auto-converted)")
    p.add_argument("--top", default="")
    p.add_argument("--bottom", default="")
    p.add_argument("--mid", default="", help="Optional intermediate waypoint (same selection syntax as --top/--bottom).")
    p.add_argument("--interactive", action="store_true")
    p.add_argument("--step", type=float, default=1.0)
    p.add_argument("--eps", type=float, default=0.25)
    p.add_argument("--noH", action="store_true")
    p.add_argument("--noHet", action="store_true")
    p.add_argument("--vdwjson", type=str, default="")
    p.add_argument("--rings", type=int, default=24)
    p.add_argument("--outprefix", type=str, default="holepy_out")
    p.add_argument("--probe", type=float, default=0.0)
    p.add_argument("--conductivity", type=float, default=1.5)
    p.add_argument("--occupancy", choices=["hydro", "electro", "radii"], default="hydro")
    p.add_argument("--hydroscale", choices=["raw", "01"], default="raw")
    p.add_argument("--electroscale", choices=["raw", "01"], default="raw")
    p.add_argument("--passable_json", type=str, default="")
    p.add_argument("--centerline", choices=["straight", "curved"], default="straight")
    p.add_argument("--adaptive", action="store_true")
    p.add_argument("--slope_thresh", type=float, default=0.5)
    p.add_argument("--max_refine", type=int, default=3)
    p.add_argument("--curve_radius", type=float, default=2.0)
    p.add_argument("--curve_iters", type=int, default=3)
    p.add_argument("--no-true-area", dest="true_area", action="store_false",
                    help="Skip rasterized true cross-sectional area/volume (faster; falls back to circular-radius approximation only)")
    p.add_argument("--area-grid-step", type=float, default=0.25, help="Grid spacing (A) for area rasterization")
    p.add_argument("--area-min-radius", type=float, default=4.0, help="Minimum half-extent (A) of the per-slice area search box")
    p.add_argument("--area-max-radius", type=float, default=20.0, help="Maximum half-extent (A) of the per-slice area search box")
    p.add_argument("--area-radius-factor", type=float, default=3.0, help="Area search box half-extent = this * local clearance, clamped to [min,max]")
    p.add_argument("--surface", action="store_true", help="Also write a density map (.mrc) for true isosurface rendering of the pore lumen")
    p.add_argument("--surface-voxel", type=float, default=0.5, help="Voxel spacing (A) for the density map")
    p.add_argument("--surface-margin", type=float, default=3.0, help="Extra padding (A) around the swept pore region in the density map")
    p.set_defaults(true_area=True)
    ns = p.parse_args(args=args)

    try:
        pdb_path = ensure_legacy_pdb(Path(ns.pdb))
    except StructureTooLargeForLegacyPDB as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 4

    atoms = load_pdb_atoms(pdb_path, include_h=not ns.noH, include_hetatm=not ns.noHet)
    if ns.interactive or (not ns.top and not ns.bottom):
        print("Enter TOP:", file=sys.stderr)
        ns.top = input().strip()
        print("Enter BOTTOM:", file=sys.stderr)
        ns.bottom = input().strip()
    top_sel = parse_residue_tokens(ns.top)
    bot_sel = parse_residue_tokens(ns.bottom)
    ca_top = ca_positions_for(atoms, top_sel)
    ca_bot = ca_positions_for(atoms, bot_sel)
    if ca_top.size == 0 or ca_bot.size == 0:
        print("ERROR: Could not find CA atoms.", file=sys.stderr)
        return 3
    c_top = ca_top.mean(axis=0)
    c_bot = ca_bot.mean(axis=0)

    c_mid = None
    if ns.mid.strip():
        mid_sel = parse_residue_tokens(ns.mid)
        ca_mid = ca_positions_for(atoms, mid_sel)
        if ca_mid.size == 0:
            print("ERROR: Could not find CA atoms for --mid selection.", file=sys.stderr)
            return 3
        c_mid = ca_mid.mean(axis=0)

    custom_vdw = {}
    if ns.vdwjson:
        with open(ns.vdwjson, 'r') as jf:
            custom_vdw = normalize_vdw_keys(json.load(jf))

    coords = np.array([[a.x, a.y, a.z] for a in atoms], float)
    radii = np.array([vdw_radius(a.element, custom_vdw) for a in atoms], float)
    metas = [(a.chain, a.resname, a.resi, a.icode) for a in atoms]

    if c_mid is not None:
        waypoints = [c_bot, c_mid, c_top]
        centers = build_waypoint_centers(coords, radii, waypoints, ns.step, ns.centerline, ns.curve_radius, ns.curve_iters)
        rows = profile_along_centers(
            coords, radii, centers, ns.eps, metas, ns.hydroscale, ns.electroscale, ns.occupancy,
            true_area=ns.true_area, probe=ns.probe, area_grid_step=ns.area_grid_step,
            area_min_radius=ns.area_min_radius, area_max_radius=ns.area_max_radius,
            area_radius_factor=ns.area_radius_factor)
        if ns.adaptive:
            print("NOTE: --adaptive has no effect together with --mid; ignoring.", file=sys.stderr)
    elif ns.centerline == 'straight':
        rows, u, L = profile_along_axis(
            coords, radii, c_bot, c_top, ns.step, ns.eps, metas, adaptive=ns.adaptive,
            slope_thresh=ns.slope_thresh, max_refine=ns.max_refine,
            hydro_scale=ns.hydroscale, electro_scale=ns.electroscale, occupancy_metric=ns.occupancy,
            true_area=ns.true_area, probe=ns.probe, area_grid_step=ns.area_grid_step,
            area_min_radius=ns.area_min_radius, area_max_radius=ns.area_max_radius,
            area_radius_factor=ns.area_radius_factor)
    else:
        centers, u, L = construct_centers_curved(coords, radii, c_bot, c_top, ns.step, ns.curve_radius, ns.curve_iters)
        rows = profile_along_centers(
            coords, radii, centers, ns.eps, metas, ns.hydroscale, ns.electroscale, ns.occupancy,
            true_area=ns.true_area, probe=ns.probe, area_grid_step=ns.area_grid_step,
            area_min_radius=ns.area_min_radius, area_max_radius=ns.area_max_radius,
            area_radius_factor=ns.area_radius_factor)

    # volumes & conductance
    volume_geom = _trap_volume_A3(rows, 0.0)
    volume_access = _trap_volume_A3(rows, float(ns.probe))
    volume_true_geom = _trap_true_volume_A3(rows, 'area_A2') if ns.true_area else None
    volume_true_access = _trap_true_volume_A3(rows, 'area_access_A2') if ns.true_area else None
    G_S, R_pore, R_access = _conductance_proxy_S(rows, ns.conductivity)
    G_nS = G_S * 1e9
    # passability reporting
    pass_radii = {}
    if ns.passable_json:
        try:
            with open(ns.passable_json, 'r') as pf:
                pass_radii = json.load(pf)
        except Exception:
            pass
    if not pass_radii:
        pass_radii = {'water': 1.4, 'na': 1.02, 'k': 1.38, 'ca': 1.00}
    pass_report = _passability_report(rows, pass_radii)

    density_map_path = None
    if ns.surface:
        surf_index = SpatialIndex(coords, radii)
        try:
            values_zyx, origin, voxel = build_density_grid(
                surf_index, rows, probe=ns.probe, voxel_size=ns.surface_voxel, margin=ns.surface_margin,
                min_half_extent=ns.area_min_radius, max_half_extent=ns.area_max_radius,
                radius_factor=ns.area_radius_factor)
        except ValueError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 5
        outprefix_tmp = Path(ns.outprefix)
        if outprefix_tmp.suffix.lower() == '.csv':
            outprefix_tmp = outprefix_tmp.with_suffix('')
        density_map_path = outprefix_tmp.with_name(outprefix_tmp.stem + '_density.mrc')
        write_density_map(density_map_path, origin, voxel, values_zyx)

    outprefix = Path(ns.outprefix)
    if outprefix.suffix.lower() == '.csv':
        outprefix = outprefix.with_suffix('')
    csv_path = outprefix.with_suffix('.csv')
    pdb_center = outprefix.with_name(outprefix.stem + '_centerline.pdb')
    pdb_mesh = outprefix.with_name(outprefix.stem + '_mesh.pdb')
    summary_path = outprefix.with_name(outprefix.stem + '_summary.json')

    with open(summary_path, 'w') as jf:
        json.dump({
            'pdb': str(Path(ns.pdb).resolve()), 'top': ns.top, 'bottom': ns.bottom, 'mid': (ns.mid or None), 'centerline': ns.centerline,
            'adaptive': bool(ns.adaptive), 'slope_thresh': float(ns.slope_thresh), 'max_refine': int(ns.max_refine),
            'curve_radius_A': float(ns.curve_radius), 'curve_iters': int(ns.curve_iters),
            'step_A': float(ns.step), 'eps_A': float(ns.eps), 'probe_A': float(ns.probe),
            'length_A': float(rows[-1]['s_A'] if rows else 0.0),
            'volume_geometric_A3': float(volume_geom), 'volume_accessible_A3': float(volume_access),
            'volume_true_geometric_A3': (float(volume_true_geom) if volume_true_geom is not None else None),
            'volume_true_accessible_A3': (float(volume_true_access) if volume_true_access is not None else None),
            'true_area_enabled': bool(ns.true_area),
            'area_grid_step_A': float(ns.area_grid_step),
            'conductivity_S_per_m': float(ns.conductivity),
            'R_pore_ohm': float(R_pore), 'R_access_ohm': float(R_access), 'R_total_ohm': float(R_pore + R_access),
            'G_nS': float(G_nS),
            'passability': pass_report,
            'min_radius_A': float(min([r['radius_A'] for r in rows]) if rows else 0.0),
            'num_samples': len(rows),
            'occupancy_metric': ns.occupancy, 'hydroscale': ns.hydroscale, 'electroscale': ns.electroscale,
            'no_hydrogens': bool(ns.noH), 'ignore_hetatm': bool(ns.noHet),
            'density_map': (str(density_map_path) if density_map_path else None),
            'density_map_voxel_A': (float(ns.surface_voxel) if density_map_path else None),
        }, jf, indent=2)

    write_csv(csv_path, rows)
    write_centerline_pdb(pdb_center, rows)
    write_mesh_pdb(pdb_mesh, axis_u=(c_top - c_bot), rows=rows, rings=ns.rings)
    print(f"Wrote: {csv_path}")
    print(f"Wrote: {pdb_center}")
    print(f"Wrote: {pdb_mesh}")
    print(f"Wrote: {summary_path}")
    if density_map_path:
        print(f"Wrote: {density_map_path}")
        print(f"  -> In ChimeraX: open {density_map_path}  then  volume #<N> level 0")
        print(f"     (level 0 is the accessible pore boundary at probe radius {ns.probe:.2f} A)")
    return 0


# --- GUI entrypoint (used by your ChimeraX tool) ---
def run_pyhole(
    session=None, model=None, pdb=None, *,
    top="", bottom="", mid="",
    vdwjson="", outprefix="pyhole_out",
    step=1.0, eps=0.25, rings=24, noH=False, noHet=False,
    probe=0.0, conductivity=1.5,
    centerline="straight",
    adaptive=False, slope_thresh=0.5, max_refine=3,
    curve_radius=2.0, curve_iters=3,
    occupancy="hydro", hydroscale="raw", electroscale="raw",
    passable_json="",
    true_area=True, area_grid_step=0.25, area_min_radius=4.0, area_max_radius=20.0, area_radius_factor=3.0,
    surface=False, surface_voxel=0.5, surface_margin=3.0,
):
    """Compute pore profile and write outputs. Returns file paths (plus summary)."""
    # 0) Resolve PDB path (GUI passes pdb=..., so we typically skip exporting)
    pdb_path = None
    if pdb:
        pdb_path = Path(pdb)
    elif session is not None and model is not None:
        # fallback export (UI should avoid calling this from worker thread)
        from chimerax.core.commands import run as cxrun
        import tempfile as _tempfile
        id_str = str(getattr(model, 'id_string', getattr(model, 'id', 'model'))).lstrip('#')
        tmp_root = Path(getattr(session, 'user_data_dir', '') or _tempfile.gettempdir())
        tmp_root.mkdir(parents=True, exist_ok=True)
        tmp = tmp_root / f"pyhole_tmp_{id_str}.pdb"
        cxrun(session, f'save "{tmp}" models #{id_str} format pdb')
        pdb_path = tmp
    else:
        raise ValueError("run_pyhole: provide 'pdb' or (session and model).")

    # mmCIF (or anything else gemmi can read) gets transparently converted
    # to a temp legacy PDB file; load_pdb_atoms() only ever sees legacy PDB.
    load_path = ensure_legacy_pdb(pdb_path)

    atoms = load_pdb_atoms(load_path, include_h=not noH, include_hetatm=not noHet)
    top_sel = parse_residue_tokens(top)
    bot_sel = parse_residue_tokens(bottom)
    top_ca = ca_positions_for(atoms, top_sel)
    bot_ca = ca_positions_for(atoms, bot_sel)
    if top_ca.size == 0 or bot_ca.size == 0:
        raise ValueError("Could not find CA for top/bottom selections.")
    c_top = top_ca.mean(axis=0)
    c_bot = bot_ca.mean(axis=0)

    c_mid = None
    if mid and mid.strip():
        mid_sel = parse_residue_tokens(mid)
        mid_ca = ca_positions_for(atoms, mid_sel)
        if mid_ca.size == 0:
            raise ValueError("Could not find CA for --mid selection.")
        c_mid = mid_ca.mean(axis=0)

    custom = {}
    if vdwjson:
        with open(vdwjson, 'r') as jf:
            custom = normalize_vdw_keys(json.load(jf))
    coords = np.array([[a.x, a.y, a.z] for a in atoms], float)
    radii = np.array([vdw_radius(a.element, custom) for a in atoms], float)
    meta = [(a.chain, a.resname, a.resi, a.icode) for a in atoms]

    occ = (occupancy or "hydro").lower()
    if occ.startswith("hydro"):
        occ = "hydro"
    elif occ.startswith("elect"):
        occ = "electro"
    elif occ.startswith("rad"):
        occ = "radii"
    else:
        occ = "hydro"

    area_kwargs = dict(true_area=true_area, probe=probe, area_grid_step=area_grid_step,
                        area_min_radius=area_min_radius, area_max_radius=area_max_radius,
                        area_radius_factor=area_radius_factor)

    if c_mid is not None:
        waypoints = [c_bot, c_mid, c_top]
        centers = build_waypoint_centers(coords, radii, waypoints, step, centerline, curve_radius, curve_iters)
        rows = profile_along_centers(coords, radii, centers, eps, meta, hydroscale, electroscale, occ, **area_kwargs)
    elif centerline == 'curved':
        centers, u, L = construct_centers_curved(coords, radii, c_bot, c_top, step, curve_radius, curve_iters)
        rows = profile_along_centers(coords, radii, centers, eps, meta, hydroscale, electroscale, occ, **area_kwargs)
    else:
        rows, u, L = profile_along_axis(
            coords, radii, c_bot, c_top, step, eps, meta, adaptive, slope_thresh, max_refine,
            hydroscale, electroscale, occ, **area_kwargs)

    outp = Path(outprefix)
    if outp.suffix.lower() == '.csv':
        outp = outp.with_suffix('')
    csv_path = outp.with_suffix('.csv')
    mesh_path = outp.with_name(outp.stem + '_mesh.pdb')
    center_path = outp.with_name(outp.stem + '_centerline.pdb')
    summary_path = outp.with_name(outp.stem + '_summary.json')

    # volumes & conductance proxy
    volume_geom = _trap_volume_A3(rows, 0.0)
    volume_access = _trap_volume_A3(rows, float(probe))
    volume_true_geom = _trap_true_volume_A3(rows, 'area_A2') if true_area else None
    volume_true_access = _trap_true_volume_A3(rows, 'area_access_A2') if true_area else None
    G_S, R_pore, R_access = _conductance_proxy_S(rows, conductivity)
    G_nS = G_S * 1e9

    # passability (optional file like CLI)
    pass_radii = {}
    if passable_json:
        try:
            with open(passable_json, 'r') as pf:
                pass_radii = json.load(pf)
        except Exception:
            pass
    if not pass_radii:
        pass_radii = {'water': 1.4, 'na': 1.02, 'k': 1.38, 'ca': 1.00}
    pass_report = _passability_report(rows, pass_radii)

    density_map_path = None
    if surface:
        surf_index = SpatialIndex(coords, radii)
        values_zyx, origin, voxel = build_density_grid(
            surf_index, rows, probe=probe, voxel_size=surface_voxel, margin=surface_margin,
            min_half_extent=area_min_radius, max_half_extent=area_max_radius, radius_factor=area_radius_factor)
        density_map_path = outp.with_name(outp.stem + '_density.mrc')
        write_density_map(density_map_path, origin, voxel, values_zyx)

    # write files
    write_csv(csv_path, rows)
    write_mesh_pdb(mesh_path, rows, axis_u=(c_top - c_bot), rings=rings)
    write_centerline_pdb(center_path, rows)
    with open(summary_path, 'w') as jf:
        json.dump({
            'pdb': str(pdb_path.resolve()), 'top': top, 'bottom': bottom, 'mid': (mid or None), 'centerline': centerline,
            'adaptive': bool(adaptive), 'slope_thresh': float(slope_thresh), 'max_refine': int(max_refine),
            'curve_radius_A': float(curve_radius), 'curve_iters': int(curve_iters),
            'step_A': float(step), 'eps_A': float(eps),
            'probe_A': float(probe), 'length_A': float(rows[-1]['s_A'] if rows else 0.0),
            'volume_geometric_A3': float(volume_geom), 'volume_accessible_A3': float(volume_access),
            'volume_true_geometric_A3': (float(volume_true_geom) if volume_true_geom is not None else None),
            'volume_true_accessible_A3': (float(volume_true_access) if volume_true_access is not None else None),
            'true_area_enabled': bool(true_area), 'area_grid_step_A': float(area_grid_step),
            'conductivity_S_per_m': float(conductivity),
            'R_pore_ohm': float(R_pore), 'R_access_ohm': float(R_access), 'R_total_ohm': float(R_pore + R_access),
            'G_nS': float(G_nS), 'passability': pass_report,
            'min_radius_A': float(min([r['radius_A'] for r in rows]) if rows else 0.0),
            'num_samples': len(rows), 'occupancy_metric': occ, 'hydroscale': hydroscale, 'electroscale': electroscale,
            'no_hydrogens': bool(noH), 'ignore_hetatm': bool(noHet),
            'density_map': (str(density_map_path) if density_map_path else None),
            'density_map_voxel_A': (float(surface_voxel) if density_map_path else None),
        }, jf, indent=2)

    return {
        "csv": str(csv_path),
        "mesh_pdb": str(mesh_path),
        "centerline_pdb": str(center_path),
        "summary_json": str(summary_path),
        "density_map": (str(density_map_path) if density_map_path else None),
        "g_proxy_S": float(G_S),
        "R_proxy_Ohm": float(R_pore + R_access),
    }

if __name__ == '__main__':
    sys.exit(main())
