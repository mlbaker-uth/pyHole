#!/usr/bin/env python3
"""
pyhole_plotter.py — Make publication-quality radius-vs-axis plots from pyHole outputs.

Features
- Accepts one or more pyHole result prefixes (or CSV/JSON files).
- Draws radius_A (Å) vs s_A (Å) with consistent axis limits.
- Shades passability "blocked spans" from the *_summary.json (species-selectable).
- Annotates min radius and conductance proxy (G_nS).
- Supports overlays or an N×M grid with per-panel titles.
- Saves PNG and/or PDF.
- Optional secondary axis for hydrophobicity/electrostatics/occupancy.
- Optional axis swap (vertical profile): radius on X, s_A on Y.
- NEW: Secondary-axis direction control (asc/desc) and default colors: radius BLACK, electrostatics RED, hydrophobicity BLUE.

Examples
  # Single plot
  python pyhole_plotter.py outputs/sample --out fig1C --ylim 0.5,8.0 --species water

  # Overlay with custom colors (comma list)
  python pyhole_plotter.py stateA stateB --overlay --labels "A,B" --primary_color "black,orange" --out overlay

  # 1x5 grid
  python pyhole_plotter.py P1 P2 P3 P4 P5 --grid 1x5 --out fig2 --titles "Prot1,Prot2,Prot3,Prot4,Prot5"

  # Vertical profile with electrostatics as secondary axis, descending scale
  python pyhole_plotter.py outputs/sample --swap_axes --secondary electro --sec_ylim -1,1 --sec_order desc --out fig_vertical
"""

import argparse, json, sys
from pathlib import Path
from typing import List, Optional, Tuple
import pandas as pd
import matplotlib.pyplot as plt

def _parse_pair_any(arg, *, flag_name='ylim') -> Optional[Tuple[float, float]]:
    """Accept either [a b], "a,b", or "a b" and return (a,b) floats; return None if arg is None."""
    if arg is None:
        return None
    if isinstance(arg, list):
        s = " ".join(arg)
    else:
        s = str(arg)
    s = s.strip().replace(',', ' ')
    toks = [t for t in s.split() if t]
    if len(toks) != 2:
        raise SystemExit(f"--{flag_name} requires two values (e.g., --{flag_name} 0.5 8.0 or --{flag_name}=0.5,8.0)")
    try:
        return float(toks[0]), float(toks[1])
    except Exception as e:
        raise SystemExit(f"--{flag_name} values must be numeric: {toks}") from e

def _parse_hlines(s: Optional[str]):
    out = []
    if not s:
        return out
    for tok in s.split(','):
        tok = tok.strip()
        if not tok:
            continue
        if ':' in tok:
            y, lbl = tok.split(':', 1)
            out.append((float(y), lbl.strip()))
        else:
            out.append((float(tok), None))
    return out

def _detect_files(arg: str):
    """Return (csv_path, summary_path, label) for a prefix or CSV/JSON path."""
    p = Path(arg)
    if p.suffix.lower() == '.csv':
        csv_path = p
        summary_path = p.with_name(p.stem + '_summary.json')
        label = p.stem
    elif p.suffix.lower() == '.json' and p.name.endswith('_summary.json'):
        summary_path = p
        csv_path = p.with_name(p.name.replace('_summary.json', '.csv'))
        label = p.stem.replace('_summary', '')
    else:
        csv_path = p.with_suffix('.csv')
        summary_path = p.with_name(p.name + '_summary.json')
        label = p.name
    return csv_path, summary_path, label

def _load_one(arg: str):
    csv_path, summary_path, label = _detect_files(arg)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    if summary_path.exists():
        with open(summary_path, 'r') as f:
            summary = json.load(f)
    else:
        sys.stderr.write(f"[warn] Summary JSON not found: {summary_path}\n")
        summary = {}
    df = pd.read_csv(csv_path)
    if 's_A' not in df.columns or 'radius_A' not in df.columns:
        raise ValueError(f"{csv_path} missing required columns s_A and radius_A")
    return df, summary, label

def _shade_blocked(ax, summary: dict, species: Optional[str], *, swap_axes: bool=False):
    if not summary:
        return
    passes = summary.get('passability', {})
    if not passes:
        return
    items = [(species, passes.get(species))] if (species and species in passes) else list(passes.items())
    for spec, info in items:
        if not info:
            continue
        spans = info.get('blocked_spans', [])
        for b in spans:
            x0 = b.get('start_s_A', None)
            x1 = b.get('end_s_A', None)
            if x0 is None or x1 is None:
                continue
            if swap_axes:
                ax.axhspan(x0, x1, color='0.85', alpha=0.5, lw=0, label=None)
            else:
                ax.axvspan(x0, x1, color='0.85', alpha=0.5, lw=0, label=None)

def _annotate_stats(ax, summary: dict):
    if not summary:
        return
    txts = []
    if 'min_radius_A' in summary:
        try:
            txts.append(f"min r = {float(summary['min_radius_A']):.2f} Å")
        except Exception:
            pass
    if 'G_nS' in summary:
        try:
            txts.append(f"G ≈ {float(summary['G_nS']):.2f} nS")
        except Exception:
            pass
    if not txts:
        return
    ax.text(0.98, 0.02, "  •  ".join(txts), transform=ax.transAxes,
            ha='right', va='bottom', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='0.7'))

def _apply_paper_style():
    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "axes.linewidth": 0.8,
        "grid.linewidth": 0.5,
    })

def _pick_secondary_column(df, which: str):
    cols_lower = {c.lower(): c for c in df.columns}
    def pick(*names):
        for n in names:
            if n in df.columns:
                return n
            if n in cols_lower:
                return cols_lower[n]
        return None
    if which == 'hydro':
        col = pick('hydro_index', 'hydropathy', 'hydro')
        label = "Hydropathy index"
    elif which == 'electro':
        col = pick('electro_index', 'electrostatics', 'electro')
        label = "Electrostatics index"
    else:
        col = pick('occ_value', 'occupancy', 'occ')
        label = "Occupancy value"
    return col, label

def plot_single(df, summary, *, ax=None, title=None, ylim=None, species=None, hlines=None,
                color='black', label=None, swap_axes=False, secondary=None, sec_ylim=None, sec_label=None,
                sec_color=None, sec_order='asc'):
    if ax is None:
        fig, ax = plt.subplots(figsize=(4,3))
    # main line
    if swap_axes:
        ax.plot(df['radius_A'], df['s_A'], lw=1.8, color=color, label=label)
        ax.set_xlabel("Radius (Å)")
        ax.set_ylabel("Axial coordinate s (Å)")
        if ylim:
            ax.set_xlim(*ylim)  # ylim refers to radius range
        if hlines:
            for y, lbl in hlines:
                ax.axvline(y, color='0.75', lw=0.8, ls='--')
                if lbl:
                    ax.text(y, ax.get_ylim()[1], f" {lbl}", rotation=90,
                            ha='left', va='top', fontsize=8, color='0.35')
        _shade_blocked(ax, summary, species, swap_axes=True)
    else:
        ax.plot(df['s_A'], df['radius_A'], lw=1.8, color=color, label=label)
        ax.set_xlabel("Axial coordinate s (Å)")
        ax.set_ylabel("Radius (Å)")
        if ylim:
            ax.set_ylim(*ylim)
        if hlines:
            for y, lbl in hlines:
                ax.axhline(y, color='0.75', lw=0.8, ls='--')
                if lbl:
                    y0,y1 = ax.get_ylim()
                    frac = (y - y0)/(y1 - y0) if y1 != y0 else 0.0
                    ax.text(0.01, frac, f" {lbl}", transform=ax.transAxes,
                            ha='left', va='center', fontsize=8, color='0.35')
        _shade_blocked(ax, summary, species, swap_axes=False)

    # secondary axis
    if secondary:
        col, auto_label = _pick_secondary_column(df, secondary)
        if col is not None:
            # choose default color by type if none provided
            use_sec_color = sec_color
            if use_sec_color is None:
                if secondary == 'electro':
                    use_sec_color = 'red'
                elif secondary == 'hydro':
                    use_sec_color = 'blue'
                else:
                    use_sec_color = '0.4'  # gray for occupancy
            if swap_axes:
                ax2 = ax.twiny()
                ax2.plot(df[col], df['s_A'], lw=1.2, alpha=0.95, color=use_sec_color)
                ax2.set_xlabel(sec_label or auto_label)
                if sec_ylim:
                    ax2.set_xlim(*sec_ylim)
                if sec_order == 'desc':
                    ax2.invert_xaxis()
            else:
                ax2 = ax.twinx()
                ax2.plot(df['s_A'], df[col], lw=1.2, alpha=0.95, color=use_sec_color)
                ax2.set_ylabel(sec_label or auto_label)
                if sec_ylim:
                    ax2.set_ylim(*sec_ylim)
                if sec_order == 'desc':
                    ax2.invert_yaxis()
        else:
            sys.stderr.write(f"[warn] Secondary column not found for '{secondary}' in columns: {list(df.columns)}\n" )

    _annotate_stats(ax, summary)
    if title:
        ax.set_title(title)
    if label:
        ax.legend(frameon=False, loc='upper right')
    ax.grid(True, alpha=0.25)
    return ax

def main(argv=None):
    ap = argparse.ArgumentParser(description="Plot pyHole radius profiles from CSV/summary outputs.")
    ap.add_argument('inputs', nargs='+', help="Prefixes or CSV/JSON files from pyHole runs")
    ap.add_argument('--out', default='pyhole_plot', help="Output basename (without extension)")
    ap.add_argument('--overlay', action='store_true', help="Overlay all inputs in one axes")
    ap.add_argument('--grid', default=None, help="Grid like '1x5' for multi-panel layout")
    ap.add_argument('--labels', default=None, help="Comma-separated labels matching inputs")
    ap.add_argument('--titles', default=None, help="Comma-separated titles (grid mode)")
    ap.add_argument('--ylim', nargs='+', default=['0.5','8.0'], help="Radius range (lo hi) or 'lo,hi'; default 0.5 8.0")
    ap.add_argument('--species', default='water', help="Passability species to shade; default 'water'. Use '' to shade all")
    ap.add_argument('--hlines', default='1.4:water', help="Reference lines 'y[:label],y[:label],...' (e.g., '1.0:Ca2+,1.4:water')")
    ap.add_argument('--pdf', action='store_true', help="Also save PDF alongside PNG")
    ap.add_argument('--style_paper', action='store_true', help="Apply compact journal-like styling")
    ap.add_argument('--swap_axes', action='store_true', help="Swap axes: radius on X, s_A on Y (vertical profile)")
    ap.add_argument('--secondary', choices=['hydro','electro','occ'], default=None,
                    help="Plot a secondary curve: hydrophobicity ('hydro'), electrostatics ('electro'), or occupancy ('occ')")
    ap.add_argument('--sec_ylim', nargs='+', default=None,
                    help="Limits for the secondary axis as two numbers or 'a,b'. Applies to Y when normal, X when --swap_axes.")
    ap.add_argument('--sec_label', default=None, help="Override label for the secondary axis")
    ap.add_argument('--sec_order', choices=['asc','desc'], default='asc',
                    help="Secondary axis direction: asc (small→large, default) or desc (large→small)")
    ap.add_argument('--primary_color', default='black',
                    help="Primary curve color; for overlay/grid, a comma list applies per series/panel (e.g., 'black,orange,green')")
    ap.add_argument('--secondary_color', default=None, help="Secondary curve color (single color). Defaults: red for electro, blue for hydro, gray for occ." )
    args = ap.parse_args(argv)

    if args.style_paper:
        _apply_paper_style()

    # Parse limits
    args.ylim = _parse_pair_any(args.ylim, flag_name='ylim')
    args.sec_ylim = _parse_pair_any(args.sec_ylim, flag_name='sec_ylim') if args.sec_ylim is not None else None

    labels = [s.strip() for s in args.labels.split(',')] if args.labels else None
    titles = [s.strip() for s in args.titles.split(',')] if args.titles else None
    hlines = _parse_hlines(args.hlines)

    # Primary colors list
    primary_colors = [c.strip() for c in args.primary_color.split(',')] if args.primary_color else ['black']

    datasets = []
    for inp in args.inputs:
        df, summary, label = _load_one(inp)
        datasets.append((df, summary, label))

    # overlay mode
    if args.overlay:
        fig, ax = plt.subplots(figsize=(4.8,3.4))
        for i, (df, summary, label) in enumerate(datasets):
            lbl = labels[i] if labels and i < len(labels) else label
            color = primary_colors[i % len(primary_colors)] if primary_colors else 'black'
            plot_single(df, summary, ax=ax, title=None, ylim=args.ylim,
                        species=args.species if args.species else None,
                        hlines=hlines, color=color, label=lbl,
                        swap_axes=args.swap_axes, secondary=args.secondary,
                        sec_ylim=args.sec_ylim, sec_label=args.sec_label,
                        sec_color=args.secondary_color, sec_order=args.sec_order)
        fig.tight_layout()
        out_png = f"{args.out}.png"
        plt.savefig(out_png)
        if args.pdf:
            plt.savefig(f"{args.out}.pdf")
        print(f"Wrote {out_png}")
        return 0

    # grid mode
    if args.grid:
        try:
            nr, nc = args.grid.lower().split('x')
            nr, nc = int(nr), int(nc)
        except Exception:
            raise SystemExit("--grid must be like '1x5' or '2x3'")
        if nr * nc < len(datasets):
            raise SystemExit(f"--grid {nr}x{nc} has fewer panels than inputs ({len(datasets)})")
        fig, axes = plt.subplots(nr, nc, figsize=(nc*3.6, nr*2.8), squeeze=False)
        for idx, (df, summary, label) in enumerate(datasets):
            r, c = divmod(idx, nc)
            ttl = titles[idx] if titles and idx < len(titles) else label
            color = primary_colors[idx % len(primary_colors)] if primary_colors else 'black'
            plot_single(df, summary, ax=axes[r][c], title=ttl, ylim=args.ylim,
                        species=args.species if args.species else None,
                        hlines=hlines, color=color, label=None,
                        swap_axes=args.swap_axes, secondary=args.secondary,
                        sec_ylim=args.sec_ylim, sec_label=args.sec_label,
                        sec_color=args.secondary_color, sec_order=args.sec_order)
        # hide unused axes
        for idx in range(len(datasets), nr*nc):
            r, c = divmod(idx, nc)
            axes[r][c].axis('off')
        fig.tight_layout()
        out_png = f"{args.out}.png"
        plt.savefig(out_png)
        if args.pdf:
            plt.savefig(f"{args.out}.pdf")
        print(f"Wrote {out_png}")
        return 0

    # single
    df, summary, label = datasets[0]
    color = primary_colors[0] if primary_colors else 'black'
    fig, ax = plt.subplots(figsize=(4.8,3.4))
    plot_single(df, summary, ax=ax, title=label, ylim=args.ylim,
                species=args.species if args.species else None,
                hlines=hlines, color=color, label=None,
                swap_axes=args.swap_axes, secondary=args.secondary,
                sec_ylim=args.sec_ylim, sec_label=args.sec_label,
                sec_color=args.secondary_color, sec_order=args.sec_order)
    fig.tight_layout()
    out_png = f"{args.out}.png"
    plt.savefig(out_png)
    if args.pdf:
        plt.savefig(f"{args.out}.pdf")
    print(f"Wrote {out_png}")
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
