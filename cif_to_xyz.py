"""
cif_to_xyz.py  —  CIF → XYZ converter for coordination compounds
═══════════════════════════════════════════════════════════════════════════════
Strategy:
  1. Parse the asymmetric unit DIRECTLY from the CIF text (no pymatgen
     symmetry expansion yet) — this gives raw fractional coordinates exactly
     as deposited, including atoms outside [0,1), which is intentional by the
     crystallographer to keep the molecule contiguous.
  2. Resolve disorder using _atom_site_disorder_assembly / _disorder_group:
     for each assembly keep only the group with the highest occupancy; drop
     all other groups. Unassigned sites (assembly=".") are always kept.
  3. Build a distance-based bond graph in Cartesian space (no PBC needed
     because the asymmetric unit already has contiguous coordinates).
  4. Find connected fragments; classify as complex/solvent/counterion/unknown.
  5. Keep only the metal-containing fragment.
  6. Write Cartesian coordinates to XYZ.

Dependencies:
    pip install pymatgen

Usage:
    python cif_to_xyz.py structure.cif
    python cif_to_xyz.py structure.cif -o result.xyz --metal Ir --verbose
    python cif_to_xyz.py structure.cif --keep-counterions
    python cif_to_xyz.py structure.cif --keep-all
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger("cif_to_xyz")

# ─── Molecule databases ───────────────────────────────────────────────────────

SOLVENT_FORMULAS: frozenset[str] = frozenset({
    "H2 O1", "H2O",
    "C1 H4 O1", "C2 H6 O1", "C3 H8 O1",
    "C2 H3 N1", "C3 H5 N1",
    "C3 H7 N1 O1", "C2 H6 N2 O1",
    "C3 H6 O1", "C4 H8 O1", "C4 H8 O2", "C4 H10 O1",
    "C1 H2 Cl2", "C1 H1 Cl3", "C1 Cl4", "C2 H4 Cl2",
    "C6 H6", "C7 H8", "C8 H10",
    "C5 H12", "C6 H12", "C6 H14", "C7 H16",
    "C2 H6 O1 S1", "C5 H5 N1", "C1 H3 N1 O2",
})

COUNTERION_FORMULAS: frozenset[str] = frozenset({
    "Cl1", "Br1", "I1", "F1",
    "B1 F4", "F6 P1", "F3 O3 S1", "C1 F3 O3 S1",
    "N1 O3", "C2 O4", "O4 S1", "As1 F6", "Sb1 F6", "B1 C24 H20",
})

TRANSITION_METALS: frozenset[str] = frozenset({
    "Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn",
    "Y","Zr","Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd",
    "Hf","Ta","W","Re","Os","Ir","Pt","Au","Hg",
    "La","Ce","Pr","Nd","Pm","Sm","Eu","Gd","Tb",
    "Dy","Ho","Er","Tm","Yb","Lu",
})

# Covalent radii (Å) for bond detection
COVALENT_RADII: dict[str, float] = {
    "H":0.31,"C":0.76,"N":0.71,"O":0.66,"F":0.57,"P":1.07,"S":1.05,
    "Cl":1.02,"Br":1.20,"I":1.39,
    "Ir":1.41,"Ru":1.46,"Rh":1.42,"Pd":1.39,"Pt":1.36,"Os":1.44,
    "Au":1.36,"Ag":1.45,"Cu":1.32,"Ni":1.24,"Co":1.26,"Fe":1.32,
    "Mn":1.61,"Cr":1.39,"V":1.53,"Ti":1.60,"Sc":1.70,
    "Zn":1.22,"Cd":1.44,"Hg":1.32,
}
DEFAULT_RADIUS = 1.5

# ─── Step 1: parse CIF ────────────────────────────────────────────────────────

def _strip_uncertainty(val: str) -> float:
    """Convert '0.7557(2)' or '0.7557' to float."""
    return float(val.split("(")[0])


def parse_cif(cif_path: Path) -> tuple[np.ndarray, list[dict]]:
    """
    Parse a CIF file directly.

    Returns
    -------
    lattice_matrix : (3,3) array, Cartesian lattice vectors (rows = a, b, c)
    sites          : list of dicts with keys:
                       label, symbol, frac (np.ndarray shape (3,)),
                       occupancy, assembly, group
    """
    text = cif_path.read_text(encoding="utf-8", errors="replace")

    # ── Cell parameters ───────────────────────────────────────────────────────
    def _cell(key):
        m = re.search(key + r"\s+([\d.]+(?:\(\d+\))?)", text)
        return _strip_uncertainty(m.group(1)) if m else None

    a = _cell("_cell_length_a")
    b = _cell("_cell_length_b")
    c = _cell("_cell_length_c")
    alpha = np.radians(_cell("_cell_angle_alpha") or 90.0)
    beta  = np.radians(_cell("_cell_angle_beta")  or 90.0)
    gamma = np.radians(_cell("_cell_angle_gamma") or 90.0)

    # Fractional → Cartesian matrix (standard crystallographic convention)
    cos_a, cos_b, cos_g = np.cos(alpha), np.cos(beta), np.cos(gamma)
    sin_g = np.sin(gamma)
    vol_factor = np.sqrt(1 - cos_a**2 - cos_b**2 - cos_g**2
                         + 2*cos_a*cos_b*cos_g)
    M = np.array([
        [a,          b*cos_g,      c*cos_b                       ],
        [0,          b*sin_g,      c*(cos_a - cos_b*cos_g)/sin_g ],
        [0,          0,            c*vol_factor/sin_g             ],
    ])  # columns are a, b, c vectors; M @ frac = cart

    # ── Atom site loop ────────────────────────────────────────────────────────
    blocks = re.split(r"\nloop_", text)
    atom_block = next(
        (bl for bl in blocks
         if "_atom_site_label" in bl and "_atom_site_fract_x" in bl),
        None
    )
    if atom_block is None:
        raise ValueError("No _atom_site loop found in CIF")

    blines = atom_block.splitlines()
    headers: list[str] = []
    raw_rows: list[list[str]] = []
    in_headers = False

    for line in blines:
        s = line.strip()
        if s.startswith("_atom_site"):
            headers.append(s)
            in_headers = True
        elif in_headers and s and not s.startswith("_") and not s.startswith("#"):
            raw_rows.append(s.split())
        elif in_headers and (s.startswith("_") or s.startswith("loop_")):
            break

    hi = {h: i for i, h in enumerate(headers)}

    def col(name, default=None):
        return hi.get(name, default)

    IL = col("_atom_site_label");       assert IL is not None
    IT = col("_atom_site_type_symbol"); assert IT is not None
    IX = col("_atom_site_fract_x");     assert IX is not None
    IY = col("_atom_site_fract_y");     assert IY is not None
    IZ = col("_atom_site_fract_z");     assert IZ is not None
    IO = col("_atom_site_occupancy")
    IA = col("_atom_site_disorder_assembly")
    IG = col("_atom_site_disorder_group")

    sites: list[dict] = []
    for r in raw_rows:
        if len(r) <= max(IX, IY, IZ):
            continue
        try:
            frac = np.array([
                _strip_uncertainty(r[IX]),
                _strip_uncertainty(r[IY]),
                _strip_uncertainty(r[IZ]),
            ])
        except ValueError:
            continue

        sites.append({
            "label":    r[IL],
            "symbol":   r[IT],
            "frac":     frac,
            "occupancy": _strip_uncertainty(r[IO]) if IO is not None else 1.0,
            "assembly": r[IA] if IA is not None else ".",
            "group":    r[IG] if IG is not None else ".",
        })

    logger.info(f"Parsed {len(sites)} sites from asymmetric unit")
    return M, sites


# ─── Step 2: resolve disorder ─────────────────────────────────────────────────

def resolve_disorder(sites: list[dict],
                     min_occupancy: float = 0.5) -> list[dict]:
    """
    Keep only the major conformer for each disorder assembly.

    Rules:
      - assembly="." → always keep (ordered atom)
      - For each assembly letter (A, B, C, …):
          find the group with the highest mean occupancy → keep only that group
      - After group selection, drop any site with occupancy < min_occupancy
        (catches residual minor-conformer sites not labelled with a group)
    """
    # Group sites by assembly
    by_assembly: dict[str, list[dict]] = defaultdict(list)
    for s in sites:
        by_assembly[s["assembly"]].append(s)

    kept: list[dict] = []

    for assembly, asm_sites in by_assembly.items():
        if assembly == ".":
            # Ordered — keep all
            kept.extend(asm_sites)
            continue

        # Find groups within this assembly
        groups: dict[str, list[dict]] = defaultdict(list)
        for s in asm_sites:
            groups[s["group"]].append(s)

        # Pick the group with highest mean occupancy
        best_group = max(
            groups,
            key=lambda g: np.mean([s["occupancy"] for s in groups[g]])
        )
        mean_occ = np.mean([s["occupancy"] for s in groups[best_group]])
        logger.info(
            f"  Assembly {assembly}: keeping group {best_group} "
            f"(mean occ={mean_occ:.2f}), "
            f"dropping {len(asm_sites) - len(groups[best_group])} atoms"
        )
        kept.extend(groups[best_group])

    # Final low-occupancy filter
    before = len(kept)
    kept = [s for s in kept if s["occupancy"] >= min_occupancy]
    n_dropped = before - len(kept)
    if n_dropped:
        logger.info(f"  Dropped {n_dropped} sites with occupancy < {min_occupancy}")

    logger.info(f"Disorder resolved: {len(kept)} sites kept (was {len(sites)})")
    return kept


# ─── Step 3: build bond graph ─────────────────────────────────────────────────

def frac_to_cart(frac: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Convert fractional to Cartesian coordinates. M columns = lattice vectors."""
    return M @ frac


def build_bond_graph_cart(symbols: list[str],
                          coords: np.ndarray,
                          tolerance: float = 0.4) -> dict[int, set[int]]:
    """
    Build covalent bond graph from Cartesian coordinates (no PBC).
    Two atoms are bonded if distance < (r_i + r_j) * (1 + tolerance).
    """
    n = len(symbols)
    radii = np.array([COVALENT_RADII.get(s, DEFAULT_RADIUS) for s in symbols])
    adj: dict[int, set[int]] = defaultdict(set)

    for i in range(n):
        for j in range(i + 1, n):
            threshold = (radii[i] + radii[j]) * (1 + tolerance)
            dist = float(np.linalg.norm(coords[i] - coords[j]))
            if dist < threshold:
                adj[i].add(j)
                adj[j].add(i)

    return adj


# ─── Step 4: connected components ─────────────────────────────────────────────

def connected_components(adj: dict[int, set[int]],
                          n: int) -> list[list[int]]:
    visited = [False] * n
    components: list[list[int]] = []
    for start in range(n):
        if visited[start]:
            continue
        comp: list[int] = []
        queue = [start]
        visited[start] = True
        while queue:
            node = queue.pop()
            comp.append(node)
            for nb in adj.get(node, []):
                if not visited[nb]:
                    visited[nb] = True
                    queue.append(nb)
        components.append(sorted(comp))
    return components


# ─── Step 5: classify and filter ──────────────────────────────────────────────

def hill_formula(symbols: list[str]) -> str:
    count: dict[str, int] = defaultdict(int)
    for s in symbols:
        count[s] += 1
    order = [el for el in ("C", "H") if el in count]
    order += sorted(k for k in count if k not in ("C", "H"))
    return " ".join(f"{el}{count[el]}" for el in order)


def contains_metal(indices: list[int], symbols: list[str],
                   target: Optional[str] = None) -> bool:
    for i in indices:
        el = symbols[i]
        if target:
            if el == target:
                return True
        elif el in TRANSITION_METALS:
            return True
    return False


def classify_fragment(indices: list[int], symbols: list[str],
                      target_metal: Optional[str]) -> str:
    syms = [symbols[i] for i in indices]
    formula = hill_formula(syms)
    if contains_metal(indices, symbols, target_metal):
        return "complex"
    if len(indices) == 1:
        return "counterion"
    if formula in SOLVENT_FORMULAS:
        return "solvent"
    if formula in COUNTERION_FORMULAS:
        return "counterion"
    return "unknown"


# ─── Main extraction ──────────────────────────────────────────────────────────

def extract_complex(
    symbols: list[str],
    coords: np.ndarray,
    target_metal: Optional[str] = None,
    keep_counterions: bool = False,
    keep_all: bool = False,
    one_molecule: bool = True,
) -> tuple[list[str], np.ndarray, dict]:
    """
    Extract the coordination complex from Cartesian coordinates.
    Returns (symbols, coords centred at CoM, report).
    """
    adj = build_bond_graph_cart(symbols, coords)
    components = connected_components(adj, len(symbols))
    logger.info(f"Molecular fragments: {len(components)}")

    classified: dict[str, list[list[int]]] = defaultdict(list)
    for comp in components:
        label = classify_fragment(comp, symbols, target_metal)
        syms  = [symbols[i] for i in comp]
        logger.debug(f"  [{label:12s}] {len(comp):4d} atoms  {hill_formula(syms)}")
        classified[label].append(comp)

    report = {k: len(v) for k, v in classified.items()}

    if not classified["complex"]:
        logger.warning("No metal fragment found — using largest fragment")
        classified["complex"].append(max(components, key=len))

    n_cx = len(classified["complex"])
    if n_cx > 1:
        if one_molecule:
            best = max(classified["complex"], key=len)
            logger.info(f"Z={n_cx}: keeping 1 molecule ({len(best)} atoms)")
            classified["complex"] = [best]
        else:
            logger.info(f"Z={n_cx}: keeping all")

    selected: list[int] = [i for frag in classified["complex"] for i in frag]

    if keep_all:
        for label in ("solvent", "counterion", "unknown"):
            selected += [i for frag in classified[label] for i in frag]
    else:
        if keep_counterions:
            selected += [i for frag in classified["counterion"] for i in frag]
            logger.info(f"Counterions kept: {len(classified['counterion'])}")
        for label in ("solvent", "counterion", "unknown"):
            frags = classified[label]
            if not frags or (label == "counterion" and keep_counterions):
                continue
            fmls = [hill_formula([symbols[i] for i in f]) for f in frags]
            msg  = f"Removed {len(frags)} {label}: {', '.join(fmls)}"
            logger.warning(msg + " — use --keep-all to retain") \
                if label == "unknown" else logger.info(msg)

    selected = sorted(set(selected))
    out_symbols = [symbols[i] for i in selected]
    out_coords  = coords[selected].copy()

    # Centre at centre of mass
    try:
        from pymatgen.core import Element as PMGElement
        masses = np.array([float(PMGElement(s).atomic_mass) for s in out_symbols])
    except Exception:
        # Fallback: approximate masses
        approx = {"H":1,"C":12,"N":14,"O":16,"F":19,"P":31,"S":32,
                  "Cl":35,"Br":80,"I":127,"Ir":192,"Ru":101,"Pt":195,
                  "Pd":106,"Rh":103,"Os":190,"Au":197,"Re":186}
        masses = np.array([approx.get(s, 50) for s in out_symbols], dtype=float)

    com = (out_coords * masses[:, None]).sum(0) / masses.sum()
    out_coords -= com

    return out_symbols, out_coords, report


# ─── I/O ─────────────────────────────────────────────────────────────────────

def write_xyz(symbols: list[str], coords: np.ndarray,
              path: Path, comment: str = "") -> None:
    lines = [str(len(symbols)), comment]
    for sym, (x, y, z) in zip(symbols, coords):
        lines.append(f"{sym:<4s}  {x:14.8f}  {y:14.8f}  {z:14.8f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info(f"XYZ written: {path}  ({len(symbols)} atoms)")


def cif_to_xyz(
    cif_path: Path,
    output_path: Optional[Path] = None,
    target_metal: Optional[str] = None,
    keep_counterions: bool = False,
    keep_all: bool = False,
    one_molecule: bool = True,
    min_occupancy: float = 0.5,
) -> Path:
    """
    Convert a CIF file to XYZ, retaining only the coordination complex.

    Parameters
    ----------
    cif_path         : input CIF file
    output_path      : output XYZ (default: <cif_stem>.xyz)
    target_metal     : metal centre symbol, e.g. 'Ir'
    keep_counterions : include counterions in output
    keep_all         : no filtering — keep all atoms
    one_molecule     : keep only one molecule when Z > 1 (default: True)
    min_occupancy    : discard sites with occupancy below this (default 0.5)
    """
    if output_path is None:
        output_path = cif_path.with_suffix(".xyz")

    # 1. Parse asymmetric unit directly from CIF text
    M, sites = parse_cif(cif_path)

    # 2. Resolve disorder using assembly/group labels
    logger.info("Resolving disorder...")
    sites = resolve_disorder(sites, min_occupancy=min_occupancy)

    # 3. Convert fractional → Cartesian (preserving coords outside [0,1))
    symbols = [s["symbol"] for s in sites]
    coords  = np.array([frac_to_cart(s["frac"], M) for s in sites])

    logger.info(f"Clean asymmetric unit: {len(symbols)} atoms")

    # 4-5. Find and extract the complex
    out_symbols, out_coords, report = extract_complex(
        symbols, coords,
        target_metal=target_metal,
        keep_counterions=keep_counterions,
        keep_all=keep_all,
        one_molecule=one_molecule,
    )

    comment = (
        f"Source: {cif_path.name} | formula: {hill_formula(out_symbols)} | "
        + " | ".join(f"{k}={v}" for k, v in report.items())
    )
    write_xyz(out_symbols, out_coords, output_path, comment)
    return output_path


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Convert CIF to XYZ for coordination compounds",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cif_to_xyz.py complex.cif --metal Ir
  python cif_to_xyz.py complex.cif --metal Ir -o mol.xyz --verbose
  python cif_to_xyz.py complex.cif --keep-counterions
  python cif_to_xyz.py complex.cif --keep-all
  python cif_to_xyz.py complex.cif --all-molecules
  python cif_to_xyz.py complex.cif --min-occupancy 0.4

Batch:
  for f in *.cif; do python cif_to_xyz.py "$f" --metal Ir; done
        """,
    )
    ap.add_argument("cif", type=Path)
    ap.add_argument("-o", "--output", type=Path, default=None)
    ap.add_argument("--metal", type=str, default=None)
    ap.add_argument("--keep-counterions", action="store_true")
    ap.add_argument("--keep-all", action="store_true")
    ap.add_argument("--all-molecules", action="store_true")
    ap.add_argument("--min-occupancy", type=float, default=0.5)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s  %(message)s",
    )

    if not args.cif.exists():
        sys.exit(f"File not found: {args.cif}")

    try:
        out = cif_to_xyz(
            cif_path=args.cif,
            output_path=args.output,
            target_metal=args.metal,
            keep_counterions=args.keep_counterions,
            keep_all=args.keep_all,
            one_molecule=not args.all_molecules,
            min_occupancy=args.min_occupancy,
        )
        print(f"Done: {out}")
    except Exception as exc:
        logger.error(str(exc))
        if args.verbose:
            import traceback; traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
