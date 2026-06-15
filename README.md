# CIF → XYZ Converter for Coordination Compounds

A Python tool that extracts a single coordination complex molecule from a CIF 
crystallographic file — removing solvent, counterions, and disorder.

![after](illustration.png)

## What it does

- Resolves crystallographic disorder (keeps the major conformer)
- Removes solvent molecules (DCM, MeOH, MeCN, THF, and others)
- Removes counterions (PF₆⁻, BF₄⁻, Cl⁻, and others)
- Handles Z' < 1 structures by applying symmetry operations to recover 
  the full molecule (e.g. dimers where only half is in the asymmetric unit)
- Outputs standard XYZ format ready for DFT calculations or ML pipelines

## Installation

```bash
git clone https://github.com/katenykhrikova/cif-to-xyz.git
cd cif-to-xyz
pip install -r requirements.txt
```

## Usage

**Python:**
```python
from cif_to_xyz import cif_to_xyz
from pathlib import Path

out = cif_to_xyz(Path("complex.cif"), target_metal="Ir")
```

**Command line:**
```bash
python cif_to_xyz.py complex.cif --metal Ir
python cif_to_xyz.py complex.cif --metal Ir --keep-counterions
python cif_to_xyz.py complex.cif --keep-all   # no filtering
```

**Batch processing:**
```bash
for f in structures/*.cif; do python cif_to_xyz.py "$f" --metal Ir; done
```

## Web app

A Streamlit web interface is available at: [https://cif-to-xyz-converter.streamlit.app]

Upload a CIF file, select options, and download the resulting XYZ — 
no installation required.

## Options

| Option | Default | Description |
|--------|---------|-------------|
| `--metal` | auto | Metal centre element symbol (Ir, Ru, Pt, Co...) |
| `--min-occupancy` | 0.5 | Discard sites below this occupancy |
| `--keep-counterions` | off | Retain counterions in output |
| `--all-molecules` | off | Keep all Z molecules when Z > 1 |
| `--keep-all` | off | No filtering — keep all atoms |

## Requirements

- Python 3.10+
- numpy
- scipy
- pymatgen (optional — used for atomic masses)

## Tested on

Ir(III) cyclometalated complexes from the Cambridge Structural Database, 
including structures with:
- Disordered solvent (DCM, MeCN)
- Disordered ligand conformers across multiple assembly labels
- Z' = 0.5 (half-molecule in asymmetric unit, e.g. Ir dimers)
- Z = 4 (four molecules in unit cell)