"""
app.py  —  Streamlit web app for CIF → XYZ conversion
══════════════════════════════════════════════════════
Run locally:
    pip install streamlit
    streamlit run app.py

Deploy to Streamlit Cloud (free):
    1. Push cif_to_xyz.py + app.py + requirements.txt to a GitHub repo
    2. Go to share.streamlit.io → New app → select your repo
    3. Set Main file path: app.py → Deploy
"""

import io
import streamlit as st
from pathlib import Path
from cif_to_xyz import cif_to_xyz, hill_formula

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="CIF → XYZ converter",
    page_icon="⚛️",
    layout="centered",
)

st.title("⚛️ CIF → XYZ converter")
st.caption("Extracts the coordination complex from a CIF file, removes solvent and counterions.")

# ── File upload ───────────────────────────────────────────────────────────────

uploaded = st.file_uploader("Upload a CIF file", type=["cif"])

# ── Options ───────────────────────────────────────────────────────────────────

with st.expander("Options", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        metal = st.text_input(
            "Metal centre",
            placeholder="Ir, Ru, Pt, Co …",
            help="Element symbol of the metal centre. Leave blank to auto-detect any transition metal.",
        )
        min_occ = st.slider(
            "Min occupancy",
            min_value=0.0, max_value=1.0, value=0.5, step=0.05,
            help="Sites with occupancy below this are discarded (removes minor disorder components).",
        )
    with col2:
        keep_counterions = st.checkbox(
            "Keep counterions",
            help="Retain PF₆⁻, BF₄⁻, Cl⁻, etc. in the output.",
        )
        all_molecules = st.checkbox(
            "All molecules (Z > 1)",
            help="Keep all symmetry-equivalent molecules. Default: one molecule only.",
        )
        keep_all = st.checkbox(
            "Keep all atoms",
            help="No filtering — keep solvent, counterions, and everything else.",
        )

# ── Conversion ────────────────────────────────────────────────────────────────

if uploaded is not None:
    if st.button("Convert", type="primary", use_container_width=True):

        # Write uploaded file to a temp path so cif_to_xyz can read it
        tmp_cif = Path(f"/tmp/{uploaded.name}")
        tmp_xyz = tmp_cif.with_suffix(".xyz")
        tmp_cif.write_bytes(uploaded.getvalue())

        with st.spinner("Converting…"):
            try:
                out_path = cif_to_xyz(
                    cif_path=tmp_cif,
                    output_path=tmp_xyz,
                    target_metal=metal.strip() or None,
                    keep_counterions=keep_counterions,
                    keep_all=keep_all,
                    one_molecule=not all_molecules,
                    min_occupancy=min_occ,
                )

                xyz_text = out_path.read_text(encoding="utf-8")
                lines = xyz_text.splitlines()
                n_atoms = int(lines[0])
                comment = lines[1]

                # Parse formula from comment
                formula = ""
                if "formula:" in comment:
                    formula = comment.split("formula:")[1].split("|")[0].strip()

                # ── Results ──────────────────────────────────────────────────
                st.success(f"Done — {n_atoms} atoms extracted")

                m1, m2 = st.columns(2)
                m1.metric("Atoms", n_atoms)
                m2.metric("Formula", formula)

                # Download button
                st.download_button(
                    label="⬇️ Download XYZ",
                    data=xyz_text,
                    file_name=tmp_cif.stem + ".xyz",
                    mime="text/plain",
                    use_container_width=True,
                )

                # Preview
                with st.expander("Preview (first 20 atoms)"):
                    st.code("\n".join(lines[:22]), language="text")

            except Exception as e:
                st.error(f"Conversion failed: {e}")
                st.exception(e)
else:
    st.info("Upload a CIF file to get started.")

# ── Footer ────────────────────────────────────────────────────────────────────

st.divider()
st.caption(
    "Handles disorder (keeps major conformer), removes crystallisation solvent, "
    "and extracts a single molecule when Z > 1. "
    "Works with monoclinic, triclinic, and orthorhombic cells."
)
