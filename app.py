"""
app.py  —  Streamlit web app for CIF → XYZ conversion with 3D viewer
══════════════════════════════════════════════════════════════════════
Run locally:
    pip install streamlit py3Dmol stmol
    streamlit run app.py

Deploy to Streamlit Cloud (free):
    1. Push cif_to_xyz.py + app.py + requirements.txt to a GitHub repo
    2. Go to share.streamlit.io → New app → select your repo
    3. Set Main file path: app.py → Deploy
"""

import streamlit as st
from pathlib import Path

try:
    import py3Dmol
    from stmol import showmol
    HAS_3DMOL = True
except ImportError:
    HAS_3DMOL = False

from cif_to_xyz import cif_to_xyz

# ── Colour scheme for common elements (CPK) ───────────────────────────────────

CPK_COLOURS = {
    "H": "0xFFFFFF", "C": "0x404040", "N": "0x3050F8", "O": "0xFF0D0D",
    "F": "0x90E050", "Cl": "0x1FF01F", "Br": "0xA62929", "I": "0x940094",
    "S": "0xFFFF30", "P": "0xFF8000",
    "Ir": "0x175487", "Ru": "0x248F8F", "Rh": "0x0A7D8C", "Pd": "0x006985",
    "Pt": "0xD0D0E0", "Os": "0x266696", "Re": "0x267DAB", "Au": "0xFFD123",
    "Co": "0xF090A0", "Ni": "0x50D050", "Cu": "0xC88033", "Fe": "0xE06633",
    "Mn": "0x9C7AC7", "Zn": "0x7D80B0",
}

VDW_RADII = {
    "H": 0.31, "C": 0.77, "N": 0.75, "O": 0.73, "F": 0.71,
    "Cl": 0.99, "Br": 1.14, "I": 1.33, "S": 1.03, "P": 1.06,
    "Ir": 1.41, "Ru": 1.46, "Rh": 1.42, "Pd": 1.39, "Pt": 1.36,
    "Os": 1.44, "Co": 1.26, "Ni": 1.24, "Cu": 1.32, "Fe": 1.32,
}


def render_molecule(xyz_text: str, style: str = "stick", width: int = 600, height: int = 400):
    """Render XYZ coordinates with py3Dmol."""
    view = py3Dmol.view(width=width, height=height)
    view.addModel(xyz_text, "xyz")

    if style == "stick":
        view.setStyle({"stick": {"colorscheme": "Jmol", "radius": 0.15}})
    elif style == "ball_and_stick":
        view.setStyle({"stick": {"colorscheme": "Jmol", "radius": 0.10},
                       "sphere": {"colorscheme": "Jmol", "scale": 0.35}})
    elif style == "sphere":
        view.setStyle({"sphere": {"colorscheme": "Jmol", "scale": 0.5}})

    view.setBackgroundColor("0xffffff")
    view.zoomTo()
    view.spin(False)
    showmol(view, height=height, width=width)


# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="CIF → XYZ converter",
    page_icon="⚛️",
    layout="centered",
)

st.title("⚛️ CIF → XYZ converter")
st.caption("Extracts the coordination complex from a CIF file, removes solvent and counterions.")

if not HAS_3DMOL:
    st.warning(
        "3D viewer disabled — install py3Dmol and stmol:  "
        "`pip install py3Dmol stmol`"
    )

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
            help="Sites with occupancy below this are discarded.",
        )
    with col2:
        keep_counterions = st.checkbox("Keep counterions")
        all_molecules    = st.checkbox("All molecules (Z > 1)")
        keep_all         = st.checkbox("Keep all atoms")

# ── Conversion ────────────────────────────────────────────────────────────────

if uploaded is not None:
    if st.button("Convert", type="primary", use_container_width=True):

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
                lines    = xyz_text.splitlines()
                n_atoms  = int(lines[0])
                comment  = lines[1]

                formula = ""
                if "formula:" in comment:
                    formula = comment.split("formula:")[1].split("|")[0].strip()

                # Store in session so viewer persists after widget interactions
                st.session_state["xyz_text"] = xyz_text
                st.session_state["n_atoms"]  = n_atoms
                st.session_state["formula"]  = formula
                st.session_state["stem"]     = tmp_cif.stem

            except Exception as e:
                st.error(f"Conversion failed: {e}")
                st.exception(e)

# ── Results (shown after conversion and on re-render) ─────────────────────────

if "xyz_text" in st.session_state:
    xyz_text = st.session_state["xyz_text"]
    n_atoms  = st.session_state["n_atoms"]
    formula  = st.session_state["formula"]
    stem     = st.session_state["stem"]

    st.success(f"Done — {n_atoms} atoms")

    m1, m2 = st.columns(2)
    m1.metric("Atoms", n_atoms)
    m2.metric("Formula", formula)

    st.download_button(
        label="⬇️ Download XYZ",
        data=xyz_text,
        file_name=stem + ".xyz",
        mime="text/plain",
        use_container_width=True,
    )

    # ── 3D viewer ─────────────────────────────────────────────────────────────
    if HAS_3DMOL:
        st.subheader("3D structure")

        viz_style = st.radio(
            "Display style",
            ["stick", "ball_and_stick", "sphere"],
            horizontal=True,
            format_func=lambda s: s.replace("_", " "),
        )
        render_molecule(xyz_text, style=viz_style)

    # ── Raw preview ───────────────────────────────────────────────────────────
    with st.expander("XYZ preview (first 20 atoms)"):
        st.code("\n".join(xyz_text.splitlines()[:22]), language="text")

else:
    st.info("Upload a CIF file to get started.")

# ── Footer ────────────────────────────────────────────────────────────────────

st.divider()
st.caption(
    "Handles disorder (keeps major conformer), removes crystallisation solvent, "
    "and extracts a single molecule when Z > 1."
)
