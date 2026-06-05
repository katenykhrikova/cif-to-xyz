"""
app.py  —  Streamlit web app for CIF → XYZ conversion with 3D viewer
══════════════════════════════════════════════════════════════════════
Run locally:
    pip install streamlit py3Dmol
    streamlit run app.py
"""

import streamlit as st
import streamlit.components.v1 as components
import py3Dmol
from pathlib import Path
from cif_to_xyz import cif_to_xyz

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

                st.session_state["xyz_text"] = xyz_text
                st.session_state["n_atoms"]  = n_atoms
                st.session_state["formula"]  = formula
                st.session_state["stem"]     = tmp_cif.stem

            except Exception as e:
                st.error(f"Conversion failed: {e}")
                st.exception(e)

# ── Results ───────────────────────────────────────────────────────────────────

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
    st.subheader("3D structure")

    viz_style = st.radio(
        "Display style",
        ["stick", "ball_and_stick", "sphere"],
        horizontal=True,
        format_func=lambda s: s.replace("_", " "),
    )

    view = py3Dmol.view(width=600, height=450)
    view.addModel(xyz_text, "xyz")

    if viz_style == "stick":
        view.setStyle({"stick": {"colorscheme": "Jmol", "radius": 0.15}})
    elif viz_style == "ball_and_stick":
        view.setStyle({
            "stick":  {"colorscheme": "Jmol", "radius": 0.10},
            "sphere": {"colorscheme": "Jmol", "scale": 0.35},
        })
    elif viz_style == "sphere":
        view.setStyle({"sphere": {"colorscheme": "Jmol", "scale": 0.45}})

    view.setBackgroundColor("0xffffff")
    view.zoomTo()

    # Render via st.components — no stmol needed
    components.html(view._make_html(), height=450, scrolling=False)

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
