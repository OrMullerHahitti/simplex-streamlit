"""
Vibe Simplex landing page with education-friendly UX guidance.
"""

import streamlit as st
import sys
from pathlib import Path

# Add parent directory to path to import vibe_simplex
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.theme import apply_custom_theme, get_theme_toggle
from utils.state import initialize_session_state
from utils.gallery import load_problem_gallery

st.set_page_config(
    page_title="Vibe Simplex - LP Solver",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

initialize_session_state()
apply_custom_theme()

with st.sidebar:
    get_theme_toggle()
    st.markdown("---")
    st.markdown(
        """
        ### Navigation
        - 📝 **Problem Input**
        - 🚀 **Solver**
        - 📊 **Analysis**
        - 🔍 **Debugger**
        - 📚 **Examples**
        """
    )

try:
    gallery = load_problem_gallery()
except FileNotFoundError:
    gallery = []

example_count = len(gallery)
category_count = len({p.category for p in gallery})
beginner_count = sum(1 for p in gallery if p.difficulty.lower() == "beginner")
two_var_ready = sum(1 for p in gallery if len(p.objective) == 2)

st.markdown(
    """
    <div style='text-align:center;padding:2rem 0;'>
        <h1 style='margin-bottom:0.6rem;'>📊 Vibe Simplex Studio</h1>
        <p style='font-size:1.1rem;color:var(--text-secondary);'>
            A classroom-ready workspace for modelling, solving, and explaining linear programs.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

metric_cols = st.columns(4)
metric_cols[0].metric("Example Library", example_count)
metric_cols[1].metric("Learning Tracks", category_count)
metric_cols[2].metric("Beginner Friendly", beginner_count)
metric_cols[3].metric("2D Visual-Ready", two_var_ready)

st.divider()

intro_cols = st.columns([2, 1])
with intro_cols[0]:
    st.subheader("Why Vibe Simplex?")
    st.markdown(
        """
        - **Model** LP problems with a friendly form builder (objectives, ≤/≥/= constraints).
        - **Solve & debug** every iteration via the Simplex debugger and tableau snapshots.
        - **Explain** outcomes with sensitivity, dual analysis, and ready-to-share visuals.
        - **Learn faster** using the curated gallery of starter, classic, and advanced problems.
        """
    )
with intro_cols[1]:
    st.success(
        """
        Tip: Keep “Enable Step Recording” on in the Solver. It powers the Debugger page, 
        lets you replay pivots, and makes live instruction effortless.
        """
    )

st.divider()

st.subheader("🧭 Quick Tour")
tour_cols = st.columns(3)

tour_cols[0].markdown(
    """
    **1 · Define**
    - Head to *Problem Input*
    - Set variables & constraints
    - Capture the LP in session state
    """
)

tour_cols[1].markdown(
    """
    **2 · Solve**
    - Run Simplex with tolerances you trust
    - Watch feasibility, optimality, or warnings
    - Store results for downstream pages
    """
)

tour_cols[2].markdown(
    """
    **3 · Explore**
    - Visualize 2D regions when applicable
    - Inspect dual, sensitivity, and slack data
    - Rewind steps in the Debugger for storytelling
    """
)

st.divider()

st.subheader("📌 Learning Paths")
learn_tabs = st.tabs(["Solve", "Analyze", "Teach"])

with learn_tabs[0]:
    st.markdown(
        """
        - Start with a gallery example or your own coefficients.
        - Keep an eye on constraint previews to avoid negative RHS slips.
        - Use success/warning banners as checkpoints before moving on.
        """
    )
with learn_tabs[1]:
    st.markdown(
        """
        - Sensitivity tables break down shadow prices and reduced costs in plain English.
        - Complementary slackness checks confirm optimality conditions per constraint.
        - 2D plots highlight binding constraints and optimal iso-profit lines.
        """
    )
with learn_tabs[2]:
    st.markdown(
        """
        - Load a curated example, run the solver, then share the Debugger slider live.
        - Use the insights list inside each example as a ready-made speaking outline.
        - Contribute new scenarios by editing `ui/assets/problem_gallery.json`—data drives the UI.
        """
    )

st.divider()

if gallery:
    st.subheader("✨ Featured Classroom Examples")
    for problem in gallery[:3]:
        cols = st.columns([3, 1])
        cols[0].markdown(f"**{problem.name}** — {problem.description}")
        tags = ", ".join(problem.tags)
        cols[0].caption(f"Tags: {tags}")
        cols[1].markdown(f"Category: **{problem.category}**")
        cols[1].markdown(f"Difficulty: **{problem.difficulty}**")
    st.caption("Browse the Examples page to load any of these instantly.")
else:
    st.info("Add entries to `ui/assets/problem_gallery.json` to populate the featured list.")

st.divider()

st.subheader("🚀 Quick Start Checklist")
st.markdown(
    """
    1. Open **Problem Input** → configure and create your LP.
    2. Head to **Solver** → run Simplex with step recording enabled.
    3. Visit **Debugger** → scrub iterations when teaching or reviewing.
    4. Inspect **Analysis** → pull shadow prices, reduced costs, and RHS ranges.
    5. Keep the **Examples** gallery handy for inspiration and reproducible demos.
    """
)

with st.expander("Need a concrete example?", expanded=False):
    st.markdown(
        """
        **Maximize** `Z = 3x₁ + 5x₂`

        Subject to:
        - `2x₁ + 3x₂ ≤ 8`
        - `x₁ + x₂ ≤ 4`
        - `x₁, x₂ ≥ 0`

        Load “Simple 2D Problem” from the gallery to see the solver, debugger, and 2D view in action.
        """
    )

st.divider()

st.subheader("📬 Keep the gallery fresh")
st.markdown(
    """
    - **Open an issue** describing a real-world scenario or teaching goal.
    - **Edit the gallery JSON** to add objectives, constraints, tags, and insights.
    - Once merged, every page automatically reflects the new content—no extra wiring.
    """
)

st.markdown(
    """
    <div style='text-align:center;color:var(--text-secondary);padding:1rem 0;'>
        Built with ❤️ using Streamlit · Powered by the Vibe Simplex engine
    </div>
    """,
    unsafe_allow_html=True,
)
