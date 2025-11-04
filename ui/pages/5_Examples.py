"""
Examples Page - Education-focused LP gallery.
"""

import streamlit as st
import sys
from collections import Counter
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from utils.theme import apply_custom_theme
from utils.state import initialize_session_state
from utils.gallery import load_problem_gallery, ExampleProblem

# Page config
st.set_page_config(page_title="Examples", page_icon="📚", layout="wide")

# Initialize UI baseline
initialize_session_state()
apply_custom_theme()

# Header copy
st.title("📚 Learning Library")
st.caption("Hand-picked exercises for teaching, self-study, and live demos.")
st.divider()

# Load gallery entries
try:
    problems = load_problem_gallery()
except FileNotFoundError as exc:
    st.error(f"Cannot load gallery data: {exc}")
    st.stop()

# Quick stats for orientation
total = len(problems)
category_counts = Counter(p.category for p in problems)
beginner_ready = sum(1 for p in problems if p.difficulty.lower() == "beginner")
two_var_ready = sum(1 for p in problems if len(p.objective) == 2)

metric_cols = st.columns(4)
metric_cols[0].metric("Examples", total)
metric_cols[1].metric("Categories", len(category_counts))
metric_cols[2].metric("Beginner Ready", beginner_ready)
metric_cols[3].metric("2D Visual-Ready", two_var_ready)

st.divider()

st.subheader("🎯 Find the right exercise")
st.markdown("Mix search, filters, and quick toggles to narrow the library.")

categories = sorted(category_counts.keys())
all_tags = sorted({tag for problem in problems for tag in problem.tags})
all_difficulties = sorted({p.difficulty for p in problems})

filter_row1 = st.columns([2, 1])
with filter_row1[0]:
    search_term = st.text_input(
        "Search by title, description, or context",
        placeholder="e.g. visualization, slack, production planning",
    ).strip().lower()
with filter_row1[1]:
    tag_filter = st.multiselect("Focus tags", options=all_tags)

filter_row2 = st.columns(3)
with filter_row2[0]:
    category_filter = st.multiselect("Categories", categories, default=categories)
with filter_row2[1]:
    difficulty_filter = st.multiselect(
        "Difficulty",
        options=all_difficulties,
        default=all_difficulties,
    )
with filter_row2[2]:
    two_var_only = st.toggle("Show only 2-variable problems", value=False, key="two_var_only")

beginner_only = st.toggle("Highlight beginner-friendly only", value=False, key="beginner_only")


def matches(problem: ExampleProblem) -> bool:
    """Apply search and filter logic."""

    if search_term:
        haystack = f"{problem.name} {problem.description} {problem.context}".lower()
        if search_term not in haystack:
            return False

    if category_filter and problem.category not in category_filter:
        return False

    if difficulty_filter and problem.difficulty not in difficulty_filter:
        return False

    if tag_filter and not set(tag_filter).issubset(problem.tags):
        return False

    if two_var_only and len(problem.objective) != 2:
        return False

    if beginner_only and problem.difficulty.lower() != "beginner":
        return False

    return True


filtered = [p for p in problems if matches(p)]

if not filtered:
    st.info("No examples match those filters yet. Widen your search or clear a toggle.")
else:
    st.markdown(f"**Showing {len(filtered)} example(s).**")

st.divider()


def render_tag_badge(tag: str) -> str:
    return (
        "<span style='background:var(--background-tertiary);padding:0.25rem 0.65rem;"
        "border-radius:999px;font-size:0.8rem;margin-right:0.3rem;'>"
        f"{tag}</span>"
    )


for problem in filtered:
    with st.container():
        header_cols = st.columns([3, 1])

        with header_cols[0]:
            st.markdown(f"### {problem.name}")
            st.markdown(problem.description)
            tags_html = "".join(render_tag_badge(tag) for tag in problem.tags)
            if tags_html:
                st.markdown(tags_html, unsafe_allow_html=True)

        with header_cols[1]:
            st.markdown(f"**Category:** {problem.category}")
            st.markdown(f"**Difficulty:** {problem.difficulty}")
            st.markdown(f"**Variables:** {len(problem.objective)}")

        insight_col1, insight_col2 = st.columns([3, 1])
        with insight_col1:
            if problem.insights:
                st.markdown("**Why use this example?**")
                for note in problem.insights:
                    st.markdown(f"- {note}")
        with insight_col2:
            if st.button(f"Load {problem.name}", key=f"load_{problem.slug}"):
                try:
                    lp = problem.to_linear_program()
                    st.session_state.current_problem = lp
                    st.session_state.current_result = None
                    st.session_state.debugger = None
                    st.session_state.num_variables = lp.num_variables
                    st.session_state.num_constraints = lp.num_constraints
                    st.session_state.objective_sense = lp.sense
                    st.success("Problem loaded. Jump to the Solver when ready!")
                except Exception as exc:  # pragma: no cover
                    st.error(f"Unable to load problem: {exc}")

        with st.expander("Full context & formulation", expanded=False):
            st.markdown(problem.context)
            st.markdown("#### Mathematical Formulation")
            obj_text = " + ".join(
                f"{problem.objective[i]:.2f}x{i+1}" for i in range(len(problem.objective))
            )
            st.code(f"{problem.sense.capitalize()}imize: Z = {obj_text}")

            st.markdown("**Subject to:**")
            for constraint in problem.constraints:
                coeffs = " + ".join(
                    f"{constraint.coefficients[i]:.2f}x{i+1}"
                    for i in range(len(constraint.coefficients))
                )
                st.code(f"{coeffs} {constraint.sense} {constraint.rhs:.2f}")

            st.code("xᵢ ≥ 0 (non-negativity)")
            st.markdown("#### Expected Solution")
            st.info(problem.expected_solution)

        st.divider()

# Contribute instructions
st.subheader("📬 Add your own teaching example")
st.markdown(
    """
    - **Open an issue** describing the learning goal and formulation, or
    - **Edit `ui/assets/problem_gallery.json`** directly in GitHub and append an entry using this template:
    """
)

st.code(
    """{
  "slug": "unique-id",
  "name": "Problem title",
  "description": "One-liner",
  "category": "Starter | Operations | Classic | Educational",
  "difficulty": "Beginner | Intermediate | Advanced",
  "tags": ["visualization", "shadow-prices"],
  "sense": "max|min",
  "objective": [3.0, 5.0],
  "constraints": [{"coefficients": [2.0, 3.0], "rhs": 8.0, "sense": "<="}],
  "context": "Markdown-friendly narrative",
  "expected_solution": "Optimal Z = …",
  "insights": ["Key teaching note"]
}
""",
    language="json",
)

st.markdown(
    """
    Once merged, the new example will appear here automatically and becomes available to the Solver,
    Debugger, and Analysis pages—no extra wiring required.
    """
)
