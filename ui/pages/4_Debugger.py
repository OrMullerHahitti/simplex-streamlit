"""
Debugger Page - Step-by-Step Simplex Exploration
"""

import streamlit as st
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from utils.theme import apply_custom_theme, get_theme_colors
from utils.state import initialize_session_state

# Page config
st.set_page_config(page_title="Debugger", page_icon="🔍", layout="wide")

# Initialize
initialize_session_state()
apply_custom_theme()

# Header
st.title("🔍 Step-by-Step Debugger")
st.markdown("Navigate through each iteration of the simplex algorithm")
st.markdown("---")

# Check if debugger data exists
if st.session_state.debugger is None:
    st.warning("⚠️ No debugger data available. Please solve a problem with **Enable Step Recording** checked in the **Solver** page.")
    st.stop()

debugger = st.session_state.debugger
steps = debugger.steps()

if not steps:
    st.error("❌ No steps recorded")
    st.stop()

# Navigation controls
st.markdown("### Navigation")

col1, col2, col3, col4, col5 = st.columns([1, 1, 2, 1, 1])

with col1:
    if st.button("⏮️ First", use_container_width=True):
        st.session_state.debugger_cursor = 0

with col2:
    if st.button("◀️ Previous", use_container_width=True):
        st.session_state.debugger_cursor = max(0, st.session_state.debugger_cursor - 1)

with col3:
    current_step = st.slider(
        "Iteration",
        min_value=0,
        max_value=len(steps) - 1,
        value=st.session_state.debugger_cursor,
        key="step_slider",
    )
    st.session_state.debugger_cursor = current_step

with col4:
    if st.button("▶️ Next", use_container_width=True):
        st.session_state.debugger_cursor = min(len(steps) - 1, st.session_state.debugger_cursor + 1)

with col5:
    if st.button("⏭️ Last", use_container_width=True):
        st.session_state.debugger_cursor = len(steps) - 1

# Get current step
current_idx = st.session_state.debugger_cursor
step = steps[current_idx]

st.markdown("---")

# Step information
st.markdown(f"### Iteration {step.iteration}")

# Metrics
metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)

with metrics_col1:
    st.metric("Iteration", step.iteration)

with metrics_col2:
    st.metric("Objective Value", f"{step.objective_value:.6f}")

with metrics_col3:
    if step.entering_variable:
        st.metric("Entering Variable", step.entering_variable, delta="↗ Entering")
    else:
        st.metric("Entering Variable", "None", delta="✓ Optimal")

with metrics_col4:
    if step.leaving_variable:
        st.metric("Leaving Variable", step.leaving_variable, delta="↙ Leaving")
    else:
        st.metric("Leaving Variable", "None", delta="—")

st.markdown("---")

# Tableau display
st.markdown("### Simplex Tableau")

# Convert tableau to DataFrame
tableau_array = np.array(step.tableau)
num_rows, num_cols = tableau_array.shape

# Create column headers (variables + RHS)
num_vars = num_cols - 1
var_names = [f"x{i+1}" for i in range(num_vars)] + ["RHS"]

# Create row headers (basis + Z)
row_names = step.basis + ["Z"]

# Create DataFrame
tableau_df = pd.DataFrame(
    tableau_array,
    columns=var_names,
    index=row_names,
)

# Style the tableau
def style_tableau(val):
    """Apply color styling to tableau cells"""
    try:
        val_float = float(val)
        if abs(val_float) < 1e-9:
            return 'background-color: rgba(128, 128, 128, 0.1)'
        elif val_float > 0:
            return 'background-color: rgba(0, 212, 255, 0.1)'
        else:
            return 'background-color: rgba(255, 107, 107, 0.1)'
    except:
        return ''

# Highlight pivot if available
if step.pivot is not None:
    pivot_row, pivot_col = step.pivot

    st.markdown(f"""
    <div style='padding: 1rem; background-color: var(--background-secondary);
    border-left: 4px solid var(--warning); border-radius: 4px; margin-bottom: 1rem;'>
    <strong>Pivot Operation:</strong> Row {pivot_row + 1} (leaving: {step.leaving_variable}),
    Column {pivot_col + 1} (entering: {step.entering_variable})
    </div>
    """, unsafe_allow_html=True)

# Display tableau with formatting
st.dataframe(
    tableau_df.style.format("{:.6f}").applymap(style_tableau),
    use_container_width=True,
    height=min(400, (num_rows + 1) * 35 + 38),
)

st.markdown("---")

# Current basis information
st.markdown("### Current Basis")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Basic Variables:**")
    basis_info = []
    for idx, var_name in enumerate(step.basis):
        value = step.tableau[idx][-1]  # RHS value
        basis_info.append(f"- **{var_name}** = {value:.6f}")

    st.markdown("\n".join(basis_info))

with col2:
    st.markdown("**Objective Function:**")
    st.code(f"Z = {step.objective_value:.6f}", language="text")

    if step.entering_variable is None:
        st.success("✅ **Optimal solution reached!**")
    else:
        st.info(f"⏳ **Next**: {step.entering_variable} enters, {step.leaving_variable} leaves")

st.markdown("---")

# Iteration explanation
st.markdown("### Iteration Explanation")

with st.expander("📖 What's happening in this iteration?", expanded=True):
    if step.iteration == 0:
        st.markdown(f"""
        #### Initial Tableau (Iteration 0)

        This is the starting point of the simplex algorithm.

        **Basis:** {", ".join(step.basis)}
        - The initial basis typically consists of slack variables
        - The current solution is at the origin (all decision variables = 0)

        **Objective Value:** {step.objective_value:.6f}
        - This is the objective function value at the current basic feasible solution

        **Next Step:**
        - The algorithm will identify which non-basic variable should enter the basis
        - This is the variable with the most positive reduced cost (in maximization)
        """)

    elif step.entering_variable is None:
        st.markdown(f"""
        #### Final Iteration (Optimal Solution)

        The simplex algorithm has terminated successfully!

        **Why Optimal?**
        - All reduced costs (bottom row) are ≤ 0
        - No non-basic variable can improve the objective function
        - The current basis provides the optimal solution

        **Optimal Basis:** {", ".join(step.basis)}

        **Optimal Objective Value:** {step.objective_value:.6f}

        **Optimal Solution:**
        """)

        for idx, var_name in enumerate(step.basis):
            value = step.tableau[idx][-1]
            st.markdown(f"- **{var_name}** = {value:.6f}")

    else:
        st.markdown(f"""
        #### Iteration {step.iteration}

        **Current Basis:** {", ".join(step.basis)}

        **Entering Variable:** {step.entering_variable}
        - This variable has the most positive reduced cost
        - Bringing it into the basis will improve the objective function

        **Leaving Variable:** {step.leaving_variable}
        - This variable leaves the basis to maintain feasibility
        - Determined by the minimum ratio test

        **Pivot Operation:**
        - The pivot element is at row {step.pivot[0] if step.pivot else "?"}, column {step.pivot[1] if step.pivot else "?"}
        - Row operations will transform this element to 1 and other elements in its column to 0

        **Objective Value:** {step.objective_value:.6f}
        - The objective improves with each iteration (for non-degenerate problems)

        **Next Iteration:**
        - After pivoting, the basis will be updated
        - {step.entering_variable} will replace {step.leaving_variable}
        - The algorithm continues until optimality is reached
        """)

# Progress indicator
st.markdown("---")
st.markdown("### Progress")

progress_value = (step.iteration) / max(len(steps) - 1, 1)
st.progress(progress_value)

st.markdown(f"""
<div style='text-align: center; padding: 1rem;'>
    <strong>Iteration {step.iteration} of {len(steps) - 1}</strong>
    ({progress_value * 100:.1f}% complete)
</div>
""", unsafe_allow_html=True)

# Navigation hints
if current_idx < len(steps) - 1:
    st.info("💡 Use the navigation buttons or slider to move to the next iteration")
else:
    st.success("🎉 You've reached the final iteration! This is the optimal solution.")
