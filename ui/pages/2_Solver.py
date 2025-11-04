"""
Solver Page - Run Simplex Algorithm and Display Results
"""

import streamlit as st
import sys
from pathlib import Path
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from vibe_simplex.solver import SimplexSolver, SimplexDebugger
from utils.theme import apply_custom_theme
from utils.state import initialize_session_state, save_problem_to_history

# Page config
st.set_page_config(page_title="Solver", page_icon="🚀", layout="wide")

# Initialize
initialize_session_state()
apply_custom_theme()

# Header
st.title("🚀 Simplex Solver")
st.markdown("Run the simplex algorithm and view comprehensive results")
st.markdown("---")

st.markdown(
    """
    <div class='custom-card' style='text-align:center;'>
        <div style='font-size:1.25rem;font-weight:600;'>Instead of Solve, try to run step by step.</div>
        <div style='font-size:0.95rem;color:var(--text-secondary);margin-top:0.35rem;'>
            Keep "Enable Step Recording" switched on to replay each pivot in the Debugger after you load a problem.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("---")

# Check if problem exists
if st.session_state.current_problem is None:
    st.warning("⚠️ No problem defined. Please go to **Problem Input** to create a problem first.")
    st.stop()

lp = st.session_state.current_problem

# Display problem summary
st.markdown("### Current Problem")

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("**Objective Function:**")
    obj_text = f"{lp.sense.capitalize()}imize: Z = " + " + ".join(
        [f"{lp.objective[i]:.2f}x{i+1}" for i in range(lp.num_variables)]
    )
    st.code(obj_text, language="text")

    st.markdown("**Constraints:**")
    for idx, constraint in enumerate(lp.constraints):
        constraint_text = " + ".join(
            [f"{constraint.coefficients[i]:.2f}x{i+1}" for i in range(lp.num_variables)]
        )
        st.code(f"{constraint_text} {constraint.sense} {constraint.rhs:.2f}", language="text")

with col2:
    st.metric("Variables", lp.num_variables)
    st.metric("Constraints", lp.num_constraints)
    st.metric("Objective", lp.sense.upper())

st.markdown("---")

# Solver controls
st.markdown("### Solver Configuration")

col1, col2, col3 = st.columns(3)

with col1:
    tolerance = st.number_input(
        "Tolerance",
        min_value=1e-12,
        max_value=1e-3,
        value=1e-9,
        format="%.2e",
        help="Numerical tolerance for the solver",
    )

with col2:
    max_iterations = st.number_input(
        "Max Iterations",
        min_value=100,
        max_value=100000,
        value=10000,
        step=100,
        help="Maximum number of simplex iterations",
    )

with col3:
    enable_debugger = st.checkbox(
        "Enable Step Recording",
        value=True,
        help="Record each iteration for step-by-step analysis",
    )

st.markdown("---")

# Run solver button
col1, col2, col3 = st.columns([1, 1, 1])

with col2:
    run_solver = st.button("▶️ Run Solver", type="primary", use_container_width=True)

if run_solver:
    with st.spinner("🔄 Running simplex algorithm..."):
        try:
            # Initialize solver
            solver = SimplexSolver(tolerance=tolerance, max_iterations=max_iterations)

            # Initialize debugger if enabled
            debugger = SimplexDebugger() if enable_debugger else None

            # Solve
            result = solver.solve(lp, debugger=debugger)

            # Save results
            st.session_state.current_result = result
            st.session_state.debugger = debugger

            # Save to history
            save_problem_to_history(lp, result)

            st.success("✅ Solver completed successfully!")

        except Exception as e:
            st.error(f"❌ Error running solver: {str(e)}")
            st.stop()

# Display results if available
if st.session_state.current_result is not None:
    result = st.session_state.current_result

    st.markdown("---")
    st.markdown("### Solution Results")

    # Status banner
    if result.status == "optimal":
        st.success(f"✅ {result.message}")
    elif result.status == "unbounded":
        st.warning(f"⚠️ {result.message}")
    else:
        st.error(f"❌ {result.message}")

    # Metrics
    if result.status == "optimal":
        st.markdown("#### Key Metrics")

        metrics_cols = st.columns(4)

        with metrics_cols[0]:
            st.metric(
                "Optimal Value",
                f"{result.objective_value:.4f}",
                help="The optimal value of the objective function",
            )

        with metrics_cols[1]:
            st.metric(
                "Iterations",
                len(result.steps) - 1,
                help="Number of simplex iterations performed",
            )

        with metrics_cols[2]:
            num_basic = sum(1 for v in result.variable_values.values() if v > tolerance)
            st.metric(
                "Basic Variables",
                num_basic,
                help="Number of non-zero variables in the solution",
            )

        with metrics_cols[3]:
            num_tight = sum(1 for s in result.slack_values.values() if s < tolerance)
            st.metric(
                "Tight Constraints",
                num_tight,
                help="Number of constraints at their limits",
            )

        st.markdown("---")

        # Variable values
        st.markdown("#### Decision Variables")

        var_data = []
        for var_name, value in result.variable_values.items():
            var_data.append({
                "Variable": var_name,
                "Value": f"{value:.6f}",
                "Status": "Basic" if value > tolerance else "Non-basic",
            })

        var_df = pd.DataFrame(var_data)
        st.dataframe(
            var_df,
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("---")

        # Slack variables
        st.markdown("#### Slack Variables")

        slack_data = []
        for slack_name, value in result.slack_values.items():
            constraint_idx = int(slack_name[1:]) - 1
            slack_data.append({
                "Slack Variable": slack_name,
                "Value": f"{value:.6f}",
                "Constraint": f"Constraint {constraint_idx + 1}",
                "Status": "Tight (binding)" if value < tolerance else "Loose (non-binding)",
            })

        slack_df = pd.DataFrame(slack_data)
        st.dataframe(
            slack_df,
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("---")

        # Solution interpretation
        st.markdown("#### Solution Interpretation")

        with st.expander("📊 What does this solution mean?", expanded=True):
            st.markdown(f"""
            The simplex algorithm found an **optimal solution** after **{len(result.steps) - 1} iterations**.

            **Optimal Objective Value:** {result.objective_value:.4f}
            - This is the best possible value for your objective function given the constraints.

            **Decision Variable Values:**
            """)

            for var_name, value in result.variable_values.items():
                st.markdown(f"- **{var_name}** = {value:.6f}")

            st.markdown(f"""

            **Constraint Analysis:**
            """)

            for idx, (slack_name, value) in enumerate(result.slack_values.items()):
                constraint_idx = int(slack_name[1:]) - 1
                if value < tolerance:
                    st.markdown(f"- **Constraint {constraint_idx + 1}** is **binding** (at its limit)")
                else:
                    st.markdown(f"- **Constraint {constraint_idx + 1}** has **{value:.4f} slack** (not at limit)")

    else:
        # Unbounded or infeasible
        st.markdown("#### Problem Status")
        st.markdown(f"**Status:** {result.status.upper()}")
        st.markdown(f"**Message:** {result.message}")

        if result.status == "unbounded":
            st.info("""
            💡 **What does unbounded mean?**

            The problem is unbounded, meaning the objective function can be increased indefinitely
            without violating any constraints. This usually indicates:
            - Missing constraints in the problem formulation
            - Incorrect constraint definitions
            - A modeling error in the original problem

            **Recommendation:** Review your constraints and ensure the problem is properly bounded.
            """)

    # Navigation hints
    st.markdown("---")
    st.info("""
    💡 **Next Steps:**
    - View **Analysis** page for sensitivity and dual analysis
    - View **Debugger** page for step-by-step iteration details
    - Navigate to **Problem Input** to solve a different problem
    """)

else:
    st.info("👆 Click **Run Solver** above to solve the current problem")
