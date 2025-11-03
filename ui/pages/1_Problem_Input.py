"""
Problem Input Page - Define Linear Programming Problems
"""

import streamlit as st
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from vibe_simplex.models import LinearProgram, Constraint
from utils.theme import apply_custom_theme
from utils.state import initialize_session_state, clear_current_problem

# Page config
st.set_page_config(page_title="Problem Input", page_icon="📝", layout="wide")

# Initialize
initialize_session_state()
apply_custom_theme()

# Header
st.title("📝 Problem Input")
st.markdown("Define your linear programming problem with an intuitive form interface")
st.markdown("---")

# Instructions
with st.expander("ℹ️ How to use this form", expanded=False):
    st.markdown("""
    ### Defining Your LP Problem

    1. **Choose objective sense**: Select whether you want to maximize or minimize
    2. **Set problem dimensions**: Specify number of decision variables and constraints
    3. **Define objective function**: Enter coefficients for each variable
    4. **Add constraints**: For each constraint, enter:
       - Coefficients for each variable
       - Constraint type (≤, ≥, or =)
       - Right-hand side value

    ### Important Notes
    - Current version supports **maximization** problems only
    - Constraints must be in **≤** form
    - All RHS values must be **non-negative**

    ### Example
    ```
    Maximize: 3x₁ + 5x₂
    Subject to:
        2x₁ + 3x₂ ≤ 8
        x₁ + x₂ ≤ 4
        x₁, x₂ ≥ 0
    ```
    """)

# Main form
st.markdown("### Problem Configuration")

col1, col2, col3 = st.columns(3)

with col1:
    objective_sense = st.selectbox(
        "Objective Sense",
        options=["max", "min"],
        index=0,
        help="Choose whether to maximize or minimize the objective function",
        key="objective_sense_select",
    )
    if objective_sense == "min":
        st.warning("⚠️ Note: Current version only supports maximization. Please convert your minimization problem.")

with col2:
    num_variables = st.number_input(
        "Number of Variables",
        min_value=2,
        max_value=10,
        value=st.session_state.num_variables,
        step=1,
        help="Number of decision variables (x₁, x₂, ...)",
    )
    st.session_state.num_variables = num_variables

with col3:
    num_constraints = st.number_input(
        "Number of Constraints",
        min_value=1,
        max_value=10,
        value=st.session_state.num_constraints,
        step=1,
        help="Number of constraint equations",
    )
    st.session_state.num_constraints = num_constraints

st.markdown("---")

# Objective function input
st.markdown("### Objective Function")
st.markdown(f"**{objective_sense.capitalize()}imize:** Z = " + " + ".join([f"c{i+1}·x{i+1}" for i in range(num_variables)]))

objective_cols = st.columns(num_variables)
objective_coeffs = []

for i in range(num_variables):
    with objective_cols[i]:
        coeff = st.number_input(
            f"c{i+1} (x{i+1})",
            value=0.0,
            step=0.1,
            format="%.2f",
            key=f"obj_coeff_{i}",
            help=f"Coefficient for variable x{i+1}",
        )
        objective_coeffs.append(coeff)

st.markdown("---")

# Constraints input
st.markdown("### Constraints")
st.markdown("Define each constraint in the form: a₁x₁ + a₂x₂ + ... {≤,≥,=} b")

constraints_data = []
valid_constraints = True

for c in range(num_constraints):
    st.markdown(f"#### Constraint {c+1}")

    cols = st.columns(num_variables + 3)

    # Coefficient inputs
    constraint_coeffs = []
    for i in range(num_variables):
        with cols[i]:
            coeff = st.number_input(
                f"x{i+1}",
                value=0.0,
                step=0.1,
                format="%.2f",
                key=f"constraint_{c}_var_{i}",
                help=f"Coefficient for x{i+1} in constraint {c+1}",
            )
            constraint_coeffs.append(coeff)

    # Constraint sense
    with cols[num_variables]:
        sense = st.selectbox(
            "Type",
            options=["<=", ">=", "="],
            index=0,
            key=f"constraint_{c}_sense",
            help="Constraint type",
        )
        if sense != "<=":
            st.warning("⚠️ Only ≤ supported")
            valid_constraints = False

    # RHS value
    with cols[num_variables + 1]:
        rhs = st.number_input(
            "RHS",
            value=0.0,
            step=0.1,
            format="%.2f",
            key=f"constraint_{c}_rhs",
            help="Right-hand side value",
        )
        if rhs < 0:
            st.error("❌ RHS must be ≥ 0")
            valid_constraints = False

    # Preview
    with cols[num_variables + 2]:
        st.markdown("**Preview:**")
        constraint_text = " + ".join([f"{constraint_coeffs[i]:.2f}x{i+1}" for i in range(num_variables)])
        st.code(f"{constraint_text} {sense} {rhs:.2f}", language="text")

    constraints_data.append({
        "coefficients": constraint_coeffs,
        "sense": sense,
        "rhs": rhs,
    })

    if c < num_constraints - 1:
        st.markdown("---")

st.markdown("---")

# Action buttons
st.markdown("### Actions")

col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("✅ Create Problem", type="primary", use_container_width=True):
        try:
            # Validate inputs
            if objective_sense != "max":
                st.error("❌ Only maximization problems are currently supported")
            elif not valid_constraints:
                st.error("❌ Please fix constraint validation errors above")
            elif sum(objective_coeffs) == 0:
                st.error("❌ Objective function cannot have all zero coefficients")
            else:
                # Create LinearProgram
                constraints = [
                    Constraint(
                        coefficients=c["coefficients"],
                        rhs=c["rhs"],
                        sense=c["sense"],
                    )
                    for c in constraints_data
                ]

                lp = LinearProgram(
                    objective=objective_coeffs,
                    constraints=constraints,
                    sense=objective_sense,
                )

                # Save to session state
                st.session_state.current_problem = lp
                st.session_state.current_result = None  # Clear previous results
                st.session_state.debugger = None

                st.success("✅ Problem created successfully!")
                st.balloons()

                # Show problem summary
                st.markdown("#### Problem Summary")
                st.markdown(f"**Objective:** {objective_sense.capitalize()}imize Z")
                st.code(
                    "Z = " + " + ".join([f"{objective_coeffs[i]:.2f}x{i+1}" for i in range(num_variables)]),
                    language="text"
                )
                st.markdown("**Subject to:**")
                for idx, c in enumerate(constraints_data):
                    constraint_text = " + ".join([f"{c['coefficients'][i]:.2f}x{i+1}" for i in range(num_variables)])
                    st.code(f"{constraint_text} {c['sense']} {c['rhs']:.2f}", language="text")

                st.info("💡 Navigate to **Solver** page to run the simplex algorithm!")

        except Exception as e:
            st.error(f"❌ Error creating problem: {str(e)}")

with col2:
    if st.button("🔄 Reset Form", use_container_width=True):
        clear_current_problem()
        st.rerun()

with col3:
    if st.button("📋 Load Example", use_container_width=True):
        st.info("💡 Navigate to **Examples** page to load pre-built problems")

with col4:
    if st.session_state.current_problem is not None:
        st.success("✓ Problem Ready")
    else:
        st.info("⧗ No Problem")

# Current problem status
if st.session_state.current_problem is not None:
    st.markdown("---")
    st.markdown("### Current Problem Status")

    lp = st.session_state.current_problem

    status_col1, status_col2, status_col3, status_col4 = st.columns(4)

    with status_col1:
        st.metric("Variables", lp.num_variables)

    with status_col2:
        st.metric("Constraints", lp.num_constraints)

    with status_col3:
        st.metric("Objective", lp.sense.upper())

    with status_col4:
        if st.session_state.current_result is not None:
            st.metric("Status", "Solved ✅")
        else:
            st.metric("Status", "Unsolved ⧗")
