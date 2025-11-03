"""
Examples Page - Pre-built LP Problems Library
"""

import streamlit as st
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from vibe_simplex.models import LinearProgram, Constraint
from utils.theme import apply_custom_theme
from utils.state import initialize_session_state

# Page config
st.set_page_config(page_title="Examples", page_icon="📚", layout="wide")

# Initialize
initialize_session_state()
apply_custom_theme()

# Header
st.title("📚 Example Problems Library")
st.markdown("Pre-built linear programming problems to explore and learn")
st.markdown("---")

# Example problems database
EXAMPLES = {
    "Simple 2D Problem": {
        "description": "A basic 2-variable problem perfect for visualization",
        "category": "Basic",
        "objective": [3.0, 5.0],
        "constraints": [
            Constraint([2.0, 3.0], 8.0, "<="),
            Constraint([1.0, 1.0], 4.0, "<="),
        ],
        "sense": "max",
        "context": """
        **Problem Context:**
        A company produces two products (x₁ and x₂) with the following characteristics:
        - Product 1 contributes $3 profit per unit
        - Product 2 contributes $5 profit per unit

        **Constraints:**
        - Resource 1: 2x₁ + 3x₂ ≤ 8 (limited resource availability)
        - Resource 2: x₁ + x₂ ≤ 4 (production capacity)

        **Goal:** Maximize total profit
        """,
        "expected_solution": "Optimal: Z = 13.33, x₁ = 0, x₂ = 2.67",
    },
    "Production Planning": {
        "description": "Classic production planning with multiple constraints",
        "category": "Classic",
        "objective": [40.0, 30.0],
        "constraints": [
            Constraint([1.0, 1.0], 12.0, "<="),  # Labor hours
            Constraint([2.0, 1.0], 16.0, "<="),  # Machine hours
            Constraint([1.0, 0.0], 8.0, "<="),   # Product 1 limit
        ],
        "sense": "max",
        "context": """
        **Problem Context:**
        A factory produces two types of products with different profit margins:
        - Product A: $40 profit per unit
        - Product B: $30 profit per unit

        **Constraints:**
        - Labor hours: 1 hour for A, 1 hour for B, max 12 hours available
        - Machine hours: 2 hours for A, 1 hour for B, max 16 hours available
        - Product A has a maximum demand of 8 units

        **Goal:** Maximize profit from production
        """,
        "expected_solution": "Optimal: Z = 400, x₁ = 8, x₂ = 0",
    },
    "Resource Allocation": {
        "description": "Allocate limited resources optimally",
        "category": "Classic",
        "objective": [5.0, 4.0],
        "constraints": [
            Constraint([6.0, 4.0], 24.0, "<="),
            Constraint([1.0, 2.0], 6.0, "<="),
            Constraint([-1.0, 1.0], 1.0, "<="),
        ],
        "sense": "max",
        "context": """
        **Problem Context:**
        Optimize resource allocation between two activities:
        - Activity 1: Returns 5 units of value
        - Activity 2: Returns 4 units of value

        **Constraints:**
        - Resource A: 6x₁ + 4x₂ ≤ 24
        - Resource B: x₁ + 2x₂ ≤ 6
        - Balance constraint: -x₁ + x₂ ≤ 1

        **Goal:** Maximize total value
        """,
        "expected_solution": "Optimal: Z = 21, x₁ = 3, x₂ = 1.5",
    },
    "Diet Problem": {
        "description": "Minimize cost while meeting nutritional requirements",
        "category": "Classic",
        "objective": [0.6, 1.0],
        "constraints": [
            Constraint([2.0, 1.0], 4.0, "<="),
            Constraint([1.0, 3.0], 6.0, "<="),
        ],
        "sense": "max",
        "context": """
        **Problem Context:**
        Select food items to maximize nutrition within budget:
        - Food A: 0.6 nutrition units per dollar
        - Food B: 1.0 nutrition units per dollar

        **Constraints:**
        - Constraint 1: 2x₁ + x₂ ≤ 4
        - Constraint 2: x₁ + 3x₂ ≤ 6

        **Goal:** Maximize nutrition (simplified from minimization)

        *Note: This is a maximization version of the classic diet problem*
        """,
        "expected_solution": "Optimal: Z = 2.4, x₁ = 0, x₂ = 2",
    },
    "Tight Constraints": {
        "description": "Problem with multiple binding constraints",
        "category": "Educational",
        "objective": [1.0, 1.0],
        "constraints": [
            Constraint([1.0, 0.0], 4.0, "<="),
            Constraint([0.0, 1.0], 6.0, "<="),
            Constraint([1.0, 1.0], 8.0, "<="),
        ],
        "sense": "max",
        "context": """
        **Problem Context:**
        Educational example showing constraint interactions:
        - Equal contribution from both variables to objective

        **Constraints:**
        - x₁ ≤ 4
        - x₂ ≤ 6
        - x₁ + x₂ ≤ 8

        **Learning Goal:**
        - Observe which constraints are binding at optimum
        - Understand shadow prices for different constraints
        """,
        "expected_solution": "Optimal: Z = 8, x₁ = 4, x₂ = 4",
    },
    "Corner Point Example": {
        "description": "Solution at a specific corner of feasible region",
        "category": "Educational",
        "objective": [2.0, 3.0],
        "constraints": [
            Constraint([1.0, 2.0], 10.0, "<="),
            Constraint([2.0, 1.0], 10.0, "<="),
        ],
        "sense": "max",
        "context": """
        **Problem Context:**
        Simple problem to illustrate corner point solutions:
        - Objective: Maximize 2x₁ + 3x₂

        **Constraints:**
        - x₁ + 2x₂ ≤ 10
        - 2x₁ + x₂ ≤ 10

        **Learning Goal:**
        - Visualize how the optimal solution occurs at a corner point
        - See the iso-profit line tangent to the feasible region
        """,
        "expected_solution": "Optimal: Z = 16.67, x₁ = 1.67, x₂ = 5",
    },
}

# Category filter
st.markdown("### Browse Examples")

categories = sorted(set(ex["category"] for ex in EXAMPLES.values()))
selected_category = st.selectbox(
    "Filter by Category",
    options=["All"] + categories,
    index=0,
)

# Display examples
filtered_examples = {
    name: details
    for name, details in EXAMPLES.items()
    if selected_category == "All" or details["category"] == selected_category
}

st.markdown(f"**Showing {len(filtered_examples)} example(s)**")
st.markdown("---")

# Display each example
for example_name, details in filtered_examples.items():
    with st.container():
        # Header
        col1, col2 = st.columns([3, 1])

        with col1:
            st.markdown(f"### {example_name}")
            st.markdown(f"*{details['description']}*")

        with col2:
            st.markdown(f"**Category:** {details['category']}")

        # Expandable details
        with st.expander("📖 View Details", expanded=False):
            # Context
            st.markdown(details["context"])

            # Mathematical formulation
            st.markdown("#### Mathematical Formulation")

            obj_text = f"{details['sense'].capitalize()}imize: Z = " + " + ".join(
                [f"{details['objective'][i]:.1f}x{i+1}" for i in range(len(details['objective']))]
            )
            st.code(obj_text, language="text")

            st.markdown("**Subject to:**")
            for idx, constraint in enumerate(details["constraints"]):
                constraint_text = " + ".join(
                    [f"{constraint.coefficients[i]:.1f}x{i+1}" for i in range(len(constraint.coefficients))]
                )
                st.code(f"{constraint_text} {constraint.sense} {constraint.rhs:.1f}", language="text")

            st.code("x₁, x₂ ≥ 0 (non-negativity)", language="text")

            # Expected solution
            st.markdown("#### Expected Solution")
            st.info(f"📊 {details['expected_solution']}")

        # Load button
        if st.button(f"📥 Load '{example_name}'", key=f"load_{example_name}", use_container_width=False):
            try:
                # Create LinearProgram
                lp = LinearProgram(
                    objective=details["objective"],
                    constraints=details["constraints"],
                    sense=details["sense"],
                )

                # Save to session state
                st.session_state.current_problem = lp
                st.session_state.current_result = None
                st.session_state.debugger = None

                # Update form state
                st.session_state.num_variables = len(details["objective"])
                st.session_state.num_constraints = len(details["constraints"])
                st.session_state.objective_sense = details["sense"]

                st.success(f"✅ Loaded '{example_name}' successfully!")
                st.balloons()

                st.info("💡 Navigate to **Solver** page to run this problem, or **Problem Input** to modify it.")

            except Exception as e:
                st.error(f"❌ Error loading example: {str(e)}")

        st.markdown("---")

# Current problem status
if st.session_state.current_problem is not None:
    st.markdown("### Currently Loaded Problem")

    lp = st.session_state.current_problem

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Variables", lp.num_variables)

    with col2:
        st.metric("Constraints", lp.num_constraints)

    with col3:
        st.metric("Objective", lp.sense.upper())

    st.success("✅ A problem is currently loaded and ready to solve!")

# Footer with tips
st.markdown("---")
st.markdown("""
### 💡 Tips for Learning

- **Start with Simple 2D Problem** to see visualization
- **Try Production Planning** for a classic application
- **Explore Tight Constraints** to understand shadow prices
- **Compare different examples** to see how constraints affect solutions

After loading an example:
1. Go to **Solver** to find the solution
2. Check **Analysis** for sensitivity and dual information
3. Use **Debugger** to see step-by-step iterations
4. View **Visualization** for 2D problems
""")
