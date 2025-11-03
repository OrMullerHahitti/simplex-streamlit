"""
Vibe Simplex - Professional Linear Programming Solver
Main landing page with branding and quick start guide
"""

import streamlit as st
import sys
from pathlib import Path

# Add parent directory to path to import vibe_simplex
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.theme import apply_custom_theme, get_theme_toggle
from utils.state import initialize_session_state

# Page configuration
st.set_page_config(
    page_title="Vibe Simplex - LP Solver",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize session state and apply theme
initialize_session_state()
apply_custom_theme()

# Header with branding
st.markdown(
    """
    <div style='text-align: center; padding: 2rem 0;'>
        <h1 style='font-size: 3rem; margin-bottom: 0.5rem;'>
            📊 Vibe Simplex
        </h1>
        <p style='font-size: 1.2rem; color: var(--text-secondary); margin-top: 0;'>
            Professional Linear Programming Solver
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Theme toggle in sidebar
with st.sidebar:
    get_theme_toggle()
    st.markdown("---")

    st.markdown("### Navigation")
    st.markdown("""
    - 📝 **Problem Input** - Define your LP problem
    - 🚀 **Solver** - Run and visualize the solution
    - 📊 **Analysis** - Sensitivity & dual analysis
    - 🔍 **Debugger** - Step-by-step exploration
    - 📚 **Examples** - Pre-built LP problems
    """)

# Main content - Quick start guide
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.markdown("## Welcome to Vibe Simplex")

    st.markdown("""
    A powerful, interactive tool for solving and understanding linear programming problems
    using the **Simplex Method**.
    """)

    st.markdown("### ✨ Key Features")

    features_col1, features_col2 = st.columns(2)

    with features_col1:
        st.markdown("""
        **🎯 Core Functionality**
        - Solve linear programming problems
        - Step-by-step execution tracking
        - Optimal solution identification
        - Unbounded/infeasible detection

        **📊 Visualization**
        - Interactive 2D constraint plots
        - Feasible region highlighting
        - Iso-profit line animation
        - Optimal point identification
        """)

    with features_col2:
        st.markdown("""
        **🔬 Advanced Analysis**
        - Sensitivity analysis
        - Shadow prices & reduced costs
        - Allowable ranges
        - Dual problem analysis

        **💼 Professional Tools**
        - Export to PDF/Excel
        - Problem history
        - Dark/Light themes
        - Example problem library
        """)

    st.markdown("---")

    st.markdown("### 🚀 Quick Start")

    st.markdown("""
    1. **Navigate to Problem Input** (sidebar) to define your LP problem
    2. **Specify objective function** (maximize or minimize)
    3. **Add constraints** (≤, ≥, or = constraints)
    4. **Run the solver** to find optimal solution
    5. **Explore results** with visualizations and analysis
    """)

    st.markdown("---")

    st.markdown("### 📖 Example Problem")

    with st.expander("Click to see a sample problem", expanded=False):
        st.markdown("""
        **Problem**: Maximize profit from producing two products

        **Objective Function:**
        ```
        Maximize: Z = 3x₁ + 5x₂
        ```

        **Constraints:**
        ```
        2x₁ + 3x₂ ≤ 8   (Resource 1)
        x₁ + x₂ ≤ 4     (Resource 2)
        x₁, x₂ ≥ 0      (Non-negativity)
        ```

        **Solution:**
        - Optimal value: Z = 13.33
        - x₁ = 0.0, x₂ = 2.67

        Try this in the **Examples** page or enter it manually in **Problem Input**!
        """)

    st.markdown("---")

    # Call to action
    st.markdown("""
    <div style='text-align: center; padding: 2rem 0;'>
        <p style='font-size: 1.1rem;'>
            Ready to get started? Use the navigation menu on the left to begin!
        </p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: var(--text-secondary); padding: 1rem 0;'>
    <p>Built with ❤️ using Streamlit | Powered by Vibe Simplex Engine</p>
</div>
""", unsafe_allow_html=True)
