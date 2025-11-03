"""
Analysis Page - Sensitivity and Dual Analysis
"""

import streamlit as st
import sys
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from utils.theme import apply_custom_theme, get_theme_colors
from utils.state import initialize_session_state
from utils.visualization import create_2d_constraint_plot

# Page config
st.set_page_config(page_title="Analysis", page_icon="📊", layout="wide")

# Initialize
initialize_session_state()
apply_custom_theme()

# Header
st.title("📊 Sensitivity & Dual Analysis")
st.markdown("Comprehensive analysis of the optimal solution")
st.markdown("---")

# Check if results exist
if st.session_state.current_result is None:
    st.warning("⚠️ No solution available. Please solve a problem first in the **Solver** page.")
    st.stop()

if st.session_state.current_problem is None:
    st.warning("⚠️ No problem defined.")
    st.stop()

result = st.session_state.current_result
lp = st.session_state.current_problem

if result.status != "optimal":
    st.error(f"❌ Analysis only available for optimal solutions. Current status: {result.status}")
    st.stop()

# Tabs for different analyses
tab1, tab2, tab3 = st.tabs(["📈 Sensitivity Analysis", "🔄 Dual Analysis", "📉 Visualization"])

# ===== SENSITIVITY ANALYSIS =====
with tab1:
    st.markdown("### Sensitivity Analysis")

    st.markdown("""
    Sensitivity analysis examines how changes in problem parameters affect the optimal solution.
    """)

    # Shadow Prices
    st.markdown("#### Shadow Prices (Dual Variables)")

    st.markdown("""
    Shadow prices represent the rate of change in the objective function value
    with respect to a unit increase in the right-hand side of a constraint.
    """)

    shadow_data = []
    for constraint_name, shadow_price in result.sensitivity.shadow_prices.items():
        constraint_idx = int(constraint_name.split("_")[1])
        shadow_data.append({
            "Constraint": f"Constraint {constraint_idx}",
            "Shadow Price": f"{shadow_price:.6f}",
            "Interpretation": f"Objective changes by {shadow_price:.6f} per unit increase in RHS",
        })

    shadow_df = pd.DataFrame(shadow_data)
    st.dataframe(shadow_df, use_container_width=True, hide_index=True)

    with st.expander("ℹ️ Understanding Shadow Prices"):
        st.markdown("""
        - **Positive shadow price**: Increasing the RHS of this constraint would improve the objective
        - **Zero shadow price**: This constraint is not binding; changing its RHS won't affect the objective
        - **Magnitude**: Indicates how valuable an additional unit of the resource is
        """)

    st.markdown("---")

    # Reduced Costs
    st.markdown("#### Reduced Costs")

    st.markdown("""
    Reduced costs indicate how much the objective coefficient must improve
    before a non-basic variable enters the basis.
    """)

    reduced_data = []
    for var_name, reduced_cost in result.sensitivity.reduced_costs.items():
        is_basic = result.variable_values.get(var_name, 0.0) > 1e-9
        reduced_data.append({
            "Variable": var_name,
            "Reduced Cost": f"{reduced_cost:.6f}",
            "Status": "Basic" if is_basic else "Non-basic",
            "Interpretation": "In basis" if is_basic else f"Coefficient must improve by {abs(reduced_cost):.6f}",
        })

    reduced_df = pd.DataFrame(reduced_data)
    st.dataframe(reduced_df, use_container_width=True, hide_index=True)

    with st.expander("ℹ️ Understanding Reduced Costs"):
        st.markdown("""
        - **Zero reduced cost**: Variable is basic (in the optimal solution)
        - **Negative reduced cost**: Variable is non-basic; coefficient must increase to make it worthwhile
        - **Magnitude**: Amount the objective coefficient must change before variable becomes basic
        """)

    st.markdown("---")

    # RHS Ranges
    st.markdown("#### Right-Hand Side Sensitivity Ranges")

    st.markdown("""
    These ranges show how much the RHS of each constraint can change
    while maintaining the current basis (and shadow prices).
    """)

    ranges_data = []
    for constraint_name, ranges in result.sensitivity.rhs_ranges.items():
        constraint_idx = int(constraint_name.split("_")[1])
        current_rhs = lp.constraints[constraint_idx - 1].rhs

        allowable_increase = ranges.get("allowable_increase", float("inf"))
        allowable_decrease = ranges.get("allowable_decrease", float("inf"))

        ranges_data.append({
            "Constraint": f"Constraint {constraint_idx}",
            "Current RHS": f"{current_rhs:.4f}",
            "Allowable Increase": f"{allowable_increase:.4f}" if allowable_increase != float("inf") else "∞",
            "Allowable Decrease": f"{allowable_decrease:.4f}" if allowable_decrease != float("inf") else "∞",
            "Lower Bound": f"{current_rhs - allowable_decrease:.4f}" if allowable_decrease != float("inf") else "0",
            "Upper Bound": f"{current_rhs + allowable_increase:.4f}" if allowable_increase != float("inf") else "∞",
        })

    ranges_df = pd.DataFrame(ranges_data)
    st.dataframe(ranges_df, use_container_width=True, hide_index=True)

    # Visualize RHS ranges
    st.markdown("#### RHS Range Visualization")

    theme = st.session_state.get("theme", "dark")
    colors = get_theme_colors(theme)

    fig_ranges = go.Figure()

    for idx, row_data in enumerate(ranges_data):
        constraint_name = row_data["Constraint"]
        current = float(row_data["Current RHS"])
        lower = float(row_data["Lower Bound"]) if row_data["Lower Bound"] != "0" else 0
        upper = float(row_data["Upper Bound"]) if row_data["Upper Bound"] != "∞" else current * 2

        # Add range bar
        fig_ranges.add_trace(go.Bar(
            x=[upper - lower],
            y=[constraint_name],
            orientation='h',
            base=lower,
            name=constraint_name,
            marker=dict(color=colors["primary"], opacity=0.6),
            hovertemplate=f"{constraint_name}<br>Range: [{lower:.2f}, {upper:.2f}]<extra></extra>",
        ))

        # Add current value marker
        fig_ranges.add_trace(go.Scatter(
            x=[current],
            y=[constraint_name],
            mode='markers',
            name=f"{constraint_name} (current)",
            marker=dict(size=12, color=colors["warning"], symbol="diamond"),
            hovertemplate=f"Current: {current:.2f}<extra></extra>",
            showlegend=False,
        ))

    fig_ranges.update_layout(
        title="RHS Sensitivity Ranges",
        xaxis_title="RHS Value",
        yaxis_title="Constraint",
        plot_bgcolor=colors["background"],
        paper_bgcolor=colors["background"],
        font=dict(color=colors["text-primary"]),
        height=300,
        showlegend=False,
    )

    st.plotly_chart(fig_ranges, use_container_width=True)

# ===== DUAL ANALYSIS =====
with tab2:
    st.markdown("### Dual Problem Analysis")

    st.markdown("""
    The dual problem provides an alternative perspective on the linear program.
    By strong duality, the optimal dual objective equals the optimal primal objective.
    """)

    # Dual variables
    st.markdown("#### Dual Variables")

    dual_data = []
    for var_name, value in result.dual.dual_variables.items():
        dual_data.append({
            "Dual Variable": var_name,
            "Value": f"{value:.6f}",
            "Interpretation": f"Shadow price for constraint {var_name[1:]}",
        })

    dual_df = pd.DataFrame(dual_data)
    st.dataframe(dual_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    # Dual objective
    st.markdown("#### Dual Objective Value")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Primal Objective", f"{result.objective_value:.6f}")

    with col2:
        st.metric("Dual Objective", f"{result.dual.dual_objective_value:.6f}")

    with col3:
        duality_gap = abs(result.objective_value - result.dual.dual_objective_value)
        st.metric("Duality Gap", f"{duality_gap:.2e}")

    if duality_gap < 1e-6:
        st.success("✅ Strong duality holds: Primal and dual objectives are equal")
    else:
        st.warning(f"⚠️ Duality gap detected: {duality_gap:.6f}")

    st.markdown("---")

    # Complementary slackness
    st.markdown("#### Complementary Slackness Conditions")

    st.markdown("""
    For an optimal solution, complementary slackness requires:
    - If a primal constraint is not binding (slack > 0), its dual variable = 0
    - If a dual variable > 0, its corresponding primal constraint is binding
    """)

    cs_data = []
    for slack_name, is_satisfied in result.dual.complementary_slackness.items():
        constraint_idx = int(slack_name[1:])
        slack_value = result.slack_values.get(slack_name, 0.0)
        dual_var = result.dual.dual_variables.get(f"y{constraint_idx}", 0.0)

        cs_data.append({
            "Constraint": f"Constraint {constraint_idx}",
            "Slack Value": f"{slack_value:.6f}",
            "Dual Variable": f"{dual_var:.6f}",
            "Product": f"{slack_value * dual_var:.2e}",
            "CS Satisfied": "✅ Yes" if is_satisfied else "❌ No",
        })

    cs_df = pd.DataFrame(cs_data)
    st.dataframe(cs_df, use_container_width=True, hide_index=True)

    all_satisfied = all(result.dual.complementary_slackness.values())
    if all_satisfied:
        st.success("✅ All complementary slackness conditions are satisfied")
    else:
        st.error("❌ Some complementary slackness conditions are violated")

    with st.expander("ℹ️ Understanding Complementary Slackness"):
        st.markdown("""
        Complementary slackness is a key optimality condition stating:
        - **Primal slack · Dual variable = 0** for each constraint
        - Either the constraint is binding (slack = 0) OR the dual variable is 0

        This condition, along with primal and dual feasibility, guarantees optimality.
        """)

# ===== VISUALIZATION =====
with tab3:
    st.markdown("### 2D Constraint Visualization")

    if lp.num_variables == 2:
        st.markdown("""
        Interactive visualization showing constraints, feasible region, corner points,
        and the optimal solution.
        """)

        try:
            theme = st.session_state.get("theme", "dark")
            fig = create_2d_constraint_plot(lp, result, theme)
            st.plotly_chart(fig, use_container_width=True)

            # Additional insights
            st.markdown("#### Key Insights")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Optimal Point:**")
                st.code(
                    f"x₁ = {result.variable_values.get('x1', 0.0):.6f}\n"
                    f"x₂ = {result.variable_values.get('x2', 0.0):.6f}\n"
                    f"Z = {result.objective_value:.6f}"
                )

            with col2:
                st.markdown("**Binding Constraints:**")
                binding = []
                for idx, (slack_name, value) in enumerate(result.slack_values.items()):
                    if value < 1e-6:
                        binding.append(f"Constraint {idx + 1}")

                if binding:
                    st.write("\n".join([f"- {c}" for c in binding]))
                else:
                    st.write("None (interior solution)")

        except Exception as e:
            st.error(f"❌ Error creating visualization: {str(e)}")

    else:
        st.info(f"""
        📊 2D visualization is only available for problems with exactly 2 variables.

        Your problem has **{lp.num_variables} variables**.

        For higher-dimensional problems, use the sensitivity and dual analysis above
        to understand the solution characteristics.
        """)
