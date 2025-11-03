from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import html
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from vibe_simplex.models import Constraint, LinearProgram, SimplexResult, SimplexStep
from vibe_simplex.solver import SimplexDebugger, SimplexSolver


st.set_page_config(
    page_title="Vibe Simplex Studio",
    page_icon="✨",
    layout="wide",
)

st.markdown(
    """
    <style>
        body {
            background: radial-gradient(circle at top, #111827, #020617);
        }
        .stApp {
            background: linear-gradient(160deg, #020617 0%, #0b1120 45%, #111827 100%);
            color: #e2e8f0;
        }
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0b1120, #020617);
            border-right: 1px solid rgba(148, 163, 184, 0.2);
        }
        .stButton > button {
            background: linear-gradient(135deg, #6366f1, #8b5cf6);
            border: none;
            color: white;
            padding: 0.6rem 1.2rem;
            border-radius: 0.8rem;
            font-weight: 600;
            box-shadow: 0 12px 24px rgba(99, 102, 241, 0.25);
        }
        .stButton > button:hover {
            box-shadow: 0 12px 32px rgba(99, 102, 241, 0.35);
        }
        .metric-card {
            background: rgba(30, 41, 59, 0.6);
            border-radius: 1rem;
            padding: 1rem 1.25rem;
            border: 1px solid rgba(148, 163, 184, 0.15);
            box-shadow: inset 0 0 0 1px rgba(99, 102, 241, 0.15);
        }
        .subsection-title {
            font-size: 1.2rem;
            font-weight: 600;
            color: #cbd5f5;
            margin-bottom: 0.5rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }
        .dataframe {
            background-color: rgba(15, 23, 42, 0.85);
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def _collect_problem_definition() -> Tuple[
    List[float],
    List[Dict[str, Sequence[float]]],
    bool,
    int,
    int,
]:
    with st.sidebar:
        st.header("Problem Setup")
        st.caption("All constraints are of the form ≤ with non-negative decision variables.")

        num_vars = st.number_input("Number of decision variables", 2, 6, 2)
        num_constraints = st.number_input("Number of constraints", 1, 6, 2)

        objective_coeffs: List[float] = []
        st.subheader("Objective Function")
        for var_idx in range(num_vars):
            coeff = st.number_input(
                f"Coefficient for x{var_idx + 1}",
                value=1.0,
                step=0.5,
                key=f"obj_{var_idx}",
            )
            objective_coeffs.append(float(coeff))

        st.markdown("### Constraints")
        constraints: List[Dict[str, Sequence[float]]] = []
        for constraint_idx in range(num_constraints):
            cols = st.columns([3] * num_vars + [2])
            coeffs: List[float] = []
            for var_idx in range(num_vars):
                coeffs.append(
                    float(
                        cols[var_idx].number_input(
                            f"a{constraint_idx + 1}{var_idx + 1}",
                            value=1.0 if var_idx == constraint_idx else 0.0,
                            step=0.5,
                            key=f"const_{constraint_idx}_{var_idx}",
                        )
                    )
                )
            rhs = float(
                cols[-1].number_input(
                    f"b{constraint_idx + 1}",
                    value=5.0,
                    step=0.5,
                    key=f"rhs_{constraint_idx}",
                )
            )
            constraints.append({"coefficients": coeffs, "rhs": rhs})

        st.markdown("---")
        run_requested = st.button("🚀 Run Simplex", type="primary")

    return objective_coeffs, constraints, run_requested, num_vars, num_constraints


def _run_simplex(
    objective: Sequence[float],
    constraints_data: Sequence[Dict[str, Sequence[float]]],
) -> SimplexResultPayload:
    constraints = [
        Constraint(coefficients=item["coefficients"], rhs=item["rhs"])
        for item in constraints_data
    ]
    lp = LinearProgram(objective=objective, constraints=constraints)
    solver = SimplexSolver()
    debugger = SimplexDebugger()
    result = solver.solve(lp, debugger)
    return SimplexResultPayload(result=result, steps=debugger.steps())


@dataclass
class SimplexResultPayload:
    result: SimplexResult
    steps: Sequence[SimplexStep]


def _render_metrics(result: SimplexResult) -> None:
    st.markdown('<div class="subsection-title">Optimal Solution</div>', unsafe_allow_html=True)
    metric_cols = st.columns([1.2, 1, 1, 1])

    objective_value = result.objective_value or 0.0
    metric_cols[0].markdown(
        f"""
        <div class="metric-card">
            <div style="font-size:0.85rem;color:#9ca3af;">Objective Value</div>
            <div style="font-size:2rem;font-weight:700;color:#fcd34d;">{objective_value:.3f}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    variables_html = "".join(
        f"<div><span style='color:#a5b4fc;'>x{idx + 1}</span> = {value:.3f}</div>"
        for idx, value in enumerate(result.variable_values.values())
    )

    metric_cols[1].markdown(
        f"""
        <div class="metric-card">
            <div style="font-size:0.85rem;color:#9ca3af;">Decision Variables</div>
            <div style="font-size:1.05rem;line-height:1.6;">{variables_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    aux_entries = [
        (name, value, "#2dd4bf") for name, value in result.slack_values.items()
    ] + [
        (name, value, "#fca5a5") for name, value in result.surplus_values.items()
    ]
    aux_html = "".join(
        f"<div><span style='color:{color};'>{name}</span> = {value:.3f}</div>"
        for name, value, color in aux_entries
    )
    metric_cols[2].markdown(
        f"""
        <div class="metric-card">
            <div style="font-size:0.85rem;color:#9ca3af;">Slack / Surplus</div>
            <div style="font-size:1.05rem;line-height:1.6;">{aux_html or '—'}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    dual_html = "".join(
        f"<div><span style='color:#fca5a5;'>y{idx + 1}</span> = {value:.3f}</div>"
        for idx, value in enumerate(result.dual.dual_variables.values())
    )
    metric_cols[3].markdown(
        f"""
        <div class="metric-card">
            <div style="font-size:0.85rem;color:#9ca3af;">Dual Solution</div>
            <div style="font-size:1.05rem;line-height:1.6;">{dual_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_stepper(
    steps: Sequence[SimplexStep],
    num_vars: int,
    num_constraints: int,
) -> None:
    st.markdown('<div class="subsection-title">Step-by-Step Debugger</div>', unsafe_allow_html=True)
    if not steps:
        st.info("Run the solver to enable step-by-step inspection.")
        return

    slider = st.slider(
        "Select iteration",
        min_value=0,
        max_value=len(steps) - 1,
        value=len(steps) - 1,
    )
    step = steps[slider]

    total_vars = len(step.tableau[0]) - 1
    slack_vars = max(total_vars - num_vars, 0)
    columns = (
        [f"x{i + 1}" for i in range(num_vars)]
        + [f"s{i + 1}" for i in range(slack_vars)]
        + ["RHS"]
    )
    row_labels = [f"Constraint {i + 1}" for i in range(len(step.tableau) - 1)] + [
        "Objective"
    ]
    df = pd.DataFrame(step.tableau, columns=columns, index=row_labels)

    def highlight(_: pd.DataFrame) -> pd.DataFrame:
        styles = pd.DataFrame("", index=df.index, columns=df.columns)
        if step.pivot:
            p_row, p_col = step.pivot
            styles.iat[p_row, p_col] = "background-color:#f97316;color:#020617;font-weight:600;"
        return styles

    st.dataframe(df.style.format("{:.3f}").apply(highlight, axis=None))

    info_cols = st.columns(4)
    info_cols[0].metric("Phase", step.phase)
    info_cols[1].metric("Iteration", step.iteration)
    info_cols[2].metric("Entering Variable", step.entering_variable or "—")
    info_cols[3].metric("Leaving Variable", step.leaving_variable or "—")

    objective_label = (
        "Phase I Objective (sum of artificials)"
        if step.phase.startswith("Phase I")
        else "Objective Value"
    )
    details_text = html.escape(step.details or "")
    st.markdown(
        f"""
        <div class="metric-card" style="margin-top:0.75rem;">
            <div style="font-size:0.85rem;color:#9ca3af;">{objective_label}</div>
            <div style="font-size:2rem;font-weight:700;color:#fcd34d;">{step.objective_value:.3f}</div>
            <div style="margin-top:0.5rem;line-height:1.6;">{details_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_sensitivity(result: SimplexResult) -> None:
    st.markdown('<div class="subsection-title">Sensitivity Analysis</div>', unsafe_allow_html=True)
    tab1, tab2 = st.tabs(["Shadow Prices", "Objective / RHS Impact"])

    with tab1:
        df_shadow = pd.DataFrame(
            list(result.sensitivity.shadow_prices.items()),
            columns=["Constraint", "Shadow Price"],
        )
        st.dataframe(df_shadow.style.format({"Shadow Price": "{:.3f}"}))

    with tab2:
        reduced_df = pd.DataFrame(
            list(result.sensitivity.reduced_costs.items()),
            columns=["Variable", "Reduced Cost"],
        )
        rhs_df = pd.DataFrame(result.sensitivity.rhs_ranges).T.rename(
            columns={
                "allowable_increase": "Allowable Increase",
                "allowable_decrease": "Allowable Decrease",
            }
        )
        col_a, col_b = st.columns(2)
        col_a.dataframe(reduced_df.style.format({"Reduced Cost": "{:.3f}"}))
        col_b.dataframe(
            rhs_df.style.format(
                {"Allowable Increase": "{:.3f}", "Allowable Decrease": "{:.3f}"}
            )
        )


def _render_dual(result: SimplexResult) -> None:
    st.markdown('<div class="subsection-title">Dual Analysis</div>', unsafe_allow_html=True)
    dual_df = pd.DataFrame(
        list(result.dual.dual_variables.items()),
        columns=["Dual Variable", "Value"],
    )
    col1, col2 = st.columns([1.5, 1])
    col1.dataframe(dual_df.style.format({"Value": "{:.3f}"}))
    col2.metric("Dual Objective", result.dual.dual_objective_value or 0.0)
    slack_df = pd.DataFrame(
        [
            {"Slack": name, "Complementary Slackness": "Satisfied" if ok else "Violated"}
            for name, ok in result.dual.complementary_slackness.items()
        ]
    )
    col2.dataframe(slack_df)


def _render_feasible_region(
    constraints: Sequence[Dict[str, Sequence[float]]],
    objective: Sequence[float],
    result: SimplexResult,
) -> None:
    st.markdown('<div class="subsection-title">2D Geometry</div>', unsafe_allow_html=True)
    if len(objective) != 2:
        st.info("2D visualisation is available when working with two decision variables.")
        return

    a = np.array([constraint["coefficients"] for constraint in constraints], dtype=float)
    b = np.array([constraint["rhs"] for constraint in constraints], dtype=float)

    max_x = []
    max_y = []
    for row, rhs in zip(a, b):
        if row[0] > 0:
            max_x.append(rhs / row[0])
        if row[1] > 0:
            max_y.append(rhs / row[1])
    max_x_val = max(max(max_x, default=5.0), result.variable_values.get("x1", 0.0)) + 1.0
    max_y_val = max(max(max_y, default=5.0), result.variable_values.get("x2", 0.0)) + 1.0

    grid = 250
    x = np.linspace(0, max_x_val, grid)
    y = np.linspace(0, max_y_val, grid)
    X, Y = np.meshgrid(x, y)
    mask = np.ones_like(X, dtype=bool)
    for row, rhs in zip(a, b):
        mask &= row[0] * X + row[1] * Y <= rhs + 1e-9

    feasible_points = np.column_stack((X[mask], Y[mask]))
    fig = go.Figure()
    if feasible_points.size:
        fig.add_trace(
            go.Scatter(
                x=feasible_points[:, 0],
                y=feasible_points[:, 1],
                mode="markers",
                marker=dict(size=4, color="#60a5fa", opacity=0.55),
                name="Feasible Region",
                showlegend=False,
            )
        )

    for idx, (row, rhs) in enumerate(zip(a, b), start=1):
        if abs(row[1]) > 1e-9:
            line_x = np.linspace(0, max_x_val, 200)
            line_y = (rhs - row[0] * line_x) / row[1]
        else:
            line_x = np.full(200, rhs / row[0] if row[0] != 0 else 0)
            line_y = np.linspace(0, max_y_val, 200)
        fig.add_trace(
            go.Scatter(
                x=line_x,
                y=line_y,
                mode="lines",
                line=dict(width=2, dash="dash"),
                name=f"Constraint {idx}",
            )
        )

    objective_value = result.objective_value or 0.0
    c1, c2 = objective
    if abs(c2) > 1e-9:
        obj_x = np.linspace(0, max_x_val, 200)
        obj_y = (objective_value - c1 * obj_x) / c2
        fig.add_trace(
            go.Scatter(
                x=obj_x,
                y=obj_y,
                mode="lines",
                line=dict(color="#f472b6", width=3),
                name="Objective Iso-line",
            )
        )
    elif abs(c1) > 1e-9:
        x_value = objective_value / c1
        fig.add_trace(
            go.Scatter(
                x=[x_value, x_value],
                y=[0, max_y_val],
                mode="lines",
                line=dict(color="#f472b6", width=3),
                name="Objective Iso-line",
            )
        )

    fig.add_trace(
        go.Scatter(
            x=[result.variable_values.get("x1", 0.0)],
            y=[result.variable_values.get("x2", 0.0)],
            mode="markers+text",
            marker=dict(size=12, color="#facc15"),
            text=["Optimal"],
            textposition="top center",
            name="Optimal Solution",
        )
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(17, 24, 39, 0.0)",
        plot_bgcolor="rgba(15, 23, 42, 0.8)",
        xaxis_title="x₁",
        yaxis_title="x₂",
        legend=dict(orientation="h", y=-0.2, x=0),
        margin=dict(l=40, r=20, t=30, b=60),
    )
    st.plotly_chart(fig, use_container_width=True)


def main() -> None:
    objective, constraints, run_requested, num_vars, num_constraints = (
        _collect_problem_definition()
    )

    if run_requested:
        try:
            payload = _run_simplex(objective, constraints)
            st.session_state["simplex_payload"] = payload
            st.session_state["problem_meta"] = {
                "objective": objective,
                "constraints": constraints,
                "num_vars": num_vars,
                "num_constraints": num_constraints,
            }
            st.success("Simplex analysis completed successfully!")
        except Exception as exc:  # noqa: BLE001
            st.session_state["simplex_payload"] = None
            st.error(f"Unable to solve the problem: {exc}")

    payload: SimplexResultPayload | None = st.session_state.get("simplex_payload")
    meta = st.session_state.get("problem_meta", {})

    st.title("✨ Vibe Simplex Studio")
    st.caption("Interactive simplex solver with debugging, geometry, sensitivity, and dual insights.")

    if not payload:
        st.info("Configure your linear program in the sidebar and press **Run Simplex** to begin.")
        return

    result = payload.result
    if result.status == "optimal":
        _render_metrics(result)
    elif result.status == "unbounded":
        st.warning("The linear program is unbounded. Review the formulation for missing constraints.")
    else:
        st.error("The linear program is infeasible under the current assumptions.")
        return

    st.markdown("---")
    _render_stepper(payload.steps, meta.get("num_vars", 2), meta.get("num_constraints", 2))

    st.markdown("---")
    col_left, col_right = st.columns([1.5, 1])
    with col_left:
        _render_feasible_region(meta.get("constraints", []), meta.get("objective", []), result)
    with col_right:
        _render_sensitivity(result)

    st.markdown("---")
    _render_dual(result)


if __name__ == "__main__":
    main()
