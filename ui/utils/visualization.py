"""
Visualization utilities for Vibe Simplex UI
"""

import numpy as np
import plotly.graph_objects as go
from typing import List, Tuple, Optional
from vibe_simplex.models import LinearProgram, SimplexResult


def create_2d_constraint_plot(
    lp: LinearProgram,
    result: Optional[SimplexResult] = None,
    theme: str = "dark",
) -> go.Figure:
    """
    Create interactive 2D plot showing constraints, feasible region, and optimal point.

    Only works for 2-variable problems.
    """

    if lp.num_variables != 2:
        raise ValueError("2D visualization only supports problems with exactly 2 variables")

    # Determine plot bounds
    max_rhs = max(c.rhs for c in lp.constraints)
    x_max = max_rhs * 1.5
    y_max = max_rhs * 1.5

    # Create figure
    fig = go.Figure()

    # Theme colors
    if theme == "dark":
        bg_color = "#0E1117"
        grid_color = "#2D3748"
        text_color = "#FAFAFA"
        constraint_colors = ["#FF6B6B", "#4ECDC4", "#FFD93D", "#51CF66", "#00D4FF"]
    else:
        bg_color = "#FFFFFF"
        grid_color = "#E2E8F0"
        text_color = "#1A202C"
        constraint_colors = ["#E00", "#7928CA", "#F5A623", "#0070F3", "#F81CE5"]

    # Plot constraint lines
    for idx, constraint in enumerate(lp.constraints):
        a, b = constraint.coefficients[0], constraint.coefficients[1]
        rhs = constraint.rhs

        # Calculate line points
        if abs(b) > 1e-9:
            # Line: ax + by = rhs => y = (rhs - ax) / b
            x_points = np.array([0, x_max])
            y_points = (rhs - a * x_points) / b
        else:
            # Vertical line: x = rhs/a
            x_points = np.array([rhs / a, rhs / a])
            y_points = np.array([0, y_max])

        color = constraint_colors[idx % len(constraint_colors)]

        fig.add_trace(
            go.Scatter(
                x=x_points,
                y=y_points,
                mode="lines",
                name=f"Constraint {idx + 1}",
                line=dict(color=color, width=2),
                hovertemplate=f"Constraint {idx + 1}<br>x1=%{{x:.2f}}<br>x2=%{{y:.2f}}<extra></extra>",
            )
        )

    # Calculate and plot feasible region
    if lp.num_constraints >= 1:
        try:
            # Generate grid
            x_grid = np.linspace(0, x_max, 200)
            y_grid = np.linspace(0, y_max, 200)
            X, Y = np.meshgrid(x_grid, y_grid)

            # Check feasibility at each point
            Z = np.ones_like(X)
            for constraint in lp.constraints:
                a, b = constraint.coefficients[0], constraint.coefficients[1]
                rhs = constraint.rhs

                # For <= constraints
                if constraint.sense == "<=":
                    Z = Z * (a * X + b * Y <= rhs + 1e-6)

            # Plot feasible region with gradient
            fig.add_trace(
                go.Contour(
                    x=x_grid,
                    y=y_grid,
                    z=Z,
                    showscale=False,
                    contours=dict(
                        start=0.5,
                        end=1,
                        size=0.5,
                    ),
                    colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(0, 212, 255, 0.2)"]],
                    hoverinfo="skip",
                    name="Feasible Region",
                )
            )

        except Exception:
            pass  # Skip feasible region if calculation fails

    # Plot optimal point if result is available
    if result is not None and result.status == "optimal":
        x1 = result.variable_values.get("x1", 0.0)
        x2 = result.variable_values.get("x2", 0.0)

        fig.add_trace(
            go.Scatter(
                x=[x1],
                y=[x2],
                mode="markers",
                name="Optimal Point",
                marker=dict(
                    size=15,
                    color="#FFD93D",
                    symbol="star",
                    line=dict(color=text_color, width=2),
                ),
                hovertemplate=f"Optimal Point<br>x1={x1:.4f}<br>x2={x2:.4f}<br>Z={result.objective_value:.4f}<extra></extra>",
            )
        )

        # Add iso-profit line through optimal point
        c1, c2 = lp.objective[0], lp.objective[1]
        if abs(c2) > 1e-9:
            z_opt = result.objective_value
            x_iso = np.array([0, x_max])
            y_iso = (z_opt - c1 * x_iso) / c2

            fig.add_trace(
                go.Scatter(
                    x=x_iso,
                    y=y_iso,
                    mode="lines",
                    name="Optimal Iso-profit",
                    line=dict(color="#FFD93D", width=3, dash="dash"),
                    hovertemplate=f"Z = {z_opt:.4f}<extra></extra>",
                )
            )

    # Calculate corner points
    corner_points = calculate_corner_points(lp, x_max, y_max)
    if corner_points:
        corner_x = [p[0] for p in corner_points]
        corner_y = [p[1] for p in corner_points]

        fig.add_trace(
            go.Scatter(
                x=corner_x,
                y=corner_y,
                mode="markers",
                name="Corner Points",
                marker=dict(size=10, color="#4ECDC4", symbol="circle"),
                hovertemplate="Corner Point<br>x1=%{x:.4f}<br>x2=%{y:.4f}<extra></extra>",
            )
        )

    # Layout
    fig.update_layout(
        title="2D Constraint Plot with Feasible Region",
        xaxis_title="x₁",
        yaxis_title="x₂",
        xaxis=dict(
            range=[0, x_max],
            gridcolor=grid_color,
            zerolinecolor=grid_color,
        ),
        yaxis=dict(
            range=[0, y_max],
            gridcolor=grid_color,
            zerolinecolor=grid_color,
        ),
        plot_bgcolor=bg_color,
        paper_bgcolor=bg_color,
        font=dict(color=text_color),
        hovermode="closest",
        showlegend=True,
        legend=dict(
            x=1.05,
            y=1,
            bgcolor="rgba(0,0,0,0)",
            bordercolor=grid_color,
            borderwidth=1,
        ),
        height=600,
    )

    return fig


def calculate_corner_points(
    lp: LinearProgram,
    x_max: float,
    y_max: float,
) -> List[Tuple[float, float]]:
    """Calculate corner points of the feasible region."""

    corner_points = []

    # Add origin
    if is_feasible_point(lp, (0, 0)):
        corner_points.append((0, 0))

    # Intersections with axes
    for constraint in lp.constraints:
        a, b = constraint.coefficients[0], constraint.coefficients[1]
        rhs = constraint.rhs

        # x-axis intersection (y=0)
        if abs(a) > 1e-9:
            x = rhs / a
            if 0 <= x <= x_max and is_feasible_point(lp, (x, 0)):
                corner_points.append((x, 0))

        # y-axis intersection (x=0)
        if abs(b) > 1e-9:
            y = rhs / b
            if 0 <= y <= y_max and is_feasible_point(lp, (0, y)):
                corner_points.append((0, y))

    # Intersections between constraints
    for i, c1 in enumerate(lp.constraints):
        for j, c2 in enumerate(lp.constraints):
            if i >= j:
                continue

            point = line_intersection(c1, c2)
            if point is not None:
                x, y = point
                if 0 <= x <= x_max and 0 <= y <= y_max:
                    if is_feasible_point(lp, point):
                        corner_points.append(point)

    # Remove duplicates
    unique_points = []
    for point in corner_points:
        is_duplicate = False
        for existing in unique_points:
            if abs(point[0] - existing[0]) < 1e-6 and abs(point[1] - existing[1]) < 1e-6:
                is_duplicate = True
                break
        if not is_duplicate:
            unique_points.append(point)

    return unique_points


def line_intersection(c1, c2) -> Optional[Tuple[float, float]]:
    """Calculate intersection point of two constraint lines."""

    a1, b1, rhs1 = c1.coefficients[0], c1.coefficients[1], c1.rhs
    a2, b2, rhs2 = c2.coefficients[0], c2.coefficients[1], c2.rhs

    det = a1 * b2 - a2 * b1

    if abs(det) < 1e-9:
        return None  # Parallel lines

    x = (b2 * rhs1 - b1 * rhs2) / det
    y = (a1 * rhs2 - a2 * rhs1) / det

    return (x, y)


def is_feasible_point(lp: LinearProgram, point: Tuple[float, float]) -> bool:
    """Check if a point satisfies all constraints."""

    x, y = point

    for constraint in lp.constraints:
        a, b = constraint.coefficients[0], constraint.coefficients[1]
        value = a * x + b * y

        if constraint.sense == "<=":
            if value > constraint.rhs + 1e-6:
                return False
        elif constraint.sense == ">=":
            if value < constraint.rhs - 1e-6:
                return False
        else:  # "="
            if abs(value - constraint.rhs) > 1e-6:
                return False

    return True
