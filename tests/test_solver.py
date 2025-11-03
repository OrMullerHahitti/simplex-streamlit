import pytest

from vibe_simplex.models import Constraint, LinearProgram
from vibe_simplex.solver import SimplexDebugger, SimplexSolver


def build_sample_lp() -> LinearProgram:
    return LinearProgram(
        objective=[3.0, 5.0],
        constraints=[
            Constraint([2.0, 3.0], 8.0),
            Constraint([1.0, 1.0], 4.0),
        ],
    )


def test_simplex_solver_finds_optimum_and_records_steps() -> None:
    lp = build_sample_lp()
    solver = SimplexSolver()
    debugger = SimplexDebugger()

    result = solver.solve(lp, debugger)

    assert result.status == "optimal"
    assert result.objective_value == pytest.approx(40 / 3)
    assert result.variable_values["x1"] == pytest.approx(0.0)
    assert result.variable_values["x2"] == pytest.approx(8 / 3)
    assert len(result.steps) == len(debugger.steps())
    assert result.steps[-1].phase == "Phase II"
    assert result.steps[-1].objective_value == pytest.approx(result.objective_value)

    pivot_steps = [step for step in result.steps if step.pivot]
    assert pivot_steps, "Expected at least one pivot step."
    first_pivot = pivot_steps[0]
    assert first_pivot.entering_variable == "x2"
    assert first_pivot.leaving_variable == "s1"
    assert all(step.details for step in result.steps)


def test_sensitivity_and_dual_analysis_are_generated() -> None:
    lp = build_sample_lp()
    solver = SimplexSolver()

    result = solver.solve(lp)

    shadow = result.sensitivity.shadow_prices
    reduced = result.sensitivity.reduced_costs
    ranges = result.sensitivity.rhs_ranges
    dual = result.dual

    assert shadow["constraint_1"] == pytest.approx(5 / 3)
    assert shadow["constraint_2"] == pytest.approx(0.0)
    assert reduced["x1"] == pytest.approx(-1 / 3)
    assert reduced["x2"] == pytest.approx(0.0)
    assert ranges["constraint_1"]["allowable_decrease"] == pytest.approx(8.0)
    assert ranges["constraint_2"]["allowable_decrease"] == pytest.approx(4 / 3)
    assert dual.dual_objective_value == pytest.approx(result.objective_value)
    assert dual.dual_variables["y1"] == pytest.approx(5 / 3)
    assert dual.dual_variables["y2"] == pytest.approx(0.0)
    assert all(dual.complementary_slackness.values())


def test_debugger_allows_navigation() -> None:
    lp = build_sample_lp()
    solver = SimplexSolver()
    debugger = SimplexDebugger()

    solver.solve(lp, debugger)

    assert debugger.current().iteration == 0
    assert debugger.next().iteration == 1
    assert debugger.previous().iteration == 0


def test_solver_handles_minimisation_with_greater_equal_constraints() -> None:
    lp = LinearProgram(
        objective=[1.0, 1.0],
        constraints=[
            Constraint([2.0, 1.0], 6.0, sense=">="),
            Constraint([1.0, 1.0], 4.0, sense=">="),
        ],
        sense="min",
    )
    solver = SimplexSolver()

    result = solver.solve(lp)

    assert result.status == "optimal"
    assert result.objective_value == pytest.approx(4.0)
    assert result.variable_values["x1"] == pytest.approx(2.0)
    assert result.variable_values["x2"] == pytest.approx(2.0)
    assert all(step.phase in {"Phase I", "Phase II"} for step in result.steps)
    assert {"Phase I", "Phase II"} <= {step.phase for step in result.steps}
    for value in result.surplus_values.values():
        assert value == pytest.approx(0.0, abs=1e-8)


def test_solver_handles_equality_constraints_via_phase_one() -> None:
    lp = LinearProgram(
        objective=[1.0, 1.0],
        constraints=[
            Constraint([1.0, 1.0], 4.0, sense="="),
            Constraint([1.0, 0.0], 3.0),
            Constraint([0.0, 1.0], 3.0),
        ],
    )
    solver = SimplexSolver()

    result = solver.solve(lp)

    assert result.status == "optimal"
    assert result.objective_value == pytest.approx(4.0)
    assert any(step.phase == "Phase I" for step in result.steps)
    assert result.steps[-1].phase == "Phase II"
    assert result.steps[-1].objective_value == pytest.approx(4.0)


def test_solver_handles_negative_rhs_by_flipping_constraints() -> None:
    lp = LinearProgram(
        objective=[3.0, 2.0],
        constraints=[
            Constraint([-1.0, 2.0], -2.0, sense="<="),
            Constraint([1.0, 0.0], 4.0),
        ],
    )
    solver = SimplexSolver()

    result = solver.solve(lp)

    assert result.status == "optimal"
    assert result.objective_value == pytest.approx(14.0)
    assert result.variable_values["x1"] == pytest.approx(4.0)
    assert result.variable_values["x2"] == pytest.approx(1.0)
    assert {"Phase I", "Phase II"} <= {step.phase for step in result.steps}


def test_infeasible_problem_detected() -> None:
    lp = LinearProgram(
        objective=[1.0, 1.0],
        constraints=[
            Constraint([1.0, 1.0], 2.0, sense="<="),
            Constraint([1.0, 1.0], 5.0, sense=">="),
        ],
    )
    solver = SimplexSolver()

    result = solver.solve(lp)

    assert result.status == "infeasible"
    assert result.objective_value is None
    assert "infeasible" in result.message.lower()


def test_redundant_equality_reports_cleanup() -> None:
    lp = LinearProgram(
        objective=[1.0, 1.0],
        constraints=[
            Constraint([1.0, 1.0], 5.0),
            Constraint([1.0, 0.0], 4.0),
            Constraint([0.0, 0.0], 0.0, sense="="),
        ],
    )
    solver = SimplexSolver()

    result = solver.solve(lp)

    assert result.status == "optimal"
    assert any("remains basic" in step.details for step in result.steps)


def test_unbounded_problem_is_detected() -> None:
    lp = LinearProgram(
        objective=[1.0, 0.0],
        constraints=[
            Constraint([-1.0, 1.0], 2.0),
            Constraint([0.0, 1.0], 5.0),
        ],
    )
    solver = SimplexSolver()

    result = solver.solve(lp)

    assert result.status == "unbounded"
    assert result.objective_value is None
    assert "unbounded" in result.message.lower()


def test_constraint_and_program_validation() -> None:
    constraint = Constraint([1.0, 2.0], 5.0)
    with pytest.raises(ValueError):
        constraint.validate(3)

    invalid_constraint = Constraint([1.0], 1.0, sense="=>")
    with pytest.raises(ValueError):
        invalid_constraint.validate(1)

    with pytest.raises(ValueError):
        LinearProgram(objective=[1.0], constraints=[], sense="max")

    with pytest.raises(ValueError):
        LinearProgram(
            objective=[1.0],
            constraints=[Constraint([1.0], 1.0)],
            sense="maximize",
        )

    lp = build_sample_lp()
    assert lp.num_variables == 2
    assert lp.num_constraints == 2


def test_debugger_reset_clears_history() -> None:
    debugger = SimplexDebugger()
    assert debugger.current() is None
    assert debugger.next() is None
    assert debugger.previous() is None

    lp = build_sample_lp()
    solver = SimplexSolver()
    solver.solve(lp, debugger)

    assert debugger.current() is not None
    debugger.reset()
    assert debugger.current() is None
    assert len(debugger.steps()) == 0
