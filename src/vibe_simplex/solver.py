from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .models import (
    Constraint,
    LinearProgram,
    SensitivityReport,
    SimplexResult,
    SimplexStep,
    SimplexStatus,
    DualReport,
)


class SimplexDebugger:
    """Stores the recorded simplex steps and offers step-by-step navigation."""

    def __init__(self) -> None:
        self._steps: List[SimplexStep] = []
        self._cursor: int = 0

    def reset(self) -> None:
        self._steps.clear()
        self._cursor = 0

    def record(self, step: SimplexStep) -> None:
        self._steps.append(step)

    def steps(self) -> Sequence[SimplexStep]:
        return tuple(self._steps)

    def current(self) -> Optional[SimplexStep]:
        if not self._steps:
            return None
        return self._steps[self._cursor]

    def next(self) -> Optional[SimplexStep]:
        if self._cursor + 1 < len(self._steps):
            self._cursor += 1
            return self._steps[self._cursor]
        return None

    def previous(self) -> Optional[SimplexStep]:
        if self._cursor - 1 >= 0:
            self._cursor -= 1
            return self._steps[self._cursor]
        return None


@dataclass
class _StandardForm:
    objective_multiplier: float
    original_variable_count: int
    constraint_matrix: np.ndarray
    rhs: np.ndarray
    objective: np.ndarray
    phase_one_objective: Optional[np.ndarray]
    variable_names: List[str]
    slack_names: List[str] = field(default_factory=list)
    surplus_names: List[str] = field(default_factory=list)
    artificial_names: List[str] = field(default_factory=list)
    constraint_slack_map: List[Optional[str]] = field(default_factory=list)
    constraint_surplus_map: List[Optional[str]] = field(default_factory=list)
    artificial_index_list: List[int] = field(default_factory=list)
    initial_basis: List[int] = field(default_factory=list)

    @property
    def requires_phase_one(self) -> bool:
        return bool(self.artificial_names)

    @property
    def artificial_indices(self) -> List[int]:
        return list(self.artificial_index_list)


class SimplexSolver:
    """Primal simplex solver supporting ≤, ≥, and = constraints via a two-phase method."""

    def __init__(self, tolerance: float = 1e-9, max_iterations: int = 20_000) -> None:
        self.tolerance = tolerance
        self.max_iterations = max_iterations

    def solve(
        self, lp: LinearProgram, debugger: Optional[SimplexDebugger] = None
    ) -> SimplexResult:
        standard = self._to_standard_form(lp)
        tableau = self._initialize_tableau(standard)
        basis = list(standard.initial_basis)
        steps: List[SimplexStep] = []
        iteration = 0
        status: SimplexStatus = "optimal"

        if debugger is not None:
            debugger.reset()

        # Phase I: remove artificial variables to reach feasibility
        if standard.requires_phase_one:
            self._set_objective_row(tableau, basis, standard.phase_one_objective)
            self._record_step(
                tableau=tableau,
                basis=basis,
                standard=standard,
                phase="Phase I",
                iteration=iteration,
                entering=None,
                leaving=None,
                pivot=None,
                details="Initial Phase I tableau — drive artificial variables to zero.",
                steps=steps,
                debugger=debugger,
                cost_vector=standard.phase_one_objective,
            )
            status, iteration = self._perform_simplex(
                tableau=tableau,
                basis=basis,
                standard=standard,
                steps=steps,
                debugger=debugger,
                iteration=iteration,
                phase="Phase I",
                cost_vector=standard.phase_one_objective,
            )

            if status != "optimal":
                return self._build_infeasible_result(
                    lp, standard, tableau, basis, steps
                )

            phase_one_obj = -self._compute_phase_objective(
                tableau, basis, standard.phase_one_objective
            )
            if phase_one_obj > self.tolerance:
                return self._build_infeasible_result(
                    lp, standard, tableau, basis, steps
                )

            iteration, tableau, basis = self._remove_artificial_variables(
                tableau=tableau,
                basis=basis,
                standard=standard,
                steps=steps,
                debugger=debugger,
                iteration=iteration,
            )

        # Phase II: optimise original objective
        self._set_objective_row(tableau, basis, standard.objective)
        phase_label = "Phase II"
        details = (
            "Phase II initial tableau — optimise the original objective."
            if standard.requires_phase_one
            else "Initial tableau — optimise the original objective."
        )
        self._record_step(
            tableau=tableau,
            basis=basis,
            standard=standard,
            phase=phase_label,
            iteration=iteration,
            entering=None,
            leaving=None,
            pivot=None,
            details=details,
            steps=steps,
            debugger=debugger,
            cost_vector=standard.objective,
        )
        status, iteration = self._perform_simplex(
            tableau=tableau,
            basis=basis,
            standard=standard,
            steps=steps,
            debugger=debugger,
            iteration=iteration,
            phase=phase_label,
            cost_vector=standard.objective,
        )

        result = self._build_result(
            lp=lp,
            standard=standard,
            tableau=tableau,
            basis=basis,
            status=status,
            steps=steps,
        )
        return result

    # ---------------------------------------------------------------------
    # Tableau utilities
    # ---------------------------------------------------------------------
    def _initialize_tableau(self, standard: _StandardForm) -> np.ndarray:
        rows, cols = standard.constraint_matrix.shape
        tableau = np.zeros((rows + 1, cols + 1), dtype=float)
        tableau[:-1, :-1] = standard.constraint_matrix
        tableau[:-1, -1] = standard.rhs
        return tableau

    def _set_objective_row(
        self,
        tableau: np.ndarray,
        basis: Sequence[int],
        cost_vector: Optional[np.ndarray],
    ) -> None:
        tableau[-1, :] = 0.0
        if cost_vector is None:
            return
        tableau[-1, :-1] = -cost_vector
        for row_index, basis_index in enumerate(basis):
            cost = cost_vector[basis_index]
            if abs(cost) > self.tolerance:
                tableau[-1, :] += cost * tableau[row_index, :]

    # ---------------------------------------------------------------------
    # Core simplex phase execution
    # ---------------------------------------------------------------------
    def _perform_simplex(
        self,
        tableau: np.ndarray,
        basis: List[int],
        standard: _StandardForm,
        steps: List[SimplexStep],
        debugger: Optional[SimplexDebugger],
        iteration: int,
        phase: str,
        cost_vector: Optional[np.ndarray],
    ) -> Tuple[SimplexStatus, int]:
        while True:
            entering = self._choose_entering_variable(tableau)
            if entering is None:
                self._record_step(
                    tableau=tableau,
                    basis=basis,
                    standard=standard,
                    phase=phase,
                    iteration=iteration,
                    entering=None,
                    leaving=None,
                    pivot=None,
                    details=f"{phase}: no positive reduced costs — optimality reached.",
                    steps=steps,
                    debugger=debugger,
                    cost_vector=cost_vector,
                )
                return "optimal", iteration

            leaving = self._choose_leaving_variable(tableau, entering)
            entering_name = standard.variable_names[entering]
            if leaving is None:
                self._record_step(
                    tableau=tableau,
                    basis=basis,
                    standard=standard,
                    phase=phase,
                    iteration=iteration,
                    entering=entering,
                    leaving=None,
                    pivot=None,
                    details=(
                        f"{phase}: entering variable {entering_name} leads to an "
                        "unbounded increase in the objective."
                    ),
                    steps=steps,
                    debugger=debugger,
                    cost_vector=cost_vector,
                )
                return "unbounded", iteration

            iteration += 1
            if iteration > self.max_iterations:
                raise RuntimeError("Maximum simplex iterations exceeded.")

            leaving_index = basis[leaving]
            leaving_name = standard.variable_names[leaving_index]
            self._pivot(tableau, leaving, entering)
            basis[leaving] = entering

            details = (
                f"{phase}: pivot on row {leaving + 1}, column {entering + 1}; "
                f"{entering_name} enters, {leaving_name} leaves the basis."
            )
            self._record_step(
                tableau=tableau,
                basis=basis,
                standard=standard,
                phase=phase,
                iteration=iteration,
                entering=entering,
                leaving=leaving_index,
                pivot=(leaving, entering),
                details=details,
                steps=steps,
                debugger=debugger,
                cost_vector=cost_vector,
            )

    def _remove_artificial_variables(
        self,
        tableau: np.ndarray,
        basis: List[int],
        standard: _StandardForm,
        steps: List[SimplexStep],
        debugger: Optional[SimplexDebugger],
        iteration: int,
    ) -> Tuple[int, np.ndarray, List[int]]:
        artificial_set = set(standard.artificial_indices)
        if not artificial_set:
            return iteration, tableau, basis

        redundant_rows: List[int] = []
        for row, var_index in enumerate(list(basis)):
            if var_index not in artificial_set:
                continue

            found = False
            for candidate in range(tableau.shape[1] - 1):
                if candidate in artificial_set:
                    continue
                if abs(tableau[row, candidate]) > self.tolerance:
                    iteration += 1
                    leaving_name = standard.variable_names[var_index]
                    entering_name = standard.variable_names[candidate]
                    self._pivot(tableau, row, candidate)
                    basis[row] = candidate
                    details = (
                        "Phase I cleanup: replaced artificial variable "
                        f"{leaving_name} with {entering_name}."
                    )
                    self._record_step(
                        tableau=tableau,
                        basis=basis,
                        standard=standard,
                        phase="Phase I",
                        iteration=iteration,
                        entering=candidate,
                        leaving=var_index,
                        pivot=(row, candidate),
                        details=details,
                        steps=steps,
                        debugger=debugger,
                        cost_vector=standard.phase_one_objective,
                    )
                    found = True
                    break
            if not found:
                details = (
                    "Phase I cleanup: artificial variable "
                    f"{standard.variable_names[var_index]} remains basic but "
                    "has zero value; constraint is redundant."
                )
                self._record_step(
                    tableau=tableau,
                    basis=basis,
                    standard=standard,
                    phase="Phase I",
                    iteration=iteration,
                    entering=None,
                    leaving=var_index,
                    pivot=None,
                    details=details,
                    steps=steps,
                    debugger=debugger,
                    cost_vector=standard.phase_one_objective,
                )
                redundant_rows.append(row)
        if redundant_rows:
            for row in sorted(redundant_rows, reverse=True):
                tableau = np.delete(tableau, row, axis=0)
                basis.pop(row)
                standard.constraint_matrix = np.delete(
                    standard.constraint_matrix, row, axis=0
                )
                standard.rhs = np.delete(standard.rhs, row, axis=0)
                if row < len(standard.constraint_slack_map):
                    standard.constraint_slack_map.pop(row)
                if row < len(standard.constraint_surplus_map):
                    standard.constraint_surplus_map.pop(row)

        tableau, basis = self._drop_artificial_columns(tableau, basis, standard)
        return iteration, tableau, basis

    def _drop_artificial_columns(
        self,
        tableau: np.ndarray,
        basis: Sequence[int],
        standard: _StandardForm,
    ) -> Tuple[np.ndarray, List[int]]:
        artificial_set = set(standard.artificial_indices)
        if not artificial_set:
            return tableau, list(basis)

        keep_indices = [
            idx for idx in range(tableau.shape[1] - 1) if idx not in artificial_set
        ]
        mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(keep_indices)}

        new_tableau = np.zeros((tableau.shape[0], len(keep_indices) + 1), dtype=float)
        new_tableau[:, :-1] = tableau[:, keep_indices]
        new_tableau[:, -1] = tableau[:, -1]

        new_basis = [mapping[idx] for idx in basis]

        standard.constraint_matrix = standard.constraint_matrix[:, keep_indices]
        standard.objective = standard.objective[keep_indices]
        if standard.phase_one_objective is not None:
            standard.phase_one_objective = standard.phase_one_objective[keep_indices]
        standard.variable_names = [standard.variable_names[idx] for idx in keep_indices]
        standard.artificial_names = []
        standard.artificial_index_list = []

        return new_tableau, new_basis

    # ---------------------------------------------------------------------
    # Tableau helpers and numerical routines
    # ---------------------------------------------------------------------
    def _choose_entering_variable(self, tableau: np.ndarray) -> Optional[int]:
        reduced_costs = -tableau[-1, :-1]
        candidates = np.where(reduced_costs > self.tolerance)[0]
        if candidates.size == 0:
            return None
        return int(candidates[np.argmax(reduced_costs[candidates])])

    def _choose_leaving_variable(
        self, tableau: np.ndarray, entering: int
    ) -> Optional[int]:
        column = tableau[:-1, entering]
        rhs = tableau[:-1, -1]
        ratios: List[Tuple[float, int]] = []
        for idx, value in enumerate(column):
            if value > self.tolerance:
                ratios.append((rhs[idx] / value, idx))
        if not ratios:
            return None
        _, leaving = min(ratios, key=lambda item: (item[0], item[1]))
        return leaving

    def _pivot(self, tableau: np.ndarray, row: int, col: int) -> None:
        pivot_value = tableau[row, col]
        tableau[row, :] /= pivot_value
        for r in range(tableau.shape[0]):
            if r == row:
                continue
            factor = tableau[r, col]
            if abs(factor) > self.tolerance:
                tableau[r, :] -= factor * tableau[row, :]

    def _record_step(
        self,
        tableau: np.ndarray,
        basis: Sequence[int],
        standard: _StandardForm,
        phase: str,
        iteration: int,
        entering: Optional[int],
        leaving: Optional[int],
        pivot: Optional[Tuple[int, int]],
        details: str,
        steps: List[SimplexStep],
        debugger: Optional[SimplexDebugger],
        cost_vector: Optional[np.ndarray],
    ) -> None:
        entering_name = (
            standard.variable_names[entering] if entering is not None else None
        )
        leaving_name = None
        if leaving is not None:
            leaving_idx = leaving
            if leaving < len(standard.variable_names):
                leaving_name = standard.variable_names[leaving_idx]
            else:
                leaving_name = f"var_{leaving_idx}"

        objective_value = self._compute_display_objective(
            tableau=tableau,
            basis=basis,
            standard=standard,
            phase=phase,
            cost_vector=cost_vector,
        )
        step = SimplexStep(
            iteration=iteration,
            phase=phase,
            tableau=tableau.copy().tolist(),
            basis=[standard.variable_names[idx] for idx in basis],
            entering_variable=entering_name,
            leaving_variable=leaving_name,
            pivot=pivot,
            objective_value=objective_value,
            details=details,
        )
        steps.append(step)
        if debugger is not None:
            debugger.record(step)

    def _compute_phase_objective(
        self,
        tableau: np.ndarray,
        basis: Sequence[int],
        cost_vector: Optional[np.ndarray],
    ) -> float:
        if cost_vector is None:
            return 0.0
        basis_values = tableau[:-1, -1]
        return float(np.dot(cost_vector[list(basis)], basis_values))

    def _compute_display_objective(
        self,
        tableau: np.ndarray,
        basis: Sequence[int],
        standard: _StandardForm,
        phase: str,
        cost_vector: Optional[np.ndarray],
    ) -> float:
        phase_value = self._compute_phase_objective(tableau, basis, cost_vector)
        if phase == "Phase I":
            return -phase_value
        return phase_value * standard.objective_multiplier

    # ---------------------------------------------------------------------
    # Result assembly
    # ---------------------------------------------------------------------
    def _build_result(
        self,
        lp: LinearProgram,
        standard: _StandardForm,
        tableau: np.ndarray,
        basis: Sequence[int],
        status: SimplexStatus,
        steps: Sequence[SimplexStep],
    ) -> SimplexResult:
        if status == "unbounded":
            return SimplexResult(
                status="unbounded",
                objective_value=None,
                variable_values={},
                slack_values={},
                surplus_values={},
                steps=steps,
                sensitivity=SensitivityReport(),
                dual=DualReport(),
                message="The linear program is unbounded.",
            )

        objective_value = self._compute_phase_objective(
            tableau, basis, standard.objective
        )
        objective_value *= standard.objective_multiplier

        variable_values = self._extract_solution(tableau, basis, standard.variable_names)
        primal_values = {
            name: variable_values.get(name, 0.0)
            for name in standard.variable_names[: standard.original_variable_count]
        }
        slack_values = {
            name: variable_values.get(name, 0.0) for name in standard.slack_names
        }
        surplus_values = {
            name: variable_values.get(name, 0.0) for name in standard.surplus_names
        }

        sensitivity = SensitivityReport(
            shadow_prices=self._compute_shadow_prices(standard, basis),
            reduced_costs=self._compute_reduced_costs(
                standard, tableau, basis, standard.original_variable_count
            ),
            rhs_ranges=self._compute_rhs_ranges(standard, basis),
        )
        dual = self._compute_dual_report(standard, basis, slack_values, surplus_values)

        return SimplexResult(
            status="optimal",
            objective_value=objective_value,
            variable_values=primal_values,
            slack_values=slack_values,
            surplus_values=surplus_values,
            steps=steps,
            sensitivity=sensitivity,
            dual=dual,
            message="Optimal solution found.",
        )

    def _build_infeasible_result(
        self,
        lp: LinearProgram,
        standard: _StandardForm,
        tableau: np.ndarray,
        basis: Sequence[int],
        steps: Sequence[SimplexStep],
    ) -> SimplexResult:
        return SimplexResult(
            status="infeasible",
            objective_value=None,
            variable_values={},
            slack_values={},
            surplus_values={},
            steps=steps,
            sensitivity=SensitivityReport(),
            dual=DualReport(),
            message="The linear program is infeasible.",
        )

    # ---------------------------------------------------------------------
    # Post-solution analytics
    # ---------------------------------------------------------------------
    def _extract_solution(
        self, tableau: np.ndarray, basis: Sequence[int], variable_names: Sequence[str]
    ) -> Dict[str, float]:
        solution = np.zeros(len(variable_names), dtype=float)
        for row_index, var_index in enumerate(basis):
            solution[var_index] = tableau[row_index, -1]
        return {
            variable_names[i]: float(max(solution[i], 0.0))
            for i in range(len(variable_names))
        }

    def _compute_shadow_prices(
        self, standard: _StandardForm, basis: Sequence[int]
    ) -> Dict[str, float]:
        m = standard.constraint_matrix.shape[0]
        basis_matrix = standard.constraint_matrix[:, basis]
        try:
            b_inv = np.linalg.inv(basis_matrix)
        except np.linalg.LinAlgError as exc:
            raise RuntimeError(
                "Basis matrix is singular; cannot compute dual variables."
            ) from exc
        cost_vector = standard.objective[basis]
        shadow_prices = cost_vector @ b_inv
        return {f"constraint_{i + 1}": float(shadow_prices[i]) for i in range(m)}

    def _compute_reduced_costs(
        self,
        standard: _StandardForm,
        tableau: np.ndarray,
        basis: Sequence[int],
        primal_variable_count: int,
    ) -> Dict[str, float]:
        total_variables = len(standard.variable_names)
        reduced_costs: Dict[str, float] = {}
        for idx in range(total_variables):
            name = standard.variable_names[idx]
            if idx in basis:
                reduced_costs[name] = 0.0
            else:
                reduced_costs[name] = float(-tableau[-1, idx])
        return {
            name: value
            for name, value in reduced_costs.items()
            if name in standard.variable_names[:primal_variable_count]
        }

    def _compute_rhs_ranges(
        self, standard: _StandardForm, basis: Sequence[int]
    ) -> Dict[str, Dict[str, float]]:
        m = standard.constraint_matrix.shape[0]
        basis_matrix = standard.constraint_matrix[:, basis]
        try:
            b_inv = np.linalg.inv(basis_matrix)
        except np.linalg.LinAlgError:
            return {}

        x_b = b_inv @ standard.rhs
        ranges: Dict[str, Dict[str, float]] = {}
        for idx in range(m):
            column = b_inv[:, idx]
            increases: List[float] = []
            decreases: List[float] = []
            for row_idx, coeff in enumerate(column):
                if abs(coeff) <= self.tolerance:
                    continue
                if coeff < 0:
                    increases.append(x_b[row_idx] / (-coeff))
                else:
                    decreases.append(x_b[row_idx] / coeff)
            allowable_increase = min(increases) if increases else float("inf")
            allowable_decrease = min(decreases) if decreases else float("inf")
            ranges[f"constraint_{idx + 1}"] = {
                "allowable_increase": float(allowable_increase),
                "allowable_decrease": float(allowable_decrease),
            }
        return ranges

    def _compute_dual_report(
        self,
        standard: _StandardForm,
        basis: Sequence[int],
        slack_values: Dict[str, float],
        surplus_values: Dict[str, float],
    ) -> DualReport:
        m = standard.constraint_matrix.shape[0]
        basis_matrix = standard.constraint_matrix[:, basis]
        try:
            b_inv = np.linalg.inv(basis_matrix)
        except np.linalg.LinAlgError:
            return DualReport()

        cost_vector = standard.objective[basis]
        dual_variables = cost_vector @ b_inv
        dual_dict = {f"y{i + 1}": float(dual_variables[i]) for i in range(m)}

        complementary_slackness: Dict[str, bool] = {}
        for idx in range(m):
            dual_name = f"y{idx + 1}"
            dual_value = dual_dict.get(dual_name, 0.0)
            slack_name = (
                standard.constraint_slack_map[idx]
                if idx < len(standard.constraint_slack_map)
                else None
            )
            surplus_name = (
                standard.constraint_surplus_map[idx]
                if idx < len(standard.constraint_surplus_map)
                else None
            )
            if slack_name:
                value = slack_values.get(slack_name, 0.0)
                complementary_slackness[slack_name] = (
                    abs(dual_value * value) <= self.tolerance
                )
            if surplus_name:
                value = surplus_values.get(surplus_name, 0.0)
                complementary_slackness[surplus_name] = (
                    abs(dual_value * value) <= self.tolerance
                )

        dual_objective = float(dual_variables @ standard.rhs)
        dual_objective *= standard.objective_multiplier

        return DualReport(
            dual_variables=dual_dict,
            dual_objective_value=dual_objective,
            complementary_slackness=complementary_slackness,
        )

    # ---------------------------------------------------------------------
    # Standard form conversion
    # ---------------------------------------------------------------------
    def _to_standard_form(self, lp: LinearProgram) -> _StandardForm:
        n = lp.num_variables
        m = lp.num_constraints
        objective_multiplier = 1.0 if lp.sense == "max" else -1.0
        processed_constraints: List[Tuple[np.ndarray, float, str]] = []

        for constraint in lp.constraints:
            coeffs = np.array(constraint.coefficients, dtype=float)
            rhs = float(constraint.rhs)
            sense = constraint.sense
            if rhs < -self.tolerance:
                coeffs = -coeffs
                rhs = -rhs
                if sense == "<=":
                    sense = ">="
                elif sense == ">=":
                    sense = "<="
            processed_constraints.append((coeffs, rhs, sense))

        slack_count = sum(1 for _, _, s in processed_constraints if s == "<=")
        surplus_count = sum(1 for _, _, s in processed_constraints if s == ">=")
        artificial_count = sum(1 for _, _, s in processed_constraints if s != "<=")

        total_variables = n + slack_count + surplus_count + artificial_count
        matrix = np.zeros((m, total_variables), dtype=float)
        rhs_vector = np.zeros(m, dtype=float)
        variable_names: List[Optional[str]] = [f"x{i + 1}" for i in range(n)] + [
            None
        ] * (total_variables - n)

        slack_names: List[str] = []
        surplus_names: List[str] = []
        artificial_names: List[str] = []
        artificial_indices: List[int] = []
        constraint_slack_map: List[Optional[str]] = []
        constraint_surplus_map: List[Optional[str]] = []
        basis: List[int] = []

        slack_offset = n
        surplus_offset = n + slack_count
        artificial_offset = n + slack_count + surplus_count
        slack_cursor = 0
        surplus_cursor = 0
        artificial_cursor = 0

        for row_index, (coeffs, rhs, sense) in enumerate(processed_constraints):
            matrix[row_index, :n] = coeffs
            rhs_vector[row_index] = rhs

            if sense == "<=":
                slack_col = slack_offset + slack_cursor
                matrix[row_index, slack_col] = 1.0
                name = f"s{len(slack_names) + 1}"
                slack_names.append(name)
                variable_names[slack_col] = name
                basis.append(slack_col)
                slack_cursor += 1
                constraint_slack_map.append(name)
                constraint_surplus_map.append(None)
            elif sense == ">=":
                surplus_col = surplus_offset + surplus_cursor
                matrix[row_index, surplus_col] = -1.0
                surplus_name = f"e{len(surplus_names) + 1}"
                surplus_names.append(surplus_name)
                variable_names[surplus_col] = surplus_name
                surplus_cursor += 1
                artificial_col = artificial_offset + artificial_cursor
                matrix[row_index, artificial_col] = 1.0
                artificial_name = f"a{len(artificial_names) + 1}"
                artificial_names.append(artificial_name)
                variable_names[artificial_col] = artificial_name
                basis.append(artificial_col)
                artificial_cursor += 1
                artificial_indices.append(artificial_col)
                constraint_slack_map.append(None)
                constraint_surplus_map.append(surplus_name)
            else:  # sense == "="
                artificial_col = artificial_offset + artificial_cursor
                matrix[row_index, artificial_col] = 1.0
                artificial_name = f"a{len(artificial_names) + 1}"
                artificial_names.append(artificial_name)
                variable_names[artificial_col] = artificial_name
                basis.append(artificial_col)
                artificial_cursor += 1
                artificial_indices.append(artificial_col)
                constraint_slack_map.append(None)
                constraint_surplus_map.append(None)

        # Fill any remaining unnamed variables (should not happen)
        for idx, name in enumerate(variable_names):
            if name is None:
                variable_names[idx] = f"v{idx + 1}"

        converted_objective = np.zeros(total_variables, dtype=float)
        converted_objective[:n] = np.array(lp.objective, dtype=float) * objective_multiplier

        phase_one_objective = None
        if artificial_count:
            phase_one_objective = np.zeros(total_variables, dtype=float)
            start = n + slack_count + surplus_count
            phase_one_objective[start : start + artificial_count] = -1.0

        return _StandardForm(
            objective_multiplier=objective_multiplier,
            original_variable_count=n,
            constraint_matrix=matrix,
            rhs=rhs_vector,
            objective=converted_objective,
            phase_one_objective=phase_one_objective,
            variable_names=[str(name) for name in variable_names],
            slack_names=slack_names,
            surplus_names=surplus_names,
            artificial_names=artificial_names,
            constraint_slack_map=constraint_slack_map,
            constraint_surplus_map=constraint_surplus_map,
            artificial_index_list=artificial_indices,
            initial_basis=basis,
        )
