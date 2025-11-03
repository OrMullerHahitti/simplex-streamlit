from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Literal, Optional, Sequence, Tuple


ConstraintSense = Literal["<=", ">=", "="]
ObjectiveSense = Literal["max", "min"]
SimplexStatus = Literal["optimal", "unbounded", "infeasible"]


@dataclass(frozen=True)
class Constraint:
    """Represents a single linear constraint."""

    coefficients: Sequence[float]
    rhs: float
    sense: ConstraintSense = "<="

    def validate(self, expected_length: int) -> None:
        if len(self.coefficients) != expected_length:
            raise ValueError(
                "Constraint coefficient length "
                f"{len(self.coefficients)} does not match expected {expected_length}"
            )
        if self.sense not in {"<=", ">=", "="}:
            raise ValueError(f"Unsupported constraint sense: {self.sense}")


@dataclass(frozen=True)
class LinearProgram:
    """Container for the linear program definition."""

    objective: Sequence[float]
    constraints: Sequence[Constraint]
    sense: ObjectiveSense = "max"

    def __post_init__(self) -> None:
        if self.sense not in {"max", "min"}:
            raise ValueError(f"Unsupported objective sense: {self.sense}")
        if not self.constraints:
            raise ValueError("Linear program must contain at least one constraint")
        width = len(self.objective)
        for constraint in self.constraints:
            constraint.validate(width)

    @property
    def num_variables(self) -> int:
        return len(self.objective)

    @property
    def num_constraints(self) -> int:
        return len(self.constraints)


@dataclass(frozen=True)
class SimplexStep:
    """Snapshot of an iteration in the simplex algorithm."""

    iteration: int
    phase: str
    tableau: List[List[float]]
    basis: List[str]
    entering_variable: Optional[str]
    leaving_variable: Optional[str]
    pivot: Optional[Tuple[int, int]]
    objective_value: float
    details: str = ""


@dataclass(frozen=True)
class SensitivityReport:
    """Summarises sensitivity insights derived from the final tableau."""

    shadow_prices: Dict[str, float] = field(default_factory=dict)
    reduced_costs: Dict[str, float] = field(default_factory=dict)
    rhs_ranges: Dict[str, Dict[str, float]] = field(default_factory=dict)


@dataclass(frozen=True)
class DualReport:
    """Captures dual problem insights derived from the primal optimum."""

    dual_variables: Dict[str, float] = field(default_factory=dict)
    dual_objective_value: Optional[float] = None
    complementary_slackness: Dict[str, bool] = field(default_factory=dict)


@dataclass(frozen=True)
class SimplexResult:
    """Final outcome of the simplex algorithm."""

    status: SimplexStatus
    objective_value: Optional[float]
    variable_values: Dict[str, float] = field(default_factory=dict)
    slack_values: Dict[str, float] = field(default_factory=dict)
    surplus_values: Dict[str, float] = field(default_factory=dict)
    steps: Sequence[SimplexStep] = field(default_factory=list)
    sensitivity: SensitivityReport = field(default_factory=SensitivityReport)
    dual: DualReport = field(default_factory=DualReport)
    message: str = ""
