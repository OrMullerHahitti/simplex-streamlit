from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from vibe_simplex.models import Constraint, LinearProgram
from vibe_simplex.solver import SimplexDebugger, SimplexSolver


def main() -> None:
    """Run a quick demo of the simplex solver in the terminal."""
    lp = LinearProgram(
        objective=[3.0, 5.0],
        constraints=[
            Constraint([2.0, 3.0], 8.0),
            Constraint([1.0, 1.0], 4.0),
        ],
    )
    solver = SimplexSolver()
    debugger = SimplexDebugger()
    result = solver.solve(lp, debugger)

    print("Status:", result.status)
    print("Optimal objective value:", result.objective_value)
    print("Variables:", result.variable_values)
    print("Slack variables:", result.slack_values)
    if result.surplus_values:
        print("Surplus variables:", result.surplus_values)
    print("\nSimplex iterations:")
    for step in debugger.steps():
        pivot = f" pivot@{step.pivot}" if step.pivot else ""
        print(
            f"  [{step.phase}] iter {step.iteration}: obj={step.objective_value:.3f} "
            f"enter={step.entering_variable or '-'} "
            f"leave={step.leaving_variable or '-'}{pivot}"
        )
        if step.details:
            print(f"      {step.details}")
    print(
        "\nTip: launch the interactive experience with "
        "`streamlit run app.py`"
    )


if __name__ == "__main__":
    main()
