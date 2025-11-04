"""Utility helpers for the curated example gallery."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

from vibe_simplex.models import Constraint, LinearProgram

ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"
DEFAULT_GALLERY_PATH = ASSETS_DIR / "problem_gallery.json"


@dataclass
class ExampleProblem:
    """Structured metadata describing a showcase linear program."""

    slug: str
    name: str
    description: str
    category: str
    difficulty: str
    tags: Sequence[str]
    sense: str
    objective: Sequence[float]
    constraints: Sequence[Constraint]
    context: str
    expected_solution: str
    insights: Sequence[str]

    def to_linear_program(self) -> LinearProgram:
        """Convert the gallery entry into a LinearProgram instance."""

        return LinearProgram(
            objective=list(self.objective),
            constraints=list(self.constraints),
            sense=self.sense,
        )


def _load_raw_gallery(path: Path) -> List[dict]:
    """Load raw gallery JSON as dictionaries."""

    if not path.exists():
        raise FileNotFoundError(f"Gallery data not found at {path}")

    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def load_problem_gallery(path: Path | None = None) -> List[ExampleProblem]:
    """Return gallery entries as ExampleProblem objects."""

    path = path or DEFAULT_GALLERY_PATH
    raw_entries = _load_raw_gallery(path)

    problems: List[ExampleProblem] = []
    for entry in raw_entries:
        constraints = [
            Constraint(
                coefficients=item["coefficients"],
                rhs=item["rhs"],
                sense=item["sense"],
            )
            for item in entry["constraints"]
        ]

        problems.append(
            ExampleProblem(
                slug=entry["slug"],
                name=entry["name"],
                description=entry["description"],
                category=entry["category"],
                difficulty=entry["difficulty"],
                tags=entry.get("tags", []),
                sense=entry["sense"],
                objective=entry["objective"],
                constraints=constraints,
                context=entry["context"],
                expected_solution=entry["expected_solution"],
                insights=entry.get("insights", []),
            )
        )

    return problems
