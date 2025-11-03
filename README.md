## Vibe Simplex Studio

Interactive simplex playground featuring step-by-step debugging, sensitivity insights, dual analysis, and 2D visualisation.

### Highlights
- **Primal simplex solver** with detailed iteration snapshots and optional debugger.
- **Full constraint support** enabling maximisation or minimisation with ≤, ≥, and = constraints via a two-phase simplex method.
- **Appealing Streamlit UI** for configuring problems, replaying pivots, and exploring results.
- **Sensitivity analysis** exposing shadow prices, reduced costs, and RHS stability ranges.
- **Dual analysis** with complementary slackness checks.
- **2D geometry view** for problems with two decision variables, including the optimal objective iso-line.
- **Comprehensive tests** (`pytest` + coverage) delivering >95% line coverage.

### Getting Started
1. Ensure [`uv`](https://github.com/astral-sh/uv) is installed (the environment is pre-initialised).
2. Install dependencies and sync the virtual environment:
   ```bash
   uv sync
   ```
3. Run the terminal demo (prints the simplex trace for a sample LP):
   ```bash
   uv run python main.py
   ```
4. Launch the interactive UI:
   ```bash
   uv run streamlit run app.py
   ```

### Using the UI
1. Configure the number of variables and constraints from the sidebar.
2. Enter objective coefficients and constraint rows (supports `≤`, `≥`, and `=` with non-negative decision variables).
3. Press **Run Simplex** to compute the solution.
4. Explore:
   - **Optimal Solution** metrics cards.
   - **Step-by-Step Debugger** slider to replay each pivot, with phase-aware narratives, highlighted pivots, and objective progression.
   - **2D Geometry** (when applicable) displaying feasible region, constraints, and optimal iso-line.
   - **Sensitivity & Dual Analysis** tabs for shadow prices, reduced costs, RHS ranges, dual variables, and complementary slackness verdicts.

### Running Tests
Execute the full suite with coverage reporting:
```bash
uv run pytest --cov=vibe_simplex --cov-report=term-missing
```

### Project Layout
- `app.py` – Streamlit application entry point.
- `main.py` – CLI sample run that prints simplex steps.
- `src/vibe_simplex/` – Solver implementation and data models.
- `tests/` – Pytest-based coverage-focused unit tests.
