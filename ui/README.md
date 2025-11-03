# Vibe Simplex UI

Professional web interface for the Vibe Simplex linear programming solver.

## Features

- 📝 **Problem Input** - Intuitive form-based LP problem definition
- 🚀 **Solver** - Run simplex algorithm with comprehensive results
- 📊 **Analysis** - Sensitivity analysis and dual problem insights
- 🔍 **Debugger** - Step-by-step iteration exploration
- 📚 **Examples** - Pre-built LP problems library
- 🎨 **Themes** - Dark and light mode support
- 📈 **Visualization** - Interactive 2D constraint plots (for 2-variable problems)

## Quick Start

### Running the Application

From the project root directory:

```bash
# Using Streamlit directly
uv run streamlit run ui/Home.py

# Or using the run script
uv run python ui/run.py
```

The application will open in your default web browser at `http://localhost:8501`

### Using the Application

1. **Define a Problem**
   - Go to "Problem Input" page
   - Set number of variables and constraints
   - Enter objective function coefficients
   - Define constraints
   - Click "Create Problem"

2. **Solve the Problem**
   - Navigate to "Solver" page
   - Configure solver settings (optional)
   - Click "Run Solver"
   - View comprehensive results

3. **Analyze the Solution**
   - Go to "Analysis" page for:
     - Sensitivity analysis (shadow prices, reduced costs, ranges)
     - Dual problem analysis
     - 2D visualization (for 2-variable problems)

4. **Debug Step-by-Step**
   - Visit "Debugger" page
   - Navigate through each simplex iteration
   - View tableau transformations
   - Understand the algorithm's progression

5. **Try Examples**
   - Browse "Examples" page
   - Load pre-built problems
   - Learn from classic LP formulations

## Project Structure

```
ui/
├── Home.py                 # Landing page
├── pages/                  # Multi-page app pages
│   ├── 1_Problem_Input.py  # LP problem input form
│   ├── 2_Solver.py         # Solver execution and results
│   ├── 3_Analysis.py       # Sensitivity and dual analysis
│   ├── 4_Debugger.py       # Step-by-step debugger
│   └── 5_Examples.py       # Example problems library
├── utils/                  # Utility modules
│   ├── theme.py            # Theme management and custom CSS
│   ├── state.py            # Session state management
│   └── visualization.py    # Plotting utilities
├── assets/                 # Static assets (if needed)
├── tests/                  # UI tests
├── run.py                  # Launch script
└── README.md              # This file
```

## Features in Detail

### Theme Support

Toggle between dark and light themes using the button in the sidebar. Both themes are professionally designed with:
- Consistent color palettes
- Smooth transitions
- Optimized contrast for readability
- Custom styled components

### 2D Visualization

For problems with exactly 2 variables, the Analysis page provides:
- Interactive Plotly charts
- Constraint lines
- Shaded feasible region
- Corner points
- Optimal point highlighting
- Iso-profit lines

### Step-by-Step Debugger

The debugger records each simplex iteration and allows you to:
- Navigate forward/backward through iterations
- View complete tableau at each step
- See entering and leaving variables
- Understand pivot operations
- Track objective value progression

### Sensitivity Analysis

Comprehensive sensitivity analysis including:
- **Shadow Prices**: Marginal value of each resource
- **Reduced Costs**: Cost of bringing non-basic variables into the basis
- **RHS Ranges**: Allowable changes in constraint right-hand sides
- **Visual Range Charts**: Graphical representation of sensitivity ranges

### Dual Analysis

Full dual problem analysis with:
- Dual variable values
- Dual objective value
- Duality gap verification
- Complementary slackness conditions

## Current Limitations

- **Maximization Only**: Current version supports maximization problems only
- **≤ Constraints**: Only less-than-or-equal-to constraints are supported
- **Non-negative RHS**: All right-hand side values must be non-negative
- **2D Visualization**: Graphical plots only available for 2-variable problems

## Tips for Best Experience

1. **Start with Examples**: Load a pre-built problem to get familiar with the interface
2. **Enable Step Recording**: Always enable when solving to use the debugger
3. **Try 2D Problems First**: To see the full visualization capabilities
4. **Explore Analysis**: Don't skip the sensitivity and dual analysis pages
5. **Use Dark Mode**: Optimized for extended use

## Troubleshooting

### Application won't start
- Ensure you're in the project root directory
- Check that all dependencies are installed: `uv sync`
- Verify Streamlit is installed: `uv run streamlit --version`

### Import errors
- Make sure you're running from the project root
- The `src/` directory must be in the parent of `ui/`

### Visualization not showing
- Check that the problem has exactly 2 variables
- Ensure Plotly is installed
- Try refreshing the page

## Development

To modify or extend the UI:

1. **Add new pages**: Create `N_Page_Name.py` in `pages/` directory
2. **Modify theme**: Edit `utils/theme.py`
3. **Add utilities**: Add helper functions to appropriate `utils/*.py` file
4. **Test changes**: Streamlit auto-reloads when files change

## License

Part of the Vibe Simplex project.
