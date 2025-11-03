"""
Session state management for Vibe Simplex UI
"""

import streamlit as st
from typing import Optional
from vibe_simplex.models import LinearProgram, SimplexResult
from vibe_simplex.solver import SimplexDebugger


def initialize_session_state() -> None:
    """Initialize all session state variables."""

    # Theme
    if "theme" not in st.session_state:
        st.session_state.theme = "dark"

    # Current problem
    if "current_problem" not in st.session_state:
        st.session_state.current_problem: Optional[LinearProgram] = None

    # Solver results
    if "current_result" not in st.session_state:
        st.session_state.current_result: Optional[SimplexResult] = None

    # Debugger
    if "debugger" not in st.session_state:
        st.session_state.debugger: Optional[SimplexDebugger] = None

    # Problem history
    if "problem_history" not in st.session_state:
        st.session_state.problem_history = []

    # Input form state
    if "num_variables" not in st.session_state:
        st.session_state.num_variables = 2

    if "num_constraints" not in st.session_state:
        st.session_state.num_constraints = 2

    if "objective_sense" not in st.session_state:
        st.session_state.objective_sense = "max"

    # Debugger navigation state
    if "debugger_cursor" not in st.session_state:
        st.session_state.debugger_cursor = 0


def toggle_theme() -> None:
    """Toggle between dark and light theme."""
    st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"


def save_problem_to_history(problem: LinearProgram, result: SimplexResult) -> None:
    """Save a solved problem to history."""
    import datetime

    entry = {
        "timestamp": datetime.datetime.now(),
        "problem": problem,
        "result": result,
    }
    st.session_state.problem_history.append(entry)


def clear_current_problem() -> None:
    """Clear the current problem and results."""
    st.session_state.current_problem = None
    st.session_state.current_result = None
    st.session_state.debugger = None
    st.session_state.debugger_cursor = 0
