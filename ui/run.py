"""
Launch script for Vibe Simplex UI

Usage:
    uv run python ui/run.py
    OR
    uv run streamlit run ui/Home.py
"""

import subprocess
import sys
from pathlib import Path

def main():
    """Launch the Streamlit application."""

    ui_dir = Path(__file__).parent
    home_file = ui_dir / "Home.py"

    if not home_file.exists():
        print(f"Error: {home_file} not found!")
        sys.exit(1)

    print("🚀 Launching Vibe Simplex UI...")
    print(f"📂 UI Directory: {ui_dir}")
    print(f"📄 Home File: {home_file}")
    print("\n" + "="*60)
    print("Starting Streamlit server...")
    print("="*60 + "\n")

    # Launch streamlit
    subprocess.run([
        "streamlit", "run", str(home_file),
        "--theme.base", "dark",
        "--theme.primaryColor", "#00D4FF",
    ])

if __name__ == "__main__":
    main()
