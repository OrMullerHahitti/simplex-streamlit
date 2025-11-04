"""
Theme management and custom CSS for Vibe Simplex UI
"""

import streamlit as st


def get_theme_colors(theme: str = "dark") -> dict:
    """Get color scheme for the specified theme."""

    if theme == "dark":
        return {
            "primary": "#00D4FF",
            "secondary": "#FF6B6B",
            "accent": "#4ECDC4",
            "success": "#51CF66",
            "warning": "#FFD93D",
            "error": "#FF6B6B",
            "background": "#0E1117",
            "background-secondary": "#1A1D24",
            "background-tertiary": "#262730",
            "text-primary": "#FAFAFA",
            "text-secondary": "#A0AEC0",
            "border": "#2D3748",
            "shadow": "rgba(0, 0, 0, 0.3)",
        }
    else:  # light theme
        return {
            "primary": "#2563EB",
            "secondary": "#D946EF",
            "accent": "#0EA5E9",
            "success": "#10B981",
            "warning": "#FBBF24",
            "error": "#EF4444",
            "background": "#FDFDFE",
            "background-secondary": "#F5F7FB",
            "background-tertiary": "#E7ECF4",
            "text-primary": "#0F172A",
            "text-secondary": "#475569",
            "border": "#CBD5F5",
            "shadow": "rgba(15, 23, 42, 0.08)",
        }


def apply_custom_theme() -> None:
    """Apply custom CSS theme based on current session state."""

    theme = st.session_state.get("theme", "dark")
    colors = get_theme_colors(theme)

    custom_css = f"""
    <style>
        /* Root variables */
        :root {{
            --primary: {colors['primary']};
            --secondary: {colors['secondary']};
            --accent: {colors['accent']};
            --success: {colors['success']};
            --warning: {colors['warning']};
            --error: {colors['error']};
            --background: {colors['background']};
            --background-secondary: {colors['background-secondary']};
            --background-tertiary: {colors['background-tertiary']};
            --text-primary: {colors['text-primary']};
            --text-secondary: {colors['text-secondary']};
            --border: {colors['border']};
            --shadow: {colors['shadow']};
        }}

        /* Main background */
        .stApp {{
            background-color: var(--background);
            color: var(--text-primary);
        }}

        /* Sidebar styling */
        [data-testid="stSidebar"] {{
            background-color: var(--background-secondary);
            border-right: 1px solid var(--border);
        }}

        /* Headers */
        h1, h2, h3, h4, h5, h6 {{
            color: var(--text-primary) !important;
            font-weight: 600;
        }}

        h1 {{
            background: linear-gradient(135deg, var(--primary), var(--accent));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }}

        /* Buttons */
        .stButton > button {{
            background: linear-gradient(135deg, var(--primary), var(--accent));
            color: white;
            border: none;
            border-radius: 8px;
            padding: 0.5rem 2rem;
            font-weight: 500;
            transition: all 0.3s ease;
            box-shadow: 0 4px 6px var(--shadow);
        }}

        .stButton > button:hover {{
            transform: translateY(-2px);
            box-shadow: 0 6px 12px var(--shadow);
        }}

        /* Input fields */
        .stTextInput > div > div > input,
        .stNumberInput > div > div > input,
        .stSelectbox > div > div {{
            background-color: var(--background-secondary);
            border: 1px solid var(--border);
            border-radius: 8px;
            color: var(--text-primary);
            transition: all 0.2s ease;
        }}

        .stTextInput > div > div > input:focus,
        .stNumberInput > div > div > input:focus {{
            border-color: var(--primary);
            box-shadow: 0 0 0 2px rgba(0, 212, 255, 0.2);
        }}

        /* Cards/Containers */
        .element-container {{
            transition: all 0.3s ease;
        }}

        /* Metrics */
        [data-testid="stMetricValue"] {{
            color: var(--primary);
            font-size: 2rem;
            font-weight: 600;
        }}

        [data-testid="stMetricLabel"] {{
            color: var(--text-secondary);
            font-size: 0.9rem;
        }}

        /* Expanders */
        .streamlit-expanderHeader {{
            background-color: var(--background-secondary);
            border: 1px solid var(--border);
            border-radius: 8px;
            color: var(--text-primary);
            transition: all 0.2s ease;
        }}

        .streamlit-expanderHeader:hover {{
            border-color: var(--primary);
            background-color: var(--background-tertiary);
        }}

        /* Tables */
        table {{
            background-color: var(--background-secondary);
            border-radius: 8px;
            overflow: hidden;
        }}

        thead tr th {{
            background: linear-gradient(135deg, var(--primary), var(--accent));
            color: white !important;
            font-weight: 600;
            padding: 1rem;
        }}

        tbody tr {{
            border-bottom: 1px solid var(--border);
            transition: all 0.2s ease;
        }}

        tbody tr:hover {{
            background-color: var(--background-tertiary);
        }}

        tbody tr td {{
            padding: 0.75rem 1rem;
            color: var(--text-primary);
        }}

        /* Code blocks */
        code {{
            background-color: var(--background-secondary);
            color: var(--accent);
            padding: 0.2rem 0.4rem;
            border-radius: 4px;
            font-family: 'Monaco', 'Menlo', monospace;
        }}

        pre {{
            background-color: var(--background-secondary);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 1rem;
        }}

        /* Divider */
        hr {{
            border-color: var(--border);
            margin: 2rem 0;
        }}

        /* Success/Warning/Error boxes */
        .stSuccess {{
            background-color: rgba(81, 207, 102, 0.1);
            border-left: 4px solid var(--success);
            border-radius: 4px;
        }}

        .stWarning {{
            background-color: rgba(255, 217, 61, 0.1);
            border-left: 4px solid var(--warning);
            border-radius: 4px;
        }}

        .stError {{
            background-color: rgba(255, 107, 107, 0.1);
            border-left: 4px solid var(--error);
            border-radius: 4px;
        }}

        /* Tooltips */
        .tooltip {{
            position: relative;
            display: inline-block;
            border-bottom: 1px dotted var(--text-secondary);
            cursor: help;
        }}

        /* Custom card styling */
        .custom-card {{
            background-color: var(--background-secondary);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 1.5rem;
            margin: 1rem 0;
            box-shadow: 0 2px 8px var(--shadow);
            transition: all 0.3s ease;
        }}

        .custom-card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 16px var(--shadow);
        }}

        /* Progress bar */
        .stProgress > div > div > div > div {{
            background: linear-gradient(135deg, var(--primary), var(--accent));
        }}

        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 8px;
        }}

        .stTabs [data-baseweb="tab"] {{
            background-color: var(--background-secondary);
            border-radius: 8px 8px 0 0;
            color: var(--text-secondary);
            padding: 0.75rem 1.5rem;
            font-weight: 500;
        }}

        .stTabs [aria-selected="true"] {{
            background-color: var(--background-tertiary);
            color: var(--primary);
            border-bottom: 2px solid var(--primary);
        }}

        /* File uploader */
        [data-testid="stFileUploader"] {{
            background-color: var(--background-secondary);
            border: 2px dashed var(--border);
            border-radius: 8px;
            padding: 2rem;
            transition: all 0.2s ease;
        }}

        [data-testid="stFileUploader"]:hover {{
            border-color: var(--primary);
            background-color: var(--background-tertiary);
        }}

        /* Animations */
        @keyframes fadeIn {{
            from {{
                opacity: 0;
                transform: translateY(10px);
            }}
            to {{
                opacity: 1;
                transform: translateY(0);
            }}
        }}

        .fade-in {{
            animation: fadeIn 0.5s ease;
        }}

        /* Scrollbar */
        ::-webkit-scrollbar {{
            width: 10px;
            height: 10px;
        }}

        ::-webkit-scrollbar-track {{
            background: var(--background);
        }}

        ::-webkit-scrollbar-thumb {{
            background: var(--border);
            border-radius: 5px;
        }}

        ::-webkit-scrollbar-thumb:hover {{
            background: var(--primary);
        }}
    </style>
    """

    st.markdown(custom_css, unsafe_allow_html=True)


def get_theme_toggle() -> None:
    """Render theme toggle button in sidebar."""

    current_theme = st.session_state.get("theme", "dark")

    if current_theme == "dark":
        button_text = "☀️ Light Mode"
        button_help = "Switch to light theme"
    else:
        button_text = "🌙 Dark Mode"
        button_help = "Switch to dark theme"

    if st.button(button_text, help=button_help, use_container_width=True):
        from utils.state import toggle_theme

        toggle_theme()
        st.rerun()
