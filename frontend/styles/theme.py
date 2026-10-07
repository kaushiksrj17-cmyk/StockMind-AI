"""Centralized AI Financial Intelligence Design System & Theme Engine.

Defines the institutional palette, design tokens, responsive typography,
CSS injection, and uniform Plotly chart styling for StockMind-AI.
"""

from pathlib import Path
from typing import Any, Dict, Optional
import plotly.graph_objects as go
import streamlit as st

# ==============================================================================
# 1. Primary Design System Palette Tokens
# ==============================================================================
BG_PRIMARY = "#070B14"
BG_SECONDARY = "#0B1220"
BG_TERTIARY = "#111827"
PANEL = "#121B2B"
PANEL_HOVER = "#172338"
BORDER = "#243247"
BORDER_SUBTLE = "rgba(36, 50, 71, 0.6)"

ACCENT_CYAN = "#22D3EE"
ACCENT_BLUE = "#3B82F6"
ACCENT_PURPLE = "#8B5CF6"

BULLISH = "#22C55E"
BULLISH_SOFT = "#163A2A"
BEARISH = "#EF4444"
BEARISH_SOFT = "#3A1A22"

WARNING = "#F59E0B"
WARNING_SOFT = "rgba(245, 158, 11, 0.15)"
NEUTRAL = "#94A3B8"
NEUTRAL_SOFT = "rgba(148, 163, 184, 0.15)"

TEXT_PRIMARY = "#F8FAFC"
TEXT_SECONDARY = "#CBD5E1"
TEXT_MUTED = "#64748B"

# Full Mode Dictionary
THEME_PALETTE = {
    "dark": {
        "bg_primary": BG_PRIMARY,
        "bg_secondary": BG_SECONDARY,
        "bg_tertiary": BG_TERTIARY,
        "panel": PANEL,
        "panel_hover": PANEL_HOVER,
        "border": BORDER,
        "border_subtle": BORDER_SUBTLE,
        "accent_cyan": ACCENT_CYAN,
        "accent_blue": ACCENT_BLUE,
        "accent_purple": ACCENT_PURPLE,
        "bullish": BULLISH,
        "bullish_soft": BULLISH_SOFT,
        "bearish": BEARISH,
        "bearish_soft": BEARISH_SOFT,
        "warning": WARNING,
        "warning_soft": WARNING_SOFT,
        "neutral": NEUTRAL,
        "text_primary": TEXT_PRIMARY,
        "text_secondary": TEXT_SECONDARY,
        "text_muted": TEXT_MUTED,
        "grid_color": "rgba(36, 50, 71, 0.6)",
        "plot_bg": "#0B1220",
        "paper_bg": "#121B2B",
    },
    "light": {
        "bg_primary": "#F8FAFC",
        "bg_secondary": "#F1F5F9",
        "bg_tertiary": "#E2E8F0",
        "panel": "#FFFFFF",
        "panel_hover": "#F8FAFC",
        "border": "#CBD5E1",
        "border_subtle": "rgba(203, 213, 225, 0.8)",
        "accent_cyan": "#0284C7",
        "accent_blue": "#2563EB",
        "accent_purple": "#7C3AED",
        "bullish": "#16A34A",
        "bullish_soft": "#DCFCE7",
        "bearish": "#DC2626",
        "bearish_soft": "#FEE2E2",
        "warning": "#D97706",
        "warning_soft": "#FEF3C7",
        "neutral": "#64748B",
        "text_primary": "#0F172A",
        "text_secondary": "#334155",
        "text_muted": "#64748B",
        "grid_color": "rgba(203, 213, 225, 0.6)",
        "plot_bg": "#FFFFFF",
        "paper_bg": "#FFFFFF",
    },
}


def get_theme_colors(theme_mode: str = "dark") -> Dict[str, str]:
    """Retrieve full palette dictionary for current theme mode."""
    mode = "light" if str(theme_mode).strip().lower() == "light" else "dark"
    return THEME_PALETTE[mode]


# ==============================================================================
# 2. Financial Semantic Signal System
# ==============================================================================
def get_semantic_signal(signal_name: str, theme_mode: str = "dark") -> Dict[str, str]:
    """Return consistent semantic color tokens and glow styling for market signals."""
    t = get_theme_colors(theme_mode)
    s = str(signal_name).upper().strip()

    if s in ("STRONG BULLISH", "STRONG_BULLISH"):
        return {
            "label": "STRONG BULLISH",
            "color": t["bullish"],
            "bg": t["bullish_soft"],
            "border": f"rgba(34, 197, 94, 0.6)",
            "glow": "0 0 16px rgba(34, 197, 94, 0.4)",
            "badge_class": "badge-live",
        }
    elif s in ("BULLISH", "BUY"):
        return {
            "label": "BULLISH",
            "color": t["bullish"],
            "bg": t["bullish_soft"],
            "border": f"rgba(34, 197, 94, 0.4)",
            "glow": "none",
            "badge_class": "badge-live",
        }
    elif s in ("STRONG BEARISH", "STRONG_BEARISH"):
        return {
            "label": "STRONG BEARISH",
            "color": t["bearish"],
            "bg": t["bearish_soft"],
            "border": f"rgba(239, 68, 68, 0.6)",
            "glow": "0 0 16px rgba(239, 68, 68, 0.4)",
            "badge_class": "badge-replay",
        }
    elif s in ("BEARISH", "SELL"):
        return {
            "label": "BEARISH",
            "color": t["bearish"],
            "bg": t["bearish_soft"],
            "border": f"rgba(239, 68, 68, 0.4)",
            "glow": "none",
            "badge_class": "badge-replay",
        }
    elif s in ("WARNING", "HIGH VOLATILITY", "REPLAY"):
        return {
            "label": s,
            "color": t["warning"],
            "bg": t["warning_soft"],
            "border": f"rgba(245, 158, 11, 0.4)",
            "glow": "none",
            "badge_class": "badge-replay",
        }
    elif s in ("HISTORICAL",):
        return {
            "label": "HISTORICAL",
            "color": t["accent_blue"],
            "bg": "rgba(59, 130, 246, 0.12)",
            "border": "rgba(59, 130, 246, 0.4)",
            "glow": "none",
            "badge_class": "badge-historical",
        }
    elif s in ("PREDICTION", "AI PREDICTION"):
        return {
            "label": "PREDICTION",
            "color": t["accent_purple"],
            "bg": "rgba(139, 92, 246, 0.12)",
            "border": "rgba(139, 92, 246, 0.4)",
            "glow": "none",
            "badge_class": "badge-prediction",
        }
    else:  # NEUTRAL
        return {
            "label": "NEUTRAL",
            "color": t["neutral"],
            "bg": t["neutral_soft"],
            "border": f"rgba(148, 163, 184, 0.3)",
            "glow": "none",
            "badge_class": "badge-historical",
        }


# ==============================================================================
# 3. Standardized Plotly Chart Styling
# ==============================================================================
def apply_terminal_chart_theme(
    fig: go.Figure,
    theme_mode: str = "dark",
    height: int = 400,
    title: Optional[str] = None,
    showlegend: bool = True,
) -> go.Figure:
    """Apply uniform institutional terminal styling to any Plotly figure."""
    t = get_theme_colors(theme_mode)

    layout_update = dict(
        paper_bgcolor=t["paper_bg"],
        plot_bgcolor=t["plot_bg"],
        font=dict(family="JetBrains Mono, monospace", color=t["text_secondary"], size=10),
        margin=dict(l=12, r=12, t=36 if title else 14, b=12),
        height=height,
        showlegend=showlegend,
    )

    if title:
        layout_update["title"] = dict(
            text=f"<b>{title}</b>",
            font=dict(size=12, color=t["text_primary"], family="Plus Jakarta Sans"),
            x=0.01,
            xanchor="left",
        )

    if showlegend:
        layout_update["legend"] = dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=9, color=t["text_secondary"]),
            bgcolor="rgba(0,0,0,0)",
        )

    fig.update_layout(**layout_update)
    fig.update_xaxes(
        gridcolor=t["grid_color"],
        showgrid=True,
        zeroline=False,
        tickfont=dict(family="JetBrains Mono", size=9, color=t["text_muted"]),
    )
    fig.update_yaxes(
        gridcolor=t["grid_color"],
        showgrid=True,
        zeroline=False,
        tickfont=dict(family="JetBrains Mono", size=9, color=t["text_muted"]),
    )
    return fig


# ==============================================================================
# 4. Central CSS Injection Engine
# ==============================================================================
def apply_terminal_theme(theme_mode: str = "dark") -> None:
    """Inject custom CSS to create the unified institutional AI terminal aesthetic."""
    css_path = Path(__file__).resolve().parent / "theme.css"
    css_content = ""
    if css_path.exists():
        css_content = css_path.read_text(encoding="utf-8")

    t = get_theme_colors(theme_mode)

    # Inject theme CSS and dynamic mode variables
    injection = f"""
    <style>
        {css_content}

        :root {{
            --bg-primary: {t['bg_primary']};
            --bg-secondary: {t['bg_secondary']};
            --bg-tertiary: {t['bg_tertiary']};
            --panel: {t['panel']};
            --panel-hover: {t['panel_hover']};
            --border: {t['border']};
            --border-subtle: {t['border_subtle']};
            --text-primary: {t['text_primary']};
            --text-secondary: {t['text_secondary']};
            --text-muted: {t['text_muted']};
            --accent-cyan: {t['accent_cyan']};
            --accent-blue: {t['accent_blue']};
            --accent-purple: {t['accent_purple']};
            --bullish: {t['bullish']};
            --bullish-soft: {t['bullish_soft']};
            --bearish: {t['bearish']};
            --bearish-soft: {t['bearish_soft']};
            --warning: {t['warning']};
            --warning-soft: {t['warning_soft']};
            --neutral: {t['neutral']};
        }}

        .stApp {{
            background-color: {t['bg_primary']} !important;
            color: {t['text_primary']} !important;
        }}

        section[data-testid="stSidebar"] {{
            background-color: {t['bg_secondary']} !important;
            border-right: 1px solid {t['border']} !important;
        }}

        /* Sidebar navigation section header styling */
        [data-testid="stNavSectionHeader"],
        [data-testid="stSidebarNav"] [data-testid="stNavSectionHeader"],
        header[data-testid="stNavSectionHeader"] {{
            color: {t['accent_cyan']} !important;
            background: rgba(34, 211, 238, 0.08) !important;
            border-left: 3px solid {t['accent_cyan']} !important;
            border-radius: 4px !important;
            padding: 6px 10px !important;
            margin-top: 14px !important;
            margin-bottom: 6px !important;
            font-size: 0.74rem !important;
            font-weight: 800 !important;
            letter-spacing: 0.09em !important;
            text-transform: uppercase !important;
            display: flex !important;
            align-items: center !important;
            justify-content: space-between !important;
            cursor: pointer !important;
        }}

        [data-testid="stNavSectionHeader"] * {{
            color: {t['accent_cyan']} !important;
            font-weight: 800 !important;
        }}

        [data-testid="stNavSectionHeader"] svg {{
            fill: {t['accent_cyan']} !important;
            color: {t['accent_cyan']} !important;
        }}

        /* Sidebar navigation links */
        [data-testid="stSidebarNavLink"],
        [data-testid="stSidebarNav"] a {{
            color: {t['text_primary']} !important;
            font-size: 0.84rem !important;
            font-weight: 500 !important;
        }}

        [data-testid="stSidebarNavLink"] span,
        [data-testid="stSidebarNavLink"] p {{
            color: {t['text_primary']} !important;
        }}

        [data-testid="stSidebarNavLink"]:hover,
        [data-testid="stSidebarNav"] a:hover {{
            background: rgba(36, 50, 71, 0.7) !important;
            color: #FFFFFF !important;
            border-left: 3px solid {t['accent_cyan']} !important;
        }}

        [data-testid="stSidebarNavLink"][aria-current="page"],
        [data-testid="stSidebarNav"] a[aria-current="page"] {{
            background: rgba(34, 211, 238, 0.15) !important;
            color: {t['accent_cyan']} !important;
            border-left: 3px solid {t['accent_cyan']} !important;
            font-weight: 700 !important;
        }}

        [data-testid="stSidebarNavLink"][aria-current="page"] span,
        [data-testid="stSidebarNavLink"][aria-current="page"] p {{
            color: {t['accent_cyan']} !important;
            font-weight: 700 !important;
        }}
    </style>
    """
    st.markdown(injection, unsafe_allow_html=True)
