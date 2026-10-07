"""Risk Intelligence Gauges: VaR, Volatility, Sharpe Ratio, Max Drawdown."""

from typing import Optional
import plotly.graph_objects as go
from frontend.styles.theme import get_theme_colors


def render_risk_gauge(
    title: str,
    value: float,
    min_val: float = 0.0,
    max_val: float = 100.0,
    unit: str = "%",
    thresholds: Optional[dict] = None,
    theme_mode: str = "dark",
) -> go.Figure:
    """Render a financial terminal risk gauge with color zones using central theme tokens."""
    t = get_theme_colors(theme_mode)

    # Default color zones
    if not thresholds:
        thresholds = {
            "green": [min_val, min_val + (max_val - min_val) * 0.4],
            "amber": [min_val + (max_val - min_val) * 0.4, min_val + (max_val - min_val) * 0.75],
            "red": [min_val + (max_val - min_val) * 0.75, max_val],
        }

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            number={"suffix": unit, "font": {"family": "JetBrains Mono", "size": 22, "color": t["text_primary"]}},
            title={"text": f"<b>{title}</b>", "font": {"family": "Plus Jakarta Sans", "size": 11, "color": t["text_muted"]}},
            gauge={
                "axis": {"range": [min_val, max_val], "tickcolor": t["text_muted"], "tickfont": {"size": 9, "family": "JetBrains Mono"}},
                "bar": {"color": t["accent_cyan"], "thickness": 0.25},
                "bgcolor": "rgba(255, 255, 255, 0.03)",
                "borderwidth": 1,
                "bordercolor": t["border"],
                "steps": [
                    {"range": thresholds["green"], "color": "rgba(34, 197, 94, 0.25)"},
                    {"range": thresholds["amber"], "color": "rgba(245, 158, 11, 0.25)"},
                    {"range": thresholds["red"], "color": "rgba(239, 68, 68, 0.25)"},
                ],
            },
        )
    )

    fig.update_layout(
        paper_bgcolor=t["paper_bg"],
        plot_bgcolor=t["plot_bg"],
        margin=dict(l=15, r=15, t=30, b=10),
        height=180,
    )
    return fig
