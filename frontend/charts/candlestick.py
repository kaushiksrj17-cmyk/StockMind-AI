"""Interactive Plotly Candlestick & Volume Chart with Moving Averages."""

from typing import Any, Dict, List
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from frontend.styles.theme import get_theme_colors


def render_candlestick_chart(
    candles: List[Dict[str, Any]],
    symbol: str = "RELIANCE",
    timeframe: str = "1d",
    show_sma: bool = True,
    theme_mode: str = "dark",
) -> go.Figure:
    """Generate an institutional-grade candlestick and volume subplot chart."""
    t = get_theme_colors(theme_mode)

    if not candles:
        # Return empty placeholder figure
        fig = go.Figure()
        fig.add_annotation(
            text="No historical candle data available for this instrument/timeframe",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=13, color=t["text_muted"], family="JetBrains Mono"),
        )
        fig.update_layout(
            paper_bgcolor=t["paper_bg"],
            plot_bgcolor=t["plot_bg"],
            height=450,
            margin=dict(l=10, r=10, t=30, b=10),
        )
        return fig

    # Extract series
    timestamps = [c.get("timestamp") for c in candles]
    opens = [float(c.get("open", 0.0)) for c in candles]
    highs = [float(c.get("high", 0.0)) for c in candles]
    lows = [float(c.get("low", 0.0)) for c in candles]
    closes = [float(c.get("close", 0.0)) for c in candles]
    volumes = [float(c.get("volume", 0.0)) for c in candles]

    # Calculate SMAs if requested
    sma20 = []
    sma50 = []
    for i in range(len(closes)):
        if i >= 19:
            sma20.append(sum(closes[i - 19 : i + 1]) / 20.0)
        else:
            sma20.append(None)

        if i >= 49:
            sma50.append(sum(closes[i - 49 : i + 1]) / 50.0)
        else:
            sma50.append(None)

    # Subplot with 2 rows: Candlestick (78%) and Volume (22%)
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=(None, None),
        row_width=[0.22, 0.78],
    )

    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=timestamps,
            open=opens,
            high=highs,
            low=lows,
            close=closes,
            name=f"{symbol} Price",
            increasing_line_color=t["bullish"],
            increasing_fillcolor=t["bullish"],
            decreasing_line_color=t["bearish"],
            decreasing_fillcolor=t["bearish"],
        ),
        row=1,
        col=1,
    )

    # Moving Averages
    if show_sma:
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=sma20,
                name="EMA 20",
                line=dict(color=t["accent_cyan"], width=1.5),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=sma50,
                name="SMA 50",
                line=dict(color=t["warning"], width=1.5, dash="dot"),
            ),
            row=1,
            col=1,
        )

    # Volume Bar Chart
    vol_colors = [
        "rgba(34, 197, 94, 0.45)" if closes[i] >= opens[i] else "rgba(239, 68, 68, 0.45)"
        for i in range(len(closes))
    ]
    fig.add_trace(
        go.Bar(
            x=timestamps,
            y=volumes,
            name="Volume",
            marker_color=vol_colors,
        ),
        row=2,
        col=1,
    )

    # Institutional Terminal Layout
    fig.update_layout(
        title=dict(
            text=f"<b>{symbol}</b> · {timeframe.upper()} Candlestick & Volume Profile",
            font=dict(size=12, color=t["text_primary"], family="Plus Jakarta Sans"),
            x=0.01,
            xanchor="left",
        ),
        paper_bgcolor=t["paper_bg"],
        plot_bgcolor=t["plot_bg"],
        font=dict(family="JetBrains Mono, monospace", color=t["text_secondary"], size=10),
        xaxis_rangeslider_visible=False,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=9, color=t["text_secondary"]),
            bgcolor="rgba(0,0,0,0)",
        ),
        margin=dict(l=10, r=10, t=36, b=10),
        height=500,
    )

    fig.update_xaxes(
        gridcolor=t["grid_color"],
        showgrid=True,
        zeroline=False,
        tickfont=dict(family="JetBrains Mono", size=9, color=t["text_muted"]),
        row=1,
        col=1,
    )
    fig.update_xaxes(
        gridcolor=t["grid_color"],
        showgrid=True,
        zeroline=False,
        tickfont=dict(family="JetBrains Mono", size=9, color=t["text_muted"]),
        row=2,
        col=1,
    )
    fig.update_yaxes(
        gridcolor=t["grid_color"],
        showgrid=True,
        zeroline=False,
        title=dict(text="Price (₹)", font=dict(size=9, color=t["text_muted"])),
        tickfont=dict(family="JetBrains Mono", size=9, color=t["text_muted"]),
        row=1,
        col=1,
    )
    fig.update_yaxes(
        gridcolor=t["grid_color"],
        showgrid=True,
        zeroline=False,
        title=dict(text="Vol", font=dict(size=9, color=t["text_muted"])),
        tickfont=dict(family="JetBrains Mono", size=9, color=t["text_muted"]),
        row=2,
        col=1,
    )

    return fig
