"""Technical Indicator Visualizations: RSI, MACD, and Bollinger Bands."""

import math
from typing import Any, Dict, List
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from frontend.styles.theme import get_theme_colors


def _calculate_rsi(closes: List[float], period: int = 14) -> List[float]:
    """Calculate Relative Strength Index (RSI)."""
    if len(closes) <= period:
        return [50.0] * len(closes)

    deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    gains = [max(d, 0.0) for d in deltas]
    losses = [max(-d, 0.0) for d in deltas]

    rsi_values = [50.0] * (period + 1)
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period

    for i in range(period, len(deltas)):
        gain = gains[i]
        loss = losses[i]
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period

        if avg_loss == 0:
            rsi_values.append(100.0)
        else:
            rs = avg_gain / avg_loss
            rsi_values.append(100.0 - (100.0 / (1.0 + rs)))

    return rsi_values[: len(closes)]


def render_rsi_chart(
    candles: List[Dict[str, Any]],
    symbol: str = "RELIANCE",
    period: int = 14,
    theme_mode: str = "dark",
) -> go.Figure:
    """Render interactive RSI oscillator with 70/30 thresholds."""
    t = get_theme_colors(theme_mode)

    if not candles:
        fig = go.Figure()
        fig.update_layout(paper_bgcolor=t["paper_bg"], plot_bgcolor=t["plot_bg"], height=220)
        return fig

    timestamps = [c.get("timestamp") for c in candles]
    closes = [float(c.get("close", 0.0)) for c in candles]
    rsi_vals = _calculate_rsi(closes, period=period)

    fig = go.Figure()

    # Overbought zone (70-100)
    fig.add_hrect(
        y0=70,
        y1=100,
        fillcolor="rgba(239, 68, 68, 0.08)",
        line_width=0,
        annotation_text="Overbought (70)",
        annotation_position="top left",
        annotation_font_size=9,
        annotation_font_color=t["bearish"],
    )
    # Oversold zone (0-30)
    fig.add_hrect(
        y0=0,
        y1=30,
        fillcolor="rgba(34, 197, 94, 0.08)",
        line_width=0,
        annotation_text="Oversold (30)",
        annotation_position="bottom left",
        annotation_font_size=9,
        annotation_font_color=t["bullish"],
    )

    # Threshold dashed lines
    fig.add_hline(y=70, line_dash="dash", line_color="rgba(239, 68, 68, 0.4)", line_width=1)
    fig.add_hline(y=50, line_dash="dot", line_color="rgba(148, 163, 184, 0.3)", line_width=1)
    fig.add_hline(y=30, line_dash="dash", line_color="rgba(34, 197, 94, 0.4)", line_width=1)

    # RSI Curve
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=rsi_vals,
            name=f"RSI ({period})",
            line=dict(color=t["accent_purple"], width=2),
        )
    )

    current_rsi = rsi_vals[-1] if rsi_vals else 50.0
    fig.update_layout(
        title=dict(
            text=f"<b>RSI ({period})</b> · Current: <b>{current_rsi:.1f}</b>",
            font=dict(size=11, color=t["text_primary"], family="Plus Jakarta Sans"),
            x=0.01,
            xanchor="left",
        ),
        paper_bgcolor=t["paper_bg"],
        plot_bgcolor=t["plot_bg"],
        font=dict(family="JetBrains Mono, monospace", color=t["text_secondary"], size=10),
        yaxis=dict(range=[0, 100], zeroline=False, gridcolor=t["grid_color"], tickfont=dict(size=9, color=t["text_muted"])),
        xaxis=dict(showgrid=True, gridcolor=t["grid_color"], tickfont=dict(size=9, color=t["text_muted"])),
        margin=dict(l=10, r=10, t=30, b=10),
        height=220,
        showlegend=False,
    )
    return fig


def render_macd_chart(
    candles: List[Dict[str, Any]],
    theme_mode: str = "dark",
) -> go.Figure:
    """Render MACD (12, 26, 9) oscillator and histogram."""
    t = get_theme_colors(theme_mode)

    if not candles or len(candles) < 26:
        fig = go.Figure()
        fig.update_layout(paper_bgcolor=t["paper_bg"], plot_bgcolor=t["plot_bg"], height=220)
        return fig

    timestamps = [c.get("timestamp") for c in candles]
    closes = [float(c.get("close", 0.0)) for c in candles]

    # Calculate EMA 12 and EMA 26
    def calc_ema(values: List[float], span: int) -> List[float]:
        alpha = 2.0 / (span + 1.0)
        ema = [values[0]]
        for v in values[1:]:
            ema.append(alpha * v + (1.0 - alpha) * ema[-1])
        return ema

    ema12 = calc_ema(closes, 12)
    ema26 = calc_ema(closes, 26)
    macd_line = [e12 - e26 for e12, e26 in zip(ema12, ema26)]
    signal_line = calc_ema(macd_line, 9)
    hist = [m - s for m, s in zip(macd_line, signal_line)]

    fig = go.Figure()

    # Histogram
    hist_colors = [
        t["bullish"] if h >= 0 else t["bearish"] for h in hist
    ]
    fig.add_trace(
        go.Bar(
            x=timestamps,
            y=hist,
            name="Histogram",
            marker_color=hist_colors,
        )
    )

    # MACD & Signal Lines
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=macd_line,
            name="MACD Line",
            line=dict(color=t["accent_cyan"], width=1.5),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=signal_line,
            name="Signal (9)",
            line=dict(color=t["warning"], width=1.5, dash="dot"),
        )
    )

    fig.update_layout(
        title=dict(
            text="<b>MACD (12, 26, 9)</b> Oscillator",
            font=dict(size=11, color=t["text_primary"], family="Plus Jakarta Sans"),
            x=0.01,
            xanchor="left",
        ),
        paper_bgcolor=t["paper_bg"],
        plot_bgcolor=t["plot_bg"],
        font=dict(family="JetBrains Mono, monospace", color=t["text_secondary"], size=10),
        xaxis=dict(showgrid=True, gridcolor=t["grid_color"], tickfont=dict(size=9, color=t["text_muted"])),
        yaxis=dict(showgrid=True, gridcolor=t["grid_color"], zeroline=True, zerolinecolor=t["border"], tickfont=dict(size=9, color=t["text_muted"])),
        margin=dict(l=10, r=10, t=30, b=10),
        height=220,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor="rgba(0,0,0,0)", font=dict(size=9, color=t["text_secondary"])),
    )
    return fig


def render_bollinger_bands(
    candles: List[Dict[str, Any]],
    symbol: str = "RELIANCE",
    period: int = 20,
    std_dev: float = 2.0,
    theme_mode: str = "dark",
) -> go.Figure:
    """Render price chart with Bollinger Bands (Upper, SMA, Lower)."""
    t = get_theme_colors(theme_mode)

    if not candles:
        fig = go.Figure()
        fig.update_layout(paper_bgcolor=t["paper_bg"], plot_bgcolor=t["plot_bg"], height=350)
        return fig

    timestamps = [c.get("timestamp") for c in candles]
    closes = [float(c.get("close", 0.0)) for c in candles]

    upper_band = []
    lower_band = []
    mid_band = []

    for i in range(len(closes)):
        if i >= period - 1:
            window = closes[i - period + 1 : i + 1]
            mean = sum(window) / period
            variance = sum((x - mean) ** 2 for x in window) / period
            stdev = math.sqrt(variance)
            mid_band.append(mean)
            upper_band.append(mean + std_dev * stdev)
            lower_band.append(mean - std_dev * stdev)
        else:
            mid_band.append(None)
            upper_band.append(None)
            lower_band.append(None)

    fig = go.Figure()

    # Price Line
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=closes,
            name=f"{symbol} Close",
            line=dict(color=t["text_primary"], width=2),
        )
    )

    # Upper Band
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=upper_band,
            name="Upper Band (+2σ)",
            line=dict(color="rgba(34, 211, 238, 0.4)", width=1, dash="dash"),
        )
    )

    # Lower Band with fill to upper
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=lower_band,
            name="Lower Band (-2σ)",
            fill="tonexty",
            fillcolor="rgba(34, 211, 238, 0.05)",
            line=dict(color="rgba(34, 211, 238, 0.4)", width=1, dash="dash"),
        )
    )

    # Mid Band (SMA)
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=mid_band,
            name=f"Middle ({period} SMA)",
            line=dict(color=t["warning"], width=1.5),
        )
    )

    fig.update_layout(
        title=dict(
            text=f"<b>{symbol}</b> · Bollinger Bands ({period}, {std_dev}σ)",
            font=dict(size=12, color=t["text_primary"], family="Plus Jakarta Sans"),
            x=0.01,
            xanchor="left",
        ),
        paper_bgcolor=t["paper_bg"],
        plot_bgcolor=t["plot_bg"],
        font=dict(family="JetBrains Mono, monospace", color=t["text_secondary"], size=10),
        xaxis=dict(showgrid=True, gridcolor=t["grid_color"], tickfont=dict(size=9, color=t["text_muted"])),
        yaxis=dict(showgrid=True, gridcolor=t["grid_color"], title=dict(text="Price (₹)", font=dict(size=9, color=t["text_muted"])), tickfont=dict(size=9, color=t["text_muted"])),
        margin=dict(l=10, r=10, t=35, b=10),
        height=350,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor="rgba(0,0,0,0)", font=dict(size=9, color=t["text_secondary"])),
    )
    return fig


def render_ai_score_gauge(
    score: float,
    classification: str = "NEUTRAL",
    theme_mode: str = "dark",
) -> go.Figure:
    """Render high-conviction institutional gauge chart for the 0-100 AI Technical Score."""
    t = get_theme_colors(theme_mode)

    if score >= 80:
        bar_color = t["bullish"]
    elif score >= 60:
        bar_color = t["accent_cyan"]
    elif score >= 40:
        bar_color = t["warning"]
    elif score >= 20:
        bar_color = "#F97316"
    else:
        bar_color = t["bearish"]

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            domain={"x": [0, 1], "y": [0, 1]},
            number={"suffix": "/100", "font": {"family": "JetBrains Mono", "size": 30, "color": bar_color}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": t["text_muted"], "tickfont": {"size": 9, "family": "JetBrains Mono"}},
                "bar": {"color": bar_color, "thickness": 0.28},
                "bgcolor": "rgba(255,255,255,0.03)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 20], "color": "rgba(239, 68, 68, 0.15)"},
                    {"range": [20, 40], "color": "rgba(249, 115, 22, 0.12)"},
                    {"range": [40, 60], "color": "rgba(245, 158, 11, 0.12)"},
                    {"range": [60, 80], "color": "rgba(34, 211, 238, 0.12)"},
                    {"range": [80, 100], "color": "rgba(34, 197, 94, 0.15)"},
                ],
                "threshold": {
                    "line": {"color": t["text_primary"], "width": 3},
                    "thickness": 0.8,
                    "value": score,
                },
            },
        )
    )

    fig.update_layout(
        paper_bgcolor=t["paper_bg"],
        plot_bgcolor=t["plot_bg"],
        font=dict(family="JetBrains Mono, monospace", color=t["text_secondary"]),
        height=210,
        margin=dict(l=15, r=15, t=20, b=15),
    )
    return fig


def render_comprehensive_indicator_chart(
    analysis: Dict[str, Any],
    theme_mode: str = "dark",
) -> go.Figure:
    """Render interactive candlestick chart with overlays for SMA 20/50/200, EMA 21, VWAP, Bollinger Bands, and Key Levels."""
    t = get_theme_colors(theme_mode)

    chart_data = analysis.get("chart_data", {})
    if not chart_data or not chart_data.get("dates"):
        fig = go.Figure()
        fig.update_layout(paper_bgcolor=t["paper_bg"], plot_bgcolor=t["plot_bg"], height=450)
        return fig

    dates = chart_data.get("dates", [])
    open_prices = chart_data.get("open", [])
    high_prices = chart_data.get("high", [])
    low_prices = chart_data.get("low", [])
    close_prices = chart_data.get("close", [])
    vol_values = chart_data.get("volume", [])

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.78, 0.22],
    )

    # 1. Candlestick trace
    fig.add_trace(
        go.Candlestick(
            x=dates,
            open=open_prices,
            high=high_prices,
            low=low_prices,
            close=close_prices,
            name="OHLC Bars",
            increasing_line_color=t["bullish"],
            decreasing_line_color=t["bearish"],
        ),
        row=1,
        col=1,
    )

    # 2. Moving Average Overlays
    if chart_data.get("sma_20"):
        fig.add_trace(
            go.Scatter(x=dates, y=chart_data["sma_20"], name="SMA 20", line=dict(color=t["accent_cyan"], width=1.5)),
            row=1,
            col=1,
        )
    if chart_data.get("sma_50"):
        fig.add_trace(
            go.Scatter(x=dates, y=chart_data["sma_50"], name="SMA 50", line=dict(color=t["warning"], width=1.5)),
            row=1,
            col=1,
        )
    if chart_data.get("sma_200"):
        fig.add_trace(
            go.Scatter(x=dates, y=chart_data["sma_200"], name="SMA 200", line=dict(color=t["accent_purple"], width=2.0)),
            row=1,
            col=1,
        )
    if chart_data.get("ema_21"):
        fig.add_trace(
            go.Scatter(x=dates, y=chart_data["ema_21"], name="EMA 21", line=dict(color=t["accent_blue"], width=1.2, dash="dash")),
            row=1,
            col=1,
        )
    if chart_data.get("vwap"):
        fig.add_trace(
            go.Scatter(x=dates, y=chart_data["vwap"], name="VWAP", line=dict(color="#F97316", width=1.5, dash="dot")),
            row=1,
            col=1,
        )

    # 3. Bollinger Bands (Upper, Lower)
    if chart_data.get("bb_upper") and chart_data.get("bb_lower"):
        fig.add_trace(
            go.Scatter(x=dates, y=chart_data["bb_upper"], name="BB Upper (2σ)", line=dict(color="rgba(255,255,255,0.25)", width=1, dash="dot")),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(x=dates, y=chart_data["bb_lower"], name="BB Lower (2σ)", line=dict(color="rgba(255,255,255,0.25)", width=1, dash="dot")),
            row=1,
            col=1,
        )

    # 4. Support and Resistance Pivot Lines
    sr = analysis.get("support_resistance", {})
    nearest_sup = sr.get("nearest_support")
    nearest_res = sr.get("nearest_resistance")
    if nearest_sup:
        fig.add_hline(
            y=nearest_sup,
            line=dict(color="rgba(34, 197, 94, 0.6)", width=1.2, dash="dash"),
            annotation_text=f"Support ₹{nearest_sup:,.2f}",
            annotation_position="top left",
            annotation_font_size=9,
            annotation_font_color=t["bullish"],
            row=1,
            col=1,
        )
    if nearest_res:
        fig.add_hline(
            y=nearest_res,
            line=dict(color="rgba(239, 68, 68, 0.6)", width=1.2, dash="dash"),
            annotation_text=f"Resistance ₹{nearest_res:,.2f}",
            annotation_position="bottom left",
            annotation_font_size=9,
            annotation_font_color=t["bearish"],
            row=1,
            col=1,
        )

    # 5. Volume Bar subplot
    vol_colors = [
        t["bullish"] if close_prices[i] >= open_prices[i] else t["bearish"]
        for i in range(len(close_prices))
    ]
    fig.add_trace(
        go.Bar(
            x=dates,
            y=vol_values,
            name="Volume",
            marker_color=vol_colors,
            opacity=0.6,
        ),
        row=2,
        col=1,
    )

    # Layout styling
    sym = analysis.get("symbol", "EQUITY")
    price = analysis.get("current_price", 0.0)
    chg = analysis.get("price_change_pct", 0.0)
    fig.update_layout(
        title=dict(
            text=f"<b>{sym}</b> · ₹{price:,.2f} ({chg:+.2f}%) · Institutional Technical Chart",
            font=dict(size=12, color=t["text_primary"], family="Plus Jakarta Sans"),
            x=0.01,
            xanchor="left",
        ),
        paper_bgcolor=t["paper_bg"],
        plot_bgcolor=t["plot_bg"],
        font=dict(family="JetBrains Mono, monospace", color=t["text_secondary"], size=10),
        xaxis=dict(showgrid=True, gridcolor=t["grid_color"], rangeslider=dict(visible=False), tickfont=dict(size=9, color=t["text_muted"])),
        xaxis2=dict(showgrid=True, gridcolor=t["grid_color"], tickfont=dict(size=9, color=t["text_muted"])),
        yaxis=dict(showgrid=True, gridcolor=t["grid_color"], title=dict(text="Price (₹)", font=dict(size=9, color=t["text_muted"])), tickfont=dict(size=9, color=t["text_muted"])),
        yaxis2=dict(showgrid=False, title=dict(text="Volume", font=dict(size=9, color=t["text_muted"])), tickfont=dict(size=9, color=t["text_muted"])),
        margin=dict(l=10, r=10, t=35, b=10),
        height=480,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor="rgba(0,0,0,0)", font=dict(size=9, color=t["text_secondary"])),
    )
    return fig
