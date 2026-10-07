"""Indian Stock Market Treemap Heatmap Visualization."""

from typing import Any, Dict, List
import plotly.graph_objects as go
from frontend.styles.theme import get_theme_colors


def render_market_heatmap(
    market_data: List[Dict[str, Any]],
    theme_mode: str = "dark",
) -> go.Figure:
    """Render a sectoral treemap heatmap color-coded by percentage return using unified theme."""
    t = get_theme_colors(theme_mode)
    is_dark = str(theme_mode).lower() == "dark"

    if not market_data:
        # Default representative Indian market sectors and blue-chips
        market_data = [
            {"symbol": "RELIANCE", "name": "Reliance", "sector": "Energy & Petrochem", "market_cap": 1950000, "change_pct": 1.45},
            {"symbol": "TCS", "name": "TCS", "sector": "Information Technology", "market_cap": 1420000, "change_pct": -0.85},
            {"symbol": "INFY", "name": "Infosys", "sector": "Information Technology", "market_cap": 710000, "change_pct": -1.20},
            {"symbol": "HDFCBANK", "name": "HDFC Bank", "sector": "Banking & Finance", "market_cap": 1280000, "change_pct": 0.65},
            {"symbol": "ICICIBANK", "name": "ICICI Bank", "sector": "Banking & Finance", "market_cap": 840000, "change_pct": 2.15},
            {"symbol": "SBIN", "name": "State Bank of India", "sector": "Banking & Finance", "market_cap": 720000, "change_pct": 1.10},
            {"symbol": "TATAMOTORS", "name": "Tata Motors", "sector": "Automobile", "market_cap": 360000, "change_pct": 3.40},
            {"symbol": "BHARTIARTL", "name": "Bharti Airtel", "sector": "Telecommunications", "market_cap": 950000, "change_pct": 0.35},
            {"symbol": "ITC", "name": "ITC Ltd", "sector": "Consumer Goods", "market_cap": 610000, "change_pct": -0.40},
            {"symbol": "LT", "name": "Larsen & Toubro", "sector": "Infrastructure", "market_cap": 480000, "change_pct": 1.80},
        ]

    # Treemap hierarchy: Market -> Sector -> Symbol
    ids = ["Indian Market"]
    labels = ["Indian Market"]
    parents = [""]
    values = [sum(item.get("market_cap", 100000) for item in market_data)]
    colors = [0.0]
    customdata = ["All Sectors"]

    # Sectors
    sectors = sorted(list(set(item.get("sector", "Other") for item in market_data)))
    for sec in sectors:
        sec_items = [i for i in market_data if i.get("sector") == sec]
        sec_cap = sum(i.get("market_cap", 100000) for i in sec_items)
        avg_change = sum(i.get("change_pct", 0.0) * i.get("market_cap", 100000) for i in sec_items) / (sec_cap or 1)
        ids.append(f"sec_{sec}")
        labels.append(sec)
        parents.append("Indian Market")
        values.append(sec_cap)
        colors.append(avg_change)
        customdata.append(f"Sector Avg: {avg_change:+.2f}%")

    # Stocks
    for item in market_data:
        sym = item.get("symbol", "UNKNOWN")
        sec = item.get("sector", "Other")
        cap = item.get("market_cap", 100000)
        chg = item.get("change_pct", 0.0)
        ids.append(f"stock_{sym}")
        labels.append(f"<b>{sym}</b><br>{chg:+.2f}%")
        parents.append(f"sec_{sec}")
        values.append(cap)
        colors.append(chg)
        customdata.append(f"Cap: ₹{cap:,.0f} Cr | Change: {chg:+.2f}%")

    colorscale = [
        [0.0, t["bearish"]],
        [0.35, t["bearish_soft"] if is_dark else "#FEE2E2"],
        [0.5, "#1E293B" if is_dark else "#E2E8F0"],
        [0.65, t["bullish_soft"] if is_dark else "#DCFCE7"],
        [1.0, t["bullish"]],
    ]

    fig = go.Figure(
        go.Treemap(
            ids=ids,
            labels=labels,
            parents=parents,
            values=values,
            branchvalues="total",
            marker=dict(
                colors=colors,
                colorscale=colorscale,
                cmid=0.0,
                cmin=-3.0,
                cmax=3.0,
                colorbar=dict(
                    title=dict(text="Change %", font=dict(size=10, color=t["text_secondary"])),
                    tickfont=dict(size=9, color=t["text_muted"], family="JetBrains Mono"),
                    thickness=12,
                    len=0.7,
                ),
            ),
            hoverinfo="label+value+text",
            textinfo="label",
            textfont=dict(family="JetBrains Mono, monospace", size=12, color=t["text_primary"]),
        )
    )

    fig.update_layout(
        paper_bgcolor=t["paper_bg"],
        plot_bgcolor=t["plot_bg"],
        font=dict(family="JetBrains Mono, monospace", color=t["text_secondary"], size=10),
        margin=dict(l=6, r=6, t=10, b=6),
        height=520,
    )
    return fig
