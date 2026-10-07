"""Price Action & Candlestick Pattern Recognition Module.

Computes Support and Resistance levels (swing pivots and dynamic clustering),
Fibonacci retracement levels, and algorithmic candlestick pattern recognition
(Doji, Hammer, Inverted Hammer, Shooting Star, Hanging Man, Engulfing,
Morning/Evening Star, Three White Soldiers / Black Crows, Marubozu).
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


def compute_support_resistance(
    df: pd.DataFrame,
    window: int = 10,
    max_levels: int = 5,
    tolerance_pct: float = 0.015,
) -> Dict[str, Any]:
    """Calculate key Support and Resistance levels using local swing extremes and clustering.

    Args:
        df: DataFrame with 'high', 'low', 'close' columns.
        window: Lookback/lookahead window for local pivot detection.
        max_levels: Maximum number of discrete support and resistance levels to return.
        tolerance_pct: Percentage threshold to cluster nearby pivot levels together.

    Returns:
        Dict with current_price, support_levels, resistance_levels, nearest_support, nearest_resistance.
    """
    if len(df) < window * 2:
        current_price = float(df["close"].iloc[-1]) if len(df) > 0 else 0.0
        return {
            "current_price": round(current_price, 2),
            "support_levels": [round(current_price * 0.95, 2)],
            "resistance_levels": [round(current_price * 1.05, 2)],
            "nearest_support": round(current_price * 0.95, 2),
            "nearest_resistance": round(current_price * 1.05, 2),
            "pivot_points": {"pivot": round(current_price, 2)},
        }

    high = df["high"].values
    low = df["low"].values
    close = df["close"].values
    current_price = float(close[-1])

    # 1. Identify local swing highs and swing lows
    swing_highs = []
    swing_lows = []

    for i in range(window, len(df) - 1):
        # Local peak
        if high[i] == np.max(high[max(0, i - window) : min(len(df), i + window + 1)]):
            swing_highs.append(high[i])
        # Local trough
        if low[i] == np.min(low[max(0, i - window) : min(len(df), i + window + 1)]):
            swing_lows.append(low[i])

    # 2. Cluster nearby levels to eliminate noise
    def cluster_levels(levels: List[float], tol: float) -> List[float]:
        if not levels:
            return []
        sorted_levels = sorted(levels)
        clusters: List[List[float]] = []
        for lvl in sorted_levels:
            if not clusters:
                clusters.append([lvl])
            else:
                last_cluster = clusters[-1]
                avg_last = sum(last_cluster) / len(last_cluster)
                if abs(lvl - avg_last) / avg_last <= tol:
                    last_cluster.append(lvl)
                else:
                    clusters.append([lvl])
        # Return median of each cluster
        return [float(np.median(c)) for c in clusters]

    clustered_highs = cluster_levels(swing_highs, tolerance_pct)
    clustered_lows = cluster_levels(swing_lows, tolerance_pct)

    # 3. Categorize into Supports (below current price) and Resistances (above current price)
    raw_supports = [lvl for lvl in (clustered_lows + clustered_highs) if lvl < current_price]
    raw_resistances = [lvl for lvl in (clustered_lows + clustered_highs) if lvl > current_price]

    raw_supports = sorted(list(set(raw_supports)), reverse=True)
    raw_resistances = sorted(list(set(raw_resistances)))

    # Fallbacks if historical range didn't produce levels
    if not raw_supports:
        raw_supports = [float(np.min(low))]
    if not raw_resistances:
        raw_resistances = [float(np.max(high))]

    supports = [round(lvl, 2) for lvl in raw_supports[:max_levels]]
    resistances = [round(lvl, 2) for lvl in raw_resistances[:max_levels]]

    nearest_support = supports[0] if supports else round(current_price * 0.95, 2)
    nearest_resistance = resistances[0] if resistances else round(current_price * 1.05, 2)

    # Standard Floor Trader Pivot Points from latest complete bar
    last_h = float(high[-1])
    last_l = float(low[-1])
    last_c = float(close[-1])
    pivot = (last_h + last_l + last_c) / 3.0
    r1 = (2.0 * pivot) - last_l
    s1 = (2.0 * pivot) - last_h
    r2 = pivot + (last_h - last_l)
    s2 = pivot - (last_h - last_l)

    return {
        "current_price": round(current_price, 2),
        "support_levels": supports,
        "resistance_levels": resistances,
        "nearest_support": nearest_support,
        "nearest_resistance": nearest_resistance,
        "pivot_points": {
            "pivot": round(pivot, 2),
            "r1": round(r1, 2),
            "r2": round(r2, 2),
            "s1": round(s1, 2),
            "s2": round(s2, 2),
        },
    }


def compute_fibonacci_levels(
    df: pd.DataFrame,
    lookback: int = 60,
) -> Dict[str, Any]:
    """Compute standard Fibonacci retracement levels from the swing high and low over lookback.

    Standard ratios: 0.0% (High), 23.6%, 38.2%, 50.0%, 61.8%, 78.6%, 100.0% (Low),
    along with extension levels: 127.2%, 161.8%.
    """
    if len(df) == 0:
        return {}

    sub_df = df.tail(lookback)
    swing_high = float(sub_df["high"].max())
    swing_low = float(sub_df["low"].min())
    current_price = float(sub_df["close"].iloc[-1])
    diff = swing_high - swing_low

    if diff <= 0:
        diff = current_price * 0.05
        swing_high = current_price * 1.025
        swing_low = current_price * 0.975

    # Retracements measuring pullbacks from swing high to swing low
    levels = {
        "0.0% (High)": round(swing_high, 2),
        "23.6%": round(swing_high - 0.236 * diff, 2),
        "38.2%": round(swing_high - 0.382 * diff, 2),
        "50.0%": round(swing_high - 0.500 * diff, 2),
        "61.8% (Golden)": round(swing_high - 0.618 * diff, 2),
        "78.6%": round(swing_high - 0.786 * diff, 2),
        "100.0% (Low)": round(swing_low, 2),
        "127.2% (Ext)": round(swing_high + 0.272 * diff, 2),
        "161.8% (Ext)": round(swing_high + 0.618 * diff, 2),
    }

    # Identify nearest fib level
    closest_level_name = min(levels.keys(), key=lambda k: abs(levels[k] - current_price))

    return {
        "swing_high": round(swing_high, 2),
        "swing_low": round(swing_low, 2),
        "current_price": round(current_price, 2),
        "levels": levels,
        "nearest_level": closest_level_name,
        "nearest_level_price": levels[closest_level_name],
    }


def detect_candlestick_patterns(
    df: pd.DataFrame,
    lookback_bars: int = 5,
) -> List[Dict[str, Any]]:
    """Detect prominent candlestick patterns across recent price action bars.

    Recognized patterns:
    - Doji (Neutral / Indecision)
    - Hammer (Bullish Reversal)
    - Inverted Hammer (Bullish Reversal)
    - Shooting Star (Bearish Reversal)
    - Hanging Man (Bearish Reversal)
    - Bullish Engulfing (Bullish Reversal)
    - Bearish Engulfing (Bearish Reversal)
    - Morning Star (Bullish 3-bar Reversal)
    - Evening Star (Bearish 3-bar Reversal)
    - Three White Soldiers (Strong Bullish Continuation)
    - Three Black Crows (Strong Bearish Continuation)
    - Bullish Marubozu / Bearish Marubozu (Strong Momentum)

    Args:
        df: DataFrame with 'open', 'high', 'low', 'close'.
        lookback_bars: How many recent bars to evaluate for patterns.

    Returns:
        List of detected pattern objects sorted by recency and relevance.
    """
    if len(df) < 3:
        return []

    patterns: List[Dict[str, Any]] = []
    n = len(df)
    start_idx = max(2, n - lookback_bars)

    for i in range(start_idx, n):
        row = df.iloc[i]
        prev1 = df.iloc[i - 1]
        prev2 = df.iloc[i - 2]

        o, h, l, c = float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"])
        o1, h1, l1, c1 = float(prev1["open"]), float(prev1["high"]), float(prev1["low"]), float(prev1["close"])
        o2, h2, l2, c2 = float(prev2["open"]), float(prev2["high"]), float(prev2["low"]), float(prev2["close"])

        body = abs(c - o)
        candle_range = h - l if (h - l) > 0 else 0.0001
        upper_shadow = h - max(o, c)
        lower_shadow = min(o, c) - l
        is_bullish = c > o
        is_bearish = c < o
        bars_ago = n - 1 - i

        # 1. Doji
        if body <= candle_range * 0.10:
            patterns.append({
                "name": "Doji",
                "bias": "NEUTRAL",
                "significance": "MODERATE",
                "bars_ago": bars_ago,
                "description": f"Candle body represents less than 10% of total range, indicating acute market indecision.",
            })

        # 2. Hammer & Hanging Man (Small body near top, long lower shadow >= 2x body, minimal upper shadow)
        if lower_shadow >= 2.0 * body and upper_shadow <= candle_range * 0.15 and body > 0.05 * candle_range:
            # Context: if preceded by a downward bar -> Hammer (Bullish), if in uptrend -> Hanging Man (Bearish)
            if c1 < o1:
                patterns.append({
                    "name": "Hammer",
                    "bias": "BULLISH",
                    "significance": "HIGH",
                    "bars_ago": bars_ago,
                    "description": f"Long lower shadow ({lower_shadow:.2f}) rejected selling pressure after a pullback; bullish reversal signal.",
                })
            else:
                patterns.append({
                    "name": "Hanging Man",
                    "bias": "BEARISH",
                    "significance": "MODERATE",
                    "bars_ago": bars_ago,
                    "description": f"Long lower shadow appeared after an advance, warning that buyers are losing unilateral control.",
                })

        # 3. Inverted Hammer & Shooting Star (Small body near bottom, long upper shadow >= 2x body, minimal lower shadow)
        if upper_shadow >= 2.0 * body and lower_shadow <= candle_range * 0.15 and body > 0.05 * candle_range:
            if c1 < o1:
                patterns.append({
                    "name": "Inverted Hammer",
                    "bias": "BULLISH",
                    "significance": "MODERATE",
                    "bars_ago": bars_ago,
                    "description": f"Upper shadow shows buyers attempted an aggressive rally after a downtrend.",
                })
            else:
                patterns.append({
                    "name": "Shooting Star",
                    "bias": "BEARISH",
                    "significance": "HIGH",
                    "bars_ago": bars_ago,
                    "description": f"Long upper shadow rejected intraday highs after an advance; bearish reversal signal.",
                })

        # 4. Bullish Engulfing
        if is_bullish and (c1 < o1) and (c > o1) and (o < c1):
            patterns.append({
                "name": "Bullish Engulfing",
                "bias": "BULLISH",
                "significance": "HIGH",
                "bars_ago": bars_ago,
                "description": f"Bullish green bar completely engulfed prior red candle's body, signaling strong demand takeover.",
            })

        # 5. Bearish Engulfing
        if is_bearish and (c1 > o1) and (c < o1) and (o > c1):
            patterns.append({
                "name": "Bearish Engulfing",
                "bias": "BEARISH",
                "significance": "HIGH",
                "bars_ago": bars_ago,
                "description": f"Bearish red candle engulfed preceding green candle's body, indicating sudden institutional distribution.",
            })

        # 6. Marubozu (Almost no shadows, massive conviction body)
        if body >= candle_range * 0.90:
            if is_bullish:
                patterns.append({
                    "name": "Bullish Marubozu",
                    "bias": "BULLISH",
                    "significance": "HIGH",
                    "bars_ago": bars_ago,
                    "description": f"Full green candle with negligible shadows; buyers dominated throughout the entire session.",
                })
            else:
                patterns.append({
                    "name": "Bearish Marubozu",
                    "bias": "BEARISH",
                    "significance": "HIGH",
                    "bars_ago": bars_ago,
                    "description": f"Full red candle with negligible shadows; sellers dominated throughout the entire session.",
                })

        # 7. Morning Star (Bearish candle -> Small body gap down -> Strong Bullish candle closing > 50% into 1st candle)
        body1 = abs(c1 - o1)
        body2 = abs(c2 - o2)
        if (c2 < o2) and (body1 <= (h1 - l1) * 0.35) and is_bullish and (c > o2 - (body2 * 0.5)):
            patterns.append({
                "name": "Morning Star",
                "bias": "BULLISH",
                "significance": "VERY_HIGH",
                "bars_ago": bars_ago,
                "description": f"Three-candle reversal: downward impulse followed by star consolidation and strong bullish confirmation.",
            })

        # 8. Evening Star (Bullish candle -> Small body gap up -> Strong Bearish candle closing > 50% into 1st candle)
        if (c2 > o2) and (body1 <= (h1 - l1) * 0.35) and is_bearish and (c < o2 + (body2 * 0.5)):
            patterns.append({
                "name": "Evening Star",
                "bias": "BEARISH",
                "significance": "VERY_HIGH",
                "bars_ago": bars_ago,
                "description": f"Three-candle reversal: upward surge followed by top exhaustion and sharp bearish distribution.",
            })

        # 9. Three White Soldiers (Three consecutive strong bullish candles, each closing higher near high)
        if is_bullish and (c1 > o1) and (c2 > o2) and (c > c1 > c2) and (o > o1 > o2):
            patterns.append({
                "name": "Three White Soldiers",
                "bias": "BULLISH",
                "significance": "VERY_HIGH",
                "bars_ago": bars_ago,
                "description": f"Three consecutive higher green closes showing relentless institutional accumulation.",
            })

        # 10. Three Black Crows (Three consecutive strong bearish candles, each closing lower near low)
        if is_bearish and (c1 < o1) and (c2 < o2) and (c < c1 < c2) and (o < o1 < o2):
            patterns.append({
                "name": "Three Black Crows",
                "bias": "BEARISH",
                "significance": "VERY_HIGH",
                "bars_ago": bars_ago,
                "description": f"Three consecutive lower red closes signaling aggressive institutional capitulation.",
            })

    # Deduplicate patterns with identical name and bars_ago
    seen = set()
    deduped = []
    for p in patterns:
        key = (p["name"], p["bars_ago"])
        if key not in seen:
            seen.add(key)
            deduped.append(p)

    return deduped
