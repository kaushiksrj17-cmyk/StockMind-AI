"""Institutional Market Anomaly Detection Engine for StockMind-AI.

Implements:
1. Isolation Forest for multivariate structural outlier identification.
2. Price Anomaly Detection via rolling statistical Z-scores and gap dynamics.
3. Volume Spike Detection relative to rolling 20-period baseline.
4. Volatility Anomaly Detection via True Range and Bollinger bandwidth expansion.
5. Price-Volume Divergence Detection (Absorption, Churn, Exhaustion).
6. Severity Classification (LOW, MEDIUM, HIGH, CRITICAL).
7. Institutional deterministic explanations for risk audits.
"""

from dataclasses import dataclass, field
import datetime
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


@dataclass
class AnomalyItem:
    """Individual detected anomaly event."""
    index: int
    timestamp: str
    anomaly_type: str  # PRICE_SHOCK, VOLUME_SPIKE, VOLATILITY_BURST, PRICE_VOLUME_DIVERGENCE, STRUCTURAL_OUTLIER
    severity: str      # LOW, MEDIUM, HIGH, CRITICAL
    score: float       # Normalized anomaly severity score (0.0 - 1.0)
    description: str
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "timestamp": self.timestamp,
            "anomaly_type": self.anomaly_type,
            "severity": self.severity,
            "score": round(self.score, 3),
            "description": self.description,
            "metrics": self.metrics,
        }


@dataclass
class AnomalyReport:
    """Comprehensive anomaly intelligence report for a symbol."""
    symbol: str
    has_anomalies: bool
    total_detected: int
    critical_count: int
    high_count: int
    medium_count: int
    low_count: int
    is_current_bar_anomalous: bool
    current_anomaly_score: float
    current_severity: str
    primary_explanation: str
    divergence_status: str
    anomalies: List[AnomalyItem] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "has_anomalies": self.has_anomalies,
            "total_detected": self.total_detected,
            "critical_count": self.critical_count,
            "high_count": self.high_count,
            "medium_count": self.medium_count,
            "low_count": self.low_count,
            "is_current_bar_anomalous": self.is_current_bar_anomalous,
            "current_anomaly_score": round(self.current_anomaly_score, 3),
            "current_severity": self.current_severity,
            "primary_explanation": self.primary_explanation,
            "divergence_status": self.divergence_status,
            "anomalies": [a.to_dict() for a in self.anomalies[-20:]],
        }


class AnomalyEngine:
    """Multi-dimensional anomaly detection using statistical models and machine learning."""

    def __init__(
        self,
        contamination: float = 0.05,
        price_z_threshold: float = 2.5,
        volume_spike_multiplier: float = 2.5,
        volatility_expansion_threshold: float = 2.0,
        random_state: int = 42,
    ) -> None:
        self.contamination = contamination
        self.price_z_threshold = price_z_threshold
        self.volume_spike_multiplier = volume_spike_multiplier
        self.volatility_expansion_threshold = volatility_expansion_threshold
        self.random_state = random_state

    def detect_anomalies(
        self,
        df: pd.DataFrame,
        symbol: str = "ASSET",
    ) -> AnomalyReport:
        """Analyze OHLCV dataframe and generate comprehensive anomaly diagnosis.
        
        Requires columns: ['open', 'high', 'low', 'close', 'volume'].
        Optional: ['timestamp' or 'date'].
        """
        if df is None or len(df) < 15:
            return AnomalyReport(
                symbol=symbol,
                has_anomalies=False,
                total_detected=0,
                critical_count=0,
                high_count=0,
                medium_count=0,
                low_count=0,
                is_current_bar_anomalous=False,
                current_anomaly_score=0.0,
                current_severity="NONE",
                primary_explanation="Insufficient price bars for statistical baseline.",
                divergence_status="NORMAL",
                anomalies=[],
            )

        # Standardize column names
        data = df.copy()
        data.columns = [str(c).strip().lower() for c in data.columns]
        close = data["close"].astype(float).values
        open_p = data["open"].astype(float).values if "open" in data.columns else close
        high = data["high"].astype(float).values if "high" in data.columns else close
        low = data["low"].astype(float).values if "low" in data.columns else close
        volume = data["volume"].astype(float).values if "volume" in data.columns else np.ones_like(close)

        n = len(close)
        timestamps = []
        if "timestamp" in data.columns:
            timestamps = [str(t) for t in data["timestamp"]]
        elif "date" in data.columns:
            timestamps = [str(t) for t in data["date"]]
        else:
            timestamps = [f"Bar-{i}" for i in range(n)]

        # 1. Feature Engineering for Outlier Analytics
        returns = np.zeros(n)
        returns[1:] = (close[1:] - close[:-1]) / np.maximum(close[:-1], 1e-6)

        # Rolling 20-period return standard deviation & mean
        ret_mean = pd.Series(returns).rolling(window=20, min_periods=5).mean().fillna(0.0).values
        ret_std = pd.Series(returns).rolling(window=20, min_periods=5).std().fillna(1e-4).values
        ret_std = np.where(ret_std < 1e-5, 1e-4, ret_std)
        z_scores = (returns - ret_mean) / ret_std

        # Volume baseline (SMA 20)
        vol_series = pd.Series(volume)
        vol_sma20 = vol_series.rolling(window=20, min_periods=5).mean().fillna(1.0).values
        vol_sma20 = np.where(vol_sma20 < 1.0, 1.0, vol_sma20)
        vol_ratios = volume / vol_sma20

        # True Range and Volatility
        tr = np.maximum(high - low, np.maximum(abs(high - np.roll(close, 1)), abs(low - np.roll(close, 1))))
        tr[0] = high[0] - low[0]
        atr20 = pd.Series(tr).rolling(window=20, min_periods=5).mean().fillna(tr[0]).values
        atr20 = np.where(atr20 < 1e-4, 1e-4, atr20)
        volatility_ratios = tr / atr20

        # 2. Run Isolation Forest
        # Features: [return, abs(return), volume_ratio, tr/close, close_to_open_pct]
        feat_return = returns
        feat_abs_ret = np.abs(returns)
        feat_vol_ratio = np.clip(vol_ratios, 0.0, 15.0)
        feat_tr_pct = tr / np.maximum(close, 1e-4)
        feat_body_pct = (close - open_p) / np.maximum(open_p, 1e-4)

        X = np.column_stack([feat_return, feat_abs_ret, feat_vol_ratio, feat_tr_pct, feat_body_pct])
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        iso_anomalies = np.zeros(n, dtype=int)
        iso_scores = np.zeros(n, dtype=float)

        if n >= 25:
            try:
                clf = IsolationForest(
                    contamination=self.contamination,
                    random_state=self.random_state,
                    n_estimators=100,
                )
                clf.fit(X)
                # predict: -1 for anomaly, 1 for inlier
                iso_preds = clf.predict(X)
                iso_anomalies = np.where(iso_preds == -1, 1, 0)
                # decision_function: lower score = more anomalous. Invert & normalize to 0..1
                raw_scores = clf.decision_function(X)
                # Scale raw score such that low negative raw scores map to higher anomaly probability
                min_s, max_s = np.min(raw_scores), np.max(raw_scores)
                if max_s > min_s:
                    iso_scores = 1.0 - (raw_scores - min_s) / (max_s - min_s)
                else:
                    iso_scores = np.zeros(n)
            except Exception:
                iso_anomalies = np.zeros(n, dtype=int)
                iso_scores = np.zeros(n, dtype=float)

        # 3. Detect and Classify Anomalies Per Bar
        detected_items: List[AnomalyItem] = []

        for i in range(5, n):
            z = float(z_scores[i])
            v_ratio = float(vol_ratios[i])
            v_volat = float(volatility_ratios[i])
            is_iso = bool(iso_anomalies[i] == 1)
            iso_s = float(iso_scores[i])
            ret_pct = float(returns[i] * 100.0)
            c_val = float(close[i])

            # Check Anomaly Criteria
            is_price_anomaly = abs(z) >= self.price_z_threshold
            is_vol_spike = v_ratio >= self.volume_spike_multiplier
            is_volat_burst = v_volat >= self.volatility_expansion_threshold

            # Price-Volume Divergence Check
            is_divergence = False
            div_type = "NONE"
            if v_ratio >= 2.2 and abs(ret_pct) < 0.25:
                is_divergence = True
                div_type = "HIGH_VOLUME_CHURN"  # Heavy institutional turnover with zero price advancement
            elif ret_pct >= 2.0 and v_ratio <= 0.60:
                is_divergence = True
                div_type = "EXHAUSTION_RALLY"   # Price pushing up on drying liquidity
            elif ret_pct <= -2.0 and v_ratio <= 0.60:
                is_divergence = True
                div_type = "LOW_VOLUME_DRIFT"   # Price dropping without institutional selling pressure
            elif v_ratio >= 2.8 and ret_pct < -0.8 and (close[i] - low[i]) > (high[i] - close[i]):
                is_divergence = True
                div_type = "ABSORPTION_HAMMER"   # Aggressive selling absorbed by buyers at lows

            # Trigger condition
            if is_price_anomaly or is_vol_spike or is_volat_burst or is_divergence or is_iso:
                # Determine Severity
                anomaly_type = "STRUCTURAL_OUTLIER"
                score = 0.50
                severity = "LOW"
                explanations = []

                if is_price_anomaly and is_vol_spike:
                    anomaly_type = "PRICE_VOLUME_SURGE"
                    score = min(0.95, 0.6 + abs(z) * 0.08 + v_ratio * 0.05)
                    severity = "CRITICAL" if abs(z) >= 3.5 or v_ratio >= 4.0 else "HIGH"
                    explanations.append(f"Price moved {ret_pct:+.2f}% (Z-Score: {z:+.2f}) on {v_ratio:.1f}x average volume")
                elif is_price_anomaly:
                    anomaly_type = "PRICE_SHOCK"
                    score = min(0.95, 0.5 + abs(z) * 0.1)
                    if abs(z) >= 3.5 or abs(ret_pct) >= 7.0:
                        severity = "CRITICAL"
                    elif abs(z) >= 2.8 or abs(ret_pct) >= 3.5:
                        severity = "HIGH"
                    else:
                        severity = "MEDIUM"
                    explanations.append(f"Abnormal price displacement of {ret_pct:+.2f}% (Z-Score: {z:+.2f})")
                elif is_vol_spike:
                    anomaly_type = "VOLUME_SPIKE"
                    score = min(0.85, 0.4 + v_ratio * 0.08)
                    severity = "HIGH" if v_ratio >= 4.0 else "MEDIUM"
                    explanations.append(f"Trading volume spiked {v_ratio:.1f}x relative to 20-period baseline")
                elif is_divergence:
                    anomaly_type = "PRICE_VOLUME_DIVERGENCE"
                    score = 0.65
                    severity = "MEDIUM"
                    explanations.append(f"Price-volume divergence detected ({div_type.replace('_', ' ')})")
                elif is_volat_burst:
                    anomaly_type = "VOLATILITY_BURST"
                    score = min(0.80, 0.4 + v_volat * 0.1)
                    severity = "MEDIUM" if v_volat < 3.0 else "HIGH"
                    explanations.append(f"Intraday true range expanded {v_volat:.1f}x above normal ATR")
                elif is_iso:
                    anomaly_type = "STRUCTURAL_OUTLIER"
                    score = max(0.55, iso_s)
                    severity = "MEDIUM" if score < 0.75 else "HIGH"
                    explanations.append(f"Isolation Forest identified multi-factor structural anomaly (Outlier Score: {iso_s:.2f})")

                desc = ". ".join(explanations) + "."

                item = AnomalyItem(
                    index=i,
                    timestamp=timestamps[i],
                    anomaly_type=anomaly_type,
                    severity=severity,
                    score=score,
                    description=desc,
                    metrics={
                        "return_pct": round(ret_pct, 2),
                        "z_score": round(z, 2),
                        "volume_ratio": round(v_ratio, 2),
                        "volatility_ratio": round(v_volat, 2),
                        "close": round(c_val, 2),
                        "isolation_forest_score": round(iso_s, 3),
                    },
                )
                detected_items.append(item)

        # Summary Metrics
        critical_cnt = sum(1 for a in detected_items if a.severity == "CRITICAL")
        high_cnt = sum(1 for a in detected_items if a.severity == "HIGH")
        med_cnt = sum(1 for a in detected_items if a.severity == "MEDIUM")
        low_cnt = sum(1 for a in detected_items if a.severity == "LOW")

        # Current (Latest Bar) Status
        is_cur_anom = False
        cur_score = 0.0
        cur_sev = "NORMAL"
        cur_expl = "Normal statistical regime. Price and volume flow conform to baseline parameters."

        if len(detected_items) > 0 and detected_items[-1].index == (n - 1):
            latest = detected_items[-1]
            is_cur_anom = True
            cur_score = latest.score
            cur_sev = latest.severity
            cur_expl = f"Active {latest.severity} Anomaly: {latest.description}"

        # Current Divergence Status
        cur_v_ratio = float(vol_ratios[-1])
        cur_ret_pct = float(returns[-1] * 100.0)
        div_status = "SYNCHRONIZED"
        if cur_v_ratio >= 2.2 and abs(cur_ret_pct) < 0.3:
            div_status = "CHURN / STALLING (Accumulation or Distribution battle)"
        elif cur_ret_pct > 2.0 and cur_v_ratio < 0.65:
            div_status = "EXHAUSTION (Price up on low participation)"
        elif cur_ret_pct < -2.0 and cur_v_ratio < 0.65:
            div_status = "UNSUPPORTED PULLBACK (Selling without volume)"
        elif cur_v_ratio > 3.0 and cur_ret_pct > 1.5:
            div_status = "INSTITUTIONAL ACCUMULATION BREAKOUT"
        elif cur_v_ratio > 3.0 and cur_ret_pct < -1.5:
            div_status = "INSTITUTIONAL DISTRIBUTION SELLOFF"

        return AnomalyReport(
            symbol=symbol.upper(),
            has_anomalies=len(detected_items) > 0,
            total_detected=len(detected_items),
            critical_count=critical_cnt,
            high_count=high_cnt,
            medium_count=med_cnt,
            low_count=low_cnt,
            is_current_bar_anomalous=is_cur_anom,
            current_anomaly_score=cur_score,
            current_severity=cur_sev,
            primary_explanation=cur_expl,
            divergence_status=div_status,
            anomalies=detected_items,
        )


# Global singleton instance
anomaly_engine = AnomalyEngine()
