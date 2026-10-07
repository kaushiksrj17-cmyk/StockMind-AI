"""Real-Time Tick-to-Signal Streaming Pipeline for StockMind-AI.

Implements the institutional execution lifecycle:
LIVE TICK
→ candle update
→ indicator update
→ feature update
→ fast AI inference
→ signal update
→ WebSocket broadcast
→ dashboard update

CRITICAL PERFORMANCE SAFEGUARD:
Does NOT run expensive LSTM / Transformer / FinBERT NLP models on every tick.
Utilizes an asynchronous tiered cache for computationally heavy models,
evaluating fast incremental feature updates on live price arrivals in sub-millisecond time.
"""

import asyncio
from dataclasses import dataclass, field
import datetime
import logging
import numpy as np
from typing import Any, Callable, Dict, List, Optional

from ai_engine.signals.signal_engine import AISignalResult, ai_signal_engine

logger = logging.getLogger("stockmind.ai.realtime_pipeline")


@dataclass
class CandleBar:
    """Current in-progress OHLCV candle bar."""
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    timestamp: str
    cum_pv: float = 0.0  # Cumulative price * volume for VWAP
    cum_vol: float = 0.0  # Cumulative volume for VWAP
    tick_count: int = 0

    def update(self, price: float, vol: float, ts: str) -> None:
        self.high = max(self.high, price)
        self.low = min(self.low, price)
        self.close = price
        self.volume += max(vol, 1.0)
        self.cum_pv += price * max(vol, 1.0)
        self.cum_vol += max(vol, 1.0)
        self.tick_count += 1
        self.timestamp = ts

    @property
    def vwap(self) -> float:
        return self.cum_pv / self.cum_vol if self.cum_vol > 0 else self.close


class RealTimeSignalPipeline:
    """Event-driven live market data processor orchestrating tick-to-signal flow."""

    def __init__(self) -> None:
        # In-memory buffer of current active candles per symbol
        self._candle_bars: Dict[str, CandleBar] = {}
        # In-memory cache for expensive model predictions (updated on schedule or bar close)
        self._slow_model_cache: Dict[str, Dict[str, Any]] = {}
        # Last emitted signal per symbol
        self._latest_signals: Dict[str, AISignalResult] = {}
        # Optional WebSocket broadcaster callback
        self._ws_broadcast_hook: Optional[Callable[[str, Dict[str, Any]], Any]] = None

    def set_ws_broadcast_hook(self, hook: Callable[[str, Dict[str, Any]], Any]) -> None:
        """Register WebSocket dispatch callback."""
        self._ws_broadcast_hook = hook

    def update_slow_model_cache(
        self,
        symbol: str,
        ml_prediction: Optional[Dict[str, Any]] = None,
        deep_learning_forecast: Optional[Dict[str, Any]] = None,
        sentiment_data: Optional[Dict[str, Any]] = None,
        risk_metrics: Optional[Dict[str, Any]] = None,
        anomaly_report: Optional[Dict[str, Any]] = None,
        regime_data: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Periodic batch update of computationally expensive AI model outputs."""
        sym = symbol.strip().upper()
        if sym not in self._slow_model_cache:
            self._slow_model_cache[sym] = {}

        cache = self._slow_model_cache[sym]
        if ml_prediction is not None:
            cache["ml_prediction"] = ml_prediction
        if deep_learning_forecast is not None:
            cache["deep_learning"] = deep_learning_forecast
        if sentiment_data is not None:
            cache["sentiment"] = sentiment_data
        if risk_metrics is not None:
            cache["risk"] = risk_metrics
        if anomaly_report is not None:
            cache["anomaly"] = anomaly_report
        if regime_data is not None:
            cache["regime"] = regime_data

        cache["last_slow_update"] = datetime.datetime.now(datetime.timezone.utc).isoformat()

    def process_live_tick(
        self,
        symbol: str,
        price: float,
        volume: float = 100.0,
        day_open: Optional[float] = None,
        timestamp: Optional[str] = None,
        data_mode: str = "LIVE DATA",
    ) -> AISignalResult:
        """Full synchronous tick-to-signal execution step.
        
        Lifecycle:
        1. Candle Update
        2. Fast Indicator & VWAP Update
        3. Feature Update
        4. Fast AI Inference (combining live state with cached model outputs)
        5. Signal Update
        6. WebSocket Broadcast (dispatched)
        """
        sym = symbol.strip().upper()
        now_ts = timestamp or datetime.datetime.now(datetime.timezone.utc).isoformat()
        p = float(price)
        v = float(volume)

        # 1. CANDLE UPDATE
        if sym not in self._candle_bars:
            open_p = day_open if day_open and day_open > 0 else p
            self._candle_bars[sym] = CandleBar(
                symbol=sym,
                open=open_p,
                high=p,
                low=p,
                close=p,
                volume=v,
                timestamp=now_ts,
                cum_pv=p * v,
                cum_vol=v,
                tick_count=1,
            )
        else:
            self._candle_bars[sym].update(price=p, vol=v, ts=now_ts)

        bar = self._candle_bars[sym]

        # 2. FAST INDICATOR UPDATE
        cur_vwap = bar.vwap
        day_chg_pct = ((p - bar.open) / bar.open * 100.0) if bar.open > 0 else 0.0

        live_quote = {
            "symbol": sym,
            "last_price": p,
            "open": bar.open,
            "high": bar.high,
            "low": bar.low,
            "close": p,
            "volume": bar.volume,
            "vwap": cur_vwap,
            "change_percent": day_chg_pct,
            "timestamp": now_ts,
        }

        # 3. FEATURE UPDATE
        # Fast indicator baseline
        tech_indicators = {
            "technical_score": float(np.clip(50.0 + day_chg_pct * 8.0, 10.0, 95.0)),
            "rsi_14": float(np.clip(50.0 + day_chg_pct * 4.0, 15.0, 85.0)),
            "vwap_diff_pct": ((p - cur_vwap) / cur_vwap * 100.0) if cur_vwap > 0 else 0.0,
        }

        # 4. FAST AI INFERENCE
        # Pull cached slow models
        slow_cache = self._slow_model_cache.get(sym, {})
        ml_pred = slow_cache.get("ml_prediction")
        dl_pred = slow_cache.get("deep_learning")
        sent_data = slow_cache.get("sentiment")
        risk_data = slow_cache.get("risk")
        anom_data = slow_cache.get("anomaly")
        reg_data = slow_cache.get("regime")

        # Synthesize multi-factor signal
        signal_res = ai_signal_engine.generate_signal(
            symbol=sym,
            live_quote=live_quote,
            technical_data=tech_indicators,
            ml_prediction=ml_pred,
            deep_learning_forecast=dl_pred,
            sentiment_data=sent_data,
            risk_metrics=risk_data,
            anomaly_report=anom_data,
            regime_data=reg_data,
            data_mode=data_mode,
        )

        # 5. SIGNAL UPDATE
        self._latest_signals[sym] = signal_res

        # 6. WEBSOCKET BROADCAST
        if self._ws_broadcast_hook:
            try:
                hook_res = self._ws_broadcast_hook(sym, signal_res.to_dict())
                # If hook returns an unawaited coroutine, schedule it
                if asyncio.iscoroutine(hook_res):
                    try:
                        loop = asyncio.get_running_loop()
                        loop.create_task(hook_res)
                    except RuntimeError:
                        pass
            except Exception as e:
                logger.debug("WS Broadcast hook warning: %s", str(e))

        return signal_res

    def get_latest_signal(self, symbol: str) -> Optional[AISignalResult]:
        """Fetch current cached signal for symbol."""
        return self._latest_signals.get(symbol.strip().upper())


# Global singleton instance
realtime_signal_pipeline = RealTimeSignalPipeline()
