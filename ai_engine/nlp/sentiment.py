"""Financial Sentiment Analysis Engine for StockMind-AI.

Implements:
1. FinBERT transformer abstraction (configuration-gated, non-blocking fallback).
2. Institutional Financial Lexicon & Loughran-McDonald domain dictionary model.
3. Multi-word financial idioms, negation detection, and intensifier scoring.
4. News impact score quantification (0–100 scale).
"""

from dataclasses import dataclass, field
from enum import Enum
import math
import os
import re
from typing import Any, Dict, List, Optional, Tuple


class SentimentLabel(str, Enum):
    POSITIVE = "POSITIVE"
    NEUTRAL = "NEUTRAL"
    NEGATIVE = "NEGATIVE"
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"


@dataclass
class SentimentResult:
    """Standardized financial sentiment assessment."""
    label: str  # POSITIVE, NEUTRAL, NEGATIVE
    score: float  # Continuous polarity: -1.0 (extremely bearish) to +1.0 (extremely bullish)
    confidence: float  # Confidence probability: 0.0 to 1.0
    impact_score: float  # Estimated market impact magnitude: 0.0 to 100.0
    positive_prob: float
    neutral_prob: float
    negative_prob: float
    model_name: str
    key_phrases: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "direction": "BULLISH" if self.score > 0.15 else ("BEARISH" if self.score < -0.15 else "NEUTRAL"),
            "score": round(self.score, 4),
            "confidence": round(self.confidence, 4),
            "confidence_pct": round(self.confidence * 100.0, 1),
            "impact_score": round(self.impact_score, 1),
            "probabilities": {
                "positive": round(self.positive_prob, 4),
                "neutral": round(self.neutral_prob, 4),
                "negative": round(self.negative_prob, 4),
            },
            "model_name": self.model_name,
            "key_phrases": self.key_phrases,
        }


# ============================================================================
# 1. INSTITUTIONAL FINANCIAL LEXICONS (Loughran-McDonald & NSE Domain)
# ============================================================================

FINANCIAL_POSITIVE_TERMS: Dict[str, float] = {
    # Earnings & Performance
    "profit": 1.0, "surge": 1.2, "growth": 0.8, "rally": 1.1, "outperform": 1.3,
    "beat": 1.2, "exceeded": 1.1, "record": 1.2, "expansion": 0.9, "all-time high": 1.5,
    "bullish": 1.2, "gain": 0.8, "jumped": 1.0, "soared": 1.3, "dividend": 0.7,
    "bonus": 0.8, "accretive": 1.2, "synergy": 0.9, "margin expansion": 1.4,
    "operating leverage": 1.2, "ebitda growth": 1.3, "net profit": 1.0,
    # Corporate Actions & Orders
    "deal": 0.9, "bags": 1.1, "contract win": 1.3, "commissioning": 1.0,
    "electrolyzer": 0.8, "capacity addition": 1.1, "partnership": 0.8,
    "buyback": 1.2, "share repurchase": 1.2, "debt reduction": 1.4, "deleveraging": 1.3,
    "cash flow positive": 1.3, "upgrade": 1.4, "upgraded": 1.4, "buy rating": 1.3,
    "overweight": 1.1, "strong buy": 1.5, "raised guidance": 1.5, "guidance hike": 1.5,
    "order book": 0.9, "inflow": 0.9, "deposit growth": 1.0, "credit growth": 0.9,
    # Regulatory & Approvals
    "approval": 1.1, "cleared": 1.0, "green light": 1.2, "patent granted": 1.2,
    "nod": 0.8, "fdi inflow": 1.1, "fii buying": 1.2, "net buyers": 1.1,
}

FINANCIAL_NEGATIVE_TERMS: Dict[str, float] = {
    # Earnings & Performance
    "loss": -1.1, "slump": -1.2, "plunge": -1.4, "tumble": -1.3, "decline": -0.8,
    "missed": -1.1, "disappoints": -1.2, "bearish": -1.1, "drop": -0.8, "fell": -0.7,
    "sank": -1.2, "drag": -0.9, "headwind": -1.0, "margin compression": -1.4,
    "ebitda contraction": -1.3, "guidance cut": -1.5, "downgraded": -1.4, "downgrade": -1.4,
    "sell rating": -1.3, "underweight": -1.1, "all-time low": -1.5, "weak": -0.8,
    # Risk, Distress & Debt
    "debt default": -1.8, "default": -1.6, "insolvency": -1.8, "bankruptcy": -1.9,
    "npa": -1.3, "bad loan": -1.4, "write-off": -1.3, "impairment": -1.3,
    "credit rating cut": -1.5, "liquidity crunch": -1.6, "run on bank": -1.9,
    "recession": -1.2, "slowdown": -0.9, "inflation spike": -1.1, "rate hike": -0.8,
    # Regulatory, Litigation & Governance
    "probe": -1.4, "investigation": -1.3, "sebi inquiry": -1.6, "ed raid": -1.8,
    "fraud": -1.9, "scam": -1.9, "whistleblower": -1.3, "resignation": -1.1,
    "fine": -1.1, "penalty": -1.2, "show cause notice": -1.3, "ban": -1.5,
    "suspended": -1.5, "lawsuit": -1.2, "litigation": -1.1, "audit query": -1.4,
    "promoter pledge": -1.2, "fii selling": -1.1, "net sellers": -1.1,
}

NEGATION_TERMS = {
    "not", "no", "never", "neither", "nor", "none", "without",
    "hardly", "scarcely", "barely", "fails to", "failed to", "unable to",
}

INTENSIFIERS: Dict[str, float] = {
    "sharply": 1.4, "massively": 1.5, "significantly": 1.3, "substantially": 1.3,
    "strongly": 1.3, "heavily": 1.3, "record": 1.4, "unprecedented": 1.5,
    "drastically": 1.4, "dramatically": 1.4, "steeply": 1.3,
}

DIMINISHERS: Dict[str, float] = {
    "slightly": 0.6, "marginally": 0.5, "partially": 0.7, "modestly": 0.7,
    "somewhat": 0.7, "fractionally": 0.5,
}

HIGH_IMPACT_CATALYSTS = [
    "rbi mpc", "repo rate", "fed", "quarterly results", "q1", "q2", "q3", "q4",
    "net profit", "acquisition", "merger", "sebi", "ed", "us fda", "usfda",
    "dividend", "bonus issue", "buyback", "delisting", "default", "electrolyzer",
    "green hydrogen", "multi-year deal", "contract", "guidance",
]


# ============================================================================
# 2. LEXICON-BASED FINANCIAL SENTIMENT MODEL
# ============================================================================

class FinancialLexiconSentimentModel:
    """Deterministic, zero-download financial sentiment engine grounded in domain lexicons."""

    def __init__(self) -> None:
        self.model_name = "FinancialLexicon-v1 (Loughran-McDonald Adapted)"

    def predict(self, text: str) -> SentimentResult:
        """Analyze text using domain-aware parsing, negation, and impact evaluation."""
        if not text or not text.strip():
            return SentimentResult(
                label=SentimentLabel.NEUTRAL.value,
                score=0.0,
                confidence=0.50,
                impact_score=10.0,
                positive_prob=0.33,
                neutral_prob=0.34,
                negative_prob=0.33,
                model_name=self.model_name,
                key_phrases=[],
            )

        text_lower = text.lower()
        words = re.findall(r"\b[a-z0-9\-']+\b", text_lower)

        pos_score = 0.0
        neg_score = 0.0
        found_phrases: List[str] = []

        # Check multi-word phrases first
        for phrase, weight in sorted(FINANCIAL_POSITIVE_TERMS.items(), key=lambda x: -len(x[0])):
            if " " in phrase and phrase in text_lower:
                pos_score += weight
                found_phrases.append(f"+{phrase}")

        for phrase, weight in sorted(FINANCIAL_NEGATIVE_TERMS.items(), key=lambda x: -len(x[0])):
            if " " in phrase and phrase in text_lower:
                neg_score += abs(weight)
                found_phrases.append(f"-{phrase}")

        # Scan words with 3-token lookback negation & intensifiers
        n = len(words)
        for i, word in enumerate(words):
            multiplier = 1.0
            is_negated = False

            # Check previous 3 tokens for negation or intensifiers
            start_window = max(0, i - 3)
            window_prev = words[start_window:i]

            for prev_word in window_prev:
                if prev_word in NEGATION_TERMS:
                    is_negated = True
                if prev_word in INTENSIFIERS:
                    multiplier *= INTENSIFIERS[prev_word]
                elif prev_word in DIMINISHERS:
                    multiplier *= DIMINISHERS[prev_word]

            if word in FINANCIAL_POSITIVE_TERMS and " " not in word:
                val = FINANCIAL_POSITIVE_TERMS[word] * multiplier
                if is_negated:
                    neg_score += val * 0.9  # Negated positive becomes negative
                    found_phrases.append(f"negated +{word}")
                else:
                    pos_score += val
                    found_phrases.append(f"+{word}")

            elif word in FINANCIAL_NEGATIVE_TERMS and " " not in word:
                val = abs(FINANCIAL_NEGATIVE_TERMS[word]) * multiplier
                if is_negated:
                    pos_score += val * 0.8  # Negated negative becomes mild positive
                    found_phrases.append(f"negated -{word}")
                else:
                    neg_score += val
                    found_phrases.append(f"-{word}")

        # Normalize total net polarity between -1.0 and +1.0 via hyperbolic tangent
        raw_net = pos_score - neg_score
        total_intensity = pos_score + neg_score

        if total_intensity > 0:
            score = math.tanh(raw_net / 2.5)
        else:
            score = 0.0

        # Compute probability distribution
        if total_intensity == 0:
            pos_prob = 0.15
            neg_prob = 0.15
            neu_prob = 0.70
            label = SentimentLabel.NEUTRAL.value
            confidence = 0.70
        else:
            denom = total_intensity + 1.2
            pos_prob = round(pos_score / denom, 4)
            neg_prob = round(neg_score / denom, 4)
            neu_prob = max(round(1.0 - (pos_prob + neg_prob), 4), 0.05)

            # Renormalize softmax-style
            s = pos_prob + neg_prob + neu_prob
            pos_prob, neg_prob, neu_prob = pos_prob / s, neg_prob / s, neu_prob / s

            if score > 0.18:
                label = SentimentLabel.POSITIVE.value
                confidence = min(0.55 + (abs(score) * 0.40), 0.96)
            elif score < -0.18:
                label = SentimentLabel.NEGATIVE.value
                confidence = min(0.55 + (abs(score) * 0.40), 0.96)
            else:
                label = SentimentLabel.NEUTRAL.value
                confidence = neu_prob

        # Calculate News Impact Score (0 to 100)
        # Factor 1: Polarity extremity (0 to 45 pts)
        polarity_pts = abs(score) * 45.0

        # Factor 2: High impact catalyst presence (0 to 40 pts)
        catalyst_count = sum(1 for cat in HIGH_IMPACT_CATALYSTS if cat in text_lower)
        catalyst_pts = min(catalyst_count * 18.0, 40.0)

        # Factor 3: Intensity & length (0 to 15 pts)
        intensity_pts = min(total_intensity * 3.5, 15.0)

        raw_impact = polarity_pts + catalyst_pts + intensity_pts
        impact_score = min(max(raw_impact, 12.0), 98.0)

        return SentimentResult(
            label=label,
            score=round(score, 4),
            confidence=round(confidence, 4),
            impact_score=round(impact_score, 1),
            positive_prob=round(pos_prob, 4),
            neutral_prob=round(neu_prob, 4),
            negative_prob=round(neg_prob, 4),
            model_name=self.model_name,
            key_phrases=list(dict.fromkeys(found_phrases))[:6],
        )

    def predict_batch(self, texts: List[str]) -> List[SentimentResult]:
        return [self.predict(t) for t in texts]


# ============================================================================
# 3. OPTIONAL TRANSFORMER-BASED FINBERT ADAPTER
# ============================================================================

class FinBERTSentimentModel:
    """FinBERT Transformer model adapter.

    Only activated when configured in environment or settings (e.g. FINBERT_ENABLED=true).
    Gracefully avoids unconfigured huge downloads.
    """

    def __init__(self, model_name: Optional[str] = None) -> None:
        self.model_name = model_name or os.getenv("FINBERT_MODEL_NAME", "ProsusAI/finbert")
        self._pipeline = None
        self._is_ready = False

        # Only initialize if explicitly configured in environment
        enabled = os.getenv("FINBERT_ENABLED", "false").lower() in ["true", "1", "yes"]
        if enabled:
            self._try_load_model()

    def _try_load_model(self) -> None:
        """Attempt to load transformer pipeline if package is present."""
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
            import torch

            device = 0 if torch.cuda.is_available() else -1
            self._pipeline = pipeline(
                "text-classification",
                model=self.model_name,
                tokenizer=self.model_name,
                device=device,
                top_k=None,
            )
            self._is_ready = True
        except Exception:
            self._is_ready = False

    def is_available(self) -> bool:
        return self._is_ready and self._pipeline is not None

    def predict(self, text: str) -> Optional[SentimentResult]:
        """Perform transformer inference if model is loaded; otherwise return None."""
        if not self.is_available():
            return None

        try:
            outputs = self._pipeline(text[:512])[0]
            scores = {item["label"].lower(): item["score"] for item in outputs}
            pos = scores.get("positive", 0.0)
            neg = scores.get("negative", 0.0)
            neu = scores.get("neutral", 0.0)

            net_score = pos - neg
            if net_score > 0.15:
                label = SentimentLabel.POSITIVE.value
                conf = pos
            elif net_score < -0.15:
                label = SentimentLabel.NEGATIVE.value
                conf = neg
            else:
                label = SentimentLabel.NEUTRAL.value
                conf = neu

            impact = min(max(abs(net_score) * 60.0 + 20.0, 10.0), 98.0)

            return SentimentResult(
                label=label,
                score=round(net_score, 4),
                confidence=round(conf, 4),
                impact_score=round(impact, 1),
                positive_prob=round(pos, 4),
                neutral_prob=round(neu, 4),
                negative_prob=round(neg, 4),
                model_name=f"FinBERT ({self.model_name})",
                key_phrases=[],
            )
        except Exception:
            return None


# ============================================================================
# 4. UNIFIED FINANCIAL SENTIMENT ANALYZER
# ============================================================================

class FinancialSentimentAnalyzer:
    """Unified Financial Sentiment Engine for StockMind-AI.

    Provides transparent fallback to FinancialLexicon when FinBERT is unconfigured,
    ensuring high reliability, speed, and institutional financial fidelity.
    """

    def __init__(self) -> None:
        self.finbert = FinBERTSentimentModel()
        self.lexicon = FinancialLexiconSentimentModel()

    def analyze(self, text: str) -> SentimentResult:
        """Analyze text using the best available configured engine."""
        if self.finbert.is_available():
            result = self.finbert.predict(text)
            if result is not None:
                return result

        # Production fallback: domain lexicon engine
        return self.lexicon.predict(text)

    def analyze_news_item(self, title: str, summary: str = "") -> SentimentResult:
        """Analyze headline with higher weighting, combined with summary."""
        combined = f"{title}. {summary}".strip()
        headline_res = self.analyze(title)

        if not summary:
            return headline_res

        body_res = self.analyze(summary)

        # Weighted combination: headline 65%, body 35%
        comb_score = (headline_res.score * 0.65) + (body_res.score * 0.35)
        comb_impact = max(headline_res.impact_score, (headline_res.impact_score * 0.70) + (body_res.impact_score * 0.30))

        if comb_score > 0.15:
            label = SentimentLabel.POSITIVE.value
        elif comb_score < -0.15:
            label = SentimentLabel.NEGATIVE.value
        else:
            label = SentimentLabel.NEUTRAL.value

        conf = max(headline_res.confidence, body_res.confidence)
        pos_prob = (headline_res.positive_prob * 0.65) + (body_res.positive_prob * 0.35)
        neg_prob = (headline_res.negative_prob * 0.65) + (body_res.negative_prob * 0.35)
        neu_prob = (headline_res.neutral_prob * 0.65) + (body_res.neutral_prob * 0.35)

        phrases = list(dict.fromkeys(headline_res.key_phrases + body_res.key_phrases))[:6]

        return SentimentResult(
            label=label,
            score=round(comb_score, 4),
            confidence=round(conf, 4),
            impact_score=round(comb_impact, 1),
            positive_prob=round(pos_prob, 4),
            neutral_prob=round(neu_prob, 4),
            negative_prob=round(neg_prob, 4),
            model_name=headline_res.model_name,
            key_phrases=phrases,
        )


# Global singleton instance
sentiment_analyzer = FinancialSentimentAnalyzer()
