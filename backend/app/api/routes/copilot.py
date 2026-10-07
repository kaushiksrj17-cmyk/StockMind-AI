"""AI Market Copilot API routes."""

from fastapi import APIRouter, HTTPException, status

from ai_engine.copilot.copilot_engine import ai_copilot
from backend.app.schemas.copilot import (
    CopilotChatRequest,
    CopilotChatResponse,
)

router = APIRouter(prefix="/copilot", tags=["AI Market Copilot"])


@router.post(
    "/chat",
    response_model=CopilotChatResponse,
    summary="Ask AI Market Copilot a financial question",
    description=(
        "Processes queries using live market quotes, technical indicators, ML & DL forecasts, "
        "news sentiment, risk metrics, and fundamentals, strictly distinguishing FACT, "
        "CURRENT MARKET DATA, MODEL PREDICTION, INFERENCE, and UNCERTAINTY."
    ),
)
async def copilot_chat(payload: CopilotChatRequest) -> CopilotChatResponse:
    """Conversational endpoint for the institutional AI Market Copilot."""
    try:
        response = ai_copilot.answer_query(
            query=payload.query,
            symbol=payload.symbol,
            conversation_history=payload.history,
        )
        return CopilotChatResponse(**response.to_dict())
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Copilot inference failed: {str(e)}",
        )


@router.get(
    "/context/{symbol}",
    summary="Get raw quantitative context snapshot for Copilot",
    description="Inspect the full multi-factor context matrix used by the Copilot for a ticker.",
)
async def get_copilot_context(symbol: str):
    """Retrieve full context matrix for a symbol."""
    sym = symbol.strip().upper()
    try:
        ctx = ai_copilot.gather_context(sym)
        return {
            "symbol": ctx.symbol,
            "current_market_data": ctx.current_market_data,
            "technicals": ctx.technicals,
            "predictions": ctx.predictions,
            "news_sentiment": ctx.news_sentiment,
            "risk_metrics": ctx.risk_metrics,
            "fundamentals": ctx.fundamentals,
            "regime": ctx.regime,
            "anomalies": ctx.anomalies,
            "news_count": len(ctx.news_articles),
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to gather Copilot context for {sym}: {str(e)}",
        )
