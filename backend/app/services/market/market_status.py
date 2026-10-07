"""Indian Stock Market (NSE/BSE) Trading Hours and Status Engine.

Evaluates real-time market state based on IST (Asia/Kolkata, UTC+5:30)
trading sessions: Pre-Open, Regular Market, Post-Closing, and Holidays.
"""

from datetime import datetime, time, timezone, timedelta
from typing import Optional, Set
from pydantic import BaseModel, Field

# Indian Standard Time (UTC+5:30) offset
IST_OFFSET = timezone(timedelta(hours=5, minutes=30))

# Exchange Session Timings (IST)
PRE_OPEN_START = time(9, 0, 0)
REGULAR_MARKET_START = time(9, 15, 0)
REGULAR_MARKET_END = time(15, 30, 0)
POST_CLOSE_START = time(15, 40, 0)
POST_CLOSE_END = time(16, 0, 0)

# Major Indian Exchange Scheduled Holidays (Format: "YYYY-MM-DD")
STANDARD_NSE_HOLIDAYS_2026: Set[str] = {
    "2026-01-26",  # Republic Day
    "2026-03-03",  # Mahashivratri
    "2026-03-20",  # Holi
    "2026-04-03",  # Good Friday
    "2026-04-14",  # Dr. Ambedkar Jayanti
    "2026-05-01",  # Maharashtra Day
    "2026-08-15",  # Independence Day
    "2026-10-02",  # Mahatma Gandhi Jayanti
    "2026-10-20",  # Dussehra
    "2026-11-08",  # Diwali Laxmi Pujan
    "2026-11-10",  # Diwali Balipratipada
    "2026-12-25",  # Christmas
}


class MarketStatus(BaseModel):
    """Indian market status response model."""
    exchange: str = Field(..., description="Exchange identifier: NSE or BSE")
    status: str = Field(..., description="Status: OPEN, CLOSED, PRE_OPEN, POST_CLOSE")
    is_trading_open: bool = Field(..., description="True if normal trading order matching is active")
    current_time_ist: str = Field(..., description="Current ISO timestamp in IST timezone")
    trading_start_ist: str = Field(default="09:15:00", description="Regular trading start time")
    trading_end_ist: str = Field(default="15:30:00", description="Regular trading end time")
    timezone: str = Field(default="Asia/Kolkata", description="Exchange local timezone")
    day_of_week: str = Field(..., description="Day name (Monday-Sunday)")
    message: str = Field(..., description="Descriptive status narrative")


def get_market_status(
    exchange: str = "NSE",
    reference_time: Optional[datetime] = None,
) -> MarketStatus:
    """Calculate current market session state for NSE or BSE.

    Args:
        exchange: Exchange code (NSE or BSE).
        reference_time: Optional datetime override (UTC or timezone-aware); defaults to current UTC.

    Returns:
        MarketStatus: Detailed market trading status.
    """
    clean_exchange = exchange.strip().upper()
    if clean_exchange not in ("NSE", "BSE"):
        clean_exchange = "NSE"

    # Convert to Indian Standard Time (IST)
    if reference_time is None:
        now_utc = datetime.now(timezone.utc)
    else:
        now_utc = reference_time if reference_time.tzinfo else reference_time.replace(tzinfo=timezone.utc)

    now_ist = now_utc.astimezone(IST_OFFSET)
    current_time = now_ist.time()
    date_str = now_ist.strftime("%Y-%m-%d")
    day_name = now_ist.strftime("%A")
    weekday = now_ist.weekday()  # Monday = 0, Sunday = 6

    # 1. Weekend Check (Saturday = 5, Sunday = 6)
    if weekday in (5, 6):
        return MarketStatus(
            exchange=clean_exchange,
            status="CLOSED",
            is_trading_open=False,
            current_time_ist=now_ist.isoformat(),
            day_of_week=day_name,
            message=f"{clean_exchange} is closed for the weekend ({day_name}). Regular trading resumes Monday at 09:15 IST.",
        )

    # 2. Public Holiday Check
    if date_str in STANDARD_NSE_HOLIDAYS_2026:
        return MarketStatus(
            exchange=clean_exchange,
            status="CLOSED",
            is_trading_open=False,
            current_time_ist=now_ist.isoformat(),
            day_of_week=day_name,
            message=f"{clean_exchange} is closed today ({date_str}) for an official exchange holiday.",
        )

    # 3. Pre-Open Session (09:00 - 09:15 IST)
    if PRE_OPEN_START <= current_time < REGULAR_MARKET_START:
        return MarketStatus(
            exchange=clean_exchange,
            status="PRE_OPEN",
            is_trading_open=False,
            current_time_ist=now_ist.isoformat(),
            day_of_week=day_name,
            message=f"{clean_exchange} is in Pre-Market session (order entry/matching). Regular trading opens at 09:15 IST.",
        )

    # 4. Regular Market Session (09:15 - 15:30 IST)
    if REGULAR_MARKET_START <= current_time <= REGULAR_MARKET_END:
        return MarketStatus(
            exchange=clean_exchange,
            status="OPEN",
            is_trading_open=True,
            current_time_ist=now_ist.isoformat(),
            day_of_week=day_name,
            message=f"{clean_exchange} is actively OPEN for regular trading.",
        )

    # 5. Post-Close Session (15:40 - 16:00 IST)
    if POST_CLOSE_START <= current_time <= POST_CLOSE_END:
        return MarketStatus(
            exchange=clean_exchange,
            status="POST_CLOSE",
            is_trading_open=False,
            current_time_ist=now_ist.isoformat(),
            day_of_week=day_name,
            message=f"{clean_exchange} is in Post-Market closing session.",
        )

    # 6. Outside Hours
    if current_time < PRE_OPEN_START:
        msg = f"{clean_exchange} is closed. Pre-market opens at 09:00 IST, regular trading at 09:15 IST."
    else:
        msg = f"{clean_exchange} has closed for the day. Trading closed at 15:30 IST."

    return MarketStatus(
        exchange=clean_exchange,
        status="CLOSED",
        is_trading_open=False,
        current_time_ist=now_ist.isoformat(),
        day_of_week=day_name,
        message=msg,
    )
