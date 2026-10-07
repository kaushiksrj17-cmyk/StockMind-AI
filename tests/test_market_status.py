"""Unit tests for Indian Market (NSE/BSE) Trading Session Status Engine."""

from datetime import datetime, timezone, timedelta
import pytest
from backend.app.services.market.market_status import (
    IST_OFFSET,
    get_market_status,
)


def test_weekend_market_status():
    """Verify market is reported as CLOSED on Saturday and Sunday."""
    # 2026-10-10 is Saturday (UTC 06:00 -> IST 11:30)
    saturday_utc = datetime(2026, 10, 10, 6, 0, 0, tzinfo=timezone.utc)
    status_nse = get_market_status(exchange="NSE", reference_time=saturday_utc)
    assert status_nse.status == "CLOSED"
    assert status_nse.is_trading_open is False
    assert status_nse.day_of_week == "Saturday"
    assert "weekend" in status_nse.message.lower()

    # 2026-10-11 is Sunday
    sunday_utc = datetime(2026, 10, 11, 6, 0, 0, tzinfo=timezone.utc)
    status_bse = get_market_status(exchange="BSE", reference_time=sunday_utc)
    assert status_bse.status == "CLOSED"
    assert status_bse.is_trading_open is False


def test_official_holiday_market_status():
    """Verify market is reported as CLOSED on official exchange holiday."""
    # 2026-01-26 is Republic Day (Monday)
    holiday_utc = datetime(2026, 1, 26, 6, 0, 0, tzinfo=timezone.utc)
    status = get_market_status(exchange="NSE", reference_time=holiday_utc)
    assert status.status == "CLOSED"
    assert status.is_trading_open is False
    assert "official exchange holiday" in status.message.lower()


def test_pre_open_session_status():
    """Verify pre-open session (09:00 - 09:15 IST) on a regular weekday."""
    # Wednesday 2026-10-07 at 03:35 UTC -> 09:05 IST
    pre_open_utc = datetime(2026, 10, 7, 3, 35, 0, tzinfo=timezone.utc)
    status = get_market_status(exchange="NSE", reference_time=pre_open_utc)
    assert status.status == "PRE_OPEN"
    assert status.is_trading_open is False
    assert "pre-market" in status.message.lower()


def test_regular_trading_session_status():
    """Verify regular trading hours (09:15 - 15:30 IST) on a regular weekday."""
    # Wednesday 2026-10-07 at 05:30 UTC -> 11:00 IST
    regular_utc = datetime(2026, 10, 7, 5, 30, 0, tzinfo=timezone.utc)
    status = get_market_status(exchange="NSE", reference_time=regular_utc)
    assert status.status == "OPEN"
    assert status.is_trading_open is True
    assert status.trading_start_ist == "09:15:00"
    assert status.trading_end_ist == "15:30:00"


def test_post_close_session_status():
    """Verify post-closing session (15:40 - 16:00 IST)."""
    # Wednesday 2026-10-07 at 10:15 UTC -> 15:45 IST
    post_close_utc = datetime(2026, 10, 7, 10, 15, 0, tzinfo=timezone.utc)
    status = get_market_status(exchange="BSE", reference_time=post_close_utc)
    assert status.status == "POST_CLOSE"
    assert status.is_trading_open is False


def test_after_hours_status():
    """Verify closed status after market closes for the night."""
    # Wednesday 2026-10-07 at 14:30 UTC -> 20:00 IST
    after_hours_utc = datetime(2026, 10, 7, 14, 30, 0, tzinfo=timezone.utc)
    status = get_market_status(exchange="NSE", reference_time=after_hours_utc)
    assert status.status == "CLOSED"
    assert status.is_trading_open is False
    assert "has closed for the day" in status.message.lower()
