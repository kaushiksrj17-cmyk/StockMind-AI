"""Data formatting utilities for financial figures, numbers, and timestamps."""

from datetime import datetime
from typing import Optional, Union


def format_inr(val: Optional[Union[float, int]]) -> str:
    """Format numeric value into Indian Rupee string (₹)."""
    if val is None:
        return "—"
    if val < 0:
        return f"-₹{abs(val):,.2f}"
    return f"₹{val:,.2f}"


def format_currency(val: Optional[Union[float, int]], currency: str = "INR") -> str:
    """Format numeric value into readable currency string (defaults to INR ₹)."""
    if val is None:
        return "—"
    if currency == "INR":
        return format_inr(val)
    symbol = "$"
    if val < 0:
        return f"-{symbol}{abs(val):,.2f}"
    return f"{symbol}{val:,.2f}"


def format_percentage(val: Optional[Union[float, int]], show_sign: bool = True) -> str:
    """Format float into percentage string with optional +/- indicator."""
    if val is None:
        return "—"
    sign = "+" if (show_sign and val >= 0) else ""
    return f"{sign}{val:.2f}%"


def format_volume(val: Optional[Union[float, int]]) -> str:
    """Format volume integer into K/M/B shorthand."""
    if val is None or val == 0:
        return "0"
    if val >= 1_000_000_000:
        v = val / 1_000_000_000
        return f"{v:g}B"
    if val >= 1_000_000:
        v = val / 1_000_000
        return f"{v:g}M"
    if val >= 1_000:
        return f"{val / 1_000:.1f}K"
    return str(val)


def format_timestamp(dt: Optional[Union[datetime, str]]) -> str:
    """Format datetime object or ISO string into human readable terminal clock."""
    if not dt:
        return "—"
    if isinstance(dt, str):
        try:
            # Handle ISO parsing
            dt = datetime.fromisoformat(dt.replace("Z", "+00:00"))
        except Exception:
            return dt
    return dt.strftime("%H:%M:%S UTC")
