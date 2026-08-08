"""Timezone helpers — everything internally is Australia/Sydney; display is AEST/AEDT."""

from datetime import datetime
from zoneinfo import ZoneInfo

SYDNEY = ZoneInfo("Australia/Sydney")


def now_sydney() -> datetime:
    """Return current time in Australia/Sydney."""
    return datetime.now(SYDNEY)


def to_sydney(dt: datetime) -> datetime:
    """Attach or convert a datetime to Australia/Sydney."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=SYDNEY)
    return dt.astimezone(SYDNEY)


def is_asx_open_time(dt: datetime) -> bool:
    """True if Sydney time is in the ASX continuous session window (10:00-16:00)."""
    syd = to_sydney(dt)
    return syd.weekday() < 5 and 10 <= syd.hour < 16


def next_asx_session(dt: datetime) -> datetime:
    """Return the next ASX trading session date at 10:00 Sydney."""
    syd = to_sydney(dt)
    candidate = syd.replace(hour=10, minute=0, second=0, microsecond=0)
    if candidate <= syd:
        candidate = candidate + __import__("datetime").timedelta(days=1)
    # Skip weekends (no full holiday calendar in MVP)
    while candidate.weekday() >= 5:
        candidate = candidate + __import__("datetime").timedelta(days=1)
    return candidate


def asx_holiday_list(year: int) -> list:
    """Minimal fixed ASX holidays. Extend with a calendar feed for production."""
    # Standard observed dates vary by year; this is a simple best-effort list.
    fixed = [
        f"{year}-01-01",
        f"{year}-01-26",
        f"{year}-04-25",
        f"{year}-12-25",
        f"{year}-12-26",
    ]
    return fixed
