"""Date and time tools."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import Field

from ..registry import registry


@registry.register("time")
def current_time(
    timezone: Annotated[
        str, Field(description="IANA time zone name, e.g. 'America/Lima' or 'UTC'")
    ] = "UTC",
) -> str:
    """Return the current date and time in the given time zone as ISO 8601."""
    try:
        zone = ZoneInfo(timezone)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise ValueError(f"Unknown time zone '{timezone}'") from exc
    now = datetime.now(zone)
    return f"{now.isoformat(timespec='seconds')} ({now.strftime('%A')})"
