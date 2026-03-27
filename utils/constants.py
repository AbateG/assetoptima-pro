"""
Shared constants for AssetOptima Pro.
"""

from __future__ import annotations

from typing import Final

STATUS_COLOR_MAP: Final[dict[str, str]] = {
    "Green": "#1E8449",
    "Amber": "#D68910",
    "Yellow": "#D68910",
    "Red": "#C0392B",
    "Critical": "#922B21",
    "Compliant": "#1E8449",
    "Watchlist": "#D68910",
    "Breach": "#C0392B",
    "Refinance Candidate": "#2874A6",
    "Leader": "#1E8449",
    "Competitive": "#229954",
    "At Risk": "#D68910",
    "Weak": "#C0392B",
    "Strong Hold": "#1E8449",
    "Hold": "#229954",
    "Watch": "#D68910",
    "Refinance": "#2874A6",
    "Sell": "#C0392B",
    "High": "#C0392B",
    "Medium": "#D68910",
    "Low": "#1E8449",
    "Delayed": "#C0392B",
    "Upcoming": "#D68910",
}

DEFAULT_STATUS_COLOR: Final[str] = "#5D6D7E"

DEFAULT_CURRENCY_STR: Final[str] = "\$0"
DEFAULT_PERCENT_STR: Final[str] = "0.0%"
DEFAULT_MULTIPLE_STR: Final[str] = "0.00x"
DEFAULT_DAYS_STR: Final[str] = "N/A"