"""
Display formatting helpers for AssetOptima Pro.
Optimized for deterministic, safe, and strict financial formatting.
"""

from __future__ import annotations

from typing import Any
import math

import pandas as pd

from utils.coercion import is_missing, parse_float, parse_int
from utils.constants import (
    DEFAULT_CURRENCY_STR,
    DEFAULT_DAYS_STR,
    DEFAULT_MULTIPLE_STR,
    DEFAULT_PERCENT_STR,
)


def _safe_decimals(decimals: int, fallback: int) -> int:
    """
    Normalize decimals to a non-negative integer.
    """
    try:
        decimals_int = int(decimals)
        return max(0, decimals_int)
    except (TypeError, ValueError):
        return fallback


def fmt_currency(
    value: Any,
    decimals: int = 0,
    default: str = DEFAULT_CURRENCY_STR,
    include_sign: bool = False,
) -> str:
    """
    Format a value as currency safely (e.g., -\$1,250,000 or +\$1,250,000).
    """
    decimals = _safe_decimals(decimals, 0)
    numeric_val = parse_float(value, default=float("nan"))
    if pd.isna(numeric_val):
        return default

    if math.isclose(numeric_val, 0.0, abs_tol=1e-9):
        return f"${0:,.{decimals}f}"

    sign = ""
    if numeric_val < 0:
        sign = "-"
    elif include_sign and numeric_val > 0:
        sign = "+"

    return f"{sign}${abs(numeric_val):,.{decimals}f}"


def fmt_pct(
    value: Any,
    decimals: int = 1,
    default: str = DEFAULT_PERCENT_STR,
    include_sign: bool = False,
) -> str:
    """
    Format a decimal value as a percentage safely.

    Assumes numeric inputs are decimal fractions:
    - 0.145 -> 14.5%

    For strings ending in '%', the numeric content is treated as an already
    expressed percent:
    - '95%' -> 95.0%
    - '12.5%' -> 12.5%
    """
    decimals = _safe_decimals(decimals, 1)

    if isinstance(value, str) and value.strip().endswith("%"):
        numeric_percent = parse_float(value, default=float("nan"))
        if pd.isna(numeric_percent):
            return default

        if math.isclose(numeric_percent, 0.0, abs_tol=1e-9):
            return f"0.{'0' * decimals}%"

        sign = ""
        if numeric_percent < 0:
            sign = "-"
        elif include_sign and numeric_percent > 0:
            sign = "+"

        return f"{sign}{abs(numeric_percent):,.{decimals}f}%"

    numeric_val = parse_float(value, default=float("nan"))
    if pd.isna(numeric_val):
        return default

    if math.isclose(numeric_val, 0.0, abs_tol=1e-9):
        return f"0.{'0' * decimals}%"

    sign = ""
    if numeric_val < 0:
        sign = "-"
    elif include_sign and numeric_val > 0:
        sign = "+"

    return f"{sign}{abs(numeric_val):.{decimals}%}"


def fmt_multiple(
    value: Any,
    decimals: int = 2,
    default: str = DEFAULT_MULTIPLE_STR,
) -> str:
    """
    Format a value as a financial multiple safely (e.g. 1.25x).
    """
    decimals = _safe_decimals(decimals, 2)
    numeric_val = parse_float(value, default=float("nan"))
    if pd.isna(numeric_val):
        return default
    return f"{numeric_val:,.{decimals}f}x"


def fmt_days(value: Any, default: str = DEFAULT_DAYS_STR) -> str:
    """
    Format a day-count metric safely.

    Returns default for missing or non-numeric values.
    """
    if is_missing(value):
        return default

    numeric_val = parse_float(value, default=float("nan"))
    if pd.isna(numeric_val):
        return default

    return f"{parse_int(numeric_val)} days"


def fmt_bool(value: Any, true_label: str = "Yes", false_label: str = "No") -> str:
    """
    Format a boolean-like value explicitly.
    """
    if isinstance(value, bool):
        return true_label if value else false_label
    return false_label


def format_metric_value(metric_name: str, value: Any) -> str:
    """
    Robust heuristic formatter for dynamic metric tables.
    Uses substring matching instead of brittle strict tokenization.
    """
    if is_missing(value):
        return "N/A"

    if isinstance(value, bool):
        return fmt_bool(value)

    metric_lower = str(metric_name).strip().lower()

    # 1. Multiples (DSCR, MOIC, Equity Multiple)
    if any(t in metric_lower for t in ["dscr", "multiple", "moic"]):
        return fmt_multiple(value)

    # 2. Currency / Dollars
    curr_keywords = [
        "revenue", "expense", "noi", "value", "rent", "cost", "cash",
        "gain", "balance", "capex", "appreciation", "price", "variance",
    ]
    if any(t in metric_lower for t in curr_keywords):
        return fmt_currency(
            value,
            include_sign=any(t in metric_lower for t in ["gain", "gap", "variance"])
        )

    # 3. Percentages & Rates
    pct_keywords = [
        "cap", "irr", "yield", "ltv", "occupancy", "vacancy",
        "margin", "pct", "rate", "growth",
    ]
    if any(t in metric_lower for t in pct_keywords) and "capex" not in metric_lower:
        return fmt_pct(value, include_sign=True)

    # 4. Fallback for unknown numeric metrics
    num_val = parse_float(value, default=float("nan"))
    if pd.isna(num_val):
        return str(value)

    if abs(num_val) >= 1000:
        return fmt_currency(num_val)

    return f"{num_val:,.2f}"