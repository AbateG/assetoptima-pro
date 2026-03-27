"""
Type coercion and safe parsing helpers for AssetOptima Pro.
"""

from __future__ import annotations

from typing import Any

import math
import pandas as pd


def ensure_dataframe(value: Any) -> pd.DataFrame:
    """
    Return a DataFrame if value is a DataFrame; otherwise return an empty DataFrame.
    """
    return value if isinstance(value, pd.DataFrame) else pd.DataFrame()


def ensure_dict(value: Any) -> dict[str, Any]:
    """
    Return a dict if value is a dict; otherwise return an empty dict.
    """
    return value if isinstance(value, dict) else {}


def ensure_list(value: Any) -> list[Any]:
    """
    Return a list if value is a list; otherwise return an empty list.
    """
    return value if isinstance(value, list) else []


def is_missing(value: Any) -> bool:
    """
    Return True for scalar missing values such as None, NaN, and pandas NA.

    This helper is intentionally scalar-safe and avoids ambiguous truth-value
    errors that can occur when calling pd.isna(...) on list-like objects.
    """
    if value is None:
        return True

    # Avoid treating container-like values as scalar missing checks.
    if isinstance(value, (list, tuple, dict, set, pd.Series, pd.DataFrame)):
        return False

    try:
        result = pd.isna(value)
        return bool(result) if isinstance(result, (bool, int)) else False
    except Exception:
        return False


def safe_str(value: Any, default: str = "") -> str:
    """
    Safely convert a value to string.

    Returns default for None/NaN-like scalar values or conversion failures.
    """
    try:
        if is_missing(value):
            return default
        return str(value)
    except Exception:
        return default


def parse_float(value: Any, default: float = 0.0) -> float:
    """
    Safely parse a value into float.

    Handles:
    - None / NaN
    - numeric types
    - strings like '\$1,234', '95%', '1,250.5'

    Notes
    -----
    - Percent-like strings are parsed as their literal numeric content:
      '95%' -> 95.0
      '12.5%' -> 12.5

      This function does NOT convert percentages to decimals. That decision is
      left to higher-level formatting/parsing logic.
    """
    try:
        if is_missing(value):
            return default

        if isinstance(value, bool):
            return float(value)

        if isinstance(value, str):
            cleaned = (
                value.strip()
                .replace(",", "")
                .replace("$", "")
                .replace("%", "")
            )
            if cleaned == "":
                return default
            return float(cleaned)

        return float(value)
    except (TypeError, ValueError):
        return default


def parse_int(value: Any, default: int = 0) -> int:
    """
    Safely parse a value into int by truncating toward zero.

    Examples
    --------
    '3.9' -> 3
    '-3.9' -> -3
    """
    try:
        numeric_val = parse_float(value, default=float("nan"))
        if math.isnan(numeric_val):
            return default
        return int(numeric_val)
    except (TypeError, ValueError):
        return default


def safe_series_numeric(series: pd.Series) -> pd.Series:
    """
    Coerce a pandas Series to numeric, setting invalid values to NaN.
    """
    return pd.to_numeric(series, errors="coerce")