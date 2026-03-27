from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from utils.coercion import safe_series_numeric

logger = logging.getLogger(__name__)


def has_columns(df: pd.DataFrame, columns: list[str]) -> bool:
    """
    Return True if ALL required columns exist in the DataFrame.
    """
    return isinstance(df, pd.DataFrame) and all(col in df.columns for col in columns)


def is_valid_dataframe(
    df: Any,
    required_columns: list[str] | None = None,
    min_rows: int = 0,
) -> bool:
    """
    Validate a DataFrame for downstream use.
    """
    if not isinstance(df, pd.DataFrame):
        return False
    if required_columns and not has_columns(df, required_columns):
        return False
    return len(df) >= min_rows


def first_existing_column(
    df: pd.DataFrame,
    candidates: list[str],
    default: str | None = None,
) -> str | None:
    """
    Return the first column in *candidates* that exists in *df*.
    """
    if not isinstance(df, pd.DataFrame):
        return default
    for col in candidates:
        if col in df.columns:
            return col
    return default


def safe_mean(
    df: pd.DataFrame,
    column: str,
    default: float = 0.0,
) -> float:
    """
    Safely compute the mean of a DataFrame column.
    """
    if not is_valid_dataframe(df, [column], min_rows=1):
        return default
    series = safe_series_numeric(df[column]).dropna()
    return float(series.mean()) if not series.empty else default


def safe_sum(
    df: pd.DataFrame,
    column: str,
    default: float = 0.0,
) -> float:
    """
    Safely compute the sum of a DataFrame column.
    """
    if not is_valid_dataframe(df, [column], min_rows=1):
        return default
    series = safe_series_numeric(df[column]).dropna()
    return float(series.sum()) if not series.empty else default


def validate_market_data(data: Any) -> tuple[bool, list[str]]:
    """
    Validate a market data bundle dictionary.

    Checks that *data* is a non-empty dict and that every expected top-level
    key is present. Missing keys are collected into an error list rather
    than raising immediately so the caller can decide how to handle partial
    data.

    Returns
    -------
    tuple[bool, list[str]]
        (is_valid, error_messages)
    """
    errors: list[str] = []

    if not isinstance(data, dict):
        errors.append(f"Market data must be a dict, got {type(data).__name__}")
        return False, errors

    if not data:
        errors.append("Market data dict is empty")
        return False, errors

    required_keys: list[str] = [
        "summary",
        "comp_table",
        "rent_distribution",
        "occupancy_comparison",
        "positioning",
        "narrative",
        "flags",
        "warnings",
    ]
    for key in required_keys:
        if key not in data:
            errors.append(f"Missing required key: {key}")

    df_keys = [
        "comp_table",
        "rent_distribution",
        "occupancy_comparison",
        "peer_benchmarking",
    ]
    for key in df_keys:
        if key in data and not isinstance(data[key], pd.DataFrame):
            errors.append(
                f"Key '{key}' must be a DataFrame, got {type(data[key]).__name__}"
            )

    dict_keys = [
        "summary",
        "positioning",
        "market_trends",
        "market_heatmap",
        "market_sentiment",
        "forecast_indicators",
        "geographic_data",
    ]
    for key in dict_keys:
        if key in data and not isinstance(data[key], dict):
            errors.append(
                f"Key '{key}' must be a dict, got {type(data[key]).__name__}"
            )

    list_keys = ["flags", "warnings"]
    for key in list_keys:
        if key in data and not isinstance(data[key], list):
            errors.append(
                f"Key '{key}' must be a list, got {type(data[key]).__name__}"
            )

    if "narrative" in data and not isinstance(data["narrative"], str):
        errors.append(
            f"Key 'narrative' must be a str, got {type(data['narrative']).__name__}"
        )

    if errors:
        logger.debug(
            "validate_market_data found %d issue(s): %s",
            len(errors),
            errors,
        )

    return len(errors) == 0, errors


def check_data_integrity(
    df: pd.DataFrame,
    required_columns: list[str] | None = None,
    min_rows: int = 1,
    max_null_pct: float = 0.50,
    label: str = "DataFrame",
) -> tuple[bool, list[str]]:
    """
    Check a DataFrame for common data-integrity issues.

    Runs the following checks in order and collects all failures:
    1. Is the object actually a DataFrame?
    2. Does it meet the minimum row count?
    3. Are all required columns present?
    4. Does any required column exceed the null threshold?
    """
    issues: list[str] = []

    if not isinstance(df, pd.DataFrame):
        issues.append(f"{label}: expected a DataFrame, got {type(df).__name__}")
        return False, issues

    if len(df) < min_rows:
        issues.append(
            f"{label}: has {len(df)} row(s), minimum required is {min_rows}"
        )

    if required_columns:
        for col in required_columns:
            if col not in df.columns:
                issues.append(f"{label}: missing required column '{col}'")
                continue

            if len(df) > 0:
                null_pct = float(df[col].isna().mean())
                if null_pct > max_null_pct:
                    issues.append(
                        f"{label}['{col}']: {null_pct:.0%} null values "
                        f"(threshold {max_null_pct:.0%})"
                    )

    if issues:
        logger.debug(
            "check_data_integrity found %d issue(s) in %s: %s",
            len(issues),
            label,
            issues,
        )

    return len(issues) == 0, issues