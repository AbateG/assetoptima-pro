from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd

try:
    import streamlit as st
except Exception:  # pragma: no cover
    st = None

from modules.data_loader import (
    get_covenant_for_property,
    get_property_list,
    get_property_name_map,
    load_debt_covenants,
)

# ============================================================================
# CONSTANTS
# ============================================================================

DSCR_WATCH_BUFFER = 0.10
LTV_WATCH_BUFFER = 0.05

REPORTING_URGENT_DAYS = 15
REPORTING_WARNING_DAYS = 30

MATURITY_URGENT_DAYS = 180
MATURITY_WARNING_DAYS = 365

REFI_DSCR_STRONG = 1.50
REFI_LTV_MAX = 0.70
REFI_DAYS_TO_MATURITY_MAX = 365

RAG_GREEN = "Green"
RAG_AMBER = "Amber"
RAG_RED = "Red"
RAG_GREY = "Grey"

REFI_READY = "Refinance Candidate"
WATCHLIST = "Watchlist"
COMPLIANT = "Compliant"
BREACH = "Breach"


# ============================================================================
# CACHE COMPATIBILITY
# ============================================================================

def _cache_data_compat(*args: Any, **kwargs: Any) -> Callable:
    """Streamlit-compatible cache decorator with no-op fallback."""
    if st is not None:
        return st.cache_data(*args, **kwargs)

    def decorator(func: Callable) -> Callable:
        return func

    return decorator


# ============================================================================
# INTERNAL HELPERS
# ============================================================================

def _safe_float(value: Any, default: float = 0.0) -> float:
    """Safely convert a value to float."""
    try:
        if pd.isna(value):
            return default
        if isinstance(value, str):
            cleaned = value.strip().replace("$", "").replace(",", "").replace("%", "")
            if cleaned == "":
                return default
            return float(cleaned)
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalize_ratio(value: Any, default: float = 0.0) -> float:
    """
    Normalize percentage-like values to decimal form.

    Intended for LTV / interest-rate style values, e.g.
    70 -> 0.70, 0.70 -> 0.70.
    """
    numeric = _safe_float(value, default=default)
    if abs(numeric) > 1.0:
        return numeric / 100.0
    return numeric


def _ltv_headroom_pct_points(max_ltv: float, actual_ltv: float) -> float:
    """Return LTV headroom in percentage points."""
    return (max_ltv - actual_ltv) * 100.0


def _assign_dscr_rag(dscr_actual: float, dscr_required: float) -> str:
    """Assign DSCR RAG based on covenant headroom."""
    headroom = dscr_actual - dscr_required
    if headroom < 0:
        return RAG_RED
    if headroom <= DSCR_WATCH_BUFFER:
        return RAG_AMBER
    return RAG_GREEN


def _assign_ltv_rag(ltv_actual: float, ltv_max: float) -> str:
    """Assign LTV RAG based on covenant headroom."""
    headroom = ltv_max - ltv_actual
    if headroom < 0:
        return RAG_RED
    if headroom <= LTV_WATCH_BUFFER:
        return RAG_AMBER
    return RAG_GREEN


def _assign_deadline_rag(days_remaining: float, urgent_days: int, warning_days: int) -> str:
    """
    Assign deadline RAG.

    Missing deadlines are Grey rather than Green.
    """
    if pd.isna(days_remaining):
        return RAG_GREY
    if days_remaining <= urgent_days:
        return RAG_RED
    if days_remaining <= warning_days:
        return RAG_AMBER
    return RAG_GREEN


def _overall_rag(*statuses: str) -> str:
    """Return worst-case RAG across status inputs."""
    filtered = [s for s in statuses if s in {RAG_GREEN, RAG_AMBER, RAG_RED, RAG_GREY}]
    if RAG_RED in filtered:
        return RAG_RED
    if RAG_AMBER in filtered:
        return RAG_AMBER
    if RAG_GREY in filtered:
        return RAG_GREY
    return RAG_GREEN


def _coerce_covenant_df(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce expected covenant columns and fill missing optional columns."""
    df = df.copy()

    defaults: dict[str, Any] = {
        "Property_ID": None,
        "Property_Name": None,
        "Loan_Name": "",
        "Lender": "",
        "Loan_Type": "",
        "Loan_Balance": 0.0,
        "Interest_Rate": 0.0,
        "DSCR_Actual": 0.0,
        "DSCR_Requirement": 0.0,
        "LTV": 0.0,
        "Max_LTV_Allowed": 0.0,
        "Days_to_Reporting": np.nan,
        "Days_to_Maturity": np.nan,
        "Annual_Debt_Service": 0.0,
    }

    for col, default in defaults.items():
        if col not in df.columns:
            df[col] = default

    numeric_cols = [
        "Loan_Balance",
        "DSCR_Actual",
        "DSCR_Requirement",
        "Days_to_Reporting",
        "Days_to_Maturity",
        "Annual_Debt_Service",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["Interest_Rate"] = df["Interest_Rate"].apply(lambda x: _normalize_ratio(x, default=0.0))
    df["LTV"] = df["LTV"].apply(lambda x: _normalize_ratio(x, default=0.0))
    df["Max_LTV_Allowed"] = df["Max_LTV_Allowed"].apply(lambda x: _normalize_ratio(x, default=0.0))

    return df


def _get_covenant_rows(property_id: str) -> pd.DataFrame:
    """
    Return all covenant rows for one property.

    Raises
    ------
    ValueError
        If no covenant rows exist for the property.
    """
    rows = get_covenant_for_property(property_id)
    if rows is None or len(rows) == 0:
        raise ValueError(f"No debt covenant record found for property: {property_id}")
    return _coerce_covenant_df(rows)


def _row_to_summary(row: pd.Series, fallback_property_id: str) -> dict[str, Any]:
    """Convert one covenant row to a standardized summary dict."""
    dscr_actual = _safe_float(row.get("DSCR_Actual"), default=0.0)
    dscr_required = _safe_float(row.get("DSCR_Requirement"), default=0.0)

    ltv_actual = _normalize_ratio(row.get("LTV"), default=0.0)
    ltv_max = _normalize_ratio(row.get("Max_LTV_Allowed"), default=0.0)

    dscr_headroom = dscr_actual - dscr_required
    ltv_headroom_pct_pts = _ltv_headroom_pct_points(ltv_max, ltv_actual)

    dscr_rag = _assign_dscr_rag(dscr_actual, dscr_required)
    ltv_rag = _assign_ltv_rag(ltv_actual, ltv_max)

    days_to_reporting = _safe_float(row.get("Days_to_Reporting"), default=np.nan)
    days_to_maturity = _safe_float(row.get("Days_to_Maturity"), default=np.nan)

    reporting_rag = _assign_deadline_rag(
        days_remaining=days_to_reporting,
        urgent_days=REPORTING_URGENT_DAYS,
        warning_days=REPORTING_WARNING_DAYS,
    )
    maturity_rag = _assign_deadline_rag(
        days_remaining=days_to_maturity,
        urgent_days=MATURITY_URGENT_DAYS,
        warning_days=MATURITY_WARNING_DAYS,
    )

    annual_debt_service = _safe_float(row.get("Annual_Debt_Service"), default=0.0)
    implied_noi_required = dscr_required * annual_debt_service

    overall_rag = _overall_rag(dscr_rag, ltv_rag, reporting_rag, maturity_rag)

    if dscr_rag == RAG_RED or ltv_rag == RAG_RED:
        compliance_label = BREACH
    elif overall_rag in {RAG_RED, RAG_AMBER}:
        compliance_label = WATCHLIST
    elif (
        dscr_actual >= REFI_DSCR_STRONG
        and ltv_actual <= REFI_LTV_MAX
        and pd.notna(days_to_maturity)
        and days_to_maturity <= REFI_DAYS_TO_MATURITY_MAX
    ):
        compliance_label = REFI_READY
    else:
        compliance_label = COMPLIANT

    return {
        "Property_ID": row.get("Property_ID", fallback_property_id),
        "Property_Name": row.get("Property_Name", fallback_property_id),
        "Loan_Name": row.get("Loan_Name", ""),
        "Lender": row.get("Lender", ""),
        "Loan_Type": row.get("Loan_Type", ""),
        "Loan_Balance": _safe_float(row.get("Loan_Balance"), default=0.0),
        "Interest_Rate": _normalize_ratio(row.get("Interest_Rate"), default=0.0),
        "DSCR_Actual": dscr_actual,
        "DSCR_Requirement": dscr_required,
        "DSCR_Headroom": dscr_headroom,
        "DSCR_RAG": dscr_rag,
        "LTV_Actual": ltv_actual,
        "LTV_Max": ltv_max,
        "LTV_Headroom_Pct_Pts": ltv_headroom_pct_pts,
        "LTV_RAG": ltv_rag,
        "Days_to_Reporting": days_to_reporting,
        "Reporting_RAG": reporting_rag,
        "Days_to_Maturity": days_to_maturity,
        "Maturity_RAG": maturity_rag,
        "Annual_Debt_Service": annual_debt_service,
        "Implied_NOI_Required": implied_noi_required,
        "Overall_RAG": overall_rag,
        "Compliance_Label": compliance_label,
    }


def _worst_label(labels: pd.Series) -> str:
    """Return worst-case compliance label."""
    if BREACH in set(labels):
        return BREACH
    if WATCHLIST in set(labels):
        return WATCHLIST
    if REFI_READY in set(labels):
        return REFI_READY
    return COMPLIANT


def _worst_rag(statuses: pd.Series) -> str:
    """Return worst-case RAG from a series."""
    ranked = {RAG_GREEN: 0, RAG_GREY: 1, RAG_AMBER: 2, RAG_RED: 3}
    inv = {v: k for k, v in ranked.items()}
    max_rank = statuses.map(ranked).fillna(0).max()
    return inv.get(int(max_rank), RAG_GREEN)


# ============================================================================
# PUBLIC FUNCTIONS
# ============================================================================

@_cache_data_compat(show_spinner=False)
def get_property_compliance_summary(property_id: str) -> dict[str, Any]:
    """
    Return property-level debt compliance summary.

    If multiple loan rows exist for a property, this function rolls them up to
    a single property-level summary using conservative worst-case logic.
    """
    rows = _get_covenant_rows(property_id)
    loan_summaries = pd.DataFrame([_row_to_summary(row, property_id) for _, row in rows.iterrows()])

    if loan_summaries.empty:
        raise ValueError(f"No debt covenant record found for property: {property_id}")

    weighted_balance = loan_summaries["Loan_Balance"].fillna(0.0)
    total_balance = float(weighted_balance.sum())

    def _weighted_avg(col: str) -> float:
        if total_balance > 0:
            return float((loan_summaries[col] * weighted_balance).sum() / total_balance)
        return float(loan_summaries[col].mean()) if len(loan_summaries) > 0 else 0.0

    summary = {
        "Property_ID": property_id,
        "Property_Name": str(loan_summaries["Property_Name"].iloc[0]),
        "Loan_Count": int(len(loan_summaries)),
        "Primary_Loan_Name": str(loan_summaries["Loan_Name"].iloc[0]) if len(loan_summaries) > 0 else "",
        "Primary_Lender": str(loan_summaries["Lender"].iloc[0]) if len(loan_summaries) > 0 else "",
        "Total_Loan_Balance": total_balance,
        "Weighted_Interest_Rate": _weighted_avg("Interest_Rate"),
        "Weighted_DSCR_Actual": _weighted_avg("DSCR_Actual"),
        "Weighted_DSCR_Requirement": _weighted_avg("DSCR_Requirement"),
        "Weighted_DSCR_Headroom": _weighted_avg("DSCR_Headroom"),
        "Worst_DSCR_RAG": _worst_rag(loan_summaries["DSCR_RAG"]),
        "Weighted_LTV_Actual": _weighted_avg("LTV_Actual"),
        "Weighted_LTV_Max": _weighted_avg("LTV_Max"),
        "Weighted_LTV_Headroom_Pct_Pts": _weighted_avg("LTV_Headroom_Pct_Pts"),
        "Worst_LTV_RAG": _worst_rag(loan_summaries["LTV_RAG"]),
        "Nearest_Reporting_Days": (
            float(loan_summaries["Days_to_Reporting"].min())
            if loan_summaries["Days_to_Reporting"].notna().any()
            else np.nan
        ),
        "Nearest_Maturity_Days": (
            float(loan_summaries["Days_to_Maturity"].min())
            if loan_summaries["Days_to_Maturity"].notna().any()
            else np.nan
        ),
        "Reporting_RAG": _worst_rag(loan_summaries["Reporting_RAG"]),
        "Maturity_RAG": _worst_rag(loan_summaries["Maturity_RAG"]),
        "Annual_Debt_Service": float(loan_summaries["Annual_Debt_Service"].sum()),
        "Implied_NOI_Required": float(loan_summaries["Implied_NOI_Required"].sum()),
        "Overall_RAG": _worst_rag(loan_summaries["Overall_RAG"]),
        "Compliance_Label": _worst_label(loan_summaries["Compliance_Label"]),
    }

    # Backward-compatible field aliases for older consumers.
    summary["Loan_Name"] = summary["Primary_Loan_Name"]
    summary["Lender"] = summary["Primary_Lender"]
    summary["Loan_Balance"] = summary["Total_Loan_Balance"]
    summary["Interest_Rate"] = summary["Weighted_Interest_Rate"]
    summary["DSCR_Actual"] = summary["Weighted_DSCR_Actual"]
    summary["DSCR_Requirement"] = summary["Weighted_DSCR_Requirement"]
    summary["DSCR_Headroom"] = summary["Weighted_DSCR_Headroom"]
    summary["DSCR_RAG"] = summary["Worst_DSCR_RAG"]
    summary["LTV_Actual"] = summary["Weighted_LTV_Actual"]
    summary["LTV_Max"] = summary["Weighted_LTV_Max"]
    summary["LTV_Headroom_Pct_Pts"] = summary["Weighted_LTV_Headroom_Pct_Pts"]
    summary["LTV_RAG"] = summary["Worst_LTV_RAG"]
    summary["Days_to_Reporting"] = summary["Nearest_Reporting_Days"]
    summary["Days_to_Maturity"] = summary["Nearest_Maturity_Days"]

    return summary


@_cache_data_compat(show_spinner=False)
def get_compliance_summary_table() -> pd.DataFrame:
    """Return property-level debt compliance summary table."""
    rows = []
    property_name_map = get_property_name_map()

    for property_id in get_property_list():
        try:
            summary = get_property_compliance_summary(property_id)
        except ValueError:
            continue

        rows.append(
            {
                "Property_ID": property_id,
                "Property_Name": property_name_map.get(property_id, summary.get("Property_Name", property_id)),
                "Loan_Count": summary["Loan_Count"],
                "Total_Loan_Balance": summary["Total_Loan_Balance"],
                "DSCR_Actual": summary["DSCR_Actual"],
                "DSCR_Requirement": summary["DSCR_Requirement"],
                "DSCR_Headroom": summary["DSCR_Headroom"],
                "DSCR_RAG": summary["DSCR_RAG"],
                "LTV_Actual": summary["LTV_Actual"],
                "LTV_Max": summary["LTV_Max"],
                "LTV_Headroom_Pct_Pts": summary["LTV_Headroom_Pct_Pts"],
                "LTV_RAG": summary["LTV_RAG"],
                "Days_to_Reporting": summary["Days_to_Reporting"],
                "Days_to_Maturity": summary["Days_to_Maturity"],
                "Overall_RAG": summary["Overall_RAG"],
                "Compliance_Label": summary["Compliance_Label"],
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    rag_sort = {RAG_RED: 0, RAG_AMBER: 1, RAG_GREY: 2, RAG_GREEN: 3}
    df["RAG_Sort"] = df["Overall_RAG"].map(rag_sort).fillna(9)

    return (
        df.sort_values(["RAG_Sort", "Days_to_Maturity"], ascending=[True, True], na_position="last")
        .drop(columns="RAG_Sort")
        .reset_index(drop=True)
    )


@_cache_data_compat(show_spinner=False)
def get_portfolio_compliance_kpis() -> dict[str, Any]:
    """Return portfolio-level debt compliance KPIs."""
    summary_table = get_compliance_summary_table()
    if summary_table.empty:
        return {
            "Property_Count": 0,
            "Loan_Count": 0,
            "Breach_Count": 0,
            "Watchlist_Count": 0,
            "Compliant_Count": 0,
            "Refinance_Candidate_Count": 0,
            "Average_DSCR": 0.0,
            "Average_LTV": 0.0,
            "Nearest_Reporting_Days": np.nan,
            "Nearest_Maturity_Days": np.nan,
            "Total_Loan_Balance": 0.0,
        }

    breach_count = int((summary_table["Compliance_Label"] == BREACH).sum())
    watchlist_count = int((summary_table["Compliance_Label"] == WATCHLIST).sum())
    compliant_count = int((summary_table["Compliance_Label"] == COMPLIANT).sum())
    refinance_candidate_count = int((summary_table["Compliance_Label"] == REFI_READY).sum())

    average_dscr = float(summary_table["DSCR_Actual"].mean())
    average_ltv = float(summary_table["LTV_Actual"].mean())

    nearest_reporting_days = (
        float(summary_table["Days_to_Reporting"].min())
        if summary_table["Days_to_Reporting"].notna().any()
        else np.nan
    )
    nearest_maturity_days = (
        float(summary_table["Days_to_Maturity"].min())
        if summary_table["Days_to_Maturity"].notna().any()
        else np.nan
    )

    return {
        "Property_Count": int(len(summary_table)),
        "Loan_Count": int(summary_table["Loan_Count"].sum()) if "Loan_Count" in summary_table.columns else int(len(summary_table)),
        "Breach_Count": breach_count,
        "Watchlist_Count": watchlist_count,
        "Compliant_Count": compliant_count,
        "Refinance_Candidate_Count": refinance_candidate_count,
        "Average_DSCR": average_dscr,
        "Average_LTV": average_ltv,
        "Nearest_Reporting_Days": nearest_reporting_days,
        "Nearest_Maturity_Days": nearest_maturity_days,
        "Total_Loan_Balance": float(summary_table["Total_Loan_Balance"].sum()) if "Total_Loan_Balance" in summary_table.columns else 0.0,
    }


@_cache_data_compat(show_spinner=False)
def get_covenant_breach_table() -> pd.DataFrame:
    """Return property-level table of debt breaches."""
    rows = []

    for property_id in get_property_list():
        try:
            summary = get_property_compliance_summary(property_id)
        except ValueError:
            continue

        if summary["Compliance_Label"] == BREACH:
            rows.append(summary)

    return pd.DataFrame(rows).reset_index(drop=True)


@_cache_data_compat(show_spinner=False)
def get_upcoming_deadlines(days_ahead: int = 45) -> pd.DataFrame:
    """
    Return all debt reporting / maturity deadlines within the specified horizon.

    Operates at the loan-row level so multiple deadlines per property are preserved.
    """
    if days_ahead < 0:
        return pd.DataFrame()

    all_rows = _coerce_covenant_df(load_debt_covenants().copy())
    if all_rows.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []

    for _, row in all_rows.iterrows():
        property_id = row.get("Property_ID")
        property_name = row.get("Property_Name", property_id)
        loan_name = row.get("Loan_Name", "")

        days_reporting = _safe_float(row.get("Days_to_Reporting"), default=np.nan)
        if pd.notna(days_reporting) and days_reporting <= days_ahead:
            rows.append(
                {
                    "Property_ID": property_id,
                    "Property_Name": property_name,
                    "Loan_Name": loan_name,
                    "Deadline_Type": "Reporting",
                    "Days_Remaining": days_reporting,
                    "RAG_Status": _assign_deadline_rag(
                        days_reporting,
                        REPORTING_URGENT_DAYS,
                        REPORTING_WARNING_DAYS,
                    ),
                }
            )

        days_maturity = _safe_float(row.get("Days_to_Maturity"), default=np.nan)
        if pd.notna(days_maturity) and days_maturity <= days_ahead:
            rows.append(
                {
                    "Property_ID": property_id,
                    "Property_Name": property_name,
                    "Loan_Name": loan_name,
                    "Deadline_Type": "Maturity",
                    "Days_Remaining": days_maturity,
                    "RAG_Status": _assign_deadline_rag(
                        days_maturity,
                        MATURITY_URGENT_DAYS,
                        MATURITY_WARNING_DAYS,
                    ),
                }
            )

    deadline_df = pd.DataFrame(rows)
    if deadline_df.empty:
        return deadline_df

    rag_sort = {RAG_RED: 0, RAG_AMBER: 1, RAG_GREY: 2, RAG_GREEN: 3}
    deadline_df["RAG_Sort"] = deadline_df["RAG_Status"].map(rag_sort).fillna(9)

    return (
        deadline_df.sort_values(["RAG_Sort", "Days_Remaining", "Deadline_Type"], ascending=[True, True, True], na_position="last")
        .drop(columns="RAG_Sort")
        .reset_index(drop=True)
    )


@_cache_data_compat(show_spinner=False)
def get_refinance_watchlist() -> pd.DataFrame:
    """
    Return property-level refinance watchlist.

    Includes breach/watchlist properties plus near-maturity properties.
    """
    rows = []

    for property_id in get_property_list():
        try:
            summary = get_property_compliance_summary(property_id)
        except ValueError:
            continue

        include = (
            summary["Compliance_Label"] in {REFI_READY, WATCHLIST, BREACH}
            or (
                pd.notna(summary["Days_to_Maturity"])
                and summary["Days_to_Maturity"] <= REFI_DAYS_TO_MATURITY_MAX
            )
        )

        if include:
            rows.append(
                {
                    "Property_ID": summary["Property_ID"],
                    "Property_Name": summary["Property_Name"],
                    "Loan_Count": summary["Loan_Count"],
                    "Primary_Loan_Name": summary["Primary_Loan_Name"],
                    "Total_Loan_Balance": summary["Total_Loan_Balance"],
                    "DSCR_Actual": summary["DSCR_Actual"],
                    "LTV_Actual": summary["LTV_Actual"],
                    "Days_to_Maturity": summary["Days_to_Maturity"],
                    "Days_to_Reporting": summary["Days_to_Reporting"],
                    "Compliance_Label": summary["Compliance_Label"],
                    "Overall_RAG": summary["Overall_RAG"],
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    sort_priority = {BREACH: 0, WATCHLIST: 1, REFI_READY: 2, COMPLIANT: 3}
    df["Sort_Priority"] = df["Compliance_Label"].map(sort_priority).fillna(9)

    return (
        df.sort_values(["Sort_Priority", "Days_to_Maturity"], ascending=[True, True], na_position="last")
        .drop(columns="Sort_Priority")
        .reset_index(drop=True)
    )


@_cache_data_compat(show_spinner=False)
def get_lender_exposure_summary() -> pd.DataFrame:
    """Return lender-level exposure summary."""
    df = _coerce_covenant_df(load_debt_covenants().copy())
    if df.empty or "Lender" not in df.columns:
        return pd.DataFrame()

    summary = (
        df.groupby("Lender", dropna=False)
        .agg(
            Loan_Count=("Lender", "count"),
            Property_Count=("Property_ID", "nunique"),
            Total_Loan_Balance=("Loan_Balance", "sum"),
            Average_DSCR=("DSCR_Actual", "mean"),
            Average_LTV=("LTV", "mean"),
            Nearest_Maturity_Days=("Days_to_Maturity", "min"),
        )
        .reset_index()
    )

    return summary.sort_values("Total_Loan_Balance", ascending=False).reset_index(drop=True)