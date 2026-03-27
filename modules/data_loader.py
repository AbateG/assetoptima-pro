from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

try:
    import streamlit as st
except Exception:  # pragma: no cover
    st = None


# ============================================================================
# SECTION 0 — PATH CONFIGURATION
# ============================================================================

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

FILE_PATHS: dict[str, Path] = {
    "properties": DATA_DIR / "properties.csv",
    "monthly_performance": DATA_DIR / "monthly_performance.csv",
    "market_comps": DATA_DIR / "market_comps.csv",
    "business_plan": DATA_DIR / "business_plan.csv",
    "debt_covenants": DATA_DIR / "debt_covenants.csv",
    "underwriting": DATA_DIR / "underwriting_assumptions.csv",
}


# ============================================================================
# SECTION 1 — CANONICAL SCHEMA DEFINITIONS
# ============================================================================

CANONICAL_REQUIRED_COLUMNS: dict[str, list[str]] = {
    "properties": [
        "Property_ID",
        "Property_Name",
        "Market",
        "Units",
        "Acquisition_Date",
        "Purchase_Price",
        "Current_Value",
        "Stabilized_Occupancy",
        "Current_Occupancy",
        "Avg_Rent_per_Unit",
        "Value_Add_Phase",
    ],
    "monthly_performance": [
        "Property_ID",
        "Year_Month",
        "Budgeted_Revenue",
        "Actual_Revenue",
        "Budgeted_Expenses",
        "Actual_Expenses",
        "Budgeted_NOI",
        "Actual_NOI",
        "Occupancy",
        "Avg_Actual_Rent",
    ],
    "market_comps": [
        "Property_ID",
        "Comp_Name",
        "Comp_Market",
        "Avg_Rent",
        "Comp_Occupancy",
        "Rent_Growth_YoY",
    ],
    "business_plan": [
        "Property_ID",
        "Initiative_ID",
        "Initiative",
        "Category",
        "Target_Completion",
        "Status",
        "Percent_Complete",
        "Budget",
        "Actual_Spend",
        "Expected_NOI_Lift",
    ],
    "debt_covenants": [
        "Property_ID",
        "Loan_Balance",
        "Interest_Rate",
        "Loan_Type",
        "DSCR_Requirement",
        "DSCR_Actual",
        "LTV",
        "Max_LTV_Allowed",
        "Next_Reporting_Date",
        "Maturity_Date",
        "Annual_Debt_Service",
    ],
    "underwriting": [
        "Property_ID",
        "Underwritten_Occupancy",
        "Underwritten_Rent_Growth_Pct",
        "Underwritten_Expense_Growth_Pct",
        "Underwritten_Exit_Cap_Rate",
        "Underwritten_NOI_Year1",
        "Underwritten_IRR",
        "Target_Hold_Years",
        "Underwritten_Purchase_Cap_Rate",
    ],
}

OPTIONAL_COLUMNS: dict[str, list[str]] = {
    "properties": [
        "Year_Built",
        "Parking_Spaces",
        "Amenity_Score",
        "Submarket",
    ],
    "monthly_performance": [
        "Concessions",
        "Bad_Debt",
        "Repairs_Maintenance",
        "Payroll",
        "Marketing",
        "Utilities",
        "Insurance",
        "Admin",
    ],
    "market_comps": [
        "Distance_Miles",
        "Units",
        "Amenity_Score",
        "Year_Built",
        "Unit_Class",
    ],
    "business_plan": [
        "Risk_Level",
        "Owner",
        "Notes",
    ],
    "debt_covenants": [
        "Loan_Name",
        "Lender",
    ],
    "underwriting": [
        "Underwritten_EGI_Year1",
        "Underwritten_OpEx_Year1",
        "Underwritten_Vacancy_Pct",
        "Underwritten_CapEx_Reserve_Per_Unit",
    ],
}

COLUMN_ALIASES: dict[str, dict[str, str]] = {
    "properties": {
        "Region": "Market",
        "Asset_Phase": "Value_Add_Phase",
        "Unit_Count": "Units",
        "Acquisition_Value": "Purchase_Price",
    },
    "monthly_performance": {
        "Budget_NOI": "Budgeted_NOI",
        "Budget_Revenue": "Budgeted_Revenue",
        "Budget_Expenses": "Budgeted_Expenses",
        "Actual_Occupancy": "Occupancy",
        "Month": "Year_Month",
    },
    "market_comps": {
        "Market_Rent": "Avg_Rent",
        "Comp_Monthly_Rent": "Avg_Rent",
        "Occupancy": "Comp_Occupancy",
        "Comp_Rent_Growth": "Rent_Growth_YoY",
    },
    "business_plan": {},
    "debt_covenants": {
        "LTV_Actual": "LTV",
        "LTV_Max": "Max_LTV_Allowed",
        "Max_LTV": "Max_LTV_Allowed",
    },
    "underwriting": {
        "Exit_Cap_Rate": "Underwritten_Exit_Cap_Rate",
        "Purchase_Cap_Rate": "Underwritten_Purchase_Cap_Rate",
        "Year_1_NOI": "Underwritten_NOI_Year1",
        "Target_IRR": "Underwritten_IRR",
    },
}


# ============================================================================
# SECTION 2 — SHARED CONSTANTS
# ============================================================================

RAG_REVENUE_WARN_PCT = -3.0
RAG_REVENUE_CRIT_PCT = -8.0
RAG_EXPENSE_WARN_PCT = 3.0
RAG_EXPENSE_CRIT_PCT = 8.0
RAG_NOI_WARN_PCT = -3.0
RAG_NOI_CRIT_PCT = -8.0

DSCR_WATCH_BUFFER = 0.10
LTV_WATCH_BUFFER = 0.05
REPORTING_CRITICAL_DAYS = 7
REPORTING_URGENT_DAYS = 14

VALUE_ADD_PHASE_ORDER = [
    "Lease_Up",
    "Renovation_In_Progress",
    "Renovation_Complete",
    "Stabilized",
]


# ============================================================================
# SECTION 3 — CACHE COMPATIBILITY
# ============================================================================

def _cache_data_compat(*args: Any, **kwargs: Any) -> Callable:
    """
    Streamlit-compatible cache decorator with a no-op fallback for non-Streamlit
    execution environments.
    """
    if st is not None:
        return st.cache_data(*args, **kwargs)

    def decorator(func: Callable) -> Callable:
        return func

    return decorator


# ============================================================================
# SECTION 4 — PRIVATE UTILITY FUNCTIONS
# ============================================================================

def _safe_read_csv(file_key: str) -> pd.DataFrame:
    """Read one CSV from the data directory with all fields initially as strings."""
    path = FILE_PATHS.get(file_key)
    if path is None:
        raise KeyError(
            f"[DataLoader] Unknown file key '{file_key}'. "
            f"Valid keys: {list(FILE_PATHS.keys())}"
        )
    if not path.exists():
        raise FileNotFoundError(
            f"\n[DataLoader] Required data file not found:\n"
            f"  Expected path : {path}\n"
            f"  Action needed : Create '{path.name}' inside the data/ folder.\n"
            f"  All required CSV files must be present before running the app."
        )
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def _normalize_column_aliases(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """Normalize legacy/alias column names to canonical names."""
    alias_map = COLUMN_ALIASES.get(dataset_name, {})
    rename_map: dict[str, str] = {}

    for alias, canonical in alias_map.items():
        if alias in df.columns and canonical not in df.columns:
            rename_map[alias] = canonical

    if rename_map:
        df = df.rename(columns=rename_map)

    duplicate_aliases = [
        alias for alias, canonical in alias_map.items()
        if alias in df.columns and canonical in df.columns
    ]
    if duplicate_aliases:
        df = df.drop(columns=duplicate_aliases)

    return df


def _validate_columns(df: pd.DataFrame, dataset_name: str) -> None:
    """Validate that canonical required columns exist."""
    required = CANONICAL_REQUIRED_COLUMNS.get(dataset_name, [])
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            f"\n[DataLoader] Schema validation failed for '{dataset_name}'.\n"
            f"  Missing canonical columns : {missing}\n"
            f"  Found columns             : {list(df.columns)}\n"
            f"  Check CSV structure and alias mappings."
        )


def _ensure_optional_columns(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """Ensure optional columns exist so downstream modules see a stable schema."""
    df = df.copy()
    for col in OPTIONAL_COLUMNS.get(dataset_name, []):
        if col not in df.columns:
            df[col] = np.nan
    return df


def _cast_numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Cast existing columns to numeric."""
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _normalize_percent_like_series(series: pd.Series) -> pd.Series:
    """
    Normalize percentage-like values to decimal form.

    Examples
    --------
    95   -> 0.95
    0.95 -> 0.95
    5    -> 0.05
    """
    numeric = pd.to_numeric(series, errors="coerce")
    return pd.Series(
        np.where(numeric.abs() > 1.0, numeric / 100.0, numeric),
        index=series.index,
    )


def _normalize_category_values(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """Normalize selected categorical fields to stable canonical values."""
    if dataset_name == "properties" and "Value_Add_Phase" in df.columns:
        phase_map = {
            "lease up": "Lease_Up",
            "lease_up": "Lease_Up",
            "lease-up": "Lease_Up",
            "renovation in progress": "Renovation_In_Progress",
            "renovation_in_progress": "Renovation_In_Progress",
            "renovation-in-progress": "Renovation_In_Progress",
            "renovation complete": "Renovation_Complete",
            "renovation_complete": "Renovation_Complete",
            "renovation-complete": "Renovation_Complete",
            "stabilized": "Stabilized",
        }
        norm = (
            df["Value_Add_Phase"]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.lower()
            .map(phase_map)
        )
        df["Value_Add_Phase"] = np.where(
            pd.notna(norm),
            norm,
            df["Value_Add_Phase"].fillna("").astype(str).str.strip(),
        )

    if dataset_name == "business_plan" and "Status" in df.columns:
        status_map = {
            "complete": "Complete",
            "completed": "Complete",
            "in progress": "In_Progress",
            "in_progress": "In_Progress",
            "not started": "Not_Started",
            "not_started": "Not_Started",
        }
        norm = (
            df["Status"]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.lower()
            .map(status_map)
        )
        df["Status"] = np.where(
            pd.notna(norm),
            norm,
            df["Status"].fillna("").astype(str).str.strip(),
        )

    if dataset_name == "business_plan" and "Risk_Level" in df.columns:
        risk_map = {
            "low": "Low",
            "medium": "Medium",
            "high": "High",
        }
        norm = (
            df["Risk_Level"]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.lower()
            .map(risk_map)
        )
        df["Risk_Level"] = np.where(
            pd.notna(norm),
            norm,
            df["Risk_Level"].fillna("").astype(str).str.strip(),
        )

    return df


def _rag_from_variance(
    series: pd.Series,
    warn_threshold: float,
    crit_threshold: float,
    unfavorable_direction: str = "negative",
) -> pd.Series:
    """Assign a RAG label to a variance series."""
    result = pd.Series("Green", index=series.index, dtype=str)

    if unfavorable_direction == "negative":
        amber_mask = series <= warn_threshold
        red_mask = series <= crit_threshold
    else:
        amber_mask = series >= warn_threshold
        red_mask = series >= crit_threshold

    result.loc[amber_mask] = "Amber"
    result.loc[red_mask] = "Red"
    result.loc[series.isna()] = "Grey"
    return result


def _urgency_from_days(
    series: pd.Series,
    critical_days: int,
    urgent_days: int,
    critical_label: str = "Critical",
    urgent_label: str = "Urgent",
    normal_label: str = "Normal",
    unknown_label: str = "Unknown",
) -> pd.Series:
    """Classify urgency from days-until event, preserving NaN as Unknown."""
    result = pd.Series(normal_label, index=series.index, dtype=str)
    result.loc[series <= urgent_days] = urgent_label
    result.loc[series <= critical_days] = critical_label
    result.loc[series.isna()] = unknown_label
    return result


def _prepare_dataset(file_key: str) -> pd.DataFrame:
    """Read and normalize a raw dataset to canonical column names."""
    df = _safe_read_csv(file_key)
    df = _normalize_column_aliases(df, file_key)
    _validate_columns(df, file_key)
    df = _ensure_optional_columns(df, file_key)
    df = _normalize_category_values(df, file_key)
    return df


# ============================================================================
# SECTION 5 — PROPERTIES LOADER
# ============================================================================

@_cache_data_compat(show_spinner=False)
def load_properties() -> pd.DataFrame:
    """Load and return the portfolio master reference table in canonical form."""
    df = _prepare_dataset("properties")

    numeric_cols = [
        "Units",
        "Purchase_Price",
        "Current_Value",
        "Stabilized_Occupancy",
        "Current_Occupancy",
        "Avg_Rent_per_Unit",
        "Year_Built",
        "Parking_Spaces",
        "Amenity_Score",
    ]
    df = _cast_numeric(df, numeric_cols)

    for col in ["Stabilized_Occupancy", "Current_Occupancy"]:
        if col in df.columns:
            df[col] = _normalize_percent_like_series(df[col])

    df["Acquisition_Date"] = pd.to_datetime(df["Acquisition_Date"], errors="coerce")

    today = pd.Timestamp.today().normalize()

    df["Acquisition_Year"] = df["Acquisition_Date"].dt.year.astype("Int64")
    holding_months = (
        (today.year - df["Acquisition_Date"].dt.year) * 12
        + (today.month - df["Acquisition_Date"].dt.month)
    )
    df["Holding_Months"] = holding_months.where(df["Acquisition_Date"].notna(), pd.NA)
    df["Holding_Months"] = df["Holding_Months"].clip(lower=0).astype("Int64")

    df["Appreciation"] = df["Current_Value"] - df["Purchase_Price"]
    df["Appreciation_Pct"] = np.where(
        df["Purchase_Price"] > 0,
        (df["Appreciation"] / df["Purchase_Price"]) * 100.0,
        np.nan,
    )
    df["Occupancy_Gap"] = df["Stabilized_Occupancy"] - df["Current_Occupancy"]
    df["Annual_GPR"] = df["Avg_Rent_per_Unit"] * df["Units"] * 12.0

    df["Region"] = df["Market"]
    df["Asset_Phase"] = df["Value_Add_Phase"]

    return df.sort_values("Property_ID").reset_index(drop=True)


# ============================================================================
# SECTION 6 — MONTHLY PERFORMANCE LOADER
# ============================================================================

@_cache_data_compat(show_spinner=False)
def load_monthly_performance() -> pd.DataFrame:
    """Load and return monthly performance data in canonical form."""
    df = _prepare_dataset("monthly_performance")

    numeric_cols = [
        "Budgeted_Revenue",
        "Actual_Revenue",
        "Budgeted_Expenses",
        "Actual_Expenses",
        "Budgeted_NOI",
        "Actual_NOI",
        "Occupancy",
        "Avg_Actual_Rent",
        "Concessions",
        "Bad_Debt",
        "Repairs_Maintenance",
        "Payroll",
        "Marketing",
        "Utilities",
        "Insurance",
        "Admin",
    ]
    df = _cast_numeric(df, numeric_cols)

    if "Occupancy" in df.columns:
        df["Occupancy"] = _normalize_percent_like_series(df["Occupancy"])

    df["Year_Month"] = pd.to_datetime(df["Year_Month"], errors="coerce")
    df["Period"] = df["Year_Month"].dt.to_period("M")

    df = df.sort_values(["Property_ID", "Year_Month"]).reset_index(drop=True)

    df["Revenue_Variance"] = df["Actual_Revenue"] - df["Budgeted_Revenue"]
    df["Revenue_Var_Pct"] = np.where(
        df["Budgeted_Revenue"] > 0,
        (df["Revenue_Variance"] / df["Budgeted_Revenue"]) * 100.0,
        np.nan,
    )

    df["Expense_Variance"] = df["Actual_Expenses"] - df["Budgeted_Expenses"]
    df["Expense_Var_Pct"] = np.where(
        df["Budgeted_Expenses"] > 0,
        (df["Expense_Variance"] / df["Budgeted_Expenses"]) * 100.0,
        np.nan,
    )

    df["NOI_Variance"] = df["Actual_NOI"] - df["Budgeted_NOI"]
    df["NOI_Var_Pct"] = np.where(
        df["Budgeted_NOI"] > 0,
        (df["NOI_Variance"] / df["Budgeted_NOI"]) * 100.0,
        np.nan,
    )

    df["Revenue_RAG"] = _rag_from_variance(
        df["Revenue_Var_Pct"],
        warn_threshold=RAG_REVENUE_WARN_PCT,
        crit_threshold=RAG_REVENUE_CRIT_PCT,
        unfavorable_direction="negative",
    )
    df["Expense_RAG"] = _rag_from_variance(
        df["Expense_Var_Pct"],
        warn_threshold=RAG_EXPENSE_WARN_PCT,
        crit_threshold=RAG_EXPENSE_CRIT_PCT,
        unfavorable_direction="positive",
    )
    df["NOI_RAG"] = _rag_from_variance(
        df["NOI_Var_Pct"],
        warn_threshold=RAG_NOI_WARN_PCT,
        crit_threshold=RAG_NOI_CRIT_PCT,
        unfavorable_direction="negative",
    )

    concessions = df["Concessions"].fillna(0.0)
    bad_debt = df["Bad_Debt"].fillna(0.0)

    df["EGI"] = (df["Actual_Revenue"] - concessions - bad_debt).clip(lower=0)

    df["Expense_Ratio"] = np.where(
        df["Actual_Revenue"] > 0,
        df["Actual_Expenses"] / df["Actual_Revenue"],
        np.nan,
    )
    df["NOI_Margin"] = np.where(
        df["Actual_Revenue"] > 0,
        df["Actual_NOI"] / df["Actual_Revenue"],
        np.nan,
    )

    df["Revenue_MoM_Chg"] = df.groupby("Property_ID")["Actual_Revenue"].pct_change() * 100.0
    df["NOI_MoM_Chg"] = df.groupby("Property_ID")["Actual_NOI"].pct_change() * 100.0

    df["Trailing_3M_NOI"] = (
        df.groupby("Property_ID")["Actual_NOI"]
        .transform(lambda s: s.rolling(window=3, min_periods=1).sum())
    )
    df["Trailing_12M_NOI"] = (
        df.groupby("Property_ID")["Actual_NOI"]
        .transform(lambda s: s.rolling(window=12, min_periods=1).sum())
    )

    df["Budget_NOI"] = df["Budgeted_NOI"]
    df["Actual_Occupancy"] = df["Occupancy"]

    return df.reset_index(drop=True)


# ============================================================================
# SECTION 7 — MARKET COMPS LOADER
# ============================================================================

@_cache_data_compat(show_spinner=False)
def load_market_comps() -> pd.DataFrame:
    """Load and return market comps in canonical form."""
    df = _prepare_dataset("market_comps")

    numeric_cols = [
        "Avg_Rent",
        "Comp_Occupancy",
        "Rent_Growth_YoY",
        "Distance_Miles",
        "Units",
        "Amenity_Score",
        "Year_Built",
    ]
    df = _cast_numeric(df, numeric_cols)

    for col in ["Comp_Occupancy", "Rent_Growth_YoY"]:
        if col in df.columns:
            df[col] = _normalize_percent_like_series(df[col])

    df["Comp_Annual_Rent"] = df["Avg_Rent"] * 12.0
    df["Comp_Rank"] = (
        df.groupby("Property_ID")["Avg_Rent"]
        .rank(ascending=False, method="min")
        .astype("Int64")
    )

    df["Market_Rent"] = df["Avg_Rent"]
    df["Comp_Monthly_Rent"] = df["Avg_Rent"]
    df["Occupancy"] = df["Comp_Occupancy"]

    return df.sort_values(["Property_ID", "Avg_Rent"], ascending=[True, False]).reset_index(drop=True)


# ============================================================================
# SECTION 8 — BUSINESS PLAN LOADER
# ============================================================================

@_cache_data_compat(show_spinner=False)
def load_business_plan() -> pd.DataFrame:
    """Load and return business plan tracker data."""
    df = _prepare_dataset("business_plan")

    numeric_cols = [
        "Percent_Complete",
        "Budget",
        "Actual_Spend",
        "Expected_NOI_Lift",
    ]
    df = _cast_numeric(df, numeric_cols)

    if "Percent_Complete" in df.columns:
        df["Percent_Complete"] = _normalize_percent_like_series(df["Percent_Complete"])

    df["Target_Completion"] = pd.to_datetime(df["Target_Completion"], errors="coerce")
    today = pd.Timestamp.today().normalize()

    df["Budget_Variance"] = df["Actual_Spend"] - df["Budget"]
    df["Budget_Var_Pct"] = np.where(
        df["Budget"] > 0,
        df["Budget_Variance"] / df["Budget"],
        np.nan,
    )
    df["Budget_Utilized_Pct"] = np.where(
        df["Budget"] > 0,
        df["Actual_Spend"] / df["Budget"],
        np.nan,
    )

    df["Days_to_Completion"] = (df["Target_Completion"] - today).dt.days
    df["Is_Overdue"] = (
        (df["Days_to_Completion"] < 0)
        & (df["Status"].fillna("").astype(str).str.strip() != "Complete")
    )

    proxy_cap_rate = 0.0525
    df["NOI_Lift_Per_Dollar"] = np.where(
        df["Budget"] > 0,
        df["Expected_NOI_Lift"] / df["Budget"],
        np.nan,
    )
    df["Implied_Value_Add"] = np.where(
        proxy_cap_rate > 0,
        df["Expected_NOI_Lift"] / proxy_cap_rate,
        np.nan,
    )

    # Use a more standard incremental return on cost definition.
    df["Return_on_Cost"] = np.where(
        df["Budget"] > 0,
        df["Expected_NOI_Lift"] / df["Budget"],
        np.nan,
    )

    status_colors = {
        "Complete": "#1B7F4F",
        "In_Progress": "#1B4F72",
        "Not_Started": "#7F8C8D",
    }
    risk_colors = {
        "Low": "#1B7F4F",
        "Medium": "#D4AC0D",
        "High": "#C0392B",
    }

    df["Status_Color"] = df["Status"].map(status_colors).fillna("#7F8C8D")
    df["Risk_Color"] = df["Risk_Level"].map(risk_colors).fillna("#7F8C8D")

    def _completion_rag(row: pd.Series) -> str:
        if row["Status"] == "Complete":
            return "Green"
        if bool(row["Is_Overdue"]):
            return "Red"

        budget_overrun = row.get("Budget_Var_Pct", np.nan)
        if pd.notna(budget_overrun) and budget_overrun > 0.15:
            return "Red"

        risk = str(row.get("Risk_Level", "Low")).strip()
        days = row["Days_to_Completion"] if pd.notna(row["Days_to_Completion"]) else np.nan

        if risk == "High":
            return "Amber"
        if pd.notna(days) and days <= 60:
            return "Amber"
        if pd.isna(days):
            return "Grey"
        return "Green"

    df["Completion_RAG"] = df.apply(_completion_rag, axis=1)

    return df.sort_values(["Property_ID", "Initiative_ID"]).reset_index(drop=True)


# ============================================================================
# SECTION 9 — DEBT COVENANTS LOADER
# ============================================================================

@_cache_data_compat(show_spinner=False)
def load_debt_covenants() -> pd.DataFrame:
    """Load and return debt covenant data in canonical form."""
    df = _prepare_dataset("debt_covenants")

    numeric_cols = [
        "Loan_Balance",
        "Interest_Rate",
        "DSCR_Requirement",
        "DSCR_Actual",
        "LTV",
        "Max_LTV_Allowed",
        "Annual_Debt_Service",
    ]
    df = _cast_numeric(df, numeric_cols)

    for col in ["LTV", "Max_LTV_Allowed"]:
        if col in df.columns:
            df[col] = _normalize_percent_like_series(df[col])

    if "Interest_Rate" in df.columns:
        df["Interest_Rate"] = _normalize_percent_like_series(df["Interest_Rate"])

    df["Next_Reporting_Date"] = pd.to_datetime(df["Next_Reporting_Date"], errors="coerce")
    df["Maturity_Date"] = pd.to_datetime(df["Maturity_Date"], errors="coerce")

    today = pd.Timestamp.today().normalize()

    df["DSCR_Headroom"] = df["DSCR_Actual"] - df["DSCR_Requirement"]
    df["DSCR_Status"] = np.where(
        df["DSCR_Headroom"] < 0,
        "Breach",
        np.where(df["DSCR_Headroom"] < DSCR_WATCH_BUFFER, "Watch", "Compliant"),
    )
    df["DSCR_RAG"] = np.where(
        df["DSCR_Status"] == "Breach",
        "Red",
        np.where(df["DSCR_Status"] == "Watch", "Amber", "Green"),
    )

    df["LTV_Headroom"] = df["Max_LTV_Allowed"] - df["LTV"]
    df["LTV_Status"] = np.where(
        df["LTV_Headroom"] < 0,
        "Breach",
        np.where(df["LTV_Headroom"] < LTV_WATCH_BUFFER, "Watch", "Compliant"),
    )
    df["LTV_RAG"] = np.where(
        df["LTV_Status"] == "Breach",
        "Red",
        np.where(df["LTV_Status"] == "Watch", "Amber", "Green"),
    )

    df["Days_to_Reporting"] = (df["Next_Reporting_Date"] - today).dt.days
    df["Days_to_Maturity"] = (df["Maturity_Date"] - today).dt.days

    df["Reporting_Urgency"] = _urgency_from_days(
        df["Days_to_Reporting"],
        critical_days=REPORTING_CRITICAL_DAYS,
        urgent_days=REPORTING_URGENT_DAYS,
        critical_label="Critical",
        urgent_label="Urgent",
        normal_label="Normal",
        unknown_label="Unknown",
    )
    df["Maturity_Urgency"] = _urgency_from_days(
        df["Days_to_Maturity"],
        critical_days=180,
        urgent_days=365,
        critical_label="Critical",
        urgent_label="Watch",
        normal_label="Normal",
        unknown_label="Unknown",
    )

    rag_rank = {"Red": 2, "Amber": 1, "Green": 0}
    inv_rank = {2: "Red", 1: "Amber", 0: "Green"}

    df["Overall_RAG"] = (
        pd.concat(
            [
                df["DSCR_RAG"].map(rag_rank).fillna(0),
                df["LTV_RAG"].map(rag_rank).fillna(0),
            ],
            axis=1,
        )
        .max(axis=1)
        .map(inv_rank)
    )

    df["Annual_Interest"] = df["Loan_Balance"] * df["Interest_Rate"]
    df["Implied_NOI_Required"] = df["DSCR_Requirement"] * df["Annual_Debt_Service"]

    df["LTV_Actual"] = df["LTV"]
    df["LTV_Max"] = df["Max_LTV_Allowed"]

    return df.sort_values("Property_ID").reset_index(drop=True)


# ============================================================================
# SECTION 10 — UNDERWRITING LOADER
# ============================================================================

@_cache_data_compat(show_spinner=False)
def load_underwriting() -> pd.DataFrame:
    """Load and return underwriting assumptions in canonical form."""
    df = _prepare_dataset("underwriting")

    numeric_cols = [
        "Underwritten_Occupancy",
        "Underwritten_Rent_Growth_Pct",
        "Underwritten_Expense_Growth_Pct",
        "Underwritten_Exit_Cap_Rate",
        "Underwritten_NOI_Year1",
        "Underwritten_IRR",
        "Target_Hold_Years",
        "Underwritten_Purchase_Cap_Rate",
        "Underwritten_EGI_Year1",
        "Underwritten_OpEx_Year1",
        "Underwritten_Vacancy_Pct",
        "Underwritten_CapEx_Reserve_Per_Unit",
    ]
    df = _cast_numeric(df, numeric_cols)

    percent_like_cols = [
        "Underwritten_Occupancy",
        "Underwritten_Rent_Growth_Pct",
        "Underwritten_Expense_Growth_Pct",
        "Underwritten_Exit_Cap_Rate",
        "Underwritten_IRR",
        "Underwritten_Purchase_Cap_Rate",
        "Underwritten_Vacancy_Pct",
    ]
    for col in percent_like_cols:
        if col in df.columns:
            df[col] = _normalize_percent_like_series(df[col])

    df["Exit_Cap_Rate"] = df["Underwritten_Exit_Cap_Rate"]
    df["Purchase_Cap_Rate"] = df["Underwritten_Purchase_Cap_Rate"]
    df["Year_1_NOI"] = df["Underwritten_NOI_Year1"]
    df["Target_IRR"] = df["Underwritten_IRR"]

    return df.sort_values("Property_ID").reset_index(drop=True)


# ============================================================================
# SECTION 11 — COMPOSITE AND MERGED LOADERS
# ============================================================================

@_cache_data_compat(show_spinner=False)
def load_all() -> dict[str, pd.DataFrame]:
    """Load all datasets."""
    return {
        "properties": load_properties(),
        "monthly_performance": load_monthly_performance(),
        "market_comps": load_market_comps(),
        "business_plan": load_business_plan(),
        "debt_covenants": load_debt_covenants(),
        "underwriting": load_underwriting(),
    }


@_cache_data_compat(show_spinner=False)
def load_performance_with_properties() -> pd.DataFrame:
    """Return performance joined to property metadata."""
    perf = load_monthly_performance()
    props = load_properties()[
        [
            "Property_ID",
            "Property_Name",
            "Market",
            "Region",
            "Units",
            "Value_Add_Phase",
            "Asset_Phase",
            "Purchase_Price",
            "Current_Value",
            "Stabilized_Occupancy",
            "Annual_GPR",
        ]
    ]
    merged = perf.merge(props, on="Property_ID", how="left")
    return merged.sort_values(["Property_ID", "Year_Month"]).reset_index(drop=True)


@_cache_data_compat(show_spinner=False)
def load_covenants_with_properties() -> pd.DataFrame:
    """Return debt covenants joined to property metadata."""
    covenants = load_debt_covenants()
    props = load_properties()[["Property_ID", "Property_Name", "Market", "Region", "Units"]]
    merged = covenants.merge(props, on="Property_ID", how="left")
    return merged.sort_values("Property_ID").reset_index(drop=True)


@_cache_data_compat(show_spinner=False)
def load_business_plan_with_properties() -> pd.DataFrame:
    """Return business plan joined to property metadata."""
    bp = load_business_plan()
    props = load_properties()[
        ["Property_ID", "Property_Name", "Market", "Region", "Units", "Value_Add_Phase", "Asset_Phase"]
    ]
    merged = bp.merge(props, on="Property_ID", how="left")
    return merged.sort_values(["Property_ID", "Initiative_ID"]).reset_index(drop=True)


@_cache_data_compat(show_spinner=False)
def load_market_comps_with_subject() -> pd.DataFrame:
    """Return market comps enriched with subject property metadata."""
    comps = load_market_comps()
    props = load_properties()[
        [
            "Property_ID",
            "Property_Name",
            "Avg_Rent_per_Unit",
            "Current_Occupancy",
            "Stabilized_Occupancy",
        ]
    ].rename(
        columns={
            "Avg_Rent_per_Unit": "Subject_Rent",
            "Current_Occupancy": "Subject_Occupancy",
            "Stabilized_Occupancy": "Subject_Stabilized_Occ",
        }
    )

    merged = comps.merge(props, on="Property_ID", how="left")

    merged["Rent_Premium_vs_Comp"] = merged["Subject_Rent"] - merged["Avg_Rent"]
    merged["Rent_Premium_Pct"] = np.where(
        merged["Avg_Rent"] > 0,
        (merged["Rent_Premium_vs_Comp"] / merged["Avg_Rent"]) * 100.0,
        np.nan,
    )
    merged["Occupancy_Gap_vs_Comp"] = merged["Subject_Occupancy"] - merged["Comp_Occupancy"]

    return merged.sort_values(["Property_ID", "Avg_Rent"], ascending=[True, False]).reset_index(drop=True)


# ============================================================================
# SECTION 12 — PORTFOLIO-LEVEL AGGREGATE HELPERS
# ============================================================================

@_cache_data_compat(show_spinner=False)
def get_portfolio_kpis() -> dict[str, float]:
    """Compute top-level portfolio KPIs."""
    props = load_properties()
    perf = load_monthly_performance()
    covenants = load_debt_covenants()
    bp = load_business_plan()

    total_units = float(props["Units"].sum())

    perf_sorted = perf.sort_values(["Property_ID", "Year_Month"]).copy()
    perf_sorted = perf_sorted[perf_sorted["Year_Month"].notna()]

    t12 = (
        perf_sorted.groupby("Property_ID", group_keys=False)
        .apply(lambda g: g.drop_duplicates(subset=["Period"], keep="last").tail(12))
        .reset_index(drop=True)
    )

    total_t12_rev = float(t12["Actual_Revenue"].sum())
    total_t12_exp = float(t12["Actual_Expenses"].sum())
    total_t12_noi = float(t12["Actual_NOI"].sum())
    total_t12_budg = float(t12["Budgeted_NOI"].sum())

    noi_vs_budget_pct = (
        ((total_t12_noi - total_t12_budg) / total_t12_budg) * 100.0
        if total_t12_budg > 0 else np.nan
    )
    noi_margin = total_t12_noi / total_t12_rev if total_t12_rev > 0 else np.nan
    expense_ratio = total_t12_exp / total_t12_rev if total_t12_rev > 0 else np.nan

    latest_perf = perf_sorted.groupby("Property_ID").last().reset_index()
    latest_with_units = latest_perf.merge(
        props[["Property_ID", "Units"]],
        on="Property_ID",
        how="left",
    )
    units_sum = latest_with_units["Units"].sum()
    weighted_occ = (
        (latest_with_units["Occupancy"] * latest_with_units["Units"]).sum() / units_sum
        if units_sum > 0 else np.nan
    )

    covenant_by_property = (
        covenants.groupby("Property_ID")["Overall_RAG"]
        .agg(lambda s: "Red" if (s == "Red").any() else ("Amber" if (s == "Amber").any() else "Green"))
        .reset_index()
    )
    assets_in_breach = int((covenant_by_property["Overall_RAG"] == "Red").sum())
    assets_on_watch = int((covenant_by_property["Overall_RAG"] == "Amber").sum())

    total_loan_balance = float(covenants["Loan_Balance"].sum())
    total_annual_debt_service = float(covenants["Annual_Debt_Service"].sum())
    portfolio_dscr = total_t12_noi / total_annual_debt_service if total_annual_debt_service > 0 else np.nan

    active_bp = bp[bp["Status"] != "Complete"]
    total_expected_noi_lift = float(active_bp["Expected_NOI_Lift"].sum())

    return {
        "total_units": total_units,
        "total_noi": total_t12_noi,
        "weighted_avg_occupancy": weighted_occ,
        "portfolio_dscr": portfolio_dscr,
        "assets_in_breach": assets_in_breach,
        "assets_on_watch": assets_on_watch,
        "total_trailing_12m_noi": total_t12_noi,
        "total_trailing_12m_rev": total_t12_rev,
        "total_trailing_12m_exp": total_t12_exp,
        "portfolio_noi_margin": noi_margin,
        "portfolio_expense_ratio": expense_ratio,
        "total_budgeted_noi_t12": total_t12_budg,
        "noi_vs_budget_pct": noi_vs_budget_pct,
        "total_loan_balance": total_loan_balance,
        "total_annual_debt_service": total_annual_debt_service,
        "total_expected_noi_lift": total_expected_noi_lift,
    }


# ============================================================================
# SECTION 13 — CONVENIENCE HELPERS
# ============================================================================

@_cache_data_compat(show_spinner=False)
def get_property_list() -> list[str]:
    """Return sorted property IDs."""
    return sorted(load_properties()["Property_ID"].dropna().unique().tolist())


@_cache_data_compat(show_spinner=False)
def get_property_name_map() -> dict[str, str]:
    """Return Property_ID -> Property_Name mapping."""
    props = load_properties()[["Property_ID", "Property_Name"]]
    return dict(zip(props["Property_ID"], props["Property_Name"]))


@_cache_data_compat(show_spinner=False)
def get_property_display_options() -> list[str]:
    """Return 'Property_ID — Property_Name' display labels."""
    name_map = get_property_name_map()
    return [f"{pid} — {name}" for pid, name in sorted(name_map.items())]


def parse_property_selection(selection: str) -> str:
    """Extract Property_ID from display label."""
    if not isinstance(selection, str):
        raise TypeError("selection must be a string")
    if " — " in selection:
        return selection.split(" — ", 1)[0].strip()
    return selection.strip()


def get_performance_for_property(
    property_id: str,
    trailing_months: int | None = None,
) -> pd.DataFrame:
    """Return performance rows for one property."""
    perf = load_monthly_performance()
    filtered = perf[perf["Property_ID"] == property_id].copy()
    filtered = filtered.sort_values("Year_Month").reset_index(drop=True)

    if trailing_months is not None:
        if trailing_months <= 0:
            return filtered.iloc[0:0].copy()
        if len(filtered) > trailing_months:
            filtered = filtered.tail(trailing_months).reset_index(drop=True)

    return filtered


def get_comps_for_property(property_id: str) -> pd.DataFrame:
    """Return comp rows for one property."""
    comps = load_market_comps_with_subject()
    return comps[comps["Property_ID"] == property_id].sort_values("Avg_Rent", ascending=False).reset_index(drop=True)


def get_business_plan_for_property(property_id: str) -> pd.DataFrame:
    """Return business plan rows for one property."""
    bp = load_business_plan_with_properties()
    return bp[bp["Property_ID"] == property_id].sort_values("Initiative_ID").reset_index(drop=True)


def get_covenant_for_property(property_id: str) -> pd.DataFrame:
    """Return covenant rows for one property."""
    covenants = load_covenants_with_properties()
    return covenants[covenants["Property_ID"] == property_id].reset_index(drop=True)


def get_underwriting_for_property(property_id: str) -> pd.DataFrame:
    """Return underwriting rows for one property."""
    uw = load_underwriting()
    return uw[uw["Property_ID"] == property_id].reset_index(drop=True)


@_cache_data_compat(show_spinner=False)
def get_schema_diagnostics() -> dict[str, Any]:
    """Return a lightweight diagnostics snapshot for loaded datasets."""
    diagnostics: dict[str, Any] = {}

    for dataset_name in FILE_PATHS:
        try:
            if dataset_name == "properties":
                df = load_properties()
            elif dataset_name == "monthly_performance":
                df = load_monthly_performance()
            elif dataset_name == "market_comps":
                df = load_market_comps()
            elif dataset_name == "business_plan":
                df = load_business_plan()
            elif dataset_name == "debt_covenants":
                df = load_debt_covenants()
            else:
                df = load_underwriting()

            diagnostics[dataset_name] = {
                "rows": int(len(df)),
                "columns": list(df.columns),
                "missing_required_columns": [
                    col for col in CANONICAL_REQUIRED_COLUMNS[dataset_name] if col not in df.columns
                ],
            }
        except Exception as exc:
            diagnostics[dataset_name] = {
                "error": str(exc),
            }

    return diagnostics


# ============================================================================
# SECTION 14 — MODULE SELF-TEST
# ============================================================================

if __name__ == "__main__":
    import sys

    print("\n" + "═" * 68)
    print("  AssetOptima Pro — Data Loader Self-Test")
    print("═" * 68)

    all_passed = True

    loaders = {
        "properties": load_properties,
        "monthly_performance": load_monthly_performance,
        "market_comps": load_market_comps,
        "business_plan": load_business_plan,
        "debt_covenants": load_debt_covenants,
        "underwriting": load_underwriting,
        "performance_with_properties": load_performance_with_properties,
        "covenants_with_properties": load_covenants_with_properties,
        "business_plan_with_properties": load_business_plan_with_properties,
        "market_comps_with_subject": load_market_comps_with_subject,
    }

    for name, loader_fn in loaders.items():
        try:
            df = loader_fn()
            print(f"  ✓  {name:<38} {len(df):>4} rows  {len(df.columns):>3} cols")
        except Exception as exc:
            print(f"  ✗  {name:<38} FAILED — {exc}")
            all_passed = False

    print()

    try:
        props = load_properties()
        perf = load_monthly_performance()
        comps = load_market_comps()
        cov = load_debt_covenants()
        uw = load_underwriting()

        assert "Market" in props.columns
        assert "Value_Add_Phase" in props.columns
        assert "Budgeted_NOI" in perf.columns
        assert "Comp_Occupancy" in comps.columns
        assert "LTV" in cov.columns
        assert "Max_LTV_Allowed" in cov.columns
        assert "Underwritten_Exit_Cap_Rate" in uw.columns
        assert "Underwritten_IRR" in uw.columns

        print("  ✓  Canonical schema checks passed")
    except Exception as exc:
        print(f"  ✗  Canonical schema checks FAILED — {exc}")
        all_passed = False

    print()

    try:
        kpis = get_portfolio_kpis()
        print("  Portfolio KPI snapshot:")
        print(f"    Total Units              : {kpis['total_units']:,.0f}")
        print(f"    T12 Portfolio NOI        : ${kpis['total_noi']:>15,.0f}")
        print(f"    Weighted Avg Occupancy   : {kpis['weighted_avg_occupancy']:.1%}")
        print(f"    Portfolio DSCR           : {kpis['portfolio_dscr']:.2f}x")
        print(f"    Assets in Breach         : {kpis['assets_in_breach']}")
        print(f"    Assets on Watch          : {kpis['assets_on_watch']}")
        print(f"    NOI Lift Pipeline        : ${kpis['total_expected_noi_lift']:>15,.0f}")
    except Exception as exc:
        print(f"  ✗  Portfolio KPI calculation FAILED — {exc}")
        all_passed = False

    print()

    try:
        diagnostics = get_schema_diagnostics()
        print("  Diagnostics summary:")
        for dataset_name, info in diagnostics.items():
            if "error" in info:
                print(f"    {dataset_name:<20} ERROR — {info['error']}")
            else:
                print(
                    f"    {dataset_name:<20} rows={info['rows']:<4} "
                    f"missing_required={len(info['missing_required_columns'])}"
                )
    except Exception as exc:
        print(f"  ✗  Diagnostics generation FAILED — {exc}")
        all_passed = False

    print()
    print("═" * 68)

    if all_passed:
        print("  ALL TESTS PASSED — canonical data foundation is ready.")
        print("═" * 68 + "\n")
        sys.exit(0)
    else:
        print("  ONE OR MORE TESTS FAILED — fix errors above before")
        print("  proceeding to downstream module refactors.")
        print("═" * 68 + "\n")
        sys.exit(1)