"""
pages/6_Compliance_and_Reporting.py

Compliance and Reporting Center page for AssetOptima Pro.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Optional

import pandas as pd
import streamlit as st

from modules.debt_compliance import (
    get_compliance_summary_table,
    get_covenant_breach_table,
    get_lender_exposure_summary,
    get_portfolio_compliance_kpis,
    get_property_compliance_summary,
    get_refinance_watchlist,
    get_upcoming_deadlines,
)
from modules.data_loader import get_property_display_options, parse_property_selection
from utils.coercion import ensure_dataframe, ensure_dict, parse_float, parse_int, safe_str
from utils.formatters import fmt_currency, fmt_days, fmt_multiple, fmt_pct
from utils.ui_helpers import render_compliance_card, render_warnings

# ---------------------------------------------------------------------------
# Logger — host application owns root logger configuration.
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Page config — must be the first Streamlit call in the module.
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Compliance & Reporting | AssetOptima Pro",
    page_icon="📋",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_TTL_SECONDS: int = 300  # 5-minute cache TTL across all data loaders
_DEADLINE_DAYS_AHEAD: int = 90  # Upcoming-deadline horizon
_DEFAULT_CARD_COUNT: int = 5   # Max properties shown by default in multiselect

# RAG risk ranking used for sorting — lower rank = higher risk
_RAG_RANK: dict[str, int] = {"Red": 0, "Amber": 1, "Green": 2}

# Columns that carry DSCR, LTV, and currency values in each table
_DSCR_COLS = ["DSCR_Actual", "DSCR_Requirement", "DSCR_Headroom"]
_LTV_PCT_COLS_SUMMARY = ["LTV_Actual", "LTV_Max"]
_LTV_PCT_COLS_BREACH = ["LTV_Actual_Dec", "LTV_Max_Dec"]
_CURRENCY_COLS_WATCHLIST = ["Loan_Balance"]
_CURRENCY_COLS_LENDER = ["Total_Loan_Balance"]


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _fmt_pts(value: Any) -> str:
    """
    Format a basis-point / percentage-point headroom value.

    Guards against ``None`` and ``NaN`` — both produce ``"N/A"`` instead
    of crashing with a ``TypeError`` inside ``.map()``.
    """
    parsed = parse_float(value)
    if parsed is None:
        return "N/A"
    return f"{parsed:.1f} pts"


def _safe_int_or_none(value: Any) -> Optional[int]:
    """
    Return ``parse_int(value)`` or ``None`` when the value is missing/NaN.

    Replaces the fragile ``pd.notna`` scalar pattern used inside ``.map()``.
    """
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return parse_int(value)


def _fmt_multiple_safe(value: Any) -> str:
    """Wrap ``fmt_multiple`` with a ``None``/``NaN`` guard."""
    parsed = parse_float(value)
    if parsed is None:
        return "N/A"
    return fmt_multiple(parsed)


def _fmt_pct_safe(value: Any, *, include_sign: bool = False) -> str:
    """Wrap ``fmt_pct`` with a ``None``/``NaN`` guard."""
    parsed = parse_float(value)
    if parsed is None:
        return "N/A"
    return fmt_pct(parsed, include_sign=include_sign)


def _fmt_currency_safe(value: Any) -> str:
    """Wrap ``fmt_currency`` with a ``None``/``NaN`` guard."""
    parsed = parse_float(value)
    if parsed is None:
        return "N/A"
    return fmt_currency(parsed)


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner=False, ttl=_TTL_SECONDS)
def get_property_options() -> list[str]:
    """Load property selector options (cached)."""
    try:
        options = get_property_display_options()
        return options if isinstance(options, list) else []
    except Exception:
        logger.exception("Failed to load property options")
        return []


@st.cache_data(show_spinner=False, ttl=_TTL_SECONDS)
def load_portfolio_bundle() -> dict[str, Any]:
    """
    Load portfolio-wide compliance outputs with partial-failure tolerance.

    Each loader is isolated so a single broken data source does not blank
    the entire page.  All values are coerced to clean types here — callers
    receive ready-to-use objects with no second coercion pass.
    """
    bundle: dict[str, Any] = {
        "kpis": {},
        "summary_table": pd.DataFrame(),
        "breach_table": pd.DataFrame(),
        "deadlines": pd.DataFrame(),
        "watchlist": pd.DataFrame(),
        "lender_exposure": pd.DataFrame(),
        "warnings": [],
    }

    loaders: dict[str, Any] = {
        "kpis": get_portfolio_compliance_kpis,
        "summary_table": get_compliance_summary_table,
        "breach_table": get_covenant_breach_table,
        "deadlines": lambda: get_upcoming_deadlines(days_ahead=_DEADLINE_DAYS_AHEAD),
        "watchlist": get_refinance_watchlist,
        "lender_exposure": get_lender_exposure_summary,
    }

    for key, loader in loaders.items():
        try:
            bundle[key] = loader()
        except Exception:
            logger.exception(
                "Failed loading compliance bundle component: %s", key
            )
            bundle["warnings"].append(
                f"'{key}' data failed to load — partial results shown."
            )

    # Coerce once; callers receive clean types.
    bundle["kpis"] = ensure_dict(bundle["kpis"])
    bundle["summary_table"] = ensure_dataframe(bundle["summary_table"])
    bundle["breach_table"] = ensure_dataframe(bundle["breach_table"])
    bundle["deadlines"] = ensure_dataframe(bundle["deadlines"])
    bundle["watchlist"] = ensure_dataframe(bundle["watchlist"])
    bundle["lender_exposure"] = ensure_dataframe(bundle["lender_exposure"])
    return bundle


@st.cache_data(show_spinner=False, ttl=_TTL_SECONDS)
def load_property_summary(property_id: str) -> tuple[dict[str, Any], bool]:
    """
    Load one property's compliance summary safely.

    Returns a ``(data, success)`` tuple so callers can surface failures
    to the user without silent data gaps.
    """
    try:
        return ensure_dict(get_property_compliance_summary(property_id)), True
    except Exception:
        logger.exception(
            "Failed to load property compliance summary (property_id=%s)",
            property_id,
        )
        return {}, False


# ---------------------------------------------------------------------------
# Display formatters
# ---------------------------------------------------------------------------


def build_deadlines_display(df: pd.DataFrame) -> pd.DataFrame:
    """Return a display-formatted copy of the upcoming-deadlines table."""
    df = ensure_dataframe(df)
    if df.empty:
        return df

    display_df = df.copy()
    for col in ["Days_to_Reporting", "Days_to_Maturity", "Days_Remaining"]:
        if col in display_df.columns:
            display_df[col] = display_df[col].map(_safe_int_or_none)
    return display_df


def build_compliance_summary_display(df: pd.DataFrame) -> pd.DataFrame:
    """Return a display-formatted copy of the portfolio compliance summary."""
    df = ensure_dataframe(df)
    if df.empty:
        return df

    display_df = df.copy()

    for col in _DSCR_COLS:
        if col in display_df.columns:
            display_df[col] = display_df[col].map(_fmt_multiple_safe)

    for col in _LTV_PCT_COLS_SUMMARY:
        if col in display_df.columns:
            display_df[col] = display_df[col].map(_fmt_pct_safe)

    if "LTV_Headroom_Pct_Pts" in display_df.columns:
        display_df["LTV_Headroom_Pct_Pts"] = display_df[
            "LTV_Headroom_Pct_Pts"
        ].map(_fmt_pts)

    return display_df


def build_breach_display(df: pd.DataFrame) -> pd.DataFrame:
    """Return a display-formatted copy of the covenant breach table."""
    df = ensure_dataframe(df)
    if df.empty:
        return df

    display_df = df.copy()

    for col in _DSCR_COLS:
        if col in display_df.columns:
            display_df[col] = display_df[col].map(_fmt_multiple_safe)

    for col in _LTV_PCT_COLS_BREACH:
        if col in display_df.columns:
            display_df[col] = display_df[col].map(_fmt_pct_safe)

    if "LTV_Headroom_Pct_Pts" in display_df.columns:
        display_df["LTV_Headroom_Pct_Pts"] = display_df[
            "LTV_Headroom_Pct_Pts"
        ].map(_fmt_pts)

    return display_df


def build_watchlist_display(df: pd.DataFrame) -> pd.DataFrame:
    """Return a display-formatted copy of the refinance watchlist."""
    df = ensure_dataframe(df)
    if df.empty:
        return df

    display_df = df.copy()

    for col in _CURRENCY_COLS_WATCHLIST:
        if col in display_df.columns:
            display_df[col] = display_df[col].map(_fmt_currency_safe)

    if "DSCR_Actual" in display_df.columns:
        display_df["DSCR_Actual"] = display_df["DSCR_Actual"].map(
            _fmt_multiple_safe
        )

    if "LTV_Actual" in display_df.columns:
        display_df["LTV_Actual"] = display_df["LTV_Actual"].map(_fmt_pct_safe)

    return display_df


def build_lender_exposure_display(df: pd.DataFrame) -> pd.DataFrame:
    """Return a display-formatted copy of the lender exposure summary."""
    df = ensure_dataframe(df)
    if df.empty:
        return df

    display_df = df.copy()

    for col in _CURRENCY_COLS_LENDER:
        if col in display_df.columns:
            display_df[col] = display_df[col].map(_fmt_currency_safe)

    if "Average_DSCR" in display_df.columns:
        display_df["Average_DSCR"] = display_df["Average_DSCR"].map(
            _fmt_multiple_safe
        )

    if "Average_LTV" in display_df.columns:
        display_df["Average_LTV"] = display_df["Average_LTV"].map(_fmt_pct_safe)

    return display_df


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def build_placeholder_export_bytes(label: str) -> bytes:
    """
    Build uniquely-stamped placeholder bytes for each export download button.

    Each call embeds the current UTC timestamp so the three export buttons
    produce distinct file contents rather than identical blobs.
    """
    timestamp = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    text = (
        f"Export: {label}\n"
        f"Application: AssetOptima Pro\n"
        f"Generated: {timestamp}\n\n"
        f"This placeholder export confirms the Compliance & Reporting page is\n"
        f"wired correctly. Replace this output with real report generation once\n"
        f"modules/report_generator.py is implemented.\n"
    )
    return text.encode("utf-8")


def get_highest_risk_asset(
    summary_table: pd.DataFrame,
) -> Optional[pd.Series]:
    """
    Return the highest-risk asset row from the compliance summary table.

    Sorts first by RAG status (Red → Amber → Green), then by
    ``Days_to_Maturity`` ascending (soonest maturity = most urgent).
    Numeric coercion is applied before sorting to handle mixed-type columns.
    """
    summary_table = ensure_dataframe(summary_table)
    if summary_table.empty or "Overall_RAG" not in summary_table.columns:
        return None

    risk_sorted = summary_table.copy()
    risk_sorted["_Risk_Rank"] = (
        risk_sorted["Overall_RAG"].map(_RAG_RANK).fillna(9)
    )

    sort_cols = ["_Risk_Rank"]
    ascending = [True]

    if "Days_to_Maturity" in risk_sorted.columns:
        risk_sorted["Days_to_Maturity"] = pd.to_numeric(
            risk_sorted["Days_to_Maturity"], errors="coerce"
        )
        sort_cols.append("Days_to_Maturity")
        ascending.append(True)

    risk_sorted = risk_sorted.sort_values(sort_cols, ascending=ascending)
    risk_sorted.drop(columns=["_Risk_Rank"], inplace=True)

    return risk_sorted.iloc[0] if not risk_sorted.empty else None


def _series_get(row: pd.Series, col: str, default: str = "") -> str:
    """
    Safely retrieve a value from a ``pd.Series`` by column name.

    Avoids ``Series.get()`` which can silently return ``None`` for
    index mismatches; uses explicit ``in row.index`` guard instead.
    """
    return safe_str(row[col] if col in row.index else default, default)


# ---------------------------------------------------------------------------
# Section renderers
# ---------------------------------------------------------------------------


def _render_portfolio_kpis(kpis: dict[str, Any]) -> None:
    """Render the six primary KPI metrics and two deadline metrics."""
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    with k1:
        st.metric("Loan Count", f"{parse_int(kpis.get('Loan_Count', 0)):,}")
    with k2:
        st.metric("Breaches", f"{parse_int(kpis.get('Breach_Count', 0)):,}")
    with k3:
        st.metric(
            "Watchlist Loans",
            f"{parse_int(kpis.get('Watchlist_Count', 0)):,}",
        )
    with k4:
        st.metric(
            "Compliant Loans",
            f"{parse_int(kpis.get('Compliant_Count', 0)):,}",
        )
    with k5:
        st.metric("Avg DSCR", _fmt_multiple_safe(kpis.get("Average_DSCR")))
    with k6:
        st.metric("Avg LTV", _fmt_pct_safe(kpis.get("Average_LTV")))

    s1, s2 = st.columns(2)
    with s1:
        st.metric(
            "Nearest Reporting Deadline",
            fmt_days(kpis.get("Nearest_Reporting_Days")),
        )
    with s2:
        st.metric(
            "Nearest Loan Maturity",
            fmt_days(kpis.get("Nearest_Maturity_Days")),
        )


def _render_covenant_cards(property_options: list[str]) -> None:
    """Render per-property covenant compliance cards."""
    st.subheader("Property-Level Covenant Monitoring")

    if not property_options:
        st.info("No property options are available for covenant monitoring.")
        return

    default_selection = property_options[:_DEFAULT_CARD_COUNT]
    selected_properties = st.multiselect(
        "Select properties to display covenant cards",
        options=property_options,
        default=default_selection,
        help=f"Defaults to the first {_DEFAULT_CARD_COUNT} properties. "
             "Add or remove to customise the view.",
    )

    if not selected_properties:
        st.info("Select at least one property to display covenant monitoring cards.")
        return

    columns = st.columns(2)
    for idx, selected in enumerate(selected_properties):
        property_id = parse_property_selection(selected)
        if not property_id:
            st.warning(f"Could not parse property ID for '{selected}' — skipping.")
            continue

        summary, success = load_property_summary(property_id)
        if not success:
            st.warning(
                f"Compliance data for **{selected}** could not be loaded. "
                "It may be excluded from covenant cards until data is available."
            )

        with columns[idx % 2]:
            render_compliance_card(summary)


def _render_deadlines_and_exports(deadlines: pd.DataFrame) -> None:
    """Render the upcoming-deadlines table and export download buttons."""
    left_col, right_col = st.columns((1.2, 1.0))

    with left_col:
        st.subheader("Upcoming Deadlines")
        if deadlines.empty:
            st.success(
                f"No reporting or maturity deadlines within the next "
                f"{_DEADLINE_DAYS_AHEAD} days."
            )
        else:
            st.dataframe(
                build_deadlines_display(deadlines),
                use_container_width=True,
                hide_index=True,
            )

    with right_col:
        st.subheader("Reporting Exports")
        st.info(
            "Export buttons are connected to placeholder outputs. "
            "They will generate real reports once `report_generator.py` "
            "is implemented.",
            icon="ℹ️",
        )

        _export_buttons = [
            (
                "⬇️ Monthly Investor Package (Excel)",
                "Monthly Investor Package Excel",
                "monthly_investor_package_placeholder.txt",
            ),
            (
                "⬇️ Covenant Compliance Report (PDF)",
                "Covenant Compliance Report PDF",
                "covenant_compliance_report_placeholder.txt",
            ),
            (
                "⬇️ Executive Presentation Deck (PPTX)",
                "Executive Presentation Deck PPTX",
                "executive_presentation_deck_placeholder.txt",
            ),
        ]

        for btn_label, export_label, filename in _export_buttons:
            st.download_button(
                label=btn_label,
                data=build_placeholder_export_bytes(export_label),
                file_name=filename,
                mime="text/plain",
                use_container_width=True,
            )


def _render_data_tabs(
    summary_table: pd.DataFrame,
    breach_table: pd.DataFrame,
    watchlist: pd.DataFrame,
    lender_exposure: pd.DataFrame,
) -> None:
    """Render the four data tabs: Summary, Breach, Watchlist, Lender Exposure."""
    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "Compliance Summary",
            "Breach Table",
            "Refinance Watchlist",
            "Lender Exposure",
        ]
    )

    with tab1:
        st.subheader("Portfolio Compliance Summary")
        if summary_table.empty:
            st.info("No compliance summary available.")
        else:
            st.dataframe(
                build_compliance_summary_display(summary_table),
                use_container_width=True,
                hide_index=True,
            )

    with tab2:
        st.subheader("Covenant Breach Table")
        if breach_table.empty:
            st.success("No active covenant breaches identified.")
        else:
            st.dataframe(
                build_breach_display(breach_table),
                use_container_width=True,
                hide_index=True,
            )

    with tab3:
        st.subheader("Refinance / Watchlist")
        if watchlist.empty:
            st.info("No refinance or watchlist loans currently identified.")
        else:
            st.dataframe(
                build_watchlist_display(watchlist),
                use_container_width=True,
                hide_index=True,
            )

    with tab4:
        st.subheader("Lender Exposure Summary")
        if lender_exposure.empty:
            st.info("No lender exposure summary available.")
        else:
            st.dataframe(
                build_lender_exposure_display(lender_exposure),
                use_container_width=True,
                hide_index=True,
            )


def _render_analyst_takeaway(
    kpis: dict[str, Any],
    summary_table: pd.DataFrame,
) -> None:
    """Render the Analyst Takeaway narrative block."""
    st.subheader("Analyst Takeaway")

    highest_risk = get_highest_risk_asset(summary_table)

    if highest_risk is not None:
        property_name = _series_get(
            highest_risk, "Property_Name", "Unknown Asset"
        )
        compliance_label = _series_get(
            highest_risk, "Compliance_Label", "Watchlist"
        )
        overall_rag = _series_get(highest_risk, "Overall_RAG", "Amber")

        breach_count = parse_int(kpis.get("Breach_Count", 0))
        watchlist_count = parse_int(kpis.get("Watchlist_Count", 0))
        nearest_reporting = fmt_days(kpis.get("Nearest_Reporting_Days"))

        st.write(
            f"The highest-priority compliance item is **{property_name}**, "
            f"currently labeled **{compliance_label}** with overall covenant "
            f"status **{overall_rag}**. "
            f"Portfolio-wide, there are **{breach_count:,} breach asset"
            f"{'s' if breach_count != 1 else ''}** and "
            f"**{watchlist_count:,} additional watchlist loan"
            f"{'s' if watchlist_count != 1 else ''}**, "
            f"with the nearest reporting deadline in **{nearest_reporting}**."
        )
    else:
        st.write(
            "No material compliance concerns are currently identified "
            "across the portfolio."
        )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Render the Compliance and Reporting Center page."""
    st.title("📋 Compliance and Reporting Center")
    st.caption(
        "Monitor debt covenant compliance, upcoming lender deadlines, "
        "refinance watchlist items, and reporting workflow readiness "
        "across the portfolio."
    )

    # ── Portfolio bundle ─────────────────────────────────────────────────────
    try:
        bundle = load_portfolio_bundle()
    except Exception:
        logger.exception("Fatal error loading compliance portfolio bundle")
        st.error(
            "⚠️ Unable to load portfolio compliance data. "
            "Please refresh or contact support."
        )
        st.stop()

    render_warnings(bundle.get("warnings", []))

    # Unpack once — already coerced inside the loader; no second pass needed.
    kpis = bundle["kpis"]
    summary_table = bundle["summary_table"]
    breach_table = bundle["breach_table"]
    deadlines = bundle["deadlines"]
    watchlist = bundle["watchlist"]
    lender_exposure = bundle["lender_exposure"]

    # ── Portfolio KPIs ───────────────────────────────────────────────────────
    st.markdown("---")
    _render_portfolio_kpis(kpis)
    st.markdown("---")

    # ── Covenant cards ───────────────────────────────────────────────────────
    property_options = get_property_options()
    _render_covenant_cards(property_options)
    st.markdown("---")

    # ── Deadlines + Exports ──────────────────────────────────────────────────
    _render_deadlines_and_exports(deadlines)
    st.markdown("---")

    # ── Data tabs ────────────────────────────────────────────────────────────
    _render_data_tabs(summary_table, breach_table, watchlist, lender_exposure)
    st.markdown("---")

    # ── Analyst Takeaway ─────────────────────────────────────────────────────
    _render_analyst_takeaway(kpis, summary_table)


if __name__ == "__main__":
    main()