"""
pages/2_Asset_Deep_Dive.py
--------------------------
Production-grade analyst workbench for property-level performance review.

Key design contracts:
- All percentage-like metrics displayed with fmt_pct() are expected to be decimals:
    0.125 -> 12.5%
- This page defensively normalizes ambiguous ratio inputs (e.g. 92 vs 0.92).
- All loader outputs are coerced before downstream use.
"""

from __future__ import annotations

import html
import logging
from typing import Any, Final

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from modules.data_loader import (
    get_performance_for_property,
    get_property_display_options,
    parse_property_selection,
)
from modules.market_analysis import generate_market_narrative
from modules.recommendation_engine import (
    generate_executive_commentary,
    get_property_recommendations,
    get_property_watchlist_score,
)
from modules.variance_analysis import (
    build_line_item_breakdown,
    build_variance_summary_table,
    compare_actual_vs_underwriting,
    get_expense_trend,
    get_monthly_noi_trend,
    get_t12_summary,
)
from utils.coercion import ensure_dataframe, ensure_dict, parse_float, safe_str
from utils.formatters import fmt_currency, fmt_multiple, fmt_pct, format_metric_value
from utils.ui_helpers import extract_property_name, rag_color, render_warnings
from utils.validators import first_existing_column

def _configure_logger() -> logging.Logger:
    """Configure and return the module logger without duplicate handlers."""
    _logger = logging.getLogger(__name__)
    if not _logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        )
        _logger.addHandler(handler)
        _logger.setLevel(logging.INFO)
    return _logger

logger = _configure_logger()

_COLOR_REVENUE:   Final[str] = "#1B4F72"
_COLOR_EXPENSES:  Final[str] = "#AF601A"
_COLOR_NOI:       Final[str] = "#1E8449"
_COLOR_OCCUPANCY: Final[str] = "#85C1E9"
_COLOR_RENT:      Final[str] = "#1B4F72"

_PERIOD_COL_CANDIDATES:  Final[tuple[str, ...]] = ("Year_Month", "Period")
_OCC_COL_CANDIDATES:     Final[tuple[str, ...]] = ("Occupancy", "Actual_Occupancy")
_RENT_COL_CANDIDATES:    Final[tuple[str, ...]] = ("Avg_Actual_Rent",)
_REVENUE_COL_CANDIDATES: Final[tuple[str, ...]] = ("Actual_Revenue",)
_UNITS_COL_CANDIDATES:   Final[tuple[str, ...]] = ("Units",)
_SORT_COL_CANDIDATES:    Final[tuple[str, ...]] = ("Priority_Rank", "Recommendation_Score")

_VARIANCE_CURRENCY_COLS: Final[tuple[str, ...]] = ("Budgeted", "Budget", "Actual", "Dollar_Variance")
_LINE_ITEM_PCT_COLS:     Final[tuple[str, ...]] = ("Pct_Variance", "Share_of_Total_Pct")

_NOI_TRACES: Final[tuple[tuple[str, str, str, int], ...]] = (
    ("Actual_Revenue",  "Revenue",  _COLOR_REVENUE,  3),
    ("Actual_Expenses", "Expenses", _COLOR_EXPENSES, 3),
    ("Actual_NOI",      "NOI",      _COLOR_NOI,      4),
)

st.set_page_config(
    page_title="Asset Deep Dive | AssetOptima Pro",
    page_icon="🏢",
    layout="wide",
)

def _fmt_currency_plain(x: Any) -> str:
    """Format as plain currency with no sign prefix."""
    return fmt_currency(x, decimals=0)

def _fmt_currency_signed(x: Any) -> str:
    """Format as currency with explicit +/- sign prefix."""
    return fmt_currency(x, decimals=0, include_sign=True)

def _fmt_pct_signed(x: Any) -> str:
    """Format a decimal ratio as percentage with explicit +/- sign."""
    return fmt_pct(x, decimals=1, include_sign=True)

def _normalize_ratio(value: Any, default: float = 0.0) -> float:
    """
    Normalize ratio-like values to decimal form expected by fmt_pct().

    Examples:
        0.922 -> 0.922
        92.2  -> 0.922
        16.3  -> 0.163
    """
    val = parse_float(value, default)
    if abs(val) > 1.0:
        return val / 100.0
    return val

def _sort_by_period(df: pd.DataFrame, period_col: str) -> pd.DataFrame:
    """
    Sort a DataFrame chronologically using a parsed datetime helper when possible.
    Falls back to raw sort if parsing fails.
    """
    if df.empty or period_col not in df.columns:
        return df

    out = df.copy()
    out["_period_dt"] = pd.to_datetime(out[period_col], errors="coerce")

    if out["_period_dt"].notna().any():
        out = out.sort_values("_period_dt")
    else:
        out = out.sort_values(period_col)

    return out.drop(columns=["_period_dt"], errors="ignore")

def _warn_if_suspicious_metrics(metrics: dict[str, Any]) -> None:
    """Surface warnings for suspicious KPI values that may indicate scaling/data issues."""
    uw_noi_gap = parse_float(metrics.get("uw_noi_gap"), 0.0)
    avg_occ = parse_float(metrics.get("avg_occupancy"), 0.0)

    if abs(uw_noi_gap) > 1.0:
        st.warning(
            "Underwriting NOI Gap exceeds 100%. This may be valid if underwriting NOI "
            "was near zero, but it can also indicate a source-data or scaling issue."
        )

    if not (0.0 <= avg_occ <= 1.0):
        st.warning(
            "Average occupancy is outside the expected 0%–100% range after normalization. "
            "Please verify source data."
        )

@st.cache_data(show_spinner=False, ttl=300)
def get_property_options() -> list[str]:
    """Load and cache property display options."""
    try:
        options = get_property_display_options()
        return options if isinstance(options, list) else []
    except Exception:
        logger.exception("Failed to load property display options")
        return []

@st.cache_data(show_spinner=False, ttl=300)
def load_property_bundle(property_id: str) -> dict[str, Any]:
    """
    Load all data components for a single property with independent fallbacks.
    """
    bundle: dict[str, Any] = {
        "perf":                 pd.DataFrame(),
        "t12_summary":          {},
        "variance_table":       pd.DataFrame(),
        "noi_trend":            pd.DataFrame(),
        "expense_trend":        pd.DataFrame(),
        "line_item_breakdown":  pd.DataFrame(),
        "uw_compare":           {},
        "recommendations":      pd.DataFrame(),
        "watchlist":            {},
        "executive_commentary": "",
        "market_narrative":     "",
        "warnings":             [],
    }

    loaders: dict[str, Any] = {
        "perf":                 lambda: get_performance_for_property(property_id),
        "t12_summary":          lambda: get_t12_summary(property_id),
        "variance_table":       lambda: build_variance_summary_table(property_id, trailing_months=12),
        "noi_trend":            lambda: get_monthly_noi_trend(property_id, trailing_months=12),
        "expense_trend":        lambda: get_expense_trend(property_id, trailing_months=12),
        "line_item_breakdown":  lambda: build_line_item_breakdown(property_id, trailing_months=12),
        "uw_compare":           lambda: compare_actual_vs_underwriting(property_id),
        "recommendations":      lambda: get_property_recommendations(property_id),
        "watchlist":            lambda: get_property_watchlist_score(property_id),
        "executive_commentary": lambda: generate_executive_commentary(property_id),
        "market_narrative":     lambda: generate_market_narrative(property_id),
    }

    for key, loader in loaders.items():
        try:
            bundle[key] = loader()
        except Exception:
            logger.exception("Failed loading bundle component=%s for property_id=%s", key, property_id)
            bundle["warnings"].append(f"'{key}' failed to load.")

    df_keys = (
        "perf", "variance_table", "noi_trend", "expense_trend",
        "line_item_breakdown", "recommendations",
    )
    dict_keys = ("t12_summary", "uw_compare", "watchlist")
    str_keys = ("executive_commentary", "market_narrative")

    for key in df_keys:
        bundle[key] = ensure_dataframe(bundle[key])
    for key in dict_keys:
        bundle[key] = ensure_dict(bundle[key])
    for key in str_keys:
        bundle[key] = safe_str(bundle[key])

    return bundle

def extract_property_metrics(
    t12_summary: dict[str, Any],
    uw_compare: dict[str, Any],
    watchlist: dict[str, Any],
) -> dict[str, Any]:
    """
    Extract and normalize display-ready scalar metrics.

    All percentage-like outputs returned here are decimals suitable for fmt_pct().
    """
    def _t12(key_lower: str, key_upper: str, default: float = 0.0) -> float:
        return parse_float(t12_summary.get(key_lower, t12_summary.get(key_upper, default)))

    def _uw(key_lower: str, key_upper: str, default: float = 0.0) -> float:
        return parse_float(uw_compare.get(key_lower, uw_compare.get(key_upper, default)))

    return {
        "t12_revenue":      _t12("actual_revenue", "Actual_Revenue"),
        "t12_expenses":     _t12("actual_expenses", "Actual_Expenses"),
        "t12_noi":          _t12("actual_noi", "Actual_NOI"),
        "avg_occupancy":    _normalize_ratio(_t12("avg_occupancy_pct", "avg_occupancy")),
        "noi_margin":       _normalize_ratio(_t12("noi_margin", "NOI_Margin")),
        "expense_ratio":    _normalize_ratio(_t12("expense_ratio", "Expense_Ratio")),
        "uw_noi_gap":       _normalize_ratio(_uw("noi_gap_pct", "NOI_Gap_Pct")),
        "watchlist_score":  parse_float(watchlist.get("Watchlist_Score", 0.0)),
        "watchlist_bucket": safe_str(watchlist.get("Watchlist_Bucket", "Green"), "Green"),
    }

def build_variance_display(df: pd.DataFrame) -> pd.DataFrame:
    """Format variance summary table for display."""
    df = ensure_dataframe(df)
    if df.empty:
        return df

    display_df = df.copy()

    for col in _VARIANCE_CURRENCY_COLS:
        if col in display_df.columns:
            fmt_fn = _fmt_currency_signed if col == "Dollar_Variance" else _fmt_currency_plain
            display_df[col] = display_df[col].map(fmt_fn)

    if "Pct_Variance" in display_df.columns:
        display_df["Pct_Variance"] = display_df["Pct_Variance"].map(_fmt_pct_signed)

    return display_df

def build_line_item_display(df: pd.DataFrame) -> pd.DataFrame:
    """Format expense line-item breakdown for display."""
    df = ensure_dataframe(df)
    if df.empty:
        return df

    display_df = df.copy()

    for col in _VARIANCE_CURRENCY_COLS:
        if col in display_df.columns:
            fmt_fn = _fmt_currency_signed if col == "Dollar_Variance" else _fmt_currency_plain
            display_df[col] = display_df[col].map(fmt_fn)

    for col in _LINE_ITEM_PCT_COLS:
        if col in display_df.columns:
            display_df[col] = display_df[col].map(_fmt_pct_signed)

    return display_df

def build_underwriting_display(uw_compare: dict[str, Any]) -> pd.DataFrame:
    """Convert underwriting comparison dict into a two-column display table."""
    uw_compare = ensure_dict(uw_compare)
    if not uw_compare:
        return pd.DataFrame()

    df = pd.DataFrame({
        "Metric": list(uw_compare.keys()),
        "Value": list(uw_compare.values()),
    })
    df["Value"] = [
        format_metric_value(metric, value)
        for metric, value in zip(df["Metric"], df["Value"])
    ]
    return df

def build_noi_trend_chart(noi_df: pd.DataFrame) -> go.Figure | None:
    """Build revenue / expense / NOI trend chart."""
    noi_df = ensure_dataframe(noi_df)
    period_col = first_existing_column(noi_df, list(_PERIOD_COL_CANDIDATES))
    if noi_df.empty or period_col is None:
        return None

    noi_df = _sort_by_period(noi_df, period_col)
    noi_df = noi_df.tail(12).copy()

    fig = go.Figure()
    for col, label, color, width in _NOI_TRACES:
        if col in noi_df.columns:
            series = pd.to_numeric(noi_df[col], errors="coerce")
            if series.notna().any():
                fig.add_trace(go.Scatter(
                    x=noi_df[period_col],
                    y=series,
                    mode="lines+markers",
                    name=label,
                    line=dict(color=color, width=width),
                ))

    if not fig.data:
        return None

    fig.update_layout(
        title="NOI, Revenue & Expense Trend (T12)",
        xaxis_title="Period",
        yaxis_title="Amount ($)",
        hovermode="x unified",
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend_title_text="Metric",
    )
    return fig

def build_expense_stack_chart(expense_df: pd.DataFrame) -> go.Figure | None:
    """Build stacked bar chart of expense categories over time."""
    expense_df = ensure_dataframe(expense_df)
    period_col = first_existing_column(expense_df, list(_PERIOD_COL_CANDIDATES))
    if expense_df.empty or period_col is None:
        return None

    expense_df = _sort_by_period(expense_df, period_col)
    expense_df = expense_df.tail(12).copy()

    value_cols = [c for c in expense_df.columns if c != period_col]
    if not value_cols:
        return None

    fig = go.Figure()
    palette = px.colors.qualitative.Set2

    for idx, col in enumerate(value_cols):
        series = pd.to_numeric(expense_df[col], errors="coerce")
        if series.notna().any():
            fig.add_trace(go.Bar(
                x=expense_df[period_col],
                y=series,
                name=col.replace("_", " "),
                marker_color=palette[idx % len(palette)],
            ))

    if not fig.data:
        return None

    fig.update_layout(
        title="Expense Category Trend (T12)",
        xaxis_title="Period",
        yaxis_title="Expenses ($)",
        barmode="stack",
        hovermode="x unified",
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend_title_text="Category",
    )
    return fig

def build_occupancy_rent_chart(perf_df: pd.DataFrame) -> go.Figure | None:
    """
    Build dual-axis chart of occupancy (bars) and average rent (line).

    Occupancy is normalized to decimal form for plotting.
    """
    perf_df = ensure_dataframe(perf_df)
    period_col = first_existing_column(perf_df, list(_PERIOD_COL_CANDIDATES))
    if perf_df.empty or period_col is None:
        return None

    occ_col = first_existing_column(perf_df, list(_OCC_COL_CANDIDATES))
    rent_col = first_existing_column(perf_df, list(_RENT_COL_CANDIDATES))

    df = _sort_by_period(perf_df, period_col).tail(12).copy()

    # Occupancy
    if occ_col:
        occ = pd.to_numeric(df[occ_col], errors="coerce")
        if occ.dropna().notna().any():
            if occ.dropna().max() > 1:
                occ = occ / 100.0
        df["Occ_Dec"] = occ.fillna(0.0)
    else:
        df["Occ_Dec"] = 0.0

    # Rent
    if rent_col:
        df["Rent_Display"] = pd.to_numeric(df[rent_col], errors="coerce")
    else:
        revenue_col = first_existing_column(df, list(_REVENUE_COL_CANDIDATES))
        units_col = first_existing_column(df, list(_UNITS_COL_CANDIDATES))

        if revenue_col:
            if units_col:
                units_series = pd.to_numeric(df[units_col], errors="coerce")
                units_series = units_series.fillna(method="ffill").fillna(method="bfill")
                units_series = units_series.clip(lower=1.0)
            else:
                units_series = pd.Series([1.0] * len(df), index=df.index, dtype=float)

            occupied_units = (units_series * df["Occ_Dec"]).clip(lower=1.0)
            df["Rent_Display"] = pd.to_numeric(df[revenue_col], errors="coerce") / occupied_units
        else:
            df["Rent_Display"] = pd.NA

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df[period_col],
        y=df["Occ_Dec"],
        name="Occupancy",
        marker_color=_COLOR_OCCUPANCY,
        yaxis="y1",
    ))

    if pd.to_numeric(df["Rent_Display"], errors="coerce").notna().any():
        fig.add_trace(go.Scatter(
            x=df[period_col],
            y=pd.to_numeric(df["Rent_Display"], errors="coerce"),
            mode="lines+markers",
            name="Avg Rent / Unit",
            line=dict(color=_COLOR_RENT, width=3),
            yaxis="y2",
        ))

    fig.update_layout(
        title="Occupancy & Average Rent Trend (T12)",
        xaxis_title="Period",
        yaxis=dict(title="Occupancy", tickformat=".0%"),
        yaxis2=dict(title="Avg Rent ($)", overlaying="y", side="right"),
        hovermode="x unified",
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend_title_text="Metric",
    )
    return fig

def build_recommendation_cards(df: pd.DataFrame) -> None:
    """Render recommendation cards safely."""
    df = ensure_dataframe(df)
    if df.empty:
        st.success("No material recommendation items at this time.")
        return

    sort_col = first_existing_column(df, list(_SORT_COL_CANDIDATES))
    render_df = df.sort_values(sort_col, ascending=False) if sort_col else df.copy()

    for _, row in render_df.iterrows():
        priority = safe_str(row.get("Priority", "Medium"), "Medium")
        category = safe_str(row.get("Category", "General"), "General")
        title = safe_str(row.get("Title", "Recommendation"), "Recommendation")
        recommendation = safe_str(row.get("Recommendation", ""))
        rationale = safe_str(row.get("Rationale", ""))

        st.markdown(
            f"""
            <div style="border-left: 6px solid {rag_color(priority)}; padding: 0.75rem 1rem;
                        margin-bottom: 0.75rem; background-color: #F8F9FA;
                        border-radius: 0.35rem;">
                <strong>{html.escape(priority)} | {html.escape(category)}</strong><br>
                <strong>{html.escape(title)}</strong><br>
                <span>{html.escape(recommendation)}</span><br>
                <em>{html.escape(rationale)}</em>
            </div>
            """,
            unsafe_allow_html=True,
        )

def _render_chart_or_info(fig: go.Figure | None, message: str) -> None:
    """Render a Plotly chart or an informational message."""
    if fig is not None:
        st.plotly_chart(fig, width="stretch")
    else:
        st.info(message)

def _render_df_or_info(df: pd.DataFrame, message: str) -> None:
    """Render DataFrame or informational message."""
    if not df.empty:
        st.dataframe(df, width="stretch", hide_index=True)
    else:
        st.info(message)

def render_kpi_rows(metrics: dict[str, Any]) -> None:
    """Render top-level KPI rows."""
    col_rev, col_exp, col_noi, col_occ, col_ws = st.columns(5)
    with col_rev:
        st.metric("T12 Revenue", fmt_currency(metrics["t12_revenue"]))
    with col_exp:
        st.metric("T12 Expenses", fmt_currency(metrics["t12_expenses"]))
    with col_noi:
        st.metric("T12 NOI", fmt_currency(metrics["t12_noi"]))
    with col_occ:
        st.metric("Avg Occupancy", fmt_pct(metrics["avg_occupancy"]))
    with col_ws:
        st.metric("Watchlist Score", f"{metrics['watchlist_score']:.1f}")

def main() -> None:
    """Main page entry point."""
    st.title("🏢 Asset Deep Dive")
    st.caption(
        "Analyst workbench for reviewing property-level performance, "
        "variances, market context, and recommended actions."
    )

    options = get_property_options()
    if not options:
        st.error("No property options are available.")
        st.stop()

    selected = st.selectbox("Select Property", options=options, index=0)
    property_id = parse_property_selection(selected)
    if not property_id:
        st.error("Unable to parse the selected property.")
        st.stop()

    property_name = extract_property_name(selected, property_id)

    try:
        bundle = load_property_bundle(property_id)
        render_warnings(bundle.get("warnings", []))

        perf_df = ensure_dataframe(bundle.get("perf"))
        t12_summary = ensure_dict(bundle.get("t12_summary"))
        uw_compare = ensure_dict(bundle.get("uw_compare"))
        watchlist = ensure_dict(bundle.get("watchlist"))
        variance_table = ensure_dataframe(bundle.get("variance_table"))
        noi_trend = ensure_dataframe(bundle.get("noi_trend"))
        expense_trend = ensure_dataframe(bundle.get("expense_trend"))
        line_item_df = ensure_dataframe(bundle.get("line_item_breakdown"))
        recommendations = ensure_dataframe(bundle.get("recommendations"))
        exec_commentary = safe_str(bundle.get("executive_commentary")) or "No executive commentary available."
        market_narrative = safe_str(bundle.get("market_narrative")) or "No market narrative available."

        if perf_df.empty:
            st.error("No performance data available for the selected property.")
            st.stop()

        metrics = extract_property_metrics(t12_summary, uw_compare, watchlist)
        st.markdown("---")
        render_kpi_rows(metrics)
        _warn_if_suspicious_metrics(metrics)
        st.markdown("---")

        col_exec, col_market = st.columns(2)
        with col_exec:
            st.subheader("Executive Commentary")
            st.info(exec_commentary)
        with col_market:
            st.subheader("Market Narrative")
            st.info(market_narrative)
        st.markdown("---")

        col_variance, col_recs = st.columns((1.5, 1.0))
        with col_variance:
            st.subheader("Variance Analysis (Trailing 12 Months)")
            _render_df_or_info(
                build_variance_display(variance_table),
                "No variance data available.",
            )
        with col_recs:
            st.subheader("Top Recommendations")
            build_recommendation_cards(recommendations)
        st.markdown("---")

        fig_noi = build_noi_trend_chart(noi_trend)
        fig_occ_rent = build_occupancy_rent_chart(perf_df)
        fig_expense = build_expense_stack_chart(expense_trend)

        col_noi_chart, col_occ_chart = st.columns(2)
        with col_noi_chart:
            _render_chart_or_info(fig_noi, "No NOI trend data available.")
        with col_occ_chart:
            _render_chart_or_info(fig_occ_rent, "No occupancy / rent trend data available.")

        _render_chart_or_info(fig_expense, "No expense trend data available.")
        st.markdown("---")

        tab_expense, tab_uw, tab_notes = st.tabs([
            "Expense Line-Item Detail",
            "Underwriting Comparison",
            "Analyst Notes",
        ])

        with tab_expense:
            _render_df_or_info(
                build_line_item_display(line_item_df),
                "No expense detail available.",
            )
        with tab_uw:
            _render_df_or_info(
                build_underwriting_display(uw_compare),
                "No underwriting comparison available.",
            )
        with tab_notes:
            # Placeholder for analyst notes — implementation pending
            st.info("Analyst notes feature coming soon.")

    except Exception as exc:
        logger.exception("Asset Deep Dive page failed for property_id=%s", property_id)
        st.error("⚠️ Page Error: Unable to load the Asset Deep Dive page.")
        st.exception(exc)


if __name__ == "__main__":
    main()