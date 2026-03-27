"""
pages/1_Portfolio_Overview.py
-----------------------------
Executive portfolio dashboard for AssetOptima Pro.

Changelog (this version):
- Fixed four-quote SyntaxError on module docstring
- Removed persist="disk" from cache — prevents stale cross-restart data
- load_portfolio_data() now returns safe fallback instead of re-raising
- Replaced all lambda default-arg workarounds with named formatter helpers
- All bundle.get() values coerced to correct types in main() before use
- Fixed bar chart variance color logic to use proper numeric coercion
- Fixed px.colors.sequential.Blues[3] single-string misuse — now a named constant
- render_kpis() uses .get() with fallbacks — no KeyError on missing metrics
- Replaced inline ternary st.* statements with explicit if/else blocks
- Compliance tab guards against nested dict/list values in DataFrame display
- Logger configured via dedicated function — no duplicate handlers
- All functions documented with Google-style docstrings
- Column name groups extracted as module-level constants
- build_executive_summary markdown-escapes dynamic string values
"""

from __future__ import annotations

import logging
from typing import Any, Final

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from modules.data_loader import get_portfolio_kpis, load_all
from modules.debt_compliance import get_portfolio_compliance_kpis
from modules.market_analysis import get_market_summary_table
from modules.recommendation_engine import get_portfolio_watchlist
from modules.valuation import get_valuation_summary_table
from utils.coercion import ensure_dataframe, ensure_dict, parse_float, parse_int
from utils.formatters import fmt_currency, fmt_multiple, fmt_pct
from utils.ui_helpers import render_warnings
from utils.validators import (
    first_existing_column,
    is_valid_dataframe,
    safe_mean,
    safe_sum,
)

def _configure_logger() -> logging.Logger:
    """
    Configure and return the module-level logger.

    Prevents duplicate handler attachment across Streamlit reruns.

    Returns:
        Configured Logger instance.
    """
    _logger = logging.getLogger(__name__)
    if not _logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
            )
        )
        _logger.addHandler(handler)
        _logger.setLevel(logging.INFO)
    return _logger


logger = _configure_logger()

PAGE_TITLE: Final[str] = "Portfolio Overview | AssetOptima Pro"
PAGE_ICON:  Final[str] = "📊"
_NOI_LINE_COLOR:    Final[str] = "#1B4F72"
_BUDGET_LINE_COLOR: Final[str] = "#7F8C8D"
_VARIANCE_POS_COLOR: Final[str] = "#27AE60"
_VARIANCE_NEG_COLOR: Final[str] = "#E74C3C"
_BAR_BASE_COLOR:    Final[str] = "#2E86C1"   # replaces fragile Blues[3] index

# Column name groups — defined once, referenced everywhere
_WATCHLIST_PREFERRED_COLS: Final[tuple[str, ...]] = (
    "Property_ID",
    "Property_Name",
    "Market",
    "Watchlist_Score",
    "Watchlist_Bucket",
    "Top_Priority",
    "Top_Recommendation",
)
_MARKET_CURRENCY_COLS: Final[tuple[str, ...]] = (
    "Subject_Rent",
    "Comp_Avg_Rent",
    "Rent_Premium",
)
_MARKET_PCT_COLS: Final[tuple[str, ...]] = (
    "Rent_Premium_Pct",
    "Subject_Occupancy",
    "Comp_Avg_Occupancy",
    "Occupancy_Gap",
    "Rent_Growth_Gap",
)
_VALUATION_CURRENCY_COLS: Final[tuple[str, ...]] = (
    "Acquisition_Value",
    "Indicated_Value",
    "Unrealized_Gain",
    "Value_Per_Unit",
)
_VALUATION_PCT_COLS: Final[tuple[str, ...]] = (
    "Unrealized_Gain_Pct",
    "Implied_Cap_Rate",
    "vs_Benchmark_Pct",
)
_NOI_TREND_REQUIRED_COLS: Final[tuple[str, ...]] = ("Year_Month", "Actual_NOI")
_BUDGET_COL_CANDIDATES: Final[tuple[str, ...]] = ("Budgeted_NOI", "Budget_NOI")
_PHASE_COL_CANDIDATES:  Final[tuple[str, ...]] = ("Value_Add_Phase", "Asset_Phase")

st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON,
    layout="wide",
)

def _fmt_currency_plain(x: Any) -> str:
    """Format as plain currency with no sign prefix."""
    return fmt_currency(x, decimals=0)


def _fmt_currency_signed(x: Any) -> str:
    """Format as currency with explicit +/- sign prefix."""
    return fmt_currency(x, decimals=0, include_sign=True)


def _fmt_pct_signed(x: Any) -> str:
    """Format as percentage with explicit +/- sign prefix."""
    return fmt_pct(x, decimals=1, include_sign=True)


def _fmt_pct_plain(x: Any) -> str:
    """Format as plain percentage with no sign prefix."""
    return fmt_pct(x, decimals=1, include_sign=False)

@st.cache_data(show_spinner=True, max_entries=5, ttl=300)
def load_portfolio_data() -> dict[str, Any]:
    """
    Load all portfolio data sources with per-source graceful fallbacks.

    ``persist="disk"`` is intentionally omitted — disk persistence causes
    stale data to survive app restarts and is inappropriate for a live
    dashboard. Use ``ttl`` for time-based expiry instead.

    Each data source is fetched independently so a failure in one module
    does not prevent the rest of the dashboard from rendering.

    Returns:
        Data bundle dict with all expected keys guaranteed to be present.
        Returns safe empty defaults on complete failure rather than
        re-raising, so the dashboard can render a graceful error state.
    """
    def _safe_df(fn: Any, *args: Any) -> pd.DataFrame:
        """Call fn(*args), coerce to DataFrame, log on failure."""
        try:
            return ensure_dataframe(fn(*args))
        except Exception:
            logger.exception("Data source failed: %s", getattr(fn, "__name__", str(fn)))
            return pd.DataFrame()

    def _safe_dict(fn: Any, *args: Any) -> dict[str, Any]:
        """Call fn(*args), coerce to dict, log on failure."""
        try:
            result = fn(*args)
            return ensure_dict(result)
        except Exception:
            logger.exception("Data source failed: %s", getattr(fn, "__name__", str(fn)))
            return {}

    try:
        raw = ensure_dict(load_all())
    except Exception:
        logger.exception("load_all() failed — returning empty bundle")
        raw = {}

    return {
        "data":             raw,
        "properties":       ensure_dataframe(raw.get("properties")),
        "monthly_perf":     ensure_dataframe(raw.get("monthly_performance")),
        "portfolio_kpis":   _safe_dict(get_portfolio_kpis),
        "watchlist":        _safe_df(get_portfolio_watchlist),
        "market_summary":   _safe_df(get_market_summary_table),
        "compliance_kpis":  _safe_dict(get_portfolio_compliance_kpis),
        "valuation_summary": _safe_df(get_valuation_summary_table),
        "warnings":         [],
    }

def extract_dashboard_metrics(bundle: dict[str, Any]) -> dict[str, float | int]:
    """
    Extract and validate top-level dashboard KPI metrics from a data bundle.

    Args:
        bundle: Data bundle returned by ``load_portfolio_data()``.

    Returns:
        Dict of validated scalar metrics ready for display.
        All values are guaranteed to be present with safe numeric defaults.
    """
    portfolio_kpis    = ensure_dict(bundle.get("portfolio_kpis"))
    watchlist         = ensure_dataframe(bundle.get("watchlist"))
    market_summary    = ensure_dataframe(bundle.get("market_summary"))
    compliance_kpis   = ensure_dict(bundle.get("compliance_kpis"))
    valuation_summary = ensure_dataframe(bundle.get("valuation_summary"))

    watchlist_assets = 0
    if is_valid_dataframe(watchlist, ["Watchlist_Bucket"], min_rows=1):
        watchlist_assets = int(
            (watchlist["Watchlist_Bucket"].astype(str) != "Green").sum()
        )

    return {
        "total_noi":               parse_float(portfolio_kpis.get("total_noi", 0.0)),
        "weighted_avg_occupancy":  parse_float(portfolio_kpis.get("weighted_avg_occupancy", 0.0)),
        "portfolio_dscr":          parse_float(portfolio_kpis.get("portfolio_dscr", 0.0)),
        "total_units":             parse_int(portfolio_kpis.get("total_units", 0)),
        "assets_in_breach":        parse_int(portfolio_kpis.get("assets_in_breach", 0)),
        "watchlist_assets":        watchlist_assets,
        "avg_market_positioning":  safe_mean(market_summary, "Positioning_Score", 0.0),
        "refi_candidates":         parse_int(compliance_kpis.get("Refinance_Candidate_Count", 0)),
        "total_unrealized_gain":   safe_sum(valuation_summary, "Unrealized_Gain", 0.0),
    }

def build_portfolio_noi_trend(perf_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate monthly performance data into a portfolio-level NOI trend.

    Args:
        perf_df: Monthly performance DataFrame (one row per property per period).

    Returns:
        Aggregated trend DataFrame indexed by ``Year_Month``, or an empty
        DataFrame when input is insufficient.
    """
    perf_df = ensure_dataframe(perf_df)
    if not is_valid_dataframe(perf_df, list(_NOI_TREND_REQUIRED_COLS), min_rows=1):
        return pd.DataFrame()

    agg_dict: dict[str, str] = {"Actual_NOI": "sum"}
    for col in ("Actual_Revenue", "Actual_Expenses"):
        if col in perf_df.columns:
            agg_dict[col] = "sum"

    budget_col = first_existing_column(perf_df, list(_BUDGET_COL_CANDIDATES))
    if budget_col:
        agg_dict[budget_col] = "sum"

    trend = (
        perf_df.groupby("Year_Month", dropna=False)
        .agg(agg_dict)
        .reset_index()
        .sort_values("Year_Month")
    )

    if budget_col and budget_col in trend.columns:
        trend["NOI_Variance"] = trend["Actual_NOI"] - trend[budget_col]
        trend["NOI_Variance_Pct"] = trend["NOI_Variance"] / (
            trend[budget_col].replace(0, pd.NA)
        )

    return trend


def build_watchlist_display(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare the watchlist DataFrame for display.

    Rounds ``Watchlist_Score`` and filters to preferred column order.

    Args:
        df: Raw watchlist DataFrame.

    Returns:
        Display-ready DataFrame, or empty DataFrame if input is empty.
    """
    df = ensure_dataframe(df)
    if df.empty:
        return df

    display_df = df.copy()

    if "Watchlist_Score" in display_df.columns:
        display_df["Watchlist_Score"] = (
            pd.to_numeric(display_df["Watchlist_Score"], errors="coerce").round(1)
        )

    existing_cols = [c for c in _WATCHLIST_PREFERRED_COLS if c in display_df.columns]
    return display_df[existing_cols].copy()


def build_market_display(df: pd.DataFrame) -> pd.DataFrame:
    """
    Format the market summary DataFrame for display.

    Applies currency and percentage formatting to relevant columns.

    Args:
        df: Raw market summary DataFrame.

    Returns:
        Display-ready DataFrame, or empty DataFrame if input is empty.
    """
    df = ensure_dataframe(df)
    if df.empty:
        return df

    display_df = df.copy()

    for col in _MARKET_CURRENCY_COLS:
        if col in display_df.columns:
            display_df[col] = display_df[col].map(_fmt_currency_plain)

    for col in _MARKET_PCT_COLS:
        if col in display_df.columns:
            display_df[col] = display_df[col].map(_fmt_pct_signed)

    if "Positioning_Score" in display_df.columns:
        display_df["Positioning_Score"] = (
            pd.to_numeric(display_df["Positioning_Score"], errors="coerce").round(1)
        )

    return display_df


def build_valuation_display(df: pd.DataFrame) -> pd.DataFrame:
    """
    Format the valuation summary DataFrame for display.

    Args:
        df: Raw valuation summary DataFrame.

    Returns:
        Display-ready DataFrame, or empty DataFrame if input is empty.
    """
    df = ensure_dataframe(df)
    if df.empty:
        return df

    display_df = df.copy()

    for col in _VALUATION_CURRENCY_COLS:
        if col in display_df.columns:
            fmt_fn = _fmt_currency_signed if col == "Unrealized_Gain" else _fmt_currency_plain
            display_df[col] = display_df[col].map(fmt_fn)

    for col in _VALUATION_PCT_COLS:
        if col in display_df.columns:
            display_df[col] = display_df[col].map(_fmt_pct_signed)

    return display_df


def _flatten_compliance_kpis(compliance_kpis: dict[str, Any]) -> dict[str, Any]:
    """
    Flatten a compliance KPIs dict for safe single-row DataFrame display.

    Nested dicts and list values are converted to strings so
    ``pd.DataFrame([...])`` does not produce broken multi-level columns
    or raise on unhashable types.

    Args:
        compliance_kpis: Raw compliance KPIs dict.

    Returns:
        Flattened dict with all values as scalar-safe types.
    """
    flat: dict[str, Any] = {}
    for key, val in compliance_kpis.items():
        if isinstance(val, (dict, list)):
            flat[key] = str(val)
        else:
            flat[key] = val
    return flat


def build_executive_summary(
    metrics: dict[str, float | int],
    watchlist: pd.DataFrame,
    properties: pd.DataFrame,
) -> str:
    """
    Generate a plain-English executive summary paragraph from portfolio metrics.

    All numeric values are formatted and LaTeX triggers ($) are escaped before
    being embedded in markdown to prevent special characters from corrupting rendering.
    """
    watchlist  = ensure_dataframe(watchlist)
    properties = ensure_dataframe(properties)

    total_units    = int(metrics.get("total_units", 0))
    property_count = len(properties)
    total_noi      = float(metrics.get("total_noi", 0.0))

    # Using bullet points and bold headers for a more attractive, readable dashboard
    parts: list[str] = [
        f"* **Scale & Yield:** The portfolio encompasses **{total_units:,} units** across "
        f"**{property_count:,} properties**, generating a robust "
        f"**{fmt_currency(total_noi)}** in annual NOI."
    ]

    if not watchlist.empty and "Watchlist_Bucket" in watchlist.columns:
        buckets     = watchlist["Watchlist_Bucket"].astype(str)
        red_count   = int((buckets == "Red").sum())
        amber_count = int((buckets == "Amber").sum())
        if red_count > 0 or amber_count > 0:
            parts.append(
                f"* **Risk Management:** Active monitoring is required for "
                f"**{red_count} high-priority (Red)** and "
                f"**{amber_count} medium-priority (Amber)** assets."
            )

    assets_in_breach = int(metrics.get("assets_in_breach", 0))
    if assets_in_breach > 0:
        parts.append(
            f"* **Compliance Alert:** ⚠️ **{assets_in_breach} asset{'s' if assets_in_breach != 1 else ''}** "
            f"{'are' if assets_in_breach != 1 else 'is'} currently in debt covenant breach."
        )

    avg_positioning = float(metrics.get("avg_market_positioning", 0.0))
    position_label  = (
        "strong"   if avg_positioning > 70 else
        "moderate" if avg_positioning > 50 else
        "weak"
    )
    parts.append(
        f"* **Market Stance:** Average market positioning remains **{position_label}** "
        f"(score: {avg_positioning:.1f}/100)."
    )

    gain = float(metrics.get("total_unrealized_gain", 0.0))
    if gain > 0:
        parts.append(
            f"* **Value Creation:** Portfolio unrealized gain stands impressively at **{fmt_currency(gain)}**, "
            f"indicating highly positive value creation."
        )
    elif gain < 0:
        parts.append(
            f"* **Valuation Pressure:** Portfolio unrealized value movement stands at **{fmt_currency(gain)}**, "
            f"suggesting near-term valuation pressure."
        )

    # The magic fix: Join with newlines and escape all dollar signs to prevent the LaTeX rendering bug
    final_summary = "\n".join(parts).replace("$", "\\$")
    return final_summary

def build_portfolio_noi_chart(trend_df: pd.DataFrame) -> go.Figure | None:
    """
    Build the portfolio NOI trend line + budget overlay + variance bar chart.

    Args:
        trend_df: Aggregated NOI trend DataFrame from ``build_portfolio_noi_trend()``.

    Returns:
        Plotly Figure, or None if data is insufficient for a meaningful chart.
    """
    trend_df = ensure_dataframe(trend_df)
    if not is_valid_dataframe(trend_df, list(_NOI_TREND_REQUIRED_COLS), min_rows=2):
        return None

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=trend_df["Year_Month"],
        y=trend_df["Actual_NOI"],
        mode="lines+markers",
        name="Actual NOI",
        line=dict(color=_NOI_LINE_COLOR, width=3),
    ))

    budget_col = first_existing_column(trend_df, list(_BUDGET_COL_CANDIDATES))
    if budget_col and trend_df[budget_col].notna().any():
        fig.add_trace(go.Scatter(
            x=trend_df["Year_Month"],
            y=trend_df[budget_col],
            mode="lines+markers",
            name="Budget NOI",
            line=dict(color=_BUDGET_LINE_COLOR, width=2, dash="dash"),
        ))

    if "NOI_Variance" in trend_df.columns:
        # Use pd.to_numeric for correct coercion — values are already numeric
        # scalars here, but this is explicit and safe for any edge cases
        variance_numeric = pd.to_numeric(trend_df["NOI_Variance"], errors="coerce")
        bar_colors = [
            _VARIANCE_POS_COLOR if (pd.notna(v) and v >= 0) else _VARIANCE_NEG_COLOR
            for v in variance_numeric
        ]
        fig.add_trace(go.Bar(
            x=trend_df["Year_Month"],
            y=variance_numeric,
            name="Variance",
            marker_color=bar_colors,
            opacity=0.28,
            yaxis="y2",
        ))

    fig.update_layout(
        title="Portfolio NOI Trend",
        xaxis_title="Period",
        yaxis_title="NOI ($)",
        yaxis2=dict(title="Variance ($)", overlaying="y", side="right"),
        hovermode="x unified",
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    return fig


def build_market_allocation_chart(df: pd.DataFrame) -> go.Figure | None:
    """
    Build a donut chart showing portfolio unit allocation by market.

    Args:
        df: Properties DataFrame with ``Market``, ``Units``, and ``Property_ID`` columns.

    Returns:
        Plotly Figure, or None if data is insufficient.
    """
    df = ensure_dataframe(df)
    if not is_valid_dataframe(df, ["Market", "Units", "Property_ID"], min_rows=1):
        return None

    alloc = (
        df.groupby("Market", dropna=False)
        .agg(
            Units=("Units", "sum"),
            Property_Count=("Property_ID", "count"),
            Avg_Units=("Units", "mean"),
        )
        .reset_index()
        .sort_values("Units", ascending=False)
    )
    alloc["Market"] = alloc["Market"].fillna("Unknown")

    fig = px.pie(
        alloc,
        names="Market",
        values="Units",
        custom_data=["Property_Count", "Avg_Units"],
        hole=0.45,
        title="Portfolio Allocation by Market",
        color_discrete_sequence=px.colors.sequential.Blues_r,
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    fig.update_layout(plot_bgcolor="white", paper_bgcolor="white", title_x=0.5)
    return fig


def build_phase_distribution_chart(df: pd.DataFrame) -> go.Figure | None:
    """
    Build a bar chart showing property count by value-add phase.

    Args:
        df: Properties DataFrame with a phase column and ``Property_ID``/``Units``.

    Returns:
        Plotly Figure, or None if no phase column is found or data is insufficient.
    """
    df = ensure_dataframe(df)
    phase_col = first_existing_column(df, list(_PHASE_COL_CANDIDATES))
    if not phase_col or not is_valid_dataframe(df, ["Property_ID", "Units"], min_rows=1):
        return None

    phase_df = (
        df.groupby(phase_col, dropna=False)
        .agg(
            Property_Count=("Property_ID", "count"),
            Units=("Units", "sum"),
            Avg_Units=("Units", "mean"),
        )
        .reset_index()
        .rename(columns={phase_col: "Value_Add_Phase"})
        .sort_values("Property_Count", ascending=False)
    )
    phase_df["Value_Add_Phase"] = phase_df["Value_Add_Phase"].fillna("Unknown")

    fig = go.Figure(data=[go.Bar(
        x=phase_df["Value_Add_Phase"],
        y=phase_df["Property_Count"],
        marker_color=_BAR_BASE_COLOR,
        text=phase_df["Property_Count"],
        textposition="outside",
    )])
    fig.update_layout(
        title="Value-Add Phase Distribution",
        xaxis_title="Value-Add Phase",
        yaxis_title="Property Count",
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=False,
    )
    return fig

def render_kpis(metrics: dict[str, float | int]) -> None:
    """
    Render primary and secondary KPI metric card rows.

    Uses ``.get()`` with safe defaults on all metric access so a missing
    key never raises a ``KeyError``.

    Args:
        metrics: Dict produced by ``extract_dashboard_metrics()``.
    """
    st.markdown("### Key Performance Indicators")
    cols = st.columns(5)
    with cols[0]:
        st.metric("Portfolio NOI",      fmt_currency(metrics.get("total_noi", 0.0)))
    with cols[1]:
        st.metric("Weighted Occupancy", fmt_pct(metrics.get("weighted_avg_occupancy", 0.0)))
    with cols[2]:
        st.metric("Portfolio DSCR",     fmt_multiple(metrics.get("portfolio_dscr", 0.0)))
    with cols[3]:
        st.metric("Total Units",        f"{int(metrics.get('total_units', 0)):,}")
    with cols[4]:
        st.metric("Assets in Breach",   f"{int(metrics.get('assets_in_breach', 0))}")

    st.markdown("### Secondary Metrics")
    cols = st.columns(4)
    with cols[0]:
        st.metric("Watchlist Assets",   f"{int(metrics.get('watchlist_assets', 0))}")
    with cols[1]:
        st.metric("Market Positioning", f"{float(metrics.get('avg_market_positioning', 0.0)):.1f}")
    with cols[2]:
        st.metric("Refi Candidates",    f"{int(metrics.get('refi_candidates', 0))}")
    with cols[3]:
        st.metric("Unrealized Gain",    fmt_currency(metrics.get("total_unrealized_gain", 0.0)))


def _render_chart_or_info(fig: go.Figure | None, message: str) -> None:
    """
    Render a Plotly chart, or an info notice if no chart is available.

    Replaces the inline ternary ``st.plotly_chart(...) if fig else st.info(...)``
    anti-pattern with an explicit, readable conditional.

    Args:
        fig:     Plotly Figure to render, or None.
        message: Info message to display when ``fig`` is None.
    """
    if fig is not None:
        st.plotly_chart(fig, width="stretch")
    else:
        st.info(message)


def _render_analytics_grid(
    noi_fig: go.Figure | None,
    market_fig: go.Figure | None,
    phase_fig: go.Figure | None,
    watchlist_display: pd.DataFrame,
) -> None:
    """
    Render the 2×2 analytics chart and watchlist grid.

    Args:
        noi_fig:           NOI trend figure or None.
        market_fig:        Market allocation figure or None.
        phase_fig:         Phase distribution figure or None.
        watchlist_display: Prepared watchlist DataFrame.
    """
    left_main, right_main = st.columns((1.8, 1.2))
    with left_main:
        _render_chart_or_info(noi_fig, "Insufficient data for NOI trend visualization.")
    with right_main:
        _render_chart_or_info(market_fig, "Insufficient data for market allocation chart.")

    bottom_left, bottom_right = st.columns((1.2, 1.8))
    with bottom_left:
        _render_chart_or_info(phase_fig, "Insufficient data for phase distribution chart.")
    with bottom_right:
        st.subheader("🚨 Priority Watchlist")
        if watchlist_display.empty:
            st.info("No watchlist items identified.")
        else:
            st.dataframe(watchlist_display, width="stretch", hide_index=True)


def _render_tabs(
    market_display: pd.DataFrame,
    compliance_kpis: dict[str, Any],
    valuation_display: pd.DataFrame,
) -> None:
    """
    Render the Market / Compliance / Valuation detail tabs.

    Args:
        market_display:    Formatted market summary DataFrame.
        compliance_kpis:   Compliance KPIs dict.
        valuation_display: Formatted valuation summary DataFrame.
    """
    tabs = st.tabs([
        "🏙️ Market Positioning",
        "⚖️ Compliance Snapshot",
        "💰 Valuation Snapshot",
    ])

    with tabs[0]:
        if not market_display.empty:
            st.dataframe(market_display, width="stretch", hide_index=True)
        else:
            st.info("No market positioning data available.")

    with tabs[1]:
        if compliance_kpis:
            flat = _flatten_compliance_kpis(compliance_kpis)
            st.dataframe(
                pd.DataFrame([flat]),
                width="stretch",
                hide_index=True,
            )
        else:
            st.info("No compliance data available.")

    with tabs[2]:
        if not valuation_display.empty:
            st.dataframe(valuation_display, width="stretch", hide_index=True)
        else:
            st.info("No valuation data available.")

def main() -> None:
    """
    Main entry point for the Portfolio Overview page.

    Orchestrates data loading, metric extraction, chart construction,
    and all UI section rendering. All bundle values are coerced to their
    correct types before being passed downstream so no helper function
    needs to defend against None.
    """
    st.title("📊 Portfolio Overview")
    st.caption(
        "Executive dashboard for multifamily portfolio performance, "
        "risk assessment, and strategic insights."
    )

    try:
        with st.spinner("Loading portfolio analytics..."):
            bundle = load_portfolio_data()

        # Coerce all bundle values to their correct types once —
        # downstream functions receive clean, typed inputs
        properties        = ensure_dataframe(bundle.get("properties"))
        monthly_perf      = ensure_dataframe(bundle.get("monthly_perf"))
        watchlist         = ensure_dataframe(bundle.get("watchlist"))
        market_summary    = ensure_dataframe(bundle.get("market_summary"))
        valuation_summary = ensure_dataframe(bundle.get("valuation_summary"))
        compliance_kpis   = ensure_dict(bundle.get("compliance_kpis"))

        render_warnings(bundle.get("warnings", []))
        metrics = extract_dashboard_metrics(bundle)

        render_kpis(metrics)
        st.markdown("---")

        st.markdown("### 📈 Portfolio Analytics")

        trend_df   = build_portfolio_noi_trend(monthly_perf)
        noi_fig    = build_portfolio_noi_chart(trend_df)
        market_fig = build_market_allocation_chart(properties)
        phase_fig  = build_phase_distribution_chart(properties)

        _render_analytics_grid(
            noi_fig,
            market_fig,
            phase_fig,
            build_watchlist_display(watchlist),
        )

        st.markdown("---")

        _render_tabs(
            build_market_display(market_summary),
            compliance_kpis,
            build_valuation_display(valuation_summary),
        )

        st.markdown("---")

        st.subheader("🎯 Executive Summary")
        st.info(
            build_executive_summary(metrics, watchlist, properties), 
            icon="📋"
        )

        st.markdown("---")
        st.caption(
            "Data as of latest reporting period | "
            "Refresh dashboard for latest updates"
        )

    except Exception as exc:
        logger.exception("Portfolio Overview page failed")
        st.error("⚠️ Dashboard Error: Unable to load portfolio dashboard.")
        st.exception(exc)


if __name__ == "__main__":
    main()