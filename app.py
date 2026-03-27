from __future__ import annotations

import logging
import math
from typing import Any, Final

import pandas as pd
import streamlit as st

from modules.data_loader import get_portfolio_kpis, load_all
from modules.market_analysis import get_market_summary_table
from modules.recommendation_engine import get_portfolio_watchlist
from modules.valuation import get_valuation_summary_table
from modules.variance_analysis import get_portfolio_noi_variance_summary


def _configure_logger() -> logging.Logger:
    """
    Configure and return the module-level logger.

    Ensures no duplicate handlers are attached across reloads,
    which is critical in Streamlit's execution model.

    Returns:
        Configured Logger instance.
    """
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


APP_TITLE:    Final[str] = "AssetOptima Pro"
APP_ICON:     Final[str] = "🏙️"
APP_SUBTITLE: Final[str] = "Multifamily Asset Management Intelligence Platform"

PRIORITY_DISPLAY_MAP: Final[dict[str, str]] = {
    "High":   "🔴 High",
    "Medium": "🟡 Medium",
    "Low":    "🟢 Low",
}

VALUATION_CURRENCY_COLS: Final[tuple[str, ...]] = (
    "Acquisition_Value",
    "Indicated_Value",
    "Unrealized_Gain",
    "Value_Per_Unit",
)
VALUATION_PCT_COLS: Final[tuple[str, ...]] = (
    "Unrealized_Gain_Pct",
    "Implied_Cap_Rate",
    "vs_Benchmark_Pct",
)
VARIANCE_MONEY_COLS: Final[tuple[str, ...]] = (
    "T12_NOI_Actual",
    "T12_NOI_Budget",
    "NOI_Variance_Dollar",
)


def _empty_df() -> pd.DataFrame:
    """
    Return a new empty DataFrame.

    Never use a module-level shared empty DataFrame as a constant —
    downstream mutations would corrupt all references to it.

    Returns:
        Fresh empty pd.DataFrame.
    """
    return pd.DataFrame()


def _default_data_bundle() -> dict[str, Any]:
    """
    Return a fresh default data bundle with safe empty values.

    Returns:
        Dict with all expected bundle keys set to safe defaults.
    """
    return {
        "data":             {},
        "properties":       _empty_df(),
        "portfolio_kpis":   {},
        "market_summary":   _empty_df(),
        "watchlist":        _empty_df(),
        "valuation_summary": _empty_df(),
        "variance_summary": _empty_df(),
    }


st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help":    "https://github.com/danielabategaray/assetoptima-pro",
        "Report a bug": "https://github.com/danielabategaray/assetoptima-pro/issues",
        "About":       f"{APP_TITLE} — {APP_SUBTITLE}",
    },
)


def ensure_dataframe(value: Any) -> pd.DataFrame:
    """
    Coerce a value to a DataFrame.

    Args:
        value: Any object to coerce.

    Returns:
        The original value if already a DataFrame, otherwise a new empty DataFrame.
    """
    return value if isinstance(value, pd.DataFrame) else _empty_df()


def _is_nan_safe(value: Any) -> bool:
    """
    Safely test whether a scalar value is NaN without raising on array-like input.

    ``pd.isna()`` raises ``ValueError`` when passed an array or Series because
    the truth value is ambiguous. This wrapper handles that edge case.

    Args:
        value: Any scalar or object to test.

    Returns:
        True if the value is NaN or None, False otherwise.
    """
    if value is None:
        return True
    try:
        # math.isnan is safe for scalars and faster than pd.isna for this purpose
        if isinstance(value, float):
            return math.isnan(value)
        # For non-float scalars, pd.isna is safe
        result = pd.isna(value)
        # Guard against array-like returns
        return bool(result) if not hasattr(result, "__len__") else False
    except (TypeError, ValueError):
        return False


def parse_float(value: Any, default: float = 0.0) -> float:
    """
    Safely parse an arbitrary value into a float.

    Handles None, NaN, numeric types, and strings containing
    commas, dollar signs, and percent signs.

    Args:
        value:   The value to parse.
        default: Fallback value when parsing fails.

    Returns:
        Parsed float, or ``default`` on failure.
    """
    if _is_nan_safe(value):
        return default
    try:
        if isinstance(value, str):
            cleaned = value.strip().replace(",", "").replace("$", "").replace("%", "")
            return float(cleaned) if cleaned else default
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_int(value: Any, default: int = 0) -> int:
    """
    Safely parse an arbitrary value into an int.

    Args:
        value:   The value to parse.
        default: Fallback value when parsing fails.

    Returns:
        Parsed int, or ``default`` on failure.
    """
    return int(parse_float(value, default=float(default)))


def _build_sign(numeric_val: float, include_sign: bool) -> str:
    """
    Determine the sign prefix string for a formatted number.

    Args:
        numeric_val:  The numeric value being formatted.
        include_sign: Whether to prepend '+' for positive values.

    Returns:
        One of '-', '+', or ''.
    """
    if numeric_val < 0:
        return "-"
    if include_sign and numeric_val > 0:
        return "+"
    return ""


def fmt_currency(
        value: Any,
        decimals: int = 0,
        default: str = "$0",
        include_sign: bool = False,
    ) -> str:
    """
    Format a value as a currency string.

    Args:
        value:        The value to format.
        decimals:     Decimal places to render.
        default:      String returned when the value cannot be parsed.
        include_sign: Prepend '+' for positive values when True.

    Returns:
        Formatted currency string, e.g. ``"\$1,250,000"`` or ``"+\$50,000"``.
    """
    numeric_val = parse_float(value, default=float("nan"))
    if _is_nan_safe(numeric_val):
        return default

    sign = _build_sign(numeric_val, include_sign)
    return f"{sign}${abs(numeric_val):,.{decimals}f}"


def fmt_pct(
    value: Any,
    decimals: int = 1,
    default: str = "0.0%",
    include_sign: bool = False,
    assume_fraction_when_abs_le_1: bool = True,
) -> str:
    """
    Format a value as a percentage string.

    When ``assume_fraction_when_abs_le_1`` is True, values with
    absolute magnitude ≤ 1 are treated as fractions (e.g. 0.95 → 95.0%).
    Values greater than 1 are treated as already-percentage points
    (e.g. 12.3 → 12.3%).

    Note:
        Sign prefix is built separately from the numeric formatting to
        avoid double-application that occurs when Python's built-in
        ``%``-format spec is combined with a manually prepended sign.

    Args:
        value:                      The value to format.
        decimals:                   Decimal places to render.
        default:                    String returned when parsing fails.
        include_sign:               Prepend '+' for positive values when True.
        assume_fraction_when_abs_le_1: Treat abs(value) ≤ 1 as a fraction.

    Returns:
        Formatted percentage string, e.g. ``"95.0%"`` or ``"+2.5%"``.
    """
    numeric_val = parse_float(value, default=float("nan"))
    if _is_nan_safe(numeric_val):
        return default

    sign    = _build_sign(numeric_val, include_sign)
    abs_val = abs(numeric_val)

    if assume_fraction_when_abs_le_1 and abs_val <= 1.0:
        # Multiply by 100 manually to keep sign logic clean
        body = f"{abs_val * 100:.{decimals}f}%"
    else:
        body = f"{abs_val:.{decimals}f}%"

    return f"{sign}{body}"


def fmt_multiple(value: Any, decimals: int = 2, default: str = "0.00x") -> str:
    """
    Format a value as an equity/debt multiple (e.g. ``"1.25x"``).

    Args:
        value:    The value to format.
        decimals: Decimal places to render.
        default:  String returned when parsing fails.

    Returns:
        Formatted multiple string.
    """
    numeric_val = parse_float(value, default=float("nan"))
    if _is_nan_safe(numeric_val):
        return default
    return f"{numeric_val:,.{decimals}f}x"


def safe_mean(df: pd.DataFrame, column: str, default: float = 0.0) -> float:
    """
    Compute the mean of a DataFrame column, coercing non-numeric values.

    Args:
        df:      Source DataFrame.
        column:  Column name to average.
        default: Fallback when the column is missing or all-NaN.

    Returns:
        Column mean as float, or ``default``.
    """
    if df.empty or column not in df.columns:
        return default
    series = pd.to_numeric(df[column], errors="coerce").dropna()
    return float(series.mean()) if not series.empty else default


def safe_sum(df: pd.DataFrame, column: str, default: float = 0.0) -> float:
    """
    Compute the sum of a DataFrame column, coercing non-numeric values.

    Args:
        df:      Source DataFrame.
        column:  Column name to sum.
        default: Fallback when the column is missing or all-NaN.

    Returns:
        Column sum as float, or ``default``.
    """
    if df.empty or column not in df.columns:
        return default
    series = pd.to_numeric(df[column], errors="coerce").dropna()
    return float(series.sum()) if not series.empty else default


def _fmt_currency_plain(x: Any) -> str:
    """Format as plain currency (no sign)."""
    return fmt_currency(x, decimals=0)


def _fmt_currency_signed(x: Any) -> str:
    """Format as currency with explicit +/- sign."""
    return fmt_currency(x, decimals=0, include_sign=True)


def _fmt_pct_signed(x: Any) -> str:
    """Format as percentage with explicit +/- sign."""
    return fmt_pct(x, decimals=1, include_sign=True)


@st.cache_data(show_spinner=True, ttl=300, max_entries=10)
def load_app_data() -> dict[str, Any]:
    """
    Load all core application datasets with graceful per-source fallbacks.

    Each data source is fetched independently so a failure in one module
    does not prevent the rest of the dashboard from rendering.

    Returns:
        Data bundle dict; falls back to ``_default_data_bundle()`` on
        complete failure.
    """
    try:
        raw_data = load_all()
        raw_data = raw_data if isinstance(raw_data, dict) else {}

        def _load_safe(fn: Any, *args: Any) -> pd.DataFrame:
            """Call ``fn(*args)`` and ensure the result is a DataFrame."""
            try:
                return ensure_dataframe(fn(*args))
            except Exception:
                logger.exception("Data source failed: %s", getattr(fn, "__name__", fn))
                return _empty_df()

        def _load_kpis_safe() -> dict[str, Any]:
            try:
                result = get_portfolio_kpis()
                return result if isinstance(result, dict) else {}
            except Exception:
                logger.exception("Portfolio KPI load failed")
                return {}

        return {
            "data":             raw_data,
            "properties":       ensure_dataframe(raw_data.get("properties")),
            "portfolio_kpis":   _load_kpis_safe(),
            "market_summary":   _load_safe(get_market_summary_table),
            "watchlist":        _load_safe(get_portfolio_watchlist),
            "valuation_summary": _load_safe(get_valuation_summary_table),
            "variance_summary": _load_safe(get_portfolio_noi_variance_summary),
        }

    except Exception:
        logger.exception("Critical failure in load_app_data — returning empty bundle")
        return _default_data_bundle()


def extract_portfolio_metrics(bundle: dict[str, Any]) -> dict[str, float | int]:
    """
    Extract and validate top-level portfolio KPI metrics from a data bundle.

    Args:
        bundle: Data bundle returned by ``load_app_data()``.

    Returns:
        Dict of validated scalar metrics ready for display.
    """
    properties       = ensure_dataframe(bundle.get("properties"))
    portfolio_kpis   = bundle.get("portfolio_kpis") or {}
    market_summary   = ensure_dataframe(bundle.get("market_summary"))
    valuation_summary = ensure_dataframe(bundle.get("valuation_summary"))
    watchlist        = ensure_dataframe(bundle.get("watchlist"))

    watchlist_assets = 0
    if not watchlist.empty and "Watchlist_Bucket" in watchlist.columns:
        watchlist_assets = int(
            (watchlist["Watchlist_Bucket"].astype(str) != "Green").sum()
        )

    return {
        "property_count":      len(properties),
        "total_units":         parse_int(portfolio_kpis.get("total_units", 0)),
        "portfolio_noi":       parse_float(portfolio_kpis.get("total_noi", 0.0)),
        "weighted_occupancy":  parse_float(portfolio_kpis.get("weighted_avg_occupancy", 0.0)),
        "portfolio_dscr":      parse_float(portfolio_kpis.get("portfolio_dscr", 0.0)),
        "assets_in_breach":    parse_int(portfolio_kpis.get("assets_in_breach", 0)),
        "avg_market_score":    safe_mean(market_summary, "Positioning_Score"),
        "total_unrealized_gain": safe_sum(valuation_summary, "Unrealized_Gain"),
        "watchlist_assets":    watchlist_assets,
    }


def render_sidebar() -> None:
    """Render the application sidebar with navigation and data notice."""
    with st.sidebar:
        st.title(f"{APP_ICON} {APP_TITLE}")
        st.caption(APP_SUBTITLE)

        st.markdown("### Navigation")
        st.markdown(
            """
            **Core Modules:**
            1. **Portfolio Overview** — Executive dashboard
            2. **Asset Deep Dive** — Property-level analysis
            3. **Forecast & Valuation** — Financial modeling
            4. **Market Intelligence** — Competitive analysis
            5. **Business Plan Tracker** — Initiative monitoring
            6. **Compliance & Reporting** — Regulatory oversight
            """
        )
        st.markdown("---")
        with st.expander("ℹ️ Data Notice", expanded=False):
            st.info(
                "All data is entirely fictitious, anonymized, and used solely "
                "for demonstration purposes."
            )


def render_header() -> None:
    """Render the main page title, subtitle, and description."""
    st.title(f"{APP_ICON} {APP_TITLE}")
    st.subheader(APP_SUBTITLE)
    st.markdown(
        "**AssetOptima Pro** is an interactive portfolio analytics platform designed "
        "to demonstrate professional multifamily real estate asset management workflows."
    )
    st.markdown("---")


def render_portfolio_snapshot(metrics: dict[str, float | int]) -> None:
    """
    Render the top-level KPI metric card grid.

    Args:
        metrics: Dict produced by ``extract_portfolio_metrics()``.
    """
    st.header("📊 Portfolio Snapshot")

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    with k1:
        st.metric("Properties",   f"{metrics['property_count']:,}")
    with k2:
        st.metric("Total Units",  f"{metrics['total_units']:,}")
    with k3:
        st.metric(
            "Portfolio NOI",
            fmt_currency(metrics["portfolio_noi"]),
            help="Net Operating Income",
        )
    with k4:
        st.metric(
            "Weighted Occupancy",
            fmt_pct(metrics["weighted_occupancy"]),
            help="Portfolio-weighted average occupancy rate",
        )
    with k5:
        st.metric(
            "Portfolio DSCR",
            fmt_multiple(metrics["portfolio_dscr"]),
            help="Debt Service Coverage Ratio",
        )
    with k6:
        st.metric("Assets in Breach", f"{metrics['assets_in_breach']}")

    s1, s2, s3 = st.columns(3)
    with s1:
        st.metric("Market Positioning", f"{metrics['avg_market_score']:.1f}")
    with s2:
        st.metric("Unrealized Gain",    fmt_currency(metrics["total_unrealized_gain"]))
    with s3:
        st.metric("Watchlist Assets",   f"{metrics['watchlist_assets']}")


def render_variance_summary(variance_summary: pd.DataFrame) -> None:
    """
    Render the trailing-12-month NOI performance and variance table.

    Args:
        variance_summary: DataFrame from ``get_portfolio_noi_variance_summary()``.
    """
    st.header("📈 Performance & Variance Summary (Trailing 12 Months)")

    # ensure_dataframe already called in main() — but guard here for
    # direct/unit-test calls to this function in isolation
    variance_summary = ensure_dataframe(variance_summary)
    if variance_summary.empty:
        st.info("No variance data available.")
        st.markdown("---")
        return

    display_df = variance_summary.copy()

    for col in VARIANCE_MONEY_COLS:
        if col in display_df.columns:
            fmt_fn = _fmt_currency_signed if col == "NOI_Variance_Dollar" else _fmt_currency_plain
            display_df[col] = display_df[col].map(fmt_fn)

    if "NOI_Variance_Pct" in display_df.columns:
        display_df["NOI_Variance_Pct"] = display_df["NOI_Variance_Pct"].map(_fmt_pct_signed)

    st.dataframe(display_df, use_container_width=True, hide_index=True)
    st.markdown("---")


def render_watchlist_snapshot(watchlist: pd.DataFrame) -> None:
    """
    Render a compact top-5 portfolio watchlist preview.

    Args:
        watchlist: DataFrame from ``get_portfolio_watchlist()``.
    """
    st.markdown("### 🚨 Portfolio Watchlist")

    watchlist = ensure_dataframe(watchlist)
    if watchlist.empty:
        st.info("No watchlist records available.")
        return

    display_df = watchlist.head(5).copy()

    if "Watchlist_Score" in display_df.columns:
        display_df["Watchlist_Score"] = (
            pd.to_numeric(display_df["Watchlist_Score"], errors="coerce").round(1)
        )

    if "Top_Priority" in display_df.columns:
        mapped = display_df["Top_Priority"].astype(str).map(PRIORITY_DISPLAY_MAP)
        display_df["Top_Priority"] = mapped.fillna(display_df["Top_Priority"])

    st.dataframe(display_df, use_container_width=True, hide_index=True)


def render_valuation_snapshot(valuation_summary: pd.DataFrame) -> None:
    """
    Render a compact top-5 valuation summary preview.

    Args:
        valuation_summary: DataFrame from ``get_valuation_summary_table()``.
    """
    st.markdown("### 💰 Valuation Snapshot")

    valuation_summary = ensure_dataframe(valuation_summary)
    if valuation_summary.empty:
        st.info("No valuation snapshot available.")
        return

    display_df = valuation_summary.head(5).copy()

    for col in VALUATION_CURRENCY_COLS:
        if col in display_df.columns:
            display_df[col] = display_df[col].map(_fmt_currency_plain)

    for col in VALUATION_PCT_COLS:
        if col in display_df.columns:
            display_df[col] = display_df[col].map(_fmt_pct_signed)

    st.dataframe(display_df, use_container_width=True, hide_index=True)


def render_feature_snapshots(
    watchlist: pd.DataFrame,
    valuation_summary: pd.DataFrame,
) -> None:
    """
    Render the side-by-side watchlist and valuation highlight panels.

    Args:
        watchlist:         Watchlist DataFrame.
        valuation_summary: Valuation summary DataFrame.
    """
    st.header("🔍 Feature Highlights")
    col1, col2 = st.columns((1.1, 1.0))
    with col1:
        render_watchlist_snapshot(watchlist)
    with col2:
        render_valuation_snapshot(valuation_summary)


def main() -> None:
    """
    Main application entry point.

    Orchestrates sidebar, data loading, and all top-level page sections.
    All bundle values are coerced to DataFrames before being passed to
    render functions so no render function needs to defend against None.
    """
    render_sidebar()

    try:
        with st.spinner("Loading portfolio data..."):
            bundle = load_app_data()

        # Coerce all DataFrame values once here — render functions
        # receive clean types and don't need defensive re-coercion
        watchlist         = ensure_dataframe(bundle.get("watchlist"))
        valuation_summary = ensure_dataframe(bundle.get("valuation_summary"))
        variance_summary  = ensure_dataframe(bundle.get("variance_summary"))

        metrics = extract_portfolio_metrics(bundle)

        render_header()
        render_portfolio_snapshot(metrics)
        render_variance_summary(variance_summary)
        render_feature_snapshots(watchlist, valuation_summary)

        st.success(
            "🚀 Ready to explore? Use the navigation menu on the left "
            "to access detailed analysis modules."
        )

    except Exception:
        logger.exception("Unhandled application error in main()")
        st.error(
            "⚠️ An unexpected error occurred while loading the dashboard. "
            "Please refresh the page or contact support."
        )


if __name__ == "__main__":
    main()