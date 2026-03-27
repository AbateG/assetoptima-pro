"""
pages/3_Forecast_and_Valuation.py
----------------------------------
Interactive financial modeling workbench for AssetOptima Pro.

Provides property-level forecast modeling, IRR sensitivity analysis,
hold/sell decision support, and multi-method valuation reconciliation.

Design goals:
- Stable across Streamlit reruns
- Defensive at all external data boundaries
- Cached heavy computations
- Clear, modular rendering
- Safe formatting of dynamic content
- Better production resilience
"""

from __future__ import annotations

import html
import logging
import types
from dataclasses import dataclass
from typing import Any, Final, Callable

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from modules.data_loader import get_property_display_options, parse_property_selection
from modules.forecasting import (
    build_irr_sensitivity_matrix,
    build_property_forecast,
    compare_forecast_vs_underwriting,
    get_base_assumptions_for_property,
    get_forecast_as_dataframe,
    get_hold_sell_recommendation,
    get_sensitivity_matrix_display,
    calculate_annual_debt_service,
)
from modules.valuation import (
    build_scenario_valuation,
    build_valuation_reconciliation,
    check_value_vs_benchmark,
    get_value_bridge,
)
from utils.coercion import ensure_dataframe, ensure_dict, parse_float, parse_int, safe_str
from utils.formatters import fmt_currency, fmt_multiple, fmt_pct, format_metric_value
from utils.ui_helpers import extract_property_name, rag_color, render_warnings


# =============================================================================
# LOGGING
# =============================================================================

def _configure_logger() -> logging.Logger:
    """Configure and return the module-level logger."""
    module_logger = logging.getLogger(__name__)
    if not module_logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        )
        module_logger.addHandler(handler)
        module_logger.setLevel(logging.INFO)
        module_logger.propagate = False
    return module_logger


logger = _configure_logger()


# =============================================================================
# CONSTANTS
# =============================================================================

DEFAULT_ASSUMPTIONS: Final = types.MappingProxyType({
    "rent_growth": 0.03,
    "vacancy_rate": 0.05,
    "expense_growth": 0.03,
    "capex_per_unit": 300,
    "exit_cap_rate": 0.055,
    "hold_years": 5,
    "purchase_price": 10_000_000,
    "ltv": 0.70,
    "interest_rate": 0.06,
})

SCENARIO_CURRENCY_COLS: Final[tuple[str, ...]] = (
    "Direct_Cap_Value",
    "DCF_Value",
    "Sales_Comp_Value",
    "Concluded_Value",
    "Value_Per_Unit",
    "Unrealized_Gain",
)

SCENARIO_PCT_COLS: Final[tuple[str, ...]] = (
    "Cap_Rate",
    "Unrealized_Gain_Pct",
)

RATE_TOKENS: Final[frozenset[str]] = frozenset({
    "RATE", "IRR", "YIELD", "VACANCY", "GROWTH", "MARGIN",
})

FORECAST_DF_KEYS: Final[tuple[str, ...]] = (
    "forecast_df",
    "value_bridge_df",
    "sensitivity_matrix",
)

FORECAST_DICT_KEYS: Final[tuple[str, ...]] = (
    "forecast",
    "hold_sell",
    "forecast_vs_uw",
    "valuation",
    "scenario_valuation",
    "benchmark_check",
)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass(frozen=True)
class ForecastAssumptions:
    """Immutable container for user-controlled forecast assumptions."""
    rent_growth: float
    vacancy_rate: float
    expense_growth: float
    capex_per_unit: int
    exit_cap_rate: float
    hold_years: int


# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="Forecast & Valuation | AssetOptima Pro",
    page_icon="📈",
    layout="wide",
)


# =============================================================================
# GENERAL HELPERS
# =============================================================================

def _fmt_currency_plain(value: Any) -> str:
    """Format as currency with no sign prefix."""
    return fmt_currency(value, decimals=0)


def _fmt_currency_signed(value: Any) -> str:
    """Format as currency with explicit sign prefix."""
    return fmt_currency(value, decimals=0, include_sign=True)


def _fmt_pct_plain(value: Any) -> str:
    """Format as plain percentage with one decimal place."""
    return fmt_pct(value, decimals=1, include_sign=False)


def _fmt_pct_signed(value: Any) -> str:
    """Format as signed percentage with one decimal place."""
    return fmt_pct(value, decimals=1, include_sign=True)


def _fmt_pct_zero_dp(value: Any) -> str:
    """Format as percentage with zero decimal places."""
    return fmt_pct(value, decimals=0, include_sign=False)


def _to_numeric_series(series: pd.Series, fill_value: float | None = 0.0) -> pd.Series:
    """Safely coerce a Series to numeric."""
    numeric = pd.to_numeric(series, errors="coerce")
    if fill_value is not None:
        numeric = numeric.fillna(fill_value)
    return numeric


def _coerce_bundle_types(bundle: dict[str, Any]) -> dict[str, Any]:
    """Coerce known bundle keys to stable expected types."""
    clean = dict(bundle)

    for key in FORECAST_DF_KEYS:
        clean[key] = ensure_dataframe(clean.get(key))

    for key in FORECAST_DICT_KEYS:
        clean[key] = ensure_dict(clean.get(key))

    warnings = clean.get("warnings", [])
    clean["warnings"] = warnings if isinstance(warnings, list) else []

    return clean


def _clamp_float(value: Any, min_value: float, max_value: float, default: float) -> float:
    """Parse and clamp float values safely."""
    parsed = parse_float(value, default)
    return min(max(parsed, min_value), max_value)


def _clamp_int(value: Any, min_value: int, max_value: int, default: int) -> int:
    """Parse and clamp integer values safely."""
    parsed = parse_int(value, default)
    return min(max(parsed, min_value), max_value)


def _clean_metric_label(raw_label: str) -> str:
    """Convert raw metric names into presentation-friendly labels."""
    clean_name = str(raw_label).replace("_", " ").title()
    replacements = {
        "Uw": "UW",
        "Irr": "IRR",
        "Noi": "NOI",
        "Dcf": "DCF",
        "Ltv": "LTV",
        "Pct": "(%)",
        "Ppt": "(ppt)",
    }
    for old, new in replacements.items():
        clean_name = clean_name.replace(old, new)
    return clean_name


def _sort_forecast_df(df: pd.DataFrame) -> pd.DataFrame:
    """Sort forecast data by Year when possible."""
    df = ensure_dataframe(df).copy()
    if df.empty or "Year" not in df.columns:
        return df

    sort_key = pd.to_numeric(df["Year"], errors="coerce")
    df = df.assign(_year_sort_key=sort_key).sort_values("_year_sort_key", na_position="last")
    return df.drop(columns=["_year_sort_key"])


def _safe_html_text(value: Any, default: str = "") -> str:
    """Convert any value to escaped display-safe HTML text."""
    return html.escape(safe_str(value, default))


def _has_meaningful_number(value: Any) -> bool:
    """Return True if value can be parsed into a finite numeric value."""
    try:
        num = float(value)
        return np.isfinite(num)
    except Exception:
        return False


def _validate_assumptions(assumptions: ForecastAssumptions) -> list[str]:
    """Return human-readable warnings for aggressive or unusual assumptions."""
    warnings: list[str] = []

    if assumptions.rent_growth > 0.06:
        warnings.append("Rent growth assumption is aggressive relative to typical stabilized multifamily expectations.")
    if assumptions.vacancy_rate < 0.02:
        warnings.append("Vacancy assumption is very low and may understate operational risk.")
    if assumptions.expense_growth < 0.01:
        warnings.append("Expense growth assumption is very low and may understate cost inflation.")
    if assumptions.exit_cap_rate < 0.045:
        warnings.append("Exit cap rate is very low and may inflate terminal valuation.")
    if assumptions.capex_per_unit < 100:
        warnings.append("CapEx reserve per unit appears light and may understate recurring capital needs.")
    if assumptions.hold_years <= 3:
        warnings.append("Short hold periods can make valuation and IRR outputs more sensitive to terminal assumptions.")

    return warnings


# =============================================================================
# DATA LOADING
# =============================================================================

@st.cache_data(show_spinner=False, ttl=300)
def get_property_options() -> list[str]:
    """Load and cache the property display options."""
    try:
        options = get_property_display_options()
        if not isinstance(options, list):
            return []
        return [item for item in options if isinstance(item, str)]
    except Exception:
        logger.exception("Failed to load property options")
        return []


@st.cache_data(show_spinner=False, ttl=300)
def get_base_assumptions(property_id: str) -> dict[str, Any]:
    """Load property-specific assumptions merged over project defaults."""
    try:
        assumptions = ensure_dict(get_base_assumptions_for_property(property_id))
        return {**DEFAULT_ASSUMPTIONS, **assumptions}
    except Exception:
        logger.exception("Failed to load base assumptions for property_id=%s", property_id)
        return dict(DEFAULT_ASSUMPTIONS)


@st.cache_data(show_spinner=True, ttl=300)
def load_forecast_bundle(property_id: str, assumptions: ForecastAssumptions) -> dict[str, Any]:
    """
    Build and cache the complete forecast and valuation bundle.

    Each component is loaded independently so isolated failures degrade gracefully.
    """
    bundle: dict[str, Any] = {
        "forecast": {},
        "forecast_df": pd.DataFrame(),
        "hold_sell": {},
        "forecast_vs_uw": {},
        "valuation": {},
        "scenario_valuation": {},
        "benchmark_check": {},
        "value_bridge_df": pd.DataFrame(),
        "sensitivity_matrix": pd.DataFrame(),
        "warnings": [],
    }

    try:
        bundle["forecast"] = ensure_dict(
            build_property_forecast(
                property_id=property_id,
                rent_growth=assumptions.rent_growth,
                vacancy_rate=assumptions.vacancy_rate,
                expense_growth=assumptions.expense_growth,
                capex_per_unit=assumptions.capex_per_unit,
                exit_cap_rate=assumptions.exit_cap_rate,
                hold_years=assumptions.hold_years,
            )
        )
    except Exception:
        logger.exception("build_property_forecast failed for property_id=%s", property_id)
        bundle["warnings"].append("Forecast model failed to build.")

    forecast = ensure_dict(bundle.get("forecast"))

    forecast_dependent_loaders: dict[str, Callable[[], Any]] = {
        "forecast_df": lambda: get_forecast_as_dataframe(forecast),
        "hold_sell": lambda: get_hold_sell_recommendation(property_id, forecast),
        "forecast_vs_uw": lambda: compare_forecast_vs_underwriting(property_id, forecast),
        "valuation": lambda: build_valuation_reconciliation(
            property_id=property_id,
            cap_rate_override=assumptions.exit_cap_rate,
            forecast=forecast,
        ),
    }

    independent_loaders: dict[str, Callable[[], Any]] = {
        "scenario_valuation": lambda: build_scenario_valuation(property_id),
        "value_bridge_df": lambda: get_value_bridge(property_id),
        "sensitivity_matrix": lambda: build_irr_sensitivity_matrix(
            property_id=property_id,
            base_rent_growth=assumptions.rent_growth,
            base_vacancy_rate=assumptions.vacancy_rate,
            base_expense_growth=assumptions.expense_growth,
            base_capex_per_unit=assumptions.capex_per_unit,
            base_hold_years=assumptions.hold_years,
        ),
    }

    if forecast:
        for key, loader in forecast_dependent_loaders.items():
            try:
                bundle[key] = loader()
            except Exception:
                logger.exception("Bundle component '%s' failed for property_id=%s", key, property_id)
                bundle["warnings"].append(f"{key!r} failed to load.")
    else:
        bundle["warnings"].append("Forecast-dependent analyses were skipped because the forecast output was unavailable.")

    for key, loader in independent_loaders.items():
        try:
            bundle[key] = loader()
        except Exception:
            logger.exception("Bundle component '%s' failed for property_id=%s", key, property_id)
            bundle["warnings"].append(f"{key!r} failed to load.")

    try:
        valuation = ensure_dict(bundle.get("valuation"))
        concluded_value = valuation.get("concluded_value")
        if _has_meaningful_number(concluded_value):
            bundle["benchmark_check"] = ensure_dict(
                check_value_vs_benchmark(property_id, parse_float(concluded_value, 0.0))
            )
        else:
            bundle["warnings"].append("Benchmark comparison was skipped because concluded valuation was unavailable.")
    except Exception:
        logger.exception("Benchmark check failed for property_id=%s", property_id)
        bundle["warnings"].append("Benchmark check failed to load.")

    return _coerce_bundle_types(bundle)


# =============================================================================
# DATA TRANSFORMATION
# =============================================================================

def _build_recon_df(valuation: dict[str, Any]) -> pd.DataFrame:
    """Build the valuation reconciliation table."""
    valuation = ensure_dict(valuation)
    weights = ensure_dict(valuation.get("weights_used"))

    direct_cap = ensure_dict(valuation.get("direct_cap"))
    sales_comp = ensure_dict(valuation.get("sales_comp"))
    dcf = ensure_dict(valuation.get("dcf"))

    rows = [
        {
            "Method": "Direct Cap",
            "Indicated_Value": parse_float(direct_cap.get("indicated_value", 0.0), 0.0),
            "Weight": parse_float(weights.get("income_approach", 0.0), 0.0),
        },
        {
            "Method": "Sales Comp",
            "Indicated_Value": parse_float(sales_comp.get("blended_comp_value", 0.0), 0.0),
            "Weight": parse_float(weights.get("sales_comp_approach", 0.0), 0.0),
        },
        {
            "Method": "DCF",
            "Indicated_Value": parse_float(dcf.get("total_dcf_value", 0.0), 0.0),
            "Weight": parse_float(weights.get("dcf_approach", 0.0), 0.0),
        },
    ]

    df = pd.DataFrame(rows)
    df["Weighted_Value"] = _to_numeric_series(df["Indicated_Value"]) * _to_numeric_series(df["Weight"])
    return df


def _format_forecast_column(col: str, series: pd.Series) -> pd.Series:
    """Apply display formatting to a forecast column based on explicit rules."""
    if col == "Year":
        numeric_years = pd.to_numeric(series, errors="coerce")
        return numeric_years.map(lambda x: "" if pd.isna(x) else str(int(x)))

    numeric_series = _to_numeric_series(series)

    col_upper = col.upper()

    if "DSCR" in col_upper:
        return numeric_series.map(fmt_multiple)

    if any(token in col_upper for token in RATE_TOKENS):
        return numeric_series.map(_fmt_pct_plain)

    return numeric_series.map(_fmt_currency_plain)


def _format_sensitivity_label(value: Any) -> str:
    """Format a sensitivity axis label as a percentage when numeric."""
    if isinstance(value, (float, int, np.floating, np.integer)):
        return f"{float(value):.1%}"
    return str(value)


def _format_sensitivity_cell(value: float) -> str:
    """Format a sensitivity heatmap cell as a percentage, or N/A."""
    try:
        if np.isnan(value):
            return "N/A"
    except TypeError:
        return "N/A"
    return f"{float(value):.1%}"


def _infer_sensitivity_axis_titles(matrix_df: pd.DataFrame) -> tuple[str, str, str]:
    """
    Infer sensitivity chart title and axis labels conservatively.

    If upstream metadata is unavailable, we use generic labels instead of
    potentially incorrect financial labels.
    """
    # Conservative default: do not overclaim specific dimensions.
    return (
        "IRR Sensitivity Analysis",
        "Scenario Dimension",
        "Scenario Dimension",
    )


# =============================================================================
# CHART BUILDERS
# =============================================================================

def build_forecast_chart(
    forecast_df: pd.DataFrame,
    debt_service: float = 0.0,
) -> go.Figure | None:
    """
    Build a combined forecast chart for EGI, NOI, Levered CF, and debt service.
    """
    forecast_df = _sort_forecast_df(ensure_dataframe(forecast_df))
    if forecast_df.empty or "Year" not in forecast_df.columns:
        return None

    years = forecast_df["Year"]
    fig = go.Figure()

    if "EGI" in forecast_df.columns:
        fig.add_trace(
            go.Scatter(
                x=years,
                y=_to_numeric_series(forecast_df["EGI"]),
                name="Gross Income (EGI)",
                mode="lines+markers",
                line=dict(color="#1B4F72", width=2, dash="dot"),
            )
        )

    if "NOI" in forecast_df.columns:
        fig.add_trace(
            go.Scatter(
                x=years,
                y=_to_numeric_series(forecast_df["NOI"]),
                name="Net Operating Income (NOI)",
                mode="lines+markers",
                line=dict(color="#1E8449", width=4),
                fill="tonexty" if "EGI" in forecast_df.columns else None,
                fillcolor="rgba(30, 132, 73, 0.10)",
            )
        )

    if "Levered_CF" in forecast_df.columns:
        fig.add_trace(
            go.Bar(
                x=years,
                y=_to_numeric_series(forecast_df["Levered_CF"]),
                name="Levered Cash Flow",
                marker_color="#85C1E9",
                opacity=0.75,
                yaxis="y2",
            )
        )

    debt_service = parse_float(debt_service, 0.0)
    if debt_service > 0:
        fig.add_hline(
            y=debt_service,
            line_dash="dash",
            line_color="#C0392B",
            annotation_text=f"Indicative Debt Service: {fmt_currency(debt_service, decimals=0)}",
            annotation_position="bottom right",
        )

    fig.update_layout(
        title="Cash Flow Dynamics & Debt Coverage",
        xaxis=dict(title="Hold Year", dtick=1, gridcolor="#F0F2F6"),
        yaxis=dict(
            title="Operating Totals ($)",
            side="left",
            tickformat="$.2s",
        ),
        yaxis2=dict(
            title="Net Cash Flow ($)",
            side="right",
            overlaying="y",
            showgrid=False,
            tickformat="$.2s",
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        hovermode="x unified",
    )
    return fig


def build_sensitivity_heatmap(matrix_df: pd.DataFrame) -> go.Figure | None:
    """
    Build an IRR sensitivity heatmap from a matrix DataFrame.

    Uses conservative generic axis labels unless explicit upstream metadata is
    available. This avoids displaying potentially incorrect financial labels.
    """
    matrix_df = ensure_dataframe(matrix_df)
    if matrix_df.empty:
        return None

    try:
        numeric_df = matrix_df.apply(pd.to_numeric, errors="coerce")
        z = numeric_df.to_numpy(dtype=float)

        if z.size == 0 or np.isnan(z).all():
            return None

        text_arr = np.vectorize(_format_sensitivity_cell)(z)
        x_labels = [_format_sensitivity_label(c) for c in numeric_df.columns]
        y_labels = [_format_sensitivity_label(i) for i in numeric_df.index]

        title, x_title, y_title = _infer_sensitivity_axis_titles(numeric_df)

        fig = go.Figure(
            data=go.Heatmap(
                z=z,
                x=x_labels,
                y=y_labels,
                colorscale="RdYlGn",
                text=text_arr,
                texttemplate="%{text}",
                hoverongaps=False,
                colorbar=dict(title="IRR"),
            )
        )
        fig.update_layout(
            title=title,
            xaxis_title=x_title,
            yaxis_title=y_title,
            plot_bgcolor="white",
            paper_bgcolor="white",
        )
        return fig

    except Exception:
        logger.exception("Failed to build sensitivity heatmap")
        return None


def build_value_bridge_chart(df: pd.DataFrame) -> go.Figure | None:
    """
    Build a value bridge waterfall chart.

    If a 'Measure' column exists, it will be used. Otherwise, the chart falls
    back to treating all components as relative changes.
    """
    df = ensure_dataframe(df)
    if df.empty or not {"Component", "Value"}.issubset(df.columns):
        return None

    clean_labels = df["Component"].astype(str).str.replace("_", " ", regex=False)
    numeric_values = _to_numeric_series(df["Value"])

    if "Measure" in df.columns:
        measure = df["Measure"].astype(str).str.lower().tolist()
    else:
        measure = ["relative"] * len(df)

    fig = go.Figure(
        go.Waterfall(
            name="Value Bridge",
            orientation="v",
            measure=measure,
            x=clean_labels,
            y=numeric_values,
            text=[_fmt_currency_signed(v) for v in numeric_values],
            textposition="outside",
            increasing={"marker": {"color": "#2ECC71"}},
            decreasing={"marker": {"color": "#E74C3C"}},
            totals={"marker": {"color": "#34495E"}},
        )
    )

    fig.update_layout(
        title="Value Bridge — Components of Value Change",
        xaxis_title="",
        yaxis_title="Value ($)",
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=False,
        margin=dict(b=80),
    )
    return fig


# =============================================================================
# UI RENDERING
# =============================================================================

def render_assumption_controls(base: dict[str, Any]) -> ForecastAssumptions:
    """
    Render sidebar assumption controls and return the selected assumptions.
    """
    base = ensure_dict(base)

    st.sidebar.header("Forecast Assumptions")

    rent_growth = st.sidebar.slider(
        "Rent Growth",
        min_value=0.00,
        max_value=0.08,
        value=_clamp_float(base.get("rent_growth"), 0.00, 0.08, 0.03),
        step=0.005,
        format="%.3f",
    )
    vacancy_rate = st.sidebar.slider(
        "Vacancy Rate",
        min_value=0.00,
        max_value=0.20,
        value=_clamp_float(base.get("vacancy_rate"), 0.00, 0.20, 0.05),
        step=0.005,
        format="%.3f",
    )
    expense_growth = st.sidebar.slider(
        "Expense Growth",
        min_value=0.00,
        max_value=0.08,
        value=_clamp_float(base.get("expense_growth"), 0.00, 0.08, 0.03),
        step=0.005,
        format="%.3f",
    )
    capex_per_unit = st.sidebar.slider(
        "CapEx Reserve per Unit",
        min_value=0,
        max_value=1000,
        value=_clamp_int(base.get("capex_per_unit"), 0, 1000, 300),
        step=25,
    )
    exit_cap_rate = st.sidebar.slider(
        "Exit Cap Rate",
        min_value=0.035,
        max_value=0.085,
        value=_clamp_float(base.get("exit_cap_rate"), 0.035, 0.085, 0.055),
        step=0.0025,
        format="%.4f",
    )
    hold_years = st.sidebar.slider(
        "Hold Period (Years)",
        min_value=3,
        max_value=10,
        value=_clamp_int(base.get("hold_years"), 3, 10, 5),
        step=1,
    )

    assumptions = ForecastAssumptions(
        rent_growth=rent_growth,
        vacancy_rate=vacancy_rate,
        expense_growth=expense_growth,
        capex_per_unit=capex_per_unit,
        exit_cap_rate=exit_cap_rate,
        hold_years=hold_years,
    )

    assumption_warnings = _validate_assumptions(assumptions)
    if assumption_warnings:
        st.sidebar.caption("Assumption checks")
        for warning in assumption_warnings:
            st.sidebar.warning(warning)

    return assumptions


def _render_chart_or_info(fig: go.Figure | None, message: str) -> None:
    """Render a Plotly chart or an info message."""
    if fig is not None:
        st.plotly_chart(fig, width="stretch")
    else:
        st.info(message)


def _render_df_or_info(df: pd.DataFrame, message: str) -> None:
    """Render a DataFrame or an info message."""
    df = ensure_dataframe(df)
    if not df.empty:
        st.dataframe(df, width="stretch", hide_index=True)
    else:
        st.info(message)


def render_kpi_row(forecast: dict[str, Any], valuation: dict[str, Any]) -> None:
    """Render the top KPI cards."""
    forecast = ensure_dict(forecast)
    valuation = ensure_dict(valuation)

    cols = st.columns(5)
    kpis = [
        ("Exit Value", fmt_currency(forecast.get("exit_value", 0.0))),
        ("Unlevered IRR", fmt_pct(forecast.get("unlevered_irr", 0.0))),
        ("Equity Multiple", fmt_multiple(forecast.get("equity_multiple", 0.0))),
        ("Concluded Value", fmt_currency(valuation.get("concluded_value", 0.0))),
        ("Implied Cap Rate", fmt_pct(valuation.get("concluded_cap_rate", 0.0))),
    ]

    for col, (label, value) in zip(cols, kpis):
        with col:
            st.metric(label, value)


def render_hold_sell_card(hold_sell: dict[str, Any]) -> None:
    """Render the hold/sell recommendation card safely."""
    hold_sell = ensure_dict(hold_sell)

    recommendation = safe_str(hold_sell.get("recommendation", "Watch"), "Watch")
    confidence = safe_str(hold_sell.get("confidence", "Moderate"), "Moderate")
    color = rag_color(recommendation)

    safe_rec = html.escape(recommendation)
    safe_conf = html.escape(confidence)

    st.subheader("Hold / Sell Recommendation")
    st.markdown(
        f"""
        <div style="border-left: 8px solid {color}; padding: 1rem;
                    background-color: #F8F9FA; border-radius: 0.35rem;">
            <strong>{safe_rec}</strong><br>
            Confidence: <strong>{safe_conf}</strong>
        </div>
        """,
        unsafe_allow_html=True,
    )

    rationale = hold_sell.get("rationale", [])
    if isinstance(rationale, list) and rationale:
        st.markdown("**Rationale**")
        for item in rationale:
            st.markdown(f"- {html.escape(safe_str(item))}")


def render_valuation_reconciliation(
    valuation: dict[str, Any],
    benchmark_check: dict[str, Any],
) -> None:
    """Render valuation reconciliation and benchmark summary."""
    valuation = ensure_dict(valuation)
    benchmark_check = ensure_dict(benchmark_check)

    st.subheader("Valuation Reconciliation")

    recon_df = _build_recon_df(valuation)
    display_df = recon_df.copy()
    display_df["Indicated_Value"] = display_df["Indicated_Value"].map(_fmt_currency_plain)
    display_df["Weight"] = display_df["Weight"].map(_fmt_pct_zero_dp)
    display_df["Weighted_Value"] = display_df["Weighted_Value"].map(_fmt_currency_plain)
    st.dataframe(display_df, width="stretch", hide_index=True)

    weight_sum = recon_df["Weight"].sum() if not recon_df.empty else 0.0
    if abs(weight_sum - 1.0) > 0.05:
        st.caption(f"Note: valuation method weights sum to {fmt_pct(weight_sum, decimals=0)} rather than 100%.")

    rag_status = safe_str(benchmark_check.get("RAG_Status", "N/A"))
    benchmark_ppu = fmt_currency(benchmark_check.get("benchmark_ppu", 0.0))
    variance_pct = fmt_pct(benchmark_check.get("variance_pct", 0.0), include_sign=True)

    st.caption(
        f"Benchmark check: {html.escape(rag_status)} | "
        f"Benchmark PPU {benchmark_ppu} | "
        f"Variance {variance_pct}"
    )


def render_cash_flow_section(
    forecast_df: pd.DataFrame,
    debt_service: float = 0.0,
) -> None:
    """Render forecast table and chart."""
    forecast_df = _sort_forecast_df(ensure_dataframe(forecast_df))

    col_table, col_chart = st.columns((1.4, 1.0))

    with col_table:
        st.subheader("Cash Flow Projection")
        if forecast_df.empty:
            st.info("No forecast table available.")
        else:
            display_df = forecast_df.copy()
            for col in display_df.columns:
                display_df[col] = _format_forecast_column(col, display_df[col])
            st.dataframe(display_df, width="stretch", hide_index=True)

    with col_chart:
        fig = build_forecast_chart(forecast_df, debt_service=debt_service)
        _render_chart_or_info(fig, "No forecast chart available.")
        if debt_service > 0:
            st.caption("Debt service overlay is indicative and based on current financing assumptions, not necessarily a fully modeled future debt structure.")


def render_sensitivity_section(bundle: dict[str, Any]) -> None:
    """Render sensitivity heatmap and matrix table."""
    bundle = _coerce_bundle_types(bundle)

    st.subheader("IRR Sensitivity Heatmap")
    sensitivity_matrix = ensure_dataframe(bundle.get("sensitivity_matrix"))

    fig = build_sensitivity_heatmap(sensitivity_matrix)
    _render_chart_or_info(fig, "No IRR sensitivity heatmap available.")

    with st.expander("View Sensitivity Matrix Table"):
        try:
            display = ensure_dataframe(get_sensitivity_matrix_display(sensitivity_matrix))
            _render_df_or_info(display, "No sensitivity matrix table available.")
        except Exception:
            logger.exception("Failed to render sensitivity matrix table")
            st.info("Unable to display sensitivity matrix table.")


def render_scenario_valuation(scenario_valuation: dict[str, Any]) -> None:
    """Render Bear / Base / Bull scenario valuation."""
    scenario_valuation = ensure_dict(scenario_valuation)

    st.subheader("Bear / Base / Bull Scenario Valuation")
    scenario_table = ensure_dataframe(scenario_valuation.get("summary_table"))
    if scenario_table.empty:
        st.info("No scenario valuation available.")
        return

    display_df = scenario_table.copy()

    if "Scenario" in display_df.columns:
        scenario_order = {"Bear": 0, "Base": 1, "Bull": 2}
        display_df = display_df.assign(
            _scenario_order=display_df["Scenario"].astype(str).map(scenario_order).fillna(999)
        ).sort_values("_scenario_order").drop(columns=["_scenario_order"])

    for col in SCENARIO_CURRENCY_COLS:
        if col in display_df.columns:
            formatter = _fmt_currency_signed if col == "Unrealized_Gain" else _fmt_currency_plain
            display_df[col] = _to_numeric_series(display_df[col]).map(formatter)

    for col in SCENARIO_PCT_COLS:
        if col in display_df.columns:
            display_df[col] = _to_numeric_series(display_df[col]).map(_fmt_pct_plain)

    st.dataframe(display_df, width="stretch", hide_index=True)


def render_value_bridge_section(bundle: dict[str, Any]) -> None:
    """Render value bridge and underwriting comparison."""
    bundle = _coerce_bundle_types(bundle)

    col_bridge, col_uw = st.columns((1.25, 1.0))

    with col_bridge:
        st.subheader("Value Bridge")
        value_bridge_df = ensure_dataframe(bundle.get("value_bridge_df"))
        fig = build_value_bridge_chart(value_bridge_df)
        _render_chart_or_info(fig, "No value bridge available.")

    with col_uw:
        st.subheader("Underwriting vs Actual / Forecast")
        compare_dict = ensure_dict(bundle.get("forecast_vs_uw"))

        if not compare_dict:
            st.info("No underwriting comparison available.")
        else:
            ordered_items = list(compare_dict.items())
            display_df = pd.DataFrame({
                "Metric": [_clean_metric_label(metric) for metric, _ in ordered_items],
                "Value": [format_metric_value(metric, value) for metric, value in ordered_items],
            })
            st.dataframe(display_df, width="stretch", hide_index=True)


def render_analyst_interpretation(
    property_name: str,
    forecast: dict[str, Any],
    valuation: dict[str, Any],
    hold_sell: dict[str, Any],
    benchmark_check: dict[str, Any],
) -> None:
    """
    Render a robust, readable analyst interpretation using safe HTML.

    This avoids markdown corruption from dynamic values and adds actual
    interpretation beyond simple metric restatement.
    """
    forecast = ensure_dict(forecast)
    valuation = ensure_dict(valuation)
    hold_sell = ensure_dict(hold_sell)
    benchmark_check = ensure_dict(benchmark_check)

    st.subheader("Analyst Interpretation")

    exit_value = parse_float(forecast.get("exit_value", 0.0), 0.0)
    unlevered_irr = parse_float(forecast.get("unlevered_irr", 0.0), 0.0)
    equity_multiple = parse_float(forecast.get("equity_multiple", 0.0), 0.0)
    concluded_value = parse_float(valuation.get("concluded_value", 0.0), 0.0)

    recommendation = safe_str(hold_sell.get("recommendation", "Watch"), "Watch")
    confidence = safe_str(hold_sell.get("confidence", "Moderate"), "Moderate")
    rag_status = safe_str(benchmark_check.get("RAG_Status", "N/A"), "N/A")
    benchmark_variance = parse_float(benchmark_check.get("variance_pct", 0.0), 0.0)

    property_label = safe_str(property_name, "Selected Property").replace("_", " ")

    exit_text = fmt_currency(exit_value, decimals=0)
    irr_text = fmt_pct(unlevered_irr, decimals=1)
    multiple_text = fmt_multiple(equity_multiple)
    value_text = fmt_currency(concluded_value, decimals=0)
    variance_text = fmt_pct(benchmark_variance, decimals=1, include_sign=True)

    if unlevered_irr >= 0.18:
        return_view = "strong projected returns"
    elif unlevered_irr >= 0.12:
        return_view = "moderate projected returns"
    else:
        return_view = "relatively modest projected returns"

    if benchmark_variance > 0.05:
        valuation_view = f"value appears above benchmark by {variance_text}"
    elif benchmark_variance < -0.05:
        valuation_view = f"value appears below benchmark by {variance_text}"
    else:
        valuation_view = f"value appears broadly in line with benchmark at {variance_text}"

    recommendation_view = {
        "Sell": "This typically suggests crystallizing value may currently be more attractive than continuing the hold.",
        "Hold": "This typically suggests current ownership remains defensible under the modeled assumptions.",
        "Refinance": "This typically suggests capital restructuring may improve investor outcomes while preserving upside.",
        "Watch": "This suggests a more balanced setup where near-term monitoring remains important.",
    }.get(recommendation, "This suggests the opportunity should be interpreted with caution and additional review.")

    safe_name = _safe_html_text(property_label, "Selected Property")
    safe_exit = _safe_html_text(exit_text)
    safe_irr = _safe_html_text(irr_text)
    safe_multiple = _safe_html_text(multiple_text)
    safe_value = _safe_html_text(value_text)
    safe_rec = _safe_html_text(recommendation)
    safe_conf = _safe_html_text(confidence)
    safe_rag = _safe_html_text(rag_status)
    safe_return_view = _safe_html_text(return_view)
    safe_valuation_view = _safe_html_text(valuation_view)
    safe_rec_view = _safe_html_text(recommendation_view)

    st.markdown(
        f"""
        <div style="padding: 1rem 1.1rem; background-color: #F8F9FA; border-radius: 0.5rem; line-height: 1.65;">
            <p style="margin-top: 0;">
                Using the selected assumptions for <strong>{safe_name}</strong>, the model indicates an
                exit value of <strong>{safe_exit}</strong>, an unlevered IRR of <strong>{safe_irr}</strong>,
                an equity multiple of <strong>{safe_multiple}</strong>, and a concluded blended valuation of
                <strong>{safe_value}</strong>.
            </p>
            <p>
                Overall, this implies <strong>{safe_return_view}</strong>. Relative to benchmark pricing,
                <strong>{safe_valuation_view}</strong>, with the current benchmark status shown as
                <strong>{safe_rag}</strong>.
            </p>
            <p style="margin-bottom: 0;">
                The current recommendation is <strong>{safe_rec}</strong> with
                <strong>{safe_conf}</strong> confidence. {safe_rec_view}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    """Main entry point for the Forecast & Valuation page."""
    st.title("📈 Forecast and Valuation")
    st.caption(
        "Interactive financial modeling workbench for forecasting property "
        "performance, testing assumptions, evaluating hold/sell outcomes, "
        "and reconciling value."
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
    base = get_base_assumptions(property_id)
    assumptions = render_assumption_controls(base)

    try:
        base_price = max(parse_float(base.get("purchase_price", 10_000_000), 10_000_000), 0.0)
        current_ltv = min(max(parse_float(base.get("ltv", 0.70), 0.70), 0.0), 1.0)
        current_rate = max(parse_float(base.get("interest_rate", 0.06), 0.06), 0.0)

        annual_debt = parse_float(
            calculate_annual_debt_service(
                purchase_price=base_price,
                ltv=current_ltv,
                annual_rate=current_rate,
            ),
            0.0,
        )
    except Exception:
        logger.exception("Failed to calculate annual debt service for property_id=%s", property_id)
        annual_debt = 0.0

    try:
        bundle = _coerce_bundle_types(load_forecast_bundle(property_id, assumptions))
        render_warnings(bundle.get("warnings", []))

        forecast = ensure_dict(bundle.get("forecast"))
        forecast_df = ensure_dataframe(bundle.get("forecast_df"))
        hold_sell = ensure_dict(bundle.get("hold_sell"))
        valuation = ensure_dict(bundle.get("valuation"))
        benchmark_check = ensure_dict(bundle.get("benchmark_check"))
        scenario_valuation = ensure_dict(bundle.get("scenario_valuation"))

        st.markdown("---")
        render_kpi_row(forecast, valuation)
        st.markdown("---")

        col_hold, col_recon = st.columns((1.1, 1.2))
        with col_hold:
            render_hold_sell_card(hold_sell)
        with col_recon:
            render_valuation_reconciliation(valuation, benchmark_check)
        st.markdown("---")

        render_cash_flow_section(forecast_df, debt_service=annual_debt)
        st.markdown("---")

        col_sensitivity, col_scenario = st.columns((1.25, 1.0))
        with col_sensitivity:
            render_sensitivity_section(bundle)
        with col_scenario:
            render_scenario_valuation(scenario_valuation)
        st.markdown("---")

        render_value_bridge_section(bundle)
        st.markdown("---")

        render_analyst_interpretation(
            property_name=property_name,
            forecast=forecast,
            valuation=valuation,
            hold_sell=hold_sell,
            benchmark_check=benchmark_check,
        )

    except Exception as exc:
        logger.exception("Forecast and Valuation page failed for property_id=%s", property_id)
        st.error("⚠️ Page Error: Unable to load the Forecast & Valuation page.")
        st.exception(exc)


if __name__ == "__main__":
    main()