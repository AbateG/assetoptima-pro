"""
pages/5_Business_Plan_Tracker.py

Business Plan Tracker page for AssetOptima Pro.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from modules.business_plan_tracker import (
    calculate_renovation_roi,
    get_budget_vs_actual_summary,
    get_delayed_initiatives,
    get_gantt_timeline_data,
    get_initiative_progress_table,
    get_portfolio_business_plan_kpis,
    get_property_business_plan_summary,
    get_property_initiative_ranking,
)
from modules.data_loader import get_property_display_options, parse_property_selection
from utils.coercion import ensure_dataframe, ensure_dict, parse_float, parse_int, safe_str
from utils.formatters import fmt_currency, fmt_pct
from utils.ui_helpers import extract_property_name, rag_color, render_warnings
from utils.validators import has_columns

# ---------------------------------------------------------------------------
# Logger — no basicConfig here; let the host application configure the root
# logger.  This prevents clobbering multi-page app log settings.
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Page config — must be the first Streamlit call in the module
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Business Plan Tracker | AssetOptima Pro",
    page_icon="🛠️",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_COLORS: dict[str, str] = {
    "budget_bar": "#85C1E9",
    "actual_bar": "#1B4F72",
    "rag_green": "#1E8449",
    "rag_amber": "#D68910",
    "rag_red": "#C0392B",
    "card_bg": "#F8F9FA",
}

_RAG_COLOR_MAP: dict[str, str] = {
    "Green": _COLORS["rag_green"],
    "Amber": _COLORS["rag_amber"],
    "Red": _COLORS["rag_red"],
}

_TTL_SECONDS: int = 300  # 5-minute cache TTL for all data loaders


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner=False, ttl=_TTL_SECONDS)
def get_property_options() -> list[str]:
    """Load property selector options."""
    try:
        options = get_property_display_options()
        return options if isinstance(options, list) else []
    except Exception:
        logger.exception("Failed to load property options")
        return []


@st.cache_data(show_spinner=False, ttl=_TTL_SECONDS)
def load_portfolio_bundle() -> dict[str, Any]:
    """
    Load portfolio-level business plan outputs with partial-failure tolerance.

    Each loader failure is caught independently so that a single broken
    data source does not blank the entire page.
    """
    bundle: dict[str, Any] = {
        "portfolio_kpis": {},
        "budget_summary": pd.DataFrame(),
        "delayed_all": pd.DataFrame(),
        "ranking": pd.DataFrame(),
        "warnings": [],
    }

    loaders: dict[str, Any] = {
        "portfolio_kpis": get_portfolio_business_plan_kpis,
        "budget_summary": get_budget_vs_actual_summary,
        "delayed_all": get_delayed_initiatives,
        "ranking": get_property_initiative_ranking,
    }

    for key, loader in loaders.items():
        try:
            bundle[key] = loader()
        except Exception:
            logger.exception("Failed loading portfolio business plan component: %s", key)
            bundle["warnings"].append(f"'{key}' data failed to load — partial results shown.")

    # Coerce once here; callers receive clean types with no double-coercion.
    bundle["portfolio_kpis"] = ensure_dict(bundle["portfolio_kpis"])
    bundle["budget_summary"] = ensure_dataframe(bundle["budget_summary"])
    bundle["delayed_all"] = ensure_dataframe(bundle["delayed_all"])
    bundle["ranking"] = ensure_dataframe(bundle["ranking"])
    return bundle


@st.cache_data(show_spinner=False, ttl=_TTL_SECONDS)
def load_property_bundle(property_id: str) -> dict[str, Any]:
    """
    Load property-level business plan outputs with partial-failure tolerance.

    Keyed on ``property_id`` so each property is cached independently.
    """
    bundle: dict[str, Any] = {
        "summary": {},
        "progress_table": pd.DataFrame(),
        "budget_summary": pd.DataFrame(),
        "timeline": pd.DataFrame(),
        "roi": {},
        "delayed": pd.DataFrame(),
        "warnings": [],
    }

    loaders: dict[str, Any] = {
        "summary": lambda: get_property_business_plan_summary(property_id),
        "progress_table": lambda: get_initiative_progress_table(property_id),
        "budget_summary": lambda: get_budget_vs_actual_summary(property_id),
        "timeline": lambda: get_gantt_timeline_data(property_id),
        "roi": lambda: calculate_renovation_roi(property_id),
        "delayed": lambda: get_delayed_initiatives(property_id),
    }

    for key, loader in loaders.items():
        try:
            bundle[key] = loader()
        except Exception:
            logger.exception(
                "Failed loading property business plan component: %s (property_id=%s)",
                key,
                property_id,
            )
            bundle["warnings"].append(f"'{key}' data failed to load — partial results shown.")

    bundle["summary"] = ensure_dict(bundle["summary"])
    bundle["progress_table"] = ensure_dataframe(bundle["progress_table"])
    bundle["budget_summary"] = ensure_dataframe(bundle["budget_summary"])
    bundle["timeline"] = ensure_dataframe(bundle["timeline"])
    bundle["roi"] = ensure_dict(bundle["roi"])
    bundle["delayed"] = ensure_dataframe(bundle["delayed"])
    return bundle


# ---------------------------------------------------------------------------
# Chart builders
# ---------------------------------------------------------------------------


def build_budget_vs_actual_chart(df: pd.DataFrame) -> Optional[go.Figure]:
    """Return a grouped bar chart of budget vs actual spend, or ``None``."""
    df = ensure_dataframe(df)
    x_col = "Property_Name" if "Property_Name" in df.columns else "Property_ID"

    if df.empty or not has_columns(df, [x_col, "Total_Budget", "Total_Actual_Spend"]):
        return None

    chart_df = df.copy()
    chart_df["Total_Budget"] = pd.to_numeric(chart_df["Total_Budget"], errors="coerce")
    chart_df["Total_Actual_Spend"] = pd.to_numeric(
        chart_df["Total_Actual_Spend"], errors="coerce"
    )

    fig = go.Figure(
        data=[
            go.Bar(
                x=chart_df[x_col],
                y=chart_df["Total_Budget"],
                name="Budget",
                marker_color=_COLORS["budget_bar"],
            ),
            go.Bar(
                x=chart_df[x_col],
                y=chart_df["Total_Actual_Spend"],
                name="Actual Spend",
                marker_color=_COLORS["actual_bar"],
            ),
        ]
    )
    fig.update_layout(
        title="Budget vs Actual Spend",
        xaxis_title="Property",
        yaxis_title="Amount ($)",
        barmode="group",
        hovermode="x unified",
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    return fig


def build_gantt_chart(df: pd.DataFrame) -> Optional[go.Figure]:
    """Return a Gantt-style initiative timeline figure, or ``None``."""
    df = ensure_dataframe(df)
    required_cols = ["Display_Start", "Display_End", "Initiative"]
    if df.empty or not has_columns(df, required_cols):
        return None

    chart_df = df.copy()
    has_rag = "Timeline_RAG" in chart_df.columns

    # Build kwargs conditionally so we never pass None to Plotly.
    px_kwargs: dict[str, Any] = {
        "x_start": "Display_Start",
        "x_end": "Display_End",
        "y": "Initiative",
        "title": "Initiative Timeline",
    }
    if has_rag:
        px_kwargs["color"] = "Timeline_RAG"
        px_kwargs["color_discrete_map"] = _RAG_COLOR_MAP

    fig = px.timeline(chart_df, **px_kwargs)
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(
        xaxis_title="Timeline",
        yaxis_title="Initiative",
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    return fig


# ---------------------------------------------------------------------------
# Display builders
# ---------------------------------------------------------------------------


def _normalize_completion(raw: Any) -> float:
    """
    Normalise a completion value to the [0.0, 1.0] range.

    Accepts percentages expressed as 0–100 or 0–1 fractions.
    """
    value = parse_float(raw)  # returns 0.0 on None/NaN
    if value is None:
        return 0.0
    # Values > 1.0 are assumed to be expressed as a 0–100 percentage.
    normalised = value / 100.0 if value > 1.0 else value
    return max(0.0, min(1.0, normalised))


def _fmt_pct_safe(value: Any, *, include_sign: bool = False) -> str:
    """Return a formatted percentage string, guarding against None/NaN."""
    return fmt_pct(parse_float(value) or 0.0, include_sign=include_sign)


def build_progress_cards(progress_df: pd.DataFrame) -> None:
    """Render one styled progress card per initiative row."""
    progress_df = ensure_dataframe(progress_df)
    if progress_df.empty:
        st.info("No initiative progress data available.")
        return

    for _, row in progress_df.iterrows():
        title = safe_str(row.get("Initiative"), "Initiative")
        status = safe_str(row.get("Status"), "")
        completion = _normalize_completion(row.get("Percent_Complete"))
        progress_rag = safe_str(row.get("Progress_RAG"), "Green")
        color = rag_color(progress_rag)

        budget = parse_float(row.get("Budget")) or 0.0
        actual = parse_float(row.get("Actual_Spend")) or 0.0
        roc = parse_float(row.get("Return_on_Cost_Calc")) or 0.0

        st.markdown(
            f"""
            <div style="
                border-left: 6px solid {color};
                padding: 0.75rem 1rem;
                margin-bottom: 0.75rem;
                background-color: {_COLORS['card_bg']};
                border-radius: 0.35rem;
            ">
                <strong>{title}</strong><br>
                Status: {status}<br>
                Budget: {fmt_currency(budget)} &nbsp;|&nbsp;
                Actual: {fmt_currency(actual)} &nbsp;|&nbsp;
                ROC: {fmt_pct(roc)}
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.progress(completion, text=f"Completion: {completion:.0%}")


def _apply_currency_columns(
    df: pd.DataFrame,
    cols: list[str],
    *,
    sign_col: str = "Budget_Variance",
) -> pd.DataFrame:
    """Apply ``fmt_currency`` to named columns in-place and return ``df``."""
    for col in cols:
        if col in df.columns:
            include_sign = col == sign_col
            # Use a default-argument capture to avoid late-binding closure bug.
            df[col] = df[col].map(
                lambda x, _sign=include_sign: fmt_currency(x, include_sign=_sign)
            )
    return df


def build_delayed_display(df: pd.DataFrame) -> pd.DataFrame:
    """Return a display-formatted copy of the delayed-initiatives table."""
    df = ensure_dataframe(df)
    if df.empty:
        return df

    display_df = df.copy()
    display_df = _apply_currency_columns(
        display_df,
        ["Budget", "Actual_Spend", "Budget_Variance", "Expected_NOI_Lift"],
    )

    if "Percent_Complete" in display_df.columns:
        display_df["Percent_Complete"] = display_df["Percent_Complete"].map(
            lambda x: fmt_pct(_normalize_completion(x))
        )

    return display_df


def build_progress_detail_display(df: pd.DataFrame) -> pd.DataFrame:
    """Return a display-formatted copy of the initiative-detail table."""
    df = ensure_dataframe(df)
    if df.empty:
        return df

    display_df = df.copy()
    display_df = _apply_currency_columns(
        display_df,
        ["Budget", "Actual_Spend", "Budget_Variance", "Expected_NOI_Lift"],
    )

    for col in ["Budget_Var_Pct", "Return_on_Cost_Calc", "Percent_Complete"]:
        if col not in display_df.columns:
            continue
        if col == "Percent_Complete":
            display_df[col] = display_df[col].map(
                lambda x: fmt_pct(_normalize_completion(x))
            )
        else:
            # include_sign captured by default arg to avoid closure bug
            display_df[col] = display_df[col].map(
                lambda x, _s=True: fmt_pct(parse_float(x) or 0.0, include_sign=_s)
            )

    return display_df


def build_ranking_display(df: pd.DataFrame) -> pd.DataFrame:
    """Return a display-formatted copy of the portfolio ranking table."""
    df = ensure_dataframe(df)
    if df.empty:
        return df

    display_df = df.copy()

    for col in ["Expected_NOI_Lift", "Implied_Value_Creation"]:
        if col in display_df.columns:
            display_df[col] = display_df[col].map(fmt_currency)

    for col in ["Weighted_Completion_Pct", "Budget_Variance_Pct", "Average_Return_on_Cost"]:
        if col in display_df.columns:
            display_df[col] = display_df[col].map(
                lambda x: fmt_pct(parse_float(x) or 0.0, include_sign=True)
            )

    if "Execution_Score" in display_df.columns:
        display_df["Execution_Score"] = (
            pd.to_numeric(display_df["Execution_Score"], errors="coerce").round(1)
        )

    return display_df


# ---------------------------------------------------------------------------
# Section renderers — decomposed from main() for readability & testability
# ---------------------------------------------------------------------------


def _render_portfolio_kpis(kpis: dict[str, Any]) -> None:
    """Render the six primary portfolio KPI metrics."""
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    with k1:
        st.metric("Initiatives", f"{parse_int(kpis.get('Initiative_Count', 0)):,}")
    with k2:
        st.metric("Completed", f"{parse_int(kpis.get('Completed_Count', 0)):,}")
    with k3:
        st.metric("In Progress", f"{parse_int(kpis.get('In_Progress_Count', 0)):,}")
    with k4:
        st.metric("Delayed", f"{parse_int(kpis.get('Delayed_Count', 0)):,}")
    with k5:
        st.metric("Total Budget", fmt_currency(kpis.get("Total_Budget", 0.0)))
    with k6:
        st.metric("Expected NOI Lift", fmt_currency(kpis.get("Expected_NOI_Lift", 0.0)))

    s1, s2, s3 = st.columns(3)
    with s1:
        st.metric("Actual Spend", fmt_currency(kpis.get("Total_Actual_Spend", 0.0)))
    with s2:
        st.metric(
            "Budget Variance",
            _fmt_pct_safe(kpis.get("Budget_Variance_Pct"), include_sign=True),
        )
    with s3:
        st.metric(
            "Weighted Completion",
            _fmt_pct_safe(kpis.get("Weighted_Completion_Pct")),
        )


def _render_portfolio_section(
    kpis: dict[str, Any],
    budget_summary: pd.DataFrame,
    delayed: pd.DataFrame,
) -> None:
    """Render the full portfolio-level section."""
    st.markdown("---")
    _render_portfolio_kpis(kpis)
    st.markdown("---")

    col_chart, col_delayed = st.columns((1.2, 1.0))

    with col_chart:
        fig = build_budget_vs_actual_chart(budget_summary)
        if fig is None:
            st.info("No portfolio budget vs actual data available.")
        else:
            st.plotly_chart(fig, use_container_width=True)

    with col_delayed:
        st.subheader("Portfolio Delayed Initiative Alerts")
        if delayed.empty:
            st.success("No delayed initiatives currently identified.")
        else:
            st.dataframe(
                build_delayed_display(delayed),
                use_container_width=True,
                hide_index=True,
            )


def _render_property_kpis(
    summary: dict[str, Any],
    roi: dict[str, Any],
) -> None:
    """Render property-level KPI and ROI metrics."""
    p1, p2, p3, p4, p5 = st.columns(5)
    with p1:
        st.metric("Initiatives", f"{parse_int(summary.get('Initiative_Count', 0)):,}")
    with p2:
        st.metric("Delayed", f"{parse_int(summary.get('Delayed_Count', 0)):,}")
    with p3:
        st.metric(
            "Weighted Completion",
            _fmt_pct_safe(summary.get("Weighted_Completion_Pct")),
        )
    with p4:
        st.metric(
            "Budget Variance",
            _fmt_pct_safe(summary.get("Budget_Variance_Pct"), include_sign=True),
        )
    with p5:
        st.metric("Overall Status", safe_str(summary.get("Overall_RAG"), "Green"))

    r1, r2, r3, r4 = st.columns(4)
    with r1:
        st.metric("Spend / Unit", fmt_currency(roi.get("Spend_Per_Unit", 0.0)))
    with r2:
        st.metric("NOI Lift / Unit", fmt_currency(roi.get("NOI_Lift_Per_Unit", 0.0)))
    with r3:
        st.metric(
            "Return on Cost",
            _fmt_pct_safe(roi.get("Return_on_Cost"), include_sign=True),
        )
    with r4:
        st.metric(
            "Implied Value Creation",
            fmt_currency(roi.get("Implied_Value_Creation", 0.0)),
        )


def _render_property_section(
    property_name: str,
    bundle: dict[str, Any],
    portfolio_ranking: pd.DataFrame,
) -> None:
    """Render the full property-level section."""
    summary = bundle["summary"]
    progress_table = bundle["progress_table"]
    timeline_df = bundle["timeline"]
    roi = bundle["roi"]
    property_delayed = bundle["delayed"]

    st.subheader(f"Property Execution Summary — {property_name}")
    _render_property_kpis(summary, roi)

    st.markdown("---")

    col_progress, col_gantt = st.columns((1.0, 1.25))

    with col_progress:
        st.subheader("Initiative Progress")
        build_progress_cards(progress_table)

    with col_gantt:
        st.subheader("Gantt Timeline")
        fig = build_gantt_chart(timeline_df)
        if fig is None:
            st.info("No timeline data available.")
        else:
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["Initiative Detail", "Delayed Alerts", "Property Ranking"])

    with tab1:
        st.subheader("Initiative Detail Table")
        if progress_table.empty:
            st.info("No initiative detail available.")
        else:
            st.dataframe(
                build_progress_detail_display(progress_table),
                use_container_width=True,
                hide_index=True,
            )

    with tab2:
        st.subheader("Delayed Initiative Alerts")
        if property_delayed.empty:
            st.success("No delayed initiatives for this property.")
        else:
            st.dataframe(
                build_delayed_display(property_delayed),
                use_container_width=True,
                hide_index=True,
            )

    with tab3:
        st.subheader("Portfolio Execution Ranking")
        if portfolio_ranking.empty:
            st.info("No ranking data available.")
        else:
            st.dataframe(
                build_ranking_display(portfolio_ranking),
                use_container_width=True,
                hide_index=True,
            )

    st.markdown("---")

    initiative_count = parse_int(summary.get("Initiative_Count", 0))
    st.subheader("Analyst Takeaway")
    st.write(
        f"For **{property_name}**, the current business plan reflects "
        f"**{initiative_count:,} initiative{'s' if initiative_count != 1 else ''}**, "
        f"with weighted completion of "
        f"**{_fmt_pct_safe(summary.get('Weighted_Completion_Pct'))}** and "
        f"budget variance of "
        f"**{_fmt_pct_safe(summary.get('Budget_Variance_Pct'), include_sign=True)}**. "
        f"Projected value creation based on expected NOI lift is approximately "
        f"**{fmt_currency(roi.get('Implied_Value_Creation', 0.0))}**."
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Render the Business Plan Tracker page."""
    st.title("🛠️ Business Plan Tracker")
    st.caption(
        "Track value-add execution across renovation, leasing, and operational initiatives "
        "with progress, spend, schedule, and ROI visibility."
    )

    # ── Portfolio section ────────────────────────────────────────────────────
    try:
        portfolio_bundle = load_portfolio_bundle()
    except Exception:
        logger.exception("Fatal error loading portfolio bundle")
        st.error("⚠️ Unable to load portfolio data. Please refresh or contact support.")
        st.stop()

    render_warnings(portfolio_bundle.get("warnings", []))

    _render_portfolio_section(
        kpis=portfolio_bundle["portfolio_kpis"],
        budget_summary=portfolio_bundle["budget_summary"],
        delayed=portfolio_bundle["delayed_all"],
    )

    # ── Property selector ────────────────────────────────────────────────────
    property_options = get_property_options()
    if not property_options:
        st.error("No property options are available.")
        st.stop()

    selected_property_display = st.selectbox(
        "Select Property",
        options=property_options,
        index=0,
    )

    property_id = parse_property_selection(selected_property_display)
    if not property_id:
        st.error("Unable to parse the selected property. Please choose a valid option.")
        st.stop()

    property_name = extract_property_name(selected_property_display, property_id)

    # ── Property section ─────────────────────────────────────────────────────
    with st.spinner(f"Loading data for {property_name}…"):
        try:
            property_bundle = load_property_bundle(property_id)
        except Exception:
            logger.exception(
                "Fatal error loading property bundle (property_id=%s)", property_id
            )
            st.error(
                f"⚠️ Unable to load data for **{property_name}**. "
                "Please refresh or select a different property."
            )
            st.stop()

    render_warnings(property_bundle.get("warnings", []))

    _render_property_section(
        property_name=property_name,
        bundle=property_bundle,
        portfolio_ranking=portfolio_bundle["ranking"],
    )


if __name__ == "__main__":
    main()