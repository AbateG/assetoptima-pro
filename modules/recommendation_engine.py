from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import streamlit as st

from modules.business_plan_tracker import (
    calculate_renovation_roi,
    get_delayed_initiatives,
    get_property_business_plan_summary,
)
from modules.data_loader import get_property_list, get_property_name_map, load_properties
from modules.debt_compliance import get_property_compliance_summary
from modules.forecasting import (
    build_property_forecast,
    get_base_assumptions_for_property,
    get_hold_sell_recommendation,
)
from modules.market_analysis import (
    generate_market_narrative,
    get_competitive_positioning_score,
    get_market_watchlist_flags,
    get_subject_vs_comp_summary,
)
from modules.valuation import build_valuation_reconciliation
from modules.variance_analysis import (
    build_variance_summary_table,
    compare_actual_vs_underwriting,
    get_consecutive_variance_months,
    get_noi_trend_direction,
    get_t12_summary,
)


# ============================================================================
# ENUMS / CONSTANTS
# ============================================================================

class Priority(str, Enum):
    CRITICAL = "Critical"
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"


class Category(str, Enum):
    PERFORMANCE = "Performance"
    CAPITAL = "Capital"
    MARKET = "Market"
    EXECUTION = "Execution"
    COMPLIANCE = "Compliance"
    STRATEGY = "Strategy"


class WatchlistBucket(str, Enum):
    RED = "Red"
    AMBER = "Amber"
    GREEN = "Green"


class ComplianceLabel(str, Enum):
    BREACH = "Breach"
    WATCHLIST = "Watchlist"
    REFINANCE_CANDIDATE = "Refinance Candidate"
    COMPLIANT = "Compliant"
    UNKNOWN = "Unknown"


class TrendDirection(str, Enum):
    DECLINING = "declining"
    FLAT = "flat"
    IMPROVING = "improving"
    UNKNOWN = "unknown"


class StrategicPosture(str, Enum):
    HOLD = "Hold"
    WATCH = "Watch"
    SELL = "Sell"
    REFINANCE = "Refinance"
    UNKNOWN = "Unknown"


WATCHLIST_RED_THRESHOLD = 75.0
WATCHLIST_AMBER_THRESHOLD = 45.0
MAX_RECOMMENDATIONS_PER_PROPERTY = 6

WEIGHT_COMPLIANCE = 30.0
WEIGHT_PERFORMANCE = 20.0
WEIGHT_FORECAST = 15.0
WEIGHT_VALUATION = 10.0
WEIGHT_MARKET = 10.0
WEIGHT_EXECUTION = 15.0

_EXPECTED_WEIGHT_SUM = 100.0
_WEIGHT_TOLERANCE = 1e-9


# ============================================================================
# DOMAIN MODELS
# ============================================================================

@dataclass(frozen=True)
class PropertyContext:
    property_id: str
    property_name: str


@dataclass(frozen=True)
class PerformanceInput:
    has_red_variance: bool
    has_amber_variance: bool
    worst_variance_line_item: str | None
    revenue_underperformance_streak_months: int
    noi_trend: TrendDirection
    noi_variance_pct: float  # decimal form, e.g. -0.08


@dataclass(frozen=True)
class ForecastInput:
    strategic_posture: StrategicPosture


@dataclass(frozen=True)
class ValuationInput:
    unrealized_gain_pct: float  # decimal form
    outperforming_thesis: bool
    noi_gap_pct: float | None


@dataclass(frozen=True)
class MarketInput:
    rent_premium_pct: float
    occupancy_gap: float
    positioning_score: float
    positioning_rag: str
    flags: tuple[str, ...]


@dataclass(frozen=True)
class ExecutionInput:
    delayed_count: int
    budget_variance_pct: float  # decimal form
    total_actual_spend: float
    renovation_return_on_cost: float | None
    top_delayed_initiative: str | None


@dataclass(frozen=True)
class ComplianceInput:
    compliance_label: ComplianceLabel
    dscr_actual: float | None
    dscr_headroom: float | None
    days_to_maturity: int | None
    days_to_reporting: int | None
    overall_rag: str


@dataclass(frozen=True)
class RecommendationSnapshot:
    context: PropertyContext
    performance: PerformanceInput
    forecast: ForecastInput
    valuation: ValuationInput
    market: MarketInput
    execution: ExecutionInput
    compliance: ComplianceInput


@dataclass(frozen=True)
class RecommendationRecord:
    property_id: str
    property_name: str
    category: Category
    priority: Priority
    title: str
    recommendation: str
    rationale: str
    metric_value: Any


@dataclass(frozen=True)
class WatchlistScore:
    property_id: str
    property_name: str
    watchlist_score: float
    watchlist_bucket: WatchlistBucket
    compliance_score: float
    performance_score: float
    forecast_score: float
    market_score: float
    execution_score: float
    valuation_score: float
    valuation_signal: float


# ============================================================================
# LOW-LEVEL HELPERS
# ============================================================================

def _assert_weight_configuration() -> None:
    total = (
        WEIGHT_COMPLIANCE
        + WEIGHT_PERFORMANCE
        + WEIGHT_FORECAST
        + WEIGHT_VALUATION
        + WEIGHT_MARKET
        + WEIGHT_EXECUTION
    )
    if abs(total - _EXPECTED_WEIGHT_SUM) > _WEIGHT_TOLERANCE:
        raise ValueError(f"Composite watchlist weights must sum to 100.0; got {total}")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if pd.isna(value):
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _normalize_percent_like(value: Any, default: float = 0.0) -> float:
    """
    Normalize percent-like input to decimal form.
    Examples:
    - 0.08 -> 0.08
    - 8 -> 0.08
    - -3 -> -0.03
    """
    numeric = _safe_float(value, default=default)
    if abs(numeric) > 1.0:
        return numeric / 100.0
    return numeric


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping/dict-like object")
    return value


def _require_dataframe(value: Any, name: str) -> pd.DataFrame:
    if not isinstance(value, pd.DataFrame):
        raise ValueError(f"{name} must be a pandas DataFrame")
    return value


def _normalize_trend(value: Any) -> TrendDirection:
    text = str(value).strip().lower()
    if text == "declining":
        return TrendDirection.DECLINING
    if text == "flat":
        return TrendDirection.FLAT
    if text == "improving":
        return TrendDirection.IMPROVING
    return TrendDirection.UNKNOWN


def _normalize_posture(value: Any) -> StrategicPosture:
    text = str(value).strip().lower()
    if text == "hold":
        return StrategicPosture.HOLD
    if text == "watch":
        return StrategicPosture.WATCH
    if text == "sell":
        return StrategicPosture.SELL
    if text == "refinance":
        return StrategicPosture.REFINANCE
    return StrategicPosture.UNKNOWN


def _normalize_compliance_label(value: Any) -> ComplianceLabel:
    text = str(value).strip().lower()
    if text == "breach":
        return ComplianceLabel.BREACH
    if text == "watchlist":
        return ComplianceLabel.WATCHLIST
    if text == "refinance candidate":
        return ComplianceLabel.REFINANCE_CANDIDATE
    if text == "compliant":
        return ComplianceLabel.COMPLIANT
    return ComplianceLabel.UNKNOWN


def _priority_rank(priority: Priority) -> int:
    mapping = {
        Priority.CRITICAL: 0,
        Priority.HIGH: 1,
        Priority.MEDIUM: 2,
        Priority.LOW: 3,
    }
    return mapping[priority]


def _category_rank(category: Category) -> int:
    mapping = {
        Category.COMPLIANCE: 0,
        Category.PERFORMANCE: 1,
        Category.EXECUTION: 2,
        Category.MARKET: 3,
        Category.CAPITAL: 4,
        Category.STRATEGY: 5,
    }
    return mapping[category]


def _append_recommendation(
    recommendations: list[RecommendationRecord],
    *,
    context: PropertyContext,
    category: Category,
    priority: Priority,
    title: str,
    recommendation: str,
    rationale: str,
    metric_value: Any = None,
) -> None:
    recommendations.append(
        RecommendationRecord(
            property_id=context.property_id,
            property_name=context.property_name,
            category=category,
            priority=priority,
            title=title,
            recommendation=recommendation,
            rationale=rationale,
            metric_value=metric_value,
        )
    )


def _recommendations_to_frame(recommendations: Sequence[RecommendationRecord]) -> pd.DataFrame:
    rows = [
        {
            "Property_ID": r.property_id,
            "Property_Name": r.property_name,
            "Category": r.category.value,
            "Priority": r.priority.value,
            "Title": r.title,
            "Recommendation": r.recommendation,
            "Rationale": r.rationale,
            "Metric_Value": r.metric_value,
            "_Priority_Rank": _priority_rank(r.priority),
            "_Category_Rank": _category_rank(r.category),
        }
        for r in recommendations
    ]

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(
            columns=[
                "Property_ID",
                "Property_Name",
                "Category",
                "Priority",
                "Title",
                "Recommendation",
                "Rationale",
                "Metric_Value",
            ]
        )

    return (
        df.sort_values(["_Priority_Rank", "_Category_Rank", "Title"])
        .drop(columns=["_Priority_Rank", "_Category_Rank"])
        .head(MAX_RECOMMENDATIONS_PER_PROPERTY)
        .reset_index(drop=True)
    )


# ============================================================================
# DATA VALIDATION / INTEGRATION HELPERS
# ============================================================================

def _get_property_row(property_id: str) -> pd.Series:
    props = load_properties().copy()
    if "Property_ID" not in props.columns:
        raise ValueError("properties table missing required column 'Property_ID'")

    subset = props.loc[props["Property_ID"] == property_id]
    if subset.empty:
        raise ValueError(f"Property not found: {property_id}")
    if len(subset) > 1:
        raise ValueError(f"Duplicate Property_ID in properties table: {property_id}")
    return subset.iloc[0]


def _extract_property_context(property_id: str) -> PropertyContext:
    row = _get_property_row(property_id)
    return PropertyContext(
        property_id=property_id,
        property_name=str(row.get("Property_Name", property_id)),
    )


def _extract_performance_input(property_id: str) -> PerformanceInput:
    variance_table = _require_dataframe(
        build_variance_summary_table(property_id, trailing_months=12),
        "variance_table",
    )
    t12_summary = _require_mapping(get_t12_summary(property_id), "t12_summary")

    noi_trend = _normalize_trend(get_noi_trend_direction(property_id, window_months=12))
    streak = _safe_int(
        get_consecutive_variance_months(
            property_id=property_id,
            line_item_type="Revenue",
            direction="negative",
            threshold_pct=3.0,
        ),
        default=0,
    )

    has_red = False
    has_amber = False
    worst_line_item: str | None = None

    if not variance_table.empty and "RAG_Status" in variance_table.columns:
        red_rows = variance_table.loc[variance_table["RAG_Status"] == "Red"]
        amber_rows = variance_table.loc[variance_table["RAG_Status"] == "Amber"]
        has_red = len(red_rows) > 0
        has_amber = len(amber_rows) > 0

        source_rows = red_rows if has_red else amber_rows
        if not source_rows.empty and "Line_Item" in source_rows.columns:
            worst_line_item = str(source_rows.iloc[0]["Line_Item"])

    return PerformanceInput(
        has_red_variance=has_red,
        has_amber_variance=has_amber,
        worst_variance_line_item=worst_line_item,
        revenue_underperformance_streak_months=max(streak, 0),
        noi_trend=noi_trend,
        noi_variance_pct=_normalize_percent_like(t12_summary.get("noi_variance_pct"), default=0.0),
    )


def _extract_forecast_input(property_id: str) -> ForecastInput:
    assumptions = _require_mapping(
        get_base_assumptions_for_property(property_id),
        "base_assumptions",
    )

    required_keys = {
        "rent_growth",
        "vacancy_rate",
        "expense_growth",
        "capex_per_unit",
        "exit_cap_rate",
        "hold_years",
    }
    missing = required_keys - set(assumptions.keys())
    if missing:
        raise ValueError(f"base_assumptions missing keys: {sorted(missing)}")

    forecast = build_property_forecast(
        property_id=property_id,
        rent_growth=assumptions["rent_growth"],
        vacancy_rate=assumptions["vacancy_rate"],
        expense_growth=assumptions["expense_growth"],
        capex_per_unit=assumptions["capex_per_unit"],
        exit_cap_rate=assumptions["exit_cap_rate"],
        hold_years=int(assumptions["hold_years"]),
    )

    hold_sell = _require_mapping(
        get_hold_sell_recommendation(property_id, forecast),
        "hold_sell_recommendation",
    )

    return ForecastInput(
        strategic_posture=_normalize_posture(hold_sell.get("recommendation", "Unknown"))
    )


def _extract_valuation_input(property_id: str) -> ValuationInput:
    valuation = _require_mapping(
        build_valuation_reconciliation(property_id),
        "valuation",
    )
    uw_compare = _require_mapping(
        compare_actual_vs_underwriting(property_id),
        "underwriting_comparison",
    )

    return ValuationInput(
        unrealized_gain_pct=_normalize_percent_like(
            valuation.get("unrealized_gain_pct"), default=0.0
        ),
        outperforming_thesis=bool(uw_compare.get("outperforming_thesis", False)),
        noi_gap_pct=(
            _normalize_percent_like(uw_compare.get("noi_gap_pct"), default=np.nan)
            if pd.notna(uw_compare.get("noi_gap_pct"))
            else None
        ),
    )


def _extract_market_input(property_id: str) -> MarketInput:
    summary = _require_mapping(
        get_subject_vs_comp_summary(property_id),
        "market_summary",
    )
    score = _require_mapping(
        get_competitive_positioning_score(property_id),
        "market_score",
    )
    flags = get_market_watchlist_flags(property_id)
    if not isinstance(flags, list):
        raise ValueError("market watchlist flags must be a list")

    positioning_score = _safe_float(score.get("Positioning_Score"), default=50.0)
    if not (0.0 <= positioning_score <= 100.0):
        raise ValueError(f"Positioning_Score must be in [0, 100], got {positioning_score}")

    return MarketInput(
        rent_premium_pct=_normalize_percent_like(summary.get("Rent_Premium_Pct"), default=0.0),
        occupancy_gap=_normalize_percent_like(summary.get("Occupancy_Gap"), default=0.0),
        positioning_score=positioning_score,
        positioning_rag=str(score.get("RAG_Status", "")),
        flags=tuple(str(x) for x in flags),
    )


def _extract_execution_input(property_id: str) -> ExecutionInput:
    bp_summary = _require_mapping(
        get_property_business_plan_summary(property_id),
        "business_plan_summary",
    )
    bp_roi = _require_mapping(
        calculate_renovation_roi(property_id),
        "business_plan_roi",
    )
    delayed = _require_dataframe(
        get_delayed_initiatives(property_id),
        "delayed_initiatives",
    )

    top_delayed: str | None = None
    if not delayed.empty:
        if "Initiative" in delayed.columns:
            top_delayed = str(delayed.iloc[0]["Initiative"])
        else:
            top_delayed = "Delayed initiative"

    roc_raw = bp_roi.get("Return_on_Cost")
    roc = _normalize_percent_like(roc_raw, default=np.nan) if pd.notna(roc_raw) else None

    return ExecutionInput(
        delayed_count=max(_safe_int(bp_summary.get("Delayed_Count"), default=0), 0),
        budget_variance_pct=_normalize_percent_like(
            bp_summary.get("Budget_Variance_Pct"),
            default=0.0,
        ),
        total_actual_spend=max(
            _safe_float(bp_summary.get("Total_Actual_Spend"), default=0.0),
            0.0,
        ),
        renovation_return_on_cost=roc,
        top_delayed_initiative=top_delayed,
    )


def _extract_compliance_input(property_id: str) -> ComplianceInput:
    compliance = _require_mapping(
        get_property_compliance_summary(property_id),
        "compliance_summary",
    )

    label = _normalize_compliance_label(compliance.get("Compliance_Label", "Unknown"))

    days_to_maturity_raw = compliance.get("Days_to_Maturity")
    days_to_reporting_raw = compliance.get("Days_to_Reporting")

    return ComplianceInput(
        compliance_label=label,
        dscr_actual=(
            _safe_float(compliance.get("DSCR_Actual"), default=np.nan)
            if pd.notna(compliance.get("DSCR_Actual"))
            else None
        ),
        dscr_headroom=(
            _safe_float(compliance.get("DSCR_Headroom"), default=np.nan)
            if pd.notna(compliance.get("DSCR_Headroom"))
            else None
        ),
        days_to_maturity=(
            max(_safe_int(days_to_maturity_raw), 0)
            if pd.notna(days_to_maturity_raw)
            else None
        ),
        days_to_reporting=(
            max(_safe_int(days_to_reporting_raw), 0)
            if pd.notna(days_to_reporting_raw)
            else None
        ),
        overall_rag=str(compliance.get("Overall_RAG", "")),
    )


@st.cache_data(show_spinner=False)
def _get_property_analytics_snapshot(property_id: str) -> RecommendationSnapshot:
    """
    Build a fully validated, typed analytics snapshot.

    This is the single integration boundary for upstream modules.
    Downstream recommendation logic should rely only on this validated model.
    """
    return RecommendationSnapshot(
        context=_extract_property_context(property_id),
        performance=_extract_performance_input(property_id),
        forecast=_extract_forecast_input(property_id),
        valuation=_extract_valuation_input(property_id),
        market=_extract_market_input(property_id),
        execution=_extract_execution_input(property_id),
        compliance=_extract_compliance_input(property_id),
    )


# ============================================================================
# PURE BUSINESS LOGIC
# ============================================================================

def _build_recommendations(snapshot: RecommendationSnapshot) -> list[RecommendationRecord]:
    """
    Pure deterministic recommendation generator.
    """
    recs: list[RecommendationRecord] = []
    ctx = snapshot.context
    perf = snapshot.performance
    fcst = snapshot.forecast
    val = snapshot.valuation
    market = snapshot.market
    execution = snapshot.execution
    compliance = snapshot.compliance

    # ----------------------------------------------------------------------
    # Performance
    # ----------------------------------------------------------------------
    if perf.has_red_variance:
        worst_line = perf.worst_variance_line_item or "operating performance"
        _append_recommendation(
            recs,
            context=ctx,
            category=Category.PERFORMANCE,
            priority=Priority.HIGH,
            title="Address red variance drivers",
            recommendation=f"Launch an operating review focused on {worst_line} variance and immediate corrective actions.",
            rationale="One or more line items are materially off budget, indicating near-term earnings pressure on NOI.",
            metric_value=worst_line,
        )
    elif perf.has_amber_variance:
        worst_line = perf.worst_variance_line_item or "operating performance"
        _append_recommendation(
            recs,
            context=ctx,
            category=Category.PERFORMANCE,
            priority=Priority.MEDIUM,
            title="Monitor amber variance items",
            recommendation=f"Track {worst_line} variance monthly and validate whether the variance is temporary or structural.",
            rationale="Moderate budget variance is present and may deteriorate without intervention.",
            metric_value=worst_line,
        )

    if perf.revenue_underperformance_streak_months >= 3:
        _append_recommendation(
            recs,
            context=ctx,
            category=Category.PERFORMANCE,
            priority=Priority.HIGH,
            title="Stabilize recurring revenue underperformance",
            recommendation="Coordinate with the property manager to reset pricing, concessions, and leasing strategy over the next 30 days.",
            rationale=f"Revenue has underperformed budget for {perf.revenue_underperformance_streak_months} consecutive months.",
            metric_value=perf.revenue_underperformance_streak_months,
        )

    if perf.noi_trend == TrendDirection.DECLINING:
        _append_recommendation(
            recs,
            context=ctx,
            category=Category.PERFORMANCE,
            priority=Priority.HIGH,
            title="Arrest declining NOI trend",
            recommendation="Prepare a 90-day NOI recovery plan with revenue and expense action items by accountable owner.",
            rationale="Trailing NOI trend is declining, suggesting performance deterioration beyond a single-month anomaly.",
            metric_value=perf.noi_trend.value,
        )
    elif perf.noi_trend == TrendDirection.IMPROVING:
        _append_recommendation(
            recs,
            context=ctx,
            category=Category.PERFORMANCE,
            priority=Priority.LOW,
            title="Sustain operational momentum",
            recommendation="Maintain current operating cadence and document best practices that can be applied elsewhere in the portfolio.",
            rationale="NOI trend is improving, indicating recent operational gains are taking hold.",
            metric_value=perf.noi_trend.value,
        )

    # ----------------------------------------------------------------------
    # Strategy / forecast
    # ----------------------------------------------------------------------
    posture = fcst.strategic_posture
    strategy_priority = (
        Priority.MEDIUM if posture in {StrategicPosture.HOLD, StrategicPosture.WATCH}
        else Priority.LOW
    )
    _append_recommendation(
        recs,
        context=ctx,
        category=Category.STRATEGY,
        priority=strategy_priority,
        title=f"Strategic posture: {posture.value}",
        recommendation=f"Maintain current strategic stance as '{posture.value}' pending updated monthly performance.",
        rationale="Forecast-based hold/sell analysis provides a forward-looking strategic recommendation.",
        metric_value=posture.value,
    )

    if posture == StrategicPosture.SELL:
        _append_recommendation(
            recs,
            context=ctx,
            category=Category.CAPITAL,
            priority=Priority.HIGH,
            title="Evaluate disposition path",
            recommendation="Prepare an updated broker opinion of value and disposition timing analysis for investment committee review.",
            rationale="Forward return profile is weak relative to target and may support harvesting value or limiting downside.",
            metric_value=posture.value,
        )
    elif posture == StrategicPosture.REFINANCE:
        _append_recommendation(
            recs,
            context=ctx,
            category=Category.CAPITAL,
            priority=Priority.MEDIUM,
            title="Test refinance alternatives",
            recommendation="Engage debt capital markets or lending contacts to assess refinance proceeds and debt-service savings.",
            rationale="Forecast metrics suggest the asset may support refinancing or recapitalization options.",
            metric_value=posture.value,
        )

    # ----------------------------------------------------------------------
    # Valuation / thesis
    # ----------------------------------------------------------------------
    if val.unrealized_gain_pct >= 0.20:
        _append_recommendation(
            recs,
            context=ctx,
            category=Category.CAPITAL,
            priority=Priority.LOW,
            title="Document embedded value creation",
            recommendation="Update valuation support and business plan achievements for lender, investor, and IC communication.",
            rationale="The asset shows meaningful unrealized gain relative to acquisition basis.",
            metric_value=val.unrealized_gain_pct,
        )
    elif val.unrealized_gain_pct < 0.0:
        _append_recommendation(
            recs,
            context=ctx,
            category=Category.CAPITAL,
            priority=Priority.HIGH,
            title="Reassess asset basis and recovery plan",
            recommendation="Re-underwrite the asset using current operations and market assumptions to define realistic hold expectations.",
            rationale="Current indicated value is below acquisition basis, suggesting value erosion or delayed execution.",
            metric_value=val.unrealized_gain_pct,
        )

    if not val.outperforming_thesis:
        _append_recommendation(
            recs,
            context=ctx,
            category=Category.STRATEGY,
            priority=Priority.MEDIUM,
            title="Revisit original investment thesis",
            recommendation="Compare actual operating outcomes against underwriting assumptions and reset forward expectations where needed.",
            rationale="Actual performance is not fully supporting the original acquisition thesis.",
            metric_value=val.noi_gap_pct,
        )

    # ----------------------------------------------------------------------
    # Market
    # ----------------------------------------------------------------------
    if market.rent_premium_pct <= -0.05:
        _append_recommendation(
            recs,
            context=ctx,
            category=Category.MARKET,
            priority=Priority.HIGH,
            title="Close rent gap to market",
            recommendation="Review unit-by-unit pricing, concessions, amenity positioning, and leasing scripts to reduce discounting versus comps.",
            rationale="The asset is leasing materially below comparable properties, signaling pricing or product-positioning issues.",
            metric_value=market.rent_premium_pct,
        )

    if market.occupancy_gap <= -0.02:
        _append_recommendation(
            recs,
            context=ctx,
            category=Category.MARKET,
            priority=Priority.HIGH,
            title="Recover occupancy shortfall",
            recommendation="Implement a targeted leasing and retention plan focused on lead conversion, renewal pricing, and resident experience.",
            rationale="Occupancy trails the competitive set by more than 200 basis points.",
            metric_value=market.occupancy_gap,
        )

    if market.positioning_rag == "Red":
        _append_recommendation(
            recs,
            context=ctx,
            category=Category.MARKET,
            priority=Priority.HIGH,
            title="Reposition asset competitively",
            recommendation="Refresh the competitive positioning strategy and determine whether product, pricing, or execution is driving weak market standing.",
            rationale="Composite market positioning score indicates the asset is underperforming peers.",
            metric_value=market.positioning_score,
        )

    has_market_rec = any(r.category == Category.MARKET for r in recs)
    if not has_market_rec:
        cautionary_flags = [flag for flag in market.flags if "No material" not in flag]
        if cautionary_flags:
            _append_recommendation(
                recs,
                context=ctx,
                category=Category.MARKET,
                priority=Priority.LOW,
                title="Monitor market positioning signals",
                recommendation="Continue tracking local pricing, occupancy, and comp movement through the monthly operating review.",
                rationale=cautionary_flags[0],
                metric_value=cautionary_flags[0],
            )

    # ----------------------------------------------------------------------
    # Execution
    # ----------------------------------------------------------------------
    if execution.delayed_count > 0:
        _append_recommendation(
            recs,
            context=ctx,
            category=Category.EXECUTION,
            priority=Priority.HIGH,
            title="Resolve delayed initiatives",
            recommendation="Escalate delayed business plan items with updated owners, revised completion dates, and quantified NOI impact.",
            rationale=f"{execution.delayed_count} initiatives are delayed or at immediate timing risk.",
            metric_value=execution.delayed_count,
        )

    if execution.budget_variance_pct > 0.10:
        _append_recommendation(
            recs,
            context=ctx,
            category=Category.EXECUTION,
            priority=Priority.HIGH,
            title="Control CapEx budget drift",
            recommendation="Review open contracts, change orders, and contingency usage to contain further spend overruns.",
            rationale="Actual initiative spend is materially above approved budget.",
            metric_value=execution.budget_variance_pct,
        )

    if (
        execution.renovation_return_on_cost is not None
        and execution.renovation_return_on_cost < 0.08
        and execution.total_actual_spend > 0.0
    ):
        _append_recommendation(
            recs,
            context=ctx,
            category=Category.EXECUTION,
            priority=Priority.MEDIUM,
            title="Re-underwrite renovation ROI",
            recommendation="Validate whether planned capital deployment is still justified by achievable rent lift and NOI gains.",
            rationale="Business plan ROI appears weak relative to value-add expectations.",
            metric_value=execution.renovation_return_on_cost,
        )

    if (
        execution.top_delayed_initiative is not None
        and len(recs) < MAX_RECOMMENDATIONS_PER_PROPERTY
    ):
        _append_recommendation(
            recs,
            context=ctx,
            category=Category.EXECUTION,
            priority=Priority.MEDIUM,
            title="Prioritize highest-risk delayed initiative",
            recommendation=f"Create an action plan to unblock {execution.top_delayed_initiative} and protect expected NOI timing.",
            rationale="At least one delayed initiative may be impairing business plan execution velocity.",
            metric_value=execution.top_delayed_initiative,
        )

    # ----------------------------------------------------------------------
    # Compliance
    # ----------------------------------------------------------------------
    if compliance.compliance_label == ComplianceLabel.BREACH:
        _append_recommendation(
            recs,
            context=ctx,
            category=Category.COMPLIANCE,
            priority=Priority.CRITICAL,
            title="Address covenant breach immediately",
            recommendation="Prepare lender communication, root-cause analysis, and a corrective action plan for covenant restoration.",
            rationale="The asset is currently in covenant breach and requires immediate management attention.",
            metric_value=compliance.dscr_actual,
        )
    elif compliance.compliance_label == ComplianceLabel.WATCHLIST:
        _append_recommendation(
            recs,
            context=ctx,
            category=Category.COMPLIANCE,
            priority=Priority.HIGH,
            title="Maintain lender watchlist monitoring",
            recommendation="Track covenant headroom monthly and pre-wire potential mitigation steps before a formal breach occurs.",
            rationale="Covenant headroom or timing risk is tight enough to justify elevated monitoring.",
            metric_value=compliance.dscr_headroom,
        )
    elif compliance.compliance_label == ComplianceLabel.REFINANCE_CANDIDATE:
        _append_recommendation(
            recs,
            context=ctx,
            category=Category.CAPITAL,
            priority=Priority.MEDIUM,
            title="Evaluate refinance execution",
            recommendation="Refresh debt options and confirm whether refinancing improves proceeds, pricing, or debt-service profile.",
            rationale="Loan metrics and maturity profile suggest a refinance opportunity.",
            metric_value=compliance.days_to_maturity,
        )

    if compliance.days_to_reporting is not None and compliance.days_to_reporting <= 15:
        _append_recommendation(
            recs,
            context=ctx,
            category=Category.COMPLIANCE,
            priority=Priority.HIGH,
            title="Prepare upcoming lender reporting",
            recommendation="Finalize covenant package and lender reporting support ahead of deadline.",
            rationale="A reporting deadline is imminent.",
            metric_value=compliance.days_to_reporting,
        )

    # ----------------------------------------------------------------------
    # Fallback
    # ----------------------------------------------------------------------
    if not recs:
        _append_recommendation(
            recs,
            context=ctx,
            category=Category.STRATEGY,
            priority=Priority.LOW,
            title="No material action items",
            recommendation="Continue routine monitoring and monthly reporting cadence.",
            rationale="No material issues were identified across current analytics.",
            metric_value=None,
        )

    return recs


def _compute_watchlist_score(snapshot: RecommendationSnapshot) -> WatchlistScore:
    """
    Pure deterministic watchlist scoring function.
    """
    _assert_weight_configuration()

    ctx = snapshot.context
    perf = snapshot.performance
    fcst = snapshot.forecast
    val = snapshot.valuation
    market = snapshot.market
    execution = snapshot.execution
    compliance = snapshot.compliance

    # Compliance score
    if compliance.compliance_label == ComplianceLabel.BREACH:
        compliance_score = 100.0
    elif compliance.compliance_label == ComplianceLabel.WATCHLIST:
        compliance_score = 65.0
    elif compliance.compliance_label == ComplianceLabel.REFINANCE_CANDIDATE:
        compliance_score = 35.0
    else:
        compliance_score = 10.0

    # Performance score
    performance_score = 20.0
    if perf.noi_trend == TrendDirection.DECLINING:
        performance_score += 35.0
    elif perf.noi_trend == TrendDirection.FLAT:
        performance_score += 15.0

    if perf.noi_variance_pct < -0.08:
        performance_score += 45.0
    elif perf.noi_variance_pct < -0.03:
        performance_score += 25.0

    performance_score = _clamp(performance_score, 0.0, 100.0)

    # Forecast score
    forecast_score = 15.0
    if fcst.strategic_posture in {StrategicPosture.SELL, StrategicPosture.WATCH}:
        forecast_score += 35.0
    elif fcst.strategic_posture == StrategicPosture.REFINANCE:
        forecast_score += 20.0

    if val.unrealized_gain_pct < 0.0:
        forecast_score += 25.0

    forecast_score = _clamp(forecast_score, 0.0, 100.0)

    # Market score is inverse of positioning quality
    market_score = _clamp(100.0 - market.positioning_score, 0.0, 100.0)

    # Execution score
    execution_score = 10.0
    execution_score += min(execution.delayed_count * 15.0, 45.0)

    if execution.budget_variance_pct > 0.10:
        execution_score += 25.0
    elif execution.budget_variance_pct > 0.05:
        execution_score += 15.0

    execution_score = _clamp(execution_score, 0.0, 100.0)

    # Valuation score:
    # + positive unrealized gain lowers risk
    # + negative unrealized gain raises risk
    # map gain_pct linearly so:
    #   +20% => 30 risk
    #    0%  => 50 risk
    #   -20% => 70 risk
    valuation_score = _clamp(50.0 - (val.unrealized_gain_pct * 100.0), 0.0, 100.0)

    total_score = (
        compliance_score * (WEIGHT_COMPLIANCE / 100.0)
        + performance_score * (WEIGHT_PERFORMANCE / 100.0)
        + forecast_score * (WEIGHT_FORECAST / 100.0)
        + market_score * (WEIGHT_MARKET / 100.0)
        + execution_score * (WEIGHT_EXECUTION / 100.0)
        + valuation_score * (WEIGHT_VALUATION / 100.0)
    )
    total_score = _clamp(total_score, 0.0, 100.0)

    if total_score >= WATCHLIST_RED_THRESHOLD:
        bucket = WatchlistBucket.RED
    elif total_score >= WATCHLIST_AMBER_THRESHOLD:
        bucket = WatchlistBucket.AMBER
    else:
        bucket = WatchlistBucket.GREEN

    return WatchlistScore(
        property_id=ctx.property_id,
        property_name=ctx.property_name,
        watchlist_score=total_score,
        watchlist_bucket=bucket,
        compliance_score=compliance_score,
        performance_score=performance_score,
        forecast_score=forecast_score,
        market_score=market_score,
        execution_score=execution_score,
        valuation_score=valuation_score,
        valuation_signal=val.unrealized_gain_pct,
    )


# ============================================================================
# PUBLIC API
# ============================================================================

@st.cache_data(show_spinner=False)
def get_property_recommendations(property_id: str) -> pd.DataFrame:
    snapshot = _get_property_analytics_snapshot(property_id)
    recommendations = _build_recommendations(snapshot)
    return _recommendations_to_frame(recommendations)


@st.cache_data(show_spinner=False)
def get_property_watchlist_score(property_id: str) -> dict[str, Any]:
    snapshot = _get_property_analytics_snapshot(property_id)
    score = _compute_watchlist_score(snapshot)
    return {
        "Property_ID": score.property_id,
        "Property_Name": score.property_name,
        "Watchlist_Score": score.watchlist_score,
        "Watchlist_Bucket": score.watchlist_bucket.value,
        "Compliance_Score": score.compliance_score,
        "Performance_Score": score.performance_score,
        "Forecast_Score": score.forecast_score,
        "Market_Score": score.market_score,
        "Execution_Score": score.execution_score,
        "Valuation_Score": score.valuation_score,
        "Valuation_Signal": score.valuation_signal,
    }


@st.cache_data(show_spinner=False)
def get_portfolio_watchlist() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for property_id in get_property_list():
        watch = get_property_watchlist_score(property_id)
        recs = get_property_recommendations(property_id)

        top_title = recs.iloc[0]["Title"] if not recs.empty else "No material action items"
        top_priority = recs.iloc[0]["Priority"] if not recs.empty else Priority.LOW.value

        rows.append(
            {
                "Property_ID": property_id,
                "Property_Name": watch["Property_Name"],
                "Watchlist_Score": watch["Watchlist_Score"],
                "Watchlist_Bucket": watch["Watchlist_Bucket"],
                "Top_Priority": top_priority,
                "Top_Recommendation": top_title,
            }
        )

    return pd.DataFrame(rows).sort_values("Watchlist_Score", ascending=False).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def get_portfolio_recommendation_summary() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    property_name_map = get_property_name_map()

    for property_id in get_property_list():
        recs = get_property_recommendations(property_id)

        critical_count = int((recs["Priority"] == Priority.CRITICAL.value).sum()) if not recs.empty else 0
        high_count = int((recs["Priority"] == Priority.HIGH.value).sum()) if not recs.empty else 0
        medium_count = int((recs["Priority"] == Priority.MEDIUM.value).sum()) if not recs.empty else 0

        top_priority = recs.iloc[0]["Priority"] if not recs.empty else Priority.LOW.value
        top_title = recs.iloc[0]["Title"] if not recs.empty else "No material action items"

        rows.append(
            {
                "Property_ID": property_id,
                "Property_Name": property_name_map.get(property_id, property_id),
                "Top_Priority": top_priority,
                "Top_Recommendation": top_title,
                "Critical_Count": critical_count,
                "High_Count": high_count,
                "Medium_Count": medium_count,
                "Recommendation_Count": len(recs),
            }
        )

    df = pd.DataFrame(rows)
    rank_map = {
        Priority.CRITICAL.value: 0,
        Priority.HIGH.value: 1,
        Priority.MEDIUM.value: 2,
        Priority.LOW.value: 3,
    }
    df["Priority_Rank"] = df["Top_Priority"].map(rank_map).fillna(9)

    return (
        df.sort_values(
            ["Priority_Rank", "Critical_Count", "High_Count"],
            ascending=[True, False, False],
        )
        .drop(columns="Priority_Rank")
        .reset_index(drop=True)
    )


@st.cache_data(show_spinner=False)
def generate_executive_commentary(property_id: str) -> str:
    snapshot = _get_property_analytics_snapshot(property_id)
    watch = _compute_watchlist_score(snapshot)
    recs = get_property_recommendations(property_id)
    market_text = generate_market_narrative(property_id)

    top_action = recs.iloc[0]["Recommendation"] if not recs.empty else "Continue standard monitoring."
    top_priority = recs.iloc[0]["Priority"] if not recs.empty else Priority.LOW.value

    return (
        f"{snapshot.context.property_name} is currently classified as a "
        f"{watch.watchlist_bucket.value} watchlist asset with a composite score of "
        f"{watch.watchlist_score:.1f}. Top priority is {top_priority.lower()} and centers on: "
        f"{top_action} Debt status is {snapshot.compliance.compliance_label.value.lower()} "
        f"with overall covenant RAG of {snapshot.compliance.overall_rag}. {market_text}"
    )


# ============================================================================
# SELF-TESTS
# ============================================================================

def _run_unit_like_self_tests() -> None:
    """
    Pure-logic tests independent of upstream data quality.
    """
    _assert_weight_configuration()

    snapshot = RecommendationSnapshot(
        context=PropertyContext(property_id="P1", property_name="Test Asset"),
        performance=PerformanceInput(
            has_red_variance=True,
            has_amber_variance=False,
            worst_variance_line_item="Revenue",
            revenue_underperformance_streak_months=4,
            noi_trend=TrendDirection.DECLINING,
            noi_variance_pct=-0.10,
        ),
        forecast=ForecastInput(strategic_posture=StrategicPosture.SELL),
        valuation=ValuationInput(
            unrealized_gain_pct=-0.12,
            outperforming_thesis=False,
            noi_gap_pct=-0.08,
        ),
        market=MarketInput(
            rent_premium_pct=-0.06,
            occupancy_gap=-0.03,
            positioning_score=32.0,
            positioning_rag="Red",
            flags=("Weak positioning.",),
        ),
        execution=ExecutionInput(
            delayed_count=2,
            budget_variance_pct=0.12,
            total_actual_spend=150000.0,
            renovation_return_on_cost=0.05,
            top_delayed_initiative="Unit renovations",
        ),
        compliance=ComplianceInput(
            compliance_label=ComplianceLabel.BREACH,
            dscr_actual=0.92,
            dscr_headroom=-0.08,
            days_to_maturity=120,
            days_to_reporting=10,
            overall_rag="Red",
        ),
    )

    recs = _build_recommendations(snapshot)
    assert len(recs) >= 1, "Expected recommendations from stressed snapshot"
    assert any(r.priority == Priority.CRITICAL for r in recs), "Expected critical compliance recommendation"

    score = _compute_watchlist_score(snapshot)
    assert 0.0 <= score.watchlist_score <= 100.0, "Watchlist score out of bounds"
    assert score.watchlist_bucket in {
        WatchlistBucket.RED,
        WatchlistBucket.AMBER,
        WatchlistBucket.GREEN,
    }, "Invalid watchlist bucket"

    stable_snapshot = RecommendationSnapshot(
        context=PropertyContext(property_id="P2", property_name="Stable Asset"),
        performance=PerformanceInput(
            has_red_variance=False,
            has_amber_variance=False,
            worst_variance_line_item=None,
            revenue_underperformance_streak_months=0,
            noi_trend=TrendDirection.IMPROVING,
            noi_variance_pct=0.01,
        ),
        forecast=ForecastInput(strategic_posture=StrategicPosture.HOLD),
        valuation=ValuationInput(
            unrealized_gain_pct=0.15,
            outperforming_thesis=True,
            noi_gap_pct=0.02,
        ),
        market=MarketInput(
            rent_premium_pct=0.01,
            occupancy_gap=0.01,
            positioning_score=78.0,
            positioning_rag="Green",
            flags=("No material market positioning concerns identified.",),
        ),
        execution=ExecutionInput(
            delayed_count=0,
            budget_variance_pct=0.00,
            total_actual_spend=0.0,
            renovation_return_on_cost=None,
            top_delayed_initiative=None,
        ),
        compliance=ComplianceInput(
            compliance_label=ComplianceLabel.COMPLIANT,
            dscr_actual=1.45,
            dscr_headroom=0.25,
            days_to_maturity=365,
            days_to_reporting=45,
            overall_rag="Green",
        ),
    )

    stable_score = _compute_watchlist_score(stable_snapshot)
    assert stable_score.watchlist_score < score.watchlist_score, (
        "Stable snapshot should score better than stressed snapshot"
    )


def _run_integration_self_tests() -> None:
    """
    Integration tests against current upstream modules.
    """
    print("Running recommendation_engine.py self-tests...\n")

    property_ids = get_property_list()
    assert isinstance(property_ids, list), "get_property_list() must return a list"
    assert len(property_ids) >= 1, "Expected at least one property."

    for property_id in property_ids:
        snapshot = _get_property_analytics_snapshot(property_id)
        recs = get_property_recommendations(property_id)
        watch = get_property_watchlist_score(property_id)
        commentary = generate_executive_commentary(property_id)

        assert isinstance(snapshot, RecommendationSnapshot), f"Snapshot type invalid for {property_id}"
        assert isinstance(recs, pd.DataFrame), f"Recommendations must be DataFrame for {property_id}"
        assert len(recs) >= 1, f"Expected at least 1 recommendation for {property_id}"

        assert isinstance(watch, dict), f"Watchlist score must be dict for {property_id}"
        assert 0.0 <= watch["Watchlist_Score"] <= 100.0, f"Watchlist score out of bounds for {property_id}"

        assert isinstance(commentary, str) and len(commentary) > 40, (
            f"Executive commentary invalid for {property_id}"
        )

        top_rec = recs.iloc[0]
        print(
            f"✓ {property_id:<8} | "
            f"Watchlist Score: {watch['Watchlist_Score']:>5.1f} | "
            f"Bucket: {watch['Watchlist_Bucket']:<5} | "
            f"Top Action: {top_rec['Title']}"
        )

    watchlist = get_portfolio_watchlist()
    assert len(watchlist) == len(property_ids), "Portfolio watchlist row count mismatch."

    rec_summary = get_portfolio_recommendation_summary()
    assert len(rec_summary) == len(property_ids), "Recommendation summary row count mismatch."

    print("\nPortfolio watchlist:")
    print(watchlist.to_string(index=False))

    print("\nPortfolio recommendation summary:")
    print(rec_summary.to_string(index=False))

    print("\nALL RECOMMENDATION ENGINE TESTS PASSED")


def _run_self_tests() -> None:
    _run_unit_like_self_tests()
    _run_integration_self_tests()


if __name__ == "__main__":
    _run_self_tests()