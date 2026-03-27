from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd
import streamlit as st

from modules.data_loader import (
    get_business_plan_for_property,
    get_comps_for_property,
    get_performance_for_property,
    get_property_list,
    get_property_name_map,
    get_underwriting_for_property,
    load_properties,
)
from modules.forecasting import build_property_forecast, get_base_assumptions_for_property


# ============================================================================
# CONSTANTS
# ============================================================================

# Reconciliation weights
INCOME_APPROACH_WEIGHT = 0.50
SALES_COMP_APPROACH_WEIGHT = 0.30
DCF_APPROACH_WEIGHT = 0.20

# Cap rate adjustments
CAP_RATE_QUALITY_PREMIUM = 0.0025
CAP_RATE_AGE_PREMIUM = 0.0050
CAP_RATE_LOCATION_DISCOUNT = 0.0025

# Scenario labels
SCENARIO_BEAR = "Bear"
SCENARIO_BASE = "Base"
SCENARIO_BULL = "Bull"

# Scenario cap rate flex
BEAR_CAP_RATE_SPREAD = 0.0075
BULL_CAP_RATE_SPREAD = 0.0050

# Regional value-per-unit benchmarks
VALUE_PER_UNIT_BENCHMARKS = {
    "Northeast": 350_000,
    "Southeast": 225_000,
    "Midwest": 175_000,
    "Southwest": 200_000,
    "West_Coast": 425_000,
}

PRIMARY_MARKETS = {"Northeast", "West_Coast"}

LEASE_UP_CAP_RATE_PREMIUM = 0.0050
RENO_IN_PROGRESS_CAP_RATE_PREMIUM = 0.0075
RENOVATION_COMPLETE_CAP_RATE_DISCOUNT = 0.0015

DCF_LEASE_UP_DISCOUNT_RATE_PREMIUM = 0.0100
DCF_RENO_DISCOUNT_RATE_PREMIUM = 0.0125

LEASE_UP_COMP_DISCOUNT = 0.08
RENO_IN_PROGRESS_COMP_DISCOUNT = 0.12
RENOVATION_COMPLETE_COMP_PREMIUM = 0.03
STABILIZED_COMP_PREMIUM = 0.02

VALUE_BENCHMARK_GREEN_THRESHOLD = 0.20
VALUE_BENCHMARK_AMBER_THRESHOLD = 0.40

MIN_CAP_RATE = 0.0350
MAX_CAP_RATE = 0.0950
DEFAULT_EXIT_CAP_RATE = 0.0550
DEFAULT_DISCOUNT_RATE = 0.10
DEFAULT_YEAR_BUILT = 2005
DEFAULT_REGION_BENCHMARK = 250_000


# ============================================================================
# INTERNAL HELPERS
# ============================================================================

def _safe_float(value: Any, default: float = 0.0) -> float:
    """
    Safely convert a value to float.
    """
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalize_ratio(value: Any, default: float = 0.0) -> float:
    """
    Normalize a ratio-like value to decimal form.
    """
    numeric = _safe_float(value, default=default)
    if abs(numeric) > 1.0:
        return numeric / 100.0
    return numeric


def _gross_rent_multiplier(value: float, annual_gross_rent: float) -> float:
        """Calculate GRM safely with logical bounds."""
        floor_rent = max(annual_gross_rent, 10000.0)
        raw_grm = value / floor_rent
        return min(raw_grm, 25.0)


def _implied_property_value(noi: float, cap_rate: float) -> float:
    """
    Calculate implied value from NOI and cap rate.
    """
    if cap_rate <= 0:
        return np.nan
    return noi / cap_rate


def _get_property_row(property_id: str) -> pd.Series:
    """
    Return property row for one asset.
    """
    props = load_properties()
    subset = props[props["Property_ID"] == property_id]
    if subset.empty:
        raise ValueError(f"Property not found: {property_id}")
    return subset.iloc[0]


def _get_underwriting_row(property_id: str) -> pd.Series:
    """
    Return underwriting row for one asset.
    """
    row = get_underwriting_for_property(property_id)
    if row is None or len(row) == 0:
        raise ValueError(f"No underwriting record found for property: {property_id}")
    return row


def _get_property_units(prop_row: pd.Series) -> float:
    """
    Return unit count with safe floor.
    """
    return max(_safe_float(prop_row.get("Units"), default=1.0), 1.0)


def _get_property_region(prop_row: pd.Series) -> str:
    """
    Return geographic benchmark key for property.

    Preference order:
    1. Region if populated
    2. Market
    """
    region = str(prop_row.get("Region", "")).strip()
    if region:
        return region
    return str(prop_row.get("Market", "")).strip()


def _get_property_phase(prop_row: pd.Series) -> str:
    """
    Return canonical value-add phase.
    """
    return str(prop_row.get("Value_Add_Phase", "")).strip()


def _get_property_year_built(prop_row: pd.Series) -> int:
    """
    Return year built with safe default.
    """
    value = _safe_float(prop_row.get("Year_Built"), default=DEFAULT_YEAR_BUILT)
    return int(value)


def _get_acquisition_value(prop_row: pd.Series, uw_row: pd.Series) -> float:
    """
    Derive acquisition value from property master or underwriting.
    """
    direct_value = _safe_float(prop_row.get("Purchase_Price"), default=0.0)
    if direct_value > 0:
        return direct_value

    year1_noi = _safe_float(uw_row.get("Underwritten_NOI_Year1"), default=0.0)
    purchase_cap = _normalize_ratio(
        uw_row.get("Underwritten_Purchase_Cap_Rate"),
        default=0.0,
    )

    if year1_noi > 0 and purchase_cap > 0:
        return year1_noi / purchase_cap

    return 0.0


def _get_t12_metrics(property_id: str) -> dict[str, float]:
    """
    Calculate trailing 12-month property operating metrics.
    """
    perf = get_performance_for_property(property_id).copy()
    if perf.empty:
        raise ValueError(f"No performance data found for property: {property_id}")

    perf = perf.sort_values("Year_Month").tail(12)

    actual_noi = _safe_float(perf["Actual_NOI"].sum(), default=0.0)
    actual_revenue = _safe_float(perf["Actual_Revenue"].sum(), default=0.0)
    actual_expenses = _safe_float(perf["Actual_Expenses"].sum(), default=0.0)

    avg_occupancy = (
        float(pd.to_numeric(perf["Occupancy"], errors="coerce").mean())
        if "Occupancy" in perf.columns
        else 0.0
    )
    avg_occupancy = _normalize_ratio(avg_occupancy, default=0.0)

    concessions = _safe_float(
        perf.get("Concessions", pd.Series(dtype=float)).sum(),
        default=0.0,
    )
    bad_debt = _safe_float(
        perf.get("Bad_Debt", pd.Series(dtype=float)).sum(),
        default=0.0,
    )

    annual_gpr = actual_revenue + concessions + bad_debt
    if annual_gpr <= 0:
        annual_gpr = actual_revenue

    return {
        "noi": actual_noi,
        "revenue": actual_revenue,
        "expenses": actual_expenses,
        "occupancy": avg_occupancy,
        "annual_gpr": annual_gpr,
    }


def _derive_market_cap_rate(property_id: str) -> tuple[float, str]:
    """
    Derive property-specific market cap rate using underwriting and adjustments.
    """
    prop = _get_property_row(property_id)
    uw = _get_underwriting_row(property_id)

    raw_base_cap = uw.get("Underwritten_Exit_Cap_Rate")
    base_cap = _normalize_ratio(raw_base_cap, default=0.0)

    if base_cap > 0:
        cap_rate = base_cap
        source = "Underwritten_Exit_Cap_Rate"
    else:
        cap_rate = DEFAULT_EXIT_CAP_RATE
        source = "Default_Exit_Cap_Rate"

    phase = _get_property_phase(prop)
    region = _get_property_region(prop)
    year_built = _get_property_year_built(prop)

    if phase not in {"Stabilized", "Renovation_Complete"}:
        cap_rate += CAP_RATE_QUALITY_PREMIUM

    if (pd.Timestamp.today().year - year_built) > 20:
        cap_rate += CAP_RATE_AGE_PREMIUM

    if region in PRIMARY_MARKETS:
        cap_rate -= CAP_RATE_LOCATION_DISCOUNT

    if phase == "Lease_Up":
        cap_rate += LEASE_UP_CAP_RATE_PREMIUM
    elif phase == "Renovation_In_Progress":
        cap_rate += RENO_IN_PROGRESS_CAP_RATE_PREMIUM
    elif phase == "Renovation_Complete":
        cap_rate -= RENOVATION_COMPLETE_CAP_RATE_DISCOUNT

    cap_rate = min(max(cap_rate, MIN_CAP_RATE), MAX_CAP_RATE)
    return cap_rate, source


def _derive_discount_rate(property_id: str) -> float:
    """
    Derive DCF discount rate from underwriting IRR and risk adjustments.
    """
    prop = _get_property_row(property_id)
    uw = _get_underwriting_row(property_id)
    phase = _get_property_phase(prop)

    discount_rate = _normalize_ratio(
        uw.get("Underwritten_IRR"),
        default=DEFAULT_DISCOUNT_RATE,
    )
    if discount_rate <= 0:
        discount_rate = DEFAULT_DISCOUNT_RATE

    if phase == "Lease_Up":
        discount_rate += DCF_LEASE_UP_DISCOUNT_RATE_PREMIUM
    elif phase == "Renovation_In_Progress":
        discount_rate += DCF_RENO_DISCOUNT_RATE_PREMIUM

    return max(discount_rate, 0.06)


def _validate_weights() -> None:
    """
    Validate that reconciliation weights sum to 1.0.
    """
    total = (
        INCOME_APPROACH_WEIGHT
        + SALES_COMP_APPROACH_WEIGHT
        + DCF_APPROACH_WEIGHT
    )
    if not np.isclose(total, 1.0):
        raise ValueError(f"Valuation weights must sum to 1.0, got {total:.6f}")


# ============================================================================
# PUBLIC VALUATION FUNCTIONS
# ============================================================================

@st.cache_data(show_spinner=False)
def get_direct_cap_value(
    property_id: str,
    cap_rate_override: Optional[float] = None,
) -> dict[str, Any]:
    """
    Calculate value via direct capitalization.
    """
    prop = _get_property_row(property_id)
    t12 = _get_t12_metrics(property_id)

    if cap_rate_override is not None:
        cap_rate_applied = float(cap_rate_override)
        cap_rate_source = "Override"
    else:
        cap_rate_applied, cap_rate_source = _derive_market_cap_rate(property_id)

    indicated_value = _implied_property_value(t12["noi"], cap_rate_applied)
    units = _get_property_units(prop)
    value_per_unit = indicated_value / units if units > 0 else np.nan
    implied_grm = _gross_rent_multiplier(indicated_value, t12["annual_gpr"])

    return {
        "noi_used": t12["noi"],
        "cap_rate_applied": cap_rate_applied,
        "indicated_value": indicated_value,
        "value_per_unit": value_per_unit,
        "implied_grm": implied_grm,
        "cap_rate_source": cap_rate_source,
    }


@st.cache_data(show_spinner=False)
def get_dcf_value(
    property_id: str,
    forecast: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """
    Calculate value via discounted cash flow.
    """
    prop = _get_property_row(property_id)

    if forecast is None:
        assumptions = get_base_assumptions_for_property(property_id)
        forecast = build_property_forecast(
            property_id=property_id,
            rent_growth=assumptions["rent_growth"],
            vacancy_rate=assumptions["vacancy_rate"],
            expense_growth=assumptions["expense_growth"],
            capex_per_unit=assumptions["capex_per_unit"],
            exit_cap_rate=assumptions["exit_cap_rate"],
            hold_years=int(assumptions["hold_years"]),
        )

    discount_rate_used = _derive_discount_rate(property_id)

    operating_cfs = list(forecast.get("unlevered_cf", []))
    hold_years = int(forecast.get("hold_years", len(operating_cfs)))
    terminal_value = _safe_float(forecast.get("net_sale_proceeds_unlevered"), default=0.0)
    terminal_cap_rate = _normalize_ratio(
        forecast.get("exit_cap_rate"),
        default=0.0,
    )

    pv_operating_cfs = 0.0
    for year_index, cf in enumerate(operating_cfs, start=1):
        pv_operating_cfs += _safe_float(cf, default=0.0) / (
            (1 + discount_rate_used) ** year_index
        )

    if hold_years > 0:
        pv_terminal_value = terminal_value / ((1 + discount_rate_used) ** hold_years)
    else:
        pv_terminal_value = terminal_value

    total_dcf_value = pv_operating_cfs + pv_terminal_value

    units = _get_property_units(prop)
    value_per_unit = total_dcf_value / units if units > 0 else np.nan

    return {
        "pv_operating_cfs": pv_operating_cfs,
        "pv_terminal_value": pv_terminal_value,
        "total_dcf_value": total_dcf_value,
        "value_per_unit": value_per_unit,
        "discount_rate_used": discount_rate_used,
        "terminal_cap_rate": terminal_cap_rate,
        "hold_years": hold_years,
    }


@st.cache_data(show_spinner=False)
def get_sales_comp_value(property_id: str) -> dict[str, Any]:
    """
    Calculate value using market/sales comparison proxy methods.
    """
    prop = _get_property_row(property_id)
    units = _get_property_units(prop)
    region = _get_property_region(prop)
    phase = _get_property_phase(prop)
    t12 = _get_t12_metrics(property_id)

    comps = get_comps_for_property(property_id).copy()
    ppu_benchmark = _safe_float(
        VALUE_PER_UNIT_BENCHMARKS.get(region),
        default=DEFAULT_REGION_BENCHMARK,
    )

    if comps.empty:
        ppu_indicated_value = ppu_benchmark * units
        implied_grm = _gross_rent_multiplier(
            ppu_indicated_value,
            max(t12["annual_gpr"], 1.0),
        )
        blended_comp_value = ppu_indicated_value
        return {
            "implied_grm": implied_grm,
            "grm_indicated_value": blended_comp_value,
            "ppu_benchmark": ppu_benchmark,
            "ppu_indicated_value": ppu_indicated_value,
            "blended_comp_value": blended_comp_value,
            "value_per_unit": blended_comp_value / units,
            "comps_used_count": 0,
        }

    if "Comp_Occupancy" not in comps.columns and "Occupancy" in comps.columns:
        comps["Comp_Occupancy"] = comps["Occupancy"]
    if "Avg_Rent" not in comps.columns and "Market_Rent" in comps.columns:
        comps["Avg_Rent"] = comps["Market_Rent"]

    comp_values = []
    comp_weights = []

    for _, row in comps.iterrows():
        comp_annual_rent = _safe_float(row.get("Comp_Annual_Rent"), default=0.0)
        comp_occ = _normalize_ratio(row.get("Comp_Occupancy"), default=0.95)
        comp_growth = _normalize_ratio(row.get("Rent_Growth_YoY"), default=0.03)
        distance = _safe_float(row.get("Distance_Miles"), default=2.0)

        comp_ppu = ppu_benchmark
        comp_ppu *= 1.0 + ((comp_occ - 0.93) * 0.50)
        comp_ppu *= 1.0 + (comp_growth * 1.50)

        implied_comp_grm = (
            comp_ppu / comp_annual_rent
            if comp_annual_rent > 0
            else np.nan
        )

        if pd.notna(implied_comp_grm) and implied_comp_grm > 0:
            distance_weight = 1.0 / max(distance, 0.25)
            quality_weight = 1.0 + max(comp_occ - 0.92, -0.05)
            growth_weight = 1.0 + max(comp_growth, -0.02)
            weight = distance_weight * quality_weight * growth_weight

            comp_values.append(implied_comp_grm)
            comp_weights.append(weight)

    if comp_values:
        implied_grm = float(np.average(comp_values, weights=comp_weights))
    else:
        implied_grm = _gross_rent_multiplier(
            ppu_benchmark * units,
            max(t12["annual_gpr"], 1.0),
        )

    grm_indicated_value = implied_grm * t12["annual_gpr"]

    occupancy_adj = 1.0 + ((t12["occupancy"] - 0.93) * 0.75)

    phase_adj = 1.0
    if phase == "Lease_Up":
        phase_adj -= LEASE_UP_COMP_DISCOUNT
    elif phase == "Renovation_In_Progress":
        phase_adj -= RENO_IN_PROGRESS_COMP_DISCOUNT
    elif phase == "Renovation_Complete":
        phase_adj += RENOVATION_COMPLETE_COMP_PREMIUM
    elif phase == "Stabilized":
        phase_adj += STABILIZED_COMP_PREMIUM

    ppu_indicated_value = ppu_benchmark * units * max(0.75, occupancy_adj * phase_adj)
    blended_comp_value = (grm_indicated_value * 0.60) + (ppu_indicated_value * 0.40)
    value_per_unit = blended_comp_value / units if units > 0 else np.nan

    return {
        "implied_grm": implied_grm,
        "grm_indicated_value": grm_indicated_value,
        "ppu_benchmark": ppu_benchmark,
        "ppu_indicated_value": ppu_indicated_value,
        "blended_comp_value": blended_comp_value,
        "value_per_unit": value_per_unit,
        "comps_used_count": int(len(comps)),
    }


@st.cache_data(show_spinner=False)
def build_valuation_reconciliation(
    property_id: str,
    cap_rate_override: Optional[float] = None,
    forecast: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """
    Reconcile valuation methods into concluded value.
    """
    _validate_weights()

    prop = _get_property_row(property_id)
    uw = _get_underwriting_row(property_id)
    units = _get_property_units(prop)
    t12 = _get_t12_metrics(property_id)

    direct_cap = get_direct_cap_value(
        property_id,
        cap_rate_override=cap_rate_override,
    )
    dcf = get_dcf_value(property_id, forecast=forecast)
    sales_comp = get_sales_comp_value(property_id)

    concluded_value = (
        direct_cap["indicated_value"] * INCOME_APPROACH_WEIGHT
        + sales_comp["blended_comp_value"] * SALES_COMP_APPROACH_WEIGHT
        + dcf["total_dcf_value"] * DCF_APPROACH_WEIGHT
    )

    acquisition_value = _get_acquisition_value(prop, uw)
    concluded_value_per_unit = concluded_value / units if units > 0 else np.nan
    unrealized_gain = concluded_value - acquisition_value
    unrealized_gain_pct = (
        unrealized_gain / acquisition_value
        if acquisition_value > 0
        else np.nan
    )
    concluded_cap_rate = t12["noi"] / concluded_value if concluded_value > 0 else np.nan

    base_cap_rate = direct_cap["cap_rate_applied"]
    bear_cap_rate = min(base_cap_rate + BEAR_CAP_RATE_SPREAD, MAX_CAP_RATE)
    bull_cap_rate = max(base_cap_rate - BULL_CAP_RATE_SPREAD, MIN_CAP_RATE)

    bear_direct = get_direct_cap_value(
        property_id,
        cap_rate_override=bear_cap_rate,
    )["indicated_value"]
    bull_direct = get_direct_cap_value(
        property_id,
        cap_rate_override=bull_cap_rate,
    )["indicated_value"]

    value_range_low = min(bear_direct, concluded_value)
    value_range_high = max(bull_direct, concluded_value)
    value_range_spread = value_range_high - value_range_low

    return {
        "direct_cap": direct_cap,
        "dcf": dcf,
        "sales_comp": sales_comp,
        "concluded_value": concluded_value,
        "concluded_value_per_unit": concluded_value_per_unit,
        "acquisition_value": acquisition_value,
        "unrealized_gain": unrealized_gain,
        "unrealized_gain_pct": unrealized_gain_pct,
        "concluded_cap_rate": concluded_cap_rate,
        "value_range_low": value_range_low,
        "value_range_high": value_range_high,
        "value_range_spread": value_range_spread,
        "weights_used": {
            "income_approach": INCOME_APPROACH_WEIGHT,
            "sales_comp_approach": SALES_COMP_APPROACH_WEIGHT,
            "dcf_approach": DCF_APPROACH_WEIGHT,
        },
    }


@st.cache_data(show_spinner=False)
def build_scenario_valuation(property_id: str) -> dict[str, Any]:
    """
    Build bear/base/bull valuation scenarios.
    """
    base_assumptions = get_base_assumptions_for_property(property_id)
    base_cap_rate, _ = _derive_market_cap_rate(property_id)

    scenario_cap_rates = {
        SCENARIO_BEAR: min(base_cap_rate + BEAR_CAP_RATE_SPREAD, MAX_CAP_RATE),
        SCENARIO_BASE: base_cap_rate,
        SCENARIO_BULL: max(base_cap_rate - BULL_CAP_RATE_SPREAD, MIN_CAP_RATE),
    }

    output: dict[str, Any] = {}
    rows = []

    for scenario_name, scenario_cap_rate in scenario_cap_rates.items():
        forecast = build_property_forecast(
            property_id=property_id,
            rent_growth=base_assumptions["rent_growth"],
            vacancy_rate=base_assumptions["vacancy_rate"],
            expense_growth=base_assumptions["expense_growth"],
            capex_per_unit=base_assumptions["capex_per_unit"],
            exit_cap_rate=scenario_cap_rate,
            hold_years=int(base_assumptions["hold_years"]),
        )

        rec = build_valuation_reconciliation(
            property_id=property_id,
            cap_rate_override=scenario_cap_rate,
            forecast=forecast,
        )
        output[scenario_name] = rec

        rows.append(
            {
                "Scenario": scenario_name,
                "Cap_Rate": scenario_cap_rate,
                "Direct_Cap_Value": rec["direct_cap"]["indicated_value"],
                "DCF_Value": rec["dcf"]["total_dcf_value"],
                "Sales_Comp_Value": rec["sales_comp"]["blended_comp_value"],
                "Concluded_Value": rec["concluded_value"],
                "Value_Per_Unit": rec["concluded_value_per_unit"],
                "Unrealized_Gain": rec["unrealized_gain"],
                "Unrealized_Gain_Pct": rec["unrealized_gain_pct"],
            }
        )

    summary_table = pd.DataFrame(rows)
    scenario_order = {SCENARIO_BEAR: 0, SCENARIO_BASE: 1, SCENARIO_BULL: 2}
    summary_table["Scenario_Order"] = summary_table["Scenario"].map(scenario_order)
    summary_table = (
        summary_table.sort_values("Scenario_Order")
        .drop(columns="Scenario_Order")
        .reset_index(drop=True)
    )

    output["summary_table"] = summary_table
    return output


@st.cache_data(show_spinner=False)
def get_value_bridge(property_id: str) -> pd.DataFrame:
    """
    Build a value bridge from acquisition value to current indicated value.
    """
    prop = _get_property_row(property_id)
    uw = _get_underwriting_row(property_id)
    reconciliation = build_valuation_reconciliation(property_id)
    t12 = _get_t12_metrics(property_id)

    acquisition_value = reconciliation["acquisition_value"]
    current_indicated_value = reconciliation["concluded_value"]

    current_cap_rate = reconciliation["direct_cap"]["cap_rate_applied"]
    purchase_cap_rate = _normalize_ratio(
        uw.get("Underwritten_Purchase_Cap_Rate"),
        default=current_cap_rate,
    )
    year1_noi = _safe_float(
        uw.get("Underwritten_NOI_Year1"),
        default=t12["noi"],
    )

    noi_growth_contribution = 0.0
    if purchase_cap_rate > 0:
        noi_growth_contribution = (t12["noi"] - year1_noi) / purchase_cap_rate

    cap_rate_compression_contribution = 0.0
    if purchase_cap_rate > 0 and current_cap_rate > 0:
        cap_rate_compression_contribution = (
            (t12["noi"] / current_cap_rate) - (t12["noi"] / purchase_cap_rate)
        )

    business_plan = get_business_plan_for_property(property_id).copy()
    expected_noi_lift = 0.0
    if not business_plan.empty and "Expected_NOI_Lift" in business_plan.columns:
        expected_noi_lift = _safe_float(
            business_plan["Expected_NOI_Lift"].fillna(0).sum(),
            default=0.0,
        )

    renovation_value_add = (
        expected_noi_lift / current_cap_rate if current_cap_rate > 0 else 0.0
    )

    subtotal = (
        acquisition_value
        + noi_growth_contribution
        + cap_rate_compression_contribution
        + renovation_value_add
    )
    
    direct_cap_value = reconciliation["direct_cap"]["indicated_value"]
    direct_cap_variance = direct_cap_value - subtotal
    reconciliation_variance = current_indicated_value - direct_cap_value

    rows = [
        {
            "Component": "Acquisition_Value",
            "Value": acquisition_value,
            "Is_Positive": True,
        },
        {
            "Component": "NOI_Growth_Contribution",
            "Value": noi_growth_contribution,
            "Is_Positive": noi_growth_contribution >= 0,
        },
        {
            "Component": "Cap_Rate_Compression_Contribution",
            "Value": cap_rate_compression_contribution,
            "Is_Positive": cap_rate_compression_contribution >= 0,
        },
        {
            "Component": "Renovation_Value_Add",
            "Value": renovation_value_add,
            "Is_Positive": renovation_value_add >= 0,
        },
    ]

    if abs(direct_cap_variance) > 1.0:
        rows.append(
            {
                "Component": "Direct_Cap_Execution_Variance",
                "Value": direct_cap_variance,
                "Is_Positive": direct_cap_variance >= 0,
            }
        )
    if abs(reconciliation_variance) > 1.0:
        rows.append(
            {
                "Component": "Valuation_Methodology_Premium",
                "Value": reconciliation_variance,
                "Is_Positive": reconciliation_variance >= 0,
            }
        )

    bridge = pd.DataFrame(rows)
    bridge["Cumulative_Value"] = bridge["Value"].cumsum()
    return bridge


@st.cache_data(show_spinner=False)
def check_value_vs_benchmark(property_id: str, indicated_value: float) -> dict[str, Any]:
    """
    Compare concluded value per unit against regional benchmark.
    Safely handles missing data (NaN) and outputs UI-ready RAG status.
    """
    prop = _get_property_row(property_id)
    units = _get_property_units(prop)
    region = _get_property_region(prop)

    benchmark_ppu = _safe_float(
        VALUE_PER_UNIT_BENCHMARKS.get(region),
        default=DEFAULT_REGION_BENCHMARK,
    )
    
    # Safely calculate actual PPU
    actual_ppu = indicated_value / units if units > 0 else np.nan
    
    # Mathematically correct pure decimal variance
    variance_pct = (
        (actual_ppu - benchmark_ppu) / benchmark_ppu
        if benchmark_ppu > 0 and pd.notna(actual_ppu)
        else np.nan
    )

    # 🛡️ THE FIX: Shield against the NaN "False Red" trap
    if pd.isna(variance_pct):
        rag_status = "⚪ N/A (Missing Data)"
    else:
        abs_variance_pct = abs(variance_pct)
        # Using your original thresholds with visual UI enhancements
        if abs_variance_pct <= VALUE_BENCHMARK_GREEN_THRESHOLD:
            rag_status = "🟢 Green"
        elif abs_variance_pct <= VALUE_BENCHMARK_AMBER_THRESHOLD:
            rag_status = "🟡 Amber"
        else:
            rag_status = "🔴 Red"

    return {
        "benchmark_ppu": benchmark_ppu,
        "actual_ppu": actual_ppu,
        "variance_pct": variance_pct, # Outputs pure decimal for your fmt_pct
        "is_above_benchmark": bool(actual_ppu > benchmark_ppu) if pd.notna(actual_ppu) else False,
        "RAG_Status": rag_status,
    }


@st.cache_data(show_spinner=False)
def get_valuation_summary_table() -> pd.DataFrame:
    """
    Build portfolio-level valuation summary table.
    """
    rows = []
    property_name_map = get_property_name_map()

    for property_id in get_property_list():
        rec = build_valuation_reconciliation(property_id)
        benchmark_check = check_value_vs_benchmark(
            property_id,
            rec["concluded_value"],
        )

        rows.append(
            {
                "Property_ID": property_id,
                "Property_Name": property_name_map.get(property_id, property_id),
                "Acquisition_Value": rec["acquisition_value"],
                "Indicated_Value": rec["concluded_value"],
                "Unrealized_Gain": rec["unrealized_gain"],
                "Unrealized_Gain_Pct": rec["unrealized_gain_pct"],
                "Value_Per_Unit": rec["concluded_value_per_unit"],
                "Implied_Cap_Rate": rec["concluded_cap_rate"],
                "vs_Benchmark_Pct": benchmark_check["variance_pct"],
                "RAG_Status": benchmark_check["RAG_Status"],
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values("Unrealized_Gain_Pct", ascending=False)
        .reset_index(drop=True)
    )


# ============================================================================
# SELF-TESTS
# ============================================================================

def _assert_between(value: float, low: float, high: float, message: str) -> None:
    """
    Assert numeric value lies within range.
    """
    if not (low <= value <= high):
        raise AssertionError(message)


def _run_self_tests() -> None:
    """
    Run module self-tests.
    """
    print("Running valuation.py self-tests...\n")

    _validate_weights()
    property_ids = get_property_list()
    assert len(property_ids) >= 1, "Expected at least one property."

    for property_id in property_ids:
        direct_cap = get_direct_cap_value(property_id)
        dcf = get_dcf_value(property_id)
        sales_comp = get_sales_comp_value(property_id)
        reconciliation = build_valuation_reconciliation(property_id)
        scenario_output = build_scenario_valuation(property_id)
        bridge = get_value_bridge(property_id)
        benchmark = check_value_vs_benchmark(
            property_id,
            reconciliation["concluded_value"],
        )

        assert (
            direct_cap["indicated_value"] > 0
        ), f"Direct cap must be positive for {property_id}"
        assert (
            direct_cap["implied_grm"] > 0
        ), f"Direct cap GRM must be positive for {property_id}"

        assert dcf["total_dcf_value"] > 0, f"DCF value must be positive for {property_id}"
        assert (
            dcf["pv_terminal_value"] > 0
        ), f"DCF terminal value must be positive for {property_id}"

        assert (
            sales_comp["implied_grm"] > 0
        ), f"Sales comp GRM must be positive for {property_id}"
        assert (
            sales_comp["blended_comp_value"] > 0
        ), f"Sales comp value must be positive for {property_id}"

        assert (
            reconciliation["concluded_value"] > 0
        ), f"Concluded value must be positive for {property_id}"
        _assert_between(
            reconciliation["concluded_value"],
            reconciliation["value_range_low"],
            reconciliation["value_range_high"],
            f"Concluded value must sit inside value range for {property_id}",
        )

        scenario_table = scenario_output["summary_table"]
        assert isinstance(
            scenario_table, pd.DataFrame
        ), f"Scenario table must be DataFrame for {property_id}"
        assert len(scenario_table) == 3, (
            f"Scenario table must contain 3 rows for {property_id}"
        )

        bear_value = scenario_output[SCENARIO_BEAR]["concluded_value"]
        base_value = scenario_output[SCENARIO_BASE]["concluded_value"]
        bull_value = scenario_output[SCENARIO_BULL]["concluded_value"]
        assert bear_value <= base_value <= bull_value, (
            f"Bear/Base/Bull ordering broken for {property_id}"
        )

        bridge_total = _safe_float(bridge["Value"].sum(), default=0.0)
        assert abs(bridge_total - reconciliation["concluded_value"]) <= 1.0, (
            f"Value bridge must reconcile to within \$1 for {property_id}"
        )

        assert benchmark["benchmark_ppu"] > 0, (
            f"Benchmark PPU must be positive for {property_id}"
        )

        print(
            f"✓ {property_id:<8} | "
            f"Direct Cap: ${direct_cap['indicated_value']:,.0f} | "
            f"DCF: ${dcf['total_dcf_value']:,.0f} | "
            f"Sales Comp: ${sales_comp['blended_comp_value']:,.0f} | "
            f"Concluded: ${reconciliation['concluded_value']:,.0f}"
        )

    portfolio_table = get_valuation_summary_table()
    assert len(portfolio_table) == len(property_ids), (
        "Valuation summary table row count mismatch."
    )

    print("\nPortfolio valuation summary:")
    print(
        portfolio_table[
            [
                "Property_ID",
                "Indicated_Value",
                "Unrealized_Gain",
                "Unrealized_Gain_Pct",
                "Value_Per_Unit",
                "Implied_Cap_Rate",
                "RAG_Status",
            ]
        ].to_string(index=False)
    )

    print("\nALL VALUATION TESTS PASSED")


if __name__ == "__main__":
    _run_self_tests()