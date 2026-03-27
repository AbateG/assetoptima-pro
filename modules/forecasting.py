import os
import sys
import warnings
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from modules.data_loader import (
    load_properties,
    get_performance_for_property,
    get_underwriting_for_property,
    get_covenant_for_property,
    load_debt_covenants,
)
from modules.kpi_calculations import (
    unlevered_irr,
    levered_irr,
    equity_multiple,
    net_present_value,
    debt_service_coverage_ratio,
    fmt_currency,
    fmt_percent,
    fmt_irr,
    fmt_multiple,
    rag_to_emoji,
)

DEFAULT_RENT_GROWTH:      float = 0.035
DEFAULT_VACANCY_RATE:     float = 0.060
DEFAULT_EXPENSE_GROWTH:   float = 0.025
DEFAULT_CAPEX_PER_UNIT:   float = 300.0
DEFAULT_EXIT_CAP_RATE:    float = 0.0525
DEFAULT_HOLD_YEARS:       int   = 5
DEFAULT_DISCOUNT_RATE:    float = 0.08
DEFAULT_SELLING_COSTS_PCT: float = 0.015

# New Global Constants for Clamping
MIN_CAP_RATE:             float = 0.035
MAX_CAP_RATE:             float = 0.090

SENSITIVITY_CAP_RATE_RANGE: list[float] = [
    0.0425, 0.0475, 0.0525, 0.0575, 0.0625
]
SENSITIVITY_RENT_GROWTH_RANGE: list[float] = [
    0.010, 0.025, 0.035, 0.045, 0.055
]

IRR_STRONG_HOLD_PREMIUM:  float = 0.020
IRR_SELL_DISCOUNT:        float = 0.010
EQUITY_MULTIPLE_STRONG:   float = 1.75
EQUITY_MULTIPLE_WEAK:     float = 1.30
DSCR_REFINANCE_TRIGGER:   float = 1.50
NOI_OUTPERFORM_THRESHOLD: float = 0.05

LEASE_UP_PHASES:           list[str] = ["Lease_Up"]
RENOVATION_PHASES:         list[str] = ["Renovation_In_Progress"]
STABILIZED_PHASES:         list[str] = ["Stabilized", "Renovation_Complete"]

LEASE_UP_VACANCY_ADJUSTMENT: float = 0.08
RENOVATION_VACANCY_ADJUSTMENT: float = 0.05


def get_base_assumptions_for_property(property_id: str) -> dict:
    """
    Derive contextually appropriate forecast assumption defaults for a single
    property by combining its original underwriting assumptions with its most
    recent actual operating performance.
    """
    props    = load_properties()
    prop_row = props[props["Property_ID"] == property_id]
    if prop_row.empty:
        raise ValueError(
            f"Property '{property_id}' not found in properties dataset."
        )
    prop = prop_row.iloc[0]

    property_name  = str(prop.get("Property_Name", property_id))
    phase          = str(prop.get("Value_Add_Phase", "Stabilized"))
    
    raw_units      = prop.get("Units", 0)
    num_units      = int(raw_units) if pd.notna(raw_units) else 0
    
    raw_acq        = prop.get("Purchase_Price", 0.0)
    acq_value      = float(raw_acq) if pd.notna(raw_acq) else 0.0
    
    raw_months     = prop.get("Holding_Months", 0)
    months_held    = int(raw_months) if pd.notna(raw_months) else 0

    is_lease_up   = phase in LEASE_UP_PHASES
    is_renovation = phase in RENOVATION_PHASES

    uw     = get_underwriting_for_property(property_id)
    uw_row = uw.iloc[0] if not uw.empty else {}

    raw_rg = uw_row.get("Underwritten_Rent_Growth_Pct", DEFAULT_RENT_GROWTH) if len(uw_row) else DEFAULT_RENT_GROWTH
    uw_rent_growth = float(raw_rg) if pd.notna(raw_rg) else DEFAULT_RENT_GROWTH
    raw_eg = uw_row.get("Underwritten_Expense_Growth_Pct", DEFAULT_EXPENSE_GROWTH) if len(uw_row) else DEFAULT_EXPENSE_GROWTH
    uw_expense_growth = float(raw_eg) if pd.notna(raw_eg) else DEFAULT_EXPENSE_GROWTH
    uw_cap_rate       = float(uw_row.get("Underwritten_Purchase_Cap_Rate", DEFAULT_EXIT_CAP_RATE)) if len(uw_row) else DEFAULT_EXIT_CAP_RATE
    uw_hold_years     = int(uw_row.get("Target_Hold_Years",                DEFAULT_HOLD_YEARS))    if len(uw_row) else DEFAULT_HOLD_YEARS
    target_irr        = float(uw_row.get("Underwritten_IRR",               0.14))                  if len(uw_row) else 0.14                          if len(uw_row) else 0.14

    perf = get_performance_for_property(property_id)
    data_quality = "Full"

    if perf.empty:
        base_noi      = 0.0
        base_revenue  = 0.0
        base_expenses = 0.0
        recent_rg     = DEFAULT_RENT_GROWTH
        recent_vac    = DEFAULT_VACANCY_RATE
        data_quality  = "Estimated"
    else:
        perf_sorted   = perf.sort_values("Period")
        t12           = perf_sorted.tail(12)
        base_noi      = float(t12["Actual_NOI"].sum())
        base_revenue  = float(t12["Actual_Revenue"].sum())
        base_expenses = float(t12["Actual_Expenses"].sum())

        t6 = perf_sorted.tail(6)
        if len(t6) >= 2:
            raw_first = t6["Actual_Revenue"].iloc[0]
            first_rev = float(raw_first) if pd.notna(raw_first) else 0.0

            raw_last = t6["Actual_Revenue"].iloc[-1]
            last_rev = float(raw_last) if pd.notna(raw_last) else 0.0

            if first_rev > 0:
                six_month_growth = (last_rev / first_rev) - 1.0
                # FIXED: Proper compounding rather than multiplying by 2
                recent_rg        = ((1.0 + six_month_growth) ** 2) - 1.0
            else:
                recent_rg = DEFAULT_RENT_GROWTH
        else:
            recent_rg = DEFAULT_RENT_GROWTH
            data_quality = "Partial"

        latest = perf_sorted.iloc[-1]
        raw_occ = latest.get("Occupancy", 1.0 - DEFAULT_VACANCY_RATE)
        occ_val = float(raw_occ) if pd.notna(raw_occ) else (1.0 - DEFAULT_VACANCY_RATE)
        
        occ_decimal = occ_val if occ_val <= 1.0 else occ_val / 100.0
        recent_vac = max(0.0, 1.0 - occ_decimal)

    blended_rg = (uw_rent_growth + recent_rg) / 2.0
    blended_rg = float(np.clip(blended_rg, 0.000, 0.080))

    if is_lease_up:
        adj_vac = recent_vac + LEASE_UP_VACANCY_ADJUSTMENT
    elif is_renovation:
        adj_vac = recent_vac + RENOVATION_VACANCY_ADJUSTMENT
    else:
        adj_vac = recent_vac
    adj_vac = float(np.clip(adj_vac, 0.020, 0.200))

    if phase in STABILIZED_PHASES:
        capex_pu = DEFAULT_CAPEX_PER_UNIT * 1.50
    elif is_renovation:
        capex_pu = DEFAULT_CAPEX_PER_UNIT * 0.75
    else:
        capex_pu = DEFAULT_CAPEX_PER_UNIT

    # FIXED: Use global constants for min/max
    exit_cap = float(np.clip(uw_cap_rate + 0.0025, MIN_CAP_RATE, MAX_CAP_RATE))

    months_remaining  = max(0, (uw_hold_years * 12) - months_held)
    years_remaining   = max(1, min(10, int(np.ceil(months_remaining / 12))))

    covenant = get_covenant_for_property(property_id)
    if not covenant.empty:
        cov_row = covenant.iloc[0]

        raw_lb = cov_row.get("Loan_Balance", 0.0)
        loan_balance = float(raw_lb) if pd.notna(raw_lb) else 0.0

        raw_ads = cov_row.get("Annual_Debt_Service", 0.0)
        annual_debt_svc = float(raw_ads) if pd.notna(raw_ads) else 0.0
    else:
        loan_balance    = acq_value * 0.60
        annual_debt_svc = loan_balance * 0.055

    # FIXED: Removed the erroneous `if data_quality == "Full": data_quality = "Partial"` block here

    return {
        "property_id":         property_id,
        "property_name":       property_name,
        "phase":               phase,
        "num_units":           num_units,

        "rent_growth":         blended_rg,
        "vacancy_rate":        adj_vac,
        "expense_growth":      uw_expense_growth,
        "capex_per_unit":      capex_pu,
        "exit_cap_rate":       exit_cap,
        "hold_years":          years_remaining,

        "base_noi":            base_noi,
        "base_revenue":        base_revenue,
        "base_expenses":       base_expenses,
        "acquisition_value":   acq_value,
        "loan_balance":        loan_balance,
        "annual_debt_service": annual_debt_svc,
        "target_irr":          target_irr,

        "is_lease_up":         is_lease_up,
        "is_renovation":       is_renovation,
        "months_held":         months_held,
        "data_quality":        data_quality,
    }


def build_property_forecast(
    property_id:    str,
    rent_growth:    Optional[float] = None,
    vacancy_rate:   Optional[float] = None,
    expense_growth: Optional[float] = None,
    capex_per_unit: Optional[float] = None,
    exit_cap_rate:  Optional[float] = None,
    hold_years:     Optional[int]   = None,
) -> dict:
    """
    Build a complete multi-year pro forma cash flow model for a single
    property, anchored to trailing 12-month actual NOI as the Year 1
    starting point.
    """
    base = get_base_assumptions_for_property(property_id)

    rent_growth    = rent_growth    if rent_growth    is not None else base["rent_growth"]
    vacancy_rate   = vacancy_rate   if vacancy_rate   is not None else base["vacancy_rate"]
    expense_growth = expense_growth if expense_growth is not None else base["expense_growth"]
    capex_per_unit = capex_per_unit if capex_per_unit is not None else base["capex_per_unit"]
    exit_cap_rate  = exit_cap_rate  if exit_cap_rate  is not None else base["exit_cap_rate"]
    hold_years     = hold_years     if hold_years     is not None else base["hold_years"]

    if not (1 <= hold_years <= 10):
        raise ValueError(
            f"hold_years must be between 1 and 10. Got {hold_years}."
        )

    base_revenue  = base["base_revenue"]
    base_expenses = base["base_expenses"]
    base_noi      = base["base_noi"]

    if base_revenue == 0 and base["acquisition_value"] > 0:
        # FIXED: Reverse the 25bps expansion to approximate the entry cap rate, bounded for safety
        entry_cap_rate = max(0.01, base["exit_cap_rate"] - 0.0025)
        implied_noi   = base["acquisition_value"] * entry_cap_rate
        base_revenue  = implied_noi / 0.40   # assume 40% NOI margin
        base_expenses = base_revenue * 0.60
        base_noi      = implied_noi

    acq_value       = base["acquisition_value"]
    loan_balance    = base["loan_balance"]
    annual_debt_svc = base["annual_debt_service"]
    num_units       = base["num_units"]

    years_list:       list[int]   = []
    gpr_list:         list[float] = []
    vac_loss_list:    list[float] = []
    egi_list:         list[float] = []
    opex_list:        list[float] = []
    noi_list:         list[float] = []
    capex_list:       list[float] = []
    unlev_cf_list:    list[float] = []
    debt_svc_list:    list[float] = []
    lev_cf_list:      list[float] = []
    dscr_list:        list[float] = []

    capex_annual = capex_per_unit * num_units

    for y in range(1, hold_years + 1):
        gpr  = base_revenue  * ((1 + rent_growth)    ** y)
        opex = base_expenses * ((1 + expense_growth) ** y)
        vac  = gpr * vacancy_rate
        egi  = gpr - vac
        noi  = egi - opex

        unlev_cf = noi - capex_annual
        lev_cf   = unlev_cf - annual_debt_svc
        dscr     = (noi / annual_debt_svc) if annual_debt_svc > 0 else 999.0

        years_list.append(y)
        gpr_list.append(round(gpr, 2))
        vac_loss_list.append(round(vac, 2))
        egi_list.append(round(egi, 2))
        opex_list.append(round(opex, 2))
        noi_list.append(round(noi, 2))
        capex_list.append(round(capex_annual, 2))
        unlev_cf_list.append(round(unlev_cf, 2))
        debt_svc_list.append(round(annual_debt_svc, 2))
        lev_cf_list.append(round(lev_cf, 2))
        dscr_list.append(round(dscr, 4))

    exit_noi          = noi_list[-1]
    exit_value        = exit_noi / exit_cap_rate if exit_cap_rate > 0 else 0.0
    selling_costs     = exit_value * DEFAULT_SELLING_COSTS_PCT
    net_proceeds_unlev = exit_value - selling_costs
    net_proceeds_lev   = net_proceeds_unlev - loan_balance

    unlev_cf_exit = unlev_cf_list.copy()
    unlev_cf_exit[-1] = round(unlev_cf_exit[-1] + net_proceeds_unlev, 2)

    lev_cf_exit = lev_cf_list.copy()
    lev_cf_exit[-1] = round(lev_cf_exit[-1] + net_proceeds_lev, 2)

    equity_invested   = acq_value - loan_balance
    unlev_cf_full     = [-acq_value]     + unlev_cf_exit
    lev_cf_full       = [-equity_invested] + lev_cf_exit

    try:
        u_irr = unlevered_irr(unlev_cf_full)
    except Exception:
        u_irr = float("nan")

    try:
        l_irr = levered_irr(lev_cf_full)
    except Exception:
        l_irr = float("nan")

    total_lev_distrib  = sum(lev_cf_exit)
    total_operating_lev_cf = sum(lev_cf_list)
    
    eq_multiple = equity_multiple(
        total_distributions=total_operating_lev_cf, 
        exit_proceeds=net_proceeds_lev, 
        initial_equity=equity_invested
    ) if equity_invested > 0 else float("nan")

    try:
        npv_unlev = net_present_value(unlev_cf_full, DEFAULT_DISCOUNT_RATE)
    except Exception:
        npv_unlev = float("nan")

    target_irr    = base["target_irr"]
    irr_vs_target = (u_irr - target_irr) if not np.isnan(u_irr) else float("nan")
    beats_target  = bool(u_irr > target_irr) if not np.isnan(u_irr) else False

    assumptions_used = {
        "rent_growth":         rent_growth,
        "vacancy_rate":        vacancy_rate,
        "expense_growth":      expense_growth,
        "capex_per_unit":      capex_per_unit,
        "exit_cap_rate":       exit_cap_rate,
        "hold_years":          hold_years,
        "base_noi":            base_noi,
        "base_revenue":        base_revenue,
        "base_expenses":       base_expenses,
        "acquisition_value":   acq_value,
        "loan_balance":        loan_balance,
        "annual_debt_service": annual_debt_svc,
    }

    return {
        "property_id":    property_id,
        "property_name":  base["property_name"],
        "num_units":      num_units,
        "phase":          base["phase"],
        "data_quality":   base["data_quality"],

        "assumptions":    assumptions_used,

        "years":              years_list,
        "gpr":                gpr_list,
        "vacancy_loss":       vac_loss_list,
        "egi":                egi_list,
        "operating_expenses": opex_list,
        "noi":                noi_list,
        "capex_reserve":      capex_list,
        "unlevered_cf":       unlev_cf_list,
        "debt_service":       debt_svc_list,
        "levered_cf":         lev_cf_list,
        "dscr":               dscr_list,

        "exit_noi":                      round(exit_noi, 2),
        "exit_value":                    round(exit_value, 2),
        "selling_costs":                 round(selling_costs, 2),
        "net_sale_proceeds_unlevered":   round(net_proceeds_unlev, 2),
        "net_sale_proceeds_levered":     round(net_proceeds_lev, 2),

        "unlevered_cf_with_exit": unlev_cf_exit,
        "levered_cf_with_exit":   lev_cf_exit,

        "equity_invested":    round(equity_invested, 2),
        "unlevered_irr":      round(u_irr, 6)  if not np.isnan(u_irr) else float("nan"),
        "levered_irr":        round(l_irr, 6)  if not np.isnan(l_irr) else float("nan"),
        "equity_multiple":    round(eq_multiple, 4) if not np.isnan(eq_multiple) else float("nan"),
        "npv_unlevered":      round(npv_unlev, 2)   if not np.isnan(npv_unlev)   else float("nan"),
        "total_unlevered_cf": round(sum(unlev_cf_exit), 2),
        "total_levered_cf":   round(total_lev_distrib, 2),

        "target_irr":     target_irr,
        "irr_vs_target":  round(irr_vs_target, 6) if not np.isnan(irr_vs_target) else float("nan"),
        "beats_target":   beats_target,
    }


def build_irr_sensitivity_matrix(
    property_id:        str,
    base_rent_growth:   Optional[float] = None,
    base_vacancy_rate:  Optional[float] = None,
    base_expense_growth: Optional[float] = None,
    base_capex_per_unit: Optional[float] = None,
    base_hold_years:    Optional[int]   = None,
    cap_rate_range:     Optional[list]  = None,
    rent_growth_range:  Optional[list]  = None,
) -> pd.DataFrame:
    """
    Build a 5x5 IRR sensitivity matrix varying exit cap rate (rows) against
    rent growth rate (columns) for a single property.
    """
    cap_rates    = cap_rate_range    or SENSITIVITY_CAP_RATE_RANGE
    rent_growths = rent_growth_range or SENSITIVITY_RENT_GROWTH_RANGE

    row_labels = [f"{r*100:.2f}%" for r in cap_rates]
    col_labels = [f"{g*100:.1f}%" for g in rent_growths]

    matrix: list[list] = []

    for cap in cap_rates:
        row_vals: list = []
        for rg in rent_growths:
            try:
                fc = build_property_forecast(
                    property_id     = property_id,
                    rent_growth     = rg,
                    vacancy_rate    = base_vacancy_rate,
                    expense_growth  = base_expense_growth,
                    capex_per_unit  = base_capex_per_unit,
                    exit_cap_rate   = cap,
                    hold_years      = base_hold_years,
                )
                irr_val = fc["unlevered_irr"]
                if not np.isnan(irr_val) and -0.50 <= irr_val <= 1.00:
                    row_vals.append(round(irr_val, 6))
                else:
                    row_vals.append(float("nan"))
            except Exception:
                row_vals.append(float("nan"))
        matrix.append(row_vals)

    df = pd.DataFrame(matrix, index=row_labels, columns=col_labels)
    df.index.name   = "Exit_Cap_Rate"
    df.columns.name = "Rent_Growth"

    return df


def compare_forecast_vs_underwriting(
    property_id: str,
    forecast: Optional[dict] = None,
) -> dict:
    """
    Compare Year 1 forecast outputs against the original underwriting
    assumptions for a single property.
    """
    if forecast is None:
        forecast = build_property_forecast(property_id)

    uw     = get_underwriting_for_property(property_id)
    uw_row = uw.iloc[0] if not uw.empty else {}

    raw_noi = uw_row.get("Underwritten_NOI_Year1", 0.0) if len(uw_row) else 0.0
    uw_noi = float(raw_noi) if pd.notna(raw_noi) else 0.0

    raw_occ = uw_row.get("Underwritten_Occupancy", 0.0) if len(uw_row) else 0.0
    uw_occ = float(raw_occ) if pd.notna(raw_occ) else 0.0

    raw_cap = uw_row.get("Underwritten_Purchase_Cap_Rate", 0.0) if len(uw_row) else 0.0
    uw_cap = float(raw_cap) if pd.notna(raw_cap) else 0.0

    raw_irr = uw_row.get("Underwritten_IRR", 0.0) if len(uw_row) else 0.0
    target_irr = float(raw_irr) if pd.notna(raw_irr) else 0.0
    
    uw_cap_pct  = uw_cap 

    perf       = get_performance_for_property(property_id)
    t12_noi = float(perf.sort_values("Period").tail(12)["Actual_NOI"].sum()) \
          if not perf.empty else 0.0

    fc_y1_noi  = forecast["noi"][0] if forecast["noi"] else 0.0
    fc_irr     = forecast["unlevered_irr"]
    fc_eq_mult = forecast["equity_multiple"]
    fc_exit    = forecast["exit_value"]
    acq_value  = forecast["assumptions"]["acquisition_value"]
    vac_rate   = forecast["assumptions"]["vacancy_rate"]

    def _pct_var_decimal(a: float, b: float) -> float:
        return ((a - b) / abs(b)) if b != 0 else 0.0

    actual_vs_uw_pct     = _pct_var_decimal(t12_noi,  uw_noi)
    forecast_vs_uw_pct   = _pct_var_decimal(fc_y1_noi, uw_noi)
    forecast_vs_actual   = _pct_var_decimal(fc_y1_noi, t12_noi)
    
    irr_vs_target_ppt    = (fc_irr - target_irr) if not np.isnan(fc_irr) else float("nan")
    irr_beats            = bool(fc_irr > target_irr) if not np.isnan(fc_irr) else False
    
    proj_appreciation    = fc_exit - acq_value
    proj_apprec_pct      = _pct_var_decimal(fc_exit, acq_value)
    
    implied_occ          = 1.0 - vac_rate

    return {
        "property_id":                   property_id,

        "underwritten_year1_noi":        uw_noi,
        "actual_t12_noi":                t12_noi,
        "forecast_year1_noi":            fc_y1_noi,
        
        "actual_vs_uw_pct":              actual_vs_uw_pct,
        "forecast_vs_uw_pct":            forecast_vs_uw_pct,
        "forecast_vs_actual_pct":        forecast_vs_actual,

        "target_irr":                    target_irr,
        "forecast_unlevered_irr":        fc_irr,
        "irr_vs_target_ppt":             irr_vs_target_ppt,
        "irr_beats_target":              irr_beats,

        "implied_acquisition_cap_rate":  uw_cap_pct,
        "forecast_exit_value":           round(fc_exit, 2),
        "acquisition_value":             acq_value,
        "projected_appreciation":        round(proj_appreciation, 2),
        "projected_appreciation_pct":    proj_apprec_pct,

        "forecast_equity_multiple":      fc_eq_mult,

        "underwritten_occupancy_pct":    uw_occ,
        "forecast_vacancy_rate_pct":     vac_rate,
        "implied_forecast_occupancy":    implied_occ,

        "data_quality":                  forecast["data_quality"],
    }


def calculate_annual_debt_service(
    purchase_price: float, 
    ltv: float, 
    annual_rate: float, 
    amort_years: int = 30
) -> float:
    """
    Calculates the annual principal and interest payment (Debt Service).
    Returns 0.0 if no loan exists.
    """
    if ltv <= 0 or annual_rate <= 0:
        return 0.0
        
    loan_amount = purchase_price * ltv
    monthly_rate = annual_rate / 12
    num_payments = amort_years * 12
    
    monthly_payment = (
        loan_amount * (monthly_rate * (1 + monthly_rate) ** num_payments) / 
        ((1 + monthly_rate) ** num_payments - 1)
    )
    
    return monthly_payment * 12


def build_portfolio_forecast_summary(
    assumption_overrides: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Build a one-row-per-property forecast summary for the entire portfolio.
    """
    props    = load_properties()
    overrides = assumption_overrides or {}
    rows: list[dict] = []

    for _, prop in props.iterrows():
        pid    = str(prop["Property_ID"])
        name   = str(prop["Property_Name"])
                
        raw_phase = prop.get("Value_Add_Phase", prop.get("Phase", ""))
        phase  = str(raw_phase) if pd.notna(raw_phase) else "Stabilized"
                
        raw_units = prop.get("Units", prop.get("Total_Units", 0))
        units  = int(raw_units) if pd.notna(raw_units) else 0

        prop_overrides = overrides.get(pid, {})

        try:
            fc = build_property_forecast(property_id=pid, **prop_overrides)
            reco = get_hold_sell_recommendation(
                property_id    = pid,
                forecast       = fc,
            )

            base_noi  = fc["assumptions"]["base_noi"]
            y1_noi    = fc["noi"][0] if fc["noi"] else 0.0
            y1_growth = round(((y1_noi - base_noi) / abs(base_noi) * 100), 2) \
                        if base_noi != 0 else 0.0
            dscr_y1   = fc["dscr"][0] if fc["dscr"] else float("nan")

            rows.append({
                "Property_ID":              pid,
                "Property_Name":            name,
                "Phase":                    phase,
                "Num_Units":                units,
                "Base_NOI":                 round(base_noi, 2),
                "Y1_Projected_NOI":         round(y1_noi, 2),
                "Y1_NOI_Growth_Pct":        y1_growth,
                "Exit_Value":               fc["exit_value"],
                "Equity_Multiple":          fc["equity_multiple"],
                "Unlevered_IRR":            fc["unlevered_irr"],
                "Target_IRR":               fc["target_irr"],
                "Beats_Target":             fc["beats_target"],
                "IRR_vs_Target_Ppt":        round(fc["irr_vs_target"] * 100, 2)
                                            if not np.isnan(fc["irr_vs_target"]) else float("nan"),
                "Hold_Years":               fc["assumptions"]["hold_years"],
                "Exit_Cap_Rate":            fc["assumptions"]["exit_cap_rate"],
                "Rent_Growth_Assumption":   fc["assumptions"]["rent_growth"],
                "DSCR_Y1":                  round(dscr_y1, 4),
                "Data_Quality":             fc["data_quality"],
                "Hold_Recommendation":      reco["recommendation"],
            })

        except Exception as exc:
            rows.append({
                "Property_ID":              pid,
                "Property_Name":            name,
                "Phase":                    phase,
                "Num_Units":                units,
                "Base_NOI":                 float("nan"),
                "Y1_Projected_NOI":         float("nan"),
                "Y1_NOI_Growth_Pct":        float("nan"),
                "Exit_Value":               float("nan"),
                "Equity_Multiple":          float("nan"),
                "Unlevered_IRR":            float("nan"),
                "Target_IRR":               float("nan"),
                "Beats_Target":             False,
                "IRR_vs_Target_Ppt":        float("nan"),
                "Hold_Years":               DEFAULT_HOLD_YEARS,
                "Exit_Cap_Rate":            DEFAULT_EXIT_CAP_RATE,
                "Rent_Growth_Assumption":   DEFAULT_RENT_GROWTH,
                "DSCR_Y1":                  float("nan"),
                "Data_Quality":             "Error",
                "Hold_Recommendation":      f"Error: {exc}",
            })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = (
            df.sort_values("Unlevered_IRR", ascending=False)
              .reset_index(drop=True)
        )

    return df


def get_hold_sell_recommendation(
    property_id: str,
    forecast:    Optional[dict] = None,
) -> dict:
    """
    Generate a rule-based hold, sell, refinance, or watch recommendation
    for a single property based on its forecast return metrics, debt
    coverage, NOI trend, and remaining hold period.
    """
    if forecast is None:
        forecast = build_property_forecast(property_id)

    u_irr        = forecast["unlevered_irr"]
    target_irr   = forecast["target_irr"]
    eq_mult      = forecast["equity_multiple"]
    hold_years   = forecast["assumptions"]["hold_years"]
    data_quality = forecast["data_quality"]
    prop_name    = forecast["property_name"]
    dscr_y1      = forecast["dscr"][0] if forecast["dscr"] else float("nan")

    base   = get_base_assumptions_for_property(property_id)
    uw     = get_underwriting_for_property(property_id)
    uw_row = uw.iloc[0] if not uw.empty else {}
    uw_cap = float(uw_row.get("Purchase_Cap_Rate", 0)) if len(uw_row) else 0.0
    cur_cap = forecast["assumptions"]["exit_cap_rate"]
    
    cap_compression_bps = (uw_cap - cur_cap) * 10_000   
    
    irr_available     = not np.isnan(u_irr)
    irr_vs_target     = (u_irr - target_irr) if irr_available else float("nan")
    irr_vs_target_bps = (u_irr - target_irr) * 10_000 if irr_available else float("nan")

    rationale:      list[str] = []
    signals_sell:   int       = 0
    signals_refi:   int       = 0
    signals_hold:   int       = 0
    signals_watch:  int       = 0

    if irr_available and not np.isnan(eq_mult):
        if (u_irr < (target_irr - IRR_SELL_DISCOUNT)) and (eq_mult < EQUITY_MULTIPLE_WEAK):
            signals_sell += 1
            rationale.append(
                f"SELL SIGNAL: IRR {fmt_irr(u_irr)} is {abs(irr_vs_target_bps):.0f}bps "
                f"below target {fmt_irr(target_irr)} and equity multiple "
                f"{fmt_multiple(eq_mult)} is below {EQUITY_MULTIPLE_WEAK:.2f}x threshold."
            )

    if hold_years <= 1 and irr_available and u_irr > target_irr:
        signals_sell += 1
        rationale.append(
            f"SELL SIGNAL: Hold period is {hold_years} year(s) remaining. "
            f"IRR beats target — disciplined exit recommended to lock in returns."
        )

    if not np.isnan(dscr_y1) and dscr_y1 < 1.10:
        signals_sell += 1
        rationale.append(
            f"SELL SIGNAL: Year 1 projected DSCR {dscr_y1:.2f}x is critically low "
            f"(below 1.10x). Debt coverage at risk."
        )

    if (
        not np.isnan(dscr_y1)
        and dscr_y1 >= DSCR_REFINANCE_TRIGGER
        and irr_available
        and irr_vs_target >= IRR_STRONG_HOLD_PREMIUM
    ):
        signals_refi += 1
        rationale.append(
            f"REFINANCE SIGNAL: DSCR {dscr_y1:.2f}x exceeds {DSCR_REFINANCE_TRIGGER:.2f}x "
            f"trigger and IRR beats target by "
            f"{irr_vs_target_bps:.0f}bps. Strong candidate to pull forward equity."
        )

    if cap_compression_bps > 50 and hold_years > 2:
        signals_refi += 1
        rationale.append(
            f"REFINANCE SIGNAL: Cap rate has compressed "
            f"{cap_compression_bps:.0f}bps vs purchase cap rate. "
            f"With {hold_years} year(s) remaining, refinancing now crystallises embedded gains."
        )

    if irr_available and not np.isnan(eq_mult):
        if irr_vs_target >= IRR_STRONG_HOLD_PREMIUM and eq_mult >= EQUITY_MULTIPLE_STRONG:
            signals_hold += 2   
            rationale.append(
                f"STRONG HOLD SIGNAL: IRR {fmt_irr(u_irr)} beats target by "
                f"{irr_vs_target_bps:.0f}bps and equity multiple "
                f"{fmt_multiple(eq_mult)} exceeds {EQUITY_MULTIPLE_STRONG:.2f}x."
            )
        elif u_irr > target_irr:
            signals_hold += 1
            rationale.append(
                f"HOLD SIGNAL: IRR {fmt_irr(u_irr)} exceeds target {fmt_irr(target_irr)} "
                f"by {irr_vs_target_bps:.0f}bps. Continue executing business plan."
            )

    if data_quality == "Estimated" or not irr_available:
        signals_watch += 1
        rationale.append(
            "WATCH NOTE: Forecast is based on estimated or incomplete data. "
            "Validate actuals before presenting to investment committee."
        )
    
    if signals_sell > 0 and signals_hold > 0:
        recommendation = "Watch (Risk Alert)"
        confidence     = "Low"
        rationale.insert(0, "⚠️ CONFLICTING SIGNALS: Asset shows strong theoretical returns but has severe underlying risk (e.g., debt coverage). Requires manual analyst review.")
    
    elif signals_sell >= 2:
        recommendation = "Sell"
        confidence     = "High"
    elif signals_sell == 1:
        recommendation = "Sell"
        confidence     = "Medium"
        
    elif signals_refi >= 2:
        recommendation = "Refinance"
        confidence     = "High"
    elif signals_refi == 1 and signals_sell == 0:
        recommendation = "Refinance"
        confidence     = "Medium"
        
    elif signals_hold >= 2:
        recommendation = "Strong Hold"
        confidence     = "High"
    elif signals_hold == 1:
        recommendation = "Hold"
        confidence     = "Medium" if signals_watch == 0 else "Low"
        
    else:
        recommendation = "Watch"
        confidence     = "Low" if (signals_watch > 0 or not irr_available) else "Medium"

    if len(rationale) == 0:
        rationale.append(
            "WATCH: No strong directional signals. Monitor performance against "
            "business plan and reassess at next quarterly review."
        )

    key_metrics = {
        "unlevered_irr":     u_irr,
        "target_irr":        target_irr,
        "equity_multiple":   eq_mult,
        "dscr_y1":           dscr_y1,
        "hold_years":        hold_years,
        "irr_vs_target_ppt": irr_vs_target,
    }

    return {
        "property_id":    property_id,
        "property_name":  prop_name,
        "recommendation": recommendation,
        "rationale":      rationale,
        "confidence":     confidence,
        "key_metrics":    key_metrics,
    }

def get_forecast_as_dataframe(forecast: dict) -> pd.DataFrame:
    """
    Convert the year-by-year arrays from a build_property_forecast() output
    dict into a structured DataFrame for display in tables and charts.
    """
    n = len(forecast["years"])
    if n == 0:
        return pd.DataFrame()

    df = pd.DataFrame({
        "Year":                  forecast["years"],
        "GPR":                   forecast["gpr"],
        "Vacancy_Loss":          forecast["vacancy_loss"],
        "EGI":                   forecast["egi"],
        "Operating_Expenses":    forecast["operating_expenses"],
        "NOI":                   forecast["noi"],
        "CapEx_Reserve":         forecast["capex_reserve"],
        "Unlevered_CF":          forecast["unlevered_cf"],
        "Debt_Service":          forecast["debt_service"],
        "Levered_CF":            forecast["levered_cf"],
        "DSCR":                  forecast["dscr"],
        "Unlevered_CF_With_Exit": forecast["unlevered_cf_with_exit"],
        "Levered_CF_With_Exit":   forecast["levered_cf_with_exit"],
    })

    return df.reset_index(drop=True)


def get_sensitivity_matrix_display(matrix_df: pd.DataFrame) -> pd.DataFrame:
    """
    Format a raw IRR sensitivity matrix (decimals) as a display-ready
    DataFrame with percentage strings, suitable for direct rendering in
    a Streamlit st.dataframe() call or as a Plotly heatmap annotation.
    """
    def _fmt(v):
        if pd.isna(v):
            return "N/A"
        return f"{v * 100:.2f}%"

    return matrix_df.applymap(_fmt)


if __name__ == "__main__":
    """
    Run with:  python modules/forecasting.py
    All tests must pass before this module is consumed by any page or
    downstream module.
    """

    PASS = "✓"
    FAIL = "✗"
    errors:  list[str]                  = []
    results: list[tuple[str, str, str]] = []

    def _check(condition: bool, test_name: str, detail: str = "") -> None:
        if condition:
            results.append((PASS, test_name, detail))
        else:
            results.append((FAIL, test_name, detail))
            errors.append(test_name)

    print("\n" + "=" * 68)
    print("  AssetOptima Pro — forecasting.py  SELF-TEST")
    print("=" * 68 + "\n")

    PROPERTIES = ["PROP001", "PROP002", "PROP003", "PROP004", "PROP005"]

    VALID_RECOMMENDATIONS = {
        "Strong Hold", "Hold", "Watch", "Refinance", "Sell"
    }
    VALID_CONFIDENCE = {"High", "Medium", "Low"}
    VALID_DATA_QUALITY = {"Full", "Partial", "Estimated", "Error"}

    print("─── Section 1: get_base_assumptions_for_property ───\n")

    required_base_keys = {
        "property_id", "property_name", "phase", "num_units",
        "rent_growth", "vacancy_rate", "expense_growth",
        "capex_per_unit", "exit_cap_rate", "hold_years",
        "base_noi", "base_revenue", "base_expenses",
        "acquisition_value", "loan_balance", "annual_debt_service",
        "target_irr", "is_lease_up", "is_renovation",
        "months_held", "data_quality",
    }

    for pid in PROPERTIES:
        try:
            base = get_base_assumptions_for_property(pid)

            missing = required_base_keys - set(base.keys())
            _check(
                len(missing) == 0,
                f"{pid} all required keys present in base assumptions",
                str(missing) if missing else "OK",
            )

            _check(
                base["property_id"] == pid,
                f"{pid} property_id field matches",
                base["property_id"],
            )

            _check(
                0.0 <= base["rent_growth"] <= 0.08,
                f"{pid} rent_growth clamped to [0%, 8%]",
                f"{base['rent_growth']*100:.2f}%",
            )

            _check(
                0.02 <= base["vacancy_rate"] <= 0.20,
                f"{pid} vacancy_rate clamped to [2%, 20%]",
                f"{base['vacancy_rate']*100:.2f}%",
            )

            _check(
                MIN_CAP_RATE <= base["exit_cap_rate"] <= MAX_CAP_RATE,
                f"{pid} exit_cap_rate clamped to bounds",
                f"{base['exit_cap_rate']*100:.3f}%",
            )

            _check(
                1 <= base["hold_years"] <= 10,
                f"{pid} hold_years in [1, 10]",
                str(base["hold_years"]),
            )

            _check(
                base["num_units"] > 0,
                f"{pid} num_units > 0",
                str(base["num_units"]),
            )

            _check(
                base["acquisition_value"] > 0,
                f"{pid} acquisition_value > 0",
                fmt_currency(base["acquisition_value"]),
            )

            _check(
                base["capex_per_unit"] > 0,
                f"{pid} capex_per_unit > 0",
                f"${base['capex_per_unit']:.0f}",
            )

            _check(
                isinstance(base["is_lease_up"], bool),
                f"{pid} is_lease_up is bool",
                str(base["is_lease_up"]),
            )

            _check(
                isinstance(base["is_renovation"], bool),
                f"{pid} is_renovation is bool",
                str(base["is_renovation"]),
            )

            _check(
                base["data_quality"] in VALID_DATA_QUALITY,
                f"{pid} data_quality is valid",
                base["data_quality"],
            )

        except Exception as exc:
            _check(
                False,
                f"{pid} get_base_assumptions_for_property raised exception",
                str(exc),
            )

    try:
        base2 = get_base_assumptions_for_property("PROP002")
        _check(
            base2["is_lease_up"] is True,
            "PROP002 flagged as is_lease_up=True",
            f"phase={base2['phase']}",
        )
        _check(
            base2["vacancy_rate"] > DEFAULT_VACANCY_RATE,
            "PROP002 vacancy_rate elevated above default due to Lease_Up phase",
            f"{base2['vacancy_rate']*100:.2f}% vs default {DEFAULT_VACANCY_RATE*100:.1f}%",
        )
    except Exception as exc:
        _check(False, "PROP002 lease-up vacancy check", str(exc))

    try:
        base4 = get_base_assumptions_for_property("PROP004")
        _check(
            base4["is_renovation"] is True,
            "PROP004 flagged as is_renovation=True",
            f"phase={base4['phase']}",
        )
    except Exception as exc:
        _check(False, "PROP004 renovation flag check", str(exc))

    try:
        get_base_assumptions_for_property("PROPXXX")
        _check(False, "ValueError raised for unknown property_id in base assumptions", "No exception")
    except ValueError:
        _check(True, "ValueError raised for unknown property_id in base assumptions", "ValueError raised")
    except Exception as exc:
        _check(False, "ValueError raised for unknown property_id in base assumptions", str(exc))

    print()

    print("─── Section 2: build_property_forecast ───\n")

    required_forecast_keys = {
        "property_id", "property_name", "num_units", "phase",
        "data_quality", "assumptions",
        "years", "gpr", "vacancy_loss", "egi", "operating_expenses",
        "noi", "capex_reserve", "unlevered_cf", "debt_service",
        "levered_cf", "dscr",
        "exit_noi", "exit_value", "selling_costs",
        "net_sale_proceeds_unlevered", "net_sale_proceeds_levered",
        "unlevered_cf_with_exit", "levered_cf_with_exit",
        "equity_invested", "unlevered_irr", "levered_irr",
        "equity_multiple", "npv_unlevered",
        "total_unlevered_cf", "total_levered_cf",
        "target_irr", "irr_vs_target", "beats_target",
    }

    for pid in PROPERTIES:
        try:
            fc = build_property_forecast(pid)

            missing = required_forecast_keys - set(fc.keys())
            _check(
                len(missing) == 0,
                f"{pid} all required keys present in forecast",
                str(missing) if missing else "OK",
            )

            hold = fc["assumptions"]["hold_years"]

            for arr_name in ["years", "gpr", "noi", "dscr", "levered_cf"]:
                _check(
                    len(fc[arr_name]) == hold,
                    f"{pid} {arr_name} array length == hold_years ({hold})",
                    str(len(fc[arr_name])),
                )

            _check(
                fc["years"] == list(range(1, hold + 1)),
                f"{pid} years array is [1..{hold}]",
                str(fc["years"]),
            )

            if len(fc["gpr"]) >= 2:
                gpr_increasing = all(
                    fc["gpr"][i] >= fc["gpr"][i - 1]
                    for i in range(1, len(fc["gpr"]))
                )
                _check(
                    gpr_increasing,
                    f"{pid} GPR is non-decreasing across forecast years",
                    f"Y1={fmt_currency(fc['gpr'][0])} Y{hold}={fmt_currency(fc['gpr'][-1])}",
                )

            _check(
                all(n > 0 for n in fc["noi"]),
                f"{pid} all projected NOI values are positive",
                f"min NOI={fmt_currency(min(fc['noi']))}",
            )

            _check(
                fc["exit_value"] > 0,
                f"{pid} exit_value > 0",
                fmt_currency(fc["exit_value"]),
            )

            cap  = fc["assumptions"]["exit_cap_rate"]
            implied_exit = fc["exit_noi"] / cap if cap > 0 else 0
            _check(
                abs(fc["exit_value"] - implied_exit) < 1.0,
                f"{pid} exit_value = exit_noi / exit_cap_rate (within $1)",
                f"{fmt_currency(fc['exit_value'])} vs {fmt_currency(implied_exit)}",
            )

            if fc["data_quality"] == "Full":
                _check(
                    not np.isnan(fc["unlevered_irr"]),
                    f"{pid} unlevered_irr is not NaN (Full data quality)",
                    fmt_irr(fc["unlevered_irr"]),
                )

            if fc["equity_invested"] > 0 and not np.isnan(fc["equity_multiple"]):
                implied_mult = fc["total_levered_cf"] / fc["equity_invested"]
                _check(
                    abs(fc["equity_multiple"] - implied_mult) < 0.01,
                    f"{pid} equity_multiple is internally consistent",
                    f"{fc['equity_multiple']:.4f}x vs implied {implied_mult:.4f}x",
                )

            for i in range(hold - 1):
                _check(
                    fc["unlevered_cf"][i] == fc["unlevered_cf_with_exit"][i],
                    f"{pid} unlevered_cf_with_exit Year {i+1} == base CF",
                    f"{fc['unlevered_cf'][i]} == {fc['unlevered_cf_with_exit'][i]}",
                )

            if not np.isnan(fc["unlevered_irr"]):
                expected_beats = fc["unlevered_irr"] > fc["target_irr"]
                _check(
                    fc["beats_target"] == expected_beats,
                    f"{pid} beats_target is consistent with IRR vs target",
                    f"IRR={fmt_irr(fc['unlevered_irr'])} target={fmt_irr(fc['target_irr'])}",
                )

        except Exception as exc:
            _check(False, f"{pid} build_property_forecast raised exception", str(exc))

    try:
        fc3 = build_property_forecast("PROP003")
        _check(
            fc3["beats_target"] is True,
            "PROP003 (portfolio star) beats target IRR",
            f"IRR={fmt_irr(fc3['unlevered_irr'])} target={fmt_irr(fc3['target_irr'])}",
        )
    except Exception as exc:
        _check(False, "PROP003 beats target IRR check", str(exc))

    try:
        fc3 = build_property_forecast("PROP003")
        fc4 = build_property_forecast("PROP004")
        if not np.isnan(fc3["unlevered_irr"]) and not np.isnan(fc4["unlevered_irr"]):
            _check(
                fc3["unlevered_irr"] > fc4["unlevered_irr"],
                "PROP003 IRR > PROP004 IRR (star outperforms distressed)",
                f"PROP003={fmt_irr(fc3['unlevered_irr'])} "
                f"PROP004={fmt_irr(fc4['unlevered_irr'])}",
            )
    except Exception as exc:
        _check(False, "PROP003 vs PROP004 IRR comparison", str(exc))

    try:
        fc2 = build_property_forecast("PROP002")
        fc3 = build_property_forecast("PROP003")
        _check(
            fc2["assumptions"]["vacancy_rate"] > fc3["assumptions"]["vacancy_rate"],
            "PROP002 (Lease_Up) vacancy > PROP003 (Stabilized) vacancy",
            f"PROP002={fc2['assumptions']['vacancy_rate']*100:.1f}% "
            f"PROP003={fc3['assumptions']['vacancy_rate']*100:.1f}%",
        )
    except Exception as exc:
        _check(False, "PROP002 vs PROP003 vacancy comparison", str(exc))

    try:
        fc_base     = build_property_forecast("PROP003")
        fc_override = build_property_forecast("PROP003", rent_growth=0.06)
        _check(
            fc_override["gpr"][-1] > fc_base["gpr"][-1],
            "Explicit rent_growth=0.06 override increases terminal GPR vs default",
            f"override={fmt_currency(fc_override['gpr'][-1])} "
            f"base={fmt_currency(fc_base['gpr'][-1])}",
        )
    except Exception as exc:
        _check(False, "rent_growth override test", str(exc))

    for hold_test in [1, 10]:
        try:
            fc_h = build_property_forecast("PROP001", hold_years=hold_test)
            _check(
                len(fc_h["years"]) == hold_test,
                f"hold_years={hold_test} produces correct array length",
                str(len(fc_h["years"])),
            )
        except Exception as exc:
            _check(False, f"hold_years={hold_test} boundary test", str(exc))

    for bad_hold in [0, 11]:
        try:
            build_property_forecast("PROP001", hold_years=bad_hold)
            _check(
                False,
                f"ValueError raised for hold_years={bad_hold}",
                "No exception",
            )
        except ValueError:
            _check(
                True,
                f"ValueError raised for hold_years={bad_hold}",
                "ValueError raised",
            )
        except Exception as exc:
            _check(False, f"ValueError raised for hold_years={bad_hold}", str(exc))

    print()

    print("─── Section 3: build_irr_sensitivity_matrix ───\n")

    for pid in PROPERTIES:
        try:
            mat = build_irr_sensitivity_matrix(pid)

            _check(
                isinstance(mat, pd.DataFrame),
                f"{pid} sensitivity matrix is DataFrame",
                "",
            )

            _check(
                mat.shape == (5, 5),
                f"{pid} sensitivity matrix shape is (5, 5)",
                str(mat.shape),
            )

            _check(
                mat.index.name == "Exit_Cap_Rate",
                f"{pid} matrix index named Exit_Cap_Rate",
                str(mat.index.name),
            )

            _check(
                mat.columns.name == "Rent_Growth",
                f"{pid} matrix columns named Rent_Growth",
                str(mat.columns.name),
            )

            numeric_ok = all(
                pd.api.types.is_float_dtype(mat[c]) for c in mat.columns
            )
            _check(
                numeric_ok,
                f"{pid} all matrix values are float dtype",
                "",
            )

            non_nan_count = mat.notna().sum().sum()
            _check(
                non_nan_count >= 15,
                f"{pid} at least 15 of 25 IRR cells are non-NaN",
                f"{non_nan_count}/25 non-NaN",
            )

            mid_row = mat.iloc[2]
            non_nan_vals = mid_row.dropna().values
            if len(non_nan_vals) >= 2:
                _check(
                    bool(non_nan_vals[-1] > non_nan_vals[0]),
                    f"{pid} higher rent growth → higher IRR (middle row)",
                    f"low_rg={non_nan_vals[0]*100:.2f}% "
                    f"high_rg={non_nan_vals[-1]*100:.2f}%",
                )

            mid_col = mat.iloc[:, 2]
            non_nan_cap = mid_col.dropna().values
            if len(non_nan_cap) >= 2:
                _check(
                    bool(non_nan_cap[0] > non_nan_cap[-1]),
                    f"{pid} lower exit cap rate → higher IRR (middle column)",
                    f"low_cap={non_nan_cap[0]*100:.2f}% "
                    f"high_cap={non_nan_cap[-1]*100:.2f}%",
                )

        except Exception as exc:
            _check(
                False,
                f"{pid} build_irr_sensitivity_matrix raised exception",
                str(exc),
            )

    try:
        custom_caps  = [0.04, 0.05, 0.06, 0.07, 0.08]
        custom_growths = [0.00, 0.02, 0.04, 0.06, 0.08]
        mat_custom = build_irr_sensitivity_matrix(
            "PROP003",
            cap_rate_range=custom_caps,
            rent_growth_range=custom_growths,
        )
        _check(
            mat_custom.shape == (5, 5),
            "Custom cap/growth ranges produce (5,5) matrix",
            str(mat_custom.shape),
        )
    except Exception as exc:
        _check(False, "Custom sensitivity range test", str(exc))

    print()

    print("─── Section 4: compare_forecast_vs_underwriting ───\n")

    required_compare_keys = {
        "property_id",
        "underwritten_year1_noi", "actual_t12_noi", "forecast_year1_noi",
        "actual_vs_uw_pct", "forecast_vs_uw_pct", "forecast_vs_actual_pct",
        "target_irr", "forecast_unlevered_irr",
        "irr_vs_target_ppt", "irr_beats_target",
        "implied_acquisition_cap_rate", "forecast_exit_value",
        "acquisition_value", "projected_appreciation",
        "projected_appreciation_pct", "forecast_equity_multiple",
        "underwritten_occupancy_pct", "forecast_vacancy_rate_pct",
        "implied_forecast_occupancy", "data_quality",
    }

    for pid in PROPERTIES:
        try:
            comp = compare_forecast_vs_underwriting(pid)

            missing = required_compare_keys - set(comp.keys())
            _check(
                len(missing) == 0,
                f"{pid} all required comparison keys present",
                str(missing) if missing else "OK",
            )

            _check(
                comp["property_id"] == pid,
                f"{pid} property_id field correct",
                "",
            )

            _check(
                comp["underwritten_year1_noi"] > 0,
                f"{pid} underwritten_year1_noi > 0",
                fmt_currency(comp["underwritten_year1_noi"]),
            )

            _check(
                comp["actual_t12_noi"] > 0,
                f"{pid} actual_t12_noi > 0",
                fmt_currency(comp["actual_t12_noi"]),
            )

            _check(
                comp["forecast_year1_noi"] > 0,
                f"{pid} forecast_year1_noi > 0",
                fmt_currency(comp["forecast_year1_noi"]),
            )

            _check(
                comp["forecast_exit_value"] > comp["acquisition_value"] * 0.5,
                f"{pid} forecast_exit_value is plausible (> 50% of acquisition)",
                f"{fmt_currency(comp['forecast_exit_value'])} vs "
                f"acq {fmt_currency(comp['acquisition_value'])}",
            )

            _check(
                0 < comp["implied_forecast_occupancy"] <= 100,
                f"{pid} implied_forecast_occupancy in (0, 100]",
                f"{comp['implied_forecast_occupancy']:.1f}%",
            )

            _check(
                isinstance(comp["irr_beats_target"], bool),
                f"{pid} irr_beats_target is bool",
                str(comp["irr_beats_target"]),
            )

        except Exception as exc:
            _check(
                False,
                f"{pid} compare_forecast_vs_underwriting raised exception",
                str(exc),
            )

    try:
        fc3   = build_property_forecast("PROP003")
        comp3 = compare_forecast_vs_underwriting("PROP003", forecast=fc3)
        _check(
            comp3["forecast_year1_noi"] == fc3["noi"][0],
            "Pre-built forecast Y1 NOI matches comparison output",
            f"{fmt_currency(comp3['forecast_year1_noi'])} == "
            f"{fmt_currency(fc3['noi'][0])}",
        )
    except Exception as exc:
        _check(False, "Pre-built forecast consistency test", str(exc))

    print()

    print("─── Section 5: build_portfolio_forecast_summary ───\n")

    try:
        summary = build_portfolio_forecast_summary()

        _check(
            isinstance(summary, pd.DataFrame),
            "build_portfolio_forecast_summary returns DataFrame",
            "",
        )

        _check(
            len(summary) == 5,
            "Portfolio summary has 5 rows (one per property)",
            str(len(summary)),
        )

        required_summary_cols = {
            "Property_ID", "Property_Name", "Phase", "Num_Units",
            "Base_NOI", "Y1_Projected_NOI", "Y1_NOI_Growth_Pct",
            "Exit_Value", "Equity_Multiple", "Unlevered_IRR",
            "Target_IRR", "Beats_Target", "IRR_vs_Target_Ppt",
            "Hold_Years", "Exit_Cap_Rate", "Rent_Growth_Assumption",
            "DSCR_Y1", "Data_Quality", "Hold_Recommendation",
        }
        missing_cols = required_summary_cols - set(summary.columns)
        _check(
            len(missing_cols) == 0,
            "Portfolio summary has all required columns",
            str(missing_cols) if missing_cols else "OK",
        )

        _check(
            summary.iloc[0]["Property_ID"] == "PROP003",
            "PROP003 appears first in IRR-sorted portfolio summary",
            f"First row: {summary.iloc[0]['Property_ID']} "
            f"IRR={fmt_irr(summary.iloc[0]['Unlevered_IRR'])}",
        )

        irrs = summary["Unlevered_IRR"].dropna().tolist()
        _check(
            irrs == sorted(irrs, reverse=True),
            "Portfolio summary sorted descending by Unlevered_IRR",
            str([f"{i*100:.1f}%" for i in irrs]),
        )

        bad_recos = [
            r for r in summary["Hold_Recommendation"]
            if r not in VALID_RECOMMENDATIONS and not str(r).startswith("Error")
        ]
        _check(
            len(bad_recos) == 0,
            "All Hold_Recommendation values are valid",
            str(bad_recos) if bad_recos else "OK",
        )

        _check(
            summary["Beats_Target"].dtype == bool or
            all(isinstance(v, (bool, np.bool_)) for v in summary["Beats_Target"]),
            "Beats_Target column contains boolean values",
            str(summary["Beats_Target"].tolist()),
        )

    except Exception as exc:
        _check(
            False,
            "build_portfolio_forecast_summary raised exception",
            str(exc),
        )

    try:
        overrides = {"PROP003": {"rent_growth": 0.06, "hold_years": 3}}
        summary_ov = build_portfolio_forecast_summary(overrides)
        prop003_row = summary_ov[summary_ov["Property_ID"] == "PROP003"]
        _check(
            not prop003_row.empty,
            "PROP003 present in overridden summary",
            "",
        )
        _check(
            int(prop003_row.iloc[0]["Hold_Years"]) == 3,
            "PROP003 hold_years override applied correctly",
            str(prop003_row.iloc[0]["Hold_Years"]),
        )
    except Exception as exc:
        _check(False, "Portfolio summary with overrides test", str(exc))

    print()

    print("─── Section 6: get_hold_sell_recommendation ───\n")

    required_reco_keys = {
        "property_id", "property_name", "recommendation",
        "rationale", "confidence", "key_metrics",
    }

    required_key_metrics = {
        "unlevered_irr", "target_irr", "equity_multiple",
        "dscr_y1", "hold_years", "irr_vs_target_ppt",
    }

    for pid in PROPERTIES:
        try:
            reco = get_hold_sell_recommendation(pid)

            missing = required_reco_keys - set(reco.keys())
            _check(
                len(missing) == 0,
                f"{pid} all required recommendation keys present",
                str(missing) if missing else "OK",
            )

            _check(
                reco["recommendation"] in VALID_RECOMMENDATIONS,
                f"{pid} recommendation is valid",
                reco["recommendation"],
            )

            _check(
                reco["confidence"] in VALID_CONFIDENCE,
                f"{pid} confidence level is valid",
                reco["confidence"],
            )

            _check(
                isinstance(reco["rationale"], list) and len(reco["rationale"]) > 0,
                f"{pid} rationale is non-empty list",
                f"{len(reco['rationale'])} reason(s)",
            )

            _check(
                all(isinstance(r, str) for r in reco["rationale"]),
                f"{pid} all rationale items are strings",
                "",
            )

            missing_km = required_key_metrics - set(reco["key_metrics"].keys())
            _check(
                len(missing_km) == 0,
                f"{pid} all key_metrics fields present",
                str(missing_km) if missing_km else "OK",
            )

        except Exception as exc:
            _check(
                False,
                f"{pid} get_hold_sell_recommendation raised exception",
                str(exc),
            )

    try:
        reco3 = get_hold_sell_recommendation("PROP003")
        _check(
            reco3["recommendation"] not in {"Sell"},
            "PROP003 recommendation is not Sell",
            f"recommendation={reco3['recommendation']} "
            f"confidence={reco3['confidence']}",
        )
    except Exception as exc:
        _check(False, "PROP003 recommendation not Sell check", str(exc))

    try:
        reco4 = get_hold_sell_recommendation("PROP004")
        _check(
            reco4["recommendation"] != "Strong Hold",
            "PROP004 recommendation is not Strong Hold",
            f"recommendation={reco4['recommendation']}",
        )
    except Exception as exc:
        _check(False, "PROP004 not Strong Hold check", str(exc))

    try:
        fc1   = build_property_forecast("PROP001")
        reco_pre  = get_hold_sell_recommendation("PROP001", forecast=fc1)
        reco_auto = get_hold_sell_recommendation("PROP001")
        _check(
            reco_pre["recommendation"] == reco_auto["recommendation"],
            "PROP001 recommendation consistent with/without pre-built forecast",
            f"pre={reco_pre['recommendation']} auto={reco_auto['recommendation']}",
        )
    except Exception as exc:
        _check(False, "PROP001 pre-built vs auto forecast recommendation consistency", str(exc))

    print()

    print("─── Section 7: helper functions ───\n")

    for pid in PROPERTIES:
        try:
            fc  = build_property_forecast(pid)
            df  = get_forecast_as_dataframe(fc)
            hold = fc["assumptions"]["hold_years"]

            _check(
                isinstance(df, pd.DataFrame),
                f"{pid} get_forecast_as_dataframe returns DataFrame",
                "",
            )

            _check(
                len(df) == hold,
                f"{pid} forecast DataFrame has {hold} rows",
                str(len(df)),
            )

            expected_cols = {
                "Year", "GPR", "Vacancy_Loss", "EGI",
                "Operating_Expenses", "NOI", "CapEx_Reserve",
                "Unlevered_CF", "Debt_Service", "Levered_CF",
                "DSCR", "Unlevered_CF_With_Exit", "Levered_CF_With_Exit",
            }
            missing_df_cols = expected_cols - set(df.columns)
            _check(
                len(missing_df_cols) == 0,
                f"{pid} forecast DataFrame has all expected columns",
                str(missing_df_cols) if missing_df_cols else "OK",
            )

            _check(
                list(df["NOI"]) == fc["noi"],
                f"{pid} DataFrame NOI matches forecast dict noi array",
                "",
            )

        except Exception as exc:
            _check(
                False,
                f"{pid} get_forecast_as_dataframe raised exception",
                str(exc),
            )

    try:
        mat      = build_irr_sensitivity_matrix("PROP003")
        mat_disp = get_sensitivity_matrix_display(mat)

        _check(
            mat_disp.shape == (5, 5),
            "get_sensitivity_matrix_display returns (5,5) shape",
            str(mat_disp.shape),
        )

        for i in range(5):
            for j in range(5):
                val = mat_disp.iloc[i, j]
                if val != "N/A":
                    _check(
                        val.endswith("%"),
                        f"Sensitivity display cell [{i},{j}] ends with '%'",
                        val,
                    )
                    break   
            else:
                continue
            break

        nan_mask     = mat.isna()
        if nan_mask.any().any():
            nan_i, nan_j = [(i, j) for i in range(5) for j in range(5)
                            if nan_mask.iloc[i, j]][0]
            _check(
                mat_disp.iloc[nan_i, nan_j] == "N/A",
                "NaN IRR cell displays as 'N/A'",
                f"cell [{nan_i},{nan_j}]",
            )

    except Exception as exc:
        _check(
            False,
            "get_sensitivity_matrix_display raised exception",
            str(exc),
        )

    print()

    print("─── Section 8: analytical integrity ───\n")

    try:
        fc_low_cap  = build_property_forecast("PROP003", exit_cap_rate=0.04)
        fc_high_cap = build_property_forecast("PROP003", exit_cap_rate=0.07)
        _check(
            fc_low_cap["exit_value"] > fc_high_cap["exit_value"],
            "Lower exit cap rate → higher exit value (PROP003)",
            f"4%={fmt_currency(fc_low_cap['exit_value'])} "
            f"7%={fmt_currency(fc_high_cap['exit_value'])}",
        )
    except Exception as exc:
        _check(False, "Cap rate / exit value inverse relationship test", str(exc))

    try:
        fc_low_vac  = build_property_forecast("PROP003", vacancy_rate=0.03)
        fc_high_vac = build_property_forecast("PROP003", vacancy_rate=0.15)
        _check(
            fc_low_vac["noi"][0] > fc_high_vac["noi"][0],
            "Lower vacancy rate → higher Year 1 NOI (PROP003)",
            f"3% vac NOI={fmt_currency(fc_low_vac['noi'][0])} "
            f"15% vac NOI={fmt_currency(fc_high_vac['noi'][0])}",
        )
    except Exception as exc:
        _check(False, "Vacancy / NOI relationship test", str(exc))

    try:
        fc_low_capex  = build_property_forecast("PROP003", capex_per_unit=100)
        fc_high_capex = build_property_forecast("PROP003", capex_per_unit=700)
        _check(
            fc_low_capex["unlevered_cf"][0] > fc_high_capex["unlevered_cf"][0],
            "Lower CapEx per unit → higher Year 1 unlevered CF",
            f"$100/unit CF={fmt_currency(fc_low_capex['unlevered_cf'][0])} "
            f"$700/unit CF={fmt_currency(fc_high_capex['unlevered_cf'][0])}",
        )
    except Exception as exc:
        _check(False, "CapEx / unlevered CF relationship test", str(exc))

    try:
        fc_3yr = build_property_forecast("PROP003", hold_years=3)
        fc_7yr = build_property_forecast("PROP003", hold_years=7)
        _check(
            fc_7yr["total_unlevered_cf"] > fc_3yr["total_unlevered_cf"],
            "7-year hold produces more total unlevered CF than 3-year hold",
            f"7yr={fmt_currency(fc_7yr['total_unlevered_cf'])} "
            f"3yr={fmt_currency(fc_3yr['total_unlevered_cf'])}",
        )
    except Exception as exc:
        _check(False, "Hold period / total CF relationship test", str(exc))

    try:
        fc_dscr = build_property_forecast("PROP001")
        implied_dscr = (
            fc_dscr["noi"][0] / fc_dscr["assumptions"]["annual_debt_service"]
            if fc_dscr["assumptions"]["annual_debt_service"] > 0
            else 999.0
        )
        _check(
            abs(fc_dscr["dscr"][0] - implied_dscr) < 0.001,
            "PROP001 Year 1 DSCR = NOI / annual_debt_service (within 0.001x)",
            f"dscr={fc_dscr['dscr'][0]:.4f} implied={implied_dscr:.4f}",
        )
    except Exception as exc:
        _check(False, "DSCR calculation integrity test", str(exc))

    try:
        fc_m = build_property_forecast("PROP005")
        implied_noi_y1 = fc_m["egi"][0] - fc_m["operating_expenses"][0]
        _check(
            abs(fc_m["noi"][0] - implied_noi_y1) < 1.0,
            "PROP005 Year 1 NOI = EGI - Operating_Expenses (within $1)",
            f"noi={fmt_currency(fc_m['noi'][0])} "
            f"implied={fmt_currency(implied_noi_y1)}",
        )
    except Exception as exc:
        _check(False, "NOI = EGI - OpEx integrity test", str(exc))

    try:
        fc_lev = build_property_forecast("PROP002")
        for y_idx in range(len(fc_lev["years"])):
            implied_lev = (
                fc_lev["unlevered_cf"][y_idx]
                - fc_lev["debt_service"][y_idx]
            )
            _check(
                abs(fc_lev["levered_cf"][y_idx] - implied_lev) < 1.0,
                f"PROP002 Year {y_idx+1} levered_CF = unlevered_CF - debt_service",
                f"lev={fmt_currency(fc_lev['levered_cf'][y_idx])} "
                f"implied={fmt_currency(implied_lev)}",
            )
    except Exception as exc:
        _check(False, "Levered CF integrity test", str(exc))

    print()

    print("=" * 68)
    print("  DETAILED RESULTS")
    print("=" * 68)
    for status, name, detail in results:
        suffix = f"  [{detail}]" if detail else ""
        print(f"  {status}  {name}{suffix}")

    print()
    print("=" * 68)

    total  = len(results)
    passed = sum(1 for s, _, _ in results if s == PASS)
    failed = total - passed

    print(f"  TOTAL  : {total}")
    print(f"  PASSED : {passed}")
    print(f"  FAILED : {failed}")
    print("=" * 68)

    if errors:
        print("\n  FAILED TESTS:")
        for e in errors:
            print(f"    {FAIL}  {e}")
        print()
        sys.exit(1)
    else:
        print()
        print("  ALL TESTS PASSED")
        print()
        sys.exit(0)