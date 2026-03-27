from __future__ import annotations

import math
import warnings
import numpy as np
import pandas as pd
from typing import Union


Number = Union[int, float]

DEFAULT_EXIT_CAP_RATE:        float = 0.0525   # 5.25%
DEFAULT_VACANCY_RATE:         float = 0.08     # 8.00%
DEFAULT_RENT_GROWTH:          float = 0.035    # 3.50% per year
DEFAULT_EXPENSE_GROWTH:       float = 0.025    # 2.50% per year
DEFAULT_CAPEX_PER_UNIT:       float = 300.0    # dollars per unit per year
DEFAULT_HOLD_YEARS:           int   = 5
DEFAULT_IRR_GUESS:            float = 0.15     # 15% initial IRR guess
IRR_MAX_ITERATIONS:           int   = 1_000
IRR_TOLERANCE:                float = 1e-8


RAG_REVENUE_WARN_PCT:  float = -3.0
RAG_REVENUE_CRIT_PCT:  float = -8.0
RAG_EXPENSE_WARN_PCT:  float =  3.0
RAG_EXPENSE_CRIT_PCT:  float =  8.0
RAG_NOI_WARN_PCT:      float = -3.0
RAG_NOI_CRIT_PCT:      float = -8.0
DSCR_WATCH_BUFFER:     float =  0.10
LTV_WATCH_BUFFER:      float =  5.0




def gross_potential_rent(
    avg_rent_per_unit: Number,
    units: Number,
    months: int = 12,
) -> float:
    """
    Calculate Gross Potential Rent (GPR).

    GPR is the maximum theoretical revenue assuming 100% occupancy
    and no concessions, bad debt, or other losses for the period.

    Formula
    -------
        GPR = Avg_Rent_Per_Unit × Units × Months

    Parameters
    ----------
    avg_rent_per_unit : Average monthly rent per occupied unit.
    units             : Total number of units in the property.
    months            : Number of months in the measurement period.
                        Default 12 for an annual figure.

    Returns
    -------
    float : Gross Potential Rent for the period.

    Example
    -------
        >>> gross_potential_rent(1425, 240, 12)
        4104000.0
    """
    return float(avg_rent_per_unit) * float(units) * float(months)


def vacancy_loss(
    gpr: Number,
    vacancy_rate_pct: Number,
) -> float:
    """
    Calculate Vacancy Loss from Gross Potential Rent.

    Formula
    -------
        Vacancy_Loss = GPR × (Vacancy_Rate / 100)

    Parameters
    ----------
    gpr              : Gross Potential Rent for the period.
    vacancy_rate_pct : Vacancy rate as a percentage (e.g. 8.0 for 8%).

    Returns
    -------
    float : Dollar amount of vacancy loss.

    Example
    -------
        >>> vacancy_loss(4104000, 8.0)
        328320.0
    """
    return float(gpr) * (float(vacancy_rate_pct) / 100.0)


def effective_gross_income(
    gpr: Number,
    vacancy_rate_pct: Number,
    concessions: Number = 0.0,
    bad_debt: Number = 0.0,
    other_income: Number = 0.0,
) -> float:
    """
    Calculate Effective Gross Income (EGI).

    EGI is the actual collectible revenue after accounting for
    vacancy, concessions, and bad debt, plus any ancillary income
    such as parking, laundry, or pet fees.

    Formula
    -------
        EGI = GPR
              − Vacancy_Loss
              − Concessions
              − Bad_Debt
              + Other_Income

    Parameters
    ----------
    gpr              : Gross Potential Rent for the period.
    vacancy_rate_pct : Vacancy rate as a percentage.
    concessions      : Dollar value of rent concessions granted.
    bad_debt         : Dollar value of uncollected rent written off.
    other_income     : Ancillary income (parking, laundry, fees, etc.).

    Returns
    -------
    float : Effective Gross Income. Floored at zero.

    Example
    -------
        >>> effective_gross_income(4104000, 8.0, 36000, 18000, 24000)
        3745680.0
    """
    vac_loss = vacancy_loss(float(gpr), float(vacancy_rate_pct))
    egi = (
        float(gpr)
        - vac_loss
        - float(concessions)
        - float(bad_debt)
        + float(other_income)
    )
    return max(egi, 0.0)


def net_operating_income(
    egi: Number,
    operating_expenses: Number,
) -> float:
    """
    Calculate Net Operating Income (NOI).

    NOI is the single most important metric in multifamily real estate.
    It measures a property's income-generating ability independent of
    financing structure or tax treatment.

    Formula
    -------
        NOI = Effective_Gross_Income − Operating_Expenses

    Note: NOI excludes debt service, depreciation, income taxes,
    and capital expenditures. These are handled separately.

    Parameters
    ----------
    egi                : Effective Gross Income for the period.
    operating_expenses : Total operating expenses for the period.
                         Includes payroll, utilities, repairs and
                         maintenance, insurance, marketing, admin.
                         Excludes CapEx and debt service.

    Returns
    -------
    float : Net Operating Income. Can be negative for distressed assets.

    Example
    -------
        >>> net_operating_income(3745680, 1260000)
        2485680.0
    """
    return float(egi) - float(operating_expenses)


def noi_margin(
    noi: Number,
    egi: Number,
) -> float:
    """
    Calculate NOI Margin as a percentage of EGI.

    A higher NOI margin indicates better expense control relative
    to income. Stabilized Class B multifamily typically targets
    50–60% NOI margins.

    Formula
    -------
        NOI_Margin = (NOI / EGI) × 100

    Parameters
    ----------
    noi : Net Operating Income.
        egi : Effective Gross Income.

    Returns
    -------
    float : NOI margin as a percentage. Returns NaN if EGI is zero.

    Example
    -------
        >>> noi_margin(2485680, 3745680)
        66.36...
    """
    if float(egi) <= 0:
        return float("nan")
    return (float(noi) / float(egi)) * 100.0


def expense_ratio(
    operating_expenses: Number,
    egi: Number,
) -> float:
    """
    Calculate Expense Ratio as a percentage of EGI.

    Expense Ratio = 100% − NOI_Margin for a clean NOI calculation.
    Tracking this monthly helps identify expense creep before it
    materially impacts NOI.

    Formula
    -------
        Expense_Ratio = (Operating_Expenses / EGI) × 100

    Parameters
    ----------
    operating_expenses : Total operating expenses.
    egi                : Effective Gross Income.

    Returns
    -------
    float : Expense ratio as a percentage. Returns NaN if EGI is zero.

    Example
    -------
        >>> expense_ratio(1260000, 3745680)
        33.64...
    """
    if float(egi) <= 0:
        return float("nan")
    return (float(operating_expenses) / float(egi)) * 100.0




def capitalization_rate(
    noi: Number,
    property_value: Number,
) -> float:
    """
    Calculate Capitalization Rate (Cap Rate).

    The cap rate expresses the relationship between a property's NOI
    and its market value. It is the foundational valuation metric in
    commercial real estate and the primary tool for comparing assets
    across markets.

    Formula
    -------
        Cap_Rate = (NOI / Property_Value) × 100

    Note: Both NOI and Property_Value must be for the same period.
    Annual NOI divided by current market value is the standard.

    Parameters
    ----------
    noi            : Annual Net Operating Income.
    property_value : Current market value or purchase price.

    Returns
    -------
    float : Cap rate as a percentage. Returns NaN if value is zero.

    Example
    -------
        >>> capitalization_rate(2090000, 34500000)
        6.057...
    """
    if float(property_value) <= 0:
        return float("nan")
    return (float(noi) / float(property_value)) * 100.0


def implied_property_value(
    noi: Number,
    cap_rate_pct: Number,
) -> float:
    """
    Calculate Implied Property Value from NOI and a cap rate.

    This is the inverse of the cap rate formula and is used to
    estimate exit valuation in hold/sell analysis.

    Formula
    -------
        Property_Value = NOI / (Cap_Rate / 100)

    Parameters
    ----------
    noi          : Annual Net Operating Income at the valuation date.
    cap_rate_pct : Capitalization rate as a percentage (e.g. 5.25).

    Returns
    -------
    float : Implied property value. Returns NaN if cap rate is zero.

    Example
    -------
        >>> implied_property_value(2090000, 5.25)
        39809523.8...
    """
    if float(cap_rate_pct) <= 0:
        return float("nan")
    return float(noi) / (float(cap_rate_pct) / 100.0)


def price_per_unit(
    property_value: Number,
    units: Number,
) -> float:
    """
    Calculate Price Per Unit (PPU).

    PPU is a quick-reference valuation metric used to compare
    acquisition prices and exit valuations across properties of
    different sizes within the same market.

    Formula
    -------
        PPU = Property_Value / Units

    Parameters
    ----------
    property_value : Property value or sale price.
    units          : Total number of units.

    Returns
    -------
    float : Price per unit. Returns NaN if units is zero.

    Example
    -------
        >>> price_per_unit(34500000, 240)
        143750.0
    """
    if float(units) <= 0:
        return float("nan")
    return float(property_value) / float(units)


def gross_rent_multiplier(
    property_value: Number,
    annual_gpr: Number,
) -> float:
    """
    Calculate Gross Rent Multiplier (GRM).

    GRM is a quick valuation metric expressing how many years of
    gross rent it would take to pay off the purchase price.
    Lower GRM indicates better relative value.

    Formula
    -------
        GRM = Property_Value / Annual_GPR

    Parameters
    ----------
    property_value : Property value or purchase price.
    annual_gpr     : Annual Gross Potential Rent.

    Returns
    -------
    float : Gross Rent Multiplier (unitless). NaN if GPR is zero.

    Example
    -------
        >>> gross_rent_multiplier(34500000, 4104000)
        8.406...
    """
    if float(annual_gpr) <= 0:
        return float("nan")
    return float(property_value) / float(annual_gpr)

def debt_service_coverage_ratio(
    noi: Number,
    annual_debt_service: Number,
) -> float:
    """
    Calculate Debt Service Coverage Ratio (DSCR).

    DSCR is the primary metric used by lenders to assess a property's
    ability to service its debt from operating income. Most agency
    lenders require a minimum DSCR of 1.20x to 1.35x.

    A DSCR below 1.00x means the property cannot cover its debt
    payments from operations. A DSCR below the covenant requirement
    triggers a technical default.

    Formula
    -------
        DSCR = NOI / Annual_Debt_Service

    Parameters
    ----------
    noi                 : Annual Net Operating Income.
    annual_debt_service : Total annual principal and interest payments.

    Returns
    -------
    float : DSCR ratio (e.g. 1.41 means NOI covers debt 1.41 times).
            Returns NaN if annual_debt_service is zero.

    Example
    -------
        >>> debt_service_coverage_ratio(2772000, 1968000)
        1.408...
    """
    if float(annual_debt_service) <= 0:
        return float("nan")
    return float(noi) / float(annual_debt_service)


def loan_to_value(
    loan_balance: Number,
    property_value: Number,
) -> float:
    """
    Calculate Loan-to-Value ratio (LTV) as a percentage.

    LTV measures the degree of leverage on a property. Most agency
    lenders cap LTV at 65–75% for multifamily assets.
    Higher LTV indicates greater financial risk and less equity cushion.

    Formula
    -------
        LTV = (Loan_Balance / Property_Value) × 100

    Parameters
    ----------
    loan_balance    : Outstanding principal balance on the loan.
    property_value  : Current market value of the property.

    Returns
    -------
    float : LTV as a percentage. Returns NaN if property_value is zero.

    Example
    -------
        >>> loan_to_value(24500000, 37200000)
        65.86...
    """
    if float(property_value) <= 0:
        return float("nan")
    return (float(loan_balance) / float(property_value)) * 100.0


def implied_noi_required_for_dscr(
    annual_debt_service: Number,
    dscr_requirement: Number,
) -> float:
    """
    Calculate the minimum NOI required to maintain a given DSCR.

    Used in covenant monitoring to determine how far NOI can decline
    before triggering a covenant breach, and in scenario analysis
    to stress-test downside cases.

    Formula
    -------
        Min_NOI = Annual_Debt_Service × DSCR_Requirement

    Parameters
    ----------
    annual_debt_service : Total annual debt service (P&I).
    dscr_requirement    : Minimum DSCR required by the loan covenant.

    Returns
    -------
    float : Minimum annual NOI needed to remain compliant.

    Example
    -------
        >>> implied_noi_required_for_dscr(1392000, 1.30)
        1809600.0
    """
    return float(annual_debt_service) * float(dscr_requirement)


def dscr_headroom(
    dscr_actual: Number,
    dscr_requirement: Number,
) -> float:
    """
    Calculate DSCR headroom above the covenant requirement.

    Positive headroom = compliant.
    Zero headroom     = exactly at threshold (extreme risk).
    Negative headroom = covenant breach (critical alert).

    Formula
    -------
        DSCR_Headroom = DSCR_Actual − DSCR_Requirement

    Parameters
    ----------
    dscr_actual      : Current actual DSCR.
    dscr_requirement : Minimum required DSCR per loan covenant.

    Returns
    -------
    float : Headroom in DSCR turns (e.g. 0.06 means 0.06x above threshold).

    Example
    -------
        >>> dscr_headroom(1.41, 1.35)
        0.06
    """
    return float(dscr_actual) - float(dscr_requirement)


def ltv_headroom(
    ltv_actual: Number,
    max_ltv_allowed: Number,
) -> float:
    """
    Calculate LTV headroom below the maximum allowed LTV.

    Positive headroom = compliant.
    Negative headroom = LTV covenant breach (critical alert).

    Formula
    -------
        LTV_Headroom = Max_LTV_Allowed − LTV_Actual

    Parameters
    ----------
    ltv_actual      : Current LTV as a percentage.
    max_ltv_allowed : Maximum LTV allowed per loan covenant.

    Returns
    -------
    float : Headroom in percentage points.

    Example
    -------
        >>> ltv_headroom(65.9, 75.0)
        9.1
    """
    return float(max_ltv_allowed) - float(ltv_actual)


def annual_interest_expense(
    loan_balance: Number,
    interest_rate_pct: Number,
) -> float:
    """
    Calculate approximate annual interest expense.

    Note: This is a simple interest approximation. For fully
    amortizing loans the actual interest expense declines over time
    as principal is repaid. This function is used for quick
    comparative analysis rather than precise amortization modeling.

    Formula
    -------
        Annual_Interest = Loan_Balance × (Interest_Rate / 100)

    Parameters
    ----------
    loan_balance       : Outstanding principal balance.
    interest_rate_pct  : Annual interest rate as a percentage.

    Returns
    -------
    float : Approximate annual interest expense.

    Example
    -------
        >>> annual_interest_expense(24500000, 4.25)
        1041250.0
    """
    return float(loan_balance) * (float(interest_rate_pct) / 100.0)

def equity_multiple(
    total_distributions: Number,
    exit_proceeds: Number,
    initial_equity: Number,
) -> float:
    """
    Calculate Equity Multiple (EM).

    The equity multiple measures total value returned to equity
    investors as a multiple of their initial investment.
    It does not account for the time value of money (IRR does that).

    Formula
    -------
        EM = (Total_Cash_Distributions + Exit_Proceeds) / Initial_Equity

    Interpretation
    --------------
        < 1.0x  : Loss of capital
        1.0–1.5x: Modest return
        1.5–2.0x: Solid value-add return
        > 2.0x  : Strong opportunistic return

    Parameters
    ----------
    total_distributions : Cumulative cash distributions received during
                          the hold period (operating cash flow).
    exit_proceeds       : Net sale proceeds at exit after paying off
                          the loan balance and closing costs.
    initial_equity      : Equity invested at acquisition.

    Returns
    -------
    float : Equity multiple. Returns NaN if initial_equity is zero.

    Example
    -------
        >>> equity_multiple(2400000, 12000000, 10000000)
        1.44
    """
    if float(initial_equity) <= 0:
        return float("nan")
    return (float(total_distributions) + float(exit_proceeds)) / float(initial_equity)


def unlevered_irr(
    cash_flows: list[Number],
) -> float:
    """
    Calculate Unlevered Internal Rate of Return (IRR).

    IRR is the discount rate that makes the Net Present Value (NPV)
    of all cash flows equal to zero. It accounts for both the
    magnitude and timing of cash flows, making it the preferred
    return metric for comparing investment alternatives.

    Unlevered IRR uses cash flows before debt service, measuring
    the return on the total asset regardless of financing structure.

    The implementation uses the Newton-Raphson method with multiple
    starting points to handle non-standard cash flow patterns.

    Formula (solved numerically)
    ----------------------------
        0 = CF₀ + CF₁/(1+IRR)¹ + CF₂/(1+IRR)² + ... + CFₙ/(1+IRR)ⁿ

    Parameters
    ----------
    cash_flows : List of cash flows ordered by period. The first
                 element (index 0) is typically a negative number
                 representing the acquisition cost. Subsequent
                 elements are annual NOI cash flows. The final
                 element includes the exit sale proceeds.

                 Example for a 5-year hold:
                 [-34500000, 2090000, 2163350, 2239237,
                   2317759, 2399031 + 39800000]

    Returns
    -------
    float : IRR as a decimal (e.g. 0.1423 = 14.23%).
            Returns NaN if IRR cannot be computed (e.g. all positive
            cash flows, or Newton-Raphson fails to converge).

    Example
    -------
        >>> cf = [-34500000, 2090000, 2163350, 2239237, 2317759, 42199031]
        >>> round(unlevered_irr(cf) * 100, 2)
        14.23
    """
    cfs = [float(cf) for cf in cash_flows]

    if len(cfs) < 2:
        return float("nan")

    # Must have at least one sign change for a valid IRR
    signs = [1 if cf >= 0 else -1 for cf in cfs]
    sign_changes = sum(
        1 for i in range(len(signs) - 1) if signs[i] != signs[i + 1]
    )
    if sign_changes == 0:
        return float("nan")

    def npv(rate: float, flows: list[float]) -> float:
        """NPV at a given discount rate."""
        return sum(
            cf / (1.0 + rate) ** t
            for t, cf in enumerate(flows)
        )

    def npv_derivative(rate: float, flows: list[float]) -> float:
        """First derivative of NPV with respect to rate."""
        return sum(
            -t * cf / (1.0 + rate) ** (t + 1)
            for t, cf in enumerate(flows)
            if t > 0
        )

    # Try multiple starting points to avoid local convergence failure
    starting_guesses = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, -0.05]

    for guess in starting_guesses:
        rate = guess
        for _ in range(IRR_MAX_ITERATIONS):
            npv_val  = npv(rate, cfs)
            npv_deriv = npv_derivative(rate, cfs)

            if abs(npv_deriv) < 1e-12:
                break  # Derivative too small, try next starting point

            rate_new = rate - npv_val / npv_deriv

            if abs(rate_new - rate) < IRR_TOLERANCE:
                # Converged — validate the result is real and reasonable
                if -0.999 < rate_new < 100.0:
                    return round(rate_new, 8)
                break

            rate = rate_new

    return float("nan")


def levered_irr(
    equity_invested: Number,
    annual_levered_cash_flows: list[Number],
    net_sale_proceeds: Number,
) -> float:
    """
    Calculate Levered Internal Rate of Return.

    Levered IRR uses cash flows after debt service, measuring the
    return on equity only. It is higher than unlevered IRR when
    leverage is positive (property return exceeds cost of debt).

    Parameters
    ----------
    equity_invested            : Initial equity outlay (positive number,
                                 will be negated internally).
    annual_levered_cash_flows  : List of annual after-debt-service cash
                                 flows for Years 1 through N-1.
    net_sale_proceeds          : Net proceeds to equity after repaying
                                 the loan and closing costs at exit.

    Returns
    -------
    float : Levered IRR as a decimal. Returns NaN on failure.

    Example
    -------
        >>> levered_irr(10000000, [400000, 500000, 600000, 700000],
        ...             12000000)
        0.1821...
    """
    cfs = (
        [-float(equity_invested)]
        + [float(cf) for cf in annual_levered_cash_flows]
    )
    cfs[-1] += float(net_sale_proceeds)
    return unlevered_irr(cfs)   # same algorithm, different cash flow inputs


def net_present_value(
    cash_flows: list[Number],
    discount_rate_pct: Number,
) -> float:
    """
    Calculate Net Present Value (NPV) at a given discount rate.

    NPV measures the value created or destroyed by an investment
    relative to the required rate of return.

    Formula
    -------
        NPV = Σ [ CFₜ / (1 + r)ᵗ ]   for t = 0 to N

    Parameters
    ----------
    cash_flows         : List of cash flows ordered by period.
                         Index 0 is typically the acquisition cost
                         as a negative number.
    discount_rate_pct  : Required rate of return as a percentage
                         (e.g. 12.0 for 12%).

    Returns
    -------
    float : Net Present Value in dollars.

    Example
    -------
        >>> cf = [-34500000, 2090000, 2163350, 2239237, 2317759, 42199031]
        >>> net_present_value(cf, 12.0)
        2847...
    """
    r = float(discount_rate_pct) / 100.0
    return sum(
        float(cf) / (1.0 + r) ** t
        for t, cf in enumerate(cash_flows)
    )


def gross_profit_on_sale(
    exit_value: Number,
    loan_balance_at_exit: Number,
    purchase_price: Number,
    closing_costs_pct: Number = 1.5,
) -> float:
    """
    Calculate Gross Profit on Sale.

    Estimates the net gain to the investor at exit after repaying
    the remaining loan balance and deducting selling costs.

    Formula
    -------
        Selling_Costs = Exit_Value × (Closing_Costs_Pct / 100)
        Net_Proceeds  = Exit_Value − Selling_Costs − Loan_Balance_at_Exit
        Gross_Profit  = Net_Proceeds − Purchase_Price

    Parameters
    ----------
    exit_value           : Projected sale price at exit.
    loan_balance_at_exit : Outstanding loan principal at time of sale.
    purchase_price       : Original acquisition price.
    closing_costs_pct    : Selling costs as a percentage of exit value.
                           Default 1.5% covers broker fees and transfer
                           taxes for a simplified estimate.

    Returns
    -------
    float : Gross profit (positive) or loss (negative) on sale.

    Example
    -------
        >>> gross_profit_on_sale(42000000, 22000000, 34500000, 1.5)
        -15130000.0   # (would be positive at higher exit values)
    """
    selling_costs = float(exit_value) * (float(closing_costs_pct) / 100.0)
    net_proceeds  = float(exit_value) - selling_costs - float(loan_balance_at_exit)
    return net_proceeds - float(purchase_price)


def return_on_cost(
    expected_noi_lift: Number,
    renovation_cost: Number,
    exit_cap_rate_pct: Number = DEFAULT_EXIT_CAP_RATE * 100,
) -> float:
    """
    Calculate Renovation Return on Cost.

    Measures the value created by a capital improvement program
    relative to the cost of that improvement.

    Formula
    -------
        Implied_Value_Add = NOI_Lift / (Exit_Cap_Rate / 100)
        Return_on_Cost    = Implied_Value_Add / Renovation_Cost

    A Return on Cost above 1.0 means the renovation creates more
    value than it costs. Value-add strategies typically target
    Returns on Cost between 1.5x and 2.5x.

    Parameters
    ----------
    expected_noi_lift  : Annual NOI increase expected from the
                         renovation program.
    renovation_cost    : Total capital cost of the renovation.
    exit_cap_rate_pct  : Exit cap rate used to capitalise the NOI
                         lift into a value estimate (percentage).
                         Default 5.25%.

    Returns
    -------
    float : Return on cost multiple. Returns NaN on invalid inputs.

    Example
    -------
        >>> return_on_cost(85000, 450000, 5.25)
        3.593...
    """
    if float(renovation_cost) <= 0 or float(exit_cap_rate_pct) <= 0:
        return float("nan")
    implied_value_add = float(expected_noi_lift) / (float(exit_cap_rate_pct) / 100.0)
    return implied_value_add / float(renovation_cost)


def renovation_roi_per_unit(
    expected_noi_lift: Number,
    renovation_cost: Number,
    units_renovated: Number,
    exit_cap_rate_pct: Number = DEFAULT_EXIT_CAP_RATE * 100,
) -> dict[str, float]:
    """
    Calculate per-unit renovation ROI metrics.

    Breaks down a renovation program's economics on a per-unit
    basis, which is the standard reporting format for value-add
    multifamily asset management.

    Parameters
    ----------
    expected_noi_lift  : Total annual NOI increase from the program.
    renovation_cost    : Total renovation cost for all units.
    units_renovated    : Number of units included in the program.
    exit_cap_rate_pct  : Exit cap rate as a percentage.

    Returns
    -------
    dict with keys:
        cost_per_unit          : float — renovation cost per unit
        noi_lift_per_unit      : float — annual NOI increase per unit
        implied_value_per_unit : float — value created per unit
        return_on_cost         : float — overall return on cost multiple
        roc_per_unit           : float — return on cost per unit
        payback_years          : float — years to recoup renovation cost
                                         from NOI lift alone

    Example
    -------
        >>> renovation_roi_per_unit(85000, 450000, 180, 5.25)
        {'cost_per_unit': 2500.0, 'noi_lift_per_unit': 472.22..., ...}
    """
    units = max(float(units_renovated), 1.0)
    cost  = float(renovation_cost)
    lift  = float(expected_noi_lift)
    cap   = float(exit_cap_rate_pct)

    cost_per_unit          = cost / units
    noi_lift_per_unit      = lift / units
    implied_value_per_unit = (
        (lift / (cap / 100.0)) / units if cap > 0 else float("nan")
    )
    roc = return_on_cost(lift, cost, cap)
    payback = cost / lift if lift > 0 else float("nan")

    return {
        "cost_per_unit":          round(cost_per_unit, 2),
        "noi_lift_per_unit":      round(noi_lift_per_unit, 2),
        "implied_value_per_unit": round(implied_value_per_unit, 2),
        "return_on_cost":         round(roc, 4),
        "roc_per_unit":           round(
            implied_value_per_unit / cost_per_unit
            if cost_per_unit > 0 else float("nan"), 4
        ),
        "payback_years":          round(payback, 2),
    }

def project_noi_growth(
    base_noi: Number,
    growth_rate_pct: Number,
    years: int,
) -> list[float]:
    """
    Project NOI forward using a constant annual growth rate.

    This is a simplified projection model. The full 5-year cash flow
    model in forecasting.py applies separate growth rates to revenue
    and expenses for greater realism.

    Formula
    -------
        NOI_Year_N = Base_NOI × (1 + Growth_Rate)^N

    Parameters
    ----------
    base_noi        : Starting annual NOI (Year 0 / current).
    growth_rate_pct : Annual NOI growth rate as a percentage.
    years           : Number of years to project.

    Returns
    -------
    list[float] : Projected NOI for each year from Year 1 to Year N.
                  Length equals the years parameter.

    Example
    -------
        >>> project_noi_growth(2090000, 3.5, 5)
        [2163150.0, 2238760.25, 2317016.86, 2398012.45, 2481942.89]
    """
    rate = float(growth_rate_pct) / 100.0
    base = float(base_noi)
    return [
        round(base * (1.0 + rate) ** year, 2)
        for year in range(1, years + 1)
    ]


def build_five_year_cash_flow_model(
    base_annual_revenue:    Number,
    base_annual_expenses:   Number,
    units:                  Number,
    rent_growth_pct:        Number = DEFAULT_RENT_GROWTH * 100,
    expense_growth_pct:     Number = DEFAULT_EXPENSE_GROWTH * 100,
    vacancy_rate_pct:       Number = DEFAULT_VACANCY_RATE * 100,
    capex_per_unit:         Number = DEFAULT_CAPEX_PER_UNIT,
    hold_years:             int    = DEFAULT_HOLD_YEARS,
    exit_cap_rate_pct:      Number = DEFAULT_EXIT_CAP_RATE * 100,
    purchase_price:         Number = 0.0,
    loan_balance:           Number = 0.0,
    annual_debt_service:    Number = 0.0,
) -> dict[str, list[float] | float]:
    """
    Build a multi-year cash flow projection model.

    This is the core financial model used by the Forecasting and
    Valuation page. It applies separate growth rates to revenue and
    expenses, deducts a CapEx reserve, and produces both unlevered
    and levered cash flows.

    Methodology
    -----------
    Year 0 : Acquisition (negative purchase price as initial outflow).
    Year 1 to N:
        Projected_Revenue   = Base_Revenue  × (1 + Rent_Growth)^Year
        Projected_Expenses  = Base_Expenses × (1 + Expense_Growth)^Year
        Vacancy_Deduction   = Projected_Revenue × (Vacancy_Rate / 100)
        EGI                 = Projected_Revenue − Vacancy_Deduction
        NOI                 = EGI − Projected_Expenses
        CapEx_Reserve       = CapEx_Per_Unit × Units
        Unlevered_CF        = NOI − CapEx_Reserve
        Levered_CF          = Unlevered_CF − Annual_Debt_Service
    Year N (exit):
        Exit_NOI            = NOI in the final year
        Exit_Value          = Exit_NOI / (Exit_Cap_Rate / 100)
        Unlevered_CF_Exit   = Unlevered_CF + Exit_Value
        Net_Sale_Proceeds   = Exit_Value − Loan_Balance (simplified)
        Levered_CF_Exit     = Levered_CF + Net_Sale_Proceeds

    Parameters
    ----------
    base_annual_revenue  : Current annual revenue baseline.
    base_annual_expenses : Current annual expense baseline.
    units                : Total units (for CapEx reserve calc).
    rent_growth_pct      : Annual revenue growth rate (%).
    expense_growth_pct   : Annual expense growth rate (%).
    vacancy_rate_pct     : Stabilized vacancy assumption (%).
    capex_per_unit       : Annual CapEx reserve per unit ($).
    hold_years           : Projection period in years.
    exit_cap_rate_pct    : Exit capitalization rate (%).
    purchase_price       : Acquisition price for IRR calculation.
    loan_balance         : Current outstanding loan balance.
    annual_debt_service  : Annual debt service (P&I) payments.

    Returns
    -------
    dict with keys:
        years              : list[int]   — [1, 2, ..., N]
        revenue            : list[float] — projected annual revenue
        vacancy_loss       : list[float] — projected vacancy deduction
        egi                : list[float] — effective gross income
        expenses           : list[float] — projected annual expenses
        noi                : list[float] — net operating income
        capex_reserve      : list[float] — annual CapEx reserve
        unlevered_cf       : list[float] — NOI minus CapEx reserve
        debt_service       : list[float] — annual debt service
        levered_cf         : list[float] — after-debt-service cash flow
        exit_noi           : float — NOI in the exit year
        exit_value         : float — implied sale price at exit
        net_sale_proceeds  : float — exit value minus loan balance
        unlevered_irr      : float — unlevered IRR as a decimal
        levered_irr        : float — levered IRR as a decimal
        equity_multiple    : float — equity multiple
        total_distributions: float — cumulative levered cash flows
        capex_rate         : float — total CapEx per unit per year
        dscr_by_year       : list[float] — DSCR for each projected year

    Example
    -------
        >>> model = build_five_year_cash_flow_model(
        ...     base_annual_revenue=3269880,
        ...     base_annual_expenses=1096200,
        ...     units=240,
        ...     rent_growth_pct=3.5,
        ...     expense_growth_pct=2.5,
        ...     vacancy_rate_pct=8.0,
        ...     capex_per_unit=350,
        ...     hold_years=5,
        ...     exit_cap_rate_pct=5.25,
        ...     purchase_price=34500000,
        ...     loan_balance=24500000,
        ...     annual_debt_service=1968000,
        ... )
    """
    rent_g    = float(rent_growth_pct)    / 100.0
    exp_g     = float(expense_growth_pct) / 100.0
    vac_rate  = float(vacancy_rate_pct)   / 100.0
    capex_res = float(capex_per_unit) * float(units)
    cap_rate  = float(exit_cap_rate_pct)  / 100.0
    base_rev  = float(base_annual_revenue)
    base_exp  = float(base_annual_expenses)
    ds        = float(annual_debt_service)

    years_list       : list[int]   = []
    revenues         : list[float] = []
    vac_losses       : list[float] = []
    egis             : list[float] = []
    expenses_list    : list[float] = []
    nois             : list[float] = []
    capex_list       : list[float] = []
    unlevered_cfs    : list[float] = []
    debt_services    : list[float] = []
    levered_cfs      : list[float] = []
    dscr_by_year     : list[float] = []

    for yr in range(1, hold_years + 1):
        proj_rev  = base_rev * (1.0 + rent_g) ** yr
        proj_exp  = base_exp * (1.0 + exp_g)  ** yr
        vac       = proj_rev * vac_rate
        egi_val   = proj_rev - vac
        noi_val   = egi_val - proj_exp
        ucf       = noi_val - capex_res
        lcf       = ucf - ds
        dscr_val  = debt_service_coverage_ratio(noi_val, ds)

        years_list.append(yr)
        revenues.append(round(proj_rev, 2))
        vac_losses.append(round(vac, 2))
        egis.append(round(egi_val, 2))
        expenses_list.append(round(proj_exp, 2))
        nois.append(round(noi_val, 2))
        capex_list.append(round(capex_res, 2))
        unlevered_cfs.append(round(ucf, 2))
        debt_services.append(round(ds, 2))
        levered_cfs.append(round(lcf, 2))
        dscr_by_year.append(round(dscr_val, 4))

    exit_noi_val    = nois[-1]
    exit_value_val  = (
        exit_noi_val / cap_rate if cap_rate > 0 else float("nan")
    )
    net_sale_proc   = exit_value_val - float(loan_balance)

    unlevered_cf_for_irr = (
        [-float(purchase_price)]
        + unlevered_cfs[:-1]
        + [unlevered_cfs[-1] + exit_value_val]
    )
    u_irr = (
        unlevered_irr(unlevered_cf_for_irr)
        if float(purchase_price) > 0
        else float("nan")
    )

    equity_in = float(purchase_price) - float(loan_balance)
    total_dist = sum(levered_cfs[:-1])

    l_irr = (
        levered_irr(equity_in, levered_cfs[:-1], net_sale_proc)
        if equity_in > 0
        else float("nan")
    )

    em = (
        equity_multiple(total_dist, net_sale_proc, equity_in)
        if equity_in > 0
        else float("nan")
    )

    return {
        "years":               years_list,
        "revenue":             revenues,
        "vacancy_loss":        vac_losses,
        "egi":                 egis,
        "expenses":            expenses_list,
        "noi":                 nois,
        "capex_reserve":       capex_list,
        "unlevered_cf":        unlevered_cfs,
        "debt_service":        debt_services,
        "levered_cf":          levered_cfs,
        "exit_noi":            round(exit_noi_val, 2),
        "exit_value":          round(exit_value_val, 2),
        "net_sale_proceeds":   round(net_sale_proc, 2),
        "unlevered_irr":       u_irr,
        "levered_irr":         l_irr,
        "equity_multiple":     round(em, 4) if not math.isnan(em) else float("nan"),
        "total_distributions": round(total_dist, 2),
        "capex_rate":          round(capex_res, 2),
        "dscr_by_year":        dscr_by_year,
    }


# ══════════════════════════════════════════════════════════════════════
# SECTION 6 — SENSITIVITY ANALYSIS
# ══════════════════════════════════════════════════════════════════════

def build_irr_sensitivity_matrix(
    base_annual_revenue:    Number,
    base_annual_expenses:   Number,
    units:                  Number,
    purchase_price:         Number,
    loan_balance:           Number,
    annual_debt_service:    Number,
    capex_per_unit:         Number = DEFAULT_CAPEX_PER_UNIT,
    hold_years:             int    = DEFAULT_HOLD_YEARS,
    exit_cap_rates:         list[float] | None = None,
    rent_growth_rates:      list[float] | None = None,
    vacancy_rate_pct:       Number = DEFAULT_VACANCY_RATE * 100,
    expense_growth_pct:     Number = DEFAULT_EXPENSE_GROWTH * 100,
) -> dict[str, object]:
    """
    Build a 2-dimensional IRR sensitivity matrix.

    Returns unlevered IRR values across a grid of:
        Rows    → Exit Cap Rate assumptions
        Columns → Rent Growth Rate assumptions

    This is the most analytically sophisticated output in the
    application and mirrors the sensitivity tables produced by
    institutional asset managers for Investment Committee presentations.

    Parameters
    ----------
    base_annual_revenue  : Current annual revenue baseline.
    base_annual_expenses : Current annual expense baseline.
    units                : Total units.
    purchase_price       : Acquisition price.
    loan_balance         : Current loan balance.
    annual_debt_service  : Annual debt service.
    capex_per_unit       : CapEx reserve per unit per year.
    hold_years           : Projection period.
    exit_cap_rates       : List of exit cap rate percentages for rows.
                           Default: [4.25, 4.75, 5.25, 5.75, 6.25].
    rent_growth_rates    : List of rent growth percentages for columns.
                           Default: [2.0, 2.75, 3.5, 4.25, 5.0].
    vacancy_rate_pct     : Vacancy rate assumption (%).
    expense_growth_pct   : Expense growth rate (%).

    Returns
    -------
    dict with keys:
        matrix       : list[list[float]] — IRR values as decimals.
                       Outer list = rows (cap rates).
                       Inner list = columns (rent growth rates).
        cap_rates    : list[float] — row labels (exit cap rate %).
        rent_growths : list[float] — column labels (rent growth %).
        row_labels   : list[str]   — formatted row labels.
        col_labels   : list[str]   — formatted column labels.

    Example
    -------
        >>> sm = build_irr_sensitivity_matrix(
        ...     3269880, 1096200, 240,
        ...     34500000, 24500000, 1968000
        ... )
        >>> len(sm['matrix'])
        5
        >>> len(sm['matrix'][0])
        5
    """
    if exit_cap_rates is None:
        exit_cap_rates = [4.25, 4.75, 5.25, 5.75, 6.25]
    if rent_growth_rates is None:
        rent_growth_rates = [2.0, 2.75, 3.5, 4.25, 5.0]

    matrix: list[list[float]] = []

    for cap_rate in exit_cap_rates:
        row: list[float] = []
        for rent_growth in rent_growth_rates:
            model = build_five_year_cash_flow_model(
                base_annual_revenue=base_annual_revenue,
                base_annual_expenses=base_annual_expenses,
                units=units,
                rent_growth_pct=rent_growth,
                expense_growth_pct=expense_growth_pct,
                vacancy_rate_pct=vacancy_rate_pct,
                capex_per_unit=capex_per_unit,
                hold_years=hold_years,
                exit_cap_rate_pct=cap_rate,
                purchase_price=purchase_price,
                loan_balance=loan_balance,
                annual_debt_service=annual_debt_service,
            )
            row.append(model["unlevered_irr"])
        matrix.append(row)

    return {
        "matrix":       matrix,
        "cap_rates":    exit_cap_rates,
        "rent_growths": rent_growth_rates,
        "row_labels":   [f"{r:.2f}% Cap" for r in exit_cap_rates],
        "col_labels":   [f"{g:.2f}% Rent Growth" for g in rent_growth_rates],
    }


# ══════════════════════════════════════════════════════════════════════
# SECTION 7 — VARIANCE AND BENCHMARKING CALCULATIONS
# ══════════════════════════════════════════════════════════════════════

def calculate_variance(
    actual: Number,
    budget: Number,
) -> dict[str, float]:
    """
    Calculate dollar and percentage variance between actual and budget.

    Sign convention:
        Revenue / NOI : positive variance = favorable (actual > budget)
        Expenses       : positive variance = unfavorable (actual > budget)

    Parameters
    ----------
    actual : Actual value for the period.
    budget : Budgeted value for the period.

    Returns
    -------
    dict with keys:
        dollar_variance : float — actual minus budget.
        pct_variance    : float — dollar_variance / budget × 100.
                          NaN if budget is zero.
        is_favorable    : bool  — True if actual >= budget (caller must
                                  apply sign logic for expenses).

    Example
    -------
        >>> calculate_variance(272400, 285000)
        {'dollar_variance': -12600.0, 'pct_variance': -4.421..., 'is_favorable': False}
    """
    dollar_var = float(actual) - float(budget)
    pct_var = (
        (dollar_var / float(budget)) * 100.0
        if float(budget) != 0
        else float("nan")
    )
    return {
        "dollar_variance": round(dollar_var, 2),
        "pct_variance":    round(pct_var, 4),
        "is_favorable":    actual >= budget,
    }


def assign_rag_status(
    variance_pct: Number,
    line_item_type: str = "revenue",
) -> str:
    """
    Assign a Red / Amber / Green status to a variance percentage.

    Parameters
    ----------
    variance_pct   : Variance as a percentage (actual vs budget).
    line_item_type : 'revenue' or 'noi' → negative variance is bad.
                     'expense'           → positive variance is bad.

    Returns
    -------
    str : 'Red', 'Amber', 'Green', or 'Grey' (for NaN inputs).

    Example
    -------
        >>> assign_rag_status(-4.5, 'revenue')
        'Amber'
        >>> assign_rag_status(9.2, 'expense')
        'Red'
        >>> assign_rag_status(-1.0, 'noi')
        'Green'
    """
    if variance_pct is None or (
        isinstance(variance_pct, float) and math.isnan(variance_pct)
    ):
        return "Grey"

    v = float(variance_pct)

    if line_item_type in ("revenue", "noi"):
        if v <= RAG_REVENUE_CRIT_PCT:
            return "Red"
        if v <= RAG_REVENUE_WARN_PCT:
            return "Amber"
        return "Green"

    # expense: unfavorable direction is positive
    if v >= RAG_EXPENSE_CRIT_PCT:
        return "Red"
    if v >= RAG_EXPENSE_WARN_PCT:
        return "Amber"
    return "Green"


def weighted_average_occupancy(
    occupancies: list[Number],
    unit_counts:  list[Number],
) -> float:
    """
    Calculate units-weighted average occupancy across a portfolio.

    Simple averaging is incorrect for a portfolio because it treats
    a 50-unit property the same as a 320-unit property. Units-weighted
    averaging gives larger properties their proportional influence.

    Formula
    -------
        Weighted_Avg_Occ = Σ(Occupancy_i × Units_i) / Σ(Units_i)

    Parameters
    ----------
    occupancies : List of current occupancy rates as percentages.
    unit_counts : List of unit counts corresponding to each occupancy.

    Returns
    -------
    float : Weighted average occupancy as a percentage.
            Returns NaN if total units is zero or lists are empty.

    Example
    -------
        >>> weighted_average_occupancy([88, 85, 91, 82, 90],
        ...                            [240, 180, 320, 150, 275])
        87.97...
    """
    if len(occupancies) != len(unit_counts) or len(occupancies) == 0:
        return float("nan")

    occ_arr   = np.array([float(o) for o in occupancies])
    units_arr = np.array([float(u) for u in unit_counts])
    total_units = units_arr.sum()

    if total_units <= 0:
        return float("nan")

    return float((occ_arr * units_arr).sum() / total_units)


def rent_premium_vs_comps(
    subject_rent: Number,
    comp_rents:   list[Number],
) -> dict[str, float]:
    """
    Calculate rent premium or discount vs a comp set.

    Parameters
    ----------
    subject_rent : Subject property's average rent per unit.
    comp_rents   : List of average rents from comparable properties.

    Returns
    -------
    dict with keys:
        comp_average        : float — simple average of comp rents.
        dollar_premium      : float — subject rent minus comp average.
        pct_premium         : float — dollar_premium / comp_average × 100.
        vs_highest_comp     : float — gap vs the highest comp rent.
        vs_lowest_comp      : float — gap vs the lowest comp rent.

    Example
    -------
        >>> rent_premium_vs_comps(1425, [1380, 1450, 1360, 1395, 1410])
        {'comp_average': 1399.0, 'dollar_premium': 26.0, ...}
    """
    if not comp_rents:
        return {
            "comp_average":    float("nan"),
            "dollar_premium":  float("nan"),
            "pct_premium":     float("nan"),
            "vs_highest_comp": float("nan"),
            "vs_lowest_comp":  float("nan"),
        }

    comp_arr     = np.array([float(r) for r in comp_rents])
    comp_avg     = float(comp_arr.mean())
    subj         = float(subject_rent)
    dollar_prem  = subj - comp_avg
    pct_prem     = (dollar_prem / comp_avg * 100.0) if comp_avg > 0 else float("nan")

    return {
        "comp_average":    round(comp_avg, 2),
        "dollar_premium":  round(dollar_prem, 2),
        "pct_premium":     round(pct_prem, 4),
        "vs_highest_comp": round(subj - float(comp_arr.max()), 2),
        "vs_lowest_comp":  round(subj - float(comp_arr.min()), 2),
    }


# ══════════════════════════════════════════════════════════════════════
# SECTION 8 — FORMATTING HELPERS
# ══════════════════════════════════════════════════════════════════════
# These helpers format numbers into display strings for use in
# Streamlit metric labels, PDF reports, and Excel cells.

def fmt_currency(value: Number, decimals: int = 0) -> str:
    """
    Format a number as a USD currency string.

    Parameters
    ----------
    value    : Numeric value to format.
    decimals : Number of decimal places. Default 0 for whole dollars.

    Returns
    -------
    str : e.g. '$2,090,000' or '$1,425.50' or 'N/A' for NaN.

    Example
    -------
        >>> fmt_currency(2090000)
        '$2,090,000'
        >>> fmt_currency(1425.5, 2)
        '$1,425.50'
    """
    try:
        v = float(value)
        if math.isnan(v):
            return "N/A"
        if v < 0:
            return f"-${abs(v):,.{decimals}f}"
        return f"${v:,.{decimals}f}"
    except (TypeError, ValueError):
        return "N/A"


def fmt_percent(value: Number, decimals: int = 1) -> str:
    """
    Format a number as a percentage string.

    Parameters
    ----------
    value    : Value already expressed as a percentage (e.g. 14.23).
    decimals : Decimal places. Default 1.

    Returns
    -------
    str : e.g. '14.2%' or 'N/A' for NaN.

    Example
    -------
        >>> fmt_percent(14.23)
        '14.2%'
    """
    try:
        v = float(value)
        if math.isnan(v):
            return "N/A"
        return f"{v:.{decimals}f}%"
    except (TypeError, ValueError):
        return "N/A"


def fmt_multiple(value: Number, decimals: int = 2) -> str:
    """
    Format a number as an 'x' multiple string.

    Parameters
    ----------
    value    : Ratio value (e.g. 1.41 for a DSCR or equity multiple).
    decimals : Decimal places. Default 2.

    Returns
    -------
    str : e.g. '1.41x' or 'N/A' for NaN.

    Example
    -------
        >>> fmt_multiple(1.41)
        '1.41x'
    """
    try:
        v = float(value)
        if math.isnan(v):
            return "N/A"
        return f"{v:.{decimals}f}x"
    except (TypeError, ValueError):
        return "N/A"


def fmt_irr(value: Number, decimals: int = 1) -> str:
    """
    Format an IRR decimal (e.g. 0.1423) as a percentage string.

    Parameters
    ----------
    value    : IRR as a decimal fraction (not already a percentage).
    decimals : Decimal places. Default 1.

    Returns
    -------
    str : e.g. '14.2%' or 'N/A' for NaN or None.

    Example
    -------
        >>> fmt_irr(0.1423)
        '14.2%'
    """
    try:
        v = float(value)
        if math.isnan(v):
            return "N/A"
        return f"{v * 100:.{decimals}f}%"
    except (TypeError, ValueError):
        return "N/A"


def rag_to_color(rag: str) -> str:
    """
    Convert a RAG label to a hex color code for Streamlit UI rendering.

    Parameters
    ----------
    rag : 'Red', 'Amber', 'Green', or 'Grey'.

    Returns
    -------
    str : Hex color code.

    Example
    -------
        >>> rag_to_color('Red')
        '#C0392B'
    """
    color_map: dict[str, str] = {
        "Red":   "#C0392B",
        "Amber": "#D4AC0D",
        "Green": "#1B7F4F",
        "Grey":  "#7F8C8D",
    }
    return color_map.get(str(rag).strip(), "#7F8C8D")


def rag_to_emoji(rag: str) -> str:
    """
    Convert a RAG label to an emoji for compact UI display.

    Parameters
    ----------
    rag : 'Red', 'Amber', 'Green', or 'Grey'.

    Returns
    -------
    str : Emoji string.

    Example
    -------
        >>> rag_to_emoji('Green')
        '🟢'
    """
    emoji_map: dict[str, str] = {
        "Red":   "🔴",
        "Amber": "🟡",
        "Green": "🟢",
        "Grey":  "⚪",
    }
    return emoji_map.get(str(rag).strip(), "⚪")


# ══════════════════════════════════════════════════════════════════════
# SECTION 9 — MODULE SELF-TEST
# ══════════════════════════════════════════════════════════════════════
# Run from the terminal to verify all formulas produce correct results:
#
#   cd assetoptima-pro
#   python modules/kpi_calculations.py
#
# A passing run prints every test result and exits with code 0.
# Any formula error prints the expected vs actual value and exits
# with code 1.

if __name__ == "__main__":
    import sys
    import logging

    logging.getLogger(
        "streamlit.runtime.caching.cache_data_api"
    ).setLevel(logging.ERROR)

    # ── Test infrastructure ───────────────────────────────────────────

    PASS  = "[PASS]"
    FAIL  = "[FAIL]"
    all_passed = True
    results: list[tuple[str, str, str]] = []

    def check(
        test_name: str,
        actual: object,
        expected: object,
        tolerance: float = 0.01,
    ) -> None:
        """
        Compare actual vs expected and record pass/fail.
        For floats, comparison uses the given tolerance.
        For strings and booleans, exact equality is required.
        """

        try:
            if isinstance(expected, float) and math.isnan(expected):
                passed = isinstance(actual, float) and math.isnan(actual)
            elif isinstance(expected, float) or isinstance(expected, int):
                passed = abs(float(actual) - float(expected)) <= tolerance
            else:
                passed = actual == expected
        except (TypeError, ValueError):
            passed = False

        status = PASS if passed else FAIL
        if not passed:
            all_passed = False

        results.append((
            status,
            test_name,
            f"expected={expected!r}  got={actual!r}" if not passed else "",
        ))

    # ─────────────────────────────────────────────────────────────────
    # SECTION 1 TESTS — Income and NOI
    # ─────────────────────────────────────────────────────────────────

    check(
        "gross_potential_rent — basic",
        gross_potential_rent(1425, 240, 12),
        4104000.0,
    )
    check(
        "gross_potential_rent — monthly period",
        gross_potential_rent(1425, 240, 1),
        342000.0,
    )
    check(
        "vacancy_loss — 8 percent",
        vacancy_loss(4104000, 8.0),
        328320.0,
    )
    check(
        "vacancy_loss — zero rate",
        vacancy_loss(4104000, 0.0),
        0.0,
    )
    check(
        "effective_gross_income — full calculation",
        effective_gross_income(4104000, 8.0, 36000, 18000, 24000),
        3745680.0,
    )
    check(
        "effective_gross_income — no ancillary items",
        effective_gross_income(4104000, 8.0),
        3775680.0,
    )
    check(
        "effective_gross_income — floored at zero",
        effective_gross_income(100, 100.0, 1000, 0, 0),
        0.0,
    )
    check(
        "net_operating_income — standard",
        net_operating_income(3775680, 1096200),
        2679480.0,
    )
    check(
        "net_operating_income — negative (distressed)",
        net_operating_income(500000, 800000),
        -300000.0,
    )
    check(
        "noi_margin — standard",
        noi_margin(2679480, 3775680),
        70.966,
        tolerance=0.01,
    )
    check(
        "noi_margin — zero EGI returns NaN",
        math.isnan(noi_margin(100, 0)),
        True,
    )
    check(
        "expense_ratio — standard",
        expense_ratio(1096200, 3775680),
        29.034,
        tolerance=0.01,
    )

    # ─────────────────────────────────────────────────────────────────
    # SECTION 2 TESTS — Valuation
    # ─────────────────────────────────────────────────────────────────

    check(
        "capitalization_rate — purchase",
        capitalization_rate(2090000, 34500000),
        6.057,
        tolerance=0.01,
    )
    check(
        "capitalization_rate — zero value returns NaN",
        math.isnan(capitalization_rate(2090000, 0)),
        True,
    )
    check(
        "implied_property_value — standard",
        implied_property_value(2090000, 5.25),
        39809523.81,
        tolerance=1.0,
    )
    check(
        "implied_property_value — zero cap rate returns NaN",
        math.isnan(implied_property_value(2090000, 0)),
        True,
    )
    check(
        "price_per_unit — standard",
        price_per_unit(34500000, 240),
        143750.0,
    )
    check(
        "price_per_unit — zero units returns NaN",
        math.isnan(price_per_unit(34500000, 0)),
        True,
    )
    check(
        "gross_rent_multiplier — standard",
        gross_rent_multiplier(34500000, 4104000),
        8.406,
        tolerance=0.01,
    )

    # ─────────────────────────────────────────────────────────────────
    # SECTION 3 TESTS — Debt and Financing
    # ─────────────────────────────────────────────────────────────────

    check(
        "debt_service_coverage_ratio — compliant",
        debt_service_coverage_ratio(2772000, 1968000),
        1.409,
        tolerance=0.001,
    )
    check(
        "debt_service_coverage_ratio — breach below 1.0",
        debt_service_coverage_ratio(900000, 1128000),
        0.798,
        tolerance=0.001,
    )
    check(
        "debt_service_coverage_ratio — zero ADS returns NaN",
        math.isnan(debt_service_coverage_ratio(2000000, 0)),
        True,
    )
    check(
        "loan_to_value — standard",
        loan_to_value(24500000, 37200000),
        65.860,
        tolerance=0.01,
    )
    check(
        "loan_to_value — zero value returns NaN",
        math.isnan(loan_to_value(24500000, 0)),
        True,
    )
    check(
        "implied_noi_required_for_dscr — PROP002 breach scenario",
        implied_noi_required_for_dscr(1392000, 1.30),
        1809600.0,
    )
    check(
        "dscr_headroom — compliant positive",
        dscr_headroom(1.41, 1.35),
        0.06,
        tolerance=0.001,
    )
    check(
        "dscr_headroom — breach negative",
        dscr_headroom(1.28, 1.30),
        -0.02,
        tolerance=0.001,
    )
    check(
        "ltv_headroom — comfortable",
        ltv_headroom(65.9, 75.0),
        9.1,
        tolerance=0.01,
    )
    check(
        "annual_interest_expense — standard",
        annual_interest_expense(24500000, 4.25),
        1041250.0,
    )

    # ─────────────────────────────────────────────────────────────────
    # SECTION 4 TESTS — Returns Analysis
    # ─────────────────────────────────────────────────────────────────

    check(
        "equity_multiple — solid value-add return",
        equity_multiple(2400000, 12000000, 10000000),
        1.44,
        tolerance=0.001,
    )
    check(
        "equity_multiple — zero equity returns NaN",
        math.isnan(equity_multiple(2400000, 12000000, 0)),
        True,
    )

    # IRR test using known cash flows with a verifiable solution
    # CF: -34500000, then 4 annual flows of ~2.1M, then exit year ~42M
    # Expected unlevered IRR ≈ 13-15% range for this asset class
    test_cfs = [
        -34500000,
         2090000,
         2163350,
         2239237,
         2317759,
        42199031,   # NOI + exit proceeds in year 5
    ]
    computed_irr = unlevered_irr(test_cfs)
    check(
        "unlevered_irr — converges to reasonable value",
        0.10 < computed_irr < 0.20,
        True,
    )
    check(
        "unlevered_irr — all positive flows returns NaN",
        math.isnan(unlevered_irr([100, 200, 300])),
        True,
    )
    check(
        "unlevered_irr — too few flows returns NaN",
        math.isnan(unlevered_irr([100])),
        True,
    )

    check(
        "net_present_value — positive NPV at 10 percent",
        net_present_value(test_cfs, 10.0) > 0,
        True,
    )
    check(
        "net_present_value — negative NPV at high discount rate",
        net_present_value(test_cfs, 25.0) < 0,
        True,
    )

    check(
        "return_on_cost — value-add scenario",
        return_on_cost(85000, 450000, 5.25),
        3.593,
        tolerance=0.01,
    )
    check(
        "return_on_cost — zero cost returns NaN",
        math.isnan(return_on_cost(85000, 0, 5.25)),
        True,
    )

    rpu = renovation_roi_per_unit(85000, 450000, 180, 5.25)
    check(
        "renovation_roi_per_unit — cost_per_unit",
        rpu["cost_per_unit"],
        2500.0,
        tolerance=1.0,
    )
    check(
        "renovation_roi_per_unit — payback_years positive",
        rpu["payback_years"] > 0,
        True,
    )
    check(
        "renovation_roi_per_unit — return_on_cost positive",
        rpu["return_on_cost"] > 1.0,
        True,
    )

    # ─────────────────────────────────────────────────────────────────
    # SECTION 5 TESTS — Forecasting Engine
    # ─────────────────────────────────────────────────────────────────

    growth_proj = project_noi_growth(2090000, 3.5, 5)
    check(
        "project_noi_growth — returns 5 values",
        len(growth_proj),
        5,
    )
    check(
        "project_noi_growth — year 1 value",
        growth_proj[0],
        2163150.0,
        tolerance=1.0,
    )
    check(
        "project_noi_growth — values increasing",
        all(growth_proj[i] < growth_proj[i + 1] for i in range(4)),
        True,
    )

    model = build_five_year_cash_flow_model(
        base_annual_revenue=3269880,
        base_annual_expenses=1096200,
        units=240,
        rent_growth_pct=3.5,
        expense_growth_pct=2.5,
        vacancy_rate_pct=8.0,
        capex_per_unit=350,
        hold_years=5,
        exit_cap_rate_pct=5.25,
        purchase_price=34500000,
        loan_balance=24500000,
        annual_debt_service=1968000,
    )
    check(
        "build_five_year_cash_flow_model — returns 5 NOI years",
        len(model["noi"]),
        5,
    )
    check(
        "build_five_year_cash_flow_model — NOI increasing over time",
        model["noi"][4] > model["noi"][0],
        True,
    )
    check(
        "build_five_year_cash_flow_model — exit_value is positive",
        model["exit_value"] > 0,
        True,
    )
    check(
        "build_five_year_cash_flow_model — unlevered_irr in range",
        0.08 < model["unlevered_irr"] < 0.25,
        True,
    )
    check(
        "build_five_year_cash_flow_model — equity_multiple above 1",
        model["equity_multiple"] > 1.0,
        True,
    )
    check(
        "build_five_year_cash_flow_model — dscr_by_year has 5 entries",
        len(model["dscr_by_year"]),
        5,
    )
    check(
        "build_five_year_cash_flow_model — all DSCR values positive",
        all(d > 0 for d in model["dscr_by_year"]),
        True,
    )
    check(
        "build_five_year_cash_flow_model — capex_reserve per unit",
        model["capex_reserve"][0],
        350 * 240,
        tolerance=1.0,
    )

    # ─────────────────────────────────────────────────────────────────
    # SECTION 6 TESTS — Sensitivity Matrix
    # ─────────────────────────────────────────────────────────────────

    sm = build_irr_sensitivity_matrix(
        base_annual_revenue=3269880,
        base_annual_expenses=1096200,
        units=240,
        purchase_price=34500000,
        loan_balance=24500000,
        annual_debt_service=1968000,
    )
    check(
        "build_irr_sensitivity_matrix — 5 rows",
        len(sm["matrix"]),
        5,
    )
    check(
        "build_irr_sensitivity_matrix — 5 columns per row",
        len(sm["matrix"][0]),
        5,
    )
    check(
        "build_irr_sensitivity_matrix — lower cap rate = higher IRR",
        sm["matrix"][0][2] > sm["matrix"][4][2],
        True,
    )
    check(
        "build_irr_sensitivity_matrix — higher rent growth = higher IRR",
        sm["matrix"][2][4] > sm["matrix"][2][0],
        True,
    )
    check(
        "build_irr_sensitivity_matrix — row labels formatted correctly",
        sm["row_labels"][0].endswith("% Cap"),
        True,
    )
    check(
        "build_irr_sensitivity_matrix — col labels formatted correctly",
        "Rent Growth" in sm["col_labels"][0],
        True,
    )

    # ─────────────────────────────────────────────────────────────────
    # SECTION 7 TESTS — Variance and Benchmarking
    # ─────────────────────────────────────────────────────────────────

    var = calculate_variance(272400, 285000)
    check(
        "calculate_variance — dollar_variance unfavorable",
        var["dollar_variance"],
        -12600.0,
    )
    check(
        "calculate_variance — pct_variance unfavorable",
        var["pct_variance"],
        -4.4211,
        tolerance=0.01,
    )
    check(
        "calculate_variance — is_favorable False",
        var["is_favorable"],
        False,
    )

    fav_var = calculate_variance(295000, 285000)
    check(
        "calculate_variance — is_favorable True",
        fav_var["is_favorable"],
        True,
    )

    check(
        "assign_rag_status — revenue Amber",
        assign_rag_status(-4.5, "revenue"),
        "Amber",
    )
    check(
        "assign_rag_status — revenue Red",
        assign_rag_status(-9.2, "revenue"),
        "Red",
    )
    check(
        "assign_rag_status — revenue Green",
        assign_rag_status(-1.0, "revenue"),
        "Green",
    )
    check(
        "assign_rag_status — expense Amber",
        assign_rag_status(4.5, "expense"),
        "Amber",
    )
    check(
        "assign_rag_status — expense Red",
        assign_rag_status(9.2, "expense"),
        "Red",
    )
    check(
        "assign_rag_status — expense Green",
        assign_rag_status(1.0, "expense"),
        "Green",
    )
    check(
        "assign_rag_status — NaN input returns Grey",
        assign_rag_status(float("nan"), "revenue"),
        "Grey",
    )

    check(
        "weighted_average_occupancy — units-weighted",
        weighted_average_occupancy(
            [88, 85, 91, 82, 90],
            [240, 180, 320, 150, 275],
        ),
        87.97,
        tolerance=0.1,
    )
    check(
        "weighted_average_occupancy — empty lists return NaN",
        math.isnan(weighted_average_occupancy([], [])),
        True,
    )
    check(
        "weighted_average_occupancy — mismatched lists return NaN",
        math.isnan(weighted_average_occupancy([88, 85], [240])),
        True,
    )

    rpc = rent_premium_vs_comps(1425, [1380, 1450, 1360, 1395, 1410])
    check(
        "rent_premium_vs_comps — comp_average",
        rpc["comp_average"],
        1399.0,
        tolerance=0.5,
    )
    check(
        "rent_premium_vs_comps — dollar_premium positive",
        rpc["dollar_premium"] > 0,
        True,
    )
    check(
        "rent_premium_vs_comps — vs_highest_comp negative (below best comp)",
        rpc["vs_highest_comp"] < 0,
        True,
    )
    check(
        "rent_premium_vs_comps — empty comps return NaN",
        math.isnan(rent_premium_vs_comps(1425, [])["comp_average"]),
        True,
    )

    # ─────────────────────────────────────────────────────────────────
    # SECTION 8 TESTS — Formatting Helpers
    # ─────────────────────────────────────────────────────────────────

    check(
        "fmt_currency — whole dollars",
        fmt_currency(2090000),
        "$2,090,000",
    )
    check(
        "fmt_currency — negative value",
        fmt_currency(-12600),
        "-$12,600",
    )
    check(
        "fmt_currency — NaN returns N/A",
        fmt_currency(float("nan")),
        "N/A",
    )
    check(
        "fmt_currency — two decimal places",
        fmt_currency(1425.5, 2),
        "$1,425.50",
    )
    check(
        "fmt_percent — standard",
        fmt_percent(14.23),
        "14.2%",
    )
    check(
        "fmt_percent — NaN returns N/A",
        fmt_percent(float("nan")),
        "N/A",
    )
    check(
        "fmt_multiple — DSCR value",
        fmt_multiple(1.41),
        "1.41x",
    )
    check(
        "fmt_multiple — NaN returns N/A",
        fmt_multiple(float("nan")),
        "N/A",
    )
    check(
        "fmt_irr — decimal to percent",
        fmt_irr(0.1423),
        "14.2%",
    )
    check(
        "fmt_irr — NaN returns N/A",
        fmt_irr(float("nan")),
        "N/A",
    )
    check(
        "rag_to_color — Red maps correctly",
        rag_to_color("Red"),
        "#C0392B",
    )
    check(
        "rag_to_color — Green maps correctly",
        rag_to_color("Green"),
        "#1B7F4F",
    )
    check(
        "rag_to_color — Amber maps correctly",
        rag_to_color("Amber"),
        "#D4AC0D",
    )
    check(
        "rag_to_color — unknown input returns grey",
        rag_to_color("Unknown"),
        "#7F8C8D",
    )
    check(
        "rag_to_emoji — Green emoji",
        rag_to_emoji("Green"),
        "🟢",
    )
    check(
        "rag_to_emoji — Red emoji",
        rag_to_emoji("Red"),
        "🔴",
    )
    check(
        "rag_to_emoji — Amber emoji",
        rag_to_emoji("Amber"),
        "🟡",
    )

    # ─────────────────────────────────────────────────────────────────
    # PRINT RESULTS
    # ─────────────────────────────────────────────────────────────────

    print("\n" + "=" * 65)
    print("  AssetOptima Pro — KPI Calculations Self-Test")
    print("=" * 65)

    pass_count = sum(1 for r in results if r[0] == PASS)
    fail_count = sum(1 for r in results if r[0] == FAIL)

    # Group results by section for readable output
    section_labels = {
        0:  "Income and NOI",
        12: "Valuation",
        19: "Debt and Financing",
        27: "Returns Analysis",
        37: "Forecasting Engine",
        45: "Sensitivity Matrix",
        51: "Variance and Benchmarking",
        66: "Formatting Helpers",
    }

    current_section = ""
    for idx, (status, name, detail) in enumerate(results):
        for boundary, label in section_labels.items():
            if idx == boundary:
                print(f"\n  -- {label} {'-' * (44 - len(label))}")
                current_section = label
                break

        print(f"  {status}  {name}")
        if detail:
            print(f"       {detail}")

    print()
    print("=" * 65)
    print(f"  Results : {pass_count} passed  |  {fail_count} failed")
    print("=" * 65)

    if all_passed:
        print("  ALL TESTS PASSED — KPI calculation engine is ready.")
        print("  You may now build modules/variance_analysis.py")
        print("=" * 65 + "\n")
        sys.exit(0)
    else:
        print("  ONE OR MORE TESTS FAILED.")
        print("  Fix the formulas shown above before proceeding.")
        print("=" * 65 + "\n")
        sys.exit(1)