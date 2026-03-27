"""
Reusable UI helpers for AssetOptima Pro Streamlit pages.
"""

from __future__ import annotations

from typing import Any
import html
import re

import streamlit as st

from utils.constants import DEFAULT_STATUS_COLOR, STATUS_COLOR_MAP
from utils.coercion import ensure_dict, safe_str
from utils.formatters import fmt_days, fmt_multiple, fmt_pct


def rag_color(status: str) -> str:
    """
    Map a status label to a configured color.

    Matching is whitespace-trimmed and case-insensitive against configured keys.
    """
    normalized = str(status).strip().lower()
    normalized_map = {key.lower(): value for key, value in STATUS_COLOR_MAP.items()}
    return normalized_map.get(normalized, DEFAULT_STATUS_COLOR)


def extract_property_name(selected_property_display: str, property_id: str) -> str:
    """
    Extract a readable property name from a display string.
    Optimized to handle hyphens, en-dashes, and em-dashes safely.
    """
    parts = re.split(r"\s+[-–—]\s+", str(selected_property_display), maxsplit=1)

    if len(parts) == 2 and parts[1].strip():
        return parts[1].strip()

    return property_id


def render_warnings(warnings_list: list[str]) -> None:
    """
    Render a list of non-fatal load/model warnings.
    """
    for warning in warnings_list:
        st.warning(f"Partial load warning: {warning}")


def render_compliance_card(summary: dict[str, Any]) -> None:
    """
    Render a property-level compliance summary card.

    This is placed in ui_helpers because it is a reusable visual component.
    """
    summary = ensure_dict(summary)
    if not summary:
        st.info("No compliance summary available for this property.")
        return

    overall_color = rag_color(safe_str(summary.get("Overall_RAG", "Green"), "Green"))
    dscr_color = rag_color(safe_str(summary.get("DSCR_RAG", "Green"), "Green"))
    ltv_color = rag_color(safe_str(summary.get("LTV_RAG", "Green"), "Green"))
    label_color = rag_color(safe_str(summary.get("Compliance_Label", "Compliant"), "Compliant"))

    property_name = html.escape(safe_str(summary.get("Property_Name", summary.get("Property_ID", ""))))
    loan_name = html.escape(safe_str(summary.get("Loan_Name", "")))
    lender = html.escape(safe_str(summary.get("Lender", "")))
    label = html.escape(safe_str(summary.get("Compliance_Label", "Compliant"), "Compliant"))

    st.markdown(
        f"""
        <div style="
            border-left: 8px solid {overall_color};
            padding: 1rem;
            background-color: #F8F9FA;
            border-radius: 0.35rem;
            margin-bottom: 1rem;
        ">
            <div style="font-size: 1.1rem; font-weight: 700;">
                {property_name}
            </div>
            <div style="margin-top: 0.35rem;">
                <strong>Loan:</strong> {loan_name}<br>
                <strong>Lender:</strong> {lender}<br>
                <strong>Status:</strong>
                <span style="color: {label_color}; font-weight: 700;">
                    {label}
                </span>
            </div>
            <hr style="margin: 0.75rem 0;">
            <div>
                <strong>DSCR:</strong>
                <span style="color: {dscr_color}; font-weight: 700;">
                    {fmt_multiple(summary.get("DSCR_Actual", 0.0))}
                </span>
                vs. req. {fmt_multiple(summary.get("DSCR_Requirement", 0.0))}<br>
                <strong>LTV:</strong>
                <span style="color: {ltv_color}; font-weight: 700;">
                    {fmt_pct(summary.get("LTV_Actual", 0.0))}
                </span>
                vs. max {fmt_pct(summary.get("LTV_Max", 0.0))}<br>
                <strong>Reporting:</strong> {fmt_days(summary.get("Days_to_Reporting"))}<br>
                <strong>Maturity:</strong> {fmt_days(summary.get("Days_to_Maturity"))}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )