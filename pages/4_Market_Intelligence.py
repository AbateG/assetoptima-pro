"""
pages/4_Market_Intelligence.py
----------------------------------

Advanced market competitive analysis dashboard.

PATCHES / IMPROVEMENTS IN THIS VERSION:
- Safer individually-guarded imports for modules.market_analysis functions
- Corrected Market Trends chart:
  * top panel uses dual y-axes (Market Rent $, Occupancy %)
  * bottom panel uses percent axis for growth rates
- Better map messaging with actionable install instructions
- Stronger chart robustness:
  * safer numeric coercion
  * per-series percentage normalization helpers
  * safer subject matching by Property_ID when available
  * better chart fallbacks on malformed upstream data
- Preserved previous fixes:
  * Pydantic mutable defaults
  * O(n) trace bug fix
  * optional dependencies
  * TypedDict property_id key
  * parse_float misuse fix
  * logging and cache behavior
"""

from __future__ import annotations

import functools
import logging
import time
import traceback
from contextlib import contextmanager
from collections.abc import Generator, Callable
from typing import Any, TypedDict, Final, ClassVar, cast
from enum import Enum
from datetime import datetime

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
import streamlit as st
from pydantic import BaseModel, Field, ValidationError

# ---------------------------------------------------------------------------
# Module-level logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional heavy dependencies — degrade gracefully if missing
# ---------------------------------------------------------------------------
try:
    import cachetools
    _CACHETOOLS_AVAILABLE = True
except ImportError:
    _CACHETOOLS_AVAILABLE = False
    logger.warning("cachetools not installed — using dict-based fallback cache")

try:
    import folium
    from streamlit_folium import folium_static
    _FOLIUM_AVAILABLE = True
except ImportError:
    _FOLIUM_AVAILABLE = False
    logger.warning("folium / streamlit_folium not installed — map visualization disabled")

try:
    from scipy import stats as _scipy_stats  # noqa: F401
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False
    logger.warning("scipy not installed — statistical features limited")

try:
    import geopandas as gpd  # noqa: F401
    _GEOPANDAS_AVAILABLE = True
except ImportError:
    _GEOPANDAS_AVAILABLE = False
    logger.warning("geopandas not installed — geographic features limited")

# ---------------------------------------------------------------------------
# Internal modules
# ---------------------------------------------------------------------------
from modules.data_loader import get_property_display_options, parse_property_selection
from utils.coercion import (
    ensure_dataframe,
    ensure_dict,
    ensure_list,
    parse_float,
    parse_int,
    safe_str,
)
from utils.formatters import fmt_currency, fmt_pct, format_metric_value
from utils.ui_helpers import extract_property_name, rag_color, render_warnings
from utils.validators import has_columns, validate_market_data, check_data_integrity

# ---------------------------------------------------------------------------
# Individually guarded imports for modules.market_analysis
# ---------------------------------------------------------------------------

def _fallback_empty_dict(*args: Any, **kwargs: Any) -> dict[str, Any]:
    return {}


def _fallback_empty_df(*args: Any, **kwargs: Any) -> pd.DataFrame:
    return pd.DataFrame()


def _fallback_empty_list(*args: Any, **kwargs: Any) -> list[Any]:
    return []


def _fallback_empty_str(*args: Any, **kwargs: Any) -> str:
    return ""


def _safe_import_market_analysis(
    func_name: str, fallback: Callable[..., Any]
) -> Callable[..., Any]:
    try:
        module = __import__("modules.market_analysis", fromlist=[func_name])
        func = getattr(module, func_name, None)
        if callable(func):
            return cast(Callable[..., Any], func)
        logger.warning(
            "modules.market_analysis.%s is missing or not callable — using fallback",
            func_name,
        )
        return fallback
    except Exception as exc:
        logger.warning(
            "Failed to import modules.market_analysis.%s: %s — using fallback",
            func_name,
            exc,
        )
        return fallback


generate_market_narrative = _safe_import_market_analysis(
    "generate_market_narrative", _fallback_empty_str
)
get_comp_table = _safe_import_market_analysis(
    "get_comp_table", _fallback_empty_df
)
get_competitive_positioning_score = _safe_import_market_analysis(
    "get_competitive_positioning_score", _fallback_empty_dict
)
get_market_watchlist_flags = _safe_import_market_analysis(
    "get_market_watchlist_flags", _fallback_empty_list
)
get_occupancy_comparison_data = _safe_import_market_analysis(
    "get_occupancy_comparison_data", _fallback_empty_df
)
get_rent_distribution_data = _safe_import_market_analysis(
    "get_rent_distribution_data", _fallback_empty_df
)
get_subject_vs_comp_summary = _safe_import_market_analysis(
    "get_subject_vs_comp_summary", _fallback_empty_dict
)
get_market_trends = _safe_import_market_analysis(
    "get_market_trends", _fallback_empty_dict
)
get_peer_benchmarking = _safe_import_market_analysis(
    "get_peer_benchmarking", _fallback_empty_df
)
get_market_heatmap_data = _safe_import_market_analysis(
    "get_market_heatmap_data", _fallback_empty_dict
)
generate_market_sentiment = _safe_import_market_analysis(
    "generate_market_sentiment", _fallback_empty_dict
)
get_market_forecast_indicators = _safe_import_market_analysis(
    "get_market_forecast_indicators", _fallback_empty_dict
)
get_geographic_competition_data = _safe_import_market_analysis(
    "get_geographic_competition_data", _fallback_empty_dict
)


# =============================================================================
# CONFIGURATION & TYPES
# =============================================================================

class MarketConfig(BaseModel):
    """Market Intelligence configuration with safe mutable defaults."""

    page_title: str = "Market Intelligence | AssetOptima Pro"
    page_icon: str = "📍"
    cache_ttl: int = 600
    cache_max_entries: int = 50
    max_comps_to_display: int = 20
    enable_map_visualization: bool = True
    enable_sentiment_analysis: bool = True
    enable_market_forecasting: bool = True
    chart_height: int = 500

    default_assumptions: dict[str, float] = Field(
        default_factory=lambda: {
            "rent_growth_forecast": 0.03,
            "vacancy_trend": -0.01,
            "supply_growth": 0.02,
            "demand_growth": 0.025,
            "cap_rate_trend": -0.0025,
        }
    )
    theme_colors: dict[str, str] = Field(
        default_factory=lambda: {
            "primary": "#1B4F72",
            "secondary": "#7F8C8D",
            "subject": "#1B4F72",
            "comp": "#85C1E9",
            "positive": "#27AE60",
            "negative": "#E74C3C",
            "warning": "#F39C12",
            "neutral": "#95A5A6",
            "map_primary": "#2980B9",
            "map_secondary": "#3498DB",
        }
    )
    positioning_thresholds: dict[str, tuple[float, float, str]] = Field(
        default_factory=lambda: {
            "excellent": (80.0, 100.0, "🟢"),
            "good": (60.0, 80.0, "🟡"),
            "fair": (40.0, 60.0, "🟠"),
            "poor": (0.0, 40.0, "🔴"),
        }
    )


class MarketBundle(TypedDict, total=False):
    property_id: str
    summary: dict[str, Any]
    comp_table: pd.DataFrame
    rent_distribution: pd.DataFrame
    occupancy_comparison: pd.DataFrame
    positioning: dict[str, Any]
    narrative: str
    flags: list[str]
    warnings: list[str]
    market_trends: dict[str, Any]
    peer_benchmarking: pd.DataFrame
    market_heatmap: dict[str, Any]
    market_sentiment: dict[str, Any]
    forecast_indicators: dict[str, Any]
    geographic_data: dict[str, Any]
    last_updated: datetime


class MarketMetrics(TypedDict):
    subject_rent: float
    comp_avg_rent: float
    rent_premium_pct: float
    subject_occupancy: float
    occupancy_gap: float
    comp_avg_occupancy: float
    rent_growth_gap: float
    comp_count: int
    positioning_score: float
    market_rank: int
    market_share_pct: float
    revenue_growth_yoy: float
    noi_growth_yoy: float


class PositioningCategory(str, Enum):
    EXCELLENT = "Excellent"
    GOOD = "Good"
    FAIR = "Fair"
    POOR = "Poor"


CONFIG: Final[MarketConfig] = MarketConfig()
pio.templates.default = "plotly_white"


# =============================================================================
# PERFORMANCE MONITORING
# =============================================================================

class MarketPerformanceMonitor:
    """Singleton for market analysis performance monitoring."""

    _instance: ClassVar["MarketPerformanceMonitor | None"] = None

    def __new__(cls) -> "MarketPerformanceMonitor":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._timings = {}
            cls._instance._market_loads = {}
        return cls._instance

    @contextmanager
    def time_market_analysis(
        self, property_id: str | None, analysis_type: str
    ) -> Generator[None, None, None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            key = f"{property_id}:{analysis_type}"
            self._timings[key] = elapsed

    def record_market_load(self, property_id: str, load_time: float) -> None:
        self._market_loads[property_id] = {
            "load_time": load_time,
            "timestamp": datetime.now(),
        }

    def get_stats(self) -> dict[str, Any]:
        return {
            "timings": self._timings,
            "market_loads": self._market_loads,
            "total_properties_analyzed": len(self._market_loads),
        }


def monitor_market_analysis(func: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        property_id: str | None = kwargs.get("property_id") or (
            args[0] if args and isinstance(args[0], str) else None
        )
        analysis_type = kwargs.get("analysis_type", func.__name__)
        monitor = MarketPerformanceMonitor()
        with monitor.time_market_analysis(property_id, analysis_type):
            return func(*args, **kwargs)

    return wrapper


# =============================================================================
# CACHING
# =============================================================================

class _DictCache:
    """Minimal dict-based fallback when cachetools is unavailable."""

    def __init__(self, maxsize: int = 200) -> None:
        self._data: dict[str, Any] = {}
        self.maxsize = maxsize

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def __setitem__(self, key: str, value: Any) -> None:
        if len(self._data) >= self.maxsize:
            self._data.pop(next(iter(self._data)))
        self._data[key] = value

    def __delitem__(self, key: str) -> None:
        del self._data[key]

    def __len__(self) -> int:
        return len(self._data)

    def keys(self):  # noqa: ANN201
        return self._data.keys()

    def clear(self) -> None:
        self._data.clear()


class MarketCache:
    """Cache manager for market data with optional cachetools backend."""

    def __init__(self, maxsize: int = 200) -> None:
        if _CACHETOOLS_AVAILABLE:
            self._cache: Any = cachetools.LRUCache(maxsize=maxsize)
        else:
            self._cache = _DictCache(maxsize=maxsize)
        self._sentiment_cache: dict[str, dict[str, Any]] = {}

    def get(self, property_id: str, data_type: str) -> Any | None:
        return self._cache.get(f"{property_id}:{data_type}")

    def set(self, property_id: str, data_type: str, value: Any) -> None:
        self._cache[f"{property_id}:{data_type}"] = value

    def get_sentiment(self, market_id: str) -> dict[str, Any] | None:
        return self._sentiment_cache.get(market_id)

    def set_sentiment(self, market_id: str, sentiment: dict[str, Any]) -> None:
        self._sentiment_cache[market_id] = sentiment

    def clear_property(self, property_id: str) -> None:
        keys_to_remove = [
            k for k in list(self._cache.keys()) if k.startswith(f"{property_id}:")
        ]
        for key in keys_to_remove:
            del self._cache[key]

    def clear_all(self) -> None:
        self._cache.clear()
        self._sentiment_cache.clear()

    def stats(self) -> dict[str, Any]:
        all_keys = list(self._cache.keys())
        return {
            "size": len(self._cache),
            "maxsize": getattr(self._cache, "maxsize", "N/A"),
            "properties_cached": len(set(k.split(":")[0] for k in all_keys)),
            "sentiment_cached": len(self._sentiment_cache),
        }


market_cache = MarketCache(maxsize=200)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _safe_numeric_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def _normalize_percent_series(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.apply(lambda x: x / 100.0 if pd.notna(x) and abs(x) > 1 else x)


def _normalize_occupancy_series(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.apply(lambda x: x / 100.0 if pd.notna(x) and x > 1 else x)


def _find_subject_mask(
    df: pd.DataFrame,
    subject_label: str,
    property_id: str | None = None,
    label_col: str = "Label",
    name_col: str = "Property_Name",
) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=bool)

    if property_id and "Property_ID" in df.columns:
        return df["Property_ID"].astype(str) == str(property_id)

    if label_col in df.columns:
        return df[label_col].astype(str).str.strip() == str(subject_label).strip()

    if name_col in df.columns:
        return df[name_col].astype(str).str.strip() == str(subject_label).strip()

    return pd.Series(False, index=df.index)


def _append_validation_warnings(bundle: MarketBundle) -> None:
    try:
        summary = ensure_dict(bundle.get("summary", {}))
        comp_table = ensure_dataframe(bundle.get("comp_table"))
        result_summary = validate_market_data(summary)
        result_comp = check_data_integrity(comp_table)

        if result_summary is False:
            bundle.setdefault("warnings", []).append("Summary validation failed.")
        if result_comp is False:
            bundle.setdefault("warnings", []).append("Competitive set integrity check failed.")
    except Exception as exc:
        logger.warning("Validation helpers raised an exception: %s", exc)


# =============================================================================
# DATA LOADING
# =============================================================================

@st.cache_data(
    show_spinner=True,
    max_entries=CONFIG.cache_max_entries,
    ttl=CONFIG.cache_ttl,
    persist="disk",
)
def get_property_options() -> list[str]:
    """Get property display options with enhanced error handling."""
    try:
        options = get_property_display_options()
        if not isinstance(options, list):
            logger.warning("Property options not a list — converting")
            options = list(options) if hasattr(options, "__iter__") else []
        return sorted(set(str(opt) for opt in options if opt))
    except Exception as exc:
        logger.error("Failed to load property options: %s", exc)
        return []


@st.cache_data(
    show_spinner=False,
    max_entries=CONFIG.cache_max_entries,
    ttl=CONFIG.cache_ttl,
)
def load_market_bundle(property_id: str) -> MarketBundle:
    """
    Load comprehensive market analysis bundle.
    """
    start_time = time.perf_counter()

    bundle: MarketBundle = {
        "property_id": property_id,
        "summary": {},
        "comp_table": pd.DataFrame(),
        "rent_distribution": pd.DataFrame(),
        "occupancy_comparison": pd.DataFrame(),
        "positioning": {},
        "narrative": "",
        "flags": [],
        "warnings": [],
        "market_trends": {},
        "peer_benchmarking": pd.DataFrame(),
        "market_heatmap": {},
        "market_sentiment": {},
        "forecast_indicators": {},
        "geographic_data": {},
        "last_updated": datetime.now(),
    }

    core_loaders: dict[str, Callable[[], Any]] = {
        "summary": lambda: get_subject_vs_comp_summary(property_id),
        "comp_table": lambda: get_comp_table(property_id),
        "rent_distribution": lambda: get_rent_distribution_data(property_id),
        "occupancy_comparison": lambda: get_occupancy_comparison_data(property_id),
        "positioning": lambda: get_competitive_positioning_score(property_id),
        "narrative": lambda: generate_market_narrative(property_id),
        "flags": lambda: get_market_watchlist_flags(property_id),
    }

    for key, loader in core_loaders.items():
        try:
            result = loader()
            if result is not None:
                bundle[key] = result  # type: ignore[literal-required]
        except Exception as exc:
            logger.error("Failed loading %s for %s: %s", key, property_id, exc)
            bundle["warnings"].append(
                f"{key.replace('_', ' ').title()} failed to load"
            )

    advanced_loaders: dict[str, Callable[[], Any]] = {
        "market_trends": lambda: get_market_trends(property_id),
        "peer_benchmarking": lambda: get_peer_benchmarking(property_id),
        "market_heatmap": lambda: get_market_heatmap_data(property_id),
        "market_sentiment": lambda: generate_market_sentiment(property_id),
        "forecast_indicators": lambda: get_market_forecast_indicators(property_id),
        "geographic_data": lambda: get_geographic_competition_data(property_id),
    }

    _sentiment_keys = {"market_sentiment", "forecast_indicators"}
    _geo_keys = {"geographic_data"}
    _forecast_keys = {"forecast_indicators"}

    for key, loader in advanced_loaders.items():
        if key in _sentiment_keys and not CONFIG.enable_sentiment_analysis:
            continue
        if key in _geo_keys and not CONFIG.enable_map_visualization:
            continue
        if key in _forecast_keys and not CONFIG.enable_market_forecasting:
            continue

        try:
            result = loader()
            if result is not None:
                bundle[key] = result  # type: ignore[literal-required]
        except Exception as exc:
            logger.warning("Optional loader %s failed: %s", key, exc)

    for df_key in (
        "comp_table",
        "rent_distribution",
        "occupancy_comparison",
        "peer_benchmarking",
    ):
        bundle[df_key] = ensure_dataframe(bundle.get(df_key))  # type: ignore[literal-required]

    for dict_key in (
        "summary",
        "positioning",
        "market_trends",
        "market_heatmap",
        "market_sentiment",
        "forecast_indicators",
        "geographic_data",
    ):
        bundle[dict_key] = ensure_dict(bundle.get(dict_key))  # type: ignore[literal-required]

    bundle["narrative"] = safe_str(bundle.get("narrative", ""))
    bundle["flags"] = [safe_str(x) for x in ensure_list(bundle.get("flags", []))]
    bundle["warnings"] = [safe_str(x) for x in ensure_list(bundle.get("warnings", []))]

    _append_validation_warnings(bundle)

    load_time = time.perf_counter() - start_time
    MarketPerformanceMonitor().record_market_load(property_id, load_time)

    return bundle


def extract_market_metrics(bundle: MarketBundle) -> MarketMetrics:
    """
    Extract and validate market metrics from a bundle.
    """
    summary = ensure_dict(bundle.get("summary", {}))
    positioning = ensure_dict(bundle.get("positioning", {}))
    peer_benchmarking = ensure_dataframe(bundle.get("peer_benchmarking"))

    metrics: MarketMetrics = {
        "subject_rent": parse_float(summary.get("Subject_Rent", 0.0)),
        "comp_avg_rent": parse_float(summary.get("Comp_Avg_Rent", 0.0)),
        "rent_premium_pct": parse_float(summary.get("Rent_Premium_Pct", 0.0)),
        "subject_occupancy": parse_float(summary.get("Subject_Occupancy", 0.0)),
        "occupancy_gap": parse_float(summary.get("Occupancy_Gap", 0.0)),
        "comp_avg_occupancy": parse_float(summary.get("Comp_Avg_Occupancy", 0.0)),
        "rent_growth_gap": parse_float(summary.get("Rent_Growth_Gap", 0.0)),
        "comp_count": parse_int(summary.get("Comp_Count", 0)),
        "positioning_score": parse_float(positioning.get("Positioning_Score", 0.0)),
        "market_rank": parse_int(positioning.get("Market_Rank", 0)),
        "market_share_pct": parse_float(summary.get("Market_Share_Pct", 0.0)),
        "revenue_growth_yoy": parse_float(summary.get("Revenue_Growth_YoY", 0.0)),
        "noi_growth_yoy": parse_float(summary.get("NOI_Growth_YoY", 0.0)),
    }

    if metrics["subject_occupancy"] > 1:
        metrics["subject_occupancy"] /= 100.0
    if metrics["comp_avg_occupancy"] > 1:
        metrics["comp_avg_occupancy"] /= 100.0
    if abs(metrics["occupancy_gap"]) > 1:
        metrics["occupancy_gap"] /= 100.0

    if metrics["market_rank"] == 0 and not peer_benchmarking.empty:
        pid = bundle.get("property_id", "")
        if "Market_Rank" in peer_benchmarking.columns and "Property_ID" in peer_benchmarking.columns:
            subject_rows = peer_benchmarking[
                peer_benchmarking["Property_ID"].astype(str) == str(pid)
            ]
            if not subject_rows.empty:
                metrics["market_rank"] = parse_int(subject_rows["Market_Rank"].iloc[0])

    return metrics


def get_positioning_category(score: float) -> tuple[str, str, str]:
    """
    Return (category_label, emoji_icon, hex_color) for a positioning score.
    """
    score = max(0.0, min(100.0, score))

    for category, (low, high, icon) in CONFIG.positioning_thresholds.items():
        if category == "excellent":
            if low <= score <= high:
                return category.title(), icon, rag_color(category.title())
        else:
            if low <= score < high:
                return category.title(), icon, rag_color(category.title())

    return "Unknown", "⚪", "#95A5A6"


# =============================================================================
# VISUALIZATION COMPONENTS
# =============================================================================

def build_rent_distribution_chart(
    df: pd.DataFrame,
    subject_label: str,
    property_id: str | None = None,
    show_percentiles: bool = True,
) -> go.Figure:
    """Build enhanced rent comparison chart with percentile reference lines."""
    df = ensure_dataframe(df)
    if df.empty or not has_columns(df, ["Label", "Monthly_Rent"]):
        return go.Figure()

    chart_df = df.copy()
    chart_df["Monthly_Rent"] = pd.to_numeric(chart_df["Monthly_Rent"], errors="coerce")
    chart_df = chart_df.dropna(subset=["Monthly_Rent"])
    if chart_df.empty:
        return go.Figure()

    is_subject = _find_subject_mask(chart_df, subject_label, property_id, label_col="Label")
    subject_data = chart_df[is_subject]
    comp_data = chart_df[~is_subject]

    fig = go.Figure()

    if not comp_data.empty:
        fig.add_trace(
            go.Bar(
                x=comp_data["Label"],
                y=comp_data["Monthly_Rent"],
                name="Competitors",
                marker_color=CONFIG.theme_colors["comp"],
                opacity=0.7,
                hovertemplate="<b>%{x}</b><br>Rent: $%{y:,.0f}<extra></extra>",
            )
        )

    if not subject_data.empty:
        first_label = (
            subject_data["Label"].iloc[0]
            if "Label" in subject_data.columns and not subject_data.empty
            else subject_label
        )
        fig.add_trace(
            go.Scatter(
                x=[first_label],
                y=[subject_data["Monthly_Rent"].iloc[0]],
                mode="markers",
                name="Subject Property",
                marker=dict(
                    color=CONFIG.theme_colors["subject"],
                    size=20,
                    symbol="star",
                    line=dict(color="white", width=2),
                ),
                hovertemplate="<b>Subject Property</b><br>Rent: $%{y:,.0f}<extra></extra>",
            )
        )

    if show_percentiles and not comp_data.empty:
        rents = comp_data["Monthly_Rent"].dropna()
        if len(rents) >= 5:
            for pct in (25, 50, 75):
                value = float(np.percentile(rents, pct))
                fig.add_hline(
                    y=value,
                    line_dash="dash",
                    line_color="gray",
                    opacity=0.5,
                    annotation_text=f"P{pct}: ${value:,.0f}",
                    annotation_position="right",
                )

    fig.update_layout(
        height=CONFIG.chart_height,
        title="Rent Distribution: Subject vs Competitive Set",
        xaxis_title="Property",
        yaxis_title="Monthly Rent ($)",
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=True,
        hovermode="x unified",
        xaxis=dict(tickangle=-45),
    )
    return fig


def build_occupancy_chart(
    df: pd.DataFrame,
    subject_label: str,
    property_id: str | None = None,
) -> go.Figure:
    """
    Build enhanced occupancy comparison chart with per-row normalization.
    """
    df = ensure_dataframe(df)
    if df.empty or not has_columns(df, ["Label", "Occupancy"]):
        return go.Figure()

    chart_df = df.copy()
    chart_df["Occupancy"] = _normalize_occupancy_series(chart_df["Occupancy"])
    chart_df = chart_df.dropna(subset=["Occupancy"])
    if chart_df.empty:
        return go.Figure()

    is_subject = _find_subject_mask(chart_df, subject_label, property_id, label_col="Label")
    subject_data = chart_df[is_subject]
    comp_data = chart_df[~is_subject]

    fig = go.Figure()

    if not comp_data.empty:
        fig.add_trace(
            go.Bar(
                x=comp_data["Label"],
                y=comp_data["Occupancy"],
                name="Competitors",
                marker_color=CONFIG.theme_colors["comp"],
                opacity=0.7,
                hovertemplate="<b>%{x}</b><br>Occupancy: %{y:.1%}<extra></extra>",
            )
        )

    if not subject_data.empty:
        fig.add_trace(
            go.Bar(
                x=[subject_data["Label"].iloc[0]],
                y=[subject_data["Occupancy"].iloc[0]],
                name="Subject Property",
                marker_color=CONFIG.theme_colors["subject"],
                hovertemplate="<b>Subject Property</b><br>Occupancy: %{y:.1%}<extra></extra>",
            )
        )

    if not comp_data.empty:
        market_avg = float(comp_data["Occupancy"].mean())
        fig.add_hline(
            y=market_avg,
            line_dash="dash",
            line_color=CONFIG.theme_colors["neutral"],
            annotation_text=f"Market Avg: {market_avg:.1%}",
            annotation_position="right",
        )

    fig.update_layout(
        height=CONFIG.chart_height,
        title="Occupancy Comparison: Subject vs Competitive Set",
        xaxis_title="Property",
        yaxis_title="Occupancy",
        yaxis_tickformat=".0%",
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=True,
        hovermode="x unified",
        xaxis=dict(tickangle=-45),
    )
    return fig


def build_positioning_radar_chart(positioning: dict[str, Any]) -> go.Figure:
    """Build radar chart for competitive positioning."""
    positioning = ensure_dict(positioning)
    if not positioning:
        return go.Figure()

    categories = ["Rent", "Occupancy", "Growth", "Location", "Quality", "Amenities"]
    subscore_keys = [
        "Rent_Subscore",
        "Occupancy_Subscore",
        "Growth_Subscore",
        "Distance_Subscore",
        "Quality_Subscore",
        "Amenity_Subscore",
    ]

    values = [max(0.0, min(100.0, parse_float(positioning.get(k, 0.0)))) for k in subscore_keys]

    fig = go.Figure(
        data=go.Scatterpolar(
            r=values,
            theta=categories,
            fill="toself",
            fillcolor="rgba(26, 82, 118, 0.3)",
            line_color=CONFIG.theme_colors["primary"],
            name="Positioning Subscores",
        )
    )
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        height=400,
        title="Competitive Positioning Radar Chart",
        showlegend=True,
    )
    return fig


def build_market_heatmap_visualization(heatmap_data: dict[str, Any]) -> go.Figure:
    """Build market heatmap visualization."""
    if not heatmap_data or "matrix" not in heatmap_data:
        return go.Figure()

    matrix_df = ensure_dataframe(heatmap_data["matrix"])
    if matrix_df.empty:
        return go.Figure()

    try:
        z_values = matrix_df.apply(pd.to_numeric, errors="coerce").values
    except Exception:
        z_values = matrix_df.values

    fig = go.Figure(
        data=go.Heatmap(
            z=z_values,
            x=list(matrix_df.columns),
            y=list(matrix_df.index),
            colorscale="RdYlGn",
            text=z_values,
            texttemplate="%{text:.2f}",
            textfont={"size": 10},
            hovertemplate=(
                "Metric: %{y}<br>Property: %{x}<br>Score: %{z:.2f}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        height=500,
        title="Market Competitive Heatmap",
        xaxis_title="Properties",
        yaxis_title="Metrics",
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    return fig


def build_market_trends_chart(market_trends: dict[str, Any]) -> go.Figure:
    """
    Build market trends chart with better axis semantics.

    Top panel:
    - Market Rent on left y-axis ($)
    - Occupancy on right y-axis (%)

    Bottom panel:
    - Rent Growth and NOI Growth on percent axis
    """
    if not market_trends or "trend_data" not in market_trends:
        return go.Figure()

    trend_df = ensure_dataframe(market_trends["trend_data"])
    if trend_df.empty or "Period" not in trend_df.columns:
        return go.Figure()

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.12,
        subplot_titles=("Market Levels", "Growth Rates"),
        specs=[[{"secondary_y": True}], [{"secondary_y": False}]],
    )

    period = trend_df["Period"]

    if "Market_Rent" in trend_df.columns:
        market_rent = pd.to_numeric(trend_df["Market_Rent"], errors="coerce")
        fig.add_trace(
            go.Scatter(
                x=period,
                y=market_rent,
                mode="lines+markers",
                name="Market Rent",
                line=dict(color=CONFIG.theme_colors["primary"], width=2),
                hovertemplate="Period: %{x}<br>Market Rent: $%{y:,.0f}<extra></extra>",
            ),
            row=1,
            col=1,
            secondary_y=False,
        )

    if "Occupancy" in trend_df.columns:
        occupancy = _normalize_occupancy_series(trend_df["Occupancy"])
        fig.add_trace(
            go.Scatter(
                x=period,
                y=occupancy,
                mode="lines+markers",
                name="Occupancy",
                line=dict(color=CONFIG.theme_colors["positive"], width=2),
                hovertemplate="Period: %{x}<br>Occupancy: %{y:.1%}<extra></extra>",
            ),
            row=1,
            col=1,
            secondary_y=True,
        )

    if "Rent_Growth" in trend_df.columns:
        rent_growth = _normalize_percent_series(trend_df["Rent_Growth"])
        fig.add_trace(
            go.Scatter(
                x=period,
                y=rent_growth,
                mode="lines+markers",
                name="Rent Growth",
                line=dict(color=CONFIG.theme_colors["warning"], width=2),
                hovertemplate="Period: %{x}<br>Rent Growth: %{y:.2%}<extra></extra>",
            ),
            row=2,
            col=1,
        )

    if "NOI_Growth" in trend_df.columns:
        noi_growth = _normalize_percent_series(trend_df["NOI_Growth"])
        fig.add_trace(
            go.Scatter(
                x=period,
                y=noi_growth,
                mode="lines+markers",
                name="NOI Growth",
                line=dict(color=CONFIG.theme_colors["secondary"], width=2),
                hovertemplate="Period: %{x}<br>NOI Growth: %{y:.2%}<extra></extra>",
            ),
            row=2,
            col=1,
        )

    fig.update_yaxes(
        title_text="Market Rent ($)",
        row=1,
        col=1,
        secondary_y=False,
    )
    fig.update_yaxes(
        title_text="Occupancy",
        tickformat=".0%",
        row=1,
        col=1,
        secondary_y=True,
    )
    fig.update_yaxes(
        title_text="Growth",
        tickformat=".1%",
        row=2,
        col=1,
    )

    fig.update_layout(
        height=650,
        title="Market Trends Over Time",
        xaxis2_title="Period",
        plot_bgcolor="white",
        paper_bgcolor="white",
        hovermode="x unified",
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
        ),
        margin=dict(t=80, l=50, r=50, b=50),
    )
    return fig


def build_peer_benchmarking_chart(
    peer_benchmarking: pd.DataFrame,
    property_name: str,
    property_id: str | None = None,
) -> go.Figure:
    """
    Build peer benchmarking chart using two vectorized traces.
    """
    peer_benchmarking = ensure_dataframe(peer_benchmarking)
    if peer_benchmarking.empty or "Property_Name" not in peer_benchmarking.columns:
        return go.Figure()

    chart_df = peer_benchmarking.copy()
    if "Benchmark_Score" in chart_df.columns:
        chart_df["Benchmark_Score"] = pd.to_numeric(
            chart_df["Benchmark_Score"], errors="coerce"
        )
        chart_df = chart_df.sort_values("Benchmark_Score", ascending=False)

    chart_df = chart_df.head(CONFIG.max_comps_to_display)

    is_subject_mask = _find_subject_mask(
        chart_df,
        property_name,
        property_id,
        label_col="Property_Name",
        name_col="Property_Name",
    )
    comp_df = chart_df[~is_subject_mask]
    subject_df = chart_df[is_subject_mask]

    fig = go.Figure()

    if not comp_df.empty:
        fig.add_trace(
            go.Bar(
                x=comp_df["Property_Name"],
                y=comp_df.get("Benchmark_Score", pd.Series(dtype=float)),
                name="Competitors",
                marker_color=CONFIG.theme_colors["comp"],
                hovertemplate="<b>%{x}</b><br>Score: %{y:.1f}<extra></extra>",
            )
        )

    if not subject_df.empty:
        fig.add_trace(
            go.Bar(
                x=subject_df["Property_Name"],
                y=subject_df.get("Benchmark_Score", pd.Series(dtype=float)),
                name="Subject Property",
                marker_color=CONFIG.theme_colors["subject"],
                hovertemplate="<b>%{x}</b><br>Score: %{y:.1f}<extra></extra>",
            )
        )

    fig.update_layout(
        height=CONFIG.chart_height,
        title="Peer Benchmarking Analysis",
        xaxis_title="Property",
        yaxis_title="Benchmark Score",
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=True,
        barmode="overlay",
        xaxis=dict(tickangle=-45),
    )
    return fig


def build_market_map(
    geographic_data: dict[str, Any], property_name: str
) -> "folium.Map | None":
    """
    Build interactive market map.
    """
    if not _FOLIUM_AVAILABLE:
        logger.warning("folium not installed — skipping map build")
        return None

    if not geographic_data or "coordinates" not in geographic_data:
        logger.info("No geographic coordinates available for map")
        return None

    try:
        coords = geographic_data["coordinates"]
        subject_coords: dict[str, float] = coords.get(
            "subject", {"lat": 39.8283, "lon": -98.5795}
        )
        s_lat = subject_coords.get("lat", 39.8283)
        s_lon = subject_coords.get("lon", -98.5795)

        market_map = folium.Map(
            location=[s_lat, s_lon],
            zoom_start=12,
            tiles="CartoDB positron",
        )

        folium.Marker(
            [s_lat, s_lon],
            popup=f"<b>{property_name}</b><br>Subject Property",
            icon=folium.Icon(color="blue", icon="star", prefix="fa"),
            tooltip="Subject Property",
        ).add_to(market_map)

        for comp in coords.get("competitors", []):
            c_lat = comp.get("lat")
            c_lon = comp.get("lon")
            if c_lat is None or c_lon is None:
                logger.warning(
                    "Competitor %s missing lat/lon — skipping",
                    comp.get("name"),
                )
                continue

            folium.Marker(
                [c_lat, c_lon],
                popup=(
                    f"<b>{comp.get('name', 'Competitor')}</b>"
                    f"<br>Rent: ${parse_float(comp.get('rent', 0)):,.0f}"
                ),
                icon=folium.Icon(color="gray", icon="home"),
                tooltip=comp.get("name", "Competitor"),
            ).add_to(market_map)

        if "market_boundary" in geographic_data:
            folium.Polygon(
                locations=geographic_data["market_boundary"],
                color="blue",
                fill=True,
                fill_color="blue",
                fill_opacity=0.1,
                popup="Market Boundary",
            ).add_to(market_map)

        return market_map

    except Exception as exc:
        logger.error("Failed to build market map: %s", exc)
        return None


# =============================================================================
# DATA DISPLAY BUILDERS
# =============================================================================

def build_comp_display(df: pd.DataFrame) -> pd.DataFrame:
    """Build enhanced competitive set detail table."""
    df = ensure_dataframe(df)
    if df.empty:
        return df

    display_df = df.copy()

    for col in ["Market_Rent", "Rent_Premium_vs_Comp", "Value_Per_Unit", "NOI"]:
        if col in display_df.columns:
            include_sign = col == "Rent_Premium_vs_Comp"
            display_df[col] = display_df[col].apply(
                lambda x, _sign=include_sign: fmt_currency(x, include_sign=_sign)
            )

    pct_sign_cols = {"Rent_Premium_Pct", "Occupancy_Gap_vs_Comp"}
    for col in [
        "Occupancy",
        "Rent_Growth_YoY",
        "Rent_Premium_Pct",
        "Occupancy_Gap_vs_Comp",
        "Revenue_Growth",
        "NOI_Margin",
    ]:
        if col in display_df.columns:
            include_sign = col in pct_sign_cols
            display_df[col] = display_df[col].apply(
                lambda x, _sign=include_sign: fmt_pct(x, include_sign=_sign)
            )

    for col in [
        "Comp_Relevance_Score",
        "Quality_Score",
        "Amenity_Score",
        "Location_Score",
    ]:
        if col in display_df.columns:
            display_df[col] = pd.to_numeric(display_df[col], errors="coerce").round(2)

    if "Comp_Rank" in display_df.columns:
        display_df["Comp_Rank"] = pd.to_numeric(display_df["Comp_Rank"], errors="coerce")

    return display_df


def build_subscore_display(positioning: dict[str, Any]) -> pd.DataFrame:
    """
    Build enhanced positioning subscore display.
    """
    positioning = ensure_dict(positioning)
    if not positioning:
        return pd.DataFrame()

    subscore_data = []
    for key, value in positioning.items():
        if "_Subscore" not in key:
            continue
        subscore_name = key.replace("_Subscore", "").replace("_", " ")
        score = parse_float(value) if value is not None else 0.0
        weight_raw = positioning.get(f"{key}_Weight")
        weight = parse_float(weight_raw) if weight_raw is not None else 0.1
        subscore_data.append(
            {"Metric": subscore_name, "Score": score, "Weight": weight}
        )

    if not subscore_data:
        return pd.DataFrame()

    df = pd.DataFrame(subscore_data)
    df["Weighted_Score"] = (df["Score"] * df["Weight"]).round(1)
    df["Score"] = df["Score"].round(1)
    df["Weight"] = df["Weight"].map(lambda x: f"{x:.0%}")
    return df


def build_market_sentiment_display(sentiment: dict[str, Any]) -> pd.DataFrame:
    """Build market sentiment display."""
    sentiment = ensure_dict(sentiment)
    if not sentiment:
        return pd.DataFrame()

    rows = [
        {
            "Indicator": k.replace("_", " ").title(),
            "Value": v,
            "Sentiment": (
                "Positive" if v > 0.5 else ("Negative" if v < -0.5 else "Neutral")
            ),
        }
        for k, v in sentiment.items()
        if isinstance(v, (int, float))
    ]
    return pd.DataFrame(rows)


def build_forecast_indicators_display(forecast_indicators: dict[str, Any]) -> pd.DataFrame:
    """Build forecast indicators display."""
    forecast_indicators = ensure_dict(forecast_indicators)
    if not forecast_indicators:
        return pd.DataFrame()

    rows = [
        {
            "Indicator": k.replace("_", " ").title(),
            "Current": v.get("current", 0.0),
            "Forecast": v.get("forecast", 0.0),
            "Change": v.get("change", 0.0),
            "Confidence": v.get("confidence", 0.0),
        }
        for k, v in forecast_indicators.items()
        if isinstance(v, dict)
    ]
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["Current"] = df.apply(
        lambda row: format_metric_value(row["Indicator"], row["Current"]), axis=1
    )
    df["Forecast"] = df.apply(
        lambda row: format_metric_value(row["Indicator"], row["Forecast"]), axis=1
    )
    df["Change"] = df["Change"].apply(lambda x: fmt_pct(x, include_sign=True))
    df["Confidence"] = df["Confidence"].apply(fmt_pct)
    return df


# =============================================================================
# UI COMPONENTS
# =============================================================================

def render_market_header(property_name: str, property_id: str) -> None:
    """Render market analysis header with refresh and export actions."""
    col1, col2, col3 = st.columns([3, 1, 1])

    with col1:
        st.title("📍 Market Intelligence")
        st.caption(
            f"Advanced competitive analysis for **{property_name}** — "
            "Real-time market positioning, peer benchmarking, and strategic insights."
        )

    with col2:
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.cache_data.clear()
            market_cache.clear_property(property_id)
            st.rerun()

    with col3:
        if st.button("📊 Export Report", use_container_width=True):
            st.info("Export functionality coming soon")


def render_market_metrics(metrics: MarketMetrics) -> None:
    """Render market metrics in a two-row structured grid."""
    st.markdown("### 📊 Market Metrics Overview")

    cols = st.columns(5)
    primary = [
        ("Subject Rent", fmt_currency(metrics["subject_rent"]), "Current market rent for subject property"),
        ("Comp Avg Rent", fmt_currency(metrics["comp_avg_rent"]), "Average rent of competitive set"),
        ("Rent Premium", fmt_pct(metrics["rent_premium_pct"], include_sign=True), "Rent premium/discount vs competitive set"),
        ("Subject Occupancy", fmt_pct(metrics["subject_occupancy"]), "Current occupancy rate"),
        ("Occupancy Gap", fmt_pct(metrics["occupancy_gap"], include_sign=True), "Occupancy gap vs competitive set"),
    ]
    for col, (label, value, help_text) in zip(cols, primary):
        col.metric(label=label, value=value, help=help_text)

    cols2 = st.columns(5)
    rank_val = f"#{metrics['market_rank']}" if metrics["market_rank"] > 0 else "N/A"
    category, _, _ = get_positioning_category(metrics["positioning_score"])
    secondary = [
        ("Comp Avg Occupancy", fmt_pct(metrics["comp_avg_occupancy"]), "Average occupancy of competitive set"),
        ("Revenue Growth Gap", fmt_pct(metrics["rent_growth_gap"], include_sign=True), "Revenue growth gap vs peers"),
        ("Competitors", f"{metrics['comp_count']:,}", "Number of identified competitors"),
        ("Market Rank", rank_val, "Rank within market"),
        ("Positioning Score", f"{metrics['positioning_score']:.1f}", f"Competitive positioning score ({category})"),
    ]
    for col, (label, value, help_text) in zip(cols2, secondary):
        col.metric(label=label, value=value, help=help_text)


def render_positioning_card(positioning: dict[str, Any], metrics: MarketMetrics) -> None:
    """Render enhanced positioning card with inline CSS styling."""
    positioning = ensure_dict(positioning)
    if not positioning:
        return

    score = metrics["positioning_score"]
    category, icon, color = get_positioning_category(score)
    label = safe_str(positioning.get("Positioning_Label", "Competitive"), "Competitive")

    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, {color}20 0%, {color}10 100%);
            border-left: 8px solid {color};
            padding: 1.5rem;
            border-radius: 0.75rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        ">
            <div style="display:flex;justify-content:space-between;align-items:center;">
                <div>
                    <h2 style="margin:0;font-size:1.5rem;color:{color};">
                        {icon} {category} Positioning
                    </h2>
                    <p style="margin:0.25rem 0 0 0;color:#6c757d;">
                        Composite Score: <strong>{score:.1f}/100</strong> | {label}
                    </p>
                </div>
                <div style="
                    background:{color};
                    color:white;
                    padding:0.5rem 1rem;
                    border-radius:2rem;
                    font-size:1.25rem;
                    font-weight:bold;
                ">
                    {score:.0f}
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_insights_panels(narrative: str, flags: list[str], property_name: str) -> None:
    """Render narrative and watchlist-flags panels side by side."""
    col1, col2 = st.columns(2)

    with col1:
        with st.expander("📖 Market Narrative", expanded=True):
            if narrative:
                st.markdown(narrative)
            else:
                st.info(f"No market narrative available for {property_name}.")

    with col2:
        with st.expander("🚨 Watchlist Flags", expanded=True):
            if flags:
                for flag in flags:
                    st.markdown(f"⚠️ {safe_str(flag)}")
            else:
                st.success("✅ No material market watchlist flags identified.")


# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main() -> None:
    """Main application entry point."""
    st.set_page_config(
        page_title=CONFIG.page_title,
        page_icon=CONFIG.page_icon,
        layout="wide",
        initial_sidebar_state="expanded",
    )

    with st.spinner("🔄 Loading property options..."):
        property_options = get_property_options()

    if not property_options:
        st.error("❌ No property options are available.")
        st.info("Please check data connectivity and try again.")
        st.stop()

    selected_property_display: str = st.selectbox(
        "Select Property",
        options=property_options,
        index=0,
        help="Choose a property for market intelligence analysis",
    )  # type: ignore[assignment]

    if not selected_property_display:
        st.warning("Please select a property")
        st.stop()

    property_id = parse_property_selection(selected_property_display)
    if not property_id:
        st.error("Unable to parse the selected property.")
        st.stop()

    property_name = extract_property_name(selected_property_display, property_id)
    render_market_header(property_name, property_id)

    try:
        with st.spinner("🔍 Analysing market competition…"):
            bundle = load_market_bundle(property_id)

        render_warnings(bundle.get("warnings", []))
        metrics = extract_market_metrics(bundle)

        render_positioning_card(bundle.get("positioning", {}), metrics)
        render_market_metrics(metrics)
        st.markdown("---")

        render_insights_panels(
            bundle.get("narrative", ""),
            bundle.get("flags", []),
            property_name,
        )
        st.markdown("---")

        st.subheader("📊 Market Analysis Visualizations")
        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            rent_fig = build_rent_distribution_chart(
                bundle.get("rent_distribution", pd.DataFrame()),
                property_name,
                property_id=property_id,
                show_percentiles=True,
            )
            if rent_fig.data:
                st.plotly_chart(rent_fig, use_container_width=True)
            else:
                st.info("📭 No rent distribution data available.")

        with chart_col2:
            occ_fig = build_occupancy_chart(
                bundle.get("occupancy_comparison", pd.DataFrame()),
                property_name,
                property_id=property_id,
            )
            if occ_fig.data:
                st.plotly_chart(occ_fig, use_container_width=True)
            else:
                st.info("📭 No occupancy comparison data available.")

        st.markdown("#### 🎯 Competitive Positioning Analysis")
        radar_col1, radar_col2 = st.columns((1.2, 1.0))

        with radar_col1:
            radar_fig = build_positioning_radar_chart(bundle.get("positioning", {}))
            if radar_fig.data:
                st.plotly_chart(radar_fig, use_container_width=True)
            else:
                st.info("📭 No positioning radar chart available.")

        with radar_col2:
            subscore_display = build_subscore_display(bundle.get("positioning", {}))
            if not subscore_display.empty:
                st.dataframe(
                    subscore_display,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Weight": st.column_config.Column(
                            "Weight", help="Weight in composite score calculation"
                        ),
                        "Weighted_Score": st.column_config.Column(
                            "Weighted Score", help="Weight-adjusted score"
                        ),
                    },
                )
            else:
                st.info("📭 No subscore data available.")

        st.markdown("---")

        st.subheader("🔬 Advanced Market Analysis")
        trends_col, heat_col = st.columns((1.25, 1.0))

        with trends_col:
            st.markdown("##### 📈 Market Trends")
            trends_fig = build_market_trends_chart(bundle.get("market_trends", {}))
            if trends_fig.data:
                st.plotly_chart(trends_fig, use_container_width=True)
            else:
                st.info("📭 No market trends data available.")

        with heat_col:
            st.markdown("##### 🔥 Competitive Heatmap")
            heatmap_fig = build_market_heatmap_visualization(
                bundle.get("market_heatmap", {})
            )
            if heatmap_fig.data:
                st.plotly_chart(heatmap_fig, use_container_width=True)
            else:
                st.info("📭 No market heatmap available.")

        st.markdown("##### 🏆 Peer Benchmarking")
        peer_fig = build_peer_benchmarking_chart(
            bundle.get("peer_benchmarking", pd.DataFrame()),
            property_name,
            property_id=property_id,
        )
        if peer_fig.data:
            st.plotly_chart(peer_fig, use_container_width=True)
        else:
            st.info("📭 No peer benchmarking data available.")

        if CONFIG.enable_map_visualization:
            st.markdown("##### 🗺️ Geographic Competition Map")
            geographic_data = ensure_dict(bundle.get("geographic_data", {}))

            if not geographic_data:
                st.info("📭 No geographic competition data available.")
            elif not _FOLIUM_AVAILABLE:
                st.warning(
                    "📭 Map visualization is unavailable because the optional packages "
                    "`folium` and `streamlit-folium` are not installed in the current environment."
                )
                st.code("pip install folium streamlit-folium")
            else:
                market_map = build_market_map(geographic_data, property_name)
                if market_map is not None:
                    folium_static(market_map)
                else:
                    st.info(
                        "📭 Geographic data is present, but the map could not be rendered. "
                        "Check coordinate structure in `geographic_data`."
                    )

        st.markdown("---")

        st.subheader("📋 Competitive Set Detail")
        comp_display = build_comp_display(bundle.get("comp_table", pd.DataFrame()))
        if not comp_display.empty:
            st.dataframe(
                comp_display,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Comp_Rank": st.column_config.NumberColumn(
                        "Rank", format="%d", help="Rank within competitive set"
                    ),
                    "Comp_Relevance_Score": st.column_config.NumberColumn(
                        "Relevance Score",
                        format="%.2f",
                        help="Relative comp relevance score",
                    ),
                },
            )
        else:
            st.info("📭 No competitive set data available.")

        st.markdown("---")

        with st.expander("📊 Advanced Market Intelligence", expanded=False):
            tabs = st.tabs(
                ["📈 Market Forecast", "😊 Market Sentiment", "📊 Performance Metrics"]
            )

            with tabs[0]:
                if CONFIG.enable_market_forecasting and bundle.get("forecast_indicators"):
                    forecast_display = build_forecast_indicators_display(
                        bundle["forecast_indicators"]
                    )
                    if not forecast_display.empty:
                        st.dataframe(
                            forecast_display, use_container_width=True, hide_index=True
                        )
                    else:
                        st.info("📭 No forecast indicators available.")
                else:
                    st.info("Market forecasting is disabled or not available.")

            with tabs[1]:
                if CONFIG.enable_sentiment_analysis and bundle.get("market_sentiment"):
                    sentiment_display = build_market_sentiment_display(
                        bundle["market_sentiment"]
                    )
                    if not sentiment_display.empty:
                        st.dataframe(
                            sentiment_display,
                            use_container_width=True,
                            hide_index=True,
                        )
                    else:
                        st.info("📭 No market sentiment data available.")
                else:
                    st.info("Sentiment analysis is disabled or not available.")

            with tabs[2]:
                perf_stats = MarketPerformanceMonitor().get_stats()
                if perf_stats["timings"]:
                    timing_df = pd.DataFrame(
                        list(perf_stats["timings"].items()),
                        columns=["Analysis", "Time (s)"],
                    )
                    st.dataframe(
                        timing_df.sort_values("Time (s)", ascending=False),
                        use_container_width=True,
                        hide_index=True,
                    )

                cache_stats = market_cache.stats()
                st.metric(
                    "Cache Usage",
                    f"{cache_stats['size']}/{cache_stats['maxsize']}",
                )

                if st.button("Clear Market Cache", use_container_width=True):
                    market_cache.clear_property(property_id)
                    st.success("Market cache cleared")
                    st.rerun()

        st.markdown("---")
        st.subheader("🎯 Analyst Takeaway")

        with st.expander("📝 Detailed Analysis", expanded=True):
            category, icon, _ = get_positioning_category(metrics["positioning_score"])
            rank_display = (
                f"#{metrics['market_rank']}" if metrics["market_rank"] > 0 else "N/A"
            )
            st.write(
                f"""
**Market Position Summary for {property_name}**

The property demonstrates **{category.lower()}** competitive positioning
with a composite score of **{metrics['positioning_score']:.1f}/100**.

**Key Strengths:**
- **Rent Premium:** {fmt_pct(metrics['rent_premium_pct'], include_sign=True)} vs competitive set
- **Market Rank:** {rank_display}
- **Occupancy:** {fmt_pct(metrics['subject_occupancy'])} \
(Gap: {fmt_pct(metrics['occupancy_gap'], include_sign=True)})

**Areas for Attention:**
- **Revenue Growth Gap:** {fmt_pct(metrics['rent_growth_gap'], include_sign=True)} vs peers
- **Competitor Count:** {metrics['comp_count']:,} identified competitors

**Strategic Recommendations:**
1. **Monitor** competitive rent changes monthly
2. **Focus** on maintaining occupancy premium
3. **Investigate** revenue growth drivers
4. **Review** positioning quarterly
                """
            )

        last_updated: datetime = bundle.get("last_updated", datetime.now())
        st.markdown("---")
        st.caption(
            f"📅 Market analysis as of {last_updated.strftime('%Y-%m-%d %H:%M')} | "
            "🔄 Refresh for latest competitive data | "
            "⚠️ Market conditions may change rapidly"
        )

    except ValidationError as exc:
        logger.error("Market data validation error: %s", exc)
        st.error("⚠️ **Data Validation Error**: Please check market data consistency.")
        with st.expander("Technical Details"):
            st.code(str(exc))

    except RuntimeError as exc:
        logger.error("Runtime error: %s", exc)
        st.error(f"⚠️ **Runtime Error**: {exc}")
        st.info("Please try refreshing the page or contact support.")

    except Exception as exc:
        logger.error("Unexpected error: %s\n%s", exc, traceback.format_exc())
        st.error("⚠️ **Application Error**: An unexpected error occurred.")

        with st.expander("🔧 Debug Information", expanded=False):
            st.code(traceback.format_exc())

        st.info(
            "💡 **Troubleshooting Tips:**\n"
            "1. Refresh the page\n"
            "2. Try selecting a different property\n"
            "3. Clear browser cache\n"
            "4. Contact support if issue persists"
        )


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()