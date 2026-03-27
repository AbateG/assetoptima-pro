# AssetOptima Pro

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/AbateG/assetoptima-pro/main/app.py)

AssetOptima Pro is an interactive, analytics-driven platform designed to demonstrate professional multifamily real estate asset management workflows. Built with Streamlit, it integrates comprehensive financial modeling, performance analytics, market intelligence, and compliance tracking to provide a holistic view of portfolio management—from executive dashboards to property-level deep dives.

This platform serves as an educational and demonstration tool for real estate professionals, analysts, and developers interested in modern data-driven approaches to multifamily asset management. All data is entirely fictitious and anonymized, with no reference to real companies, properties, individuals, or portfolios.

## 🚀 Key Features

### Executive Portfolio Overview
- **Key Performance Indicators (KPIs)**: Portfolio NOI, weighted occupancy, DSCR, total units, assets in breach.
- **Trend Analysis**: Monthly NOI trends with variance overlays.
- **Market Allocation**: Unit and property distribution by market.
- **Value-Add Phase Insights**: Distribution across lease-up, stabilization, renovation phases.
- **Watchlist Management**: Priority assets requiring attention based on performance, valuation, and compliance signals.
- **Market Positioning**: Average competitive positioning scores across the portfolio.

### Asset Deep Dive
- **Property Selector**: Interactive dropdown with name and ID mapping.
- **Trailing 12-Month Summary**: Revenue, expense, and NOI variances with RAG (Red/Amber/Green) status.
- **Variance Analysis**: Breakdown by line items (e.g., repairs, payroll, marketing) with monthly trends.
- **Financial Trends**: Revenue, expense, and NOI over time; stacked expense breakdowns.
- **Occupancy and Rent Trends**: Dual-axis charts showing operational metrics.
- **Underwriting Comparison**: Actual performance vs. original acquisition assumptions.

### Forecast & Valuation
- **Interactive Financial Modeling**: Slider-based assumption testing (rent growth, vacancy, expense growth, CapEx, exit cap rate, hold period).
- **5-Year Cash Flow Projections**: Revenue, expenses, NOI, CapEx reserves, debt service, DSCR, and levered/unlevered cash flows.
- **Hold/Sell Recommendations**: Rule-based analysis considering IRR, equity multiple, DSCR, and market conditions.
- **Valuation Reconciliation**: Weighted combination of direct capitalization, discounted cash flow (DCF), and sales comparison approaches.
- **Scenario Analysis**: Bear/base/bull valuations with sensitivity heatmaps for IRR.
- **Value Bridge**: Decomposition of value creation from acquisition to current indicated value.

### Market Intelligence
- **Competitive Positioning**: Subject property vs. comp set analysis (rent premiums, occupancy gaps, growth differentials).
- **Comp Tables**: Detailed comparable property data with relevance scores.
- **Rent and Occupancy Distributions**: Visual comparisons across subject and comps.
- **Market Narratives**: AI-generated summaries of positioning and risks.
- **Watchlist Flags**: Automated alerts for market-related concerns.

### Compliance & Reporting
- **Debt Covenant Monitoring**: DSCR and LTV headroom, breach detection, reporting urgency.
- **Business Plan Tracking**: Initiative progress, budget variances, NOI lift expectations.
- **Portfolio Compliance KPIs**: Breach counts, refinance candidates, reporting timelines.

### Advanced Analytics
- **Forecasting Engine**: Compound growth projections with separate revenue/expense assumptions.
- **Valuation Models**: Multi-approach reconciliation with scenario flexing.
- **IRR Sensitivity**: Heatmaps for exit cap rate and rent growth variations.
- **Recommendation Logic**: Deterministic rules for hold/sell/refinance decisions based on financial metrics.

## 🏗️ Architecture

The platform is modular and built for extensibility:

- **Modules**: Core logic separated into functional areas (e.g., `data_loader.py` for data ingestion, `forecasting.py` for projections, `valuation.py` for value estimation).
- **Pages**: Streamlit UI components for user interaction, pulling from cached module outputs.
- **Data Layer**: CSV-based datasets normalized to canonical schemas with validation and alias handling.
- **Caching**: Streamlit's `@st.cache_data` for performance optimization across sessions.
- **Dependencies**: Python libraries for data manipulation (pandas, numpy), visualization (plotly), and UI (streamlit).

### Project Structure
```
assetoptima-pro/
├── app.py                          # Main Streamlit entry point
├── remove_docstrings.py            # Utility for code optimization
├── modules/                        # Core analytics modules
│   ├── data_loader.py              # Data ingestion and normalization
│   ├── kpi_calculations.py         # Financial formulas and metrics
│   ├── forecasting.py              # Cash flow projections
│   ├── valuation.py                # Valuation methodologies
│   ├── variance_analysis.py        # Performance variance tracking
│   ├── market_analysis.py          # Competitive positioning
│   ├── recommendation_engine.py    # Action recommendations
│   ├── business_plan_tracker.py    # Initiative monitoring
│   ├── debt_compliance.py          # Covenant oversight
│   └── report_generator.py         # Output formatting
├── pages/                          # Streamlit page components
│   ├── 1_Portfolio_Overview.py     # Executive dashboard
│   ├── 2_Asset_Deep_Dive.py        # Property workbench
│   ├── 3_Forecast_and_Valuation.py # Modeling interface
│   ├── 4_Market_Intelligence.py    # Comp analysis
│   ├── 5_Business_Plan_Tracker.py  # Initiative tracking
│   └── 6_Compliance_and_Reporting.py # Regulatory monitoring
├── data/                           # Fictional datasets (not included)
│   ├── properties.csv
│   ├── monthly_performance.csv
│   ├── market_comps.csv
│   ├── business_plan.csv
│   ├── debt_covenants.csv
│   └── underwriting_assumptions.csv
├── venv/                           # Virtual environment (ignored)
├── .kilo/                          # Agent configurations
└── README.md, LICENSE.md, etc.     # Documentation
```

## 📊 Methodology

### Financial Modeling
- **NOI Calculation**: Effective Gross Income minus operating expenses.
- **IRR**: Newton-Raphson solved for cash flow series.
- **Valuation Weights**: 50% direct cap, 30% sales comp, 20% DCF.
- **Variance Thresholds**: Revenue/NOI: Warn at -3%, critical at -8%; Expenses: Warn at 3%, critical at 8%.

### Data Assumptions
- All metrics annualized where appropriate.
- Occupancy and rates as decimals (e.g., 0.95 = 95%).
- Growth rates compounded annually.
- Cap rates and IRRs as percentages converted to decimals internally.

## 🛠️ Installation and Setup

### Prerequisites
- Python 3.8+
- pip for package management

### Steps
1. **Clone the Repository**:
   ```bash
    git clone https://github.com/AbateG/assetoptima-pro.git
   cd assetoptima-pro
   ```

2. **Set Up Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   (Note: `requirements.txt` should include streamlit, pandas, numpy, plotly, etc.)

4. **Prepare Data**:
   - Create `data/` directory.
   - Add fictional CSV files matching the canonical schemas in `modules/data_loader.py`.

5. **Run the App**:
   ```bash
   streamlit run app.py
   ```
   Open the provided URL in your browser.

### Self-Tests
Run module diagnostics:
```bash
python modules/data_loader.py
python modules/valuation.py
```

## 📈 Usage

1. **Portfolio Overview**: Start here for high-level KPIs and trends.
2. **Asset Deep Dive**: Select a property to analyze performance and variances.
3. **Forecast & Valuation**: Adjust assumptions and view projections/recommendations.
4. Navigate via sidebar or page tabs for specialized insights.

All outputs are interactive and responsive to user inputs.

## 🤝 Contributing

This is a demonstration project. For suggestions or extensions:
- Open an issue with detailed feedback.
- Submit pull requests for enhancements (e.g., additional metrics or visualizations).
- Ensure new code includes docstrings and passes self-tests.

## 📜 License

See [LICENSE.md](LICENSE.md) for details.

## 🙏 Acknowledgments

- Inspired by institutional real estate platforms.
- Built for educational purposes in data science and finance.
- Fictional data generated for realism without real-world attribution.

## 👤 Author

Developed by Daniel Abate Garay.

## 🔗 Sharing and Deployment

This project is designed for public sharing on platforms like GitHub, Streamlit Community Cloud, and LinkedIn to demonstrate advanced data science and financial modeling skills.

