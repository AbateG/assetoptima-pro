# Architecture Overview

AssetOptima Pro is structured for modularity, testability, and performance.

## Core Principles
- **Separation of Concerns**: Modules handle logic; pages handle UI.
- **Data Contracts**: Canonical schemas ensure consistency.
- **Caching**: Streamlit caching reduces recomputation.
- **Error Resilience**: Safe fallbacks for missing data.

## Module Dependencies
- `data_loader` → All modules (provides normalized data).
- `kpi_calculations` → `forecasting`, `valuation`.
- `forecasting` → `valuation`, `recommendation_engine`.
- Pages depend on modules but not vice versa.

## Performance Optimizations
- Cached data loading.
- Vectorized pandas operations.
- Lazy evaluation in charts.