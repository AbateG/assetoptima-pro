# Data Schema

## Required Columns by Dataset
- `properties`: Property_ID, Property_Name, Market, Units, etc.
- `monthly_performance`: Property_ID, Year_Month, Actual_Revenue, etc.
- Details in `modules/data_loader.py`.

## Assumptions
- Dates in YYYY-MM-DD.
- Rates as decimals or percentages (normalized internally).
- All values numeric except IDs/names.