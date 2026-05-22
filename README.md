# Nifty Data Layer & Hypothesis Validation

This package fetches real NSE data and tests the **specific hypothesis derived
from the literature review**: that there exists a **Variance Risk Premium (VRP)**
in Nifty options, with the bulk of the premium captured by holding
**short-option positions overnight** rather than purely intraday.

## Why this hypothesis (not opening range breakout)

After a literature review, the strongest empirical evidence in Nifty options
points to:

1. **Implied vol > realized vol on average** (positive VRP)
2. **Day–night asymmetry**: overnight short-option returns positive,
   intraday short-option returns negative or weakly positive
3. ORB on Indian indices loses money after realistic costs

So we're testing the effect with the strongest prior, not the trendiest one.

## What this package does NOT do

- It does not claim demonstrated edge before validation runs.
- It does not bypass broker authentication. You must have your own API key.
- It does not execute trades. Paper trading and live execution are handled separately.

## Modules

- `brokers/`       — vendor-specific data fetchers (Kite, Upstox, AngelOne)
- `data_store.py`  — local Parquet cache so we don't re-download
- `hypothesis_h1_vrp.py`     — does IV > RV on a rolling basis? (Step 2a)
- `hypothesis_h2_daynight.py` — is overnight short-option return > intraday? (Step 2b)
- `hypothesis_h3_costs.py`   — does the effect survive realistic costs?
- `validation_report.py`     — produces the report the strategy depends on

## Workflow

```
1. pip install -r requirements.txt
2. Copy config.example.yaml -> config.yaml, add your broker API credentials
3. python -m nifty_data_layer.fetch --start 2022-01-01 --end 2024-12-31
4. python -m nifty_data_layer.hypothesis_h1_vrp
5. python -m nifty_data_layer.hypothesis_h2_daynight
6. python -m nifty_data_layer.hypothesis_h3_costs
7. python -m nifty_data_layer.validation_report
```

Only after all three hypothesis tests pass should the strategy code be run on real data.
