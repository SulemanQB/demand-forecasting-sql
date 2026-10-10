# Retail demand forecasting with SQL features

28-day SKU demand forecasts for a UK online gift retailer. DuckDB builds the model features in SQL, and a global LightGBM model is scored against seasonal-naive baselines on a rolling-origin backtest.

**Headline.** On a 28-day horizon across 10 rolling origins, LightGBM pooled WAPE was 0.833938: 37.27% lower than the weekly seasonal naive (WAPE 1.329340); 42.29% lower than the yearly seasonal naive (WAPE 1.445168); 22.68% lower than the 28-day moving average (WAPE 1.078537).

LightGBM WAPE was lower than all three baselines on each of the 10 origins. Pooled MAE was 42.7183 versus 68.0952 for the weekly seasonal naive. Full precision is in `reports/metrics.csv`. The sentences above are copied from `reports/headline.txt`.

## Why this matters

A buyer setting a four-week replenishment order needs a unit forecast for each core SKU. Too high, and cash sits in stock. Too low, and the order is missed, especially on the run-up to Christmas. The decision metric here is WAPE, absolute error divided by actual units, so a miss on a high-volume day counts more than a miss on a quiet day.

## Data

[UCI Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii), donated by Daqing Chen. Invoices from a UK online retailer, 1 December 2009 to 9 December 2011. No account is required. `make download` fetches the zip, checks SHA-256 `572e36277c2390fbfde10664750731e0a86f55e33470d91919085f0408e67bfb`, and writes `data/raw/online_retail_ii.parquet`. That file is gitignored.

This run kept 1,044,848 invoice rows after dropping the duplicated 1–9 December 2010 overlap between the two yearly sheets. Cleaning left 930,509 UK product lines. Rules:

- United Kingdom only.
- Stock codes that start with a digit. Postage, bank charges, adjustments, and gift vouchers are excluded.
- Price and quantity must be positive. A sale is removed when the same customer returns the same quantity of that SKU within 21 days.
- Among 936 SKUs with at least 180 distinct sale days, the top 40 by units. They account for 1,585,714 of 8,919,810 cleaned units (17.78%).
- One row per SKU per day from 2009-12-01 to 2011-12-09. Days with no sale are zero units.

Examples from that assortment: 85123A cream hanging heart T-light holder (83,889 units, 603 sale days), 85099B jumbo bag red retrospot (86,792 units, 585 sale days), 84077 world war 2 gliders (96,970 units, 447 sale days). The full list is in `reports/build_stats.json`.

## Architecture

```
download script
  -> data/raw/online_retail_ii.parquet
  -> DuckDB raw_invoices
  -> sql/01_clean.sql
       clean lines, daily sales, SKU filter, complete daily panel
  -> sql/02_features.sql
       as-of lags, rolling mean/std, trailing price, direct 1..28 day rows
  -> baselines and LightGBM
  -> rolling-origin backtest
  -> reports/metrics.csv, reports/metrics.md, plots
```

Each supervised row predicts a target date from an origin `horizon` days earlier. Lags, rolling means, and rolling standard deviations are SQL window functions that close on that origin. The weekly seasonal naive (season 7) and the same weekday last year (364 days, so the weekday lines up) are joins to dates on or before the origin. Target-day calendar fields, including days until Christmas, are known when the forecast is made. The price feature is the last realized unit price as of the origin divided by its trailing 28-day mean.

LightGBM is refit at every origin: one global model, all 40 SKUs, horizons 1–28 together, MAE objective (`regression_l1`), SKU id categorical. Fixed settings: learning rate 0.05, 63 leaves, minimum 40 rows per leaf, feature fraction 0.9, bagging fraction 0.8, L2 1.0, at most 400 rounds, early stopping after 40 rounds on the 28 days before the origin, seed 42. Predictions are clipped at zero. These settings are in `src/demandforecast/config.py` and were not searched on the backtest.

The first origin is 2011-02-22. By then the training cut already contains 56 daily as-of dates with the 364-day lag defined, and the 28-day horizon still falls inside the sample. Later origins step forward 28 days through 2011-11-01 (10 folds, 11,200 SKU-days).

On this run the backtest stage took 24.9 seconds. Mean gain was highest for the weekly seasonal-naive feature, then target day of week, the 28-day rolling mean, SKU id, and days to Christmas (`reports/feature_importance.csv`).

## Reproduce

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
make test
make all
```

`make all` downloads, builds the feature tables, and writes `reports/`. A cached parquet is reused. `python -m demandforecast download --force` fetches the file again.

Commands: `make download`, `make build-features`, `make backtest`, `make test`.

Continuous integration runs the unit tests, including SQL feature values and the as-of rule on a toy table. It does not download the UCI file. When `reports/predictions.csv` is present, the tests recompute the metrics from those predictions and check them against `reports/metrics.csv`.

## Results

WAPE = sum(|forecast − actual|) / sum(|actual|). sMAPE is the mean of 2|forecast − actual| / (|actual| + |forecast|), with 0/0 counted as 0. Rates below are ratios, not percentages. The display rounds rates to 6 decimal places and MAE/RMSE to 4; `reports/metrics.csv` has full precision.

Pooled rows:

| fold | origin | model | points | MAE | RMSE | WAPE | sMAPE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| pooled |  | weekly seasonal naive | 11200 | 68.0952 | 212.5820 | 1.329340 | 0.969943 |
| pooled |  | yearly seasonal naive | 11200 | 74.0285 | 287.4805 | 1.445168 | 0.961749 |
| pooled |  | 28-day moving average | 11200 | 55.2479 | 142.6519 | 1.078537 | 1.232403 |
| pooled |  | LightGBM | 11200 | 42.7183 | 143.6927 | 0.833938 | 1.227377 |

Pooled RMSE was 143.6927 for LightGBM, 212.5820 for the weekly seasonal naive, 287.4805 for the yearly seasonal naive, and 142.6519 for the 28-day moving average. Pooled sMAPE was 1.227377 for LightGBM and 0.969943 for the weekly seasonal naive (0.961749 for the yearly seasonal naive, 1.232403 for the 28-day mean). The per-fold table is in `reports/metrics.md`.

The chart sums units across the 40 SKUs in the last window. The table scores each SKU-day.

![Last forecast window, sum of 40 SKUs](reports/forecast_vs_actual.png)

![WAPE by fold](reports/wape_by_fold.png)

## Limitations

- Recorded sales are not demand. Stockouts are not flagged, so some zeros are censored.
- If a customer buys the same quantity twice and returns it once, both sales can be removed. Anonymous returns stay unmatched.
- 31 of the 40 SKUs already sell on 2009-12-01. The other nine start later, and the daily spine still fills the earlier days with zeros.
- There is no promotion calendar. WAPE above 1 on the naive forecasts comes from bulk order days: a weekly copy often places that spike on the wrong date, and absolute error then exceeds volume.
- The long tail, and any SKU with fewer than 180 sale days, is out of scope. The extract ends on 9 December 2011. On this 28-day grid the last scored day is 29 November 2011, before the final Christmas week.
- Two years is a short sample for a yearly lag. The model is a single fixed configuration.
- sMAPE, which treats a quiet day like a busy one, favors the seasonal naive forecasts on this sample. RMSE favors the 28-day mean by a small margin (142.6519 versus 143.6927).

## Next steps

- Pinball loss or an explicit stockout cost, so the point forecast can feed a safety-stock rule.
- Reconcile SKU forecasts to a category total.
- Use a forward price and promo calendar when one exists.
- A separate path for SKUs that do not yet have 180 sale days.

## Author

Suleman Qamar, Sydney.

## License

MIT. The UCI extract is downloaded at runtime and is not in this repository.
