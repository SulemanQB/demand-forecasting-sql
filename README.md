# Retail demand forecasting with SQL features

28-day SKU demand forecasts for a UK online gift retailer. DuckDB builds the model features in SQL, and a global LightGBM model is scored against seasonal-naive baselines on a rolling-origin backtest.

## Why this matters

A buyer setting a four-week replenishment order needs a unit forecast for each core SKU. Too high, and cash sits in stock. Too low, and the order is missed, especially on the run-up to Christmas. The number that matters here is volume-weighted error (WAPE) at SKU-day level, against baselines a planning team already uses.

## Data

[UCI Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii), donated by Daqing Chen. The file is a UK online retailer's invoices from 1 December 2009 to 9 December 2011. No account is required. `make download` fetches the zip, checks its SHA-256, and writes a parquet file under `data/`, which is gitignored.

The modelled subset:

- United Kingdom invoices only.
- Stock codes that start with a digit. Postage, bank charges, adjustments, and gift vouchers are left out.
- Price and quantity must be positive. A sale is removed when the same customer returns the same quantity of that SKU within 21 days. Anonymous returns are not matched.
- The two yearly sheets both contain 1–9 December 2010. Those days are kept from the later sheet only.
- Among SKUs with at least 180 distinct sale days, the top 40 by units.
- A daily spine from the first to the last cleaned invoice date. Days with no sale are zero units.

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

Each supervised row predicts a target date from an origin `horizon` days earlier. Lags, rolling means, and rolling standard deviations are SQL window functions that close on that origin. The weekly seasonal naive (season 7) and the same weekday last year (364 days, so the weekday lines up) are joins to dates on or before the origin. Target-day calendar fields, including days until Christmas, are known when the forecast is made. The price feature is the last realized unit price as of the origin divided by its trailing 28-day mean. The price paid on the target day is not used.

LightGBM is refit at every origin: one global model, all selected SKUs, horizons 1–28, MAE objective, SKU id as a categorical. The 28 days immediately before the origin are the early-stopping validation set. Predictions are clipped at zero. The configuration is fixed in `src/demandforecast/config.py`. It is not a search over this backtest.

Origins start only once the training cut contains 56 daily as-of dates with the 364-day lag defined, and each origin has a full 28-day horizon inside the sample. Folds step forward 28 days.

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

Continuous integration runs the unit tests, including the SQL feature checks on a toy table. It does not download the UCI file.

## Results

The backtest writes `reports/metrics.csv`, `reports/metrics.md`, `reports/headline.txt`, and the plots. The headline in this section is that file, copied after the run.

## Limitations

- Sales are not demand. Stockouts are not flagged, so some zeros are censored.
- If a customer buys the same quantity twice and returns it once, both sales can be removed.
- Returns with no customer id are not matched.
- The daily spine starts at the first UK invoice in the extract, so a SKU can show zeros before it was actually listed.
- There is no promotion calendar.
- The long tail and items that sell on fewer than 180 days are out of scope. The file ends on 9 December 2011, so the backtest does not score the last Christmas week of 2011.
- Two years is a short sample for a yearly lag.
- Hyperparameters are a single fixed setting.

## Next steps

- Pinball loss or stockout cost, so the point forecast can feed a safety-stock rule.
- Reconcile SKU forecasts to a category total.
- Use a forward price and promo calendar when one exists.
- A separate path for SKUs that do not yet have 180 sale days.

## Author

Suleman Qamar, Sydney.

## License

MIT. The UCI extract is downloaded at runtime and is not in this repository.
