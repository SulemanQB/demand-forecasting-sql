# Backtest metrics

At the 28-day order quantity (400 SKU-origins across 10 origins), LightGBM Tweedie WAPE was 0.355129 (bias -0.063205), 10.32% lower than the 28-day moving average (WAPE 0.395996); an origin-cluster 95% interval for the WAPE gap (Tweedie minus the moving average) is [-0.090274, +0.009449] and includes zero; Tweedie was lower on 6 of 10 origins.

The largest SKU-origin share of the net absolute-error reduction versus the moving average was SKU 22197 at origin 2011-06-14 (20.75% of the net; actual 1626, Tweedie 1828.21, moving average 6175.00).

LightGBM L1, fit on absolute error, had order-quantity WAPE 0.522505 (bias -0.485450).

Dropping SKU-origins with no sale in the 90 days before the origin (dates after origin minus 90 days, through the origin), without refitting, leaves 391 of 400 SKU-origins. On that slice, order-quantity WAPE was 0.353240 for Tweedie (bias -0.065094) and 0.395996 for the moving average, 10.80% lower for Tweedie.

Horizon cuts are partial sums of the fitted 28-day forecasts, not refits. days 1-7: Tweedie WAPE 0.571241 is 0.70% higher than the moving average (WAPE 0.567251); days 1-14: Tweedie WAPE 0.437721 is 5.23% lower than the moving average (WAPE 0.461885).

In Q1, the lowest backtest-volume quartile (descriptive, not a feature; 49108 units, mean order 491.08), order-quantity WAPE was 0.561226 for Tweedie (bias +0.332390), 0.553820 for the moving average (bias +0.024864), and 0.433232 for L1 (bias -0.273265). Tweedie was above the 28-day actual on 77 of 100 SKU-origins, against 42 of 100 for the moving average.

SKU 84270 (ASSTD DESIGN BUBBLE GUM RING) has actual 0 on all 10 scored origins. The last positive day in the panel is 2010-12-06, and it sold 20866 units before the first origin. The moving average forecasts 0.00 units in total; Tweedie forecasts 1240.10. Its share of the net absolute-error reduction versus the moving average is -5.92%. A negative share increases Tweedie's absolute error.

LightGBM refit on the SKUs with the most units before the first origin. Not a slice of the top-40 model. On this 20-SKU refit, order-quantity WAPE was 0.373500 for Tweedie (bias -0.104242) and 0.396441 for the moving average (5.79% lower for Tweedie), across 200 SKU-origins. Same SKUs scored from the top-40 model. The model was not refit. That slice has Tweedie WAPE 0.357125 and moving-average WAPE 0.396441 (9.92% lower for Tweedie). The refit assortment includes 84270, which has no sales in the scored window.

Order-quantity bias, (forecast minus actual) / actual, was weekly seasonal naive +0.054680; yearly seasonal naive +0.342904; 28-day moving average -0.050239; LightGBM L1 -0.485450; LightGBM Tweedie -0.063205.

Summing the assortment before scoring, daily aggregate WAPE was 0.568517 for the weekly seasonal naive; 0.725395 for the yearly seasonal naive; 0.588800 for the 28-day moving average; 0.537963 for the LightGBM L1; 0.393766 for the LightGBM Tweedie.

On individual SKU-days, Tweedie WAPE was 0.933069 (bias -0.063205) and L1 WAPE was 0.824654 (bias -0.485450), against 1.053000 for the 28-day moving average and 1.276659 for the weekly seasonal naive. The daily score is a shape score; the order quantity above is the replenishment comparison.

On the weekly track (ISO weeks, horizon 4, 10 origins, 400 SKU-origins), the LightGBM Tweedie has the lowest summed-horizon WAPE. 1-week seasonal naive WAPE 0.612399 (bias -0.033896); 52-week seasonal naive WAPE 0.639445 (bias +0.341604); 4-week moving average WAPE 0.422572 (bias -0.042427); LightGBM Tweedie WAPE 0.399551 (bias -0.003710).

Pooled daily MAE was 42.6956 for Tweedie, 37.7347 for L1, 48.1835 for the 28-day moving average, 58.4178 for the weekly seasonal naive, and 66.7329 for the yearly seasonal naive. On the same rows, the relative MAE gap equals the relative WAPE gap because both scale with total absolute error.

Pooled daily RMSE was 125.0627 for Tweedie, 129.6483 for L1, 128.8235 for the 28-day moving average, 193.7641 for the weekly seasonal naive, and 228.8161 for the yearly seasonal naive.

Pooled daily sMAPE was 1.190162 for Tweedie, 1.177392 for L1, 1.141511 for the 28-day moving average, 0.964985 for the weekly seasonal naive, and 1.002279 for the yearly seasonal naive.

WAPE, bias, and sMAPE in this table are ratios (sMAPE uses the 0/0 = 0 convention and is not multiplied by 100). Bias is (sum of forecast minus actual) / sum of actual. Negative bias is an under-forecast. The CSV stores the same figures at full precision. Displayed rates are rounded to 6 decimal places and MAE/RMSE to 4.

| fold | origin | model | points | MAE | RMSE | WAPE | sMAPE | bias |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 2011-02-22 | weekly seasonal naive | 1120 | 46.6955 | 107.9231 | 1.112674 | 1.020630 | -0.237921 |
| 1 | 2011-02-22 | yearly seasonal naive | 1120 | 75.5054 | 459.1680 | 1.799162 | 1.035228 | +0.643576 |
| 1 | 2011-02-22 | 28-day moving average | 1120 | 41.7151 | 93.5444 | 0.993997 | 1.109306 | -0.163287 |
| 1 | 2011-02-22 | LightGBM L1 | 1120 | 33.9924 | 94.8507 | 0.809981 | 1.148510 | -0.481322 |
| 1 | 2011-02-22 | LightGBM Tweedie | 1120 | 39.7636 | 91.3094 | 0.947497 | 1.170900 | -0.007893 |
| 2 | 2011-03-22 | weekly seasonal naive | 1120 | 69.7839 | 179.4763 | 1.474679 | 0.976743 | +0.257358 |
| 2 | 2011-03-22 | yearly seasonal naive | 1120 | 60.2580 | 148.4530 | 1.273377 | 1.069243 | -0.067302 |
| 2 | 2011-03-22 | 28-day moving average | 1120 | 48.4682 | 119.5841 | 1.024233 | 1.147888 | -0.113151 |
| 2 | 2011-03-22 | LightGBM L1 | 1120 | 39.0866 | 123.5263 | 0.825981 | 1.209583 | -0.578565 |
| 2 | 2011-03-22 | LightGBM Tweedie | 1120 | 42.5525 | 118.7066 | 0.899222 | 1.177843 | -0.159967 |
| 3 | 2011-04-19 | weekly seasonal naive | 1120 | 57.6152 | 151.1001 | 1.237254 | 0.987451 | +0.048797 |
| 3 | 2011-04-19 | yearly seasonal naive | 1120 | 74.0259 | 254.2870 | 1.589665 | 1.040734 | +0.419308 |
| 3 | 2011-04-19 | 28-day moving average | 1120 | 52.1084 | 109.7637 | 1.119000 | 1.236591 | +0.016202 |
| 3 | 2011-04-19 | LightGBM L1 | 1120 | 42.1391 | 111.8185 | 0.904915 | 1.301964 | -0.432015 |
| 3 | 2011-04-19 | LightGBM Tweedie | 1120 | 46.2082 | 107.1243 | 0.992297 | 1.285748 | -0.043751 |
| 4 | 2011-05-17 | weekly seasonal naive | 1120 | 54.1884 | 176.5303 | 1.134645 | 0.974091 | -0.054908 |
| 4 | 2011-05-17 | yearly seasonal naive | 1120 | 65.8429 | 204.0205 | 1.378676 | 0.951026 | +0.168035 |
| 4 | 2011-05-17 | 28-day moving average | 1120 | 52.9281 | 172.2273 | 1.108256 | 1.166290 | -0.024940 |
| 4 | 2011-05-17 | LightGBM L1 | 1120 | 40.8287 | 172.6209 | 0.854907 | 1.193052 | -0.476186 |
| 4 | 2011-05-17 | LightGBM Tweedie | 1120 | 45.7763 | 169.2486 | 0.958506 | 1.208849 | -0.071065 |
| 5 | 2011-06-14 | weekly seasonal naive | 1120 | 43.1625 | 117.6738 | 1.136015 | 0.916950 | -0.032570 |
| 5 | 2011-06-14 | yearly seasonal naive | 1120 | 59.7866 | 132.2976 | 1.573554 | 1.039380 | +0.418833 |
| 5 | 2011-06-14 | 28-day moving average | 1120 | 48.7135 | 105.1736 | 1.282114 | 1.211475 | +0.256968 |
| 5 | 2011-06-14 | LightGBM L1 | 1120 | 32.4649 | 99.6459 | 0.854459 | 1.194102 | -0.353337 |
| 5 | 2011-06-14 | LightGBM Tweedie | 1120 | 40.8668 | 97.5675 | 1.075593 | 1.261716 | +0.166100 |
| 6 | 2011-07-12 | weekly seasonal naive | 1120 | 55.6045 | 176.9075 | 1.271089 | 0.986702 | -0.032228 |
| 6 | 2011-07-12 | yearly seasonal naive | 1120 | 59.5571 | 169.2438 | 1.361445 | 0.997686 | +0.260557 |
| 6 | 2011-07-12 | 28-day moving average | 1120 | 43.6327 | 136.1865 | 0.997420 | 1.108052 | -0.131462 |
| 6 | 2011-07-12 | LightGBM L1 | 1120 | 35.4945 | 136.2105 | 0.811385 | 1.162242 | -0.522335 |
| 6 | 2011-07-12 | LightGBM Tweedie | 1120 | 38.7158 | 133.9644 | 0.885023 | 1.157480 | -0.172919 |
| 7 | 2011-08-09 | weekly seasonal naive | 1120 | 70.6411 | 245.3567 | 1.701609 | 0.978760 | +0.588438 |
| 7 | 2011-08-09 | yearly seasonal naive | 1120 | 64.8616 | 147.4417 | 1.562392 | 0.968130 | +0.497355 |
| 7 | 2011-08-09 | 28-day moving average | 1120 | 46.9762 | 101.1813 | 1.131567 | 1.188492 | +0.053747 |
| 7 | 2011-08-09 | LightGBM L1 | 1120 | 34.2341 | 101.2466 | 0.824635 | 1.219604 | -0.502695 |
| 7 | 2011-08-09 | LightGBM Tweedie | 1120 | 39.8961 | 97.0665 | 0.961022 | 1.237186 | -0.056223 |
| 8 | 2011-09-06 | weekly seasonal naive | 1120 | 50.6339 | 121.3676 | 1.064817 | 0.934864 | -0.134252 |
| 8 | 2011-09-06 | yearly seasonal naive | 1120 | 62.4652 | 140.3893 | 1.313624 | 0.987996 | +0.188535 |
| 8 | 2011-09-06 | 28-day moving average | 1120 | 45.3452 | 101.8651 | 0.953597 | 1.090681 | -0.126967 |
| 8 | 2011-09-06 | LightGBM L1 | 1120 | 37.4438 | 102.2162 | 0.787432 | 1.138624 | -0.464098 |
| 8 | 2011-09-06 | LightGBM Tweedie | 1120 | 40.0702 | 97.9390 | 0.842665 | 1.118663 | -0.136412 |
| 9 | 2011-10-04 | weekly seasonal naive | 1120 | 56.5946 | 207.4351 | 1.255964 | 0.921487 | +0.102402 |
| 9 | 2011-10-04 | yearly seasonal naive | 1120 | 70.8768 | 222.1488 | 1.572917 | 1.038569 | +0.472775 |
| 9 | 2011-10-04 | 28-day moving average | 1120 | 50.0471 | 174.7682 | 1.110659 | 1.135687 | +0.055283 |
| 9 | 2011-10-04 | LightGBM L1 | 1120 | 37.7398 | 176.3275 | 0.837533 | 1.142465 | -0.443231 |
| 9 | 2011-10-04 | LightGBM Tweedie | 1120 | 43.8581 | 172.4931 | 0.973310 | 1.193693 | -0.035731 |
| 10 | 2011-11-01 | weekly seasonal naive | 1120 | 79.2580 | 338.3625 | 1.364103 | 0.952177 | +0.054476 |
| 10 | 2011-11-01 | yearly seasonal naive | 1120 | 74.1491 | 217.8618 | 1.276174 | 0.894797 | +0.469935 |
| 10 | 2011-11-01 | 28-day moving average | 1120 | 51.9006 | 142.9247 | 0.893256 | 1.020654 | -0.224464 |
| 10 | 2011-11-01 | LightGBM L1 | 1120 | 43.9236 | 145.7983 | 0.755965 | 1.063776 | -0.559552 |
| 10 | 2011-11-01 | LightGBM Tweedie | 1120 | 49.2489 | 132.7541 | 0.847618 | 1.089545 | -0.067209 |
| pooled |  | weekly seasonal naive | 11200 | 58.4178 | 193.7641 | 1.276659 | 0.964985 | +0.054680 |
| pooled |  | yearly seasonal naive | 11200 | 66.7329 | 228.8161 | 1.458377 | 1.002279 | +0.342904 |
| pooled |  | 28-day moving average | 11200 | 48.1835 | 128.8235 | 1.053000 | 1.141511 | -0.050239 |
| pooled |  | LightGBM L1 | 11200 | 37.7347 | 129.6483 | 0.824654 | 1.177392 | -0.485450 |
| pooled |  | LightGBM Tweedie | 11200 | 42.6956 | 125.0627 | 0.933069 | 1.190162 | -0.063205 |
