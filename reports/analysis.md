# Evaluation note

This note is written from the CSV files in `reports/`. The replenishment comparison is the 28-day total for one SKU at one origin. Daily WAPE is reported beside it as a shape score. Volume quartiles are computed from backtest actuals and are not model features. Demand class uses only history through the origin, starting at the first positive day.

At the 28-day order quantity (400 SKU-origins across 10 origins), LightGBM Tweedie WAPE was 0.355129 (bias -0.063205), 10.32% lower than the 28-day moving average (WAPE 0.395996); an origin-cluster 95% interval for the WAPE gap (Tweedie minus the moving average) is [-0.090274, +0.009449] and includes zero; Tweedie was lower on 6 of 10 origins.

## Order-quantity uncertainty and labeled cuts

The largest SKU-origin share of the net absolute-error reduction versus the moving average was SKU 22197 at origin 2011-06-14 (20.75% of the net; actual 1626, Tweedie 1828.21, moving average 6175.00).

LightGBM L1, fit on absolute error, had order-quantity WAPE 0.522505 (bias -0.485450).

Dropping SKU-origins with no sale in the 90 days before the origin (dates after origin minus 90 days, through the origin), without refitting, leaves 391 of 400 SKU-origins. On that slice, order-quantity WAPE was 0.353240 for Tweedie (bias -0.065094) and 0.395996 for the moving average, 10.80% lower for Tweedie.

Horizon cuts are partial sums of the fitted 28-day forecasts, not refits. days 1-7: Tweedie WAPE 0.571241 is 0.70% higher than the moving average (WAPE 0.567251); days 1-14: Tweedie WAPE 0.437721 is 5.23% lower than the moving average (WAPE 0.461885).

In Q1, the lowest backtest-volume quartile (descriptive, not a feature; 49108 units, mean order 491.08), order-quantity WAPE was 0.561226 for Tweedie (bias +0.332390), 0.553820 for the moving average (bias +0.024864), and 0.433232 for L1 (bias -0.273265). Tweedie was above the 28-day actual on 77 of 100 SKU-origins, against 42 of 100 for the moving average.

SKU 84270 (ASSTD DESIGN BUBBLE GUM RING) has actual 0 on all 10 scored origins. The last positive day in the panel is 2010-12-06, and it sold 20866 units before the first origin. The moving average forecasts 0.00 units in total; Tweedie forecasts 1240.10. Its share of the net absolute-error reduction versus the moving average is -5.92%. A negative share increases Tweedie's absolute error.

LightGBM refit on the SKUs with the most units before the first origin. Not a slice of the top-40 model. On this 20-SKU refit, order-quantity WAPE was 0.373500 for Tweedie (bias -0.104242) and 0.396441 for the moving average (5.79% lower for Tweedie), across 200 SKU-origins. Same SKUs scored from the top-40 model. The model was not refit. That slice has Tweedie WAPE 0.357125 and moving-average WAPE 0.396441 (9.92% lower for Tweedie). The refit assortment includes 84270, which has no sales in the scored window.

Horizon cuts (order quantity):

| days | evaluation | model | points | WAPE | bias |
| --- | --- | --- | --- | --- | --- |
| 1-7 | partial_sum_not_a_refit | 28-day moving average | 400 | 0.567251 | -0.098285 |
| 1-7 | partial_sum_not_a_refit | LightGBM L1 | 400 | 0.607465 | -0.501729 |
| 1-7 | partial_sum_not_a_refit | LightGBM Tweedie | 400 | 0.571241 | -0.063422 |
| 8-14 | partial_sum_not_a_refit | 28-day moving average | 400 | 0.583832 | -0.048655 |
| 8-14 | partial_sum_not_a_refit | LightGBM L1 | 400 | 0.610325 | -0.492410 |
| 8-14 | partial_sum_not_a_refit | LightGBM Tweedie | 400 | 0.544067 | -0.075710 |
| 15-21 | partial_sum_not_a_refit | 28-day moving average | 400 | 0.570639 | +0.103803 |
| 15-21 | partial_sum_not_a_refit | LightGBM L1 | 400 | 0.536636 | -0.404182 |
| 15-21 | partial_sum_not_a_refit | LightGBM Tweedie | 400 | 0.521451 | +0.054101 |
| 22-28 | partial_sum_not_a_refit | 28-day moving average | 400 | 0.591624 | -0.127007 |
| 22-28 | partial_sum_not_a_refit | LightGBM L1 | 400 | 0.615523 | -0.527577 |
| 22-28 | partial_sum_not_a_refit | LightGBM Tweedie | 400 | 0.565939 | -0.144299 |
| 1-14 | partial_sum_not_a_refit | 28-day moving average | 400 | 0.461885 | -0.074135 |
| 1-14 | partial_sum_not_a_refit | LightGBM L1 | 400 | 0.559207 | -0.497194 |
| 1-14 | partial_sum_not_a_refit | LightGBM Tweedie | 400 | 0.437721 | -0.069401 |
| 1-28 | partial_sum_not_a_refit | 28-day moving average | 400 | 0.395996 | -0.050239 |
| 1-28 | partial_sum_not_a_refit | LightGBM L1 | 400 | 0.522505 | -0.485450 |
| 1-28 | partial_sum_not_a_refit | LightGBM Tweedie | 400 | 0.355129 | -0.063205 |

Active-in-window slice (order quantity):

| grain | model | points | WAPE | bias |
| --- | --- | --- | --- | --- |
| order_total | 28-day moving average | 391 | 0.395996 | -0.050239 |
| order_total | LightGBM L1 | 391 | 0.520769 | -0.487186 |
| order_total | LightGBM Tweedie | 391 | 0.353240 | -0.065094 |

Pre-cutoff top-SKU refit, then the same SKUs sliced from the top-40 model:

refit_pre_cutoff_top_n. LightGBM refit on the SKUs with the most units before the first origin. Not a slice of the top-40 model.

| grain | model | points | WAPE | bias |
| --- | --- | --- | --- | --- |
| order_total | 28-day moving average | 200 | 0.396441 | -0.055483 |
| order_total | LightGBM L1 | 200 | 0.514967 | -0.473191 |
| order_total | LightGBM Tweedie | 200 | 0.373500 | -0.104242 |

slice_of_top40_not_a_refit. Same SKUs scored from the top-40 model. The model was not refit.

| grain | model | points | WAPE | bias |
| --- | --- | --- | --- | --- |
| order_total | 28-day moving average | 200 | 0.396441 | -0.055483 |
| order_total | LightGBM L1 | 200 | 0.540741 | -0.511510 |
| order_total | LightGBM Tweedie | 200 | 0.357125 | -0.090894 |

## Order quantity, bias, and the daily aggregate

| grain | model | points | WAPE | bias |
| --- | --- | --- | --- | --- |
| daily | weekly seasonal naive | 11200 | 1.276659 | +0.054680 |
| order_total | weekly seasonal naive | 400 | 0.557692 | +0.054680 |
| daily_aggregate | weekly seasonal naive | 280 | 0.568517 | +0.054680 |
| daily | yearly seasonal naive | 11200 | 1.458377 | +0.342904 |
| order_total | yearly seasonal naive | 400 | 0.646218 | +0.342904 |
| daily_aggregate | yearly seasonal naive | 280 | 0.725395 | +0.342904 |
| daily | 28-day moving average | 11200 | 1.053000 | -0.050239 |
| order_total | 28-day moving average | 400 | 0.395996 | -0.050239 |
| daily_aggregate | 28-day moving average | 280 | 0.588800 | -0.050239 |
| daily | LightGBM L1 | 11200 | 0.824654 | -0.485450 |
| order_total | LightGBM L1 | 400 | 0.522505 | -0.485450 |
| daily_aggregate | LightGBM L1 | 280 | 0.537963 | -0.485450 |
| daily | LightGBM Tweedie | 11200 | 0.933069 | -0.063205 |
| order_total | LightGBM Tweedie | 400 | 0.355129 | -0.063205 |
| daily_aggregate | LightGBM Tweedie | 280 | 0.393766 | -0.063205 |

Bias is (sum of forecast minus actual) / sum of actual. A negative value under-forecasts the total.

## Daily point metrics

| model | points | MAE | RMSE | WAPE | sMAPE | bias |
| --- | --- | --- | --- | --- | --- | --- |
| weekly seasonal naive | 11200 | 58.4178 | 193.7641 | 1.276659 | 0.964985 | +0.054680 |
| yearly seasonal naive | 11200 | 66.7329 | 228.8161 | 1.458377 | 1.002279 | +0.342904 |
| 28-day moving average | 11200 | 48.1835 | 128.8235 | 1.053000 | 1.141511 | -0.050239 |
| LightGBM L1 | 11200 | 37.7347 | 129.6483 | 0.824654 | 1.177392 | -0.485450 |
| LightGBM Tweedie | 11200 | 42.6956 | 125.0627 | 0.933069 | 1.190162 | -0.063205 |

Per-fold rows are in `metrics.md`. On the same rows, a relative MAE gap equals the relative WAPE gap.

## Weekly track

ISO weeks (Monday start), horizon 4. The 1-week naive repeats the origin week. The 52-week naive is the target week minus 364 days. The first and last weeks of the panel can be shorter than seven days.

| grain | model | points | WAPE | bias |
| --- | --- | --- | --- | --- |
| daily | 1-week seasonal naive | 1600 | 0.734789 | -0.033896 |
| order_total | 1-week seasonal naive | 400 | 0.612399 | -0.033896 |
| daily_aggregate | 1-week seasonal naive | 40 | 0.244064 | -0.033896 |
| daily | 52-week seasonal naive | 1600 | 0.938724 | +0.341604 |
| order_total | 52-week seasonal naive | 400 | 0.639445 | +0.341604 |
| daily_aggregate | 52-week seasonal naive | 40 | 0.446379 | +0.341604 |
| daily | 4-week moving average | 1600 | 0.591555 | -0.042427 |
| order_total | 4-week moving average | 400 | 0.422572 | -0.042427 |
| daily_aggregate | 4-week moving average | 40 | 0.228902 | -0.042427 |
| daily | LightGBM Tweedie | 1600 | 0.584924 | -0.003710 |
| order_total | LightGBM Tweedie | 400 | 0.399551 | -0.003710 |
| daily_aggregate | LightGBM Tweedie | 40 | 0.199747 | -0.003710 |

## Lumpy SKUs

A SKU-origin is lumpy when ADI > 1.32 and CV² > 0.49 on history from the first positive day through the origin. Croston, SBA, and TSB use alpha 0.1, are flat across the horizon, and are not included in the all-SKU WAPE. Croston and SBA do not update on trailing zeros. TSB updates the demand probability every period after the first demand.

| model | points | WAPE | bias |
| --- | --- | --- | --- |
| Croston | 7728 | 1.137167 | -0.033915 |
| SBA | 7728 | 1.113449 | -0.082219 |
| TSB | 7728 | 1.124895 | -0.039376 |
| 28-day moving average | 7728 | 1.134960 | -0.046231 |
| LightGBM Tweedie | 7728 | 1.002714 | -0.101772 |
| LightGBM L1 | 7728 | 0.882282 | -0.548452 |

## Error by segment

| segment | grain | model | points | actual units | WAPE | bias |
| --- | --- | --- | --- | --- | --- | --- |
| volume_quartile: Q1 | daily | 28-day moving average | 2800 | 49108.0 | 1.204466 | +0.024864 |
| volume_quartile: Q1 | order_total | 28-day moving average | 100 | 49108.0 | 0.553820 | +0.024864 |
| volume_quartile: Q1 | daily | LightGBM L1 | 2800 | 49108.0 | 0.976028 | -0.273265 |
| volume_quartile: Q1 | order_total | LightGBM L1 | 100 | 49108.0 | 0.433232 | -0.273265 |
| volume_quartile: Q1 | daily | LightGBM Tweedie | 2800 | 49108.0 | 1.265599 | +0.332390 |
| volume_quartile: Q1 | order_total | LightGBM Tweedie | 100 | 49108.0 | 0.561226 | +0.332390 |
| volume_quartile: Q2 | daily | 28-day moving average | 2800 | 86933.0 | 1.022347 | -0.022903 |
| volume_quartile: Q2 | order_total | 28-day moving average | 100 | 86933.0 | 0.373794 | -0.022903 |
| volume_quartile: Q2 | daily | LightGBM L1 | 2800 | 86933.0 | 0.794486 | -0.384755 |
| volume_quartile: Q2 | order_total | LightGBM L1 | 100 | 86933.0 | 0.414577 | -0.384755 |
| volume_quartile: Q2 | daily | LightGBM Tweedie | 2800 | 86933.0 | 0.934622 | +0.043151 |
| volume_quartile: Q2 | order_total | LightGBM Tweedie | 100 | 86933.0 | 0.287469 | +0.043151 |
| volume_quartile: Q3 | daily | 28-day moving average | 2800 | 122850.0 | 1.085088 | -0.068067 |
| volume_quartile: Q3 | order_total | 28-day moving average | 100 | 122850.0 | 0.437086 | -0.068067 |
| volume_quartile: Q3 | daily | LightGBM L1 | 2800 | 122850.0 | 0.844980 | -0.498010 |
| volume_quartile: Q3 | order_total | LightGBM L1 | 100 | 122850.0 | 0.539030 | -0.498010 |
| volume_quartile: Q3 | daily | LightGBM Tweedie | 2800 | 122850.0 | 0.934459 | -0.089029 |
| volume_quartile: Q3 | order_total | LightGBM Tweedie | 100 | 122850.0 | 0.324310 | -0.089029 |
| volume_quartile: Q4 | daily | 28-day moving average | 2800 | 253602.0 | 1.018634 | -0.065516 |
| volume_quartile: Q4 | order_total | 28-day moving average | 100 | 253602.0 | 0.353140 | -0.065516 |
| volume_quartile: Q4 | daily | LightGBM L1 | 2800 | 253602.0 | 0.795836 | -0.554971 |
| volume_quartile: Q4 | order_total | LightGBM L1 | 100 | 253602.0 | 0.568784 | -0.554971 |
| volume_quartile: Q4 | daily | LightGBM Tweedie | 2800 | 253602.0 | 0.867471 | -0.163758 |
| volume_quartile: Q4 | order_total | LightGBM Tweedie | 100 | 253602.0 | 0.353343 | -0.163758 |
| demand_class: erratic | daily | 28-day moving average | 3472 | 186994.0 | 0.910332 | -0.057216 |
| demand_class: erratic | order_total | 28-day moving average | 124 | 186994.0 | 0.298721 | -0.057216 |
| demand_class: erratic | daily | LightGBM L1 | 3472 | 186994.0 | 0.724340 | -0.375783 |
| demand_class: erratic | order_total | LightGBM L1 | 124 | 186994.0 | 0.412894 | -0.375783 |
| demand_class: erratic | daily | LightGBM Tweedie | 3472 | 186994.0 | 0.811838 | +0.003927 |
| demand_class: erratic | order_total | LightGBM Tweedie | 124 | 186994.0 | 0.283046 | +0.003927 |
| demand_class: lumpy | daily | 28-day moving average | 7728 | 325499.0 | 1.134960 | -0.046231 |
| demand_class: lumpy | order_total | 28-day moving average | 276 | 325499.0 | 0.451879 | -0.046231 |
| demand_class: lumpy | daily | LightGBM L1 | 7728 | 325499.0 | 0.882282 | -0.548452 |
| demand_class: lumpy | order_total | LightGBM L1 | 276 | 325499.0 | 0.585475 | -0.548452 |
| demand_class: lumpy | daily | LightGBM Tweedie | 7728 | 325499.0 | 1.002714 | -0.101772 |
| demand_class: lumpy | order_total | LightGBM Tweedie | 276 | 325499.0 | 0.396540 | -0.101772 |

The full segment table, including the naive forecasts, is `segment_metrics.csv`.

## Inventory cost

For each SKU-origin the order quantity is the 28-day sum. Units short are actual minus forecast when that is positive; units over are the reverse. Holding cost per unit equals the origin unit price (the last realized price, not the price on the target day). Stockout cost per unit is 3 times holding cost. Cost = units over × price + 3 × units short × price. Rows with a missing or non-positive price stay in the unit totals and drop out of the money totals.

Stockout-to-holding ratio 3:1. Holding basis: origin unit price. Value and cost are in the dataset currency (sterling).

| model | SKU-origins | missing price | units short | units over | value short | value over | cost |
| --- | --- | --- | --- | --- | --- | --- | --- |
| LightGBM Tweedie | 400 | 0 | 107196.78 | 74804.45 | 138,959.40 | 123,111.67 | 539,989.86 |
| LightGBM L1 | 400 | 0 | 258284.98 | 9495.19 | 368,180.33 | 19,584.06 | 1,124,125.06 |
| 28-day moving average | 400 | 0 | 114346.00 | 88599.00 | 165,569.20 | 126,213.59 | 622,921.20 |

Highest Tweedie costs:

| origin | sku | class | units short | units over | cost |
| --- | --- | --- | --- | --- | --- |
| 2011-07-12 | 84879 | erratic | 3550.49 | 0.00 | 18,000.99 |
| 2011-11-01 | 22197 | lumpy | 6364.11 | 0.00 | 15,185.61 |
| 2011-03-22 | 85099B | erratic | 1512.90 | 0.00 | 14,221.71 |
| 2011-03-22 | 22423 | erratic | 222.46 | 0.00 | 10,807.30 |
| 2011-08-09 | 85099B | erratic | 1704.97 | 0.00 | 10,639.03 |

The SKU-origin table is `inventory_by_sku_origin.csv`. L1 is the absolute-error contrast. The moving average is included because it is the order-quantity baseline.

## Sample and reconciliation

Selection cutoff 2011-02-22, which is the first daily origin. Ranks use dates strictly before that day. 40 SKUs were taken from 430 with at least 180 sale days in the window. They are 17.96% of units before the cutoff and, as a descriptive full-period share only, 16.49% of cleaned units from 2009-12-01 to 2011-12-09. Examples: 85123A WHITE HANGING HEART T-LIGHT HOLDER (60,447 units before the cutoff, 360 sale days); 84077 WORLD WAR 2 GLIDERS ASSTD DESIGNS (56,976 units before the cutoff, 263 sale days); 21212 PACK OF 72 RETROSPOT CAKE CASES (51,436 units before the cutoff, 348 sale days). Cancellation matches: 3670 (sales lines 935027, clean lines 931357). Clean quantity 8950843 equals daily units 8950843. Panel rows 29560 and distinct keys 29560. Supervised rows with as-of on or after the target: 0. Weekly versus daily unit mismatches: 0.

Charts: `order_wape_by_fold.png` (28-day order-quantity WAPE by origin), `bias.png` (bias by grain, and order-quantity WAPE), `forecast_vs_actual.png`, `wape_by_fold.png`.
