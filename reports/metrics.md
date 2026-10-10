# Backtest metrics

On a 28-day horizon across 10 rolling origins, LightGBM pooled WAPE was 0.833938: 37.27% lower than the weekly seasonal naive (WAPE 1.329340); 42.29% lower than the yearly seasonal naive (WAPE 1.445168); 22.68% lower than the 28-day moving average (WAPE 1.078537).

Pooled MAE was 42.7183 for LightGBM and 68.0952 for the weekly seasonal naive (37.27% lower). The relative MAE reduction matches the WAPE reduction because both are proportional to total absolute error.

Pooled RMSE was 143.6927 for LightGBM, 212.5820 for the weekly seasonal naive, 287.4805 for the yearly seasonal naive, and 142.6519 for the 28-day moving average.

Pooled sMAPE was 1.227377 for LightGBM, 0.969943 for the weekly seasonal naive, 0.961749 for the yearly seasonal naive, and 1.232403 for the 28-day moving average.

WAPE and sMAPE in this table are ratios (sMAPE uses the 0/0 = 0 convention and is not multiplied by 100). The CSV stores the same figures at full precision. Displayed rates are rounded to 6 decimal places and MAE/RMSE to 4.

| fold | origin | model | points | MAE | RMSE | WAPE | sMAPE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 2011-02-22 | weekly seasonal naive | 1120 | 53.5161 | 133.4237 | 1.175439 | 0.997353 |
| 1 | 2011-02-22 | yearly seasonal naive | 1120 | 77.1286 | 466.1100 | 1.694070 | 0.952412 |
| 1 | 2011-02-22 | 28-day moving average | 1120 | 46.6085 | 113.8961 | 1.023721 | 1.220685 |
| 1 | 2011-02-22 | LightGBM | 1120 | 38.1106 | 117.2352 | 0.837070 | 1.231223 |
| 2 | 2011-03-22 | weekly seasonal naive | 1120 | 80.5259 | 210.0962 | 1.593023 | 0.974101 |
| 2 | 2011-03-22 | yearly seasonal naive | 1120 | 61.7446 | 162.0506 | 1.221478 | 0.994540 |
| 2 | 2011-03-22 | 28-day moving average | 1120 | 54.5166 | 135.0088 | 1.078489 | 1.272304 |
| 2 | 2011-03-22 | LightGBM | 1120 | 42.9046 | 139.0861 | 0.848771 | 1.295469 |
| 3 | 2011-04-19 | weekly seasonal naive | 1120 | 59.0036 | 152.7273 | 1.263073 | 0.943388 |
| 3 | 2011-04-19 | yearly seasonal naive | 1120 | 68.9214 | 210.8179 | 1.475382 | 0.943215 |
| 3 | 2011-04-19 | 28-day moving average | 1120 | 55.3858 | 111.9386 | 1.185628 | 1.355344 |
| 3 | 2011-04-19 | LightGBM | 1120 | 42.5482 | 113.3486 | 0.910818 | 1.378615 |
| 4 | 2011-05-17 | weekly seasonal naive | 1120 | 60.5089 | 186.3936 | 1.173710 | 0.948684 |
| 4 | 2011-05-17 | yearly seasonal naive | 1120 | 68.9170 | 208.8442 | 1.336803 | 0.854333 |
| 4 | 2011-05-17 | 28-day moving average | 1120 | 56.8552 | 179.8477 | 1.102838 | 1.273231 |
| 4 | 2011-05-17 | LightGBM | 1120 | 44.2260 | 180.8801 | 0.857866 | 1.255255 |
| 5 | 2011-06-14 | weekly seasonal naive | 1120 | 50.5107 | 132.5496 | 1.177896 | 0.912979 |
| 5 | 2011-06-14 | yearly seasonal naive | 1120 | 59.8000 | 138.4042 | 1.394520 | 1.002904 |
| 5 | 2011-06-14 | 28-day moving average | 1120 | 55.1698 | 114.6377 | 1.286546 | 1.319190 |
| 5 | 2011-06-14 | LightGBM | 1120 | 36.9352 | 110.7331 | 0.861318 | 1.260168 |
| 6 | 2011-07-12 | weekly seasonal naive | 1120 | 67.8295 | 210.7299 | 1.332532 | 1.019810 |
| 6 | 2011-07-12 | yearly seasonal naive | 1120 | 61.7357 | 197.8223 | 1.212819 | 0.984553 |
| 6 | 2011-07-12 | 28-day moving average | 1120 | 52.9838 | 172.1915 | 1.040884 | 1.198415 |
| 6 | 2011-07-12 | LightGBM | 1120 | 43.0668 | 173.2279 | 0.846062 | 1.229491 |
| 7 | 2011-08-09 | weekly seasonal naive | 1120 | 77.4313 | 258.0847 | 1.727997 | 1.004621 |
| 7 | 2011-08-09 | yearly seasonal naive | 1120 | 81.0259 | 376.9226 | 1.808217 | 1.000706 |
| 7 | 2011-08-09 | 28-day moving average | 1120 | 55.6304 | 114.9215 | 1.241477 | 1.266448 |
| 7 | 2011-08-09 | LightGBM | 1120 | 37.8636 | 114.1819 | 0.844983 | 1.264213 |
| 8 | 2011-09-06 | weekly seasonal naive | 1120 | 63.1375 | 157.8629 | 1.157311 | 0.968904 |
| 8 | 2011-09-06 | yearly seasonal naive | 1120 | 70.7893 | 161.1765 | 1.297568 | 0.997120 |
| 8 | 2011-09-06 | 28-day moving average | 1120 | 53.7641 | 113.7818 | 0.985496 | 1.164645 |
| 8 | 2011-09-06 | LightGBM | 1120 | 43.2847 | 113.8852 | 0.793408 | 1.139892 |
| 9 | 2011-10-04 | weekly seasonal naive | 1120 | 68.8616 | 225.0831 | 1.333074 | 0.914532 |
| 9 | 2011-10-04 | yearly seasonal naive | 1120 | 86.2098 | 297.8257 | 1.668914 | 0.998761 |
| 9 | 2011-10-04 | 28-day moving average | 1120 | 57.0232 | 182.4981 | 1.103896 | 1.187042 |
| 9 | 2011-10-04 | LightGBM | 1120 | 43.0644 | 183.8332 | 0.833673 | 1.161800 |
| 10 | 2011-11-01 | weekly seasonal naive | 1120 | 99.6268 | 357.3981 | 1.362950 | 1.015054 |
| 10 | 2011-11-01 | yearly seasonal naive | 1120 | 104.0125 | 424.0677 | 1.422949 | 0.888947 |
| 10 | 2011-11-01 | 28-day moving average | 1120 | 64.5413 | 158.6505 | 0.882961 | 1.066728 |
| 10 | 2011-11-01 | LightGBM | 1120 | 55.1792 | 160.4667 | 0.754882 | 1.057648 |
| pooled |  | weekly seasonal naive | 11200 | 68.0952 | 212.5820 | 1.329340 | 0.969943 |
| pooled |  | yearly seasonal naive | 11200 | 74.0285 | 287.4805 | 1.445168 | 0.961749 |
| pooled |  | 28-day moving average | 11200 | 55.2479 | 142.6519 | 1.078537 | 1.232403 |
| pooled |  | LightGBM | 11200 | 42.7183 | 143.6927 | 0.833938 | 1.227377 |
