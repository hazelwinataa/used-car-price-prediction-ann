# Evidence map and metric qualifications

All figures are taken from saved `SOAL_1.ipynb` notebook outputs (converted to WebP), with the PDF reproduction preserved in `reports/`.

| Saved notebook (zero-based) cell | Output |
|---:|---|
| 107 | Selling price distribution before log transform |
| 167 | Numeric-feature correlation heatmap |
| 234 | Selling price distribution after log transform |
| 316 | Baseline model training curves |
| 321 | Baseline train/validation/test metrics |
| 342 | Modified model training curves |
| 345 | Modified train/validation/test metrics |
| 347 | Modified test actual vs predicted and residual plots |
| 350 | Final side-by-side comparison of ANN candidates |

The network was assessed on `log1p(selling_price)`, so the recorded numerical errors do **not** represent USD error. The cleaned cohort contains 1,648 records; the unavailable original dataset prevents independently auditing each filtering step or rerunning the pipeline. Test set size is 330.
