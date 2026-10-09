# Exercise 5 · Open-set learning with SPL and MPL

`osr_learning.py` implements single-pseudo-label (SPL) and multi-pseudo-label
(MPL) learning. Each training function returns a predictor with class labels and
knownness scores. SPL compares nearest known and unknown samples. MPL represents
each known class by a centroid and assigns a separate pseudo-class to each
known-unknown training sample.

## Run

Place `challenge_train_data.csv` in `../group/data/`. The final column contains
integer class labels, with `-1` for unknown samples. Open PowerShell in this folder:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
$env:PYTHONPATH = (Resolve-Path "..\group\src").Path
.\.venv\Scripts\python.exe osr_learning.py
```

The script imports `Config` from the group package. Its entry point demonstrates
prediction on the first 50 training rows; the nine-split evaluation harness is
not included in this repository.

## Validation results

| Method | AUCROC | DIR at FAR=1% | Balanced rank-1 |
| --- | --- | --- | --- |
| SPL | 0.979724 | 0.867395 | 0.979235 |
| MPL | 0.985263 | 0.916393 | 0.988525 |

Values are averaged over nine validation splits. See the [report](reports/report.txt)
for the split setup and method details.


[Exercise overview](../README.md) · [Project](../../README.md)
