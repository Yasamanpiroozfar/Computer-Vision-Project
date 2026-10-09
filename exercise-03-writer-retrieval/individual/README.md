# Exercise 3 · PCA whitening and reciprocal query expansion

The extension fits PCA on training VLAD vectors, whitens and normalizes the test
representation, then applies reciprocal query expansion (R-QE). Writer labels
are used only for evaluation.

## Run

Run the group pipeline first to generate `enc_train.pkl.gz` and `enc_test.pkl.gz`
in `../group/`. The test labels are read from `../group/data/`.
Open PowerShell in this folder:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
New-Item -ItemType Directory -Force runs | Out-Null
.\.venv\Scripts\python.exe exercise3.py --enc-train ../group/enc_train.pkl.gz --enc-test ../group/enc_test.pkl.gz --labels-test ../group/data/icdar17_labels_test.txt --pca-dims 128,256,512,1024 --rqe-k 1,2,3,5 --combined-k 1,2,3,4,5 --best-pca-dim 1024 --output runs/individual_exercise3_results.txt
```

## Results

| Method | Top-1 | mAP |
| --- | --- | --- |
| VLAD baseline | 82.19% | 0.6297 |
| PCA whitening, 1024D | 86.94% | 0.7171 |
| R-QE on VLAD, k=3 | 82.75% | 0.6539 |
| PCA 1024D + R-QE, k=4 | 86.78% | **0.7481** |
| PCA 1024D + R-QE, k=2 | **87.06%** | 0.7377 |

PCA with R-QE at `k=4` gives the highest mAP; `k=2` gives the highest Top-1 score.
This parameter study uses a separate baseline run from the group experiment.

[Report](reports/individual_exercise3_report.txt) ·
[Full parameter study](submitted_results/individual_exercise3_results.txt)


[Exercise overview](../README.md) · [Project](../../README.md)
