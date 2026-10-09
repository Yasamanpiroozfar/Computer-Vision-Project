# Exercise 3 · Writer retrieval with VLAD and Exemplar SVM

The retrieval pipeline extracts SIFT descriptors with keypoint orientation set
to zero, applies Hellinger normalization and aggregates descriptors into VLAD
vectors using a 100-center visual vocabulary. Retrieval uses cosine similarity,
followed by an Exemplar SVM transformation of the representation.

## Run

Place the ICDAR17 images and label files in [data/](data/README.md), then open
PowerShell in this folder:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe skeleton_new.py --in_train data/train --in_test data/test --labels_train data/icdar17_labels_train.txt --labels_test data/icdar17_labels_test.txt --suffix_train .png --suffix_test .jpg
```

Adjust `--suffix_train` and `--suffix_test` to match the image files.
`--powernorm` enables signed square-root normalization and is off by default.

The script saves `mus.pkl.gz`, `enc_train.pkl.gz` and `enc_test.pkl.gz` in this
folder. The encoding caches contain VLAD vectors for the individual extension;
the Exemplar SVM transformation is evaluated in memory afterwards.

## Results

| Representation | Top-1 | mAP |
| --- | --- | --- |
| VLAD | 81.94% | 0.6305 |
| VLAD + Exemplar SVM | 87.86% | 0.7417 |

[Report](reports/report.pdf) · [Experiment log](submitted_results/output.txt)


[Exercise overview](../README.md) · [Project](../../README.md)
