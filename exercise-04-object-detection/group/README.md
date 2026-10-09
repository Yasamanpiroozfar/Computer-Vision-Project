# Exercise 4 · Selective Search

Selective Search generates region proposals by hierarchically merging image
segments using colour, texture, size and fill similarities. The experiment uses
nine images from the `arthist`, `chrisarch` and `classarch` collections.

The runner uses HSV colour and Felzenszwalb segmentation with scale 100,
sigma 0.8 and minimum segment size 20. Images are resized to a maximum search
dimension of 420 pixels.

## Run

Place the three image collections under [data/](data/README.md), then open
PowerShell in this folder:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe code/main.py
```

Proposal visualizations and counts are written to `results/selective_search/`.
At least one input image is required.

## Results

- [Report](reports/writeup_5_1_selective_search.pdf) and [LaTeX source](reports/writeup_5_1_selective_search.tex)
- [Proposal contact sheet](submitted_results/selective_search/contact_sheet_moderate_filter.jpg)
- [Proposal counts](submitted_results/selective_search/proposal_counts.csv)


[Exercise overview](../README.md) · [Project](../../README.md)
