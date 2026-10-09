# Exercise 4 · Balloon detection

The pipeline generates Selective Search proposals, assigns training labels using
intersection over union, extracts 512-dimensional pretrained ResNet-18 features
and trains a linear SVM. A score threshold and non-maximum suppression produce
the final detections.

## Run

Place the balloon dataset in [data/balloon_dataset/](data/README.md), with
`train`, `valid` and `test` splits. Each split needs its images and
`_annotations.coco.json`. Open PowerShell in this folder:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe code/pipeline.py
```

The first run downloads torchvision's ResNet-18 weights if they are not cached.
Intermediate files go to `work/balloon_pipeline/`. Each run replaces generated
outputs in `results/balloon_pipeline/`.

## Configuration

| Parameter | Value |
| --- | --- |
| Positive / negative IoU thresholds | 0.75 / 0.25 |
| SVM | Linear, C=0.01, balanced class weights |
| PCA | Disabled |
| Score threshold / NMS IoU | 0.4 / 0.3 |
| Top proposals for evaluation | 200 |

## Results

| Metric | Value |
| --- | --- |
| mAP@0.50:0.95 | 0.0916125 |
| AP@0.50 | 0.3830015 |
| Mean average best overlap of test proposals | 0.6813746 |
| Proposal recall at IoU 0.50 / 0.75 | 1.00 / 0.20 |
| Final detections / ground-truth boxes | 36 / 25 |

See the [detection evaluation](submitted_results/balloon_pipeline/evaluation.json)
and [classifier training metrics](submitted_results/balloon_pipeline/training_metrics.json).

[Report](reports/writeup_5_2_detection_pipeline.pdf) ·
[LaTeX source](reports/writeup_5_2_detection_pipeline.tex) ·
[Detection examples](submitted_results/balloon_pipeline/visualizations)


[Exercise overview](../README.md) · [Project](../../README.md)
