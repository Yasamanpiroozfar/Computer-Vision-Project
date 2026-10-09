# Exercise 2 · HDR from JPG exposures

`individual_ex02.py` estimates a camera response, linearizes twelve JPG exposures
and merges them into an HDR image. Exposure times are read from EXIF metadata,
and the sequence is sorted from longest to shortest exposure.

## Run

Place the exposure sequence in [data/hdr-jpg/](data/README.md), keeping its EXIF
metadata intact. Open PowerShell in this folder:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe individual_ex02.py --input-dir data/hdr-jpg --output-dir results
```

The script writes full-resolution images, diagnostic plots and measurements to `results/`.

## Results

- [Method and limitations](reports/discussion.txt)
- [Estimated camera response](submitted_results/estimated_camera_response.png)
- [HDR comparison](submitted_results/hdr_comparison.jpg)
- [Linearity check](submitted_results/linearization_check.png)
- [Exposure-ratio measurements](submitted_results/linearization_ratios.csv)


[Exercise overview](../README.md) · [Project](../../README.md)
