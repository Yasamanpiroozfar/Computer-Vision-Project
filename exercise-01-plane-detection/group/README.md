# Exercise 1 · RANSAC plane detection

`ex01.py` detects the floor and box-top planes in a Kinect point cloud. It applies
morphological filtering, selects connected regions and estimates the separation
between the planes. The main fits use an 8 mm inlier threshold and 1,000 RANSAC
iterations.

## Run

Place `example4kinect.mat` in `data/`, then open PowerShell in this folder:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe ex01.py
```

The script displays the point clouds and fitted planes in Matplotlib windows.
RANSAC sampling is random, so estimates can vary between runs. Collinear samples
are an unhandled edge case in the group implementation.


[Exercise overview](../README.md) · [Project](../../README.md)
