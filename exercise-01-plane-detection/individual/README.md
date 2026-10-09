# Exercise 1 · MLESAC and Preemptive RANSAC

This experiment compares RANSAC with the exercise's residual-cost MLESAC variant:
inliers contribute their point-to-plane residual, while outliers contribute a
constant `gamma`. The cost differs from the probabilistic mixture likelihood
used in standard MLESAC.

The Preemptive RANSAC study evaluates hypothesis counts `M ∈ {64, 256, 1024}`
and block sizes `B ∈ {64, 256, 1024}`. The selected configuration is `M=1024, B=256`.

## Run

Place the Kinect MAT files in [data/](data/README.md), then open PowerShell in this folder:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe individual_ex01.py --data-dir data --output-dir results
```

The default input is `example4kinect.mat`; use `--input-file` to select another
MAT file. Figures and measurements are written to `results/`.

## Results

- [Discussion](reports/discussion.txt)
- [Inlier-threshold study](submitted_results/mlesac_epsilon_summary.csv)
- [Preemptive RANSAC parameter study](submitted_results/preemptive_parameter_summary.csv)
- [Hypothesis comparison](submitted_results/three_M_comparison.png)


[Exercise overview](../README.md) · [Project](../../README.md)
