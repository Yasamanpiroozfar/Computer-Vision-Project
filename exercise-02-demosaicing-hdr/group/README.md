# Exercise 2 · Demosaicing and RAW HDR

`solution_ex02.ipynb` follows the image-formation pipeline from a GBRG Bayer
pattern to demosaicing, gamma and logarithmic curves, gray-world white balance,
sensor-linearity analysis, RAW HDR merging and iCAM06 tone mapping. The final
task combines the processing steps in `process_raw`.

## Run

Place the RAW and NPY inputs under `exercise_2_data/`; see the
[input list](exercise_2_data/README.md). Start PowerShell in this `group` folder:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m jupyterlab solution_ex02.ipynb
```

Select the environment's Python kernel and run the cells in order. The notebook
uses its working directory as `BASE_DIR`, so start Jupyter from this folder.
Image exports are written to `outputs_ex02/`.

## Example outputs

| Task | Preview |
| --- | --- |
| Bayer pattern | [GBRG](submitted_results/notebook-previews/01-bayer-preview.png) |
| Demosaicing | [Reconstructed image](submitted_results/notebook-previews/02-demosaicing.png) |
| Luminosity | [Gamma](submitted_results/notebook-previews/03-gamma.png) · [Log curve](submitted_results/notebook-previews/03-logarithmic-curve.png) |
| White balance | [Gray world](submitted_results/notebook-previews/04-white-balance.png) |
| Sensor linearity | [Plot](submitted_results/ex05_sensor_linearity_plot.png) |
| RAW HDR | [Log tone map](submitted_results/notebook-previews/06-hdr.png) |
| iCAM06 | [Setting 1](submitted_results/notebook-previews/07-icam06-setting-1.png) · [Setting 2](submitted_results/notebook-previews/07-icam06-setting-2.png) |


[Exercise overview](../README.md) · [Project](../../README.md)
