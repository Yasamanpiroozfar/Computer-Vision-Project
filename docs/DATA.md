# Data and environments

The repository includes code, reports and example results. Course datasets and
pretrained models must be obtained separately and placed in the following locations.
Paths are relative to each exercise folder.

| Exercise | Inputs | Location |
| --- | --- | --- |
| 1 · Plane detection | Kinect MAT files | [Group data](../exercise-01-plane-detection/group/data/README.md) and [individual data](../exercise-01-plane-detection/individual/data/README.md) |
| 2 · RAW processing | CR3 exposures and NPY Bayer data | [group/exercise_2_data/](../exercise-02-demosaicing-hdr/group/exercise_2_data/README.md) |
| 2 · JPG HDR | Twelve exposures with EXIF metadata | [individual/data/hdr-jpg/](../exercise-02-demosaicing-hdr/individual/data/README.md) |
| 3 · Writer retrieval | ICDAR17 images and writer labels | [group/data/](../exercise-03-writer-retrieval/group/data/README.md) |
| 3 · Retrieval extension | VLAD encodings from the group pipeline | `group/enc_train.pkl.gz` and `group/enc_test.pkl.gz` |
| 4 · Selective Search | Art and architecture image collections | [group/data/](../exercise-04-object-detection/group/data/README.md) |
| 4 · Balloon detection | COCO images and annotations in train/valid/test splits | [individual/data/balloon_dataset/](../exercise-04-object-detection/individual/data/README.md) |
| 5 · Face recognition | Image sequences, ONNX model and evaluation files | [group/data/](../exercise-05-face-recognition/group/data/README.md) |
| 5 · Open-set learning | Challenge training CSV | `group/data/challenge_train_data.csv` |

## Python environments

Create a separate environment for each component and install its `requirements.txt`.
The READMEs use Python 3.10 and PowerShell commands. Most dependencies are unpinned,
so results may vary with library versions and random sampling.

On Linux or macOS, create an environment with `python3 -m venv .venv` and use
`.venv/bin/python` in place of the Windows executable. For Exercise 5, set
`PYTHONPATH` to the group package before running module commands:

```bash
# From exercise-05-face-recognition/group
export PYTHONPATH="$PWD/src"

# From exercise-05-face-recognition/individual
export PYTHONPATH="$(cd ../group/src && pwd)"
```

Start the Exercise 2 notebook from its `group/` folder. The face-recognition
image-sequence tools require a graphical desktop for their OpenCV windows.

[Back to project](../README.md)
