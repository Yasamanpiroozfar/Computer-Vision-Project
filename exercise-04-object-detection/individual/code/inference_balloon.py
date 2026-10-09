"""Inference wrapper for the Exercise 5.2 detector.

Usage from project root after training:
    python code/inference_balloon.py path/to/image.jpg
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python code/inference_balloon.py path/to/image.jpg")
    project_root = Path(__file__).resolve().parents[1]
    cmd = [
        sys.executable,
        str(project_root / "code" / "balloon_pipeline.py"),
        "--mode",
        "infer",
        "--image",
        sys.argv[1],
    ]
    raise SystemExit(subprocess.call(cmd, cwd=str(project_root)))
