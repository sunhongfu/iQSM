# Legacy Code

This directory contains original research code preserved for reference and reproducibility.
The production inference pipeline is in [`inference.py`](../inference.py) and [`models/`](../models/).

## matlab/

MATLAB scripts for running iQSM from MATLAB (calls Python internally via the
`demo_single_echo.m` entry point) and helper functions (`iQSM_fcns/`), including
data simulation scripts used to generate the training dataset.

## python/

Original Python evaluation and training scripts from the paper.
These are **not** used by the Gradio app or `run.py`.

### Evaluation variants

| Directory | Description |
|-----------|-------------|
| `python/PythonCodes/Evaluation/LearnableLapLayer/` | **Active model** — learnable LoT layer (v2 checkpoints) |
| `python/PythonCodes/Evaluation/DataFidelityVersion/` | Data-fidelity variant |
| `python/PythonCodes/Evaluation/` | Other evaluation scripts |

### Training code

`python/PythonCodes/Training/` contains scripts to train iQSM and iQFM from scratch.
See the README in the parent directory for dataset preparation instructions.
