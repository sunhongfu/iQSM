# iQSM – Instant Quantitative Susceptibility Mapping

**Instant Tissue Field and Magnetic Susceptibility Mapping from MRI Raw Phase using Laplacian Enabled Deep Neural Networks**

[NeuroImage 2022](https://www.sciencedirect.com/science/article/pii/S1053811922005274) &nbsp;|&nbsp; [arXiv](https://arxiv.org/abs/2111.07665) &nbsp;|&nbsp; [HuggingFace](https://huggingface.co/sunhongfu/iQSM) &nbsp;|&nbsp; [deepMRI collection](https://github.com/sunhongfu/deepMRI)

iQSM performs single-step, end-to-end local field (iQFM) and susceptibility (QSM) reconstruction directly from raw MRI phase — no separate background field removal step needed. It uses a large-stencil Laplacian preprocessed deep neural network (LoT-Unet).

> **Tip:** For data with resolution finer than 0.7 mm isotropic, interpolate to 1 mm before reconstruction for best results.

## Highlights

- **DICOM folder input** — drop your raw multi-echo GRE folder; phase and magnitude are auto-separated by `ImageType`, echoes are grouped by `EchoTime`, and TEs / voxel size / B0 are read from headers. Works in both the **web app** and the **CLI** (`--dicom_dir`).
- **NIfTI / MAT input** — multiple 3D phase echoes or a single 4D volume (`.nii`, `.nii.gz`, `.mat` v5/v7.3 all supported).
- **Multi-echo support** — the CLI and web app both run iQSM on each echo and combine via magnitude × TE² weighted averaging when magnitude is supplied.
- **Browser-based UI** — collapsible sections, live progress, slice slider, shape verification (mask vs. phase), per-run "equivalent CLI command" log entry, dark-mode auto-open with port auto-fallback.
- **Two outputs** — `iQSM.nii.gz` (susceptibility, χ) and `iQFM.nii.gz` (tissue field, LFS).

## Layout

- `app.py` — Gradio web app for browser-based inference.
- `run.py` — command-line driver (DICOM folder, 4D NIfTI, per-echo files, or YAML config).
- `inference.py` — pure-Python inference pipeline (single-echo run_iqsm).
- `data_utils.py` — NIfTI / MAT / DICOM loaders and shape utilities (DICOM splits phase from magnitude).
- `models/` — LoT-Unet architecture.

---

## Overview

![Framework](figs/fig1.png)

*Fig. 1: iQFM and iQSM framework using the proposed LoT-Unet architecture.*

![Results](figs/fig2.png)

*Fig. 2: Comparison of QSM methods on ICH patients. Red arrows indicate artifacts near hemorrhage sources.*

---

## Quick Start

### 1. Get the code

**Option A — Git**

```bash
git clone https://github.com/sunhongfu/iQSM.git
cd iQSM
```

**Option B — Download ZIP**

1. Open the GitHub repository page.
2. Click **Code** → **Download ZIP**.
3. Unzip and open a terminal in the folder.

---

### 2. Install dependencies

A fresh virtual environment is the recommended way — it isolates iQSM's dependencies from anything else on your system and avoids version conflicts.

You need Python 3.10 or 3.11. Check your version:

```bash
python --version
```

If Python is not installed, download it from [python.org](https://www.python.org/downloads/). On Windows, tick **Add Python to PATH** during installation.

**Create and activate a virtual environment:**

macOS / Linux:
```bash
python -m venv venv
source venv/bin/activate
```

Windows:
```powershell
python -m venv venv
venv\Scripts\activate
```

You should see `(venv)` in your prompt. Run this activation command each time you open a new terminal.

**Install PyTorch.** Go to [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/), select your OS and CUDA version, and copy the install command. For example:

CUDA 12.4 (recommended if you have an NVIDIA GPU):
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

CPU only (slower, but works without a GPU):
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

**Install remaining dependencies.** Pick one of the two options below depending on whether you want the browser-based web app:

- **Web app + Command-Line** (recommended for most users):

  ```bash
  pip install -r requirements-webapp.txt
  ```

  Adds Gradio and Matplotlib for the browser UI and slice previews.

- **Command-Line only** (lighter install, fewer dependencies, no web stack):

  ```bash
  pip install -r requirements.txt
  ```

  Skips Gradio and its ~18 transitive packages (FastAPI, Pydantic, Uvicorn, etc.). Recommended for headless servers, HPC clusters, or environments where Gradio's deps conflict with other tools.

---

### 3. Download checkpoints (and optionally demo data)

Large files (checkpoints and demo data) are excluded from git and hosted on Hugging Face: [sunhongfu/iQSM](https://huggingface.co/sunhongfu/iQSM/tree/main).

**Download checkpoints** (required, one-time):

```bash
python run.py --download-checkpoints
```

**Optional — download demo data:**

```bash
python run.py --download-demo
```

This places sample NIfTI files in `demo/`. See [Run Demo Examples](#run-demo-examples) below.

**Manual download (optional).** If the auto-download fails (e.g. behind a firewall), grab the files from Hugging Face and place them as follows:

```text
iQSM/
├── checkpoints/
│   ├── iQSM_50_v2.pth
│   ├── LPLayer_chi_50_v2.pth
│   ├── iQFM_40_v2.pth
│   └── LoTLayer_lfs_40_v2.pth
└── demo/
    ├── ph_single_echo.nii.gz
    ├── mask_single_echo.nii.gz
    └── params.json
```

---

### 4. Run

Choose the web app (recommended) or the command-line interface.

---

## Web App

```bash
python app.py
```

The app picks port `7860` by default; if it's busy it falls back automatically (`7861`, `7862`, …). Your default browser opens at the dark-themed URL once the server is ready.

### Usage walk-through

The page is organised top-to-bottom; each section is a collapsible accordion.

#### 1. Phase Input

Two tabs — choose **one**:

##### 📁 DICOM Folder *(recommended)*
Click **Select DICOM Folder** and pick the folder containing your multi-echo GRE DICOMs. The app:
- walks the folder recursively (any extension or none — `.dcm`, `.ima`, `.dicom`, …);
- splits **phase** vs **magnitude** via `ImageType` (`P`/`PHASE` vs `M`/`MAGNITUDE`);
- groups by `EchoTime`, sorts each echo by `ImagePositionPatient`;
- normalises phase values to radians where needed;
- builds an LPS→RAS NIfTI affine and writes one NIfTI per modality (`dcm_converted_phase[_4d].nii.gz` and, when present, `dcm_converted_magnitude[_4d].nii.gz`);
- auto-fills **Echo Times**, **Voxel size**, and **B0** from the headers.

Mixed-study folders, single-echo phase folders, or folders with no phase image are rejected with a clear message instead of producing wrong results.

##### 📄 NIfTI / MAT files *(advanced)*
Click **Add Phase NIfTI / MAT** and pick:
- multiple 3D files (one per echo) — `.nii`, `.nii.gz`, or `.mat`; or
- a single 4D volume of shape `(X, Y, Z, n_echoes)`.

Files with unsupported extensions are dropped with a Gradio warning toast. `.mat` files are accepted in **both v5 (default `save`) and v7.3 (`save -v7.3`)** formats.

#### 2. Processing Order
Lists every phase file that will be fed to the pipeline (sorted in natural numeric order: `mag1`, `mag2`, …, `mag10`). Below the list, a one-line shape summary tells you whether all files share the same volume dimensions.

- Click ✕ next to any file to remove it (≥2 files only).
- Click **✕ Remove all phase files** to wipe the list.

#### 3. Echo Times (ms)
A single textbox accepts two equivalent formats:

- **Comma-separated values** — one per echo: `4, 8, 12, 16, 20`
- **Compact `first_TE : spacing : count`** — uniform spacing only: `4 : 4 : 5` expands to `4, 8, 12, 16, 20`

Auto-filled when you use the DICOM Folder tab.

#### 4. Magnitude *(optional)*
Used for **magnitude × TE² weighted averaging** when reconstructing multi-echo data. Without it, multi-echo results use a simple mean. Auto-filled when DICOM input includes magnitude images.

#### 5. Brain Mask *(optional)*
Click **Select Brain Mask** to provide a BET (or any binary) mask. Supported: `.nii`, `.nii.gz`, `.mat` (v5 or v7.3).

After upload, the field shows: `Loaded: BET_mask.nii · Shape: 192 × 256 × 176 · dtype: uint8 · ✓ matches phase`. **Without a mask, all voxels are processed**.

Click **✕ Remove Brain Mask** to clear it.

#### 6. Acquisition & Hyper-parameters *(collapsed by default)*
- **Voxel size** (mm) — overrides NIfTI header.
- **B0** (Tesla) — defaults to 3.0.
- **Mask erosion radius** (voxels) — defaults to 3.
- **Reverse phase sign** — enable if iron-rich deep grey matter appears dark (rather than bright) in the QSM output.

#### 7. Run Reconstruction
Hit the green **Run Reconstruction** button. Three sections appear in sequence:

- **Log** — streaming console output, including a *RUN CONFIGURATION* block that prints the equivalent command-line invocation so you can reproduce the run from a terminal.
- **Results** — `iQSM.nii.gz` (susceptibility) and `iQFM.nii.gz` (tissue field) downloadable when the run completes.
- **Visualisation** — middle-slice grayscale preview of both maps with a **Z-slice slider**. Display windows (`QSM` ± 0.2 ppm, `LFS` ± 0.05 ppm) are editable.

GPU memory is released between runs, so you can upload a new dataset and re-run without restarting the page.

---

## Command-Line Interface

The pipeline can also be driven from the terminal using a YAML config file or explicit arguments.

### Config file

```bash
python run.py --config config.yaml
```

Example `config.yaml`:

```yaml
data_dir: demo

# Pick one input style:
# Style A — DICOM folder:
# dicom_dir: dicoms
# Style B1 — multiple 3D phase echoes:
# echo_files: [phase_e1.nii, phase_e2.nii, phase_e3.nii]
# te_ms: [4, 8, 12]
# Style B2 — single 4D phase NIfTI:
# echo_4d: phase_4d.nii.gz
# te_ms: [4, 8, 12]
# Style B3 — legacy single-echo:
phase: ph_single_echo.nii.gz
te_ms: [20]

mask: mask_single_echo.nii.gz
b0: 3.0
eroded_rad: 3
output: ./iqsm_output
```

### Direct Command-Line

DICOM folder (multi-echo phase + magnitude GRE — TEs/voxel size/B0 auto-detected):

```bash
python run.py --dicom_dir Data/your_subject_dicoms --mask BET_mask.nii
```

The folder is walked recursively; phase vs magnitude is split via `ImageType`, grouped by `EchoTime`, sorted by `ImagePositionPatient`, and saved as `dcm_converted_phase[_4d].nii.gz` and `dcm_converted_magnitude[_4d].nii.gz` inside `<output>/dicom_converted_nii/`. Pass `--te_ms` only if you want to override the headers.

Multiple 3D phase NIfTI echoes:

```bash
python run.py \
  --data_dir Data/your_subject \
  --echo_files ph1.nii ph2.nii ph3.nii \
  --te_ms 4 8 12 \
  --mag mag_4d.nii.gz \
  --mask BET_mask.nii
```

Single 4D phase NIfTI:

```bash
python run.py \
  --echo_4d phase_4d.nii.gz \
  --te_ms 4 8 12 16 20 \
  --mag mag_4d.nii.gz \
  --mask BET_mask.nii
```

Legacy single-echo (3D phase, TE in **seconds**):

```bash
python run.py --phase ph.nii.gz --te 0.020 --mask mask.nii.gz
```

MATLAB inputs:

```bash
python run.py \
  --data_dir Data/your_subject \
  --echo_files ph1.mat ph2.mat ph3.mat \
  --te_ms 4 8 12 \
  --mask BET_mask.mat
```

`.mat` inputs (v5 or v7.3) must contain a single numeric array per file.

### Arguments at a glance

- Input (mutually exclusive): `--dicom_dir`, `--echo_files`, `--echo_4d`, `--phase`.
- Echo times: `--te_ms` (preferred, milliseconds) or `--te` (seconds, legacy).
- Optional: `--mag`, `--mask` (or `--bet_mask`), `--voxel-size`, `--b0`, `--eroded-rad`, `--reverse-phase-sign`.
- Output: `--output` (default `./iqsm_output`).
- Setup: `--download-checkpoints`, `--download-demo`.
- `--data_dir` is **optional** (defaults to current working directory) — relative input paths are resolved against it.

---

## Run Demo Examples

Once you've downloaded the demo data (`python run.py --download-demo`), you can try iQSM in any of the following ways. The demo is a single-echo GRE acquisition (TE = 20 ms, 3 T, 1×1×1 mm isotropic).

### Option 1 — Web app

```bash
python app.py
```

Open the auto-launched browser tab, drop the demo files into the **NIfTI / MAT files** tab, paste `20` into Echo Times, and add `mask_single_echo.nii.gz` as Brain Mask. Hit **Run Reconstruction**.

### Option 2 — Command line

```bash
python run.py \
  --phase demo/ph_single_echo.nii.gz \
  --te_ms 20 \
  --mask demo/mask_single_echo.nii.gz
```

### Option 3 — YAML config

```bash
python run.py --config config.yaml
```

All three options produce the same outputs: `iQSM.nii.gz` (susceptibility) and `iQFM.nii.gz` (tissue field). View in [FSLeyes](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FSLeyes), [ITK-SNAP](http://www.itksnap.org/), or [3D Slicer](https://www.slicer.org/).

---

## Citation

```bibtex
@article{gao2022instant,
  title={Instant tissue field and magnetic susceptibility mapping from MRI raw phase using Laplacian enabled deep neural networks},
  journal={NeuroImage},
  year={2022},
  doi={10.1016/j.neuroimage.2022.119327}
}
```

---

[⬆ top](#iqsm--instant-quantitative-susceptibility-mapping) &nbsp;|&nbsp; [deepMRI collection](https://github.com/sunhongfu/deepMRI)
