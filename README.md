# iQSM – Instant Quantitative Susceptibility Mapping

**Instant Tissue Field and Magnetic Susceptibility Mapping from MRI Raw Phase using Laplacian Enabled Deep Neural Networks**

[NeuroImage 2022](https://www.sciencedirect.com/science/article/pii/S1053811922005274) &nbsp;|&nbsp; [arXiv](https://arxiv.org/abs/2111.07665) &nbsp;|&nbsp; [HuggingFace](https://huggingface.co/sunhongfu/iQSM) &nbsp;|&nbsp; [deepMRI collection](https://github.com/sunhongfu/deepMRI)

iQSM performs single-step, end-to-end local field (iQFM) and susceptibility (QSM) reconstruction directly from raw MRI phase — **no separate background-field-removal step required**. It uses a large-stencil Laplacian-preprocessed deep neural network (LoT-Unet).

> **Tip:** for data with resolution finer than 0.7 mm isotropic, interpolate to 1 mm before reconstruction for best results.

## Highlights

- **Single-step QSM from raw phase** — phase unwrapping and background field removal happen inside the network.
- **NIfTI / MAT input** — phase + (optional) magnitude as 3D-per-echo files or a single 4D volume; `.nii`, `.nii.gz`, `.mat` v5/v7.3.
- **Multi-echo magnitude × TE² weighted combination** — automatic TE²-only fallback when no magnitude is provided.
- **Echo Selection / Recombine** — after a multi-echo run, exclude noisy short-TE echoes and recombine in seconds (no re-inference).
- **Browser-based UI** — collapsible sections, live progress log, slice slider, orientation-preview panels (last-echo phase / magnitude / mask), shape verification, dark-mode auto-open with port auto-fallback.
- **Standalone DICOM helper** — `dicom_to_nifti.py` converts raw GRE DICOMs (phase + magnitude **or** real + imaginary, with optional GE slice-direction chopper correction) into the NIfTI files the web app and CLI consume.
- **Optional iQFM** — toggleable tissue-field output (`iQFM.nii.gz`); requires a brain mask.

## Layout

| File / folder | Purpose |
|---|---|
| `app.py` | Gradio web app for browser-based inference. |
| `run.py` | Command-line driver (per-echo files, 4D NIfTI, converted folder, or YAML config). |
| `dicom_to_nifti.py` | Standalone DICOM → NIfTI converter (run once, before `app.py` / `run.py`). |
| `inference.py` | Pure-Python iQSM / iQFM pipeline. |
| `data_utils.py` | NIfTI / MAT loaders, shape utilities, DICOM splitter (used by `dicom_to_nifti.py`). |
| `models/` | LoT-Unet architecture. |
| `config.yaml` | Example YAML config for `run.py --config`. |

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

A fresh virtual environment isolates iQSM's dependencies and avoids version conflicts. You need **Python 3.10 or 3.11**.

```bash
python --version    # check
```

If Python is missing, install from [python.org](https://www.python.org/downloads/). On Windows, tick **Add Python to PATH** during installation.

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

You should see `(venv)` in your prompt. Re-activate each new terminal.

**Install PyTorch.** Go to [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/), pick your OS / CUDA version, and run the command it gives you. Examples:

```bash
# NVIDIA GPU (CUDA 12.4):
pip install torch --index-url https://download.pytorch.org/whl/cu124

# CPU only (works without GPU, slower):
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

**Install the rest.** Pick one — depending on whether you want the browser UI:

- **Web app + CLI** (recommended):
  ```bash
  pip install -r requirements-webapp.txt
  ```

- **CLI only** (lighter, no Gradio / FastAPI / Pydantic / Uvicorn):
  ```bash
  pip install -r requirements.txt
  ```

---

### 3. Download checkpoints (and optionally demo data)

Checkpoints and demo data are excluded from git and hosted on Hugging Face: [sunhongfu/iQSM](https://huggingface.co/sunhongfu/iQSM/tree/main).

**Checkpoints** (required, one-time):
```bash
python run.py --download-checkpoints
```

**Demo data** (optional, see [Run Demo Examples](#run-demo-examples)):
```bash
python run.py --download-demo
```

**Manual download** (if behind a firewall) — grab the files from Hugging Face and place them as:

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

If you're starting from raw DICOMs, do the [DICOM → NIfTI conversion step](#dicom--nifti-conversion) first. Then choose the [Web App](#web-app) (recommended) or the [Command-Line Interface](#command-line-interface).

---

## DICOM → NIfTI conversion

The web app and CLI consume **NIfTI** files, not raw DICOMs (uploading thousands of small DICOMs through a browser is brittle, and the local conversion is one short command). `dicom_to_nifti.py` walks one or more folders, splits modalities by the DICOM `ImageType` tag (with fall-backs to `ComplexImageComponent` and the GE private tag `(0043, 102f)`), groups slices by `EchoTime`, and emits ready-to-use NIfTIs plus a `params.json` you can copy values out of.

It accepts **either** of the modality combinations a scanner may export:

- **phase (P/PHASE) + magnitude (M/MAGNITUDE)** — used directly, or
- **real (R/REAL) + imaginary (I/IMAGINARY)** — phase and magnitude are derived from the complex signal:
  ```
  phase     = angle(R + 1j·I)
  magnitude = |R + 1j·I|
  ```

When both pairs are present in the same folder, **real + imaginary is preferred**.

### Combined folder (any/all modalities mixed together)

```bash
python dicom_to_nifti.py --dicom_dir /path/to/dicoms --out_dir ./converted
```

### Separate folders, phase + magnitude

```bash
python dicom_to_nifti.py \
  --phase_dir /path/to/phase_dicoms \
  --mag_dir   /path/to/magnitude_dicoms \
  --out_dir   ./converted
```

### Separate folders, real + imaginary (phase + magnitude derived)

```bash
python dicom_to_nifti.py \
  --real_dir /path/to/real_dicoms \
  --imag_dir /path/to/imaginary_dicoms \
  --out_dir  ./converted
```

### GE slice-direction chopper (real + imaginary only)

GE 3D-GRE recon inserts an alternating ±1 along the slice direction in image space (a missing `fftshift` in their pipeline). Magnitude is invariant under this, but phase derived from raw real + imag shows π flips on every other slice — looking like massive wrapping. The script automatically applies the `(-1)^z` chopper when `Manufacturer = GE MEDICAL SYSTEMS`. Override:

```bash
python dicom_to_nifti.py --dicom_dir /path/to/dicoms --chopper on   # always apply
python dicom_to_nifti.py --dicom_dir /path/to/dicoms --chopper off  # never apply
```

`--chopper auto` (default) only fires for GE. `--chopper` has no effect when phase + magnitude DICOMs are used directly.

### After conversion

You'll see a copy-paste-friendly summary:

```text
─── Acquisition values ───
  Echo Times (ms)  : 4.92, 9.84, 14.76, 19.68, 24.6
  Voxel size (mm)  : 1 1 2
  B0 (Tesla)       : 3.0
──────────────────────────
```

The output folder will contain (names depend on echo count):

```text
converted/
├── dcm_converted_phase[_4d].nii.gz
├── dcm_converted_magnitude[_4d].nii.gz
└── params.json
```

`python dicom_to_nifti.py --help` lists all flags. The output folder feeds directly into:

- the **web app** (drop the NIfTIs into the upload buttons; copy values from `params.json` into the form), or
- the **CLI** (`run.py --from_converted ./converted` reads `params.json` automatically — no retyping).

---

## Web App

```bash
python app.py
```

The app picks port `7860` by default; if it's busy it falls back automatically (`7861`, `7862`, …). Your default browser opens once the server is ready.

### Usage walk-through

#### 1. Phase + Magnitude Input

Two side-by-side upload buttons:

- **Add Phase NIfTI / MAT** (left, required)
- **Add Magnitude NIfTI / MAT (optional)** (right)

Each accepts one 4D file or multiple 3D files (one per echo). Supported: `.nii`, `.nii.gz`, `.mat` (v5 or v7.3).

**Phase is required.** **Magnitude is optional** — used for magnitude × TE² weighting in multi-echo combination; without it, multi-echo falls back to TE²-only weighting (uniform magnitude).

Have raw DICOMs? See [DICOM → NIfTI conversion](#dicom--nifti-conversion).

#### 2. Processing Order

Two parallel lists (Phase left, Magnitude right) showing every uploaded file in natural numeric order (`mag1`, `mag2`, …, `mag10`). Each list shows a **shape summary** so mismatches are obvious before you run, plus an explicit "✕ Remove all …" button per modality. When both modalities are supplied, the two columns must have the same echo count.

#### 3. Echo Times (ms)

A single textbox accepts two equivalent formats:

- Comma-separated values (any spacing) — `2.4, 3.6, 9.2, 20.8`
- Compact `first_TE : spacing : count` — `4.5 : 5.0 : 5` expands to `4.5, 9.5, 14.5, 19.5, 24.5`

Voxel size is auto-filled from the first NIfTI's header on upload (overridable below).

#### 4. Brain Mask *(optional but recommended)*

This section is **open by default**. A brain mask:
- improves **iQSM reconstruction quality** (background voxels excluded), and
- enables **iQFM** — the local tissue field map (the background-field-removal result).

Default mask erosion is 3 voxels; adjust under **Acquisition & Hyper-parameters** below if you'd rather keep more cortical brain region.

⚠️ **Make sure the mask is oriented and aligned to the phase / magnitude volumes.** After the run finishes you can confirm in the **Visualisation** panel — the brain-mask preview shares the same slice slider as the phase / magnitude previews.

#### 5. Acquisition & Hyper-parameters *(collapsed by default)*

| Field | Notes |
|---|---|
| Voxel size (mm) | Overrides NIfTI header. Auto-filled on upload. |
| B0 (Tesla) | Defaults to 3.0. |
| Mask erosion radius (voxels) | Disabled (and 0) when no mask is provided; defaults to 3 once a mask is supplied. |
| Reverse phase sign | Enable if iron-rich deep grey matter appears dark (rather than bright) in the QSM output. |
| Run iQFM (tissue field) | Opt-in, mask-required. Produces an additional `iQFM.nii.gz`. |

#### 6. Run Reconstruction

Click the green **Run Reconstruction** button. Below it you'll see, in order:

- **Log** — streaming console output, including a *RUN CONFIGURATION* block that prints the equivalent CLI invocation so you can reproduce the run from a terminal.
- **Visualisation** — middle-slice grayscale preview with a **Z-slice slider**. Display windows for QSM (± 0.2 ppm) and LFS (± 0.05 ppm, only when iQFM ran) are editable. Below those, three **orientation-preview** panels show the last-echo raw phase, last-echo raw magnitude, and brain mask (auto-windowed) sharing the same slice slider so you can verify alignment.
- **Echo Selection (refine combination)** — visible after every multi-echo run; disabled (greyed out) for single-echo runs. Uncheck the echoes you want to **exclude** (early echoes with short TEs may produce artifacts) and click **🔁 Recombine selected echoes**. The per-echo files above are reused, so this is fast (no re-inference). The recombined files use a versioned name (`iQSM_recombined_e2_e3_e4.nii.gz`) so the original all-echoes combination stays available.
- **Results** — every produced file listed for download. Click a file size to download a single file, or click **📦 Download all (ZIP)** at the bottom for the whole bundle. Each Echo-Selection recombine refreshes the bundle.

GPU memory is released between runs, so you can upload a new dataset and re-run without restarting the page.

The web app accepts files up to **5 GB** (`max_file_size="5gb"` on launch).

#### Running over SSH

The launch script detects `SSH_CONNECTION` and skips the auto-open browser step (which would only try to launch a browser on the remote box). It prints the URL and a port-forward hint instead:

```text
Running over SSH — auto-open skipped.
Open this URL in your local browser:
  http://127.0.0.1:7860/
If the host isn't reachable from your laptop, forward the port:
  ssh -L 7860:127.0.0.1:7860 <user>@<host>
```

---

## Command-Line Interface

`run.py` can be driven via flags, a YAML config, or the converted-folder shortcut. It always validates inputs and prints an *Equivalent command-line invocation* line so a web-app run can be reproduced verbatim from a terminal.

### One-step from a converted DICOM folder

After [DICOM conversion](#dicom--nifti-conversion):

```bash
python run.py --from_converted ./converted --mask BET_mask.nii
```

`--from_converted` reads `phase.nii.gz`, `magnitude.nii.gz`, and `params.json` (TEs, voxel size, B0) automatically — no retyping.

### Multiple 3D phase NIfTI echoes

```bash
python run.py \
  --data_dir Data/your_subject \
  --echo_files ph1.nii ph2.nii ph3.nii \
  --te_ms 4 8 12 \
  --mag mag_4d.nii.gz \
  --mask BET_mask.nii
```

### Single 4D phase NIfTI

```bash
python run.py \
  --echo_4d phase_4d.nii.gz \
  --te_ms 4 8 12 16 20 \
  --mag mag_4d.nii.gz \
  --mask BET_mask.nii
```

### Single-echo (3D phase, TE in **seconds**)

```bash
python run.py --phase ph.nii.gz --te 0.020 --mag mag.nii.gz --mask mask.nii.gz
```

### MATLAB inputs

```bash
python run.py \
  --data_dir Data/your_subject \
  --echo_files ph1.mat ph2.mat ph3.mat \
  --te_ms 4 8 12 \
  --mag mag.mat \
  --mask BET_mask.mat
```

`.mat` files (v5 or v7.3) must contain a single numeric array per file.

### YAML config

```bash
python run.py --config config.yaml
```

Example `config.yaml`:

```yaml
data_dir: demo
phase: ph_single_echo.nii.gz
te_ms: [20]
mag: mag.nii.gz
mask: mask_single_echo.nii.gz
b0: 3.0
eroded_rad: 3
output: ./iqsm_output
```

### Arguments at a glance

- **Input** (mutually exclusive): `--from_converted`, `--echo_files`, `--echo_4d`, `--phase`.
- **Echo times**: `--te_ms` (preferred, milliseconds) or `--te` (seconds, legacy).
- **Required**: a phase-input flag (above).
- **Optional**: `--mag` (omit → TE²-only weighting in multi-echo), `--mask` (or `--bet_mask`), `--voxel-size`, `--b0`, `--eroded-rad`, `--reverse-phase-sign`, `--no-iqfm`.
- **Output**: `--output` (default `./iqsm_output`).
- **Setup**: `--download-checkpoints`, `--download-demo`.
- `--data_dir` is **optional** (defaults to current working directory) — relative input paths are resolved against it.

`python run.py --help` lists everything.

---

## Run Demo Examples

After `python run.py --download-demo`, try iQSM in any of the following ways. The bundled demo is a single-echo GRE acquisition (TE = 20 ms, 3 T, 1×1×1 mm isotropic).

### Option 1 — Web app

```bash
python app.py
```

Drop the demo files into the upload buttons, paste `20` into Echo Times, add `mask_single_echo.nii.gz` as Brain Mask. Hit **Run Reconstruction**.

### Option 2 — Command line

```bash
python run.py \
  --phase demo/ph_single_echo.nii.gz \
  --te_ms 20 \
  --mag demo/mag_single_echo.nii.gz \
  --mask demo/mask_single_echo.nii.gz
```

### Option 3 — YAML config

```bash
python run.py --config config.yaml
```

All three options produce the same output: `iQSM.nii.gz` (susceptibility), and `iQFM.nii.gz` (tissue field) if iQFM was enabled. View in [FSLeyes](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FSLeyes), [ITK-SNAP](http://www.itksnap.org/), or [3D Slicer](https://www.slicer.org/).

---

## Troubleshooting

- **Web app upload fails with "Connection errored out / Load failed"** — usually a stale browser tab from before the server was restarted. Close all tabs from earlier sessions and open a fresh one. If still failing on a real run with very large files, increase `max_file_size` in `app.launch(...)` (currently `"5gb"`).
- **SSH "channel N: open failed: connect failed: Connection refused"** — comes from your local SSH client, not Gradio. Browser is hitting the forwarded port before Gradio has bound it (race during startup), or a stale tab is polling. After `* Running on local URL` prints, refresh once. To silence the noise add `-q -o LogLevel=ERROR` to your `ssh` command.
- **Phase from real + imaginary looks "wrapped" every other slice** — that's the GE FFT-shift quirk. Re-run conversion with `--chopper on`.
- **iron-rich deep grey matter appears dark in the QSM output** — flip the phase sign convention with `--reverse-phase-sign 1` (CLI) or the *Reverse phase sign* checkbox (web app).
- **Checkpoint download fails behind a firewall** — manually grab the four `.pth` files from [Hugging Face](https://huggingface.co/sunhongfu/iQSM/tree/main) and drop them in `checkpoints/`.

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
