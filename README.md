# iQSM – Instant Quantitative Susceptibility Mapping

**Instant Tissue Field and Magnetic Susceptibility Mapping from MRI Raw Phase using Laplacian Enabled Deep Neural Networks**

[NeuroImage 2022](https://www.sciencedirect.com/science/article/pii/S1053811922005274) &nbsp;|&nbsp; [arXiv](https://arxiv.org/abs/2111.07665) &nbsp;|&nbsp; [HuggingFace](https://huggingface.co/sunhongfu/iQSM) &nbsp;|&nbsp; [deepMRI collection](https://github.com/sunhongfu/deepMRI)

iQSM performs single-step, end-to-end local field (iQFM) and susceptibility (QSM) reconstruction directly from raw MRI phase — no separate background field removal step needed. It uses a large-stencil Laplacian preprocessed deep neural network (LoT-Unet).

> **Tip:** For data with resolution finer than 0.7 mm isotropic, interpolate to 1 mm before reconstruction for best results.

---

## Overview

![Framework](https://www.dropbox.com/s/7bxkyu1utxux76k/Figs_1.png?raw=1)

*Fig. 1: iQFM and iQSM framework using the proposed LoT-Unet architecture.*

![Results](https://www.dropbox.com/s/9jt391q22sgber6/Figs_2.png?raw=1)

*Fig. 2: Comparison of QSM methods on ICH patients. Red arrows indicate artifacts near hemorrhage sources.*

---

## Which Setup Should I Use?

| I want to… | Best option |
|---|---|
| Just try it quickly, no coding | **Docker** (Option 1) |
| Use the web app on a shared server | **Docker** or **Conda** |
| Run from the command line / scripts | **Conda** or **pip** |
| Call from MATLAB | **MATLAB wrapper** (requires Conda or pip) |
| Use an NVIDIA GPU | **Docker** (GPU mode) or **Conda/pip** |

---

## Option 1 — Docker (Web App, Recommended)

**Best for:** Windows, macOS (including Apple Silicon), Linux. No Python setup needed.

**Requirements:** [Docker Desktop](https://docs.docker.com/get-docker/) (or Docker Engine on Linux).

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/sunhongfu/iQSM.git
cd iQSM

# 2. Download model weights (run once, on the host — not inside Docker)
python run.py --download-checkpoints

# 3. (Optional) Download demo data to try the app
python run.py --download-demo

# 4. Start the app
docker compose up
```

Open **http://localhost:7860** in your browser.

> The `demo/` and `checkpoints/` folders are bind-mounted into the container — files downloaded on the host are immediately visible inside Docker without a restart.

### Enable NVIDIA GPU (Linux only)

1. Install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).
2. Edit `docker-compose.yml`: set `TORCH_VARIANT: cu121`.
3. Uncomment the GPU block at the bottom of `docker-compose.yml`.
4. Rebuild and start:

```bash
docker compose build
docker compose up
```

---

## Option 2 — Conda (Web App + CLI)

**Best for:** Users with Anaconda or Miniconda already installed.

**Requirements:** [Anaconda](https://www.anaconda.com/download) or [Miniconda](https://docs.anaconda.com/miniconda/).

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/sunhongfu/iQSM.git
cd iQSM

# 2. Create and activate the environment
conda env create -f environment.yml
conda activate iqsm

# 3a. Launch the web app
python app.py
#     → open http://localhost:7860

# 3b. Or use the command line directly
python run.py --download-demo          # download demo data
python run.py --download-checkpoints   # download model weights
python run.py --phase ph.nii.gz --te 0.020 --mask mask.nii.gz
```

> **GPU note:** `environment.yml` installs `pytorch-cuda=12.1` by default. To install CPU-only, remove that line from `environment.yml` before running `conda env create`.

---

## Option 3 — pip (Web App + CLI)

**Best for:** Users who prefer pip, or already have a Python environment.

**Requirements:** Python 3.10+.

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/sunhongfu/iQSM.git
cd iQSM

# 2. Install PyTorch (choose one):
pip install torch                                                        # Apple Silicon or CPU
pip install torch --index-url https://download.pytorch.org/whl/cpu      # Linux/Windows CPU-only
pip install torch --index-url https://download.pytorch.org/whl/cu121    # Linux/Windows NVIDIA GPU

# 3. Install remaining dependencies
pip install -r requirements.txt

# 4a. Launch the web app
python app.py
#     → open http://localhost:7860

# 4b. Or use the command line directly
python run.py --download-demo          # download demo data
python run.py --download-checkpoints   # download model weights
python run.py --phase ph.nii.gz --te 0.020 --mask mask.nii.gz
```

---

## Option 4 — MATLAB Wrapper

**Best for:** Users already working in MATLAB who want to call iQSM directly.

**Requirements:** MATLAB R2017b+, and a working Python environment (Conda or pip — see Options 2/3 above).

> **Windows users:** Run `iQSM_fcns/ConfigurePython.m` first and update the `pyExec` variable to your Python executable path.

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/sunhongfu/iQSM.git

# 2. Set up Python environment (Conda or pip, see Options 2/3 above)
#    Model weights are downloaded automatically on first inference.
```

```matlab
% Run the demo scripts
demo_single_echo
demo_multi_echo
```

### Function signature

```matlab
QSM = iQSM(phase, TE, 'mag', mag, 'mask', mask, ...
            'voxel_size', [1,1,1], 'B0', 3, 'B0_dir', [0,0,1], ...
            'output_dir', pwd);
```

| Parameter | Required | Description |
|---|---|---|
| `phase` | ✓ | 3D (single-echo) or 4D (multi-echo) GRE phase volume |
| `TE` | ✓ | Echo time(s) in seconds — e.g. `20e-3` or `[4,8,12]*1e-3` |
| `mag` | | Magnitude volume (default: ones) |
| `mask` | | Brain mask (default: ones) |
| `voxel_size` | | Resolution in mm (default: `[1 1 1]`) |
| `B0_dir` | | B0 direction unit vector (default: `[0 0 1]` for axial) |
| `B0` | | Field strength in Tesla (default: `3`) |
| `output_dir` | | Output folder (default: current directory) |

---

## Downloading Checkpoints and Demo Data

Model weights and demo data are hosted on [Hugging Face Hub](https://huggingface.co/sunhongfu/iQSM).

```bash
# Download model weights into checkpoints/
python run.py --download-checkpoints

# Download demo data into demo/ and print the run command
python run.py --download-demo
```

> **Docker users:** Run these commands on the **host machine** before or after starting the container — not inside Docker. The folders are bind-mounted, so files appear immediately without a restart.

Files are also cached in `~/.cache/huggingface/hub/` for reuse.

---

## Web App Features

- Upload phase NIfTI (`.nii` / `.nii.gz`)
- Optionally upload a brain mask
- Click **⬇ Load Demo Data** to auto-fill all fields with the demo dataset
- Click **▶ Run Reconstruction** to generate QSM and tissue field maps
- Download output NIfTI files — view in [FSLeyes](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FSLeyes), [ITK-SNAP](http://www.itksnap.org/), or [3D Slicer](https://www.slicer.org/)

---

## Command Line Reference

```bash
# Show all options
python run.py --help

# Download demo data (prints example run command)
python run.py --download-demo

# Basic single-echo reconstruction
python run.py --phase ph.nii.gz --te 0.020 --mask mask.nii.gz

# Override output directory
python run.py --phase ph.nii.gz --te 0.020 --output ./my_output/

# Use a YAML config file
python run.py --config config.yaml
```

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
