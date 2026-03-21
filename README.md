# iQSM – Instant Quantitative Susceptibility Mapping

**Instant Tissue Field and Magnetic Susceptibility Mapping from MRI Raw Phase using Laplacian Enabled Deep Neural Networks**

[NeuroImage 2022](https://www.sciencedirect.com/science/article/pii/S1053811922005274) &nbsp;|&nbsp; [arXiv](https://arxiv.org/abs/2111.07665) &nbsp;|&nbsp; [data & checkpoints](https://www.dropbox.com/sh/9kmbytgf3jpj7bh/AACUZJ1KlJ1AFCPMIVyRFJi5a?dl=0) &nbsp;|&nbsp; [deepMRI collection](https://github.com/sunhongfu/deepMRI)

iQSM enables single-step (end-to-end) local field and QSM reconstruction directly from raw MRI phase images, using a large-stencil Laplacian preprocessed deep neural network (LoT-Unet) — no separate background field removal needed.

> **Update (March 2025):** New user-friendly MATLAB wrappers for iQSM/iQFM/iQSM+/xQSM/xQSM+ with simpler syntax — see the [iQSM+](https://github.com/sunhongfu/iQSM_Plus) repo.

> **Tip:** For data with resolution finer than 0.7 mm isotropic, interpolate to 1 mm before reconstruction for best results.

---

## Overview

### Framework

![Whole Framework](https://www.dropbox.com/s/7bxkyu1utxux76k/Figs_1.png?raw=1)

Fig. 1: Overview of the iQFM and iQSM framework using the proposed LoT-Unet architecture, composed of a tailored Lap-Layer and a 3D residual U-net.

### Representative Results

![Representative Results](https://www.dropbox.com/s/9jt391q22sgber6/Figs_2.png?raw=1)

Fig. 2: Comparison of different QSM methods on three ICH patients. Red arrows indicate artifacts near hemorrhage sources.

---

## Requirements

- Python 3.7+, PyTorch 1.8+
- NVIDIA GPU (CUDA 10.0+)
- MATLAB R2017b+ (for MATLAB wrapper)
- FSL (for BET brain mask extraction)

Tested on: CentOS 7.8 (Tesla V100), macOS 12 / Ubuntu 19.10 (GTX 1060).

---

## Quick Start — Web App (no MATLAB needed)

The easiest way to run iQSM. Pretrained checkpoints and demo data download automatically.

### Option A: Docker (recommended — zero setup)

```bash
git clone https://github.com/sunhongfu/iQSM.git
cd iQSM
docker compose up
```

Open **http://localhost:7860** in your browser.
Click **⚡ Run demo** to try it instantly, or upload your own phase NIfTI.

> For NVIDIA GPU support, edit `docker-compose.yml` and set `TORCH_VARIANT=cu121`, then uncomment the GPU block.

### Option B: Conda

```bash
git clone https://github.com/sunhongfu/iQSM.git
cd iQSM

conda env create -f environment.yml
conda activate iqsm

python app.py
```

Open **http://localhost:7860** in your browser.

### Option C: pip

```bash
git clone https://github.com/sunhongfu/iQSM.git
cd iQSM

pip install torch          # CPU — or see requirements.txt for GPU options
pip install -r requirements.txt

python app.py
```

Open **http://localhost:7860** in your browser.

> **Checkpoints** (`iQSM_UnetPart.pth`, `iQFM_UnetPart.pth`) are downloaded automatically
> into `iQSM_fcns/` on first run (~34 MB total). They are also available manually from
> [Dropbox](https://www.dropbox.com/sh/9kmbytgf3jpj7bh/AACUZJ1KlJ1AFCPMIVyRFJi5a?dl=0).

---

## Quick Start — MATLAB Wrapper

### 1. Clone and download checkpoints

```bash
git clone https://github.com/sunhongfu/iQSM.git
```

Download checkpoints and demo data from [Dropbox](https://www.dropbox.com/sh/9kmbytgf3jpj7bh/AACUZJ1KlJ1AFCPMIVyRFJi5a?dl=0) and place `.pth` files in `iQSM_fcns/`.

### 2. Run on demo data

```matlab
% Single-echo
demo_single_echo

% Multi-echo
demo_multi_echo
```

---

## MATLAB Wrapper Usage

```matlab
QSM = iQSM(phase, TE, 'mag', mag, 'mask', mask, 'voxel_size', [1,1,1], 'B0', 3, 'B0_dir', [0,0,1]);
```

**Compulsory inputs:**
- `phase` — 3D (single-echo) or 4D (multi-echo) GRE phase volume
- `TE` — echo time(s) in seconds, e.g. `20e-3` or `[4,8,12]*1e-3`

**Optional inputs:**
- `mag` — magnitude volume (default: ones)
- `mask` — brain mask (default: ones)
- `voxel_size` — resolution in mm (default: `[1 1 1]`)
- `B0_dir` — B0 direction (default: `[0 0 1]` for axial)
- `B0` — field strength in Tesla (default: `3`)
- `output_dir` — output folder (default: current directory)

---

## Code Structure

| File | Description |
|------|-------------|
| `demo_single_echo.m` | Full pipeline on single-echo simulated data |
| `demo_multi_echo.m` | Full pipeline on multi-echo in vivo data |
| `PythonCodes/Evaluation/Inference.py` | PyTorch inference API |
| `PythonCodes/Evaluation/LoT_Unet.py` | LoT-Unet model |
| `PythonCodes/Training/` | Training scripts for iQSM and iQFM |

---

## Training

```matlab
% Prepare training data
matlab -nodisplay -r PrepareFullSizedImages
matlab -nodisplay -r cropQSMs
matlab -nodisplay -r GenerateHealthyPatches
matlab -nodisplay -r Gen_HemoCal
```

```bash
# Train networks
python PythonCodes/Training/FixedLapLayer/TrainiQSM/TrainiQSM.py
python PythonCodes/Training/FixedLapLayer/TrainiQFM/TrainiQFM.py
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
