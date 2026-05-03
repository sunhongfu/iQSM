"""
iQSM – Command-line interface

Setup (first time):
    python run.py --download-demo           # download demo NIfTIs → demo/
    python run.py --download-checkpoints    # download model weights → checkpoints/

Run:
    python run.py --config config.yaml
    python run.py --phase ph.nii.gz --te 0.020 --mask mask.nii.gz
    python run.py --config config.yaml --output ./other/   # CLI overrides config
    python run.py --help
"""

import argparse
import os
import tempfile
import urllib.request
from pathlib import Path

import yaml


def _mat_to_nii(mat_path: str) -> str:
    """Convert a .mat file to a temporary NIfTI. Returns the temp path."""
    import numpy as np
    import nibabel as nib
    import scipy.io
    mat = scipy.io.loadmat(mat_path)
    arrays = {k: v for k, v in mat.items()
              if not k.startswith("_") and isinstance(v, np.ndarray) and v.ndim >= 3}
    if not arrays:
        raise SystemExit(f"Error: no 3D+ numeric array found in {mat_path}")
    arr = max(arrays.values(), key=lambda a: a.size).squeeze().astype(np.float32)
    tmp = tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False)
    nib.save(nib.Nifti1Image(arr, np.eye(4)), tmp.name)
    return tmp.name

_HF_REPO  = "sunhongfu/iQSM"
_HF_BASE  = f"https://huggingface.co/{_HF_REPO}/resolve/main"
_HERE     = os.path.dirname(os.path.abspath(__file__))
_DEMO_DIR = os.path.join(_HERE, "demo")
_CKPT_DIR = os.path.join(_HERE, "checkpoints")

_DEMO_FILES = [
    "demo/ph_single_echo.nii.gz",
    "demo/mask_single_echo.nii.gz",
    "demo/params.json",
]
_CKPT_FILENAMES = [
    "iQSM_50_v2.pth",
    "LPLayer_chi_50_v2.pth",
    "iQFM_40_v2.pth",
    "LoTLayer_lfs_40_v2.pth",
]


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def _download_to(hf_path: str, local_name: str, dest_dir: str) -> str:
    """Download one file from HuggingFace into dest_dir, return local path."""
    url = f"{_HF_BASE}/{hf_path}"
    local = os.path.join(dest_dir, local_name)
    urllib.request.urlretrieve(url, local)
    return local


def cmd_download_demo():
    import json
    print(f"Fetching demo data from huggingface.co/{_HF_REPO} → demo/")
    os.makedirs(_DEMO_DIR, exist_ok=True)

    for hf_path in _DEMO_FILES:
        local_name = os.path.basename(hf_path)
        local = os.path.join(_DEMO_DIR, local_name)
        if os.path.exists(local):
            print(f"  {local_name} already present, skipping.")
            continue
        print(f"  {local_name} …", end=" ", flush=True)
        _download_to(hf_path, local_name, _DEMO_DIR)
        print("ok")

    with open(os.path.join(_DEMO_DIR, "params.json")) as f:
        p = json.load(f)
    te      = p["TE_seconds"]
    vox     = p["voxel_size_mm"]
    b0      = p["B0_Tesla"]
    sign    = p["phase_sign_convention"]
    eroded  = p.get("eroded_rad", 3)
    mat     = "×".join(str(x) for x in p.get("matrix_size", []))
    te_str  = str(te) if isinstance(te, (int, float)) else " ".join(f"{v:.4g}" for v in te)
    vox_str = " ".join(str(v) for v in vox)
    print(f"""
Demo dataset: {p.get("description", "")}
  Matrix:  {mat}
  Voxel:   {vox_str} mm
  TE:      {te_str} s
  B0:      {b0} T

To run reconstruction on this data:

    python run.py \\
        --phase  demo/ph_single_echo.nii.gz \\
        --mask   demo/mask_single_echo.nii.gz \\
        --te     {te_str} \\
        --b0     {b0} \\
        --voxel-size {vox_str} \\
        --eroded-rad {eroded} \\
        --reverse-phase-sign {1 if sign == 1 else 0} \\
        --output ./iqsm_demo_output/

Or copy config.yaml, fill in the paths above, and run:

    python run.py --config config.yaml
""")


def cmd_download_checkpoints():
    print(f"Fetching model checkpoints from huggingface.co/{_HF_REPO} → checkpoints/")
    os.makedirs(_CKPT_DIR, exist_ok=True)
    for name in _CKPT_FILENAMES:
        local = os.path.join(_CKPT_DIR, name)
        if os.path.exists(local):
            print(f"  {name} already present, skipping.")
            continue
        print(f"  {name} …", end=" ", flush=True)
        _download_to(name, name, _CKPT_DIR)
        print("ok")
    print("\nCheckpoints downloaded. Inference will also download them automatically on first use.\n")


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def _load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Pre-parse action flags before building the full parser.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--download-demo",        action="store_true")
    pre.add_argument("--download-checkpoints", action="store_true")
    pre.add_argument("--config", metavar="FILE")
    known, _ = pre.parse_known_args()

    if known.download_demo:
        cmd_download_demo()
        return

    if known.download_checkpoints:
        cmd_download_checkpoints()
        return

    config = _load_config(known.config) if known.config else {}

    parser = argparse.ArgumentParser(
        description="iQSM: Instant QSM reconstruction from raw MRI phase.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--download-demo",        action="store_true",
                        help="Download demo NIfTIs into demo/ and show how to run them.")
    parser.add_argument("--download-checkpoints", action="store_true",
                        help="Download model checkpoints into checkpoints/.")
    parser.add_argument("--config",     metavar="FILE",
                        help="YAML config file. CLI arguments override config values.")
    parser.add_argument("--phase",      metavar="FILE",
                        help="Wrapped phase NIfTI (.nii / .nii.gz), single-echo 3D.")
    parser.add_argument("--te",         type=float, metavar="SEC",
                        help="Echo time in seconds (e.g. 0.020).")
    parser.add_argument("--mask",       metavar="FILE", default=None,
                        help="Brain mask NIfTI (optional; ones if omitted).")
    parser.add_argument("--output",     metavar="DIR",  default="./iqsm_output",
                        help="Output directory.")
    parser.add_argument("--b0",         type=float, default=3.0,
                        help="B0 field strength in Tesla.")
    parser.add_argument("--voxel-size", nargs=3, type=float, metavar=("X", "Y", "Z"),
                        default=None,
                        help="Voxel size in mm. Reads from NIfTI header if omitted.")
    parser.add_argument("--eroded-rad", type=int, default=3, metavar="N",
                        help="Mask erosion radius in voxels.")
    parser.add_argument("--reverse-phase-sign", type=int, choices=[0, 1], default=0,
                        help="Reverse the phase sign: 0 = no (default), 1 = yes. "
                             "Set to 1 if iron-rich deep grey matter appears dark (rather than bright) in the QSM output.")
    parser.set_defaults(**config)
    args = parser.parse_args()

    if not args.phase:
        parser.error("--phase is required (or set 'phase' in config.yaml).")
    if args.te is None:
        parser.error("--te is required (or set 'te' in config.yaml).")
    if args.te <= 0:
        parser.error("--te must be positive (value in seconds, e.g. 0.020).")

    if Path(args.phase).suffix.lower() == ".mat":
        print(f"Converting phase .mat → NIfTI …")
        args.phase = _mat_to_nii(args.phase)
        if args.voxel_size is None:
            args.voxel_size = [1.0, 1.0, 1.0]
            print("Note: no --voxel-size given for .mat input — defaulting to 1×1×1 mm")
    if args.mask and Path(args.mask).suffix.lower() == ".mat":
        print(f"Converting mask .mat → NIfTI …")
        args.mask = _mat_to_nii(args.mask)

    from inference import run_iqsm, CheckpointNotFoundError
    try:
        qsm_path, lfs_path = run_iqsm(
            phase_nii_path=args.phase,
            te=args.te,
            mask_nii_path=args.mask,
            voxel_size=args.voxel_size,
            b0=args.b0,
            eroded_rad=args.eroded_rad,
            phase_sign=(1 if args.reverse_phase_sign else -1),
            output_dir=args.output,
        )
    except CheckpointNotFoundError as exc:
        print(f"\nError: {exc}\n", flush=True)
        raise SystemExit(1)

    print(f"\nOutputs:")
    print(f"  QSM (susceptibility): {qsm_path}")
    print(f"  LFS (tissue field):   {lfs_path}")


if __name__ == "__main__":
    main()
