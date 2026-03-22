"""
iQSM – Command-line interface

Setup (first time — pre-warms the Hugging Face cache):
    python run.py --download-demo           # fetch demo NIfTIs
    python run.py --download-checkpoints    # fetch model weights

Run:
    python run.py --config config.yaml
    python run.py --phase ph.nii.gz --te 0.020 --mask mask.nii.gz
    python run.py --config config.yaml --output ./other/   # CLI overrides config
    python run.py --help

Files are cached automatically by huggingface_hub (~/.cache/huggingface/hub/).
"""

import argparse

import yaml

_HF_REPO = "sunhongfu/iQSM"

_DEMO_FILES = [
    "demo/ph_single_echo.nii.gz",
    "demo/mask_single_echo.nii.gz",
]
_CKPT_FILES = [
    "iQSM_50_v2.pth",
    "LPLayer_chi_50_v2.pth",
    "iQFM_40_v2.pth",
    "LoTLayer_lfs_40_v2.pth",
]


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def _hf_pull(filenames: list[str]) -> dict[str, str]:
    """Download files from HF Hub (cached after first run). Returns {filename: local_path}."""
    from huggingface_hub import hf_hub_download
    paths = {}
    for filename in filenames:
        print(f"  {filename} …", end=" ", flush=True)
        path = hf_hub_download(repo_id=_HF_REPO, filename=filename)
        print(f"ok  →  {path}")
        paths[filename] = path
    return paths


def cmd_download_demo():
    print(f"Fetching demo data from huggingface.co/{_HF_REPO} …")
    paths = _hf_pull(_DEMO_FILES)
    phase = paths["demo/ph_single_echo.nii.gz"]
    mask  = paths["demo/mask_single_echo.nii.gz"]
    print(f"""
Demo dataset: single-echo in-vivo brain, 1×1×1 mm, TE=20 ms, B0=3T

To run reconstruction on this data:

    python run.py \\
        --phase  {phase} \\
        --mask   {mask} \\
        --te     0.020 \\
        --b0     3.0 \\
        --voxel-size 1 1 1 \\
        --eroded-rad 3 \\
        --phase-sign 1 \\
        --output ./iqsm_demo_output/

Or copy config.yaml, fill in the paths above, and run:

    python run.py --config config.yaml
""")


def cmd_download_checkpoints():
    print(f"Fetching model checkpoints from huggingface.co/{_HF_REPO} …")
    _hf_pull(_CKPT_FILES)
    print("\nCheckpoints cached. (Also fetched automatically on first inference.)\n")


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
                        help="Pre-warm HF cache with demo NIfTIs and show how to run them.")
    parser.add_argument("--download-checkpoints", action="store_true",
                        help="Pre-warm HF cache with model checkpoints.")
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
    parser.add_argument("--phase-sign", type=int, choices=[-1, 1], default=-1,
                        help="Phase sign convention: -1 (default) or +1.")
    parser.set_defaults(**config)
    args = parser.parse_args()

    if not args.phase:
        parser.error("--phase is required (or set 'phase' in config.yaml).")
    if args.te is None:
        parser.error("--te is required (or set 'te' in config.yaml).")
    if args.te <= 0:
        parser.error("--te must be positive (value in seconds, e.g. 0.020).")

    from inference import run_iqsm
    qsm_path, lfs_path = run_iqsm(
        phase_nii_path=args.phase,
        te=args.te,
        mask_nii_path=args.mask,
        voxel_size=args.voxel_size,
        b0=args.b0,
        eroded_rad=args.eroded_rad,
        phase_sign=args.phase_sign,
        output_dir=args.output,
    )

    print(f"\nOutputs:")
    print(f"  QSM (susceptibility): {qsm_path}")
    print(f"  LFS (tissue field):   {lfs_path}")


if __name__ == "__main__":
    main()
