"""
iQSM – Command-line interface
Usage:
    python run.py --demo                                        # download & run demo
    python run.py --config config.yaml                         # from config file
    python run.py --phase ph.nii.gz --te 0.020 --mask mask.nii.gz
    python run.py --config config.yaml --output ./other/       # CLI overrides config
    python run.py --help
"""

import argparse
import os
import tempfile
import urllib.request

import yaml

from inference import run_iqsm


# ---------------------------------------------------------------------------
# Demo data (mirrors app.py)
# ---------------------------------------------------------------------------
_DEMO_BASE      = "https://github.com/sunhongfu/iQSM/releases/download/v1.0-demo"
_DEMO_CACHE_DIR = os.path.join(tempfile.gettempdir(), "iqsm_demo")


def _download_demo() -> tuple[str, str]:
    os.makedirs(_DEMO_CACHE_DIR, exist_ok=True)
    files = {
        "ph_single_echo.nii.gz":   f"{_DEMO_BASE}/ph_single_echo.nii.gz",
        "mask_single_echo.nii.gz": f"{_DEMO_BASE}/mask_single_echo.nii.gz",
    }
    for name, url in files.items():
        dest = os.path.join(_DEMO_CACHE_DIR, name)
        if not os.path.exists(dest):
            print(f"Downloading {name} …")
            urllib.request.urlretrieve(url, dest)
    return (
        os.path.join(_DEMO_CACHE_DIR, "ph_single_echo.nii.gz"),
        os.path.join(_DEMO_CACHE_DIR, "mask_single_echo.nii.gz"),
    )


def run_demo(output_dir: str):
    print("── iQSM demo ──────────────────────────────────────────────────")
    print(f"  Data cached at: {_DEMO_CACHE_DIR}")
    print("  Parameters: 1×1×1 mm, TE=20 ms, B0=3T, phase_sign=+1")
    print("────────────────────────────────────────────────────────────────")
    phase_path, mask_path = _download_demo()
    qsm_path, lfs_path = run_iqsm(
        phase_nii_path=phase_path,
        te=0.020,
        mask_nii_path=mask_path,
        voxel_size=[1, 1, 1],
        b0=3.0,
        eroded_rad=3,
        phase_sign=1,
        output_dir=output_dir,
    )
    print(f"\nOutputs:")
    print(f"  QSM (susceptibility): {qsm_path}")
    print(f"  LFS (tissue field):   {lfs_path}")


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
    # Pre-parse --demo and --config before building the full parser.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--demo",   action="store_true")
    pre.add_argument("--config", metavar="FILE")
    known, _ = pre.parse_known_args()

    if known.demo:
        pre2 = argparse.ArgumentParser(add_help=False)
        pre2.add_argument("--output", default="./iqsm_demo_output")
        pre2.add_argument("--demo", action="store_true")
        a, _ = pre2.parse_known_args()
        run_demo(a.output)
        return

    config = _load_config(known.config) if known.config else {}

    parser = argparse.ArgumentParser(
        description="iQSM: Instant QSM reconstruction from raw MRI phase.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--demo",       action="store_true",
                        help="Download and run the built-in demo dataset.")
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
    parser.add_argument("--voxel-size", nargs=3, type=float, metavar=("X","Y","Z"),
                        default=None,
                        help="Voxel size in mm. Reads from NIfTI header if omitted.")
    parser.add_argument("--eroded-rad", type=int, default=3, metavar="N",
                        help="Mask erosion radius in voxels.")
    parser.add_argument("--phase-sign", type=int, choices=[-1, 1], default=-1,
                        help="Phase sign convention: -1 (default) or +1.")
    parser.set_defaults(**config)
    args = parser.parse_args()

    if not args.phase:
        parser.error("--phase is required (or use --demo, or set 'phase' in config.yaml).")
    if args.te is None:
        parser.error("--te is required (or set 'te' in config.yaml).")
    if args.te <= 0:
        parser.error("--te must be positive (value in seconds, e.g. 0.020).")

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
