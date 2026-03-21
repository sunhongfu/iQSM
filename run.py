"""
iQSM – Command-line interface
Usage:
    python run.py --phase ph.nii.gz --te 0.020 --mask mask.nii.gz --output ./results/
    python run.py --help
"""

import argparse
import sys

from inference import run_iqsm


def main():
    parser = argparse.ArgumentParser(
        description="iQSM: Instant QSM reconstruction from raw MRI phase.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--phase",   required=True, metavar="FILE",
                        help="Wrapped phase NIfTI (.nii / .nii.gz), single-echo 3D.")
    parser.add_argument("--te",      required=True, type=float, metavar="SEC",
                        help="Echo time in seconds (e.g. 0.020).")
    parser.add_argument("--mask",    metavar="FILE", default=None,
                        help="Brain mask NIfTI (optional; ones if omitted).")
    parser.add_argument("--output",  metavar="DIR",  default="./iqsm_output",
                        help="Output directory for QSM and LFS NIfTI files.")
    parser.add_argument("--b0",      type=float, default=3.0,
                        help="B0 field strength in Tesla.")
    parser.add_argument("--voxel-size", nargs=3, type=float, metavar=("X","Y","Z"),
                        default=None,
                        help="Voxel size in mm. Reads from NIfTI header if omitted.")
    parser.add_argument("--eroded-rad", type=int, default=3, metavar="N",
                        help="Mask erosion radius in voxels.")
    parser.add_argument("--phase-sign", type=int, choices=[-1, 1], default=-1,
                        help="Phase sign convention: -1 (default) or +1.")

    args = parser.parse_args()

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
