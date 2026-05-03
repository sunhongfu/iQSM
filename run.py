"""
iQSM — Command-line interface

First-time setup (download checkpoints + demo data):
    python run.py --download-checkpoints
    python run.py --download-demo

Run from raw DICOMs (recommended — phase + magnitude auto-separated, TEs read
from headers):
    python run.py --dicom_dir /path/to/dicoms

Run from pre-converted NIfTI / MAT files:
    python run.py --echo_files ph1.nii ph2.nii ph3.nii --te_ms 4 8 12
    python run.py --echo_4d phase_4d.nii.gz --te_ms 4 8 12 --mag mag_4d.nii.gz
    python run.py --phase ph.nii.gz --te 0.020 --mask mask.nii.gz

YAML config (any CLI arg can be set instead in the config):
    python run.py --config config.yaml
"""

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np
import yaml
from huggingface_hub import hf_hub_download

from data_utils import (
    load_array_with_affine,
    load_dicom_qsm_folder,
)

REPO_ROOT = Path(__file__).resolve().parent

HF_REPO_ID = "sunhongfu/iQSM"
CHECKPOINT_FILES = [
    "iQSM_50_v2.pth",
    "LPLayer_chi_50_v2.pth",
    "iQFM_40_v2.pth",
    "LoTLayer_lfs_40_v2.pth",
]
DEMO_FILES = [
    "demo/ph_single_echo.nii.gz",
    "demo/mask_single_echo.nii.gz",
    "demo/params.json",
]


# ---------------------------------------------------------------------------
# HuggingFace download helpers (mirrors DeepRelaxo)
# ---------------------------------------------------------------------------

def _download_files(file_list, local_subdir):
    target_dir = REPO_ROOT / local_subdir
    target_dir.mkdir(parents=True, exist_ok=True)
    for remote_path in file_list:
        local_name = Path(remote_path).name
        destination = target_dir / local_name
        downloaded_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=remote_path,
            repo_type="model",
        )
        shutil.copyfile(downloaded_path, destination)
        print(f"Downloaded: {destination}")


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

def _resolve_path(base_dir, value):
    if value is None:
        return None
    p = Path(value)
    return p if p.is_absolute() else Path(base_dir) / p


# ---------------------------------------------------------------------------
# MAT → NIfTI staging
# ---------------------------------------------------------------------------

def _stage_input(path, work_dir, suffix=""):
    """Copy NIfTI as-is or convert MAT → NIfTI. Returns staged path."""
    src = Path(path)
    if src.suffix.lower() == ".mat":
        arr, _ = load_array_with_affine(src)
        dst = Path(work_dir) / (src.stem + suffix + ".nii.gz")
        nib.save(nib.Nifti1Image(arr.astype(np.float32), np.eye(4)), str(dst))
        print(f"Converted MAT → NIfTI: {src.name} → {dst.name}")
        return dst
    return src


# ---------------------------------------------------------------------------
# Multi-echo combiner — runs run_iqsm() per echo and averages outputs
# ---------------------------------------------------------------------------

def _run_multi_echo(phase_paths, te_values_s, mag_path, mask_path, voxel_size,
                    b0, eroded_rad, phase_sign, output_dir):
    """Run iQSM/iQFM on each echo and combine with magnitude × TE² weighting
    (or simple mean if no magnitude). Returns (qsm_path, lfs_path)."""
    from inference import run_iqsm

    work_dir = Path(output_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    # If magnitude is 4D, split into per-echo files
    mag_per_echo = [None] * len(phase_paths)
    if mag_path:
        mag_img = nib.load(str(mag_path))
        mag_data = mag_img.get_fdata(dtype=np.float32)
        if mag_data.ndim == 4:
            if mag_data.shape[3] != len(phase_paths):
                raise ValueError(
                    f"Magnitude has {mag_data.shape[3]} echoes but phase has "
                    f"{len(phase_paths)} — counts must match."
                )
            for i in range(mag_data.shape[3]):
                p = work_dir / f"mag_echo{i+1}.nii.gz"
                nib.save(nib.Nifti1Image(mag_data[:, :, :, i], mag_img.affine), str(p))
                mag_per_echo[i] = str(p)
        else:
            # 3D magnitude → reuse for every echo (simple weighting)
            mag_per_echo = [str(mag_path)] * len(phase_paths)

    qsm_volumes, lfs_volumes = [], []
    affine = None
    for i, (ppath, te_s) in enumerate(zip(phase_paths, te_values_s)):
        print(f"\n--- Echo {i+1}/{len(phase_paths)}  (TE = {te_s*1000:g} ms) ---")
        echo_out = work_dir / f"echo{i+1}_output"
        q_path, l_path = run_iqsm(
            phase_nii_path=str(ppath),
            te=float(te_s),
            mask_nii_path=str(mask_path) if mask_path else None,
            voxel_size=voxel_size,
            b0=b0,
            eroded_rad=eroded_rad,
            phase_sign=phase_sign,
            output_dir=str(echo_out),
        )
        q_img = nib.load(q_path)
        if affine is None:
            affine = q_img.affine
        qsm_volumes.append(q_img.get_fdata(dtype=np.float32))
        lfs_volumes.append(nib.load(l_path).get_fdata(dtype=np.float32))

    print(f"\nAveraging {len(qsm_volumes)} echoes …")
    qsm_stack = np.stack(qsm_volumes, axis=-1)
    lfs_stack = np.stack(lfs_volumes, axis=-1)

    if mag_path and all(m is not None for m in mag_per_echo):
        print("  Magnitude × TE² weighted averaging")
        mag_data = np.stack(
            [nib.load(m).get_fdata(dtype=np.float32) for m in mag_per_echo],
            axis=-1,
        )
        te_bc = np.array(te_values_s, dtype=np.float32).reshape(1, 1, 1, -1)
        weights = (mag_data * te_bc) ** 2
        denom = weights.sum(axis=-1, keepdims=True)
        denom[denom == 0] = 1.0
        qsm_avg = (weights * qsm_stack).sum(axis=-1) / denom.squeeze(-1)
        lfs_avg = (weights * lfs_stack).sum(axis=-1) / denom.squeeze(-1)
    else:
        print("  Simple mean (no magnitude provided)")
        qsm_avg = np.mean(qsm_stack, axis=-1)
        lfs_avg = np.mean(lfs_stack, axis=-1)

    qsm_path = str(work_dir / "iQSM.nii.gz")
    lfs_path = str(work_dir / "iQFM.nii.gz")
    nib.save(nib.Nifti1Image(qsm_avg.astype(np.float32), affine), qsm_path)
    nib.save(nib.Nifti1Image(lfs_avg.astype(np.float32), affine), lfs_path)
    return qsm_path, lfs_path


# ---------------------------------------------------------------------------
# 4D NIfTI → per-echo 3D files
# ---------------------------------------------------------------------------

def _split_4d(path, work_dir):
    img = nib.load(str(path))
    data = img.get_fdata(dtype=np.float32)
    if data.ndim != 4:
        return [path]
    out = []
    for i in range(data.shape[3]):
        p = Path(work_dir) / f"phase_echo{i+1}.nii.gz"
        nib.save(nib.Nifti1Image(data[:, :, :, i], img.affine), str(p))
        out.append(p)
    return out


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def _build_parser():
    parser = argparse.ArgumentParser(
        description="iQSM: Instant QSM reconstruction from raw MRI phase.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--config", metavar="FILE",
                        help="YAML config file. Any flag below can be set there.")

    parser.add_argument("--data_dir", metavar="DIR",
                        help="Resolve relative input paths against this folder. "
                             "Defaults to the current working directory.")

    parser.add_argument(
        "--dicom_dir", metavar="DIR",
        help="Folder of multi-echo GRE phase (and, optionally, magnitude) DICOMs. "
             "Files are walked recursively, split into phase vs. magnitude via "
             "ImageType, grouped by EchoTime, sorted by ImagePositionPatient, "
             "and saved as one NIfTI per modality in <output>/dicom_converted_nii/. "
             "TE values, voxel size and B0 are auto-detected from headers.",
    )
    parser.add_argument("--echo_files", nargs="+", metavar="FILE",
                        help="Multiple 3D phase NIfTI / MAT files (one per echo).")
    parser.add_argument("--echo_4d", metavar="FILE",
                        help="Single 4D phase NIfTI / MAT (echoes in last dim).")
    parser.add_argument("--phase", metavar="FILE",
                        help="Single 3D phase NIfTI / MAT (legacy single-echo input).")

    parser.add_argument("--te", nargs="+", type=float, metavar="SEC",
                        help="Echo time(s) in **seconds**, e.g. --te 0.020.")
    parser.add_argument("--te_ms", nargs="+", type=float, metavar="MS",
                        help="Echo time(s) in **milliseconds**, e.g. --te_ms 4 8 12.")

    parser.add_argument("--mag", metavar="FILE",
                        help="Magnitude NIfTI / MAT (3D or 4D). Used for "
                             "magnitude × TE² weighted averaging on multi-echo input.")
    parser.add_argument("--mask", metavar="FILE",
                        help="Brain mask NIfTI / MAT (optional; ones if omitted).")
    parser.add_argument("--bet_mask", metavar="FILE",
                        help="Alias for --mask.")

    parser.add_argument("--output", metavar="DIR", default="./iqsm_output",
                        help="Output directory.")
    parser.add_argument("--b0", type=float, default=3.0,
                        help="B0 field strength in Tesla (default: 3.0).")
    parser.add_argument("--voxel-size", nargs=3, type=float, metavar=("X", "Y", "Z"),
                        default=None,
                        help="Voxel size in mm. Reads from NIfTI header if omitted.")
    parser.add_argument("--eroded-rad", type=int, default=3, metavar="N",
                        help="Mask erosion radius in voxels (default: 3).")
    parser.add_argument("--reverse-phase-sign", type=int, choices=[0, 1], default=0,
                        help="0 = no (default), 1 = yes. Set to 1 if iron-rich "
                             "deep grey matter appears dark in the QSM output.")

    parser.add_argument("--download-checkpoints", action="store_true",
                        help="Download model weights from HuggingFace and exit.")
    parser.add_argument("--download-demo", action="store_true",
                        help="Download demo NIfTIs from HuggingFace and exit.")
    return parser


# ---------------------------------------------------------------------------
# Config merge
# ---------------------------------------------------------------------------

def _apply_config_defaults(args, parser):
    """Merge YAML config into args. CLI values win over config values."""
    if not args.config:
        return args, None
    config_path = Path(args.config)
    if not config_path.exists():
        parser.error(f"Config file not found: {config_path}")
    with open(config_path) as f:
        cfg = yaml.safe_load(f) or {}
    config_dir = config_path.parent
    # Only fill in fields the user didn't pass on the CLI
    user_provided = set()
    for tok in sys.argv[1:]:
        if tok.startswith("--"):
            user_provided.add(tok.lstrip("-").replace("-", "_"))
    for key, value in cfg.items():
        attr = key.replace("-", "_")
        if attr in user_provided:
            continue
        if hasattr(args, attr):
            setattr(args, attr, value)
    return args, config_dir


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = _build_parser()
    args = parser.parse_args()

    if args.download_checkpoints:
        _download_files(CHECKPOINT_FILES, "checkpoints")
        return
    if args.download_demo:
        _download_files(DEMO_FILES, "demo")
        return

    args, config_dir = _apply_config_defaults(args, parser)

    # Path resolution base: data_dir from CLI, else config_dir, else cwd
    if args.data_dir:
        data_dir = Path(args.data_dir).resolve()
    elif config_dir is not None:
        data_dir = config_dir.resolve()
    else:
        data_dir = Path.cwd()

    # Resolve "phase" / "te" alternative spellings from config
    if getattr(args, "te_ms", None) is None and args.te is None:
        # Some configs may use simply 'te'
        pass

    # ── Determine input mode ────────────────────────────────────────────
    given = sum(x is not None for x in [args.dicom_dir, args.echo_files,
                                        args.echo_4d, args.phase])
    if given == 0:
        parser.error(
            "No phase input. Use one of: --dicom_dir, --echo_files, --echo_4d, --phase "
            "(or set 'phase' / 'echoes' / 'echo_4d' / 'dicom_dir' in --config)."
        )
    if given > 1:
        parser.error(
            "Provide exactly one of: --dicom_dir, --echo_files, --echo_4d, --phase."
        )

    # Output dir
    if config_dir is not None and not Path(args.output).is_absolute():
        output_dir = (config_dir / args.output).resolve()
    else:
        output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Working dir (for converted NIfTIs and intermediate files)
    work_dir = Path(tempfile.mkdtemp(prefix="iqsm_run_"))

    # ── Resolve phase files + TE values ─────────────────────────────────
    phase_paths = []
    mag_path_resolved = None  # may be set by DICOM parsing

    if args.dicom_dir:
        dicom_path = _resolve_path(data_dir, args.dicom_dir)
        if not dicom_path.is_dir():
            parser.error(f"--dicom_dir is not a directory: {dicom_path}")
        file_list = [str(p) for p in dicom_path.rglob("*") if p.is_file()]
        if not file_list:
            parser.error(f"--dicom_dir contains no files: {dicom_path}")
        nii_out = output_dir / "dicom_converted_nii"
        print(f"Parsing DICOMs from {dicom_path}")
        print(f"Writing converted NIfTI files to {nii_out}")
        result = load_dicom_qsm_folder(file_list, nii_out)
        phase_path = result["phase_path"]
        mag_path_resolved = str(result["mag_path"]) if result["mag_path"] else None
        # Split if 4D
        phase_paths = _split_4d(phase_path, work_dir) if len(result["te_values_s"]) > 1 \
            else [phase_path]
        te_values_s = list(result["te_values_s"])
        # Auto-fill voxel size and B0 from DICOM if user didn't specify
        if args.voxel_size is None and result["voxel_size"]:
            args.voxel_size = result["voxel_size"]
            print(f"Voxel size from DICOM: {args.voxel_size}")
        if args.b0 == 3.0 and result["b0"] is not None:
            args.b0 = float(result["b0"])
            print(f"B0 from DICOM: {args.b0} T")
        # User-supplied TEs take precedence
        if args.te_ms is not None:
            if len(args.te_ms) != len(te_values_s):
                parser.error(
                    f"--te_ms count ({len(args.te_ms)}) doesn't match the number "
                    f"of parsed echoes ({len(te_values_s)})."
                )
            te_values_s = [t / 1000.0 for t in args.te_ms]
            print(f"Using user-supplied TEs (ms): {args.te_ms}")
        elif args.te is not None:
            if len(args.te) != len(te_values_s):
                parser.error(
                    f"--te count ({len(args.te)}) doesn't match the number "
                    f"of parsed echoes ({len(te_values_s)})."
                )
            te_values_s = list(args.te)
            print(f"Using user-supplied TEs (s): {args.te}")

    elif args.echo_4d:
        if args.te_ms is None and args.te is None:
            parser.error("when using --echo_4d, provide --te_ms or --te")
        src = _resolve_path(data_dir, args.echo_4d)
        staged = _stage_input(src, work_dir)
        phase_paths = _split_4d(staged, work_dir)
        if len(phase_paths) == 1:
            # Was actually 3D — single echo
            te_values_s = ([args.te_ms[0] / 1000.0] if args.te_ms else [args.te[0]])
        else:
            te_values_s = ([t / 1000.0 for t in args.te_ms] if args.te_ms
                           else list(args.te))
            if len(te_values_s) != len(phase_paths):
                parser.error(
                    f"Number of TEs ({len(te_values_s)}) does not match "
                    f"echoes in --echo_4d ({len(phase_paths)})."
                )

    elif args.echo_files:
        if args.te_ms is None and args.te is None:
            parser.error("when using --echo_files, provide --te_ms or --te")
        te_values_s = ([t / 1000.0 for t in args.te_ms] if args.te_ms
                       else list(args.te))
        if len(args.echo_files) != len(te_values_s):
            parser.error("--echo_files and TE counts must match.")
        for f in args.echo_files:
            src = _resolve_path(data_dir, f)
            phase_paths.append(_stage_input(src, work_dir))

    else:  # legacy --phase
        if args.te_ms is None and args.te is None:
            parser.error("when using --phase, provide --te_ms or --te")
        src = _resolve_path(data_dir, args.phase)
        staged = _stage_input(src, work_dir)
        # Could be 3D or 4D — split if 4D
        try:
            shape = nib.load(str(staged)).shape
        except Exception:
            shape = (None,)
        if len(shape) == 4:
            phase_paths = _split_4d(staged, work_dir)
            te_values_s = ([t / 1000.0 for t in args.te_ms] if args.te_ms
                           else list(args.te))
            if len(phase_paths) != len(te_values_s):
                parser.error(
                    f"Number of TEs ({len(te_values_s)}) does not match "
                    f"echoes in --phase 4D file ({len(phase_paths)})."
                )
        else:
            phase_paths = [staged]
            if args.te_ms:
                if len(args.te_ms) != 1:
                    parser.error("Single 3D phase requires exactly one TE.")
                te_values_s = [args.te_ms[0] / 1000.0]
            else:
                if len(args.te) != 1:
                    parser.error("Single 3D phase requires exactly one TE.")
                te_values_s = [args.te[0]]

    # ── Stage magnitude / mask ──────────────────────────────────────────
    mag_arg = args.mag or mag_path_resolved
    if mag_arg:
        mag_src = _resolve_path(data_dir, mag_arg)
        mag_path = _stage_input(mag_src, work_dir, suffix="_mag")
    else:
        mag_path = None

    mask_value = args.mask or args.bet_mask
    if mask_value:
        mask_src = _resolve_path(data_dir, mask_value)
        mask_path = _stage_input(mask_src, work_dir, suffix="_mask")
    else:
        mask_path = None

    # Default voxel size for MAT-only input
    if args.voxel_size is None and all(Path(p).suffix.lower() == ".mat"
                                       for p in [args.echo_4d or "",
                                                 *(args.echo_files or [])] if p):
        args.voxel_size = [1.0, 1.0, 1.0]
        print("Note: no --voxel-size given for MAT input — defaulting to 1×1×1 mm")

    if any(t <= 0 for t in te_values_s):
        parser.error("All echo times must be positive.")

    phase_sign = 1 if args.reverse_phase_sign else -1

    # ── Print summary ───────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("iQSM RUN CONFIGURATION")
    print("=" * 60)
    if len(phase_paths) == 1:
        print(f"Input mode      : Single 3D phase")
        print(f"Phase file      : {Path(phase_paths[0]).name}")
    else:
        print(f"Input mode      : Multi-echo  ({len(phase_paths)} echoes)")
        for i, (p, te_s) in enumerate(zip(phase_paths, te_values_s), 1):
            print(f"  Echo {i}: {Path(p).name}    TE = {te_s*1000:g} ms")
    print(f"TE (s)          : {', '.join(f'{t:g}' for t in te_values_s)}")
    print(f"Magnitude       : {Path(mag_path).name if mag_path else '(none)'}")
    print(f"Brain mask      : {Path(mask_path).name if mask_path else '(none)'}")
    if args.voxel_size:
        print(f"Voxel size (mm) : {' '.join(f'{v:.4g}' for v in args.voxel_size)}")
    else:
        print(f"Voxel size (mm) : (from NIfTI header)")
    print(f"B0 (T)          : {args.b0}")
    print(f"Mask erosion    : {args.eroded_rad} voxels")
    print(f"Reverse phase   : {'yes' if phase_sign == 1 else 'no'}")
    print(f"Output dir      : {output_dir}")
    print("=" * 60)
    print()

    # ── Run reconstruction ──────────────────────────────────────────────
    from inference import run_iqsm, CheckpointNotFoundError
    try:
        if len(phase_paths) == 1:
            qsm_path, lfs_path = run_iqsm(
                phase_nii_path=str(phase_paths[0]),
                te=float(te_values_s[0]),
                mag_nii_path=str(mag_path) if mag_path else None,
                mask_nii_path=str(mask_path) if mask_path else None,
                voxel_size=args.voxel_size,
                b0=args.b0,
                eroded_rad=args.eroded_rad,
                phase_sign=phase_sign,
                output_dir=str(output_dir),
            )
        else:
            qsm_path, lfs_path = _run_multi_echo(
                phase_paths=phase_paths,
                te_values_s=te_values_s,
                mag_path=mag_path,
                mask_path=mask_path,
                voxel_size=args.voxel_size,
                b0=args.b0,
                eroded_rad=args.eroded_rad,
                phase_sign=phase_sign,
                output_dir=output_dir,
            )
    except CheckpointNotFoundError as exc:
        print(f"\nError: {exc}\n", flush=True)
        raise SystemExit(1)

    print()
    print(f"Outputs:")
    print(f"  QSM (susceptibility): {qsm_path}")
    print(f"  LFS (tissue field):   {lfs_path}")


if __name__ == "__main__":
    main()
