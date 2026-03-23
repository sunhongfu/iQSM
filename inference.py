"""
Pure Python inference pipeline for iQSM / iQFM.

Reconstructs QSM (chi) and tissue field (lfs) from raw MRI phase without MATLAB.
Uses the learnable LoT-layer architecture (iQSM v2 checkpoints).
"""

import os
import tempfile

import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
from scipy.ndimage import binary_erosion

from models.lot_unet import LoT_Unet
from models.unet import Unet
from models.unet_blocks import LoTLayer

# ---------------------------------------------------------------------------
# Checkpoint management — local checkpoints/ first, then HF Hub
# ---------------------------------------------------------------------------
_HF_REPO  = "sunhongfu/iQSM"
_HERE     = os.path.dirname(os.path.abspath(__file__))
_CKPT_DIR = os.path.join(_HERE, "checkpoints")

_CKPT_FILENAMES = [
    "iQSM_50_v2.pth",
    "LPLayer_chi_50_v2.pth",
    "iQFM_40_v2.pth",
    "LoTLayer_lfs_40_v2.pth",
]


class CheckpointNotFoundError(Exception):
    """Raised when model checkpoint files have not been downloaded yet."""


_CKPT_NOT_FOUND_MSG = """\
Model weights not found in checkpoints/.

Run this command on the host machine (outside Docker) before starting the app:

    python run.py --download-checkpoints

This downloads the weights into the checkpoints/ folder that Docker mounts.
Once done, click Run again — no restart needed.\
"""


def _ckpt(filename: str) -> str:
    """Return local path to a checkpoint, raising CheckpointNotFoundError if absent."""
    local = os.path.join(_CKPT_DIR, filename)
    if os.path.exists(local):
        return local
    raise CheckpointNotFoundError(_CKPT_NOT_FOUND_MSG)


# ---------------------------------------------------------------------------
# Laplacian convolution kernel
# ---------------------------------------------------------------------------
_CONV_OP = np.array(
    [
        [[1/13, 3/26, 1/13], [3/26, 3/13, 3/26], [1/13, 3/26, 1/13]],
        [[3/26, 3/13, 3/26], [3/13, -44/13, 3/13], [3/26, 3/13, 3/26]],
        [[1/13, 3/26, 1/13], [3/26, 3/13, 3/26], [1/13, 3/26, 1/13]],
    ],
    dtype=np.float32,
)

_model_cache: dict = {}


def _load_lot_layer(ckpt_path: str, device: torch.device) -> LoTLayer:
    conv_op = torch.from_numpy(_CONV_OP).unsqueeze(0).unsqueeze(0)
    layer = LoTLayer(conv_op)
    layer = nn.DataParallel(layer)
    layer.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    return layer.module


def get_models(device: torch.device):
    """Load (or return cached) iQSM and iQFM models."""
    key = str(device)
    if key in _model_cache:
        return _model_cache[key]

    lot_chi = _load_lot_layer(_ckpt("LPLayer_chi_50_v2.pth"), device)
    unet_chi = Unet(4, 16, 1)
    unet_chi = nn.DataParallel(unet_chi)
    unet_chi.load_state_dict(torch.load(_ckpt("iQSM_50_v2.pth"), map_location=device, weights_only=True))
    unet_chi = unet_chi.module

    lot_lfs = _load_lot_layer(_ckpt("LoTLayer_lfs_40_v2.pth"), device)
    unet_lfs = Unet(4, 16, 1)
    unet_lfs = nn.DataParallel(unet_lfs)
    unet_lfs.load_state_dict(torch.load(_ckpt("iQFM_40_v2.pth"), map_location=device, weights_only=True))
    unet_lfs = unet_lfs.module

    iqsm = LoT_Unet(lot_chi, unet_chi).to(device).eval()
    iqfm = LoT_Unet(lot_lfs, unet_lfs).to(device).eval()

    _model_cache[key] = (iqsm, iqfm)
    return iqsm, iqfm


# ---------------------------------------------------------------------------
# Preprocessing helpers
# ---------------------------------------------------------------------------

def _make_sphere(radius):
    c = np.arange(-radius, radius + 1)
    x, y, z = np.meshgrid(c, c, c, indexing='ij')
    return (x**2 + y**2 + z**2) <= radius**2


def _zero_pad(arr, multiple=16):
    pad_spec, positions = [], []
    for s in arr.shape[:3]:
        total = (multiple - s % multiple) % multiple
        before = total // 2
        pad_spec.append((before, total - before))
        positions.append((before, before + s))
    if arr.ndim == 4:
        pad_spec.append((0, 0))
    return np.pad(arr, pad_spec), positions


def _zero_remove(arr, positions):
    (x1, x2), (y1, y2), (z1, z2) = positions
    return arr[x1:x2, y1:y2, z1:z2]


# ---------------------------------------------------------------------------
# Main reconstruction entry point
# ---------------------------------------------------------------------------

def run_iqsm(
    phase_nii_path: str,
    te: float,
    *,
    mag_nii_path: str | None = None,
    mask_nii_path: str | None = None,
    voxel_size: list | None = None,
    b0: float = 3.0,
    eroded_rad: int = 3,
    phase_sign: int = -1,
    output_dir: str | None = None,
    progress_fn=None,
) -> tuple[str, str]:
    """
    Run iQSM + iQFM reconstruction in pure Python.

    Parameters
    ----------
    phase_nii_path : str  – wrapped phase NIfTI (3D single-echo)
    te : float            – echo time in seconds
    mag_nii_path : str    – magnitude NIfTI (optional, unused by iQSM)
    mask_nii_path : str   – brain mask NIfTI (optional)
    voxel_size : list     – [x,y,z] mm override (reads from header if None)
    b0 : float            – field strength in Tesla (default 3.0)
    eroded_rad : int      – mask erosion radius in voxels (default 3)
    phase_sign : int      – +1 or -1 sign convention flip (default -1)
    output_dir : str      – output directory (temp dir if None)

    Returns
    -------
    (qsm_path, lfs_path) – paths to QSM and tissue field NIfTI files
    """
    def _log(frac, msg):
        print(f"[{frac:.0%}] {msg}")
        if progress_fn:
            progress_fn(frac, msg)

    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="iqsm_")
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    _log(0.0, f"Device: {device}")

    _log(0.05, "Loading phase …")
    phase_img = nib.load(phase_nii_path)
    phase = phase_img.get_fdata(dtype=np.float32)
    affine = phase_img.affine

    phase = float(phase_sign) * phase.astype(np.float32)

    if mask_nii_path is not None:
        _log(0.10, "Loading mask …")
        mask = nib.load(mask_nii_path).get_fdata(dtype=np.float32)
    else:
        mask = np.ones(phase.shape[:3], dtype=np.float32)
        eroded_rad = 0

    if eroded_rad > 0:
        _log(0.15, f"Eroding mask (radius={eroded_rad}) …")
        mask = binary_erosion(mask > 0.5, structure=_make_sphere(eroded_rad)).astype(np.float32)

    phase_pad, positions = _zero_pad(phase)
    mask_pad, _ = _zero_pad(mask)

    _log(0.25, "Loading models …")
    iqsm, iqfm = get_models(device)

    te_t = torch.tensor([te], dtype=torch.float32).to(device)
    b0_t = torch.tensor([b0], dtype=torch.float32).to(device)
    phase_t = torch.from_numpy(phase_pad).float().unsqueeze(0).unsqueeze(0).to(device)
    mask_t  = torch.from_numpy(mask_pad).float().unsqueeze(0).unsqueeze(0).to(device)

    _log(0.35, "Running iQSM …")
    with torch.inference_mode():
        pred_chi = iqsm(phase_t, mask_t, te_t, b0_t) * mask_t
        _log(0.65, "Running iQFM …")
        pred_lfs = iqfm(phase_t, mask_t, te_t, b0_t) * mask_t

    def _to_numpy(t):
        return t.squeeze().cpu().numpy().astype(np.float32)

    chi = _zero_remove(_to_numpy(pred_chi), positions)
    lfs = _zero_remove(_to_numpy(pred_lfs), positions)

    _log(0.90, "Saving …")
    qsm_path = os.path.join(output_dir, "iQSM.nii.gz")
    lfs_path = os.path.join(output_dir, "iQFM.nii.gz")
    nib.save(nib.Nifti1Image(chi, affine), qsm_path)
    nib.save(nib.Nifti1Image(lfs, affine), lfs_path)

    _log(1.0, f"Done! Saved to {output_dir}")
    return qsm_path, lfs_path
