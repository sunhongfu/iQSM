"""
iQSM – Gradio Web Interface
=====================================
Clinician-friendly web UI for Quantitative Susceptibility Mapping (QSM).

Launch:
    python app.py                   # CPU
    python app.py --share           # public Gradio link
    python app.py --server-port 8080

Docker:
    docker compose up               # see docker-compose.yml
"""

import argparse
import os
import re
import tempfile
import traceback
import urllib.request

import gradio as gr
import nibabel as nib
import numpy as np

from inference import run_iqsm


# ---------------------------------------------------------------------------
# Demo data – single-echo in-vivo brain, 1×1×1 mm, B0=3T, TE=20ms
# ---------------------------------------------------------------------------
_DEMO_BASE = (
    "https://github.com/sunhongfu/iQSM/releases/download/v1.0-demo"
)
_DEMO_PHASE = f"{_DEMO_BASE}/ph_single_echo.nii.gz"
_DEMO_MASK  = f"{_DEMO_BASE}/mask_single_echo.nii.gz"
_DEMO_CACHE_DIR = os.path.join(tempfile.gettempdir(), "iqsm_demo")

# Demo acquisition parameters
_DEMO_TE         = 0.020   # seconds
_DEMO_B0         = 3.0     # Tesla
_DEMO_VOX        = "1 1 1" # mm
_DEMO_ERODED_RAD = 3
_DEMO_PHASE_SIGN = False   # negate_phase checkbox (False = phase_sign -1)


def _download_demo() -> tuple[str, str]:
    """Download demo files (cached after first run). Returns (phase_path, mask_path)."""
    os.makedirs(_DEMO_CACHE_DIR, exist_ok=True)
    phase_path = os.path.join(_DEMO_CACHE_DIR, "ph_single_echo.nii.gz")
    mask_path  = os.path.join(_DEMO_CACHE_DIR, "mask_single_echo.nii.gz")

    for url, path in [(_DEMO_PHASE, phase_path), (_DEMO_MASK, mask_path)]:
        if not os.path.exists(path):
            print(f"Downloading demo file: {url}")
            try:
                urllib.request.urlretrieve(url, path)
            except Exception as exc:
                raise gr.Error(
                    f"Could not download demo data from GitHub Releases.\n{exc}\n\n"
                    "Please upload your own phase NIfTI file instead."
                )
    return phase_path, mask_path


def load_and_run_demo(progress=gr.Progress(track_tqdm=True)):
    """Download demo data, fill all form fields, and run reconstruction."""
    def _progress(frac, msg):
        progress(frac, desc=msg)

    _progress(0.0, "Downloading demo data …")
    try:
        phase_path, mask_path = _download_demo()
    except gr.Error:
        raise
    except Exception as exc:
        raise gr.Error(str(exc))

    output_dir = tempfile.mkdtemp(prefix="iqsm_demo_")
    try:
        qsm_path, lfs_path = run_iqsm(
            phase_nii_path=phase_path,
            te=_DEMO_TE,
            mask_nii_path=mask_path,
            voxel_size=[1, 1, 1],
            b0=_DEMO_B0,
            eroded_rad=_DEMO_ERODED_RAD,
            phase_sign=-1,
            output_dir=output_dir,
            progress_fn=_progress,
        )
    except Exception:
        raise gr.Error("Demo reconstruction failed.\n\n" + traceback.format_exc())

    try:
        ax_img, cor_img, sag_img = _make_slice_figure(qsm_path)
    except Exception:
        ax_img = cor_img = sag_img = None

    demo_info = (
        f"Demo data cached at: {_DEMO_CACHE_DIR}\n"
        f"  Phase:  ph_single_echo.nii.gz\n"
        f"  Mask:   mask_single_echo.nii.gz\n"
        "Parameters: 1×1×1 mm, TE=20 ms, B0=3T"
    )
    status = "✅ Demo complete! Download QSM and LFS NIfTI files below."

    # Return: input field updates + output results
    return (
        # --- input fields ---
        phase_path,                         # phase_file
        mask_path,                          # mask_file
        str(_DEMO_TE),                      # te_str
        _DEMO_VOX,                          # voxel_str
        _DEMO_B0,                           # b0_val
        _DEMO_ERODED_RAD,                   # eroded_rad
        _DEMO_PHASE_SIGN,                   # negate_phase
        gr.update(value=demo_info, visible=True),  # demo_info_box
        # --- output results ---
        status, qsm_path, lfs_path, ax_img, cor_img, sag_img,
    )


# ---------------------------------------------------------------------------
# Metadata extraction from uploaded NIfTI
# ---------------------------------------------------------------------------

def extract_nii_metadata(file_obj):
    """
    Called when a phase NIfTI is uploaded.
    Returns updates for: te_str, voxel_str, b0_val
    Auto-fills voxel size from the header; attempts to parse TE from descrip.
    """
    if file_obj is None:
        return gr.update(), gr.update(), gr.update()
    try:
        img = nib.load(file_obj.name)
        zooms = img.header.get_zooms()
        voxel_str = f"{zooms[0]:.4g} {zooms[1]:.4g} {zooms[2]:.4g}"

        # Try to extract TE from the NIfTI descrip field (best-effort)
        te_update = gr.update()
        try:
            descrip = img.header.get("descrip", b"")
            if isinstance(descrip, (bytes, bytearray)):
                descrip = descrip.decode("utf-8", errors="ignore")
            descrip = descrip.strip()
            # Match "TE=20", "TE=0.020", "TE=20ms", "TE=20 ms" (case-insensitive)
            m = re.search(r"TE\s*=\s*([\d.]+)\s*(ms)?", descrip, re.IGNORECASE)
            if m:
                te_val = float(m.group(1))
                if m.group(2) and m.group(2).lower() == "ms":
                    te_val /= 1000.0
                te_update = gr.update(value=str(te_val))
        except Exception:
            pass

        return te_update, gr.update(value=voxel_str), gr.update()
    except Exception:
        return gr.update(), gr.update(), gr.update()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_floats(text: str, name: str, n: int | None = None) -> list[float]:
    try:
        vals = [float(v) for v in text.replace(",", " ").split()]
    except ValueError:
        raise gr.Error(f"'{name}' must be numbers separated by spaces or commas.")
    if n is not None and len(vals) != n:
        raise gr.Error(f"'{name}' must have exactly {n} values, got {len(vals)}.")
    return vals


def _make_slice_figure(nii_path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    vol = nib.load(nii_path).get_fdata(dtype=np.float32)
    vmin, vmax = np.percentile(vol, [2, 98])
    vol_n = np.clip((vol - vmin) / max(vmax - vmin, 1e-6), 0, 1)

    slices = {
        "Axial":    vol_n[:, :, vol_n.shape[2] // 2].T,
        "Coronal":  vol_n[:, vol_n.shape[1] // 2, :].T,
        "Sagittal": vol_n[vol_n.shape[0] // 2, :, :].T,
    }

    imgs = []
    for title, sl in slices.items():
        fig, ax = plt.subplots(figsize=(4, 4), dpi=100)
        ax.imshow(sl, cmap="gray", origin="lower", aspect="equal")
        ax.set_title(title, fontsize=12)
        ax.axis("off")
        fig.tight_layout(pad=0.5)
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
        imgs.append(buf[:, :, :3].copy())
        plt.close(fig)

    return imgs[0], imgs[1], imgs[2]


# ---------------------------------------------------------------------------
# Core reconstruction callback
# ---------------------------------------------------------------------------

def reconstruct(
    phase_file,
    te_str,
    mask_file,
    voxel_str,
    b0_val,
    eroded_rad,
    negate_phase,
    progress=gr.Progress(track_tqdm=True),
):
    if phase_file is None:
        raise gr.Error("Please upload a phase NIfTI file.")
    if not te_str.strip():
        raise gr.Error("Please enter the echo time (TE).")

    te_vals = _parse_floats(te_str, "Echo time (TE)", n=1)
    te = te_vals[0]
    if te <= 0:
        raise gr.Error("Echo time must be positive. Enter value in seconds (e.g. 0.020).")

    voxel_size = None
    if voxel_str.strip():
        voxel_size = _parse_floats(voxel_str, "Voxel size", n=3)
        if any(v <= 0 for v in voxel_size):
            raise gr.Error("Voxel sizes must be positive.")

    phase_sign = 1 if negate_phase else -1
    output_dir = tempfile.mkdtemp(prefix="iqsm_out_")

    def _progress(frac, msg):
        progress(frac, desc=msg)

    try:
        qsm_path, lfs_path = run_iqsm(
            phase_nii_path=phase_file if isinstance(phase_file, str) else phase_file.name,
            te=te,
            mask_nii_path=(mask_file if isinstance(mask_file, str) else mask_file.name) if mask_file else None,
            voxel_size=voxel_size,
            b0=float(b0_val),
            eroded_rad=int(eroded_rad),
            phase_sign=phase_sign,
            output_dir=output_dir,
            progress_fn=_progress,
        )
    except Exception:
        raise gr.Error(
            "Reconstruction failed. Check the log for details.\n\n"
            + traceback.format_exc()
        )

    try:
        ax_img, cor_img, sag_img = _make_slice_figure(qsm_path)
    except Exception:
        ax_img = cor_img = sag_img = None

    status = "✅ Reconstruction complete! Download QSM and LFS NIfTI files below."
    return status, qsm_path, lfs_path, ax_img, cor_img, sag_img


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

TITLE = "iQSM – QSM Reconstruction"
DESCRIPTION = """
**Quantitative Susceptibility Mapping (QSM)** from MRI phase data
using the *iQSM* deep learning model ([paper](https://doi.org/10.1002/mrm.28578)).

**Quick-start:** Upload your phase NIfTI — voxel size is filled automatically.
Enter TE in seconds and click **Run Reconstruction**.
Or click **⚡ Run demo** to try it instantly on a built-in brain dataset.
"""


def build_ui():
    with gr.Blocks(title=TITLE) as demo:
        gr.Markdown(f"# {TITLE}")
        gr.Markdown(DESCRIPTION)

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Input")
                phase_file = gr.File(
                    label="Phase NIfTI (.nii / .nii.gz)",
                    file_types=[".nii", ".gz"],
                )

                gr.Markdown("### Echo time")
                te_str = gr.Textbox(
                    label="TE (seconds)",
                    placeholder="e.g.  0.020   — auto-filled if found in NIfTI header",
                )

                negate_phase = gr.Checkbox(
                    label="Reverse phase sign (opposite scanner convention)",
                    value=False,
                )

                gr.Markdown("### Optional inputs")
                mask_file = gr.File(
                    label="Brain mask NIfTI (optional)",
                    file_types=[".nii", ".gz"],
                )

                gr.Markdown("### Acquisition parameters")
                with gr.Row():
                    b0_val = gr.Number(
                        label="B0 field strength (Tesla)",
                        value=3.0,
                        minimum=0.1,
                        maximum=14.0,
                        step=0.5,
                    )
                    eroded_rad = gr.Slider(
                        label="Mask erosion radius (voxels)",
                        minimum=0,
                        maximum=10,
                        step=1,
                        value=3,
                    )

                voxel_str = gr.Textbox(
                    label="Voxel size (mm)  — auto-filled from NIfTI header",
                    placeholder="e.g.  1 1 2",
                )

                with gr.Row():
                    run_btn  = gr.Button("▶ Run Reconstruction", variant="primary", size="lg")
                    demo_btn = gr.Button("⚡ Run demo", variant="secondary", size="lg")

                demo_info_box = gr.Textbox(
                    label="Demo data info",
                    lines=4,
                    interactive=False,
                    visible=False,
                )

            with gr.Column(scale=1):
                gr.Markdown("### Results")
                status_box = gr.Textbox(
                    label="Status",
                    lines=2,
                    interactive=False,
                    placeholder="Reconstruction output will appear here …",
                )
                qsm_file = gr.File(label="⬇ Download QSM NIfTI (susceptibility)")
                lfs_file = gr.File(label="⬇ Download LFS NIfTI (tissue field)")

                gr.Markdown("#### Preview (QSM, middle slice)")
                with gr.Row():
                    axial_img    = gr.Image(label="Axial",    show_label=True)
                    coronal_img  = gr.Image(label="Coronal",  show_label=True)
                    sagittal_img = gr.Image(label="Sagittal", show_label=True)

        # Auto-fill voxel size (and TE if in header) when NIfTI is uploaded
        phase_file.change(
            fn=extract_nii_metadata,
            inputs=[phase_file],
            outputs=[te_str, voxel_str, b0_val],
        )

        _run_outputs  = [status_box, qsm_file, lfs_file, axial_img, coronal_img, sagittal_img]
        _demo_outputs = [
            phase_file, mask_file, te_str, voxel_str, b0_val, eroded_rad, negate_phase,
            demo_info_box,
        ] + _run_outputs

        run_btn.click(
            fn=reconstruct,
            inputs=[phase_file, te_str, mask_file, voxel_str, b0_val, eroded_rad, negate_phase],
            outputs=_run_outputs,
        )

        demo_btn.click(
            fn=load_and_run_demo,
            inputs=[],
            outputs=_demo_outputs,
        )

        gr.Markdown(
            "---\n"
            "**Citation:** Sun H, et al. *Leveraging deep neural networks for quantitative susceptibility mapping via "
            "residual learning and linear fitting.* Magnetic Resonance in Medicine, 2021. "
            "[doi:10.1002/mrm.28578](https://doi.org/10.1002/mrm.28578)\n\n"
            "**Source code:** [github.com/sunhongfu/iQSM](https://github.com/sunhongfu/iQSM)"
        )

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="iQSM Gradio server")
    parser.add_argument("--share", action="store_true", help="Create public Gradio link")
    parser.add_argument("--server-port", type=int, default=7860)
    parser.add_argument("--server-name", type=str, default="0.0.0.0")
    args = parser.parse_args()

    demo = build_ui()
    demo.launch(
        theme=gr.themes.Soft(),
        share=args.share,
        server_name=args.server_name,
        server_port=args.server_port,
        show_error=True,
        allowed_paths=[tempfile.gettempdir()],
    )
