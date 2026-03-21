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
import tempfile
import traceback

import gradio as gr
import nibabel as nib
import numpy as np

from inference import run_iqsm


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
            phase_nii_path=phase_file.name,
            te=te,
            mask_nii_path=mask_file.name if mask_file else None,
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


TITLE = "iQSM – QSM Reconstruction"
DESCRIPTION = """
**Quantitative Susceptibility Mapping (QSM)** from MRI phase data
using the *iQSM* deep learning model ([paper](https://doi.org/10.1002/mrm.28578)).

**Quick-start:**
1. Upload your wrapped phase NIfTI file.
2. Enter the echo time in **seconds** (e.g. `0.020`).
3. Adjust parameters as needed and click **Run Reconstruction**.
4. Download the QSM (susceptibility) and LFS (tissue field) results.
"""


def build_ui():
    with gr.Blocks(title=TITLE, theme=gr.themes.Soft()) as demo:
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
                    placeholder="e.g.  0.020",
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
                    label="Voxel size override (mm, optional)",
                    placeholder="e.g.  1 1 2",
                )

                run_btn = gr.Button("▶ Run Reconstruction", variant="primary", size="lg")

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

        run_btn.click(
            fn=reconstruct,
            inputs=[phase_file, te_str, mask_file, voxel_str, b0_val, eroded_rad, negate_phase],
            outputs=[status_box, qsm_file, lfs_file, axial_img, coronal_img, sagittal_img],
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
        share=args.share,
        server_name=args.server_name,
        server_port=args.server_port,
        show_error=True,
        allowed_paths=[tempfile.gettempdir()],
    )
