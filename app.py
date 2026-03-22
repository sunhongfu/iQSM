"""
iQSM – Gradio Web Interface

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
_DEMO_BASE = "https://github.com/sunhongfu/iQSM/releases/download/v1.0-demo"
_DEMO_PHASE = f"{_DEMO_BASE}/ph_single_echo.nii.gz"
_DEMO_MASK  = f"{_DEMO_BASE}/mask_single_echo.nii.gz"
_DEMO_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "demo")

_DEMO_TE         = 0.020
_DEMO_B0         = 3.0
_DEMO_VOX        = "1 1 1"
_DEMO_ERODED_RAD = 3
_DEMO_PHASE_SIGN = True   # negate_phase checkbox (True = phase_sign +1)


def _download_demo() -> tuple[str, str]:
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


def load_demo_data(progress=gr.Progress(track_tqdm=True)):
    """Download demo files and populate all input fields. Does not run reconstruction."""
    progress(0.0, desc="Downloading demo data …")
    try:
        phase_path, mask_path = _download_demo()
    except gr.Error:
        raise
    except Exception as exc:
        raise gr.Error(str(exc))

    demo_info = (
        f"Cached at: {_DEMO_CACHE_DIR}\n"
        f"  ph_single_echo.nii.gz   (phase)\n"
        f"  mask_single_echo.nii.gz (mask)\n"
        f"Parameters: 1×1×1 mm · TE = 20 ms · B0 = 3 T\n"
        f"Ready — click ▶ Run Reconstruction to proceed."
    )

    return (
        phase_path, mask_path,
        str(_DEMO_TE), _DEMO_VOX,
        _DEMO_B0, _DEMO_ERODED_RAD,
        _DEMO_PHASE_SIGN,
        gr.update(value=demo_info, visible=True),
    )


# ---------------------------------------------------------------------------
# Metadata extraction from uploaded NIfTI
# ---------------------------------------------------------------------------

def extract_nii_metadata(file_obj):
    if file_obj is None:
        return gr.update(), gr.update(), gr.update()
    try:
        img = nib.load(file_obj.name)
        zooms = img.header.get_zooms()
        voxel_str = f"{zooms[0]:.4g} {zooms[1]:.4g} {zooms[2]:.4g}"

        te_update = gr.update()
        try:
            descrip = img.header.get("descrip", b"")
            if isinstance(descrip, (bytes, bytearray)):
                descrip = descrip.decode("utf-8", errors="ignore")
            m = re.search(r"TE\s*=\s*([\d.]+)\s*(ms)?", descrip.strip(), re.IGNORECASE)
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


_DISPLAY_VMIN = -0.2   # ppm  (QSM)
_DISPLAY_VMAX =  0.2   # ppm
_LFS_VMIN     = -0.05  # ppm  (LFS tissue field)
_LFS_VMAX     =  0.05  # ppm


def _make_slice_figure(nii_path: str, vmin: float, vmax: float):
    """Render axial/coronal/sagittal middle slices; return PNG file paths."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    vol = nib.load(nii_path).get_fdata(dtype=np.float32)

    raw_slices = [
        vol[:, :, vol.shape[2] // 2].T,
        vol[:, vol.shape[1] // 2, :].T,
        vol[vol.shape[0] // 2, :, :].T,
    ]

    out_dir = tempfile.mkdtemp(prefix="iqsm_preview_")
    paths = []
    for i, sl in enumerate(raw_slices):
        fig, ax = plt.subplots(figsize=(4, 4), dpi=120)
        ax.imshow(sl, cmap="gray", origin="lower", aspect="equal", vmin=vmin, vmax=vmax)
        ax.axis("off")
        fig.patch.set_facecolor("#111827")
        ax.set_facecolor("#111827")
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        path = os.path.join(out_dir, f"slice_{i}.png")
        fig.savefig(path, dpi=120, bbox_inches="tight", pad_inches=0,
                    facecolor="#111827")
        plt.close(fig)
        paths.append(path)

    return paths[0], paths[1], paths[2]


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
        raise gr.Error("Echo time must be positive (enter seconds, e.g. 0.020).")

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
            "Reconstruction failed — check the log for details.\n\n"
            + traceback.format_exc()
        )

    try:
        ax_qsm, cor_qsm, sag_qsm = _make_slice_figure(qsm_path, _DISPLAY_VMIN, _DISPLAY_VMAX)
    except Exception:
        ax_qsm = cor_qsm = sag_qsm = None

    try:
        ax_lfs, cor_lfs, sag_lfs = _make_slice_figure(lfs_path, _LFS_VMIN, _LFS_VMAX)
    except Exception:
        ax_lfs = cor_lfs = sag_lfs = None

    status = "✅ Done — download QSM and LFS files below."
    return status, qsm_path, lfs_path, ax_qsm, cor_qsm, sag_qsm, ax_lfs, cor_lfs, sag_lfs


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

_CSS = """
/* ── Typography ──────────────────────────────────────────────── */
.gradio-container {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Inter",
                 Roboto, "Helvetica Neue", Arial, sans-serif !important;
    max-width: 1280px !important;
    margin: 0 auto !important;
}

/* ── App header ──────────────────────────────────────────────── */
.app-header {
    background: linear-gradient(135deg, #0c2340 0%, #1a4f8a 55%, #2471b5 100%);
    border-radius: 10px;
    padding: 22px 28px;
    margin-bottom: 4px;
}
.app-header h1 {
    color: #ffffff !important;
    font-size: 1.45rem !important;
    font-weight: 700 !important;
    margin: 0 0 5px 0 !important;
    letter-spacing: -0.02em !important;
    line-height: 1.2 !important;
}
.app-header p {
    color: rgba(255,255,255,0.72) !important;
    font-size: 0.875rem !important;
    margin: 0 !important;
    line-height: 1.55 !important;
}
.app-header a { color: #93c5fd !important; text-decoration: none; }
.app-header a:hover { text-decoration: underline !important; }

/* ── Section labels ──────────────────────────────────────────── */
.sec-label {
    font-size: 0.68rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: #64748b !important;
    margin: 0 0 6px 0 !important;
    padding-bottom: 6px !important;
    border-bottom: 1px solid #e2e8f0 !important;
    display: block !important;
}

/* ── Action buttons ──────────────────────────────────────────── */
#run-btn > button {
    font-size: 0.975rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.01em !important;
    height: 52px !important;
    border-radius: 8px !important;
}
#demo-btn > button {
    height: 52px !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
}

/* ── Demo info box ────────────────────────────────────────────── */
#demo-info textarea {
    background: #f0f9ff !important;
    border-color: #bae6fd !important;
    font-size: 0.82rem !important;
    font-family: ui-monospace, "Cascadia Code", "Fira Code", monospace !important;
    color: #0c4a6e !important;
}

/* ── Status box ──────────────────────────────────────────────── */
#status-box textarea {
    font-size: 0.875rem !important;
    color: #1e293b !important;
}

/* ── Preview images ──────────────────────────────────────────── */
#preview-row .image-container { border-radius: 6px !important; overflow: hidden !important; }

/* ── Footer ──────────────────────────────────────────────────── */
.app-footer {
    font-size: 0.775rem !important;
    color: #94a3b8 !important;
    text-align: center !important;
    padding: 12px 0 2px 0 !important;
    border-top: 1px solid #e2e8f0 !important;
    margin-top: 4px !important;
    line-height: 1.6 !important;
}
.app-footer a { color: #64748b !important; text-decoration: none; }
.app-footer a:hover { text-decoration: underline !important; }
"""

_THEME = gr.themes.Default(
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
    font_mono=["ui-monospace", "SFMono-Regular", "Menlo", "monospace"],
    primary_hue=gr.themes.colors.blue,
    neutral_hue=gr.themes.colors.slate,
)

TITLE = "iQSM — Quantitative Susceptibility Mapping"


def build_ui():
    with gr.Blocks(title=TITLE) as demo:

        # ── Header ──────────────────────────────────────────────────────
        gr.HTML("""
        <div class="app-header">
          <h1>iQSM &mdash; Quantitative Susceptibility Mapping</h1>
          <p>
            Deep learning QSM reconstruction from single-echo MRI phase
            (<a href="https://doi.org/10.1002/mrm.28578">Sun et al., MRM 2021</a>).
            Upload your phase NIfTI, verify parameters, and reconstruct.
            New here? Click <strong style="color:#bfdbfe">⬇ Load Demo Data</strong> to prefill all fields, then click Run.
          </p>
        </div>
        """)

        with gr.Row(equal_height=False):

            # ── Left column: Inputs ──────────────────────────────────────
            with gr.Column(scale=5, min_width=340):

                gr.HTML('<p class="sec-label">Input</p>')

                phase_file = gr.File(
                    label="Phase NIfTI (.nii / .nii.gz)",
                    file_types=[".nii", ".gz"],
                )
                te_str = gr.Textbox(
                    label="Echo time — TE (seconds)",
                    placeholder="e.g.  0.020",
                    info="Auto-filled from NIfTI header when available. Enter value in seconds.",
                )
                negate_phase = gr.Checkbox(
                    label="Reverse phase sign",
                    value=False,
                    info="Enable if using the opposite scanner convention "
                         "(veins appear bright in the QSM output).",
                )

                mask_file = gr.File(
                    label="Brain mask NIfTI (optional — full volume used if omitted)",
                    file_types=[".nii", ".gz"],
                )
                with gr.Row():
                    b0_val = gr.Number(
                        label="B0 field strength (Tesla)",
                        value=3.0, minimum=0.1, maximum=14.0, step=0.5,
                    )
                    eroded_rad = gr.Slider(
                        label="Mask erosion (voxels)",
                        minimum=0, maximum=10, step=1, value=3,
                    )
                voxel_str = gr.Textbox(
                    label="Voxel size — x y z (mm)",
                    placeholder="e.g.  1 1 2",
                    info="Auto-filled from NIfTI header. Override if the values look wrong.",
                )

                with gr.Row():
                    run_btn  = gr.Button(
                        "▶  Run Reconstruction", variant="primary",
                        size="lg", elem_id="run-btn", scale=3,
                    )
                    demo_btn = gr.Button(
                        "⬇  Load Demo Data", variant="secondary",
                        size="lg", elem_id="demo-btn", scale=1,
                    )

                demo_info_box = gr.Textbox(
                    label="Demo dataset",
                    lines=4, interactive=False, visible=False, elem_id="demo-info",
                )

            # ── Right column: Results ────────────────────────────────────
            with gr.Column(scale=5, min_width=340):

                gr.HTML('<p class="sec-label">Results</p>')

                status_box = gr.Textbox(
                    label="Status",
                    lines=2, interactive=False,
                    placeholder="Results will appear here after reconstruction …",
                    elem_id="status-box",
                )
                with gr.Row():
                    qsm_file = gr.File(label="QSM — susceptibility map (.nii.gz)")
                    lfs_file = gr.File(label="LFS — tissue field (.nii.gz)")

                gr.HTML('<p class="sec-label" style="margin-top:14px">Preview — QSM (−0.2 to 0.2 ppm)</p>')
                with gr.Row():
                    axial_img    = gr.Image(label="Axial",    show_label=True, height=200)
                    coronal_img  = gr.Image(label="Coronal",  show_label=True, height=200)
                    sagittal_img = gr.Image(label="Sagittal", show_label=True, height=200)

                gr.HTML('<p class="sec-label" style="margin-top:10px">Preview — LFS tissue field (−0.05 to 0.05 ppm)</p>')
                with gr.Row():
                    axial_lfs    = gr.Image(label="Axial",    show_label=True, height=200)
                    coronal_lfs  = gr.Image(label="Coronal",  show_label=True, height=200)
                    sagittal_lfs = gr.Image(label="Sagittal", show_label=True, height=200)

        # ── Footer ──────────────────────────────────────────────────────
        gr.HTML("""
        <div class="app-footer">
          Sun H, et al. <em>Leveraging deep neural networks for quantitative susceptibility mapping
          via residual learning and linear fitting.</em> Magn Reson Med, 2021.
          <a href="https://doi.org/10.1002/mrm.28578">doi:10.1002/mrm.28578</a>
          &nbsp;·&nbsp;
          <a href="https://github.com/sunhongfu/iQSM">github.com/sunhongfu/iQSM</a>
        </div>
        """)

        # ── Wiring ───────────────────────────────────────────────────────
        phase_file.change(
            fn=extract_nii_metadata,
            inputs=[phase_file],
            outputs=[te_str, voxel_str, b0_val],
        )

        _run_outputs  = [status_box, qsm_file, lfs_file,
                         axial_img, coronal_img, sagittal_img,
                         axial_lfs, coronal_lfs, sagittal_lfs]
        _demo_outputs = [
            phase_file, mask_file, te_str, voxel_str, b0_val, eroded_rad, negate_phase,
            demo_info_box,
        ]

        run_btn.click(
            fn=reconstruct,
            inputs=[phase_file, te_str, mask_file, voxel_str, b0_val, eroded_rad, negate_phase],
            outputs=_run_outputs,
        )
        demo_btn.click(
            fn=load_demo_data,
            inputs=[],
            outputs=_demo_outputs,
        )

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="iQSM Gradio server")
    parser.add_argument("--share",       action="store_true", help="Create public Gradio link")
    parser.add_argument("--server-port", type=int, default=7860)
    parser.add_argument("--server-name", type=str, default="0.0.0.0")
    args = parser.parse_args()

    demo = build_ui()
    demo.launch(
        theme=_THEME,
        css=_CSS,
        share=args.share,
        server_name=args.server_name,
        server_port=args.server_port,
        show_error=True,
        allowed_paths=[tempfile.gettempdir()],
    )
