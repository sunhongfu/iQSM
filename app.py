"""
iQSM – Gradio Web Interface

Launch:
    python app.py                   # CPU
    python app.py --server-port 8080

Docker:
    docker compose up               # see docker-compose.yml
"""

import argparse
import os
import re
import tempfile
import traceback

import gradio as gr
import nibabel as nib
import numpy as np

from inference import run_iqsm, CheckpointNotFoundError


# ---------------------------------------------------------------------------
# Demo data – single-echo in-vivo brain, 1×1×1 mm, B0=3T, TE=20ms
# ---------------------------------------------------------------------------
_HERE     = os.path.dirname(os.path.abspath(__file__))
_DEMO_DIR = os.path.join(_HERE, "demo")


_DEMO_FILES_HF = [
    "ph_single_echo.nii.gz",
    "mask_single_echo.nii.gz",
    "params.json",
]
_DEMO_HF_BASE = "https://huggingface.co/sunhongfu/iQSM/resolve/main/demo"

def _demo_not_found_html() -> str:
    s = '<div style="color:#dc2626;font-size:0.875rem;line-height:1.6">'
    s += '<p style="font-weight:700;margin:0 0 6px">⚠ Demo data not found in <code>demo/</code></p>'
    s += '<p style="margin:0 0 4px"><strong>Option A — Python (run on the host, not inside Docker):</strong></p>'
    s += '<pre style="background:#fef2f2;padding:6px 10px;border-radius:4px;margin:0 0 10px;font-size:0.8rem">python run.py --download-demo</pre>'
    s += '<p style="margin:0 0 4px"><strong>Option B — Manual download (no Python needed):</strong></p>'
    s += '<p style="margin:0 0 4px">Download all three files and place them in the <code>demo/</code> folder:</p>'
    s += '<ul style="margin:0 0 10px;padding-left:18px">'
    for f in _DEMO_FILES_HF:
        s += f'<li><a href="{_DEMO_HF_BASE}/{f}?download=true" target="_blank" style="color:#dc2626">{f}</a></li>'
    s += '</ul>'
    s += '<p style="margin:0">Then click <strong>⬇ Load Demo Data</strong> again — no Docker restart needed.</p>'
    s += '</div>'
    return s


def _find_demo_nii(name: str) -> str:
    """Return path to name (or name without .gz if macOS auto-extracted it)."""
    p = os.path.join(_DEMO_DIR, name)
    if os.path.exists(p):
        return p
    if name.endswith(".gz"):
        p2 = os.path.join(_DEMO_DIR, name[:-3])
        if os.path.exists(p2):
            return p2
    raise FileNotFoundError(name)


def _load_demo_files() -> tuple[str, str, dict]:
    """Load demo NIfTIs + params.json from local demo/ folder."""
    import json
    if not os.path.exists(os.path.join(_DEMO_DIR, "params.json")):
        raise FileNotFoundError()
    phase_path = _find_demo_nii("ph_single_echo.nii.gz")
    mask_path  = _find_demo_nii("mask_single_echo.nii.gz")
    with open(os.path.join(_DEMO_DIR, "params.json")) as f:
        params = json.load(f)
    return phase_path, mask_path, params


def load_demo_data():
    """Load demo files and populate all input fields. Does not run reconstruction."""
    _no_change = (gr.update(),) * 8  # phase, mask, te, vox, b0, eroded, negate, demo_info
    try:
        phase_path, mask_path, params = _load_demo_files()
    except FileNotFoundError:
        return (*_no_change, _demo_not_found_html())
    except Exception as exc:
        err = f'<p style="color:#dc2626;font-size:0.875rem">⚠ Error loading demo: {exc}</p>'
        return (*_no_change, err)

    try:
        te       = params["TE_seconds"]
        te_str   = str(te) if isinstance(te, (int, float)) else ", ".join(f"{v:.4g}" for v in te)
        vox      = params["voxel_size_mm"]
        vox_str  = " ".join(f"{v:.4g}" for v in vox)
        b0       = params["B0_Tesla"]
        eroded   = params.get("eroded_rad", 3)
        negate   = params["phase_sign_convention"] == 1
        mat      = params.get("matrix_size", "")
        mat_str  = "×".join(str(x) for x in mat) if mat else ""
    except Exception as exc:
        err = f'<p style="color:#dc2626;font-size:0.875rem">⚠ Error reading demo params: {exc}</p>'
        return (*_no_change, err)

    demo_info = (
        f"demo/ph_single_echo.nii.gz   (phase)\n"
        f"demo/mask_single_echo.nii.gz (mask)\n"
        f"Matrix: {mat_str} · Voxel: {vox_str} mm · TE: {te_str} s · B0: {b0} T\n"
        f"Ready — click ▶ Run Reconstruction to proceed."
    )

    return (
        phase_path, mask_path,
        te_str, vox_str,
        b0, eroded, negate,
        gr.update(value=demo_info, visible=True),
        "",
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


def _status_html(msg: str, ok: bool = True) -> str:
    import html as _html
    color = "#16a34a" if ok else "#dc2626"
    if ok:
        return f'<p style="color:{color};font-weight:600;margin:0">{_html.escape(msg)}</p>'
    return f'<pre style="color:{color};font-weight:600;margin:0;white-space:pre-wrap;font-family:inherit;font-size:0.875rem">{_html.escape(msg)}</pre>'


_HF_BASE = "https://huggingface.co/sunhongfu/iQSM/resolve/main"
_CKPT_FILES = [
    "iQSM_50_v2.pth",
    "LPLayer_chi_50_v2.pth",
    "iQFM_40_v2.pth",
    "LoTLayer_lfs_40_v2.pth",
]

def _ckpt_not_found_html() -> str:
    s = '<div style="color:#dc2626;font-size:0.875rem;line-height:1.6">'
    s += '<p style="font-weight:700;margin:0 0 6px">⚠ Model weights not found in <code>checkpoints/</code></p>'
    s += '<p style="margin:0 0 4px"><strong>Option A — Python (run on the host, not inside Docker):</strong></p>'
    s += '<pre style="background:#fef2f2;padding:6px 10px;border-radius:4px;margin:0 0 10px;font-size:0.8rem">python run.py --download-checkpoints</pre>'
    s += '<p style="margin:0 0 4px"><strong>Option B — Manual download (no Python needed):</strong></p>'
    s += '<p style="margin:0 0 4px">Download all four files and place them in the <code>checkpoints/</code> folder:</p>'
    s += '<ul style="margin:0 0 10px;padding-left:18px">'
    for f in _CKPT_FILES:
        s += f'<li><a href="{_HF_BASE}/{f}?download=true" target="_blank" style="color:#dc2626">{f}</a></li>'
    s += '</ul>'
    s += '<p style="margin:0">Then click <strong>▶ Run Reconstruction</strong> again — no Docker restart needed.</p>'
    s += '</div>'
    return s


_DISPLAY_VMIN = -0.2   # ppm  (QSM)
_DISPLAY_VMAX =  0.2   # ppm
_LFS_VMIN     = -0.05  # ppm  (LFS tissue field)
_LFS_VMAX     =  0.05  # ppm


def _make_slice_figure(nii_path: str, vmin: float, vmax: float) -> np.ndarray:
    """Render axial/coronal/sagittal middle slices as one combined figure; return numpy array."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    vol = nib.load(nii_path).get_fdata(dtype=np.float32)
    slices = [
        (vol[:, :, vol.shape[2] // 2].T, "Axial"),
        (vol[:, vol.shape[1] // 2, :].T, "Coronal"),
        (vol[vol.shape[0] // 2, :, :].T, "Sagittal"),
    ]

    bg = "#111827"
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.patch.set_facecolor(bg)
    for ax, (sl, label) in zip(axes, slices):
        ax.imshow(sl, cmap="gray", origin="lower", aspect="equal", vmin=vmin, vmax=vmax)
        ax.set_title(label, color="white", fontsize=11, pad=5)
        ax.axis("off")
        ax.set_facecolor(bg)
    plt.subplots_adjust(left=0.01, right=0.99, top=0.92, bottom=0.01, wspace=0.04)
    import io
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight", pad_inches=0.1, facecolor=bg)
    plt.close(fig)
    buf.seek(0)
    arr = plt.imread(buf)          # float32 RGBA 0-1
    return (arr[:, :, :3] * 255).astype(np.uint8)


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
    except CheckpointNotFoundError:
        return _ckpt_not_found_html(), None, None, None, None
    except Exception:
        tb = traceback.format_exc()
        print(tb, flush=True)
        return _status_html("Reconstruction failed — check the terminal / Docker log for the full error.", ok=False), None, None, None, None

    try:
        qsm_img = _make_slice_figure(qsm_path, _DISPLAY_VMIN, _DISPLAY_VMAX)
    except Exception:
        qsm_img = None

    try:
        lfs_img = _make_slice_figure(lfs_path, _LFS_VMIN, _LFS_VMAX)
    except Exception:
        lfs_img = None

    return _status_html("✅ Done — download QSM and LFS files below."), qsm_path, lfs_path, qsm_img, lfs_img


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
#status-box {
    font-size: 0.875rem !important;
    min-height: 60px !important;
    padding: 4px 0 !important;
}

/* ── Preview images ──────────────────────────────────────────── */
#preview-row .image-container { border-radius: 6px !important; }

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

/* ── Theme toggle button ─────────────────────────────────────── */
.app-header { position: relative !important; }
#theme-toggle {
    position: absolute !important;
    top: 16px !important;
    right: 16px !important;
    background: rgba(255,255,255,0.15) !important;
    color: #ffffff !important;
    border: 1px solid rgba(255,255,255,0.35) !important;
    border-radius: 6px !important;
    padding: 6px 14px !important;
    font-size: 0.8rem !important;
    font-weight: 500 !important;
    cursor: pointer !important;
    transition: background 0.2s !important;
}
#theme-toggle:hover { background: rgba(255,255,255,0.28) !important; }

/* ── Hide Gradio share button ─────────────────────────────────── */
.share-button { display: none !important; }

/* ── Preview images: click-to-fullscreen, no buttons ────────────
   Click the image to enter browser fullscreen; click again to exit. */
#qsm-preview img, #lfs-preview img { cursor: zoom-in !important; }
img:fullscreen, img:-webkit-full-screen {
    object-fit: contain !important;
    background: #000 !important;
    cursor: zoom-out !important;
    width: 100vw !important;
    height: 100vh !important;
}
#qsm-preview button, #lfs-preview button,
#qsm-preview .icon-buttons, #lfs-preview .icon-buttons { display: none !important; }
"""

_THEME = gr.themes.Default(
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
    font_mono=["ui-monospace", "SFMono-Regular", "Menlo", "monospace"],
    primary_hue=gr.themes.colors.blue,
    neutral_hue=gr.themes.colors.slate,
)

_HEAD = """<script>
(function() {
    // ── Theme toggle ─────────────────────────────────────────────
    var key = 'iqsm-theme';
    var saved = localStorage.getItem(key) || 'light';
    document.documentElement.classList.toggle('dark', saved === 'dark');
    document.addEventListener('click', function(e) {
        var t = (e.composedPath && e.composedPath()[0]) || e.target;
        if (t && t.id === 'theme-toggle') {
            var next = document.documentElement.classList.contains('dark') ? 'light' : 'dark';
            document.documentElement.classList.toggle('dark', next === 'dark');
            localStorage.setItem(key, next);
            t.textContent = next === 'dark' ? '\u2600 Light mode' : '\u263d Dark mode';
        }
    });

    // ── Preview images: click → browser fullscreen; click again → exit ──
    document.addEventListener('click', function(e) {
        if (document.fullscreenElement) {
            document.exitFullscreen();
            e.preventDefault();
            e.stopImmediatePropagation();
            return;
        }
        var img = ((e.composedPath && e.composedPath()[0]) || e.target).closest('img');
        if (!img) return;
        if (img.closest('#qsm-preview, #lfs-preview')) {
            e.preventDefault();
            e.stopImmediatePropagation();
            img.requestFullscreen().catch(console.error);
        }
    }, true);
})();
</script>"""

TITLE = "iQSM — Quantitative Susceptibility Mapping"


def build_ui():
    with gr.Blocks(title=TITLE) as demo:

        # ── Header ──────────────────────────────────────────────────────
        gr.HTML("""
        <div class="app-header">
          <button id="theme-toggle">&#x263D; Dark mode</button>
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

                status_box = gr.HTML(
                    value='<p style="color:#94a3b8;font-size:0.875rem;margin:0">Results will appear here after reconstruction…</p>',
                    elem_id="status-box",
                )
                with gr.Row():
                    qsm_file = gr.File(label="QSM — susceptibility map (.nii.gz)")
                    lfs_file = gr.File(label="LFS — tissue field (.nii.gz)")

                gr.HTML('<p class="sec-label" style="margin-top:14px">Preview — QSM (−0.2 to 0.2 ppm) &nbsp;<span style="font-weight:400;font-size:0.78rem;color:#94a3b8">click to fullscreen</span></p>')
                qsm_preview = gr.Image(
                    show_label=False, interactive=False,
                    elem_id="qsm-preview", height=220,
                )

                gr.HTML('<p class="sec-label" style="margin-top:10px">Preview — LFS tissue field (−0.05 to 0.05 ppm) &nbsp;<span style="font-weight:400;font-size:0.78rem;color:#94a3b8">click to fullscreen</span></p>')
                lfs_preview = gr.Image(
                    show_label=False, interactive=False,
                    elem_id="lfs-preview", height=220,
                )

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

        _run_outputs  = [status_box, qsm_file, lfs_file, qsm_preview, lfs_preview]
        _demo_outputs = [
            phase_file, mask_file, te_str, voxel_str, b0_val, eroded_rad, negate_phase,
            demo_info_box, status_box,
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
    parser.add_argument("--server-port", type=int, default=7860)
    parser.add_argument("--server-name", type=str, default="0.0.0.0")
    args = parser.parse_args()

    demo = build_ui()
    demo.launch(
        theme=_THEME,
        css=_CSS,
        head=_HEAD,
        server_name=args.server_name,
        server_port=args.server_port,
        show_error=True,
        allowed_paths=[tempfile.gettempdir(), _DEMO_DIR],
    )
