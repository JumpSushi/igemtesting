#!/usr/bin/env python3


import base64
import io
import os
import traceback
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import tifffile
from flask import Flask, jsonify, render_template, request
from werkzeug.utils import secure_filename

from pipeline import (
    DEFAULT_SIRNA_MAP,
    CHANNEL_LABELS,
    detect_beads,
    intensities_to_binary,
    binary_to_code,
    load_multichannel_tiff,
    load_sirna_map,
    lookup_sirna,
    measure_channel_intensities,
    load_truth_csv,
    match_beads_to_truth,
)
from generate_test_image import make_image

# flask app config
app = Flask(__name__)

class _NumpyEncoder(app.json_provider_class):
    def default(self, o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)

app.json_provider_class = _NumpyEncoder
app.json = _NumpyEncoder(app)
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024  # max upload: 200MB

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {".tif", ".tiff"}


def _safe_path(filename: str) -> Path:
    # sanitize & validate file path
    name = secure_filename(filename)
    if not name:
        raise ValueError("Invalid filename.")
    ext = Path(name).suffix.lower()
    if ext and ext not in ALLOWED_EXTENSIONS and not name.endswith(".csv"):
        raise ValueError(f"File type not allowed: {ext}")
    return (UPLOAD_DIR / name).resolve()


# pipeline wrapper

def _run_on_channels(channels: np.ndarray, sirna_map: dict, params: dict):
    # run full decode pipeline, return results
    blobs = detect_beads(
        channels,
        min_sigma=params["min_sigma"],
        max_sigma=params["max_sigma"],
        threshold=params["blob_threshold"],
        overlap=params["overlap"],
    )

    if len(blobs) == 0:
        return None, None, None, None, (
            "No beads detected. Try lowering the LoG threshold "
            "(e.g. 0.02) or adjusting Min/Max sigma."
        )

    intensities = measure_channel_intensities(
        channels, blobs, aperture_factor=params["aperture"])
    binary      = intensities_to_binary(intensities, method=params["threshold_method"])
    codes       = binary_to_code(binary)
    identities  = lookup_sirna(codes, sirna_map)

    n_ch      = channels.shape[0]
    ch_labels = CHANNEL_LABELS[:n_ch]

    rows = []
    for idx, (blob, inten, code, identity) in enumerate(
            zip(blobs, intensities, codes, identities)):
        row: dict = {
            "bead_id":   idx,
            "row_px":    round(float(blob[0]), 1),
            "col_px":    round(float(blob[1]), 1),
            "radius_px": round(float(blob[2]), 1),
            "code":      code,
            "siRNA":     identity,
        }
        for label, val in zip(ch_labels, inten):
            row[f"{label}"] = round(float(val), 5)
        rows.append(row)

    df     = pd.DataFrame(rows)
    img_b64 = _render_annotated(channels, blobs, codes, identities)
    return df, img_b64, blobs, codes, None


def _render_annotated(channels, blobs, codes, identities) -> str:
    # draw circles & labels on image
    projection = channels.max(axis=0)
    vmax = np.percentile(projection, 99.5)

    fig, ax = plt.subplots(figsize=(10, 8), facecolor="#0d0d14")
    ax.set_facecolor("#0d0d14")
    ax.imshow(projection, cmap="inferno", vmin=0, vmax=vmax, interpolation="nearest")
    ax.axis("off")

    unique_ids = sorted(set(identities))
    n          = max(2, len(unique_ids))
    try:
        cmap = matplotlib.colormaps.get_cmap("tab20").resampled(n)
    except AttributeError:
        cmap = plt.cm.get_cmap("tab20", n)
    id_color = {sid: cmap(i) for i, sid in enumerate(unique_ids)}

    for (row, col, radius), code, identity in zip(blobs, codes, identities):
        color = id_color[identity]
        ax.add_patch(mpatches.Circle(
            (col, row), radius * 1.5,
            linewidth=0.9, edgecolor=color, facecolor="none", alpha=0.9))
        ax.text(col + radius * 1.8, row, code,
                color=color, fontsize=4, va="center", fontfamily="monospace")

    handles = [mpatches.Patch(color=id_color[s], label=s) for s in unique_ids]
    ax.legend(handles=handles, loc="upper right", fontsize=5,
              ncol=max(1, len(handles) // 12),
              framealpha=0.6, facecolor="#1a1a2e",
              edgecolor="#333355", labelcolor="white")

    fig.tight_layout(pad=0.3)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def _render_channels(channels: np.ndarray, ch_labels: list) -> list:
    # render each channel as grayscale
    result = []
    for ch, label in zip(channels, ch_labels):
        vmax = np.percentile(ch, 99.5)
        fig, ax = plt.subplots(figsize=(2.4, 2.4), facecolor="#0d0d14")
        ax.set_facecolor("#0d0d14")
        ax.imshow(ch, cmap="gray", vmin=0, vmax=max(float(vmax), 1e-6),
                  interpolation="nearest")
        ax.set_title(label, color="#ccccdd", fontsize=9, pad=4)
        ax.axis("off")
        fig.tight_layout(pad=0.2)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        result.append(base64.b64encode(buf.read()).decode("ascii"))
    return result


def _parse_params(source) -> dict:
    # defaults tuned by sweep: avg F1 = 0.9916 across 5 seeds
    return {
        "min_sigma":        float(source.get("min_sigma",        1.5)),
        "max_sigma":        float(source.get("max_sigma",        6.0)),
        "blob_threshold":   float(source.get("blob_threshold",   0.03)),
        "overlap":          float(source.get("overlap",          0.7)),
        "aperture":         float(source.get("aperture",         0.6)),
        "threshold_method": str(source.get("threshold_method",   "gmm")),
    }


# routes

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/generate", methods=["POST"])
def api_generate():
    try:
        body    = request.get_json(silent=True) or {}
        n_beads = max(1, min(int(body.get("n_beads", 150)), 2000))
        noise   = float(body.get("noise",   0.03))
        seed    = int(body.get("seed",      42))

        image, truth = make_image(n_beads=n_beads, noise_level=noise, seed=seed)

        tif_path = _safe_path("test_image.tif")
        truth_path = _safe_path("test_image_truth.csv")
        tifffile.imwrite(str(tif_path), image, photometric="minisblack")
        truth.to_csv(str(truth_path), index=False)

        params = _parse_params({})
        df, img_b64, blobs, codes, err = _run_on_channels(image, DEFAULT_SIRNA_MAP, params)
        if err:
            return jsonify({"error": err}), 400

        # Compare with truth
        identities = [DEFAULT_SIRNA_MAP.get(code, "UNKNOWN") for code in codes]
        comparison = match_beads_to_truth(blobs, codes, identities, truth)

        n_ch      = image.shape[0]
        ch_labels = CHANNEL_LABELS[:n_ch]
        return jsonify({
            "beads":        df.to_dict(orient="records"),
            "image":        img_b64,
            "channels_b64": _render_channels(image, ch_labels),
            "ch_labels":    ch_labels,
            "n_beads":      len(df),
            "n_codes":      df["siRNA"].nunique(),
            "session_file": "test_image.tif",
            "comparison":   comparison,
            "has_truth":    True,
        })
    except Exception:
        return jsonify({"error": traceback.format_exc()}), 500


@app.route("/api/run", methods=["POST"])
def api_run():
    try:
        params  = _parse_params(request.form)
        ch_str  = request.form.get("channels", "0 1 2 3 4")
        indices = [int(x) for x in ch_str.split() if x.strip().isdigit()]
        if not indices:
            return jsonify({"error": "Invalid channel indices."}), 400

        # optional sirna map upload
        sirna_map = DEFAULT_SIRNA_MAP
        map_file  = request.files.get("sirna_map")
        if map_file and map_file.filename:
            map_path = _safe_path(map_file.filename)
            map_file.save(str(map_path))
            sirna_map = load_sirna_map(map_path)

        # file upload or use session file
        img_file = request.files.get("image")
        if img_file and img_file.filename:
            ext = Path(img_file.filename).suffix.lower()
            if ext not in ALLOWED_EXTENSIONS:
                return jsonify({"error": "Only .tif / .tiff files are accepted."}), 400
            tif_path = _safe_path(img_file.filename)
            img_file.save(str(tif_path))
        else:
            session_file = request.form.get("session_file", "")
            if not session_file:
                return jsonify({"error": "No image provided."}), 400
            tif_path = _safe_path(session_file)

        if not tif_path.exists():
            return jsonify({"error": f"Image not found: {tif_path.name}"}), 400

        channels           = load_multichannel_tiff(tif_path, indices)
        df, img_b64, blobs, codes, err = _run_on_channels(channels, sirna_map, params)
        if err:
            return jsonify({"error": err}), 400

        n_ch      = channels.shape[0]
        ch_labels = CHANNEL_LABELS[:n_ch]
        
        # Check if truth file exists
        comparison = None
        has_truth = False
        truth_path = UPLOAD_DIR / "test_image_truth.csv"
        if truth_path.exists():
            try:
                truth_df = load_truth_csv(truth_path)
                identities = [sirna_map.get(code, "UNKNOWN") for code in codes]
                comparison = match_beads_to_truth(blobs, codes, identities, truth_df)
                has_truth = True
            except Exception as e:
                print(f"Warning: Could not load truth data: {e}")
        
        result = {
            "beads":        df.to_dict(orient="records"),
            "image":        img_b64,
            "channels_b64": _render_channels(channels, ch_labels),
            "ch_labels":    ch_labels,
            "n_beads":      len(df),
            "n_codes":      df["siRNA"].nunique(),
            "session_file": tif_path.name,
        }
        
        if has_truth and comparison:
            result["comparison"] = comparison
            result["has_truth"] = True
        
        return jsonify(result)
    except Exception:
        return jsonify({"error": traceback.format_exc()}), 500


# praytothepythongodsandletitrun

if __name__ == "__main__":
    import webbrowser
    port = int(os.environ.get("PORT", 5050))
    webbrowser.open(f"http://localhost:{port}")
    app.run(host="127.0.0.1", port=port, debug=False)
