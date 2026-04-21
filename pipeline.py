#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
import matplotlib
matplotlib.use("Agg")         
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from skimage.feature import blob_log
from skimage.filters import threshold_otsu
from skimage.draw import disk as draw_disk

# 21 knock-bead codes (TODO: get actual gene names from paper, i'm lazy will do someday)
DEFAULT_SIRNA_MAP: dict[str, str] = {
    # single
    "10000": "siRNA-A",
    "01000": "siRNA-B",
    "00100": "siRNA-C",
    "00010": "siRNA-D",
    "00001": "siRNA-E",
    # pairs
    "11000": "siRNA-AB",
    "10100": "siRNA-AC",
    "10010": "siRNA-AD",
    "10001": "siRNA-AE",
    "01100": "siRNA-BC",
    "01010": "siRNA-BD",
    "01001": "siRNA-BE",
    "00110": "siRNA-CD",
    "00101": "siRNA-CE",
    "00011": "siRNA-DE",
    # triplets
    "11100": "siRNA-ABC",
    "01110": "siRNA-BCD",
    "00111": "siRNA-CDE",
    "10110": "siRNA-ACD",
    "10011": "siRNA-ADE",
    # all 5
    "11111": "siRNA-ABCDE",
}

CHANNEL_LABELS = ["QD547", "QD572", "QD604", "QD641", "AF647"]


# image loading

def _normalise(arr: np.ndarray) -> np.ndarray:
    # normalize to 0-1
    arr = arr.astype(np.float32)
    lo, hi = arr.min(), arr.max()
    if hi > lo:
        arr = (arr - lo) / (hi - lo)
    return arr


def load_multichannel_tiff(path: Path, channel_indices: list[int]) -> np.ndarray:
    # load TIFF, handle 2D/3D/4D, return (C, H, W)
    raw = tifffile.imread(str(path))
    raw = _normalise(raw)

    ndim = raw.ndim

    # Collapse leading Z or T axis
    if ndim == 4:
        raw = raw.max(axis=0)      # max-project; shape → (C, H, W) or (H, W, C)
        ndim = 3

    if ndim == 2:
        print(f"  [warn] Single-channel image; replicating across {len(channel_indices)} channels.")
        return np.stack([raw] * len(channel_indices))

    if ndim == 3:
        # guess which axis is channels
        if raw.shape[0] <= 16 and raw.shape[0] < raw.shape[2]:
            img_chw = raw                       # (C, H, W)
        else:
            img_chw = np.moveaxis(raw, -1, 0)  # (H, W, C) → (C, H, W)

        n_available = img_chw.shape[0]
        bad = [i for i in channel_indices if i >= n_available]
        if bad:
            raise ValueError(
                f"Requested channel indices {bad} but image only has "
                f"{n_available} channels (0-indexed)."
            )
        return img_chw[channel_indices]

    raise ValueError(f"Unsupported image shape: {raw.shape}")


def load_channel_files(paths: list[Path]) -> np.ndarray:
    # load separate channel files and stack
    channels = []
    for p in paths:
        raw = tifffile.imread(str(p))
        raw = _normalise(raw)
        if raw.ndim == 3:
            raw = raw.max(axis=0)  # flatten z-stack
        if raw.ndim != 2:
            raise ValueError(f"Expected 2-D image from {p}, got shape {raw.shape}")
        channels.append(raw)
    return np.stack(channels)


# bead detection (LoG)

def detect_beads(
    channels: np.ndarray,
    min_sigma: float = 2.0,
    max_sigma: float = 8.0,
    threshold: float = 0.05,
    overlap: float = 0.5,
) -> np.ndarray:
    # find beads, returns (N, 3) array
    projection = channels.sum(axis=0)
    lo, hi = projection.min(), projection.max()
    if hi > lo:
        projection = (projection - lo) / (hi - lo)

    blobs = blob_log(
        projection,
        min_sigma=min_sigma,
        max_sigma=max_sigma,
        num_sigma=12,
        threshold=threshold,
        overlap=overlap,
    )

    if blobs.size == 0:
        return np.empty((0, 3), dtype=np.float32)

    blobs[:, 2] *= np.sqrt(2)   # sigma → approximate radius
    return blobs.astype(np.float32)


# measure intensities per channel

def measure_channel_intensities(
    channels: np.ndarray,
    blobs: np.ndarray,
    aperture_factor: float = 1.0,
) -> np.ndarray:
    # aperture + background subtraction, returns (N, C)
    n_beads = len(blobs)
    n_channels, h, w = channels.shape
    intensities = np.zeros((n_beads, n_channels), dtype=np.float32)

    for i, (row, col, radius) in enumerate(blobs):
        row_i, col_i = int(round(row)), int(round(col))
        r_inner = max(1, int(round(radius * aperture_factor)))
        r_outer = r_inner + max(2, int(round(radius * 0.6)))

        rr_fg, cc_fg = draw_disk((row_i, col_i), r_inner, shape=(h, w))

        bg_mask = np.zeros((h, w), dtype=bool)
        rr_bg, cc_bg = draw_disk((row_i, col_i), r_outer, shape=(h, w))
        bg_mask[rr_bg, cc_bg] = True
        bg_mask[rr_fg, cc_fg] = False

        for c in range(n_channels):
            ch = channels[c]
            fg_mean = ch[rr_fg, cc_fg].mean() if len(rr_fg) > 0 else 0.0
            bg_mean = ch[bg_mask].mean() if bg_mask.any() else 0.0
            intensities[i, c] = fg_mean - bg_mean

    return intensities


# binarize intensities

def intensities_to_binary(
    intensities: np.ndarray,
    method: str = "otsu",
) -> np.ndarray:
    # threshold: otsu, percentile, or gmm
    n_beads, n_channels = intensities.shape
    binary = np.zeros((n_beads, n_channels), dtype=np.uint8)

    for c in range(n_channels):
        col = intensities[:, c]

        if method == "otsu":
            thresh = threshold_otsu(col) if col.max() > col.min() else col.mean()

        elif method == "percentile":
            thresh = np.percentile(col, 50)

        elif method == "gmm":
            try:
                from sklearn.mixture import GaussianMixture
            except ImportError:
                raise ImportError(
                    "scikit-learn is required for --threshold-method gmm. "
                    "Install it with: pip install scikit-learn"
                )
            gm = GaussianMixture(n_components=2, random_state=42)
            gm.fit(col.reshape(-1, 1))
            thresh = gm.means_.flatten().mean()

        else:
            raise ValueError(f"Unknown threshold method: {method!r}")

        binary[:, c] = (col > thresh).astype(np.uint8)

    return binary


# binary to code string

def binary_to_code(binary: np.ndarray) -> list[str]:
    # convert to 5-bit string
    return ["".join(str(b) for b in row) for row in binary]


def lookup_sirna(codes: list[str], sirna_map: dict[str, str]) -> list[str]:
    return [sirna_map.get(code, "UNKNOWN") for code in codes]


# compare detected beads to truth data

def load_truth_csv(csv_path: Path) -> pd.DataFrame:
    # load truth data with expected columns: bead_id, row_px, col_px, code, siRNA
    df = pd.read_csv(csv_path)
    required = {"row_px", "col_px", "code", "siRNA"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Truth CSV missing columns: {missing}")
    return df


def match_beads_to_truth(
    detected_blobs: np.ndarray,
    detected_codes: list[str],
    detected_identities: list[str],
    truth_df: pd.DataFrame,
    max_distance: float = 10.0,
) -> dict:
    """
    Match detected beads to truth beads by spatial proximity.
    Returns detailed comparison metrics and per-bead match information.
    """
    n_detected = len(detected_blobs)
    n_truth = len(truth_df)
    
    # Initialize results
    matches = []
    matched_truth_indices = set()
    
    # For each detected bead, find closest truth bead
    for det_idx, (blob, det_code, det_identity) in enumerate(
        zip(detected_blobs, detected_codes, detected_identities)
    ):
        det_row, det_col = blob[0], blob[1]
        
        # Compute distance to all truth beads
        truth_rows = truth_df["row_px"].values
        truth_cols = truth_df["col_px"].values
        distances = np.sqrt((truth_rows - det_row)**2 + (truth_cols - det_col)**2)
        
        min_dist_idx = np.argmin(distances)
        min_dist = distances[min_dist_idx]
        
        # Check if match is within threshold and not already matched
        is_matched = min_dist <= max_distance and min_dist_idx not in matched_truth_indices
        
        if is_matched:
            matched_truth_indices.add(min_dist_idx)
            truth_row = truth_df.iloc[min_dist_idx]
            truth_code = str(truth_row["code"])
            truth_identity = str(truth_row["siRNA"])
            
            matches.append({
                "detected_idx": det_idx,
                "truth_idx": min_dist_idx,
                "det_row": float(det_row),
                "det_col": float(det_col),
                "truth_row": float(truth_row["row_px"]),
                "truth_col": float(truth_row["col_px"]),
                "distance": float(min_dist),
                "det_code": det_code,
                "truth_code": truth_code,
                "code_match": det_code == truth_code,
                "det_identity": det_identity,
                "truth_identity": truth_identity,
                "identity_match": det_identity == truth_identity,
                "status": "correct" if det_code == truth_code else "code_mismatch",
            })
        else:
            # False positive (detected but no matching truth)
            matches.append({
                "detected_idx": det_idx,
                "truth_idx": None,
                "det_row": float(det_row),
                "det_col": float(det_col),
                "truth_row": None,
                "truth_col": None,
                "distance": float(min_dist) if n_truth > 0 else None,
                "det_code": det_code,
                "truth_code": None,
                "code_match": False,
                "det_identity": det_identity,
                "truth_identity": None,
                "identity_match": False,
                "status": "false_positive",
            })
    
    # Find unmatched truth beads (false negatives)
    false_negatives = []
    for truth_idx, (idx, truth_row) in enumerate(truth_df.iterrows()):
        if truth_idx not in matched_truth_indices:
            false_negatives.append({
                "truth_idx": truth_idx,
                "truth_row": float(truth_row["row_px"]),
                "truth_col": float(truth_row["col_px"]),
                "truth_code": str(truth_row["code"]),
                "truth_identity": str(truth_row["siRNA"]),
                "status": "false_negative",
            })
    
    # Calculate metrics
    true_positives = sum(1 for m in matches if m["status"] != "false_positive" and m["code_match"])
    code_matches = sum(1 for m in matches if m["code_match"])
    identity_matches = sum(1 for m in matches if m["identity_match"])
    false_positives = sum(1 for m in matches if m["status"] == "false_positive")
    false_negatives_count = len(false_negatives)
    
    # Avoid division by zero
    precision = code_matches / n_detected if n_detected > 0 else 0.0
    recall = code_matches / n_truth if n_truth > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "matches": matches,
        "false_negatives": false_negatives,
        "metrics": {
            "n_detected": int(n_detected),
            "n_truth": int(n_truth),
            "true_positives": int(true_positives),
            "false_positives": int(false_positives),
            "false_negatives": int(false_negatives_count),
            "code_matches": int(code_matches),
            "identity_matches": int(identity_matches),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "code_accuracy": float(code_matches / n_detected) if n_detected > 0 else 0.0,
        }
    }


# load sirna map from csv

def load_sirna_map(csv_path: Path) -> dict[str, str]:
    df = pd.read_csv(csv_path, dtype=str)
    if "code" not in df.columns or "siRNA" not in df.columns:
        raise ValueError(f"{csv_path} must have columns 'code' and 'siRNA'.")
    return dict(zip(df["code"].str.strip(), df["siRNA"].str.strip()))


def save_sirna_map_template(csv_path: Path) -> None:
    rows = [{"code": code, "siRNA": name} for code, name in DEFAULT_SIRNA_MAP.items()]
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"Saved siRNA map template → {csv_path}")
    print("Edit the 'siRNA' column with your actual gene targets, then re-run.")


# plot beads on image

def visualize(
    image_path: Path,
    channels: np.ndarray,
    blobs: np.ndarray,
    codes: list[str],
    identities: list[str],
    output_path: Path,
) -> None:
    # save annotated visualization
    projection = channels.max(axis=0)
    vmax = np.percentile(projection, 99.5)

    fig, ax = plt.subplots(figsize=(14, 11))
    ax.imshow(projection, cmap="gray", vmin=0, vmax=vmax, interpolation="nearest")
    ax.set_title(f"{image_path.name}  ·  {len(blobs)} beads detected", fontsize=11)
    ax.axis("off")

    for (row, col, radius), code, identity in zip(blobs, codes, identities):
        circ = mpatches.Circle(
            (col, row), radius * 1.5,
            linewidth=0.8, edgecolor="#00ff88", facecolor="none", alpha=0.85,
        )
        ax.add_patch(circ)
        ax.text(
            col + radius * 1.6, row,
            f"{code}\n{identity}",
            color="#ffdd44", fontsize=4.5, va="center",
            fontfamily="monospace", linespacing=1.3,
        )

    plt.tight_layout()
    fig.savefig(str(output_path), dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Visualization saved → {output_path}")


# main pipeline

def run_pipeline(args) -> pd.DataFrame:
    print("\n--- knock-bead decoder ---")

    # load sirna map
    map_path = Path(args.sirna_map)
    if map_path.exists():
        sirna_map = load_sirna_map(map_path)
        print(f"  siRNA map : {map_path}  ({len(sirna_map)} entries)")
    else:
        sirna_map = DEFAULT_SIRNA_MAP
        print(f"  siRNA map : built-in default  ({len(sirna_map)} entries)")
        print(f"  Tip: run --save-map-template to create an editable {args.sirna_map}")

    # load image
    if args.channel_files:
        paths = [Path(p) for p in args.channel_files]
        print(f"  Channels  : {[p.name for p in paths]}")
        channels = load_channel_files(paths)
        image_path = paths[0]
    else:
        image_path = Path(args.image)
        print(f"  Image     : {image_path}")
        print(f"  Channels  : indices {args.channels}")
        channels = load_multichannel_tiff(image_path, args.channels)

    n_ch = channels.shape[0]
    print(f"  Shape     : {channels.shape}  (C × H × W)")

    # find beads
    print(
        f"  detecting beads  "
        f"(min_σ={args.min_sigma}, max_σ={args.max_sigma}, "
        f"threshold={args.blob_threshold}) …"
    )
    blobs = detect_beads(
        channels,
        min_sigma=args.min_sigma,
        max_sigma=args.max_sigma,
        threshold=args.blob_threshold,
        overlap=args.overlap,
    )
    print(f"  Found {len(blobs)} beads")

    if len(blobs) == 0:
        print(
            "\n  No beads detected.\n"
            "  Suggestions:\n"
            "    • Lower --blob-threshold (e.g. 0.02)\n"
            "    • Adjust --min-sigma / --max-sigma to match bead size in pixels\n"
            "    • Check --channels points to the correct channel indices"
        )
        return pd.DataFrame()

    # measure intensity per channel
    intensities = measure_channel_intensities(channels, blobs, aperture_factor=args.aperture)

    # threshold to binary
    binary = intensities_to_binary(intensities, method=args.threshold_method)
    codes = binary_to_code(binary)
    identities = lookup_sirna(codes, sirna_map)

    # make results table
    ch_labels = CHANNEL_LABELS[:n_ch]
    rows = []
    for idx, (blob, inten, binar, code, identity) in enumerate(
        zip(blobs, intensities, binary, codes, identities)
    ):
        row: dict = {
            "bead_id": idx,
            "row_px": float(blob[0]),
            "col_px": float(blob[1]),
            "radius_px": float(blob[2]),
            "code": code,
            "siRNA": identity,
        }
        for label, val in zip(ch_labels, inten):
            row[f"{label}_intensity"] = round(float(val), 6)
        for label, val in zip(ch_labels, binar):
            row[f"{label}_binary"] = int(val)
        rows.append(row)

    df = pd.DataFrame(rows)

    # save results
    output_csv = Path(args.output)
    df.to_csv(str(output_csv), index=False)
    print(f"  Results   : {output_csv}  ({len(df)} beads)")

    # Optional visualisation
    if args.visualize:
        vis_path = output_csv.with_suffix(".png")
        visualize(image_path, channels, blobs, codes, identities, vis_path)

    # Print code distribution summary
    print("\n  ── Code distribution ──────────────────────────────────────────")
    counts = (
        df.groupby(["code", "siRNA"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    unknown = counts[counts["siRNA"] == "UNKNOWN"]
    known = counts[counts["siRNA"] != "UNKNOWN"]
    for _, r in pd.concat([known, unknown]).iterrows():
        bar = "█" * min(50, int(r["count"]))
        print(f"    {r['code']}  {r['siRNA']:<22s}  {r['count']:5d}  {bar}")

    return df


# ─── CLI ─────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Decode Knock-bead fluorescence codes from multichannel microscopy images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Input
    input_group = p.add_mutually_exclusive_group()
    input_group.add_argument(
        "image", nargs="?",
        help="Path to a multichannel TIFF image.",
    )
    input_group.add_argument(
        "--channel-files", nargs="+", metavar="CH.TIF",
        help="One single-channel TIFF per fluorophore (QD547 QD572 QD604 QD641 AF647).",
    )

    # Channel selection
    p.add_argument(
        "--channels", type=int, nargs="+", default=[0, 1, 2, 3, 4],
        help="0-based channel indices within the TIFF (ignored with --channel-files).",
    )

    # Output
    p.add_argument("--output", default="bead_results.csv", help="Output CSV path.")
    p.add_argument(
        "--sirna-map", default="siRNA_map.csv",
        help="CSV with 'code' and 'siRNA' columns.",
    )

    # Detection parameters
    p.add_argument("--min-sigma", type=float, default=2.0,
                   help="Min blob sigma for LoG detector (≈ min bead radius / √2 px).")
    p.add_argument("--max-sigma", type=float, default=8.0,
                   help="Max blob sigma for LoG detector.")
    p.add_argument("--blob-threshold", type=float, default=0.05,
                   help="LoG threshold — lower finds more (dimmer) blobs.")
    p.add_argument("--overlap", type=float, default=0.5,
                   help="Max overlap before two blobs are merged (0–1).")
    p.add_argument("--aperture", type=float, default=1.0,
                   help="Scale factor applied to detected radius for the signal aperture.")

    # Binarisation
    p.add_argument(
        "--threshold-method", choices=["otsu", "percentile", "gmm"], default="otsu",
        help="Per-channel binarisation strategy.",
    )

    # Extras
    p.add_argument("--visualize", action="store_true",
                   help="Save an annotated PNG alongside the CSV.")
    p.add_argument("--save-map-template", action="store_true",
                   help="Write siRNA_map.csv and exit (edit it, then re-run).")

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.save_map_template:
        save_sirna_map_template(Path(args.sirna_map))
        sys.exit(0)

    if not args.image and not args.channel_files:
        parser.error("Provide either a positional image path or --channel-files.")

    run_pipeline(args)


if __name__ == "__main__":
    main()
