#!/usr/bin/env python3
# synthetic bead image generator
# outputs: test_image.tif (5 channels) + truth.csv

import argparse
import numpy as np
import pandas as pd
import tifffile
from pathlib import Path

# all 21 codes
CODES = [
    "10000", "01000", "00100", "00010", "00001",
    "11000", "10100", "10010", "10001", "01100",
    "01010", "01001", "00110", "00101", "00011",
    "11100", "01110", "00111", "10110", "10011",
    "11111",
]


def make_image(
    n_beads: int = 150,
    image_size: int = 512,
    bead_radius: float = 4.0,
    signal_intensity: float = 0.8,
    noise_level: float = 0.03,
    background_level: float = 0.04,
    seed: int = 42,
) -> tuple[np.ndarray, pd.DataFrame]:
    # generate fake bead image + truth table
    rng = np.random.default_rng(seed)
    n_channels = 5
    H = W = image_size

    # add background noise
    image = rng.normal(loc=background_level, scale=noise_level,
                       size=(n_channels, H, W)).astype(np.float32)
    image = np.clip(image, 0, None)

    # place beads away from edges
    margin = int(bead_radius * 3)
    rows = rng.integers(margin, H - margin, size=n_beads)
    cols = rng.integers(margin, W - margin, size=n_beads)
    codes = [CODES[i % len(CODES)] for i in range(n_beads)]

    # build gaussian PSF
    sigma = bead_radius / 2.5
    k = int(bead_radius * 4) | 1  # odd size
    ax = np.arange(k) - k // 2
    kernel = np.exp(-0.5 * (ax[:, None] ** 2 + ax[None, :] ** 2) / sigma ** 2)
    kernel /= kernel.max()

    kh, kw = kernel.shape
    half_h, half_w = kh // 2, kw // 2

    for row, col, code in zip(rows, cols, codes):
        # clamp to image bounds
        r0 = max(0, row - half_h);  r1 = min(H, row + half_h + 1)
        c0 = max(0, col - half_w);  c1 = min(W, col + half_w + 1)

        kr0 = half_h - (row - r0);  kr1 = kr0 + (r1 - r0)
        kc0 = half_w - (col - c0);  kc1 = kc0 + (c1 - c0)

        patch = kernel[kr0:kr1, kc0:kc1]

        for ch in range(n_channels):
            if code[ch] == "1":
                # bright bead with jitter
                amp = signal_intensity * rng.uniform(0.85, 1.0)
                image[ch, r0:r1, c0:c1] += (amp * patch).astype(np.float32)
            else:
                # faint ghost (autofluorescence)
                image[ch, r0:r1, c0:c1] += (0.05 * patch).astype(np.float32)

    image = np.clip(image, 0, 1).astype(np.float32)

    sirna_labels = {
        "10000": "siRNA-A",   "01000": "siRNA-B",   "00100": "siRNA-C",
        "00010": "siRNA-D",   "00001": "siRNA-E",   "11000": "siRNA-AB",
        "10100": "siRNA-AC",  "10010": "siRNA-AD",  "10001": "siRNA-AE",
        "01100": "siRNA-BC",  "01010": "siRNA-BD",  "01001": "siRNA-BE",
        "00110": "siRNA-CD",  "00101": "siRNA-CE",  "00011": "siRNA-DE",
        "11100": "siRNA-ABC", "01110": "siRNA-BCD", "00111": "siRNA-CDE",
        "10110": "siRNA-ACD", "10011": "siRNA-ADE", "11111": "siRNA-ABCDE",
    }

    truth = pd.DataFrame({
        "bead_id": range(n_beads),
        "row_px":  rows,
        "col_px":  cols,
        "code":    codes,
        "siRNA":   [sirna_labels[c] for c in codes],
    })

    return image, truth


def main():
    p = argparse.ArgumentParser(
        description="Generate a synthetic multichannel bead TIFF for pipeline testing.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--beads",      type=int,   default=150,    help="Number of beads to place.")
    p.add_argument("--size",       type=int,   default=512,    help="Image width/height in pixels.")
    p.add_argument("--radius",     type=float, default=4.0,    help="Bead PSF radius in pixels.")
    p.add_argument("--signal",     type=float, default=0.8,    help="Peak bead intensity (0–1).")
    p.add_argument("--noise",      type=float, default=0.03,   help="Background noise std-dev.")
    p.add_argument("--background", type=float, default=0.04,   help="Background DC level.")
    p.add_argument("--seed",       type=int,   default=42,     help="Random seed.")
    p.add_argument("--output",     default="test_image.tif",   help="Output TIFF path.")
    args = p.parse_args()

    image, truth = make_image(
        n_beads=args.beads,
        image_size=args.size,
        bead_radius=args.radius,
        signal_intensity=args.signal,
        noise_level=args.noise,
        background_level=args.background,
        seed=args.seed,
    )

    out = Path(args.output)
    tifffile.imwrite(str(out), image, photometric="minisblack")
    truth_path = out.with_name(out.stem + "_truth.csv")
    truth.to_csv(str(truth_path), index=False)

    print(f"Saved {out}           — shape {image.shape}, dtype {image.dtype}")
    print(f"Saved {truth_path}  — {len(truth)} beads, {truth['code'].nunique()} unique codes")
    print()
    print("Run the decoder:")
    print(f"  python pipeline.py {out} --visualize")


if __name__ == "__main__":
    main()
