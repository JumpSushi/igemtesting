#!/usr/bin/env python3
"""
Parameter sweep to find optimal decoding settings.
Phase 1: sweeps blob detection params
Phase 2: sweeps intensity measurement params using best detection
"""
import itertools, sys
import numpy as np
from testgen import make_image
from pipeline import (
    detect_beads, measure_channel_intensities,
    intensities_to_binary, binary_to_code,
    lookup_sirna, match_beads_to_truth, DEFAULT_SIRNA_MAP,
)

SEEDS = [42, 7, 99, 314, 512]

def score(image, truth, min_sigma, max_sigma, threshold, overlap, aperture, method):
    try:
        blobs = detect_beads(image, min_sigma=min_sigma, max_sigma=max_sigma,
                             threshold=threshold, overlap=overlap)
        if len(blobs) == 0:
            return 0.0, 0, 0
        intens = measure_channel_intensities(image, blobs, aperture_factor=aperture)
        binary = intensities_to_binary(intens, method=method)
        codes  = binary_to_code(binary)
        ids    = lookup_sirna(codes, DEFAULT_SIRNA_MAP)
        cmp    = match_beads_to_truth(blobs, codes, ids, truth)
        m      = cmp["metrics"]
        return m["f1_score"], m["n_detected"], m["code_accuracy"]
    except Exception as e:
        return 0.0, 0, 0


print("Generating test images …")
cases = [make_image(n_beads=120, noise_level=0.03, seed=s) for s in SEEDS]
print(f"  {len(cases)} images ready.\n")

# ── Phase 1: detection parameters (fixed aperture=1.0, method=otsu) ────────
print("Phase 1: sweeping detection parameters …")
det_grid = list(itertools.product(
    [1.5, 2.0, 2.5, 3.0],        # min_sigma
    [6.0, 8.0, 10.0, 12.0],      # max_sigma
    [0.01, 0.02, 0.03, 0.05, 0.07, 0.10],  # blob_threshold
    [0.3, 0.5, 0.7],             # overlap
))
print(f"  {len(det_grid)} combinations × {len(cases)} seeds …")

best_det = {"f1": -1}
for min_s, max_s, thr, ovl in det_grid:
    if min_s >= max_s:
        continue
    f1s = [score(img, truth, min_s, max_s, thr, ovl, 1.0, "otsu")[0]
           for img, truth in cases]
    avg = np.mean(f1s)
    if avg > best_det["f1"]:
        best_det = dict(f1=avg, min_sigma=min_s, max_sigma=max_s,
                        threshold=thr, overlap=ovl)

print(f"  Best detection  f1={best_det['f1']:.4f}")
print(f"    min_sigma={best_det['min_sigma']}, max_sigma={best_det['max_sigma']}")
print(f"    threshold={best_det['threshold']}, overlap={best_det['overlap']}\n")

# ── Phase 2: intensity/binarisation parameters ──────────────────────────────
print("Phase 2: sweeping intensity parameters …")
int_grid = list(itertools.product(
    [0.6, 0.8, 1.0, 1.2, 1.5],  # aperture
    ["otsu", "gmm"],             # threshold method
))
print(f"  {len(int_grid)} combinations × {len(cases)} seeds …")

best_int = {"f1": -1}
for ap, meth in int_grid:
    f1s = [score(img, truth,
                 best_det["min_sigma"], best_det["max_sigma"],
                 best_det["threshold"], best_det["overlap"],
                 ap, meth)[0]
           for img, truth in cases]
    avg = np.mean(f1s)
    if avg > best_int["f1"]:
        best_int = dict(f1=avg, aperture=ap, method=meth)

print(f"  Best intensity  f1={best_int['f1']:.4f}")
print(f"    aperture={best_int['aperture']}, method={best_int['method']}\n")

# ── Final summary ────────────────────────────────────────────────────────────
best = {**best_det, **best_int}
print("=" * 55)
print("OPTIMAL PARAMETERS")
print("=" * 55)
print(f"  min_sigma        = {best['min_sigma']}")
print(f"  max_sigma        = {best['max_sigma']}")
print(f"  blob_threshold   = {best['threshold']}")
print(f"  overlap          = {best['overlap']}")
print(f"  aperture_factor  = {best['aperture']}")
print(f"  threshold_method = {best['method']}")
print(f"  avg F1 (5 seeds) = {best['f1']:.4f}")
print("=" * 55)
