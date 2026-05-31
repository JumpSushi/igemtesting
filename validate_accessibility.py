"""
validate_accessibility.py
--------------------------
For each siRNA candidate in a scan CSV:
  1. Fetch the target mRNA from NCBI (or use cached FASTA)
  2. Run RNAfold (ViennaRNA) to get per-nucleotide base-pair probabilities
  3. Score the target site: average UNPAIRED probability across the 19-nt window
     → higher = more accessible = better

Usage:
  python validate_accessibility.py scan_asgr1.csv --accession NM_001671 --score-min 4
  python validate_accessibility.py scan_asgr1.csv --accession NM_001671 --score-min 4 --top 10
"""

import argparse
import csv
import sys
import time
import requests
import RNA   # ViennaRNA

NCBI_EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
HEADERS = {"User-Agent": "igem-26/siRNA-validator contact@example.com"}


def fetch_fasta_seq(accession: str) -> str:
    """Return raw nucleotide sequence (no header, uppercase) for accession."""
    resp = requests.get(
        NCBI_EFETCH,
        params={"db": "nuccore", "id": accession, "rettype": "fasta", "retmode": "text"},
        headers=HEADERS,
        timeout=30,
    )
    resp.raise_for_status()
    lines = resp.text.strip().splitlines()
    return "".join(l.strip() for l in lines if not l.startswith(">")).upper()


def parse_position(pos_str: str):
    """'502-520' → (502, 520) 1-based inclusive."""
    parts = pos_str.split("-")
    return int(parts[0]), int(parts[1])


def score_accessibility(bpp_matrix, start: int, end: int) -> float:
    """
    Average probability of being UNPAIRED across positions start..end (1-based).
    bpp_matrix[i][j] = prob that base i is paired with base j (0-based internally).
    We use RNA.pfl_fold_up which gives per-nucleotide unpaired probability directly.
    """
    # bpp is accessed via the pr attribute; unpaired prob = 1 - sum of all pair probs for that base
    # We'll use the unpaired_probability array returned by pfl_fold
    raise NotImplementedError  # handled below via pfl_fold_up


def run_rnafold(seq: str):
    """
    Run RNAfold partition function on seq.
    Returns array of per-nucleotide unpaired probabilities (1-indexed, index 0 unused).
    """
    # fold_compound with partition function
    fc = RNA.fold_compound(seq)
    _, _ = fc.pf()  # run partition function
    # Get base pair probabilities; compute unpaired prob per nucleotide
    bppm = fc.bpp()  # (n+1) x (n+1) matrix, 1-indexed
    n = len(seq)
    unpaired = []
    for i in range(1, n + 1):
        paired_prob = sum(bppm[i][j] for j in range(1, n + 1) if j != i)
        unpaired.append(1.0 - min(paired_prob, 1.0))
    return unpaired  # index 0 = position 1


def accessibility_score(unpaired: list, start: int, end: int) -> float:
    """Mean unpaired probability over [start, end] (1-based inclusive)."""
    window = unpaired[start - 1 : end]
    if not window:
        return 0.0
    return sum(window) / len(window)


def main():
    parser = argparse.ArgumentParser(description="Score siRNA candidates for mRNA accessibility via RNAfold.")
    parser.add_argument("csv", help="Input scan CSV (e.g. scan_asgr1.csv)")
    parser.add_argument("--accession", required=True, help="NCBI mRNA accession to fold (e.g. NM_001671)")
    parser.add_argument("--score-min", type=int, default=0, help="Only show candidates with score >= this")
    parser.add_argument("--validated-only", action="store_true", help="Only show siDirect2-validated candidates")
    parser.add_argument("--top", type=int, default=20, help="Show top N by accessibility (default: 20)")
    parser.add_argument("--out", default=None, help="Output CSV (optional)")
    args = parser.parse_args()

    # 1. Fetch mRNA sequence
    print(f"[1/3] Fetching {args.accession} from NCBI…")
    seq = fetch_fasta_seq(args.accession)
    print(f"      Sequence length: {len(seq):,} nt")

    # 2. Run RNAfold partition function
    print(f"[2/3] Running RNAfold partition function (this takes ~10–30s for short mRNAs)…")
    t0 = time.time()
    unpaired = run_rnafold(seq)
    print(f"      Done in {time.time()-t0:.1f}s")

    # 3. Load candidates and score
    print(f"[3/3] Scoring candidates from {args.csv}…")
    with open(args.csv, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Filter
    if args.score_min:
        rows = [r for r in rows if r.get("score") and int(float(r["score"])) >= args.score_min]
    if args.validated_only:
        rows = [r for r in rows if r.get("sidirect2_validated", "").strip().lower() in ("true", "1", "yes")]

    # Score accessibility
    results = []
    for r in rows:
        try:
            start, end = parse_position(r["position"])
        except Exception:
            continue
        if end > len(seq):
            continue
        acc = accessibility_score(unpaired, start, end)
        r["accessibility"] = round(acc, 4)
        results.append(r)

    # Sort by accessibility descending
    results.sort(key=lambda x: float(x["accessibility"]), reverse=True)
    top = results[:args.top]

    # Print table
    print()
    print(f"{'Pos':<14} {'Sequence':<22} {'Score':>5} {'siD2':>5} {'Access':>7}  Interpretation")
    print("-" * 75)
    for r in top:
        acc = float(r["accessibility"])
        seq_disp = (r.get("sequence_dna") or r.get("sequence_rna") or "").lower().replace("u", "t")
        score = r.get("score", "—")
        validated = "✓" if r.get("sidirect2_validated", "").strip().lower() in ("true","1","yes") else "·"
        if acc >= 0.7:
            label = "✓ very accessible"
        elif acc >= 0.5:
            label = "~ moderately accessible"
        elif acc >= 0.3:
            label = "! partially structured"
        else:
            label = "✗ likely buried"
        print(f"{r['position']:<14} {seq_disp:<22} {score:>5} {validated:>5} {acc:>7.3f}  {label}")

    print()
    print(f"Accessibility = mean unpaired probability (0=fully paired/buried, 1=fully open)")
    print(f"Showed top {len(top)} of {len(results)} filtered candidates.")

    # Optional CSV output
    if args.out and results:
        fieldnames = list(results[0].keys())
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(results)
        print(f"Written → {args.out}")


if __name__ == "__main__":
    main()
