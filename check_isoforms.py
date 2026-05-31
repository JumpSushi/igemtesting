#!/usr/bin/env python3
"""
check_isoforms.py — Phase 5b isoform coverage check for siRNA candidates

For each candidate in a scan CSV, fetches every RefSeq mRNA transcript variant
for the target gene and reports what fraction contain the candidate sequence.

Candidates present in ALL isoforms are marked 'ALL'.
Candidates present in SOME are marked 'PARTIAL (n/total)'.
Candidates absent from all non-reference isoforms are marked 'REFERENCE_ONLY'.

Usage:
    python check_isoforms.py scan_mttp.csv --gene MTTP
    python check_isoforms.py scan_pcsk9.csv --gene PCSK9 --score-min 5 --validated-only
    python check_isoforms.py scan_apob.csv --gene APOB --max-candidates 50 --api-key YOUR_KEY

Output:
    isoform_<gene>.csv  — original columns + isoform_coverage, isoform_hits, isoform_total
"""

import argparse
import csv
import sys
import time

import requests

NCBI_ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
NCBI_EFETCH  = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
NCBI_EPOST   = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/epost.fcgi"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; siRNA-isoform-checker/1.0; iGEM research use)"
}


# ── NCBI helpers ──────────────────────────────────────────────────────────────

def _params(extra: dict, api_key: str | None) -> dict:
    p = {"retmode": "text", **extra}
    if api_key:
        p["api_key"] = api_key
    return p


def fetch_all_accessions(gene: str, api_key: str | None = None) -> list[str]:
    """
    Return all RefSeq mRNA accessions (NM_*) for *gene* in Homo sapiens.
    Strategy: eSearch gene db → gene ID → eFetch gene_table (plain text) → parse NM_ lines.
    This is the most reliable method — gene_table always lists every transcript variant.
    """
    import re as _re
    import xml.etree.ElementTree as ET

    def _get(url, params):
        if api_key:
            params["api_key"] = api_key
        r = requests.get(url, params=params, headers=HEADERS, timeout=30)
        r.raise_for_status()
        return r

    # 1. Resolve gene symbol → Gene ID
    resp = _get(NCBI_ESEARCH, {
        "db":      "gene",
        "term":    f"{gene}[Gene Name] AND Homo sapiens[Organism]",
        "retmax":  "5",
        "retmode": "xml",
    })
    root = ET.fromstring(resp.text)
    gene_ids = [el.text for el in root.findall(".//Id") if el.text]
    if not gene_ids:
        return []
    gene_id = gene_ids[0]

    # 2. Fetch gene table — lists every mRNA variant as "NM_XXXXXX.V"
    time.sleep(0.35 if not api_key else 0.11)
    resp = _get(NCBI_EFETCH, {
        "db":      "gene",
        "id":      gene_id,
        "rettype": "gene_table",
        "retmode": "text",
    })
    # Extract all NM_ accessions (strip version suffix)
    accessions = list(dict.fromkeys(
        m.group(1)
        for m in _re.finditer(r'\b(NM_\d+)\.\d+', resp.text)
    ))
    return accessions


def fetch_fasta_seq(accession: str, api_key: str | None = None) -> str | None:
    """Return the raw uppercase nucleotide sequence for *accession*, or None on error."""
    try:
        resp = requests.get(
            NCBI_EFETCH,
            params=_params({
                "db":      "nuccore",
                "id":      accession,
                "rettype": "fasta",
            }, api_key),
            headers=HEADERS,
            timeout=30,
        )
        resp.raise_for_status()
        lines = resp.text.strip().splitlines()
        return "".join(l.strip() for l in lines if not l.startswith(">")).upper()
    except Exception as e:
        print(f"    [warn] Could not fetch {accession}: {e}", file=sys.stderr)
        return None


# ── Core logic ────────────────────────────────────────────────────────────────

def check_coverage(
    sequence_dna: str,
    isoform_seqs: dict[str, str],
) -> tuple[str, int, int]:
    """
    Check whether *sequence_dna* is present (exact substring) in each isoform.

    Returns:
        (label, hits, total)
        label: 'ALL' | 'PARTIAL' | 'NONE'
    """
    seq = sequence_dna.upper().replace("U", "T")
    hits = sum(1 for s in isoform_seqs.values() if seq in s)
    total = len(isoform_seqs)
    if hits == total:
        label = "ALL"
    elif hits > 0:
        label = "PARTIAL"
    else:
        label = "NONE"
    return label, hits, total


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Check siRNA candidates against all gene isoforms")
    ap.add_argument("csv_in",           help="Input scan CSV (e.g. scan_mttp.csv)")
    ap.add_argument("--gene",           required=True, help="Gene symbol (e.g. MTTP)")
    ap.add_argument("--score-min",      type=int, default=0,   help="Only check candidates with score >= N")
    ap.add_argument("--validated-only", action="store_true",   help="Only check siDirect2-validated candidates")
    ap.add_argument("--max-candidates", type=int, default=200, help="Max candidates to check (default 200)")
    ap.add_argument("--api-key",        default=None,          help="NCBI API key (10 req/s vs 3 req/s)")
    ap.add_argument("--out",            default=None,          help="Output CSV path (default: isoform_<gene>.csv)")
    args = ap.parse_args()

    out_path = args.out or f"isoform_{args.gene.lower()}.csv"

    # ── 1. Read candidates ────────────────────────────────────────────────────
    with open(args.csv_in, newline="") as fh:
        rows = list(csv.DictReader(fh))

    candidates = rows
    if args.score_min > 0:
        candidates = [r for r in candidates if _int(r.get("score")) >= args.score_min]
    if args.validated_only:
        candidates = [r for r in candidates if r.get("sidirect2_validated", "").lower() in ("true", "1", "yes")]
    candidates = candidates[: args.max_candidates]

    if not candidates:
        print("[!] No candidates match the filter criteria.", file=sys.stderr)
        sys.exit(1)

    print(f"[1/3] Loaded {len(candidates)} candidate(s) from {args.csv_in}")

    # ── 2. Fetch all isoform sequences ────────────────────────────────────────
    print(f"[2/3] Fetching all RefSeq mRNA accessions for {args.gene} …")
    accessions = fetch_all_accessions(args.gene, args.api_key)
    if not accessions:
        print(f"[!] No NM_ accessions found for {args.gene}. Check gene symbol.", file=sys.stderr)
        sys.exit(1)

    print(f"      Found {len(accessions)} accession(s): {', '.join(accessions)}")

    isoform_seqs: dict[str, str] = {}
    for acc in accessions:
        delay = 0.11 if args.api_key else 0.34  # stay under rate limit
        time.sleep(delay)
        seq = fetch_fasta_seq(acc, args.api_key)
        if seq:
            isoform_seqs[acc] = seq
            print(f"      {acc}: {len(seq):,} nt")

    if not isoform_seqs:
        print("[!] Could not fetch any isoform sequences.", file=sys.stderr)
        sys.exit(1)

    # ── 3. Check each candidate ───────────────────────────────────────────────
    print(f"[3/3] Checking {len(candidates)} candidate(s) against {len(isoform_seqs)} isoform(s) …")

    all_count = partial_count = none_count = 0
    out_rows = []

    for i, row in enumerate(candidates, 1):
        seq_dna = row.get("sequence_dna", "").replace("U", "T").upper()
        if not seq_dna:
            # fall back to RNA → DNA
            seq_dna = row.get("sequence_rna", "").replace("U", "T").upper()

        label, hits, total = check_coverage(seq_dna, isoform_seqs)

        row["isoform_coverage"] = label
        row["isoform_hits"]     = hits
        row["isoform_total"]    = total
        out_rows.append(row)

        if label == "ALL":
            all_count += 1
        elif label == "PARTIAL":
            partial_count += 1
        else:
            none_count += 1

        if i % 25 == 0 or i == len(candidates):
            print(f"      {i}/{len(candidates)}  ALL={all_count}  PARTIAL={partial_count}  NONE={none_count}")

    # ── Write output ──────────────────────────────────────────────────────────
    if out_rows:
        fieldnames = list(out_rows[0].keys())
        with open(out_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(out_rows)

    print(f"\n[result] Written {len(out_rows)} row(s) → {out_path}")
    print(f"         ALL isoforms : {all_count}")
    print(f"         PARTIAL      : {partial_count}")
    print(f"         NONE         : {none_count}")
    print()
    print("Recommendation: prefer candidates labelled ALL for pan-isoform knockdown.")
    print("                PARTIAL candidates may be isoform-specific — check which isoforms they miss.")


def _int(v) -> int:
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return 0


if __name__ == "__main__":
    main()
