#!/usr/bin/env python3
"""
validate_offtarget.py — Phase 6 BLAST off-target validation for siRNA candidates

For each candidate guide strand, submits to NCBI BLAST (blastn-short, refseq_rna,
Homo sapiens) and flags any non-target transcript hit with ≤ MAX_MISMATCHES
mismatches covering ≥ 80 % of the guide.

Usage:
    # Check top candidates (score ≥ 5) for MTTP
    python validate_offtarget.py scan_mttp.csv --gene MTTP

    # Only check siDirect2-validated candidates
    python validate_offtarget.py scan_apob.csv --gene APOB --validated-only

    # Wider net: flag hits with ≤ 3 mismatches, check up to 30 candidates
    python validate_offtarget.py scan_mttp.csv --gene MTTP --max-mismatches 3 --max-candidates 30

    # With NCBI API key (removes 3 req/s rate limit)
    python validate_offtarget.py scan_mttp.csv --gene MTTP --api-key YOUR_KEY
"""

import argparse
import csv
import sys
import time
import xml.etree.ElementTree as ET

import requests

BLAST_URL = "https://blast.ncbi.nlm.nih.gov/blast/Blast.cgi"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; siRNA-offtarget-checker/1.0; "
        "iGEM research use)"
    )
}


# ── NCBI BLAST helpers ────────────────────────────────────────────────────────

def blast_submit(sequence: str, max_hits: int = 100, api_key: str | None = None) -> str:
    """Submit a blastn-short job; return the RID string."""
    params: dict = {
        "CMD":          "Put",
        "PROGRAM":      "blastn",
        "DATABASE":     "refseq_rna",
        "QUERY":        sequence,
        "HITLIST_SIZE": str(max_hits),
        "EXPECT":       "1000",
        "WORD_SIZE":    "7",
        "FILTER":       "L",               # low-complexity mask
        "ENTREZ_QUERY": "Homo sapiens[Organism]",
        "FORMAT_TYPE":  "XML",
        "TASK":         "blastn-short",    # optimised for <50 nt queries
    }
    if api_key:
        params["api_key"] = api_key

    resp = requests.post(BLAST_URL, data=params, headers=HEADERS, timeout=60)
    resp.raise_for_status()

    for line in resp.text.splitlines():
        if "RID =" in line:
            return line.strip().split("=")[1].strip()
    raise RuntimeError(f"RID not found in BLAST response:\n{resp.text[:400]}")


def blast_poll(rid: str, poll_interval: int = 12, api_key: str | None = None) -> str:
    """Block until the BLAST job is READY; return the XML result string."""
    params: dict = {"CMD": "Get", "RID": rid, "FORMAT_TYPE": "XML"}
    if api_key:
        params["api_key"] = api_key

    while True:
        time.sleep(poll_interval)
        resp = requests.get(BLAST_URL, params=params, headers=HEADERS, timeout=60)
        resp.raise_for_status()
        text = resp.text
        if "Status=READY" in text:
            return text
        if "Status=FAILED" in text:
            raise RuntimeError(f"BLAST job {rid} failed")
        if "Status=UNKNOWN" in text:
            raise RuntimeError(f"BLAST job {rid} expired / unknown RID")
        # Status=WAITING — keep polling


def parse_blast_xml(
    xml_text: str,
    query_len: int,
    max_mismatches: int,
    target_gene: str,
    min_coverage: float = 0.80,
) -> list[dict]:
    """
    Parse BLAST XML output.

    Returns every HSP where:
      - alignment covers ≥ *min_coverage* of the query
      - mismatches ≤ *max_mismatches*

    Each result dict contains:
        accession, title, align_len, identity, mismatches,
        coverage_pct, q_start, q_end, is_self, is_offtarget
    """
    hits: list[dict] = []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        print(f"[warning] XML parse error: {exc}", file=sys.stderr)
        return hits

    for hit in root.iter("Hit"):
        title     = hit.findtext("Hit_def", "") or ""
        accession = hit.findtext("Hit_accession", "") or ""

        # Iterate HSPs; report best (first) qualifying one per hit
        for hsp in hit.iter("Hsp"):
            align_len = int(hsp.findtext("Hsp_align-len", "0") or 0)
            identity  = int(hsp.findtext("Hsp_identity",  "0") or 0)
            mismatches = align_len - identity
            q_start   = int(hsp.findtext("Hsp_query-from", "0") or 0)
            q_end     = int(hsp.findtext("Hsp_query-to",   "0") or 0)

            coverage = align_len / query_len if query_len > 0 else 0

            if coverage < min_coverage:
                continue
            if mismatches > max_mismatches:
                continue

            is_self = target_gene.upper() in title.upper()
            hits.append({
                "accession":    accession,
                "title":        title[:100],
                "align_len":    align_len,
                "identity":     identity,
                "mismatches":   mismatches,
                "coverage_pct": round(coverage * 100, 1),
                "q_start":      q_start,
                "q_end":        q_end,
                "is_self":      is_self,
                "is_offtarget": not is_self,
            })
            break  # best HSP only per hit

    return hits


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 6: BLAST off-target check for siRNA candidates"
    )
    parser.add_argument("csv",  help="Scan CSV produced by scrape_sirna.py")
    parser.add_argument("--gene", required=True,
                        help="Target gene name (e.g. MTTP) — used to distinguish self-hits")
    parser.add_argument("--out", default=None,
                        help="Output CSV path (default: offtarget_<gene>.csv)")
    parser.add_argument("--score-min", type=int, default=5,
                        help="Include candidates with score ≥ this (default: 5)")
    parser.add_argument("--validated-only", action="store_true",
                        help="Only BLAST siDirect2-validated candidates")
    parser.add_argument("--max-mismatches", type=int, default=2,
                        help="Flag off-target hits with ≤ this many mismatches (default: 2)")
    parser.add_argument("--min-coverage", type=float, default=0.80,
                        help="Minimum query coverage fraction to consider (default: 0.80)")
    parser.add_argument("--max-candidates", type=int, default=20,
                        help="Max candidates to submit to BLAST (default: 20)")
    parser.add_argument("--poll-interval", type=int, default=12,
                        help="Seconds between BLAST status polls (default: 12)")
    parser.add_argument("--api-key", default=None,
                        help="NCBI API key (optional; raises rate limit to 10 req/s)")
    args = parser.parse_args()

    out_file = args.out or f"offtarget_{args.gene.lower()}.csv"

    # ── Load + filter candidates ─────────────────────────────────────────────
    with open(args.csv, newline="") as f:
        rows = list(csv.DictReader(f))

    candidates: list[dict] = []
    for r in rows:
        try:
            score = float(r.get("score") or 0)
        except ValueError:
            score = 0.0
        validated = str(r.get("sidirect2_validated", "")).strip().lower() in ("true", "1", "yes")

        if args.validated_only and not validated:
            continue
        if score >= args.score_min or validated:
            candidates.append(r)

    # Sort by score desc, then take top N
    candidates.sort(key=lambda r: float(r.get("score") or 0), reverse=True)
    candidates = candidates[: args.max_candidates]

    if not candidates:
        sys.exit(
            f"[error] No candidates pass filters "
            f"(score>={args.score_min}, validated_only={args.validated_only})"
        )

    print(f"[phase 6] BLAST off-target validation")
    print(f"          Target gene    : {args.gene.upper()}")
    print(f"          Candidates     : {len(candidates)}")
    print(f"          Max mismatches : ≤{args.max_mismatches}")
    print(f"          Min coverage   : ≥{int(args.min_coverage * 100)}%")
    print(f"          Output         : {out_file}")
    print()

    results: list[dict] = []

    for i, cand in enumerate(candidates, 1):
        seq_rna = (cand.get("sequence_rna") or cand.get("sequence_dna") or "").strip()
        seq_dna = seq_rna.replace("U", "T")  # BLAST requires DNA notation
        if not seq_dna:
            continue

        pos   = cand.get("position", "?")
        score = cand.get("score", "?")
        val   = str(cand.get("sidirect2_validated", "")).lower() in ("true", "1", "yes")

        print(
            f"  [{i:>2}/{len(candidates)}] {pos:<14} {seq_dna}  "
            f"score={score}{'  ✓siD2' if val else ''}",
            end="  ",
            flush=True,
        )

        try:
            rid  = blast_submit(seq_dna, api_key=args.api_key)
            xml  = blast_poll(rid, poll_interval=args.poll_interval, api_key=args.api_key)
            hits = parse_blast_xml(
                xml,
                query_len=len(seq_dna),
                max_mismatches=args.max_mismatches,
                target_gene=args.gene,
                min_coverage=args.min_coverage,
            )

            offtargets = [h for h in hits if h["is_offtarget"]]
            self_hits  = [h for h in hits if h["is_self"]]

            if not offtargets:
                print("✓ clean")
                flag = "CLEAN"
            else:
                print(f"⚠  {len(offtargets)} off-target(s): "
                      + ", ".join(h["title"][:40] for h in offtargets[:3]))
                flag = "OFFTARGET"

            results.append({
                "position":             pos,
                "sequence_dna":         seq_dna,
                "sequence_rna":         seq_rna,
                "score":                score,
                "sidirect2_validated":  val,
                "self_hits":            len(self_hits),
                "offtarget_count":      len(offtargets),
                "offtarget_genes":      "; ".join(h["title"]     for h in offtargets[:5]),
                "offtarget_mismatches": "; ".join(str(h["mismatches"]) for h in offtargets[:5]),
                "offtarget_accessions": "; ".join(h["accession"] for h in offtargets[:5]),
                "flag":                 flag,
            })

        except Exception as exc:
            print(f"ERROR: {exc}")
            results.append({
                "position":             pos,
                "sequence_dna":         seq_dna,
                "sequence_rna":         seq_rna,
                "score":                score,
                "sidirect2_validated":  val,
                "self_hits":            "",
                "offtarget_count":      "",
                "offtarget_genes":      f"ERROR: {exc}",
                "offtarget_mismatches": "",
                "offtarget_accessions": "",
                "flag":                 "ERROR",
            })

        # Polite delay between submissions (NCBI: 3 req/s without key, 10 with key)
        time.sleep(3 if not args.api_key else 0.5)

    # ── Write output CSV ─────────────────────────────────────────────────────
    if results:
        fieldnames = list(results[0].keys())
        with open(out_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

    # ── Summary ──────────────────────────────────────────────────────────────
    n_clean    = sum(1 for r in results if r["flag"] == "CLEAN")
    n_offtgt   = sum(1 for r in results if r["flag"] == "OFFTARGET")
    n_error    = sum(1 for r in results if r["flag"] == "ERROR")

    print()
    print(f"{'='*70}")
    print(f"  RESULTS  —  {args.gene.upper()}")
    print(f"{'='*70}")
    print(f"  Clean (no off-targets)  : {n_clean}")
    print(f"  Off-target hits flagged : {n_offtgt}")
    if n_error:
        print(f"  Errors                  : {n_error}")
    print(f"  Output written          : {out_file}")
    print(f"{'='*70}")
    print()
    print(f"{'Position':<14} {'Sequence':<28} {'Score':<7} {'Validated':<10} {'Off-tgts':<9} Flag")
    print("-" * 80)
    for r in results:
        print(
            f"{r['position']:<14} {r['sequence_dna']:<28} {r['score']:<7} "
            f"{'yes' if r['sidirect2_validated'] else 'no':<10} "
            f"{str(r['offtarget_count']):<9} {r['flag']}"
        )


if __name__ == "__main__":
    main()
