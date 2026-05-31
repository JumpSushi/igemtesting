#!/usr/bin/env python3
"""
scrape_sirna.py — full siRNA design pipeline (Phases 1-5)

Phase 1  Novel target recommendations (unchecked genes, high iGEM novelty)
Phase 2  Local pre-screening: length 19-27 nt, AA/UU termini, no 4+ repeats
Phase 3  G/C content window 30-52 %
Phase 4  Position-specific efficiency rules (pos 1/10/13/19)
Phase 5  siDirect2 cross-validation (NCBI FASTA → web scrape)

Usage:
    # Full pipeline (local screen + siDirect2 validation)
    python scrape_sirna.py NM_139314 --out candidates.csv

    # Local screening only (no web call to siDirect2)
    python scrape_sirna.py NM_139314 --local-only

    # Show recommended novel target genes for iGEM/original research
    python scrape_sirna.py --list-targets

    # Override CDS manually
    python scrape_sirna.py NM_139314 --cds-start 172 --cds-end 1500
"""

import argparse
import csv
import re
import sys
import time

import requests
from bs4 import BeautifulSoup

NCBI_EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
SIDIRECT2_DESIGN = "http://sidirect2.rnai.jp/design.cgi"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; siRNA-scraper/1.0; "
        "iGEM research use)"
    )
}

# ── Phase 1 — Novel target gene list ─────────────────────────────────────────
# Genes WITHOUT commercially approved siRNA drugs or advanced clinical trials.
# High priority for original iGEM / academic research submissions.
NOVEL_TARGETS: dict[str, dict] = {
    "ANGPTL4": {
        "accession":   "NM_139314",
        "description": "Angiopoietin-like 4 — inhibits lipoprotein lipase; \n"
                        "                regulates triglyceride metabolism",
        "cds_hint":    172,   # first ATG at nt 172 in reference transcript
    },
    "APOB": {
        "accession":   "NM_000384",
        "description": "Apolipoprotein B-100 — core structural protein of \n"
                        "                LDL; primary target for LDL-C lowering",
        "cds_hint":    None,
    },
    "CETP": {
        "accession":   "NM_000078",
        "description": "Cholesteryl ester transfer protein — transfers \n"
                        "                cholesteryl esters from HDL to LDL/VLDL",
        "cds_hint":    None,
    },
    "MTTP": {
        "accession":   "NM_000253",
        "description": "Microsomal triglyceride transfer protein — required \n"
                        "                for VLDL/chylomicron assembly",
        "cds_hint":    None,
    },
    "ASGR1": {
        "accession":   "NM_001671",
        "description": "Asialoglycoprotein receptor 1 — liver-specific; \n"
                        "                enables hepatocyte-targeted siRNA delivery",
        "cds_hint":    None,
    },
    "SREBF1": {
        "accession":   "NM_004176",
        "description": "SREBP-1 — master transcription factor for fatty \n"
                        "                acid / lipogenic gene programs",
        "cds_hint":    None,
    },
    "SREBF2": {
        "accession":   "NM_004599",
        "description": "SREBP-2 — master transcription factor for \n"
                        "                cholesterol biosynthesis",
        "cds_hint":    None,
    },
    "APOA5": {
        "accession":   "NM_052968",
        "description": "Apolipoprotein A-V — key regulator of plasma \n"
                        "                triglyceride levels via LPL activation",
        "cds_hint":    None,
    },
    "PCSK9": {
        "accession":   "NM_174936",
        "description": "Proprotein convertase subtilisin/kexin type 9 — \n"
                        "                degrades LDL receptors; target of Inclisiran",
        "cds_hint":    None,
    },
    "APOC3": {
        "accession":   "NM_000040",
        "description": "Apolipoprotein C-III — inhibits lipoprotein lipase \n"
                        "                and hepatic uptake; target of Plozasiran",
        "cds_hint":    None,
    },
    "ANGPTL3": {
        "accession":   "NM_014495",
        "description": "Angiopoietin-like 3 — inhibits LPL and EL; \n"
                        "                target of Zodasiran",
        "cds_hint":    None,
    },
    "LPA": {
        "accession":   "NM_005577",
        "description": "Lipoprotein(a) — independent CVD risk factor; \n"
                        "                target of Lepodisiran / Olpasiran / SLN360",
        "cds_hint":    None,
    },
    "PNPLA3": {
        "accession":   "NM_025225",
        "description": "Patatin-like phospholipase domain 3 — mutations \n"
                        "                impair liver lipid metabolism; ARO-PNPLA3 phase 1",
        "cds_hint":    None,
    },
    "ALK7": {
        "accession":   "NM_145259",
        "description": "Activin receptor-like kinase 7 (ACVR1C) — \n"
                        "                receptor regulating fat storage; SA030 phase 1",
        "cds_hint":    None,
    },
    "HSD17B13": {
        "accession":   "NM_178135",
        "description": "17β-hydroxysteroid dehydrogenase 13 — enzyme \n"
                        "                localised in liver lipid droplets; ALN-HSD / ARO-HSD phase 2",
        "cds_hint":    None,
    },
    "DGAT2": {
        "accession":   "NM_032564",
        "description": "Diacylglycerol O-acyltransferase 2 — key enzyme \n"
                        "                for triglyceride synthesis in the liver",
        "cds_hint":    None,
    },
}
# Genes with approved drugs / advanced clinical programs (kept for reference)
ESTABLISHED_TARGETS: set[str] = set()  # all now in NOVEL_TARGETS


# ── Phase 2-4 — Local screening helpers ──────────────────────────────────────

def parse_fasta_seq(fasta: str) -> str:
    """Return the raw uppercase nucleotide sequence from a FASTA string."""
    lines = fasta.strip().splitlines()
    return "".join(l.strip() for l in lines if not l.startswith(">")).upper()


def find_first_atg(seq: str) -> int:
    """
    Return 1-based position of the first ATG codon in *seq*.
    Returns -1 if no ATG is found.
    """
    pos = seq.upper().find("ATG")
    return pos + 1 if pos >= 0 else -1


def gc_content(seq: str) -> float:
    """Return G/C percentage (0-100) for *seq*."""
    s = seq.upper()
    return (s.count("G") + s.count("C")) / len(s) * 100 if s else 0.0


def has_homopolymer(seq: str, n: int = 4) -> bool:
    """Return True if *seq* contains a run of >= *n* identical bases."""
    seq = seq.upper()
    for base in "ACGT":
        if base * n in seq:
            return True
    return False


def _check_position_rules(window: str) -> dict[str, bool]:
    """
    Position-specific efficiency rules applied to a 19-27 nt window
    (DNA notation, uppercase, 1-based positions, sense/passenger strand).

    Rule     Pos  Requirement          Biochemical rationale
    ──────── ───  ──────────────────── ──────────────────────────────────────────
    P1_GC   1    G or C               Passenger 5' stability → asymmetric RISC loading
    P10_U   10   U (T in DNA)         Aligns with Ago2 catalytic centre
    P13_nG  13   not G                Prevents steric block of RISC recognition
    P19_AU  19   A or T               Guide 5' = complement → U or A preferred
    """
    w = window.upper()
    return {
        "pos1_GC":    w[0]  in ("G", "C"),
        "pos10_U":    w[9]  == "T",         # T in DNA = U in RNA
        "pos13_notG": w[12] != "G",
        "pos19_AU":   len(w) > 18 and w[18] in ("A", "T"),
    }


def screen_local(
    seq: str,
    length: int = 21,
    gc_min: float = 30.0,
    gc_max: float = 52.0,
    repeat_n: int = 4,
    length_min: int | None = None,
    length_max: int | None = None,
) -> list[dict]:
    """
    Slide windows of length *length_min*–*length_max* nt across *seq*
    (CDS region recommended).  If length_min/max are not set, uses *length*.
    Apply Phase 2-4 rules and return all candidates that pass hard
    constraints, sorted by soft score descending.

    Hard constraints (any failure → candidate excluded):
      • G/C content 30–52 %
      • No homopolymer run ≥ 4 nt

    Soft criteria (each worth 1 point, max score 6):
      • Starts with AA            (U6/H1 promoter compatibility; AA(xxx)UU motif)
      • Ends with TT              (3' UU overhang stability)
      • Position 1  = G or C      (passenger 5' stability → asymmetric RISC loading)
      • Position 10 = U/T         (Ago2 catalytic alignment)
      • Position 13 ≠ G           (RISC recognition)
      • Position 19 = A or T      (guide 5' end = U or A preferred; Ui-Tei / thermodynamic asymmetry)
    """
    seq = seq.upper()
    candidates: list[dict] = []

    lo = length_min if length_min is not None else length
    hi = length_max if length_max is not None else length

    for ln in range(lo, hi + 1):
        for i in range(len(seq) - ln + 1):
            window = seq[i : i + ln]
            if len(window) < ln:
                break

            # ── Hard constraints ────────────────────────────────────────────
            gc = gc_content(window)
            if not (gc_min <= gc <= gc_max):
                continue
            if has_homopolymer(window, repeat_n):
                continue

            # ── Soft criteria ───────────────────────────────────────────────
            aa_start = window[:2] == "AA"
            uu_end   = window[-2:] == "TT"       # TT in DNA → UU 3' overhang in RNA
            pos      = _check_position_rules(window)

            score = sum([
                aa_start,
                uu_end,
                pos["pos1_GC"],
                pos["pos10_U"],
                pos["pos13_notG"],
                pos["pos19_AU"],
            ])

            candidates.append({
                "position":     f"{i + 1}-{i + ln}",
                "sequence_dna": window,
                "sequence_rna": window.replace("T", "U"),
                "gc_pct":       round(gc, 1),
                "aa_start":     aa_start,
                "uu_end":       uu_end,
                **pos,
                "score":        score,
            })

    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates


def _extract_cds_seq(full_seq: str, cds_start: int, cds_end: int) -> str:
    """Slice 1-based [cds_start, cds_end] from *full_seq* (inclusive)."""
    return full_seq[cds_start - 1 : cds_end]


# ── NCBI helpers ─────────────────────────────────────────────────────────────

def fetch_fasta(accession: str) -> str:
    """Return the raw FASTA sequence string for *accession*."""
    resp = requests.get(
        NCBI_EFETCH,
        params={"db": "nuccore", "id": accession, "rettype": "fasta", "retmode": "text"},
        headers=HEADERS,
        timeout=30,
    )
    resp.raise_for_status()
    return resp.text.strip()


def fetch_genbank(accession: str) -> str:
    """Return the GenBank flat-file text for *accession*."""
    resp = requests.get(
        NCBI_EFETCH,
        params={"db": "nuccore", "id": accession, "rettype": "gb", "retmode": "text"},
        headers=HEADERS,
        timeout=30,
    )
    resp.raise_for_status()
    return resp.text


def parse_cds_range(gb_text: str) -> tuple[int, int] | None:
    """
    Extract CDS start..end positions from a GenBank record.
    Returns 1-based integers or None if not found.
    """
    # Matches lines like:  CDS   883..1602
    # Also handles complement(n..n) and join() — we take the outermost numbers.
    cds_match = re.search(r"^\s+CDS\s+(?:complement\()?(?:join\()?(\d+)\.\.(\d+)", gb_text, re.MULTILINE)
    if cds_match:
        return int(cds_match.group(1)), int(cds_match.group(2))
    return None


# ── siDirect2 helper ─────────────────────────────────────────────────────────

def design_sirna(
    fasta_text: str,
    *,
    species: str = "hs_refseq230",
    gc_min: int = 30,
    gc_max: int = 52,
    cds_start: int | None = None,
    cds_end: int | None = None,
    seed_tm_max: float = 21.5,
) -> list[dict]:
    """
    Submit *fasta_text* to siDirect2 and return a list of candidate dicts.

    Each dict has keys:
        target_position, target_sequence,
        guide_21nt, passenger_21nt,
        functional_selection,
        guide_seed_tm, passenger_seed_tm,
        gc_content,
        guide_off_targets_0, guide_off_targets_1plus, guide_off_targets_2plus,
        guide_off_targets_3plus,
        passenger_off_targets_0, passenger_off_targets_1minus,
        passenger_off_targets_2minus, passenger_off_targets_3minus,
        target_position_constraint, consec_gc_ok, consec_at_ok
    """
    payload: dict = {
        "yourSeq": fasta_text,
        # algorithms
        "uitei": "1",
        # seed duplex filter
        "seedTm": "1",
        "seedTmMax": str(seed_tm_max),
        # specificity check
        "spe": species,
        "hidenonspe": "1",
        "hitcount": "1",
        # GC content
        "percentGC": "1",
        "percentGCMin": str(gc_min),
        "percentGCMax": str(gc_max),
        # avoid repeats
        "consGC": "1",
        "consGCmax": "4",
        "consAT": "1",
        "consATmax": "4",
        # only show siRNAs matching all criteria
        "hide": "1",
    }

    if cds_start is not None and cds_end is not None:
        payload["pos"] = "1"
        payload["posStart"] = str(cds_start)
        payload["posEnd"] = str(cds_end)

    resp = requests.post(
        SIDIRECT2_DESIGN,
        data=payload,
        headers=HEADERS,
        timeout=120,
    )
    resp.raise_for_status()
    return _parse_results(resp.text)


def _parse_results(html: str) -> list[dict]:
    soup = BeautifulSoup(html, "lxml")

    # Find the "Effective siRNA candidates" table
    candidates: list[dict] = []
    tables = soup.find_all("table")
    target_table = None
    for table in tables:
        header_text = table.get_text(" ", strip=True).lower()
        if "target position" in header_text and "guide" in header_text:
            target_table = table
            break

    if target_table is None:
        # Check for "no siRNA" message
        page_text = soup.get_text(" ", strip=True)
        if "no sirna" in page_text.lower() or "0 sirna" in page_text.lower():
            return []
        # Dump a snippet for debugging
        print("[warning] could not locate results table; page snippet:", file=sys.stderr)
        print(page_text[:800], file=sys.stderr)
        return []

    rows = target_table.find_all("tr")
    # First two rows are usually header rows; data starts at row 2 or 3
    data_rows = [r for r in rows if r.find("td")]

    for row in data_rows:
        cells = [td.get_text(" ", strip=True) for td in row.find_all("td")]
        # siDirect2 table columns (as of v2.1):
        # 0  target position
        # 1  target sequence (21nt target + 2nt overhang)
        # 2  guide 21nt  (5'→3')
        # 3  passenger 21nt (5'→3')
        # 4  functional siRNA selection
        # 5  guide seed-duplex Tm
        # 6  passenger seed-duplex Tm
        # 7  specificity: guide mismatches against any off-target (0+)
        # 8  guide 1(+)  ...  (further specificity columns)
        # 15 target position constraint
        # 16 contiguous G's or C's
        # 17 contiguous A's or T's
        # 18 GC content
        if len(cells) < 10:
            continue
        # Skip header rows — real data rows have a numeric range like "908-930"
        if not re.match(r"^\d+[-–]\d+$", cells[0].strip()):
            continue

        def _cell(i: int, default: str = "") -> str:
            return cells[i] if i < len(cells) else default

        candidates.append({
            "target_position":         _cell(0),
            "target_sequence":         _cell(1),
            "guide_21nt":              _cell(2),
            "passenger_21nt":          _cell(3),
            "functional_selection":    _cell(4),
            "guide_seed_tm":           _cell(5),
            "passenger_seed_tm":       _cell(6),
            "guide_off_targets_0plus": _cell(7),
            "guide_off_targets_1plus": _cell(8),
            "guide_off_targets_2plus": _cell(9),
            "guide_off_targets_3plus": _cell(10),
            "pass_off_targets_0minus": _cell(11),
            "pass_off_targets_1minus": _cell(12),
            "pass_off_targets_2minus": _cell(13),
            "pass_off_targets_3minus": _cell(14),
            "target_pos_constraint":   _cell(15),
            "consec_gc":               _cell(16),
            "consec_at":               _cell(17),
            "gc_content":              _cell(18),
        })

    return candidates


# ── CLI ───────────────────────────────────────────────────────────────────────

def _print_novel_targets() -> None:
    width = 74
    print("=" * width)
    print("  RECOMMENDED NOVEL TARGETS  (unchecked — high iGEM innovation value)")
    print("=" * width)
    for gene, info in NOVEL_TARGETS.items():
        print(f"  {gene:<10}  {info['accession']:<14}  {info['description']}")
    print()
    print("  AVOID (approved drugs / advanced clinical programs):")
    print(f"  {', '.join(sorted(ESTABLISHED_TARGETS))}")
    print("=" * width)


def _print_local_summary(candidates: list[dict], top: int = 15) -> None:
    print()
    print(f"{'Pos':<12} {'Sequence (DNA)':<24} {'GC%':<6} "
          f"{'AA':<4} {'TT':<4} {'P1':<5} {'P10':<4} {'P13':<5} {'P19':<6} {'Score'}")
    print("-" * 80)
    for c in candidates[:top]:
        def _yn(v: bool) -> str:
            return "\u2713" if v else "\u00b7"
        seq_display = c['sequence_dna'].lower()
        print(
            f"{c['position']:<12} {seq_display:<24} {c['gc_pct']:<6.1f} "
            f"{_yn(c['aa_start']):<4} {_yn(c['uu_end']):<4} "
            f"{_yn(c['pos1_GC']):<5} {_yn(c['pos10_U']):<4} "
            f"{_yn(c['pos13_notG']):<5} {_yn(c['pos19_AU']):<6} "
            f"{c['score']}/6"
        )
    if len(candidates) > top:
        print(f"  … {len(candidates) - top} more in output CSV")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "siRNA design pipeline: NCBI fetch → local Phase 2-4 screening "
            "→ siDirect2 Phase 5 validation."
        )
    )
    parser.add_argument(
        "accession", nargs="?",
        help="NCBI mRNA accession, e.g. NM_139314  (omit with --list-targets)")
    parser.add_argument("--list-targets", action="store_true",
                        help="Print recommended novel target genes and exit")
    parser.add_argument("--out", default="candidates.csv",
                        help="Output CSV (default: candidates.csv)")
    parser.add_argument("--local-only", action="store_true",
                        help="Run Phase 2-4 local screening only; skip siDirect2")
    parser.add_argument("--species", default="hs_refseq230",
                        help="siDirect2 specificity-check DB (default: hs_refseq230)")
    parser.add_argument("--gc-min", type=float, default=30.0)
    parser.add_argument("--gc-max", type=float, default=52.0)
    parser.add_argument("--length", type=int, default=21,
                        help="siRNA length for local screening (default: 21, overridden by --length-min/max)")
    parser.add_argument("--length-min", type=int, default=None,
                        help="Minimum siRNA length for range scan (default: use --length)")
    parser.add_argument("--length-max", type=int, default=None,
                        help="Maximum siRNA length for range scan (default: use --length)")
    parser.add_argument("--seed-tm-max", type=float, default=21.5)
    parser.add_argument("--cds-start", type=int, default=None,
                        help="Override CDS start (1-based)")
    parser.add_argument("--cds-end", type=int, default=None,
                        help="Override CDS end (1-based)")
    parser.add_argument("--no-cds-filter", action="store_true",
                        help="Screen / submit the whole transcript, not just CDS")
    args = parser.parse_args()

    # ── --list-targets ───────────────────────────────────────────────────────
    if args.list_targets:
        _print_novel_targets()
        return

    if not args.accession:
        parser.error("accession is required (or use --list-targets)")

    # Warn if user picked a well-established target
    gene_upper = args.accession.upper()
    if gene_upper in ESTABLISHED_TARGETS:
        print(f"[warning] {gene_upper} already has approved drugs/advanced clinical "
              f"programs. Consider a novel target (run --list-targets).", file=sys.stderr)

    # ── Step 1: fetch FASTA from NCBI ────────────────────────────────────────
    print(f"[1/4] Fetching FASTA for {args.accession} from NCBI …")
    fasta = fetch_fasta(args.accession)
    if not fasta or fasta.startswith("Error"):
        sys.exit(f"[error] NCBI returned: {fasta[:200]}")
    full_seq = parse_fasta_seq(fasta)
    print(f"      Sequence length: {len(full_seq):,} nt")

    # ── Step 2: locate CDS ───────────────────────────────────────────────────
    cds_start = args.cds_start
    cds_end   = args.cds_end

    if not args.no_cds_filter and cds_start is None:
        print("[2/4] Fetching GenBank record to detect CDS …")
        time.sleep(0.4)   # be polite to NCBI rate limits
        gb = fetch_genbank(args.accession)
        cds = parse_cds_range(gb)
        if cds:
            cds_start, cds_end = cds
            print(f"      CDS annotated: {cds_start}..{cds_end}  "
                  f"({cds_end - cds_start + 1} nt)")
        else:
            # Fall back to first ATG rule
            atg = find_first_atg(full_seq)
            if atg > 0:
                cds_start = atg
                cds_end   = len(full_seq)
                print(f"      CDS not annotated; using first ATG at pos {cds_start}")
            else:
                print("      [warning] No ATG found; screening full transcript")
    else:
        print("[2/4] CDS detection skipped (--no-cds-filter)")

    # ── Step 3: local Phase 2-4 screening ────────────────────────────────────
    lmin = args.length_min if args.length_min is not None else args.length
    lmax = args.length_max if args.length_max is not None else args.length
    length_desc = f"{lmin}-{lmax}" if lmin != lmax else str(lmin)
    print(f"[3/4] Running local Phase 2-4 screening "
          f"(length={length_desc} nt, GC {args.gc_min:.0f}-{args.gc_max:.0f}%) …")

    if cds_start and cds_end and not args.no_cds_filter:
        # Scan from CDS+100 (skip ribosome zone) through end of transcript (incl. 3'UTR)
        scan_start = cds_start + 100
        screen_seq = full_seq[scan_start - 1:]   # to transcript end
        seq_label  = f"CDS+100..3'UTR ({scan_start}..{len(full_seq)})"
        pos_offset = scan_start - 1
    else:
        screen_seq = full_seq
        seq_label  = "full transcript"
        pos_offset = 0

    local_hits = screen_local(
        screen_seq,
        length=args.length,
        gc_min=args.gc_min,
        gc_max=args.gc_max,
        length_min=args.length_min,
        length_max=args.length_max,
    )

    # Adjust positions back to transcript coordinates and annotate region
    for h in local_hits:
        start_str, end_str = h["position"].split("-")
        abs_start = int(start_str) + pos_offset
        abs_end   = int(end_str)   + pos_offset
        h["position"] = f"{abs_start}-{abs_end}"
        # Annotate region relative to CDS
        if cds_start and cds_end:
            if abs_end < cds_start:
                h["region"] = "5UTR"
            elif abs_start > cds_end:
                h["region"] = "3UTR"
            else:
                h["region"] = "CDS"
        else:
            h["region"] = "unknown"

    print(f"      Screened {seq_label}: "
          f"{len(screen_seq):,} nt → {len(local_hits)} candidate(s) pass Phase 2-4")

    if local_hits:
        _print_local_summary(local_hits)

    # ── Step 4 (optional): siDirect2 Phase 5 validation ─────────────────────
    sidirect_hits: list[dict] = []
    if not args.local_only:
        print(f"\n[4/4] Submitting to siDirect2 for Phase 5 off-target validation …")
        # Pass +100 offset to siDirect2 target range to skip ribosome zone
        sd_range_start = (cds_start + 100) if cds_start else None
        sidirect_hits = design_sirna(
            fasta,
            species=args.species,
            gc_min=int(args.gc_min),
            gc_max=int(args.gc_max),
            cds_start=sd_range_start,
            cds_end=None,   # scan to end of transcript (incl. 3'UTR)
            seed_tm_max=args.seed_tm_max,
        )
        print(f"      siDirect2 returned {len(sidirect_hits)} validated candidate(s)")
        if sidirect_hits:
            print()
            print(f"{'Position':<12} {'Guide (21 nt)':<24} {'GC%':<6} {'Seed Tm'}")
            print("-" * 58)
            for c in sidirect_hits:
                print(f"{c['target_position']:<12} {c['guide_21nt']:<24} "
                      f"{c['gc_content']:<6} {c['guide_seed_tm']}")
    else:
        print("\n[4/4] siDirect2 skipped (--local-only)")

    # ── Write output CSV ─────────────────────────────────────────────────────
    if not local_hits and not sidirect_hits:
        print("\n[result] No candidates found.")
        return

    rows: list[dict] = []

    def _parse_pos(pos_str: str) -> tuple[int, int]:
        """Parse '861-881' → (861, 881)."""
        parts = pos_str.replace("–", "-").split("-")
        return int(parts[0]), int(parts[1])

    def _overlaps(pos_a: str, pos_b: str) -> bool:
        """Return True if two position strings share at least 1 nt of overlap."""
        try:
            s1, e1 = _parse_pos(pos_a)
            s2, e2 = _parse_pos(pos_b)
            return s1 <= e2 and s2 <= e1
        except (ValueError, IndexError):
            return False

    # Build list of (target_seq_DNA, guide_seed_tm, passenger_seed_tm) for validated hits
    sidirect_entries = []
    for c in sidirect_hits:
        seq_dna = c["target_sequence"].replace("U", "T").replace("u", "t").upper().split()[0]
        sidirect_entries.append((seq_dna, c.get("guide_seed_tm", ""), c.get("passenger_seed_tm", "")))

    # Cross-reference: annotate local hits with siDirect2 validation flag + Tm values.
    # A local candidate is validated ONLY if its exact DNA sequence is a substring
    # of a siDirect2-validated target sequence (or vice versa for longer candidates).
    # This avoids false positives from mere positional overlap.
    for h in local_hits:
        row = dict(h)
        cand_seq = h["sequence_dna"].upper()
        matched_tm_guide = ""
        matched_tm_pass = ""
        validated = False
        for sd_seq, g_tm, p_tm in sidirect_entries:
            if cand_seq in sd_seq or sd_seq in cand_seq:
                validated = True
                matched_tm_guide = g_tm
                matched_tm_pass = p_tm
                break
        row["sidirect2_validated"] = validated
        row["sidirect2_guide_seed_tm"] = matched_tm_guide
        row["sidirect2_passenger_seed_tm"] = matched_tm_pass
        rows.append(row)

    # Append any siDirect2 hits not already in local list (e.g. if --no-cds-filter)
    local_positions = {h["position"] for h in local_hits}
    for c in sidirect_hits:
        if c["target_position"] not in local_positions:
            rows.append({
                "position":        c["target_position"],
                "sequence_dna":    c["target_sequence"].replace("U", "T"),
                "sequence_rna":    c["target_sequence"],
                "gc_pct":          c["gc_content"],
                "aa_start":        "",
                "uu_end":          "",
                "pos1_A":     "",
                "pos10_U":    "",
                "pos13_notG": "",
                "pos19_A":    "",
                "score":           "",
                "sidirect2_validated": True,
                "sidirect2_guide_seed_tm":     c.get("guide_seed_tm", ""),
                "sidirect2_passenger_seed_tm": c.get("passenger_seed_tm", ""),
                "guide_21nt":      c["guide_21nt"],
                "passenger_21nt":  c["passenger_21nt"],
                "guide_off_targets_0plus": c.get("guide_off_targets_0plus", ""),
            })

    print(f"\n[result] Writing {len(rows)} total row(s) → {args.out}")
    fieldnames = list(rows[0].keys())
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    # Summary
    validated = sum(1 for r in rows if r.get("sidirect2_validated") is True)
    high_score = sum(1 for r in rows if isinstance(r.get("score"), int) and r["score"] >= 5)
    print(f"       Phase 2-4 pass:      {len(local_hits)}")
    print(f"       Score \u22655/6 (optimal): {high_score}")
    if not args.local_only:
        print(f"       siDirect2 validated: {validated}")


if __name__ == "__main__":
    main()
