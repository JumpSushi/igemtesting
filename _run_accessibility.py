import RNA, requests, csv

NCBI_EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
HEADERS = {"User-Agent": "igem-26/siRNA-validator"}

GENES = [
    ("ASGR1",      "scan_asgr1.csv",      "NM_001671"),
    ("ASGR1-V2",   "scan_asgr1_v2.csv",   "NM_001445022.1"),
    ("ASGR1-ISOB", "scan_asgr1_isob.csv", "NM_001197216.3"),
    ("APOA5",      "scan_apoa5.csv",       "NM_052968.5"),
    ("APOA5-V2",   "scan_apoa5_v2.csv",   "NM_001371904.1"),
    ("APOA5-V3",   "scan_apoa5_v3.csv",   "NM_001166598.2"),
]

def fetch_seq(acc):
    r = requests.get(NCBI_EFETCH, params={"db":"nuccore","id":acc,"rettype":"fasta","retmode":"text"}, headers=HEADERS, timeout=30)
    return "".join(l for l in r.text.splitlines() if not l.startswith(">")).upper()

def rnafold_unpaired(seq):
    fc = RNA.fold_compound(seq)
    fc.pf()
    bppm = fc.bpp()
    n = len(seq)
    return [1.0 - min(sum(bppm[i][j] for j in range(1, n+1) if j != i), 1.0) for i in range(1, n+1)]

def best_per_region(rows):
    def ps(r):
        try: return int(r["position"].split("-")[0])
        except: return 9999
    rows = sorted(rows, key=ps)
    regions = []
    last = -999
    for r in rows:
        s = ps(r)
        sc = int(r["score"]) if r.get("score") else 0
        ln = len(r["sequence_dna"])
        if s - last > 10:
            regions.append(r)
            last = s
        else:
            psc = int(regions[-1]["score"]) if regions[-1].get("score") else 0
            pln = len(regions[-1]["sequence_dna"])
            if sc > psc or (sc == psc and ln < pln):
                regions[-1] = r
                last = s
    return regions

for gene, csvfile, acc in GENES:
    print(f"\n=== {gene} ({acc}) ===")
    with open(csvfile) as f:
        rows = list(csv.DictReader(f))
    validated = [r for r in rows if r.get("sidirect2_validated","").strip().lower() in ("true","1")]
    best = best_per_region(validated)
    print(f"  {len(validated)} validated → {len(best)} region(s). Fetching + folding...")
    mrna = fetch_seq(acc)
    unpaired = rnafold_unpaired(mrna)
    print(f"  {'Sequence':<28}  {'Position':>12}  {'Score':>5}  {'Access':>7}  Label")
    print(f"  {'-'*70}")
    for r in best:
        sd = r["sequence_dna"].upper().replace("U","T")
        idx = mrna.find(sd)
        if idx == -1:
            print(f"  {r['sequence_dna'].lower():<28}  {'NOT FOUND':>12}")
            continue
        av = sum(unpaired[idx:idx+len(sd)]) / len(sd)
        lbl = "very open" if av >= 0.7 else "moderate" if av >= 0.5 else "partial" if av >= 0.3 else "buried"
        print(f"  {r['sequence_dna'].lower():<28}  {str(idx+1)+'-'+str(idx+len(sd)):>12}  {r.get('score','?'):>5}  {av:>7.3f}  {lbl}")
