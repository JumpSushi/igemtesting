"""BLAST top shared APOA5 candidates (score>=4) against Homo sapiens refseq_rna."""
import requests, time, re

BLAST_URL = "https://blast.ncbi.nlm.nih.gov/blast/Blast.cgi"
HEADERS = {"User-Agent": "igem-26/siRNA-validator bioinfo@example.com"}

# Top shared candidates, score>=4, one per region
CANDIDATES = [
    ("pos~101", "GGCTCTTCTTTCAGCGTTT",       "score=5"),
    ("pos~203", "AAGACAGCCTTGAGCAAGA",        "score=4"),
    ("pos~254", "CTCAACAATATGAACAAGTT",       "score=4 siD2✓"),
    ("pos~452", "CCTACACGATGGATCTGAT",        "score=4"),
    ("pos~628", "AAAGAGCTCTTCCACCCAT",        "score=4"),
    ("pos~1048","AACAGACAGTGGCAAGGTT",        "score=5"),
    ("pos~1101","AGACATCACTCACAGCCTT",        "score=4"),
]

def submit(seq):
    r = requests.post(BLAST_URL, data={
        "CMD": "Put", "PROGRAM": "blastn", "DATABASE": "refseq_rna",
        "QUERY": seq, "ENTREZ_QUERY": "Homo sapiens[Organism]",
        "WORD_SIZE": 7, "EXPECT": 10, "FORMAT_TYPE": "Text",
        "SHORT_QUERY_ADJUST": "true", "HITLIST_SIZE": 20,
    }, headers=HEADERS)
    m = re.search(r"RID = (\w+)", r.text)
    return m.group(1) if m else None

def poll(rid):
    for _ in range(25):
        time.sleep(12)
        r = requests.get(BLAST_URL, params={
            "CMD": "Get", "RID": rid, "FORMAT_TYPE": "Text", "HITLIST_SIZE": 20
        }, headers=HEADERS)
        if "Status=WAITING" in r.text:
            continue
        if "Status=FAILED" in r.text or "Status=UNKNOWN" in r.text:
            return None
        return r.text
    return None

def parse_hits(text, seq):
    lines = text.splitlines()
    hits = []
    in_hits = False
    for line in lines:
        if "Sequences producing" in line:
            in_hits = True
            continue
        if in_hits:
            if line.strip() == "":
                if hits:
                    break
                continue
            # parse tabular hit line: description ... bits  evalue  ident
            # Format: NM_xxx.x Description...   bits  evalue  ident%
            m = re.match(r"^(\S+\s+.+?)\s{2,}(\S+)\s+(\S+)\s+(\S+)\s*$", line)
            if m:
                hits.append({
                    "desc": m.group(1).strip()[:65],
                    "bits": m.group(2),
                    "evalue": m.group(3),
                    "ident": m.group(4),
                })
    return hits

# Submit all at once
print("Submitting BLAST jobs...")
jobs = []
for region, seq, label in CANDIDATES:
    rid = submit(seq)
    print(f"  {region} {seq} → RID={rid}  [{label}]")
    jobs.append((region, seq, label, rid))
    time.sleep(3)  # polite delay between submissions

print("\nWaiting for results (polling every 12s)...\n")

for region, seq, label, rid in jobs:
    if not rid:
        print(f"[{region}] SUBMISSION FAILED")
        continue
    print(f"{'='*80}")
    print(f"APOA5 {region}  seq={seq}  [{label}]")
    text = poll(rid)
    if not text:
        print("  TIMEOUT or FAILED")
        continue
    hits = parse_hits(text, seq)
    if not hits:
        print("  No significant hits (or parse error)")
        # print raw snippet
        for line in text.splitlines():
            if "No significant" in line or "Sequences producing" in line:
                print(" ", line)
    else:
        print(f"  {'#':<3} {'Description':<65} {'E-val':>8}  {'Ident':>6}")
        print(f"  {'-'*90}")
        for i, h in enumerate(hits[:12], 1):
            # highlight non-APOA5
            flag = "  "
            if "APOA5" not in h["desc"] and "apolipoprotein A-V" not in h["desc"].lower():
                flag = "!!"
            print(f"  {i:<3} {h['desc']:<65} {h['evalue']:>8}  {h['ident']:>6}  {flag}")
    print()
