#!/usr/bin/env python3


import base64
import io
import json as _json
import math
import os
import re
import threading
import time as _time
import traceback
import uuid
from pathlib import Path

import requests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import tifffile
from flask import Flask, jsonify, render_template, request, session, redirect, url_for
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename

from scrape_sirna import (
    fetch_fasta,
    parse_fasta_seq,
    fetch_genbank,
    parse_cds_range,
    find_first_atg,
    screen_local,
    design_sirna,
    NOVEL_TARGETS,
    HEADERS as _SCRAPE_HEADERS,
)

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
from testgen import make_image

# flask app config
app = Flask(__name__)

class _JsonCantHandleNumpyTypes(app.json_provider_class):
    def default(self, o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)

app.json_provider_class = _JsonCantHandleNumpyTypes
app.json = _JsonCantHandleNumpyTypes(app)
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024  # max upload: 200MB
app.secret_key = os.environ.get("FLASK_SECRET", "igem26-dev-secret-change-in-prod")

_PASSWORD_HASH = generate_password_hash("wowwhatasecurepassword")

@app.before_request
def _require_login():
    if request.endpoint in ("login", "static"):
        return
    if not session.get("authed"):
        if request.path.startswith("/api/"):
            return jsonify({"error": "Unauthorized"}), 401
        return redirect(url_for("login", next=request.path))

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {".tif", ".tiff"}


def _dont_get_path_traversaled(filename: str) -> Path:
    # sanitize & validate file path
    cleaned_name = secure_filename(filename)
    if not cleaned_name:
        raise ValueError("Invalid filename.")
    file_extension = Path(cleaned_name).suffix.lower()
    if file_extension and file_extension not in ALLOWED_EXTENSIONS and not cleaned_name.endswith(".csv"):
        raise ValueError(f"File type not allowed: {file_extension}")
    return (UPLOAD_DIR / cleaned_name).resolve()


# pipeline wrapper

def _do_the_science(channels: np.ndarray, gene_lookup: dict, knobs: dict):
    # run full decode pipeline, return results
    round_things = detect_beads(
        channels,
        min_sigma=knobs["min_sigma"],
        max_sigma=knobs["max_sigma"],
        threshold=knobs["blob_threshold"],
        overlap=knobs["overlap"],
    )

    if len(round_things) == 0:
        return None, None, None, None, (
            "No beads detected. Try lowering the LoG threshold "
            "(e.g. 0.02) or adjusting Min/Max sigma."
        )

    brightness_vals = measure_channel_intensities(
        channels, round_things, aperture_factor=knobs["aperture"])
    on_off      = intensities_to_binary(brightness_vals, method=knobs["threshold_method"])
    codes       = binary_to_code(on_off)
    gene_names  = lookup_sirna(codes, gene_lookup)

    n           = channels.shape[0]
    labels      = CHANNEL_LABELS[:n]

    bead_data = []
    for i, (dot, ints, c, gname) in enumerate(
            zip(round_things, brightness_vals, codes, gene_names)):
        entry: dict = {
            "bead_id":   i,
            "row_px":    round(float(dot[0]), 1),
            "col_px":    round(float(dot[1]), 1),
            "radius_px": round(float(dot[2]), 1),
            "code":      c,
            "siRNA":     gname,
        }
        for lbl, v in zip(labels, ints):
            entry[f"{lbl}"] = round(float(v), 5)
        bead_data.append(entry)

    table = pd.DataFrame(bead_data)
    pic   = _put_circles_on_it(channels, round_things, codes, gene_names)
    return table, pic, round_things, codes, None


def _put_circles_on_it(channels, round_things, codes, gene_names) -> str:
    # draw circles & labels on image
    squished = channels.max(axis=0)
    top = np.percentile(squished, 99.5)

    fig, ax = plt.subplots(figsize=(10, 8), facecolor="#0d0d14")
    ax.set_facecolor("#0d0d14")
    ax.imshow(squished, cmap="inferno", vmin=0, vmax=top, interpolation="nearest")
    ax.axis("off")

    all_genes = sorted(set(gene_names))
    n         = max(2, len(all_genes))
    try:
        cm = matplotlib.colormaps.get_cmap("tab20").resampled(n)
    except AttributeError:
        cm = plt.cm.get_cmap("tab20", n)
    gene_colors = {sid: cm(i) for i, sid in enumerate(all_genes)}

    for (row, col, radius), c, gname in zip(round_things, codes, gene_names):
        color = gene_colors[gname]
        ax.add_patch(mpatches.Circle(
            (col, row), radius * 1.5,
            linewidth=0.9, edgecolor=color, facecolor="none", alpha=0.9))
        ax.text(col + radius * 1.8, row, c,
                color=color, fontsize=4, va="center", fontfamily="monospace")

    swatches = [mpatches.Patch(color=gene_colors[s], label=s) for s in all_genes]
    ax.legend(swatches, [s.get_label() for s in swatches], loc="upper right", fontsize=5,
              ncol=max(1, len(swatches) // 12),
              framealpha=0.6, facecolor="#1a1a2e",
              edgecolor="#333355", labelcolor="white")

    fig.tight_layout(pad=0.3)
    b = io.BytesIO()
    fig.savefig(b, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    b.seek(0)
    return base64.b64encode(b.read()).decode("ascii")


def _grayscale_viewer_thingy(channels: np.ndarray, labels: list) -> list:
    # render each channel as grayscale
    pics = []
    for ch, lbl in zip(channels, labels):
        top = np.percentile(ch, 99.5)
        fig, ax = plt.subplots(figsize=(2.4, 2.4), facecolor="#0d0d14")
        ax.set_facecolor("#0d0d14")
        ax.imshow(ch, cmap="gray", vmin=0, vmax=max(float(top), 1e-6),
                  interpolation="nearest")
        ax.set_title(lbl, color="#ccccdd", fontsize=9, pad=4)
        ax.axis("off")
        fig.tight_layout(pad=0.2)
        b = io.BytesIO()
        fig.savefig(b, format="png", dpi=110, bbox_inches="tight")
        plt.close(fig)
        b.seek(0)
        pics.append(base64.b64encode(b.read()).decode("ascii"))
    return pics


def _grab_all_the_knobs(source) -> dict:
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

@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        pw = request.form.get("password", "")
        if check_password_hash(_PASSWORD_HASH, pw):
            session["authed"] = True
            return redirect(request.args.get("next") or "/")
        error = "Incorrect password."
    return render_template("login.html", error=error)

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

@app.route("/")
def homepage():
    known_genes = sorted(NOVEL_TARGETS.keys())
    return render_template("designer.html", known_genes=known_genes)


@app.route("/old")
def old_homepage():
    return render_template("index.html")


@app.route("/api/generate", methods=["POST"])
def cook_fake_data():
    try:
        stuff     = request.get_json(silent=True) or {}
        count     = max(1, min(int(stuff.get("n_beads", 150)), 2000))
        fuzziness = float(stuff.get("noise",   0.03))
        rng       = int(stuff.get("seed",      42))

        fake_img, answers = make_image(n_beads=count, noise_level=fuzziness, seed=rng)

        tif_loc      = _dont_get_path_traversaled("test_image.tif")
        answer_sheet = _dont_get_path_traversaled("test_image_truth.csv")
        tifffile.imwrite(str(tif_loc), fake_img, photometric="minisblack")
        answers.to_csv(str(answer_sheet), index=False)

        knobs = _grab_all_the_knobs({})
        table, pic, round_things, codes, err = _do_the_science(fake_img, DEFAULT_SIRNA_MAP, knobs)
        if err:
            return jsonify({"error": err}), 400

        # Compare with truth
        gene_names = [DEFAULT_SIRNA_MAP.get(c, "UNKNOWN") for c in codes]
        score      = match_beads_to_truth(round_things, codes, gene_names, answers)

        n      = fake_img.shape[0]
        labels = CHANNEL_LABELS[:n]
        return jsonify({
            "beads":        table.to_dict(orient="records"),
            "image":        pic,
            "channels_b64": _grayscale_viewer_thingy(fake_img, labels),
            "ch_labels":    labels,
            "n_beads":      len(table),
            "n_codes":      table["siRNA"].nunique(),
            "session_file": "test_image.tif",
            "comparison":   score,
            "has_truth":    True,
        })
    except Exception:
        return jsonify({"error": traceback.format_exc()}), 500


@app.route("/api/run", methods=["POST"])
def actually_analyze():
    try:
        knobs    = _grab_all_the_knobs(request.form)
        chans    = request.form.get("channels", "0 1 2 3 4")
        nums     = [int(x) for x in chans.split() if x.strip().isdigit()]
        if not nums:
            return jsonify({"error": "Invalid channel indices."}), 400

        # optional sirna map upload
        gene_lookup = DEFAULT_SIRNA_MAP
        gene_file   = request.files.get("sirna_map")
        if gene_file and gene_file.filename:
            gene_path = _dont_get_path_traversaled(gene_file.filename)
            gene_file.save(str(gene_path))
            gene_lookup = load_sirna_map(gene_path)

        # file upload or use session file
        tiff_upload = request.files.get("image")
        if tiff_upload and tiff_upload.filename:
            file_extension = Path(tiff_upload.filename).suffix.lower()
            if file_extension not in ALLOWED_EXTENSIONS:
                return jsonify({"error": "Only .tif / .tiff files are accepted."}), 400
            tif_loc = _dont_get_path_traversaled(tiff_upload.filename)
            tiff_upload.save(str(tif_loc))
        else:
            old_tif = request.form.get("session_file", "")
            if not old_tif:
                return jsonify({"error": "No image provided."}), 400
            tif_loc = _dont_get_path_traversaled(old_tif)

        if not tif_loc.exists():
            return jsonify({"error": f"Image not found: {tif_loc.name}"}), 400

        channels                              = load_multichannel_tiff(tif_loc, nums)
        table, pic, round_things, codes, err  = _do_the_science(channels, gene_lookup, knobs)
        if err:
            return jsonify({"error": err}), 400

        n      = channels.shape[0]
        labels = CHANNEL_LABELS[:n]

        # Check if truth file exists
        score        = None
        found_answers = False
        answer_sheet  = UPLOAD_DIR / "test_image_truth.csv"
        if answer_sheet.exists():
            try:
                answers    = load_truth_csv(answer_sheet)
                gene_names = [gene_lookup.get(c, "UNKNOWN") for c in codes]
                score      = match_beads_to_truth(round_things, codes, gene_names, answers)
                found_answers = True
            except Exception as e:
                print(f"Warning: Could not load truth data: {e}")

        response = {
            "beads":        table.to_dict(orient="records"),
            "image":        pic,
            "channels_b64": _grayscale_viewer_thingy(channels, labels),
            "ch_labels":    labels,
            "n_beads":      len(table),
            "n_codes":      table["siRNA"].nunique(),
            "session_file": tif_loc.name,
        }

        if found_answers and score:
            response["comparison"] = score
            response["has_truth"]  = True

        return jsonify(response)
    except Exception:
        return jsonify({"error": traceback.format_exc()}), 500


# ── siRNA candidate viewer ────────────────────────────────────────────────────

_GENE_FILES = {
    "ASGR1":        "scan_asgr1.csv",
    "ASGR1-V2":     "scan_asgr1_v2.csv",
    "ASGR1-ISOB":   "scan_asgr1_isob.csv",
    "APOA5":        "scan_apoa5.csv",
    "APOA5-V2":     "scan_apoa5_v2.csv",
    "APOA5-V3":     "scan_apoa5_v3.csv",
}

_GENE_ACCESSIONS = {
    "ASGR1":        "NM_001671",
    "ASGR1-V2":     "NM_001445022.1",
    "ASGR1-ISOB":   "NM_001197216.3",
    "APOA5":        "NM_052968.5",
    "APOA5-V2":     "NM_001371904.1",
    "APOA5-V3":     "NM_001166598.2",
}

@app.route("/sirna")
def sirna_viewer():
    return render_template("sirna.html", genes=list(_GENE_FILES.keys()), accessions=_GENE_ACCESSIONS)


@app.route("/isoforms")
def isoforms_viewer():
    return render_template("isoforms.html")


@app.route("/api/sirna/<gene>/fasta")
def sirna_fasta(gene: str):
    gene = gene.upper()
    if gene not in _GENE_ACCESSIONS:
        return jsonify({"error": f"Unknown gene: {gene}"}), 404
    accession = _GENE_ACCESSIONS[gene]
    try:
        resp = requests.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
            params={"db": "nuccore", "id": accession, "rettype": "fasta", "retmode": "text"},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.text, 200, {"Content-Type": "text/plain"}
    except Exception as e:
        return jsonify({"error": str(e)}), 502


@app.route("/api/sirna/<gene>")
def sirna_data(gene: str):
    gene = gene.upper()
    if gene not in _GENE_FILES:
        return jsonify({"error": f"Unknown gene: {gene}"}), 404
    csv_path = Path(_GENE_FILES[gene])
    if not csv_path.exists():
        return jsonify({"error": f"No scan file found for {gene}. Run scrape_sirna.py first."}), 404
    try:
        df = pd.read_csv(csv_path)
        # normalise legacy column name (old scans used pos19_notGC, new scans use pos19_A)
        if "pos19_notGC" in df.columns and "pos19_A" not in df.columns:
            df = df.rename(columns={"pos19_notGC": "pos19_A"})
        # normalise column types
        for col in ["aa_start", "uu_end", "pos1_A", "pos10_U", "pos13_notG", "pos19_A", "sidirect2_validated"]:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: bool(x) if pd.notna(x) else False)
        if "score" in df.columns:
            df["score"] = pd.to_numeric(df["score"], errors="coerce")
        records = df.where(pd.notna(df), None).to_dict(orient="records")
        # Replace any remaining float NaN/inf (not caught by pd.notna) with None
        import math
        def sanitize(v):
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                return None
            return v
        records = [{k: sanitize(v) for k, v in row.items()} for row in records]
        validated = int(df["sidirect2_validated"].sum()) if "sidirect2_validated" in df.columns else 0
        score5 = int((df["score"] >= 5).sum()) if "score" in df.columns else 0
        score6 = int((df["score"] >= 6).sum()) if "score" in df.columns else 0
        return jsonify({
            "gene": gene,
            "total": len(df),
            "validated": validated,
            "score5": score5,
            "score6": score6,
            "records": records,
        })
    except Exception:
        return jsonify({"error": traceback.format_exc()}), 500


# ── siRNA Designer ───────────────────────────────────────────────────────────

DESIGNER_SAVE_DIR = Path("designer_saved")
DESIGNER_SAVE_DIR.mkdir(exist_ok=True)

_designer_jobs: dict = {}   # job_id → {status, log, result, error, gene}


def _search_gene_isoforms(gene_name: str, organism: str = "Homo sapiens") -> list[dict]:
    """Search NCBI nuccore for RefSeq NM_ mRNA isoforms of *gene_name*."""
    term = f"{gene_name}[Gene Name] AND {organism}[Organism] AND mRNA[Filter]"
    try:
        r1 = requests.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
            params={"db": "nuccore", "term": term, "retmode": "json", "retmax": 30},
            headers=_SCRAPE_HEADERS,
            timeout=30,
        )
        r1.raise_for_status()
        ids = r1.json().get("esearchresult", {}).get("idlist", [])
        if not ids:
            return []

        _time.sleep(0.4)
        r2 = requests.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi",
            params={"db": "nuccore", "id": ",".join(ids), "retmode": "json"},
            headers=_SCRAPE_HEADERS,
            timeout=30,
        )
        r2.raise_for_status()
        summaries = r2.json().get("result", {})

        isoforms = []
        for uid in ids:
            s = summaries.get(uid, {})
            if not isinstance(s, dict):
                continue
            acc = s.get("accessionversion", "")
            if acc.startswith(("NM_", "NR_")):
                isoforms.append({
                    "uid":       uid,
                    "accession": acc,
                    "title":     s.get("title", ""),
                    "length":    s.get("slen", 0),
                })
        return isoforms
    except Exception:
        return []


def _log_job(job_id: str, step: str, msg: str) -> None:
    if job_id in _designer_jobs:
        _designer_jobs[job_id]["log"].append({"step": step, "msg": msg})


def _run_pipeline_job(
    job_id: str,
    gene_name: str,
    accessions: list,
    species: str,
    gc_min: float = 30.0,
    gc_max: float = 52.0,
) -> None:
    """Background thread: run full siRNA design pipeline for every accession."""
    job = _designer_jobs[job_id]
    all_records: list[dict] = []

    try:
        for acc in accessions:
            # ── 1. Fetch FASTA ───────────────────────────────────────────────
            _log_job(job_id, f"fetch_{acc}", f"Fetching FASTA for {acc} from NCBI…")
            fasta    = fetch_fasta(acc)
            full_seq = parse_fasta_seq(fasta)
            _log_job(job_id, f"fetch_{acc}_done",
                     f"{acc}: {len(full_seq):,} nt")
            _time.sleep(0.4)

            # ── 2. Detect CDS ────────────────────────────────────────────────
            _log_job(job_id, f"cds_{acc}", f"Fetching GenBank record for {acc}…")
            gb  = fetch_genbank(acc)
            cds = parse_cds_range(gb)
            if cds:
                cds_start, cds_end = cds
                _log_job(job_id, f"cds_{acc}_done",
                         f"{acc}: CDS {cds_start}..{cds_end}")
                scan_start = cds_start + 100
            else:
                atg = find_first_atg(full_seq)
                if atg > 0:
                    cds_start = atg
                    cds_end   = len(full_seq)
                    scan_start = atg + 100
                    _log_job(job_id, f"cds_{acc}_done",
                             f"{acc}: no CDS annotated — using first ATG at {atg}")
                else:
                    cds_start = cds_end = None
                    scan_start = 1
                    _log_job(job_id, f"cds_{acc}_done",
                             f"{acc}: no ATG found — screening full transcript")

            screen_seq = full_seq[scan_start - 1:]
            pos_offset = scan_start - 1
            _time.sleep(0.4)

            # ── 3. Local Phase 2-4 screening ─────────────────────────────────
            _log_job(job_id, f"screen_{acc}",
                     f"Local Phase 2-4 screening {acc} ({len(screen_seq):,} nt)…")
            local_hits = screen_local(
                screen_seq,
                gc_min=gc_min,
                gc_max=gc_max,
                length_min=19,
                length_max=27,
            )

            for h in local_hits:
                s, e = h["position"].split("-")
                abs_s = int(s) + pos_offset
                abs_e = int(e) + pos_offset
                h["position"]  = f"{abs_s}-{abs_e}"
                h["accession"] = acc
                if cds_start and cds_end:
                    if abs_e < cds_start:    h["region"] = "5UTR"
                    elif abs_s > cds_end:    h["region"] = "3UTR"
                    else:                    h["region"] = "CDS"
                else:
                    h["region"] = "unknown"

            _log_job(job_id, f"screen_{acc}_done",
                     f"{acc}: {len(local_hits)} candidate(s) pass Phase 2-4")

            # ── 4. siDirect2 Phase 5 validation ──────────────────────────────
            _log_job(job_id, f"sidirect_{acc}",
                     f"Submitting {acc} to siDirect2 (off-target validation)…")
            sd_range_start = (cds_start + 100) if cds_start else None
            try:
                sd_hits = design_sirna(
                    fasta,
                    species=species,
                    gc_min=int(gc_min),
                    gc_max=int(gc_max),
                    cds_start=sd_range_start,
                    cds_end=None,
                )
                _log_job(job_id, f"sidirect_{acc}_done",
                         f"{acc}: {len(sd_hits)} siDirect2-validated candidate(s)")
            except Exception as sd_err:
                sd_hits = []
                _log_job(job_id, f"sidirect_{acc}_warn",
                         f"{acc}: siDirect2 error — {str(sd_err)[:80]}")

            # ── 5. Cross-reference ────────────────────────────────────────────
            sd_entries = []
            for c in sd_hits:
                seq_dna = c["target_sequence"].replace("U", "T").upper().split()[0]
                sd_entries.append((seq_dna, c.get("guide_seed_tm", ""),
                                   c.get("passenger_seed_tm", "")))

            rows: list[dict] = []
            for h in local_hits:
                row      = dict(h)
                cand_seq = h["sequence_dna"].upper()
                validated, g_tm, p_tm = False, "", ""
                for sd_seq, gt, pt in sd_entries:
                    if cand_seq in sd_seq or sd_seq in cand_seq:
                        validated, g_tm, p_tm = True, gt, pt
                        break
                row["sidirect2_validated"]          = validated
                row["sidirect2_guide_seed_tm"]      = g_tm
                row["sidirect2_passenger_seed_tm"]  = p_tm
                rows.append(row)

            local_positions = {h["position"] for h in local_hits}
            for c in sd_hits:
                if c["target_position"] not in local_positions:
                    rows.append({
                        "accession":                  acc,
                        "position":                   c["target_position"],
                        "sequence_dna":               c["target_sequence"].replace("U", "T"),
                        "sequence_rna":               c["target_sequence"],
                        "gc_pct":                     c["gc_content"],
                        "aa_start":                   None,
                        "uu_end":                     None,
                        "pos1_GC":                    None,
                        "pos10_U":                    None,
                        "pos13_notG":                 None,
                        "pos19_AU":                   None,
                        "score":                      None,
                        "region":                     "unknown",
                        "sidirect2_validated":        True,
                        "sidirect2_guide_seed_tm":    c.get("guide_seed_tm", ""),
                        "sidirect2_passenger_seed_tm":c.get("passenger_seed_tm", ""),
                        "guide_21nt":                 c.get("guide_21nt", ""),
                        "passenger_21nt":             c.get("passenger_21nt", ""),
                    })

            all_records.extend(rows)
            _time.sleep(0.4)

        _enrich_seed_tm(all_records)
        job["result"] = all_records
        job["status"] = "done"
        _log_job(job_id, "done",
                 f"Pipeline complete — {len(all_records)} total candidate(s) "
                 f"across {len(accessions)} accession(s)")

    except Exception:
        err = traceback.format_exc()
        job["error"]  = err
        job["status"] = "error"
        _log_job(job_id, "error", f"Pipeline failed: {err[:200]}")


def _calc_seed_tm(sequence_dna: str) -> float | None:
    """
    Estimate guide seed-duplex Tm for a ≥20 nt target sequence (DNA notation).
    Seed = positions 2-8 of the guide strand (antisense, 5'→3').
    Uses the Wallace-rule approximation: Tm = 2*(A+U) + 4*(G+C).
    Matches the siDirect2 threshold convention (default cutoff 21.5 °C).
    """
    seq = sequence_dna.upper().replace("U", "T")
    if len(seq) < 20:
        return None
    _comp = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G'}
    guide = ''.join(_comp.get(b, 'N') for b in reversed(seq))
    seed  = guide[1:8]        # positions 2-8 (1-based), 0-indexed [1:8]
    if len(seed) < 7:
        return None
    au = seed.count('A') + seed.count('T')
    gc = seed.count('G') + seed.count('C')
    return round(2.0 * au + 4.0 * gc, 1)


def _enrich_seed_tm(records: list) -> list:
    """Fill sidirect2_guide_seed_tm for every record that lacks one."""
    for r in records:
        if not r.get('sidirect2_guide_seed_tm'):
            seq = r.get('sequence_dna') or ''
            tm  = _calc_seed_tm(seq)
            r['sidirect2_guide_seed_tm'] = str(tm) if tm is not None else ''
    return records


def _sanitize_records(records: list) -> list:
    """Replace float NaN/Inf with None for JSON serialisation."""
    def _s(v):
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return None
        return v
    return [{k: _s(v) for k, v in row.items()} for row in records]


@app.route("/designer")
def designer_page():
    return redirect(url_for("homepage"))


@app.route("/api/designer/search", methods=["POST"])
def designer_search():
    data = request.get_json(silent=True) or {}
    raw  = (data.get("gene") or "").strip()
    gene = re.sub(r"[^A-Z0-9_\-]", "", raw.upper())
    if not gene:
        return jsonify({"error": "Gene name required"}), 400
    try:
        isoforms = _search_gene_isoforms(gene)
        return jsonify({"gene": gene, "isoforms": isoforms})
    except Exception:
        return jsonify({"error": traceback.format_exc()}), 500


@app.route("/api/designer/run", methods=["POST"])
def designer_run():
    data = request.get_json(silent=True) or {}
    gene = re.sub(r"[^A-Z0-9_\-]", "", (data.get("gene") or "").strip().upper())
    accs = [
        re.sub(r"[^A-Z0-9_.\-]", "", a.strip())
        for a in (data.get("accessions") or [])
        if isinstance(a, str)
    ]
    allowed_species = {"hs_refseq230", "mm_refseq230", "hs_ensembl", "mm_ensembl"}
    species = data.get("species", "hs_refseq230")
    if species not in allowed_species:
        species = "hs_refseq230"
    if not gene or not accs:
        return jsonify({"error": "gene and accessions are required"}), 400

    job_id = str(uuid.uuid4())[:8]
    _designer_jobs[job_id] = {
        "status": "running", "log": [],
        "result": None, "error": None, "gene": gene,
    }
    threading.Thread(
        target=_run_pipeline_job,
        args=(job_id, gene, accs, species),
        daemon=True,
    ).start()
    return jsonify({"job_id": job_id})


@app.route("/api/designer/job/<job_id>")
def designer_job_status(job_id: str):
    safe_id = re.sub(r"[^a-f0-9\-]", "", job_id.lower())
    job = _designer_jobs.get(safe_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    result = _sanitize_records(job["result"]) if job["result"] else None
    return jsonify({
        "status": job["status"],
        "log":    job["log"],
        "result": result,
        "error":  job.get("error"),
        "gene":   job.get("gene"),
    })


@app.route("/api/designer/save", methods=["POST"])
def designer_save():
    data = request.get_json(silent=True) or {}
    gene = re.sub(r"[^A-Z0-9_\-]", "", (data.get("gene") or "").strip().upper())
    if not gene:
        return jsonify({"error": "Gene name required"}), 400
    records = data.get("records", [])
    if not isinstance(records, list):
        return jsonify({"error": "records must be a list"}), 400

    import datetime
    name_label = re.sub(r"[^A-Z0-9_\-]", "", (data.get("save_label") or "").strip().upper())
    filename = f"{gene}_{name_label}" if name_label else gene
    save_obj = {
        "gene":       gene,
        "species":    re.sub(r"[^a-z0-9_]", "", str(data.get("species", "hs_refseq230"))),
        "accessions": [re.sub(r"[^A-Z0-9_.\-]", "", str(a)) for a in (data.get("accessions") or [])],
        "records":    records,
        "saved_at":   datetime.datetime.utcnow().isoformat(),
    }
    save_path = (DESIGNER_SAVE_DIR / f"{filename}.json").resolve()
    if not str(save_path).startswith(str(DESIGNER_SAVE_DIR.resolve())):
        return jsonify({"error": "Invalid path"}), 400
    with open(save_path, "w") as fh:
        _json.dump(save_obj, fh)
    return jsonify({"ok": True, "name": filename})


@app.route("/api/designer/delete/<name>", methods=["DELETE"])
def designer_delete(name: str):
    safe_name = re.sub(r"[^A-Z0-9_\-]", "", name.upper())
    del_path = (DESIGNER_SAVE_DIR / f"{safe_name}.json").resolve()
    if not str(del_path).startswith(str(DESIGNER_SAVE_DIR.resolve())):
        return jsonify({"error": "Invalid path"}), 400
    if not del_path.exists():
        return jsonify({"error": "Not found"}), 404
    del_path.unlink()
    return jsonify({"ok": True})


@app.route("/api/designer/fasta/<accession>")
def designer_fasta(accession: str):
    """Fetch FASTA for a given NCBI accession and return as plain text."""
    safe_acc = re.sub(r"[^A-Za-z0-9_\.\-]", "", accession)
    if not safe_acc:
        return jsonify({"error": "Invalid accession"}), 400
    try:
        resp = requests.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
            params={"db": "nuccore", "id": safe_acc, "rettype": "fasta", "retmode": "text"},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.text, 200, {"Content-Type": "text/plain"}
    except Exception as e:
        return jsonify({"error": str(e)}), 502


@app.route("/api/designer/saved")
def designer_saved_list():
    items = []
    for p in sorted(DESIGNER_SAVE_DIR.glob("*.json")):
        try:
            with open(p) as fh:
                meta = _json.load(fh)
            recs  = meta.get("records", [])
            total = len(recs)
            val   = sum(1 for r in recs if r.get("sidirect2_validated"))
            s6    = sum(1 for r in recs if r.get("score") is not None and int(r["score"]) >= 6)
            items.append({
                "name":       p.stem,
                "gene":       meta.get("gene", p.stem),
                "saved_at":   meta.get("saved_at"),
                "total":      total,
                "validated":  val,
                "score6":     s6,
                "species":    meta.get("species"),
                "accessions": meta.get("accessions", []),
            })
        except Exception:
            items.append({"name": p.stem, "gene": p.stem})
    return jsonify(items)


@app.route("/api/designer/load/<name>")
def designer_load(name: str):
    safe_name = re.sub(r"[^A-Z0-9_\-]", "", name.upper())
    load_path = (DESIGNER_SAVE_DIR / f"{safe_name}.json").resolve()
    if not str(load_path).startswith(str(DESIGNER_SAVE_DIR.resolve())):
        return jsonify({"error": "Invalid path"}), 400
    if not load_path.exists():
        return jsonify({"error": "Not found"}), 404
    with open(load_path) as fh:
        data = _json.load(fh)
    records = _enrich_seed_tm(data.get("records", []))
    data["records"] = _sanitize_records(records)
    return jsonify(data)


# praytothepythongodsandletitrun

if __name__ == "__main__":
    import webbrowser
    server_port = int(os.environ.get("PORT", 5050))
    webbrowser.open(f"http://localhost:{server_port}")
    app.run(host="127.0.0.1", port=server_port, debug=False)
