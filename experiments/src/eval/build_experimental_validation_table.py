#!/usr/bin/env python3
"""Build prospective experimental-validation tables (Table 4 + Supplement S3).

Merges StableProt predictions with literature labels from the Sea6 carrageenase
sheet, deduplicates the two overlapping high-activity carrageenase rows, and
writes cohort aggregates plus a single per-sequence table.
"""
from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
EV = ROOT / "experimental_validation"
MERGED = EV / "results_and_plots" / "all_experimental_predictions_merged.csv"
SEA6 = EV / "sea6_data_embeddings_RESULTS (1).csv"
TEST_PRED = EV / "results_and_plots" / "test_data_prospective_predictions.csv"
OUT_JSON = ROOT / "paper" / "writeup" / "tables" / "experimental_validation_numbers.json"
OUT_MD_TABLE4 = ROOT / "paper" / "writeup" / "tables" / "table5_experimental_validation.md"
OUT_MD_S3 = ROOT / "paper" / "writeup" / "tables" / "table_s3_experimental_validation.md"
OUT_HTML_S3 = ROOT / "paper" / "writeup" / "tables" / "table_s3_experimental_validation.html"
OUT_HTML_TABLE4 = ROOT / "paper" / "writeup" / "tables" / "table4_experimental_validation.html"
OUT_CSV = EV / "results_and_plots" / "experimental_validation_scored.csv"

THRESHOLD = 50.0
# Predictions in the merged CSV were exported with a wider effective scale (~2.0).
# Rescale half-widths to the in-distribution calibration constant used in §3.4.
CALIB_C = 1.56
EXPORT_C = 2.0


def parse_temperature(raw: str) -> tuple[str, float | None]:
    """Return (display string, parsed °C midpoint if numeric)."""
    if raw is None:
        return "—", None
    text = str(raw).strip()
    if not text or text.lower() in {"nan", "—", "-", "not specified", "not available"}:
        return text or "—", None

    low = text.lower()
    if "thermostable" in low and "thermolabile" not in low:
        return text, None
    if "thermolabile" in low:
        return text, None

    nums = [float(x) for x in re.findall(r"(\d+(?:\.\d+)?)", text.replace("–", "-"))]
    if not nums:
        return text, None
    if len(nums) >= 2 and ("-" in text or "–" in text):
        return text, sum(nums[:2]) / 2.0
    if text.startswith("~") or text.startswith("≥") or text.startswith(">"):
        return text, nums[0]
    return text, nums[0]


def infer_threshold_class(reference_raw: str, parsed: float | None, *, allow_pred_class: str | None = None) -> str | None:
    """Thermostable / thermolabile at 50 °C for Tier 1."""
    low = (reference_raw or "").lower().replace("≥", ">=").replace("°", "")
    if "not specified" in low or not reference_raw or reference_raw == "—":
        return None
    if "thermostable" in low and (">=" in low or "at or above" in low):
        return "thermostable"
    if "thermolabile" in low:
        return "thermolabile"
    if parsed is not None:
        return "thermostable" if parsed >= THRESHOLD else "thermolabile"
    if allow_pred_class:
        pc = allow_pred_class.lower()
        if "thermostable" in pc:
            return "thermostable"
        if "thermolabile" in pc:
            return "thermolabile"
    return None


def rescaled_interval(mu: float, hw_raw: float) -> tuple[float, float, float]:
    hw = hw_raw * CALIB_C / EXPORT_C
    return mu - hw, mu + hw, hw


def tier1_ok(mu: float, ref_class: str | None) -> bool | None:
    if ref_class is None:
        return None
    pred_class = "thermostable" if mu >= THRESHOLD else "thermolabile"
    return pred_class == ref_class


def tier2_ok(ci_low: float, ci_high: float, ref_class: str | None) -> bool | None:
    if ref_class is None:
        return None
    if ref_class == "thermostable":
        return ci_high >= THRESHOLD
    return ci_low < THRESHOLD


def truncate_seq(seq: str, head: int = 18, tail: int = 8) -> str:
    if len(seq) <= head + tail + 3:
        return seq
    return f"{seq[:head]}…{seq[-tail:]}"


def load_sea6_labels() -> dict[str, dict]:
    labels: dict[str, dict] = {}
    with open(SEA6, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            seq = (row.get("Sequence") or "").strip()
            if len(seq) < 20:
                continue
            name = (row.get("Pritein names") or row.get("Protein names") or "").strip()
            temp = (row.get("TEMPARATURE INFO") or "").strip()
            if seq not in labels:
                labels[seq] = {"id": name, "reference_raw": temp}
    return labels


@dataclass
class Row:
    cohort: str
    protein_id: str
    sequence: str
    sequence_display: str
    reference_raw: str
    reference_parsed: float | None
    predicted_tm: float
    half_width: float
    ci_low: float
    ci_high: float
    abs_error: float | None
    tier1: bool | None
    tier2: bool | None
    reference_type: str


def build_rows() -> tuple[list[Row], list[Row]]:
    """Build prospective experimental-validation rows under Biological Consensus Deduplication.

    - 5OCR: N=5 unique engineered variants
    - Sea6 Carrageenases: N=13 unique enzymes (11 scored)
    - High-Activity Carrageenases: N=2 unique reference enzymes
    - Thermostable Lipases: N=37 unique consensus sequences (35 pure + 2 consensus)
    - Thermolabile Lipases: N=32 unique consensus sequences (28 pure + 4 consensus)
    Total: 87 unique sequences across the prospective suite (85 scored non-redundant).

    Returns:
      (all_5cohort_rows, non_redundant_unique_rows)
    """
    merged = pd.read_csv(MERGED)
    sea6 = load_sea6_labels()
    test_pred = pd.read_csv(TEST_PRED)
    carr_pred = pd.read_csv(EV / "results_and_plots" / "carrageenase_predictions.csv")

    ts_raw = merged[merged["dataset_category"] == "Thermostable Lipases"]
    tl_raw = merged[merged["dataset_category"] == "Thermolabile Lipases"]

    ts_seqs = set(ts_raw["sequence"].dropna().unique())
    tl_seqs = set(tl_raw["sequence"].dropna().unique())
    overlap_seqs = ts_seqs.intersection(tl_seqs)

    # Biological Consensus Classification for the 6 cross-isolate sequences:
    # 1. Thermostable monoacylglycerol lipase (233 aa) -> Thermostable (>=50 °C)
    # 2. Lipase_1 / Lipase_3 (410 aa) -> Thermostable (>=50 °C)
    # 3. Spore germination lipase LipC (237 aa) -> Thermolabile (<50 °C)
    # 4. Phospholipase YtpA (267 aa) -> Thermolabile (<50 °C)
    # 5. Triacylglycerol lipase (324 aa) -> Thermolabile (<50 °C)
    # 6. Triacylglycerol lipase (330 aa) -> Thermolabile (<50 °C)
    consensus_ts_seqs = (ts_seqs - overlap_seqs).union({s for s in overlap_seqs if len(s) in [233, 410]})
    consensus_tl_seqs = (tl_seqs - overlap_seqs).union({s for s in overlap_seqs if len(s) in [237, 267, 324, 330]})

    rows: list[Row] = []

    # ── Cohort 1: High-Activity Carrageenases (N=2) ──
    for _, r in carr_pred.iterrows():
        seq = str(r["sequence"])
        mu = float(r["tm_pred"])
        lo, hi, hw = rescaled_interval(mu, float(r["tm_conf"]))
        exp_t = float(r["exp_t_opt"])
        ref_raw = f"{exp_t:.1f} °C (optimum)"
        err = abs(mu - exp_t)
        ref_class = "thermolabile" if exp_t < THRESHOLD else "thermostable"
        rows.append(
            Row(
                cohort="High-Activity Carrageenases",
                protein_id=str(r["name"]),
                sequence=seq,
                sequence_display=truncate_seq(seq),
                reference_raw=ref_raw,
                reference_parsed=exp_t,
                predicted_tm=round(mu, 2),
                half_width=round(hw, 2),
                ci_low=round(lo, 2),
                ci_high=round(hi, 2),
                abs_error=round(err, 2),
                tier1=tier1_ok(mu, ref_class),
                tier2=tier2_ok(lo, hi, ref_class),
                reference_type="Measured activity optimum (Topt)",
            )
        )

    # ── Cohort 2: Codon-Optimised 5OCR Series (N=5) ──
    ocr = merged[merged["dataset_category"] == "Codon-Optimized 5OCR Series"].drop_duplicates(subset=["sequence"]).copy()
    for _, r in ocr.iterrows():
        sid = str(r["sequence_id"])
        if sid.startswith("Wild_Type"):
            ref_raw = "Thermolabile (<50 °C)"
        else:
            ref_raw = "Thermostable (≥50 °C)"
        ref_parsed = parse_temperature(ref_raw)[1]
        ref_class = infer_threshold_class(ref_raw, ref_parsed)
        mu = float(r["predicted_tm_C"])
        lo, hi, hw = rescaled_interval(mu, float(r["uncertainty_C"]))
        rows.append(
            Row(
                cohort="Codon-Optimized 5OCR",
                protein_id=sid.replace(" (ddG=", " ").split(")")[0],
                sequence=str(r["sequence"]),
                sequence_display=truncate_seq(str(r["sequence"])),
                reference_raw=ref_raw,
                reference_parsed=ref_parsed,
                predicted_tm=round(mu, 2),
                half_width=round(hw, 2),
                ci_low=round(lo, 2),
                ci_high=round(hi, 2),
                abs_error=None,
                tier1=tier1_ok(mu, ref_class),
                tier2=tier2_ok(lo, hi, ref_class),
                reference_type="Threshold label + engineered series",
            )
        )

    # ── Cohort 3: Sea6 Marine Carrageenases (N=13 unique) ──
    for _, r in test_pred.drop_duplicates(subset=["sequence"]).iterrows():
        seq = str(r["sequence"])
        meta = sea6.get(seq, {})
        ref_raw = meta.get("reference_raw", "Not specified")
        ref_disp, ref_parsed = parse_temperature(ref_raw)
        ref_class = infer_threshold_class(ref_disp, ref_parsed)
        mu = float(r["tm_pred"])
        lo, hi, hw = rescaled_interval(mu, float(r["tm_conf"]))
        err = abs(mu - ref_parsed) if ref_parsed is not None else None
        pid = str(r["id"]).split(" (Seq_")[0].strip()
        rows.append(
            Row(
                cohort="Sea6 Marine Carrageenases",
                protein_id=pid,
                sequence=seq,
                sequence_display=truncate_seq(seq),
                reference_raw=ref_disp,
                reference_parsed=round(ref_parsed, 2) if ref_parsed is not None else None,
                predicted_tm=round(mu, 2),
                half_width=round(hw, 2),
                ci_low=round(lo, 2),
                ci_high=round(hi, 2),
                abs_error=round(err, 2) if err is not None else None,
                tier1=tier1_ok(mu, ref_class),
                tier2=tier2_ok(lo, hi, ref_class),
                reference_type="Literature stability / activity optimum",
            )
        )

    # ── Cohort 4: Consensus Thermostable Lipases (N=37 unique) ──
    # Pick source row from ts_raw (or tl_raw if not found)
    for seq in consensus_ts_seqs:
        sub = ts_raw[ts_raw["sequence"] == seq]
        if len(sub) == 0:
            sub = tl_raw[tl_raw["sequence"] == seq]
        r = sub.iloc[0]
        ref_raw = "Thermostable (≥50 °C)"
        ref_parsed = 50.0
        ref_class = "thermostable"
        mu = float(r["predicted_tm_C"])
        lo, hi, hw = rescaled_interval(mu, float(r["uncertainty_C"]))
        rows.append(
            Row(
                cohort="Thermostable Lipases",
                protein_id=str(r["sequence_id"]),
                sequence=str(r["sequence"]),
                sequence_display=truncate_seq(str(r["sequence"])),
                reference_raw=ref_raw,
                reference_parsed=ref_parsed,
                predicted_tm=round(mu, 2),
                half_width=round(hw, 2),
                ci_low=round(lo, 2),
                ci_high=round(hi, 2),
                abs_error=None,
                tier1=tier1_ok(mu, ref_class),
                tier2=tier2_ok(lo, hi, ref_class),
                reference_type="Threshold label (≥50 °C)",
            )
        )

    # ── Cohort 5: Consensus Thermolabile Lipases (N=32 unique) ──
    for seq in consensus_tl_seqs:
        sub = tl_raw[tl_raw["sequence"] == seq]
        if len(sub) == 0:
            sub = ts_raw[ts_raw["sequence"] == seq]
        r = sub.iloc[0]
        ref_raw = "Thermolabile (<50 °C)"
        ref_parsed = 49.9
        ref_class = "thermolabile"
        mu = float(r["predicted_tm_C"])
        lo, hi, hw = rescaled_interval(mu, float(r["uncertainty_C"]))
        rows.append(
            Row(
                cohort="Thermolabile Lipases",
                protein_id=str(r["sequence_id"]),
                sequence=str(r["sequence"]),
                sequence_display=truncate_seq(str(r["sequence"])),
                reference_raw=ref_raw,
                reference_parsed=ref_parsed,
                predicted_tm=round(mu, 2),
                half_width=round(hw, 2),
                ci_low=round(lo, 2),
                ci_high=round(hi, 2),
                abs_error=None,
                tier1=tier1_ok(mu, ref_class),
                tier2=tier2_ok(lo, hi, ref_class),
                reference_type="Threshold label (<50 °C)",
            )
        )

    # Non-redundant unique rows across all 5 cohorts (5OCR + Sea6 + TS Lip + TL Lip = 87 unique, 85 scored)
    # Excludes the 2 high-activity carrageenase duplicates since they are already in Sea6
    unique_rows: list[Row] = []
    seen = set()
    for r in rows:
        if r.cohort == "High-Activity Carrageenases":
            continue
        if r.sequence not in seen:
            unique_rows.append(r)
            seen.add(r.sequence)

    return rows, unique_rows


def pct(num: int, den: int) -> str:
    if den == 0:
        return "—"
    return f"{100.0 * num / den:.1f}% ({num}/{den})"


def summarize(rows: list[Row], unique_rows: list[Row]) -> dict:
    cohorts = {}
    order = [
        "High-Activity Carrageenases",
        "Codon-Optimized 5OCR",
        "Sea6 Marine Carrageenases",
        "Thermostable Lipases",
        "Thermolabile Lipases",
    ]
    for name in order:
        sub = [r for r in rows if r.cohort == name]
        t1 = [r.tier1 for r in sub if r.tier1 is not None]
        t2 = [r.tier2 for r in sub if r.tier2 is not None]
        errs = [r.abs_error for r in sub if r.abs_error is not None]
        cohorts[name] = {
            "n": len(sub),
            "reference_type": sub[0].reference_type if sub else "",
            "point_mae": round(sum(errs) / len(errs), 2) if errs else None,
            "point_mae_n": len(errs),
            "tier1": pct(sum(t1), len(t1)),
            "tier1_num": sum(t1),
            "tier1_den": len(t1),
            "tier2": pct(sum(t2), len(t2)),
            "tier2_num": sum(t2),
            "tier2_den": len(t2),
            "mean_half_width": round(sum(r.half_width for r in sub) / len(sub), 1) if sub else None,
        }

    # Combined non-redundant total (79 scored, 81 total)
    t1_u = [r.tier1 for r in unique_rows if r.tier1 is not None]
    t2_u = [r.tier2 for r in unique_rows if r.tier2 is not None]
    errs_u = [r.abs_error for r in unique_rows if r.abs_error is not None]
    cohorts["Combined Prospective Total"] = {
        "n": len(unique_rows),
        "reference_type": "Multi-tier prospective experimental suite",
        "point_mae": round(sum(errs_u) / len(errs_u), 2) if errs_u else None,
        "point_mae_n": len(errs_u),
        "tier1": pct(sum(t1_u), len(t1_u)),
        "tier1_num": sum(t1_u),
        "tier1_den": len(t1_u),
        "tier2": pct(sum(t2_u), len(t2_u)),
        "tier2_num": sum(t2_u),
        "tier2_den": len(t2_u),
        "mean_half_width": round(sum(r.half_width for r in unique_rows) / len(unique_rows), 1),
    }
    return cohorts


def write_markdown(rows: list[Row], cohorts: dict) -> None:
    lines = [
        "# Table 4 / Table 5: Prospective evaluation across 5 laboratory and engineered datasets",
        "",
        f"Intervals use \\(c = {CALIB_C}\\) (\\(\\mu \\pm 1.96\\,c\\,\\sigma\\), half-widths rescaled from "
        f"the export scale \\(c = {EXPORT_C}\\)). Tier 1 is point classification at 50 °C. Tier 2 is whether "
        "the 95% confidence interval is consistent with the reference class.",
        "",
        "| Cohort | N | Reference type | Point agreement | Tier 1 ↑ | Tier 2 ↑ | Mean 95% half-width |",
        "|:---|:---:|:---|:---:|:---:|:---:|:---:|",
    ]
    order = [
        "High-Activity Carrageenases",
        "Codon-Optimized 5OCR",
        "Sea6 Marine Carrageenases",
        "Thermostable Lipases",
        "Thermolabile Lipases",
        "Combined Prospective Total",
    ]
    for name in order:
        c = cohorts[name]
        if c.get("point_mae") is not None:
            point = f"**{c['point_mae']:.2f} °C** (n = {c['point_mae_n']})"
        else:
            point = "not available"
        bold = "**" if name == "Combined Prospective Total" else ""
        end_bold = "**" if name == "Combined Prospective Total" else ""
        lines.append(
            f"| {bold}{name}{end_bold} | {c['n']} | {c['reference_type']} | {point} | "
            f"{c['tier1']} | {c['tier2']} | {c['mean_half_width']} °C |"
        )
    OUT_MD_TABLE4.write_text("\n".join(lines) + "\n")

    s3 = [
        "# Supplementary Table S3: Prospective evaluation across all 5 datasets, per sequence",
        "",
        f"All {len(rows)} prospective sequence records (Option A: deduplicated clean sequences across all 5 cohorts). "
        "Full amino-acid sequences are in `experimental_validation/results_and_plots/experimental_validation_scored.csv`. "
        f"Intervals use \\(c = {CALIB_C}\\).",
        "",
        "| Cohort | ID | Sequence | Reference | Parsed (°C) | Pred Tm | ±95% | 95% interval | |Δ| | Tier 1 | Tier 2 |",
        "|:---|:---|:---|:---|:---:|:---:|:---:|:---|:---:|:---:|:---:|",
    ]
    for r in rows:
        t1 = "✓" if r.tier1 else ("✗" if r.tier1 is False else "—")
        t2 = "✓" if r.tier2 else ("✗" if r.tier2 is False else "—")
        parsed = "—" if r.reference_parsed is None else f"{r.reference_parsed:.1f}"
        err = "—" if r.abs_error is None else f"{r.abs_error:.2f}"
        s3.append(
            f"| {r.cohort} | {r.protein_id} | `{r.sequence_display}` | {r.reference_raw} | {parsed} | "
            f"{r.predicted_tm:.2f} | ±{r.half_width:.2f} | [{r.ci_low:.2f}, {r.ci_high:.2f}] | {err} | {t1} | {t2} |"
        )
    OUT_MD_S3.write_text("\n".join(s3) + "\n")


def write_html(rows: list[Row], cohorts: dict) -> None:
    def esc(s: str) -> str:
        return (
            str(s)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )

    t4_lines = []
    order = [
        "High-Activity Carrageenases",
        "Codon-Optimized 5OCR",
        "Sea6 Marine Carrageenases",
        "Thermostable Lipases",
        "Thermolabile Lipases",
        "Combined Prospective Total",
    ]
    for name in order:
        c = cohorts[name]
        if c.get("point_mae") is not None:
            point = f'<span class="num"><b>{c["point_mae"]:.2f}&deg;C</b> (n={c["point_mae_n"]})</span>'
        else:
            point = "not available"
        cls = ' class="sep ours"' if name == "Combined Prospective Total" else ""
        t1 = c["tier1"]
        t2 = c["tier2"]
        if name in {"High-Activity Carrageenases", "Codon-Optimized 5OCR"}:
            t1 = f'<span class="num"><b>{t1}</b></span>'
            t2 = f'<span class="num"><b>{t2}</b></span>'
        else:
            t1 = f'<span class="num">{t1}</span>'
            t2 = f'<span class="num">{t2}</span>'
        t4_lines.append(
            f'<tr{cls}><td>{esc(name)}</td><td class="num">{c["n"]}</td>'
            f'<td style="text-align:left">{esc(c["reference_type"])}</td>'
            f'<td>{point}</td><td>{t1}</td><td>{t2}</td>'
            f'<td class="num">{c["mean_half_width"]}&deg;C</td></tr>'
        )
    OUT_HTML_TABLE4.write_text("\n".join(t4_lines) + "\n")

    s3_lines = []
    current = None
    for r in rows:
        if r.cohort != current:
            current = r.cohort
            s3_lines.append(
                f'<tr class="sep"><td colspan="10" style="text-align:left;font-weight:600;padding-top:10px">'
                f'{esc(current)}</td></tr>'
            )
        t1 = "&#10003;" if r.tier1 else ("&#10007;" if r.tier1 is False else "&mdash;")
        t2 = "&#10003;" if r.tier2 else ("&#10007;" if r.tier2 is False else "&mdash;")
        parsed = "&mdash;" if r.reference_parsed is None else f'{r.reference_parsed:.1f}'
        err = "&mdash;" if r.abs_error is None else f'{r.abs_error:.2f}'
        s3_lines.append(
            f'<tr><td style="text-align:left">{esc(r.protein_id)}</td>'
            f'<td style="text-align:left;font-family:monospace;font-size:.72rem">{esc(r.sequence_display)}</td>'
            f'<td style="text-align:left">{esc(r.reference_raw)}</td><td class="num">{parsed}</td>'
            f'<td class="num">{r.predicted_tm:.2f}</td><td class="num">&plusmn;{r.half_width:.2f}</td>'
            f'<td class="num">[{r.ci_low:.2f}, {r.ci_high:.2f}]</td><td class="num">{err}</td>'
            f'<td>{t1}</td><td>{t2}</td></tr>'
        )
    OUT_HTML_S3.write_text("\n".join(s3_lines) + "\n")


def main() -> None:
    rows, unique_rows = build_rows()
    cohorts = summarize(rows, unique_rows)

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "calib_c": CALIB_C,
        "export_c": EXPORT_C,
        "n_total_records": len(rows),
        "n_unique_sequences": len(unique_rows),
        "cohorts": cohorts,
        "rows": [asdict(r) for r in rows],
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2) + "\n")

    pd.DataFrame([asdict(r) for r in rows]).to_csv(OUT_CSV, index=False)
    write_markdown(rows, cohorts)
    write_html(rows, cohorts)

    print(json.dumps(cohorts, indent=2))
    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_CSV}")


if __name__ == "__main__":
    main()
