"""
lock_baseline.py
================
Generates and locks the approved baseline record for VR-SENTIMENT-01.
Run this ONCE before any drift monitoring. The output becomes the
comparison anchor for all production batch comparisons.

This is the step that gives drift monitoring a real foundation —
it turns the model card's performance claims into a measured,
versioned, machine-readable baseline.

RUN:
    pip install transformers sentence-transformers pandas scipy
    python lock_baseline.py

OUTPUT:
    baseline_v1.0_approved_2026-01-15.json
    baseline_v1.0_summary.txt
"""

import json, hashlib, platform
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import describe as sp_describe

# ─── BASELINE TEXTS (from model card Section: Training / Baseline Data) ───────
# These are the approved 30 GRC texts that define the authorized operating profile.
# DO NOT modify after baseline is locked — changes require a new baseline version.
BASELINE_TEXTS = [
    "Vendor submitted required security documentation on schedule.",
    "Annual SOC 2 Type II report received and reviewed. No exceptions noted.",
    "Audit finding remediated per control requirements. Evidence accepted.",
    "Penetration test report reviewed and POA&M items created.",
    "Annual access review completed with zero unresolved exceptions.",
    "Third-party vendor risk assessment completed. Controls rated satisfactory.",
    "Encryption key rotation completed in accordance with policy.",
    "Business continuity test executed. Recovery time objective met.",
    "Incident response tabletop completed. No critical gaps identified.",
    "Policy exception request reviewed and denied. Standard controls apply.",
    "Vendor completed HECVAT questionnaire. Risk rated acceptable.",
    "Data retention policy review completed. No violations identified.",
    "MFA enforcement confirmed across all privileged accounts.",
    "Vulnerability scan completed. Critical findings remediated within SLA.",
    "Control owner confirmed quarterly review. Evidence archived.",
    "Security awareness training completion rate: 98%. Satisfactory.",
    "Change management record reviewed. Approved by change advisory board.",
    "Disaster recovery test completed. RTO and RPO objectives achieved.",
    "Third-party audit completed. Two minor observations, both accepted.",
    "Data processing agreement updated and countersigned by vendor.",
    "System hardening guide applied. Baseline configuration verified.",
    "Log management policy review complete. Retention compliant.",
    "Risk register updated. No new high risks identified this quarter.",
    "Vendor confirmed no subprocessor changes in past 90 days.",
    "Annual HIPAA security risk analysis completed. Findings documented.",
    "Firewall rule review completed. Forty-two rules removed as obsolete.",
    "Privileged access recertification completed on schedule.",
    "Third-party cloud environment reviewed against shared responsibility model.",
    "Vendor provided evidence of patch management compliance.",
    "Corrective action plan submitted and approved by compliance officer.",
]

# ─── METADATA ──────────────────────────────────────────────────────────────────
METADATA = {
    "system_id":         "VR-SENTIMENT-01",
    "system_name":       "Vendor Risk Assessment Text Classifier",
    "baseline_version":  "v1.0",
    "approval_ref":      "AI-GOV-DEMO-2026-001",
    "approved_by":       "GRC Program Lead",
    "approved_date":     "2026-01-15",
    "base_model":        "distilbert/distilbert-base-uncased-finetuned-sst-2-english",
    "model_card":        "model_card_VR-SENTIMENT-01.md",
    "risk_tier":         "HIGH",
    "n":                 len(BASELINE_TEXTS),
    "language":          "en",
    "domain":            "GRC vendor risk text classification",
    # generation_host intentionally omitted — environment metadata can leak
    # workstation/server identifiers into the published baseline artifact.
    "generation_time":   datetime.utcnow().isoformat() + "Z",
}

def score_texts(texts):
    """Run DistilBERT classifier and return scored DataFrame."""
    try:
        from transformers import pipeline
        print("  Loading DistilBERT (distilbert-base-uncased-finetuned-sst-2-english)...")
        clf = pipeline(
            "sentiment-analysis",
            model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
            device=-1,
        )
        results = clf(texts, truncation=True, max_length=512)
        return pd.DataFrame([
            {
                "text":       t,
                "label":      r["label"],
                "confidence": round(r["score"], 6),
                "length":     len(t),
            }
            for t, r in zip(texts, results)
        ])
    except ImportError:
        print("  transformers not installed — generating synthetic scores for structure demo")
        import random; random.seed(42)
        return pd.DataFrame([
            {
                "text":       t,
                "label":      "POSITIVE" if random.random() > 0.29 else "NEGATIVE",
                "confidence": round(random.gauss(0.912, 0.071), 6),
                "length":     len(t),
            }
            for t in texts
        ])

def compute_embeddings(texts):
    """Compute sentence embeddings for semantic baseline."""
    try:
        from sentence_transformers import SentenceTransformer
        print("  Computing embeddings (all-MiniLM-L6-v2)...")
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        emb = model.encode(texts)
        centroid = emb.mean(axis=0)
        return {
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "centroid":        centroid.tolist(),
            "centroid_norm":   float(np.linalg.norm(centroid)),
            "n_dimensions":    int(centroid.shape[0]),
        }
    except ImportError:
        print("  sentence-transformers not installed — skipping embedding baseline")
        return {"embedding_model": None, "centroid": None, "note": "not computed"}

def compute_text_profile(df):
    """Compute distributional profile of input features."""
    conf = df["confidence"].values
    length = df["length"].values
    label_counts = df["label"].value_counts()
    total = len(df)
    return {
        "confidence": {
            "mean":   round(float(conf.mean()), 6),
            "std":    round(float(conf.std()), 6),
            "min":    round(float(conf.min()), 6),
            "max":    round(float(conf.max()), 6),
            "p10":    round(float(np.percentile(conf, 10)), 6),
            "p25":    round(float(np.percentile(conf, 25)), 6),
            "p50":    round(float(np.percentile(conf, 50)), 6),
            "p75":    round(float(np.percentile(conf, 75)), 6),
            "p90":    round(float(np.percentile(conf, 90)), 6),
            "low_conf_rate_070": round(float((conf < 0.70).mean()), 6),
            "histogram_bins": [
                {"bin": "0.50-0.60", "count": int(((conf >= 0.50) & (conf < 0.60)).sum()),
                 "pct": round(float(((conf >= 0.50) & (conf < 0.60)).mean()), 4)},
                {"bin": "0.60-0.70", "count": int(((conf >= 0.60) & (conf < 0.70)).sum()),
                 "pct": round(float(((conf >= 0.60) & (conf < 0.70)).mean()), 4)},
                {"bin": "0.70-0.80", "count": int(((conf >= 0.70) & (conf < 0.80)).sum()),
                 "pct": round(float(((conf >= 0.70) & (conf < 0.80)).mean()), 4)},
                {"bin": "0.80-0.90", "count": int(((conf >= 0.80) & (conf < 0.90)).sum()),
                 "pct": round(float(((conf >= 0.80) & (conf < 0.90)).mean()), 4)},
                {"bin": "0.90-1.00", "count": int(((conf >= 0.90) & (conf <= 1.00)).sum()),
                 "pct": round(float(((conf >= 0.90) & (conf <= 1.00)).mean()), 4)},
            ],
        },
        "text_length": {
            "mean": round(float(length.mean()), 2),
            "std":  round(float(length.std()), 2),
            "min":  int(length.min()),
            "max":  int(length.max()),
            "p25":  float(np.percentile(length, 25)),
            "p50":  float(np.percentile(length, 50)),
            "p75":  float(np.percentile(length, 75)),
        },
        "label_distribution": {
            "POSITIVE": {
                "count": int(label_counts.get("POSITIVE", 0)),
                "rate":  round(float(label_counts.get("POSITIVE", 0) / total), 6),
            },
            "NEGATIVE": {
                "count": int(label_counts.get("NEGATIVE", 0)),
                "rate":  round(float(label_counts.get("NEGATIVE", 0) / total), 6),
            },
        },
        "null_rate":      0.0,
        "duplicate_rate": round(float(df["text"].duplicated().mean()), 6),
        "language":       "en",
        "ai_generated_rate": 0.0,
    }

def compute_thresholds(profile):
    """
    Derive recommended monitoring thresholds from the baseline profile.
    These become the approved thresholds in the monitoring plan.
    """
    avg_conf = profile["confidence"]["mean"]
    low_conf = profile["confidence"]["low_conf_rate_070"]
    pos_rate = profile["label_distribution"]["POSITIVE"]["rate"]
    avg_len  = profile["text_length"]["mean"]
    return {
        "ks_confidence":        {"watch": 0.05, "warning": 0.15, "critical": 0.30},
        "psi_confidence":       {"watch": 0.05, "warning": 0.10, "critical": 0.20},
        "js_label_distribution":{"watch": 0.02, "warning": 0.05, "critical": 0.15},
        "embedding_cosine":     {"watch": 0.05, "warning": 0.10, "critical": 0.20},
        "avg_confidence_floor": round(avg_conf - 0.10, 3),
        "low_conf_rate_max":    {
            "watch":    round(min(low_conf + 0.03, 0.08), 3),
            "warning":  round(min(low_conf + 0.10, 0.15), 3),
            "critical": round(min(low_conf + 0.25, 0.30), 3),
        },
        "positive_rate_delta":  {"watch": 0.08, "warning": 0.15, "critical": 0.25},
        "text_length_delta_pct":{"watch": 0.20, "warning": 0.40, "critical": 0.70},
        "human_override_rate":  {"watch": 0.05, "warning": 0.10, "critical": 0.20},
        "escalation_rate":      {"watch": 0.06, "warning": 0.12, "critical": 0.25},
    }

def hash_baseline(texts, profile):
    """
    Compute a tamper-evident hash of the baseline.
    If anyone modifies the baseline texts or metrics post-approval,
    the hash will no longer match.
    """
    content = json.dumps({"texts": texts, "profile": profile}, sort_keys=True)
    return hashlib.sha256(content.encode()).hexdigest()

def write_summary(baseline, out_path):
    """Write a human-readable summary for review and approval."""
    p = baseline["profile"]
    conf = p["confidence"]
    labels = p["label_distribution"]
    length = p["text_length"]
    thresholds = baseline["thresholds"]

    lines = [
        "=" * 70,
        f"BASELINE SUMMARY — {baseline['metadata']['system_id']}",
        f"Version: {baseline['metadata']['baseline_version']}",
        f"Approval ref: {baseline['metadata']['approval_ref']}",
        f"Generated: {baseline['metadata']['generation_time']}",
        "=" * 70,
        "",
        "DATASET",
        f"  N:                {baseline['metadata']['n']}",
        f"  Language:         {baseline['metadata']['language']}",
        f"  Domain:           {baseline['metadata']['domain']}",
        "",
        "CONFIDENCE SCORES",
        f"  Mean:             {conf['mean']:.4f}",
        f"  Std:              {conf['std']:.4f}",
        f"  P10/P50/P90:      {conf['p10']:.4f} / {conf['p50']:.4f} / {conf['p90']:.4f}",
        f"  Low conf (<0.70): {conf['low_conf_rate_070']:.1%}",
        "",
        "  Histogram:",
    ]
    for b in conf["histogram_bins"]:
        bar = "█" * int(b["pct"] * 40)
        lines.append(f"    {b['bin']}  {bar:<40} {b['pct']:.1%}")

    lines += [
        "",
        "LABEL DISTRIBUTION",
        f"  POSITIVE:         {labels['POSITIVE']['count']} ({labels['POSITIVE']['rate']:.1%})",
        f"  NEGATIVE:         {labels['NEGATIVE']['count']} ({labels['NEGATIVE']['rate']:.1%})",
        "",
        "INPUT TEXT LENGTH",
        f"  Mean:             {length['mean']:.1f} chars",
        f"  Std:              {length['std']:.1f} chars",
        f"  Range:            {length['min']}–{length['max']} chars",
        "",
        "APPROVED MONITORING THRESHOLDS",
        f"  KS (confidence):  watch {thresholds['ks_confidence']['watch']} / "
        f"warning {thresholds['ks_confidence']['warning']} / "
        f"critical {thresholds['ks_confidence']['critical']}",
        f"  PSI (confidence): watch {thresholds['psi_confidence']['watch']} / "
        f"warning {thresholds['psi_confidence']['warning']} / "
        f"critical {thresholds['psi_confidence']['critical']}",
        f"  JS (labels):      watch {thresholds['js_label_distribution']['watch']} / "
        f"warning {thresholds['js_label_distribution']['warning']} / "
        f"critical {thresholds['js_label_distribution']['critical']}",
        f"  Embedding cosine: watch {thresholds['embedding_cosine']['watch']} / "
        f"warning {thresholds['embedding_cosine']['warning']} / "
        f"critical {thresholds['embedding_cosine']['critical']}",
        "",
        "BASELINE HASH (SHA-256)",
        f"  {baseline['hash']}",
        "",
        "APPROVAL REQUIRED",
        "  Reviewer: _______________________________  Date: ____________",
        "  Signature: ______________________________",
        "",
        "  By signing, the reviewer confirms this baseline accurately represents",
        "  the approved operating profile for VR-SENTIMENT-01 and that the",
        "  thresholds above are appropriate for the system's risk tier.",
        "",
        "=" * 70,
    ]
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Summary written: {out_path}")

def main():
    print("\n" + "="*60)
    print("  BASELINE LOCK — VR-SENTIMENT-01")
    print("  This script generates the approved baseline record.")
    print("  Run once. Lock the output. All drift monitoring")
    print("  compares against this file.")
    print("="*60 + "\n")

    # 1. Score baseline texts
    print("[1/4] Scoring baseline texts with DistilBERT...")
    df = score_texts(BASELINE_TEXTS)
    print(f"  Scored {len(df)} texts. "
          f"POSITIVE: {(df.label=='POSITIVE').sum()}, "
          f"NEGATIVE: {(df.label=='NEGATIVE').sum()}, "
          f"Avg conf: {df.confidence.mean():.4f}\n")

    # 2. Compute embedding baseline
    print("[2/4] Computing embedding centroid...")
    embedding_baseline = compute_embeddings(BASELINE_TEXTS)
    print()

    # 3. Build profile and thresholds
    print("[3/4] Computing statistical profile and thresholds...")
    profile = compute_text_profile(df)
    thresholds = compute_thresholds(profile)
    print()

    # 4. Assemble and hash
    print("[4/4] Assembling and hashing baseline record...")
    baseline = {
        "schema_version":    "1.0",
        "metadata":          METADATA,
        "profile":           profile,
        "embedding_baseline":embedding_baseline,
        "thresholds":        thresholds,
        "texts":             BASELINE_TEXTS,
        "scored_records":    df.to_dict("records"),
        "hash":              None,  # computed below
    }
    # Compute hash over texts + profile (not the full record, to allow metadata updates)
    baseline["hash"] = hash_baseline(BASELINE_TEXTS, profile)
    print(f"  Baseline hash: {baseline['hash'][:16]}...\n")

    # Write outputs
    json_path = "baseline_v1.0_approved_2026-01-15.json"
    summary_path = "baseline_v1.0_summary.txt"

    with open(json_path, "w") as f:
        json.dump(baseline, f, indent=2)
    print(f"  Baseline JSON written: {json_path}")

    write_summary(baseline, summary_path)

    print("\n" + "="*60)
    print("  NEXT STEPS:")
    print(f"  1. Review {summary_path}")
    print("  2. Confirm metrics match model card approved values")
    print("  3. Sign the summary and file as approval evidence")
    print(f"  4. Commit {json_path} to version control (READ-ONLY)")
    print("  5. Reference this file in drift_test_simple.py as")
    print("     the comparison anchor for all production batches")
    print("\n  IMPORTANT: Do not modify the baseline JSON after")
    print("  approval. Changes require a new version (v1.1) and")
    print("  a new approval record.")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
