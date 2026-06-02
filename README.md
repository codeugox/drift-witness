# drift-witness

A runnable demonstration of AI drift monitoring for GRC workflows.

This Phase 1 demo monitors a DistilBERT sentiment classifier applied
to synthetic GRC vendor-risk text. It shows how an AI governance
control can connect model behavior, drift thresholds, and
framework-mapped evidence in a single reproducible workflow.

---

## At a glance

| | |
|---|---|
| **Demo target** | DistilBERT sentiment classifier (`distilbert-base-uncased-finetuned-sst-2-english`) |
| **Domain** | Synthetic GRC vendor-risk text |
| **Metrics** | KS, PSI, JS, embedding centroid cosine distance, low-confidence rate |
| **Evidence** | Markdown summary, JSON evidence record (linked to baseline by SHA-256), Evidently HTML report |
| **Judge** | Optional; Claude Sonnet 4.6 by default, swappable via `ANTHROPIC_JUDGE_MODEL` |
| **Frameworks** | NIST AI RMF, ISO 42001, EU AI Act Article 72, SR 26-2 |
| **Scope** | Local demo (Phase 1). Autonomous delivery layer, full Drift Monitoring Plan, and calibrated judge rubric are not in this repo. |
| **License** | Apache 2.0 |

---

## What this is

`drift-witness` is a local demo of an AI drift-monitoring control.
It compares a production-like text batch against an approved,
SHA-256-locked baseline and writes evidence artifacts that a GRC
reviewer can inspect.

Each run:

- verifies the approved baseline hash,
- computes KS, PSI, JS, embedding drift, and low-confidence rate,
- optionally calls an LLM judge for framework-mapped interpretation,
- writes a Markdown summary, JSON evidence record, and Evidently HTML report.

---

## Quickstart

Requires Python 3.10 or newer. No GPU required; all models run on
CPU. First run downloads ~500MB of model weights (DistilBERT +
all-MiniLM-L6-v2); subsequent runs are local.

```bash
# Clone
git clone https://github.com/YOUR_HANDLE/drift-witness.git
cd drift-witness

# Set up an isolated Python environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Step 1: Lock the approved baseline (run once)
python lock_baseline.py

# Step 2: Smoke-test without an API key (proves the pipeline works)
python drift_test_simple.py --scenario stable --skip-judge

# Step 3: Provide your Anthropic key, then run the full loop
export ANTHROPIC_API_KEY=your-anthropic-key
#  Alternative: copy .env.example to .env and source it
#  (the script reads from the environment, not the file):
#      cp .env.example .env && set -a && source .env && set +a

python drift_test_simple.py --scenario stable
python drift_test_simple.py --scenario critical
```

The repo ships with two scenarios:

1. `stable` — representative healthy vendor-risk traffic.
2. `critical` — mixed-language, AI-generated, incident-heavy inputs
   that trigger material drift.

---

## Expected output

The output blocks below show a full run *with* the LLM judge. If you
ran the smoke test from Quickstart Step 2 with `--skip-judge`, the
statistical metrics (KS, PSI, JS, Embedding, Low Conf Rate) will
match. The risk rating will show `NOT_ASSESSED` and trigger level
`PENDING_HUMAN_REVIEW` because the judge step was skipped.

### Stable scenario

```
KS Statistic:        0.1333  ◉ WATCH
PSI Confidence:      0.0448  ◆ STABLE   (4 quartile bins)
JS Divergence:       0.0239  ◉ WATCH
Embedding Cosine:    0.0418  ◆ STABLE
Low Conf Rate:       6.7%   (matches baseline)
RISK RATING:   LOW
TRIGGER LEVEL: WATCH
```

A stable batch will not produce zero drift on an N=30 baseline.
Minor variation is normal and should not be treated as a material
alert.

### Critical scenario

```
KS Statistic:        0.4333  ■ CRITICAL
PSI Confidence:      0.5122  ■ CRITICAL  (4 quartile bins)
JS Divergence:       0.3579  ■ CRITICAL
Embedding Cosine:    0.1297  ▲ WARNING
Low Conf Rate:       0.0%   (vs 6.7% baseline)
RISK RATING:   CRITICAL
TRIGGER LEVEL: INCIDENT
```

The notable signal is the low-confidence rate: it drops to zero. On
a naive dashboard, this would read as "more confident predictions →
improvement." The LLM judge correctly identifies it as
**paradoxical overconfidence** — a known failure mode of fine-tuned
transformer models applied outside their training domain. That
translation — delivered as reviewable evidence rather than a
dashboard widget — is what this demo is built around.

### A note on PSI

PSI (Population Stability Index) is a familiar
financial-services / model-risk stability metric, computed here
over the classifier confidence distribution. PSI by itself does
not prove accuracy degradation; it indicates that the score
population has moved away from the approved baseline, and should
be interpreted alongside KS, JS, embedding drift, and
low-confidence rate.

Because the public demo baseline is N=30, PSI uses **4 quartile
bins** (not the more common 10 deciles) to avoid sparse-bin
artifacts. Production deployments with larger baselines should
define approved score bands or use decile bins.

### Reproducibility and custom inputs

The two scenarios use fixed text batches, so the statistical
metrics (KS, PSI, JS, embedding drift, low-confidence rate)
reproduce exactly across runs. The LLM judge narrative will vary
slightly between runs because Claude's output is not
bit-deterministic; risk rating and trigger level are stable in
practice but should be treated as judgment, not a fingerprint.

To experiment with other inputs today, edit the `STABLE_TEXTS` or
`CRITICAL_TEXTS` lists in `drift_test_simple.py`. A dedicated
experiment mode that accepts a custom batch file may be added in a
later release.

---

## How to read the outputs

Each run produces three artifact types:

| Artifact | Audience | Purpose |
|---|---|---|
| Drift summary (`drift_summary_*.md`) | GRC / audit reviewer | Plain-English conclusion, metric table with deterministic interpretations, required actions |
| Evidence JSON (`evidence/drift_evidence_*.json`) | Audit trail / automation | Machine-readable record linked to the baseline hash |
| Evidently HTML (`drift_report_*.html`) | Technical reviewer | Visual distribution comparison |

Start with the **summary**, use the **JSON** for traceability and audit
chains, and open the **HTML** report for visual inspection of the
underlying distributions. All three are generated from the same run
state, so metric values and conclusions are designed to stay aligned
across artifacts.

The summary is regenerated on every run; the metric-interpretation
column is deterministic (the same metric/status pair always renders
the same interpretation), while the conclusion, key findings, and
required actions are taken from the LLM judge's structured output.
Running with `--skip-judge` still produces a full summary file; the
judge-derived sections are labeled as pending human review.

---

## Architecture

```diagram
╭──────────────────────╮        ╭───────────────────────────╮
│ lock_baseline.py     │        │  baseline_v1.0_*.json     │
│  · score 30 texts    │───────▶│  + SHA-256 integrity hash │
│  · profile + thresh  │        │  + summary.txt (approve)  │
╰──────────────────────╯        ╰──────┬────────────────────╯
                                       │ tamper-evident link
                                       ▼
╭───────────────────────╮       ╭──────────────────────────╮
│ drift_test_simple.py  │──read─│ hash-verify on every run │
│  [1] load baseline    │       │  fail-closed exit 1      │
│  [2] score prod batch │       ╰──────────────────────────╯
│  [3] KS/PSI/JS/emb    │
│  [4] Evidently report │       ╭──────────────────────────╮
│  [5] LLM judge        │──────▶│  drift_report_*.html     │
│      ↓                │       ╰──────────────────────────╯
│      defensive parser │
│      ↓                │       ╭──────────────────────────╮
│  write evidence       │──────▶│  evidence/drift_*.json   │
╰───────────────────────╯       │  · linked to baseline    │
                                │  · framework-mapped      │
                                │  · revalidation flag     │
                                ╰──────────────────────────╯
```

The classifier is the demonstration target; the reusable
contribution is the control pattern. The judge is swappable via
`ANTHROPIC_JUDGE_MODEL`. The pattern generalizes to any classifier,
embedding model, or RAG pipeline.

---

## Repository layout

**Core files**

```
drift-witness/
├── README.md
├── LICENSE                              ← Apache 2.0 (add via GitHub at publish)
├── requirements.txt
├── .env.example
├── .gitignore
├── lock_baseline.py                     ← generate the approved baseline
├── drift_test_simple.py                 ← run the monitor
├── model_card_VR-SENTIMENT-01.md        ← model card with measured values
└── findings_demo_run_2026-05-29.md      ← analysis of the first run
```

**Sample outputs** — shipped so reviewers can inspect the format
without running the demo. Fresh local runs generate new files with
ISO-date suffixes.

```
├── baseline_v1.0_approved_2026-01-15.json
├── baseline_v1.0_summary.txt
├── drift_summary_stable.md           ← human-readable audit summary
├── drift_summary_critical.md
├── drift_report_stable.html          ← Evidently visual report
├── drift_report_critical.html
└── evidence/
    ├── drift_evidence_2026-06-01_stable.json   ← machine-readable record
    └── drift_evidence_2026-06-01_critical.json
```

---

## Why this matters

The Epic Sepsis Model example shows why post-market monitoring
matters: reported performance can look strong while
deployment-window performance collapses. External validation found
the model caught roughly one in three actual sepsis cases at
recommended thresholds. Reported 87% accuracy fell to 53% when
re-evaluated in the only window where the prediction could change
clinician behavior. Confidence distribution shift, embedding-level
domain drift, and override rate are the kinds of pre-outcome
signals a post-market monitoring program should evaluate.

The gap this demo focuses on is translation. MLOps observability
tools (Evidently, Arize Phoenix, NannyML, Fiddler, WhyLabs) are
strong at drift telemetry; AI governance platforms (OneTrust, IBM
watsonx.governance, ServiceNow AI Control Tower, Credo AI, Holistic
AI) are strong at documentation and policy. Regulated teams need
the layer between them: statistical drift translated into
reviewable, framework-mapped evidence — including a continuous
audit trail when the LLM judge fails or returns unparseable output.

---

## Key findings from the first end-to-end run

See [`findings_demo_run_2026-05-29.md`](findings_demo_run_2026-05-29.md)
for the full analysis. Two findings worth reading first:

1. **The phantom baseline problem.** The model card's "approved
   values" were initially drafted as an *estimated expectation* of
   model behavior, before `lock_baseline.py` was run against the
   approved corpus. Measurement produced the inverse on several
   fields (claimed POSITIVE rate 71% vs. measured 43.3%). This is
   exactly the inversion an audit would flag, and why baseline lock
   files must be the source of truth — not estimates of what the
   baseline "ought to" look like.

2. **Paradoxical overconfidence on out-of-distribution inputs.** On
   the critical batch, low-confidence rate dropped to 0.0% (from
   6.7% baseline) while 67% of predictions registered confidence
   above 0.99 (vs. 27% baseline). The model became *more* confident
   on inputs it understood *least*. The LLM judge correctly
   identified this as a known OOD failure mode. A pure-statistics
   monitor would have called it an improvement.

---

## Scope and limitations

### Included in Phase 1

- Baseline locking and integrity verification
- KS / PSI / JS / embedding drift / low-confidence rate metrics
- Optional LLM judge (Claude Sonnet 4.6 by default)
- Sample summaries, evidence JSON, and Evidently HTML reports
- Model card with measured baseline values

### Demo limitations

- Baseline N=30. Production recommendation is 500+ submissions
  with confirmed risk outcomes.
- LLM judge is uncalibrated. Production use requires human-labeled
  GRC examples and an agreement metric such as Cohen's kappa ≥ 0.70.
- DistilBERT is fine-tuned on SST-2 sentiment, not on GRC text. It
  is intentionally a demonstration target; drift on a
  mismatched-domain classifier is exactly the kind of issue this
  monitor is designed to surface.
- Synthetic GRC text only.
- Not a turnkey production control. The autonomous delivery layer,
  full Model Drift Monitoring Plan, and calibrated framework-mapped
  judge rubric are maintained separately as a private/commercial
  layer.

---

## Security notes

This demo is designed so that:

- API keys are read from environment variables, not hardcoded.
- Generated evidence does not intentionally include API keys or
  secrets.
- Sample evidence uses synthetic GRC text.
- The baseline integrity check (SHA-256 over `texts` + `profile`)
  fails closed: a tampered baseline causes the monitor to exit 1,
  so cron / CI / scheduler layers do not misclassify a tampered run
  as successful.
- Production text included in the judge prompt is wrapped and
  explicitly marked as untrusted data, with an instruction to the
  judge to ignore any directive inside it. This is a demo-grade
  defense.

**Production deployments should additionally:**

- Add input filtering and output validation around the judge call.
- Redact, hash, or restrict access to raw production samples when
  those samples appear in evidence.

---

## Framework mapping

| What this implements | NIST AI RMF | ISO 42001 | EU AI Act | SR 26-2 |
|---|---|---|---|---|
| Approved baseline (locked + hashed) | MAP 1.5 | Cl. 6.1, Annex A.6.2.3 | Art. 9 | Ongoing model monitoring |
| Confidence distribution drift (KS, PSI) | MEASURE 2.5 | Cl. 9.1 | Art. 9 | Outcomes analysis |
| Label distribution drift (JS) | MEASURE 2.6 | Cl. 9.1 | Art. 9 | Outcomes analysis |
| Performance metrics + thresholds | MEASURE 2.7 | Cl. 9.1 | Art. 15 robustness | Ongoing model monitoring |
| Threshold + response protocol | MANAGE 2.2 | Cl. 10.1 | Art. 9 post-market | Change management |
| Signed evidence log | GOVERN 1.7 | Cl. 7.5 | Art. 12 logging | Documentation |
| LLM-as-judge assessment | MEASURE 2.5 | Cl. 9.3 | Art. 9 | Outcomes analysis |
| Defensive parser → continuous audit | GOVERN 1.7 | Cl. 8.1 | Art. 12 | Documentation continuity |

**Scope note:** SR 26-2 applies to traditional model-risk
management for banking organizations but excludes generative and
agentic AI pending further federal banking-regulator guidance. The
DistilBERT classifier and its drift monitoring are treated here as
traditional ML monitoring; the agent delivery layer is intentionally
out of scope for this public Phase 1 demo. EU AI Act Article 72
post-market monitoring remains a useful comparator for continuous
monitoring expectations.

---

## Citations and references

**Epic Sepsis Model failure:**
- Wong, A. et al. *External Validation of a Widely Implemented
  Proprietary Sepsis Prediction Model in Hospitalized Patients.*
  JAMA Internal Medicine, 2021. doi:10.1001/jamainternmed.2021.2626
- Kamran, F. et al. *Evaluation of Sepsis Prediction Models before
  Onset of Treatment.* NEJM AI, 2024.

**Regulatory references:**
- NIST AI RMF 1.0 (January 2023), NIST AI 600-1 GenAI Profile (July
  2024)
- ISO/IEC 42001:2023, ISO/IEC 42005:2025, ISO/IEC 42006:2025
- EU AI Act (Regulation 2024/1689); Omnibus political agreement,
  May 7, 2026
- SR 26-2 / OCC Bulletin 2026-13 (April 17, 2026)

**Models used in demo:**
- `distilbert-base-uncased-finetuned-sst-2-english` (Hugging Face,
  91.3% accuracy on SST-2 dev set)
- `sentence-transformers/all-MiniLM-L6-v2`
- `claude-sonnet-4-6` (Anthropic, released Feb 17, 2026)

---

## License

Apache 2.0.

Code is Apache 2.0 — use it commercially, modify it, fork it. The
sample model card and sample evidence artifacts in this repo are
provided as reference documentation. The agent delivery layer, full
Model Drift Monitoring Plan, and calibrated framework-mapped rubric
are maintained separately as a private/commercial layer.

---

## Contact and author

Senior federal GRC practitioner with a 10+ year background in
FDIC / FSA / FFIEC environments (CISSP, PMP). Current focus: AI
governance, post-market monitoring, and the regulatory wave
reaching financial services, healthcare, and FedRAMP-bound SaaS
in 2026–2027.

Open an issue for technical questions, bugs, or implementation
patterns worth discussing in public. For organizations preparing
for AI governance audits, FedRAMP AI authorization, or EU AI Act
readiness, direct contact via LinkedIn (link in profile).
