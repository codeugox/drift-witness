# Model Card — VR-SENTIMENT-01
# Vendor Risk Assessment Text Classifier
# GRC Drift Monitoring Demo System

---

## System Identity

| Field | Value |
|---|---|
| **System ID** | VR-SENTIMENT-01 |
| **System Name** | Vendor Risk Assessment Text Classifier |
| **Version** | v1.0 |
| **Base Model** | distilbert/distilbert-base-uncased-finetuned-sst-2-english |
| **Base Model Source** | https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english |
| **Approval Reference** | AI-GOV-DEMO-2026-001 |
| **Baseline Locked** | 2026-01-15 |
| **Approved By** | GRC Program Lead |
| **Risk Tier** | HIGH (outputs used to prioritize human review queue) |

---

## Intended Use

**Primary use:** Classify GRC vendor risk submission text as requiring immediate attention
(NEGATIVE) or satisfactory (POSITIVE), to support automated triage of vendor risk review queues
in regulated financial sector environments.

**Intended users:** GRC analysts, vendor risk program managers, and audit teams at regulated
financial institutions using an automated first-pass triage tool. All NEGATIVE outputs must
receive human review before action is taken. This system is advisory-only for NEGATIVE classifications.

**Intended deployment context:** English-language vendor submissions, audit findings, and
compliance attestations from regulated financial sector clients. Standard template-based
submissions. Single-submission classification (not multi-document or conversation).

**What this system should NOT be used for:**
- Final risk decisions without human review
- Non-English language submissions
- AI-generated vendor attestations (see Known Limitations)
- Legal or regulatory compliance determinations
- Any use case outside vendor risk text classification

---

## Base Model Documentation

The base model (DistilBERT fine-tuned on SST-2) was originally trained on the Stanford
Sentiment Treebank — a dataset of movie review sentences. Its Hugging Face model card
reports 91.3% accuracy on the SST-2 development set.

**Important scope note:** This base model was NOT trained on GRC or vendor risk text.
It is applied here via transfer learning of sentiment signal to a GRC classification task.
The baseline documented below reflects performance on GRC text specifically, not on the
SST-2 benchmark. These are different tasks with different populations.

---

## Training / Baseline Data Profile

The approved baseline dataset for drift monitoring comparison is:

| Field | Value |
|---|---|
| **Baseline Dataset File** | baseline_v1.0_approved_2026-01-15.json |
| **Baseline N** | 30 texts (demo) · Production recommendation: 500+ |
| **Collection Method** | Representative GRC practitioner-authored samples |
| **Composition** | ~60% vendor attestations, ~25% audit findings, ~15% incident/exception |
| **Language** | English (100%) |
| **Template Compliance** | Standard enterprise GRC submission format |
| **AI-Generated Content** | 0% — all texts human-authored |
| **Date Range** | January 2026 |
| **Population Description** | Standard enterprise vendor risk submissions from regulated financial sector clients |

**Baseline texts represent:** Normal operational GRC activity — vendor documentation submissions,
audit finding remediations, access reviews, control validations, and compliance confirmations.
Texts are English-language, template-compliant, and free of exception/incident language.

**What the baseline does NOT represent:**
- Policy exception requests (higher complexity language)
- Incident reports (urgent/crisis language)
- Multi-language submissions
- AI-generated compliance attestations
- Acquisition or M&A transition language
- Submissions longer than ~120 characters (model performance degrades)

---

## Approved Performance Metrics (Baseline)

These metrics define the authorized performance standard. Any production batch showing
degradation beyond threshold triggers the drift response protocol.

**Source of truth:** Values below are populated from the locked baseline file
`baseline_v1.0_approved_2026-01-15.json` (SHA-256:
`7d5e387a36a200a584b11d45762ba11124f3b2b0f564cf7182b6215a3f3d626d`). They are
**measured**, not estimated. Any divergence between this table and the locked
baseline file is a documentation error and the baseline file governs.

| Metric | Approved Value | Watch Threshold | Warning Threshold | Critical Threshold |
|---|---|---|---|---|
| Avg Confidence Score | 0.9237 | <0.87 | <0.80 | <0.70 |
| Confidence Std Dev | 0.1155 | — | — | — |
| Low Confidence Rate (<0.70) | 6.7% | >8% | >15% | >30% |
| POSITIVE Rate | 43.3% | ±8pp | ±15pp | ±25pp |
| NEGATIVE Rate | 56.7% | ±8pp | ±15pp | ±25pp |
| Very High Confidence Rate (>0.99) | 26.7% | — | — | — |
| Avg Input Text Length | 64.3 chars | >±20% | >±40% | >±70% |
| Input Text Length Range | 56–75 chars | — | — | — |
| KS Statistic (confidence) | ~0.000 | >0.05 | >0.15 | >0.30 |
| JS Divergence (labels) | ~0.000 | >0.02 | >0.05 | >0.15 |
| PSI (confidence) [†](#psi-footnote) | 0.0000 (self vs. self) | >0.05 | >0.10 | >0.20 |
| Embedding Centroid Cosine | ~0.000 | >0.05 | >0.10 | >0.20 |

**Calibration:** Model confidence scores are not independently calibrated against
GRC-specific ground truth. Confidence scores should be treated as relative ordering
signals, not probability estimates. ECE not validated for this deployment.

<a id="psi-footnote"></a>
**† PSI footnote.** PSI of the baseline distribution against itself is 0 by
definition; the "Measured Baseline" column shows that floor. PSI is meaningful
only when comparing a production batch to the approved baseline. The demo uses
**4 quartile bins** rather than 10 deciles because the baseline is N=30 and
decile bins would average ~3 records each (sparse-bin instability). Production
deployments with larger baselines should use approved score bands or decile
bins, and should validate thresholds against the production monitoring plan
(common banking rules of thumb are <0.10 stable / 0.10–0.25 moderate / >0.25
material; the demo values above are deliberately stricter for a small-N
synthetic baseline).

### Note on the label distribution

The 43.3% POSITIVE / 56.7% NEGATIVE split is *not* what a GRC reader would intuit
when looking at the baseline texts — every baseline text describes a routine,
successful, compliant operation that a human would consider "positive." The
distribution reflects how DistilBERT-SST-2 classifies based on **sentiment-language
patterns learned from movie reviews**, not on GRC operational meaning. Words such
as *required*, *risk*, *removed*, *obsolete*, *test*, *exception* carry mild-to-
strong negative sentiment in the SST-2 training distribution even when used in
GRC-positive contexts. This is the domain transfer gap quantified.

This is the canonical evidence for why estimated model card values must be
replaced with measured values before approval. An earlier draft of this card
estimated POSITIVE = 71% by reasoning about GRC intent. The locked baseline
shows the opposite. Measurement governs.

---

## Fairness Profile

| Segment | Baseline Accuracy (Estimated) | Notes |
|---|---|---|
| Vendor Attestations | ~93% | Primary training domain; most representative texts |
| Audit Findings | ~90% | Secondary domain; some domain-specific language |
| Incident / Exception | ~87% | Underrepresented in baseline; monitor closely |
| Non-English submissions | Not validated | Out of scope; schema drift alert should fire |
| AI-generated text | Not validated | Out of scope; embedding drift alert should fire |

**Fairness limitation:** Demographic fairness analysis was not performed. This system
classifies text content, not individuals. Indirect fairness risks (vendor size, geography,
language) have not been formally assessed.

---

## Known Limitations

1. **Domain transfer gap.** This model was trained on movie review sentiment (SST-2),
   not GRC text. It generalizes well to simple positive/negative GRC language but
   may underperform on technical compliance language, legal qualifications, or
   complex multi-clause submissions.

2. **English only.** Model was trained exclusively on English text. Non-English
   inputs will produce unreliable outputs. Schema/population drift monitoring
   should catch language shifts before they compound.

3. **Length sensitivity.** Performance degrades on submissions longer than
   approximately 120 characters. Inputs are truncated at 512 tokens. Long
   exception narratives or legal disclosures may receive lower-confidence
   or incorrect classifications.

4. **AI-generated content.** The baseline contains no AI-generated text.
   Vendor submissions generated by LLMs (ChatGPT, Claude, etc.) may produce
   embedding drift signals that reflect stylistic shift rather than
   risk-relevant semantic shift. These must be investigated, not auto-dismissed.

5. **No ground truth validation on GRC text.** The 91.3% accuracy figure
   applies to SST-2 movie reviews. Accuracy on GRC text has not been
   independently validated with labeled GRC outcomes.

6. **Binary classification only.** Outputs are POSITIVE/NEGATIVE with a
   confidence score. No severity gradation within NEGATIVE. Human reviewers
   must determine severity and action for all flagged submissions.

7. **Demo dataset size.** The baseline N=30 is appropriate for demonstration
   purposes. Production deployments should establish a baseline of 500+
   submissions validated against actual risk outcomes.

8. **Paradoxical overconfidence on out-of-distribution inputs (empirically
   observed, 2026-05-29).** On a critical-drift test batch containing
   non-English text, AI-generated attestations, and incident/crisis language,
   the model produced **zero low-confidence predictions** (vs 6.7% on baseline)
   and 67% of predictions registered confidence above 0.99 (vs 27% on baseline).
   The model became *more* confident on OOD inputs, not less. A naive monitor
   reads "low-conf rate dropped → improved performance"; the correct
   interpretation is that strong lexical signals in OOD text cause the model
   to commit harder to a class without understanding the input. **Reviewers
   must treat a sudden drop in low-confidence rate as a drift signal in its
   own right, not as a performance improvement.** See
   `findings_demo_run_2026-05-29.md` for the run that produced this evidence.

9. **Right-for-the-wrong-reason and wrong-for-the-right-reason classifications.**
   The model can produce classifications that are correct from the sentiment
   perspective and incorrect from the GRC risk perspective, and vice versa.
   Example from the 2026-05-29 demo run: an AI-generated compliance attestation
   ("This AI-generated compliance attestation confirms all controls are
   implemented per framework v4.2.") classified POSITIVE with confidence 0.85,
   which a GRC reviewer would flag as high-risk evidence requiring scrutiny.
   Statistical drift monitoring cannot catch this — the model is operating
   within its baseline. Human review of POSITIVE classifications in elevated
   risk categories is a required compensating control.

---

## Ethical Considerations

- **Automation bias risk:** Reviewers may over-rely on POSITIVE classifications
  and reduce scrutiny of submissions the model did not flag. Human review
  procedures must explicitly address this.
- **Vendor equity:** If model performance varies across vendor size, geography,
  or submission style, systematic bias in vendor risk ratings may result.
  Segment performance monitoring is required.
- **Opacity:** DistilBERT is a black-box classifier from the reviewer's
  perspective. Reviewers should not be expected to justify a NEGATIVE
  classification based on the model's internal reasoning.

---

## Drift Monitoring Plan Reference

This model card anchors the following drift monitoring controls:

| Control | Reference |
|---|---|
| Monitoring Plan Document | Model_Drift_Monitoring_Plan_VR-SENTIMENT-01_v1.0 |
| Baseline Dataset File | baseline_v1.0_approved_2026-01-15.json |
| Monitoring Script | drift_test_simple.py |
| LLM Judge | Claude Sonnet 4.6 (claude-sonnet-4-6) — overridable via `ANTHROPIC_JUDGE_MODEL` |
| Scheduled Delivery | OpenClaw heartbeat / ai-drift-monitor.md skill |
| Evidence Directory | evidence/ |
| NIST AI RMF Mapping | MAP 1.5, MEASURE 2.3, 2.4, 2.5, MANAGE 4.1 |
| ISO 42001 Mapping | Clause 9.1, A.6.2.3, A.6.2.6 |
| EU AI Act Mapping | Article 72 (post-market monitoring) |

---

## Retraining and Decommission Criteria

**Retrain when:**
- KS statistic >0.30 sustained across two consecutive production batches
- Estimated accuracy on labeled sample drops below 80%
- Material domain shift confirmed (e.g., non-English inputs become standard)
- Base model deprecated by Hugging Face / NousResearch

**Decommission when:**
- Revalidation fails twice following retraining
- Use case scope changes beyond the documented intended use
- Better-validated GRC-specific classifier becomes available

---

## Version History

| Version | Date | Change | Approved By |
|---|---|---|---|
| v1.0 | 2026-01-15 | Initial baseline. N=30 demo dataset. | GRC Program Lead |
| v1.0.1 | 2026-05-29 | Performance metrics table reconciled to locked baseline file (measured values replaced earlier estimates: POSITIVE 71%→43.3%, avg conf 0.912→0.9237, length 54.3→64.3, low-conf 4.1%→6.7%). Added Known Limitations #8 (paradoxical overconfidence on OOD inputs) and #9 (right-for-wrong-reason classifications), both from empirical evidence in `findings_demo_run_2026-05-29.md`. No model behavior change; documentation correction only. | GRC Program Lead |

---

## Demo vs. Production Note

This model card documents a **demonstration system** built to show how drift
monitoring works. For a production deployment:

1. Baseline N should be 500+ submissions with confirmed risk outcomes
2. Performance metrics should be validated against GRC-specific ground truth
3. Fairness profile should include segment analysis across vendor types
4. Calibration should be validated against historical accuracy on labeled data
5. An independent model validation should be performed before authorization
6. This model card should be reviewed and approved by both the model owner
   and an independent second-line reviewer

The demo system is architecturally identical to a production system.
The governance documents (this card, the monitoring plan, the evidence schema)
are structured like production artifacts, with demo-scale data and
demo-scale validation. Production deployment requires the calibration,
sample size, and independent validation steps listed above.
