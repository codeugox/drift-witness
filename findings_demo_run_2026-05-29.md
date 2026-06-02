# Findings — First End-to-End Demo Run
## VR-SENTIMENT-01 Drift Monitor · 2026-05-29

This document captures observations from the first end-to-end run of
the DistilBERT drift monitor against both a STABLE and a CRITICAL
production batch. The purpose is to document measured behavior,
control implications, and follow-up actions for the public Phase 1
demo. The run sequence: original end-to-end run on 2026-05-29; PSI
added on 2026-05-31 (see addendum); shipped sample evidence
regenerated with PSI on 2026-06-01.

---

## Executive summary

- The initial estimated model-card baseline did not match measured
  model behavior; the locked baseline is the canonical source.
- The critical batch produced compound drift across confidence,
  label distribution, embedding space, and input characteristics.
- The model became *more* confident on out-of-distribution inputs —
  a paradoxical overconfidence signal, not an improvement.
- The LLM judge and defensive parser translated statistical signals
  into reviewable governance evidence without breaking the audit
  trail when the first judge response truncated mid-JSON.

---

## Finding 1 — The "phantom baseline" problem

### What we observed

The model card [model_card_VR-SENTIMENT-01.md](model_card_VR-SENTIMENT-01.md)
claims an "approved baseline" with these values:

| Metric | Model card claim | Measured reality (locked baseline) |
|---|---|---|
| POSITIVE rate | **71%** | **43.3%** |
| NEGATIVE rate | 29% | 56.7% |
| Avg confidence | 0.912 | 0.9237 |
| Low-conf rate (<0.70) | 4.1% | 6.7% |
| Avg input length | 54.3 chars | 64.3 chars |

The POSITIVE/NEGATIVE rates are essentially **inverted**.

### Source of discrepancy

The initial model card metrics were drafted before `lock_baseline.py`
was run against the approved corpus. As a result, the values
reflected an *estimated expectation* of model behavior rather than
*measured output* from the deployed classifier. The estimate
reasoned at the **GRC semantic layer**: these texts describe
routine, successful, compliant operations → therefore "positive" →
therefore the model should classify them as POSITIVE.

That reasoning is wrong. DistilBERT-SST-2 does not classify by GRC meaning.
It classifies by **sentiment language patterns learned from movie reviews.**
Many baseline texts that a human GRC reader would call positive
("Vendor submitted required security documentation on schedule",
"Risk register updated. No new high risks identified", "Firewall rule
review completed. Forty-two rules removed as obsolete") contain lexical
features — *required*, *risk*, *removed*, *obsolete*, *test*, *exception* —
that the SST-2 training distribution associates with negative sentiment.

### Control implication

**Even frontier LLMs cannot estimate small-model baselines reliably**,
because they reason about the *semantic intent* of the input while the
deployed model reasons about the *lexical features* it was trained on.
The two abstractions diverge, sometimes spectacularly, in transfer-learning
deployments — which is what most production "AI" actually is.

**The control implication:** Model card baseline metrics MUST come from
measurement on the actual approved corpus using the actual deployed model
artifact. Not from estimation. Not from documentation borrowed from the
upstream model card. Not from the team's intuition. From running the
model and counting.

This is exactly why `lock_baseline.py` exists — to convert phantom
baselines into measured ones, then SHA-256 hash the result so nobody
edits the values after approval.

### Control update

The model card's "Approved Performance Metrics" table was reconciled
to the locked baseline JSON. The baseline file is now the canonical
source for measured values, while the model card explains why those
values may look counterintuitive to a GRC reader.

---

## Finding 2 — What actually pushes the model to drift

The CRITICAL_TEXTS batch was designed to trigger drift. We can now
decompose precisely *which* shifts each metric is responding to.

### The numbers

| Dimension | Baseline (locked) | STABLE batch | CRITICAL batch |
|---|---|---|---|
| N | 30 | 30 | 30 |
| POS rate | 43.3% | 40.0% | **3.3%** |
| Mean confidence | 0.9237 | 0.9281 | **0.9716** |
| Low-conf rate (<0.70) | 6.7% | 6.7% | **0.0%** |
| Very-high-conf (>0.99) | 27% (8/30) | 23% (7/30) | **67% (20/30)** |
| Mean length | 64.3 chars | 63.8 chars | **87.4 chars** |
| KS statistic | — | 0.13 (WATCH) | **0.43 (CRITICAL)** |
| PSI confidence | — | 0.0448 (STABLE) | **0.5122 (CRITICAL)** |
| JS divergence | — | 0.02 (WATCH) | **0.36 (CRITICAL)** |
| Embedding cosine | — | 0.04 (STABLE) | **0.13 (WARNING)** |

### The five drivers, ranked by contribution

**1. Negative-sentiment lexicon dominance (biggest driver of label-distribution drift)**

The critical batch is saturated with words that SST-2 training data
strongly associates with negative sentiment: *expired*, *delayed*,
*unresolved*, *refuses*, *no successor*, *unreachable*, *overdue*,
*departed*, *unresponsive*, *exfiltration*, *breach*, *ransomware*,
*disputed*, *contradicts*, *unauthorized*. The result: 29 of 30 inputs
classify NEGATIVE with extreme confidence (median 0.994).

This is what pushed POS rate from 43% → 3% and the JS divergence to 0.36.

**2. Crisis / urgency markers (amplifier on lexicon drift)**

"URGENT:", "CRITICAL:", "ESCALATION REQUIRED:" — these tokens are rare
in the baseline corpus and common in the critical batch. They push the
model toward extreme confidence rather than nuanced confidence.

**3. Out-of-distribution language (language shift)**

Three of 30 critical texts are non-English (Spanish and French). The
SST-2 model was trained only on English. Interestingly, DistilBERT *still*
classified them confidently — Spanish "no presentó" and French
"n'a pas fourni" contain enough recognizable negation tokens that the
model produced confident NEGATIVE predictions. **The model is right for
the wrong reason** — it doesn't understand the texts; it pattern-matched
on negation.

**4. Length shift (driver of embedding-space drift)**

Baseline mean length 64 chars vs. critical mean length 87 chars (+37%).
This contributes meaningfully to the embedding centroid cosine distance
(0.13) — `all-MiniLM-L6-v2` is sensitive to length and content density,
not just topic.

**5. Topical / domain shift (driver of embedding-space drift)**

Critical batch covers ransomware events, ITAR violations, M&A transitions,
LLM-generated attestations, legal holds, regulatory examinations — none
of which appear in the baseline. Even normalized for length, the embedding
centroid would move because the topical space is genuinely different.

### Why multi-metric monitoring matters

The drift isn't *one* shift — it's a **stacked compound shift across four
dimensions** (lexicon, urgency markers, language, length, topic).
That's exactly why monitoring needs multiple independent metrics rather
than just one. KS catches lexicon-driven confidence shift; JS catches
the label collapse; embedding cosine catches the topical/length shift;
low-confidence rate catches the "is the model uncertain?" signal.

A single-metric monitor would have missed at least one of these. The
multi-metric design isn't academic. It's necessary because real drift
events stack drivers.

---

## Finding 3 — The paradoxical overconfidence phenomenon

### What we observed

On the critical batch, the **low-confidence rate dropped to 0.0%** (vs
6.7% on baseline). 67% of predictions came back with confidence above
0.99 (vs 27% on baseline). The model became *more* confident on
out-of-distribution inputs, not less.

### Why this is counterintuitive but expected

A naive interpretation of "confidence" is "the model's estimate of how
likely its prediction is correct." Under that interpretation, OOD inputs
should produce LOWER confidence — the model "doesn't know" how to classify
them and should signal uncertainty.

That interpretation is wrong for fine-tuned transformer classifiers.
Confidence scores from a softmax output are not calibrated probability
estimates — they're **the softness of the model's commitment to a class
given its learned features.** When OOD inputs happen to *contain rich
lexical signals* (in our case, strong negative-sentiment words), the
model commits MORE strongly to the class those lexical signals support,
because the features it's anchored on are highly present. The model
doesn't know it's outside its training distribution. It only knows that
the words it knows how to weight are all pulling in the same direction.

### Governance interpretation by the LLM judge

Claude Sonnet 4.6 flagged this explicitly in its CRITICAL assessment:

> "Zero low-confidence predictions (0.0%) versus the approved 6.7%
> baseline is a paradoxical overconfidence signal — rather than indicating
> improved performance, this pattern is consistent with out-of-distribution
> inputs being misclassified with spuriously high certainty, a known
> failure mode of fine-tuned transformer models applied outside their
> training domain."

A pure statistical monitor would have read "low-conf rate dropped — that's
good, more confident predictions." A naive dashboard would have shown a
green metric. The LLM judge translated the math into the *risk-relevant*
interpretation: *absence of uncertainty signal is not the same as actual
certainty.*

**This is the most important governance signal from the run.**

---

## Finding 4 — "Right for the wrong reason" classifications

The critical batch had one POSITIVE classification:

> "This AI-generated compliance attestation confirms all controls are
> implemented per framework v4.2." → **POSITIVE, confidence 0.85**

A GRC reader would flag this as one of the **highest-risk** items in the
batch: an AI-generated attestation is exactly the kind of evidence that
should NOT receive an automatic POSITIVE rating without human review.

But the SST-2 model has no concept of "AI-generated content is suspect
in compliance contexts." It sees "confirms all controls are implemented"
and classifies it as positive sentiment, with reasonable confidence.

**The pattern this exposes:** sentiment-language classifiers can produce
classifications that are *correct from the sentiment perspective and
exactly wrong from the risk perspective.* No amount of statistical drift
monitoring catches this — the model is operating within its baseline.
You need either (a) human review of every POSITIVE classification in
elevated-risk vendor categories, or (b) a domain-specific classifier
trained on actual GRC labels.

This is a model card limitation, not a drift event. The model card
already calls out "AI-generated content not validated" but the demo run
shows what that limitation looks like when it actually fires.

---

## Finding 5 — The defensive judge parser earned its keep on Run 1

The first CRITICAL run truncated mid-JSON at `max_tokens=900` because
the verbose CRITICAL narrative exceeded budget. The hardened
`_parse_judge_json` helper caught it, wrote a structured "human review
required" evidence record with the truncated raw response preserved,
and the monitoring loop did not crash.

We then bumped `max_tokens=1500` and the second run completed cleanly.

### The governance point

The judge will fail in production. Models truncate, return malformed JSON,
get rate-limited, time out, get deprecated. Treating a parse failure as
a monitoring event (with auto-promotion to "human review required") rather
than as a crash means **the audit trail stays continuous** even when
components misbehave. That is a control implementation detail an
auditor or model-risk reviewer can evaluate.

---

## Resulting control updates

1. **Model card values reconciled to the locked baseline.** The
   approved performance metrics now match
   `baseline_v1.0_approved_2026-01-15.json` and
   `baseline_v1.0_summary.txt`.

2. **Known limitations updated.** The model card now documents
   paradoxical overconfidence on out-of-distribution inputs as an
   empirical finding, not only a theoretical limitation.

3. **Known classification failure patterns documented.** The model
   card now distinguishes drift from baseline model-purpose
   failures, including "right for the wrong reason" classifications.

4. **Evidence traceability strengthened.** Public sample outputs
   link the run evidence to the approved baseline hash and model
   card.

---

## Summary observations

- The classifier became more confident on out-of-distribution
  inputs, not less: 0.0% low-confidence calls versus 6.7% on
  baseline.
- The estimated model-card POSITIVE rate was 71%; measured baseline
  output was 43.3%, showing why baseline values must be generated
  from actual model runs.
- The critical batch demonstrated compound drift across lexicon,
  urgency markers, language, length, and topic.
- Defensive judge parsing preserved the monitoring record when the
  first CRITICAL judge response truncated mid-JSON.

---

## Addendum — 2026-05-31: PSI added to runtime

**What changed:** Added `compute_psi()` to `drift_test_simple.py`. The
baseline JSON already declared PSI thresholds (watch 0.05 / warning 0.10
/ critical 0.20), but the runtime was not computing or reporting PSI.
That documentation/runtime gap is now closed.

**Why now:** PSI is a familiar and widely used stability metric in
banking and credit model risk. For the target audience (FFIEC / FDIC /
federal GRC), shipping without PSI would have read as an oversight.
Adding it strengthens the financial-services positioning of the demo.

**Implementation choices:**

1. **4 quartile bins, not 10 deciles.** With baseline N=30, decile bins
   would average ~3 records each and produce unstable PSI. Quartile bins
   give ~7-8 baseline records per bin, which is defensible at this
   scale. The `compute_psi()` docstring documents the path to deciles
   for production-scale baselines.
2. **Baseline-quantile binning, not equal-width.** Equal-width bins on
   production data would conflate distribution change with binning
   artifacts. Quantile cuts from the baseline (with `-inf` / `+inf`
   endpoints) keep the comparison honest.
3. **Epsilon smoothing (1e-6)** to avoid `log(0)` when a production bin
   is empty.
4. **Explicit interpretation note in the judge prompt.** The judge is
   told that PSI is an investigation signal over confidence-score
   population stability, not direct proof of degradation. This prevents
   the judge from treating PSI as standalone evidence of accuracy loss.

**Measured PSI (2026-05-31 re-run):**

| Scenario | PSI | Status |
|---|---:|---|
| stable | 0.0448 | STABLE |
| critical | 0.5122 | CRITICAL |

These match the values predicted in the pre-implementation review.

**Judge behavior with PSI:** On the critical batch, the judge naturally
referenced PSI as the **second of three corroborating distribution
metrics** (alongside KS and JS), explicitly applied the interpretation
caveat from the prompt ("interpreted per model card guidance as an
investigation signal, not standalone proof of degradation"), and used
PSI to support — not replace — the paradoxical-overconfidence finding.
This is the kind of cautious, multi-metric reasoning a model-risk
reviewer expects.

**Files updated in this change:**

- `drift_test_simple.py` — new `compute_psi()` function plus wiring
- `README.md` — expected-output blocks, framework mapping, PSI note
- `model_card_VR-SENTIMENT-01.md` — PSI row footnote
- `evidence/drift_evidence_2026-06-01_stable.json` and
  `evidence/drift_evidence_2026-06-01_critical.json` — regenerated
  with PSI and aligned to the shipped sample summaries and reports
- HTML reports — now generated with a date suffix
  (`drift_report_<scenario>_YYYY-MM-DD.html`); the shipped sample
  reports keep the undated convention.

**What PSI does NOT do.** PSI does not prove the model is wrong. It
indicates the score population has moved relative to the approved
baseline. The governance action is investigation, not automatic
conclusion. This is the language used in the model card footnote and
the judge prompt note.
