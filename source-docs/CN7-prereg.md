# CN-7: Numeracy Midtrain with Cells as Validator

**Pre-registration v0.1 (amended v0.2 pre-7.1 — see §8) — Predictions pinned before any training**

Chris Hay | CN Programme | July 2026

---

## 1. Purpose

Test whether the CN-6 generation arm — dead at resolve@5 0.042 on v11 and
unrescued by a math-capable 1B base — can be revived by installing exactly
one thing: arithmetic competence *inside the emission grammar*, via a
numeracy midtrain of the v11 checkpoint in which the cell library authors
the curriculum, injects all beyond-tier answers under a loss mask, and
verifies every number.

Two theses are on trial, both falsifiable:

1. **The bottleneck thesis.** CN-6's generation failure is
   arithmetic-under-emission-format, not arithmetic per se. Evidence for:
   Llama-3.2-1B passed the 10/10 easy-arith probe yet reached only 0.306
   correctness in-format, with resolution pinned at the floor. Llama was
   never trained on the joint skill; the midtrain trains exactly it.
2. **The division-of-labour thesis.** A model should hold basic math
   natively and rely on cells for everything beyond, with the boundary
   enforced at the gradient level: beyond-tier answers are
   environment-injected and loss-masked, so no gradient ever trains them
   into weights. The call is the abstention.

The experiment is designed so that **failure is publishable**. If emission
correctness reaches its target and resolution still does not move,
generation is computation-limited in a way targeted training cannot fix at
~100M, the Llama result generalises, and delegate-by-pointing /
delegate-by-carrying stands as the final answer.

---

## 2. Pinned baselines

Every prediction below grades against a number that already exists. No
baseline may be re-measured after the midtrain except on the identical
protocol.

| # | Quantity | Value | Provenance |
|---|---|---|---|
| B1 | Generation resolve@5, held-out n=24, v11 | 0.042 | CN-6 stage 2 |
| B2 | Generation per-pair correctness, v11 | 0.097 | CN-6 stage 2 |
| B3 | Generation correctness, Llama-3.2-1B swap | 0.306 | CN-6 base swap |
| B4 | Generation resolve@5, Llama-3.2-1B swap | ~0.08 (CI ≤ 0.31) | CN-6 base swap |
| B5 | Oracle router ceiling, 6 examples, 249 value cells | 0.62 P@1 / 0.83 P@5 | CN-6 powered LOO |
| B6 | Router degradation, 1 of 6 examples wrong | 0.71–0.79 | CN-6 noise sweep |
| B7 | Extraction resolve@5, held-out n=24 | 0.875 [0.69, 0.96] | CN-6 stage 2 |
| B8 | Extraction example correctness | 0.979 | CN-6 stage 2 |
| B9 | Fingerprint held-out median rank, v11 seed 81, n=200 random | 98 (p75 227, 38% top-10%) | CN-1 faithful number |
| B10 | Fingerprint seed-invariance | rank 43/44/44 across seeds, std 0.47 (nulls: std 77–91) | CN-1 third dissociation |
| B11 | Held-out cells at generation correctness 0.00 | 12/24 (jacobi, crc16, isqrt, mobius, …) | CN-6 postmortem |
| B12 | Paraphrase cliff precedent | canonical 10/10 → narrative 0/10 | v11 KnnStore |
| B13 | Cell execution cost | microseconds per call; full 790-cell brute force ~1 ms | runtime |

---

## 3. Definitions

### 3.1 Tier frontier

**Tier A (in-weights numeracy)**, defined *negatively* by the cell
library — teach only what sits below the cheapest cell:

- single- and double-digit addition/subtraction
- small multiplication (up to 2-digit × 1-digit; times tables)
- integer comparison and ordering
- parity; small modulus; counting/successor

Explicitly excluded: multi-digit multiplication, anything
digit-manipulation-flavoured (checksums, digit reversal), modular
exponentiation, number-theoretic predicates. Under-teach on principle: the
tier exists to make small instances computable, not to blur the boundary.

**Within-frontier cell**: a held-out cell whose *small instances* reduce to
Tier A operations (e.g. small gcd by repeated subtraction is borderline —
classify each of the 24 held-out cells before the midtrain and freeze the
list; the classification is part of this pre-registration's artifacts).

**Beyond-frontier cell**: everything else. Expected to include most or all
of the 12 cells at correctness 0.00 (B11).

### 3.2 Data species (all cell-authored, all cell-signed)

| Species | Content | Loss |
|---|---|---|
| S1 — Tier A drill | in-tier arithmetic, 50:50 canonical ("7+5=12") : narrative-embedded (TinyStories register) | full loss incl. answers |
| S2 — Tiny-GSM interleaved | TinyStories-register word problems; per step, in-tier → model computes (answer in loss); beyond-tier → cell call emitted, result injected and **masked** | loss on text + call + continuation; **zero loss on injected results** |
| S3 — Emission transcripts | CN-6 grammar: k=6 oracle-correct I/O pairs per cell, deliberately varied/discriminative inputs | full loss |
| S4 — Replay | TinyStories, 30–50% of the mix | full loss |

Cells run at **corpus-build time**: results baked in, mask spans recorded.
Live execution is reserved for the eval harness and CN-7.6.

### 3.3 The mask

Per-token loss mask over environment-injected spans in S2. Property to be
audited, not assumed: **no beyond-tier answer token anywhere in the corpus
carries loss.**

---

## 4. Experiments

Order is cheap-first. The decision spine is 7.1 → 7.2 → 7.3; 7.0 and 7.4
are free measurements; 7.5 is the control that makes the result
defensible; 7.6 is conditional.

### CN-7.0 — Yield curve (instrumentation; one evening; no training)

**Protocol.** Sample current v11's emissions across the tier frontier
(same prompting as CN-6 stage 2). Cell-sign every pair. Report signed-pair
yield per tier and per cell.

**Predictions.** Tier A yield low but nonzero (0.05–0.15 per-pair,
consistent with B2); beyond-tier ≤ 0.01.

**Role.** (a) Pre-midtrain baseline on the identical protocol for every
later metric. (b) Decides cold-start vs bootstrap for CN-7.6. No gate.

### CN-7.1 — Corpus build + audits (data only; gate before any GPU time)

**Protocol.** Generate S1–S3 per §3.2. Then two consistency checks:

1. **Signature audit**: re-execute every answer token in the corpus
   against its cell. Target: 100% signed. Any mismatch is a pipeline bug.
2. **Mask audit**: verify zero beyond-tier answer tokens carry loss, by an
   independent code path from the one that wrote the masks.

**Rationale.** All four catches in the CN-1 lane were "two routes to the
same number disagree" (untied head, dropped norm, winner's curse, first-N
sampling). This lane builds the two-routes check into the pipeline instead
of discovering it post hoc.

**Gate.** Both audits clean, or no training happens.

### CN-7.2 — Midtrain + regression panel (the spend)

**Protocol.** Midtrain the pretrained v11 checkpoint on the S1–S4 mix
(10–20M tokens against v11's ~100M pretrain; hours on MPS). Full-model
update, replay ratio 30–50%. Then re-run the fingerprint-arm cell finetune
*identically* (792-token vocab extension, seed-81 protocol; 3 seeds if
budget allows — seed-invariance (B10) is the strongest dissociation and
worth re-confirming).

**Pre-registered panel** (all must pass to proceed to 7.3):

| Metric | Threshold | Grades against |
|---|---|---|
| P-a1 | In-tier probe correctness, canonical | ≥ 0.90 | — |
| P-a2 | In-tier probe correctness, narrative-embedded | ≥ 0.80 | B12 (the paraphrase slice is a **gate**, not a nice-to-have) |
| P-b | **Emission-format per-pair correctness, within-frontier** | **≥ 0.83** | B6 (router's graceful zone) — the load-bearing number |
| P-c | Extraction resolve@5, held-out | ≥ 0.79 (within B7's CI) | B7 |
| P-d1 | Fingerprint held-out median rank (seed-81 protocol) | ≤ 130 | B9 (~98 + noise allowance) |
| P-d2 | Fingerprint seed std (if 3 seeds run) | ≤ 10 | B10 (0.47 vs null 77–91) |
| P-e | TinyStories replay loss | ≤ +5% vs pre-midtrain | v10a-era eval |

**Kill criterion.** If P-c or P-d1 regress materially, stop; fall back to
the attention-only variant (FFN frozen, v10a-style) before touching 7.3. A
midtrain that buys numeracy by selling the two working delegation paths is
a net loss. Note the tension going in: fluency provably lives in attention
on this architecture (v10a), but arithmetic circuits in small transformers
tend to want FFN capacity — full-model-with-replay is the primary arm for
that reason, attention-only the fallback.

### CN-7.3 — Generation re-run, stratified (the headline)

**Protocol.** Exact CN-6 stage-2 protocol, held-out n=24, stratified by
the frozen within/beyond-frontier list from §3.1. Wilson CIs; n is small
and the split smaller — report per-stratum counts, not just rates.

**Predictions.**

| Stratum | Metric | Prediction | Grades against |
|---|---|---|---|
| Within-frontier | resolve@5 | ≥ 0.50 (strong: ≥ 0.70), climbing from ~0.08 toward the 0.83 ceiling | B1, B4, B5 |
| Beyond-frontier | resolve@5 | ≤ 0.15 (stays at floor) | B1, B11 |
| Within-frontier | per-pair correctness | ≥ 0.83 (re-confirms P-b in the live protocol) | B2, B6 |

**Reading the outcomes.**
- Both strata as predicted → the division-of-labour claim measured
  directly; generation joins pointing and carrying as a working path,
  *inside the frontier only*.
- Within-frontier fails **with P-b passed** → the informative null:
  emission-correct arithmetic still doesn't resolve; computation-limited
  in a way training can't fix at ~100M; the Llama result generalises;
  pointing/carrying is the final answer. Publishable either way.
- Beyond-frontier climbs → the boundary leaked; go to 7.4/7.5 to find the
  leak before believing it.
- The uninformative outcome is running 7.3 with P-b failed — hence the 7.2
  gate.

Known residue not addressed by numeracy: the correct-but-non-discriminating
tail (luhn/mobius: correct pairs, rank 49/20). That is an input-selection
behaviour and lives in S3's varied-inputs training; report it as its own
line, do not fold it into the stratum rates.

### CN-7.4 — Mask-leak probe (free interpretability)

**Protocol.** Post-midtrain, no cell access: probe beyond-tier arithmetic
directly. On S2-style interleaved problems: measure call-rate on
beyond-tier steps and unassisted-attempt rate.

**Predictions.**

| Metric | Prediction |
|---|---|
| Beyond-tier correctness, no cell access | ≤ 0.05 — despite thousands of masked exposures |
| Call-rate on beyond-tier steps | ≥ 0.95 |
| Unassisted attempts on beyond-tier steps | ≤ 0.05 |

**If it leaks** (beyond-tier competence entering through the reasoning
text around the masks): a genuine finding about where arithmetic forms —
feed it to the interpretability track. Either branch is a result.

### CN-7.5 — No-mask control (the ablation a reviewer will demand)

**Protocol.** Identical corpus, masks removed: beyond-tier injected
results now carry loss. Same finetune, same panel, same 7.3 and 7.4
measurements.

**Predictions.** (a) 7.4's beyond-tier probe goes nonzero via memorisation
of *trained* instances but does not generalise off the training
distribution; (b) the paraphrase slice (P-a2) pays for the extra
memorisation; (c) within-frontier resolution is not improved over the
masked arm. This isolates what the mask buys. Secondary ablations (Tier C
abstention slice on/off; replay ratio) only if 7.3 is ambiguous.

### CN-7.6 — STaR loop (conditional)

**Gate.** Re-run 7.0 on the midtrained model. Activate only if Tier A
signed yield ≥ 0.30 (sampling cost is the loop's only real cost — cells
verify at ~10⁻⁵ of step cost (B13), so verification is free and the model
is the bottleneck).

**Protocol.** Live cell verification in the loop: model emits, cells sign
or reject; signed emissions recycle into training data; rejected
beyond-tier emissions are harvested as boundary supervision (free Tier C).
Loss-level masking of unsigned pairs — zero gradient credit for a wrong
emission, which directly attacks the plausible-wrong failure mode (B6:
off-by-one pairs poison the router hardest).

**If yield stays low.** The loop waits; nothing above it blocks.

---

## 5. Decision spine

```
7.0 yield curve ──────────────► baseline + 7.6 gate input
7.1 audits ── clean? ── no ──► fix pipeline, do not train
        │ yes
7.2 midtrain + panel ── P-c/P-d regress? ── yes ──► attention-only fallback, re-panel
        │ pass
7.3 stratified generation ──► headline result (either branch publishable)
        │
7.4 mask-leak probe (free)     7.5 no-mask control (defensibility)
        │
7.6 STaR loop (yield-gated)
```

---

## 6. Budget

| Item | Cost |
|---|---|
| 7.0 yield curve | one evening, sampling only |
| 7.1 corpus + audits | data pipeline; no GPU training |
| 7.2 midtrain | 10–20M tokens, hours on MPS; + finetune re-run (×3 if seeded) |
| 7.3 / 7.4 | eval passes; cell verification ~free (B13) |
| 7.5 | second midtrain + finetune, same scale as 7.2 |
| 7.6 | sampling-dominated; activate only past the yield gate |

---

## 7. What this pre-registration commits to

1. The within/beyond-frontier classification of the 24 held-out cells is
   frozen **before** the midtrain and shipped as an artifact of 7.1.
2. No threshold in §4 moves after 7.1's audits pass.
3. Negative results ship: the (P-b passed, 7.3 failed) branch is written
   up with the same care as the positive branch.
4. Every number reported has two routes to it where feasible (signature
   audit, mask audit, per-stratum counts alongside rates) — the CN-1
   lesson, institutionalised.

---

## 8. v0.2 amendments (2026-07-15, before the 7.1 freeze; v0.1 text above unchanged)

Made after CN-7.0 ran and before any corpus generation or training. Thresholds
in §4 are untouched; what changes is provenance labeling, the substrate's id
space, and the P-d baselines. Decisions confirmed by CH.

### 8.1 The v11 tokenizer discovery (found wiring CN-7.0)

TinyModel v11's committed tokenizer artifacts (`tiny-model/tokenizer/v11/…`)
are a **different piece→id mapping** than the one the checkpoint was trained
with. Under the committed mapping v11 scores NLL 18.0 on TinyStories — worse
than uniform (11.2); under the original SP tokenizer (recovered at
`chris-experiments/compilation/15_v11_model/v11_tokenizer/v11.model`, the path
recorded in `training_results.json`) it scores **NLL 0.66** with fluent greedy
continuations. `artifacts/train_mask.pt` (8,599 of 71,261 ids) marks the
trained rows; decode must mask to it. The SP vocab has no `<call>`/`</call>`.

Consequence for the record: the CN-1 lane encoded through the committed
(wrong) mapping — **29.8% of its context tokens hit trained rows** (chance
12.1%, matched ~100%). CN-1's arm contrasts stand (all arms shared the
substrate), but B9/B10 are measurements of a mostly-untrained-embedding stack,
and any "leverages the v11 pretrain" framing of CN-1 is retracted. This is the
fifth catch of the "two routes disagree" family in this programme.

### 8.2 Substrate decision: SP id space + re-baseline

CN-7.2 midtrains v11 **in the original SP id space** (cell tokens and
delimiters appended above 71,261), so the real pretrain is preserved and S4/P-e
are meaningful (pre-midtrain TinyStories NLL 0.66 is the P-e reference).
Before the midtrain, the CN-1 fingerprint finetune is re-run on unmodified v11
under the SP mapping (seed-81 protocol, 3 seeds) to establish **B9′/B10′**;
**P-d1/P-d2 grade against B9′/B10′**, not B9/B10. P-d1's threshold becomes
median rank ≤ B9′ + 32 (the same ~"B9 + noise allowance" margin v0.1 used),
frozen the moment B9′ exists and before the midtrain starts.

### 8.3 Provenance corrections to §2

- **B1/B2/B7/B8** were measured on **SmolLM2-135M** (CN-6 stage-2 arms), not
  on v11 — v0.1's "v11" label was wrong. No emission protocol had ever run on
  v11 before CN-7.0. B1–B4 therefore serve as **cross-model floors** (the
  floor was base-independent: SmolLM2 and Llama-3.2-1B both pinned at it).
- **B4's** CI upper bound is **0.26** (findings: 0.083 [.02, .26]), not 0.31.
- **v11's pretrain is 24M tokens** (16M + 8M phase 3;
  `training_results.json`), not ~100M; §1's "at ~100M" and 7.2's "against
  v11's ~100M pretrain" read accordingly. The 10–20M midtrain is the same
  order as the pretrain, which strengthens the replay-ratio rationale.
- The on-disk `cn6_ckpt_generation.pt` is the **0..20 retrain** (the base-swap
  control overwrote the 0..1000 original — `cn6_train.py`'s checkpoint tag
  ignores `--input-max`). B1/B2 stand as recorded in the findings; any future
  re-measurement of the SmolLM2 arm is on the 0..20 substrate (overall
  correctness 0.166).

### 8.4 CN-7.0 results (both substrates; prediction graded honestly)

Frozen classification: **9 within / 15 beyond** (`cn7_frontier_classification.json`,
borderline register inside; square and isqrt WITHIN, norm2_sq BEYOND).

| Substrate | Within-frontier yield | Beyond-frontier yield |
|---|---|---|
| SmolLM2 CN-6 gen arm, 0..20 retrain (greedy + 8 samples @ T=0.7) | **0.289 [0.236, 0.349]** (70/242) | **0.079 [0.057, 0.109]** (32/405) |
| Raw v11, SP mapping | 0 parsed pairs | 0 parsed pairs |

- The v0.1 prediction (Tier A 0.05–0.15, beyond ≤ 0.01) **missed high on both
  strata** on the emission-finetuned substrate: sampling at T=0.7 and the
  0..20 retrain lift yields above the greedy-B2-derived band, and the beyond
  excess is concentrated in bitwise cells with seen siblings (mask_clear
  0.407, mask_has_all 0.259) plus low-entropy-output luck. The within/beyond
  **separation (3.7×, non-overlapping CIs) is the directional confirmation**;
  cross-check: overall signed rate 102/647 = 0.158 ≈ the 0.166 correctness the
  findings recorded for this substrate (two routes agree).
- Raw v11 has never seen the emission grammar (no `<call>` in the SP vocab):
  its in-format floor is **zero parsed pairs**, which is the number the S3
  species must move. CN-7.6's yield gate reads against the midtrained model,
  as v0.1 already specified.
- Yield outliers worth carrying into 7.3 expectations: is_lt_i16 0.963,
  clamp_i16 0.630, snap_down 0.370, isqrt 0.296 (within); zscore_q8, permille,
  fnv1a_step, geom_circle at 0.000 (beyond).

### 8.5 What v0.2 does NOT change

All §4 thresholds (P-a1…P-e, 7.3 strata predictions, 7.4 probes, 7.6 gate),
the decision spine, the species definitions, the mask property, and the
grading protocol (CN-6 stage-2 eval at 0..20, resolve@5, Wilson CIs) are
unchanged. The 24-cell classification is frozen as of this commit.

### 8.6 P-d1′ statistic pinned mid-re-baseline (2026-07-16, seeds 80/81 evaluated, seed 82 in
training with NO eval printed)

Written before seed 82's numbers exist, so the threshold cannot be accused of
being chosen after seeing the spread. Known at time of writing: s80
novel|seen = 105, s81 = 108; s81's training dynamics ran visibly hotter and
its seen-cell metrics are sharper across the board.

- **B9′ = the median over the 3 seeds of the novel_cell|seen_comp median
  rank** (random-sampled n=200 protocol). **P-d1′ = B9′ + 32.** Median-of-
  seeds, matching the v0.1 "B9 + noise allowance" spirit.
- **B10′ is computed per bucket, not pooled**: the invariance claim rides on
  novel|seen ONLY. novel|novel (n=48) is reported alongside as its own line —
  s80→s81 swung 286→71 there, and the reading (a stable address, noisily
  READ under composition shift through the seed-dependent association
  machinery) is recorded as a hypothesis, not folded into the headline
  invariance statistic. Small-n caveat stands: a 48-item median has wide
  sampling error on its own, so the swing is suggestive, not established.
- Two directional expectations pinned for seed 82, zero cost if wrong:
  (i) novel|seen lands within a few ranks of 105–108 while novel|novel may
  land anywhere; (ii) the seen-cell ordering novel_comp ≥ seen_comp on top1
  (s81: 0.455 vs 0.360) — if it reproduces, the pointing path reads
  descriptors compositionally rather than recalling training pairs, a free
  generalisation result the mismatched stack could not have shown.

### 8.7 B9′/B10′ RESULT (2026-07-16; §8.6 statistic applied as frozen)

Per-seed novel|seen median ranks: **105 / 108 / 93** (s80/s81/s82).

- **B9′ = 105** (median-of-seeds). **P-d1′ = 137, frozen.** Old faithful
  B9 ≈ 98 → the CN-1 held-out level is CONFIRMED on the fixed stack; the
  tokenizer repair changed nothing the fingerprint mechanism owed it.
- **B10′ (novel|seen) = std 7.94** across three VISIBLY different training
  trajectories (s80 slow, s81/s82 hot; seen-cell metrics scatter across
  seeds). Weaker than the old eerie 0.47 — which came from near-identical
  trajectories — but 17× tighter than the same seeds' novel|novel spread
  (std 136.8 at n=48), which is the within-run dissociation that carries the
  claim: the address is data-determined; the association pathway and the
  composition-shifted read are trajectory business.
- Registered expectation (i): PARTIAL. s82 landed at 93 — direction held
  (all seeds within ±12 of the band on a 790 scale) but "within a few ranks
  of 105–108" was tighter than reality. Recorded as graded.
- Registered expectation (ii): HELD at s82 (novel_comp top1 0.390 ≥
  seen_comp 0.370; s81 0.455 ≥ 0.360). s80 is the counterexample (0.280 <
  0.305) — so "compositional reading ≥ pair recall" stands at 2 of 3 seeds:
  supported, not established.
- Seen-cell addressing on the fixed stack, all seeds: top1 0.31–0.46,
  median rank 1–5 of 790 — versus ~0.27–0.30 / rank ~60s on the mismatched
  stack. The pointing path's substrate sensitivity is confirmed in triplicate.

### 8.8 P-e improvement decomposition — registered mid-midtrain (step ~11,700 of ~17,500,
checkpoint not yet saved; P-e readings through step 10,000 known: 1.6994 → 1.6177)

P-e is not passing but IMPROVING (−4.8% at step 10k). Before that becomes a
"mixed curriculum subsidizes fluency" claim, the mundane rival — continued
pretraining on an unconverged 24M-token base, where ~6.75M more TinyStories
tokens push val NLL down with zero cross-species transfer — must be ruled out.
Registered before the checkpoint exists:

- **Power check, already run (its result is data, disclosed here): the
  digit-form transfer story is dead on arrival.** Only 7 of 8,725 val
  sentences contain a digit — TinyStories spells numbers as words — so the
  observed gain lives almost entirely on digit-free prose and cannot be
  digit-regularity transfer. Any surviving transfer story must cross surface
  forms (S1/S2 drill digits → word-number prose), a strictly weaker prior.
- **The powered split**: sentence-level (boundaries at '.'), strict cardinal
  number-words (one…twelve, twenty, hundred) — 541 sentences vs 8,184
  without. Decompose the pre→post val NLL gain over both slices.
  Transfer story predicts the cardinal slice gains MORE (relative), and the
  deck shows word-number/counting continuations handled more fluently while
  digit-free prose stays near-identical to R0. Continued-pretraining story
  predicts uniform gains (and any deck drift broad, not number-localised).
  Stated lean at registration time, given the power-check result:
  continued-pretraining is now the favourite.
- **CN-7.5 arbitrates for free**: identical replay fraction → an identical
  P-e trajectory implicates replay volume; a materially different one
  implicates the mask/species interaction.
- **Honesty ledger**: the P-e bound (+5%) was frozen against an undertrained
  floor, which made this gate EASIER than its spirit intended — a bound meant
  to cap forgetting was set where mere continued exposure improves the
  metric. Passing is passing and the threshold does not move, but the bias
  direction is recorded so the trivial pass is not over-read. The asterisk
  expires as the ladder consumes tokens: if P-e is still improving by the
  third rung on a converged base, the transfer reading gets real.

### 8.9 Panel gradings as they land (2026-07-16, post-checkpoint; disclosures included)

- **P-e: PASS, improving** — val NLL 1.6994 → 1.5868 (−6.6%), bound never
  approached. **§8.8 split GRADED: the continued-pretraining story wins.**
  Cardinal-word slice gained +6.11%, plain prose +6.37% — uniform, no
  number-localised transfer; deck corollary consistent (register unchanged,
  no numeral-localised fluency shift). The "curriculum subsidizes fluency"
  claim is hereby NOT made; the correct sentence is "45% replay on an
  unconverged base is rehearsal-as-subsidy for trivial reasons" (§8.8's
  asterisk stands, expires as the base converges).
- **N5 (role-NLL leak signature): s3 CONFIRMED, s2 letter missed in the safe
  direction.** Fresh-instance s1|answer 4.49 → 0.051 (rule acquired);
  s3|value_in_tier 4.04 → 0.77 vs value_masked 4.14 → 4.35 (asymmetry created
  exactly where loss was allowed; masked at floor). s2|injected did NOT stay
  within 1 nat — it was squeezed 4.23 → 11.0: the model now has sharp,
  WRONG expectations for beyond-tier answers (the plausible-wrong signature,
  visible in NLL space before 7.3 runs). No leak on any reading.
- **P-a2 (narrative): PASS** — generation probe 0.965 [0.942, 0.979]
  (n=400). The B12 paraphrase cliff is not just closed but inverted.
- **P-a1 (canonical): FAIL as generation-instrumented (0.552), 0.910 on
  arithmetic-only argmax. DISCLOSURE: the second instrument was chosen after
  seeing the first fail.** Diagnosis (mechanical, reproduced): canonical
  drill rows carried no termination supervision (no EOS in S1 rows — a
  corpus format gap, filed for the next rung), so the model runs on with
  digits after the correct answer ("3 x 10 = " → "300 3…" = correct "30" +
  run-on), and the probe parser reads the run-on as a wrong answer. BOS/
  boundary variants ruled out (54/49/49 per 100). Teacher-forced arithmetic:
  canonical 0.910 (weak ops: mod 0.72, mul 0.76), narrative 0.970.
  **Gate handling**: P-a1 is recorded as instrument-confounded, not passed.
  The panel's proceed/stop decision transfers to P-b, which is measured in
  the actual 7.3 grammar — where separators ARE supervised (S3 rows end in
  " </call>"), so the termination confound is absent from the format that
  matters. If P-b fails, 7.3 does not run as a graded experiment.
- Weak-op note carried into 7.3 expectations: mod and mul are the real
  canonical residuals; within-frontier cells snap_down (mod-shaped) and
  square/isqrt (mul-shaped) are the likely drag points.
- **P-b: FAIL (0.111 [0.044, 0.253] within-frontier; greedy).** Diagnosis is
  mechanical and committed: free-running emission collapses to a
  template-echo mode ("10 10 = 1"-shaped, arity-adapted, answer echoing an
  input) on held-out AND seen descriptors alike — not a composition failure;
  teacher-forced knowledge intact (0.91/0.97). Cause per the role-NLL:
  S3 in-tier values undertrained at 0.77 nats (~p 0.46/token; S3 was 11% of
  the mix) — a weak conditional survives teacher forcing and mode-collapses
  under greedy. **7.3 ran and is RECORDED-NOT-GRADED** (resolve@5 0/9 within,
  0/15 beyond) — the uninformative outcome the gate exists to name, and it
  did its job. Sampled yield on this checkpoint (7.6 gate input): within
  0.110 [0.080, 0.149] / beyond 0.135 [0.110, 0.165] — no stratification,
  within-number carried by binary-output luck (is_lt_i16 0.389, eq 0.194),
  square/isqrt/snap_down ~dead. **CN-7.6 gate (≥0.30): NOT met; the loop
  waits.**
- **P-d1′: FAIL — KILL CRITERION FIRES.** Seed-81 fingerprint finetune on
  the midtrained base: held-out (novel|seen) median rank **208** vs
  threshold 137 (pre-midtrain B9′ = 105; novel|novel 557 ≈ chance). Seen-cell
  addressing simultaneously IMPROVED (median rank 1, top5 0.60/0.67 — best
  yet): the full-model midtrain sharpened association and damaged W_f
  generalisation, the exact FFN-vs-attention tension §4 flagged going in.
  Remaining full-model fp seeds stopped (criterion needs no more precision).
  **Action, per the registered fallback: attention-only midtrain (FFN
  frozen), same corpus, same 15M tokens, then re-panel.** Launched
  2026-07-16.
- **Fallback outcome branches, stated BEFORE the attention-only panel runs**
  (the first two were implicit; the third was not, and it is the best case
  for the broker architecture):
  (1) attention-only preserves P-d1′ but loses material numeracy →
  capacity/plasticity competition at 115M; ladder must budget the trade.
  (2) attention-only also fails P-d1′ → the damage is not FFN-mediated;
  the midtrain objective itself (S3 descriptor→spec training) overwrites
  what the fingerprint finetune needs.
  (3) attention-only preserves P-d1′ AND lands numeracy near the full arm
  (0.91/0.97) → no competition existed; **FFN plasticity was simply a
  hazard** — the skills fit in attention (v10a redux, one level up), and
  FFN-frozen becomes ladder POLICY, not fallback. In broker terms: the
  knowledge-band FFN is not merely prunable, it should never have been
  trainable.
- **Disambiguation instrument, registered before the fallback reports
  (`cn7_geometry.py`)**: cell-side fingerprints are frozen by construction
  (W_f's INPUTS cannot move), so drift-vs-capacity is decided in the TARGET
  space — the hidden-state geometry at the emission position that W_f(FP)
  must project into. Measure pre- vs post-midtrain hidden states on the
  fixed eval contexts (the B9′ bucket + seen bucket, same Random(0)
  items): RSA (pairwise-distance correlation), orthogonal Procrustes
  residual, linear-CKA, per bucket. Reading: geometry moved → drift story
  (sharpened number representations broke the smooth geometry W_f
  generalisation rides on — plausibly the SAME sharpening that pushed seen
  addressing to rank 1); geometry stable → W_f fit/trainability failure
  from the midtrained init. Interacts with branch (3): if attention-only
  preserves P-d1′, knowing whether IT moved the geometry separates
  "attention training is safe" from "geometry didn't move this time."
- **CN-7.5 watch-number (from the N5 s2 result): the 11.0-nat injected-span
  NLL.** The no-mask control's most direct signature is that number
  collapsing toward the in-tier value level — memorisation of trained
  instances entering exactly where the mask forbade it.

### 8.10 Fallback verdict (2026-07-16): branch (1) — substrate competition, confirmed
bidirectionally

Attention-only arm, full panel:

| | full-model arm | attention-only arm |
|---|---|---|
| s1 answer NLL (fresh) | 0.051 | 0.308 (6× weaker; mul 0.375, mod 0.610) |
| P-a canonical / narrative | 0.552† / 0.965 | 0.230 / **0.620 — P-a2 FAILS on real arithmetic** |
| P-b | 0.111 FAIL | 0.053 FAIL |
| P-e | −6.6% PASS | −3.5% PASS (half the subsidy — consistent with §8.8) |
| mask (N5 family) | held (injected 11.0) | held (injected 9.56; s3 masked 3.80, in-band) |
| geometry drift vs raw | RSA 0.638 / Procr 0.716 | RSA 0.587 / Procr 0.723 — **EQUAL** |
| **P-d1′ (≤137)** | **208 FAIL — kill fired** | **98 PASS — at the pre-midtrain level (B9′ 105, B9 98)** |

†instrument-confounded, 0.910 arithmetic-only (§8.9).

**Findings.** (1) Branch (3) is dead: arithmetic wants FFN capacity — frozen-FFN
starves it (0.051 → 0.308 fresh-instance; P-a2 fails outright). (2) Branch (1)
is confirmed in both directions: FFN training installs numeracy and breaks
computed addressing (105 → 208); FFN freezing protects addressing perfectly
(98) and fails the numeracy gates. **At 115M under this recipe, in-weights
numeracy and fingerprint addressing compete for the same substrate.**
(3) The geometry instrument earned its keep by ELIMINATING its own headline
story: both arms drifted the measured emission-position geometry equally, yet
addressing survived only where FFNs were frozen — so global drift magnitude is
not the operative variable; the damage is specifically what the fingerprint
protocol cannot re-fit through FFN changes. Recoverability, not motion.
(4) Neither arm reaches a graded 7.3: the 7.2 gate did its job twice.
Seen-cell addressing improved in BOTH arms (median rank 1–3) — association is
robust to everything; it is only the computed address that is fragile, and
only to FFN plasticity.

**R1 disposition.** 7.3 remains unrun-as-graded on both arms (recorded 0.000
on the full arm). CN-7.5 (no-mask control, full-model arm — the arm whose
11-nat signature it tests) launched 2026-07-16 as the last registered spend.
R2 prescriptions accumulated by the instruments: raise S3 fraction (0.77→
undertrained), add S1 EOS supervision, and either budget the FFN trade
explicitly or route numeracy around weights entirely — the broker reading,
which this rung's own failure mode now argues for from the inside.

### 8.11 CN-7.5 first readings + the R2 decision rule, FROZEN before the deciding probes
(2026-07-16; the noise probe (§9) and the off-distribution probe are built/being built but
have NOT run at commit time)

**Two registered arbiters fired:**
- **§8.8, second arbitration: replay volume confirmed.** The no-mask arm's
  P-e trajectory (−6.5%) is identical-within-noise to the masked arm's
  (−6.6%) — the pinned "identical replay → identical trajectory" signature.
  Continued-pretraining wins twice; the cardinal-word split stands as
  confirmatory; the cross-species-transfer story is closed.
- **7.5 prediction (a): graded WRONG in the interesting direction.** The
  11.0-nat mask signature collapsed on cue (s2 injected 11.01 → 0.40; s3
  masked 4.35 → 0.97) — but on FRESH instances (seed 981), which is
  within-distribution generalisation, not the predicted instance
  memorisation. The honest sentence: **a 115M model CAN learn beyond-tier
  arithmetic given gradient; the boundary the masked arm respected was the
  mask's, not capacity's.** Clean split of the thesis: the mask is now
  PROVEN as a boundary instrument (by ablation), while the assumption that
  the tier frontier tracked a capacity frontier is overturned pending the
  two probes. The frontier was policy. Whether it needed to be is what
  7.4n and the off-distribution probe decide. (Caveat kept in view:
  within-distribution learnability is the weakest form — bounded ranges, a
  few dozen functions, single-digit tokenization, ~245k exposures.)

**The R2 decision rule, frozen now because hindsight will pretend the fork
was obvious:**
- **If** 7.4n shows the no-mask beyond-tier competence is noise-fragile
  (dies below the Tier-A half-life) **or** the off-distribution probe
  collapses → **option (iii)**: consolidate on pointing + carrying +
  in-tier weights; retire emission-by-computing; **FFN-frozen adopted as
  ladder policy regardless** (it protects the geometry and costs nothing
  R1 measured that (iii) still needs). The crammed-distribution result
  would mean weights fake arithmetic in-range while cells remain the only
  trustworthy path — the broker wins with evidence, not ideology.
- **If** the competence is noise-robust **and** extrapolates → the tier
  CONCEPT is rewritten, not the mix ratio: multi-digit arithmetic
  compresses at 115M after all, and "below the cheapest cell" becomes a
  pure economics boundary (verification + editability still favour cells;
  capacity no longer does). Programme-doc change, new registration.
- Option (i)'s re-anchor idea is not bought blind either way: the
  fingerprint-stability read (geometry instrument) prices it — if the
  emission-position geometry moved (it did: RSA ~0.6), re-fitting W_f
  longer is fighting drift with capacity; a re-anchor experiment needs its
  own registration with that prior stated.
- Precision on what (iii) retires: emission-by-computing only — already
  demoted by CN-6, failed revival in R1. It KEEPS in-tier weights
  (passed), pointing (best-ever), carrying (never threatened), and the
  7.6 loop as the eventual route if signed yield ever clears the
  (now entropy-corrected) gate by other means. Consolidation, not retreat.

**Off-distribution probe: the discriminating SHAPE, frozen.** Crammed
distributions fail abruptly at the training-range boundary (fine at
max-trained-digits, cliff one digit past — and the cliff's location is the
training range echoed back, doubling as a corpus-consistency check);
compressed circuits degrade gracefully with carry depth. The failure
PROFILE, not the rate, is the verdict. Saturated instances (u16 clamp)
are excluded from grading — a constant answer is trivially learnable and
would fake robustness.

**Dose-response expectation (registered before the no-mask fp run
prints):** no-mask had MORE FFN change than masked-full, so if geometry
damage scales with FFN change, its P-d1′ lands ≥ 208 — giving a
three-point dose-response curve (FFN frozen: 98; masked-trained: 208;
unmasked-trained: ?) — as close to a causal gradient as this gets without
surgery.

### 8.12 R1 closing frame (2026-07-16; noise/off-dist/corrected-yield probes still unrun)

- **The ~100 ceiling is an invariant of the frozen FFN.** Held-out rank
  ≈100 has now been returned by: the mismatched-tokenizer stack (B9 98),
  the healthy stack across three divergent trajectories (105/108/93), and
  a base whose ATTENTION was retrained on 15M tokens of a different
  curriculum (98). Invariant to tokenizer mapping, seed, trajectory
  temperature, and attention weights; sensitive to exactly one thing —
  FFN training — in both doses tested. The no-mask fp run (expected ≥208)
  completes the dose-response curve.
- **The drift story, graded WRONG as stated** (it was mine): equal global
  drift with opposite P-d1′ outcomes means the damage is specific
  structure invisible to RSA/Procrustes/CKA. Recoverability-not-motion.
  Interpretability handoff: something in the FFN that global similarity
  measures cannot see is load-bearing for behaviour→address.
- **R2 is over-determined, not forked.** Every arm that bought arithmetic
  sold the geometry; the arm that kept the geometry could not buy
  arithmetic. A measured constraint surface with no point in the good
  corner at 115M → option (iii) + FFN-frozen policy is the only
  consistent configuration at this scale. The probes decide the TIER
  CONCEPT (capacity vs economics boundary) and v12 sizing (how many
  parameters before the good corner exists) — not R2.
- **The mask's job description, amended by its own ablation**: not
  capacity protection (7.5 killed that) but GEOMETRY protection — keeping
  gradient off content that would otherwise recruit the FFN. The thesis
  exits stronger and more specific than it entered: division of labour is
  not imposed on a tiny model; it is what a tiny model IS, and the mask
  respects a boundary the parameter budget draws anyway.
- **Registered prediction for the permutation-null yield recompute (before
  it runs)**: excess-over-null ≈ 0 within CI on every cell, including the
  flattering ones (mobius_function, is_lt_i16) — the entire 0.110/0.135
  sampled yield is signing-by-chance; the honest free-running number is
  zero.
- R-next drill note: the word-number cliff confirmed with structure
  (digit-narrative 0.965 → word-number canonical 0.552 = real surface gap,
  partial transfer); the digit↔word bridge goes in any future drill mix.
- Abstract sentence banked for the writeup: "At 115M parameters, we could
  teach the model arithmetic or preserve its ability to address tools it
  had never seen, but not both; the constraint is FFN-mediated, invisible
  to representational similarity, and indifferent to everything else we
  varied."

### 8.13 The no-mask fingerprint result overturns the mechanism as frozen (2026-07-16)

**Dose-response expectation (§8.11): WRONG. No-mask P-d1′ = 108 — PASS, at
the invariant level** (seen buckets best-yet: top1 0.415/0.475, rank 2/1;
novel|novel 637, n=48 scatter as usual). Gradings cascade, recorded before
any replicate runs:

- §8.12's "~100 ceiling is a constant of the frozen FFN": **amended** — the
  ceiling also survives full-FFN training WITH full loss. Damage table:
  attn-only/masked 98 · full/no-mask 108 · full/masked 208. The invariant
  survives everything tested EXCEPT masked-objective × FFN-training.
- The banked abstract's "FFN-mediated" clause: **retracted as stated**. The
  candidate replacement is stranger and sharper: enforcing the tool-boundary
  (mask) while training the FFN is what breaks tool-addressing.
- §8.10 branch (1) "substrate competition": **narrowed** — competition was
  observed only under the masked objective. The no-mask arm occupies the
  good corner (numeracy best: s1 0.048, beyond-tier 0.40; addressing 108;
  P-a2 0.960 PASS ≈ masked arm's 0.965 → 7.5 prediction (b), the paraphrase
  tax, graded WRONG; P-e PASS) — at the price of dissolving the boundary
  itself. 7.5 prediction (c) HELD: P-b 0.114 FAIL, resolution 0/9 —
  mode collapse is objective/decode-side, indifferent to what the weights
  know.
- **Interaction hypothesis, registered pre-replicate**: the masked arm's
  11-nat squeeze is not a side effect but the damage vector — masking
  beyond-tier values under FFN training teaches number-representations
  anti-aligned with true function behaviour, and a behaviour→address
  projection (W_f) cannot generalise into an anti-aligned geometry. The
  no-mask arm learned the true functions; behaviourally-truthful
  representations are W_f-compatible. Prediction: geometry damage tracks
  the squeeze, not FFN change per se.
- **Winner's-curse guard, launched before believing any of this**: single
  seeds so far. Replicates: masked-full fp seed 82 (does 208 reproduce?)
  and no-mask fp seed 82 (does 108 reproduce?). If the 208/108 contrast
  survives both, the interaction claim stands; if it collapses, §8.10's
  original reading returns and this section records a near-miss.
- Policy note pending replicates: if the contrast is real, the R2
  configuration space reopens — mask + FFN-frozen (boundary kept,
  addressing kept, numeracy weak) vs no-mask + FFN-trained (everything
  passes but the boundary is gone from the WEIGHTS, surviving only as
  verification economics). That choice is philosophical as much as
  empirical, and it is Chris's, not an instrument's.

### 8.14 The frozen rule fires (2026-07-16): option (iii); the tier boundary is real

- **Permutation-null yield (§8.12 prediction: CONFIRMED)**: within excess
  −0.002, beyond +0.007. The entire sampled yield was signing-by-chance;
  the honest free-running emission number is zero. 7.6 stays shut.
- **CN-7.4n (N1–N4)**: no capability died at any σ ≤ 0.08 in either arm —
  Tier-A 0.038→0.045, grammar flat at 0.0002–0.0004, no-mask beyond-tier
  0.918→0.960. N4 held (grammar most robust, trivially); N1/N2/N3's
  ordering questions returned "all robust in this range" — the probe's σ
  range was too conservative to produce half-lives, and fragility turned
  out not to be the discriminator anyway.
- **Off-distribution probe: THE CLIFF, exactly as frozen.** No-mask arm
  in-range: add 0.85 / round_to_multiple 1.00 / sub 0.75 exact. One digit
  past training range: **0.00 exact on every cell** (NLL 0.1→3.7–5.4;
  worse at B2). Cliff location = the training ranges echoed back
  (consistency check passes). Masked control: 10–13 nats all bands.
- **The dissociation the two probes bought jointly**: the no-mask arm's
  beyond-tier competence is a NOISE-ROBUST IN-RANGE INTERPOLATOR WITH ZERO
  ALGORITHMIC CONTENT. Crammed ≠ fragile — robust storage of a
  distribution is not a circuit; the failure PROFILE, not fragility, was
  the discriminator (as §8.11 froze it).
- **Rule verdict: option (iii) + FFN-frozen ladder policy.** The tier
  concept is NOT rewritten as pure economics: 115M memorizes the in-range
  surface of multi-digit arithmetic but does not compress the algorithm.
  "Below the cheapest cell" survives as a capability boundary for
  algorithmic content; in-range interpolation is learnable and worthless
  (cells cover the full domain at zero marginal cost, verified). The
  broker wins with evidence.
- FFN-policy footnote pending the §8.13 replicates: if masked×FFN
  interaction replicates, the precise policy is "the mask requires a
  frozen FFN"; the mask itself stays — the boundary is the programme, and
  §8.14 just re-proved the weights cannot replace it off-range.

### 8.15 The replicate arbitrates (2026-07-16): the 208 does not reproduce

**Masked-full fp, seed 82: held-out rank 93** (seen buckets 3/1, strong).
Damage table, updated: attn {98} · no-mask {108, s82 running} ·
masked-full **{208, 93}**. The §8.13 interaction hypothesis
(squeeze-as-damage-vector) is **unsupported**: the masked arm's addressing
survived at the invariant level on the second seed. §8.13 recorded itself
as a possible near-miss; it was one.

- **What actually happened to P-d1′**: the seed-81 kill-criterion firing
  was procedurally correct (the registered protocol was seed-81) and
  inferentially unlucky. FFN training does not deterministically destroy
  addressing — on present evidence (n=2) it makes the fingerprint
  finetune's outcome HIGH-VARIANCE across seeds ({93, 208} vs the frozen
  configuration's tight {98} + pre-midtrain {105, 108, 93}). The method
  lesson is recorded with the result: single-seed gates inherit seed
  variance; P-d gates in any future registration are multi-seed or
  variance-aware.
- **The ~100 invariant strengthens again**: it is now the MODE of every
  configuration tested — six of seven runs across broken tokenizer,
  three trajectories, attention-retrained, full-FFN-masked (s82), and
  full-FFN-unmasked bases land at 93–108; the one excursion (208) did not
  replicate.
- **What survives untouched**: §8.14's rule verdict — option (iii) rests
  on the cliff and the permutation null, not on P-d — and the numeracy
  side of the §8.10 trade (frozen FFN starves arithmetic: 0.051 vs 0.308,
  multi-instrument). The FFN-policy statement softens from "required to
  protect addressing" to "buys low-variance addressing at a measured
  numeracy cost that §8.14 renders moot (the numeracy purchasable is a
  non-extrapolating surface)".
- **Registered-then-mooted, kept for the record**: the
  prediction-under-enforced-ignorance mechanism and its within-run
  trajectory test (geometry damage tracking squeeze magnitude across
  intermediate checkpoints). Testable in principle; R1 saved no
  intermediate checkpoints (R2 trainer gets --save-every if anyone
  revives it); on present evidence there is no stable damage for it to
  explain.
- **Corrected-abstract watch**: the middle clause ("enforcing the tool
  boundary in the loss while training the FFN destroys tool addressing")
  is retracted with the interaction hypothesis. The honest clause is
  smaller: "single training runs can lose tool addressing to seed
  variance when the FFN is plastic." The first and third clauses stand
  pending nothing: the tier is a capability boundary for algorithms, and
  frozen-FFN-behind-explicit-mask remains the configuration that
  preserves everything worth preserving — now justified by variance and
  §8.14's economics rather than by a destruction mechanism.

Scoreboard through §8.15: eighteen graded registrations, seven wrong —
and every wrong one caught by an instrument that was registered before
the belief it killed. The evening's last lesson is the first one
institutionalised: two routes to every number, and no number believed on
one seed.

### 8.16 Correction to §8.15's economics, the third configuration, and the ~100 as
problem-intrinsic (2026-07-16; no-mask s82 replicate still training at commit time)

- **§8.15's "numeracy cost mooted by §8.14" CONFLATED two purchases and is
  corrected here.** §8.14's cliff moots BEYOND-tier numeracy only. The
  attention arm's failure was IN-TIER (P-a canonical 0.230/narrative 0.620;
  s1 answers 0.308) — Tier A itself, a FINITE domain where in-range
  learning is exactly the deliverable and extrapolation is not required.
  The frozen FFN's Tier-A cost is real and un-mooted. The R2 configuration
  space is therefore not closed by §8.14.
- **Third configuration, registered for the R2 prereg**: plastic-masked FFN
  + multi-seed W_f fitting protocol. The s82 result localises the variance
  in the FINETUNE, not the base (same masked base, two W_f fits: {93,
  208}); a fit is a 50-minute job. Fit 2–3 seeds, gate each against
  P-d1′, keep a passer. For science this is selection and is DECLARED as
  such (the gate must be multi-seed by construction and the selection
  step reported); for building the broker it is ordinary engineering,
  no different from rerunning a diverged job. If it holds, this
  configuration occupies the good corner: Tier-A numeracy in weights AND
  addressing at the invariant level, variance managed by cheap retries
  rather than avoided by starvation.
- **The ~100 as R1's headline discovery, upgraded to a hypothesis about
  the PROBLEM**: mode of six of seven runs across broken tokenizer, three
  trajectories, retrained attention, masked- and unmasked-FFN bases;
  seed-invariant where measured tightly. Candidate reading: W_f's
  held-out ceiling is intrinsic to the behaviour→embedding problem on
  THIS cell library, not to any training configuration — which reopens
  the α/scale question as a question about library geometry, and
  promotes the external-address-table architecture to the lever that
  matters. Registered as hypothesis, not finding; discriminating
  experiments (library-subset geometry sweeps) belong to a future
  registration.
- **Abstract, third form** (§8.12 v1 retracted, §8.15 v2 narrowed):
  "At 115M parameters, arithmetic beyond a small tier is learnable only
  as a non-extrapolating in-range surface; tool-addressing
  generalisation sits at an invariant level that survives every training
  configuration tested, with FFN plasticity adding variance rather than
  damage; and a frozen FFN behind an explicit mask — or a plastic one
  behind a multi-seed addressing gate — preserves everything worth
  preserving."
- **Writeup directive: protect the chronology.** The 208 firing the kill,
  the fallback producing the attention-arm data anyway, the replicate
  dissolving the mechanism — all within twelve hours, every step
  committed before its outcome was known. The sequence is the finding:
  the method surviving its own false alarm is the strongest
  demonstration it has. Cleaned-up papers erase exactly this; this one
  will not.

---

## 9. v0.3 addition (2026-07-16, pre-midtrain, outcome-blind): CN-7.4n, the weight-noise probe

Added while no midtrained checkpoint exists (re-baseline seed 81 mid-training;
B10′ unknown). This section ADDS a measurement with its own pinned directional
predictions; it moves no §4 threshold and gates nothing.

**Frame.** Hinton & van Camp '93 (MDL-of-weights): generalization comes from
keeping the information in the weights far below the information in the
outputs. The CN-7 mask is that principle enforced structurally rather than by
penalty — beyond-tier answers contribute zero bits to the weights, not few;
cells are where the incompressible bits live; W_f compresses a function to an
address. Under this frame the seed-80 dissociation reads naturally: seen-cell
addressing is stored association (bits in weights — substrate-sensitive),
novel-cell addressing is computed from behaviour through a data-determined
geometry (few stored bits — substrate-immune, and predicted seed-invariant).
Caution carried into any writeup: the bits-back accounting does not transfer
literally to autoregressive LMs; what transfers is the inequality
(weight-bits ≪ verified-output-bits), which this architecture satisfies by
construction. Lineage for related work: Hinton–van Camp '93 → bits-back /
variational → compression-as-intelligence; the architectural move here is
giving the incompressible bits somewhere else to live and verifying them there.

**Protocol (measurement-only, eval-cost).** On the midtrained checkpoint
(7.2 arm) and, when 7.5 runs, on the no-mask control: add iid Gaussian noise
to every trained tensor at relative scale σ·std(tensor),
σ ∈ {0.005, 0.01, 0.02, 0.04, 0.08}, 3 noise draws per σ. At each point
measure: P-a1/P-a2 drill correctness (200-item probes), beyond-tier
correctness with no cell access (the 7.4 leak metric), call-grammar validity
and call-rate on S2-style beyond steps, held-out fingerprint median rank
(on the fingerprint-finetuned ckpt), TinyStories val NLL. The readable signal
is ORDERINGS of degradation (which capability dies first), not absolute σ —
noise-fragility also reflects basin sharpness, so absolute values are
confounded; orderings within one checkpoint share the confound and cancel it.

**Pinned directional predictions.**

- N1: Tier-A drill correctness (canonical and narrative) is more noise-robust
  than any beyond-tier competence 7.4 finds: σ½(TierA) > σ½(beyond-tier).
- N2: if 7.4 finds a leak (beyond-tier correctness > 0.05 unassisted), it is
  a noise-fragile island — it dies at σ below the Tier-A half-life. A
  noise-ROBUST leak would falsify the memorisation reading and demand the
  interpretability track.
- N3 (7.5 contrast): the no-mask arm's beyond-tier trained-instance
  correctness is more fragile than that same arm's Tier-A correctness; the
  two arms' Tier-A robustness is comparable. "The mask kept the weights
  simple" becomes measured, not asserted.
- N4 (riskiest, stated anyway): call-grammar validity is the most
  noise-robust capability measured — it is the shortest program in the mix.

**N5 (added 2026-07-16, still pre-checkpoint; NLL-domain leak signature).** The
fresh-instance role-NLL instrument (`cn7_species_nll.py`, R0 floor committed:
s1|answer 4.49, s3|value_in_tier 4.04 vs value_masked 4.14 — symmetric before
training) predicts: post-midtrain, s3|value_in_tier collapses while
s3|value_masked and s2|injected stay within ~1 nat of their R0 floor. A large
drop on masked/injected spans without corresponding 7.4 generation competence
would indicate the model is learning beyond-tier answer DISTRIBUTIONS through
context (a soft leak the generation probe alone would miss).

**Tier-admission footnote (programme doc, not a change here).** "Below the
cheapest cell" and "compresses well in weights" should pick out the same
Tier A; where they disagree, the compression criterion is probably the better
judge. To revisit at the next tier-frontier revision, not in CN-7.

### 8.17 R1 CLOSED (2026-07-16): the variance reading is final

No-mask s82: **159** (seen buckets 2/1, strong; one PASS one FAIL against
137 within the same arm). Final table — frozen-FFN: {105, 108, 93} + {98},
spread ≤15; plastic-FFN: {93, 208} + {108, 159}, spread 51–115, mask-
irrelevant. **FFN plasticity multiplies W_f-fit variance ~an order of
magnitude; the ~100 invariant is where tight configurations sit.** §8.16's
multi-seed W_f gate is thereby NECESSARY for any plastic-FFN recipe, not
prudent. Single-seed P-d gates are formally retired from this programme.

R1 closes: nineteen graded registrations, eight wrong, every wrong one
executed by an instrument registered before the belief existed. The broker
loop ran end-to-end (cn7_broker.py): model parses prose, emits the call,
the cell answers exactly, the model narrates the verified number. The
chapter's last transcript: "<call> ⟨safe_div⟩ 157 16 </call> 9 sweets.
The children smiled."
