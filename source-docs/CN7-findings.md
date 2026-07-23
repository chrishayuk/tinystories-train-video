# Cell-Native Model Architectures — CN-0/CN-1/CN-2 findings

Status: **CN-0 complete, gate not met, CN-3 scoped out for Gemma-class
models** (hyperparameter sweep, operation breadth, narrative contrastive
probe + null test, all in wave 2) — three alternative explanations for the
generalization gap were tested and killed (under-tuning, addition being
representative, narrative lacking the information), and the mechanism that
survives (a fast-forming, flat, non-computational numeral encoding) gives a
principled reason for the scope-out, not just a failed number. Full verdict
and programme redirect (CN-1 next, not CN-3) in `## CN-0, read against the
gate, after two waves` below. **CN-1: a slice-0 toy pilot, three iterations,
concluded** — the pilot's real headline is architectural: weight tying
(embeddings ↔ output projection) is a load-bearing precondition of the whole
fingerprint-embedding hypothesis, not a debugging footnote, found by two arms
silently reading 0.000 everywhere before any number was trusted. Given
tying, fingerprint-init cleanly beats random-init on trained cells
(0.993-1.000 vs. a variable, sometimes much worse mean) — held loosely, as
an init-quality effect, not the novel claim. The harder novelty/held-out
question (gate ii) got a genuine, pre-registered test in iteration 3 (a
compositional grid, bar stated before the run: fingerprint-init > 0.5 /
random-init <= 0.25 on a held-out combination) and came back a clean FAIL —
both arms at exactly 0.000 — closing the toy-scale question rather than
leaving it open: no compositional generalization to modulate at toy scale,
full stop. Per the pre-registered fork, gate (ii) now moves to the real
build, not a fourth toy iteration — see `## CN-1 slice-0` below for all
three iterations' diagnoses. **CN-2: G2 complete at the
harness level** — measurement (60-problem battery, wrong-number baseline
0.016 after the plan-IR i32 signed lane closed the verifier's last coverage
hole), then the correction loop (truncate at a cell80-refuted claim, assert
the verified equation, deterministically continue): scoped wrong-number
rate **0.016 → 0.000 at ~4.8% overhead**, residue 100% non-arithmetic
(unscoped claims / wrong plans), per the pre-registered null. In-decoder
version couples to CN-1. All experiments are defined in
`cell-native-architectures.md`. Kicked off 2026-07-12. Code lives in
`cell-native-architectures/`.

## Infrastructure map (established before any code was written)

The doc's dependency line names LARQL, the Gemma circuit map, and the L30
injection result as available prerequisites. None of that lives inside
`cell80` — it's spread across sibling repos under `~/chris-source/`:

- **LARQL** (`~/chris-source/larql`, Rust, live, daily commits) — loads
  Gemma 3 4B natively, serves an OpenAI-compatible HTTP API
  (`larql serve <vindex> --port 8080`). `chuk-larql` (Python) is its
  superseded predecessor.
- **The "100% P(target)" / seq-exact 1.00 injection result** is real, from
  `chris-experiments/arithmetic_mechanism/a9_residual_alu.py`
  (`A9_VERDICT.md`) — a bespoke per-decode-step `mlx_lm` loop, not built on
  any packaged tool.
- **`chuk-mcp-lazarus`** (`~/chris-source/chuk-ai/mcp-servers/chuk-mcp-lazarus`)
  is prior-art naming precedent for LARQL's mech-interp primitives, not a
  dependency of either LARQL or the A9 result. Its `extract_activations`/
  `prefill_to_layer` tools are usable for readout but weren't used here —
  the already-proven `chris-experiments/arithmetic_mechanism` taps were
  faster to reuse.
- **cell80-py's `CellHost`** (`load`/`run`/`solve`) is solid and required no
  new plumbing — `host.solve(plan_json)` (the M2 plan-IR path used by
  `cell80/examples/m3_gsm8k_smoketest.rs`) verifies `add`/`sub`/`mul`/`div`
  on u32 operands in ~10ms after JIT, cached thereafter.

## CN-0 slice-0 — operand readout

**Script:** `cell-native-architectures/cn0_operand_readout.py`. Vendors the
`mlx_lm` load + per-layer last-position residual capture from
`arithmetic_mechanism/a1_trace.py`, and the Fourier/helix design-matrix
scaffolding (periods 2/5/10/100) from `a2c_helix_rotation.py` — reused in
the **read** direction (residual → operand value) rather than a2c's
**write** direction (value → injection vector).

**Setup:** Gemma 3 4B (`google/gemma-3-4b-it`, local MLX weights), 4 surface-
form families (`digit`, `word`, `mixed`, `narrative`) × 40 addition problems
each (`a, b ∈ [1, 99]`), residual captured at L12–L22 (straddling the doc's
L13–L21 band), three probe families (linear ridge, Fourier/codebook ridge,
small MLX MLP), scored on **held-out-family** exact-pair recovery (train on
3 families, test on the 4th, rotate) plus a pooled 80/20 random-split
baseline for reference.

**Result** (full run, 33s wall-clock, 160 forward passes):

| held out   | best probe | best layer | exact-pair |
|---|---|---|---|
| digit      | fourier | L22 | 0.150 |
| word       | fourier | L21 | 0.225 |
| mixed      | fourier | L22 | 0.250 |
| narrative  | —       | —   | 0.000 |

Linear and MLP probes sit at floor (0.000, one stray 0.025 hit) at every
layer for every held-out family. The **Fourier/codebook probe** is the only
one that shows a real signal, and it's structured, not noise: near-zero
through L12–16, then rising through L17–22, peaking at the top of the swept
band. This is consistent with `a2c_helix_rotation.py`'s prior finding that
Gemma encodes small numbers via small-period Fourier/clock components
rather than linearly — a probe that respects that structure is the one that
picks up signal here. `narrative` (the "Sam had... found... now has"
surface form) is flatly unrecoverable at this sample size — the operands
sit many tokens before the tap position, unlike the other three families
where they're adjacent to it.

**Read against the gate:** nowhere near the ≥95% gate, and every held-out
family is also below the 80% kill line. **This is not read as a kill.** At
N=40/family (~120 pooled training examples) against a 2560-dim residual,
the linear and MLP probes are almost certainly underpowered rather than
genuinely floor — ridge with an untuned λ=1.0 and an MLP given only ~120
examples both have plausible headroom a larger N would recover. The
Fourier probe's monotonic-ish rise toward L21–22 is the one piece of signal
worth trusting from this pilot: **the next run should sweep further past
L22** (the doc's own band stops at L21; this pilot's peak sits at its
*upper* edge) and scale N by 5–10× before treating the gate/kill as
actually evaluated.

**Not run this slice:** hyperparameter sweep on ridge λ, longer MLP
training, layers beyond L22, multiplication/subtraction (addition only).

## CN-0 rerun — 5× data, tuned λ, layers extended to L28

Addressed all three of slice-0's open items: `N_PER_FAMILY` 40→200 (800
prompts total), ridge `λ` 1.0→0.3, MLP epochs 300→400, layer sweep extended
from L12–22 to L12–28. Results saved separately from the slice-0 pilot
(`cn0_operand_readout_results_slice0_pilot.json` vs the current
`cn0_operand_readout_results.json`).

| held out   | best probe | best layer | exact-pair (was, N=40) |
|---|---|---|---|
| digit      | fourier | L24 | **0.400** (0.150) |
| word       | fourier | L23 | **0.365** (0.225) |
| mixed      | fourier | L23 | **0.575** (0.250) |
| narrative  | fourier | L25 | **0.030** (0.000) |

Every family roughly doubled or better. Linear and MLP are still at floor
everywhere (0.000–0.010) — this isn't a probe-family question, it's
specifically the Fourier/codebook probe carrying the signal, more strongly
now that it has more data to fit against. The peak layer moved from L21–22
to **L23–25**, past the doc's originally-scoped L13–21 band, confirming
last slice's suspicion that the band needed extending, not just re-running.

**The one genuinely new finding: a random-split vs. held-out-family gap.**
The pooled 80/20 random split (train and test both drawn from all 4
families, so the test set can contain surface forms the probe *has* seen
during training) hits **85.6% exact-pair recovery at L23** — close to the
95% gate, and the Fourier probe crosses 60–70% at several nearby layers
too. That's a large gap from the held-out-family numbers above (max 57.5%,
mixed). Read together, these say the operand information is strongly
present in the residual stream and well-localized (L23, respects the
Fourier/clock encoding) *within* a surface-form distribution, but doesn't
transfer *across* surface forms — most acutely for `narrative` phrasing,
which stays at 0–3% at every layer regardless of probe, even with 5× the
data. The mechanism doesn't generalize the way CN-0's gate wants it to; it
memorizes-per-surface-form rather than reading a form-invariant operand
representation.

**Read against the gate:** still below both the 95% gate and the 80% kill
line for every held-out family — narrowly missing "kill" only because the
trend is still visibly climbing with N (not flat), and `mixed` at 57.5% is
within plausible reach of 80% with more data or a wider probe family (a
proper MLP hyperparameter sweep wasn't done — `MLP_HIDDEN`/`MLP_LR` are
still slice-0 defaults). The random-split/held-out-family gap is the more
interesting result than the gate/kill verdict itself: it reframes the open
question from "can operands be read out at all" (yes, clearly, 85.6%) to
"what would it take to make that readout invariant to surface form"
(unanswered — narrative's near-total failure suggests the gap isn't just
about probe capacity).

**Not run this rerun either:** a real hyperparameter sweep (λ and MLP
architecture were single fixed choices, not searched), multiplication/
subtraction, or a dedicated narrative-vs-others contrastive probe design
(the current setup treats all 4 families symmetrically, but narrative's
failure mode looks qualitatively different from the other three's).

## CN-0 wave 2 — hyperparameter sweep, operation breadth, the narrative
## contrastive probe, and the null test that mattered most

Addresses all three items the rerun left open. **Read against the gate: still
scoping, not a pass or a kill** — this wave sharpens *why*, it doesn't move
the verdict.

### The one clean, load-bearing finding: linear sees nothing, Fourier sees almost everything

Across every op, every held-out family, every layer, every feature
representation tested this wave (tap, operand-position, mean-pooled) and
last wave: **linear probes never exceed 0.225**, often sitting at exactly
0.000. **Fourier probes range 0.5-1.0** on the same data. This is not a
margin, it's categorical. Pre-registering the Fourier/codebook family
*cheapest-second* (`cell-native-architectures.md`'s own "cheapest first"
ordering) is the reason CN-0 has a real signal to argue about at all — a
linear-only probe would have read as a clean kill (<80% everywhere, in fact
near 0% everywhere) and scoped out depths 2-3 on what would have been a false
negative. Operand information lives in a periodic/helical encoding, not a
linear one, exactly as the Fourier/clock literature on number representations
predicts — and it now has a receipt on Gemma 3 4B specifically.

### Hyperparameter sweep: rules out under-tuning, doesn't move the number

`cn0_operand_readout.py` rewritten with a real, honest sweep: ridge λ over
9 values, MLP over 4 hidden sizes × 3 learning rates — all selected via a
**nested inner validation split carved out of the pooled training families**
(never the true held-out family itself, which would leak the test set into
hyperparameter selection and inflate the number). Addition, full sweep:

| held out | exact-pair (tuned) | exact-pair (single fixed choice, prior wave) |
|---|---:|---:|
| digit | 0.400 | 0.400 |
| word | 0.365 | 0.365 |
| mixed | 0.575 | 0.575 |
| narrative | 0.030 | 0.030 |

**Bit-for-bit identical to the untuned baseline.** A real, properly
cross-validated sweep found nothing the single fixed choice hadn't already
found. This closes the "maybe it's just under-tuned" hypothesis the prior
wave left open: 57.5% (mixed, the ceiling so far) is very likely close to a
genuine representational limit for this probe class on held-out surface
forms, not a tuning artifact.

### Operation breadth: subtraction and multiplication generalize much better than addition — narrative doesn't, for any of them

Extended the battery to `sub`/`mul` (same 4 surface-form families, same
tuned-probe pipeline, `CN0_OPS` env var). Held-out-family exact-pair, best
probe/layer:

| held out | add | sub | mul |
|---|---:|---:|---:|
| digit | 0.400 | **0.850** | 0.585 |
| word | 0.365 | 0.660 | **0.865** |
| mixed | 0.575 | **0.925** | **0.895** |
| narrative | 0.030 | 0.385 | 0.120 |

Subtraction and multiplication both clear the 80% kill line on `mixed`
(0.925/0.895), and subtraction clears it on `digit` too (0.850) —
addition never clears 80% anywhere. **Narrative is the hardest family for
every operation tested, by a wide margin, though its floor moves**:
0.030 (add) -> 0.120 (mul) -> 0.385 (sub). None of the three operations
reach the 95% gate on any held-out family. Read plainly: operand-readout
generalization is *operation-dependent*, and addition — the only operation
the prior wave tested — happens to be the hardest case, not a representative
one. A CN-0 verdict based on addition alone would have been reading the
floor of the distribution, not its center.

### Mean-pooling: fixes in-distribution recovery, actively hurts cross-family generalization

Added a `CN0_FEATURE=mean_pooled` mode (mean over every token position's
residual, not just the last-token tap) to test whether the tap position was
simply the wrong read. Addition, held-out-family, tuned:

| held out | tap | mean-pooled |
|---|---:|---:|
| digit | 0.400 | 0.125 |
| word | 0.365 | 0.050 |
| mixed | 0.575 | 0.095 |
| narrative | 0.030 | 0.005 |

**Worse across the board, including narrative itself.** Mean-pooling
averages in each surface form's own filler tokens (narrative's "Sam had...
marbles and found... more" carries far more non-operand tokens than
digit's "94 + 62 = "), so the pooled vector becomes *more*
surface-form-entangled, not less — the opposite of what would help
cross-family transfer. A fix that helps in-distribution readout can still
actively hurt the generalization the gate actually measures; the two
questions ("is the information there" vs. "does the read transfer across
surface form") do not share an answer.

### The narrative contrastive probe: real information, precisely characterized — and the null test that mattered most

**Hypothesis:** narrative's near-total tap-based failure (0.030, both prior
waves) reflects a wrong-tap-position artifact, not missing information —
narrative embeds operands many tokens before the tap ("Sam had `{a}`
marbles and found `{b}` more. Sam now has "), unlike the other three
families where operands sit immediately before it.

**Method** (`cn0_narrative_probe.py`, new): locate each operand's own last
digit-token index via prefix tokenization (verified by hand: in "Sam had 72
marbles...", the '2' of '72' sits at token index 5, the tap at index 17).
Compare three feature reads, in-distribution (random 80/20 split within
narrative alone, N=200): **tap** (the original method), **operand_positions**
(concat of the residual at each operand's own token), **mean_pooled**.

| feature | best exact-pair (in-distribution) | best layer |
|---|---:|---:|
| tap | 0.925 | L5 |
| operand_positions | 1.000 | flat, L2-L26 |
| mean_pooled | 0.975 | L2 |

The hypothesis was right: reading at the operand's own position (or pooling
broadly) recovers the information almost perfectly *in-distribution* — tap
alone, at the right layer, already gets to 0.925 within narrative's own
distribution (vs. 0.030 when trained on the other three families and tested
on narrative, the held-out-family number). The information was never
missing; it just doesn't transfer from other surface forms' tap-position
geometry.

**A 1.000, flat across 8 layers, is not something to report without a null
check — so before writing any of this up, one was run.** Extended the layer
sweep down to the raw token embedding itself (zero transformer computation)
and the first few layers (L0, L1, L2, L5), on the same operand_positions
feature:

| layer | tap (fourier) | operand_positions (fourier) |
|---|---:|---:|
| embed (raw, 0 layers) | 0.000 | 0.000 |
| L0 | 0.000 | 0.975 |
| L1 | 0.725 | 0.950 |
| L2 | 0.750 | 1.000 |
| L5 | **0.925** | 1.000 |
| L19-26 | 0.525-0.725 | 1.000 (flat) |

**The raw embedding reads at exactly 0.000 — this is not the probe decoding
the tokenizer.** If it were, the trivially-available, lossless token
identity at the embedding layer would already recover close to 1.000, and it
doesn't. Something the model computes is necessary. But the more precise
finding is that whatever that something is, it forms almost immediately (by
L0-L1) and then stays essentially flat through L26 for `operand_positions` —
consistent with the residual stream's additive architecture (once an early
layer writes a clean, Fourier-decodable numeral code into a token's own
position, later layers have no strong pressure to overwrite it there). This
means `operand_positions`'s 1.000 is real, but it is better described as
reading a fast-forming, stable **numeral encoding at that token's own
position**, not evidence of extended in-flight arithmetic computation
building up with depth. `tap`, by contrast, shows a real depth profile in
this same test (0 at embed/L0, rising to a peak of 0.925 at L5, declining
through L19-26) — a genuinely different, depth-dependent signature.

**A structural point that limits what any of this can mean for CN-3, stated
plainly so the write-up doesn't overclaim:** `operand_positions` and
`mean_pooled` are not features the prosthetic (CN-3) can actually use.
CN-3 needs the operands read from the in-flight residual *at the decision
point*, before the model commits to a continuation — a single tap, not a
read that requires already knowing where the operands sit in text that may
not even be fully known yet in a fully general setting, and not a pool over
positions decided after the fact. **`tap` is the only feature representation
this experiment tested that CN-3 could actually deploy**, and `tap`'s
narrative number is 0.725 in-distribution / 0.030 held-out-family — the
harder, real number. `operand_positions`/`mean_pooled`'s strong results are
a genuine scientific finding (the information exists, cleanly, in the
residual stream) but not an architectural one (it is not available where
CN-3 would need to read it).

### CN-0, read against the gate, after two waves — the gate is not met, and CN-3 scopes out

**Precision on the pre-registered wording first, since it matters:** the
*gate* (>=95% held-out-family exact-pair) has not been met anywhere, by any
operation, family, feature representation, or hyperparameter setting, across
either wave. The literal *kill trigger* as originally worded ("no family
exceeds 80% anywhere in the band") did **not** strictly fire either — sub
and mul both clear 80% on several families (sub digit 0.850, sub mixed
0.925, mul mixed 0.895, mul word 0.865). So this is not "the pre-registered
kill line tripped, mechanically." It's a judgment call, made explicitly, on
the accumulated evidence — and that evidence now supports drawing it rather
than deferring it further:

1. **Three alternative explanations for the gap were tested and killed, not
   assumed.** Under-tuning: dead (the nested-validated sweep reproduces the
   untuned baseline bit-for-bit — a strong null). "Addition is
   representative": dead — it was the *hardest* case tested, not a typical
   one. "Narrative lacks the information": dead — `operand_positions` reads
   1.000, and the embedding-layer control reads exactly **0.000**, ruling
   out "the probe is just decoding the tokenizer" (the specific artifact the
   1.000 raised suspicion of).
2. **The finding the null test actually produced is a clean negative for
   CN-3's own premise.** The operand encoding forms by L0-L1 and stays flat
   through L26 — a fast-forming *numeral* encoding ("this token is the
   number 47"), not evidence of extended in-flight computation ("the model
   is currently computing 47 x 3"). If there's no extended computation,
   there is no in-flight operand *state* for a prosthetic to intercept in
   the first place — this is consistent with the standing observation
   (elsewhere in this session's work) that Gemma's small-number arithmetic
   behaves like lookup rather than computation. Two independent readings
   now agree on that shape.
3. **The one apparent fix doesn't transfer, and its failure mode is
   diagnostic.** Mean-pooling helps in-distribution (0.975-1.000) but
   *actively hurts* held-out-family generalization on every family tested
   (0.005-0.125, worse than tap's own 0.030-0.575) — the signature of
   fitting each surface form's own statistics, not recovering a
   surface-invariant representation.
4. **The strong features are architecturally unavailable to CN-3 regardless
   of their probe score.** `operand_positions`/`mean_pooled` require
   already knowing where the operands sit, or reading after the whole
   sequence is in, neither of which is available at the decision point
   CN-3 needs. `tap` is the only feature CN-3 could actually deploy, and
   `tap`'s held-out-family narrative number is 0.030 — the real number, not
   the encouraging one.
5. **This is the second independent time this readout machinery has hit
   surface-form brittleness on this model** (alongside the KnnStore result:
   canonical 10/10 -> paraphrase 0/10). Two independent measurements of the
   same failure mode is a property of what's there to read, not a
   coincidence to route around with a better probe.

**Verdict: CN-3 (the prosthetic, depth-2 landing) scopes out for
Gemma-class models.** Not "parked pending a better probe" — the operand
information required does not exist in a form both (a) surface-invariant
and (b) available at the decision point, and the mechanism (fast-forming,
flat, no extended computation) gives a principled reason not to expect that
to change with more probing effort on this model. CN-0 did exactly the job
its own pre-registration assigned it: settle the question cheaply (days, on
existing instrumentation) before a month is spent on CN-3's surgical build.

**What this does *not* touch:** depths 1 and 3 never depended on residual
readout. **CN-1** (cell tokens + fingerprint embeddings) and **CN-6**
(behavioural-spec emission) have the model *emit* the call in the token
stream — the boundary lives where the model already succeeds at surface
statistics, not where it needs an in-flight numeric intermediate. **CN-4**
(the routed organ, TinyModel) is untouched and arguably sharper for this
result: CN-0 asked "can the operands be *read* from what's already there,"
and the answer is no — CN-4 asks "can a model be *trained* to put something
readable there," a different question this result doesn't answer either
way. The flat-from-L0 finding explains *why* nothing is there today
(nothing in training ever required an intermediate operand representation)
without saying anything about whether training pressure could create one.

**Programme redirect: CN-1 is next**, not CN-3. The infusion thesis is
unharmed — only its most surgical (residual-stream) form is closed for this
model class; the token-stream form (a model that *learned to call* a cell)
carries the practical claim forward untouched.

**Remaining open items, now secondary to the CN-1 redirect:**
1. Root-cause `spin_pool`'s remaining concurrency bug (bug 3, carried over
   from wave 1, still open).
2. CN-1's H1 factory hasn't been scoped yet.
3. A proper multi-split average (not a single random 80/20 split) for the
   in-distribution numbers above would tighten them, but doesn't change the
   verdict — noted for anyone who later wants a cleaner citation, not as a
   blocker.

## CN-1 slice-0 — a toy pilot, not the pre-registered build

**TL;DR: the pilot's most important output is an architectural result, not an incident —
weight tying is a load-bearing precondition of the whole fingerprint-embedding hypothesis,
not an implementation detail — and it's stated here as a claim. On trained cells,
fingerprint-init cleanly beats random-init given tied weights (0.993-1.000 vs. a variable,
sometimes much worse mean), but that result travels with its caveat stated up front: toy
scale, trained cells, an effect closer to "a good initialization converges faster" than the
novel "embedding is the behaviour" claim. That harder claim — a fingerprint-placed embedding
gives an unseen cell a meaningful address — went through three iterations: the first two
each found a distinct reason the harness couldn't ask its own question; the third, run
against a bar pre-registered *before* the run (fingerprint-init > 0.5, random-init <= 0.25
on a properly-designed held-out combination), came back a clean, pre-registered **FAIL** —
both arms scored exactly 0.000. Per the fork stated in advance, that FAIL is itself the
answer: a from-scratch toy model this small has no compositional generalization to speak
of, so there is no substrate on which any embedding strategy could show an advantage. Gate
(ii) is not decided by this pilot — it moves to the real build, exactly as pre-registered,
with a hard stop on further toy iterations.**

CN-1's full spec is "the programme's first real training spend" across five repos (TinyModel
v11, the H1 factory, an ~800-cell vocabulary, ported constrained decoding). Research before
building found ~40% assembly / ~60% new construction, and a scope adjustment discovered
mid-research: TinyModel v11 is PyTorch (not MLX) and its tokenizer loads a pre-built,
immutable `.vocab.bin` with no `add_tokens` API — extending it means rebuilding the vocab
and recompiling a Rust/PyO3 extension. Per the user's explicit choice, this pass built a
**slice-0 pilot** instead: a small, self-contained MLX toy transformer + toy vocabulary
(trivial to add cell tokens to, since it's defined in the script, not loaded from a file),
mirroring CN-0's own "find the bugs in the apparatus before treating numbers as science"
discipline.

### What was built

- **`cell80/examples/dump_fingerprints.rs`** — a thin CLI wrapper around `Fingerprint::compute`
  (already public, no new fingerprinting logic): prints each named cell's fingerprint over
  `DEFAULT_PROBES` as JSON, called from Python via `subprocess`. The one new piece of Rust
  needed — `cell80-py` has no `Fingerprint` binding, so this is the pilot's stand-in for one,
  not the eventual shape.
- **`experiments/cell-native-architectures/cn1_pilot.py`** — rewritten across three
  iterations as each one's diagnosis reshaped the corpus design (iterations 1-2 used
  `chuk_math_gym`'s `ArithmeticGenerator` for the arithmetic cells, cross-checked against
  `cell80-py`'s `CellHost`; iteration 3's compositional grid, the final shape, generates all
  6 cells directly via `CellHost` — no independent domain generator needed once the corpus
  is a synthetic grid, since cell80's own execution *is* the label either way, and there's
  no separate spec for "is 12 >= 7" to diverge from). A small causal
  transformer (3 layers, dim 64) trained from scratch, comparing (b) random-init vs. (c)
  fingerprint-init cell-token embedding rows (a fixed linear projection of each cell's
  fingerprint vector, not learned) on the identical corpus/split.

### The architectural result: weight tying is a precondition, not a debugging footnote

The first full run tied nothing: the toy model's output projection was a separate learned
`nn.Linear`, not tied to the input embeddings. Result: both arms scored exactly 0.000 on
every cell, trained or held out. **Stated as a claim, because that's what it is: a
fingerprint-initialized cell embedding can only influence a prediction in a model whose
output projection shares weights with its input embeddings — the fingerprint has to be
*reachable from the output side* to steer anything.** Untied, the output head has no reason
to reflect embedding-space geometry at all, so a fingerprint-placed vector is invisible to
every downstream prediction regardless of how well it's placed. This is a real, non-obvious
precondition on the entire "the embedding is the behaviour" hypothesis — it constrains which
model architectures CN-1's approach can ever work in (tied-weight only) — not an incident to
footnote. It is also a small vindication of reading the signature correctly: two arms both
reading exactly 0.000 is the signature of a broken harness, not a false hypothesis, and it
was read that way before a single number was trusted, not written up as a null result. Fixed
by tying `logits = hidden @ embed.weight^T` (matching TinyModel v11's own `lm_head.weight =
embed.weight`) before any further result was treated as real.

### Receipts (post-fix, N=300/cell, 60 epochs, identical corpus/split for both arms)

**Iteration 1** — held out `is_ge` (shares `is_gt`'s prompt *shape*: "X > Y ?" vs "X >= Y ?")
and `argmax3` (shares no trained cell's structure at all — arity 3, disjoint vocabulary):

| cell | held? | (b) random-init | (c) fingerprint-init |
|---|---|---:|---:|
| add_sat, sub_sat, mul_sat, is_gt, discount_percent | trained | mean 0.993 | mean 0.997 |
| is_ge | held-out | 0.000 | 0.000 |
| argmax3 | held-out | 0.000 | 0.000 |

(An even earlier run, before the held-out set was reconsidered at all, held out
`discount_percent`/`argmax3` — trained-cell means (b) 0.640 / (c) 1.000, held-out both
0.000 — the first evidence that fingerprint-init's trained-cell advantage is real and not
small.)

**Iteration 2** — `is_ge`'s defining input token (`>=`) never appeared anywhere in training,
so the model had never processed it once; redesigned the held-out cells (`mul_sat`, `is_ge`)
to use templates that **recombine tokens already trained elsewhere** ("discount" via
`discount_percent`, "->" via `add_sat`/`sub_sat`, "?" via `is_gt") in a sequence never seen
together, and moved `argmax3` to trained (so "max" gets real gradient signal too, rather
than being a second untestable held-out case):

| cell | held? | (b) random-init | (c) fingerprint-init |
|---|---|---:|---:|
| add_sat, sub_sat, argmax3, is_gt, discount_percent | trained | mean 0.997 | mean 1.000 |
| mul_sat ("`a discount b ->`") | held-out | 0.000 | 0.000 |
| is_ge ("`a discount b ?`") | held-out | 0.000 | 0.000 |

**Still 0.000 for both arms, even with every individual token pre-trained.** This is a
different, deeper finding than iteration 1's: it's not that an input token was unprocessed —
it's that the specific **combination** ("discount" immediately followed by "->", never seen
together) is itself effectively novel to the model. A well-placed target embedding doesn't
help if the model never produces a hidden state resembling that novel sequence in the first
place — the bottleneck moved from the output side (which iteration 1 diagnosed) to the input
side (which iteration 2 diagnosed), and neither is what CN-1's own gate (ii) is actually
about.

**Iteration 3, pre-registered, the last one.** Both prior failures are "the model can't reach
a hidden state where the embedding could matter" wearing different clothes — neither is a
fingerprint result, both are capacity/compositional-generalization results. **Bar stated
before this run, not after:** PASS if fingerprint-init's accuracy on a held-out combination
exceeds 0.5 while random-init's stays <=0.25 (near the ~1/6 chance level for 6 candidate
cells); FAIL if both land in the same range. Design: a genuine 3x2 compositional grid —
`CATEGORY in {cat1,cat2,cat3} x VARIANT in {var1,var2}`, each combination mapping to one of
6 pilot cells (`add_sat`, `sub_sat`, `is_gt`, `is_ge`, `discount_percent`, `mul_sat`),
template `"{a} {cat} {b} {var} ->"` uniform across the whole grid. Trained on 5 of 6
combinations with 300 examples each — every category and every variant token gets heavy
exposure across *multiple* partners, a genuine basis for learning that category and variant
compose independently to select a cell, not memorizing 5 point facts. Held out exactly 1
combination (`cat3+var2` = `mul_sat`) entirely.

| combo | held? | (b) random-init | (c) fingerprint-init |
|---|---|---:|---:|
| cat1+var1 (add_sat), cat1+var2 (sub_sat), cat2+var1 (is_gt), cat2+var2 (is_ge), cat3+var1 (discount_percent) | trained | mean 0.993 | mean 1.000 |
| cat3+var2 (mul_sat) | held-out | **0.000** | **0.000** |

**Pre-registered verdict: FAIL.** Trained combinations hit 0.993-1.000 for both arms — the
model clearly learned the compositional *structure* well enough to place every trained
combination correctly — but the held-out combination scores exactly 0.000 regardless of
embedding strategy. Per the fork agreed before running this: a FAIL here means toy scale
cannot test gate (ii) at all — there is no compositional generalization capacity in a model
this small, trained this briefly, for any embedding placement to modulate — not "try a
fourth corpus." Stopping here, as pre-registered.

**A precision that must not blur, for any future reader: this FAIL is not evidence against
the fingerprint hypothesis, and must never be cited as such.** Both arms reading *exactly*
0.000 is the signature of a floor, not a comparison — there is no difference between the
arms to interpret, in either direction, because neither arm reached a hidden state the
embedding could have influenced at all. "Gate (ii) FAIL" here means "this pilot found gate
(ii) untestable at this scale," full stop — not "fingerprint-init failed to beat random-init
on held-out cells." The verdict is recorded exactly as pre-registered (a FAIL against the
stated bar, honestly written down rather than reinterpreted after the fact), but the
*mechanism* — zero compositional capacity, not a losing comparison — is what any citation of
this result must carry forward. A model with genuine capacity may still show gate (ii)'s
claim clean; this pilot has not spoken to that either way.

### What this shows

- **On trained cells, fingerprint-init is a clean, consistent win at equal training budget —
  held loosely, for a stated reason.** Random-init's per-cell accuracy is noticeably more
  variable and sometimes much worse at this same budget (0.640 mean in the earliest run,
  individual cells as low as 0.233-0.333); fingerprint-init reaches 0.993-1.000 on every
  trained cell across both iterations. At toy scale, on cells the model *is* trained on, this
  is close to what any structured initialization would give over an unstructured one — a
  convergence-speed effect as much as a semantic one. It supports "fingerprints are a useful
  init," which is real but is not CN-1's novel claim.
- **Weight tying is load-bearing, confirmed by exactly the failure signature that should
  raise suspicion.** Both arms silently reading 0.000 everywhere is diagnostic in itself —
  it says "the harness is broken," not "the hypothesis is false" — and treating it that way
  (rather than writing up a null) is what let the real result surface. Any future CN-1 work,
  toy or real, needs to verify tying explicitly, and — per the design's own logic — the same
  reachability requirement extends to CN-4 (the routed organ): if a fingerprint-derived
  representation needs to be reachable from where the model reads results, the organ's
  result-projection needs to land somewhere the model can actually read, and that's worth
  checking before CN-4's design hardens, not after.

### What this does *not* show — and why, precisely, across three attempts

- **The held-out/novelty question — CN-1's actual gate (ii), the part that matters — is
  decided as "not testable at toy scale," which is a real, pre-registered answer, not a
  loose end.** Three independently-diagnosed reasons, each sharper than the last: (1) a
  held-out cell's own defining input token can be entirely absent from training, giving the
  model no processed representation of that token at all (iteration 1); (2) even when every
  individual token is familiar, the specific *combination* can still be effectively novel
  (iteration 2); (3) even a properly-designed compositional grid — heavy, multi-partner
  exposure to every individual category and variant token, a genuine basis for learning
  composition as a rule — still could not get either arm above 0.000 on the one held-out
  combination (iteration 3, against a bar pre-registered before the run). All three are the
  harness/model failing to reach a decision point the embedding could influence, not the
  fingerprint hypothesis failing an available test — but iteration 3's pre-registered FAIL is
  the one that closes the loop: it rules out "just need a better corpus" as the explanation
  and points at model capacity/scale directly, which no amount of further toy-corpus
  redesign can fix.
- **A concrete, real requirement for the real H1 factory, regardless of how gate (ii)
  eventually lands.** Iteration 2 found "share vocabulary" isn't enough — the corpus needs
  genuine compositional coverage (multiple prefix/suffix combinations recombining flexibly)
  for a held-out combination to be a fair test at all. Iteration 3 confirms that requirement
  is necessary but shows it may not be sufficient at small model scale — the real H1 factory
  and a real (non-toy) model are where this actually gets decided.
- **This is 2 arms of 3, next-token argmax of 3, from-scratch of the real thing.** Arm (a)
  (the prompted `cell_solve` baseline) isn't meaningfully testable with a from-scratch toy
  model with no prompting ability. Evaluation is next-token accuracy at the cell-call
  position, not full constrained generation (porting LARQL's `generate_constrained`/
  `OpNameMask` pattern into MLX is real work, not attempted here). The toy model is trained
  from scratch, not TinyModel v11 — the real tokenizer/vocab-extension problem is untouched.
- **N=6-7 pilot cells, one seed, one architecture size, one training budget** — not a
  systematic sweep, and per the pre-registered stop, not extended further at toy scale.

### Reproduce it

```
cargo build --release -p cell80 --example dump_fingerprints
python3 experiments/cell-native-architectures/cn1_pilot.py   # ~10-15s on M3, no GPU training wait
```

### What would raise confidence further — at the real build, not another toy iteration

Per the pre-registered fork: this pilot's job is done. It validated the harness (weight
tying), found and named a real corpus requirement (compositional coverage), and used that
requirement to run gate (ii)'s first real test — which came back a clean FAIL, closing the
toy-scale question rather than leaving it open. What's next belongs to the real CN-1 build,
not a fourth corpus redesign:

- **Test gate (ii) with a real model at the scale CN-1 was always specified for**
  (TinyModel v11 or comparable, a real H1 factory corpus with genuine compositional
  coverage, a real training spend) — a model with actual capacity may show the
  compositional generalization this toy one didn't, and that's the honest next test, not a
  repeat of this one at larger toy scale.
- Scale N and epochs on the *trained-cell* question specifically (still open, and cheap to
  answer even at toy scale): does fingerprint-init only give a *training-speed* advantage,
  or a ceiling the random arm never reaches even given much more budget?
- Add arm (a) via a simple prompted baseline once a real model is in play (even a toy
  from-scratch model can be given a fixed in-context example set to "prompt" from, as a
  rough proxy, if a toy-scale check is wanted first).
- Keep the weight-tying and compositional-coverage findings in view when scoping the H1
  factory and CN-4's design — both are real constraints this pilot found for the cost of a
  toy, not open questions to re-derive later.

## CN-2 slice-0 — verified decoding, real result obtained

**Script:** `cell-native-architectures/cn2_verified_decoding.py`. Sends a
15-problem hand-authored GSM8K-style battery to a running LARQL server,
asks the model to show each arithmetic step as `A op B = C`, extracts every
such span by regex, and independently re-derives it via
`cell80_py.CellHost.solve()` (the same plan-IR path
`m3_gsm8k_smoketest.rs` uses). Verified against synthetic text end-to-end
(extraction + `add`/`sub`/`mul`/`div` verification all correct) before
touching a real server.

**Initial blocker, since fixed in the sibling `larql` repo** (see next
section for the full account): `/v1/chat/completions` hung indefinitely and
`/v1/completions` hung on anything past a trivial 3-token request. Root
causes found, fixed, and verified: a missing request timeout (ported an
existing fix from `/v1/infer`) and an unguarded raw-pointer read in the
default Q4_K hand-asm matvec kernel. A third, deeper concurrency bug in the
custom `spin_pool` thread pool remains — worked around via
`LARQL_SPIN_POOL=0` (routes through `rayon` instead; the file's own docs
call the two paths numerically identical), which the actual run below used.

**Result** (full 15-problem run, `LARQL_SPIN_POOL=0`, zero crashes):

```json
{
  "n_problems": 15, "n_spans": 8,
  "n_match": 8, "n_mismatch": 0, "n_escalated": 0,
  "agreement_rate": 1.0, "wrong_number_rate": 0.0,
  "final_answer_accuracy": 1.0
}
```

Every arithmetic span the model wrote in `A op B = C` form (8 of 15
problems produced at least one — the rest solved without showing intermediate
steps, still landing on a correct final answer) matched cell80's exact
computation. 15/15 final answers were correct. N is small (a slice-0 pilot,
not the full pre-registered GSM8K battery) — read this as "the measurement
pipeline works and the first real numbers are clean," not as a settled
wrong-number-rate baseline. The real CN-2 signal (does injection/resampling
move the needle) needs a larger battery and, per the gate's own design, some
genuinely wrong model arithmetic to correct — this pilot's model happened to
get everything right unaided.

## CN-2 rerun — 60-problem battery, and two real harness bugs found along the way

Extended `BATTERY` from 15 hand-authored problems to 60 (15 + 45
programmatically generated, larger numbers and 3–4-step chains, ground
truth computed in Python rather than by hand — `_gen_battery()` in the
script). Goal: give the model, and CN-2's measurement, something to
actually get wrong — the slice-0 pilot's 15 problems were easy enough that
the model went 15/15.

**That surfaced two real bugs in the harness itself, both worth recording
since they'd silently corrupt any future rerun that didn't catch them:**

1. **Regex span extraction matched chain fragments, not real claims.**
   `SPAN_RE` finds any `A op B = C` substring, so a model line like
   `"437 + 127 + 207 = 771"` (a genuine 3-operand sum) partially matches as
   the 2-operand substring `"127 + 207 = 771"` — real arithmetic (127+207
   = 334) that was never actually claimed to equal 771, so verifying it
   produced a **false-positive mismatch**. Same failure mode, worse, on a
   self-verification decomposition (`"6 * 578 = 6 * (500+70+8) = ... =
   3468"`, correct, flagged wrong) and on a degenerate repetition loop
   (`"359 + 144 = 499 + 1 = 500 + 1 = 500 + 1 = ..."` repeating to the
   token limit). Fixed by rejecting any match with an arithmetic operator
   or `=` immediately adjacent (prefix *or* suffix, skipping whitespace) —
   a genuine standalone equation has neither; a chain fragment has one or
   both. A label prefix like `"Total shirts sold = 45 + 38 = 83"` still
   matches correctly (the character before `45` is `=`, not an operator).
2. **The `SYSTEM` prompt's "A op B = C" phrasing was read as literal text,
   not a placeholder.** On 55/60 completions (including some of the exact
   same problems that worked cleanly in the slice-0 pilot) the model wrote
   lines like `"12 op 7 = 19"` — literally copying the word "op" instead
   of substituting `+`. Fixed by rewording the instruction and adding a
   concrete example (`"write '12 + 7 = 19', not '12 op 7 = 19'"`). Span
   coverage went from 9 verifiable spans (60 problems) to 127 after the
   fix — more than a 10× improvement in what the measurement could
   actually see.

**Final result** (60 problems, `LARQL_SPIN_POOL=0`, fixed harness + fixed
prompt, offline-reprocessed against the saved deterministic completions
once the extraction fix landed so the model didn't need to be re-queried
twice):

```json
{
  "n_problems": 60, "n_spans": 127,
  "n_match": 122, "n_mismatch": 2, "n_escalated": 3,
  "agreement_rate": 0.961, "wrong_number_rate": 0.016,
  "final_answer_accuracy": 0.883
}
```

- **2 genuine caught arithmetic errors** (real signal, not extraction
  artifacts): `68 * 31 = 2088` (correct: 2108) and `1569 + 299 = 1888`
  (correct: 1868).
- **3 escalations, and they're not model errors either** — all three are
  `636 - 710 = -74`-shaped (subtraction producing a negative
  intermediate). The model's arithmetic is actually correct on all three;
  cell80's plan IR is unsigned (`u32`-based) and has no representation for
  a negative intermediate, so it escalates (`needs_wider_math`) rather
  than silently producing a wrong answer. Real, honest coverage gap in the
  verifier, not a wrong-number-rate data point either way — worth fixing
  before CN-2 scales further (a battery with more subtraction-into-negative
  steps would just keep escalating instead of verifying).
- **Final-answer accuracy improved from 80% → 88.3%** between the
  "op"-placeholder run and the fixed-prompt run on the *identical* 60
  problems — forcing genuinely explicit intermediate arithmetic (rather
  than a broken placeholder the model partially ignored) measurably helped
  the model get more final answers right, a secondary but real finding
  about prompting for verified decoding.

**Read against the gate:** `wrong_number_rate = 0.016` is now a real,
trustworthy first baseline number (previously: 0.0 on too-easy problems,
then an artifact-inflated 0.1 before the extraction fix). It's the number
CN-2's eventual injection/resampling build should be compared against —
this slice still doesn't do injection, just measurement.

## LARQL fixes — three real bugs found in the sibling repo, two fixed

Reproducing CN-2's server hang required going into `~/chris-source/larql`
(a separate, actively-developed sibling repo, not part of `cell80`). Full
account, since these are real defects future sessions (in either repo)
should know about:

**Bug 1 — missing request timeout, FIXED.** `/v1/completions` and
`/v1/chat/completions` (`larql-server/src/routes/openai/{completions,chat}.rs`)
did a bare `spawn_blocking(...).await` with no deadline. A slow/stuck
generation call holds `LoadedModel.weights`'s write guard for as long as the
spawned thread runs; with no timeout, every subsequent request queues on
that guard forever — one slow request wedges the whole server. `/v1/infer`
already had this exact fix (`run_infer_with_timeout`, commit `660e6afb`,
"BUG-infer-deadlock §5.6") — it was never ported to the OpenAI-compat
routes. Ported the same `tokio::time::timeout(state.infer_timeout, handle)`
pattern to both. Verified: rebuilt, reinstalled, confirmed clean 200s where
the old binary hung indefinitely.

**Bug 2 — unguarded raw-pointer read in the default Q4_K asm kernel, FIXED.**
The hand-written `asm!` kernel (`q4k_q8k_matvec_asm_v3` and 8 sibling
functions in `larql-compute/src/cpu/ops/q4k_q8k_dot.rs`) is the **default**
matvec path (`LARQL_Q4K_ASM=0` opts out, not in). Its only guard against a
caller passing an activation buffer (`q8k_x.qs`) shorter than the `cols` it's
about to read was `debug_assert_eq!(q8k_x.qs.len(), cols)` — compiled to
nothing in this workspace's release profile (`[profile.release]` never sets
`debug-assertions`). The asm kernel takes `q8k_x.qs.as_ptr()` as a bare
pointer with no length of its own and no per-iteration bound in the asm
block itself — a real unguarded OOB read in production, not a debug-only
guard. Added a real runtime check (`q8k_shape_ok`, zero-fills and returns
early on mismatch, matching the file's own existing convention for the
`w.len() < rows*row_bytes` check right next to it) across all 9 call sites
carrying this pattern (Q4_K scalar/neon/neon_2row/asm/asm_v2/asm_v3, the
fused gate-up neon/asm pair, plus the Q6_K family for consistency, though
Q6_K's own `qs` access turned out to already be bounds-checked). Verified:
full `larql-compute`/`larql-inference`/`larql-server` test suite green
(1240+744+other passed, the two apparent `larql-inference` failures were
pre-existing test-parallelism flakiness on the shared `spin_pool::global()`
singleton, confirmed by re-running in isolation and by a clean full re-run),
all 28 `q4k_q8k_dot` tests including every scalar/neon/asm bit-exact parity
check still pass.

**Bug 3 — a real concurrency bug in `spin_pool`, NOT root-caused despite
extensive effort, hardened + worked around.** Even after both fixes above,
the server still crashed (SIGSEGV) on repeated requests — 3 separate crash
reports, fault addresses `0x2800` (10240, matches `features_per_layer`),
`0xa00` (2560, matches `hidden_size`), and `0x1`, all localizing to
`larql_compute::cpu::spin_pool`'s own dispatch closure
(`par_chunks_mut`/`run_chunks`), not the matvec kernels. Hardened one real,
provable defect found in that file — `chunk.min(total - start)` at
`spin_pool.rs:349`/`:384` is an **unchecked subtraction** that silently
wraps to a huge `usize` in a release build if `start` ever exceeds `total`,
feeding a wild length into `from_raw_parts_mut`; changed to `saturating_sub`
with a zero-length early-return.

Went further to try to actually root-cause it: wrote two new stress tests
(`stress_realistic_decode_shape_no_corruption`,
`stress_concurrent_realistic_decode_shape_no_corruption`) that exercise the
exact public `par_chunks_mut` entry point at the real gemma-3-4b-it
dimensions — sequential at production scale (108K+ dispatches), then
genuinely concurrent (6 threads, varying dispatch shapes, ~98K dispatches),
then again with artificial per-element busy-work to match the real asm
kernel's timing (ruling out a spin/yield/park timing dependency). All
clean, zero corruption, across every variant. A companion audit (separate
agent) traced the full mmap ownership chain from vindex file on disk to
the `&[u8]` slices the kernels read and found no unmap/reload/UAF hazard —
the vindex loads once at startup before the listener binds, and the
timeout-drop pattern from bug 1 doesn't dangle any reference (the abandoned
thread owns real `Arc` clones). Also audited every other `par_chunks_mut`
call site in the codebase for the same unguarded-pointer shape as bug 2 —
found none. **Conclusion: the bug resists both static analysis and
synthetic reproduction; finding it needs a live debugger session on an
actual crash (`lldb` attach + repeated requests until it faults), which
this session didn't have set up.** Root cause is open.

**Empirically confirmed mitigation:** `LARQL_SPIN_POOL=0` (documented in the
file's own header comment as routing through `rayon` instead, "either way
the arithmetic is identical — only *which threads run which chunks*
differs") ran the full 15-request CN-2 battery with zero crashes.

**State of the sibling repo: committed and pushed** (explicit request,
2026-07-12), three scoped commits on `origin/main`:
- `600dcc66` — bug 1 (timeout)
- `addbc267` — bug 2 (unguarded asm pointer)
- `7e5b84b8` — bug 3 hardening + the two new stress tests, commit message
  is explicit that root cause remains open

## CN-2 follow-up — the plan-IR signed lane, and 3 escalations become matches

The unsigned-plan-IR gap above (roadmap item 9's fifth finding) is now
**fixed** (2026-07-12, same day): `cell80/src/plan.rs` gained an `i32` repr —
the signed lane the escalations were asking for.

**Design, since backend zero has no native signed-32:** i32 values ride the
existing u32 state fields as **two's-complement bits** (the dialect doc's own
observation — signed add/sub/mul are bit-identical to u32 patterns — made
load-bearing). The renderer emits its own sign discipline instead of leaning
on a signed type: add/sub are the wrapping u32 ops plus the textbook
sign-rule overflow check (same-signs-in/different-sign-out for add, its
mirror for sub → `needs_wider_math`, exactly as the unsigned lane escalates);
mul/div convert to magnitudes branch-free (`(x ^ mask) - mask`), run the
existing unsigned checks, and reapply the result sign — division truncates
toward zero, rustc `i32` semantics. The range is symmetric by policy:
`i32::MIN` has no negation, so parse refuses it and any op that would produce
it escalates — which is what makes the magnitude trick unconditionally safe.
`nonneg` becomes a real check on i32 (it renders as nothing for u32), and
`exact_div` becomes a magnitude question. Mixed `int`/`i32` ops are a render
kill like every other repr pair — the extractor opts into the signed lane
explicitly. 5 new tests in `cell80/tests/plan.rs` (the literal `636 - 710 =
-74` case, chains through negative intermediates, the kill classes, canonical
rendering, the counterfactual battery on a negative answer).

**CN-2 re-verified against the saved 60-problem completions** (temperature-0
completions are deterministic, so `--reprocess` — added to the harness — is
exactly a rerun minus the model; the harness now sends `repr: "i32"`
quantities and decodes two's-complement answers):

```json
{
  "n_problems": 60, "n_spans": 127,
  "n_match": 125, "n_mismatch": 2, "n_escalated": 0,
  "agreement_rate": 0.984, "wrong_number_rate": 0.016,
  "final_answer_accuracy": 0.883
}
```

All three previously-escalated spans (`636 - 710 = -74`, `452 - 493 = -41`,
`480 - 734 = -254`) now **verify as matches** — the model was right each
time, and cell80 can finally say so. The only remaining non-matches are the
2 genuine caught model arithmetic errors (`68 * 31 = 2088`, `1569 + 299 =
1888`), which is the measurement working as designed: **agreement 0.961 →
0.984 with zero escalations**, and the wrong-number-rate baseline (0.016)
now stands on a battery with no verifier coverage holes. CN-2 is ready for
the G2 build proper (verified decoding *with* resampling on mismatch).

## CN-2 G2 — the correction loop: scoped wrong numbers go to zero, and the
## residue is exactly the pre-registered null

**Script:** `cell-native-architectures/cn2_g2_resample.py` (harness-level
slice of the G2 design — correction between requests, not yet inside
LARQL's decode loop). Baseline pass as before; then, for any span cell80
refutes, the completion is **truncated at the refuted claim, the verified
equation is asserted in its place** (the model's own notation, cell80's
number), and the model **continues from the corrected prefix** via
`/v1/completions` with the Gemma chat template rendered byte-identically to
the chat route (system turn + user turn + open model turn + partial). Greedy
decoding end to end, so the whole pipeline — baseline, refutation,
correction, continuation — is deterministic. Loop until no refuted span
remains (cap 4; never hit).

**Result** (60 problems, `--max-tokens 400` matching the committed
baseline, `larql-server --infer-timeout-secs 300` — note `larql serve`
does *not* forward that flag, run the server binary directly):

```json
{
  "before": { "n_spans": 127, "n_mismatch": 2,
              "wrong_number_rate": 0.016, "final_answer_accuracy": 0.883 },
  "after":  { "n_spans": 127, "n_mismatch": 0,
              "wrong_number_rate": 0.0,   "final_answer_accuracy": 0.883 },
  "n_corrections_total": 2,
  "t_baseline_total_s": 1049.2, "t_correction_total_s": 50.1
}
```

- **Wrong-number rate 0.016 → 0.000.** Both genuinely wrong claims
  (`68 * 31 = 2088` → 2108, `1569 + 299 = 1888` → 1868) were caught,
  corrected, and the corrected continuations verify clean — no correction
  cascades, no new wrong spans introduced. Cost: **~4.8% wall-clock
  overhead** (50s on 1049s), two extra requests over the whole battery.
- **The baseline reproduced the committed run to the token.** 56/60
  completions byte-identical two days and two server restarts apart; the 4
  divergent rows are pure prefix-extensions (same greedy stream, the older
  run just stopped earlier), with spans and finals unchanged. An earlier
  same-day run at `--max-tokens 120` produced 14 prefix-truncations of the
  same streams and *zero* token-level divergence — worth stating as a
  measured property: LARQL greedy decoding is reproducible across restarts,
  which is what makes "resample deterministically from a corrected prefix"
  a meaningful operation at all.
- **Neither corrected problem's final answer flipped — and the dissection
  is the finding.** Problem 21: after `68 * 31` is fixed, the model's
  *other* error (`359 + 144 = 499`, should be 503) sits at the head of a
  degenerate repetition chain (`499 + 1 = 500 + 1 = 500 + 1 = …` to the
  token limit) that the extractor rightly refuses to read as standalone
  claims — an **unscoped** wrong number, invisible to the span grammar by
  construction. Problem 27: after correction, *every arithmetic step
  verifies* and the final is still wrong because the model solved a
  different problem than the question asks — a **wrong plan**, which
  arithmetic verification cannot and should not touch. One artifact worth
  recording from the 120-token run: truncation can *forge* a
  standalone-looking claim out of a chain fragment (`… 500 + 1 = 50` cut
  mid-number matched as a claim and got "corrected"); at the correct token
  budget the artifact disappears, but an in-decoder G2 should treat an
  unterminated trailing equation as unverifiable, not as a claim.

**Read against the gate.** The G2 gate asks for a significant wrong-number
reduction at negligible overhead: **the scoped wrong-number rate went to
zero at ~4.8% wall-clock** (and the spec's per-token latency budget is not
even touched — correction fires on 2 of 60 problems). The pre-registered
kill-side observation also landed on schedule: the residual final-answer
errors are 100% non-arithmetic ("models' wrong numbers are mostly
unscoped / mostly plans") — on *this* battery the model's scoped arithmetic
was already 98.4% right, so correction's headline value is the **guarantee**
(every arithmetic claim in the output is now exactly right, signed by an
executed cell) rather than an accuracy lift. The natural next steps are the
in-decoder version (span grammar at decode time, the CN-1 error-correction
layer) and a battery seeded with harder arithmetic where the scoped
wrong-number rate is high enough for accuracy to move.

## Wave 1 status: both experiments have real, well-powered first results

CN-0's 5× rerun and CN-2's 60-problem rerun (above) are both done. Neither
was a one-shot — CN-0 needed the sample-size/λ/layer-range tuning from the
slice-0 pilot's own recommendations, and CN-2 needed two real harness bugs
(regex chain-matching, the "op" placeholder prompt) fixed mid-flight before
its numbers were trustworthy. Worth internalizing for whoever runs CN-1/
CN-3 next: a slice-0 pilot's job is exactly this — finding the bugs in the
measurement apparatus itself before treating its numbers as science.

## CN-1 real build — infrastructure map (step 2) and the axis-A held-out draw (step 3)

The real build is pre-registered (`cell-native-architectures-cn1-preregistration.md`). This
section records the two things done before any training spend: the cross-repo infrastructure
map that turns the pre-registration's "in-scope engineering" into a concrete assembly list,
and the axis-A held-out draw — committed, per the pre-registration's order of operations,
**before** any corpus exists.

### Infrastructure map — three repos, what exists vs. what must be written

- **`tiny-model/model` (TinyModel v11).** PyTorch, not MLX — the pre-registration's "MLX"
  tag was wrong and is corrected there. 115M params, dim 512, 20 layers, vocab 71261, and
  **weight tying is native** (`model.py:136`, `self.lm_head.weight = self.embed.weight`;
  `tie_embeddings: true` in config) — so the pilot's hard-won precondition holds for free.
  Trained on ~24M TinyStories tokens on an M3 via MPS, so from-scratch/continue training on
  the M3 is realistic per the repo's own design. **Must be written:** an embedding-resize
  utility (none exists; `load_state_dict` is `strict=True`), a checkpoint-resume path (the
  trainer only trains from random init), and an autoregressive generate loop (there is no
  decode loop at all — `forward()` returns raw logits and stops).
- **`tiny-model/tokenizer` (v11.vocab.bin).** The format's reader *and* writer are both
  public in `v11-core` (`Vocab::load` / `Vocab::save`); IDs are `u32` with **no runtime
  ceiling** (the 72000 cap is a builder-truncation knob only). So the ~790 cell tokens +
  call delimiters go in by **append-only re-serialization** at the tail (ids from 71260 up),
  leaving every existing row — and every trained embedding — untouched. The PyO3 binding
  (`v11.Tokenizer`) derives `vocab_size` from `vocab.len()`, so a larger artifact loads with
  zero binding changes. Atomicity of the new tokens comes from longest-match (no per-piece
  `is_special` flag exists); add both the plain and `▁`-prefixed form of each name.
- **`larql` (constrained decoding).** `OpNameMask` (`crates/larql-inference/src/experts/mask.rs`)
  is the port target: not a token trie but a per-step logit mask that re-derives grammar
  state from the decoded text and admits only tokens whose surface continues a valid op-name
  prefix (the closing `"` is unmasked only once `so_far` is a complete name). The entire
  coupling to the sampler is one closure, `FnMut(&[u32] generated_ids, &mut Vec<f32> logits)`,
  applied after **dense** LM-head scoring and before sampling. Reimplementing that single
  seam in the TinyModel generate loop carries the whole pattern. (There is no "span grammar /
  G2" subsystem in larql — that machinery is cell80-side, `cn2_g2_resample.py` lineage,
  exactly as the pre-registration assumes.)

### Library dump — `dump_library` (all 790 cells, not 249)

`cell80/examples/dump_library.rs` walks the library via `discover_cell_files` and emits one
JSONL row per cell — `name`, `pack`, `family_hash` (the identity-grade SHA-256 over
canonical source; `source_hash` is non-cryptographic and is not used as identity), `arity`,
`ret`, and the `DEFAULT_PROBES` fingerprint — into `cn1_library.jsonl`. One artifact feeds
both the axis-A draw and `W_f` later. A naive `Cartridge::compile` auto-detects only
`run`/`main` entries and silently dropped **541 of 790** cells (state cells and non-`run`
entries); the fix was to compile through the canonical CLI/admission path, newly exposed as
`cell80::library_cartridge` (was `pub(crate)`), which parses each cell's `//!` metadata
including its declared entry. Result: **790/790 dumped, 0 skipped**, all names unique, all
790 `family_hash`es distinct (no behavioural-duplicate collisions), across 42 packs. Arity:
541 arity-0 state cells, 92 unary, 104 binary, 53 ternary — i.e. **249 value cells** are the
natural call targets for an arithmetic-shaped corpus.

### Axis-A held-out draw — `cn1_axis_a_draw.py` → `cn1_axis_a_heldout.json`

Deterministic (seed 80, cells sorted by name within pack, `random.Random(seed).sample`), so
the exact set reproduces from `cn1_library.jsonl` + the script; verified identical on rerun.
Stratified by pack: `round(0.10 * size)` per pack, clamped to `[0, size-1]` so **no pack is
wholly held out**; 7 small packs (size < 5) contribute 0 held-out and serve only as seen
siblings. **Result: 79/790 = 10.0% overall held out**, of which 24 are value cells
(24/249 = 9.6%): **7 unary, 12 binary, 5 ternary**, plus 55 state cells. Arity is recorded
per held-out cell because gate (ii) is only *testable* on cells the corpus would otherwise
invoke — the 24 value cells — while the draw still honors the frozen "10% of the vocabulary"
text over all 790 (a held-out state cell simply yields an empty eval bucket, which is honest,
not a bug). The full 79-cell list with `family_hash`es is committed in
`cn1_axis_a_heldout.json`; this is the pre-registered record, timestamped before corpus
generation. These cells never appear as a call target in either corpus source; their
fingerprints (already in `cn1_library.jsonl`) are what `W_f` must turn into a usable address.

### Tokenizer extension (step 2a) — 792 atomic tokens appended, atomicity verified

Done and verified, non-destructively. Two new reusable examples in the tiny-model tokenizer
workspace (`v11-core/examples/`): `append_user_tokens` (append-only re-serialize via the
public `Vocab::save`, self-checking that every added token encodes to one id) and
`check_call_encoding` (the contextual check). The base `v11.vocab.bin`
(sha256 `873f44de…905b`, 71260 pieces) is **untouched**; the extended artifact
`v11-cells.vocab.bin` is written into the experiment dir.

**Token design (recorded so the corpus, `W_f`, and the mask agree):** one atomic token per
cell, natural surface `<cell:NAME>` (angle-bracket + colon-namespaced so it is not word-like
and cannot arise from ordinary text; NAME unique across all 790), plus two delimiters
`<call>` / `</call>`. Corpus form is space-delimited — `... <call> <cell:NAME> <args> </call>` —
so each is its own `▁`-prefixed chunk. **The crucial detail the pilot's toy vocab hid:** the
pretokenizer prepends `▁` (U+2581) to every whitespace-delimited run, so the *stored piece*
must be `▁<cell:NAME>` while the *corpus text* is the natural `<cell:NAME>`; the tool owns
that `▁` and the self-check encodes the natural form. Because each cell is **one** token,
constrained decoding (step 2b) is a single-step mask over a fixed id set — no per-character
op-name FSM as in LARQL, whose op names span multiple subword tokens.

**Result:** 792 tokens appended (2 delimiters + 790 cells), vocab 71260 → 72052, **contiguous
ids 71260..72051** (`<call>`=71260, `</call>`=71261, cells 71262..72051), map 1:1 with the
library (no missing/extra). Standalone self-check: all 792 → one id. Contextual check (cell
token embedded in word-problem lines with digit operands and delimiters): every cell token
stays exactly one id, delimiters one id each. The authoritative `{surface → id}` map is
`cn1_cell_token_map.json` (committed); `v11-cells.vocab.bin` is a build product, regenerable
from the pinned base vocab + `cn1_cell_tokens.txt` via `append_user_tokens`, so it is not
committed. All axis-A held-out cells are in the vocabulary (held-out = never *called* in
training, not absent as a token — constrained decoding must be able to emit them, which is
gate (ii)'s whole point).

### Three-way-tied W_f + resize (step 2c) and constrained decoding (step 2b) — apparatus validated

Both done and validated on the **real** v11 checkpoint (CPU, seconds, no training), before any
corpus exists — the pilot's discipline, applied to the real build. `cn1_model.py` and
`cn1_decode.py`, each with a structural self-test that passes.

**`cn1_model.py` — the architectural core.** `resize_embedding` grows the pretrained tied
embed/lm_head 71261 → 72052 rows; the self-test confirms all 71261 trained rows are preserved
**byte-for-byte** and `lm_head.weight` re-ties to `embed.weight` (same storage). `CN1Model`
overrides the forward so the effective embedding matrix — used for *both* the input lookup and
the tied output head — has its cell-token rows (71262..72051) supplied by a shared MLP
`W_f: fingerprint → d_model` (arm c) or by the base free params (arm b, the ablation).
Fingerprints are encoded 40-d (20 scaled probe values + a 20-d "ran cleanly" mask, so a
returned-0 and a trap/halt don't collapse to the same point). The self-test proves the four
load-bearing properties: (1) trained rows preserved + tying after resize; (2) arm-c cell rows
`== W_f(FP)` for a seen *and* a held-out cell, on the same matrix for input and output;
(3) **the gate-(ii) mechanism** — a gradient step optimizing only a *seen* cell's row moved a
*held-out* cell's row (W_f grad-norm ~77, held-out row moved ~2.3), because held-out cells have
no free params and can only move through the shared projection the seen cells train; (4) arm-b
cell rows are free base params, no W_f. A forward pass yields logits `(1, 5, 72052)`.

**`cn1_decode.py` — constrained decoding.** Because each cell is one atomic token (step 2a),
the LARQL `OpNameMask` port collapses to a **single-step mask over a fixed id set**: the one
grammar transition is "the token after `<call>` must be a `<cell:*>` id", applied at the same
`FnMut(generated_ids, &mut logits)` seam (dense logits → mask → pick). Self-test: after
`<call>`, exactly the 790 cell ids stay finite and the argmax is always a cell id; **all 79
axis-A held-out cells are in the allowed set** (emittable — without this gate (ii) is
impossible, since the mask must measure *selection*, not vocabulary membership); and an
end-to-end constrained `generate` on the real arm-c model emits a valid cell token after
`<call>`. Step 2's smoke slice (harness runs end-to-end, gates deliberately not evaluated) is
thereby satisfied.

**Step 2 (2a tokenizer + 2b decoding + 2c model) and step 3 (axis-A) are complete and
validated.** What remains before a training number: step 4 (CN-2 harvest + corpus generation,
two factorized held-out axes), step 5 (eval batteries + G2-reachability classification), step 6
(train arms, evaluate against the pre-registered gates). Also still to write on the model side:
the training loop itself (freeze policy for the base, W_f + new rows trainable), a
checkpoint-resume path, and argument-encoding in the call grammar (operands after the cell
token) — the last is a step-4 corpus decision.

### H1 factory — corpus generator (step 4, source 1) built

No H1 factory existed (it was "spec'd" everywhere, never built); the building blocks did — the
`CellHost` exact oracle, the `cell_eval/examples_gen.py` verified-I/O + sibling-dedup engine,
`retrieval.jsonl`'s hand-authored queries, chuk-math-gym, and steps 2–3's grammar/axis-A. The
factory that assembles them into `(context, <call><cell:NAME>args</call>, verified result)`
rows is `cn1_corpus.py`.

**Design decision (the scientific crux of gate (ii)), recorded:** each call's context is a
small set of **behavioral I/O demonstrations** of the operation (`a b = r ; …`) plus a query.
A cell's fingerprint *is* its behavior on the probe battery, so "demonstrated behavior →
`W_f(fingerprint)` region" is the smoothest substrate for a held-out cell to inherit an
address from its trained siblings — an arbitrary linguistic name could not transfer, a
behavioral cue can. Every demonstration is the oracle's own output, so rows are verified by
construction (`halt == returned` required; escalating draws dropped, `trapped_ops` carried).

**Axes (never conflated):** axis A = the 24 held-out value cells never appear as a call target
(train side); axis B = composition is (surface *template* × *pack*), with a stratified set of
(template, pack) pairs held out so every template appears with other packs and every pack with
other templates — a held-out (template, pack) is attributable to composition, not an unseen
token. Three uniform templates (`eq` / `arrow` / `io`) that work for any cell, so the
factorization needs no per-cell natural language (the library mostly lacks summaries — 0/120
sampled value cells had one).

**First generation (`--per-cell 30`, seed 80, ~2s, deterministic):** all **four eval buckets
populated and disjoint** — seen×seen 6054 (the in-distribution held-out slice), seen-cell×
novel-comp 520, novel-cell×seen-comp 668, novel×novel 40 — plus 6066 train rows. Coverage
222/225 seen value cells; the 3 stragglers are the `units` pack (needs matching unit-code
operands, not random u16 — a documented v1 gap). `cn1_corpus.py` + `cn1_corpus_stats.json` are
committed; the corpus JSONLs are regenerable build products (not committed). Ratio/scale are
CLI knobs; the real run raises `--per-cell` and can thin the seen×seen bucket. Still to add:
source 2 (the CN-2 harvest, mix ratio reported then) and sibling-discrimination of the demos
(so a held-out cell's context uniquely identifies it — the `examples_gen.py` `co_match` engine
is the reusable primitive).

### Training harness (step 6 apparatus) + the smoke-slice finding that stops the spend

The training loop (`cn1_train.py`) is built and runs end-to-end on M3/MPS: freeze policy knob,
both arms, loss supervised at the cell-token position (predict the cell from the prefix ending
in `<call>` — full-sequence CE is swamped by irreducible loss on digit/format tokens the frozen
base can't improve). Two MPS/mechanics issues found and fixed along the way: `index_copy` is
unimplemented on MPS (rewrote the effective-matrix assembly as a `torch.cat` of contiguous
slices — cell tokens are a tail range), and device-safety in the generate loop.

**Then the smoke run surfaced a real, load-bearing design finding — exactly what the step-2
smoke slice is for — and it stops the training spend until resolved.** Trained on the
behavioral-demonstration corpus, both a fully-frozen base and a top-4-unfrozen base **collapse
to predicting a single cell regardless of context** (cell-accuracy ≈ 0.005, loss plateaus just
under `ln(790)`). `cn1_probe_separation.py` isolates the mechanism rigorously: the base's hidden
state at the `<call>` position has **cosine 0.982 within a cell (same operation, different
operands) vs. 0.985 between different cells — separation −0.003**. The base collapses every
arithmetic-demonstration context to essentially one point; different operations are no more
distinct than the same operation on different operands.

**What this means (a corpus/representation finding, not a training bug).** My corpus design
grounded each call in behavioral I/O demonstrations — principled for gate (ii) (the fingerprint
*is* behavior, so demonstrated-behavior → `W_f(fingerprint)` is the smoothest transfer path) —
but it made the base task **few-shot function identification**: infer which of 249 functions the
demos show, with no surface cue, since operands are re-randomized every example. A TinyStories
base neither represents arithmetic demonstrations discriminatively (probe above) nor learns to
in 200 smoke steps, so there is no context signal for *any* embedding strategy (W_f or free
rows) to condition on — gate (ii) is not even reachable, for the same structural reason the toy
pilot's was: no substrate the address could modulate. The apparatus is validated; the corpus
grounding and the model's representation capacity are mismatched.

**The design fork this forces (before the real spend), and the recommendation.** Behavioral
grounding alone is too hard; a pure name/descriptor cue transfers to held-out cells no better
than the pilot's iteration-1 (novel token, no address). The resolution has to give seen cells a
*learnable* context→cell signal while keeping a held-out→fingerprint transfer path — i.e. a
**compositional descriptor** (operation-attribute words drawn from a controlled vocabulary,
composed per cell) that (a) lets the model learn description→cell on seen cells and (b) lets a
held-out cell's description, built from attribute words seen with other cells, land in the
fingerprint region `W_f` maps that behavior to. That is the pilot's compositional-grid lesson at
real scale, and it needs either synthesized per-cell descriptors (the library lacks summaries —
0/120 sampled) or a training phase that first teaches the base to *read* demonstrations. This is
a genuine scientific choice about corpus grounding, surfaced here — deliberately — before a
GPU-hour is spent, which is the entire point of pre-registering a smoke slice.

**Resolution chosen and validated: compositional descriptors.** The corpus now grounds each call
in a compositional operation *description* built from the cell's own snake_case name-words +
pack, expanded through a controlled abbreviation vocabulary (`sat`→saturating, `mul`→multiply,
`i16`→signed, …) so words recur heavily across the library (`op multiply saturating kind safe
arith`). Seen cells get a learnable description→cell signal; a held-out cell's description reuses
words seen with other cells, and `W_f(fingerprint)` places it near its behavioral siblings —
both transfer paths available (`cn1_corpus.py --grounding descriptor`, the default; `behavioral`
kept for the ablation).

The frozen-base separation probe still reads ≈0 *with* descriptors (+0.004) — but that measures
the frozen, anisotropic representation (all hidden states sit at cosine ~0.98); it is pessimistic
because it can't see what a *trained* model extracts. The decisive test is learnability, and it
passes: **a controlled diagnostic (10 cells, ~90 examples each, top-6 unfrozen) climbs from
cell-accuracy 0.08 → ~0.40 and loss 5.47 → 3.12 in 400 steps**, learning distinctive cells
(`rotr16`) cleanly (constrained HITs); confusion is confined to behaviorally-and-lexically
near-identical siblings (`leading_zeros`/`leading_ones`/`trailing_ones`). That root-causes the
earlier collapse: it was **data density** — ~6 examples/cell over 225 cells against a 790-way
output — not the harness or the approach. The behavioral-only corpus could not do even this
(separation ≈0 *and* no discriminative token to attend to); descriptors supply the signal.

**First full-library run (arm fingerprint, one seed — NOT a gate verdict).** 790-cell descriptor
corpus (~108 train examples/seen cell), top-8 unfrozen, 5000 steps (~23 min M3). Loss 6.78 →
~4.2 (chance `ln 790 = 6.67`), training cell-acc ~0.12. Constrained top-1 per eval bucket:

| bucket | top-1 |
|---|---|
| seen-cell × seen-comp (in-distribution) | 0.220 |
| seen-cell × novel-comp | 0.165 |
| **novel-cell × seen-comp** (gate (ii) signal) | **0.000** |
| **novel-cell × novel-comp** (gate (ii) signal) | **0.000** |

Read carefully, and NOT as a gate verdict: the model **does** learn seen cells (0.22 is ~170×
the 1/790 chance) and **does** generalize to novel *compositions* of seen cells (0.165) — so
unlike the toy pilot it reached hidden states the embeddings influence. But held-out **cells**
score top-1 0.000. Three reasons this is not yet gate (ii)'s answer: (a) top-1 is blunt — a
held-out cell landing at rank 3 of 790 still scores 0, and rank was not measured (the harness now
reports top-5 + median rank, and saves the checkpoint — both gaps this run exposed); (b)
in-distribution accuracy is only 0.22, so the model is under-powered — asking it to place unseen
cells when it is 22% on seen ones is premature; (c) one arm, one seed, no random-init baseline
(gate (ii) is a *contrast*). A stronger, better-instrumented run (denser corpus ~180/cell,
top-12, 8000 steps, rank metrics) is in flight to disambiguate "no transfer" from "partial
transfer below top-1", followed by the random arm.

### Gate (ii) mechanism CONFIRMED — the fingerprint-vs-random contrast on held-out cells

The decisive A/B: both arms trained identically (dense corpus, top-12, 8000 steps) on the SAME
descriptor-grounded corpus — identical contexts, so the descriptor's contribution is controlled
for; the **only** difference is where a cell token's embedding row comes from (arm c: shared
`W_f(fingerprint)`; arm b: a free learned row). Full rank distribution of the true cell among the
790 masked candidates, on the **held-out** bucket (novel-cell × seen-comp, n=200; chance median
rank ≈ 395). **Faithful median rank** (training-time eval, live model):

| | held-out median rank | seen median rank (control) |
|---|---|---|
| **fingerprint** | **56** | 62 |
| **random** | **619** (worse than chance) | 45 |

**This is the first genuine gate-(ii) positive in the programme.** On cells the model never saw
called, fingerprint embeddings rank them near the top of the 790-cell library (median 56);
random embeddings rank them *worse than chance* (619 — an untrained free row is actively
suppressed as the trained rows rise). The control rules out a general arm difference: on *seen*
cells the arms are comparable (random 45, fingerprint 62). So the held-out gap is specifically the
fingerprint projection — the mechanism the pre-registration named. The toy pilot could never reach
this test (0.000/0.000 capacity floor).

**Provenance note (do not quote top-10% fractions here as headline figures).** A top-10%
(rank < 79) distribution was computed via `cn1_eval_ckpt.py`, but that path reloaded checkpoints
that did **not** save the trained `norm` (the bug found later — see the norm-fix entry), so its
absolute fractions are from a slightly-unfaithful model and are **parked**, not reported. The
median ranks above are faithful (live-model training-time eval). Faithful top-10% fractions from a
single code path arrive with the 3-seed runs (checkpoints now save the norm).

**What is and isn't established (no overclaim).** This confirms the *mechanism* — behaviour-
derived embeddings give unseen cells usable addresses — as a **ranking** signal. It does **not**
clear the pre-registered gate-(ii) *bar*, which is top-1 (`(c) ≥ 0.5`): held-out top-1/top-5 are
0.000 for both arms, because the whole model is under-powered (even *seen* cells are only ~0.065
top-1 / ~0.24 top-10%). So the honest verdict is **"mechanism confirmed, invocation not yet"**:
the fingerprint address is real and strong at the rank level, and converting it to top-1 needs a
stronger model (more capacity/steps/data), not a different mechanism. Also: one seed per arm; the
pre-registered gate needs 3.

### CORRECTION (2026-07-14): two headline numbers were first-N sampling artifacts

> The eval harness scored the **first 200 items** of `cn1_corpus_eval.jsonl`, which is **grouped by
> cell** — so it over-weighted whichever held-out cells appear first (and their ~12 repeats each).
> Two headline numbers were inflated by this and are **corrected**:
> - **Held-out median rank: ~21 → ~114 of 790** (robust random sample, n=150: median 114, mean 209).
> - **Confusion enrichment: 6.73× → ~2.7× median / ~4.2× mean** vs the all-790 null (confusion
>   agreement 0.14 median / 0.22 mean; null 0.053). Diagnosis (`cn1_null_diagnosis.py`): the
>   agreement function is identical across routes (diff 0.0000), so this is population/sampling, not
>   a broken metric; the two nulls (all-790 0.053, same-arity-value 0.147) are each correct for their
>   population, and the confusions span types (37% same-arity, 18% state) so all-790 is a reasonable
>   comparator.
>
> **What survives:** the arm *contrast* — fingerprint ≪ shuffled ≈ random on held-out — used the
> *same items for all three arms*, so the double dissociation and the "fingerprint does what random
> can't" conclusion **stand**. What was inflated is the *absolute* address quality (rank, enrichment
> magnitude). Net: mechanism real, **usable level weaker than previously stated** (median rank ~114,
> per-cell top-50 recall 0.25). Authoritative absolute numbers come from a **random-sampled** re-eval
> of the faithful checkpoints; the in-flight training evals are contrast-valid but absolute-inflated
> (first-N). Fourth consistency-check catch (random-vs-first-N), same shape as the previous three.
> The tables below retain their original (first-N) numbers as the reasoning trail; read absolute
> levels against this correction.

### The shuffled control + the double dissociation (the result that makes it strong)

The one control a skeptic reaches for: is the held-out signal *behaviour*, or just a well-
conditioned shared projection / name-similarity? Arm **(s) shuffled** — identical `W_f`, identical
geometry, but each cell assigned a *different* cell's fingerprint (seeded derangement) — answers
it. Run at a fixed config across all three arms (top-16, LR linear decay, 8000 steps, dense
corpus, seed 80). Median rank of the true cell among 790:

| arm | seen top-1 | seen rank | **held-out rank** (novel-cell × seen-comp) |
|---|---|---|---|
| fingerprint | 0.27 | 72 | **43** |
| shuffled | 0.475 | 2 | **566** (worse than chance) |
| random | 0.785 | 0 | **519** (worse than chance) |

Two things line up, and together they are decisive:

1. **Held-out transfer is behavioural.** Scrambling the behaviour↔cell correspondence collapses
   held-out ranking from 43 to 566 — from ~9× better than chance to *worse* than chance, alongside
   random's 519. So the address signal is specifically the fingerprint↔behaviour correspondence,
   not the projection layer and not name-similarity. The pre-registered "behavioural" outcome.
2. **A double dissociation kills the "better init" alternative.** On *seen* cells the ordering
   **inverts**: fingerprint is *worst* (top-1 0.27, rank 72), shuffled better (0.475, rank 2),
   random best (0.785, rank 0) — a clean monotonic pattern along the "freedom to memorize" axis
   (behavioural constraint < arbitrary projection < fully-free rows). The skeptic's default
   ("fingerprint just has a better-conditioned init / the shared projection aids optimization")
   predicts fingerprint ≥ shuffled *everywhere*; the seen-cell inversion is the opposite. What
   remains is the mechanism the hypothesis names: **behavioural geometry constrains similar cells
   to similar rows — costing rank-1 precision on seen cells, buying an address for unseen ones.**
   Generalization traded against memorization, along the axis the hypothesis predicts. The
   inversion was not designed; it is a prediction the hypothesis makes that nobody wrote down
   first — now **pre-registered before the 3-seed run** so replication is confirmatory.

The LR-decay/top-16 config also lifted seen top-1 4× (0.065 → 0.27) over the un-decayed top-12
run, while held-out top-1 stayed pinned at 0.000 — the exact pattern the base swap exists to
disambiguate (ceiling vs. no-prior). Flagged and **not yet reported as a result:** the
`novel_cell × novel_comp` bucket (n=48) shows a sign flip (shuffled 292, *better* than chance,
vs 566 worse on seen-comp) — under-powered, to be resolved by 3 seeds / larger n before it enters
the findings.

**Status of the CN-1 real build:** the **gate-(ii) mechanism is confirmed with a controlled,
double-dissociated, mechanistically-explained positive** — behaviour-as-address is real, not a
hypothesis. Owed before it is a *gate*: 3 seeds (the registered inversion prediction), and top-1
on a base with a relevant prior (the SmolLM2 swap — running). Also owed for the full programme:
gate (i) vs the prompted baseline, step-5 eval batteries + G2-reachability, gate (iii), and the
CN-2 harvest (source 2). The v11 result stands as written — "mechanism confirmed, invocation not
yet."

### SmolLM2 swap — preliminary read (PARKED: norm-less reload), and an MPS crash lesson

The first swap run trained the fingerprint arm to completion (step 8000, checkpoint saved) then
**crashed in the eval**: the full-vocab LM-head matmul on MPS (`hidden @ w.t()`, V=49944) hits a
hard MPSGraph shape-inference bug on the trained tensors (a fresh model dodges it; weights are
finite, not NaN). `set -e` + a monolithic queue meant that one crash destroyed the whole batch —
shuffled/random/seeds never ran. Fixes: HF eval moved to **CPU** (checkpoint saved first anyway;
~800 single-example forwards), and the runner hardened (no `set -e`, each run independent). Both
CN-1 infra bugs so far — the pilot's untied head and this one — were caught by *consistency
checks*, not failing tests; here it was reload-crashes-where-fresh-doesn't.

**Preliminary SmolLM2 fingerprint (CPU eval of the saved checkpoint — PARKED, because that
checkpoint predates the norm-save; norm moved v11 shuffled 313→566, so treat as directional):**
held-out (novel_cell × seen_comp, n=200): **top-1 0.000, top-5 0.180, median rank 21, 88% in
top-10%**; seen: top-1 0.360, median 9. Read directionally against v11 (held-out median 43,
top-5 0): the code/math prior **sharpens held-out ranking** (median 21 vs 43; 88% vs ~65% top-10%)
and **top-5 goes 0 → 0.18** — the rank signal is starting to climb toward the top — but **top-1
still hasn't converted**. Right at the prior-vs-capacity boundary: prior clearly helps, doesn't yet
fully crack top-1. The faithful number (norm saved, one path) + the shuffled/random control arms +
the within-base inversion check come from the resilient re-run now in flight. Nothing here is a
verdict; it is the parked preliminary.

### Top-k confusion analysis — the plateau, and a reframe that was later RETRACTED

> **RETRACTED (2026-07-14):** the "neighbourhood-resolution / top-1 is mechanism-forbidden"
> reading below was withdrawn. The confusions sit at ~0.44 agreement (loosely related, *not*
> siblings), and a held-out cell has median 0 genuine near-duplicates (≥0.8) — so nothing
> structurally forbids rank 1. The honest position: mechanism confirmed, usable level insufficient
> (per-cell top-50 recall 0.25), top-1 not excluded, cause undetermined (capacity / fingerprint
> resolution / corpus). See the pre-registration's RETRACTION amendment. The 6.7× enrichment and
> the double dissociation stand; the *reframe* does not. Text kept below as the reasoning trail.

The preliminary held-out plateau (median rank ~21, 88% top-10%, top-5 0.18, **top-1 0.000**) is a
plateau *shape*, not a near-miss: if it were capacity the whole distribution would shift and top-1
would lift with it; instead the mass is pressed against the top and the final pick fails. Tested
directly on the fingerprint checkpoint (`cn1_confusion_analysis.py`) — **what occupies the ranks
above the true cell?**

- **Confusions vs true cell: mean fingerprint agreement 0.436, vs 0.065 for random — 6.7× more
  behaviourally similar than chance.**
- **Same-family (pack) rate of confusions: 0.112 vs 0.027 base rate — 4.1×.**

The cells beating the true held-out cell are its **behavioural siblings** — the cells that do
almost the same thing. This is the *same* property that causes the seen-cell inversion: fingerprints
place behaviourally-similar cells near each other, so the arm that reaches rank 21 is exactly the
arm that fills ranks 1–20 with near-identical cells. **The mechanism's strength and its top-1
ceiling are one property.** (A minority of held-out cells — e.g. `swap_bytes` — rank far worse with
low-agreement confusions; those are genuine capacity misses, not neighbourhood resolution, and are
a named residual, not the dominant pattern.)

**What this reframes.** Behaviour-as-address resolves to a behavioural *neighbourhood*, not a
point — so the pre-registered top-1 bar (c ≥ 0.5) is likely the **wrong bar for this mechanism**,
and "more capacity" would not fix a structural ceiling. The architecture already answers it: the
model emits an address that lands the neighbourhood (rank ~21, 88% top-10%), and the **shipped
fused behavioural router (0.859) disambiguates within it by execution** — the F2 two-tier design,
which is the correct division of labour (model locates, runtime resolves) and degrades gracefully
as the library grows. The claim moves from "the model picks the right unseen cell" (hard, maybe
structurally capped) to **"the model locates an unseen cell's behavioural neighbourhood from a
description of what it needs, and execution resolves the rest"** — a better claim, because it is
the one the substrate is built to deliver. Registered as a superseding interpretation in the
pre-registration before the faithful re-run.

**Methodological note (third instance).** This was caught the same way as both CN-1 bugs: a
consistency check, not a failing test (reload-crashes-where-fresh-doesn't; before that
arm-vs-arm-signature and reload-vs-training-rank). Standing method for this lane: compute
load-bearing numbers by two routes; a disagreement is the finding.

**Companion rule (2026-07-17, after three same-day instances spanning three repos —
whitelist-read-as-census, trained-subset-read-as-id-space, spaCy-skeleton-assumed-equivalent-to-
frozen-normalizer; corpus-atlas gate A0 / equivalence audit):** no derived artifact enters a
claim without its construction attached. An artifact whose construction is unrecorded gets
silently promoted to ground truth by the next reader; all three instances were caught only by
re-deriving the artifact from its source and diffing. Receipts, generalized from matches to
instruments.

### Probe-richness sweep (model-free) — the confusions are genuinely distinct, not spuriously merged

Tests whether the 20-probe fingerprint is too coarse to separate the model's ~0.44 confusions, or
whether those cells are truly that similar (`cn1_probe_richness.py`, execution only, no model).
Mean agreement of each held-out value cell's coarse top-20 neighbourhood, as the probe battery
grows: **20→0.344, 100→0.259, 500→0.247, 2000→0.245**; fraction of that neighbourhood still ≥0.7
agreement: 0.119→0.085.

- **[CORRECTED]** The sweep's 0.344→0.245 drop was mostly **winner's curse**, not over-merge: the
  top-20 was selected *by* the noisy 20-probe estimate, so re-measuring less noisily must fall. The
  bias control (`cn1_probe_bias_control.py`, 3000 *random* same-arity pairs, chosen by nothing) is
  decisive: 20-probe agreement 0.1625 vs an independent rich battery 0.1293 — **mean over-merge
  +0.033, median +0.018.** So the battery genuinely runs high, but only by ~3 points; of the
  sweep's −0.099 drop, ~0.033 is real bias and ~0.066 was selection regression. The original
  "~28% over-merge" is **retracted** as ~3× inflated. A richer address would sharpen only
  *modestly* — so the probe-richness retrain is defensible but **low priority**; capacity/corpus
  likely dominate the per-cell recall gap.
- **But it plateaus at ~0.245 by 100 probes** — the confusion cells are **genuinely** ~0.245-
  similar (they differ on ~75% of inputs), not artifacts that dissolve under rich probing. So the
  confusions are **behaviourally distinct**, which *confirms the retraction*: nothing structural
  forbids rank 1 — the address can resolve these in principle.
- **Net:** fingerprint resolution is a *contributing, tractable* cause (richer probes should help
  modestly), not the whole story; capacity and/or corpus likely dominate the per-cell recall gap.
  Whether a richer fingerprint actually moves the model's per-cell recall is a retrain experiment
  (the model-free sweep only establishes the cells are separable in principle — they are).

### Faithful arms: the pre-registered inversion REPLICATES 3/3 seeds; the swap dissociates too

All nine runs completed (resilient runner + CPU-eval fix held; no failures). Numbers are first-N
eval as-run (contrast-valid across arms, absolute-inflated — authoritative random-sampled absolutes
pending), seen-top1 / held-out-median-rank:

| | seed 80 | seed 81 | seed 82 |
|---|---|---|---|
| fingerprint | 0.27 / 43 | 0.295 / 44 | 0.285 / 44 |
| shuffled | 0.475 / 566 | 0.43 / 432 | 0.49 / 345 |
| random | 0.785 / 519 | 0.745 / 447 | 0.76 / 331 |

**The pre-registered prediction holds in all three seeds:** (a) fingerprint underperforms shuffled
on **seen top-1** (0.27<0.475, 0.295<0.43, 0.285<0.49) *and* (b) outperforms it on **held-out rank**
(43<566, 44<432, 44<345) — and fingerprint beats random on held-out in every seed too. The
fingerprint held-out rank is strikingly stable (43/44/44). This is the confirmatory outcome: a
prediction the mechanism made *before* the data, which could have failed, held across seeds. The
double dissociation is not a one-seed artifact.

**The SmolLM2 base swap replicates the dissociation at a different base** (first-N eval): seen top-1
fp 0.36 < shuf 0.52 < **rand 1.0** (random memorizes seen cells *perfectly* — the extreme of the
freedom-to-memorize axis); held-out rank fp **20** ≪ shuf 272 < rand 404. So the mechanism is
**base-independent** — the inversion and the held-out advantage reproduce on a code/math-pretrained
Llama, not just the TinyStories base. Held-out top-1 is still 0 on SmolLM2 (the retired bar); seen
top-1 rose modestly (0.27→0.36) — the prior helps but does not crack top-1, consistent with the
K_exec reframe that top-1 was never the target.

**Caveat (unchanged):** absolute ranks here are first-N-inflated; the authoritative absolute levels
(to read against K_exec) come from the random-sampled checkpoint re-eval (`cn1_eval_ckpt.py`, now
shuffled; seeds 81/82 checkpoints have the saved norm and are faithful; seed 80's predate the
norm-fix). The *contrast* and the *inversion replication* are matched-item and stand regardless.

### Authoritative random-sampled re-eval (v11 seed 81, faithful) — and it clears K_exec at 790

`cn1_eval_ckpt.py` (now random-sampled, norm loaded) on the seed-81 faithful checkpoints. Held-out
(novel_cell × seen_comp, n=200), median rank / mean / % in top-10%:

| arm | held-out median | mean | top-10% | seen median (control) |
|---|---|---|---|---|
| fingerprint | **98** | 187 | **0.38** | 29 |
| shuffled | 498 | 481 | 0.045 | 25 |
| random | 539 | 520 | 0.0 | 2 |

- **The contrast is robust under authoritative sampling:** fingerprint held-out median 98 ≪ shuffled
  498 ≈ random 539; 38% of held-out cells in the top-10% vs 4.5% vs 0%. The seen control holds (all
  comparable, random best at 2 — the freedom-to-memorize extreme). The corrected absolute median is
  ~98 (first-N had inflated it to 44; consistent with the ~114 robust proxy).
- **Read against the execution-derived target:** median 98 < **K_exec ≈ 260 (CPU)**, and p75 = 227
  < 260 — so at 790 cells the fingerprint places **most** held-out cells inside the per-token
  executable budget. **The mechanism already clears the usable bar at today's scale.** What it does
  as the library grows — absolute rank flat (usable) vs fractional 12.4% (→ ~124k at 10⁶, unusable
  on CPU) — is the single remaining question, and it is exactly the next experiment (the library-
  scale curve). (Swap/SmolLM2 authoritative re-eval is pending an HF checkpoint-eval; the v11
  faithful number is the solid one.)

### Library-scale curve (hypothesis a) — UNDERPOWERED (not FAIL); threshold sits inside the CI

> **VERDICT CORRECTION (2026-07-14):** an earlier revision of this section called this a
> "pre-registered FAIL." That was wrong, and registering an unsupported FAIL is the same error as
> claiming an unsupported PASS (the sixth catch, same shape). The point estimate is α = 0.624, but
> **SE(α) = 0.088 and the 95% CI is [0.38, 0.87]** (6 points, 4 df, residual s = 0.149) — the
> threshold **α < 0.54 sits comfortably inside the interval**, so the experiment **cannot decide
> pass from fail. Verdict: UNDERPOWERED.** The non-monotonicity (rank at N=175 is 32, *below* N=114's
> 34) is the noise floor announcing itself. What *is* established: **α < 1 with confidence** (upper
> CI 0.87 < 1.0 → sublinear growth confirmed) and **lift over chance grows monotonically 1.7×→4.1×**
> — the mechanism gets relatively stronger as the library grows. Envelope across the CI: α=0.38 →
> usable to ~22M cells; α=0.62 → ~390k; α=0.87 → ~68k — so even the pessimistic end is ~10⁵ cells
> (130× the current library). This decides whether the pitch says "millions" or "hundreds of
> thousands," not whether the thing works.

The deciding experiment (`cn1_scale_curve.py`; W_f-only retrain on the frozen seed-81 transformer,
validated to reproduce the full-model rank: 96 vs ~98 at N=790). Held-out median rank vs library
size N (each holding the 24 axis-A cells in, retrained on the subset's seen cells):

| N | held-out rank | chance (N/2) | lift over chance |
|---|---|---|---|
| 114 | 34 | 57 | 1.7× |
| 175 | 32 | 88 | 2.7× |
| 270 | 45 | 135 | 3.0× |
| 415 | 76 | 208 | 2.7× |
| 640 | 87 | 320 | 3.7× |
| 790 | 96 | 395 | 4.1× |

**Log-log fit: rank(N) ≈ 98·(N/790)^α, α = 0.624 [95% CI 0.38–0.87]. Threshold α < 0.54 is INSIDE
the CI → the experiment cannot decide (UNDERPOWERED; see the verdict-correction note above).** The
point extrapolation is rank(10⁶) ≈ 8,462 vs GPU K_exec 4,718 (factor 1.8, well inside what a ±0.09
SE on α produces: α=0.54→4,718, α=0.70→~15,000, both in the interval).

**What is established (independent of the underpowered pass/fail): the geometry holds sublinearly:**
- **Lift over chance grows with library size (1.7× → 4.1×)** — the hoped-for shape: absolute rank
  grows sublinearly (α < 1) while chance grows linearly, so the mechanism gets *relatively* stronger
  as the library grows. It is genuinely doing behavioural work at every scale tested.
- **Usable envelope spans the CI: ~68k (α=0.87) to ~22M (α=0.38) cells, point estimate ~390k**
  (GPU @ 4.8%). So *even the pessimistic end of the interval is ~10⁵ cells — 130× the current
  library.* Whether it reaches 10⁶ standalone is the undecided question; that nothing is blocked for
  a long time either way is not.
- **Why underpowered:** 6 points over <1 decade (114→790) extrapolated 3+ orders of magnitude — the
  non-monotonic N=175 point is the noise floor, SE(α)=0.088, CI [0.38,0.87]. Plus a small-N
  training-amount confound (fewer seen cells trained at small N; α direction ambiguous).

**What it means for the programme.** Hypothesis (a) — behavioural geometry holding as the library
grows — is **neither confirmed nor refuted; the curve as built cannot decide it.** What stands: the
geometry holds **sublinearly** (α < 1 with confidence, CI nowhere near 1) and lift grows monotonically
(1.7×→4.1×) — the mechanism gets relatively stronger with scale. **The resolution is another decade
of N (~5–10k cells)** to halve the CI *and* reach the softmax ceiling — which requires **synthetic
library expansion** (fingerprint-perturbed clones, or composition/cost-discovery output), the **same
build hypothesis (b) needs. One build answers both**, and it is more informative than CN-6, whose
premise depends on how this lands. The honest headline: *behaviour-as-address is a confirmed
mechanism whose scaling exponent is measured but not yet pinned (α = 0.62 [0.38, 0.87]); the next
experiment is synthetic expansion to 10⁴ cells, which decides both the exponent and the token-vocab
ceiling.*

### Synthetic scale curve to 10⁴ — α tightens to [0.53, 0.82], two methods agree, sublinear robust

Extended the library 790 → 10⁴ with 9,210 **density-matched** synthetic cells (fingerprint clones,
30% of probes resampled from per-probe marginals; synthetic nearest-real agreement median 0.70 ≈ real
0.70–0.75, slightly denser on the mean → conservative). Held-out rank among N using the **fixed
seed-81 W_f** (`cn1_synth_scale.py`; N=790 reproduces the seed-81 rank, 86 vs ~96 — path validated):

| N | rank | chance | lift | N | rank | chance | lift |
|---|---|---|---|---|---|---|---|
| 790 | 86 | 395 | 4.6× | 5,000 | 245 | 2,500 | 10.2× |
| 1,500 | 113 | 750 | 6.6× | 8,000 | 386 | 4,000 | 10.4× |
| 3,000 | 162 | 1,500 | 9.3× | 10,000 | 462 | 5,000 | 10.8× |

**α = 0.673, SE 0.053, 95% CI [0.53, 0.82]** over the synthetic decade — **consistent with the
retrained real-curve α = 0.624 [0.38, 0.87]** (two different methods — retrain-per-N vs fixed-W_f
with structured distractors — agree, which is the reassuring cross-check). The extra decade cut SE
from 0.088 to 0.053.

Reads:
- **Sublinear is now firmly established** (upper CI 0.82 < 1); **lift over chance keeps growing
  (4.6× → 10.8×)** through 10⁴ — the mechanism gets relatively stronger at every scale tested.
- **The pass threshold α < 0.54 now sits at the *lower edge* of the CI (0.53)** — so the picture has
  shifted from "undecided" to "leaning past the 10⁶ GPU window": point estimate α ≈ 0.62–0.67,
  extrapolated rank(10⁶) ≈ 8.5k–10.5k vs GPU K_exec 4,718 (~2× over). Not a clean fail (0.53 barely
  passes), but no longer symmetric around the threshold.
- **Usable envelope (GPU @ 4.8%): ~300k cells** at α=0.673 (≈390k at 0.62) — both methods land at
  "hundreds of thousands," short of millions standalone.
- **No softmax cliff** in the fixed-W_f regime: rank grows as a smooth power law through the
  ~2,500-candidate mark, no discontinuity. **But this does NOT test hypothesis (b)** — the address is
  fixed, not a *learned* softmax over 10⁴ tokens; the learned-routing bottleneck still needs the
  retrain-with-10⁴-tokens build and remains owed.

**Net:** the two-method agreement + the tighter synthetic CI converge on **α ≈ 0.65, sublinear,
usable to ~10⁵–10⁶⁻ cells, reaching full 10⁶ standalone is borderline-unlikely** (threshold at the CI
edge). Behaviour-as-address scales to hundreds of thousands of cells with a fixed address; the full
million, and the *learned* routing question, route to the two-tier/CN-6 path — with a measured
exponent now, not a guess.

### CN-6 stage 1 — the runtime tier resolves held-out cells from examples (premise validated)

Before training a model to emit I/O examples (the expensive CN-6 build), we validated the tier the
whole thing rests on: does `CellHost.route` (the pure behavioural router) resolve a **held-out** cell
from a few I/O examples? `cn6_router_check.py`, 249 value cells loaded, random examples:

| cells | k=3 | k=6 |
|---|---|---|
| held-out (n=24) | P@1 0.625, P@5 0.833 | **P@1 0.750, P@5 0.833** |
| seen (n=40, control) | P@1 0.525, P@5 0.825 | P@1 0.725, P@5 0.950 |

**Premise holds, and for the right reason.** Held-out cells resolve **as well as seen cells** (P@1
0.75 vs 0.725) — because the router resolves by *execution*, not by any learned representation, so it
is training-independent by construction. That is exactly the property that makes the two-tier design
library-size-invariant: the runtime doesn't care whether a cell was in training. And these are
**random** examples — the discriminating `co_match`-selected examples plus text fusion are what lift
the fused router to 0.859, so **0.75 is a floor.** (Caveat: 249-cell library, not the full 790; P@1
would soften over more cells but the held-out/seen parity is the load-bearing fact.)

**Correction: 0.75 is a CEILING, not a floor.** The stage-1 examples came from an *oracle* (we ran
the true cell). A deployed model won't have the cell — that's the point of retrieval — so the real
question is whether a *model's imperfect* examples still resolve. Two prerequisites, done before any
training spend:

**(a) Noise sensitivity (`cn6_noise_check.py`, model-free).** Corrupt j of 6 examples and watch P@1:

| corruption | clean | 1 of 6 | 2 of 6 |
|---|---|---|---|
| random | 0.875 | 0.792 | 0.750 |
| off-by-one | 0.875 | 0.792 | 0.583 |
| sibling | 0.875 | 0.708 | 0.667 |

`route` **degrades gracefully — no cliff** (it ranks by degree of match, already tolerant), so the
router does **not** need majority/confidence changes as a prerequisite. But **plausible-wrong errors
(off-by-one, sibling — the realistic model failures) hurt more than random**, because a near-miss
output pulls resolution toward a *different* cell that really produces it. The real ceiling CN-6
routes into is **~0.58–0.88, set by the model's own error rate**, not the oracle's 0.875.

**(b) The circularity, resolved in the design.** To emit a *correct* (input,output) pair the model
must compute the function — but if it can, why does it need the cell? CN-6's design must answer this,
and the noise result makes the answer clean: **the emitted examples are EASY instances (small-case
bootstrap) or EXTRACTED from the task context — never the hard target computation the cell is for.**
The model demonstrates the *pattern* cheaply (2+2=4, or copies (200,15)→170 from the problem); the
cell does the *hard instance* (4823×9917) exactly. The circularity dissolves because the value is
"cheap pattern demonstration → exact hard computation," and the router's demonstrated tolerance (a)
covers the model's occasional slips on the easy cases. The **stronger** deployment variant is
extraction (examples from the task, the equipped-query 0.859 case) — noted as the primary
real-world path; pure generation (bootstrap) carries the model's empirical error rate, which the
gate must measure against the (a) tolerance band rather than assume.

**(c) Discriminativeness of canonical examples (`cn6_canonical_check.py`, model-free).**

> **RETRACTED then RE-RUN AT POWER (2026-07-14).** A first pass at **n=24** read a width gradient
> (tiny 0.625 → wide 0.750) as a ~15–22% "canonical inputs discriminate worse" effect, and a whole
> **compute-vs-discriminate "bind"** (wide=discriminating but uncomputable; canonical=computable but
> non-discriminating) that flipped the fork to *extraction-primary*. **Both were n=24 noise.** Two
> grounds, both correct: (i) at n=24, SE≈0.10, so wide−round was ~1.2σ — and it wasn't even monotone
> (tiny beat round); (ii) the bind conflated input *width* with computational *difficulty*, which are
> orthogonal (`is_even(65534)`, `max3(1000,2000,3000)` are wide **and** trivial).

**Powered re-run — leave-one-out over all 249 value cells** (resolution is training-independent, so
discriminativeness is a library property; n=249 collapses the bars). Router P@1 by input pool, ±SE:
**tiny(0..10) 0.546±.032 · round(0..100) 0.620±.031 · mid(0..1000) 0.600±.032 · wide(0..65535)
0.662±.031.** wide−round = **+0.042, z = +1.0 — not significant**, and not monotone (round > mid). So
**there is no width–discriminativeness gradient**; the only hint is the *tiniest* pool (single
digits) sitting a little low. **The bind is dissolved, generation is not capped, and extraction
should NOT become primary on this evidence.** (Fourth "artifact of how I sampled" catch in CN-6
alone; the α-curve error class, avoided this time before it redirected the spend.)

**What the powered check DOES establish:** the router's rank-1 resolution over the full 249-cell
library from 6 oracle examples is **~0.62 P@1** (≈0.83 P@5 from stage 1) — roughly flat across input
scales. That is the real ceiling CN-6 grades against — set by the inherent difficulty of resolving
1-of-249 from 6 pairs, not by input width — and it rises with more/better examples. Combined with the
noise result (plausible-wrong errors, the LLM kind, degrade gracefully): the honest generation
ceiling is **~0.55–0.62 base P@1** (or ~0.83 P@5, which the two-tier execution stage then resolves),
no width penalty.

**Consolidated stage-2 design (restored to the honest fork):** (i) emitted examples are
easy/extracted, never the hard target (dissolves the circularity); (ii) grade end-to-end P@1/P@5
against the **~0.62 / ~0.83** router ceiling; (iii) **build both arms as co-equal** — generation
(headline: "delegate by demonstrating"; covers unequipped queries) and extraction (deployment: the
equipped-query 0.859 path). If forced to one, **generation**, because its null is informative and
extraction almost certainly works if generation does. No fork flip; the checks left the original
framing standing, at a properly-powered ceiling.

### CN-6 stage 2 — corpus + resolution eval harness (graded on P@5, CIs baked in)

Built and validated before the training spend. Two corpora (`cn6_corpus.py`): **generation** (target
= a fresh example set the model must produce from the descriptor — genuinely hard, e.g.
`op lcm ... <call> 728 943 = 31144 ; …` requires computing lcm) and **extraction** (demos in context,
target copies them). ~13k train rows each, 24 held-out cells eval-only.

The eval (`cn6_eval.py`) grades on **end-to-end resolution, not router P@1** — the two-tier pipeline
needs the true cell in an *executable* top-k, which execution then confirms exactly — and every
number ships with a Wilson CI (the terrain is noisy: n=24 held cells, five sampling catches so far).
**Oracle ceiling** (correct examples, validates parse→route→resolution + gives the powered bar):

| | resolve@1 | resolve@5 | resolve@10 |
|---|---|---|---|
| all value cells (n=243) | 0.514 [.45,.58] | **0.749 [.69,.80]** | **0.835 [.78,.88]** |
| held-out only (n=24) | 0.625 [.43,.79] | 0.792 [.60,.91] | 0.833 [.64,.93] |

So the **operating ceiling is resolve@5 ≈ 0.75 / resolve@10 ≈ 0.84** (all-cells, tight); resolve@1
(0.51) is not the metric. Stage 2's gate: does the model's *emitted* spec resolve near this ceiling,
with emitted-example *correctness* reported alongside (a spec can fail by being wrong or by being
non-discriminating — correctness separates them). Remaining: `cn6_train.py` (train the base to emit
specs, both arms), then the model-generation eval against this ceiling.

### CN-6 stage 2 RESULT — extraction works, generation is compute-capped (prediction held)

Both arms trained (SmolLM2-135M, spec-loss on the emitted examples; generation plateaued at 0.85 —
can't compute perfectly; extraction hit 0.000 — a copy task). End-to-end resolution on held-out
cells (n=24, model-generated specs → parse → router → execution), Wilson CIs:

| arm | resolve@1 | resolve@5 | resolve@10 | emitted-example correctness |
|---|---|---|---|---|
| **extraction** | 0.708 [.51,.85] | **0.875 [.69,.96]** | 0.917 [.74,.98] | **0.979** |
| generation | 0.042 [.01,.20] | 0.042 [.01,.20] | 0.083 [.02,.26] | 0.097 |

**Extraction is a working system.** Given a task that carries I/O examples, the model copies them
(98% fidelity) and the runtime routes to the right cell — **including held-out cells never in the
token vocabulary** — at resolve@5 0.875, statistically at the oracle ceiling (0.79 held-out / 0.75
all-cells). This is **library-size-invariant invocation, end to end, working today** for the
equipped-query case — and it's the same 0.859 equipped-vs-0.387-paraphrase story from the other
direction: when the query carries executable content, the pipeline resolves.

**Generation is compute-capped, and the diagnostic proves it's not a router/mechanism failure.** Its
resolution (0.042) *tracks its correctness* (0.097) — exactly the pre-stated prediction. A 135M model
cannot compute lcm/mul for held-out cells, so it emits wrong examples the (working) router can't
resolve. The circularity bites at this scale: to generate a *correct* example you must compute the
function; for held-out arithmetic the model can't. This is the **informative null** — the model
delegates only where it can already compute — not a defect in the pipeline (extraction proves the
runtime resolves fine). A larger/math-stronger base would lift generation's correctness and with it
its resolution; the cap is the base model, measured.

**Precision, before this travels (the summary above is slightly generous):**
- **The extraction model-side contribution is *copying*, not *specifying*.** Correctness 0.979 = it
  lifts pairs already in the prompt; the *router* (training-independent) does all the resolution
  (0.875 ≈ the 0.83 oracle ceiling). So the honest model-side claim is thin: "the model copies I/O
  pairs from context," not "the model learned to specify the computation it needs." It invites the
  fair question *why is the model in the loop at all if the examples are already there* — you could
  route straight from the context. The model's residual contribution is deciding **that** a cell is
  wanted and **which** pairs to lift — real but small.
- **The deployment envelope is narrower than "works today."** Extraction needs the task to *carry*
  I/O examples (few-shot prompts, spreadsheets, worked examples). Most real queries don't — "compute
  the compound interest on this loan" has no worked pair. **The general case is the *unequipped*
  one — exactly where generation failed.** So CN-6's success case is the case where the answer was
  partly given; the case that matters most is still open.
- **Keep the retraction straight.** What was retracted (correctly) is the *width/discriminativeness*
  "bind" — the powered data says input width doesn't matter, and generation did **not** fail for
  that reason. What bit is the **original circularity** (to emit a correct example you must compute
  the function), which was live from the start and is what the correctness-0.097 data confirms. A
  reader must not conclude the retracted argument was vindicated; a different one was.
- **"Generation works once the base can compute" is a CONJECTURE, not a result.** Plausible
  (correctness scales with model size; discriminativeness is now measured flat, so correctness is
  the only remaining blocker) — but untested. **The deciding experiment (a swap, not a redesign):
  run the generation arm on a base that can do easy arithmetic** (Qwen2.5-1.5B/3B or similar —
  something that gets `lcm(4,6)=12`). If correctness climbs *and* resolution tracks it toward the
  ~0.83 ceiling → the unequipped case works and CN-6 is done properly. If correctness climbs but
  resolution doesn't → something else is wrong, learned cheaply. This is now the most informative
  unrun experiment in the lane.

### CN-6 stage 2 DECIDING RESULT — the conjecture is not confirmed at 1B; generation is computation-limited, and that is the point

The pre-stated deciding experiment (line above) was a *swap, not a redesign*: run the generation
arm on a base that can actually do easy arithmetic, regenerate the corpus with small (0..20) inputs
the base can compute, and ask whether correctness climbs *and resolution tracks it toward the ~0.83
ceiling*. Base chosen: **Llama-3.2-1B** (10/10 on an easy-arithmetic probe; standard arch; M3-fits).
gemma-4-E2B is likely stronger but is `model_type: gemma4` — nested text config, 262k vocab — a heavy
non-standard fine-tune; Llama-1B clears the precondition, so it decides the question. Fine-tuned
light-touch (top-6 layers + tied embeddings, lr 2e-4, 3000 steps), eval on the same 24 held-out cells.

**First: the comparison bar had to be re-measured at 0..20.** The old 0.83 ceiling used 0..1000
inputs. Oracle-correct examples at 0..20 (leave-one-out): all-cells resolve@5 0.711 [.65,.76],
**held-out 0.833 [.64,.93]** — *identical* to the 0..1000 ceiling. Shrinking inputs to what the base
can compute costs nothing at the ceiling (consistent with the powered width-null). So the bar stands.

**The result — correctness climbed, resolution did not move.** (held-out n=24, Wilson CIs)

| base / inputs | emitted-example correctness | resolve@5 | vs ceiling 0.833 |
|---|---|---|---|
| SmolLM2-135M / 0..1000 (orig) | 0.097 | 0.042 [.01,.20] | floor |
| SmolLM2-135M / 0..20 (control) | 0.167 | 0.000 [.00,.14] | floor |
| **Llama-3.2-1B / 0..20** | **0.306** | **0.083 [.02,.26]** | floor |

This is the pre-registered **"correctness climbs but resolution doesn't"** branch — *"something else
is wrong, learned cheaply."* Here is the something else, in three verified pieces.

**(1) The decode-collapse was a symptom, not the cause — proven by a diversity sweep.** Greedy
generation collapses onto one token: it emits `15 = 45 ; 15 = 45 ; 15 = 45` (repeated input; and
15×3, not 15²). The tempting read is "just a decoding artifact — sample and it'll resolve." It is not.
Sampling *does* diversify inputs, and resolution *does not follow*:

| decode | mean distinct inputs/spec | correctness | resolve@5 |
|---|---|---|---|
| greedy | 2.08 | 0.306 | 0.083 [.02,.26] |
| sample T=0.7 ×1 | 2.92 | 0.215 | 0.083 [.02,.26] |
| sample T=0.8 ×3 (union) | 8.58 | 0.204 | 0.042 [.01,.20] |
| sample T=1.0 ×3 (union) | 8.33 | 0.194 | 0.125 [.04,.31] |

4× more input diversity leaves resolve@5 statistically flat (~0.08; every CI overlaps, all far below
0.833). Diversity and correctness **trade off one-for-one** — sampling varies the inputs but then the
base can't compute the varied outputs, so correctness drops and net resolution is pinned. Unioning
many sampled specs (nsample 3) *lowers* resolution: the extra pairs are mostly wrong and poison the
router — the exact plausible-wrong penalty measured in the noise-sensitivity check. **Resolution is
computation-limited, not decode-limited.** The greedy `15,15,15` is the model repeating the few
canonical pairs it actually knows; forced to vary, it fabricates.

**(2) The correctness climb is real and base-driven, but nowhere near enough.** The control isolates
it: at fixed 0..20, SmolLM2→Llama moves correctness 0.167→0.306 (+0.14; the base is the lever), while
input-range alone moves SmolLM2 0.097→0.167 (+0.07, within noise on n=24). So a compute-capable base
*does* emit more correct examples — the conjecture's first clause holds. But 0.306 is far too low:
**12/24 held-out cells sit at correctness 0.00.** The held-out slice (axis-A, a structural draw) is
dominated by *specialized* functions — `jacobi_symbol`, `crc16_step`, `mobius_function`, `isqrt`,
`fnv1a_step`, `zscore_q8`, `norm2_sq` — which are **not the easy arithmetic Llama aced**. The
conjecture quietly assumed the targets were base-computable; for a random library slice they mostly
are not. `isqrt(19)` → the model emits 0; `square(15)` → 45. This is a base-capability ceiling,
measured, not a pipeline defect.

**(3) Even where the base computes correctly, discrimination isn't guaranteed.** Among Llama's four
fully-correct specs, resolution is 2/4 (0.50, n=4 — directional only): `median3` and `between_exclusive`
emit distinct correct examples and resolve at rank 4; `luhn_check` and `mobius_function` are computed
*correctly* but emit low-entropy outputs ({0,1}, {−1,0,1}) on small inputs that dozens of cells share,
so they land at rank 49 / rank 20. A residual correct-but-non-discriminating tail survives even past
the computation wall.

**Verdict.** The conjecture *"generation works once the base can compute"* is **not confirmed** at 1B,
and the experiment says why cheaply: to write a *resolving* spec the model must **compute the target
function** — the very work cells exist to offload — so free-form generation inherits the base's
capability ceiling, and (residually) a discrimination penalty on low-entropy functions. Where the base
computes, it mostly resolves (2/4); but a small base computes too little of a real library for the arm
to be useful, and no decoding trick buys the missing correctness. This does not *refute* the thesis —
it **sharpens** it. The two invocation paths that work do so precisely because they never ask the model
to compute: **extraction** (0.875) copies I/O pairs already present and lets the training-independent
router resolve; **CN-1's fingerprint address** projects *behaviour* to a token, so the model narrows by
identity, not by re-deriving the function. Free-form spec *generation* is the path that re-imports the
computation problem, and it fails exactly there. That is a clean boundary, and it is the CN-6 result:
**delegate by pointing (identity) or by carrying (examples), not by re-computing (generation).**

Open only as a scaling footnote, not a live claim: a math-specialist or tool-augmented base, or a
held-out set filtered to base-computable functions, would raise generation's correctness and with it
its resolution on that subset — but it cannot escape the structural point that generation must compute
what identity/extraction merely reference. Apparatus: `cn6_corpus.py --input-max`, `cn6_train.py
--base/--input-max`, `cn6_eval.py --sample/--nsample/--input-max` (KV-cached greedy/sampled decode,
per-cell correctness|rank split, fully-correct discrimination bar), `cn6_inspect.py` (per-example
oracle diff). Six configs, one held-out set, every number with its CI.

## Immediate next steps (not yet done)

1. Root-cause `spin_pool`'s remaining concurrency bug (bug 3) for real —
   needs a live debugger session on an actual crash, not more static or
   synthetic work (both were tried extensively and came up clean).
2. CN-0: a real hyperparameter sweep (λ and MLP architecture were single
   fixed choices even in the rerun), multiplication/subtraction beyond
   addition, and a dedicated narrative-vs-others contrastive probe design —
   narrative's near-total failure (0–3% at every layer, both runs) looks
   qualitatively different from the other three families', not just a
   smaller-N version of the same gap.
3. ~~CN-2: fix the plan-IR's unsigned-only limitation~~ **Done** (the i32
   signed lane): all 3 escalations became verified matches, agreement
   0.984, zero coverage holes left in the verifier. ~~Then the real G2
   build~~ **Done too** (the correction loop, section above): scoped
   wrong-number rate 0.016 → 0.000 at ~4.8% overhead; the residue is
   unscoped claims and wrong plans, per the pre-registered null. Still
   open for CN-2: the in-decoder span grammar (couples to CN-1), and a
   harder battery where scoped arithmetic errors are frequent enough for
   final-answer accuracy to move.
4. Wave 2 (CN-1's H1 factory, CN-3's prosthetic) hasn't been scoped yet.

## CN-7 R1 — numeracy midtrain with cells as validator: the mask holds, the panel fires twice, and the substrate trade is measured

Pre-registration: `cell-native-architectures-cn7-preregistration.md` (v0.1 pinned
85fcbab before any training; amendments §8, each committed before the
measurement it governs existed). One day end-to-end, 2026-07-15/16. Every
checkpoint sha256-manifested and write-protected (`cn7_ckpt_manifest.json`);
overwrite guards added to all CN trainers after the §8.3 lesson — and the
guard caught a real name collision mid-run (dcaf303).

**Substrate repair first (§8.1).** Wiring CN-7.0 exposed that TinyModel v11's
committed tokenizer artifacts are a different piece→id mapping than the
checkpoint was trained with (NLL 18.0 — worse than uniform — through the
committed mapping; 0.66 through the recovered SP model). CN-1 had encoded
through the wrong mapping (29.8% trained-row hits). CN-7 moved to the original
SP id space and re-baselined: **B9′ = 105** (per-seed 105/108/93 across three
visibly different trajectories; novel|seen std 7.9 vs novel|novel std 137 —
the invariance dissociation, reproduced on a healthy stack), so CN-1's
held-out level was never a substrate artifact; and seen-cell addressing
jumped to median rank 1–5 of 790 (top1 0.31–0.46), which WAS substrate-starved.

**CN-7.1 (7.55M tok/epoch, 45.0% replay): both audits clean.** 188,520
corpus claims re-derived from row text by independent parsers and re-executed
against cells — 100% signed; zero beyond-tier answer tokens carry loss; no
held-out cell appears anywhere. The audit's first run flagged 5 failures that
were bugs in the AUDIT (canonical/narrative disambiguation) — the two-routes
design surfacing its own defects, as intended.

**The midtrain worked wherever it was allowed to.** Fresh-instance
(seed-981) role-NLL: s1 answers 4.49 → **0.051** (succ 0.0002, parity 0.0006,
add 0.012; residuals mul 0.138, mod 0.274); call grammar 6.11 → 0.0002;
P-e improved 6.6% (bound never approached; §8.8 split graded: gains uniform
across cardinal-word vs plain sentences → continued pretraining on an
unconverged 24M-token base, NOT cross-species transfer — the digit-split
power check had already killed the digit-form story, 7/8,725 val sentences);
deck register unchanged; narrative probe 0.965 (B12's cliff inverted).

**The mask held, with a behavioural signature.** Three instruments:
s3 masked values stayed AT floor (4.14 → 4.35) while in-tier twins collapsed
(4.04 → 0.77); s2 injected spans were actively SQUEEZED away from floor
(4.23 → **11.0 nats**) — after 15M tokens containing those exact spans, the
model treats its own training data's injected values as e^-11-improbable:
not absence of gradient but presence of maximal surprise. The
call-is-the-abstention thesis now has a number; CN-7.5 (running) watches it
collapse when the mask is removed.

**Both failure gates fired, and the gate design converted failures into
mechanism.** P-b failed (0.111): free-running emission mode-collapses to a
template echo ("10 10 = 1", arity-adapted) on seen AND held-out descriptors —
not a composition failure; s3 values were undertrained (0.77 nats ≈ p 0.46;
S3 was 11% of the mix). 7.3 ran recorded-not-graded (0/9, 0/15); the sampled
yield (0.110/0.135, unstratified, binary-output luck) left the 7.6 STaR gate
unmet. Then P-d1′ failed (held-out rank **208** vs threshold 137) and the
kill criterion fired: the full-model midtrain broke computed addressing while
association simultaneously improved to its best-ever (rank 1).

**The fallback resolved the mechanism (§8.10).** Attention-only arm, same
corpus/tokens: P-d1′ **98 — at the pre-midtrain level** — while fresh-instance
numeracy landed 6× weaker (0.308; P-a2 fails outright at 0.620). Both arms
warped the measured emission-position geometry EQUALLY (RSA 0.59/0.64,
Procrustes ~0.72, uniform across buckets), so drift magnitude is not the
operative variable; what kills the fingerprint pathway is specifically FFN
changes the fp protocol cannot re-fit. **At 115M under this recipe,
in-weights numeracy and computed cell-addressing compete for the same
substrate: FFN training installs arithmetic and breaks addressing; FFN
freezing protects addressing perfectly and starves arithmetic.** Association
is robust to everything — only the computed address is fragile, and only to
FFN plasticity. Three pre-stated branches; branch (3) (FFN-freezing-as-free-
policy) is dead.

**R1 disposition.** No graded 7.3 on either arm — the 7.2 gate did its job
twice, which is the panel working, not the experiment failing. Open: CN-7.5
no-mask control (running; the 11-nat watch-number), CN-7.4n noise probe
(registered, N1–N4), 3-seed P-d2′ if the ladder continues. R2 prescriptions,
all instrument-derived: raise S3 fraction, add S1 EOS supervision, and either
budget the FFN trade explicitly or route numeracy around weights entirely.
The rung's own arc is the programme's thesis performed by the programme: the
model that tried to hold everything in weights broke the delegation path it
needed most, and the recovery was to freeze the parts that store and let the
interface adapt.

### CN-7 R1 addendum — 7.5, the probes, the replicates, and the loop closing (§8.13–8.17)

The section above was written before the evening's second act; the graded record is
prereg §8.13–8.17. In brief: **7.5's watch-number collapsed on cue** (injected spans
11.0 → 0.40 nats — but on FRESH instances: within-distribution learning, not instance
memorisation; prediction (a) wrong in the interesting direction). The **off-distribution
probe returned the frozen cliff signature** — 0.75–1.00 exact in-range, 0.00 exact one
digit past, on all six cells, cliff at the corpus boundary — while the **noise probe found
everything robust** (σ ≤ 0.08): crammed ≠ fragile; redundant storage is noise-robust while
containing zero algorithm. Corrected MDL operationalization: compression is what survives
leaving the distribution. The **permutation-null yield** confirmed its registered
prediction exactly (excess −0.002/+0.007): the sampled yield was entirely signing-by-
chance. The frozen §8.11 rule fired: **option (iii); the tier boundary survives as a
capability boundary for algorithms** (in-range surface learnable, worthless against cells).

The P-d saga resolved as variance, not mechanism: the 208 kill-firing did NOT replicate
(masked s82: 93), the §8.13 mask×FFN interaction died with it, and the final table —
frozen-FFN {105,108,93}+{98} spread ≤15 vs plastic-FFN {93,208}+{108,159} spread 51–115 —
says **FFN plasticity multiplies W_f-fit variance ~an order of magnitude, mask-irrelevant**.
The ~100 held-out ceiling is the mode of every configuration tested (hypothesis registered:
problem-intrinsic to the library's behavioural geometry). Single-seed P-d gates retired;
§8.16's multi-seed W_f gate is necessary for any plastic-FFN recipe; §8.15's "cost mooted"
line corrected in §8.16 (the frozen FFN's IN-tier cost is real — Tier A is a finite domain
where in-range learning is the deliverable).

Final scoreboard: nineteen graded registrations, eight wrong, every wrong one executed by
an instrument registered before the belief existed — including two of the reviewer's own.
And the day ended with **the broker loop running end-to-end for the first time**
(`cn7_broker.py`): model parses prose, emits `<call> ⟨safe_div⟩ 157 16 </call>`, the cell
answers 9 in microseconds, the model narrates the verified number — " 9 sweets. The
children smiled." The masked model's greedy output leaves a literal hole where the answer
belongs; the runtime fills it. That is the architecture, running.
