# Roadmap — where this is and what to do next

Written 2026-07-25. `script/SCRIPT.md` is the video plan; this is the *state* of the
work behind it, so a cold start can pick up without re-deriving anything.

---

## Done and verifiable

- **Act 1e's base model exists and is published.** 16M tokens, published v11
  tokenizer, `chrishayuk/v11-tinystories-115m-base`, model sha
  `1841e058…`. Loss 11.07 → 1.70 in ~120 min on M3/MPS. First logged loss is
  `ln(71260)` to two decimals, measured not asserted.
- **The corpus is registered and content-addressed.**
  `tiny-model/mathonly-midtrain` v1, `DSV-20260725-160944-00002`. Registered with the
  **bootstrap admin token**, which minted no key — so anonymous catalog reads still
  work and v-tokenizers' CI (including fork PRs) is untouched. See
  chuk-datasets-server's roadmap for why that matters and what would force the door
  open.
- **The corpus is now deterministic, and cross-machine.** `ds.shuffle(seed, buffer_size)`
  on a *streaming* dataset was the culprit: the seed controlled the shuffle but not
  what arrived to be shuffled. Replaced with a seeded stride over the unshuffled
  pinned revision. A Colab T4 independently rebuilt it byte-identically to the Mac —
  proven by `--expect-sha ff7bf26b…` in the entrypoint, which refuses before using
  any GPU.
- **Dispatch to Colab works**, via `colab_cell` → join → `submit_run`, entrypoint
  `midtrain`. The T4 runs ~2.2× the Mac (2,630 vs 1,183 tok/s).
- **The S1-EOS remediation is applied.** R1 registered this failure and its fix and
  neither had been done here: 0 of 96,548 rows carried EOS, so free-running
  generation ran on into template-echo (P-b 0.111 free vs 0.91/0.97 teacher-forced).
  90,000 drill rows now terminate; replay rows deliberately do not.

## In flight

| run | seed | state |
|---|---|---|
| `EXEC-…-00028` | 80 | training on Colab, identity-checked |
| `EXEC-…-00029/30` | 81, 82 | queued behind it |
| `EXEC-…-00031` | — | smoke pretrain, queued: the end-to-end test of the checkpoint upload path |
| `EXEC-…-00027` | — | **zombie**, crash-looping on an old code unit, ignores cancels |

---

## Next actions, in order

### 1. Fix checkpoint upload before trusting any run's output ⛔ blocking

Checkpoints are **not landing**. `list_checkpoints` is empty for every run today
despite `[harness ckpt] step_1500` in the logs. Root cause is identified in
`chuk-compute-worker/src/outputs.rs` — a failed upload is marked collected and never
retried, and the `Artifact` message is gated on success, so the loss is silent. Full
write-up in gpu-training-harness's ROADMAP.

Until this is fixed, **a finished run's checkpoint dies with its Colab runtime.**
That is the single most important thing outstanding, because everything below
consumes a checkpoint.

Interim rescue if a run must be saved: a second notebook cell that globs
`/tmp/**/ckpt/step_*/model.safetensors` and pushes `.ready` dirs to HF. Safe
alongside training (`.ready` is touched last).

### 2. Register the CN-7 follow-up experiment, and attach runs to it

Runs **are** mirrored to chuk-experiments — but into
`gpu-training-harness/harness-runs`, whose own hypothesis says those are
*"infrastructure dry-runs and unattached scratch work — not research conclusions."*
Correct, because no submission passed `experiment_ref`.

CN-7's `next_action` asks for this arm to be registered as a follow-up under
`cell-native-architectures`. Do that **before** the rerun, then pass
`experiment_ref` on each seed. Register the **pre-registered outcome map** from
SCRIPT.md § "PRE-REGISTERED outcome map" as the design — it was written before any
result, and that is the only thing that makes it worth anything.

Also: `whoami` reports `experiments_key_set: false`, so runs mirror under the shared
default identity. Fixable on the dashboard's Team screen.

### 3. Publish properly to HuggingFace

**The rig has no HF integration at all** — checkpoints go R2Hot → R2Final → Drive.
Publishing is manual and always will be until someone adds it.

`training/publish_pretrain_hf.py` is the right tool (five refusals, and it verifies
by loading the *downloaded* artifact and generating). But **its card asserts "the
base pretrain only… no maths mid-training" and "It cannot do arithmetic"**, both
false for a mathonly checkpoint. Add a `--phase {pretrain,mathonly}` that swaps
those sections. Do not ship the pretrain card for a midtrain checkpoint.

Intermediate checkpoints belong in a separate bucket repo
(`…-mathonly-ckpts`, `seed80/step_N/`), not a headline model repo — the `--curve`
probe wants several, and they are research artifacts rather than a release.

### 4. Add pre-flight to the Colab cell

Drafted this session; not yet committed. Checks hardware (GPU present, ≥13GB free,
disk), services (chuk-train, chuk-datasets, corpus identity registered), HF token
**present and write-scoped**, and base model reachable. Add: chuk-experiments
reachable, `experiment_ref` **resolves**, and `experiment_ref` **is set** for a
research run — a seed replicate silently filed under `harness-runs` is worse than a
failure, because the numbers exist and aren't findable.

The strongest addition is a **step-0 probe checkpoint** in the trainer, so a broken
upload path fails in seconds rather than at step 1,500.

### 5. Then, and only then, read the result

`training/heldout_probe_mathonly.py --curve`, n≥250/band.

**Read `rung` on the `taught` band first.** While it reads `none` or `prior`,
nothing else in the table is interpretable — a fact-unseen zero is a
not-yet-learned, not a failure to generalise. This is row 0 of the pre-registered
map and the most tempting error available.

Current signal points at row 0: `7 + 5 -> 11` was stable across three checkpoints,
and `11` for both `7+5` and `3+4` is the answer marginal, not magnitude.

---

## Known-bad numbers in SCRIPT.md

- **Act 3's replay baseline of 1.6079 is MPS.** Colab reports **1.4904** for the same
  checkpoint and code. Most of that gap was different validation data (the corpus
  divergence, now fixed), but until it is re-measured on one device with one corpus,
  **do not quote a cross-device replay delta.**
- Every 🔶 in Act 3 is still CN-7's mixed run.
- The `call grammar 6.11 → 0.0002` line is deleted and cannot return: this arm is
  cell-free by construction, so it has no call grammar to measure.

## Deferred

- **Phase 3** (frozen-FFN, 8M tokens). Act 1f's `/full` vs `/compiled` needs it and
  Act 4c calls back to it. Cheap; nothing depends on it yet.
- **Act 4** — the long pole, unscoped: cell-token vocab extension (71,260 → ~72,050),
  zero-gradient answer masking, the 11-nats measurement and its unmasked control,
  and `cn7_broker.py`'s checkpoint. `/broker` and `run_broker.sh` stay blocked.
- **An S2-analogue** (in-tier word problems, no cell content) if Act 3's narrative
  arithmetic comes back weak. Note the trap: an S2-analogue carrying `S2_BEYOND`
  frames would reproduce CN-7.5's no-mask arm, which §8.11 already graded and
  rejected — and would teach interpolation *through* the tier boundary Act 3c
  depends on. In-tier operands only.
- **Fetch the corpus instead of rebuilding it.** Registration uploaded a manifest,
  not bytes: `"locations": []`. Needs an R2 location, then the entrypoint fetches and
  verifies like it already does for the base checkpoint.
