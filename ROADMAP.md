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

- **Seed 80 is complete, and its whole curve is safe.** `EXEC-…-00028`, 12.00M
  tokens, 22,012 steps. All 15 checkpoints (step 1500 → 22012) are on the Hub at
  `chrishayuk/v11-tinystories-115m-mathonly-ckpts`, each verified against the
  sha256 the Hub itself serves. They were rescued off the worker, not uploaded by
  the rig — see below.
- **Arithmetic arrived.** By 16M tokens the samples read `7 + 5 = 12` and
  `Lily had 3 apples… → 7 apples`, both correct, while the same sum in number
  words still reads `11 apples`. That is the taught band clearing, which is the
  gate the outcome map puts before everything else.
- **The CN-7 follow-up is registered.** `cn-8-mathonly-midtrain-cliff` under
  `cell-native-architectures`, with SCRIPT.md's pre-registered outcome map as its
  design and a logical run per seed. Seeds 81/82 carry `experiment_ref`; seed 80
  predates the experiment and is joined by run id, with its caveats recorded.

## In flight

| run | seed | state |
|---|---|---|
| `EXEC-…-00028` | 80 | **complete**, checkpoints rescued to the Hub |
| — | 81, 82 | to resubmit on unit `adf33a7a` once the fixed CP + worker are live |
| — | — | `ckpt-path-preflight` to rerun first: the end-to-end proof the path works |

---

## Next actions, in order

### 1. Checkpoint upload — root cause found, fixed, deploying ⛔ still blocking

**The recorded diagnosis was wrong**, and the wrong part mattered: this was never
the worker. `outputs.rs` marking a failed upload collected is a real bug and is
also fixed, but it is not what was happening — the uploads *succeeded*, which is
why no log line ever appeared. Only failures log.

`ingest_checkpoint` read the checkpoint back out of R2 to sha256 it:

```rust
let model = self.artifacts.get(...).await?;   // Vec<u8>
let model_hash = hex::encode(Sha256::digest(&model));
```

`model.safetensors` is **606,621,456 bytes**. The control plane is a
shared-cpu-1x with **256 MB**. Not slow, not flaky — a 606MB `Vec` cannot exist
in a 256MB machine, so ingest could never complete for any checkpoint, ever. It
also explains the repeated `running` events on 00028: those are the worker
reconnecting after the control plane died.

Fixed in gpu-training-harness (`64f23ab`, `b28b6bb`): the worker hashes each file
as it uploads it — the bytes are already in hand, and the wire has carried unused
`sha256`/`bytes` fields since M1 — and the CP uses that instead of fetching. Plus
bounded retries, so `collected` means *uploaded*.

**The second fix is the one that stops this recurring.** The archive sweep runs on
a timer over every completed run with a checkpoint, and it reached Drive the same
way. Fixing ingest alone would have moved the crash, not removed it: checkpoints
would finally land and the next sweep would crash-loop the CP. Hot→final never
needed the bytes (server-side CopyObject), so that is now unconditional; only the
Drive leg is gated, and refuses above 64MiB rather than attempting it.

Still open, and why Drive is not yet a real tier:

- **Drive cannot take a checkpoint at all** until `upload_to_path` streams. Today
  a checkpoint reaches R2Final and stops, so it lives on R2's lifecycle rules
  rather than in the canonical copy.
- **256 MB re-enables nothing.** Nothing now needs to hold a checkpoint, but both
  the ingest fallback and the Drive leg are permanently disabled by it.

`training/colab_rescue_cell.py` is the tourniquet and it works — 15 checkpoints,
9.1GB, hash-verified. Keep it until a run's checkpoints demonstrably land by
themselves. It proves write access by creating the repo up front rather than
reading `role`, because a fine-grained token reports `fineGrained`, not `read`,
and sails past the obvious check before 403ing half a gigabyte later.

### 2. ~~Register the CN-7 follow-up~~ — done, but one thing is not

`whoami` still reports `experiments_key_set: false`, so runs mirror under the
shared default identity. Fixable on the dashboard's Team screen.

### 3. Publish to HuggingFace — `--phase` landed; the join now exists

`publish_pretrain_hf.py --phase {pretrain,mathonly}` swaps the lede, the limits
section, the training table's provenance rows, the tags and `not_included`. A
fifth refusal checks it against the checkpoint's own `meta.json` phase, and runs
*before* the tokenizer and vocabulary guards — those pass happily on a
correctly-built checkpoint being published under the wrong story. Verified: the
pretrain card renders byte-identical to what is on the Hub today.

It would have refused every mathonly checkpoint anyway, because guard 1 wants
`tokenizer_hash` and the trainer wrote five fields, none of them a join. It now
writes the tokenizer hash, the base repo and sha the entrypoint verified, the
corpus identity it re-proved, the seed, the run id and the emergence samples.

**Seed 80's rescued checkpoints predate that** and carry none of it, so they are
not publishable as a headline model without reconstructing the join by
measurement from the sandbox that produced them. Seeds 81/82 will not have this
problem. Decide whether seed 80 needs re-running for that reason alone — its
*result* is fine, only its provenance is thin.

### 4. Pre-flight for the Colab cell — the step-0 probe landed, the cell has not

`train_mathonly.py` now writes a `step_0` checkpoint before the first gradient,
so a broken write path shows up in seconds rather than forty minutes in. It makes
the failure visible early, **not fatal** — the trainer has no credentials to ask
whether the bytes landed, so checking `list_checkpoints` a minute in is the other
half and belongs to whoever dispatched the run.

Still to write: the cell-side pre-flight. Hardware (GPU present, ≥13GB free,
disk), services (chuk-train, chuk-datasets, corpus identity registered), HF token
**proved writable, not assumed from its role**, base model reachable,
chuk-experiments reachable, and `experiment_ref` both **resolving** and **set**
for a research run — a seed replicate silently filed under `harness-runs` is
worse than a failure, because the numbers exist and aren't findable.

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
