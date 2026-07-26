# Roadmap — where this is and what to do next

Written 2026-07-25, updated 2026-07-26. `script/SCRIPT.md` is the video plan and
`script/RECORD.md` is the shooting script; this is the *state* of the work behind
them, so a cold start can pick up without re-deriving anything.

---

## 2026-07-26 — what changed, and the one thing that hurt

**Act 4 is no longer the long pole.** Its training arm is self-contained here, it
has run, delegation works, `/broker` is built into the REPL and the closing shot is
typed live rather than canned. What Act 4 still lacks is *instruments*, not weights.

**Both arms are measured and paired**, seed 80, same base, same budget, same
schedule, same 710 held-out rows:

```
absorb  (CN-8, maths)  1.5893 → 1.6976   +6.8%   fails CN-7's registered P-e (≤+5%)
delegate (CN-9, cells) 1.5940 → 1.5694   −1.5%   better than the model it started from
```

**A learning-rate bug was found and fixed** (`07bfb7d`). The decay was denominated
in steps via a step count estimated from an assumed 30 tokens per row — a property
of the *corpus*, not the recipe, and the two arms do not share it. At the same 12M
budget the maths arm annealed to 12% of peak and the cells arm stopped at 52%. Every
pre-fix run is superseded.

**The two probes ran for the first time, and one overturned the act.**
`cliff_probe` gives 0.99/1.00/1.00/0.87 in-range collapsing to ~0 one digit past,
with the base model flat across all three bands — so the midtrain *creates* the
cliff rather than revealing it. `heldout_probe` puts fact-unseen at **0.98**, which
is row 2 of the pre-registered map, not row 1. "It memorised a table and generalised
to nothing" is false and is out of the script.

**What hurt: a run died at 96% and there was nothing to resume from.** See next
actions.

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

### 0. A resume path — the highest-value engineering left ⛔

**`cells-s80` reached step 12,000 of ~12,530 and was lost entirely.** Colab dropped
the worker, the control plane requeued the run, and the run restarted from zero,
because nothing in this repo can resume. Three minutes from the end, an hour thrown
away. It has happened once and the exposure is permanent until this is built.

Four things are missing, and only the first is hard:

1. **Optimizer and scheduler state in the checkpoint.** `write_harness_ckpt` writes
   `model.safetensors` + `meta.json` + `.ready` — weights only. AdamW's moments and
   the LR schedule's position are both needed, and they are ~2× the model in bytes,
   so this interacts with the 256MB control-plane ceiling in § 1.
2. **A `--resume` flag** on `train_mathonly.py`, `train_cells.py` and
   `train_phase3.py`, loading the newest complete `step_*` and continuing.
3. **Data-order recovery.** `epoch_batches()` reshuffles from a seeded RNG whose
   state dies with the process, so a resumed run is not bit-identical to an
   uninterrupted one. Either persist the RNG state or accept and *document* that
   resumed runs are not reproducible from the seed alone — the second is fine, the
   silent version is not.
4. **CP-side handoff.** `submit_run`'s docstring already claims a run "resumes from
   its last checkpoint if the worker is lost mid-run". That is not true of any unit
   here today; the control plane re-dispatches and the entrypoint starts over. Either
   make it true or correct the docstring.

**Until it exists, treat every run as all-or-nothing** and prefer shorter budgets
over longer ones on Colab.

### 0a. A worker can heartbeat while its runner is dead, and nothing notices ⛔

**Observed 2026-07-26, and it cost 1h45m of a T4.** Two seconds after
`mathonly-s81` completed, `cells-s82` was assigned to the same worker. It never
started. The control plane then cycled it — `assigned` →
`assignment_timed_out` → `queued` → `assigned` — **seventy times over 103
minutes**, on a strict ~90-second period, while the worker reported a healthy
2.9-second heartbeat throughout.

Telemetry from the middle of it says plainly that nothing was running:

```
gpu_util 0 · gpu_mem_used_bytes 0 · gpu_power_w 11.7 · gpu_temp_c 42 · cpu_util 0.02
```

The run has **no log lines at all**, so the job never reached the worker's
executor. The heartbeat thread outlived whatever claims and runs work, and from
the control plane's side those are indistinguishable: a worker that says
`connected` every three seconds looks identical to a working one.

**Nothing was lost** — the three queued runs are intact and everything already
completed is safe. This is wasted lease, not wasted work. But it is a silent
failure of exactly the kind this project keeps finding, and it compounds § 0:
with no resume, a worker that quietly stops accepting work is the second way to
lose an afternoon without being told.

Four things worth doing, roughly in order of how cheap they are:

1. **Give up on a worker that repeatedly fails to start.** Seventy identical
   cycles is not a retry policy, it is a loop. After N consecutive
   `assignment_timed_out` for the same (run, worker) pair, mark the worker
   unhealthy and stop assigning to it.
2. **Say so.** Nothing surfaced this — `list_runs` showed `queued`/`assigned`,
   which reads as "waiting its turn". It took reading 141 lifecycle events to
   see the loop. A run whose assignment has timed out more than once should be
   visibly distinct from one that is simply waiting.
3. **Heartbeat the runner, not the process.** The liveness signal should come
   from the component that claims jobs, so that its death shows up as a
   disconnect rather than as unexplained idleness.
4. **Treat GPU-idle-while-assigned as a signal.** The telemetry already knows:
   `gpu_util 0` on a worker holding an assignment is either a very long fetch
   or a dead runner, and both are worth a warning.

**Operational note until then:** if a queue stops advancing, check
`run_events` rather than `list_runs`, and restart the Colab worker. The queued
runs re-assign to a fresh worker with no loss.

### 0b. The lease does not self-drain, and I described it as though it did

`colab_cell(lease_min=240)` produced a worker with **no lease registered** —
`lease_status` returns `no lease`, and the worker ran 4.9 hours without draining. The
drain-window behaviour in the bootstrap cell is either not wired or not reported.
Worth knowing before planning a shoot around it: a Colab worker runs until Colab
kills it, and § 0 is what makes that expensive.

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

- ~~**Act 3's replay baseline of 1.6079 is MPS**, against Colab's 1.4904.~~
  **Settled 2026-07-25, and the answer is that the gap was never the device.**
  With the corpus fixed, both machines report a pre-midtrain TinyStories val NLL
  of **1.5893** on a 710-row validation set derived from a 96,894-row corpus —
  the same figure to four decimals on MPS and on a T4. The old 1.6079/1.4904
  spread was *entirely* the corpus divergence (663 rows vs 710); device
  arithmetic contributed nothing measurable.

  So a cross-device delta is quotable now, provided both sides share a corpus
  build. The harness's standing rule — keep a replicate set on one device —
  still holds for a different reason: it is about seed variance, not this.

- **Seed 80's own replay delta: val NLL 1.5893 → 1.7182** over 12M tokens. The
  midtrain costs story-modelling loss; that is the forgetting number Act 3 has
  to be honest about, and it is measured on one device with one corpus.
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
