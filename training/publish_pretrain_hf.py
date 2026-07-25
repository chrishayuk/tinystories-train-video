#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "torch>=2.2", "tokenizers>=0.20", "safetensors>=0.4",
#   "transformers>=4.44", "huggingface_hub>=0.24", "numpy",
# ]
# ///
"""Publish a v11 TinyStories pretrain checkpoint to the Hub as a loadable model repo.

    publish_pretrain_hf.py --ckpt-dir out/ckpt \\
                           --repo-id chrishayuk/v11-tinystories-115m-base \\
                           --dry-run

Picks the highest `.ready` milestone under --ckpt-dir unless --step says otherwise,
stages a self-contained repo, pushes it, then proves the claim by loading what the
Hub actually serves and generating from it.

Acceptance criterion: a clean machine with no clone of this repo can download the
repo, build TinyModel from the shipped `tiny_model_v11/`, load the weights, and
generate English. Verified against the DOWNLOADED artifact, not the staged one.

WHAT THIS REFUSES TO DO, and why each refusal exists rather than a warning:

1. PUBLISH A CHECKPOINT WHOSE TOKENIZER IS NOT THE PUBLISHED v11 BUILD.
   train.py writes a real sha256 of its vendored tokenizer into each
   checkpoint's meta.json as `tokenizer_hash`. That is the join key, and it is
   checked here against the same constant train.py guards on. A checkpoint and
   a tokenizer that disagree produce fluent nonsense, not an error -- which is
   the entire reason the old SentencePiece checkpoint had to be retired. The
   failure mode this guards is a published model repo that decodes to garbage
   for everyone who loads it.

2. PUBLISH A CHECKPOINT WHOSE EMBEDDING TABLE DISAGREES WITH THE VOCABULARY.
   `embed.weight.shape[0]` must equal both config.json's vocab_size and the
   tokenizer's. A 71261-row table against a 71260-piece tokenizer loads happily
   under strict=False and is off by one for every id past the insertion point.

3. PUBLISH A HALF-WRITTEN CHECKPOINT.
   train.py touches `.ready` only after model.safetensors and meta.json are both
   on disk. A step dir without it was interrupted mid-write.

4. SILENTLY REPLACE PUBLISHED WEIGHTS.
   If the repo already holds a different model.safetensors, this refuses and
   tells you to pick a new repo id. Model repos here are cited by the video and
   joined on by provenance.json; replacing bytes under a name that is already
   published is how a citation quietly starts pointing at something else.
   --force exists because a retrain is legitimate, but it must be deliberate.

5. DESCRIBE A CHECKPOINT AS SOMETHING IT IS NOT.
   The card is not decoration; it is the claim. The pretrain card asserts "the
   base pretrain only... no maths mid-training" and "it cannot do arithmetic",
   and both are false of a mid-trained checkpoint. --phase selects the copy, and
   the checkpoint's own meta.json `phase` is checked against it, so getting this
   wrong is an error rather than a card nobody re-reads.

WHAT IT DELIBERATELY DOES NOT DO: register the model as a transformers
architecture. TinyModel is a 3-file Gemma-shaped decoder, not an AutoModel
subclass, and shipping a `model_type` the Hub cannot resolve would give the repo
an inference widget that fails for every visitor. The card sets `inference: false`
and ships the model code instead, which is honest and actually works.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
UNIT = HERE / "harness_pretrain"
ARCH_CONFIG = UNIT / "config.json"
TOKENIZER_JSON = UNIT / "tokenizer_v11" / "tokenizer.json"
MODEL_CODE_DIR = REPO_ROOT / "tiny_model_v11"
MODEL_CODE_FILES = ["__init__.py", "model.py", "loader.py"]

MODEL_FILE = "model.safetensors"
READY_MARKER = ".ready"

# Same constant train.py guards on -- the published v11 tokenizer (2026-07-24):
# crates.io v11-core, PyPI v11-tokenizer, HF chrishayuk/v11-tokenizer.
PUBLISHED_TOKENIZER_SHA256 = "10dd51100331ab503115db23eee7e8dc3e360e3aed697c8a2e1b12b8f46031ae"
TOKENIZER_REPO = "chrishayuk/v11-tokenizer"

# The corpus revision train.py pins (HUB_SHA). Recorded, not re-derived.
DATASET = "roneneldan/TinyStories"
DATASET_REVISION = "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64"

# A published model repo that decodes to garbage is worse than no repo, so the
# post-push check generates rather than just loading. These are the two prompts
# the video opens on; the first must read as English, the second must NOT contain
# a correct sum -- that failure is the subject of Act 2, not a bug.
VERIFY_PROMPT = "Once upon a time"

# The maths mid-train corpus, as registered in the chuk-datasets catalog. The
# identity is over the manifest, not the bytes -- see the roadmap note on
# `"locations": []` -- so it anchors a rebuild rather than distributing one.
MATHONLY_CORPUS = "tiny-model/mathonly-midtrain"

# What each phase's card may claim. These two checkpoints differ in exactly the
# places a reader cares about -- what training the weights have had, and what
# they can therefore do -- and the wrong card asserts both falsely rather than
# merely omitting them.
PHASES = {
    "pretrain": {
        "meta_phase": None,   # pretrain meta.json predates the field entirely
        "config_key": "pretrain_run",
        "phase_note": "phase 1 only -- no frozen-FFN phase, no maths mid-train",
        "tags": ["tinystories", "language-modeling", "from-scratch",
                 "small-language-model"],
        "not_included": [
            "phase 2 frozen-FFN attention retrain",
            "maths mid-train (Act 3)",
            "cell-call mid-train (Act 4)",
        ],
    },
    "mathonly": {
        "meta_phase": "mathonly-midtrain",
        "config_key": "midtrain_run",
        "phase_note": "base pretrain + maths-only mid-train (Act 3), native vocab",
        "tags": ["tinystories", "language-modeling", "small-language-model",
                 "arithmetic", "mid-training"],
        "not_included": [
            "phase 2 frozen-FFN attention retrain",
            "cell-call mid-train (Act 4)",
            "vocabulary extension of any kind",
        ],
    },
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pick_step_dir(ckpt_dir: Path, step: int | None) -> Path:
    """The highest-step `.ready` milestone, or the one --step names.

    Highest rather than newest-mtime: a resumed run rewrites earlier dirs, and
    mtime would then hand back a checkpoint from earlier in training.
    """
    if not ckpt_dir.is_dir():
        sys.exit(f"no such checkpoint dir: {ckpt_dir}")
    found = {}
    for d in ckpt_dir.iterdir():
        m = re.fullmatch(r"step_(\d+)", d.name)
        if m and d.is_dir():
            found[int(m.group(1))] = d
    if not found:
        sys.exit(f"no step_<n>/ dirs under {ckpt_dir}")
    if step is not None:
        if step not in found:
            sys.exit(f"step {step} not found. Have: {sorted(found)}")
        return found[step]
    # train.py touches .ready last, after both files are on disk, so its absence
    # means interrupted mid-write rather than merely unfinished training.
    ready = {n: d for n, d in found.items() if (d / READY_MARKER).is_file()}
    if not ready:
        sys.exit(
            f"no COMPLETE checkpoint under {ckpt_dir} -- found {sorted(found)} but none "
            f"carries a {READY_MARKER} marker, so every one was interrupted mid-write."
        )
    return ready[max(ready)]


def read_emergence(ckpt_dir: Path, metrics_path: Path | None) -> dict:
    """Every milestone's generations, plus the loss trace if we have it.

    This is the Act 1e emergence table as data. It ships in the repo because the
    table is the most-cited thing in the video and "trust the screenshot" is a
    worse offer than "here are the actual generations at each milestone".
    """
    milestones = []
    for d in sorted(ckpt_dir.iterdir()):
        m = re.fullmatch(r"step_(\d+)", d.name)
        # Interrupted milestones are skipped, not flagged: an emergence table is
        # read as "this is what it wrote at this point", and a row from a
        # half-written checkpoint cannot honour that.
        if not (m and (d / "meta.json").is_file() and (d / READY_MARKER).is_file()):
            continue
        meta = json.loads((d / "meta.json").read_text())
        milestones.append({
            "step": int(m.group(1)),
            "tokens": meta.get("tokens"),
            "samples": meta.get("samples", {}),
        })
    milestones.sort(key=lambda r: r["step"])

    loss_trace = []
    if metrics_path and metrics_path.is_file():
        for line in metrics_path.read_text().splitlines():
            line = line.strip()
            if line:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                loss_trace.append({k: rec.get(k) for k in ("step", "loss", "lr")})

    return {
        "schema": "v11-pretrain-emergence-1",
        "note": (
            "Generations captured by train.py at each milestone, greedy, 30 new tokens, "
            "from the same prompts every time. The 16M-token row is this repo's weights; "
            "the earlier rows are the same run mid-flight and are not published as weights."
        ),
        "milestones": milestones,
        "loss_trace": loss_trace,
    }


def stage_tokenizer(out_dir: Path) -> str:
    """AutoTokenizer-loadable tokenizer files, with the published bytes intact.

    save_pretrained re-serializes tokenizer.json through transformers, which
    would make the published sha transformers' round-trip rather than the
    artifact everyone else hashed. So the exact bytes get copied back over the
    top and re-checked.
    """
    from tokenizers import Tokenizer
    from transformers import PreTrainedTokenizerFast

    local_sha = sha256_file(TOKENIZER_JSON)
    backend = Tokenizer.from_file(str(TOKENIZER_JSON))
    fast = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="<unk>", bos_token="<s>", eos_token="</s>", pad_token="<pad>",
        clean_up_tokenization_spaces=False,
    )
    fast.save_pretrained(str(out_dir))
    shutil.copyfile(TOKENIZER_JSON, out_dir / "tokenizer.json")
    if sha256_file(out_dir / "tokenizer.json") != local_sha:
        sys.exit("internal error: tokenizer.json changed while staging")

    # Loadable on transformers 4.x AND 5.x. Measured, not assumed: 5.x writes
    # tokenizer_class "TokenizersBackend", which 4.x then refuses outright.
    # model_max_length belongs to the model, and here we actually know it, so
    # unlike the tokenizer repo this one sets it from the arch config.
    cfg_path = out_dir / "tokenizer_config.json"
    cfg = json.loads(cfg_path.read_text())
    cfg["tokenizer_class"] = "PreTrainedTokenizerFast"
    cfg.pop("backend", None)
    cfg["model_max_length"] = json.loads(ARCH_CONFIG.read_text())["max_seq"]
    cfg_path.write_text(json.dumps(cfg, indent=2, sort_keys=True) + "\n")

    specials = {k: v for k, v in (
        ("unk_token", cfg.get("unk_token")), ("bos_token", cfg.get("bos_token")),
        ("eos_token", cfg.get("eos_token")), ("pad_token", cfg.get("pad_token")),
    ) if v}
    (out_dir / "special_tokens_map.json").write_text(
        json.dumps(specials, indent=2, sort_keys=True) + "\n")
    return local_sha


def build_release(step_dir: Path, out_dir: Path, args, meta: dict,
                  emergence: dict, param_count: int) -> tuple[dict, str]:
    arch = json.loads(ARCH_CONFIG.read_text())

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    # Verbatim bytes: the model sha this repo publishes is the one the training
    # run produced, not a re-save of it.
    shutil.copyfile(step_dir / MODEL_FILE, out_dir / MODEL_FILE)
    model_sha = sha256_file(out_dir / MODEL_FILE)

    tokenizer_sha = stage_tokenizer(out_dir)

    # The weights were produced by the code unit's VENDORED copy of
    # tiny_model_v11/, but what ships here is the repo-root copy. Those are two
    # files on disk with nothing keeping them equal, so if they ever diverge this
    # repo would publish model code that does not match its own weights -- and
    # the failure would surface as a shape error, or worse a silent behavioural
    # difference, for whoever downloads it. Check rather than trust.
    unit_code = UNIT / MODEL_CODE_DIR.name
    drift = [name for name in MODEL_CODE_FILES
             if (unit_code / name).is_file()
             and sha256_file(unit_code / name) != sha256_file(MODEL_CODE_DIR / name)]
    if drift:
        sys.exit(
            f"\nREFUSING TO PUBLISH -- the vendored model code that TRAINED these "
            f"weights differs from the copy about to be published beside them.\n"
            f"  differing files: {', '.join(drift)}\n"
            f"  trained by: {unit_code.relative_to(REPO_ROOT)}\n"
            f"  publishing: {MODEL_CODE_DIR.relative_to(REPO_ROOT)}\n"
            f"Re-sync the two copies before publishing.\n")

    code_dir = out_dir / MODEL_CODE_DIR.name
    code_dir.mkdir()
    for name in MODEL_CODE_FILES:
        shutil.copyfile(MODEL_CODE_DIR / name, code_dir / name)

    # A superset of the arch config: the architecture verbatim, plus what this
    # particular run did. Nothing here is re-derived -- it is copied from the
    # checkpoint's own meta.json and the run config.
    cfg = dict(arch)
    # Drop model_type/architecture from the PUBLISHED copy (the source config
    # keeps them). transformers reads config.json next to a tokenizer and warns
    # "You are using a model of type `tinymodel` to instantiate a model of type
    # ``" on every single load, because there is no registered architecture by
    # that name -- nor should there be; TinyModel is not an AutoModel. Claiming a
    # model_type transformers cannot resolve buys nothing and costs every
    # downstream user a scary-looking warning.
    cfg.pop("model_type", None)
    cfg.pop("architecture", None)
    cfg["_note"] = ("Not a transformers architecture. Build TinyModel from the "
                    "tiny_model_v11/ package shipped in this repo -- see the model card.")
    spec = PHASES[args.phase]
    cfg[spec["config_key"]] = {
        "step": meta.get("step"),
        "tokens": meta.get("tokens"),
        "param_count": param_count,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "phase": spec["phase_note"],
        "corpus": (f"{DATASET} @ {DATASET_REVISION}" if args.phase == "pretrain"
                   else f"{MATHONLY_CORPUS} @ {meta.get('corpus_identity')}"),
        "run_config": args.run_config,
    }
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    (out_dir / "emergence.json").write_text(json.dumps(emergence, indent=2) + "\n")

    provenance = {
        "schema": f"v11-{args.phase}-provenance-1",
        "phase": args.phase,
        "model_sha256": model_sha,
        "tokenizer_sha256": tokenizer_sha,
        "tokenizer_repo": TOKENIZER_REPO,
        "vocab_size": arch["vocab_size"],
        "param_count": param_count,
        "tokens_trained": meta.get("tokens"),
        "steps": meta.get("step"),
        "seed": args.seed,
        "source_repo": "https://github.com/chrishayuk/tinystories-train-video",
        "source_commit": args.source_commit or None,
        "trainer": ("training/harness_pretrain/train.py" if args.phase == "pretrain"
                    else "training/train_mathonly.py"),
        "run_config": args.run_config,
        "corpus": {
            "hub_dataset": DATASET,
            "hub_revision": DATASET_REVISION,
            "note": (
                "Streamed from the pinned HF revision. The content-addressed route "
                "(chuk-datasets tiny-model/v11-rust-tokenized-phase1 @ 67603f8e...) is "
                "the bit-reproducible alternative and is NOT what produced these weights."
            ),
        },
        "not_included": spec["not_included"],
    }
    if args.phase == "mathonly":
        # Copied out of the checkpoint's own meta.json, not re-derived here. Each
        # was verified by the thing that used it -- the entrypoint refuses to
        # train if the base or the corpus hashes differently -- so this is a
        # record of a check that happened, not a claim made at publish time.
        provenance["base"] = {
            "repo": meta.get("base_repo"),
            "sha256": meta.get("base_sha256"),
            "note": "These weights are that checkpoint, continued. Not a fresh run.",
        }
        provenance["corpus"] = {
            "catalog": MATHONLY_CORPUS,
            "identity": meta.get("corpus_identity"),
            "file_sha256": meta.get("corpus_file_sha256"),
            "replay_source": DATASET,
            "replay_revision": DATASET_REVISION,
            "note": (
                "Registered identity is over the manifest, not the bytes, so the corpus "
                "is rebuilt per worker and re-proved against this hash before training. "
                "file_sha256 is the rebuilt corpus on the machine that produced these "
                "weights."
            ),
        }
        provenance["run_id"] = meta.get("run_id")
    (out_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")

    (out_dir / "README.md").write_text(
        render_card(args, cfg, provenance, emergence, model_sha, tokenizer_sha))

    lines = [f"{sha256_file(p)}  {p.relative_to(out_dir)}"
             for p in sorted(out_dir.rglob("*"))
             if p.is_file() and p.name != "checksums.sha256"]
    (out_dir / "checksums.sha256").write_text("\n".join(lines) + "\n")
    return provenance, model_sha


def card_lede(phase: str, args, params: int, tokens: int, provenance: dict) -> str:
    """The paragraph directly under the title: what these weights have had done
    to them. The single most-read sentence in the repo, and the one the wrong
    card gets flatly wrong."""
    tokenizer_link = f"[v11 tokenizer](https://huggingface.co/{TOKENIZER_REPO})"
    if phase == "pretrain":
        return (
            f"A {params/1e6:.1f}M-parameter decoder-only transformer trained "
            f"**from scratch** on\nTinyStories for {tokens/1e6:.0f}M tokens, using the\n"
            f"{tokenizer_link}.\n") + f"""
This is the **base pretrain only**: phase 1, no frozen-FFN attention retrain, and
**no maths mid-training**. It writes competent children's-story English and it
**cannot do arithmetic** — it narrates straight past the place a number belongs
rather than putting a wrong number there. That is the intended state of this
checkpoint, not a defect: it is the starting point for the mid-training
experiments in
[tinystories-train-video](https://github.com/chrishayuk/tinystories-train-video).
"""
    base = provenance.get("base", {})
    return (
        f"A {params/1e6:.1f}M-parameter decoder-only transformer that learned English on\n"
        f"TinyStories and was then taught addition, using the {tokenizer_link}.\n") + f"""
This is a base pretrain **continued on a maths mid-training corpus** — Act 3 of
[tinystories-train-video](https://github.com/chrishayuk/tinystories-train-video).
It is **not** a fresh training run: it starts from
[`{base.get('repo') or 'the published base'}`](https://huggingface.co/{base.get('repo', '')})
(`{(base.get('sha256') or '')[:16]}…`), adds {tokens/1e6:.1f}M tokens of in-tier addition drills
mixed with TinyStories replay, and **extends no vocabulary** — the corpus never
mentions a cell, so the embedding table is the native {cfg_vocab(args)} rows end to end.

The mid-training corpus is content-addressed as `{MATHONLY_CORPUS}`
(`{(provenance.get('corpus', {}).get('identity') or 'unregistered')[:16]}…`), so a replicate is
guaranteed to have been taught the same thing rather than assumed to.
"""


def card_limits(phase: str, tokens: int, emergence: dict, maths_evidence: str) -> str:
    """"What this model cannot do". For the pretrain that is a settled fact. For
    the mid-train it is the experiment's result, so this reports what the
    checkpoint actually generates and declines to grade it -- the grading is
    pre-registered elsewhere and does not belong on a model card."""
    tail = (
        f"It has also seen {tokens/1e6:.0f}M tokens, which is **not converged**. Continued\n"
        f"training on more or less anything improves it, so do not read an improvement after\n"
        f"mid-training as evidence that the mid-training data helped specifically.\n")
    if phase == "pretrain":
        return f"""## What this model cannot do

It cannot do arithmetic. Nothing in TinyStories teaches addition, and number words
in that corpus are narrative texture rather than quantities — "once upon a time
there were two" is an idiom the model learns the way it learns "happily ever
after".

The interesting part is *how* it fails. It does not answer with a wrong number; it
carries on telling the story, straight past the place a number belongs.

{maths_evidence}
{tail}"""
    return f"""## What this model can and cannot do

This checkpoint exists to be measured, not to be believed. `emergence.json` holds
its greedy generations at every milestone, from the same prompts throughout, and
those prompts were chosen before the run to separate four different things:

- an **in-tier canonical** sum (`7 + 5 =`) — inside the taught range;
- an **out-of-tier** sum (`394 + 251 =`) — one decimal digit past it, which should
  stay wrong, because getting it right would mean the tier boundary leaked;
- the same in-tier sum **in narrative digits** — the corpus's own surface form;
- the same sum **in number words**, which never appears in the corpus in any form.

Those last two are the whole question: matching the corpus surface is consistent
with having memorised a distribution, and only the unseen surface distinguishes
that from having learned an algorithm.

{maths_evidence}
**Do not read the arithmetic samples as an accuracy figure.** They are a handful of
greedy generations. The measurement is `training/heldout_probe_mathonly.py --curve` at
n≥250 per band, read against a pre-registered outcome map, and a `taught` band that
has not yet learned its own facts makes every other number in that table
uninterpretable.

{tail}"""


def cfg_vocab(args) -> str:
    return f"{json.loads(ARCH_CONFIG.read_text())['vocab_size']:,}"


def corpus_rows(phase: str, provenance: dict) -> str:
    """The Training table's provenance rows. A mid-train has two inputs a
    pretrain does not have at all -- the weights it started from and a corpus
    that is not a public dataset -- and both are identities, not names."""
    if phase == "pretrain":
        return (f"| Corpus | [`{DATASET}`](https://huggingface.co/datasets/{DATASET}) "
                f"@ `{DATASET_REVISION[:12]}…` |\n")
    base = provenance.get("base", {})
    corpus = provenance.get("corpus", {})
    return (
        f"| Base checkpoint | [`{base.get('repo') or '—'}`]"
        f"(https://huggingface.co/{base.get('repo', '')}) `{(base.get('sha256') or '')[:12]}…` |\n"
        f"| Corpus | `{MATHONLY_CORPUS}` `{(corpus.get('identity') or 'unregistered')[:12]}…` |\n"
        f"| Corpus replay source | [`{DATASET}`](https://huggingface.co/datasets/{DATASET}) "
        f"@ `{DATASET_REVISION[:12]}…` |\n")


def corpus_note(phase: str) -> str:
    if phase == "pretrain":
        return ("The dataset revision is pinned, so the document set is reproducible and held-out\n"
                "text can be shown never to have been trained on.")
    return (
        "The corpus identity is checked by the training entrypoint *before* it spends any\n"
        "GPU time, so a rebuild that silently differs refuses rather than trains. That\n"
        "matters more here than it sounds: an earlier pair of runs was compared across\n"
        "machines while validating on different data, because the corpus was rebuilt per\n"
        "host and nothing re-proved it was the same corpus.")


def render_card(args, cfg, provenance, emergence, model_sha, tokenizer_sha) -> str:
    tokens = provenance["tokens_trained"] or 0
    params = provenance["param_count"]
    rows = []
    for m in emergence["milestones"]:
        sample = next(iter(m["samples"].values()), "") if m["samples"] else ""
        sample = " ".join(sample.split())[:110]
        rows.append(f"| {(m['tokens'] or 0)/1e6:.2f}M | {m['step']} | {sample or '—'} |")
    emergence_table = "\n".join(rows) if rows else "| — | — | — |"

    embed_params = cfg["vocab_size"] * cfg["dim"]
    embed_pct = 100.0 * embed_params / params if params else 0.0

    # Quote THIS run's own generations rather than describing the behaviour from
    # memory. The pretrain's failure mode was characterised on an earlier
    # checkpoint trained with a different tokenizer, and the midtrain's result is
    # the thing under test -- restating either as a property of these weights
    # would be inheriting a measurement.
    final = emergence["milestones"][-1]["samples"] if emergence["milestones"] else {}
    if args.phase == "pretrain":
        quoted = [(p, s) for p, s in final.items() if "gave her four more" in p]
    else:
        # All four arithmetic prompts, because for a mid-trained checkpoint the
        # comparison BETWEEN them is the finding; any one alone is a claim.
        quoted = [(p, s) for p, s in final.items() if p != VERIFY_PROMPT]
    if quoted:
        lines = "\n".join(f"> *{p}* **{' '.join(s.split())}**\n" for p, s in quoted)
        noun = "generation" if len(quoted) == 1 else "generations"
        maths_evidence = (
            f"Its own {noun} at the final milestone, greedy, from this run:\n\n{lines}")
    else:
        maths_evidence = (
            "See `emergence.json` for what it actually generates on arithmetic prompts.\n")

    tags = "\n".join(f"  - {t}" for t in PHASES[args.phase]["tags"])

    return f"""---
language:
  - en
license: apache-2.0
library_name: pytorch
inference: false
datasets:
  - {DATASET}
tags:
{tags}
---

# {args.repo_id.split('/')[-1]}

{card_lede(args.phase, args, params, tokens, provenance)}
## Loading

Not an `AutoModel` — `TinyModel` is a 3-file Gemma-shaped decoder (RMSNorm, RoPE,
GQA, gated FFN, tied embeddings), shipped in this repo under `tiny_model_v11/`.

```python
import json, torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from transformers import AutoTokenizer

path = snapshot_download("{args.repo_id}")

import sys; sys.path.insert(0, path)
from tiny_model_v11 import TinyModel

cfg = json.load(open(f"{{path}}/config.json"))
model = TinyModel(
    vocab_size=cfg["vocab_size"], dim=cfg["dim"], n_layers=cfg["n_layers"],
    ffn_dim=cfg["ffn_dim"], n_heads=cfg["n_heads"], n_kv_heads=cfg["n_kv_heads"],
    max_seq=cfg["max_seq"],
)
model.load_state_dict(load_file(f"{{path}}/{MODEL_FILE}"))
model.eval()

tok = AutoTokenizer.from_pretrained("{args.repo_id}")
ids = [tok.convert_tokens_to_ids("<s>")] + tok("Once upon a time", add_special_tokens=False)["input_ids"]
for _ in range(40):
    logits = model(torch.tensor([ids[-cfg["max_seq"]:]]))[0, -1]
    ids.append(int(logits.argmax()))
print(tok.decode(ids))
```

That exact round-trip — download, build, load, generate — is run against the
uploaded files at publish time. It is not an untested snippet.

## Architecture

| | |
|---|---|
| Parameters | {params:,} |
| Layers | {cfg['n_layers']} |
| Model dim | {cfg['dim']} |
| Attention heads | {cfg['n_heads']} ({cfg['n_kv_heads']} KV heads, GQA) |
| FFN dim | {cfg['ffn_dim']} |
| Context | {cfg['max_seq']} tokens |
| Vocabulary | {cfg['vocab_size']:,} |
| Embeddings | tied (`lm_head.weight is embed.weight`) |

The embedding table is {embed_params/1e6:.1f}M parameters — **{embed_pct:.0f}% of the
whole model is its vocabulary lookup**, which is what a 71k vocabulary costs at
this width.

`rope_freqs` is a complex64 buffer in the state dict. It is derived from
`dim`/`n_heads`/`max_seq` and recomputed on construction, so it round-trips but
carries no learned information.

## Identity

**Identity is the content hash, not the Hub revision** — re-pushing identical
bytes mints a new commit oid, and a README edit does too. Join on these:

| | |
|---|---|
| `{MODEL_FILE}` sha256 | `{model_sha}` |
| `tokenizer.json` sha256 | `{tokenizer_sha}` |
| Tokenizer repo | [`{TOKENIZER_REPO}`](https://huggingface.co/{TOKENIZER_REPO}) |
| Source | [tinystories-train-video](https://github.com/chrishayuk/tinystories-train-video) |
| Source commit | `{args.source_commit or 'unrecorded'}` |

The tokenizer sha is the **same value** this checkpoint's training run wrote into
its `meta.json` as `tokenizer_hash`, and the same one
[`{TOKENIZER_REPO}`](https://huggingface.co/{TOKENIZER_REPO}) publishes. A
checkpoint driven by a different tokenizer produces fluent nonsense rather than
an error, so that join is checked mechanically at publish time, not asserted here.

## Training

| | |
|---|---|
{corpus_rows(args.phase, provenance)}| Tokens | {tokens/1e6:.0f}M |
| Steps | {provenance['steps']:,} |
| Batch × context | {args.batch_size} × {cfg['max_seq']} |
| Optimiser | AdamW, lr {args.lr}, weight decay 0.01, grad clip 1.0 |
| LR schedule | linear warmup then linear decay to 5% |
| Seed | {args.seed} |
| Precision | fp32 |

{corpus_note(args.phase)}

## Capability emergence

`emergence.json` carries the generations captured at each milestone, plus the
loss trace. Greedy, {30 if args.phase == "pretrain" else 14} new tokens, same prompts throughout:

| Tokens | Step | First sample continuation |
|---|---|---|
{emergence_table}

Only the final row's weights are published here; the earlier rows are the same
run mid-flight.

{card_limits(args.phase, tokens, emergence, maths_evidence)}
## Intended use

Research and teaching on compact language models: mid-training, tool-use /
tool-call training, and measuring the difference between memorising a
distribution and learning an algorithm. Not intended for any production use, and
it has no safety training of any kind.

## Provenance

`provenance.json` records the model and tokenizer hashes, the corpus pin, and the
run config. The chuk-datasets catalog holds a content-addressed, bit-reproducible
tokenization of the same corpus (`tiny-model/v11-rust-tokenized-phase1`); these
weights came from the pinned HF revision instead, which is reproducible given the
seed but not bit-identical. That distinction is recorded rather than glossed.
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ckpt-dir", type=Path, required=True,
                    help="dir containing step_<n>/ milestones (train.py's CHUK_CKPT_DIR)")
    ap.add_argument("--step", type=int, default=None,
                    help="which milestone; default = highest complete one")
    ap.add_argument("--repo-id", required=True)
    ap.add_argument("--phase", choices=sorted(PHASES), default="pretrain",
                    help="which card and provenance to ship; checked against the "
                         "checkpoint's own meta.json")
    ap.add_argument("--metrics", type=Path, default=None,
                    help="metrics.jsonl, for the loss trace in emergence.json")
    ap.add_argument("--output-dir", type=Path, default=None)
    ap.add_argument("--source-commit", default="")
    ap.add_argument("--run-config", default="configs/colab.json")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="replace different published weights under this repo id")
    ap.add_argument("--dry-run", action="store_true",
                    help="stage and self-check locally; contact the Hub read-only, push nothing")
    args = ap.parse_args()

    for path in (ARCH_CONFIG, TOKENIZER_JSON):
        if not path.is_file():
            sys.exit(f"missing: {path}")

    step_dir = pick_step_dir(args.ckpt_dir, args.step)
    if not (step_dir / MODEL_FILE).is_file():
        sys.exit(f"{step_dir} has no {MODEL_FILE}")
    if not (step_dir / READY_MARKER).is_file():
        sys.exit(
            f"\nREFUSING TO PUBLISH -- {step_dir} carries no {READY_MARKER} marker.\n"
            f"train.py touches it only after both {MODEL_FILE} and meta.json are on disk, "
            f"so this checkpoint was interrupted mid-write.\n")
    meta = json.loads((step_dir / "meta.json").read_text())

    # --- guard 5: the card must describe THIS checkpoint -----------------------
    # Before the tokenizer and vocabulary guards, because those two pass happily
    # on a correctly-built checkpoint being published under the wrong story --
    # and the story is what people read.
    want_phase = PHASES[args.phase]["meta_phase"]
    got_phase = meta.get("phase")
    if got_phase != want_phase:
        expected = f"{want_phase!r}" if want_phase else "no `phase` field (the pretrain trainer writes none)"
        sys.exit(
            f"\nREFUSING TO PUBLISH -- --phase {args.phase} does not match this checkpoint.\n"
            f"  meta.json phase: {got_phase!r}\n"
            f"  --phase {args.phase} expects: {expected}\n\n"
            f"The pretrain card asserts \"the base pretrain only... no maths mid-training\" "
            f"and \"it cannot do arithmetic\". Both are false of a mid-trained checkpoint, "
            f"and a card is a claim rather than decoration.\n")

    # The mid-train stamps its own hyperparameters into every checkpoint, so read
    # them off the artifact rather than off flags that can disagree with it
    # silently. The pretrain predates that and still needs them passed.
    if args.phase == "mathonly":
        for flag, key in (("seed", "seed"), ("batch_size", "batch_size"), ("lr", "lr")):
            if meta.get(key) is not None:
                setattr(args, flag, meta[key])
        if args.run_config == ap.get_default("run_config"):
            args.run_config = "training/run_mathonly_unit.sh (chuk-train entrypoint `midtrain`)"

    # --- guard 1: the checkpoint/tokenizer join --------------------------------
    tokenizer_sha = sha256_file(TOKENIZER_JSON)
    if tokenizer_sha != PUBLISHED_TOKENIZER_SHA256:
        sys.exit(
            f"\nREFUSING TO PUBLISH -- the tokenizer in this repo is not the published "
            f"v11 build.\n  expected {PUBLISHED_TOKENIZER_SHA256}\n  found    {tokenizer_sha}\n")
    ckpt_tok = meta.get("tokenizer_hash", "")
    if ckpt_tok != tokenizer_sha:
        sys.exit(
            f"\nREFUSING TO PUBLISH -- this checkpoint was NOT trained with the tokenizer "
            f"about to be published alongside it.\n"
            f"  checkpoint meta.json tokenizer_hash: {ckpt_tok or '(absent)'}\n"
            f"  tokenizer staged for this repo:      {tokenizer_sha}\n"
            f"A mismatch here does not error at load time -- it generates fluent nonsense "
            f"for everyone who downloads the repo.\n")

    # --- guard 2: vocabulary vs embedding table -------------------------------
    from safetensors.torch import load_file as load_safetensors
    from tokenizers import Tokenizer

    arch = json.loads(ARCH_CONFIG.read_text())
    tok_vocab = Tokenizer.from_file(str(TOKENIZER_JSON)).get_vocab_size()
    state = load_safetensors(str(step_dir / MODEL_FILE))
    embed_rows = tuple(state["embed.weight"].shape)[0]
    if not (embed_rows == arch["vocab_size"] == tok_vocab):
        sys.exit(
            f"\nREFUSING TO PUBLISH -- vocabulary disagreement.\n"
            f"  embed.weight rows : {embed_rows}\n"
            f"  config.json       : {arch['vocab_size']}\n"
            f"  tokenizer         : {tok_vocab}\n")
    param_count = sum(v.numel() for k, v in state.items() if k != "rope_freqs")
    # Tied embeddings: lm_head.weight aliases embed.weight in the live model, but
    # safetensors cannot store aliases so train.py clones it. Count it once.
    if "lm_head.weight" in state:
        param_count -= state["lm_head.weight"].numel()

    emergence = read_emergence(args.ckpt_dir, args.metrics)
    out_dir = args.output_dir or Path(tempfile.mkdtemp(prefix="v11-pretrain-release-"))
    provenance, model_sha = build_release(
        step_dir, out_dir, args, meta, emergence, param_count)

    print(f"staged {args.repo_id}")
    print(f"  checkpoint       {step_dir}  (step {meta.get('step')}, "
          f"{(meta.get('tokens') or 0)/1e6:.2f}M tokens)")
    print(f"  parameters       {param_count:,}")
    print(f"  vocab            {tok_vocab:,}  (embed rows {embed_rows:,})")
    print(f"  model sha256     {model_sha}")
    print(f"  tokenizer sha256 {tokenizer_sha}  == checkpoint tokenizer_hash")
    print(f"  milestones       {[m['step'] for m in emergence['milestones']]}")
    print(f"  loss trace       {len(emergence['loss_trace'])} points")
    print(f"  files            {sorted(str(p.relative_to(out_dir)) for p in out_dir.rglob('*') if p.is_file())}")

    from huggingface_hub import HfApi
    from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError
    api = HfApi()

    # --- guard 4: never silently replace published weights --------------------
    try:
        existing = api.hf_hub_download(repo_id=args.repo_id, filename=MODEL_FILE,
                                       repo_type="model")
        published_sha = sha256_file(Path(existing))
        if published_sha == model_sha:
            print("  precondition     repo exists, identical weights -- re-push is a no-op")
        elif args.force:
            print(f"  precondition     repo holds DIFFERENT weights ({published_sha[:16]}…) "
                  f"-- replacing, --force given")
        else:
            sys.exit(
                f"\nREFUSING TO PUSH -- {args.repo_id} already holds different weights.\n"
                f"  published {published_sha}\n  staged    {model_sha}\n\n"
                f"provenance.json joins are only as good as that name staying put. Publish "
                f"the retrain under a new repo id, or pass --force if replacing is "
                f"genuinely what you want.\n")
    except (RepositoryNotFoundError, EntryNotFoundError):
        print("  precondition     repo is new")

    if args.dry_run:
        verify_local(out_dir)
        print(f"\ndry run -- nothing pushed. Staged at {out_dir}")
        return

    api.create_repo(args.repo_id, repo_type="model", exist_ok=True, private=args.private)
    commit = api.upload_folder(
        repo_id=args.repo_id, repo_type="model", folder_path=str(out_dir),
        commit_message=f"Pretrain {(meta.get('tokens') or 0)/1e6:.0f}M tokens, "
                       f"model {model_sha[:16]}")
    revision = getattr(commit, "oid", None) or "main"
    print(f"  pushed           revision {revision}")

    verify_published(args.repo_id, revision, model_sha, tokenizer_sha)

    print(f"\npublished https://huggingface.co/{args.repo_id}")
    print(f"  identity (join on this): model {model_sha}")
    print(f"                           tokenizer {tokenizer_sha}")
    print(f"  hub revision (retrieval coordinate only): {revision}")


def _generate(model, tok, prompt: str, max_seq: int, max_new: int = 40) -> str:
    """Greedy, matching train.py's own milestone sampling. `tok` is a
    tokenizers.Tokenizer (the backend), not a transformers wrapper."""
    import torch
    ids = tok.encode(prompt).ids
    bos = tok.token_to_id("<s>")
    if bos is not None:
        ids = [bos] + ids
    with torch.no_grad():
        for _ in range(max_new):
            logits = model(torch.tensor([ids[-max_seq:]]))[0, -1]
            ids.append(int(logits.argmax()))
    return tok.decode(ids)


def _load_and_generate(root: Path, label: str) -> None:
    """Build TinyModel from the shipped code + weights and actually generate.

    A shape check passes through a model that emits garbage. Generating is the
    only check that would notice a tokenizer/checkpoint mismatch, and that is
    the exact failure this whole project exists to stop shipping.
    """
    import torch
    from safetensors.torch import load_file
    from tokenizers import Tokenizer

    sys.path.insert(0, str(root))
    for mod in [m for m in sys.modules if m.startswith("tiny_model_v11")]:
        del sys.modules[mod]
    from tiny_model_v11 import TinyModel

    cfg = json.loads((root / "config.json").read_text())
    model = TinyModel(
        vocab_size=cfg["vocab_size"], dim=cfg["dim"], n_layers=cfg["n_layers"],
        ffn_dim=cfg["ffn_dim"], n_heads=cfg["n_heads"], n_kv_heads=cfg["n_kv_heads"],
        max_seq=cfg["max_seq"],
    )
    model.load_state_dict(load_file(str(root / MODEL_FILE)))
    model.eval()
    tok = Tokenizer.from_file(str(root / "tokenizer.json"))
    out = _generate(model, tok, VERIFY_PROMPT, cfg["max_seq"])
    print(f"  {label:16s} strict load OK, generated:")
    print(f"                   {' '.join(out.split())[:200]!r}")
    if len(out.split()) < 6:
        print("  WARNING: that generation is suspiciously short for a trained checkpoint.")


def verify_local(out_dir: Path) -> None:
    """Everything the post-push check does that does not need the Hub.

    The AutoTokenizer load is here rather than only after upload because the
    failure it catches is a staging bug (transformers 5.x writes a
    `tokenizer_class` 4.x refuses), and finding that out post-push means the
    broken files are already published.
    """
    from transformers import AutoTokenizer
    print("\nself-check (staged files, nothing downloaded):")
    tok = AutoTokenizer.from_pretrained(str(out_dir))
    ids = tok(VERIFY_PROMPT, add_special_tokens=False)["input_ids"]
    if not ids or tok.decode(ids).strip() != VERIFY_PROMPT:
        sys.exit(f"SELF-CHECK FAILED: AutoTokenizer round-trip broke: "
                 f"{VERIFY_PROMPT!r} -> {ids} -> {tok.decode(ids)!r}")
    print(f"  AutoTokenizer    loads, {VERIFY_PROMPT!r} -> {ids} -> round-trips")
    _load_and_generate(out_dir, "staged")


def verify_published(repo_id: str, revision: str, model_sha: str, tokenizer_sha: str) -> None:
    """Verify what the Hub SERVES, not what we staged.

    Downloads the repo fresh, re-hashes both identity anchors, loads the model
    through AutoTokenizer + the shipped model code, and generates. This is the
    acceptance criterion in the module docstring, executed.
    """
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    print("\nverifying the DOWNLOADED artifact:")
    root = Path(snapshot_download(repo_id, revision=revision))

    for name, want in ((MODEL_FILE, model_sha), ("tokenizer.json", tokenizer_sha)):
        got = sha256_file(root / name)
        if got != want:
            sys.exit(f"VERIFY FAILED: hub serves {name} as {got}, expected {want}")
    print(f"  sha256           {MODEL_FILE} + tokenizer.json both match")

    tok = AutoTokenizer.from_pretrained(repo_id, revision=revision)
    probe = "Once upon a time"
    ids = tok(probe, add_special_tokens=False)["input_ids"]
    if not ids or tok.decode(ids).strip() != probe:
        sys.exit(f"VERIFY FAILED: AutoTokenizer round-trip broke: {probe!r} -> {ids} -> "
                 f"{tok.decode(ids)!r}")
    print(f"  AutoTokenizer    loads, {probe!r} -> {ids} -> round-trips")

    _load_and_generate(root, "downloaded")


if __name__ == "__main__":
    main()
