#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["torch>=2.2", "tokenizers>=0.20", "safetensors>=0.4", "pyarrow>=14", "numpy"]
# ///
"""Interactive REPL for TinyModel v11 — for typing prompts live on camera.

  uv run repl.py

Type a prompt, press enter, watch it generate token by token. Nothing is
pre-canned; every word on screen is produced live.

This is the ONLY tool the live shoot uses (SCRIPT.md § ONE REPL): the whole
session runs in here rather than cutting out to `bat` and one-shot scripts.
Hence /config, /params, /data and /loop, which cover Act 1a-1d.

Commands (type these instead of a prompt):
  /config          the architecture config — Act 1a
  /params          where the 115.1M parameters go — Act 1a
  /data [--tokens] TinyStories at the pinned revision — Act 1c
  /loop            the training loop itself — Act 1d
  /next <prompt>   top-10 next-word predictions instead of generating
  /slots           all four Act 2a number slots at once, with the summary
  /greedy          always take the most likely token (deterministic)
  /sample          sample with temperature (default)
  /temp 0.8        set sampling temperature
  /len 60          set max tokens to generate
  /full            switch to model_full.pt   (after phase 1/2, 16M tokens) — default
  /compiled        switch to model_compiled.pt (after phase 3, frozen FFN; not run yet)
  /mathonly        switch to model_mathonly.pt (Act 3: maths mid-trained, no cells)
  /broker          let the model call Z80 cells — Act 4
  /slow            add a delay per token, for camera pacing
  /fast            no delay (default)
  /help  /quit

Ctrl-C stops a generation without leaving the REPL.

Uses the PUBLISHED v11 tokenizer (vocab 71260) and refuses to run against any
checkpoint built on a different vocabulary.

NO CHECKPOINT IS LOADED UNTIL YOU GENERATE. That is deliberate, and not just an
optimisation: Acts 1a-1d happen *before the model exists* in the video's own
story, and a REPL that had to load trained weights before it could print the
config would be quietly admitting the ending. /config, /params, /data and /loop
all run against a repo with no checkpoint in it at all.
"""

import json
import os
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from demo_common import (
    ARCH_CONFIG, ARTEFACTS, BOLD, DIM, GREEN, NUMBER_WORDS, RESET, TOKENIZER,
    TRAIN_PY, V11Tokenizer, check_vocab, missing_checkpoint_message,
)


# Act 2a. Slots where a NUMBER is the only sensible continuation -- the point
# being what the model ranks, not what it generates. Kept next to the code that
# uses them so the figures the script quotes are re-derivable from one place.
#
# Ordered as a ladder from pure idiom down to arithmetic, because the *gradient*
# is the finding: this model is not blind to numbers, it is fluent in the shapes
# they appear in and cannot combine them. Chosen by measuring ~20 candidates on
# the published checkpoint (2026-07-25) and keeping the ones that separate.
#
# Row 2 is the one that matters. It scores 91.8% with the count sequence ranked
# correctly -- three > four > five > six -- so the distribution looks like a model
# that can count. Then generate from it and it emits "three, four, four, five,
# five, five..." forever, because each step is a local next-token guess with no
# state, and TinyStories almost never counts past five. Metric says competent,
# behaviour says otherwise: Act 3c's whole argument, two acts early.
SLOT_PROMPTS = [
    ("Once upon a time there were", "story idiom — no counting required"),
    ("Lily counted her toys. One, two,", "reciting a sequence — LOOKS like counting"),
    ("She counted the apples. There were", "counting context, no arithmetic"),
    ("There were five ducks. Two swam away. Now there are",
     "needs 5 − 2 = three — watch it echo 'two' from the prompt"),
    ("Anna is four years old. Next year she will be", "needs 4 + 1 = five"),
]


def no_model_yet(path):
    """Print and carry on -- this is a REPL, and /config, /params, /data and
    /loop all still work with no weights anywhere."""
    print(missing_checkpoint_message(path))
    print(f"  {DIM}/config, /params, /data and /loop need no checkpoint "
          f"and work now.{RESET}\n")
    return False


try:
    import readline  # noqa: F401  — arrow keys and history in input()
except ImportError:
    pass


# The cell's return is printed in a DIFFERENT colour from the model's tokens, and
# labelled. That distinction IS Act 4's argument: if the answer looks like model
# output, the act has no point.
CYAN = "\033[96m"

# The cells checkpoint lives outside model_v11/artifacts/ because it has its own
# vocabulary -- 72,052 rows against config.json's 71,260 -- so load_from_artifacts
# cannot build it and export_repl_checkpoint.py rightly refuses it.
CELLS_CKPT = HERE / "runs" / "cells-s80" / "ckpt" / "step_12504"
CELLS_TOKEN_MAP = HERE / "training" / "data" / "cells" / "cells_token_map.json"
CELL_SOURCES = Path.home() / "chris-source" / "cell80" / "cell80" / "cells"


def find_cell80():
    """cell80_py lives only in the uv cache -- there is no installable pin.

    Try the import first in case PYTHONPATH is already set, then walk the cache.
    Most cached builds are too old and reject safe_div.rs's `a / b`, so a build
    that imports is not necessarily one that works; the compile error surfaces at
    add_source() time with a clear message, which is soon enough.
    """
    try:
        import cell80_py
        return cell80_py
    except ImportError:
        pass
    cache = Path.home() / ".cache" / "uv" / "archive-v0"
    for d in sorted(cache.glob("*/cell80_py"), reverse=True):
        sys.path.insert(0, str(d.parent))
        try:
            import cell80_py
            return cell80_py
        except ImportError:
            sys.path.pop(0)
    return None


class Session:
    def __init__(self):
        self.temperature = 0.8
        self.top_k = 40
        self.greedy = False
        self.max_new = 60
        self.delay = 0.0
        # Broker mode: the cells checkpoint plus a live Z80. Off until /broker.
        self.broker = False
        self.cells = None
        # phase 1 of the Act 1e lineage. NOT model_compiled.pt: that is the
        # phase-3 frozen-FFN checkpoint, and phase 3 has not been run on the
        # published tokenizer yet.
        self.checkpoint = "model_full.pt"
        self.sp = None
        self.model = None
        self.config = None

    def tokenizer(self):
        """The tokenizer alone — no weights. /data needs it; /config and
        /params do not need either."""
        if self.sp is None:
            self.sp = V11Tokenizer(TOKENIZER)
        return self.sp

    def ensure_model(self):
        """Load on first use rather than at startup. Returns False if there is
        no checkpoint, having already said so."""
        if self.model is not None:
            return True
        return self.load()

    def load(self, checkpoint=None):
        from tiny_model_v11 import load_from_artifacts

        # Don't commit self.checkpoint until the load succeeds. A failed switch
        # used to leave the session pointing at a checkpoint that isn't there,
        # so the next prompt failed too and the REPL looked broken rather than
        # just "that one doesn't exist yet".
        want = checkpoint or self.checkpoint
        self.tokenizer()

        path = ARTEFACTS / "artifacts" / want
        if not path.exists():
            return no_model_yet(path)
        self.checkpoint = want

        print(f"{DIM}loading {self.checkpoint} …{RESET}", end=" ", flush=True)
        t0 = time.time()
        self.model, self.config = load_from_artifacts(
            ARTEFACTS, checkpoint=self.checkpoint)
        print(f"{DIM}{time.time()-t0:.1f}s{RESET}")
        check_vocab(self.config, self.sp, self.checkpoint)
        # LEAVING BROKER MODE IS PART OF LOADING SOMETHING ELSE. Without this,
        # /mathonly swaps the weights but leaves self.broker set, and the next
        # /broker sees the flag, returns early and never reloads -- so Act 4 runs
        # the MATHS checkpoint in broker mode and emits no call at all. The shoot
        # order goes /broker (cold open) → /mathonly (3b) → /broker (4c), so this
        # is the normal path through the video, not a corner.
        self.broker = False
        n = sum(p.numel() for p in self.model.parameters())
        print(f"{DIM}  {n/1e6:.1f}M params · {self.config.n_layers} layers · "
              f"dim {self.config.dim} · vocab {self.config.vocab_size:,} · "
              f"{self.device}{RESET}")
        return True

    def load_broker(self):
        """Switch to the cells checkpoint and wake a Z80. Returns False on failure.

        Three steps rather than a load, and all three are forced by the same fact:
        this is the one model in the project with an extended vocabulary. Build at
        the base 71,260, grow the tied embedding to 72,052, then load the weights.
        `train_cells.py` does exactly this on the worker.
        """
        # Already in broker mode with the cells weights actually loaded.
        if self.broker and self.model is not None:
            return True
        # Coming back after /full or /mathonly: the Z80 host and its compiled
        # cells survive the switch, so only the weights need reloading. Keeping
        # the host matters -- compiling a cell is the slow part, and the second
        # /broker in the running order is thirty seconds from the closing shot.
        reentry = self.cells is not None
        if not CELLS_CKPT.exists():
            print(f"\n  {BOLD}No cells checkpoint.{RESET}\n")
            print(f"  {DIM}Expected {CELLS_CKPT.relative_to(HERE)}.")
            print(f"  Pull it from the control plane, or run Act 4b's training"
                  f" arm.{RESET}\n")
            return False
        cell80 = find_cell80()
        if cell80 is None:
            print(f"\n  {BOLD}cell80_py not found.{RESET}\n")
            print(f"  {DIM}It exists only in the uv cache — there is no installable")
            print(f"  pin. Set PYTHONPATH to a build that compiles safe_div.rs;")
            print(f"  run_broker.sh carries the recipe for finding one.{RESET}\n")
            return False

        import json
        from safetensors.torch import load_file
        from tiny_model_v11 import load_from_artifacts
        sys.path.insert(0, str(HERE / "training"))
        from train_cells import resize_embedding

        tmap = json.loads(CELLS_TOKEN_MAP.read_text())
        self.tokenizer()
        print(f"{DIM}loading cells checkpoint …{RESET}", end=" ", flush=True)
        t0 = time.time()
        model, cfg = load_from_artifacts(ARTEFACTS, checkpoint="model_full.pt",
                                         device="cpu")
        resize_embedding(model, tmap["extended_vocab"])
        state = load_file(str(CELLS_CKPT / "model.safetensors"))
        model.load_state_dict(state, strict=True)
        device = ("mps" if torch.backends.mps.is_available()
                  else "cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.config = model.to(device).eval(), cfg
        print(f"{DIM}{time.time()-t0:.1f}s{RESET}")

        if not reentry:
            self.cells = {
                "call": tmap["call"], "close": tmap["close"],
                "inv": {v: k for k, v in tmap["cells"].items()},
                "host": cell80.CellHost(), "handles": {},
            }
        self.broker = True
        self.checkpoint = "cells (broker)"
        print(f"{DIM}  vocab {tmap['base_vocab']:,} → "
              f"{tmap['extended_vocab']:,} · Z80 ready · "
              f"model tokens in {GREEN}green{RESET}{DIM}, "
              f"cell returns in {CYAN}cyan{RESET}{DIM}{RESET}\n")
        return True

    def run_cell(self, span):
        """Execute the call this </call> just closed. Returns its result or None."""
        c = self.cells
        toks = [x for x in span if x in c["inv"]]
        if not toks:
            return None
        name = c["inv"][toks[0]]
        digits = [x for x in span if x < self.sp.get_piece_size()]
        args = [int(s) for s in self.sp.decode(digits).split()
                if s.lstrip("-").isdigit()]
        if name not in c["handles"]:
            src = next(CELL_SOURCES.rglob(f"{name}.rs"), None)
            if src is None:
                return None
            c["host"].add_source(name, src.read_text())
            c["handles"][name] = c["host"].load(name)
        r = c["host"].run(c["handles"][name], args)
        res = r["result"] if r.get("halt") == "returned" else None
        print(f"\n  {CYAN}[cell]{RESET} {name}({', '.join(map(str, args))}) "
              f"{DIM}→{RESET} {BOLD}{res}{RESET}", flush=True)
        return res

    @property
    def device(self):
        return next(self.model.parameters()).device

    def encode(self, text):
        ids = self.sp.encode(text)
        if self.sp.bos_id() >= 0:
            ids = [self.sp.bos_id()] + ids
        return ids

    @torch.no_grad()
    def stream(self, prompt):
        """Generate, printing each token as it arrives."""
        max_seq = self.model.rope_freqs.shape[0]
        ids = self.encode(prompt)
        shown = self.sp.decode(ids)

        print(f"  {prompt}", end="", flush=True)
        print(GREEN, end="", flush=True)

        n = 0
        t0 = time.time()
        try:
            for _ in range(self.max_new):
                window = ids[-max_seq:]
                logits = self.model(
                    torch.tensor([window], device=self.device))[0, -1].float()

                if self.greedy:
                    nxt = int(logits.argmax())
                else:
                    logits = logits / self.temperature
                    if self.top_k:
                        kth = torch.topk(logits, self.top_k).values[-1]
                        logits[logits < kth] = -float("inf")
                    nxt = int(torch.multinomial(torch.softmax(logits, -1), 1))

                if nxt == self.sp.eos_id():
                    break

                ids.append(nxt)
                n += 1

                # BROKER MODE. The special tokens sit above the tokenizer's range,
                # so they cannot go through decode() -- they are printed as markers
                # and the surrounding runs decoded around them. On </call> the span
                # back to <call> is executed and the verified digits are spliced in,
                # in the cell's colour rather than the model's.
                if self.broker:
                    c = self.cells
                    if nxt in (c["call"], c["close"]) or nxt in c["inv"]:
                        marker = ("<call>" if nxt == c["call"] else
                                  "</call>" if nxt == c["close"] else
                                  f"⟨{c['inv'][nxt]}⟩")
                        print(f"{RESET}{CYAN} {marker}{RESET}{GREEN}",
                              end="", flush=True)
                        shown = self.sp.decode([i for i in ids
                                                if i < self.sp.get_piece_size()])
                        if nxt == c["close"]:
                            back = next(i for i, x in enumerate(reversed(ids))
                                        if x == c["call"])
                            res = self.run_cell(ids[len(ids) - 1 - back:])
                            if res is not None:
                                print(f"  {CYAN}{res}{RESET}{GREEN}",
                                      end="", flush=True)
                                ids.extend(self.sp.encode(f" {res}"))
                                shown = self.sp.decode(
                                    [i for i in ids if i < self.sp.get_piece_size()])
                        if self.delay:
                            time.sleep(self.delay)
                        continue
                # decode-whole-then-diff keeps spacing correct. Hold back any
                # trailing U+FFFD: the model emits byte-fallback tokens that
                # only form valid UTF-8 once the next byte arrives, and printing
                # a half-formed character puts mojibake on screen.
                # In broker mode the id list carries tokens the tokenizer has never
                # heard of, so decode only the ones it can.
                decodable = ([i for i in ids if i < self.sp.get_piece_size()]
                             if self.broker else ids)
                full = self.sp.decode(decodable).rstrip("�")
                if len(full) > len(shown):
                    print(full[len(shown):], end="", flush=True)
                    shown = full

                if self.delay:
                    time.sleep(self.delay)
        except KeyboardInterrupt:
            print(f"{RESET}{DIM} ⏹{RESET}", end="")

        dt = time.time() - t0
        rate = f"{n/dt:.0f} tok/s" if dt > 0 and n else "—"
        mode = "greedy" if self.greedy else f"t={self.temperature}"
        print(f"{RESET}\n  {DIM}{n} tokens · {dt:.1f}s · {rate} · {mode}{RESET}")

        # SAY SO WHEN NOTHING CAME BACK. The model can stop on its first token --
        # it did exactly that when a cell-shaped prompt was typed at the maths
        # checkpoint, which is a real thing to do on camera by mistake. What the
        # viewer saw was the prompt, then silence, then the prompt again: no
        # output, no error, nothing to explain. Baffling silence is the worst
        # failure mode available to a tool whose whole job is to be filmed, and
        # it costs one dim line to make it legible instead.
        if n == 0:
            print(f"  {DIM}no output — it stopped immediately. Usually the prompt")
            print(f"  belongs to a different checkpoint than the one loaded"
                  f" ({self.checkpoint}).{RESET}")
        print()

    @torch.no_grad()
    def next_words(self, prompt, k=10):
        ids = self.encode(prompt)
        logits = self.model(torch.tensor([ids], device=self.device))[0, -1].float()
        probs = torch.softmax(logits, -1)
        vals, idx = torch.topk(probs, k)
        top = [(self.sp.id_to_piece(int(i)), float(v)) for v, i in zip(vals, idx)]

        print(f"\n  {prompt} {BOLD}___{RESET}\n")
        for piece, p in top:
            word = piece.lstrip("▁")
            hot = word.lower() in NUMBER_WORDS
            bar = "█" * max(1, round(p * 40))
            colour = GREEN if hot else DIM
            print(f"    {colour}{word:<14}{RESET} {p:6.3f}  {colour}{bar}{RESET}")

        share = sum(p for t, p in top if t.lstrip("▁").lower() in NUMBER_WORDS)
        print(f"\n  number-word mass in top {k}: {BOLD}{share:.1%}{RESET}\n")
        return top, share

    @torch.no_grad()
    def slots(self, k=10):
        """Act 2a's four number slots at once, with the summary computed.

        /next is the on-camera beat -- typed one at a time so the bar chart
        renders live. This is the measuring tool: it re-derives the four figures
        the script quotes, which have to be re-read after any retrain because
        they are properties of the weights, not of the video.
        """
        print(f"\n  {BOLD}Four slots where a number is the only sensible next word{RESET}")
        print("  " + "─" * 55)
        rows = []
        for prompt, note in SLOT_PROMPTS:
            ids = self.encode(prompt)
            logits = self.model(
                torch.tensor([ids], device=self.device))[0, -1].float()
            vals, idx = torch.topk(torch.softmax(logits, -1), k)
            top = [(self.sp.id_to_piece(int(i)), float(v)) for v, i in zip(vals, idx)]
            share = sum(p for t, p in top if t.lstrip("▁").lower() in NUMBER_WORDS)
            rows.append((prompt, note, top, share))
            head = "  ".join(
                f"{GREEN if t.lstrip('▁').lower() in NUMBER_WORDS else DIM}"
                f"{t.lstrip('▁')} {p:.3f}{RESET}" for t, p in top[:5])
            print(f"\n  {DIM}{note}{RESET}")
            print(f"  {prompt} {BOLD}___{RESET}")
            print(f"    {head}")
            print(f"    number-word mass: {BOLD}{share:.1%}{RESET}")

        # Computed, never hardcoded: these figures are properties of whichever
        # checkpoint is loaded, and a retrain silently invalidates any constant.
        top_share, bottom_share = rows[0][3], rows[-1][3]
        print(f"\n  {DIM}{top_share:.1%} at the top of the ladder, "
              f"{bottom_share:.1%} at the bottom — a gradient, not a cliff.{RESET}")
        print(f"  {DIM}Row 2 is the trap: high mass, the count sequence ranked "
              f"correctly, and{RESET}")
        print(f"  {DIM}it still cannot count. Generate from it and see "
              f"(/greedy, then type it).{RESET}\n")


BANNER = f"""
{BOLD}TinyModel v11{RESET} — trained on TinyStories, 16M tokens
type a prompt and press enter · {DIM}/help for commands · ctrl-c stops generation{RESET}
"""


def show_config():
    """Act 1a — the architecture. What `bat …/config.json` used to do, minus
    leaving the REPL. Prints the arch keys only: the file also carries a
    `training` block and (once exported) an `exported_from` block, which are
    provenance rather than architecture and just crowd the frame."""
    cfg = json.loads(ARCH_CONFIG.read_text())
    print(f"\n{BOLD}The entire specification of the model{RESET}")
    print("─" * 37)
    print(f"{DIM}  {ARCH_CONFIG.relative_to(HERE)}{RESET}\n")
    keys = ["dim", "n_layers", "n_heads", "n_kv_heads", "ffn_dim", "max_seq",
            "vocab_size", "rope_theta", "tie_embeddings"]
    for k in keys:
        if k in cfg:
            v = cfg[k]
            v = f"{v:,}" if isinstance(v, int) and v > 999 else str(v)
            hot = k in ("vocab_size", "tie_embeddings")
            print(f"    {k:<16} {GREEN if hot else BOLD}{v}{RESET}")
    print(f"\n{DIM}  vocab_size is the published v11 tokenizer. Embeddings are tied,")
    print(f"  so the table is counted once — see /params.{RESET}\n")


def show_loop():
    """Act 1d — the training loop, from the file that actually ran. Highlights
    the five lines the VO names (forward, loss, backward, step, zero-grad)
    rather than asking the viewer to find them in a wall of source."""
    src = TRAIN_PY.read_text().splitlines()
    start = next((i for i, l in enumerate(src) if "for batch in stream_batches" in l), None)
    if start is None:
        print(f"  {DIM}could not find the loop in {TRAIN_PY}{RESET}")
        return
    marks = ("logits = model(", "loss = F.cross_entropy", "loss.backward()",
             "opt.step()", "opt.zero_grad()")
    print(f"\n{BOLD}The whole of training{RESET}")
    print("─" * 21)
    print(f"{DIM}  {TRAIN_PY.relative_to(HERE)}:{start+1}{RESET}\n")
    for i, line in enumerate(src[start:start + 16], start + 1):
        hot = any(m in line for m in marks)
        print(f"  {DIM}{i:>4}{RESET}  {GREEN if hot else ''}{line}{RESET}")
    print(f"\n{DIM}  Forward, loss, backward, step, zero-grad. That is the part")
    print(f"  people imagine is complicated.{RESET}\n")


def main():
    s = Session()
    print(BANNER)
    cfg = json.loads(ARCH_CONFIG.read_text())
    # Architecture from config.json, not from a loaded model: nothing is loaded
    # yet. See the module docstring for why that is deliberate.
    print(f"{DIM}{cfg['n_layers']} layers · dim {cfg['dim']} · vocab "
          f"{cfg['vocab_size']:,} · no checkpoint loaded yet{RESET}\n")

    while True:
        try:
            line = input(f"{BOLD}>{RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue

        if line.startswith("/"):
            cmd, _, rest = line.partition(" ")
            cmd, rest = cmd.lower(), rest.strip()

            if cmd in ("/quit", "/q", "/exit"):
                break
            elif cmd == "/help":
                print(__doc__.split("Commands (type these instead of a prompt):")[1]
                      .split("Uses the PUBLISHED")[0])
            elif cmd == "/config":
                show_config()
            elif cmd == "/params":
                import show_params
                show_params.main()
            elif cmd == "/data":
                import show_data
                show_data.main(rest.split() if rest else [])
            elif cmd == "/loop":
                show_loop()
            elif cmd == "/slots":
                if s.ensure_model():
                    s.slots()
            elif cmd == "/broker":
                s.load_broker()
            elif cmd == "/next":
                if rest and s.ensure_model():
                    s.next_words(rest)
                elif not rest:
                    print(f"  {DIM}usage: /next <prompt>{RESET}")
            elif cmd == "/greedy":
                s.greedy = True
                print(f"  {DIM}greedy — most likely token every time{RESET}")
            elif cmd == "/sample":
                s.greedy = False
                print(f"  {DIM}sampling at t={s.temperature}{RESET}")
            elif cmd == "/temp":
                try:
                    s.temperature = float(rest)
                    s.greedy = False
                    print(f"  {DIM}temperature {s.temperature}{RESET}")
                except ValueError:
                    print(f"  {DIM}usage: /temp 0.8{RESET}")
            elif cmd == "/len":
                try:
                    s.max_new = int(rest)
                    print(f"  {DIM}max {s.max_new} tokens{RESET}")
                except ValueError:
                    print(f"  {DIM}usage: /len 60{RESET}")
            elif cmd == "/full":
                s.load("model_full.pt")
            elif cmd == "/compiled":
                s.load("model_compiled.pt")
            elif cmd == "/mathonly":
                s.load("model_mathonly.pt")
            elif cmd == "/slow":
                s.delay = float(rest) if rest else 0.04
                print(f"  {DIM}{s.delay}s per token{RESET}")
            elif cmd == "/fast":
                s.delay = 0.0
                print(f"  {DIM}no delay{RESET}")
            else:
                print(f"  {DIM}unknown command {cmd} — /help{RESET}")
            continue

        if s.ensure_model():
            s.stream(line)


if __name__ == "__main__":
    main()
