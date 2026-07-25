#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["torch>=2.2", "tokenizers>=0.20", "datasets>=2.18", "numpy"]
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
  /broker          let the model call Z80 cells — Act 4 (not built yet)
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
# uses them so the four figures the script quotes are re-derivable from one place.
SLOT_PROMPTS = [
    ("Once upon a time there were", "story idiom — no counting required"),
    ("She counted the apples. There were", "counting, but no arithmetic"),
    ("Lily had three apples and Tom gave her four more. Now Lily has",
     "needs 3 + 4 = seven"),
    ("Tom had two cats and one dog. Altogether he had", "needs 2 + 1 = three"),
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


class Session:
    def __init__(self):
        self.temperature = 0.8
        self.top_k = 40
        self.greedy = False
        self.max_new = 60
        self.delay = 0.0
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
        n = sum(p.numel() for p in self.model.parameters())
        print(f"{DIM}  {n/1e6:.1f}M params · {self.config.n_layers} layers · "
              f"dim {self.config.dim} · vocab {self.config.vocab_size:,} · "
              f"{self.device}{RESET}")
        return True

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
                # decode-whole-then-diff keeps spacing correct. Hold back any
                # trailing U+FFFD: the model emits byte-fallback tokens that
                # only form valid UTF-8 once the next byte arrives, and printing
                # a half-formed character puts mojibake on screen.
                full = self.sp.decode(ids).rstrip("�")
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
        print(f"{RESET}\n  {DIM}{n} tokens · {dt:.1f}s · {rate} · {mode}{RESET}\n")

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
        idiom, counting = rows[0][3], rows[1][3]
        arith = [r[3] for r in rows[2:]]
        reach = [t.lstrip("▁") for t, _ in rows[2][2][:3]]
        print(f"\n  {DIM}idiom {idiom:.1%} · counting {counting:.1%} · "
              f"arithmetic {min(arith):.1%}–{max(arith):.1%} "
              f"(reaches for {', '.join(reach)}){RESET}")
        print(f"  {DIM}Three tiers: it knows a quantity belongs after \"she counted\","
              f" and cannot combine two.{RESET}\n")


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


def broker_not_built():
    print(f"\n  {BOLD}/broker is not built yet.{RESET}\n")
    print("  It needs Act 4's cell-call checkpoint, which was trained on the")
    print("  retired 71261-vocab tokenizer and has to be rebuilt on the published")
    print("  one before the model can emit a call at all.\n")
    print(f"  {DIM}See SCRIPT.md \"What still needs running\" item 5c. The")
    print(f"  non-interactive version is ./run_broker.sh, blocked for the same")
    print(f"  reason.{RESET}\n")


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
                broker_not_built()
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
