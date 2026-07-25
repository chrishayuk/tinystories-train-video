#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["torch>=2.2", "tokenizers>=0.20", "numpy"]
# ///
"""Interactive REPL for TinyModel v11 — for typing prompts live on camera.

  uv run repl.py

Type a prompt, press enter, watch it generate token by token. Nothing is
pre-canned; every word on screen is produced live.

Commands (type these instead of a prompt):
  /next <prompt>   top-10 next-word predictions instead of generating
  /greedy          always take the most likely token (deterministic)
  /sample          sample with temperature (default)
  /temp 0.8        set sampling temperature
  /len 60          set max tokens to generate
  /full            switch to model_full.pt   (after phase 1/2, 16M tokens)
  /compiled        switch to model_compiled.pt (after phase 3, frozen FFN)
  /mathonly        switch to model_mathonly.pt (Act 3: maths mid-trained, no cells)
  /slow            add a delay per token, for camera pacing
  /fast            no delay (default)
  /help  /quit

Ctrl-C stops a generation without leaving the REPL.

Uses the PUBLISHED v11 tokenizer (vocab 71260) and refuses to run against any
checkpoint built on a different vocabulary.
"""

import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

TOKENIZER = HERE / "tokenizer" / "tokenizer.json"
ARTEFACTS = HERE / "model_v11"

DIM, GREEN, BOLD, RESET = "\033[2m", "\033[92m", "\033[1m", "\033[0m"

NUMBER_WORDS = {"zero", "one", "two", "three", "four", "five", "six", "seven",
                "eight", "nine", "ten", "eleven", "twelve"}


def no_model_yet(path):
    sys.exit(
        f"\n  no checkpoint at {path}\n\n"
        f"  The pre-existing 71261-vocab model has been retired -- every demo\n"
        f"  that generates text now runs on the Act 1e model, which has not been\n"
        f"  trained yet. See SCRIPT.md, \"What still needs running\" item 1.\n\n"
        f"  show_data.py, show_params.py and the v11 CLI need no checkpoint\n"
        f"  and work today.\n")

class V11Tokenizer:
    """The published v11 build (pip install v11-tokenizer / HF
    chrishayuk/v11-tokenizer), wrapped in the small SentencePiece-shaped
    interface the rest of this file uses. Verified by hash on load, so what
    goes on camera is provably the published artifact."""

    SHA256 = "10dd51100331ab503115db23eee7e8dc3e360e3aed697c8a2e1b12b8f46031ae"

    def __init__(self, path):
        import hashlib
        from tokenizers import Tokenizer
        if not path.exists():
            sys.exit(f"missing tokenizer: {path}")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != self.SHA256:
            sys.exit(f"{path} is not the published v11 build\n"
                     f"  expected {self.SHA256}\n  found    {actual}\n"
                     f"  re-fetch: huggingface.co/chrishayuk/v11-tokenizer")
        self.t = Tokenizer.from_file(str(path))
        self._inv = {i: p for p, i in self.t.get_vocab().items()}

    def encode(self, text):
        return self.t.encode(text).ids

    def decode(self, ids):
        return self.t.decode(ids)

    def id_to_piece(self, i):
        return self._inv.get(i, "?")

    def _special(self, tok):
        i = self.t.token_to_id(tok)
        return -1 if i is None else i

    def bos_id(self):
        return self._special("<s>")

    def eos_id(self):
        return self._special("</s>")

    def get_piece_size(self):
        return self.t.get_vocab_size()


def check_vocab(config, tok, checkpoint):
    """The published tokenizer has 71260 pieces. A checkpoint built against
    any other vocabulary cannot be driven by it -- the ids would mean
    different things and generation is fluent nonsense at perplexity ~1e7.
    Refuse instead, and say what to do about it."""
    if config.vocab_size != tok.get_piece_size():
        sys.exit(
            f"\n  checkpoint/tokenizer mismatch -- refusing to generate.\n"
            f"    {checkpoint}: vocab {config.vocab_size:,}\n"
            f"    published v11 tokenizer: vocab {tok.get_piece_size():,}\n\n"
            f"  This checkpoint predates the published tokenizer and is retired.\n"
            f"  Every demo now runs on the Act 1e model -- see SCRIPT.md,\n"
            f'  "What still needs running" items 1 and 5.\n')


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
        self.checkpoint = "model_compiled.pt"
        self.sp = None
        self.model = None
        self.config = None

    def load(self, checkpoint=None):
        from tiny_model_v11 import load_from_artifacts

        if checkpoint:
            self.checkpoint = checkpoint
        if self.sp is None:
            self.sp = V11Tokenizer(TOKENIZER)

        path = ARTEFACTS / "artifacts" / self.checkpoint
        if not path.exists():
            no_model_yet(path)

        print(f"{DIM}loading {self.checkpoint} …{RESET}", end=" ", flush=True)
        t0 = time.time()
        self.model, self.config = load_from_artifacts(
            ARTEFACTS, checkpoint=self.checkpoint)
        print(f"{DIM}{time.time()-t0:.1f}s{RESET}")
        check_vocab(self.config, self.sp, self.checkpoint)
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


BANNER = f"""
{BOLD}TinyModel v11{RESET} — trained on TinyStories, 16M + 8M tokens
type a prompt and press enter · {DIM}/help for commands · ctrl-c stops generation{RESET}
"""


def main():
    s = Session()
    if not s.load():
        sys.exit(1)

    n = sum(p.numel() for p in s.model.parameters())
    print(BANNER)
    print(f"{DIM}{n/1e6:.1f}M params · {s.config.n_layers} layers · dim {s.config.dim}"
          f" · vocab {s.config.vocab_size:,} · {s.device}{RESET}\n")

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
                print(__doc__.split("Commands")[1].split("Uses the NATIVE")[0])
            elif cmd == "/next":
                if rest:
                    s.next_words(rest)
                else:
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

        s.stream(line)


if __name__ == "__main__":
    main()
