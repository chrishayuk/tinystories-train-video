"""Shared plumbing for the on-camera demos — repl.py and cold_open.py.

Not a script: it declares no PEP 723 block and is never run directly. The two
demos stay independently `uv run`-able; this only stops them carrying two copies
of the tokenizer wrapper and the vocabulary guard, which is how those two copies
quietly disagree.

Everything here is the part that MUST be identical between them: which tokenizer
file, which sha it has to hash to, what counts as a number word, and what happens
when a checkpoint and a tokenizer disagree. A drift in any of those changes what
goes on screen.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

TOKENIZER = HERE / "tokenizer" / "tokenizer.json"
ARTEFACTS = HERE / "model_v11"
ARCH_CONFIG = HERE / "training" / "harness_pretrain" / "config.json"
TRAIN_PY = HERE / "training" / "harness_pretrain" / "train.py"

DIM, GREEN, BOLD, RESET = "\033[2m", "\033[92m", "\033[1m", "\033[0m"

# Act 2a counts how much next-token mass lands on a number word. Twelve is the
# ceiling because that is where TinyStories' own number words stop being common,
# not because of anything about the tokenizer -- "thirteen" through "sixteen" are
# also single pieces in v11. See SCRIPT.md, "One thing NOT to overclaim".
NUMBER_WORDS = {"zero", "one", "two", "three", "four", "five", "six", "seven",
                "eight", "nine", "ten", "eleven", "twelve"}

# The published v11 tokenizer: crates.io v11-core, PyPI v11-tokenizer, HF
# chrishayuk/v11-tokenizer. Same constant train.py, publish_pretrain_hf.py and
# export_repl_checkpoint.py each guard on.
PUBLISHED_TOKENIZER_SHA256 = "10dd51100331ab503115db23eee7e8dc3e360e3aed697c8a2e1b12b8f46031ae"


class V11Tokenizer:
    """The published v11 build, wrapped in the small SentencePiece-shaped
    interface the demos use. Verified by hash on load, so an id that goes on
    camera is provably from the published artifact."""

    SHA256 = PUBLISHED_TOKENIZER_SHA256

    def __init__(self, path=TOKENIZER):
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
    """A checkpoint built against a different vocabulary cannot be driven by
    this tokenizer -- the ids mean different things, and the result is fluent
    nonsense rather than an error. So refuse, loudly, in both demos."""
    if config.vocab_size != tok.get_piece_size():
        sys.exit(
            f"\n  checkpoint/tokenizer mismatch -- refusing to generate.\n"
            f"    {checkpoint}: vocab {config.vocab_size:,}\n"
            f"    published v11 tokenizer: vocab {tok.get_piece_size():,}\n\n"
            f"  That checkpoint predates the published tokenizer and is retired.\n"
            f"  Export a current one:\n"
            f"    uv run training/export_repl_checkpoint.py --ckpt-dir run_pretrain/ckpt\n")


def missing_checkpoint_message(path: Path) -> str:
    """Returned rather than printed, because the two callers need different
    endings: repl.py prints it and carries on (the commands that need no weights
    still work), cold_open.py exits on it."""
    extra = ""
    if path.name == "model_compiled.pt":
        extra = (
            "  model_compiled.pt is the PHASE 3 checkpoint (frozen FFN, attention\n"
            "  retrained). The Act 1e run is phase 1 only, so it does not exist on\n"
            "  this lineage yet -- that phase has not been run. Try model_full.pt.\n\n")
    elif path.name == "model_mathonly.pt":
        extra = (
            "  model_mathonly.pt comes from Act 3's maths mid-train, which has not\n"
            "  been run yet. See SCRIPT.md \"What still needs running\" item 1.\n\n")
    shown = path.relative_to(HERE) if path.is_relative_to(HERE) else path
    return (
        f"\n  no checkpoint at {shown}\n\n{extra}"
        f"  Export one from a train.py run:\n"
        f"    uv run training/export_repl_checkpoint.py --ckpt-dir run_pretrain/ckpt\n\n"
        f"  Or fetch the published base model:\n"
        f"    huggingface.co/chrishayuk/v11-tinystories-115m-base\n")
