#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["tokenizers>=0.20", "pyarrow>=14"]
# ///
"""Act 1c — what the model is actually trained on.

  uv run show_data.py
  uv run show_data.py --rows 5
  uv run show_data.py --tokens

Streams the SAME pinned revision of TinyStories the model was trained on
(revision f54c09f), so what's on screen is what went in.

--tokens additionally shows one story turned into the integers the model
actually sees, and works out how much of the 16M-token budget one story is.
"""

import argparse
import io
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
TOKENIZER = HERE / "tokenizer" / "tokenizer.json"   # the published v11 build

# the pinned revision used throughout — same documents, every run
HUB_SHA = "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64"
TOKENS_PHASE1 = 16_000_000

DIM, GREEN, BOLD, RESET = "\033[2m", "\033[92m", "\033[1m", "\033[0m"

# The first training shard, addressed by revision. The sha is IN THE URL rather
# than in a `revision=` keyword, which suits the beat this feeds: the address is
# the pin, and a viewer can paste it into a browser.
DATA_URL = ("https://huggingface.co/datasets/roneneldan/TinyStories/resolve/"
            f"{HUB_SHA}/data/train-00000-of-00004-2d5a1467fff1081b.parquet")


class _HTTPRangeFile(io.RawIOBase):
    """The seekable file parquet wants, backed by HTTP range requests.

    Parquet keeps its footer at the END of the file, so a reader seeks there
    first, learns where the row groups are, and then reads only the one it wants.
    That means the whole 249MB shard never moves -- three range requests is the
    entire cost of the first thousand stories.
    """

    def __init__(self, url: str):
        from urllib.request import Request, urlopen
        self._Request, self._urlopen = Request, urlopen
        self.url, self.pos, self.requests = url, 0, 0
        with urlopen(Request(url, method="HEAD")) as r:
            self.size = int(r.headers["Content-Length"])

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset, whence=io.SEEK_SET):
        self.pos = (offset if whence == io.SEEK_SET
                    else self.pos + offset if whence == io.SEEK_CUR
                    else self.size + offset)
        return self.pos

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        if n == 0 or self.pos >= self.size:
            return b""
        end = min(self.pos + n, self.size) - 1
        req = self._Request(self.url, headers={"Range": f"bytes={self.pos}-{end}"})
        with self._urlopen(req) as resp:
            data = resp.read()
        self.requests += 1
        self.pos += len(data)
        return data


def fetch_stories(rows: int, skip: int) -> list[str]:
    """The first `rows` stories after `skip`, read from the pinned parquet shard.

    THIS USED TO GO THROUGH `datasets` STREAMING, AND THAT BROKE THE REPL. Loading
    a HuggingFace dataset in a process that has imported torch leaves the two
    libraries' teardown waiting on each other: /data worked, printed, returned to
    the prompt -- and then the REPL could never exit. Reproduced down to three
    lines (`import torch; import show_data; show_data.main([])`) and killed at
    five minutes. show_data.py alone was always fine, because nothing in it
    imports torch, which is exactly why the bug survived being "verified".

    Reading the parquet directly needs pyarrow and nothing else, so the REPL never
    loads `datasets` at all -- and it is faster besides: three range requests
    against a 249MB shard rather than a streaming connection.

    IT HAS TO BE THE PARQUET, NOT `TinyStories-train.txt`. The plain-text file is
    the obvious shortcut and it is subtly wrong: it collapses paragraph breaks,
    so its first story is 699 characters where the parquet's is 701. The model was
    trained through the parquet, and this act's claim is that what is on screen is
    what went in -- two characters is enough to make that false.
    """
    import pyarrow.parquet as pq

    want = rows + skip
    handle = _HTTPRangeFile(DATA_URL)
    pf = pq.ParquetFile(handle)
    out: list[str] = []
    for group in range(pf.num_row_groups):
        out += pf.read_row_group(group, columns=["text"]).column("text").to_pylist()
        if len(out) >= want:
            break
    if len(out) < want:
        raise SystemExit(f"\nshard holds {len(out)} stories; asked for {want}.\n")
    return [s.strip() for s in out[skip:want]]


def main(argv=None):
    ap = argparse.ArgumentParser(prog="/data")
    ap.add_argument("--rows", type=int, default=3, help="stories to show")
    ap.add_argument("--tokens", action="store_true", help="also show tokenization")
    ap.add_argument("--skip", type=int, default=0, help="skip N stories first")
    args = ap.parse_args(argv)

    print(f"\n{BOLD}TinyStories{RESET} — the entire education of this model")
    print(f"{DIM}roneneldan/TinyStories, revision {HUB_SHA[:12]} (pinned){RESET}\n")

    stories = fetch_stories(args.rows, args.skip)

    for i, text in enumerate(stories, 1):
        print(f"  {DIM}── story {i + args.skip} {'─' * 56}{RESET}")
        for line in text.split("\n"):
            print(f"  {line}")
        print()

    print(f"{DIM}  Synthetic. Deliberately tiny vocabulary. Written to answer:")
    print(f"  how small can a language model be and still write real English?{RESET}\n")

    if not args.tokens:
        return

    # V11Tokenizer rather than a bare Tokenizer.from_file: it verifies the
    # file's sha256 against the published build. This act puts token IDs on
    # screen, and "you can check these" is the claim -- so the tokenizer that
    # produced them has to be provably the published one, exactly as repl.py
    # requires before it will generate.
    from demo_common import V11Tokenizer
    tok = V11Tokenizer(TOKENIZER)

    text = stories[0]
    ids = tok.encode(text)

    print(f"{BOLD}  And here is that first story as the model sees it{RESET}\n")
    head = text[:110].replace("\n", " ")
    print(f"  {DIM}text  {RESET}{head}…\n")
    print(f"  {DIM}pieces{RESET} {[tok.id_to_piece(i) for i in ids[:18]]}…\n")
    print(f"  {DIM}ids   {RESET}{ids[:18]}…\n")

    n = len(ids)
    print(f"  one story = {BOLD}{n:,} tokens{RESET}"
          f" · {len(text):,} characters"
          f" · {n/len(text):.3f} tokens/char")
    print(f"  phase 1 budget = {TOKENS_PHASE1:,} tokens"
          f" ≈ {BOLD}{TOKENS_PHASE1/n:,.0f} stories{RESET}\n")


if __name__ == "__main__":
    main()
