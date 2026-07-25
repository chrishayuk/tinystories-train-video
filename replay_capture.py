"""Record a training run's terminal output so it can be replayed for camera.

Not a script. Imported by the trainers that live in this repo; `training/replay_run.py`
plays the result back.

Why this exists: every training run in this video is measured in hours, and none of
them is watchable in real time. Replaying a finished run is also the *more* faithful
option than re-running one for the camera -- a re-run produces a second run whose
checkpoints get discarded, while the recording shows the run that actually produced
the published weights. So the timing has to be captured while it happens, because it
cannot be recovered afterwards: stalls, checkpoint writes and sample generations are
most of what the wall clock consists of and none of them leave a trace in a metrics
file.

`training/harness_pretrain/train.py` carries its own inline copy rather than
importing this, and deliberately so: it is a self-contained chuk-train code unit that
gets shipped to workers, the same reason it vendors `tiny_model_v11/` and the
tokenizer. This file is the canonical version; keep them in step.
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

REPLAY_LOG = "train_replay.jsonl"


class Tee:
    """Timestamp every line written to `stream` into a JSONL replay log.

    One small append per printed line -- nothing against a training step. Never
    fails the run: if the log cannot be written the flag flips and training carries
    on with only the replay lost.
    """

    def __init__(self, stream, path: Path, t0: float | None = None):
        self._stream = stream
        self._path = path
        self._t0 = time.time() if t0 is None else t0
        self._buf = ""
        self._ok = True

    def write(self, text):
        self._stream.write(text)
        self._buf += text
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            self._record(line)
        return len(text)

    def _record(self, line: str):
        if not self._ok:
            return
        try:
            with self._path.open("a") as f:
                f.write(json.dumps({"t": round(time.time() - self._t0, 3),
                                    "line": line}) + "\n")
        except OSError:
            self._ok = False

    def flush(self):
        self._stream.flush()

    def __getattr__(self, name):
        return getattr(self._stream, name)


def start_capture(run_dir: Path, log_name: str = REPLAY_LOG) -> Path | None:
    """Truncate the replay log and route stdout through a Tee. Returns its path.

    Truncates rather than appends: a replay log describes one run, and a file
    holding two concatenated runs replays as nonsense with no obvious tell.
    """
    try:
        run_dir.mkdir(parents=True, exist_ok=True)
        path = run_dir / log_name
        path.write_text("")
        sys.stdout = Tee(sys.stdout, path)
        return path
    except OSError:
        return None  # replay is a nicety; training is not


def completed_run(run_dir: Path, done_marker: str,
                  log_names: tuple[str, ...] = (REPLAY_LOG,)) -> Path | None:
    """The log of a run that reached `done_marker`, or None.

    Checking for the marker rather than merely for the file is the point: a run that
    died halfway leaves a log behind too, and replaying it would show training
    stopping for no reason.
    """
    for name in log_names:
        path = run_dir / name
        if path.is_file() and done_marker in path.read_text():
            return path
    return None


def max_token_id(rows) -> int:
    """Highest id in a pre-tokenized corpus -- for the guard below."""
    return max((max(r["ids"]) for r in rows if r.get("ids")), default=-1)


def check_corpus_vocab(rows, vocab_size: int, corpus_path) -> None:
    """Refuse to train on a corpus whose ids fall outside this vocabulary.

    The failure this catches is silent, which is why it is a refusal. A corpus built
    with a different tokenizer is a flat array of plausible-looking integers; train
    on it and the loss curve looks fine and every generation is fluent nonsense.
    An out-of-range id is the cheap half of the check -- it catches a *larger* id
    space, e.g. a corpus built with the retired 71261-piece SentencePiece build
    against this 71260-piece model.
    """
    hi = max_token_id(rows)
    if hi >= vocab_size:
        raise SystemExit(
            f"\nREFUSING TO TRAIN -- corpus contains token id {hi}, outside this "
            f"model's vocabulary (size {vocab_size}).\n"
            f"  corpus: {corpus_path}\n"
            f"That is a different tokenizer's id space. Rebuild the corpus with the "
            f"published v11 build:\n"
            f"  uv run training/build_mathonly_corpus.py --drill 90000 --seed 90\n")
