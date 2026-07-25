#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# ///
"""Replay a finished training run's terminal output, for camera.

    replay_run.py run_pretrain --speed 60 --max-gap 2

A 16M-token pretrain takes ~2h and is the Act 1 centrepiece. Nobody films two
hours. This plays a *completed* run's output back at a speed that reads, with the
original timing preserved in shape.

WHY REPLAY IS THE MORE FAITHFUL OPTION, not a compromise. The obvious
alternative is to re-run training with a screen capture going. But `seed 42` and
a pinned corpus make the trajectory reproducible, which means a re-run produces a
*second* run -- same numbers, different artifact -- whose checkpoints you then
throw away. Replaying the recorded log shows **the run that actually produced the
published weights**: every loss value on screen is the one that made
`chrishayuk/v11-tinystories-115m-base`, and `train.log` + `metrics.jsonl` +
`ckpt/step_*/meta.json` are all still on disk to check it against.

What it is NOT is live. Do not narrate it in the present tense as if the model
were being trained while you speak. It is a screen recording of a real run, which
is exactly what pre-recording means -- see SCRIPT.md § PRE-RECORD.

TIMING. Two sources, in order of preference:

  1. `train_replay.jsonl` -- written by train.py itself, one
     {"t": <seconds since start>, "line": ...} per line. Exact.
  2. `train.log` + `metrics.jsonl` -- reconstructed. Every "step N/M ... T tok/s"
     line carries a cumulative rate, so elapsed = N * tokens_per_step / T
     recovers its timestamp; everything between two of those is interpolated.
     Accurate to a second or so, which is far finer than any playback speed.

(2) exists because the 2026-07-25 run predates (1). Runs after that get exact
timing for free.

--max-gap is the one that matters on camera. The HF-streaming route stalls for
30-60s at a time while the shuffle buffer refills; at 60x that is still a
one-second freeze, and a freeze reads as a crash. Capping the gap keeps the
*shape* of the timing -- fast stretches stay fast -- while bounding dead air.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

STEP_RE = re.compile(r"^\[v11-pretrain\] step (\d+)/(\d+) .*?([\d.]+) tok/s\s*$")
DIM, GREEN, BOLD, RESET = "\033[2m", "\033[92m", "\033[1m", "\033[0m"


def load_exact(path: Path) -> list[tuple[float, str]]:
    """train.py's own replay log: timestamps are recorded, not inferred."""
    out = []
    for raw in path.read_text().splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            rec = json.loads(raw)
        except json.JSONDecodeError:
            continue
        out.append((float(rec["t"]), rec["line"]))
    return out


def load_reconstructed(log: Path, tokens_per_step: int) -> list[tuple[float, str]]:
    """Recover per-line timing from the step lines' cumulative tokens/sec.

    Anchors are exact wherever a "step N/M ... T tok/s" line appears; everything
    between two anchors is spread evenly across the interval. Checkpoint blocks
    land inside those gaps, which is right -- writing 600MB and generating three
    samples is most of what the gap consists of.
    """
    lines = log.read_text().splitlines()
    anchors: dict[int, float] = {0: 0.0}
    for i, line in enumerate(lines):
        m = STEP_RE.match(line)
        if m:
            step, rate = int(m.group(1)), float(m.group(3))
            if rate > 0:
                anchors[i] = step * tokens_per_step / rate
    if len(anchors) < 2:
        sys.exit(f"{log}: no 'step N/M ... tok/s' lines to recover timing from")
    anchors[len(lines) - 1] = max(anchors.values())

    idx = sorted(anchors)
    out = []
    for i, line in enumerate(lines):
        if i in anchors:
            out.append((anchors[i], line))
            continue
        # bracket i between the nearest anchors either side and interpolate
        lo = max(j for j in idx if j <= i)
        hi = min((j for j in idx if j >= i), default=lo)
        if hi == lo:
            out.append((anchors[lo], line))
        else:
            frac = (i - lo) / (hi - lo)
            out.append((anchors[lo] + frac * (anchors[hi] - anchors[lo]), line))
    return out


def find_source(run_dir: Path) -> tuple[str, Path]:
    exact = run_dir / "train_replay.jsonl"
    if exact.is_file():
        return "exact", exact
    log = run_dir / "train.log"
    if log.is_file():
        return "reconstructed", log
    sys.exit(f"{run_dir}: need train_replay.jsonl or train.log")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run_dir", type=Path, nargs="?", default=Path("run_pretrain"),
                    help="dir holding train.log / train_replay.jsonl / metrics.jsonl")
    ap.add_argument("--speed", type=float, default=60.0,
                    help="playback multiplier (default 60x)")
    ap.add_argument("--realtime", action="store_true",
                    help="original timing, unscaled — all ~2h of it")
    ap.add_argument("--max-gap", type=float, default=2.0,
                    help="cap any single pause, in playback seconds (0 = uncapped)")
    ap.add_argument("--tokens-per-step", type=int, default=1024,
                    help="batch_size * max_seq, for reconstructing timing")
    ap.add_argument("--plain", action="store_true", help="no highlighting")
    args = ap.parse_args()

    kind, path = find_source(args.run_dir)
    events = (load_exact(path) if kind == "exact"
              else load_reconstructed(path, args.tokens_per_step))
    if not events:
        sys.exit(f"{path}: nothing to replay")

    speed = 1.0 if args.realtime else max(args.speed, 1e-6)
    total = events[-1][0]
    print(f"{DIM}replaying {path} ({kind} timing) — {len(events)} lines, "
          f"{total/60:.0f} min of run at {speed:g}x"
          f"{'' if not args.max_gap else f', gaps capped at {args.max_gap:g}s'}"
          f"{RESET}\n", file=sys.stderr)

    prev = 0.0
    for t, line in events:
        gap = max(0.0, t - prev) / speed
        if args.max_gap:
            gap = min(gap, args.max_gap)
        if gap:
            time.sleep(gap)
        prev = t
        if args.plain:
            print(line, flush=True)
        else:
            # The two things a viewer is meant to track: the loss falling, and a
            # milestone landing. Everything else is texture.
            hot = "checkpoint step_" in line or line.startswith("  '")
            print(f"{GREEN if hot else ''}{line}{RESET}" if hot else line, flush=True)


if __name__ == "__main__":
    main()
