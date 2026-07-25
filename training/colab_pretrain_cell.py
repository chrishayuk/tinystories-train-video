# ── v11 TinyStories pretrain · one Colab cell ────────────────────────────────
# Paste into ONE cell of a Colab notebook set to Runtime → Change runtime type →
# T4 GPU, then run it. It pretrains the 115M base model from scratch on
# TinyStories (16M tokens, the published v11 tokenizer, NO maths mid-train) and
# publishes the result to the Hub.
#
# Needs no chuk-datasets key, no chuk-train control plane and no join token:
# configs/colab.json carries no `data:` block, so train.py streams TinyStories
# straight from HuggingFace at its pinned revision. Roughly 45-75 min on a T4.
#
# BEFORE YOU RUN IT: add your Hub token as a Colab secret named HF_TOKEN (key
# icon in the left sidebar, "Notebook access" on). It must be a WRITE token. The
# cell checks this up front, on purpose -- discovering a read-only token after an
# hour of training is the one failure worth spending three seconds to prevent.
#
# REHEARSE IT FIRST. Set CONFIG = "configs/smoke.json" and PUBLISH = False: same
# code path, same data source, 19 steps instead of 15,625, done in ~2 minutes.
# That exercises the install, the clone, the tokenizer guard, the HF stream, the
# training loop and the checkpoint writer before you spend an hour on them.

REPO_URL = "https://github.com/chrishayuk/tinystories-train-video.git"
REPO_ID = "chrishayuk/v11-tinystories-115m-base"   # where the model gets published
CONFIG = "configs/colab.json"                      # 16M tokens, HF-streamed, seed 42
WORK = "/content/v11-pretrain"                     # metrics + checkpoints land here
PUBLISH = True                                     # False = train only, publish later
PRIVATE = False                                    # True = private Hub repo

import json, os, re, subprocess, sys
from pathlib import Path

def sh(cmd, **kw):
    """Run and stream straight into the cell output -- no capture, so train.py's
    per-step lines appear live rather than in one lump at the end."""
    print(f"\n$ {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=True, **kw)

# ── 1 · hardware ─────────────────────────────────────────────────────────────
import torch
if not torch.cuda.is_available():
    raise SystemExit(
        "No CUDA device. Runtime → Change runtime type → T4 GPU, then rerun.\n"
        "On CPU this run takes days, not an hour -- refusing rather than starting."
    )
print(f"GPU: {torch.cuda.get_device_name(0)}  torch {torch.__version__}")

# ── 2 · Hub credentials, BEFORE the hour of training ─────────────────────────
if PUBLISH:
    try:
        from google.colab import userdata
        os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
    except Exception:
        if not os.environ.get("HF_TOKEN"):
            from huggingface_hub import notebook_login
            notebook_login()

    from huggingface_hub import HfApi
    who = HfApi().whoami()
    role = (who.get("auth") or {}).get("accessToken", {}).get("role")
    print(f"Hub: {who.get('name')} (token role: {role or 'unknown'})")
    if role == "read":
        raise SystemExit(
            f"That HF_TOKEN is READ-ONLY, so the push at the end would fail after "
            f"~an hour of training. Mint a write token at "
            f"https://huggingface.co/settings/tokens and update the Colab secret."
        )
    owner = REPO_ID.split("/")[0]
    if owner != who.get("name") and owner not in {o["name"] for o in who.get("orgs", [])}:
        raise SystemExit(
            f"Token belongs to {who.get('name')!r} but REPO_ID is under {owner!r}, "
            f"which is neither that account nor one of its orgs -- the push would 403."
        )

# ── 3 · code ─────────────────────────────────────────────────────────────────
src = Path("/content/tinystories-train-video")
if src.is_dir():
    sh(f"git -C {src} pull --ff-only")
else:
    sh(f"git clone --depth 1 {REPO_URL} {src}")
commit = subprocess.run(f"git -C {src} rev-parse HEAD", shell=True,
                        capture_output=True, text=True).stdout.strip()
print(f"source commit: {commit}")

for f in (src / CONFIG, src / "training/publish_pretrain_hf.py"):
    if not f.is_file():
        raise SystemExit(f"{f.relative_to(src)} is not in the clone -- push it to "
                         f"{REPO_URL} first, this cell trains from the pushed tree.")

# Deliberately does NOT install torch: Colab ships a CUDA-matched build and a
# naive `pip install torch` can silently replace it with a mismatched or CPU
# wheel. Only what this run adds on top.
sh('pip install -q "tokenizers>=0.20" "datasets>=2.18" "pyarrow>=14" '
   '"safetensors>=0.4" "huggingface_hub>=0.24" "transformers>=4.44"')

# ── 4 · train ────────────────────────────────────────────────────────────────
work = Path(WORK); (work / "ckpt").mkdir(parents=True, exist_ok=True)
env = dict(os.environ,
           CHUK_CONFIG=str(src / CONFIG),
           CHUK_METRICS=str(work / "metrics.jsonl"),
           CHUK_CKPT_DIR=str(work / "ckpt"))

# Auto-resume from the newest complete milestone if this cell is being rerun
# after a disconnect. Honest about what that does and doesn't restore: weights
# and step counter carry over, but the HF stream restarts from the top, so a
# resumed run re-reads the corpus prefix it already saw. Fine for a 16M-token
# run over a ~400M-token corpus; not equivalent to an uninterrupted run.
done = {int(m.group(1)): d for d in (work / "ckpt").iterdir()
        if (m := re.fullmatch(r"step_(\d+)", d.name)) and (d / ".ready").is_file()}
if done:
    latest = done[max(done)]
    env["CHUK_RESUME_CKPT"] = str(latest)
    print(f"resuming from {latest.name} (corpus stream restarts from the top)")

print(f"\ntraining -- config {CONFIG}, checkpoints in {work/'ckpt'}")
print("Keep this tab open and visible; Colab reclaims idle runtimes.\n")
sh(f"python3 {src/'training/harness_pretrain/train.py'}", env=env, cwd=str(src))

# ── 5 · publish ──────────────────────────────────────────────────────────────
if not PUBLISH:
    print(f"\nTrained. Checkpoints in {work/'ckpt'} -- publish with "
          f"training/publish_pretrain_hf.py when ready.")
else:
    cfg = json.loads((src / CONFIG).read_text())
    sh(f"python3 {src/'training/publish_pretrain_hf.py'} "
       f"--ckpt-dir {work/'ckpt'} --metrics {work/'metrics.jsonl'} "
       f"--repo-id {REPO_ID} --run-config {CONFIG} --source-commit {commit} "
       f"--seed {cfg.get('seed', 42)} --batch-size {cfg.get('batch_size', 4)} "
       f"--lr {cfg.get('lr', 3e-4)}" + (" --private" if PRIVATE else ""),
       env=env, cwd=str(src))
    print(f"\nhttps://huggingface.co/{REPO_ID}")
    print("Download the checkpoints before closing the tab if you want the "
          "intermediate milestone weights -- only the final ones get published.")
