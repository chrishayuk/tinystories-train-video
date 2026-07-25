# ── Rescue checkpoints stranded on a worker · one Colab cell ─────────────────
# Paste into a cell of the notebook that is RUNNING THE WORKER, and run it.
#
# WHY THIS EXISTS. The harness's output collector marks a checkpoint "collected"
# whether or not its upload succeeded, and gates the `Artifact` message on
# success -- so a failed upload is silent. `list_checkpoints` stays empty, the
# logs say `[harness ckpt] step_1500`, and the weights exist only on the worker.
# When the Colab runtime is reclaimed they go with it. Root cause and fix are in
# gpu-training-harness's ROADMAP; this is the tourniquet, not the surgery.
#
# WHAT IT DOES. Walks the worker's job sandboxes, finds every checkpoint the
# trainer finished writing, and pushes it to a Hub repo keyed by run id.
#
# SAFE TO RUN WHILE TRAINING CONTINUES. It only touches step dirs carrying a
# `.ready` marker, which train_mathonly.py touches last -- after both
# model.safetensors and meta.json are on disk. It writes nothing into the
# sandbox and never deletes anything.
#
# COLAB RUNS ONE CELL AT A TIME. The worker cell blocks, so to run this you must
# interrupt it (■). That is fine: the sandbox outlives the job -- the worker only
# clears a sandbox when it *creates* one for that job id, never on exit -- so
# every checkpoint from every run this runtime has executed is still on disk.
# Re-run the worker cell afterwards to rejoin the fleet and pick up the queue.
#
# BEFORE YOU RUN IT: HF_TOKEN as a Colab secret (key icon, "Notebook access"
# on), and it must be a WRITE token.

REPO_ID = "chrishayuk/v11-tinystories-115m-mathonly-ckpts"
SANDBOX_ROOT = "/tmp"          # where the worker makes chuk-job-<run id>/
ONLY_RUN = ""                  # "" = every run found on this box
DRY_RUN = False                # True = report what it would push, push nothing
PRIVATE = False                # these are research artifacts, not a release

import hashlib, json, os, re, sys
from pathlib import Path

MODEL_FILE = "model.safetensors"
READY_MARKER = ".ready"
SANDBOX_PREFIX = "chuk-job-"   # chuk-compute-worker/src/sandbox.rs
CKPT_SUBDIR = "ckpt"           # chuk-train-controlplane/src/jobspec.rs

# ── 1 · Hub credentials, BEFORE reading half a gigabyte off disk ─────────────
try:
    from google.colab import userdata
    os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
except Exception:
    if not os.environ.get("HF_TOKEN"):
        from huggingface_hub import notebook_login
        notebook_login()

from huggingface_hub import HfApi
from huggingface_hub.utils import RepositoryNotFoundError

api = HfApi()
who = api.whoami()
role = (who.get("auth") or {}).get("accessToken", {}).get("role")
print(f"Hub: {who.get('name')} (token role: {role or 'unknown'})")
if role == "read":
    raise SystemExit(
        "That HF_TOKEN is READ-ONLY. Mint a write token at "
        "https://huggingface.co/settings/tokens and update the Colab secret."
    )
owner = REPO_ID.split("/")[0]
if owner != who.get("name") and owner not in {o["name"] for o in who.get("orgs", [])}:
    raise SystemExit(
        f"Token belongs to {who.get('name')!r} but REPO_ID is under {owner!r}, "
        f"which is neither that account nor one of its orgs -- the push would 403."
    )

# Prove the token can write, rather than inferring it from `role`. A
# fine-grained token reports role "fineGrained" -- not "read" -- so the check
# above waves it through, and it then 403s on repo creation AFTER half a
# gigabyte has been read and hashed. Creating the repo is idempotent
# (exist_ok), costs one request, and is the only thing that actually answers
# "may this token write here".
if not DRY_RUN:
    try:
        api.create_repo(REPO_ID, repo_type="model", exist_ok=True, private=PRIVATE)
    except Exception as error:
        raise SystemExit(
            f"This token cannot write to {REPO_ID}:\n\n  {error}\n\n"
            f"If it is a FINE-GRAINED token, it needs BOTH permissions, and the\n"
            f"first is easy to miss because the repo does not exist yet:\n"
            f"  - Repositories -> 'Create' (or pre-create {REPO_ID} by hand)\n"
            f"  - Repositories -> 'Write' access to it\n"
            f"Edit it at https://huggingface.co/settings/tokens, then rerun this "
            f"cell.\nNothing has been uploaded and nothing on this box has been "
            f"touched, so rerunning is free."
        ) from None
    print(f"Hub: write access to {REPO_ID} confirmed")

# ── 2 · what is on this box ─────────────────────────────────────────────────
def complete_checkpoints(root: Path):
    """(run_id, step, dir) for every checkpoint the trainer finished writing.

    Sorted by run then step. A step dir without `.ready` was interrupted
    mid-write and is skipped rather than reported -- a half-written
    model.safetensors that uploads cleanly is the worst outcome available here,
    because it looks rescued.
    """
    found = []
    for sandbox in sorted(root.glob(f"{SANDBOX_PREFIX}*")):
        run_id = sandbox.name[len(SANDBOX_PREFIX):]
        if ONLY_RUN and run_id != ONLY_RUN:
            continue
        for d in sorted((sandbox / CKPT_SUBDIR).glob("step_*")):
            m = re.fullmatch(r"step_(\d+)", d.name)
            if m and (d / READY_MARKER).is_file() and (d / MODEL_FILE).is_file():
                found.append((run_id, int(m.group(1)), d))
    return sorted(found, key=lambda r: (r[0], r[1]))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


# The run's own terminal output, and the loss series behind it. Both are tiny
# next to a checkpoint and neither is recoverable once the runtime goes.
#
# train_replay.jsonl is the one that matters for the video: replay_run.py plays
# a FINISHED run back at camera speed, and this is its exact-timing source
# ({"t": seconds since start, "line": ...} per printed line). Without it, replay
# falls back to reconstructing timestamps from tokens-per-second in the log
# text. With it, what goes on screen is the run that actually produced these
# weights -- on the T4 that produced them, at its real pace.
RUN_ARTIFACTS = {
    "train_replay.jsonl": "unit/run_mathonly/train_replay.jsonl",
    "metrics.jsonl": "metrics.jsonl",
}


ckpts = complete_checkpoints(Path(SANDBOX_ROOT))
if not ckpts:
    raise SystemExit(
        f"No complete checkpoints under {SANDBOX_ROOT}/{SANDBOX_PREFIX}*/{CKPT_SUBDIR}/.\n"
        f"Either no run has reached its first save boundary on this runtime, or "
        f"this is not the notebook running the worker."
    )

runs = sorted({run for run, _, _ in ckpts})
print(f"\n{len(ckpts)} complete checkpoint(s) across {len(runs)} run(s):")
for run in runs:
    steps = [s for r, s, _ in ckpts if r == run]
    print(f"  {run}  steps {steps}")

# ── 3 · what the Hub already holds ───────────────────────────────────────────
# Files over ~10MB go to LFS, and an LFS object's oid IS the sha256 of the file.
# So "is this already up there, byte for byte?" is one metadata call, not a
# 460MB download per checkpoint.
published = {}
try:
    for info in api.list_repo_tree(REPO_ID, repo_type="model", recursive=True):
        lfs = getattr(info, "lfs", None)
        if lfs is not None:
            published[info.path] = getattr(lfs, "sha256", None) or lfs.get("sha256")
    print(f"\n{REPO_ID} already holds {len(published)} LFS file(s)")
except RepositoryNotFoundError:
    print(f"\n{REPO_ID} does not exist yet -- it will be created")

# ── 4 · push ────────────────────────────────────────────────────────────────
pushed, skipped, mismatched = [], [], []
for run_id, step, d in ckpts:
    path_in_repo = f"{run_id}/step_{step}"
    local_sha = sha256_file(d / MODEL_FILE)
    size_mb = (d / MODEL_FILE).stat().st_size / 1e6
    remote_sha = published.get(f"{path_in_repo}/{MODEL_FILE}")

    if remote_sha == local_sha:
        print(f"  {path_in_repo:52s} already published ({local_sha[:16]}…)")
        skipped.append(path_in_repo)
        continue
    if remote_sha is not None:
        # Same run, same step, different bytes. Two different training runs
        # cannot share a run id, so this means something is wrong upstream --
        # refuse the whole rescue rather than overwrite a research artifact.
        mismatched.append((path_in_repo, remote_sha, local_sha))
        continue

    meta = {}
    if (d / "meta.json").is_file():
        meta = json.loads((d / "meta.json").read_text())
    print(f"  {path_in_repo:52s} {size_mb:7.1f} MB  {local_sha[:16]}…  "
          f"{(meta.get('tokens') or 0)/1e6:.2f}M tok")
    if DRY_RUN:
        pushed.append(path_in_repo)
        continue

    # Staged outside the sandbox on purpose: the sandbox belongs to a job that
    # may still be running, and a rescue that mutates what it is rescuing is a
    # rescue you cannot repeat. The run id is the only thing that joins these
    # bytes back to a seed, a corpus identity and a base checkpoint, and
    # chuk-train is where those live -- so it is recorded rather than resolved.
    rescue = Path("/tmp") / f"rescue-{run_id}-step_{step}.json"
    rescue.write_text(json.dumps({
        "schema": "v11-mathonly-rescue-1",
        "run_id": run_id,
        "step": step,
        "model_sha256": local_sha,
        "meta": meta,
        "note": (
            "Rescued off the worker by training/colab_rescue_cell.py because the "
            "harness's checkpoint upload failed silently. Resolve run_id with "
            "chuk-train run_status for the seed, code-unit sha and timings."
        ),
    }, indent=2) + "\n")

    api.upload_folder(
        repo_id=REPO_ID, repo_type="model", folder_path=str(d),
        path_in_repo=path_in_repo,
        ignore_patterns=[READY_MARKER],
        commit_message=f"Rescue {run_id} step {step} ({local_sha[:16]})",
    )
    api.upload_file(
        repo_id=REPO_ID, repo_type="model", path_or_fileobj=str(rescue),
        path_in_repo=f"{path_in_repo}/rescue.json",
        commit_message=f"Rescue {run_id} step {step} provenance",
    )
    pushed.append(path_in_repo)

if mismatched:
    print("\nREFUSING -- the Hub already holds DIFFERENT bytes for:")
    for path_in_repo, remote, local in mismatched:
        print(f"  {path_in_repo}\n    published {remote}\n    on disk   {local}")
    raise SystemExit(
        "A run id identifies one training run, so the same step cannot honestly "
        "have two sets of weights. Resolve that before pushing anything else."
    )

# ── 4b · the run's log and metrics ──────────────────────────────────────────
# Discovered from the sandboxes rather than from the checkpoint list, so a
# re-run of this cell still rescues logs after the checkpoints are already up.
logs_pushed = []
for sandbox in sorted(Path(SANDBOX_ROOT).glob(f"{SANDBOX_PREFIX}*")):
    run_id = sandbox.name[len(SANDBOX_PREFIX):]
    if ONLY_RUN and run_id != ONLY_RUN:
        continue
    for name, rel in RUN_ARTIFACTS.items():
        src = sandbox / rel
        if not src.is_file() or src.stat().st_size == 0:
            continue
        path_in_repo = f"{run_id}/{name}"
        print(f"  {path_in_repo:52s} {src.stat().st_size/1e6:7.2f} MB  "
              f"{sum(1 for _ in src.open('rb')):,} lines")
        if DRY_RUN:
            logs_pushed.append(path_in_repo)
            continue
        api.upload_file(
            repo_id=REPO_ID, repo_type="model", path_or_fileobj=str(src),
            path_in_repo=path_in_repo,
            commit_message=f"Rescue {run_id} {name}",
        )
        logs_pushed.append(path_in_repo)

if logs_pushed:
    print(f"\nrescued {len(logs_pushed)} run artifact(s). Replay one on the Mac with:")
    print("  from huggingface_hub import hf_hub_download; import shutil, pathlib")
    print(f"  p = hf_hub_download({REPO_ID!r}, '<run id>/train_replay.jsonl')")
    print("  d = pathlib.Path('run_mathonly'); d.mkdir(exist_ok=True)")
    print("  shutil.copyfile(p, d / 'train_replay.jsonl')")
    print("  # then: uv run training/replay_run.py run_mathonly --speed 60 --max-gap 2")
else:
    print("\nNo train_replay.jsonl or metrics.jsonl found. If this run trained with "
          "--smoke, capture is deliberately off; otherwise the sandbox is gone.")

# ── 5 · verify what the Hub SERVES, not what we sent ────────────────────────
if DRY_RUN:
    print(f"\ndry run -- {len(pushed)} checkpoint(s) would be pushed, "
          f"{len(skipped)} already there. Nothing uploaded.")
else:
    serving = {}
    for info in api.list_repo_tree(REPO_ID, repo_type="model", recursive=True):
        lfs = getattr(info, "lfs", None)
        if lfs is not None:
            serving[info.path] = getattr(lfs, "sha256", None) or lfs.get("sha256")
    bad = []
    for run_id, step, d in ckpts:
        key = f"{run_id}/step_{step}/{MODEL_FILE}"
        if key in [f"{p}/{MODEL_FILE}" for p in pushed]:
            want = sha256_file(d / MODEL_FILE)
            if serving.get(key) != want:
                bad.append((key, serving.get(key), want))
    if bad:
        print("\nVERIFY FAILED -- the Hub is not serving what was sent:")
        for key, got, want in bad:
            print(f"  {key}\n    serving {got}\n    sent    {want}")
        raise SystemExit("Do not release this runtime; the rescue did not land.")
    print(f"\nrescued {len(pushed)} checkpoint(s), {len(skipped)} already published")
    print(f"https://huggingface.co/{REPO_ID}")

print("\nPull one back with:")
print("  from huggingface_hub import hf_hub_download")
print(f"  hf_hub_download({REPO_ID!r}, '<run id>/step_<n>/{MODEL_FILE}')")
print("\nRe-run the worker cell to rejoin the fleet and pick up the queue.")
