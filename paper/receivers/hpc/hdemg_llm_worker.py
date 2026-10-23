"""Consume the finite Qwen campaign on one GPU, recording use and aborting stalls."""

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(path)


def gpu_sample(uuid=None):
    command = ["nvidia-smi", "--query-gpu=uuid,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw", "--format=csv,noheader,nounits"]
    if uuid:
        command += ["--id=" + uuid]
    lines = subprocess.check_output(command, text=True, timeout=20).strip().splitlines()
    if len(lines) != 1:
        raise RuntimeError("Select exactly one GPU UUID when reusing an existing allocation")
    values = [s.strip() for s in lines[0].split(",")]
    return dict(zip(("uuid", "gpu_percent", "memory_percent", "memory_used_mib", "memory_total_mib", "power_watts"),
                    [values[0], *map(float, values[1:])]))


def cell_path(root, cell):
    subject, seed, arm = cell
    return root / f"subject_{subject:02d}/fits/{arm}_{seed}"


def stop_children(active):
    for process, _log, _start in active.values():
        if process.poll() is None:
            process.terminate()
    deadline = time.monotonic() + 120
    for process, log, _start in active.values():
        try:
            process.wait(timeout=max(1, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        log.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--workers", type=int, choices=(1, 2, 4, 6, 8), required=True)
    parser.add_argument("--gpu-uuid")
    parser.add_argument("--max-seconds", type=float, required=True)
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()
    if args.max_seconds < 300:
        raise ValueError("Insufficient time for a checkpointed run")
    lock = (args.root / "worker.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    protocol = json.loads((args.root / "protocol.json").read_text())
    cells = protocol["cells"]
    assert len(cells) == 108
    runner = "paper.receivers.run_hdemg_llm_full"
    repo = Path(__file__).resolve().parents[3]  # -m needs the repository root as working directory
    # --reference/--matched/--pilot use the runner's defaults under paper/results.
    common = ["--root", str(args.root)]
    sample = gpu_sample(args.gpu_uuid)
    if sample["memory_used_mib"] > 1200:
        raise RuntimeError(f"GPU already occupied; refusing to share: {sample}")
    uuid = sample["uuid"]
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=uuid, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    monitor_path = args.root / "monitoring" / f"{run_id}.jsonl"
    monitor_path.parent.mkdir(exist_ok=True)
    active = {}
    stopping = False

    def request_stop(_sig, _frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    start = time.monotonic()
    deadline = start + args.max_seconds - 150
    phase, last_sample = "fit", 0.
    measures = []
    failure = None
    try:
        with monitor_path.open("a", buffering=1) as monitor:
            while not stopping and time.monotonic() < deadline:
                suffix = "fit.json" if phase == "fit" else "test.json"
                for index, (process, log, _started) in list(active.items()):
                    code = process.poll()
                    if code is not None:
                        log.close()
                        del active[index]
                        if code:
                            raise RuntimeError(f"{phase} cell {index} exited with {code}; see {log.name}")
                pending = [i for i, cell in enumerate(cells) if not (cell_path(args.root, cell) / suffix).exists()]
                if not pending and not active:
                    if phase == "fit":
                        write(args.root / "worker_status.json", {"phase": "test_gate", "fits_complete": 108})
                        subprocess.run([args.python, "-m", runner, "gate", *common], cwd=repo, env=environment, check=True, timeout=600)
                        phase = "test"
                        continue
                    subprocess.run([args.python, "-m", runner, "report", *common], cwd=repo, env=environment, check=True, timeout=300)
                    phase = "complete"
                    break
                for index in pending:
                    if index in active:
                        continue
                    if len(active) >= args.workers:
                        break
                    out = cell_path(args.root, cells[index])
                    out.mkdir(parents=True, exist_ok=True)
                    log = (out / f"{phase}-{run_id}.log").open("a", buffering=1)
                    process = subprocess.Popen([args.python, "-u", "-m", runner, phase, *common, "--index", str(index)],
                                               cwd=repo, stdout=log, stderr=subprocess.STDOUT, env=environment)
                    active[index] = (process, log, time.time())
                now = time.monotonic()
                if now - last_sample >= 30:
                    sample = gpu_sample(uuid)
                    measures.append(sample)
                    progress = {}
                    for index, (process, log, launched) in active.items():
                        path = cell_path(args.root, cells[index]) / "progress.json"
                        heartbeat = max(launched, Path(log.name).stat().st_mtime,
                                        path.stat().st_mtime if path.exists() else 0)
                        if time.time() - heartbeat > 300:
                            raise RuntimeError(f"Cell {index} has made no progress for five minutes")
                        progress[str(index)] = {"pid": process.pid, "step": json.loads(path.read_text())["step"] if path.exists() else 0}
                    status = {"timestamp": datetime.now(timezone.utc).isoformat(), "phase": phase,
                              "fits_complete": sum((cell_path(args.root, c) / "fit.json").exists() for c in cells),
                              "tests_complete": sum((cell_path(args.root, c) / "test.json").exists() for c in cells),
                              "workers": args.workers, "active": progress, "gpu": sample,
                              "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "slurm_step_id": os.environ.get("SLURM_STEP_ID")}
                    monitor.write(json.dumps(status) + "\n")
                    write(args.root / "worker_status.json", status)
                    print(json.dumps(status), flush=True)
                    last_sample = now
                time.sleep(2)
    except BaseException as exc:
        failure = repr(exc)
        raise
    finally:
        stop_children(active)
        receipt = {"phase": phase, "failure": failure, "wall_seconds": time.monotonic() - start,
                   "workers": args.workers, "gpu_uuid": uuid, "samples": len(measures),
                   "mean_gpu_percent": sum(m["gpu_percent"] for m in measures) / len(measures) if measures else None,
                   "peak_memory_mib": max((m["memory_used_mib"] for m in measures), default=0),
                   "ended_at": datetime.now(timezone.utc).isoformat()}
        write(monitor_path.with_suffix(".summary.json"), receipt)
        write(args.root / "worker_status.json", receipt)
    return 0 if phase == "complete" else 75


if __name__ == "__main__":
    raise SystemExit(main())
