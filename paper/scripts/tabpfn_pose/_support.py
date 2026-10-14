"""Helpers shared with the other TabPFN modules: file hashing, split-separation check, GPU monitor."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import threading
import time


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def validate_separation(windows, samples=6000):
    """Reject duplicated/overlapping parent windows across train/val/test."""
    grouped = {}
    for row in windows:
        key = int(row["subject"]), int(row["task"])
        grouped.setdefault(key, []).append((int(row["start"]), row["split"]))
    for key, values in grouped.items():
        values.sort()
        for (a, sa), (b, sb) in zip(values, values[1:]):
            if b < a + samples:
                raise ValueError(f"Overlapping parent windows: {key}, {a}/{sa}, {b}/{sb}")


class Monitor:
    def __init__(self, path, interval=10):
        self.path, self.interval = Path(path), interval
        self.stop = threading.Event()
        self.phase = "initializing"
        self.rows = 0

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        return self

    def _loop(self):
        import torch
        while not self.stop.is_set():
            row = dict(time=time.time(), pid=os.getpid(), phase=self.phase,
                       slurm_job_id=os.environ.get("SLURM_JOB_ID"))
            try:
                if torch.cuda.is_available():
                    props = torch.cuda.get_device_properties(0)
                    uuid = str(props.uuid)
                    if not uuid.startswith("GPU-"):
                        uuid = "GPU-" + uuid
                    output = subprocess.check_output([
                        "nvidia-smi", "--id=" + uuid,
                        "--query-gpu=uuid,utilization.gpu,memory.used,memory.total,power.draw",
                        "--format=csv,noheader,nounits"], text=True, timeout=5).strip()
                    values = [v.strip() for v in output.split(",")]
                    row.update(gpu_uuid=values[0], gpu_utilization_percent=float(values[1]),
                               memory_used_mib=float(values[2]), memory_total_mib=float(values[3]),
                               power_watts=float(values[4]))
            except Exception as exc:
                row["monitor_error"] = str(exc)
            with self.path.open("a") as stream:
                stream.write(json.dumps(row) + "\n")
            self.rows += 1
            self.stop.wait(self.interval)

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join(timeout=6)
