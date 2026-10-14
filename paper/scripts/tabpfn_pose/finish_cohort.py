"""Run the latency stage once the existing cohort process has released the GPU."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--source",type=Path,required=True)
    p.add_argument("--model-path",type=Path,required=True)
    args=p.parse_args()
    deadline=time.monotonic()+3*3600
    pid=json.loads((args.root/"process.json").read_text())["pid"]
    while time.monotonic()<deadline:
        proc=Path(f"/proc/{pid}/cmdline")
        try:
            active=b"tabpfn_pose.cohort" in proc.read_bytes()
        except FileNotFoundError:
            active=False
        completed=(args.root/"results.json").exists() and json.loads((args.root/"results.json").read_text()).get("complete")
        if not active:
            if not completed:
                raise RuntimeError("Cohort stopped without a complete result; latency stage not run")
            break
        time.sleep(20)
    else:
        raise TimeoutError("Cohort did not complete within the bounded three-hour wait")
    if (args.root/"resource-benchmark.json").exists():
        print("Resource benchmark already complete",flush=True)
        return
    command=[sys.executable,"-u","-m","tabpfn_pose.benchmark_cohort",
        "--root",str(args.root.resolve()),"--source",str(args.source.resolve()),"--model-path",str(args.model_path.resolve())]
    print("Cohort finished; beginning isolated latency benchmark",flush=True)
    subprocess.run(command,check=True,cwd=Path(__file__).resolve().parents[1])
    print("Latency benchmark complete",flush=True)


if __name__=="__main__":
    main()
