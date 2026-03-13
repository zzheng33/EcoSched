#!/usr/bin/env python3
"""
Run a single-node co-scheduling experiment:
  - ResNet50 training (ML/dl.py)
  - miniWeather benchmark (SPEC)

The script:
  1) Splits app budgets across each app's assigned GPUs
  2) Applies per-GPU power caps via geopmwrite
  3) Launches both applications concurrently
  4) Stores stdout/stderr logs under a timestamped run directory
  5) Resets touched GPUs to a default cap on exit
"""

import argparse
import json
import math
import os
import shlex
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


HOME = Path.home()
DEFAULT_ML_PYTHON = HOME / "env/ml/bin/python3"
DEFAULT_ML_DIR = HOME / "power/ML"
DEFAULT_DL_SCRIPT = DEFAULT_ML_DIR / "dl.py"
DEFAULT_MINIWEATHER_DIR = HOME / "benchmark/spec/miniWeather/cpp/build"
DEFAULT_OUT_ROOT = HOME / "power/GPGPU/data/H100/cosched_runs"
SPEC_ENV_SETUP_SHELL = (
    "source /etc/profile >/dev/null 2>&1 || true; "
    "source /etc/profile.d/modules.sh >/dev/null 2>&1 || true; "
    "module use /soft/modulefiles; "
    "module load cuda/12.3.0; "
    "module load cmake; "
    "module load gcc/12.2.0; "
    "module load openmpi/4.1.1-gcc; "
    "module load public_mkl/2019; "
    "export CUDA_DIR=/soft/compilers/cuda/cuda-12.3.0; "
    "export PCM_NO_MSR=1; "
    "export PCM_KEEP_NMI_WATCHDOG=1; "
)


def parse_gpu_csv(value: str) -> List[int]:
    items = [x.strip() for x in value.split(",") if x.strip()]
    if not items:
        raise argparse.ArgumentTypeError("GPU list cannot be empty")
    out: List[int] = []
    for item in items:
        try:
            g = int(item)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"Invalid GPU id: {item}") from exc
        if g < 0:
            raise argparse.ArgumentTypeError(f"GPU id must be non-negative: {item}")
        out.append(g)
    if len(set(out)) != len(out):
        raise argparse.ArgumentTypeError(f"Duplicate GPU ids are not allowed: {value}")
    return out


def run_checked(cmd: List[str], *, env: Optional[Dict[str, str]] = None, cwd: Optional[Path] = None) -> None:
    proc = subprocess.run(cmd, env=env, cwd=str(cwd) if cwd else None, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}")


def query_gpu_power_limits(gpu_id: int) -> Tuple[float, float]:
    cmd = [
        "nvidia-smi",
        "-i",
        str(gpu_id),
        "--query-gpu=power.min_limit,power.max_limit",
        "--format=csv,noheader,nounits",
    ]
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        check=False,
    )
    if proc.returncode != 0:
        # Conservative fallback for H100-like configs used in this repo.
        return 200.0, 700.0

    line = proc.stdout.strip().splitlines()[0]
    parts = [x.strip() for x in line.split(",")]
    if len(parts) != 2:
        return 200.0, 700.0
    try:
        return float(parts[0]), float(parts[1])
    except ValueError:
        return 200.0, 700.0


def set_gpu_cap(gpu_id: int, watts: int) -> None:
    cmd = [
        "geopmwrite",
        "NVML::GPU_POWER_LIMIT_CONTROL",
        "gpu",
        str(gpu_id),
        str(watts),
    ]
    run_checked(cmd)


def clamp_cap_for_gpu(gpu_id: int, requested_watts: float) -> Tuple[int, str]:
    min_w, max_w = query_gpu_power_limits(gpu_id)
    clamped = max(min_w, min(max_w, requested_watts))
    cap_w = int(round(clamped))
    note = ""
    if abs(cap_w - requested_watts) > 1e-6:
        note = (
            f"GPU{gpu_id}: requested {requested_watts:.2f}W, "
            f"clamped to {cap_w}W (limits {min_w:.1f}-{max_w:.1f}W)"
        )
    return cap_w, note


def build_run_dir(root: Path) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = root / f"resnet50_miniweather_{ts}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run co-scheduled ResNet50 + miniWeather experiment.")

    p.add_argument("--resnet-gpus", type=parse_gpu_csv, default=[0, 1, 2], help="GPU ids for ResNet50, e.g. 0,1,2")
    p.add_argument("--miniweather-gpus", type=parse_gpu_csv, default=[3], help="GPU ids for miniWeather, e.g. 3")

    p.add_argument("--resnet-budget-w", type=float, default=1000.0, help="Total GPU budget for ResNet50 (W)")
    p.add_argument("--miniweather-budget-w", type=float, default=1000.0, help="Total GPU budget for miniWeather (W)")
    p.add_argument("--resnet-numa", type=int, default=0, help="NUMA node id for ResNet50 CPU+memory binding")
    p.add_argument("--miniweather-numa", type=int, default=1, help="NUMA node id for miniWeather CPU+memory binding")

    p.add_argument("--reset-cap-w", type=int, default=700, help="Per-GPU cap to restore at the end")
    p.add_argument("--cap-settle-sec", type=float, default=0.5, help="Sleep after cap programming")

    p.add_argument("--ml-python", type=Path, default=DEFAULT_ML_PYTHON, help="Python executable for ML env")
    p.add_argument("--ml-script", type=Path, default=DEFAULT_DL_SCRIPT, help="Path to ML/dl.py")
    p.add_argument("--ml-workdir", type=Path, default=DEFAULT_ML_DIR, help="Working directory for ML run")
    p.add_argument("--ml-model", type=str, default="resnet50")
    p.add_argument("--ml-batch-size", type=int, default=8192)
    p.add_argument("--ml-epochs", type=int, default=3)
    p.add_argument("--ml-lr", type=float, default=0.001)

    p.add_argument(
        "--miniweather-workdir",
        type=Path,
        default=DEFAULT_MINIWEATHER_DIR,
        help="miniWeather build directory (contains ./parallelfor)",
    )
    p.add_argument("--miniweather-bin", type=str, default="./parallelfor", help="miniWeather executable path")

    p.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT, help="Directory for run logs")
    return p


def terminate_all(procs: Dict[str, subprocess.Popen]) -> None:
    for _, p in procs.items():
        if p.poll() is None:
            p.terminate()
    deadline = time.time() + 10.0
    while time.time() < deadline:
        if all(p.poll() is not None for p in procs.values()):
            return
        time.sleep(0.2)
    for _, p in procs.items():
        if p.poll() is None:
            p.kill()


def main() -> int:
    args = build_parser().parse_args()

    if shutil.which("geopmwrite") is None:
        raise RuntimeError("geopmwrite not found in PATH. Please load required modules first.")
    if shutil.which("numactl") is None:
        raise RuntimeError("numactl not found in PATH.")
    if not args.ml_python.exists():
        raise RuntimeError(f"ML python not found: {args.ml_python}")
    if not args.ml_script.exists():
        raise RuntimeError(f"ML script not found: {args.ml_script}")
    if not args.ml_workdir.exists():
        raise RuntimeError(f"ML workdir not found: {args.ml_workdir}")
    if not args.miniweather_workdir.exists():
        raise RuntimeError(f"miniWeather workdir not found: {args.miniweather_workdir}")

    overlap = set(args.resnet_gpus).intersection(set(args.miniweather_gpus))
    if overlap:
        raise RuntimeError(f"GPU overlap detected between apps: {sorted(overlap)}")

    resnet_per_gpu = args.resnet_budget_w / len(args.resnet_gpus)
    mini_per_gpu = args.miniweather_budget_w / len(args.miniweather_gpus)

    cap_plan: Dict[int, int] = {}
    clamp_notes: List[str] = []

    for g in args.resnet_gpus:
        cap, note = clamp_cap_for_gpu(g, resnet_per_gpu)
        cap_plan[g] = cap
        if note:
            clamp_notes.append(f"[ResNet50] {note}")
    for g in args.miniweather_gpus:
        cap, note = clamp_cap_for_gpu(g, mini_per_gpu)
        cap_plan[g] = cap
        if note:
            clamp_notes.append(f"[miniWeather] {note}")

    run_dir = build_run_dir(args.out_root)
    meta_path = run_dir / "run_meta.json"
    resnet_log = run_dir / "resnet50.log"
    miniweather_log = run_dir / "miniweather.log"

    metadata = {
        "resnet_gpus": args.resnet_gpus,
        "miniweather_gpus": args.miniweather_gpus,
        "resnet_budget_w": args.resnet_budget_w,
        "miniweather_budget_w": args.miniweather_budget_w,
        "resnet_numa": args.resnet_numa,
        "miniweather_numa": args.miniweather_numa,
        "resnet_per_gpu_requested_w": resnet_per_gpu,
        "miniweather_per_gpu_requested_w": mini_per_gpu,
        "cap_plan_w": {str(k): v for k, v in sorted(cap_plan.items())},
        "clamp_notes": clamp_notes,
        "ml_command": [
            str(args.ml_python),
            str(args.ml_script),
            "--model",
            args.ml_model,
            "--num-gpus",
            str(len(args.resnet_gpus)),
            "--batch-size",
            str(args.ml_batch_size),
            "--epochs",
            str(args.ml_epochs),
            "--lr",
            str(args.ml_lr),
        ],
    }
    meta_path.write_text(json.dumps(metadata, indent=2))

    print(f"[INFO] Run directory: {run_dir}")
    if clamp_notes:
        print("[WARN] Cap clamp notes:")
        for note in clamp_notes:
            print(f"  - {note}")
    print("[INFO] Applying caps:")
    for gpu_id in sorted(cap_plan):
        print(f"  GPU{gpu_id}: {cap_plan[gpu_id]}W")

    touched_gpus = sorted(cap_plan.keys())
    for gpu_id in touched_gpus:
        set_gpu_cap(gpu_id, cap_plan[gpu_id])
    time.sleep(args.cap_settle_sec)

    procs: Dict[str, subprocess.Popen] = {}
    start_ts = time.time()
    try:
        ml_env = os.environ.copy()
        ml_env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in args.resnet_gpus)
        ml_env["PYTHONUNBUFFERED"] = "1"

        ml_cmd = [
            "numactl",
            "--cpunodebind",
            str(args.resnet_numa),
            "--membind",
            str(args.resnet_numa),
            str(args.ml_python),
            str(args.ml_script),
            "--model",
            args.ml_model,
            "--num-gpus",
            str(len(args.resnet_gpus)),
            "--batch-size",
            str(args.ml_batch_size),
            "--epochs",
            str(args.ml_epochs),
            "--lr",
            str(args.ml_lr),
        ]

        mini_gpu_csv = ",".join(str(g) for g in args.miniweather_gpus)
        # Map each rank to one GPU from miniweather_gpus.
        rank_shell = (
            f'GPU_CSV="{mini_gpu_csv}"; '
            'IFS="," read -ra GPU_ARR <<< "$GPU_CSV"; '
            'GPU_ID="${GPU_ARR[$OMPI_COMM_WORLD_LOCAL_RANK]}"; '
            'if [[ -z "$GPU_ID" ]]; then echo "No GPU mapped for local rank $OMPI_COMM_WORLD_LOCAL_RANK"; exit 2; fi; '
            'export CUDA_VISIBLE_DEVICES="$GPU_ID"; '
            f'{args.miniweather_bin}'
        )
        mini_launch_shell = (
            SPEC_ENV_SETUP_SHELL
            + f"numactl --cpunodebind {args.miniweather_numa} --membind {args.miniweather_numa} "
            + f"mpirun -np {len(args.miniweather_gpus)} bash -lc {shlex.quote(rank_shell)} "
        )
        mini_cmd = [
            "bash",
            "-lc",
            mini_launch_shell,
        ]

        with resnet_log.open("w") as ml_fh, miniweather_log.open("w") as mini_fh:
            procs["resnet50"] = subprocess.Popen(
                ml_cmd,
                cwd=str(args.ml_workdir),
                env=ml_env,
                stdout=ml_fh,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
            )
            procs["miniweather"] = subprocess.Popen(
                mini_cmd,
                cwd=str(args.miniweather_workdir),
                env=os.environ.copy(),
                stdout=mini_fh,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
            )

            remaining = set(procs.keys())
            while remaining:
                for name in list(remaining):
                    rc = procs[name].poll()
                    if rc is not None:
                        print(f"[INFO] {name} exited with code {rc}")
                        remaining.remove(name)
                        if rc != 0:
                            print(f"[ERROR] {name} failed, terminating other process.")
                            terminate_all(procs)
                            return 1
                time.sleep(1.0)

        elapsed = time.time() - start_ts
        print(f"[INFO] Experiment completed successfully in {elapsed:.2f}s")
        print(f"[INFO] Logs: {resnet_log} , {miniweather_log}")
        return 0
    except KeyboardInterrupt:
        print("[WARN] Interrupted by user, terminating processes.")
        terminate_all(procs)
        return 130
    finally:
        print("[INFO] Resetting touched GPUs to default cap:", args.reset_cap_w)
        for gpu_id in touched_gpus:
            try:
                set_gpu_cap(gpu_id, args.reset_cap_w)
            except Exception as exc:  # noqa: BLE001
                print(f"[WARN] Failed to reset GPU{gpu_id}: {exc}")


if __name__ == "__main__":
    sys.exit(main())
