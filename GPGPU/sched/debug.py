#!/usr/bin/env python3
"""
Debug: run a single benchmark with explicit GPU assignment.

Launches the real application with the specified CUDA_VISIBLE_DEVICES and
GPU count, using the same command construction as run_cosched.py.

Usage:
    python3 debug.py --app pot3d    --gpus 0,1   --gpu-count 2
    python3 debug.py --app bert     --gpus 2,3   --gpu-count 2
    python3 debug.py --app resnet50 --gpus 0,1,2 --gpu-count 3
    python3 debug.py --app gpt2     --gpus 2,3   --gpu-count 2 --numa 1
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

HOME = Path.home()
SCRIPT_DIR = HOME / "power/GPGPU/script"
SPEC_SCRIPT_DIR = SCRIPT_DIR / "run_benchmark/spec_script"
ECP_SCRIPT_DIR = SCRIPT_DIR / "run_benchmark/ecp_script"

SPEC_ENV_SETUP = (
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

TORCHRUN_APPS = {'bert', 'gpt2'}
ML_DL_APPS = {'resnet50'}

ML_PYTHON = HOME / "env/ml/bin/python3"
ML_SCRIPT = HOME / "power/ML/dl.py"
ML_WORKDIR = HOME / "power/ML"
ML_BATCH_SIZE = 2048
ML_EPOCHS = 3
ML_LR = 0.001

SPEC_APPS = {'pot3d', 'minisweep', 'lbm', 'cloverleaf', 'tealeaf',
             'miniweather', 'hpgmg'}


def build_command(app, gpu_ids, gpu_count, numa_node):
    """Build the launch command for the given app, same as run_cosched.py."""
    gpu_csv = ",".join(str(g) for g in gpu_ids)

    if app in SPEC_APPS:
        script_path = SPEC_SCRIPT_DIR / f"{app}.sh"
        shell_cmd = (
            SPEC_ENV_SETUP
            + f"export CUDA_VISIBLE_DEVICES={gpu_csv}; "
            + f"numactl --cpunodebind={numa_node} --membind={numa_node} "
            + f"bash {script_path} {gpu_count}"
        )
        cmd = ["bash", "-lc", shell_cmd]
        env = os.environ.copy()
        cwd = None

    elif app in TORCHRUN_APPS:
        app_dir = 'bert-large' if app == 'bert' else app
        shell_cmd = (
            f"export CUDA_VISIBLE_DEVICES={gpu_csv}; "
            f"export TOKENIZERS_PARALLELISM=false; "
            f"source {HOME}/env/ml/bin/activate; "
            f"cd {HOME}/benchmark/ECP/{app_dir}; "
            f"numactl --cpunodebind={numa_node} --membind={numa_node} "
            f"torchrun --nproc_per_node={gpu_count} training.py; "
            f"deactivate"
        )
        cmd = ["bash", "-lc", shell_cmd]
        env = os.environ.copy()
        cwd = None

    elif app in ML_DL_APPS:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_csv
        cmd = [
            "numactl",
            f"--cpunodebind={numa_node}",
            f"--membind={numa_node}",
            str(ML_PYTHON),
            str(ML_SCRIPT),
            "--model", app,
            "--num-gpus", str(gpu_count),
            "--batch-size", str(ML_BATCH_SIZE),
            "--epochs", str(ML_EPOCHS),
            "--lr", str(ML_LR),
        ]
        cwd = str(ML_WORKDIR)

    else:
        print(f"ERROR: unknown app '{app}'")
        sys.exit(1)

    return cmd, env, cwd


def main():
    all_apps = sorted(SPEC_APPS | TORCHRUN_APPS | ML_DL_APPS)

    parser = argparse.ArgumentParser(
        description="Debug: run a single benchmark with explicit GPU assignment")
    parser.add_argument("--app", type=str, required=True, choices=all_apps,
                        help="Benchmark name")
    parser.add_argument("--gpus", type=str, required=True,
                        help="Comma-separated GPU IDs, e.g. '0,1' or '2,3'")
    parser.add_argument("--gpu-count", type=int, required=True,
                        help="Number of GPUs to pass to the app")
    parser.add_argument("--numa", type=int, default=0, choices=[0, 1],
                        help="NUMA node for CPU/memory binding (default: 0)")

    args = parser.parse_args()

    gpu_ids = [int(g) for g in args.gpus.split(",")]

    print(f"App:                  {args.app}")
    print(f"CUDA_VISIBLE_DEVICES: {args.gpus}")
    print(f"GPU count to app:     {args.gpu_count}")
    print(f"NUMA node:            {args.numa}")
    print("=" * 60)

    cmd, env, cwd = build_command(args.app, gpu_ids, args.gpu_count, args.numa)
    print(f"Command: {' '.join(cmd)}")
    print("=" * 60)

    start = time.time()
    proc = subprocess.Popen(
        cmd, env=env, cwd=cwd,
    )
    rc = proc.wait()
    elapsed = time.time() - start

    print("=" * 60)
    print(f"Exit code: {rc}")
    print(f"Runtime:   {elapsed:.2f}s")


if __name__ == "__main__":
    main()
