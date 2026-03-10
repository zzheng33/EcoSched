#!/usr/bin/env python3
"""
Unified tenant runner (app launch only; no power-cap management).

Behavior:
  - Auto-detects app family from exp_power_cap.py app lists (case-sensitive).
  - Uses existing benchmark wrapper scripts for non-ML apps.
  - Uses ML/dl.py for resnet50 and vgg16.
  - Uses ECP scripts for bert and gpt2.

# ML (explicit GPUs allowed)
python3 /home/ac.zzheng/power/GPGPU/script/co-sched/tenant.py \
  --app resnet50 --gpus 1,2,3 --numa 0

# SPEC (must be prefix-from-0)
python3 /home/ac.zzheng/power/GPGPU/script/co-sched/tenant.py \
  --app miniweather --gpus 0,1,2 --numa 1

# Single-GPU app
python3 /home/ac.zzheng/power/GPGPU/script/co-sched/tenant.py \
  --app aobench --gpus 2 --numa 1

"""


import argparse
import os
import shlex
import shutil
import signal
import subprocess
import sys
from pathlib import Path


HOME = Path.home()
SCRIPT_ROOT = HOME / "power/GPGPU/script"
RUN_BENCH_ROOT = SCRIPT_ROOT / "run_benchmark"

DEFAULT_ML_PYTHON = HOME / "env/ml/bin/python3"
DEFAULT_ML_WORKDIR = HOME / "power/ML"
DEFAULT_ML_SCRIPT = DEFAULT_ML_WORKDIR / "dl.py"

TENANT1_BIND = [
    0, 80, 1, 81, 2, 82, 3, 83, 4, 84, 5, 85, 6, 86, 7, 87, 8, 88, 9, 89,
    10, 90, 11, 91, 12, 92, 13, 93, 14, 94, 15, 95, 16, 96, 17, 97, 18, 98,
    19, 99, 20, 100, 21, 101, 22, 102, 23, 103, 24, 104, 25, 105, 26, 106,
    27, 107, 28, 108, 29, 109, 30, 110, 31, 111, 32, 112, 33, 113, 34, 114,
    35, 115, 36, 116, 37, 117, 38, 118, 39, 119,
]

TENANT2_BIND = [
    40, 120, 41, 121, 42, 122, 43, 123, 44, 124, 45, 125, 46, 126, 47, 127,
    48, 128, 49, 129, 50, 130, 51, 131, 52, 132, 53, 133, 54, 134, 55, 135,
    56, 136, 57, 137, 58, 138, 59, 139, 60, 140, 61, 141, 62, 142, 63, 143,
    64, 144, 65, 145, 66, 146, 67, 147, 68, 148, 69, 149, 70, 150, 71, 151,
    72, 152, 73, 153, 74, 154, 75, 155, 76, 156, 77, 157, 78, 158, 79, 159,
]


# Keep these lists aligned with exp_power_cap.py (case-sensitive by design).
HEC_BENCHMARKS = [
    "addBiasResidualLayerNorm",
    "aobench",
    "background-subtract",
    "chacha20",
    "convolution3D",
    "dropout",
    "extrema",
    "fft",
    "kalman",
    "knn",
    "softmax",
    "stencil3d",
    "zmddft",
    "zoom",
]
ALTIS_BENCHMARKS_LEVEL = {
    "level0": ["maxflops"],
    "level1": ["bfs", "gemm", "gups", "pathfinder", "sort"],
    "level2": [
        "cfd",
        "cfd_double",
        "fdtd2d",
        "kmeans",
        "lavamd",
        "nw",
        "particlefilter_float",
        "particlefilter_naive",
        "raytracing",
        "srad",
        "where",
    ],
}
ECP_BENCHMARKS = [
    "XSBench",
    "miniGAN",
    "CRADL",
    "sw4lite",
    "Laghos",
    "bert",
    "UNet",
    "Resnet50",
    "lammps",
    "gromacs",
    "NAMD",
]
SPEC_BENCHMARKS = ["lbm", "cloverleaf", "tealeaf", "minisweep", "pot3d", "miniweather", "hpgmg"]
ML_MODELS = ["resnet50", "gpt2", "vgg16"]

DL_MODELS = {"resnet50", "vgg16"}

# Per your requirement:
# - multi-GPU apps: SPEC, ml_models, bert, gpt2 (up to 4 GPUs)
# - all others single-GPU only
MULTI_GPU_APPS = set(SPEC_BENCHMARKS) | set(ML_MODELS) | {"bert", "gpt2"}
ALL_APPS = (
    set(HEC_BENCHMARKS)
    | set(ECP_BENCHMARKS)
    | set(SPEC_BENCHMARKS)
    | set(ML_MODELS)
    | set(ALTIS_BENCHMARKS_LEVEL["level0"])
    | set(ALTIS_BENCHMARKS_LEVEL["level1"])
    | set(ALTIS_BENCHMARKS_LEVEL["level2"])
)
SINGLE_GPU_APPS = ALL_APPS - MULTI_GPU_APPS

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


def parse_gpu_csv(value):
    items = [x.strip() for x in value.split(",") if x.strip()]
    if not items:
        raise argparse.ArgumentTypeError("GPU list cannot be empty")
    out = []
    for item in items:
        try:
            gid = int(item)
        except ValueError as exc:
            raise argparse.ArgumentTypeError("Invalid GPU id: {}".format(item)) from exc
        if gid < 0:
            raise argparse.ArgumentTypeError("GPU id must be non-negative: {}".format(item))
        out.append(gid)
    if len(set(out)) != len(out):
        raise argparse.ArgumentTypeError("Duplicate GPU ids are not allowed: {}".format(value))
    return out


def build_parser():
    p = argparse.ArgumentParser(description="Unified tenant runner for ML/SPEC/ECP/ALTIS/HEC apps.")
    p.add_argument("--app", required=True, help="Application name (case-sensitive)")
    p.add_argument("--gpus", type=parse_gpu_csv, required=True, help="GPU ids, e.g. 0,1,2")
    p.add_argument("--numa", type=int, required=True, help="NUMA node id (required)")

    # ML (dl.py) options
    p.add_argument("--ml-python", type=Path, default=DEFAULT_ML_PYTHON, help="Python executable for ML env")
    p.add_argument("--ml-workdir", type=Path, default=DEFAULT_ML_WORKDIR, help="ML working directory")
    p.add_argument("--ml-script", type=Path, default=DEFAULT_ML_SCRIPT, help="Path to ML/dl.py")
    p.add_argument("--batch-size", type=int, default=8192, help="Batch size for dl.py models")
    p.add_argument("--epochs", type=int, default=3, help="Epochs for dl.py models")
    p.add_argument("--lr", type=float, default=0.001, help="Learning rate for dl.py models")
    return p


def validate_app_and_gpus(app, gpus):
    if app not in ALL_APPS:
        raise RuntimeError("Unknown app '{}'. Case-sensitive app list: {}".format(app, ", ".join(sorted(ALL_APPS))))

    if len(gpus) > 4:
        raise RuntimeError("Too many GPUs ({}). Max supported is 4.".format(len(gpus)))

    if app in SINGLE_GPU_APPS and len(gpus) != 1:
        raise RuntimeError("App '{}' is single-GPU only; received GPUs={}".format(app, gpus))

    if app in SPEC_BENCHMARKS:
        expected = list(range(len(gpus)))
        if gpus != expected:
            raise RuntimeError(
                "SPEC app '{}' requires prefix GPUs from 0 due wrapper mapping. "
                "Expected {}, got {}.".format(app, expected, gpus)
            )


def resolve_script_for_app(app):
    # ML models that run through dl.py are handled separately.
    if app in DL_MODELS:
        return None

    # Per requirement: bert/gpt2 route through ECP scripts.
    if app in {"bert", "gpt2"}:
        return RUN_BENCH_ROOT / "ecp_script" / "{}.sh".format(app)

    if app in SPEC_BENCHMARKS:
        return RUN_BENCH_ROOT / "spec_script" / "{}.sh".format(app)

    if app in HEC_BENCHMARKS:
        return RUN_BENCH_ROOT / "hec_script" / "{}.sh".format(app)

    if app in ECP_BENCHMARKS:
        return RUN_BENCH_ROOT / "ecp_script" / "{}.sh".format(app)

    for level, apps in ALTIS_BENCHMARKS_LEVEL.items():
        if app in apps:
            return RUN_BENCH_ROOT / "altis_script" / level / "{}.sh".format(app)

    # Should never hit if ALL_APPS is complete.
    raise RuntimeError("No script mapping found for app '{}'".format(app))


def resolve_physcpubind_from_numa(numa):
    if numa == 0:
        return ",".join(str(x) for x in TENANT1_BIND)
    if numa == 1:
        return ",".join(str(x) for x in TENANT2_BIND)
    raise RuntimeError(
        "Unsupported --numa {} for predefined tenant bindings. Use 0 (tenant1) or 1 (tenant2).".format(numa)
    )


def build_ml_cmd(args):
    if shutil.which("numactl") is None:
        raise RuntimeError("numactl not found in PATH.")
    if not args.ml_python.exists():
        raise RuntimeError("ML python not found: {}".format(args.ml_python))
    if not args.ml_script.exists():
        raise RuntimeError("ML script not found: {}".format(args.ml_script))
    if not args.ml_workdir.exists():
        raise RuntimeError("ML workdir not found: {}".format(args.ml_workdir))

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in args.gpus)
    env["PYTHONUNBUFFERED"] = "1"
    physcpubind = resolve_physcpubind_from_numa(args.numa)

    cmd = [
        "numactl",
        "--physcpubind",
        physcpubind,
        "--membind",
        str(args.numa),
        str(args.ml_python),
        str(args.ml_script),
        "--model",
        args.app,
        "--num-gpus",
        str(len(args.gpus)),
        "--batch-size",
        str(args.batch_size),
        "--epochs",
        str(args.epochs),
        "--lr",
        str(args.lr),
    ]
    return cmd, env, str(args.ml_workdir)


def build_script_cmd(args):
    script_path = resolve_script_for_app(args.app)
    if script_path is None:
        raise RuntimeError("Internal mapping error for app '{}'".format(args.app))
    if not script_path.exists():
        raise RuntimeError("Wrapper script not found for app '{}': {}".format(args.app, script_path))

    if shutil.which("numactl") is None:
        raise RuntimeError("numactl not found in PATH.")

    physcpubind = resolve_physcpubind_from_numa(args.numa)
    gpu_csv = ",".join(str(g) for g in args.gpus)
    launch_shell = (
        SPEC_ENV_SETUP_SHELL
        + "export CUDA_VISIBLE_DEVICES={}; ".format(shlex.quote(gpu_csv))
        + "numactl --physcpubind {} ".format(shlex.quote(physcpubind))
        + "bash {} {} ".format(shlex.quote(str(script_path)), len(args.gpus))
    )
    cmd = ["bash", "-lc", launch_shell]
    return cmd, None, None


def main():
    args = build_parser().parse_args()
    validate_app_and_gpus(args.app, args.gpus)

    if args.app in DL_MODELS:
        cmd, env, cwd = build_ml_cmd(args)
    else:
        cmd, env, cwd = build_script_cmd(args)

    proc = None

    try:
        physcpubind = resolve_physcpubind_from_numa(args.numa)
        print("[INFO] App: {}".format(args.app))
        print("[INFO] GPUs: {}".format(args.gpus))
        print("[INFO] NUMA: {}".format(args.numa))
        print("[INFO] physcpubind: {}".format(physcpubind))
        print("[INFO] Starting command:")
        print("       {}".format(" ".join(cmd)))
        proc = subprocess.Popen(cmd, env=env, cwd=cwd)
        return proc.wait()
    except KeyboardInterrupt:
        print("[WARN] Interrupted by user.")
        if proc and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
        return 130


if __name__ == "__main__":
    sys.exit(main())
