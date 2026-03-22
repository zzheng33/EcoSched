"""Shared configuration for co-scheduling scripts."""

import os
from pathlib import Path

HOME = Path.home()

SYSTEM = os.environ.get("SYSTEM", "H100")

IDLE_POWER_PER_GPU = {"V100": 43.0, "A100": 53.0, "H100": 70.0}

# System-specific benchmark roots
SPEC_BENCHMARK_ROOT = {
    "V100": HOME / "benchmark/spec-V100",
    "A100": HOME / "benchmark/spec-A100",
    "H100": HOME / "benchmark/spec-H100",
}
CUDA_BUILD_DIR = {
    "V100": "build-sm70",
    "A100": "build-sm80",
    "H100": "build-sm90",
}

RESULTS_DIR = HOME / f"power/GPGPU/coSched/results/{SYSTEM}"
PERF_METRICS_FILE = HOME / f"power/GPGPU/data/{SYSTEM}/perf_metrics.txt"
SCRIPT_DIR = HOME / "power/GPGPU/script"
SPEC_SCRIPT_DIR = SCRIPT_DIR / "run_benchmark/spec_script"
ECP_SCRIPT_DIR = SCRIPT_DIR / "run_benchmark/ecp_script"
CUDA_SCRIPT_DIR = SCRIPT_DIR / "run_benchmark/cuda_script"

TOTAL_GPUS = 4

# NUMA-to-GPU mapping
NUMA0_GPUS = [0, 1]
NUMA1_GPUS = [2, 3]

# Environment setup for SPEC/CUDA benchmarks (MPI + CUDA modules)
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

# ---------------------------------------------------------------------------
# App categories — controls which command builder is used
# ---------------------------------------------------------------------------
SPEC_APPS = {
    'pot3d', 'minisweep', 'lbm', 'cloverleaf', 'tealeaf',
    'miniweather', 'hpgmg',
}

CUDA_APPS = {
    'conjugateGradientMultiDeviceCG', 'MonteCarloMultiGPU',
    'simpleCUBLASXT', 'simpleCUFFT_MGPU', 'simpleCUFFT_2d_MGPU',
    'simpleMultiGPU', 'simpleP2P', 'simpleIPC',
    'streamOrderedAllocationP2P', 'streamOrderedAllocationIPC',
}

TORCHRUN_APPS = {'bert', 'gpt2'}

ML_DL_APPS = {'resnet50', 'resnet101', 'resnet152', 'vgg16', 'vgg19'}

ML_PYTHON = HOME / "env/ml/bin/python3"
ML_SCRIPT = HOME / "power/ML/dl.py"
ML_WORKDIR = HOME / "power/ML"
ML_MIN_PER_GPU_CAP = 200
ML_MAX_PER_GPU_CAP = 700
ML_BATCH_SIZE = 2048
ML_BATCH_SIZE_OVERRIDE = {
    'V100': {'resnet50': 512, 'resnet101': 512, 'resnet152': 512, 'vgg19': 512,'vgg16': 512},
    'H100': {'resnet50': 2048, 'resnet101': 2048, 'resnet152': 1024, 'vgg19': 2048, 'vgg16': 2048},
}
ML_EPOCHS = 3
ML_LR = 0.001



# 'simpleCUBLASXT', 'simpleCUFFT_MGPU', 'simpleCUFFT_2d_MGPU' 'hpgmg' 'simpleMultiGPU'
DEFAULT_JOB_QUEUE = [
    'minisweep', 'lbm', 'cloverleaf', 'tealeaf',
    'miniweather', 'bert', 'gpt2', 'resnet50',
    'conjugateGradientMultiDeviceCG', 'pot3d','MonteCarloMultiGPU',
    'simpleP2P', 'streamOrderedAllocationP2P',
    "resnet101", "resnet152", "vgg19", "vgg16",
]

# DEFAULT_JOB_QUEUE = [
#     'gpt2'
# ]

# ---------------------------------------------------------------------------
# Workload presets for controlled mixed-workload studies.
# Inline comments show the best standalone GPU count from perf_metrics.txt.
# ---------------------------------------------------------------------------
WORKLOAD_PRESETS = {
    "V100": {
        "low_opp": [
            "cloverleaf",                    # best_gpu=4
            "lbm",                           # best_gpu=4
            "minisweep",                     # best_gpu=4
            "pot3d",                         # best_gpu=4
            "tealeaf",                       # best_gpu=4
            "bert",                          # best_gpu=4
            "gpt2",                          # best_gpu=4
            "resnet50",                      # best_gpu=4
            "simpleP2P",                     # best_gpu=2
            "streamOrderedAllocationP2P",    # best_gpu=2
        ],
        "med_opp": [
            "cloverleaf",                    # best_gpu=4
            "pot3d",                         # best_gpu=4
            "bert",                          # best_gpu=4
            "resnet50",                      # best_gpu=4
            "resnet152",                     # best_gpu=4
            "MonteCarloMultiGPU",            # best_gpu=1
            "vgg16",                         # best_gpu=3
            "MonteCarloMultiGPU",            # best_gpu=1
            "simpleP2P",                     # best_gpu=2
            "streamOrderedAllocationP2P",    # best_gpu=2
        ],
        "high_opp": [
            "MonteCarloMultiGPU",            # best_gpu=1
            "vgg16",                         # best_gpu=3
            "MonteCarloMultiGPU",            # best_gpu=1
            "simpleP2P",                     # best_gpu=2
            "streamOrderedAllocationP2P",    # best_gpu=2
            "simpleP2P",                     # best_gpu=2
            "vgg16",                         # best_gpu=3
            "simpleP2P",                     # best_gpu=2
            "vgg16",                         # best_gpu=3
            "streamOrderedAllocationP2P",    # best_gpu=2
        ],
    },
    "A100": {
        "low_opp": [
            "cloverleaf",                    # best_gpu=4
            "lbm",                           # best_gpu=4
            "minisweep",                     # best_gpu=4
            "pot3d",                         # best_gpu=4
            "tealeaf",                       # best_gpu=4
            "bert",                          # best_gpu=4
            "gpt2",                          # best_gpu=4
            "resnet50",                      # best_gpu=4
            "streamOrderedAllocationP2P",    # best_gpu=2
            "conjugateGradientMultiDeviceCG",# best_gpu=2
            
        ],
        "med_opp": [
            "cloverleaf",                    # best_gpu=4
            "lbm",                           # best_gpu=4
            "pot3d",                         # best_gpu=4
            "bert",                          # best_gpu=4
            "resnet50",                      # best_gpu=4
            "hpgmg",                         # best_gpu=1
            "resnet101",                     # best_gpu=2
            "vgg16",                         # best_gpu=1
            "MonteCarloMultiGPU",            # best_gpu=1
            "simpleP2P",                     # best_gpu=2
        ],
        "high_opp": [
            "resnet152",                     # best_gpu=2
            "simpleP2P",                     # best_gpu=2
            "resnet101",                     # best_gpu=2
            "resnet152",                     # best_gpu=2
            "vgg16",                         # best_gpu=1
            "vgg19",                         # best_gpu=1
            "MonteCarloMultiGPU",            # best_gpu=1
            "conjugateGradientMultiDeviceCG",# best_gpu=2
            "simpleP2P",                     # best_gpu=2
            "streamOrderedAllocationP2P",    # best_gpu=2
        ],
    },
    "H100": {
        "low_opp": [
            "cloverleaf",                    # best_gpu=4
            "lbm",                           # best_gpu=4
            "minisweep",                     # best_gpu=4
            "pot3d",                         # best_gpu=4
            "tealeaf",                       # best_gpu=4
            "bert",                          # best_gpu=4
            "resnet50",                      # best_gpu=4
            "resnet101",                     # best_gpu=4
            "miniweather",                   # best_gpu=1
            "gpt2",                          # best_gpu=3
        ],
        "med_opp": [
            "cloverleaf",                    # best_gpu=4
            "lbm",                           # best_gpu=4
            "pot3d",                         # best_gpu=4
            "bert",                          # best_gpu=4
            "resnet50",                      # best_gpu=4
            "hpgmg",                         # best_gpu=1
            "gpt2",                          # best_gpu=3
            "vgg16",                         # best_gpu=1
            "MonteCarloMultiGPU",            # best_gpu=1
            "simpleP2P",                     # best_gpu=2
        ],
        "high_opp": [
            "hpgmg",                         # best_gpu=1
            "miniweather",                   # best_gpu=1
            "gpt2",                          # best_gpu=3
            "streamOrderedAllocationP2P",    # best_gpu=2 
            "vgg16",                         # best_gpu=1
            "vgg19",                         # best_gpu=1
            "MonteCarloMultiGPU",            # best_gpu=1
            "simpleP2P",                     # best_gpu=2
            "streamOrderedAllocationP2P",    # best_gpu=2
            "gpt2",                          # best_gpu=3
        ],
    },
}
