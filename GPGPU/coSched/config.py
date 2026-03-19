"""Shared configuration for co-scheduling scripts."""

from pathlib import Path

HOME = Path.home()
RESULTS_DIR = HOME / "power/GPGPU/coSched/results"
PERF_METRICS_FILE = HOME / "power/GPGPU/data/H100/perf_metrics.txt"
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
ML_BATCH_SIZE = 2048
ML_EPOCHS = 3
ML_LR = 0.001

# ---------------------------------------------------------------------------
# Co-scheduling mode only (not sequential): deprecated
# Predicted optimal GPU counts from the notebook's EDP model
# (dram_sum proxy, w=0.0, alpha=1.30)
# ---------------------------------------------------------------------------
PREDICTED_GPU_COUNTS = {
    'pot3d':       1,
    'minisweep':   4,
    'lbm':         4,
    'cloverleaf':  4,
    'tealeaf':     4,
    'miniweather': 1,
    'hpgmg':       2,
    'bert':        4,
    'gpt2':        3,
    'resnet50':    3,
}

# 'simpleCUBLASXT', 'simpleCUFFT_MGPU', 'simpleCUFFT_2d_MGPU'
DEFAULT_JOB_QUEUE = [
    'pot3d', 'minisweep', 'lbm', 'cloverleaf', 'tealeaf',
    'miniweather', 'bert', 'gpt2', 'resnet50', 'hpgmg',
    'conjugateGradientMultiDeviceCG', 'MonteCarloMultiGPU',
    'simpleMultiGPU', 'simpleP2P', 'streamOrderedAllocationP2P',
    "resnet101", "resnet152", "vgg19", "vgg16",
]
