"""Shared configuration for co-scheduling scripts."""

from pathlib import Path

HOME = Path.home()
RESULTS_DIR = HOME / "power/GPGPU/coSched/results"
PERF_METRICS_FILE = HOME / "power/GPGPU/data/H100/perf_metrics.txt"
SCRIPT_DIR = HOME / "power/GPGPU/script"
SPEC_SCRIPT_DIR = SCRIPT_DIR / "run_benchmark/spec_script"
ECP_SCRIPT_DIR = SCRIPT_DIR / "run_benchmark/ecp_script"

TOTAL_GPUS = 4

# NUMA-to-GPU mapping
NUMA0_GPUS = [0, 1]
NUMA1_GPUS = [2, 3]

#  co-scheduling mode only (not sequential): depreciated
# Predicted optimal GPU counts from the notebook's EDP model
# (dram_sum proxy, w=0.0, alpha=1.30)
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
    'simpleMultiGPU', 'simpleP2P','streamOrderedAllocationP2P',
    "resnet101", "resnet152", "vgg19", "vgg16",
]
