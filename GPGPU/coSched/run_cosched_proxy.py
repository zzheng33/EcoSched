#!/usr/bin/env python3
"""
Order-only proxy scheduler for co-scheduling benchmarks on 4 H100 GPUs.

This launcher leaves the existing schedulers untouched. It uses fixed GPU counts
from the notebook prediction and uses the proxy prediction only to decide the
execution order after the first queue-head launch.
"""

import argparse
import math
import os
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

HOME = Path.home()
SCRIPT_DIR = HOME / "power/GPGPU/script"
SPEC_SCRIPT_DIR = SCRIPT_DIR / "run_benchmark/spec_script"

TOTAL_GPUS = 4
NUMA0_GPUS = [0, 1]
NUMA1_GPUS = [2, 3]
PROFILE_METRICS_PATH = HOME / "power/GPGPU/data/H100/edp_metrics.txt"
ACTIVITY_ALPHA = 1.30

DEFAULT_JOB_QUEUE = [
    "pot3d", "minisweep", "lbm", "cloverleaf", "tealeaf",
    "miniweather", "bert", "gpt2", "resnet50", "hpgmg",
]

FIXED_GPU_COUNTS = {
    "pot3d": 1,
    "minisweep": 4,
    "lbm": 4,
    "cloverleaf": 4,
    "tealeaf": 4,
    "miniweather": 1,
    "hpgmg": 2,
    "bert": 4,
    "gpt2": 3,
    "resnet50": 3,
}

TORCHRUN_APPS = {"bert", "gpt2"}
ML_DL_APPS = {"resnet50"}

ML_PYTHON = HOME / "env/ml/bin/python3"
ML_SCRIPT = HOME / "power/ML/dl.py"
ML_WORKDIR = HOME / "power/ML"
ML_BATCH_SIZE = 2048
ML_EPOCHS = 3
ML_LR = 0.001

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


@dataclass
class ProfileOption:
    gpu_count: int
    performance: float
    avg_power: float
    dram_sum: float
    sm_sum: float
    norm_rt: float = 0.0
    norm_edp: float = 0.0


@dataclass
class PlannedLaunch:
    app: str
    gpu_count: int
    numa_node: int
    gpu_ids: List[int]
    proxy_score: float


@dataclass
class RunningJob:
    app: str
    gpu_ids: List[int]
    numa_node: int
    process: subprocess.Popen
    start_time: float
    proxy_score: float
    stdout_sink: object


class PowerMonitor:
    def __init__(self, num_gpus: int = TOTAL_GPUS, interval: float = 0.3):
        self.num_gpus = num_gpus
        self.interval = interval
        self._samples = []
        self._stop = threading.Event()
        self._thread = None

    def _query_power(self):
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
                encoding="utf-8",
                stderr=subprocess.DEVNULL,
            )
            return [float(x.strip()) for x in out.strip().split("\n")]
        except Exception:
            return [0.0] * self.num_gpus

    def _run(self):
        while not self._stop.is_set():
            t = time.time()
            powers = self._query_power()
            self._samples.append((t, powers))
            self._stop.wait(self.interval)

    def start(self):
        self._samples.clear()
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def total_energy_j(self):
        if len(self._samples) < 2:
            return 0.0
        total = 0.0
        for i in range(1, len(self._samples)):
            t0, p0 = self._samples[i - 1]
            t1, p1 = self._samples[i]
            dt = t1 - t0
            avg_power = sum((a + b) / 2.0 for a, b in zip(p0, p1))
            total += avg_power * dt
        return total

    def print_summary(self):
        energy_j = self.total_energy_j()
        n = len(self._samples)
        duration = self._samples[-1][0] - self._samples[0][0] if n >= 2 else 0.0
        print(f"\nPower monitoring: {n} samples over {duration:.1f}s")
        print(f"Total GPU energy: {energy_j:.2f} J ({energy_j / 1000.0:.2f} kJ)")


def _safe_float(text: str) -> float:
    value = float(text)
    if math.isnan(value):
        return 0.0
    return value


def load_fixed_proxy_scores(metrics_path: Path, fixed_gpu_counts: Dict[str, int]) -> Dict[str, ProfileOption]:
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing profile metrics file: {metrics_path}")

    section_re = re.compile(r"^===== .*?/([^/ ]+) =====$")
    current_app = None
    rows: Dict[str, List[ProfileOption]] = {}

    for raw_line in metrics_path.read_text().splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = section_re.match(line)
        if match:
            current_app = match.group(1)
            rows[current_app] = []
            continue
        if current_app is None or line.startswith("cap=") or line.startswith("gpu_count"):
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        rows[current_app].append(
            ProfileOption(
                gpu_count=int(parts[0]),
                performance=_safe_float(parts[1]),
                avg_power=_safe_float(parts[2]),
                dram_sum=_safe_float(parts[3]),
                sm_sum=_safe_float(parts[4]),
            )
        )

    selected: Dict[str, ProfileOption] = {}
    for app, required_gpu_count in fixed_gpu_counts.items():
        options = rows.get(app, [])
        if not options:
            continue
        max_activity = max(option.dram_sum for option in options)
        if max_activity <= 0.0:
            max_activity = 1.0
        for option in options:
            activity = max(option.dram_sum, 1e-9)
            option.norm_rt = (max_activity / activity) ** ACTIVITY_ALPHA
            option.norm_edp = option.norm_rt * option.avg_power * option.gpu_count
        matched = next((option for option in options if option.gpu_count == required_gpu_count), None)
        if matched is None:
            raise ValueError(f"No profile row for {app} at {required_gpu_count} GPUs")
        selected[app] = matched
    return selected


def allocate_gpus_numa(needed: int, tenant_numa: int, gpus_in_use: set) -> Optional[List[int]]:
    if tenant_numa == 0:
        local_gpus = [gpu for gpu in NUMA0_GPUS if gpu not in gpus_in_use]
        remote_gpus = [gpu for gpu in NUMA1_GPUS if gpu not in gpus_in_use]
    else:
        local_gpus = [gpu for gpu in NUMA1_GPUS if gpu not in gpus_in_use]
        remote_gpus = [gpu for gpu in NUMA0_GPUS if gpu not in gpus_in_use]
    available = local_gpus + remote_gpus
    if len(available) >= needed:
        return available[:needed]
    return None


def remote_gpu_count(gpu_ids: Sequence[int], tenant_numa: int) -> int:
    local = set(NUMA0_GPUS if tenant_numa == 0 else NUMA1_GPUS)
    return sum(1 for gpu in gpu_ids if gpu not in local)


def choose_best_numa(needed: int, gpus_in_use: set) -> Optional[Tuple[int, List[int]]]:
    candidates = []
    for numa_node in (0, 1):
        gpu_ids = allocate_gpus_numa(needed, numa_node, gpus_in_use)
        if gpu_ids is None:
            continue
        candidates.append((remote_gpu_count(gpu_ids, numa_node), numa_node, gpu_ids))
    if not candidates:
        return None
    _, numa_node, gpu_ids = min(candidates, key=lambda item: (item[0], item[1]))
    return numa_node, gpu_ids


def build_spec_command(app: str, gpu_ids: List[int], numa_node: int):
    gpu_count = len(gpu_ids)
    gpu_csv = ",".join(str(gpu) for gpu in gpu_ids)
    script_path = SPEC_SCRIPT_DIR / f"{app}.sh"
    shell_cmd = (
        SPEC_ENV_SETUP
        + f"export CUDA_VISIBLE_DEVICES={gpu_csv}; "
        + f"numactl --cpunodebind={numa_node} --membind={numa_node} "
        + f"bash {script_path} {gpu_count}"
    )
    return ["bash", "-lc", shell_cmd], os.environ.copy(), None


def build_torchrun_command(app: str, gpu_ids: List[int], numa_node: int):
    gpu_count = len(gpu_ids)
    gpu_csv = ",".join(str(gpu) for gpu in gpu_ids)
    shell_cmd = (
        f"export CUDA_VISIBLE_DEVICES={gpu_csv}; "
        f"export TOKENIZERS_PARALLELISM=false; "
        f"source {HOME}/env/ml/bin/activate; "
        f"cd {HOME}/benchmark/ECP/{app if app != 'bert' else 'bert-large'}; "
        f"numactl --cpunodebind={numa_node} --membind={numa_node} "
        f"torchrun --nproc_per_node={gpu_count} training.py; "
        f"deactivate"
    )
    return ["bash", "-lc", shell_cmd], os.environ.copy(), None


def build_ml_dl_command(app: str, gpu_ids: List[int], numa_node: int):
    gpu_count = len(gpu_ids)
    gpu_csv = ",".join(str(gpu) for gpu in gpu_ids)
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
    return cmd, env, ML_WORKDIR


def build_command(app: str, gpu_ids: List[int], numa_node: int):
    if app in TORCHRUN_APPS:
        return build_torchrun_command(app, gpu_ids, numa_node)
    if app in ML_DL_APPS:
        return build_ml_dl_command(app, gpu_ids, numa_node)
    return build_spec_command(app, gpu_ids, numa_node)


def _devnull():
    return open(os.devnull, "w")


def make_launch(app: str, fixed_profiles: Dict[str, ProfileOption], gpus_in_use: set) -> Optional[PlannedLaunch]:
    profile = fixed_profiles[app]
    best = choose_best_numa(profile.gpu_count, gpus_in_use)
    if best is None:
        return None
    numa_node, gpu_ids = best
    return PlannedLaunch(
        app=app,
        gpu_count=profile.gpu_count,
        numa_node=numa_node,
        gpu_ids=gpu_ids,
        proxy_score=profile.norm_edp,
    )


def pick_lowest_proxy_job(pending: Sequence[str], fixed_profiles: Dict[str, ProfileOption], gpus_in_use: set) -> Optional[PlannedLaunch]:
    feasible = []
    for app in pending:
        launch = make_launch(app, fixed_profiles, gpus_in_use)
        if launch is not None:
            feasible.append(launch)
    if not feasible:
        return None
    return min(feasible, key=lambda launch: (launch.proxy_score, launch.gpu_count, pending.index(launch.app)))


def run_sequential(job_queue: List[str], fixed_profiles: Dict[str, ProfileOption]):
    monitor = PowerMonitor()
    monitor.start()

    completed = []
    wall_start = time.time()

    print(f"Sequential execution: {len(job_queue)} apps on {TOTAL_GPUS} GPUs")
    print(f"NUMA 0 GPUs: {NUMA0_GPUS}  |  NUMA 1 GPUs: {NUMA1_GPUS}")
    print("=" * 80)

    for app in job_queue:
        launch = make_launch(app, fixed_profiles, set())
        if launch is None:
            raise RuntimeError(f"Unable to allocate GPUs for {app}")

        cmd, env, cwd = build_command(app, launch.gpu_ids, launch.numa_node)
        elapsed = time.time() - wall_start
        print(
            f"  t={elapsed:8.2f}s | START {app:<15} | "
            f"{launch.gpu_count} GPUs {launch.gpu_ids} | NUMA {launch.numa_node} | proxy={launch.proxy_score:.2f}"
        )

        app_start = time.time()
        devnull = _devnull()
        proc = subprocess.Popen(
            cmd,
            env=env,
            cwd=str(cwd) if cwd else None,
            stdout=devnull,
            stderr=subprocess.STDOUT,
        )
        rc = proc.wait()
        runtime = time.time() - app_start
        devnull.close()

        elapsed = time.time() - wall_start
        status = "OK" if rc == 0 else f"FAILED(rc={rc})"
        print(f"  t={elapsed:8.2f}s | END   {app:<15} | runtime={runtime:.2f}s | {status}")
        completed.append((app, launch.gpu_count, launch.gpu_ids, launch.numa_node, runtime, rc, launch.proxy_score))

    total_time = time.time() - wall_start
    print("\n" + "=" * 80)
    print("Sequential summary:")
    print(f"{'App':<15} {'#GPUs':>6} {'GPU IDs':>12} {'NUMA':>5} {'Runtime (s)':>12} {'Proxy':>10} {'Status':>10}")
    print("-" * 90)
    for app, gpu_count, gpu_ids, numa_node, runtime, rc, proxy_score in completed:
        status = "OK" if rc == 0 else "FAILED"
        print(
            f"{app:<15} {gpu_count:>6} {str(gpu_ids):>12} {numa_node:>5} "
            f"{runtime:>12.2f} {proxy_score:>10.2f} {status:>10}"
        )
    print("-" * 90)
    print(f"\nTotal makespan: {total_time:.2f}s")
    monitor.stop()
    monitor.print_summary()


def run_proxy(job_queue: List[str], fixed_profiles: Dict[str, ProfileOption], max_concurrent: int, metrics_path: Path):
    monitor = PowerMonitor()
    pending = list(job_queue)
    running: List[RunningJob] = []
    completed = []
    gpus_in_use: set = set()
    wall_start = time.time()
    first_launch_done = False

    print(f"Proxy co-scheduling {len(pending)} apps on {TOTAL_GPUS} GPUs (max {max_concurrent} concurrent)")
    print(f"NUMA 0 GPUs: {NUMA0_GPUS}  |  NUMA 1 GPUs: {NUMA1_GPUS}")
    print(f"Metrics source: {metrics_path}")
    print("Fixed GPU counts from notebook prediction; proxy used only for ordering")
    print("=" * 80)

    monitor.start()

    while pending or running:
        while pending and len(running) < max_concurrent:
            if not first_launch_done:
                launch = make_launch(pending[0], fixed_profiles, gpus_in_use)
                if launch is None:
                    break
                first_launch_done = True
            else:
                launch = pick_lowest_proxy_job(pending, fixed_profiles, gpus_in_use)
                if launch is None:
                    break

            cmd, env, cwd = build_command(launch.app, launch.gpu_ids, launch.numa_node)
            devnull = _devnull()
            elapsed = time.time() - wall_start
            print(
                f"  t={elapsed:8.2f}s | START {launch.app:<15} | "
                f"{launch.gpu_count} GPUs {launch.gpu_ids} | NUMA {launch.numa_node} | proxy={launch.proxy_score:.2f}"
            )
            proc = subprocess.Popen(
                cmd,
                env=env,
                cwd=str(cwd) if cwd else None,
                stdout=devnull,
                stderr=subprocess.STDOUT,
            )
            running.append(
                RunningJob(
                    app=launch.app,
                    gpu_ids=launch.gpu_ids,
                    numa_node=launch.numa_node,
                    process=proc,
                    start_time=time.time(),
                    proxy_score=launch.proxy_score,
                    stdout_sink=devnull,
                )
            )
            gpus_in_use.update(launch.gpu_ids)
            pending.remove(launch.app)

        if not running:
            if pending:
                print("ERROR: pending jobs remain but no feasible fixed-count launch was found.", file=sys.stderr)
                sys.exit(1)
            break

        time.sleep(1.0)
        for job in list(running):
            rc = job.process.poll()
            if rc is None:
                continue
            elapsed = time.time() - wall_start
            runtime = time.time() - job.start_time
            gpus_in_use -= set(job.gpu_ids)
            running.remove(job)
            job.stdout_sink.close()
            status = "OK" if rc == 0 else f"FAILED(rc={rc})"
            print(
                f"  t={elapsed:8.2f}s | END   {job.app:<15} | "
                f"freed {len(job.gpu_ids)} GPUs {job.gpu_ids} | "
                f"NUMA {job.numa_node} | runtime={runtime:.2f}s | {status}"
            )
            completed.append((job.app, len(job.gpu_ids), job.gpu_ids, job.numa_node, runtime, rc, job.proxy_score))

    total_time = time.time() - wall_start
    print("\n" + "=" * 80)
    print("Proxy co-schedule summary:")
    print(f"{'App':<15} {'#GPUs':>6} {'GPU IDs':>12} {'NUMA':>5} {'Runtime (s)':>12} {'Proxy':>10} {'Status':>10}")
    print("-" * 90)
    for app, gpu_count, gpu_ids, numa_node, runtime, rc, proxy_score in completed:
        status = "OK" if rc == 0 else "FAILED"
        print(
            f"{app:<15} {gpu_count:>6} {str(gpu_ids):>12} {numa_node:>5} "
            f"{runtime:>12.2f} {proxy_score:>10.2f} {status:>10}"
        )
    print("-" * 90)
    print(f"\nTotal makespan: {total_time:.2f}s")
    monitor.stop()
    monitor.print_summary()


def main():
    parser = argparse.ArgumentParser(
        description="Run an order-only proxy scheduler on 4 H100 GPUs."
    )
    parser.add_argument(
        "--jobs", nargs="+", default=DEFAULT_JOB_QUEUE,
        help=f"Job queue (app names in order). Default: {DEFAULT_JOB_QUEUE}",
    )
    parser.add_argument(
        "--policy", choices=["proxy", "sequential"], default="proxy",
        help="Execution mode (default: proxy)",
    )
    parser.add_argument(
        "--max-concurrent", type=int, default=2,
        help="Max concurrent apps (default: 2)",
    )
    parser.add_argument(
        "--metrics-file", type=Path, default=PROFILE_METRICS_PATH,
        help=f"Path to edp_metrics.txt (default: {PROFILE_METRICS_PATH})",
    )

    args = parser.parse_args()
    fixed_profiles = load_fixed_proxy_scores(args.metrics_file, FIXED_GPU_COUNTS)

    missing = [app for app in args.jobs if app not in fixed_profiles]
    if missing:
        print(f"ERROR: Missing fixed profile rows for: {missing}", file=sys.stderr)
        sys.exit(1)

    print("Job queue and fixed GPU counts:")
    for app in args.jobs:
        profile = fixed_profiles[app]
        print(f"  {app:<15} -> {profile.gpu_count} GPUs, proxy={profile.norm_edp:.2f}")
    print()

    if args.policy == "sequential":
        run_sequential(args.jobs, fixed_profiles)
    else:
        run_proxy(args.jobs, fixed_profiles, args.max_concurrent, args.metrics_file)


if __name__ == "__main__":
    main()
