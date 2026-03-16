#!/usr/bin/env python3
"""
Co-schedule benchmarks on 4 H100 GPUs with NUMA-aware GPU mapping.

Policies:
  - fcfs:       First-Come-First-Served (queue order)
  - best-fit:   Pick the job whose GPU count best fills remaining GPUs
  - sequential: Run apps one by one (baseline, no co-scheduling)

NUMA topology (2-socket):
  NUMA 0: GPUs 0, 1   (tenant 1)
  NUMA 1: GPUs 2, 3   (tenant 2)

Scheduling rules:
  - At most 2 apps run concurrently (tenant 1 on NUMA 0, tenant 2 on NUMA 1)
  - Tenant 1 binds CPU/memory to NUMA 0, prefers GPUs 0,1
  - Tenant 2 binds CPU/memory to NUMA 1, prefers GPUs 2,3
  - If an app needs 3 GPUs, it takes both from its own NUMA side + 1 from
    the other side (only allowed when running alone)
  - If an app needs 4 GPUs, it runs alone using all GPUs

Usage:
    python3 run_cosched.py --policy best-fit           # best-fit (default)
    python3 run_cosched.py --policy fcfs               # FCFS
    python3 run_cosched.py --policy sequential         # sequential baseline
    python3 run_cosched.py --policy best-fit --dry-run # dry-run simulation
    python3 run_cosched.py --both                      # sequential then best-fit
"""

import argparse
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

HOME = Path.home()
RESULTS_DIR = HOME / "power/GPGPU/coSched/results"
SCRIPT_DIR = HOME / "power/GPGPU/script"
SPEC_SCRIPT_DIR = SCRIPT_DIR / "run_benchmark/spec_script"
ECP_SCRIPT_DIR = SCRIPT_DIR / "run_benchmark/ecp_script"

TOTAL_GPUS = 4

# NUMA-to-GPU mapping
NUMA0_GPUS = [0, 1]
NUMA1_GPUS = [2, 3]

# Environment setup for SPEC benchmarks (MPI + CUDA modules)
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

DEFAULT_JOB_QUEUE = [
    'pot3d', 'minisweep', 'lbm', 'cloverleaf', 'tealeaf',
    'miniweather', 'bert', 'gpt2', 'resnet50', 'hpgmg',
]

# Benchmarks that use torchrun (bert, gpt2)
TORCHRUN_APPS = {'bert', 'gpt2'}

# Benchmarks that use dl.py via the ML python env (resnet50, etc.)
ML_DL_APPS = {'resnet50'}

ML_PYTHON = HOME / "env/ml/bin/python3"
ML_SCRIPT = HOME / "power/ML/dl.py"
ML_WORKDIR = HOME / "power/ML"
ML_BATCH_SIZE = 2048
ML_EPOCHS = 3
ML_LR = 0.001

# All others are SPEC/MPI-based


class TeeStream(object):
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()


class PowerMonitor:
    """
    Background thread that samples all GPU power every `interval` seconds
    via nvidia-smi. Computes total energy (J) across all GPUs when stopped.
    """

    def __init__(self, num_gpus=TOTAL_GPUS, interval=0.3):
        self.num_gpus = num_gpus
        self.interval = interval
        self._samples = []   # list of (timestamp, [gpu0_W, gpu1_W, ...])
        self._stop = threading.Event()
        self._thread = None

    def _query_power(self):
        try:
            out = subprocess.check_output(
                ['nvidia-smi', '--query-gpu=power.draw',
                 '--format=csv,noheader,nounits'],
                encoding='utf-8', stderr=subprocess.DEVNULL,
            )
            return [float(x.strip()) for x in out.strip().split('\n')]
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
        """Compute total energy in Joules using trapezoidal integration."""
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
        energy_kj = energy_j / 1000.0
        n = len(self._samples)
        if n >= 2:
            duration = self._samples[-1][0] - self._samples[0][0]
        else:
            duration = 0.0
        print(f"\nPower monitoring: {n} samples over {duration:.1f}s")
        print(f"Total GPU energy: {energy_j:.2f} J ({energy_kj:.2f} kJ)")


@dataclass
class RunningJob:
    app: str
    gpu_ids: List[int]
    numa_node: int
    process: subprocess.Popen
    start_time: float


def allocate_gpus_numa(needed: int, tenant_numa: int, gpus_in_use: set) -> Optional[List[int]]:
    """
    Allocate GPUs with NUMA affinity.

    - Prefer GPUs from the tenant's own NUMA node first.
    - If more GPUs are needed (e.g., 3 GPUs), spill to the other NUMA side.
    - Never allocate GPUs already in use.
    """
    if tenant_numa == 0:
        local_gpus = [g for g in NUMA0_GPUS if g not in gpus_in_use]
        remote_gpus = [g for g in NUMA1_GPUS if g not in gpus_in_use]
    else:
        local_gpus = [g for g in NUMA1_GPUS if g not in gpus_in_use]
        remote_gpus = [g for g in NUMA0_GPUS if g not in gpus_in_use]

    available = local_gpus + remote_gpus
    if len(available) >= needed:
        return available[:needed]
    return None


def pick_numa_for_tenant(running_jobs: List[RunningJob]) -> Optional[int]:
    """
    Pick the NUMA node for the next tenant.

    - If no jobs running, assign NUMA 0.
    - If one job running on NUMA 0, assign NUMA 1 (and vice versa).
    - If both NUMA nodes occupied, return None (cannot schedule).
    """
    if not running_jobs:
        return 0
    used_numas = {j.numa_node for j in running_jobs}
    if 0 not in used_numas:
        return 0
    if 1 not in used_numas:
        return 1
    return None  # both occupied


# ---------------------------------------------------------------------------
# Command builders — each returns (cmd, env, cwd)
# All commands are wrapped with numactl for CPU/memory binding.
# ---------------------------------------------------------------------------

def build_spec_command(app: str, gpu_ids: List[int], numa_node: int):
    """Build command for SPEC/MPI benchmarks with NUMA binding."""
    gpu_count = len(gpu_ids)
    gpu_csv = ",".join(str(g) for g in gpu_ids)
    script_path = SPEC_SCRIPT_DIR / f"{app}.sh"

    shell_cmd = (
        SPEC_ENV_SETUP
        + f"export CUDA_VISIBLE_DEVICES={gpu_csv}; "
        + f"numactl --cpunodebind={numa_node} --membind={numa_node} "
        + f"bash {script_path} {gpu_count}"
    )
    cmd = ["bash", "-lc", shell_cmd]
    env = os.environ.copy()
    return cmd, env, None


def build_torchrun_command(app: str, gpu_ids: List[int], numa_node: int):
    """Build command for ML benchmarks using torchrun (bert, gpt2) with NUMA binding."""
    gpu_count = len(gpu_ids)
    gpu_csv = ",".join(str(g) for g in gpu_ids)

    # Wrap the torchrun call with numactl via a shell command
    shell_cmd = (
        f"export CUDA_VISIBLE_DEVICES={gpu_csv}; "
        f"export TOKENIZERS_PARALLELISM=false; "
        f"source {HOME}/env/ml/bin/activate; "
        f"cd {HOME}/benchmark/ECP/{app if app != 'bert' else 'bert-large'}; "
        f"numactl --cpunodebind={numa_node} --membind={numa_node} "
        f"torchrun --nproc_per_node={gpu_count} training.py; "
        f"deactivate"
    )
    cmd = ["bash", "-lc", shell_cmd]
    env = os.environ.copy()
    return cmd, env, None


def build_ml_dl_command(app: str, gpu_ids: List[int], numa_node: int):
    """Build command for ML models using dl.py (resnet50, etc.) with NUMA binding."""
    gpu_count = len(gpu_ids)
    gpu_csv = ",".join(str(g) for g in gpu_ids)

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
    """Dispatch to the right command builder based on app type."""
    if app in TORCHRUN_APPS:
        return build_torchrun_command(app, gpu_ids, numa_node)
    elif app in ML_DL_APPS:
        return build_ml_dl_command(app, gpu_ids, numa_node)
    else:
        return build_spec_command(app, gpu_ids, numa_node)


def _devnull():
    return open(os.devnull, "w")


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

def run_sequential(
    job_queue: List[str],
    gpu_counts: Dict[str, int],
):
    """Run each app one by one (baseline, no co-scheduling)."""
    monitor = PowerMonitor()
    monitor.start()

    completed = []
    wall_start = time.time()

    print(f"Sequential execution: {len(job_queue)} apps on {TOTAL_GPUS} GPUs")
    print(f"NUMA 0 GPUs: {NUMA0_GPUS}  |  NUMA 1 GPUs: {NUMA1_GPUS}")
    print("=" * 80)

    for app in job_queue:
        needed = gpu_counts[app]
        numa = 0  # always use NUMA 0 when running alone
        gpu_ids = allocate_gpus_numa(needed, numa, set())

        cmd, env, cwd = build_command(app, gpu_ids, numa)

        elapsed = time.time() - wall_start
        print(f"  t={elapsed:8.2f}s | START {app:<15} | "
              f"{needed} GPUs {gpu_ids} | NUMA {numa}")

        app_start = time.time()
        devnull = _devnull()
        proc = subprocess.Popen(
            cmd, env=env,
            cwd=str(cwd) if cwd else None,
            stdout=devnull, stderr=subprocess.STDOUT,
        )
        rc = proc.wait()
        runtime = time.time() - app_start
        devnull.close()

        elapsed = time.time() - wall_start
        status = "OK" if rc == 0 else f"FAILED(rc={rc})"
        print(f"  t={elapsed:8.2f}s | END   {app:<15} | "
              f"runtime={runtime:.2f}s | {status}")

        completed.append({
            'app': app,
            'gpu_count': needed,
            'gpu_ids': gpu_ids,
            'numa_node': numa,
            'runtime': runtime,
            'return_code': rc,
        })

    total_time = time.time() - wall_start

    print("\n" + "=" * 80)
    print("Sequential summary:")
    print(f"{'App':<15} {'#GPUs':>6} {'GPU IDs':>12} {'Runtime (s)':>12}")
    print("-" * 52)
    for r in completed:
        print(f"{r['app']:<15} {r['gpu_count']:>6} {str(r['gpu_ids']):>12} "
              f"{r['runtime']:>12.2f}")
    print("-" * 52)
    print(f"\nTotal makespan: {total_time:.2f}s")
    monitor.stop()
    monitor.print_summary()


def _pick_next_app(candidates, gpu_counts, free_gpu_count, policy):
    """Pick the next app to schedule from candidates.

    - 'fcfs':     first app in queue order that fits.
    - 'best-fit': app whose GPU count is closest to free_gpu_count
                  (fills all remaining GPUs, minimizes waste).
    """
    if policy == "fcfs":
        for app in candidates:
            if gpu_counts[app] <= free_gpu_count:
                return app
        return None

    # best-fit: prefer largest GPU count that fits (smallest waste)
    best_app = None
    best_diff = free_gpu_count + 1
    for app in candidates:
        needed = gpu_counts[app]
        if needed <= free_gpu_count:
            diff = free_gpu_count - needed  # 0 = perfect fit
            if diff < best_diff:
                best_diff = diff
                best_app = app
    return best_app


def run_cosched(
    job_queue: List[str],
    gpu_counts: Dict[str, int],
    max_concurrent: int,
    dry_run: bool,
    policy: str = "best-fit",
):
    """Run co-scheduling on 4 GPUs with NUMA-aware placement.

    policy: 'fcfs' = first-come-first-served, 'best-fit' = fill remaining GPUs.
    """
    monitor = PowerMonitor()

    pending = list(job_queue)
    running: List[RunningJob] = []
    completed = []
    gpus_in_use: set = set()
    wall_start = time.time()

    print(f"Co-scheduling {len(pending)} apps on {TOTAL_GPUS} GPUs "
          f"(max {max_concurrent} concurrent, policy={policy})")
    print(f"NUMA 0 GPUs: {NUMA0_GPUS}  |  NUMA 1 GPUs: {NUMA1_GPUS}")
    print("=" * 80)

    if dry_run:
        print("\n[DRY RUN] Simulating schedule order:\n")
        sim_pending = list(pending)
        sim_running = []  # (app, gpu_count, gpu_ids, numa, end_time)
        sim_gpus_in_use = set()
        sim_numas_in_use = set()
        sim_time = 0.0

        est_runtime = {
            'pot3d': 133.09, 'minisweep': 43.61, 'lbm': 18.82,
            'cloverleaf': 16.83, 'tealeaf': 16.21, 'miniweather': 57.01,
            'hpgmg': 10.78, 'bert': 21.03, 'gpt2': 14.80, 'resnet50': 20.81,
        }

        while sim_pending or sim_running:
            scheduled = True
            while scheduled and sim_pending and len(sim_running) < max_concurrent:
                scheduled = False

                if not sim_numas_in_use:
                    numa = 0
                elif 0 not in sim_numas_in_use:
                    numa = 0
                elif 1 not in sim_numas_in_use:
                    numa = 1
                else:
                    break

                free_gpu_count = TOTAL_GPUS - len(sim_gpus_in_use)
                app = _pick_next_app(sim_pending, gpu_counts, free_gpu_count, policy)
                if app is None:
                    break

                needed = gpu_counts[app]
                gpu_ids = allocate_gpus_numa(needed, numa, sim_gpus_in_use)
                if gpu_ids is not None:
                    rt = est_runtime.get(app, 30.0)
                    end = sim_time + rt
                    sim_running.append((app, needed, gpu_ids, numa, end))
                    sim_gpus_in_use.update(gpu_ids)
                    sim_numas_in_use.add(numa)
                    sim_pending.remove(app)
                    scheduled = True
                    print(f"  t={sim_time:8.2f}s | START {app:<15} | "
                          f"{needed} GPUs {gpu_ids} | NUMA {numa} | "
                          f"ends ~t={end:.2f}s")
                else:
                    break

            if not sim_running:
                break

            sim_running.sort(key=lambda x: x[4])
            app, ngpu, gids, numa, end = sim_running.pop(0)
            sim_time = end
            sim_gpus_in_use -= set(gids)
            sim_numas_in_use.discard(numa)
            print(f"  t={sim_time:8.2f}s | END   {app:<15} | "
                  f"freed {ngpu} GPUs {gids} | NUMA {numa}")

        print(f"\nEstimated makespan: ~{sim_time:.2f}s")
        return

    # --- Actual execution ---
    monitor.start()

    while pending or running:
        # Try to schedule pending apps (Best-Fit)
        scheduled_this_round = True
        while scheduled_this_round and pending and len(running) < max_concurrent:
            scheduled_this_round = False

            # Pick NUMA node for this tenant
            numa = pick_numa_for_tenant(running)
            if numa is None:
                break  # both NUMA nodes occupied

            # Pick next app based on policy
            free_gpu_count = TOTAL_GPUS - len(gpus_in_use)
            app = _pick_next_app(pending, gpu_counts, free_gpu_count, policy)
            if app is None:
                break  # no app fits

            needed = gpu_counts[app]

            # Allocate GPUs with NUMA affinity
            gpu_ids = allocate_gpus_numa(needed, numa, gpus_in_use)
            if gpu_ids is None:
                break  # shouldn't happen since _best_fit_pick checked

            # Launch the app
            cmd, env, cwd = build_command(app, gpu_ids, numa)
            devnull = _devnull()

            elapsed = time.time() - wall_start
            print(f"  t={elapsed:8.2f}s | START {app:<15} | "
                  f"{needed} GPUs {gpu_ids} | NUMA {numa}")

            proc = subprocess.Popen(
                cmd,
                env=env,
                cwd=str(cwd) if cwd else None,
                stdout=devnull,
                stderr=subprocess.STDOUT,
            )

            job = RunningJob(
                app=app,
                gpu_ids=gpu_ids,
                numa_node=numa,
                process=proc,
                start_time=time.time(),
            )
            running.append(job)
            gpus_in_use.update(gpu_ids)
            pending.remove(app)
            scheduled_this_round = True

        if not running:
            break

        # Poll running processes
        time.sleep(1.0)
        for job in list(running):
            rc = job.process.poll()
            if rc is not None:
                elapsed = time.time() - wall_start
                runtime = time.time() - job.start_time
                gpus_in_use -= set(job.gpu_ids)
                running.remove(job)

                status = "OK" if rc == 0 else f"FAILED(rc={rc})"
                print(f"  t={elapsed:8.2f}s | END   {job.app:<15} | "
                      f"freed {len(job.gpu_ids)} GPUs {job.gpu_ids} | "
                      f"NUMA {job.numa_node} | runtime={runtime:.2f}s | {status}")

                completed.append({
                    'app': job.app,
                    'gpu_count': len(job.gpu_ids),
                    'gpu_ids': job.gpu_ids,
                    'numa_node': job.numa_node,
                    'runtime': runtime,
                    'return_code': rc,
                })

    total_time = time.time() - wall_start

    # Print summary
    print("\n" + "=" * 80)
    print("Co-schedule summary:")
    print(f"{'App':<15} {'#GPUs':>6} {'GPU IDs':>12} {'NUMA':>5} "
          f"{'Runtime (s)':>12}")
    print("-" * 54)
    for r in completed:
        print(f"{r['app']:<15} {r['gpu_count']:>6} {str(r['gpu_ids']):>12} "
              f"{r['numa_node']:>5} {r['runtime']:>12.2f}")
    print("-" * 54)
    print(f"\nTotal makespan: {total_time:.2f}s")
    monitor.stop()
    monitor.print_summary()


def _results_log_path(args: argparse.Namespace) -> Path:
    args.results_dir.mkdir(parents=True, exist_ok=True)
    if args.both:
        policy = args.policy if args.policy != "sequential" else "best-fit"
        safe_policy = policy.replace("-", "_")
        name = f"run_cosched_both_{safe_policy}.txt"
    elif args.policy == "sequential":
        name = "run_cosched_sequential.txt"
    else:
        safe_policy = args.policy.replace("-", "_")
        prefix = "run_cosched_dryrun" if args.dry_run else "run_cosched"
        name = f"{prefix}_{safe_policy}.txt"
    return args.results_dir / name


def main():
    parser = argparse.ArgumentParser(
        description="Run co-scheduled benchmarks on 4 H100 GPUs (NUMA-aware)")

    parser.add_argument(
        "--jobs", nargs="+", default=DEFAULT_JOB_QUEUE,
        help=f"Job queue (app names in order). Default: {DEFAULT_JOB_QUEUE}")
    parser.add_argument(
        "--policy", type=str, default="best-fit",
        choices=["fcfs", "best-fit", "sequential"],
        help="Scheduling policy (default: best-fit)")
    parser.add_argument(
        "--max-concurrent", type=int, default=2,
        help="Max number of concurrent apps (default: 2)")
    parser.add_argument(
        "--both", action="store_true",
        help="Run sequential first, then co-scheduled (uses --policy for phase 2)")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print simulated schedule without launching anything")
    parser.add_argument(
        "--gpu-override", nargs="+", default=None,
        help="Override GPU counts as app:count pairs, e.g. bert:2 gpt2:2")
    parser.add_argument(
        "--results-dir", type=Path, default=RESULTS_DIR,
        help="Directory for fixed run logs. Default: {}".format(RESULTS_DIR))

    args = parser.parse_args()
    log_path = _results_log_path(args)
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    with log_path.open("w", encoding="utf-8") as log_file:
        tee = TeeStream(original_stdout, log_file)
        sys.stdout = tee
        sys.stderr = tee
        try:
            print(f"Results log: {log_path}")

            gpu_counts = dict(PREDICTED_GPU_COUNTS)

            # Apply overrides
            if args.gpu_override:
                for item in args.gpu_override:
                    app, count = item.split(":")
                    gpu_counts[app] = int(count)

            # Validate
            for app in args.jobs:
                if app not in gpu_counts:
                    print(f"ERROR: No GPU count defined for '{app}'", file=sys.stderr)
                    sys.exit(1)
                if gpu_counts[app] > TOTAL_GPUS:
                    print(f"ERROR: {app} needs {gpu_counts[app]} GPUs but only "
                          f"{TOTAL_GPUS} available", file=sys.stderr)
                    sys.exit(1)

            print("Job queue and GPU assignments:")
            for app in args.jobs:
                print(f"  {app:<15} -> {gpu_counts[app]} GPUs")
            print()

            if args.both:
                print("=" * 80)
                print("  PHASE 1: SEQUENTIAL")
                print("=" * 80)
                run_sequential(
                    job_queue=args.jobs,
                    gpu_counts=gpu_counts,
                )
                print("\n\n")
                policy = args.policy if args.policy != "sequential" else "best-fit"
                print("=" * 80)
                print(f"  PHASE 2: CO-SCHEDULED ({policy.upper()})")
                print("=" * 80)
                run_cosched(
                    job_queue=args.jobs,
                    gpu_counts=gpu_counts,
                    max_concurrent=args.max_concurrent,
                    dry_run=False,
                    policy=policy,
                )
            elif args.policy == "sequential":
                run_sequential(
                    job_queue=args.jobs,
                    gpu_counts=gpu_counts,
                )
            else:
                run_cosched(
                    job_queue=args.jobs,
                    gpu_counts=gpu_counts,
                    max_concurrent=args.max_concurrent,
                    dry_run=args.dry_run,
                    policy=args.policy,
                )
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr


if __name__ == "__main__":
    main()
