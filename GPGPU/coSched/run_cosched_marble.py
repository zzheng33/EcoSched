#!/usr/bin/env python3
"""Single-node Marble baseline.

Marble uses brief profiling to select, for each application, the GPU count that
minimizes runtime (best performance). It then runs the existing single-node
NUMA-aware co-scheduler in FCFS mode with those fixed GPU counts.
"""

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, Sequence, Tuple

from run_cosched import (
    DEFAULT_JOB_QUEUE,
    NUMA0_GPUS,
    NUMA1_GPUS,
    RESULTS_DIR,
    TOTAL_GPUS,
    TeeStream,
    allocate_gpus_numa,
    run_cosched,
)

DEFAULT_METRICS_FILE = Path("/home/ac.zzheng/power/GPGPU/data/H100/perf_metrics.txt")
DEFAULT_PERF_TOL = 0.05
SECTION_RE = re.compile(r"^===== .*?/([^/ ]+) =====$")


def parse_runtime_rows(metrics_path: Path, selected_jobs: Sequence[str]) -> Dict[str, Dict[int, float]]:
    raw_rows: Dict[str, Dict[int, float]] = {}
    current_job = None

    for raw_line in metrics_path.read_text().splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = SECTION_RE.match(line)
        if match:
            current_job = match.group(1)
            raw_rows.setdefault(current_job, {})
            continue
        if current_job is None or line.startswith("cap=") or line.startswith("gpu_count"):
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        gpu_count = int(parts[0])
        runtime_s = float(parts[1])
        raw_rows[current_job][gpu_count] = runtime_s

    missing = [job for job in selected_jobs if job not in raw_rows]
    if missing:
        raise ValueError("Missing jobs in metrics file: {}".format(missing))
    return {job: raw_rows[job] for job in selected_jobs}


def select_marble_gpu_counts(
    runtime_rows: Dict[str, Dict[int, float]],
    perf_tol: float,
) -> Dict[str, Tuple[int, float]]:
    selected = {}
    for app, rows in runtime_rows.items():
        min_runtime = min(rows.values())
        runtime_limit = min_runtime * (1.0 + perf_tol)
        feasible = sorted((gpu_count, runtime_s) for gpu_count, runtime_s in rows.items() if runtime_s <= runtime_limit + 1e-9)
        gpu_count, runtime_s = min(feasible, key=lambda item: (item[0], item[1]))
        selected[app] = (gpu_count, runtime_s)
    return selected


def _results_log_path(results_dir: Path, dry_run: bool) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    name = "run_cosched_marble_dryrun.txt" if dry_run else "run_cosched_marble.txt"
    return results_dir / name


def run_marble_dry(job_queue: Sequence[str], marble_modes: Dict[str, Tuple[int, float]], max_concurrent: int) -> None:
    pending = list(job_queue)
    sim_running = []  # (app, gpu_count, gpu_ids, numa, end_time)
    sim_gpus_in_use = set()
    sim_numas_in_use = set()
    sim_time = 0.0

    print(
        "Marble dry-run: {} apps on {} GPUs (max {} concurrent, fcfs queue)".format(
            len(pending), TOTAL_GPUS, max_concurrent
        )
    )
    print("NUMA 0 GPUs: {}  |  NUMA 1 GPUs: {}".format(NUMA0_GPUS, NUMA1_GPUS))
    print("=" * 80)
    print("\n[DRY RUN] Simulating Marble schedule order:\n")

    while pending or sim_running:
        scheduled = True
        while scheduled and pending and len(sim_running) < max_concurrent:
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
            app = next((candidate for candidate in pending if marble_modes[candidate][0] <= free_gpu_count), None)
            if app is None:
                break

            needed, runtime_s = marble_modes[app]
            gpu_ids = allocate_gpus_numa(needed, numa, sim_gpus_in_use)
            if gpu_ids is None:
                break

            end_time = sim_time + runtime_s
            sim_running.append((app, needed, gpu_ids, numa, end_time))
            sim_gpus_in_use.update(gpu_ids)
            sim_numas_in_use.add(numa)
            pending.remove(app)
            scheduled = True
            print(
                "  t={:8.2f}s | START {:<15} | {} GPUs {} | NUMA {} | ends ~t={:.2f}s".format(
                    sim_time,
                    app,
                    needed,
                    gpu_ids,
                    numa,
                    end_time,
                )
            )

        if not sim_running:
            break

        sim_running.sort(key=lambda item: item[4])
        app, gpu_count, gpu_ids, numa, end_time = sim_running.pop(0)
        sim_time = end_time
        sim_gpus_in_use -= set(gpu_ids)
        sim_numas_in_use.discard(numa)
        print(
            "  t={:8.2f}s | END   {:<15} | freed {} GPUs {} | NUMA {}".format(
                sim_time,
                app,
                gpu_count,
                gpu_ids,
                numa,
            )
        )

    print("\nEstimated makespan: ~{:.2f}s".format(sim_time))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the single-node Marble baseline.")
    parser.add_argument(
        "--metrics-file",
        type=Path,
        default=DEFAULT_METRICS_FILE,
        help="Path to perf_metrics.txt. Default: {}".format(DEFAULT_METRICS_FILE),
    )
    parser.add_argument(
        "--jobs",
        nargs="+",
        default=DEFAULT_JOB_QUEUE,
        help="Job queue (app names in order). Default: {}".format(DEFAULT_JOB_QUEUE),
    )
    parser.add_argument(
        "--perf-tol",
        type=float,
        default=DEFAULT_PERF_TOL,
        help="Performance tolerance relative to the fastest mode. Marble picks the smallest GPU count within (1 + perf_tol) of the minimum runtime. Default: {}".format(DEFAULT_PERF_TOL),
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=2,
        help="Max number of concurrent apps on the node. Default: 2",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate Marble scheduling without launching jobs.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=RESULTS_DIR,
        help="Directory for fixed run logs. Default: {}".format(RESULTS_DIR),
    )
    args = parser.parse_args()

    runtime_rows = parse_runtime_rows(args.metrics_file, args.jobs)
    marble_modes = select_marble_gpu_counts(runtime_rows, args.perf_tol)
    gpu_counts = {app: gpu_count for app, (gpu_count, _) in marble_modes.items()}

    log_path = _results_log_path(args.results_dir, args.dry_run)
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    with log_path.open("w", encoding="utf-8") as log_file:
        tee = TeeStream(original_stdout, log_file)
        sys.stdout = tee
        sys.stderr = tee
        try:
            print("Results log: {}".format(log_path))
            print("Marble policy: pick the smallest GPU count within {:.2%} of the minimum profiled runtime, then FCFS co-schedule.".format(args.perf_tol))
            print("Metrics source: {}".format(args.metrics_file))
            print("Selected Marble GPU counts:")
            for app in args.jobs:
                gpu_count, runtime_s = marble_modes[app]
                print("  {:<15} -> {} GPUs (runtime={:.2f}s)".format(app, gpu_count, runtime_s))
            print()
            if args.dry_run:
                run_marble_dry(args.jobs, marble_modes, args.max_concurrent)
            else:
                run_cosched(
                    job_queue=list(args.jobs),
                    gpu_counts=gpu_counts,
                    max_concurrent=args.max_concurrent,
                    dry_run=False,
                    policy="fcfs",
                )
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr


if __name__ == "__main__":
    main()
