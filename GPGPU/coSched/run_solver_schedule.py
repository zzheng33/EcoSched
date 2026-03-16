#!/usr/bin/env python3
"""Run applications according to a saved solver schedule.

Expected input:
- The text output produced by solve_energy_optimal_cpsat.py redirected to a file.

Execution policy:
- Jobs are parsed from the solver's schedule table.
- Each job keeps the GPU count and placement chosen by the solver.
- Jobs are launched when their planned start time is reached and the required
  GPUs are free.
- If actual runtimes differ from the solver plan, later launches slip until the
  required GPUs become available.
- All console output is also recorded in a timestamped text file under results/.
"""

import argparse
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, NamedTuple, Optional, Sequence, Tuple

from run_cosched import (
    NUMA0_GPUS,
    NUMA1_GPUS,
    PowerMonitor,
    TOTAL_GPUS,
    build_command,
)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = SCRIPT_DIR / "results"
DEFAULT_SCHEDULE_FILE = DEFAULT_RESULTS_DIR / "solver_schedule.txt"


class PlannedJob(NamedTuple):
    order_idx: int
    start_s: float
    end_s: float
    app: str
    gpu_count: int
    placement: Tuple[int, int]
    runtime_s: float


class RunningJob(object):
    def __init__(self, plan: PlannedJob, gpu_ids: List[int], numa_node: int,
                 process: subprocess.Popen, devnull, actual_start_time: float):
        self.plan = plan
        self.gpu_ids = gpu_ids
        self.numa_node = numa_node
        self.process = process
        self.devnull = devnull
        self.actual_start_time = actual_start_time


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


TABLE_ROW_RE = re.compile(
    r"^\s*(\d+)\s+"
    r"([0-9]+(?:\.[0-9]+)?)\s+"
    r"([0-9]+(?:\.[0-9]+)?)\s+"
    r"([A-Za-z0-9_.-]+)\s+"
    r"(\d+)\s+"
    r"\((\d),(\d)\)\s+"
    r"([0-9]+(?:\.[0-9]+)?)"
)


def parse_solver_schedule(schedule_file: Path) -> List[PlannedJob]:
    plans = []
    for line in schedule_file.read_text().splitlines():
        match = TABLE_ROW_RE.match(line)
        if not match:
            continue
        plans.append(
            PlannedJob(
                order_idx=int(match.group(1)),
                start_s=float(match.group(2)),
                end_s=float(match.group(3)),
                app=match.group(4),
                gpu_count=int(match.group(5)),
                placement=(int(match.group(6)), int(match.group(7))),
                runtime_s=float(match.group(8)),
            )
        )

    if not plans:
        raise ValueError(
            "No schedule rows found in {}. Pass the text output from "
            "solve_energy_optimal_cpsat.py.".format(schedule_file)
        )

    plans.sort(key=lambda item: (item.start_s, item.order_idx, item.app))
    return plans


def print_planned_decisions(plans: Sequence[PlannedJob]):
    print("Planned solver decisions:")
    print("=" * 80)
    for plan in plans:
        print(
            "  {}. t={:.2f}s | {:<15} | {} GPUs | placement {} | planned_end={:.2f}s".format(
                plan.order_idx,
                plan.start_s,
                plan.app,
                plan.gpu_count,
                plan.placement,
                plan.end_s,
            )
        )
    print()


def home_numa_for_placement(placement: Tuple[int, int]) -> int:
    use0, use1 = placement
    if use0 == 0:
        return 1
    if use1 == 0:
        return 0
    return 0 if use0 >= use1 else 1


def allocate_gpu_ids_for_placement(
    placement: Tuple[int, int],
    gpus_in_use: set,
) -> Optional[Tuple[List[int], int]]:
    need0, need1 = placement
    free0 = [gpu for gpu in NUMA0_GPUS if gpu not in gpus_in_use]
    free1 = [gpu for gpu in NUMA1_GPUS if gpu not in gpus_in_use]

    if len(free0) < need0 or len(free1) < need1:
        return None

    numa_node = home_numa_for_placement(placement)
    if numa_node == 0:
        gpu_ids = free0[:need0] + free1[:need1]
    else:
        gpu_ids = free1[:need1] + free0[:need0]
    return gpu_ids, numa_node


def _devnull():
    return open(os.devnull, "w")


def dry_run(plans: Sequence[PlannedJob]):
    print("Dry-run solver schedule")
    print("=" * 80)
    for plan in plans:
        print(
            "  t={:8.2f}s | PLAN  {:<15} | {} GPUs | placement {}".format(
                plan.start_s, plan.app, plan.gpu_count, plan.placement
            )
        )


def run_planned_schedule(plans: Sequence[PlannedJob], poll_interval: float):
    pending = list(plans)
    running = []
    completed = []
    gpus_in_use = set()
    monitor = PowerMonitor()
    wall_start = time.time()

    print("Solver-driven execution: {} apps on {} GPUs".format(len(plans), TOTAL_GPUS))
    print("NUMA 0 GPUs: {}  |  NUMA 1 GPUs: {}".format(NUMA0_GPUS, NUMA1_GPUS))
    print("=" * 80)

    monitor.start()

    try:
        while pending or running:
            launched = True
            while launched and pending and len(running) < 2:
                launched = False
                elapsed = time.time() - wall_start

                due_jobs = [plan for plan in pending if plan.start_s <= elapsed + 1e-6]
                if not due_jobs:
                    break

                selected = None
                for plan in due_jobs:
                    allocation = allocate_gpu_ids_for_placement(plan.placement, gpus_in_use)
                    if allocation is not None:
                        selected = (plan, allocation[0], allocation[1])
                        break

                if selected is None:
                    break

                plan, gpu_ids, numa_node = selected
                cmd, env, cwd = build_command(plan.app, gpu_ids, numa_node)
                devnull = _devnull()

                elapsed = time.time() - wall_start
                delay = elapsed - plan.start_s
                print(
                    "  t={:8.2f}s | START {:<15} | {} GPUs {} | NUMA {} | "
                    "planned={:.2f}s | delay={:+.2f}s".format(
                        elapsed,
                        plan.app,
                        plan.gpu_count,
                        gpu_ids,
                        numa_node,
                        plan.start_s,
                        delay,
                    )
                )

                proc = subprocess.Popen(
                    cmd,
                    env=env,
                    cwd=str(cwd) if cwd else None,
                    stdout=devnull,
                    stderr=subprocess.STDOUT,
                )
                running.append(RunningJob(plan, gpu_ids, numa_node, proc, devnull, time.time()))
                gpus_in_use.update(gpu_ids)
                pending.remove(plan)
                launched = True

            if running:
                time.sleep(poll_interval)
                for job in list(running):
                    rc = job.process.poll()
                    if rc is None:
                        continue

                    elapsed = time.time() - wall_start
                    runtime = time.time() - job.actual_start_time
                    gpus_in_use -= set(job.gpu_ids)
                    running.remove(job)
                    job.devnull.close()

                    status = "OK" if rc == 0 else "FAILED(rc={})".format(rc)
                    print(
                        "  t={:8.2f}s | END   {:<15} | freed {} GPUs {} | NUMA {} | "
                        "runtime={:.2f}s | {}".format(
                            elapsed,
                            job.plan.app,
                            len(job.gpu_ids),
                            job.gpu_ids,
                            job.numa_node,
                            runtime,
                            status,
                        )
                    )
                    completed.append({
                        "app": job.plan.app,
                        "gpu_count": job.plan.gpu_count,
                        "gpu_ids": job.gpu_ids,
                        "numa_node": job.numa_node,
                        "planned_start": job.plan.start_s,
                        "runtime": runtime,
                        "return_code": rc,
                    })
                continue

            if pending:
                elapsed = time.time() - wall_start
                wait_s = max(0.05, min(poll_interval, pending[0].start_s - elapsed))
                time.sleep(wait_s)

    finally:
        for job in running:
            try:
                job.process.terminate()
            except Exception:
                pass
            try:
                job.devnull.close()
            except Exception:
                pass
        monitor.stop()

    total_time = time.time() - wall_start

    print("\n" + "=" * 80)
    print("Solver-driven summary:")
    print(
        "{:<15} {:>6} {:>12} {:>5} {:>12} {:>12}".format(
            "App", "#GPUs", "GPU IDs", "NUMA", "Planned (s)", "Runtime (s)"
        )
    )
    print("-" * 74)
    for item in completed:
        print(
            "{:<15} {:>6} {:>12} {:>5} {:>12.2f} {:>12.2f}".format(
                item["app"],
                item["gpu_count"],
                str(item["gpu_ids"]),
                item["numa_node"],
                item["planned_start"],
                item["runtime"],
            )
        )
    print("-" * 74)
    print("\nTotal makespan: {:.2f}s".format(total_time))
    monitor.print_summary()


def main():
    parser = argparse.ArgumentParser(
        description="Run applications using a saved solver schedule."
    )
    parser.add_argument(
        "--schedule-file",
        type=Path,
        default=DEFAULT_SCHEDULE_FILE,
        help="Text file containing the stdout from solve_energy_optimal_cpsat.py",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and print the planned solver schedule without launching apps",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=1.0,
        help="Process polling interval in seconds. Default: 1.0",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Directory for timestamped run logs. Default: {}".format(DEFAULT_RESULTS_DIR),
    )
    args = parser.parse_args()

    plans = parse_solver_schedule(args.schedule_file)

    results_dir = args.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode = "dryrun" if args.dry_run else "run"
    log_path = results_dir / "solver_execution_{}.txt".format(mode)

    original_stdout = sys.stdout
    original_stderr = sys.stderr
    with log_path.open("w") as log_file:
        sys.stdout = TeeStream(original_stdout, log_file)
        sys.stderr = TeeStream(original_stderr, log_file)
        try:
            print("Results log: {}".format(log_path))
            print_planned_decisions(plans)
            if args.dry_run:
                dry_run(plans)
                return

            run_planned_schedule(plans, args.poll_interval)
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr


if __name__ == "__main__":
    main()
