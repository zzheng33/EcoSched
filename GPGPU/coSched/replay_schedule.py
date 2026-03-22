#!/usr/bin/env python3
"""Replay a schedule extracted from an EcoPack (or any co-scheduler) log.

Usage:
    # Parse and replay an EcoPack log:
    python3 replay_schedule.py --log results/H100/mix/EcoPack_cmab_run.txt

    # Parse, save the schedule JSON, and replay:
    python3 replay_schedule.py --log results/H100/mix/EcoPack_cmab_run.txt \
        --save-schedule schedule.json

    # Replay from a previously saved schedule JSON:
    python3 replay_schedule.py --schedule schedule.json

    # Dry-run (parse & print, no execution):
    python3 replay_schedule.py --log results/H100/mix/EcoPack_cmab_run.txt --dry-run

    # Multiple repeat runs:
    python3 replay_schedule.py --log results/H100/mix/EcoPack_cmab_run.txt --repeats 3

Execution model:
    - Jobs are launched in the same order as the original run.
    - Start times are NOT replayed as absolute times; instead, each job is
      launched as soon as the required GPUs are free (event-driven).
    - At most 2 jobs run concurrently (same constraint as the original schedulers).
    - GPU placement (NUMA node, GPU IDs) matches the original schedule.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

from run_cosched_sequential import (
    NUMA0_GPUS,
    NUMA1_GPUS,
    PowerMonitor,
    TOTAL_GPUS,
    base_app_name,
    build_command,
)
from config import APP_LOG_ENABLED, SYSTEM

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = SCRIPT_DIR / "results" / SYSTEM


# ── Data structures ──────────────────────────────────────────────────────────

class ScheduleEntry(NamedTuple):
    order: int
    app: str
    gpu_count: int
    gpu_ids: List[int]
    numa_node: int


class RunningJob:
    def __init__(self, entry: ScheduleEntry, process: subprocess.Popen,
                 log_file, start_time: float):
        self.entry = entry
        self.process = process
        self.log_file = log_file
        self.start_time = start_time


class TeeStream:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
        return len(data)

    def flush(self):
        for s in self.streams:
            s.flush()


# ── Parsers ──────────────────────────────────────────────────────────────────

# Matches EcoPack / co-scheduler START lines:
#   t=    0.00s | START pot3d           | 2 GPUs [0, 1] | NUMA 0 | ...
START_RE = re.compile(
    r"t=\s*([0-9.]+)s\s*\|\s*START\s+(\S+)"
    r"\s*\|\s*(\d+)\s+GPUs?\s+\[([0-9,\s]+)\]"
    r"\s*\|\s*NUMA\s+(\d+)"
)


def parse_ecopack_log(log_path: Path) -> List[ScheduleEntry]:
    """Extract the launch schedule from an EcoPack / co-scheduler log file."""
    entries = []
    for line in log_path.read_text().splitlines():
        m = START_RE.search(line)
        if not m:
            continue
        start_t = float(m.group(1))
        app = m.group(2)
        gpu_count = int(m.group(3))
        gpu_ids = [int(g.strip()) for g in m.group(4).split(",")]
        numa_node = int(m.group(5))
        entries.append((start_t, app, gpu_count, gpu_ids, numa_node))

    if not entries:
        raise ValueError(f"No START lines found in {log_path}")

    # Sort by original start time to preserve launch order
    entries.sort(key=lambda e: e[0])
    return [
        ScheduleEntry(order=i, app=app, gpu_count=gc, gpu_ids=gids, numa_node=numa)
        for i, (_, app, gc, gids, numa) in enumerate(entries)
    ]


def load_schedule_json(json_path: Path) -> List[ScheduleEntry]:
    """Load a schedule from a JSON file."""
    data = json.loads(json_path.read_text())
    return [
        ScheduleEntry(
            order=i,
            app=e["app"],
            gpu_count=e["gpu_count"],
            gpu_ids=e["gpu_ids"],
            numa_node=e["numa_node"],
        )
        for i, e in enumerate(data)
    ]


def save_schedule_json(entries: Sequence[ScheduleEntry], json_path: Path):
    """Save a schedule to a JSON file."""
    data = [
        {
            "app": e.app,
            "gpu_count": e.gpu_count,
            "gpu_ids": e.gpu_ids,
            "numa_node": e.numa_node,
        }
        for e in entries
    ]
    json_path.write_text(json.dumps(data, indent=2) + "\n")
    print(f"Schedule saved to {json_path}")


# ── Execution ────────────────────────────────────────────────────────────────

def allocate_if_free(entry: ScheduleEntry, gpus_in_use: set) -> bool:
    """Check if the required GPUs are all free."""
    return all(g not in gpus_in_use for g in entry.gpu_ids)


def print_schedule(entries: Sequence[ScheduleEntry]):
    print("Replay schedule ({} apps):".format(len(entries)))
    print("=" * 72)
    for e in entries:
        print("  {:2d}. {:<30s} | {} GPUs {} | NUMA {}".format(
            e.order, e.app, e.gpu_count, e.gpu_ids, e.numa_node))
    print()


# ── Co-run pair analysis ─────────────────────────────────────────────────────

# Matches START and END lines with timestamps
EVENT_RE = re.compile(
    r"t=\s*([0-9.]+)s\s*\|\s*(START|END)\s+(\S+)"
    r"\s*\|\s*(?:freed\s+)?(\d+)\s+GPUs?\s+\[([0-9,\s]+)\]"
    r"\s*\|\s*NUMA\s+(\d+)"
)
RUNTIME_RE = re.compile(r"runtime=([0-9.]+)s")


def analyze_corun_pairs(log_path: Path):
    """Extract co-running pairs and their runtimes from an EcoPack log."""
    # Parse all events
    events = []
    for line in log_path.read_text().splitlines():
        m = EVENT_RE.search(line)
        if not m:
            continue
        t = float(m.group(1))
        event_type = m.group(2)
        app = m.group(3)
        gpu_count = int(m.group(4))
        gpu_ids = [int(g.strip()) for g in m.group(5).split(",")]
        numa = int(m.group(6))
        runtime = None
        if event_type == "END":
            rm = RUNTIME_RE.search(line)
            if rm:
                runtime = float(rm.group(1))
        events.append({
            "t": t, "type": event_type, "app": app,
            "gpu_count": gpu_count, "gpu_ids": gpu_ids,
            "numa": numa, "runtime": runtime,
        })

    # Simulate to find overlapping pairs
    active = {}  # app -> {start_t, gpu_ids, numa}
    pairs = []   # list of (app, co_runner_or_None, gpu_ids, numa, runtime)

    for ev in events:
        if ev["type"] == "START":
            active[ev["app"]] = {
                "start_t": ev["t"], "gpu_ids": ev["gpu_ids"], "numa": ev["numa"],
            }
        elif ev["type"] == "END":
            app = ev["app"]
            info = active.pop(app, None)
            if info is None:
                continue
            # Who else was running during this app's lifetime?
            co_runners = [a for a in active if a != app]
            pairs.append({
                "app": app,
                "gpu_count": ev["gpu_count"],
                "gpu_ids": info["gpu_ids"],
                "numa": info["numa"],
                "runtime": ev["runtime"],
                "co_runners": co_runners,
            })

    return pairs


def print_corun_analysis(pairs):
    """Print co-run pair analysis."""
    print("Co-run pair analysis:")
    print("=" * 80)
    print("{:<30s} {:>6} {:>10} {:>10}  {:<30s}".format(
        "App", "#GPUs", "NUMA", "Runtime", "Co-runner(s)"))
    print("-" * 90)
    for p in pairs:
        co = ", ".join(p["co_runners"]) if p["co_runners"] else "(alone)"
        print("{:<30s} {:>6} {:>10} {:>10.2f}  {:<30s}".format(
            p["app"], p["gpu_count"], p["numa"],
            p["runtime"] if p["runtime"] else 0, co))
    print()


def build_pair_entries(app_a: str, gpus_a: List[int], numa_a: int,
                       app_b: str, gpus_b: List[int], numa_b: int) -> List[ScheduleEntry]:
    """Build a 2-entry schedule for a co-run pair test."""
    return [
        ScheduleEntry(order=0, app=app_a, gpu_count=len(gpus_a),
                      gpu_ids=gpus_a, numa_node=numa_a),
        ScheduleEntry(order=1, app=app_b, gpu_count=len(gpus_b),
                      gpu_ids=gpus_b, numa_node=numa_b),
    ]


def run_replay(entries: Sequence[ScheduleEntry], poll_interval: float = 1.0,
               results_dir=None, max_concurrent: int = 2):
    """Execute the schedule event-driven: launch next job when GPUs are free."""
    pending = list(entries)
    running: List[RunningJob] = []
    completed = []
    gpus_in_use: set = set()
    monitor = PowerMonitor()
    wall_start = time.time()

    print("Replay execution: {} apps on {} GPUs".format(len(entries), TOTAL_GPUS))
    print("NUMA 0 GPUs: {}  |  NUMA 1 GPUs: {}".format(NUMA0_GPUS, NUMA1_GPUS))
    print("=" * 80)

    monitor.start()

    try:
        while pending or running:
            # Try to launch as many pending jobs as possible
            launched = True
            while launched and pending and len(running) < max_concurrent:
                launched = False
                for entry in list(pending):
                    if not allocate_if_free(entry, gpus_in_use):
                        continue

                    cmd, env, cwd = build_command(
                        base_app_name(entry.app), entry.gpu_ids, entry.numa_node)

                    app_log = open(os.devnull, "w")
                    if results_dir is not None:
                        Path(results_dir).mkdir(parents=True, exist_ok=True)
                        app_log = open(Path(results_dir) / "log.txt", "a")

                    elapsed = time.time() - wall_start
                    print("  t={:8.2f}s | START {:<30s} | {} GPUs {} | NUMA {}".format(
                        elapsed, entry.app, entry.gpu_count, entry.gpu_ids,
                        entry.numa_node))

                    proc = subprocess.Popen(
                        cmd, env=env,
                        cwd=str(cwd) if cwd else None,
                        stdout=app_log, stderr=subprocess.STDOUT,
                    )
                    running.append(RunningJob(entry, proc, app_log, time.time()))
                    gpus_in_use.update(entry.gpu_ids)
                    pending.remove(entry)
                    launched = True
                    break  # re-check from top of pending after each launch

            # Poll running jobs
            time.sleep(poll_interval)
            for job in list(running):
                rc = job.process.poll()
                if rc is None:
                    continue

                elapsed = time.time() - wall_start
                runtime = time.time() - job.start_time
                gpus_in_use -= set(job.entry.gpu_ids)
                running.remove(job)
                job.log_file.close()

                status = "OK" if rc == 0 else "FAILED(rc={})".format(rc)
                print("  t={:8.2f}s | END   {:<30s} | freed {} GPUs {} | NUMA {} | "
                      "runtime={:.2f}s | {}".format(
                          elapsed, job.entry.app, len(job.entry.gpu_ids),
                          job.entry.gpu_ids, job.entry.numa_node, runtime, status))

                completed.append({
                    "app": job.entry.app,
                    "gpu_count": job.entry.gpu_count,
                    "gpu_ids": job.entry.gpu_ids,
                    "numa_node": job.entry.numa_node,
                    "runtime": runtime,
                    "return_code": rc,
                })

    finally:
        for job in running:
            try:
                job.process.terminate()
            except Exception:
                pass
            try:
                job.log_file.close()
            except Exception:
                pass
        monitor.stop()

    total_time = time.time() - wall_start

    # Summary
    print("\n" + "=" * 80)
    print("Replay summary:")
    print("{:<30s} {:>6} {:>14} {:>5} {:>12}".format(
        "App", "#GPUs", "GPU IDs", "NUMA", "Runtime (s)"))
    print("-" * 72)
    for item in completed:
        print("{:<30s} {:>6} {:>14} {:>5} {:>12.2f}".format(
            item["app"], item["gpu_count"], str(item["gpu_ids"]),
            item["numa_node"], item["runtime"]))
    print("-" * 72)
    print("\nTotal makespan: {:.2f}s".format(total_time))
    monitor.print_summary()


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Replay a co-scheduler schedule for repeated measurements.")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--log", type=Path,
                     help="EcoPack / co-scheduler log file to parse")
    src.add_argument("--schedule", type=Path,
                     help="Previously saved schedule JSON file")
    parser.add_argument("--save-schedule", type=Path, default=None,
                        help="Save parsed schedule to this JSON file")
    parser.add_argument("--dry-run", action="store_true",
                        help="Parse and print schedule without executing")
    parser.add_argument("--analyze", action="store_true",
                        help="Analyze co-run pairs from log and exit")
    parser.add_argument("--pair", type=str, nargs=2, metavar=("APP_A", "APP_B"),
                        help="Run a specific co-run pair test (uses GPU assignments from log)")
    parser.add_argument("--solo", type=str, metavar="APP",
                        help="Run a single app alone (uses GPU/NUMA assignment from log)")
    parser.add_argument("--repeats", type=int, default=1,
                        help="Number of repeat runs (default: 1)")
    parser.add_argument("--poll-interval", type=float, default=1.0,
                        help="Process polling interval in seconds (default: 1.0)")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR,
                        help="Directory for run logs (default: {})".format(
                            DEFAULT_RESULTS_DIR))
    parser.add_argument("--no-app-log", action="store_true",
                        help="Disable logging application stdout/stderr")
    args = parser.parse_args()

    # --analyze: show co-run pairs and exit
    if args.analyze:
        if not args.log:
            parser.error("--analyze requires --log")
        pairs = analyze_corun_pairs(args.log)
        print_corun_analysis(pairs)
        return

    # Parse or load schedule
    if args.log:
        entries = parse_ecopack_log(args.log)
        print(f"Parsed {len(entries)} jobs from {args.log}\n")
    else:
        entries = load_schedule_json(args.schedule)
        print(f"Loaded {len(entries)} jobs from {args.schedule}\n")

    # --solo or --pair: build a subset schedule from the parsed entries
    entry_map = {e.app: e for e in entries}

    if args.solo:
        if args.solo not in entry_map:
            parser.error(f"App '{args.solo}' not found in schedule. "
                         f"Available: {list(entry_map.keys())}")
        e = entry_map[args.solo]
        entries = [ScheduleEntry(order=0, app=e.app, gpu_count=e.gpu_count,
                                 gpu_ids=e.gpu_ids, numa_node=e.numa_node)]
        print(f"Solo run: {args.solo}\n")

    elif args.pair:
        app_a_name, app_b_name = args.pair
        if app_a_name not in entry_map:
            parser.error(f"App '{app_a_name}' not found in schedule. "
                         f"Available: {list(entry_map.keys())}")
        if app_b_name not in entry_map:
            parser.error(f"App '{app_b_name}' not found in schedule. "
                         f"Available: {list(entry_map.keys())}")
        ea, eb = entry_map[app_a_name], entry_map[app_b_name]
        entries = build_pair_entries(
            ea.app, ea.gpu_ids, ea.numa_node,
            eb.app, eb.gpu_ids, eb.numa_node,
        )
        print(f"Co-run pair test: {app_a_name} vs {app_b_name}\n")

    # Optionally save
    if args.save_schedule:
        save_schedule_json(entries, args.save_schedule)

    print_schedule(entries)

    if args.dry_run:
        return

    # Run (possibly multiple repeats)
    results_dir = args.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    app_log_dir = None if (args.no_app_log or not APP_LOG_ENABLED) else results_dir

    if args.solo:
        pair_tag = f"solo_{args.solo}"
    elif args.pair:
        pair_tag = f"pair_{args.pair[0]}_vs_{args.pair[1]}"
    else:
        pair_tag = "replay"

    for run_idx in range(args.repeats):
        run_label = f"run{run_idx + 1}" if args.repeats > 1 else "run"
        log_path = results_dir / f"{pair_tag}_{run_label}.txt"

        original_stdout = sys.stdout
        original_stderr = sys.stderr

        with log_path.open("w") as log_file:
            sys.stdout = TeeStream(original_stdout, log_file)
            sys.stderr = TeeStream(original_stderr, log_file)
            try:
                if args.repeats > 1:
                    print(f"\n{'#' * 80}")
                    print(f"# {pair_tag} {run_idx + 1} / {args.repeats}")
                    print(f"{'#' * 80}\n")
                print(f"Results log: {log_path}")
                run_replay(entries, args.poll_interval, results_dir=app_log_dir)
            finally:
                sys.stdout = original_stdout
                sys.stderr = original_stderr

        print(f"\nRun {run_idx + 1} saved to {log_path}")

    if args.repeats > 1:
        print(f"\nAll {args.repeats} runs complete. Logs in {results_dir}/{pair_tag}_run*.txt")


if __name__ == "__main__":
    main()
