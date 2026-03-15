#!/usr/bin/env python3
"""Exact offline energy-aware co-scheduling solver using OR-Tools CP-SAT.

Problem model:
- All jobs are available at time 0.
- Each job chooses exactly one GPU count from the rows available in edp_metrics.txt.
- At most 2 jobs may run concurrently.
- The node has 2 NUMA domains and 4 GPUs total, with capacity 2 GPUs per NUMA.
- Feasible placements follow the topology-aware patterns discussed in the project:
  * 1 GPU: (1,0) or (0,1)
  * 2 GPUs: (2,0) or (0,2)
  * 3 GPUs: (2,1) or (1,2)
  * 4 GPUs: (2,2)
- Runtime and average power are known for each (job, gpu_count) pair.
- Cross-application interference is ignored.

Objective:
    minimize total node energy
  = sum(active job energy) + idle_power * total_idle_gpu_time

The script prints the chosen GPU count, placement, start/end times, and a final
execution order.
"""

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, NamedTuple, Sequence, Tuple

try:
    from ortools.sat.python import cp_model
except ImportError:
    raise SystemExit(
        "ortools is not installed for this interpreter. Run with "
        "/home/ac.zzheng/venv_sched/bin/python or install ortools first."
    )

TIME_SCALE = 100
POWER_SCALE = 100
DEFAULT_IDLE_POWER = 70.0
DEFAULT_THREADS = 8
DEFAULT_TIME_LIMIT_S = 20.0
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = SCRIPT_DIR / "results"
DEFAULT_SCHEDULE_OUTPUT = DEFAULT_RESULTS_DIR / "solver_schedule.txt"
DEFAULT_JOBS = [
    "pot3d", "minisweep", "lbm", "cloverleaf", "tealeaf",
    "miniweather", "hpgmg", "bert", "gpt2", "resnet50",
]


class ModeRow(NamedTuple):
    job: str
    gpu_count: int
    runtime_s: float
    duration_ticks: int
    avg_power_w: float
    total_power_w: float
    active_energy_j: float


class Config(NamedTuple):
    job: str
    gpu_count: int
    placement: Tuple[int, int]
    runtime_s: float
    duration_ticks: int
    avg_power_w: float
    total_power_w: float
    active_energy_j: float
    objective_coeff: int


class SolveRecord(NamedTuple):
    start_s: float
    end_s: float
    job: str
    gpu_count: int
    placement: Tuple[int, int]
    runtime_s: float
    avg_power_w: float
    total_power_w: float
    active_energy_j: float


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


def placement_candidates(gpu_count: int) -> List[Tuple[int, int]]:
    if gpu_count == 1:
        return [(1, 0), (0, 1)]
    if gpu_count == 2:
        return [(2, 0), (0, 2)]
    if gpu_count == 3:
        return [(2, 1), (1, 2)]
    if gpu_count == 4:
        return [(2, 2)]
    raise ValueError("Unsupported gpu count: {}".format(gpu_count))


def parse_metrics(metrics_path: Path, selected_jobs: Sequence[str]) -> Dict[str, Dict[int, ModeRow]]:
    section_re = re.compile(r"^===== .*?/([^/ ]+) =====$")
    current_job = None
    rows = {}

    for raw_line in metrics_path.read_text().splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = section_re.match(line)
        if match:
            current_job = match.group(1)
            rows.setdefault(current_job, {})
            continue
        if current_job is None or line.startswith("cap=") or line.startswith("gpu_count"):
            continue

        parts = line.split()
        if len(parts) < 3:
            continue

        gpu_count = int(parts[0])
        runtime_s = float(parts[1])
        avg_power_w = float(parts[2])
        duration_ticks = int(round(runtime_s * TIME_SCALE))
        total_power_w = gpu_count * avg_power_w
        rows[current_job][gpu_count] = ModeRow(
            job=current_job,
            gpu_count=gpu_count,
            runtime_s=runtime_s,
            duration_ticks=duration_ticks,
            avg_power_w=avg_power_w,
            total_power_w=total_power_w,
            active_energy_j=runtime_s * total_power_w,
        )

    missing = [job for job in selected_jobs if job not in rows]
    if missing:
        raise ValueError("Missing jobs in metrics file: {}".format(missing))

    return {job: rows[job] for job in selected_jobs}


def build_configs(
    parsed: Dict[str, Dict[int, ModeRow]],
    selected_jobs: Sequence[str],
    idle_power_w: float,
) -> Tuple[Dict[str, List[Config]], int]:
    idle_power_scaled = int(round(idle_power_w * POWER_SCALE))
    configs_by_job = {}
    horizon = 0

    for job in selected_jobs:
        job_configs = []
        max_duration = 0
        for gpu_count in sorted(parsed[job]):
            row = parsed[job][gpu_count]
            max_duration = max(max_duration, row.duration_ticks)
            total_power_scaled = int(round(row.total_power_w * POWER_SCALE))
            objective_coeff = row.duration_ticks * (total_power_scaled - idle_power_scaled * gpu_count)
            for placement in placement_candidates(gpu_count):
                job_configs.append(
                    Config(
                        job=job,
                        gpu_count=gpu_count,
                        placement=placement,
                        runtime_s=row.runtime_s,
                        duration_ticks=row.duration_ticks,
                        avg_power_w=row.avg_power_w,
                        total_power_w=row.total_power_w,
                        active_energy_j=row.active_energy_j,
                        objective_coeff=objective_coeff,
                    )
                )
        configs_by_job[job] = job_configs
        horizon += max_duration

    return configs_by_job, horizon


def solve_schedule(
    metrics_path: Path,
    jobs: Sequence[str],
    idle_power_w: float,
    threads: int,
    time_limit_s: float,
):
    parsed = parse_metrics(metrics_path, jobs)
    configs_by_job, horizon = build_configs(parsed, jobs, idle_power_w)
    idle_power_scaled = int(round(idle_power_w * POWER_SCALE))

    model = cp_model.CpModel()
    cmax = model.NewIntVar(0, horizon, "cmax")

    all_intervals = []
    numa0_demands = []
    numa1_demands = []
    concurrency_demands = []
    vars_by_job = {}
    objective_terms = [4 * idle_power_scaled * cmax]

    for job in jobs:
        presences = []
        job_vars = []
        for idx, cfg in enumerate(configs_by_job[job]):
            label = "{}_g{}_p{}{}_{}".format(job, cfg.gpu_count, cfg.placement[0], cfg.placement[1], idx)
            start = model.NewIntVar(0, horizon, "s_{}".format(label))
            end = model.NewIntVar(0, horizon, "e_{}".format(label))
            use = model.NewBoolVar("x_{}".format(label))
            interval = model.NewOptionalIntervalVar(start, cfg.duration_ticks, end, use, "i_{}".format(label))

            model.Add(end <= cmax).OnlyEnforceIf(use)

            presences.append(use)
            all_intervals.append(interval)
            numa0_demands.append(cfg.placement[0])
            numa1_demands.append(cfg.placement[1])
            concurrency_demands.append(1)
            objective_terms.append(cfg.objective_coeff * use)
            job_vars.append((cfg, use, start, end))

        model.Add(sum(presences) == 1)
        vars_by_job[job] = job_vars

    model.AddCumulative(all_intervals, numa0_demands, 2)
    model.AddCumulative(all_intervals, numa1_demands, 2)
    model.AddCumulative(all_intervals, concurrency_demands, 2)
    model.Minimize(sum(objective_terms))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_s
    solver.parameters.num_search_workers = threads
    solver.parameters.log_search_progress = False

    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError("Solver did not find a schedule. Status={}".format(status))

    schedule = []
    for job in jobs:
        chosen = None
        for cfg, use, start, end in vars_by_job[job]:
            if solver.Value(use):
                chosen = SolveRecord(
                    start_s=solver.Value(start) / float(TIME_SCALE),
                    end_s=solver.Value(end) / float(TIME_SCALE),
                    job=job,
                    gpu_count=cfg.gpu_count,
                    placement=cfg.placement,
                    runtime_s=cfg.runtime_s,
                    avg_power_w=cfg.avg_power_w,
                    total_power_w=cfg.total_power_w,
                    active_energy_j=cfg.active_energy_j,
                )
                break
        if chosen is None:
            raise RuntimeError("No chosen configuration found for {}".format(job))
        schedule.append(chosen)

    schedule.sort(key=lambda rec: (rec.start_s, rec.end_s, rec.job))
    makespan_s = max((rec.end_s for rec in schedule), default=0.0)
    active_energy_j = sum(rec.active_energy_j for rec in schedule)
    busy_gpu_time = sum(rec.gpu_count * rec.runtime_s for rec in schedule)
    idle_energy_j = idle_power_w * max(0.0, 4.0 * makespan_s - busy_gpu_time)
    total_energy_j = active_energy_j + idle_energy_j
    objective_value = solver.ObjectiveValue() / float(TIME_SCALE * POWER_SCALE)

    return {
        "status": status,
        "schedule": schedule,
        "makespan_s": makespan_s,
        "active_energy_j": active_energy_j,
        "idle_energy_j": idle_energy_j,
        "total_energy_j": total_energy_j,
        "objective_value": objective_value,
        "wall_time_s": solver.WallTime(),
        "best_bound": solver.BestObjectiveBound() / float(TIME_SCALE * POWER_SCALE),
    }


def print_summary(result, idle_power_w: float):
    schedule = result["schedule"]
    print("Exact offline energy-optimal schedule")
    print("=" * 96)
    print("Idle power per GPU: {:.2f} W".format(idle_power_w))
    print("Makespan: {:.2f} s".format(result["makespan_s"]))
    print(
        "Active energy: {:.2f} J ({:.2f} kJ)".format(
            result["active_energy_j"], result["active_energy_j"] / 1000.0
        )
    )
    print(
        "Idle energy: {:.2f} J ({:.2f} kJ)".format(
            result["idle_energy_j"], result["idle_energy_j"] / 1000.0
        )
    )
    print(
        "Total energy: {:.2f} J ({:.2f} kJ)".format(
            result["total_energy_j"], result["total_energy_j"] / 1000.0
        )
    )
    print("Objective value: {:.2f} J".format(result["objective_value"]))
    print("Best bound: {:.2f} J".format(result["best_bound"]))
    print("Solver wall time: {:.2f} s".format(result["wall_time_s"]))
    print()
    print(
        "{:<4} {:>8} {:>8} {:<15} {:>4} {:>8} {:>8} {:>10} {:>10} {:>12}".format(
            "#", "Start", "End", "App", "GPU", "Place", "Runtime", "P_gpu(W)", "P_tot(W)", "E_act(J)"
        )
    )
    print("-" * 96)
    for idx, rec in enumerate(schedule, 1):
        place = "({},{})".format(rec.placement[0], rec.placement[1])
        print(
            "{:<4} {:>8.2f} {:>8.2f} {:<15} {:>4} {:>8} {:>8.2f} {:>10.2f} {:>10.2f} {:>12.2f}".format(
                idx,
                rec.start_s,
                rec.end_s,
                rec.job,
                rec.gpu_count,
                place,
                rec.runtime_s,
                rec.avg_power_w,
                rec.total_power_w,
                rec.active_energy_j,
            )
        )
    print()
    print("Execution order:")
    for idx, rec in enumerate(schedule, 1):
        print(
            "  {}. t={:.2f}s start {} on {} GPU(s) at placement {}".format(
                idx, rec.start_s, rec.job, rec.gpu_count, rec.placement
            )
        )


def main():
    parser = argparse.ArgumentParser(description="Solve the exact offline energy-aware co-scheduling problem with CP-SAT.")
    parser.add_argument(
        "--metrics-file",
        type=Path,
        default=Path("/home/ac.zzheng/power/GPGPU/data/H100/edp_metrics.txt"),
        help="Path to edp_metrics.txt",
    )
    parser.add_argument(
        "--jobs",
        nargs="+",
        default=DEFAULT_JOBS,
        help="Jobs to schedule. Default: {}".format(DEFAULT_JOBS),
    )
    parser.add_argument(
        "--idle-power",
        type=float,
        default=DEFAULT_IDLE_POWER,
        help="Idle power per GPU in watts. Default: {}".format(DEFAULT_IDLE_POWER),
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=DEFAULT_THREADS,
        help="CP-SAT worker threads. Default: {}".format(DEFAULT_THREADS),
    )
    parser.add_argument(
        "--time-limit",
        type=float,
        default=DEFAULT_TIME_LIMIT_S,
        help="CP-SAT time limit in seconds. Default: {}".format(DEFAULT_TIME_LIMIT_S),
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=DEFAULT_SCHEDULE_OUTPUT,
        help="Schedule output text file. Default: {}".format(DEFAULT_SCHEDULE_OUTPUT),
    )
    args = parser.parse_args()

    output_file = args.output_file
    output_file.parent.mkdir(parents=True, exist_ok=True)

    original_stdout = sys.stdout
    original_stderr = sys.stderr
    with output_file.open("w") as log_file:
        sys.stdout = TeeStream(original_stdout, log_file)
        sys.stderr = TeeStream(original_stderr, log_file)
        try:
            print("Solver schedule output: {}".format(output_file))
            result = solve_schedule(
                metrics_path=args.metrics_file,
                jobs=args.jobs,
                idle_power_w=args.idle_power,
                threads=args.threads,
                time_limit_s=args.time_limit,
            )
            print_summary(result, args.idle_power)
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr


if __name__ == "__main__":
    main()
