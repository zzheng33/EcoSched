#!/usr/bin/env python3
"""Hand-crafted event-driven online co-scheduler.

This launcher uses per-application profiled metrics from perf_metrics.txt and
makes decisions at each scheduling event:
- enumerate feasible actions on the currently free GPUs
- predict normalized runtime from the best available per-app signal
  (dram_sum, else fp_sum/gpu_count, else sm_sum/gpu_count)
- score each action with a hand-crafted online objective
- launch the minimum-score action

Policy design:
- predicted normalized runtime is used as a guardrail: only modes within a
  slowdown tolerance of the app's best predicted mode are online-feasible
- the main score term is normalized energy regret (or EDP regret)
- idle GPUs are penalized using a weight derived from idle/busy power
- actions that leave a residual GPU budget few remaining jobs can use receive a
  blocking penalty
"""

import argparse
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

from config import (
    DEFAULT_JOB_QUEUE,
    NUMA0_GPUS,
    NUMA1_GPUS,
    TOTAL_GPUS,
)
from run_cosched_sequential import (
    PowerMonitor,
    allocate_gpus_numa,
    build_command,
    pick_numa_for_tenant,
)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = SCRIPT_DIR / "results"
DEFAULT_IDLE_POWER = 70.0
DEFAULT_SLOWDOWN_TOL = 0.2
DEFAULT_SCORE_METRIC = "energy"
DEFAULT_ANCHOR_APP = "pot3d"
DEFAULT_MAX_CONCURRENT = 2


class ModeInfo(NamedTuple):
    app: str
    gpu_count: int
    runtime_s: float
    avg_power_w: float
    total_power_w: float
    active_energy_j: float
    edp: float
    dram_sum: float
    sm_sum: float
    fp_sum: float
    predictor_name: str
    predictor_value: float
    norm_runtime: float
    norm_energy: float
    norm_edp: float


class ActionEval(NamedTuple):
    score: float
    metric_regret: float
    idle_frac: float
    blocking_penalty: float
    runtime_regret: float


class CMABDecision(NamedTuple):
    predicted_reward: float
    exploration_bonus: float
    ucb_value: float


class CMABInterval(NamedTuple):
    features: Tuple[float, float]
    idle_frac: float
    busy_total_power_w: float
    start_time_s: float
    sample_start_idx: int
    action_label: str


class LaunchSpec(NamedTuple):
    mode: ModeInfo
    gpu_ids: List[int]
    numa_node: int


class RunningJob(object):
    def __init__(self, app, mode, gpu_ids, numa_node, process, start_time, devnull):
        self.app = app
        self.mode = mode
        self.gpu_ids = gpu_ids
        self.numa_node = numa_node
        self.process = process
        self.start_time = start_time
        self.devnull = devnull


class SimJob(NamedTuple):
    app: str
    mode: ModeInfo
    gpu_ids: List[int]
    numa_node: int
    end_time: float


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


SECTION_RE = re.compile(r"^===== .*?/([^/ ]+) =====$")


def _predictor_name_and_value(rows: Dict[int, Tuple[float, float, float, float, float, float, float, float]]) -> Tuple[str, Dict[int, float]]:
    dram_values = {gpu_count: values[5] for gpu_count, values in rows.items()}
    if any(value > 0.0 for value in dram_values.values()):
        return "dram_sum", dram_values

    fp_per_gpu = {
        gpu_count: (values[7] / float(gpu_count))
        for gpu_count, values in rows.items()
    }
    if any(value > 0.0 for value in fp_per_gpu.values()):
        return "fp_sum/gpu_count", fp_per_gpu

    sm_per_gpu = {
        gpu_count: (values[6] / float(gpu_count))
        for gpu_count, values in rows.items()
    }
    return "sm_sum/gpu_count", sm_per_gpu


def _predicted_norm_runtime(predictor_values: Dict[int, float]) -> Dict[int, float]:
    max_value = max(predictor_values.values())
    min_value = min(predictor_values.values())

    if max_value <= 0.0:
        return {gpu_count: 1.0 for gpu_count in predictor_values}

    if abs(max_value - min_value) < 1e-12:
        return {gpu_count: 1.0 for gpu_count in predictor_values}

    norm_runtime = {}
    for gpu_count, value in predictor_values.items():
        if value <= 0.0:
            raise ValueError(
                "Predictor value must be positive for all modes when using predictor-based runtime; "
                "got {} for {} GPUs".format(value, gpu_count)
            )
        norm_runtime[gpu_count] = max_value / value
    return norm_runtime


def parse_metrics(metrics_path: Path, selected_jobs: Sequence[str]) -> Dict[str, Dict[int, ModeInfo]]:
    raw_rows = {}
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
        if len(parts) < 6:
            raise ValueError(
                "Expected 6 columns per metrics row in {} but got {} for line: {}".format(
                    metrics_path, len(parts), line
                )
            )
        gpu_count = int(parts[0])
        runtime_s = float(parts[1])
        avg_power_w = float(parts[2])
        dram_sum = float(parts[3])
        sm_sum = float(parts[4])
        fp_sum = float(parts[5])
        total_power_w = gpu_count * avg_power_w
        active_energy_j = runtime_s * total_power_w
        edp = active_energy_j * runtime_s
        raw_rows[current_job][gpu_count] = (
            runtime_s,
            avg_power_w,
            total_power_w,
            active_energy_j,
            edp,
            dram_sum,
            sm_sum,
            fp_sum,
        )

    missing = [job for job in selected_jobs if job not in raw_rows]
    if missing:
        raise ValueError("Missing jobs in metrics file: {}".format(missing))

    parsed = {}
    for job in selected_jobs:
        rows = raw_rows[job]
        predictor_name, predictor_values = _predictor_name_and_value(rows)
        norm_runtime_by_gpu = _predicted_norm_runtime(predictor_values)
        predicted_energy = {
            gpu_count: rows[gpu_count][2] * norm_runtime_by_gpu[gpu_count]
            for gpu_count in rows
        }
        predicted_edp = {
            gpu_count: rows[gpu_count][2] * (norm_runtime_by_gpu[gpu_count] ** 2)
            for gpu_count in rows
        }
        min_predicted_energy = min(predicted_energy.values())
        min_predicted_edp = min(predicted_edp.values())

        parsed[job] = {}
        for gpu_count, values in rows.items():
            runtime_s, avg_power_w, total_power_w, active_energy_j, edp, dram_sum, sm_sum, fp_sum = values
            parsed[job][gpu_count] = ModeInfo(
                app=job,
                gpu_count=gpu_count,
                runtime_s=runtime_s,
                avg_power_w=avg_power_w,
                total_power_w=total_power_w,
                active_energy_j=active_energy_j,
                edp=edp,
                dram_sum=dram_sum,
                sm_sum=sm_sum,
                fp_sum=fp_sum,
                predictor_name=predictor_name,
                predictor_value=predictor_values[gpu_count],
                norm_runtime=norm_runtime_by_gpu[gpu_count],
                norm_energy=predicted_energy[gpu_count] / min_predicted_energy,
                norm_edp=predicted_edp[gpu_count] / min_predicted_edp,
            )
    return parsed


def select_online_modes(
    parsed: Dict[str, Dict[int, ModeInfo]],
    slowdown_tol: float,
) -> Dict[str, Tuple[ModeInfo, ...]]:
    selected = {}
    for app, rows in parsed.items():
        modes = tuple(rows[g] for g in sorted(rows))
        feasible = [m for m in modes if m.norm_runtime <= 1.0 + slowdown_tol + 1e-9]
        if not feasible:
            min_norm_rt = min(m.norm_runtime for m in modes)
            feasible = [m for m in modes if abs(m.norm_runtime - min_norm_rt) < 1e-9]
        selected[app] = tuple(sorted(feasible, key=lambda m: (m.gpu_count, m.norm_energy, m.norm_edp)))
    return selected


class LinUCBPolicy(object):
    def __init__(self, idle_weight: float, alpha: float, prior_strength: float = 10.0):
        self.alpha = alpha
        self.prior_strength = prior_strength
        self.A = [
            [prior_strength, 0.0],
            [0.0, prior_strength],
        ]
        self.b = [
            -prior_strength * 1.0,
            -prior_strength * idle_weight,
        ]

    def features(self, evaluation: ActionEval) -> Tuple[float, float]:
        return (evaluation.metric_regret, evaluation.idle_frac)

    def _inverse(self) -> List[List[float]]:
        a, b = self.A[0]
        c, d = self.A[1]
        det = a * d - b * c
        if abs(det) < 1e-12:
            raise ValueError('Singular LinUCB covariance matrix')
        inv_det = 1.0 / det
        return [
            [d * inv_det, -b * inv_det],
            [-c * inv_det, a * inv_det],
        ]

    def theta(self) -> Tuple[float, float]:
        inv = self._inverse()
        return (
            inv[0][0] * self.b[0] + inv[0][1] * self.b[1],
            inv[1][0] * self.b[0] + inv[1][1] * self.b[1],
        )

    def evaluate(self, features: Tuple[float, float]) -> CMABDecision:
        inv = self._inverse()
        theta0, theta1 = self.theta()
        x0, x1 = features
        predicted_reward = theta0 * x0 + theta1 * x1
        inv_x0 = inv[0][0] * x0 + inv[0][1] * x1
        inv_x1 = inv[1][0] * x0 + inv[1][1] * x1
        variance = max(0.0, x0 * inv_x0 + x1 * inv_x1)
        exploration_bonus = self.alpha * math.sqrt(variance)
        return CMABDecision(
            predicted_reward=predicted_reward,
            exploration_bonus=exploration_bonus,
            ucb_value=predicted_reward + exploration_bonus,
        )

    def update(self, features: Tuple[float, float], reward: float) -> None:
        x0, x1 = features
        self.A[0][0] += x0 * x0
        self.A[0][1] += x0 * x1
        self.A[1][0] += x1 * x0
        self.A[1][1] += x1 * x1
        self.b[0] += reward * x0
        self.b[1] += reward * x1


def mean_busy_power(parsed: Dict[str, Dict[int, ModeInfo]]) -> float:
    values = []
    for rows in parsed.values():
        values.extend(mode.avg_power_w for mode in rows.values())
    return sum(values) / float(len(values))


def action_key(action: Sequence[ModeInfo]) -> Tuple[Tuple[str, int], ...]:
    return tuple(sorted((mode.app, mode.gpu_count) for mode in action))


def enumerate_actions(
    pending: Sequence[str],
    candidate_modes: Dict[str, Tuple[ModeInfo, ...]],
    free_gpus: int,
    free_slots: int,
) -> List[Tuple[ModeInfo, ...]]:
    if free_gpus <= 0 or free_slots <= 0:
        return []

    actions = []
    seen = set()

    for app in pending:
        for mode in candidate_modes[app]:
            if mode.gpu_count <= free_gpus:
                action = (mode,)
                key = action_key(action)
                if key not in seen:
                    seen.add(key)
                    actions.append(action)

    if free_slots < 2:
        return actions

    for i, app_a in enumerate(pending):
        for app_b in pending[i + 1:]:
            for mode_a in candidate_modes[app_a]:
                for mode_b in candidate_modes[app_b]:
                    if mode_a.gpu_count + mode_b.gpu_count > free_gpus:
                        continue
                    action = tuple(sorted((mode_a, mode_b), key=lambda m: (m.app, m.gpu_count)))
                    key = action_key(action)
                    if key not in seen:
                        seen.add(key)
                        actions.append(action)

    return actions


def score_action(
    action: Sequence[ModeInfo],
    pending: Sequence[str],
    candidate_modes: Dict[str, Tuple[ModeInfo, ...]],
    free_gpus: int,
    score_metric: str,
    idle_weight: float,
    blocking_weight: float,
) -> ActionEval:
    used_gpus = sum(mode.gpu_count for mode in action)
    leftover_gpus = max(0, free_gpus - used_gpus)
    remaining_jobs = [app for app in pending if app not in {mode.app for mode in action}]

    if score_metric == "edp":
        regrets = [mode.norm_edp - 1.0 for mode in action]
    else:
        regrets = [mode.norm_energy - 1.0 for mode in action]
    metric_regret = sum(regrets) / float(len(regrets))

    runtime_regret = sum((mode.norm_runtime - 1.0) for mode in action) / float(len(action))
    idle_frac = leftover_gpus / float(TOTAL_GPUS)

    if leftover_gpus == 0 or not remaining_jobs:
        blocking_penalty = 0.0
    else:
        compatible = 0
        for app in remaining_jobs:
            if any(mode.gpu_count <= leftover_gpus for mode in candidate_modes[app]):
                compatible += 1
        fit_fraction = compatible / float(len(remaining_jobs))
        blocking_penalty = 1.0 - fit_fraction

    score = metric_regret + idle_weight * idle_frac + blocking_weight * blocking_penalty
    return ActionEval(
        score=score,
        metric_regret=metric_regret,
        idle_frac=idle_frac,
        blocking_penalty=blocking_penalty,
        runtime_regret=runtime_regret,
    )


def _spill_count(gpu_ids: Sequence[int], numa_node: int) -> int:
    local = set(NUMA0_GPUS if numa_node == 0 else NUMA1_GPUS)
    return sum(1 for gpu in gpu_ids if gpu not in local)


def materialize_action(
    action: Sequence[ModeInfo],
    running: Sequence[RunningJob],
    gpus_in_use: set,
) -> Optional[List[LaunchSpec]]:
    if len(running) + len(action) > DEFAULT_MAX_CONCURRENT:
        return None

    if running:
        if len(action) != 1:
            return None
        numa_node = pick_numa_for_tenant(list(running))
        if numa_node is None:
            return None
        mode = action[0]
        gpu_ids = allocate_gpus_numa(mode.gpu_count, numa_node, set(gpus_in_use))
        if gpu_ids is None:
            return None
        return [LaunchSpec(mode=mode, gpu_ids=gpu_ids, numa_node=numa_node)]

    if len(action) == 1:
        mode = action[0]
        gpu_ids = allocate_gpus_numa(mode.gpu_count, 0, set(gpus_in_use))
        if gpu_ids is None:
            return None
        return [LaunchSpec(mode=mode, gpu_ids=gpu_ids, numa_node=0)]

    if len(action) != 2:
        return None

    best = None
    assignments = [(0, 1), (1, 0)]
    for numa_a, numa_b in assignments:
        used = set(gpus_in_use)
        specs = [None, None]
        indexed = list(enumerate(((action[0], numa_a), (action[1], numa_b))))
        indexed.sort(key=lambda item: (-item[1][0].gpu_count, item[0]))
        feasible = True
        spill = 0
        for original_idx, (mode, numa_node) in indexed:
            gpu_ids = allocate_gpus_numa(mode.gpu_count, numa_node, used)
            if gpu_ids is None:
                feasible = False
                break
            spill += _spill_count(gpu_ids, numa_node)
            used.update(gpu_ids)
            specs[original_idx] = LaunchSpec(mode=mode, gpu_ids=gpu_ids, numa_node=numa_node)
        if feasible:
            candidate = (spill, tuple((spec.numa_node, tuple(spec.gpu_ids)) for spec in specs), specs)
            if best is None or candidate < best:
                best = candidate

    return None if best is None else best[2]


def pick_best_action(
    pending: Sequence[str],
    running: Sequence[RunningJob],
    gpus_in_use: set,
    candidate_modes: Dict[str, Tuple[ModeInfo, ...]],
    score_metric: str,
    idle_weight: float,
    blocking_weight: float,
    anchor_app: Optional[str],
    anchor_started: bool,
    policy: str,
    cmab_policy: Optional[LinUCBPolicy],
) -> Tuple[Optional[Tuple[ModeInfo, ...]], Optional[List[LaunchSpec]], Optional[ActionEval], Optional[CMABDecision]]:
    free_gpus = TOTAL_GPUS - len(gpus_in_use)
    free_slots = DEFAULT_MAX_CONCURRENT - len(running)
    actions = enumerate_actions(pending, candidate_modes, free_gpus, free_slots)
    if anchor_app and not anchor_started and anchor_app in pending:
        actions = [action for action in actions if any(mode.app == anchor_app for mode in action)]

    best_item = None
    for action in actions:
        launches = materialize_action(action, running, gpus_in_use)
        if launches is None:
            continue
        evaluation = score_action(action, pending, candidate_modes, free_gpus, score_metric, idle_weight, blocking_weight)
        if policy == "cmab":
            if cmab_policy is None:
                raise ValueError("cmab policy selected without a LinUCB policy instance")
            decision = cmab_policy.evaluate(cmab_policy.features(evaluation))
            tie_key = (
                -decision.ucb_value,
                evaluation.idle_frac,
                evaluation.runtime_regret,
                -sum(mode.gpu_count for mode in action),
                action_key(action),
            )
        else:
            decision = None
            tie_key = (
                evaluation.score,
                evaluation.idle_frac,
                evaluation.runtime_regret,
                -sum(mode.gpu_count for mode in action),
                action_key(action),
            )
        candidate = (tie_key, action, launches, evaluation, decision)
        if best_item is None or candidate < best_item:
            best_item = candidate

    if best_item is None:
        return None, None, None, None
    _, action, launches, evaluation, decision = best_item
    return action, launches, evaluation, decision


def _action_label(action: Sequence[ModeInfo]) -> str:
    return "+".join("{}:{}GPU".format(mode.app, mode.gpu_count) for mode in action)


def _energy_between_samples(samples, start_idx: int, end_idx: Optional[int] = None) -> float:
    if end_idx is None:
        end_idx = len(samples)
    if end_idx - start_idx < 2:
        return 0.0
    total = 0.0
    begin = max(start_idx + 1, 1)
    for i in range(begin, end_idx):
        t0, p0 = samples[i - 1]
        t1, p1 = samples[i]
        dt = t1 - t0
        avg_power = sum((a + b) / 2.0 for a, b in zip(p0, p1))
        total += avg_power * dt
    return total


def _make_cmab_interval(
    action: Sequence[ModeInfo],
    evaluation: ActionEval,
    running: Sequence[RunningJob],
    start_time_s: float,
    sample_start_idx: int,
) -> CMABInterval:
    busy_total_power_w = sum(job.mode.total_power_w for job in running)
    return CMABInterval(
        features=(evaluation.metric_regret, evaluation.idle_frac),
        idle_frac=evaluation.idle_frac,
        busy_total_power_w=busy_total_power_w,
        start_time_s=start_time_s,
        sample_start_idx=sample_start_idx,
        action_label=_action_label(action),
    )


def _compute_cmab_reward(
    interval: CMABInterval,
    duration_s: float,
    idle_power_w: float,
    idle_weight: float,
    mean_busy_power_w: float,
    total_energy_j: Optional[float] = None,
) -> Tuple[float, float]:
    if duration_s <= 0.0:
        return -(idle_weight * interval.idle_frac), 0.0

    idle_gpus = int(round(interval.idle_frac * TOTAL_GPUS))
    if total_energy_j is None:
        total_energy_j = duration_s * (interval.busy_total_power_w + idle_power_w * idle_gpus)

    idle_energy_j = idle_power_w * idle_gpus * duration_s
    active_energy_j = max(0.0, total_energy_j - idle_energy_j)
    denom = mean_busy_power_w * TOTAL_GPUS * duration_s
    realized_metric = 0.0 if denom <= 0.0 else active_energy_j / denom
    reward = -(realized_metric + idle_weight * interval.idle_frac)
    return reward, realized_metric


def _cmab_update_message(
    interval: CMABInterval,
    reward: float,
    realized_metric: float,
    cmab_policy: LinUCBPolicy,
) -> str:
    theta0, theta1 = cmab_policy.theta()
    return (
        "    CMAB update | action={} | x=({:.4f}, {:.4f}) | realized_metric={:.4f} | "
        "reward={:.4f} | theta=({:.4f}, {:.4f})"
    ).format(
        interval.action_label,
        interval.features[0],
        interval.features[1],
        realized_metric,
        reward,
        theta0,
        theta1,
    )


def _devnull():
    return open(os.devnull, "w")


def run_dry(
    job_queue: Sequence[str],
    candidate_modes: Dict[str, Tuple[ModeInfo, ...]],
    score_metric: str,
    idle_weight: float,
    blocking_weight: float,
    anchor_app: Optional[str],
    policy: str,
    cmab_policy: Optional[LinUCBPolicy],
    idle_power: float,
    mean_busy_power_w: float,
):
    pending = list(job_queue)
    running = []
    gpus_in_use = set()
    sim_time = 0.0
    anchor_started = anchor_app is None
    pending_interval = None

    print(
        "Online dry-run: {} apps on {} GPUs (metric={}, slowdown_tol modes pre-filtered, policy={})".format(
            len(pending), TOTAL_GPUS, score_metric, policy
        )
    )
    print("NUMA 0 GPUs: {}  |  NUMA 1 GPUs: {}".format(NUMA0_GPUS, NUMA1_GPUS))
    print("=" * 96)

    while pending or running:
        scheduled = True
        while scheduled and pending and len(running) < DEFAULT_MAX_CONCURRENT:
            scheduled = False
            action, launches, evaluation, cmab_decision = pick_best_action(
                pending,
                running,
                gpus_in_use,
                candidate_modes,
                score_metric,
                idle_weight,
                blocking_weight,
                anchor_app,
                anchor_started,
                policy,
                cmab_policy,
            )
            if not launches:
                break
            for spec in launches:
                end_time = sim_time + spec.mode.runtime_s
                running.append(SimJob(spec.mode.app, spec.mode, spec.gpu_ids, spec.numa_node, end_time))
                gpus_in_use.update(spec.gpu_ids)
                pending.remove(spec.mode.app)
                if anchor_app and spec.mode.app == anchor_app:
                    anchor_started = True
                message = (
                    "  t={:8.2f}s | START {:<15} | {} GPUs {} | NUMA {} | score={:.4f} | "
                    "regret={:.4f} idle={:.2f} block={:.2f}"
                ).format(
                    sim_time,
                    spec.mode.app,
                    spec.mode.gpu_count,
                    spec.gpu_ids,
                    spec.numa_node,
                    evaluation.score,
                    evaluation.metric_regret,
                    evaluation.idle_frac,
                    evaluation.blocking_penalty,
                )
                if cmab_decision is not None:
                    message += " | cmab_ucb={:.4f} mu={:.4f} bonus={:.4f}".format(
                        cmab_decision.ucb_value,
                        cmab_decision.predicted_reward,
                        cmab_decision.exploration_bonus,
                    )
                print(message)
            if policy == "cmab":
                pending_interval = _make_cmab_interval(action, evaluation, running, sim_time, 0)
                break
            scheduled = True

        if not running:
            break

        running.sort(key=lambda job: (job.end_time, job.app))
        finished = running.pop(0)
        if policy == "cmab" and pending_interval is not None and cmab_policy is not None:
            duration = finished.end_time - pending_interval.start_time_s
            reward, realized_metric = _compute_cmab_reward(
                pending_interval,
                duration,
                idle_power,
                idle_weight,
                mean_busy_power_w,
            )
            cmab_policy.update(pending_interval.features, reward)
            print(_cmab_update_message(pending_interval, reward, realized_metric, cmab_policy))
            pending_interval = None
        sim_time = finished.end_time
        gpus_in_use -= set(finished.gpu_ids)
        print(
            "  t={:8.2f}s | END   {:<15} | freed {} GPUs {} | NUMA {} | runtime={:.2f}s".format(
                sim_time,
                finished.app,
                len(finished.gpu_ids),
                finished.gpu_ids,
                finished.numa_node,
                finished.mode.runtime_s,
            )
        )

    print("\nEstimated makespan: {:.2f}s".format(sim_time))


def run_online(
    job_queue: Sequence[str],
    candidate_modes: Dict[str, Tuple[ModeInfo, ...]],
    score_metric: str,
    idle_weight: float,
    blocking_weight: float,
    poll_interval: float,
    anchor_app: Optional[str],
    policy: str,
    cmab_policy: Optional[LinUCBPolicy],
    idle_power: float,
    mean_busy_power_w: float,
):
    pending = list(job_queue)
    running = []
    completed = []
    gpus_in_use = set()
    wall_start = time.time()
    monitor = PowerMonitor()
    anchor_started = anchor_app is None
    pending_interval = None

    print(
        "Online co-scheduling: {} apps on {} GPUs (metric={}, policy={})".format(
            len(pending), TOTAL_GPUS, score_metric, policy
        )
    )
    print("NUMA 0 GPUs: {}  |  NUMA 1 GPUs: {}".format(NUMA0_GPUS, NUMA1_GPUS))
    print("=" * 96)

    monitor.start()
    try:
        while pending or running:
            scheduled = True
            while scheduled and pending and len(running) < DEFAULT_MAX_CONCURRENT:
                scheduled = False
                action, launches, evaluation, cmab_decision = pick_best_action(
                    pending,
                    running,
                    gpus_in_use,
                    candidate_modes,
                    score_metric,
                    idle_weight,
                    blocking_weight,
                    anchor_app,
                    anchor_started,
                    policy,
                    cmab_policy,
                )
                if not launches:
                    break
                for spec in launches:
                    cmd, env, cwd = build_command(spec.mode.app, spec.gpu_ids, spec.numa_node)
                    devnull = _devnull()
                    elapsed = time.time() - wall_start
                    message = (
                        "  t={:8.2f}s | START {:<15} | {} GPUs {} | NUMA {} | score={:.4f} | "
                        "regret={:.4f} idle={:.2f} block={:.2f}"
                    ).format(
                        elapsed,
                        spec.mode.app,
                        spec.mode.gpu_count,
                        spec.gpu_ids,
                        spec.numa_node,
                        evaluation.score,
                        evaluation.metric_regret,
                        evaluation.idle_frac,
                        evaluation.blocking_penalty,
                    )
                    if cmab_decision is not None:
                        message += " | cmab_ucb={:.4f} mu={:.4f} bonus={:.4f}".format(
                            cmab_decision.ucb_value,
                            cmab_decision.predicted_reward,
                            cmab_decision.exploration_bonus,
                        )
                    print(message)
                    proc = subprocess.Popen(
                        cmd,
                        env=env,
                        cwd=str(cwd) if cwd else None,
                        stdout=devnull,
                        stderr=subprocess.STDOUT,
                    )
                    running.append(
                        RunningJob(
                            app=spec.mode.app,
                            mode=spec.mode,
                            gpu_ids=spec.gpu_ids,
                            numa_node=spec.numa_node,
                            process=proc,
                            start_time=time.time(),
                            devnull=devnull,
                        )
                    )
                    gpus_in_use.update(spec.gpu_ids)
                    pending.remove(spec.mode.app)
                    if anchor_app and spec.mode.app == anchor_app:
                        anchor_started = True
                if policy == "cmab":
                    pending_interval = _make_cmab_interval(
                        action,
                        evaluation,
                        running,
                        time.time(),
                        len(monitor._samples),
                    )
                    break
                scheduled = True

            if not running:
                break

            time.sleep(poll_interval)
            finished_jobs = [job for job in list(running) if job.process.poll() is not None]
            if not finished_jobs:
                continue

            if policy == "cmab" and pending_interval is not None and cmab_policy is not None:
                now = time.time()
                samples = list(monitor._samples)
                total_energy_j = _energy_between_samples(samples, pending_interval.sample_start_idx, len(samples))
                duration = now - pending_interval.start_time_s
                reward, realized_metric = _compute_cmab_reward(
                    pending_interval,
                    duration,
                    idle_power,
                    idle_weight,
                    mean_busy_power_w,
                    total_energy_j=total_energy_j,
                )
                cmab_policy.update(pending_interval.features, reward)
                print(_cmab_update_message(pending_interval, reward, realized_metric, cmab_policy))
                pending_interval = None

            for job in finished_jobs:
                rc = job.process.poll()
                elapsed = time.time() - wall_start
                runtime = time.time() - job.start_time
                gpus_in_use -= set(job.gpu_ids)
                running.remove(job)
                job.devnull.close()
                print(
                    "  t={:8.2f}s | END   {:<15} | freed {} GPUs {} | NUMA {} | runtime={:.2f}s".format(
                        elapsed,
                        job.app,
                        len(job.gpu_ids),
                        job.gpu_ids,
                        job.numa_node,
                        runtime,
                    )
                )
                completed.append({
                    "app": job.app,
                    "gpu_count": job.mode.gpu_count,
                    "gpu_ids": job.gpu_ids,
                    "numa_node": job.numa_node,
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
                job.devnull.close()
            except Exception:
                pass
        monitor.stop()

    total_time = time.time() - wall_start

    print("\n" + "=" * 96)
    print("Online co-schedule summary:")
    print("{:<15} {:>6} {:>12} {:>5} {:>12}".format("App", "#GPUs", "GPU IDs", "NUMA", "Runtime (s)"))
    print("-" * 60)
    for item in completed:
        print(
            "{:<15} {:>6} {:>12} {:>5} {:>12.2f}".format(
                item["app"],
                item["gpu_count"],
                str(item["gpu_ids"]),
                item["numa_node"],
                item["runtime"],
            )
        )
    print("-" * 60)
    print("\nTotal makespan: {:.2f}s".format(total_time))
    monitor.print_summary()


def main():
    parser = argparse.ArgumentParser(description="Run the hand-crafted event-driven online co-scheduler.")
    parser.add_argument(
        "--metrics-file",
        type=Path,
        default=Path("/home/ac.zzheng/power/GPGPU/data/H100/perf_metrics.txt"),
        help="Path to perf_metrics.txt",
    )
    parser.add_argument(
        "--jobs",
        nargs="+",
        default=DEFAULT_JOB_QUEUE,
        help="Job queue. Default: {}".format(DEFAULT_JOB_QUEUE),
    )
    parser.add_argument(
        "--policy",
        choices=["heuristic", "cmab"],
        default="heuristic",
        help="Action selection policy. 'heuristic' keeps the original fixed score, 'cmab' uses warm-started LinUCB.",
    )
    parser.add_argument(
        "--slowdown-tol",
        type=float,
        default=DEFAULT_SLOWDOWN_TOL,
        help="Only consider modes with normalized runtime <= 1 + slowdown_tol. Default: {}".format(DEFAULT_SLOWDOWN_TOL),
    )
    parser.add_argument(
        "--score-metric",
        choices=["energy", "edp"],
        default=DEFAULT_SCORE_METRIC,
        help="Primary normalized regret term used in the action score.",
    )
    parser.add_argument(
        "--idle-power",
        type=float,
        default=DEFAULT_IDLE_POWER,
        help="Idle power per GPU in watts. Default: {}".format(DEFAULT_IDLE_POWER),
    )
    parser.add_argument(
        "--idle-weight",
        type=float,
        default=None,
        help="Override idle penalty weight. Default derives from idle_power / mean_busy_power.",
    )
    parser.add_argument(
        "--ucb-alpha",
        type=float,
        default=0.5,
        help="Exploration strength for --policy cmab. Default: 0.5",
    )
    parser.add_argument(
        "--blocking-weight",
        type=float,
        default=0,
        help="Override residual blocking penalty weight. Default equals idle_weight.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate decisions using profiled runtimes instead of launching jobs.",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=1.0,
        help="Polling interval in seconds for real execution. Default: 1.0",
    )
    parser.add_argument(
        "--anchor-app",
        type=str,
        default=DEFAULT_ANCHOR_APP,
        help="App that must be included in the first chosen action if it is waiting. Default: {}".format(DEFAULT_ANCHOR_APP),
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Directory for timestamped run logs. Default: {}".format(DEFAULT_RESULTS_DIR),
    )
    args = parser.parse_args()

    parsed = parse_metrics(args.metrics_file, args.jobs)
    candidate_modes = select_online_modes(parsed, args.slowdown_tol)

    busy_power = mean_busy_power(parsed)
    idle_weight = args.idle_weight if args.idle_weight is not None else args.idle_power / busy_power
    blocking_weight = args.blocking_weight if args.blocking_weight is not None else idle_weight
    cmab_policy = LinUCBPolicy(idle_weight=idle_weight, alpha=args.ucb_alpha) if args.policy == "cmab" else None

    results_dir = args.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    mode = "dryrun" if args.dry_run else "run"
    if args.policy == "heuristic":
        log_path = results_dir / "EcoPack_{}.txt".format(mode)
    else:
        log_path = results_dir / "EcoPack_{}_{}.txt".format(args.policy, mode)

    original_stdout = sys.stdout
    original_stderr = sys.stderr
    with log_path.open("w") as log_file:
        sys.stdout = TeeStream(original_stdout, log_file)
        sys.stderr = TeeStream(original_stderr, log_file)
        try:
            print("Results log: {}".format(log_path))
            print("Online policy parameters:")
            print("  policy            = {}".format(args.policy))
            print("  score_metric      = {}".format(args.score_metric))
            print("  slowdown_tol      = {:.2f}".format(args.slowdown_tol))
            print("  idle_power        = {:.2f} W".format(args.idle_power))
            print("  mean_busy_power   = {:.2f} W".format(busy_power))
            print("  idle_weight       = {:.4f}".format(idle_weight))
            print("  blocking_weight   = {:.4f}".format(blocking_weight))
            if cmab_policy is not None:
                theta0, theta1 = cmab_policy.theta()
                print("  ucb_alpha         = {:.4f}".format(args.ucb_alpha))
                print("  cmab_theta_init   = ({:.4f}, {:.4f})".format(theta0, theta1))
            print("  anchor_app        = {}".format(args.anchor_app if args.anchor_app else "None"))
            print("  candidate modes:")
            for app in args.jobs:
                desc = [
                    "{}GPU(rt={:.2f}, pred={}={:.4f}, nrt={:.2f}, ne={:.2f}, nedp={:.2f})".format(
                        mode.gpu_count,
                        mode.runtime_s,
                        mode.predictor_name,
                        mode.predictor_value,
                        mode.norm_runtime,
                        mode.norm_energy,
                        mode.norm_edp,
                    )
                    for mode in candidate_modes[app]
                ]
                print("    {:<15} -> {}".format(app, ", ".join(desc)))
            print()

            if args.dry_run:
                run_dry(
                    args.jobs,
                    candidate_modes,
                    args.score_metric,
                    idle_weight,
                    blocking_weight,
                    args.anchor_app,
                    args.policy,
                    cmab_policy,
                    args.idle_power,
                    busy_power,
                )
            else:
                run_online(
                    args.jobs,
                    candidate_modes,
                    args.score_metric,
                    idle_weight,
                    blocking_weight,
                    args.poll_interval,
                    args.anchor_app,
                    args.policy,
                    cmab_policy,
                    args.idle_power,
                    busy_power,
                )
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr


if __name__ == "__main__":
    main()
