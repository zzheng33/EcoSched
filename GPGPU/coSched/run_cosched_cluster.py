#!/usr/bin/env python3
"""Cluster-level co-scheduler for two remote 4-GPU H100 nodes.

This controller runs on a login node and launches jobs remotely via ssh on:
- hopper00
- hopper01

Each remote node keeps the same local scheduling model as run_cosched.py:
- 4 GPUs total
- 2 NUMA domains
- at most 2 concurrent jobs per node
- NUMA-aware GPU placement

The controller performs cluster-level dispatch while reusing the existing
per-node GPU-count assignments and command builders.
"""

import argparse
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

from run_cosched import (
    DEFAULT_JOB_QUEUE,
    NUMA0_GPUS,
    NUMA1_GPUS,
    PREDICTED_GPU_COUNTS,
    RESULTS_DIR,
    TOTAL_GPUS,
    allocate_gpus_numa,
    build_command,
    pick_numa_for_tenant,
)

DEFAULT_CLUSTER_HOSTS = ("hopper00", "hopper01")
DEFAULT_MAX_CONCURRENT = 2
DEFAULT_ESTIMATED_RUNTIME = {
    "pot3d": 133.09,
    "minisweep": 43.61,
    "lbm": 18.82,
    "cloverleaf": 16.83,
    "tealeaf": 16.21,
    "miniweather": 57.01,
    "hpgmg": 10.78,
    "bert": 21.03,
    "gpt2": 14.80,
    "resnet50": 20.81,
}


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


@dataclass
class RunningRemoteJob:
    host: str
    app: str
    gpu_ids: List[int]
    numa_node: int
    process: subprocess.Popen
    start_time: float
    devnull: object


@dataclass
class SimRemoteJob:
    host: str
    app: str
    gpu_ids: List[int]
    numa_node: int
    end_time: float


@dataclass
class ClusterNode:
    host: str
    running: List[object] = field(default_factory=list)
    gpus_in_use: Set[int] = field(default_factory=set)

    @property
    def free_gpu_count(self) -> int:
        return TOTAL_GPUS - len(self.gpus_in_use)


def _devnull():
    return open(os.devnull, "w")


def _pick_next_app(
    candidates: Sequence[str],
    gpu_counts: Dict[str, int],
    free_gpu_count: int,
    policy: str,
) -> Optional[str]:
    if policy == "fcfs":
        for app in candidates:
            if gpu_counts[app] <= free_gpu_count:
                return app
        return None

    best_app = None
    best_diff = free_gpu_count + 1
    for app in candidates:
        needed = gpu_counts[app]
        if needed <= free_gpu_count:
            diff = free_gpu_count - needed
            if diff < best_diff:
                best_diff = diff
                best_app = app
    return best_app


def _node_sort_key(node: ClusterNode) -> Tuple[bool, int, str]:
    active = bool(node.running)
    return (not active, node.free_gpu_count, node.host)


def _ssh_args(ssh_options: Sequence[str]) -> List[str]:
    args = ["ssh"]
    for option in ssh_options:
        args.extend(["-o", option])
    return args


def _shell_join(parts: Sequence[str]) -> str:
    return " ".join(shlex.quote(part) for part in parts)


def build_remote_payload(app: str, gpu_ids: List[int], numa_node: int) -> str:
    cmd, env, cwd = build_command(app, gpu_ids, numa_node)
    pieces: List[str] = ["set -e"]

    if cwd:
        pieces.append("cd {}".format(shlex.quote(str(cwd))))

    cuda_visible = env.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible and cuda_visible != os.environ.get("CUDA_VISIBLE_DEVICES"):
        pieces.append("export CUDA_VISIBLE_DEVICES={}".format(shlex.quote(cuda_visible)))

    if len(cmd) >= 3 and cmd[0] == "bash" and cmd[1] == "-lc":
        pieces.append(cmd[2])
    else:
        pieces.append(_shell_join(cmd))

    return "; ".join(pieces)


def build_remote_command(
    host: str,
    app: str,
    gpu_ids: List[int],
    numa_node: int,
    ssh_options: Sequence[str],
) -> List[str]:
    payload = build_remote_payload(app, gpu_ids, numa_node)
    return _ssh_args(ssh_options) + [host, "bash", "-lc", payload]


def _cluster_has_running(nodes: Dict[str, ClusterNode]) -> bool:
    return any(node.running for node in nodes.values())


def _try_schedule_one(
    node: ClusterNode,
    pending: List[str],
    gpu_counts: Dict[str, int],
    policy: str,
    max_concurrent: int,
) -> Optional[Tuple[str, int, List[int], int]]:
    if len(node.running) >= max_concurrent:
        return None

    numa = pick_numa_for_tenant(node.running)
    if numa is None:
        return None

    app = _pick_next_app(pending, gpu_counts, node.free_gpu_count, policy)
    if app is None:
        return None

    needed = gpu_counts[app]
    gpu_ids = allocate_gpus_numa(needed, numa, node.gpus_in_use)
    if gpu_ids is None:
        return None
    return app, needed, gpu_ids, numa


def _results_log_path(results_dir: Path, policy: str, dry_run: bool) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    safe_policy = policy.replace("-", "_")
    prefix = "run_cosched_cluster_dryrun" if dry_run else "run_cosched_cluster"
    return results_dir / "{}_{}.txt".format(prefix, safe_policy)


def run_cluster_dry(
    nodes: Dict[str, ClusterNode],
    job_queue: Sequence[str],
    gpu_counts: Dict[str, int],
    policy: str,
    max_concurrent: int,
) -> None:
    pending = list(job_queue)
    completed = []
    sim_time = 0.0

    print(
        "Cluster dry-run: {} apps across {} nodes (policy={}, max {} per node)".format(
            len(pending), len(nodes), policy, max_concurrent
        )
    )
    print("Hosts: {}".format(", ".join(nodes)))
    print("Per-node topology: NUMA0 {} | NUMA1 {}".format(NUMA0_GPUS, NUMA1_GPUS))
    print("=" * 96)

    while pending or _cluster_has_running(nodes):
        scheduled_this_round = True
        while scheduled_this_round and pending:
            scheduled_this_round = False
            for node in sorted(nodes.values(), key=_node_sort_key):
                launch = _try_schedule_one(node, pending, gpu_counts, policy, max_concurrent)
                if launch is None:
                    continue
                app, needed, gpu_ids, numa = launch
                runtime = DEFAULT_ESTIMATED_RUNTIME.get(app, 30.0)
                end_time = sim_time + runtime
                node.running.append(
                    SimRemoteJob(
                        host=node.host,
                        app=app,
                        gpu_ids=gpu_ids,
                        numa_node=numa,
                        end_time=end_time,
                    )
                )
                node.gpus_in_use.update(gpu_ids)
                pending.remove(app)
                scheduled_this_round = True
                print(
                    "  t={:8.2f}s | START {:<15} | host {:<8} | {} GPUs {} | NUMA {} | "
                    "ends ~t={:.2f}s".format(
                        sim_time,
                        app,
                        node.host,
                        needed,
                        gpu_ids,
                        numa,
                        end_time,
                    )
                )

        if not _cluster_has_running(nodes):
            break

        next_job = min(
            (job for node in nodes.values() for job in node.running),
            key=lambda item: (item.end_time, item.host, item.app),
        )
        node = nodes[next_job.host]
        node.running.remove(next_job)
        node.gpus_in_use -= set(next_job.gpu_ids)
        sim_time = next_job.end_time
        completed.append(
            {
                "app": next_job.app,
                "host": next_job.host,
                "gpu_count": len(next_job.gpu_ids),
                "gpu_ids": next_job.gpu_ids,
                "numa_node": next_job.numa_node,
                "runtime": DEFAULT_ESTIMATED_RUNTIME.get(next_job.app, 30.0),
            }
        )
        print(
            "  t={:8.2f}s | END   {:<15} | host {:<8} | freed {} GPUs {} | NUMA {}".format(
                sim_time,
                next_job.app,
                next_job.host,
                len(next_job.gpu_ids),
                next_job.gpu_ids,
                next_job.numa_node,
            )
        )

    print("\n" + "=" * 96)
    print("Cluster dry-run summary:")
    print("{:<15} {:<8} {:>6} {:>12} {:>5} {:>12}".format("App", "Host", "#GPUs", "GPU IDs", "NUMA", "Runtime (s)"))
    print("-" * 74)
    for item in completed:
        print(
            "{:<15} {:<8} {:>6} {:>12} {:>5} {:>12.2f}".format(
                item["app"],
                item["host"],
                item["gpu_count"],
                str(item["gpu_ids"]),
                item["numa_node"],
                item["runtime"],
            )
        )
    print("-" * 74)
    print("\nEstimated makespan: ~{:.2f}s".format(sim_time))


def run_cluster(
    nodes: Dict[str, ClusterNode],
    job_queue: Sequence[str],
    gpu_counts: Dict[str, int],
    policy: str,
    max_concurrent: int,
    poll_interval: float,
    ssh_options: Sequence[str],
) -> None:
    pending = list(job_queue)
    completed = []
    wall_start = time.time()

    print(
        "Cluster co-scheduling: {} apps across {} nodes (policy={}, max {} per node)".format(
            len(pending), len(nodes), policy, max_concurrent
        )
    )
    print("Hosts: {}".format(", ".join(nodes)))
    print("Per-node topology: NUMA0 {} | NUMA1 {}".format(NUMA0_GPUS, NUMA1_GPUS))
    print("=" * 96)

    try:
        while pending or _cluster_has_running(nodes):
            scheduled_this_round = True
            while scheduled_this_round and pending:
                scheduled_this_round = False
                for node in sorted(nodes.values(), key=_node_sort_key):
                    launch = _try_schedule_one(node, pending, gpu_counts, policy, max_concurrent)
                    if launch is None:
                        continue
                    app, needed, gpu_ids, numa = launch
                    cmd = build_remote_command(node.host, app, gpu_ids, numa, ssh_options)
                    devnull = _devnull()
                    elapsed = time.time() - wall_start
                    print(
                        "  t={:8.2f}s | START {:<15} | host {:<8} | {} GPUs {} | NUMA {}".format(
                            elapsed,
                            app,
                            node.host,
                            needed,
                            gpu_ids,
                            numa,
                        )
                    )
                    proc = subprocess.Popen(
                        cmd,
                        stdout=devnull,
                        stderr=subprocess.STDOUT,
                    )
                    node.running.append(
                        RunningRemoteJob(
                            host=node.host,
                            app=app,
                            gpu_ids=gpu_ids,
                            numa_node=numa,
                            process=proc,
                            start_time=time.time(),
                            devnull=devnull,
                        )
                    )
                    node.gpus_in_use.update(gpu_ids)
                    pending.remove(app)
                    scheduled_this_round = True

            if not _cluster_has_running(nodes):
                break

            time.sleep(poll_interval)
            for node in nodes.values():
                for job in list(node.running):
                    rc = job.process.poll()
                    if rc is None:
                        continue
                    elapsed = time.time() - wall_start
                    runtime = time.time() - job.start_time
                    node.gpus_in_use -= set(job.gpu_ids)
                    node.running.remove(job)
                    job.devnull.close()
                    status = "OK" if rc == 0 else "FAILED(rc={})".format(rc)
                    print(
                        "  t={:8.2f}s | END   {:<15} | host {:<8} | freed {} GPUs {} | "
                        "NUMA {} | runtime={:.2f}s | {}".format(
                            elapsed,
                            job.app,
                            job.host,
                            len(job.gpu_ids),
                            job.gpu_ids,
                            job.numa_node,
                            runtime,
                            status,
                        )
                    )
                    completed.append(
                        {
                            "app": job.app,
                            "host": job.host,
                            "gpu_count": len(job.gpu_ids),
                            "gpu_ids": job.gpu_ids,
                            "numa_node": job.numa_node,
                            "runtime": runtime,
                            "return_code": rc,
                        }
                    )
    finally:
        for node in nodes.values():
            for job in list(node.running):
                try:
                    job.process.terminate()
                except Exception:
                    pass
                try:
                    job.devnull.close()
                except Exception:
                    pass
                node.running.remove(job)

    total_time = time.time() - wall_start

    print("\n" + "=" * 96)
    print("Cluster co-schedule summary:")
    print(
        "{:<15} {:<8} {:>6} {:>12} {:>5} {:>12}".format(
            "App",
            "Host",
            "#GPUs",
            "GPU IDs",
            "NUMA",
            "Runtime (s)",
        )
    )
    print("-" * 74)
    for item in completed:
        print(
            "{:<15} {:<8} {:>6} {:>12} {:>5} {:>12.2f}".format(
                item["app"],
                item["host"],
                item["gpu_count"],
                str(item["gpu_ids"]),
                item["numa_node"],
                item["runtime"],
            )
        )
    print("-" * 74)
    print("\nTotal makespan: {:.2f}s".format(total_time))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run cluster-level co-scheduling across hopper00 and hopper01."
    )
    parser.add_argument(
        "--jobs",
        nargs="+",
        default=DEFAULT_JOB_QUEUE,
        help="Job queue (app names in order). Default: {}".format(DEFAULT_JOB_QUEUE),
    )
    parser.add_argument(
        "--hosts",
        nargs="+",
        default=list(DEFAULT_CLUSTER_HOSTS),
        help="Remote hosts to use. Default: {}".format(list(DEFAULT_CLUSTER_HOSTS)),
    )
    parser.add_argument(
        "--policy",
        type=str,
        default="best-fit",
        choices=["fcfs", "best-fit"],
        help="Scheduling policy (default: best-fit)",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=DEFAULT_MAX_CONCURRENT,
        help="Max number of concurrent apps per node (default: {})".format(DEFAULT_MAX_CONCURRENT),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate schedule order without launching remote jobs",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=1.0,
        help="Polling interval in seconds for actual remote execution. Default: 1.0",
    )
    parser.add_argument(
        "--gpu-override",
        nargs="+",
        default=None,
        help="Override GPU counts as app:count pairs, e.g. bert:2 gpt2:2",
    )
    parser.add_argument(
        "--ssh-option",
        action="append",
        default=[],
        help="Extra ssh -o option, e.g. --ssh-option BatchMode=yes",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=RESULTS_DIR,
        help="Directory for fixed run logs. Default: {}".format(RESULTS_DIR),
    )
    args = parser.parse_args()

    gpu_counts = dict(PREDICTED_GPU_COUNTS)
    if args.gpu_override:
        for item in args.gpu_override:
            app, count = item.split(":")
            gpu_counts[app] = int(count)

    for app in args.jobs:
        if app not in gpu_counts:
            print("ERROR: No GPU count defined for '{}'".format(app), file=sys.stderr)
            sys.exit(1)
        if gpu_counts[app] > TOTAL_GPUS:
            print(
                "ERROR: {} needs {} GPUs but only {} available per node".format(
                    app, gpu_counts[app], TOTAL_GPUS
                ),
                file=sys.stderr,
            )
            sys.exit(1)

    log_path = _results_log_path(args.results_dir, args.policy, args.dry_run)
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    with log_path.open("w", encoding="utf-8") as log_file:
        tee = TeeStream(original_stdout, log_file)
        sys.stdout = tee
        sys.stderr = tee
        try:
            print("Results log: {}".format(log_path))
            print("Cluster hosts: {}".format(", ".join(args.hosts)))
            print("Job queue and GPU assignments:")
            for app in args.jobs:
                print("  {:<15} -> {} GPUs".format(app, gpu_counts[app]))
            print()

            nodes = {host: ClusterNode(host=host) for host in args.hosts}
            if args.dry_run:
                run_cluster_dry(nodes, args.jobs, gpu_counts, args.policy, args.max_concurrent)
            else:
                run_cluster(
                    nodes,
                    args.jobs,
                    gpu_counts,
                    args.policy,
                    args.max_concurrent,
                    args.poll_interval,
                    args.ssh_option,
                )
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr


if __name__ == "__main__":
    main()
