#!/bin/bash
set -euo pipefail

# Usage: bash run.sh <V100|A100|H100>
export SYSTEM="${1:?Usage: run.sh <V100|A100|H100>}"

PYTHON="$HOME/venv_sched/bin/python"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

$PYTHON run_cosched_sequential.py --policy sequential --sequential-gpu-strategy best


$PYTHON run_cosched_sequential.py --policy sequential --sequential-gpu-strategy max


$PYTHON solve_energy_optimal_cpsat.py --time-limit 300


$PYTHON run_solver_schedule.py


$PYTHON run_cosched_marble.py


$PYTHON ecoPack.py --policy cmab


$PYTHON ecoPack.py --policy heuristic


