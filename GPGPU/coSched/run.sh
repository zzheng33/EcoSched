#!/bin/bash
set -euo pipefail

# Usage: bash run.sh <V100|A100|H100> <vh|a100>
#
# SYSTEM  – GPU type (V100, A100, H100)
# SERVER  – Physical machine (vh = V100/H100 server, a100 = A100 server)
#
# Examples:
#   bash run.sh H100 vh
#   bash run.sh A100 a100

export SYSTEM="${1:?Usage: run.sh <V100|A100|H100> <vh|a100>}"
export SERVER="${2:?Usage: run.sh <V100|A100|H100> <vh|a100>}"

# Python interpreter: vh server uses a scheduling venv, a100 uses system python3
if [ "$SERVER" = "vh" ]; then
    PYTHON="$HOME/venv_sched/bin/python"
else
    PYTHON="python3"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

$PYTHON run_cosched_sequential.py --policy sequential --sequential-gpu-strategy best


$PYTHON run_cosched_sequential.py --policy sequential --sequential-gpu-strategy max


$PYTHON solve_energy_optimal_cpsat.py --time-limit 120


$PYTHON run_solver_schedule.py


$PYTHON run_cosched_marble.py


$PYTHON ecoPack.py --policy cmab
