#!/bin/bash
set -euo pipefail

# Usage: bash run.sh <V100|A100|H100> <jlse|cc>
#
# SYSTEM  – GPU type (V100, A100, H100)
# SERVER  – Physical machine (jlse = V100/H100 server, cc = A100 server)
#
# Examples:
#   bash run.sh H100 jlse
#   bash run.sh A100 cc

export SYSTEM="${1:?Usage: run.sh <V100|A100|H100> <jlse|cc>}"
export SERVER="${2:?Usage: run.sh <V100|A100|H100> <jlse|cc>}"

# Python interpreter: jlse server uses a scheduling venv, cc uses system python3
if [ "$SERVER" = "jlse" ]; then
    PYTHON="$HOME/venv_sched/bin/python"
else
    PYTHON="python3"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# $PYTHON run_cosched_sequential.py --policy sequential --sequential-gpu-strategy best


# $PYTHON run_cosched_sequential.py --policy sequential --sequential-gpu-strategy max


# $PYTHON solve_energy_optimal_cpsat.py --time-limit 120


# $PYTHON run_solver_schedule.py


# $PYTHON run_cosched_marble.py


$PYTHON ecoPack.py --policy cmab
