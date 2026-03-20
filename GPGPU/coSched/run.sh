#!/bin/bash
set -euo pipefail

# System: V100, A100, or H100
# Make sure config.py has SYSTEM = "V100" before running.

PYTHON="$HOME/venv_sched/bin/python"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# $PYTHON solve_energy_optimal_cpsat.py --time-limit 1200


# $PYTHON run_solver_schedule.py


# $PYTHON run_cosched_sequential.py --policy sequential


# $PYTHON run_cosched_marble.py


$PYTHON ecoPack.py


