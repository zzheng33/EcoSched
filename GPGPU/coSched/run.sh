#!/bin/bash
set -euo pipefail

# System: V100, A100, or H100
# Make sure config.py has SYSTEM = "V100" before running.

PYTHON="$HOME/venv_sched/bin/python"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "===== 1. Offline solver (CP-SAT) ====="
$PYTHON solve_energy_optimal_cpsat.py --time-limit 600

echo ""
echo "===== 2. Solver schedule executor ====="
$PYTHON run_solver_schedule.py

echo ""
echo "===== 3. Sequential baseline ====="
$PYTHON run_cosched_sequential.py --policy sequential

echo ""
echo "===== 4. Marble baseline ====="
$PYTHON run_cosched_marble.py

echo ""
echo "===== 5. EcoPack ====="
$PYTHON ecoPack.py

echo ""
echo "===== All done ====="
