#!/bin/bash
set -euo pipefail

# Usage:
#   bash run_presets.sh <V100|A100|H100> [all|low|med|high]
#
# Examples:
#   bash run_presets.sh H100 all
#   bash run_presets.sh V100 low

SYSTEM_ARG="${1:?Usage: run_presets.sh <V100|A100|H100> [all|low|med|high]}"
PRESET_ARG="${2:-all}"

export SYSTEM="$SYSTEM_ARG"

PYTHON="$HOME/venv_sched/bin/python"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

case "$PRESET_ARG" in
    all)
        PRESET_KEYS=(low_opp med_opp high_opp)
        PRESET_DIRS=(low med high)
        ;;
    low)
        PRESET_KEYS=(low_opp)
        PRESET_DIRS=(low)
        ;;
    med)
        PRESET_KEYS=(med_opp)
        PRESET_DIRS=(med)
        ;;
    high)
        PRESET_KEYS=(high_opp)
        PRESET_DIRS=(high)
        ;;
    *)
        echo "Unknown preset '$PRESET_ARG'. Expected one of: all, low, med, high" >&2
        exit 1
        ;;
esac

get_jobs_for_preset() {
    local system_name="$1"
    local preset_key="$2"
    "$PYTHON" - <<PY
from config import WORKLOAD_PRESETS
system_name = "${system_name}"
preset_key = "${preset_key}"
try:
    jobs = WORKLOAD_PRESETS[system_name][preset_key]
except KeyError as exc:
    raise SystemExit(f"Missing workload preset for {system_name}/{preset_key}: {exc}")
print(" ".join(jobs))
PY
}

run_one_preset() {
    local system_name="$1"
    local preset_key="$2"
    local preset_dir_name="$3"

    local results_dir="$HOME/power/GPGPU/coSched/results/${system_name}/${preset_dir_name}"
    local solver_schedule_file="${results_dir}/solver_schedule.txt"
    local jobs

    jobs="$(get_jobs_for_preset "$system_name" "$preset_key")"
    mkdir -p "$results_dir"

    echo
    echo "================================================================================"
    echo "SYSTEM      : $system_name"
    echo "PRESET      : $preset_key"
    echo "RESULTS DIR : $results_dir"
    echo "JOBS        : $jobs"
    echo "================================================================================"

    "$PYTHON" run_cosched_sequential.py \
        --policy sequential \
        --sequential-gpu-strategy best \
        --results-dir "$results_dir" \
        --jobs $jobs

    "$PYTHON" run_cosched_sequential.py \
        --policy sequential \
        --sequential-gpu-strategy max \
        --results-dir "$results_dir" \
        --jobs $jobs

    "$PYTHON" solve_energy_optimal_cpsat.py \
        --time-limit 120 \
        --output-file "$solver_schedule_file" \
        --jobs $jobs

    "$PYTHON" run_solver_schedule.py \
        --schedule-file "$solver_schedule_file" \
        --results-dir "$results_dir"

    "$PYTHON" run_cosched_marble.py \
        --results-dir "$results_dir" \
        --jobs $jobs

    "$PYTHON" ecoPack.py \
        --policy cmab \
        --results-dir "$results_dir" \
        --jobs $jobs
}

for idx in "${!PRESET_KEYS[@]}"; do
    run_one_preset "$SYSTEM_ARG" "${PRESET_KEYS[$idx]}" "${PRESET_DIRS[$idx]}"
done
