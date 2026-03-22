#!/bin/bash
set -euo pipefail

# Usage: bash setup_env.sh <jlse|cc>
#
# Creates the Python virtual environments needed on each server.
#   jlse  – V100/H100 server: creates ~/venv_sched and ~/env/ml
#   cc    – A100 server:      creates ~/venv_sched only (system python3 for ML)

SERVER="${1:?Usage: setup_env.sh <jlse|cc>}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REQ_FILE="$SCRIPT_DIR/requirements.txt"

echo "=== Setting up environments for server: $SERVER ==="

# --- Scheduling venv (both servers) ---
SCHED_VENV="$HOME/venv_sched"
if [ -d "$SCHED_VENV" ]; then
    echo "Scheduling venv already exists at $SCHED_VENV, skipping creation."
else
    echo "Creating scheduling venv at $SCHED_VENV ..."
    python3 -m venv "$SCHED_VENV"
fi
echo "Installing scheduling dependencies ..."
"$SCHED_VENV/bin/pip" install --upgrade pip
"$SCHED_VENV/bin/pip" install ortools

# --- ML venv (jlse only) ---
if [ "$SERVER" = "jlse" ]; then
    ML_VENV="$HOME/env/ml"
    if [ -d "$ML_VENV" ]; then
        echo "ML venv already exists at $ML_VENV, skipping creation."
    else
        echo "Creating ML venv at $ML_VENV ..."
        mkdir -p "$(dirname "$ML_VENV")"
        python3 -m venv "$ML_VENV"
    fi
    echo "Installing ML dependencies ..."
    "$ML_VENV/bin/pip" install --upgrade pip
    "$ML_VENV/bin/pip" install torch torchvision numpy datasets transformers
else
    echo "CC server: ML packages use system python3, skipping ML venv."
    echo "Make sure torch, torchvision, numpy, datasets, transformers are installed system-wide."
fi

echo
echo "=== Done ==="
echo "Scheduling venv : $SCHED_VENV"
if [ "$SERVER" = "jlse" ]; then
    echo "ML venv         : $HOME/env/ml"
fi
