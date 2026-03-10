#!/bin/bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "Usage: $0 <cap_gpu0> <cap_gpu1> <cap_gpu2> <cap_gpu3>" >&2
  exit 1
fi

CAP0="$1"
CAP1="$2"
CAP2="$3"
CAP3="$4"

geopmwrite NVML::GPU_POWER_LIMIT_CONTROL gpu 0 "$CAP0"
geopmwrite NVML::GPU_POWER_LIMIT_CONTROL gpu 1 "$CAP1"
geopmwrite NVML::GPU_POWER_LIMIT_CONTROL gpu 2 "$CAP2"
geopmwrite NVML::GPU_POWER_LIMIT_CONTROL gpu 3 "$CAP3"
