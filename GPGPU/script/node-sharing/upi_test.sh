#!/bin/bash

BANDWIDTHTEST=/soft/compilers/cuda/cuda-12.3.0/extras/demo_suite/bandwidthTest
OUTDIR=/home/ac.zzheng/power/GPGPU/script/co-sched/upi_results
mkdir -p "$OUTDIR"

echo "=== UPI Bandwidth Test ==="
echo "Results will be saved to $OUTDIR"

# Test 1: Local NUMA — GPU 0 from NUMA 0 (no UPI)
echo ""
echo ">>> Test 1: Local NUMA — GPU 0 from NUMA 0 CPUs/memory (no UPI)"
CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=0 --membind=0 \
  $BANDWIDTHTEST --mode=shmoo --csv 2>&1 | tee "$OUTDIR/test1_local_gpu0_numa0.csv"

# Test 2: Remote NUMA — GPU 2 from NUMA 0 (crosses UPI)
echo ""
echo ">>> Test 2: Remote NUMA — GPU 2 from NUMA 0 CPUs/memory (crosses UPI)"
CUDA_VISIBLE_DEVICES=2 numactl --cpunodebind=0 --membind=0 \
  $BANDWIDTHTEST --mode=shmoo --csv 2>&1 | tee "$OUTDIR/test2_remote_gpu2_numa0.csv"

# Test 3: Local NUMA — GPU 2 from NUMA 1 (no UPI, control)
echo ""
echo ">>> Test 3: Local NUMA — GPU 2 from NUMA 1 CPUs/memory (no UPI, control)"
CUDA_VISIBLE_DEVICES=2 numactl --cpunodebind=1 --membind=1 \
  $BANDWIDTHTEST --mode=shmoo --csv 2>&1 | tee "$OUTDIR/test3_local_gpu2_numa1.csv"

# Test 4: Contention — both tenants crossing UPI simultaneously
echo ""
echo ">>> Test 4: Contention — GPU 2 from NUMA 0 + GPU 1 from NUMA 1 (both cross UPI)"
CUDA_VISIBLE_DEVICES=2 numactl --cpunodebind=0 --membind=0 \
  $BANDWIDTHTEST --mode=shmoo --csv 2>"$OUTDIR/test4a_stderr.log" | tee "$OUTDIR/test4a_gpu2_from_numa0.csv" &
PID1=$!

CUDA_VISIBLE_DEVICES=1 numactl --cpunodebind=1 --membind=1 \
  $BANDWIDTHTEST --mode=shmoo --csv 2>"$OUTDIR/test4b_stderr.log" | tee "$OUTDIR/test4b_gpu1_from_numa1.csv" &
PID2=$!

wait $PID1 $PID2
echo ""
echo "=== All tests complete. Results in $OUTDIR ==="