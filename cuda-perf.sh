#!/bin/bash

# This script executes the CUDA `cuda-multi-layer-nn` program with an
# increasing input size and measures the throughput (processed data items/seconds).

CPU_PROG=./multi-layer-nn    # name of the executable
GPU_PROG=./cuda-multi-layer-nn    # name of the executable
K=4000                         # number of layers
NREPS=5                      # number of replications.
CORES=`cat /proc/cpuinfo | grep processor | wc -l` # number of (logical) cores
WORKLOAD=(131072 262144 524288 1048576 2097152)  # input sizes (2^17, 2^18, 2^19, 2^20, 2^21)

if [ ! -f "$CPU_PROG" ]; then
    echo
    echo "$CPU_PROG not found"
    echo
    exit 1
fi

if [ ! -f "$GPU_PROG" ]; then
    echo
    echo "$GPU_PROG not found"
    echo
    exit 1
fi

echo "N,K,cpu_t1,cpu_t2,cpu_t3,cpu_t4,cpu_t5,gpu_t1,gpu_t2,gpu_t3,gpu_t4,gpu_t5,gpu_t1_shared,gpu_t2_shared,gpu_t3_shared,gpu_t4_shared,gpu_t5_shared,cpu_th1,cpu_th2,cpu_th3,cpu_th4,cpu_th5,gpu_th1,gpu_th2,gpu_th3,gpu_th4,gpu_th5,gpu_th1_shared,gpu_th2_shared,gpu_th3_shared,gpu_th4_shared,gpu_th5_shared"
for N in ${WORKLOAD[@]}; do
    CPU_TIMES=()
    GPU_TIMES=()
    GPU_TIMES_SHARED=()

    CPU_THROUGHPUTS=()
    GPU_THROUGHPUTS=()
    GPU_THROUGHPUTS_SHARED=()
    for rep in `seq $NREPS`; do
        # CPU
        CPU_OUTPUT="$( OMP_NUM_THREADS=$CORES "$CPU_PROG" $N $K)"
        CPU_EXEC_TIME=$(echo "$CPU_OUTPUT" | grep "Execution time" | sed 's/Execution time //' | grep -o -E '[0-9]+\.[0-9]+')
        CPU_THROUGHPUT=$(echo "$CPU_OUTPUT" | grep "Throughput" | sed -n '1p' | grep -o -E '[0-9]+\.[0-9]+')

        CPU_TIMES+=("$CPU_EXEC_TIME")
        CPU_THROUGHPUTS+=("$CPU_THROUGHPUT")

        # GPU
        GPU_OUTPUT="$("$GPU_PROG" $N $K)"
        GPU_EXEC_TIME=$(echo "$GPU_OUTPUT" | grep "No shared memory" | grep -o -E '[0-9]+\.[0-9]+s' | grep -o -E '[0-9]+\.[0-9]+')
        GPU_THROUGHPUT=$(echo "$GPU_OUTPUT" | grep "Throughput" | sed -n '1p' | grep -o -E '[0-9]+\.[0-9]+')
        GPU_EXEC_TIME_SHARED=$(echo "$GPU_OUTPUT" | grep "Shared memory" | grep -o -E '[0-9]+\.[0-9]+s' | grep -o -E '[0-9]+\.[0-9]+')
        GPU_THROUGHPUT_SHARED=$(echo "$GPU_OUTPUT" | grep "Throughput" | sed -n '2p' | grep -o -E '[0-9]+\.[0-9]+')

        GPU_TIMES+=("$GPU_EXEC_TIME")
        GPU_THROUGHPUTS+=("$GPU_THROUGHPUT")
        GPU_TIMES_SHARED+=("$GPU_EXEC_TIME_SHARED")
        GPU_THROUGHPUTS_SHARED+=("$GPU_THROUGHPUT_SHARED")
    done
    echo "$N,$K,${CPU_TIMES[0]},${CPU_TIMES[1]},${CPU_TIMES[2]},${CPU_TIMES[3]},${CPU_TIMES[4]},${GPU_TIMES[0]},${GPU_TIMES[1]},${GPU_TIMES[2]},${GPU_TIMES[3]},${GPU_TIMES[4]},${GPU_TIMES_SHARED[0]},${GPU_TIMES_SHARED[1]},${GPU_TIMES_SHARED[2]},${GPU_TIMES_SHARED[3]},${GPU_TIMES_SHARED[4]},${CPU_THROUGHPUTS[0]},${CPU_THROUGHPUTS[1]},${CPU_THROUGHPUTS[2]},${CPU_THROUGHPUTS[3]},${CPU_THROUGHPUTS[4]},${GPU_THROUGHPUTS[0]},${GPU_THROUGHPUTS[1]},${GPU_THROUGHPUTS[2]},${GPU_THROUGHPUTS[3]},${GPU_THROUGHPUTS[4]},${GPU_THROUGHPUTS_SHARED[0]},${GPU_THROUGHPUTS_SHARED[1]},${GPU_THROUGHPUTS_SHARED[2]},${GPU_THROUGHPUTS_SHARED[3]},${GPU_THROUGHPUTS_SHARED[4]}"
done
