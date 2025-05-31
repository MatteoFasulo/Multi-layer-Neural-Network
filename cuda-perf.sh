#!/bin/bash

# This script executes the CUDA `cuda-multi-layer-nn` program with an
# increasing input size and measures the throughput (processed data items/seconds).

CPU_PROG=./multi-layer-nn    # name of the executable
GPU_PROG=./cuda-multi-layer-nn    # name of the executable
K=1000                         # number of layers
NREPS=5                      # number of replications.
CORES=`cat /proc/cpuinfo | grep processor | wc -l` # number of (logical) cores
WORKLOAD=(131072 262144 524288 1048576 2097152)  # input sizes (2^17, 2^18, 2^19, 2^20, 2^21)

if [ ! -f "$CPU_PROG" ]; then
    echo
    echo "$CPU_PROG not found. Please compile it (e.g., gcc -std=c99 -Wall -Wpedantic -fopenmp multi-layer-nn.c -o multi-layer-nn -lm)."
    echo
    exit 1
fi

if [ ! -f "$GPU_PROG" ]; then
    echo
    echo "$GPU_PROG not found. Please compile it (e.g., nvcc cuda-multi-layer-nn.cu -o cuda-multi-layer-nn)."
    echo
    exit 1
fi

# CSV Header
echo "N,K,cpu_t1,cpu_t2,cpu_t3,cpu_t4,cpu_t5,gpu_t_no_shared1,gpu_t_no_shared2,gpu_t_no_shared3,gpu_t_no_shared4,gpu_t_no_shared5,gpu_t_shared1,gpu_t_shared2,gpu_t_shared3,gpu_t_shared4,gpu_t_shared5,cpu_th1,cpu_th2,cpu_th3,cpu_th4,cpu_th5,gpu_th_no_shared1,gpu_th_no_shared2,gpu_th_no_shared3,gpu_th_no_shared4,gpu_th_no_shared5,gpu_th_shared1,gpu_th_shared2,gpu_th_shared3,gpu_th_shared4,gpu_th_shared5"

for N in ${WORKLOAD[@]}; do
    CPU_TIMES=()
    GPU_TIMES_NO_SHARED=() # Renamed for clarity
    GPU_TIMES_SHARED=()

    CPU_THROUGHPUTS=()
    GPU_THROUGHPUTS_NO_SHARED=() # Renamed for clarity
    GPU_THROUGHPUTS_SHARED=()

    echo "Running N=$N, K=$K" >&2 # Print progress to stderr

    for rep in $(seq $NREPS); do
        # CPU
        # Assuming the OpenMP program prints "Execution time X.XXXXs" and "Throughput: YYY.YYY items/second"
        # Or "Time taken: X.XXXXs"
        CPU_OUTPUT="$( OMP_NUM_THREADS=$CORES "$CPU_PROG" $N $K )"
        # Try to match "Execution time X.Xs" or "Time taken: X.Xs"
        CPU_EXEC_TIME=$(echo "$CPU_OUTPUT" | grep -E "(Execution time|Time taken:)" | sed -E 's/Execution time |Time taken: //; s/s//' | grep -o -E '[0-9]+\.[0-9]+' | head -n 1)
        CPU_THROUGHPUT=$(echo "$CPU_OUTPUT" | grep "Throughput:" | sed -E 's/Throughput: | items\/second//g' | grep -o -E '[0-9]+\.[0-9]+' | head -n 1)

        CPU_TIMES+=("$CPU_EXEC_TIME")
        CPU_THROUGHPUTS+=("$CPU_THROUGHPUT")

        # GPU
        GPU_OUTPUT="$("$GPU_PROG" $N $K)"

        # GPU No Shared Memory
        # Output: Time (no shared): X.XXXX s, Throughput: YYYYYY.YY items/sec
        GPU_LINE_NO_SHARED=$(echo "$GPU_OUTPUT" | grep "Time (no shared):")
        GPU_EXEC_TIME_NO_SHARED=$(echo "$GPU_LINE_NO_SHARED" | sed -E 's/Time \(no shared\): ([0-9]+\.[0-9]+) s.*/\1/')
        GPU_THROUGHPUT_NO_SHARED=$(echo "$GPU_LINE_NO_SHARED" | sed -E 's/.*Throughput: ([0-9]+\.[0-9]+) items.*/\1/')
        
        # GPU Shared Memory
        # Output: Time (shared): X.XXXX s, Throughput: YYYYYY.YY items/sec, Speedup: Z.ZZx
        GPU_LINE_SHARED=$(echo "$GPU_OUTPUT" | grep "Time (shared):")
        GPU_EXEC_TIME_SHARED=$(echo "$GPU_LINE_SHARED" | sed -E 's/Time \(shared\): ([0-9]+\.[0-9]+) s.*/\1/')
        GPU_THROUGHPUT_SHARED=$(echo "$GPU_LINE_SHARED" | sed -E 's/.*Throughput: ([0-9]+\.[0-9]+) items.*/\1/')


        GPU_TIMES_NO_SHARED+=("$GPU_EXEC_TIME_NO_SHARED")
        GPU_THROUGHPUTS_NO_SHARED+=("$GPU_THROUGHPUT_NO_SHARED")
        GPU_TIMES_SHARED+=("$GPU_EXEC_TIME_SHARED")
        GPU_THROUGHPUTS_SHARED+=("$GPU_THROUGHPUT_SHARED")

        # Check if values were extracted (simple check for empty string)
        if [ -z "$CPU_EXEC_TIME" ] || [ -z "$GPU_EXEC_TIME_NO_SHARED" ] || [ -z "$GPU_EXEC_TIME_SHARED" ]; then
            echo "Error: Failed to parse output for N=$N, K=$K, rep=$rep. Check program output format." >&2
            echo "CPU Output:" >&2
            echo "$CPU_OUTPUT" >&2
            echo "GPU Output:" >&2
            echo "$GPU_OUTPUT" >&2
            # exit 1 # Optionally exit on parsing error
        fi
    done

    # Output CSV Row
    echo "$N,$K,${CPU_TIMES[0]},${CPU_TIMES[1]},${CPU_TIMES[2]},${CPU_TIMES[3]},${CPU_TIMES[4]},${GPU_TIMES_NO_SHARED[0]},${GPU_TIMES_NO_SHARED[1]},${GPU_TIMES_NO_SHARED[2]},${GPU_TIMES_NO_SHARED[3]},${GPU_TIMES_NO_SHARED[4]},${GPU_TIMES_SHARED[0]},${GPU_TIMES_SHARED[1]},${GPU_TIMES_SHARED[2]},${GPU_TIMES_SHARED[3]},${GPU_TIMES_SHARED[4]},${CPU_THROUGHPUTS[0]},${CPU_THROUGHPUTS[1]},${CPU_THROUGHPUTS[2]},${CPU_THROUGHPUTS[3]},${CPU_THROUGHPUTS[4]},${GPU_THROUGHPUTS_NO_SHARED[0]},${GPU_THROUGHPUTS_NO_SHARED[1]},${GPU_THROUGHPUTS_NO_SHARED[2]},${GPU_THROUGHPUTS_NO_SHARED[3]},${GPU_THROUGHPUTS_NO_SHARED[4]},${GPU_THROUGHPUTS_SHARED[0]},${GPU_THROUGHPUTS_SHARED[1]},${GPU_THROUGHPUTS_SHARED[2]},${GPU_THROUGHPUTS_SHARED[3]},${GPU_THROUGHPUTS_SHARED[4]}"
done