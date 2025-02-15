#!/bin/bash

# This script executes the parallel `multi-layer-nn` program with an
# increasing number of processors $p$, from 1 up to the number of logical
# cores. For each value of $p$, the program sets the input size so
# that the amount of per-processor work is kept constant while the total
# amount of work increases.

PROG=./multi-layer-nn  # name of the executable
N0=1048576              # base problem size (number of input neurons) for p=1 (2^20)
K=1000                  # number of layers
CORES=`cat /proc/cpuinfo | grep processor | wc -l`  # number of (logical) cores
NREPS=5                # nmber of replications

if [ ! -f "$PROG" ]; then
    echo
    echo "$PROG not found"
    echo
    exit 1
fi

echo -e "p,t1,t2,t3,t4,t5"

for p in `seq $CORES`; do
    TIMES=()
    # Compute problem size (N) for weak scaling
    # According to weak scaling: N = N0 * p
    N=$(awk "BEGIN {print $N0 * $p}")
    for rep in `seq $NREPS`; do
        EXEC_TIME="$( OMP_NUM_THREADS=$p "$PROG" $N $K | grep "Execution time" | sed 's/Execution time //' | grep -o -E '[0-9]+[.][0-9]+')"
        TIMES+=("$EXEC_TIME")
    done
    echo "$p,${TIMES[0]},${TIMES[1]},${TIMES[2]},${TIMES[3]},${TIMES[4]}"
done
