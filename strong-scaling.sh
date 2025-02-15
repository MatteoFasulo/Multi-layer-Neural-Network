#!/bin/bash

# This script executes the parallel `multi-layer-nn` program with an
# increasing number of processors $p$, from 1 up to the number of logical
# cores. For each value of $p$, the program runs with a fixed input size $N$ so
# that the total amount of work is kept constant.

PROG=./multi-layer-nn    # name of the executable
N=1048576          # problem size; 2^20
K=1000             # number of layers
CORES=`cat /proc/cpuinfo | grep processor | wc -l` # number of (logical) cores
NREPS=5                 # number of replications.

if [ ! -f "$PROG" ]; then
    echo
    echo "$PROG not found"
    echo
    exit 1
fi

echo "p,t1,t2,t3,t4,t5"

for p in `seq $CORES`; do
    TIMES=()
    for rep in `seq $NREPS`; do
        EXEC_TIME="$( OMP_NUM_THREADS=$p "$PROG" $N $K | grep "Execution time" | sed 's/Execution time //' | grep -o -E '[0-9]+[.][0-9]+')"
        TIMES+=("$EXEC_TIME")
    done
    echo "$p,${TIMES[0]},${TIMES[1]},${TIMES[2]},${TIMES[3]},${TIMES[4]}"
done