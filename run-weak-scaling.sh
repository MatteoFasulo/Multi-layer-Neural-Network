#!/bin/bash
# run-omp-program.sh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task 8
#SBATCH --time=1-01:00:00
#SBATCH --output slurm-%j.out
#SBATCH --partition=l40
echo "== Running OpenMP version =="
gcc -std=c99 -Wall -Wpedantic -fopenmp multi-layer-nn.c -o multi-layer-nn -lm
#nvcc cuda-multi-layer-nn.cu -o cuda-multi-layer-nn

bash weak-scaling.sh > csv/weak_scaling.csv
echo "== End of Job =="
