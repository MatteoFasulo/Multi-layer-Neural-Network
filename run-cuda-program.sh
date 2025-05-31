#!/bin/bash
# run-cuda-program.sh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task 8
#SBATCH --time=1-01:00:00
#SBATCH --output slurm-%j.out
#SBATCH --partition=l40
echo "== Compiling CUDA =="
#gcc -std=c99 -Wall -Wpedantic -fopenmp multi-layer-nn.c -o multi-layer-nn -lm
nvcc cuda-multi-layer-nn.cu -o cuda-multi-layer-nn
#echo "== Test with CUDA =="
#./cuda-multi-layer-nn 1048576 100000 3
echo "== Running CUDA version =="
bash cuda-perf.sh > csv/cuda_perf1000.csv
echo "== End of Job =="
