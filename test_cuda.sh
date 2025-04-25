#!/bin/bash
# run-cuda-program.sh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task 4
#SBATCH --time=1-01:00:00
#SBATCH --output slurm-%j.out
#SBATCH --partition=rtx2080
echo "== Compiling CUDA =="
gcc -std=c99 -Wall -Wpedantic -fopenmp multi-layer-nn.c -o multi-layer-nn -lm
nvcc cuda-multi-layer-nn.cu -o cuda-multi-layer-nn
echo "== Test with CUDA =="
./cuda-multi-layer-nn 65536 1000
./cuda-multi-layer-nn 65536 1000

./cuda-multi-layer-nn 131072 1000

./cuda-multi-layer-nn 262144 1000

./cuda-multi-layer-nn 524288 1000

./cuda-multi-layer-nn 1048576 1000
./cuda-multi-layer-nn 1048576 1000
echo "== End of Job =="