/****************************************************************************
 *
 * cuda-multi-layer-nn.cu - Multi Layer Neural Network with CUDA
 *
 * Last updated in 2025 by Matteo Fasulo <mat.fasulo@gmail.com>
 *
 * To the extent possible under law, the author(s) have dedicated all 
 * copyright and related and neighboring rights to this software to the 
 * public domain worldwide. This software is distributed without any warranty.
 *
 * You should have received a copy of the CC0 Public Domain Dedication
 * along with this software. If not, see 
 * <http://creativecommons.org/publicdomain/zero/1.0/>. 
 *
 * --------------------------------------------------------------------------
 *
 * Compile with:
 * nvcc cuda-multi-layer-nn.cu -o cuda-multi-layer-nn
 *
 * Run with:
 * ./cuda-multi-layer-nn [N] [K]
 *
 * (N = first layer n° neurons; default 1024)
 * (K = number of layers; default 2)
 *
 ****************************************************************************/

#include "hpc.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define BLKDIM 1024
#define R      3
#define RADIUS ((R-1)/2)
#define BIAS   0.2f

/* Define the NeuralNet struct */
/**
 * Neural network struct.
 * @param base The base pointer for the allocated memory.
 * @param x The input layer.
 * @param W The weights.
 * @param y The output layer.
 */
 typedef struct {
    float *base;   /* original cudaMalloc pointer */
    float *x;      /* input buffer */
    float *W;      /* weight buffer */
    float *y;      /* output buffer */
} NeuralNet;

/* Compute input size for layer `t`: layer 1 sees N, layer 2 sees N-(R-1), ... */
static inline int input_size(int N0, int layer) {
    return N0 - (layer-1)*(R-1);
}

/* Compute output size of layer `t`: #neurons = N0 - t*(R-1) */
static inline int output_size(int N0, int layer) {
    return N0 - layer*(R-1);
}

/* Allocate one big chunk for x, W, and y, then slice it up */
void allocateNeuralNet(NeuralNet *net, int N, int M) {
    size_t size_x = (N + 2 * RADIUS) * sizeof(float);
    size_t size_W = N * R * sizeof(float);
    size_t size_y = (M + 2 * RADIUS) * sizeof(float);
    size_t total_size = size_x + size_W + size_y;

    cudaError_t err = cudaMalloc(&net->base, total_size);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    
    net->x = net->base;
    net->W = (float *)((char *)net->base + size_x);
    net->y = (float *)((char *)net->base + size_x + size_W);
}

/* Free the single allocation */
void freeNeuralNet(NeuralNet *net) {
    if (net->base) {
        cudaFree(net->base);
        net->base = net->x = net->W = net->y = NULL;
    }
}

/* Define the Sigmoid function using the math.h lib*/
/**
 * Sigmoid activation function.
 *
 * @param v Input value.
 * @return The sigmoid of the input value.
 */
__device__ float sigmoid(const float v)
{
    return 1.0f / (1.0f + expf(-v));
}

/* Define the fill function */
/**
 * Fill an array with random values.
 *
 * @param arr The array to fill.
 * @param size The size of the array.
 */
void fill(float *arr, size_t size)
{
    for (size_t i = 0; i < size; ++i)
    {
        arr[i] = (float)rand() / (float)RAND_MAX;
    }
}

/* CPU-side throughput calculation: sum output sizes of layers 1..K-1 */
double compute_throughput(long long N0, int K, double secs) {
    long long total = 0;
    for (int t = 1; t <= K-1; ++t)
        total += output_size((int)N0, t);
    return total / secs;
}

/* Define the forward_propagation kernel without shared memory */
__global__ void forward_propagation(
    const float* x,
    const float* W,
    float* y,
    int out_size
) {
    const int index = threadIdx.x + blockIdx.x * blockDim.x + RADIUS;
    if (index >= out_size) return;

    float sum = BIAS;

    #pragma unroll
    for (int offset = -RADIUS; offset <= RADIUS; offset++) {
        sum += x[index + offset] * W[index + out_size * (offset + RADIUS)];
    }

    y[index] = sigmoid(sum);
}

/* Define the forward_propagation kernel with shared memory */
__global__ void forward_propagation_shared(
    const float* x,
    const float* W,
    float* y,
    int in_size,
    int out_size
) {
    // Shared memory for the input stencil window
    __shared__ float temp[BLKDIM + 2 * RADIUS];

    const int gindex = threadIdx.x + blockIdx.x * blockDim.x + RADIUS;
    const int lindex = threadIdx.x + RADIUS;

    /* Read input elements into shared memory */
    temp[lindex] = x[gindex];
    if (threadIdx.x < RADIUS) {
        temp[lindex - RADIUS] = x[gindex - RADIUS];
        temp[lindex + blockDim.x] = x[gindex + blockDim.x];
    }
    __syncthreads();

    float sum = BIAS;

    #pragma unroll
    for (int offset = -RADIUS ; offset <= RADIUS ; offset++) {
        sum += temp[lindex + offset] * W[gindex + out_size * (offset + RADIUS)];
    }

    y[gindex] = sigmoid(sum);
}

int main(int argc, char *argv[])
{
    float *h_x, *h_W, *h_y, *h_y_shared; // Host memory for x, W, and y
    int N = BLKDIM;  // Number of neurons in the first layer
    int K = 2;     // Number of layers
    int M;
    double tstart, tstop, tnoshared, tshared; // Timers
    double throughput, shared_throughput; // Throughput in items per second

    if (argc > 3)
    {
        fprintf(stderr, "Usage: %s [N (default %d)] [K (default %d)]\n",
                argv[0], N, K);
        return EXIT_FAILURE;
    }
    if (argc >= 2)
        N = atoi(argv[1]);
    if (argc == 3)
        K = atoi(argv[2]);

    // Validate the input arguments
    if (K < 2)
    {
        fprintf(stderr, "K must be greater than 1.\n");
        return EXIT_FAILURE;
    }
    if (N < 1)
    {
        fprintf(stderr, "N must be a positive integer.\n");
        return EXIT_FAILURE;
    }

    // Check that input is a power of 2
    if (N & (N - 1))
    {
        fprintf(stderr, "N must be a power of 2.\n");
        return EXIT_FAILURE;
    }

    // Check that R is odd
    if (R % 2 == 0)
    {
        fprintf(stderr, "R must be odd.\n");
        return EXIT_FAILURE;
    }

    // Compute the size of the first output layer
    M = N - (R - 1);

    NeuralNet d_nn;
    allocateNeuralNet(&d_nn, N, M);

    // Allocate host memory for x, W
    h_x = (float *)malloc((N+2*RADIUS)*sizeof(float)); fill(h_x, (N+2*RADIUS));
    h_W = (float *)malloc(N*R*sizeof(float)); fill(h_W, N * R);
    
    // Copy x and W to device memory
    cudaMemcpy(d_nn.x, h_x, (N+2*RADIUS)*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_nn.W, h_W, N*R*sizeof(float), cudaMemcpyHostToDevice);

    int final_layer_size = output_size(N, K - 1);

    printf("Number of neurons (N) = %d, Number of layers (K) = %d, Radius (R) = %d\n", N, K, R);
    printf("BLKDIM = %d\n", BLKDIM);

    /**
     ** Forward propagation without shared memory
     **/
    printf("No shared memory:\t");

    // Start the time
    tstart = hpc_gettime();
    
    // Forward propagation without shared memory
    for (int t = 1; t <= K - 1; t++)
    {
        int out_sz = output_size(N, t);

        // Launch the kernel
        forward_propagation<<<(out_sz + BLKDIM - 1) / BLKDIM, BLKDIM>>>(
            d_nn.x, d_nn.W, d_nn.y,
            out_sz
        );

        if (t < K - 1)
        {
            // Swap the input and output arays if we are not at the last layer
            float *temp = d_nn.x;
            d_nn.x = d_nn.y;
            d_nn.y = temp;
        }
    }
    cudaDeviceSynchronize();
    // Stop the time
    tstop = hpc_gettime();
    tnoshared = tstop - tstart;
    printf("%fs\n", tnoshared);

    // Calculate throughput
    throughput = compute_throughput(N, K, tnoshared);
    printf("Throughput:\t\t%f items/second\n", throughput);

    // Copy the output layer back to the host
    h_y = (float *)malloc((final_layer_size+(2*RADIUS)) * sizeof(float)); 
    cudaMemcpy(h_y, d_nn.y, (final_layer_size+(2*RADIUS)) * sizeof(float), cudaMemcpyDeviceToHost);

    // Clean up host and device allocations.
    freeNeuralNet(&d_nn);

    // Reallocate memory for the device
    NeuralNet d_nn_shared;
    allocateNeuralNet(&d_nn_shared, N, M);
    
    // Copy x and W to device memory
    cudaMemcpy(d_nn_shared.x, h_x, (N+2*RADIUS) * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_nn_shared.W, h_W, N * R * sizeof(float), cudaMemcpyHostToDevice);


    printf("Shared memory:\t\t");

    tstart = hpc_gettime();
    // Forward propagation with shared memory
    for (int t = 1; t <= K - 1; t++)
    {
        int in_sz = input_size(N, t);
        int out_sz = output_size(N, t);
        
        // Launch the kernel
        forward_propagation_shared<<<(out_sz + BLKDIM - 1) / BLKDIM, BLKDIM>>>(
            d_nn_shared.x, d_nn_shared.W, d_nn_shared.y,
            in_sz, out_sz
        );

        if (t < K - 1)
        {
            // Swap the input and output arays if we are not at the last layer
            float *temp = d_nn_shared.x;
            d_nn_shared.x = d_nn_shared.y;
            d_nn_shared.y = temp;
        }
    }
    cudaDeviceSynchronize();
    // Stop the time
    tstop = hpc_gettime();
    tshared = tstop - tstart;
    // Print the time and speedup w.r.t the non-shared memory version
    printf("%fs (%.2fx speedup)\n", tshared, tnoshared / tshared);

    // Calculate throughput
    shared_throughput = compute_throughput(N, K, tshared);
    printf("Throughput:\t\t%f items/second\n", shared_throughput);

    // Copy the output layer back to the host
    h_y_shared = (float *)malloc((final_layer_size+(2*RADIUS)) * sizeof(float)); 
    cudaMemcpy(h_y_shared, d_nn_shared.y, (final_layer_size+(2*RADIUS)) * sizeof(float), cudaMemcpyDeviceToHost);

    // Check if the results are the same
    int good = memcmp(h_y, h_y_shared, final_layer_size * sizeof(float)) == 0;
    // Compare only the actual output size, excluding padding
    if (good)
        printf("Results are the same.\n");
    else
        printf("Results are different!\n");

    // Clean up host and device allocations.
    freeNeuralNet(&d_nn_shared);
    free(h_x); h_x = NULL;
    free(h_W); h_W = NULL;
    free(h_y); h_y = NULL;
    free(h_y_shared); h_y_shared = NULL;

    return EXIT_SUCCESS;;
}
