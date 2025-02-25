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
#define R 3
#define RADIUS R / 2
#define BIAS 0.0

/* Define the NeuralNet struct */
/**
 * Neural network struct.
 *
 * @param x The input layer.
 * @param W The weights.
 * @param y The output layer.
 */
struct __align__(16) NeuralNet {
    float *x;
    float *W;
    float *y;
};

/* Define the allocateNeuralNet function */
/**
 * Allocate memory for the neural network.
 *
 * @param net The neural network to allocate.
 * @param N The number of neurons in the first layer.
 * @param M The size of the output layer.
 */
void allocateNeuralNet(NeuralNet &net, const int N, const int M) {
    size_t size_x = (N+2*RADIUS) * sizeof(float);
    size_t size_W = N * R * sizeof(float);
    size_t size_y = (M+2*RADIUS) * sizeof(float);
    size_t total_size = size_x + size_W + size_y;
    
    
    // cudaMalloc returns memory that is usually aligned to 256 bytes.
    float *base_ptr = nullptr;
    cudaMalloc((void **)&base_ptr, total_size);
    
    // Slice up the contiguous allocation.
    net.x = base_ptr;
    net.W = (float *)(((char *)base_ptr) + size_x);
    net.y = (float *)(((char *)base_ptr) + size_x + size_W);
}

/* Free the memory allocated for the neural network */
/**
 * @param net The neural network to free.
 * @return void
 */
void freeNeuralNet(NeuralNet &net) {
    if (net.x) cudaFree(net.x);
    if (net.W) cudaFree(net.W);
    if (net.y) cudaFree(net.y);
    net.x = net.W = net.y = nullptr;
}

/* Define the Sigmoid function using the math.h lib*/
/**
 * Sigmoid activation function.
 *
 * @param x Input value.
 * @return The sigmoid of the input value.
 */
__device__ float sigmoid(const float x)
{
    return x;
}

/* Define the fill function */
/**
 * Fill an array with random values.
 *
 * @param array The array to fill.
 * @param size The size of the array.
 */
void fill(float *array, const size_t size)
{
    for (size_t i = 0; i < size; i++)
    {
        array[i] = 1.0f;
    }
}

/* Define the compute_layer_size function */
/**
 * Compute the size of a layer in the network.
 *
 * @param N The number of neurons in the first layer.
 * @param t The layer index.
 * @return The size of the layer.
 */
int compute_layer_size(const int N, const int t)
{
    return N - t*(R - 1);
}

/* Define the forward_propagation kernel without shared memory */
/**
 * Forward propagation kernel. Each thread computes the output of a single neuron in the output layer.
 *
 * @param NeuralNet The neural network.
 * @param out_size The size of the output layer.
 */
__global__ void forward_propagation(
    const NeuralNet net,
    const int out_size
) {

    /* Compute the index of the current thread */
    const int index = threadIdx.x + blockIdx.x * blockDim.x + RADIUS;

    /* Compute the output for this thread */
    if (index < out_size + RADIUS) {
        float sum = BIAS;
        for (int offset = -RADIUS; offset <= RADIUS; offset++) {
            int row = offset + RADIUS; // 0,1,2 for R=3
            sum += net.x[index + offset] * net.W[row * out_size + index];
        }
        net.y[index] = sigmoid(sum);
    }
}

/* Define the forward_propagation kernel with shared memory */
/**
 * Forward propagation kernel optimized for 1D stencil computation.
 *
 * @param NeuralNet The neural network.
 * @param in_size The size of the input layer.
 * @param out_size The size of the output layer.
 */
__global__ void forward_propagation_shared(
    const NeuralNet net,
    const int in_size,
    const int out_size
) {
    // Shared memory for the input stencil window
    __shared__ float temp[BLKDIM + 2 * RADIUS];
    __shared__ float shared_W[R][BLKDIM];

    int lindex = threadIdx.x + RADIUS;
    int gindex = threadIdx.x + blockIdx.x * blockDim.x + RADIUS;
    float sum = BIAS;

    // Load center element with bounds checking
    if (gindex < in_size) {  // Add size check for main element
        temp[lindex] = net.x[gindex];
    } else {
        temp[lindex] = 0;
    }

    // Load weights into shared memory
    // Each thread loads R/blockDim.x weights
    for (int row = threadIdx.x; row < R; row += blockDim.x) {
        if (gindex < out_size + RADIUS) {
            shared_W[row][threadIdx.x] = net.W[row * out_size + gindex];
        }
    }

    // Load halo elements with bounds checking
    if (threadIdx.x < RADIUS) {
        // Left halo
        if (gindex >= RADIUS) {
            temp[lindex - RADIUS] = net.x[gindex - RADIUS];
        } else {
            temp[lindex - RADIUS] = 0;
        }
        
        // Right halo
        if (gindex + blockDim.x < in_size) {
            temp[lindex + blockDim.x] = net.x[gindex + blockDim.x];
        } else {
            temp[lindex + blockDim.x] = 0;
        }
    }
    __syncthreads(); 
    
    // Compute only for valid output indices
    if (gindex < out_size + RADIUS) {
        
        for (int offset = -RADIUS; offset <= RADIUS; offset++) {
            int row = offset + RADIUS; // 0,1,2 for R=3
            sum += temp[lindex + offset] * shared_W[row][threadIdx.x];
        }
        net.y[gindex] = sigmoid(sum);
    }
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

    // Compute the size of the first output layer
    M = N - (R - 1);

    NeuralNet d_nn;
    allocateNeuralNet(d_nn, N, M);

    // Allocate host memory for x, W
    h_x = (float *)malloc((N+2*RADIUS) * sizeof(float)); fill(h_x, (N+2*RADIUS));
    h_W = (float *)malloc(N * R * sizeof(float)); fill(h_W, N * R);
    
    // Copy x and W to device memory
    cudaMemcpy(d_nn.x, h_x, (N+2*RADIUS) * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_nn.W, h_W, N * R * sizeof(float), cudaMemcpyHostToDevice);

    int final_layer_size = compute_layer_size(N, K - 1);

    printf("N = %d, K = %d, R = %d\n", N, K, R);
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
        int input_layer_size = compute_layer_size(N, t - 1);
        int output_layer_size = compute_layer_size(N, t);

        // Launch the kernel
        forward_propagation<<<(output_layer_size + RADIUS + BLKDIM - 1)/BLKDIM, BLKDIM>>>(
            d_nn,
            output_layer_size
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
    throughput = (double)(N * R) / tnoshared;
    printf("Throughput:\t\t%f items/second\n", throughput);

    // Copy the output layer back to the host
    h_y = (float *)malloc((final_layer_size+2*RADIUS) * sizeof(float)); 
    cudaMemcpy(h_y, d_nn.y, (final_layer_size+2*RADIUS) * sizeof(float), cudaMemcpyDeviceToHost);

    // Clean up host and device allocations.
    freeNeuralNet(d_nn);

    // Reallocate memory for the device
    NeuralNet d_nn_shared;
    allocateNeuralNet(d_nn_shared, N, M);
    
    // Copy x and W to device memory
    cudaMemcpy(d_nn_shared.x, h_x, (N+2*RADIUS) * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_nn_shared.W, h_W, N * R * sizeof(float), cudaMemcpyHostToDevice);


    printf("Shared memory:\t\t");

    tstart = hpc_gettime();
    // Forward propagation with shared memory
    for (int t = 1; t <= K - 1; t++)
    {
        int input_layer_size = compute_layer_size(N, t - 1);
        int output_layer_size = compute_layer_size(N, t);
        
        // Launch the kernel
        forward_propagation_shared<<<(output_layer_size + RADIUS + BLKDIM - 1)/BLKDIM, BLKDIM>>>(
            d_nn_shared,
            input_layer_size,
            output_layer_size
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
    shared_throughput = (double)(N * R) / tshared;
    printf("Throughput:\t\t%f items/second\n", shared_throughput);

    // Copy the output layer back to the host
    h_y_shared = (float *)malloc((final_layer_size+2*RADIUS) * sizeof(float)); 
    cudaMemcpy(h_y_shared, d_nn_shared.y, (final_layer_size+2*RADIUS) * sizeof(float), cudaMemcpyDeviceToHost);

    // Check if the results are the same
    for (int i = RADIUS; i < final_layer_size+RADIUS; i++)
    {
        if (fabs(h_y[i] - h_y_shared[i]) > 1e-6)
        {
            fprintf(stderr, "Results do not match at index %d: %f != %f\n", i, h_y[i], h_y_shared[i]);
            return EXIT_FAILURE;
        }
    }
    printf("Test OK, results match\n");

    // Clean up host and device allocations.
    freeNeuralNet(d_nn_shared);
    free(h_x); free(h_W); free(h_y); free(h_y_shared);

    return EXIT_SUCCESS;
}