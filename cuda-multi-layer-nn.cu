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
#define BIAS 0.2

/* Define the NeuralNet struct */
/**
 * Neural network struct.
 *
 * @param x The input layer.
 * @param W The weights.
 * @param y The output layer.
 */
struct NeuralNet {
    float *x;
    float *W;
    float *y;
} __align__(16);

/* Define the allocateNeuralNet function */
/**
 * Allocate memory for the neural network.
 *
 * @param net The neural network to allocate.
 * @param N The number of neurons in the first layer.
 * @param M The size of the output layer.
 */
void allocateNeuralNet(NeuralNet &net, const int N, const int M) {
    size_t size_x = N * sizeof(float);
    size_t size_W = N * R * sizeof(float);
    size_t size_y = M * sizeof(float);
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
    if (net.x) {
        cudaFree(net.x);
        net.x = net.W = net.y = nullptr;
    }
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
    return 1.0f / (1.0f + exp(-x));
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
        array[i] = ((float)rand() / RAND_MAX);
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

/* Define the compute_throughput function */
/**
 * Compute the throughput of the network.
 *
 * @param time The time taken to compute the network.
 * @param N The number of neurons in the first layer.
 * @param K The number of layers.
 * @return The throughput of the network.
 */
double compute_throughput(const double time, const int N, const int K)
{
    int processed_items = 0;
    for (int t = 1; t <= K - 1; t++)
    {
        processed_items += compute_layer_size(N, t);
    }
    return processed_items / time;

}

/* Define the forward_propagation kernel without shared memory */
/**
 * Forward propagation kernel. Each thread computes the output of a single neuron in the output layer.
 *
 * @param NeuralNet The neural network.
 * @param in_size Current input layer size.
 */
__global__ void forward_propagation(
    const NeuralNet net,
    const int in_size
) {

    /* Compute the index of the current thread */
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int output_size = in_size - R + 1;


    /* Compute the output for this thread */
    if (i < output_size) {
        float sum = BIAS;
        sum += net.x[i + 0] * net.W[(i * R) + 0] +
               net.x[i + 1] * net.W[(i * R) + 1] +
               net.x[i + 2] * net.W[(i * R) + 2];

        net.y[i] = sigmoid(sum);
    }
}

/* Define the forward_propagation kernel with shared memory */
/**
 * Forward propagation kernel with shared memory. Each thread computes the output of a single neuron in the output layer. Shared memory is used to store the input layer since it is accessed multiple times by adjacent threads.
 *
 * @param NeuralNet The neural network.
 * @param in_size Current input layer size.
 */
__global__ void forward_propagation_shared(
    const NeuralNet net,
    const int in_size
) {

    // Declare shared memory
    __shared__ float shared_x[BLKDIM + R - 1];

    /* Compute the index of the current thread */
    int global_index = threadIdx.x + blockIdx.x * blockDim.x;
    int local_index = threadIdx.x; 

    // Load the main portion of input into shared memory
    if (global_index < in_size) {
        shared_x[local_index] = net.x[global_index];
    } else {
        shared_x[local_index] = 0.0f;
    }

    // Load the extra (R - 1) elements for the boundaries.
    if (threadIdx.x < R - 1) {
        int load_index = global_index + blockDim.x;
        shared_x[local_index + blockDim.x] = (load_index < in_size) ? net.x[load_index] : 0.0f;
    }

    __syncthreads();
    
    // Compute the output layer size
    int output_size = in_size - R + 1;
    
    // Only threads corresponding to valid output indices compute the result.
    if (global_index < output_size) {
        float sum = BIAS;
        // Manual unrolling since R is known at compile time.
        sum += shared_x[local_index + 0] * net.W[(global_index * R) + 0] +
               shared_x[local_index + 1] * net.W[(global_index * R) + 1] +
               shared_x[local_index + 2] * net.W[(global_index * R) + 2];

        net.y[global_index] = sigmoid(sum);
    }
}

int main(int argc, char *argv[])
{
    float *h_x, *h_W, *h_y, *h_y_shared; // Host memory for x, W, and y
    int N = 1024;  // Number of neurons in the first layer
    int K = 2;     // Number of layers
    int M; // Size of the output layer
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

    // Compute the size of the output layer
    M = N - R + 1;

    NeuralNet d_nn;
    allocateNeuralNet(d_nn, N, M);

    // Allocate host memory for x, W
    h_x = (float *)malloc(N * sizeof(float)); fill(h_x, N);
    h_W = (float *)malloc(N * R * sizeof(float)); fill(h_W, N * R);
    
    // Copy x and W to device memory
    cudaMemcpy(d_nn.x, h_x, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_nn.W, h_W, N * R * sizeof(float), cudaMemcpyHostToDevice);

    int final_layer_size = compute_layer_size(N, K - 1);

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
        forward_propagation<<<(output_layer_size + BLKDIM - 1)/BLKDIM, BLKDIM>>>(
            d_nn,
            input_layer_size
        );

        if (t < K - 1)
        {
            // Swap the input and output arays if we are not at the last layer
            cudaMemcpy(d_nn.x, d_nn.y, output_layer_size * sizeof(float), cudaMemcpyDeviceToDevice);
        }
    }
    cudaDeviceSynchronize();
    // Stop the time
    tstop = hpc_gettime();
    tnoshared = tstop - tstart;
    printf("%fs\n", tnoshared);

    // Calculate throughput
    throughput = compute_throughput(tnoshared, N, K);
    printf("Throughput: %f items/second\n", throughput);

    // Copy the output layer back to the host
    h_y = (float *)malloc(final_layer_size * sizeof(float)); 
    cudaMemcpy(h_y, d_nn.y, final_layer_size * sizeof(float), cudaMemcpyDeviceToHost);

    // Clean up host and device allocations.
    freeNeuralNet(d_nn);

    // Reallocate memory for the device
    NeuralNet d_nn_shared;
    allocateNeuralNet(d_nn_shared, N, M);
    
    // Copy x and W to device memory
    cudaMemcpy(d_nn_shared.x, h_x, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_nn_shared.W, h_W, N * R * sizeof(float), cudaMemcpyHostToDevice);


    printf("Shared memory:\t\t");

    tstart = hpc_gettime();
    // Forward propagation with shared memory
    for (int t = 1; t <= K - 1; t++)
    {
        int input_layer_size = compute_layer_size(N, t - 1);
        int output_layer_size = compute_layer_size(N, t);
        
        // Launch the kernel
        forward_propagation_shared<<<(output_layer_size + BLKDIM - 1)/BLKDIM, BLKDIM>>>(
            d_nn_shared,
            input_layer_size
        );

        if (t < K - 1)
        {
            // Swap the input and output arays if we are not at the last layer
            cudaMemcpy(d_nn_shared.x, d_nn_shared.y, output_layer_size * sizeof(float), cudaMemcpyDeviceToDevice);
        }
    }
    cudaDeviceSynchronize();
    // Stop the time
    tstop = hpc_gettime();
    tshared = tstop - tstart;
    // Print the time and speedup w.r.t the non-shared memory version
    printf("%fs (%.2fx speedup)\n", tshared, tnoshared / tshared);

    // Calculate throughput
    shared_throughput = compute_throughput(tshared, N, K);
    printf("Throughput: %f items/second\n", shared_throughput);

    // Copy the output layer back to the host
    h_y_shared = (float *)malloc(final_layer_size * sizeof(float)); 
    cudaMemcpy(h_y_shared, d_nn_shared.y, final_layer_size * sizeof(float), cudaMemcpyDeviceToHost);

    // Check if the results are the same
    for (int i = 0; i < final_layer_size; i++)
    {
        if (fabs(h_y[i] - h_y_shared[i]) > 1e-6)
        {
            fprintf(stderr, "Results do not match at index %d: %f != %f\n", i, h_y[i], h_y_shared[i]);
            return EXIT_FAILURE;
        }
    }
    printf("Results match.\n");

    // Clean up host and device allocations.
    freeNeuralNet(d_nn_shared);
    free(h_x); free(h_W); free(h_y); free(h_y_shared);

    return EXIT_SUCCESS;
}