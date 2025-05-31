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
 * @param x The input layer (padded).
 * @param W The weights.
 * @param y The output layer (padded).
 */
 typedef struct {
    float *base;   /* original cudaMalloc pointer */
    float *x;      /* input buffer (padded: N_max + 2*RADIUS) */
    float *W;      /* weight buffer (N_max * R) */
    float *y;      /* output buffer (padded: M_max + 2*RADIUS, M_max = N_max - (R-1)) */
} NeuralNet;

/* Compute output size of layer `t` (unpadded): #neurons = N0 - t*(R-1) */
/* layer_num_1_based is the computation step number (1 to K-1) */
static inline int unpadded_output_size(int N0, int layer_num_1_based) {
    return N0 - layer_num_1_based * (R - 1);
}

/* Allocate one big chunk for x, W, and y, then slice it up */
/* N_initial is the number of neurons in the very first input layer (unpadded) */
void allocateNeuralNet(NeuralNet *net, int N_initial) {
    size_t size_x_padded = (N_initial + 2 * RADIUS) * sizeof(float);
    size_t size_W = (size_t)N_initial * R * sizeof(float);
    size_t size_y_padded = (N_initial + 2 * RADIUS) * sizeof(float);
    size_t total_size = size_x_padded + size_W + size_y_padded;
    cudaError_t err;

    err = cudaMalloc((void**)&net->base, total_size);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    net->x = net->base;
    net->W = (float *)((char *)net->base + size_x_padded);
    net->y = (float *)((char *)net->base + size_x_padded + size_W);
}

/* Free the single allocation */
void freeNeuralNet(NeuralNet *net) {
    if (net->base) {
        cudaFree(net->base);
        net->base = net->x = net->W = net->y = NULL;
    }
}

/* Define the Sigmoid function */
__device__ float sigmoid_device(const float v)
{
    return 1.0f / (1.0f + expf(-v));
}

/* Define the fill function */
void fill(float *arr, size_t size)
{
    size_t i;
    for (i = 0; i < size; ++i)
    {
        arr[i] = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f; /* e.g., -1 to 1 */
    }
}

/* CPU-side throughput calculation: sum output sizes of layers 1..K-1 */
double compute_throughput(long long N0_initial, int K_total_layers, double secs) {
    long long total_neurons_computed = 0;
    int t;
    /* K_total_layers includes the input layer. K-1 computation steps. */
    for (t = 1; t <= K_total_layers - 1; ++t) { /* t is the computation step number */
        total_neurons_computed += unpadded_output_size((int)N0_initial, t);
    }
    if (secs == 0) return 0;
    return (double)total_neurons_computed / secs;
}

/*
 * Forward propagation kernel without shared memory.
 */
__global__ void forward_propagation(
    const float* x_padded,
    const float* W,
    float* y_padded,
    int current_out_size_unpadded
) {
    const int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    int offset;
    int k_stencil_element;

    if (gtid >= current_out_size_unpadded) return;

    /* padded_idx_center is where gtid's output is written and where its central input is read */
    const int padded_idx_center = gtid + RADIUS;
    float sum = BIAS;

    for (offset = -RADIUS, k_stencil_element = 0; offset <= RADIUS; ++offset, ++k_stencil_element) {
        sum += x_padded[padded_idx_center + offset] * W[gtid * R + k_stencil_element];
    }

    y_padded[padded_idx_center] = sigmoid_device(sum);
}

/*
 * Forward propagation kernel with shared memory,
 */
__global__ void forward_propagation_shared(
    const float* x_padded,
    const float* W,
    float* y_padded,
    int N0_initial_padded_size,
    int current_out_size_unpadded
) {
    __shared__ float s_tile[BLKDIM + 2 * RADIUS];

    const int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    const int gindex_padded_center = gtid + RADIUS;
    const int lindex_shared_center = threadIdx.x + RADIUS;

    int k_stencil_element;
    float sum;

    /* Read input elements into shared memory */
    if (gindex_padded_center < N0_initial_padded_size) {
        s_tile[lindex_shared_center] = x_padded[gindex_padded_center];
    } else {
        s_tile[lindex_shared_center] = 0.0f;
    }

    /* First RADIUS threads load left and right halos for the block */
    if (threadIdx.x < RADIUS) {
        /* Load left halo element: s_tile[lindex_shared_center - RADIUS] from x_padded[gindex_padded_center - RADIUS] */
        if (gindex_padded_center - RADIUS >= 0) {
            s_tile[lindex_shared_center - RADIUS] = x_padded[gindex_padded_center - RADIUS];
        } else {
             s_tile[lindex_shared_center - RADIUS] = 0.0f;
        }

        /* Load right halo element: s_tile[lindex_shared_center + BLKDIM] from x_padded[gindex_padded_center + BLKDIM] */
        if (gindex_padded_center + BLKDIM < N0_initial_padded_size) { // Check upper bound of x_padded
            s_tile[lindex_shared_center + BLKDIM] = x_padded[gindex_padded_center + BLKDIM];
        } else {
            s_tile[lindex_shared_center + BLKDIM] = 0.0f; // Pad if attempting to read beyond end
        }
    }
    __syncthreads();

    /* only compute if gtid is a valid output neuron for this layer */
    if (gtid >= current_out_size_unpadded) return;

    sum = BIAS;

    for (k_stencil_element = 0; k_stencil_element < R; ++k_stencil_element) {
        int offset_from_center = k_stencil_element - RADIUS;
        sum += s_tile[lindex_shared_center + offset_from_center] * W[gtid * R + k_stencil_element];
    }

    y_padded[gindex_padded_center] = sigmoid_device(sum);
}

int main(int argc, char *argv[])
{
    float *h_x_padded_initial, *h_W, *h_y_check_no_shared, *h_y_check_shared;
    int N0_initial = 1024;
    int K_total_layers = 2;
    double tstart, tstop, tnoshared = 0.0, tshared = 0.0;
    double throughput_val, shared_throughput_val;
    NeuralNet d_nn;
    NeuralNet d_nn_shared;
    int t; /* Loop variable for layers */
    int final_layer_unpadded_size;
    int current_layer_unpadded_output_size;
    int i; /* Generic loop variable */
    int verification_failed = 0;

    if (argc > 3) {
        fprintf(stderr, "Usage: %s [N_initial (default %d)] [K_total_layers (default %d)]\n",
                argv[0], N0_initial, K_total_layers);
        return EXIT_FAILURE;
    }
    if (argc >= 2) N0_initial = atoi(argv[1]);
    if (argc >= 3) K_total_layers = atoi(argv[2]);

    /* Validate inputs */
    if (K_total_layers < 2) {
        fprintf(stderr, "K_total_layers must be at least 2 (input + one output layer).\n");
        return EXIT_FAILURE;
    }
    if (N0_initial < 1) {
        fprintf(stderr, "N_initial must be a positive integer.\n");
        return EXIT_FAILURE;
    }
    if (R % 2 == 0) {
        fprintf(stderr, "R (stencil size) must be odd for a centered stencil.\n");
        return EXIT_FAILURE;
    }
    if ( (K_total_layers > 1) && (unpadded_output_size(N0_initial, K_total_layers - 1) < 1) ) {
        fprintf(stderr, "Network configuration results in zero or negative neurons in the final layer.\n");
        fprintf(stderr, "N0=%d, K=%d, R=%d. Final layer would have %d neurons.\n",
                N0_initial, K_total_layers, R, unpadded_output_size(N0_initial, K_total_layers-1));
        return EXIT_FAILURE;
    }
    if (N0_initial < R && K_total_layers > 1) {
         fprintf(stderr, "N_initial (%d) must be at least R (%d) for the first computation layer if K > 1.\n", N0_initial, R);
         return EXIT_FAILURE;
    }

    printf("Configuration: N_initial=%d, K_total_layers=%d, R=%d, RADIUS=%d, BLKDIM=%d\n",
           N0_initial, K_total_layers, R, RADIUS, BLKDIM);

    /* Allocate host memory */
    h_x_padded_initial = (float *)malloc((N0_initial + 2 * RADIUS) * sizeof(float));
    h_W = (float *)malloc((size_t)N0_initial * R * sizeof(float));
    if (!h_x_padded_initial || !h_W) {
        fprintf(stderr, "Host malloc failed.\n");
        return EXIT_FAILURE;
    }

    /* Fill initial host data */
    /* Initialize padding regions in h_x_padded_initial to 0 */
    for (i = 0; i < RADIUS; ++i) {
        h_x_padded_initial[i] = 0.0f;
        h_x_padded_initial[N0_initial + RADIUS + i] = 0.0f;
    }
    fill(h_x_padded_initial + RADIUS, N0_initial); /* Fill the actual N0_initial data */
    fill(h_W, (size_t)N0_initial * R);

    /* --- Test without shared memory --- */
    printf("No shared memory:\t");
    allocateNeuralNet(&d_nn, N0_initial);
    cudaMemcpy(d_nn.x, h_x_padded_initial, (N0_initial + 2 * RADIUS) * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_nn.W, h_W, (size_t)N0_initial * R * sizeof(float), cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();
    tstart = hpc_gettime();

    for (t = 1; t <= K_total_layers - 1; ++t) {
        current_layer_unpadded_output_size = unpadded_output_size(N0_initial, t);
        if (current_layer_unpadded_output_size < 1) {
             fprintf(stderr, "Layer %d would have %d neurons. Stopping.\n", t, current_layer_unpadded_output_size);
             break;
        }

        forward_propagation<<<(current_layer_unpadded_output_size + BLKDIM - 1) / BLKDIM, BLKDIM>>>(
            d_nn.x, d_nn.W, d_nn.y,
            current_layer_unpadded_output_size
        );

        /* Swap buffers for next iteration, unless it's the last one */
        if (t < K_total_layers - 1) {
            float *temp_ptr = d_nn.x;
            d_nn.x = d_nn.y;
            d_nn.y = temp_ptr;
        }
    }
    cudaDeviceSynchronize();
    tstop = hpc_gettime();
    tnoshared = tstop - tstart;

    final_layer_unpadded_size = unpadded_output_size(N0_initial, K_total_layers - 1);
    if (final_layer_unpadded_size < 1 && K_total_layers > 1) final_layer_unpadded_size = 0;
    
    h_y_check_no_shared = (float *)malloc((final_layer_unpadded_size + 2*RADIUS) * sizeof(float));
    if (!h_y_check_no_shared && final_layer_unpadded_size > 0) { fprintf(stderr, "Host malloc failed for no_shared results.\n"); return EXIT_FAILURE; }

    if (final_layer_unpadded_size > 0) {
        cudaMemcpy(h_y_check_no_shared, d_nn.y, (final_layer_unpadded_size + 2*RADIUS) * sizeof(float), cudaMemcpyDeviceToHost);
    }
    
    printf("%fs\n", tnoshared);
    throughput_val = compute_throughput(N0_initial, K_total_layers, tnoshared);
    printf("Throughput:\t\t%f items/second\n", throughput_val);
    freeNeuralNet(&d_nn);


    /* --- Test with shared memory --- */
    printf("Shared memory:\t\t");
    allocateNeuralNet(&d_nn_shared, N0_initial);
    cudaMemcpy(d_nn_shared.x, h_x_padded_initial, (N0_initial + 2 * RADIUS) * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_nn_shared.W, h_W, (size_t)N0_initial * R * sizeof(float), cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();
    tstart = hpc_gettime();

    for (t = 1; t <= K_total_layers - 1; ++t) {
        current_layer_unpadded_output_size = unpadded_output_size(N0_initial, t);
         if (current_layer_unpadded_output_size < 1) break;

        forward_propagation_shared<<<(current_layer_unpadded_output_size + BLKDIM - 1) / BLKDIM, BLKDIM>>>(
            d_nn_shared.x,
            d_nn_shared.W,
            d_nn_shared.y,
            N0_initial + 2 * RADIUS, /* Total elements in the padded x buffer */
            current_layer_unpadded_output_size
        );
        if (t < K_total_layers - 1) {
            float *temp_ptr = d_nn_shared.x;
            d_nn_shared.x = d_nn_shared.y;
            d_nn_shared.y = temp_ptr;
        }
    }
    cudaDeviceSynchronize();
    tstop = hpc_gettime();
    tshared = tstop - tstart;

    h_y_check_shared = (float *)malloc((final_layer_unpadded_size + 2*RADIUS) * sizeof(float));
     if (!h_y_check_shared && final_layer_unpadded_size > 0) { fprintf(stderr, "Host malloc failed for shared results.\n"); return EXIT_FAILURE; }

    if (final_layer_unpadded_size > 0) {
        cudaMemcpy(h_y_check_shared, d_nn_shared.y, (final_layer_unpadded_size + 2*RADIUS) * sizeof(float), cudaMemcpyDeviceToHost);
    }

    printf("%fs (%.2fx speedup)\n", tshared, tnoshared / tshared);
    shared_throughput_val = compute_throughput(N0_initial, K_total_layers, tshared);
    printf("Throughput:\t\t%f items/second\n", shared_throughput_val);
    freeNeuralNet(&d_nn_shared);

    /* --- Verification --- */
    if (final_layer_unpadded_size > 0) {
        printf("\nVerifying results...\n");
        verification_failed = 0;
        for (i = 0; i < final_layer_unpadded_size; ++i) {
            /* skip padding */
            float val_no_shared = h_y_check_no_shared[i + RADIUS];
            float val_shared = h_y_check_shared[i + RADIUS];
            if (fabsf(val_no_shared - val_shared) > 1e-5) {
                fprintf(stderr, "Verification FAILED at index %d (unpadded): NoShared=%.6f, Shared=%.6f, Diff=%.6f\n",
                        i, val_no_shared, val_shared, fabsf(val_no_shared - val_shared));
                verification_failed = 1;
                break;
            }
        }
        if (!verification_failed) {
            printf("Verification PASSED.\n");
        }
    } else {
        printf("\nNo output neurons in final layer to verify (K=%d, N0=%d).\n", K_total_layers, N0_initial);
    }


    /* Clean up host memory */
    free(h_x_padded_initial);
    free(h_W);
    if (final_layer_unpadded_size > 0) {
        free(h_y_check_no_shared);
        free(h_y_check_shared);
    }

    return verification_failed ? EXIT_FAILURE : EXIT_SUCCESS;
}