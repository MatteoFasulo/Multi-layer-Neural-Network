/****************************************************************************
 *
 * multi-layer-nn.c - Multi Layer Neural Network with OpenMP
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
 * gcc -std=c99 -Wall -Wpedantic -fopenmp multi-layer-nn.c -o multi-layer-nn -lm
 *
 * Run with:
 * ./multi-layer-nn [N] [K]
 *
 * (N = first layer n° neurons; default 1024)
 * (K = number of layers; default 2)
 *
 ****************************************************************************/

#include "hpc.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

#define R 3
#define BIAS 0.2


/* Define the Sigmoid function using the math.h lib*/
/**
 * Sigmoid activation function.
 *
 * @param x Input value.
 * @return The sigmoid of the input value.
 */
double sigmoid(const double x)
{
    return 1.0 / (1.0 + exp(-x));
}

/* Define the fill function */
/**
 * Fill an array with random values.
 *
 * @param array The array to fill.
 * @param size The size of the array.
 */
void fill( double *array, const size_t size)
{
    for (size_t i = 0; i < size; i++) 
    {
        array[i] = ((double)rand() / RAND_MAX);
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

/* Define the forward_pass function */
/**
 * Perform the forward pass of the neural network.
 *
 * @param x The input array.
 * @param W The weights array.
 * @param y The output array.
 * @param n The size of the output array.
 */
void forward_pass(
    const double *x,
    const double *W,
    double *y,
    const int n
) { 
    
    #pragma omp parallel for schedule(static) default(none) shared(x, W, y, n)
    for (int i = 0; i < n; i++) {
        double sum = BIAS;
        for (int j = 0; j < R; j++) {
            sum += x[i + j] * W[(i * R) + j];
        }

        y[i] = sigmoid(sum);
    }
}

int main(const int argc, char *argv[] )
{
    double *x, *W, *y; // Input, weights, output
    int N = 1024; // Number of neurons in the first layer
    int K = 2; // Number of layers
    int M; // Size of the output layer
    double tstart, tstop; // Timing variables

    if (argc > 3) {
        fprintf(stderr, "Usage: %s [N (default %d)] [K (default %d)]\n", argv[0], N, K);
        return EXIT_FAILURE;
    }
    
    if (argc >= 2)
        N = atoi(argv[1]);
    if (argc == 3)
        K = atoi(argv[2]);

    // Validate the input arguments
    if (K < 2) {
        printf("K must be greater than 1\n");
        return EXIT_FAILURE;
    }

    if (N < 1)
    {
        fprintf(stderr, "N must be a positive integer.\n");
        return EXIT_FAILURE;
    }

    // Compute the size of the output layer
    M = N - R + 1;

    // Allocate memory for the input, weights, and output arrays
    x = (double *)malloc(N * sizeof(double)); fill(x, N);
    W = (double *)malloc(N * R * sizeof(double)); fill(W, N * R);
    y = (double *)malloc(M * sizeof(double));

    /**
     ** Forward propagation
     **/
    // Start the time
    tstart = hpc_gettime();
    // Forward pass
    for (int t = 1; t <= K - 1; t++)
    {
        int output_layer_size = compute_layer_size(N, t);

        // Do the forward pass
        forward_pass(
            x, 
            W,
            y, 
            output_layer_size
        );

        // Swap the input and output arrays if we are not at the last layer
        if (t < K - 1)
        {
            double *temp = x;
            x = y;
            y = temp;
        }
    }
    // Stop the time
    tstop = hpc_gettime();
    printf("\nExecution time %fs\n", tstop - tstart);

    // Calculate throughput
    printf("Throughput: %f items/second\n", (double)N / (tstop - tstart));

    // Free the memory
    free(x); free(W); free(y);

    return EXIT_SUCCESS;
}