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
 
 /* Compute input size for layer `t`: layer 1 sees N, layer 2 sees N-(R-1), ... */
 static inline int input_size(int N0, int layer) {
     return N0 - (layer-1)*(R-1);
 }
 
 /* Compute output size of layer `t`: #neurons = N0 - t*(R-1) */
 static inline int output_size(int N0, int layer) {
     return N0 - layer*(R-1);
 }
 
 /* Define the Sigmoid function using the math.h lib*/
 /**
  * Sigmoid activation function.
  *
  * @param v Input value.
  * @return The sigmoid of the input value.
  */
 float sigmoid(const float v)
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
 void fill( float *arr, const size_t size)
 {
     for (size_t i = 0; i < size; i++) 
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
     const float *x,
     const float *W,
     float *y,
     int out_size
 ) { 
     
     #pragma omp parallel for schedule(static) default(none) shared(x, W, y, out_size)
     for (int i = 0; i < out_size; i++) {
         float sum = BIAS;
         // Manual loop unrolling since R is fixed at 3
         sum += x[i + 0] * W[(i * R) + 0];
         sum += x[i + 1] * W[(i * R) + 1];
         sum += x[i + 2] * W[(i * R) + 2];
 
         y[i] = sigmoid(sum);
     }
 }
 
 int main(const int argc, char *argv[] )
 {
     float *x, *W, *y; // Input, weights, output
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
     x = (float *)malloc(N * sizeof(float)); fill(x, N);
     W = (float *)malloc(N * R * sizeof(float)); fill(W, N * R);
     y = (float *)malloc(M * sizeof(float));
 
     /**
      ** Forward propagation
      **/
     // Start the time
     tstart = hpc_gettime();
     // Forward pass
     for (int t = 1; t <= K - 1; t++)
     {
         int out_sz = output_size(N, t);
 
         // Do the forward pass
         forward_pass(
             x, 
             W,
             y, 
             out_sz
         );
 
         // Swap the input and output arrays if we are not at the last layer
         if (t < K - 1)
         {
             float *temp = x;
             x = y;
             y = temp;
         }
     }
     // Stop the time
     tstop = hpc_gettime();
     printf("\nExecution time %fs\n", tstop - tstart);
 
     // Calculate throughput
     printf("Throughput: %f items/second\n", compute_throughput(N, K, tstop - tstart));
 
     // Free the memory
     free(x); free(W); free(y);
 
     return EXIT_SUCCESS;
 }