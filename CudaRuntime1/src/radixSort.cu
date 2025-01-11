#ifndef __CUDACC__ 
#define __CUDACC__ // Needed for Visual Studio to recognize __syncthreads and atomicAdd
#endif

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "../include/radixSort.h"
#include <stdio.h>

// Kernel to perform counting sort based on the digit at the given place value
__global__ void countSortKernel(int* d_input, int* d_output, int* d_hist, int num_elements, int digit_place) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Check for valid index
    if (idx >= num_elements) return;

    // Count occurrences of each digit in the input array
    int digit = (d_input[idx] / digit_place) % 10;

    // Use atomicAdd on global memory to update the histogram
    atomicAdd(&d_hist[digit], 1);

    __syncthreads(); // Make sure histogram updates are done

    // Perform prefix sum (cumulative sum) on the histogram using shared memory
    __shared__ int hist[10];
    if (threadIdx.x < 10) {
        hist[threadIdx.x] = d_hist[threadIdx.x];
    }

    __syncthreads();

    // Compute prefix sum for histogram in shared memory
    for (int stride = 1; stride < 10; stride *= 2) {
        if (threadIdx.x >= stride) {
            hist[threadIdx.x] += hist[threadIdx.x - stride];
        }
        __syncthreads();
    }

    // After computing prefix sum, place elements in the correct position
    if (idx < num_elements) {
        int digit = (d_input[idx] / digit_place) % 10;
        int pos = hist[digit] - 1;  // Get the correct position from the cumulative histogram
        d_output[pos] = d_input[idx];   // Place the element in the output array

        // Synchronize threads before proceeding
        __syncthreads();

        // Update histogram to track the next position for this digit
        atomicAdd(&d_hist[digit], 1);
    }
}

void radixSort(int* d_input, int* d_output, int num_elements) {
    const int threads_per_block = 256;
    const int blocks = (num_elements + threads_per_block - 1) / threads_per_block;

    int* d_hist;
    cudaMalloc(&d_hist, 10 * sizeof(int));  // Allocate memory for the histogram
    cudaMemset(d_hist, 0, 10 * sizeof(int)); // Initialize the histogram to 0

    // Loop over each digit place (1's, 10's, 100's, etc.)
    for (int digit_place = 1; digit_place <= 1000000; digit_place *= 10) {
        countSortKernel << <blocks, threads_per_block >> > (d_input, d_output, d_hist, num_elements, digit_place);

        // Debug: Check for kernel errors
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
        }

        cudaDeviceSynchronize();  // Synchronize to ensure all threads are done

        // Debug: Verify the output after each kernel execution
        int* temp = d_input;
        d_input = d_output;
        d_output = temp;

        // Reset the histogram for the next pass
        cudaMemset(d_hist, 0, 10 * sizeof(int));
    }

    cudaFree(d_hist);  // Free the histogram memory
}
