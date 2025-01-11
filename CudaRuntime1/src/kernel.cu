
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "../include/radixSort.h"

#include <stdio.h>

int main() {
    const int num_elements{ 10 };
    int h_input[num_elements] = { 154860, 5245, 2289, 2689, 1589, 16878, 222, 5966, 15612, 2387 };
    int h_output[num_elements];

    int* d_input = nullptr;
    int* d_output = nullptr;

    cudaMalloc(&d_input, num_elements * sizeof(int));
    cudaMalloc(&d_output, num_elements * sizeof(int));

    cudaMemcpy(d_input, h_input, num_elements * sizeof(int), cudaMemcpyHostToDevice);

    radixSort(d_input, d_output, num_elements);


    cudaMemcpy(h_output, d_output, num_elements * sizeof(int), cudaMemcpyDeviceToHost);

    printf("Sorted output: "); // I use printf instead of cout because device code doesn't support cout
    for (int i = 0; i < num_elements; i++) {
        printf("%d ", h_output[i]);
    }
    printf("\n");

    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
