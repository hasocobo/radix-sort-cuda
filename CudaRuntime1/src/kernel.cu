#include <iostream>
#include <string>
#include <stdio.h>
#include <thrust/host_vector.h>
#include <cmath>
#include <chrono>
#include <thrust/device_vector.h>
#include "../include/utils.h"
#include "../include/radixSort.h"
#include "../include/timer.h"
#include "../include/cpuRadixSort.h"



void testWithRandomNumbers() {
    const size_t numElems = pow(2, 20);
    thrust::host_vector<unsigned int> h_inputVals(numElems);
    thrust::host_vector<unsigned int> h_inputPos(numElems);

    // Generate random 4-digit numbers
    for (size_t i = 0; i < numElems; ++i) {
        h_inputVals[i] = 1000 + rand() % 9000;
        h_inputPos[i] = i;
    }
    /*
    std::cout << "Elements before the sorting: ";
    for (size_t i = 0; i < numElems; ++i) {
        std::cout << h_inputVals[i] << " ";
    }
    */

    thrust::device_vector<unsigned int> d_inputVals = h_inputVals;
    thrust::device_vector<unsigned int> d_inputPos = h_inputPos;
    thrust::device_vector<unsigned int> d_outputVals(numElems);
    thrust::device_vector<unsigned int> d_outputPos(numElems);

    GpuTimer timer;
    timer.Start();

    radixSort(thrust::raw_pointer_cast(d_inputVals.data()),
        thrust::raw_pointer_cast(d_inputPos.data()),
        thrust::raw_pointer_cast(d_outputVals.data()),
        thrust::raw_pointer_cast(d_outputPos.data()),
        numElems);

    timer.Stop();
    /*
    std::cout << "\n\nElements after the sorting: ";
    for (size_t i = 0; i < numElems; ++i) {
        std::cout << d_outputVals[i] << " ";
    }
    */
    std::cout << "\nSorting completed in GPU in : " << timer.Elapsed() << " ms\n";

    // Validate results
    thrust::host_vector<unsigned int> h_outputVals = d_outputVals;
    thrust::host_vector<unsigned int> h_outputPos = d_outputPos;
    thrust::host_vector<unsigned int> h_referenceVals(numElems);
    thrust::host_vector<unsigned int> h_referencePos(numElems);

    // Measure time for CPU sorting
    auto cpuStart = std::chrono::high_resolution_clock::now();

    cpuRadixSort(h_inputVals.data(), h_inputPos.data(),
        h_referenceVals.data(), h_referencePos.data(),
        numElems);

    auto cpuEnd = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpuElapsed = cpuEnd - cpuStart;

    std::cout << "\nSorting completed in CPU in: " << cpuElapsed.count() << " ms\n";

    checkResultsExact(h_referenceVals.data(), h_outputVals.data(), numElems);
    checkResultsExact(h_referencePos.data(), h_outputPos.data(), numElems);

    std::cout << "Tests successful\n";
    }

void listCudaDevices() {
    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);

    if (error != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(error) << std::endl;
        return;
    }

    if (deviceCount == 0) {
        std::cout << "No CUDA-compatible devices found." << std::endl;
        return;
    }

    std::cout << "Number of CUDA-compatible devices: " << deviceCount << std::endl;

    for (int device = 0; device < deviceCount; ++device) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, device);

        std::cout << "\nDevice " << device << ": " << deviceProp.name << std::endl;
        std::cout << "  Compute capability: " << deviceProp.major << "." << deviceProp.minor << std::endl;
        std::cout << "  Total global memory: " << deviceProp.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
        std::cout << "  Shared memory per block: " << deviceProp.sharedMemPerBlock / 1024 << " KB" << std::endl;
        std::cout << "  Registers per block: " << deviceProp.regsPerBlock << std::endl;
        std::cout << "  Warp size: " << deviceProp.warpSize << std::endl;
        std::cout << "  Max threads per block: " << deviceProp.maxThreadsPerBlock << std::endl;
        std::cout << "  Max threads dimensions: (" << deviceProp.maxThreadsDim[0] << ", "
            << deviceProp.maxThreadsDim[1] << ", " << deviceProp.maxThreadsDim[2] << ")" << std::endl;
        std::cout << "  Max grid size: (" << deviceProp.maxGridSize[0] << ", "
            << deviceProp.maxGridSize[1] << ", " << deviceProp.maxGridSize[2] << ")" << std::endl;
        std::cout << "  Memory clock rate: " << deviceProp.memoryClockRate / 1000 << " MHz" << std::endl;
        std::cout << "  Memory bus width: " << deviceProp.memoryBusWidth << "-bit" << std::endl;
    }
}


int main(int argc, char** argv) {
    //listCudaDevices();
    std::cout << "Running test with random numbers...\n";
    testWithRandomNumbers();
    
    return 0;
}
