#include <iostream>
#include <string>
#include <stdio.h>
#include <algorithm>
#include <thrust/host_vector.h>
#include <cmath>
#include <chrono>
#include <thrust/device_vector.h>
#include "../include/utils.h"
#include "../include/radixSort.h"
#include "../include/timer.h"
#include "../include/cpuRadixSort.h"



void testWithRandomNumbers(const size_t numElems) {
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
    std::cout << "\nSorting completed in GPU(radix sort) in : " << timer.Elapsed() << " ms\n";

    // Validate results
    thrust::host_vector<unsigned int> h_outputVals = d_outputVals;
    thrust::host_vector<unsigned int> h_outputPos = d_outputPos;
    thrust::host_vector<unsigned int> h_referenceVals(numElems);
    thrust::host_vector<unsigned int> h_referencePos(numElems);

    // Measure time for CPU radix sorting
    auto cpuStart = std::chrono::high_resolution_clock::now();

    cpuRadixSort(h_inputVals.data(), h_inputPos.data(),
        h_referenceVals.data(), h_referencePos.data(),
        numElems);

    auto cpuEnd = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpuElapsed = cpuEnd - cpuStart;

    std::cout << "\nSorting completed in CPU(radix sort) in: " << cpuElapsed.count() << " ms\n";    
    
    // Measure time for CPU quick sorting
    cpuStart = std::chrono::high_resolution_clock::now();

    std::sort(h_inputVals.begin(), h_inputVals.end());

    cpuEnd = std::chrono::high_resolution_clock::now();
    cpuElapsed = cpuEnd - cpuStart;

    std::cout << "\nSorting completed in CPU(quick sort) in: " << cpuElapsed.count() << " ms\n";

    /*Check Results
    checkResultsExact(h_referenceVals.data(), h_outputVals.data(), numElems);
    checkResultsExact(h_referencePos.data(), h_outputPos.data(), numElems);
    std::cout << "Tests successful\n";
    */
    }


int main(int argc, char** argv) {
    const size_t numElems = pow(2, 20);
    std::cout << "Running test with " << numElems << " random numbers...\n";
    testWithRandomNumbers(numElems);
    
    return 0;
}
