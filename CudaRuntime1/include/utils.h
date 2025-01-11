#pragma once

#include <cuda_runtime.h>
#include <iostream>

#define CUDA_CHECK(call)                                         \
    do {                                                        \
        cudaError_t error = call;                               \
        if (error != cudaSuccess) {                             \
            std::cerr << "CUDA Error: " << cudaGetErrorString(error) << "\n"; \
            exit(1);                                            \
        }                                                       \
    } while (0)