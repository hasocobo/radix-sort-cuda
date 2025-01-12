#pragma once

#include "cuda_runtime.h"

void radixSort
(
    unsigned int* const inputVals,
    unsigned int* const inputPos,
    unsigned int* const outputVals,
    unsigned int* const outputPos,
    const size_t numElems
);