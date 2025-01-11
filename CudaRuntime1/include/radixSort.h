#pragma once

#include "cuda_runtime.h"

void radixSort(int* d_input, int* d_output, int num_of_elements);