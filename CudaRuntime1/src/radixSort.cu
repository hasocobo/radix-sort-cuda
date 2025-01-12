#ifndef __CUDACC__ 
#define __CUDACC__ // Needed for Visual Studio to recognize __syncthreads
#endif

#define BLOCK_SIZE 1024

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "../include/radixSort.h"
#include "../include/utils.h"

#include <math.h>
#include <stdio.h>

// for a thread tid, check if a (bit & inputVals[tid]) == 0
__global__
void checkBit(unsigned int* const dInputVals, unsigned int* const dOutputPredicate,
	const unsigned int bit, const size_t numElems)
{
	const unsigned int id = blockDim.x * blockIdx.x + threadIdx.x;
	if (id >= numElems)
		return;

	int predicate = ((dInputVals[id] & bit) == 0);
	dOutputPredicate[id] = predicate;
}

// Flips the bits in the list for example: 0 1 0 1 1 1 0 -> 1 0 1 0 0 0 1
__global__
void flipBit(unsigned int* const dList, const size_t numElems)
{
	const unsigned int id = blockDim.x * blockIdx.x + threadIdx.x;
	if (id >= numElems)
		return;

	dList[id] = ((dList[id] + 1) % 2);
}

// Blelloch scan or also known as partial prefix sum makes a sum operation in the list similar to += operator:
// scan(0 1 0 1 1 1) -> 0 0 1 1 2 3
__global__
void partialPrefixSum(unsigned int* const dList, unsigned int* const dBlockSums, const size_t numElems)
{
	extern __shared__ unsigned int sBlockScan[];

	const unsigned int tid = threadIdx.x;
	const unsigned int id = blockDim.x * blockIdx.x + tid;

	if (id >= numElems)
		sBlockScan[tid] = 0;
	else
		sBlockScan[tid] = dList[id];
	__syncthreads();

	unsigned int i;
	for (i = 2; i <= blockDim.x; i <<= 1) {
		if ((tid + 1) % i == 0) {
			unsigned int neighborOffset = i >> 1;
			sBlockScan[tid] += sBlockScan[tid - neighborOffset];
		}
		__syncthreads();
	}
	i >>= 1;
	if (tid == (blockDim.x - 1)) {
		dBlockSums[blockIdx.x] = sBlockScan[tid];
		sBlockScan[tid] = 0;
	}
	__syncthreads();

	for (i = i; i >= 2; i >>= 1) {
		if ((tid + 1) % i == 0) {
			unsigned int neighborOffset = i >> 1;
			unsigned int oldNeighbor = sBlockScan[tid - neighborOffset];
			sBlockScan[tid - neighborOffset] = sBlockScan[tid];
			sBlockScan[tid] += oldNeighbor;
		}
		__syncthreads();
	}

	if (id < numElems) {
		dList[id] = sBlockScan[tid];
	}
}

__global__
void incrementPrefixSum(unsigned int* const dPredicateScan,
	unsigned int* const dBlockSumScan, const size_t numElems)
{
	const unsigned int id = blockDim.x * blockIdx.x + threadIdx.x;
	if (id >= numElems)
		return;

	dPredicateScan[id] += dBlockSumScan[blockIdx.x];
}

// swap elements to their new locations.
__global__
void scatter(unsigned int* const dInput, unsigned int* const dOutput,
	unsigned int* const dPredicateTrueScan, unsigned int* const dPredicateFalseScan,
	unsigned int* const dPredicateFalse, unsigned int* const dNumPredicateTrueElements,
	const size_t numElems)
{
	const unsigned int id = blockDim.x * blockIdx.x + threadIdx.x;
	if (id >= numElems)
		return;

	unsigned int newLoc;
	if (dPredicateFalse[id] == 1) {
		newLoc = dPredicateFalseScan[id] + *dNumPredicateTrueElements;
	}
	else {
		newLoc = dPredicateTrueScan[id];
	}

	if (newLoc >= numElems)
		printf("ALERT dPredicateFalse[id]: %i newLoc: %i numElems: %i\n", dPredicateFalse[id], newLoc, numElems);

	dOutput[newLoc] = dInput[id];
}

unsigned int* dPredicate; // predicate: (x & 1) == 0 that is if a bit is 0, predicate is 1 and vice versa
unsigned int* dPredicateTrueScan; // prefix sum of predicates that are 1
unsigned int* dPredicateFalseScan; // "" "" "" "" that are 0
unsigned int* dNumPredicateTrueElements; // predicate elements that are 1 
unsigned int* dNumPredicateFalseElements; // predicate elements that are 0
unsigned int* dBlockSums;

void radixSort(unsigned int* const dInputVals,
	unsigned int* const dInputPos,
	unsigned int* const dOutputVals,
	unsigned int* const dOutputPos,
	const size_t numElems)
{
	int blockSize = BLOCK_SIZE;

	size_t size = sizeof(unsigned int) * numElems;
	int gridSize = ceil(float(numElems) / float(blockSize));

	checkCudaErrors(cudaMalloc((void**)&dPredicate, size));
	checkCudaErrors(cudaMalloc((void**)&dPredicateTrueScan, size));
	checkCudaErrors(cudaMalloc((void**)&dPredicateFalseScan, size));
	checkCudaErrors(cudaMalloc((void**)&dNumPredicateTrueElements, sizeof(unsigned int)));
	checkCudaErrors(cudaMalloc((void**)&dNumPredicateFalseElements, sizeof(unsigned int)));
	checkCudaErrors(cudaMalloc((void**)&dBlockSums, gridSize * sizeof(unsigned int)));

	unsigned int nsb; //next significant bit
	unsigned int maxBits = 31;
	for (unsigned int bit = 0; bit < maxBits; bit++) {
		nsb = 1 << bit;

		if ((bit + 1) % 2 == 1) {
			checkBit << <gridSize, blockSize >> > (dInputVals, dPredicate, nsb, numElems);
		}
		else {
			checkBit << <gridSize, blockSize >> > (dOutputVals, dPredicate, nsb, numElems);
		}
		
		cudaDeviceSynchronize();
		
		checkCudaErrors(cudaGetLastError());

		
		checkCudaErrors(cudaMemcpy(dPredicateTrueScan, dPredicate, size, cudaMemcpyDeviceToDevice));
		checkCudaErrors(cudaMemset(dBlockSums, 0, gridSize * sizeof(unsigned int)));


		partialPrefixSum << <gridSize, blockSize, sizeof(unsigned int)* blockSize >> >
			(dPredicateTrueScan, dBlockSums, numElems);
		
		cudaDeviceSynchronize();

		checkCudaErrors(cudaGetLastError());


		partialPrefixSum << <1, BLOCK_SIZE, sizeof(unsigned int)* BLOCK_SIZE >> >
			(dBlockSums, dNumPredicateTrueElements, gridSize);
		
		cudaDeviceSynchronize();
		
		checkCudaErrors(cudaGetLastError());


		incrementPrefixSum << <gridSize, blockSize >> >
			(dPredicateTrueScan, dBlockSums, numElems);
		
		cudaDeviceSynchronize();
		
		checkCudaErrors(cudaGetLastError());


		flipBit << <gridSize, blockSize >> >
			(dPredicate, numElems);
		
		cudaDeviceSynchronize();
		
		checkCudaErrors(cudaGetLastError());

		
		checkCudaErrors(cudaMemcpy(dPredicateFalseScan, dPredicate, size, cudaMemcpyDeviceToDevice));
		checkCudaErrors(cudaMemset(dBlockSums, 0, gridSize * sizeof(unsigned int)));

		
		partialPrefixSum << <gridSize, blockSize, sizeof(unsigned int)* blockSize >> >
			(dPredicateFalseScan, dBlockSums, numElems);
		cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

		
		partialPrefixSum << <1, BLOCK_SIZE, sizeof(unsigned int)* BLOCK_SIZE >> >
			(dBlockSums, dNumPredicateFalseElements, gridSize);
		cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

		
		incrementPrefixSum << <gridSize, blockSize >> >
			(dPredicateFalseScan, dBlockSums, numElems);
		cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

		
		// if bit is 0
		if ((bit + 1) % 2 == 1) {
			scatter << <gridSize, blockSize >> >
				(dInputVals, dOutputVals, dPredicateTrueScan, dPredicateFalseScan,
				dPredicate, dNumPredicateTrueElements, numElems);
			
			cudaDeviceSynchronize();
			
			checkCudaErrors(cudaGetLastError());
		

			scatter << <gridSize, blockSize >> > (dInputPos, dOutputPos, dPredicateTrueScan, dPredicateFalseScan,
				dPredicate, dNumPredicateTrueElements, numElems);
			
			cudaDeviceSynchronize();
			
			checkCudaErrors(cudaGetLastError());
		}
		else {
			scatter << <gridSize, blockSize >> > (dOutputVals, dInputVals, dPredicateTrueScan, dPredicateFalseScan,
				dPredicate, dNumPredicateTrueElements, numElems);
			
			cudaDeviceSynchronize();
			
			checkCudaErrors(cudaGetLastError());
			
			
			scatter << <gridSize, blockSize >> > (dOutputPos, dInputPos, dPredicateTrueScan, dPredicateFalseScan,
				dPredicate, dNumPredicateTrueElements, numElems);
			
			cudaDeviceSynchronize();
			
			checkCudaErrors(cudaGetLastError());
		}
	}

	checkCudaErrors(cudaFree(dPredicate));
	checkCudaErrors(cudaFree(dPredicateTrueScan));
	checkCudaErrors(cudaFree(dPredicateFalseScan));
	checkCudaErrors(cudaFree(dNumPredicateTrueElements));
	checkCudaErrors(cudaFree(dNumPredicateFalseElements));
	checkCudaErrors(cudaFree(dBlockSums));
}
