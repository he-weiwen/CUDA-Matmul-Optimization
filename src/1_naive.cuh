#pragma once

/*
 * Kernel 1: Naive SGEMM Implementation
 *
 * Goal: Get a working baseline implementation
 * Expected performance: ~300 GFLOPs
 *
 * Each thread computes ONE element of the output matrix C.
 *
 * C[row][col] = sum_{k=0}^{K-1} A[row][k] * B[k][col]
 *
 * Reference diagram: ../blog_reference/images/naive-kernel.png
 * Reference implementation: ../SGEMM_CUDA/src/kernels/1_naive.cuh
 *
 * TODO:
 * 1. Map blockIdx and threadIdx to row and column indices
 * 2. Add bounds checking for non-square matrices
 * 3. Implement the dot product loop
 * 4. Apply alpha and beta scaling
 */

__global__ void sgemm_naive(int M, int N, int K, float alpha,
                            const float *A, const float *B, float beta, float *C) {
    // TODO: Calculate the row index for this thread
    // Hint: Use blockIdx.y, blockDim.y, and threadIdx.y
    // row = block.x + 
    const int row = blockIdx.y * blockDim.y + threadIdx.y;  // FIX THIS

    // TODO: Calculate the column index for this thread
    // Hint: Use blockIdx.x, blockDim.x, and threadIdx.x
    const int col = blockIdx.x * blockDim.x + threadIdx.x;  // FIX THIS

    // TODO: Add bounds checking
    // We need this because the matrix dimensions may not be
    // perfectly divisible by the block dimensions
    if (row < M && col < N) {  // FIX THIS: Check row < M && col < N
        float sum = 0.0f;

        // TODO: Implement the dot product
        // Loop over K, accumulating A[row][k] * B[k][col]
        for (int k = 0; k < K; k++) {
            // TODO: Calculate the correct indices into A and B
            // Remember: matrices are stored in row-major order
            // A[row][k] is at index: row * K + k
            // B[k][col] is at index: k * N + col
            sum += A[row * K + k] * B[k * N + col];  // FIX THIS
        }

        // TODO: Write the result with alpha/beta scaling
        // C[row][col] = alpha * sum + beta * C[row][col]
        // C[row * N + col] = ...
        C[row * N + col] = alpha * sum + beta * C[row * N + col];
    }
}

/*
 * Questions to think about after implementing:
 *
 * 1. Why is blockDim set to (32, 32)? What happens with other values?
 *      ans: presumably because an SM is 1024 threads and 32 * 32 = 1024, so we don't waste any threads in a block?
 * 2. When threads 0-31 (a warp) execute line:
 *        sum += A[row * K + k] * B[k * N + col]
 *    - What memory addresses do they access in A? A[k]. A[K + k], ... A[31 * K + k]
 *    - What memory addresses do they access in B? B[k], B[k + 1], ..., B[k + 31]
 *    - Are these accesses coalesced? presumably memory access to A is not coalesced but the memory access to B is?
 *
 * 3. How many global memory loads does each thread perform?
 *    - For A: __K__ loads per thread
 *    - For B: __K__ loads per thread
 *    - Total for the kernel: __2 * K + 1__
 *
 * 4. Use Nsight Compute to measure:
 *    - Memory throughput achieved
 *    - Theoretical max memory throughput of RTX 4090
 *    - What percentage are we achieving?
 */
