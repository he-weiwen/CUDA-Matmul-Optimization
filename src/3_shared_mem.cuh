#pragma once

/*
 * Kernel 3: Shared Memory Cache-Blocking
 *
 * Goal: Reduce global memory traffic by caching tiles in shared memory
 * Expected performance: ~3,000 GFLOPs (blog) → ~6,600+ GFLOPs (RTX 4090)
 *
 * Problem with Kernels 1 & 2:
 * - Each thread loads K elements from A and K elements from B
 * - Total GMEM loads: M * N * 2K (massive!)
 * - Data is re-loaded many times across different threads
 *
 * Solution: Shared Memory (SMEM)
 * - SMEM is ~10-20x faster than GMEM (~19 TB/s vs ~1 TB/s)
 * - Shared across all threads in a block
 * - Load a tile once into SMEM, reuse it many times
 *
 * Reference diagrams:
 * - Memory hierarchy: ../blog_reference/images/memory-hierarchy-in-gpus.png
 * - Cache blocking: ../blog_reference/images/cache-blocking.png
 *
 * Reference implementation: ../SGEMM_CUDA/src/kernels/3_kernel_shared_mem_blocking.cuh
 *
 * Algorithm:
 * 1. Each block computes a BM x BN tile of C
 * 2. Loop over K in chunks of BK:
 *    a. Cooperatively load BM x BK tile of A into SMEM
 *    b. Cooperatively load BK x BN tile of B into SMEM
 *    c. __syncthreads() - wait for all loads to complete
 *    d. Each thread computes partial dot product using SMEM data
 *    e. __syncthreads() - wait before loading next tile
 * 3. Write final results to C
 */

#define BM 32  // Block tile size in M dimension
#define BN 32  // Block tile size in N dimension
#define BK 32  // Block tile size in K dimension

__global__ void sgemm_shared_mem(int M, int N, int K, float alpha,
                                  const float *A, const float *B, float beta, float *C) {
    // Shared memory for tiles of A and B
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    // Block position in the output matrix
    const int bx = blockIdx.x;  // Column block index
    const int by = blockIdx.y;  // Row block index

    // Thread position within the block
    const int tx = threadIdx.x;  // Column within block (0..BN-1)
    const int ty = threadIdx.y;  // Row within block (0..BM-1)

    // Global position this thread is responsible for in C
    const int row = by * BM + ty;
    const int col = bx * BN + tx;

    // Accumulator for the dot product
    float sum = 0.0f;

    // Loop over tiles along the K dimension
    // Each iteration processes BK elements of the dot product
    for (int bk = 0; bk < K; bk += BK) {
        // TODO: Cooperatively load tile of A into shared memory
        // Each thread loads one element: As[ty][tx] = A[row][bk + tx]
        // But we need to handle bounds: what if (bk + tx) >= K or row >= M?
        //
        // A is M x K, stored row-major
        // We want to load A[row][bk + tx] where:
        //   - row = by * BM + ty
        //   - col_in_A = bk + tx
        //
        // Hint: Use 0.0f for out-of-bounds accesses
        if (row < M && (bk + tx) < K) {
            As[ty][tx] = A[row * K + (bk + tx)];
        } else {
            As[ty][tx] = 0.0f;
        }

        // TODO: Cooperatively load tile of B into shared memory
        // Each thread loads one element: Bs[ty][tx] = B[bk + ty][col]
        // B is K x N, stored row-major
        // We want to load B[bk + ty][col] where:
        //   - row_in_B = bk + ty
        //   - col = bx * BN + tx
        if ((bk + ty) < K && col < N) {
            Bs[ty][tx] = B[(bk + ty) * N + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }

        // CRITICAL: Wait for all threads to finish loading
        // Without this, some threads might read uninitialized SMEM!
        __syncthreads();

        // TODO: Compute partial dot product using shared memory
        // sum += As[ty][0] * Bs[0][tx] + As[ty][1] * Bs[1][tx] + ... + As[ty][BK-1] * Bs[BK-1][tx]
        for (int k = 0; k < BK; k++) {
            sum += As[ty][k] * Bs[k][tx];
        }

        // CRITICAL: Wait before loading next tile
        // Without this, fast threads might overwrite SMEM while slow threads still read!
        __syncthreads();
    }

    // Write result to global memory (with bounds check)
    if (row < M && col < N) {
        C[row * N + col] = alpha * sum + beta * C[row * N + col];
    }
}

/*
 * Questions:
 *
 * 1. Why do we need TWO __syncthreads() calls per iteration?
 *    - First one:
 *    - Second one:
 *
 * 2. How much SMEM does this kernel use per block?
 *    - As: BM * BK * sizeof(float) = ___ bytes
 *    - Bs: BK * BN * sizeof(float) = ___ bytes
 *    - Total: ___ bytes
 *    - RTX 4090 has 128KB shared memory per SM. How many blocks can run concurrently?
 *
 * 3. How many GMEM loads does this kernel perform?
 *    - Before (kernel 2): M * N * 2K loads
 *    - Now: ??? (hint: each element of A and B is loaded once per block that needs it)
 *
 * 4. What's the arithmetic intensity now?
 *    - FLOPs per block: BM * BN * 2K
 *    - Bytes loaded per block: (BM * K + K * BN) * sizeof(float)
 *    - Ratio: ???
 *
 * 5. Memory access pattern in SMEM:
 *    - As[ty][k]: all threads in a warp have same ty, different tx (but k is same for all)
 *                 → all threads access same row of As → broadcast, OK
 *    - Bs[k][tx]: all threads in a warp have same ty, different tx
 *                 → threads access Bs[k][0], Bs[k][1], ..., Bs[k][31] → consecutive, OK
 *    - Any bank conflicts? (32 banks, 4 bytes each)
 */
