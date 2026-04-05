#pragma once

#include <cuda_runtime.h>

// Include your kernel implementations
#include "1_naive.cuh"
#include "2_coalesced.cuh"
#include "3_shared_mem.cuh"
// Uncomment as you implement each kernel:
// #include "4_1d_blocktiling.cuh"
// #include "5_2d_blocktiling.cuh"
// #include "6_vectorized.cuh"
// #include "9_autotuned.cuh"
// #include "10_warptiling.cuh"

#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))

void run_kernel(int kernel_num, int M, int N, int K,
                float alpha, float *A, float *B, float beta, float *C) {
    switch (kernel_num) {
        case 1: {
            // Kernel 1: Naive implementation
            dim3 blockDim(32, 32);
            dim3 gridDim(CEIL_DIV(N, 32), CEIL_DIV(M, 32));
            sgemm_naive<<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
            break;
        }

        case 2: {
            // Kernel 2: Global memory coalescing (correct mapping)
            dim3 blockDim(BLOCKSIZE, BLOCKSIZE);
            dim3 gridDim(CEIL_DIV(N, BLOCKSIZE), CEIL_DIV(M, BLOCKSIZE));
            sgemm_coalesced<<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
            break;
        }

        case 20: {
            // Kernel 20: NON-coalesced version for comparison
            dim3 blockDim(BLOCKSIZE, BLOCKSIZE);
            dim3 gridDim(CEIL_DIV(M, BLOCKSIZE), CEIL_DIV(N, BLOCKSIZE));
            sgemm_NOT_coalesced<<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
            break;
        }

        case 3: {
            // Kernel 3: Shared memory cache-blocking
            dim3 blockDim(BN, BM);  // (32, 32)
            dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
            sgemm_shared_mem<<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
            break;
        }

        // Uncomment and implement as you progress:
        /*
        case 2: {
            // Kernel 2: Global memory coalescing
            dim3 blockDim(32, 32);
            dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
            sgemm_coalesced<<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
            break;
        }

        case 3: {
            // Kernel 3: Shared memory caching
            const int BM = 32, BN = 32, BK = 32;
            dim3 blockDim(BN, BM);
            dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
            sgemm_shared<<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
            break;
        }

        case 4: {
            // Kernel 4: 1D blocktiling
            const int BM = 64, BN = 64, BK = 8, TM = 8;
            dim3 blockDim((BM * BN) / TM);
            dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
            sgemm_1d_blocktiling<BM, BN, BK, TM><<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
            break;
        }

        case 5: {
            // Kernel 5: 2D blocktiling
            const int BM = 128, BN = 128, BK = 8, TM = 8, TN = 8;
            dim3 blockDim((BM / TM) * (BN / TN));
            dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
            sgemm_2d_blocktiling<BM, BN, BK, TM, TN><<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
            break;
        }

        case 6: {
            // Kernel 6: Vectorized memory access
            const int BM = 128, BN = 128, BK = 8, TM = 8, TN = 8;
            dim3 blockDim((BM / TM) * (BN / TN));
            dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
            sgemm_vectorized<BM, BN, BK, TM, TN><<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
            break;
        }

        case 9: {
            // Kernel 9: Autotuned parameters
            // TODO: Fill in your optimal parameters after tuning
            const int BM = 128, BN = 128, BK = 16, TM = 8, TN = 8;
            dim3 blockDim((BM / TM) * (BN / TN));
            dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
            sgemm_autotuned<BM, BN, BK, TM, TN><<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
            break;
        }

        case 10: {
            // Kernel 10: Warptiling
            // TODO: Fill in warp tiling parameters
            break;
        }
        */

        default:
            printf("Kernel %d not implemented yet!\n", kernel_num);
            break;
    }
}
