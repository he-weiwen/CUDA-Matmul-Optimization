#pragma once

#include <cuda_runtime.h>

// Include your kernel implementations
#include "1_naive.cuh"
#include "2_coalesced.cuh"
#include "3_shared_mem.cuh"
#include "4_1d_blocktiling.cuh"
#include "5_2d_blocktiling.cuh"
#include "6_vectorized.cuh"
#include "9_autotuned.cuh"
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
            dim3 blockDim(SM_BN, SM_BM);  // (32, 32)
            dim3 gridDim(CEIL_DIV(N, SM_BN), CEIL_DIV(M, SM_BM));
            sgemm_shared_mem<<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
            break;
        }

        case 4: {
            // Kernel 4: 1D Blocktiling
            // BM=64, BN=64, BK=8, TM=8
            // Each thread computes TM=8 outputs
            // blockDim = BM * BN / TM = 64 * 64 / 8 = 512 threads
            const int BM4 = 64, BN4 = 64, BK4 = 8, TM4 = 8;
            dim3 blockDim4((BM4 * BN4) / TM4);  // 512 threads
            dim3 gridDim4(CEIL_DIV(N, BN4), CEIL_DIV(M, BM4));
            sgemm_1d_blocktiling<BM4, BN4, BK4, TM4><<<gridDim4, blockDim4>>>(M, N, K, alpha, A, B, beta, C);
            break;
        }

        case 5: {
            // Kernel 5: 2D Blocktiling
            // BM=128, BN=128, BK=8, TM=8, TN=8
            // Each thread computes TM*TN = 64 outputs
            // blockDim = BM*BN / (TM*TN) = 128*128 / 64 = 256 threads
            const int BM5 = 128, BN5 = 128, BK5 = 8, TM5 = 8, TN5 = 8;
            dim3 blockDim5((BM5 * BN5) / (TM5 * TN5));  // 256 threads
            dim3 gridDim5(CEIL_DIV(N, BN5), CEIL_DIV(M, BM5));
            sgemm_2d_blocktiling<BM5, BN5, BK5, TM5, TN5><<<gridDim5, blockDim5>>>(M, N, K, alpha, A, B, beta, C);
            break;
        }

        case 6: {
            // Kernel 6: Vectorized memory access
            // Same tile sizes as kernel 5, but with float4 loads and transposed As
            const int BM6 = 128, BN6 = 128, BK6 = 8, TM6 = 8, TN6 = 8;
            dim3 blockDim6((BM6 * BN6) / (TM6 * TN6));  // 256 threads
            dim3 gridDim6(CEIL_DIV(N, BN6), CEIL_DIV(M, BM6));
            sgemm_vectorized<BM6, BN6, BK6, TM6, TN6><<<gridDim6, blockDim6>>>(M, N, K, alpha, A, B, beta, C);
            break;
        }

        case 9: {
            // Kernel 9: Autotuned — best params from sweep on RTX 4090
            const int BM9 = 128, BN9 = 128, BK9 = 16, TM9 = 8, TN9 = 8;
            dim3 blockDim9(256);
            dim3 gridDim9(CEIL_DIV(N, BN9), CEIL_DIV(M, BM9));
            sgemm_autotuned<BM9, BN9, BK9, TM9, TN9><<<gridDim9, blockDim9>>>(M, N, K, alpha, A, B, beta, C);
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
