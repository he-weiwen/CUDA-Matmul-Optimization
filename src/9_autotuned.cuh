#pragma once

/*
 * Kernel 9: Autotuned SGEMM
 *
 * Based on kernel 6 (vectorized + transposed As) with two generalizations:
 *
 * 1. STRIDED float4 SMEM loading — decouples BK from thread count
 *    Kernel 6 required: numThreads * 4 = BM * BK (exact 1:1 float4 mapping)
 *    Now: each thread may load multiple float4's via a strided loop
 *
 * 2. WARP ITERATION — allows larger block tiles with fixed thread count
 *    Each thread can compute multiple TM×TN patches within the block tile.
 *    WM = TM * 16, WN = TN * 16 define "warp tile" dimensions.
 *    WMITER = BM / WM, WNITER = BN / WN iterations per thread.
 *
 * These changes decouple all 5 parameters (BM, BN, BK, TM, TN) so we can
 * sweep them independently to find the optimal configuration for RTX 4090.
 *
 * Reference: ../SGEMM_CUDA/src/kernels/9_kernel_autotuned.cuh
 */

#define CEIL_DIV_9(M, N) (((M) + (N) - 1) / (N))

template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void __launch_bounds__(256)
    sgemm_autotuned(int M, int N, int K, float alpha, float *A, float *B,
                    float beta, float *C) {
    const uint cRow = blockIdx.y;
    const uint cCol = blockIdx.x;

    // Warp tile dimensions: each warp covers WM × WN of the block tile
    // 16 comes from: 256 threads / 16 = 16 threads per warp-row-group
    // (a warp of 32 threads is laid out as threadRow × threadCol within WM × WN)
    constexpr int WM = TM * 16;
    constexpr int WN = TN * 16;
    // How many warp-tile iterations to cover the full block tile
    constexpr int WMITER = CEIL_DIV_9(BM, WM);
    constexpr int WNITER = CEIL_DIV_9(BN, WN);

    // Thread position within warp tile (not block tile!)
    const int threadCol = threadIdx.x % (WN / TN);  // 0..15
    const int threadRow = threadIdx.x / (WN / TN);  // 0..15

    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    A += cRow * BM * K;
    B += cCol * BN;
    C += cRow * BM * N + cCol * BN;

    // SMEM loading indices — strided float4 loads
    const uint innerRowA = threadIdx.x / (BK / 4);
    const uint innerColA = threadIdx.x % (BK / 4);
    constexpr uint rowStrideA = (256 * 4) / BK;
    const uint innerRowB = threadIdx.x / (BN / 4);
    const uint innerColB = threadIdx.x % (BN / 4);
    constexpr uint rowStrideB = 256 / (BN / 4);

    // Results: WMITER * WNITER patches of TM × TN each
    float threadResults[WMITER * WNITER * TM * TN] = {0.0};
    float regM[TM] = {0.0};
    float regN[TN] = {0.0};

    for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
        // Strided float4 load of A (transposed into SMEM)
        for (uint offset = 0; offset + rowStrideA <= BM; offset += rowStrideA) {
            float4 tmp = reinterpret_cast<float4 *>(
                &A[(innerRowA + offset) * K + innerColA * 4])[0];
            As[(innerColA * 4 + 0) * BM + innerRowA + offset] = tmp.x;
            As[(innerColA * 4 + 1) * BM + innerRowA + offset] = tmp.y;
            As[(innerColA * 4 + 2) * BM + innerRowA + offset] = tmp.z;
            As[(innerColA * 4 + 3) * BM + innerRowA + offset] = tmp.w;
        }
        // Strided float4 load of B
        for (uint offset = 0; offset + rowStrideB <= BK; offset += rowStrideB) {
            reinterpret_cast<float4 *>(
                &Bs[(innerRowB + offset) * BN + innerColB * 4])[0] =
                reinterpret_cast<float4 *>(
                    &B[(innerRowB + offset) * N + innerColB * 4])[0];
        }
        __syncthreads();

        // Warp iteration: each thread processes WMITER × WNITER patches
        for (uint wmIdx = 0; wmIdx < WMITER; ++wmIdx) {
            for (uint wnIdx = 0; wnIdx < WNITER; ++wnIdx) {
                for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
                    for (uint i = 0; i < TM; i += 4) {
                        float4 tmp = reinterpret_cast<float4 *>(
                            &As[dotIdx * BM + (wmIdx * WM) + threadRow * TM + i])[0];
                        regM[i] = tmp.x; regM[i+1] = tmp.y; regM[i+2] = tmp.z; regM[i+3] = tmp.w;
                    }
                    for (uint i = 0; i < TN; i += 4) {
                        float4 tmp = reinterpret_cast<float4 *>(
                            &Bs[dotIdx * BN + (wnIdx * WN) + threadCol * TN + i])[0];
                        regN[i] = tmp.x; regN[i+1] = tmp.y; regN[i+2] = tmp.z; regN[i+3] = tmp.w;
                    }
                    for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
                        for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
                            threadResults[(wmIdx * TM + resIdxM) * (WNITER * TN) +
                                          wnIdx * TN + resIdxN] +=
                                regM[resIdxM] * regN[resIdxN];
                        }
                    }
                }
            }
        }
        __syncthreads();
        A += BK;
        B += BK * N;
    }

    // Write back all patches
    for (uint wmIdx = 0; wmIdx < WMITER; ++wmIdx) {
        for (uint wnIdx = 0; wnIdx < WNITER; ++wnIdx) {
            float *C_interim = C + (wmIdx * WM * N) + (wnIdx * WN);
            for (uint resIdxM = 0; resIdxM < TM; resIdxM += 1) {
                for (uint resIdxN = 0; resIdxN < TN; resIdxN += 4) {
                    float4 tmp = reinterpret_cast<float4 *>(
                        &C_interim[(threadRow * TM + resIdxM) * N +
                                   threadCol * TN + resIdxN])[0];
                    const int i = (wmIdx * TM + resIdxM) * (WNITER * TN) +
                                  wnIdx * TN + resIdxN;
                    tmp.x = alpha * threadResults[i + 0] + beta * tmp.x;
                    tmp.y = alpha * threadResults[i + 1] + beta * tmp.y;
                    tmp.z = alpha * threadResults[i + 2] + beta * tmp.z;
                    tmp.w = alpha * threadResults[i + 3] + beta * tmp.w;
                    reinterpret_cast<float4 *>(
                        &C_interim[(threadRow * TM + resIdxM) * N +
                                   threadCol * TN + resIdxN])[0] = tmp;
                }
            }
        }
    }
}
