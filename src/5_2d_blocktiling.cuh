#pragma once

/*
 * Kernel 5: 2D Blocktiling
 *
 * Goal: Each thread computes a TM x TN tile of outputs (not just TM x 1)
 * Expected performance: ~35,000 GFLOPs on RTX 4090 (~1.8x over kernel 4)
 *
 * Problem with Kernel 4:
 * - Each thread computes TM x 1 outputs
 * - Per dot step: loads TM from As + 1 from Bs = (TM+1) SMEM reads for 2*TM FLOPs
 * - Arithmetic intensity = 2*TM / (TM+1) = 1.78 for TM=8
 * - ncu showed L1/SMEM throughput at 80% — we're bottlenecked on SMEM reads
 *
 * Solution: 2D Blocktiling
 * - Each thread computes TM x TN outputs (e.g., 8x8 = 64 elements!)
 * - Per dot step: load TM from As + TN from Bs into REGISTERS
 * - Then compute the OUTER PRODUCT: TM * TN = 64 FMAs from just 16 register values
 * - Arithmetic intensity = 2*TM*TN / (TM+TN) = 128/16 = 8.0 (4.5x better!)
 *
 * Key changes from Kernel 4:
 * 1. Thread position: threadCol = threadIdx.x % (BN/TN), threadRow = threadIdx.x / (BN/TN)
 *    Each thread now owns a TM x TN patch, not a TM x 1 strip.
 *
 * 2. Fewer threads per block: BM*BN / (TM*TN)
 *    e.g., 128*128 / (8*8) = 256 threads (vs 512 in kernel 4)
 *
 * 3. Each thread loads MULTIPLE elements into SMEM (strided loop)
 *    Since we have fewer threads but same-size SMEM tiles, each thread
 *    must load more than one element. This decouples BK from TM.
 *
 * 4. Inner loop does an OUTER PRODUCT instead of a dot product:
 *    regM[TM] x regN[TN] → accumulate into threadResults[TM*TN]
 *
 * Parameters: BM=128, BN=128, BK=8, TM=8, TN=8
 *   blockDim = 128*128 / (8*8) = 256 threads
 *   Grid = CEIL_DIV(N, 128) x CEIL_DIV(M, 128)
 *   SMEM: As[128*8] + Bs[8*128] = 8KB per tile pair
 *
 * Reference diagram: ../blog_reference/images/kernel_5_2D_blocktiling.png
 * Reference diagram: ../blog_reference/images/kernel_5_reg_blocking.png
 * Reference implementation: ../SGEMM_CUDA/src/kernels/5_kernel_2D_blocktiling.cuh
 */

template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void sgemm_2d_blocktiling(int M, int N, int K, float alpha,
                                      const float *A, const float *B, float beta,
                                      float *C) {
    // ─── Step 1: Block and thread position ───────────────────────────

    const uint cRow = blockIdx.y;
    const uint cCol = blockIdx.x;

    // Total threads in this block
    const uint numThreads = BM * BN / (TM * TN);

    // TODO: Map thread to a 2D position in the output tile
    //
    // In kernel 4: threadCol = tid % BN (one column per thread)
    // Now: each thread owns TN columns, so there are BN/TN "thread columns"
    //
    //   threadCol = threadIdx.x % (BN / TN)  → which TN-wide column group
    //   threadRow = threadIdx.x / (BN / TN)  → which TM-tall row group
    //
    // This thread computes outputs at:
    //   rows: [threadRow*TM .. threadRow*TM + TM - 1]
    //   cols: [threadCol*TN .. threadCol*TN + TN - 1]
    const int threadCol = threadIdx.x % (BN / TN);  // FIX THIS
    const int threadRow = threadIdx.x / (BN / TN);  // FIX THIS

    // ─── Step 2: Shared memory ───────────────────────────────────────

    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    // ─── Step 3: Advance pointers to this block's position ───────────

    A += cRow * BM * K;  // FIX THIS: advance to row cRow*BM
    B += cCol * BN * 1;  // FIX THIS: advance to col cCol*BN
    C += cRow * BM * N + cCol * BN * 1;  // FIX THIS: advance to (cRow*BM, cCol*BN)

    // ─── Step 4: SMEM loading indices ────────────────────────────────
    //
    // In kernel 4: each thread loaded exactly 1 element of As and Bs.
    // Now we have fewer threads (256) but same tile sizes:
    //   As = BM * BK = 128 * 8 = 1024 elements
    //   Bs = BK * BN = 8 * 128 = 1024 elements
    //
    // 256 threads loading 1024 elements → each thread loads 4 elements
    // via a STRIDED LOOP:
    //
    //   strideA = numThreads / BK = 256 / 8 = 32
    //   Thread loads rows: innerRowA, innerRowA + strideA, innerRowA + 2*strideA, ...
    //   until all BM rows are covered (128/32 = 4 iterations)
    //
    // Same idea for Bs:
    //   strideB = numThreads / BN = 256 / 128 = 2
    //   Thread loads rows: innerRowB, innerRowB + strideB, ...
    //   until all BK rows are covered (8/2 = 4 iterations)

    const uint innerRowA = threadIdx.x / BK;
    const uint innerColA = threadIdx.x % BK;
    const uint strideA = numThreads / BK;

    const uint innerRowB = threadIdx.x / BN;
    const uint innerColB = threadIdx.x % BN;
    const uint strideB = numThreads / BN;

    // ─── Step 5: Register storage ────────────────────────────────────

    // TM * TN results per thread (8*8 = 64 floats in registers!)
    float threadResults[TM * TN] = {0.0};
    // Register caches for the current dot-product step
    float regM[TM] = {0.0};
    float regN[TN] = {0.0};

    // ─── Step 6: Main loop over K dimension ──────────────────────────

    for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
        // --- 6a: Load tiles into SMEM (strided loop) ---
        //
        // TODO: Load As tile. Each thread loads multiple rows.
        // for (loadOffset = 0; loadOffset < BM; loadOffset += strideA)
        //     As[(innerRowA + loadOffset) * BK + innerColA] =
        //         A[(innerRowA + loadOffset) * K + innerColA]
        for (uint loadOffset = 0; loadOffset < BM; loadOffset += strideA) {
            // FIX THIS
            As[(innerRowA + loadOffset) * BK + innerColA] = A[(innerRowA + loadOffset) * K + innerColA];
        }

        // TODO: Load Bs tile. Each thread loads multiple rows.
        // for (loadOffset = 0; loadOffset < BK; loadOffset += strideB)
        //     Bs[(innerRowB + loadOffset) * BN + innerColB] =
        //         B[(innerRowB + loadOffset) * N + innerColB]
        for (uint loadOffset = 0; loadOffset < BK; loadOffset += strideB) {
            // FIX THIS
            Bs[(innerRowB + loadOffset) * BN + innerColB] = B[(innerRowB + loadOffset) * N + innerColB];
        }

        __syncthreads();

        // Advance pointers for next tile
        A += BK;  // FIX THIS
        B += BK * N;  // FIX THIS

        // --- 6b: Compute outer products from SMEM ---
        //
        // For each dotIdx in 0..BK-1:
        //   1. Load TM values from As column into regM
        //      regM[i] = As[(threadRow*TM + i) * BK + dotIdx]  for i in 0..TM-1
        //
        //   2. Load TN values from Bs row into regN
        //      regN[j] = Bs[dotIdx * BN + threadCol*TN + j]    for j in 0..TN-1
        //
        //   3. Outer product: regM[i] * regN[j] for all i,j
        //      threadResults[i * TN + j] += regM[i] * regN[j]
        //
        // This is the key insight: TM+TN = 16 SMEM loads produce TM*TN = 64 FMAs!

        for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
            // TODO: Load TM values from As into regM
            for (uint i = 0; i < TM; ++i) {
                regM[i] = As[(threadRow * TM + i) * BK + dotIdx];  // FIX THIS
            }
            // TODO: Load TN values from Bs into regN
            for (uint i = 0; i < TN; ++i) {
                regN[i] = Bs[dotIdx * BN + threadCol * TN + i];  // FIX THIS
            }
            // TODO: Outer product accumulation
            for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
                for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
                    // FIX THIS
                    threadResults[resIdxM * TN + resIdxN] += regM[resIdxM] * regN[resIdxN];
                }
            }
        }
        __syncthreads();
    }

    // ─── Step 7: Write TM x TN results to global memory ─────────────
    //
    // TODO: Write threadResults back to C
    // Output positions:
    //   row = threadRow * TM + resIdxM
    //   col = threadCol * TN + resIdxN
    //   C[row * N + col] = alpha * threadResults[...] + beta * C[row * N + col]

    for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
        for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
            // FIX THIS
            int row = threadRow * TM + resIdxM;
            int col = threadCol * TN + resIdxN;
            C[row * N + col] = alpha * threadResults[resIdxM * TN + resIdxN] + beta * C[row * N + col];
        }
    }
}

/*
 * Questions to think about:
 *
 * 1. Why does the strided SMEM loading pattern work?
 *    - 256 threads, As has 1024 elements
 *    - strideA = 256/8 = 32, so thread loads rows: innerRowA, +32, +64, +96
 *    - 4 iterations × 256 threads = 1024 loads = exactly fills As
 *
 * 2. Why is the outer product so powerful?
 *    - Kernel 4: 1 Bs load → TM = 8 FMAs (reuse in 1D)
 *    - Kernel 5: TM As loads + TN Bs loads → TM*TN = 64 FMAs (reuse in 2D)
 *    - Same total SMEM bandwidth, but 4x more compute per load
 *
 * 3. How many registers does this kernel use?
 *    - threadResults: TM*TN = 64 floats
 *    - regM: TM = 8 floats
 *    - regN: TN = 8 floats
 *    - Total: ~80 floats = 320 bytes in registers per thread
 *    - 256 threads × 80 regs = 20,480 registers per block
 *    - RTX 4090 has 65,536 registers per SM → 3 blocks can fit
 *
 * 4. How does the thread-to-output mapping differ?
 *    - Kernel 4: threadCol = tid % BN       (one column per thread)
 *    - Kernel 5: threadCol = tid % (BN/TN)  (TN columns per thread)
 *    - Each thread now owns a 2D patch, not a 1D strip
 *
 * 5. Why does decoupling BK from TM matter?
 *    - Kernel 4: required BM*BK == blockDim → BK = BN/TM (tied together)
 *    - Kernel 5: strided loading means any BK works (within SMEM limits)
 *    - More freedom to tune parameters independently
 */
