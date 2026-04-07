#pragma once

/*
 * Kernel 6: Vectorized Memory Access
 *
 * Goal: Use 128-bit (float4) loads for GMEM and SMEM to reduce instruction count
 * Expected performance: ~40,000 GFLOPs on RTX 4090 (~1.1x over kernel 5)
 *
 * Problem with Kernel 5:
 * - GMEM loads use 32-bit (1 float) loads → 4 separate instructions per 4 floats
 * - SMEM loads of regM are strided (stride BK) → can't vectorize
 *
 * Solution: Two key changes
 *
 * 1. TRANSPOSE A in SMEM
 *    Kernel 5: As[m * BK + k]  → regM[i] = As[(threadRow*TM+i) * BK + dotIdx]
 *              Consecutive i → stride BK in memory (can't vectorize)
 *
 *    Kernel 6: As[k * BM + m]  → regM[i] = As[dotIdx * BM + threadRow*TM + i]
 *              Consecutive i → stride 1 in memory (CAN vectorize with float4!)
 *
 * 2. FLOAT4 for GMEM loads
 *    Each thread loads 4 floats at once using reinterpret_cast<float4*>
 *    One LDG.E.128 instruction instead of four LDG.E.32
 *
 * 3. FLOAT4 for C write-back
 *    Same idea — write 4 results at once per store instruction
 *
 * Bonus: float4 means each thread loads 4 elements, so 256 threads × 4 = 1024
 * elements = exactly the tile size. No strided loops needed!
 *
 * Parameters: same as kernel 5 (BM=128, BN=128, BK=8, TM=8, TN=8)
 *
 * Reference diagram: ../blog_reference/images/kernel_6_As_transpose.png
 * Reference implementation: ../SGEMM_CUDA/src/kernels/6_kernel_vectorize.cuh
 */

template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void sgemm_vectorized(int M, int N, int K, float alpha,
                                  float *A, float *B, float beta, float *C) {
    // ─── Step 1: Block and thread position (same as kernel 5) ────────

    const uint cRow = blockIdx.y;
    const uint cCol = blockIdx.x;

    const int threadCol = threadIdx.x % (BN / TN);
    const int threadRow = threadIdx.x / (BN / TN);

    // ─── Step 2: Shared memory (same layout sizes, but As is TRANSPOSED) ─

    __shared__ float As[BM * BK];  // Will be stored as As[k * BM + m] (transposed!)
    __shared__ float Bs[BK * BN];  // Same as kernel 5: Bs[k * BN + n]

    // ─── Step 3: Advance pointers (same as kernel 5) ────────────────

    A += cRow * BM * K;
    B += cCol * BN;
    C += cRow * BM * N + cCol * BN;

    // ─── Step 4: SMEM loading indices (changed for float4!) ──────────
    //
    // Each thread now loads 4 floats at once (float4 = 128 bits).
    // So we divide the column dimension by 4 when computing indices:
    //
    // For A (BM × BK = 128 × 8):
    //   Each row has BK=8 floats = 2 float4's
    //   innerRowA = threadIdx.x / (BK/4) = threadIdx.x / 2  → range 0..127 = BM rows
    //   innerColA = threadIdx.x % (BK/4) = threadIdx.x % 2  → which float4 group (0 or 1)
    //   256 threads × 4 floats = 1024 = BM*BK ✓  (no strided loop needed!)
    //
    // For B (BK × BN = 8 × 128):
    //   Each row has BN=128 floats = 32 float4's
    //   innerRowB = threadIdx.x / (BN/4) = threadIdx.x / 32  → range 0..7 = BK rows
    //   innerColB = threadIdx.x % (BN/4) = threadIdx.x % 32  → which float4 group (0..31)
    //   256 threads × 4 floats = 1024 = BK*BN ✓  (no strided loop needed!)

    // TODO: Calculate loading indices
    const uint innerRowA = threadIdx.x / (BK / 4);  // FIX THIS
    const uint innerColA = threadIdx.x % (BK / 4);  // FIX THIS
    const uint innerRowB = threadIdx.x / (BN / 4);  // FIX THIS
    const uint innerColB = threadIdx.x % (BN / 4);  // FIX THIS

    // ─── Step 5: Register storage (same as kernel 5) ─────────────────

    float threadResults[TM * TN] = {0.0};
    float regM[TM] = {0.0};
    float regN[TN] = {0.0};

    // ─── Step 6: Main loop ───────────────────────────────────────────

    for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
        // --- 6a: Load A into SMEM with TRANSPOSE + float4 ---
        //
        // We load 4 consecutive elements from A's row (along K dimension)
        // using float4, then scatter them into TRANSPOSED positions in As.
        //
        // GMEM read: A[innerRowA * K + innerColA * 4] → loads 4 consecutive K-values
        //   float4 tmp = reinterpret_cast<float4*>(&A[innerRowA * K + innerColA * 4])[0];
        //
        // SMEM write (transposed): As[k][m] = As[k * BM + m]
        //   As[(innerColA*4 + 0) * BM + innerRowA] = tmp.x;
        //   As[(innerColA*4 + 1) * BM + innerRowA] = tmp.y;
        //   As[(innerColA*4 + 2) * BM + innerRowA] = tmp.z;
        //   As[(innerColA*4 + 3) * BM + innerRowA] = tmp.w;
        //
        // Why transpose? So that later, loading regM with consecutive i:
        //   regM[i] = As[dotIdx * BM + threadRow*TM + i]  → contiguous addresses!
        float4 tmp = reinterpret_cast<float4*>(&A[innerRowA * K + innerColA * 4])[0];
        As[(innerColA * 4 + 0) * BM + innerRowA] = tmp.x;
        As[(innerColA * 4 + 1) * BM + innerRowA] = tmp.y;
        As[(innerColA * 4 + 2) * BM + innerRowA] = tmp.z;
        As[(innerColA * 4 + 3) * BM + innerRowA] = tmp.w;

        // FIX THIS: load float4 from A and store transposed into As


        // --- 6b: Load B into SMEM with float4 (no transpose needed) ---
        //
        // Bs layout is Bs[k * BN + n] — consecutive n is already contiguous.
        // Load 4 consecutive column values from B with float4:
        //
        //   reinterpret_cast<float4*>(&Bs[innerRowB * BN + innerColB * 4])[0] =
        //       reinterpret_cast<float4*>(&B[innerRowB * N + innerColB * 4])[0];

        // FIX THIS: vectorized load of B into Bs

        float4 tmpB = reinterpret_cast<float4*>(&B[innerRowB * N + innerColB * 4])[0];
        reinterpret_cast<float4*>(&Bs[innerRowB * BN + innerColB * 4])[0] = tmpB;


        __syncthreads();

        // Advance pointers
        A += BK;
        B += BK * N;

        // --- 6c: Compute outer products (same as kernel 5, but As is transposed) ---
        //
        // regM load changes due to transpose:
        //   Kernel 5: regM[i] = As[(threadRow*TM + i) * BK + dotIdx]  (row-major)
        //   Kernel 6: regM[i] = As[dotIdx * BM + threadRow*TM + i]    (transposed)
        //
        // regN load stays the same:
        //   regN[j] = Bs[dotIdx * BN + threadCol*TN + j]

        for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
            // Vectorized SMEM loads: 2× LDS.128 instead of 8× LDS.32 per register array
            for (uint i = 0; i < TM; i += 4) {
                float4 tmp = reinterpret_cast<float4*>(&As[dotIdx * BM + threadRow * TM + i])[0];
                regM[i] = tmp.x; regM[i+1] = tmp.y; regM[i+2] = tmp.z; regM[i+3] = tmp.w;
            }
            for (uint i = 0; i < TN; i += 4) {
                float4 tmp = reinterpret_cast<float4*>(&Bs[dotIdx * BN + threadCol * TN + i])[0];
                regN[i] = tmp.x; regN[i+1] = tmp.y; regN[i+2] = tmp.z; regN[i+3] = tmp.w;
            }
            // Outer product (same as kernel 5)
            for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
                for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
                    threadResults[resIdxM * TN + resIdxN] +=
                        regM[resIdxM] * regN[resIdxN];
                }
            }
        }
        __syncthreads();
    }

    // ─── Step 7: Vectorized write-back with float4 ──────────────────
    //
    // Instead of writing 1 float at a time, write 4 at once.
    // Inner loop steps by 4: resIdxN += 4
    //
    // For each row (resIdxM):
    //   For each group of 4 columns (resIdxN = 0, 4, ...):
    //     1. Load existing C values as float4 (for beta scaling)
    //     2. Apply alpha * result + beta * old
    //     3. Write back as float4
    //
    // Address: C[(threadRow*TM + resIdxM) * N + threadCol*TN + resIdxN]

    for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
        for (uint resIdxN = 0; resIdxN < TN; resIdxN += 4) {
            // FIX THIS: vectorized write-back
            //   1. Load C as float4
            //   2. Apply alpha/beta to each component
            //   3. Store back as float4
            float4 c = reinterpret_cast<float4*>(
                &C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN])[0];
            c.x = alpha * threadResults[resIdxM * TN + resIdxN + 0] + beta * c.x;
            c.y = alpha * threadResults[resIdxM * TN + resIdxN + 1] + beta * c.y;
            c.z = alpha * threadResults[resIdxM * TN + resIdxN + 2] + beta * c.z;
            c.w = alpha * threadResults[resIdxM * TN + resIdxN + 3] + beta * c.w;
            reinterpret_cast<float4*>(
                &C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN])[0] = c;
        }
    }
}

/*
 * Questions to think about:
 *
 * 1. Why does transposing A enable vectorized SMEM reads?
 *    - Before: As[m][k] → regM[i] reads As[(threadRow*TM+i)*BK + dotIdx]
 *      Consecutive i → addresses differ by BK (strided, can't float4)
 *    - After:  As[k][m] → regM[i] reads As[dotIdx*BM + threadRow*TM+i]
 *      Consecutive i → addresses differ by 1 (contiguous, CAN float4)
 *
 * 2. Why can we eliminate strided loops?
 *    - 256 threads × 4 floats/thread = 1024 elements = BM*BK = BK*BN
 *    - Perfect 1:1 mapping with float4 — each thread loads one float4
 *
 * 3. What alignment requirements does float4 have?
 *    - float4 loads require 16-byte alignment
 *    - A and B must be aligned (cudaMalloc guarantees at least 256-byte alignment)
 *    - SMEM arrays are naturally aligned at their start
 *
 * 4. The transpose during GMEM→SMEM load:
 *    - We read 4 consecutive K-values from A (coalesced in GMEM)
 *    - We write them to 4 different rows of As (scattered in SMEM)
 *    - SMEM scatter writes are fine — no coalescing requirement
 *    - The payoff comes later in the compute phase with contiguous regM loads
 *
 * 5. Can we vectorize the regM load too?
 *    - Yes! With TM=8 and contiguous layout, we could use float4 for regM:
 *      for (i = 0; i < TM; i += 4) {
 *          float4 tmp = reinterpret_cast<float4*>(&As[dotIdx*BM + threadRow*TM + i])[0];
 *          regM[i]=tmp.x; regM[i+1]=tmp.y; regM[i+2]=tmp.z; regM[i+3]=tmp.w;
 *      }
 *    - The reference implementation doesn't do this, but it's possible.
 */
