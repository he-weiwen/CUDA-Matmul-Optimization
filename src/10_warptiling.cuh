#pragma once

#include <cute/layout.hpp>

/*
 * Kernel 10: Warptiling
 *
 * Goal: Explicit three-level tiling hierarchy for maximum ILP
 * Expected performance: >90% of cuBLAS
 *
 * The hierarchy:
 *   Block tile (BM×BN)         — maps to one thread block on one SM
 *     Warp tile (WM×WN)        — maps to one warp (32 threads)
 *       Thread tile (TM×TN)    — maps to one thread's register file
 *
 * Why this helps over kernel 9:
 * - Warps are explicitly mapped to non-overlapping SMEM regions
 * - Loads are separated from computes: load ALL regM/regN, THEN compute all outer products
 * - This lets the warp scheduler issue FMAs while loads are in flight → more ILP
 * - Each warp's SMEM accesses are localized → fewer bank conflicts across warps
 *
 * Parameters: BM=128, BN=128, BK=16, WM=64, WN=64, WNITER=2, TM=8, TN=4
 *   NUM_THREADS = (BM/WM) * (BN/WN) * 32 = 2 * 2 * 32 = 128
 *   WMITER = (WM*WN) / (32*TM*TN*WNITER) = 4096/2048 = 2
 *   WSUBM = WM/WMITER = 32,  WSUBN = WN/WNITER = 32
 *
 * Thread-in-warp mapping (within each 32×32 subtile):
 *   threadColInWarp = tid_in_warp % (WSUBN/TN) = tid % 8  → 0..7
 *   threadRowInWarp = tid_in_warp / (WSUBN/TN) = tid / 8  → 0..3
 *   Each thread: 8 rows × 4 cols = 32 outputs per subtile
 *   32 threads × 32 outputs = 1024 = 32×32 subtile ✓
 *
 * Reference diagram: ../blog_reference/images/kernel_10_warp_tiling.png
 * Reference implementation: ../SGEMM_CUDA/src/kernels/10_kernel_warptiling.cuh
 */

#define CEIL_DIV_10(M, N) (((M) + (N) - 1) / (N))
constexpr int WARPSIZE = 32;

// ─── Helper: Load GMEM → SMEM (same as kernel 9) ────────────────────────

template <const int BM, const int BN, const int BK,
          const int rowStrideA, const int rowStrideB>
__device__ void loadFromGmem(int N, int K,
                              const float *A, const float *B,
                              float *As, float *Bs,
                              int innerRowA, int innerColA,
                              int innerRowB, int innerColB) {
    using namespace cute;
    constexpr auto smemA = make_layout(make_shape(Int<BM>{}, Int<BK>{}),
                                      make_stride(Int<1>{}, Int<BM>{}));
    constexpr auto smemB = make_layout(make_shape(Int<BK>{}, Int<BN>{}),
                                      make_stride(Int<BN>{}, Int<1>{}));
    /*
      each thread load 4 contiguous elements to shared memory
        A: transposed
        B: straight
    */
    for (uint offset = 0; offset + rowStrideA <= BM; offset += rowStrideA) {
        const float4 tmp = reinterpret_cast<const float4 *>(
            &A[(innerRowA + offset) * K + innerColA * 4])[0];
        As[smemA(innerRowA + offset, innerColA * 4 + 0)] = tmp.x;
        As[smemA(innerRowA + offset, innerColA * 4 + 1)] = tmp.y;
        As[smemA(innerRowA + offset, innerColA * 4 + 2)] = tmp.z;
        As[smemA(innerRowA + offset, innerColA * 4 + 3)] = tmp.w;
    }
    // Strided float4 load of B
    for (uint offset = 0; offset + rowStrideB <= BK; offset += rowStrideB) {
        reinterpret_cast<float4 *>(
            &Bs[smemB(innerRowB + offset, innerColB * 4)])[0] =
            reinterpret_cast<const float4 *>(
                &B[(innerRowB + offset) * N + innerColB * 4])[0];
    }
}

// ─── Helper: Process SMEM → Registers → Outer Products ──────────────────

template <const int BM, const int BN, const int BK,
          const int WM, const int WN,
          const int WMITER, const int WNITER,
          const int WSUBM, const int WSUBN,
          const int TM, const int TN>
__device__ void processFromSmem(float *regM, float *regN, float *threadResults,
                                 const float *As, const float *Bs,
                                 const uint warpRow, const uint warpCol,
                                 const uint threadRowInWarp,
                                 const uint threadColInWarp) {
    using namespace cute;
    constexpr auto smemA = make_layout(make_shape(Int<BM>{}, Int<BK>{}),
                                      make_stride(Int<1>{}, Int<BM>{}));
    constexpr auto smemB = make_layout(make_shape(Int<BK>{}, Int<BN>{}),
                                      make_stride(Int<BN>{}, Int<1>{}));
    // (subtile, element) -> register index; packed within each thread.
    constexpr auto regA = make_layout(make_shape(Int<WMITER>{}, Int<TM>{}),
                                     make_stride(Int<TM>{}, Int<1>{}));
    constexpr auto regB = make_layout(make_shape(Int<WNITER>{}, Int<TN>{}),
                                     make_stride(Int<TN>{}, Int<1>{}));
    constexpr auto accum = make_layout(make_shape(Int<WMITER * TM>{}, Int<WNITER * TN>{}),
                                      make_stride(Int<WNITER * TN>{}, Int<1>{}));
    // The same packed register coordinates map to separated matrix patches.
    constexpr auto rows = make_layout(make_shape(Int<WMITER>{}, Int<TM>{}),
                                     make_stride(Int<WSUBM>{}, Int<1>{}));
    constexpr auto cols = make_layout(make_shape(Int<WNITER>{}, Int<TN>{}),
                                     make_stride(Int<WSUBN>{}, Int<1>{}));
    const uint rowBase = warpRow * WM + threadRowInWarp * TM;
    const uint colBase = warpCol * WN + threadColInWarp * TN;
    for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
        // ── Phase 1: Load ALL regM values for all WMITER subtile rows ──
        // This front-loads the SMEM reads so FMAs can overlap with loads
        for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
            for (uint i = 0; i < TM; ++i) {
                regM[regA(wSubRowIdx, i)] = As[smemA(rowBase + rows(wSubRowIdx, i), dotIdx)];
            }
        }

        // ── Phase 2: Load ALL regN values for all WNITER subtile cols ──
        for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
            for (uint i = 0; i < TN; ++i) {
                regN[regB(wSubColIdx, i)] = Bs[smemB(dotIdx, colBase + cols(wSubColIdx, i))];
            }
        }

        // ── Phase 3: Compute ALL outer products ──
        // regM and regN are fully populated → pure FMA work
        for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
            for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
                for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
                    for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
                        const auto m = regA(wSubRowIdx, resIdxM);
                        const auto n = regB(wSubColIdx, resIdxN);
                        threadResults[accum(m, n)] += regM[m] * regN[n];
                    }
                }
            }
        }
    }
}

// ─── Main kernel ─────────────────────────────────────────────────────────

template <const int BM, const int BN, const int BK,
          const int WM, const int WN, const int WNITER,
          const int TM, const int TN, const int NUM_THREADS>
__global__ void __launch_bounds__(NUM_THREADS)
    sgemm_warptiling(int M, int N, int K, float alpha, float *A, float *B,
                      float beta, float *C) {
    const uint cRow = blockIdx.y;
    const uint cCol = blockIdx.x;

    // ─── Warp-level position ─────────────────────────────────────────
    //
    // Which warp am I in? Where does my warp sit in the block tile?
    //
    //   Block tile (BM×BN) is divided into (BM/WM) × (BN/WN) warp tiles
    //   e.g., 128/64 × 128/64 = 2×2 = 4 warps
    //
    //   warpIdx = threadIdx.x / 32            → which warp (0..3)
    //   warpCol = warpIdx % (BN / WN)         → warp's column position (0..1)
    //   warpRow = warpIdx / (BN / WN)         → warp's row position (0..1)

    const uint warpIdx = threadIdx.x / 32;    // which warp (0..3)
    const uint warpCol = warpIdx % (BN / WN); // warp's column position (0..1)
    const uint warpRow = warpIdx / (BN / WN); // warp's row position  (0..1)

    // ─── Warp subtile dimensions ─────────────────────────────────────
    //
    // Each warp tile (WM×WN) is further divided into WMITER × WNITER subtiles.
    // WNITER is a template parameter; WMITER is derived:
    //
    //   Total outputs per warp tile = WM * WN
    //   Outputs per subtile pass = 32 threads × TM × TN (per thread)
    //   Subtile passes needed = WM*WN / (32*TM*TN)
    //   = WMITER * WNITER
    //   → WMITER = (WM*WN) / (32*TM*TN*WNITER)

    constexpr uint WMITER = (WM * WN) / (WARPSIZE * TM * TN * WNITER);
    constexpr uint WSUBM = WM / WMITER;  // subtile height (e.g., 64/2 = 32)
    constexpr uint WSUBN = WN / WNITER;  // subtile width  (e.g., 64/2 = 32)

    // ─── Thread position within warp subtile ─────────────────────────
    //
    // Within each WSUBM×WSUBN subtile, 32 threads are mapped as:
    //   threadColInWarp = tid_in_warp % (WSUBN / TN)
    //   threadRowInWarp = tid_in_warp / (WSUBN / TN)

    const uint threadIdxInWarp = threadIdx.x % 32;
    const uint threadColInWarp = threadIdxInWarp % (WSUBN / TN);
    const uint threadRowInWarp = threadIdxInWarp / (WSUBN / TN);

    // ─── Shared memory ───────────────────────────────────────────────

    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    // ─── Advance pointers ────────────────────────────────────────────
    //
    // A and B: advance to block's position (same as before)
    // C: advance to WARP's output region (not just block's!)

    A += cRow * BM * K;
    B += cCol * BN;
    C += (cRow * BM + warpRow * WM) * N + cCol * BN + warpCol * WN;

    // ─── SMEM loading indices (same strided float4 pattern) ──────────

    const uint innerRowA = threadIdx.x / (BK / 4);
    const uint innerColA = threadIdx.x % (BK / 4);
    constexpr uint rowStrideA = (NUM_THREADS * 4) / BK;
    const uint innerRowB = threadIdx.x / (BN / 4);
    const uint innerColB = threadIdx.x % (BN / 4);
    constexpr uint rowStrideB = NUM_THREADS / (BN / 4);

    // ─── Register storage ────────────────────────────────────────────
    //
    // Note: regM and regN are sized for ALL WMITER/WNITER subtiles!
    // This is what enables the load-compute separation.

    float threadResults[WMITER * TM * WNITER * TN] = {0.0};
    float regM[WMITER * TM] = {0.0};  // e.g., 2*8 = 16 floats
    float regN[WNITER * TN] = {0.0};  // e.g., 2*4 = 8 floats

    // ─── Main loop ───────────────────────────────────────────────────

    for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
        loadFromGmem<BM, BN, BK, rowStrideA, rowStrideB>(
            N, K, A, B, As, Bs, innerRowA, innerColA, innerRowB, innerColB);
        __syncthreads();

        processFromSmem<BM, BN, BK, WM, WN, WMITER, WNITER, WSUBM, WSUBN, TM, TN>(
            regM, regN, threadResults, As, Bs,
            warpRow, warpCol, threadRowInWarp, threadColInWarp);

        A += BK;
        B += BK * N;
        __syncthreads();
    }

    // Each thread owns four TM x TN patches in its warp tile.
    using namespace cute;
    constexpr auto accum = make_layout(
        make_shape(Int<WMITER>{}, Int<WNITER>{}, Int<TM>{}, Int<TN>{}),
        make_stride(Int<TM * WNITER * TN>{}, Int<TN>{}, Int<WNITER * TN>{}, Int<1>{}));
    const auto output = make_layout(make_shape(Int<WM>{}, Int<WN>{}),
                                    make_stride(N, Int<1>{}));
    constexpr auto rows = make_layout(make_shape(Int<WMITER>{}, Int<TM>{}),
                                     make_stride(Int<WSUBM>{}, Int<1>{}));
    constexpr auto cols = make_layout(make_shape(Int<WNITER>{}, Int<TN>{}),
                                     make_stride(Int<WSUBN>{}, Int<1>{}));
    for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
        for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {

            for (uint resIdxM = 0; resIdxM < TM; resIdxM += 1) {
                for (uint resIdxN = 0; resIdxN < TN; resIdxN += 4) {
                    const auto outIdx = output(
                        threadRowInWarp * TM + rows(wSubRowIdx, resIdxM),
                        threadColInWarp * TN + cols(wSubColIdx, resIdxN));
                    const auto resultIdx = accum(wSubRowIdx, wSubColIdx, resIdxM, resIdxN);
                    float4 result = make_float4(
                        alpha * threadResults[resultIdx],
                        alpha * threadResults[resultIdx + 1],
                        alpha * threadResults[resultIdx + 2],
                        alpha * threadResults[resultIdx + 3]);
                    // When beta is zero, C need not contain initialized values.
                    if (beta != 0.0f) {
                        const float4 previous =
                            *reinterpret_cast<const float4 *>(&C[outIdx]);
                        result.x += beta * previous.x;
                        result.y += beta * previous.y;
                        result.z += beta * previous.z;
                        result.w += beta * previous.w;
                    }
                    *reinterpret_cast<float4 *>(&C[outIdx]) = result;
                }
            }
        }
    }
}

/*
 * Questions to think about:
 *
 * 1. Why does explicit warp mapping help?
 *    - In kernel 9, threads are mapped globally — warps end up interleaved
 *    - In kernel 10, each warp owns a contiguous SMEM region
 *    - Less cross-warp bank conflict, better data locality per warp
 *
 * 2. Why separate load and compute phases?
 *    - Phase 1-2: issue SMEM loads (latency ~20 cycles)
 *    - Phase 3: issue FMAs using loaded values
 *    - The warp scheduler can overlap loads with FMAs from different iterations
 *    - More independent instructions in flight → better ILP
 *
 * 3. Why TM=8, TN=4 (asymmetric)?
 *    - 32 threads in a warp: threadRowInWarp = 0..3, threadColInWarp = 0..7
 *    - 4 rows × 8 cols of threads = 32 ✓
 *    - TM=8 per row-thread, TN=4 per col-thread → 32×32 subtile
 *    - The asymmetry matches the warp's 4×8 thread layout
 *
 * 4. Register budget:
 *    - threadResults: WMITER*TM*WNITER*TN = 2*8*2*4 = 128 floats
 *    - regM: WMITER*TM = 16 floats
 *    - regN: WNITER*TN = 8 floats
 *    - Total: ~152 floats ≈ 152 registers per thread
 *    - 128 threads × 152 regs = 19,456 < 65,536 per SM → ~3 blocks per SM
 *
 * 5. Why NUM_THREADS=128 (only 4 warps)?
 *    - Fewer threads = more registers per thread = more work per thread
 *    - Same tradeoff cuBLAS makes (202 regs/thread, low occupancy)
 *    - 4 warps × 4 schedulers = 1 warp per scheduler (minimal, but each fully utilized)
 */
