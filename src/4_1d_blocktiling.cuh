#pragma once

/*
 * Kernel 4: 1D Blocktiling
 *
 * Goal: Each thread computes TM output elements instead of just 1
 * Expected performance: ~18,000 GFLOPs on RTX 4090 (~3x over kernel 3)
 *
 * Problem with Kernel 3:
 * - Each thread computes 1 element of C
 * - Loads 1 value from As and 1 from Bs per multiply
 * - Arithmetic intensity = 2 FLOPs / 2 SMEM reads = 1.0
 *   (very low — SMEM bandwidth becomes the bottleneck)
 *
 * Solution: 1D Blocktiling
 * - Each thread computes TM  elements in a COLUMN of the output tile
 * - For each dot product step:
 *     1. Load Bs[dotIdx][threadCol] ONCE into a register
 *     2. Reuse it across TM multiplies with TM values from As
 * - Arithmetic intensity =2*TM FLOPs / (TM+1) SMEM reads ≈ 2.0 for large TM
 *
 * Key change: 1D thread block instead of 2D!
 * - blockDim = (BM * BN) / TM  (e.g., 64*64/8 = 512 threads)
 * - threadCol = threadIdx.x % BN   (column position in output tile)
 * - threadRow = threadIdx.x / BN   (row-group; each group owns TM rows)
 *
 * Reference diagram: ../blog_reference/images/kernel_4_1D_blocktiling.png
 * Reference implementation: ../SGEMM_CUDA/src/kernels/4_kernel_1D_blocktiling.cuh
 *
 * Tile layout (BM=64, BN=64, TM=8):
 *
 *        BN=64 columns
 *       |<------------>|
 *   --- +──────────────+
 *    T  | thread(r,c)  |  Each thread computes a vertical
 *    M  | computes     |  strip of TM=8 elements at column
 *    =  | this column  |  threadCol, rows threadRow*TM to
 *    8  | strip        |  threadRow*TM + TM - 1
 *   --- +──────────────+
 *
 *   Number of row-groups = BM / TM = 64/8 = 8
 *   Threads per row-group = BN = 64
 *   Total threads = 8 * 64 = 512
 */
// <d0, d1, d2> -> <d0
template <const int BM, const int BN, const int BK, const int TM>
__global__ void sgemm_1d_blocktiling(
    int M, int N, int K, 
    float alpha,
    const float *A, const float *B, 
    float beta,
    float *C) {
    // ─── Step 1: Block and thread position ───────────────────────────

    // Which block-tile of C are we computing?
    const uint cRow = blockIdx.y;  // block row in output matrix
    const uint cCol = blockIdx.x;  // block col in output matrix

    // TODO: Map this thread to a position in the output tile
    // With a 1D thread block, we derive row-group and column from threadIdx.x:
    //   threadCol = threadIdx.x % BN   → column within the tile (0..BN-1)
    //   threadRow = threadIdx.x / BN   → row-group index (0..BM/TM-1)
    // This thread computes TM elements: rows [threadRow*TM .. threadRow*TM+TM-1]
    // at column threadCol.
    const int tid = threadIdx.x;
    const int threadCol = tid % BN;  // FIX THIS
    const int threadRowGrp = tid / BN;  // FIX THIS

    // ─── Step 2: Shared memory ───────────────────────────────────────

    // Allocate SMEM for one tile of A (BM x BK) and one tile of B (BK x BN)
    // Using 1D arrays (row-major layout)
    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    // ─── Step 3: Advance pointers to this block's starting position ──

    // TODO: Move A to the start of this block's row-strip
    // A covers rows [cRow*BM .. cRow*BM+BM-1], starting at column 0
    // So advance A by cRow * BM * K elements
    A += (cRow * BM) * K;  // FIX THIS

    // TODO: Move B to the start of this block's column-strip
    // B covers columns [cCol*BN .. cCol*BN+BN-1], starting at row 0
    // So advance B by cCol * BN elements
    B += cCol * BN * 1;  // FIX THIS

    // TODO: Move C to the top-left corner of this block's output tile
    // C[cRow*BM][cCol*BN] = C[cRow*BM*N + cCol*BN]
    C += cRow * BM * N + cCol * BN;  // FIX THIS

    // ─── Step 4: SMEM loading indices ────────────────────────────────

    // Each thread loads exactly 1 element of As and 1 element of Bs.
    // This requires: BM * BK == blockDim.x AND BK * BN == blockDim.x
    // (With BM=BN=64, BK=8: 64*8 = 512 = blockDim.x)
    //
    // For As (BM x BK, row-major): linearize threadIdx.x into (row, col)
    //   innerRowA = threadIdx.x / BK   → which row (0..BM-1)
    //   innerColA = threadIdx.x % BK   → which col (0..BK-1)
    //
    // For Bs (BK x BN, row-major): linearize threadIdx.x into (row, col)
    //   innerRowB = threadIdx.x / BN   → which row (0..BK-1)
    //   innerColB = threadIdx.x % BN   → which col (0..BN-1)

    // TODO: Calculate SMEM loading indices
    const uint innerColA = threadIdx.x % BK;  // FIX THIS
    const uint innerRowA = threadIdx.x / BK;  // FIX THIS
    const uint innerColB = threadIdx.x % BN;  // FIX THIS
    const uint innerRowB = threadIdx.x / BN;  // FIX THIS

    // ─── Step 5: Register storage for results ────────────────────────

    // Each thread accumulates TM results in registers (fastest storage!)
    float threadResults[TM] = {0.0};

    // ─── Step 6: Main loop over K dimension ──────────────────────────

    for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
        // --- 6a: Load one tile of A and B into SMEM ---
        //
        // TODO: Load As[innerRowA][innerColA] from A[innerRowA][innerColA]
        // Remember: A pointer was already advanced to the block's row-strip,
        // and we advance it by BK each iteration (see step 6c).
        // As is stored as 1D: As[innerRowA * BK + innerColA]
        // A element is at: A[innerRowA * K + innerColA]
        As[innerRowA * BK + innerColA] = A[innerRowA * K + innerColA];  // FIX THIS

        // TODO: Load Bs[innerRowB][innerColB] from B[innerRowB][innerColB]
        // Bs is stored as 1D: Bs[innerRowB * BN + innerColB]
        // B element is at: B[innerRowB * N + innerColB]
        Bs[innerRowB * BN + innerColB] = B[innerRowB * N + innerColB];  // FIX THIS

        __syncthreads();

        // --- 6b: Advance A and B pointers for next tile ---
        // TODO: A moves right by BK columns, B moves down by BK rows
        A += BK * 1;  // FIX THIS
        B += BK * N;  // FIX THIS

        // --- 6c: Compute partial dot products from SMEM ---
        //
        // For each position along the K-tile (dotIdx = 0..BK-1):
        //   1. Load Bs[dotIdx][threadCol] into a register (reused TM times!)
        //   2. For each of the TM rows this thread owns:
        //      threadResults[i] += As[threadRow*TM + i][dotIdx] * tmpB
        //
        // Why is dotIdx the OUTER loop?
        //   → tmpB = Bs[dotIdx][threadCol] is loaded ONCE and reused TM times
        //   → This is the key optimization: register reuse!
        //
        // TODO: Implement the nested loop
        for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
            // TODO: Cache Bs value in a register
            float tmpB = Bs[dotIdx * BN + threadCol * 1];  // FIX THIS: load from Bs

            for (uint resIdx = 0; resIdx < TM; ++resIdx) {
                // TODO: Accumulate: As[(threadRow*TM + resIdx)][dotIdx] * tmpB
                threadResults[resIdx] += As[(threadRowGrp * TM + resIdx) * BK + dotIdx] * tmpB;  // FIX THIS
            }
        }
        __syncthreads();
    }

    // ─── Step 7: Write results to global memory ─────────────────────

    // TODO: Write TM results back to C
    // This thread's outputs are at:
    //   C[(threadRow*TM + resIdx)][threadCol]
    //   = C[(threadRow*TM + resIdx) * N + threadCol]
    //
    // Apply alpha/beta scaling as usual.
    for (uint resIdx = 0; resIdx < TM; ++resIdx) {
        // FIX THIS: write threadResults[resIdx] to the correct C location
        C[(threadRowGrp * TM + resIdx) * N + threadCol] =
            alpha * threadResults[resIdx] +
            beta * C[(threadRowGrp * TM + resIdx) * N + threadCol];
    }
}

/*
 * Questions to think about:
 *
 * 1. Why is blockDim now 1D instead of 2D?
 *    - In kernel 3: blockDim = (BN, BM) = (32, 32) = 1024 threads, each computes 1 element
 *    - Here: blockDim = BM*BN/TM = 64*64/8 = 512 threads, each computes TM=8 elements
 *    - Same output tile size (64x64 = 4096 elements), fewer threads, more work per thread
 *
 * 2. Why does dotIdx (K loop) go on the OUTSIDE?
 *    - So we load tmpB = Bs[dotIdx][threadCol] ONCE per dotIdx
 *    - Then reuse it for all TM rows
 *    - If loops were swapped, we'd reload tmpB TM times (wasteful)
 *
 * 3. How does SMEM loading work with 512 threads and BM*BK = 512 elements?
 *    - Perfect 1:1 mapping — each thread loads exactly 1 element
 *    - No thread is idle, no element is loaded twice
 *    - Constraint: BM * BK must equal blockDim.x
 *
 * 4. What is the arithmetic intensity now?
 *    - Per dot product step: TM multiplies + TM adds = 2*TM FLOPs
 *    - SMEM loads: TM values from As + 1 value from Bs = (TM+1) loads
 *    - Intensity = 2*TM / (TM+1) ≈ 1.78 for TM=8 (vs 1.0 in kernel 3)
 *
 * 5. Are GMEM loads during SMEM population coalesced?
 *    - A: innerColA = threadIdx.x % BK. Threads 0-31 get innerColA = 0,1,...,7,0,1,...
 *         → Stride-BK pattern, NOT perfectly coalesced (BK < 32)
 *    - B: innerColB = threadIdx.x % BN. Threads 0-31 get innerColB = 0,1,...,31
 *         → Consecutive columns → COALESCED
 *    - This is addressed in kernel 6 with vectorized loads.
 *
 * 6. Why do we need bounds checking for non-square / non-aligned matrices?
 *    (Omitted here for clarity, but needed in production)
 */
