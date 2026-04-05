#pragma once

/*
 * Kernel 2: Global Memory Coalescing
 *
 * Goal: Fix the memory access pattern so warps access consecutive addresses
 * Expected performance: ~2,000 GFLOPs (blog) → ~4,400+ GFLOPs (RTX 4090)
 *
 * Problem with Kernel 1:
 * - Threads 0-31 (a warp) have same row, different cols
 * - A[row * K + k]: all threads load SAME address (wasteful broadcast)
 * - B[k * N + col]: threads load consecutive addresses (good!)
 *
 * The fix: Swap row/col mapping so threads in a warp have:
 * - Different rows, same col → A becomes coalesced
 * - But wait... then B becomes non-coalesced!
 *
 * Actually, let's think more carefully...
 *
 * Reference diagrams:
 * - Problem: ../blog_reference/images/Naive_kernel_mem_coalescing.png
 * - Solution: ../blog_reference/images/Naive_kernel_improved_access.png
 * - Warp mapping: ../blog_reference/images/threadId_to_warp_mapping.png
 *
 * Reference implementation: ../SGEMM_CUDA/src/kernels/2_kernel_global_mem_coalesce.cuh
 *
 * Key insight: The issue isn't row vs col assignment, it's HOW we map
 * threadIdx to matrix positions. We need consecutive threads to access
 * consecutive memory, which means they should differ in the INNERMOST
 * dimension of the matrix layout.
 *
 * For row-major matrices:
 * - A[row][k] stored as A[row * K + k] - consecutive in k
 * - B[k][col] stored as B[k * N + col] - consecutive in col
 * - C[row][col] stored as C[row * N + col] - consecutive in col
 *
 * So for coalesced access:
 * - A: threads should differ in k (but we iterate over k, so this is tricky)
 * - B: threads should differ in col ✓ (already good in kernel 1!)
 * - C: threads should differ in col ✓
 *
 * The REAL issue in kernel 1: We used a 2D block where threadIdx.x maps to col.
 * But warp formation goes: thread 0=(0,0), thread 1=(1,0), ..., thread 31=(31,0)
 * So threads 0-31 have threadIdx.x = 0,1,...,31 and threadIdx.y = 0
 * This means they have SAME row, DIFFERENT cols → B is coalesced, A is broadcast
 *
 * To improve: Use a 1D block or remap the indexing!
 */

// Block size - using 32 to match warp size for cleaner analysis
#define BLOCKSIZE 32

__global__ void sgemm_coalesced(int M, int N, int K, float alpha,
                                const float *A, const float *B, float beta, float *C) {
    // Key change: Swap the mapping!
    // In kernel 1: row from blockIdx.y/threadIdx.y, col from blockIdx.x/threadIdx.x
    // Now: row from blockIdx.x/threadIdx.x, col from blockIdx.y/threadIdx.y
    //
    // This means threads 0-31 (which have consecutive threadIdx.x) now have
    // consecutive ROWS instead of consecutive COLS.
    //
    // For A[row * K + k]: consecutive rows means addresses differ by K (still strided!)
    // For B[k * N + col]: same col means same address (broadcast)
    //
    // Hmm, this just swaps the problem! Let's think differently...
    //
    // ACTUAL SOLUTION: The blog uses a different approach.
    // Keep the thread-to-output mapping, but change which thread loads which data
    // during the computation. See the reference implementation.
    //
    // For this kernel, the key insight is:
    // - Use blockIdx.x for COLUMN blocks (so C writes are coalesced)
    // - Use blockIdx.y for ROW blocks
    // - Map threadIdx.x to column (consecutive threads → consecutive cols)
    // - Map threadIdx.y to row
    //
    // Wait, that's what kernel 1 already does! Let me re-read the blog...
    //
    // Ah! The blog's kernel 1 had the OPPOSITE mapping:
    // - row = blockIdx.x * blockDim.x + threadIdx.x  (row from x!)
    // - col = blockIdx.y * blockDim.y + threadIdx.y  (col from y!)
    //
    // That means threads 0-31 have different ROWS, same COL.
    // A[row * K + k] with different rows = strided by K (bad!)
    // B[k * N + col] with same col = broadcast (wasteful but ok)
    //
    // The FIX is to swap to:
    // - row = blockIdx.y * blockDim.y + threadIdx.y
    // - col = blockIdx.x * blockDim.x + threadIdx.x
    //
    // Now threads 0-31 have same ROW, different COL.
    // A[row * K + k] with same row = broadcast (ok)
    // B[k * N + col] with different cols = consecutive addresses (coalesced!)
    // C[row * N + col] with different cols = consecutive addresses (coalesced!)

    // TODO: Implement the coalesced version
    // The key is ensuring that the COLUMN index varies with threadIdx.x
    // so that consecutive threads access consecutive memory addresses

    const int col = blockIdx.x * BLOCKSIZE + threadIdx.x;  // x → col for coalescing
    const int row = blockIdx.y * BLOCKSIZE + threadIdx.y;  // y → row

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            // A[row][k] - same row for threads in warp (broadcast)
            // B[k][col] - different col for threads in warp (coalesced!)
            sum += A[row * K + k] * B[k * N + col];
        }
        // C[row][col] - different col for threads in warp (coalesced!)
        C[row * N + col] = alpha * sum + beta * C[row * N + col];
    }
}

/*
 * Wait... this looks identical to kernel 1!
 *
 * The issue is that YOUR kernel 1 was already correctly ordered!
 * The blog's "naive" kernel had them swapped (row from x, col from y),
 * which caused the coalescing issue.
 *
 * So for you, kernel 1 and kernel 2 will perform similarly because
 * you already had the correct coalescing pattern.
 *
 * To see the difference, try the WRONG mapping below and compare:
 */

__global__ void sgemm_NOT_coalesced(int M, int N, int K, float alpha,
                                     const float *A, const float *B, float beta, float *C) {
    // WRONG: row from x, col from y
    // This causes threads 0-31 to have different rows, same col
    // B access becomes broadcast, C write becomes strided
    const int row = blockIdx.x * BLOCKSIZE + threadIdx.x;  // WRONG for coalescing
    const int col = blockIdx.y * BLOCKSIZE + threadIdx.y;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = alpha * sum + beta * C[row * N + col];
    }
}

/*
 * Questions:
 *
 * 1. Run both sgemm_coalesced and sgemm_NOT_coalesced.
 *    What's the performance difference?
 *
 * 2. In sgemm_NOT_coalesced, what memory addresses do threads 0-31 access for: tx = 0-31, ty = 0
 *    - A[row * K + k]? Are they consecutive? A[k], A[K + k], ..., A[32K +k] for k = 0...N
 *    - B[k * N + col]? Are they consecutive? B[k * N] for k = 0...N
 *    - C[row * N + col]? Are they consecutive? C[0], C[N], ... C[31N]
 *
 * 3. Why does the WRITE to C matter for performance?
 *    (Hint: global memory writes can also be coalesced)
 *
 * 4. The blog shows ~6x improvement from coalescing. Will you see similar gains?
 *    (Hint: Your kernel 1 was already coalesced for B and C!)
 */
