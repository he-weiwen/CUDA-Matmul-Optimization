# CUDA MatMul Optimization Progress

## Goal
Learn and reproduce the optimizations from [siboehm's CUDA SGEMM blog](https://siboehm.com/articles/22/CUDA-MMM), achieving >90% of cuBLAS performance on RTX 4090.

## Setup Complete

| Item | Location |
|------|----------|
| Blog reference (full content + images) | `blog_reference/CUDA_MatMul_Optimization.md` |
| 27 diagrams | `blog_reference/images/` |
| Reference implementations | `SGEMM_CUDA/src/kernels/` |
| Your workspace | `my_kernels/` |
| Study plan | `my_kernels/STUDY_PLAN.md` |
| Notes | `my_kernels/notes/` |

## Hardware Baseline

- **GPU:** RTX 4090 (24GB, compute capability 8.9)
- **CUDA:** 13.0
- **cuBLAS performance:** ~52,000-57,000 GFLOPs (varies by run)

## Progress

### Completed Kernels

| Kernel | Status | GFLOPs | % of cuBLAS | Notes |
|--------|--------|--------|-------------|-------|
| 1 - Naive | Done | 4,848 | 9.3% | Baseline implementation |
| 2 - Coalesced | Done | 4,843 | 9.3% | Same as K1 (yours was already coalesced) |
| 20 - NOT Coalesced | Done | 601 | 1.1% | Comparison showing 8x penalty |
| 3 - Shared Memory | Done | 5,600 | 10.7% | 1.16x over K1 |
| 4 - 1D Blocktiling | Done | 19,169 | 33.2% | 3.4x over K3 |
| 5 - 2D Blocktiling | Done | 37,290 | 71.7% | 1.9x over K4 |
| 6 - Vectorized | Done | 44,156 | 81.5% | 1.18x over K5 |
| 9 - Autotuned | Done | 45,404 | 85.3% | BK=16 from parameter sweep |

### Remaining Kernels

| Kernel | Expected GFLOPs | Key Concept |
|--------|-----------------|-------------|
| 10 - Warptiling | ~48,000+ | Three-level tiling: Block -> Warp -> Thread |

## Key Learnings

### 1. Memory Coalescing (Notes: `notes/01_memory_coalescing.md`)

- Threads 0-31 in a warp have `threadIdx.x = 0..31`, `threadIdx.y = 0`
- For coalesced access: map `threadIdx.x` to the **column** (innermost dimension)
- Non-coalesced access causes **8x slowdown** (kernel 20 demo)

### 2. Shared Memory Cache-Blocking (Kernel 3)

- SMEM is ~10-20x faster than GMEM
- Load tiles into SMEM, compute, then load next tile
- Need `__syncthreads()` after loading AND before next load
- Modest improvement (1.16x) because arithmetic intensity still low

### 3. 1D Blocktiling (Kernel 4)

- Each thread computes TM=8 outputs instead of 1
- 1D thread block: `threadCol = tid % BN`, `threadRow = tid / BN` (row-group index)
- Actual rows: `threadRow * TM + resIdx`
- Pointer advancement simplifies indexing (shift A/B/C before the loop)
- Constraint: `BK = BN / TM` (each thread loads exactly 1 element of As and Bs)
- With BM=BN=64, BK=TM=8: blockDim=512 threads, SMEM=4KB per tile pair

### 4. GPU Memory Hierarchy (Corrected Understanding)

- **Registers** (256 KB/SM, ~1 cycle) = physically separate SRAM (register file)
- **Shared memory + L1 cache** (128 KB/SM, ~20-30 cycles) = same physical "unified data cache"
- Registers are NOT part of L1 — they are a distinct, faster structure
- Source: CUDA Programming Guide: "Each SM contains a local register file, a unified data cache"

### 5. Nsight Compute Profiling

- `ncu --set basic -o <file>` for quick overview; `--set full` for detailed analysis
- Need `sudo` (or set `NVreg_RestrictProfilingToAdminUsers=0`)
- Use full path with sudo: `sudo /usr/local/cuda/bin/ncu ...`

**Kernel 4 ncu profile findings:**
- L1/SMEM throughput: 79.7% (bottleneck — too many SMEM reads per FLOP)
- Compute throughput: 79.6% (well balanced with memory)
- DRAM throughput: 4.5% (good — SMEM tiling working)
- Occupancy: 65.7% (not the bottleneck)
- 48 registers/thread

**cuBLAS ncu profile (for comparison):**
- Compute throughput: 80.3%, Memory throughput: 42.6%
- 202 registers/thread, occupancy 16.7%
- Low occupancy is intentional: massive register reuse + ILP hides latency
- ~63% of theoretical FP32 peak (82.6 TFLOPS)

**Identifying performance issues in ncu:**
- Uncoalesced global access: sectors per request >> 4.0
- Shared memory bank conflicts: wavefronts per request > 1.0
- Both require `--set full`

### 6. 2D Blocktiling (Kernel 5)

- Each thread computes TM x TN = 8x8 = 64 outputs via outer product
- Arithmetic intensity: 2*TM*TN / (TM+TN) = 128/16 = 8.0 (vs 1.78 in kernel 4)
- Thread mapping: `threadCol = tid % (BN/TN)`, `threadRow = tid / (BN/TN)`
- Fewer threads (256 vs 512), more work per thread
- Strided SMEM loading: stride = numThreads / tile_width (interleaved, not contiguous, for GMEM coalescing)

### 7. Vectorization (Kernel 6)

- float4 (128-bit) loads reduce instruction count: 88 → 18 GMEM loads, 64 → 16 GMEM stores
- Verified in PTX: `ld.global.f32` → `ld.global.v4.f32`, `ld.shared.f32` → `ld.shared.v4.f32`
- **nvcc does NOT auto-vectorize SMEM loads** — needs explicit `reinterpret_cast<float4*>`
- Transposing As in SMEM serves two purposes:
  1. Makes regM loads contiguous (stride 1 instead of BK) → enables float4 SMEM reads
  2. Eliminates 2-way shared memory bank conflicts between threads with different threadRow
- float4 SMEM reads reduce instruction count (LDS.128 vs 4x LDS.32) but same bandwidth

### 8. Register Arrays and SROA

- `float regM[TM]` compiles to registers via LLVM's SROA (Scalar Replacement of Aggregates)
- SROA is a **general-purpose mid-end pass** (target-independent, not NVPTX-specific)
- Lives in `llvm/lib/Transforms/Scalar/SROA.cpp`
- Works when all array indices are compile-time constants (after loop unrolling)
- If indices are dynamic, array spills to local memory (DRAM) — massive performance penalty

### 9. Tiling vs Interleaving

- **Interleave** when consecutive threads must hit consecutive GMEM addresses (coalescing)
- **Tile** for everything else (output ownership, computation layout)
- SMEM loading uses interleaved pattern for the GMEM read side; SMEM write side doesn't care

### 10. Autotuning (Kernel 9)

- Strided float4 SMEM loading decouples BK from thread count
- Warp iteration (WMITER/WNITER) allows larger block tiles with fixed thread count
- **Best params for RTX 4090:** BM=128, BN=128, BK=16, TM=8, TN=8
- BK=16 beats BK=8 by ~5%: halves `__syncthreads()` barrier count
- BK=32 hurts: SMEM usage limits occupancy
- BM=BN=256 causes register spills (threadResults too large)
- Stable benchmarking requires: thermal warm-up (50 cuBLAS runs), 10 warmup + 30 measured runs
- **Vectorization is context-dependent:** float4 reg loads helped kernel 6 but must re-tune after code changes

## Next Steps

1. **Kernel 10: Warptiling** — three-level tiling hierarchy (Block -> Warp -> Thread) for the final push toward cuBLAS

## Commands

```bash
cd my_kernels/build

# Build
cmake --build .

# Run specific kernel
./sgemm <kernel_number>

# Run with different matrix size
./sgemm <kernel_number> <size>

# Run parameter sweep
./sweep

# Profile (need sudo for full counters)
sudo /usr/local/cuda/bin/ncu --set full -o <output_name> ./build/sgemm <kernel_number>

# Read profile report
ncu -i <file>.ncu-rep

# Dump PTX for inspection
nvcc -ptx -o /tmp/kernels.ptx sgemm.cu -I src --std=c++17 -arch=sm_89
```

## File Structure

```
my_kernels/
├── CMakeLists.txt
├── sgemm.cu              # Main runner with benchmarking
├── sweep.cu              # Parameter sweep for autotuning
├── STUDY_PLAN.md         # Full study plan
├── PROGRESS.md           # This file
├── build/                # Build directory
├── notes/
│   └── 01_memory_coalescing.md
└── src/
    ├── kernels.cuh       # Kernel selector
    ├── 1_naive.cuh       # Done
    ├── 2_coalesced.cuh   # Done (includes kernel 20)
    ├── 3_shared_mem.cuh  # Done
    ├── 4_1d_blocktiling.cuh  # Done
    ├── 5_2d_blocktiling.cuh  # Done
    ├── 6_vectorized.cuh      # Done
    └── 9_autotuned.cuh       # Done
```
