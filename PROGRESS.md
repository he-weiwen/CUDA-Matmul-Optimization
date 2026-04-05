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
- **cuBLAS performance:** ~52,000 GFLOPs (significantly faster than blog's A6000 at ~23,000)

## Progress

### Completed Kernels

| Kernel | Status | GFLOPs | % of cuBLAS | Notes |
|--------|--------|--------|-------------|-------|
| 1 - Naive | ✅ Done | 4,848 | 9.3% | You implemented this |
| 2 - Coalesced | ✅ Done | 4,843 | 9.3% | Same as K1 (yours was already coalesced) |
| 20 - NOT Coalesced | ✅ Done | 601 | 1.1% | Comparison showing 8× penalty |
| 3 - Shared Memory | ✅ Done | 5,600 | 10.7% | 1.16× improvement |

### Remaining Kernels

| Kernel | Expected GFLOPs | Key Concept |
|--------|-----------------|-------------|
| 4 - 1D Blocktiling | ~18,000 | Each thread computes TM outputs |
| 5 - 2D Blocktiling | ~35,000 | Each thread computes TM×TN outputs |
| 6 - Vectorized | ~40,000 | float4 loads, transposed SMEM |
| 9 - Autotuned | ~42,000 | Parameter optimization |
| 10 - Warptiling | ~48,000 | Warp-level tiling hierarchy |

## Key Learnings So Far

### 1. Memory Coalescing (Notes: `notes/01_memory_coalescing.md`)

- Threads 0-31 in a warp have `threadIdx.x = 0..31`, `threadIdx.y = 0`
- For coalesced access: map `threadIdx.x` to the **column** (innermost dimension)
- Non-coalesced access causes **8× slowdown** (kernel 20 demo)

```cpp
// GOOD: col from threadIdx.x → consecutive memory
const int col = blockIdx.x * blockDim.x + threadIdx.x;
const int row = blockIdx.y * blockDim.y + threadIdx.y;

// BAD: row from threadIdx.x → strided memory
const int row = blockIdx.x * blockDim.x + threadIdx.x;
const int col = blockIdx.y * blockDim.y + threadIdx.y;
```

### 2. Shared Memory Cache-Blocking

- SMEM is ~10-20× faster than GMEM
- Load tiles into SMEM, compute, then load next tile
- Need `__syncthreads()` after loading AND before next load
- Modest improvement (1.16×) because arithmetic intensity still low

## Next Steps

1. **Answer questions in `3_shared_mem.cuh`** (optional but recommended)
2. **Kernel 4: 1D Blocktiling** - This is where big gains happen (~3× from K3)
   - Each thread computes TM elements instead of 1
   - Dramatically increases arithmetic intensity

## Commands

```bash
cd my_kernels/build

# Build
cmake --build .

# Run specific kernel
./sgemm <kernel_number>

# Run with different matrix size
./sgemm <kernel_number> <size>

# Example: run kernel 3 with 2048x2048 matrices
./sgemm 3 2048
```

## File Structure

```
my_kernels/
├── CMakeLists.txt
├── sgemm.cu              # Main runner with benchmarking
├── STUDY_PLAN.md         # Full study plan
├── PROGRESS.md           # This file
├── build/                # Build directory
├── notes/
│   └── 01_memory_coalescing.md
└── src/
    ├── kernels.cuh       # Kernel selector
    ├── 1_naive.cuh       # ✅ Complete
    ├── 2_coalesced.cuh   # ✅ Complete
    └── 3_shared_mem.cuh  # ✅ Complete (review questions)
```

## To Continue

Run: `cd /home/whe302/cuda/my_kernels/build && ./sgemm 3` to verify setup still works, then ask to proceed with Kernel 4.
