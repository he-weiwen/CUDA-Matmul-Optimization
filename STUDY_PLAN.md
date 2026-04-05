# CUDA MatMul Optimization Study Plan

## Overview

You will implement **10 progressively optimized SGEMM kernels** from scratch, targeting your RTX 4090 (compute capability 8.9). Each step builds on the previous one.

**Goal:** Achieve >90% of cuBLAS performance (~21,000+ GFLOPs on 4096x4096 matrices)

---

## Directory Structure

```
my_kernels/
├── STUDY_PLAN.md          # This file
├── CMakeLists.txt         # Build configuration
├── sgemm.cu               # Main runner with benchmarking
├── src/
│   ├── kernels.cuh        # Kernel selector
│   ├── 1_naive.cuh        # Your implementations
│   ├── 2_coalesced.cuh
│   └── ...
└── benchmarks/            # Performance results
```

**Reference implementations:** `../SGEMM_CUDA/src/kernels/`
**Reference diagrams:** `../blog_reference/images/`

---

## Study Milestones

### Phase 1: Foundations (Kernels 1-3)
Understanding memory access patterns and the memory hierarchy.

| Kernel | Target GFLOPs | Key Concepts |
|--------|--------------|--------------|
| 1 - Naive | ~300 | Basic CUDA kernel structure, thread indexing |
| 2 - Coalesced | ~2,000 | Memory coalescing, warp behavior |
| 3 - Shared Memory | ~3,000 | SMEM, cache blocking, __syncthreads() |

### Phase 2: Register Optimization (Kernels 4-5)
Maximizing computation per memory access.

| Kernel | Target GFLOPs | Key Concepts |
|--------|--------------|--------------|
| 4 - 1D Blocktiling | ~8,500 | Register reuse, arithmetic intensity |
| 5 - 2D Blocktiling | ~16,000 | Outer product, TM×TN tiles |

### Phase 3: Memory Optimization (Kernels 6-8)
Fine-tuning memory access patterns.

| Kernel | Target GFLOPs | Key Concepts |
|--------|--------------|--------------|
| 6 - Vectorized | ~18,000 | float4, LDS.128, transposed SMEM |
| 7 - Bank Conflicts | ~16,500 | SMEM bank layout, conflict resolution |
| 8 - Bank Offset | ~16,500 | Alternative bank conflict solution |

### Phase 4: Advanced (Kernels 9-10)
Final optimizations for peak performance.

| Kernel | Target GFLOPs | Key Concepts |
|--------|--------------|--------------|
| 9 - Autotuned | ~20,000 | Parameter search, occupancy |
| 10 - Warptiling | ~22,000 | Warp-level hierarchy, ILP |

---

## Detailed Steps for Each Kernel

### Kernel 1: Naive Implementation
**Diagram:** `../blog_reference/images/naive-kernel.png`

**What to implement:**
1. Each thread computes ONE element of C
2. Simple nested loop: `C[row][col] = sum(A[row][k] * B[k][col])`
3. 2D grid/block configuration

**Key questions to understand:**
- [ ] How does `blockIdx` and `threadIdx` map to matrix positions?
- [ ] Why is this kernel slow? (Check: `../blog_reference/images/naive_kernel_mem_access.png`)

**Validation:** Compare output against CPU implementation or cuBLAS

---

### Kernel 2: Global Memory Coalescing
**Diagrams:**
- `../blog_reference/images/Naive_kernel_mem_coalescing.png` (problem)
- `../blog_reference/images/GMEM_coalescing.png` (solution)

**What to change:**
1. Swap the thread-to-row/col mapping
2. Consecutive threads should access consecutive memory addresses

**Key questions:**
- [ ] What is a warp and why does it matter?
- [ ] How does the hardware combine memory requests?
- [ ] Draw the memory access pattern for threads 0-31 before and after

---

### Kernel 3: Shared Memory Cache-Blocking
**Diagrams:**
- `../blog_reference/images/memory-hierarchy-in-gpus.png`
- `../blog_reference/images/cache-blocking.png`

**What to implement:**
1. Declare `__shared__` arrays for tiles of A and B
2. Load BK×BM tile of A and BK×BN tile of B into SMEM
3. Compute partial results from SMEM
4. Loop over K dimension in BK-sized chunks
5. Use `__syncthreads()` to ensure all threads complete loading

**Key questions:**
- [ ] Why is shared memory faster than global memory?
- [ ] What happens if you forget `__syncthreads()`?
- [ ] How do you choose BK, BM, BN sizes?

---

### Kernel 4: 1D Blocktiling
**Diagram:** `../blog_reference/images/kernel_4_1D_blocktiling.png`

**What to change:**
1. Each thread computes TM results (not 1)
2. Store results in registers: `float threadResults[TM]`
3. Load A values into registers, reuse for all TM outputs

**Key questions:**
- [ ] How does computing more per thread reduce memory pressure?
- [ ] What is "arithmetic intensity" and why does it matter?
- [ ] How many registers does each thread use now?

---

### Kernel 5: 2D Blocktiling
**Diagrams:**
- `../blog_reference/images/kernel_5_2D_blocktiling.png`
- `../blog_reference/images/kernel_5_reg_blocking.png`

**What to change:**
1. Each thread computes TM × TN results
2. Inner loop computes outer product: regA[i] * regB[j]
3. Adjust thread block dimensions accordingly

**Key questions:**
- [ ] What is the optimal TM, TN for your GPU?
- [ ] How does 2D tiling increase arithmetic intensity vs 1D?
- [ ] What limits how large TM and TN can be?

---

### Kernel 6: Vectorized Memory Access
**Diagram:** `../blog_reference/images/kernel_6_As_transpose.png`

**What to implement:**
1. Store A transposed in SMEM: `As[col][row]` instead of `As[row][col]`
2. Use `float4` for global memory loads
3. Use `reinterpret_cast<float4*>` for vectorized access

**Key questions:**
- [ ] Why does transposing A enable wider SMEM loads?
- [ ] How do you ensure 128-bit alignment for float4?
- [ ] Check assembly: are you seeing LDS.128 and LDG.E.128?

---

### Kernels 7-8: Bank Conflicts
**What to understand:**
1. SMEM is organized into 32 banks
2. Threads in same warp accessing same bank = conflict
3. Two solutions: linearize indexing OR add padding offset

---

### Kernel 9: Autotuning
**What to do:**
1. Create parameter sweep: BM, BN, BK, TM, TN
2. Run benchmarks for each combination
3. Find optimal parameters for your RTX 4090

---

### Kernel 10: Warptiling
**Diagrams:**
- `../blog_reference/images/WarpSchedulers.png`
- `../blog_reference/images/kernel_10_warp_tiling.png`

**What to implement:**
1. Three-level hierarchy: Block → Warp → Thread
2. Explicit warp-level tiling between SMEM and registers
3. Maximize instruction-level parallelism

---

## Benchmarking Checklist

For each kernel, record:
- [ ] GFLOPs achieved at 4096×4096
- [ ] Memory throughput (GB/s)
- [ ] Occupancy (%)
- [ ] % of cuBLAS performance

## Commands

```bash
# Build your kernels
cd my_kernels && mkdir build && cd build && cmake .. && make

# Run kernel N
./sgemm <N>

# Profile with Nsight Compute
ncu --set full ./sgemm <N>
```

---

## Tips

1. **Start simple, verify correctness first** - Use small matrices (64×64) to debug
2. **Always compare against cuBLAS** - It's your ground truth
3. **Profile before optimizing** - Let the data guide you
4. **Read the reference AFTER attempting** - Try for 30 minutes, then check
5. **Draw memory access patterns** - Visual understanding is crucial
