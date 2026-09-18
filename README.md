# CUDA Matmul Optimization

This repo contains a progressive FP32 SGEMM study in CUDA.

## Optimization progression

Results below are for **4096 x 4096 x 4096 FP32 GEMM on an RTX 4090**.
Early timings are historical measurements from [PROGRESS.md](PROGRESS.md),
collected before the benchmark setup was unified. Later comparisons come from
separate controlled rounds; compare small differences within the same round.
Percentages use the corresponding cuBLAS reference: historical reported ratios
for kernels 1–9, and ratios of median throughputs for the layout
comparison. The cuBLAS baseline varies between measurement rounds.
**Inspect** identifies an NCU check where no measured before/after counters
were saved in the notes.

| Step | What changed | NCU evidence or statistic to inspect | Recorded performance |
|---|---|---|---|
| 1: baseline | One output per thread; already coalesced. | Inspect L1/TEX global-load sectors/request and executed global-load instructions. | 4.848 TFLOP/s (9.3% of cuBLAS). |
| 1 → 2: coalescing | Map neighboring lanes to neighboring columns; same access pattern as kernel 1. | No improvement expected over kernel 1. Inspect sectors/request against deliberately strided kernel 20. | 4.843 TFLOP/s (9.3% of cuBLAS); kernel 20 gets 0.601 TFLOP/s (1.1% of cuBLAS), about 8x slower. |
| 2 → 3: shared-memory tiling | Cooperatively load 32x32 tiles and reuse inputs within a block. | Inspect global-load instruction/sector counts and shared-memory throughput. | 5.600 TFLOP/s (10.7% of cuBLAS), about 1.16x. |
| 3 → 4: 1D register tiling | Compute eight outputs per thread, reusing one B operand. | Recorded L1/shared throughput 79.7%, compute throughput 79.6%, DRAM throughput 4.5%; 48 registers/thread. | 19.169 TFLOP/s (33.2% of cuBLAS), about 3.42x. |
| 4 → 5: 2D register tiling | Compute an 8x8 outer product per thread, reusing A and B. | Inspect shared-load instructions per FFMA; source-level reuse rises from 1.78 to 8 FLOPs per shared float loaded. | 37.290 TFLOP/s (71.7% of cuBLAS), about 1.95x. |
| 5 → 6: vectorization | Transpose shared A; use `float4` loads and output stores. | Inspect executed load/store counts and SASS `LDG.E.128` / `LDS.128` for wider accesses. | 44.156 TFLOP/s (81.5% of cuBLAS), about 1.18x. |
| 6 → 9: autotuning | Select 128x128x16 block tiles; doubling K depth halves tile transitions. | Inspect executed barriers, shared bytes/block, occupancy limits, and spills across candidates. | 45.404 TFLOP/s (85.3% of cuBLAS), about 1.03x in the historical table. |
| 9 → 10: warp tiling (`c3e9824`) | Add 64x64 warp tiles; 128 threads own 128 accumulators each. | Initial profile: 184 registers/thread, register limit of 2 blocks/SM, issue slots busy 64.70%. | No isolated event-timed 9 → 10 comparison retained here. |
| 10: CuTe layouts (`93142e7`) | Replace manual offset bookkeeping with layout indexing. | Registers 184 → 166; register block limit 2 → 3; occupancy 17.19% → 23.38%; issue slots busy 64.70% → 72.76%; zero spills. | **NCU duration** 2.83 → 2.57 ms; separate unprofiled baseline: 48.34 TFLOP/s (93.5% of cuBLAS). |
| 10: padding (`6fcd918`) | Pad shared A's stride from 128 to 132 floats, preserving 16-byte alignment. | Shared-store conflicts 50,331,648 → 16,777,216; registers stay at 166, shared bytes 16,384 → 16,640, block limit stays at 3. | Paired medians: 2.843 → 2.799 ms; 48.34 TFLOP/s (93.5% of cuBLAS) → 49.11 TFLOP/s (95.7% of cuBLAS), **1.6% faster**. |

Kernels 7–8 have no separate implementation here; shared-layout experiments
are documented with kernel 10. Saved CuTe profiles:
`/tmp/kernel10-profile/{before,after}.details.txt`.
See the [layout comparison](notes/02_kernel10_shared_layouts.md) for the
measured effect of padding.

Overall, recorded throughput rises from about **4.8 TFLOP/s (9.3% of cuBLAS)
to 49 TFLOP/s (~94% of cuBLAS)**, roughly 10x across the study. The ~94%
figure comes from a separate paired comparison against cuBLAS FP32 pedantic.

## Canonical Benchmark Spec

The official comparison mode in this repo is now:

- row-major `C = A @ B`
- FP32 inputs, FP32 accumulation, FP32 outputs
- cuBLAS reference via `cublasGemmEx(..., CUBLAS_COMPUTE_32F_PEDANTIC, ...)`
- 50 thermal warmups, then 10 warmups + 30 timed runs

This means the canonical reference is **cuBLAS FP32 pedantic**, not the faster explicit TF32 tensor-core path.

Throughput is `2*M*N*K / seconds`. Measure runtime outside NCU: replay can
distort the application's event timings. Raw counter counts above are per
profiled launch.

For L1/TEX `Sectors/Req`, aligned contiguous accesses by 32 active lanes
ideally use 4 sectors for scalar FP32 or 16 for `float4`; interpret the ratio
using the instruction width. For shared memory, compare bank conflicts and
wavefronts with the access's ideal work. See NVIDIA's
[NCU metric definitions](https://docs.nvidia.com/nsight-compute/ProfilingGuide/#memory-tables).

## Build

```bash
cd /home/whe302/cuda/my_kernels
cmake -S . -B build
cmake --build build -j
```

Kernel 10 uses only CuTe's header-only layout abstraction for indexing. CMake
looks for CUTLASS headers in `../cutedsl/cutlass/include` by default. To use a
different checkout, configure with `-DCUTLASS_INCLUDE_DIR=/path/to/cutlass/include`.
Loads, stores, arithmetic, and synchronization remain handwritten CUDA.

## Existing CUDA kernels

```bash
./build/sgemm 0 4096
./build/sgemm 6 4096
./build/sgemm 9 4096
./build/sgemm 10 4096
```

Capture a separate profile after ten kernel warmups:

```bash
ncu --set full --kernel-name regex:sgemm_warptiling \
  --launch-skip 10 --launch-count 1 -o /tmp/kernel10 ./build/sgemm 10 4096
```
