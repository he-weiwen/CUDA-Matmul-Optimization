# Kernel 10: padding versus swizzling

Tested on RTX 4090, CUDA 13.4.92, sm_89, 4096 x 4096 FP32 GEMM.
All variants use CuTe only for layouts; copies, arithmetic and synchronization
remain handwritten CUDA. Experiments used isolated source copies. Padding +4 was subsequently selected
for the working kernel; it reduces, rather than eliminates, A-store bank conflicts.

## Conclusion

Padding is the simpler representation: change the shared A stride from BM to
BM + PAD and allocate (BM + PAD) * BK floats. Swizzling composes an XOR mapping
with the original layout and needs more care to retain efficient vector loads.

Neither removing all bank conflicts nor using fewer shared-memory bytes guarantees
a faster kernel. Padding by four was a small gain (~1.6%) in the final paired
comparison, similar to adding explicit vector reads without changing the layout.
The optimized swizzle eliminated conflicts but was ~1.1% slower than baseline.
These small differences should not be treated as universal wins/losses.

## Final controlled comparison

Seven invocations per variant, randomized order, 50 cuBLAS thermal warmups,
10 kernel warmups and 30 event-timed launches per invocation. Each invocation
verified results against cuBLAS. GPU clocks were not locked. The TFLOP/s ranges
below are min/max across invocations; the table uses medians for comparisons.

| Variant | Median ms | Median TFLOP/s | TFLOP/s range | Registers/thread | Shared bytes | NCU shared-store conflicts |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 2.843 | 48.34 | 47.41–48.65 | 166 | 16384 | 50,331,648 |
| baseline-vector | 2.797 | 49.15 | 48.63–50.53 | 167 | 16384 | 50,331,648 |
| pad4 | 2.799 | 49.11 | 48.50–49.46 | 166 | 16640 | 16,777,216 |
| swizzle-base-vector | 2.874 | 47.83 | 47.51–49.07 | 168 | 16384 | 0 |

All four allow three blocks/SM. All have 96 static LDS.128 instructions, no scalar
LDS instructions, and zero shared-load bank conflicts. The swizzle adds address
instructions: LOP3.LUT count rises from 30 to 97. It preserves 16 KiB shared
storage; pad4 adds 256 bytes/block. Neither spills to local memory.

## Initial layout-only comparison

Five invocations each, randomized order. This is a separate measurement round:
compare variants within this table, not small differences across rounds.

| Variant | Median TFLOP/s | Registers/thread | Store conflicts | Main tradeoff |
|---|---:|---:|---:|---|
| Baseline | 49.14 | 166 | 50,331,648 | Vector reads; 3 blocks/SM |
| Padding +1 | 43.20 | 202 | 16,777,216 | Some scalar/64-bit reads; 2 blocks/SM |
| Padding +2 | 43.95 | 198 | 0 | Some 64-bit reads; 2 blocks/SM |
| Padding +4 | 49.56 | 166 | 16,777,216 | Keeps 128-bit reads; 3 blocks/SM |
| Direct swizzle | 32.69 | 168 | 0 | Compiler loses vectorization of A reads |

Padding +2 removes store conflicts but its 130-float column stride breaks
16-byte alignment on odd K columns. Padding +4 preserves 16-byte alignment but
only reduces the scalar A stores from four-way to two-way conflicts.

The direct swizzle generated 256 scalar LDS and only 32 LDS.128 instructions
(the latter are B reads), plus 584 LOP3.LUT instructions. It is physically
aligned, but the compiler failed to recover the contiguous A accesses through
individually swizzled indices.

## CuTe representations

Padding, used consistently for both writes and reads:

```cpp
constexpr auto smemA = cute::make_layout(
    cute::make_shape(cute::Int<BM>{}, cute::Int<BK>{}),
    cute::make_stride(cute::Int<1>{}, cute::Int<BM + PAD>{}));
__shared__ float As[(BM + PAD) * BK];
```

Swizzle, specialized here to BM=128 and BK=16:

```cpp
#include <cute/swizzle_layout.hpp>
constexpr auto smemA = cute::composition(
    cute::Swizzle<2, 3, 6>{},
    cute::make_layout(
        cute::make_shape(cute::Int<128>{}, cute::Int<16>{}),
        cute::make_stride(cute::Int<1>{}, cute::Int<128>{})));
```

For the linear index x = row + 128*k, Swizzle<2,3,6> XORs index bits 9–10
into bits 3–4. Equivalently:

```cpp
physical = 128*k + (row ^ ((k / 4) * 8));
```

This sends the four conflicting lanes to four different groups of eight banks.
The low three bits remain unchanged, preserving contiguous eight-float patches.
It does not require extra storage.

## Making the swizzle competitive

1. Applying explicit float4 A reads to the direct swizzle restores LDS.128, but
   results in 181 registers/thread and only two blocks/SM (~44.86 TFLOP/s).
2. Computing the swizzled address once per thread's patch reduces redundant
   address work, but scalar source reads still don't auto-vectorize (~42.26 TFLOP/s).
3. Combining patch-base addressing with explicit float4 reads produces the final
   swizzle-base-vector variant: 168 registers, three blocks/SM, and 96 LDS.128.

The final read pattern is:

```cpp
const auto smemRowBase = smemA(rowBase, dotIdx);
// For this TM=8, WSUBM=32 layout, offsets within each patch don't change
// the two swizzled row bits. Both patches remain aligned.
const float4 values = *reinterpret_cast<const float4 *>(
    &As[smemRowBase + rows(wSubRowIdx, i)]); // i advances by four
```

The optimized variant includes static assertions for the tile/patch assumptions.
The unswizzled explicit-vector control is important: it gets a small gain without
changing bank conflicts, so small runtime changes cannot all be credited to
conflict elimination.

## Validation and artifacts

All variants passed signed-input CPU-reference checks for 256x384 output with
K=16 or 80, alpha=1/-0.75/0, beta=0/0.5, including NaN-filled initial C when
beta=0. All benchmark invocations passed the existing cuBLAS check at 4096 square.
Padding +1/+2/+4 and the optimized swizzle were also run under compute-sanitizer
memcheck and racecheck. Full-tile restrictions still apply.

Full source variants, build scripts, timing JSON, SASS dumps, sanitizer logs and
NCU reports are in `/tmp/kernel10-layouts/`. Each variant directory contains its
own source, benchmark executable and `profile.ncu-repz`. NCU profiling used:

```sh
ncu --set full --kernel-name regex:sgemm_warptiling \
  --launch-skip 10 --launch-count 1 --force-overwrite \
  -o <report> <variant>/sgemm 10 4096
```

NCU runs were separate from unprofiled timing. Replayed application's event
timings are not benchmark results. Raw shared-bank-conflict counts above are
per profiled launch. Candidate patches are saved alongside this note so the
variants can be recreated even after temporary artifacts expire. They apply to
the unpadded CuTe kernel at commit `93142e7`, before the padding commit; apply
each independently in a separate checkout of that revision.
