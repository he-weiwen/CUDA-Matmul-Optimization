# Memory Coalescing Analysis

## Thread Layout in a 32×32 Block

Threads are linearized as: `tid = threadIdx.x + threadIdx.y * blockDim.x`

For a 32×32 block, threads 0-31 (first warp) have:
```
Thread 0:  threadIdx.x = 0,  threadIdx.y = 0
Thread 1:  threadIdx.x = 1,  threadIdx.y = 0
Thread 2:  threadIdx.x = 2,  threadIdx.y = 0
...
Thread 31: threadIdx.x = 31, threadIdx.y = 0
```

**Key:** Threads 0-31 have `threadIdx.x = 0..31` and `threadIdx.y = 0` (constant).

---

## Kernel 1 & 2 (Coalesced Version)

```cpp
const int row = blockIdx.y * blockDim.y + threadIdx.y;  // y → row
const int col = blockIdx.x * blockDim.x + threadIdx.x;  // x → col
```

For threads 0-31 (assuming blockIdx = (0,0)):
```
Thread 0:  row = 0 + 0 = 0,  col = 0 + 0  = 0
Thread 1:  row = 0 + 0 = 0,  col = 0 + 1  = 1
Thread 2:  row = 0 + 0 = 0,  col = 0 + 2  = 2
...
Thread 31: row = 0 + 0 = 0,  col = 0 + 31 = 31
```

**Result:** row = 0 (same), col = 0,1,2,...,31 (consecutive)

| Access | Formula | Thread 0 | Thread 1 | ... | Thread 31 | Coalesced? |
|--------|---------|----------|----------|-----|-----------|------------|
| A[row*K+k] | A[0*K+k] | A[k] | A[k] | ... | A[k] | Broadcast (same addr) |
| B[k*N+col] | B[k*N+col] | B[k*N+0] | B[k*N+1] | ... | B[k*N+31] | **YES** |
| C[row*N+col] | C[0*N+col] | C[0] | C[1] | ... | C[31] | **YES** |

---

## Kernel 20 (NOT Coalesced Version)

```cpp
const int row = blockIdx.x * BLOCKSIZE + threadIdx.x;  // x → row (SWAPPED!)
const int col = blockIdx.y * BLOCKSIZE + threadIdx.y;  // y → col (SWAPPED!)
```

For threads 0-31 (assuming blockIdx = (0,0)):
```
Thread 0:  row = 0 + 0  = 0,   col = 0 + 0 = 0
Thread 1:  row = 0 + 1  = 1,   col = 0 + 0 = 0
Thread 2:  row = 0 + 2  = 2,   col = 0 + 0 = 0
...
Thread 31: row = 0 + 31 = 31,  col = 0 + 0 = 0
```

**Result:** row = 0,1,2,...,31 (consecutive), col = 0 (same)

| Access | Formula | Thread 0 | Thread 1 | ... | Thread 31 | Coalesced? |
|--------|---------|----------|----------|-----|-----------|------------|
| A[row*K+k] | A[row*K+k] | A[k] | A[K+k] | ... | A[31K+k] | **NO** (stride K) |
| B[k*N+col] | B[k*N+0] | B[k*N] | B[k*N] | ... | B[k*N] | Broadcast (same addr) |
| C[row*N+col] | C[row*N+0] | C[0] | C[N] | ... | C[31N] | **NO** (stride N) |

---

## Summary Table

| Kernel | row varies? | col varies? | A coalesced? | B coalesced? | C coalesced? |
|--------|-------------|-------------|--------------|--------------|--------------|
| 1 & 2  | No (same)   | Yes (0-31)  | Broadcast    | **YES**      | **YES**      |
| 20     | Yes (0-31)  | No (same)   | **NO** (stride K) | Broadcast | **NO** (stride N) |

---

## Performance Results

| Kernel | GFLOPs | % of cuBLAS | Notes |
|--------|--------|-------------|-------|
| 1 (naive, already coalesced) | 4,850 | 9.2% | B and C coalesced |
| 2 (explicitly coalesced) | 4,839 | 9.2% | Same as kernel 1 |
| **20 (NOT coalesced)** | **601** | **1.1%** | **8× slower!** |

---

## Why Non-Coalesced is Slow

When threads in a warp access strided memory (e.g., stride K or stride N):
- Each thread's request goes to a different cache line
- Instead of 1 memory transaction for 32 threads, you get up to 32 separate transactions
- Memory bandwidth is wasted fetching data that won't be used

For kernel 20:
- A reads: 32 separate transactions (stride K)
- C writes: 32 separate transactions (stride N)
- Result: ~8× slower than coalesced version

---

## Key Takeaway

For row-major matrices, ensure consecutive threads (threadIdx.x varying) access the **column** dimension, not the row dimension. This makes memory addresses consecutive.

```cpp
// GOOD: threadIdx.x → col (consecutive memory)
const int col = blockIdx.x * blockDim.x + threadIdx.x;
const int row = blockIdx.y * blockDim.y + threadIdx.y;

// BAD: threadIdx.x → row (strided memory)
const int row = blockIdx.x * blockDim.x + threadIdx.x;
const int col = blockIdx.y * blockDim.y + threadIdx.y;
```
