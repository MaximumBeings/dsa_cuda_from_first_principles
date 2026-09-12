# Chapter 6: Stream Compaction

**What you will understand by the end of this chapter:**

- Why stream compaction — filtering an array down to elements that satisfy a predicate, while preserving relative order — is Chapter 5's exclusive scan wearing a disguise, not a new algorithm.
- How to generalize single-block compaction to arbitrarily large `N` with the identical three-kernel pattern Chapter 5.3 used for scan.
- How stable partitioning (keeping BOTH the elements that pass and the elements that fail, split into two ordered groups) reuses the same scan twice — the exact operation Part 3's radix sort will need once per bit.

**What you need to know first:**

- Chapter 5 in full, especially Section 5.2's exclusive scan (up-sweep/down-sweep) and Section 5.3's multi-block combination pattern, both reused directly and unchanged in this chapter.

---

Reduction (Chapter 4) answers "what is the total." Scan (Chapter 5) answers "what is the running total at every position." Stream compaction answers a third, related question: "which positions do I actually want to keep, and where does each one belong once the gaps are closed up." The answer turns out to need nothing new — just Chapter 5's own scan, pointed at a 0/1 flag array instead of the data itself.

## 6.1 Compaction via Exclusive Scan: Flags, Scan, Scatter

### Intuition

Filtering `[0, 1, 2, 3, 4, 5, 6]` down to multiples of 3 should produce `[0, 3, 6]`, in that same relative order. A sequential pass does this with one running output index, incrementing it each time an element is kept. The parallel version needs to know, for every kept element, how many OTHER kept elements come before it — which is exactly what an exclusive scan of a "was this element kept" flag array computes.

### Background

The kernel below computes a flag (1.0 if the predicate holds, 0.0 otherwise) for every position, exclusive-scans those flags with Chapter 5.2's exact up-sweep/down-sweep shape, and then every kept element scatters itself directly to the output position its own scanned flag value names — no further bookkeeping needed, because that scanned value already IS a count of every earlier kept element.

```cpp
#include <cstdio>
#include <vector>

// Chapter 6.1 -- stream compaction filters an array down to only the
// elements satisfying some predicate, PRESERVING their relative order.
// The naive sequential way is a single pass with a running output index.
// The parallel way turns out to be Chapter 5's exclusive scan wearing a
// disguise: compute a 0/1 "keep this element" flag for every position,
// exclusive-scan the flags, and each kept element's scanned value IS its
// correct output position -- every element strictly before it that was
// also kept has already been counted into that scan value.

#define BLOCK_SIZE 256

__device__ __forceinline__ bool keep_predicate(float v) {
    // Keep multiples of 3. An arbitrary, easily-checked-by-hand predicate,
    // not a special property this algorithm depends on -- any predicate
    // producing a 0/1 flag per element works identically.
    int iv = (int)v;
    return (iv % 3) == 0;
}

__global__ void compact_single_block(const float* g_in, float* g_out, int* g_count, int n) {
    __shared__ float flags[BLOCK_SIZE];   // doubles as the scan buffer
    __shared__ float vals[BLOCK_SIZE];
    int tid = threadIdx.x;

    vals[tid] = (tid < n) ? g_in[tid] : 0.0f;
    flags[tid] = (tid < n && keep_predicate(vals[tid])) ? 1.0f : 0.0f;
    __syncthreads();

    // Exclusive scan of `flags`, in place -- Chapter 5.2's exact
    // up-sweep/down-sweep shape, reused unchanged.
    int offset = 1;
    for (int d = n >> 1; d > 0; d >>= 1) {
        __syncthreads();
        if (tid < d) {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            flags[bi] += flags[ai];
        }
        offset *= 2;
    }
    __syncthreads();
    float total_kept = flags[n - 1];   // captured before zeroing, exactly like Chapter 5.3
    if (tid == 0) flags[n - 1] = 0.0f;
    for (int d = 1; d < n; d *= 2) {
        offset >>= 1;
        __syncthreads();
        if (tid < d) {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            float t = flags[ai];
            flags[ai] = flags[bi];
            flags[bi] += t;
        }
    }
    __syncthreads();

    // Scatter: a kept element's scanned value is exactly its output slot.
    if (tid < n && keep_predicate(vals[tid])) {
        g_out[(int)flags[tid]] = vals[tid];
    }
    if (tid == 0) *g_count = (int)total_kept;
}

// ---- Host-side simulation of the identical flag/scan/scatter arithmetic ----

struct CompactionResult {
    std::vector<float> out;
    int count;
};

CompactionResult simulate_compact(const std::vector<float>& in) {
    int n = (int)in.size();
    std::vector<float> flags(n);
    for (int i = 0; i < n; i++) {
        int iv = (int)in[i];
        flags[i] = ((iv % 3) == 0) ? 1.0f : 0.0f;
    }

    int offset = 1;
    for (int d = n >> 1; d > 0; d >>= 1) {
        for (int tid = 0; tid < d; tid++) {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            flags[bi] += flags[ai];
        }
        offset *= 2;
    }
    float total_kept = flags[n - 1];
    flags[n - 1] = 0.0f;
    for (int d = 1; d < n; d *= 2) {
        offset >>= 1;
        for (int tid = 0; tid < d; tid++) {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            float t = flags[ai];
            flags[ai] = flags[bi];
            flags[bi] += t;
        }
    }

    CompactionResult r;
    r.out.assign((size_t)total_kept, 0.0f);
    r.count = (int)total_kept;
    for (int i = 0; i < n; i++) {
        int iv = (int)in[i];
        if ((iv % 3) == 0) {
            r.out[(int)flags[i]] = in[i];
        }
    }
    return r;
}

std::vector<float> reference_compact(const std::vector<float>& in) {
    std::vector<float> out;
    for (float v : in) {
        if (((int)v % 3) == 0) out.push_back(v);
    }
    return out;
}

int main() {
    printf("=== Section 6.1: single-block stream compaction via exclusive scan ===\n\n");

    const int N = BLOCK_SIZE;
    std::vector<float> vals(N);
    for (int i = 0; i < N; i++) vals[i] = (float)i;   // 0, 1, 2, ..., 255

    auto result = simulate_compact(vals);
    auto ref = reference_compact(vals);

    bool correct = (result.out.size() == ref.size());
    if (correct) {
        for (size_t i = 0; i < ref.size(); i++) {
            if (result.out[i] != ref[i]) { correct = false; break; }
        }
    }

    printf("N = %d, predicate = 'value is a multiple of 3'\n", N);
    printf("compacted count: %d (independent reference count: %d)\n", result.count, (int)ref.size());
    printf("first 8 kept values:  ");
    for (int i = 0; i < 8; i++) printf("%.0f ", result.out[i]);
    printf("\nlast 8 kept values:   ");
    for (size_t i = result.out.size() - 8; i < result.out.size(); i++) printf("%.0f ", result.out[i]);
    printf("\n\n");
    printf("relative order preserved and matches independent reference exactly: %s\n",
           correct ? "yes" : "NO -- BUG");

    bool ok = correct && (result.count == 86);   // 0,3,...,255 -> 86 multiples of 3
    printf("\nself-check: compaction correct, count = 86 multiples of 3 in [0,255]: %s\n",
           ok ? "confirmed" : "MISMATCH");
    return ok ? 0 : 1;
}
```

Running this program produces:

```text
=== Section 6.1: single-block stream compaction via exclusive scan ===

N = 256, predicate = 'value is a multiple of 3'
compacted count: 86 (independent reference count: 86)
first 8 kept values:  0 3 6 9 12 15 18 21 
last 8 kept values:   234 237 240 243 246 249 252 255 

relative order preserved and matches independent reference exactly: yes

self-check: compaction correct, count = 86 multiples of 3 in [0,255]: confirmed
```

86 multiples of 3 in `[0, 255]`, in their original order, matched exactly against an independent single-pass reference. The scan did all the actual position bookkeeping; the scatter step is a single conditional write.

!!! warning "[COMMON TRAP] Forgetting the scan must be EXCLUSIVE, not inclusive"
    An inclusive scan of the flags would give a kept element at position `i` a value that already counts ITSELF, one too many for a correct zero-based output index. This is not a minor off-by-one to patch after the fact — it is the precise reason Chapter 5 built an exclusive scan (Section 5.2) at all, rather than stopping at Section 5.1's inclusive Hillis-Steele version: compaction specifically needs "how many kept elements came STRICTLY before me," which is the exclusive scan's exact definition, not an easily-adjusted variant of the inclusive one.

## 6.2 Multi-Block Stream Compaction

### Intuition

Section 6.1 compacted one block. A real compaction needs `N` larger than that, and the combination step across blocks looks exactly like Chapter 5.3's multi-block scan for the same underlying reason: each block CAN compact its own segment correctly in isolation, but every block's kept elements then need shifting by how many elements every earlier block kept — an exclusive scan of per-block kept-COUNTS, rather than per-block sums, but otherwise the identical pattern.

### Background

Kernel 1 below is Section 6.1's exact compaction, run once per block into a same-sized local segment, additionally recording each block's own kept-count. Kernel 2 exclusive-scans those per-block counts — the identical scan shape from Chapter 5, reused at a much smaller scale, exactly as Chapter 5.3 reused it for sums. Kernel 3 copies each block's locally-compacted elements into their final position, shifted by that block's now-scanned offset.

```cpp
#include <cstdio>
#include <vector>

// Chapter 6.2 -- Section 6.1 compacted exactly one block's worth of data.
// A real compaction needs N larger than one block, and the combination
// step looks like Chapter 5.3's multi-block scan for the same reason:
// each block can compact ITSELF correctly in isolation, but every
// block's kept elements need shifting by how many elements every
// EARLIER block kept -- an exclusive scan of per-block kept-counts,
// exactly Chapter 5.3's block_sums idea applied to a count instead of a
// sum of values.

#define BLOCK_SIZE 256

__device__ __forceinline__ bool keep_predicate(float v) {
    int iv = (int)v;
    return (iv % 3) == 0;
}

// Kernel 1: each block compacts its OWN segment locally (Section 6.1's
// exact flag/scan/scatter shape) into a same-sized local segment of
// g_local_compacted, and records how many elements it kept.
__global__ void compact_local(const float* g_in, float* g_local_compacted,
                               int* g_block_counts, int n_per_block) {
    __shared__ float flags[BLOCK_SIZE];
    __shared__ float vals[BLOCK_SIZE];
    int tid = threadIdx.x;
    int base = blockIdx.x * n_per_block;

    vals[tid] = g_in[base + tid];
    flags[tid] = keep_predicate(vals[tid]) ? 1.0f : 0.0f;
    __syncthreads();

    int offset = 1;
    for (int d = n_per_block >> 1; d > 0; d >>= 1) {
        __syncthreads();
        if (tid < d) {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            flags[bi] += flags[ai];
        }
        offset *= 2;
    }
    __syncthreads();
    if (tid == 0) {
        g_block_counts[blockIdx.x] = (int)flags[n_per_block - 1];
        flags[n_per_block - 1] = 0.0f;
    }
    for (int d = 1; d < n_per_block; d *= 2) {
        offset >>= 1;
        __syncthreads();
        if (tid < d) {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            float t = flags[ai];
            flags[ai] = flags[bi];
            flags[bi] += t;
        }
    }
    __syncthreads();
    if (keep_predicate(vals[tid])) {
        g_local_compacted[base + (int)flags[tid]] = vals[tid];
    }
}

// Kernel 2: exclusive-scan the (short) per-block kept-counts -- the
// identical scan shape, reused at a second, smaller scale, exactly as
// Chapter 5.3 did for sums.
__global__ void scan_block_counts(int* g_block_counts, int num_blocks) {
    __shared__ float temp[BLOCK_SIZE];
    int tid = threadIdx.x;
    temp[tid] = (tid < num_blocks) ? (float)g_block_counts[tid] : 0.0f;
    __syncthreads();

    int offset = 1;
    for (int d = num_blocks >> 1; d > 0; d >>= 1) {
        __syncthreads();
        if (tid < d) {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            temp[bi] += temp[ai];
        }
        offset *= 2;
    }
    __syncthreads();
    if (tid == 0) temp[num_blocks - 1] = 0.0f;
    for (int d = 1; d < num_blocks; d *= 2) {
        offset >>= 1;
        __syncthreads();
        if (tid < d) {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            float t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }
    __syncthreads();
    if (tid < num_blocks) g_block_counts[tid] = (int)temp[tid];   // now each block's GLOBAL offset
}

// Kernel 3: copy each block's locally-compacted elements to their final
// global position, shifted by that block's now-scanned offset.
__global__ void stitch_blocks(const float* g_local_compacted, float* g_out,
                               const int* g_block_offsets, const int* g_local_counts,
                               int n_per_block) {
    int tid = threadIdx.x;
    int block = blockIdx.x;
    int base = block * n_per_block;
    if (tid < g_local_counts[block]) {
        g_out[g_block_offsets[block] + tid] = g_local_compacted[base + tid];
    }
}

// ---- Host-side simulation of the identical three-kernel arithmetic ----

void simulate_compact_local(const std::vector<float>& in, std::vector<float>& local_compacted,
                             std::vector<int>& block_counts, int num_blocks, int n_per_block) {
    for (int block = 0; block < num_blocks; block++) {
        int base = block * n_per_block;
        std::vector<float> flags(n_per_block);
        for (int i = 0; i < n_per_block; i++) {
            flags[i] = (((int)in[base + i] % 3) == 0) ? 1.0f : 0.0f;
        }
        int offset = 1;
        for (int d = n_per_block >> 1; d > 0; d >>= 1) {
            for (int tid = 0; tid < d; tid++) {
                int ai = offset * (2 * tid + 1) - 1;
                int bi = offset * (2 * tid + 2) - 1;
                flags[bi] += flags[ai];
            }
            offset *= 2;
        }
        block_counts[block] = (int)flags[n_per_block - 1];
        flags[n_per_block - 1] = 0.0f;
        for (int d = 1; d < n_per_block; d *= 2) {
            offset >>= 1;
            for (int tid = 0; tid < d; tid++) {
                int ai = offset * (2 * tid + 1) - 1;
                int bi = offset * (2 * tid + 2) - 1;
                float t = flags[ai];
                flags[ai] = flags[bi];
                flags[bi] += t;
            }
        }
        for (int i = 0; i < n_per_block; i++) {
            if (((int)in[base + i] % 3) == 0) {
                local_compacted[base + (int)flags[i]] = in[base + i];
            }
        }
    }
}

void simulate_scan_block_counts(std::vector<int>& block_counts, int num_blocks) {
    std::vector<float> temp(num_blocks);
    for (int i = 0; i < num_blocks; i++) temp[i] = (float)block_counts[i];
    int offset = 1;
    for (int d = num_blocks >> 1; d > 0; d >>= 1) {
        for (int tid = 0; tid < d; tid++) {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            temp[bi] += temp[ai];
        }
        offset *= 2;
    }
    temp[num_blocks - 1] = 0.0f;
    for (int d = 1; d < num_blocks; d *= 2) {
        offset >>= 1;
        for (int tid = 0; tid < d; tid++) {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            float t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }
    for (int i = 0; i < num_blocks; i++) block_counts[i] = (int)temp[i];
}

std::vector<float> reference_compact(const std::vector<float>& in) {
    std::vector<float> out;
    for (float v : in) if (((int)v % 3) == 0) out.push_back(v);
    return out;
}

int main() {
    printf("=== Section 6.2: multi-block stream compaction, three kernels ===\n\n");

    const int NUM_BLOCKS = 8;
    const int N = NUM_BLOCKS * BLOCK_SIZE;   // 2048
    std::vector<float> vals(N);
    for (int i = 0; i < N; i++) vals[i] = (float)i;

    std::vector<float> local_compacted(N, -1.0f);
    std::vector<int> block_counts(NUM_BLOCKS, 0);
    simulate_compact_local(vals, local_compacted, block_counts, NUM_BLOCKS, BLOCK_SIZE);

    printf("kernel 1: %d blocks each locally compact their own %d elements\n", NUM_BLOCKS, BLOCK_SIZE);
    printf("per-block local kept-counts: ");
    for (int c : block_counts) printf("%d ", c);
    printf("\n\n");

    std::vector<int> local_counts_copy = block_counts;
    simulate_scan_block_counts(block_counts, NUM_BLOCKS);
    printf("kernel 2: exclusive-scan the %d per-block kept-counts\n", NUM_BLOCKS);
    printf("per-block global offsets:    ");
    for (int c : block_counts) printf("%d ", c);
    printf("\n\n");

    int total_kept = local_counts_copy[NUM_BLOCKS - 1] + block_counts[NUM_BLOCKS - 1];
    std::vector<float> out(total_kept, -2.0f);
    for (int block = 0; block < NUM_BLOCKS; block++) {
        int base = block * BLOCK_SIZE;
        for (int j = 0; j < local_counts_copy[block]; j++) {
            out[block_counts[block] + j] = local_compacted[base + j];
        }
    }
    printf("kernel 3: stitch each block's local results into their final global position\n\n");

    auto ref = reference_compact(vals);
    bool correct = (out.size() == ref.size());
    int first_mismatch = -1;
    if (correct) {
        for (size_t i = 0; i < ref.size(); i++) {
            if (out[i] != ref[i]) { correct = false; first_mismatch = (int)i; break; }
        }
    }

    printf("N = %d, total kept = %d (independent reference count: %d)\n", N, total_kept, (int)ref.size());
    printf("matches independent reference exactly, including order across every block\n");
    printf("boundary: %s\n", correct ? "yes" : "NO -- BUG");
    if (!correct) printf("first mismatch at output index %d\n", first_mismatch);

    printf("\nvalues straddling the boundary between block 1's kept elements and block 2's:\n");
    int b1_end = block_counts[2];   // block 2's offset marks where block 1's contribution ends
    for (int i = b1_end - 3; i <= b1_end + 2; i++) {
        printf("  out[%3d] = %6.0f   reference = %6.0f\n", i, out[i], ref[i]);
    }

    bool ok = correct;
    printf("\nself-check: multi-block compaction matches the independent reference across\n");
    printf("all %d kept elements: %s\n", total_kept, ok ? "confirmed" : "MISMATCH");
    return ok ? 0 : 1;
}
```

Running this program produces:

```text
=== Section 6.2: multi-block stream compaction, three kernels ===

kernel 1: 8 blocks each locally compact their own 256 elements
per-block local kept-counts: 86 85 85 86 85 85 86 85 

kernel 2: exclusive-scan the 8 per-block kept-counts
per-block global offsets:    0 86 171 256 342 427 512 598 

kernel 3: stitch each block's local results into their final global position

N = 2048, total kept = 683 (independent reference count: 683)
matches independent reference exactly, including order across every block
boundary: yes

values straddling the boundary between block 1's kept elements and block 2's:
  out[168] =    504   reference =    504
  out[169] =    507   reference =    507
  out[170] =    510   reference =    510
  out[171] =    513   reference =    513
  out[172] =    516   reference =    516
  out[173] =    519   reference =    519

self-check: multi-block compaction matches the independent reference across
all 683 kept elements: confirmed
```

683 kept elements across 2048, matched exactly against an independent reference at every position — including the values that straddle a block boundary, where a mistake in the offset arithmetic would show up immediately.

!!! warning "[COMMON TRAP] Reusing Section 6.1's total-count trick incorrectly across blocks"
    Section 6.1 captured a single block's total kept-count directly from the scan's own root value. It is tempting to assume block `k`'s CORRECT global count is available the same way from kernel 1 alone. It is not: kernel 1 only knows block `k`'s LOCAL count, with no visibility into any other block's contribution — exactly Chapter 4.3's point about blocks having no way to synchronize with each other mid-kernel. Getting each block's correct GLOBAL starting offset genuinely requires kernel 2's separate scan over all blocks' local counts; there is no way to shortcut it from within kernel 1 alone.

## 6.3 Stable Partitioning: Keeping Both Sides

### Intuition

Sections 6.1 and 6.2 discarded elements that failed the predicate. A stable PARTITION keeps everything instead, splitting the array into two groups: elements that pass the predicate, in their original order, followed by elements that fail it, also in their original order. This is not a new algorithm — it is Section 6.1's scan run twice, once on a "keep" flag array and once on a "discard" flag array, with the discard group's positions starting right after the keep group's own total count.

### Background

The kernel below factors the exclusive-scan shape into a small reusable device function specifically because this section needs it twice on two different flag arrays in the same kernel. Every element scatters to one of two possible destinations: the kept scan's value directly, if it passed; or the total kept count plus the discard scan's value, if it didn't. Part 3's radix sort will call this exact same operation once per bit of a number being sorted, separating elements with a 0 in the current digit from elements with a 1.

```cpp
#include <cstdio>
#include <vector>

// Chapter 6.3 -- Sections 6.1-6.2 DISCARDED elements that failed the
// predicate. A stable partition keeps everything, splitting the array
// into two groups instead: elements that pass the predicate first (in
// their original relative order), followed by elements that fail it
// (also in their original relative order). This is not a new algorithm
// so much as running Section 6.1's exact scan twice -- once on the
// "keep" flags, once on the "discard" flags -- and placing the discard
// group's output positions right after the keep group's own total. Part
// 3's radix sort reuses exactly this pattern once per bit, to separate
// elements with a 0 in the current digit from elements with a 1.

#define BLOCK_SIZE 256

__device__ __forceinline__ bool keep_predicate(float v) {
    int iv = (int)v;
    return (iv % 3) == 0;
}

// A single, reusable exclusive-scan-in-shared-memory helper, applied
// twice below to two DIFFERENT flag arrays -- Section 6.1's shape,
// factored out because this section genuinely needs it twice.
__device__ void exclusive_scan_shared(float* data, int n, int tid, float* out_total) {
    int offset = 1;
    for (int d = n >> 1; d > 0; d >>= 1) {
        __syncthreads();
        if (tid < d) {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            data[bi] += data[ai];
        }
        offset *= 2;
    }
    __syncthreads();
    if (tid == 0) { *out_total = data[n - 1]; data[n - 1] = 0.0f; }
    for (int d = 1; d < n; d *= 2) {
        offset >>= 1;
        __syncthreads();
        if (tid < d) {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            float t = data[ai];
            data[ai] = data[bi];
            data[bi] += t;
        }
    }
    __syncthreads();
}

__global__ void stable_partition(const float* g_in, float* g_out, int n) {
    __shared__ float vals[BLOCK_SIZE];
    __shared__ float keep_flags[BLOCK_SIZE];
    __shared__ float discard_flags[BLOCK_SIZE];
    __shared__ float total_kept;
    __shared__ float total_discarded;
    int tid = threadIdx.x;

    vals[tid] = g_in[tid];
    bool keep = keep_predicate(vals[tid]);
    keep_flags[tid] = keep ? 1.0f : 0.0f;
    discard_flags[tid] = keep ? 0.0f : 1.0f;
    __syncthreads();

    exclusive_scan_shared(keep_flags, n, tid, &total_kept);
    exclusive_scan_shared(discard_flags, n, tid, &total_discarded);

    if (keep) {
        g_out[(int)keep_flags[tid]] = vals[tid];
    } else {
        g_out[(int)total_kept + (int)discard_flags[tid]] = vals[tid];
    }
}

// ---- Host-side simulation of the identical two-scan arithmetic ----

std::vector<float> exclusive_scan_host(std::vector<float> data, float* out_total) {
    int n = (int)data.size();
    int offset = 1;
    for (int d = n >> 1; d > 0; d >>= 1) {
        for (int tid = 0; tid < d; tid++) {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            data[bi] += data[ai];
        }
        offset *= 2;
    }
    *out_total = data[n - 1];
    data[n - 1] = 0.0f;
    for (int d = 1; d < n; d *= 2) {
        offset >>= 1;
        for (int tid = 0; tid < d; tid++) {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            float t = data[ai];
            data[ai] = data[bi];
            data[bi] += t;
        }
    }
    return data;
}

std::vector<float> simulate_stable_partition(const std::vector<float>& in) {
    int n = (int)in.size();
    std::vector<float> keep_flags(n), discard_flags(n);
    for (int i = 0; i < n; i++) {
        bool keep = (((int)in[i]) % 3) == 0;
        keep_flags[i] = keep ? 1.0f : 0.0f;
        discard_flags[i] = keep ? 0.0f : 1.0f;
    }
    float total_kept, total_discarded;
    auto keep_scan = exclusive_scan_host(keep_flags, &total_kept);
    auto discard_scan = exclusive_scan_host(discard_flags, &total_discarded);

    std::vector<float> out(n);
    for (int i = 0; i < n; i++) {
        bool keep = (((int)in[i]) % 3) == 0;
        if (keep) out[(int)keep_scan[i]] = in[i];
        else out[(int)total_kept + (int)discard_scan[i]] = in[i];
    }
    return out;
}

std::vector<float> reference_stable_partition(const std::vector<float>& in) {
    std::vector<float> out;
    for (float v : in) if (((int)v % 3) == 0) out.push_back(v);
    for (float v : in) if (((int)v % 3) != 0) out.push_back(v);
    return out;
}

int main() {
    printf("=== Section 6.3: stable partition, both sides kept ===\n\n");

    const int N = BLOCK_SIZE;
    std::vector<float> vals(N);
    for (int i = 0; i < N; i++) vals[i] = (float)i;

    auto result = simulate_stable_partition(vals);
    auto ref = reference_stable_partition(vals);

    bool correct = (result.size() == ref.size());
    if (correct) {
        for (size_t i = 0; i < ref.size(); i++) {
            if (result[i] != ref[i]) { correct = false; break; }
        }
    }

    printf("N = %d, predicate = 'value is a multiple of 3'\n", N);
    printf("first 6 outputs (kept group, front):      ");
    for (int i = 0; i < 6; i++) printf("%.0f ", result[i]);
    printf("\nvalues straddling kept->discarded boundary (indices 83-88):\n  ");
    for (int i = 83; i <= 88; i++) printf("%.0f ", result[i]);
    printf("\nlast 6 outputs (discarded group, back):   ");
    for (int i = N - 6; i < N; i++) printf("%.0f ", result[i]);
    printf("\n\n");

    printf("matches independent two-pass reference partition exactly: %s\n",
           correct ? "yes" : "NO -- BUG");

    // Both groups individually preserve their ORIGINAL relative order --
    // check this directly, not just overall array equality.
    bool kept_ordered = true, discarded_ordered = true;
    float last_kept = -1.0f, last_discarded = -1.0f;
    for (int i = 0; i < 86; i++) {
        if (result[i] <= last_kept) kept_ordered = false;
        last_kept = result[i];
    }
    for (int i = 86; i < N; i++) {
        if (result[i] <= last_discarded) discarded_ordered = false;
        last_discarded = result[i];
    }

    bool ok = correct && kept_ordered && discarded_ordered;
    printf("\nself-check: partition correct, both the kept group and the discarded group\n");
    printf("independently preserve their original relative order: %s\n", ok ? "confirmed" : "MISMATCH");
    return ok ? 0 : 1;
}
```

Running this program produces:

```text
=== Section 6.3: stable partition, both sides kept ===

N = 256, predicate = 'value is a multiple of 3'
first 6 outputs (kept group, front):      0 3 6 9 12 15 
values straddling kept->discarded boundary (indices 83-88):
  249 252 255 1 2 4 
last 6 outputs (discarded group, back):   247 248 250 251 253 254 

matches independent two-pass reference partition exactly: yes

self-check: partition correct, both the kept group and the discarded group
independently preserve their original relative order: confirmed
```

Both groups independently preserve their original relative order — verified directly, not just as an overall array match — and the boundary between them lands exactly where the kept group's own total count says it should.

!!! warning "[COMMON TRAP] Assuming a partition's two groups can share one flag array"
    It is tempting to compute only the "keep" flags and derive the discard group's positions as `i - keep_scan[i]` (position minus how many kept elements came before it), reasoning that keep and discard flags are just complements of each other. This actually works arithmetically for the SIMPLE two-way case Section 6.3 builds — but it stops generalizing the moment a future chapter needs a partition into more than two groups (radix sort's later, multi-bit-at-once variants sort into 4 or 16 buckets per pass, not 2). Scanning a separate flag array per group, as this section does even for the two-group case, is the pattern that generalizes; the arithmetic shortcut for exactly two groups is a special case worth recognizing, not the general technique worth building the habit around.

## Chapter Summary

Section 6.1 showed stream compaction is Chapter 5's exclusive scan applied to a 0/1 predicate-flag array: the scan value at each kept position is already that element's correct output slot, verified against an independent reference for 86 multiples of 3 in 256 elements. Section 6.2 generalized this past one block with the identical three-kernel pattern Chapter 5.3 used for scan — local compaction, a scan of per-block counts, and a stitching kernel — verified across every block boundary for 2048 elements. Section 6.3 extended compaction to a stable two-way partition by running the same scan twice, on complementary flag arrays, foreshadowing exactly the operation Part 3's radix sort performs once per bit. Chapter 7 completes Part 1 with histograms — counting how many elements fall into each of several buckets, a problem that looks like compaction's opposite (many elements can land in the SAME bucket) but turns out to share real structure with it.

## Self-Check Questions

1. Section 6.1's predicate keeps multiples of 3. If the predicate instead kept multiples of 4 over the same `[0, 255]` range, how many elements would the compacted output contain?
2. Explain concretely, using Section 6.1's own scatter step (`g_out[(int)flags[tid]] = vals[tid]`), why an INCLUSIVE scan of the same flags would place every kept element one position too far to the right.
3. Section 6.2's kernel 1 records `g_block_counts[blockIdx.x]` from the LOCAL scan's root value, before zeroing it. Why can this local count NOT simply be summed directly (without a further scan) to get block 5's correct global starting offset?
4. Using Section 6.2's own per-block local kept-counts (86, 85, 85, 86, 85, 85, 86, 85), verify by hand that the reported global offset for block 4 (342) is correct.
5. Section 6.3 warns that the `i - keep_scan[i]` shortcut for the discard group's position does not generalize past two groups. Explain concretely what breaks if a predicate instead sorted elements into 4 groups (based on, say, `value % 4`) using an analogous shortcut.
6. Both the kept and discarded groups in Section 6.3's output are verified to independently preserve their original relative order. Explain why this specific property — not just "the right elements ended up in the right group" — is what makes a partition STABLE.

## Where We Go Next

Chapter 7 completes Part 1 with histograms: counting how many elements fall into each of several buckets. It looks like the opposite of compaction (many elements can legitimately share one destination, rather than each needing a unique one), but this chapter's exclusive-scan machinery reappears there too, in a genuinely different role — turning per-bucket counts into per-bucket starting offsets, the same problem Section 6.2 just solved for per-block counts.

## Worked Solutions

**1.** Multiples of 4 in `[0, 255]` inclusive: `floor(255/4) + 1 = 63 + 1 = 64` elements (0, 4, 8, ..., 252).

**2.** An inclusive scan's value at position `i` already includes position `i`'s OWN flag in the running total. For a kept element (`flag[i] = 1`), the inclusive scan value at `i` is `(exclusive scan value at i) + 1` -- one more than the correct zero-based output slot Section 6.1's exclusive scan produces, which would place every kept element exactly one slot to the right of where it belongs (and, for the very first kept element, write to output index 1 instead of 0, leaving index 0 permanently empty).

**3.** Kernel 1 has no visibility into any other block's data or results — it is a single, isolated block that can only ever know its OWN local count (Chapter 4.3's own point about blocks having no way to synchronize with each other mid-kernel). Block 5's correct global offset needs the SUM of every block-0-through-4's local count, which requires reading all of THEIR local counts too — information that does not exist anywhere kernel 1's single block invocation could reach, which is exactly why kernel 2 is a separate, later launch operating over the complete array of all blocks' local counts at once.

**4.** Block 4's offset should be the sum of every earlier block's local count: blocks 0-3, with counts 86, 85, 85, 86. Sum: `86 + 85 + 85 + 86 = 342`, matching the reported offset exactly.

**5.** The two-group shortcut works because there are exactly two possible destinations, and knowing "how many elements before me are in the OTHER group" is recoverable by subtraction (`i - keep_scan[i]`) precisely because every element is in one group or the other — no third option. With 4 groups (`value % 4` giving group 0, 1, 2, or 3), knowing how many elements before position `i` belong to group 2, specifically, cannot be recovered by subtracting group 2's own scan from `i`, because `i` also includes elements from groups 0, 1, and 3 that the subtraction has no way to separate out — each of the 4 groups genuinely needs its OWN scanned flag array, exactly as Section 6.3's warning states.

**6.** A partition could place the "right" elements in the "right" group while still scrambling their order within that group (for instance, writing kept elements to their correct slots but in reverse order) — this would still correctly separate the two groups but would not be STABLE. Stability specifically means each group, examined on its own, appears in the same relative order the elements had in the original array — the property Section 6.3 checks directly and separately from overall correctness, because a partition can satisfy one without the other.
