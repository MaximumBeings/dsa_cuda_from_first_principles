# Chapter 5: Scan (Prefix Sum)

**What you will understand by the end of this chapter:**

- Why scan is a genuinely harder problem than reduction, even though both look like "combine everything with addition" at first glance.
- How the Hillis-Steele scan gets a first, correct parallel answer, and why its work is O(n log n) rather than O(n) despite matching reduction's span exactly.
- How Blelloch's two-sweep, work-efficient scan trades that away for a real, measured span cost, and how to combine per-block scans into one correct multi-block result.

**What you need to know first:**

- Chapter 4 in full: a real reduction kernel, its divergence-aware rewrite, and its two-phase multi-block generalization.
- Chapter 3's work-span vocabulary, which this chapter uses to compare three genuinely different scan designs on equal footing.

---

Reduction collapses `n` elements to one value. Scan (also called prefix sum) asks for something that looks similar but is not: an output at EVERY position, each one the running total of everything up to and including it. This chapter builds three working versions — a straightforward one, a work-efficient one, and one that scales past a single block — and measures, rather than assumes, what each one actually costs.

## 5.1 What Scan Computes, and the Hillis-Steele Algorithm

### Intuition

An inclusive scan turns `[3, 1, 4, 1, 5]` into `[3, 4, 8, 9, 14]` — each output position is the sum of every input up to and including that position. Chapter 4's reduction tree does not directly give this: its intermediate levels hold partial sums of specific subranges on the way to one final total, not a running total ending at every position. Getting every position's own running total needs a different access pattern.

### The Sequential (CPU) Baseline

A CPU computes an inclusive scan with exactly one pass and one running variable:

```
void scan_inclusive_cpu(const float* in, float* out, int n) {
    float running = 0.0f;
    for (int i = 0; i < n; i++) {
        running += in[i];
        out[i] = running;
    }
}
```

O(n) work, not O(n log n), and no notion of "steps" or double-buffering at all, because each position's answer is trivially available the moment the position before it has been processed. The GPU version below needs `log2(n)` STEPS and touches nearly every position at EACH step specifically because no single GPU thread can see every earlier position's running total the way one sequential loop naturally can — the extra work is the price of turning one sequential dependency chain into something many threads can execute at once.

### The Concept, In Detail

The Hillis-Steele scan gets there by brute force, correctly: at step `d` (1, 2, 4, ..., up to `n/2`), every position `i >= d` adds the value `d` positions behind it. Traced on a small 8-element example (all ones, so the correct inclusive answer at position `i` is simply `i+1`):

```
Hillis-Steele inclusive scan, n=8, input = [1, 1, 1, 1, 1, 1, 1, 1]

start:                    1  1  1  1  1  1  1  1

d=1  out[i] = in[i] + (i>=1 ? in[i-1] : 0)   -- every position adds the
                                                 value 1 slot behind it
                          1  2  2  2  2  2  2  2

d=2  out[i] = in[i] + (i>=2 ? in[i-2] : 0)   -- every position adds the
                                                 value 2 slots behind it
                          1  2  3  4  4  4  4  4

d=4  out[i] = in[i] + (i>=4 ? in[i-4] : 0)   -- every position adds the
                                                 value 4 slots behind it
                          1  2  3  4  5  6  7  8

log2(8) = 3 steps, and every position now holds its own correct running
total -- not just position 7 (the "reduction" answer), but every one.
```

The kernel double-buffers between two shared-memory arrays (writing this step's results into a fresh array rather than overwriting the one still being read) because a single in-place array would let one thread read a value another thread already overwrote earlier in the very same step — exactly the kind of same-step read-after-write hazard Chapter 2's shared-memory staging already taught this book to watch for.

The cost is visible directly from the diagram: at `d=1`, 7 of the 8 positions do an addition; at `d=2`, 6 positions do; at `d=4`, 4 positions do. Nearly every step touches nearly every position — for `n=256` (this section's actual size), the exact total is `sum of (n-d)` for `d = 1, 2, 4, ..., 128`, which comes to `255+254+252+248+240+224+192+128 = 1793` additions, in `log2(256) = 8` steps.

### Code and Verification

```cpp
#include <cstdio>
#include <vector>

// Chapter 5.1 -- reduction (Chapter 4) collapses n elements to ONE value.
// Scan (prefix sum) asks for something genuinely harder: EVERY output
// position i needs the running total of every position up to and
// including it (an inclusive scan), not just the grand total. A tree
// reduction's shape does not directly give this -- its intermediate
// levels hold partial sums of specific SUBRANGES, not running totals
// ending at every position. The first working parallel scan needs a
// different access pattern, not just reduction's tree reused as-is.

#define BLOCK_SIZE 256

// Hillis-Steele inclusive scan: at each step d (1, 2, 4, ..., n/2), every
// position i >= d adds the value d positions behind it. After log2(n)
// steps, position i holds the sum of positions [i-n+1 .. i] that are
// actually in range -- the full inclusive running total. Double-buffered
// because a naive in-place version would read a value one thread already
// overwrote earlier in the SAME step.
__global__ void scan_hillis_steele(const float* g_in, float* g_out, int n) {
    __shared__ float buf_a[BLOCK_SIZE];
    __shared__ float buf_b[BLOCK_SIZE];
    int tid = threadIdx.x;

    buf_a[tid] = (tid < n) ? g_in[tid] : 0.0f;
    __syncthreads();

    float* src = buf_a;
    float* dst = buf_b;
    for (int d = 1; d < n; d *= 2) {
        if (tid >= d) {
            dst[tid] = src[tid] + src[tid - d];
        } else {
            dst[tid] = src[tid];
        }
        __syncthreads();
        float* tmp = src; src = dst; dst = tmp;
    }
    if (tid < n) g_out[tid] = src[tid];
}

// ---- Host-side simulation of the identical double-buffered arithmetic ----

std::vector<float> simulate_hillis_steele(std::vector<float> vals) {
    int n = (int)vals.size();
    std::vector<float> src = vals, dst = vals;
    for (int d = 1; d < n; d *= 2) {
        for (int tid = 0; tid < n; tid++) {
            dst[tid] = (tid >= d) ? (src[tid] + src[tid - d]) : src[tid];
        }
        src = dst;
    }
    return src;
}

std::vector<float> reference_inclusive_scan(const std::vector<float>& vals) {
    std::vector<float> out(vals.size());
    float running = 0.0f;
    for (size_t i = 0; i < vals.size(); i++) {
        running += vals[i];
        out[i] = running;
    }
    return out;
}

int main() {
    printf("=== Section 5.1: Hillis-Steele inclusive scan ===\n\n");

    std::vector<float> vals(BLOCK_SIZE);
    for (int i = 0; i < BLOCK_SIZE; i++) vals[i] = 1.0f;   // all-ones: out[i] should be i+1

    auto result = simulate_hillis_steele(vals);
    auto ref = reference_inclusive_scan(vals);

    bool correct = true;
    for (int i = 0; i < BLOCK_SIZE; i++) {
        if (result[i] != ref[i]) correct = false;
    }
    printf("block size = %d, input = all ones, so out[i] should equal i+1\n", BLOCK_SIZE);
    printf("matches independent reference inclusive scan at every position: %s\n",
           correct ? "yes" : "NO -- BUG");
    printf("first 8 outputs:  ");
    for (int i = 0; i < 8; i++) printf("%.0f ", result[i]);
    printf("\nlast 8 outputs:   ");
    for (int i = BLOCK_SIZE - 8; i < BLOCK_SIZE; i++) printf("%.0f ", result[i]);
    printf("\n\n");

    // Count genuine total additions (work), and span (number of steps).
    long long total_additions = 0;
    int span = 0;
    for (int d = 1; d < BLOCK_SIZE; d *= 2) {
        total_additions += (BLOCK_SIZE - d);   // positions tid >= d each do one addition
        span++;
    }
    printf("work (total additions) = %lld, span (number of steps) = %d\n", total_additions, span);
    printf("compare to reduction's work for the same n: exactly n-1 = %d additions, in the\n",
           BLOCK_SIZE - 1);
    printf("same log2(n) = %d steps. Scan's span matches reduction's span exactly -- but its\n",
           span);
    printf("work is dramatically higher: every step here does WORK PROPORTIONAL TO N, not\n");
    printf("halving like reduction's tree does, because scan needs an output at every\n");
    printf("position, not just one final value at the root.\n");

    bool ok = correct && (total_additions > (BLOCK_SIZE - 1)) && (span == 8);
    printf("\nself-check: scan correct, work (%lld) exceeds reduction's work (%d) for the\n",
           total_additions, BLOCK_SIZE - 1);
    printf("same n, span = 8: %s\n", ok ? "confirmed" : "MISMATCH");
    return ok ? 0 : 1;
}
```

**Compile and run:**

```bash
nvcc -arch=sm_80 13_hillis_steele_scan.cu -o hillis_steele_scan
./hillis_steele_scan
```

**Sample input:** `BLOCK_SIZE = 256` elements, all set to `1.0`, so the correct inclusive scan is simply `1, 2, 3, ..., 256`.

**Sample output:**

```text
=== Section 5.1: Hillis-Steele inclusive scan ===

block size = 256, input = all ones, so out[i] should equal i+1
matches independent reference inclusive scan at every position: yes
first 8 outputs:  1 2 3 4 5 6 7 8 
last 8 outputs:   249 250 251 252 253 254 255 256 

work (total additions) = 1793, span (number of steps) = 8
compare to reduction's work for the same n: exactly n-1 = 255 additions, in the
same log2(n) = 8 steps. Scan's span matches reduction's span exactly -- but its
work is dramatically higher: every step here does WORK PROPORTIONAL TO N, not
halving like reduction's tree does, because scan needs an output at every
position, not just one final value at the root.

self-check: scan correct, work (1793) exceeds reduction's work (255) for the
same n, span = 8: confirmed
```

Correct at every position, matched against an independent sequential scan. But the work tells a different story than reduction did: 1793 additions for 256 elements, against reduction's 255 for the identical `n` — over 7x more, in the identical `log2(n) = 8` steps.

!!! warning "[COMMON TRAP] Assuming scan and reduction cost the same because their spans match"
    Both algorithms need `log2(n)` steps — Chapter 3's span vocabulary would call them span-equivalent, and it is tempting to stop the comparison there. Reduction's span comes cheap: each level does HALF the work of the level before, so total work stays `O(n)`. Hillis-Steele's span comes expensive: every level still touches nearly every position, because every position needs its own answer, not just the tree's root. Matching spans do not imply matching costs — Section 5.2 shows the work Hillis-Steele leaves on the table can, in fact, be recovered, but not for free.

## 5.2 Blelloch's Work-Efficient Scan: A Genuine Trade, Not a Free Improvement

### Intuition

Section 5.1's O(n log n) work comes from touching nearly every position at every step. Blelloch's scan instead builds an EXPLICIT binary tree over the data in two sweeps: an up-sweep that is exactly Chapter 4's reduction (each level halves the active count, same shape, same O(n) total work), then a down-sweep that distributes the tree's stored partial sums back down, turning "just the total" into a running total at every position.

### The Sequential (CPU) Baseline

The exclusive version is the identical one-pass CPU loop as Section 5.1's baseline, just writing the running total BEFORE adding the current element instead of after:

```
void scan_exclusive_cpu(const float* in, float* out, int n) {
    float running = 0.0f;
    for (int i = 0; i < n; i++) {
        out[i] = running;
        running += in[i];
    }
}
```

Still O(n) work, still no up-sweep or down-sweep of any kind. Blelloch's two-sweep tree exists purely to recover this same O(n) work bound in a form that many GPU threads can execute in O(log n) steps; a single CPU thread never needed a tree to begin with, since it already gets O(n) work for free from one straightforward loop.

### The Concept, In Detail

Traced by hand on the same 8-element, all-ones example makes both sweeps concrete. The up-sweep is a reduction tree, exactly Chapter 4's shape, just written with explicit tree indices instead of a shared-memory halving loop:

```
UP-SWEEP (identical shape to Chapter 4's reduction tree), n=8:

start:                          1  1  1  1  1  1  1  1

d=4 (stride 1): combine pairs (0,1) (2,3) (4,5) (6,7)
                                1  2  1  2  1  2  1  2

d=2 (stride 2): combine pairs (1,3) (5,7)
                                1  2  1  4  1  2  1  4

d=1 (stride 4): combine pair (3,7)
                                1  2  1  4  1  2  1  8   <- index 7 = TRUE TOTAL

zero the last element (addition's identity element) to start an
EXCLUSIVE scan:                1  2  1  4  1  2  1  0
```

The down-sweep then walks the identical tree structure backward. At each node the up-sweep visited, the LEFT child receives the parent's OLD value, and the RIGHT child receives the parent's old value PLUS what the left child just received:

```
DOWN-SWEEP (walking the same tree backward), continuing from above:

d=1 (stride 4): at the pair (3,7): left=3 gets old right(7)=0,
                right=7 gets old right(7) + old left(3) = 0+4=4
                                1  2  1  0  1  2  1  4

d=2 (stride 2): at pairs (1,3) and (5,7):
                                1  0  1  2  1  4  1  6

d=4 (stride 1): at pairs (0,1) (2,3) (4,5) (6,7):
                                0  1  2  3  4  5  6  7   <- correct exclusive scan
```

Every position now holds the sum of everything strictly BEFORE it, matching the exclusive-scan definition exactly. The up-sweep alone already matches Hillis-Steele's entire span (`log2(n)` levels); the down-sweep adds a second, equally deep pass on top — doubling the span in exchange for cutting the work down to barely more than reduction's own `O(n)`.

### Code and Verification

```cpp
#include <cstdio>
#include <vector>

// Chapter 5.2 -- Section 5.1's Hillis-Steele scan is correct but does
// O(n log n) total additions for an O(n)-sized problem, because every
// one of its log2(n) steps touches nearly every position. Blelloch's
// work-efficient scan instead builds an implicit binary tree over the
// data in TWO sweeps: an up-sweep that is exactly Chapter 4's reduction
// (each level halves the number of active positions, same as before),
// followed by a down-sweep that distributes the tree's partial sums back
// down to produce an EXCLUSIVE prefix at every position. This section
// measures both the work AND the span this trade costs, rather than
// assuming one improves for free.
__global__ void scan_blelloch_exclusive(const float* g_in, float* g_out, int n) {
    __shared__ float temp[256];
    int tid = threadIdx.x;
    temp[tid] = (tid < n) ? g_in[tid] : 0.0f;
    __syncthreads();

    // Up-sweep (reduce): identical shape to Chapter 4's tree reduction --
    // each level halves the number of active threads.
    int offset = 1;
    for (int d = n >> 1; d > 0; d >>= 1) {
        __syncthreads();
        if (tid < d) {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            temp[bi] += temp[ai];
        }
        offset *= 2;
    }

    __syncthreads();
    if (tid == 0) temp[n - 1] = 0.0f;   // identity element, for an EXCLUSIVE scan

    // Down-sweep: distributes each node's stored partial sum back down to
    // its two children -- the left child gets the parent's old value, the
    // right child gets the parent's old value plus the left child's own
    // old value. This is what actually turns "just the total" back into
    // "a running total at every position."
    for (int d = 1; d < n; d *= 2) {
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
    if (tid < n) g_out[tid] = temp[tid];
}

// ---- Host-side simulation of the identical two-sweep arithmetic ----

std::vector<float> simulate_blelloch(std::vector<float> vals) {
    int n = (int)vals.size();
    std::vector<float> temp = vals;

    int offset = 1;
    for (int d = n >> 1; d > 0; d >>= 1) {
        for (int tid = 0; tid < d; tid++) {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            temp[bi] += temp[ai];
        }
        offset *= 2;
    }

    temp[n - 1] = 0.0f;
    for (int d = 1; d < n; d *= 2) {
        offset >>= 1;
        for (int tid = 0; tid < d; tid++) {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            float t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }
    return temp;
}

std::vector<float> reference_exclusive_scan(const std::vector<float>& vals) {
    std::vector<float> out(vals.size());
    float running = 0.0f;
    for (size_t i = 0; i < vals.size(); i++) {
        out[i] = running;
        running += vals[i];
    }
    return out;
}

int main() {
    printf("=== Section 5.2: Blelloch's work-efficient exclusive scan ===\n\n");

    const int N = 256;
    std::vector<float> vals(N, 1.0f);   // all-ones: exclusive out[i] should equal i

    auto result = simulate_blelloch(vals);
    auto ref = reference_exclusive_scan(vals);
    bool correct = true;
    for (int i = 0; i < N; i++) if (result[i] != ref[i]) correct = false;

    printf("N = %d, input = all ones, so exclusive out[i] should equal i\n", N);
    printf("matches independent reference exclusive scan at every position: %s\n",
           correct ? "yes" : "NO -- BUG");
    printf("first 8 outputs:  ");
    for (int i = 0; i < 8; i++) printf("%.0f ", result[i]);
    printf("\nlast 8 outputs:   ");
    for (int i = N - 8; i < N; i++) printf("%.0f ", result[i]);
    printf("\n\n");

    // Genuine work and span counts for both sweeps.
    long long upsweep_additions = 0;
    int upsweep_span = 0;
    for (int d = N >> 1; d > 0; d >>= 1) { upsweep_additions += d; upsweep_span++; }

    long long downsweep_additions = 0;
    int downsweep_span = 0;
    for (int d = 1; d < N; d *= 2) { downsweep_additions += d; downsweep_span++; }

    long long total_additions = upsweep_additions + downsweep_additions;
    int total_span = upsweep_span + downsweep_span;

    printf("up-sweep:   %lld additions across %d levels (identical shape to Chapter 4's\n",
           upsweep_additions, upsweep_span);
    printf("            reduction tree for the same N)\n");
    printf("down-sweep: %lld additions across %d levels\n", downsweep_additions, downsweep_span);
    printf("total work = %lld additions, total span = %d levels\n\n", total_additions, total_span);

    const long long hillis_steele_work = 1793;
    const int hillis_steele_span = 8;
    printf("Section 5.1's Hillis-Steele scan, same N: work = %lld, span = %d\n",
           hillis_steele_work, hillis_steele_span);
    printf("Blelloch's work-efficient scan, same N:   work = %lld, span = %d\n",
           total_additions, total_span);
    printf("work changes by %.2fx (less), span changes by %.2fx (MORE) -- this is a genuine\n",
           (double)hillis_steele_work / (double)total_additions,
           (double)total_span / (double)hillis_steele_span);
    printf("trade, not a strict improvement: Blelloch's two sequential sweeps double the\n");
    printf("number of synchronization levels needed, in exchange for doing barely more\n");
    printf("than reduction's own O(n) work instead of Hillis-Steele's O(n log n).\n");

    bool ok = correct && (total_additions < hillis_steele_work) && (total_span > hillis_steele_span);
    printf("\nself-check: exclusive scan correct, work strictly less than Hillis-Steele's,\n");
    printf("span strictly more: %s\n", ok ? "confirmed" : "MISMATCH");
    return ok ? 0 : 1;
}
```

**Compile and run:**

```bash
nvcc -arch=sm_80 14_blelloch_work_efficient_scan.cu -o blelloch_scan
./blelloch_scan
```

**Sample input:** `N = 256` elements, all set to `1.0`, so the correct exclusive scan is simply `0, 1, 2, ..., 255`.

**Sample output:**

```text
=== Section 5.2: Blelloch's work-efficient exclusive scan ===

N = 256, input = all ones, so exclusive out[i] should equal i
matches independent reference exclusive scan at every position: yes
first 8 outputs:  0 1 2 3 4 5 6 7 
last 8 outputs:   248 249 250 251 252 253 254 255 

up-sweep:   255 additions across 8 levels (identical shape to Chapter 4's
            reduction tree for the same N)
down-sweep: 255 additions across 8 levels
total work = 510 additions, total span = 16 levels

Section 5.1's Hillis-Steele scan, same N: work = 1793, span = 8
Blelloch's work-efficient scan, same N:   work = 510, span = 16
work changes by 3.52x (less), span changes by 2.00x (MORE) -- this is a genuine
trade, not a strict improvement: Blelloch's two sequential sweeps double the
number of synchronization levels needed, in exchange for doing barely more
than reduction's own O(n) work instead of Hillis-Steele's O(n log n).

self-check: exclusive scan correct, work strictly less than Hillis-Steele's,
span strictly more: confirmed
```

Correct against an independent exclusive-scan reference, and the comparison against Section 5.1 is direct and genuine: 3.52x less work, but exactly 2x more span. Blelloch's two sweeps are each `log2(n)` levels — the up-sweep alone matches Hillis-Steele's entire span, and the down-sweep doubles it.

!!! warning "[COMMON TRAP] Assuming 'work-efficient' means 'strictly better'"
    Both this section's own self-check and its name warn against exactly this: "work-efficient" describes ONE axis, not overall superiority. A machine with abundant spare parallelism and a memory system that is the real bottleneck benefits from Blelloch's lower work despite its higher span; a machine (or a problem size) where synchronization overhead dominates might do better with Hillis-Steele's shorter span despite its higher work. Neither algorithm is "the" right scan — Chapter 3's whole point was that work and span are independent measurements, and this section is the clearest demonstration yet of a real trade-off between them rather than a strict win.

## 5.3 Multi-Block Scan: Combining Per-Block Results

### Intuition

Sections 5.1 and 5.2 scanned exactly one block. A real scan needs `N` far larger than one block holds — and unlike reduction, where combining per-block partial sums is a second small reduction, scan's per-block results need CORRECTING, not just combining: block 1's local scan is only correct relative to block 1's own start, and every one of its values needs block 0's total added on top to be correct for the whole array.

### The Sequential (CPU) Baseline

Exactly like Chapter 4.3, the single-pass CPU loop from Section 5.2 handles `N = 2048` (or any larger `N`) without modification — a CPU scan never needs to know about "blocks" at all. This section's three-kernel design (local scan, scan-of-totals, offset-broadcast) exists entirely to work around a single GPU block's limited shared memory and thread count; it recovers the identical answer the one-line CPU loop already computes trivially, just structured so many blocks can each do a bounded piece of the work.

### The Concept, In Detail

Three kernels, reusing Section 5.2's exact exclusive-scan shape twice — once on the real data, once on a much shorter array of per-block totals:

```
N = 2048 elements across NUM_BLOCKS = 8 blocks of 256 elements each

KERNEL 1 (8 blocks, run independently): each block exclusive-scans its
OWN 256 elements (Section 5.2's exact shape), separately recording its
own TRUE total -- captured right before the up-sweep's last element gets
zeroed for the exclusive-scan identity:

  block totals:  1774  1790  1806  1783  1786  1802  1792  1782

each block's own 256 outputs are only correct RELATIVE TO THAT BLOCK's
own start -- block 1's values still need block 0's entire total (1774)
added on top before they are correct for the whole 2048-element array.

KERNEL 2 (1 block, over just the 8 block totals): exclusive-scan those
totals with the IDENTICAL scan shape, turning each block's total into
the OFFSET that block's own results need:

  block totals:  1774  1790  1806  1783  1786  1802  1792  1782
  offsets:          0  1774  3564  5370  7153  8939 10741 12533
  (block b's offset = sum of every block's total strictly before b --
  scan's own definition, applied one level up)

KERNEL 3 (8 blocks again): add each block's own offset into every one
of its 256 local results -- every one of block 3's 256 values gets
+5370, every one of block 7's gets +12533, and so on.
```

The three kernels are not three different algorithms — kernels 1 and 2 run the literal same scan shape, just at two different scales (2048 elements split into 8 groups of 256, then those 8 group totals scanned directly), and kernel 3 is a simple broadcast-add.

### Code and Verification

```cpp
#include <cstdio>
#include <vector>

// Chapter 5.3 -- Sections 5.1-5.2 scanned exactly one block's worth of
// data. A real scan, like Chapter 4's reduction, has to handle N larger
// than one block. The classic three-kernel design reuses Section 5.2's
// exclusive scan TWICE: once per block on the actual data, and once on
// the (much shorter) array of per-block totals -- then a third kernel
// adds each block's scanned offset back into its own local results.

#define BLOCK_SIZE 256

// Kernel 1: each block exclusive-scans its own BLOCK_SIZE elements
// (Section 5.2's up-sweep/down-sweep, unchanged), but ALSO records its
// pre-zeroed total into g_block_sums[blockIdx.x] -- the one piece of
// information kernel 3 will need to correct every element in this block.
__global__ void scan_block_local(const float* g_in, float* g_out, float* g_block_sums, int n) {
    __shared__ float temp[BLOCK_SIZE];
    int tid = threadIdx.x;
    int base = blockIdx.x * n;
    temp[tid] = (tid < n) ? g_in[base + tid] : 0.0f;
    __syncthreads();

    int offset = 1;
    for (int d = n >> 1; d > 0; d >>= 1) {
        __syncthreads();
        if (tid < d) {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            temp[bi] += temp[ai];
        }
        offset *= 2;
    }

    __syncthreads();
    if (tid == 0) {
        g_block_sums[blockIdx.x] = temp[n - 1];   // this block's true total, before zeroing
        temp[n - 1] = 0.0f;
    }

    for (int d = 1; d < n; d *= 2) {
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
    if (tid < n) g_out[base + tid] = temp[tid];
}

// Kernel 2: exclusive-scan the (short) array of block totals -- the
// IDENTICAL kernel shape as kernel 1, just launched once, on a single
// block, over NUM_BLOCKS elements instead of BLOCK_SIZE.
__global__ void scan_block_sums(float* g_block_sums, int num_blocks) {
    __shared__ float temp[BLOCK_SIZE];
    int tid = threadIdx.x;
    temp[tid] = (tid < num_blocks) ? g_block_sums[tid] : 0.0f;
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
    if (tid < num_blocks) g_block_sums[tid] = temp[tid];   // now holds each block's OFFSET
}

// Kernel 3: add each block's now-scanned offset into every element of
// that block's own local scan result -- turning "correct within this
// block" into "correct across the whole array."
__global__ void add_block_offsets(float* g_out, const float* g_block_offsets, int n) {
    int tid = threadIdx.x;
    int base = blockIdx.x * n;
    g_out[base + tid] += g_block_offsets[blockIdx.x];
}

// ---- Host-side simulation of the identical three-kernel arithmetic ----

void simulate_scan_block_local(const std::vector<float>& in, std::vector<float>& out,
                                std::vector<float>& block_sums, int num_blocks, int n) {
    for (int block = 0; block < num_blocks; block++) {
        std::vector<float> temp(n);
        for (int i = 0; i < n; i++) temp[i] = in[block * n + i];

        int offset = 1;
        for (int d = n >> 1; d > 0; d >>= 1) {
            for (int tid = 0; tid < d; tid++) {
                int ai = offset * (2 * tid + 1) - 1;
                int bi = offset * (2 * tid + 2) - 1;
                temp[bi] += temp[ai];
            }
            offset *= 2;
        }
        block_sums[block] = temp[n - 1];
        temp[n - 1] = 0.0f;
        for (int d = 1; d < n; d *= 2) {
            offset >>= 1;
            for (int tid = 0; tid < d; tid++) {
                int ai = offset * (2 * tid + 1) - 1;
                int bi = offset * (2 * tid + 2) - 1;
                float t = temp[ai];
                temp[ai] = temp[bi];
                temp[bi] += t;
            }
        }
        for (int i = 0; i < n; i++) out[block * n + i] = temp[i];
    }
}

void simulate_scan_block_sums(std::vector<float>& block_sums, int num_blocks) {
    std::vector<float> temp = block_sums;
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
    block_sums = temp;
}

std::vector<float> reference_exclusive_scan(const std::vector<float>& vals) {
    std::vector<float> out(vals.size());
    float running = 0.0f;
    for (size_t i = 0; i < vals.size(); i++) {
        out[i] = running;
        running += vals[i];
    }
    return out;
}

int main() {
    printf("=== Section 5.3: multi-block scan, three kernels ===\n\n");

    const int NUM_BLOCKS = 8;
    const int N = NUM_BLOCKS * BLOCK_SIZE;   // 2048 total elements
    std::vector<float> vals(N);
    for (int i = 0; i < N; i++) vals[i] = (float)((i % 13) + 1);   // varied, deterministic values

    std::vector<float> out(N, 0.0f);
    std::vector<float> block_sums(NUM_BLOCKS, 0.0f);

    simulate_scan_block_local(vals, out, block_sums, NUM_BLOCKS, BLOCK_SIZE);
    printf("kernel 1: %d blocks each exclusive-scan their own %d elements\n", NUM_BLOCKS, BLOCK_SIZE);
    printf("per-block totals (before scan 2): ");
    for (float s : block_sums) printf("%.0f ", s);
    printf("\n\n");

    simulate_scan_block_sums(block_sums, NUM_BLOCKS);
    printf("kernel 2: exclusive-scan the %d block totals themselves\n", NUM_BLOCKS);
    printf("per-block offsets (after scan 2):  ");
    for (float s : block_sums) printf("%.0f ", s);
    printf("\n\n");

    for (int block = 0; block < NUM_BLOCKS; block++) {
        for (int i = 0; i < BLOCK_SIZE; i++) {
            out[block * BLOCK_SIZE + i] += block_sums[block];
        }
    }
    printf("kernel 3: add each block's offset into its own %d local results\n\n", BLOCK_SIZE);

    auto ref = reference_exclusive_scan(vals);
    bool correct = true;
    int first_mismatch = -1;
    for (int i = 0; i < N; i++) {
        if (out[i] != ref[i]) { correct = false; if (first_mismatch < 0) first_mismatch = i; }
    }

    printf("N = %d across %d blocks, matches independent single-pass reference exclusive\n", N, NUM_BLOCKS);
    printf("scan at every position: %s\n", correct ? "yes" : "NO -- BUG");
    if (!correct) printf("first mismatch at index %d\n", first_mismatch);

    printf("\nvalues straddling a block boundary (indices %d-%d, around block 1/2's edge):\n",
           BLOCK_SIZE - 3, BLOCK_SIZE + 2);
    for (int i = BLOCK_SIZE - 3; i <= BLOCK_SIZE + 2; i++) {
        printf("  out[%4d] = %6.0f   reference = %6.0f\n", i, out[i], ref[i]);
    }

    bool ok = correct;
    printf("\nself-check: three-kernel multi-block scan matches the independent reference\n");
    printf("across all %d elements, including every block boundary: %s\n", N, ok ? "confirmed" : "MISMATCH");
    return ok ? 0 : 1;
}
```

**Compile and run:**

```bash
nvcc -arch=sm_80 15_multi_block_scan.cu -o multi_block_scan
./multi_block_scan
```

**Sample input:** `N = 2048` elements, all set to values in `[1, 8]` deterministically varied by position, launched as `NUM_BLOCKS = 8` blocks of 256 elements each.

**Sample output:**

```text
=== Section 5.3: multi-block scan, three kernels ===

kernel 1: 8 blocks each exclusive-scan their own 256 elements
per-block totals (before scan 2): 1774 1790 1806 1783 1786 1802 1792 1782 

kernel 2: exclusive-scan the 8 block totals themselves
per-block offsets (after scan 2):  0 1774 3564 5370 7153 8939 10741 12533 

kernel 3: add each block's offset into its own 256 local results

N = 2048 across 8 blocks, matches independent single-pass reference exclusive
scan at every position: yes

values straddling a block boundary (indices 253-258, around block 1/2's edge):
  out[ 253] =   1750   reference =   1750
  out[ 254] =   1757   reference =   1757
  out[ 255] =   1765   reference =   1765
  out[ 256] =   1774   reference =   1774
  out[ 257] =   1784   reference =   1784
  out[ 258] =   1795   reference =   1795

self-check: three-kernel multi-block scan matches the independent reference
across all 2048 elements, including every block boundary: confirmed
```

2048 elements across 8 blocks, matched exactly against an independent single-pass reference at every position — including the values straddling each block boundary, where a bug in the offset arithmetic would show up immediately and did not.

!!! warning "[COMMON TRAP] Forgetting kernel 2 needs the SAME algorithm, not a simpler one"
    It is tempting to combine block totals with something simpler than a full exclusive scan, reasoning that there are far fewer of them. But what kernel 3 needs from kernel 2 is precisely an EXCLUSIVE scan of the block totals — block `k`'s correct offset is the sum of every block's total strictly before it, exactly scan's own definition, just applied one level up. Reusing the identical kernel shape for both levels is not a coincidence or a simplification for exposition; it is the actual, correct algorithm, and it generalizes: a fourth level (scanning offsets of offsets) would use the identical shape again for an array too large even for kernel 2's single block.

## Chapter Summary

Section 5.1's Hillis-Steele scan gave a first correct answer at every position, in the same `log2(n)` span as reduction, but at nearly 7x the work for a 256-element block. Section 5.2's Blelloch scan recovered a 3.52x reduction in that work by reusing reduction's own tree shape for an up-sweep, followed by a down-sweep — at a genuine, measured cost of exactly 2x the span, a real trade rather than a strict improvement. Section 5.3 generalized past one block with a three-kernel design that reuses the identical exclusive-scan shape at two levels — once on the real data, once on the much shorter array of per-block totals — verified exactly against an independent reference across every block boundary. Part 1 continues with stream compaction, which turns out to need exactly this chapter's scan as its own core building block.

## Self-Check Questions

1. For a 512-element block (instead of this chapter's 256), how many total additions would Section 5.1's Hillis-Steele scan need? (Use the same per-step formula the code computes: sum of `n-d` for `d = 1, 2, 4, ..., n/2`.)
2. Section 5.2's up-sweep and down-sweep each do exactly 255 additions for N=256. Explain why these two numbers are equal, using the tree-structure argument from the up-sweep's own resemblance to Chapter 4's reduction.
3. A teammate proposes skipping kernel 2 in Section 5.3 and instead having kernel 3 add up ALL preceding blocks' totals with a sequential loop over blocks. For 1000 blocks, compare this proposal's span to kernel 2's actual span, using Chapter 3's vocabulary.
4. Section 5.2 sets `temp[n-1] = 0` right after the up-sweep to get an EXCLUSIVE scan. What would the down-sweep produce instead if that line were removed, and would the result still be a valid scan of some kind?
5. Using Section 5.3's own block totals (1774, 1790, 1806, 1783, 1786, 1802, 1792, 1782), verify by hand that the reported per-block offset for block 3 (5370) is correct.
6. Why does kernel 1 in Section 5.3 need to capture `g_block_sums[blockIdx.x]` BEFORE zeroing `temp[n-1]`, rather than after?

## Where We Go Next

Chapter 6 builds stream compaction — filtering an array down to only the elements that satisfy some condition, while preserving their relative order — directly on top of this chapter's exclusive scan: a boolean "keep this element" flag, scanned, gives exactly the output position each kept element belongs at. Chapter 7 then builds histograms, the last of Part 1's four core primitives, before Part 2 turns to linear data structures built from all four.

## Worked Solutions

**1.** For `n=512`, the steps are `d = 1, 2, 4, 8, 16, 32, 64, 128, 256`, and each step contributes `n - d` additions: `511 + 510 + 508 + 504 + 496 + 480 + 448 + 384 + 256 = 4097`. (Following the same pattern as `n=256`'s 1793, roughly doubling as `n` doubles, since both the number of steps grows by one AND each step's contribution grows.)

**2.** The up-sweep is structurally identical to Chapter 4's reduction tree for the same N, which does exactly `n-1` additions total (one addition per internal node of a binary tree over `n` leaves, and a binary tree over `n` leaves has exactly `n-1` internal nodes). The down-sweep walks the SAME tree structure in reverse, visiting the identical set of internal nodes once each — same node count, same one-addition-per-node cost, hence the identical `n-1 = 255` total.

**3.** A sequential loop over `p` preceding blocks' totals has span `p` (each addition depends on the running total before it — Chapter 3's own sequential-sum example, applied to blocks instead of elements). For 1000 blocks, that is span 1000. Kernel 2's actual scan has span `2 * log2(1000)`, approximately `2 * 10 = 20`, (its own up-sweep plus down-sweep, rounding 1000 up to the next power of two). The sequential proposal is roughly 50x worse in span for the identical correctness guarantee, precisely because it discards the same independence-across-blocks insight Chapter 3 spent an entire chapter establishing.

**4.** Without zeroing `temp[n-1]`, the down-sweep would distribute the TRUE total (not zero) as the starting value fed into the tree, producing, at each position, the sum of everything up to and including a shifted starting point rather than a clean running total from zero — specifically, it would no longer satisfy either the standard exclusive-scan definition (running total of strictly-preceding elements, starting from 0) or the inclusive-scan definition; it would be a valid down-sweep of SOME initial value, but not a scan of the original array by either standard definition, since the "zero" identity element is what the exclusive scan mathematically requires at the tree's root before distributing.

**5.** Block 3's offset should be the sum of every EARLIER block's total: blocks 0, 1, and 2, with totals 1774, 1790, and 1806. Sum: `1774 + 1790 + 1806 = 5370`, matching the reported offset exactly.

**6.** `temp[n-1]` after the up-sweep holds the block's TRUE total (the actual sum of all its elements) — exactly the value kernel 2 needs to correctly compute this block's offset relative to every other block. Zeroing it first would record 0 as this block's "total," making every later block's offset calculation silently wrong (each one would be missing this block's actual contribution), while still not affecting kernel 1's own local scan correctness — the bug would be invisible within any single block and only surface once kernel 3 tried to combine blocks that all reported a total of zero.
