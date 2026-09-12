# Chapter 4: Reduction, For Real

**What you will understand by the end of this chapter:**

- How to write a real, launchable CUDA reduction kernel using the tree shape Chapter 3 already traced, and why the most obvious way to write it interacts badly with Chapter 1's warp divergence model.
- Why changing which threads a fixed-size active set selects — not how many threads are active — genuinely eliminates most of that divergence, measured directly rather than assumed.
- How a two-phase design (grid-stride accumulation, then two tree-reduction launches) generalizes a single block's reduction to arbitrarily large `N`.

**What you need to know first:**

- Part 0 in full: warp divergence's issue-pass model (Chapter 1), grid-stride loops and shared-memory staging (Chapter 2), and the work-span model applied to a traced reduction (Chapter 3).

---

Part 0 built the vocabulary; Part 1 starts spending it. This chapter is the first in the book to write and genuinely compile a complete, real CUDA kernel intended to be launched for actual work, rather than a kernel written primarily to host-simulate a specific cost model. Reduction is the right place to start because Chapter 3 already traced its dependency structure in the abstract — this chapter turns that trace into working code, and finds a real, measurable divergence cost hiding in the most natural way to write it.

## 4.1 A Naive Reduction Kernel, and Its Divergence Cost

### Intuition

Chapter 3's tree reduction combined adjacent pairs, then adjacent pairs of those results. The most direct way to write "which threads are active at step `s`" as an index test is `tid % (2*s) == 0` — thread 0, thread `2s`, thread `4s`, and so on. This looks harmless. It is not: Chapter 1 established that a warp's cost is the number of *distinct paths* its 32 lanes take, and a modulus test recurs *inside every warp's own lane numbering*, not just at the boundary between warps.

### The Sequential (CPU) Baseline

On a CPU, reduction is nothing but one loop:

```cpp
#include <cstdio>
#include <vector>

// Chapter 4.1 -- The Sequential (CPU) Baseline.
// On a single CPU thread, reduction is nothing but one loop: one running
// variable, O(n) work, and no notion of "warps" or "divergence" at all,
// because there is only ever one thread doing the adding.

float reduce_cpu(const float* data, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) sum += data[i];
    return sum;
}

int main() {
    printf("=== Section 4.1 CPU baseline: sequential reduction ===\n\n");

    const int N = 10;
    std::vector<float> data(N);
    for (int i = 0; i < N; i++) data[i] = (float)i;

    printf("input (%d elements): ", N);
    for (int i = 0; i < N; i++) printf("%.0f ", data[i]);
    printf("\n\n");

    float sum = reduce_cpu(data.data(), N);
    float expected = (float)(N * (N - 1) / 2);
    bool ok = (sum == expected);

    printf("reduce_cpu sum: %.1f\n", sum);
    printf("expected (closed form n*(n-1)/2): %.1f\n", expected);
    printf("\nself-check: sequential reduction matches closed-form sum: %s\n",
           ok ? "confirmed" : "MISMATCH");
    return ok ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++17 -Wall -Wextra -O2 31_reduce_cpu_baseline.cpp -o reduce_cpu_baseline
./reduce_cpu_baseline
```

**Sample input:** a 10-element array, values `0, 1, 2, ..., 9`.

**Sample output:**

```text
=== Section 4.1 CPU baseline: sequential reduction ===

input (10 elements): 0 1 2 3 4 5 6 7 8 9 

reduce_cpu sum: 45.0
expected (closed form n*(n-1)/2): 45.0

self-check: sequential reduction matches closed-form sum: confirmed
```

O(n) work, one running variable, no notion of "warps" or "divergence" at all, because there is only ever one thread doing the adding. Warp divergence is not a cost this loop could ever pay — it is a cost that only appears once the identical reduction is rewritten to run across many threads at once, which is exactly what the rest of this section does.

### The Concept, In Detail

A 256-thread block has exactly 8 warps of 32 lanes each. At every step of the reduction, SOME threads are "active" (they still have an addition to do) and the rest are not. What determines a warp's issue-pass cost, by Chapter 1's own rule, is not how many of ITS 32 lanes are active — it is whether they are ALL active, ALL inactive, or a MIX. A mix costs 2 issue-passes (the warp's scheduler issues the instruction once for the active lanes and once more, wastefully, for the inactive ones); uniform costs 1; fully idle costs 0.

`tid % (2*s) == 0` places its active lanes at 0, `2s`, `4s`, `6s`, ... — evenly spread across the ENTIRE 256-thread range, at every step, no matter how small `s` gets. Trace warp 0 (block-local lanes 0-31) directly:

```
interleaved addressing: warp 0's own 32 lanes, A = active, . = inactive

s=1   (active where tid %   2 == 0):
      A.A.A.A.A.A.A.A.A.A.A.A.A.A.A.A.A.A.A.A.A.A.A.A.A.A.A.A.A.A.A.A
      16 of 32 lanes active, alternating every OTHER lane -> MIXED (2 passes)

s=16  (active where tid %  32 == 0):
      A...............................
      only lane 0 active -- STILL mixed (2 passes) -- same cost as s=1

s=128 (active where tid % 256 == 0):
      A...............................
      only lane 0 active -- STILL mixed (2 passes) -- same cost as s=16
```

Lane 0 of warp 0 is global thread 0 — the thread that ends up holding the reduction's final answer — so it never stops being active, and warp 0 never becomes uniform, at ANY of the 8 steps. What actually changes step to step is how many of the OTHER 7 warps in the block still have any active lane left at all:

```
step_s:          1    2    4    8   16   32   64  128
active warps:    8    8    8    8    8    4    2    1     (out of 8 total)
each active warp is MIXED until it drops to 0 active lanes, so it costs 2
passes right up until the step where it goes fully idle:
issue passes:   16   16   16   16   16    8    4    2     -> sums to 94
```

The kernel below reduces one block's shared-memory tile using exactly this interleaved-addressing test. Its host-side model walks the identical per-step active/inactive pattern and counts issue-passes per warp with Chapter 1's own rule: a warp with a genuine mix of active and inactive lanes pays 2 passes, a uniformly active or uniformly idle warp pays 1 or 0.

### Code and Verification

```cpp
#include <cstdio>
#include <vector>

// Chapter 4.1 -- Chapter 3 traced a reduction's work and span in the
// abstract. This is the first REAL, launchable CUDA kernel in this book:
// a single block reduces its own shared-memory tile to one value, using
// exactly the tree shape Chapter 3, Section 3.2 already traced level by
// level. The first, most obvious way to write the per-level "which
// threads are still active" test turns out to interact badly with
// Chapter 1's warp divergence model -- and this section measures that
// interaction directly rather than asserting it.

#define BLOCK_SIZE 256

// Interleaved addressing: at step s (1, 2, 4, ..., BLOCK_SIZE/2), the
// threads that do work are exactly those with `tid % (2*s) == 0`. This
// is the most direct translation of "combine adjacent pairs, then
// combine adjacent pairs of those" into an index test.
__global__ void reduce_interleaved(const float* g_in, float* g_out, int n) {
    __shared__ float sdata[BLOCK_SIZE];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = (idx < n) ? g_in[idx] : 0.0f;
    __syncthreads();

    for (int s = 1; s < blockDim.x; s *= 2) {
        if (tid % (2 * s) == 0) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) g_out[blockIdx.x] = sdata[0];
}

// ---- Host-side model: walk the exact same per-step active/inactive
// ---- pattern the kernel's own `tid % (2*s) == 0` test produces, and
// ---- count issue-passes per warp using Chapter 1's own model: 1 pass if
// ---- a warp's active lanes agree (all active or all inactive), 2 passes
// ---- if a warp has a genuine mix, 0 passes if every lane in the warp
// ---- has nothing left to do this step. ----

const int WARP_SIZE = 32;

struct StepDivergence {
    int step_s;
    long long active_threads;
    int passes;   // summed across all warps in the block for this one step
};

std::vector<StepDivergence> trace_interleaved_divergence(int block_size) {
    std::vector<StepDivergence> trace;
    int num_warps = block_size / WARP_SIZE;
    for (int s = 1; s < block_size; s *= 2) {
        StepDivergence sd;
        sd.step_s = s;
        sd.active_threads = 0;
        sd.passes = 0;
        for (int w = 0; w < num_warps; w++) {
            bool any_active = false, any_inactive = false;
            for (int lane = 0; lane < WARP_SIZE; lane++) {
                int tid = w * WARP_SIZE + lane;
                bool active = (tid % (2 * s) == 0);
                if (active) { any_active = true; sd.active_threads++; }
                else any_inactive = true;
            }
            if (any_active && any_inactive) sd.passes += 2;
            else if (any_active) sd.passes += 1;
            // fully inactive warp: 0 passes, nothing to schedule
        }
        trace.push_back(sd);
    }
    return trace;
}

float reference_sum(const std::vector<float>& vals) {
    float s = 0.0f;
    for (float v : vals) s += v;
    return s;
}

// Host-side simulation of the kernel's OWN arithmetic (not just an
// independent reference) -- walks the identical shared-memory tile and
// step loop the device code above describes, to confirm the algorithm
// itself is correct before trusting the divergence trace built on top
// of its index pattern.
float simulate_reduce_interleaved(std::vector<float> sdata) {
    int block_size = (int)sdata.size();
    for (int s = 1; s < block_size; s *= 2) {
        for (int tid = 0; tid < block_size; tid++) {
            if (tid % (2 * s) == 0 && tid + s < block_size) {
                sdata[tid] += sdata[tid + s];
            }
        }
    }
    return sdata[0];
}

int main() {
    printf("=== Section 4.1: a real reduction kernel, and its divergence cost ===\n\n");

    std::vector<float> vals(BLOCK_SIZE);
    for (int i = 0; i < BLOCK_SIZE; i++) vals[i] = (float)(i + 1);

    float ref = reference_sum(vals);
    float sim = simulate_reduce_interleaved(vals);
    bool correct = (sim == ref);
    printf("block size = %d, reference sum = %.1f, kernel-arithmetic simulation = %.1f\n",
           BLOCK_SIZE, ref, sim);
    printf("simulation matches independent reference: %s\n\n", correct ? "yes" : "NO -- BUG");

    auto trace = trace_interleaved_divergence(BLOCK_SIZE);
    printf("%-8s %-16s %-10s\n", "step_s", "active_threads", "issue_passes");
    long long total_passes = 0;
    for (const auto& sd : trace) {
        printf("%-8d %-16lld %-10d\n", sd.step_s, sd.active_threads, sd.passes);
        total_passes += sd.passes;
    }
    printf("\ntotal issue-passes across all %d steps (interleaved addressing): %lld\n",
           (int)trace.size(), total_passes);
    printf("an ideal, non-divergent reduction issuing exactly 1 pass per step per active\n");
    printf("warp would need far fewer -- interleaved addressing's `tid %% (2*s) == 0` test\n");
    printf("spreads its active/inactive boundary across every warp's own lane numbering,\n");
    printf("so nearly every warp stays mixed (2 passes) far longer than the shrinking\n");
    printf("active-thread count alone would suggest.\n");

    bool ok = correct && (total_passes > (int)trace.size());   // strictly worse than 1/step ideal
    printf("\nself-check: kernel arithmetic correct, and total passes exceed the %d-step\n",
           (int)trace.size());
    printf("ideal minimum (confirming genuine, measured divergence overhead): %s\n",
           ok ? "confirmed" : "MISMATCH");
    return ok ? 0 : 1;
}
```

**Compile and run:**

```bash
nvcc -arch=sm_80 10_naive_interleaved_reduction.cu -o naive_reduction
./naive_reduction
```

**Sample input:** a 256-element block (`BLOCK_SIZE = 256`), values `0, 1, 2, ..., 255`, reduced by addition.

**Sample output:**

```text
=== Section 4.1: a real reduction kernel, and its divergence cost ===

block size = 256, reference sum = 32896.0, kernel-arithmetic simulation = 32896.0
simulation matches independent reference: yes

step_s   active_threads   issue_passes
1        128              16        
2        64               16        
4        32               16        
8        16               16        
16       8                16        
32       4                8         
64       2                4         
128      1                2         

total issue-passes across all 8 steps (interleaved addressing): 94
an ideal, non-divergent reduction issuing exactly 1 pass per step per active
warp would need far fewer -- interleaved addressing's `tid % (2*s) == 0` test
spreads its active/inactive boundary across every warp's own lane numbering,
so nearly every warp stays mixed (2 passes) far longer than the shrinking
active-thread count alone would suggest.

self-check: kernel arithmetic correct, and total passes exceed the 8-step
ideal minimum (confirming genuine, measured divergence overhead): confirmed
```

Every one of the first five steps keeps every single warp in the block divergent — not because more threads are active than the tree shape requires, but because the modulus test's active/inactive boundary falls inside every warp's own 32 lanes, every time, until `s` grows large enough that an entire warp's worth of lane numbers shares the same outcome.

!!! warning "[COMMON TRAP] Assuming divergence disappears once most threads are idle"
    By step `s=16`, only 8 of 256 threads are still doing useful work — it is tempting to assume a kernel this idle must be cheap. The trace shows the opposite: every one of the block's 8 warps is still divergent at that step, because interleaved addressing places exactly one active lane inside every warp, no matter how few threads remain active overall. Thread *count* and warp *divergence* are different measurements — Chapter 1 already warned against conflating them, and this section is where that warning becomes a concrete, measured kernel cost rather than an abstract rule.

## 4.2 Sequential Addressing: Fixing the Index Test, Not the Thread Count

### Intuition

Section 4.1's divergence problem is not that too many threads participate — it is *which* threads participate. If the active threads at every step are a single *contiguous* block starting from thread 0, that active/inactive boundary can fall inside at most one warp at any given step, no matter how the boundary shrinks.

### The Sequential (CPU) Baseline

The CPU baseline is unchanged from Section 4.1 — the identical one-loop function computes the identical sum with the identical O(n) work, regardless of which GPU index test this section is about to change:

```cpp
#include <cstdio>
#include <vector>

// Chapter 4.2 -- The Sequential (CPU) Baseline.
// Unchanged from Section 4.1: the identical one-loop reduce_cpu computes
// the identical sum with the identical O(n) work, regardless of which
// GPU index test Section 4.2 is about to change. Demonstrated here on a
// differently-sized array purely to show the function does not care.

float reduce_cpu(const float* data, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) sum += data[i];
    return sum;
}

int main() {
    printf("=== Section 4.2 CPU baseline: the identical reduce_cpu, unchanged ===\n\n");

    const int N = 16;
    std::vector<float> data(N);
    for (int i = 0; i < N; i++) data[i] = (float)i;

    printf("input (%d elements): ", N);
    for (int i = 0; i < N; i++) printf("%.0f ", data[i]);
    printf("\n\n");

    float sum = reduce_cpu(data.data(), N);
    float expected = (float)(N * (N - 1) / 2);
    bool ok = (sum == expected);

    printf("reduce_cpu sum: %.1f\n", sum);
    printf("expected (closed form n*(n-1)/2): %.1f\n", expected);
    printf("\nsame function, a different N -- nothing about this loop changes when the\n");
    printf("GPU kernel's index test changes in Section 4.2, because a CPU reduction\n");
    printf("never had warps or divergence to begin with.\n");
    printf("\nself-check: sequential reduction matches closed-form sum: %s\n",
           ok ? "confirmed" : "MISMATCH");
    return ok ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++17 -Wall -Wextra -O2 32_reduce_cpu_baseline_unchanged.cpp -o reduce_cpu_baseline_unchanged
./reduce_cpu_baseline_unchanged
```

**Sample input:** a 16-element array, values `0, 1, 2, ..., 15` — a different size from Section 4.1's, purely to demonstrate the function does not care.

**Sample output:**

```text
=== Section 4.2 CPU baseline: the identical reduce_cpu, unchanged ===

input (16 elements): 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 

reduce_cpu sum: 120.0
expected (closed form n*(n-1)/2): 120.0

same function, a different N -- nothing about this loop changes when the
GPU kernel's index test changes in Section 4.2, because a CPU reduction
never had warps or divergence to begin with.

self-check: sequential reduction matches closed-form sum: confirmed
```

What Section 4.2 improves is entirely a property of how the GPU version divides its work across warps; the sequential version never had that property to begin with, which is exactly why the fix belongs entirely on the GPU side.

### The Concept, In Detail

Changing the test from `tid % (2*s) == 0` to `tid < s`, with `s` halving from `blockDim.x/2` down to 1, makes the active set a single contiguous run `[0, s)` instead of a scattered, evenly-spread set. A contiguous run has exactly ONE boundary — the point where thread ID `s-1` (active) meets thread ID `s` (inactive) — and that single boundary can only ever fall inside ONE warp's 32 consecutive lane numbers. Every OTHER warp is entirely on one side of it: either every one of its lanes is below `s` (uniform, 1 pass) or every one of its lanes is at or above `s` (fully idle, 0 passes).

```
sequential addressing: warp 0's own 32 lanes, test is tid < s

s=128 (active where tid < 128): warps 0-3 are ENTIRELY active (uniform),
                                 warps 4-7 are ENTIRELY inactive -- the
                                 boundary falls BETWEEN warp 3 and warp 4,
                                 inside NO warp's own 32 lanes.

s=16  (active where tid < 16):  warp 0's own lanes:
                                 AAAAAAAAAAAAAAAA................
                                 (lanes 0-15 active, 16-31 inactive) --
                                 the boundary falls INSIDE warp 0 only;
                                 every other warp is entirely inactive.
```

Once `s` drops below 32 (one warp's width), the single boundary is permanently trapped inside warp 0 for the rest of the reduction — an IRREDUCIBLE cost, not a bug, since a tree whose active count has shrunk below one warp's size cannot possibly keep that one remaining warp uniform:

```
step_s:          128   64   32   16    8    4    2    1
active warps:      4    2    1    1    1    1    1    1    (out of 8 total)
mixed warps:       0    0    0    1    1    1    1    1    (boundary inside it)
uniform warps:     4    2    1    0    0    0    0    0
issue passes:      4    2    1    2    2    2    2    2    -> sums to 17
```

The kernel below is otherwise identical to Section 4.1's — same shared-memory tile, same number of steps, same total useful work — and its host-side model uses the identical warp-by-warp counting rule, reporting Section 4.1's own total inline for a direct comparison.

### Code and Verification

```cpp
#include <cstdio>
#include <vector>

// Chapter 4.2 -- Section 4.1's interleaved addressing scattered its
// active/inactive boundary across every warp's own lane numbering,
// keeping nearly every warp divergent for most of the reduction. The
// fix changes only WHICH threads the test selects, not how many: instead
// of `tid % (2*s) == 0` (a boundary that recurs inside every warp), use
// `tid < s` (a single boundary that only ever falls inside AT MOST ONE
// warp at a time, because active threads are always a contiguous block
// starting from thread 0).

#define BLOCK_SIZE 256

__global__ void reduce_sequential(const float* g_in, float* g_out, int n) {
    __shared__ float sdata[BLOCK_SIZE];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = (idx < n) ? g_in[idx] : 0.0f;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s /= 2) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) g_out[blockIdx.x] = sdata[0];
}

const int WARP_SIZE = 32;

struct StepDivergence {
    int step_s;
    long long active_threads;
    int passes;
};

std::vector<StepDivergence> trace_sequential_divergence(int block_size) {
    std::vector<StepDivergence> trace;
    int num_warps = block_size / WARP_SIZE;
    for (int s = block_size / 2; s > 0; s /= 2) {
        StepDivergence sd;
        sd.step_s = s;
        sd.active_threads = 0;
        sd.passes = 0;
        for (int w = 0; w < num_warps; w++) {
            bool any_active = false, any_inactive = false;
            for (int lane = 0; lane < WARP_SIZE; lane++) {
                int tid = w * WARP_SIZE + lane;
                bool active = (tid < s);
                if (active) { any_active = true; sd.active_threads++; }
                else any_inactive = true;
            }
            if (any_active && any_inactive) sd.passes += 2;
            else if (any_active) sd.passes += 1;
        }
        trace.push_back(sd);
    }
    return trace;
}

float reference_sum(const std::vector<float>& vals) {
    float s = 0.0f;
    for (float v : vals) s += v;
    return s;
}

float simulate_reduce_sequential(std::vector<float> sdata) {
    int block_size = (int)sdata.size();
    for (int s = block_size / 2; s > 0; s /= 2) {
        for (int tid = 0; tid < block_size; tid++) {
            if (tid < s) {
                sdata[tid] += sdata[tid + s];
            }
        }
    }
    return sdata[0];
}

int main() {
    printf("=== Section 4.2: sequential addressing, measured against Section 4.1 ===\n\n");

    std::vector<float> vals(BLOCK_SIZE);
    for (int i = 0; i < BLOCK_SIZE; i++) vals[i] = (float)(i + 1);

    float ref = reference_sum(vals);
    float sim = simulate_reduce_sequential(vals);
    bool correct = (sim == ref);
    printf("block size = %d, reference sum = %.1f, kernel-arithmetic simulation = %.1f\n",
           BLOCK_SIZE, ref, sim);
    printf("simulation matches independent reference: %s\n\n", correct ? "yes" : "NO -- BUG");

    auto trace = trace_sequential_divergence(BLOCK_SIZE);
    printf("%-8s %-16s %-10s\n", "step_s", "active_threads", "issue_passes");
    long long total_passes = 0;
    for (const auto& sd : trace) {
        printf("%-8d %-16lld %-10d\n", sd.step_s, sd.active_threads, sd.passes);
        total_passes += sd.passes;
    }
    printf("\ntotal issue-passes across all %d steps (sequential addressing): %lld\n",
           (int)trace.size(), total_passes);

    // Section 4.1's own measured total, reproduced here for a direct,
    // in-program comparison rather than asking the reader to flip back.
    const long long interleaved_total = 94;
    printf("Section 4.1's interleaved addressing total, for the identical block size and\n");
    printf("identical amount of useful work: %lld\n", interleaved_total);
    printf("sequential addressing's reduction in issue-passes: %lld -> %lld (%.2fx fewer)\n",
           interleaved_total, total_passes, (double)interleaved_total / (double)total_passes);
    printf("both kernels compute the IDENTICAL sum -- this is purely a divergence-cost\n");
    printf("reduction from changing which threads a fixed-size active set selects, not a\n");
    printf("change in how much reduction work happens.\n");

    bool ok = correct && (total_passes < interleaved_total);
    printf("\nself-check: kernel arithmetic correct, and total passes strictly less than\n");
    printf("Section 4.1's interleaved-addressing total: %s\n", ok ? "confirmed" : "MISMATCH");
    return ok ? 0 : 1;
}
```

**Compile and run:**

```bash
nvcc -arch=sm_80 11_sequential_addressing_reduction.cu -o sequential_reduction
./sequential_reduction
```

**Sample input:** the identical 256-element block and values as Section 4.1 (`0, 1, 2, ..., 255`), so the two kernels' results can be compared directly.

**Sample output:**

```text
=== Section 4.2: sequential addressing, measured against Section 4.1 ===

block size = 256, reference sum = 32896.0, kernel-arithmetic simulation = 32896.0
simulation matches independent reference: yes

step_s   active_threads   issue_passes
128      128              4         
64       64               2         
32       32               1         
16       16               2         
8        8                2         
4        4                2         
2        2                2         
1        1                2         

total issue-passes across all 8 steps (sequential addressing): 17
Section 4.1's interleaved addressing total, for the identical block size and
identical amount of useful work: 94
sequential addressing's reduction in issue-passes: 94 -> 17 (5.53x fewer)
both kernels compute the IDENTICAL sum -- this is purely a divergence-cost
reduction from changing which threads a fixed-size active set selects, not a
change in how much reduction work happens.

self-check: kernel arithmetic correct, and total passes strictly less than
Section 4.1's interleaved-addressing total: confirmed
```

A 5.53x reduction in issue-passes, for two kernels that compute the bit-identical sum. Only one warp is ever divergent at a time now — every other warp is either fully active (uniform, 1 pass) or has already finished and gone idle (0 passes).

!!! warning "[COMMON TRAP] Assuming sequential addressing eliminates ALL divergence"
    Section 4.2's own trace shows divergence does not reach zero — steps `s=16` down to `s=1` still show 2 passes each, because a contiguous active block smaller than one warp (32 threads) still splits that one warp between active and inactive lanes. Sequential addressing eliminates the *spurious* divergence interleaved addressing added on top of the tree shape's own structure; it cannot eliminate the *irreducible* divergence of a tree whose last few levels, by definition, have fewer active threads than one warp holds. Later chapters that reduce within a single warp (using warp-shuffle instructions instead of shared memory for the final steps) address exactly this remaining, structural cost — not a bug in this section's kernel, but a limit intrinsic to any shared-memory tree once it shrinks below 32 active threads.

## 4.3 Multi-Block Reduction: Beyond One Block's Capacity

### Intuition

Sections 4.1 and 4.2 reduced exactly 256 elements — one block's worth. A real reduction has to handle `N` far larger than any single block can hold in shared memory. The standard answer combines three ideas this book has already built separately: a grid-stride loop lets each thread pre-accumulate several elements before the tree even starts, each block's tree reduces its own threads down to one partial sum, and a second, smaller launch reduces the resulting (much shorter) array of per-block partial sums.

### The Sequential (CPU) Baseline

The identical single loop from Section 4.1 handles `N = 100,000` exactly as easily as `N = 256` — a CPU loop does not care how large `N` is, it simply iterates more times:

```cpp
#include <cstdio>
#include <cmath>
#include <vector>

// Chapter 4.3 -- The Sequential (CPU) Baseline.
// The identical single loop from Section 4.1 handles N=100,000 exactly
// as easily as N=256 -- a CPU loop does not care how large N is, it
// simply iterates more times. There is no CPU equivalent of the GPU
// version's two-phase, two-kernel design at all.

float reduce_cpu(const float* data, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) sum += data[i];
    return sum;
}

int main() {
    printf("=== Section 4.3 CPU baseline: one loop, no phases, N=100000 ===\n\n");

    const int N = 100000;
    std::vector<float> data(N);
    for (int i = 0; i < N; i++) data[i] = (float)((i % 97) + 1);

    double ref = 0.0;
    for (int i = 0; i < N; i++) ref += data[i];

    float sum = reduce_cpu(data.data(), N);
    bool ok = (std::fabs((double)sum - ref) < 1.0);

    printf("N = %d elements, values (i %% 97) + 1\n\n", N);
    printf("reduce_cpu sum (float):        %.4f\n", sum);
    printf("independent reference (double): %.4f\n", ref);
    printf("match within floating-point tolerance: %s\n\n", ok ? "yes" : "NO -- BUG");
    printf("one loop, one running variable -- no blocks, no phases, no launch boundary,\n");
    printf("regardless of whether N is 256 or 100,000.\n");
    printf("\nself-check: sequential reduction matches independent reference: %s\n",
           ok ? "confirmed" : "MISMATCH");
    return ok ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++17 -Wall -Wextra -O2 33_reduce_cpu_baseline_multiblock.cpp -o reduce_cpu_baseline_multiblock
./reduce_cpu_baseline_multiblock
```

**Sample input:** `N = 100000` elements, values `(i % 97) + 1` — the identical values the two-phase GPU version below uses.

**Sample output:**

```text
=== Section 4.3 CPU baseline: one loop, no phases, N=100000 ===

N = 100000 elements, values (i % 97) + 1

reduce_cpu sum (float):        4899685.0000
independent reference (double): 4899685.0000
match within floating-point tolerance: yes

one loop, one running variable -- no blocks, no phases, no launch boundary,
regardless of whether N is 256 or 100,000.

self-check: sequential reduction matches independent reference: confirmed
```

There is no CPU equivalent of this section's two-phase, two-kernel design at all: splitting the work across blocks, giving each block its own shared-memory tile, and combining partial sums with a second launch are all concessions to how a GPU's shared memory and grid dimensions work, not anything a sequential reduction ever needs.

### The Concept, In Detail

The whole design is two ordinary kernels with one, unavoidable gap between them:

```
N = 100,000 elements, NUM_BLOCKS = 64, BLOCK_SIZE = 256 (16,384 total threads)

PHASE 1 (64 blocks, launched together):
  thread (block b, lane t): idx = b*256 + t, stride = 64*256 = 16384
    acc = in[idx] + in[idx+16384] + in[idx+32768] + ...   (~6.1 elements/thread)
  each block's 256 partial sums then tree-reduce (Section 4.2's exact shape)
  down to ONE value per block:

      block 0 -> partials[0]   block 1 -> partials[1]  ...  block 63 -> partials[63]

-------------------------- KERNEL LAUNCH BOUNDARY --------------------------
  no __syncthreads() can cross this line: blocks are not guaranteed to run
  concurrently, or in any particular order (Chapter 1's own point about
  block scheduling) -- only a WHOLE NEW LAUNCH guarantees every block above
  has finished and written its partial sum before phase 2 reads any of them.
------------------------------------------------------------------------

PHASE 2 (1 block, 256 threads, launched once):
  reads all 64 partials into shared memory (lanes 64-255 pad with 0),
  tree-reduces them (the identical Section 4.2 shape) down to ONE final value.
```

The grid-stride loop means phase 1's cost per thread grows with `N`, but the STRUCTURE never changes — the same 64 blocks, the same tree shape, regardless of whether `N` is 100,000 or 100,000,000. What phase 2 needs, structurally, is for the NUMBER of phase-1 blocks to fit inside one block's own thread count, so a single further launch can finish the job. Chapter 3's span vocabulary generalizes cleanly here: phase 1's tree is `log2(256) = 8` levels, phase 2's tree is `log2(64) = 6` levels, and the one unavoidable launch boundary between them counts as exactly 1 more — a total span of `8 + 1 + 6 = 15`, fixed regardless of how large `N` grows.

The two kernels below implement exactly this. `reduce_phase1` combines Chapter 2's grid-stride loop with Section 4.2's sequential-addressing tree; `reduce_phase2` reduces the resulting array of per-block partials with the identical tree shape, in a single block.

### Code and Verification

```cpp
#include <cstdio>
#include <vector>
#include <cmath>

// Chapter 4.3 -- Sections 4.1-4.2 reduced exactly one block's worth of
// data (256 elements). A real reduction has to handle N far larger than
// one block can hold. The standard two-phase design combines three ideas
// this book has already built separately: Chapter 2's grid-stride loop
// (each thread first accumulates several elements from global memory
// into a register), Section 4.2's sequential-addressing tree (each
// block then reduces its own threads' partial sums), and a second,
// smaller launch of the SAME kernel shape to combine the per-block
// results -- because the per-block partial sums are themselves just a
// smaller array that needs reducing.

#define BLOCK_SIZE 256

// Phase 1: each thread accumulates a grid-stride subset of the N input
// elements into one register (exactly Chapter 2, Section 2.1's loop
// shape), then the block tree-reduces its threads' partial sums using
// Section 4.2's sequential addressing.
__global__ void reduce_phase1(const float* g_in, float* g_partials, int n) {
    __shared__ float sdata[BLOCK_SIZE];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    float acc = 0.0f;
    for (int i = idx; i < n; i += stride) {
        acc += g_in[i];
    }
    sdata[tid] = acc;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s /= 2) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) g_partials[blockIdx.x] = sdata[0];
}

// Phase 2: a single block reduces the (small) array of per-block
// partials from phase 1 down to one final value, using the identical
// sequential-addressing tree.
__global__ void reduce_phase2(const float* g_partials, float* g_out, int num_partials) {
    __shared__ float sdata[BLOCK_SIZE];
    int tid = threadIdx.x;
    sdata[tid] = (tid < num_partials) ? g_partials[tid] : 0.0f;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s /= 2) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) g_out[0] = sdata[0];
}

// ---- Host-side simulation of the exact same two-phase arithmetic ----

float reference_sum(const std::vector<float>& vals) {
    float s = 0.0f;
    for (float v : vals) s += v;
    return s;
}

std::vector<float> simulate_phase1(const std::vector<float>& in, int num_blocks, int block_size) {
    int n = (int)in.size();
    int stride = num_blocks * block_size;
    std::vector<float> partials(num_blocks, 0.0f);

    for (int block = 0; block < num_blocks; block++) {
        std::vector<float> sdata(block_size, 0.0f);
        for (int tid = 0; tid < block_size; tid++) {
            int idx = block * block_size + tid;
            float acc = 0.0f;
            for (int i = idx; i < n; i += stride) acc += in[i];
            sdata[tid] = acc;
        }
        for (int s = block_size / 2; s > 0; s /= 2) {
            for (int tid = 0; tid < s; tid++) sdata[tid] += sdata[tid + s];
        }
        partials[block] = sdata[0];
    }
    return partials;
}

float simulate_phase2(std::vector<float> partials, int block_size) {
    int num_partials = (int)partials.size();
    std::vector<float> sdata(block_size, 0.0f);
    for (int tid = 0; tid < block_size; tid++) {
        sdata[tid] = (tid < num_partials) ? partials[tid] : 0.0f;
    }
    for (int s = block_size / 2; s > 0; s /= 2) {
        for (int tid = 0; tid < s; tid++) sdata[tid] += sdata[tid + s];
    }
    return sdata[0];
}

int main() {
    printf("=== Section 4.3: a two-phase multi-block reduction ===\n\n");

    const int N = 100000;                 // far larger than any one block can hold
    const int NUM_BLOCKS = 64;             // chosen so the partials array fits in one block
    std::vector<float> vals(N);
    for (int i = 0; i < N; i++) vals[i] = (float)((i % 97) + 1);   // bounded, deterministic values

    float ref = reference_sum(vals);

    auto partials = simulate_phase1(vals, NUM_BLOCKS, BLOCK_SIZE);
    float total_threads = (float)(NUM_BLOCKS * BLOCK_SIZE);
    printf("N = %d elements, phase 1 launch = %d blocks x %d threads = %d total threads\n",
           N, NUM_BLOCKS, BLOCK_SIZE, NUM_BLOCKS * BLOCK_SIZE);
    printf("each thread's grid-stride loop covers roughly %.2f elements on average\n\n",
           (double)N / total_threads);

    float final_sum = simulate_phase2(partials, BLOCK_SIZE);
    bool correct = (std::fabs(final_sum - ref) < 1e-2f);   // float accumulation, small tolerance

    printf("phase 1 produced %d per-block partial sums\n", (int)partials.size());
    printf("phase 2 reduced those %d partials to one final value\n\n", (int)partials.size());
    printf("independent reference sum:      %.4f\n", ref);
    printf("two-phase reduction result:     %.4f\n", final_sum);
    printf("match within floating-point tolerance: %s\n\n", correct ? "yes" : "NO -- BUG");

    // Span, generalized from Chapter 3's single-block trace: phase 1's
    // tree has log2(BLOCK_SIZE) levels, plus one kernel-launch boundary
    // (an implicit global synchronization no in-kernel __syncthreads()
    // can provide, since different blocks cannot be relied on to run
    // concurrently -- Chapter 1's own point about block scheduling), plus
    // phase 2's log2(NUM_BLOCKS) levels.
    int phase1_span = (int)std::log2((double)BLOCK_SIZE);
    int phase2_span = (int)std::log2((double)NUM_BLOCKS);
    int total_span = phase1_span + 1 + phase2_span;   // +1 for the kernel-launch boundary
    printf("span, generalizing Chapter 3's single-block trace: phase 1 = %d levels, one\n",
           phase1_span);
    printf("kernel-launch boundary (blocks cannot synchronize with each other mid-kernel,\n");
    printf("only via a whole new launch), phase 2 = %d levels -- total span = %d, completely\n",
           phase2_span, total_span);
    printf("independent of how large N grows, as long as NUM_BLOCKS <= BLOCK_SIZE so a\n");
    printf("single phase-2 launch can finish the job.\n");

    bool ok = correct && (total_span == phase1_span + phase2_span + 1);
    printf("\nself-check: two-phase result matches reference, total span = %d: %s\n",
           total_span, ok ? "confirmed" : "MISMATCH");
    return ok ? 0 : 1;
}
```

**Compile and run:**

```bash
nvcc -arch=sm_80 12_multi_block_reduction.cu -o multi_block_reduction
./multi_block_reduction
```

**Sample input:** `N = 100000` elements, values `(i % 97) + 1` for `i` in `[0, N)` (bounded, deterministic), launched as `NUM_BLOCKS = 64` blocks of `BLOCK_SIZE = 256` threads for phase 1.

**Sample output:**

```text
=== Section 4.3: a two-phase multi-block reduction ===

N = 100000 elements, phase 1 launch = 64 blocks x 256 threads = 16384 total threads
each thread's grid-stride loop covers roughly 6.10 elements on average

phase 1 produced 64 per-block partial sums
phase 2 reduced those 64 partials to one final value

independent reference sum:      4899685.0000
two-phase reduction result:     4899685.0000
match within floating-point tolerance: yes

span, generalizing Chapter 3's single-block trace: phase 1 = 8 levels, one
kernel-launch boundary (blocks cannot synchronize with each other mid-kernel,
only via a whole new launch), phase 2 = 6 levels -- total span = 15, completely
independent of how large N grows, as long as NUM_BLOCKS <= BLOCK_SIZE so a
single phase-2 launch can finish the job.

self-check: two-phase result matches reference, total span = 15: confirmed
```

100,000 elements, reduced through 16,384 threads' grid-stride accumulation and two tree-reduction launches, matching an independent reference sum exactly. The span analysis generalizes Chapter 3's single-block trace cleanly: phase 1's tree levels, plus one launch boundary, plus phase 2's tree levels — a total that stays fixed regardless of how large `N` grows, as long as the number of blocks launched in phase 1 fits within one block's thread count for phase 2.

!!! warning "[COMMON TRAP] Assuming more blocks always finishes faster"
    It is tempting to launch as many blocks as possible in phase 1, reasoning that more parallelism is always better. Section 4.3's own span analysis shows the limit this hits directly: phase 2 needs the number of phase-1 blocks to fit within ONE block's thread count, or phase 2 itself needs to become multi-level (a phase 2a, phase 2b, ...), adding exactly the kind of extra launch-boundary span this section just introduced. The right number of phase-1 blocks is not "as many as the hardware allows" — it is the number that keeps phase 2 a single, simple launch, trading a larger per-thread grid-stride workload in phase 1 for a bounded, predictable span overall.

## Chapter Summary

Section 4.1 wrote this book's first real, launchable reduction kernel and found a genuine, measured divergence cost in the most natural way to express "which threads are still active": interleaved addressing kept every warp divergent for most of the reduction, totaling 94 issue-passes for a 256-element block. Section 4.2 changed only the index test — sequential addressing, a contiguous active block instead of a scattered modulus pattern — and measured a 5.53x reduction to 17 issue-passes, for the identical sum. Section 4.3 generalized beyond one block's capacity with a two-phase design (grid-stride accumulation, two tree-reduction launches), verified against an independent reference for 100,000 elements, and showed the resulting span stays fixed regardless of how large `N` grows, as long as the per-block partial count fits one further block's launch. The next chapter, scan (prefix sum), reuses this same tree machinery for a genuinely harder problem: not one final value, but a running result at every position.

## Self-Check Questions

1. For a 128-thread block (instead of this chapter's 256), how many steps would Section 4.1's interleaved-addressing loop need, and at which step does the active/inactive boundary first align with a full warp?
2. Section 4.2's trace shows 4 issue-passes at `step_s=128` (the largest step). Using the warp-counting rule (0/1/2 passes per warp), explain exactly why this step costs 4, not 2.
3. A colleague suggests skipping Section 4.2 entirely and using `if (tid % 2 == 0)` combined with halving the ACTIVE COUNT differently to avoid the interleaved pattern. Using this chapter's own vocabulary, explain what specifically distinguishes a "contiguous active block" from any other way of selecting the same number of active threads.
4. Section 4.3 chose `NUM_BLOCKS = 64` for `N = 100000` and `BLOCK_SIZE = 256`. What is the largest `N` this exact configuration (64 blocks, 256 threads/block, one phase-2 launch) could handle while still covering every element in phase 1's grid-stride loop, and does that maximum depend on `N` at all?
5. Explain concretely why the boundary between `reduce_phase1` and `reduce_phase2` cannot be replaced with a single additional `__syncthreads()` call inside one combined kernel.
6. Using Section 4.3's span formula (phase 1 levels + 1 + phase 2 levels), what would the total span become if `NUM_BLOCKS` were increased to 1024 instead of 64, keeping `BLOCK_SIZE = 256`? What practical problem does this immediately create for `reduce_phase2` as written?

## Where We Go Next

Chapter 5 builds scan (prefix sum) on this same tree-shaped foundation, but with a genuinely harder dependency structure: every output position needs the running total of every position before it, not just the grand total Chapter 4's reduction produces. The work-span vocabulary Chapter 3 built, and the divergence-aware kernel-writing habits this chapter measured directly, both carry forward unchanged.

## Worked Solutions

**1.** A 128-thread block needs `log2(128) = 7` steps (`s = 1, 2, 4, 8, 16, 32, 64`). The active/inactive boundary first aligns with a full warp boundary once `2*s >= 32`, i.e. at `s=16` (`2*s=32`) — at that step, `tid % 32 == 0` selects exactly one active lane per warp, the same pattern this chapter's own 256-thread trace showed at its own `s=16` step, since the warp size (32) is what matters, not the block size.

**2.** At `step_s=128` for a 256-thread block, the test is `tid < 128`. Threads 0-127 (warps 0-3) are all active; threads 128-255 (warps 4-7) are all inactive. Every one of the 4 active warps is uniformly active (1 pass each = 4 passes total); every one of the 4 inactive warps costs 0 passes. Total: 4 passes, matching the trace exactly, and none of it is divergence — it is 4 separate uniform (non-divergent) passes, one per active warp, run because there are 4 independent warps that each need to issue the addition once.

**3.** The distinguishing property is that a contiguous active block's boundary (the point where "active" becomes "inactive" as thread ID increases) occurs at exactly ONE location among all `blockDim.x` threads, so it can fall inside at most one warp's 32 consecutive lane numbers at any step. A modulus-based or otherwise scattered selection of the identical NUMBER of active threads can place multiple such boundaries throughout the full thread range, each one potentially falling inside a different warp, multiplying the number of warps that end up divergent even though the total active-thread count is unchanged.

**4.** The grid-stride loop's `stride = num_blocks * block_size = 64 * 256 = 16384` covers `N` correctly for ANY `N`, including values far larger than 100,000 — a thread's loop `for (int i = idx; i < n; i += stride)` simply takes more iterations as `N` grows, exactly Chapter 2's own point about grid-stride loops handling arbitrary `N` without relaunching with different dimensions. So this configuration's maximum `N` is not bounded by phase 1 at all; what IS fixed regardless of `N` is phase 2's requirement that `NUM_BLOCKS` (64) fit within `BLOCK_SIZE` (256) threads, which has nothing to do with how large `N` is.

**5.** `__syncthreads()` only synchronizes threads WITHIN one block — it has no effect on, and no visibility into, any other block's threads. Phase 1 launches 64 independent blocks that are not guaranteed to execute concurrently, in any particular order, or even overlap in time at all (Chapter 1's own point about block scheduling). No barrier expressible inside a single kernel's code can make one block wait for another block it has no handle to; the only mechanism that guarantees every phase-1 block has fully completed and written its partial sum is the CPU-side code's own wait for the first kernel launch to finish before issuing the second.

**6.** Total span would become `8 + 1 + log2(1024) = 8 + 1 + 10 = 19`, only modestly larger. The immediate PRACTICAL problem is unrelated to span: `reduce_phase2` as written launches with `BLOCK_SIZE` (256) threads and reads `g_partials[tid]` only for `tid < num_partials` — with 1024 partials and only 256 threads available to read them in one block, 768 of the partial sums would never be read at all, silently producing a wrong answer rather than a span or performance problem. Handling this correctly requires either a third phase or giving phase 2 its own grid-stride loop over the partials array, not simply accepting a larger span number.
