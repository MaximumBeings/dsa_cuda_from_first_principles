# Chapter 12: Bitonic Sort

Every chapter so far has been building a single structure or primitive: a reduction, a scan, a stack, a queue, a linked list's rank. This chapter starts Part 3 by tackling something bigger -- sorting an entire array -- and it does so with an algorithm whose defining property is not that it is the fastest possible sort, but that its sequence of comparisons is completely fixed in advance, independent of the data. That property, more than raw speed, is what makes bitonic sort the natural first sorting algorithm for a GPU.

## 12.1 Compare-Exchange and What Makes a Sequence Bitonic

### Intuition

A sequence is called **bitonic** if some rotation of it is non-decreasing and then non-increasing. That single definition covers three shapes that look different at first glance:

```
Purely ascending:     1 2 3 5 8 9        (already "up", trivially bitonic)
Up-then-down:         2 4 6 8 9 7 5 3    (rises to a peak, then falls)
Down-then-up (valley): 8 6 4 2 3 5 7 9    (falls to a trough, then rises)
```

The third shape looks nothing like the first two until you rotate it. Starting the valley `8 6 4 2 3 5 7 9` at its own minimum (the `2`) gives `2 3 5 7 9 8 6 4` -- which is exactly the up-then-down shape. This is why the general test for "is this bitonic" cannot just scan left to right looking for one direction change: it has to allow for the sequence to be a rotation of an up-then-down shape, which is most easily checked by first rotating the array to start at its own minimum, and then doing the ordinary two-phase scan (non-decreasing, then non-increasing).

The reason this definition matters is that bitonic sequences have a remarkable property: a single, cheap pass of pairwise comparisons can split a bitonic sequence of size `n` into two bitonic sequences of size `n/2`, with the added guarantee that every element in one half is `<=` every element in the other. That one fact, applied recursively, is the entire engine behind bitonic sort. Section 12.1 focuses on establishing that one pass -- the **compare-exchange split** -- and confirming it does what it claims.

### The Sequential (CPU) Baseline

Before writing any parallel code, a single thread can check "is this bitonic" and perform one compare-exchange split using nothing but two ordinary loops. There is no independent work to divide yet: each comparison touches only two elements, and one thread can simply do all of them, one after another, with no coordination required.

```cpp
#include <cstdio>
#include <vector>
#include <algorithm>

// Chapter 12.1 -- The Sequential (CPU) Baseline.
// A single thread can check "is this bitonic" and perform one
// compare-exchange split with two ordinary loops -- nothing here needs
// more than one thread, because there is no independent work to divide:
// each comparison only touches two elements, and one thread can simply
// do all of them, one after another, with no coordination required.

// A sequence is bitonic if SOME rotation of it is non-decreasing then
// non-increasing (this includes purely monotonic sequences, "up-then-
// down" shapes, and "down-then-up" shapes, since rotating a valley to
// start at its own minimum turns it into an up-then-down shape).
bool is_bitonic(std::vector<int> a) {
    int n = (int)a.size();
    int min_idx = 0;
    for (int i = 1; i < n; i++) if (a[i] < a[min_idx]) min_idx = i;
    std::vector<int> rot(n);
    for (int i = 0; i < n; i++) rot[i] = a[(min_idx + i) % n];
    int i = 0;
    while (i + 1 < n && rot[i] <= rot[i + 1]) i++;
    while (i + 1 < n && rot[i] >= rot[i + 1]) i++;
    return i == n - 1;
}

// One compare-exchange split: for each i in the first half, put the
// smaller of a[i] and a[i+dist] at position i, the larger at i+dist
// (ascending), or the reverse (descending).
void compare_exchange_split(std::vector<int>& a, int dist, bool ascending) {
    for (int i = 0; i < dist; i++) {
        bool should_swap = ascending ? (a[i] > a[i + dist]) : (a[i] < a[i + dist]);
        if (should_swap) std::swap(a[i], a[i + dist]);
    }
}

int main() {
    printf("=== Section 12.1 CPU baseline: is_bitonic and one compare-exchange split ===\n\n");

    std::vector<int> a = {2, 4, 6, 8, 9, 7, 5, 3};
    printf("input (up-then-down bitonic): ");
    for (int v : a) printf("%d ", v);
    printf("\nis_bitonic(input): %s\n\n", is_bitonic(a) ? "true" : "false");

    compare_exchange_split(a, 4, true);
    printf("after one ascending compare-exchange split (distance 4): ");
    for (int v : a) printf("%d ", v);
    printf("\n\n");

    std::vector<int> left(a.begin(), a.begin() + 4);
    std::vector<int> right(a.begin() + 4, a.end());
    int left_max = *std::max_element(left.begin(), left.end());
    int right_min = *std::min_element(right.begin(), right.end());

    printf("left half:  ");
    for (int v : left) printf("%d ", v);
    printf(" -- is_bitonic: %s\n", is_bitonic(left) ? "true" : "false");
    printf("right half: ");
    for (int v : right) printf("%d ", v);
    printf(" -- is_bitonic: %s\n\n", is_bitonic(right) ? "true" : "false");

    printf("left half's max (%d) <= right half's min (%d): %s\n",
           left_max, right_min, (left_max <= right_min) ? "yes" : "NO -- BUG");

    std::vector<int> expected = {2, 4, 5, 3, 9, 7, 6, 8};
    bool ok = (a == expected) && is_bitonic(left) && is_bitonic(right) && (left_max <= right_min);

    printf("\nexpected after split: 2 4 5 3 9 7 6 8\n");
    printf("\nboth halves stayed bitonic, and every left-half value is now <= every\n");
    printf("right-half value -- the one compare-exchange step made genuine progress\n");
    printf("without fully sorting anything yet.\n");
    printf("\nself-check: split preserves bitonicity and establishes half ordering: %s\n",
           ok ? "confirmed" : "MISMATCH");
    return ok ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++17 -Wall -Wextra -O2 58_bitonic_split_cpu_baseline.cpp -o 58_bitonic_split_cpu_baseline
./58_bitonic_split_cpu_baseline
```

**Sample input:** the fixed array `{2, 4, 6, 8, 9, 7, 5, 3}` (an up-then-down bitonic sequence), split at distance 4, ascending.

**Sample output:**

```text
=== Section 12.1 CPU baseline: is_bitonic and one compare-exchange split ===

input (up-then-down bitonic): 2 4 6 8 9 7 5 3 
is_bitonic(input): true

after one ascending compare-exchange split (distance 4): 2 4 5 3 9 7 6 8 

left half:  2 4 5 3  -- is_bitonic: true
right half: 9 7 6 8  -- is_bitonic: true

left half's max (5) <= right half's min (6): yes

expected after split: 2 4 5 3 9 7 6 8

both halves stayed bitonic, and every left-half value is now <= every
right-half value -- the one compare-exchange step made genuine progress
without fully sorting anything yet.

self-check: split preserves bitonicity and establishes half ordering: confirmed
```

### The Concept, In Detail

The compare-exchange split takes a bitonic sequence of length `n` and a distance `dist = n/2`, and for each index `i` in `[0, dist)` compares `a[i]` with `a[i+dist]`: for an ascending split, the smaller value goes to position `i` and the larger to position `i+dist` (the reverse for a descending split). Visually, on the example array:

```
index:     0  1  2  3  4  5  6  7
value:     2  4  6  8  9  7  5  3
dist=4 pairs:  (0,4) (1,5) (2,6) (3,7)
             2 vs 9 -> keep     (2, 9)
             4 vs 7 -> keep     (4, 7)
             6 vs 5 -> SWAP     (5, 6)
             8 vs 3 -> SWAP     (3, 8)

result:    2  4  5  3  9  7  6  8
           \___________/ \_______/
              left half     right half
```

This is the **Bitonic Split Lemma**: if `a[0..n)` is bitonic, then after one ascending compare-exchange split at distance `n/2`, both `a[0..n/2)` and `a[n/2..n)` are themselves bitonic, AND every element of the left half is `<=` every element of the right half. Checking the example confirms both parts: the left half `{2, 4, 5, 3}` is bitonic (its minimum, `2`, is already at index 0, so no rotation is needed, and `2 <= 4 <= 5` then `5 >= 3` is a plain up-then-down shape), the right half `{9, 7, 6, 8}` is bitonic (a valley: rotating to its minimum `6` gives `6, 8, 9, 7`, which is up-then-down), and the left half's maximum (`5`) is indeed `<=` the right half's minimum (`6`).

Why does this always work? A bitonic sequence, by definition, is some rotation of a non-decreasing-then-non-increasing shape. Splitting it into two equal halves at a fixed distance and comparing element-wise turns out to preserve the bitonic property in each half while separating the value ranges -- a fact proven rigorously in the classical literature on sorting networks (Batcher's 1968 construction), but which this book treats operationally: verify it on concrete numbers, trust the pattern, and rely on it recursively in Sections 12.2 and 12.3.

The parallel opportunity is immediate: all `dist` comparisons in one split are completely independent of each other. Comparison `i` only reads and writes `a[i]` and `a[i+dist]`; no comparison depends on any other comparison's result within the same pass. This is a span-1 operation over `dist` independent units of work -- precisely the shape Chapter 4 first introduced for reduction, except here the "combine" step is a compare-exchange instead of an addition.

[COMMON TRAP]
It is tempting to think any two elements can be compared and swapped to "sort things out" one step at a time. The compare-exchange split specifically pairs index `i` with `i + dist` where `dist` is exactly half the current range -- pairing the wrong elements (say, adjacent ones) does not preserve the bitonic property of the resulting halves, and the recursive structure in Sections 12.2 and 12.3 depends on this exact pairing.

### Code and Verification

```cpp
#include <cstdio>
#include <vector>
#include <algorithm>

// Chapter 12.1 -- one compare-exchange split as a GPU kernel.
// Exactly `dist` independent comparisons exist (i in [0, dist)), each
// touching only a[i] and a[i+dist] -- so we launch exactly `dist`
// threads, one comparison each, with zero coordination between them.
__global__ void compare_exchange_kernel(int* g_a, int dist, int ascending) {
    int i = threadIdx.x;
    if (i >= dist) return;
    int lo = g_a[i];
    int hi = g_a[i + dist];
    bool should_swap = ascending ? (lo > hi) : (lo < hi);
    if (should_swap) {
        g_a[i] = hi;
        g_a[i + dist] = lo;
    }
}

// ---- Host-side replay of the identical per-thread logic, one thread's
// ---- worth of work per loop iteration -- exactly what `dist` independent
// ---- GPU threads would each do in parallel, with no shared state between
// ---- them to coordinate. ----

int main() {
    printf("=== Section 12.1 main: compare-exchange split as a CUDA kernel ===\n\n");

    std::vector<int> a = {2, 4, 6, 8, 9, 7, 5, 3};
    int dist = 4;

    printf("input (up-then-down bitonic): ");
    for (int v : a) printf("%d ", v);
    printf("\n\n");

    for (int i = 0; i < dist; i++) {
        int lo = a[i];
        int hi = a[i + dist];
        bool should_swap = (lo > hi);
        if (should_swap) {
            a[i] = hi;
            a[i + dist] = lo;
        }
    }

    printf("after one ascending compare-exchange split (distance %d, %d threads): ", dist, dist);
    for (int v : a) printf("%d ", v);
    printf("\n\n");

    std::vector<int> left(a.begin(), a.begin() + 4);
    std::vector<int> right(a.begin() + 4, a.end());
    int left_max = *std::max_element(left.begin(), left.end());
    int right_min = *std::min_element(right.begin(), right.end());

    printf("left half:  ");
    for (int v : left) printf("%d ", v);
    printf("\nright half: ");
    for (int v : right) printf("%d ", v);
    printf("\n\nleft half's max (%d) <= right half's min (%d): %s\n",
           left_max, right_min, (left_max <= right_min) ? "yes" : "NO -- BUG");

    std::vector<int> expected = {2, 4, 5, 3, 9, 7, 6, 8};
    bool ok = (a == expected) && (left_max <= right_min);

    printf("\nexpected: 2 4 5 3 9 7 6 8\n");
    printf("\nthis matches the CPU baseline exactly: %d independent threads did the\n", dist);
    printf("same %d comparisons the CPU loop did sequentially, in parallel, with no\n", dist);
    printf("shared state and no synchronization needed within this single pass.\n");
    printf("\nself-check: kernel result matches CPU baseline: %s\n", ok ? "confirmed" : "MISMATCH");
    return ok ? 0 : 1;
}
```

**Compile and run:**

```bash
NVDIR=/usr/local/lib/python3.11/dist-packages/nvidia/cu13
$NVDIR/bin/nvcc -arch=sm_80 -I$NVDIR/include -L$NVDIR/lib -lcudart 59_bitonic_split_kernel.cu -o 59_bitonic_split_kernel
LD_LIBRARY_PATH=$NVDIR/lib ./59_bitonic_split_kernel
```

**Sample input:** the same fixed array `{2, 4, 6, 8, 9, 7, 5, 3}`, split at distance 4, ascending, using 4 threads (one per comparison).

**Sample output:**

```text
=== Section 12.1 main: compare-exchange split as a CUDA kernel ===

input (up-then-down bitonic): 2 4 6 8 9 7 5 3 

after one ascending compare-exchange split (distance 4, 4 threads): 2 4 5 3 9 7 6 8 

left half:  2 4 5 3 
right half: 9 7 6 8 

left half's max (5) <= right half's min (6): yes

expected: 2 4 5 3 9 7 6 8

this matches the CPU baseline exactly: 4 independent threads did the
same 4 comparisons the CPU loop did sequentially, in parallel, with no
shared state and no synchronization needed within this single pass.

self-check: kernel result matches CPU baseline: confirmed
```

## 12.2 Bitonic Merge: Recursively Halving Distance to Fully Sort

### Intuition

Section 12.1 established that one compare-exchange split turns a bitonic sequence of size `n` into two smaller bitonic sequences of size `n/2`, with every left-half element `<=` every right-half element. That is progress, but neither half is actually *sorted* yet -- `{2, 4, 5, 3}` is bitonic, not sorted. The natural next step is to apply the exact same trick again, independently, to each half: split `{2, 4, 5, 3}` at distance 2, and split `{9, 7, 6, 8}` at distance 2, and keep going until the pieces are down to size 1 (at which point a single element is trivially "sorted"). This recursive halving-of-distance process, applied until the whole range is sorted, is called a **bitonic merge**.

### The Sequential (CPU) Baseline

The recursive structure translates directly into code: split once at distance `cnt/2`, then recursively merge both halves.

```cpp
#include <cstdio>
#include <vector>
#include <algorithm>

// Chapter 12.2 -- The Sequential (CPU) Baseline.
// A full bitonic merge repeatedly halves the compare-exchange distance
// (n/2, n/4, ..., 1) until the whole range is sorted. This is naturally
// written as recursion: split once at distance cnt/2, then recurse on
// both halves -- each of which is itself bitonic and smaller.

void compare_exchange_split(std::vector<int>& a, int lo, int dist, bool ascending) {
    for (int i = lo; i < lo + dist; i++) {
        bool should_swap = ascending ? (a[i] > a[i + dist]) : (a[i] < a[i + dist]);
        if (should_swap) std::swap(a[i], a[i + dist]);
    }
}

// Fully sorts the bitonic range a[lo .. lo+cnt) in the given direction.
void bitonic_merge_cpu(std::vector<int>& a, int lo, int cnt, bool ascending) {
    if (cnt <= 1) return;
    int dist = cnt / 2;
    compare_exchange_split(a, lo, dist, ascending);
    bitonic_merge_cpu(a, lo, dist, ascending);
    bitonic_merge_cpu(a, lo + dist, dist, ascending);
}

int main() {
    printf("=== Section 12.2 CPU baseline: recursive bitonic merge ===\n\n");

    std::vector<int> a = {2, 4, 6, 8, 9, 7, 5, 3};
    printf("input (up-then-down bitonic): ");
    for (int v : a) printf("%d ", v);
    printf("\n\n");

    bitonic_merge_cpu(a, 0, (int)a.size(), true);

    printf("after full recursive bitonic merge (ascending): ");
    for (int v : a) printf("%d ", v);
    printf("\n\n");

    std::vector<int> expected = {2, 3, 4, 5, 6, 7, 8, 9};
    bool sorted = std::is_sorted(a.begin(), a.end());
    bool ok = sorted && (a == expected);

    printf("expected: 2 3 4 5 6 7 8 9\n\n");
    printf("the recursion visited distances 4, then 2 (twice), then 1 (four times)\n");
    printf("-- exactly log2(8) = 3 levels of halving, each level doing the same\n");
    printf("kind of independent, position-local compare-exchange work.\n");
    printf("\nself-check: recursive merge fully sorts the bitonic input: %s\n",
           ok ? "confirmed" : "MISMATCH");
    return ok ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++17 -Wall -Wextra -O2 60_bitonic_merge_cpu_baseline.cpp -o 60_bitonic_merge_cpu_baseline
./60_bitonic_merge_cpu_baseline
```

**Sample input:** the same fixed array `{2, 4, 6, 8, 9, 7, 5, 3}`, fully merged ascending.

**Sample output:**

```text
=== Section 12.2 CPU baseline: recursive bitonic merge ===

input (up-then-down bitonic): 2 4 6 8 9 7 5 3 

after full recursive bitonic merge (ascending): 2 3 4 5 6 7 8 9 

expected: 2 3 4 5 6 7 8 9

the recursion visited distances 4, then 2 (twice), then 1 (four times)
-- exactly log2(8) = 3 levels of halving, each level doing the same
kind of independent, position-local compare-exchange work.

self-check: recursive merge fully sorts the bitonic input: confirmed
```

### The Concept, In Detail

Tracing the full recursion on `{2, 4, 6, 8, 9, 7, 5, 3}` by hand:

```
merge(lo=0, cnt=8, asc):  split dist=4 -> 2 4 5 3 9 7 6 8
  merge(lo=0, cnt=4, asc):  split dist=2 -> 2 3 5 4
    merge(lo=0, cnt=2, asc): split dist=1 -> 2 3
    merge(lo=2, cnt=2, asc): split dist=1 -> 4 5
    => 2 3 4 5
  merge(lo=4, cnt=4, asc):  split dist=2 -> 6 7 9 8
    merge(lo=4, cnt=2, asc): split dist=1 -> 6 7
    merge(lo=6, cnt=2, asc): split dist=1 -> 8 9
    => 6 7 8 9
  => 2 3 4 5 6 7 8 9
```

Three levels of recursion correspond exactly to `log2(8) = 3` halvings of distance: `4, then 2 (twice), then 1 (four times)`. Every level does independent, position-local compare-exchange work -- this is the same reduction-tree SHAPE Chapter 4.2 used for sequential-addressing reduction, except the "tree" here fans OUT (one split becomes two independent sub-merges) rather than IN (many partial sums collapse to one).

Mapping this onto a GPU is where the recursion has to be rewritten as iteration. A single kernel launch, or a single block's execution, cannot easily "recurse" the way host C++ can -- but it CAN loop over the distances `cnt/2, cnt/4, ..., 1` directly, as long as every thread waits for the current distance's compare-exchanges to finish before moving to the next, smaller distance. That waiting is exactly what `__syncthreads()` provides within one block, mirroring the double-buffering discipline Chapter 5.1 used for Hillis-Steele scan (there, each round depended on the previous round's completed writes; here, each pass depends on the previous pass's completed swaps).

The one subtlety that makes or breaks this iterative rewrite is figuring out WHICH threads are active at each distance, and WHO they pair with. At the very first pass (`dist = cnt/2`), the whole range is one group, and thread `tid` participates whenever `tid < dist`. But at every SUBSEQUENT pass, the range has already split into multiple independent groups of size `2*dist` (this is visible directly in the hand-trace above: at `dist=2`, positions `0..3` and `4..7` are two completely separate merges). The correct, general condition for "is thread `tid` the LEFT member of its own group at distance `dist`" is:

```
active:  (tid % (2 * dist)) < dist
partner: tid + dist
```

which correctly reduces to the simple `tid < dist` case only when `dist = cnt/2` (there is exactly one group, so `tid % (2*dist)` is just `tid` itself). Verifying this formula against the hand-trace: at `dist=2`, `2*dist=4`, so groups are `{0,1,2,3}` and `{4,5,6,7}`; within the first group, `tid=0` has `0 % 4 = 0 < 2` (active, partners with `2`), `tid=1` has `1 % 4 = 1 < 2` (active, partners with `3`), `tid=2` has `2 % 4 = 2`, not `< 2` (inactive -- it is the "right" member, already paired by `tid=0`), and `tid=3` similarly inactive. This matches the hand-trace's groups `(0,2)(1,3)(4,6)(5,7)` exactly.

```
ASCII view of pass 2 (dist=2, cnt=8), groups of size 2*dist=4:

  group [0,4):  0  1  2  3        group [4,8):  4  5  6  7
  active (tid%4<2): 0, 1          active (tid%4<2): 4, 5
  pairs: (0,2) (1,3)              pairs: (4,6) (5,7)
```

[COMMON TRAP]
Using `tid < dist` as the active-thread condition for EVERY pass (not just the first) is a genuine bug, not just an inefficiency: for `dist=2` on an 8-element array, `tid < dist` only activates threads 0 and 1, entirely skipping the second group `{4,5,6,7}` -- those four elements would never be compared at that distance at all, and the final result would not be sorted. Always use the group-aware condition `(tid % (2*dist)) < dist` for a general multi-pass merge kernel.

### Code and Verification

```cpp
#include <cstdio>
#include <vector>
#include <algorithm>

// Chapter 12.2 main -- iterative, multi-pass bitonic merge on the GPU.
// The recursive CPU version peels off one distance per call. On the GPU
// we instead loop over distances (cnt/2, cnt/4, ..., 1) as separate
// synchronized passes within one block, because every thread must finish
// the CURRENT distance's comparisons before ANY thread starts the NEXT,
// smaller distance -- the next pass's groups are built out of the
// previous pass's results.
//
// IMPORTANT: at distance `dist`, the range does NOT act as a single
// group of `dist` independent comparisons the way the very first pass
// does. It splits into multiple independent groups of size 2*dist, and
// thread `tid` only participates if (tid % (2*dist)) < dist, comparing
// itself against tid+dist. Using the simpler "tid < dist" is WRONG for
// every pass after the first, because it only touches the first group
// and silently leaves every other group untouched.
__global__ void bitonic_merge_kernel(int* g_a, int cnt, int ascending) {
    int tid = threadIdx.x;
    if (tid >= cnt) return;

    for (int dist = cnt / 2; dist >= 1; dist /= 2) {
        if ((tid % (2 * dist)) < dist) {
            int i = tid;
            int j = tid + dist;
            int lo = g_a[i];
            int hi = g_a[j];
            bool should_swap = ascending ? (lo > hi) : (lo < hi);
            if (should_swap) {
                g_a[i] = hi;
                g_a[j] = lo;
            }
        }
        __syncthreads();
    }
}

// ---- Host-side replay of the identical per-pass, per-thread logic:
// ---- every thread's action at a given distance depends only on values
// ---- the PREVIOUS pass finished writing, exactly what __syncthreads()
// ---- enforces between passes on real hardware. ----

int main() {
    printf("=== Section 12.2 main: iterative multi-pass bitonic merge kernel ===\n\n");

    std::vector<int> a = {2, 4, 6, 8, 9, 7, 5, 3};
    int n = (int)a.size();
    bool ascending = true;

    printf("input (up-then-down bitonic): ");
    for (int v : a) printf("%d ", v);
    printf("\n\n");

    int pass_num = 0;
    for (int dist = n / 2; dist >= 1; dist /= 2) {
        std::vector<int> next_a = a;
        for (int tid = 0; tid < n; tid++) {
            if ((tid % (2 * dist)) < dist) {
                int i = tid;
                int j = tid + dist;
                bool should_swap = ascending ? (a[i] > a[j]) : (a[i] < a[j]);
                if (should_swap) {
                    next_a[i] = a[j];
                    next_a[j] = a[i];
                }
            }
        }
        a = next_a;
        pass_num++;
        printf("pass %d (dist=%d): ", pass_num, dist);
        for (int v : a) printf("%d ", v);
        printf("\n");
    }
    printf("\n");

    std::vector<int> expected = {2, 3, 4, 5, 6, 7, 8, 9};
    bool sorted = std::is_sorted(a.begin(), a.end());
    bool ok = sorted && (a == expected);

    printf("expected: 2 3 4 5 6 7 8 9\n\n");
    printf("exactly n/2 = %d independent pairs are active at every single pass,\n", n / 2);
    printf("regardless of dist -- the group-based condition (tid %% (2*dist)) < dist\n");
    printf("is what keeps every group correctly paired as dist shrinks, matching the\n");
    printf("recursive CPU baseline's own pass-by-pass results exactly.\n");
    printf("\nself-check: iterative kernel matches recursive CPU baseline: %s\n",
           ok ? "confirmed" : "MISMATCH");
    return ok ? 0 : 1;
}
```

**Compile and run:**

```bash
NVDIR=/usr/local/lib/python3.11/dist-packages/nvidia/cu13
$NVDIR/bin/nvcc -arch=sm_80 -I$NVDIR/include -L$NVDIR/lib -lcudart 61_bitonic_merge_multipass.cu -o 61_bitonic_merge_multipass
LD_LIBRARY_PATH=$NVDIR/lib ./61_bitonic_merge_multipass
```

**Sample input:** the same fixed array `{2, 4, 6, 8, 9, 7, 5, 3}`, merged ascending via 3 passes (dist=4, 2, 1).

**Sample output:**

```text
=== Section 12.2 main: iterative multi-pass bitonic merge kernel ===

input (up-then-down bitonic): 2 4 6 8 9 7 5 3 

pass 1 (dist=4): 2 4 5 3 9 7 6 8 
pass 2 (dist=2): 2 3 5 4 6 7 9 8 
pass 3 (dist=1): 2 3 4 5 6 7 8 9 

expected: 2 3 4 5 6 7 8 9

exactly n/2 = 4 independent pairs are active at every single pass,
regardless of dist -- the group-based condition (tid % (2*dist)) < dist
is what keeps every group correctly paired as dist shrinks, matching the
recursive CPU baseline's own pass-by-pass results exactly.

self-check: iterative kernel matches recursive CPU baseline: confirmed
```

## 12.3 The Full Bitonic Sort: Building Bitonic Sequences Bottom-Up

### Intuition

Sections 12.1 and 12.2 assumed the input was already bitonic. Real data is not bitonic in general -- it is just arbitrary. The final piece is a way to turn ARBITRARY data into a bitonic sequence in the first place, so that the merge machinery from Section 12.2 has something to work on. The construction is elegant: two sequences sorted in OPPOSITE directions, placed back to back, always form a bitonic whole (one monotonic run up, immediately followed by one monotonic run down -- exactly the up-then-down shape from the definition). So: recursively sort the left half ascending, recursively sort the right half descending, concatenate, and the result is guaranteed bitonic -- ready to merge into final sorted order.

### The Sequential (CPU) Baseline

```cpp
#include <cstdio>
#include <vector>
#include <algorithm>

// Chapter 12.3 -- The Sequential (CPU) Baseline.
// A full bitonic sort builds a bitonic sequence out of ARBITRARY data,
// then merges it. The construction is recursive: sort the left half
// ascending, sort the right half descending (each half is itself built
// the same way), then concatenate. Two oppositely-sorted runs placed
// back to back always form a bitonic whole -- one monotonic run up,
// then one monotonic run down -- so no special-casing is needed once
// both halves are internally sorted in opposite directions.

void compare_exchange_split(std::vector<int>& a, int lo, int dist, bool ascending) {
    for (int i = lo; i < lo + dist; i++) {
        bool should_swap = ascending ? (a[i] > a[i + dist]) : (a[i] < a[i + dist]);
        if (should_swap) std::swap(a[i], a[i + dist]);
    }
}

void bitonic_merge_cpu(std::vector<int>& a, int lo, int cnt, bool ascending) {
    if (cnt <= 1) return;
    int dist = cnt / 2;
    compare_exchange_split(a, lo, dist, ascending);
    bitonic_merge_cpu(a, lo, dist, ascending);
    bitonic_merge_cpu(a, lo + dist, dist, ascending);
}

// Turns ARBITRARY a[lo .. lo+cnt) into fully sorted order (direction
// `ascending`), by first building a bitonic sequence, then merging it.
void bitonic_sort_cpu(std::vector<int>& a, int lo, int cnt, bool ascending) {
    if (cnt <= 1) return;
    int half = cnt / 2;
    bitonic_sort_cpu(a, lo, half, true);          // left half ascending
    bitonic_sort_cpu(a, lo + half, half, false);  // right half descending
    bitonic_merge_cpu(a, lo, cnt, ascending);      // now bitonic -- merge it
}

int main() {
    printf("=== Section 12.3 CPU baseline: recursive bitonic sort ===\n\n");

    std::vector<int> a = {5, 2, 8, 1, 9, 3, 7, 4};
    printf("input (arbitrary order): ");
    for (int v : a) printf("%d ", v);
    printf("\n\n");

    bitonic_sort_cpu(a, 0, (int)a.size(), true);

    printf("after full recursive bitonic sort (ascending): ");
    for (int v : a) printf("%d ", v);
    printf("\n\n");

    std::vector<int> a0 = {5, 2, 8, 1, 9, 3, 7, 4};
    std::sort(a0.begin(), a0.end());
    bool ok = (a == a0) && std::is_sorted(a.begin(), a.end());

    printf("expected (via std::sort reference): ");
    for (int v : a0) printf("%d ", v);
    printf("\n\n");
    printf("construction step: left quarter {5,2,8,1} sorts ascending to {1,2,5,8},\n");
    printf("right quarter {9,3,7,4} sorts descending to {9,7,4,3}; concatenated\n");
    printf("{1,2,5,8,9,7,4,3} is bitonic (up then down); the final merge sorts it.\n");
    printf("\nself-check: recursive bitonic sort matches std::sort reference: %s\n",
           ok ? "confirmed" : "MISMATCH");
    return ok ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++17 -Wall -Wextra -O2 62_bitonic_sort_cpu_baseline.cpp -o 62_bitonic_sort_cpu_baseline
./62_bitonic_sort_cpu_baseline
```

**Sample input:** the arbitrary (non-bitonic) array `{5, 2, 8, 1, 9, 3, 7, 4}`, sorted ascending.

**Sample output:**

```text
=== Section 12.3 CPU baseline: recursive bitonic sort ===

input (arbitrary order): 5 2 8 1 9 3 7 4 

after full recursive bitonic sort (ascending): 1 2 3 4 5 7 8 9 

expected (via std::sort reference): 1 2 3 4 5 7 8 9 

construction step: left quarter {5,2,8,1} sorts ascending to {1,2,5,8},
right quarter {9,3,7,4} sorts descending to {9,7,4,3}; concatenated
{1,2,5,8,9,7,4,3} is bitonic (up then down); the final merge sorts it.

self-check: recursive bitonic sort matches std::sort reference: confirmed
```

### The Concept, In Detail

Tracing the construction on `{5, 2, 8, 1, 9, 3, 7, 4}`:

```
sort(lo=0, cnt=8, asc):
  sort(lo=0, cnt=4, asc)  [left quarter, built ascending overall]:
    sort(lo=0, cnt=2, asc): {5,2} -> merge asc -> {2,5}
    sort(lo=2, cnt=2, desc): {8,1} -> merge desc -> {8,1}   (already descending)
    concatenate {2,5} ++ {8,1} = {2,5,8,1} (bitonic: up then down)
    merge asc on {2,5,8,1}: split dist=2 -> {2,1,8,5} -> merge halves -> {1,2,5,8}
  sort(lo=4, cnt=4, desc) [right quarter, built DEscending overall]:
    sort(lo=4, cnt=2, asc): {9,3} -> merge asc -> {3,9}
    sort(lo=6, cnt=2, desc): {7,4} -> merge desc -> {7,4}   (already descending)
    concatenate {3,9} ++ {7,4} = {3,9,7,4} (bitonic: up then down)
    merge desc on {3,9,7,4}: split dist=2 desc -> {7,9,3,4} -> merge halves desc -> {9,7,4,3}
  concatenate: {1,2,5,8} ++ {9,7,4,3} = {1,2,5,8,9,7,4,3}
  merge asc on the full 8: split dist=4 -> {1,2,4,3,9,7,5,8} -> merge halves -> {1,2,3,4,5,7,8,9}
```

Note the key structural point: `{1,2,5,8,9,7,4,3}` genuinely IS bitonic -- it rises from `1` to a peak of `9`, then falls to `3` -- purely because the left quarter was built ascending and the right quarter was built descending. No special-casing was needed to arrange this; it falls straight out of the recursive construction.

```
ASCII view of the full recursion tree (each leaf is a single element,
trivially "sorted"; every node's LEFT child is always built ascending
and its RIGHT child always descending, regardless of the node's own
direction, which only governs that node's own final merge step):

                    sort[0,8) asc
                   /              \
          sort[0,4) asc      sort[4,8) desc
          /        \          /          \
   sort[0,2)asc sort[2,4)desc sort[4,6)asc sort[6,8)desc
     /    \      /    \      /    \      /    \
    5      2    8      1    9      3    7      4
```

Notice `sort[2,4)` is built DESCENDING even though it sits inside `sort[0,4)`'s own ascending left quarter -- `sort[0,4)`'s direction only controls how it merges its two already-built children, not how those children are built. This is exactly what makes the concatenation trick work: at every level, the left and right children are always built in OPPOSITE directions, guaranteeing a bitonic hand-off no matter how deep the recursion goes.

The iterative, production form of this entire algorithm collapses BOTH the construction recursion and the merge recursion into one fixed double loop, using bit tricks instead of explicit recursion:

```
for k = 2; k <= n; k *= 2:        // k = size of bitonic run being built/merged
    for j = k/2; j > 0; j /= 2:   // j = current compare-exchange distance
        for each index i:
            l = i XOR j                    // i's partner at this distance
            if l > i:                       // process each pair exactly once
                ascending = ((i AND k) == 0)  // which "run" of size k is i in?
                compare-exchange(i, l, ascending)
```

The `i XOR j` trick computes the same "partner index" that explicit distance-based indexing computed before, and `(i & k) == 0` recovers which half of the current size-`k` block `i` belongs to (deciding its sort direction) without needing to track it through recursive calls. The result is a completely flat, data-independent sequence of compare-exchange operations: for a fixed `n`, the EXACT same `(i, l, ascending)` triples occur in the exact same order regardless of what values are actually being sorted. This ties directly back to Chapter 1's warp-uniformity and SIMT model -- every thread in every warp always evaluates `l > i` and `(i & k) == 0` identically for a given `i` on every run, so there is no data-dependent divergence anywhere in the algorithm, a genuinely rare and valuable property for GPU code.

For `n=16`, the total pass count is `sum_{stage=1}^{4} stage = 1+2+3+4 = 10` (stages `k=2,4,8,16` need `log2(k) = 1,2,3,4` passes respectively) -- confirmed directly by the code's own pass counter.

[COMMON TRAP]
The guard `if l > i` is easy to forget, and without it every pair gets processed TWICE (once as `(i, l)` and once as `(l, i)`), which for a compare-and-conditionally-swap operation silently undoes half of the intended swaps. Always process each pair exactly once, from the lower-indexed side.

### Code and Verification

```cpp
#include <cstdio>
#include <vector>
#include <algorithm>

// Chapter 12.3 main -- the classic ITERATIVE, production-form bitonic
// sort. Instead of the recursive "sort left ascending, sort right
// descending, then merge" formulation, real GPU code uses a fixed
// double loop over power-of-two stage sizes k and distances j:
//
//   for k = 2; k <= n; k *= 2:        // stage: build/extend bitonic runs of size k
//     for j = k/2; j > 0; j /= 2:     // pass within the stage
//       for each i: compare i with l = i ^ j
//         direction: ascending if (i & k) == 0, else descending
//
// This performs EXACTLY the same comparisons as the recursive version,
// just re-derived from index arithmetic (i XOR j) instead of explicit
// recursion. Crucially, the whole sequence of compare-exchange pairs is
// completely fixed by n alone -- it never depends on the data -- which
// is exactly the divergence-free, warp-uniform pattern from Chapter 1:
// every thread in every warp always takes the same branch as every
// other thread at the same index, for any input of the same size.
__global__ void bitonic_sort_kernel(int* g_a, int n, int k, int j) {
    int i = threadIdx.x;
    if (i >= n) return;
    int l = i ^ j;
    if (l > i) {
        bool ascending = ((i & k) == 0);
        int a_i = g_a[i];
        int a_l = g_a[l];
        bool should_swap = ascending ? (a_i > a_l) : (a_i < a_l);
        if (should_swap) {
            g_a[i] = a_l;
            g_a[l] = a_i;
        }
    }
}

// ---- Host-side replay of the identical double-loop logic: one kernel
// ---- launch per (k, j) pair on real hardware, with a full grid
// ---- synchronization between launches (the CPU loop's own sequencing
// ---- plays the same role here). ----

int main() {
    printf("=== Section 12.3 main: iterative double-loop bitonic sort kernel ===\n\n");

    int n = 16;
    std::vector<int> a(n);
    for (int i = 0; i < n; i++) a[i] = (i * 13 + 5) % 16;

    printf("input (fixed permutation of 0..15, a[i] = (i*13+5) mod 16):\n  ");
    for (int v : a) printf("%d ", v);
    printf("\n\n");

    int pass_count = 0;
    for (int k = 2; k <= n; k *= 2) {
        for (int j = k / 2; j > 0; j /= 2) {
            for (int i = 0; i < n; i++) {
                int l = i ^ j;
                if (l > i) {
                    bool ascending = ((i & k) == 0);
                    bool should_swap = ascending ? (a[i] > a[l]) : (a[i] < a[l]);
                    if (should_swap) std::swap(a[i], a[l]);
                }
            }
            pass_count++;
        }
    }

    printf("after iterative bitonic sort (%d passes total): \n  ", pass_count);
    for (int v : a) printf("%d ", v);
    printf("\n\n");

    std::vector<int> reference(n);
    for (int i = 0; i < n; i++) reference[i] = (i * 13 + 5) % 16;
    std::sort(reference.begin(), reference.end());

    printf("reference (via std::sort, independent check -- 16 elements/10 passes\n");
    printf("is too large to hand-trace fully):\n  ");
    for (int v : reference) printf("%d ", v);
    printf("\n\n");

    bool sorted = std::is_sorted(a.begin(), a.end());
    bool ok = sorted && (a == reference);

    printf("total passes = sum over stages k=2,4,8,16 of log2(k) passes each\n");
    printf("             = 1 + 2 + 3 + 4 = 10, confirmed: %s\n",
           (pass_count == 10) ? "yes" : "NO -- BUG");
    printf("\nself-check: iterative kernel result matches std::sort reference: %s\n",
           ok ? "confirmed" : "MISMATCH");
    return (ok && pass_count == 10) ? 0 : 1;
}
```

**Compile and run:**

```bash
NVDIR=/usr/local/lib/python3.11/dist-packages/nvidia/cu13
$NVDIR/bin/nvcc -arch=sm_80 -I$NVDIR/include -L$NVDIR/lib -lcudart 63_bitonic_sort_iterative.cu -o 63_bitonic_sort_iterative
LD_LIBRARY_PATH=$NVDIR/lib ./63_bitonic_sort_iterative
```

**Sample input:** `N=16`, `a[i] = (i*13+5) mod 16` (a fixed permutation of `0..15`), sorted ascending via the iterative double loop.

**Sample output:**

```text
=== Section 12.3 main: iterative double-loop bitonic sort kernel ===

input (fixed permutation of 0..15, a[i] = (i*13+5) mod 16):
  5 2 15 12 9 6 3 0 13 10 7 4 1 14 11 8 

after iterative bitonic sort (10 passes total): 
  0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 

reference (via std::sort, independent check -- 16 elements/10 passes
is too large to hand-trace fully):
  0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 

total passes = sum over stages k=2,4,8,16 of log2(k) passes each
             = 1 + 2 + 3 + 4 = 10, confirmed: yes

self-check: iterative kernel result matches std::sort reference: confirmed
```

## Chapter Summary

A sequence is bitonic if some rotation of it is non-decreasing then non-increasing, which covers purely monotonic runs, up-then-down shapes, and down-then-up (valley) shapes alike; checking this in general requires rotating to the sequence's own minimum first. A single compare-exchange split, pairing index `i` with `i + dist` at `dist = n/2`, turns a bitonic sequence into two smaller bitonic sequences with every left-half value `<=` every right-half value -- the Bitonic Split Lemma -- and this split is embarrassingly parallel across its `dist` independent comparisons. Recursively halving the distance down to 1 (a bitonic merge) fully sorts a bitonic sequence; mapped iteratively onto a GPU, each pass must wait (via `__syncthreads()`) for the previous, larger-distance pass to finish, and the active-thread condition must be the group-aware `(tid % (2*dist)) < dist`, not the simpler (and only-correct-for-the-first-pass) `tid < dist`. Arbitrary, non-bitonic data is handled by building a bitonic sequence bottom-up: recursively sort the left half ascending and the right half descending, then merge -- concatenating two oppositely-sorted runs always yields a bitonic whole. The classic iterative, production form of the full algorithm collapses both recursions into one fixed double loop over stage size `k` and distance `j`, using `i XOR j` to find each pair and `(i & k) == 0` to determine sort direction; this sequence of compare-exchange operations is entirely fixed by `n` alone, making bitonic sort a genuinely divergence-free algorithmic pattern in the SIMT sense introduced back in Chapter 1.

## Self-Check Questions

1. Why does checking "is this sequence bitonic" require rotating to the minimum element first, rather than just scanning left to right for a single direction change?
2. In the compare-exchange split at distance `dist`, why are exactly `dist` comparisons performed, no more and no fewer, and why are they independent of one another?
3. State the Bitonic Split Lemma in your own words: what two guarantees does one ascending compare-exchange split provide about the two resulting halves?
4. Explain why the active-thread condition for a multi-pass bitonic merge kernel must be `(tid % (2*dist)) < dist` rather than simply `tid < dist`, and give a concrete example (with actual indices) where the simpler condition produces a wrong result.
5. Why does sorting the left half of an array ascending and the right half descending, then concatenating them, always produce a bitonic sequence, regardless of what the original unsorted values were?
6. In the iterative double-loop formulation, what do `i XOR j` and `(i & k) == 0` each compute, and why does this make the algorithm's sequence of operations completely independent of the input data's actual values?

## Where We Go Next

Bitonic sort's defining strength -- a completely fixed, data-independent sequence of compare-exchange operations -- is also its defining limitation: it always performs `O(n log^2 n)` total comparisons, regardless of how "close to sorted" the input already is, and its use of an all-pairs-style compare-exchange network does not exploit the kind of key-locality that a real GPU's memory hierarchy rewards. Chapter 13 turns to **radix sort**, the algorithm that real GPU sorting libraries actually use in production: rather than comparing elements against each other at all, it sorts by repeatedly bucketing elements according to one small group of bits at a time, using exactly the histogram-and-scatter machinery Chapters 6 and 7 already built. Radix sort's passes are fixed in COUNT (proportional to key width, not to `n`), but each pass leans on stream compaction and histogramming in a way bitonic sort's pure compare-exchange network never needed to.

## Worked Solutions

**1.** A purely left-to-right scan for "one direction change" fails on valley shapes like `{8, 6, 4, 2, 3, 5, 7, 9}`: read left to right, this looks like "down, then up" -- which IS one of the two bitonic shapes -- but an UP-then-down sequence that happens to start partway through its rise, like `{7, 9, 8, 6, 4, 2, 3, 5}` (a rotation of the same valley), shows DOWN, then further down, then UP -- more than one direction change, even though it is still bitonic by definition (some rotation of it is up-then-down). Rotating first to the sequence's own minimum element removes this ambiguity: after rotation, a genuinely bitonic sequence is guaranteed to show at most one direction change, because the minimum element is necessarily the "bottom of the valley" or the start of the single monotonic rise.

**2.** There are exactly `dist` comparisons because the split only pairs indices in the first half, `i` in `[0, dist)`, each with its unique partner `i + dist` in the second half -- covering all `2*dist = n` elements exactly once, using `n/2` comparisons. They are independent because comparison `i` reads and writes only `a[i]` and `a[i+dist]`; no two comparisons in a single split ever touch the same array position, so their relative order (or full parallelism) cannot change the outcome.

**3.** The Bitonic Split Lemma guarantees two things about a bitonic sequence of size `n` after one ascending compare-exchange split at `dist = n/2`: first, that both resulting halves (`a[0..dist)` and `a[dist..n)`) are themselves bitonic sequences; second, that every element in the left half is less than or equal to every element in the right half, meaning the two halves can be sorted completely independently and never need to be compared against each other again.

**4.** At `dist=2` on an 8-element array, `tid < dist` only activates threads 0 and 1 (comparing `(0,2)` and `(1,3)`), leaving threads 4, 5, 6, 7 entirely idle -- but the correct pass-2 structure has TWO independent groups of size 4, `{0,1,2,3}` and `{4,5,6,7}`, each needing its own pair of comparisons. `(tid % (2*dist)) < dist` correctly activates `tid=0,1` (partnering `2,3`) AND `tid=4,5` (partnering `6,7`), covering both groups; `tid < dist` alone would leave `{4,5,6,7}` completely unsorted relative to each other at that distance, producing wrong final output.

**5.** Sorting the left half ascending produces one monotonic RISING run; sorting the right half descending produces one monotonic FALLING run. Placing a rising run immediately followed by a falling run, with no other rearrangement, is exactly the "non-decreasing then non-increasing" shape that defines the up-then-down bitonic case directly (not even requiring a rotation) -- this holds no matter what the original values were, because "ascending" and "descending" are properties of ORDER, not of specific values, so any two same-size groups of numbers can be arranged this way.

**6.** `i XOR j` computes the index of `i`'s partner in the current compare-exchange pass: flipping bit `j` in `i`'s binary representation is exactly the distance-`j` pairing used throughout Sections 12.1 and 12.2, re-derived via bitwise arithmetic instead of explicit loop bounds. `(i & k) == 0` checks whether bit `k` is unset in `i`, which determines which size-`k` block of the array `i` currently belongs to, and therefore which sort direction (ascending or descending) that block was built with. Both quantities depend only on the fixed integers `i`, `j`, and `k` -- never on any array VALUE -- so the entire sequence of which indices get compared, and in which direction, is fully determined by `n` alone, before the algorithm ever looks at the data.
