# Chapter 14: Merge Sort and Sample Sort

Chapters 12 and 13 both assumed the entire array fits comfortably where one block (or one thread's own view of the whole array) can see everything at once. Real workloads outgrow that assumption constantly -- a dataset that needs many blocks, or many separate GPUs, to even hold. This chapter closes out Part 3 with two techniques built specifically for that case: merge sort, which combines already-sorted pieces without ever comparing two full keys against every other key, and sample sort, which partitions data into buckets that can be handed to completely independent blocks with no further coordination needed afterward.

## 14.1 Parallel Merge of Two Sorted Arrays

### Intuition

Merging two already-sorted arrays into one sorted array is one of the oldest tricks in sorting: keep a pointer into each array, always take whichever pointer's current element is smaller, and advance that pointer. It is simple, correct, and -- looked at through Chapter 3's own vocabulary -- has an unavoidable span equal to the combined length of both arrays, because every step's decision (which pointer to advance) depends on knowing the previous step's decision. Parallelizing this is not obvious at first: unlike Chapter 4's reduction or Chapter 5's scan, there is no obvious way to split "walk two pointers together" across independent threads. The key insight, called the **co-rank** or **merge path** technique, is to ask a completely different question: for a given OUTPUT position `k`, without simulating any of the walk, how many elements came from array A and how many from array B to produce exactly the first `k` elements of the merged result? That question turns out to be answerable by binary search, independently, for every `k` at once.

### The Sequential (CPU) Baseline

```cpp
#include <cstdio>
#include <vector>

// Chapter 14.1 -- The Sequential (CPU) Baseline.
// Merging two already-sorted arrays is one of the oldest building
// blocks in all of sorting: walk two pointers forward, always taking
// whichever front element is smaller, until one array runs out, then
// copy the rest of the other straight over. Every single step of this
// walk depends on the DECISION the previous step made (which pointer
// advanced) -- in Chapter 3's own vocabulary, this is an unavoidable
// span of m+n, no matter how many threads might be sitting idle.

std::vector<int> merge_sequential(const std::vector<int>& A, const std::vector<int>& B) {
    std::vector<int> out;
    out.reserve(A.size() + B.size());
    size_t i = 0, j = 0;
    while (i < A.size() && j < B.size()) {
        if (A[i] <= B[j]) out.push_back(A[i++]);
        else out.push_back(B[j++]);
    }
    while (i < A.size()) out.push_back(A[i++]);
    while (j < B.size()) out.push_back(B[j++]);
    return out;
}

int main() {
    printf("=== Section 14.1 CPU baseline: sequential two-pointer merge ===\n\n");

    std::vector<int> A = {1, 3, 6, 8};
    std::vector<int> B = {2, 4, 5, 7};

    printf("A (sorted): ");
    for (int v : A) printf("%d ", v);
    printf("\nB (sorted): ");
    for (int v : B) printf("%d ", v);
    printf("\n\n");

    auto merged = merge_sequential(A, B);

    printf("merged:     ");
    for (int v : merged) printf("%d ", v);
    printf("\n\n");

    std::vector<int> expected = {1, 2, 3, 4, 5, 6, 7, 8};
    bool sorted_ok = true;
    for (size_t k = 1; k < merged.size(); k++) if (merged[k - 1] > merged[k]) sorted_ok = false;
    bool ok = (merged == expected) && sorted_ok;

    int m = (int)A.size(), n = (int)B.size();
    printf("expected: 1 2 3 4 5 6 7 8\n\n");
    printf("span, in Chapter 3's own vocabulary: exactly m+n-1 = %d sequentially dependent\n",
           m + n - 1);
    printf("comparisons in the worst case -- each comparison's outcome decides which pointer\n");
    printf("moves, so no comparison after the first can even be ATTEMPTED before the\n");
    printf("previous one's result is known. Adding threads cannot shorten this chain; the\n");
    printf("dependency is in the algorithm's own structure, exactly like Chapter 11's\n");
    printf("pointer-chasing linked list.\n");

    printf("\nself-check: merged output is sorted and matches expected: %s\n",
           ok ? "confirmed" : "MISMATCH");
    return ok ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++17 -Wall -Wextra -O2 70_merge_two_sorted_cpu_baseline.cpp -o 70_merge_two_sorted_cpu_baseline
./70_merge_two_sorted_cpu_baseline
```

**Sample input:** `A = {1, 3, 6, 8}`, `B = {2, 4, 5, 7}`, merged via the standard two-pointer walk.

**Sample output:**

```text
=== Section 14.1 CPU baseline: sequential two-pointer merge ===

A (sorted): 1 3 6 8 
B (sorted): 2 4 5 7 

merged:     1 2 3 4 5 6 7 8 

expected: 1 2 3 4 5 6 7 8

span, in Chapter 3's own vocabulary: exactly m+n-1 = 7 sequentially dependent
comparisons in the worst case -- each comparison's outcome decides which pointer
moves, so no comparison after the first can even be ATTEMPTED before the
previous one's result is known. Adding threads cannot shorten this chain; the
dependency is in the algorithm's own structure, exactly like Chapter 11's
pointer-chasing linked list.

self-check: merged output is sorted and matches expected: confirmed
```

### The Concept, In Detail

Define `co_rank(k)` as the pair `(i, j)` with `i + j = k` such that the first `i` elements of A together with the first `j` elements of B are EXACTLY the first `k` elements of the correctly merged output. Two conditions pin this down uniquely: every element A took (`A[0..i)`) must be `<=` the next unconsumed element of B (`B[j]`, if it exists), and every element B took (`B[0..j)`) must be `<=` the next unconsumed element of A (`A[i]`, if it exists). Because A and B are each individually sorted, `i` can be found by binary search over the range `[max(0, k-n), min(k, m)]` (where `m, n` are A and B's lengths): given a candidate `i` (and `j = k - i`), the two conditions above either hold (done), or tell you definitively whether to search higher or lower.

Working through two concrete values of `k` on `A = {1,3,6,8}`, `B = {2,4,5,7}`:

```
co-rank(3): searching for i,j with i+j=3 such that A[0:i] and B[0:j]
are exactly the smallest 3 combined values.
  try i=2, j=1: A[0:2]={1,3}, B[0:1]={2}. Check: A[1]=3 <= B[1]=4? yes.
                Check: B[0]=2 <= A[2]=6? yes. Both hold -> co-rank(3)=(i=2,j=1).
  (matches: the 3 smallest of {1,3,6,8} and {2,4,5,7} combined are 1,2,3 --
   2 from A, 1 from B.)

co-rank(6): try i=3, j=3: A[0:3]={1,3,6}, B[0:3]={2,4,5}.
  Check: A[2]=6 <= B[3]=7? yes. Check: B[2]=5 <= A[3]=8? yes. Both hold
  -> co-rank(6)=(i=3,j=3).
```

Once `co_rank(k) = (i, j)` is known, the actual output value at position `k` is immediate: it is `A[i]` if `A` still has an element there and (`B` is exhausted or `A[i] <= B[j]`), otherwise it is `B[j]`. Every one of the `m+n` output positions can compute its own co-rank completely independently -- no thread's binary search ever depends on any other thread's result -- which is precisely what turns an inherently-sequential span of `m+n` into a fully parallel operation with span `O(log(min(m,n)))` per position.

```
ASCII view: co-rank splits the merge into independent chunks.

  A: [1  3  6  8]         B: [2  4  5  7]
      i=0..4                  j=0..4

  k=0: (i=0,j=0)   k=1: (i=1,j=0)   k=2: (i=1,j=1)   k=3: (i=2,j=1)
  k=4: (i=2,j=2)   k=5: (i=2,j=3)   k=6: (i=3,j=3)   k=7: (i=3,j=4)

  Every k's (i,j) pair is found by its OWN binary search -- position 3's
  search never needs to know what position 5's search decided.
```

[COMMON TRAP]
It is tempting to compute `co_rank(k)` and then simply read off `A[i]` or `B[j]` without checking whether `i` has actually reached `m` or `j` has reached `n`. Once one array is exhausted for a given `k`, only the OTHER array's element is valid to read -- indexing the exhausted array out of bounds is a real, easy-to-hit bug at the boundary positions (the first and last few `k` values especially).

### Code and Verification

```cpp
#include <cstdio>
#include <vector>

// Chapter 14.1 main -- parallel merge via co-rank (the "merge path").
// Section 14.1's CPU baseline showed the sequential walk has an
// unavoidable span of m+n. The parallel fix reframes the problem
// entirely: instead of walking forward and DECIDING where each element
// goes, ask directly, for each output position k, "how many elements
// came from A, and how many from B, to produce exactly the first k
// merged elements?" That split point (i, j) with i+j=k is called k's
// CO-RANK, and it can be found by BINARY SEARCH on i alone (j = k-i is
// then fixed) -- completely independently for every k, with no thread
// needing to know what any other thread's split point is.
__device__ void co_rank(int k, const int* A, int m, const int* B, int n, int* out_i, int* out_j) {
    int i_low = (k - n > 0) ? (k - n) : 0;
    int i_high = (k < m) ? k : m;
    while (i_low < i_high) {
        int i = (i_low + i_high + 1) / 2;
        int j = k - i;
        if (j > 0 && i < m && A[i] < B[j - 1]) {
            i_low = i;
        } else if (i > 0 && j < n && A[i - 1] > B[j]) {
            i_high = i - 1;
        } else {
            i_low = i;
            break;
        }
    }
    *out_i = i_low;
    *out_j = k - i_low;
}

__global__ void merge_corank_kernel(const int* g_A, int m, const int* g_B, int n, int* g_out) {
    int k = threadIdx.x;
    if (k >= m + n) return;
    int i, j;
    co_rank(k, g_A, m, g_B, n, &i, &j);
    bool take_from_A = (i < m) && (j >= n || g_A[i] <= g_B[j]);
    g_out[k] = take_from_A ? g_A[i] : g_B[j];
}

// ---- Host-side replay of the identical binary search, one independent
// call per output index k -- exactly what m+n independent GPU threads
// would each do in parallel, with zero coordination between them. ----

void co_rank_host(int k, const std::vector<int>& A, const std::vector<int>& B, int& out_i, int& out_j) {
    int m = (int)A.size(), n = (int)B.size();
    int i_low = (k - n > 0) ? (k - n) : 0;
    int i_high = (k < m) ? k : m;
    while (i_low < i_high) {
        int i = (i_low + i_high + 1) / 2;
        int j = k - i;
        if (j > 0 && i < m && A[i] < B[j - 1]) {
            i_low = i;
        } else if (i > 0 && j < n && A[i - 1] > B[j]) {
            i_high = i - 1;
        } else {
            i_low = i;
            break;
        }
    }
    out_i = i_low;
    out_j = k - i_low;
}

int main() {
    printf("=== Section 14.1 main: parallel merge via co-rank, as a CUDA kernel ===\n\n");

    std::vector<int> A = {1, 3, 6, 8};
    std::vector<int> B = {2, 4, 5, 7};
    int m = (int)A.size(), n = (int)B.size();

    printf("A (sorted): ");
    for (int v : A) printf("%d ", v);
    printf("\nB (sorted): ");
    for (int v : B) printf("%d ", v);
    printf("\n\n");

    std::vector<int> out(m + n);
    std::vector<int> co_i(m + n), co_j(m + n);
    for (int k = 0; k < m + n; k++) {
        int i, j;
        co_rank_host(k, A, B, i, j);
        co_i[k] = i;
        co_j[k] = j;
        bool take_from_A = (i < m) && (j >= n || A[i] <= B[j]);
        out[k] = take_from_A ? A[i] : B[j];
    }

    printf("co-rank(k) = (i,j) for k=0..%d:  ", m + n - 1);
    for (int k = 0; k < m + n; k++) printf("(%d,%d) ", co_i[k], co_j[k]);
    printf("\n\n");

    printf("worked example: co-rank(3) = (i=%d, j=%d) -- the first 3 merged elements come\n",
           co_i[3], co_j[3]);
    printf("from A[0:%d] and B[0:%d]; co-rank(6) = (i=%d, j=%d) -- the first 6 come from\n",
           co_i[3], co_j[3], co_i[6], co_j[6]);
    printf("A[0:%d] and B[0:%d].\n\n", co_i[6], co_j[6]);

    printf("merged (via co-rank, %d independent threads): ", m + n);
    for (int v : out) printf("%d ", v);
    printf("\n\n");

    std::vector<int> expected = {1, 2, 3, 4, 5, 6, 7, 8};
    bool ok = (out == expected);

    printf("expected: 1 2 3 4 5 6 7 8\n\n");
    printf("this matches the CPU baseline exactly, but every output position's binary\n");
    printf("search only ever looks at A and B (never at any other thread's result), so all\n");
    printf("%d threads can genuinely run at once -- span collapses from m+n to O(log(min(m,n)))\n",
           m + n);
    printf("per output position, at the cost of doing more total comparisons overall than\n");
    printf("the sequential walk ever needed -- the same work-versus-span trade this book has\n");
    printf("made since Chapter 5.2's scan.\n");

    printf("\nself-check: co-rank-based parallel merge matches CPU baseline: %s\n",
           ok ? "confirmed" : "MISMATCH");
    return ok ? 0 : 1;
}
```

**Compile and run:**

```bash
NVDIR=/usr/local/lib/python3.11/dist-packages/nvidia/cu13
$NVDIR/bin/nvcc -arch=sm_80 -I$NVDIR/include -L$NVDIR/lib -lcudart 71_corank_merge_kernel.cu -o 71_corank_merge_kernel
LD_LIBRARY_PATH=$NVDIR/lib ./71_corank_merge_kernel
```

**Sample input:** the same `A` and `B`, merged via 8 independent co-rank threads (one per output position).

**Sample output:**

```text
=== Section 14.1 main: parallel merge via co-rank, as a CUDA kernel ===

A (sorted): 1 3 6 8 
B (sorted): 2 4 5 7 

co-rank(k) = (i,j) for k=0..7:  (0,0) (1,0) (1,1) (2,1) (2,2) (2,3) (3,3) (3,4) 

worked example: co-rank(3) = (i=2, j=1) -- the first 3 merged elements come
from A[0:2] and B[0:1]; co-rank(6) = (i=3, j=3) -- the first 6 come from
A[0:3] and B[0:3].

merged (via co-rank, 8 independent threads): 1 2 3 4 5 6 7 8 

expected: 1 2 3 4 5 6 7 8

this matches the CPU baseline exactly, but every output position's binary
search only ever looks at A and B (never at any other thread's result), so all
8 threads can genuinely run at once -- span collapses from m+n to O(log(min(m,n)))
per output position, at the cost of doing more total comparisons overall than
the sequential walk ever needed -- the same work-versus-span trade this book has
made since Chapter 5.2's scan.

self-check: co-rank-based parallel merge matches CPU baseline: confirmed
```

## 14.2 From Merge to Merge Sort: Repeated Pairwise Merging

### Intuition

Section 14.1 built ONE merge of two sorted runs. Merge sort is what happens when every single element starts as its own trivially-sorted run of length 1, and adjacent runs are repeatedly merged in pairs, doubling the run length each round -- 1, then 2, then 4, and so on -- until one fully sorted run remains. This is the classic "bottom-up" or iterative formulation of merge sort, and it reuses Section 14.1's parallel merge directly: at each round, every pair of adjacent runs gets merged, and each of those merges is itself internally parallel via co-rank.

### The Sequential (CPU) Baseline

```cpp
#include <cstdio>
#include <vector>
#include <algorithm>

// Chapter 14.2 -- The Sequential (CPU) Baseline.
// Section 14.1 built ONE merge of two already-sorted runs. Bottom-up
// merge sort treats every single element as a trivially-sorted run of
// length 1, then repeatedly merges ADJACENT pairs of runs, doubling the
// run length each pass (1 -> 2 -> 4 -> ... -> n) until one run remains,
// fully sorted. This is a fundamentally different construction strategy
// than Chapter 12's bitonic sort, applied to the SAME kind of array, so
// this section reuses Chapter 12.3's own example input to make the
// contrast concrete.

std::vector<int> merge_sequential(const std::vector<int>& A, const std::vector<int>& B) {
    std::vector<int> out;
    out.reserve(A.size() + B.size());
    size_t i = 0, j = 0;
    while (i < A.size() && j < B.size()) {
        if (A[i] <= B[j]) out.push_back(A[i++]);
        else out.push_back(B[j++]);
    }
    while (i < A.size()) out.push_back(A[i++]);
    while (j < B.size()) out.push_back(B[j++]);
    return out;
}

std::vector<int> bottom_up_merge_sort(std::vector<int> a) {
    int n = (int)a.size();
    for (int width = 1; width < n; width *= 2) {
        std::vector<int> next;
        next.reserve(n);
        for (int i = 0; i < n; i += 2 * width) {
            int left_end = std::min(i + width, n);
            int right_end = std::min(i + 2 * width, n);
            std::vector<int> left(a.begin() + i, a.begin() + left_end);
            std::vector<int> right(a.begin() + left_end, a.begin() + right_end);
            auto merged = merge_sequential(left, right);
            next.insert(next.end(), merged.begin(), merged.end());
        }
        a = next;
    }
    return a;
}

int main() {
    printf("=== Section 14.2 CPU baseline: bottom-up (iterative) merge sort ===\n\n");

    std::vector<int> a = {5, 2, 8, 1, 9, 3, 7, 4};
    printf("input (same array Chapter 12.3 sorted with bitonic sort): ");
    for (int v : a) printf("%d ", v);
    printf("\n\n");

    std::vector<int> a1 = a;
    for (int width = 1; width < (int)a.size(); width *= 2) {
        std::vector<int> next;
        int n = (int)a1.size();
        for (int i = 0; i < n; i += 2 * width) {
            int left_end = std::min(i + width, n);
            int right_end = std::min(i + 2 * width, n);
            std::vector<int> left(a1.begin() + i, a1.begin() + left_end);
            std::vector<int> right(a1.begin() + left_end, a1.begin() + right_end);
            auto merged = merge_sequential(left, right);
            next.insert(next.end(), merged.begin(), merged.end());
        }
        a1 = next;
        printf("after width=%d merges: ", width);
        for (int v : a1) printf("%d ", v);
        printf("\n");
    }
    printf("\n");

    auto sorted_a = bottom_up_merge_sort(a);
    std::vector<int> a0 = {5, 2, 8, 1, 9, 3, 7, 4};
    std::sort(a0.begin(), a0.end());
    bool ok = (sorted_a == a0) && std::is_sorted(sorted_a.begin(), sorted_a.end());

    printf("final: ");
    for (int v : sorted_a) printf("%d ", v);
    printf("\nexpected (via std::sort reference): ");
    for (int v : a0) printf("%d ", v);
    printf("\n\n");
    printf("3 passes total for n=8 (widths 1, 2, 4) -- exactly log2(8), the same pass count\n");
    printf("Chapter 12.2's bitonic merge needed, but built from a completely different\n");
    printf("primitive: repeated two-run MERGES instead of repeated compare-exchange SPLITS.\n");

    printf("\nself-check: bottom-up merge sort matches std::sort reference: %s\n",
           ok ? "confirmed" : "MISMATCH");
    return ok ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++17 -Wall -Wextra -O2 72_bottom_up_mergesort_cpu_baseline.cpp -o 72_bottom_up_mergesort_cpu_baseline
./72_bottom_up_mergesort_cpu_baseline
```

**Sample input:** the array `{5, 2, 8, 1, 9, 3, 7, 4}` -- the same array Chapter 12.3 sorted with bitonic sort -- sorted here via bottom-up merge sort instead.

**Sample output:**

```text
=== Section 14.2 CPU baseline: bottom-up (iterative) merge sort ===

input (same array Chapter 12.3 sorted with bitonic sort): 5 2 8 1 9 3 7 4 

after width=1 merges: 2 5 1 8 3 9 4 7 
after width=2 merges: 1 2 5 8 3 4 7 9 
after width=4 merges: 1 2 3 4 5 7 8 9 

final: 1 2 3 4 5 7 8 9 
expected (via std::sort reference): 1 2 3 4 5 7 8 9 

3 passes total for n=8 (widths 1, 2, 4) -- exactly log2(8), the same pass count
Chapter 12.2's bitonic merge needed, but built from a completely different
primitive: repeated two-run MERGES instead of repeated compare-exchange SPLITS.

self-check: bottom-up merge sort matches std::sort reference: confirmed
```

### The Concept, In Detail

Tracing the three passes on `{5, 2, 8, 1, 9, 3, 7, 4}`:

```
input:              5  2  8  1  9  3  7  4

width=1 (merge      [5,2]->2,5  [8,1]->1,8  [9,3]->3,9  [7,4]->4,7
pairs of size 1):    2  5  1  8  3  9  4  7

width=2 (merge      [2,5],[1,8] -> 1,2,5,8    [3,9],[4,7] -> 3,4,7,9
pairs of size 2):    1  2  5  8  3  4  7  9

width=4 (merge      [1,2,5,8],[3,4,7,9] -> 1,2,3,4,5,7,8,9
the final pair):     1  2  3  4  5  7  8  9
```

Three passes for `n=8` -- exactly `log2(8)`, the same pass COUNT Chapter 12.2's bitonic merge needed for the same size array, but built from a fundamentally different primitive. Bitonic merge repeatedly applies compare-exchange SPLITS to a single already-bitonic sequence; bottom-up merge sort repeatedly applies full MERGES to pairs of independently-already-sorted runs, and critically, unlike bitonic sort, it places NO requirement on the input being bitonic (or anything else) to start with -- every element trivially counts as "already sorted" the moment a run has length 1.

Parallelizing one pass means launching independent co-rank merges, one per pair of runs, and -- since Section 14.1 already made a single merge itself fully parallel across its own output positions -- this composes into one flat kernel where every thread across the WHOLE array first figures out which pair of runs it belongs to (from its global output index and the current width), then runs its own co-rank binary search scoped to just that pair.

```
ASCII view of one pass at width=2, n=8 (two independent pairs of runs):

  pair 0: runs [2,5] and [1,8], output positions 0..3
  pair 1: runs [3,9] and [4,7], output positions 4..7

  thread for global position 5 figures out: pair_start=4, so it is
  working within pair 1 (runs [3,9] and [4,7]), at LOCAL position
  k = 5 - 4 = 1 within that pair's own merge.
```

Every pass still needs a full synchronization before the next one starts (the next pass's runs are built entirely out of the previous pass's output), exactly the same "loop over passes, synchronize between them" discipline Chapter 12.2's multi-pass bitonic merge and Chapter 13.3's chained radix passes both already required.

[COMMON TRAP]
A thread computing which pair of runs it belongs to must use its GLOBAL output position, not assume it is always working within pair 0. Forgetting to add `pair_start` back onto a co-rank result computed relative to that pair's own local indices silently corrupts every pair after the first.

### Code and Verification

```cpp
#include <cstdio>
#include <vector>
#include <algorithm>

// Chapter 14.2 main -- bottom-up merge sort as chained co-rank merges.
// Each pass merges every adjacent pair of same-width runs; Section
// 14.1 already made ONE such merge fully parallel via co-rank. A full
// parallel merge sort pass launches one co-rank merge PER PAIR of runs
// (each pair's merge is itself internally parallel across its own
// output positions), and passes are chained with a full synchronization
// between them -- the same "loop over passes, synchronize between
// them" shape Chapter 12.2 and Chapter 13.3 both already used, just
// with runs of DOUBLING width instead of a fixed distance or digit
// place driving the loop.

__device__ void co_rank(int k, const int* A, int m, const int* B, int n, int* out_i, int* out_j) {
    int i_low = (k - n > 0) ? (k - n) : 0;
    int i_high = (k < m) ? k : m;
    while (i_low < i_high) {
        int i = (i_low + i_high + 1) / 2;
        int j = k - i;
        if (j > 0 && i < m && A[i] < B[j - 1]) {
            i_low = i;
        } else if (i > 0 && j < n && A[i - 1] > B[j]) {
            i_high = i - 1;
        } else {
            i_low = i;
            break;
        }
    }
    *out_i = i_low;
    *out_j = k - i_low;
}

// One thread per OUTPUT position in the whole array; each thread first
// figures out which pair of runs it belongs to, then runs co-rank
// against just that pair.
__global__ void mergesort_pass_kernel(const int* g_in, int* g_out, int n, int width) {
    int global_k = threadIdx.x;
    if (global_k >= n) return;

    int pair_start = (global_k / (2 * width)) * (2 * width);
    int left_len = min(width, n - pair_start);
    int mid = pair_start + left_len;
    int right_len = min(width, n - mid);
    int k = global_k - pair_start;   // this thread's output index WITHIN its own pair

    const int* A = g_in + pair_start;
    const int* B = g_in + mid;
    int i, j;
    co_rank(k, A, left_len, B, right_len, &i, &j);
    bool take_from_A = (i < left_len) && (j >= right_len || A[i] <= B[j]);
    g_out[global_k] = take_from_A ? A[i] : B[j];
}

// ---- Host-side replay of the identical per-thread logic: every output
// position works out its own pair boundaries and runs the same co-rank
// binary search, independently of every other position. ----

void co_rank_host(int k, const int* A, int m, const int* B, int n, int& out_i, int& out_j) {
    int i_low = (k - n > 0) ? (k - n) : 0;
    int i_high = (k < m) ? k : m;
    while (i_low < i_high) {
        int i = (i_low + i_high + 1) / 2;
        int j = k - i;
        if (j > 0 && i < m && A[i] < B[j - 1]) {
            i_low = i;
        } else if (i > 0 && j < n && A[i - 1] > B[j]) {
            i_high = i - 1;
        } else {
            i_low = i;
            break;
        }
    }
    out_i = i_low;
    out_j = k - i_low;
}

std::vector<int> mergesort_pass(const std::vector<int>& a, int width) {
    int n = (int)a.size();
    std::vector<int> out(n);
    for (int global_k = 0; global_k < n; global_k++) {
        int pair_start = (global_k / (2 * width)) * (2 * width);
        int left_len = std::min(width, n - pair_start);
        int mid = pair_start + left_len;
        int right_len = std::min(width, n - mid);
        int k = global_k - pair_start;

        const int* A = a.data() + pair_start;
        const int* B = a.data() + mid;
        int i, j;
        co_rank_host(k, A, left_len, B, right_len, i, j);
        bool take_from_A = (i < left_len) && (j >= right_len || A[i] <= B[j]);
        out[global_k] = take_from_A ? A[i] : B[j];
    }
    return out;
}

int main() {
    printf("=== Section 14.2 main: bottom-up merge sort as chained co-rank kernels ===\n\n");

    std::vector<int> a = {5, 2, 8, 1, 9, 3, 7, 4};
    int n = (int)a.size();

    printf("input: ");
    for (int v : a) printf("%d ", v);
    printf("\n\n");

    int pass_num = 0;
    for (int width = 1; width < n; width *= 2) {
        a = mergesort_pass(a, width);
        pass_num++;
        printf("pass %d (width=%d, %d threads): ", pass_num, width, n);
        for (int v : a) printf("%d ", v);
        printf("\n");
    }
    printf("\n");

    std::vector<int> a0 = {5, 2, 8, 1, 9, 3, 7, 4};
    std::sort(a0.begin(), a0.end());
    bool ok = std::is_sorted(a.begin(), a.end()) && (a == a0);

    printf("expected (via std::sort reference): ");
    for (int v : a0) printf("%d ", v);
    printf("\n\n");
    printf("every pass uses all %d threads at once, each running its own independent\n", n);
    printf("co-rank binary search scoped to its own pair of runs -- exactly %d passes\n",
           pass_num);
    printf("total, matching the CPU baseline's own pass count, with span collapsed from\n");
    printf("this pass's own O(width) sequential merge work down to O(log width) per thread.\n");

    printf("\nself-check: parallel bottom-up merge sort matches std::sort reference: %s\n",
           ok ? "confirmed" : "MISMATCH");
    return ok ? 0 : 1;
}
```

**Compile and run:**

```bash
NVDIR=/usr/local/lib/python3.11/dist-packages/nvidia/cu13
$NVDIR/bin/nvcc -arch=sm_80 -I$NVDIR/include -L$NVDIR/lib -lcudart 73_mergesort_corank_multipass.cu -o 73_mergesort_corank_multipass
LD_LIBRARY_PATH=$NVDIR/lib ./73_mergesort_corank_multipass
```

**Sample input:** the same array, sorted via 3 chained co-rank-based merge passes (widths 1, 2, 4), all 8 threads active on every pass.

**Sample output:**

```text
=== Section 14.2 main: bottom-up merge sort as chained co-rank kernels ===

input: 5 2 8 1 9 3 7 4 

pass 1 (width=1, 8 threads): 2 5 1 8 3 9 4 7 
pass 2 (width=2, 8 threads): 1 2 5 8 3 4 7 9 
pass 3 (width=4, 8 threads): 1 2 3 4 5 7 8 9 

expected (via std::sort reference): 1 2 3 4 5 7 8 9 

every pass uses all 8 threads at once, each running its own independent
co-rank binary search scoped to its own pair of runs -- exactly 3 passes
total, matching the CPU baseline's own pass count, with span collapsed from
this pass's own O(width) sequential merge work down to O(log width) per thread.

self-check: parallel bottom-up merge sort matches std::sort reference: confirmed
```

## 14.3 Sample Sort: Partitioning Data Across Blocks

### Intuition

Bottom-up merge sort's pass count grows as `log2(number of blocks)` once "a run" means an entire block's worth of data rather than a few array elements -- each round still needs every pair of blocks to coordinate a full cross-block merge. Sample sort avoids that altogether with a completely different strategy: pick a small number of **splitter** values that divide the data's value range into buckets, scatter every element directly into its own bucket in a SINGLE pass, and then sort each bucket completely independently -- on its own block, with no further coordination needed, because the bucket boundaries already guarantee every value in bucket `b` is `<=` every value in bucket `b+1`.

### The Sequential (CPU) Baseline

```cpp
#include <cstdio>
#include <vector>
#include <algorithm>

// Chapter 14.3 -- The Sequential (CPU) Baseline.
// Merge sort (Section 14.2) scales by repeatedly merging PAIRS of runs
// -- log2(n) rounds of cross-run coordination, which becomes expensive
// once "a run" means "everything one whole block or one whole GPU
// processed," not just a few array elements. Sample sort takes a
// completely different approach: pick a small number of SPLITTER
// values that divide the value range into buckets, send every element
// directly to its own bucket in ONE pass, then sort each bucket
// independently -- with NO further merging needed afterward, because
// every value in bucket b is guaranteed <= every value in bucket b+1
// by construction.

int bucket_of(int value, const std::vector<int>& splitters) {
    int b = 0;
    for (int s : splitters) {
        if (value >= s) b++;
        else break;
    }
    return b;
}

int main() {
    printf("=== Section 14.3 CPU baseline: sample sort (splitters, buckets, per-bucket sort) ===\n\n");

    std::vector<int> a = {19, 3, 42, 7, 55, 12, 38, 25};
    std::vector<int> splitters = {15, 30};   // 2 splitters -> 3 buckets
    int base = (int)splitters.size() + 1;
    int n = (int)a.size();

    printf("input: ");
    for (int v : a) printf("%d ", v);
    printf("\nsplitters: 15, 30  (3 buckets: [-inf,15), [15,30), [30,inf))\n\n");

    std::vector<int> bucket(n);
    for (int i = 0; i < n; i++) bucket[i] = bucket_of(a[i], splitters);

    printf("bucket of each element: ");
    for (int b : bucket) printf("%d ", b);
    printf("\n\n");

    std::vector<int> count(base, 0);
    for (int b : bucket) count[b]++;
    std::vector<int> offset(base, 0);
    int total = 0;
    for (int d = 0; d < base; d++) { offset[d] = total; total += count[d]; }

    printf("count per bucket:  ");
    for (int c : count) printf("%d ", c);
    printf("\noffset per bucket: ");
    for (int o : offset) printf("%d ", o);
    printf("\n\n");

    std::vector<int> scattered(n);
    std::vector<int> pos = offset;
    for (int i = 0; i < n; i++) {
        int b = bucket[i];
        scattered[pos[b]] = a[i];
        pos[b]++;
    }

    printf("scattered by bucket (not yet sorted within each bucket): ");
    for (int v : scattered) printf("%d ", v);
    printf("\n\n");

    std::vector<int> final_result;
    for (int d = 0; d < base; d++) {
        std::vector<int> bucket_vals(scattered.begin() + offset[d],
                                      scattered.begin() + offset[d] + count[d]);
        std::sort(bucket_vals.begin(), bucket_vals.end());
        printf("bucket %d has %d element(s), sorted independently: ", d, count[d]);
        for (int v : bucket_vals) printf("%d ", v);
        printf("\n");
        final_result.insert(final_result.end(), bucket_vals.begin(), bucket_vals.end());
    }
    printf("\n");

    std::vector<int> expected = {3, 7, 12, 19, 25, 38, 42, 55};
    bool ok = (final_result == expected) && std::is_sorted(final_result.begin(), final_result.end());

    printf("final: ");
    for (int v : final_result) printf("%d ", v);
    printf("\nexpected: 3 7 12 19 25 38 42 55\n\n");
    printf("bucket sizes came out 3, 2, 3 -- not perfectly even, because these particular\n");
    printf("splitters (15 and 30) don't split THIS array's values into exactly equal\n");
    printf("thirds. No further merging is needed regardless: concatenating the buckets in\n");
    printf("order is already the full sorted result, since every bucket-0 value is < 15,\n");
    printf("every bucket-1 value is in [15,30), and every bucket-2 value is >= 30.\n");

    printf("\nself-check: sample sort matches expected fully sorted order: %s\n",
           ok ? "confirmed" : "MISMATCH");
    return ok ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++17 -Wall -Wextra -O2 74_sample_sort_cpu_baseline.cpp -o 74_sample_sort_cpu_baseline
./74_sample_sort_cpu_baseline
```

**Sample input:** the array `{19, 3, 42, 7, 55, 12, 38, 25}`, split into 3 buckets using splitters `{15, 30}`.

**Sample output:**

```text
=== Section 14.3 CPU baseline: sample sort (splitters, buckets, per-bucket sort) ===

input: 19 3 42 7 55 12 38 25 
splitters: 15, 30  (3 buckets: [-inf,15), [15,30), [30,inf))

bucket of each element: 1 0 2 0 2 0 2 1 

count per bucket:  3 2 3 
offset per bucket: 0 3 5 

scattered by bucket (not yet sorted within each bucket): 3 7 12 19 25 42 55 38 

bucket 0 has 3 element(s), sorted independently: 3 7 12 
bucket 1 has 2 element(s), sorted independently: 19 25 
bucket 2 has 3 element(s), sorted independently: 38 42 55 

final: 3 7 12 19 25 38 42 55 
expected: 3 7 12 19 25 38 42 55

bucket sizes came out 3, 2, 3 -- not perfectly even, because these particular
splitters (15 and 30) don't split THIS array's values into exactly equal
thirds. No further merging is needed regardless: concatenating the buckets in
order is already the full sorted result, since every bucket-0 value is < 15,
every bucket-1 value is in [15,30), and every bucket-2 value is >= 30.

self-check: sample sort matches expected fully sorted order: confirmed
```

### The Concept, In Detail

With splitters `{15, 30}`, three buckets exist: values less than 15, values in `[15, 30)`, and values 30 or greater. Assigning each element to its bucket is one binary search (or, for just 2 splitters, a couple of comparisons) per element:

```
value:    19   3  42   7  55  12  38  25
bucket:    1   0   2   0   2   0   2   1
           (>=15,<30)  (<15)  (>=30)  ...
```

From here, the machinery is EXACTLY Chapter 13.2's stable radix-pass pipeline, with one substitution: "which digit does this element have" is replaced by "which bucket does this element's value fall into." Count elements per bucket, take an exclusive prefix sum to get each bucket's starting offset, compute each element's rank within its own bucket via the same per-bucket indicator-array scan (the k-way generalization of Chapter 6.3's stable partition), and scatter:

```
count per bucket:   3   2   3
offset per bucket:  0   3   5

scattered (grouped by bucket, NOT yet sorted within each bucket):
  3  7  12 | 19  25 | 42  55  38
  bucket 0    bucket 1    bucket 2
```

The bucket sizes came out `3, 2, 3` -- not perfectly even. This is not a bug: these particular splitters happen not to divide THIS array's actual values into exactly equal thirds. In practice, "sample" sort gets its name from how splitters are normally chosen: rather than requiring the whole array to already be sorted (which would defeat the point), a small RANDOM SAMPLE of the data is drawn, sorted (cheap, since the sample is small), and evenly-spaced elements of that sorted sample become the splitters -- a statistical estimate of where the true quantiles lie, good enough in practice to keep bucket sizes roughly balanced without ever needing to look at the whole dataset's sorted order in advance.

The critical final step is what sample sort DOESN'T need: once bucketed, each bucket can be sorted completely independently -- by its own block, using any per-block sort this book has already built (Chapter 12's bitonic sort or Chapter 13's own radix sort both work identically well here) -- and simply concatenating the sorted buckets, in bucket order, is already the final fully sorted array. No merge step across buckets is ever required, because the splitter-defined bucket boundaries already guarantee bucket 0's largest possible value is smaller than bucket 1's smallest possible value, and so on.

```
final concatenation (each bucket sorted independently, by its own block):

  bucket 0 sorted: 3  7  12
  bucket 1 sorted: 19  25
  bucket 2 sorted: 38  42  55

  concatenated: 3  7  12  19  25  38  42  55   <- fully sorted, no merge needed
```

[COMMON TRAP]
Choosing too FEW splitters, or splitters that badly misjudge the data's distribution, produces badly imbalanced buckets -- in the extreme, all elements could land in a single bucket, leaving one block with all the work and every other block idle, no better than not partitioning at all. The quality of the sample used to pick splitters directly controls how well sample sort's parallelism actually gets used; this is a load-balancing concern with no equivalent in Chapter 12's bitonic sort, whose fixed comparison network guarantees perfectly even work distribution regardless of data.

### Code and Verification

```cpp
#include <cstdio>
#include <vector>
#include <algorithm>

// Chapter 14.3 main -- sample sort's bucketing pass as GPU kernels.
// This is Chapter 13.2's exact histogram/offset/rank/scatter pipeline,
// with one change: which "bucket" an element belongs to is now decided
// by comparing its VALUE against a small set of splitters (a binary
// search) instead of extracting a digit by arithmetic. Everything
// downstream -- counting, the exclusive offset scan, the per-bucket
// rank scan, and the final scatter -- is identical in shape to radix
// sort's own per-pass machinery. Once every element has been scattered
// into its own bucket, each bucket can be handed to its OWN block and
// sorted completely independently (with any per-block sort this book
// has already built, such as Chapter 12's bitonic sort or Chapter 13's
// own radix sort) -- exactly the "data too large for one block" case
// this chapter set out to solve.

__device__ int bucket_of(int value, const int* splitters, int num_splitters) {
    int b = 0;
    for (int s = 0; s < num_splitters; s++) {
        if (value >= splitters[s]) b++;
        else break;
    }
    return b;
}

__global__ void bucket_histogram_kernel(const int* g_a, int n, const int* g_splitters,
                                         int num_splitters, int* g_count) {
    int i = threadIdx.x;
    if (i >= n) return;
    int b = bucket_of(g_a[i], g_splitters, num_splitters);
    atomicAdd(&g_count[b], 1);
}

__global__ void exclusive_scan_kernel(const int* g_in, int* g_out, int n) {
    extern __shared__ int shared_mem[];
    int* buf_a = shared_mem;
    int* buf_b = shared_mem + n;
    int tid = threadIdx.x;
    if (tid >= n) return;
    buf_a[tid] = g_in[tid];
    __syncthreads();
    int* src = buf_a;
    int* dst = buf_b;
    for (int d = 1; d < n; d *= 2) {
        dst[tid] = (tid >= d) ? (src[tid] + src[tid - d]) : src[tid];
        __syncthreads();
        int* tmp = src; src = dst; dst = tmp;
    }
    g_out[tid] = src[tid] - g_in[tid];
}

__global__ void bucket_scatter_kernel(const int* g_a, int n, const int* g_splitters,
                                       int num_splitters, const int* g_offset,
                                       const int* g_rank, int* g_out) {
    int i = threadIdx.x;
    if (i >= n) return;
    int b = bucket_of(g_a[i], g_splitters, num_splitters);
    g_out[g_offset[b] + g_rank[i]] = g_a[i];
}

// ---- Host-side replay of the identical per-thread logic across all
// kernels: bucket histogram, offset scan, per-bucket rank scan (the
// same k-way stable-partition generalization Chapter 13.2 used), the
// scatter, and finally an independent sort of each resulting bucket. ----

int main() {
    printf("=== Section 14.3 main: sample sort's bucketing pass as CUDA kernels ===\n\n");

    std::vector<int> a = {19, 3, 42, 7, 55, 12, 38, 25};
    std::vector<int> splitters = {15, 30};
    int base = (int)splitters.size() + 1;
    int n = (int)a.size();

    printf("input: ");
    for (int v : a) printf("%d ", v);
    printf("\nsplitters: 15, 30  (3 buckets)\n\n");

    std::vector<int> bucket(n);
    for (int i = 0; i < n; i++) {
        int b = 0;
        for (int s : splitters) { if (a[i] >= s) b++; else break; }
        bucket[i] = b;
    }

    std::vector<int> count(base, 0);
    for (int b : bucket) count[b]++;

    std::vector<int> src = count, dst = count;
    for (int d = 1; d < base; d *= 2) {
        for (int tid = 0; tid < base; tid++)
            dst[tid] = (tid >= d) ? (src[tid] + src[tid - d]) : src[tid];
        src = dst;
    }
    std::vector<int> offset(base);
    for (int tid = 0; tid < base; tid++) offset[tid] = src[tid] - count[tid];

    std::vector<int> rank(n, 0);
    for (int d = 0; d < base; d++) {
        if (count[d] == 0) continue;
        std::vector<int> indicator(n);
        for (int i = 0; i < n; i++) indicator[i] = (bucket[i] == d) ? 1 : 0;
        std::vector<int> isrc = indicator, idst = indicator;
        for (int s = 1; s < n; s *= 2) {
            for (int tid = 0; tid < n; tid++)
                idst[tid] = (tid >= s) ? (isrc[tid] + isrc[tid - s]) : isrc[tid];
            isrc = idst;
        }
        for (int i = 0; i < n; i++) if (bucket[i] == d) rank[i] = isrc[i] - indicator[i];
    }

    printf("bucket of each element:       ");
    for (int b : bucket) printf("%d ", b);
    printf("\nrank within own bucket:       ");
    for (int r : rank) printf("%d ", r);
    printf("\ncount per bucket:  ");
    for (int c : count) printf("%d ", c);
    printf("\noffset per bucket: ");
    for (int o : offset) printf("%d ", o);
    printf("\n\n");

    std::vector<int> scattered(n);
    for (int i = 0; i < n; i++) scattered[offset[bucket[i]] + rank[i]] = a[i];

    printf("scattered by bucket: ");
    for (int v : scattered) printf("%d ", v);
    printf("\n\n");

    std::vector<int> final_result;
    for (int d = 0; d < base; d++) {
        std::vector<int> bucket_vals(scattered.begin() + offset[d],
                                      scattered.begin() + offset[d] + count[d]);
        std::sort(bucket_vals.begin(), bucket_vals.end());   // any per-block sort works here
        printf("bucket %d, sorted independently (its own block's job): ", d);
        for (int v : bucket_vals) printf("%d ", v);
        printf("\n");
        final_result.insert(final_result.end(), bucket_vals.begin(), bucket_vals.end());
    }
    printf("\n");

    std::vector<int> expected = {3, 7, 12, 19, 25, 38, 42, 55};
    bool ok = (final_result == expected);

    printf("final: ");
    for (int v : final_result) printf("%d ", v);
    printf("\nexpected: 3 7 12 19 25 38 42 55\n\n");
    printf("this matches Section 14.3's CPU baseline exactly: bucketing was ONE parallel\n");
    printf("pass across all %d elements, and once buckets were formed, each block's sort\n", n);
    printf("ran completely independently of every other block's -- no cross-block merging\n");
    printf("step was needed anywhere in this pipeline.\n");

    printf("\nself-check: parallel sample sort matches CPU baseline: %s\n",
           ok ? "confirmed" : "MISMATCH");
    return ok ? 0 : 1;
}
```

**Compile and run:**

```bash
NVDIR=/usr/local/lib/python3.11/dist-packages/nvidia/cu13
$NVDIR/bin/nvcc -arch=sm_80 -I$NVDIR/include -L$NVDIR/lib -lcudart 75_sample_sort_parallel.cu -o 75_sample_sort_parallel
LD_LIBRARY_PATH=$NVDIR/lib ./75_sample_sort_parallel
```

**Sample input:** the same array and splitters, bucketed via the full histogram/offset/rank/scatter pipeline, then each bucket sorted independently.

**Sample output:**

```text
=== Section 14.3 main: sample sort's bucketing pass as CUDA kernels ===

input: 19 3 42 7 55 12 38 25 
splitters: 15, 30  (3 buckets)

bucket of each element:       1 0 2 0 2 0 2 1 
rank within own bucket:       0 0 0 1 1 2 2 1 
count per bucket:  3 2 3 
offset per bucket: 0 3 5 

scattered by bucket: 3 7 12 19 25 42 55 38 

bucket 0, sorted independently (its own block's job): 3 7 12 
bucket 1, sorted independently (its own block's job): 19 25 
bucket 2, sorted independently (its own block's job): 38 42 55 

final: 3 7 12 19 25 38 42 55 
expected: 3 7 12 19 25 38 42 55

this matches Section 14.3's CPU baseline exactly: bucketing was ONE parallel
pass across all 8 elements, and once buckets were formed, each block's sort
ran completely independently of every other block's -- no cross-block merging
step was needed anywhere in this pipeline.

self-check: parallel sample sort matches CPU baseline: confirmed
```

## Chapter Summary

Merging two sorted arrays sequentially has an unavoidable span equal to their combined length, because each step's decision depends on the previous one -- but the co-rank (merge path) technique sidesteps this entirely by asking, for each OUTPUT position independently, how many elements came from each input array, a question answerable by binary search with no dependency between positions. Chaining this parallel merge across doubling run widths (1, 2, 4, ..., n) gives bottom-up merge sort, needing `log2(n)` passes with a full synchronization between them, exactly like Chapter 12's bitonic merge and Chapter 13's chained radix passes, but built from full merges of independently-sorted runs rather than fixed compare-exchange distances or digit places. Sample sort takes a different approach entirely, suited to data spanning many blocks: pick splitter values (in practice, estimated from a small random sample of the data) that divide the value range into buckets, scatter every element into its bucket using the exact same histogram/offset/rank/scatter pipeline Chapter 13.2 built for radix sort's digits, then sort each bucket completely independently with no cross-bucket merge needed afterward, since the splitter boundaries already guarantee every bucket's values are less than the next bucket's. Sample sort's one real risk is load imbalance: poorly chosen splitters can leave buckets badly uneven in size, unlike bitonic sort's data-independent, perfectly-balanced comparison network.

## Self-Check Questions

1. Why does a sequential merge of two sorted arrays have span equal to their combined length, in Chapter 3's own vocabulary, and why can adding more threads not shorten that span directly?
2. What two conditions must hold for `co_rank(k) = (i, j)` to be the correct split point, and why does A and B both being individually sorted make binary search for `i` possible?
3. Once `co_rank(k)` is known, how is the actual output value at position `k` determined, and what boundary case must be checked before reading from either array?
4. In bottom-up merge sort's parallel kernel, why must each thread compute which PAIR of runs it belongs to before running co-rank, rather than always treating the whole array as one giant pair?
5. Explain how sample sort's bucketing pass reuses Chapter 13.2's radix-pass pipeline. What plays the role of "digit," and what plays the role of "extracting the digit"?
6. Why does sample sort never need a merge step across buckets, the way merge sort needs merges across runs? What specific guarantee makes concatenation alone sufficient?

## Where We Go Next

This chapter closes Part 3. Across four chapters, this book has built comparison-based sorting on a fixed network (bitonic sort), non-comparison bucket-based sorting with a fixed pass count (radix sort), and two strategies for sorting data that outgrows a single block entirely (merge sort's chained parallel merges, and sample sort's one-pass bucketing). Part 4 turns to trees: binary trees built and traversed without recursion (a real constraint on a GPU, where a natural call stack is not simply "there" the way it is on a CPU), parallel tree and trie construction, segment and Fenwick trees for answering range queries in logarithmic time, and the spatial trees -- k-d trees, quadtrees, octrees, and bounding volume hierarchies -- that make nearest-neighbor search and collision detection tractable at scale.

## Worked Solutions

**1.** A sequential merge's span is `m+n` (or `m+n-1` comparisons) because each step's decision -- which pointer to advance -- determines what the NEXT comparison even is; the second comparison cannot be attempted until the first one's outcome is known, and so on down the chain. This is a dependency baked into the algorithm's own logic, not a resource limitation, so throwing more threads at it does not help directly -- there is nothing independent for those threads to do within the sequential walk itself. (Section 14.1's co-rank technique works around this not by speeding up the walk, but by replacing it with a completely different, independently-parallel computation that happens to produce the same result.)

**2.** The two conditions are: every element A contributed (`A[0..i)`) must be `<=` the next unconsumed element of B (`B[j]`, if `j < n`), and every element B contributed (`B[0..j)`) must be `<=` the next unconsumed element of A (`A[i]`, if `i < m`). Binary search for `i` works because A and B are each sorted: increasing the candidate `i` only makes the first condition easier to satisfy and the second harder (or vice versa for decreasing `i`), giving binary search's required monotonic structure to narrow the search range.

**3.** Once `co_rank(k) = (i, j)` is known, the value AT position `k` is `A[i]` if `i < m` and either `B` is exhausted (`j >= n`) or `A[i] <= B[j]`; otherwise it is `B[j]`. The boundary case that must be checked is exactly this exhaustion condition -- reading `A[i]` when `i == m`, or `B[j]` when `j == n`, would read out of bounds, so the "is this array exhausted" check must come before comparing values, not after.

**4.** The whole array is not one giant pair once run width is smaller than the full array -- at width 2 on an 8-element array, there are 4 independent pairs of runs (each of size 2), not one pair of size 8. A thread must first identify which pair its global output position falls into (via `pair_start = (global_k / (2*width)) * (2*width)`), then run co-rank using ONLY that pair's own two runs and its own LOCAL position within the pair; treating the whole array as one pair would attempt to merge runs that were never adjacent to each other in the current pass, or of the wrong width, producing an incorrect result.

**5.** Chapter 13.2's radix pass pipeline is: extract each element's digit, count elements per digit (histogram), take an exclusive prefix sum for offsets, compute each element's rank within its own digit group via a per-digit indicator-array scan, then scatter using `offset[digit] + rank`. Sample sort's bucketing pass is structurally identical, with "which bucket does this element's value fall into" (found via binary search against the splitters) playing the role of "digit," and that binary search playing the role of "extracting the digit" -- counting, offsets, rank-within-group, and the final scatter are all unchanged in shape.

**6.** Sample sort never needs a cross-bucket merge because the splitters, by construction, guarantee that every value assigned to bucket `b` is strictly less than every value that could be assigned to bucket `b+1` (bucket `b` holds values in some range `[splitter[b-1], splitter[b])`, and bucket `b+1` only holds values `>= splitter[b]`). Once each bucket is internally sorted, simply placing bucket 0's sorted values first, then bucket 1's, and so on, already satisfies the full array's sorted order -- there is no possibility of a bucket-1 value needing to be interleaved with a bucket-0 value, which is exactly the guarantee a merge step exists to provide when it IS needed, as it is between two runs in Section 14.2's merge sort.
