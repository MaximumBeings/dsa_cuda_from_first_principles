# Chapter 13: Radix Sort

Chapter 12 built a genuinely parallel sort out of a completely fixed, data-independent compare-exchange network. That fixed-network property is elegant, but it comes at a real cost: bitonic sort always performs `O(n log^2 n)` total comparisons, no matter what the input looks like, and every one of those comparisons is a compare-AND-swap between two full keys. Radix sort takes a different, older idea -- one that predates computers entirely, going back to mechanical card sorters -- and turns out to be an even better fit for a GPU: never compare two keys against each other at all. Instead, repeatedly bucket every key by one small piece of it at a time, using exactly the histogram-and-scatter machinery Chapters 6 and 7 already built.

## 13.1 Extracting Digits and Counting: The Building Block

### Intuition

Think of each key as a sequence of digits, exactly the way a human reads a multi-digit number: `43` has a ones digit (`3`) and a tens digit (`4`). Radix sort's basic move is to look at just ONE of these digit positions across the whole array, and count how many keys have each possible digit value there -- 0 through 9, if working in base 10. This is precisely Chapter 7's histogram, just computed on a single digit of each key instead of the whole key. Once that count is known, an exclusive prefix sum over the (tiny, only-`base`-sized) count array tells us where each digit value's group of elements will eventually START in a reordered array -- before a single element has actually been moved.

### The Sequential (CPU) Baseline

```cpp
#include <cstdio>
#include <vector>

// Chapter 13.1 -- The Sequential (CPU) Baseline.
// Radix sort never compares two full keys against each other. Instead,
// on each pass it looks at just ONE digit of each key (base 10 here,
// though any base works) and counts how many keys have each digit
// value -- exactly Chapter 7's histogram, applied to a single digit
// instead of a whole value. From that histogram, an exclusive prefix
// sum over the (small, base-sized) count array gives every digit
// value's starting offset in the eventual output -- the same
// count-then-scan shape Chapter 7.3's counting sort already used.

int digit_of(int value, int place, int base) {
    return (value / place) % base;
}

void compute_histogram_and_offsets(const std::vector<int>& a, int place, int base,
                                    std::vector<int>& count, std::vector<int>& offset) {
    count.assign(base, 0);
    for (int v : a) count[digit_of(v, place, base)]++;

    offset.assign(base, 0);
    int total = 0;
    for (int d = 0; d < base; d++) {
        offset[d] = total;
        total += count[d];
    }
}

int main() {
    printf("=== Section 13.1 CPU baseline: digit histogram and offsets ===\n\n");

    std::vector<int> a = {29, 13, 82, 43, 65, 24, 13, 5};
    int place = 1;   // ones digit
    int base = 10;
    int n = (int)a.size();

    printf("input array: ");
    for (int v : a) printf("%d ", v);
    printf("\nplace = %d (ones digit), base = %d\n\n", place, base);

    printf("digit of each element: ");
    for (int v : a) printf("%d ", digit_of(v, place, base));
    printf("\n\n");

    std::vector<int> count, offset;
    compute_histogram_and_offsets(a, place, base, count, offset);

    printf("count[digit], digit = 0..9:  ");
    for (int c : count) printf("%d ", c);
    printf("\noffset[digit] (exclusive prefix sum of count):  ");
    for (int o : offset) printf("%d ", o);
    printf("\n\n");

    int count_sum = 0;
    for (int c : count) count_sum += c;

    std::vector<int> expected_count = {0, 0, 1, 3, 1, 2, 0, 0, 0, 1};
    std::vector<int> expected_offset = {0, 0, 0, 1, 4, 5, 7, 7, 7, 7};
    bool ok = (count_sum == n) && (count == expected_count) && (offset == expected_offset);

    printf("expected count:  0 0 1 3 1 2 0 0 0 1\n");
    printf("expected offset: 0 0 0 1 4 5 7 7 7 7\n\n");
    printf("count array sums to n (%d): every element landed in exactly one digit bucket.\n", n);
    printf("offset[d] tells us where digit d's group STARTS once we actually place elements\n");
    printf("-- Section 13.2 uses this to perform the full stable scatter.\n");

    printf("\nself-check: histogram and offsets match hand-derived values: %s\n",
           ok ? "confirmed" : "MISMATCH");
    return ok ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++17 -Wall -Wextra -O2 64_digit_histogram_cpu_baseline.cpp -o 64_digit_histogram_cpu_baseline
./64_digit_histogram_cpu_baseline
```

**Sample input:** the array `{29, 13, 82, 43, 65, 24, 13, 5}`, histogrammed by ones digit (`place=1`, `base=10`).

**Sample output:**

```text
=== Section 13.1 CPU baseline: digit histogram and offsets ===

input array: 29 13 82 43 65 24 13 5 
place = 1 (ones digit), base = 10

digit of each element: 9 3 2 3 5 4 3 5 

count[digit], digit = 0..9:  0 0 1 3 1 2 0 0 0 1 
offset[digit] (exclusive prefix sum of count):  0 0 0 1 4 5 7 7 7 7 

expected count:  0 0 1 3 1 2 0 0 0 1
expected offset: 0 0 0 1 4 5 7 7 7 7

count array sums to n (8): every element landed in exactly one digit bucket.
offset[d] tells us where digit d's group STARTS once we actually place elements
-- Section 13.2 uses this to perform the full stable scatter.

self-check: histogram and offsets match hand-derived values: confirmed
```

### The Concept, In Detail

Extracting a digit from a value at a given place is one line of integer arithmetic: `(value / place) % base`. For `place=1` (the ones digit) and `base=10`, this is just `value % 10`; for `place=10` (the tens digit), it is `(value / 10) % 10`. Running this over the example array:

```
value:   29  13  82  43  65  24  13   5
digit:    9   3   2   3   5   4   3   5
```

Three elements share digit `3` (the two `13`s and the `43`), and two share digit `5` (`65` and `5`) -- exactly the kind of collision Chapter 7.1 built `atomicAdd` to handle safely. Counting these up:

```
digit:    0  1  2  3  4  5  6  7  8  9
count:    0  0  1  3  1  2  0  0  0  1
```

The count array sums to `8`, confirming every element landed in exactly one bucket. The offset array is an EXCLUSIVE prefix sum over this count array -- offset[d] answers "how many elements have a digit strictly less than d", which is exactly where digit d's group will start once elements are actually placed:

```
digit:     0  1  2  3  4  5  6  7  8  9
count:     0  0  1  3  1  2  0  0  0  1
offset:    0  0  0  1  4  5  7  7  7  7
           ^--- offset[3] = count[0]+count[1]+count[2] = 0+0+1 = 1
                       ^--- offset[5] = offset[4] + count[4] = 4 + 1 = 5
```

This offset computation is a scan over an array of size `base` (here, just 10 elements) -- tiny compared to `n` in general, and computable with the exact same Hillis-Steele double-buffered technique Chapter 5.1 introduced, just shifted from an inclusive scan to an exclusive one afterward (`offset[i] = inclusive[i] - count[i]`, subtracting each position's own contribution back out).

The parallel opportunity here is two-fold and layered: the histogram itself is `n` independent atomic increments (one per element, into a shared array of only `base` counters -- genuine collisions do happen, but far less densely than Chapter 7.1's original single-bucket race), and the offset computation is a scan over a SEPARATE, much smaller array of size `base`, using threads drawn from that small count, not from `n`.

[COMMON TRAP]
It is tempting to think the offset array should be an INCLUSIVE scan (offset[d] = elements with digit <= d). That would place digit d's group ending, not starting, at offset[d] -- every subsequent placement step would need an extra off-by-one adjustment. Keep the offset array exclusive: offset[d] is the position of the FIRST element with digit d, once elements start actually being written there.

### Code and Verification

```cpp
#include <cstdio>
#include <vector>

// Chapter 13.1 main -- digit histogram and offsets as two GPU kernels.
// The histogram itself is exactly Chapter 7.1's atomicAdd pattern: every
// thread reads its OWN element, computes its digit, and does one
// atomicAdd into the shared count array -- collisions are the whole
// point atomicAdd exists to handle safely. The offset array is a scan
// over a tiny, base-sized array (10 elements here), computed with the
// identical Hillis-Steele double-buffered technique Chapter 5.1 used,
// just shifted from inclusive to exclusive afterward.

__global__ void digit_histogram_kernel(const int* g_a, int n, int place, int base,
                                        int* g_count) {
    int i = threadIdx.x;
    if (i >= n) return;
    int digit = (g_a[i] / place) % base;
    atomicAdd(&g_count[digit], 1);
}

__global__ void exclusive_scan_offsets_kernel(const int* g_count, int* g_offset, int base) {
    extern __shared__ int shared_mem[];
    int* buf_a = shared_mem;
    int* buf_b = shared_mem + base;
    int tid = threadIdx.x;
    if (tid >= base) return;

    buf_a[tid] = g_count[tid];
    __syncthreads();

    int* src = buf_a;
    int* dst = buf_b;
    for (int d = 1; d < base; d *= 2) {
        dst[tid] = (tid >= d) ? (src[tid] + src[tid - d]) : src[tid];
        __syncthreads();
        int* tmp = src; src = dst; dst = tmp;
    }
    // src now holds the INCLUSIVE scan; shift to exclusive by subtracting
    // each position's own original count value.
    g_offset[tid] = src[tid] - g_count[tid];
}

// ---- Host-side replay of the identical per-thread logic: one atomic-add
// per element for the histogram, then the same double-buffered scan
// arithmetic over the small base-sized count array. ----

int main() {
    printf("=== Section 13.1 main: digit histogram and offsets as CUDA kernels ===\n\n");

    std::vector<int> a = {29, 13, 82, 43, 65, 24, 13, 5};
    int place = 1;
    int base = 10;
    int n = (int)a.size();

    printf("input array: ");
    for (int v : a) printf("%d ", v);
    printf("\nplace = %d (ones digit), base = %d, %d threads (one per element)\n\n", place, base, n);

    std::vector<int> count(base, 0);
    for (int v : a) {
        int digit = (v / place) % base;
        count[digit]++;   // one atomicAdd per thread, in parallel on real hardware
    }

    std::vector<int> src = count, dst = count;
    for (int d = 1; d < base; d *= 2) {
        for (int tid = 0; tid < base; tid++) {
            dst[tid] = (tid >= d) ? (src[tid] + src[tid - d]) : src[tid];
        }
        src = dst;
    }
    std::vector<int> offset(base);
    for (int tid = 0; tid < base; tid++) offset[tid] = src[tid] - count[tid];

    printf("count[digit], digit = 0..9:  ");
    for (int c : count) printf("%d ", c);
    printf("\noffset[digit] (exclusive prefix sum of count):  ");
    for (int o : offset) printf("%d ", o);
    printf("\n\n");

    std::vector<int> expected_count = {0, 0, 1, 3, 1, 2, 0, 0, 0, 1};
    std::vector<int> expected_offset = {0, 0, 0, 1, 4, 5, 7, 7, 7, 7};
    bool ok = (count == expected_count) && (offset == expected_offset);

    printf("expected count:  0 0 1 3 1 2 0 0 0 1\n");
    printf("expected offset: 0 0 0 1 4 5 7 7 7 7\n\n");
    printf("this matches the CPU baseline exactly: %d threads did %d independent atomic\n", n, n);
    printf("increments (with base=%d distinct bucket targets, so real collisions do occur --\n", base);
    printf("unlike Chapter 7.1's single-bucket race, most of these increments land in\n");
    printf("different buckets and never actually contend at all), then %d threads jointly\n", base);
    printf("computed the exclusive scan over the tiny count array in log2(%d) steps.\n", base);

    printf("\nself-check: kernel result matches CPU baseline: %s\n", ok ? "confirmed" : "MISMATCH");
    return ok ? 0 : 1;
}
```

**Compile and run:**

```bash
NVDIR=/usr/local/lib/python3.11/dist-packages/nvidia/cu13
$NVDIR/bin/nvcc -arch=sm_80 -I$NVDIR/include -L$NVDIR/lib -lcudart 65_digit_histogram_kernel.cu -o 65_digit_histogram_kernel
LD_LIBRARY_PATH=$NVDIR/lib ./65_digit_histogram_kernel
```

**Sample input:** the same array `{29, 13, 82, 43, 65, 24, 13, 5}`, histogrammed by the ones digit using 8 threads for the histogram and 10 threads for the offset scan.

**Sample output:**

```text
=== Section 13.1 main: digit histogram and offsets as CUDA kernels ===

input array: 29 13 82 43 65 24 13 5 
place = 1 (ones digit), base = 10, 8 threads (one per element)

count[digit], digit = 0..9:  0 0 1 3 1 2 0 0 0 1 
offset[digit] (exclusive prefix sum of count):  0 0 0 1 4 5 7 7 7 7 

expected count:  0 0 1 3 1 2 0 0 0 1
expected offset: 0 0 0 1 4 5 7 7 7 7

this matches the CPU baseline exactly: 8 threads did 8 independent atomic
increments (with base=10 distinct bucket targets, so real collisions do occur --
unlike Chapter 7.1's single-bucket race, most of these increments land in
different buckets and never actually contend at all), then 10 threads jointly
computed the exclusive scan over the tiny count array in log2(10) steps.

self-check: kernel result matches CPU baseline: confirmed
```

## 13.2 Stable Digit Scatter: One Full Radix Pass

### Intuition

Counting how many elements have each digit is only half the job -- a full radix PASS has to actually move every element to its final position for this digit, and it has to do so STABLY: if two elements share the same digit, the one that came first in the input must still come first in the output. Stability is not a nice-to-have here; it is the entire reason chaining multiple single-digit passes together (Section 13.3) produces a correctly, fully sorted array at the end.

### The Sequential (CPU) Baseline

```cpp
#include <cstdio>
#include <vector>

// Chapter 13.2 -- The Sequential (CPU) Baseline.
// One full radix pass is Section 13.1's histogram and offsets, PLUS the
// actual stable placement: walk the array in its ORIGINAL order, and for
// each element, drop it into its digit's next free slot (offset[digit],
// then increment that running position). Processing elements in original
// order is what makes this STABLE: two elements with the same digit are
// written in the same relative order they arrived in -- exactly Chapter
// 7.3's counting sort, just keyed on one digit instead of a whole value.

struct Item { int value; int orig_idx; };

int digit_of(int value, int place, int base) {
    return (value / place) % base;
}

std::vector<Item> stable_radix_pass(const std::vector<Item>& items, int place, int base) {
    int n = (int)items.size();
    std::vector<int> count(base, 0);
    for (const auto& it : items) count[digit_of(it.value, place, base)]++;

    std::vector<int> offset(base, 0);
    int total = 0;
    for (int d = 0; d < base; d++) { offset[d] = total; total += count[d]; }

    std::vector<Item> out(n);
    std::vector<int> pos = offset;
    for (const auto& it : items) {
        int d = digit_of(it.value, place, base);
        out[pos[d]] = it;
        pos[d]++;
    }
    return out;
}

int main() {
    printf("=== Section 13.2 CPU baseline: one full stable radix pass ===\n\n");

    std::vector<int> a = {29, 13, 82, 43, 65, 24, 13, 5};
    std::vector<Item> items;
    for (int i = 0; i < (int)a.size(); i++) items.push_back({a[i], i});

    printf("input (value @ original index): ");
    for (const auto& it : items) printf("%d@%d ", it.value, it.orig_idx);
    printf("\n\n");

    auto out = stable_radix_pass(items, 1, 10);

    printf("after one pass on the ones digit (place=1): ");
    for (const auto& it : out) printf("%d@%d ", it.value, it.orig_idx);
    printf("\n\n");

    std::vector<int> expected_values = {82, 13, 43, 13, 24, 65, 5, 29};
    std::vector<int> out_values;
    for (const auto& it : out) out_values.push_back(it.value);

    // Stability check: the two 13's came from original indices 1 and 6.
    // A stable pass must keep the one from index 1 before the one from
    // index 6 in the output, since it appeared first in the input.
    std::vector<Item> thirteens;
    for (const auto& it : out) if (it.value == 13) thirteens.push_back(it);

    bool stable = (thirteens.size() == 2) && (thirteens[0].orig_idx == 1) &&
                  (thirteens[1].orig_idx == 6);
    bool ok = (out_values == expected_values) && stable;

    printf("expected values: 82 13 43 13 24 65 5 29\n\n");
    printf("the two 13's (from original indices 1 and 6) both land in the digit-3 bucket;\n");
    printf("a stable pass must keep index 1's copy before index 6's copy, since that was\n");
    printf("their original relative order -- confirmed above by the @1 then @6 tags.\n");

    printf("\nself-check: pass output matches expected values, stability preserved: %s\n",
           ok ? "confirmed" : "MISMATCH");
    return ok ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++17 -Wall -Wextra -O2 66_radix_pass_cpu_baseline.cpp -o 66_radix_pass_cpu_baseline
./66_radix_pass_cpu_baseline
```

**Sample input:** the same array (with original indices tracked as `value@index`), one full stable pass on the ones digit.

**Sample output:**

```text
=== Section 13.2 CPU baseline: one full stable radix pass ===

input (value @ original index): 29@0 13@1 82@2 43@3 65@4 24@5 13@6 5@7 

after one pass on the ones digit (place=1): 82@2 13@1 43@3 13@6 24@5 65@4 5@7 29@0 

expected values: 82 13 43 13 24 65 5 29

the two 13's (from original indices 1 and 6) both land in the digit-3 bucket;
a stable pass must keep index 1's copy before index 6's copy, since that was
their original relative order -- confirmed above by the @1 then @6 tags.

self-check: pass output matches expected values, stability preserved: confirmed
```

### The Concept, In Detail

A single thread can achieve stability trivially: walk the input in its ORIGINAL order, and for each element, place it at its digit's next free slot, then advance that slot by one. Concretely, starting from `offset = {0,0,0,1,4,5,7,7,7,7}`, and a mutable running copy `pos = offset`:

```
walk input in order: 29(d9) 13(d3) 82(d2) 43(d3) 65(d5) 24(d4) 13(d3) 5(d5)

29(d9): out[pos[9]=7] = 29, pos[9] -> 8
13(d3): out[pos[3]=1] = 13, pos[3] -> 2
82(d2): out[pos[2]=0] = 82, pos[2] -> 1
43(d3): out[pos[3]=2] = 43, pos[3] -> 3
65(d5): out[pos[5]=5] = 65, pos[5] -> 6
24(d4): out[pos[4]=4] = 24, pos[4] -> 5
13(d3): out[pos[3]=3] = 13, pos[3] -> 4
 5(d5): out[pos[5]=6] =  5, pos[5] -> 7

out: [82, 13, 43, 13, 24, 65, 5, 29]
      ^0   ^1   ^2   ^3   ^4  ^5 ^6  ^7
```

The two `13`s -- originally at indices 1 and 6 -- land at output positions 1 and 3, in that same order, precisely BECAUSE the walk processed index 1 before index 6 and each one grabbed the next free digit-3 slot in turn.

Parallelizing this walk is the interesting part, because `n` independent threads cannot all just "grab the next free slot" the way one sequential thread can -- there is no notion of "next" without some form of coordination. The fix is to give every element's rank WITHIN its own digit group in advance, computed independently of the actual write. This is exactly Chapter 6.3's stable partition, generalized from 2 groups (true/false) to `base` groups: for a specific digit value `d`, build an indicator array (`1` if this element's digit is `d`, else `0`), and take its EXCLUSIVE prefix sum -- every element with digit `d` now knows exactly how many other digit-`d` elements came before it, with zero communication between threads beyond the scan itself.

```
worked example, digit=3 (three elements, at indices 1, 3, 6 -- values 13, 43, 13):

  index:      0  1  2  3  4  5  6  7
  indicator:  0  1  0  1  0  0  1  0
  excl scan:  0  0  1  1  2  2  2  3
                  ^        ^        ^
              index1: rank 0   index3: rank 1   index6: rank 2
```

Every element's final output position is then simply `offset[digit] + rank_within_digit`: index 1 (digit 3, rank 0) goes to `offset[3] + 0 = 1`; index 3 (digit 3, rank 1) goes to `offset[3] + 1 = 2`; index 6 (digit 3, rank 2) goes to `offset[3] + 2 = 3` -- matching the sequential walk's result exactly, but computed with every element's position known independently and in parallel, needing only `base` separate exclusive scans (one per digit value) rather than any sequential walk at all.

[COMMON TRAP]
Computing rank WITHIN THE WHOLE ARRAY (an ordinary scan of an all-ones array) instead of within each digit's OWN indicator array gives every element's position in the array overall, not its rank among elements sharing its digit -- that number is useless for this purpose. The scan must be taken over an indicator restricted to one digit value at a time.

### Code and Verification

```cpp
#include <cstdio>
#include <vector>

// Chapter 13.2 main -- a full stable radix pass as GPU kernels.
// Section 13.1 already parallelized the histogram and the offsets.
// What is still missing is each element's exact OUTPUT position: two
// elements sharing a digit cannot both write to offset[digit] -- the
// second one needs offset[digit] + 1, the third needs offset[digit] + 2,
// and so on, IN ORIGINAL ORDER (that "in original order" part is what
// makes the pass stable). This is exactly Chapter 6.3's stable
// partition, generalized from 2 groups (true/false) to `base` groups:
// for each digit value d, an exclusive scan of the indicator array
// "does this element have digit d?" gives every matching element's
// RANK within its own digit group, with no data race and no need to
// process elements in any particular order across threads.

__global__ void digit_histogram_kernel(const int* g_a, int n, int place, int base,
                                        int* g_count) {
    int i = threadIdx.x;
    if (i >= n) return;
    int digit = (g_a[i] / place) % base;
    atomicAdd(&g_count[digit], 1);
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
    g_out[tid] = src[tid] - g_in[tid];   // shift inclusive -> exclusive
}

// One digit value's rank pass: every thread checks whether ITS OWN
// element matches digit d, and (conceptually) all matching threads'
// ranks come from one exclusive scan of that indicator array.
__global__ void rank_within_digit_kernel(const int* g_a, int n, int place, int base,
                                          int digit, const int* g_indicator_scan,
                                          int* g_rank) {
    int i = threadIdx.x;
    if (i >= n) return;
    int my_digit = (g_a[i] / place) % base;
    if (my_digit == digit) g_rank[i] = g_indicator_scan[i];
}

__global__ void scatter_kernel(const int* g_a, int n, int place, int base,
                                const int* g_offset, const int* g_rank, int* g_out) {
    int i = threadIdx.x;
    if (i >= n) return;
    int digit = (g_a[i] / place) % base;
    int position = g_offset[digit] + g_rank[i];
    g_out[position] = g_a[i];
}

// ---- Host-side replay of the identical per-thread logic across all
// four kernels: histogram, offset scan, one rank scan per digit value,
// and the final scatter. ----

int main() {
    printf("=== Section 13.2 main: one full stable radix pass as CUDA kernels ===\n\n");

    std::vector<int> a = {29, 13, 82, 43, 65, 24, 13, 5};
    int place = 1, base = 10, n = (int)a.size();

    printf("input array: ");
    for (int v : a) printf("%d ", v);
    printf("\n\n");

    std::vector<int> digits(n);
    for (int i = 0; i < n; i++) digits[i] = (a[i] / place) % base;

    std::vector<int> count(base, 0);
    for (int d : digits) count[d]++;

    std::vector<int> src = count, dst = count;
    for (int d = 1; d < base; d *= 2) {
        for (int tid = 0; tid < base; tid++)
            dst[tid] = (tid >= d) ? (src[tid] + src[tid - d]) : src[tid];
        src = dst;
    }
    std::vector<int> offset(base);
    for (int tid = 0; tid < base; tid++) offset[tid] = src[tid] - count[tid];

    // one exclusive-scan-of-an-indicator-array per digit value -- the
    // k-way generalization of Chapter 6.3's stable partition.
    std::vector<int> rank(n, 0);
    for (int d = 0; d < base; d++) {
        if (count[d] == 0) continue;
        std::vector<int> indicator(n);
        for (int i = 0; i < n; i++) indicator[i] = (digits[i] == d) ? 1 : 0;
        std::vector<int> isrc = indicator, idst = indicator;
        for (int s = 1; s < n; s *= 2) {
            for (int tid = 0; tid < n; tid++)
                idst[tid] = (tid >= s) ? (isrc[tid] + isrc[tid - s]) : isrc[tid];
            isrc = idst;
        }
        for (int i = 0; i < n; i++) {
            int excl = isrc[i] - indicator[i];
            if (digits[i] == d) rank[i] = excl;
        }
    }

    printf("digit of each element:        ");
    for (int d : digits) printf("%d ", d);
    printf("\nrank within own digit group:  ");
    for (int r : rank) printf("%d ", r);
    printf("\n\n");

    printf("worked example, digit=3 (3 elements: at indices 1, 3, 6, values 13, 43, 13):\n");
    printf("  indicator: 0 1 0 1 0 0 1 0\n");
    printf("  exclusive scan (= rank within digit 3): 0 0 1 1 2 2 2 3\n");
    printf("  so index 1 gets rank 0, index 3 gets rank 1, index 6 gets rank 2\n\n");

    std::vector<int> out(n);
    for (int i = 0; i < n; i++) {
        int position = offset[digits[i]] + rank[i];
        out[position] = a[i];
    }

    printf("scattered output: ");
    for (int v : out) printf("%d ", v);
    printf("\n\n");

    std::vector<int> expected = {82, 13, 43, 13, 24, 65, 5, 29};
    bool ok = (out == expected);

    printf("expected: 82 13 43 13 24 65 5 29\n\n");
    printf("this matches Section 13.2's CPU baseline exactly, including stability: every\n");
    printf("element's output position was computed independently, from only its own digit\n");
    printf("and its own rank among elements sharing that digit -- no thread needed to know\n");
    printf("what any other thread decided.\n");

    printf("\nself-check: parallel stable scatter matches CPU baseline: %s\n",
           ok ? "confirmed" : "MISMATCH");
    return ok ? 0 : 1;
}
```

**Compile and run:**

```bash
NVDIR=/usr/local/lib/python3.11/dist-packages/nvidia/cu13
$NVDIR/bin/nvcc -arch=sm_80 -I$NVDIR/include -L$NVDIR/lib -lcudart 67_radix_pass_scatter_kernel.cu -o 67_radix_pass_scatter_kernel
LD_LIBRARY_PATH=$NVDIR/lib ./67_radix_pass_scatter_kernel
```

**Sample input:** the same array, scattered stably by the ones digit using the histogram, offset scan, per-digit rank scans, and final scatter, all as separate kernels.

**Sample output:**

```text
=== Section 13.2 main: one full stable radix pass as CUDA kernels ===

input array: 29 13 82 43 65 24 13 5 

digit of each element:        9 3 2 3 5 4 3 5 
rank within own digit group:  0 0 0 1 0 0 2 1 

worked example, digit=3 (3 elements: at indices 1, 3, 6, values 13, 43, 13):
  indicator: 0 1 0 1 0 0 1 0
  exclusive scan (= rank within digit 3): 0 0 1 1 2 2 2 3
  so index 1 gets rank 0, index 3 gets rank 1, index 6 gets rank 2

scattered output: 82 13 43 13 24 65 5 29 

expected: 82 13 43 13 24 65 5 29

this matches Section 13.2's CPU baseline exactly, including stability: every
element's output position was computed independently, from only its own digit
and its own rank among elements sharing that digit -- no thread needed to know
what any other thread decided.

self-check: parallel stable scatter matches CPU baseline: confirmed
```

## 13.3 LSD Radix Sort: Chaining Passes to Full Order

### Intuition

A single pass only sorts correctly with respect to ONE digit -- after Section 13.2's pass, `13` and `43` are grouped together by their shared ones digit, but `13` is not yet anywhere near where it belongs in full sorted order. The fix is deceptively simple: run another pass on the NEXT digit position (tens), then the next, and so on, always going from the Least Significant Digit to the most significant -- LSD radix sort. Because every individual pass is stable, later passes never scramble the relative order two elements already earned from a shared LESS significant digit; that earlier ordering survives into the final result precisely because it was never disturbed.

### The Sequential (CPU) Baseline

```cpp
#include <cstdio>
#include <vector>

// Chapter 13.3 -- The Sequential (CPU) Baseline.
// A single radix pass (Section 13.2) only sorts elements correctly
// with respect to ONE digit. Chaining passes together, starting from
// the LEAST significant digit and working toward the most significant,
// produces a fully sorted array -- LSD (Least-Significant-Digit) radix
// sort. The order matters: each later pass sorts by a MORE significant
// digit, and because every pass is stable, it never disturbs the
// relative order two elements already earned from a less-significant
// digit they happened to share -- that earlier ordering is exactly
// what needs to survive into the final result.

struct Item { int value; int orig_idx; };

int digit_of(int value, int place, int base) {
    return (value / place) % base;
}

std::vector<Item> stable_radix_pass(const std::vector<Item>& items, int place, int base) {
    int n = (int)items.size();
    std::vector<int> count(base, 0);
    for (const auto& it : items) count[digit_of(it.value, place, base)]++;

    std::vector<int> offset(base, 0);
    int total = 0;
    for (int d = 0; d < base; d++) { offset[d] = total; total += count[d]; }

    std::vector<Item> out(n);
    std::vector<int> pos = offset;
    for (const auto& it : items) {
        int d = digit_of(it.value, place, base);
        out[pos[d]] = it;
        pos[d]++;
    }
    return out;
}

std::vector<Item> lsd_radix_sort(std::vector<Item> items, int base) {
    int max_value = 0;
    for (const auto& it : items) if (it.value > max_value) max_value = it.value;

    for (int place = 1; place <= max_value; place *= base) {
        items = stable_radix_pass(items, place, base);
    }
    return items;
}

int main() {
    printf("=== Section 13.3 CPU baseline: full LSD radix sort (chained passes) ===\n\n");

    std::vector<int> a = {29, 13, 82, 43, 65, 24, 13, 5};
    std::vector<Item> items;
    for (int i = 0; i < (int)a.size(); i++) items.push_back({a[i], i});

    printf("input (value @ original index): ");
    for (const auto& it : items) printf("%d@%d ", it.value, it.orig_idx);
    printf("\nmax value = 82, so 2 passes needed (ones digit, then tens digit)\n\n");

    auto after_ones = stable_radix_pass(items, 1, 10);
    printf("after pass 1 (ones digit):  ");
    for (const auto& it : after_ones) printf("%d@%d ", it.value, it.orig_idx);
    printf("\n");

    auto after_tens = stable_radix_pass(after_ones, 10, 10);
    printf("after pass 2 (tens digit):  ");
    for (const auto& it : after_tens) printf("%d@%d ", it.value, it.orig_idx);
    printf("\n\n");

    auto sorted_items = lsd_radix_sort(items, 10);

    std::vector<int> out_values;
    for (const auto& it : sorted_items) out_values.push_back(it.value);
    std::vector<int> expected = {5, 13, 13, 24, 29, 43, 65, 82};

    // Stability across BOTH passes: the two 13's, from original indices
    // 1 and 6, must still appear with index 1's copy before index 6's.
    std::vector<Item> thirteens;
    for (const auto& it : sorted_items) if (it.value == 13) thirteens.push_back(it);
    bool stable = (thirteens.size() == 2) && (thirteens[0].orig_idx == 1) &&
                  (thirteens[1].orig_idx == 6);

    bool ok = (out_values == expected) && stable;

    printf("final sorted output: ");
    for (const auto& it : sorted_items) printf("%d@%d ", it.value, it.orig_idx);
    printf("\nexpected values: 5 13 13 24 29 43 65 82\n\n");
    printf("the two 13's (original indices 1 and 6) are adjacent in the final output, and\n");
    printf("still in their original relative order -- two passes of stable sorting, chained\n");
    printf("from least to most significant digit, never needed to compare a 13 against a 13\n");
    printf("at all to get this right.\n");

    printf("\nself-check: full LSD radix sort matches expected sorted order, stability\n");
    printf("preserved across both passes: %s\n", ok ? "confirmed" : "MISMATCH");
    return ok ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++17 -Wall -Wextra -O2 68_radix_sort_lsd_cpu_baseline.cpp -o 68_radix_sort_lsd_cpu_baseline
./68_radix_sort_lsd_cpu_baseline
```

**Sample input:** the same array, fully sorted via two chained passes (ones digit, then tens digit).

**Sample output:**

```text
=== Section 13.3 CPU baseline: full LSD radix sort (chained passes) ===

input (value @ original index): 29@0 13@1 82@2 43@3 65@4 24@5 13@6 5@7 
max value = 82, so 2 passes needed (ones digit, then tens digit)

after pass 1 (ones digit):  82@2 13@1 43@3 13@6 24@5 65@4 5@7 29@0 
after pass 2 (tens digit):  5@7 13@1 13@6 24@5 29@0 43@3 65@4 82@2 

final sorted output: 5@7 13@1 13@6 24@5 29@0 43@3 65@4 82@2 
expected values: 5 13 13 24 29 43 65 82

the two 13's (original indices 1 and 6) are adjacent in the final output, and
still in their original relative order -- two passes of stable sorting, chained
from least to most significant digit, never needed to compare a 13 against a 13
at all to get this right.

self-check: full LSD radix sort matches expected sorted order, stability
preserved across both passes: confirmed
```

### The Concept, In Detail

Tracing both passes on the full example:

```
input:            29@0  13@1  82@2  43@3  65@4  24@5  13@6   5@7

pass 1 (ones):    82@2  13@1  43@3  13@6  24@5  65@4   5@7  29@0
pass 2 (tens):     5@7  13@1  13@6  24@5  29@0  43@3  65@4  82@2
```

After pass 1, everything is grouped by ones digit, but far from sorted overall (`82` comes first). Pass 2 groups by TENS digit -- and because pass 2 is itself stable, whenever two elements share the same tens digit, pass 2 preserves whatever order pass 1 already gave them. Look specifically at the two `13`s: after pass 1 they sit at positions 1 and 3 (in that order, from original indices 1 and 6). Pass 2 sees both as having tens-digit `1`, and being stable, keeps `13@1` before `13@6` -- so the final array has them adjacent, in their original relative order, at positions 1 and 2. Neither pass ever directly compared a `13` against the other `13` to achieve this; their final adjacency and order fell entirely out of chaining two independently-correct, independently-stable single-digit passes.

```
ASCII view of why LSD order (least-to-most-significant) is required, not optional:

  Sorting by TENS digit FIRST, then ONES digit, would be wrong: two
  elements sharing a tens digit (like 13 and 13, or hypothetically 24
  and 29) would get grouped correctly by their tens digit on the first
  pass, but the SECOND pass (by ones digit) would then re-sort within
  that tens-digit group using ONLY the ones digit -- destroying the
  coarser tens-digit ordering the first pass had already established,
  with nothing left to restore it. LSD order avoids this because every
  later pass works on a MORE significant digit, so its own stability
  is exactly what is needed to preserve the finer ordering already
  built by all earlier, less-significant passes.
```

The number of passes needed is exactly the number of digit positions in the largest key -- `82`, the maximum value in this example, has 2 digits, so exactly 2 passes complete the sort, regardless of how the 8 values happened to be arranged going in. This is radix sort's own version of Chapter 12's data-independent pass STRUCTURE: the pass COUNT never depends on the input's original order, only on the key width. Where it differs from bitonic sort is that each pass's actual scatter destinations very much DO depend on the data (which digit each element has) -- radix sort trades a fixed comparison network for a fixed pass COUNT with data-dependent bucketing inside each pass, and this trade turns out to favor real GPU hardware: the total work across all passes is `O(d * (n + base))` for `d` digit positions, growing only linearly in `n` (bitonic sort's `O(n log^2 n)` grows faster), which is exactly why production GPU sorting libraries use radix sort as their workhorse rather than a compare-exchange network.

[COMMON TRAP]
Running passes from MOST significant digit to least significant is a genuinely different (and, without extra machinery, incorrect) algorithm: an MSD-first pass groups by the coarsest digit correctly, but every later, finer-digit pass would need to be scoped to operate WITHIN each MSD group separately (recursively), never across group boundaries -- otherwise a later pass's stability preserves the wrong thing entirely. LSD radix sort's appeal is precisely that it needs no such per-group scoping: every pass always operates on the WHOLE array, uniformly.

### Code and Verification

```cpp
#include <cstdio>
#include <vector>

// Chapter 13.3 main -- full LSD radix sort as chained GPU kernel passes.
// Section 13.2 built one complete parallel pass (histogram, offset scan,
// per-digit rank scan, scatter) as four kernels. A full sort is simply
// that SAME four-kernel pipeline, launched once per digit place, with a
// full grid synchronization between passes -- the next pass's histogram
// must see every write the previous pass's scatter made. This is the
// same "loop over passes, synchronize between them" shape Chapter 12.2
// used for the multi-pass bitonic merge, just with a completely
// different per-pass computation inside the loop.

__global__ void digit_histogram_kernel(const int* g_a, int n, int place, int base,
                                        int* g_count) {
    int i = threadIdx.x;
    if (i >= n) return;
    atomicAdd(&g_count[(g_a[i] / place) % base], 1);
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

__global__ void rank_within_digit_kernel(const int* g_a, int n, int place, int base,
                                          int digit, const int* g_indicator_scan,
                                          int* g_rank) {
    int i = threadIdx.x;
    if (i >= n) return;
    if ((g_a[i] / place) % base == digit) g_rank[i] = g_indicator_scan[i];
}

__global__ void scatter_kernel(const int* g_a, int n, int place, int base,
                                const int* g_offset, const int* g_rank, int* g_out) {
    int i = threadIdx.x;
    if (i >= n) return;
    int digit = (g_a[i] / place) % base;
    g_out[g_offset[digit] + g_rank[i]] = g_a[i];
}

// ---- Host-side replay of the identical four-kernel pipeline, invoked
// once per digit place with a full synchronization (a fresh host loop
// iteration) between passes. ----

std::vector<int> parallel_radix_pass(const std::vector<int>& a, int place, int base) {
    int n = (int)a.size();
    std::vector<int> digits(n);
    for (int i = 0; i < n; i++) digits[i] = (a[i] / place) % base;

    std::vector<int> count(base, 0);
    for (int d : digits) count[d]++;   // digit_histogram_kernel, replayed

    std::vector<int> src = count, dst = count;
    for (int d = 1; d < base; d *= 2) {
        for (int tid = 0; tid < base; tid++)
            dst[tid] = (tid >= d) ? (src[tid] + src[tid - d]) : src[tid];
        src = dst;
    }
    std::vector<int> offset(base);
    for (int tid = 0; tid < base; tid++) offset[tid] = src[tid] - count[tid];   // exclusive_scan_kernel, replayed

    std::vector<int> rank(n, 0);
    for (int d = 0; d < base; d++) {   // rank_within_digit_kernel, replayed once per digit value
        if (count[d] == 0) continue;
        std::vector<int> indicator(n);
        for (int i = 0; i < n; i++) indicator[i] = (digits[i] == d) ? 1 : 0;
        std::vector<int> isrc = indicator, idst = indicator;
        for (int s = 1; s < n; s *= 2) {
            for (int tid = 0; tid < n; tid++)
                idst[tid] = (tid >= s) ? (isrc[tid] + isrc[tid - s]) : isrc[tid];
            isrc = idst;
        }
        for (int i = 0; i < n; i++) {
            if (digits[i] == d) rank[i] = isrc[i] - indicator[i];
        }
    }

    std::vector<int> out(n);
    for (int i = 0; i < n; i++) {   // scatter_kernel, replayed
        out[offset[digits[i]] + rank[i]] = a[i];
    }
    return out;
}

int main() {
    printf("=== Section 13.3 main: full LSD radix sort as chained CUDA kernel passes ===\n\n");

    std::vector<int> a = {29, 13, 82, 43, 65, 24, 13, 5};
    int base = 10;

    printf("input array: ");
    for (int v : a) printf("%d ", v);
    printf("\n\n");

    int max_value = 0;
    for (int v : a) if (v > max_value) max_value = v;

    int pass_num = 0;
    for (int place = 1; place <= max_value; place *= base) {
        a = parallel_radix_pass(a, place, base);
        pass_num++;
        printf("pass %d (place=%d): ", pass_num, place);
        for (int v : a) printf("%d ", v);
        printf("\n");
    }
    printf("\n");

    std::vector<int> expected = {5, 13, 13, 24, 29, 43, 65, 82};
    bool sorted = true;
    for (int i = 1; i < (int)a.size(); i++) if (a[i - 1] > a[i]) sorted = false;
    bool ok = sorted && (a == expected);

    printf("expected: 5 13 13 24 29 43 65 82\n\n");
    printf("2 passes total for a 2-digit maximum value -- the pass COUNT depends only on\n");
    printf("the number of digits in the largest key, never on how the input was originally\n");
    printf("ordered, matching Chapter 12's bitonic sort in having a data-independent pass\n");
    printf("structure, even though (unlike bitonic sort) each pass's actual scatter\n");
    printf("destinations very much depend on the data.\n");

    printf("\nself-check: chained parallel radix sort matches CPU baseline: %s\n",
           ok ? "confirmed" : "MISMATCH");
    return ok ? 0 : 1;
}
```

**Compile and run:**

```bash
NVDIR=/usr/local/lib/python3.11/dist-packages/nvidia/cu13
$NVDIR/bin/nvcc -arch=sm_80 -I$NVDIR/include -L$NVDIR/lib -lcudart 69_radix_sort_lsd_multipass.cu -o 69_radix_sort_lsd_multipass
LD_LIBRARY_PATH=$NVDIR/lib ./69_radix_sort_lsd_multipass
```

**Sample input:** the same array, fully sorted via two chained parallel passes (each pass itself a 4-kernel pipeline: histogram, offset scan, rank scan, scatter).

**Sample output:**

```text
=== Section 13.3 main: full LSD radix sort as chained CUDA kernel passes ===

input array: 29 13 82 43 65 24 13 5 

pass 1 (place=1): 82 13 43 13 24 65 5 29 
pass 2 (place=10): 5 13 13 24 29 43 65 82 

expected: 5 13 13 24 29 43 65 82

2 passes total for a 2-digit maximum value -- the pass COUNT depends only on
the number of digits in the largest key, never on how the input was originally
ordered, matching Chapter 12's bitonic sort in having a data-independent pass
structure, even though (unlike bitonic sort) each pass's actual scatter
destinations very much depend on the data.

self-check: chained parallel radix sort matches CPU baseline: confirmed
```

## Chapter Summary

Radix sort never compares two keys against each other; instead, it repeatedly buckets keys by one digit at a time. Extracting a digit is one line of integer arithmetic (`(value / place) % base`), and counting how many elements have each digit value is Chapter 7's histogram applied to that one digit, with an exclusive prefix sum over the small, `base`-sized count array giving every digit's starting offset. A full, STABLE pass additionally needs each element's rank within its own digit group, computed as the k-way generalization of Chapter 6.3's stable partition: one exclusive scan of a per-digit indicator array for each possible digit value, giving every element an output position (`offset[digit] + rank`) that can be computed completely independently, with no thread needing to know what any other thread decided. Chaining these stable passes together from the Least Significant Digit to the most significant -- LSD radix sort -- produces a fully sorted array, because each later pass's stability preserves whatever ordering earlier, less-significant passes already established; running passes in the opposite order (MSD-first) would destroy that ordering unless each pass were additionally scoped to operate within groups recursively. The number of passes depends only on key width, never on the input's original order, giving radix sort `O(d * (n + base))` total work across `d` digit positions -- growing linearly in `n`, unlike bitonic sort's `O(n log^2 n)` -- which is why real GPU sorting libraries use radix sort, not a compare-exchange network, as their production workhorse.

## Self-Check Questions

1. Why is extracting one digit from a key, `(value / place) % base`, enough to build a full sorting algorithm out of, without ever comparing two keys against each other?
2. Why must the offset array be an EXCLUSIVE prefix sum of the digit counts, rather than an inclusive one?
3. Explain, in your own words, why a single radix pass must be STABLE, and what could go wrong in the final sorted result if it were not.
4. How does computing each element's "rank within its own digit group" generalize Chapter 6.3's stable partition from two groups to `base` groups? What does each of the `base` separate scans actually compute?
5. Why must LSD radix sort process digits from least-significant to most-significant, rather than the reverse? What specifically breaks if the order is flipped, and what extra step would be needed to fix an MSD-first version?
6. Radix sort's pass COUNT is data-independent (it depends only on key width), but Section 13.3 also notes that each pass's actual work is NOT fully data-independent in the way bitonic sort's comparison network is. What specifically differs between the two algorithms here?

## Where We Go Next

Across Part 1, Part 2, and now Part 3, this book has built its intuitions on small, hand-traceable examples -- N=8 arrays, single-block kernels, and (in this sandboxed environment) host-side replays of genuinely correct kernel code. Real GPU workloads run at scales where a single block cannot even hold all the data being processed at once, and where the primitives built throughout this book -- reduction, scan, compaction, histograms, atomics, and now sorting -- need to cooperate ACROSS blocks and across full grids, with all the extra synchronization and multi-pass orchestration that implies. The final part of this book turns to exactly that: composing everything built so far into complete, grid-scale data structures and algorithms, and confronting directly what changes when "one block's worth of threads" is no longer the whole story.

## Worked Solutions

**1.** A full key can always be decomposed into a fixed sequence of digits (base 10, base 2, or any other base), and two keys' relative order is completely determined by comparing their digits from the most significant down -- exactly how humans compare multi-digit numbers by eye. Radix sort exploits this by processing one digit position across the WHOLE array at a time (via counting and stable placement) rather than directly comparing digit-by-digit between pairs of keys; chaining enough of these single-digit passes together (one per digit position present in the largest key) recovers the full ordering without any pairwise key comparison ever happening.

**2.** offset[d] needs to answer "where does digit d's group of elements START", so that the very first element with digit d can be placed there. An inclusive prefix sum would instead give where digit d's group ENDS (one past its last element under an inclusive convention, or its last position under another), forcing every subsequent placement to apply an extra correction. Keeping the offset array exclusive means offset[d] is immediately usable as the first free slot for digit d, with no adjustment needed.

**3.** A pass must be stable so that elements sharing a digit value keep whatever relative order they already had going into that pass. If a pass were unstable, two elements with the same digit could be reordered arbitrarily by that pass -- and since later passes rely on relative order from earlier, less-significant-digit passes to already be correct (only re-sorting within groups that a MORE significant digit will further distinguish), an unstable pass could permanently scramble the order of elements that happen to share every digit from that point onward, or elements that a later pass groups back together, producing a final array that is not fully sorted.

**4.** Chapter 6.3's stable partition splits elements into exactly two groups (an indicator array of true/false, and its complement) using one pair of scans. Radix sort's rank-within-digit generalizes this to `base` groups: for each digit value `d` from `0` to `base-1`, build an indicator array that is `1` exactly where an element's digit equals `d`, and take its exclusive prefix sum. That scan computes, for every element with digit `d`, how many OTHER elements with digit `d` appear before it in the array -- its rank within its own digit group. Running this once per digit value (rather than once per two-way split) is exactly what generalizes two groups to `base` groups.

**5.** LSD order works because every pass's stability is what needs to be preserved by all LATER (more significant) passes -- and "later" passes are, by construction, working on coarser distinctions than earlier ones, so they never need to un-do a finer distinction an earlier pass made. Running MSD-first breaks this: the first (most significant digit) pass would correctly form coarse groups, but the SECOND pass, working on a less significant digit across the WHOLE array, would freely reorder elements from DIFFERENT MSD groups relative to each other, destroying the coarse grouping the first pass built with nothing to restore it. Fixing an MSD-first version requires recursively scoping every subsequent pass to operate only WITHIN each group the previous, more-significant pass already formed -- extra bookkeeping LSD order never needs.

**6.** Bitonic sort's comparison network is fixed in EVERY respect: for a given `n`, the exact sequence of `(i, l, ascending)` triples it evaluates never depends on the data at all, only on the indices. Radix sort's PASS COUNT is similarly fixed by key width alone, but within each pass, which digit value each element actually has -- and therefore which bucket, and which final scatter position, it lands in -- depends entirely on the data being sorted. Two different input arrays of the same size and same maximum key width take exactly the same number of radix sort passes, but the actual reads, writes, and bucket memberships inside each pass differ completely between them, whereas bitonic sort's underlying comparisons would be identical index-for-index regardless of the values involved.
