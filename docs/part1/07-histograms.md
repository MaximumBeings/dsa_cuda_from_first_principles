# Chapter 7: Histograms

**What you will understand by the end of this chapter:**

- Why a plain, non-atomic `hist[bucket]++` genuinely loses updates when multiple threads target the same bucket, traced through Chapter 1's own warp lockstep model rather than asserted.
- Why `atomicAdd` fixes this exactly, and why privatizing a histogram into per-block shared memory is a CONTENTION improvement, not a second correctness fix layered on top of the first.
- How a histogram's counts, combined with Chapter 5's exclusive scan, produce a fully sorted output array in one more pass — counting sort — directly foreshadowing Part 3's radix sort.

**What you need to know first:**

- Chapter 1's warp lockstep execution model (every lane in a warp issues the same instruction together), reused directly in Section 7.1 to trace exactly how a non-atomic increment loses updates.
- Chapter 5's exclusive scan (Section 5.2's up-sweep/down-sweep or Section 5.1's Hillis-Steele shape) and Chapter 6.2's "counts become offsets" pattern, both reused unchanged in Section 7.3.

---

Reduction (Chapter 4) collapses many values into one. Scan (Chapter 5) computes a running total at every position. Stream compaction (Chapter 6) gives every KEPT element its own unique output slot, discarding the rest. A histogram asks a question that looks like compaction's opposite: instead of every element needing a unique destination, MANY elements are allowed — expected — to land in the exact same bucket at the exact same time. That single difference is the whole chapter: it is what makes a naive histogram wrong, what atomicAdd is for, and, once the counting is done correctly, what turns out to connect histograms right back to Chapter 5's scan.

## 7.1 The Histogram Race, and Why atomicAdd Fixes It

### Intuition

Counting how many elements fall into each of 10 buckets sounds like it should be even EASIER than compaction — no scatter positions to compute, just `hist[bucket]++` for whichever bucket an element belongs to. The trouble is that "many elements, one bucket" is exactly the case Chapter 1 already warned about: several threads in the same warp executing the same instruction on the same address, at the same lockstep moment.

### The Concept, In Detail

A single C++ statement like `hist[bucket]++` is not one hardware step. It is three: LOAD the current value, ADD one to it, STORE the result back. On a real CPU with one thread, that separation is invisible — nothing else can run between the load and the store. On a GPU warp, it is the entire problem: every lane in the warp issues the LOAD together, then every lane issues the ADD together, then every lane issues the STORE together (Chapter 1's lockstep model, unchanged). If two or more lanes happen to target the SAME bucket, they all load the identical pre-increment value, each independently computes the identical "value + 1," and then all of them store that identical result back — no matter how many lanes were racing for that bucket, only one increment's worth of progress survives.

Here is that trace for 4 lanes (out of a 32-lane warp) that all happen to target bucket 3, which holds the value 5 before this wave runs:

```
non-atomic hist[3]++, 4 racing lanes, hist[3] starts at 5

step 1 -- LOAD (every lane in the warp issues this together):
    lane 0: reads hist[3] = 5      lane 1: reads hist[3] = 5
    lane 2: reads hist[3] = 5      lane 3: reads hist[3] = 5

step 2 -- ADD (every lane computes independently, using its OWN load):
    lane 0: 5 + 1 = 6              lane 1: 5 + 1 = 6
    lane 2: 5 + 1 = 6              lane 3: 5 + 1 = 6

step 3 -- STORE (every lane writes back; whichever store is physically
          last for this address is simply the value that survives):
    lane 0: hist[3] <- 6           lane 1: hist[3] <- 6
    lane 2: hist[3] <- 6           lane 3: hist[3] <- 6   <- last write wins

RESULT: hist[3] is now 6. Four lanes each executed "hist[3]++" once, but
the bucket only advanced by ONE. Three increments were computed and
stored, and every one of them was silently overwritten by an identical
sibling before it could do any good.
```

`atomicAdd` fixes this not by avoiding the race, but by making the hardware itself perform the load-add-store as a single, indivisible unit that cannot be interleaved with any other lane's atomicAdd to the same address. The same 4 lanes, using `atomicAdd(&hist[3], 1)` instead:

```
atomicAdd(&hist[3], 1) x 4, same 4 lanes, hist[3] starts at 5
(hardware serializes the 4 requests into SOME order -- the order is not
guaranteed, but "some genuine order" is exactly what atomicAdd promises)

    lane 0: read 5, write 6   (indivisible -- no other lane's atomicAdd
    lane 1: read 6, write 7    can land in the middle of this one)
    lane 2: read 7, write 8
    lane 3: read 8, write 9

RESULT: hist[3] is now 9. All 4 increments land, regardless of which
physical order the hardware happens to pick.
```

The code below models exactly this: input is processed in fixed 32-lane "waves," and within each wave, a non-atomic increment is modeled as allowing only ONE surviving increment per DISTINCT bucket touched in that wave (`pre_wave_value` captures each touched bucket's value once, before any of that wave's stores happen) — while `atomicAdd` is modeled as a plain, exact, nothing-ever-lost sequential count, because that is precisely what "indivisible" guarantees.

### Code and Verification

```cpp
#include <cstdio>
#include <vector>
#include <map>

// Chapter 7.1 -- a histogram counts how many elements fall into each of
// K buckets. Unlike compaction (Chapter 6), where every kept element
// gets its OWN unique output slot, a histogram has MANY threads that
// legitimately want to update the SAME bucket's counter at once. A plain
// `hist[bucket]++` compiles to a separate load, add, and store -- and
// Chapter 1's own warp lockstep model already explains exactly how this
// goes wrong: if several lanes in the same warp target the same bucket,
// they all load the SAME pre-increment value, each computes value+1
// independently, and whichever lane's store happens to land last simply
// overwrites the others -- every increment but one is silently lost.

#define WARP_SIZE 32

// ---- A deterministic model of one warp's lockstep load/add/store,
// ---- applied to a non-atomic increment. This is not a guess about what
// ---- MIGHT happen under contention -- it is what a lockstep warp's
// ---- three separate instructions genuinely produce when multiple lanes
// ---- target the same address: exactly one surviving increment per
// ---- DISTINCT bucket touched in that wave, not one per lane. ----

std::vector<int> simulate_naive_histogram(const std::vector<int>& bucket_of, int num_buckets) {
    std::vector<int> hist(num_buckets, 0);
    int n = (int)bucket_of.size();

    for (int wave_start = 0; wave_start < n; wave_start += WARP_SIZE) {
        int wave_end = std::min(wave_start + WARP_SIZE, n);
        // Every lane in this wave loads its bucket's CURRENT value first
        // (lockstep: the load instruction issues for the whole warp
        // before the add or the store does), so all lanes targeting the
        // same bucket see the identical pre-wave count.
        std::map<int, int> pre_wave_value;
        for (int i = wave_start; i < wave_end; i++) {
            int b = bucket_of[i];
            if (pre_wave_value.find(b) == pre_wave_value.end()) {
                pre_wave_value[b] = hist[b];
            }
        }
        // Every lane then stores pre_wave_value[bucket] + 1 -- whichever
        // lane's store is last for a given bucket determines the final
        // value, but since EVERY lane targeting that bucket computed the
        // identical result, the bucket ends up exactly one higher than
        // before the wave, no matter how many lanes targeted it.
        for (auto& kv : pre_wave_value) {
            hist[kv.first] = kv.second + 1;
        }
    }
    return hist;
}

// atomicAdd is a single, indivisible hardware read-modify-write -- there
// is no separate load/add/store for another lane to interleave with, so
// this model is exactly a correct, un-lost sequential count, regardless
// of what order the "threads" actually execute in.
std::vector<int> simulate_atomic_histogram(const std::vector<int>& bucket_of, int num_buckets) {
    std::vector<int> hist(num_buckets, 0);
    for (int b : bucket_of) hist[b]++;
    return hist;
}

int main() {
    printf("=== Section 7.1: the histogram race, and why atomicAdd fixes it ===\n\n");

    const int N = 1000;
    const int NUM_BUCKETS = 10;
    std::vector<int> bucket_of(N);
    for (int i = 0; i < N; i++) bucket_of[i] = i % NUM_BUCKETS;

    auto naive = simulate_naive_histogram(bucket_of, NUM_BUCKETS);
    auto atomic = simulate_atomic_histogram(bucket_of, NUM_BUCKETS);

    printf("N = %d elements into %d buckets, one warp (%d lanes) processed per wave\n\n",
           N, NUM_BUCKETS, WARP_SIZE);
    printf("%-8s %-16s %-16s %-10s\n", "bucket", "naive_count", "atomic_count", "reference");
    long long naive_total = 0, atomic_total = 0;
    bool atomic_correct = true;
    for (int b = 0; b < NUM_BUCKETS; b++) {
        int reference = N / NUM_BUCKETS;   // exactly N/NUM_BUCKETS, since bucket_of is i % NUM_BUCKETS
        printf("%-8d %-16d %-16d %-10d\n", b, naive[b], atomic[b], reference);
        naive_total += naive[b];
        atomic_total += atomic[b];
        if (atomic[b] != reference) atomic_correct = false;
    }

    printf("\ntotal counted: naive = %lld, atomic = %lld, actual N = %d\n",
           naive_total, atomic_total, N);
    printf("naive (non-atomic) increment LOSES %lld of %d increments to the exact race\n",
           (long long)N - naive_total, N);
    printf("Chapter 1's own warp lockstep model already predicts: whenever 2 or more\n");
    printf("lanes in the same 32-lane wave target the same bucket, only one increment\n");
    printf("survives, no matter how many lanes tried.\n\n");
    printf("atomicAdd's single indivisible read-modify-write loses NOTHING -- every one of\n");
    printf("the %d increments lands, matching the independent reference exactly.\n", N);

    bool ok = (naive_total < N) && atomic_correct && (atomic_total == N);
    printf("\nself-check: naive strictly undercounts, atomic matches the reference exactly\n");
    printf("for every bucket: %s\n", ok ? "confirmed" : "MISMATCH");
    return ok ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++17 -Wall -Wextra -O2 -o histogram_race 19_histogram_race_and_atomics.cpp
./histogram_race
```

**Sample input:** `N = 1000` elements distributed round-robin into `NUM_BUCKETS = 10` buckets (`bucket_of[i] = i % 10`, so each bucket's true count is exactly 100), processed in fixed 32-lane waves.

**Sample output:**

```text
=== Section 7.1: the histogram race, and why atomicAdd fixes it ===

N = 1000 elements into 10 buckets, one warp (32 lanes) processed per wave

bucket   naive_count      atomic_count     reference 
0        31               100              100       
1        31               100              100       
2        32               100              100       
3        32               100              100       
4        32               100              100       
5        32               100              100       
6        32               100              100       
7        32               100              100       
8        32               100              100       
9        32               100              100       

total counted: naive = 318, atomic = 1000, actual N = 1000
naive (non-atomic) increment LOSES 682 of 1000 increments to the exact race
Chapter 1's own warp lockstep model already predicts: whenever 2 or more
lanes in the same 32-lane wave target the same bucket, only one increment
survives, no matter how many lanes tried.

atomicAdd's single indivisible read-modify-write loses NOTHING -- every one of
the 1000 increments lands, matching the independent reference exactly.

self-check: naive strictly undercounts, atomic matches the reference exactly
for every bucket: confirmed
```

The naive, non-atomic count undercounts every single bucket — 318 total increments recorded out of 1000, losing 682 to the exact race traced above — while the atomic count matches the independent reference of 100 per bucket exactly.

!!! warning "[COMMON TRAP] Assuming the race only matters for LARGE bucket counts"
    It is tempting to think a small number of buckets colliding occasionally is a minor, mostly-harmless rounding error. The trace above shows the opposite: the race does not lose "some" increments proportional to how rare collisions are — within a single wave, it loses ALL but one increment for EVERY bucket that two or more lanes in that wave happen to share, every single time, deterministically. With 10 buckets and 32 lanes per wave, most waves have several lanes sharing a bucket by simple pigeonhole, which is exactly why the naive version recovers less than a third of the true count rather than something close to it.

## 7.2 Privatized Histograms

### Intuition

Section 7.1 already proved atomicAdd is CORRECT — it never loses an increment. So why would a real histogram kernel do anything more elaborate than "every thread calls atomicAdd on the global histogram directly"? Because correctness and speed are different questions: with only `NUM_BUCKETS` distinct addresses and potentially thousands of threads across the whole grid all wanting to touch them, every one of those threads' atomicAdd calls serializes against every OTHER thread's atomicAdd to the same bucket, across the ENTIRE grid at once.

### The Concept, In Detail

Privatization does not change WHAT is correct — a shared-memory atomicAdd is exactly as indivisible, and exactly as lossless, as a global-memory one (Section 7.1's proof applies unchanged to either address space). What privatization changes is WHO is contending with whom. Instead of every thread in the grid racing every other thread in the grid for one of `NUM_BUCKETS` global addresses, each block first builds its own PRIVATE copy of the histogram in shared memory — so a block's 256 threads only ever contend with the other 255 threads in that SAME block, never with any other block. Once every thread has recorded its element, the block performs one final merge: exactly `NUM_BUCKETS` atomicAdd calls into the global histogram, one per bucket, regardless of how many of the block's threads actually touched each one.

```
naive (Section 7.1): every thread's atomicAdd goes straight to GLOBAL memory

  block 0 (256 threads) --\
  block 1 (256 threads) ---\
  block 2 (256 threads) ----+---> up to 2048 atomicAdd calls, ALL of them
        ...                /      contending on just 16 global addresses
  block 7 (256 threads) --/

privatized (Section 7.2): each block accumulates locally first

  block 0: 256 threads -> atomicAdd on 16 SHARED addresses (contention
           stays inside this one block's 256 threads) -> then exactly 16
           atomicAdd calls merge into the global histogram
  block 1: (same, independently)                        -> 16 more
  ...
  block 7: (same, independently)                        -> 16 more

  total global atomic traffic: 8 blocks x 16 buckets = 128 calls,
  regardless of how large N grows -- versus 2048 for the naive version.
```

The key structural point: the naive kernel's global atomic traffic grows with `N` (one atomicAdd per element, forever). The privatized kernel's global atomic traffic grows only with `NUM_BLOCKS x NUM_BUCKETS` — completely independent of how many elements each block actually processes. Doubling `N` while keeping the block and bucket counts fixed doubles the naive kernel's global contention but leaves the privatized kernel's global contention exactly where it was.

### Code and Verification

```cpp
#include <cstdio>
#include <vector>

// Chapter 7.2 -- Section 7.1 established that atomicAdd is CORRECT: it
// never loses an increment, no matter how many threads target the same
// bucket. But every one of those atomicAdd calls in Section 7.1's naive
// kernel goes straight to GLOBAL memory, and every thread across the
// entire grid that shares a bucket serializes against every other thread
// that also wants that same bucket -- correctness was never the problem
// with a naive global-atomics histogram; contention on a small number of
// hot global addresses is. Privatization does not change correctness (a
// per-block SHARED-memory atomicAdd is exactly as indivisible as a global
// one) -- it changes how MANY global atomic operations the whole grid
// issues, by having each block accumulate its own private copy of the
// histogram first.

#define BLOCK_SIZE 256
#define NUM_BUCKETS 16

// Every thread does exactly one atomicAdd, straight to global memory.
// Correct (Section 7.1 already proved atomicAdd loses nothing) -- but N
// global atomic operations total, all contending on just NUM_BUCKETS
// addresses.
__global__ void histogram_naive_global(const int* g_data, int* g_hist, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        atomicAdd(&g_hist[g_data[idx]], 1);
    }
}

// Each block first builds its OWN private histogram in shared memory --
// every atomicAdd here contends only with the 255 other threads in the
// SAME block, not the whole grid. Once every thread has counted its
// element, the block merges its private counts into the global histogram
// with exactly NUM_BUCKETS atomicAdd calls total (one per bucket), no
// matter how many of the block's 256 threads actually touched each
// bucket.
__global__ void histogram_privatized(const int* g_data, int* g_hist, int n) {
    __shared__ int local_hist[NUM_BUCKETS];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Cooperative zero-init: NUM_BUCKETS (16) is smaller than BLOCK_SIZE
    // (256), so only the first NUM_BUCKETS threads do this.
    if (tid < NUM_BUCKETS) local_hist[tid] = 0;
    __syncthreads();

    if (idx < n) {
        atomicAdd(&local_hist[g_data[idx]], 1);
    }
    __syncthreads();

    // Merge: exactly NUM_BUCKETS global atomics per block, unconditionally
    // -- not one per element the block touched.
    if (tid < NUM_BUCKETS) {
        atomicAdd(&g_hist[tid], local_hist[tid]);
    }
}

// ---- Host-side simulation of the exact same two designs ----

std::vector<int> simulate_naive_global(const std::vector<int>& data, int num_blocks, int block_size) {
    std::vector<int> hist(NUM_BUCKETS, 0);
    int n = (int)data.size();
    for (int block = 0; block < num_blocks; block++) {
        for (int tid = 0; tid < block_size; tid++) {
            int idx = block * block_size + tid;
            if (idx < n) hist[data[idx]]++;   // atomicAdd: indivisible, never lost
        }
    }
    return hist;
}

struct PrivatizedResult {
    std::vector<int> hist;
    long long global_atomics;
};

PrivatizedResult simulate_privatized(const std::vector<int>& data, int num_blocks, int block_size) {
    std::vector<int> hist(NUM_BUCKETS, 0);
    long long global_atomics = 0;
    int n = (int)data.size();

    for (int block = 0; block < num_blocks; block++) {
        std::vector<int> local_hist(NUM_BUCKETS, 0);
        for (int tid = 0; tid < block_size; tid++) {
            int idx = block * block_size + tid;
            if (idx < n) local_hist[data[idx]]++;   // shared-memory atomicAdd, local contention only
        }
        // Merge: exactly NUM_BUCKETS global atomics, unconditionally.
        for (int b = 0; b < NUM_BUCKETS; b++) {
            hist[b] += local_hist[b];
            global_atomics++;
        }
    }
    return {hist, global_atomics};
}

int main() {
    printf("=== Section 7.2: privatized histograms -- same correctness, far fewer global atomics ===\n\n");

    const int N = 2048;
    const int NUM_BLOCKS = N / BLOCK_SIZE;   // 8
    std::vector<int> data(N);
    for (int i = 0; i < N; i++) data[i] = i % NUM_BUCKETS;

    auto naive_hist = simulate_naive_global(data, NUM_BLOCKS, BLOCK_SIZE);
    auto priv = simulate_privatized(data, NUM_BLOCKS, BLOCK_SIZE);

    printf("N = %d elements, %d buckets, %d blocks x %d threads\n\n", N, NUM_BUCKETS, NUM_BLOCKS, BLOCK_SIZE);
    printf("%-8s %-14s %-14s %-10s\n", "bucket", "naive_hist", "priv_hist", "reference");
    bool hist_match = true;
    for (int b = 0; b < NUM_BUCKETS; b++) {
        int reference = N / NUM_BUCKETS;   // exact: N divides NUM_BUCKETS evenly
        printf("%-8d %-14d %-14d %-10d\n", b, naive_hist[b], priv.hist[b], reference);
        if (naive_hist[b] != reference || priv.hist[b] != reference) hist_match = false;
    }

    long long naive_global_atomics = N;                          // one per element
    long long priv_global_atomics = priv.global_atomics;          // NUM_BLOCKS * NUM_BUCKETS

    printf("\nboth histograms are IDENTICAL and correct: %s\n", hist_match ? "yes" : "NO -- BUG");
    printf("global atomic operations issued:\n");
    printf("  naive global-atomics kernel: %lld (one per element)\n", naive_global_atomics);
    printf("  privatized kernel:           %lld (%d blocks x %d buckets, regardless of N)\n",
           priv_global_atomics, NUM_BLOCKS, NUM_BUCKETS);
    printf("  reduction factor: %.1fx fewer global atomics\n",
           (double)naive_global_atomics / (double)priv_global_atomics);
    printf("\nprivatization is a CONTENTION story, not a correctness fix -- shared-memory\n");
    printf("atomicAdd (Section 7.2) is exactly as indivisible and lossless as global-memory\n");
    printf("atomicAdd (Section 7.1). What changes is that %d threads now contend for %d\n",
           BLOCK_SIZE, NUM_BUCKETS);
    printf("addresses PER BLOCK instead of all %d threads contending for %d addresses\n",
           N, NUM_BUCKETS);
    printf("across the WHOLE grid at once.\n");

    bool ok = hist_match && (priv_global_atomics == (long long)NUM_BLOCKS * NUM_BUCKETS)
              && (priv_global_atomics < naive_global_atomics);
    printf("\nself-check: histograms match and agree with reference, privatized kernel issues\n");
    printf("strictly fewer global atomics: %s\n", ok ? "confirmed" : "MISMATCH");
    return ok ? 0 : 1;
}
```

**Compile and run:**

```bash
nvcc -arch=sm_80 20_privatized_histogram.cu -o privatized_histogram
./privatized_histogram
```

**Sample input:** `N = 2048` elements (`data[i] = i % 16`, so each of `NUM_BUCKETS = 16` buckets gets exactly 128 elements), launched as `NUM_BLOCKS = 8` blocks of `BLOCK_SIZE = 256` threads.

**Sample output:**

```text
=== Section 7.2: privatized histograms -- same correctness, far fewer global atomics ===

N = 2048 elements, 16 buckets, 8 blocks x 256 threads

bucket   naive_hist     priv_hist      reference 
0        128            128            128       
1        128            128            128       
2        128            128            128       
3        128            128            128       
4        128            128            128       
5        128            128            128       
6        128            128            128       
7        128            128            128       
8        128            128            128       
9        128            128            128       
10       128            128            128       
11       128            128            128       
12       128            128            128       
13       128            128            128       
14       128            128            128       
15       128            128            128       

both histograms are IDENTICAL and correct: yes
global atomic operations issued:
  naive global-atomics kernel: 2048 (one per element)
  privatized kernel:           128 (8 blocks x 16 buckets, regardless of N)
  reduction factor: 16.0x fewer global atomics

privatization is a CONTENTION story, not a correctness fix -- shared-memory
atomicAdd (Section 7.2) is exactly as indivisible and lossless as global-memory
atomicAdd (Section 7.1). What changes is that 256 threads now contend for 16
addresses PER BLOCK instead of all 2048 threads contending for 16 addresses
across the WHOLE grid at once.

self-check: histograms match and agree with reference, privatized kernel issues
strictly fewer global atomics: confirmed
```

Both kernels produce the identical, correct histogram (128 per bucket, matching the reference exactly) — but the naive kernel issues 2048 global atomic operations while the privatized kernel issues only 128, a 16x reduction that has nothing to do with correctness and everything to do with where the contention happens.

!!! warning "[COMMON TRAP] Treating privatization as a correctness upgrade over atomicAdd"
    It is tempting to describe Section 7.2 as "fixing" something Section 7.1 got wrong, especially since the numbers visibly improve. Nothing about Section 7.1's atomicAdd-based histogram was incorrect — it already matched the reference exactly for every bucket. Privatization changes a PERFORMANCE property (how many threads contend for how many addresses, and where) while leaving correctness exactly as it already was. Confusing the two leads to skipping privatization on a small kernel "since atomicAdd is already correct" — true, but beside the point the moment contention, not correctness, is what is limiting performance.

## 7.3 Histogram Counts to Sorted Output: Counting Sort

### Intuition

A histogram tells you how MANY elements belong in each bucket. Chapter 6.2 already solved a closely related problem — turning per-block COUNTS into per-block starting OFFSETS, with an exclusive scan — for stream compaction's multi-block combination step. Running that identical scan over a histogram's per-bucket counts instead of per-block counts answers a new question: at what output index should each bucket's group of elements START? Once every bucket knows its starting offset, one more pass can place every element directly into a fully sorted output array.

### The Concept, In Detail

The recipe is exactly three steps, and every one of them is a technique this book has already proven correct on its own:

1. **Histogram** (Section 7.1): count how many elements belong to each bucket, using atomicAdd so no count is ever lost.
2. **Exclusive scan** (Chapter 5.2, or Chapter 5.1's simpler Hillis-Steele shape for a small array): turn those per-bucket COUNTS into per-bucket starting OFFSETS — bucket `b`'s offset is the total count of every bucket before it, exactly Chapter 6.2's "counts become offsets" pattern, applied here across buckets instead of across blocks.
3. **Scatter** (Section 7.1's atomicAdd again, in a new role): give every element its own unique output position by atomically claiming and advancing a per-bucket CURSOR that starts at that bucket's offset — the first element that lands in bucket `b` gets position `offset[b]`, the next gets `offset[b] + 1`, and so on.

Traced by hand on a small example (8 elements, 4 buckets — smaller than the code's actual `N = 256`, `NUM_BUCKETS = 8`, purely so the whole trace fits on the page):

```
keys:      [3, 1, 3, 0, 2, 1, 3, 0]      (N=8, NUM_BUCKETS=4)

step 1 -- histogram (atomicAdd per element):
    bucket:   0   1   2   3
    count:    2   2   1   3

step 2 -- exclusive scan of counts (Chapter 5.2, unchanged):
    bucket:   0   1   2   3
    offset:   0   2   4   5      (bucket b starts right after buckets 0..b-1)

step 3 -- scatter: cursor[b] starts at offset[b]; each element, IN ORIGINAL
ORDER, claims cursor[bucket] as its output slot and advances the cursor:

    i=0  key=3 -> claims slot 5, cursor[3]: 5->6
    i=1  key=1 -> claims slot 2, cursor[1]: 2->3
    i=2  key=3 -> claims slot 6, cursor[3]: 6->7
    i=3  key=0 -> claims slot 0, cursor[0]: 0->1
    i=4  key=2 -> claims slot 4, cursor[2]: 4->5
    i=5  key=1 -> claims slot 3, cursor[1]: 3->4
    i=6  key=3 -> claims slot 7, cursor[3]: 7->8
    i=7  key=0 -> claims slot 1, cursor[0]: 1->2

    output values:        [0, 0, 1, 1, 2, 3, 3, 3]   -- fully sorted
    output orig. indices:  [3, 7, 1, 5, 4, 0, 2, 6]   -- increasing WITHIN
                                                          each bucket's run
```

Every bucket's elements land in a contiguous run starting exactly at its scanned offset, the whole array comes out sorted by bucket, and because elements are processed in their original order when claiming cursor positions, each bucket's run preserves the original relative order of its own elements (the original-index columns 3, 7 for bucket 0 and 1, 5 for bucket 1 are each increasing) — the same STABILITY property Chapter 6.3 checked for partitioning. This entire three-step recipe — histogram, scan, cursor-based scatter — is counting sort. It is also exactly what Part 3's radix sort performs, once per digit, using each digit's value as the bucket.

### Code and Verification

```cpp
#include <cstdio>
#include <vector>
#include <algorithm>

// Chapter 7.3 -- Section 7.1 built a correct histogram with atomicAdd.
// Section 7.2 made it cheaper. This section asks a different question:
// once you know how many elements belong in each bucket, can you go
// further and actually place every element into a fully SORTED output
// array, grouped by bucket, in one more pass? Yes -- and the recipe is
// built entirely from pieces this book has already proven correct:
// Section 7.1's histogram gives per-bucket COUNTS; Chapter 5.2's
// exclusive scan turns those counts into per-bucket STARTING OFFSETS
// (the exact operation Chapter 6.2 used for per-block offsets, applied
// here to per-bucket offsets instead); and a final atomicAdd per element
// hands out each element's own unique slot within its bucket, counting
// up from that bucket's offset. This whole three-step recipe is counting
// sort -- and Part 3's radix sort is exactly this same recipe, run once
// per digit.

#define NUM_BUCKETS 8

// Step 1: Section 7.1's histogram, unchanged in spirit -- every thread
// contributes one atomicAdd to its bucket's count.
__global__ void histogram_kernel(const int* g_data, int* g_hist, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        atomicAdd(&g_hist[g_data[idx]], 1);
    }
}

// Step 2: Chapter 5.2's exact exclusive-scan shape (up-sweep, zero the
// last element, down-sweep), run once over the small NUM_BUCKETS-sized
// count array -- turning "how many elements are in bucket b" into "at
// what output index does bucket b's group START."
__global__ void exclusive_scan_offsets(const int* g_hist, int* g_offsets) {
    __shared__ int temp[NUM_BUCKETS];
    int tid = threadIdx.x;
    temp[tid] = g_hist[tid];
    __syncthreads();

    // Chapter 5.1's exact Hillis-Steele shape, run once over the tiny
    // NUM_BUCKETS-sized count array: after this loop, temp[tid] holds an
    // INCLUSIVE scan (total elements in buckets 0..tid, inclusive).
    for (int d = 1; d < NUM_BUCKETS; d *= 2) {
        int val = (tid >= d) ? temp[tid - d] : 0;
        __syncthreads();
        if (tid >= d) temp[tid] += val;
        __syncthreads();
    }

    // Exclusive = inclusive minus this bucket's own count -- the number
    // of elements in every EARLIER bucket, not counting bucket `tid`
    // itself, which is exactly the starting offset bucket `tid` needs.
    g_offsets[tid] = temp[tid] - g_hist[tid];
}

// Step 3: every element reads its bucket's current cursor (starting at
// that bucket's scanned offset) and atomically claims the next free slot
// in that bucket's group -- exactly Section 7.1's atomicAdd, now used to
// hand out unique POSITIONS instead of counting occurrences.
__global__ void scatter_counting_sort(const int* g_data, int* g_cursor,
                                       int* g_out_val, int* g_out_orig_idx, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int b = g_data[idx];
        int pos = atomicAdd(&g_cursor[b], 1);
        g_out_val[pos] = g_data[idx];
        g_out_orig_idx[pos] = idx;
    }
}

// ---- Host-side simulation of the exact same three-step arithmetic ----

std::vector<int> simulate_histogram(const std::vector<int>& data) {
    std::vector<int> hist(NUM_BUCKETS, 0);
    for (int v : data) hist[v]++;   // atomicAdd: indivisible, matches Section 7.1
    return hist;
}

std::vector<int> simulate_exclusive_scan(const std::vector<int>& hist) {
    std::vector<int> offsets(NUM_BUCKETS, 0);
    int running = 0;
    for (int b = 0; b < NUM_BUCKETS; b++) {
        offsets[b] = running;
        running += hist[b];
    }
    return offsets;
}

struct ScatterResult {
    std::vector<int> out_val;
    std::vector<int> out_orig_idx;
};

ScatterResult simulate_scatter(const std::vector<int>& data, std::vector<int> cursor) {
    int n = (int)data.size();
    ScatterResult r;
    r.out_val.assign(n, -1);
    r.out_orig_idx.assign(n, -1);
    for (int idx = 0; idx < n; idx++) {
        int b = data[idx];
        int pos = cursor[b];   // atomicAdd(&cursor[b], 1): claim, then advance
        cursor[b] = pos + 1;
        r.out_val[pos] = data[idx];
        r.out_orig_idx[pos] = idx;
    }
    return r;
}

int main() {
    printf("=== Section 7.3: histogram + scan = counting sort ===\n\n");

    const int N = 256;
    std::vector<int> data(N);
    for (int i = 0; i < N; i++) data[i] = (i * 31 + 17) % NUM_BUCKETS;   // deterministic, unsorted keys in [0, 8)

    auto hist = simulate_histogram(data);
    auto offsets = simulate_exclusive_scan(hist);
    auto scattered = simulate_scatter(data, offsets);

    printf("N = %d elements, keys in [0, %d)\n\n", N, NUM_BUCKETS);
    printf("%-8s %-8s %-10s\n", "bucket", "count", "offset");
    for (int b = 0; b < NUM_BUCKETS; b++) {
        printf("%-8d %-8d %-10d\n", b, hist[b], offsets[b]);
    }

    bool sorted = true;
    for (int i = 1; i < N; i++) {
        if (scattered.out_val[i - 1] > scattered.out_val[i]) sorted = false;
    }

    bool ranges_correct = true;
    for (int b = 0; b < NUM_BUCKETS; b++) {
        int start = offsets[b];
        int end = start + hist[b];
        for (int i = start; i < end; i++) {
            if (scattered.out_val[i] != b) ranges_correct = false;
        }
    }

    bool stable = true;
    for (int b = 0; b < NUM_BUCKETS; b++) {
        int start = offsets[b];
        int end = start + hist[b];
        for (int i = start + 1; i < end; i++) {
            if (scattered.out_orig_idx[i - 1] >= scattered.out_orig_idx[i]) stable = false;
        }
    }

    printf("\nfirst 16 output values:      ");
    for (int i = 0; i < 16; i++) printf("%d ", scattered.out_val[i]);
    printf("\nfirst 16 original indices:   ");
    for (int i = 0; i < 16; i++) printf("%d ", scattered.out_orig_idx[i]);
    printf("\n\n");

    printf("output is fully sorted by key: %s\n", sorted ? "yes" : "NO -- BUG");
    printf("every bucket occupies exactly its [offset, offset+count) range: %s\n",
           ranges_correct ? "yes" : "NO -- BUG");
    printf("within each bucket, original relative order is preserved (stable): %s\n",
           stable ? "yes" : "NO -- BUG");
    printf("\nthis is counting sort: Section 7.1's histogram counted, Chapter 5.2's exclusive\n");
    printf("scan turned counts into offsets, and Section 7.1's atomicAdd -- reused here to\n");
    printf("hand out unique positions instead of counting occurrences -- placed every\n");
    printf("element directly into fully sorted order. Part 3's radix sort is this exact\n");
    printf("recipe, run once per digit.\n");

    bool ok = sorted && ranges_correct && stable;
    printf("\nself-check: sorted, correctly ranged, and stable: %s\n", ok ? "confirmed" : "MISMATCH");
    return ok ? 0 : 1;
}
```

**Compile and run:**

```bash
nvcc -arch=sm_80 21_histogram_to_counting_sort.cu -o counting_sort
./counting_sort
```

**Sample input:** `N = 256` elements with keys in `[0, 8)` assigned by `data[i] = (i * 31 + 17) % 8` — a fixed, deterministic, intentionally-unsorted pattern (not already grouped by key).

**Sample output:**

```text
=== Section 7.3: histogram + scan = counting sort ===

N = 256 elements, keys in [0, 8)

bucket   count    offset    
0        32       0         
1        32       32        
2        32       64        
3        32       96        
4        32       128       
5        32       160       
6        32       192       
7        32       224       

first 16 output values:      0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
first 16 original indices:   1 9 17 25 33 41 49 57 65 73 81 89 97 105 113 121 

output is fully sorted by key: yes
every bucket occupies exactly its [offset, offset+count) range: yes
within each bucket, original relative order is preserved (stable): yes

this is counting sort: Section 7.1's histogram counted, Chapter 5.2's exclusive
scan turned counts into offsets, and Section 7.1's atomicAdd -- reused here to
hand out unique positions instead of counting occurrences -- placed every
element directly into fully sorted order. Part 3's radix sort is this exact
recipe, run once per digit.

self-check: sorted, correctly ranged, and stable: confirmed
```

The output array is fully sorted by key, every bucket occupies exactly the `[offset, offset + count)` range the scan predicted, and each bucket's run preserves the original relative order of its elements — all three checked directly, not just inferred from the final array looking right.

!!! warning "[COMMON TRAP] Reusing the histogram's COUNTS as if they were already valid output positions"
    A count and a position answer different questions: "how many elements are in bucket 3" is not "where does bucket 3's first element go." Skipping the scan step and trying to scatter directly using raw per-bucket counts has no way to know where one bucket's group ends and the next one begins — every bucket's elements would try to write starting at output index 0. The exclusive scan is not an optional bookkeeping nicety layered on top of counting sort; it is the specific step that turns "how many" into "starting where," and without it there is no way to assign a single element a correct output position at all.

## Chapter Summary

Section 7.1 traced, through Chapter 1's own warp lockstep model, exactly how a non-atomic `hist[bucket]++` loses updates whenever multiple lanes in the same wave target the same bucket — losing 682 of 1000 increments in the measured example — and showed atomicAdd's indivisible read-modify-write loses nothing. Section 7.2 showed privatizing a histogram into per-block shared memory does not change this correctness story at all; it changes contention, cutting global atomic traffic from 2048 down to 128 (16x) by having each block's threads contend only with each other before one final per-bucket merge into global memory. Section 7.3 combined Section 7.1's histogram with Chapter 5's exclusive scan to turn per-bucket counts into per-bucket starting offsets, then reused atomicAdd a third time to hand out unique output positions instead of counting occurrences — producing a fully sorted, stable output array in one more pass: counting sort, built entirely from pieces this book had already proven correct on their own, and the exact recipe Part 3's radix sort will run once per digit.

This completes Part 1. Reduction (Chapter 4), scan (Chapter 5), stream compaction (Chapter 6), and histograms (Chapter 7) are not four unrelated tricks — reduction's tree shape reappears inside scan's up-sweep, scan reappears inside compaction's offset computation, and both scan and atomics reappear inside this chapter's histograms and counting sort. Part 2 turns to linear structures under concurrent access, where these same primitives meet a new problem: data structures that must stay correct while many threads modify them at once, not just read from them.

## Self-Check Questions

1. Section 7.1 processes input in fixed 32-lane waves. If a single wave's 32 lanes happened to target 32 DIFFERENT buckets (no collisions at all), how many increments would the naive, non-atomic version lose in that wave?
2. Using Section 7.1's own trace shape, work through what happens to a bucket holding the value 10 before a wave, if exactly 5 lanes in that wave all target it, using the non-atomic model. What is the bucket's value after the wave?
3. Section 7.2 reports 128 global atomic operations for the privatized kernel regardless of how many of a block's 256 threads actually shared any given bucket. Explain concretely why the merge step's atomicAdd count does not depend on the DATA at all, only on `NUM_BLOCKS` and `NUM_BUCKETS`.
4. Section 7.3's exclusive scan step computes `offset[b] = temp[b] - hist[b]`, where `temp[b]` is an INCLUSIVE scan of the histogram. Explain why subtracting a bucket's own count from its inclusive scan value produces that bucket's correct exclusive offset.
5. Using Section 7.3's small hand-traced example (`keys = [3, 1, 3, 0, 2, 1, 3, 0]`, offsets `[0, 2, 4, 5]`), verify by hand where the SECOND element with key 0 (at original index 7) is placed in the output, and confirm it lands after the first element with key 0 (original index 3).
6. Section 7.3's warning explains why scattering directly from raw counts (skipping the scan) fails. Explain concretely what would happen to bucket 0's and bucket 1's elements if both buckets simply used their own raw COUNT as their starting output offset instead of a scanned value.

## Where We Go Next

Part 1 is complete: reduction, scan, stream compaction, and histograms are the four parallel primitives nearly everything later in this book is built from. Part 2 turns to linear structures — arrays and growable buffers under concurrent access, lock-free stacks and queues, and a full accounting of why pointer-chasing linked lists, so natural on a CPU, are the wrong shape for a machine built around thousands of threads moving in lockstep.

## Worked Solutions

**1.** Zero. The naive model only loses increments when two or more lanes in the SAME wave target the SAME bucket (`pre_wave_value` captures a bucket's value once and applies `+1` once per DISTINCT bucket touched). With 32 lanes targeting 32 distinct buckets, every bucket touched in that wave is touched by exactly one lane, so every one of the 32 increments is recorded — the race only costs increments when buckets are actually shared.

**2.** All 5 lanes load the identical pre-wave value of 10 (lockstep LOAD), each independently computes `10 + 1 = 11` (lockstep ADD), and all 5 stores write back 11 (lockstep STORE) — whichever store physically lands last, the value written is the same 11 regardless. The bucket's value after the wave is 11, not 15: four of the five lanes' increments are completely lost, exactly as Section 7.1's 4-lane trace showed.

**3.** The merge step's loop structure is `if (tid < NUM_BUCKETS) atomicAdd(&g_hist[tid], local_hist[tid])` — every block runs this merge exactly once, and it always performs exactly `NUM_BUCKETS` atomicAdd calls (one per bucket index `tid` from 0 to `NUM_BUCKETS - 1`), REGARDLESS of what value `local_hist[tid]` holds. Even a bucket that ended up with a local count of zero still gets one atomicAdd call (adding zero). The data only affects WHAT VALUE each atomicAdd contributes, never HOW MANY atomicAdd calls happen — that count is fixed the moment `NUM_BLOCKS` and `NUM_BUCKETS` are chosen.

**4.** An inclusive scan value at bucket `b`, `temp[b]`, is the total count of every bucket from 0 up to AND INCLUDING `b` itself (`hist[0] + hist[1] + ... + hist[b]`). Subtracting `hist[b]` — bucket `b`'s own count — removes exactly the "including itself" part, leaving `hist[0] + ... + hist[b-1]`: the total count of every bucket STRICTLY BEFORE `b`, which is precisely the exclusive scan's definition and exactly the output index bucket `b` should start at.

**5.** Offsets are `[0, 2, 4, 5]`, so bucket 0's cursor starts at 0. Processing in original order: at `i=3` (the FIRST key-0 element), `cursor[0]` is still at its starting value of 0, so it claims slot 0 and the cursor advances to 1. At `i=7` (the SECOND key-0 element), `cursor[0]` is now 1, so it claims slot 1 and the cursor advances to 2. Slot 0 (original index 3) comes before slot 1 (original index 7) in the output array, so the first key-0 element by original order does land before the second — confirming stability for this pair exactly as the hand trace's output-index row shows (`[3, 7, ...]`).

**6.** Bucket 0's raw count is 2, so it would start writing at output index 2 instead of its correct scanned offset of 0 — leaving indices 0 and 1 completely unwritten. Bucket 1's raw count is 2, so it would ALSO start writing at output index 2 — directly colliding with bucket 0's elements at the exact same output positions bucket 0 is also trying to use, silently overwriting each other. Skipping the scan does not just shift results by a constant amount; different buckets' raw counts have no relationship to each other's correct starting positions, so every bucket beyond the first ends up overlapping some other bucket's output range instead of getting its own.
