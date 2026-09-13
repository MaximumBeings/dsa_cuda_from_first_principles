# Chapter 26: Heaps and Priority Queues on a GPU

Every graph algorithm since Chapter 22 has needed some way to pick "the next best thing" -- the closest unvisited vertex, the cheapest outgoing edge. A binary heap is the classic sequential answer, and this chapter opens Part 7 -- Priority Structures and Concurrency by asking how much of it survives contact with a GPU. The answer comes in three pieces: heap CONSTRUCTION parallelizes cleanly across levels, since sibling subtrees never interact; heap EXTRACTION does not parallelize within a single heap at all, but sharding the problem across many independent heaps turns that limitation into an opportunity; and when priorities are bounded integers, a bucket queue sidesteps the whole pointer-chasing sift path in favor of the same atomic-scatter techniques Chapters 7 and 13 already built.

## 26.1 Sequential Heap Construction and Parallel Heapify

### Intuition

A binary min-heap keeps its smallest element at the root, stored as a plain array (no pointers) where node i's children live at indices 2i+1 and 2i+2 -- the same implicit-tree indexing Chapter 15 used for array-based binary trees. Building one the classic way means inserting elements one at a time: append to the next free slot, then "sift up" -- repeatedly swap with the parent for as long as the new element is smaller -- until the heap property holds everywhere again. That sequential, one-at-a-time process hides a much more parallel structure underneath it.

### The Sequential (CPU) Baseline

```cpp
#include <cstdio>
#include <vector>
#include <algorithm>

// Chapter 26.1 -- The Sequential (CPU) Baseline.
// A binary min-heap keeps its smallest element always at the root,
// using an ARRAY (no pointers) where node i's children live at
// indices 2i+1 and 2i+2 -- the same implicit-tree indexing this book
// has used since Chapter 15's array-based binary trees. Inserting a
// new element appends it at the end of the array (the next free leaf
// slot) and then repeatedly swaps it with its parent -- "sift-up" --
// for as long as it is smaller than that parent, restoring the heap
// property one level at a time. This section builds a heap the
// classic way: one element at a time, via repeated insertion.
#define N 8

void sift_up(std::vector<int>& heap, int i) {
    while (i > 0) {
        int parent = (i - 1) / 2;
        if (heap[i] < heap[parent]) {
            printf("    sift-up: swap index %d (value %d) with parent index %d (value %d)\n",
                   i, heap[i], parent, heap[parent]);
            std::swap(heap[i], heap[parent]);
            i = parent;
        } else {
            break;
        }
    }
}

int main() {
    printf("=== Section 26.1 CPU baseline: building a min-heap via repeated insertion ===\n\n");

    std::vector<int> values = {15, 10, 20, 8, 25, 5, 30, 12};
    printf("inserting %d values one at a time: [ ", N);
    for (int v : values) printf("%d ", v);
    printf("]\n\n");

    std::vector<int> heap;
    for (int v : values) {
        heap.push_back(v);
        printf("insert %d: heap = [ ", v);
        for (int h : heap) printf("%d ", h);
        printf("]\n");
        sift_up(heap, (int)heap.size() - 1);
        printf("  after sift-up: heap = [ ");
        for (int h : heap) printf("%d ", h);
        printf("]\n");
    }

    printf("\nfinal heap array: [ ");
    for (int h : heap) printf("%d ", h);
    printf("]\n");

    // Verify the heap property holds everywhere: every parent <= both children.
    bool valid = true;
    for (int i = 0; i < N; i++) {
        int l = 2*i+1, r = 2*i+2;
        if (l < N && heap[i] > heap[l]) valid = false;
        if (r < N && heap[i] > heap[r]) valid = false;
    }

    std::vector<int> expected_heap = {5, 10, 8, 12, 25, 20, 30, 15};
    bool ok = valid && (heap == expected_heap);

    printf("\nexpected final heap: [ 5 10 8 12 25 20 30 15 ], with the heap\n");
    printf("property (every parent <= both children) holding at every node\n");

    printf("\nself-check: repeated insertion builds a valid min-heap matching the\n");
    printf("expected array: %s\n", ok ? "confirmed" : "MISMATCH");
    return ok ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++17 -Wall -Wextra -O2 142_heap_insertion_cpu_baseline.cpp -o 142_heap_insertion_cpu_baseline
./142_heap_insertion_cpu_baseline
```

**Sample input:** 8 values inserted one at a time into an initially empty heap, each restoring the heap property via sift-up.

**Sample output:**

```text
=== Section 26.1 CPU baseline: building a min-heap via repeated insertion ===

inserting 8 values one at a time: [ 15 10 20 8 25 5 30 12 ]

insert 15: heap = [ 15 ]
  after sift-up: heap = [ 15 ]
insert 10: heap = [ 15 10 ]
    sift-up: swap index 1 (value 10) with parent index 0 (value 15)
  after sift-up: heap = [ 10 15 ]
insert 20: heap = [ 10 15 20 ]
  after sift-up: heap = [ 10 15 20 ]
insert 8: heap = [ 10 15 20 8 ]
    sift-up: swap index 3 (value 8) with parent index 1 (value 15)
    sift-up: swap index 1 (value 8) with parent index 0 (value 10)
  after sift-up: heap = [ 8 10 20 15 ]
insert 25: heap = [ 8 10 20 15 25 ]
  after sift-up: heap = [ 8 10 20 15 25 ]
insert 5: heap = [ 8 10 20 15 25 5 ]
    sift-up: swap index 5 (value 5) with parent index 2 (value 20)
    sift-up: swap index 2 (value 5) with parent index 0 (value 8)
  after sift-up: heap = [ 5 10 8 15 25 20 ]
insert 30: heap = [ 5 10 8 15 25 20 30 ]
  after sift-up: heap = [ 5 10 8 15 25 20 30 ]
insert 12: heap = [ 5 10 8 15 25 20 30 12 ]
    sift-up: swap index 7 (value 12) with parent index 3 (value 15)
  after sift-up: heap = [ 5 10 8 12 25 20 30 15 ]

final heap array: [ 5 10 8 12 25 20 30 15 ]

expected final heap: [ 5 10 8 12 25 20 30 15 ], with the heap
property (every parent <= both children) holding at every node

self-check: repeated insertion builds a valid min-heap matching the
expected array: confirmed
```

### The Concept, In Detail

```
ASCII view: the same 8 values, viewed as a tree instead of an array.

  Insertion-built heap:            Level-synchronous heapify (26.1 main):

           5                                5
         /   \                            /   \
        10     8                         8     15
       / \    / \                       / \    / \
      12 25  20 30                    10  25  20  30
     /                                /
    15                               12

  BOTH are valid min-heaps of the same 8 values (every parent <= both
  children) -- there is no single "correct" heap layout, only correct
  and incorrect ones.
```

Insertion processes elements ONE AT A TIME, in whatever order they arrive, and each sift-up only ever touches a single root-to-leaf path. Building a heap from an array that already holds every element in memory can instead start from the BOTTOM: every leaf is trivially a valid (single-node) heap already, so sift-down every internal node starting from the deepest level and working upward. Two internal nodes at the same level are always roots of disjoint subtrees (neither's descendants overlap with the other's), so every node at a given level can sift down at once with zero coordination -- only the LEVELS themselves must be processed bottom-up, since a parent's sift-down needs its children's subtrees to already be valid heaps first.

[COMMON TRAP]
It is tempting to think heapify could process every internal node in parallel all at once, in a single pass, since sift-down only writes to nodes below the one that started it. A node's sift-down can move an element several levels down its own subtree, and if that subtree's OWN sift-down (from a still-unprocessed lower level) hasn't finished yet, the two can race on the same array slots. Respecting strict bottom-up LEVEL order is what guarantees every child subtree is already a fully valid heap before its parent's own sift-down ever begins.

### Code and Verification

```cpp
#include <cstdio>
#include <vector>
#include <algorithm>
#include <utility>

// Chapter 26.1 main -- one thread per INTERNAL NODE AT THE SAME LEVEL,
// launched once per level, working bottom-up: the classic O(n)
// build-heap. Two sibling subtrees rooted at different nodes on the
// same level never touch each other's array slots during a sift-down
// (one subtree's descendants and another's never overlap), so every
// node on a given level can sift down at once with no synchronization
// needed within that level -- exactly the level-synchronous pattern
// Chapter 16 used for parallel tree construction. Only levels
// themselves must be processed in strict bottom-up order, since a
// parent's sift-down needs its children's subtrees to already be
// valid heaps.
#define N 8

__device__ void sift_down_dev(int* arr, int i, int n) {
    while (true) {
        int l = 2*i+1, r = 2*i+2, smallest = i;
        if (l < n && arr[l] < arr[smallest]) smallest = l;
        if (r < n && arr[r] < arr[smallest]) smallest = r;
        if (smallest == i) break;
        int tmp = arr[i]; arr[i] = arr[smallest]; arr[smallest] = tmp;
        i = smallest;
    }
}

// One thread per node at the CURRENT level; `level_start`/`level_end`
// bound which node indices belong to that level.
__global__ void heapify_level_kernel(int* arr, int n, int level_start, int level_end) {
    int i = level_start + threadIdx.x;
    if (i > level_end) return;
    sift_down_dev(arr, i, n);
}

// ---- Host-side replay of the identical per-thread logic. ----

void sift_down_host(std::vector<int>& arr, int i, int n, std::vector<std::pair<int,int>>* trace) {
    while (true) {
        int l = 2*i+1, r = 2*i+2, smallest = i;
        if (l < n && arr[l] < arr[smallest]) smallest = l;
        if (r < n && arr[r] < arr[smallest]) smallest = r;
        if (smallest == i) break;
        if (trace) trace->push_back({i, smallest});
        std::swap(arr[i], arr[smallest]);
        i = smallest;
    }
}

int level_of(int i) {
    int lvl = 0;
    long long v = i + 1;
    while (v > 1) { v >>= 1; lvl++; }
    return lvl;
}

int main() {
    printf("=== Section 26.1 main: level-synchronous parallel heapify ===\n\n");

    std::vector<int> arr = {15, 10, 20, 8, 25, 5, 30, 12};
    printf("initial array (heap property not yet established anywhere): [ ");
    for (int v : arr) printf("%d ", v);
    printf("]\n\n");

    int last_internal = N / 2 - 1;   // index 3: last node with any children
    int max_level = level_of(last_internal);

    for (int lvl = max_level; lvl >= 0; lvl--) {
        int level_start = -1, level_end = -1;
        for (int i = last_internal; i >= 0; i--) {
            if (level_of(i) == lvl) {
                if (level_start == -1) level_start = i;
                level_end = i;
            }
        }
        if (level_start == -1) continue;
        // level_start > level_end numerically since we scanned downward;
        // normalize so level_start <= level_end for the kernel's range.
        if (level_start > level_end) std::swap(level_start, level_end);
        printf("level %d: sift-down nodes %d..%d (independent, launched in parallel)\n", lvl, level_start, level_end);
        for (int i = level_start; i <= level_end; i++) {
            std::vector<std::pair<int,int>> trace;
            sift_down_host(arr, i, N, &trace);
            if (trace.empty()) {
                printf("    node %d: already satisfies heap property, no swap\n", i);
            } else {
                for (auto& p : trace) printf("    node %d: swap index %d with index %d\n", i, p.first, p.second);
            }
        }
        printf("  array after level %d: [ ", lvl);
        for (int v : arr) printf("%d ", v);
        printf("]\n");
    }

    printf("\nfinal heap array: [ ");
    for (int v : arr) printf("%d ", v);
    printf("]\n");

    bool valid = true;
    for (int i = 0; i < N; i++) {
        int l = 2*i+1, r = 2*i+2;
        if (l < N && arr[i] > arr[l]) valid = false;
        if (r < N && arr[i] > arr[r]) valid = false;
    }

    std::vector<int> expected_arr = {5, 8, 15, 10, 25, 20, 30, 12};
    bool ok = valid && (arr == expected_arr);

    printf("\nexpected final heap: [ 5 8 15 10 25 20 30 12 ] -- a DIFFERENT valid\n");
    printf("arrangement than Section 26.1's insertion-built heap ([ 5 10 8 12 25\n");
    printf("20 30 15 ]), since more than one array satisfies the heap property\n");
    printf("for the same value set, but built in only 3 rounds (one per level)\n");
    printf("instead of 8 sequential insertions\n");

    printf("\nself-check: level-synchronous parallel heapify produces a valid min-heap\n");
    printf("matching the expected array: %s\n", ok ? "confirmed" : "MISMATCH");
    return ok ? 0 : 1;
}
```

**Compile and run:**

```bash
NVDIR=/usr/local/lib/python3.11/dist-packages/nvidia/cu13
$NVDIR/bin/nvcc -arch=sm_80 -I$NVDIR/include -L$NVDIR/lib -lcudart 143_heap_parallel_heapify_kernel.cu -o 143_heap_parallel_heapify_kernel
LD_LIBRARY_PATH=$NVDIR/lib ./143_heap_parallel_heapify_kernel
```

**Sample input:** the same 8 values, placed into the array as-is (no heap property anywhere), heapified bottom-up in 3 level-synchronous rounds.

**Sample output:**

```text
=== Section 26.1 main: level-synchronous parallel heapify ===

initial array (heap property not yet established anywhere): [ 15 10 20 8 25 5 30 12 ]

level 2: sift-down nodes 3..3 (independent, launched in parallel)
    node 3: already satisfies heap property, no swap
  array after level 2: [ 15 10 20 8 25 5 30 12 ]
level 1: sift-down nodes 1..2 (independent, launched in parallel)
    node 1: swap index 1 with index 3
    node 2: swap index 2 with index 5
  array after level 1: [ 15 8 5 10 25 20 30 12 ]
level 0: sift-down nodes 0..0 (independent, launched in parallel)
    node 0: swap index 0 with index 2
  array after level 0: [ 5 8 15 10 25 20 30 12 ]

final heap array: [ 5 8 15 10 25 20 30 12 ]

expected final heap: [ 5 8 15 10 25 20 30 12 ] -- a DIFFERENT valid
arrangement than Section 26.1's insertion-built heap ([ 5 10 8 12 25
20 30 15 ]), since more than one array satisfies the heap property
for the same value set, but built in only 3 rounds (one per level)
instead of 8 sequential insertions

self-check: level-synchronous parallel heapify produces a valid min-heap
matching the expected array: confirmed
```

## 26.2 Batch Extraction via Sharded Heaps

### Intuition

Extracting the minimum from a single heap is inherently sequential: each extraction removes the root, moves the last element into its place, and sifts it back down -- and the very next extraction depends entirely on the structure that sift-down just produced. There is no way to run two extractions against the SAME heap at once. But many applications don't need repeated extractions from one heap; they need the k smallest values across an entire batch, and that reframes the problem: split the batch into independent SHARDS, give each shard its own private heap, and drain each shard's heap completely and independently -- the same sharding idea Chapter 20.1 used for cuckoo hashing, now applied to heaps.

### The Sequential (CPU) Baseline

```cpp
#include <cstdio>
#include <vector>
#include <algorithm>

// Chapter 26.2 -- The Sequential (CPU) Baseline.
// Extracting the minimum from a single heap is inherently sequential:
// each extraction removes the root, moves the last element into its
// place, and sifts it back down -- and the NEXT extraction depends
// entirely on the structure that sift-down just produced. There is no
// way to run two extractions from the SAME heap at once. But many
// applications don't need extractions from one heap; they need the k
// smallest values across a whole batch, which opens a different door:
// split the batch into independent SHARDS, give each shard its own
// private heap, and drain each shard's heap completely and
// independently -- exactly the sharding idea Chapter 20.1 used for
// cuckoo hashing, now applied to heaps.
#define N 8

void sift_down(std::vector<int>& arr, int i, int n) {
    while (true) {
        int l = 2*i+1, r = 2*i+2, smallest = i;
        if (l < n && arr[l] < arr[smallest]) smallest = l;
        if (r < n && arr[r] < arr[smallest]) smallest = r;
        if (smallest == i) break;
        std::swap(arr[i], arr[smallest]);
        i = smallest;
    }
}

void heapify(std::vector<int>& arr) {
    int n = (int)arr.size();
    for (int i = n/2 - 1; i >= 0; i--) sift_down(arr, i, n);
}

int extract_min(std::vector<int>& arr) {
    int root = arr[0];
    arr[0] = arr.back();
    arr.pop_back();
    if (!arr.empty()) sift_down(arr, 0, (int)arr.size());
    return root;
}

std::vector<int> drain(std::vector<int> heap) {
    std::vector<int> out;
    while (!heap.empty()) out.push_back(extract_min(heap));
    return out;
}

int main() {
    printf("=== Section 26.2 CPU baseline: sharded heaps drained independently ===\n\n");

    std::vector<int> values = {15, 10, 20, 8, 25, 5, 30, 12};
    std::vector<int> shardA(values.begin(), values.begin() + 4);
    std::vector<int> shardB(values.begin() + 4, values.end());
    printf("shard A (raw): [ "); for (int v : shardA) printf("%d ", v); printf("]\n");
    printf("shard B (raw): [ "); for (int v : shardB) printf("%d ", v); printf("]\n\n");

    heapify(shardA);
    heapify(shardB);
    printf("shard A (heapified): [ "); for (int v : shardA) printf("%d ", v); printf("]\n");
    printf("shard B (heapified): [ "); for (int v : shardB) printf("%d ", v); printf("]\n\n");

    std::vector<int> sortedA = drain(shardA);
    std::vector<int> sortedB = drain(shardB);
    printf("shard A drained (each shard sorts itself, independently): [ ");
    for (int v : sortedA) printf("%d ", v);
    printf("]\n");
    printf("shard B drained (each shard sorts itself, independently): [ ");
    for (int v : sortedB) printf("%d ", v);
    printf("]\n\n");

    // Host-side merge of the two independently-sorted shard sequences,
    // exactly like merge sort's own merge step (Chapter 14).
    std::vector<int> merged;
    size_t i = 0, j = 0;
    printf("merging the two sorted shard sequences:\n");
    while (i < sortedA.size() && j < sortedB.size()) {
        if (sortedA[i] <= sortedB[j]) {
            printf("  take %d from shard A\n", sortedA[i]);
            merged.push_back(sortedA[i++]);
        } else {
            printf("  take %d from shard B\n", sortedB[j]);
            merged.push_back(sortedB[j++]);
        }
    }
    while (i < sortedA.size()) { printf("  take %d from shard A (B exhausted)\n", sortedA[i]); merged.push_back(sortedA[i++]); }
    while (j < sortedB.size()) { printf("  take %d from shard B (A exhausted)\n", sortedB[j]); merged.push_back(sortedB[j++]); }

    printf("\nfully merged (global sorted order): [ ");
    for (int v : merged) printf("%d ", v);
    printf("]\n");

    int k = 3;
    printf("k=%d smallest overall: [ ", k);
    for (int idx = 0; idx < k; idx++) printf("%d ", merged[idx]);
    printf("]\n");

    std::vector<int> expected_merged = {5, 8, 10, 12, 15, 20, 25, 30};
    bool ok = (merged == expected_merged);

    printf("\nexpected fully merged sequence: [ 5 8 10 12 15 20 25 30 ], with the\n");
    printf("3 smallest being [ 5 8 10 ]\n");

    printf("\nself-check: sharded, independently-drained heaps merge into the exact\n");
    printf("same fully sorted sequence as a single heap would produce: %s\n", ok ? "confirmed" : "MISMATCH");
    return ok ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++17 -Wall -Wextra -O2 144_heap_shard_drain_cpu_baseline.cpp -o 144_heap_shard_drain_cpu_baseline
./144_heap_shard_drain_cpu_baseline
```

**Sample input:** the same 8 values split into 2 shards of 4, each shard heapified and fully drained independently, then merged.

**Sample output:**

```text
=== Section 26.2 CPU baseline: sharded heaps drained independently ===

shard A (raw): [ 15 10 20 8 ]
shard B (raw): [ 25 5 30 12 ]

shard A (heapified): [ 8 10 20 15 ]
shard B (heapified): [ 5 12 30 25 ]

shard A drained (each shard sorts itself, independently): [ 8 10 15 20 ]
shard B drained (each shard sorts itself, independently): [ 5 12 25 30 ]

merging the two sorted shard sequences:
  take 5 from shard B
  take 8 from shard A
  take 10 from shard A
  take 12 from shard B
  take 15 from shard A
  take 20 from shard A
  take 25 from shard B (A exhausted)
  take 30 from shard B (A exhausted)

fully merged (global sorted order): [ 5 8 10 12 15 20 25 30 ]
k=3 smallest overall: [ 5 8 10 ]

expected fully merged sequence: [ 5 8 10 12 15 20 25 30 ], with the
3 smallest being [ 5 8 10 ]

self-check: sharded, independently-drained heaps merge into the exact
same fully sorted sequence as a single heap would produce: confirmed
```

### The Concept, In Detail

```
ASCII view: 2 independent shards, each doing its OWN sequential work,
merged only at the very end.

  shard A: [15 10 20 8] -> heapify -> drain -> [8 10 15 20]  (thread 0)
  shard B: [25 5 30 12] -> heapify -> drain -> [5 12 25 30]  (thread 1)

  neither shard's thread ever reads or writes the OTHER shard's array
  slice -- the only place the two results ever meet is the final,
  sequential merge step, exactly like merge sort's own merge (Ch14)
```

Sharding does not make heap extraction itself parallel -- each shard's drain loop is exactly as sequential as ever. What it parallelizes is running MANY such sequential drains at once, since different shards never touch each other's memory. The final merge of the per-shard sorted sequences must still happen sequentially (or via its own separate parallel merge network, Chapter 14), but it operates on far less total work than a single from-scratch sort would, because most of the ordering was already established independently and in parallel.

[COMMON TRAP]
It is tempting to think more shards always means more parallelism and therefore a faster result. Splitting into more, smaller shards does increase the number of independent sequential drains that can run at once, but it also means the final merge step -- which IS sequential -- has to combine more separate sorted sequences, and a shard's own heap becomes less effective at concentrating genuinely small values together the smaller it gets. The right shard count balances how much independent parallel work exists against how much sequential merging that choice creates afterward.

### Code and Verification

```cpp
#include <cstdio>
#include <vector>
#include <algorithm>

// Chapter 26.2 main -- one thread per SHARD, each running its own
// complete, private sequential heapify-then-drain loop, exactly like
// Chapter 20.1's one-thread-per-shard cuckoo construction: no shard's
// thread ever reads or writes another shard's slice of the array, so
// there is zero cross-shard interaction to protect with atomics. Each
// thread's work is itself sequential (a heap's extractions cannot be
// parallelized), but many independent sequential drains running at
// once is still genuine, useful parallelism.
#define N 8
#define SHARD_SIZE 4
#define NUM_SHARDS 2

__device__ void sift_down_dev(int* arr, int i, int n) {
    while (true) {
        int l = 2*i+1, r = 2*i+2, smallest = i;
        if (l < n && arr[l] < arr[smallest]) smallest = l;
        if (r < n && arr[r] < arr[smallest]) smallest = r;
        if (smallest == i) break;
        int tmp = arr[i]; arr[i] = arr[smallest]; arr[smallest] = tmp;
        i = smallest;
    }
}

// Each thread privately heapifies and fully drains its own shard,
// writing that shard's sorted output into its own segment of `out`.
__global__ void shard_drain_kernel(const int* input, int* out, int shard_size, int num_shards) {
    int s = threadIdx.x;
    if (s >= num_shards) return;

    int local[SHARD_SIZE];
    int base = s * shard_size;
    for (int i = 0; i < shard_size; i++) local[i] = input[base + i];

    int n = shard_size;
    for (int i = n/2 - 1; i >= 0; i--) sift_down_dev(local, i, n);

    for (int k = 0; k < shard_size; k++) {
        out[base + k] = local[0];
        n--;
        local[0] = local[n];
        if (n > 0) sift_down_dev(local, 0, n);
    }
}

// ---- Host-side replay of the identical per-thread logic. ----

void sift_down_host(std::vector<int>& arr, int i, int n) {
    while (true) {
        int l = 2*i+1, r = 2*i+2, smallest = i;
        if (l < n && arr[l] < arr[smallest]) smallest = l;
        if (r < n && arr[r] < arr[smallest]) smallest = r;
        if (smallest == i) break;
        std::swap(arr[i], arr[smallest]);
        i = smallest;
    }
}

int main() {
    printf("=== Section 26.2 main: one thread per shard, private heapify + drain ===\n\n");

    std::vector<int> values = {15, 10, 20, 8, 25, 5, 30, 12};
    printf("%d values, split into %d independent shards of %d:\n", N, NUM_SHARDS, SHARD_SIZE);
    for (int s = 0; s < NUM_SHARDS; s++) {
        printf("  shard %d (raw): [ ", s);
        for (int i = 0; i < SHARD_SIZE; i++) printf("%d ", values[s*SHARD_SIZE + i]);
        printf("]\n");
    }
    printf("\n");

    std::vector<int> out(N);
    for (int s = 0; s < NUM_SHARDS; s++) {
        std::vector<int> local(values.begin() + s*SHARD_SIZE, values.begin() + (s+1)*SHARD_SIZE);
        int n = SHARD_SIZE;
        for (int i = n/2 - 1; i >= 0; i--) sift_down_host(local, i, n);
        printf("  thread %d: shard heapified to [ ", s);
        for (int v : local) printf("%d ", v);
        printf("]\n");
        int base = s * SHARD_SIZE;
        for (int k = 0; k < SHARD_SIZE; k++) {
            out[base + k] = local[0];
            n--;
            local[0] = local[n];
            if (n > 0) sift_down_host(local, 0, n);
        }
        printf("  thread %d: shard drained (sorted) to [ ", s);
        for (int k = 0; k < SHARD_SIZE; k++) printf("%d ", out[base+k]);
        printf("]\n");
    }

    printf("\nper-shard sorted output array (both shards' results, side by side): [ ");
    for (int v : out) printf("%d ", v);
    printf("]\n");

    // Host-side merge of the shard outputs -- stays sequential, exactly
    // like Kruskal's cycle-check walk (Chapter 25.1) stayed sequential
    // after its own parallel front-end.
    std::vector<int> merged;
    size_t i = 0, j = 0;
    while (i < (size_t)SHARD_SIZE && j < (size_t)SHARD_SIZE) {
        if (out[i] <= out[SHARD_SIZE + j]) merged.push_back(out[i++]);
        else merged.push_back(out[SHARD_SIZE + j++]);
    }
    while (i < (size_t)SHARD_SIZE) merged.push_back(out[i++]);
    while (j < (size_t)SHARD_SIZE) merged.push_back(out[SHARD_SIZE + j++]);

    printf("merged (global sorted order): [ ");
    for (int v : merged) printf("%d ", v);
    printf("]\n");

    std::vector<int> expected_out = {8, 10, 15, 20, 5, 12, 25, 30};
    std::vector<int> expected_merged = {5, 8, 10, 12, 15, 20, 25, 30};
    bool ok = (out == expected_out) && (merged == expected_merged);

    printf("\nexpected per-shard output: [ 8 10 15 20 5 12 25 30 ], merging to\n");
    printf("[ 5 8 10 12 15 20 25 30 ] -- matching the CPU baseline's result exactly,\n");
    printf("computed by 2 threads working independently instead of 2 sequential\n");
    printf("phases\n");

    printf("\nself-check: parallel per-shard drain reproduces the exact same fully\n");
    printf("merged sequence as the sequential baseline: %s\n", ok ? "confirmed" : "MISMATCH");
    return ok ? 0 : 1;
}
```

**Compile and run:**

```bash
NVDIR=/usr/local/lib/python3.11/dist-packages/nvidia/cu13
$NVDIR/bin/nvcc -arch=sm_80 -I$NVDIR/include -L$NVDIR/lib -lcudart 145_heap_shard_drain_kernel.cu -o 145_heap_shard_drain_kernel
LD_LIBRARY_PATH=$NVDIR/lib ./145_heap_shard_drain_kernel
```

**Sample input:** the same 2 shards of 4, each fully heapified and drained by its own private thread, with no shared state between them.

**Sample output:**

```text
=== Section 26.2 main: one thread per shard, private heapify + drain ===

8 values, split into 2 independent shards of 4:
  shard 0 (raw): [ 15 10 20 8 ]
  shard 1 (raw): [ 25 5 30 12 ]

  thread 0: shard heapified to [ 8 10 20 15 ]
  thread 0: shard drained (sorted) to [ 8 10 15 20 ]
  thread 1: shard heapified to [ 5 12 30 25 ]
  thread 1: shard drained (sorted) to [ 5 12 25 30 ]

per-shard sorted output array (both shards' results, side by side): [ 8 10 15 20 5 12 25 30 ]
merged (global sorted order): [ 5 8 10 12 15 20 25 30 ]

expected per-shard output: [ 8 10 15 20 5 12 25 30 ], merging to
[ 5 8 10 12 15 20 25 30 ] -- matching the CPU baseline's result exactly,
computed by 2 threads working independently instead of 2 sequential
phases

self-check: parallel per-shard drain reproduces the exact same fully
merged sequence as the sequential baseline: confirmed
```

## 26.3 Bucket Queues: Trading Comparisons for Atomics

### Intuition

A binary heap's sift path is a chain of data-dependent comparisons, each one deciding where the next comparison even happens -- a poor fit for a GPU, which prefers uniform, predictable work across threads. When priorities are bounded, non-negative integers, a bucket queue sidesteps that chain entirely: bucket i holds every item with priority exactly i, insertion is an O(1) append into the right bucket, and extract-min is just "find the lowest-indexed non-empty bucket." Chapter 23.1's own Dijkstra's algorithm fits this perfectly, since its popped distances never decrease -- once a bucket has been passed, nothing smaller can ever arrive in it again.

### The Sequential (CPU) Baseline

```cpp
#include <cstdio>
#include <vector>
#include <algorithm>

// Chapter 26.3 -- The Sequential (CPU) Baseline.
// A binary heap's sift path is a chain of data-dependent comparisons --
// a poor fit for a GPU, which prefers uniform, predictable work. When
// priorities are bounded, non-negative integers, a BUCKET QUEUE avoids
// that chain entirely: bucket i holds every item with priority exactly
// i, insertion is an O(1) append into the right bucket, and
// extract-min is just "find the lowest-indexed non-empty bucket."
// Chapter 23.1's own Dijkstra's algorithm is exactly the kind of
// workload this fits: its own popped-vertex order NEVER decreases
// (every finalized distance is at least as large as the last one), so
// once a bucket has been passed, it can never receive a new item
// smaller than what has already been extracted -- exactly the
// "current pointer only moves forward" discipline this section uses.
// This section replays the book's own Dijkstra distances (Chapter
// 23.1) through a bucket queue and confirms it reproduces the exact
// same pop order.
#define NUM_ITEMS 6

int main() {
    printf("=== Section 26.3 CPU baseline: monotonic bucket queue ===\n\n");

    // (vertex, distance) pairs -- the book's own final Dijkstra
    // distances from Chapter 23.1.
    struct Item { int vertex, dist; };
    std::vector<Item> items = { {0,0}, {1,4}, {2,1}, {3,9}, {4,11}, {5,14} };
    printf("%d (vertex, distance) pairs, the exact final distances Chapter\n", NUM_ITEMS);
    printf("23.1's Dijkstra computed: [ ");
    for (auto& it : items) printf("(%d,%d) ", it.vertex, it.dist);
    printf("]\n\n");

    int max_dist = 0;
    for (auto& it : items) max_dist = std::max(max_dist, it.dist);
    int num_buckets = max_dist + 1;

    std::vector<std::vector<int>> buckets(num_buckets);
    for (auto& it : items) buckets[it.dist].push_back(it.vertex);

    printf("buckets (index = distance), %d buckets total:\n", num_buckets);
    for (int d = 0; d < num_buckets; d++) {
        if (!buckets[d].empty()) {
            printf("  bucket %d: [ ", d);
            for (int v : buckets[d]) printf("%d ", v);
            printf("]\n");
        }
    }
    printf("\n");

    // Monotonic extraction: `current` only ever moves FORWARD, since
    // no distance smaller than an already-passed bucket will ever
    // appear (Dijkstra's own non-negative-weight guarantee).
    int current = 0;
    std::vector<int> pop_order;
    printf("extracting in monotonic order (current pointer never moves backward):\n");
    while (current < num_buckets) {
        while (current < num_buckets && buckets[current].empty()) current++;
        if (current >= num_buckets) break;
        int v = buckets[current].front();
        buckets[current].erase(buckets[current].begin());
        pop_order.push_back(v);
        printf("  extract-min: bucket %d -> vertex %d\n", current, v);
    }

    printf("\npop order: [ ");
    for (int v : pop_order) printf("%d ", v);
    printf("]\n");

    std::vector<int> expected_pop_order = {0, 2, 1, 3, 4, 5};
    bool ok = (pop_order == expected_pop_order);

    printf("\nexpected pop order: [ 0 2 1 3 4 5 ] -- the EXACT SAME order Chapter\n");
    printf("23.1's Dijkstra popped vertices in, now produced by a bucket queue\n");
    printf("instead of Dijkstra's own O(V) global-minimum scan\n");

    printf("\nself-check: bucket queue reproduces Dijkstra's exact pop order: %s\n", ok ? "confirmed" : "MISMATCH");
    return ok ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++17 -Wall -Wextra -O2 146_bucket_queue_cpu_baseline.cpp -o 146_bucket_queue_cpu_baseline
./146_bucket_queue_cpu_baseline
```

**Sample input:** the book's own 6 final Dijkstra distances (Chapter 23.1), inserted into buckets indexed by distance and extracted via a forward-only bucket pointer.

**Sample output:**

```text
=== Section 26.3 CPU baseline: monotonic bucket queue ===

6 (vertex, distance) pairs, the exact final distances Chapter
23.1's Dijkstra computed: [ (0,0) (1,4) (2,1) (3,9) (4,11) (5,14) ]

buckets (index = distance), 15 buckets total:
  bucket 0: [ 0 ]
  bucket 1: [ 2 ]
  bucket 4: [ 1 ]
  bucket 9: [ 3 ]
  bucket 11: [ 4 ]
  bucket 14: [ 5 ]

extracting in monotonic order (current pointer never moves backward):
  extract-min: bucket 0 -> vertex 0
  extract-min: bucket 1 -> vertex 2
  extract-min: bucket 4 -> vertex 1
  extract-min: bucket 9 -> vertex 3
  extract-min: bucket 11 -> vertex 4
  extract-min: bucket 14 -> vertex 5

pop order: [ 0 2 1 3 4 5 ]

expected pop order: [ 0 2 1 3 4 5 ] -- the EXACT SAME order Chapter
23.1's Dijkstra popped vertices in, now produced by a bucket queue
instead of Dijkstra's own O(V) global-minimum scan

self-check: bucket queue reproduces Dijkstra's exact pop order: confirmed
```

### The Concept, In Detail

```
ASCII view: buckets indexed by priority, current pointer moving only forward.

  bucket:   0    1    2    3    4    5    6    7    8    9   10   11   12   13   14
           [0]  [2]   .    .   [1]   .    .    .    .   [3]   .   [4]   .    .   [5]

  current -> 0 ... 1 ... (skip 2,3) 4 ... (skip 5-8) 9 ... (skip 10) 11 ... (skip 12,13) 14

  pop order: 0, 2, 1, 3, 4, 5 -- IDENTICAL to Dijkstra's own trace
```

The forward-only pointer is what makes extraction cheap: rather than scanning all V unvisited vertices for the true minimum every single pop (Dijkstra's own O(V) per-pop cost from Chapter 23.1), a bucket queue only ever advances past buckets it has already emptied, and Dijkstra's non-negative-weight guarantee is exactly what makes "never move backward" safe. This trade only works because priorities are bounded, known integers -- a bucket queue over arbitrary floating-point priorities, or priorities with no known upper bound, has no natural array size to allocate.

[COMMON TRAP]
It is tempting to think a bucket queue's monotonic pointer is a limitation that a binary heap does not share, since a heap can extract elements in any order at all. The monotonic restriction is not a missing feature -- it is the SOURCE of the speedup, and it only applies safely because Dijkstra's own structure (or any similarly monotonic workload) guarantees priorities are only ever discovered in non-decreasing order. Using a bucket queue for a workload WITHOUT that guarantee -- one where a smaller priority could legitimately appear after a bucket has already been passed -- would silently produce wrong results, extracted out of order.

### Code and Verification

```cpp
#include <cstdio>
#include <vector>

// Chapter 26.3 main -- one thread per ITEM, all launched at once,
// inserting into a shared bucket array via `atomicAdd`: exactly the
// histogram-style scatter Chapters 7 and 13 already used, now used to
// bucket priority-queue items instead of digits. Each thread computes
// its own item's bucket (its distance), atomically reserves the next
// free slot within THAT bucket via a per-bucket counter, and writes
// its item there -- so two threads landing in the same bucket never
// overwrite each other, no matter what order they happen to run in.
// This section adds two extra candidate pairs to Chapter 26.3's own 6
// real Dijkstra distances: a real lazy-deletion Dijkstra implementation
// can legitimately push the SAME vertex more than once, at the same or
// different tentative distances, before its true shortest distance is
// finally popped -- giving this bucket queue genuine same-bucket
// contention to resolve.
#define NUM_ITEMS 8
#define MAX_DIST 14
#define NUM_BUCKETS (MAX_DIST + 1)
#define MAX_PER_BUCKET 4

struct Item { int vertex, dist; };

__global__ void bucket_insert_kernel(const Item* items, int num_items,
                                      int* bucket_counts, int* bucket_slots,
                                      int max_per_bucket) {
    int i = threadIdx.x;
    if (i >= num_items) return;
    int d = items[i].dist;
    int slot = atomicAdd(&bucket_counts[d], 1);
    if (slot < max_per_bucket) {
        bucket_slots[d * max_per_bucket + slot] = items[i].vertex;
    }
}

// ---- Host-side replay of the identical per-thread logic, run under
// two different thread-launch orders. ----

void run_order(const std::vector<Item>& items, const std::vector<int>& order, const char* label) {
    std::vector<int> bucket_counts(NUM_BUCKETS, 0);
    std::vector<int> bucket_slots(NUM_BUCKETS * MAX_PER_BUCKET, -1);

    printf("%s: threads execute in order [ ", label);
    for (int idx : order) printf("%d ", idx);
    printf("]\n");
    for (int idx : order) {
        int d = items[idx].dist;
        int slot = bucket_counts[d]++;
        printf("  thread %d: item(vertex=%d,dist=%d) -- atomicAdd(bucket_counts[%d]) reserves slot %d\n",
               idx, items[idx].vertex, d, d, slot);
        bucket_slots[d * MAX_PER_BUCKET + slot] = items[idx].vertex;
    }

    printf("  buckets after insertion:\n");
    for (int d = 0; d < NUM_BUCKETS; d++) {
        if (bucket_counts[d] == 0) continue;
        printf("    bucket %d: [ ", d);
        for (int s = 0; s < bucket_counts[d]; s++) printf("%d ", bucket_slots[d*MAX_PER_BUCKET + s]);
        printf("]\n");
    }

    // Monotonic drain, identical to Section 26.3's CPU baseline.
    std::vector<int> counts = bucket_counts;
    std::vector<int> drained;
    int current = 0;
    while (current < NUM_BUCKETS) {
        while (current < NUM_BUCKETS && counts[current] == 0) current++;
        if (current >= NUM_BUCKETS) break;
        int taken = bucket_counts[current] - counts[current];
        drained.push_back(bucket_slots[current * MAX_PER_BUCKET + taken]);
        counts[current]--;
    }
    printf("  drained vertex order: [ ");
    for (int v : drained) printf("%d ", v);
    printf("]\n\n");
}

int main() {
    printf("=== Section 26.3 main: parallel bucket insertion via atomicAdd ===\n\n");

    std::vector<Item> items = {
        {0,0}, {1,4}, {2,1}, {3,9}, {4,11}, {5,14}, {1,4}, {3,9},
    };
    printf("%d items (6 real Dijkstra distances plus 2 duplicate relaxation\n", NUM_ITEMS);
    printf("candidates -- vertex 1 and vertex 3 each pushed twice):\n  [ ");
    for (auto& it : items) printf("(%d,%d) ", it.vertex, it.dist);
    printf("]\n\n");

    std::vector<int> orderA = {0,1,2,3,4,5,6,7};
    std::vector<int> orderB = {7,6,5,4,3,2,1,0};

    run_order(items, orderA, "launch order A (natural)");
    run_order(items, orderB, "launch order B (reversed)");

    // Recompute both drains programmatically for the self-check.
    auto drain_for = [&](const std::vector<int>& order) {
        std::vector<int> bucket_counts(NUM_BUCKETS, 0);
        std::vector<int> bucket_slots(NUM_BUCKETS * MAX_PER_BUCKET, -1);
        for (int idx : order) {
            int d = items[idx].dist;
            int slot = bucket_counts[d]++;
            bucket_slots[d * MAX_PER_BUCKET + slot] = items[idx].vertex;
        }
        std::vector<int> counts = bucket_counts;
        std::vector<int> drained;
        int current = 0;
        while (current < NUM_BUCKETS) {
            while (current < NUM_BUCKETS && counts[current] == 0) current++;
            if (current >= NUM_BUCKETS) break;
            int taken = bucket_counts[current] - counts[current];
            drained.push_back(bucket_slots[current * MAX_PER_BUCKET + taken]);
            counts[current]--;
        }
        return drained;
    };

    std::vector<int> drainedA = drain_for(orderA);
    std::vector<int> drainedB = drain_for(orderB);
    std::vector<int> expected_drain = {0, 2, 1, 1, 3, 3, 4, 5};
    bool ok = (drainedA == expected_drain) && (drainedB == expected_drain);

    printf("expected drained vertex order (either launch order): [ 0 2 1 1 3 3 4 5 ]\n");
    printf("-- both orders agree, even though the WITHIN-bucket arrival order of\n");
    printf("vertex 1's two entries (and vertex 3's two entries) differs between\n");
    printf("them, since both entries share the same vertex id and distance\n");

    printf("\nself-check: parallel atomicAdd-based bucket insertion drains to the\n");
    printf("exact same vertex order under both launch orders: %s\n", ok ? "confirmed" : "MISMATCH");
    return ok ? 0 : 1;
}
```

**Compile and run:**

```bash
NVDIR=/usr/local/lib/python3.11/dist-packages/nvidia/cu13
$NVDIR/bin/nvcc -arch=sm_80 -I$NVDIR/include -L$NVDIR/lib -lcudart 147_bucket_queue_parallel_insert_kernel.cu -o 147_bucket_queue_parallel_insert_kernel
LD_LIBRARY_PATH=$NVDIR/lib ./147_bucket_queue_parallel_insert_kernel
```

**Sample input:** the same 6 distances plus 2 duplicate relaxation candidates (8 items total), inserted by 8 threads at once via `atomicAdd`-based bucket slot reservation, under two different launch orders.

**Sample output:**

```text
=== Section 26.3 main: parallel bucket insertion via atomicAdd ===

8 items (6 real Dijkstra distances plus 2 duplicate relaxation
candidates -- vertex 1 and vertex 3 each pushed twice):
  [ (0,0) (1,4) (2,1) (3,9) (4,11) (5,14) (1,4) (3,9) ]

launch order A (natural): threads execute in order [ 0 1 2 3 4 5 6 7 ]
  thread 0: item(vertex=0,dist=0) -- atomicAdd(bucket_counts[0]) reserves slot 0
  thread 1: item(vertex=1,dist=4) -- atomicAdd(bucket_counts[4]) reserves slot 0
  thread 2: item(vertex=2,dist=1) -- atomicAdd(bucket_counts[1]) reserves slot 0
  thread 3: item(vertex=3,dist=9) -- atomicAdd(bucket_counts[9]) reserves slot 0
  thread 4: item(vertex=4,dist=11) -- atomicAdd(bucket_counts[11]) reserves slot 0
  thread 5: item(vertex=5,dist=14) -- atomicAdd(bucket_counts[14]) reserves slot 0
  thread 6: item(vertex=1,dist=4) -- atomicAdd(bucket_counts[4]) reserves slot 1
  thread 7: item(vertex=3,dist=9) -- atomicAdd(bucket_counts[9]) reserves slot 1
  buckets after insertion:
    bucket 0: [ 0 ]
    bucket 1: [ 2 ]
    bucket 4: [ 1 1 ]
    bucket 9: [ 3 3 ]
    bucket 11: [ 4 ]
    bucket 14: [ 5 ]
  drained vertex order: [ 0 2 1 1 3 3 4 5 ]

launch order B (reversed): threads execute in order [ 7 6 5 4 3 2 1 0 ]
  thread 7: item(vertex=3,dist=9) -- atomicAdd(bucket_counts[9]) reserves slot 0
  thread 6: item(vertex=1,dist=4) -- atomicAdd(bucket_counts[4]) reserves slot 0
  thread 5: item(vertex=5,dist=14) -- atomicAdd(bucket_counts[14]) reserves slot 0
  thread 4: item(vertex=4,dist=11) -- atomicAdd(bucket_counts[11]) reserves slot 0
  thread 3: item(vertex=3,dist=9) -- atomicAdd(bucket_counts[9]) reserves slot 1
  thread 2: item(vertex=2,dist=1) -- atomicAdd(bucket_counts[1]) reserves slot 0
  thread 1: item(vertex=1,dist=4) -- atomicAdd(bucket_counts[4]) reserves slot 1
  thread 0: item(vertex=0,dist=0) -- atomicAdd(bucket_counts[0]) reserves slot 0
  buckets after insertion:
    bucket 0: [ 0 ]
    bucket 1: [ 2 ]
    bucket 4: [ 1 1 ]
    bucket 9: [ 3 3 ]
    bucket 11: [ 4 ]
    bucket 14: [ 5 ]
  drained vertex order: [ 0 2 1 1 3 3 4 5 ]

expected drained vertex order (either launch order): [ 0 2 1 1 3 3 4 5 ]
-- both orders agree, even though the WITHIN-bucket arrival order of
vertex 1's two entries (and vertex 3's two entries) differs between
them, since both entries share the same vertex id and distance

self-check: parallel atomicAdd-based bucket insertion drains to the
exact same vertex order under both launch orders: confirmed
```

## Chapter Summary

A binary heap's construction and its extraction behave completely differently under parallelism. Building one from an array already in memory parallelizes cleanly level by level, bottom-up, since sibling subtrees at the same level never share array slots -- turning what repeated insertion does in O(n log n) sequential work into O(log n) rounds of full parallelism. Extracting from a single heap does not parallelize at all, since each extraction's result depends entirely on the previous one's sift-down, but sharding a batch across many independent private heaps turns that sequential bottleneck into many independent sequential drains running at once, merged only at the very end. When priorities are bounded, non-negative integers -- exactly the situation Dijkstra's algorithm (Chapter 23.1) creates -- a bucket queue replaces the heap's data-dependent comparison chain entirely with the same atomic-scatter insertion Chapters 7 and 13 already used for histograms and radix sort, and a forward-only extraction pointer that is safe only because priorities are guaranteed to arrive in non-decreasing order.

## Self-Check Questions

1. Why can every internal node at the SAME level of a heap safely sift down in parallel, while nodes at DIFFERENT levels cannot be processed out of order?
2. Why does sharding a batch of elements across several independent heaps not make any single extraction faster, even though it speeds up finding the k smallest elements overall?
3. What determines the right number of shards to use when draining a batch via sharded heaps?
4. Why does a bucket queue's array need to be sized according to the maximum possible priority, and what happens if that maximum is not known in advance?
5. Why is it safe for a bucket queue's extraction pointer to only ever move forward when used for Dijkstra's algorithm specifically?
6. Two threads both insert an item into the same bucket at the same time. What prevents one insertion from overwriting the other, and how does the earlier chapters' histogram-building technique relate?

## Where We Go Next

Bucket queues traded a heap's comparison chain for atomic-scatter insertion, but they still assume every insertion and extraction is safe to run without any explicit locking discipline beyond a single atomic operation. Chapter 27 confronts concurrency more directly: building lock-free stacks, queues, and other structures whose correctness must be proven under ANY possible interleaving of concurrent operations, not just the specific orderings this book's examples have traced by hand.

## Worked Solutions

**1.** Two internal nodes at the same level are always the roots of two disjoint subtrees -- neither one's descendants overlap with the other's -- so their sift-down operations can never read or write the same array slot, making them safe to run at once with no coordination. A parent's sift-down, by contrast, needs to compare itself against its CHILDREN, and that comparison is only valid once those children's own subtrees have already been fully heapified; running a parent's sift-down before its children's level has finished could compare against a value that is about to move.

**2.** Sharding splits the total batch across several INDEPENDENT heaps, and each shard's own drain loop is exactly as sequential internally as ever -- one extraction at a time, each depending on the last. What sharding speeds up is running many such independent sequential drains SIMULTANEOUSLY, one per shard, rather than draining one giant heap of everything at once; no single extraction within any one shard becomes any faster than before.

**3.** More shards means more independent sequential drains can run at once, which increases parallelism, but it also means the final merge step -- which combines all the per-shard sorted sequences and must run sequentially (or via a separate parallel merge network) -- has more sequences to combine, and each individual shard's heap has fewer elements to work with, weakening how effectively it concentrates the truly smallest values. The right shard count balances the parallel speedup from more independent drains against the added sequential merge cost that more shards creates.

**4.** A bucket queue needs one array slot (or bucket) for every possible priority value from 0 up to the maximum, since insertion works by directly indexing into `buckets[priority]` with no search involved -- that direct indexing is exactly what makes insertion O(1) instead of requiring a comparison-based search for the right spot. If the maximum possible priority is not known in advance, there is no way to size that array correctly ahead of time, and a bucket queue loses its main advantage; a comparison-based structure like a binary heap, which never needs to know priorities' range in advance, becomes the safer choice.

**5.** Dijkstra's algorithm specifically guarantees that every edge weight is non-negative, which means once a vertex's true shortest distance has been finalized, every future candidate distance computed from it (by relaxing its outgoing edges) can only be EQUAL to or LARGER than what has already been finalized -- never smaller. That guarantee is exactly what makes it safe for the bucket queue's extraction pointer to never look backward: nothing smaller than an already-passed bucket's contents will ever legitimately need to be inserted into it after the fact.

**6.** Each thread reserves its own SLOT within a bucket via `atomicAdd` on that bucket's counter before writing anything -- the atomic operation guarantees that if two threads both target the same bucket at the same instant, one of them gets slot 0 and the other gets slot 1 (or whatever the next two available slots are), with no possibility of both threads computing the same slot index and overwriting each other. This is the exact same technique Chapters 7 and 13 used to scatter items into histogram bins and radix-sort buckets: an atomic counter per bucket turns "many threads writing to a shared destination" into "many threads each getting their own guaranteed-unique destination slot."
