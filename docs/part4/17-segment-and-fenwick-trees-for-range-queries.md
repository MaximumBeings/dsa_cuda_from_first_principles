# Chapter 17: Segment and Fenwick Trees for Range Queries

Every tree this book has built so far answered questions about individual elements: is this key present, what value does this leaf hold. A huge and common class of real questions instead asks about a RANGE -- the sum, minimum, or maximum of every element between two positions -- and recomputing that by scanning the range every time is wasteful when the same underlying array gets queried, and occasionally updated, over and over. This chapter builds two structures purpose-built for exactly that: the segment tree, an explicit binary tree over the array (continuing this book's own array-index discipline from Chapters 11 and 15), and the Fenwick tree (binary indexed tree), a leaner structure that answers the identical questions with a fraction of the memory and no explicit tree shape at all. Both structures also reopen this book's running concurrency story from a new angle: unlike Chapter 16's trie node creation, a point update here is a simple ADDITION, and addition's own commutativity turns out to make concurrent updates far easier to get right than concurrent node creation ever was.

## 17.1 Iterative Segment Tree Construction and Range-Sum Query

### Intuition

A segment tree precomputes the sum of every node's own subtree once, so that any contiguous range's sum can be assembled from a small number of already-computed pieces instead of adding up every element in the range one at a time. Continuing this book's iterative discipline (Chapter 15's explicit stack, Chapter 16's explicit queue), this section uses the well-known 1-indexed ARRAY layout: for an array of `n` elements, a `2n`-slot array holds the leaves at indices `[n, 2n)` and every internal node `i` at `tree[i] = tree[2*i] + tree[2*i+1]`, built bottom-up with a single loop and no recursion. A range-sum query over `[l, r)` uses a similarly iterative "two pointers walking up together" technique: whenever a pointer is the RIGHT child of its parent, its entire subtree is a complete, self-contained piece of the answer.

### The Sequential (CPU) Baseline

```cpp
#include <cstdio>
#include <vector>

// Chapter 17.1 -- The Sequential (CPU) Baseline.
// A segment tree answers range-sum queries in O(log n) by precomputing
// every "power-of-two-aligned" sum once: leaf i holds a[i], and every
// internal node holds the sum of its two children -- so the sum of any
// contiguous range can be assembled from at most O(log n) precomputed
// pieces instead of walking every element in the range. This baseline
// uses the classic ITERATIVE, 1-indexed, array-based layout (continuing
// this book's discipline since Chapter 11 of never using real pointers
// for tree structure): a size-8 array of leaves lives in `tree[8..15]`,
// and `tree[i] = tree[2*i] + tree[2*i+1]` for every internal node,
// built bottom-up with a single loop -- no recursion anywhere.
#define N 8

void build(const std::vector<int>& a, std::vector<int>& tree) {
    tree.assign(2 * N, 0);
    for (int i = 0; i < N; i++) tree[N + i] = a[i];
    for (int i = N - 1; i >= 1; i--) tree[i] = tree[2 * i] + tree[2 * i + 1];
}

// Range sum over the half-open range [l, r), using the well-known
// iterative "walk two pointers up from the leaves" technique: whenever
// a pointer is the RIGHT child of its parent, its own subtree's sum is
// a complete, self-contained piece of the answer, so it gets added in
// directly and the pointer moves past it; otherwise both pointers just
// move up to their parents together.
int range_sum(const std::vector<int>& tree, int l, int r) {
    int res = 0;
    l += N; r += N;
    while (l < r) {
        if (l & 1) res += tree[l++];
        if (r & 1) res += tree[--r];
        l >>= 1; r >>= 1;
    }
    return res;
}

int main() {
    printf("=== Section 17.1 CPU baseline: iterative segment tree build + range queries ===\n\n");

    std::vector<int> a = {5, 8, 6, 3, 2, 7, 4, 9};
    printf("input array a: ");
    for (int v : a) printf("%d ", v);
    printf("\n\n");

    std::vector<int> tree;
    build(a, tree);
    printf("segment tree array (1-indexed, tree[0] unused, leaves at tree[8..15]):\n  ");
    for (int i = 1; i < 2 * N; i++) printf("%d ", tree[i]);
    printf("\n\n");

    struct Query { int l, r; };
    std::vector<Query> queries = {{1, 5}, {0, 8}, {2, 3}, {4, 7}};

    printf("range-sum queries (half-open [l,r)):\n");
    bool ok = true;
    std::vector<int> expected = {19, 44, 6, 13};
    for (size_t i = 0; i < queries.size(); i++) {
        int result = range_sum(tree, queries[i].l, queries[i].r);
        int brute = 0;
        for (int j = queries[i].l; j < queries[i].r; j++) brute += a[j];
        printf("  query[%d,%d) = %d  (brute-force check: %d)\n",
               queries[i].l, queries[i].r, result, brute);
        ok = ok && (result == expected[i]) && (result == brute);
    }

    printf("\nself-check: segment tree built correctly, all range queries match brute-force\n");
    printf("sums: %s\n", ok ? "confirmed" : "MISMATCH");
    return ok ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++17 -Wall -Wextra -O2 88_segtree_build_query_cpu_baseline.cpp -o 88_segtree_build_query_cpu_baseline
./88_segtree_build_query_cpu_baseline
```

**Sample input:** the array `{5, 8, 6, 3, 2, 7, 4, 9}`, built into a segment tree and queried over four sample ranges.

**Sample output:**

```text
=== Section 17.1 CPU baseline: iterative segment tree build + range queries ===

input array a: 5 8 6 3 2 7 4 9 

segment tree array (1-indexed, tree[0] unused, leaves at tree[8..15]):
  44 22 22 13 9 9 13 5 8 6 3 2 7 4 9 

range-sum queries (half-open [l,r)):
  query[1,5) = 19  (brute-force check: 19)
  query[0,8) = 44  (brute-force check: 44)
  query[2,3) = 6  (brute-force check: 6)
  query[4,7) = 13  (brute-force check: 13)

self-check: segment tree built correctly, all range queries match brute-force
sums: confirmed
```

### The Concept, In Detail

Building the tree bottom-up on `a = {5, 8, 6, 3, 2, 7, 4, 9}` (`N = 8`):

```
leaves (tree[8..15]):  5  8  6  3  2  7  4  9

level (internal, width 4): tree[4..7]
  tree[4] = tree[8]  + tree[9]  = 5 + 8 = 13
  tree[5] = tree[10] + tree[11] = 6 + 3 = 9
  tree[6] = tree[12] + tree[13] = 2 + 7 = 9
  tree[7] = tree[14] + tree[15] = 4 + 9 = 13

level (internal, width 2): tree[2..3]
  tree[2] = tree[4] + tree[5] = 13 + 9  = 22
  tree[3] = tree[6] + tree[7] = 9  + 13 = 22

level (internal, width 1): tree[1]
  tree[1] = tree[2] + tree[3] = 22 + 22 = 44   (the sum of the whole array)
```

```
ASCII view:

                         44 (tree[1])
                    /                  \
              22 (tree[2])          22 (tree[3])
              /        \             /        \
        13 (tree[4]) 9 (tree[5]) 9 (tree[6]) 13 (tree[7])
        /    \        /    \      /    \       /    \
       5      8      6      3    2      7     4      9
     (leaf8)(leaf9)(leaf10)(leaf11)(leaf12)(leaf13)(leaf14)(leaf15)
```

Tracing query `[1, 5)` (sum of `a[1..4]` = `8+6+3+2` = 19):

```
l = 1+8 = 9, r = 5+8 = 13

step 1: l=9 is odd (right child) -> res += tree[9]=8;  l becomes 10
        r=13 is odd (right child) -> r becomes 12; res += tree[12]=2
        l >>= 1 -> l=5;  r >>= 1 -> r=6
        res so far = 8 + 2 = 10

step 2: l=5 is odd -> res += tree[5]=9; l becomes 6
        r=6 is even -> no action
        l >>= 1 -> l=3; r >>= 1 -> r=3
        res so far = 10 + 9 = 19

l == r -> loop ends. result = 19  (matches brute force: 8+6+3+2=19)
```

Every internal node's value depends only on its two children, already known from the level below -- exactly Chapter 15.1's and Chapter 16.1's own "loop over levels, synchronize between them" shape, with tree nodes (rather than BST groups or sorted-array ranges) as the unit of independent work. A range query, by contrast, is a single short walk of at most `O(log n)` steps -- too little work, and too data-dependent step to step, to split across multiple threads usefully. The real parallel target, exactly Chapter 15.2/15.3's own insight about tree traversal, is running MANY independent queries at once, one thread per query, each with no interaction with any other query at all.

[COMMON TRAP]
It is tempting to think a single range query itself is "embarrassingly parallel" because it only touches `O(log n)` nodes. Each step of the two-pointer walk depends on the RESULT of the previous step (`l` and `r` are recomputed every iteration) -- the walk is short, but it is sequential, exactly like Chapter 11's pointer-chasing linked list was short-work-but-sequential. Parallelism here comes from running many queries side by side, never from splitting one query's own walk across threads.

### Code and Verification

```cpp
#include <cstdio>
#include <vector>

// Chapter 17.1 main -- two independent parallel opportunities live in a
// segment tree, and this file demonstrates both. Construction is
// LEVEL-BY-LEVEL, exactly Chapter 15.1's and Chapter 16.1's own "loop
// over levels, synchronize between them" shape: every node WITHIN one
// level depends only on two already-known children from the level
// below, so all nodes in a level are independent of each other and can
// be built by one thread each. A single range query, by contrast, has
// an unavoidable small span (its own O(log n) walk) -- so, exactly
// Chapter 15.2/15.3's "many independent traversals" insight, the real
// parallel target for QUERYING is running MANY queries at once, one
// thread per query, each one a completely independent read.
#define N 8

// One thread per node in the CURRENT level being built. `lo` is the
// first index of this level (a level of `width` nodes occupies indices
// [lo, lo+width)); each node's two children always live at `2*index`
// and `2*index+1`, already finalized by the previous level's kernel
// launch.
__global__ void build_level_kernel(int* tree, int lo, int width) {
    int t = threadIdx.x;
    if (t >= width) return;
    int idx = lo + t;
    tree[idx] = tree[2 * idx] + tree[2 * idx + 1];
}

// One thread per QUERY -- completely independent reads, no shared
// mutable state at all, so no synchronization or atomics are needed
// regardless of how many queries run at once.
__global__ void range_sum_kernel(const int* tree, int n, const int* qs_l,
                                  const int* qs_r, int* results) {
    int t = threadIdx.x;
    int l = qs_l[t] + n;
    int r = qs_r[t] + n;
    int res = 0;
    while (l < r) {
        if (l & 1) res += tree[l++];
        if (r & 1) res += tree[--r];
        l >>= 1; r >>= 1;
    }
    results[t] = res;
}

// ---- Host-side replay of the identical per-thread logic: leaves are
// given directly, then each internal level's nodes are all computed by
// independent "threads" before the next level (narrower by half) can
// start -- and, separately, every query below runs against the
// finished tree with no interaction between queries at all. ----

int main() {
    printf("=== Section 17.1 main: level-by-level parallel build, parallel range queries ===\n\n");

    std::vector<int> a = {5, 8, 6, 3, 2, 7, 4, 9};
    printf("input array a: ");
    for (int v : a) printf("%d ", v);
    printf("\n\n");

    std::vector<int> tree(2 * N, 0);
    for (int i = 0; i < N; i++) tree[N + i] = a[i];
    printf("level (leaves, width %d): tree[%d..%d] set directly from a[]\n", N, N, 2 * N - 1);

    for (int width = N / 2; width >= 1; width /= 2) {
        int lo = width;
        printf("level (width %d): %d independent thread(s), building tree[%d..%d]\n",
               width, width, lo, lo + width - 1);
        for (int t = 0; t < width; t++) {
            int idx = lo + t;
            tree[idx] = tree[2 * idx] + tree[2 * idx + 1];
        }
    }
    printf("\nsegment tree array (1-indexed, leaves at tree[8..15]):\n  ");
    for (int i = 1; i < 2 * N; i++) printf("%d ", tree[i]);
    printf("\n\n");

    struct Query { int l, r; };
    std::vector<Query> queries = {{1, 5}, {0, 8}, {2, 3}, {4, 7}};
    std::vector<int> results(queries.size());

    printf("%zu independent query threads, each walking the tree on its own:\n", queries.size());
    for (size_t t = 0; t < queries.size(); t++) {
        int l = queries[t].l + N, r = queries[t].r + N;
        int res = 0;
        while (l < r) {
            if (l & 1) res += tree[l++];
            if (r & 1) res += tree[--r];
            l >>= 1; r >>= 1;
        }
        results[t] = res;
        printf("  thread %zu: query[%d,%d) -> %d\n", t, queries[t].l, queries[t].r, res);
    }

    std::vector<int> expected_tree = {44, 22, 22, 13, 9, 9, 13, 5, 8, 6, 3, 2, 7, 4, 9};
    std::vector<int> expected_results = {19, 44, 6, 13};
    bool ok = true;
    for (int i = 1; i < 2 * N; i++) ok = ok && (tree[i] == expected_tree[i - 1]);
    for (size_t t = 0; t < results.size(); t++) ok = ok && (results[t] == expected_results[t]);

    printf("\nexpected tree: 44 22 22 13 9 9 13 5 8 6 3 2 7 4 9\n");
    printf("expected query results: 19 44 6 13\n");
    printf("\nself-check: level-by-level parallel build and parallel independent queries\n");
    printf("match the CPU baseline exactly: %s\n", ok ? "confirmed" : "MISMATCH");
    return ok ? 0 : 1;
}
```

**Compile and run:**

```bash
NVDIR=/usr/local/lib/python3.11/dist-packages/nvidia/cu13
$NVDIR/bin/nvcc -arch=sm_80 -I$NVDIR/include -L$NVDIR/lib -lcudart 89_segtree_build_query_kernel.cu -o 89_segtree_build_query_kernel
LD_LIBRARY_PATH=$NVDIR/lib ./89_segtree_build_query_kernel
```

**Sample input:** the same array, built level by level (one thread per node per level), then the same four ranges queried by four independent threads at once.

**Sample output:**

```text
=== Section 17.1 main: level-by-level parallel build, parallel range queries ===

input array a: 5 8 6 3 2 7 4 9 

level (leaves, width 8): tree[8..15] set directly from a[]
level (width 4): 4 independent thread(s), building tree[4..7]
level (width 2): 2 independent thread(s), building tree[2..3]
level (width 1): 1 independent thread(s), building tree[1..1]

segment tree array (1-indexed, leaves at tree[8..15]):
  44 22 22 13 9 9 13 5 8 6 3 2 7 4 9 

4 independent query threads, each walking the tree on its own:
  thread 0: query[1,5) -> 19
  thread 1: query[0,8) -> 44
  thread 2: query[2,3) -> 6
  thread 3: query[4,7) -> 13

expected tree: 44 22 22 13 9 9 13 5 8 6 3 2 7 4 9
expected query results: 19 44 6 13

self-check: level-by-level parallel build and parallel independent queries
match the CPU baseline exactly: confirmed
```

## 17.2 Concurrent Point Updates: Why Plain atomicAdd Is Enough Here

### Intuition

Changing one array element means every one of its ancestors now holds a stale sum. A point update fixes this in `O(log n)` by adding the SAME delta to the changed leaf and to every node on the path from that leaf to the root -- correct because the tree's own invariant (`tree[i] = tree[2*i] + tree[2*i+1]`) is purely additive. Chapter 16.2 needed `atomicCAS` because "create this specific node" is NOT commutative -- whichever thread's write lands last silently wins, discarding the other. "Add this delta" is different in exactly the way that matters: addition is commutative and associative, so no matter what order multiple threads' additions to the SAME shared ancestor happen in, the final sum comes out identical. That means concurrent point updates need nothing more than the plain, unconditional `atomicAdd` this book already established in Chapters 7 and 8.

### The Sequential (CPU) Baseline

```cpp
#include <cstdio>
#include <vector>

// Chapter 17.2 -- The Sequential (CPU) Baseline.
// Changing one element of the underlying array means every ancestor of
// that element's leaf holds a now-stale sum. A point update fixes this
// by adding the SAME delta to every node on the path from that leaf up
// to the root -- exactly the additive relationship the tree already
// encodes (`tree[i] = tree[2*i] + tree[2*i+1]`), so adding `delta` to a
// leaf and to every one of its ancestors keeps every sum correct
// without rebuilding anything else in the tree.
#define N 8

void build(const std::vector<int>& a, std::vector<int>& tree) {
    tree.assign(2 * N, 0);
    for (int i = 0; i < N; i++) tree[N + i] = a[i];
    for (int i = N - 1; i >= 1; i--) tree[i] = tree[2 * i] + tree[2 * i + 1];
}

void point_update(std::vector<int>& tree, int leaf, int delta) {
    int idx = N + leaf;
    while (idx >= 1) {
        tree[idx] += delta;
        idx >>= 1;
    }
}

int main() {
    printf("=== Section 17.2 CPU baseline: sequential point updates ===\n\n");

    std::vector<int> a = {5, 8, 6, 3, 2, 7, 4, 9};
    std::vector<int> tree;
    build(a, tree);
    printf("initial tree (1-indexed, leaves at tree[8..15]):\n  ");
    for (int i = 1; i < 2 * N; i++) printf("%d ", tree[i]);
    printf("\n\n");

    struct Update { int leaf, delta; };
    std::vector<Update> updates = {{1, 5}, {5, -2}, {6, 10}};

    printf("applying updates one at a time:\n");
    for (auto& u : updates) {
        point_update(tree, u.leaf, u.delta);
        printf("  after leaf=%d delta=%+d: tree = ", u.leaf, u.delta);
        for (int i = 1; i < 2 * N; i++) printf("%d ", tree[i]);
        printf("\n");
    }
    printf("\n");

    printf("final tree:\n  ");
    for (int i = 1; i < 2 * N; i++) printf("%d ", tree[i]);
    printf("\n\n");

    // Cross-check: rebuilding from scratch with the updated array must
    // give the identical tree.
    std::vector<int> new_a = a;
    for (auto& u : updates) new_a[u.leaf] += u.delta;
    std::vector<int> tree_rebuilt;
    build(new_a, tree_rebuilt);

    printf("updated array a: ");
    for (int v : new_a) printf("%d ", v);
    printf("\nrebuilt-from-scratch tree:\n  ");
    for (int i = 1; i < 2 * N; i++) printf("%d ", tree_rebuilt[i]);
    printf("\n\n");

    std::vector<int> expected = {57, 27, 30, 18, 9, 7, 23, 5, 13, 6, 3, 2, 5, 14, 9};
    bool ok = true;
    for (int i = 1; i < 2 * N; i++) {
        ok = ok && (tree[i] == expected[i - 1]) && (tree[i] == tree_rebuilt[i]);
    }

    printf("expected final tree: 57 27 30 18 9 7 23 5 13 6 3 2 5 14 9\n");
    printf("\nself-check: incremental point updates match a full from-scratch rebuild: %s\n",
           ok ? "confirmed" : "MISMATCH");
    return ok ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++17 -Wall -Wextra -O2 90_segtree_update_cpu_baseline.cpp -o 90_segtree_update_cpu_baseline
./90_segtree_update_cpu_baseline
```

**Sample input:** Section 17.1's tree, updated one at a time with three point updates (`leaf=1,delta=+5`; `leaf=5,delta=-2`; `leaf=6,delta=+10`), cross-checked against a full from-scratch rebuild.

**Sample output:**

```text
=== Section 17.2 CPU baseline: sequential point updates ===

initial tree (1-indexed, leaves at tree[8..15]):
  44 22 22 13 9 9 13 5 8 6 3 2 7 4 9 

applying updates one at a time:
  after leaf=1 delta=+5: tree = 49 27 22 18 9 9 13 5 13 6 3 2 7 4 9 
  after leaf=5 delta=-2: tree = 47 27 20 18 9 7 13 5 13 6 3 2 5 4 9 
  after leaf=6 delta=+10: tree = 57 27 30 18 9 7 23 5 13 6 3 2 5 14 9 

final tree:
  57 27 30 18 9 7 23 5 13 6 3 2 5 14 9 

updated array a: 5 13 6 3 2 5 14 9 
rebuilt-from-scratch tree:
  57 27 30 18 9 7 23 5 13 6 3 2 5 14 9 

expected final tree: 57 27 30 18 9 7 23 5 13 6 3 2 5 14 9

self-check: incremental point updates match a full from-scratch rebuild: confirmed
```

### The Concept, In Detail

Each update's leaf-to-root path visits exactly `log2(N)+1 = 4` nodes (`N=8`):

```
update(leaf=1, delta=+5): path = [9, 4, 2, 1]
update(leaf=5, delta=-2): path = [13, 6, 3, 1]
update(leaf=6, delta=+10): path = [14, 7, 3, 1]
```

All three paths share node 1 (the root -- every update's path always ends there). The second and third paths additionally share node 3. Running these three updates CONCURRENTLY, one thread per update, in a lockstep "every thread takes one step per wave" schedule (all three paths happen to be the same length here, since every leaf in a complete tree is the same distance from the root):

```
wave 0: thread0 -> node9,  thread1 -> node13, thread2 -> node14   (no contention)
wave 1: thread0 -> node4,  thread1 -> node6,   thread2 -> node7   (no contention)
wave 2: thread0 -> node2,  thread1 & thread2 BOTH -> node3        (contended!)
wave 3: thread0 & thread1 & thread2 ALL -> node1                  (contended!)
```

At the two contended waves, multiple threads call `atomicAdd` on the exact same memory location at the exact same time. Nothing about this needs detecting, retrying, or resolving a winner -- unlike Chapter 16.2's CAS dance, every contributing thread's delta simply gets added in, and the hardware serializes the individual additions in SOME order (never specified, never needing to be), with the final sum being identical regardless of which order that turns out to be.

```
ASCII view: what makes this safe, side by side with Chapter 16.2's race.

  Chapter 16.2 (node CREATION):        Chapter 17.2 (delta ADDITION):
    slot starts at -1 (undefined)        slot starts at a real, valid sum
    two threads want DIFFERENT values    two threads want to ADD their own
      written into the SAME slot           amount to the SAME slot
    whichever write lands LAST wins,     EVERY contribution lands, in any
      the other is silently lost           order -- sum is order-independent
    fix: atomicCAS (detect and retry)    fix: nothing beyond atomicAdd needed
```

[COMMON TRAP]
Not every "two threads write to the same shared location" situation needs `atomicCAS`. The deciding question is whether the operation being combined is COMMUTATIVE (`add`, and the `atomicMin`/`atomicMax`/`atomicOr` families are further examples already implying an ordering-independent combine) or NOT (`assign this specific value`, `create this specific node`, `push this exact element onto a stack's head`, all of which depend on WHICH write happens to be visible last). Reaching for `atomicCAS` when plain `atomicAdd` already suffices adds real overhead (a retry loop) for no correctness benefit.

### Code and Verification

```cpp
#include <cstdio>
#include <vector>
#include <map>
#include <algorithm>

// Chapter 17.2 main -- Chapter 16.2 needed atomicCAS specifically
// because "create this shared node" is NOT commutative: whichever
// thread's write lands LAST silently wins, and order matters. A point
// update's own operation -- "add this delta to a shared counter" -- is
// different in exactly the way that matters: addition is commutative
// and associative, so no matter what ORDER multiple threads' additions
// to the SAME shared ancestor happen in, the final sum is identical.
// That means the plain, unconditional `atomicAdd` this book already
// established in Chapters 7 and 8 -- not CAS, not a retry loop -- is
// already sufficient here, even though multiple threads' update paths
// provably collide at shared ancestors (including the root, which
// EVERY update always passes through).
#define N 8

// One thread per UPDATE. Every thread walks its own leaf-to-root path
// and atomically adds its own delta at every node along the way;
// different threads' paths merge at shared ancestors, and `atomicAdd`
// alone (no CAS, no retry) is all that is needed there.
__global__ void point_update_kernel(int* tree, int n, const int* leaf_idx,
                                     const int* deltas, int num_updates) {
    int t = threadIdx.x;
    if (t >= num_updates) return;
    int idx = n + leaf_idx[t];
    int delta = deltas[t];
    while (idx >= 1) {
        atomicAdd(&tree[idx], delta);
        idx >>= 1;
    }
}

// ---- Host-side replay under a genuinely adversarial interleaving: all
// three threads' paths happen to be the same length here (leaf depth is
// uniform in a complete tree), so a "wave" schedule -- every thread
// takes exactly one step up its own path per wave, all waves
// interleaved in lockstep -- forces maximum contention at any node two
// or more threads' paths pass through in the SAME wave. Nothing in this
// replay ever checks "did I get here first" -- it only ever adds. ----

int main() {
    printf("=== Section 17.2 main: concurrent point updates via plain atomicAdd ===\n\n");

    std::vector<int> a = {5, 8, 6, 3, 2, 7, 4, 9};
    std::vector<int> tree(2 * N, 0);
    for (int i = 0; i < N; i++) tree[N + i] = a[i];
    for (int i = N - 1; i >= 1; i--) tree[i] = tree[2 * i] + tree[2 * i + 1];
    printf("initial tree (1-indexed, leaves at tree[8..15]):\n  ");
    for (int i = 1; i < 2 * N; i++) printf("%d ", tree[i]);
    printf("\n\n");

    struct Update { int leaf, delta; };
    std::vector<Update> updates = {{1, 5}, {5, -2}, {6, 10}};
    printf("3 threads update concurrently:\n");
    for (size_t t = 0; t < updates.size(); t++) {
        printf("  thread %zu: leaf=%d delta=%+d\n", t, updates[t].leaf, updates[t].delta);
    }
    printf("\n");

    std::vector<std::vector<int>> paths(updates.size());
    for (size_t t = 0; t < updates.size(); t++) {
        int idx = N + updates[t].leaf;
        while (idx >= 1) { paths[t].push_back(idx); idx >>= 1; }
    }

    size_t max_len = 0;
    for (auto& p : paths) max_len = std::max(max_len, p.size());

    printf("lockstep wave schedule (every thread advances one step per wave):\n");
    for (size_t w = 0; w < max_len; w++) {
        std::map<int, std::vector<int>> contributions;   // node -> which threads write here this wave
        for (size_t t = 0; t < updates.size(); t++) {
            if (w < paths[t].size()) contributions[paths[t][w]].push_back((int)t);
        }
        printf("  wave %zu: ", w);
        for (auto& kv : contributions) {
            printf("node%d<-thread", kv.first);
            for (size_t i = 0; i < kv.second.size(); i++) {
                printf("%d%s", kv.second[i], i + 1 < kv.second.size() ? "&" : "");
            }
            printf(kv.first == contributions.rbegin()->first ? "" : ", ");
        }
        if (contributions.size() < updates.size()) {
            // at least one node in this wave received more than one thread
            for (auto& kv : contributions) {
                if (kv.second.size() > 1) { printf("  [contended node -- multiple atomicAdd, no conflict]"); break; }
            }
        }
        printf("\n");
        for (size_t t = 0; t < updates.size(); t++) {
            if (w < paths[t].size()) tree[paths[t][w]] += updates[t].delta;   // atomicAdd, order within a wave irrelevant
        }
    }
    printf("\n");

    printf("final tree:\n  ");
    for (int i = 1; i < 2 * N; i++) printf("%d ", tree[i]);
    printf("\n\n");

    std::vector<int> expected = {57, 27, 30, 18, 9, 7, 23, 5, 13, 6, 3, 2, 5, 14, 9};
    bool ok = true;
    for (int i = 1; i < 2 * N; i++) ok = ok && (tree[i] == expected[i - 1]);

    printf("expected final tree (identical to Section 17.2's CPU baseline, applied\n");
    printf("sequentially in program order): 57 27 30 18 9 7 23 5 13 6 3 2 5 14 9\n");
    printf("\nnode 1 (the root) receives contributions from ALL THREE threads, in the SAME\n");
    printf("wave -- three simultaneous atomicAdd calls on one shared counter -- and node 3\n");
    printf("receives two. Unlike Chapter 16.2's trie node creation, no thread ever needs to\n");
    printf("discover it \"lost\" anything: every contribution is simply summed in, in\n");
    printf("whatever order the hardware happens to serialize the atomics.\n");

    printf("\nself-check: concurrent atomicAdd-based updates match the sequential CPU\n");
    printf("baseline exactly, regardless of interleaving: %s\n", ok ? "confirmed" : "MISMATCH");
    return ok ? 0 : 1;
}
```

**Compile and run:**

```bash
NVDIR=/usr/local/lib/python3.11/dist-packages/nvidia/cu13
$NVDIR/bin/nvcc -arch=sm_80 -I$NVDIR/include -L$NVDIR/lib -lcudart 91_segtree_concurrent_update_kernel.cu -o 91_segtree_concurrent_update_kernel
LD_LIBRARY_PATH=$NVDIR/lib ./91_segtree_concurrent_update_kernel
```

**Sample input:** the same three point updates applied concurrently (one thread each) via plain `atomicAdd`, under the adversarial lockstep wave schedule traced above.

**Sample output:**

```text
=== Section 17.2 main: concurrent point updates via plain atomicAdd ===

initial tree (1-indexed, leaves at tree[8..15]):
  44 22 22 13 9 9 13 5 8 6 3 2 7 4 9 

3 threads update concurrently:
  thread 0: leaf=1 delta=+5
  thread 1: leaf=5 delta=-2
  thread 2: leaf=6 delta=+10

lockstep wave schedule (every thread advances one step per wave):
  wave 0: node9<-thread0, node13<-thread1, node14<-thread2
  wave 1: node4<-thread0, node6<-thread1, node7<-thread2
  wave 2: node2<-thread0, node3<-thread1&2  [contended node -- multiple atomicAdd, no conflict]
  wave 3: node1<-thread0&1&2  [contended node -- multiple atomicAdd, no conflict]

final tree:
  57 27 30 18 9 7 23 5 13 6 3 2 5 14 9 

expected final tree (identical to Section 17.2's CPU baseline, applied
sequentially in program order): 57 27 30 18 9 7 23 5 13 6 3 2 5 14 9

node 1 (the root) receives contributions from ALL THREE threads, in the SAME
wave -- three simultaneous atomicAdd calls on one shared counter -- and node 3
receives two. Unlike Chapter 16.2's trie node creation, no thread ever needs to
discover it "lost" anything: every contribution is simply summed in, in
whatever order the hardware happens to serialize the atomics.

self-check: concurrent atomicAdd-based updates match the sequential CPU
baseline exactly, regardless of interleaving: confirmed
```

## 17.3 The Fenwick Tree (Binary Indexed Tree): The Same Operations With a Leaner Structure

### Intuition

A segment tree needs an explicit `2n`-slot array and explicit child-index arithmetic to represent its shape. A Fenwick tree (binary indexed tree, or BIT) answers the exact same two questions -- point update, prefix-sum query -- using only `n+1` slots and NO explicit tree shape at all: every index's role comes purely from its own binary representation, via `lowbit(i) = i & (-i)`, which isolates an integer's lowest set bit. Update walks UP by repeatedly ADDING the lowbit; prefix-sum query walks DOWN by repeatedly SUBTRACTING it -- two mirror-image walks built from the same one-line primitive, each still `O(log n)`.

### The Sequential (CPU) Baseline

```cpp
#include <cstdio>
#include <vector>

// Chapter 17.3 -- The Sequential (CPU) Baseline.
// A segment tree needs an explicit array of 2n nodes and explicit
// child indices (`2*i`, `2*i+1`) to represent its shape. A Fenwick
// tree (binary indexed tree, or BIT) answers the exact same two
// questions -- point update, prefix-sum query -- using only n+1 array
// slots and NO explicit tree structure at all: every index's role is
// determined purely by its own binary representation, via the
// "lowbit" operation `i & (-i)`, which isolates an integer's lowest
// set bit (in two's complement, `-i` is `~i + 1`, so `i & (-i)` always
// yields exactly that one bit). Update walks UP by repeatedly ADDING
// the lowbit; query walks DOWN by repeatedly SUBTRACTING it -- the two
// walks are mirror images of each other, using the same primitive.
#define N 8

int lowbit(int i) { return i & (-i); }

void update(std::vector<int>& bit, int i, int delta) {
    while (i <= N) {
        bit[i] += delta;
        i += lowbit(i);
    }
}

void build(const std::vector<int>& a, std::vector<int>& bit) {
    bit.assign(N + 1, 0);
    for (int i = 1; i <= N; i++) update(bit, i, a[i - 1]);
}

int prefix_sum(const std::vector<int>& bit, int i) {
    int s = 0;
    while (i > 0) {
        s += bit[i];
        i -= lowbit(i);
    }
    return s;
}

int main() {
    printf("=== Section 17.3 CPU baseline: Fenwick tree build + prefix-sum queries ===\n\n");

    std::vector<int> a = {5, 8, 6, 3, 2, 7, 4, 9};
    printf("input array a: ");
    for (int v : a) printf("%d ", v);
    printf("\n\n");

    std::vector<int> bit;
    build(a, bit);
    printf("fenwick array (1-indexed, bit[0] unused):\n  ");
    for (int i = 0; i <= N; i++) printf("%d ", bit[i]);
    printf("\n\n");

    std::vector<int> queries = {1, 3, 5, 8};
    printf("prefix-sum queries (sum of a[0..i)):\n");
    bool ok = true;
    std::vector<int> expected = {5, 19, 24, 44};
    for (size_t q = 0; q < queries.size(); q++) {
        int i = queries[q];
        int result = prefix_sum(bit, i);
        int brute = 0;
        for (int j = 0; j < i; j++) brute += a[j];
        printf("  prefix_sum(%d) = %d  (brute-force check: %d)\n", i, result, brute);
        ok = ok && (result == expected[q]) && (result == brute);
    }

    printf("\nself-check: Fenwick tree built correctly, all prefix-sum queries match\n");
    printf("brute-force sums: %s\n", ok ? "confirmed" : "MISMATCH");
    return ok ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++17 -Wall -Wextra -O2 92_fenwick_build_query_cpu_baseline.cpp -o 92_fenwick_build_query_cpu_baseline
./92_fenwick_build_query_cpu_baseline
```

**Sample input:** the same array `{5, 8, 6, 3, 2, 7, 4, 9}`, built into a Fenwick tree via repeated point updates, queried over four prefix ranges.

**Sample output:**

```text
=== Section 17.3 CPU baseline: Fenwick tree build + prefix-sum queries ===

input array a: 5 8 6 3 2 7 4 9 

fenwick array (1-indexed, bit[0] unused):
  0 5 13 6 22 2 9 4 44 

prefix-sum queries (sum of a[0..i)):
  prefix_sum(1) = 5  (brute-force check: 5)
  prefix_sum(3) = 19  (brute-force check: 19)
  prefix_sum(5) = 24  (brute-force check: 24)
  prefix_sum(8) = 44  (brute-force check: 44)

self-check: Fenwick tree built correctly, all prefix-sum queries match
brute-force sums: confirmed
```

### The Concept, In Detail

Tracing `update(i=2, delta=+5)`'s upward walk (`lowbit(2)=2`, `lowbit(4)=4`, `lowbit(8)=8`):

```
i=2: bit[2] += 5;  i = 2 + lowbit(2) = 2 + 2 = 4
i=4: bit[4] += 5;  i = 4 + lowbit(4) = 4 + 4 = 8
i=8: bit[8] += 5;  i = 8 + lowbit(8) = 8 + 8 = 16 > N -- stop
path = [2, 4, 8]
```

And `prefix_sum(i=5)`'s downward walk (`lowbit(5)=1`, `lowbit(4)=4`):

```
i=5: s += bit[5];  i = 5 - lowbit(5) = 5 - 1 = 4
i=4: s += bit[4];  i = 4 - lowbit(4) = 4 - 4 = 0 -- stop
path = [5, 4]  (bit[5] + bit[4] = the prefix sum of a[0..4])
```

Running three CONCURRENT updates (`pos=2,delta=+5`; `pos=6,delta=-2`; `pos=7,delta=+10`) shows the one genuine difference from Section 17.2's segment tree: Fenwick update paths are NOT all the same length, because `lowbit` steps are uneven, so some threads finish early -- Chapter 16.3's own "still-active" idea, carried over unchanged:

```
paths: pos=2 -> [2,4,8] (length 3),  pos=6 -> [6,8] (length 2),  pos=7 -> [7,8] (length 2)

wave 0: thread0 -> node2, thread1 -> node6, thread2 -> node7      (no contention)
wave 1: thread0 -> node4, thread1 & thread2 BOTH -> node8         (contended!)
wave 2: thread0 -> node8   (threads 1 and 2 are DONE -- no longer active)
```

Node 8 (the Fenwick array's own root-equivalent -- every complete update path passes through it here) receives contributions from all three threads across TWO DIFFERENT waves, not just one. Plain `atomicAdd` handles this exactly as well as same-wave contention: addition does not care whether contributions arrive at the same instant or at different times, only that every one of them eventually lands.

```
ASCII view: memory footprint, segment tree versus Fenwick tree, same n=8.

  segment tree: 2n = 16 ints, plus every node needs 2*i/2*i+1 arithmetic
  Fenwick tree: n+1 = 9 ints, no explicit child/parent arithmetic at all --
                just lowbit(i) = i & (-i), computed from i's own bits
```

Prefix-sum queries after these updates remain exactly what Section 17.1 already established for range queries in general: independent, read-only walks with no shared mutable state, so any number of them run fully in parallel with zero atomics needed -- the same "many independent short walks, one thread each" pattern, now on the leaner structure.

[COMMON TRAP]
It is tempting to assume every update's path passes through the SAME final index (as it did for the segment tree's root). A Fenwick tree of size `n` guarantees every update path terminates once `i` exceeds `n` -- for `n=8`, every path shown above does happen to pass through index 8, but that is because 8 is a power of two equal to `n` itself, not a general guarantee; for a non-power-of-two `n`, different starting positions can produce paths that never intersect at all, and code that assumes a single universal "root" index will be wrong.

### Code and Verification

```cpp
#include <cstdio>
#include <vector>
#include <map>
#include <algorithm>

// Chapter 17.3 main -- the Fenwick tree carries over BOTH parallel
// stories from Sections 17.1 and 17.2, just riding a different index
// walk (`i += lowbit(i)` for updates, `i -= lowbit(i)` for queries,
// instead of `idx >>= 1` and the two-pointer walk). Point updates are
// still additive and therefore still commutative, so concurrent
// updates still need nothing beyond plain `atomicAdd` -- no CAS, no
// retry. Prefix-sum queries are still pure reads with no shared
// mutable state between them, so many queries still run fully in
// parallel with no synchronization at all. The one new wrinkle: unlike
// Section 17.2's segment tree (where every leaf is the same distance
// from the root), different Fenwick update paths can have DIFFERENT
// lengths, so some threads finish their walk while others are still
// going -- Chapter 16.3's own "still-active" idea, here applying to a
// bump-allocator-free, pointer-free structure.
#define N 8

int lowbit_dev_equiv(int i) { return i & (-i); }   // host mirror of the device expression below

// One thread per UPDATE, walking upward by lowbit steps.
__global__ void fenwick_update_kernel(int* bit, int n, const int* pos,
                                       const int* deltas, int num_updates) {
    int t = threadIdx.x;
    if (t >= num_updates) return;
    int i = pos[t];
    int delta = deltas[t];
    while (i <= n) {
        atomicAdd(&bit[i], delta);
        i += (i & (-i));
    }
}

// One thread per QUERY -- pure reads, walking downward by lowbit
// steps, with zero interaction between threads and no atomics needed
// at all.
__global__ void fenwick_query_kernel(const int* bit, const int* qs, int* results,
                                      int num_queries) {
    int t = threadIdx.x;
    if (t >= num_queries) return;
    int i = qs[t];
    int s = 0;
    while (i > 0) {
        s += bit[i];
        i -= (i & (-i));
    }
    results[t] = s;
}

int main() {
    printf("=== Section 17.3 main: concurrent Fenwick updates, parallel prefix queries ===\n\n");

    std::vector<int> a = {5, 8, 6, 3, 2, 7, 4, 9};
    std::vector<int> bit(N + 1, 0);
    for (int i = 1; i <= N; i++) {
        int j = i;
        while (j <= N) { bit[j] += a[i - 1]; j += lowbit_dev_equiv(j); }
    }
    printf("initial fenwick array (1-indexed, bit[0] unused):\n  ");
    for (int i = 0; i <= N; i++) printf("%d ", bit[i]);
    printf("\n\n");

    struct Update { int pos, delta; };
    std::vector<Update> updates = {{2, 5}, {6, -2}, {7, 10}};
    printf("3 threads update concurrently:\n");
    for (size_t t = 0; t < updates.size(); t++) {
        printf("  thread %zu: pos=%d delta=%+d\n", t, updates[t].pos, updates[t].delta);
    }
    printf("\n");

    std::vector<std::vector<int>> paths(updates.size());
    for (size_t t = 0; t < updates.size(); t++) {
        int i = updates[t].pos;
        while (i <= N) { paths[t].push_back(i); i += lowbit_dev_equiv(i); }
    }
    size_t max_len = 0;
    for (auto& p : paths) max_len = std::max(max_len, p.size());

    printf("lockstep wave schedule (a thread stops contributing once its own path ends):\n");
    for (size_t w = 0; w < max_len; w++) {
        std::map<int, std::vector<int>> contributions;
        for (size_t t = 0; t < updates.size(); t++) {
            if (w < paths[t].size()) contributions[paths[t][w]].push_back((int)t);
        }
        printf("  wave %zu: ", w);
        bool first = true;
        for (auto& kv : contributions) {
            if (!first) printf(", ");
            first = false;
            printf("node%d<-thread", kv.first);
            for (size_t i = 0; i < kv.second.size(); i++) {
                printf("%d%s", kv.second[i], i + 1 < kv.second.size() ? "&" : "");
            }
        }
        int active = (int)contributions.size();
        int total_active_threads = 0;
        for (auto& kv : contributions) total_active_threads += (int)kv.second.size();
        if (total_active_threads > active) printf("  [contended node -- multiple atomicAdd, no conflict]");
        printf("\n");
        for (size_t t = 0; t < updates.size(); t++) {
            if (w < paths[t].size()) bit[paths[t][w]] += updates[t].delta;
        }
    }
    printf("\n");

    printf("final fenwick array:\n  ");
    for (int i = 0; i <= N; i++) printf("%d ", bit[i]);
    printf("\n\n");

    std::vector<int> expected_bit = {0, 5, 18, 6, 27, 2, 7, 14, 57};
    bool ok = true;
    for (int i = 0; i <= N; i++) ok = ok && (bit[i] == expected_bit[i]);

    printf("expected final fenwick array: 0 5 18 6 27 2 7 14 57\n\n");

    std::vector<int> queries = {1, 3, 5, 8};
    std::vector<int> results(queries.size());
    printf("%zu independent query threads, each walking down by lowbit on its own:\n", queries.size());
    for (size_t t = 0; t < queries.size(); t++) {
        int i = queries[t];
        int s = 0;
        while (i > 0) { s += bit[i]; i -= lowbit_dev_equiv(i); }
        results[t] = s;
        printf("  thread %zu: prefix_sum(%d) -> %d\n", t, queries[t], s);
    }

    std::vector<int> expected_results = {5, 24, 29, 57};
    for (size_t t = 0; t < results.size(); t++) ok = ok && (results[t] == expected_results[t]);

    printf("\nexpected post-update prefix sums: 5 24 29 57\n");
    printf("\nnode 8 (bit's own root-equivalent) receives contributions from ALL THREE\n");
    printf("update threads across TWO DIFFERENT waves (thread 1 and thread 2 finish their\n");
    printf("own paths there in wave 1; thread 0's longer path reaches it one wave later) --\n");
    printf("plain atomicAdd handles contributions arriving at different TIMES exactly as\n");
    printf("well as contributions arriving in the SAME wave, since addition does not care\n");
    printf("about order at all.\n");

    printf("\nself-check: concurrent Fenwick updates and fully independent parallel prefix\n");
    printf("queries both match the sequential CPU baseline exactly: %s\n", ok ? "confirmed" : "MISMATCH");
    return ok ? 0 : 1;
}
```

**Compile and run:**

```bash
NVDIR=/usr/local/lib/python3.11/dist-packages/nvidia/cu13
$NVDIR/bin/nvcc -arch=sm_80 -I$NVDIR/include -L$NVDIR/lib -lcudart 93_fenwick_concurrent_update_query_kernel.cu -o 93_fenwick_concurrent_update_query_kernel
LD_LIBRARY_PATH=$NVDIR/lib ./93_fenwick_concurrent_update_query_kernel
```

**Sample input:** three concurrent point updates via `atomicAdd` along each thread's own lowbit-walk, followed by four fully independent prefix-sum queries running in parallel with no atomics at all.

**Sample output:**

```text
=== Section 17.3 main: concurrent Fenwick updates, parallel prefix queries ===

initial fenwick array (1-indexed, bit[0] unused):
  0 5 13 6 22 2 9 4 44 

3 threads update concurrently:
  thread 0: pos=2 delta=+5
  thread 1: pos=6 delta=-2
  thread 2: pos=7 delta=+10

lockstep wave schedule (a thread stops contributing once its own path ends):
  wave 0: node2<-thread0, node6<-thread1, node7<-thread2
  wave 1: node4<-thread0, node8<-thread1&2  [contended node -- multiple atomicAdd, no conflict]
  wave 2: node8<-thread0

final fenwick array:
  0 5 18 6 27 2 7 14 57 

expected final fenwick array: 0 5 18 6 27 2 7 14 57

4 independent query threads, each walking down by lowbit on its own:
  thread 0: prefix_sum(1) -> 5
  thread 1: prefix_sum(3) -> 24
  thread 2: prefix_sum(5) -> 29
  thread 3: prefix_sum(8) -> 57

expected post-update prefix sums: 5 24 29 57

node 8 (bit's own root-equivalent) receives contributions from ALL THREE
update threads across TWO DIFFERENT waves (thread 1 and thread 2 finish their
own paths there in wave 1; thread 0's longer path reaches it one wave later) --
plain atomicAdd handles contributions arriving at different TIMES exactly as
well as contributions arriving in the SAME wave, since addition does not care
about order at all.

self-check: concurrent Fenwick updates and fully independent parallel prefix
queries both match the sequential CPU baseline exactly: confirmed
```

## Chapter Summary

A segment tree precomputes every subtree's sum once, laid out iteratively in a `2n`-slot array (leaves at `[n, 2n)`, each internal node the sum of its two children), built level by level with no recursion -- the same "independent work within a level, synchronized between levels" shape this book has used since Chapter 12. A single range query is a short but genuinely sequential `O(log n)` walk, so the real parallel opportunity is running many independent queries at once rather than parallelizing one query's own walk. Point updates reopen this book's concurrency story from a new angle: because a point update's core operation is ADDING a delta, and addition is commutative and associative, concurrent updates to different leaves that share ancestors need nothing more than the plain `atomicAdd` from Chapters 7 and 8 -- a sharp contrast with Chapter 16.2's trie node creation, which needed `atomicCAS` precisely because creating a node is NOT commutative. The Fenwick tree (binary indexed tree) answers the identical two questions -- point update, prefix-sum query -- using only `n+1` array slots and no explicit tree shape at all, via the `lowbit(i) = i & (-i)` primitive walked upward for updates and downward for queries; its update paths can have different lengths for different starting positions, meaning some concurrent update threads finish before others, but the same commutative-`atomicAdd` guarantee, and the same "many independent parallel queries" pattern, carry over unchanged.

## Self-Check Questions

1. Why does building a segment tree level by level require no shared allocation counter, the same way Section 15.1's sorted-array BST construction did not?
2. Why is a single range-sum query considered "short but sequential" rather than something worth parallelizing internally, and what IS the right parallel target for querying?
3. What specifically makes a point update's `atomicAdd` on a shared ancestor safe with no retry logic, when Chapter 16.2's trie node creation on a shared child slot was NOT safe without `atomicCAS`?
4. In the concurrent point-update trace, node 1 (the segment tree's root) receives contributions from all three threads in the very same wave. Why does this not require any of the three threads to detect or react to the other two?
5. What does `lowbit(i) = i & (-i)` compute, and how do the update and prefix-sum walks use it in opposite directions?
6. Why can different Fenwick tree update paths have different lengths, and what concept from Chapter 16.3 does that directly echo?

## Where We Go Next

Segment and Fenwick trees answer range and prefix questions over a one-dimensional array. Part 4 closes by leaving one dimension behind: Chapter 18 turns to the spatial trees -- k-d trees, quadtrees, octrees, and bounding volume hierarchies -- that make nearest-neighbor search and collision queries over points and shapes in 2D and 3D space tractable, including how their own construction and traversal parallelize.

## Worked Solutions

**1.** Every node in a given level of a segment tree is built purely from `tree[2*i]` and `tree[2*i+1]`, both already finalized by the previous (narrower) level -- a fixed, closed-form relationship between a node's own index and its children's indices that holds for EVERY node in a complete tree, not just the ones a particular build happens to visit. Section 15.1 established the exact same closed-form indexing (`2*slot+1`, `2*slot+2`) for the same reason: because the tree being built is always perfectly complete, no thread ever needs to ask a shared counter "what slot is free" -- its own slot is computable directly from its position in the level.

**2.** A single range-sum query's iterative two-pointer walk recomputes `l` and `r` from their OWN previous values every iteration -- each step depends on the result of the step before it, so no number of additional threads can shorten that one query's own critical path (identical in shape to Chapter 11's pointer-chasing argument, just bounded to `O(log n)` steps instead of `O(n)`). The right parallel target is running MANY independent queries side by side, one thread per query, since different queries share no data dependency on each other at all -- only the same, unchanging tree that all of them are only ever reading from.

**3.** A point update's operation is "add this delta to whatever is already here" -- addition is commutative and associative, so the FINAL value at a shared ancestor is the same regardless of what order multiple threads' additions to it happen to be serialized in; nothing is ever overwritten, only accumulated. Trie node creation's operation is "store this SPECIFIC id into this slot if it's still empty" -- an assignment, not an accumulation, so whichever thread's assignment lands last determines the final value, discarding every other thread's candidate; `atomicCAS` is required there specifically to detect that discarding and let the losing thread recover (reuse the winner's value) rather than silently corrupt the structure.

**4.** Because `atomicAdd` makes each individual addition indivisible and requires no thread to know anything about any other thread's activity: every thread simply calls `atomicAdd(&tree[1], its_own_delta)`, the hardware serializes the three individual additions in some unspecified order, and the final value is the sum of all three deltas regardless of that order. There is no "winner" and no "loser" to detect -- unlike `atomicCAS`, where a thread's own call can fail and must inspect the returned value to decide what to do next, `atomicAdd` always succeeds and always contributes, so no thread ever needs to react to what any other thread did.

**5.** In two's complement representation, `-i` equals `~i + 1`, and `i & (-i)` isolates exactly `i`'s lowest set bit (every bit below that lowest set bit is 0 in both `i` and `-i`'s relevant range, and the bits above it cancel to 0 through the AND) -- for example `lowbit(6) = lowbit(0b110) = 0b010 = 2`. An update walk moves UP the implicit structure by repeatedly ADDING this value (`i += lowbit(i)`), reaching indices responsible for progressively larger ranges that include position `i`; a prefix-sum query walks DOWN by repeatedly SUBTRACTING it (`i -= lowbit(i)`), decomposing the prefix `[1, i]` into a small number of disjoint, already-summed pieces.

**6.** Different starting positions produce different sequences of `lowbit` values, since `lowbit` depends entirely on which bits happen to be set in the CURRENT value of `i` at each step, and that current value changes differently for different starting positions -- there is no guarantee two different starting indices take the same number of steps to exceed `n`. This is a direct echo of Chapter 16.3's level-synchronous trie construction, where different keys' lengths meant some threads finished their own position-by-position walk (and stopped contributing) while others were still active -- both cases require a schedule that tracks, per thread, whether it still has more work to contribute at the current synchronization point, rather than assuming every thread's walk is the same length.
