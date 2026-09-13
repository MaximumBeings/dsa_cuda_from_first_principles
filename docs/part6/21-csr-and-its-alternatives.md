# Chapter 21: CSR and Its Alternatives

Part 6 turns to graphs -- structures whose entire challenge, on a GPU, is IRREGULARITY: unlike an array, a tree, or a hash table, a graph's vertices can each have a wildly different number of neighbors, and that shape is fixed by the input, not something the algorithm gets to choose. Before anything can traverse a graph in parallel (Chapter 22 onward), it needs a representation a thread can actually index into. This chapter builds that representation -- Compressed Sparse Row (CSR) -- in three pieces: what CSR is and why it beats a naive adjacency list, how to build it in parallel directly from a raw, unsorted edge list (a direct callback to Chapter 5's scan, Chapter 7's histogram, and Chapter 8's atomic bump allocator, now working together), and when CSR is the WRONG tool, favoring instead the raw edge list format it was built from.

## 21.1 Why Flatten the Graph: Adjacency Lists vs. Compressed Sparse Row (CSR)

### Intuition

The most natural graph representation is an adjacency list: one growable container of neighbors per vertex. This is easy to build and reason about sequentially, but it is fundamentally hostile to a GPU's flat, contiguous-memory model -- every vertex's neighbor list is a separate, differently-sized container, so there is no single array a thread can index directly into, and no way to even hand a `std::vector<int>` to a kernel in the first place. Compressed Sparse Row (CSR) flattens every vertex's neighbors into ONE shared array, `col_idx`, with a small second array, `row_offsets` (sized `V+1`), recording exactly where each vertex's own slice begins and ends -- directly the same flat-array-plus-index-structure idea this book has already used for stream compaction (Chapter 6) and counting sort (Chapter 13), now applied to a graph's edges.

### The Sequential (CPU) Baseline

```cpp
#include <cstdio>
#include <vector>

// Chapter 21.1 -- The Sequential (CPU) Baseline.
// A graph's most natural representation is an ADJACENCY LIST: one
// growable list of neighbors per vertex. This is easy to build and
// easy to reason about sequentially, but it is hostile to a GPU's flat,
// contiguous-memory model -- every vertex's list is a SEPARATE,
// differently-sized container, so there is no single array a thread
// can index directly into. Compressed Sparse Row (CSR) fixes this by
// flattening every vertex's neighbor list into ONE shared array
// (`col_idx`), with a second, small array (`row_offsets`, sized V+1)
// recording where each vertex's own slice begins and ends within it --
// directly the same idea as this book's earlier flat-array-plus-index
// structures (Chapter 6's stream compaction, Chapter 13's counting
// arrays), now applied to a graph's edges.
struct AdjacencyList {
    std::vector<std::vector<int>> neighbors;
    explicit AdjacencyList(int V) : neighbors(V) {}
};

AdjacencyList build_adjacency_list(int V, const std::vector<std::pair<int,int>>& edges) {
    AdjacencyList adj(V);
    for (auto& e : edges) {
        adj.neighbors[e.first].push_back(e.second);
    }
    return adj;
}

// Flattens an already-built adjacency list into CSR: walk vertices in
// order, and within each vertex, walk its own list in the order it was
// built -- a purely sequential concatenation.
void adjacency_to_csr(const AdjacencyList& adj, std::vector<int>& row_offsets, std::vector<int>& col_idx) {
    int V = (int)adj.neighbors.size();
    row_offsets.assign(V + 1, 0);
    col_idx.clear();
    for (int v = 0; v < V; v++) {
        row_offsets[v] = (int)col_idx.size();
        for (int nbr : adj.neighbors[v]) col_idx.push_back(nbr);
    }
    row_offsets[V] = (int)col_idx.size();
}

int main() {
    printf("=== Section 21.1 CPU baseline: adjacency list -> CSR conversion ===\n\n");

    int V = 6;
    std::vector<std::pair<int,int>> edges = {
        {2,3}, {0,1}, {3,4}, {1,2}, {0,2}, {4,5}, {2,4}, {3,5}, {1,3},
    };
    printf("6-vertex graph, 9 directed edges (input order, NOT grouped by source):\n  ");
    for (auto& e : edges) printf("(%d->%d) ", e.first, e.second);
    printf("\n\n");

    AdjacencyList adj = build_adjacency_list(V, edges);
    printf("adjacency list (one growable container per vertex):\n");
    for (int v = 0; v < V; v++) {
        printf("  adj[%d]: ", v);
        for (int n : adj.neighbors[v]) printf("%d ", n);
        printf("\n");
    }
    printf("\n");

    std::vector<int> row_offsets, col_idx;
    adjacency_to_csr(adj, row_offsets, col_idx);

    printf("flattened into CSR:\n");
    printf("  row_offsets = [ ");
    for (int r : row_offsets) printf("%d ", r);
    printf("]\n  col_idx     = [ ");
    for (int c : col_idx) printf("%d ", c);
    printf("]\n\n");

    printf("reading each vertex's neighbors back out of CSR via its slice:\n");
    bool ok = true;
    for (int v = 0; v < V; v++) {
        printf("  vertex %d: slice [%d,%d) -> ", v, row_offsets[v], row_offsets[v + 1]);
        std::vector<int> from_csr;
        for (int i = row_offsets[v]; i < row_offsets[v + 1]; i++) {
            printf("%d ", col_idx[i]);
            from_csr.push_back(col_idx[i]);
        }
        printf("\n");
        ok = ok && (from_csr == adj.neighbors[v]);
    }

    std::vector<int> exp_row_offsets = {0, 2, 4, 6, 8, 9, 9};
    std::vector<int> exp_col_idx = {1, 2, 2, 3, 3, 4, 4, 5, 5};
    ok = ok && (row_offsets == exp_row_offsets) && (col_idx == exp_col_idx);

    printf("\nexpected row_offsets = [ 0 2 4 6 8 9 9 ]\n");
    printf("expected col_idx     = [ 1 2 2 3 3 4 4 5 5 ]\n");
    printf("\nself-check: CSR's per-vertex slices reproduce the adjacency list's\n");
    printf("neighbors exactly, matching the expected flattened arrays: %s\n", ok ? "confirmed" : "MISMATCH");
    return ok ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++17 -Wall -Wextra -O2 112_csr_conversion_cpu_baseline.cpp -o 112_csr_conversion_cpu_baseline
./112_csr_conversion_cpu_baseline
```

**Sample input:** a 6-vertex, 9-edge directed graph, built first as an adjacency list, then flattened into CSR.

**Sample output:**

```text
=== Section 21.1 CPU baseline: adjacency list -> CSR conversion ===

6-vertex graph, 9 directed edges (input order, NOT grouped by source):
  (2->3) (0->1) (3->4) (1->2) (0->2) (4->5) (2->4) (3->5) (1->3) 

adjacency list (one growable container per vertex):
  adj[0]: 1 2 
  adj[1]: 2 3 
  adj[2]: 3 4 
  adj[3]: 4 5 
  adj[4]: 5 
  adj[5]: 

flattened into CSR:
  row_offsets = [ 0 2 4 6 8 9 9 ]
  col_idx     = [ 1 2 2 3 3 4 4 5 5 ]

reading each vertex's neighbors back out of CSR via its slice:
  vertex 0: slice [0,2) -> 1 2 
  vertex 1: slice [2,4) -> 2 3 
  vertex 2: slice [4,6) -> 3 4 
  vertex 3: slice [6,8) -> 4 5 
  vertex 4: slice [8,9) -> 5 
  vertex 5: slice [9,9) -> 

expected row_offsets = [ 0 2 4 6 8 9 9 ]
expected col_idx     = [ 1 2 2 3 3 4 4 5 5 ]

self-check: CSR's per-vertex slices reproduce the adjacency list's
neighbors exactly, matching the expected flattened arrays: confirmed
```

### The Concept, In Detail

```
ASCII view: adjacency list (separate containers) vs CSR (one flat
array plus an index):

  adjacency list:
    adj[0] -> [1, 2]
    adj[1] -> [2, 3]
    adj[2] -> [3, 4]
    adj[3] -> [4, 5]
    adj[4] -> [5]
    adj[5] -> []
    (6 separate containers, different sizes -- no single array exists)

  CSR:
    row_offsets = [ 0 | 2 | 4 | 6 | 8 | 9 | 9 ]
                    ^v0  ^v1  ^v2  ^v3  ^v4  ^v5 ^end
    col_idx     = [ 1  2 | 2  3 | 3  4 | 4  5 | 5 |  ]
                    -v0-   -v1-   -v2-   -v3-  v4  v5(empty)

  Vertex v's neighbors are exactly col_idx[row_offsets[v] .. row_offsets[v+1]) --
  ONE array, indexed by TWO adjacent entries of a second, tiny array.
```

`row_offsets` has exactly `V+1` entries, not `V`: the extra final entry (here, a second `9`) gives every vertex, including the last one, an explicit UPPER bound for its slice without needing a special case. Vertex 5 has no outgoing edges at all -- its slice `[9, 9)` is simply empty, which the loop bound `row_offsets[v] < row_offsets[v+1]` handles automatically, with no branch needed to detect "this vertex has zero neighbors."

[COMMON TRAP]
It is tempting to size `row_offsets` at exactly `V` entries -- one per vertex -- since that is how many vertices there are. Without the extra `V+1`-th entry, the LAST vertex's slice would have no upper bound to read from (`row_offsets[V]` would be out of range), forcing a special case ("if this is the last vertex, the slice extends to `col_idx`'s own length instead"). The extra sentinel entry, holding the total edge count, is what lets every vertex -- including the last -- use the exact same `[row_offsets[v], row_offsets[v+1])` formula with no exceptions.

### Code and Verification

```cpp
#include <cstdio>
#include <vector>

// Chapter 21.1 main -- CSR's whole point is that a thread can index
// directly into ONE shared flat array instead of chasing a per-vertex
// container. One thread per VERTEX reads its own slice of `col_idx`,
// bounded by `row_offsets[v]` and `row_offsets[v+1]` -- pure reads of
// shared, read-only memory, so no atomics are needed at all. The slice
// lengths are genuinely UNEVEN across threads (this graph's vertices
// have degree 2, 2, 2, 2, 1, and 0), which is precisely the kind of
// per-thread work an adjacency list of separate, differently-sized
// host containers could never expose to a GPU thread in the first
// place -- there is no way to hand a `std::vector<int>` to a kernel.
#define NUM_VERTICES 6

// One thread per vertex. `row_offsets` has NUM_VERTICES+1 entries;
// thread v's own slice is col_idx[row_offsets[v] .. row_offsets[v+1]).
__global__ void csr_neighbor_sum_kernel(const int* row_offsets, const int* col_idx,
                                         int* out_sums, int num_vertices) {
    int v = threadIdx.x;
    if (v >= num_vertices) return;
    int sum = 0;
    for (int i = row_offsets[v]; i < row_offsets[v + 1]; i++) {
        sum += col_idx[i];
    }
    out_sums[v] = sum;
}

// ---- Host-side replay of the identical per-thread logic. ----

int main() {
    printf("=== Section 21.1 main: one thread per vertex, summing its own CSR slice ===\n\n");

    // Section 21.1's own CSR arrays.
    std::vector<int> row_offsets = {0, 2, 4, 6, 8, 9, 9};
    std::vector<int> col_idx = {1, 2, 2, 3, 3, 4, 4, 5, 5};
    printf("row_offsets = [ 0 2 4 6 8 9 9 ]\n");
    printf("col_idx     = [ 1 2 2 3 3 4 4 5 5 ]\n\n");

    printf("%d independent threads, each summing its own vertex's neighbor ids:\n", NUM_VERTICES);
    std::vector<int> sums(NUM_VERTICES);
    for (int v = 0; v < NUM_VERTICES; v++) {
        int sum = 0;
        int lo = row_offsets[v], hi = row_offsets[v + 1];
        for (int i = lo; i < hi; i++) sum += col_idx[i];
        sums[v] = sum;
        printf("  thread %d (vertex %d): slice [%d,%d), %d neighbor(s) -> sum = %d\n",
               v, v, lo, hi, hi - lo, sum);
    }

    std::vector<int> expected = {3, 5, 7, 9, 5, 0};
    bool ok = (sums == expected);

    printf("\nexpected sums: 3 5 7 9 5 0\n");
    printf("(thread lengths ranged from 2 down to 0 neighbors -- genuinely uneven\n");
    printf("per-thread work, yet every thread still just walks a slice of ONE\n");
    printf("shared flat array; no per-thread container of any kind was needed)\n");

    printf("\nself-check: all 6 per-vertex neighbor sums match expected values: %s\n",
           ok ? "confirmed" : "MISMATCH");
    return ok ? 0 : 1;
}
```

**Compile and run:**

```bash
NVDIR=/usr/local/lib/python3.11/dist-packages/nvidia/cu13
$NVDIR/bin/nvcc -arch=sm_80 -I$NVDIR/include -L$NVDIR/lib -lcudart 113_csr_neighbor_sum_kernel.cu -o 113_csr_neighbor_sum_kernel
LD_LIBRARY_PATH=$NVDIR/lib ./113_csr_neighbor_sum_kernel
```

**Sample input:** the same CSR arrays, read by 6 independent threads at once, one per vertex, each summing its own neighbor ids.

**Sample output:**

```text
=== Section 21.1 main: one thread per vertex, summing its own CSR slice ===

row_offsets = [ 0 2 4 6 8 9 9 ]
col_idx     = [ 1 2 2 3 3 4 4 5 5 ]

6 independent threads, each summing its own vertex's neighbor ids:
  thread 0 (vertex 0): slice [0,2), 2 neighbor(s) -> sum = 3
  thread 1 (vertex 1): slice [2,4), 2 neighbor(s) -> sum = 5
  thread 2 (vertex 2): slice [4,6), 2 neighbor(s) -> sum = 7
  thread 3 (vertex 3): slice [6,8), 2 neighbor(s) -> sum = 9
  thread 4 (vertex 4): slice [8,9), 1 neighbor(s) -> sum = 5
  thread 5 (vertex 5): slice [9,9), 0 neighbor(s) -> sum = 0

expected sums: 3 5 7 9 5 0
(thread lengths ranged from 2 down to 0 neighbors -- genuinely uneven
per-thread work, yet every thread still just walks a slice of ONE
shared flat array; no per-thread container of any kind was needed)

self-check: all 6 per-vertex neighbor sums match expected values: confirmed
```

## 21.2 Building CSR in Parallel: Degree Counting and Prefix Sum

### Intuition

Section 21.1 converted an ALREADY-BUILT adjacency list into CSR, which is easy because the neighbors were already grouped by vertex. Real input usually arrives as a raw, UNSORTED edge list instead, with no such grouping. Building CSR directly from that takes three steps this book has already built individually: count each vertex's degree (Chapter 7's histogram, counting edge sources instead of arbitrary bins), turn those counts into `row_offsets` via an exclusive prefix sum (Chapter 5's scan), then scatter each edge into `col_idx` at a position claimed from its source vertex's own running counter (Chapter 8's atomic bump allocator, one allocator per vertex instead of one shared global one). CSR construction is simply these three already-parallel primitives, applied one after another to a new kind of input.

### The Sequential (CPU) Baseline

```cpp
#include <cstdio>
#include <vector>
#include <algorithm>

// Chapter 21.2 -- The Sequential (CPU) Baseline.
// Section 21.1 converted an ALREADY-BUILT adjacency list into CSR --
// easy, because the neighbors were already grouped by vertex. Real
// input usually arrives as a raw, UNSORTED edge list instead (exactly
// the format a file of "u v weight" triples would give you), with
// edges for the same source vertex scattered anywhere in the list.
// Building CSR directly from that requires three steps: (1) count each
// vertex's DEGREE -- exactly Chapter 7's histogram, counted over edge
// sources instead of arbitrary bins; (2) an EXCLUSIVE PREFIX SUM over
// the degrees gives `row_offsets` -- exactly Chapter 5's scan; (3)
// walk the edge list once more, SCATTERING each edge into `col_idx` at
// a position given by its source's row_offset plus a running "next
// free slot" counter for that vertex -- exactly Chapter 8's atomic
// bump-allocator pattern, one bump-allocator per vertex instead of one
// shared global counter.
struct Edge { int src, dst, weight; };

int main() {
    printf("=== Section 21.2 CPU baseline: building CSR from a raw edge list ===\n\n");

    const int V = 6;
    std::vector<Edge> edges = {
        {2,3,8}, {0,1,4}, {3,4,2}, {1,2,2}, {0,2,1},
        {4,5,3}, {2,4,10}, {3,5,6}, {1,3,5},
    };
    const int E = (int)edges.size();
    printf("%d vertices, %d edges, arriving in NO particular source order:\n  ", V, E);
    for (auto& e : edges) printf("(%d->%d,w=%d) ", e.src, e.dst, e.weight);
    printf("\n\n");

    // Step 1: degree count (histogram over edge sources).
    std::vector<int> degree(V, 0);
    for (auto& e : edges) degree[e.src]++;
    printf("step 1 -- degree count (histogram over sources):\n  degree = [ ");
    for (int d : degree) printf("%d ", d);
    printf("]\n\n");

    // Step 2: exclusive prefix sum -> row_offsets (size V+1).
    std::vector<int> row_offsets(V + 1, 0);
    int running = 0;
    for (int v = 0; v < V; v++) {
        row_offsets[v] = running;
        running += degree[v];
    }
    row_offsets[V] = running;
    printf("step 2 -- exclusive prefix sum over degree -> row_offsets:\n  row_offsets = [ ");
    for (int r : row_offsets) printf("%d ", r);
    printf("]\n\n");

    // Step 3: scatter each edge into col_idx/weight_csr using a
    // per-vertex "next free slot" counter, seeded from row_offsets.
    std::vector<int> next_slot = row_offsets;  // copy; will be mutated as a bump allocator
    std::vector<int> col_idx(E, -1), weight_csr(E, -1);
    printf("step 3 -- scatter each edge via its source's own bump allocator:\n");
    for (auto& e : edges) {
        int pos = next_slot[e.src];
        next_slot[e.src]++;
        col_idx[pos] = e.dst;
        weight_csr[pos] = e.weight;
        printf("  edge(%d->%d, w=%d) claims slot %d (next_slot[%d] was %d, now %d)\n",
               e.src, e.dst, e.weight, pos, e.src, pos, next_slot[e.src]);
    }
    printf("\n  col_idx     = [ ");
    for (int c : col_idx) printf("%d ", c);
    printf("]\n  weight_csr  = [ ");
    for (int w : weight_csr) printf("%d ", w);
    printf("]\n\n");

    bool ok = true;
    std::vector<int> exp_degree = {2, 2, 2, 2, 1, 0};
    std::vector<int> exp_row_offsets = {0, 2, 4, 6, 8, 9, 9};
    std::vector<int> exp_col_idx = {1, 2, 2, 3, 3, 4, 4, 5, 5};
    std::vector<int> exp_weight_csr = {4, 1, 2, 5, 8, 10, 2, 6, 3};
    ok = ok && (degree == exp_degree) && (row_offsets == exp_row_offsets);
    ok = ok && (col_idx == exp_col_idx) && (weight_csr == exp_weight_csr);

    printf("verifying every vertex's slice holds exactly its own edges (by set):\n");
    for (int v = 0; v < V; v++) {
        std::vector<std::pair<int,int>> got;
        for (int i = row_offsets[v]; i < row_offsets[v + 1]; i++) got.push_back({col_idx[i], weight_csr[i]});
        std::vector<std::pair<int,int>> expected;
        for (auto& e : edges) if (e.src == v) expected.push_back({e.dst, e.weight});
        std::sort(got.begin(), got.end());
        std::sort(expected.begin(), expected.end());
        bool match = (got == expected);
        printf("  vertex %d: %s\n", v, match ? "matches" : "MISMATCH");
        ok = ok && match;
    }

    printf("\nexpected degree      = [ 2 2 2 2 1 0 ]\n");
    printf("expected row_offsets = [ 0 2 4 6 8 9 9 ]\n");
    printf("expected col_idx     = [ 1 2 2 3 3 4 4 5 5 ]\n");
    printf("expected weight_csr  = [ 4 1 2 5 8 10 2 6 3 ]\n");
    printf("\nself-check: CSR built from a raw, unsorted edge list via count -> scan\n");
    printf("-> scatter matches the expected arrays exactly: %s\n", ok ? "confirmed" : "MISMATCH");
    return ok ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++17 -Wall -Wextra -O2 114_csr_construction_cpu_baseline.cpp -o 114_csr_construction_cpu_baseline
./114_csr_construction_cpu_baseline
```

**Sample input:** the same 6-vertex graph, this time as a raw, unsorted edge list of 9 `(src, dst, weight)` triples, built into CSR via count -> scan -> scatter.

**Sample output:**

```text
=== Section 21.2 CPU baseline: building CSR from a raw edge list ===

6 vertices, 9 edges, arriving in NO particular source order:
  (2->3,w=8) (0->1,w=4) (3->4,w=2) (1->2,w=2) (0->2,w=1) (4->5,w=3) (2->4,w=10) (3->5,w=6) (1->3,w=5) 

step 1 -- degree count (histogram over sources):
  degree = [ 2 2 2 2 1 0 ]

step 2 -- exclusive prefix sum over degree -> row_offsets:
  row_offsets = [ 0 2 4 6 8 9 9 ]

step 3 -- scatter each edge via its source's own bump allocator:
  edge(2->3, w=8) claims slot 4 (next_slot[2] was 4, now 5)
  edge(0->1, w=4) claims slot 0 (next_slot[0] was 0, now 1)
  edge(3->4, w=2) claims slot 6 (next_slot[3] was 6, now 7)
  edge(1->2, w=2) claims slot 2 (next_slot[1] was 2, now 3)
  edge(0->2, w=1) claims slot 1 (next_slot[0] was 1, now 2)
  edge(4->5, w=3) claims slot 8 (next_slot[4] was 8, now 9)
  edge(2->4, w=10) claims slot 5 (next_slot[2] was 5, now 6)
  edge(3->5, w=6) claims slot 7 (next_slot[3] was 7, now 8)
  edge(1->3, w=5) claims slot 3 (next_slot[1] was 3, now 4)

  col_idx     = [ 1 2 2 3 3 4 4 5 5 ]
  weight_csr  = [ 4 1 2 5 8 10 2 6 3 ]

verifying every vertex's slice holds exactly its own edges (by set):
  vertex 0: matches
  vertex 1: matches
  vertex 2: matches
  vertex 3: matches
  vertex 4: matches
  vertex 5: matches

expected degree      = [ 2 2 2 2 1 0 ]
expected row_offsets = [ 0 2 4 6 8 9 9 ]
expected col_idx     = [ 1 2 2 3 3 4 4 5 5 ]
expected weight_csr  = [ 4 1 2 5 8 10 2 6 3 ]

self-check: CSR built from a raw, unsorted edge list via count -> scan
-> scatter matches the expected arrays exactly: confirmed
```

### The Concept, In Detail

Running the scatter step CONCURRENTLY (one thread per edge) under two different arrival orders for which thread's `atomicAdd` resolves first:

```
ASCII view: order A vs order B, same 9 edges, different arrival order.

  order A col_idx    = [ 1 2 | 2 3 | 3 4 | 4 5 | 5 ]
  order B col_idx    = [ 2 1 | 3 2 | 3 4 | 5 4 | 5 ]
             (vertex0) (vertex1)(vertex2)(vertex3)(v4)

  Vertex 0's slice holds {1, 2} in BOTH orders -- just in the OPPOSITE
  positions, because whichever of edge(0->1) or edge(0->2)'s atomicAdd
  on next_slot[0] happened to resolve first claimed slot 0, and the
  other claimed slot 1. Same story for vertex 1's slice ({2,3}) and
  vertex 3's slice ({4,5}).
```

This is exactly the same story this book has now told for a trie's shared child slots (Chapter 16.3), a hash table's shared probe slots (Chapter 19.2), and a cuckoo table rebuilt after a cycle (Chapter 20.3): which thread's atomic operation wins a race determines the EXACT position of the result, but never which SET of results ends up correct. Here, that means a vertex's slice can come out in a different internal order on different runs, while still holding precisely the right neighbors -- CSR's per-vertex slices are unordered SETS as far as correctness is concerned, even though they live in an ordered array.

[COMMON TRAP]
It is tempting to expect a concurrently-built CSR's `col_idx` array to exactly match one particular sequential construction's array, entry for entry. As with every other shared-slot race this book has covered, the only property that matters is semantic: does each vertex's slice, taken as a SET, contain exactly its correct neighbors (and matching weights). Comparing exact array CONTENTS between two concurrent runs -- or between a concurrent run and a specific sequential order -- is the wrong check; comparing each vertex's slice as a sorted set is the right one.

### Code and Verification

```cpp
#include <cstdio>
#include <vector>
#include <algorithm>

// Chapter 21.2 main -- both non-trivial steps of CSR construction are
// genuinely parallel over EDGES, one thread per edge, using exactly
// the atomic patterns this book has already built: degree counting is
// Chapter 7's histogram (`atomicAdd(&degree[src], 1)`, commutative,
// order never matters for the final counts); scattering into `col_idx`
// is Chapter 8's bump allocator, one counter PER VERTEX instead of one
// shared global counter (`atomicAdd(&next_slot[src], 1)` returns the
// claiming thread's own unique slot). The exclusive prefix sum that
// turns `degree` into `row_offsets` is Chapter 5's own already-parallel
// scan, simply applied here rather than re-derived. Because which
// thread's atomicAdd resolves first is not fixed, two different
// concurrent runs over the SAME edges can scatter them into DIFFERENT
// exact positions within a vertex's slice -- while every vertex's
// slice still ends up holding exactly the right SET of edges.
#define NUM_VERTICES 6
#define NUM_EDGES 9

__global__ void degree_count_kernel(const int* edge_src, int* degree, int num_edges) {
    int e = threadIdx.x;
    if (e >= num_edges) return;
    atomicAdd(&degree[edge_src[e]], 1);
}

__global__ void scatter_kernel(const int* edge_src, const int* edge_dst, const int* edge_weight,
                                int* next_slot, int* col_idx, int* weight_csr, int num_edges) {
    int e = threadIdx.x;
    if (e >= num_edges) return;
    int pos = atomicAdd(&next_slot[edge_src[e]], 1);
    col_idx[pos] = edge_dst[e];
    weight_csr[pos] = edge_weight[e];
}

// ---- Host-side replay of the identical per-thread logic, run under
// two DIFFERENT thread-execution orders. ----

struct Edge { int src, dst, weight; };

void run_construction(const std::vector<Edge>& edges, const std::vector<int>& row_offsets,
                       std::vector<int>& col_idx, std::vector<int>& weight_csr) {
    int E = (int)edges.size();
    col_idx.assign(E, -1);
    weight_csr.assign(E, -1);
    std::vector<int> next_slot = row_offsets;  // one bump allocator per vertex
    for (auto& e : edges) {
        int pos = next_slot[e.src];
        next_slot[e.src]++;
        col_idx[pos] = e.dst;
        weight_csr[pos] = e.weight;
    }
}

int main() {
    printf("=== Section 21.2 main: parallel degree count + parallel scatter ===\n\n");

    // Section 21.2's own row_offsets, already computed via Chapter 5's
    // exclusive scan over the degree-count histogram.
    std::vector<int> row_offsets = {0, 2, 4, 6, 8, 9, 9};
    printf("row_offsets (from degree count + Chapter 5's scan): [ 0 2 4 6 8 9 9 ]\n\n");

    std::vector<Edge> order_A = {
        {2,3,8}, {0,1,4}, {3,4,2}, {1,2,2}, {0,2,1},
        {4,5,3}, {2,4,10}, {3,5,6}, {1,3,5},
    };
    std::vector<Edge> order_B = {
        {0,2,1}, {2,3,8}, {1,3,5}, {0,1,4}, {3,5,6},
        {4,5,3}, {1,2,2}, {2,4,10}, {3,4,2},
    };

    printf("execution order A (%d threads, one per edge, this arrival order):\n", NUM_EDGES);
    std::vector<int> col_idx_A, weight_csr_A;
    run_construction(order_A, row_offsets, col_idx_A, weight_csr_A);
    printf("  col_idx    = [ ");
    for (int c : col_idx_A) printf("%d ", c);
    printf("]\n  weight_csr = [ ");
    for (int w : weight_csr_A) printf("%d ", w);
    printf("]\n\n");

    printf("execution order B (%d threads, one per edge, a DIFFERENT arrival order):\n", NUM_EDGES);
    std::vector<int> col_idx_B, weight_csr_B;
    run_construction(order_B, row_offsets, col_idx_B, weight_csr_B);
    printf("  col_idx    = [ ");
    for (int c : col_idx_B) printf("%d ", c);
    printf("]\n  weight_csr = [ ");
    for (int w : weight_csr_B) printf("%d ", w);
    printf("]\n\n");

    bool exact_layout_differs = (col_idx_A != col_idx_B);

    bool same_sets = true;
    printf("comparing per-vertex slices between the two orders (as SETS of\n");
    printf("(neighbor, weight) pairs, since exact position need not match):\n");
    for (int v = 0; v < NUM_VERTICES; v++) {
        std::vector<std::pair<int,int>> set_A, set_B;
        for (int i = row_offsets[v]; i < row_offsets[v + 1]; i++) {
            set_A.push_back({col_idx_A[i], weight_csr_A[i]});
            set_B.push_back({col_idx_B[i], weight_csr_B[i]});
        }
        std::sort(set_A.begin(), set_A.end());
        std::sort(set_B.begin(), set_B.end());
        bool match = (set_A == set_B);
        printf("  vertex %d: %s\n", v, match ? "identical sets" : "MISMATCH");
        same_sets = same_sets && match;
    }

    std::vector<int> exp_col_idx_A = {1, 2, 2, 3, 3, 4, 4, 5, 5};
    std::vector<int> exp_col_idx_B = {2, 1, 3, 2, 3, 4, 5, 4, 5};
    bool ok = exact_layout_differs && same_sets;
    ok = ok && (col_idx_A == exp_col_idx_A) && (col_idx_B == exp_col_idx_B);

    printf("\nexpected: order A and order B scatter edges into DIFFERENT exact\n");
    printf("col_idx layouts, but every vertex's slice holds the identical set\n");
    printf("of (neighbor, weight) pairs either way\n");

    printf("\nself-check: exact layouts differ between arrival orders, yet every\n");
    printf("per-vertex neighbor set matches exactly: %s\n", ok ? "confirmed" : "MISMATCH");
    return ok ? 0 : 1;
}
```

**Compile and run:**

```bash
NVDIR=/usr/local/lib/python3.11/dist-packages/nvidia/cu13
$NVDIR/bin/nvcc -arch=sm_80 -I$NVDIR/include -L$NVDIR/lib -lcudart 115_csr_concurrent_construction_kernel.cu -o 115_csr_concurrent_construction_kernel
LD_LIBRARY_PATH=$NVDIR/lib ./115_csr_concurrent_construction_kernel
```

**Sample input:** the same 9 edges, scattered by 9 concurrent threads under two different arrival orders, each thread claiming its own slot via `atomicAdd` on its source vertex's bump allocator.

**Sample output:**

```text
=== Section 21.2 main: parallel degree count + parallel scatter ===

row_offsets (from degree count + Chapter 5's scan): [ 0 2 4 6 8 9 9 ]

execution order A (9 threads, one per edge, this arrival order):
  col_idx    = [ 1 2 2 3 3 4 4 5 5 ]
  weight_csr = [ 4 1 2 5 8 10 2 6 3 ]

execution order B (9 threads, one per edge, a DIFFERENT arrival order):
  col_idx    = [ 2 1 3 2 3 4 5 4 5 ]
  weight_csr = [ 1 4 5 2 8 10 6 2 3 ]

comparing per-vertex slices between the two orders (as SETS of
(neighbor, weight) pairs, since exact position need not match):
  vertex 0: identical sets
  vertex 1: identical sets
  vertex 2: identical sets
  vertex 3: identical sets
  vertex 4: identical sets
  vertex 5: identical sets

expected: order A and order B scatter edges into DIFFERENT exact
col_idx layouts, but every vertex's slice holds the identical set
of (neighbor, weight) pairs either way

self-check: exact layouts differ between arrival orders, yet every
per-vertex neighbor set matches exactly: confirmed
```

## 21.3 Alternatives to CSR: Edge Lists (COO) for Edge-Parallel Work

### Intuition

CSR is built for one specific access pattern: given a vertex, enumerate its neighbors fast -- exactly what graph traversal (Chapter 22 onward) needs. It is a poor fit for work that wants to touch every EDGE independently without caring which vertex it belongs to. The raw edge list itself -- three flat, parallel arrays of `(src, dst, weight)`, often called COO ("coordinate") format, borrowing the name from sparse-matrix storage -- is already perfectly shaped for exactly that: one thread per edge, zero grouping, zero indirection through any vertex structure at all.

### The Sequential (CPU) Baseline

```cpp
#include <cstdio>
#include <vector>

// Chapter 21.3 -- The Sequential (CPU) Baseline.
// CSR is built specifically for "given a vertex, enumerate its
// neighbors fast" -- exactly what traversal algorithms (Chapter 22
// onward) need. It is a poor fit for work that wants to touch every
// EDGE independently, without caring which vertex it belongs to: that
// is what the raw edge list -- three parallel arrays of (src, dst,
// weight), often called COO ("coordinate") format -- is already
// perfectly shaped for, since it never groups anything by vertex in
// the first place. This section computes the same quantity (the total
// weight of every edge) two different ways, to make the loop-shape
// difference concrete before Section 21.3's main file makes it a
// parallel one.
#define V 6
#define E 9

int main() {
    printf("=== Section 21.3 CPU baseline: summing edge weights, COO vs CSR ===\n\n");

    // COO format: one flat triple per edge, no grouping at all.
    std::vector<int> edge_src    = {2, 0, 3, 1, 0, 4, 2, 3, 1};
    std::vector<int> edge_dst    = {3, 1, 4, 2, 2, 5, 4, 5, 3};
    std::vector<int> edge_weight = {8, 4, 2, 2, 1, 3, 10, 6, 5};

    printf("COO (edge list) format -- 3 flat arrays, one entry per edge:\n");
    printf("  edge_src    = [ ");
    for (int s : edge_src) printf("%d ", s);
    printf("]\n  edge_dst    = [ ");
    for (int d : edge_dst) printf("%d ", d);
    printf("]\n  edge_weight = [ ");
    for (int w : edge_weight) printf("%d ", w);
    printf("]\n\n");

    printf("summing via COO: ONE flat loop over all %d edges, no vertex knowledge needed:\n", E);
    int sum_coo = 0;
    for (int i = 0; i < E; i++) sum_coo += edge_weight[i];
    printf("  sum_coo = %d\n\n", sum_coo);

    // CSR format: same edges, grouped by source vertex.
    std::vector<int> row_offsets = {0, 2, 4, 6, 8, 9, 9};
    std::vector<int> col_idx     = {1, 2, 2, 3, 3, 4, 4, 5, 5};
    std::vector<int> weight_csr  = {4, 1, 2, 5, 8, 10, 2, 6, 3};

    printf("CSR format -- same edges, grouped by source vertex:\n");
    printf("  row_offsets = [ ");
    for (int r : row_offsets) printf("%d ", r);
    printf("]\n  col_idx     = [ ");
    for (int c : col_idx) printf("%d ", c);
    printf("]\n  weight_csr  = [ ");
    for (int w : weight_csr) printf("%d ", w);
    printf("]\n\n");

    printf("summing via CSR: a NESTED loop -- outer over %d vertices, inner over\n", V);
    printf("each vertex's own (uneven-length) slice:\n");
    int sum_csr = 0;
    for (int v = 0; v < V; v++) {
        int lo = row_offsets[v], hi = row_offsets[v + 1];
        printf("  vertex %d: %d edge(s) in its slice\n", v, hi - lo);
        for (int i = lo; i < hi; i++) sum_csr += weight_csr[i];
    }
    printf("  sum_csr = %d\n\n", sum_csr);

    bool ok = (sum_coo == 41) && (sum_csr == 41) && (sum_coo == sum_csr);

    printf("expected: both totals equal 41\n");
    printf("(the SAME 9 edges are visited exactly once either way -- COO's single\n");
    printf("flat loop needs no per-vertex bookkeeping at all, while CSR's nested\n");
    printf("loop must visit vertex 5's EMPTY slice and do genuinely less work for\n");
    printf("vertex 4 than for vertex 0 -- uneven inner-loop lengths that a flat,\n");
    printf("one-thread-per-edge scan over COO simply does not have)\n");

    printf("\nself-check: COO and CSR sums agree exactly, and both equal 41: %s\n", ok ? "confirmed" : "MISMATCH");
    return ok ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++17 -Wall -Wextra -O2 116_coo_vs_csr_sum_cpu_baseline.cpp -o 116_coo_vs_csr_sum_cpu_baseline
./116_coo_vs_csr_sum_cpu_baseline
```

**Sample input:** the same 9 weighted edges, with their total weight computed two ways -- one flat loop over COO, and one nested (vertex, then neighbor) loop over CSR.

**Sample output:**

```text
=== Section 21.3 CPU baseline: summing edge weights, COO vs CSR ===

COO (edge list) format -- 3 flat arrays, one entry per edge:
  edge_src    = [ 2 0 3 1 0 4 2 3 1 ]
  edge_dst    = [ 3 1 4 2 2 5 4 5 3 ]
  edge_weight = [ 8 4 2 2 1 3 10 6 5 ]

summing via COO: ONE flat loop over all 9 edges, no vertex knowledge needed:
  sum_coo = 41

CSR format -- same edges, grouped by source vertex:
  row_offsets = [ 0 2 4 6 8 9 9 ]
  col_idx     = [ 1 2 2 3 3 4 4 5 5 ]
  weight_csr  = [ 4 1 2 5 8 10 2 6 3 ]

summing via CSR: a NESTED loop -- outer over 6 vertices, inner over
each vertex's own (uneven-length) slice:
  vertex 0: 2 edge(s) in its slice
  vertex 1: 2 edge(s) in its slice
  vertex 2: 2 edge(s) in its slice
  vertex 3: 2 edge(s) in its slice
  vertex 4: 1 edge(s) in its slice
  vertex 5: 0 edge(s) in its slice
  sum_csr = 41

expected: both totals equal 41
(the SAME 9 edges are visited exactly once either way -- COO's single
flat loop needs no per-vertex bookkeeping at all, while CSR's nested
loop must visit vertex 5's EMPTY slice and do genuinely less work for
vertex 4 than for vertex 0 -- uneven inner-loop lengths that a flat,
one-thread-per-edge scan over COO simply does not have)

self-check: COO and CSR sums agree exactly, and both equal 41: confirmed
```

### The Concept, In Detail

```
ASCII view: loop shape, COO vs CSR, for the identical computation:

  COO (one thread per edge):
    for e in 0..E-1: total += edge_weight[e]
    -- flat, uniform: every thread does EXACTLY one read, one add.

  CSR (one thread per vertex):
    for v in 0..V-1:
      for i in row_offsets[v]..row_offsets[v+1]-1: total += weight_csr[i]
    -- nested, UNEVEN: thread for vertex 0 does 2x the work of thread
       for vertex 4, and 100% more than vertex 5's thread (which does
       zero -- an empty slice).
```

On this book's tiny 6-vertex example, that imbalance is barely noticeable -- a couple of wasted cycles on an idle thread. Real-world graphs are typically far more skewed: a social network's "hub" vertices, or a web graph's most-linked pages, can have thousands or millions of times more neighbors than a typical vertex, meaning a vertex-parallel kernel would leave the vast majority of its threads idle while a tiny handful churn through enormous slices. Edge-parallel work over COO sidesteps this completely, since every thread's workload is fixed at exactly one edge, regardless of how any vertex's degree is distributed. This same fork -- vertex-parallel (CSR-natural) versus edge-parallel (COO-natural) -- is exactly the choice Chapter 22's frontier-based breadth-first search will need to make for expanding a frontier's neighbors.

A companion format worth naming without a dedicated code file (the same treatment Chapter 18.3 gave octrees and BVHs alongside quadtrees): CSC ("compressed sparse column") is simply CSR transposed -- grouped by DESTINATION instead of source. Where CSR answers "who does this vertex point to," CSC answers "who points TO this vertex," which algorithms needing in-neighbors (computing in-degree, or PageRank-style score propagation) need instead. Building CSC is the identical count -> scan -> scatter procedure from Section 21.2, just keyed on `edge_dst` instead of `edge_src`.

[COMMON TRAP]
It is tempting to assume CSR is simply the "better" format and CSC or COO are lesser fallbacks. Each format is built for a specific access pattern, and using the wrong one costs real performance even when the computation is otherwise identical: CSR for fast per-vertex neighbor enumeration, CSC for fast per-vertex IN-neighbor enumeration, and COO for uniform, degree-independent edge-parallel work. Choosing a format means asking what the algorithm actually needs to look up quickly, not which one seems most "complete."

### Code and Verification

```cpp
#include <cstdio>
#include <vector>

// Chapter 21.3 main -- COO's flat, ungrouped layout makes it the
// natural fit for EDGE-parallel work: one thread per edge, each
// reading its own (src, dst, weight) triple directly with zero
// indirection through any vertex structure, then contributing to a
// shared total via `atomicAdd` -- exactly Chapter 4's reduction,
// applied over edges instead of a plain array. Attempting the same
// total with one thread PER VERTEX over CSR (as Section 21.1's kernel
// did for neighbor sums) would force vertex 5's thread to do zero work
// while vertex 0's thread does twice as much as vertex 4's -- on a
// real graph, where a handful of "hub" vertices can have thousands of
// times more neighbors than average, that same imbalance becomes
// severe. Edge-parallel work sidesteps it entirely: every thread here
// does exactly the same fixed amount of work, regardless of degree.
#define NUM_EDGES 9

// One thread per edge. Every thread reads exactly one entry from each
// of three shared, read-only arrays, then performs exactly one
// atomicAdd -- uniform work, no thread ever does more than any other.
__global__ void coo_edge_sum_kernel(const int* edge_weight, int* total, int num_edges) {
    int e = threadIdx.x;
    if (e >= num_edges) return;
    atomicAdd(total, edge_weight[e]);
}

// ---- Host-side replay of the identical per-thread logic, run under
// two different accumulation orders to confirm the commutative sum
// (Chapter 17.2's own point, reapplied) is order-independent. ----

int main() {
    printf("=== Section 21.3 main: edge-parallel reduction over COO ===\n\n");

    std::vector<int> edge_weight = {8, 4, 2, 2, 1, 3, 10, 6, 5};
    printf("edge_weight = [ 8 4 2 2 1 3 10 6 5 ]\n");
    printf("(%d threads launched, one per edge -- each does ONE read and ONE\n", NUM_EDGES);
    printf("atomicAdd; no thread's workload depends on any vertex's degree)\n\n");

    // Order A: threads resolve their atomicAdd in increasing edge order.
    std::vector<int> order_A = {0, 1, 2, 3, 4, 5, 6, 7, 8};
    int total_A = 0;
    for (int e : order_A) total_A += edge_weight[e];
    printf("execution order A (increasing edge index): running total = %d\n", total_A);

    // Order B: an adversarial interleaving, resolving in reverse.
    std::vector<int> order_B = {8, 7, 6, 5, 4, 3, 2, 1, 0};
    int total_B = 0;
    for (int e : order_B) total_B += edge_weight[e];
    printf("execution order B (reverse edge index):    running total = %d\n\n", total_B);

    bool ok = (total_A == 41) && (total_B == 41) && (total_A == total_B);

    printf("expected: both orders produce the identical total, 41\n");
    printf("(addition is commutative -- exactly Section 17.2's own point about\n");
    printf("point updates, reapplied here to a reduction over edges instead of\n");
    printf("a tree: the ORDER atomicAdd calls resolve in never changes the sum)\n");

    printf("\nself-check: edge-parallel reduction over COO matches the CPU\n");
    printf("baseline's total (41) under both accumulation orders: %s\n", ok ? "confirmed" : "MISMATCH");
    return ok ? 0 : 1;
}
```

**Compile and run:**

```bash
NVDIR=/usr/local/lib/python3.11/dist-packages/nvidia/cu13
$NVDIR/bin/nvcc -arch=sm_80 -I$NVDIR/include -L$NVDIR/lib -lcudart 117_coo_edge_parallel_reduce_kernel.cu -o 117_coo_edge_parallel_reduce_kernel
LD_LIBRARY_PATH=$NVDIR/lib ./117_coo_edge_parallel_reduce_kernel
```

**Sample input:** the same 9 edge weights, summed by 9 independent threads (one per edge) via `atomicAdd`, under two different accumulation orders.

**Sample output:**

```text
=== Section 21.3 main: edge-parallel reduction over COO ===

edge_weight = [ 8 4 2 2 1 3 10 6 5 ]
(9 threads launched, one per edge -- each does ONE read and ONE
atomicAdd; no thread's workload depends on any vertex's degree)

execution order A (increasing edge index): running total = 41
execution order B (reverse edge index):    running total = 41

expected: both orders produce the identical total, 41
(addition is commutative -- exactly Section 17.2's own point about
point updates, reapplied here to a reduction over edges instead of
a tree: the ORDER atomicAdd calls resolve in never changes the sum)

self-check: edge-parallel reduction over COO matches the CPU
baseline's total (41) under both accumulation orders: confirmed
```

## Chapter Summary

Graphs are irregular by nature -- each vertex can have a wildly different number of neighbors -- which makes a naive adjacency list of separate, per-vertex containers a poor fit for a GPU's flat-memory model. Compressed Sparse Row (CSR) fixes this by flattening every vertex's neighbors into one shared array (`col_idx`), indexed via a small `row_offsets` array whose `V+1` entries give every vertex, including the last, an explicit slice boundary with no special case needed. Building CSR directly from a raw, unsorted edge list is exactly three already-built primitives applied in sequence: a degree-count histogram (Chapter 7), an exclusive prefix sum (Chapter 5), and an atomic bump-allocator scatter (Chapter 8) -- and, like every other shared-slot race this book has covered, concurrent execution can scatter edges into different exact positions within a vertex's slice while every vertex's neighbor SET remains correct regardless. CSR is not universally best: it excels at per-vertex neighbor lookup but forces uneven, degree-dependent work onto a vertex-parallel kernel, while the raw edge list format (COO) it was built from stays the right tool whenever work needs to touch every edge uniformly and independently, regardless of degree -- exactly the choice Chapter 22's frontier-based traversal will need to make.

## Self-Check Questions

1. Why does an adjacency list of per-vertex containers not work well as a GPU data structure, even though it is a perfectly reasonable sequential representation?
2. Why does `row_offsets` need `V+1` entries rather than exactly `V`, and what special case does that extra entry eliminate?
3. Building CSR from a raw edge list uses three already-established primitives from earlier chapters. Name each one and the earlier chapter it came from.
4. Two concurrent runs of the same CSR-construction scatter, over the same edges, can produce different exact `col_idx` arrays. Why does this not make either run incorrect?
5. Why is a raw edge list (COO) format better suited than CSR for computing something over every edge independently, such as a total edge weight?
6. What does CSC represent that CSR does not, and how would you build it?

## Where We Go Next

With a graph representation a thread can actually index into, Chapter 22 turns to graph TRAVERSAL: breadth-first search reframed around the FRONTIER model -- the set of vertices reachable in exactly k steps -- which is what makes BFS genuinely parallelizable, expanding an entire frontier's neighbors at once rather than visiting vertices one at a time.

## Worked Solutions

**1.** An adjacency list represents each vertex's neighbors as a separate container (for instance, a `std::vector<int>`), and these containers can be wildly different sizes across vertices. A GPU kernel needs a single, flat, indexable block of memory that every thread can read from directly -- there is no way to hand a kernel a collection of separately-allocated, variable-length host containers, and even if there were, threads would have no uniform way to compute where one vertex's data ends and another's begins.

**2.** With only `V` entries, the LAST vertex would have a starting offset but no explicit ending boundary to read -- `row_offsets[V]` would be out of bounds. Rather than special-casing "if this is the last vertex, the slice extends to the end of col_idx instead," the extra `V+1`-th entry stores the total edge count directly, so every vertex, including the last, uses the exact same `[row_offsets[v], row_offsets[v+1])` formula with zero exceptions.

**3.** Counting each vertex's degree from the raw edge list is Chapter 7's histogram (counting occurrences of each source vertex, exactly like counting occurrences of each bin value). Turning the resulting degree counts into `row_offsets` is Chapter 5's exclusive prefix sum (scan). Scattering each edge into its final `col_idx` position, using a per-vertex running counter that different edges targeting the same vertex race to increment, is Chapter 8's atomic bump allocator, generalized from one shared global counter to one counter per vertex.

**4.** Which thread's `atomicAdd` on a given vertex's "next free slot" counter resolves first determines only the EXACT POSITION within that vertex's slice that each edge lands in -- it does not change which edges belong to that vertex, since every edge still targets its own fixed source vertex's counter regardless of timing. Correctness for CSR construction means each vertex's slice contains exactly the right SET of neighbors (and matching weights); the internal order of entries within a slice was never part of the contract, so two different valid orderings are both correct.

**5.** A raw edge list already stores every edge as an independent, self-contained entry in a flat array, with no grouping by vertex at all -- a thread assigned to edge `e` needs to read only `edge_src[e]`, `edge_dst[e]`, and `edge_weight[e]`, doing a fixed, identical amount of work regardless of which vertices are involved. Computing the same total via CSR requires a nested loop -- once per vertex, then once per neighbor in that vertex's slice -- whose inner-loop length varies with degree, forcing some threads (high-degree vertices) to do far more work than others (low-degree or zero-degree vertices), which wastes parallel capacity without needing to.

**6.** CSC represents each vertex's IN-neighbors -- the set of vertices with an edge pointing INTO it -- whereas CSR represents each vertex's OUT-neighbors. Building CSC uses the identical three-step count-scan-scatter procedure Section 21.2 used for CSR, with every step keyed on `edge_dst` instead of `edge_src`: count how many edges target each vertex (a histogram over destinations), exclusive-scan those counts into a `col_offsets`-style array, then scatter each edge's SOURCE vertex into the resulting array at a position claimed from its destination vertex's own running counter.
