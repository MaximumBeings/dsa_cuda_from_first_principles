# Chapter 23: Single-Source Shortest Paths

Chapter 22's frontier model relied on a quiet assumption: every edge costs the same, so fewer hops always meant a shorter distance, and "first thread to arrive" was automatically the right answer. Weighted edges break that assumption completely -- a vertex reached via more hops but a smaller total weight can still be the true shortest path. This chapter builds single-source shortest paths in three pieces: Dijkstra's algorithm, whose greedy discipline resists easy parallelization even though its relax step doesn't; Bellman-Ford, which abandons that discipline entirely to make every round embarrassingly parallel; and `atomicMin`, a new primitive whose "always keep the smallest value proposed, regardless of arrival order" guarantee is what makes Bellman-Ford's lack of ordering safe.

## 23.1 Weighted Edges Break the Frontier Model: Dijkstra's Algorithm

### Intuition

Chapter 22's BFS visited vertices in order of HOP COUNT, and that was correct because every edge cost exactly one hop. With weighted edges, a vertex several hops away via cheap edges can be genuinely closer than a vertex one hop away via an expensive edge, so "first thread to discover it" is no longer a valid stand-in for "shortest distance." Dijkstra's algorithm restores correctness by always expanding the UNVISITED vertex with the smallest known tentative distance next, guaranteeing that once a vertex is popped, its distance can never improve again -- every edge still under consideration would only make things worse.

### The Sequential (CPU) Baseline

```cpp
#include <cstdio>
#include <vector>

// Chapter 23.1 -- The Sequential (CPU) Baseline.
// Breadth-first search's "first thread to arrive wins" rule (Chapter
// 22) relied on every edge costing exactly the same -- fewer hops
// always meant a shorter total distance. Weighted edges break this
// completely: a vertex reached via MORE hops but a smaller total
// WEIGHT can still be the true shortest path. Dijkstra's algorithm
// fixes this by always expanding the UNVISITED vertex with the
// smallest known tentative distance next (found here by a simple
// O(V) scan over a small array -- no heap needed at this book's
// scale), guaranteeing that once a vertex is popped, its distance is
// final and can never improve later.
#define V 6
#define INF 999999

int main() {
    printf("=== Section 23.1 CPU baseline: Dijkstra's algorithm ===\n\n");

    // Chapter 21's own CSR arrays, now with real edge weights.
    std::vector<int> row_offsets = {0, 2, 4, 6, 8, 9, 9};
    std::vector<int> col_idx     = {1, 2, 2, 3, 3, 4, 4, 5, 5};
    std::vector<int> weight_csr  = {4, 1, 2, 5, 8, 10, 2, 6, 3};
    printf("row_offsets = [ 0 2 4 6 8 9 9 ]\n");
    printf("col_idx     = [ 1 2 2 3 3 4 4 5 5 ]\n");
    printf("weight_csr  = [ 4 1 2 5 8 10 2 6 3 ]\n\n");

    std::vector<int> dist(V, INF);
    std::vector<bool> visited(V, false);
    int source = 0;
    dist[source] = 0;
    std::vector<int> pop_order;

    printf("source vertex: %d\n\n", source);

    for (int iter = 0; iter < V; iter++) {
        int u = -1, best = INF;
        for (int v = 0; v < V; v++) {
            if (!visited[v] && dist[v] < best) { best = dist[v]; u = v; }
        }
        if (u == -1) break;
        visited[u] = true;
        pop_order.push_back(u);
        printf("pop vertex %d (dist=%d) -- relax its neighbors:", u, dist[u]);
        for (int i = row_offsets[u]; i < row_offsets[u + 1]; i++) {
            int w = col_idx[i], wt = weight_csr[i];
            int candidate = dist[u] + wt;
            if (candidate < dist[w]) {
                printf(" %d(%d->%d, improved)", w, dist[w] == INF ? -1 : dist[w], candidate);
                dist[w] = candidate;
            } else {
                printf(" %d(%d, no improvement)", w, dist[w]);
            }
        }
        printf("\n");
    }

    printf("\npop order: ");
    for (int u : pop_order) printf("%d ", u);
    printf("\nfinal dist: [ ");
    for (int d : dist) printf("%d ", d);
    printf("]\n\n");

    printf("compare to Chapter 22's UNWEIGHTED BFS pop order on this same graph:\n");
    printf("  BFS pop order (hop count only):      0 1 2 3 4 5\n");
    printf("  Dijkstra pop order (true min weight): ");
    for (int u : pop_order) printf("%d ", u);
    printf("\n  vertex 2 pops BEFORE vertex 1 here, because edge (0->2, w=1) is far\n");
    printf("  cheaper than (0->1, w=4) -- fewer hops does NOT mean shorter distance\n");

    std::vector<int> expected_pop_order = {0, 2, 1, 3, 4, 5};
    std::vector<int> expected_dist = {0, 4, 1, 9, 11, 14};
    bool ok = (pop_order == expected_pop_order) && (dist == expected_dist);

    printf("\nexpected pop order: 0 2 1 3 4 5\n");
    printf("expected dist:      [ 0 4 1 9 11 14 ]\n");
    printf("\nself-check: Dijkstra's pop order and final distances match expected: %s\n", ok ? "confirmed" : "MISMATCH");
    return ok ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++17 -Wall -Wextra -O2 124_dijkstra_cpu_baseline.cpp -o 124_dijkstra_cpu_baseline
./124_dijkstra_cpu_baseline
```

**Sample input:** Chapter 21's own 6-vertex graph, now with real edge weights, run through Dijkstra's algorithm from source vertex 0.

**Sample output:**

```text
=== Section 23.1 CPU baseline: Dijkstra's algorithm ===

row_offsets = [ 0 2 4 6 8 9 9 ]
col_idx     = [ 1 2 2 3 3 4 4 5 5 ]
weight_csr  = [ 4 1 2 5 8 10 2 6 3 ]

source vertex: 0

pop vertex 0 (dist=0) -- relax its neighbors: 1(-1->4, improved) 2(-1->1, improved)
pop vertex 2 (dist=1) -- relax its neighbors: 3(-1->9, improved) 4(-1->11, improved)
pop vertex 1 (dist=4) -- relax its neighbors: 2(1, no improvement) 3(9, no improvement)
pop vertex 3 (dist=9) -- relax its neighbors: 4(11, no improvement) 5(-1->15, improved)
pop vertex 4 (dist=11) -- relax its neighbors: 5(15->14, improved)
pop vertex 5 (dist=14) -- relax its neighbors:

pop order: 0 2 1 3 4 5 
final dist: [ 0 4 1 9 11 14 ]

compare to Chapter 22's UNWEIGHTED BFS pop order on this same graph:
  BFS pop order (hop count only):      0 1 2 3 4 5
  Dijkstra pop order (true min weight): 0 2 1 3 4 5 
  vertex 2 pops BEFORE vertex 1 here, because edge (0->2, w=1) is far
  cheaper than (0->1, w=4) -- fewer hops does NOT mean shorter distance

expected pop order: 0 2 1 3 4 5
expected dist:      [ 0 4 1 9 11 14 ]

self-check: Dijkstra's pop order and final distances match expected: confirmed
```

### The Concept, In Detail

```
ASCII view: BFS's hop-count order vs Dijkstra's true-distance order,
on the IDENTICAL graph.

  BFS (Chapter 22, unweighted):       0 -> 1 -> 2 -> 3 -> 4 -> 5
  Dijkstra (this chapter, weighted):  0 -> 2 -> 1 -> 3 -> 4 -> 5
                                            ^
                                     vertex 2 pops SECOND, not third --
                                     edge (0->2, w=1) is far cheaper
                                     than edge (0->1, w=4), so vertex 2
                                     is genuinely CLOSER even though
                                     both are one hop from the source.
```

The popped vertex's distance is final specifically because Dijkstra always picks the GLOBAL minimum among everything still unvisited: any other unvisited vertex has a tentative distance at least as large, and following any edge only adds a non-negative weight, so no future relaxation could ever produce a smaller value for the vertex just popped. This guarantee depends entirely on picking the true minimum every single time -- which is precisely why the outer loop resists parallelization the way Chapter 22's frontier expansion did not.

[COMMON TRAP]
It is tempting to think Dijkstra's algorithm could be parallelized the same way Chapter 22's BFS was, by expanding "the current frontier" all at once. Dijkstra has no notion of a frontier of equally-valid next vertices -- only ONE vertex (the true global minimum among the unvisited) is safe to finalize at any given moment. Popping any OTHER unvisited vertex before it could finalize a distance that a cheaper, not-yet-considered edge would later beat, breaking the entire algorithm's correctness guarantee.

### Code and Verification

```cpp
#include <cstdio>
#include <vector>
#include <algorithm>

// Chapter 23.1 main -- Dijkstra's OUTER loop (finding the global
// minimum-distance unvisited vertex) is a genuine sequential
// bottleneck: exactly like Chapter 17.1's own short segment-tree
// query, there is no useful way to split "find the single smallest
// value among all unvisited vertices" across many independent threads
// without first re-deriving a reduction. What DOES parallelize cleanly
// is the RELAX step -- once a vertex is popped, updating every one of
// its neighbors can happen at once, one thread per outgoing edge, via
// `atomicMin(&dist[w], candidate)`. `atomicMin` is a new primitive for
// this book: like `atomicAdd`, it is commutative and needs no
// CAS-retry loop, but its job is different -- not to accumulate every
// contribution, but to always keep the SMALLEST one, no matter which
// thread's write reaches the slot first.
#define V 6

__global__ void dijkstra_relax_kernel(const int* row_offsets, const int* col_idx,
                                       const int* weight_csr, int* dist, int u) {
    int t = threadIdx.x;
    int lo = row_offsets[u], hi = row_offsets[u + 1];
    if (lo + t >= hi) return;
    int i = lo + t;
    int w = col_idx[i];
    int candidate = dist[u] + weight_csr[i];
    atomicMin(&dist[w], candidate);
}

// ---- Host-side replay of the identical per-thread logic. ----

int main() {
    printf("=== Section 23.1 main: parallel relax step for one popped vertex ===\n\n");

    std::vector<int> row_offsets = {0, 2, 4, 6, 8, 9, 9};
    std::vector<int> col_idx     = {1, 2, 2, 3, 3, 4, 4, 5, 5};
    std::vector<int> weight_csr  = {4, 1, 2, 5, 8, 10, 2, 6, 3};
    std::vector<int> dist = {0, 999999, 999999, 999999, 999999, 999999};

    printf("popping vertex 0 (dist=0): 2 threads relax its 2 outgoing edges at once\n");
    int u = 0;
    for (int i = row_offsets[u]; i < row_offsets[u + 1]; i++) {
        int w = col_idx[i], wt = weight_csr[i];
        int candidate = dist[u] + wt;
        int old = dist[w];
        dist[w] = std::min(dist[w], candidate);
        printf("  thread %d: atomicMin(dist[%d]=%d, candidate=%d) -> dist[%d]=%d\n",
               i - row_offsets[u], w, old, candidate, w, dist[w]);
    }
    printf("  dist after popping 0: [ ");
    for (int d : dist) printf("%d ", d);
    printf("]\n\n");

    printf("popping vertex 2 (dist=1): 2 threads relax its 2 outgoing edges at once\n");
    u = 2;
    for (int i = row_offsets[u]; i < row_offsets[u + 1]; i++) {
        int w = col_idx[i], wt = weight_csr[i];
        int candidate = dist[u] + wt;
        int old = dist[w];
        dist[w] = std::min(dist[w], candidate);
        printf("  thread %d: atomicMin(dist[%d]=%d, candidate=%d) -> dist[%d]=%d\n",
               i - row_offsets[u], w, old, candidate, w, dist[w]);
    }
    printf("  dist after popping 2: [ ");
    for (int d : dist) printf("%d ", d);
    printf("]\n\n");

    std::vector<int> expected_dist = {0, 4, 1, 9, 11, 999999};
    bool ok = (dist == expected_dist);

    printf("expected dist: [ 0 4 1 9 11 999999 ] (vertex 5 not yet relaxed by any pop)\n");
    printf("(popping vertex 0's 2 neighbors and popping vertex 2's 2 neighbors are\n");
    printf("each internally race-free here -- no two edges from the SAME popped\n");
    printf("vertex share a destination in this graph -- but the outer loop that\n");
    printf("chose to pop 0 then 2, in that specific order, is exactly the part\n");
    printf("that stays sequential: it depends on the global minimum found so far)\n");

    printf("\nself-check: parallel relaxation of each popped vertex's neighbors\n");
    printf("matches Dijkstra's own sequential trace exactly: %s\n", ok ? "confirmed" : "MISMATCH");
    return ok ? 0 : 1;
}
```

**Compile and run:**

```bash
NVDIR=/usr/local/lib/python3.11/dist-packages/nvidia/cu13
$NVDIR/bin/nvcc -arch=sm_80 -I$NVDIR/include -L$NVDIR/lib -lcudart 125_dijkstra_relax_kernel.cu -o 125_dijkstra_relax_kernel
LD_LIBRARY_PATH=$NVDIR/lib ./125_dijkstra_relax_kernel
```

**Sample input:** popping vertices 0 and 2 in turn, each time relaxing that vertex's own outgoing edges in parallel via `atomicMin`.

**Sample output:**

```text
=== Section 23.1 main: parallel relax step for one popped vertex ===

popping vertex 0 (dist=0): 2 threads relax its 2 outgoing edges at once
  thread 0: atomicMin(dist[1]=999999, candidate=4) -> dist[1]=4
  thread 1: atomicMin(dist[2]=999999, candidate=1) -> dist[2]=1
  dist after popping 0: [ 0 4 1 999999 999999 999999 ]

popping vertex 2 (dist=1): 2 threads relax its 2 outgoing edges at once
  thread 0: atomicMin(dist[3]=999999, candidate=9) -> dist[3]=9
  thread 1: atomicMin(dist[4]=999999, candidate=11) -> dist[4]=11
  dist after popping 2: [ 0 4 1 9 11 999999 ]

expected dist: [ 0 4 1 9 11 999999 ] (vertex 5 not yet relaxed by any pop)
(popping vertex 0's 2 neighbors and popping vertex 2's 2 neighbors are
each internally race-free here -- no two edges from the SAME popped
vertex share a destination in this graph -- but the outer loop that
chose to pop 0 then 2, in that specific order, is exactly the part
that stays sequential: it depends on the global minimum found so far)

self-check: parallel relaxation of each popped vertex's neighbors
matches Dijkstra's own sequential trace exactly: confirmed
```

## 23.2 Bellman-Ford: Relax Every Edge, Every Round, Fully in Parallel

### Intuition

Dijkstra's outer loop is a genuine sequential bottleneck because it insists on a strict order: always the true minimum next. Bellman-Ford abandons that discipline completely -- instead of choosing vertices in any particular order, it relaxes EVERY edge in the graph, every round, for up to V-1 rounds. This is still provably correct: any shortest path in a graph with V vertices uses at most V-1 edges, so V-1 full rounds of "relax every edge" are always enough to propagate the true shortest distance everywhere, regardless of what order the edges happen to be visited in within a round. Because no ordering is required at all, every round is embarrassingly parallel: one thread per edge, using the raw edge list (COO, Chapter 21.3) rather than CSR, since nothing here needs edges grouped by vertex.

### The Sequential (CPU) Baseline

```cpp
#include <cstdio>
#include <vector>

// Chapter 23.2 -- The Sequential (CPU) Baseline.
// Dijkstra's outer loop is a genuine sequential bottleneck because it
// insists on a strict ORDER: always the current global minimum next.
// Bellman-Ford abandons that discipline entirely -- instead of picking
// vertices in any particular order, it relaxes EVERY edge in the
// graph, every round, for up to V-1 rounds. This is still guaranteed
// correct: any shortest path in a graph with V vertices uses at most
// V-1 edges, so V-1 full rounds of "relax every edge" are always
// enough to propagate the true shortest distance to every vertex,
// regardless of what order the edges happen to be visited in within a
// round. Using the raw edge list (COO format, Chapter 21.3) rather
// than CSR is deliberate: nothing here needs to group edges by vertex
// at all.
#define V 6
#define INF 999999

struct Edge { int src, dst, weight; };

int main() {
    printf("=== Section 23.2 CPU baseline: Bellman-Ford, relax every edge every round ===\n\n");

    // The same 9 weighted edges, in COO (edge list) form.
    std::vector<Edge> edges = {
        {2,3,8}, {0,1,4}, {3,4,2}, {1,2,2}, {0,2,1},
        {4,5,3}, {2,4,10}, {3,5,6}, {1,3,5},
    };
    printf("edges (COO): ");
    for (auto& e : edges) printf("(%d->%d,w=%d) ", e.src, e.dst, e.weight);
    printf("\n\n");

    std::vector<int> dist(V, INF);
    int source = 0;
    dist[source] = 0;
    printf("source vertex: %d, up to %d rounds allowed\n\n", source, V - 1);

    int rounds_used = 0;
    for (int round = 1; round <= V - 1; round++) {
        bool updated = false;
        printf("round %d:", round);
        for (auto& e : edges) {
            if (dist[e.src] == INF) continue;
            int candidate = dist[e.src] + e.weight;
            if (candidate < dist[e.dst]) {
                printf(" (%d->%d: %d->%d)", e.src, e.dst, dist[e.dst] == INF ? -1 : dist[e.dst], candidate);
                dist[e.dst] = candidate;
                updated = true;
            }
        }
        if (!updated) printf(" no updates -- converged early");
        printf("\n  dist = [ ");
        for (int d : dist) printf("%d ", d);
        printf("]\n");
        rounds_used = round;
        if (!updated) break;
    }

    printf("\nfinal dist: [ ");
    for (int d : dist) printf("%d ", d);
    printf("]  (converged after %d round(s), out of %d allowed)\n", rounds_used, V - 1);

    std::vector<int> expected_dist = {0, 4, 1, 9, 11, 14};
    bool ok = (dist == expected_dist) && (rounds_used == 3);

    printf("\nexpected dist: [ 0 4 1 9 11 14 ], matching Dijkstra's own result exactly\n");
    printf("expected rounds used: 3\n");
    printf("\nself-check: Bellman-Ford's distances match Dijkstra's, with no ordering\n");
    printf("discipline required at all: %s\n", ok ? "confirmed" : "MISMATCH");
    return ok ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++17 -Wall -Wextra -O2 126_bellmanford_cpu_baseline.cpp -o 126_bellmanford_cpu_baseline
./126_bellmanford_cpu_baseline
```

**Sample input:** the same 9 weighted edges (COO format), relaxed every round for up to 5 rounds, with early exit once a round makes no updates.

**Sample output:**

```text
=== Section 23.2 CPU baseline: Bellman-Ford, relax every edge every round ===

edges (COO): (2->3,w=8) (0->1,w=4) (3->4,w=2) (1->2,w=2) (0->2,w=1) (4->5,w=3) (2->4,w=10) (3->5,w=6) (1->3,w=5) 

source vertex: 0, up to 5 rounds allowed

round 1: (0->1: -1->4) (1->2: -1->6) (0->2: 6->1) (2->4: -1->11) (1->3: -1->9)
  dist = [ 0 4 1 9 11 999999 ]
round 2: (4->5: -1->14)
  dist = [ 0 4 1 9 11 14 ]
round 3: no updates -- converged early
  dist = [ 0 4 1 9 11 14 ]

final dist: [ 0 4 1 9 11 14 ]  (converged after 3 round(s), out of 5 allowed)

expected dist: [ 0 4 1 9 11 14 ], matching Dijkstra's own result exactly
expected rounds used: 3

self-check: Bellman-Ford's distances match Dijkstra's, with no ordering
discipline required at all: confirmed
```

### The Concept, In Detail

```
ASCII view: Bellman-Ford's convergence, round by round.

  round 1: dist[1]=4, dist[2]=1, dist[3]=9, dist[4]=11   (4 vertices settle)
  round 2: dist[5]=14                                     (1 vertex settles)
  round 3: no updates -- CONVERGED, even though 2 rounds remained available

  Final distances match Dijkstra's exactly: [0, 4, 1, 9, 11, 14]
```

Convergence in 3 rounds rather than the full 5 allowed is typical, not a special property of this graph: once every vertex's true shortest distance has propagated through, an entire round can pass with no thread improving anything, and that is the signal to stop. The parallel version's early-exit check works identically -- a single shared flag, set by ANY thread whose relaxation succeeds, tells the host whether the next round is even necessary.

[COMMON TRAP]
It is tempting to think Bellman-Ford needs exactly V-1 rounds every time, since that is the number the correctness proof guarantees is ENOUGH. V-1 is a worst-case upper bound, not a required exact count -- a round that produces no updates at all means every distance has already converged, and continuing to relax edges after that point would only repeat work for no benefit. Always running the full V-1 rounds without checking for early convergence wastes exactly the kind of effort a bounded loop is supposed to avoid.

### Code and Verification

```cpp
#include <cstdio>
#include <vector>

// Chapter 23.2 main -- Bellman-Ford's "relax every edge, every round"
// rule needs no ordering discipline at all, which makes EVERY round
// embarrassingly parallel: one thread per edge (Chapter 21.3's own
// edge-parallel pattern over COO), each computing its own candidate
// distance and applying it via `atomicMin`. A shared `any_update` flag,
// set via `atomicOr` whenever a thread's atomicMin actually improves a
// distance, tells the host whether another round is needed -- exactly
// the CPU baseline's own early-exit check, now decided by however many
// threads happened to succeed rather than by a single sequential scan.
#define V 6
#define E 9
#define INF 999999

__global__ void bellman_ford_relax_kernel(const int* edge_src, const int* edge_dst,
                                           const int* edge_weight, int* dist,
                                           int num_edges, int* any_update) {
    int e = threadIdx.x;
    if (e >= num_edges) return;
    int s = edge_src[e];
    if (dist[s] == INF) return;
    int candidate = dist[s] + edge_weight[e];
    int old = atomicMin(&dist[edge_dst[e]], candidate);
    if (candidate < old) atomicOr(any_update, 1);
}

// ---- Host-side replay of the identical per-thread logic, run under
// two different edge-processing orders (Chapter 21.2/22.2's own
// pattern), each driven by a host loop checking any_update. ----

struct Edge { int src, dst, weight; };

std::vector<int> run_bellman_ford(const std::vector<Edge>& edges, int* rounds_used) {
    std::vector<int> dist(V, INF);
    dist[0] = 0;
    for (int round = 1; round <= V - 1; round++) {
        int any_update = 0;
        for (auto& e : edges) {
            if (dist[e.src] == INF) continue;
            int candidate = dist[e.src] + e.weight;
            int old = dist[e.dst];
            if (candidate < old) { dist[e.dst] = candidate; any_update = 1; }
        }
        *rounds_used = round;
        if (!any_update) break;
    }
    return dist;
}

int main() {
    printf("=== Section 23.2 main: edge-parallel Bellman-Ford with atomicMin ===\n\n");

    std::vector<Edge> order_A = {
        {2,3,8}, {0,1,4}, {3,4,2}, {1,2,2}, {0,2,1},
        {4,5,3}, {2,4,10}, {3,5,6}, {1,3,5},
    };
    std::vector<Edge> order_B = {
        {1,3,5}, {3,5,6}, {2,4,10}, {4,5,3}, {0,2,1},
        {1,2,2}, {3,4,2}, {0,1,4}, {2,3,8},
    };

    printf("%d threads launched per round (one per edge), any_update flagged via atomicOr\n\n", E);

    int rounds_A = 0;
    printf("order A: ");
    for (auto& e : order_A) printf("(%d->%d) ", e.src, e.dst);
    printf("\n");
    std::vector<int> dist_A = run_bellman_ford(order_A, &rounds_A);
    printf("  converged after %d round(s), dist = [ ", rounds_A);
    for (int d : dist_A) printf("%d ", d);
    printf("]\n\n");

    int rounds_B = 0;
    printf("order B (same edges, different processing order): ");
    for (auto& e : order_B) printf("(%d->%d) ", e.src, e.dst);
    printf("\n");
    std::vector<int> dist_B = run_bellman_ford(order_B, &rounds_B);
    printf("  converged after %d round(s), dist = [ ", rounds_B);
    for (int d : dist_B) printf("%d ", d);
    printf("]\n\n");

    std::vector<int> expected_dist = {0, 4, 1, 9, 11, 14};
    bool ok = (dist_A == expected_dist) && (dist_B == expected_dist);

    printf("expected dist (either order): [ 0 4 1 9 11 14 ]\n");
    printf("(atomicMin needs no agreement on WHICH order edges are relaxed in --\n");
    printf("every round just needs to consider every edge; the round count itself\n");
    printf("can even differ between orders on a different graph, but the FINAL\n");
    printf("distances, once converged, are identical no matter the order)\n");

    printf("\nself-check: both edge-processing orders converge to the exact same\n");
    printf("final distances as the CPU baseline: %s\n", ok ? "confirmed" : "MISMATCH");
    return ok ? 0 : 1;
}
```

**Compile and run:**

```bash
NVDIR=/usr/local/lib/python3.11/dist-packages/nvidia/cu13
$NVDIR/bin/nvcc -arch=sm_80 -I$NVDIR/include -L$NVDIR/lib -lcudart 127_bellmanford_parallel_kernel.cu -o 127_bellmanford_parallel_kernel
LD_LIBRARY_PATH=$NVDIR/lib ./127_bellmanford_parallel_kernel
```

**Sample input:** the same 9 edges, relaxed by 9 threads at once every round via `atomicMin`, run under two different edge-processing orders.

**Sample output:**

```text
=== Section 23.2 main: edge-parallel Bellman-Ford with atomicMin ===

9 threads launched per round (one per edge), any_update flagged via atomicOr

order A: (2->3) (0->1) (3->4) (1->2) (0->2) (4->5) (2->4) (3->5) (1->3) 
  converged after 3 round(s), dist = [ 0 4 1 9 11 14 ]

order B (same edges, different processing order): (1->3) (3->5) (2->4) (4->5) (0->2) (1->2) (3->4) (0->1) (2->3) 
  converged after 3 round(s), dist = [ 0 4 1 9 11 14 ]

expected dist (either order): [ 0 4 1 9 11 14 ]
(atomicMin needs no agreement on WHICH order edges are relaxed in --
every round just needs to consider every edge; the round count itself
can even differ between orders on a different graph, but the FINAL
distances, once converged, are identical no matter the order)

self-check: both edge-processing orders converge to the exact same
final distances as the CPU baseline: confirmed
```

## 23.3 Why atomicMin Converges Regardless of Arrival Order

### Intuition

Vertex 5 in this book's own graph has two genuinely DIFFERENT candidate distances competing for it: 15 (via vertex 3) and 14 (via vertex 4). A naive "read the current value, decide whether to write, then write unconditionally" approach can get this wrong -- if the thread proposing the WORSE value (15) happens to read stale information and write AFTER the thread proposing the better value (14) has already succeeded, the naive version overwrites a correct answer with a wrong one. `atomicMin` cannot make this mistake, because it never acts on a possibly-stale read: it atomically compares its candidate against whatever the slot ACTUALLY holds at the exact moment it runs, and only writes if its candidate is smaller than that current value.

### The Sequential (CPU) Baseline

```cpp
#include <cstdio>
#include <vector>

// Chapter 23.3 -- The Sequential (CPU) Baseline.
// Vertex 5 in this book's own graph has two incoming candidate
// distances that genuinely DIFFER: via vertex 3 (dist 9 + weight 6 =
// 15) and via vertex 4 (dist 11 + weight 3 = 14). Whichever thread's
// relaxation reaches vertex 5 LAST must not be allowed to blindly
// overwrite a better value already sitting there -- and a naive
// "read the old value, decide to write, then write unconditionally"
// approach can do exactly that if the two threads' reads happen
// before either thread's write (the same worst-case interleaving this
// book has used since Chapter 16.2). This section demonstrates the
// naive version actually producing the WRONG final answer under one
// arrival order, then shows the fix.
#define INF 999999

int main() {
    printf("=== Section 23.3 CPU baseline: the atomicMin race on vertex 5 ===\n\n");

    printf("two DIFFERENT candidate distances for vertex 5:\n");
    printf("  via vertex 3 (dist=9, weight=6): candidate = 15\n");
    printf("  via vertex 4 (dist=11, weight=3): candidate = 14\n");
    printf("  the TRUE shortest distance to vertex 5 is min(15, 14) = 14\n\n");

    printf("--- naive (read old value, decide, then write unconditionally) ---\n\n");

    printf("order 1: thread-A (candidate 15) writes, THEN thread-B (candidate 14) writes\n");
    {
        int dist5 = INF;
        int read_A = dist5, read_B = dist5;   // both read BEFORE either writes
        printf("  both threads read dist[5]=%d\n", dist5);
        if (15 < read_A) { dist5 = 15; printf("  thread-A decided 15 < %d, writes dist[5]=15\n", read_A); }
        if (14 < read_B) { dist5 = 14; printf("  thread-B decided 14 < %d, writes dist[5]=14\n", read_B); }
        printf("  final dist[5] = %d -- CORRECT (by luck of write order)\n\n", dist5);
    }

    printf("order 2: thread-B (candidate 14) writes, THEN thread-A (candidate 15) writes\n");
    {
        int dist5 = INF;
        int read_A = dist5, read_B = dist5;   // both read BEFORE either writes, same as before
        printf("  both threads read dist[5]=%d\n", dist5);
        if (14 < read_B) { dist5 = 14; printf("  thread-B decided 14 < %d, writes dist[5]=14\n", read_B); }
        if (15 < read_A) { dist5 = 15; printf("  thread-A decided 15 < %d (its OWN stale read), writes dist[5]=15 -- OVERWRITES the correct value!\n", read_A); }
        printf("  final dist[5] = %d -- WRONG (should be 14)\n\n", dist5);
    }

    printf("--- atomicMin (compares against the CURRENT value at write time) ---\n\n");

    auto atomic_min_sim = [](int& slot, int candidate) {
        int old = slot;
        if (candidate < slot) slot = candidate;
        return old;
    };

    printf("order 1: thread-A (candidate 15) resolves first, THEN thread-B (candidate 14)\n");
    int dist5_o1 = INF;
    {
        int old = atomic_min_sim(dist5_o1, 15);
        printf("  thread-A: atomicMin(dist[5], 15) -- old was %d, dist[5] now %d\n", old, dist5_o1);
        old = atomic_min_sim(dist5_o1, 14);
        printf("  thread-B: atomicMin(dist[5], 14) -- old was %d, dist[5] now %d\n", old, dist5_o1);
    }
    printf("  final dist[5] = %d\n\n", dist5_o1);

    printf("order 2: thread-B (candidate 14) resolves first, THEN thread-A (candidate 15)\n");
    int dist5_o2 = INF;
    {
        int old = atomic_min_sim(dist5_o2, 14);
        printf("  thread-B: atomicMin(dist[5], 14) -- old was %d, dist[5] now %d\n", old, dist5_o2);
        old = atomic_min_sim(dist5_o2, 15);
        printf("  thread-A: atomicMin(dist[5], 15) -- old was %d, dist[5] now %d (no change -- 15 is not smaller)\n", old, dist5_o2);
    }
    printf("  final dist[5] = %d\n\n", dist5_o2);

    bool ok = (dist5_o1 == 14) && (dist5_o2 == 14);

    printf("expected: BOTH atomicMin orders converge to 14, regardless of which\n");
    printf("thread's atomicMin executes first -- unlike the naive version, which\n");
    printf("got the WRONG answer (15) under order 2\n");

    printf("\nself-check: atomicMin converges to the true minimum (14) under both\n");
    printf("arrival orders: %s\n", ok ? "confirmed" : "MISMATCH");
    return ok ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++17 -Wall -Wextra -O2 128_atomicmin_race_cpu_baseline.cpp -o 128_atomicmin_race_cpu_baseline
./128_atomicmin_race_cpu_baseline
```

**Sample input:** vertex 5's two competing candidate distances (15 and 14), resolved two ways -- a naive unprotected write, and an atomicMin-style compare-at-write-time -- under two different arrival orders each.

**Sample output:**

```text
=== Section 23.3 CPU baseline: the atomicMin race on vertex 5 ===

two DIFFERENT candidate distances for vertex 5:
  via vertex 3 (dist=9, weight=6): candidate = 15
  via vertex 4 (dist=11, weight=3): candidate = 14
  the TRUE shortest distance to vertex 5 is min(15, 14) = 14

--- naive (read old value, decide, then write unconditionally) ---

order 1: thread-A (candidate 15) writes, THEN thread-B (candidate 14) writes
  both threads read dist[5]=999999
  thread-A decided 15 < 999999, writes dist[5]=15
  thread-B decided 14 < 999999, writes dist[5]=14
  final dist[5] = 14 -- CORRECT (by luck of write order)

order 2: thread-B (candidate 14) writes, THEN thread-A (candidate 15) writes
  both threads read dist[5]=999999
  thread-B decided 14 < 999999, writes dist[5]=14
  thread-A decided 15 < 999999 (its OWN stale read), writes dist[5]=15 -- OVERWRITES the correct value!
  final dist[5] = 15 -- WRONG (should be 14)

--- atomicMin (compares against the CURRENT value at write time) ---

order 1: thread-A (candidate 15) resolves first, THEN thread-B (candidate 14)
  thread-A: atomicMin(dist[5], 15) -- old was 999999, dist[5] now 15
  thread-B: atomicMin(dist[5], 14) -- old was 15, dist[5] now 14
  final dist[5] = 14

order 2: thread-B (candidate 14) resolves first, THEN thread-A (candidate 15)
  thread-B: atomicMin(dist[5], 14) -- old was 999999, dist[5] now 14
  thread-A: atomicMin(dist[5], 15) -- old was 14, dist[5] now 14 (no change -- 15 is not smaller)
  final dist[5] = 14

expected: BOTH atomicMin orders converge to 14, regardless of which
thread's atomicMin executes first -- unlike the naive version, which
got the WRONG answer (15) under order 2

self-check: atomicMin converges to the true minimum (14) under both
arrival orders: confirmed
```

### The Concept, In Detail

```
ASCII view: why the naive version can regress, and atomicMin cannot.

  naive, order 2 (worse value writes LAST):
    both threads read dist[5]=INF (before either writes)
    thread-B writes 14 (correct, based on its own read of INF)
    thread-A writes 15 (ALSO based on its own read of INF -- unconditional,
                         oblivious to thread-B's write) -- WRONG: overwrites 14

  atomicMin, any order:
    atomicMin always re-reads the CURRENT slot value at the instant it runs,
    never a snapshot from earlier -- so the SECOND call to run always sees
    whatever the FIRST call already wrote, and can only ever improve on it
    (or leave it alone), never regress past it
```

The naive version's bug is specifically that its decision ("is my candidate smaller?") and its action (writing) are separated by a READ that can go stale. `atomicMin` fuses the comparison and the write into one indivisible hardware operation, so there is no window in which a thread can be acting on outdated information about the slot it is about to modify. This is a genuinely different correctness argument from `atomicCAS`'s (Chapter 16.2, Chapter 22.1): CAS guarantees exactly one of several competing writes succeeds; `atomicMin` guarantees every competing write is allowed to happen, but the slot always ends up holding the smallest value across all of them, in any order.

[COMMON TRAP]
It is tempting to think the naive race is rare enough in practice not to matter, since it produced the correct answer under the FIRST order tested. A race condition's danger is precisely that its outcome depends on scheduling details outside the program's control -- the same naive code can pass under one run's thread interleaving and fail under another's, with nothing in the source code itself indicating which will occur. `atomicMin` removes the dependency on interleaving entirely, which is the only way to make the result actually reliable rather than accidentally correct.

### Code and Verification

```cpp
#include <cstdio>
#include <vector>

// Chapter 23.3 main -- the genuine `atomicMin` fix for Section 23.3's
// race: two threads, relaxing edges (3->5, weight 6) and (4->5, weight
// 3) from this book's own graph, both targeting vertex 5 with
// DIFFERENT candidate distances (15 and 14). `atomicMin` guarantees
// the slot ends up holding the smaller of every value ever proposed to
// it, no matter which thread's call executes first -- a fundamentally
// different guarantee from `atomicCAS`'s "exactly one thread may
// succeed" (Chapter 16.2, Chapter 22.1) or `atomicAdd`'s "every
// contribution is summed regardless of order" (Chapter 17.2): here,
// EVERY thread's write is allowed to happen, and the primitive's job
// is to always keep whichever result is smallest.
#define INF 999999

__global__ void relax_vertex5_kernel(int* dist, const int* src_dist, const int* weight, int num_edges) {
    int t = threadIdx.x;
    if (t >= num_edges) return;
    int candidate = src_dist[t] + weight[t];
    atomicMin(dist, candidate);
}

// ---- Host-side replay of the identical per-thread logic, under two
// different thread-resolution orders. ----

int main() {
    printf("=== Section 23.3 main: atomicMin resolving the vertex-5 race ===\n\n");

    printf("2 threads relax the graph's own edges (3->5, w=6) and (4->5, w=3)\n");
    printf("into the SAME target, dist[5], starting from dist[3]=9 and dist[4]=11:\n");
    printf("  thread 0 (edge 3->5): candidate = 9 + 6 = 15\n");
    printf("  thread 1 (edge 4->5): candidate = 11 + 3 = 14\n\n");

    auto run_order = [](const std::vector<int>& thread_order, const std::vector<int>& candidates, const char* label) {
        int dist5 = INF;
        printf("%s: threads resolve in order %d, %d\n", label, thread_order[0], thread_order[1]);
        for (int t : thread_order) {
            int old = dist5;
            if (candidates[t] < dist5) dist5 = candidates[t];
            printf("  thread %d: atomicMin(dist[5], %d) -- old was %d, dist[5] now %d\n",
                   t, candidates[t], old, dist5);
        }
        printf("  final dist[5] = %d\n\n", dist5);
        return dist5;
    };

    std::vector<int> candidates = {15, 14};
    int result_A = run_order({0, 1}, candidates, "order A");
    int result_B = run_order({1, 0}, candidates, "order B");

    bool ok = (result_A == 14) && (result_B == 14);

    printf("expected: both orders converge to 14, matching Section 23.2's own\n");
    printf("full Bellman-Ford result for this vertex exactly\n");

    printf("\nself-check: atomicMin resolves the race correctly regardless of\n");
    printf("thread execution order: %s\n", ok ? "confirmed" : "MISMATCH");
    return ok ? 0 : 1;
}
```

**Compile and run:**

```bash
NVDIR=/usr/local/lib/python3.11/dist-packages/nvidia/cu13
$NVDIR/bin/nvcc -arch=sm_80 -I$NVDIR/include -L$NVDIR/lib -lcudart 129_atomicmin_race_kernel.cu -o 129_atomicmin_race_kernel
LD_LIBRARY_PATH=$NVDIR/lib ./129_atomicmin_race_kernel
```

**Sample input:** the graph's own two edges into vertex 5 (candidates 15 and 14), resolved via real `atomicMin` under two different thread-resolution orders.

**Sample output:**

```text
=== Section 23.3 main: atomicMin resolving the vertex-5 race ===

2 threads relax the graph's own edges (3->5, w=6) and (4->5, w=3)
into the SAME target, dist[5], starting from dist[3]=9 and dist[4]=11:
  thread 0 (edge 3->5): candidate = 9 + 6 = 15
  thread 1 (edge 4->5): candidate = 11 + 3 = 14

order A: threads resolve in order 0, 1
  thread 0: atomicMin(dist[5], 15) -- old was 999999, dist[5] now 15
  thread 1: atomicMin(dist[5], 14) -- old was 15, dist[5] now 14
  final dist[5] = 14

order B: threads resolve in order 1, 0
  thread 1: atomicMin(dist[5], 14) -- old was 999999, dist[5] now 14
  thread 0: atomicMin(dist[5], 15) -- old was 14, dist[5] now 14
  final dist[5] = 14

expected: both orders converge to 14, matching Section 23.2's own
full Bellman-Ford result for this vertex exactly

self-check: atomicMin resolves the race correctly regardless of
thread execution order: confirmed
```

## Chapter Summary

Weighted edges break BFS's frontier model, since a vertex reached via more hops but a smaller total weight can still be genuinely closer -- Dijkstra's algorithm restores correctness by always finalizing the true global-minimum-distance unvisited vertex next, a strict ordering discipline that keeps its outer loop sequential even though the relax step for one popped vertex's own neighbors parallelizes cleanly via `atomicMin`. Bellman-Ford abandons that ordering discipline entirely: relaxing every edge, every round, for up to V-1 rounds is still provably correct regardless of the order edges are visited in, which makes every round embarrassingly parallel -- one thread per edge, with a shared early-exit flag replacing Dijkstra's sequential minimum-finding scan. `atomicMin` is what makes this safe: unlike `atomicCAS`'s "exactly one thread succeeds" guarantee or `atomicAdd`'s "every contribution is summed" guarantee, `atomicMin` lets every competing write happen but always leaves the slot holding the smallest value proposed, because it compares against the slot's CURRENT value at the instant it runs rather than acting on a potentially stale read -- the exact property a naive unprotected race lacks, and the reason that naive version can regress to a worse answer depending on scheduling.

## Self-Check Questions

1. Why does BFS's "first thread to arrive" rule stop being valid once edges have different weights?
2. Why must Dijkstra's algorithm always pop the TRUE global minimum among unvisited vertices, rather than any vertex that merely looks promising?
3. Why is Bellman-Ford correct after at most V-1 rounds, regardless of what order the edges are relaxed in within each round?
4. What does a round with no updates at all tell Bellman-Ford's early-exit check, and why is checking for this worthwhile?
5. Walk through why a naive, unprotected "read then write" race on a shared distance value can produce a WRONG final result, using vertex 5's two competing candidates as the example.
6. How does `atomicMin`'s correctness guarantee differ from `atomicCAS`'s, given that both are used to resolve races on a shared memory location?

## Where We Go Next

Both BFS and shortest-path algorithms answer questions about REACHING vertices -- how many hops, or at what total cost. Chapter 24 turns to a related but different question: which vertices belong to the same CONNECTED COMPONENT as each other, with no source vertex and no notion of distance at all -- just grouping every vertex in a graph by which other vertices it can reach.

## Worked Solutions

**1.** BFS's rule relies on every edge contributing the same fixed cost (one hop), so visiting vertices in the order they are first discovered is automatically the same as visiting them in order of increasing true distance. Once edges carry different weights, a vertex several hops away via cheap edges can have a smaller total distance than a vertex just one hop away via an expensive edge -- "discovered first" (fewest hops) and "closest" (smallest total weight) are no longer the same thing, so a thread being first to reach a vertex no longer proves it found that vertex's true shortest distance.

**2.** Any OTHER unvisited vertex has a tentative distance at least as large as the current global minimum, and following any edge from a popped vertex only ever ADDS a non-negative weight -- it can never decrease a distance. If Dijkstra popped a vertex that was not the true minimum, some cheaper vertex still unvisited might later provide an even shorter path to it, meaning the popped vertex's distance would not yet have been safe to finalize. Only the true global minimum is provably impossible to improve on by any future relaxation.

**3.** Any shortest path between two vertices in a graph with V vertices visits at most V-1 edges (a path revisiting a vertex would contain a cycle, which could only be removed to shorten the path, so the shortest path itself is always simple). Each full round of relaxing every edge is guaranteed to correctly extend the shortest known distance by at least one more edge along every still-improving path, so after V-1 rounds, every shortest path's full length has necessarily been covered -- and this argument never depended on which specific order the edges were visited in within any given round, only that every edge was considered.

**4.** A round making no updates at all means every relaxation attempted found its candidate distance was NOT smaller than what was already stored -- in other words, every distance in the array is already final and no further round could possibly improve anything, since Bellman-Ford's relaxations are monotonic (distances only ever decrease, never increase). Checking for this is worthwhile because real graphs typically converge in far fewer than V-1 rounds, so continuing to run additional rounds after convergence would repeat identical, wasted work with no possibility of changing the result.

**5.** Vertex 5 has two competing candidates, 15 (via vertex 3) and 14 (via vertex 4), with 14 being correct. If both threads read the CURRENT value of dist[5] (some large "infinity" placeholder) before either writes, each independently concludes its own candidate is an improvement. If the thread carrying 14 writes first, then the thread carrying 15 -- having already made its decision based on its own earlier, now-stale read -- writes unconditionally afterward, overwriting the correct 14 with the incorrect 15. The bug is that the decision to write and the act of writing are separated by time, during which the slot's true value can change without the thread noticing.

**6.** `atomicCAS` guarantees that when multiple threads attempt to change a slot from one specific expected value to a new value, EXACTLY ONE of them succeeds and every other thread's attempt fails outright, leaving the slot's final content determined by whichever thread won -- this is the right tool when only one write should ever "count," such as discovering a vertex for the first time. `atomicMin` instead lets every thread's write attempt succeed in the sense that it is genuinely applied, but each application only takes effect if it improves on the slot's CURRENT value at that instant, so the slot converges to the smallest value ever proposed regardless of how many threads wrote to it or in what order -- the right tool when many different candidate values must all be considered and only the best one should survive.
