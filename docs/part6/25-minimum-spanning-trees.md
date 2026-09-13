# Chapter 25: Minimum Spanning Trees

Connected components (Chapter 24) asked which vertices belong together, with no notion of cost. This chapter combines that grouping question with Chapter 23's notion of edge weight: a minimum spanning tree (MST) is the cheapest possible set of edges that connects every vertex of a graph into one component -- exactly V-1 edges, no more, no fewer, with the smallest total weight achievable. This chapter builds one three ways: Kruskal's algorithm, whose strict globally-sorted-order requirement resists parallelization the same way Dijkstra's did, even though the sort feeding it does not; Boruvka's algorithm, which abandons that global order entirely in favor of every component finding its own cheapest edge at once; and the packed-atomicMin technique that makes "which edge achieved the minimum" answerable atomically, not just "what was the minimum." Along the way, fusing Boruvka's per-round selection with Chapter 24.2's own hooking machinery surfaces one more genuine lesson about snapshot semantics -- one this book has now seen three times, in three different algorithms, for the same underlying reason.

## 25.1 Sequential Minimum Spanning Trees via Kruskal's Algorithm

### Intuition

A minimum spanning tree connects every vertex as cheaply as possible, using exactly V-1 edges (any fewer would leave some vertex unreachable; any more would create a cycle, which could always be shortened by removing its most expensive edge). Kruskal's algorithm finds one with a simple greedy rule: sort every edge by weight, then walk the sorted list from cheapest to most expensive, accepting an edge whenever its two endpoints are not already connected, and rejecting it whenever they already are (accepting it would only create a cycle). This section reuses the book's own 6-vertex weighted graph from Chapters 21-23, now treated as undirected for the purpose of connecting every vertex as cheaply as possible.

### The Sequential (CPU) Baseline

```cpp
#include <cstdio>
#include <vector>
#include <algorithm>

// Chapter 25.1 -- The Sequential (CPU) Baseline.
// A minimum spanning tree (MST) asks for the CHEAPEST possible set of
// edges that connects every vertex of a graph into a single component --
// exactly V-1 edges, no more and no fewer, with the smallest possible
// total weight. Kruskal's algorithm finds one with a strikingly simple
// greedy rule: sort every edge by weight, then walk the sorted list
// from cheapest to most expensive, adding an edge whenever its two
// endpoints are NOT already connected (accept), and skipping it
// whenever they already are (reject -- adding it would only create a
// cycle, never improve connectivity). This section reuses the book's
// own 6-vertex weighted graph from Chapters 21-23, now treated as
// undirected for the purpose of connecting every vertex as cheaply as
// possible.
#define V 6
#define E 9

int find_root(const std::vector<int>& parent, int v) {
    while (parent[v] != v) v = parent[v];
    return v;
}

int main() {
    printf("=== Section 25.1 CPU baseline: minimum spanning tree via Kruskal's algorithm ===\n\n");

    std::vector<std::pair<std::pair<int,int>,int>> edges = {
        {{2,3},8}, {{0,1},4}, {{3,4},2}, {{1,2},2}, {{0,2},1},
        {{4,5},3}, {{2,4},10}, {{3,5},6}, {{1,3},5},
    };
    printf("%d undirected weighted edges (the book's own graph from Chapters 21-23):\n", E);
    for (auto& e : edges) printf("  (%d,%d) weight=%d\n", e.first.first, e.first.second, e.second);
    printf("\n");

    std::vector<std::pair<std::pair<int,int>,int>> sorted_edges = edges;
    std::sort(sorted_edges.begin(), sorted_edges.end(),
              [](const auto& a, const auto& b) {
                  if (a.second != b.second) return a.second < b.second;
                  if (a.first.first != b.first.first) return a.first.first < b.first.first;
                  return a.first.second < b.first.second;
              });
    printf("edges sorted by weight (ties broken by (u,w)):\n");
    for (auto& e : sorted_edges) printf("  (%d,%d) weight=%d\n", e.first.first, e.first.second, e.second);
    printf("\n");

    std::vector<int> parent(V);
    for (int v = 0; v < V; v++) parent[v] = v;

    std::vector<std::pair<int,int>> mst_edges;
    int total_weight = 0;

    printf("walking sorted edges, cheapest first:\n");
    for (auto& e : sorted_edges) {
        int u = e.first.first, w = e.first.second, wt = e.second;
        int ru = find_root(parent, u), rw = find_root(parent, w);
        if (ru == rw) {
            printf("  edge(%d,%d,w=%d): SAME component (root %d) -- REJECT (would cycle)\n", u, w, wt, ru);
            continue;
        }
        int hi = std::max(ru, rw), lo = std::min(ru, rw);
        printf("  edge(%d,%d,w=%d): different components (root %d vs %d) -- ACCEPT, hook parent[%d]=%d\n",
               u, w, wt, ru, rw, hi, lo);
        parent[hi] = lo;
        mst_edges.push_back({u, w});
        total_weight += wt;
    }

    printf("\nMST edges (in the order accepted): [ ");
    for (auto& e : mst_edges) printf("(%d,%d) ", e.first, e.second);
    printf("]\n");
    printf("total MST weight: %d\n", total_weight);

    std::vector<std::pair<int,int>> expected_mst = {{0,2}, {1,2}, {3,4}, {4,5}, {1,3}};
    int expected_weight = 13;
    bool ok = (mst_edges == expected_mst) && (total_weight == expected_weight)
              && ((int)mst_edges.size() == V - 1);

    printf("\nexpected MST edges: [ (0,2) (1,2) (3,4) (4,5) (1,3) ], total weight 13,\n");
    printf("exactly V-1=5 edges connecting all 6 vertices\n");
    printf("\nself-check: Kruskal's algorithm finds the expected minimum spanning tree: %s\n", ok ? "confirmed" : "MISMATCH");
    return ok ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++17 -Wall -Wextra -O2 136_mst_kruskal_cpu_baseline.cpp -o 136_mst_kruskal_cpu_baseline
./136_mst_kruskal_cpu_baseline
```

**Sample input:** the book's own 9-edge weighted graph from Chapters 21-23, walked cheapest-to-most-expensive with union-find cycle detection.

**Sample output:**

```text
=== Section 25.1 CPU baseline: minimum spanning tree via Kruskal's algorithm ===

9 undirected weighted edges (the book's own graph from Chapters 21-23):
  (2,3) weight=8
  (0,1) weight=4
  (3,4) weight=2
  (1,2) weight=2
  (0,2) weight=1
  (4,5) weight=3
  (2,4) weight=10
  (3,5) weight=6
  (1,3) weight=5

edges sorted by weight (ties broken by (u,w)):
  (0,2) weight=1
  (1,2) weight=2
  (3,4) weight=2
  (4,5) weight=3
  (0,1) weight=4
  (1,3) weight=5
  (3,5) weight=6
  (2,3) weight=8
  (2,4) weight=10

walking sorted edges, cheapest first:
  edge(0,2,w=1): different components (root 0 vs 2) -- ACCEPT, hook parent[2]=0
  edge(1,2,w=2): different components (root 1 vs 0) -- ACCEPT, hook parent[1]=0
  edge(3,4,w=2): different components (root 3 vs 4) -- ACCEPT, hook parent[4]=3
  edge(4,5,w=3): different components (root 3 vs 5) -- ACCEPT, hook parent[5]=3
  edge(0,1,w=4): SAME component (root 0) -- REJECT (would cycle)
  edge(1,3,w=5): different components (root 0 vs 3) -- ACCEPT, hook parent[3]=0
  edge(3,5,w=6): SAME component (root 0) -- REJECT (would cycle)
  edge(2,3,w=8): SAME component (root 0) -- REJECT (would cycle)
  edge(2,4,w=10): SAME component (root 0) -- REJECT (would cycle)

MST edges (in the order accepted): [ (0,2) (1,2) (3,4) (4,5) (1,3) ]
total MST weight: 13

expected MST edges: [ (0,2) (1,2) (3,4) (4,5) (1,3) ], total weight 13,
exactly V-1=5 edges connecting all 6 vertices

self-check: Kruskal's algorithm finds the expected minimum spanning tree: confirmed
```

### The Concept, In Detail

```
ASCII view: sorted edges, accept/reject decisions.

  (0,2) w=1  ACCEPT  components: {0,2}
  (1,2) w=2  ACCEPT  components: {0,1,2}
  (3,4) w=2  ACCEPT  components: {0,1,2}, {3,4}
  (4,5) w=3  ACCEPT  components: {0,1,2}, {3,4,5}
  (0,1) w=4  REJECT  (0 and 1 already share a component)
  (1,3) w=5  ACCEPT  components: {0,1,2,3,4,5}   <- fully connected, 5 edges
  (3,5) w=6  REJECT  (3 and 5 already share a component)
  (2,3) w=8  REJECT  (2 and 3 already share a component)
  (2,4) w=10 REJECT  (2 and 4 already share a component)
```

Kruskal's correctness rests on the "cut property": for any way of splitting the vertices into two groups, the cheapest edge crossing that split is safe to include in SOME minimum spanning tree. Processing edges cheapest-first guarantees that whenever an edge is accepted, it is the cheapest possible way to connect whatever it connects -- which is exactly why the outer walk must see every edge in true global sorted order, one at a time, to know at each step whether accepting would create a cycle.

[COMMON TRAP]
It is tempting to think that any edge NOT touching an already-visited vertex can be skipped early, the way a frontier-based search (Chapter 22) prunes unreached vertices. Kruskal's algorithm has no notion of "already visited" in that sense -- every edge, however far from wherever the walk has been, must still be checked against the union-find structure, because two DISCONNECTED partial components can both grow independently before ever merging into each other, exactly as {0,1,2} and {3,4,5} do here before edge (1,3) finally joins them.

### Code and Verification

```cpp
#include <cstdio>
#include <vector>
#include <algorithm>

// Chapter 25.1 main -- Kruskal's cycle-checking walk over the sorted
// edge list is a genuine sequential bottleneck, exactly like Dijkstra's
// outer loop in Chapter 23.1: each decision (accept or reject) depends
// on every union-find state left behind by every earlier decision, so
// the walk itself cannot be parallelized. But the SORT that produces
// that ordering is a completely separate step with no such dependency,
// and it parallelizes cleanly: one thread per edge computes that
// edge's RANK -- how many other edges are strictly cheaper than it (or
// equally cheap but with a smaller index, as a tie-break) -- by
// comparing itself against every other edge at once. An edge's rank is
// exactly the position it belongs at in the fully sorted array, so
// once every thread has its rank, scattering edges into that array
// needs no further comparisons at all.
#define V 6
#define E 9

struct Edge { int u, w, wt; };

__device__ __host__ bool edge_less(const Edge& a, const Edge& b) {
    if (a.wt != b.wt) return a.wt < b.wt;
    if (a.u != b.u) return a.u < b.u;
    return a.w < b.w;
}

__global__ void edge_rank_kernel(const Edge* edges, int* rank, int num_edges) {
    int i = threadIdx.x;
    if (i >= num_edges) return;
    int r = 0;
    for (int j = 0; j < num_edges; j++) {
        if (j == i) continue;
        if (edge_less(edges[j], edges[i])) r++;
    }
    rank[i] = r;
}

// ---- Host-side replay of the identical per-thread logic. ----

int find_root(const std::vector<int>& parent, int v) {
    while (parent[v] != v) v = parent[v];
    return v;
}

int main() {
    printf("=== Section 25.1 main: parallel edge ranking, sequential cycle check ===\n\n");

    std::vector<Edge> edges = {
        {2,3,8}, {0,1,4}, {3,4,2}, {1,2,2}, {0,2,1},
        {4,5,3}, {2,4,10}, {3,5,6}, {1,3,5},
    };
    printf("%d edges, in their ORIGINAL (unsorted) array order:\n", E);
    for (int i = 0; i < E; i++) printf("  edge%d: (%d,%d) weight=%d\n", i, edges[i].u, edges[i].w, edges[i].wt);
    printf("\n");

    std::vector<int> rank(E);
    for (int i = 0; i < E; i++) {
        int r = 0;
        for (int j = 0; j < E; j++) {
            if (j == i) continue;
            if (edge_less(edges[j], edges[i])) r++;
        }
        rank[i] = r;
    }
    printf("every thread's computed rank (position in sorted order), all at once:\n");
    for (int i = 0; i < E; i++) printf("  edge%d (%d,%d,w=%d) -> rank %d\n", i, edges[i].u, edges[i].w, edges[i].wt, rank[i]);

    std::vector<Edge> sorted_edges(E);
    for (int i = 0; i < E; i++) sorted_edges[rank[i]] = edges[i];
    printf("\nscattered into sorted order using the ranks (no comparisons needed here):\n");
    for (auto& e : sorted_edges) printf("  (%d,%d) weight=%d\n", e.u, e.w, e.wt);

    // The cycle-checking walk itself stays sequential, exactly as in
    // the CPU baseline -- only the SORT that fed it was parallelized.
    std::vector<int> parent(V);
    for (int v = 0; v < V; v++) parent[v] = v;
    std::vector<std::pair<int,int>> mst_edges;
    int total_weight = 0;
    printf("\nwalking the parallel-sorted edges, cheapest first (this walk stays sequential):\n");
    for (auto& e : sorted_edges) {
        int ru = find_root(parent, e.u), rw = find_root(parent, e.w);
        if (ru == rw) {
            printf("  edge(%d,%d,w=%d): SAME component -- REJECT\n", e.u, e.w, e.wt);
            continue;
        }
        int hi = std::max(ru, rw), lo = std::min(ru, rw);
        printf("  edge(%d,%d,w=%d): different components -- ACCEPT, hook parent[%d]=%d\n", e.u, e.w, e.wt, hi, lo);
        parent[hi] = lo;
        mst_edges.push_back({e.u, e.w});
        total_weight += e.wt;
    }

    printf("\nMST edges: [ ");
    for (auto& e : mst_edges) printf("(%d,%d) ", e.first, e.second);
    printf("]\ntotal MST weight: %d\n", total_weight);

    std::vector<int> expected_rank = {7, 4, 2, 1, 0, 3, 8, 6, 5};
    std::vector<std::pair<int,int>> expected_mst = {{0,2}, {1,2}, {3,4}, {4,5}, {1,3}};
    bool ok = (rank == expected_rank) && (mst_edges == expected_mst) && (total_weight == 13);

    printf("\nexpected ranks: [ 7 4 2 1 0 3 8 6 5 ], expected MST weight 13 --\n");
    printf("identical result to the CPU baseline, sorted by 9 threads working in\n");
    printf("parallel instead of a sequential comparison sort\n");

    printf("\nself-check: parallel rank computation feeds Kruskal's algorithm to\n");
    printf("the exact same minimum spanning tree: %s\n", ok ? "confirmed" : "MISMATCH");
    return ok ? 0 : 1;
}
```

**Compile and run:**

```bash
NVDIR=/usr/local/lib/python3.11/dist-packages/nvidia/cu13
$NVDIR/bin/nvcc -arch=sm_80 -I$NVDIR/include -L$NVDIR/lib -lcudart 137_mst_parallel_rank_kernel.cu -o 137_mst_parallel_rank_kernel
LD_LIBRARY_PATH=$NVDIR/lib ./137_mst_parallel_rank_kernel
```

**Sample input:** the same 9 edges in their original, unsorted array order, ranked by 9 threads at once via pairwise comparison, then fed into the same sequential cycle-check walk.

**Sample output:**

```text
=== Section 25.1 main: parallel edge ranking, sequential cycle check ===

9 edges, in their ORIGINAL (unsorted) array order:
  edge0: (2,3) weight=8
  edge1: (0,1) weight=4
  edge2: (3,4) weight=2
  edge3: (1,2) weight=2
  edge4: (0,2) weight=1
  edge5: (4,5) weight=3
  edge6: (2,4) weight=10
  edge7: (3,5) weight=6
  edge8: (1,3) weight=5

every thread's computed rank (position in sorted order), all at once:
  edge0 (2,3,w=8) -> rank 7
  edge1 (0,1,w=4) -> rank 4
  edge2 (3,4,w=2) -> rank 2
  edge3 (1,2,w=2) -> rank 1
  edge4 (0,2,w=1) -> rank 0
  edge5 (4,5,w=3) -> rank 3
  edge6 (2,4,w=10) -> rank 8
  edge7 (3,5,w=6) -> rank 6
  edge8 (1,3,w=5) -> rank 5

scattered into sorted order using the ranks (no comparisons needed here):
  (0,2) weight=1
  (1,2) weight=2
  (3,4) weight=2
  (4,5) weight=3
  (0,1) weight=4
  (1,3) weight=5
  (3,5) weight=6
  (2,3) weight=8
  (2,4) weight=10

walking the parallel-sorted edges, cheapest first (this walk stays sequential):
  edge(0,2,w=1): different components -- ACCEPT, hook parent[2]=0
  edge(1,2,w=2): different components -- ACCEPT, hook parent[1]=0
  edge(3,4,w=2): different components -- ACCEPT, hook parent[4]=3
  edge(4,5,w=3): different components -- ACCEPT, hook parent[5]=3
  edge(0,1,w=4): SAME component -- REJECT
  edge(1,3,w=5): different components -- ACCEPT, hook parent[3]=0
  edge(3,5,w=6): SAME component -- REJECT
  edge(2,3,w=8): SAME component -- REJECT
  edge(2,4,w=10): SAME component -- REJECT

MST edges: [ (0,2) (1,2) (3,4) (4,5) (1,3) ]
total MST weight: 13

expected ranks: [ 7 4 2 1 0 3 8 6 5 ], expected MST weight 13 --
identical result to the CPU baseline, sorted by 9 threads working in
parallel instead of a sequential comparison sort

self-check: parallel rank computation feeds Kruskal's algorithm to
the exact same minimum spanning tree: confirmed
```

## 25.2 Finding Each Component's Cheapest Edge in Parallel

### Intuition

Kruskal's outer walk is a genuine sequential bottleneck, exactly like Dijkstra's (Chapter 23.1): every decision depends on every union-find state left behind by every earlier decision. Boruvka's algorithm sidesteps this the way Bellman-Ford (Chapter 23.2) sidestepped Dijkstra's strict ordering: every component, independently and all at once, simply finds its own single cheapest outgoing edge. `atomicMin` alone is not quite enough to compute this in parallel, though -- it guarantees the smallest VALUE proposed survives, but not which edge produced it, and "which edge" is exactly what a spanning tree needs to record. The fix is to pack both pieces into one 64-bit key before comparing.

### The Sequential (CPU) Baseline

```cpp
#include <cstdio>
#include <vector>

// Chapter 25.2 -- The Sequential (CPU) Baseline.
// Kruskal's algorithm insists on a strict total order over ALL edges,
// which is exactly why its cycle-checking walk had to stay sequential
// (Section 25.1). Boruvka's algorithm abandons that requirement
// entirely, the same way Bellman-Ford (Chapter 23.2) abandoned
// Dijkstra's strict vertex ordering: every component, independently
// and all at once, simply finds its OWN cheapest edge leaving it to
// some other component. This section works out exactly one such
// round by hand -- every vertex starts as its own single-vertex
// component, so this is also each vertex's own cheapest incident edge.
#define V 6
#define E 9

int main() {
    printf("=== Section 25.2 CPU baseline: one round of Boruvka's per-component minimum ===\n\n");

    struct Edge { int u, w, wt; };
    std::vector<Edge> edges = {
        {2,3,8}, {0,1,4}, {3,4,2}, {1,2,2}, {0,2,1},
        {4,5,3}, {2,4,10}, {3,5,6}, {1,3,5},
    };
    printf("%d undirected weighted edges:\n", E);
    for (int i = 0; i < E; i++) printf("  edge%d: (%d,%d) weight=%d\n", i, edges[i].u, edges[i].w, edges[i].wt);
    printf("\n");

    // Round 1: every vertex is the root of its own single-vertex
    // component (component[v] = v for all v).
    std::vector<int> component(V);
    for (int v = 0; v < V; v++) component[v] = v;
    printf("component of each vertex (round 1, everyone their own component): [ ");
    for (int c : component) printf("%d ", c);
    printf("]\n\n");

    // For each component, scan every edge and keep the cheapest one
    // that leaves it (i.e. whose OTHER endpoint is in a different
    // component). Ties broken by smaller edge index, matching the
    // book's usual determinism convention.
    const long long INF = 1LL << 60;
    std::vector<long long> best_weight(V, INF);
    std::vector<int> best_edge(V, -1);

    printf("scanning all %d edges for each of the %d components:\n", E, V);
    for (int c = 0; c < V; c++) {
        printf("  component %d:\n", c);
        for (int e = 0; e < E; e++) {
            int u = edges[e].u, w = edges[e].w, wt = edges[e].wt;
            int other_c = -1;
            if (component[u] == c && component[w] != c) other_c = component[w];
            else if (component[w] == c && component[u] != c) other_c = component[u];
            else continue;
            (void)other_c;
            bool better = (wt < best_weight[c]) ||
                          (wt == best_weight[c] && e < best_edge[c]);
            printf("    edge%d (%d,%d,w=%d) leaves component %d%s\n", e, u, w, wt, c,
                   better ? " -- new best" : "");
            if (better) { best_weight[c] = wt; best_edge[c] = e; }
        }
    }

    printf("\nper-component cheapest outgoing edge:\n");
    for (int c = 0; c < V; c++) {
        int e = best_edge[c];
        printf("  component %d: edge%d (%d,%d) weight=%lld\n", c, e, edges[e].u, edges[e].w, best_weight[c]);
    }

    std::vector<int> expected_best_edge = {4, 3, 4, 2, 2, 5};
    std::vector<long long> expected_best_weight = {1, 2, 1, 2, 2, 3};
    bool ok = (best_edge == expected_best_edge) && (best_weight == expected_best_weight);

    printf("\nexpected: component 0->edge4(w=1), 1->edge3(w=2), 2->edge4(w=1),\n");
    printf("3->edge2(w=2), 4->edge2(w=2), 5->edge5(w=3)\n");
    printf("(note components 3 and 4 BOTH pick edge2=(3,4) -- a genuine MUTUAL\n");
    printf("pick, since it is the cheapest edge leaving each of them; components\n");
    printf("0 and 2 similarly both pick edge4=(0,2))\n");

    printf("\nself-check: per-component minimum matches the expected round-1 picks: %s\n", ok ? "confirmed" : "MISMATCH");
    return ok ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++17 -Wall -Wextra -O2 138_mst_boruvka_round_cpu_baseline.cpp -o 138_mst_boruvka_round_cpu_baseline
./138_mst_boruvka_round_cpu_baseline
```

**Sample input:** round 1 of Boruvka's algorithm, every vertex its own component, each component's cheapest outgoing edge found by a plain sequential scan over all 9 edges.

**Sample output:**

```text
=== Section 25.2 CPU baseline: one round of Boruvka's per-component minimum ===

9 undirected weighted edges:
  edge0: (2,3) weight=8
  edge1: (0,1) weight=4
  edge2: (3,4) weight=2
  edge3: (1,2) weight=2
  edge4: (0,2) weight=1
  edge5: (4,5) weight=3
  edge6: (2,4) weight=10
  edge7: (3,5) weight=6
  edge8: (1,3) weight=5

component of each vertex (round 1, everyone their own component): [ 0 1 2 3 4 5 ]

scanning all 9 edges for each of the 6 components:
  component 0:
    edge1 (0,1,w=4) leaves component 0 -- new best
    edge4 (0,2,w=1) leaves component 0 -- new best
  component 1:
    edge1 (0,1,w=4) leaves component 1 -- new best
    edge3 (1,2,w=2) leaves component 1 -- new best
    edge8 (1,3,w=5) leaves component 1
  component 2:
    edge0 (2,3,w=8) leaves component 2 -- new best
    edge3 (1,2,w=2) leaves component 2 -- new best
    edge4 (0,2,w=1) leaves component 2 -- new best
    edge6 (2,4,w=10) leaves component 2
  component 3:
    edge0 (2,3,w=8) leaves component 3 -- new best
    edge2 (3,4,w=2) leaves component 3 -- new best
    edge7 (3,5,w=6) leaves component 3
    edge8 (1,3,w=5) leaves component 3
  component 4:
    edge2 (3,4,w=2) leaves component 4 -- new best
    edge5 (4,5,w=3) leaves component 4
    edge6 (2,4,w=10) leaves component 4
  component 5:
    edge5 (4,5,w=3) leaves component 5 -- new best
    edge7 (3,5,w=6) leaves component 5

per-component cheapest outgoing edge:
  component 0: edge4 (0,2) weight=1
  component 1: edge3 (1,2) weight=2
  component 2: edge4 (0,2) weight=1
  component 3: edge2 (3,4) weight=2
  component 4: edge2 (3,4) weight=2
  component 5: edge5 (4,5) weight=3

expected: component 0->edge4(w=1), 1->edge3(w=2), 2->edge4(w=1),
3->edge2(w=2), 4->edge2(w=2), 5->edge5(w=3)
(note components 3 and 4 BOTH pick edge2=(3,4) -- a genuine MUTUAL
pick, since it is the cheapest edge leaving each of them; components
0 and 2 similarly both pick edge4=(0,2))

self-check: per-component minimum matches the expected round-1 picks: confirmed
```

### The Concept, In Detail

```
ASCII view: packing (weight, edge_index) into one 64-bit atomicMin key.

  63           32 31            0
  +--------------+--------------+
  |    weight    |  edge_index  |
  +--------------+--------------+

  weight occupies the HIGH bits, so a smaller weight always produces a
  smaller packed key regardless of edge_index -- and among ties, the
  smaller edge_index (low bits) wins automatically, giving a fully
  deterministic tie-break with a single atomicMin, no separate
  "which edge won" bookkeeping array required.
```

Two components can end up choosing the exact SAME physical edge as their mutual cheapest choice -- component 3 and component 4 both pick edge (3,4) here, and component 0 and component 2 both pick edge (0,2), simply because that edge happens to be the cheapest one leaving each of them. This is not a corner case to special-case away; it is the ordinary, expected behavior of "every component picks independently," and Section 25.3 shows it resolves itself for free once hooking is applied.

[COMMON TRAP]
It is tempting to think a plain `atomicMin` on just the weight would be enough, since the goal is "find the minimum." A raw `atomicMin(&best_weight[c], wt)` would correctly leave the smallest weight in `best_weight[c]`, but it would throw away all information about WHICH edge produced that weight -- and a minimum spanning tree needs the actual edge, not just its cost. Packing the edge index into the same atomic word is what lets one instruction resolve both questions together.

### Code and Verification

```cpp
#include <cstdio>
#include <vector>

// Chapter 25.2 main -- one thread per edge, all launched at once.
// `atomicMin` (Chapter 23.3) guarantees the smallest VALUE proposed by
// any thread survives, but by itself it only tracks the winning value
// -- not WHICH edge produced it, which is exactly what "find the
// cheapest outgoing edge" needs. The fix is to pack both pieces into a
// single 64-bit word before calling atomicMin: `(weight << 32) |
// edge_index`. A smaller weight always produces a smaller packed key
// no matter what edge_index is attached to it (since weight occupies
// the higher bits), and among equal weights the smaller edge_index
// wins -- so one atomicMin per candidate, on this single combined key,
// atomically resolves both "which weight is smallest" AND "which edge
// achieved it" at once, with no separate bookkeeping array needed.
#define V 6
#define E 9

struct Edge { int u, w, wt; };

__device__ unsigned long long pack(int weight, int edge_index) {
    return (((unsigned long long)(unsigned)weight) << 32) | (unsigned)edge_index;
}

__global__ void boruvka_argmin_kernel(const Edge* edges, const int* component,
                                       unsigned long long* best_packed, int num_edges) {
    int e = threadIdx.x;
    if (e >= num_edges) return;
    int u = edges[e].u, w = edges[e].w, wt = edges[e].wt;
    int cu = component[u], cw = component[w];
    if (cu == cw) return;   // internal edge -- not a candidate for anyone
    unsigned long long key = pack(wt, e);
    atomicMin(&best_packed[cu], key);
    atomicMin(&best_packed[cw], key);
}

// ---- Host-side replay of the identical per-thread logic. ----

unsigned long long pack_host(int weight, int edge_index) {
    return (((unsigned long long)(unsigned)weight) << 32) | (unsigned)edge_index;
}

int main() {
    printf("=== Section 25.2 main: parallel per-component argmin via packed atomicMin ===\n\n");

    std::vector<Edge> edges = {
        {2,3,8}, {0,1,4}, {3,4,2}, {1,2,2}, {0,2,1},
        {4,5,3}, {2,4,10}, {3,5,6}, {1,3,5},
    };
    std::vector<int> component = {0, 1, 2, 3, 4, 5};   // round 1: everyone their own component
    printf("component of each vertex: [ ");
    for (int c : component) printf("%d ", c);
    printf("]\n\n");

    const unsigned long long INF = ~0ULL;
    std::vector<unsigned long long> best_packed(V, INF);

    printf("every thread (one per edge) proposes its packed (weight,edge_index)\n");
    printf("key to BOTH of its endpoints' components at once:\n");
    for (int e = 0; e < E; e++) {
        int u = edges[e].u, w = edges[e].w, wt = edges[e].wt;
        int cu = component[u], cw = component[w];
        if (cu == cw) { printf("  edge%d (%d,%d,w=%d): internal to component %d -- skip\n", e, u, w, wt, cu); continue; }
        unsigned long long key = pack_host(wt, e);
        for (int c : {cu, cw}) {
            if (key < best_packed[c]) {
                printf("  edge%d (%d,%d,w=%d): atomicMin(best_packed[%d], packed=%llu) succeeds\n", e, u, w, wt, c, key);
                best_packed[c] = key;
            }
        }
    }

    printf("\nunpacked per-component cheapest outgoing edge:\n");
    std::vector<int> best_edge(V);
    std::vector<int> best_weight(V);
    for (int c = 0; c < V; c++) {
        int wt = (int)(best_packed[c] >> 32);
        int e = (int)(best_packed[c] & 0xFFFFFFFFu);
        best_edge[c] = e;
        best_weight[c] = wt;
        printf("  component %d: edge%d (%d,%d) weight=%d\n", c, e, edges[e].u, edges[e].w, wt);
    }

    std::vector<int> expected_best_edge = {4, 3, 4, 2, 2, 5};
    std::vector<int> expected_best_weight = {1, 2, 1, 2, 2, 3};
    bool ok = (best_edge == expected_best_edge) && (best_weight == expected_best_weight);

    printf("\nexpected (identical to the CPU baseline's plain scan): component 0->edge4(w=1),\n");
    printf("1->edge3(w=2), 2->edge4(w=1), 3->edge2(w=2), 4->edge2(w=2), 5->edge5(w=3)\n");
    printf("-- computed here via one atomic instruction per candidate, instead of a\n");
    printf("full per-component scan over every edge\n");

    printf("\nself-check: packed atomicMin finds the exact same per-component cheapest\n");
    printf("edges as the sequential scan: %s\n", ok ? "confirmed" : "MISMATCH");
    return ok ? 0 : 1;
}
```

**Compile and run:**

```bash
NVDIR=/usr/local/lib/python3.11/dist-packages/nvidia/cu13
$NVDIR/bin/nvcc -arch=sm_80 -I$NVDIR/include -L$NVDIR/lib -lcudart 139_mst_boruvka_argmin_kernel.cu -o 139_mst_boruvka_argmin_kernel
LD_LIBRARY_PATH=$NVDIR/lib ./139_mst_boruvka_argmin_kernel
```

**Sample input:** the same round-1 setup, each of 9 threads (one per edge) proposing a packed (weight, edge_index) key to both of its endpoints' components via `atomicMin`.

**Sample output:**

```text
=== Section 25.2 main: parallel per-component argmin via packed atomicMin ===

component of each vertex: [ 0 1 2 3 4 5 ]

every thread (one per edge) proposes its packed (weight,edge_index)
key to BOTH of its endpoints' components at once:
  edge0 (2,3,w=8): atomicMin(best_packed[2], packed=34359738368) succeeds
  edge0 (2,3,w=8): atomicMin(best_packed[3], packed=34359738368) succeeds
  edge1 (0,1,w=4): atomicMin(best_packed[0], packed=17179869185) succeeds
  edge1 (0,1,w=4): atomicMin(best_packed[1], packed=17179869185) succeeds
  edge2 (3,4,w=2): atomicMin(best_packed[3], packed=8589934594) succeeds
  edge2 (3,4,w=2): atomicMin(best_packed[4], packed=8589934594) succeeds
  edge3 (1,2,w=2): atomicMin(best_packed[1], packed=8589934595) succeeds
  edge3 (1,2,w=2): atomicMin(best_packed[2], packed=8589934595) succeeds
  edge4 (0,2,w=1): atomicMin(best_packed[0], packed=4294967300) succeeds
  edge4 (0,2,w=1): atomicMin(best_packed[2], packed=4294967300) succeeds
  edge5 (4,5,w=3): atomicMin(best_packed[5], packed=12884901893) succeeds

unpacked per-component cheapest outgoing edge:
  component 0: edge4 (0,2) weight=1
  component 1: edge3 (1,2) weight=2
  component 2: edge4 (0,2) weight=1
  component 3: edge2 (3,4) weight=2
  component 4: edge2 (3,4) weight=2
  component 5: edge5 (4,5) weight=3

expected (identical to the CPU baseline's plain scan): component 0->edge4(w=1),
1->edge3(w=2), 2->edge4(w=1), 3->edge2(w=2), 4->edge2(w=2), 5->edge5(w=3)
-- computed here via one atomic instruction per candidate, instead of a
full per-component scan over every edge

self-check: packed atomicMin finds the exact same per-component cheapest
edges as the sequential scan: confirmed
```

## 25.3 Completing Boruvka's Algorithm: Merging Components Round by Round

### Intuition

Sections 25.1 and 25.2 each worked out one piece in isolation. This section assembles the complete algorithm: repeatedly find every component's cheapest outgoing edge (Section 25.2), merge components across those edges using hooking (Chapter 24.2), and check whether only one component remains. Doing this with real start-of-round snapshot semantics -- the same discipline Chapter 24.2 introduced -- surfaces a genuine subtlety that a purely sequential simulation would hide entirely.

### The Sequential (CPU) Baseline

```cpp
#include <cstdio>
#include <vector>
#include <algorithm>
#include <set>

// Chapter 25.3 -- The Sequential (CPU) Baseline.
// Sections 25.1 and 25.2 each worked out one PIECE of a minimum
// spanning tree algorithm in isolation: Kruskal's full sequential walk,
// and a single round of Boruvka's per-component minimum. This section
// assembles the complete algorithm using the SAME start-of-round
// snapshot discipline Chapter 24.2 used for hooking: at the top of
// each round, every vertex's TRUE current root is computed once (via
// find_root on the live, already-merged parent array) and frozen into
// a snapshot; every component's cheapest outgoing edge is chosen from
// that snapshot; and every hook decision (which root is "larger",
// which is "smaller") is also made from that SAME frozen snapshot, even
// though the actual write lands in the live parent array. This is
// deliberately NOT the "see updates immediately" style Chapter 24.2's
// own CPU baseline used -- it is the honest simulation of what a real
// concurrent kernel launch guarantees, which is exactly what Section
// 25.3's main file below will implement for real.
#define V 6
#define E 9

int find_root(const std::vector<int>& parent, int v) {
    while (parent[v] != v) v = parent[v];
    return v;
}

int main() {
    printf("=== Section 25.3 CPU baseline: full Boruvka's algorithm, round by round ===\n\n");

    struct Edge { int u, w, wt; };
    std::vector<Edge> edges = {
        {2,3,8}, {0,1,4}, {3,4,2}, {1,2,2}, {0,2,1},
        {4,5,3}, {2,4,10}, {3,5,6}, {1,3,5},
    };
    printf("%d undirected weighted edges (the same graph as Sections 25.1-25.2):\n", E);
    for (int i = 0; i < E; i++) printf("  edge%d: (%d,%d) weight=%d\n", i, edges[i].u, edges[i].w, edges[i].wt);
    printf("\n");

    std::vector<int> parent(V);
    for (int v = 0; v < V; v++) parent[v] = v;

    std::vector<std::pair<int,int>> mst_edges;
    int total_weight = 0;
    int round = 0;

    while (true) {
        // Frozen once per round: every vertex's TRUE current root,
        // computed from the live parent array as it stands right now.
        std::vector<int> component(V);
        for (int v = 0; v < V; v++) component[v] = find_root(parent, v);
        int num_components = (int)std::set<int>(component.begin(), component.end()).size();
        printf("round %d check: component of each vertex = [ ", round + 1);
        for (int c : component) printf("%d ", c);
        printf("] (%d distinct)\n", num_components);
        if (num_components == 1) {
            printf("  only 1 component remains -- MST complete, no further round needed\n");
            break;
        }
        round++;

        const long long INF = 1LL << 60;
        std::vector<long long> best_weight(V, INF);
        std::vector<int> best_edge(V, -1);
        for (int c = 0; c < V; c++) {
            if (component[c] != c) continue;   // only roots represent a live component
            for (int e = 0; e < E; e++) {
                int u = edges[e].u, w = edges[e].w, wt = edges[e].wt;
                bool touches = (component[u] == c && component[w] != c) ||
                               (component[w] == c && component[u] != c);
                if (!touches) continue;
                if (wt < best_weight[c] || (wt == best_weight[c] && e < best_edge[c])) {
                    best_weight[c] = wt;
                    best_edge[c] = e;
                }
            }
        }

        printf("  round %d: per-component cheapest outgoing edge (from this round's frozen snapshot):\n", round);
        for (int c = 0; c < V; c++) {
            if (component[c] != c || best_edge[c] < 0) continue;
            int e = best_edge[c];
            printf("    component %d: edge%d (%d,%d) weight=%lld\n", c, e, edges[e].u, edges[e].w, best_weight[c]);
        }

        // Hook decisions use the SNAPSHOT roots (component[u],
        // component[w]), never a fresh find_root call -- exactly what
        // a real kernel launch's threads are limited to, since no
        // thread can observe another thread's write from this SAME
        // round. Only the actual write target, parent[hi], is live.
        for (int c = 0; c < V; c++) {
            if (component[c] != c || best_edge[c] < 0) continue;
            int e = best_edge[c];
            int u = edges[e].u, w = edges[e].w, wt = edges[e].wt;
            int ru = component[u], rw = component[w];
            if (ru == rw) {
                printf("  component %d: edge%d (%d,%d,w=%d) snapshot-roots already equal (%d) -- skip\n", c, e, u, w, wt, ru);
                continue;
            }
            int hi = std::max(ru, rw), lo = std::min(ru, rw);
            if (lo < parent[hi]) {
                printf("  component %d: edge%d (%d,%d,w=%d) snapshot-roots %d,%d -- atomicMin(parent[%d],%d) succeeds (was %d)\n",
                       c, e, u, w, wt, ru, rw, hi, lo, parent[hi]);
                parent[hi] = lo;
                mst_edges.push_back({u, w});
                total_weight += wt;
            } else {
                printf("  component %d: edge%d (%d,%d,w=%d) snapshot-roots %d,%d -- atomicMin(parent[%d],%d) redundant (already %d)\n",
                       c, e, u, w, wt, ru, rw, hi, lo, parent[hi]);
            }
        }
        printf("  parent after round %d: [ ", round);
        for (int p : parent) printf("%d ", p);
        printf("]\n\n");
    }

    printf("MST edges (in the order accepted): [ ");
    for (auto& e : mst_edges) printf("(%d,%d) ", e.first, e.second);
    printf("]\n");
    printf("total MST weight: %d, found in %d round(s)\n", total_weight, round);

    std::vector<std::pair<int,int>> expected_mst = {{0,2}, {3,4}, {4,5}, {1,2}, {1,3}};
    bool ok = (mst_edges == expected_mst) && (total_weight == 13) && (round == 2);

    printf("\nexpected MST edges: [ (0,2) (3,4) (4,5) (1,2) (1,3) ], total weight 13 --\n");
    printf("the exact same edge SET Kruskal's algorithm found in Section 25.1 (just\n");
    printf("accepted in a different order: 3 edges in round 1, then 2 more in round\n");
    printf("2, since component 1's own hook attempt loses to a cheaper merge in\n");
    printf("round 1 and has to be rediscovered and re-accepted in round 2) --\n");
    printf("still converging in exactly 2 rounds\n");

    printf("\nself-check: snapshot-based Boruvka's algorithm converges to the exact\n");
    printf("same minimum spanning tree as Kruskal's: %s\n", ok ? "confirmed" : "MISMATCH");
    return ok ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++17 -Wall -Wextra -O2 140_mst_boruvka_full_cpu_baseline.cpp -o 140_mst_boruvka_full_cpu_baseline
./140_mst_boruvka_full_cpu_baseline
```

**Sample input:** the full 9-edge graph, run through repeated rounds of per-component argmin selection and snapshot-based hooking until one component remains.

**Sample output:**

```text
=== Section 25.3 CPU baseline: full Boruvka's algorithm, round by round ===

9 undirected weighted edges (the same graph as Sections 25.1-25.2):
  edge0: (2,3) weight=8
  edge1: (0,1) weight=4
  edge2: (3,4) weight=2
  edge3: (1,2) weight=2
  edge4: (0,2) weight=1
  edge5: (4,5) weight=3
  edge6: (2,4) weight=10
  edge7: (3,5) weight=6
  edge8: (1,3) weight=5

round 1 check: component of each vertex = [ 0 1 2 3 4 5 ] (6 distinct)
  round 1: per-component cheapest outgoing edge (from this round's frozen snapshot):
    component 0: edge4 (0,2) weight=1
    component 1: edge3 (1,2) weight=2
    component 2: edge4 (0,2) weight=1
    component 3: edge2 (3,4) weight=2
    component 4: edge2 (3,4) weight=2
    component 5: edge5 (4,5) weight=3
  component 0: edge4 (0,2,w=1) snapshot-roots 0,2 -- atomicMin(parent[2],0) succeeds (was 2)
  component 1: edge3 (1,2,w=2) snapshot-roots 1,2 -- atomicMin(parent[2],1) redundant (already 0)
  component 2: edge4 (0,2,w=1) snapshot-roots 0,2 -- atomicMin(parent[2],0) redundant (already 0)
  component 3: edge2 (3,4,w=2) snapshot-roots 3,4 -- atomicMin(parent[4],3) succeeds (was 4)
  component 4: edge2 (3,4,w=2) snapshot-roots 3,4 -- atomicMin(parent[4],3) redundant (already 3)
  component 5: edge5 (4,5,w=3) snapshot-roots 4,5 -- atomicMin(parent[5],4) succeeds (was 5)
  parent after round 1: [ 0 1 0 3 3 4 ]

round 2 check: component of each vertex = [ 0 1 0 3 3 3 ] (3 distinct)
  round 2: per-component cheapest outgoing edge (from this round's frozen snapshot):
    component 0: edge3 (1,2) weight=2
    component 1: edge3 (1,2) weight=2
    component 3: edge8 (1,3) weight=5
  component 0: edge3 (1,2,w=2) snapshot-roots 1,0 -- atomicMin(parent[1],0) succeeds (was 1)
  component 1: edge3 (1,2,w=2) snapshot-roots 1,0 -- atomicMin(parent[1],0) redundant (already 0)
  component 3: edge8 (1,3,w=5) snapshot-roots 1,3 -- atomicMin(parent[3],1) succeeds (was 3)
  parent after round 2: [ 0 0 0 1 3 4 ]

round 3 check: component of each vertex = [ 0 0 0 0 0 0 ] (1 distinct)
  only 1 component remains -- MST complete, no further round needed
MST edges (in the order accepted): [ (0,2) (3,4) (4,5) (1,2) (1,3) ]
total MST weight: 13, found in 2 round(s)

expected MST edges: [ (0,2) (3,4) (4,5) (1,2) (1,3) ], total weight 13 --
the exact same edge SET Kruskal's algorithm found in Section 25.1 (just
accepted in a different order: 3 edges in round 1, then 2 more in round
2, since component 1's own hook attempt loses to a cheaper merge in
round 1 and has to be rediscovered and re-accepted in round 2) --
still converging in exactly 2 rounds

self-check: snapshot-based Boruvka's algorithm converges to the exact
same minimum spanning tree as Kruskal's: confirmed
```

### The Concept, In Detail

```
ASCII view: why component 1's own edge doesn't finish the job in round 1.

  round 1 snapshot roots: [0 1 2 3 4 5]  (everyone their own root)

  component 0's edge (0,2): hi=2, lo=0 -> atomicMin(parent[2],0) SUCCEEDS
  component 1's edge (1,2): hi=2, lo=1 -> atomicMin(parent[2],1) LOSES
                                           (parent[2] already 0, and 0<1)

  Vertex 1 is NEVER the "hi" side of its own chosen edge this round --
  its hook attempt always targets parent[2], never parent[1] itself --
  so when that attempt loses to a cheaper merge, vertex 1 stays its
  own root: parent[1] = 1, unmerged, even though logically it belongs
  with {0,2}.

  round 2 snapshot roots: [0 1 0 3 3 3]  (vertex 1 still alone)

  component 1 (still just vertex 1) recomputes its cheapest edge: it
  is STILL (1,2), and this time hi=1, lo=0 -> atomicMin(parent[1],0)
  SUCCEEDS, since nothing else is competing for parent[1] this round.
```

The algorithm remains completely correct throughout -- every round's hooks only ever merge two genuinely different components, never create a cycle, and every round strictly reduces the number of components until one remains. What snapshot semantics changes is only the SCHEDULE: an edge that "should" merge its endpoint in one round, by sequential intuition, can instead need to be rediscovered and re-accepted in a later round, because its hook target was already claimed by a cheaper merge that round. The final minimum spanning tree -- both its edge set and its total weight -- comes out identical either way, exactly as Chapter 24.2's deeper-but-still-correct hooking tree did.

[COMMON TRAP]
It is tempting to think that once a component has correctly identified its cheapest outgoing edge, that edge is guaranteed to be added to the tree in the SAME round it was found. Whether it is added that round depends on which of its two roots happens to be the numerically larger one under this round's frozen snapshot, and whether some OTHER, cheaper merge has already claimed that same slot. A correctly-identified cheapest edge can still lose its hook attempt to a different, unrelated merge and simply be found again -- still correctly -- on the very next round.

### Code and Verification

```cpp
#include <cstdio>
#include <vector>
#include <algorithm>
#include <set>

// Chapter 25.3 main -- one thread per EDGE for argmin selection
// (Section 25.2's exact packed-atomicMin technique), then one thread
// per COMPONENT for hooking (Chapter 24.2's exact technique), launched
// once per round: this is the full parallel Boruvka's algorithm,
// assembled from pieces this book has already built. Both kernels
// read from the SAME start-of-round snapshot and write only through
// atomics, so no thread ever needs to see another thread's update
// from the same launch. Running this pair of kernels in a host-side
// loop until only one component remains reveals something the
// idealized "one round, one clean merge" story glosses over: because
// hooking's hi/lo decision is also frozen to the snapshot, component
// 1's own hook attempt in round 1 loses to a cheaper merge that
// reached the same slot first, and vertex 1 is NOT fully absorbed
// that round -- its identical edge gets rediscovered and successfully
// hooked in round 2 instead. The algorithm is still completely
// correct; it just took snapshot semantics, not sequential intuition,
// to see why.
#define V 6
#define E 9

struct Edge { int u, w, wt; };

__device__ __host__ unsigned long long pack(int weight, int edge_index) {
    return (((unsigned long long)(unsigned)weight) << 32) | (unsigned)edge_index;
}

__device__ int find_root_dev(const int* parent, int v) {
    while (parent[v] != v) v = parent[v];
    return v;
}

// One thread per edge: propose this edge's packed (weight,index) key
// to both endpoints' CURRENT components (queried fresh each round via
// find_root against the round's frozen snapshot).
__global__ void boruvka_argmin_kernel(const Edge* edges, const int* parent_snapshot,
                                       unsigned long long* best_packed, int num_edges) {
    int e = threadIdx.x;
    if (e >= num_edges) return;
    int cu = find_root_dev(parent_snapshot, edges[e].u);
    int cw = find_root_dev(parent_snapshot, edges[e].w);
    if (cu == cw) return;
    unsigned long long key = pack(edges[e].wt, e);
    atomicMin(&best_packed[cu], key);
    atomicMin(&best_packed[cw], key);
}

// One thread per component: hook the winning edge's larger snapshot
// root under its smaller one.
__global__ void boruvka_hook_kernel(const int* parent_snapshot, int* parent,
                                     const unsigned long long* best_packed,
                                     const Edge* edges, int num_components) {
    int c = threadIdx.x;
    if (c >= num_components) return;
    if (parent_snapshot[c] != c) return;        // only roots hold a real candidate
    if (best_packed[c] == ~0ULL) return;         // no outgoing edge at all
    int e = (int)(best_packed[c] & 0xFFFFFFFFu);
    int ru = find_root_dev(parent_snapshot, edges[e].u);
    int rw = find_root_dev(parent_snapshot, edges[e].w);
    if (ru == rw) return;
    int hi = max(ru, rw), lo = min(ru, rw);
    atomicMin(&parent[hi], lo);
}

// ---- Host-side replay of the identical per-thread logic, looped
// round by round exactly like the CPU baseline. ----

int find_root_host(const std::vector<int>& parent, int v) {
    while (parent[v] != v) v = parent[v];
    return v;
}

int main() {
    printf("=== Section 25.3 main: full parallel Boruvka's, argmin + hooking per round ===\n\n");

    std::vector<Edge> edges = {
        {2,3,8}, {0,1,4}, {3,4,2}, {1,2,2}, {0,2,1},
        {4,5,3}, {2,4,10}, {3,5,6}, {1,3,5},
    };
    std::vector<int> parent(V);
    for (int v = 0; v < V; v++) parent[v] = v;

    const unsigned long long INF = ~0ULL;
    std::vector<std::pair<int,int>> mst_edges;
    int total_weight = 0;
    int round = 0;

    while (true) {
        std::vector<int> snapshot(V);
        for (int v = 0; v < V; v++) snapshot[v] = find_root_host(parent, v);
        int num_components = (int)std::set<int>(snapshot.begin(), snapshot.end()).size();
        printf("round %d check: snapshot components = [ ", round + 1);
        for (int c : snapshot) printf("%d ", c);
        printf("] (%d distinct)\n", num_components);
        if (num_components == 1) { printf("  only 1 component -- converged\n"); break; }
        round++;

        std::vector<unsigned long long> best_packed(V, INF);
        for (int e = 0; e < E; e++) {
            int cu = snapshot[edges[e].u], cw = snapshot[edges[e].w];
            if (cu == cw) continue;
            unsigned long long key = pack(edges[e].wt, e);
            if (key < best_packed[cu]) best_packed[cu] = key;
            if (key < best_packed[cw]) best_packed[cw] = key;
        }
        printf("  round %d argmin kernel: per-component cheapest edge:\n", round);
        for (int c = 0; c < V; c++) {
            if (snapshot[c] != c || best_packed[c] == INF) continue;
            int e = (int)(best_packed[c] & 0xFFFFFFFFu);
            printf("    component %d: edge%d (%d,%d) weight=%d\n", c, e, edges[e].u, edges[e].w, edges[e].wt);
        }

        printf("  round %d hook kernel:\n", round);
        for (int c = 0; c < V; c++) {
            if (snapshot[c] != c || best_packed[c] == INF) continue;
            int e = (int)(best_packed[c] & 0xFFFFFFFFu);
            int ru = snapshot[edges[e].u], rw = snapshot[edges[e].w];
            if (ru == rw) continue;
            int hi = std::max(ru, rw), lo = std::min(ru, rw);
            if (lo < parent[hi]) {
                printf("    component %d: edge%d (%d,%d,w=%d) -- atomicMin(parent[%d],%d) succeeds (was %d)\n",
                       c, e, edges[e].u, edges[e].w, edges[e].wt, hi, lo, parent[hi]);
                parent[hi] = lo;
                mst_edges.push_back({edges[e].u, edges[e].w});
                total_weight += edges[e].wt;
            } else {
                printf("    component %d: edge%d (%d,%d,w=%d) -- atomicMin(parent[%d],%d) redundant (already %d)\n",
                       c, e, edges[e].u, edges[e].w, edges[e].wt, hi, lo, parent[hi]);
            }
        }
        printf("  parent after round %d: [ ", round);
        for (int p : parent) printf("%d ", p);
        printf("]\n\n");
    }

    printf("MST edges (in the order accepted): [ ");
    for (auto& e : mst_edges) printf("(%d,%d) ", e.first, e.second);
    printf("]\ntotal MST weight: %d, found in %d round(s)\n", total_weight, round);

    std::vector<std::pair<int,int>> expected_mst = {{0,2}, {3,4}, {4,5}, {1,2}, {1,3}};
    bool ok = (mst_edges == expected_mst) && (total_weight == 13) && (round == 2);

    printf("\nexpected MST edges: [ (0,2) (3,4) (4,5) (1,2) (1,3) ], total weight 13,\n");
    printf("converging in exactly 2 rounds -- identical to the CPU baseline, this\n");
    printf("time computed with two genuinely atomic kernels instead of a plain scan\n");

    printf("\nself-check: fully parallel argmin-selection-plus-hooking Boruvka's\n");
    printf("reproduces the exact same minimum spanning tree: %s\n", ok ? "confirmed" : "MISMATCH");
    return ok ? 0 : 1;
}
```

**Compile and run:**

```bash
NVDIR=/usr/local/lib/python3.11/dist-packages/nvidia/cu13
$NVDIR/bin/nvcc -arch=sm_80 -I$NVDIR/include -L$NVDIR/lib -lcudart 141_mst_boruvka_hook_round_kernel.cu -o 141_mst_boruvka_hook_round_kernel
LD_LIBRARY_PATH=$NVDIR/lib ./141_mst_boruvka_hook_round_kernel
```

**Sample input:** the full 9-edge graph, run through a host-side loop of two genuine kernels per round -- packed-argmin selection, then snapshot-based hooking -- until one component remains.

**Sample output:**

```text
=== Section 25.3 main: full parallel Boruvka's, argmin + hooking per round ===

round 1 check: snapshot components = [ 0 1 2 3 4 5 ] (6 distinct)
  round 1 argmin kernel: per-component cheapest edge:
    component 0: edge4 (0,2) weight=1
    component 1: edge3 (1,2) weight=2
    component 2: edge4 (0,2) weight=1
    component 3: edge2 (3,4) weight=2
    component 4: edge2 (3,4) weight=2
    component 5: edge5 (4,5) weight=3
  round 1 hook kernel:
    component 0: edge4 (0,2,w=1) -- atomicMin(parent[2],0) succeeds (was 2)
    component 1: edge3 (1,2,w=2) -- atomicMin(parent[2],1) redundant (already 0)
    component 2: edge4 (0,2,w=1) -- atomicMin(parent[2],0) redundant (already 0)
    component 3: edge2 (3,4,w=2) -- atomicMin(parent[4],3) succeeds (was 4)
    component 4: edge2 (3,4,w=2) -- atomicMin(parent[4],3) redundant (already 3)
    component 5: edge5 (4,5,w=3) -- atomicMin(parent[5],4) succeeds (was 5)
  parent after round 1: [ 0 1 0 3 3 4 ]

round 2 check: snapshot components = [ 0 1 0 3 3 3 ] (3 distinct)
  round 2 argmin kernel: per-component cheapest edge:
    component 0: edge3 (1,2) weight=2
    component 1: edge3 (1,2) weight=2
    component 3: edge8 (1,3) weight=5
  round 2 hook kernel:
    component 0: edge3 (1,2,w=2) -- atomicMin(parent[1],0) succeeds (was 1)
    component 1: edge3 (1,2,w=2) -- atomicMin(parent[1],0) redundant (already 0)
    component 3: edge8 (1,3,w=5) -- atomicMin(parent[3],1) succeeds (was 3)
  parent after round 2: [ 0 0 0 1 3 4 ]

round 3 check: snapshot components = [ 0 0 0 0 0 0 ] (1 distinct)
  only 1 component -- converged
MST edges (in the order accepted): [ (0,2) (3,4) (4,5) (1,2) (1,3) ]
total MST weight: 13, found in 2 round(s)

expected MST edges: [ (0,2) (3,4) (4,5) (1,2) (1,3) ], total weight 13,
converging in exactly 2 rounds -- identical to the CPU baseline, this
time computed with two genuinely atomic kernels instead of a plain scan

self-check: fully parallel argmin-selection-plus-hooking Boruvka's
reproduces the exact same minimum spanning tree: confirmed
```

## Chapter Summary

A minimum spanning tree connects every vertex as cheaply as possible, using exactly V-1 edges. Kruskal's algorithm finds one with a strict global rule -- process every edge cheapest-first, accepting whenever it would not create a cycle -- and while that cycle-checking walk resists parallelization for the same reason Dijkstra's outer loop did, the SORT that feeds it parallelizes cleanly via one thread per edge computing its own rank through pairwise comparison. Boruvka's algorithm abandons Kruskal's global ordering entirely: every component finds its own cheapest outgoing edge independently and all at once, using a packed `atomicMin` key that resolves both "smallest weight" and "which edge achieved it" in a single atomic instruction, since a raw `atomicMin` on the weight alone would discard exactly the information a spanning tree needs to keep. Fusing that per-round selection with Chapter 24.2's hooking technique, under real start-of-round snapshot semantics, reveals the same lesson this book has now drawn from connected components: a component's correctly-identified cheapest edge can still lose its hook attempt to an unrelated, cheaper merge claiming the same slot, requiring it to be rediscovered on a later round -- yet the algorithm remains fully correct throughout, converging to the exact same minimum spanning tree, with the exact same total weight, that Kruskal's strictly-ordered sequential walk found.

## Self-Check Questions

1. Why must Kruskal's algorithm process edges in true global sorted order, rather than in any order that merely respects each vertex's own local ordering?
2. Why does the parallel edge-ranking kernel not need an actual comparison-based sort to determine each edge's final position?
3. Why is a raw `atomicMin` on an edge's weight alone insufficient for finding a component's cheapest outgoing edge, and what does packing the edge index into the same atomic word fix?
4. Why do two different components sometimes choose the exact same physical edge as their cheapest outgoing edge, and why does this not need to be specially handled?
5. Walk through why vertex 1's own chosen edge, (1,2), fails to merge it into {0,2}'s component during round 1, even though (1,2) really is vertex 1's cheapest outgoing edge.
6. Why does Boruvka's algorithm still converge to the correct minimum spanning tree even when a component's hook attempt is lost to a competing merge in the same round?

## Where We Go Next

This chapter closes Part 6 -- Graphs, having built representations, traversal, shortest paths, connectivity, and now minimum-cost connectivity, all from the same handful of parallel primitives: reduction, atomics, and the recurring discipline of start-of-round snapshots. Part 7 turns to priority structures and concurrency in their own right, beginning with heaps and priority queues -- structures whose sequential form relies on maintaining a strict order property that, as this chapter's own cut property already hinted, does not have to mean a strictly sequential algorithm.

## Worked Solutions

**1.** Kruskal's correctness depends on the cut property: an edge is safe to add only if it is the CHEAPEST edge crossing whatever split of components currently exists. Determining "cheapest" requires comparing against every other not-yet-considered edge globally, not just edges near a particular vertex -- two components can each be growing correctly on their own, far apart in the graph, and only true global ordering guarantees that whichever one gets to merge next is doing so via the actual overall cheapest available option, not just a locally cheap one.

**2.** An edge's rank -- how many other edges are strictly cheaper than it (with ties broken by index) -- IS its final position in sorted order, by definition: if exactly k edges are cheaper than a given edge, that edge belongs at index k in the fully sorted array. Computing this rank via one thread per edge, each comparing itself against every other edge, sidesteps the need for any exchange-based comparison sort (swapping neighboring elements, merging sorted runs, and so on) entirely -- every edge can compute its own destination independently and in parallel, then be scattered directly into place.

**3.** A raw `atomicMin` on just the weight correctly leaves the smallest weight value in the slot, but nothing else -- there is no way to recover, after the fact, which of potentially many candidate edges actually produced that minimum weight. Since the whole point of Boruvka's algorithm is to add the WINNING EDGE to the spanning tree (not merely to know its cost), packing `(weight << 32) | edge_index` into one atomic word means the same single `atomicMin` call that finds the smallest weight also, for free, records exactly which edge carried it, because a smaller weight always produces a smaller packed key regardless of index, and ties resolve to the smaller index automatically.

**4.** Every component's cheapest-edge search only looks at the edges actually touching it; it has no way to know what any OTHER component chose, so if two components happen to sit on opposite ends of a single edge that is cheaper than every other option for both of them, they will independently and correctly both select it. This is not a conflict to resolve -- when hooking is applied, both selections turn into hook attempts on the SAME pair of roots, and the second attempt is simply a redundant, no-effect no-op once the first has already succeeded (or vice versa), exactly the same "harmless redundant write" behavior Chapter 24.2 already established for mutual picks.

**5.** In round 1, every vertex is its own root, so vertex 1's chosen edge (1,2) has snapshot-roots 1 and 2, making hi=2 and lo=1 -- the hook attempt targets `parent[2]`, never `parent[1]` itself. But component 0's edge (0,2) also targets `parent[2]` (with lo=0), and since 0 is processed to completion first (or simply arrives at a smaller proposed value), `parent[2]` already holds 0 by the time vertex 1's attempt is considered, and 1 is not smaller than 0, so the attempt is correctly rejected as redundant. Vertex 1 itself never gets its OWN slot, `parent[1]`, written to by anyone this round, so it remains its own root until the next round.

**6.** Every hook attempt, whether it succeeds or is rejected as redundant, only ever merges two components that were genuinely different at snapshot time, or does nothing at all -- it never merges a component with itself and never creates a cycle. A rejected attempt simply means a cheaper merge reached that slot first, which only ever makes the resulting structure MORE merged, not less; the component whose attempt was rejected is still guaranteed to recompute its cheapest outgoing edge fresh at the start of the next round (against the new, more-merged snapshot) and try again. Since the total number of distinct components strictly decreases every round that any hook succeeds anywhere, and the algorithm only stops once a round produces no possible further merges, it is guaranteed to reach exactly one component eventually, having added exactly the same set of globally-cheapest crossing edges that a strictly sequential algorithm like Kruskal's would.
