# Chapter 24: Connected Components

Every algorithm since Chapter 22 has answered a question rooted at a single source vertex -- how many hops, or at what total cost, to reach every other vertex from it. Connected components asks something different: with no source vertex singled out at all, which vertices can reach which other vertices, full stop. This chapter builds that grouping three ways: flood fill, the direct sequential generalization of Chapter 22's own BFS, run repeatedly until every vertex has a label; label propagation, which spreads each component's smallest vertex id outward via `atomicMin` until every vertex agrees; and hooking, a union-find-style approach whose PARALLEL version -- unlike its sequential twin -- produces a genuinely deeper tree, setting up the pointer-jumping technique this chapter closes with to flatten it back down.

## 24.1 Sequential Connected Components via Flood Fill

### Intuition

Chapter 22's BFS started from one designated source and asked how far every other vertex was from it. Connected components drops the notion of a "source" altogether: edges are treated as UNDIRECTED (an edge lets traversal move either way), and the question is simply which vertices can reach which others through any sequence of edges at all. The classic sequential approach reuses BFS exactly as-is, just repeatedly: pick any not-yet-labeled vertex, flood-fill outward labeling everything it reaches with that vertex's own id, then move on to the next not-yet-labeled vertex and repeat until none remain. This section's graph extends the book's own 6-vertex graph with a second, separate 2-vertex component, giving flood fill something Chapters 22-23 never needed: more than one group to discover.

### The Sequential (CPU) Baseline

```cpp
#include <cstdio>
#include <vector>

// Chapter 24.1 -- The Sequential (CPU) Baseline.
// Connected components asks a fundamentally different question than
// Chapters 22-23: not "how far" or "how cheap" from a single source,
// but simply "which vertices can reach which other vertices at all,"
// with no source vertex singled out. Edges are treated as UNDIRECTED
// for this question (an edge lets traversal move either way), and the
// classic sequential approach is exactly Chapter 22's own flood-fill
// BFS, just run repeatedly: pick any not-yet-labeled vertex, flood-fill
// outward labeling everything it reaches with that vertex's own id,
// then move to the next not-yet-labeled vertex and repeat. This
// section's graph adds a second, separate 2-vertex component to
// Chapter 21-23's own 6-vertex graph, specifically to give connected
// components something Chapters 22-23 never needed: more than one
// group to discover.
#define V 8
#define UNVISITED (-1)

int main() {
    printf("=== Section 24.1 CPU baseline: connected components via flood fill ===\n\n");

    // The book's own 6-vertex graph (edges treated as undirected),
    // plus a new, separate 2-vertex component {6, 7}.
    std::vector<std::pair<int,int>> edges = {
        {2,3}, {0,1}, {3,4}, {1,2}, {0,2}, {4,5}, {2,4}, {3,5}, {1,3}, {6,7},
    };
    std::vector<std::vector<int>> adj(V);
    for (auto& e : edges) {
        adj[e.first].push_back(e.second);
        adj[e.second].push_back(e.first);
    }
    printf("undirected adjacency:\n");
    for (int v = 0; v < V; v++) {
        printf("  vertex %d: ", v);
        for (int w : adj[v]) printf("%d ", w);
        printf("\n");
    }
    printf("\n");

    std::vector<int> label(V, UNVISITED);
    for (int start = 0; start < V; start++) {
        if (label[start] != UNVISITED) continue;
        printf("flood-filling from vertex %d (its own id becomes the component label):\n", start);
        label[start] = start;
        std::vector<int> queue = {start};
        int head = 0;
        while (head < (int)queue.size()) {
            int u = queue[head++];
            for (int w : adj[u]) {
                if (label[w] == UNVISITED) {
                    label[w] = start;
                    queue.push_back(w);
                    printf("  vertex %d labeled %d (reached from %d)\n", w, start, u);
                }
            }
        }
        printf("\n");
    }

    printf("final labels: [ ");
    for (int l : label) printf("%d ", l);
    printf("]\n");

    std::vector<int> expected_label = {0, 0, 0, 0, 0, 0, 6, 6};
    bool ok = (label == expected_label);

    printf("\nexpected labels: [ 0 0 0 0 0 0 6 6 ] (2 components: {0..5} and {6,7},\n");
    printf("each labeled by its own smallest/starting vertex id)\n");
    printf("\nself-check: flood fill correctly identifies both components: %s\n", ok ? "confirmed" : "MISMATCH");
    return ok ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++17 -Wall -Wextra -O2 130_cc_floodfill_cpu_baseline.cpp -o 130_cc_floodfill_cpu_baseline
./130_cc_floodfill_cpu_baseline
```

**Sample input:** the book's own 6-vertex graph (edges now treated as undirected) plus a new, separate 2-vertex component {6, 7}.

**Sample output:**

```text
=== Section 24.1 CPU baseline: connected components via flood fill ===

undirected adjacency:
  vertex 0: 1 2 
  vertex 1: 0 2 3 
  vertex 2: 3 1 0 4 
  vertex 3: 2 4 5 1 
  vertex 4: 3 5 2 
  vertex 5: 4 3 
  vertex 6: 7 
  vertex 7: 6 

flood-filling from vertex 0 (its own id becomes the component label):
  vertex 1 labeled 0 (reached from 0)
  vertex 2 labeled 0 (reached from 0)
  vertex 3 labeled 0 (reached from 1)
  vertex 4 labeled 0 (reached from 2)
  vertex 5 labeled 0 (reached from 3)

flood-filling from vertex 6 (its own id becomes the component label):
  vertex 7 labeled 6 (reached from 6)

final labels: [ 0 0 0 0 0 0 6 6 ]

expected labels: [ 0 0 0 0 0 0 6 6 ] (2 components: {0..5} and {6,7},
each labeled by its own smallest/starting vertex id)

self-check: flood fill correctly identifies both components: confirmed
```

### The Concept, In Detail

```
ASCII view: two separate components, each labeled by its own
smallest/starting vertex id.

  component A (start = vertex 0):        component B (start = vertex 6):

      0---1                                    6---7
      |\ /|
      | X |
      |/ \|
      2---3
       \ /
        4---5

  every vertex reached from 0 is labeled 0    every vertex reached from 6
  (vertices 0,1,2,3,4,5)                      is labeled 6 (vertices 6,7)
```

Flood fill's correctness rests on a simple fact: two vertices end up with the same label if and only if some path of undirected edges connects them, since the flood-fill walk from a given start visits EVERY vertex reachable from it and nothing else. Once a vertex is labeled, later flood fills starting elsewhere skip it entirely (the `label[start] != UNVISITED` check), which is exactly why every vertex ends up with exactly one label rather than being revisited or double-counted.

[COMMON TRAP]
It is tempting to think the starting vertex for each flood fill matters to the final grouping -- that starting from vertex 3 instead of vertex 0 might produce a different set of components. Which vertices end up grouped together depends only on which edges connect them, never on which vertex happens to be visited first; only the LABEL each group receives (chosen here as the starting vertex's own id, by convention) depends on the start, not the grouping itself. Two vertices are in the same component regardless of which of them a flood fill happens to reach first.

### Code and Verification

```cpp
#include <cstdio>
#include <vector>

// Chapter 24.1 main -- flood fill processes ONE component at a time,
// finishing it completely before starting the next -- a poor fit for a
// GPU, which wants to do useful work across the WHOLE graph at once
// regardless of how many separate components it contains. Label
// propagation instead gives every vertex its own id as an initial
// label, then repeatedly has each vertex adopt the SMALLEST label
// among itself and all its neighbors, via `atomicMin` -- exactly
// Chapter 23.3's primitive, now spreading a MINIMUM value outward
// through a graph instead of a single shortest-path candidate. Every
// vertex in a component eventually converges on that component's
// smallest vertex id, entirely independent of which vertex the flood
// fill would have started from.
#define V 8
#define E 10

__global__ void label_propagation_kernel(const int* edge_a, const int* edge_b, int* label,
                                          int num_edges, int* any_change) {
    int e = threadIdx.x;
    if (e >= num_edges) return;
    int a = edge_a[e], b = edge_b[e];
    int la = label[a], lb = label[b];
    if (lb < la) { atomicMin(&label[a], lb); atomicOr(any_change, 1); }
    if (la < lb) { atomicMin(&label[b], la); atomicOr(any_change, 1); }
}

// ---- Host-side replay of the identical per-thread logic, using a
// start-of-round snapshot (no thread sees another thread's write from
// the SAME round -- a real kernel launch offers no such guarantee, but
// this matches every prior round-based host replay in this book). ----

int main() {
    printf("=== Section 24.1 main: parallel label propagation via atomicMin ===\n\n");

    std::vector<int> edge_a = {2, 0, 3, 1, 0, 4, 2, 3, 1, 6};
    std::vector<int> edge_b = {3, 1, 4, 2, 2, 5, 4, 5, 3, 7};
    printf("%d edges, %d vertices, every vertex starts as its own label:\n", E, V);
    printf("  label = [ 0 1 2 3 4 5 6 7 ]\n\n");

    std::vector<int> label(V);
    for (int v = 0; v < V; v++) label[v] = v;

    int round = 0;
    while (true) {
        round++;
        std::vector<int> snapshot = label;
        bool changed = false;
        for (int e = 0; e < E; e++) {
            int a = edge_a[e], b = edge_b[e];
            int la = snapshot[a], lb = snapshot[b];
            if (lb < label[a]) { label[a] = lb; changed = true; }
            if (la < label[b]) { label[b] = la; changed = true; }
        }
        printf("round %d: label = [ ", round);
        for (int l : label) printf("%d ", l);
        printf("]%s\n", changed ? "" : "  (no change -- converged)");
        if (!changed) break;
    }

    std::vector<int> expected_label = {0, 0, 0, 0, 0, 0, 6, 6};
    bool ok = (label == expected_label) && (round == 4);

    printf("\nexpected final label: [ 0 0 0 0 0 0 6 6 ], converging after %d rounds\n", 4);
    printf("(every vertex in {0..5} settles on label 0, every vertex in {6,7}\n");
    printf("settles on label 6 -- the SAME grouping the CPU baseline's flood\n");
    printf("fill found, discovered without ever picking a single starting\n");
    printf("vertex or processing one component at a time)\n");

    printf("\nself-check: parallel label propagation converges to the exact same\n");
    printf("component labels as flood fill: %s\n", ok ? "confirmed" : "MISMATCH");
    return ok ? 0 : 1;
}
```

**Compile and run:**

```bash
NVDIR=/usr/local/lib/python3.11/dist-packages/nvidia/cu13
$NVDIR/bin/nvcc -arch=sm_80 -I$NVDIR/include -L$NVDIR/lib -lcudart 131_cc_label_propagation_kernel.cu -o 131_cc_label_propagation_kernel
LD_LIBRARY_PATH=$NVDIR/lib ./131_cc_label_propagation_kernel
```

**Sample input:** the same 10 undirected edges, with every vertex starting as its own label and repeatedly adopting the smallest label among itself and its neighbors via `atomicMin`.

**Sample output:**

```text
=== Section 24.1 main: parallel label propagation via atomicMin ===

10 edges, 8 vertices, every vertex starts as its own label:
  label = [ 0 1 2 3 4 5 6 7 ]

round 1: label = [ 0 0 0 1 2 3 6 6 ]
round 2: label = [ 0 0 0 0 0 1 6 6 ]
round 3: label = [ 0 0 0 0 0 0 6 6 ]
round 4: label = [ 0 0 0 0 0 0 6 6 ]  (no change -- converged)

expected final label: [ 0 0 0 0 0 0 6 6 ], converging after 4 rounds
(every vertex in {0..5} settles on label 0, every vertex in {6,7}
settles on label 6 -- the SAME grouping the CPU baseline's flood
fill found, discovered without ever picking a single starting
vertex or processing one component at a time)

self-check: parallel label propagation converges to the exact same
component labels as flood fill: confirmed
```

## 24.2 Hooking: Merging Components via Union by Smaller Root

### Intuition

Label propagation (Section 24.1's parallel version) works, but a label can only travel one edge per round, so it can take as many rounds as the graph's diameter to fully converge. Hooking takes a different approach, borrowed from union-find: every vertex starts as the root of its own single-vertex tree, and for any edge whose two endpoints currently have different roots, the tree with the LARGER root is grafted ("hooked") underneath the tree with the smaller one. Once every edge connects two vertices that already share a root, every remaining tree exactly matches one connected component -- but which TREE SHAPE that process settles into depends heavily on whether hooks are seen immediately (as the sequential version sees its own updates) or only as a start-of-round snapshot (as a real parallel launch requires).

### The Sequential (CPU) Baseline

```cpp
#include <cstdio>
#include <vector>
#include <algorithm>

// Chapter 24.2 -- The Sequential (CPU) Baseline.
// Label propagation (Section 24.1) can take as many rounds as the
// graph's diameter, since a label has to hop across one edge per
// round. HOOKING takes a different approach, borrowed from union-find:
// every vertex starts as the root of its own single-vertex tree
// (`parent[v] = v`), and for every edge whose two endpoints currently
// have DIFFERENT roots, the tree with the LARGER root is grafted
// ("hooked") underneath the tree with the smaller one. Once every
// edge connects two vertices already sharing a root, every tree
// exactly matches one connected component. This sequential version
// finds each root with a straightforward walk up the current parent
// chain, seeing its OWN hooks immediately as it works through the
// edge list.
#define V 8
#define E 10

int find_root(const std::vector<int>& parent, int v) {
    while (parent[v] != v) v = parent[v];
    return v;
}

int main() {
    printf("=== Section 24.2 CPU baseline: connected components via hooking ===\n\n");

    std::vector<std::pair<int,int>> edges = {
        {2,3}, {0,1}, {3,4}, {1,2}, {0,2}, {4,5}, {2,4}, {3,5}, {1,3}, {6,7},
    };
    // Every undirected edge is considered in both directions.
    std::vector<std::pair<int,int>> directed;
    for (auto& e : edges) { directed.push_back(e); directed.push_back({e.second, e.first}); }

    std::vector<int> parent(V);
    for (int v = 0; v < V; v++) parent[v] = v;
    printf("initial parent: [ 0 1 2 3 4 5 6 7 ] (every vertex its own root)\n\n");

    int round = 0;
    while (true) {
        round++;
        bool any_hook = false;
        printf("round %d:\n", round);
        for (auto& e : directed) {
            int u = e.first, w = e.second;
            int ru = find_root(parent, u), rw = find_root(parent, w);
            if (ru == rw) continue;
            int hi = std::max(ru, rw), lo = std::min(ru, rw);
            printf("  edge(%d,%d): root(%d)=%d, root(%d)=%d -- hook parent[%d]=%d\n",
                   u, w, u, ru, w, rw, hi, lo);
            parent[hi] = lo;
            any_hook = true;
        }
        printf("  parent after round %d: [ ", round);
        for (int p : parent) printf("%d ", p);
        printf("]\n");
        if (!any_hook) { printf("  no hooks this round -- converged\n"); break; }
    }

    printf("\nfinal parent: [ ");
    for (int p : parent) printf("%d ", p);
    printf("]\n");
    printf("chain depth of each vertex (steps to its root):\n");
    for (int v = 0; v < V; v++) {
        int depth = 0, cur = v;
        while (parent[cur] != cur) { cur = parent[cur]; depth++; }
        printf("  vertex %d: root=%d, depth=%d\n", v, cur, depth);
    }

    std::vector<int> expected_parent = {0, 0, 0, 2, 2, 0, 6, 6};
    bool ok = (parent == expected_parent) && (round == 2);

    printf("\nexpected final parent: [ 0 0 0 2 2 0 6 6 ], converged after 2 rounds\n");
    printf("(1 round of real hooks, 1 confirming round with none)\n");
    printf("\nself-check: sequential hooking converges to the expected parent array: %s\n", ok ? "confirmed" : "MISMATCH");
    return ok ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++17 -Wall -Wextra -O2 132_cc_hooking_cpu_baseline.cpp -o 132_cc_hooking_cpu_baseline
./132_cc_hooking_cpu_baseline
```

**Sample input:** the same 10 undirected edges, with `find_root` walking the LIVE parent array -- seeing its own hooks the instant they happen, even within the same pass over the edge list.

**Sample output:**

```text
=== Section 24.2 CPU baseline: connected components via hooking ===

initial parent: [ 0 1 2 3 4 5 6 7 ] (every vertex its own root)

round 1:
  edge(2,3): root(2)=2, root(3)=3 -- hook parent[3]=2
  edge(0,1): root(0)=0, root(1)=1 -- hook parent[1]=0
  edge(3,4): root(3)=2, root(4)=4 -- hook parent[4]=2
  edge(1,2): root(1)=0, root(2)=2 -- hook parent[2]=0
  edge(4,5): root(4)=0, root(5)=5 -- hook parent[5]=0
  edge(6,7): root(6)=6, root(7)=7 -- hook parent[7]=6
  parent after round 1: [ 0 0 0 2 2 0 6 6 ]
round 2:
  parent after round 2: [ 0 0 0 2 2 0 6 6 ]
  no hooks this round -- converged

final parent: [ 0 0 0 2 2 0 6 6 ]
chain depth of each vertex (steps to its root):
  vertex 0: root=0, depth=0
  vertex 1: root=0, depth=1
  vertex 2: root=0, depth=1
  vertex 3: root=0, depth=2
  vertex 4: root=0, depth=2
  vertex 5: root=0, depth=1
  vertex 6: root=6, depth=0
  vertex 7: root=6, depth=1

expected final parent: [ 0 0 0 2 2 0 6 6 ], converged after 2 rounds
(1 round of real hooks, 1 confirming round with none)

self-check: sequential hooking converges to the expected parent array: confirmed
```

### The Concept, In Detail

```
ASCII view: LIVE sequential hooking vs SNAPSHOT parallel hooking, on
the identical edge list.

  sequential (sees its own hooks immediately):
    parent[3]=2 happens, then a LATER edge in the SAME pass already
    sees root(3) as root(2), not needing a second round to catch up
    -> converges to a SHALLOW tree: parent = [0,0,0,2,2,0,6,6], depth <= 2

  parallel (every thread reads one shared, unchanging snapshot):
    ALL edges compute their roots from the SAME start-of-round array --
    none of them can see hooks proposed by other threads THIS round
    -> converges to a DEEPER tree: parent = [0,0,0,1,2,3,6,6], depth <= 3
```

Both versions reach the exact same final grouping -- {0,1,2,3,4,5} and {6,7} -- because hooking's correctness never depended on the ORDER hooks happen in, only on eventually connecting every edge's endpoints under a shared root. What differs is the SHAPE of the resulting tree: the sequential version's `find_root` calls see every hook the moment it happens, so later edges in the same pass can shortcut through work already done, while a real GPU kernel launch offers no such guarantee -- every thread genuinely reads the array as it stood when the round began. That gap between "what the sequential intuition suggests" and "what actual concurrent hardware guarantees" is exactly what Section 24.2's own parallel kernel demonstrates next.

[COMMON TRAP]
It is tempting to think a parallel kernel modeled on this sequential baseline would produce the identical parent array, just computed faster. A real kernel launch gives no thread visibility into another thread's writes from the same launch -- every thread's view of the parent array is a snapshot from before the round started, not a live, continuously-updated structure the way a single sequential pass over the edge list is. Assuming otherwise is exactly how a "port the CPU code to CUDA" translation quietly produces a DIFFERENT, deeper tree than intended, or -- if the snapshot discipline is skipped entirely -- an actual data race.

### Code and Verification

```cpp
#include <cstdio>
#include <vector>
#include <algorithm>

// Chapter 24.2 main -- one thread per (directed) edge, all launched at
// once: every thread reads the parent array as it stood at the START
// of the round (no thread can see another thread's write from the
// SAME launch), computes both endpoints' current roots, and if they
// differ, attempts to hook the larger root under the smaller one via
// `atomicMin(&parent[hi], lo)`. Because every thread reads the SAME
// stale, all-self-parented snapshot, MANY different edges can propose
// DIFFERENT hook targets for the SAME root in the very first round --
// a genuine multi-way race that Section 24.2's sequential baseline,
// seeing its own updates immediately, never had to resolve.
#define V 8
#define E 10

__device__ int find_root_dev(const int* parent, int v) {
    while (parent[v] != v) v = parent[v];
    return v;
}

__global__ void hooking_kernel(const int* parent_snapshot, int* parent,
                                const int* edge_a, const int* edge_b, int num_directed_edges) {
    int e = threadIdx.x;
    if (e >= num_directed_edges) return;
    int u = edge_a[e], w = edge_b[e];
    int ru = find_root_dev(parent_snapshot, u);
    int rw = find_root_dev(parent_snapshot, w);
    if (ru == rw) return;
    int hi = max(ru, rw), lo = min(ru, rw);
    atomicMin(&parent[hi], lo);
}

// ---- Host-side replay of the identical per-thread logic. ----

int find_root_host(const std::vector<int>& parent, int v) {
    while (parent[v] != v) v = parent[v];
    return v;
}

int main() {
    printf("=== Section 24.2 main: parallel hooking via atomicMin, snapshot semantics ===\n\n");

    std::vector<std::pair<int,int>> edges = {
        {2,3}, {0,1}, {3,4}, {1,2}, {0,2}, {4,5}, {2,4}, {3,5}, {1,3}, {6,7},
    };
    std::vector<std::pair<int,int>> directed;
    for (auto& e : edges) { directed.push_back(e); directed.push_back({e.second, e.first}); }

    std::vector<int> parent(V);
    for (int v = 0; v < V; v++) parent[v] = v;

    int round = 0;
    while (true) {
        round++;
        std::vector<int> snapshot = parent;   // every thread reads THIS, unchanged, all round
        printf("round %d: snapshot = [ ", round);
        for (int p : snapshot) printf("%d ", p);
        printf("]\n");

        bool any_hook = false;
        for (auto& e : directed) {
            int u = e.first, w = e.second;
            int ru = find_root_host(snapshot, u), rw = find_root_host(snapshot, w);
            if (ru == rw) continue;
            int hi = std::max(ru, rw), lo = std::min(ru, rw);
            if (lo < parent[hi]) {
                printf("  edge(%d,%d): root(%d)=%d, root(%d)=%d -- atomicMin(parent[%d], %d) succeeds (was %d)\n",
                       u, w, u, ru, w, rw, hi, lo, parent[hi]);
                parent[hi] = lo;
                any_hook = true;
            }
        }
        printf("  parent after round %d: [ ", round);
        for (int p : parent) printf("%d ", p);
        printf("]\n");
        if (!any_hook) { printf("  no hooks this round -- converged\n"); break; }
    }

    printf("\nchain depth of each vertex (steps to its root):\n");
    for (int v = 0; v < V; v++) {
        int depth = 0, cur = v;
        while (parent[cur] != cur) { cur = parent[cur]; depth++; }
        printf("  vertex %d: root=%d, depth=%d\n", v, cur, depth);
    }

    std::vector<int> expected_parent = {0, 0, 0, 1, 2, 3, 6, 6};
    bool ok = (parent == expected_parent) && (round == 2);

    printf("\nexpected final parent: [ 0 0 0 1 2 3 6 6 ], converged after 2 rounds\n");
    printf("(the SAME 2 components as the sequential baseline, but a DEEPER,\n");
    printf("differently-shaped tree -- vertex 5 sits 3 hops from its root here,\n");
    printf("versus only 2 hops in the sequential version -- because every\n");
    printf("thread this round worked from the SAME stale, all-self snapshot\n");
    printf("instead of seeing hooks as they happened)\n");

    printf("\nself-check: parallel hooking finds the same 2 components, via a\n");
    printf("deeper tree, matching the expected parent array exactly: %s\n", ok ? "confirmed" : "MISMATCH");
    return ok ? 0 : 1;
}
```

**Compile and run:**

```bash
NVDIR=/usr/local/lib/python3.11/dist-packages/nvidia/cu13
$NVDIR/bin/nvcc -arch=sm_80 -I$NVDIR/include -L$NVDIR/lib -lcudart 133_cc_hooking_parallel_kernel.cu -o 133_cc_hooking_parallel_kernel
LD_LIBRARY_PATH=$NVDIR/lib ./133_cc_hooking_parallel_kernel
```

**Sample input:** the same 10 undirected edges (20 directed passes), every thread reading one shared start-of-round snapshot and racing via `atomicMin` to hook larger roots under smaller ones.

**Sample output:**

```text
=== Section 24.2 main: parallel hooking via atomicMin, snapshot semantics ===

round 1: snapshot = [ 0 1 2 3 4 5 6 7 ]
  edge(2,3): root(2)=2, root(3)=3 -- atomicMin(parent[3], 2) succeeds (was 3)
  edge(0,1): root(0)=0, root(1)=1 -- atomicMin(parent[1], 0) succeeds (was 1)
  edge(3,4): root(3)=3, root(4)=4 -- atomicMin(parent[4], 3) succeeds (was 4)
  edge(1,2): root(1)=1, root(2)=2 -- atomicMin(parent[2], 1) succeeds (was 2)
  edge(0,2): root(0)=0, root(2)=2 -- atomicMin(parent[2], 0) succeeds (was 1)
  edge(4,5): root(4)=4, root(5)=5 -- atomicMin(parent[5], 4) succeeds (was 5)
  edge(2,4): root(2)=2, root(4)=4 -- atomicMin(parent[4], 2) succeeds (was 3)
  edge(3,5): root(3)=3, root(5)=5 -- atomicMin(parent[5], 3) succeeds (was 4)
  edge(1,3): root(1)=1, root(3)=3 -- atomicMin(parent[3], 1) succeeds (was 2)
  edge(6,7): root(6)=6, root(7)=7 -- atomicMin(parent[7], 6) succeeds (was 7)
  parent after round 1: [ 0 0 0 1 2 3 6 6 ]
round 2: snapshot = [ 0 0 0 1 2 3 6 6 ]
  parent after round 2: [ 0 0 0 1 2 3 6 6 ]
  no hooks this round -- converged

chain depth of each vertex (steps to its root):
  vertex 0: root=0, depth=0
  vertex 1: root=0, depth=1
  vertex 2: root=0, depth=1
  vertex 3: root=0, depth=2
  vertex 4: root=0, depth=2
  vertex 5: root=0, depth=3
  vertex 6: root=6, depth=0
  vertex 7: root=6, depth=1

expected final parent: [ 0 0 0 1 2 3 6 6 ], converged after 2 rounds
(the SAME 2 components as the sequential baseline, but a DEEPER,
differently-shaped tree -- vertex 5 sits 3 hops from its root here,
versus only 2 hops in the sequential version -- because every
thread this round worked from the SAME stale, all-self snapshot
instead of seeing hooks as they happened)

self-check: parallel hooking finds the same 2 components, via a
deeper tree, matching the expected parent array exactly: confirmed
```

## 24.3 Pointer Jumping: Flattening Chains to Their Roots in O(log Depth) Rounds

### Intuition

Section 24.2's parallel hooking reached the right grouping, but at the cost of a deeper tree than the sequential version ever produced -- vertex 5 sits 3 hops from its root, and every future `find_root` call on it has to walk that entire chain. Pointer jumping, first used in Chapter 11 to flatten a linked list for parallel list ranking, solves exactly this: every vertex simultaneously sets `parent[v] = parent[parent[v]]`, which roughly halves each chain's remaining distance to its root on every round. After enough rounds, every parent pointer aims directly at its root, and any future `find_root` call becomes O(1) -- no walk required at all.

### The Sequential (CPU) Baseline

```cpp
#include <cstdio>
#include <vector>

// Chapter 24.3 -- The Sequential (CPU) Baseline.
// Section 24.2's PARALLEL hooking (file 133) converged to the right
// grouping, but to a DEEPER tree than the sequential version ever
// produced: parent = [0,0,0,1,2,3,6,6], with vertex 5 sitting 3 hops
// from its root (5 -> 3 -> 1 -> 0). A `find_root` walk over a tree
// like that costs O(depth) per call -- fine once, expensive if every
// future query has to re-walk it. POINTER JUMPING, first introduced in
// Chapter 11 to flatten a linked list for parallel list ranking, solves
// exactly this here too: every vertex simultaneously sets
// `parent[v] = parent[parent[v]]`, which halves every chain's
// remaining distance to its root on each round. After enough rounds,
// every parent pointer aims directly at its root, and every future
// `find_root` call is O(1). This sequential version simulates the
// SAME simultaneous update (every vertex reads a start-of-round
// snapshot, never a value another "thread" already jumped this round)
// one round at a time.
#define V 8

int main() {
    printf("=== Section 24.3 CPU baseline: flattening via pointer jumping ===\n\n");

    // Starting point: the DEEPER tree section 24.2's parallel hooking
    // produced (file 133's final parent array), not the sequential
    // hooking baseline's shallower one -- pointer jumping earns its
    // keep precisely on trees like this.
    std::vector<int> parent = {0, 0, 0, 1, 2, 3, 6, 6};
    printf("starting parent (from section 24.2's parallel hooking): [ ");
    for (int p : parent) printf("%d ", p);
    printf("]\n");
    printf("starting chain depth of each vertex:\n");
    for (int v = 0; v < V; v++) {
        int depth = 0, cur = v;
        while (parent[cur] != cur) { cur = parent[cur]; depth++; }
        printf("  vertex %d: root=%d, depth=%d\n", v, cur, depth);
    }
    printf("\n");

    int round = 0;
    while (true) {
        round++;
        std::vector<int> snapshot = parent;   // every vertex jumps from THIS, unchanged, all round
        printf("round %d: snapshot = [ ", round);
        for (int p : snapshot) printf("%d ", p);
        printf("]\n");

        std::vector<int> next(V);
        for (int v = 0; v < V; v++) {
            next[v] = snapshot[snapshot[v]];
            if (next[v] != snapshot[v]) {
                printf("  vertex %d: parent[%d]=%d -> parent[parent[%d]]=parent[%d]=%d\n",
                       v, v, snapshot[v], v, snapshot[v], next[v]);
            }
        }
        parent = next;
        printf("  parent after round %d: [ ", round);
        for (int p : parent) printf("%d ", p);
        printf("]\n");

        // Convergence check: is every vertex ALREADY pointing directly
        // at its root, i.e. would jumping again be a no-op for
        // everyone? Unlike hooking's "no hooks this round" test, this
        // can be checked immediately from the array we just produced --
        // no separate confirming round is needed, since the condition
        // `parent[v] == parent[parent[v]]` for every v is exactly the
        // definition of "fully flattened."
        bool flattened = true;
        for (int v = 0; v < V; v++) {
            if (parent[v] != parent[parent[v]]) { flattened = false; break; }
        }
        if (flattened) { printf("  every vertex already points directly at its root -- converged\n"); break; }
    }

    printf("\nfinal chain depth of each vertex:\n");
    for (int v = 0; v < V; v++) {
        int depth = 0, cur = v;
        while (parent[cur] != cur) { cur = parent[cur]; depth++; }
        printf("  vertex %d: root=%d, depth=%d\n", v, cur, depth);
    }

    std::vector<int> expected_parent = {0, 0, 0, 0, 0, 0, 6, 6};
    bool ok = (parent == expected_parent) && (round == 2);

    printf("\nexpected final parent: [ 0 0 0 0 0 0 6 6 ], converged after 2 rounds\n");
    printf("(the SAME 2 components as ever, now with every vertex exactly one\n");
    printf("hop from its root -- vertex 5's chain of length 3 was flattened in\n");
    printf("just 2 rounds, not 3, since pointer jumping roughly HALVES the\n");
    printf("remaining distance to the root every round: 3 -> 2 -> 1 hop(s))\n");

    printf("\nself-check: pointer jumping flattens the deeper parallel-hooking\n");
    printf("tree to the expected fully-flattened parent array: %s\n", ok ? "confirmed" : "MISMATCH");
    return ok ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++17 -Wall -Wextra -O2 134_cc_pointer_jump_cpu_baseline.cpp -o 134_cc_pointer_jump_cpu_baseline
./134_cc_pointer_jump_cpu_baseline
```

**Sample input:** the deeper tree Section 24.2's parallel hooking produced (`parent = [0,0,0,1,2,3,6,6]`), flattened via simultaneous `parent[v] = parent[parent[v]]` jumps against a start-of-round snapshot.

**Sample output:**

```text
=== Section 24.3 CPU baseline: flattening via pointer jumping ===

starting parent (from section 24.2's parallel hooking): [ 0 0 0 1 2 3 6 6 ]
starting chain depth of each vertex:
  vertex 0: root=0, depth=0
  vertex 1: root=0, depth=1
  vertex 2: root=0, depth=1
  vertex 3: root=0, depth=2
  vertex 4: root=0, depth=2
  vertex 5: root=0, depth=3
  vertex 6: root=6, depth=0
  vertex 7: root=6, depth=1

round 1: snapshot = [ 0 0 0 1 2 3 6 6 ]
  vertex 3: parent[3]=1 -> parent[parent[3]]=parent[1]=0
  vertex 4: parent[4]=2 -> parent[parent[4]]=parent[2]=0
  vertex 5: parent[5]=3 -> parent[parent[5]]=parent[3]=1
  parent after round 1: [ 0 0 0 0 0 1 6 6 ]
round 2: snapshot = [ 0 0 0 0 0 1 6 6 ]
  vertex 5: parent[5]=1 -> parent[parent[5]]=parent[1]=0
  parent after round 2: [ 0 0 0 0 0 0 6 6 ]
  every vertex already points directly at its root -- converged

final chain depth of each vertex:
  vertex 0: root=0, depth=0
  vertex 1: root=0, depth=1
  vertex 2: root=0, depth=1
  vertex 3: root=0, depth=1
  vertex 4: root=0, depth=1
  vertex 5: root=0, depth=1
  vertex 6: root=6, depth=0
  vertex 7: root=6, depth=1

expected final parent: [ 0 0 0 0 0 0 6 6 ], converged after 2 rounds
(the SAME 2 components as ever, now with every vertex exactly one
hop from its root -- vertex 5's chain of length 3 was flattened in
just 2 rounds, not 3, since pointer jumping roughly HALVES the
remaining distance to the root every round: 3 -> 2 -> 1 hop(s))

self-check: pointer jumping flattens the deeper parallel-hooking
tree to the expected fully-flattened parent array: confirmed
```

### The Concept, In Detail

```
ASCII view: vertex 5's chain, flattening one jump at a time.

  before:   5 -> 3 -> 1 -> 0        (3 hops to root)
  round 1:  5 -> 1 -> 0             (parent[5] = parent[parent[5]] = parent[3] = 1)
  round 2:  5 -> 0                  (parent[5] = parent[parent[5]] = parent[1] = 0)

  each jump roughly HALVES the remaining distance to the root:
  3 hops -> 2 hops -> 1 hop, converging in ceil(log2(3)) = 2 rounds
```

Pointer jumping's stopping condition is different from every earlier round-based algorithm in this chapter: rather than waiting for a round in which nothing changes (which would need one extra, wasted "confirming" round to detect), convergence can be checked directly from the array a round just produced -- if `parent[v] == parent[parent[v]]` already holds for every vertex, then every parent pointer already aims straight at its root, and jumping again would be a pure no-op for everyone. That is precisely the state reached here after only 2 rounds, even though vertex 5 started 3 hops from its root.

[COMMON TRAP]
It is tempting to think pointer jumping needs one round per unit of chain depth, the same way `find_root`'s ORIGINAL walk would need one step per unit of depth. Each round jumps every vertex to its GRANDPARENT, not merely its parent, which is why depth shrinks by roughly half each round rather than by one -- a chain of depth 3 flattens in 2 rounds, and a chain of depth 1000 would need only about 10, not 999.

### Code and Verification

```cpp
#include <cstdio>
#include <vector>

// Chapter 24.3 main -- one thread per vertex, all launched at once.
// Every thread does exactly ONE thing: read `parent[v]` and
// `parent[parent[v]]` from the round's start-of-round snapshot, then
// write the result into `parent[v]` -- its OWN slot, and nobody
// else's. That is the entire reason this kernel needs NO atomics at
// all, unlike every other GPU kernel this chapter has written:
// hooking (file 133) had many threads racing to write the SAME
// `parent[hi]` slot, but here thread v only ever touches index v.
// Different threads' reads can overlap freely with other threads'
// writes to DIFFERENT slots with no danger, because CUDA guarantees
// no data race exists when no two threads touch the same memory
// location and at least one of them writes.
#define V 8

__global__ void pointer_jump_kernel(const int* parent_snapshot, int* parent) {
    int v = threadIdx.x;
    if (v >= V) return;
    parent[v] = parent_snapshot[parent_snapshot[v]];
}

// ---- Host-side replay of the identical per-thread logic. ----

int main() {
    printf("=== Section 24.3 main: parallel pointer jumping, no atomics needed ===\n\n");

    std::vector<int> parent = {0, 0, 0, 1, 2, 3, 6, 6};
    printf("starting parent (from section 24.2's parallel hooking): [ ");
    for (int p : parent) printf("%d ", p);
    printf("]\n\n");

    int round = 0;
    while (true) {
        round++;
        std::vector<int> snapshot = parent;   // every thread reads THIS, unchanged, all round
        printf("round %d: snapshot = [ ", round);
        for (int p : snapshot) printf("%d ", p);
        printf("]\n");

        std::vector<int> next(V);
        for (int v = 0; v < V; v++) {
            next[v] = snapshot[snapshot[v]];   // thread v: one read of its own chain, one write to slot v
        }
        parent = next;
        printf("  parent after round %d: [ ", round);
        for (int p : parent) printf("%d ", p);
        printf("]\n");

        bool flattened = true;
        for (int v = 0; v < V; v++) {
            if (parent[v] != parent[parent[v]]) { flattened = false; break; }
        }
        if (flattened) { printf("  every vertex already points directly at its root -- converged\n"); break; }
    }

    printf("\nfinal chain depth of each vertex:\n");
    for (int v = 0; v < V; v++) {
        int depth = 0, cur = v;
        while (parent[cur] != cur) { cur = parent[cur]; depth++; }
        printf("  vertex %d: root=%d, depth=%d\n", v, cur, depth);
    }

    std::vector<int> expected_parent = {0, 0, 0, 0, 0, 0, 6, 6};
    bool ok = (parent == expected_parent) && (round == 2);

    printf("\nexpected final parent: [ 0 0 0 0 0 0 6 6 ], converged after 2 rounds,\n");
    printf("every vertex exactly one hop from its root, computed with a kernel\n");
    printf("that never called a single atomic operation: each thread only ever\n");
    printf("reads a shared snapshot and writes its own, private array slot\n");

    printf("\nself-check: parallel pointer jumping reproduces the exact same\n");
    printf("fully-flattened parent array, atomic-free: %s\n", ok ? "confirmed" : "MISMATCH");
    return ok ? 0 : 1;
}
```

**Compile and run:**

```bash
NVDIR=/usr/local/lib/python3.11/dist-packages/nvidia/cu13
$NVDIR/bin/nvcc -arch=sm_80 -I$NVDIR/include -L$NVDIR/lib -lcudart 135_cc_pointer_jump_kernel.cu -o 135_cc_pointer_jump_kernel
LD_LIBRARY_PATH=$NVDIR/lib ./135_cc_pointer_jump_kernel
```

**Sample input:** the same deeper tree, flattened by one thread per vertex, each reading a shared snapshot and writing only its own array slot -- no atomics needed anywhere in this kernel.

**Sample output:**

```text
=== Section 24.3 main: parallel pointer jumping, no atomics needed ===

starting parent (from section 24.2's parallel hooking): [ 0 0 0 1 2 3 6 6 ]

round 1: snapshot = [ 0 0 0 1 2 3 6 6 ]
  parent after round 1: [ 0 0 0 0 0 1 6 6 ]
round 2: snapshot = [ 0 0 0 0 0 1 6 6 ]
  parent after round 2: [ 0 0 0 0 0 0 6 6 ]
  every vertex already points directly at its root -- converged

final chain depth of each vertex:
  vertex 0: root=0, depth=0
  vertex 1: root=0, depth=1
  vertex 2: root=0, depth=1
  vertex 3: root=0, depth=1
  vertex 4: root=0, depth=1
  vertex 5: root=0, depth=1
  vertex 6: root=6, depth=0
  vertex 7: root=6, depth=1

expected final parent: [ 0 0 0 0 0 0 6 6 ], converged after 2 rounds,
every vertex exactly one hop from its root, computed with a kernel
that never called a single atomic operation: each thread only ever
reads a shared snapshot and writes its own, private array slot

self-check: parallel pointer jumping reproduces the exact same
fully-flattened parent array, atomic-free: confirmed
```

## Chapter Summary

Connected components abandons the single-source framing every algorithm since Chapter 22 relied on, asking instead which vertices can reach which others at all, with edges treated as undirected. Flood fill answers this sequentially by repeatedly running Chapter 22's own BFS from each not-yet-labeled vertex; its parallel counterpart, label propagation, spreads each component's smallest vertex id outward every round via `atomicMin`, converging on the identical grouping without ever picking a starting vertex. Hooking offers a different, union-find-style route to the same grouping, but exposes a genuine gap between sequential intuition and parallel reality: a sequential pass sees its own hooks immediately and settles into a shallow tree, while a real parallel kernel -- every thread confined to a single start-of-round snapshot -- produces a deeper one, even though both reach the exact same components. Pointer jumping closes the chapter by flattening that deeper tree back down, having every vertex simultaneously jump to its grandparent (`parent[v] = parent[parent[v]]`) each round; because this roughly halves every chain's remaining depth, convergence takes only about log2(depth) rounds, and because each thread only ever reads shared data and writes its own private slot, it needs no atomics at all -- the first algorithm in this chapter's trio that doesn't.

## Self-Check Questions

1. Why does connected components treat every edge as undirected, in contrast to the directed edges Chapters 22 and 23 both assumed?
2. Why does the specific starting vertex chosen for a given flood fill never change which vertices end up grouped together?
3. Label propagation and flood fill always reach the same grouping. What does label propagation gain by spreading labels via `atomicMin` instead of running flood fill's sequential passes?
4. Why does sequential hooking converge to a shallower tree than a real parallel kernel implementing the "same" algorithm, even though both reach the identical final grouping?
5. Why does pointer jumping's convergence check not need an extra "confirming" round the way label propagation's and hooking's do?
6. Why does the pointer-jumping kernel need no atomic operations at all, when both the label-propagation and hooking kernels earlier in this chapter did?

## Where We Go Next

Connected components asks which vertices belong together, with no notion of cost at all. Chapter 25 closes out Part 6 by combining that grouping question with Chapter 23's notion of edge weight: minimum spanning trees ask for the cheapest possible set of edges that connects every vertex in a graph into a single component, introducing Boruvka's algorithm -- a parallel-native approach built directly out of this chapter's own hooking and pointer-jumping machinery.

## Worked Solutions

**1.** Directed edges made sense for Chapters 22 and 23 because both were fundamentally about REACHING vertices from a specific source, where the direction of travel matters (an edge from A to B lets you go from A to B, not necessarily back). Connected components asks a different question entirely -- which vertices can reach which others AT ALL, from no particular source -- and for that question, an edge simply establishes that two vertices belong together, regardless of which direction it happens to be drawn in; treating edges as undirected is what makes "connected" mean the same thing in both directions.

**2.** Which vertices end up grouped together is determined entirely by which edges connect them -- two vertices share a component if and only if some path of edges connects them, a fact that has nothing to do with which vertex a flood fill happens to visit first. Starting from a different vertex within the same component would produce the exact same set of vertices being visited (just possibly in a different order), so only the LABEL attached to that group (chosen by convention as the starting vertex's own id) depends on the start -- never the grouping itself.

**3.** Flood fill processes one component completely before starting the next, which is a poor fit for a GPU that wants to do useful work across the WHOLE graph at once. Label propagation instead lets every vertex in every component participate simultaneously every round, via `atomicMin` -- there is no need to single out one "current" component being processed, and no thread has to wait for an entire other component's flood fill to finish before starting its own work.

**4.** Sequential hooking's `find_root` walks a LIVE parent array, so a hook performed earlier in the same pass over the edge list is immediately visible to every later edge processed in that same pass, letting later edges shortcut through work already done. A real parallel kernel launch offers no such guarantee -- every thread reads the array as it stood at the START of the round, seeing none of the other threads' hooks from that same round -- so multiple edges can propose different hook targets for the same root at once, and the resulting tree ends up both correct and genuinely deeper than the sequential version's.

**5.** Label propagation's and hooking's convergence check is "did anything change this round," which can only be answered AFTER a round has run with no effect -- requiring one extra, otherwise wasted round purely to confirm nothing changed. Pointer jumping's convergence condition, `parent[v] == parent[parent[v]]` for every vertex, can instead be evaluated directly on the array a round JUST produced, without needing to run a further round to find out that nothing would change -- the array itself already reveals whether every pointer is fully flattened.

**6.** Both the label-propagation and hooking kernels have MULTIPLE threads potentially writing to the SAME shared array slot at once (several edges into the same vertex, or several roots hooking under the same target), which is exactly the situation atomics exist to make safe. The pointer-jumping kernel assigns each thread its own private slot -- thread v reads shared data but writes ONLY `parent[v]`, and no other thread ever writes that same location -- so there is no shared-write race to protect against in the first place, and CUDA's guarantee that no data race exists when no two threads touch the same memory location (with at least one writing) applies automatically.
