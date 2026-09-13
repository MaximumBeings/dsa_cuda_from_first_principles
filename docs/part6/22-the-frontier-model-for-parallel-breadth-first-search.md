# Chapter 22: The Frontier Model for Parallel Breadth-First Search

Chapter 21 built a graph representation a thread can index into; this chapter puts it to work. Breadth-first search's sequential form processes one vertex at a time off a FIFO queue -- but every vertex at the SAME distance from the source is, in principle, independent of every other vertex at that same distance, which is exactly what makes BFS parallelizable once it is reframed around the FRONTIER: the entire set of vertices at the current distance, expanded all at once. This chapter builds that reframing in three pieces: expanding a frontier in parallel and the genuine race it exposes (two different frontier vertices discovering the same neighbor at once), compacting the newly-discovered vertices into the next frontier (a direct callback to Chapter 6's stream compaction), and choosing between vertex-parallel and edge-parallel expansion -- Chapter 21.3's CSR-vs-COO lesson, now applied to a frontier instead of a whole graph.

## 22.1 Sequential BFS and the Race to Discover a Shared Neighbor

### Intuition

Breadth-first search visits a graph outward in expanding rings from a source vertex: the source itself, then everything one edge away, then everything two edges away, and so on. The classic sequential form tracks this with one FIFO queue, dequeuing a vertex and expanding its neighbors one at a time. Reframed around explicit per-level FRONTIERS instead, the picture becomes: expand every vertex in the CURRENT frontier at once, one thread per frontier vertex, walking its own CSR slice (Chapter 21.1's own access pattern). This immediately exposes a new hazard -- two DIFFERENT frontier vertices can share a common unvisited neighbor, so two threads can race to be the one that "discovers" it, needing exactly the same `atomicCAS`-based fix Chapter 16.2 used for a trie's shared child slot.

### The Sequential (CPU) Baseline

```cpp
#include <cstdio>
#include <vector>

// Chapter 22.1 -- The Sequential (CPU) Baseline.
// Breadth-first search visits a graph in expanding "rings" outward from
// a source vertex: first the source itself, then every vertex exactly
// one edge away, then every vertex exactly two edges away, and so on.
// The classic sequential implementation tracks this with a single FIFO
// queue (here, an explicit array with head/tail indices, continuing
// this book's discipline against real pointers) -- a vertex is
// enqueued the moment it is first discovered, and dequeued once its
// own neighbors need expanding. Reusing Chapter 21's own 6-vertex CSR
// graph makes the connection explicit: BFS is the first algorithm in
// this book to actually WALK the structure Chapter 21 spent an entire
// chapter building.
#define V 6
#define UNVISITED (-1)

int main() {
    printf("=== Section 22.1 CPU baseline: sequential BFS via explicit queue ===\n\n");

    // Chapter 21's own CSR arrays for the same 6-vertex graph.
    std::vector<int> row_offsets = {0, 2, 4, 6, 8, 9, 9};
    std::vector<int> col_idx     = {1, 2, 2, 3, 3, 4, 4, 5, 5};
    printf("row_offsets = [ 0 2 4 6 8 9 9 ]\n");
    printf("col_idx     = [ 1 2 2 3 3 4 4 5 5 ]\n\n");

    std::vector<int> dist(V, UNVISITED);
    std::vector<int> queue;
    queue.reserve(V);
    int head = 0;

    int source = 0;
    dist[source] = 0;
    queue.push_back(source);
    printf("source vertex: %d\n\n", source);

    printf("dequeue order and neighbor expansion:\n");
    while (head < (int)queue.size()) {
        int u = queue[head];
        head++;
        printf("  dequeue(%d) [dist=%d], neighbors:", u, dist[u]);
        for (int i = row_offsets[u]; i < row_offsets[u + 1]; i++) {
            int w = col_idx[i];
            if (dist[w] == UNVISITED) {
                dist[w] = dist[u] + 1;
                queue.push_back(w);
                printf(" %d(newly discovered, dist=%d)", w, dist[w]);
            } else {
                printf(" %d(already visited, dist=%d)", w, dist[w]);
            }
        }
        printf("\n");
    }

    printf("\nfinal dist array: [ ");
    for (int d : dist) printf("%d ", d);
    printf("]\n\n");

    printf("grouping vertices by distance (each group is one BFS \"frontier\"):\n");
    for (int level = 0; level <= 3; level++) {
        printf("  level %d: { ", level);
        for (int v = 0; v < V; v++) if (dist[v] == level) printf("%d ", v);
        printf("}\n");
    }

    std::vector<int> expected_dist = {0, 1, 1, 2, 2, 3};
    std::vector<int> expected_visit_order = {0, 1, 2, 3, 4, 5};
    bool ok = (dist == expected_dist) && (queue == expected_visit_order);

    printf("\nexpected dist:       [ 0 1 1 2 2 3 ]\n");
    printf("expected dequeue order: 0 1 2 3 4 5\n");
    printf("\nself-check: BFS distances and dequeue order match expected values: %s\n", ok ? "confirmed" : "MISMATCH");
    return ok ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++17 -Wall -Wextra -O2 118_bfs_sequential_cpu_baseline.cpp -o 118_bfs_sequential_cpu_baseline
./118_bfs_sequential_cpu_baseline
```

**Sample input:** Chapter 21's own 6-vertex CSR graph, BFS'd from source vertex 0 via an explicit array-based FIFO queue.

**Sample output:**

```text
=== Section 22.1 CPU baseline: sequential BFS via explicit queue ===

row_offsets = [ 0 2 4 6 8 9 9 ]
col_idx     = [ 1 2 2 3 3 4 4 5 5 ]

source vertex: 0

dequeue order and neighbor expansion:
  dequeue(0) [dist=0], neighbors: 1(newly discovered, dist=1) 2(newly discovered, dist=1)
  dequeue(1) [dist=1], neighbors: 2(already visited, dist=1) 3(newly discovered, dist=2)
  dequeue(2) [dist=1], neighbors: 3(already visited, dist=2) 4(newly discovered, dist=2)
  dequeue(3) [dist=2], neighbors: 4(already visited, dist=2) 5(newly discovered, dist=3)
  dequeue(4) [dist=2], neighbors: 5(already visited, dist=3)
  dequeue(5) [dist=3], neighbors:

final dist array: [ 0 1 1 2 2 3 ]

grouping vertices by distance (each group is one BFS "frontier"):
  level 0: { 0 }
  level 1: { 1 2 }
  level 2: { 3 4 }
  level 3: { 5 }

expected dist:       [ 0 1 1 2 2 3 ]
expected dequeue order: 0 1 2 3 4 5

self-check: BFS distances and dequeue order match expected values: confirmed
```

### The Concept, In Detail

Grouping the CPU baseline's own dequeue order by distance reveals the frontier structure already implicit in FIFO order:

```
ASCII view: the graph's 4 BFS levels, and the shared-neighbor hazard.

  level 0: { 0 }
             |  \
             v   v
  level 1: { 1    2 }
             |  \  |  \
             v   v v   v
  level 2: {   3        4  }
                \       /
                 \     /
  level 3:  {       5       }

  Vertex 3 is a neighbor of BOTH vertex 1 AND vertex 2 -- level 1's
  entire frontier. If threads for 1 and 2 run at the same time, BOTH
  can see vertex 3 as unvisited at the same moment.
```

Because every thread discovering a given vertex would write the IDENTICAL distance value (both threads expanding level 1 would write `dist[3] = 2`), a naive unprotected race cannot corrupt the distance array itself -- but it CAN let both threads believe they were the one who made the discovery, which matters enormously once Section 22.2 uses "I made the discovery" to decide who gets to add the vertex to the next frontier. `atomicCAS(&dist[w], UNVISITED, level + 1)` resolves this precisely: it reports back whether the slot was still `UNVISITED` at the moment THIS thread's CAS executed, giving each thread an unambiguous, hardware-guaranteed answer to "was I first."

[COMMON TRAP]
It is tempting to think that because both racing threads would write the SAME value, no synchronization is needed at all -- unlike Chapter 19's hash-table insertion race, where different threads write different keys. The value being IDENTICAL protects the distance array's correctness, but it says nothing about how many times a vertex gets counted as newly discovered, added to a result list, or (Section 22.2) appended to the next frontier. A duplicate "discovery" of the same vertex is a real bug even when every write happens to agree on the value.

### Code and Verification

```cpp
#include <cstdio>
#include <vector>

// Chapter 22.1 main -- reframe BFS around the FRONTIER: the array of
// vertices at the current distance, all expanded AT ONCE, one thread
// per frontier vertex, walking its own CSR slice (Chapter 21.1's own
// pattern, uneven length per thread). The genuinely new hazard: two
// DIFFERENT frontier vertices can share a common unvisited neighbor --
// here, level 1's frontier {1, 2} both point to vertex 3 -- so two
// threads can race to discover the SAME vertex at the SAME moment.
// Both threads would write the identical distance value, so a naive
// unprotected "check then write" cannot corrupt `dist` itself, but it
// CAN let both threads believe they were the one who discovered
// vertex 3 -- exactly the kind of duplicate that would enqueue it
// twice into the next frontier. `atomicCAS(&dist[w], UNVISITED,
// level+1)` fixes this the same way it fixed Chapter 16.2's trie race:
// only the thread whose CAS actually flips the slot may claim the
// discovery.
#define V 6
#define UNVISITED (-1)

__global__ void frontier_expand_naive_kernel(const int* row_offsets, const int* col_idx,
                                              int* dist, const int* frontier, int frontier_size,
                                              int level, int* discovered_count) {
    int t = threadIdx.x;
    if (t >= frontier_size) return;
    int u = frontier[t];
    for (int i = row_offsets[u]; i < row_offsets[u + 1]; i++) {
        int w = col_idx[i];
        if (dist[w] == UNVISITED) {
            dist[w] = level + 1;
            atomicAdd(discovered_count, 1);   // counts DISCOVERY EVENTS, not distinct vertices
        }
    }
}

__global__ void frontier_expand_cas_kernel(const int* row_offsets, const int* col_idx,
                                            int* dist, const int* frontier, int frontier_size,
                                            int level, int* discovered_count) {
    int t = threadIdx.x;
    if (t >= frontier_size) return;
    int u = frontier[t];
    for (int i = row_offsets[u]; i < row_offsets[u + 1]; i++) {
        int w = col_idx[i];
        int old = atomicCAS(&dist[w], UNVISITED, level + 1);
        if (old == UNVISITED) {
            atomicAdd(discovered_count, 1);   // only the WINNING thread counts a discovery
        }
    }
}

// ---- Host-side replay of the identical per-thread logic, under the
// worst-case "every thread reads before any thread writes" interleaving
// (Chapter 16.2's own adversarial schedule), with a fixed tie-break:
// the lower thread index resolves first. ----

int main() {
    printf("=== Section 22.1 main: parallel frontier expansion, naive vs atomicCAS ===\n\n");

    std::vector<int> row_offsets = {0, 2, 4, 6, 8, 9, 9};
    std::vector<int> col_idx     = {1, 2, 2, 3, 3, 4, 4, 5, 5};
    std::vector<int> frontier = {1, 2};   // level 1's frontier, from Section 22.1's CPU baseline
    int level = 1;

    printf("expanding level %d's frontier {1, 2} (2 threads, one per frontier vertex):\n", level);
    printf("  thread 0 (vertex 1): neighbors 2, 3\n");
    printf("  thread 1 (vertex 2): neighbors 3, 4\n");
    printf("  vertex 3 is a SHARED unvisited neighbor of both -- a genuine race\n\n");

    // Pre-race state: vertices 0,1,2 already have distances from level 0.
    std::vector<int> dist_naive = {0, 1, 1, UNVISITED, UNVISITED, UNVISITED};
    std::vector<int> dist_cas   = dist_naive;

    printf("--- naive (unprotected check-then-write) ---\n");
    printf("wave: BOTH threads READ dist[3] BEFORE either WRITES (worst case)\n");
    int read_t0_at_3 = dist_naive[3];   // thread 0's read of dist[3], BEFORE any write
    int read_t1_at_3 = dist_naive[3];   // thread 1's read of dist[3], BEFORE any write (same moment)
    printf("  thread 0 reads dist[3]=%d -- believes it can claim vertex 3\n", read_t0_at_3);
    printf("  thread 1 reads dist[3]=%d -- believes it can claim vertex 3\n", read_t1_at_3);
    int naive_discovered = 0;
    // thread 0 (vertex 1): neighbor 2 (already visited, dist=1, skip -- no race, reads and writes atomically here for clarity)
    if (dist_naive[2] == UNVISITED) { dist_naive[2] = level + 1; naive_discovered++; }
    // thread 0 now acts on its EARLIER read of vertex 3
    if (read_t0_at_3 == UNVISITED) { dist_naive[3] = level + 1; naive_discovered++; printf("  thread 0 writes dist[3]=%d, counts a discovery\n", level + 1); }
    // thread 1 (vertex 2): neighbor 3 -- acts on ITS OWN earlier read, oblivious to thread 0's write
    if (read_t1_at_3 == UNVISITED) { dist_naive[3] = level + 1; naive_discovered++; printf("  thread 1 ALSO writes dist[3]=%d, ALSO counts a discovery -- DUPLICATE\n", level + 1); }
    // thread 1's other neighbor, 4 -- uncontended, no race
    if (dist_naive[4] == UNVISITED) { dist_naive[4] = level + 1; naive_discovered++; printf("  thread 1 writes dist[4]=%d, counts a discovery\n", level + 1); }

    printf("  dist after naive expansion: [ ");
    for (int d : dist_naive) printf("%d ", d);
    printf("]\n  discovered_count = %d (but only 2 DISTINCT vertices -- 3 and 4 -- were actually newly discovered)\n\n", naive_discovered);

    printf("--- atomicCAS-protected ---\n");
    printf("wave: both threads attempt atomicCAS(&dist[3], UNVISITED, %d); tie-break: thread 0 resolves first\n", level + 1);
    int cas_discovered = 0;
    // thread 0 (vertex 1): neighbor 2 (CAS fails, already visited)
    { int old = dist_cas[2]; if (old == UNVISITED) { dist_cas[2] = level + 1; cas_discovered++; } }
    // thread 0's CAS on vertex 3 resolves FIRST (the fixed tie-break) and succeeds
    { int old = dist_cas[3]; if (old == UNVISITED) { dist_cas[3] = level + 1; cas_discovered++; printf("  thread 0's atomicCAS on dist[3] SUCCEEDS (was UNVISITED) -- claims the discovery\n"); } }
    // thread 1's CAS on vertex 3 resolves SECOND -- the slot has already changed
    { int old = dist_cas[3]; if (old == UNVISITED) { dist_cas[3] = level + 1; cas_discovered++; }
      else { printf("  thread 1's atomicCAS on dist[3] FAILS (already %d) -- does NOT count a discovery\n", old); } }
    // thread 1's other neighbor, 4 -- uncontended, CAS succeeds
    { int old = dist_cas[4]; if (old == UNVISITED) { dist_cas[4] = level + 1; cas_discovered++; printf("  thread 1's atomicCAS on dist[4] SUCCEEDS (was UNVISITED) -- claims the discovery\n"); } }

    printf("  dist after CAS-protected expansion: [ ");
    for (int d : dist_cas) printf("%d ", d);
    printf("]\n  discovered_count = %d (exactly matching the 2 distinct newly-discovered vertices)\n\n", cas_discovered);

    std::vector<int> expected_dist = {0, 1, 1, 2, 2, UNVISITED};
    bool ok = (dist_naive == expected_dist) && (dist_cas == expected_dist);
    ok = ok && (naive_discovered == 3) && (cas_discovered == 2);

    printf("expected dist after either version: [ 0 1 1 2 2 -1 ]\n");
    printf("(vertex 5 is not yet reachable until level 2's frontier expands)\n");
    printf("expected discovered_count: naive = 3 (one duplicate), CAS = 2 (exact)\n");

    printf("\nself-check: both versions leave `dist` correct (the shared value being\n");
    printf("written was identical either way), but only atomicCAS gives an exact,\n");
    printf("duplicate-free discovery count: %s\n", ok ? "confirmed" : "MISMATCH");
    return ok ? 0 : 1;
}
```

**Compile and run:**

```bash
NVDIR=/usr/local/lib/python3.11/dist-packages/nvidia/cu13
$NVDIR/bin/nvcc -arch=sm_80 -I$NVDIR/include -L$NVDIR/lib -lcudart 119_bfs_frontier_expand_kernel.cu -o 119_bfs_frontier_expand_kernel
LD_LIBRARY_PATH=$NVDIR/lib ./119_bfs_frontier_expand_kernel
```

**Sample input:** level 1's frontier `{1, 2}`, expanded by 2 threads under a worst-case interleaving, comparing an unprotected naive kernel against an `atomicCAS`-protected one.

**Sample output:**

```text
=== Section 22.1 main: parallel frontier expansion, naive vs atomicCAS ===

expanding level 1's frontier {1, 2} (2 threads, one per frontier vertex):
  thread 0 (vertex 1): neighbors 2, 3
  thread 1 (vertex 2): neighbors 3, 4
  vertex 3 is a SHARED unvisited neighbor of both -- a genuine race

--- naive (unprotected check-then-write) ---
wave: BOTH threads READ dist[3] BEFORE either WRITES (worst case)
  thread 0 reads dist[3]=-1 -- believes it can claim vertex 3
  thread 1 reads dist[3]=-1 -- believes it can claim vertex 3
  thread 0 writes dist[3]=2, counts a discovery
  thread 1 ALSO writes dist[3]=2, ALSO counts a discovery -- DUPLICATE
  thread 1 writes dist[4]=2, counts a discovery
  dist after naive expansion: [ 0 1 1 2 2 -1 ]
  discovered_count = 3 (but only 2 DISTINCT vertices -- 3 and 4 -- were actually newly discovered)

--- atomicCAS-protected ---
wave: both threads attempt atomicCAS(&dist[3], UNVISITED, 2); tie-break: thread 0 resolves first
  thread 0's atomicCAS on dist[3] SUCCEEDS (was UNVISITED) -- claims the discovery
  thread 1's atomicCAS on dist[3] FAILS (already 2) -- does NOT count a discovery
  thread 1's atomicCAS on dist[4] SUCCEEDS (was UNVISITED) -- claims the discovery
  dist after CAS-protected expansion: [ 0 1 1 2 2 -1 ]
  discovered_count = 2 (exactly matching the 2 distinct newly-discovered vertices)

expected dist after either version: [ 0 1 1 2 2 -1 ]
(vertex 5 is not yet reachable until level 2's frontier expands)
expected discovered_count: naive = 3 (one duplicate), CAS = 2 (exact)

self-check: both versions leave `dist` correct (the shared value being
written was identical either way), but only atomicCAS gives an exact,
duplicate-free discovery count: confirmed
```

## 22.2 Compacting the Next Frontier: Stream Compaction Revisited

### Intuition

Winning the `atomicCAS` race tells a thread it is the FIRST to discover a vertex -- exactly the signal needed to decide which vertices belong in the next level's frontier array. Every winning thread claims a slot in `next_frontier` via `atomicAdd` on a shared counter, packing the newly-discovered vertices into a fresh, gap-free array: this is Chapter 6's stream compaction, with the "keep this element" test now being "did my atomicCAS just succeed" instead of a simple predicate on a value.

### The Sequential (CPU) Baseline

```cpp
#include <cstdio>
#include <vector>

// Chapter 22.2 -- The Sequential (CPU) Baseline.
// Section 22.1's FIFO queue and this section's explicit PER-LEVEL
// frontier arrays compute the identical BFS -- but the array-based
// version makes each level's boundary an explicit, separate array
// rather than an implicit run of positions inside one shared queue.
// That separation is exactly what the parallel version will need: a
// kernel launch operates on ONE frontier array at a time, producing
// the NEXT level's frontier array as its output, the same shape as
// Chapter 6's stream compaction (keep only the newly-discovered
// vertices, packed into a fresh array with no gaps).
#define V 6
#define UNVISITED (-1)

int main() {
    printf("=== Section 22.2 CPU baseline: BFS via explicit per-level frontier arrays ===\n\n");

    std::vector<int> row_offsets = {0, 2, 4, 6, 8, 9, 9};
    std::vector<int> col_idx     = {1, 2, 2, 3, 3, 4, 4, 5, 5};
    std::vector<int> dist(V, UNVISITED);

    int source = 0;
    dist[source] = 0;
    std::vector<int> frontier = {source};
    int level = 0;

    while (!frontier.empty()) {
        printf("level %d: frontier = { ", level);
        for (int v : frontier) printf("%d ", v);
        printf("}\n");

        std::vector<int> next_frontier;
        for (int u : frontier) {
            printf("  expand(%d):", u);
            for (int i = row_offsets[u]; i < row_offsets[u + 1]; i++) {
                int w = col_idx[i];
                if (dist[w] == UNVISITED) {
                    dist[w] = level + 1;
                    next_frontier.push_back(w);
                    printf(" %d(discovered, appended to next_frontier)", w);
                } else {
                    printf(" %d(already visited, skipped)", w);
                }
            }
            printf("\n");
        }
        printf("  next_frontier = { ");
        for (int v : next_frontier) printf("%d ", v);
        printf("}\n\n");

        frontier = next_frontier;
        level++;
    }

    printf("BFS terminates: level %d's frontier is empty\n\n", level);
    printf("final dist array: [ ");
    for (int d : dist) printf("%d ", d);
    printf("]\n");

    std::vector<int> expected_dist = {0, 1, 1, 2, 2, 3};
    bool ok = (dist == expected_dist) && (level == 4);

    printf("\nexpected dist: [ 0 1 1 2 2 3 ], terminating after level 4's empty frontier\n");
    printf("\nself-check: level-by-level array construction reproduces Section 22.1's\n");
    printf("exact distances: %s\n", ok ? "confirmed" : "MISMATCH");
    return ok ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++17 -Wall -Wextra -O2 120_bfs_frontier_construction_cpu_baseline.cpp -o 120_bfs_frontier_construction_cpu_baseline
./120_bfs_frontier_construction_cpu_baseline
```

**Sample input:** the same 6-vertex graph, BFS'd via explicit per-level frontier arrays instead of one shared FIFO queue, building each level's `next_frontier` by appending newly-discovered vertices.

**Sample output:**

```text
=== Section 22.2 CPU baseline: BFS via explicit per-level frontier arrays ===

level 0: frontier = { 0 }
  expand(0): 1(discovered, appended to next_frontier) 2(discovered, appended to next_frontier)
  next_frontier = { 1 2 }

level 1: frontier = { 1 2 }
  expand(1): 2(already visited, skipped) 3(discovered, appended to next_frontier)
  expand(2): 3(already visited, skipped) 4(discovered, appended to next_frontier)
  next_frontier = { 3 4 }

level 2: frontier = { 3 4 }
  expand(3): 4(already visited, skipped) 5(discovered, appended to next_frontier)
  expand(4): 5(already visited, skipped)
  next_frontier = { 5 }

level 3: frontier = { 5 }
  expand(5):
  next_frontier = { }

BFS terminates: level 4's frontier is empty

final dist array: [ 0 1 1 2 2 3 ]

expected dist: [ 0 1 1 2 2 3 ], terminating after level 4's empty frontier

self-check: level-by-level array construction reproduces Section 22.1's
exact distances: confirmed
```

### The Concept, In Detail

Running level 1's compaction under two different edge-level interleavings -- which of the two threads' `atomicCAS`+`atomicAdd` pair on vertex 3 resolves first:

```
ASCII view: interleaving A vs interleaving B, same race, different winner.

  interleaving A: t0 claims vertex 3 first -> next_frontier = [ 3, 4 ]
  interleaving B: t1 claims vertex 3 first (while also claiming 4) -> next_frontier = [ 4, 3 ]

  Both arrays hold the exact same SET, {3, 4} -- just in different
  positions, because whichever thread's atomicAdd resolved first
  claimed position 0.
```

This is the same story Chapter 21.2 told for CSR's own concurrent scatter, now one level up: a shared counter's `atomicAdd` gives every winning thread a unique, but not otherwise predictable, output position. The next level's kernel launch will read `next_frontier` and process whichever vertex sits at each position -- and since a frontier's vertices are processed as an unordered set regardless of index, the exact array order this level produces has no bearing on the correctness of the level after it.

[COMMON TRAP]
It is tempting to think a specific run's `next_frontier` array should be reproducible byte-for-byte across different executions, and to treat a differently-ordered (but equally correct) array as a bug. As with every other compaction or scatter this book has built under concurrency, the contract is about the SET of entries a frontier array holds, not their positions. Comparing frontier arrays between runs should sort both sides first, exactly as this section's own self-check does implicitly by construction.

### Code and Verification

```cpp
#include <cstdio>
#include <vector>
#include <algorithm>

// Chapter 22.2 main -- one kernel launch per level, expanding the ENTIRE
// current frontier at once: one thread per frontier vertex walks its
// own CSR slice, uses `atomicCAS` to claim each unvisited neighbor
// (Section 22.1's fix), and -- new in this section -- the WINNING
// thread also claims a slot in the compacted `next_frontier` array via
// `atomicAdd` on a shared counter. This is exactly Chapter 6's stream
// compaction: only entries that pass a test (here, "I am the thread
// that discovered this vertex") get packed into the output array, with
// no gaps and no wasted slots.
#define V 6
#define UNVISITED (-1)

__global__ void frontier_expand_compact_kernel(const int* row_offsets, const int* col_idx,
                                                int* dist, const int* frontier, int frontier_size,
                                                int level, int* next_frontier, int* next_count) {
    int t = threadIdx.x;
    if (t >= frontier_size) return;
    int u = frontier[t];
    for (int i = row_offsets[u]; i < row_offsets[u + 1]; i++) {
        int w = col_idx[i];
        int old = atomicCAS(&dist[w], UNVISITED, level + 1);
        if (old == UNVISITED) {
            int pos = atomicAdd(next_count, 1);
            next_frontier[pos] = w;
        }
    }
}

// ---- Host-side replay of the identical per-thread logic. ----

int main() {
    printf("=== Section 22.2 main: parallel frontier expansion with compaction ===\n\n");

    std::vector<int> row_offsets = {0, 2, 4, 6, 8, 9, 9};
    std::vector<int> col_idx     = {1, 2, 2, 3, 3, 4, 4, 5, 5};
    std::vector<int> dist(V, UNVISITED);
    dist[0] = 0;
    std::vector<int> frontier = {0};
    int level = 0;

    printf("--- full BFS, one kernel launch per level (natural thread order) ---\n");
    while (!frontier.empty()) {
        std::vector<int> next_frontier;
        for (int t = 0; t < (int)frontier.size(); t++) {
            int u = frontier[t];
            for (int i = row_offsets[u]; i < row_offsets[u + 1]; i++) {
                int w = col_idx[i];
                int old = dist[w];
                if (old == UNVISITED) dist[w] = level + 1;
                if (old == UNVISITED) next_frontier.push_back(w);
            }
        }
        printf("  level %d: frontier size %zu -> next_frontier = { ", level, frontier.size());
        for (int v : next_frontier) printf("%d ", v);
        printf("}\n");
        frontier = next_frontier;
        level++;
    }
    printf("  final dist: [ ");
    for (int d : dist) printf("%d ", d);
    printf("]\n\n");

    std::vector<int> expected_dist = {0, 1, 1, 2, 2, 3};
    bool ok = (dist == expected_dist) && (level == 4);

    printf("--- level 1's compaction under two DIFFERENT edge-level interleavings ---\n");
    printf("frontier {1, 2}: thread 0 (vertex 1, neighbors 2,3), thread 1 (vertex 2, neighbors 3,4)\n\n");

    // Interleaving A: thread 0's edges resolve fully before thread 1's.
    {
        std::vector<int> d = {0, 1, 1, UNVISITED, UNVISITED, UNVISITED};
        std::vector<int> next_A;
        int counter = 0;
        printf("interleaving A: t0-nbr2, t0-nbr3, t1-nbr3, t1-nbr4\n");
        auto attempt = [&](int w, const char* who) {
            int old = d[w];
            if (old == UNVISITED) { d[w] = 2; int pos = counter++; next_A.resize(std::max((size_t)pos + 1, next_A.size())); next_A[pos] = w; printf("  %s: atomicCAS(dist[%d]) succeeds -> claims next_frontier[%d]=%d\n", who, w, pos, w); }
            else printf("  %s: atomicCAS(dist[%d]) fails (already %d)\n", who, w, old);
        };
        attempt(2, "t0-nbr2"); attempt(3, "t0-nbr3"); attempt(3, "t1-nbr3"); attempt(4, "t1-nbr4");
        printf("  next_frontier (order A) = [ ");
        for (int v : next_A) printf("%d ", v);
        printf("]\n\n");
        ok = ok && (next_A == std::vector<int>{3, 4});
    }

    // Interleaving B: thread 1's edges resolve fully before thread 0's.
    {
        std::vector<int> d = {0, 1, 1, UNVISITED, UNVISITED, UNVISITED};
        std::vector<int> next_B;
        int counter = 0;
        printf("interleaving B: t1-nbr4, t1-nbr3, t0-nbr2, t0-nbr3\n");
        auto attempt = [&](int w, const char* who) {
            int old = d[w];
            if (old == UNVISITED) { d[w] = 2; int pos = counter++; next_B.resize(std::max((size_t)pos + 1, next_B.size())); next_B[pos] = w; printf("  %s: atomicCAS(dist[%d]) succeeds -> claims next_frontier[%d]=%d\n", who, w, pos, w); }
            else printf("  %s: atomicCAS(dist[%d]) fails (already %d)\n", who, w, old);
        };
        attempt(4, "t1-nbr4"); attempt(3, "t1-nbr3"); attempt(2, "t0-nbr2"); attempt(3, "t0-nbr3");
        printf("  next_frontier (order B) = [ ");
        for (int v : next_B) printf("%d ", v);
        printf("]\n\n");
        ok = ok && (next_B == std::vector<int>{4, 3});
    }

    printf("expected: order A gives [3, 4], order B gives [4, 3] -- DIFFERENT exact\n");
    printf("array positions, but the SAME set {3, 4} either way\n");

    printf("\nself-check: full multi-level BFS matches the CPU baseline exactly, and\n");
    printf("level 1's compacted frontier holds the same set under both interleavings: %s\n", ok ? "confirmed" : "MISMATCH");
    return ok ? 0 : 1;
}
```

**Compile and run:**

```bash
NVDIR=/usr/local/lib/python3.11/dist-packages/nvidia/cu13
$NVDIR/bin/nvcc -arch=sm_80 -I$NVDIR/include -L$NVDIR/lib -lcudart 121_bfs_frontier_compact_kernel.cu -o 121_bfs_frontier_compact_kernel
LD_LIBRARY_PATH=$NVDIR/lib ./121_bfs_frontier_compact_kernel
```

**Sample input:** a full 4-level BFS run via repeated per-level kernel launches, plus level 1's compaction shown again under two different edge-level interleavings.

**Sample output:**

```text
=== Section 22.2 main: parallel frontier expansion with compaction ===

--- full BFS, one kernel launch per level (natural thread order) ---
  level 0: frontier size 1 -> next_frontier = { 1 2 }
  level 1: frontier size 2 -> next_frontier = { 3 4 }
  level 2: frontier size 2 -> next_frontier = { 5 }
  level 3: frontier size 1 -> next_frontier = { }
  final dist: [ 0 1 1 2 2 3 ]

--- level 1's compaction under two DIFFERENT edge-level interleavings ---
frontier {1, 2}: thread 0 (vertex 1, neighbors 2,3), thread 1 (vertex 2, neighbors 3,4)

interleaving A: t0-nbr2, t0-nbr3, t1-nbr3, t1-nbr4
  t0-nbr2: atomicCAS(dist[2]) fails (already 1)
  t0-nbr3: atomicCAS(dist[3]) succeeds -> claims next_frontier[0]=3
  t1-nbr3: atomicCAS(dist[3]) fails (already 2)
  t1-nbr4: atomicCAS(dist[4]) succeeds -> claims next_frontier[1]=4
  next_frontier (order A) = [ 3 4 ]

interleaving B: t1-nbr4, t1-nbr3, t0-nbr2, t0-nbr3
  t1-nbr4: atomicCAS(dist[4]) succeeds -> claims next_frontier[0]=4
  t1-nbr3: atomicCAS(dist[3]) succeeds -> claims next_frontier[1]=3
  t0-nbr2: atomicCAS(dist[2]) fails (already 1)
  t0-nbr3: atomicCAS(dist[3]) fails (already 2)
  next_frontier (order B) = [ 4 3 ]

expected: order A gives [3, 4], order B gives [4, 3] -- DIFFERENT exact
array positions, but the SAME set {3, 4} either way

self-check: full multi-level BFS matches the CPU baseline exactly, and
level 1's compacted frontier holds the same set under both interleavings: confirmed
```

## 22.3 Vertex-Parallel vs. Edge-Parallel Frontier Expansion

### Intuition

One thread per frontier VERTEX, as Sections 22.1 and 22.2 used, forces uneven work onto threads whenever frontier vertices have different degrees -- exactly Chapter 21.3's CSR-vs-COO lesson, now applied to a single frontier's expansion instead of a whole-graph pass. The fix is the same: flatten the current frontier's incident edges into one list (via the frontier's own degree-count and prefix sum -- Chapter 21.2's recipe, applied to the frontier instead of the whole graph), then launch one thread per EDGE instead of one thread per vertex, giving every thread the identical, fixed amount of work regardless of any vertex's degree.

### The Sequential (CPU) Baseline

```cpp
#include <cstdio>
#include <vector>
#include <algorithm>

// Chapter 22.3 -- The Sequential (CPU) Baseline.
// Section 22.1/22.2's kernels are VERTEX-parallel: one thread per
// frontier vertex, looping over that vertex's own neighbors. Exactly
// like Chapter 21.3's warning about CSR, this forces UNEVEN work onto
// threads whenever frontier vertices have different degrees -- the
// thread assigned to a high-degree vertex does far more work than the
// thread assigned to a low-degree one, and on this book's tiny example
// the effect is modest, but on a real graph with a few extremely
// high-degree "hub" vertices, a single thread can end up doing
// thousands of times more work than its neighbors in the same launch.
// The fix, mirroring Chapter 21.3's own COO alternative, is EDGE-
// parallel expansion: flatten the current frontier's incident edges
// into one list first, then launch one thread per EDGE instead of one
// thread per vertex.
#define V 6
#define UNVISITED (-1)

int main() {
    printf("=== Section 22.3 CPU baseline: vertex-parallel vs edge-parallel work ===\n\n");

    std::vector<int> row_offsets = {0, 2, 4, 6, 8, 9, 9};
    std::vector<int> col_idx     = {1, 2, 2, 3, 3, 4, 4, 5, 5};

    // Reconstruct the same 4 frontiers Section 22.1/22.2 already built.
    std::vector<std::vector<int>> frontiers = {{0}, {1, 2}, {3, 4}, {5}};

    printf("vertex-parallel (1 thread/vertex) vs edge-parallel (1 thread/edge) work, per level:\n");
    for (size_t level = 0; level < frontiers.size(); level++) {
        std::vector<int> degrees;
        for (int v : frontiers[level]) degrees.push_back(row_offsets[v + 1] - row_offsets[v]);
        int max_deg = 0, total = 0;
        for (int d : degrees) { max_deg = std::max(max_deg, d); total += d; }
        printf("  level %zu: frontier={ ", level);
        for (int v : frontiers[level]) printf("%d ", v);
        printf("} degrees={ ");
        for (int d : degrees) printf("%d ", d);
        printf("} vertex-parallel slowest-thread=%d edges  edge-parallel threads=%d (each does exactly 1 edge)\n",
               max_deg, total);
    }

    printf("\nbuilding edge-parallel offsets for level 2's frontier {3, 4}\n");
    printf("(exactly Chapter 21.2's degree-count + prefix-sum recipe, applied to\n");
    printf("the FRONTIER's own vertices instead of the whole graph's):\n\n");

    std::vector<int> frontier2 = {3, 4};
    std::vector<int> frontier_degree;
    for (int v : frontier2) frontier_degree.push_back(row_offsets[v + 1] - row_offsets[v]);
    printf("  frontier_degree = [ ");
    for (int d : frontier_degree) printf("%d ", d);
    printf("]  (vertex 3 has degree %d, vertex 4 has degree %d)\n", frontier_degree[0], frontier_degree[1]);

    std::vector<int> frontier_offsets(frontier2.size() + 1, 0);
    int running = 0;
    for (size_t i = 0; i < frontier2.size(); i++) {
        frontier_offsets[i] = running;
        running += frontier_degree[i];
    }
    frontier_offsets[frontier2.size()] = running;
    printf("  frontier_offsets = [ ");
    for (int o : frontier_offsets) printf("%d ", o);
    printf("]  (exclusive prefix sum over frontier_degree)\n\n");

    int total_edges = frontier_offsets.back();
    printf("  %d edge-threads, each mapped back to its own frontier vertex:\n", total_edges);
    std::vector<std::pair<int,int>> edge_assignments;  // (src, dst)
    for (int e = 0; e < total_edges; e++) {
        int fi = 0;
        while (frontier_offsets[fi + 1] <= e) fi++;
        int local = e - frontier_offsets[fi];
        int src = frontier2[fi];
        int csr_index = row_offsets[src] + local;
        int dst = col_idx[csr_index];
        edge_assignments.push_back({src, dst});
        printf("    edge-thread %d: frontier_vertex_index=%d (src=%d), local_edge=%d, csr_index=%d, dst=%d\n",
               e, fi, src, local, csr_index, dst);
    }

    std::vector<std::pair<int,int>> expected_edges = {{3,4}, {3,5}, {4,5}};
    bool ok = (edge_assignments == expected_edges);
    ok = ok && (frontier_offsets == std::vector<int>{0, 2, 3});

    printf("\nexpected edge assignments: (3->4) (3->5) (4->5)\n");
    printf("(dst=5 is shared by BOTH edge-threads 1 and 2, from DIFFERENT source\n");
    printf("vertices -- the identical race Section 22.1 demonstrated, now surfacing\n");
    printf("at the EDGE level instead of the vertex level)\n");

    printf("\nself-check: frontier offsets and edge-to-vertex mapping match expected: %s\n", ok ? "confirmed" : "MISMATCH");
    return ok ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++17 -Wall -Wextra -O2 122_bfs_load_balance_cpu_baseline.cpp -o 122_bfs_load_balance_cpu_baseline
./122_bfs_load_balance_cpu_baseline
```

**Sample input:** all 4 of the graph's BFS levels, comparing vertex-parallel worst-case thread work against edge-parallel uniform work, then building the edge-parallel offsets for level 2's frontier `{3, 4}`.

**Sample output:**

```text
=== Section 22.3 CPU baseline: vertex-parallel vs edge-parallel work ===

vertex-parallel (1 thread/vertex) vs edge-parallel (1 thread/edge) work, per level:
  level 0: frontier={ 0 } degrees={ 2 } vertex-parallel slowest-thread=2 edges  edge-parallel threads=2 (each does exactly 1 edge)
  level 1: frontier={ 1 2 } degrees={ 2 2 } vertex-parallel slowest-thread=2 edges  edge-parallel threads=4 (each does exactly 1 edge)
  level 2: frontier={ 3 4 } degrees={ 2 1 } vertex-parallel slowest-thread=2 edges  edge-parallel threads=3 (each does exactly 1 edge)
  level 3: frontier={ 5 } degrees={ 0 } vertex-parallel slowest-thread=0 edges  edge-parallel threads=0 (each does exactly 1 edge)

building edge-parallel offsets for level 2's frontier {3, 4}
(exactly Chapter 21.2's degree-count + prefix-sum recipe, applied to
the FRONTIER's own vertices instead of the whole graph's):

  frontier_degree = [ 2 1 ]  (vertex 3 has degree 2, vertex 4 has degree 1)
  frontier_offsets = [ 0 2 3 ]  (exclusive prefix sum over frontier_degree)

  3 edge-threads, each mapped back to its own frontier vertex:
    edge-thread 0: frontier_vertex_index=0 (src=3), local_edge=0, csr_index=6, dst=4
    edge-thread 1: frontier_vertex_index=0 (src=3), local_edge=1, csr_index=7, dst=5
    edge-thread 2: frontier_vertex_index=1 (src=4), local_edge=0, csr_index=8, dst=5

expected edge assignments: (3->4) (3->5) (4->5)
(dst=5 is shared by BOTH edge-threads 1 and 2, from DIFFERENT source
vertices -- the identical race Section 22.1 demonstrated, now surfacing
at the EDGE level instead of the vertex level)

self-check: frontier offsets and edge-to-vertex mapping match expected: confirmed
```

### The Concept, In Detail

```
ASCII view: vertex-parallel vs edge-parallel expansion of frontier {3, 4}.

  vertex-parallel (2 threads):
    thread 0 (vertex 3): 2 edges -- checks dst=4, dst=5
    thread 1 (vertex 4): 1 edge  -- checks dst=5
    thread 0 does 2x thread 1's work

  edge-parallel (3 threads, via frontier_offsets = [0, 2, 3]):
    thread 0: edge 0 -> vertex 3's local edge 0 -> dst=4
    thread 1: edge 1 -> vertex 3's local edge 1 -> dst=5
    thread 2: edge 2 -> vertex 4's local edge 0 -> dst=5
    every thread does EXACTLY 1 edge -- no thread does more than any other
```

This tiny graph's imbalance (2 edges versus 1) is barely worth correcting on its own. Real graphs are typically far more skewed -- a handful of extremely high-degree "hub" vertices (a popular social-media account, a widely-linked web page) can sit in the SAME frontier as thousands of low-degree vertices, and a vertex-parallel kernel would leave nearly all of its threads idle while a tiny few churn through enormous slices. Edge-parallel expansion eliminates this imbalance at the frontier level exactly as Chapter 21.3 eliminated it at the whole-graph level, at the cost of one extra indirection per thread: mapping an edge index back to the frontier vertex it belongs to.

[COMMON TRAP]
It is tempting to think edge-parallel expansion is strictly better and should always be used. Building the frontier's own `frontier_offsets` array costs a degree-count-plus-prefix-sum pass over the CURRENT frontier before expansion can even begin -- overhead that vertex-parallel expansion does not pay. For a frontier that is small or has roughly uniform degrees (this book's own tiny example, or many real graphs' early BFS levels), that extra pass can cost more than the imbalance it fixes. The right choice depends on the actual degree distribution of the frontier at hand, not a fixed rule.

### Code and Verification

```cpp
#include <cstdio>
#include <vector>
#include <algorithm>
#include <utility>

// Chapter 22.3 main -- edge-parallel frontier expansion: one thread per
// INCIDENT EDGE of the current frontier (built via the frontier's own
// degree-count + prefix-sum, Chapter 21.2's recipe applied to the
// frontier instead of the whole graph), rather than one thread per
// frontier VERTEX. Each thread first locates which frontier vertex it
// belongs to via `frontier_offsets` (a linear search here, since a
// single frontier is tiny; a real implementation would binary-search
// or precompute this), then performs the IDENTICAL atomicCAS-claim +
// atomicAdd-compaction logic Section 22.2's kernel already used --
// only the granularity of what "one thread" means has changed, not
// the correctness mechanism itself.
#define UNVISITED (-1)

// Finds which frontier vertex a given edge index `e` belongs to.
__device__ int find_frontier_vertex(const int* frontier_offsets, int frontier_size, int e) {
    int fi = 0;
    while (fi < frontier_size - 1 && frontier_offsets[fi + 1] <= e) fi++;
    return fi;
}

__global__ void edge_parallel_expand_kernel(const int* row_offsets, const int* col_idx,
                                             const int* frontier, const int* frontier_offsets,
                                             int frontier_size, int* dist, int level,
                                             int* next_frontier, int* next_count) {
    int e = threadIdx.x;
    int total_edges = frontier_offsets[frontier_size];
    if (e >= total_edges) return;
    int fi = find_frontier_vertex(frontier_offsets, frontier_size, e);
    int src = frontier[fi];
    int local = e - frontier_offsets[fi];
    int csr_index = row_offsets[src] + local;
    int dst = col_idx[csr_index];
    int old = atomicCAS(&dist[dst], UNVISITED, level + 1);
    if (old == UNVISITED) {
        int pos = atomicAdd(next_count, 1);
        next_frontier[pos] = dst;
    }
}

// ---- Host-side replay of the identical per-thread logic. ----

int main() {
    printf("=== Section 22.3 main: edge-parallel frontier expansion ===\n\n");

    std::vector<int> row_offsets = {0, 2, 4, 6, 8, 9, 9};
    std::vector<int> col_idx     = {1, 2, 2, 3, 3, 4, 4, 5, 5};
    std::vector<int> frontier = {3, 4};
    std::vector<int> frontier_offsets = {0, 2, 3};
    int level = 2;

    printf("expanding level %d's frontier {3, 4} via 3 edge-threads:\n", level);
    printf("  edge-thread 0: src=3, dst=4   edge-thread 1: src=3, dst=5   edge-thread 2: src=4, dst=5\n\n");

    // Pre-race state: vertices 0,1,2 (level<=1) and 4 (discovered during
    // level 1's own expansion of vertex 2) already have distances.
    std::vector<int> dist_pre = {0, 1, 1, 2, 2, UNVISITED};

    auto run_order = [&](const std::vector<int>& edge_order, const char* label) {
        std::vector<int> dist = dist_pre;
        std::vector<int> next_frontier;
        int counter = 0;
        printf("%s: edge order %d, %d, %d\n", label, edge_order[0], edge_order[1], edge_order[2]);
        for (int e : edge_order) {
            int fi = 0;
            while (fi < (int)frontier.size() - 1 && frontier_offsets[fi + 1] <= e) fi++;
            int src = frontier[fi];
            int local = e - frontier_offsets[fi];
            int csr_index = row_offsets[src] + local;
            int dst = col_idx[csr_index];
            int old = dist[dst];
            if (old == UNVISITED) {
                dist[dst] = level + 1;
                int pos = counter++;
                next_frontier.resize(std::max((size_t)pos + 1, next_frontier.size()));
                next_frontier[pos] = dst;
                printf("  edge-thread %d (src=%d,dst=%d): atomicCAS succeeds -> claims next_frontier[%d]=%d\n",
                       e, src, dst, pos, dst);
            } else {
                printf("  edge-thread %d (src=%d,dst=%d): atomicCAS fails (dist[%d] already %d)\n",
                       e, src, dst, dst, old);
            }
        }
        printf("  next_frontier = [ ");
        for (int v : next_frontier) printf("%d ", v);
        printf("]  dist = [ ");
        for (int d : dist) printf("%d ", d);
        printf("]\n\n");
        return std::make_pair(dist, next_frontier);
    };

    auto result_A = run_order({0, 1, 2}, "order A");
    auto result_B = run_order({2, 1, 0}, "order B");

    std::vector<int> expected_dist = {0, 1, 1, 2, 2, 3};
    std::vector<int> expected_next = {5};
    bool ok = (result_A.first == expected_dist) && (result_A.second == expected_next);
    ok = ok && (result_B.first == expected_dist) && (result_B.second == expected_next);

    printf("expected: both orders leave dist = [ 0 1 1 2 2 3 ] and next_frontier = [ 5 ]\n");
    printf("(edge-thread for src=3,dst=4 is ALWAYS a no-op -- vertex 4 was already\n");
    printf("discovered during level 1's own expansion -- and exactly one of the two\n");
    printf("threads racing for dst=5 wins, regardless of which order they run in)\n");

    printf("\nself-check: edge-parallel expansion matches the vertex-parallel result\n");
    printf("exactly under both edge-processing orders: %s\n", ok ? "confirmed" : "MISMATCH");
    return ok ? 0 : 1;
}
```

**Compile and run:**

```bash
NVDIR=/usr/local/lib/python3.11/dist-packages/nvidia/cu13
$NVDIR/bin/nvcc -arch=sm_80 -I$NVDIR/include -L$NVDIR/lib -lcudart 123_bfs_edge_parallel_expand_kernel.cu -o 123_bfs_edge_parallel_expand_kernel
LD_LIBRARY_PATH=$NVDIR/lib ./123_bfs_edge_parallel_expand_kernel
```

**Sample input:** level 2's frontier `{3, 4}`, expanded by 3 edge-threads under two different edge-processing orders.

**Sample output:**

```text
=== Section 22.3 main: edge-parallel frontier expansion ===

expanding level 2's frontier {3, 4} via 3 edge-threads:
  edge-thread 0: src=3, dst=4   edge-thread 1: src=3, dst=5   edge-thread 2: src=4, dst=5

order A: edge order 0, 1, 2
  edge-thread 0 (src=3,dst=4): atomicCAS fails (dist[4] already 2)
  edge-thread 1 (src=3,dst=5): atomicCAS succeeds -> claims next_frontier[0]=5
  edge-thread 2 (src=4,dst=5): atomicCAS fails (dist[5] already 3)
  next_frontier = [ 5 ]  dist = [ 0 1 1 2 2 3 ]

order B: edge order 2, 1, 0
  edge-thread 2 (src=4,dst=5): atomicCAS succeeds -> claims next_frontier[0]=5
  edge-thread 1 (src=3,dst=5): atomicCAS fails (dist[5] already 3)
  edge-thread 0 (src=3,dst=4): atomicCAS fails (dist[4] already 2)
  next_frontier = [ 5 ]  dist = [ 0 1 1 2 2 3 ]

expected: both orders leave dist = [ 0 1 1 2 2 3 ] and next_frontier = [ 5 ]
(edge-thread for src=3,dst=4 is ALWAYS a no-op -- vertex 4 was already
discovered during level 1's own expansion -- and exactly one of the two
threads racing for dst=5 wins, regardless of which order they run in)

self-check: edge-parallel expansion matches the vertex-parallel result
exactly under both edge-processing orders: confirmed
```

## Chapter Summary

Reframing BFS around the FRONTIER -- the entire set of vertices at the current distance -- turns a sequential, one-vertex-at-a-time FIFO walk into a genuinely parallel operation: expand every frontier vertex's CSR neighbors at once. Two different frontier vertices can share a common unvisited neighbor, creating a real race to be the one that discovers it; `atomicCAS` on the distance array resolves this exactly as it resolved Chapter 16.2's trie race, with the twist that the value being written is often identical regardless of who wins, making the race's real danger a DUPLICATE discovery rather than a corrupted value. The winning thread of each race claims a slot in the next frontier via `atomicAdd`, exactly Chapter 6's stream compaction with a new predicate; as with every other concurrent scatter in this book, the resulting array's exact order can vary between runs while the SET of newly-discovered vertices stays identical. Vertex-parallel expansion (one thread per frontier vertex) forces uneven work whenever frontier vertices have different degrees, echoing Chapter 21.3's CSR-versus-COO lesson; edge-parallel expansion (one thread per incident edge, located via the frontier's own degree-count-and-prefix-sum) gives every thread uniform work at the cost of one extra layer of indirection.

## Self-Check Questions

1. Why can a naive, unprotected race between two threads discovering the same vertex leave the distance array itself correct, and yet still be a real bug?
2. What does `atomicCAS` return, and how does a thread use that return value to know whether it was the one that discovered a vertex?
3. How is claiming a slot in `next_frontier` via `atomicAdd` an instance of Chapter 6's stream compaction?
4. Two different runs of the same parallel frontier expansion can produce `next_frontier` arrays with the same vertices in different orders. Why does this not make either run incorrect?
5. Why does vertex-parallel frontier expansion risk uneven per-thread work, and how does edge-parallel expansion avoid it?
6. What extra cost does edge-parallel expansion pay that vertex-parallel expansion does not, and when might that cost not be worth paying?

## Where We Go Next

BFS answers "how many edges away" from a source, treating every edge as equally costly. Chapter 23 turns to single-source shortest paths, where edges carry different weights and the frontier model itself is no longer enough -- a vertex reached later, via a longer PATH but a smaller total WEIGHT, can still be the correct answer, requiring a different discovery rule than "first thread to arrive wins."

## Worked Solutions

**1.** If two threads race to discover the same vertex, both would compute and write the SAME distance value (their shared neighbor is at the same level from both frontier vertices), so the distance array ends up holding the correct value either way. The bug is elsewhere: if both threads separately conclude "I discovered this vertex," both would (Section 22.2) attempt to add it to the next frontier, producing a duplicate entry that wastes work at the next level and, in a more general algorithm that counts or processes each vertex once, would silently double-count it.

**2.** `atomicCAS(&slot, expected, new_value)` atomically checks whether `slot` currently equals `expected`; if so, it writes `new_value` and returns the OLD value (`expected`), and if not, it leaves `slot` unchanged and returns whatever `slot` actually held. A thread checks whether the returned old value equals `UNVISITED`: if it does, this thread's CAS was the one that actually performed the transition, so it was the discoverer; if the returned value is anything else, some other thread already changed the slot first.

**3.** Stream compaction keeps only the elements that pass some test, packing them into a fresh array with no gaps, using an `atomicAdd` on a shared counter to give each kept element its own unique output position. Frontier construction applies exactly this: the "test" a vertex passes is "did my thread's atomicCAS report that I was the one who discovered it," and only vertices passing that test claim a slot (via the identical atomicAdd-for-position pattern) in the `next_frontier` array.

**4.** Which thread's `atomicAdd` on the shared position counter resolves first determines only the INDEX a given vertex lands at within `next_frontier` -- it has no bearing on which vertices end up in the array at all, since that is decided entirely by which threads' `atomicCAS` calls succeeded, a fact that does not depend on the counter's resolution order. A frontier array's correctness is defined by the SET of vertices it contains, to be processed (in Section 22.3's case, per-vertex or per-edge) as an unordered group at the next level, so a different internal ordering carries no consequence.

**5.** Vertex-parallel expansion assigns one thread per frontier vertex, and that thread's workload is exactly that vertex's degree (its own CSR slice length) -- if frontier vertices have different degrees, some threads finish quickly while others are still working, wasting the idle threads' capacity. Edge-parallel expansion instead assigns one thread per INDIVIDUAL EDGE incident to the frontier, computed via the frontier's own degree-count-and-prefix-sum; because every thread now handles exactly one edge, no thread's workload depends on any vertex's degree, so no thread can ever be left doing more work than any other.

**6.** Edge-parallel expansion must first build `frontier_offsets` -- a degree count and prefix sum over the CURRENT frontier's own vertices -- before expansion can begin, and each edge-thread pays one extra indirection (locating which frontier vertex its edge belongs to) that a vertex-parallel thread never needs. When a frontier is small or has roughly uniform degrees across its vertices, this extra bookkeeping can cost more than the load imbalance it is meant to prevent, making vertex-parallel expansion the better choice in that case.
