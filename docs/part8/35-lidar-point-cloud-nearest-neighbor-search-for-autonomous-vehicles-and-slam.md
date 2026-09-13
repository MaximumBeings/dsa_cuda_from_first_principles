# Chapter 35: LiDAR Point-Cloud Nearest-Neighbor Search for Autonomous Vehicles and SLAM

An autonomous vehicle's LiDAR sensor produces a fresh cloud of thousands to millions of 3D points many times per second, and almost everything downstream -- obstacle detection, free-space estimation, and SLAM (Simultaneous Localization and Mapping) -- depends on answering one question over and over: for this point, which other points are nearby? Scanning every point against every other point does not scale, which is exactly why Chapter 18 built spatial trees in the first place. This chapter puts that structure to work on a genuine, if simplified, LiDAR pipeline: Section 35.1 partitions a point cloud into a quadtree, Section 35.2 uses it to prune a nearest-neighbor search from all points down to a handful, and Section 35.3 applies that same pruned search across an entire second scan to produce the point correspondences a real SLAM system uses to track how the vehicle moved.

## 35.1 Building a Quadtree Over a Point Cloud

### Intuition

A quadtree splits 2D space into four quadrants, recursively, until each region holds few enough points to search directly. Computing which quadrant a point falls into needs only 2 bits -- one for which half of x it is in, one for which half of y -- combined exactly the way Section 31.2's Morton code interleaved bits, just for a single level instead of many. Building the quadtree, then, is nothing more than computing this 2-bit code per point and appending each point to its quadrant's list, exactly Section 30.2/33.3's bucket-append pattern.

### The Sequential (CPU) Baseline

```cpp
// 196_quadtree_build_cpu_baseline.cpp
//
// Chapter 35.1 -- sequential baseline for partitioning a 2D LiDAR
// point cloud into a quadtree's four quadrants. This is Section
// 31.2's Morton-style bit code, now just ONE level deep: 1 bit for
// which half of x, 1 bit for which half of y, combined into a 2-bit
// quadrant code. Appending each point to its quadrant's list is
// Section 30.2/33.3's bucket append, run sequentially here.
//
// Compile: g++ -std=c++17 -Wall -Wextra -O2 196_quadtree_build_cpu_baseline.cpp -o 196_quadtree_build_cpu_baseline
// Run:     ./196_quadtree_build_cpu_baseline

#include <array>
#include <cstdio>
#include <vector>

constexpr int NUM_POINTS = 8;
constexpr double MID_X = 4.0, MID_Y = 4.0;

int quadrant_code(double x, double y) {
    return (x >= MID_X ? 1 : 0) + (y >= MID_Y ? 2 : 0);
}

int main() {
    double points[NUM_POINTS][2] = {
        {1, 1}, {2, 3}, {5, 1}, {6, 2}, {1, 5}, {3, 6}, {5, 5}, {6, 7},
    };

    std::printf("=== Section 35.1 CPU baseline: sequential quadtree build (one level, capacity 2) ===\n\n");
    std::printf("points: [");
    for (int i = 0; i < NUM_POINTS; ++i) {
        std::printf("(%d, (%.0f, %.0f))%s", i, points[i][0], points[i][1], (i + 1 < NUM_POINTS) ? ", " : "");
    }
    std::printf("]\n");
    std::printf("split at x=%.0f, y=%.0f -> quadrant code = (x>=mid_x?1:0) + (y>=mid_y?2:0)\n\n", MID_X, MID_Y);

    std::array<std::vector<int>, 4> quadrants;

    for (int i = 0; i < NUM_POINTS; ++i) {
        int q = quadrant_code(points[i][0], points[i][1]);
        quadrants[q].push_back(i);
        std::printf("point %d (%.0f,%.0f): quadrant code = %d -> appended to quadrant %d, now [", i,
                    points[i][0], points[i][1], q, q);
        for (size_t j = 0; j < quadrants[q].size(); ++j) {
            std::printf("%d%s", quadrants[q][j], (j + 1 < quadrants[q].size()) ? ", " : "");
        }
        std::printf("]\n");
    }

    std::printf("\nfinal quadrants:\n");
    const char* names[4] = {"SW (x<4,y<4)", "SE (x>=4,y<4)", "NW (x<4,y>=4)", "NE (x>=4,y>=4)"};
    for (int q = 0; q < 4; ++q) {
        std::printf("  quadrant %d %s: points [", q, names[q]);
        for (size_t j = 0; j < quadrants[q].size(); ++j) {
            std::printf("%d%s", quadrants[q][j], (j + 1 < quadrants[q].size()) ? ", " : "");
        }
        std::printf("]\n");
    }

    bool ok = true;
    for (int q = 0; q < 4; ++q) if (quadrants[q].size() != 2) ok = false;
    std::printf("\nself-check: exactly 2 points per quadrant (capacity satisfied, no further "
                "subdivision needed): %s\n", ok ? "confirmed" : "MISMATCH");

    return ok ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++17 -Wall -Wextra -O2 196_quadtree_build_cpu_baseline.cpp -o 196_quadtree_build_cpu_baseline
./196_quadtree_build_cpu_baseline
```

**Sample input:** 8 LiDAR points scattered across a small square scene, split at the midpoint of each axis into 4 quadrants, with exactly 2 points landing in each quadrant.

**Sample output:**

```text
=== Section 35.1 CPU baseline: sequential quadtree build (one level, capacity 2) ===

points: [(0, (1, 1)), (1, (2, 3)), (2, (5, 1)), (3, (6, 2)), (4, (1, 5)), (5, (3, 6)), (6, (5, 5)), (7, (6, 7))]
split at x=4, y=4 -> quadrant code = (x>=mid_x?1:0) + (y>=mid_y?2:0)

point 0 (1,1): quadrant code = 0 -> appended to quadrant 0, now [0]
point 1 (2,3): quadrant code = 0 -> appended to quadrant 0, now [0, 1]
point 2 (5,1): quadrant code = 1 -> appended to quadrant 1, now [2]
point 3 (6,2): quadrant code = 1 -> appended to quadrant 1, now [2, 3]
point 4 (1,5): quadrant code = 2 -> appended to quadrant 2, now [4]
point 5 (3,6): quadrant code = 2 -> appended to quadrant 2, now [4, 5]
point 6 (5,5): quadrant code = 3 -> appended to quadrant 3, now [6]
point 7 (6,7): quadrant code = 3 -> appended to quadrant 3, now [6, 7]

final quadrants:
  quadrant 0 SW (x<4,y<4): points [0, 1]
  quadrant 1 SE (x>=4,y<4): points [2, 3]
  quadrant 2 NW (x<4,y>=4): points [4, 5]
  quadrant 3 NE (x>=4,y>=4): points [6, 7]

self-check: exactly 2 points per quadrant (capacity satisfied, no further subdivision needed): confirmed
```

### The Concept, In Detail

```
ASCII view: a quadrant code is a 1-level Morton code.

  scene, split at x=4, y=4:

    y                 quadrant 2 (NW)  |  quadrant 3 (NE)
    ^                  code = 0+2=2    |   code = 1+2=3
    |                 -------------------------------------
    |                 quadrant 0 (SW)  |  quadrant 1 (SE)
    |                  code = 0+0=0    |   code = 1+0=1
    +----------> x

  code = (x >= mid_x ? 1 : 0) + (y >= mid_y ? 2 : 0)
       -- exactly Section 31.2's bit-interleave, for 1 level instead of k
```

A full quadtree keeps subdividing any quadrant that still holds more points than its capacity, level by level -- exactly Chapter 16's level-synchronous construction, one round per depth, with an explicit stack standing in for recursion at each round. This section's example needs only one round because its capacity (2 points per leaf) happens to already be satisfied everywhere after the first split.

[COMMON TRAP]
It is tempting to think this section's one-level example IS a complete quadtree implementation. A real quadtree must keep recursing into any quadrant that still exceeds capacity after a split, using the same level-synchronous, explicit-stack discipline Chapter 16 established for tries -- this section stops at one level purely because its chosen point count and capacity make that the correct final answer, not because quadtrees are inherently one level deep.

### Code and Verification

```cpp
#include <cstdio>
#include <algorithm>
#include <vector>

// Chapter 35.1 main -- concurrent quadtree build: each thread computes
// its own point's quadrant code independently (no synchronization
// needed there, exactly Section 30.1's per-particle hashing), then
// claims its slot in that quadrant's point list via a single shared
// atomicAdd cursor -- Section 30.2/33.3's bucket append, reused
// unchanged. Two threads landing in the SAME quadrant must both be
// counted and both receive distinct, correct slots, regardless of
// which one's atomicAdd happens to land first.

#define NUM_POINTS 3
#define MID_X 4.0
#define MID_Y 4.0

__device__ int quadrant_code_device(double x, double y) {
    return (x >= MID_X ? 1 : 0) + (y >= MID_Y ? 2 : 0);
}

__global__ void quadtree_build_kernel(const double* xs, const double* ys, int* cursors, int* quadrant_points) {
    int tid = threadIdx.x;
    int q = quadrant_code_device(xs[tid], ys[tid]);
    int slot = atomicAdd(&cursors[q], 1);
    quadrant_points[q * 2 + slot] = tid;
}

// ---- Host-side replay of the identical atomicAdd-cursor logic,
// ---- driving a forced landing order where two points sharing a
// ---- quadrant race, and a third, unrelated point does not. ----

int main() {
    printf("=== Section 35.1 main: concurrent quadtree build via atomicAdd-cursor append ===\n\n");

    double points[NUM_POINTS][2] = {{1, 1}, {2, 3}, {5, 1}};
    int landing_order[NUM_POINTS] = {1, 0, 2};

    printf("landing order (by point index): [ ");
    for (int i : landing_order) printf("%d ", i);
    printf("]\n\n");

    int cursors[4] = {0, 0, 0, 0};
    std::vector<int> quadrants[4];

    for (int i : landing_order) {
        double x = points[i][0], y = points[i][1];
        int q = (x >= MID_X ? 1 : 0) + (y >= MID_Y ? 2 : 0);
        int slot = cursors[q];
        cursors[q]++;
        quadrants[q].push_back(i);
        printf("  point %d (%.0f,%.0f): quadrant code = %d -> atomicAdd(cursor[%d],1) returned %d "
               "-> quadrants[%d][%d] = %d\n", i, x, y, q, q, slot, q, slot, i);
    }

    printf("\nfinal quadrant 0: [");
    for (size_t i = 0; i < quadrants[0].size(); i++) printf("%d%s", quadrants[0][i], (i + 1 < quadrants[0].size()) ? ", " : "");
    printf("]\n");
    printf("final quadrant 1: [");
    for (size_t i = 0; i < quadrants[1].size(); i++) printf("%d%s", quadrants[1][i], (i + 1 < quadrants[1].size()) ? ", " : "");
    printf("]\n");

    std::vector<int> q0_sorted = quadrants[0];
    std::sort(q0_sorted.begin(), q0_sorted.end());
    bool ok = (q0_sorted == std::vector<int>{0, 1}) && (quadrants[1] == std::vector<int>{2});
    printf("\nself-check: quadrant 0 holds both {0,1} regardless of landing order, quadrant 1\n");
    printf("holds only {2}, fully uncontended: %s\n", ok ? "confirmed" : "MISMATCH");

    return ok ? 0 : 1;
}
```

**Compile and run:**

```bash
NVDIR=/usr/local/lib/python3.11/dist-packages/nvidia/cu13
$NVDIR/bin/nvcc -arch=sm_80 -I$NVDIR/include -L$NVDIR/lib -lcudart 197_quadtree_build_kernel.cu -o 197_quadtree_build_kernel
LD_LIBRARY_PATH=$NVDIR/lib ./197_quadtree_build_kernel
```

**Sample input:** 3 concurrent threads, two of which (points 0 and 1) land in the same quadrant -- genuine contention on that quadrant's shared cursor -- with the third (point 2) fully uncontended.

**Sample output:**

```text
=== Section 35.1 main: concurrent quadtree build via atomicAdd-cursor append ===

landing order (by point index): [ 1 0 2 ]

  point 1 (2,3): quadrant code = 0 -> atomicAdd(cursor[0],1) returned 0 -> quadrants[0][0] = 1
  point 0 (1,1): quadrant code = 0 -> atomicAdd(cursor[0],1) returned 1 -> quadrants[0][1] = 0
  point 2 (5,1): quadrant code = 1 -> atomicAdd(cursor[1],1) returned 0 -> quadrants[1][0] = 2

final quadrant 0: [1, 0]
final quadrant 1: [2]

self-check: quadrant 0 holds both {0,1} regardless of landing order, quadrant 1
holds only {2}, fully uncontended: confirmed
```

## 35.2 Finding a Point's Nearest Neighbor via Pruned Search

### Intuition

The entire reason to build a spatial tree is to avoid comparing a query point against every point in the cloud. Once a query point's quadrant is known, only the points inside that SAME quadrant need to be examined at all -- for this section's 8-point, 4-quadrant example, that immediately cuts the search from 8 points down to 2. Within that small pruned set, finding the closest point is Section 25.2/32.2's packed-key argmin, unchanged: pack squared distance and local index into one key, reduce with MIN.

### The Sequential (CPU) Baseline

```cpp
// 198_nn_query_cpu_baseline.cpp
//
// Chapter 35.2 -- sequential baseline for nearest-neighbor search
// pruned by the quadtree. Finding a query point's nearest neighbor
// only needs to examine the points in the query's OWN quadrant
// (found via Section 35.1's quadrant-code computation), instead of
// scanning all 8 points -- the entire point of building a spatial
// tree. Squared distance is used throughout, since argmin is
// unaffected by the monotonic sqrt -- avoiding sqrt entirely.
//
// Compile: g++ -std=c++17 -Wall -Wextra -O2 198_nn_query_cpu_baseline.cpp -o 198_nn_query_cpu_baseline
// Run:     ./198_nn_query_cpu_baseline

#include <array>
#include <cstdio>
#include <vector>

constexpr double MID_X = 4.0, MID_Y = 4.0;

int quadrant_code(double x, double y) {
    return (x >= MID_X ? 1 : 0) + (y >= MID_Y ? 2 : 0);
}

int main() {
    double points[8][2] = {
        {1, 1}, {2, 3}, {5, 1}, {6, 2}, {1, 5}, {3, 6}, {5, 5}, {6, 7},
    };
    std::array<std::vector<int>, 4> quadrants = {
        std::vector<int>{0, 1}, std::vector<int>{2, 3}, std::vector<int>{4, 5}, std::vector<int>{6, 7}
    };

    double qx = 2.0, qy = 2.0;

    std::printf("=== Section 35.2 CPU baseline: sequential nearest-neighbor query, pruned to one quadrant ===\n\n");
    int q = quadrant_code(qx, qy);
    std::printf("query point: (%.1f, %.1f) -> quadrant code = %d\n", qx, qy, q);

    const std::vector<int>& candidates = quadrants[q];
    std::printf("pruned candidate set (quadrant %d only): [", q);
    for (size_t i = 0; i < candidates.size(); ++i) std::printf("%d%s", candidates[i], (i + 1 < candidates.size()) ? ", " : "");
    std::printf("] -- out of 8 total points\n\n");

    int best_idx = -1;
    double best_dist2 = -1.0;
    for (size_t local_i = 0; local_i < candidates.size(); ++local_i) {
        int pidx = candidates[local_i];
        double px = points[pidx][0], py = points[pidx][1];
        double dist2 = (qx - px) * (qx - px) + (qy - py) * (qy - py);
        std::printf("  candidate local_idx=%zu (point %d at (%.0f,%.0f)): squared distance = %.1f\n",
                    local_i, pidx, px, py, dist2);
        if (best_dist2 < 0 || dist2 < best_dist2) {
            best_dist2 = dist2;
            best_idx = pidx;
        }
    }

    std::printf("\nnearest neighbor: point %d at (%.0f,%.0f), squared distance = %.1f\n",
                best_idx, points[best_idx][0], points[best_idx][1], best_dist2);

    bool ok = (best_idx == 1) && (best_dist2 == 1.0);
    std::printf("\nself-check: nearest neighbor is point 1 (2,3) with squared distance 1.0: %s\n",
                ok ? "confirmed" : "MISMATCH");

    return ok ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++17 -Wall -Wextra -O2 198_nn_query_cpu_baseline.cpp -o 198_nn_query_cpu_baseline
./198_nn_query_cpu_baseline
```

**Sample input:** a query point at (2.0, 2.0), pruned to quadrant 0's 2 points instead of scanning all 8.

**Sample output:**

```text
=== Section 35.2 CPU baseline: sequential nearest-neighbor query, pruned to one quadrant ===

query point: (2.0, 2.0) -> quadrant code = 0
pruned candidate set (quadrant 0 only): [0, 1] -- out of 8 total points

  candidate local_idx=0 (point 0 at (1,1)): squared distance = 2.0
  candidate local_idx=1 (point 1 at (2,3)): squared distance = 1.0

nearest neighbor: point 1 at (2,3), squared distance = 1.0

self-check: nearest neighbor is point 1 (2,3) with squared distance 1.0: confirmed
```

### The Concept, In Detail

```
ASCII view: pruning shrinks the reduction's INPUT, not the algorithm.

  without a tree:  compare query against ALL 8 points  (8 candidates)
  with the quadtree: compare query against its OWN quadrant only
                      (2 candidates -- 6 points never even touched)

  packed-key MIN reduction, same technique as Section 25.2/32.2,
  just over a smaller array:
    point 0: dist2=2.0 -> key 2000
    point 1: dist2=1.0 -> key 1001
    min(2000, 1001) = 1001  -> decodes to point 1
```

Squared distance, not true Euclidean distance, is compared throughout -- since square root is monotonically increasing, the point with the smallest squared distance is guaranteed to also have the smallest true distance, so the expensive `sqrt` can be skipped entirely and only computed once, at the very end, if an actual distance value is ever needed.

[COMMON TRAP]
It is tempting to assume the query's own quadrant always contains the TRUE global nearest neighbor. When a query point sits close to a quadrant boundary, a point in a NEIGHBORING quadrant can genuinely be closer than anything in the query's own quadrant -- a real, correct nearest-neighbor search must also check adjacent quadrants whenever the query is within its own best-found distance of a boundary, backtracking exactly the way a correctly-implemented k-d tree or quadtree search does. This section's query point sits comfortably inside its quadrant's interior specifically so this backtracking step can be set aside without affecting the answer.

### Code and Verification

```cpp
#include <cstdio>
#include <vector>
#include <algorithm>

// Chapter 35.2 main -- within the pruned candidate set, finding the
// nearest point is Section 25.2/32.2's packed-key argmin: pack
// (squared distance, local index) into one comparable key, reduce
// with MIN via a pairwise tree (Chapter 4/31.1's exact technique),
// now over a tree-pruned candidate set of size 2 instead of a full
// 8-element array -- the reduction itself needs no change, only the
// SIZE of what it reduces shrank, because the quadtree already
// eliminated 6 of the 8 points from consideration.

__device__ long long pack_device(double dist2, int local_idx) {
    return (long long)(dist2 * 100.0 + 0.5) * 10 + local_idx;
}

__global__ void nn_reduce_kernel(long long* keys, int n) {
    int tid = threadIdx.x;
    if (tid * 2 + 1 >= n) return;
    long long a = keys[tid * 2];
    long long b = keys[tid * 2 + 1];
    keys[tid] = (a < b) ? a : b;
}

// ---- Host-side replay of the identical pack + MIN-reduction logic,
// ---- over the same pruned 2-candidate set Section 35.2's CPU
// ---- baseline computed. ----

long long pack_host(double dist2, int local_idx) {
    return (long long)(dist2 * 100.0 + 0.5) * 10 + local_idx;
}

int main() {
    printf("=== Section 35.2 main: packed-key MIN reduction over the pruned candidate set ===\n\n");

    double points[8][2] = {
        {1, 1}, {2, 3}, {5, 1}, {6, 2}, {1, 5}, {3, 6}, {5, 5}, {6, 7},
    };
    std::vector<int> quadrant0 = {0, 1};
    double qx = 2.0, qy = 2.0;
    int q = 0;

    printf("query point: (%.1f, %.1f) -> quadrant %d, candidates: [", qx, qy, q);
    for (size_t i = 0; i < quadrant0.size(); i++) printf("%d%s", quadrant0[i], (i + 1 < quadrant0.size()) ? ", " : "");
    printf("]\n\n");

    std::vector<long long> keys;
    for (size_t local_i = 0; local_i < quadrant0.size(); local_i++) {
        int pidx = quadrant0[local_i];
        double px = points[pidx][0], py = points[pidx][1];
        double dist2 = (qx - px) * (qx - px) + (qy - py) * (qy - py);
        long long key = pack_host(dist2, (int)local_i);
        keys.push_back(key);
        printf("  local_idx=%zu (point %d at (%.0f,%.0f)): squared distance = %.1f -> packed key = %lld\n",
               local_i, pidx, px, py, dist2, key);
    }

    long long best_key = std::min(keys[0], keys[1]);
    printf("\nround 1: min(%lld, %lld) -> %lld\n", keys[0], keys[1], best_key);

    int best_local = (int)(best_key % 10);
    double best_dist2 = (best_key / 10) / 100.0;
    int best_point = quadrant0[best_local];

    printf("\nwinning key %lld decodes to local_idx=%d -> point %d, squared distance = %.1f\n",
           best_key, best_local, best_point, best_dist2);

    bool ok = (best_point == 1) && (best_dist2 == 1.0);
    printf("\nself-check: the packed-key reduction's winner matches Section 35.2's CPU baseline\n");
    printf("exactly (point 1, squared distance 1.0): %s\n", ok ? "confirmed" : "MISMATCH");

    return ok ? 0 : 1;
}
```

**Compile and run:**

```bash
NVDIR=/usr/local/lib/python3.11/dist-packages/nvidia/cu13
$NVDIR/bin/nvcc -arch=sm_80 -I$NVDIR/include -L$NVDIR/lib -lcudart 199_nn_query_kernel.cu -o 199_nn_query_kernel
LD_LIBRARY_PATH=$NVDIR/lib ./199_nn_query_kernel
```

**Sample input:** the same query point and pruned 2-candidate set, resolved via a packed-key MIN reduction instead of a sequential comparison.

**Sample output:**

```text
=== Section 35.2 main: packed-key MIN reduction over the pruned candidate set ===

query point: (2.0, 2.0) -> quadrant 0, candidates: [0, 1]

  local_idx=0 (point 0 at (1,1)): squared distance = 2.0 -> packed key = 2000
  local_idx=1 (point 1 at (2,3)): squared distance = 1.0 -> packed key = 1001

round 1: min(2000, 1001) -> 1001

winning key 1001 decodes to local_idx=1 -> point 1, squared distance = 1.0

self-check: the packed-key reduction's winner matches Section 35.2's CPU baseline
exactly (point 1, squared distance 1.0): confirmed
```

## 35.3 Frame-to-Frame Point Correspondence for SLAM

### Intuition

SLAM's scan-matching step -- the computational core of algorithms like ICP (Iterative Closest Point) -- asks Section 35.2's exact question, once for every point in a NEW LiDAR frame: what is this point's nearest neighbor in the PREVIOUS frame's already-built index? The resulting list of (new_point, old_point) correspondence pairs is what a real SLAM system feeds into a separate least-squares step to estimate how far and which way the vehicle moved between the two scans -- a step this section does not implement, since it is pure linear algebra with no new parallel structure to teach, but the correspondence-finding step this section DOES build is the genuinely expensive, genuinely parallelizable part real systems offload to a GPU.

### The Sequential (CPU) Baseline

```cpp
// 200_frame_correspondence_cpu_baseline.cpp
//
// Chapter 35.3 -- sequential baseline for SLAM's scan-matching step:
// finding, for every point in a NEW LiDAR frame, its nearest neighbor
// in the PREVIOUS frame's already-built quadtree. This is exactly
// Section 35.2's pruned nearest-neighbor query, run once per new-frame
// point. The resulting (frame2_point, frame1_point) correspondence
// pairs are what a real ICP-based SLAM system would feed into a
// separate least-squares step to estimate the vehicle's motion.
//
// Compile: g++ -std=c++17 -Wall -Wextra -O2 200_frame_correspondence_cpu_baseline.cpp -o 200_frame_correspondence_cpu_baseline
// Run:     ./200_frame_correspondence_cpu_baseline

#include <array>
#include <cstdio>
#include <vector>

constexpr double MID_X = 4.0, MID_Y = 4.0;

int quadrant_code(double x, double y) {
    return (x >= MID_X ? 1 : 0) + (y >= MID_Y ? 2 : 0);
}

int main() {
    double frame1[8][2] = {
        {1, 1}, {2, 3}, {5, 1}, {6, 2}, {1, 5}, {3, 6}, {5, 5}, {6, 7},
    };
    std::array<std::vector<int>, 4> quadrants = {
        std::vector<int>{0, 1}, std::vector<int>{2, 3}, std::vector<int>{4, 5}, std::vector<int>{6, 7}
    };
    double frame2[4][2] = {
        {1.2, 0.9}, {5.1, 5.2}, {6.2, 1.8}, {3.1, 5.9},
    };
    constexpr int NUM_F2 = 4;

    std::printf("=== Section 35.3 CPU baseline: sequential frame-to-frame nearest-neighbor matching ===\n\n");
    std::printf("frame 1 (already indexed, Section 35.1's quadtree): [");
    for (int i = 0; i < 8; ++i) std::printf("(%d, (%.0f, %.0f))%s", i, frame1[i][0], frame1[i][1], (i + 1 < 8) ? ", " : "");
    std::printf("]\n");
    std::printf("frame 2 (new scan, to be matched): [");
    for (int i = 0; i < NUM_F2; ++i) std::printf("(%d, (%.1f, %.1f))%s", i, frame2[i][0], frame2[i][1], (i + 1 < NUM_F2) ? ", " : "");
    std::printf("]\n\n");

    std::vector<std::pair<int, int>> correspondences;
    for (int qi = 0; qi < NUM_F2; ++qi) {
        double qx = frame2[qi][0], qy = frame2[qi][1];
        int q = quadrant_code(qx, qy);
        const std::vector<int>& candidates = quadrants[q];
        int best_idx = -1;
        double best_dist2 = -1.0;
        for (int pidx : candidates) {
            double px = frame1[pidx][0], py = frame1[pidx][1];
            double dist2 = (qx - px) * (qx - px) + (qy - py) * (qy - py);
            if (best_dist2 < 0 || dist2 < best_dist2) {
                best_dist2 = dist2;
                best_idx = pidx;
            }
        }
        correspondences.push_back({qi, best_idx});
        std::printf("frame2 point %d (%.1f,%.1f): quadrant %d, candidates [", qi, qx, qy, q);
        for (size_t i = 0; i < candidates.size(); ++i) std::printf("%d%s", candidates[i], (i + 1 < candidates.size()) ? ", " : "");
        std::printf("] -> nearest is frame1 point %d (%.0f, %.0f), squared distance = %.4g\n",
                    best_idx, frame1[best_idx][0], frame1[best_idx][1], best_dist2);
    }

    std::printf("\ncorrespondences (frame2_point, frame1_point): [");
    for (size_t i = 0; i < correspondences.size(); ++i) {
        std::printf("(%d, %d)%s", correspondences[i].first, correspondences[i].second, (i + 1 < correspondences.size()) ? ", " : "");
    }
    std::printf("]\n");

    std::vector<std::pair<int, int>> expected = {{0, 0}, {1, 6}, {2, 3}, {3, 5}};
    bool ok = (correspondences == expected);
    std::printf("\nself-check: all 4 correspondences match hand-computed expectation exactly: %s\n",
                ok ? "confirmed" : "MISMATCH");

    return ok ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++17 -Wall -Wextra -O2 200_frame_correspondence_cpu_baseline.cpp -o 200_frame_correspondence_cpu_baseline
./200_frame_correspondence_cpu_baseline
```

**Sample input:** a second LiDAR frame of 4 points, each close to a genuine point in frame 1 (simulating the vehicle having moved slightly), matched against frame 1's already-built quadtree from Section 35.1.

**Sample output:**

```text
=== Section 35.3 CPU baseline: sequential frame-to-frame nearest-neighbor matching ===

frame 1 (already indexed, Section 35.1's quadtree): [(0, (1, 1)), (1, (2, 3)), (2, (5, 1)), (3, (6, 2)), (4, (1, 5)), (5, (3, 6)), (6, (5, 5)), (7, (6, 7))]
frame 2 (new scan, to be matched): [(0, (1.2, 0.9)), (1, (5.1, 5.2)), (2, (6.2, 1.8)), (3, (3.1, 5.9))]

frame2 point 0 (1.2,0.9): quadrant 0, candidates [0, 1] -> nearest is frame1 point 0 (1, 1), squared distance = 0.05
frame2 point 1 (5.1,5.2): quadrant 3, candidates [6, 7] -> nearest is frame1 point 6 (5, 5), squared distance = 0.05
frame2 point 2 (6.2,1.8): quadrant 1, candidates [2, 3] -> nearest is frame1 point 3 (6, 2), squared distance = 0.08
frame2 point 3 (3.1,5.9): quadrant 2, candidates [4, 5] -> nearest is frame1 point 5 (3, 6), squared distance = 0.02

correspondences (frame2_point, frame1_point): [(0, 0), (1, 6), (2, 3), (3, 5)]

self-check: all 4 correspondences match hand-computed expectation exactly: confirmed
```

### The Concept, In Detail

```
ASCII view: matching a whole new frame is many independent Section
35.2 queries, one per new point.

  frame 2 point 0 (1.2, 0.9)  -> quadrant 0 -> nearest: frame 1 point 0
  frame 2 point 1 (5.1, 5.2)  -> quadrant 3 -> nearest: frame 1 point 6
  frame 2 point 2 (6.2, 1.8)  -> quadrant 1 -> nearest: frame 1 point 3
  frame 2 point 3 (3.1, 5.9)  -> quadrant 2 -> nearest: frame 1 point 5

  every row above is a fully independent Section 35.2 query --
  none of them read or write any other row's data
```

Every frame-2 point's query reads the SAME shared, unchanging quadtree built once in Section 35.1 -- exactly Section 34.3's read-only, fully independent lookup pattern -- so all 4 (or, in a real system, millions of) correspondences can be computed simultaneously with no synchronization between them at all.

[COMMON TRAP]
It is tempting to reach for Section 34.3's shared atomicAdd-cursor output pattern here, since both sections write a variable-looking result to a shared array. A nearest-neighbor query, unlike a k-mer's seed lookup, always produces EXACTLY one result -- there is no case where zero or several matches are found -- so each thread can safely write directly to its own fixed, pre-assigned output slot `correspondences[tid]`, with no cursor and no atomics needed anywhere; reaching for atomics here would be solving a problem this section's structure does not actually have.

### Code and Verification

```cpp
#include <cstdio>
#include <vector>

// Chapter 35.3 main -- one thread per frame2 point, each querying the
// SAME shared, already-built quadtree (read-only, so no race there),
// exactly Section 33.1/34.3's independent-threads pattern. Unlike
// Section 34.3 (where a match count could be 0, 1, or several,
// needing a shared atomicAdd cursor), a nearest-neighbor query always
// produces EXACTLY one match, so each thread writes directly to its
// own dedicated output slot correspondences[tid] -- no cursor and no
// atomics needed at all.

__device__ int quadrant_code_device(double x, double y) {
    return (x >= 4.0 ? 1 : 0) + (y >= 4.0 ? 2 : 0);
}

__global__ void frame_match_kernel(const double* f1x, const double* f1y, const int* quadrant_points,
                                    const double* f2x, const double* f2y, int* correspondences) {
    int tid = threadIdx.x;
    double qx = f2x[tid], qy = f2y[tid];
    int q = quadrant_code_device(qx, qy);
    int c0 = quadrant_points[q * 2 + 0];
    int c1 = quadrant_points[q * 2 + 1];
    double d0 = (qx - f1x[c0]) * (qx - f1x[c0]) + (qy - f1y[c0]) * (qy - f1y[c0]);
    double d1 = (qx - f1x[c1]) * (qx - f1x[c1]) + (qy - f1y[c1]) * (qy - f1y[c1]);
    correspondences[tid] = (d0 < d1) ? c0 : c1;
}

// ---- Host-side replay of the identical per-thread lookup logic,
// ---- processed in an arbitrary order (reverse of index) to
// ---- demonstrate that thread execution order does not affect the
// ---- final, per-slot result. ----

int main() {
    printf("=== Section 35.3 main: concurrent frame-to-frame matching, one thread per frame2 point ===\n\n");
    printf("(threads are independent -- order shown here is arbitrary, e.g. reverse of index)\n\n");

    double frame1[8][2] = {
        {1, 1}, {2, 3}, {5, 1}, {6, 2}, {1, 5}, {3, 6}, {5, 5}, {6, 7},
    };
    std::vector<int> quadrant_points[4] = {{0, 1}, {2, 3}, {4, 5}, {6, 7}};
    double frame2[4][2] = {
        {1.2, 0.9}, {5.1, 5.2}, {6.2, 1.8}, {3.1, 5.9},
    };
    int thread_order[4] = {3, 2, 1, 0};
    int correspondences[4] = {-1, -1, -1, -1};

    for (int tid : thread_order) {
        double qx = frame2[tid][0], qy = frame2[tid][1];
        int q = (qx >= 4.0 ? 1 : 0) + (qy >= 4.0 ? 2 : 0);
        const std::vector<int>& candidates = quadrant_points[q];
        int best_idx = -1;
        double best_dist2 = -1.0;
        for (int pidx : candidates) {
            double px = frame1[pidx][0], py = frame1[pidx][1];
            double dist2 = (qx - px) * (qx - px) + (qy - py) * (qy - py);
            if (best_dist2 < 0 || dist2 < best_dist2) {
                best_dist2 = dist2;
                best_idx = pidx;
            }
        }
        correspondences[tid] = best_idx;
        printf("  thread %d (frame2 point (%.1f,%.1f)): quadrant %d, nearest = frame1 point %d "
               "-> writes correspondences[%d] = %d\n", tid, qx, qy, q, best_idx, tid, best_idx);
    }

    printf("\nfinal correspondences array: [ ");
    for (int c : correspondences) printf("%d ", c);
    printf("]\n");

    bool ok = (correspondences[0] == 0 && correspondences[1] == 6 &&
               correspondences[2] == 3 && correspondences[3] == 5);
    printf("\nself-check: correspondences array matches Section 35.3's CPU baseline exactly, "
           "in ORIGINAL index order despite arbitrary thread execution order: %s\n",
           ok ? "confirmed" : "MISMATCH");

    return ok ? 0 : 1;
}
```

**Compile and run:**

```bash
NVDIR=/usr/local/lib/python3.11/dist-packages/nvidia/cu13
$NVDIR/bin/nvcc -arch=sm_80 -I$NVDIR/include -L$NVDIR/lib -lcudart 201_frame_correspondence_kernel.cu -o 201_frame_correspondence_kernel
LD_LIBRARY_PATH=$NVDIR/lib ./201_frame_correspondence_kernel
```

**Sample input:** the same frame-2 points, each matched by its own independent thread against frame 1's quadtree, processed in an arbitrary (non-index) order to demonstrate that execution order does not affect any thread's own result.

**Sample output:**

```text
=== Section 35.3 main: concurrent frame-to-frame matching, one thread per frame2 point ===

(threads are independent -- order shown here is arbitrary, e.g. reverse of index)

  thread 3 (frame2 point (3.1,5.9)): quadrant 2, nearest = frame1 point 5 -> writes correspondences[3] = 5
  thread 2 (frame2 point (6.2,1.8)): quadrant 1, nearest = frame1 point 3 -> writes correspondences[2] = 3
  thread 1 (frame2 point (5.1,5.2)): quadrant 3, nearest = frame1 point 6 -> writes correspondences[1] = 6
  thread 0 (frame2 point (1.2,0.9)): quadrant 0, nearest = frame1 point 0 -> writes correspondences[0] = 0

final correspondences array: [ 0 6 3 5 ]

self-check: correspondences array matches Section 35.3's CPU baseline exactly, in ORIGINAL index order despite arbitrary thread execution order: confirmed
```

## Chapter Summary

A LiDAR nearest-neighbor and SLAM scan-matching pipeline needed no new parallel primitive -- it needed Chapter 18's spatial-tree idea combined with Section 31.2's bit-code partitioning, Section 25.2/32.2's packed-key reduction, and Section 30.2/33.3's bucket append. Section 35.1 showed that building a quadtree is a 1-level Morton code (a 2-bit quadrant code) combined with the same bucket-append pattern used for spatial hashing and stream compaction throughout Part 8. Section 35.2 showed that the entire benefit of a spatial tree is pruning -- reducing an argmin reduction's input from every point down to only the points that could possibly be relevant -- while the argmin reduction itself is unchanged from Section 25.2/32.2. Section 35.3 showed that SLAM's expensive scan-matching step is simply Section 35.2's query, repeated independently across every point in a new frame, with each thread writing to its own fixed slot since a nearest-neighbor query always produces exactly one result.

## Self-Check Questions

1. How does computing a quadrant code relate to Section 31.2's Morton code, and in what specific way is it simpler?
2. Section 35.1's example quadtree has only one level. What would need to happen for a quadrant to require a second level of subdivision, and what technique from earlier in the book would build that second level?
3. Why does pruning a nearest-neighbor search to one quadrant change the SIZE of Section 35.2's reduction but not the reduction algorithm itself?
4. Under what specific circumstance would Section 35.2's single-quadrant search give the WRONG answer, and what would a correct implementation need to do about it?
5. Why does Section 35.3's kernel write directly to `correspondences[tid]` instead of using the atomicAdd-cursor pattern Section 34.3 needed?
6. What real SLAM computation happens AFTER Section 35.3's correspondences are found, and why does this chapter stop short of implementing it?

## Where We Go Next

A LiDAR pipeline searches physical space for nearby points; the final case study turns to a completely different kind of "nearby" -- exchange rates between currencies, where a profitable arbitrage loop is not a geometric neighbor but a cycle in a graph whose edge weights multiply together to more than one. Chapter 36 closes Part 8, and the book, with graph-theoretic arbitrage detection in foreign-exchange markets, reusing Chapter 23's shortest-path and negative-cycle machinery on a graph where "distance" means something entirely different from every previous chapter's use of the word.

## Worked Solutions

**1.** A quadrant code, like a Morton code, interleaves one bit per spatial axis to produce a single integer that identifies a spatial region -- `(x>=mid_x?1:0) + (y>=mid_y?2:0)` is exactly a 2-bit Morton code computed at a single fixed depth. It is simpler because Section 31.2's Morton code interleaves MANY bits per axis (one per level of quantization precision, producing a fine-grained ordering across an entire scene), while a quadrant code only ever needs to distinguish 4 possible regions at ONE level, so a single comparison per axis is enough.

**2.** A quadrant would need a second level of subdivision if, after the first split, it still contained more points than the chosen capacity (2, in this section's example) -- meaning the points inside it are not yet spread out enough to search directly. Building that second level uses the identical level-synchronous, CAS-protected (or, here, atomicAdd-cursor) construction Chapter 16 established for tries and this chapter's own Section 35.1 used for the first level, simply applied again to the smaller sub-region, using an explicit stack rather than recursion to track which quadrants still need subdividing.

**3.** Pruning determines WHICH points are handed to the reduction as its input array -- restricting the search to one quadrant means only that quadrant's points ever become reduction leaves, shrinking the array's SIZE from 8 down to 2 in this section's example. The reduction algorithm itself (pack a comparable key, reduce with MIN via a pairwise tree) is completely unaffected by how many elements it is given; Section 35.2 changed what gets fed into Section 25.2/32.2's reduction, not the reduction's own logic.

**4.** Section 35.2's single-quadrant search gives the wrong answer whenever the true nearest neighbor lies in a DIFFERENT quadrant than the query point's own -- which can happen when the query point is near a quadrant boundary and a point just across that boundary is closer than any point on the query's own side. A correct implementation must check, after finding the best candidate within its own quadrant, whether that candidate's distance is larger than the query's distance to the nearest boundary; if so, it must also search the neighboring quadrant(s) within that distance, a backtracking step standard, correctly-implemented k-d tree and quadtree searches always include.

**5.** A k-mer lookup (Section 34.3) can legitimately produce zero, one, or several seed matches for a single query, so the number of output slots one thread needs is not known in advance, which is exactly why a shared atomicAdd cursor was needed to claim a variable-sized block of output space safely. A nearest-neighbor query always produces EXACTLY one result (the single closest point), so the number of slots per thread is always exactly one and always known in advance -- each thread's own fixed index already IS its unique, correct output slot, making any cursor or atomic operation unnecessary overhead for a problem that does not actually have variable multiplicity.

**6.** After correspondences are found, a real SLAM system solves a least-squares problem (commonly via SVD, singular value decomposition) over the matched point pairs to find the single rigid transformation -- a rotation and translation -- that best aligns frame 2 onto frame 1, which is the system's actual estimate of how the vehicle moved between scans. This chapter stops short of that step because it is pure linear algebra with no new parallel data structure or primitive to teach -- every technique this book has built (hashing, trees, reduction, scan, compaction) is about the SEARCH and INDEXING problem that produces the correspondences in the first place, which is the genuinely expensive, genuinely parallelizable part a real system offloads to a GPU, while the final least-squares solve is typically a small, fast computation run once per frame.
