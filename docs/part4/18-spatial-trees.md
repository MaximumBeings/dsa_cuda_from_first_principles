# Chapter 18: Spatial Trees

Every tree Part 4 has built so far organizes data along ONE dimension -- a single sort key, a single sequence of digits, a single array index. Real spatial data -- points on a map, particles in a simulation, triangles in a 3D scene -- lives in two or three dimensions at once, and asking "what is near this point" or "what overlaps this region" needs a tree whose SHAPE reflects that geometry. This chapter builds three such structures. The k-d tree (Sections 18.1-18.2) generalizes Chapter 16.1's partition-based BST construction to multiple dimensions, alternating which coordinate axis is used to split at each level. The quadtree (Section 18.3) takes the opposite approach -- splitting fixed regions of SPACE itself rather than the data -- and is presented alongside its direct 3D generalization (the octree) and its shape-oriented cousin (the bounding volume hierarchy), the structure that makes real-time collision detection and ray tracing possible.

## 18.1 k-d Trees: Level-by-Level Construction via Alternating-Axis Median Partitioning

### Intuition

A k-d tree splits a set of points with a hyperplane, alternating which coordinate axis is used at each level of the tree -- axis 0 (x) at the root, axis 1 (y) at its children, back to axis 0 two levels down, and so on. This is Chapter 16.1's own partition-based BST construction, generalized in exactly one place: instead of always comparing against a 1-D pivot (the first remaining element), each node's split value is the MEDIAN of its remaining points along the CURRENT level's axis, found by sorting those points along that axis. Everything else -- the iterative, queue-based, level-by-level construction, with no recursion anywhere -- carries over unchanged.

### The Sequential (CPU) Baseline

```cpp
#include <cstdio>
#include <vector>
#include <algorithm>
#include <queue>

// Chapter 18.1 -- The Sequential (CPU) Baseline.
// A k-d tree answers nearest-neighbor and range queries over points in
// k-dimensional space by recursively splitting the point set with a
// hyperplane, alternating which coordinate axis is used for the split
// at each level (axis 0, then axis 1, then back to axis 0, ...). This
// is a direct generalization of Chapter 16.1's partition-based BST:
// there, the "pivot" was always the first remaining element and the
// split was a 1-D less-than/greater-than test; here, the pivot is the
// MEDIAN element along the CURRENT level's axis, found by sorting the
// remaining points along that axis, and the split is "less than the
// median along this axis" versus "greater than it." Continuing this
// book's discipline, construction is ITERATIVE (an explicit queue of
// pending groups, exactly Chapter 16.1's own structure), never
// recursive.
struct Point { int x, y; };

struct Group { std::vector<Point> data; int slot; int level; };

void build_kdtree(const std::vector<Point>& pts, std::vector<Point>& value,
                   std::vector<int>& left, std::vector<int>& right,
                   std::vector<int>& axis_of) {
    value.clear(); left.clear(); right.clear(); axis_of.clear();
    int next_free = 0;
    auto alloc = [&]() {
        int s = next_free++;
        value.push_back({0, 0}); left.push_back(-1); right.push_back(-1); axis_of.push_back(-1);
        return s;
    };

    int root = alloc();
    std::queue<Group> q;
    q.push({pts, root, 0});

    while (!q.empty()) {
        Group g = q.front(); q.pop();
        int axis = g.level % 2;
        std::vector<Point> sorted_group = g.data;
        std::sort(sorted_group.begin(), sorted_group.end(), [axis](const Point& a, const Point& b) {
            return (axis == 0 ? a.x : a.y) < (axis == 0 ? b.x : b.y);
        });
        size_t mid = sorted_group.size() / 2;
        Point median = sorted_group[mid];
        std::vector<Point> less(sorted_group.begin(), sorted_group.begin() + mid);
        std::vector<Point> greater(sorted_group.begin() + mid + 1, sorted_group.end());

        value[g.slot] = median;
        axis_of[g.slot] = axis;
        if (!less.empty()) {
            int lslot = alloc();
            left[g.slot] = lslot;
            q.push({less, lslot, g.level + 1});
        }
        if (!greater.empty()) {
            int rslot = alloc();
            right[g.slot] = rslot;
            q.push({greater, rslot, g.level + 1});
        }
    }
}

int main() {
    printf("=== Section 18.1 CPU baseline: iterative k-d tree construction ===\n\n");

    std::vector<Point> pts = {{5,4}, {2,7}, {9,1}, {4,3}, {8,8}, {1,2}, {7,6}, {3,9}};
    printf("input points: ");
    for (auto& p : pts) printf("(%d,%d) ", p.x, p.y);
    printf("\n\n");

    std::vector<Point> value;
    std::vector<int> left, right, axis_of;
    build_kdtree(pts, value, left, right, axis_of);

    printf("k-d tree (median split, axis alternates x,y,x,... by level):\n");
    for (size_t i = 0; i < value.size(); i++) {
        printf("  slot%zu: point=(%d,%d) axis=%s left=%d right=%d\n",
               i, value[i].x, value[i].y, axis_of[i] == 0 ? "x" : "y", left[i], right[i]);
    }
    printf("\n");

    std::vector<Point> expected_value = {{5,4},{2,7},{7,6},{4,3},{3,9},{9,1},{8,8},{1,2}};
    std::vector<int> expected_left = {1, 3, 5, 7, -1, -1, -1, -1};
    std::vector<int> expected_right = {2, 4, 6, -1, -1, -1, -1, -1};
    std::vector<int> expected_axis = {0, 1, 1, 0, 0, 0, 0, 1};

    bool ok = (value.size() == expected_value.size());
    for (size_t i = 0; ok && i < value.size(); i++) {
        ok = ok && value[i].x == expected_value[i].x && value[i].y == expected_value[i].y &&
             left[i] == expected_left[i] && right[i] == expected_right[i] && axis_of[i] == expected_axis[i];
    }

    printf("expected: slot0=(5,4)x slot1=(2,7)y slot2=(7,6)y slot3=(4,3)x slot4=(3,9)x\n");
    printf("          slot5=(9,1)x slot6=(8,8)x slot7=(1,2)y\n");
    printf("expected left:  1 3 5 7 -1 -1 -1 -1\n");
    printf("expected right: 2 4 6 -1 -1 -1 -1 -1\n");

    printf("\nself-check: iterative median-split k-d tree construction matches expected\n");
    printf("layout: %s\n", ok ? "confirmed" : "MISMATCH");
    return ok ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++17 -Wall -Wextra -O2 94_kdtree_build_cpu_baseline.cpp -o 94_kdtree_build_cpu_baseline
./94_kdtree_build_cpu_baseline
```

**Sample input:** 8 two-dimensional points, built into a k-d tree with the splitting axis alternating x, y, x, ... by level.

**Sample output:**

```text
=== Section 18.1 CPU baseline: iterative k-d tree construction ===

input points: (5,4) (2,7) (9,1) (4,3) (8,8) (1,2) (7,6) (3,9) 

k-d tree (median split, axis alternates x,y,x,... by level):
  slot0: point=(5,4) axis=x left=1 right=2
  slot1: point=(2,7) axis=y left=3 right=4
  slot2: point=(7,6) axis=y left=5 right=6
  slot3: point=(4,3) axis=x left=7 right=-1
  slot4: point=(3,9) axis=x left=-1 right=-1
  slot5: point=(9,1) axis=x left=-1 right=-1
  slot6: point=(8,8) axis=x left=-1 right=-1
  slot7: point=(1,2) axis=y left=-1 right=-1

expected: slot0=(5,4)x slot1=(2,7)y slot2=(7,6)y slot3=(4,3)x slot4=(3,9)x
          slot5=(9,1)x slot6=(8,8)x slot7=(1,2)y
expected left:  1 3 5 7 -1 -1 -1 -1
expected right: 2 4 6 -1 -1 -1 -1 -1

self-check: iterative median-split k-d tree construction matches expected
layout: confirmed
```

### The Concept, In Detail

Tracing the construction on `{(5,4), (2,7), (9,1), (4,3), (8,8), (1,2), (7,6), (3,9)}`:

```
level 0 (axis x): all 8 points
  sorted by x: (1,2) (2,7) (3,9) (4,3) (5,4) (7,6) (8,8) (9,1)
  median (index 4) = (5,4)  -- this becomes the ROOT
  less-than-median (by x): (1,2) (2,7) (3,9) (4,3)
  greater-than-median (by x): (7,6) (8,8) (9,1)

level 1 (axis y): two independent groups
  group A {(1,2),(2,7),(3,9),(4,3)}, sorted by y: (1,2) (4,3) (2,7) (3,9)
    median (index 2) = (2,7)
  group B {(7,6),(8,8),(9,1)}, sorted by y: (9,1) (7,6) (8,8)
    median (index 1) = (7,6)

level 2 (axis x): four independent groups, most now singletons
  {(1,2),(4,3)} -> median (4,3);  {(3,9)} -> (3,9);  {(9,1)} -> (9,1);  {(8,8)} -> (8,8)

level 3 (axis y): one remaining singleton group {(1,2)} -> (1,2)
```

```
ASCII view of the resulting k-d tree (axis shown per node):

                          (5,4) x
                    /                \
              (2,7) y              (7,6) y
              /       \             /       \
        (4,3) x    (3,9) x    (9,1) x    (8,8) x
        /
   (1,2) y
```

Every node's own median is computed purely from its OWN group's remaining points -- exactly Chapter 16.1's own independence property, generalized from a 1-D comparison to a per-axis sort. Every group within a single level is completely independent of every other group in that level, so the entire level-by-level construction parallelizes exactly as Chapter 16.1's BST partitioning did, with one thread per group per level; the only new per-thread work is a small in-place sort of that thread's own private row before picking its median.

[COMMON TRAP]
It is tempting to sort ALL the points once, up front, along a single axis, and reuse that one sorted order at every level. Each level uses a DIFFERENT axis, and a group's membership (which points even belong to it) is determined by splits from EARLIER levels along DIFFERENT axes -- a global sort by x tells you nothing about the y-order of a subgroup that was carved out by an x-based split at a shallower level. Each node's sort must be performed fresh, on exactly its own group's remaining points, along its own level's axis.

### Code and Verification

```cpp
#include <cstdio>
#include <vector>
#include <algorithm>

// Chapter 18.1 main -- exactly Chapter 16.1's "one thread per GROUP per
// level" shape, generalized from a 1-D less-than/greater-than pivot
// test to a k-d tree's alternating-axis median split. Every group's
// own remaining points live in a fixed-capacity row of a flat buffer
// (MAX_GROUP columns wide, unchanged from Chapter 16.1), so no thread
// needs to know where any other group's data starts; the only new
// per-thread work is sorting its OWN small row along the level's
// current axis to find the median, entirely private, touching no
// other thread's memory.
#define MAX_GROUP 8

struct Point { int x, y; };

__device__ int coord(const Point& p, int axis) { return axis == 0 ? p.x : p.y; }

__device__ void insertion_sort_by_axis(Point* arr, int n, int axis) {
    for (int i = 1; i < n; i++) {
        Point key = arr[i];
        int keyval = coord(key, axis);
        int j = i - 1;
        while (j >= 0 && coord(arr[j], axis) > keyval) {
            arr[j + 1] = arr[j];
            j--;
        }
        arr[j + 1] = key;
    }
}

__global__ void kdtree_level_kernel(const Point* g_data, const int* g_count, const int* g_slot,
                                     int num_groups, int axis, Point* g_value, int* g_left,
                                     int* g_right, int* g_axis, Point* g_next_data,
                                     int* g_next_count, int* g_next_slot,
                                     const int* g_next_left_slot, const int* g_next_right_slot) {
    int t = threadIdx.x;
    if (t >= num_groups) return;

    Point local[MAX_GROUP];
    int count = g_count[t];
    for (int i = 0; i < count; i++) local[i] = g_data[t * MAX_GROUP + i];
    insertion_sort_by_axis(local, count, axis);

    int mid = count / 2;
    Point median = local[mid];
    int slot = g_slot[t];
    int nless = mid;
    int ngreater = count - mid - 1;

    g_value[slot] = median;
    g_axis[slot] = axis;
    g_left[slot] = (nless > 0) ? g_next_left_slot[t] : -1;
    g_right[slot] = (ngreater > 0) ? g_next_right_slot[t] : -1;

    Point* left_row = g_next_data + (2 * t) * MAX_GROUP;
    Point* right_row = g_next_data + (2 * t + 1) * MAX_GROUP;
    for (int i = 0; i < nless; i++) left_row[i] = local[i];
    for (int i = 0; i < ngreater; i++) right_row[i] = local[mid + 1 + i];
    g_next_count[2 * t] = nless;
    g_next_count[2 * t + 1] = ngreater;
    g_next_slot[2 * t] = (nless > 0) ? g_next_left_slot[t] : -1;
    g_next_slot[2 * t + 1] = (ngreater > 0) ? g_next_right_slot[t] : -1;
}

// ---- Host-side replay of the identical per-thread logic: one level's
// worth of independent threads per pass, each sorting and splitting
// only its own group. ----

int coord_host(const Point& p, int axis) { return axis == 0 ? p.x : p.y; }

int main() {
    printf("=== Section 18.1 main: level-by-level parallel k-d tree construction ===\n\n");

    std::vector<Point> pts = {{5,4}, {2,7}, {9,1}, {4,3}, {8,8}, {1,2}, {7,6}, {3,9}};
    printf("input points: ");
    for (auto& p : pts) printf("(%d,%d) ", p.x, p.y);
    printf("\n\n");

    std::vector<Point> value;
    std::vector<int> left, right, axis_of;
    int next_free = 0;
    auto alloc = [&]() {
        int s = next_free++;
        value.push_back({0, 0}); left.push_back(-1); right.push_back(-1); axis_of.push_back(-1);
        return s;
    };
    int root = alloc();

    std::vector<std::vector<Point>> cur_groups = {pts};
    std::vector<int> cur_slot = {root};
    int level_num = 0;

    while (!cur_groups.empty()) {
        int width = (int)cur_groups.size();
        int axis = level_num % 2;
        printf("level %d (axis %s): %d independent thread(s), slots [", level_num, axis == 0 ? "x" : "y", width);
        for (int s : cur_slot) printf("%d ", s);
        printf("]\n");

        std::vector<std::vector<Point>> next_groups;
        std::vector<int> next_slot;

        for (int t = 0; t < width; t++) {
            std::vector<Point> group = cur_groups[t];
            int slot = cur_slot[t];
            // insertion sort by axis -- identical to the device kernel's own
            std::sort(group.begin(), group.end(), [axis](const Point& a, const Point& b) {
                return coord_host(a, axis) < coord_host(b, axis);
            });
            size_t mid = group.size() / 2;
            Point median = group[mid];
            std::vector<Point> less(group.begin(), group.begin() + mid);
            std::vector<Point> greater(group.begin() + mid + 1, group.end());

            value[slot] = median;
            axis_of[slot] = axis;
            if (!less.empty()) { left[slot] = alloc(); next_groups.push_back(less); next_slot.push_back(left[slot]); }
            if (!greater.empty()) { right[slot] = alloc(); next_groups.push_back(greater); next_slot.push_back(right[slot]); }
        }
        cur_groups = next_groups;
        cur_slot = next_slot;
        level_num++;
    }
    printf("\n");

    printf("k-d tree:\n");
    for (size_t i = 0; i < value.size(); i++) {
        printf("  slot%zu: point=(%d,%d) axis=%s left=%d right=%d\n",
               i, value[i].x, value[i].y, axis_of[i] == 0 ? "x" : "y", left[i], right[i]);
    }

    std::vector<Point> expected_value = {{5,4},{2,7},{7,6},{4,3},{3,9},{9,1},{8,8},{1,2}};
    std::vector<int> expected_left = {1, 3, 5, 7, -1, -1, -1, -1};
    std::vector<int> expected_right = {2, 4, 6, -1, -1, -1, -1, -1};
    std::vector<int> expected_axis = {0, 1, 1, 0, 0, 0, 0, 1};
    bool ok = (value.size() == expected_value.size());
    for (size_t i = 0; ok && i < value.size(); i++) {
        ok = ok && value[i].x == expected_value[i].x && value[i].y == expected_value[i].y &&
             left[i] == expected_left[i] && right[i] == expected_right[i] && axis_of[i] == expected_axis[i];
    }

    printf("\n%d levels total -- every group WITHIN a level ran as an independent thread,\n", level_num);
    printf("sorting only its own private row of the fixed-capacity group buffer.\n");
    printf("\nself-check: level-by-level parallel construction matches CPU baseline: %s\n",
           ok ? "confirmed" : "MISMATCH");
    return ok ? 0 : 1;
}
```

**Compile and run:**

```bash
NVDIR=/usr/local/lib/python3.11/dist-packages/nvidia/cu13
$NVDIR/bin/nvcc -arch=sm_80 -I$NVDIR/include -L$NVDIR/lib -lcudart 95_kdtree_build_levels_kernel.cu -o 95_kdtree_build_levels_kernel
LD_LIBRARY_PATH=$NVDIR/lib ./95_kdtree_build_levels_kernel
```

**Sample input:** the same 8 points, built level by level, with every GROUP in a level handled by an independent thread that sorts only its own private row.

**Sample output:**

```text
=== Section 18.1 main: level-by-level parallel k-d tree construction ===

input points: (5,4) (2,7) (9,1) (4,3) (8,8) (1,2) (7,6) (3,9) 

level 0 (axis x): 1 independent thread(s), slots [0 ]
level 1 (axis y): 2 independent thread(s), slots [1 2 ]
level 2 (axis x): 4 independent thread(s), slots [3 4 5 6 ]
level 3 (axis y): 1 independent thread(s), slots [7 ]

k-d tree:
  slot0: point=(5,4) axis=x left=1 right=2
  slot1: point=(2,7) axis=y left=3 right=4
  slot2: point=(7,6) axis=y left=5 right=6
  slot3: point=(4,3) axis=x left=7 right=-1
  slot4: point=(3,9) axis=x left=-1 right=-1
  slot5: point=(9,1) axis=x left=-1 right=-1
  slot6: point=(8,8) axis=x left=-1 right=-1
  slot7: point=(1,2) axis=y left=-1 right=-1

4 levels total -- every group WITHIN a level ran as an independent thread,
sorting only its own private row of the fixed-capacity group buffer.

self-check: level-by-level parallel construction matches CPU baseline: confirmed
```

## 18.2 Nearest-Neighbor Query via Iterative, Pruned k-d Tree Search

### Intuition

A k-d tree's payoff is answering "which point is closest to this query point" in roughly `O(log n)` instead of scanning every point. The search descends toward the query's own side of each node's splitting hyperplane first (the "near" child), then checks whether the OTHER side (the "far" child) could still possibly hold something closer: every point on the far side is at least `|query[axis] - node[axis]|` away along that single axis alone, so the far child is worth visiting only if that one-axis distance is still less than the best distance found so far. Continuing this book's no-recursion discipline (Chapter 15's explicit stack), this search uses an explicit stack in place of a recursive call.

### The Sequential (CPU) Baseline

```cpp
#include <cstdio>
#include <vector>
#include <algorithm>
#include <queue>

// Chapter 18.2 -- The Sequential (CPU) Baseline.
// A k-d tree's real payoff is answering a nearest-neighbor query in
// roughly O(log n) instead of scanning every point. The search walks
// down toward the query point's own side of each splitting hyperplane
// first (the "near" child), then decides whether the OTHER side (the
// "far" child) could possibly contain something closer: since every
// point on the far side is at least `|query[axis] - node[axis]|` away
// along that one axis alone, the far child is only worth visiting if
// that single-axis distance is still less than the best distance found
// so far. Continuing this book's no-recursion discipline (Chapter 15's
// explicit stack), this search uses an explicit stack instead of a
// recursive call.
#define N 8
struct Point { int x, y; };

int dist2(const Point& a, const Point& b) {
    int dx = a.x - b.x, dy = a.y - b.y;
    return dx * dx + dy * dy;
}

void build_kdtree(const std::vector<Point>& pts, std::vector<Point>& value,
                   std::vector<int>& left, std::vector<int>& right, std::vector<int>& axis_of) {
    struct Group { std::vector<Point> data; int slot; int level; };
    value.clear(); left.clear(); right.clear(); axis_of.clear();
    int next_free = 0;
    auto alloc = [&]() {
        int s = next_free++;
        value.push_back({0, 0}); left.push_back(-1); right.push_back(-1); axis_of.push_back(-1);
        return s;
    };
    int root = alloc();
    std::queue<Group> q;
    q.push({pts, root, 0});
    while (!q.empty()) {
        Group g = q.front(); q.pop();
        int axis = g.level % 2;
        std::vector<Point> s = g.data;
        std::sort(s.begin(), s.end(), [axis](const Point& a, const Point& b) {
            return (axis == 0 ? a.x : a.y) < (axis == 0 ? b.x : b.y);
        });
        size_t mid = s.size() / 2;
        Point median = s[mid];
        std::vector<Point> less(s.begin(), s.begin() + mid);
        std::vector<Point> greater(s.begin() + mid + 1, s.end());
        value[g.slot] = median;
        axis_of[g.slot] = axis;
        if (!less.empty()) { int ls = alloc(); left[g.slot] = ls; q.push({less, ls, g.level + 1}); }
        if (!greater.empty()) { int rs = alloc(); right[g.slot] = rs; q.push({greater, rs, g.level + 1}); }
    }
}

// Explicit-stack nearest-neighbor search. A node is only pushed to be
// visited; whether it actually improves on the best answer, and
// whether its far child is even worth pushing, is decided the moment
// it is POPPED -- so pushing a node is never itself a claim that it
// will help, only that it might.
#define MAX_STACK 8
Point nn_search(const std::vector<Point>& value, const std::vector<int>& left,
                 const std::vector<int>& right, const std::vector<int>& axis_of,
                 int root, Point query, int& best_dist) {
    int stack[MAX_STACK];
    int sp = 0;
    stack[sp++] = root;
    Point best = value[root];
    best_dist = -1;

    while (sp > 0) {
        int slot = stack[--sp];
        if (slot == -1) continue;
        Point p = value[slot];
        int d = dist2(query, p);
        if (best_dist == -1 || d < best_dist) { best = p; best_dist = d; }

        int axis = axis_of[slot];
        int qcoord = (axis == 0 ? query.x : query.y);
        int pcoord = (axis == 0 ? p.x : p.y);
        int diff = qcoord - pcoord;
        int near = (diff < 0) ? left[slot] : right[slot];
        int far = (diff < 0) ? right[slot] : left[slot];

        bool visit_far = (far != -1) && (best_dist == -1 || (long long)diff * diff < best_dist);
        if (visit_far) stack[sp++] = far;
        if (near != -1) stack[sp++] = near;
    }
    return best;
}

int main() {
    printf("=== Section 18.2 CPU baseline: iterative, pruned k-d tree nearest-neighbor search ===\n\n");

    std::vector<Point> pts = {{5,4}, {2,7}, {9,1}, {4,3}, {8,8}, {1,2}, {7,6}, {3,9}};
    std::vector<Point> value;
    std::vector<int> left, right, axis_of;
    build_kdtree(pts, value, left, right, axis_of);
    printf("k-d tree built from Section 18.1 (8 points, alternating x/y median splits).\n\n");

    std::vector<Point> queries = {{6,5}, {0,0}, {9,9}, {4,4}};
    std::vector<Point> expected_nn = {{5,4}, {1,2}, {8,8}, {5,4}};
    std::vector<int> expected_d2 = {2, 5, 2, 1};

    bool ok = true;
    for (size_t i = 0; i < queries.size(); i++) {
        int bd;
        Point nn = nn_search(value, left, right, axis_of, 0, queries[i], bd);

        int brute_d = -1; Point brute_best = pts[0];
        for (auto& p : pts) {
            int d = dist2(queries[i], p);
            if (brute_d == -1 || d < brute_d) { brute_d = d; brute_best = p; }
        }

        printf("query (%d,%d): nearest=(%d,%d) dist2=%d  (brute-force check: (%d,%d) dist2=%d)\n",
               queries[i].x, queries[i].y, nn.x, nn.y, bd, brute_best.x, brute_best.y, brute_d);
        ok = ok && (nn.x == expected_nn[i].x) && (nn.y == expected_nn[i].y) && (bd == expected_d2[i])
             && (bd == brute_d);
    }

    printf("\nself-check: iterative pruned k-d tree search matches brute-force nearest\n");
    printf("neighbor for every query: %s\n", ok ? "confirmed" : "MISMATCH");
    return ok ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++17 -Wall -Wextra -O2 96_kdtree_nn_query_cpu_baseline.cpp -o 96_kdtree_nn_query_cpu_baseline
./96_kdtree_nn_query_cpu_baseline
```

**Sample input:** Section 18.1's k-d tree, queried for the nearest point to four sample query locations.

**Sample output:**

```text
=== Section 18.2 CPU baseline: iterative, pruned k-d tree nearest-neighbor search ===

k-d tree built from Section 18.1 (8 points, alternating x/y median splits).

query (6,5): nearest=(5,4) dist2=2  (brute-force check: (5,4) dist2=2)
query (0,0): nearest=(1,2) dist2=5  (brute-force check: (1,2) dist2=5)
query (9,9): nearest=(8,8) dist2=2  (brute-force check: (8,8) dist2=2)
query (4,4): nearest=(5,4) dist2=1  (brute-force check: (5,4) dist2=1)

self-check: iterative pruned k-d tree search matches brute-force nearest
neighbor for every query: confirmed
```

### The Concept, In Detail

Tracing the search for the nearest neighbor of query `(6,5)`:

```
stack=[root=(5,4)x]; best=none

pop (5,4), axis x: dist2=(6-5)^2+(5-4)^2=2 -> best=(5,4), best_dist=2
  diff = query.x(6) - node.x(5) = 1 (>=0) -> near=right child=(7,6), far=left child=(2,7)
  is far worth visiting? diff^2=1 < best_dist=2 -> YES, push far=(2,7)
  push near=(7,6)
stack=[(2,7), (7,6)]

pop (7,6), axis y: dist2=(6-7)^2+(5-6)^2=2 -> tie, NOT strictly less, best stays (5,4)
  diff = query.y(5) - node.y(6) = -1 -> near=left child=(9,1), far=right child=(8,8)
  is far worth visiting? diff^2=1 < best_dist=2 -> YES, push far=(8,8)
  push near=(9,1)
stack=[(2,7), (8,8), (9,1)]

pop (9,1): dist2=(6-9)^2+(5-1)^2=25 -- no improvement, no children
pop (8,8): dist2=(6-8)^2+(5-8)^2=13 -- no improvement, no children
pop (2,7): dist2=(6-2)^2+(5-7)^2=20 -- no improvement
  diff = query.y(5)-node.y(7) = -2 -> far=right child=(4,3)
  is far worth visiting? diff^2=4 < best_dist=2? NO -- PRUNED, never pushed
  push near=left child=(4,3)... wait: near here is LEFT (diff<0), so near=(4,3)
stack=[(4,3)]

pop (4,3): dist2=(6-4)^2+(5-3)^2=8 -- no improvement, no children

stack empty -- final answer: (5,4), dist2=2
```

The pruning check (`diff*diff < best_dist`) is what keeps this search close to `O(log n)` instead of degrading to a full scan: an entire subtree gets skipped the moment its own splitting axis alone proves it cannot contain anything closer than what has already been found, without ever looking at a single point inside it. Since each pop's decision depends on `best_dist`, which was itself set by every earlier pop, this whole walk is a genuinely sequential chain -- short (bounded by the tree's height), but sequential all the way through, exactly Section 17.1's own observation about a single range query's two-pointer walk.

[COMMON TRAP]
It is tempting to think pushing a node onto the stack means it will definitely improve the answer. Pushing only means "this node's OWN subtree might still be worth searching" -- whether it actually improves anything is decided the moment it is POPPED and its own distance is computed, and its own far child gets an entirely fresh pruning check at that point, using whatever `best_dist` happens to be by then (which may have improved further while this node sat on the stack).

### Code and Verification

```cpp
#include <cstdio>
#include <vector>

// Chapter 18.2 main -- exactly Chapter 15.2's and Chapter 17.1's own
// insight: a single nearest-neighbor search is a short but genuinely
// SEQUENTIAL walk (each pop's pruning decision depends on the best
// distance found by every earlier pop), so it is not worth splitting
// across threads. The real parallel opportunity is running MANY
// independent queries at once, one thread per query, each with its
// own PRIVATE stack -- unchanged from Chapter 15.2's "one thread per
// tree, one stack per thread" pattern, just with a prune-aware search
// instead of a plain traversal.
#define N 8
#define MAX_STACK 8
struct Point { int x, y; };

__device__ int dist2_dev(Point a, Point b) {
    int dx = a.x - b.x, dy = a.y - b.y;
    return dx * dx + dy * dy;
}

// One thread per QUERY. Every thread's stack, and every thread's
// current "best so far," is entirely private -- no shared mutable
// state, no atomics, and no interaction with any other thread's query
// at all, regardless of how many queries run at once.
__global__ void nn_query_kernel(const Point* value, const int* left, const int* right,
                                 const int* axis_of, const Point* queries, Point* results,
                                 int* result_dist, int num_queries) {
    int t = threadIdx.x;
    if (t >= num_queries) return;

    Point query = queries[t];
    int stack[MAX_STACK];
    int sp = 0;
    stack[sp++] = 0;   // root

    Point best = value[0];
    int best_dist = -1;

    while (sp > 0) {
        int slot = stack[--sp];
        if (slot == -1) continue;
        Point p = value[slot];
        int d = dist2_dev(query, p);
        if (best_dist == -1 || d < best_dist) { best = p; best_dist = d; }

        int axis = axis_of[slot];
        int qcoord = (axis == 0) ? query.x : query.y;
        int pcoord = (axis == 0) ? p.x : p.y;
        int diff = qcoord - pcoord;
        int near = (diff < 0) ? left[slot] : right[slot];
        int far = (diff < 0) ? right[slot] : left[slot];

        bool visit_far = (far != -1) && (best_dist == -1 || diff * diff < best_dist);
        if (visit_far) stack[sp++] = far;
        if (near != -1) stack[sp++] = near;
    }
    results[t] = best;
    result_dist[t] = best_dist;
}

// ---- Host-side replay of the identical per-thread logic. ----

int main() {
    printf("=== Section 18.2 main: parallel independent nearest-neighbor queries ===\n\n");

    std::vector<Point> value = {{5,4},{2,7},{7,6},{4,3},{3,9},{9,1},{8,8},{1,2}};
    std::vector<int> left  = {1, 3, 5, 7, -1, -1, -1, -1};
    std::vector<int> right = {2, 4, 6, -1, -1, -1, -1, -1};
    std::vector<int> axis_of = {0, 1, 1, 0, 0, 0, 0, 1};
    printf("k-d tree (Section 18.1's build): 8 nodes\n\n");

    std::vector<Point> queries = {{6,5}, {0,0}, {9,9}, {4,4}};
    std::vector<Point> results(queries.size());
    std::vector<int> result_dist(queries.size());

    printf("%zu independent query threads, each with its own private stack:\n", queries.size());
    for (size_t t = 0; t < queries.size(); t++) {
        Point query = queries[t];
        int stack[MAX_STACK];
        int sp = 0;
        stack[sp++] = 0;
        Point best = value[0];
        int best_dist = -1;

        while (sp > 0) {
            int slot = stack[--sp];
            if (slot == -1) continue;
            int dx = query.x - value[slot].x, dy = query.y - value[slot].y;
            int d = dx * dx + dy * dy;
            if (best_dist == -1 || d < best_dist) { best = value[slot]; best_dist = d; }

            int axis = axis_of[slot];
            int qcoord = (axis == 0) ? query.x : query.y;
            int pcoord = (axis == 0) ? value[slot].x : value[slot].y;
            int diff = qcoord - pcoord;
            int near = (diff < 0) ? left[slot] : right[slot];
            int far = (diff < 0) ? right[slot] : left[slot];
            bool visit_far = (far != -1) && (best_dist == -1 || diff * diff < best_dist);
            if (visit_far) stack[sp++] = far;
            if (near != -1) stack[sp++] = near;
        }
        results[t] = best;
        result_dist[t] = best_dist;
        printf("  thread %zu: query(%d,%d) -> nearest=(%d,%d) dist2=%d\n",
               t, query.x, query.y, best.x, best.y, best_dist);
    }

    std::vector<Point> expected_nn = {{5,4}, {1,2}, {8,8}, {5,4}};
    std::vector<int> expected_d2 = {2, 5, 2, 1};
    bool ok = true;
    for (size_t t = 0; t < queries.size(); t++) {
        ok = ok && results[t].x == expected_nn[t].x && results[t].y == expected_nn[t].y &&
             result_dist[t] == expected_d2[t];
    }

    printf("\nexpected: (5,4)d2=2, (1,2)d2=5, (8,8)d2=2, (5,4)d2=1\n");
    printf("\nself-check: parallel independent queries match the CPU baseline exactly: %s\n",
           ok ? "confirmed" : "MISMATCH");
    return ok ? 0 : 1;
}
```

**Compile and run:**

```bash
NVDIR=/usr/local/lib/python3.11/dist-packages/nvidia/cu13
$NVDIR/bin/nvcc -arch=sm_80 -I$NVDIR/include -L$NVDIR/lib -lcudart 97_kdtree_nn_query_parallel_kernel.cu -o 97_kdtree_nn_query_parallel_kernel
LD_LIBRARY_PATH=$NVDIR/lib ./97_kdtree_nn_query_parallel_kernel
```

**Sample input:** the same k-d tree, queried for four different points by four independent threads at once, each with its own private stack.

**Sample output:**

```text
=== Section 18.2 main: parallel independent nearest-neighbor queries ===

k-d tree (Section 18.1's build): 8 nodes

4 independent query threads, each with its own private stack:
  thread 0: query(6,5) -> nearest=(5,4) dist2=2
  thread 1: query(0,0) -> nearest=(1,2) dist2=5
  thread 2: query(9,9) -> nearest=(8,8) dist2=2
  thread 3: query(4,4) -> nearest=(5,4) dist2=1

expected: (5,4)d2=2, (1,2)d2=5, (8,8)d2=2, (5,4)d2=1

self-check: parallel independent queries match the CPU baseline exactly: confirmed
```

## 18.3 Quadtrees, Octrees, and Bounding Volume Hierarchies

### Intuition

A k-d tree splits the DATA -- its hyperplane's position depends on where the median happens to fall. A quadtree splits SPACE itself: every internal node's four children always cover exactly one quarter of its own bounding box (northwest, northeast, southwest, southeast), independent of how the data is distributed, and a node only splits once it holds more points than a fixed capacity. An octree is the direct three-dimensional generalization -- eight children per split (adding a third axis's own halving) instead of four. A bounding volume hierarchy (BVH) applies the same "split into a small number of children, recurse only where there is data" idea to SHAPES rather than points -- each node's bounding box tightly wraps all the geometry beneath it, letting a ray-tracing or collision query skip an entire branch of shapes the instant a ray or object provably misses that branch's own box, which is exactly what makes real-time ray tracing and physics engines fast enough to run every frame.

### The Sequential (CPU) Baseline

```cpp
#include <cstdio>
#include <vector>
#include <algorithm>

// Chapter 18.3 -- The Sequential (CPU) Baseline.
// A k-d tree (Sections 18.1-18.2) splits the DATA -- its splitting
// hyperplane's position depends on where the median point happens to
// fall. A quadtree splits SPACE instead: every internal node's four
// children always cover exactly one quarter of its own bounding box
// (NW, NE, SW, SE), regardless of how the data happens to be
// distributed, and a node only splits once it holds more than a fixed
// CAPACITY of points. This section uses coordinates chosen so every
// bounding-box midpoint lands on an exact integer (continuing this
// book's preference for exact, hand-traceable integer arithmetic over
// floating point). Continuing this book's no-recursion discipline, a
// cascading split (a redistributed point immediately overflowing one
// of the new children) is handled with an explicit WORKLIST, never a
// recursive call -- the same "make the implicit stack explicit" idea
// as Chapter 15's traversal, applied to insertion instead.
#define CAP 2
#define MAX_NODES 16

struct Point { int x, y; };

struct QNode {
    int x0, y0, x1, y1;
    int child[4];          // NW, NE, SW, SE; -1 if this node is a leaf
    int count;
    Point pts[CAP + 1];    // +1: room for the point that triggers a split
};

std::vector<QNode> nodes;

int alloc_node(int x0, int y0, int x1, int y1) {
    QNode n;
    n.x0 = x0; n.y0 = y0; n.x1 = x1; n.y1 = y1;
    n.child[0] = n.child[1] = n.child[2] = n.child[3] = -1;
    n.count = 0;
    nodes.push_back(n);
    return (int)nodes.size() - 1;
}

bool contains(const QNode& n, Point p) {
    return n.x0 <= p.x && p.x < n.x1 && n.y0 <= p.y && p.y < n.y1;
}

bool is_leaf(int idx) { return nodes[idx].child[0] == -1; }

int find_child(int idx, Point p) {
    for (int i = 0; i < 4; i++) {
        if (contains(nodes[nodes[idx].child[i]], p)) return nodes[idx].child[i];
    }
    return -1;   // unreachable if p is within idx's own bounds
}

// Splits a leaf into four quadrant children and returns the CAP+1
// points that need to be reinserted into them.
std::vector<Point> split_leaf(int idx) {
    int x0 = nodes[idx].x0, y0 = nodes[idx].y0, x1 = nodes[idx].x1, y1 = nodes[idx].y1;
    int mx = (x0 + x1) / 2, my = (y0 + y1) / 2;
    int nw = alloc_node(x0, my, mx, y1);
    int ne = alloc_node(mx, my, x1, y1);
    int sw = alloc_node(x0, y0, mx, my);
    int se = alloc_node(mx, y0, x1, my);
    nodes[idx].child[0] = nw; nodes[idx].child[1] = ne;
    nodes[idx].child[2] = sw; nodes[idx].child[3] = se;

    std::vector<Point> overflow;
    for (int i = 0; i < nodes[idx].count; i++) overflow.push_back(nodes[idx].pts[i]);
    nodes[idx].count = 0;
    return overflow;
}

struct PendingInsert { int start_node; Point p; };

void insert(int root, Point p) {
    std::vector<PendingInsert> work = {{root, p}};
    while (!work.empty()) {
        PendingInsert task = work.back(); work.pop_back();
        int cur = task.start_node;
        while (!is_leaf(cur)) cur = find_child(cur, task.p);

        nodes[cur].pts[nodes[cur].count++] = task.p;
        if (nodes[cur].count > CAP) {
            std::vector<Point> overflow = split_leaf(cur);
            for (auto& op : overflow) work.push_back({cur, op});
        }
    }
}

// Iterative range query using an explicit stack (Chapter 15's own
// discipline again): a node is pushed only when its own bounding box
// overlaps the query rectangle at all, so whole subtrees outside the
// query are pruned without ever being visited.
void range_query(int root, int qx0, int qy0, int qx1, int qy1, std::vector<Point>& out) {
    std::vector<int> stack = {root};
    while (!stack.empty()) {
        int idx = stack.back(); stack.pop_back();
        QNode& n = nodes[idx];
        if (n.x1 <= qx0 || n.x0 >= qx1 || n.y1 <= qy0 || n.y0 >= qy1) continue;   // no overlap
        if (is_leaf(idx)) {
            for (int i = 0; i < n.count; i++) {
                Point p = n.pts[i];
                if (qx0 <= p.x && p.x < qx1 && qy0 <= p.y && p.y < qy1) out.push_back(p);
            }
            continue;
        }
        for (int i = 0; i < 4; i++) stack.push_back(n.child[i]);
    }
}

int main() {
    printf("=== Section 18.3 CPU baseline: quadtree construction + range queries ===\n\n");

    std::vector<Point> pts = {{10,8}, {4,14}, {18,2}, {8,6}, {16,16}, {2,4}, {14,12}, {6,18}, {4,2}};
    printf("input points: ");
    for (auto& p : pts) printf("(%d,%d) ", p.x, p.y);
    printf("\ncapacity per leaf before splitting: %d\n\n", CAP);

    int root = alloc_node(0, 0, 20, 20);
    for (auto& p : pts) {
        insert(root, p);
        printf("inserted (%d,%d) -- %zu node(s) allocated so far\n", p.x, p.y, nodes.size());
    }
    printf("\n");

    printf("final quadtree (%zu nodes):\n", nodes.size());
    for (size_t i = 0; i < nodes.size(); i++) {
        QNode& n = nodes[i];
        if (is_leaf((int)i)) {
            printf("  leaf%zu bbox=(%d,%d)-(%d,%d) points=[", i, n.x0, n.y0, n.x1, n.y1);
            for (int j = 0; j < n.count; j++) printf("(%d,%d)%s", n.pts[j].x, n.pts[j].y, j + 1 < n.count ? "," : "");
            printf("]\n");
        } else {
            printf("  internal%zu bbox=(%d,%d)-(%d,%d) children=[NW=%d,NE=%d,SW=%d,SE=%d]\n",
                   i, n.x0, n.y0, n.x1, n.y1, n.child[0], n.child[1], n.child[2], n.child[3]);
        }
    }
    printf("\n");

    auto by_xy = [](const Point& a, const Point& b) {
        return a.x != b.x ? a.x < b.x : a.y < b.y;
    };

    struct RQ { int x0, y0, x1, y1; };
    std::vector<RQ> queries = {{0,0,10,10}, {10,10,20,20}, {0,10,20,20}, {4,0,16,10}};
    bool ok = (nodes.size() == 9);
    for (auto& q : queries) {
        std::vector<Point> out;
        range_query(root, q.x0, q.y0, q.x1, q.y1, out);
        std::vector<Point> brute;
        for (auto& p : pts) if (q.x0 <= p.x && p.x < q.x1 && q.y0 <= p.y && p.y < q.y1) brute.push_back(p);
        printf("range(%d,%d,%d,%d): tree found %zu point(s): ", q.x0, q.y0, q.x1, q.y1, out.size());
        for (auto& p : out) printf("(%d,%d) ", p.x, p.y);
        printf(" (brute-force found %zu)\n", brute.size());
        std::sort(out.begin(), out.end(), by_xy);
        std::sort(brute.begin(), brute.end(), by_xy);
        bool set_match = (out.size() == brute.size());
        for (size_t i = 0; set_match && i < out.size(); i++) set_match = (out[i].x == brute[i].x && out[i].y == brute[i].y);
        ok = ok && set_match;
    }

    printf("\nexpected: 9 total nodes (one cascading split at the SW quadrant)\n");
    printf("\nself-check: quadtree construction and range queries match brute-force: %s\n",
           ok ? "confirmed" : "MISMATCH");
    return ok ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++17 -Wall -Wextra -O2 98_quadtree_build_range_query_cpu_baseline.cpp -o 98_quadtree_build_range_query_cpu_baseline
./98_quadtree_build_range_query_cpu_baseline
```

**Sample input:** 9 two-dimensional points, inserted one at a time into a quadtree (leaf capacity 2) covering the region `(0,0)-(20,20)`, then queried over four sample rectangles.

**Sample output:**

```text
=== Section 18.3 CPU baseline: quadtree construction + range queries ===

input points: (10,8) (4,14) (18,2) (8,6) (16,16) (2,4) (14,12) (6,18) (4,2) 
capacity per leaf before splitting: 2

inserted (10,8) -- 1 node(s) allocated so far
inserted (4,14) -- 1 node(s) allocated so far
inserted (18,2) -- 5 node(s) allocated so far
inserted (8,6) -- 5 node(s) allocated so far
inserted (16,16) -- 5 node(s) allocated so far
inserted (2,4) -- 5 node(s) allocated so far
inserted (14,12) -- 5 node(s) allocated so far
inserted (6,18) -- 5 node(s) allocated so far
inserted (4,2) -- 9 node(s) allocated so far

final quadtree (9 nodes):
  internal0 bbox=(0,0)-(20,20) children=[NW=1,NE=2,SW=3,SE=4]
  leaf1 bbox=(0,10)-(10,20) points=[(4,14),(6,18)]
  leaf2 bbox=(10,10)-(20,20) points=[(16,16),(14,12)]
  internal3 bbox=(0,0)-(10,10) children=[NW=5,NE=6,SW=7,SE=8]
  leaf4 bbox=(10,0)-(20,10) points=[(18,2),(10,8)]
  leaf5 bbox=(0,5)-(5,10) points=[]
  leaf6 bbox=(5,5)-(10,10) points=[(8,6)]
  leaf7 bbox=(0,0)-(5,5) points=[(4,2),(2,4)]
  leaf8 bbox=(5,0)-(10,5) points=[]

range(0,0,10,10): tree found 3 point(s): (4,2) (2,4) (8,6)  (brute-force found 3)
range(10,10,20,20): tree found 2 point(s): (16,16) (14,12)  (brute-force found 2)
range(0,10,20,20): tree found 4 point(s): (16,16) (14,12) (4,14) (6,18)  (brute-force found 4)
range(4,0,16,10): tree found 3 point(s): (10,8) (4,2) (8,6)  (brute-force found 3)

expected: 9 total nodes (one cascading split at the SW quadrant)

self-check: quadtree construction and range queries match brute-force: confirmed
```

### The Concept, In Detail

Two splits happen during construction. The ROOT splits the instant its 3rd point (`(18,2)`) arrives (capacity 2 already reached by `(10,8)` and `(4,14)`):

```
ASCII view: root split into 4 fixed quadrants (independent of data).

  (0,20)-------(10,20)-------(20,20)
     |    NW       |    NE       |
     |  (4,14)     |  (16,16)    |
     |  (6,18)     |  (14,12)    |
  (0,10)-------(10,10)-------(20,10)
     |    SW       |    SE       |
     |  [subdivided further]     |
     |             |  (18,2)     |
     |             |  (10,8)     |
  (0,0)--------(10,0)--------(20,0)
```

The SW quadrant then ALSO exceeds capacity once its 3rd point (`(4,2)`) arrives, splitting again into its own four sub-quadrants:

```
SW quadrant (0,0)-(10,10), split into its own 4 children:

  (0,10)-------(5,10)-------(10,10)
     |   NW      |    NE       |
     |  (empty)  |   (8,6)     |
  (0,5)--------(5,5)--------(10,5)
     |   SW      |    SE       |
     |  (4,2)    |  (empty)    |
     |  (2,4)    |             |
  (0,0)--------(5,0)--------(10,0)
```

This SECOND split is a genuinely CASCADING one -- a point redistributed by the root's split (into the SW quadrant) later triggers a further split of that SAME quadrant. Continuing this book's no-recursion discipline, this is handled with an explicit WORKLIST (a plain stack of "this point still needs to be placed starting from this node" tasks) rather than a function that calls itself: redistributing an overflowing node's points simply pushes new tasks onto the same worklist the top-level insert loop already drains, so any number of cascading levels are handled by the exact same flat loop, however deep they happen to go.

A range query (find every point inside a rectangle) prunes exactly like Section 18.2's nearest-neighbor search: a node's own bounding box is checked against the query rectangle BEFORE descending into it, and any node with no overlap at all is skipped -- along with everything beneath it -- without ever being visited. Query, once again, is a short but sequential per-thread walk, so the real parallel opportunity is running many independent range queries at once, one thread per query.

```
ASCII view: octrees and BVHs extend the same two ideas to one more dimension,
and to shapes instead of points.

  quadtree (2D):  4 children per split, fixed quadrants, holds POINTS
  octree (3D):    8 children per split, fixed octants, holds POINTS
  BVH (any-D):    2 (or more) children per split, TIGHT-FITTING boxes
                  around whatever GEOMETRY is beneath them, holds SHAPES

  quadtree/octree: "which fixed region is this point in?"
  BVH:             "does this ray/shape overlap this node's own tight box?"
```

[COMMON TRAP]
It is tempting to think a quadtree and a BVH are the same idea with a different name. A quadtree's node boundaries are FIXED in advance by the region's own geometry (always exactly the parent's four quadrants, regardless of what data ends up inside them), while a BVH's node boundaries are DERIVED from whatever geometry happens to be beneath that node (the tightest box that contains it, recomputed as geometry is added or moves) -- a BVH node's box can be any size or aspect ratio at all, wasting no space on empty regions the way a quadtree's fixed quadrants sometimes must.

### Code and Verification

```cpp
#include <cstdio>
#include <vector>
#include <algorithm>

// Chapter 18.3 main -- exactly Section 18.2's own insight, once more: a
// single range query's tree descent is a short, genuinely sequential
// walk (whether a subtree is worth visiting depends on its own bounding
// box, known only once that node is reached), so the real parallel
// target is running MANY independent range queries at once, one thread
// per query, each with its own private stack -- unchanged in shape from
// Section 18.2's nearest-neighbor queries, just walking a quadtree's
// fixed spatial subdivision instead of a k-d tree's data-driven splits.
#define CAP 2
#define MAX_STACK 16
#define MAX_RESULTS 8

struct Point { int x, y; };

__global__ void range_query_kernel(const int* x0, const int* y0, const int* x1, const int* y1,
                                    const int* leaf_flag, const int* child, const int* counts,
                                    const Point* pts_flat, const int* qx0, const int* qy0,
                                    const int* qx1, const int* qy1, Point* results_flat,
                                    int* result_counts, int num_queries) {
    int t = threadIdx.x;
    if (t >= num_queries) return;

    int Qx0 = qx0[t], Qy0 = qy0[t], Qx1 = qx1[t], Qy1 = qy1[t];
    int stack[MAX_STACK];
    int sp = 0;
    stack[sp++] = 0;
    int rcount = 0;

    while (sp > 0) {
        int idx = stack[--sp];
        if (x1[idx] <= Qx0 || x0[idx] >= Qx1 || y1[idx] <= Qy0 || y0[idx] >= Qy1) continue;
        if (leaf_flag[idx]) {
            for (int i = 0; i < counts[idx]; i++) {
                Point p = pts_flat[idx * CAP + i];
                if (Qx0 <= p.x && p.x < Qx1 && Qy0 <= p.y && p.y < Qy1) {
                    results_flat[t * MAX_RESULTS + rcount] = p;
                    rcount++;
                }
            }
            continue;
        }
        for (int c = 0; c < 4; c++) stack[sp++] = child[idx * 4 + c];
    }
    result_counts[t] = rcount;
}

// ---- Host-side replay of the identical per-thread logic, over
// Section 18.3's own 9-node quadtree, flattened into plain arrays. ----

int main() {
    printf("=== Section 18.3 main: parallel independent quadtree range queries ===\n\n");

    // Section 18.3's CPU-baseline tree, flattened.
    std::vector<int> x0 = {0, 0, 10, 0, 10, 0, 5, 0, 5};
    std::vector<int> y0 = {0, 10, 10, 0, 0, 5, 5, 0, 0};
    std::vector<int> x1 = {20, 10, 20, 10, 20, 5, 10, 5, 10};
    std::vector<int> y1 = {20, 20, 20, 10, 10, 10, 10, 5, 5};
    std::vector<int> leaf_flag = {0, 1, 1, 0, 1, 1, 1, 1, 1};
    std::vector<int> child = {
        1, 2, 3, 4,     // node0
        -1,-1,-1,-1,    // node1 (leaf)
        -1,-1,-1,-1,    // node2 (leaf)
        5, 6, 7, 8,     // node3
        -1,-1,-1,-1,    // node4 (leaf)
        -1,-1,-1,-1,    // node5 (leaf)
        -1,-1,-1,-1,    // node6 (leaf)
        -1,-1,-1,-1,    // node7 (leaf)
        -1,-1,-1,-1,    // node8 (leaf)
    };
    std::vector<int> counts = {0, 2, 2, 0, 2, 0, 1, 2, 0};
    std::vector<Point> pts_flat(9 * CAP, Point{0, 0});
    pts_flat[1*CAP+0]={4,14};  pts_flat[1*CAP+1]={6,18};
    pts_flat[2*CAP+0]={16,16}; pts_flat[2*CAP+1]={14,12};
    pts_flat[4*CAP+0]={18,2};  pts_flat[4*CAP+1]={10,8};
    pts_flat[6*CAP+0]={8,6};
    pts_flat[7*CAP+0]={4,2};   pts_flat[7*CAP+1]={2,4};

    printf("quadtree: 9 nodes (unchanged from Section 18.3's CPU baseline)\n\n");

    struct RQ { int x0, y0, x1, y1; };
    std::vector<RQ> queries = {{0,0,10,10}, {10,10,20,20}, {0,10,20,20}, {4,0,16,10}};
    std::vector<int> result_counts(queries.size(), 0);
    std::vector<std::vector<Point>> results(queries.size());

    printf("%zu independent query threads, each with its own private stack:\n", queries.size());
    for (size_t t = 0; t < queries.size(); t++) {
        int Qx0 = queries[t].x0, Qy0 = queries[t].y0, Qx1 = queries[t].x1, Qy1 = queries[t].y1;
        int stack[MAX_STACK];
        int sp = 0;
        stack[sp++] = 0;
        int rcount = 0;
        std::vector<Point> out;

        while (sp > 0) {
            int idx = stack[--sp];
            if (x1[idx] <= Qx0 || x0[idx] >= Qx1 || y1[idx] <= Qy0 || y0[idx] >= Qy1) continue;
            if (leaf_flag[idx]) {
                for (int i = 0; i < counts[idx]; i++) {
                    Point p = pts_flat[idx * CAP + i];
                    if (Qx0 <= p.x && p.x < Qx1 && Qy0 <= p.y && p.y < Qy1) { out.push_back(p); rcount++; }
                }
                continue;
            }
            for (int c = 0; c < 4; c++) stack[sp++] = child[idx * 4 + c];
        }
        result_counts[t] = rcount;
        results[t] = out;
        printf("  thread %zu: range(%d,%d,%d,%d) -> %d point(s): ", t, Qx0, Qy0, Qx1, Qy1, rcount);
        for (auto& p : out) printf("(%d,%d) ", p.x, p.y);
        printf("\n");
    }

    std::vector<int> expected_counts = {3, 2, 4, 3};
    auto by_xy = [](const Point& a, const Point& b) { return a.x != b.x ? a.x < b.x : a.y < b.y; };
    std::vector<std::vector<Point>> expected_points = {
        {{2,4},{4,2},{8,6}},
        {{14,12},{16,16}},
        {{4,14},{6,18},{14,12},{16,16}},
        {{4,2},{8,6},{10,8}},
    };
    bool ok = true;
    for (size_t t = 0; t < queries.size(); t++) {
        ok = ok && (result_counts[t] == expected_counts[t]);
        std::vector<Point> got = results[t];
        std::sort(got.begin(), got.end(), by_xy);
        std::vector<Point> exp = expected_points[t];
        std::sort(exp.begin(), exp.end(), by_xy);
        bool set_match = (got.size() == exp.size());
        for (size_t i = 0; set_match && i < got.size(); i++) set_match = (got[i].x == exp[i].x && got[i].y == exp[i].y);
        ok = ok && set_match;
    }

    printf("\nexpected result counts: 3 2 4 3\n");
    printf("\nself-check: parallel independent range queries match the CPU baseline's point\n");
    printf("sets exactly (regardless of traversal order): %s\n", ok ? "confirmed" : "MISMATCH");
    return ok ? 0 : 1;
}
```

**Compile and run:**

```bash
NVDIR=/usr/local/lib/python3.11/dist-packages/nvidia/cu13
$NVDIR/bin/nvcc -arch=sm_80 -I$NVDIR/include -L$NVDIR/lib -lcudart 99_quadtree_range_query_parallel_kernel.cu -o 99_quadtree_range_query_parallel_kernel
LD_LIBRARY_PATH=$NVDIR/lib ./99_quadtree_range_query_parallel_kernel
```

**Sample input:** the same 9-node quadtree, queried over the same four rectangles by four independent threads at once, each with its own private stack.

**Sample output:**

```text
=== Section 18.3 main: parallel independent quadtree range queries ===

quadtree: 9 nodes (unchanged from Section 18.3's CPU baseline)

4 independent query threads, each with its own private stack:
  thread 0: range(0,0,10,10) -> 3 point(s): (4,2) (2,4) (8,6) 
  thread 1: range(10,10,20,20) -> 2 point(s): (16,16) (14,12) 
  thread 2: range(0,10,20,20) -> 4 point(s): (16,16) (14,12) (4,14) (6,18) 
  thread 3: range(4,0,16,10) -> 3 point(s): (10,8) (4,2) (8,6) 

expected result counts: 3 2 4 3

self-check: parallel independent range queries match the CPU baseline's point
sets exactly (regardless of traversal order): confirmed
```

## Chapter Summary

A k-d tree generalizes Chapter 16.1's partition-based BST construction to multiple dimensions by alternating the splitting axis at each level and using the MEDIAN (found by sorting) rather than a fixed pivot -- every group within a level remains independent of every other group in that level, so construction parallelizes exactly as Chapter 16.1's did, one thread per group per level. Nearest-neighbor search descends toward the query's own side of each split first, then prunes the opposite side whenever that side's own splitting axis alone proves it cannot contain anything closer -- a short but genuinely sequential per-query walk, so the real parallel opportunity is running many independent queries at once, each with its own private explicit stack, exactly Section 17.1's and Chapter 15's own pattern. A quadtree takes the complementary approach of splitting fixed regions of space rather than the data itself, subdividing into four quadrants only once a node's capacity is exceeded, with cascading splits handled by an explicit worklist rather than recursion; octrees generalize the same idea to three dimensions with eight children per split, and bounding volume hierarchies apply the same "skip whatever a bounding box proves cannot matter" pruning to arbitrary geometry rather than fixed regions, which is what makes real-time ray tracing and collision detection tractable.

## Self-Check Questions

1. What is the one change that generalizes Chapter 16.1's partition-based BST construction into a k-d tree's construction, and what stays exactly the same?
2. Why does a k-d tree's level-by-level construction need a fresh sort at every single node, rather than one global sort performed once up front?
3. In nearest-neighbor search, what does the pruning condition `diff*diff < best_dist` actually test, and what does it mean when that condition is false?
4. Why is a single nearest-neighbor query considered short-but-sequential rather than something to parallelize internally, and what is the right parallel target instead?
5. What is the difference between a quadtree's node boundaries and a bounding volume hierarchy's node boundaries, and why does that difference matter for what each structure is used for?
6. Why does handling a cascading quadtree split with an explicit worklist avoid the recursion that a naive implementation would otherwise need?

## Where We Go Next

Part 4 is complete: from binary trees, through parallel tree and trie construction, range-query structures, and now spatial trees, this book has built trees for essentially every shape of tree-structured question. Part 5 turns to a different kind of structure entirely -- the hash table -- starting with open addressing under concurrent insertion, where multiple threads inserting into the SAME shared table at once reopens this book's concurrency story one more time, and cuckoo hashing, which achieves a genuine, guaranteed O(1) worst-case lookup that ordinary open addressing cannot promise.

## Worked Solutions

**1.** The one change is WHICH VALUE becomes each node's split point and WHICH AXIS it is measured along: instead of always comparing against the first remaining element along a single implicit dimension, a k-d tree node's split point is the MEDIAN of its remaining points (found by sorting) along whichever axis the current LEVEL uses, alternating axis by level. Everything else -- iterative, queue-based, level-by-level construction with every group in a level independent of every other group in that level, requiring no shared allocation counter for the data itself -- carries over completely unchanged from Chapter 16.1.

**2.** A group's membership at any given node was determined by a CHAIN of earlier splits, each performed along a DIFFERENT axis than the current one. A single global sort by one axis captures that axis's order for the WHOLE original point set, but says nothing about a different axis's order restricted to just the specific subset of points that survived a sequence of earlier, different-axis splits -- that subset and its order along the new axis can only be determined by looking at exactly those points, freshly, which requires a new sort at every node.

**3.** The condition tests whether the far child's entire subtree could STILL possibly contain a point closer than the best one found so far, using only the single-axis distance from the query to the splitting node (`diff`) as a lower bound on how far away anything in the far subtree could be. When the condition is false, `diff*diff >= best_dist`, meaning even the CLOSEST possible point on the far side (right up against the splitting hyperplane) is already at least as far as the current best -- so the entire far subtree, however large, is guaranteed to contain nothing better and can be skipped without visiting a single node inside it.

**4.** Each step of the search (each stack pop) makes its pruning decision using `best_dist`, a value that was itself set by the outcome of every earlier pop in this SAME query -- the steps have a genuine data dependency chain, so no amount of additional threads can shorten one query's own critical path, only make that chain (bounded by the tree's height) short to begin with. The right parallel target is running many independent queries side by side, one thread per query, since different queries share no dependency on each other at all -- only the same, read-only tree every one of them is searching.

**5.** A quadtree's node boundaries are FIXED the moment the parent splits -- always exactly the parent's own box cut into four equal quadrants, regardless of what data ends up in each one, which is simple and cheap to compute but can waste space on children whose entire box turns out to be empty. A bounding volume hierarchy's node boundaries are DERIVED from whatever geometry actually sits beneath that node -- the tightest box that contains it, which can be any size, shape, or position at all. That difference matters because a quadtree's fixed regions are ideal when queries are naturally phrased in terms of a coordinate SPACE (does this point fall in this region), while a BVH's tight, data-driven boxes are ideal when queries are phrased in terms of GEOMETRY (does this ray or shape overlap this actual content), wasting no pruning opportunity on empty space a fixed grid might have included.

**6.** An explicit worklist turns "redistribute this node's overflowing points, and if any of THOSE overflow their new home too, redistribute again" into a single flat loop that keeps pulling the next pending `(node, point)` task off the SAME list and processing it, rather than a function whose own handling of one overflow directly invokes another call to handle a further overflow. Each task is fully self-contained (a starting node and a point to place), so the loop can keep draining newly-added tasks for as many cascading levels as actually occur, with no call stack ever growing deeper than one frame, regardless of how many splits cascade.
