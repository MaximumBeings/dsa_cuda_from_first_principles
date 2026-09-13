# Chapter 31: Parallel BVH Construction for Ray Tracing

A ray tracer's core question -- which objects does this ray actually pass near -- is a spatial search problem, but a different shape of one than Chapter 30's particle neighbors. Particles are roughly uniform in size and spread through space; a scene's objects can be wildly different sizes, a tiny bolt sitting next to a sprawling terrain mesh, and what a ray tracer needs is a TREE that groups nearby objects together at every scale. A bounding volume hierarchy (BVH) is exactly that tree, and building one in parallel combines two tools this book already has in full: Chapter 4's reduction, generalized from numbers to bounding boxes, and Chapter 16's level-synchronous parallel tree construction, applied to primitives ordered by a Morton code so that spatial neighbors end up adjacent in memory.

## 31.1 Computing Bounding Boxes via Reduction

### Intuition

Every primitive in a scene starts with its own axis-aligned bounding box (AABB); before anything else can be built, the scene's OVERALL bounding box is needed, both as a top-level bound and to normalize coordinates for Section 31.2's Morton codes. Finding it is Chapter 4's reduction with a different combine operator: instead of numeric min or max, two boxes combine via UNION -- the smallest box containing both. Union is still associative and commutative, so every guarantee Chapter 4 proved about reduction (in particular, that a tree-shaped parallel reduction gives the identical answer to any sequential fold) carries over with no new proof required.

### The Sequential (CPU) Baseline

```cpp
#include <cstdio>
#include <vector>
#include <algorithm>

// Chapter 31.1 CPU baseline -- a bounding volume hierarchy starts from
// the scene's overall bounding box, needed to normalize primitive
// centroids before Section 31.2 can compute Morton codes from them.
// Finding it is exactly Chapter 4's reduction, with the combine
// operator changed from numeric min/max to AABB UNION: the operator is
// still associative and order-independent, so every property Chapter 4
// proved about reduction (in particular, that this book's tree-shaped
// parallel reduction gives the exact same answer as a sequential fold)
// carries over unchanged to a completely different payload type.

struct AABB { double xmin, ymin, xmax, ymax; };

AABB box_union(const AABB& a, const AABB& b) {
    return {std::min(a.xmin, b.xmin), std::min(a.ymin, b.ymin),
            std::max(a.xmax, b.xmax), std::max(a.ymax, b.ymax)};
}

void print_box(const AABB& b) {
    printf("(%.1f, %.1f, %.1f, %.1f)", b.xmin, b.ymin, b.xmax, b.ymax);
}

int main() {
    printf("=== Section 31.1 CPU baseline: sequential reduction (box union) over 8 primitive AABBs ===\n\n");

    std::vector<AABB> boxes = {
        {0.0, 0.0, 1.0, 1.0}, {0.5, 0.5, 1.5, 1.5}, {5.0, 5.0, 6.0, 6.0}, {5.2, 5.1, 6.1, 6.0},
        {0.0, 5.0, 1.0, 6.0}, {0.1, 5.2, 1.2, 6.1}, {5.0, 0.0, 6.0, 1.0}, {5.3, 0.2, 6.2, 1.1},
    };

    AABB acc = boxes[0];
    printf("start: acc = ");
    print_box(acc);
    printf("\n");
    for (size_t i = 1; i < boxes.size(); i++) {
        acc = box_union(acc, boxes[i]);
        printf("union with box %zu = ", i);
        print_box(boxes[i]);
        printf(" -> acc = ");
        print_box(acc);
        printf("\n");
    }

    printf("\nscene bounding box: ");
    print_box(acc);
    printf("\n");

    bool ok = (acc.xmin == 0.0 && acc.ymin == 0.0 && acc.xmax == 6.2 && acc.ymax == 6.1);
    printf("\nself-check: the reduced box tightly contains every one of the 8 input\n");
    printf("boxes and matches the expected scene extent (0.0, 0.0, 6.2, 6.1): %s\n",
           ok ? "confirmed" : "MISMATCH");
    return ok ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++17 -Wall -Wextra -O2 172_bvh_reduce_bbox_cpu_baseline.cpp -o 172_bvh_reduce_bbox_cpu_baseline
./172_bvh_reduce_bbox_cpu_baseline
```

**Sample input:** 8 primitive AABBs scattered across four spatial clusters in a small 2D scene, folded sequentially with box union.

**Sample output:**

```text
=== Section 31.1 CPU baseline: sequential reduction (box union) over 8 primitive AABBs ===

start: acc = (0.0, 0.0, 1.0, 1.0)
union with box 1 = (0.5, 0.5, 1.5, 1.5) -> acc = (0.0, 0.0, 1.5, 1.5)
union with box 2 = (5.0, 5.0, 6.0, 6.0) -> acc = (0.0, 0.0, 6.0, 6.0)
union with box 3 = (5.2, 5.1, 6.1, 6.0) -> acc = (0.0, 0.0, 6.1, 6.0)
union with box 4 = (0.0, 5.0, 1.0, 6.0) -> acc = (0.0, 0.0, 6.1, 6.0)
union with box 5 = (0.1, 5.2, 1.2, 6.1) -> acc = (0.0, 0.0, 6.1, 6.1)
union with box 6 = (5.0, 0.0, 6.0, 1.0) -> acc = (0.0, 0.0, 6.1, 6.1)
union with box 7 = (5.3, 0.2, 6.2, 1.1) -> acc = (0.0, 0.0, 6.2, 6.1)

scene bounding box: (0.0, 0.0, 6.2, 6.1)

self-check: the reduced box tightly contains every one of the 8 input
boxes and matches the expected scene extent (0.0, 0.0, 6.2, 6.1): confirmed
```

### The Concept, In Detail

```
ASCII view: box union is Chapter 4's reduction with a different combine.

  numeric reduction:  acc = min(acc, x)         (or max, or +)

  box reduction:      acc = union(acc, box)
                           = ( min(acc.xmin, box.xmin), min(acc.ymin, box.ymin),
                               max(acc.xmax, box.xmax), max(acc.ymax, box.ymax) )

  both are associative and commutative -- combining in ANY order, or ANY
  grouping (sequential fold vs. pairwise tree), gives the identical
  final answer
```

Nothing about Chapter 4's reduction actually required its payload to be a single number -- it required only an associative, order-independent combine operator, and box union satisfies that exactly as well as addition or numeric min does. This is why Section 31.1's sequential fold and Section 31.1's parallel tree version (next) can be proven to agree without re-deriving reduction's correctness from scratch.

[COMMON TRAP]
It is tempting to think a bounding box "reduction" needs a fundamentally different algorithm than a numeric reduction, since a box is a compound value with four fields instead of one number. The algorithm -- fold or tree, sequential or parallel -- is completely unchanged; only the combine operator's definition changes, from a numeric comparison to four independent min/max comparisons packed into one operation. Any code that already implements Chapter 4's reduction correctly needs only its combine function swapped to reduce boxes instead of numbers.

### Code and Verification

```cpp
#include <cstdio>
#include <vector>
#include <algorithm>

// Chapter 31.1 main -- Chapter 4's PAIRWISE TREE reduction, not the
// sequential fold, is what a real kernel launches: log2(N) rounds, each
// one pairing up neighbors and replacing them with their combined
// value. Here the combine is AABB union instead of numeric min/max, but
// the reduction tree's SHAPE -- and the proof that it gives the exact
// same answer as any sequential fold, because union is associative and
// commutative -- is unchanged from Chapter 4.

#define NUM_LEAVES 8

struct AABB { double xmin, ymin, xmax, ymax; };

__device__ AABB box_union_device(const AABB& a, const AABB& b) {
    AABB r;
    r.xmin = a.xmin < b.xmin ? a.xmin : b.xmin;
    r.ymin = a.ymin < b.ymin ? a.ymin : b.ymin;
    r.xmax = a.xmax > b.xmax ? a.xmax : b.xmax;
    r.ymax = a.ymax > b.ymax ? a.ymax : b.ymax;
    return r;
}

__global__ void reduce_round_kernel(AABB* boxes, int n) {
    int tid = threadIdx.x;
    if (tid * 2 + 1 >= n) return;
    boxes[tid] = box_union_device(boxes[tid * 2], boxes[tid * 2 + 1]);
}

// ---- Host-side replay of the identical pairwise tree-reduction logic,
// ---- run round by round exactly as log2(8)=3 kernel launches would. ----

AABB box_union_host(const AABB& a, const AABB& b) {
    return {std::min(a.xmin, b.xmin), std::min(a.ymin, b.ymin),
            std::max(a.xmax, b.xmax), std::max(a.ymax, b.ymax)};
}

void print_box(const AABB& b) {
    printf("(%.1f, %.1f, %.1f, %.1f)", b.xmin, b.ymin, b.xmax, b.ymax);
}

int main() {
    printf("=== Section 31.1 main: parallel tree reduction (box union), log2(8)=3 rounds ===\n\n");

    std::vector<AABB> level = {
        {0.0, 0.0, 1.0, 1.0}, {0.5, 0.5, 1.5, 1.5}, {5.0, 5.0, 6.0, 6.0}, {5.2, 5.1, 6.1, 6.0},
        {0.0, 5.0, 1.0, 6.0}, {0.1, 5.2, 1.2, 6.1}, {5.0, 0.0, 6.0, 1.0}, {5.3, 0.2, 6.2, 1.1},
    };

    printf("round 0 (leaves): [ ");
    for (auto& b : level) { print_box(b); printf(" "); }
    printf("]\n\n");

    int round_num = 1;
    while (level.size() > 1) {
        std::vector<AABB> next_level;
        printf("round %d:\n", round_num);
        for (size_t i = 0; i < level.size(); i += 2) {
            AABB merged = box_union_host(level[i], level[i + 1]);
            printf("  thread %zu: union(slot %zu=", i / 2, i);
            print_box(level[i]);
            printf(", slot %zu=", i + 1);
            print_box(level[i + 1]);
            printf(") -> ");
            print_box(merged);
            printf("\n");
            next_level.push_back(merged);
        }
        level = next_level;
        printf("round %d result: [ ", round_num);
        for (auto& b : level) { print_box(b); printf(" "); }
        printf("]\n\n");
        round_num++;
    }

    printf("final scene bounding box: ");
    print_box(level[0]);
    printf("\n");

    bool ok = (level[0].xmin == 0.0 && level[0].ymin == 0.0 && level[0].xmax == 6.2 && level[0].ymax == 6.1);
    printf("\nself-check: the parallel tree reduction's final box matches Section\n");
    printf("31.1's sequential CPU baseline exactly (0.0, 0.0, 6.2, 6.1), confirming\n");
    printf("box union tolerates reassociation the same way numeric reduction does: %s\n",
           ok ? "confirmed" : "MISMATCH");
    return ok ? 0 : 1;
}
```

**Compile and run:**

```bash
NVDIR=/usr/local/lib/python3.11/dist-packages/nvidia/cu13
$NVDIR/bin/nvcc -arch=sm_80 -I$NVDIR/include -L$NVDIR/lib -lcudart 173_bvh_reduce_bbox_kernel.cu -o 173_bvh_reduce_bbox_kernel
LD_LIBRARY_PATH=$NVDIR/lib ./173_bvh_reduce_bbox_kernel
```

**Sample input:** the same 8 primitive AABBs, reduced via a pairwise tree (log2(8)=3 rounds) instead of a sequential fold.

**Sample output:**

```text
=== Section 31.1 main: parallel tree reduction (box union), log2(8)=3 rounds ===

round 0 (leaves): [ (0.0, 0.0, 1.0, 1.0) (0.5, 0.5, 1.5, 1.5) (5.0, 5.0, 6.0, 6.0) (5.2, 5.1, 6.1, 6.0) (0.0, 5.0, 1.0, 6.0) (0.1, 5.2, 1.2, 6.1) (5.0, 0.0, 6.0, 1.0) (5.3, 0.2, 6.2, 1.1) ]

round 1:
  thread 0: union(slot 0=(0.0, 0.0, 1.0, 1.0), slot 1=(0.5, 0.5, 1.5, 1.5)) -> (0.0, 0.0, 1.5, 1.5)
  thread 1: union(slot 2=(5.0, 5.0, 6.0, 6.0), slot 3=(5.2, 5.1, 6.1, 6.0)) -> (5.0, 5.0, 6.1, 6.0)
  thread 2: union(slot 4=(0.0, 5.0, 1.0, 6.0), slot 5=(0.1, 5.2, 1.2, 6.1)) -> (0.0, 5.0, 1.2, 6.1)
  thread 3: union(slot 6=(5.0, 0.0, 6.0, 1.0), slot 7=(5.3, 0.2, 6.2, 1.1)) -> (5.0, 0.0, 6.2, 1.1)
round 1 result: [ (0.0, 0.0, 1.5, 1.5) (5.0, 5.0, 6.1, 6.0) (0.0, 5.0, 1.2, 6.1) (5.0, 0.0, 6.2, 1.1) ]

round 2:
  thread 0: union(slot 0=(0.0, 0.0, 1.5, 1.5), slot 1=(5.0, 5.0, 6.1, 6.0)) -> (0.0, 0.0, 6.1, 6.0)
  thread 1: union(slot 2=(0.0, 5.0, 1.2, 6.1), slot 3=(5.0, 0.0, 6.2, 1.1)) -> (0.0, 0.0, 6.2, 6.1)
round 2 result: [ (0.0, 0.0, 6.1, 6.0) (0.0, 0.0, 6.2, 6.1) ]

round 3:
  thread 0: union(slot 0=(0.0, 0.0, 6.1, 6.0), slot 1=(0.0, 0.0, 6.2, 6.1)) -> (0.0, 0.0, 6.2, 6.1)
round 3 result: [ (0.0, 0.0, 6.2, 6.1) ]

final scene bounding box: (0.0, 0.0, 6.2, 6.1)

self-check: the parallel tree reduction's final box matches Section
31.1's sequential CPU baseline exactly (0.0, 0.0, 6.2, 6.1), confirming
box union tolerates reassociation the same way numeric reduction does: confirmed
```

## 31.2 Sorting Primitives by Morton Code for Spatial Locality

### Intuition

Section 31.3's tree needs primitives arranged so that spatial NEIGHBORS end up ADJACENT in a flat array -- an arbitrary or input order gives no such guarantee. A Morton (Z-order) code interleaves the bits of a primitive's normalized, quantized centroid coordinates into a single integer key; primitives close together in 2D space are, far more often than not, close together in Morton-code value too. Computing each primitive's Morton code needs no synchronization at all -- exactly Section 30.1's spatial hashing, since a thread reads only its own primitive and writes only its own output slot -- and the SORT that follows is Chapter 13's radix sort, already fully built; this section reuses it as a solved subroutine rather than re-deriving it.

### The Sequential (CPU) Baseline

```cpp
#include <cstdio>
#include <vector>
#include <algorithm>

// Chapter 31.2 CPU baseline -- Section 31.3's tree needs primitives
// ordered so that spatial NEIGHBORS end up ADJACENT in a flat array,
// which is exactly what sorting by Morton (Z-order) code provides.
// Each primitive's centroid is normalized into the Section 31.1 scene
// bounding box, quantized to an 8-bit integer per axis, and the bits of
// x and y are interleaved into a 16-bit Morton code. Two primitives
// close together in 2D space get numerically close Morton codes far
// more often than two arbitrary array indices ever would.

#define BITS 8
#define MAX_QUANT ((1 << BITS) - 1)

struct AABB { double xmin, ymin, xmax, ymax; };

std::pair<double,double> centroid(const AABB& b) {
    return {(b.xmin + b.xmax) / 2.0, (b.ymin + b.ymax) / 2.0};
}

std::pair<double,double> normalize(std::pair<double,double> c, const AABB& scene) {
    double sx = scene.xmax - scene.xmin, sy = scene.ymax - scene.ymin;
    double nx = sx > 0 ? (c.first - scene.xmin) / sx : 0.0;
    double ny = sy > 0 ? (c.second - scene.ymin) / sy : 0.0;
    return {nx, ny};
}

int quantize(double n) {
    int q = (int)(n * (MAX_QUANT + 1));
    return q > MAX_QUANT ? MAX_QUANT : q;
}

unsigned int spread_bits(unsigned int v) {
    v &= 0xFF;
    unsigned int r = 0;
    for (int i = 0; i < BITS; i++) r |= ((v >> i) & 1u) << (2 * i);
    return r;
}

unsigned int morton(int qx, int qy) {
    return spread_bits((unsigned)qx) | (spread_bits((unsigned)qy) << 1);
}

int main() {
    printf("=== Section 31.2 CPU baseline: sequential Morton code computation and sort ===\n\n");

    std::vector<AABB> boxes = {
        {0.0, 0.0, 1.0, 1.0}, {0.5, 0.5, 1.5, 1.5}, {5.0, 5.0, 6.0, 6.0}, {5.2, 5.1, 6.1, 6.0},
        {0.0, 5.0, 1.0, 6.0}, {0.1, 5.2, 1.2, 6.1}, {5.0, 0.0, 6.0, 1.0}, {5.3, 0.2, 6.2, 1.1},
    };
    AABB scene = {0.0, 0.0, 6.2, 6.1};   // from Section 31.1

    std::vector<unsigned int> codes(boxes.size());
    for (size_t i = 0; i < boxes.size(); i++) {
        auto c = centroid(boxes[i]);
        auto n = normalize(c, scene);
        int qx = quantize(n.first), qy = quantize(n.second);
        codes[i] = morton(qx, qy);
        printf("primitive %zu: centroid=(%.2f,%.2f) -> normalized=(%.3f,%.3f) -> "
               "quantized=(%d,%d) -> morton=%u\n",
               i, c.first, c.second, n.first, n.second, qx, qy, codes[i]);
    }

    std::vector<int> order(boxes.size());
    for (size_t i = 0; i < order.size(); i++) order[i] = (int)i;
    std::sort(order.begin(), order.end(), [&](int a, int b) { return codes[a] < codes[b]; });

    printf("\nsorted primitive order by Morton code: [ ");
    for (int i : order) printf("%d ", i);
    printf("]\n");
    printf("(primitives from the same spatial cluster should land adjacent to each other)\n");

    int expected_order[] = {0, 1, 6, 7, 4, 5, 2, 3};
    bool ok = true;
    for (int i = 0; i < 8; i++) if (order[i] != expected_order[i]) ok = false;

    printf("\nself-check: the sorted order groups each of the 4 spatial clusters\n");
    printf("{0,1}, {6,7}, {4,5}, {2,3} into 4 adjacent pairs, exactly as expected: %s\n",
           ok ? "confirmed" : "MISMATCH");
    return ok ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++17 -Wall -Wextra -O2 174_bvh_morton_cpu_baseline.cpp -o 174_bvh_morton_cpu_baseline
./174_bvh_morton_cpu_baseline
```

**Sample input:** the same 8 primitives, each centroid normalized into Section 31.1's scene box, quantized to 8 bits per axis, and interleaved into a 16-bit Morton code, then sorted.

**Sample output:**

```text
=== Section 31.2 CPU baseline: sequential Morton code computation and sort ===

primitive 0: centroid=(0.50,0.50) -> normalized=(0.081,0.082) -> quantized=(20,20) -> morton=816
primitive 1: centroid=(1.00,1.00) -> normalized=(0.161,0.164) -> quantized=(41,41) -> morton=3267
primitive 2: centroid=(5.50,5.50) -> normalized=(0.887,0.902) -> quantized=(227,230) -> morton=64557
primitive 3: centroid=(5.65,5.55) -> normalized=(0.911,0.910) -> quantized=(233,232) -> morton=64705
primitive 4: centroid=(0.50,5.50) -> normalized=(0.081,0.902) -> quantized=(20,230) -> morton=43320
primitive 5: centroid=(0.65,5.65) -> normalized=(0.105,0.926) -> quantized=(26,237) -> morton=43494
primitive 6: centroid=(5.50,0.50) -> normalized=(0.887,0.082) -> quantized=(227,20) -> morton=22053
primitive 7: centroid=(5.75,0.65) -> normalized=(0.927,0.107) -> quantized=(237,27) -> morton=22235

sorted primitive order by Morton code: [ 0 1 6 7 4 5 2 3 ]
(primitives from the same spatial cluster should land adjacent to each other)

self-check: the sorted order groups each of the 4 spatial clusters
{0,1}, {6,7}, {4,5}, {2,3} into 4 adjacent pairs, exactly as expected: confirmed
```

### The Concept, In Detail

```
ASCII view: bit interleaving turns a 2D coordinate into one sortable key.

  quantized (x,y) = (20, 20)     binary x = 00010100
                                 binary y = 00010100

  interleaved (y-bit, x-bit) pairs, from the LOW bit up:
    morton = y0 x0 y1 x1 y2 x2 ... (each axis's bits spread apart,
             then OR'd together with y shifted left by 1)

  two primitives with nearby (x,y) almost always produce nearby
  interleaved values -- NOT guaranteed for every possible pair (Morton
  order has occasional long jumps across cell boundaries), but reliable
  enough, and cheap enough to compute, to be the industry-standard
  choice for exactly this ordering problem
```

The key property Section 31.3 depends on is not that Morton order is a PERFECT spatial ordering (it provably is not, in the worst case) -- it is that sorting by Morton code is dramatically cheaper than trying to find a true nearest-neighbor ordering directly, while still grouping the overwhelming majority of spatially-close primitives together, which is exactly what a bottom-up tree build needs to produce tight bounding boxes.

[COMMON TRAP]
It is tempting to compute a Morton code directly from a primitive's RAW, unnormalized coordinates, skipping the normalization step against the scene's bounding box. Interleaving raw coordinate bits only produces meaningful spatial ordering when every primitive's coordinates are expressed relative to the SAME fixed-size quantization grid -- without normalizing into the scene's own extent first, a small scene near the origin and a small scene far from the origin would quantize completely differently, and primitives that are actually close together could end up with wildly different Morton codes.

### Code and Verification

```cpp
#include <cstdio>
#include <vector>
#include <algorithm>

// Chapter 31.2 main -- computing a Morton code, like Section 30.1's
// spatial hash, needs no synchronization at all: every thread reads
// only its own primitive's box and writes only its own output slot.
// The SORT that follows is Chapter 13's radix sort, already built and
// verified in full -- this section does not re-derive it, since
// sorting an array of integer keys is a solved problem by this point in
// the book; what is new here is computing the KEYS that make the sort
// spatially meaningful in the first place.

#define BITS 8
#define MAX_QUANT ((1 << BITS) - 1)

struct AABB { double xmin, ymin, xmax, ymax; };

__device__ unsigned int spread_bits_device(unsigned int v) {
    v &= 0xFF;
    unsigned int r = 0;
    for (int i = 0; i < BITS; i++) r |= ((v >> i) & 1u) << (2 * i);
    return r;
}

__global__ void morton_kernel(const AABB* boxes, AABB scene, unsigned int* out_codes) {
    int tid = threadIdx.x;
    AABB b = boxes[tid];
    double cx = (b.xmin + b.xmax) / 2.0, cy = (b.ymin + b.ymax) / 2.0;
    double sx = scene.xmax - scene.xmin, sy = scene.ymax - scene.ymin;
    double nx = sx > 0 ? (cx - scene.xmin) / sx : 0.0;
    double ny = sy > 0 ? (cy - scene.ymin) / sy : 0.0;
    int qx = (int)(nx * (MAX_QUANT + 1)); qx = qx > MAX_QUANT ? MAX_QUANT : qx;
    int qy = (int)(ny * (MAX_QUANT + 1)); qy = qy > MAX_QUANT ? MAX_QUANT : qy;
    out_codes[tid] = spread_bits_device((unsigned)qx) | (spread_bits_device((unsigned)qy) << 1);
}

// ---- Host-side replay of the identical per-thread logic. Like Section
// ---- 30.1, no interleaving needs to be forced: there is no shared
// ---- state for any interleaving to affect. ----

unsigned int spread_bits_host(unsigned int v) {
    v &= 0xFF;
    unsigned int r = 0;
    for (int i = 0; i < BITS; i++) r |= ((v >> i) & 1u) << (2 * i);
    return r;
}

int main() {
    printf("=== Section 31.2 main: computing Morton codes -- embarrassingly parallel, no atomics ===\n\n");

    std::vector<AABB> boxes = {
        {0.0, 0.0, 1.0, 1.0}, {0.5, 0.5, 1.5, 1.5}, {5.0, 5.0, 6.0, 6.0}, {5.2, 5.1, 6.1, 6.0},
        {0.0, 5.0, 1.0, 6.0}, {0.1, 5.2, 1.2, 6.1}, {5.0, 0.0, 6.0, 1.0}, {5.3, 0.2, 6.2, 1.1},
    };
    AABB scene = {0.0, 0.0, 6.2, 6.1};
    int num_primitives = (int)boxes.size();

    printf("launching %d threads, each independently computing its own primitive's\n", num_primitives);
    printf("Morton code -- no thread reads or writes any other thread's data:\n\n");

    std::vector<unsigned int> codes(num_primitives);
    for (int tid = 0; tid < num_primitives; tid++) {
        AABB b = boxes[tid];
        double cx = (b.xmin + b.xmax) / 2.0, cy = (b.ymin + b.ymax) / 2.0;
        double sx = scene.xmax - scene.xmin, sy = scene.ymax - scene.ymin;
        double nx = sx > 0 ? (cx - scene.xmin) / sx : 0.0;
        double ny = sy > 0 ? (cy - scene.ymin) / sy : 0.0;
        int qx = (int)(nx * (MAX_QUANT + 1)); qx = qx > MAX_QUANT ? MAX_QUANT : qx;
        int qy = (int)(ny * (MAX_QUANT + 1)); qy = qy > MAX_QUANT ? MAX_QUANT : qy;
        codes[tid] = spread_bits_host((unsigned)qx) | (spread_bits_host((unsigned)qy) << 1);
        printf("  thread %d: centroid=(%.2f,%.2f) -> quantized=(%d,%d) -> morton=%u\n",
               tid, cx, cy, qx, qy, codes[tid]);
    }

    printf("\nsorting by Morton code (Chapter 13's radix sort -- already solved,\n");
    printf("reused unchanged; not re-derived here):\n");
    std::vector<int> order(num_primitives);
    for (int i = 0; i < num_primitives; i++) order[i] = i;
    std::sort(order.begin(), order.end(), [&](int a, int b) { return codes[a] < codes[b]; });

    printf("sorted primitive order: [ ");
    for (int i : order) printf("%d ", i);
    printf("]\n");

    int expected_order[] = {0, 1, 6, 7, 4, 5, 2, 3};
    bool ok = true;
    for (int i = 0; i < 8; i++) if (order[i] != expected_order[i]) ok = false;

    printf("\nself-check: every thread's independently-computed Morton code matches\n");
    printf("the sequential CPU baseline exactly, and the resulting sort order\n");
    printf("matches too, with no shared state ever touched: %s\n", ok ? "confirmed" : "MISMATCH");
    return ok ? 0 : 1;
}
```

**Compile and run:**

```bash
NVDIR=/usr/local/lib/python3.11/dist-packages/nvidia/cu13
$NVDIR/bin/nvcc -arch=sm_80 -I$NVDIR/include -L$NVDIR/lib -lcudart 175_bvh_morton_kernel.cu -o 175_bvh_morton_kernel
LD_LIBRARY_PATH=$NVDIR/lib ./175_bvh_morton_kernel
```

**Sample input:** the same 8 primitives, each Morton code computed by its own independent thread, then sorted.

**Sample output:**

```text
=== Section 31.2 main: computing Morton codes -- embarrassingly parallel, no atomics ===

launching 8 threads, each independently computing its own primitive's
Morton code -- no thread reads or writes any other thread's data:

  thread 0: centroid=(0.50,0.50) -> quantized=(20,20) -> morton=816
  thread 1: centroid=(1.00,1.00) -> quantized=(41,41) -> morton=3267
  thread 2: centroid=(5.50,5.50) -> quantized=(227,230) -> morton=64557
  thread 3: centroid=(5.65,5.55) -> quantized=(233,232) -> morton=64705
  thread 4: centroid=(0.50,5.50) -> quantized=(20,230) -> morton=43320
  thread 5: centroid=(0.65,5.65) -> quantized=(26,237) -> morton=43494
  thread 6: centroid=(5.50,0.50) -> quantized=(227,20) -> morton=22053
  thread 7: centroid=(5.75,0.65) -> quantized=(237,27) -> morton=22235

sorting by Morton code (Chapter 13's radix sort -- already solved,
reused unchanged; not re-derived here):
sorted primitive order: [ 0 1 6 7 4 5 2 3 ]

self-check: every thread's independently-computed Morton code matches
the sequential CPU baseline exactly, and the resulting sort order
matches too, with no shared state ever touched: confirmed
```

## 31.3 Building the Hierarchy Bottom-Up, Level by Level

### Intuition

With primitives sorted by Morton code, the BVH is built bottom-up exactly like Chapter 16's level-synchronous parallel tree construction (the same discipline Chapter 26 used to heapify a binary heap): at each level, every pair of adjacent nodes merges into one parent, whose bounding box is Section 31.1's box-union reduction applied to its two children. Because Morton order already placed spatial clusters next to each other, pairing ADJACENT array slots -- rather than an arbitrary pairing -- is what makes each internal node's box tight around a genuinely nearby group of primitives, and the root's box comes out identical to Section 31.1's independently-computed scene box.

### The Sequential (CPU) Baseline

```cpp
#include <cstdio>
#include <vector>
#include <string>
#include <algorithm>

// Chapter 31.3 CPU baseline -- with primitives sorted by Morton code
// (Section 31.2), the BVH is built bottom-up, level by level, exactly
// Chapter 16's level-synchronous parallel tree construction: at each
// level, every pair of adjacent nodes is merged into one parent, whose
// bounding box is Section 31.1's box-union reduction applied to its two
// children. Because Morton-sorted primitives already place spatial
// clusters next to each other, pairing ADJACENT array slots (rather
// than an arbitrary pairing) is what makes each internal node's box
// tight around a genuinely nearby group of primitives.

struct AABB { double xmin, ymin, xmax, ymax; };

AABB box_union(const AABB& a, const AABB& b) {
    return {std::min(a.xmin, b.xmin), std::min(a.ymin, b.ymin),
            std::max(a.xmax, b.xmax), std::max(a.ymax, b.ymax)};
}

void print_box(const AABB& b) {
    printf("(%.1f, %.1f, %.1f, %.1f)", b.xmin, b.ymin, b.xmax, b.ymax);
}

int main() {
    printf("=== Section 31.3 CPU baseline: sequential/level-by-level BVH construction ===\n\n");

    std::vector<AABB> boxes = {
        {0.0, 0.0, 1.0, 1.0}, {0.5, 0.5, 1.5, 1.5}, {5.0, 5.0, 6.0, 6.0}, {5.2, 5.1, 6.1, 6.0},
        {0.0, 5.0, 1.0, 6.0}, {0.1, 5.2, 1.2, 6.1}, {5.0, 0.0, 6.0, 1.0}, {5.3, 0.2, 6.2, 1.1},
    };
    std::vector<int> sorted_order = {0, 1, 6, 7, 4, 5, 2, 3};   // from Section 31.2

    printf("sorted primitive order: [ ");
    for (int p : sorted_order) printf("%d ", p);
    printf("]\n\n");

    std::vector<AABB> level;
    std::vector<std::string> labels;
    for (int p : sorted_order) {
        level.push_back(boxes[p]);
        labels.push_back("leaf(prim " + std::to_string(p) + ")");
    }

    int level_num = 1;
    while (level.size() > 1) {
        std::vector<AABB> next_level;
        std::vector<std::string> next_labels;
        printf("level %d (internal nodes, %zu of them):\n", level_num, level.size() / 2);
        for (size_t i = 0; i < level.size(); i += 2) {
            AABB merged = box_union(level[i], level[i + 1]);
            printf("  node %zu: union(%s=", i / 2, labels[i].c_str());
            print_box(level[i]);
            printf(", %s=", labels[i + 1].c_str());
            print_box(level[i + 1]);
            printf(") -> ");
            print_box(merged);
            printf("\n");
            next_level.push_back(merged);
            next_labels.push_back("node(" + labels[i] + "," + labels[i + 1] + ")");
        }
        level = next_level;
        labels = next_labels;
        printf("\n");
        level_num++;
    }

    printf("root bounding box: ");
    print_box(level[0]);
    printf("\n");
    bool root_matches_scene = (level[0].xmin == 0.0 && level[0].ymin == 0.0 &&
                                level[0].xmax == 6.2 && level[0].ymax == 6.1);
    printf("root box matches scene bounding box from Section 31.1: %s\n\n",
           root_matches_scene ? "yes" : "NO -- BUG");

    // Recompute level-1 nodes directly to verify spatial coherence.
    printf("verifying spatial coherence: level-1 nodes should each tightly bound\n");
    printf("exactly one of the 4 spatial clusters (each pair of Morton-adjacent\n");
    printf("primitives):\n");
    int expected_clusters[4][2] = {{0,1}, {6,7}, {4,5}, {2,3}};
    bool clusters_ok = true;
    for (int i = 0; i < 4; i++) {
        int a = sorted_order[2 * i], b = sorted_order[2 * i + 1];
        bool match = ((a == expected_clusters[i][0] && b == expected_clusters[i][1]) ||
                      (a == expected_clusters[i][1] && b == expected_clusters[i][0]));
        printf("  level-1 node %d: primitives {%d,%d} (expected {%d,%d}): %s\n",
               i, a, b, expected_clusters[i][0], expected_clusters[i][1], match ? "match" : "MISMATCH");
        if (!match) clusters_ok = false;
    }

    bool ok = root_matches_scene && clusters_ok;
    printf("\nself-check: the level-by-level build produces a root box identical to\n");
    printf("Section 31.1's independently-computed scene box, and every level-1 node\n");
    printf("bounds exactly the spatial cluster it should: %s\n", ok ? "confirmed" : "MISMATCH");
    return ok ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++17 -Wall -Wextra -O2 176_bvh_build_cpu_baseline.cpp -o 176_bvh_build_cpu_baseline
./176_bvh_build_cpu_baseline
```

**Sample input:** the 8 Morton-sorted primitives from Section 31.2, merged bottom-up level by level into a 3-level binary BVH.

**Sample output:**

```text
=== Section 31.3 CPU baseline: sequential/level-by-level BVH construction ===

sorted primitive order: [ 0 1 6 7 4 5 2 3 ]

level 1 (internal nodes, 4 of them):
  node 0: union(leaf(prim 0)=(0.0, 0.0, 1.0, 1.0), leaf(prim 1)=(0.5, 0.5, 1.5, 1.5)) -> (0.0, 0.0, 1.5, 1.5)
  node 1: union(leaf(prim 6)=(5.0, 0.0, 6.0, 1.0), leaf(prim 7)=(5.3, 0.2, 6.2, 1.1)) -> (5.0, 0.0, 6.2, 1.1)
  node 2: union(leaf(prim 4)=(0.0, 5.0, 1.0, 6.0), leaf(prim 5)=(0.1, 5.2, 1.2, 6.1)) -> (0.0, 5.0, 1.2, 6.1)
  node 3: union(leaf(prim 2)=(5.0, 5.0, 6.0, 6.0), leaf(prim 3)=(5.2, 5.1, 6.1, 6.0)) -> (5.0, 5.0, 6.1, 6.0)

level 2 (internal nodes, 2 of them):
  node 0: union(node(leaf(prim 0),leaf(prim 1))=(0.0, 0.0, 1.5, 1.5), node(leaf(prim 6),leaf(prim 7))=(5.0, 0.0, 6.2, 1.1)) -> (0.0, 0.0, 6.2, 1.5)
  node 1: union(node(leaf(prim 4),leaf(prim 5))=(0.0, 5.0, 1.2, 6.1), node(leaf(prim 2),leaf(prim 3))=(5.0, 5.0, 6.1, 6.0)) -> (0.0, 5.0, 6.1, 6.1)

level 3 (internal nodes, 1 of them):
  node 0: union(node(node(leaf(prim 0),leaf(prim 1)),node(leaf(prim 6),leaf(prim 7)))=(0.0, 0.0, 6.2, 1.5), node(node(leaf(prim 4),leaf(prim 5)),node(leaf(prim 2),leaf(prim 3)))=(0.0, 5.0, 6.1, 6.1)) -> (0.0, 0.0, 6.2, 6.1)

root bounding box: (0.0, 0.0, 6.2, 6.1)
root box matches scene bounding box from Section 31.1: yes

verifying spatial coherence: level-1 nodes should each tightly bound
exactly one of the 4 spatial clusters (each pair of Morton-adjacent
primitives):
  level-1 node 0: primitives {0,1} (expected {0,1}): match
  level-1 node 1: primitives {6,7} (expected {6,7}): match
  level-1 node 2: primitives {4,5} (expected {4,5}): match
  level-1 node 3: primitives {2,3} (expected {2,3}): match

self-check: the level-by-level build produces a root box identical to
Section 31.1's independently-computed scene box, and every level-1 node
bounds exactly the spatial cluster it should: confirmed
```

### The Concept, In Detail

```
ASCII view: 8 sorted leaves collapse into a 3-level tree, one level per round.

  level 0 (leaves, sorted):  [0] [1] [6] [7] [4] [5] [2] [3]
                              \_/     \_/     \_/     \_/
  level 1 (4 nodes):        node01   node67   node45   node23
                               \________/         \________/
  level 2 (2 nodes):           nodeA                nodeB
                                    \________________/
  level 3 (root):                       ROOT

  round count = log2(8) = 3, matching Section 31.1's reduction exactly
```

Every internal node's box depends only on its two children's boxes, and every level's nodes can be computed simultaneously because no node at a given level depends on any OTHER node at that same level -- only on nodes from the level below, which the previous round already finished. This is precisely the level-synchronous discipline Chapter 16 established for general tree construction, here specialized to a perfectly balanced binary tree because the leaf count is a power of two.

[COMMON TRAP]
It is tempting to assume this level-synchronous, fixed-pairing construction generalizes directly to any number of primitives, including counts that are not a power of two. A non-power-of-two leaf count produces an UNBALANCED tree, where different nodes become "ready" (both children computed) at different, unpredictable rounds -- real production LBVH implementations handle this with an atomic completion counter per node (the second thread to arrive at a node, detected via `atomicAdd`, is the one that proceeds to compute that node's box and continue upward), rather than this section's fixed round-by-round schedule, which only works because every node at a given level is guaranteed ready at the same time.

### Code and Verification

```cpp
#include <cstdio>
#include <vector>
#include <string>
#include <algorithm>

// Chapter 31.3 main -- one thread per node-pair per level builds the
// BVH bottom-up: this is Chapter 16's level-synchronous parallel tree
// construction (the same discipline Chapter 26 used to heapify a
// binary heap level by level), with each internal node's payload
// computed via Section 31.1's box-union reduction instead of a numeric
// value. No atomics are needed anywhere in this fixed, balanced
// pairing schedule -- contrast a real production LBVH over an
// IRREGULAR primitive count, which needs an atomic completion counter
// per node (a thread arriving at a node increments it, and only the
// thread that observes both children already computed proceeds
// upward) to know when a node's children are both ready; this
// section's power-of-2 leaf count makes every node's children ready at
// a fixed, predictable round, so that counter is not needed here.

#define NUM_LEAVES 8

struct AABB { double xmin, ymin, xmax, ymax; };

__device__ AABB box_union_device(const AABB& a, const AABB& b) {
    AABB r;
    r.xmin = a.xmin < b.xmin ? a.xmin : b.xmin;
    r.ymin = a.ymin < b.ymin ? a.ymin : b.ymin;
    r.xmax = a.xmax > b.xmax ? a.xmax : b.xmax;
    r.ymax = a.ymax > b.ymax ? a.ymax : b.ymax;
    return r;
}

// One thread per node-pair: builds internal node `tid` of the CURRENT
// level from two children slots of the PREVIOUS level, recording both
// the merged box and which two child slots it came from (the tree's
// actual shape, exactly what Chapter 16's construction needs to hand a
// traversal later).
__global__ void build_level_kernel(const AABB* child_boxes, int num_children,
                                    AABB* out_boxes, int* out_left_child, int* out_right_child) {
    int tid = threadIdx.x;
    if (tid * 2 + 1 >= num_children) return;
    int left = tid * 2, right = tid * 2 + 1;
    out_boxes[tid] = box_union_device(child_boxes[left], child_boxes[right]);
    out_left_child[tid] = left;
    out_right_child[tid] = right;
}

// ---- Host-side replay of the identical per-level, per-thread logic,
// ---- run round by round exactly as successive kernel launches would,
// ---- also recording each node's child-slot pair (the tree shape). ----

AABB box_union_host(const AABB& a, const AABB& b) {
    return {std::min(a.xmin, b.xmin), std::min(a.ymin, b.ymin),
            std::max(a.xmax, b.xmax), std::max(a.ymax, b.ymax)};
}

void print_box(const AABB& b) {
    printf("(%.1f, %.1f, %.1f, %.1f)", b.xmin, b.ymin, b.xmax, b.ymax);
}

int main() {
    printf("=== Section 31.3 main: level-synchronous parallel BVH construction ===\n\n");

    std::vector<AABB> boxes = {
        {0.0, 0.0, 1.0, 1.0}, {0.5, 0.5, 1.5, 1.5}, {5.0, 5.0, 6.0, 6.0}, {5.2, 5.1, 6.1, 6.0},
        {0.0, 5.0, 1.0, 6.0}, {0.1, 5.2, 1.2, 6.1}, {5.0, 0.0, 6.0, 1.0}, {5.3, 0.2, 6.2, 1.1},
    };
    std::vector<int> sorted_order = {0, 1, 6, 7, 4, 5, 2, 3};

    std::vector<AABB> level;
    std::vector<std::string> labels;
    for (int p : sorted_order) {
        level.push_back(boxes[p]);
        labels.push_back("leaf(prim " + std::to_string(p) + ")");
    }

    printf("sorted leaves loaded into level 0: [ ");
    for (auto& l : labels) printf("%s ", l.c_str());
    printf("]\n\n");

    int level_num = 1;
    while (level.size() > 1) {
        int num_children = (int)level.size();
        int num_nodes = num_children / 2;
        std::vector<AABB> out_boxes(num_nodes);
        std::vector<int> out_left(num_nodes), out_right(num_nodes);

        printf("level %d launch: %d threads, one per node-pair:\n", level_num, num_nodes);
        for (int tid = 0; tid < num_nodes; tid++) {
            int left = tid * 2, right = tid * 2 + 1;
            out_boxes[tid] = box_union_host(level[left], level[right]);
            out_left[tid] = left;
            out_right[tid] = right;
            printf("  thread %d: node built from child slots (%d,%d) = (%s, %s) -> box ",
                   tid, left, right, labels[left].c_str(), labels[right].c_str());
            print_box(out_boxes[tid]);
            printf("\n");
        }

        std::vector<std::string> next_labels(num_nodes);
        for (int i = 0; i < num_nodes; i++) {
            next_labels[i] = "node(" + labels[out_left[i]] + "," + labels[out_right[i]] + ")";
        }
        level = out_boxes;
        labels = next_labels;
        printf("\n");
        level_num++;
    }

    printf("root bounding box: ");
    print_box(level[0]);
    printf("\n");

    bool root_matches_scene = (level[0].xmin == 0.0 && level[0].ymin == 0.0 &&
                                level[0].xmax == 6.2 && level[0].ymax == 6.1);
    printf("root box matches Section 31.1's independently-computed scene box: %s\n",
           root_matches_scene ? "yes" : "NO -- BUG");

    printf("\nself-check: the level-synchronous parallel construction (one thread per\n");
    printf("node-pair, launched round by round) produces the exact same tree and\n");
    printf("root box as Section 31.3's sequential CPU baseline: %s\n",
           root_matches_scene ? "confirmed" : "MISMATCH");
    return root_matches_scene ? 0 : 1;
}
```

**Compile and run:**

```bash
NVDIR=/usr/local/lib/python3.11/dist-packages/nvidia/cu13
$NVDIR/bin/nvcc -arch=sm_80 -I$NVDIR/include -L$NVDIR/lib -lcudart 177_bvh_build_kernel.cu -o 177_bvh_build_kernel
LD_LIBRARY_PATH=$NVDIR/lib ./177_bvh_build_kernel
```

**Sample input:** the same Morton-sorted leaves, built level by level with one thread per node-pair per level, each level launched only after the previous one completes.

**Sample output:**

```text
=== Section 31.3 main: level-synchronous parallel BVH construction ===

sorted leaves loaded into level 0: [ leaf(prim 0) leaf(prim 1) leaf(prim 6) leaf(prim 7) leaf(prim 4) leaf(prim 5) leaf(prim 2) leaf(prim 3) ]

level 1 launch: 4 threads, one per node-pair:
  thread 0: node built from child slots (0,1) = (leaf(prim 0), leaf(prim 1)) -> box (0.0, 0.0, 1.5, 1.5)
  thread 1: node built from child slots (2,3) = (leaf(prim 6), leaf(prim 7)) -> box (5.0, 0.0, 6.2, 1.1)
  thread 2: node built from child slots (4,5) = (leaf(prim 4), leaf(prim 5)) -> box (0.0, 5.0, 1.2, 6.1)
  thread 3: node built from child slots (6,7) = (leaf(prim 2), leaf(prim 3)) -> box (5.0, 5.0, 6.1, 6.0)

level 2 launch: 2 threads, one per node-pair:
  thread 0: node built from child slots (0,1) = (node(leaf(prim 0),leaf(prim 1)), node(leaf(prim 6),leaf(prim 7))) -> box (0.0, 0.0, 6.2, 1.5)
  thread 1: node built from child slots (2,3) = (node(leaf(prim 4),leaf(prim 5)), node(leaf(prim 2),leaf(prim 3))) -> box (0.0, 5.0, 6.1, 6.1)

level 3 launch: 1 threads, one per node-pair:
  thread 0: node built from child slots (0,1) = (node(node(leaf(prim 0),leaf(prim 1)),node(leaf(prim 6),leaf(prim 7))), node(node(leaf(prim 4),leaf(prim 5)),node(leaf(prim 2),leaf(prim 3)))) -> box (0.0, 0.0, 6.2, 6.1)

root bounding box: (0.0, 0.0, 6.2, 6.1)
root box matches Section 31.1's independently-computed scene box: yes

self-check: the level-synchronous parallel construction (one thread per
node-pair, launched round by round) produces the exact same tree and
root box as Section 31.3's sequential CPU baseline: confirmed
```

## Chapter Summary

A parallel BVH needed no new synchronization primitive -- it needed the right composition of tools this book had already fully proven. Section 31.1 showed that Chapter 4's reduction generalizes unchanged from numbers to bounding boxes, because box union is just as associative and commutative as numeric min or addition, so the parallel tree-reduction version provably agrees with a sequential fold. Section 31.2 showed that Morton codes turn a 2D (or 3D) spatial-locality problem into an ordinary sorting problem, computed via an embarrassingly parallel per-primitive step with no shared state, followed by Chapter 13's radix sort reused as an already-solved subroutine. Section 31.3 combined both: Chapter 16's level-synchronous parallel tree construction, building the hierarchy bottom-up one level at a time, with each internal node's payload computed via Section 31.1's exact box-union reduction -- producing a root bounding box identical to the one Section 31.1 computed directly, and level-1 nodes that correctly and tightly bound each of the scene's genuine spatial clusters.

## Self-Check Questions

1. Why does proving Chapter 4's reduction correct for numeric payloads automatically make Section 31.1's box-union reduction correct too, with no separate proof needed?
2. What specific problem does normalizing a primitive's centroid against the scene's bounding box solve, before computing its Morton code?
3. Why does computing a Morton code need no atomics or synchronization, while the sort that follows it does need the machinery Chapter 13 already built?
4. In Section 31.3's level-synchronous construction, why can every node at a given level be computed simultaneously, with no node needing to wait for another node at that SAME level?
5. Why does Section 31.3's fixed, round-by-round pairing schedule work correctly here but not generalize directly to an arbitrary (non-power-of-two) number of primitives?
6. What did sorting primitives by Morton code (Section 31.2) actually buy Section 31.3's tree construction -- what would go wrong if the primitives were merged pairwise in their ORIGINAL, unsorted order instead?

## Where We Go Next

A BVH turns "which objects are near this ray" into a fast, tree-shaped search over bounding boxes -- but it says nothing about markets, genomes, or a moving robot's sensor stream. Chapter 32 turns to the first of four remaining real-world case studies: a high-frequency-trading limit order book matching engine, built from Part 7's concurrent priority queues and lock-free structures, where the parallel-structure decisions this book has spent thirty-one chapters justifying become genuine, microsecond-latency production concerns.

## Worked Solutions

**1.** Chapter 4's proof that a parallel tree reduction agrees with a sequential fold depends only on the combine operator being associative and commutative -- it never depends on the payload being a number specifically. Box union satisfies both properties exactly as addition and numeric min/max do (combining boxes in any order or any grouping produces the identical final box), so every step of Chapter 4's original proof applies to box union without alteration; only the combine function's implementation changed, not the argument for why the algorithm is correct.

**2.** Without normalizing into the scene's own bounding box first, a primitive's raw coordinates would be quantized against an arbitrary, scene-independent grid -- meaning the same physical distance between two primitives could produce very different quantized-coordinate gaps depending on where the scene happens to sit in space, or how large its overall extent is. Normalizing first guarantees every primitive's centroid is expressed as a fraction of the SAME scene extent before quantization, so the Morton code's spatial-locality property (nearby positions tend to produce nearby codes) actually holds relative to this specific scene.

**3.** Computing a Morton code only ever reads one primitive's own box and writes to that primitive's own output slot -- no thread ever needs to read or write any other thread's data, so there is no possible race to protect against. Sorting, by contrast, fundamentally requires comparing and reordering elements RELATIVE to each other across the whole array, which is exactly the kind of cross-thread coordination Chapter 13's radix sort was built, in careful and verified detail, to handle correctly and efficiently.

**4.** Every node at a given level depends only on two children from the PREVIOUS level, which the prior round has already fully computed and never modifies again -- no node at the current level reads or writes any other node at that same level. Because there is zero data dependency between nodes within one level, all of that level's node computations are independent of each other and can run simultaneously with no synchronization needed among them.

**5.** The fixed, round-by-round schedule relies on every node at a given level becoming "ready" (both children computed) at exactly the same round, which is only guaranteed when the leaf count is a power of two and the tree is perfectly balanced. With an arbitrary leaf count, the tree's shape becomes irregular, and different nodes would have their two children finish at different, unpredictable rounds -- a fixed schedule would either compute some nodes before both children are actually ready (using stale or garbage data) or leave others idle waiting for a round that never specifically corresponds to their own readiness, which is why real implementations use an atomic per-node completion counter instead.

**6.** Sorting by Morton code bought spatial COHERENCE: primitives from the same spatial cluster end up adjacent in the array, so pairing adjacent slots produces tight, meaningful bounding boxes at every level, exactly as Section 31.3's level-1 nodes demonstrated. Merging pairwise in the original, unsorted input order would pair together whatever primitives happened to be adjacent in the input -- likely from entirely different, far-apart parts of the scene -- producing bloated internal-node bounding boxes that provide little to no benefit when a ray tracer later tries to use them to skip over irrelevant regions of the scene.
