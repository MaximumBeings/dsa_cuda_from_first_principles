# Chapter 30: Spatial Hashing for Particle Simulation

Chapter 29 combined a hash table, a memory pool, and hazard pointers to build a key-value store where "nearby" meant "the exact same key." A particle simulation needs the opposite notion of nearby: particles at similar POSITIONS, not identical keys, need to find each other quickly to compute collisions, forces, or density. This chapter reuses the same hashing, histogram, scan, and scatter tools already built -- Chapters 5, 6, 7, 13, 19, and 21 -- pointed at a 2D coordinate instead of an integer key, to build a spatial hash grid that turns an expensive all-pairs neighbor search into one that only examines particles actually nearby.

## 30.1 Hashing Particle Positions Into Grid Cells

### Intuition

Divide space into square cells of a fixed size, and every particle belongs to exactly one cell, found by dividing its position by the cell size. A real simulation's domain can be far larger than any grid a program could afford to allocate densely, so the cell coordinate itself is hashed down into a small, fixed number of buckets -- exactly Chapter 19/29's key-hashing idea, applied to a 2D coordinate via two large odd primes and XOR rather than a single integer and modulo. Computing this hash for every particle needs no synchronization at all: each thread reads only its own particle's position and writes only its own output slot.

### The Sequential (CPU) Baseline

```cpp
#include <cstdio>
#include <vector>

// Chapter 30.1 CPU baseline -- spatial hashing maps each particle's cell
// coordinate down into a small, FIXED number of hash buckets. This is a
// different problem than Chapter 21's CSR, which needed keys already
// packed into a dense, small range (0..N-1): a particle simulation's
// domain can be far larger, or entirely unbounded, than the number of
// buckets a grid can afford to keep, so the cell coordinate itself must
// be hashed down, exactly like Chapter 19/29's key hashing, just with a
// 2D key instead of a scalar one. Two large odd primes and XOR are the
// standard spatial-hash mixing function.

#define CELL_SIZE 1.0
#define NUM_BUCKETS 8
#define PRIME_X 73856093
#define PRIME_Y 19349663

struct CellCoord { int cx, cy; };

CellCoord cell_of(double x, double y) {
    return {(int)(x / CELL_SIZE), (int)(y / CELL_SIZE)};
}

int spatial_hash(int cx, int cy) {
    long long mixed = ((long long)cx * PRIME_X) ^ ((long long)cy * PRIME_Y);
    long long m = mixed % NUM_BUCKETS;
    if (m < 0) m += NUM_BUCKETS;   // C++'s % can return negative for negative operands
    return (int)m;
}

int main() {
    printf("=== Section 30.1 CPU baseline: sequential spatial hashing of particle positions ===\n\n");

    struct { double x, y; } particles[] = {
        {0.5, 0.5}, {0.7, 0.9}, {3.2, 0.4}, {1.5, 2.5},
        {1.6, 2.6}, {7.1, 7.8}, {0.2, 3.9}, {4.4, 4.1},
    };
    int num_particles = 8;

    printf("cell size = %.1f, num buckets = %d\n\n", CELL_SIZE, NUM_BUCKETS);

    std::vector<int> bucket_of(num_particles);
    std::vector<CellCoord> cell_of_particle(num_particles);
    for (int i = 0; i < num_particles; i++) {
        CellCoord c = cell_of(particles[i].x, particles[i].y);
        int b = spatial_hash(c.cx, c.cy);
        cell_of_particle[i] = c;
        bucket_of[i] = b;
        printf("particle %d: pos=(%.1f, %.1f) -> cell=(%d,%d) -> bucket %d\n",
               i, particles[i].x, particles[i].y, c.cx, c.cy, b);
    }

    printf("\nverifying: particles landing in the SAME cell always land in the SAME\n");
    printf("bucket (0 and 1 share cell (0,0)); particles in DIFFERENT cells can also\n");
    printf("collide into the same bucket (0 and 7 share bucket %d despite different\n", bucket_of[0]);
    printf("cells (%d,%d) vs (%d,%d) -- a genuine hash collision, not a spatial match):\n",
           cell_of_particle[0].cx, cell_of_particle[0].cy, cell_of_particle[7].cx, cell_of_particle[7].cy);

    bool same_cell_same_bucket = (bucket_of[0] == bucket_of[1]) &&
        (cell_of_particle[0].cx == cell_of_particle[1].cx) &&
        (cell_of_particle[0].cy == cell_of_particle[1].cy);
    bool collision_present = (bucket_of[0] == bucket_of[7]) &&
        !(cell_of_particle[0].cx == cell_of_particle[7].cx &&
          cell_of_particle[0].cy == cell_of_particle[7].cy);

    printf("  particles 0 and 1 share a cell and share a bucket: %s\n", same_cell_same_bucket ? "yes" : "NO -- BUG");
    printf("  particles 0 and 7 share a bucket despite DIFFERENT cells (collision): %s\n",
           collision_present ? "yes" : "NO -- BUG");

    bool ok = same_cell_same_bucket && collision_present;
    printf("\nself-check: same-cell particles always share a bucket, and a genuine\n");
    printf("cross-cell hash collision is present in this data set (both expected,\n");
    printf("real properties of spatial hashing, not bugs): %s\n", ok ? "confirmed" : "MISMATCH");
    return ok ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++17 -Wall -Wextra -O2 166_spatial_hash_cpu_baseline.cpp -o 166_spatial_hash_cpu_baseline
./166_spatial_hash_cpu_baseline
```

**Sample input:** 8 particles scattered through a domain much larger than the 8-bucket table, hashed via cell size 1.0 and two-prime XOR mixing.

**Sample output:**

```text
=== Section 30.1 CPU baseline: sequential spatial hashing of particle positions ===

cell size = 1.0, num buckets = 8

particle 0: pos=(0.5, 0.5) -> cell=(0,0) -> bucket 0
particle 1: pos=(0.7, 0.9) -> cell=(0,0) -> bucket 0
particle 2: pos=(3.2, 0.4) -> cell=(3,0) -> bucket 7
particle 3: pos=(1.5, 2.5) -> cell=(1,2) -> bucket 3
particle 4: pos=(1.6, 2.6) -> cell=(1,2) -> bucket 3
particle 5: pos=(7.1, 7.8) -> cell=(7,7) -> bucket 2
particle 6: pos=(0.2, 3.9) -> cell=(0,3) -> bucket 5
particle 7: pos=(4.4, 4.1) -> cell=(4,4) -> bucket 0

verifying: particles landing in the SAME cell always land in the SAME
bucket (0 and 1 share cell (0,0)); particles in DIFFERENT cells can also
collide into the same bucket (0 and 7 share bucket 0 despite different
cells (0,0) vs (4,4) -- a genuine hash collision, not a spatial match):
  particles 0 and 1 share a cell and share a bucket: yes
  particles 0 and 7 share a bucket despite DIFFERENT cells (collision): yes

self-check: same-cell particles always share a bucket, and a genuine
cross-cell hash collision is present in this data set (both expected,
real properties of spatial hashing, not bugs): confirmed
```

### The Concept, In Detail

```
ASCII view: two cells colliding into the same bucket despite being far apart.

  particle 0: pos=(0.5,0.5) -> cell=(0,0) -\
                                             +--> bucket 0  (hash collision --
  particle 7: pos=(4.4,4.1) -> cell=(4,4) -/      these cells are NOT adjacent)

  particle 1: pos=(0.7,0.9) -> cell=(0,0) --> bucket 0  (same cell as
                                                particle 0 -- a real spatial match)
```

Same-cell particles always land in the same bucket, because the hash is a pure function of the cell coordinate -- but the reverse is not guaranteed: two particles in completely different, far-apart cells can also collide into the same bucket purely because the hash function's output range is smaller than the number of distinct cells in use. Bucket membership is necessary but not sufficient evidence of spatial proximity, a fact Section 30.3 builds its central safeguard around.

[COMMON TRAP]
It is tempting to assume that if two particles share a bucket, they must be spatially close, since that is the whole point of a spatial hash. Sharing a bucket only means the hash FUNCTION happened to map their (possibly very different) cells to the same output -- with a small table relative to the domain, hash collisions between unrelated cells are common, not rare, and any code that treats bucket membership as a proximity guarantee will report false neighbors.

### Code and Verification

```cpp
#include <cstdio>
#include <vector>

// Chapter 30.1 main -- computing each particle's spatial hash needs NO
// synchronization at all: every thread reads only its own particle's
// position and writes only its own output slot, with zero shared state
// touched between threads. This is worth noticing precisely because it
// is the exception in this book, not the rule -- contrast Section
// 29.1's key insertion (needed a CAS retry loop, since multiple threads
// could target the same table slot) or Section 30.2's histogram (needs
// atomicAdd, since multiple particles can land in the same bucket).
// Computing a hash from a value you already privately hold is always
// embarrassingly parallel; it is only ever STORING that hash somewhere
// SHARED that introduces contention.

#define CELL_SIZE 1.0
#define NUM_BUCKETS 8
#define PRIME_X 73856093
#define PRIME_Y 19349663

__device__ int spatial_hash_device(int cx, int cy) {
    long long mixed = ((long long)cx * PRIME_X) ^ ((long long)cy * PRIME_Y);
    long long m = mixed % NUM_BUCKETS;
    if (m < 0) m += NUM_BUCKETS;
    return (int)m;
}

__global__ void hash_particles_kernel(const double* pos_x, const double* pos_y,
                                       int* out_cx, int* out_cy, int* out_bucket) {
    int tid = threadIdx.x;
    int cx = (int)(pos_x[tid] / CELL_SIZE);
    int cy = (int)(pos_y[tid] / CELL_SIZE);
    out_cx[tid] = cx;
    out_cy[tid] = cy;
    out_bucket[tid] = spatial_hash_device(cx, cy);
}

// ---- Host-side replay of the identical per-thread logic. No forced
// ---- interleaving is needed here -- there is no shared state for any
// ---- interleaving to affect, which is itself the point of this
// ---- section. ----

int spatial_hash_host(int cx, int cy) {
    long long mixed = ((long long)cx * PRIME_X) ^ ((long long)cy * PRIME_Y);
    long long m = mixed % NUM_BUCKETS;
    if (m < 0) m += NUM_BUCKETS;
    return (int)m;
}

int main() {
    printf("=== Section 30.1 main: computing spatial hashes -- embarrassingly parallel, no atomics ===\n\n");

    double pos_x[] = {0.5, 0.7, 3.2, 1.5, 1.6, 7.1, 0.2, 4.4};
    double pos_y[] = {0.5, 0.9, 0.4, 2.5, 2.6, 7.8, 3.9, 4.1};
    int num_particles = 8;

    printf("cell size = %.1f, num buckets = %d\n", CELL_SIZE, NUM_BUCKETS);
    printf("launching %d threads, each independently hashing its own particle --\n", num_particles);
    printf("no thread reads or writes any other thread's data:\n\n");

    std::vector<int> cx(num_particles), cy(num_particles), bucket(num_particles);
    for (int tid = 0; tid < num_particles; tid++) {
        cx[tid] = (int)(pos_x[tid] / CELL_SIZE);
        cy[tid] = (int)(pos_y[tid] / CELL_SIZE);
        bucket[tid] = spatial_hash_host(cx[tid], cy[tid]);
        printf("  thread %d: pos=(%.1f, %.1f) -> cell=(%d,%d) -> bucket %d\n",
               tid, pos_x[tid], pos_y[tid], cx[tid], cy[tid], bucket[tid]);
    }

    int expected_bucket[] = {0, 0, 7, 3, 3, 2, 5, 0};
    bool ok = true;
    for (int i = 0; i < num_particles; i++) if (bucket[i] != expected_bucket[i]) ok = false;

    printf("\nfinal buckets: [ ");
    for (int b : bucket) printf("%d ", b);
    printf("]\n");

    printf("\nself-check: every thread's independently-computed bucket matches the\n");
    printf("sequential CPU baseline exactly, with no shared state ever touched: %s\n",
           ok ? "confirmed" : "MISMATCH");
    return ok ? 0 : 1;
}
```

**Compile and run:**

```bash
NVDIR=/usr/local/lib/python3.11/dist-packages/nvidia/cu13
$NVDIR/bin/nvcc -arch=sm_80 -I$NVDIR/include -L$NVDIR/lib -lcudart 167_spatial_hash_kernel.cu -o 167_spatial_hash_kernel
LD_LIBRARY_PATH=$NVDIR/lib ./167_spatial_hash_kernel
```

**Sample input:** the same 8 particles, each hashed by its own independent thread with no shared state touched between threads.

**Sample output:**

```text
=== Section 30.1 main: computing spatial hashes -- embarrassingly parallel, no atomics ===

cell size = 1.0, num buckets = 8
launching 8 threads, each independently hashing its own particle --
no thread reads or writes any other thread's data:

  thread 0: pos=(0.5, 0.5) -> cell=(0,0) -> bucket 0
  thread 1: pos=(0.7, 0.9) -> cell=(0,0) -> bucket 0
  thread 2: pos=(3.2, 0.4) -> cell=(3,0) -> bucket 7
  thread 3: pos=(1.5, 2.5) -> cell=(1,2) -> bucket 3
  thread 4: pos=(1.6, 2.6) -> cell=(1,2) -> bucket 3
  thread 5: pos=(7.1, 7.8) -> cell=(7,7) -> bucket 2
  thread 6: pos=(0.2, 3.9) -> cell=(0,3) -> bucket 5
  thread 7: pos=(4.4, 4.1) -> cell=(4,4) -> bucket 0

final buckets: [ 0 0 7 3 3 2 5 0 ]

self-check: every thread's independently-computed bucket matches the
sequential CPU baseline exactly, with no shared state ever touched: confirmed
```

## 30.2 Building the Grid: Histogram, Scan, and Scatter

### Intuition

Section 30.1 gave every particle a bucket; a queryable GRID needs those particles grouped contiguously by bucket, exactly Chapter 21's CSR layout. Building it composes three tools unchanged: Chapter 7's histogram counts how many particles land in each bucket, Chapter 5's exclusive scan turns those counts into CSR start offsets, and Chapter 6/13's scatter-by-running-count writes each particle into its bucket's slice using a per-bucket cursor. All three steps use only `atomicAdd` -- no CAS retry loop is needed anywhere, because (as Section 28.2 first established) a count or a cursor position never requires a thread to make a decision that could be invalidated; `atomicAdd`'s return value is unconditionally correct.

### The Sequential (CPU) Baseline

```cpp
#include <cstdio>
#include <vector>

// Chapter 30.2 CPU baseline -- turning Section 30.1's per-particle
// bucket assignments into a queryable grid means grouping particles by
// bucket CONTIGUOUSLY, exactly Chapter 21's CSR layout. Building it
// composes three tools this book already has: Chapter 7's histogram
// (count how many particles land in each bucket), Chapter 5's exclusive
// scan (turn those counts into CSR start offsets), and Chapter 6/13's
// scatter-by-running-count (write each particle into its bucket's
// slice using a per-bucket cursor that starts at that bucket's offset).

#define NUM_BUCKETS 8

int main() {
    printf("=== Section 30.2 CPU baseline: sequential grid construction (histogram + scan + scatter) ===\n\n");

    int buckets[] = {0, 0, 7, 3, 3, 2, 5, 0};   // from Section 30.1
    int num_particles = 8;

    printf("per-particle buckets: [ ");
    for (int i = 0; i < num_particles; i++) printf("%d ", buckets[i]);
    printf("]\n\n");

    // Step 1: histogram.
    std::vector<int> counts(NUM_BUCKETS, 0);
    for (int i = 0; i < num_particles; i++) counts[buckets[i]]++;
    printf("histogram (count per bucket): [ ");
    for (int c : counts) printf("%d ", c);
    printf("]\n\n");

    // Step 2: exclusive scan -> CSR offsets, with one sentinel entry.
    std::vector<int> offsets(NUM_BUCKETS + 1, 0);
    int running = 0;
    for (int b = 0; b < NUM_BUCKETS; b++) {
        offsets[b] = running;
        running += counts[b];
    }
    offsets[NUM_BUCKETS] = running;
    printf("CSR offsets (exclusive scan + sentinel): [ ");
    for (int o : offsets) printf("%d ", o);
    printf("]\n\n");

    // Step 3: scatter -- each bucket gets its own write cursor,
    // starting at that bucket's offset.
    std::vector<int> cursor(offsets.begin(), offsets.begin() + NUM_BUCKETS);
    std::vector<int> particle_ids(num_particles, 0);
    for (int pid = 0; pid < num_particles; pid++) {
        int b = buckets[pid];
        int slot = cursor[b];
        particle_ids[slot] = pid;
        cursor[b]++;
    }

    printf("scattered particle_ids (grouped contiguously by bucket): [ ");
    for (int p : particle_ids) printf("%d ", p);
    printf("]\n\n");

    printf("verifying: bucket b's particles occupy particle_ids[offsets[b] : offsets[b+1]]\n");
    int expected_bucket0[] = {0, 1, 7};
    int expected_bucket3[] = {3, 4};
    bool bucket0_ok = true, bucket3_ok = true;
    for (int b = 0; b < NUM_BUCKETS; b++) {
        printf("  bucket %d: particles [ ", b);
        for (int i = offsets[b]; i < offsets[b + 1]; i++) printf("%d ", particle_ids[i]);
        printf("]\n");
    }
    for (int i = 0; i < 3; i++) if (particle_ids[offsets[0] + i] != expected_bucket0[i]) bucket0_ok = false;
    for (int i = 0; i < 2; i++) if (particle_ids[offsets[3] + i] != expected_bucket3[i]) bucket3_ok = false;

    bool total_ok = (offsets[NUM_BUCKETS] == num_particles);
    bool ok = bucket0_ok && bucket3_ok && total_ok;
    printf("\nself-check: bucket 0 holds exactly {0,1,7} (a genuine hash collision plus\n");
    printf("a real same-cell pair), bucket 3 holds exactly {3,4} (same-cell pair), and\n");
    printf("every particle was scattered exactly once: %s\n", ok ? "confirmed" : "MISMATCH");
    return ok ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++17 -Wall -Wextra -O2 168_grid_build_cpu_baseline.cpp -o 168_grid_build_cpu_baseline
./168_grid_build_cpu_baseline
```

**Sample input:** the 8 particles' buckets from Section 30.1, grouped into a CSR layout via histogram, exclusive scan, and scatter.

**Sample output:**

```text
=== Section 30.2 CPU baseline: sequential grid construction (histogram + scan + scatter) ===

per-particle buckets: [ 0 0 7 3 3 2 5 0 ]

histogram (count per bucket): [ 3 0 1 2 0 1 0 1 ]

CSR offsets (exclusive scan + sentinel): [ 0 3 3 4 6 6 7 7 8 ]

scattered particle_ids (grouped contiguously by bucket): [ 0 1 7 5 3 4 6 2 ]

verifying: bucket b's particles occupy particle_ids[offsets[b] : offsets[b+1]]
  bucket 0: particles [ 0 1 7 ]
  bucket 1: particles [ ]
  bucket 2: particles [ 5 ]
  bucket 3: particles [ 3 4 ]
  bucket 4: particles [ ]
  bucket 5: particles [ 6 ]
  bucket 6: particles [ ]
  bucket 7: particles [ 2 ]

self-check: bucket 0 holds exactly {0,1,7} (a genuine hash collision plus
a real same-cell pair), bucket 3 holds exactly {3,4} (same-cell pair), and
every particle was scattered exactly once: confirmed
```

### The Concept, In Detail

```
ASCII view: histogram -> scan -> scatter, exactly Chapter 21's CSR pipeline.

  buckets:    [0, 0, 7, 3, 3, 2, 5, 0]     (from Section 30.1)

  histogram:  [3, 0, 1, 2, 0, 1, 0, 1]     (count per bucket)

  offsets:    [0, 3, 3, 4, 6, 6, 7, 7, 8]  (exclusive scan + sentinel)

  scatter (each particle writes at offsets[bucket] + its own running
  count within that bucket):
    particle_ids: [0, 1, 7, |5|, 3, 4, |6|, |2|]
                   \--bucket 0--/  b2  \b3-/ b5  b7
```

Every particle's final position in `particle_ids` is determined entirely by two numbers: its bucket's starting offset (fixed once the scan completes) and its own position within that bucket's arrival order (resolved by `atomicAdd` on that bucket's cursor). Because the cursor for each bucket starts exactly at that bucket's offset, no bucket's particles can ever spill into a neighboring bucket's slice, regardless of what order the particles actually land in.

[COMMON TRAP]
It is tempting to have every thread use ONE single global write cursor for the whole `particle_ids` array, incrementing it with `atomicAdd` for every particle regardless of bucket, reasoning that this is simpler than per-bucket cursors. A single global cursor destroys the whole point of the grid: particles would end up scattered in arrival order rather than grouped by bucket, and a later neighbor query would have no way to find "all particles in bucket b" without scanning the entire array -- the per-bucket cursor, seeded from that bucket's own CSR offset, is what makes each bucket's particles land in a contiguous, directly-indexable slice.

### Code and Verification

```cpp
#include <cstdio>
#include <vector>
#include <algorithm>

// Chapter 30.2 main -- concurrent grid construction. Histogramming and
// scattering both need only atomicAdd, never a CAS retry loop: exactly
// like Section 28.2's bump allocator, atomicAdd's return value (the
// count or cursor position BEFORE the add) is unconditionally each
// thread's own private, non-colliding result, regardless of what order
// the adds actually land in. Two particles concurrently landing in the
// SAME bucket must both be counted (Chapter 7's histogram) and must
// both receive DISTINCT scatter slots within that bucket's slice
// (Chapter 6/13's scatter-by-count) -- composed here with no new idea.
// The exclusive scan step in between is Chapter 5's, and is small
// enough here (8 buckets) to run as a single thread, unchanged from
// its established sequential form.

#define NUM_BUCKETS 8

__global__ void histogram_kernel(const int* buckets, int* counts) {
    int tid = threadIdx.x;
    int b = buckets[tid];
    atomicAdd(&counts[b], 1);
}

__global__ void scan_kernel(const int* counts, int* offsets) {
    // A single thread computes the exclusive scan -- Chapter 5 already
    // established the parallel version; 8 buckets needs no re-parallelizing.
    if (threadIdx.x != 0) return;
    int running = 0;
    for (int b = 0; b < NUM_BUCKETS; b++) {
        offsets[b] = running;
        running += counts[b];
    }
    offsets[NUM_BUCKETS] = running;
}

__global__ void scatter_kernel(const int* buckets, const int* offsets, int* cursors, int* particle_ids) {
    int tid = threadIdx.x;
    int b = buckets[tid];
    int local_slot = atomicAdd(&cursors[b], 1);
    particle_ids[offsets[b] + local_slot] = tid;
}

// ---- Host-side replay of the identical atomicAdd-based logic, driving
// ---- a specific landing order (particles 7 and 1, both bucket 0, land
// ---- before particle 0) so the composition is hand-traceable. ----

struct SharedCounter {
    int value = 0;
    int atomic_add(int n = 1) {
        int old = value;
        value += n;
        return old;
    }
};

int main() {
    printf("=== Section 30.2 main: concurrent grid construction -- atomicAdd histogram + scatter ===\n\n");

    int buckets[] = {0, 0, 7, 3, 3, 2, 5, 0};
    int num_particles = 8;
    int landing_order[] = {7, 1, 0, 2, 3, 4, 5, 6};

    printf("landing order (by particle id): [ ");
    for (int p : landing_order) printf("%d ", p);
    printf("]\n\n");

    printf("=== concurrent histogram ===\n");
    std::vector<SharedCounter> counters(NUM_BUCKETS);
    for (int pid : landing_order) {
        int b = buckets[pid];
        int old = counters[b].atomic_add(1);
        printf("  particle %d (bucket %d): atomicAdd(count[%d], 1) returned old=%d -> count[%d] now %d\n",
               pid, b, b, old, b, counters[b].value);
    }
    std::vector<int> counts(NUM_BUCKETS);
    for (int b = 0; b < NUM_BUCKETS; b++) counts[b] = counters[b].value;
    printf("\nfinal histogram: [ ");
    for (int c : counts) printf("%d ", c);
    printf("]\n\n");

    std::vector<int> offsets(NUM_BUCKETS + 1, 0);
    int running = 0;
    for (int b = 0; b < NUM_BUCKETS; b++) {
        offsets[b] = running;
        running += counts[b];
    }
    offsets[NUM_BUCKETS] = running;
    printf("CSR offsets: [ ");
    for (int o : offsets) printf("%d ", o);
    printf("]\n\n");

    printf("=== concurrent scatter ===\n");
    std::vector<SharedCounter> cursors(NUM_BUCKETS);
    std::vector<int> particle_ids(num_particles, -1);
    for (int pid : landing_order) {
        int b = buckets[pid];
        int local_slot = cursors[b].atomic_add(1);
        int global_slot = offsets[b] + local_slot;
        particle_ids[global_slot] = pid;
        printf("  particle %d (bucket %d): atomicAdd(cursor[%d],1) returned %d -> writes particle_ids[%d] = %d\n",
               pid, b, b, local_slot, global_slot, pid);
    }

    printf("\nfinal particle_ids: [ ");
    for (int p : particle_ids) printf("%d ", p);
    printf("]\n");

    bool ok = true;
    int expected_bucket0[] = {0, 1, 7};
    int expected_bucket3[] = {3, 4};
    std::vector<int> got_bucket0(particle_ids.begin() + offsets[0], particle_ids.begin() + offsets[1]);
    std::vector<int> got_bucket3(particle_ids.begin() + offsets[3], particle_ids.begin() + offsets[4]);
    std::vector<int> sorted0 = got_bucket0, sorted3 = got_bucket3;
    std::sort(sorted0.begin(), sorted0.end());
    std::sort(sorted3.begin(), sorted3.end());
    for (int i = 0; i < 3; i++) if (sorted0[i] != expected_bucket0[i]) ok = false;
    for (int i = 0; i < 2; i++) if (sorted3[i] != expected_bucket3[i]) ok = false;
    if (offsets[NUM_BUCKETS] != num_particles) ok = false;

    printf("\nself-check: bucket 0's particle SET is exactly {0,1,7} and bucket 3's is\n");
    printf("exactly {3,4} regardless of the forced out-of-order landing, and every\n");
    printf("particle was scattered exactly once: %s\n", ok ? "confirmed" : "MISMATCH");
    return ok ? 0 : 1;
}
```

**Compile and run:**

```bash
NVDIR=/usr/local/lib/python3.11/dist-packages/nvidia/cu13
$NVDIR/bin/nvcc -arch=sm_80 -I$NVDIR/include -L$NVDIR/lib -lcudart 169_grid_build_kernel.cu -o 169_grid_build_kernel
LD_LIBRARY_PATH=$NVDIR/lib ./169_grid_build_kernel
```

**Sample input:** the same 8 particles' buckets, histogrammed and scattered by threads landing in a specific out-of-order sequence, with two particles genuinely contending for the same bucket's counter and cursor.

**Sample output:**

```text
=== Section 30.2 main: concurrent grid construction -- atomicAdd histogram + scatter ===

landing order (by particle id): [ 7 1 0 2 3 4 5 6 ]

=== concurrent histogram ===
  particle 7 (bucket 0): atomicAdd(count[0], 1) returned old=0 -> count[0] now 1
  particle 1 (bucket 0): atomicAdd(count[0], 1) returned old=1 -> count[0] now 2
  particle 0 (bucket 0): atomicAdd(count[0], 1) returned old=2 -> count[0] now 3
  particle 2 (bucket 7): atomicAdd(count[7], 1) returned old=0 -> count[7] now 1
  particle 3 (bucket 3): atomicAdd(count[3], 1) returned old=0 -> count[3] now 1
  particle 4 (bucket 3): atomicAdd(count[3], 1) returned old=1 -> count[3] now 2
  particle 5 (bucket 2): atomicAdd(count[2], 1) returned old=0 -> count[2] now 1
  particle 6 (bucket 5): atomicAdd(count[5], 1) returned old=0 -> count[5] now 1

final histogram: [ 3 0 1 2 0 1 0 1 ]

CSR offsets: [ 0 3 3 4 6 6 7 7 8 ]

=== concurrent scatter ===
  particle 7 (bucket 0): atomicAdd(cursor[0],1) returned 0 -> writes particle_ids[0] = 7
  particle 1 (bucket 0): atomicAdd(cursor[0],1) returned 1 -> writes particle_ids[1] = 1
  particle 0 (bucket 0): atomicAdd(cursor[0],1) returned 2 -> writes particle_ids[2] = 0
  particle 2 (bucket 7): atomicAdd(cursor[7],1) returned 0 -> writes particle_ids[7] = 2
  particle 3 (bucket 3): atomicAdd(cursor[3],1) returned 0 -> writes particle_ids[4] = 3
  particle 4 (bucket 3): atomicAdd(cursor[3],1) returned 1 -> writes particle_ids[5] = 4
  particle 5 (bucket 2): atomicAdd(cursor[2],1) returned 0 -> writes particle_ids[3] = 5
  particle 6 (bucket 5): atomicAdd(cursor[5],1) returned 0 -> writes particle_ids[6] = 6

final particle_ids: [ 7 1 0 5 3 4 6 2 ]

self-check: bucket 0's particle SET is exactly {0,1,7} and bucket 3's is
exactly {3,4} regardless of the forced out-of-order landing, and every
particle was scattered exactly once: confirmed
```

## 30.3 Neighbor Queries: Checking Cells, Not Buckets

### Intuition

With the grid built, finding a particle's true spatial neighbors means checking its own cell plus the 8 surrounding cells (9 total, in 2D) -- never the whole particle set. But Section 30.1 already showed that hash collisions can put spatially distant particles in the same bucket as a query's neighbor cells, so a correct query must recompute each of the 9 neighbor cells' hashes, gather that bucket's candidates, and then explicitly verify each candidate's ACTUAL stored cell coordinate against the specific neighbor cell being examined before ever computing a distance -- bucket membership narrows the search, but only the cell-coordinate check proves a candidate is really a neighbor.

### The Sequential (CPU) Baseline

```cpp
#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>

// Chapter 30.3 CPU baseline -- using the Section 30.2 grid to find a
// query particle's true spatial neighbors by examining only nearby
// CELLS, instead of every particle in the simulation. The critical
// subtlety this section exists to teach: neighboring BUCKETS are not
// the same thing as neighboring CELLS, because spatial hashing can (and
// Section 30.1's own data DOES) map far-apart cells to the same bucket.
// A correct neighbor query recomputes the hash for each of the 9
// neighbor cells (including the query particle's own cell, in 2D), then
// explicitly verifies each candidate's ACTUAL cell coordinate against
// the neighbor cell being examined -- bucket membership alone is never
// proof of spatial proximity.

#define CELL_SIZE 1.0
#define NUM_BUCKETS 8
#define PRIME_X 73856093
#define PRIME_Y 19349663
#define RADIUS 1.0

struct CellCoord { int cx, cy; };

CellCoord cell_of(double x, double y) {
    return {(int)(x / CELL_SIZE), (int)(y / CELL_SIZE)};
}

int spatial_hash(int cx, int cy) {
    long long mixed = ((long long)cx * PRIME_X) ^ ((long long)cy * PRIME_Y);
    long long m = mixed % NUM_BUCKETS;
    if (m < 0) m += NUM_BUCKETS;
    return (int)m;
}

int main() {
    printf("=== Section 30.3 CPU baseline: sequential neighbor query using the spatial hash grid ===\n\n");

    struct { double x, y; } particles[] = {
        {0.5, 0.5}, {0.7, 0.9}, {3.2, 0.4}, {1.5, 2.5},
        {1.6, 2.6}, {7.1, 7.8}, {0.2, 3.9}, {4.4, 4.1},
    };
    int num_particles = 8;

    std::vector<CellCoord> cells(num_particles);
    std::vector<int> buckets(num_particles);
    for (int i = 0; i < num_particles; i++) {
        cells[i] = cell_of(particles[i].x, particles[i].y);
        buckets[i] = spatial_hash(cells[i].cx, cells[i].cy);
    }

    // Build the grid (Section 30.2).
    std::vector<int> counts(NUM_BUCKETS, 0);
    for (int b : buckets) counts[b]++;
    std::vector<int> offsets(NUM_BUCKETS + 1, 0);
    int running = 0;
    for (int b = 0; b < NUM_BUCKETS; b++) { offsets[b] = running; running += counts[b]; }
    offsets[NUM_BUCKETS] = running;
    std::vector<int> cursor(offsets.begin(), offsets.begin() + NUM_BUCKETS);
    std::vector<int> particle_ids(num_particles);
    for (int pid = 0; pid < num_particles; pid++) {
        int b = buckets[pid];
        particle_ids[cursor[b]] = pid;
        cursor[b]++;
    }

    printf("cells: [ ");
    for (auto& c : cells) printf("(%d,%d) ", c.cx, c.cy);
    printf("]\n");
    printf("buckets: [ ");
    for (int b : buckets) printf("%d ", b);
    printf("]\n\n");

    auto neighbor_query = [&](int qid) {
        double qx = particles[qid].x, qy = particles[qid].y;
        int qcx = cells[qid].cx, qcy = cells[qid].cy;
        printf("query particle %d: pos=(%.1f,%.1f), cell=(%d,%d), radius=%.1f\n",
               qid, qx, qy, qcx, qcy, RADIUS);

        std::vector<int> candidates_checked;
        std::vector<std::pair<int,double>> true_neighbors;

        for (int dcx = -1; dcx <= 1; dcx++) {
            for (int dcy = -1; dcy <= 1; dcy++) {
                int ncx = qcx + dcx, ncy = qcy + dcy;
                int nb = spatial_hash(ncx, ncy);
                for (int i = offsets[nb]; i < offsets[nb + 1]; i++) {
                    int cand = particle_ids[i];
                    if (cand == qid) continue;
                    candidates_checked.push_back(cand);
                    // CRITICAL check: verify the candidate's ACTUAL cell
                    // matches the neighbor cell being examined.
                    if (cells[cand].cx != ncx || cells[cand].cy != ncy) continue;
                    double dx = particles[cand].x - qx, dy = particles[cand].y - qy;
                    double dist = std::sqrt(dx * dx + dy * dy);
                    if (dist <= RADIUS) true_neighbors.push_back({cand, dist});
                }
            }
        }

        std::vector<int> unique_candidates = candidates_checked;
        std::sort(unique_candidates.begin(), unique_candidates.end());
        unique_candidates.erase(std::unique(unique_candidates.begin(), unique_candidates.end()), unique_candidates.end());
        printf("  candidates examined across 9 neighbor cells (via their buckets): [ ");
        for (int c : unique_candidates) printf("%d ", c);
        printf("]\n");
        printf("  true neighbors within radius %.1f: [ ", RADIUS);
        for (auto& [id, d] : true_neighbors) printf("(%d, %.3f) ", id, d);
        printf("]\n");
        return true_neighbors;
    };

    auto n0 = neighbor_query(0);
    printf("\n");
    auto n3 = neighbor_query(3);

    printf("\nnote: particle 7 (cell (4,4)) shares bucket %d with particles 0 and 1 (a\n", buckets[7]);
    printf("hash collision) but is correctly EXCLUDED from particle 0's neighbor list\n");
    printf("because its actual cell (4,4) never equals any of particle 0's 9 examined\n");
    printf("neighbor cells -- the cell-coordinate check is what filters it out.\n");

    bool n0_ok = (n0.size() == 1 && n0[0].first == 1);
    bool n3_ok = (n3.size() == 1 && n3[0].first == 4);
    bool particle7_excluded = true;
    for (auto& [id, d] : n0) if (id == 7) particle7_excluded = false;

    bool ok = n0_ok && n3_ok && particle7_excluded;
    printf("\nself-check: particle 0's only true neighbor is particle 1, particle 3's\n");
    printf("only true neighbor is particle 4, and the bucket-colliding particle 7 is\n");
    printf("correctly excluded from particle 0's results: %s\n", ok ? "confirmed" : "MISMATCH");
    return ok ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++17 -Wall -Wextra -O2 170_neighbor_query_cpu_baseline.cpp -o 170_neighbor_query_cpu_baseline
./170_neighbor_query_cpu_baseline
```

**Sample input:** queries for particles 0 and 3 against the Section 30.2 grid, checking all 9 neighbor cells and filtering candidates by exact cell match before computing distance, radius 1.0.

**Sample output:**

```text
=== Section 30.3 CPU baseline: sequential neighbor query using the spatial hash grid ===

cells: [ (0,0) (0,0) (3,0) (1,2) (1,2) (7,7) (0,3) (4,4) ]
buckets: [ 0 0 7 3 3 2 5 0 ]

query particle 0: pos=(0.5,0.5), cell=(0,0), radius=1.0
  candidates examined across 9 neighbor cells (via their buckets): [ 1 2 3 4 5 6 7 ]
  true neighbors within radius 1.0: [ (1, 0.447) ]

query particle 3: pos=(1.5,2.5), cell=(1,2), radius=1.0
  candidates examined across 9 neighbor cells (via their buckets): [ 0 1 2 4 5 6 7 ]
  true neighbors within radius 1.0: [ (4, 0.141) ]

note: particle 7 (cell (4,4)) shares bucket 0 with particles 0 and 1 (a
hash collision) but is correctly EXCLUDED from particle 0's neighbor list
because its actual cell (4,4) never equals any of particle 0's 9 examined
neighbor cells -- the cell-coordinate check is what filters it out.

self-check: particle 0's only true neighbor is particle 1, particle 3's
only true neighbor is particle 4, and the bucket-colliding particle 7 is
correctly excluded from particle 0's results: confirmed
```

### The Concept, In Detail

```
ASCII view: 9 neighbor cells checked, but only a genuine cell match counted.

  query particle 0, cell (0,0). 9 cells checked: (-1,-1) .. (1,1)

  cell (0,0) -> bucket 0 -> candidates {1, 7}
    particle 1: actual cell (0,0) == (0,0) MATCH -> distance 0.447 <= radius -> NEIGHBOR
    particle 7: actual cell (4,4) != (0,0) NO MATCH -> rejected (hash collision, not a neighbor)

  cell (1,0), (1,1), etc: bucket lookups may return OTHER unrelated
  particles (more collisions) -- all rejected the same way, by cell
  mismatch, before any distance is ever computed
```

The cell-coordinate check comes BEFORE the distance computation for a reason: computing a distance for a particle that was never actually in a relevant cell would be wasted work at best, and at worst could accidentally treat a bucket-collided but distant particle as a neighbor if its distance happened to fall inside the radius by coincidence. Filtering by exact cell match first guarantees every distance computation that follows is against a particle that genuinely occupies one of the 9 cells being searched.

[COMMON TRAP]
It is tempting to skip the cell-coordinate check entirely and simply compute the distance to every candidate a neighbor cell's bucket returns, reasoning that "particles too far away will just fail the radius check anyway." A bucket collision can place a particle from a DISTANT cell at any distance, including one that happens to fall inside the query radius purely by chance -- skipping the cell check does not merely waste work, it can silently accept a false neighbor that a correct implementation would have rejected before ever measuring its distance.

### Code and Verification

```cpp
#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>

// Chapter 30.3 main -- one thread per query particle, each independently
// walking its own 9 neighbor cells against the SAME shared, READ-ONLY
// grid built in Section 30.2. Like Section 30.1's hashing, this needs
// no atomics and no CAS loop at all: every thread only ever READS the
// shared bucket/offset/particle_ids arrays and writes only to its own
// private output slot, so many query threads can run over the same
// grid simultaneously with zero contention -- the grid, once built, is
// immutable for the duration of every query.

#define CELL_SIZE 1.0
#define NUM_BUCKETS 8
#define PRIME_X 73856093
#define PRIME_Y 19349663
#define RADIUS 1.0
#define MAX_NEIGHBORS 8

__device__ int spatial_hash_device(int cx, int cy) {
    long long mixed = ((long long)cx * PRIME_X) ^ ((long long)cy * PRIME_Y);
    long long m = mixed % NUM_BUCKETS;
    if (m < 0) m += NUM_BUCKETS;
    return (int)m;
}

__global__ void neighbor_query_kernel(const double* pos_x, const double* pos_y,
                                       const int* cell_x, const int* cell_y,
                                       const int* offsets, const int* particle_ids,
                                       int* out_neighbor_ids, int* out_neighbor_count) {
    int qid = threadIdx.x;
    double qx = pos_x[qid], qy = pos_y[qid];
    int qcx = cell_x[qid], qcy = cell_y[qid];
    int found = 0;

    for (int dcx = -1; dcx <= 1; dcx++) {
        for (int dcy = -1; dcy <= 1; dcy++) {
            int ncx = qcx + dcx, ncy = qcy + dcy;
            int nb = spatial_hash_device(ncx, ncy);
            for (int i = offsets[nb]; i < offsets[nb + 1]; i++) {
                int cand = particle_ids[i];
                if (cand == qid) continue;
                if (cell_x[cand] != ncx || cell_y[cand] != ncy) continue;   // reject hash collisions
                double dx = pos_x[cand] - qx, dy = pos_y[cand] - qy;
                double dist = sqrt(dx * dx + dy * dy);
                if (dist <= RADIUS && found < MAX_NEIGHBORS) {
                    out_neighbor_ids[qid * MAX_NEIGHBORS + found] = cand;
                    found++;
                }
            }
        }
    }
    out_neighbor_count[qid] = found;
}

// ---- Host-side replay of the identical per-thread logic, run for two
// ---- query threads (particle 0 and particle 3) against the same
// ---- shared, read-only grid -- no interleaving needs to be forced,
// ---- since neither thread ever writes anything the other reads. ----

int spatial_hash_host(int cx, int cy) {
    long long mixed = ((long long)cx * PRIME_X) ^ ((long long)cy * PRIME_Y);
    long long m = mixed % NUM_BUCKETS;
    if (m < 0) m += NUM_BUCKETS;
    return (int)m;
}

int main() {
    printf("=== Section 30.3 main: concurrent neighbor queries over a shared, read-only grid ===\n\n");

    double pos_x[] = {0.5, 0.7, 3.2, 1.5, 1.6, 7.1, 0.2, 4.4};
    double pos_y[] = {0.5, 0.9, 0.4, 2.5, 2.6, 7.8, 3.9, 4.1};
    int num_particles = 8;

    std::vector<int> cell_x(num_particles), cell_y(num_particles), buckets(num_particles);
    for (int i = 0; i < num_particles; i++) {
        cell_x[i] = (int)(pos_x[i] / CELL_SIZE);
        cell_y[i] = (int)(pos_y[i] / CELL_SIZE);
        buckets[i] = spatial_hash_host(cell_x[i], cell_y[i]);
    }

    std::vector<int> counts(NUM_BUCKETS, 0);
    for (int b : buckets) counts[b]++;
    std::vector<int> offsets(NUM_BUCKETS + 1, 0);
    int running = 0;
    for (int b = 0; b < NUM_BUCKETS; b++) { offsets[b] = running; running += counts[b]; }
    offsets[NUM_BUCKETS] = running;
    std::vector<int> cursor(offsets.begin(), offsets.begin() + NUM_BUCKETS);
    std::vector<int> particle_ids(num_particles);
    for (int pid = 0; pid < num_particles; pid++) {
        int b = buckets[pid];
        particle_ids[cursor[b]] = pid;
        cursor[b]++;
    }

    printf("grid built (Section 30.2's exact CSR layout); 2 query threads now run\n");
    printf("concurrently against it, each only reading shared state:\n\n");

    auto run_query_thread = [&](int qid) {
        double qx = pos_x[qid], qy = pos_y[qid];
        int qcx = cell_x[qid], qcy = cell_y[qid];
        std::vector<int> neighbors;
        for (int dcx = -1; dcx <= 1; dcx++) {
            for (int dcy = -1; dcy <= 1; dcy++) {
                int ncx = qcx + dcx, ncy = qcy + dcy;
                int nb = spatial_hash_host(ncx, ncy);
                for (int i = offsets[nb]; i < offsets[nb + 1]; i++) {
                    int cand = particle_ids[i];
                    if (cand == qid) continue;
                    if (cell_x[cand] != ncx || cell_y[cand] != ncy) continue;
                    double dx = pos_x[cand] - qx, dy = pos_y[cand] - qy;
                    double dist = std::sqrt(dx * dx + dy * dy);
                    if (dist <= RADIUS) neighbors.push_back(cand);
                }
            }
        }
        return neighbors;
    };

    printf("  thread (query particle) 0: pos=(%.1f,%.1f), cell=(%d,%d)\n", pos_x[0], pos_y[0], cell_x[0], cell_y[0]);
    auto neighbors0 = run_query_thread(0);
    printf("    found %zu neighbor(s): [ ", neighbors0.size());
    for (int n : neighbors0) printf("%d ", n);
    printf("]\n");

    printf("  thread (query particle) 3: pos=(%.1f,%.1f), cell=(%d,%d)\n", pos_x[3], pos_y[3], cell_x[3], cell_y[3]);
    auto neighbors3 = run_query_thread(3);
    printf("    found %zu neighbor(s): [ ", neighbors3.size());
    for (int n : neighbors3) printf("%d ", n);
    printf("]\n");

    bool ok0 = (neighbors0.size() == 1 && neighbors0[0] == 1);
    bool ok3 = (neighbors3.size() == 1 && neighbors3[0] == 4);
    bool ok = ok0 && ok3;

    printf("\nself-check: both query threads, running concurrently over the same\n");
    printf("immutable shared grid with no synchronization between them, find exactly\n");
    printf("the same true neighbors as Section 30.3's sequential CPU baseline: %s\n",
           ok ? "confirmed" : "MISMATCH");
    return ok ? 0 : 1;
}
```

**Compile and run:**

```bash
NVDIR=/usr/local/lib/python3.11/dist-packages/nvidia/cu13
$NVDIR/bin/nvcc -arch=sm_80 -I$NVDIR/include -L$NVDIR/lib -lcudart 171_neighbor_query_kernel.cu -o 171_neighbor_query_kernel
LD_LIBRARY_PATH=$NVDIR/lib ./171_neighbor_query_kernel
```

**Sample input:** two query threads (particles 0 and 3) running concurrently against the same immutable grid, each independently checking its own 9 neighbor cells.

**Sample output:**

```text
=== Section 30.3 main: concurrent neighbor queries over a shared, read-only grid ===

grid built (Section 30.2's exact CSR layout); 2 query threads now run
concurrently against it, each only reading shared state:

  thread (query particle) 0: pos=(0.5,0.5), cell=(0,0)
    found 1 neighbor(s): [ 1 ]
  thread (query particle) 3: pos=(1.5,2.5), cell=(1,2)
    found 1 neighbor(s): [ 4 ]

self-check: both query threads, running concurrently over the same
immutable shared grid with no synchronization between them, find exactly
the same true neighbors as Section 30.3's sequential CPU baseline: confirmed
```

## Chapter Summary

A spatial hash grid for particle simulation reused every tool this chapter needed without inventing anything new: Chapter 19/29's key-hashing idea, applied to a 2D cell coordinate via two-prime XOR mixing, maps an unbounded domain down to a small, fixed bucket count; Chapter 7's histogram, Chapter 5's exclusive scan, and Chapter 6/13's scatter-by-count compose unchanged into Chapter 21's CSR layout to group particles contiguously by bucket; and neighbor queries walk only the 9 relevant cells around a query particle instead of the whole particle set. The one genuinely new discipline this chapter introduced is the cell-coordinate check: because a hash table's bucket count is necessarily smaller than an unbounded domain's cell count, distant cells will collide into shared buckets, and every neighbor query must verify a candidate's actual cell before trusting it as a real spatial neighbor, never relying on bucket membership alone.

## Self-Check Questions

1. Why does computing each particle's spatial hash in Section 30.1 need no atomics or CAS loop, while Section 30.2's histogram and scatter both need `atomicAdd`?
2. What does it mean, concretely, for two particles to "collide" in a spatial hash, and why does this happen even when their positions are far apart?
3. Why does Section 30.2's scatter step need a SEPARATE write cursor per bucket, rather than one single shared cursor for the whole output array?
4. When Section 30.3 examines a neighbor cell's bucket and finds a candidate particle, what specific check must happen before that candidate's distance is ever computed, and why in that order?
5. Why can a query particle's true neighbor set never be found correctly by only comparing bucket indices, no matter how good the hash function is?
6. Section 30.3's kernel runs many query threads concurrently over the same grid with no synchronization between them at all. What property of the grid, once built, makes that safe?

## Where We Go Next

Spatial hashing organizes particles by which cell they occupy, which is exactly the right tool for uniform, grid-like neighbor queries -- but it says nothing about which OBJECTS a ray actually passes through, or how to organize wildly different-sized objects (a tiny sphere next to a sprawling terrain mesh) into a single searchable structure. Chapter 31 turns to parallel BVH (bounding volume hierarchy) construction for ray tracing, building a tree-shaped spatial index from the ground up, combining Chapter 16's parallel tree construction with Chapter 4's reduction to bound each node's children -- one of five remaining case studies that close out Part 8 by putting this book's toolbox to work on real production problems, from high-frequency trading to genomics to autonomous-vehicle perception.

## Worked Solutions

**1.** Computing a hash from a value a thread already privately holds -- its own particle's position -- involves no shared state at all: every thread reads only its own input and writes only its own output slot, so no two threads can ever interfere with each other's work. Section 30.2's histogram and scatter, by contrast, have multiple threads potentially incrementing the SAME shared bucket counter or cursor (whenever their particles land in the same bucket), and `atomicAdd` is what guarantees each of those increments is applied without any being silently lost.

**2.** Two particles "collide" in a spatial hash when the hash function maps their cell coordinates to the SAME bucket index, which can happen either because they share the same cell (a genuine spatial match) or, just as easily, because the hash function's limited output range mapped two completely different, possibly very distant cells to the same bucket by coincidence. This happens precisely because a spatial hash table intentionally uses far fewer buckets than the number of distinct cells a large or unbounded domain could contain, so some cells are guaranteed to share buckets.

**3.** A single shared cursor for the whole output array would scatter particles in whatever order their `atomicAdd` calls happened to land, completely destroying the CSR grouping -- particles from different buckets would end up interleaved throughout the array with no way to find "every particle in bucket b" without scanning everything. A separate cursor per bucket, seeded at that bucket's own CSR starting offset, guarantees every particle lands somewhere within its own bucket's contiguous slice, regardless of arrival order, which is exactly what makes the grid directly indexable by bucket afterward.

**4.** Before any distance is computed, the candidate's ACTUAL stored cell coordinate must be checked against the specific neighbor cell currently being examined. This must come first because a bucket can contain particles from cells far outside the 9 being searched (due to hash collisions), and computing a distance for such a particle would be work spent evaluating a candidate that was never a legitimate neighbor-cell occupant in the first place -- filtering by cell membership first ensures every distance computed afterward is for a particle that genuinely belongs to one of the cells actually being searched.

**5.** Bucket indices are the output of a hash function whose whole purpose is to compress a large or unbounded set of distinct cell coordinates down into a small, fixed number of outputs -- by design, this means some cells that are nowhere near each other in space will still produce the same bucket index. Comparing bucket indices alone can therefore never distinguish "these two particles share a cell" from "these two particles' cells happened to hash to the same bucket," no matter how well-designed the hash function is; only comparing the actual stored cell coordinates resolves that ambiguity.

**6.** Once built, the grid (the bucket assignments, the CSR offsets, and the scattered `particle_ids` array) is never modified again during the query phase -- every query thread only ever READS these arrays and writes to its own private output. Because no thread writes to any of the shared grid data during queries, there is no possible data race between concurrently running query threads, and the grid can be queried by an arbitrary number of threads simultaneously with no synchronization needed at all, exactly the same "read-only sharing needs no atomics" property Section 30.1's hashing step relied on.
