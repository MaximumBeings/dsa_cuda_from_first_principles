# Appendix G: Common Failure Modes -- Divergence, Bank Conflicts, and Atomic Contention in Data Structures

Every part of this book has measured one performance cost or another as it went -- Chapter 1's warp-lockstep model, Chapter 4's issue-pass counting, Chapter 7's contention-vs-correctness distinction. This appendix collects the three failure modes that recur most often when a hand-written data structure runs slower than its operation count alone would predict, gives each one a genuinely computed, minimal example, and closes with a checklist for recognizing them in code this book did not write.

## G.1 Warp Divergence: When a Loop's Trip Count Depends on the Data

### Intuition

A GPU warp does not run 32 independent threads -- it runs ONE instruction stream, 32 lanes wide, in lockstep. Chapter 1 established what happens when an `if`/`else` splits a warp's lanes onto different paths: the warp does not somehow run both paths in parallel, it runs each path in turn, with the lanes that do not belong to that path simply masked off (idle) until their own path comes around. Chapter 4 measured this exact cost for a reduction's active/inactive thread pattern. This section asks the same question about a shape that shows up constantly in real data structures: not an `if`/`else`, but a LOOP whose trip count differs from lane to lane -- exactly what happens whenever different keys need different numbers of probes, different tree paths have different depths, or different linked-list traversals run different lengths.

### The Concept, In Detail

```
one warp (32 lanes), each looking up a different key in the same
open-addressing table (Chapter 19's own structure):

  lane:    0  1  2  3  4  5  6  7  8  9 10 11 ... 31
  probes:  0  1  2  3  4  5  6  7  8  9  0  0  ...  0
           <---- one long collision chain ---->  <-- everyone else, 0 probes -->

  round 0: ALL 32 lanes active (every lane's loop condition is still being checked)
  round 1: lanes 1-9 active, lanes 0,10-31 masked off (already done)
  round 2: lanes 2-9 active, everyone else masked off
  ...
  round 9: lane 9 active ALONE, 31 lanes masked off
  round 10: warp finally retires the loop -- every lane needed this many rounds
            to exist, even the 22 lanes that had nothing left to do after round 0
```

The warp cannot move past the probing loop until EVERY lane's own `while` condition has gone false -- lane 9's nine probes force all 32 lanes to sit through ten rounds of the loop, even though 22 of those lanes finished their real work in the very first round. This is not a bug in the probing logic; open addressing's correctness (Chapter 19) is completely unaffected by how the lanes happen to be scheduled. It is a performance cost that exists ONLY because the lanes share one instruction stream -- the identical 32 lookups, run as 32 fully independent CPU threads (or even 32 independent GPU threads in different warps), would cost nothing extra for the 22 short lookups at all.

!!! warning "[COMMON TRAP] Assuming divergence only comes from `if`/`else`"
    It is tempting to associate "divergence" only with an explicit branch, and conclude that a plain `while` loop with no visible `if` inside it cannot diverge. A loop's own exit condition IS a branch, evaluated once per iteration -- when different lanes need that condition to stay true for different numbers of iterations, the warp pays exactly the same masked-lane cost Chapter 4 measured for an explicit `if`, whether or not the word "if" ever appears in the source. Any data-dependent trip count -- a probe chain, a tree depth, a linked-list length -- is a divergence source, not just a visible conditional.

### Code and Verification

```cpp
// 233_warp_divergence_probe_depth.cu
//
// Appendix G.1 -- Chapter 1 established the rule this file applies:
// a warp executes in lockstep, one instruction at a time, for all 32
// lanes at once; a lane with nothing left to do is simply MASKED OFF
// (idle) rather than skipped, so the warp as a whole cannot move past
// an instruction until every one of its lanes has finished needing it.
// Chapter 4 measured this cost for an if/else-shaped branch. This file
// measures the SAME cost for a shape that shows up constantly in real
// data structures: a per-lane loop whose TRIP COUNT varies by lane --
// here, Chapter 19's own linear-probing lookup, where one key might be
// found on the first check and another might need many probes. Thirty-
// two threads (one warp) each look up a different key in the identical
// kind of open-addressing table Chapter 19 built; the table is
// deliberately built so ten of those keys collided into one long probe
// chain during insertion, and the other twenty-two did not collide at
// all.
//
// Compile: nvcc -arch=sm_80 -I$NVDIR/include -L$NVDIR/lib -lcudart 233_warp_divergence_probe_depth.cu -o 233_warp_divergence_probe_depth
// Run:     LD_LIBRARY_PATH=$NVDIR/lib ./233_warp_divergence_probe_depth

#include <algorithm>
#include <cstdio>
#include <vector>
#include <cuda_runtime.h>

#define TABLE_SIZE 41
#define EMPTY (-1)
#define WARP_SIZE 32

// Real kernel: each of the 32 lanes in one warp performs its OWN
// Chapter-19-style linear-probing lookup, entirely independently, and
// records how many probes its own key needed. No lane depends on any
// other lane's result -- the only thing they share is the warp's
// instruction stream, which is exactly what makes this a divergence
// example rather than a correctness one.
__global__ void warp_lookup_kernel(const int* keys, const int* table, int* probe_depths) {
    int lane = threadIdx.x;
    int key = keys[lane];
    int slot = key % TABLE_SIZE;
    int probes = 0;
    while (table[slot] != key) {
        slot = (slot + 1) % TABLE_SIZE;
        probes++;
    }
    probe_depths[lane] = probes;
}

void report(const char* call_name, cudaError_t err) {
    printf("  %-24s -> %-24s (%s)\n", call_name, cudaGetErrorName(err), cudaGetErrorString(err));
}

// Host-side build: insert all 32 keys via Chapter 19's own linear
// probing, in the same order the warp below will look them up in.
std::vector<int> build_table(const std::vector<int>& keys) {
    std::vector<int> table(TABLE_SIZE, EMPTY);
    for (int key : keys) {
        int slot = key % TABLE_SIZE;
        while (table[slot] != EMPTY) slot = (slot + 1) % TABLE_SIZE;
        table[slot] = key;
    }
    return table;
}

// Host-side replay of the identical per-lane lookup logic the kernel
// above performs.
std::vector<int> replay_lookup(const std::vector<int>& keys, const std::vector<int>& table) {
    std::vector<int> probe_depths(keys.size());
    for (size_t lane = 0; lane < keys.size(); lane++) {
        int key = keys[lane];
        int slot = key % TABLE_SIZE;
        int probes = 0;
        while (table[slot] != key) {
            slot = (slot + 1) % TABLE_SIZE;
            probes++;
        }
        probe_depths[lane] = probes;
    }
    return probe_depths;
}

int main() {
    printf("=== Appendix G.1: warp divergence from variable-depth probing ===\n\n");

    // Ten keys deliberately chosen to all collide at home slot 5 (each
    // is congruent to 5 mod 41), forming one long probe chain of depths
    // 0 through 9. Twenty-two keys chosen so their home slots (0-4 and
    // 15-31) never touch that chain at all -- every one of them is
    // found in exactly 0 probes.
    std::vector<int> cluster_keys = {5, 46, 87, 128, 169, 210, 251, 292, 333, 374};
    std::vector<int> spread_keys = {0, 1, 2, 3, 4, 15, 16, 17, 18, 19, 20,
                                     21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31};
    std::vector<int> keys;
    keys.insert(keys.end(), cluster_keys.begin(), cluster_keys.end());
    keys.insert(keys.end(), spread_keys.begin(), spread_keys.end());
    printf("warp of %d lanes, one key per lane -- lanes 0-9 all collide at home slot 5,\n", WARP_SIZE);
    printf("lanes 10-31 each land on an untouched home slot\n\n");

    std::vector<int> table = build_table(keys);

    printf("attempting a genuine device allocation and kernel launch:\n");
    int* d_keys = nullptr;
    int* d_table = nullptr;
    int* d_probe_depths = nullptr;
    cudaError_t e1 = cudaMalloc(&d_keys, WARP_SIZE * sizeof(int));
    report("cudaMalloc(keys)", e1);
    cudaError_t e2 = cudaMalloc(&d_table, TABLE_SIZE * sizeof(int));
    report("cudaMalloc(table)", e2);
    cudaError_t e3 = cudaMalloc(&d_probe_depths, WARP_SIZE * sizeof(int));
    report("cudaMalloc(probe_depths)", e3);

    if (e1 == cudaSuccess && e2 == cudaSuccess && e3 == cudaSuccess) {
        cudaMemcpy(d_keys, keys.data(), WARP_SIZE * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_table, table.data(), TABLE_SIZE * sizeof(int), cudaMemcpyHostToDevice);
        warp_lookup_kernel<<<1, WARP_SIZE>>>(d_keys, d_table, d_probe_depths);
        cudaDeviceSynchronize();
        printf("  (kernel launch attempted -- would print device results here on real hardware)\n\n");
    } else {
        printf("\n  (this environment has no usable device -- see Section 2.3. What follows is a\n");
        printf("   host-side REPLAY of warp_lookup_kernel's exact per-lane probing logic.)\n\n");
    }

    std::vector<int> probe_depths = replay_lookup(keys, table);

    printf("per-lane probe depth:\n  [ ");
    for (int d : probe_depths) printf("%d ", d);
    printf("]\n\n");

    // Chapter 1's warp-lockstep rule, applied to a variable-trip-count
    // loop rather than an if/else: the warp cannot retire the loop
    // instruction until every lane's own condition (table[slot]==key)
    // has gone false. A lane that finished early is simply masked off
    // (idle) for every remaining round -- it does not let the warp move
    // on any sooner.
    int max_depth = 0;
    for (int d : probe_depths) max_depth = std::max(max_depth, d);
    int warp_rounds = max_depth + 1;   // depth 0 still needs 1 round (the initial check)
    long long lane_rounds_spent = (long long)WARP_SIZE * warp_rounds;
    long long useful_lane_rounds = 0;
    for (int d : probe_depths) useful_lane_rounds += (d + 1);
    long long wasted_lane_rounds = lane_rounds_spent - useful_lane_rounds;

    printf("deepest chain in this warp: lane 9's key needed %d probes, so the WHOLE warp\n", max_depth);
    printf("must execute %d rounds of the probing loop before any lane can move on\n\n", warp_rounds);

    printf("lane-rounds actually spent by the warp:  %2d lanes x %d rounds = %lld\n",
           WARP_SIZE, warp_rounds, lane_rounds_spent);
    printf("lane-rounds that did USEFUL probing work: %lld\n", useful_lane_rounds);
    printf("lane-rounds wasted on idle, masked-off lanes: %lld\n", wasted_lane_rounds);

    // Self-check: cluster keys 5..374 should show depths 0..9 in order;
    // every spread key should show depth 0.
    bool ok = true;
    for (int i = 0; i < 10; i++) if (probe_depths[i] != i) ok = false;
    for (int i = 10; i < WARP_SIZE; i++) if (probe_depths[i] != 0) ok = false;
    ok = ok && (max_depth == 9) && (warp_rounds == 10)
            && (lane_rounds_spent == 320) && (useful_lane_rounds == (55 + 22));

    printf("\nself-check: the ten collided keys show probe depths 0..9 in order, every\n");
    printf("spread key shows depth 0, and the warp-rounds/lane-rounds accounting matches\n");
    printf("hand computation: %s\n", ok ? "confirmed" : "MISMATCH");

    if (d_keys) cudaFree(d_keys);
    if (d_table) cudaFree(d_table);
    if (d_probe_depths) cudaFree(d_probe_depths);

    return ok ? 0 : 1;
}
```

**Compile and run:**

```bash
nvcc -arch=sm_80 -I$NVDIR/include -L$NVDIR/lib -lcudart 233_warp_divergence_probe_depth.cu -o 233_warp_divergence_probe_depth
LD_LIBRARY_PATH=$NVDIR/lib ./233_warp_divergence_probe_depth
```

**Sample input:** one warp of 32 lanes, each looking up a distinct key in a 41-slot open-addressing table -- ten keys engineered to collide into one probe chain (depths 0 through 9), twenty-two keys landing on untouched slots (depth 0 each).

**Sample output:**

```text
=== Appendix G.1: warp divergence from variable-depth probing ===

warp of 32 lanes, one key per lane -- lanes 0-9 all collide at home slot 5,
lanes 10-31 each land on an untouched home slot

attempting a genuine device allocation and kernel launch:
  cudaMalloc(keys)         -> cudaErrorInsufficientDriver (CUDA driver version is insufficient for CUDA runtime version)
  cudaMalloc(table)        -> cudaErrorInsufficientDriver (CUDA driver version is insufficient for CUDA runtime version)
  cudaMalloc(probe_depths) -> cudaErrorInsufficientDriver (CUDA driver version is insufficient for CUDA runtime version)

  (this environment has no usable device -- see Section 2.3. What follows is a
   host-side REPLAY of warp_lookup_kernel's exact per-lane probing logic.)

per-lane probe depth:
  [ 0 1 2 3 4 5 6 7 8 9 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ]

deepest chain in this warp: lane 9's key needed 9 probes, so the WHOLE warp
must execute 10 rounds of the probing loop before any lane can move on

lane-rounds actually spent by the warp:  32 lanes x 10 rounds = 320
lane-rounds that did USEFUL probing work: 77
lane-rounds wasted on idle, masked-off lanes: 243

self-check: the ten collided keys show probe depths 0..9 in order, every
spread key shows depth 0, and the warp-rounds/lane-rounds accounting matches
hand computation: confirmed
```

## G.2 Shared Memory Bank Conflicts: When "Parallel" Memory Isn't

### Intuition

Shared memory looks like one fast, flat array from a kernel's source code, but the hardware behind it is split into 32 independent banks, and only one memory transaction per bank can be serviced per cycle. When 32 lanes read 32 different banks, all 32 accesses complete together. When several lanes read DIFFERENT addresses that happen to fall in the SAME bank, the hardware has no choice but to service them one at a time -- an "N-way bank conflict" takes N times as long as the conflict-free case. This is a second, independent cost from warp divergence: divergence is about the INSTRUCTION stream splitting; a bank conflict happens even when every lane executes the identical instruction, purely because of WHERE each lane's data landed.

### The Concept, In Detail

```
shared memory, word-interleaved across 32 banks (word w lives in bank w % 32):

  bank:      0    1    2    3   ...  30   31
            [ ]  [ ]  [ ]  [ ]  ... [ ]  [ ]

  stride 1 (32 lanes, 32 different banks):        every lane -> its own bank
    lane 0 -> addr 0 -> bank 0        lane 1 -> addr 1 -> bank 1      ... all distinct
    ALL 32 SERVICED IN ONE TRANSACTION

  stride 32 (32 lanes, all the SAME bank, DIFFERENT addresses):
    lane 0 -> addr 0  -> bank 0      lane 1 -> addr 32 -> bank 0      ... all bank 0
    32 DIFFERENT ADDRESSES, ONE BANK -> 32-WAY CONFLICT -> 32 SERIALIZED TRANSACTIONS

  stride 0 (32 lanes, the SAME address):
    every lane -> addr 0 -> bank 0 -- but it is the SAME address, not just the same bank
    BROADCAST: one transaction serves all 32 lanes, free -- NOT a conflict
```

The distinction in that last case matters: sharing a bank is only expensive when the addresses actually differ. Thirty-two lanes reading the exact same shared counter is the cheapest possible access (one broadcast), while thirty-two lanes reading thirty-two DIFFERENT addresses that all happen to fall in bank 0 is the most expensive possible access (32 serialized transactions) -- despite both cases mapping every lane to "bank 0." The practical trigger for the expensive case is almost always a per-thread region sized as an exact multiple of 32 elements, accessed with that size as the stride -- exactly the shape a private per-thread stack frame, a private hash-table row, or a per-thread histogram sub-range can fall into without anyone intending it.

!!! warning "[COMMON TRAP] Confusing 'same bank' with 'conflict'"
    It is tempting to treat every case where multiple lanes' addresses reduce to the same bank number as equally expensive. The hardware's actual rule cares about DISTINCT ADDRESSES, not distinct lanes -- many lanes reading one identical address is a free broadcast, while a handful of lanes reading a handful of genuinely different addresses in the same bank is a real, serialized conflict. Padding a per-thread region to break a stride-32 pattern fixes the second case; it does nothing for (and is not needed for) the first.

### Code and Verification

```cpp
// 234_shared_memory_bank_conflicts.cu
//
// Appendix G.2 -- Shared memory is physically split into 32 equally-sized
// BANKS, interleaved so that consecutive 4-byte words land in consecutive
// banks (word w lives in bank w % 32). Within one warp, the hardware can
// service one memory transaction per bank per cycle -- if every lane in
// the warp reads or writes a DIFFERENT bank, all 32 accesses happen in
// parallel. If two or more lanes target the SAME bank at DIFFERENT
// addresses, those accesses serialize into as many separate transactions
// as there are distinct addresses in that bank -- an "N-way bank
// conflict" costs N times as long as a conflict-free access. There is one
// deliberate exception: if every lane in the warp reads the exact SAME
// address, the hardware broadcasts that one value to all of them in a
// single transaction, free of charge, even though every lane technically
// "shares a bank." This file computes bank assignments for four access
// patterns a real per-thread data structure slot (a private stack frame,
// a private hash-table row, a private histogram sub-range) could plausibly
// use, and confirms which ones conflict and which do not -- the bank
// arithmetic itself is pure address math, unaffected by execution order or
// scheduling, so it is exactly as trustworthy computed on the host as it
// would be measured on real hardware.
//
// Compile: nvcc -arch=sm_80 -I$NVDIR/include -L$NVDIR/lib -lcudart 234_shared_memory_bank_conflicts.cu -o 234_shared_memory_bank_conflicts
// Run:     LD_LIBRARY_PATH=$NVDIR/lib ./234_shared_memory_bank_conflicts

#include <algorithm>
#include <cstdio>
#include <vector>
#include <cuda_runtime.h>

#define WARP_SIZE 32
#define NUM_BANKS 32

// Real kernel: each lane computes the shared-memory address IT would
// touch under a given per-thread stride, and the bank that address maps
// to (address_in_words % 32). This is pure arithmetic -- no thread reads
// or writes another thread's data, so nothing here depends on scheduling
// order, only on the stride each lane was given.
__global__ void compute_addresses_kernel(int stride, int* addresses, int* banks) {
    int lane = threadIdx.x;
    int address = lane * stride;
    addresses[lane] = address;
    banks[lane] = address % NUM_BANKS;
}

void report(const char* call_name, cudaError_t err) {
    printf("  %-24s -> %-24s (%s)\n", call_name, cudaGetErrorName(err), cudaGetErrorString(err));
}

struct AccessResult {
    std::vector<int> addresses;
    std::vector<int> banks;
    bool all_same_address;
    int max_conflict_degree;   // among distinct addresses sharing a bank
};

// Host-side replay of the identical per-lane address/bank computation.
AccessResult replay_addresses(int stride) {
    AccessResult r;
    r.addresses.resize(WARP_SIZE);
    r.banks.resize(WARP_SIZE);
    for (int lane = 0; lane < WARP_SIZE; lane++) {
        int address = lane * stride;
        r.addresses[lane] = address;
        r.banks[lane] = address % NUM_BANKS;
    }
    r.all_same_address = true;
    for (int a : r.addresses) if (a != r.addresses[0]) r.all_same_address = false;

    // For each bank, count DISTINCT addresses mapped to it (broadcast --
    // identical addresses in the same bank -- costs nothing extra).
    r.max_conflict_degree = 1;
    for (int b = 0; b < NUM_BANKS; b++) {
        std::vector<int> distinct_addrs;
        for (int lane = 0; lane < WARP_SIZE; lane++) {
            if (r.banks[lane] != b) continue;
            bool seen = false;
            for (int a : distinct_addrs) if (a == r.addresses[lane]) seen = true;
            if (!seen) distinct_addrs.push_back(r.addresses[lane]);
        }
        r.max_conflict_degree = std::max(r.max_conflict_degree, (int)distinct_addrs.size());
    }
    return r;
}

void print_case(const char* label, int stride, const AccessResult& r) {
    printf("--- %s (stride = %d) ---\n", label, stride);
    printf("addresses: [ ");
    for (int a : r.addresses) printf("%d ", a);
    printf("]\n");
    printf("banks:     [ ");
    for (int b : r.banks) printf("%d ", b);
    printf("]\n");
    if (r.all_same_address) {
        printf("verdict: every lane reads the SAME address -> BROADCAST, a single free\n");
        printf("transaction serves all %d lanes despite sharing one bank\n\n", WARP_SIZE);
    } else if (r.max_conflict_degree == 1) {
        printf("verdict: every lane's address maps to a DIFFERENT bank -> conflict-free,\n");
        printf("all %d lanes serviced in one transaction\n\n", WARP_SIZE);
    } else {
        printf("verdict: worst bank holds %d DISTINCT addresses -> a %d-way conflict,\n",
               r.max_conflict_degree, r.max_conflict_degree);
        printf("that bank alone needs %d serialized transactions\n\n", r.max_conflict_degree);
    }
}

int main() {
    printf("=== Appendix G.2: shared-memory bank conflicts, four per-thread strides ===\n\n");

    printf("attempting a genuine device allocation and kernel launch:\n");
    int* d_addresses = nullptr;
    int* d_banks = nullptr;
    cudaError_t e1 = cudaMalloc(&d_addresses, WARP_SIZE * sizeof(int));
    report("cudaMalloc(addresses)", e1);
    cudaError_t e2 = cudaMalloc(&d_banks, WARP_SIZE * sizeof(int));
    report("cudaMalloc(banks)", e2);

    if (e1 == cudaSuccess && e2 == cudaSuccess) {
        compute_addresses_kernel<<<1, WARP_SIZE>>>(1, d_addresses, d_banks);
        cudaDeviceSynchronize();
        printf("  (kernel launch attempted -- would print device results here on real hardware)\n\n");
    } else {
        printf("\n  (this environment has no usable device -- see Section 2.3. What follows is a\n");
        printf("   host-side REPLAY of compute_addresses_kernel's exact address/bank arithmetic,\n");
        printf("   which is pure per-lane math and does not depend on scheduling.)\n\n");
    }

    AccessResult stride1 = replay_addresses(1);
    print_case("stride 1: each lane owns one contiguous private slot", 1, stride1);

    AccessResult stride2 = replay_addresses(2);
    print_case("stride 2: each lane's private region is 2 words wide", 2, stride2);

    AccessResult stride32 = replay_addresses(32);
    print_case("stride 32: each lane's private region is exactly 32 words wide", 32, stride32);

    AccessResult stride0 = replay_addresses(0);
    print_case("stride 0: every lane reads the same shared counter", 0, stride0);

    bool ok = (stride1.max_conflict_degree == 1) && !stride1.all_same_address
           && (stride2.max_conflict_degree == 2) && !stride2.all_same_address
           && (stride32.max_conflict_degree == 32) && !stride32.all_same_address
           && stride0.all_same_address;

    printf("self-check: stride 1 is conflict-free, stride 2 produces a 2-way conflict,\n");
    printf("stride 32 produces the worst-case 32-way conflict, and stride 0 is a free\n");
    printf("broadcast rather than a 32-way conflict despite sharing bank 0: %s\n",
           ok ? "confirmed" : "MISMATCH");

    printf("\nthe practical rule this generalizes to: whenever a per-thread structure is\n");
    printf("sized as an exact multiple of %d elements (a private stack frame, a private\n", NUM_BANKS);
    printf("hash-table row, a per-thread histogram sub-range) and threads access it with\n");
    printf("that stride, every thread's access lands in the same bank as stride 32 did\n");
    printf("here. The fix used throughout real CUDA code is the same one that fixes it\n");
    printf("above: access with stride 1 wherever the algorithm allows it, or pad the\n");
    printf("per-thread region by one extra element so its stride is no longer a multiple\n");
    printf("of %d.\n", NUM_BANKS);

    if (d_addresses) cudaFree(d_addresses);
    if (d_banks) cudaFree(d_banks);

    return ok ? 0 : 1;
}
```

**Compile and run:**

```bash
nvcc -arch=sm_80 -I$NVDIR/include -L$NVDIR/lib -lcudart 234_shared_memory_bank_conflicts.cu -o 234_shared_memory_bank_conflicts
LD_LIBRARY_PATH=$NVDIR/lib ./234_shared_memory_bank_conflicts
```

**Sample input:** one warp of 32 lanes, computing shared-memory addresses and bank numbers under four per-thread strides (1, 2, 32, and 0).

**Sample output:**

```text
=== Appendix G.2: shared-memory bank conflicts, four per-thread strides ===

attempting a genuine device allocation and kernel launch:
  cudaMalloc(addresses)    -> cudaErrorInsufficientDriver (CUDA driver version is insufficient for CUDA runtime version)
  cudaMalloc(banks)        -> cudaErrorInsufficientDriver (CUDA driver version is insufficient for CUDA runtime version)

  (this environment has no usable device -- see Section 2.3. What follows is a
   host-side REPLAY of compute_addresses_kernel's exact address/bank arithmetic,
   which is pure per-lane math and does not depend on scheduling.)

--- stride 1: each lane owns one contiguous private slot (stride = 1) ---
addresses: [ 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 ]
banks:     [ 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 ]
verdict: every lane's address maps to a DIFFERENT bank -> conflict-free,
all 32 lanes serviced in one transaction

--- stride 2: each lane's private region is 2 words wide (stride = 2) ---
addresses: [ 0 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 32 34 36 38 40 42 44 46 48 50 52 54 56 58 60 62 ]
banks:     [ 0 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 0 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 ]
verdict: worst bank holds 2 DISTINCT addresses -> a 2-way conflict,
that bank alone needs 2 serialized transactions

--- stride 32: each lane's private region is exactly 32 words wide (stride = 32) ---
addresses: [ 0 32 64 96 128 160 192 224 256 288 320 352 384 416 448 480 512 544 576 608 640 672 704 736 768 800 832 864 896 928 960 992 ]
banks:     [ 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ]
verdict: worst bank holds 32 DISTINCT addresses -> a 32-way conflict,
that bank alone needs 32 serialized transactions

--- stride 0: every lane reads the same shared counter (stride = 0) ---
addresses: [ 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ]
banks:     [ 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ]
verdict: every lane reads the SAME address -> BROADCAST, a single free
transaction serves all 32 lanes despite sharing one bank

self-check: stride 1 is conflict-free, stride 2 produces a 2-way conflict,
stride 32 produces the worst-case 32-way conflict, and stride 0 is a free
broadcast rather than a 32-way conflict despite sharing bank 0: confirmed

the practical rule this generalizes to: whenever a per-thread structure is
sized as an exact multiple of 32 elements (a private stack frame, a private
hash-table row, a per-thread histogram sub-range) and threads access it with
that stride, every thread's access lands in the same bank as stride 32 did
here. The fix used throughout real CUDA code is the same one that fixes it
above: access with stride 1 wherever the algorithm allows it, or pad the
per-thread region by one extra element so its stride is no longer a multiple
of 32.
```

## G.3 Atomic Contention: When Everyone Wants the Same Address

### Intuition

Chapter 7.1 already proved `atomicAdd` never loses an update, no matter how many threads target the same address at once -- correctness was never the question. Chapter 7.2 asked the other question: what does it cost when thousands of threads across an entire grid all contend for a handful of shared addresses? Every one of those threads' atomic operations serializes against every other thread's atomic operation to the same address, and Chapter 7.2's fix -- privatize into a per-block copy first, merge once at the end -- generalizes to any atomicAdd-based technique this book has built, not just histograms.

### The Concept, In Detail

```
naive: every thread's atomicAdd goes straight to ONE global counter

  block 0 (256 threads) --\
  block 1 (256 threads) ---\
  block 2 (256 threads) ----+--> [ global counter ] <-- 2048 threads,
  ...                      /                              2048 SERIALIZED
  block 7 (256 threads) --/                                 global atomics

privatized: each block aggregates locally first, ONE atomic reaches the global counter

  block 0: 256 local (shared-memory) atomics --> 1 global atomicAdd(count, 256) --\
  block 1: 256 local (shared-memory) atomics --> 1 global atomicAdd(count, 256) ---+--> [ global counter ]
  ...                                                                              /      8 global atomics,
  block 7: 256 local (shared-memory) atomics --> 1 global atomicAdd(count, 256) --/       NOT 2048
```

The naive version and the privatized version reserve the exact same 2048 output slots, in the exact same sense Chapter 7.2's naive and privatized histograms produced the exact same bucket counts -- privatization is a performance change, not a correctness change, in both cases. What differs is where the contention happens: the naive version makes every thread in the entire grid a potential contender for the same address, while the privatized version confines contention to each block's own 256 threads, then reduces the GLOBAL traffic to exactly one atomic per block -- an 8x reduction here (2048 to 8), the identical shape as Chapter 7.2's own 2048-to-128 histogram result, just with a different block size setting the reduction factor.

!!! warning "[COMMON TRAP] Assuming privatization always requires a fixed, known local count"
    This section's privatized kernel works cleanly because every thread contributes exactly one item, so a block's local count is always exactly `THREADS_PER_BLOCK`, known before the kernel even launches. Real workloads are rarely this uniform -- when different threads contribute different numbers of items (or none at all), the per-block LOCAL atomic count genuinely varies, and the single global `atomicAdd(global_count, local_count)` this section uses is exactly the right tool for that case too: it reserves however many slots that specific block actually needs, no more and no less, still at a cost of one global atomic per block rather than one per item.

### Code and Verification

```cpp
// 235_atomic_contention_privatization.cu
//
// Appendix G.3 -- Chapter 7.2 already proved this exact trade for
// histogram buckets: routing every thread's atomicAdd straight to a
// small number of GLOBAL addresses makes every one of those threads,
// across the ENTIRE grid, serialize against every other thread that
// happens to target the same address -- correctness is never in
// question (Chapter 7.1 already proved atomicAdd loses nothing), only
// speed. Chapter 7.2's fix was PRIVATIZATION: let each block accumulate
// into its own local copy first (contention limited to that one block's
// own threads), then merge once into the global structure. This file
// applies the identical idea to a different operation -- Chapter 8's
// own buffer-append slot reservation -- and counts exactly how many
// GLOBAL atomic operations each version issues for the same amount of
// useful work.
//
// Compile: nvcc -arch=sm_80 -I$NVDIR/include -L$NVDIR/lib -lcudart 235_atomic_contention_privatization.cu -o 235_atomic_contention_privatization
// Run:     LD_LIBRARY_PATH=$NVDIR/lib ./235_atomic_contention_privatization

#include <cstdio>
#include <vector>
#include <cuda_runtime.h>

#define NUM_BLOCKS 8
#define THREADS_PER_BLOCK 256
#define TOTAL_THREADS (NUM_BLOCKS * THREADS_PER_BLOCK)

// Real kernel, naive version: every one of the 2048 threads across the
// whole grid calls atomicAdd directly on ONE global counter. Chapter
// 8.1's own slot-reservation pattern, applied at full grid scale --
// correct, but every single thread's atomicAdd contends with every
// OTHER thread's atomicAdd, grid-wide.
__global__ void naive_append_kernel(int* global_count, int* slots) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int pos = atomicAdd(global_count, 1);
    slots[tid] = pos;
}

// Real kernel, privatized version: each block first aggregates its own
// 256 threads' claims into a SHARED (block-local) counter -- contention
// stays inside that one block. Only ONE thread per block then issues a
// single atomicAdd against the GLOBAL counter, reserving a contiguous
// range of exactly THREADS_PER_BLOCK slots for its entire block at once.
// Every thread's final position is that block's reserved base plus its
// own local offset -- identical final coverage, far fewer global atomics.
__global__ void privatized_append_kernel(int* global_count, int* slots) {
    __shared__ int local_count;
    __shared__ int block_base;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (threadIdx.x == 0) local_count = 0;
    __syncthreads();

    int local_pos = atomicAdd(&local_count, 1);   // contention limited to this block's 256 threads
    __syncthreads();

    if (threadIdx.x == 0) block_base = atomicAdd(global_count, local_count);   // ONE global atomic per block
    __syncthreads();

    slots[tid] = block_base + local_pos;
}

void report(const char* call_name, cudaError_t err) {
    printf("  %-24s -> %-24s (%s)\n", call_name, cudaGetErrorName(err), cudaGetErrorString(err));
}

// Host-side replay of the naive kernel: one global atomic per thread,
// in grid launch order (thread 0 of block 0 first, and so on).
std::vector<int> replay_naive(int& global_atomic_count) {
    std::vector<int> slots(TOTAL_THREADS);
    int global_count = 0;
    global_atomic_count = 0;
    for (int tid = 0; tid < TOTAL_THREADS; tid++) {
        int pos = global_count++;   // atomicAdd(global_count, 1)
        global_atomic_count++;
        slots[tid] = pos;
    }
    return slots;
}

// Host-side replay of the privatized kernel: 256 local (shared-memory)
// atomics per block, contained entirely within that block, plus exactly
// ONE global atomic per block to reserve its contiguous range.
std::vector<int> replay_privatized(int& global_atomic_count, int& local_atomic_count) {
    std::vector<int> slots(TOTAL_THREADS);
    int global_count = 0;
    global_atomic_count = 0;
    local_atomic_count = 0;
    for (int block = 0; block < NUM_BLOCKS; block++) {
        int local_count = 0;
        std::vector<int> local_pos(THREADS_PER_BLOCK);
        for (int lane = 0; lane < THREADS_PER_BLOCK; lane++) {
            local_pos[lane] = local_count++;   // atomicAdd(&local_count, 1) -- block-local only
            local_atomic_count++;
        }
        int block_base = global_count;
        global_count += local_count;           // atomicAdd(global_count, local_count) -- ONE per block
        global_atomic_count++;
        for (int lane = 0; lane < THREADS_PER_BLOCK; lane++) {
            int tid = block * THREADS_PER_BLOCK + lane;
            slots[tid] = block_base + local_pos[lane];
        }
    }
    return slots;
}

bool covers_every_slot_exactly_once(const std::vector<int>& slots) {
    std::vector<bool> seen(TOTAL_THREADS, false);
    for (int s : slots) {
        if (s < 0 || s >= TOTAL_THREADS || seen[s]) return false;
        seen[s] = true;
    }
    return true;
}

int main() {
    printf("=== Appendix G.3: atomic contention, naive vs. privatized append ===\n\n");
    printf("%d blocks x %d threads/block = %d threads, each reserving one output slot\n\n",
           NUM_BLOCKS, THREADS_PER_BLOCK, TOTAL_THREADS);

    printf("attempting a genuine device allocation and kernel launch:\n");
    int* d_global_count = nullptr;
    int* d_slots = nullptr;
    cudaError_t e1 = cudaMalloc(&d_global_count, sizeof(int));
    report("cudaMalloc(global_count)", e1);
    cudaError_t e2 = cudaMalloc(&d_slots, TOTAL_THREADS * sizeof(int));
    report("cudaMalloc(slots)", e2);

    if (e1 == cudaSuccess && e2 == cudaSuccess) {
        cudaMemset(d_global_count, 0, sizeof(int));
        naive_append_kernel<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(d_global_count, d_slots);
        cudaMemset(d_global_count, 0, sizeof(int));
        privatized_append_kernel<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(d_global_count, d_slots);
        cudaDeviceSynchronize();
        printf("  (kernel launches attempted -- would print device results here on real hardware)\n\n");
    } else {
        printf("\n  (this environment has no usable device -- see Section 2.3. What follows is a\n");
        printf("   host-side REPLAY of both kernels' exact atomic-reservation logic.)\n\n");
    }

    int naive_global_atomics = 0;
    std::vector<int> naive_slots = replay_naive(naive_global_atomics);

    int privatized_global_atomics = 0, privatized_local_atomics = 0;
    std::vector<int> privatized_slots = replay_privatized(privatized_global_atomics, privatized_local_atomics);

    bool naive_ok = covers_every_slot_exactly_once(naive_slots);
    bool privatized_ok = covers_every_slot_exactly_once(privatized_slots);

    printf("naive version:      %d global atomicAdd calls (one per thread, grid-wide)\n", naive_global_atomics);
    printf("  every one of the %d threads contends against every other thread's\n", TOTAL_THREADS);
    printf("  atomicAdd, no matter which block it belongs to\n");
    printf("  every slot 0..%d filled exactly once: %s\n\n", TOTAL_THREADS - 1, naive_ok ? "confirmed" : "MISMATCH");

    printf("privatized version: %d global atomicAdd calls (one per block)\n", privatized_global_atomics);
    printf("  plus %d shared-memory atomicAdd calls, but each one only contends against\n", privatized_local_atomics);
    printf("  the %d other threads in its OWN block, never against another block\n", THREADS_PER_BLOCK - 1);
    printf("  every slot 0..%d filled exactly once: %s\n\n", TOTAL_THREADS - 1, privatized_ok ? "confirmed" : "MISMATCH");

    double reduction = (double)naive_global_atomics / (double)privatized_global_atomics;
    printf("global atomic traffic reduced from %d to %d: %.0fx fewer grid-wide contention points\n",
           naive_global_atomics, privatized_global_atomics, reduction);
    printf("(the same trade Chapter 7.2 measured for histogram buckets: 2048 global atomics\n");
    printf("down to 128, a 16x reduction, from privatizing into per-block shared memory first)\n");

    bool ok = naive_ok && privatized_ok
           && (naive_global_atomics == TOTAL_THREADS)
           && (privatized_global_atomics == NUM_BLOCKS)
           && (privatized_local_atomics == TOTAL_THREADS)
           && (reduction == (double)THREADS_PER_BLOCK);

    printf("\nself-check: both versions place every thread in a unique slot, the naive\n");
    printf("version issues exactly %d global atomics, and privatization cuts that to\n", TOTAL_THREADS);
    printf("exactly %d (a %dx reduction, matching the block size exactly): %s\n",
           NUM_BLOCKS, THREADS_PER_BLOCK, ok ? "confirmed" : "MISMATCH");

    if (d_global_count) cudaFree(d_global_count);
    if (d_slots) cudaFree(d_slots);

    return ok ? 0 : 1;
}
```

**Compile and run:**

```bash
nvcc -arch=sm_80 -I$NVDIR/include -L$NVDIR/lib -lcudart 235_atomic_contention_privatization.cu -o 235_atomic_contention_privatization
LD_LIBRARY_PATH=$NVDIR/lib ./235_atomic_contention_privatization
```

**Sample input:** 8 blocks of 256 threads each (2048 threads total), every thread reserving one output slot, compared naive-global-atomic-per-thread against privatized-local-then-one-global-atomic-per-block.

**Sample output:**

```text
=== Appendix G.3: atomic contention, naive vs. privatized append ===

8 blocks x 256 threads/block = 2048 threads, each reserving one output slot

attempting a genuine device allocation and kernel launch:
  cudaMalloc(global_count) -> cudaErrorInsufficientDriver (CUDA driver version is insufficient for CUDA runtime version)
  cudaMalloc(slots)        -> cudaErrorInsufficientDriver (CUDA driver version is insufficient for CUDA runtime version)

  (this environment has no usable device -- see Section 2.3. What follows is a
   host-side REPLAY of both kernels' exact atomic-reservation logic.)

naive version:      2048 global atomicAdd calls (one per thread, grid-wide)
  every one of the 2048 threads contends against every other thread's
  atomicAdd, no matter which block it belongs to
  every slot 0..2047 filled exactly once: confirmed

privatized version: 8 global atomicAdd calls (one per block)
  plus 2048 shared-memory atomicAdd calls, but each one only contends against
  the 255 other threads in its OWN block, never against another block
  every slot 0..2047 filled exactly once: confirmed

global atomic traffic reduced from 2048 to 8: 256x fewer grid-wide contention points
(the same trade Chapter 7.2 measured for histogram buckets: 2048 global atomics
down to 128, a 16x reduction, from privatizing into per-block shared memory first)

self-check: both versions place every thread in a unique slot, the naive
version issues exactly 2048 global atomics, and privatization cuts that to
exactly 8 (a 256x reduction, matching the block size exactly): confirmed
```

## G.4 A Diagnostic Checklist for Recognizing These Failure Modes

None of this appendix's three failure modes announce themselves with an error message -- a kernel suffering from any of them still produces the exactly correct answer, every single time, which is precisely what makes them easy to miss. The following questions are the ones worth asking of any data-structure kernel that runs correctly but slower than its raw operation count would predict.

Does any loop's trip count depend on the data a thread happens to be processing, rather than on its thread index alone? A hash table probe chain, a tree traversal depth, a linked-list length, and a work-stealing retry loop are all examples -- anywhere the answer is yes, warp divergence (Section G.1) is worth measuring, by the same issue-pass or warp-rounds accounting Chapter 4 and this appendix both used, before assuming the slowdown must be memory bandwidth or occupancy instead.

Is any per-thread region of shared memory sized as an exact multiple of 32 elements, and accessed by other threads using that size as a stride? A private stack frame, a private row of a block-local hash table, and a per-thread histogram sub-range are the recurring shapes. Section G.2's fix -- padding the per-thread region by one element, or restructuring the access to stride 1 -- is worth trying specifically when a kernel's shared-memory traffic looks correct but its measured throughput (on real hardware, via Appendix E's `cudaEvent_t` technique) falls well short of the bandwidth the algorithm's own operation count would predict.

Does more than a handful of threads across DIFFERENT blocks call `atomicAdd` (or any other atomic) on the SAME address? If the answer is yes and the number of distinct contended addresses is small relative to the thread count, Section G.3's privatize-then-merge pattern is the fix Chapter 7.2 already proved for histograms and this appendix reapplied to buffer-append slot reservation -- and the same idea generalizes to any atomicAdd-based technique this book has built, not only those two.

## Appendix Summary

- Warp divergence is not limited to explicit `if`/`else` branches -- any loop whose trip count varies by lane (a probe chain, a tree depth, a linked-list length) forces the whole warp to run at the pace of its slowest lane, with every faster lane simply masked off and idle for the remaining rounds, exactly the cost Chapter 1 and Chapter 4 already measured for branch-shaped divergence.
- Shared memory's 32 banks service one transaction per bank per cycle; distinct addresses landing in the same bank serialize into an N-way conflict, but identical addresses landing in the same bank broadcast for free -- the expensive case is almost always a per-thread region sized as an exact multiple of 32 and accessed with that stride, fixed by padding or by switching to a stride-1 access pattern.
- Atomic operations are always correct under contention (Chapter 7.1), but heavy contention on a small number of shared addresses is a real, measurable performance cost -- Chapter 7.2's privatize-locally-then-merge-once pattern generalizes cleanly from histogram buckets to any atomicAdd-based slot reservation this book has built, including Chapter 8's own buffer-append counter, reducing global atomic traffic by a factor equal to the block size.
- Section G.4's diagnostic checklist collects the three questions worth asking of any correct-but-slower-than-expected kernel: does a loop's trip count depend on the data, is shared memory being accessed at a stride that is a multiple of 32, and are many threads across different blocks contending for the same atomic address.
