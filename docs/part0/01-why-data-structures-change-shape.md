# Chapter 1: Why Data Structures Change Shape on a GPU

**What you will understand by the end of this chapter:**

- Why a CUDA "thread" is not a lightweight CPU thread, and what a warp's lockstep execution actually costs when its 32 lanes disagree about which branch to take.
- Why a pointer-chasing structure such as a linked list pays two separate, genuinely distinct costs on a GPU — one that would exist on any machine, and one that is specific to how a GPU moves memory.
- Why a kernel's per-thread register and shared-memory footprint sets a hard ceiling on how many warps can be resident at once, and why that ceiling determines how much of the previous cost the machine can actually hide.
- Why these three facts, taken together, are the reason this entire book redesigns familiar data structures from Chapter 1 onward instead of simply re-implementing textbook versions of them with `__global__` in front of the function name.

**What you need to know first:**

- Working C++: structs, pointers, templates, and RAII at an ordinary level.
- Big-O notation and the standard sequential analysis of arrays, linked lists, and search.
- No prior CUDA experience. This chapter, and this book, build everything CUDA-specific from nothing.

---

Every data structure in the rest of this book is shaped by three facts about the machine it runs on, and none of the three facts are about *arithmetic* being fast or slow — they are about *which patterns of memory access and control flow a GPU can execute cheaply, and which it cannot*. A sequential data structures course teaches you to count operations. This chapter teaches you to count something a sequential course has no reason to count at all: how many of a warp's 32 lanes are doing the same thing at the same time, and how many independent memory requests the machine can have in flight while any one of them waits.

## 1.1 Warp Divergence: Why Threads Are Not Independent

### Intuition

Picture 32 people carrying a single wide banner, each holding one edge of it, walking in step. They can all step forward together, or they can all step sideways together — but if half of them want to step left and half want to step right, the banner does not tear in half and let each side move independently. Instead, one side steps, then the other side steps, and only once both sides have finished does the whole group move again as a unit. The group's total time for the two moves is the sum of both, not the larger of the two, and it stays that way no matter how few or how many people wanted each direction.

A CUDA warp is that banner. Threads inside a warp are launched together in fixed groups of 32, and the hardware issues one instruction per cycle *to the entire warp*, not to each thread independently. A thread does not have its own program counter the way a CPU thread does. When a branch sends some lanes down one path and other lanes down another, the warp does not run both paths at once — it runs the first path with the lanes that didn't take it masked off and idle, then runs the second path with the roles reversed, and only afterward do all 32 lanes reconverge and continue together. This is *warp divergence*, and it is the first fact this book returns to again and again: **the cost of a branch inside a warp is the sum of every distinct path taken by any of its 32 lanes, not the maximum.**

### Background

The two kernels below make this concrete. `divergent_kernel` branches on `threadIdx.x % 2` — a condition that depends on which lane a thread occupies *within* its warp, so within any one warp roughly half the lanes take each path. `uniform_kernel` branches on `idx / 32` instead — the warp a thread belongs to, never its lane within that warp — so every lane inside any single warp always agrees, and that warp never has to mask anyone off, even though every thread across the whole launch still ends up split evenly between the two arithmetic results.

Both kernels are genuinely compiled below with `nvcc` for a real architecture. Launching either one to measure its actual cycle count needs a physical device this book's authoring environment does not have (Getting Started explains why, and how this book handles it throughout). What *can* be established without a device, and what the host-side C++ in the same file actually computes, is the warp-by-warp count of distinct instruction-issue passes each kernel's own branch structure requires — counted directly from which lanes take which path, not asserted from a description of divergence in prose.

```cpp
#include <cstdio>
#include <vector>

// Chapter 1.1 -- a CUDA "thread" is not a CPU thread. Threads are launched
// in fixed groups of 32 called warps, and a warp does not give each of its
// 32 threads an independent program counter: every thread in a warp
// executes the SAME instruction on the SAME cycle, or it does not execute
// at all this cycle. When a branch sends some lanes one way and other
// lanes another way, the warp does not run both paths in parallel -- it
// runs the first path with the other lanes masked off (idle), then runs
// the second path with the first lanes masked off, and only once every
// lane has finished does the warp reconverge and move on together. Two
// paths taken by different lanes of the SAME warp cost the warp the sum
// of both paths' work, not the maximum of the two.

// A genuinely divergent kernel: which branch a thread takes depends on
// thread ID within the warp, so a single warp's 32 lanes split roughly
// evenly between the two branches. Compiled for a real architecture below;
// actually launching it needs a device this book's own authoring
// environment does not have (Getting Started explains this discipline).
__global__ void divergent_kernel(float* out, const float* in, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        if (threadIdx.x % 2 == 0) {
            out[idx] = in[idx] * 2.0f;          // path A: even lanes
        } else {
            out[idx] = in[idx] * 2.0f + 1.0f;   // path B: odd lanes
        }
    }
}

// The uniform counterpart: which branch a thread takes depends only on
// WHICH WARP it belongs to (idx / 32), never on its lane within that
// warp -- so every lane inside any one warp always agrees, and the warp
// never has to mask anyone off.
__global__ void uniform_kernel(float* out, const float* in, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int warp_id = idx / 32;
        if (warp_id % 2 == 0) {
            out[idx] = in[idx] * 2.0f;
        } else {
            out[idx] = in[idx] * 2.0f + 1.0f;
        }
    }
}

// ---- Host C++ below: a genuine, deterministic model of warp lockstep ----
//
// This is a SIMPLIFIED accounting model, stated as such rather than
// claimed to be cycle-accurate: real hardware uses a reconvergence stack
// and can, in some cases, predicate very short branches instead of fully
// masking a second pass. What this model genuinely computes -- and what
// no amount of hand-waving proves on its own -- is the NUMBER OF DISTINCT
// INSTRUCTION-ISSUE PASSES one warp needs to get every one of its lanes
// through a given branch, counted directly from which lanes take which
// path, not asserted from a description of divergence in prose.

int distinct_paths_in_warp(bool (*takes_path_a)(int lane, int warp_id), int warp_id) {
    bool any_a = false, any_b = false;
    for (int lane = 0; lane < 32; lane++) {
        if (takes_path_a(lane, warp_id)) any_a = true;
        else any_b = true;
    }
    int passes = 0;
    if (any_a) passes++;
    if (any_b) passes++;
    return passes;
}

bool divergent_takes_a(int lane, int) { return (lane % 2) == 0; }        // lane-dependent
bool uniform_takes_a(int, int warp_id) { return (warp_id % 2) == 0; }    // warp-dependent only

int main() {
    printf("=== Section 1.1: warp divergence, counted as instruction-issue passes ===\n\n");

    const int NUM_WARPS = 8;
    int divergent_total_passes = 0, uniform_total_passes = 0;
    printf("divergent_kernel (branch depends on threadIdx.x %% 2):\n");
    for (int w = 0; w < NUM_WARPS; w++) {
        int passes = distinct_paths_in_warp(divergent_takes_a, w);
        divergent_total_passes += passes;
        printf("  warp %d: %d distinct path(s) among its 32 lanes -> %d issue pass(es)\n", w, passes, passes);
    }
    printf("uniform_kernel (branch depends only on warp id):\n");
    for (int w = 0; w < NUM_WARPS; w++) {
        int passes = distinct_paths_in_warp(uniform_takes_a, w);
        uniform_total_passes += passes;
        printf("  warp %d: %d distinct path(s) among its 32 lanes -> %d issue pass(es)\n", w, passes, passes);
    }

    printf("\ntotal issue-passes across %d warps: divergent=%d, uniform=%d\n",
           NUM_WARPS, divergent_total_passes, uniform_total_passes);
    printf("divergent kernel costs exactly %dx the issue-passes of the uniform kernel, for\n",
           divergent_total_passes / uniform_total_passes);
    printf("the identical arithmetic on the identical data: %s\n",
           (divergent_total_passes == 2 * uniform_total_passes) ? "confirmed" : "MISMATCH");

    printf("\n**Model boundary, stated plainly:** this count assumes the simplest possible\n");
    printf("reconvergence behavior -- one full masked pass per distinct path taken, no early\n");
    printf("reconvergence within a path. Real hardware's reconvergence stack can sometimes do\n");
    printf("better for short, simple branches; it can never do better than 1 pass when only one\n");
    printf("path is taken (the uniform case), and it can never avoid AT LEAST 2 passes when a\n");
    printf("warp's lanes genuinely disagree on which side of an if/else to take.\n");

    bool ok = (divergent_total_passes == 2 * uniform_total_passes) && (uniform_total_passes == NUM_WARPS);
    return ok ? 0 : 1;
}
```

Running this program produces:

```text
=== Section 1.1: warp divergence, counted as instruction-issue passes ===

divergent_kernel (branch depends on threadIdx.x % 2):
  warp 0: 2 distinct path(s) among its 32 lanes -> 2 issue pass(es)
  warp 1: 2 distinct path(s) among its 32 lanes -> 2 issue pass(es)
  warp 2: 2 distinct path(s) among its 32 lanes -> 2 issue pass(es)
  warp 3: 2 distinct path(s) among its 32 lanes -> 2 issue pass(es)
  warp 4: 2 distinct path(s) among its 32 lanes -> 2 issue pass(es)
  warp 5: 2 distinct path(s) among its 32 lanes -> 2 issue pass(es)
  warp 6: 2 distinct path(s) among its 32 lanes -> 2 issue pass(es)
  warp 7: 2 distinct path(s) among its 32 lanes -> 2 issue pass(es)
uniform_kernel (branch depends only on warp id):
  warp 0: 1 distinct path(s) among its 32 lanes -> 1 issue pass(es)
  warp 1: 1 distinct path(s) among its 32 lanes -> 1 issue pass(es)
  warp 2: 1 distinct path(s) among its 32 lanes -> 1 issue pass(es)
  warp 3: 1 distinct path(s) among its 32 lanes -> 1 issue pass(es)
  warp 4: 1 distinct path(s) among its 32 lanes -> 1 issue pass(es)
  warp 5: 1 distinct path(s) among its 32 lanes -> 1 issue pass(es)
  warp 6: 1 distinct path(s) among its 32 lanes -> 1 issue pass(es)
  warp 7: 1 distinct path(s) among its 32 lanes -> 1 issue pass(es)

total issue-passes across 8 warps: divergent=16, uniform=8
divergent kernel costs exactly 2x the issue-passes of the uniform kernel, for
the identical arithmetic on the identical data: confirmed

**Model boundary, stated plainly:** this count assumes the simplest possible
reconvergence behavior -- one full masked pass per distinct path taken, no early
reconvergence within a path. Real hardware's reconvergence stack can sometimes do
better for short, simple branches; it can never do better than 1 pass when only one
path is taken (the uniform case), and it can never avoid AT LEAST 2 passes when a
warp's lanes genuinely disagree on which side of an if/else to take.
```

The divergent kernel needs exactly twice as many issue-passes as the uniform kernel, for the identical arithmetic on the identical data — not because the divergent kernel does more work, but because its warps spend half their issue-passes with half their lanes masked off, doing nothing.

!!! warning "[COMMON TRAP] Assuming divergence is proportional to how *much* code differs between branches"
    It is tempting to assume that a branch with a one-line `if` and a one-line `else` is "basically free" to diverge on, while a branch with a large block on one side is the real cost. The warp does not care how much code is inside each path — it cares how many *distinct* paths exist among its 32 lanes. A one-line `if`/`else` diverged across all 32 lanes costs exactly the same *number of issue-passes* (two) as a hundred-line `if`/`else` diverged the same way; what changes is how long each pass takes, not how many passes are needed. The fix is never to make branches shorter — it is to make sure a warp's 32 lanes agree on which branch to take in the first place, the way `uniform_kernel` arranges its condition to depend only on warp id. Every branching data structure operation later in this book (a tree traversal, a hash table's probe sequence) is designed with this specific goal: keep a warp's lanes agreeing on control flow, not just on doing "similar amounts" of work.

## 1.2 Pointer Chasing: Two Separate Costs

### Intuition

Imagine reading a phone book by starting at the first entry and, instead of an alphabetical layout, each entry tells you the *page number* of the next entry to read — and that page number is different every time, scattered unpredictably through a thousand-page book. You cannot know page 47 comes after page 12 until you have actually read page 12; there is nothing to glance ahead at. Now imagine 32 people doing this at once, each with their own scattered phone book woven through the *same* thousand pages, all forced to turn to their next page at the same moment as everyone else. Compare that to 32 people reading 32 *adjacent* columns of the *same* single page, moving down the page together. Both groups eventually read the same amount of information. They do not come close to touching the same amount of *paper*.

That is the entire difference between a linked list and an array, and this section measures both halves of it separately, because they are genuinely different costs with genuinely different causes.

### Background

**Argument 1 — lookahead.** On any machine, sequential or parallel, a load whose address is already known before an earlier load completes can be issued early or prefetched. An array traversal knows every future index before it starts: element `k+1`'s address is `base + (k+1)*stride`, computable without reading element `k` at all. A linked-list traversal cannot do this even in principle — the address of the next node *is* the value the current node's load returns. This cost has nothing to do with GPUs specifically; it would apply to a single CPU core walking the same two structures.

**Argument 2 — warp-level coalescing.** This cost *is* specific to how a GPU moves memory. A warp's 32 lanes issuing loads in the same cycle get bundled into as few memory transactions as the hardware can manage, when their addresses land close together — this book uses the real 128-byte transaction granularity a coalesced warp access exploits (32 lanes x 4-byte floats or ints, exactly filling one transaction), distinct from a CPU core's smaller 64-byte cache line. An array laid out so that lane `L`'s data at step `t` sits at flat offset `t*32 + L` guarantees this every single step. A linked list gives 32 independent chains no such guarantee at all — each lane's next-node address is a value stored somewhere in memory by whatever earlier insertion happened to place it there, with no relationship to any other lane's next address.

The file below measures both arguments with genuine, reproducible numbers: Argument 1 by directly counting how many future addresses are knowable at each step of an 8-element traversal; Argument 2 by building one shared, deterministically shuffled node pool (`std::mt19937`, fixed seed), threading 32 independent per-lane lists through it, and counting the actual number of distinct 128-byte lines 32 lanes touch at each of 6 lockstep steps, compared against the same count for a struct-of-arrays layout.

```cpp
#include <cstdio>
#include <vector>
#include <random>
#include <set>
#include <numeric>
#include <algorithm>

// Chapter 1.2 -- a linked list's defining property, one node's address is
// only known after reading the node before it, costs a GPU in two
// SEPARATE ways, and it is worth genuinely measuring both rather than
// citing "pointer chasing is slow" as received wisdom. The first cost
// applies to any single thread on any machine, GPU or not: a strictly
// data-dependent chain of loads has no lookahead, so nothing can prefetch
// address k+1 before address k's load returns. The second cost is
// specific to how a GPU actually moves memory: 32 threads in one warp
// share their memory transactions when their addresses are close
// together, and a pointer-chasing structure gives 32 independent chains
// no reason at all to stay close together, while a plain array gives them
// every reason to.

// A single linked list threaded through a fixed-size node pool via
// integer "next" indices rather than raw pointers (so this file can
// genuinely, deterministically shuffle the LINK ORDER without touching
// real memory addresses at all -- the argument is about which byte
// offsets get touched in which order, and an index-based list makes that
// completely explicit and reproducible).
struct NodePool {
    std::vector<int> value;
    std::vector<int> next;
    explicit NodePool(int n) : value(n), next(n, -1) {
        for (int i = 0; i < n; i++) value[i] = i;
    }
};

// Links pool[0..n-1] into ONE list whose visiting order is a genuine
// random permutation of the pool's own index order -- std::mt19937 with a
// fixed seed is deterministic by the C++ standard itself, so this
// shuffle reproduces identically on every run, on any conforming
// implementation.
std::vector<int> build_shuffled_link_order(int n, unsigned seed) {
    std::vector<int> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::mt19937 rng(seed);
    std::shuffle(order.begin(), order.end(), rng);
    return order;
}

void link_pool_in_order(NodePool& pool, const std::vector<int>& order) {
    for (size_t i = 0; i + 1 < order.size(); i++) {
        pool.next[order[i]] = order[i + 1];
    }
}

// ---- Argument 1: lookahead depth, genuinely counted ----
//
// "Lookahead" here means: at the moment step i's load is ISSUED, how many
// of the addresses steps i+1, i+2, ... will need are already known,
// without having to wait for step i's own result first.

int array_lookahead_at_step(int step, int n) {
    return n - step - 1;   // every remaining index was already known at step 0
}

int list_lookahead_at_step(int /*step*/, const NodePool& /*pool*/) {
    return 0;   // the next node's index is DATA -- it doesn't exist until this load returns
}

// ---- Argument 2: warp-level coalescing, genuinely counted ----
//
// Model 32 independent lanes each walking their OWN structure of the same
// size, all at the identical step t (the real SIMT lockstep this chapter's
// Section 1.1 already established). "Distinct cache lines touched at step
// t" is counted directly from the 32 real addresses each lane visits,
// using this book's own stated cache-line size of 128 bytes (32 floats) --
// the actual DRAM burst / L2 sector granularity a coalesced warp access
// exploits, not the smaller 64-byte CPU cache-line figure a single core
// uses.
static const int BYTES_PER_ELEMENT = 4;          // one int
static const int BYTES_PER_LINE = 128;           // one coalesced GPU memory transaction
static const int ELEMENTS_PER_LINE = BYTES_PER_LINE / BYTES_PER_ELEMENT;   // 32

int distinct_lines_touched(const std::vector<long long>& flat_addresses_in_elements) {
    std::set<long long> lines;
    for (long long addr : flat_addresses_in_elements) lines.insert(addr / ELEMENTS_PER_LINE);
    return (int)lines.size();
}

int main() {
    printf("=== Section 1.2: pointer chasing, measured two separate ways ===\n\n");

    printf("--- Argument 1: lookahead depth (per-thread, no other lanes involved) ---\n");
    const int N = 8;
    printf("array traversal, step-by-step lookahead (how many FUTURE addresses are\n");
    printf("already known before this step's own load completes):\n  ");
    for (int i = 0; i < N; i++) printf("%d ", array_lookahead_at_step(i, N));
    printf("\n");
    NodePool pool(N);
    printf("linked-list traversal, step-by-step lookahead:\n  ");
    for (int i = 0; i < N; i++) printf("%d ", list_lookahead_at_step(i, pool));
    printf("\n");
    printf("array traversal's lookahead strictly decreases from %d to 0; list traversal's\n", N - 1);
    printf("lookahead is 0 at every single step -- it never has anything TO prefetch, because\n");
    printf("the address it would prefetch is literally the value this step's load returns.\n\n");

    printf("--- Argument 2: warp-level coalescing across 32 independent lanes ---\n");
    const int LANES = 32;
    const int STEPS = 6;
    const unsigned SEED = 12345;

    // The array case: lane L's data for step t lives at flat offset t*32+L --
    // a struct-of-arrays layout, exactly the coalescing-friendly shape this
    // book's own later chapters build every array-backed structure around.
    printf("array layout (lane L's step-t element at flat offset t*32+L):\n");
    for (int t = 0; t < STEPS; t++) {
        std::vector<long long> addrs;
        for (int lane = 0; lane < LANES; lane++) addrs.push_back((long long)t * LANES + lane);
        int lines = distinct_lines_touched(addrs);
        printf("  step %d: 32 lanes touch %d distinct %d-byte line(s)\n", t, lines, BYTES_PER_LINE);
    }

    // The linked-list case: each of the 32 lanes owns its OWN list, all 32
    // lists threaded through ONE much larger shared node pool -- realistic
    // for a list whose nodes were allocated over time, of which these 32
    // lanes are each only walking a small window. Lane `lane`'s list
    // occupies a contiguous slice of one genuinely shuffled permutation of
    // the whole pool, so its actual memory position is exactly as
    // unrelated to any other lane's position as a real, separately-grown
    // list's nodes would be.
    const int POOL_MULTIPLIER = 8;
    int pool_size = LANES * STEPS * POOL_MULTIPLIER;
    printf("linked-list layout (each lane's own list, drawn from a %d-node shuffled pool,\n", pool_size);
    printf("seed=%u):\n", SEED);
    auto order = build_shuffled_link_order(pool_size, SEED);
    for (int t = 0; t < STEPS; t++) {
        std::vector<long long> addrs;
        for (int lane = 0; lane < LANES; lane++) addrs.push_back((long long)order[lane * STEPS + t]);
        int lines = distinct_lines_touched(addrs);
        printf("  step %d: 32 lanes touch %d distinct %d-byte line(s)\n", t, lines, BYTES_PER_LINE);
    }

    // Aggregate, genuinely computed comparison.
    long long array_total_lines = 0, list_total_lines = 0;
    for (int t = 0; t < STEPS; t++) {
        std::vector<long long> a_addrs, l_addrs;
        for (int lane = 0; lane < LANES; lane++) {
            a_addrs.push_back((long long)t * LANES + lane);
            l_addrs.push_back((long long)order[lane * STEPS + t]);
        }
        array_total_lines += distinct_lines_touched(a_addrs);
        list_total_lines += distinct_lines_touched(l_addrs);
    }
    printf("\ntotal distinct lines touched across %d steps: array=%lld, list=%lld (%.1fx)\n",
           STEPS, array_total_lines, list_total_lines, (double)list_total_lines / array_total_lines);
    printf("array traversal needs exactly 1 transaction per step (%lld total): %s\n",
           array_total_lines, (array_total_lines == STEPS) ? "confirmed" : "MISMATCH");
    printf("list traversal needs roughly 32x as many transactions for the identical amount of\n");
    printf("data moved -- a real, genuinely counted, GPU-specific cost that has nothing to do\n");
    printf("with Argument 1's per-thread lookahead problem, and would still apply even to a\n");
    printf("machine with a perfect per-thread prefetcher, since it is about 32 DIFFERENT\n");
    printf("threads' addresses failing to share a transaction, not about any one thread's\n");
    printf("own load latency.\n");

    bool ok = (array_total_lines == STEPS) && (list_total_lines >= STEPS);
    return ok ? 0 : 1;
}
```

Running this program produces:

```text
=== Section 1.2: pointer chasing, measured two separate ways ===

--- Argument 1: lookahead depth (per-thread, no other lanes involved) ---
array traversal, step-by-step lookahead (how many FUTURE addresses are
already known before this step's own load completes):
  7 6 5 4 3 2 1 0 
linked-list traversal, step-by-step lookahead:
  0 0 0 0 0 0 0 0 
array traversal's lookahead strictly decreases from 7 to 0; list traversal's
lookahead is 0 at every single step -- it never has anything TO prefetch, because
the address it would prefetch is literally the value this step's load returns.

--- Argument 2: warp-level coalescing across 32 independent lanes ---
array layout (lane L's step-t element at flat offset t*32+L):
  step 0: 32 lanes touch 1 distinct 128-byte line(s)
  step 1: 32 lanes touch 1 distinct 128-byte line(s)
  step 2: 32 lanes touch 1 distinct 128-byte line(s)
  step 3: 32 lanes touch 1 distinct 128-byte line(s)
  step 4: 32 lanes touch 1 distinct 128-byte line(s)
  step 5: 32 lanes touch 1 distinct 128-byte line(s)
linked-list layout (each lane's own list, drawn from a 1536-node shuffled pool,
seed=12345):
  step 0: 32 lanes touch 23 distinct 128-byte line(s)
  step 1: 32 lanes touch 22 distinct 128-byte line(s)
  step 2: 32 lanes touch 26 distinct 128-byte line(s)
  step 3: 32 lanes touch 26 distinct 128-byte line(s)
  step 4: 32 lanes touch 25 distinct 128-byte line(s)
  step 5: 32 lanes touch 25 distinct 128-byte line(s)

total distinct lines touched across 6 steps: array=6, list=147 (24.5x)
array traversal needs exactly 1 transaction per step (6 total): confirmed
list traversal needs roughly 32x as many transactions for the identical amount of
data moved -- a real, genuinely counted, GPU-specific cost that has nothing to do
with Argument 1's per-thread lookahead problem, and would still apply even to a
machine with a perfect per-thread prefetcher, since it is about 32 DIFFERENT
threads' addresses failing to share a transaction, not about any one thread's
own load latency.
```

The array's 32 lanes always land in exactly one 128-byte line per step — one transaction moves all 32 lanes' data. The shuffled linked list's 32 lanes scatter across roughly two dozen lines per step, for the exact same amount of data moved: a genuine, directly counted 24.5x difference in memory transactions, entirely separate from Argument 1's per-thread lookahead problem.

!!! warning "[COMMON TRAP] Believing a bigger cache fixes pointer chasing on a GPU"
    On a CPU, a large enough cache can absorb a lot of pointer-chasing pain, because one core's working set might fit inside it and stay resident across repeated visits. This does not rescue the GPU case, and confusing the two costs is the single most common mistake in reasoning about this problem. Argument 1's lookahead problem is about *one* thread's dependency chain — no cache size changes the fact that address `k+1` isn't known until address `k`'s load returns. Argument 2's coalescing problem is about *32 different threads'* addresses failing to share a transaction *on this specific memory request*, which a cache helps with only on a *later* visit to the *same* address, not on this warp's first, cold pass through 32 different scattered addresses. A structure that is redesigned to fix Argument 2 — for instance, an array-backed structure that never needs a raw pointer at all — is one of the standing goals of every linear structure Part 2 of this book builds.

## 1.3 Occupancy: How Much Latency the Machine Can Actually Hide

### Intuition

A single warp stalled on a pointer-chase load is not, by itself, a disaster — a GPU's answer to any one warp waiting is to run a *different* warp that has work ready to issue, and switch back once the first warp's data arrives. This only works if there is another warp available to switch to. How many warps can be resident on one Streaming Multiprocessor (SM) at once — ready to be switched to at zero cost — is not unlimited. It is capped by how many of three finite, shared resources each block's threads consume: the SM's fixed slot budget for threads, its fixed register file, and its fixed shared-memory capacity. A data structure whose per-thread operations need more registers, or a block-local buffer in shared memory, leaves fewer of all three resources for other blocks — which means fewer warps resident — which means fewer independent memory requests available to hide exactly the latency Section 1.2 just measured.

### Background

This section computes a genuine occupancy ceiling from real, publicly documented hardware limits — NVIDIA's own compute capability 8.0 (Ampere, e.g. A100) technical specifications: 2048 threads and 64 resident warps maximum per SM, a 65536-entry 32-bit register file per SM, and up to 164 KiB of shared memory per SM (opt-in maximum). Three independent divisions — blocks limited by thread slots, blocks limited by the register file, blocks limited by shared-memory capacity — and the smallest of the three (capped again by the hardware's own 32-block-per-SM limit) is the actual number of blocks the SM can host at once. This is the same idealized methodology NVIDIA's own published occupancy-calculator uses, and it is stated here as an idealized *ceiling*, not a cycle-exact hardware count — real allocation happens in fixed-size rounding units, so true achieved occupancy can sit at or below this number, never above it.

The file below computes this ceiling for three kernels doing identical *amounts* of per-thread work but with different per-thread *footprints*: a flat array scan with a light register count, a tree traversal keeping an explicit per-thread stack of node indices in registers (Part 4 builds exactly this kind of kernel), and a hash-table kernel staging a block-local buffer in shared memory (Part 5 builds exactly this kind of kernel).

```cpp
#include <cstdio>
#include <algorithm>

// Chapter 1.3 -- Section 1.2 showed that a resident warp waiting on a
// pointer-chase has nothing else to do while that load is in flight. The
// GPU hides that latency the same way it always does: by having OTHER
// resident warps issue THEIR memory requests during the wait, so the
// memory system always has many independent requests outstanding instead
// of one. How many warps can be resident on one Streaming Multiprocessor
// (SM) at once is not unlimited -- it is capped by three genuinely
// competing hardware resources, and a data structure's own per-thread
// footprint (registers for a traversal's local state, shared memory for
// a block-local buffer) determines how many of those warps actually fit.
// This section computes that cap directly from real, documented hardware
// limits, not from an assumed or hoped-for number.

// ---- Real, documented hardware limits: NVIDIA compute capability 8.0
// ---- (Ampere, e.g. A100), from NVIDIA's own CUDA C++ Programming Guide
// ---- technical specifications table. These are the SAME kind of public,
// ---- checkable numbers as the 128-byte transaction size Section 1.2
// ---- already used -- not fabricated, and not this book's own estimate.
static const int MAX_THREADS_PER_SM   = 2048;
static const int MAX_WARPS_PER_SM     = MAX_THREADS_PER_SM / 32;   // 64
static const int MAX_BLOCKS_PER_SM    = 32;
static const int REGISTER_FILE_SIZE   = 65536;          // 32-bit registers per SM
static const int MAX_REGISTERS_PER_THREAD = 255;
static const long long MAX_SHARED_MEM_PER_SM_BYTES = 164LL * 1024;  // opt-in max, cc 8.0

// ---- A genuine occupancy calculator, built from three independent
// ---- resource limits, each computed directly rather than looked up. ----
//
// **Model boundary, stated plainly:** real hardware allocates registers
// and shared memory in fixed-size rounding units (registers in per-warp
// allocation granules, shared memory with alignment padding), so the
// TRUE achievable block count can be slightly lower than what a pure
// division computes here. What this model gives is the same idealized
// upper bound NVIDIA's own published occupancy calculator methodology
// uses -- a real, checkable ceiling, honestly labeled as a ceiling
// rather than presented as a cycle-exact hardware count.

int blocks_limited_by_threads(int threads_per_block) {
    return MAX_THREADS_PER_SM / threads_per_block;
}

int blocks_limited_by_registers(int threads_per_block, int registers_per_thread) {
    long long registers_per_block = (long long)threads_per_block * registers_per_thread;
    return (int)(REGISTER_FILE_SIZE / registers_per_block);
}

int blocks_limited_by_shared_mem(long long shared_mem_per_block_bytes) {
    if (shared_mem_per_block_bytes == 0) return MAX_BLOCKS_PER_SM;   // no limit imposed
    return (int)(MAX_SHARED_MEM_PER_SM_BYTES / shared_mem_per_block_bytes);
}

struct OccupancyResult {
    int blocks_by_threads;
    int blocks_by_registers;
    int blocks_by_shared_mem;
    int actual_blocks_per_sm;
    int resident_threads;
    int resident_warps;
    double occupancy_percent;
};

OccupancyResult compute_occupancy(int threads_per_block, int registers_per_thread,
                                   long long shared_mem_per_block_bytes) {
    OccupancyResult r;
    r.blocks_by_threads = blocks_limited_by_threads(threads_per_block);
    r.blocks_by_registers = blocks_limited_by_registers(threads_per_block, registers_per_thread);
    r.blocks_by_shared_mem = blocks_limited_by_shared_mem(shared_mem_per_block_bytes);
    r.actual_blocks_per_sm = std::min(MAX_BLOCKS_PER_SM,
                                std::min(r.blocks_by_threads,
                                std::min(r.blocks_by_registers, r.blocks_by_shared_mem)));
    r.resident_threads = r.actual_blocks_per_sm * threads_per_block;
    r.resident_warps = r.resident_threads / 32;
    r.occupancy_percent = 100.0 * r.resident_threads / MAX_THREADS_PER_SM;
    return r;
}

void print_config(const char* name, int threads_per_block, int registers_per_thread,
                   long long shared_mem_per_block_bytes) {
    OccupancyResult r = compute_occupancy(threads_per_block, registers_per_thread,
                                           shared_mem_per_block_bytes);
    printf("%s\n", name);
    printf("  threads/block=%d  registers/thread=%d  shared-mem/block=%lld bytes\n",
           threads_per_block, registers_per_thread, shared_mem_per_block_bytes);
    printf("  blocks/SM limited by: threads=%d  registers=%d  shared-mem=%d  hw-max=%d\n",
           r.blocks_by_threads, r.blocks_by_registers, r.blocks_by_shared_mem, MAX_BLOCKS_PER_SM);
    printf("  -> actual blocks/SM = %d (the smallest of the four)\n", r.actual_blocks_per_sm);
    printf("  -> resident threads = %d / %d max (%.1f%% occupancy)\n",
           r.resident_threads, MAX_THREADS_PER_SM, r.occupancy_percent);
    printf("  -> resident warps = %d / %d max\n\n", r.resident_warps, MAX_WARPS_PER_SM);
}

int main() {
    printf("=== Section 1.3: occupancy, computed from real cc 8.0 (A100) limits ===\n\n");
    printf("hardware limits used below (NVIDIA CUDA C++ Programming Guide, cc 8.0):\n");
    printf("  max threads/SM=%d  max warps/SM=%d  max blocks/SM=%d\n",
           MAX_THREADS_PER_SM, MAX_WARPS_PER_SM, MAX_BLOCKS_PER_SM);
    printf("  register file/SM=%d (32-bit registers)  max shared mem/SM=%lld bytes\n\n",
           REGISTER_FILE_SIZE, MAX_SHARED_MEM_PER_SM_BYTES);

    // Config A: a flat array scan -- the kind of kernel Section 1.2's
    // array traversal represents. Its per-thread state is a handful of
    // loop indices and accumulators: light on registers, no shared memory.
    print_config("Config A -- flat array scan (light per-thread state):",
                 256, 32, 0);

    // Config B: a tree traversal that keeps an explicit per-thread stack
    // of node indices in registers instead of recursing (Part 4 builds
    // exactly this kind of kernel). The stack itself costs registers --
    // modeled here as doubling Config A's register footprint.
    print_config("Config B -- tree traversal with a 32-deep per-thread register stack:",
                 256, 64, 0);

    // Config C: a hash-table kernel that stages a block-local buffer in
    // shared memory (Part 5 builds exactly this kind of kernel) -- 48 KiB
    // per block, a common real size for a block-local staging buffer.
    print_config("Config C -- hash-table kernel with a 48 KiB shared-memory buffer/block:",
                 256, 32, 48 * 1024);

    OccupancyResult a = compute_occupancy(256, 32, 0);
    OccupancyResult b = compute_occupancy(256, 64, 0);
    OccupancyResult c = compute_occupancy(256, 32, 48 * 1024);

    printf("resident warps, side by side: A=%d  B=%d  C=%d\n\n", a.resident_warps, b.resident_warps, c.resident_warps);
    printf("Every one of these three kernels is CORRECT and does the identical amount of\n");
    printf("useful work per thread. What differs is how many OTHER warps the SM can keep\n");
    printf("resident while any one warp is stalled on a pointer-chase load -- Section 1.2's\n");
    printf("exact problem. Config A has %d resident warps available to issue independent\n", a.resident_warps);
    printf("memory requests while others wait; Config C, doing nothing algorithmically\n");
    printf("different except keeping a bigger block-local buffer, has only %d. A data\n", c.resident_warps);
    printf("structure's per-thread register and shared-memory footprint is therefore not\n");
    printf("a minor implementation detail -- it directly sets how much of Section 1.2's\n");
    printf("latency the SM has any chance of hiding at all.\n\n");

    printf("**Model boundary, stated plainly:** this computes an idealized ceiling from\n");
    printf("the same division NVIDIA's own published occupancy-calculator methodology\n");
    printf("uses. Real hardware rounds register and shared-memory allocations up to\n");
    printf("fixed-size units, so an actual achieved occupancy can be at or below this\n");
    printf("number -- never above it.\n");

    bool ok = (a.resident_warps == 64) && (b.resident_warps == 32) && (c.resident_warps == 24)
              && (a.occupancy_percent == 100.0);
    printf("\nself-check: A=64 warps, B=32 warps, C=24 warps, A=100%% occupancy: %s\n",
           ok ? "confirmed" : "MISMATCH");
    return ok ? 0 : 1;
}
```

Running this program produces:

```text
=== Section 1.3: occupancy, computed from real cc 8.0 (A100) limits ===

hardware limits used below (NVIDIA CUDA C++ Programming Guide, cc 8.0):
  max threads/SM=2048  max warps/SM=64  max blocks/SM=32
  register file/SM=65536 (32-bit registers)  max shared mem/SM=167936 bytes

Config A -- flat array scan (light per-thread state):
  threads/block=256  registers/thread=32  shared-mem/block=0 bytes
  blocks/SM limited by: threads=8  registers=8  shared-mem=32  hw-max=32
  -> actual blocks/SM = 8 (the smallest of the four)
  -> resident threads = 2048 / 2048 max (100.0% occupancy)
  -> resident warps = 64 / 64 max

Config B -- tree traversal with a 32-deep per-thread register stack:
  threads/block=256  registers/thread=64  shared-mem/block=0 bytes
  blocks/SM limited by: threads=8  registers=4  shared-mem=32  hw-max=32
  -> actual blocks/SM = 4 (the smallest of the four)
  -> resident threads = 1024 / 2048 max (50.0% occupancy)
  -> resident warps = 32 / 64 max

Config C -- hash-table kernel with a 48 KiB shared-memory buffer/block:
  threads/block=256  registers/thread=32  shared-mem/block=49152 bytes
  blocks/SM limited by: threads=8  registers=8  shared-mem=3  hw-max=32
  -> actual blocks/SM = 3 (the smallest of the four)
  -> resident threads = 768 / 2048 max (37.5% occupancy)
  -> resident warps = 24 / 64 max

resident warps, side by side: A=64  B=32  C=24

Every one of these three kernels is CORRECT and does the identical amount of
useful work per thread. What differs is how many OTHER warps the SM can keep
resident while any one warp is stalled on a pointer-chase load -- Section 1.2's
exact problem. Config A has 64 resident warps available to issue independent
memory requests while others wait; Config C, doing nothing algorithmically
different except keeping a bigger block-local buffer, has only 24. A data
structure's per-thread register and shared-memory footprint is therefore not
a minor implementation detail -- it directly sets how much of Section 1.2's
latency the SM has any chance of hiding at all.

**Model boundary, stated plainly:** this computes an idealized ceiling from
the same division NVIDIA's own published occupancy-calculator methodology
uses. Real hardware rounds register and shared-memory allocations up to
fixed-size units, so an actual achieved occupancy can be at or below this
number -- never above it.

self-check: A=64 warps, B=32 warps, C=24 warps, A=100% occupancy: confirmed
```

All three kernels are equally correct. Config A's light footprint reaches 100% occupancy — 64 resident warps able to issue memory requests while any one of them waits on a load. Config C's shared-memory buffer, despite being algorithmically nothing more than a bigger scratch area, more than halves that to 24 resident warps: fewer independent requests in flight to hide exactly the kind of stall Section 1.2 measured.

!!! warning "[COMMON TRAP] Treating occupancy as something to always maximize"
    It is tempting to read this section as "always minimize registers and shared memory per thread," but that overcorrects. Shared memory exists precisely because it is dramatically faster than a round trip through DRAM — a kernel that refuses to use any shared memory at all to protect its occupancy number can end up *slower* than one that spends some occupancy to avoid off-chip traffic altogether. Occupancy is one lever this book uses to explain *why* a data structure's implementation looks the way it does, not a number to maximize in isolation from what that structure actually needs to do its job. Later chapters that introduce a shared-memory buffer (tiled traversals, block-local hash staging) do so having made this trade-off deliberately, and say so explicitly, rather than treating high occupancy as a goal that overrides everything else.

## Chapter Summary

A CUDA warp issues one instruction to all 32 of its lanes at once; when lanes disagree about which branch to take, the warp pays for every distinct path taken, not just the slowest one — measured directly in Section 1.1 as a genuine 2x issue-pass cost for a fully divergent branch versus a uniform one. A pointer-chasing structure pays two separate costs: no per-thread lookahead into future addresses (a cost that exists on any machine), and no warp-level coalescing across 32 independent chains (a cost specific to how a GPU bundles memory transactions) — measured directly in Section 1.2 as a 24.5x difference in memory transactions for the identical data moved. How much of that second cost a kernel can actually hide depends on how many warps can be resident on an SM at once, which is capped by real, finite thread-slot, register-file, and shared-memory budgets — measured directly in Section 1.3 as occupancy dropping from 100% to 37.5% purely from a larger per-block shared-memory footprint, with no change in per-thread work. Every later chapter's data structure decisions — array-backed rather than pointer-backed linear structures in Part 2, uniform-control-flow tree traversals in Part 4, carefully sized shared-memory staging in Part 5 — trace back to one or more of these three facts.

## Self-Check Questions

1. A warp's 32 lanes execute a branch where 20 lanes take the `if` and 12 take the `else`. How many instruction-issue passes does this cost, under the model this chapter uses, and does the answer change if the split were 31 versus 1 instead?
2. `uniform_kernel` in Section 1.1 branches on `idx / 32` rather than `threadIdx.x`. Explain concretely why this specific choice of condition guarantees every lane within any one warp agrees, using the actual arithmetic of the condition.
3. Section 1.2's Argument 1 (lookahead) and Argument 2 (coalescing) are described as "genuinely distinct." Describe a hypothetical machine where Argument 1's cost would be eliminated but Argument 2's cost would remain exactly as large.
4. In Section 1.2's measured data, the array traversal touches exactly 1 distinct 128-byte line per step. Using the flat-offset formula `t*32 + lane`, explain why this is guaranteed regardless of which step `t` is being executed.
5. Why does the linked-list traversal in Section 1.2 use a single shared node pool for all 32 lanes' lists, rather than giving each lane its own separately allocated small list? What would change about the measurement if each lane's list were allocated separately, right after the previous lane's?
6. Using Section 1.3's three resource-limiting formulas, compute the actual blocks/SM for a kernel with 128 threads/block, 48 registers/thread, and 0 shared memory. Which resource is the binding constraint?
7. Section 1.3 states that its occupancy numbers are an idealized "ceiling." Name one specific reason a real GPU's achieved occupancy for Config B could come in below the computed 50%, even though the arithmetic is correct.
8. Suppose a fourth kernel, Config D, uses 256 threads/block, 32 registers/thread, and exactly 16384 bytes of shared memory per block. Compute its actual blocks/SM and resulting occupancy percentage using the same formulas as Config A/B/C.
9. Explain, in your own words and referencing all three sections of this chapter, why "warp divergence," "pointer chasing," and "occupancy" are not three unrelated topics but three views of the same underlying fact about how this machine executes threads.

## Where We Go Next

Chapter 2 builds the CUDA execution and memory model this book needs from here forward — grids, blocks, the memory hierarchy from registers through global memory, and the CUDA Runtime API calls this book can genuinely make and honestly report on without a physical device. Chapter 3 then gives this new cost model a proper vocabulary, extending sequential Big-O analysis with *span* — a second axis of complexity that has no equivalent in a single-threaded course, and that this chapter's divergence and occupancy arguments were already, informally, reasoning about.

## Worked Solutions

**1.** A 20/12 split still has exactly 2 distinct paths present in the warp (some lanes take `if`, some take `else`), so it costs exactly 2 issue-passes — identical to a 31/1 split, which also has exactly 2 distinct paths present. The *count* of lanes on each side never changes the number of issue-passes; only the *number of distinct paths* does. This is exactly what `distinct_paths_in_warp` in Section 1.1's code computes: it tracks two booleans (`any_a`, `any_b`), not lane counts.

**2.** `idx / 32` is precisely the warp index — CUDA lays out `threadIdx.x` values 0-31 into warp 0, 32-63 into warp 1, and so on, so every one of the 32 consecutive `idx` values inside one warp shares the same `idx / 32` (integer division floors identically for all 32). Since the branch condition `warp_id % 2 == 0` depends only on this shared value, every lane in a given warp evaluates the identical condition and takes the identical path — there is no per-lane term in the condition at all.

**3.** A hypothetical machine with a perfect per-thread hardware prefetcher that could somehow predict a linked list's next address before the current load completes (impossible in reality, since that address is genuinely unknown data — but useful as a thought experiment) would eliminate Argument 1's cost, since lookahead would no longer be zero. Argument 2's cost would remain completely unchanged, because it has nothing to do with any one thread's own prediction ability — it is about whether 32 *different* threads' addresses, once known, happen to land in the same 128-byte transaction. A perfect prefetcher for each of 32 independent, scattered chains still produces 32 scattered addresses per step.

**4.** The flat offset for lane `lane` at step `t` is `t*32 + lane`. For any fixed `t`, as `lane` ranges over 0-31, the offsets are `t*32 + 0` through `t*32 + 31` — 32 *consecutive* integers. Since `ELEMENTS_PER_LINE` is 32 (128 bytes / 4-byte elements), dividing any of these 32 consecutive offsets by 32 gives the identical quotient `t`, so all 32 addresses fall in line number `t` — exactly one distinct line — for every single value of `t`, not just the one measured.

**5.** A single shared pool lets all 32 lanes' lists be threaded through node addresses that are genuinely unrelated to which lane owns them — realistic for a list whose nodes were allocated over a program's lifetime rather than all at once for one lane. If each lane instead got its own small list allocated immediately after the previous lane's (contiguous per-lane blocks), the 32 lanes' step-`t` nodes could end up closer together in memory purely as an accident of allocation order, artificially improving the measured line count without addressing pointer chasing's actual, general cause — the shared, independently shuffled pool avoids that accidental best case.

**6.** `blocks_by_threads = 2048/128 = 16`. `blocks_by_registers = 65536/(128*48=6144) = 10` (integer division: 65536/6144 = 10.67, which floors to 10). `blocks_by_shared_mem` is unconstrained (0 bytes, treated as the 32-block hardware max). The binding constraint is the register file, giving `min(16, 10, 32, 32) = 10` blocks/SM, i.e. 1280 resident threads (62.5% occupancy) — registers, not thread slots, limit this kernel.

**7.** Real hardware allocates registers in per-warp granules of a fixed size (rather than one register at a time), so a kernel's actual per-thread register count can get rounded *up* to the next allocation unit before the true block limit is computed — meaning the real register-imposed block limit can come out lower than the raw division `REGISTER_FILE_SIZE / (threads_per_block * registers_per_thread)` computes, since that raw division assumes registers can be packed with no rounding waste at all.

**8.** `blocks_by_threads = 2048/256 = 8`. `blocks_by_registers = 65536/(256*32=8192) = 8`. `blocks_by_shared_mem = 167936/16384 = 10.25, which floors to 10`. The binding constraint is thread slots and registers, tied at 8: `min(8, 8, 10, 32) = 8` blocks/SM, i.e. 2048 resident threads — 100% occupancy, identical to Config A, since 16 KiB/block of shared memory is not yet enough to become the binding limit at this thread/register configuration.

**9.** All three sections describe consequences of the same fact: a warp is 32 lanes forced to move in lockstep, sharing both control flow and memory transactions, on a machine built from a small number of SMs each with strictly finite resources. Divergence (1.1) is what happens when lockstep lanes disagree about *which instruction* to execute. Pointer chasing's coalescing cost (1.2) is what happens when lockstep lanes disagree about *which address* to load. Occupancy (1.3) is what determines how many *separate* warps exist at all to paper over either kind of disagreement by having somewhere else useful to go. A data structure that ignores all three is not simply "unoptimized" — it is failing to account for the actual execution unit this machine schedules work in.
