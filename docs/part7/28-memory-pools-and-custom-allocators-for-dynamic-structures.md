# Chapter 28: Memory Pools and Custom Allocators for Dynamic Structures

Every lock-free structure this book has built since Chapter 9 -- stacks, queues, sorted lists, hazard-protected reclaimable nodes -- has quietly assumed a fixed-capacity array of node slots was already sitting there, ready to be indexed into. That assumption was never an accident: CUDA kernels cannot call `malloc` or `new` the way host code does and expect anything resembling good performance, and a naive per-thread heap allocation under heavy contention can serialize a kernel's memory traffic into something barely faster than running it on one thread. This chapter closes Part 7 by building the machinery that actually backs that assumption -- three different allocator designs, each trading away some generality for speed and predictability, and each one a direct, deliberate reuse of a synchronization idea this book has already established rather than a new one.

## 28.1 Fixed-Size Block Pool with a Lock-Free Free List

### Intuition

The simplest allocator that supports individual, independent frees keeps every block of a single fixed size in one array, and tracks which blocks are currently free using a STACK of their indices -- alloc() pops the stack, free() pushes back onto it. This is exactly Chapter 9's lock-free stack, with one substitution: instead of pushing and popping arbitrary payload values, the stack holds block indices, and "popping a value" means "claiming ownership of that block." The same `atomicCAS` retry loop that made Chapter 9's stack safe under concurrent push and pop makes this pool safe under concurrent alloc and free, with no new idea required.

### The Sequential (CPU) Baseline

```cpp
#include <cstdio>
#include <vector>
#include <set>

// Chapter 28.1 CPU baseline -- a fixed-size block pool backed by an
// array-based free-list STACK. free_stack[0 .. top-1] holds the indices
// of every block currently available; `top` marks one past the last
// valid entry, exactly like Chapter 9's lock-free stack, except the
// values being pushed and popped are block indices instead of arbitrary
// payloads.
//
//   alloc(): pop the top of the free list  -> decrement top, return
//            free_stack[top]
//   free(i): push block i back on          -> free_stack[top] = i,
//            then increment top
//
// Popping the most recently freed block first (LIFO reuse) is what makes
// this a STACK rather than, say, a FIFO queue of free blocks -- and it
// is exactly what will let Section 28.1's parallel kernel reuse Chapter
// 9's CAS retry loop directly, with no new synchronization idea needed.

#define POOL_SIZE 8

struct BlockPool {
    std::vector<int> free_stack;
    int top;

    explicit BlockPool(int capacity) : free_stack(capacity), top(capacity) {
        for (int i = 0; i < capacity; i++) free_stack[i] = i;
    }

    int alloc() {
        if (top == 0) return -1;   // pool exhausted
        top--;
        return free_stack[top];
    }

    void free_block(int idx) {
        free_stack[top] = idx;
        top++;
    }

    void print_state(const char* label) const {
        printf("%s: [ ", label);
        for (int i = 0; i < top; i++) printf("%d ", free_stack[i]);
        printf("] (top=%d)\n", top);
    }
};

int main() {
    printf("=== Section 28.1 CPU baseline: fixed-size block pool, sequential alloc/free ===\n\n");

    BlockPool pool(POOL_SIZE);
    pool.print_state("initial free stack");
    printf("\n");

    std::vector<int> allocated;
    for (int i = 0; i < 3; i++) {
        int b = pool.alloc();
        allocated.push_back(b);
        printf("alloc() -> block %d\n", b);
        pool.print_state("  free stack now");
    }

    printf("\nfreeing block %d (the second block allocated above)\n", allocated[1]);
    pool.free_block(allocated[1]);
    pool.print_state("  free stack now");

    printf("\nalloc() again -- should reuse the just-freed block (LIFO: most\n");
    printf("recently freed comes back first):\n");
    int reused = pool.alloc();
    printf("alloc() -> block %d\n", reused);
    pool.print_state("  free stack now");
    bool reuse_ok = (reused == allocated[1]);
    printf("reused the just-freed block: %s\n", reuse_ok ? "yes" : "NO -- BUG");

    printf("\ndraining the remaining pool completely, then attempting one more\n");
    printf("alloc() past exhaustion:\n");
    // Blocks CURRENTLY live right after the reuse: allocated[0], reused
    // (== allocated[1], the same physical block, now on its second live
    // span), and allocated[2]. Draining the rest of the pool must hand
    // out every OTHER index exactly once, since none of those were ever
    // freed and reused.
    std::vector<int> live_now = {allocated[0], reused, allocated[2]};
    std::vector<int> drain_sequence;
    int drained;
    while ((drained = pool.alloc()) != -1) {
        drain_sequence.push_back(drained);
        printf("alloc() -> block %d\n", drained);
    }
    printf("alloc() -> %d (pool exhausted, top=%d)\n", drained, pool.top);

    std::vector<int> currently_live = live_now;
    currently_live.insert(currently_live.end(), drain_sequence.begin(), drain_sequence.end());
    std::set<int> distinct(currently_live.begin(), currently_live.end());
    bool all_distinct = (distinct.size() == currently_live.size());
    bool count_matches_pool = ((int)currently_live.size() == POOL_SIZE);
    bool all_in_range = true;
    for (int b : currently_live) if (b < 0 || b >= POOL_SIZE) all_in_range = false;

    printf("\ntotal blocks simultaneously live at the moment of full exhaustion: %zu\n", currently_live.size());
    printf("every one of those live block indices is distinct (no double-allocation): %s\n",
           all_distinct ? "yes" : "NO -- BUG");
    printf("exactly POOL_SIZE (%d) blocks are live once the pool is fully exhausted: %s\n",
           POOL_SIZE, count_matches_pool ? "yes" : "NO -- BUG");
    printf("every live index stayed within [0, POOL_SIZE): %s\n",
           all_in_range ? "yes" : "NO -- BUG");

    bool ok = reuse_ok && all_distinct && count_matches_pool && all_in_range;
    printf("\nself-check: LIFO reuse correct, every live block distinct and in range,\n");
    printf("pool exhausts after exactly POOL_SIZE allocations: %s\n", ok ? "confirmed" : "MISMATCH");
    return ok ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++17 -Wall -Wextra -O2 154_pool_alloc_free_cpu_baseline.cpp -o 154_pool_alloc_free_cpu_baseline
./154_pool_alloc_free_cpu_baseline
```

**Sample input:** an 8-block pool, 3 sequential allocations, freeing the second one allocated, one more allocation (which reuses it), then draining the pool completely to exhaustion.

**Sample output:**

```text
=== Section 28.1 CPU baseline: fixed-size block pool, sequential alloc/free ===

initial free stack: [ 0 1 2 3 4 5 6 7 ] (top=8)

alloc() -> block 7
  free stack now: [ 0 1 2 3 4 5 6 ] (top=7)
alloc() -> block 6
  free stack now: [ 0 1 2 3 4 5 ] (top=6)
alloc() -> block 5
  free stack now: [ 0 1 2 3 4 ] (top=5)

freeing block 6 (the second block allocated above)
  free stack now: [ 0 1 2 3 4 6 ] (top=6)

alloc() again -- should reuse the just-freed block (LIFO: most
recently freed comes back first):
alloc() -> block 6
  free stack now: [ 0 1 2 3 4 ] (top=5)
reused the just-freed block: yes

draining the remaining pool completely, then attempting one more
alloc() past exhaustion:
alloc() -> block 4
alloc() -> block 3
alloc() -> block 2
alloc() -> block 1
alloc() -> block 0
alloc() -> -1 (pool exhausted, top=0)

total blocks simultaneously live at the moment of full exhaustion: 8
every one of those live block indices is distinct (no double-allocation): yes
exactly POOL_SIZE (8) blocks are live once the pool is fully exhausted: yes
every live index stayed within [0, POOL_SIZE): yes

self-check: LIFO reuse correct, every live block distinct and in range,
pool exhausts after exactly POOL_SIZE allocations: confirmed
```

### The Concept, In Detail

```
ASCII view: the free-list stack as an array with one shared `top` index.

  free_stack: [ 0  1  2  3  4  5  6  7 ]        top = 8  (all 8 free)
                                      ^top

  alloc() -> pop:  top-- ; return free_stack[top]
  free_stack: [ 0  1  2  3  4  5  6  7 ]        top = 7  (block 7 now owned)
                                   ^top

  free(6) -> push:  free_stack[top] = 6 ; top++
  free_stack: [ 0  1  2  3  4  6  6  7 ]        top = 6  (block 6 back on top)
                                ^top          (only the first `top` entries are meaningful)
```

Every alloc() and free() touches exactly one shared value -- `top` -- which is precisely the shape Chapter 9's lock-free stack push and pop already solved: read `top`, compute what the new value should be, and commit with `atomicCAS(&top, old, new)`, retrying on failure with a freshly re-read `top` and a freshly re-read candidate block. Two threads racing to alloc() will, at worst, force one of them to retry once with an updated snapshot -- neither can ever walk away believing it owns a block the other thread already claimed, because only one thread's CAS against any given `old_top` value can ever succeed.

[COMMON TRAP]
It is tempting to think a thread can safely read `free_stack[old_top - 1]` and treat that read as final proof of which block it will receive, since the CAS check afterward "will catch any problem anyway." The CAS only protects `top` itself -- it says nothing about whether the SPECIFIC candidate block that thread read is still the right one. Because failed CAS attempts always trigger a full retry (re-reading BOTH `top` and the candidate fresh, not just retrying the same CAS with the same stale candidate), this danger never actually surfaces here, but it is exactly why the retry loop re-reads the candidate every attempt rather than only re-reading `top`.

### Code and Verification

```cpp
#include <cstdio>
#include <vector>
#include <set>

// Chapter 28.1 main -- multiple threads concurrently call alloc() (pop)
// against the SAME shared free-list stack. This is Chapter 9's lock-free
// stack pop CAS loop, applied unchanged to an array-based stack of block
// indices instead of a linked list of arbitrary payloads:
//
//   1. read old_top = *top                      (snapshot)
//   2. if old_top == 0: pool exhausted, fail
//   3. read candidate = free_stack[old_top - 1]
//   4. atomicCAS(top, old_top, old_top - 1)
//        succeeds -> this thread owns `candidate`, done
//        fails    -> some other thread's alloc() already moved top;
//                    re-read top and retry from step 1
//
// The race this protects against: two threads could both read the same
// old_top and the same candidate block, and without the CAS, both would
// believe they own it -- a double-allocation. The CAS makes only ONE of
// them actually able to move `top` from that exact old_top; the other's
// CAS is guaranteed to fail and forces a fresh read.

#define POOL_SIZE 8

__global__ void alloc_kernel(int* free_stack, int* top, int* out_block, int* out_attempts) {
    int tid = threadIdx.x;
    int attempts = 0;
    while (true) {
        attempts++;
        int old_top = *top;
        if (old_top == 0) {
            out_block[tid] = -1;
            out_attempts[tid] = attempts;
            return;
        }
        int candidate = free_stack[old_top - 1];
        int observed = atomicCAS(top, old_top, old_top - 1);
        if (observed == old_top) {
            out_block[tid] = candidate;
            out_attempts[tid] = attempts;
            return;
        }
        // CAS failed: `top` had already moved by the time this thread's
        // CAS ran. Loop back and re-read both old_top and candidate fresh.
    }
}

// ---- Host-side replay of the identical per-thread CAS-loop logic,
// ---- driving a SPECIFIC interleaving so the race is hand-traceable:
// ---- thread 1 completes uninterrupted, then while thread 0 is mid-loop
// ---- (it has already read its snapshot but not yet run its own CAS),
// ---- thread 2 is allowed to run its full loop and land its CAS first --
// ---- forcing thread 0's CAS to fail and retry with a fresh snapshot.

struct SharedTop {
    int value;
    // Models atomicCAS(&value, expected, new_val): returns the value
    // observed at the moment of the compare (the CUDA atomicCAS return
    // convention), and updates value only if it matched `expected`.
    int cas(int expected, int new_val) {
        int old = value;
        if (old == expected) value = new_val;
        return old;
    }
};

int host_alloc(int thread_id, std::vector<int>& free_stack, SharedTop& top,
               int* out_attempts, void (*interleave)() = nullptr, bool* interleave_used = nullptr) {
    int attempts = 0;
    while (true) {
        attempts++;
        int old_top = top.value;
        if (old_top == 0) {
            *out_attempts = attempts;
            printf("  T%d attempt %d: pool exhausted (top=0)\n", thread_id, attempts);
            return -1;
        }
        int candidate = free_stack[old_top - 1];
        printf("  T%d attempt %d: reads top=%d, candidate=block %d\n", thread_id, attempts, old_top, candidate);
        if (interleave && attempts == 1 && interleave_used && !*interleave_used) {
            *interleave_used = true;
            interleave();
        }
        int observed = top.cas(old_top, old_top - 1);
        if (observed == old_top) {
            printf("  T%d attempt %d: CAS(top: %d->%d) succeeded -> owns block %d\n",
                   thread_id, attempts, old_top, old_top - 1, candidate);
            *out_attempts = attempts;
            return candidate;
        } else {
            printf("  T%d attempt %d: CAS(top: expected %d, but top is now %d) FAILED -- retrying\n",
                   thread_id, attempts, old_top, observed);
        }
    }
}

int main() {
    printf("=== Section 28.1 main: concurrent alloc() via atomicCAS retry on a shared free-list stack ===\n\n");

    std::vector<int> free_stack(POOL_SIZE);
    for (int i = 0; i < POOL_SIZE; i++) free_stack[i] = i;
    SharedTop top{POOL_SIZE};

    printf("initial free stack: [ ");
    for (int b : free_stack) printf("%d ", b);
    printf("]  top = %d\n\n", top.value);

    printf("T1 completes a full, uninterrupted alloc() first:\n");
    int attempts1;
    int r1 = host_alloc(1, free_stack, top, &attempts1);
    printf("  T1 owns block %d (in %d attempt%s)\n\n", r1, attempts1, attempts1 == 1 ? "" : "s");

    printf("T0 begins alloc(): it reads its snapshot, but before T0's own CAS\n");
    printf("executes, T2 sneaks in and completes ITS alloc() first, moving top\n");
    printf("out from under T0. T0's CAS must then fail and it retries with a\n");
    printf("freshly re-read snapshot:\n\n");

    int r2 = -1, attempts2 = 0;
    bool interleave_used = false;
    // A static pointer lets the plain C-function-pointer interleave callback
    // reach the T2 call without capturing state (kernels have no closures).
    static std::vector<int>* s_free_stack;
    static SharedTop* s_top;
    static int* s_r2;
    static int* s_attempts2;
    s_free_stack = &free_stack;
    s_top = &top;
    s_r2 = &r2;
    s_attempts2 = &attempts2;
    auto t2_interrupts = []() {
        printf("  (T2 interleaves here, completing its own alloc() before T0's CAS runs)\n");
        *s_r2 = host_alloc(2, *s_free_stack, *s_top, s_attempts2);
        printf("  T2 owns block %d (in %d attempt%s)\n", *s_r2, *s_attempts2, *s_attempts2 == 1 ? "" : "s");
    };

    int attempts0;
    int r0 = host_alloc(0, free_stack, top, &attempts0, t2_interrupts, &interleave_used);
    printf("  T0 owns block %d (in %d attempt%s)\n\n", r0, attempts0, attempts0 == 1 ? "" : "s");

    printf("final top = %d\n", top.value);
    printf("final free stack (valid entries): [ ");
    for (int i = 0; i < top.value; i++) printf("%d ", free_stack[i]);
    printf("]\n\n");

    int owned[3] = {r0, r1, r2};
    printf("blocks owned: T0=%d, T1=%d, T2=%d\n", owned[0], owned[1], owned[2]);

    std::set<int> distinct(owned, owned + 3);
    bool all_distinct = (distinct.size() == 3);
    bool all_valid = true;
    for (int b : owned) if (b < 0 || b >= POOL_SIZE) all_valid = false;
    bool t0_retried = (attempts0 == 2);

    printf("\nself-check: all 3 threads received DISTINCT valid block indices despite\n");
    printf("the forced interleave, and T0 needed exactly one retry after its first\n");
    printf("CAS was preempted by T2: %s\n",
           (all_distinct && all_valid && t0_retried) ? "confirmed" : "MISMATCH");
    return (all_distinct && all_valid && t0_retried) ? 0 : 1;
}
```

**Compile and run:**

```bash
NVDIR=/usr/local/lib/python3.11/dist-packages/nvidia/cu13
$NVDIR/bin/nvcc -arch=sm_80 -I$NVDIR/include -L$NVDIR/lib -lcudart 155_pool_alloc_parallel_kernel.cu -o 155_pool_alloc_parallel_kernel
LD_LIBRARY_PATH=$NVDIR/lib ./155_pool_alloc_parallel_kernel
```

**Sample input:** the same 8-block pool, with 3 threads concurrently calling alloc() -- one thread's snapshot is deliberately made stale by a forced interleaving, so it must retry.

**Sample output:**

```text
=== Section 28.1 main: concurrent alloc() via atomicCAS retry on a shared free-list stack ===

initial free stack: [ 0 1 2 3 4 5 6 7 ]  top = 8

T1 completes a full, uninterrupted alloc() first:
  T1 attempt 1: reads top=8, candidate=block 7
  T1 attempt 1: CAS(top: 8->7) succeeded -> owns block 7
  T1 owns block 7 (in 1 attempt)

T0 begins alloc(): it reads its snapshot, but before T0's own CAS
executes, T2 sneaks in and completes ITS alloc() first, moving top
out from under T0. T0's CAS must then fail and it retries with a
freshly re-read snapshot:

  T0 attempt 1: reads top=7, candidate=block 6
  (T2 interleaves here, completing its own alloc() before T0's CAS runs)
  T2 attempt 1: reads top=7, candidate=block 6
  T2 attempt 1: CAS(top: 7->6) succeeded -> owns block 6
  T2 owns block 6 (in 1 attempt)
  T0 attempt 1: CAS(top: expected 7, but top is now 6) FAILED -- retrying
  T0 attempt 2: reads top=6, candidate=block 5
  T0 attempt 2: CAS(top: 6->5) succeeded -> owns block 5
  T0 owns block 5 (in 2 attempts)

final top = 5
final free stack (valid entries): [ 0 1 2 3 4 ]

blocks owned: T0=5, T1=7, T2=6

self-check: all 3 threads received DISTINCT valid block indices despite
the forced interleave, and T0 needed exactly one retry after its first
CAS was preempted by T2: confirmed
```

## 28.2 Bump Allocators for Single-Pass, Reset-Once Structures

### Intuition

Section 28.1's pool pays for a genuine capability: any individual block can be freed and reused independently, at any time, in any order. Many GPU workloads never need that -- a per-frame scratch buffer, or one BFS level's temporary worklist, is built fresh every pass and thrown away as a WHOLE unit before the next pass begins. For exactly that shape of workload, a bump allocator drops the free-list entirely and keeps just one counter: alloc(n) claims the next n slots and advances the counter, full stop. There is no way to free a single allocation -- only a whole-arena reset, which is also exactly why it needs no CAS retry loop at all: a single `atomicAdd` both reserves the space and reports where it starts, in one indivisible step.

### The Sequential (CPU) Baseline

```cpp
#include <cstdio>
#include <vector>

// Chapter 28.2 CPU baseline -- a bump allocator. Where Section 28.1's
// pool tracked every individual block on a free-list stack, a bump
// allocator keeps exactly ONE counter, the "bump pointer": alloc(n)
// claims the next n slots by returning the current counter value and
// then advancing it by n. There is no per-block metadata and no way to
// free a single allocation -- the entire arena can only be reclaimed as
// one unit, by resetting the counter back to zero.
//
// This trades flexibility for simplicity and speed: exactly right for
// structures that are built fresh every pass and discarded together
// (a per-frame scratch buffer, one BFS level's temporary worklist),
// wrong for anything that needs individual blocks freed independently
// mid-lifetime (that is what Section 28.1's pool is for).

#define ARENA_SIZE 32

struct BumpArena {
    int bump = 0;

    int alloc(int n) {
        if (bump + n > ARENA_SIZE) return -1;   // arena exhausted
        int start = bump;
        bump += n;
        return start;
    }

    void reset() { bump = 0; }
};

int main() {
    printf("=== Section 28.2 CPU baseline: sequential bump allocator ===\n\n");

    BumpArena arena;
    printf("arena size = %d, bump = %d\n\n", ARENA_SIZE, arena.bump);

    int requests[] = {5, 3, 8, 2, 10, 6};
    int num_requests = 6;
    std::vector<int> starts;
    bool last_failed = false;

    for (int i = 0; i < num_requests; i++) {
        int n = requests[i];
        int start = arena.alloc(n);
        starts.push_back(start);
        if (start == -1) {
            printf("alloc(%d) -> FAILED (arena exhausted, bump=%d, would need %d > %d)\n",
                   n, arena.bump, arena.bump + n, ARENA_SIZE);
            last_failed = true;
        } else {
            printf("alloc(%d) -> [%d, %d), bump now = %d\n", n, start, start + n, arena.bump);
        }
    }

    printf("\nfinal bump = %d\n", arena.bump);

    bool ranges_ok = true;
    for (int i = 0; i + 1 < num_requests; i++) {
        if (starts[i] == -1 || starts[i + 1] == -1) continue;
        if (starts[i] + requests[i] > starts[i + 1]) ranges_ok = false;
    }

    printf("\nresetting the whole arena (bump = 0) -- no individual frees, ever:\n");
    arena.reset();
    printf("bump after reset = %d\n", arena.bump);
    int r = arena.alloc(4);
    printf("alloc(4) after reset -> [%d, %d)  (reuses the very start of the arena)\n", r, r + 4);
    bool reset_reuses_start = (r == 0);

    bool exhaustion_expected = (starts[5] == -1);   // 5+3+8+2+10 = 28, +6 = 34 > 32
    bool ok = ranges_ok && reset_reuses_start && exhaustion_expected && last_failed;

    printf("\nself-check: every successful allocation's range is disjoint from and\n");
    printf("precedes the next, the sixth request correctly overflows the arena,\n");
    printf("and a reset lets the arena be reused from offset 0 again: %s\n",
           ok ? "confirmed" : "MISMATCH");
    return ok ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++17 -Wall -Wextra -O2 156_bump_allocator_cpu_baseline.cpp -o 156_bump_allocator_cpu_baseline
./156_bump_allocator_cpu_baseline
```

**Sample input:** a 32-slot arena, five successful allocations of varying sizes, a sixth that overflows, then a full reset and one more allocation from offset 0.

**Sample output:**

```text
=== Section 28.2 CPU baseline: sequential bump allocator ===

arena size = 32, bump = 0

alloc(5) -> [0, 5), bump now = 5
alloc(3) -> [5, 8), bump now = 8
alloc(8) -> [8, 16), bump now = 16
alloc(2) -> [16, 18), bump now = 18
alloc(10) -> [18, 28), bump now = 28
alloc(6) -> FAILED (arena exhausted, bump=28, would need 34 > 32)

final bump = 28

resetting the whole arena (bump = 0) -- no individual frees, ever:
bump after reset = 0
alloc(4) after reset -> [0, 4)  (reuses the very start of the arena)

self-check: every successful allocation's range is disjoint from and
precedes the next, the sixth request correctly overflows the arena,
and a reset lets the arena be reused from offset 0 again: confirmed
```

### The Concept, In Detail

```
ASCII view: atomicAdd's return value as each thread's private starting offset.

  bump = 0

  T2 requests 6:  atomicAdd(&bump, 6) returns old=0  -> T2 owns [0, 6)   bump=6
  T0 requests 10: atomicAdd(&bump,10) returns old=6  -> T0 owns [6, 16)  bump=16
  T3 requests 4:  atomicAdd(&bump, 4) returns old=16 -> T3 owns [16,20)  bump=20
  T1 requests 8:  atomicAdd(&bump, 8) returns old=20 -> range [20,28)
                    ARENA_SIZE=24 -- OVERFLOWS, T1's alloc fails,
                    but bump is now 28 -- that space is gone until reset
```

`atomicAdd` needs no compare and no retry because it never has to make a DECISION based on a value that might already be stale -- it unconditionally adds `n` and unconditionally hands back whatever the counter held immediately before, and that returned value is, by construction, a range no other thread's `atomicAdd` could also have returned. Contrast Section 28.1's alloc(): it had to decide WHICH specific block to hand out, and that decision could be invalidated by a concurrent change, which is exactly what CAS exists to detect. A bump allocator's "decision" -- how much to advance the counter by -- never depends on anything that could go stale.

[COMMON TRAP]
It is tempting to assume that once a request's `old + n` exceeds `ARENA_SIZE`, the allocator can simply "not count" that request and leave the arena exactly as it was. `atomicAdd` has ALREADY executed and ALREADY moved the counter by the time a thread can check whether its own range overflowed -- there is no way to undo an atomic add after the fact. A bump allocator that overflows leaves that space permanently stranded (unusable by anyone) until the entire arena is reset, which is precisely why bump allocators suit only workloads that reset the whole arena on a known cadence rather than running indefinitely.

### Code and Verification

```cpp
#include <cstdio>
#include <vector>
#include <algorithm>

// Chapter 28.2 main -- concurrent bump allocation via a single
// atomicAdd, directly reusing Chapter 8's atomicAdd-based growable-array
// append pattern (there, every thread always appended exactly one
// element; here, each thread requests its own variable-size span).
//
// atomicAdd(&bump, n) atomically adds n to bump and returns the value
// bump held BEFORE the add, in one indivisible step -- no retry loop is
// needed at all (contrast Section 28.1's alloc(), which needed a CAS
// retry because it had to make a DECISION -- which candidate block --
// based on a value that might already be stale by the time it acted).
// A bump allocator never needs a candidate decision: every thread's
// returned old value is unconditionally its own reserved starting
// offset, and no two threads can ever receive overlapping ranges,
// regardless of the order their atomicAdd calls actually execute in.
//
// The COMMON TRAP: atomicAdd cannot be undone. A thread whose request
// pushes bump past ARENA_SIZE has already consumed that arena space by
// the time it notices its own range overflows -- unlike Section 28.1's
// free-list alloc(), which either fully succeeds or leaves the pool
// completely untouched. An overflowing request here strands space that
// only a full arena reset (not a per-block free) can ever reclaim.

#define ARENA_SIZE 24

__global__ void bump_alloc_kernel(int* bump, const int* request_sizes, int* out_start, int* out_ok) {
    int tid = threadIdx.x;
    int n = request_sizes[tid];
    int old = atomicAdd(bump, n);
    if (old + n > ARENA_SIZE) {
        out_start[tid] = -1;
        out_ok[tid] = 0;   // overflowed -- arena space from `old` to `old+n` is now stranded
    } else {
        out_start[tid] = old;
        out_ok[tid] = 1;
    }
}

// ---- Host-side replay of the identical atomicAdd logic, driving a
// ---- specific landing order for the atomicAdd calls so the outcome is
// ---- hand-traceable regardless of thread-index order. ----

struct SharedBump {
    int value = 0;
    int atomic_add(int n) {
        int old = value;
        value += n;
        return old;
    }
};

int main() {
    printf("=== Section 28.2 main: concurrent bump allocation via a single atomicAdd ===\n\n");

    const int NUM_THREADS = 4;
    int sizes[NUM_THREADS] = {10, 8, 6, 4};   // indexed by thread id: T0=10, T1=8, T2=6, T3=4
    // The atomicAdd calls do not have to land in thread-index order --
    // this landing order is chosen to show T1's request overflowing.
    int landing_order[NUM_THREADS] = {2, 0, 3, 1};

    SharedBump bump;
    printf("arena size = %d\n", ARENA_SIZE);
    printf("threads issue atomicAdd in this landing order: [T2, T0, T3, T1]\n\n");

    int start[NUM_THREADS], ok[NUM_THREADS];
    for (int i = 0; i < NUM_THREADS; i++) {
        int tid = landing_order[i];
        int n = sizes[tid];
        int old = bump.atomic_add(n);
        if (old + n > ARENA_SIZE) {
            start[tid] = -1;
            ok[tid] = 0;
            printf("  T%d: atomicAdd(bump, %d) returned old=%d -> range [%d, %d) OVERFLOWS "
                   "arena (size %d) -- alloc FAILS (bump now %d, that space is stranded)\n",
                   tid, n, old, old, old + n, ARENA_SIZE, bump.value);
        } else {
            start[tid] = old;
            ok[tid] = 1;
            printf("  T%d: atomicAdd(bump, %d) returned old=%d -> owns range [%d, %d)  (bump now %d)\n",
                   tid, n, old, old, old + n, bump.value);
        }
    }

    printf("\nfinal bump = %d  (arena size %d)\n", bump.value, ARENA_SIZE);

    printf("\nsuccessful threads and their ranges: ");
    std::vector<std::pair<int,int>> ranges;   // (start, end)
    for (int tid = 0; tid < NUM_THREADS; tid++) {
        if (ok[tid]) {
            printf("T%d=[%d,%d) ", tid, start[tid], start[tid] + sizes[tid]);
            ranges.push_back({start[tid], start[tid] + sizes[tid]});
        }
    }
    printf("\n");

    std::sort(ranges.begin(), ranges.end());
    bool disjoint = true;
    for (size_t i = 0; i + 1 < ranges.size(); i++) {
        if (ranges[i].second > ranges[i + 1].first) disjoint = false;
    }
    printf("all successful ranges are disjoint (no overlap): %s\n", disjoint ? "yes" : "NO -- BUG");

    bool exactly_one_overflow = (ok[0] + ok[1] + ok[2] + ok[3] == 3);
    printf("exactly one thread (T1, the largest late-landing request) overflows: %s\n",
           exactly_one_overflow ? "yes" : "NO -- BUG");

    printf("\nresetting the arena as a whole and retrying T1's overflowed request:\n");
    bump.value = 0;
    int retry_old = bump.atomic_add(sizes[1]);
    bool retry_ok = (retry_old + sizes[1] <= ARENA_SIZE);
    printf("  after reset, T1's atomicAdd(bump, %d) returns old=%d -> range [%d, %d), "
           "succeeds: %s\n", sizes[1], retry_old, retry_old, retry_old + sizes[1], retry_ok ? "yes" : "no");

    bool ok_all = disjoint && exactly_one_overflow && retry_ok;
    printf("\nself-check: all concurrent atomicAdd allocations landed in disjoint,\n");
    printf("non-overlapping ranges with no retry loop needed, the one overflowing\n");
    printf("request correctly failed without corrupting anyone else's range, and a\n");
    printf("full arena reset let it succeed afterward: %s\n", ok_all ? "confirmed" : "MISMATCH");
    return ok_all ? 0 : 1;
}
```

**Compile and run:**

```bash
NVDIR=/usr/local/lib/python3.11/dist-packages/nvidia/cu13
$NVDIR/bin/nvcc -arch=sm_80 -I$NVDIR/include -L$NVDIR/lib -lcudart 157_bump_allocator_kernel.cu -o 157_bump_allocator_kernel
LD_LIBRARY_PATH=$NVDIR/lib ./157_bump_allocator_kernel
```

**Sample input:** a 24-slot arena, 4 threads requesting different sizes in a chosen landing order, one of which overflows -- then a full reset and a successful retry of the overflowed request.

**Sample output:**

```text
=== Section 28.2 main: concurrent bump allocation via a single atomicAdd ===

arena size = 24
threads issue atomicAdd in this landing order: [T2, T0, T3, T1]

  T2: atomicAdd(bump, 6) returned old=0 -> owns range [0, 6)  (bump now 6)
  T0: atomicAdd(bump, 10) returned old=6 -> owns range [6, 16)  (bump now 16)
  T3: atomicAdd(bump, 4) returned old=16 -> owns range [16, 20)  (bump now 20)
  T1: atomicAdd(bump, 8) returned old=20 -> range [20, 28) OVERFLOWS arena (size 24) -- alloc FAILS (bump now 28, that space is stranded)

final bump = 28  (arena size 24)

successful threads and their ranges: T0=[6,16) T2=[0,6) T3=[16,20) 
all successful ranges are disjoint (no overlap): yes
exactly one thread (T1, the largest late-landing request) overflows: yes

resetting the arena as a whole and retrying T1's overflowed request:
  after reset, T1's atomicAdd(bump, 8) returns old=0 -> range [0, 8), succeeds: yes

self-check: all concurrent atomicAdd allocations landed in disjoint,
non-overlapping ranges with no retry loop needed, the one overflowing
request correctly failed without corrupting anyone else's range, and a
full arena reset let it succeed afterward: confirmed
```

## 28.3 Size-Class Allocators: Routing Different Request Sizes to Separate Pools

### Intuition

Sections 28.1 and 28.2 both assumed every request was the same, known size. Real workloads rarely are, and running every request through one pool sized for the LARGEST request wastes space on every smaller one. A size-class allocator keeps several fixed-size pools side by side -- each one built exactly like Section 28.1's pool, complete with its own free-list stack -- and routes each incoming request to the smallest class still big enough to hold it. Routing itself needs no synchronization at all, since it depends only on the requesting thread's own request size; the only contention that can ever happen is between two threads independently routed to the SAME class, and that contention is resolved by that class's own Section 28.1 CAS retry loop, untouched.

### The Sequential (CPU) Baseline

```cpp
#include <cstdio>
#include <vector>

// Chapter 28.3 CPU baseline -- a size-class allocator. Section 28.1's
// single pool works well when every request is the same size, but real
// workloads request a MIX of sizes, and running every request through
// one shared pool wastes space: a request for 5 bytes served from a
// pool sized for 128-byte blocks throws away 123 bytes every time.
//
// A size-class allocator instead keeps SEVERAL fixed-size pools -- each
// one built exactly like Section 28.1's pool, with its own free-list
// stack -- and routes each incoming request to the SMALLEST class that
// is still big enough to hold it. This bounds the wasted space per
// request to (chosen class size - requested size), and, just as
// importantly for Section 28.3's parallel version, it means requests
// routed to DIFFERENT classes never contend with each other at all.

#define NUM_CLASSES 3

struct SizeClassPool {
    int block_size;
    std::vector<int> free_stack;
    int top;

    SizeClassPool(int size, int count) : block_size(size), free_stack(count), top(count) {
        for (int i = 0; i < count; i++) free_stack[i] = i;
    }

    int alloc() {
        if (top == 0) return -1;
        top--;
        return free_stack[top];
    }

    void free_block(int idx) {
        free_stack[top] = idx;
        top++;
    }
};

// Returns the index into `classes` of the smallest class size >= n, or
// -1 if no class is large enough.
int route(const int classes[NUM_CLASSES], int n) {
    for (int i = 0; i < NUM_CLASSES; i++) {
        if (n <= classes[i]) return i;
    }
    return -1;
}

int main() {
    printf("=== Section 28.3 CPU baseline: sequential size-class allocator ===\n\n");

    int class_sizes[NUM_CLASSES] = {8, 32, 128};
    int class_counts[NUM_CLASSES] = {4, 3, 2};
    std::vector<SizeClassPool> pools;
    for (int i = 0; i < NUM_CLASSES; i++) pools.emplace_back(class_sizes[i], class_counts[i]);

    printf("size classes and starting block counts: {8: 4, 32: 3, 128: 2}\n\n");

    int requests[] = {5, 20, 100, 3, 32, 9};
    int num_requests = 6;
    std::vector<int> req_class(num_requests), req_block(num_requests);

    for (int i = 0; i < num_requests; i++) {
        int n = requests[i];
        int ci = route(class_sizes, n);
        if (ci == -1) {
            printf("request(%d) -> NO CLASS FITS -- rejected\n", n);
            req_class[i] = -1;
            req_block[i] = -1;
            continue;
        }
        int block = pools[ci].alloc();
        req_class[i] = ci;
        req_block[i] = block;
        if (block == -1) {
            printf("request(%d) -> routed to class %d, but class %d is EXHAUSTED -- alloc fails\n",
                   n, class_sizes[ci], class_sizes[ci]);
        } else {
            printf("request(%d) -> routed to class %d (smallest class >= %d), got block %d "
                   "of that class (internal waste: %d bytes)\n",
                   n, class_sizes[ci], n, block, class_sizes[ci] - n);
        }
    }

    printf("\n");
    for (int i = 0; i < NUM_CLASSES; i++) {
        printf("class %d: free stack = [ ", class_sizes[i]);
        for (int j = 0; j < pools[i].top; j++) printf("%d ", pools[i].free_stack[j]);
        printf("] (top=%d)\n", pools[i].top);
    }

    printf("\nfreeing the block from request(20) (class 32) and re-requesting size 25:\n");
    int idx20 = 1;   // requests[1] == 20
    pools[req_class[idx20]].free_block(req_block[idx20]);
    printf("class %d free stack now = [ ", class_sizes[req_class[idx20]]);
    for (int j = 0; j < pools[req_class[idx20]].top; j++) printf("%d ", pools[req_class[idx20]].free_stack[j]);
    printf("] (top=%d)\n", pools[req_class[idx20]].top);

    int ci_new = route(class_sizes, 25);
    int block_new = pools[ci_new].alloc();
    bool reused = (block_new == req_block[idx20]);
    printf("request(25) -> routed to class %d, got block %d (reused the just-freed block: %s)\n",
           class_sizes[ci_new], block_new, reused ? "yes" : "no");

    bool routing_ok = (req_class[0] == 0 && req_class[1] == 1 && req_class[2] == 2 &&
                        req_class[3] == 0 && req_class[4] == 1 && req_class[5] == 1);
    bool no_cross_class_block_reuse_confusion = (ci_new == req_class[idx20]);

    bool ok = routing_ok && reused && no_cross_class_block_reuse_confusion;
    printf("\nself-check: every request routed to the smallest fitting class, and a\n");
    printf("freed block is correctly reused within its own class: %s\n", ok ? "confirmed" : "MISMATCH");
    return ok ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++17 -Wall -Wextra -O2 158_sizeclass_allocator_cpu_baseline.cpp -o 158_sizeclass_allocator_cpu_baseline
./158_sizeclass_allocator_cpu_baseline
```

**Sample input:** three size classes (8, 32, 128), six requests of varying sizes routed to the smallest fitting class, then a freed block reused within its own class.

**Sample output:**

```text
=== Section 28.3 CPU baseline: sequential size-class allocator ===

size classes and starting block counts: {8: 4, 32: 3, 128: 2}

request(5) -> routed to class 8 (smallest class >= 5), got block 3 of that class (internal waste: 3 bytes)
request(20) -> routed to class 32 (smallest class >= 20), got block 2 of that class (internal waste: 12 bytes)
request(100) -> routed to class 128 (smallest class >= 100), got block 1 of that class (internal waste: 28 bytes)
request(3) -> routed to class 8 (smallest class >= 3), got block 2 of that class (internal waste: 5 bytes)
request(32) -> routed to class 32 (smallest class >= 32), got block 1 of that class (internal waste: 0 bytes)
request(9) -> routed to class 32 (smallest class >= 9), got block 0 of that class (internal waste: 23 bytes)

class 8: free stack = [ 0 1 ] (top=2)
class 32: free stack = [ ] (top=0)
class 128: free stack = [ 0 ] (top=1)

freeing the block from request(20) (class 32) and re-requesting size 25:
class 32 free stack now = [ 2 ] (top=1)
request(25) -> routed to class 32, got block 2 (reused the just-freed block: yes)

self-check: every request routed to the smallest fitting class, and a
freed block is correctly reused within its own class: confirmed
```

### The Concept, In Detail

```
ASCII view: routing by size, then contention only within a shared class.

  request(20) -> smallest class >= 20 is 32  -> class-32 pool
  request(9)  -> smallest class >= 9  is 32  -> class-32 pool  (SAME pool as above)
  request(5)  -> smallest class >= 5  is 8   -> class-8 pool   (different pool, zero contention)
  request(100)-> smallest class >= 100 is 128-> class-128 pool (different pool, zero contention)

  class-32 pool: T0(20) and T1(9) both land here -> race resolved by
                 THAT pool's own CAS retry loop, exactly Section 28.1

  class-8 pool and class-128 pool: each has exactly one thread --
                 no CAS contention is even possible
```

Splitting one pool into several by size class does two things at once: it bounds each request's wasted space to (chosen class size minus requested size) instead of (largest class size minus requested size), and it shrinks how many threads can ever contend with one another, since only threads landing in the SAME class ever touch the same free-list stack. Neither benefit required inventing a new synchronization primitive -- both come from applying Section 28.1's exact CAS retry loop to smaller, more numerous, independent pools instead of one large shared one.

[COMMON TRAP]
It is tempting to route by picking the LARGEST class a request fits comfortably within, reasoning that this leaves more room for future growth if the caller needs slightly more space later. Routing to any class larger than the smallest one that fits wastes MORE space than necessary for every single allocation, and this book's allocators do not support growing an allocation in place at all -- the smallest fitting class is strictly the correct choice, not a tradeoff.

### Code and Verification

```cpp
#include <cstdio>
#include <vector>

// Chapter 28.3 main -- concurrent size-class allocation. Routing itself
// needs no synchronization at all: each thread decides its own class
// purely from its own requested size, with no shared state involved.
// Once routed, a thread contends ONLY with other threads routed to that
// SAME class, using Section 28.1's exact atomicCAS retry pop on that
// class's own free-list stack. Threads routed to different classes
// never touch each other's memory at all -- splitting one pool into
// several by size class is itself what shrinks how many threads can
// ever contend with one another, on top of shrinking wasted space.

#define NUM_CLASSES 3

struct DevicePool {
    int block_size;
    int* free_stack;
    int top;   // would be a separate atomic int per class on a real device
};

__device__ int route(const int* class_sizes, int num_classes, int n) {
    for (int i = 0; i < num_classes; i++) {
        if (n <= class_sizes[i]) return i;
    }
    return -1;
}

__global__ void sizeclass_alloc_kernel(int* class_sizes, int** free_stacks, int* tops,
                                        const int* request_sizes, int* out_class, int* out_block) {
    int tid = threadIdx.x;
    int n = request_sizes[tid];
    int ci = route(class_sizes, NUM_CLASSES, n);
    out_class[tid] = ci;
    if (ci == -1) {
        out_block[tid] = -1;
        return;
    }
    // Section 28.1's CAS retry pop, applied to THIS thread's routed
    // class's own free-list stack -- other classes' `tops` are never
    // touched, so only threads sharing a class ever contend.
    while (true) {
        int old_top = tops[ci];
        if (old_top == 0) {
            out_block[tid] = -1;
            return;
        }
        int candidate = free_stacks[ci][old_top - 1];
        int observed = atomicCAS(&tops[ci], old_top, old_top - 1);
        if (observed == old_top) {
            out_block[tid] = candidate;
            return;
        }
    }
}

// ---- Host-side replay of the identical routing + per-class CAS-loop
// ---- logic, driving a specific interleaving so the one genuine race
// ---- (two threads routed to the SAME class) is hand-traceable. ----

struct SharedTop {
    int value;
    int cas(int expected, int new_val) {
        int old = value;
        if (old == expected) value = new_val;
        return old;
    }
};

struct HostPool {
    int block_size;
    std::vector<int> free_stack;
    SharedTop top;
};

int route_host(const int class_sizes[NUM_CLASSES], int n) {
    for (int i = 0; i < NUM_CLASSES; i++) if (n <= class_sizes[i]) return i;
    return -1;
}

int host_alloc(HostPool& pool, int thread_id, void (*interleave)() = nullptr, bool* used = nullptr) {
    int attempts = 0;
    while (true) {
        attempts++;
        int old_top = pool.top.value;
        if (old_top == 0) {
            printf("  T%d (class %d) attempt %d: exhausted\n", thread_id, pool.block_size, attempts);
            return -1;
        }
        int candidate = pool.free_stack[old_top - 1];
        printf("  T%d (class %d) attempt %d: reads top=%d, candidate=block %d\n",
               thread_id, pool.block_size, attempts, old_top, candidate);
        if (interleave && attempts == 1 && used && !*used) {
            *used = true;
            interleave();
        }
        int observed = pool.top.cas(old_top, old_top - 1);
        if (observed == old_top) {
            printf("  T%d (class %d) attempt %d: CAS(%d->%d) succeeded -> owns block %d\n",
                   thread_id, pool.block_size, attempts, old_top, old_top - 1, candidate);
            return candidate;
        } else {
            printf("  T%d (class %d) attempt %d: CAS(expected %d, but top is now %d) FAILED -- retrying\n",
                   thread_id, pool.block_size, attempts, old_top, observed);
        }
    }
}

int main() {
    printf("=== Section 28.3 main: concurrent size-class allocation -- route, then per-class CAS ===\n\n");

    int class_sizes[NUM_CLASSES] = {8, 32, 128};
    HostPool pools[NUM_CLASSES] = {
        {8,   {0,1,2,3},   SharedTop{4}},
        {32,  {0,1,2},     SharedTop{3}},
        {128, {0,1},       SharedTop{2}},
    };

    int requests[4] = {20, 9, 5, 100};   // T0=20, T1=9, T2=5, T3=100
    int routed[4];
    for (int t = 0; t < 4; t++) routed[t] = route_host(class_sizes, requests[t]);

    printf("requests: T0=%d, T1=%d, T2=%d, T3=%d\n", requests[0], requests[1], requests[2], requests[3]);
    printf("routed to classes: T0->%d, T1->%d, T2->%d, T3->%d\n\n",
           class_sizes[routed[0]], class_sizes[routed[1]], class_sizes[routed[2]], class_sizes[routed[3]]);

    printf("T2 (class %d) and T3 (class %d) proceed with no contention (different classes):\n",
           class_sizes[routed[2]], class_sizes[routed[3]]);
    int r2 = host_alloc(pools[routed[2]], 2);
    int r3 = host_alloc(pools[routed[3]], 3);
    printf("  T2 owns class-%d block %d, T3 owns class-%d block %d\n\n",
           class_sizes[routed[2]], r2, class_sizes[routed[3]], r3);

    printf("T0 and T1 both land on class %d and contend for the SAME free-list stack;\n", class_sizes[routed[0]]);
    printf("T1 is forced to interleave before T0's CAS runs, so T0 must retry:\n\n");

    static HostPool* s_pool32;
    static int s_r1;
    s_pool32 = &pools[routed[1]];
    auto t1_interrupts = []() {
        printf("  (T1 interleaves here, completing its own alloc() on class 32 first)\n");
        s_r1 = host_alloc(*s_pool32, 1);
    };
    bool used = false;
    int r0 = host_alloc(pools[routed[0]], 0, t1_interrupts, &used);
    int r1 = s_r1;

    printf("\n  T0 owns class-%d block %d, T1 owns class-%d block %d\n\n",
           class_sizes[routed[0]], r0, class_sizes[routed[1]], r1);

    for (int i = 0; i < NUM_CLASSES; i++) {
        printf("class %d final free stack: [ ", class_sizes[i]);
        for (int j = 0; j < pools[i].top.value; j++) printf("%d ", pools[i].free_stack[j]);
        printf("] (top=%d)\n", pools[i].top.value);
    }

    bool distinct_class32 = (r0 != r1);
    bool all_valid = (r0 >= 0 && r1 >= 0 && r2 >= 0 && r3 >= 0);
    bool routing_correct = (routed[0] == 1 && routed[1] == 1 && routed[2] == 0 && routed[3] == 2);

    bool ok = distinct_class32 && all_valid && routing_correct;
    printf("\nself-check: routing sent each request to the correct smallest-fitting\n");
    printf("class, T2/T3 (different classes) had zero contention, and T0/T1 (same\n");
    printf("class) received DISTINCT blocks despite the forced interleave: %s\n",
           ok ? "confirmed" : "MISMATCH");
    return ok ? 0 : 1;
}
```

**Compile and run:**

```bash
NVDIR=/usr/local/lib/python3.11/dist-packages/nvidia/cu13
$NVDIR/bin/nvcc -arch=sm_80 -I$NVDIR/include -L$NVDIR/lib -lcudart 159_sizeclass_allocator_kernel.cu -o 159_sizeclass_allocator_kernel
LD_LIBRARY_PATH=$NVDIR/lib ./159_sizeclass_allocator_kernel
```

**Sample input:** the same three size classes, 4 threads requesting different sizes concurrently -- two of them routed to the same class and forced to contend, two routed to different classes with zero contention.

**Sample output:**

```text
=== Section 28.3 main: concurrent size-class allocation -- route, then per-class CAS ===

requests: T0=20, T1=9, T2=5, T3=100
routed to classes: T0->32, T1->32, T2->8, T3->128

T2 (class 8) and T3 (class 128) proceed with no contention (different classes):
  T2 (class 8) attempt 1: reads top=4, candidate=block 3
  T2 (class 8) attempt 1: CAS(4->3) succeeded -> owns block 3
  T3 (class 128) attempt 1: reads top=2, candidate=block 1
  T3 (class 128) attempt 1: CAS(2->1) succeeded -> owns block 1
  T2 owns class-8 block 3, T3 owns class-128 block 1

T0 and T1 both land on class 32 and contend for the SAME free-list stack;
T1 is forced to interleave before T0's CAS runs, so T0 must retry:

  T0 (class 32) attempt 1: reads top=3, candidate=block 2
  (T1 interleaves here, completing its own alloc() on class 32 first)
  T1 (class 32) attempt 1: reads top=3, candidate=block 2
  T1 (class 32) attempt 1: CAS(3->2) succeeded -> owns block 2
  T0 (class 32) attempt 1: CAS(expected 3, but top is now 2) FAILED -- retrying
  T0 (class 32) attempt 2: reads top=2, candidate=block 1
  T0 (class 32) attempt 2: CAS(2->1) succeeded -> owns block 1

  T0 owns class-32 block 1, T1 owns class-32 block 2

class 8 final free stack: [ 0 1 2 ] (top=3)
class 32 final free stack: [ 0 ] (top=1)
class 128 final free stack: [ 0 ] (top=1)

self-check: routing sent each request to the correct smallest-fitting
class, T2/T3 (different classes) had zero contention, and T0/T1 (same
class) received DISTINCT blocks despite the forced interleave: confirmed
```

## Chapter Summary

Every allocator in this chapter reused a synchronization idea this book had already established rather than inventing a new one, because the actual work of an allocator -- deciding who owns which piece of memory -- reduces to the same handful of shapes seen throughout Parts 2 and 7. A fixed-size block pool's free-list stack is Chapter 9's lock-free stack, applied to block indices instead of arbitrary values, with alloc as pop and free as push. A bump allocator needs no retry loop at all, because `atomicAdd` already reports each thread's own private starting offset unconditionally, in exchange for giving up the ability to free any single allocation -- only a full-arena reset. A size-class allocator routes each request, using no shared state at all, to the smallest of several independent Section 28.1 pools, so that requests of different sizes never contend with each other and each request wastes no more space than the gap to its own class's size. Together these three designs cover the space real allocators occupy: general-purpose reuse, single-pass throwaway speed, and size-aware routing -- and none of it required calling `malloc` from inside a kernel.

## Self-Check Questions

1. Why does Section 28.1's fixed-size block pool need a full CAS retry loop for alloc(), while Section 28.2's bump allocator needs no retry loop at all?
2. In the block pool, what specifically does a thread's failed CAS attempt on `top` indicate, and what must it do differently on its retry compared to its first attempt?
3. Why can a bump allocator's overflowing request never be "undone" once its `atomicAdd` has executed, and what does that imply about how often such an allocator should be reset?
4. In a size-class allocator, why does the routing step itself need no synchronization at all, even though the pools it routes into do?
5. If two threads request sizes that both round up to the SAME size class, what determines whether they contend with each other, and what happens if they do?
6. Why is routing a request to the smallest class that fits strictly better than routing it to any larger class the request would also fit within?

## Where We Go Next

Every structure this book has built from Chapter 4 onward -- reductions, sorts, trees, hash tables, graphs, priority queues, lock-free structures, and now the allocators that back them -- has been examined in isolation, one data structure or algorithm at a time. Part 8 closes the book with case studies that combine several of these tools at once on problems that need more than any single structure alone provides: a GPU key-value store, spatial hashing for particle simulation, and parallel BVH construction for ray tracing.

## Worked Solutions

**1.** Section 28.1's alloc() has to make a DECISION -- which specific block index to hand out -- based on a snapshot of `top` and the free-list array that another thread's concurrent alloc() or free() could invalidate before the decision is committed; CAS exists precisely to detect and recover from that staleness. Section 28.2's bump allocator makes no such decision: `atomicAdd(&bump, n)` unconditionally advances the counter and unconditionally returns the value it held immediately before, and that returned value can never coincide with what any other thread's `atomicAdd` also returned, so there is nothing a retry could ever need to correct.

**2.** A failed CAS on `top` means some other thread's alloc() or free() already changed `top` between this thread's read and its own CAS attempt, so the block index this thread computed as its candidate may no longer be the correct (or even still-free) block to hand out. On retry, the thread must re-read BOTH `top` and the candidate block fresh from the current state, not simply retry the same CAS with the same stale candidate and an updated expected value.

**3.** `atomicAdd` is a single, indivisible hardware operation -- by the time a thread can even check whether its own `old + n` exceeds `ARENA_SIZE`, the add has already happened and the counter has already moved past that point for every other thread. There is no compensating "subtract n back" operation the allocator can safely apply after the fact without risking undoing a different, valid concurrent allocation, so an overflowing request permanently strands that arena space. This means a bump allocator is appropriate only for workloads that reset the whole arena on a known, bounded cadence (once per frame, once per pass) rather than running indefinitely without ever resetting.

**4.** Routing depends only on the SIZE of the request a given thread is making, a value that thread already has privately and that no other thread can change or interfere with -- there is no shared state involved in the routing decision itself, so no synchronization is needed to make it safely. Synchronization only becomes necessary once a request has been routed and needs to actually claim a block from a specific pool's shared free-list stack, which other threads routed to that SAME class might also be trying to claim from at the same moment.

**5.** Whether two threads contend depends entirely on whether their requests are routed to the SAME size class, which happens whenever both requests are less than or equal to that class's size and greater than the next-smaller class's size (or both exceed every smaller class). If they do land in the same class, they contend for that class's shared free-list stack using that pool's own Section 28.1 CAS retry loop: one thread's CAS succeeds, and the other observes a failed CAS and retries with a freshly re-read snapshot of that same pool.

**6.** Routing to the smallest class that fits bounds the wasted space on that allocation to (chosen class size minus requested size), which is the smallest possible waste this allocator design can offer for that request; routing to any larger class only increases that gap for no benefit, since none of this chapter's allocators support growing an allocation in place later. Because there is no in-place growth to plan ahead for, there is no scenario in which reserving extra headroom by choosing a larger class actually helps -- it is pure waste with no compensating advantage.
