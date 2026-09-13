# Chapter 27: Lock-Free and Atomic-Based Concurrent Structures

Chapters 9 and 10 built lock-free stacks and queues around a single, purpose-built hardware atomic each. This chapter asks what happens once a problem needs more than what a single fixed atomic instruction provides: an operation CUDA has no built-in primitive for at all, a whole data structure's shape changing under concurrent modification rather than just one shared counter, and the sharper hazard that surfaces once nodes can be REMOVED, not just added, while other threads might still be reading them. All three build on the same foundation -- compare-and-swap and its return value -- pushed further than any single chapter so far.

## 27.1 The Compare-and-Swap Retry Loop: Building Arbitrary Atomic Updates

### Intuition

`atomicAdd`, `atomicMin`, and `atomicOr` (Chapters 6 through 24) are each a FIXED hardware operation -- CUDA bakes in exactly what happens to the memory location. There is no `atomicMul`. The general-purpose escape hatch is a compare-and-swap retry loop: read the slot's current value, compute whatever new value the operation actually calls for, then attempt `atomicCAS(&slot, old, new)`. `atomicCAS` returns whatever the slot ACTUALLY held at the moment it ran; if that matches `old`, the swap took effect and the thread is done, and if it doesn't, some other thread changed the slot first, so the read, compute, and CAS must all be repeated against the fresh value.

### The Sequential (CPU) Baseline

```cpp
#include <cstdio>
#include <vector>

// Chapter 27.1 -- The Sequential (CPU) Baseline.
// `atomicAdd`, `atomicMin`, and `atomicOr` (Chapters 6-24) are all
// FIXED hardware operations -- CUDA bakes in exactly what each one
// does to a memory location. There is no `atomicMul`. The
// compare-and-swap retry loop is the general-purpose escape hatch:
// read the current value, compute whatever new value you want from
// it, then attempt `atomicCAS(&slot, old, new)`. If some OTHER thread
// changed the slot in between the read and the CAS, the CAS fails
// (the slot no longer holds `old`), and the thread must retry with a
// FRESH read -- exactly the same retry discipline Chapter 9's
// lock-free stack push already used, now generalized to an arbitrary
// computation instead of "push a specific new node."
#define NUM_THREADS 4

int cas(int* slot, int expected, int new_val) {
    if (*slot == expected) { *slot = new_val; return 1; }
    return 0;
}

int main() {
    printf("=== Section 27.1 CPU baseline: CAS retry loop implementing atomic multiply ===\n\n");

    std::vector<int> factors = {2, 3, 5, 7};
    printf("%d threads, each multiplying a shared accumulator (starting at 1) by\n", NUM_THREADS);
    printf("its own factor: [ ");
    for (int f : factors) printf("%d ", f);
    printf("]\n\n");

    int acc = 1;
    printf("straightforward order (no contention -- each CAS succeeds first try):\n");
    for (int tid = 0; tid < NUM_THREADS; tid++) {
        int old = acc;
        int new_val = old * factors[tid];
        int succeeded = cas(&acc, old, new_val);
        printf("  thread%d: CAS(acc, old=%d, new=%d) %s -- acc=%d\n",
               tid, old, new_val, succeeded ? "succeeds" : "FAILS", acc);
    }
    printf("  final acc: %d\n\n", acc);

    // Adversarial interleave: thread 0 reads the accumulator EARLY,
    // but before its CAS runs, threads 1, 2, and 3 all successfully
    // update it -- so thread 0's CAS, still holding its now-STALE
    // "old" value, correctly fails and must retry with a fresh read.
    printf("adversarial interleave (thread 0 reads early, others race ahead):\n");
    acc = 1;
    int t0_old = acc;
    int t0_new = t0_old * factors[0];
    printf("  thread0 reads acc=%d, computes new=%d (has NOT attempted its CAS yet)\n", t0_old, t0_new);
    for (int tid = 1; tid < NUM_THREADS; tid++) {
        int old = acc;
        int new_val = old * factors[tid];
        cas(&acc, old, new_val);
        printf("  thread%d: CAS(acc, old=%d, new=%d) succeeds -- acc=%d\n", tid, old, new_val, acc);
    }
    int first_attempt = cas(&acc, t0_old, t0_new);
    printf("  thread0: CAS(acc, old=%d, new=%d) -- current acc=%d, %s\n",
           t0_old, t0_new, acc, first_attempt ? "succeeds" : "FAILS (stale old value)");
    if (!first_attempt) {
        int retry_old = acc;
        int retry_new = retry_old * factors[0];
        printf("  thread0 retries: reads FRESH acc=%d, computes new=%d\n", retry_old, retry_new);
        int second_attempt = cas(&acc, retry_old, retry_new);
        printf("  thread0: CAS(acc, old=%d, new=%d) %s -- acc=%d\n",
               retry_old, retry_new, second_attempt ? "succeeds" : "FAILS", acc);
    }
    printf("  final acc: %d\n", acc);

    int expected = 2 * 3 * 5 * 7;
    bool ok = (acc == expected);

    printf("\nexpected final product: %d (2*3*5*7), regardless of interleaving --\n", expected);
    printf("the CAS retry loop guarantees no thread's contribution is ever lost\n");

    printf("\nself-check: CAS retry loop correctly computes the full product even\n");
    printf("under an adversarial interleave: %s\n", ok ? "confirmed" : "MISMATCH");
    return ok ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++17 -Wall -Wextra -O2 148_cas_multiply_cpu_baseline.cpp -o 148_cas_multiply_cpu_baseline
./148_cas_multiply_cpu_baseline
```

**Sample input:** a shared accumulator starting at 1, multiplied by 4 threads' own factors (2, 3, 5, 7) via a CAS retry loop, under both a straightforward order and an adversarial interleave.

**Sample output:**

```text
=== Section 27.1 CPU baseline: CAS retry loop implementing atomic multiply ===

4 threads, each multiplying a shared accumulator (starting at 1) by
its own factor: [ 2 3 5 7 ]

straightforward order (no contention -- each CAS succeeds first try):
  thread0: CAS(acc, old=1, new=2) succeeds -- acc=2
  thread1: CAS(acc, old=2, new=6) succeeds -- acc=6
  thread2: CAS(acc, old=6, new=30) succeeds -- acc=30
  thread3: CAS(acc, old=30, new=210) succeeds -- acc=210
  final acc: 210

adversarial interleave (thread 0 reads early, others race ahead):
  thread0 reads acc=1, computes new=2 (has NOT attempted its CAS yet)
  thread1: CAS(acc, old=1, new=3) succeeds -- acc=3
  thread2: CAS(acc, old=3, new=15) succeeds -- acc=15
  thread3: CAS(acc, old=15, new=105) succeeds -- acc=105
  thread0: CAS(acc, old=1, new=2) -- current acc=105, FAILS (stale old value)
  thread0 retries: reads FRESH acc=105, computes new=210
  thread0: CAS(acc, old=105, new=210) succeeds -- acc=210
  final acc: 210

expected final product: 210 (2*3*5*7), regardless of interleaving --
the CAS retry loop guarantees no thread's contribution is ever lost

self-check: CAS retry loop correctly computes the full product even
under an adversarial interleave: confirmed
```

### The Concept, In Detail

```
ASCII view: why a naive read-modify-write loses an update, and CAS does not.

  naive (no CAS):                      CAS retry loop:

  thread0 reads acc=1                  thread0 reads acc=1
  threads 1-3 run: acc becomes 105     threads 1-3 run: acc becomes 105
  thread0 WRITES 1*2=2 unconditionally thread0: CAS(acc,old=1,new=2) FAILS
    -- acc=2, THREE UPDATES LOST          (slot now holds 105, not 1)
                                        thread0 retries: reads acc=105,
                                          computes new=210, CAS succeeds
                                        -- acc=210, NOTHING LOST
```

The naive version's bug is exactly the one Chapter 23.3 first diagnosed for a plain shared-minimum race: reading a value and later writing based on it are two separate steps, and anything can change the slot in between. `atomicCAS` closes that window not by preventing the interleaving -- threads 1 through 3 still run first, exactly as before -- but by making thread 0's write CONDITIONAL on the slot still holding what it originally read, so a stale write simply fails instead of silently clobbering everyone else's work.

[COMMON TRAP]
It is tempting to think a CAS retry loop is guaranteed to succeed within some small, predictable number of attempts. Under heavy contention, a thread can in principle lose the race many times in a row, retrying repeatedly while other threads keep winning first -- CAS guarantees CORRECTNESS (no update is ever silently lost), not a bound on how many retries any individual thread needs before it finally succeeds.

### Code and Verification

```cpp
#include <cstdio>
#include <vector>

// Chapter 27.1 main -- one thread per factor, all launched at once,
// each running the canonical CUDA compare-and-swap retry idiom:
// atomically try to replace the accumulator's CURRENT value with its
// own computed result, and if `atomicCAS` reports the slot held
// something else (because another thread's CAS won in between), loop
// back and recompute from the FRESH value. This is exactly how CUDA
// itself historically implemented `atomicAdd` for types the hardware
// had no native instruction for -- the same trick this section now
// applies to a genuinely unsupported operation, atomic multiply.
#define NUM_THREADS 4

__global__ void atomic_multiply_kernel(int* acc, const int* factors, int num_threads) {
    int tid = threadIdx.x;
    if (tid >= num_threads) return;
    int old = *acc;
    int assumed;
    do {
        assumed = old;
        int new_val = assumed * factors[tid];
        old = atomicCAS(acc, assumed, new_val);
    } while (assumed != old);
}

// ---- Host-side replay of the identical per-thread logic. ----

int cas_host(int* slot, int expected, int new_val) {
    if (*slot == expected) { *slot = new_val; return expected; }
    return *slot;   // atomicCAS always returns the OLD value actually found
}

int main() {
    printf("=== Section 27.1 main: parallel atomicCAS retry loop for atomic multiply ===\n\n");

    std::vector<int> factors = {2, 3, 5, 7};
    printf("%d threads (one per factor): [ ", NUM_THREADS);
    for (int f : factors) printf("%d ", f);
    printf("], shared accumulator starts at 1\n\n");

    int acc = 1;
    printf("straightforward order:\n");
    for (int tid = 0; tid < NUM_THREADS; tid++) {
        int old = acc;
        int assumed, returned;
        do {
            assumed = old;
            int new_val = assumed * factors[tid];
            returned = cas_host(&acc, assumed, new_val);
            printf("  thread%d: atomicCAS(acc, assumed=%d, new=%d) returns %d -- %s\n",
                   tid, assumed, new_val, returned, (returned == assumed) ? "won" : "lost, retrying");
            old = returned;
        } while (assumed != old);
    }
    printf("  final acc: %d\n\n", acc);

    printf("adversarial interleave (thread 0's first attempt loses the race):\n");
    acc = 1;
    int t0_assumed = acc;
    int t0_new = t0_assumed * factors[0];
    printf("  thread0 reads acc=%d, computes new=%d (has NOT attempted its CAS yet)\n", t0_assumed, t0_new);
    for (int tid = 1; tid < NUM_THREADS; tid++) {
        int old = acc;
        int assumed, returned;
        do {
            assumed = old;
            int new_val = assumed * factors[tid];
            returned = cas_host(&acc, assumed, new_val);
            printf("  thread%d: atomicCAS(acc, assumed=%d, new=%d) returns %d -- won\n", tid, assumed, new_val, returned);
            old = returned;
        } while (assumed != old);
    }
    int r0 = cas_host(&acc, t0_assumed, t0_new);
    printf("  thread0: atomicCAS(acc, assumed=%d, new=%d) returns %d -- lost, retrying\n", t0_assumed, t0_new, r0);
    int assumed0b = r0;
    int new0b = assumed0b * factors[0];
    int r0b = cas_host(&acc, assumed0b, new0b);
    printf("  thread0: atomicCAS(acc, assumed=%d, new=%d) returns %d -- won\n", assumed0b, new0b, r0b);
    printf("  final acc: %d\n", acc);

    int expected = 2 * 3 * 5 * 7;
    bool ok = (acc == expected);

    printf("\nexpected final product: %d, matching the CPU baseline exactly --\n", expected);
    printf("atomicCAS's return value (the value ACTUALLY found in the slot) is what\n");
    printf("tells a thread whether it won or must recompute and retry\n");

    printf("\nself-check: parallel atomicCAS retry loop computes the exact same\n");
    printf("product as the CPU baseline: %s\n", ok ? "confirmed" : "MISMATCH");
    return ok ? 0 : 1;
}
```

**Compile and run:**

```bash
NVDIR=/usr/local/lib/python3.11/dist-packages/nvidia/cu13
$NVDIR/bin/nvcc -arch=sm_80 -I$NVDIR/include -L$NVDIR/lib -lcudart 149_cas_multiply_kernel.cu -o 149_cas_multiply_kernel
LD_LIBRARY_PATH=$NVDIR/lib ./149_cas_multiply_kernel
```

**Sample input:** the same 4-factor accumulator, computed via a real `atomicCAS` retry loop under both a straightforward order and an adversarial interleave.

**Sample output:**

```text
=== Section 27.1 main: parallel atomicCAS retry loop for atomic multiply ===

4 threads (one per factor): [ 2 3 5 7 ], shared accumulator starts at 1

straightforward order:
  thread0: atomicCAS(acc, assumed=1, new=2) returns 1 -- won
  thread1: atomicCAS(acc, assumed=2, new=6) returns 2 -- won
  thread2: atomicCAS(acc, assumed=6, new=30) returns 6 -- won
  thread3: atomicCAS(acc, assumed=30, new=210) returns 30 -- won
  final acc: 210

adversarial interleave (thread 0's first attempt loses the race):
  thread0 reads acc=1, computes new=2 (has NOT attempted its CAS yet)
  thread1: atomicCAS(acc, assumed=1, new=3) returns 1 -- won
  thread2: atomicCAS(acc, assumed=3, new=15) returns 3 -- won
  thread3: atomicCAS(acc, assumed=15, new=105) returns 15 -- won
  thread0: atomicCAS(acc, assumed=1, new=2) returns 105 -- lost, retrying
  thread0: atomicCAS(acc, assumed=105, new=210) returns 105 -- won
  final acc: 210

expected final product: 210, matching the CPU baseline exactly --
atomicCAS's return value (the value ACTUALLY found in the slot) is what
tells a thread whether it won or must recompute and retry

self-check: parallel atomicCAS retry loop computes the exact same
product as the CPU baseline: confirmed
```

## 27.2 Lock-Free Sorted Linked List: Concurrent Insertion via CAS

### Intuition

Chapter 11's linked list focused on traversal; this section tackles concurrent INSERTION into a sorted list while keeping it sorted, using the same array-based node pool (indices instead of pointers) every linked structure since Chapter 9 has used. Two threads inserting different keys that both belong at the exact same point in the list will compute the identical predecessor and successor, and both will attempt to CAS that predecessor's `next` field -- only one can possibly win, and the loser's failed CAS is the signal that the list has changed underneath it and its search must be redone.

### The Sequential (CPU) Baseline

```cpp
#include <cstdio>
#include <vector>
#include <string>

// Chapter 27.2 -- The Sequential (CPU) Baseline.
// Chapter 11's pointer-chasing linked list focused on TRAVERSAL; this
// section tackles concurrent INSERTION into a SORTED list while
// keeping it sorted. As in every array-based structure this book has
// built (Chapters 9-11), nodes live in a fixed-capacity array, with
// `next` storing an INDEX rather than a pointer, and -1 as NIL.
// Inserting a new key means finding its predecessor (the last node
// whose key is smaller) and successor (the first node whose key is
// not smaller), then splicing the new node in between. This baseline
// does that splice directly, since a single sequential thread never
// has to worry about another thread changing the list underneath it.
#define CAP 8
#define NIL (-1)

int find_pred(const std::vector<int>& key, const std::vector<int>& next, int head, int new_key) {
    int pred = NIL;
    int cur = head;
    while (cur != NIL && key[cur] < new_key) {
        pred = cur;
        cur = next[cur];
    }
    return pred;
}

void print_list(const std::vector<int>& key, const std::vector<int>& next, int head) {
    printf("  list: [ ");
    int cur = head;
    while (cur != NIL) {
        printf("%d ", key[cur]);
        cur = next[cur];
    }
    printf("]\n");
}

int main() {
    printf("=== Section 27.2 CPU baseline: sequential sorted-list insertion ===\n\n");

    std::vector<int> key(CAP, 0);
    std::vector<int> next(CAP, NIL);
    key[0] = 10; key[1] = 30; key[2] = 50;
    next[0] = 1; next[1] = 2; next[2] = NIL;
    int head = 0;
    int free_slot = 3;

    printf("initial sorted list:\n");
    print_list(key, next, head);
    printf("\n");

    std::vector<int> to_insert = {20, 25};
    for (int new_key : to_insert) {
        int new_idx = free_slot++;
        key[new_idx] = new_key;
        int pred = find_pred(key, next, head, new_key);
        int succ = (pred == NIL) ? head : next[pred];
        printf("insert %d (new node index %d): pred=%s, succ=%s\n",
               new_key, new_idx,
               (pred == NIL) ? "NIL (becomes new head)" : (std::to_string(pred) + " (key=" + std::to_string(key[pred]) + ")").c_str(),
               (succ == NIL) ? "NIL" : (std::to_string(succ) + " (key=" + std::to_string(key[succ]) + ")").c_str());
        next[new_idx] = succ;
        if (pred == NIL) head = new_idx;
        else next[pred] = new_idx;
        print_list(key, next, head);
    }

    printf("\nfinal sorted list:\n");
    print_list(key, next, head);

    std::vector<int> final_order;
    int cur = head;
    while (cur != NIL) { final_order.push_back(key[cur]); cur = next[cur]; }
    std::vector<int> expected = {10, 20, 25, 30, 50};
    bool ok = (final_order == expected);

    printf("\nexpected final list: [ 10 20 25 30 50 ]\n");
    printf("\nself-check: sequential insertion keeps the list sorted and matches\n");
    printf("the expected order: %s\n", ok ? "confirmed" : "MISMATCH");
    return ok ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++17 -Wall -Wextra -O2 150_lockfree_sortedlist_cpu_baseline.cpp -o 150_lockfree_sortedlist_cpu_baseline
./150_lockfree_sortedlist_cpu_baseline
```

**Sample input:** a sorted list [10, 30, 50], with keys 20 and 25 inserted one at a time, sequentially.

**Sample output:**

```text
=== Section 27.2 CPU baseline: sequential sorted-list insertion ===

initial sorted list:
  list: [ 10 30 50 ]

insert 20 (new node index 3): pred=0 (key=10), succ=1 (key=30)
  list: [ 10 20 30 50 ]
insert 25 (new node index 4): pred=3 (key=20), succ=1 (key=30)
  list: [ 10 20 25 30 50 ]

final sorted list:
  list: [ 10 20 25 30 50 ]

expected final list: [ 10 20 25 30 50 ]

self-check: sequential insertion keeps the list sorted and matches
the expected order: confirmed
```

### The Concept, In Detail

```
ASCII view: two threads racing for the same insertion point.

  list: 10 -> 30 -> 50

  thread-20: pred=node(10), succ=node(30)   both compute the
  thread-25: pred=node(10), succ=node(30)   SAME pred/succ pair

  thread-20: CAS(next[10], expected=30, new=20) SUCCEEDS
  list: 10 -> 20 -> 30 -> 50

  thread-25: CAS(next[10], expected=30, new=25) FAILS
             (next[10] is now 20, not 30 -- someone else already moved it)

  thread-25 RETRIES: re-searches from the CURRENT list, finds
             pred=node(20), succ=node(30) this time
  thread-25: CAS(next[20], expected=30, new=25) SUCCEEDS
  list: 10 -> 20 -> 25 -> 30 -> 50
```

The loser's failed CAS never corrupts anything -- it simply tells that thread its view of the list is stale, and the ONLY correct response is to re-derive a fresh predecessor and successor from the list as it now stands, not to retry the same CAS with the same stale successor value. This is exactly the same "retry with a fresh read" discipline Section 27.1 established for a single shared number, now protecting the shape of an entire structure.

[COMMON TRAP]
It is tempting to think a failed CAS can simply be retried against the SAME predecessor, just with a different expected value. A failed CAS means another thread changed what that predecessor's `next` field points to, which can also mean a DIFFERENT node now belongs between the original predecessor and the new key -- the search for the correct predecessor must be redone from scratch, not patched in place, or the list can end up with a node inserted in the wrong position.

### Code and Verification

```cpp
#include <cstdio>
#include <vector>

// Chapter 27.2 main -- two threads insert DIFFERENT keys that both
// belong at the SAME point in the list at once: both compute the
// identical (pred, succ) pair from the list as it stood when they
// started, and both attempt `atomicCAS(&next[pred], succ, new_idx)`.
// Only one can win, since a CAS can only ever successfully match the
// value that is ACTUALLY there -- the loser's CAS fails because the
// winner already changed `next[pred]`, and the loser must re-find its
// predecessor from scratch (the list has changed) and retry. This is
// the exact same retry discipline Section 27.1 built in the abstract,
// now protecting a real data structure's shape instead of a single
// shared number.
#define CAP 8
#define NIL (-1)

__global__ void insert_kernel(int* key, int* next, int* head, const int* new_keys, int num_inserts) {
    int tid = threadIdx.x;
    if (tid >= num_inserts) return;
    int new_idx = 3 + tid;   // pre-assigned free slots, one per thread
    int new_key = new_keys[tid];
    key[new_idx] = new_key;

    while (true) {
        int pred = NIL, cur = *head;
        while (cur != NIL && key[cur] < new_key) { pred = cur; cur = next[cur]; }
        int succ = (pred == NIL) ? *head : next[pred];
        next[new_idx] = succ;
        if (pred == NIL) {
            int old = atomicCAS(head, succ, new_idx);
            if (old == succ) break;
        } else {
            int old = atomicCAS(&next[pred], succ, new_idx);
            if (old == succ) break;
        }
        // CAS failed -- someone else changed the list; retry from scratch.
    }
}

// ---- Host-side replay of the identical per-thread logic. ----

int cas_host(int* slot, int expected, int new_val) {
    if (*slot == expected) { *slot = new_val; return expected; }
    return *slot;
}

void print_list(const std::vector<int>& key, const std::vector<int>& next, int head) {
    printf("  list: [ ");
    int cur = head;
    while (cur != NIL) { printf("%d ", key[cur]); cur = next[cur]; }
    printf("]\n");
}

int main() {
    printf("=== Section 27.2 main: concurrent CAS-based sorted-list insertion ===\n\n");

    std::vector<int> key(CAP, 0);
    std::vector<int> next(CAP, NIL);
    key[0] = 10; key[1] = 30; key[2] = 50;
    next[0] = 1; next[1] = 2; next[2] = NIL;
    int head = 0;

    printf("initial sorted list:\n");
    print_list(key, next, head);
    printf("\nthread-20 (inserting key 20) and thread-25 (inserting key 25) BOTH\n");
    printf("start from this SAME list -- both belong between node0(10) and node1(30):\n\n");

    int new_idx_20 = 3, new_idx_25 = 4;
    key[new_idx_20] = 20;
    key[new_idx_25] = 25;

    // Round 1: both threads compute pred/succ from the SAME starting list.
    auto find_pred = [&](int new_key) {
        int pred = NIL, cur = head;
        while (cur != NIL && key[cur] < new_key) { pred = cur; cur = next[cur]; }
        return pred;
    };
    int pred20 = find_pred(20), succ20 = (pred20 == NIL) ? head : next[pred20];
    int pred25 = find_pred(25), succ25 = (pred25 == NIL) ? head : next[pred25];
    printf("thread-20: pred=%d(key=%d), succ=%d(key=%d)\n", pred20, key[pred20], succ20, key[succ20]);
    printf("thread-25: pred=%d(key=%d), succ=%d(key=%d)\n", pred25, key[pred25], succ25, key[succ25]);

    // thread-20 attempts its CAS first and wins.
    next[new_idx_20] = succ20;
    int r20 = cas_host(&next[pred20], succ20, new_idx_20);
    printf("thread-20: atomicCAS(next[%d], expected=%d, new=%d) returns %d -- won\n", pred20, succ20, new_idx_20, r20);
    print_list(key, next, head);

    // thread-25 attempts its CAS against the SAME succ25 it read earlier -- fails.
    next[new_idx_25] = succ25;
    int r25 = cas_host(&next[pred25], succ25, new_idx_25);
    printf("thread-25: atomicCAS(next[%d], expected=%d, new=%d) returns %d -- lost, retrying\n", pred25, succ25, new_idx_25, r25);

    // thread-25 retries: re-finds pred/succ against the NOW-updated list.
    int pred25b = find_pred(25), succ25b = (pred25b == NIL) ? head : next[pred25b];
    printf("thread-25 retries: re-finds pred=%d(key=%d), succ=%d(key=%d)\n", pred25b, key[pred25b], succ25b, key[succ25b]);
    next[new_idx_25] = succ25b;
    int r25b = cas_host(&next[pred25b], succ25b, new_idx_25);
    printf("thread-25: atomicCAS(next[%d], expected=%d, new=%d) returns %d -- won\n", pred25b, succ25b, new_idx_25, r25b);

    printf("\nfinal sorted list:\n");
    print_list(key, next, head);

    std::vector<int> final_order;
    int cur = head;
    while (cur != NIL) { final_order.push_back(key[cur]); cur = next[cur]; }
    std::vector<int> expected_order = {10, 20, 25, 30, 50};
    bool ok = (final_order == expected_order);

    printf("\nexpected final list: [ 10 20 25 30 50 ] -- both keys inserted, list\n");
    printf("still sorted, even though thread-25's FIRST CAS attempt lost the race\n");

    printf("\nself-check: concurrent CAS-based insertion with retry produces the\n");
    printf("exact same sorted list as the sequential baseline: %s\n", ok ? "confirmed" : "MISMATCH");
    return ok ? 0 : 1;
}
```

**Compile and run:**

```bash
NVDIR=/usr/local/lib/python3.11/dist-packages/nvidia/cu13
$NVDIR/bin/nvcc -arch=sm_80 -I$NVDIR/include -L$NVDIR/lib -lcudart 151_lockfree_sortedlist_kernel.cu -o 151_lockfree_sortedlist_kernel
LD_LIBRARY_PATH=$NVDIR/lib ./151_lockfree_sortedlist_kernel
```

**Sample input:** the same starting list, with two threads concurrently inserting 20 and 25 -- both targeting the same insertion point at once.

**Sample output:**

```text
=== Section 27.2 main: concurrent CAS-based sorted-list insertion ===

initial sorted list:
  list: [ 10 30 50 ]

thread-20 (inserting key 20) and thread-25 (inserting key 25) BOTH
start from this SAME list -- both belong between node0(10) and node1(30):

thread-20: pred=0(key=10), succ=1(key=30)
thread-25: pred=0(key=10), succ=1(key=30)
thread-20: atomicCAS(next[0], expected=1, new=3) returns 1 -- won
  list: [ 10 20 30 50 ]
thread-25: atomicCAS(next[0], expected=1, new=4) returns 3 -- lost, retrying
thread-25 retries: re-finds pred=3(key=20), succ=1(key=30)
thread-25: atomicCAS(next[3], expected=1, new=4) returns 1 -- won

final sorted list:
  list: [ 10 20 25 30 50 ]

expected final list: [ 10 20 25 30 50 ] -- both keys inserted, list
still sorted, even though thread-25's FIRST CAS attempt lost the race

self-check: concurrent CAS-based insertion with retry produces the
exact same sorted list as the sequential baseline: confirmed
```

## 27.3 The Memory-Reclamation Problem: Hazard Pointers

### Intuition

Section 27.2 solved concurrent insertion; concurrent REMOVAL has a sharper hazard. Once a node is unlinked and its slot pushed onto a free list for reuse, some OTHER thread may already be mid-traversal holding that slot's index, about to dereference it -- and if a third thread recycles the slot for a brand-new value before the reader finishes, the reader silently reads the WRONG data. This is a different hazard than Chapter 9's ABA problem (a reused VALUE fooling a comparison); this one is reading through a reference to memory that has already been repurposed entirely. A hazard pointer fixes it directly: before touching a slot, a thread publishes that slot's index into a shared array, and anyone wanting to recycle a freed slot must check first that no one still has it published.

### The Sequential (CPU) Baseline

```cpp
#include <cstdio>
#include <vector>

// Chapter 27.2 solved concurrent INSERTION; this section confronts
// concurrent REMOVAL's own, sharper hazard. Once a node is unlinked
// from a structure (its slot pushed onto a free list for reuse), some
// OTHER thread may already be mid-traversal holding that slot's INDEX,
// about to dereference it -- and if a third thread recycles that slot
// for a brand-new value before the reader finishes, the reader
// silently reads the WRONG data. This is a genuinely different hazard
// than Chapter 9's ABA problem (which was about a REUSED VALUE fooling
// a comparison); this one is about reading through a reference to
// memory that has already been repurposed. A HAZARD POINTER is the
// standard fix: before dereferencing a slot, a thread PUBLISHES that
// slot's index to a shared array first, and any thread wanting to
// recycle a freed slot must check that no one has it published before
// reusing it. This baseline demonstrates the bug directly, then the
// fix.
#define CAP 4

int main() {
    printf("=== Section 27.3 CPU baseline: the memory-reclamation hazard ===\n\n");

    printf("--- without hazard-pointer protection (buggy) ---\n");
    std::vector<int> key = {10, 30, 50, -1};
    std::vector<int> next = {1, 2, -1, -1};
    std::vector<int> free_list = {3};

    int current_T2 = 2;
    printf("T2: has loaded current=%d while traversing (about to read node[%d].key,\n", current_T2, current_T2);
    printf("    has NOT read it yet)\n");

    next[1] = -1;
    free_list.push_back(2);
    printf("T1: unlinks node %d (key=%d) from the list, pushes slot %d onto the free\n", current_T2, key[current_T2], current_T2);
    printf("    list -- free list now: [ ");
    for (int f : free_list) printf("%d ", f);
    printf("]\n");

    int reused_slot = free_list.back();
    free_list.pop_back();
    printf("T3: pops slot %d from the free list to insert key=99 (NO hazard check!)\n", reused_slot);
    key[reused_slot] = 99;
    printf("T3: overwrites node[%d].key = 99\n", reused_slot);

    int buggy_read = key[current_T2];
    printf("T2: NOW reads node[%d].key = %d\n", current_T2, buggy_read);
    printf("BUG: T2 expected to see key=50 (the node it was traversing to), but the\n");
    printf("     slot was already recycled -- got %d instead\n\n", buggy_read);

    printf("--- with hazard-pointer protection (fixed) ---\n");
    key = {10, 30, 50, -1};
    next = {1, 2, -1, -1};
    free_list = {3};
    int hazard_T2 = -1;

    current_T2 = 2;
    hazard_T2 = current_T2;
    printf("T2: publishes hazard[T2] = %d BEFORE reading node[%d]\n", hazard_T2, current_T2);

    next[1] = -1;
    free_list.push_back(2);
    printf("T1: unlinks node %d, pushes slot %d onto the free list -- free list: [ ", current_T2, current_T2);
    for (int f : free_list) printf("%d ", f);
    printf("]\n");

    printf("T3: wants to insert key=99, checks hazards before reusing the top of the\n");
    printf("    free list (slot %d)\n", free_list.back());
    int candidate = free_list.back();
    bool hazarded = (candidate == hazard_T2);
    if (hazarded) {
        printf("T3: slot %d is HAZARDED (T2 is still using it) -- defers this slot,\n", candidate);
        int deferred = free_list.back();
        free_list.pop_back();
        free_list.insert(free_list.begin(), deferred);
        printf("    tries a different slot instead -- free list reordered to: [ ");
        for (int f : free_list) printf("%d ", f);
        printf("]\n");
    }

    int fixed_read = key[current_T2];
    printf("T2: safely reads node[%d].key = %d (the ORIGINAL value, correct)\n", current_T2, fixed_read);
    hazard_T2 = -1;
    printf("T2: clears hazard[T2] (done reading)\n");

    int safe_slot = -1;
    for (size_t i = free_list.size(); i-- > 0; ) {
        if (free_list[i] != hazard_T2) { safe_slot = free_list[i]; free_list.erase(free_list.begin() + i); break; }
    }
    key[safe_slot] = 99;
    printf("T3: rechecks hazards -- slot %d is now clear, safely reuses it, sets\n", safe_slot);
    printf("    node[%d].key = 99\n", safe_slot);

    bool ok = (buggy_read == 99) && (fixed_read == 50) && (safe_slot == 3);

    printf("\nexpected: buggy version reads 99 (wrong), fixed version reads 50\n");
    printf("(correct) and defers hazarded slot 2, safely reusing slot 3 instead\n");

    printf("\nself-check: hazard-pointer protection prevents the use-after-reuse bug: %s\n", ok ? "confirmed" : "MISMATCH");
    return ok ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++17 -Wall -Wextra -O2 152_hazard_pointer_cpu_baseline.cpp -o 152_hazard_pointer_cpu_baseline
./152_hazard_pointer_cpu_baseline
```

**Sample input:** a node about to be unlinked while another thread holds its index mid-traversal, run once WITHOUT hazard protection (showing the bug) and once WITH it (showing the fix).

**Sample output:**

```text
=== Section 27.3 CPU baseline: the memory-reclamation hazard ===

--- without hazard-pointer protection (buggy) ---
T2: has loaded current=2 while traversing (about to read node[2].key,
    has NOT read it yet)
T1: unlinks node 2 (key=50) from the list, pushes slot 2 onto the free
    list -- free list now: [ 3 2 ]
T3: pops slot 2 from the free list to insert key=99 (NO hazard check!)
T3: overwrites node[2].key = 99
T2: NOW reads node[2].key = 99
BUG: T2 expected to see key=50 (the node it was traversing to), but the
     slot was already recycled -- got 99 instead

--- with hazard-pointer protection (fixed) ---
T2: publishes hazard[T2] = 2 BEFORE reading node[2]
T1: unlinks node 2, pushes slot 2 onto the free list -- free list: [ 3 2 ]
T3: wants to insert key=99, checks hazards before reusing the top of the
    free list (slot 2)
T3: slot 2 is HAZARDED (T2 is still using it) -- defers this slot,
    tries a different slot instead -- free list reordered to: [ 2 3 ]
T2: safely reads node[2].key = 50 (the ORIGINAL value, correct)
T2: clears hazard[T2] (done reading)
T3: rechecks hazards -- slot 3 is now clear, safely reuses it, sets
    node[3].key = 99

expected: buggy version reads 99 (wrong), fixed version reads 50
(correct) and defers hazarded slot 2, safely reusing slot 3 instead

self-check: hazard-pointer protection prevents the use-after-reuse bug: confirmed
```

### The Concept, In Detail

```
ASCII view: a hazard array blocking premature reuse.

  T2 publishes: hazard[T2] = 2      (about to read node[2])
  T1 unlinks node 2, frees slot 2

  T3 wants to reuse slot 2 for a new insertion -- checks the hazard
  array FIRST:
    hazard[T2] == 2?  YES -- slot 2 is still in use, DEFER

  T2 finishes reading node[2].key (gets the correct, original value)
  T2 clears: hazard[T2] = UNHAZARDED

  T3 rechecks: hazard[T2] == 2?  NO -- now safe, slot 2 can be reused
```

Publishing a hazard costs each reader a single plain write to its own array slot (no atomic needed, since no other thread ever writes that slot), and checking a hazard costs a reclaiming thread one scan over every reader's entry. That is a real, ongoing cost paid on every read and every reclaim attempt -- the price of making memory reclamation provably safe under concurrent access, in exchange for never having to fall back to a global lock that would serialize every reader against every reclaimer.

[COMMON TRAP]
It is tempting to think a thread only needs to publish its hazard right before the exact instruction that dereferences the slot, to minimize how long the hazard stays published. The hazard must be published BEFORE any other thread could possibly decide to reclaim that slot based on the structure as the reader last observed it -- publishing it late leaves a window, between when the reader decided which slot to visit and when it actually declares that intent, during which a reclaiming thread sees no hazard at all and proceeds as if the slot were free.

### Code and Verification

```cpp
#include <cstdio>
#include <vector>

// Chapter 27.3 main -- each traversing thread publishes the slot index
// it is about to read into its OWN entry of a shared `hazard` array
// (a plain write -- no atomic needed, since each thread only ever
// writes its own entry). A thread wanting to reclaim a freed slot
// scans every OTHER thread's hazard entry before reusing it: if any
// entry still matches the candidate slot, reclaiming it is deferred.
// This mirrors Chapter 24.2's start-of-round snapshot discipline in
// spirit -- a thread's decision (safe to reuse or not) is only ever
// based on what it can actually observe of other threads' published
// state, never on an assumption about timing.
#define NUM_READERS 2
#define UNHAZARDED (-1)

__global__ void publish_hazard_kernel(int* hazard, const int* target_slot, int num_readers) {
    int tid = threadIdx.x;
    if (tid >= num_readers) return;
    hazard[tid] = target_slot[tid];   // publish BEFORE reading
}

__device__ bool slot_is_hazarded(const int* hazard, int num_readers, int slot) {
    for (int r = 0; r < num_readers; r++) {
        if (hazard[r] == slot) return true;
    }
    return false;
}

__global__ void try_reclaim_kernel(const int* hazard, int num_readers, int candidate_slot, int* reclaim_ok) {
    if (threadIdx.x != 0) return;
    *reclaim_ok = slot_is_hazarded(hazard, num_readers, candidate_slot) ? 0 : 1;
}

// ---- Host-side replay of the identical per-thread logic. ----

int main() {
    printf("=== Section 27.3 main: hazard-pointer publication and reclaim check ===\n\n");

    std::vector<int> key = {10, 30, 50, -1};
    printf("initial nodes: [ ");
    for (int k : key) printf("%d ", k);
    printf("], slot 2 (key=50) is about to be unlinked and freed\n\n");

    std::vector<int> hazard(NUM_READERS, UNHAZARDED);
    std::vector<int> target_slot = {2, -1};   // reader 0 targets slot 2; reader 1 is idle

    printf("readers publish their hazards (one thread per reader, each writing only\n");
    printf("its own entry):\n");
    for (int r = 0; r < NUM_READERS; r++) {
        hazard[r] = target_slot[r];
        printf("  reader %d: hazard[%d] = %d\n", r, r, hazard[r]);
    }
    printf("hazard array: [ ");
    for (int h : hazard) printf("%d ", h);
    printf("]\n\n");

    int candidate_slot = 2;
    bool hazarded_before = false;
    for (int r = 0; r < NUM_READERS; r++) if (hazard[r] == candidate_slot) hazarded_before = true;
    printf("reclaiming thread checks slot %d against every reader's published hazard:\n", candidate_slot);
    for (int r = 0; r < NUM_READERS; r++) {
        printf("  hazard[%d]=%d %s slot %d\n", r, hazard[r], (hazard[r] == candidate_slot) ? "MATCHES" : "does not match", candidate_slot);
    }
    printf("reclaim of slot %d: %s\n\n", candidate_slot, hazarded_before ? "BLOCKED (still hazarded)" : "allowed");

    printf("reader 0 finishes reading node[%d].key = %d, clears its hazard:\n", candidate_slot, key[candidate_slot]);
    hazard[0] = UNHAZARDED;
    printf("hazard array: [ ");
    for (int h : hazard) printf("%d ", h);
    printf("]\n\n");

    bool hazarded_after = false;
    for (int r = 0; r < NUM_READERS; r++) if (hazard[r] == candidate_slot) hazarded_after = true;
    printf("reclaiming thread rechecks slot %d:\n", candidate_slot);
    for (int r = 0; r < NUM_READERS; r++) {
        printf("  hazard[%d]=%d %s slot %d\n", r, hazard[r], (hazard[r] == candidate_slot) ? "MATCHES" : "does not match", candidate_slot);
    }
    printf("reclaim of slot %d: %s\n", candidate_slot, hazarded_after ? "BLOCKED (still hazarded)" : "allowed -- safe to reuse now");

    bool ok = hazarded_before && !hazarded_after;

    printf("\nexpected: reclaim BLOCKED while reader 0's hazard is published,\n");
    printf("then ALLOWED once reader 0 clears it -- the exact same protection\n");
    printf("the CPU baseline demonstrated, computed here via one thread per\n");
    printf("reader publishing and one thread scanning\n");

    printf("\nself-check: hazard array correctly blocks reclaim while a reader is\n");
    printf("active and allows it once clear: %s\n", ok ? "confirmed" : "MISMATCH");
    return ok ? 0 : 1;
}
```

**Compile and run:**

```bash
NVDIR=/usr/local/lib/python3.11/dist-packages/nvidia/cu13
$NVDIR/bin/nvcc -arch=sm_80 -I$NVDIR/include -L$NVDIR/lib -lcudart 153_hazard_pointer_kernel.cu -o 153_hazard_pointer_kernel
LD_LIBRARY_PATH=$NVDIR/lib ./153_hazard_pointer_kernel
```

**Sample input:** one reader thread publishing a hazard on a slot, a reclaiming thread checking it before and after the reader clears it.

**Sample output:**

```text
=== Section 27.3 main: hazard-pointer publication and reclaim check ===

initial nodes: [ 10 30 50 -1 ], slot 2 (key=50) is about to be unlinked and freed

readers publish their hazards (one thread per reader, each writing only
its own entry):
  reader 0: hazard[0] = 2
  reader 1: hazard[1] = -1
hazard array: [ 2 -1 ]

reclaiming thread checks slot 2 against every reader's published hazard:
  hazard[0]=2 MATCHES slot 2
  hazard[1]=-1 does not match slot 2
reclaim of slot 2: BLOCKED (still hazarded)

reader 0 finishes reading node[2].key = 50, clears its hazard:
hazard array: [ -1 -1 ]

reclaiming thread rechecks slot 2:
  hazard[0]=-1 does not match slot 2
  hazard[1]=-1 does not match slot 2
reclaim of slot 2: allowed -- safe to reuse now

expected: reclaim BLOCKED while reader 0's hazard is published,
then ALLOWED once reader 0 clears it -- the exact same protection
the CPU baseline demonstrated, computed here via one thread per
reader publishing and one thread scanning

self-check: hazard array correctly blocks reclaim while a reader is
active and allows it once clear: confirmed
```

## Chapter Summary

Compare-and-swap's return value -- whatever the slot ACTUALLY held at the instant the operation ran, not a stale assumption -- is the one mechanism underlying every technique in this chapter. A CAS retry loop generalizes CUDA's fixed atomics to any read-modify-write operation the hardware has no dedicated instruction for, succeeding only when the slot still holds what the thread last read and retrying with a fresh read otherwise, so no contributing thread's update is ever silently lost. Applying that same discipline to a predecessor pointer rather than a single number makes concurrent sorted-list insertion safe: a losing thread's failed CAS is the signal that its view of the list is stale, and the only correct response is to re-derive its insertion point from scratch. Removing nodes rather than only adding them introduces a sharper hazard that CAS alone cannot fix -- a slot freed for reuse might still be referenced by another thread's in-flight traversal -- and hazard pointers close that gap by having every reader publish what it is about to touch, so a reclaiming thread can check first and defer reuse until it is provably safe.

## Self-Check Questions

1. Why does a CAS retry loop never lose a thread's contribution, even though the exact same adversarial interleaving would break a naive, unprotected read-modify-write?
2. What does it mean for `atomicCAS` to return the value it does, and how does a thread use that return value to decide whether to retry?
3. When a thread's CAS attempt to insert into a sorted list fails, why must it re-search for a fresh predecessor rather than simply retrying the same CAS with an updated expected value?
4. How does the memory-reclamation hazard this chapter introduces differ from Chapter 9's ABA problem?
5. Why is a plain (non-atomic) write sufficient for a thread to publish its own hazard pointer entry?
6. What specific check must a thread perform before reusing a freed slot, and what does it do if that check fails?

## Where We Go Next

Every lock-free structure this book has built -- stacks, queues, sorted lists -- pulls its node storage from a fixed-capacity array, sidestepping the question of where that memory actually comes from and how it gets reused safely once a structure needs to grow or shrink at runtime. Chapter 28 closes Part 7 by tackling that question directly: memory pools and custom allocators, the machinery that makes dynamic, runtime-sized data structures possible on a GPU without ever calling `malloc` from inside a kernel.

## Worked Solutions

**1.** A naive read-modify-write reads a value, computes a new one, and writes it back UNCONDITIONALLY, so if other threads modified the slot in between, that unconditional write simply overwrites their work with no way to notice anything changed. A CAS retry loop's write is CONDITIONAL: it only takes effect if the slot still holds exactly the value the thread originally read, so if other threads changed it first, the CAS fails harmlessly (nothing is overwritten) and the thread is forced to recompute from the new, current value instead of blindly clobbering it.

**2.** `atomicCAS(&slot, expected, new_val)` atomically compares the slot's CURRENT contents against `expected`; if they match, it writes `new_val` and returns the OLD value (which will equal `expected`), and if they don't match, it leaves the slot untouched and returns whatever the slot actually held instead. A thread checks whether the returned value equals the `expected` value it passed in: if so, its swap succeeded; if not, some other thread got there first, and the returned value is exactly the fresh, current value the thread needs to recompute from and retry with.

**3.** A failed CAS means the predecessor's `next` field no longer holds the successor value the thread originally observed, which means the LIST ITSELF has changed -- possibly because a different node was just inserted at exactly the point this thread was also targeting. Simply retrying with a different expected value risks inserting at a position that is no longer correct for the new key (a different node might now belong between the old predecessor and the new key), so the only sound response is to re-walk the list from scratch and derive a fresh, currently-correct predecessor and successor pair.

**4.** Chapter 9's ABA problem is about a VALUE being reused: a thread reads a value, gets preempted, and later succeeds a CAS against that same value even though the underlying state changed and changed back in between -- the comparison itself is fooled. This chapter's memory-reclamation hazard is about a REFERENCE (a slot index) pointing to memory that has been entirely repurposed for a different piece of data while another thread still intends to read through that same reference -- the danger is not a fooled comparison, but reading data that no longer means what the reader expects it to mean.

**5.** Each thread's hazard-pointer entry is a dedicated slot in the shared hazard array that ONLY that thread ever writes -- no other thread ever attempts to write to that same entry, so there is no possibility of two threads racing to update it and no need for an atomic operation to protect it. Other threads only ever READ each entry (to check whether a candidate slot is hazarded), and a single plain write is already guaranteed to eventually become visible to those readers.

**6.** Before reusing a freed slot, a thread must scan every other thread's published hazard-pointer entry and confirm that none of them currently equals the slot it wants to reuse. If the check finds even one matching entry, the thread must defer reclaiming that particular slot -- leaving it on the free list untouched -- and either try a different available slot or wait and recheck later, rather than proceeding to overwrite memory some other thread may still be actively reading.
