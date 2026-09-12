# Chapter 8: Concurrent Arrays and Growable Buffers

**What you will understand by the end of this chapter:**

- How Chapter 6 and Chapter 7's atomic slot-reservation technique generalizes to the broader problem of a shared, growable output buffer — and the one guarantee (relative order) it does not give you, unlike scan-based compaction.
- Why "grow the buffer when full," the ordinary single-threaded idiom, is a genuine correctness bug under GPU concurrency, not merely a slow one — traced through the identical lockstep race Chapter 7.1 already proved.
- How to build a fixed-capacity buffer that detects and reports overflow EXACTLY, for the common case where a second full counting pass is too expensive to afford.

**What you need to know first:**

- Chapter 6's exclusive-scan-based compaction and Chapter 7's atomicAdd-based histogram and counting-sort techniques — both directly reused and contrasted against a new technique in this chapter.
- Chapter 1's warp lockstep execution model, reused again in Section 8.2 to trace exactly how a non-atomic shared counter races.

---

Chapters 6 and 7 both needed to hand elements unique output positions — compaction via a scan, counting sort via a scan-then-atomic-cursor. This chapter asks the same question from a different angle: what does it take to maintain a single, SHARED, growable buffer that many threads append to over the buffer's own lifetime, not just within one kernel's compaction step? The answer reuses everything Part 1 already built — and reveals a structural limit an array's own dynamic growth runs into on this machine, one that scan and atomics alone cannot fix.

## 8.1 Reserving Slots with atomicAdd: What You Get, and What You Don't

### Intuition

Chapter 6 computed every kept element's output slot with an exclusive scan, guaranteeing both correctness AND order. There is a second, simpler-looking way to hand out unique slots for a shared append-only buffer: skip the scan, and let every qualifying thread atomically grab the NEXT free slot directly — exactly Chapter 7.3's cursor trick, with only one bucket. It is genuinely correct. What it does not give you is Chapter 6's order guarantee, and this section proves that by direct construction rather than by assertion.

### The Sequential (CPU) Baseline

```
std::vector<int> append_cpu(const std::vector<int>& data) {
    std::vector<int> out;
    for (int v : data) {
        if (v % 3 == 0) out.push_back(v);   // one thread; no reservation needed at all
    }
    return out;
}
```

`push_back` on a CPU already IS a safe, unique slot reservation, because only one thread ever calls it — there is no possibility of two calls claiming the same slot. Section 8.1's atomicAdd-based reservation exists to give many GPU threads that exact same guarantee (a unique slot, never double-claimed) when they all want to append at once, which is a problem a single CPU thread never has.

### The Concept, In Detail

`pos = atomicAdd(&count, 1); buffer[pos] = value;` gives every qualifying thread a UNIQUE position — no two threads can ever receive the same `pos`, because `atomicAdd` is indivisible (Chapter 7.1's own guarantee). What it does NOT control is WHICH thread gets WHICH position — that depends entirely on the order in which the hardware happens to service each thread's `atomicAdd` call, and Chapter 1 already established that a real GPU never promises blocks execute in any particular order, or even concurrently.

Traced on a small, illustrative example — 2 blocks of 4 elements each, keeping only even values:

```
block 0: values [0, 1, 2, 3]   ->  kept (even): 0, 2
block 1: values [4, 5, 6, 7]   ->  kept (even): 4, 6

FORWARD schedule (block 0's atomicAdd calls happen to run first):
  block 0's kept elements claim slots 0, 1 -> output[0]=0, output[1]=2
  block 1's kept elements claim slots 2, 3 -> output[2]=4, output[3]=6
  output: [0, 2, 4, 6]

REVERSED schedule (block 1's atomicAdd calls happen to run first instead):
  block 1's kept elements claim slots 0, 1 -> output[0]=4, output[1]=6
  block 0's kept elements claim slots 2, 3 -> output[2]=0, output[3]=2
  output: [4, 6, 0, 2]

SAME 4 values kept in BOTH schedules, SAME count (4) -- but a genuinely
DIFFERENT final order, purely a function of which block's atomicAdd
calls physically ran first. Chapter 6's scan-based compaction can never
produce this: its scan computes each element's slot from that element's
OWN position in the array, completely independent of execution order.
```

The code below runs this exact comparison deterministically, at real scale (256 elements, 8 blocks of 32 threads), by explicitly simulating TWO different valid block execution orders — forward and reversed — and checking what stays the same and what does not.

### Code and Verification

```cpp
#include <cstdio>
#include <vector>
#include <algorithm>

// Chapter 8.1 -- Chapter 6 built stream compaction with an exclusive scan:
// every kept element's OWN scanned flag value already told it exactly
// where to go, with no possibility of two threads claiming the same
// output slot. There is a second, simpler-looking way to solve the same
// "keep some elements, pack them together" problem: instead of scanning
// flags first, just let every element that wants to be kept grab the
// NEXT free output slot directly, via atomicAdd on a single shared
// counter -- exactly Section 7.3's cursor trick, with only one bucket.
// It is genuinely correct. What it does NOT give you, and what this
// section proves by direct construction, is Chapter 6's guarantee that
// the output preserves the original relative order of the input.

#define BLOCK_SIZE 32

// Every thread whose element passes the predicate atomically claims the
// next free slot in the output buffer and writes there. No scan needed --
// but also no control over WHICH kept element gets WHICH slot, beyond
// "whichever thread's atomicAdd the hardware happens to service first."
__global__ void compact_via_atomic_reservation(const int* g_data, int* g_out,
                                                int* g_count, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n && g_data[idx] % 3 == 0) {
        int pos = atomicAdd(g_count, 1);
        g_out[pos] = g_data[idx];
    }
}

// ---- Host-side simulation, run under TWO different valid block
// ---- execution orders -- exactly the kind of freedom a real GPU
// ---- scheduler genuinely has (Chapter 1's own point: distinct blocks
// ---- are never guaranteed to run in any particular order, or even
// ---- concurrently). Both orders are equally "correct" schedules; this
// ---- section deliberately runs both, deterministically, specifically
// ---- to show what changes and what does not. ----

std::vector<int> simulate_atomic_compaction(const std::vector<int>& data,
                                             const std::vector<int>& block_order,
                                             int block_size) {
    std::vector<int> out;
    int n = (int)data.size();
    for (int block : block_order) {
        for (int tid = 0; tid < block_size; tid++) {
            int idx = block * block_size + tid;
            if (idx < n && data[idx] % 3 == 0) {
                out.push_back(data[idx]);   // atomicAdd: claims the next slot, in whatever order blocks actually ran
            }
        }
    }
    return out;
}

int main() {
    printf("=== Section 8.1: concurrent append via atomicAdd slot reservation ===\n\n");

    const int N = 256;
    const int NUM_BLOCKS = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    std::vector<int> data(N);
    for (int i = 0; i < N; i++) data[i] = i;

    std::vector<int> forward_order(NUM_BLOCKS);
    for (int i = 0; i < NUM_BLOCKS; i++) forward_order[i] = i;
    std::vector<int> reversed_order(NUM_BLOCKS);
    for (int i = 0; i < NUM_BLOCKS; i++) reversed_order[i] = NUM_BLOCKS - 1 - i;

    auto out_forward = simulate_atomic_compaction(data, forward_order, BLOCK_SIZE);
    auto out_reversed = simulate_atomic_compaction(data, reversed_order, BLOCK_SIZE);

    printf("N = %d elements, keeping multiples of 3, %d blocks of %d threads\n\n",
           N, NUM_BLOCKS, BLOCK_SIZE);
    printf("forward block order  (0, 1, 2, ..., %d): first 8 kept values: ", NUM_BLOCKS - 1);
    for (int i = 0; i < 8; i++) printf("%d ", out_forward[i]);
    printf("\nreversed block order (%d, ..., 2, 1, 0): first 8 kept values: ", NUM_BLOCKS - 1);
    for (int i = 0; i < 8; i++) printf("%d ", out_reversed[i]);
    printf("\n\n");

    bool same_count = (out_forward.size() == out_reversed.size());
    bool same_set = false;
    if (same_count) {
        std::vector<int> sorted_forward = out_forward;
        std::vector<int> sorted_reversed = out_reversed;
        std::sort(sorted_forward.begin(), sorted_forward.end());
        std::sort(sorted_reversed.begin(), sorted_reversed.end());
        same_set = (sorted_forward == sorted_reversed);
    }
    bool same_order = same_count && (out_forward == out_reversed);

    printf("both schedules kept the same COUNT: %s (%zu vs %zu)\n",
           same_count ? "yes" : "NO -- BUG", out_forward.size(), out_reversed.size());
    printf("both schedules kept the same SET of values (order ignored): %s\n",
           same_set ? "yes" : "NO -- BUG");
    printf("both schedules produced the same ORDER in the output array: %s\n",
           same_order ? "yes" : "no -- and that is expected, not a bug");
    printf("\natomicAdd-based reservation gives every correctness guarantee EXCEPT order:\n");
    printf("the right elements, and only the right elements, always end up kept -- but\n");
    printf("WHICH slot each one lands in depends on the order blocks actually execute in,\n");
    printf("which Chapter 1 already established a real GPU never promises. Chapter 6's\n");
    printf("scan-based compaction computes each element's slot from its OWN position, so\n");
    printf("it is schedule-independent by construction; this section's atomic reservation\n");
    printf("is not, and should only replace scan when the caller genuinely does not care\n");
    printf("about relative order.\n");

    bool ok = same_count && same_set && !same_order;
    printf("\nself-check: same count, same set, DIFFERENT order across schedules: %s\n",
           ok ? "confirmed" : "MISMATCH");
    return ok ? 0 : 1;
}
```

**Compile and run:**

```bash
nvcc -arch=sm_80 22_concurrent_append_atomic_reservation.cu -o atomic_reservation
./atomic_reservation
```

**Sample input:** `N = 256` elements (`data[i] = i`), predicate = "multiple of 3" (the same predicate as Chapter 6.1), launched as 8 blocks of 32 threads, run once under a forward block order and once under a reversed block order.

**Sample output:**

```text
=== Section 8.1: concurrent append via atomicAdd slot reservation ===

N = 256 elements, keeping multiples of 3, 8 blocks of 32 threads

forward block order  (0, 1, 2, ..., 7): first 8 kept values: 0 3 6 9 12 15 18 21 
reversed block order (7, ..., 2, 1, 0): first 8 kept values: 225 228 231 234 237 240 243 246 

both schedules kept the same COUNT: yes (86 vs 86)
both schedules kept the same SET of values (order ignored): yes
both schedules produced the same ORDER in the output array: no -- and that is expected, not a bug

atomicAdd-based reservation gives every correctness guarantee EXCEPT order:
the right elements, and only the right elements, always end up kept -- but
WHICH slot each one lands in depends on the order blocks actually execute in,
which Chapter 1 already established a real GPU never promises. Chapter 6's
scan-based compaction computes each element's slot from its OWN position, so
it is schedule-independent by construction; this section's atomic reservation
is not, and should only replace scan when the caller genuinely does not care
about relative order.

self-check: same count, same set, DIFFERENT order across schedules: confirmed
```

Both schedules keep the identical 86 elements — the same count, the same set — but the actual sequence those elements land in differs completely, confirming that atomic reservation trades away order for simplicity, while Chapter 6's scan trades away nothing but a bit of extra arithmetic.

!!! warning "[COMMON TRAP] Assuming atomicAdd-based compaction is simply a faster Chapter 6"
    It is tempting to treat this section's atomic reservation as a strictly better replacement for Chapter 6's scan-based compaction, since it needs no scan step at all. It is not strictly better — it is a genuine trade, exactly like Chapter 5.2's work-versus-span trade between Hillis-Steele and Blelloch. Atomic reservation is the right choice when the caller genuinely does not care about relative order (many real workloads do not); Chapter 6's scan is the right, and only, choice when the caller needs the output to preserve the input's original relative order, because atomic reservation structurally cannot promise that on real hardware.

## 8.2 Why "Grow When Full" Breaks Under Concurrency

### Intuition

Section 8.1's `atomicAdd`-based reservation is correct because `atomicAdd` is a single, indivisible hardware operation. It is tempting to write the same "grab a slot, then use it" idea BY HAND instead — read the current size, use it as your own slot, then write the incremented size back — reasoning that it is "basically the same thing." It is not: written by hand, that is three separate steps, and Chapter 7.1 already proved exactly what happens when a warp's lanes run three separate steps in lockstep on the same shared value.

### The Sequential (CPU) Baseline

```
std::vector<int> grow_cpu(int count) {
    std::vector<int> buffer;             // std::vector grows itself automatically, safely,
    for (int i = 0; i < count; i++) {    // because only one thread is ever calling push_back
        buffer.push_back(i);
    }
    return buffer;
}
```

`std::vector::push_back` already does exactly the "check size, grow if needed, then write" pattern this section is about to show breaking — and on a CPU, with one thread, it is completely safe, which is exactly why it is most programmers' first instinct to reach for the same idea on a GPU. This section shows precisely what goes wrong the moment many threads try to run that same, otherwise-correct-looking pattern at once.

### The Concept, In Detail

Replaying Chapter 7.1's exact race, now on a buffer's SIZE counter instead of a histogram bucket. Suppose a buffer already holds 98 of its 100-slot capacity, and a fresh wave of 32 lanes all want to append one more element each, using the naive, non-atomic pattern:

```
non-atomic "check size, use as slot, increment size", 32 racing lanes,
buffer capacity=100, size BEFORE this wave = 98

step 1 -- LOAD (every lane in the wave reads the SAME, not-yet-updated
          size, exactly as Chapter 7.1's lanes all read the identical
          pre-wave bucket value):
    all 32 lanes read: size = 98

step 2 -- COMPUTE (each lane independently, using only what it loaded):
    all 32 lanes compute: my_slot = 98, my_new_size = 99

step 3 -- STORE (every lane writes both its value and the new size;
          whichever store physically lands last for a given address
          wins -- Chapter 7.1's exact last-write-wins mechanism):
    all 32 lanes write:  buffer[98] = my_value   <- SAME slot, 31 lost
    all 32 lanes write:  size <- 99              <- SAME new size

RESULT: size goes from 98 to 99 (it SHOULD be 130 -- 98 plus 32), and
buffer[98] holds only whichever lane's store happened to land last. 31
of the 32 appended values are gone, AND the size counter itself now
under-reports the buffer's true contents, silently corrupting every
future capacity check that reads it.
```

The fix this book actually recommends is not a cleverer atomic — it is to never grow a buffer mid-kernel at all. A GPU has no single atomic operation that can "reallocate a bigger buffer and copy the old contents into it," so no amount of atomicAdd correctness on the SIZE counter alone can make growth itself safe. Instead, split the work into two passes with nothing racing in between:

```
PASS 1 (a pure count, no writes at all -- Chapter 4's reduction shape,
        with no shared, externally-visible buffer position to race over):
    exactly 86 elements need a slot

    -- a single allocation of EXACTLY 86 slots happens here, between the
       two kernel launches, on the host side. There is no concurrency
       to race over at this point, because nothing else is running.

PASS 2 (Section 8.1's atomicAdd reservation, now into a buffer that is
        ALREADY guaranteed to be exactly the right size):
    every one of the 86 slots gets filled, none left empty, none
    overflowing -- because the buffer's size was never in question by
    the time this pass starts writing anything.
```

### Code and Verification

```cpp
#include <cstdio>
#include <vector>

// Chapter 8.2 -- Section 8.1's atomicAdd-based reservation is correct
// because atomicAdd is a single indivisible operation. It is tempting to
// write the SAME "grab a slot, then use it" pattern by hand instead --
// read the current size, use it as your slot, then write the incremented
// size back -- reasoning that it is "basically the same thing." It is
// not: written by hand, that is three separate steps (load, compute,
// store), and Chapter 7.1 already proved exactly what happens when a
// warp's lanes run three separate steps in lockstep on the same shared
// value. This section reruns that exact race, now on a buffer's size
// counter instead of a histogram bucket, and then shows the pattern this
// book actually recommends: never grow a buffer mid-kernel at all.

#define WARP_SIZE 32

// The buggy, non-atomic pattern: every lane loads `size`, uses it
// directly as its own output slot, computes size+1, and stores it back --
// three separate steps, exactly Chapter 7.1's LOAD / ADD / STORE shape.
struct NaiveGrowResult {
    std::vector<int> buffer;
    int final_size;
    int distinct_slots_written;
};

NaiveGrowResult simulate_naive_grow_wave(int starting_size, int capacity, int wave_size) {
    std::vector<int> buffer(capacity, -1);   // -1 marks "never written"

    // step 1 -- LOAD (every lane in the wave reads the identical,
    // not-yet-updated `size`, exactly as Chapter 7.1's lanes all read the
    // identical pre-wave bucket value).
    int loaded_size = starting_size;

    // step 2 -- each lane independently computes ITS OWN "reserved" slot
    // and the new size it believes should result, using only what it
    // loaded -- no lane has seen any other lane's result yet.
    std::vector<int> claimed_slot(wave_size);
    std::vector<int> computed_new_size(wave_size);
    for (int lane = 0; lane < wave_size; lane++) {
        claimed_slot[lane] = loaded_size;         // every lane claims the SAME slot
        computed_new_size[lane] = loaded_size + 1;  // every lane computes the SAME new size
    }

    // step 3 -- STORE (every lane writes its value AND the new size;
    // whichever store physically lands last for a given address wins --
    // Chapter 7.1's exact last-write-wins mechanism).
    int distinct_slots = 0;
    for (int lane = 0; lane < wave_size; lane++) {
        int slot = claimed_slot[lane];
        if (slot < capacity) {
            if (buffer[slot] == -1) distinct_slots++;
            buffer[slot] = lane;   // every lane in this wave targets the SAME slot
        }
    }
    int final_size = computed_new_size[wave_size - 1];   // last lane's store wins

    return {buffer, final_size, distinct_slots};
}

// The fix this book actually recommends: never grow mid-kernel. Pass 1
// counts EXACTLY how many elements need a slot (a pure count, no writes,
// no capacity assumed yet -- Chapter 4's reduction shape). Between the
// two passes, a single allocation of EXACTLY that size happens once, with
// no concurrency at all to race over. Pass 2 then uses Section 8.1's
// already-proven atomicAdd reservation to scatter into a buffer that is
// now guaranteed to be exactly the right size.
int simulate_pass1_exact_count(const std::vector<int>& data) {
    int count = 0;
    for (int v : data) {
        if (v % 3 == 0) count++;   // predicate: same "keep multiples of 3" as Section 8.1
    }
    return count;
}

std::vector<int> simulate_pass2_scatter(const std::vector<int>& data, int exact_capacity) {
    std::vector<int> buffer(exact_capacity, -1);
    int cursor = 0;   // atomicAdd(&cursor, 1) in the real kernel
    for (int v : data) {
        if (v % 3 == 0) {
            int pos = cursor;
            cursor = cursor + 1;
            buffer[pos] = v;   // exact_capacity guarantees pos is always in range
        }
    }
    return buffer;
}

int main() {
    printf("=== Section 8.2: the naive growth race, and the two-pass fix ===\n\n");

    printf("--- part 1: replaying Chapter 7.1's exact race on a SIZE counter ---\n\n");
    const int STARTING_SIZE = 98;
    const int CAPACITY = 100;
    auto naive = simulate_naive_grow_wave(STARTING_SIZE, CAPACITY, WARP_SIZE);

    printf("buffer capacity = %d, size BEFORE this wave = %d, %d lanes all append at once\n\n",
           CAPACITY, STARTING_SIZE, WARP_SIZE);
    printf("every one of the %d lanes independently loaded size=%d, so every lane computed\n",
           WARP_SIZE, STARTING_SIZE);
    printf("the SAME target slot (%d) and the SAME new size (%d)\n\n", STARTING_SIZE, STARTING_SIZE + 1);
    printf("distinct buffer slots actually written: %d (should be %d if each lane got its own slot)\n",
           naive.distinct_slots_written, WARP_SIZE);
    printf("size after the wave: %d (should be %d if all %d appends were counted)\n\n",
           naive.final_size, STARTING_SIZE + WARP_SIZE, WARP_SIZE);
    printf("%d of the %d appended values were silently overwritten in slot %d, and the size\n",
           WARP_SIZE - naive.distinct_slots_written, WARP_SIZE, STARTING_SIZE);
    printf("counter itself now under-reports how many elements the buffer actually holds --\n");
    printf("exactly Chapter 7.1's race, now corrupting a buffer's bookkeeping instead of a\n");
    printf("histogram bucket.\n\n");

    printf("--- part 2: the two-pass fix -- count exactly, allocate once, then fill ---\n\n");
    const int N = 256;
    std::vector<int> data(N);
    for (int i = 0; i < N; i++) data[i] = i;

    int exact_count = simulate_pass1_exact_count(data);
    printf("pass 1 (a pure count, no writes, no capacity assumed): exactly %d elements need a slot\n",
           exact_count);
    auto filled = simulate_pass2_scatter(data, exact_count);
    printf("a single allocation of exactly %d slots happens here, between the two kernel\n", exact_count);
    printf("launches -- no concurrency to race over, because nothing else runs at the same time\n");
    printf("pass 2 (Section 8.1's atomicAdd reservation, now into an EXACTLY sized buffer):\n\n");

    bool every_slot_filled = true;
    for (int v : filled) if (v == -1) every_slot_filled = false;
    bool count_matches = ((int)filled.size() == exact_count);

    printf("every one of the %d allocated slots was filled, none left empty: %s\n",
           exact_count, every_slot_filled ? "yes" : "NO -- BUG");
    printf("buffer size matches the exact pass-1 count, with zero wasted capacity: %s\n",
           count_matches ? "yes" : "NO -- BUG");

    bool naive_lost_data = (naive.distinct_slots_written < WARP_SIZE);
    bool ok = naive_lost_data && every_slot_filled && count_matches;
    printf("\nself-check: naive growth demonstrably loses data, two-pass fix wastes nothing\n");
    printf("and loses nothing: %s\n", ok ? "confirmed" : "MISMATCH");
    return ok ? 0 : 1;
}
```

**Compile and run:**

```bash
nvcc -arch=sm_80 23_naive_growth_race_and_two_pass_fix.cu -o growth_race_and_fix
./growth_race_and_fix
```

**Sample input:** part 1 replays a single critical wave (32 lanes, starting size 98, capacity 100) to demonstrate the race directly; part 2 uses the same `N = 256`, "multiple of 3" setup as Section 8.1 to demonstrate the two-pass fix.

**Sample output:**

```text
=== Section 8.2: the naive growth race, and the two-pass fix ===

--- part 1: replaying Chapter 7.1's exact race on a SIZE counter ---

buffer capacity = 100, size BEFORE this wave = 98, 32 lanes all append at once

every one of the 32 lanes independently loaded size=98, so every lane computed
the SAME target slot (98) and the SAME new size (99)

distinct buffer slots actually written: 1 (should be 32 if each lane got its own slot)
size after the wave: 99 (should be 130 if all 32 appends were counted)

31 of the 32 appended values were silently overwritten in slot 98, and the size
counter itself now under-reports how many elements the buffer actually holds --
exactly Chapter 7.1's race, now corrupting a buffer's bookkeeping instead of a
histogram bucket.

--- part 2: the two-pass fix -- count exactly, allocate once, then fill ---

pass 1 (a pure count, no writes, no capacity assumed): exactly 86 elements need a slot
a single allocation of exactly 86 slots happens here, between the two kernel
launches -- no concurrency to race over, because nothing else runs at the same time
pass 2 (Section 8.1's atomicAdd reservation, now into an EXACTLY sized buffer):

every one of the 86 allocated slots was filled, none left empty: yes
buffer size matches the exact pass-1 count, with zero wasted capacity: yes

self-check: naive growth demonstrably loses data, two-pass fix wastes nothing
and loses nothing: confirmed
```

The naive pattern loses 31 of 32 appends in the traced wave and under-reports its own size — exactly Chapter 7.1's race, now corrupting buffer bookkeeping instead of a histogram bucket. The two-pass fix wastes nothing and loses nothing: every one of the 86 exactly-allocated slots gets filled.

!!! warning "[COMMON TRAP] Believing atomicAdd on the size counter alone would fix this"
    It is tempting to conclude the ONLY bug in Section 8.2's naive pattern is the non-atomic size update, and that swapping in `atomicAdd(&size, 1)` for the counter would make growing the buffer mid-kernel safe. It would fix the COUNTING race — every thread would get a genuinely unique slot number — but it does nothing about the CAPACITY itself: if the buffer's true capacity is 100 and 130 threads all successfully claim unique slots via a correct atomicAdd, slots 100 through 129 still write past the end of a 100-element allocation, corrupting whatever memory happens to sit after it. atomicAdd fixes the RACE for a slot NUMBER; it cannot fix a buffer that was simply never big enough, which is exactly why Section 8.2's real fix is the two-pass count-then-allocate pattern, not a better atomic.

## 8.3 Fixed-Capacity Buffers with Overflow Detection

### Intuition

Section 8.2's two-pass fix is the right answer whenever a second full pass over the data is affordable. Plenty of real GPU workloads cannot afford one: collecting ray-triangle intersections, contact points in a physics step, or frontier edges in a graph traversal often needs a SINGLE pass, into a buffer whose capacity was fixed before the kernel even launched. This section builds the pattern actually used there.

### The Sequential (CPU) Baseline

```
int bounded_append_cpu(std::vector<int>& buffer, int value, int capacity) {
    if ((int)buffer.size() < capacity) {
        buffer.push_back(value);
        return (int)buffer.size() - 1;   // the slot just filled
    }
    return -1;   // over capacity -- caller can detect and react
}
```

On a CPU, checking the size before writing is completely safe with one thread — there is no gap for another thread's write to land in between the check and the append. Section 8.3's atomicAdd-based version exists to give many GPU threads that identical guarantee: a correct, race-free count of true demand and a correctly bounded set of writes, even when the check-then-write pattern above is being executed by many threads simultaneously instead of just one.

### The Concept, In Detail

The idea is to keep Section 8.1's atomicAdd reservation exactly as it was, but check the returned position against a fixed capacity BEFORE writing, rather than assuming a slot always exists:

```
N=300 elements, keep multiples of 3 (true demand = 100), fixed CAPACITY=64

every qualifying thread:  pos = atomicAdd(&count, 1)   <- ALWAYS correct,
                                                            never lost, exactly
                                                            Section 8.1's guarantee

  pos  0 .. 63   (pos < 64):  WRITE to buffer[pos]          -- 64 elements land
  pos 64 .. 99   (pos >= 64): DROPPED, buffer left untouched -- 36 elements dropped

final count = 100  -- EXACTLY the true demand, matching an independent
                      reference. The COUNTER never overflows or loses
                      track of anything; only the fixed-size BUFFER does,
                      and only past the point capacity actually allows.

overflow = count - capacity = 100 - 64 = 36  -- known EXACTLY, not
                                                 estimated or guessed at
```

The critical difference from Section 8.2's naive growth: nothing here is silently wrong. The `atomicAdd` counter is exactly as correct as it was in Section 8.1 — it is never the thing that overflows. Only the fixed-size buffer's WRITES are bounded, deliberately, by an explicit `if (pos < capacity)` check, and the gap between the counter's true count and the capacity tells the caller precisely how many elements overflowed, with no uncertainty at all.

### Code and Verification

```cpp
#include <cstdio>
#include <vector>

// Chapter 8.3 -- Section 8.2's two-pass fix (count exactly, allocate
// once, then fill) is the right answer whenever a second full pass over
// the data is affordable. Plenty of real GPU workloads cannot afford it:
// collecting the list of ray-triangle intersections, or contact points
// in a physics step, or edges in a graph traversal frontier, often needs
// a SINGLE pass, into a buffer whose capacity was fixed before the
// kernel even launched. This section builds the pattern actually used
// there: reserve a slot with Section 8.1's atomicAdd exactly as before,
// but now check the returned position against a fixed capacity before
// writing -- and show that the counter itself stays perfectly correct
// even when the BUFFER cannot hold everything it is asked to.

#define BLOCK_SIZE 32

// Every thread whose element passes the predicate atomically claims the
// NEXT slot, exactly Section 8.1's reservation -- but now the buffer has
// a fixed CAPACITY, and a claimed position at or past that capacity is
// deliberately left unwritten rather than corrupting adjacent memory.
__global__ void bounded_append_with_overflow_check(const int* g_data, int* g_out,
                                                     int* g_count, int n, int capacity) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n && g_data[idx] % 3 == 0) {
        int pos = atomicAdd(g_count, 1);   // always correct: every true attempt is counted
        if (pos < capacity) {
            g_out[pos] = g_data[idx];      // only write if a real slot exists
        }
        // else: intentionally dropped -- `pos - capacity` distinct elements have
        // now been dropped by the time this thread runs, and g_count still
        // reports the TRUE total demand, letting the caller detect this exactly.
    }
}

// ---- Host-side simulation of the identical bounded-reservation logic ----

struct BoundedResult {
    std::vector<int> buffer;   // sized to `capacity`, only `min(true_count, capacity)` slots meaningful
    int true_count;            // what g_count ends up holding: EVERY true attempt, lost or not
    int stored_count;          // how many elements actually landed in the buffer
};

BoundedResult simulate_bounded_append(const std::vector<int>& data, int capacity) {
    std::vector<int> buffer(capacity, -1);
    int cursor = 0;   // atomicAdd(&cursor, 1) in the real kernel
    int stored = 0;
    for (int v : data) {
        if (v % 3 == 0) {
            int pos = cursor;
            cursor = cursor + 1;           // ALWAYS advances -- atomicAdd never loses a count
            if (pos < capacity) {
                buffer[pos] = v;
                stored++;
            }
        }
    }
    return {buffer, cursor, stored};
}

int simulate_reference_count(const std::vector<int>& data) {
    int count = 0;
    for (int v : data) if (v % 3 == 0) count++;
    return count;
}

int main() {
    printf("=== Section 8.3: bounded atomic reservation with overflow detection ===\n\n");

    const int N = 300;
    const int CAPACITY = 64;
    std::vector<int> data(N);
    for (int i = 0; i < N; i++) data[i] = i;

    int reference_count = simulate_reference_count(data);
    auto result = simulate_bounded_append(data, CAPACITY);

    printf("N = %d elements, keeping multiples of 3, fixed buffer capacity = %d\n\n", N, CAPACITY);
    printf("independent reference count of true demand: %d\n", reference_count);
    printf("counter's reported true demand (g_count after the kernel): %d\n", result.true_count);
    printf("elements actually stored in the buffer:                    %d\n", result.stored_count);
    printf("elements dropped due to overflow:                          %d\n",
           result.true_count - result.stored_count);

    bool every_stored_slot_filled = true;
    for (int i = 0; i < result.stored_count; i++) {
        if (result.buffer[i] == -1) every_stored_slot_filled = false;
    }

    bool counter_matches_reference = (result.true_count == reference_count);
    bool stored_capped_at_capacity = (result.stored_count == CAPACITY);
    bool overflow_detected = (result.true_count > CAPACITY);

    printf("\ncounter exactly matches independent reference demand (never lost, even though\n");
    printf("the buffer itself overflowed): %s\n", counter_matches_reference ? "yes" : "NO -- BUG");
    printf("stored count correctly capped at capacity: %s\n", stored_capped_at_capacity ? "yes" : "NO -- BUG");
    printf("every one of the %d stored slots holds a real value, none skipped: %s\n",
           result.stored_count, every_stored_slot_filled ? "yes" : "NO -- BUG");
    printf("overflow correctly detected (true demand > capacity): %s\n", overflow_detected ? "yes" : "NO -- BUG");

    printf("\nunlike Section 8.2's naive growth race, an overflowing bounded buffer is not a\n");
    printf("silent correctness bug: the atomicAdd counter is exactly as correct here as it\n");
    printf("was in Section 8.1, so the caller always knows PRECISELY how much overflowed\n");
    printf("(%d elements here) and can react -- rerun with a bigger buffer, accept the\n",
           result.true_count - result.stored_count);
    printf("drop, or fall back to Section 8.2's two-pass exact-sizing approach.\n");

    bool ok = counter_matches_reference && stored_capped_at_capacity
              && every_stored_slot_filled && overflow_detected;
    printf("\nself-check: counter exact, storage correctly bounded, overflow correctly\n");
    printf("detected: %s\n", ok ? "confirmed" : "MISMATCH");
    return ok ? 0 : 1;
}
```

**Compile and run:**

```bash
nvcc -arch=sm_80 24_bounded_reservation_with_overflow_detection.cu -o bounded_reservation
./bounded_reservation
```

**Sample input:** `N = 300` elements, predicate = "multiple of 3" (true demand = 100 elements), fixed buffer `CAPACITY = 64` (deliberately smaller than true demand, to force a real, measured overflow).

**Sample output:**

```text
=== Section 8.3: bounded atomic reservation with overflow detection ===

N = 300 elements, keeping multiples of 3, fixed buffer capacity = 64

independent reference count of true demand: 100
counter's reported true demand (g_count after the kernel): 100
elements actually stored in the buffer:                    64
elements dropped due to overflow:                          36

counter exactly matches independent reference demand (never lost, even though
the buffer itself overflowed): yes
stored count correctly capped at capacity: yes
every one of the 64 stored slots holds a real value, none skipped: yes
overflow correctly detected (true demand > capacity): yes

unlike Section 8.2's naive growth race, an overflowing bounded buffer is not a
silent correctness bug: the atomicAdd counter is exactly as correct here as it
was in Section 8.1, so the caller always knows PRECISELY how much overflowed
(36 elements here) and can react -- rerun with a bigger buffer, accept the
drop, or fall back to Section 8.2's two-pass exact-sizing approach.

self-check: counter exact, storage correctly bounded, overflow correctly
detected: confirmed
```

The counter reports the exact true demand of 100, matching an independent reference; the buffer correctly stores exactly 64 (its full capacity, with no slot left empty); and the overflow of 36 elements is known exactly, not merely detected as "some data was lost."

!!! warning "[COMMON TRAP] Treating Section 8.3's overflow as the same bug as Section 8.2's data loss"
    Both sections end up losing some appended elements when demand exceeds what the buffer can hold, and it is tempting to see these as the same failure with different numbers. They are not: Section 8.2's naive growth loses data SILENTLY and UNPREDICTABLY — the size counter itself becomes wrong, so the caller cannot even tell how much was lost, or that anything was lost at all, and the amount varies with how threads happen to race. Section 8.3's overflow is EXACT and KNOWN — the counter always reports the true, uncorrupted demand, and the caller can compute precisely how much overflowed and decide what to do about it (retry with a bigger buffer, accept the drop, or fall back to Section 8.2's two-pass approach). One is an undetectable correctness bug; the other is a deliberate, measurable, recoverable design choice.

## Chapter Summary

Section 8.1 generalized Chapter 6 and 7's atomic slot-reservation technique to a shared append-only buffer, and proved directly — by running the identical workload under two different valid block schedules — that it guarantees uniqueness and correctness but NOT relative order, unlike Chapter 6's scan-based compaction. Section 8.2 replayed Chapter 7.1's exact lockstep race on a buffer's size counter instead of a histogram bucket, showing that a non-atomic "check, use, increment" growth pattern silently loses appended data AND corrupts its own bookkeeping — and established this book's actual answer: never grow a buffer mid-kernel, but count exactly in a first pass and allocate precisely once before a second pass fills it. Section 8.3 built the practical alternative for when a second pass is too expensive: a fixed-capacity buffer whose atomicAdd-based counter stays exactly correct even when the buffer itself overflows, turning an uncontrolled correctness bug into an exact, recoverable, measured overflow count. Part 2 now turns from these general-purpose buffer patterns to specific concurrent data structures — starting with lock-free stacks, which need a genuinely new tool this chapter's atomicAdd alone cannot provide: compare-and-swap.

## Self-Check Questions

1. Section 8.1's example kept 86 multiples of 3 from 256 elements across 8 blocks of 32 threads. If block order were reversed, how many elements would the reversed schedule keep, and would the SET of kept values change?
2. Explain concretely why Section 8.1's atomic-reservation compaction cannot be used as a drop-in replacement for Chapter 6's scan-based compaction in a context that requires preserving original relative order (for example, a log of events by timestamp).
3. Section 8.2's race trace shows `size` going from 98 to 99 instead of 130 after 32 racing lanes each append once. Using the identical non-atomic pattern, what would `size` become (and how many distinct buffer slots would actually get written) if instead only 5 lanes raced, starting from `size = 50`?
4. Explain why Section 8.2's Pass 1 (a pure count) is safe from any race at all, even though Pass 2 still needs `atomicAdd`.
5. Section 8.3 used `N = 300` and `capacity = 64`, giving 36 dropped elements. If `capacity` were instead increased to 100 (exactly matching true demand), how many elements would be dropped, and would the kernel's own `atomicAdd` logic need to change at all?
6. A teammate argues Section 8.3's overflow handling is "just as broken" as Section 8.2's naive growth, since both lose data once demand exceeds what is available. Explain the concrete difference between the two kinds of data loss.

## Where We Go Next

Part 2 turns from general-purpose buffer patterns to specific concurrent data structures. A lock-free stack needs something this chapter's `atomicAdd` alone cannot provide: a way to update a shared pointer only if it still holds the value a thread last read, so a push or pop can detect and retry when another thread got there first. That operation — compare-and-swap — is the next tool this book adds, and it builds directly on this chapter's central lesson: correctness under concurrency comes from an operation's genuine indivisibility, not from how careful the surrounding code looks.

## Worked Solutions

**1.** The reversed schedule keeps the identical 86 elements — same count, same set — because the "multiple of 3" predicate depends only on each element's own value, never on execution order. Only the ORDER those 86 values appear in the output array would differ, exactly as Section 8.1's own forward-versus-reversed comparison demonstrated for the same predicate.

**2.** Atomic reservation's output order depends on which thread's `atomicAdd` call physically executes first, which Chapter 1 already established is never guaranteed across different warps or blocks on real hardware (this book's own host-side model keeps both schedules deterministic purely for reproducibility, not because real hardware offers that guarantee). A timestamp-ordered log needs each entry's position to depend on ITS OWN original position in the input, not on scheduling — exactly what Chapter 6's exclusive scan computes, and exactly what atomic reservation structurally cannot promise.

**3.** All 5 lanes load the identical pre-wave value `size = 50` (LOAD), each independently computes `my_slot = 50` and `my_new_size = 51` (COMPUTE), and all 5 stores write back (STORE) — whichever store physically lands last, both the buffer slot and the counter reflect only ONE lane's contribution. `size` becomes 51, not 55, and exactly 1 distinct buffer slot (slot 50) actually gets written, no matter that 5 lanes each executed the pattern once — the identical structure as the 32-lane case, just with fewer lanes racing.

**4.** Pass 1 only reads data and accumulates a count — Chapter 4's own reduction is exactly this shape, and reduction's tree has no thread writing to a shared, externally-visible BUFFER position that any other thread could also target; every thread's contribution is combined through the tree's well-defined, non-overlapping dependency structure, not by racing to claim a mutable shared counter's raw value for use as an address. Pass 2 is different because it writes actual data into actual positions that must be unique per element — exactly the "claim a slot" problem `atomicAdd` exists to solve correctly.

**5.** Zero elements would be dropped, since true demand (100) would exactly equal capacity (100) — every claimed position `pos` (0 through 99) satisfies `pos < capacity`. The kernel's own `atomicAdd` logic needs no change at all: the same `if (pos < capacity)` check simply always evaluates true in this case, which is exactly why Section 8.3's pattern is described as graceful — it behaves correctly, with zero waste, whether demand is under, over, or exactly at capacity, without needing a separate code path for the exactly-matching case.

**6.** Section 8.2's naive growth loses data SILENTLY and UNPREDICTABLY — the size counter itself becomes wrong (99 instead of 130), so the caller has no way to even know how much was lost, or that anything was lost at all, and different runs with different racing lane counts would lose different, unpredictable amounts. Section 8.3's overflow is EXACT and KNOWN — the counter (100) always reports the true, uncorrupted demand, and the caller can compute precisely how much overflowed (`true_count - capacity` = 36) and decide what to do about it. Both discard data past some point, but one is an undetectable correctness bug with no way to even measure the damage, and the other is a deliberate, measurable, fully recoverable design choice.
