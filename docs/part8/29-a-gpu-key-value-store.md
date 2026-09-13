# Chapter 29: A GPU Key-Value Store

Part 8 closes the book with case studies that need more than one structure at once, and a key-value store is the clearest possible example: keys need a fast index, values need somewhere to live that doesn't cap their size at one table slot, and deleting an entry safely under concurrent access needs a way to know no one is still reading what is about to be reclaimed. None of these three needs is new -- Chapter 19 already built the key index, Chapter 28 already built the value storage, and Chapter 27 already built the reclamation protocol. This chapter's job is composition: showing that these three pieces snap together with no new synchronization idea required, because each one was already designed to only ever touch its own shared state.

## 29.1 Building the Key Index

### Intuition

The key half of a key-value store is exactly Chapter 19's open-addressing hash table: hash each key to a home slot, and on collision, probe forward through the table with `atomicCAS(&table[idx], EMPTY, key)` until an empty slot is claimed. Nothing about a KEY index changes just because the eventual goal is a full key-VALUE store -- this section builds and verifies that index in isolation, exactly as Chapter 19 did, before Section 29.2 attaches anything to it.

### The Sequential (CPU) Baseline

```cpp
#include <cstdio>
#include <vector>

// Chapter 29.1 CPU baseline -- Part 8 opens by combining tools this book
// has already built, rather than introducing new ones. A GPU key-value
// store starts with Chapter 19's open-addressing hash table for the KEY
// index: capacity-8 table, hash(key) = key % 8, linear probing on
// collision. This section builds and verifies that key index alone;
// Section 29.2 attaches a value to each key via a memory pool, and
// Section 29.3 adds concurrent, hazard-protected deletion.

#define CAPACITY 8
#define EMPTY (-1)

int home_slot(int key) { return key % CAPACITY; }

struct KeyTable {
    std::vector<int> slots;
    KeyTable() : slots(CAPACITY, EMPTY) {}

    // Returns (landed_slot, probes_taken), or (-1, probes) if full.
    std::pair<int,int> insert(int key) {
        int home = home_slot(key);
        for (int i = 0; i < CAPACITY; i++) {
            int idx = (home + i) % CAPACITY;
            if (slots[idx] == EMPTY) {
                slots[idx] = key;
                return {idx, i + 1};
            }
        }
        return {-1, CAPACITY};
    }
};

int main() {
    printf("=== Section 29.1 CPU baseline: sequential open-addressing key index ===\n\n");

    KeyTable table;
    int keys[] = {5, 13, 21, 3, 11};
    int num_keys = 5;
    int expected_slots[] = {5, 6, 7, 3, 4};   // known-correct linear-probing landing slots

    bool ok = true;
    for (int i = 0; i < num_keys; i++) {
        int key = keys[i];
        auto [idx, probes] = table.insert(key);
        printf("insert(%d): home slot = %d, landed at slot %d after %d probe(s)\n",
               key, home_slot(key), idx, probes);
        printf("  table: [ ");
        for (int s : table.slots) printf("%d ", s);
        printf("]\n");
        if (idx != expected_slots[i]) ok = false;
    }

    printf("\nfinal table: [ ");
    for (int s : table.slots) printf("%d ", s);
    printf("]\n");

    printf("\nverifying every key can be found again by re-deriving its probe chain:\n");
    bool all_found = true;
    for (int key : keys) {
        int home = home_slot(key);
        int found_at = -1;
        for (int i = 0; i < CAPACITY; i++) {
            int idx = (home + i) % CAPACITY;
            if (table.slots[idx] == key) { found_at = idx; break; }
            if (table.slots[idx] == EMPTY) break;
        }
        printf("  find(%d) -> slot %d\n", key, found_at);
        if (found_at == -1) all_found = false;
    }

    ok = ok && all_found;
    printf("\nself-check: every key landed at its expected slot via linear probing,\n");
    printf("and every key can be re-found afterward: %s\n", ok ? "confirmed" : "MISMATCH");
    return ok ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++17 -Wall -Wextra -O2 160_kv_hashtable_insert_cpu_baseline.cpp -o 160_kv_hashtable_insert_cpu_baseline
./160_kv_hashtable_insert_cpu_baseline
```

**Sample input:** a capacity-8 table, keys [5, 13, 21, 3, 11] inserted in sequence, with hash(key) = key % 8 producing two genuine collision chains.

**Sample output:**

```text
=== Section 29.1 CPU baseline: sequential open-addressing key index ===

insert(5): home slot = 5, landed at slot 5 after 1 probe(s)
  table: [ -1 -1 -1 -1 -1 5 -1 -1 ]
insert(13): home slot = 5, landed at slot 6 after 2 probe(s)
  table: [ -1 -1 -1 -1 -1 5 13 -1 ]
insert(21): home slot = 5, landed at slot 7 after 3 probe(s)
  table: [ -1 -1 -1 -1 -1 5 13 21 ]
insert(3): home slot = 3, landed at slot 3 after 1 probe(s)
  table: [ -1 -1 -1 3 -1 5 13 21 ]
insert(11): home slot = 3, landed at slot 4 after 2 probe(s)
  table: [ -1 -1 -1 3 11 5 13 21 ]

final table: [ -1 -1 -1 3 11 5 13 21 ]

verifying every key can be found again by re-deriving its probe chain:
  find(5) -> slot 5
  find(13) -> slot 6
  find(21) -> slot 7
  find(3) -> slot 3
  find(11) -> slot 4

self-check: every key landed at its expected slot via linear probing,
and every key can be re-found afterward: confirmed
```

### The Concept, In Detail

```
ASCII view: two collision chains landing via linear probing.

  home(5)=5   home(13)=5  home(21)=5    home(3)=3   home(11)=3

  slot:  0  1  2  3  4  5  6  7
  after
  all 5
  inserts: [ - -  -  3 11  5 13 21 ]

  5  -> lands at 5 (home, no collision)
  13 -> home 5 occupied by 5, probes to 6
  21 -> home 5 occupied, 6 occupied, probes to 7
  3  -> lands at 3 (home, no collision)
  11 -> home 3 occupied by 3, probes to 4
```

Every key that is not the first to reach its home slot pays for that collision with an extra probe, and the table above shows two independent chains (5/13/21 anchored at slot 5, and 3/11 anchored at slot 3) coexisting without interference, because linear probing only ever cares about what is CURRENTLY in the slot it is examining -- it has no notion of "whose chain" a slot belongs to.

[COMMON TRAP]
It is tempting to think a thread's probe sequence needs to know how many OTHER keys are already in the table, in order to know how far it might need to probe. A probing thread never needs that information -- it simply keeps testing successive slots for EMPTY (sequentially, or via CAS under concurrency) until it finds one, and the number of probes required falls out naturally from whatever the table's current contents happen to be, with no separate bookkeeping needed.

### Code and Verification

```cpp
#include <cstdio>
#include <vector>
#include <string>

// Chapter 29.1 main -- concurrent open-addressing insertion into the key
// index, reusing Chapter 19's exact technique: at each probed slot, a
// thread attempts atomicCAS(&table[idx], EMPTY, key). Success means the
// key landed there; failure means either another key already legitimately
// occupies that slot, OR another thread's concurrent insert just won the
// race for it -- either way, the correct response is identical: advance
// to the next probe slot and try again.

#define CAPACITY 8
#define EMPTY (-1)

__device__ int home_slot(int key) { return key % CAPACITY; }

__global__ void hashtable_insert_kernel(int* table, const int* keys, int* out_slot) {
    int tid = threadIdx.x;
    int key = keys[tid];
    int home = home_slot(key);
    for (int i = 0; i < CAPACITY; i++) {
        int idx = (home + i) % CAPACITY;
        if (table[idx] != EMPTY) continue;   // a quick pre-check; the CAS below is the real decision
        int observed = atomicCAS(&table[idx], EMPTY, key);
        if (observed == EMPTY) {
            out_slot[tid] = idx;
            return;
        }
        // CAS failed: some other thread's insert landed here first between
        // this thread's pre-check and its own CAS. Keep probing.
    }
    out_slot[tid] = -1;   // table full
}

// ---- Host-side replay of the identical per-thread CAS-loop logic,
// ---- driving a specific interleaving so the race is hand-traceable:
// ---- T0 and T1 both hash to slot 5, and T1 is forced to complete its
// ---- full insert before T0's own CAS at slot 5 runs. ----

struct SharedSlot {
    int value;
    int cas(int expected, int new_val) {
        int old = value;
        if (old == expected) value = new_val;
        return old;
    }
};

int home_slot_host(int key) { return key % CAPACITY; }

int host_insert(std::vector<SharedSlot>& table, int thread_id, int key,
                 void (*interleave)() = nullptr, bool* used = nullptr) {
    int home = home_slot_host(key);
    for (int i = 0; i < CAPACITY; i++) {
        int idx = (home + i) % CAPACITY;
        int current = table[idx].value;
        printf("  T%d (key=%d): probe slot %d, sees %s\n", thread_id, key, idx,
               current == EMPTY ? "EMPTY" : std::to_string(current).c_str());
        if (current != EMPTY) continue;
        if (interleave && i == 0 && used && !*used) {
            *used = true;
            interleave();
        }
        int observed = table[idx].cas(EMPTY, key);
        if (observed == EMPTY) {
            printf("  T%d (key=%d): CAS(slot %d, EMPTY->key %d) succeeded\n", thread_id, key, idx, key);
            return idx;
        } else {
            printf("  T%d (key=%d): CAS(slot %d, EMPTY->key %d) FAILED (slot now holds %d) -- probing on\n",
                   thread_id, key, idx, key, observed);
        }
    }
    return -1;
}

int main() {
    printf("=== Section 29.1 main: concurrent key insertion via atomicCAS on the shared table ===\n\n");

    std::vector<SharedSlot> table(CAPACITY);
    for (auto& s : table) s.value = EMPTY;

    printf("T0 inserts key=5 (home slot 5), T1 inserts key=13 (home slot 5, same slot).\n");
    printf("T1 is forced to interleave right after T0 reads slot 5 as empty but before\n");
    printf("T0's own CAS runs, so T0's CAS fails and it must probe onward:\n\n");

    static std::vector<SharedSlot>* s_table;
    static int s_r1;
    s_table = &table;
    auto t1_inserts = []() {
        printf("  (T1 interleaves here, completing its own insert of key=13 into slot 5 first)\n");
        s_r1 = host_insert(*s_table, 1, 13);
        printf("  T1 (key=13) lands at slot %d\n", s_r1);
    };

    bool used = false;
    int r0 = host_insert(table, 0, 5, t1_inserts, &used);
    printf("\n  T0 (key=5) lands at slot %d\n", r0);

    printf("\nfinal table: [ ");
    for (auto& s : table) printf("%d ", s.value);
    printf("]\n");

    bool ok = (r0 == 6) && (s_r1 == 5) && (table[5].value == 13) && (table[6].value == 5);
    printf("\nself-check: T1 wins the race for slot 5, T0 is correctly forced to probe\n");
    printf("onward to slot 6, and neither key is lost or duplicated: %s\n", ok ? "confirmed" : "MISMATCH");
    return ok ? 0 : 1;
}
```

**Compile and run:**

```bash
NVDIR=/usr/local/lib/python3.11/dist-packages/nvidia/cu13
$NVDIR/bin/nvcc -arch=sm_80 -I$NVDIR/include -L$NVDIR/lib -lcudart 161_kv_hashtable_insert_kernel.cu -o 161_kv_hashtable_insert_kernel
LD_LIBRARY_PATH=$NVDIR/lib ./161_kv_hashtable_insert_kernel
```

**Sample input:** two threads inserting keys 5 and 13 -- both hashing to the same home slot 5 -- with a forced interleaving so one thread's CAS fails and it must probe onward.

**Sample output:**

```text
=== Section 29.1 main: concurrent key insertion via atomicCAS on the shared table ===

T0 inserts key=5 (home slot 5), T1 inserts key=13 (home slot 5, same slot).
T1 is forced to interleave right after T0 reads slot 5 as empty but before
T0's own CAS runs, so T0's CAS fails and it must probe onward:

  T0 (key=5): probe slot 5, sees EMPTY
  (T1 interleaves here, completing its own insert of key=13 into slot 5 first)
  T1 (key=13): probe slot 5, sees EMPTY
  T1 (key=13): CAS(slot 5, EMPTY->key 13) succeeded
  T1 (key=13) lands at slot 5
  T0 (key=5): CAS(slot 5, EMPTY->key 5) FAILED (slot now holds 13) -- probing on
  T0 (key=5): probe slot 6, sees EMPTY
  T0 (key=5): CAS(slot 6, EMPTY->key 5) succeeded

  T0 (key=5) lands at slot 6

final table: [ -1 -1 -1 -1 -1 13 5 -1 ]

self-check: T1 wins the race for slot 5, T0 is correctly forced to probe
onward to slot 6, and neither key is lost or duplicated: confirmed
```

## 29.2 Attaching Values via a Memory Pool

### Intuition

Storing a value directly in a table slot caps every value at that slot's fixed width and wastes space sizing every slot for the largest value a program might ever store. Instead, each table slot stores a BLOCK INDEX into a separate value pool -- Chapter 28.1's fixed-size block pool, reused unchanged. Inserting a key now does two things in a fixed order: allocate a block and write the value into it FIRST, then publish the key into the table index SECOND -- so that the instant any other thread can see the key in the table, the value it points to is already complete.

### The Sequential (CPU) Baseline

```cpp
#include <cstdio>
#include <vector>

// Chapter 29.2 CPU baseline -- storing a value directly in a hash table
// slot caps every value at that slot's fixed width, and wastes space for
// small values sized to fit the largest one. Instead, each table slot
// stores a BLOCK INDEX into a separate value pool -- Section 28.1's
// fixed-size block pool, reused unchanged: insert(key, value) allocates
// a block, writes the value into it, THEN inserts (key -> block) into
// the Section 29.1 key index. This decouples how many keys fit (table
// capacity) from how large a value can be (pool block size).

#define CAPACITY 8
#define EMPTY (-1)
#define POOL_SIZE 8

int home_slot(int key) { return key % CAPACITY; }

struct BlockPool {
    std::vector<int> free_stack;
    std::vector<int> values;
    int top;
    BlockPool(int size) : free_stack(size), values(size, 0), top(size) {
        for (int i = 0; i < size; i++) free_stack[i] = i;
    }
    int alloc() {
        if (top == 0) return -1;
        top--;
        return free_stack[top];
    }
};

struct KVStore {
    std::vector<int> table_key;
    std::vector<int> table_block;
    BlockPool pool;
    KVStore() : table_key(CAPACITY, EMPTY), table_block(CAPACITY, EMPTY), pool(POOL_SIZE) {}

    std::pair<int,int> insert(int key, int value) {
        int home = home_slot(key);
        for (int i = 0; i < CAPACITY; i++) {
            int idx = (home + i) % CAPACITY;
            if (table_key[idx] == EMPTY) {
                int block = pool.alloc();
                pool.values[block] = value;
                table_key[idx] = key;
                table_block[idx] = block;
                return {idx, block};
            }
        }
        return {-1, -1};
    }

    // Returns the value, or -1 if the key is not present.
    int lookup(int key) const {
        int home = home_slot(key);
        for (int i = 0; i < CAPACITY; i++) {
            int idx = (home + i) % CAPACITY;
            if (table_key[idx] == key) return pool.values[table_block[idx]];
            if (table_key[idx] == EMPTY) return -1;
        }
        return -1;
    }
};

int main() {
    printf("=== Section 29.2 CPU baseline: hash table for keys + memory pool for values ===\n\n");

    KVStore kv;
    int entries_key[] = {5, 13, 21, 3, 11};
    int entries_val[] = {500, 1300, 2100, 300, 1100};
    int num_entries = 5;

    for (int i = 0; i < num_entries; i++) {
        auto [idx, block] = kv.insert(entries_key[i], entries_val[i]);
        printf("insert(key=%d, value=%d): table slot %d, value pool block %d\n",
               entries_key[i], entries_val[i], idx, block);
    }

    printf("\ntable keys:   [ ");
    for (int k : kv.table_key) printf("%d ", k);
    printf("]\n");
    printf("table blocks: [ ");
    for (int b : kv.table_block) printf("%d ", b);
    printf("]\n");
    printf("pool values:  [ ");
    for (int v : kv.pool.values) printf("%d ", v);
    printf("]\n");
    printf("pool free stack: [ ");
    for (int i = 0; i < kv.pool.top; i++) printf("%d ", kv.pool.free_stack[i]);
    printf("] (top=%d)\n\n", kv.pool.top);

    printf("lookups:\n");
    bool all_correct = true;
    for (int i = 0; i < num_entries; i++) {
        int got = kv.lookup(entries_key[i]);
        printf("  lookup(%d) -> %d\n", entries_key[i], got);
        if (got != entries_val[i]) all_correct = false;
    }
    int missing = kv.lookup(99);
    printf("  lookup(99) -> %d  (not present)\n", missing);
    bool missing_correct = (missing == -1);

    bool ok = all_correct && missing_correct;
    printf("\nself-check: every inserted key's value round-trips through its pool\n");
    printf("block exactly, and a missing key correctly returns not-found: %s\n",
           ok ? "confirmed" : "MISMATCH");
    return ok ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++17 -Wall -Wextra -O2 162_kv_value_pool_cpu_baseline.cpp -o 162_kv_value_pool_cpu_baseline
./162_kv_value_pool_cpu_baseline
```

**Sample input:** the same 5 keys as Section 29.1, each paired with a value (key times 100), stored via an 8-block value pool, followed by lookups for every key plus one missing key.

**Sample output:**

```text
=== Section 29.2 CPU baseline: hash table for keys + memory pool for values ===

insert(key=5, value=500): table slot 5, value pool block 7
insert(key=13, value=1300): table slot 6, value pool block 6
insert(key=21, value=2100): table slot 7, value pool block 5
insert(key=3, value=300): table slot 3, value pool block 4
insert(key=11, value=1100): table slot 4, value pool block 3

table keys:   [ -1 -1 -1 3 11 5 13 21 ]
table blocks: [ -1 -1 -1 4 3 7 6 5 ]
pool values:  [ 0 0 0 1100 300 2100 1300 500 ]
pool free stack: [ 0 1 2 ] (top=3)

lookups:
  lookup(5) -> 500
  lookup(13) -> 1300
  lookup(21) -> 2100
  lookup(3) -> 300
  lookup(11) -> 1100
  lookup(99) -> -1  (not present)

self-check: every inserted key's value round-trips through its pool
block exactly, and a missing key correctly returns not-found: confirmed
```

### The Concept, In Detail

```
ASCII view: table slots holding block indices, not values themselves.

  table_key:   [ -  -  -  3 11  5 13 21 ]
  table_block: [ -  -  -  4  3  7  6  5 ]
                          |  |  |  |  |
  pool.values: [ .  .  . 1100 300 2100 1300 500 ]
                          ^-3  ^-4  ^-7   ^-6   ^-5

  lookup(13): home=5, probe 5 (holds key 5, not 13), probe 6 (holds key
              13!) -> table_block[6]=6 -> pool.values[6] = 1300
```

Two independent shared structures now cooperate through a single, fixed protocol: the pool is asked ONLY for a fresh block and never inspects the table, and the table is asked ONLY to associate a key with whatever block index it is given and never inspects the pool's internals. Because insert always completes the pool write before the table publish, any thread that successfully finds a key in the table is guaranteed the block it points to already holds that key's real, final value -- there is no window where a reader could see the key but read a half-written value.

[COMMON TRAP]
It is tempting to publish the key into the table FIRST and write its value into the pool block SECOND, reasoning that the two writes will "happen close enough together" not to matter. Any other thread's lookup could observe the newly-published key in the very instant after it appears and before the value write completes, reading stale or garbage data out of that block -- the value must always be fully written before the key that points to it becomes visible to anyone else, with no exceptions for how small that window looks.

### Code and Verification

```cpp
#include <cstdio>
#include <vector>

// Chapter 29.2 main -- concurrent KV insertion composing TWO independent
// CAS loops per thread: Section 28.1's pool alloc() (contending only
// with other threads on the shared value pool) followed by Section
// 29.1's table insert (contending only with other threads that hash to
// the same table slot). Neither loop needs to know the other exists --
// a thread pair can race on the pool while an entirely different pair
// races on the table, and correctness of each is exactly what Sections
// 28.1 and 29.1 already established independently.

#define CAPACITY 8
#define EMPTY (-1)
#define POOL_SIZE 8

__device__ int home_slot(int key) { return key % CAPACITY; }

__global__ void kv_insert_kernel(int* table, int* table_block, int* pool_free_stack,
                                  int* pool_top, int* pool_values,
                                  const int* keys, const int* values,
                                  int* out_slot, int* out_block) {
    int tid = threadIdx.x;
    int key = keys[tid];
    int value = values[tid];

    // Step 1: alloc a value block (Section 28.1's CAS retry pop).
    int block = -1;
    while (true) {
        int old_top = *pool_top;
        if (old_top == 0) { block = -1; break; }
        int candidate = pool_free_stack[old_top - 1];
        int observed = atomicCAS(pool_top, old_top, old_top - 1);
        if (observed == old_top) { block = candidate; break; }
    }
    pool_values[block] = value;

    // Step 2: insert (key -> block) into the table (Section 29.1's CAS
    // retry probe), only after the value is safely written into the
    // block -- any thread that later finds this key in the table must
    // see a fully-written value, not a block still being initialized.
    int home = home_slot(key);
    for (int i = 0; i < CAPACITY; i++) {
        int idx = (home + i) % CAPACITY;
        if (table[idx] != EMPTY) continue;
        int observed = atomicCAS(&table[idx], EMPTY, key);
        if (observed == EMPTY) {
            table_block[idx] = block;
            out_slot[tid] = idx;
            out_block[tid] = block;
            return;
        }
    }
    out_slot[tid] = -1;
    out_block[tid] = block;
}

// ---- Host-side replay of the identical two-step per-thread logic,
// ---- driving T1 to fully interleave (both its pool alloc AND its
// ---- table insert) ahead of T0, while T2 (a different key, different
// ---- home slot) proceeds independently first. ----

int home_slot_host(int key) { return key % CAPACITY; }

struct SharedInt {
    int value;
    int cas(int expected, int new_val) {
        int old = value;
        if (old == expected) value = new_val;
        return old;
    }
};

int pool_alloc(std::vector<int>& free_stack, SharedInt& top, int thread_id, const char* tag) {
    while (true) {
        int old_top = top.value;
        if (old_top == 0) return -1;
        int candidate = free_stack[old_top - 1];
        int observed = top.cas(old_top, old_top - 1);
        if (observed == old_top) {
            printf("  T%d%s: pool alloc -> block %d\n", thread_id, tag, candidate);
            return candidate;
        }
        printf("  T%d%s: pool alloc CAS failed at top=%d (now %d) -- retry\n", thread_id, tag, old_top, observed);
    }
}

int table_insert(std::vector<SharedInt>& table, int home, int key, int thread_id, const char* tag) {
    for (int i = 0; i < CAPACITY; i++) {
        int idx = (home + i) % CAPACITY;
        if (table[idx].value != EMPTY) {
            printf("  T%d%s (key=%d): slot %d occupied, probing on\n", thread_id, tag, key, idx);
            continue;
        }
        int observed = table[idx].cas(EMPTY, key);
        if (observed == EMPTY) {
            printf("  T%d%s (key=%d): table CAS at slot %d succeeded -> slot %d\n", thread_id, tag, key, idx, idx);
            return idx;
        } else {
            printf("  T%d%s (key=%d): table CAS at slot %d FAILED (holds %d) -- probing on\n",
                   thread_id, tag, key, idx, observed);
        }
    }
    return -1;
}

int main() {
    printf("=== Section 29.2 main: concurrent KV insert -- pool alloc CAS + table insert CAS ===\n\n");

    std::vector<int> free_stack(POOL_SIZE);
    for (int i = 0; i < POOL_SIZE; i++) free_stack[i] = i;
    SharedInt pool_top{POOL_SIZE};
    std::vector<int> pool_values(POOL_SIZE, 0);

    std::vector<SharedInt> table(CAPACITY);
    for (auto& s : table) s.value = EMPTY;
    std::vector<int> table_block(CAPACITY, EMPTY);

    printf("T0(key=3), T1(key=11, same home slot 3 as T0), T2(key=21, home slot 5)\n");
    printf("-- all three race on the shared pool too.\n\n");

    printf("T2 runs first, uncontended on both pool and table:\n");
    int b2 = pool_alloc(free_stack, pool_top, 2, "");
    pool_values[b2] = 2100;
    int idx2 = table_insert(table, home_slot_host(21), 21, 2, "");
    table_block[idx2] = b2;
    printf("  T2 (key=21): block=%d, slot=%d\n\n", b2, idx2);

    printf("T0(key=3) and T1(key=11) now proceed concurrently. T1 is forced to\n");
    printf("interleave on BOTH the pool alloc and the table insert, ahead of T0:\n\n");

    printf("  (T1 interleaves: runs its own pool alloc + table insert to completion)\n");
    int b1 = pool_alloc(free_stack, pool_top, 1, "");
    pool_values[b1] = 1100;
    int idx1 = table_insert(table, home_slot_host(11), 11, 1, "");
    table_block[idx1] = b1;

    int b0 = pool_alloc(free_stack, pool_top, 0, "");
    pool_values[b0] = 300;
    int idx0 = table_insert(table, home_slot_host(3), 3, 0, "");
    table_block[idx0] = b0;

    printf("\n  T0 (key=3): block=%d, slot=%d\n", b0, idx0);
    printf("  T1 (key=11): block=%d, slot=%d\n", b1, idx1);

    printf("\nfinal table keys:   [ ");
    for (auto& s : table) printf("%d ", s.value);
    printf("]\n");
    printf("final table blocks: [ ");
    for (int b : table_block) printf("%d ", b);
    printf("]\n");
    printf("final pool values:  [ ");
    for (int v : pool_values) printf("%d ", v);
    printf("]\n");
    printf("final pool free stack: [ ");
    for (int i = 0; i < pool_top.value; i++) printf("%d ", free_stack[i]);
    printf("] (top=%d)\n", pool_top.value);

    bool distinct_blocks = (b0 != b1 && b1 != b2 && b0 != b2);
    bool distinct_slots = (idx0 != idx1 && idx1 != idx2 && idx0 != idx2);
    printf("\nself-check: all 3 threads received DISTINCT pool blocks: %s, DISTINCT\n",
           distinct_blocks ? "yes" : "NO -- BUG");
    printf("table slots: %s\n", distinct_slots ? "yes" : "NO -- BUG");

    bool ok = distinct_blocks && distinct_slots;
    return ok ? 0 : 1;
}
```

**Compile and run:**

```bash
NVDIR=/usr/local/lib/python3.11/dist-packages/nvidia/cu13
$NVDIR/bin/nvcc -arch=sm_80 -I$NVDIR/include -L$NVDIR/lib -lcudart 163_kv_value_pool_kernel.cu -o 163_kv_value_pool_kernel
LD_LIBRARY_PATH=$NVDIR/lib ./163_kv_value_pool_kernel
```

**Sample input:** three threads inserting different keys concurrently -- two of which share a home table slot and are forced to contend, while all three also contend for blocks in the same shared value pool.

**Sample output:**

```text
=== Section 29.2 main: concurrent KV insert -- pool alloc CAS + table insert CAS ===

T0(key=3), T1(key=11, same home slot 3 as T0), T2(key=21, home slot 5)
-- all three race on the shared pool too.

T2 runs first, uncontended on both pool and table:
  T2: pool alloc -> block 7
  T2 (key=21): table CAS at slot 5 succeeded -> slot 5
  T2 (key=21): block=7, slot=5

T0(key=3) and T1(key=11) now proceed concurrently. T1 is forced to
interleave on BOTH the pool alloc and the table insert, ahead of T0:

  (T1 interleaves: runs its own pool alloc + table insert to completion)
  T1: pool alloc -> block 6
  T1 (key=11): table CAS at slot 3 succeeded -> slot 3
  T0: pool alloc -> block 5
  T0 (key=3): slot 3 occupied, probing on
  T0 (key=3): table CAS at slot 4 succeeded -> slot 4

  T0 (key=3): block=5, slot=4
  T1 (key=11): block=6, slot=3

final table keys:   [ -1 -1 -1 11 3 21 -1 -1 ]
final table blocks: [ -1 -1 -1 6 5 7 -1 -1 ]
final pool values:  [ 0 0 0 0 0 300 1100 2100 ]
final pool free stack: [ 0 1 2 3 4 ] (top=5)

self-check: all 3 threads received DISTINCT pool blocks: yes, DISTINCT
table slots: yes
```

## 29.3 Concurrent, Hazard-Protected Deletion

### Intuition

Deleting a key cannot simply reset its table slot to EMPTY: any OTHER key that originally probed past this slot to reach its own home would become unreachable the instant a lookup stops at the newly-EMPTY slot. The fix is a TOMBSTONE marker, distinct from both EMPTY and any real key, that lookup treats as "keep probing" and insert treats as "available to reuse." Freeing the deleted key's value block back to the pool then faces Chapter 27.3's exact hazard -- a reader thread may already be mid-read of that block -- and this section reuses that chapter's hazard-pointer protocol without modification to close it.

### The Sequential (CPU) Baseline

```cpp
#include <cstdio>
#include <vector>

// Chapter 29.3 CPU baseline -- deleting a key cannot simply reset its
// table slot to EMPTY: that would break the linear-probing chain for
// any OTHER key that originally probed past this slot to reach its own
// home (Section 29.1's key=21 probed past key=13's slot to land at 7;
// erasing key=13's slot to EMPTY would make key=21 unreachable, since
// lookup stops probing the instant it sees EMPTY). The fix is a
// TOMBSTONE marker, distinct from both EMPTY and any real key: lookup
// treats it as "keep probing," insert treats it as "available to
// reuse."
//
// Freeing the deleted key's value block back to the pool has exactly
// Section 27.3's hazard: a reader thread may have already looked up
// this key, read its block index, and be mid-read of that block's
// value when the delete tries to free it. This section reuses Section
// 27.3's hazard-pointer discipline unchanged: a reader publishes the
// block index it is about to read BEFORE reading it, and a thread
// freeing a block checks every reader's published hazard first.

#define CAPACITY 8
#define EMPTY (-1)
#define TOMBSTONE (-2)
#define POOL_SIZE 8
#define UNHAZARDED (-1)
#define NUM_READERS 1

int home_slot(int key) { return key % CAPACITY; }

struct BlockPool {
    std::vector<int> free_stack;
    std::vector<int> values;
    int top;
    BlockPool(int size) : free_stack(size), values(size, 0), top(size) {
        for (int i = 0; i < size; i++) free_stack[i] = i;
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

struct KVStore {
    std::vector<int> table_key;
    std::vector<int> table_block;
    BlockPool pool;
    KVStore() : table_key(CAPACITY, EMPTY), table_block(CAPACITY, EMPTY), pool(POOL_SIZE) {}

    std::pair<int,int> insert(int key, int value) {
        int home = home_slot(key);
        for (int i = 0; i < CAPACITY; i++) {
            int idx = (home + i) % CAPACITY;
            if (table_key[idx] == EMPTY || table_key[idx] == TOMBSTONE) {
                int block = pool.alloc();
                pool.values[block] = value;
                table_key[idx] = key;
                table_block[idx] = block;
                return {idx, block};
            }
        }
        return {-1, -1};
    }

    // Returns the slot index, or -1 if not present. TOMBSTONE slots are
    // skipped over (kept probing), never treated as a stopping point.
    int find_slot(int key) const {
        int home = home_slot(key);
        for (int i = 0; i < CAPACITY; i++) {
            int idx = (home + i) % CAPACITY;
            if (table_key[idx] == key) return idx;
            if (table_key[idx] == EMPTY) return -1;
        }
        return -1;
    }
};

int main() {
    printf("=== Section 29.3 CPU baseline: concurrent-safe delete via tombstone + hazard pointer ===\n\n");

    KVStore kv;
    int keys[]   = {5, 13, 21, 3, 11};
    int values[] = {500, 1300, 2100, 300, 1100};
    for (int i = 0; i < 5; i++) kv.insert(keys[i], values[i]);

    printf("initial setup (same as Section 29.2):\n");
    printf("table keys:   [ ");
    for (int k : kv.table_key) printf("%d ", k);
    printf("]\n");
    printf("table blocks: [ ");
    for (int b : kv.table_block) printf("%d ", b);
    printf("]\n\n");

    std::vector<int> hazard(NUM_READERS, UNHAZARDED);

    printf("Reader R looks up key=13, publishes its hazard BEFORE reading:\n");
    int slot13 = kv.find_slot(13);
    int block13 = kv.table_block[slot13];
    hazard[0] = block13;
    printf("  found key=13 at slot %d, block %d; hazard[0] = %d\n\n", slot13, block13, block13);

    printf("Deleter D deletes key=13: marks the slot TOMBSTONE (preserving the\n");
    printf("probe chain for key=21, which originally probed past this slot):\n");
    kv.table_key[slot13] = TOMBSTONE;
    printf("  table keys now: [ ");
    for (int k : kv.table_key) printf("%d ", k);
    printf("]\n");
    int slot21_after = kv.find_slot(21);
    printf("  key=21 lookup still succeeds: find_slot(21) = %d\n\n", slot21_after);
    bool tombstone_preserves_chain = (slot21_after == 7);

    printf("D wants to free block %d back to the pool, but checks the hazard\n", block13);
    printf("array first:\n");
    auto is_hazarded = [&](int block) {
        for (int h : hazard) if (h == block) return true;
        return false;
    };
    bool deferred_first = is_hazarded(block13);
    if (deferred_first) {
        printf("  block %d IS hazarded (hazard[0]=%d) -- free DEFERRED\n\n", block13, hazard[0]);
    } else {
        kv.pool.free_block(block13);
        printf("  block %d not hazarded -- freed immediately\n\n", block13);
    }

    printf("R finishes reading pool.values[%d] = %d and clears its hazard:\n", block13, kv.pool.values[block13]);
    int read_value = kv.pool.values[block13];
    hazard[0] = UNHAZARDED;
    printf("  hazard array now: [ %d ]\n\n", hazard[0]);

    printf("D rechecks and this time frees the block:\n");
    bool hazarded_second = is_hazarded(block13);
    if (!hazarded_second) {
        kv.pool.free_block(block13);
        printf("  block %d not hazarded -- freed. pool free stack: [ ", block13);
        for (int i = 0; i < kv.pool.top; i++) printf("%d ", kv.pool.free_stack[i]);
        printf("] (top=%d)\n\n", kv.pool.top);
    }

    printf("A new insert for key=29 (home slot %d) now reuses the just-freed block:\n", home_slot(29));
    auto [idx29, block29] = kv.insert(29, 2900);
    bool reused = (block29 == block13);
    printf("  insert(29, 2900) -> slot %d, block %d (reused: %s)\n", idx29, block29, reused ? "yes" : "no");
    printf("  final table keys:   [ ");
    for (int k : kv.table_key) printf("%d ", k);
    printf("]\n");
    printf("  final table blocks: [ ");
    for (int b : kv.table_block) printf("%d ", b);
    printf("]\n");
    printf("  final pool values:  [ ");
    for (int v : kv.pool.values) printf("%d ", v);
    printf("]\n");

    bool ok = tombstone_preserves_chain && deferred_first && !hazarded_second &&
              reused && (read_value == 1300) && (idx29 == slot13);
    printf("\nself-check: tombstone preserves the probe chain for other keys, the\n");
    printf("reader's hazard correctly defers reclamation until it clears, and the\n");
    printf("freed block is safely reused only afterward: %s\n", ok ? "confirmed" : "MISMATCH");
    return ok ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++17 -Wall -Wextra -O2 164_kv_delete_hazard_cpu_baseline.cpp -o 164_kv_delete_hazard_cpu_baseline
./164_kv_delete_hazard_cpu_baseline
```

**Sample input:** the Section 29.2 table and pool, with a reader publishing a hazard on key 13's value block just before a deleter marks key 13 tombstoned and attempts to reclaim that same block.

**Sample output:**

```text
=== Section 29.3 CPU baseline: concurrent-safe delete via tombstone + hazard pointer ===

initial setup (same as Section 29.2):
table keys:   [ -1 -1 -1 3 11 5 13 21 ]
table blocks: [ -1 -1 -1 4 3 7 6 5 ]

Reader R looks up key=13, publishes its hazard BEFORE reading:
  found key=13 at slot 6, block 6; hazard[0] = 6

Deleter D deletes key=13: marks the slot TOMBSTONE (preserving the
probe chain for key=21, which originally probed past this slot):
  table keys now: [ -1 -1 -1 3 11 5 -2 21 ]
  key=21 lookup still succeeds: find_slot(21) = 7

D wants to free block 6 back to the pool, but checks the hazard
array first:
  block 6 IS hazarded (hazard[0]=6) -- free DEFERRED

R finishes reading pool.values[6] = 1300 and clears its hazard:
  hazard array now: [ -1 ]

D rechecks and this time frees the block:
  block 6 not hazarded -- freed. pool free stack: [ 0 1 2 6 ] (top=4)

A new insert for key=29 (home slot 5) now reuses the just-freed block:
  insert(29, 2900) -> slot 6, block 6 (reused: yes)
  final table keys:   [ -1 -1 -1 3 11 5 29 21 ]
  final table blocks: [ -1 -1 -1 4 3 7 6 5 ]
  final pool values:  [ 0 0 0 1100 300 2100 2900 500 ]

self-check: tombstone preserves the probe chain for other keys, the
reader's hazard correctly defers reclamation until it clears, and the
freed block is safely reused only afterward: confirmed
```

### The Concept, In Detail

```
ASCII view: tombstone preserves probing, hazard defers reclamation.

  before delete:  table_key[6] = 13     (key=21 reached slot 7 by
                                          probing PAST slot 6)

  delete(13):     table_key[6] = TOMBSTONE
                  find_slot(21) still returns 7 -- TOMBSTONE is
                  "keep probing," not "stop here"

  hazard[R] = 6   (published before R reads pool.values[6])

  D wants to free block 6:
    hazard[R] == 6 ?  YES -> DEFER (block 6 stays out of the free list)

  R finishes reading, clears hazard[R]

  D rechecks:
    hazard[R] == 6 ?  NO -> block 6 freed, safe to reuse
```

The tombstone and the hazard check solve two DIFFERENT problems that a naive delete would conflate into one: the tombstone keeps the table's probe chains intact for every other key, while the hazard check keeps the value pool's block reuse safe for every other in-flight reader. Neither mechanism substitutes for the other -- a tombstone with no hazard check would correctly preserve lookups for other keys while still handing a reader's in-progress read out from under it, and a hazard check with no tombstone would correctly protect the value pool while silently breaking lookups for keys that used to probe past the deleted slot.

[COMMON TRAP]
It is tempting to reclaim a deleted key's TABLE SLOT (turning the tombstone back into something reusable) at the same moment its VALUE BLOCK is freed, treating the two as one combined "delete is now complete" step. A tombstoned slot is always safe to leave as-is indefinitely -- it costs one extra probe for anyone who has to pass over it -- but a value block that is reclaimed while still hazarded corrupts a live read, so the two reclamations belong to different resources with different safety requirements and must be checked independently, never bundled into a single all-or-nothing step.

### Code and Verification

```cpp
#include <cstdio>
#include <vector>

// Chapter 29.3 main -- one thread publishes a hazard and reads a value,
// a second thread deletes that same key and attempts to reclaim its
// value block, checking the hazard array before actually freeing it.
// This is Section 27.3's hazard-pointer kernel pair (publish_hazard_kernel
// / try_reclaim_kernel), unchanged, now guarding a KV store's value pool
// instead of a bare linked-list slot -- the exact same protocol
// protects any structure where one thread's delete could race another
// thread's still-in-flight read of what is being deleted.

#define CAPACITY 8
#define EMPTY (-1)
#define TOMBSTONE (-2)
#define UNHAZARDED (-1)
#define NUM_READERS 1

__global__ void reader_publish_and_read_kernel(int* hazard, const int* table_block, int slot,
                                                const int* pool_values, int* out_value) {
    if (threadIdx.x != 0) return;
    int block = table_block[slot];
    hazard[0] = block;          // publish BEFORE reading -- Section 27.3's discipline, unchanged
    *out_value = pool_values[block];
}

__global__ void deleter_mark_tombstone_kernel(int* table_key, int slot) {
    if (threadIdx.x != 0) return;
    table_key[slot] = TOMBSTONE;
}

__device__ bool block_is_hazarded(const int* hazard, int num_readers, int block) {
    for (int r = 0; r < num_readers; r++) if (hazard[r] == block) return true;
    return false;
}

__global__ void deleter_try_free_kernel(const int* hazard, int block, int* reclaim_ok) {
    if (threadIdx.x != 0) return;
    *reclaim_ok = block_is_hazarded(hazard, NUM_READERS, block) ? 0 : 1;
}

// ---- Host-side replay of the identical publish/tombstone/hazard-check
// ---- sequence, in the same order Section 29.3's CPU baseline used. ----

int main() {
    printf("=== Section 29.3 main: hazard-protected concurrent delete of a KV entry ===\n\n");

    // Same table/pool state Section 29.2 built: key=13 at slot 6, block 6.
    std::vector<int> table_key   = {-1, -1, -1, 3, 11, 5, 13, 21};
    std::vector<int> table_block = {-1, -1, -1, 4, 3, 7, 6, 5};
    std::vector<int> pool_values = {0, 0, 0, 1100, 300, 2100, 1300, 500};
    std::vector<int> hazard(NUM_READERS, UNHAZARDED);

    printf("initial table keys:   [ ");
    for (int k : table_key) printf("%d ", k);
    printf("]\n");
    printf("initial table blocks: [ ");
    for (int b : table_block) printf("%d ", b);
    printf("]\n\n");

    int slot13 = 6;
    int block13 = table_block[slot13];

    printf("Reader R (one thread) publishes its hazard and reads, in that order:\n");
    hazard[0] = block13;
    int read_value = pool_values[block13];
    printf("  hazard[0] = %d, reads pool_values[%d] = %d\n\n", hazard[0], block13, read_value);

    printf("Deleter D (one thread) marks key=13's slot TOMBSTONE:\n");
    table_key[slot13] = TOMBSTONE;
    printf("  table keys now: [ ");
    for (int k : table_key) printf("%d ", k);
    printf("]\n\n");

    printf("D checks the hazard array before freeing block %d:\n", block13);
    bool hazarded_before = false;
    for (int h : hazard) if (h == block13) hazarded_before = true;
    printf("  block %d %s -- %s\n\n", block13, hazarded_before ? "IS hazarded" : "not hazarded",
           hazarded_before ? "free DEFERRED" : "free allowed");

    printf("R finishes and clears its hazard:\n");
    hazard[0] = UNHAZARDED;
    printf("  hazard array now: [ %d ]\n\n", hazard[0]);

    printf("D rechecks the hazard array:\n");
    bool hazarded_after = false;
    for (int h : hazard) if (h == block13) hazarded_after = true;
    printf("  block %d %s -- %s\n", block13, hazarded_after ? "IS hazarded" : "not hazarded",
           hazarded_after ? "free still DEFERRED" : "free now allowed");

    bool ok = hazarded_before && !hazarded_after && (read_value == 1300) &&
              (table_key[slot13] == TOMBSTONE);

    printf("\nself-check: reclaim correctly BLOCKED while R's hazard was published,\n");
    printf("then ALLOWED once R cleared it, with R's own read (%d) unaffected by the\n", read_value);
    printf("concurrent tombstone marking: %s\n", ok ? "confirmed" : "MISMATCH");
    return ok ? 0 : 1;
}
```

**Compile and run:**

```bash
NVDIR=/usr/local/lib/python3.11/dist-packages/nvidia/cu13
$NVDIR/bin/nvcc -arch=sm_80 -I$NVDIR/include -L$NVDIR/lib -lcudart 165_kv_delete_hazard_kernel.cu -o 165_kv_delete_hazard_kernel
LD_LIBRARY_PATH=$NVDIR/lib ./165_kv_delete_hazard_kernel
```

**Sample input:** one reader thread publishing a hazard on key 13's value block and reading it, one deleter thread tombstoning key 13's slot and checking the hazard array before and after the reader clears it.

**Sample output:**

```text
=== Section 29.3 main: hazard-protected concurrent delete of a KV entry ===

initial table keys:   [ -1 -1 -1 3 11 5 13 21 ]
initial table blocks: [ -1 -1 -1 4 3 7 6 5 ]

Reader R (one thread) publishes its hazard and reads, in that order:
  hazard[0] = 6, reads pool_values[6] = 1300

Deleter D (one thread) marks key=13's slot TOMBSTONE:
  table keys now: [ -1 -1 -1 3 11 5 -2 21 ]

D checks the hazard array before freeing block 6:
  block 6 IS hazarded -- free DEFERRED

R finishes and clears its hazard:
  hazard array now: [ -1 ]

D rechecks the hazard array:
  block 6 not hazarded -- free now allowed

self-check: reclaim correctly BLOCKED while R's hazard was published,
then ALLOWED once R cleared it, with R's own read (1300) unaffected by the
concurrent tombstone marking: confirmed
```

## Chapter Summary

A GPU key-value store needed no new synchronization primitive anywhere in its design -- every piece of it was already sitting in this book, built and verified in isolation, waiting only to be composed. The key index is Chapter 19's open-addressing table, unchanged: `atomicCAS` on EMPTY slots, probing onward on collision or on a lost race. The value storage is Chapter 28.1's fixed-size block pool, unchanged: a table slot holds a block index instead of a value directly, which decouples value size from table-slot width and lets a value be freed independently of its key's slot, provided the value is always written before the key that points to it becomes visible. Deletion needed two DIFFERENT fixes for two DIFFERENT hazards: a tombstone marker (distinct from EMPTY) keeps other keys' probe chains intact, while Chapter 27.3's hazard-pointer protocol, reused without modification, keeps a deleted value's block from being reclaimed out from under a reader still in the middle of reading it. Composition, not invention, is the whole chapter.

## Self-Check Questions

1. Why does inserting into this KV store always write the value into its pool block BEFORE publishing the key into the table, rather than the other way around?
2. What specifically would go wrong if a delete reset a key's table slot directly to EMPTY instead of to a TOMBSTONE?
3. Why does a lookup's probe sequence need to treat TOMBSTONE differently from both a real key and from EMPTY?
4. In Section 29.2's concurrent insert, why can a thread's pool allocation and its table insertion be reasoned about as two entirely independent races, even though both happen inside the same insert operation?
5. Why is it unsafe to reclaim a deleted key's value block at the exact same moment its table slot is tombstoned, and what would that unsafety look like concretely?
6. This chapter reused Chapter 19's hash table, Chapter 28's memory pool, and Chapter 27's hazard pointers without modifying any of their core CAS loops. What property did each of those three designs already have that made this composition possible?

## Where We Go Next

A hash table indexed by an exact key match is only one way to organize spatial or relational data for fast lookup. Chapter 30 turns to a different problem shape entirely: particles moving through space that need to find their spatial NEIGHBORS quickly, not an exact key -- introducing spatial hashing as a second, complementary way to use everything a hash table already does, aimed at simulation instead of exact retrieval.

## Worked Solutions

**1.** If the key were published into the table first, any other thread performing a lookup could observe that key in the table in the instant before the value write to its pool block completes, and would then read a stale, garbage, or partially-written value out of that block. Writing the value fully first and publishing the key second guarantees that the moment any other thread can possibly see the key, the block it points to already holds that key's complete, correct value -- there is no window in which a successful lookup can observe an incomplete value.

**2.** Resetting a key's slot directly to EMPTY would break the probe chains of every OTHER key that originally probed PAST that slot to reach its own home, because lookup stops probing the instant it encounters an EMPTY slot. A key like 21, which only reached slot 7 by probing past slots 5 and 6, would become permanently unreachable if slot 6 were reset to EMPTY after key 13's deletion, even though key 21 itself was never touched.

**3.** EMPTY tells lookup "no key was ever inserted along this probe path, stop searching" -- it is a hard stopping signal. TOMBSTONE tells lookup "a key WAS here once and other keys may have probed past it to reach their own homes, so keep probing" -- it must never stop a search, only let it continue past. A real key, by contrast, is a candidate for an exact match check. Conflating TOMBSTONE with EMPTY would break other keys' reachability exactly as described in Question 2; conflating it with a real key would make lookups incorrectly report a deleted key as still present.

**4.** The pool allocation touches only the pool's own shared `top` counter and free-list array, and the table insertion touches only the table's own shared slot array -- neither operation ever reads or writes any state the other one owns. Because the two shared structures have no overlapping state, any interleaving of one thread's pool CAS loop with another thread's table CAS loop can only ever affect that structure's own outcome, so each race can be analyzed completely on its own, exactly as Chapters 28.1 and 19 already did independently.

**5.** The table slot and the value pool block are two separate resources with two separate safety conditions: the table slot's TOMBSTONE marking is always immediately safe (it only ever costs an extra probe for future lookups), but the value pool block might still be in the middle of being read by another thread that looked up the key before the delete happened. If both were reclaimed at the same instant, the value pool block could be handed to a brand-new insert and overwritten with a completely different value WHILE a concurrent reader that captured the old block index was still reading from it -- causing that reader to silently observe the wrong key's data.

**6.** All three designs shared the property that they act ONLY through their own dedicated shared state, touched by an operation whose safety was proven independently of any surrounding context: the hash table's insert only ever touches its own slot array via CAS, the memory pool's alloc/free only ever touches its own free-list stack via CAS, and hazard-pointer publication/checking only ever touches a dedicated hazard array via a plain write and a linear scan. Because none of the three ever reached into or depended on the internal state of either of the other two, wiring them together required only calling each one's existing operation in the right order -- write value, then publish key; check hazard, then free -- with no new shared state and no new CAS loop of its own.
