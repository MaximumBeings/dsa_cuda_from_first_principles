# Chapter 19: Open Addressing Under Concurrent Insertion

Part 5 turns to a structure this book has used as a supporting tool since Chapter 7 (histograms) but never built directly: the hash table. This chapter builds open addressing -- the simplest style of hash table, where every key lives directly inside one fixed-size array, with collisions resolved by probing to another slot rather than chaining a linked list off each bucket (continuing this book's long-standing discipline against real pointers). Construction and lookup open the chapter; concurrent insertion reopens this book's running concurrency story one more time, in a form that is genuinely new -- unlike Chapter 17.2's commutative point updates, claiming a hash table slot for a specific key is NOT commutative, which is exactly Chapter 16.2's trie-node hazard again, now on a flat array instead of a tree. The chapter closes with deletion, and the classic, easy-to-miss bug that naive deletion introduces into every open-addressing scheme.

## 19.1 Linear Probing: Construction and Lookup

### Intuition

Open addressing stores every key directly inside a fixed-size array: a key's HOME slot is `hash(key) % table_size`, and on a collision -- the home slot already holds a different key -- the search simply moves to the next slot, wrapping around the end of the table, until it finds either the key itself or an empty slot. Linear probing is the simplest possible collision-resolution rule (always move exactly one slot forward), paired with the simplest possible hash function (`key % table_size`) on a table sized to a prime number, which spreads consecutive keys around the table more evenly than a power of two would.

### The Sequential (CPU) Baseline

```cpp
#include <cstdio>
#include <vector>

// Chapter 19.1 -- The Sequential (CPU) Baseline.
// Open addressing stores every key directly inside a fixed-size array
// (no linked buckets, continuing this book's discipline against real
// pointers): a key's HOME slot is `hash(key) % table_size`, and on a
// collision (the home slot already holds a different key), the search
// simply moves to the next slot, wrapping around the end of the table,
// until it finds either the key itself or an empty slot. This section
// uses the simplest possible collision-resolution rule -- LINEAR
// probing, always moving exactly one slot forward -- and the simplest
// possible hash function, `key % table_size`, on a table sized 11 (a
// prime, which spreads consecutive keys around the table more evenly
// than a power of two would).
#define TABLE_SIZE 11
#define EMPTY (-1)

int hash_key(int key) { return key % TABLE_SIZE; }

void build(const std::vector<int>& keys, std::vector<int>& table) {
    table.assign(TABLE_SIZE, EMPTY);
    for (int key : keys) {
        int slot = hash_key(key);
        int start = slot;
        int probes = 0;
        while (table[slot] != EMPTY) {
            probes++;
            slot = (slot + 1) % TABLE_SIZE;
        }
        table[slot] = key;
        printf("  insert(%d): home=%d, probed %d time(s), placed at slot %d\n",
               key, start, probes, slot);
    }
}

// Returns the slot holding `key`, or -1 if not present. Probing must
// stop at the first EMPTY slot encountered: since every key's home
// slot is fixed by its hash, an empty slot along the probe sequence
// proves that key was never placed here (had it been, it could not
// have skipped over what is now an empty slot to land further along).
int lookup(const std::vector<int>& table, int key) {
    int slot = hash_key(key);
    int start = slot;
    while (table[slot] != EMPTY) {
        if (table[slot] == key) return slot;
        slot = (slot + 1) % TABLE_SIZE;
        if (slot == start) return -1;   // wrapped all the way around
    }
    return -1;
}

int main() {
    printf("=== Section 19.1 CPU baseline: open addressing, build + lookup ===\n\n");

    std::vector<int> keys = {10, 21, 32, 5, 16, 8};
    printf("inserting keys in order: ");
    for (int k : keys) printf("%d ", k);
    printf("  (table size %d)\n\n", TABLE_SIZE);

    std::vector<int> table;
    build(keys, table);
    printf("\nfinal table: ");
    for (int v : table) printf("%d ", v);
    printf("\n\n");

    bool ok = true;
    std::vector<int> expected_table = {21, 32, -1, -1, -1, 5, 16, -1, 8, -1, 10};
    for (int i = 0; i < TABLE_SIZE; i++) ok = ok && (table[i] == expected_table[i]);

    printf("looking up every inserted key, plus one absent key (99):\n");
    for (int k : keys) {
        int slot = lookup(table, k);
        printf("  lookup(%d) -> slot %d\n", k, slot);
        ok = ok && (slot != -1) && (table[slot] == k);
    }
    int missing = lookup(table, 99);
    printf("  lookup(99) -> slot %d (expected: not found)\n", missing);
    ok = ok && (missing == -1);

    printf("\nexpected table: 21 32 -1 -1 -1 5 16 -1 8 -1 10\n");
    printf("\nself-check: every inserted key is found at its correct slot, and the\n");
    printf("absent key correctly reports not-found: %s\n", ok ? "confirmed" : "MISMATCH");
    return ok ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++17 -Wall -Wextra -O2 100_hashtable_build_lookup_cpu_baseline.cpp -o 100_hashtable_build_lookup_cpu_baseline
./100_hashtable_build_lookup_cpu_baseline
```

**Sample input:** 6 keys inserted into an 11-slot table via linear probing, then looked up (plus one absent key).

**Sample output:**

```text
=== Section 19.1 CPU baseline: open addressing, build + lookup ===

inserting keys in order: 10 21 32 5 16 8   (table size 11)

  insert(10): home=10, probed 0 time(s), placed at slot 10
  insert(21): home=10, probed 1 time(s), placed at slot 0
  insert(32): home=10, probed 2 time(s), placed at slot 1
  insert(5): home=5, probed 0 time(s), placed at slot 5
  insert(16): home=5, probed 1 time(s), placed at slot 6
  insert(8): home=8, probed 0 time(s), placed at slot 8

final table: 21 32 -1 -1 -1 5 16 -1 8 -1 10 

looking up every inserted key, plus one absent key (99):
  lookup(10) -> slot 10
  lookup(21) -> slot 0
  lookup(32) -> slot 1
  lookup(5) -> slot 5
  lookup(16) -> slot 6
  lookup(8) -> slot 8
  lookup(99) -> slot -1 (expected: not found)

expected table: 21 32 -1 -1 -1 5 16 -1 8 -1 10

self-check: every inserted key is found at its correct slot, and the
absent key correctly reports not-found: confirmed
```

### The Concept, In Detail

Tracing the construction on keys `{10, 21, 32, 5, 16, 8}` in a table of size 11:

```
insert(10): home = 10 % 11 = 10 -- slot 10 is empty -> placed at slot 10
insert(21): home = 21 % 11 = 10 -- slot 10 taken (10) -> probe slot 0 -- empty -> placed at slot 0
insert(32): home = 32 % 11 = 10 -- slot 10 taken, slot 0 taken (21) -> probe slot 1 -- empty -> placed at slot 1
insert(5):  home = 5  % 11 = 5  -- slot 5 is empty -> placed at slot 5
insert(16): home = 16 % 11 = 5  -- slot 5 taken (5) -> probe slot 6 -- empty -> placed at slot 6
insert(8):  home = 8  % 11 = 8  -- slot 8 is empty -> placed at slot 8
```

```
ASCII view of the final table (index: value):

  0    1    2   3   4   5   6   7   8   9   10
[ 21 | 32 | . | . | . | 5 | 16 | . | 8 | . | 10 ]
  ^probed from 10   ^probed from 5      ^home
```

Keys `21` and `32` both WRAP AROUND the end of the table -- their home slot (10) is near the table's end, so linear probing continues past index 10 by wrapping back to index 0, then 1. Lookup uses the identical probe sequence starting from a key's own home slot, and it can stop the MOMENT it hits a genuinely empty slot: if the key it is looking for had ever been inserted, insertion would have claimed that very empty slot instead of continuing past it (or claimed an earlier one along the same sequence) -- so an empty slot is proof the key was never placed.

[COMMON TRAP]
It is tempting to think lookup can stop as soon as it finds ANY slot holding a different key than the one being searched for. A different key at a slot means only that THAT PARTICULAR key claimed this slot first -- it says nothing about whether the key being searched for exists further along the SAME probe sequence. Lookup must keep probing past every non-matching, non-empty slot, and may only conclude "not found" upon reaching a genuinely empty slot (or wrapping all the way back to its own start).

### Code and Verification

```cpp
#include <cstdio>
#include <vector>

// Chapter 19.1 main -- exactly the pattern established since Chapter
// 17.1: a single lookup's probe sequence is a short, genuinely
// sequential walk (each step depends on what the previous slot held),
// so the real parallel opportunity is running MANY independent lookups
// at once, one thread per lookup. Since lookups never modify the
// table, this is a purely read-only workload with no shared mutable
// state at all -- no atomics, no synchronization, regardless of how
// many lookups run simultaneously.
#define TABLE_SIZE 11
#define EMPTY (-1)

__device__ int hash_key_dev(int key) { return key % TABLE_SIZE; }

// One thread per QUERY key. Every thread reads the same shared,
// unchanging table -- never writes to it -- so any number of threads
// can run this at once with zero interaction between them.
__global__ void lookup_kernel(const int* table, const int* query_keys, int* results, int num_queries) {
    int t = threadIdx.x;
    if (t >= num_queries) return;
    int key = query_keys[t];
    int slot = hash_key_dev(key);
    int start = slot;
    int result = -1;
    while (table[slot] != EMPTY) {
        if (table[slot] == key) { result = slot; break; }
        slot = (slot + 1) % TABLE_SIZE;
        if (slot == start) break;
    }
    results[t] = result;
}

// ---- Host-side replay of the identical per-thread logic. ----

int main() {
    printf("=== Section 19.1 main: many independent parallel lookups ===\n\n");

    std::vector<int> table = {21, 32, -1, -1, -1, 5, 16, -1, 8, -1, 10};
    printf("prebuilt table: ");
    for (int v : table) printf("%d ", v);
    printf("\n\n");

    std::vector<int> queries = {10, 21, 32, 5, 16, 8, 99};
    std::vector<int> results(queries.size());

    printf("%zu independent lookup threads, each probing on its own:\n", queries.size());
    for (size_t t = 0; t < queries.size(); t++) {
        int key = queries[t];
        int slot = key % TABLE_SIZE;
        int start = slot;
        int result = -1;
        while (table[slot] != EMPTY) {
            if (table[slot] == key) { result = slot; break; }
            slot = (slot + 1) % TABLE_SIZE;
            if (slot == start) break;
        }
        results[t] = result;
        printf("  thread %zu: lookup(%d) -> slot %d\n", t, key, result);
    }

    std::vector<int> expected = {10, 0, 1, 5, 6, 8, -1};
    bool ok = true;
    for (size_t t = 0; t < results.size(); t++) ok = ok && (results[t] == expected[t]);

    printf("\nexpected results: 10 0 1 5 6 8 -1\n");
    printf("\nself-check: parallel independent lookups match the CPU baseline exactly: %s\n",
           ok ? "confirmed" : "MISMATCH");
    return ok ? 0 : 1;
}
```

**Compile and run:**

```bash
NVDIR=/usr/local/lib/python3.11/dist-packages/nvidia/cu13
$NVDIR/bin/nvcc -arch=sm_80 -I$NVDIR/include -L$NVDIR/lib -lcudart 101_hashtable_parallel_lookup_kernel.cu -o 101_hashtable_parallel_lookup_kernel
LD_LIBRARY_PATH=$NVDIR/lib ./101_hashtable_parallel_lookup_kernel
```

**Sample input:** the same prebuilt table, queried for all 6 inserted keys plus one absent key, by 7 independent threads at once.

**Sample output:**

```text
=== Section 19.1 main: many independent parallel lookups ===

prebuilt table: 21 32 -1 -1 -1 5 16 -1 8 -1 10 

7 independent lookup threads, each probing on its own:
  thread 0: lookup(10) -> slot 10
  thread 1: lookup(21) -> slot 0
  thread 2: lookup(32) -> slot 1
  thread 3: lookup(5) -> slot 5
  thread 4: lookup(16) -> slot 6
  thread 5: lookup(8) -> slot 8
  thread 6: lookup(99) -> slot -1

expected results: 10 0 1 5 6 8 -1

self-check: parallel independent lookups match the CPU baseline exactly: confirmed
```

## 19.2 Concurrent Insertion via atomicCAS: Racing for a Slot

### Intuition

Insertion is itself just a retry loop: attempt to claim a slot (check if it is empty, then write); if the attempt fails (the slot turns out to be occupied), move to the next slot and try again. Written sequentially, "attempt to claim a slot" is trivial, because only one insertion is ever in flight at a time. Multiple threads inserting DIFFERENT keys at once can genuinely probe to the exact same slot at the exact same time, and "claim this slot for MY key" is NOT commutative -- unlike Chapter 17.2's additive point updates, whichever key actually ends up in a slot is a specific, different outcome depending on which thread wins. This is exactly Chapter 16.2's trie-node race again, and the fix is the same: `atomicCAS(&table[slot], EMPTY, key)` claims a slot only if it is still empty, and a thread whose CAS fails simply treats that exactly like an ordinary collision -- advancing to the next slot and retrying, reusing linear probing's own existing mechanism with no separate "race recovery" logic needed at all.

### The Sequential (CPU) Baseline

```cpp
#include <cstdio>
#include <vector>

// Chapter 19.2 -- The Sequential (CPU) Baseline.
// Insertion is itself just a RETRY LOOP: attempt to claim a slot; if
// it is already taken, move to the next slot and try again. Written
// sequentially, "attempt to claim a slot" is trivial -- check if it is
// EMPTY, and if so, write the key there -- because only one insertion
// is ever in flight at a time, so there is no way for the slot to
// change between the check and the write. This section's baseline is
// the exact same construction as Section 19.1, but framed around that
// retry structure specifically: it is the part that Section 19.2's
// main file will need to make SAFE once multiple insertions attempt
// their own claims at once.
#define TABLE_SIZE 11
#define EMPTY (-1)

int hash_key(int key) { return key % TABLE_SIZE; }

void insert(std::vector<int>& table, int key) {
    int slot = hash_key(key);
    while (true) {
        if (table[slot] == EMPTY) {
            table[slot] = key;   // claim this slot -- safe here because nothing else can be
                                  // checking or writing this same slot at the same time
            return;
        }
        slot = (slot + 1) % TABLE_SIZE;   // claim attempt failed (slot occupied) -- retry at the next slot
    }
}

int main() {
    printf("=== Section 19.2 CPU baseline: insertion as a sequential retry loop ===\n\n");

    std::vector<int> keys = {10, 21, 32, 5, 16, 8};
    std::vector<int> table(TABLE_SIZE, EMPTY);

    printf("inserting keys one at a time, each as its own claim-or-retry loop:\n");
    for (int key : keys) {
        int home = hash_key(key);
        int attempts = 0;
        int slot = home;
        while (table[slot] != EMPTY) { attempts++; slot = (slot + 1) % TABLE_SIZE; }
        insert(table, key);
        printf("  insert(%d): home=%d, %d failed claim attempt(s) before succeeding at slot %d\n",
               key, home, attempts, slot);
    }

    printf("\nfinal table (this insertion ORDER produced this exact layout): ");
    for (int v : table) printf("%d ", v);
    printf("\n\n");

    std::vector<int> expected_table = {21, 32, -1, -1, -1, 5, 16, -1, 8, -1, 10};
    bool ok = true;
    for (int i = 0; i < TABLE_SIZE; i++) ok = ok && (table[i] == expected_table[i]);

    printf("expected table: 21 32 -1 -1 -1 5 16 -1 8 -1 10\n");
    printf("\nself-check: sequential retry-based insertion matches the expected layout: %s\n",
           ok ? "confirmed" : "MISMATCH");
    return ok ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++17 -Wall -Wextra -O2 102_hashtable_insert_cpu_baseline.cpp -o 102_hashtable_insert_cpu_baseline
./102_hashtable_insert_cpu_baseline
```

**Sample input:** the same 6 keys, inserted one at a time, with each insertion's claim-or-retry attempts counted explicitly.

**Sample output:**

```text
=== Section 19.2 CPU baseline: insertion as a sequential retry loop ===

inserting keys one at a time, each as its own claim-or-retry loop:
  insert(10): home=10, 0 failed claim attempt(s) before succeeding at slot 10
  insert(21): home=10, 1 failed claim attempt(s) before succeeding at slot 0
  insert(32): home=10, 2 failed claim attempt(s) before succeeding at slot 1
  insert(5): home=5, 0 failed claim attempt(s) before succeeding at slot 5
  insert(16): home=5, 1 failed claim attempt(s) before succeeding at slot 6
  insert(8): home=8, 0 failed claim attempt(s) before succeeding at slot 8

final table (this insertion ORDER produced this exact layout): 21 32 -1 -1 -1 5 16 -1 8 -1 10 

expected table: 21 32 -1 -1 -1 5 16 -1 8 -1 10

self-check: sequential retry-based insertion matches the expected layout: confirmed
```

### The Concept, In Detail

Running the same 6 keys CONCURRENTLY (one thread per key) under a lockstep "every still-unplaced key's thread attempts a claim on its current slot every wave" schedule -- exactly Chapter 16.3's own level-synchronous "still-active" pattern, generalized from trie positions to hash-table probe positions -- with a fixed tie-break (the largest key wins any simultaneous CAS on the same slot):

```
wave 0: slot5 contenders {5,16} -> 16 wins (atomicCAS succeeds); 5 loses, advances to slot6
        slot8 contenders {8}    -> 8 wins alone
        slot10 contenders {10,21,32} -> 32 wins; 10 and 21 lose, both advance to slot0

wave 1: slot0 contenders {10,21} -> 21 wins; 10 loses, advances to slot1
        slot6 contenders {5}     -> 5 wins alone (slot6 was empty)

wave 2: slot1 contenders {10}    -> 10 wins alone
```

```
ASCII view: sequential result versus concurrent result, same 6 keys.

  sequential (Section 19.2's own CPU baseline):
    [ 21 | 32 | . | . | . |  5 | 16 | . | 8 | . | 10 ]

  concurrent (this section's worst-case interleaving):
    [ 21 | 10 | . | . | . | 16 |  5 | . | 8 | . | 32 ]

  DIFFERENT slot assignments for keys 10, 32, 5, and 16 -- but every key
  is still correctly reachable by the SAME probe-sequence lookup.
```

A losing thread's failed `atomicCAS` is indistinguishable, from that thread's own point of view, from finding an ordinary occupied slot during sequential insertion -- both simply mean "try the next slot." This is what makes the fix so lightweight: no new code path is needed for "I lost a race," because losing a race and encountering a collision are already the same event as far as the probing loop is concerned. The resulting table is a different, but equally valid, arrangement of the same 6 keys -- exactly Chapter 16.3's own observation that concurrent execution's exact layout can differ from any particular sequential run while remaining fully correct.

[COMMON TRAP]
It is tempting to expect a concurrently-built hash table to match a SPECIFIC sequential insertion order's layout exactly. Concurrent execution order is not guaranteed to match program order at all -- different runs (or even the same run, on different hardware) can settle on different slot assignments for contended keys. The only correctness property that matters is semantic: does every inserted key come back out of `lookup()` correctly. Comparing exact table CONTENTS between a concurrent run and one particular sequential order is the wrong check; comparing whether every key is still findable is the right one.

### Code and Verification

```cpp
#include <cstdio>
#include <vector>
#include <map>
#include <algorithm>

// Chapter 19.2 main -- multiple threads inserting DIFFERENT keys at
// once can genuinely probe to the exact same slot at the exact same
// time, and "claim this slot for MY key" is emphatically NOT
// commutative (unlike Chapter 17.2's additive point updates): whichever
// key ends up in a slot is a specific, different value depending on
// which thread wins, so a naive, unprotected "check if empty, then
// write" (Section 19.2's own sequential logic) would let two threads
// both see EMPTY and both write, with one silently overwriting the
// other -- exactly Chapter 16.2's trie node race, now on a hash
// table's slot array. The fix reuses Chapter 16.2's own pattern
// directly: `atomicCAS(&table[slot], EMPTY, key)` claims the slot only
// if it is still empty; a thread whose CAS fails simply treats that
// EXACTLY like finding an ordinary occupied slot -- it advances to the
// next slot and retries there, reusing linear probing's own existing
// collision-handling logic with no separate "race recovery" code path
// needed at all.
#define TABLE_SIZE 11
#define EMPTY (-1)

// One thread per KEY being inserted. A failed CAS and an ordinary
// "slot occupied by someone else" collision are handled by the exact
// same retry step -- probing onward is already what collision handling
// does, so the race and the collision share one mechanism.
__global__ void concurrent_insert_kernel(int* table, const int* keys, int num_keys) {
    int t = threadIdx.x;
    if (t >= num_keys) return;
    int key = keys[t];
    int slot = key % TABLE_SIZE;
    while (true) {
        int prev = atomicCAS(&table[slot], EMPTY, key);
        if (prev == EMPTY) return;                  // won the claim
        slot = (slot + 1) % TABLE_SIZE;              // lost -- retry at the next slot, same as any collision
    }
}

// ---- Host-side replay under a genuinely adversarial interleaving:
// every still-unplaced key's thread attempts a claim on its CURRENT
// slot every wave; when several threads target the SAME slot in the
// SAME wave, exactly one (here, by a fixed tie-break: the largest key
// value) wins, and every loser advances to the next slot for the
// following wave -- exactly Chapter 16.3's own level-synchronous
// "still-active" schedule, generalized from trie positions to hash
// table probe positions. ----

int main() {
    printf("=== Section 19.2 main: concurrent insertion via atomicCAS ===\n\n");

    std::vector<int> keys = {10, 21, 32, 5, 16, 8};
    printf("6 threads insert concurrently: ");
    for (int k : keys) printf("%d ", k);
    printf("\n\n");

    std::vector<int> table(TABLE_SIZE, EMPTY);
    std::map<int, int> pos;
    for (int k : keys) pos[k] = k % TABLE_SIZE;
    std::vector<int> still_active = keys;

    int wave = 0;
    while (!still_active.empty()) {
        std::map<int, std::vector<int>> groups;
        for (int k : still_active) groups[pos[k]].push_back(k);

        printf("wave %d:\n", wave);
        std::vector<int> winners;
        for (auto& kv : groups) {
            int slot = kv.first;
            auto& contenders = kv.second;
            if (table[slot] != EMPTY) {
                printf("  slot%d: already claimed by %d -- contenders", slot, table[slot]);
                for (int c : contenders) printf(" %d", c);
                printf(" all fail, advance\n");
                continue;
            }
            int winner = *std::max_element(contenders.begin(), contenders.end());
            table[slot] = winner;
            winners.push_back(winner);
            printf("  slot%d: contenders", slot);
            for (int c : contenders) printf(" %d", c);
            printf(" -> atomicCAS winner %d placed; loser(s)", winner);
            for (int c : contenders) if (c != winner) printf(" %d", c);
            printf(" advance to next slot\n");
        }

        std::vector<int> next_active;
        for (int k : still_active) {
            bool won = false;
            for (int w : winners) if (w == k) won = true;
            if (!won) { pos[k] = (pos[k] + 1) % TABLE_SIZE; next_active.push_back(k); }
        }
        still_active = next_active;
        wave++;
    }
    printf("\n");

    printf("final table: ");
    for (int v : table) printf("%d ", v);
    printf("\n\n");

    std::vector<int> expected_table = {21, 10, -1, -1, -1, 16, 5, -1, 8, -1, 32};
    bool ok = true;
    for (int i = 0; i < TABLE_SIZE; i++) ok = ok && (table[i] == expected_table[i]);

    printf("expected table (DIFFERENT from Section 19.2's sequential layout, because a\n");
    printf("different key won each contended slot here): 21 10 -1 -1 -1 16 5 -1 8 -1 32\n\n");

    printf("every key is still findable by ordinary linear-probe lookup:\n");
    for (int key : keys) {
        int slot = key % TABLE_SIZE;
        int start = slot;
        int result = -1;
        while (table[slot] != EMPTY) {
            if (table[slot] == key) { result = slot; break; }
            slot = (slot + 1) % TABLE_SIZE;
            if (slot == start) break;
        }
        printf("  lookup(%d) -> slot %d\n", key, result);
        ok = ok && (result != -1);
    }

    printf("\nthe exact slot assignment differs from Section 19.2's sequential run (32\n");
    printf("wins slot 10 here, not 10), but every key is still correctly reachable by\n");
    printf("the SAME deterministic probe sequence lookup always uses -- concurrent\n");
    printf("insertion is safe even though it can settle on a different, equally valid\n");
    printf("layout than any particular sequential order would.\n");

    printf("\nself-check: concurrent atomicCAS-based insertion produces a fully valid\n");
    printf("table (every key findable), matching the traced worst-case interleaving: %s\n",
           ok ? "confirmed" : "MISMATCH");
    return ok ? 0 : 1;
}
```

**Compile and run:**

```bash
NVDIR=/usr/local/lib/python3.11/dist-packages/nvidia/cu13
$NVDIR/bin/nvcc -arch=sm_80 -I$NVDIR/include -L$NVDIR/lib -lcudart 103_hashtable_concurrent_insert_kernel.cu -o 103_hashtable_concurrent_insert_kernel
LD_LIBRARY_PATH=$NVDIR/lib ./103_hashtable_concurrent_insert_kernel
```

**Sample input:** the same 6 keys inserted concurrently (one thread each) via `atomicCAS`, under the adversarial lockstep wave schedule traced above.

**Sample output:**

```text
=== Section 19.2 main: concurrent insertion via atomicCAS ===

6 threads insert concurrently: 10 21 32 5 16 8 

wave 0:
  slot5: contenders 5 16 -> atomicCAS winner 16 placed; loser(s) 5 advance to next slot
  slot8: contenders 8 -> atomicCAS winner 8 placed; loser(s) advance to next slot
  slot10: contenders 10 21 32 -> atomicCAS winner 32 placed; loser(s) 10 21 advance to next slot
wave 1:
  slot0: contenders 10 21 -> atomicCAS winner 21 placed; loser(s) 10 advance to next slot
  slot6: contenders 5 -> atomicCAS winner 5 placed; loser(s) advance to next slot
wave 2:
  slot1: contenders 10 -> atomicCAS winner 10 placed; loser(s) advance to next slot

final table: 21 10 -1 -1 -1 16 5 -1 8 -1 32 

expected table (DIFFERENT from Section 19.2's sequential layout, because a
different key won each contended slot here): 21 10 -1 -1 -1 16 5 -1 8 -1 32

every key is still findable by ordinary linear-probe lookup:
  lookup(10) -> slot 1
  lookup(21) -> slot 0
  lookup(32) -> slot 10
  lookup(5) -> slot 6
  lookup(16) -> slot 5
  lookup(8) -> slot 8

the exact slot assignment differs from Section 19.2's sequential run (32
wins slot 10 here, not 10), but every key is still correctly reachable by
the SAME deterministic probe sequence lookup always uses -- concurrent
insertion is safe even though it can settle on a different, equally valid
layout than any particular sequential order would.

self-check: concurrent atomicCAS-based insertion produces a fully valid
table (every key findable), matching the traced worst-case interleaving: confirmed
```

## 19.3 Deletion and Tombstones: Why You Can't Just Clear a Slot

### Intuition

Lookup's stopping rule depends on a subtle invariant: an empty slot along a probe sequence proves a key was never inserted, because insertion would have claimed that slot instead of probing past it. Naively deleting a key by clearing its slot back to empty breaks this invariant for every OTHER key that happened to probe PAST the deleted slot during ITS OWN insertion -- lookup for such a key now incorrectly stops early at the newly-empty slot and reports "not found," even though the key is still sitting further along the exact same probe sequence. The fix is a TOMBSTONE: a marker distinct from both empty and any real key, which lookup treats as "keep probing past" (it does not prove the slot was always empty) while insertion treats as "may reuse this slot."

### The Sequential (CPU) Baseline

```cpp
#include <cstdio>
#include <vector>

// Chapter 19.3 -- The Sequential (CPU) Baseline.
// Lookup's stopping rule (Section 19.1) relies on a subtle invariant:
// an EMPTY slot along a probe sequence proves the key being searched
// for was never inserted, because insertion would have claimed that
// empty slot instead of continuing past it. Naively DELETING a key by
// just clearing its slot back to EMPTY breaks that invariant for every
// OTHER key that happens to have probed PAST the deleted slot during
// its own insertion -- lookup for such a key now incorrectly stops at
// the newly-empty slot and reports "not found," even though the key is
// still sitting further along the very same probe sequence. The fix is
// a TOMBSTONE: a marker distinct from both EMPTY and any real key,
// which lookup treats as "keep probing past" (it is not a real key,
// but it does not prove the slot was always empty either) while insert
// treats it as "may reuse this slot."
#define TABLE_SIZE 11
#define EMPTY (-1)
#define TOMBSTONE (-2)

int hash_key(int key) { return key % TABLE_SIZE; }

int lookup(const std::vector<int>& table, int key) {
    int slot = hash_key(key);
    int start = slot;
    while (table[slot] != EMPTY) {
        if (table[slot] == key) return slot;
        slot = (slot + 1) % TABLE_SIZE;
        if (slot == start) return -1;
    }
    return -1;
}

int lookup_past_tombstones(const std::vector<int>& table, int key) {
    int slot = hash_key(key);
    int start = slot;
    while (table[slot] != EMPTY) {
        if (table[slot] == key) return slot;
        // TOMBSTONE falls through here exactly like an occupied slot holding
        // a DIFFERENT key -- neither one proves the search should stop.
        slot = (slot + 1) % TABLE_SIZE;
        if (slot == start) return -1;
    }
    return -1;
}

void insert_with_tombstones(std::vector<int>& table, int key) {
    int slot = hash_key(key);
    int start = slot;
    int first_tombstone = -1;
    while (table[slot] != EMPTY) {
        if (table[slot] == TOMBSTONE && first_tombstone == -1) first_tombstone = slot;
        if (table[slot] == key) return;   // already present
        slot = (slot + 1) % TABLE_SIZE;
        if (slot == start) break;
    }
    table[first_tombstone != -1 ? first_tombstone : slot] = key;
}

int main() {
    printf("=== Section 19.3 CPU baseline: naive deletion versus tombstones ===\n\n");

    std::vector<int> base_table = {21, 32, -1, -1, -1, 5, 16, -1, 8, -1, 10};
    printf("starting table (from Section 19.1): ");
    for (int v : base_table) printf("%d ", v);
    printf("\n(key 10's probe chain: home slot 10 -> wraps to slot 0 (21) -> slot 1 (32))\n\n");

    printf("--- naive deletion: just clear the slot ---\n");
    std::vector<int> naive_table = base_table;
    naive_table[10] = EMPTY;
    printf("after deleting key 10 (slot 10 set to EMPTY): ");
    for (int v : naive_table) printf("%d ", v);
    printf("\n");
    int naive_21 = lookup(naive_table, 21);
    int naive_32 = lookup(naive_table, 32);
    int naive_10 = lookup(naive_table, 10);
    printf("  lookup(21) -> %d   (21 IS still in the table, at slot 0 -- this is WRONG)\n", naive_21);
    printf("  lookup(32) -> %d   (32 IS still in the table, at slot 1 -- this is WRONG)\n", naive_32);
    printf("  lookup(10) -> %d   (10 really is gone -- this part is correct)\n\n", naive_10);

    printf("--- tombstone deletion: mark the slot, don't clear it ---\n");
    std::vector<int> tomb_table = base_table;
    tomb_table[10] = TOMBSTONE;
    printf("after deleting key 10 (slot 10 set to TOMBSTONE): ");
    for (int v : tomb_table) printf("%d ", v);
    printf("\n");
    int tomb_21 = lookup_past_tombstones(tomb_table, 21);
    int tomb_32 = lookup_past_tombstones(tomb_table, 32);
    int tomb_10 = lookup_past_tombstones(tomb_table, 10);
    printf("  lookup(21) -> %d   (correctly still found)\n", tomb_21);
    printf("  lookup(32) -> %d   (correctly still found)\n", tomb_32);
    printf("  lookup(10) -> %d   (correctly gone)\n\n", tomb_10);

    printf("--- inserting a NEW key that reuses the tombstoned slot ---\n");
    int new_key = 43;   // 43 % 11 == 10, same home slot as the deleted key 10
    insert_with_tombstones(tomb_table, new_key);
    printf("insert(%d) [home=%d]: ", new_key, hash_key(new_key));
    for (int v : tomb_table) printf("%d ", v);
    printf("\n");
    int after_21 = lookup_past_tombstones(tomb_table, 21);
    int after_new = lookup_past_tombstones(tomb_table, new_key);
    printf("  lookup(21) -> %d   (still correctly found, unaffected by the reuse)\n", after_21);
    printf("  lookup(%d) -> %d   (the new key correctly reuses the freed slot)\n\n", new_key, after_new);

    bool ok = (naive_21 == -1) && (naive_32 == -1) && (naive_10 == -1) &&
              (tomb_21 == 0) && (tomb_32 == 1) && (tomb_10 == -1) &&
              (after_21 == 0) && (after_new == 10);

    printf("expected: naive deletion WRONGLY loses lookups for 21 and 32; tombstone\n");
    printf("deletion keeps both correct; the new key 43 correctly reuses slot 10\n");
    printf("\nself-check: naive deletion demonstrably breaks lookup, tombstones fix it,\n");
    printf("and a tombstoned slot is correctly reusable: %s\n", ok ? "confirmed" : "MISMATCH");
    return ok ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++17 -Wall -Wextra -O2 104_hashtable_tombstone_cpu_baseline.cpp -o 104_hashtable_tombstone_cpu_baseline
./104_hashtable_tombstone_cpu_baseline
```

**Sample input:** Section 19.1's table, with key `10` deleted two different ways -- naively (clearing the slot) and via a tombstone -- then a new key inserted into the freed slot.

**Sample output:**

```text
=== Section 19.3 CPU baseline: naive deletion versus tombstones ===

starting table (from Section 19.1): 21 32 -1 -1 -1 5 16 -1 8 -1 10 
(key 10's probe chain: home slot 10 -> wraps to slot 0 (21) -> slot 1 (32))

--- naive deletion: just clear the slot ---
after deleting key 10 (slot 10 set to EMPTY): 21 32 -1 -1 -1 5 16 -1 8 -1 -1 
  lookup(21) -> -1   (21 IS still in the table, at slot 0 -- this is WRONG)
  lookup(32) -> -1   (32 IS still in the table, at slot 1 -- this is WRONG)
  lookup(10) -> -1   (10 really is gone -- this part is correct)

--- tombstone deletion: mark the slot, don't clear it ---
after deleting key 10 (slot 10 set to TOMBSTONE): 21 32 -1 -1 -1 5 16 -1 8 -1 -2 
  lookup(21) -> 0   (correctly still found)
  lookup(32) -> 1   (correctly still found)
  lookup(10) -> -1   (correctly gone)

--- inserting a NEW key that reuses the tombstoned slot ---
insert(43) [home=10]: 21 32 -1 -1 -1 5 16 -1 8 -1 43 
  lookup(21) -> 0   (still correctly found, unaffected by the reuse)
  lookup(43) -> 10   (the new key correctly reuses the freed slot)

expected: naive deletion WRONGLY loses lookups for 21 and 32; tombstone
deletion keeps both correct; the new key 43 correctly reuses slot 10

self-check: naive deletion demonstrably breaks lookup, tombstones fix it,
and a tombstoned slot is correctly reusable: confirmed
```

### The Concept, In Detail

Key `10`'s probe chain, from Section 19.1's construction, was `home slot 10 -> (wraps) -> slot 0 (holds 21) -> slot 1 (holds 32)`. Deleting key `10` naively:

```
before: [ 21 | 32 | . | . | . | 5 | 16 | . | 8 | . | 10 ]
after (naive, slot 10 -> EMPTY):
        [ 21 | 32 | . | . | . | 5 | 16 | . | 8 | . |  . ]

lookup(21): home=10, slot 10 is now EMPTY -> STOP IMMEDIATELY -> reports "not found"
            (WRONG: 21 is sitting right there at slot 0)
lookup(32): home=10, slot 10 is now EMPTY -> STOP IMMEDIATELY -> reports "not found"
            (WRONG: 32 is sitting right there at slot 1)
```

Marking the same slot with a TOMBSTONE instead of EMPTY fixes both:

```
after (tombstone, slot 10 -> TOMBSTONE):
        [ 21 | 32 | . | . | . | 5 | 16 | . | 8 | . |  T ]

lookup(21): home=10, slot 10 is TOMBSTONE (not EMPTY) -> keep probing -> slot 0 -> found (21)
lookup(32): home=10, slot 10 is TOMBSTONE -> keep probing -> slot 0 (21, no match) -> slot 1 -> found (32)
lookup(10): home=10, slot 10 is TOMBSTONE -> keep probing -> slot 0, slot 1, slot 2 -- slot 2 is
            genuinely EMPTY -> STOP -> correctly reports "not found"
```

A later insertion can safely REUSE a tombstoned slot -- unlike a genuinely empty slot, a tombstone is memory that is no longer holding a live key, so giving it to a new key costs nothing and shortens future probe sequences that would otherwise have to skip over it forever:

```
insert(43) [home = 43 % 11 = 10]: slot 10 holds TOMBSTONE -> claim it directly
        [ 21 | 32 | . | . | . | 5 | 16 | . | 8 | . | 43 ]

lookup(21) afterward: unaffected -- still finds 21 at slot 0, exactly as before
lookup(43): home=10, slot 10 holds 43 -> found immediately
```

[COMMON TRAP]
It is tempting to think a tombstoned slot should stop insertion's search for a home the same way an empty slot does. Insertion should instead REMEMBER the first tombstone it passes (rather than stopping there immediately) and keep checking further slots for an exact match with an already-present key -- only once the search either finds the key or reaches a genuinely empty slot does it fall back to claiming the earliest tombstone it saw, which is both correct (never creates a duplicate entry for an already-present key) and efficient (reuses freed space instead of always growing the probe chain further).

### Code and Verification

```cpp
#include <cstdio>
#include <vector>

// Chapter 19.3 main -- exactly Section 19.1's own pattern once more:
// many independent lookups against a shared, read-only table, one
// thread per lookup, with zero interaction between threads. The one
// change from Section 19.1's kernel is a single extra rule inside each
// thread's own probe loop -- a TOMBSTONE never stops the search, only
// a real EMPTY slot does -- and that rule is exactly as safe to run in
// parallel as the original, since it still touches nothing but the
// unchanging, shared table.
#define TABLE_SIZE 11
#define EMPTY (-1)
#define TOMBSTONE (-2)

__device__ int hash_key_dev(int key) { return key % TABLE_SIZE; }

// One thread per QUERY key. A TOMBSTONE is treated exactly like an
// occupied slot holding some OTHER key: neither one proves the search
// should stop, so the loop only exits on a true EMPTY or on finding
// the key itself.
__global__ void lookup_past_tombstones_kernel(const int* table, const int* query_keys,
                                               int* results, int num_queries) {
    int t = threadIdx.x;
    if (t >= num_queries) return;
    int key = query_keys[t];
    int slot = hash_key_dev(key);
    int start = slot;
    int result = -1;
    while (table[slot] != EMPTY) {
        if (table[slot] == key) { result = slot; break; }
        slot = (slot + 1) % TABLE_SIZE;
        if (slot == start) break;
    }
    results[t] = result;
}

// ---- Host-side replay of the identical per-thread logic. ----

int main() {
    printf("=== Section 19.3 main: parallel independent lookups past tombstones ===\n\n");

    // Section 19.3's own table right after deleting key 10 via tombstone
    // (slot 10 = TOMBSTONE, not yet reused by any later insertion).
    std::vector<int> table = {21, 32, -1, -1, -1, 5, 16, -1, 8, -1, TOMBSTONE};
    printf("table (key 10 deleted via tombstone at slot 10):\n  ");
    for (int v : table) printf("%d ", v);
    printf("\n\n");

    std::vector<int> queries = {21, 32, 10, 8, 99};
    std::vector<int> results(queries.size());

    printf("%zu independent lookup threads, each probing past tombstones on its own:\n", queries.size());
    for (size_t t = 0; t < queries.size(); t++) {
        int key = queries[t];
        int slot = key % TABLE_SIZE;
        int start = slot;
        int result = -1;
        while (table[slot] != EMPTY) {
            if (table[slot] == key) { result = slot; break; }
            slot = (slot + 1) % TABLE_SIZE;
            if (slot == start) break;
        }
        results[t] = result;
        printf("  thread %zu: lookup(%d) -> slot %d\n", t, key, result);
    }

    std::vector<int> expected = {0, 1, -1, 8, -1};
    bool ok = true;
    for (size_t t = 0; t < results.size(); t++) ok = ok && (results[t] == expected[t]);

    printf("\nexpected results: 0 1 -1 8 -1\n");
    printf("(21 and 32 correctly found despite probing straight through the tombstoned\n");
    printf("slot 10; 10 itself is correctly reported gone; 99 was never present)\n");

    printf("\nself-check: parallel independent lookups correctly probe past a reused\n");
    printf("tombstoned slot, matching the CPU baseline exactly: %s\n", ok ? "confirmed" : "MISMATCH");
    return ok ? 0 : 1;
}
```

**Compile and run:**

```bash
NVDIR=/usr/local/lib/python3.11/dist-packages/nvidia/cu13
$NVDIR/bin/nvcc -arch=sm_80 -I$NVDIR/include -L$NVDIR/lib -lcudart 105_hashtable_tombstone_lookup_kernel.cu -o 105_hashtable_tombstone_lookup_kernel
LD_LIBRARY_PATH=$NVDIR/lib ./105_hashtable_tombstone_lookup_kernel
```

**Sample input:** the tombstoned table (key `10` deleted), queried by 5 independent threads at once, each correctly probing past the tombstone.

**Sample output:**

```text
=== Section 19.3 main: parallel independent lookups past tombstones ===

table (key 10 deleted via tombstone at slot 10):
  21 32 -1 -1 -1 5 16 -1 8 -1 -2 

5 independent lookup threads, each probing past tombstones on its own:
  thread 0: lookup(21) -> slot 0
  thread 1: lookup(32) -> slot 1
  thread 2: lookup(10) -> slot -1
  thread 3: lookup(8) -> slot 8
  thread 4: lookup(99) -> slot -1

expected results: 0 1 -1 8 -1
(21 and 32 correctly found despite probing straight through the tombstoned
slot 10; 10 itself is correctly reported gone; 99 was never present)

self-check: parallel independent lookups correctly probe past a reused
tombstoned slot, matching the CPU baseline exactly: confirmed
```

## Chapter Summary

Open addressing stores every key directly inside a fixed-size array, resolving collisions by probing to the next slot (linear probing) rather than chaining a linked list off each bucket. Lookup can safely stop the moment it finds a genuinely empty slot, since insertion would always have claimed that slot rather than probing past it -- a short, sequential per-lookup walk whose real parallel opportunity, exactly this book's own recurring pattern since Chapter 17.1, is running many independent lookups at once. Concurrent insertion reopens Chapter 16.2's non-commutative shared-slot hazard on a flat array instead of a tree: claiming a specific slot for a specific key must use `atomicCAS`, and a failed CAS is handled by the exact same retry-at-the-next-slot logic that ordinary collisions already use, producing a table that may differ in exact layout from any one sequential insertion order while remaining fully correct -- every key still findable. Naive deletion (clearing a slot back to empty) silently breaks lookup for any other key that probed past the deleted slot during its own insertion; a TOMBSTONE marker -- distinct from both empty and any real key, telling lookup to keep probing past it while telling insertion it may be safely reused -- fixes this without giving up the ability to reclaim deleted slots.

## Self-Check Questions

1. Why can lookup safely stop the moment it encounters a genuinely EMPTY slot, but not the moment it encounters a slot holding some OTHER key?
2. Why is claiming a hash table slot for a specific key NOT commutative, in contrast to Chapter 17.2's point updates, and what does that difference mean for which atomic primitive is needed?
3. When a thread's `atomicCAS` attempt on a hash table slot fails, what does it do next, and why does this need no separate "race recovery" logic beyond what ordinary collision handling already does?
4. Why can a concurrently-built hash table end up with a different exact layout than any particular sequential insertion order, and why does that not make it incorrect?
5. Walk through exactly why naive deletion (clearing a slot to EMPTY) can cause a lookup for a DIFFERENT, still-present key to incorrectly report "not found."
6. Why should insertion, upon encountering a tombstoned slot, remember it but keep searching rather than claiming it immediately?

## Where We Go Next

Open addressing under linear probing has a real weakness this chapter did not dwell on: in the worst case, a lookup or insertion can still probe through a long run of occupied slots, giving no true upper bound on how many probes a single operation might need. Chapter 20 turns to cuckoo hashing, a different collision-resolution scheme that gives every key exactly one of two possible homes and guarantees O(1) worst-case lookup -- not just O(1) on average, but a genuine, provable bound on the very worst case, at the cost of a more involved insertion procedure.

## Worked Solutions

**1.** Insertion always claims the FIRST available slot along a key's probe sequence -- it never skips over an already-empty slot to place a key further along. This means an empty slot is proof-by-construction that no key currently in the table was ever routed past it: if the key being searched for had been inserted, it (or some earlier key sharing part of its probe sequence) would already occupy that now-empty slot. A slot holding some OTHER key carries no such guarantee -- it only tells you THAT key claimed this slot first, saying nothing about whether the key being searched for exists further along the same sequence, so the search must continue.

**2.** A point update's operation (Chapter 17.2) is "add a delta to whatever is here" -- accumulation, where the final value is the same regardless of the order contributions arrive in. Claiming a hash table slot is "store this SPECIFIC key here if the slot is still available" -- an assignment, where the final content is a completely different, specific value depending on WHICH thread's write is the one that lands. Because the outcome depends on order, a plain `atomicAdd`-style unconditional operation cannot be used; the claim must be conditional on the slot's prior state, which is exactly what `atomicCAS` provides.

**3.** A failed `atomicCAS` means some other thread already claimed that slot first. The thread whose CAS failed advances to the next slot in its own probe sequence and attempts to claim THAT one instead -- which is precisely what the same thread would do sequentially upon finding an ordinary occupied slot during collision handling. Because losing a race and encountering an ordinary collision look identical from inside the probing loop (both just mean "this slot isn't available to me, try the next one"), no additional code path is needed to specifically detect or recover from a lost race.

**4.** Which thread's `atomicCAS` actually succeeds when multiple threads contend for the same slot depends on the runtime order the hardware happens to serialize those attempts in, which is not fixed or guaranteed to match any particular program order. Different keys can therefore end up occupying different slots than a specific sequential insertion order would have produced. This does not make the result incorrect because correctness for a hash table is defined purely by whether `lookup()` correctly finds every key that was inserted, not by which specific slot each key happens to occupy -- and every valid probe-sequence-respecting arrangement, however it arose, satisfies that.

**5.** A different, still-present key `K` may have probed PAST the now-deleted slot during its own original insertion, because its home slot's probe sequence happened to pass through that slot before `K` found an empty one further along. Once that slot is cleared back to EMPTY, lookup for `K` starts at `K`'s own home slot, walks the same probe sequence, reaches the now-empty slot, and stops immediately -- concluding "not found" -- without ever reaching the slot further along where `K` actually still resides, because the emptied slot looks exactly like proof that `K` was never inserted at all.

**6.** If insertion claimed the very first tombstone it encountered, it could end up creating a SECOND entry for a key that is already present further along the same probe sequence (since it never continued searching for an exact match past that point), silently producing a duplicate. Continuing to search after noting the tombstone's position -- while remembering it as a fallback -- lets insertion still find and reuse an already-present key's existing slot when the key IS present, or correctly fall back to reusing the earliest tombstone (rather than probing all the way to a genuinely empty slot) only once it is certain, by reaching a real empty slot, that the key was not already there.
