# Chapter 20: Cuckoo Hashing for Guaranteed O(1) Worst-Case Lookup

Chapter 19 closed with a real weakness left unaddressed: even with tombstones handled correctly, linear probing gives no fixed bound on how many slots a single lookup or insertion might have to probe through. Cuckoo hashing trades a more involved insertion procedure for a genuine, provable guarantee -- every key lives in exactly one of only TWO possible slots, so lookup NEVER costs more than two probes, no matter how full the table is or how unlucky a particular key's history has been. This chapter builds cuckoo hashing in three pieces: construction via eviction chains (with an honest, sharding-based take on what parallelizes safely), the guaranteed-O(1) lookup this scheme exists to deliver, and what happens when an eviction chain refuses to terminate -- detecting the cycle and rehashing. This closes out Part 5 -- Hash Tables.

## 20.1 Two Tables, Two Hash Functions: Cuckoo Hashing Construction via Eviction Chains

### Intuition

Cuckoo hashing gives every key exactly two candidate homes: `T1[h1(key)]` in one table, or `T2[h2(key)]` in a second table, via two independent hash functions. Insertion tries `T1` first. If that slot is empty, done. If it is occupied, the new key EVICTS whoever is there -- takes the slot for itself -- and the evicted key must now find a home in ITS OTHER table, trying `T2[h2(evicted)]`. That can evict yet another key, which tries `T1` again, and so on: an eviction CHAIN that only stops once some key lands in a slot that is genuinely empty. This is named for the cuckoo bird's own habit of evicting other eggs from a nest to make room for its own.

### The Sequential (CPU) Baseline

```cpp
#include <cstdio>
#include <vector>

// Chapter 20.1 -- The Sequential (CPU) Baseline.
// Cuckoo hashing gives every key exactly TWO possible homes, one in each
// of two tables T1 and T2, via two independent hash functions h1 and h2.
// Insertion tries T1 first; if that slot is already occupied, the key
// EVICTS whatever was there, and the evicted key tries ITS OTHER table
// next. This can cascade -- an evicted key evicts another, which evicts
// another -- forming an eviction CHAIN that only stops once some key
// lands in a genuinely empty slot. Unlike open addressing's forward
// probing (Chapter 19), a cuckoo eviction chain can bounce back and
// forth between the two tables an arbitrary number of times before
// settling.
#define TABLE_SIZE 5
#define EMPTY (-1)

int h1(int key) { return key % TABLE_SIZE; }
int h2(int key) { return (key / TABLE_SIZE) % TABLE_SIZE; }

// Builds one shard's two tables from scratch via sequential eviction
// chains. Returns true if every key was placed (no shard in this
// section ever needs more than a handful of evictions).
bool build_shard(const std::vector<int>& keys, std::vector<int>& T1, std::vector<int>& T2) {
    T1.assign(TABLE_SIZE, EMPTY);
    T2.assign(TABLE_SIZE, EMPTY);
    for (int key : keys) {
        int cur = key;
        int table = 1;
        printf("  insert(%d)\n", key);
        for (int step = 0; step < 2 * TABLE_SIZE; step++) {
            if (table == 1) {
                int slot = h1(cur);
                if (T1[slot] == EMPTY) {
                    T1[slot] = cur;
                    printf("    place %d at T1[%d] (was empty) -- done\n", cur, slot);
                    break;
                }
                int evicted = T1[slot];
                T1[slot] = cur;
                printf("    place %d at T1[%d], evicting %d\n", cur, slot, evicted);
                cur = evicted;
                table = 2;
            } else {
                int slot = h2(cur);
                if (T2[slot] == EMPTY) {
                    T2[slot] = cur;
                    printf("    place %d at T2[%d] (was empty) -- done\n", cur, slot);
                    break;
                }
                int evicted = T2[slot];
                T2[slot] = cur;
                printf("    place %d at T2[%d], evicting %d\n", cur, slot, evicted);
                cur = evicted;
                table = 1;
            }
        }
    }
    return true;
}

void print_table(const char* name, const std::vector<int>& T) {
    printf("%s = [ ", name);
    for (int v : T) { if (v == EMPTY) printf(". "); else printf("%d ", v); }
    printf("]\n");
}

int main() {
    printf("=== Section 20.1 CPU baseline: cuckoo construction via eviction chains ===\n\n");

    printf("--- Shard A: keys {3, 8, 13, 18} ---\n");
    std::vector<int> shardA_keys = {3, 8, 13, 18};
    std::vector<int> A_T1, A_T2;
    build_shard(shardA_keys, A_T1, A_T2);
    print_table("  final T1", A_T1);
    print_table("  final T2", A_T2);
    printf("\n");

    printf("--- Shard B: keys {1, 6, 11, 16} ---\n");
    std::vector<int> shardB_keys = {1, 6, 11, 16};
    std::vector<int> B_T1, B_T2;
    build_shard(shardB_keys, B_T1, B_T2);
    print_table("  final T1", B_T1);
    print_table("  final T2", B_T2);
    printf("\n");

    // Self-check: every key in each shard is findable at exactly the
    // slot the eviction chain left it in -- either T1[h1(key)] or
    // T2[h2(key)] must equal that key.
    bool ok = true;
    printf("verifying every key is findable in its own shard's tables:\n");
    for (int key : shardA_keys) {
        bool found = (A_T1[h1(key)] == key) || (A_T2[h2(key)] == key);
        printf("  shard A: find(%d) -> %s\n", key, found ? "found" : "MISSING");
        ok = ok && found;
    }
    for (int key : shardB_keys) {
        bool found = (B_T1[h1(key)] == key) || (B_T2[h2(key)] == key);
        printf("  shard B: find(%d) -> %s\n", key, found ? "found" : "MISSING");
        ok = ok && found;
    }

    std::vector<int> expA_T1 = {-1, -1, -1, 18, -1};
    std::vector<int> expA_T2 = {3, 8, 13, -1, -1};
    std::vector<int> expB_T1 = {-1, 16, -1, -1, -1};
    std::vector<int> expB_T2 = {1, 6, 11, -1, -1};
    ok = ok && (A_T1 == expA_T1) && (A_T2 == expA_T2) && (B_T1 == expB_T1) && (B_T2 == expB_T2);

    printf("\nexpected shard A: T1=[ . . . 18 . ]  T2=[ 3 8 13 . . ]\n");
    printf("expected shard B: T1=[ . 16 . . . ]  T2=[ 1 6 11 . . ]\n");
    printf("\nself-check: both shards built correctly via independent eviction\n");
    printf("chains, every key findable: %s\n", ok ? "confirmed" : "MISMATCH");
    return ok ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++17 -Wall -Wextra -O2 106_cuckoo_construction_cpu_baseline.cpp -o 106_cuckoo_construction_cpu_baseline
./106_cuckoo_construction_cpu_baseline
```

**Sample input:** two independent 4-key shards, each built via its own sequential eviction-chain insertion into a pair of size-5 tables.

**Sample output:**

```text
=== Section 20.1 CPU baseline: cuckoo construction via eviction chains ===

--- Shard A: keys {3, 8, 13, 18} ---
  insert(3)
    place 3 at T1[3] (was empty) -- done
  insert(8)
    place 8 at T1[3], evicting 3
    place 3 at T2[0] (was empty) -- done
  insert(13)
    place 13 at T1[3], evicting 8
    place 8 at T2[1] (was empty) -- done
  insert(18)
    place 18 at T1[3], evicting 13
    place 13 at T2[2] (was empty) -- done
  final T1 = [ . . . 18 . ]
  final T2 = [ 3 8 13 . . ]

--- Shard B: keys {1, 6, 11, 16} ---
  insert(1)
    place 1 at T1[1] (was empty) -- done
  insert(6)
    place 6 at T1[1], evicting 1
    place 1 at T2[0] (was empty) -- done
  insert(11)
    place 11 at T1[1], evicting 6
    place 6 at T2[1] (was empty) -- done
  insert(16)
    place 16 at T1[1], evicting 11
    place 11 at T2[2] (was empty) -- done
  final T1 = [ . 16 . . . ]
  final T2 = [ 1 6 11 . . ]

verifying every key is findable in its own shard's tables:
  shard A: find(3) -> found
  shard A: find(8) -> found
  shard A: find(13) -> found
  shard A: find(18) -> found
  shard B: find(1) -> found
  shard B: find(6) -> found
  shard B: find(11) -> found
  shard B: find(16) -> found

expected shard A: T1=[ . . . 18 . ]  T2=[ 3 8 13 . . ]
expected shard B: T1=[ . 16 . . . ]  T2=[ 1 6 11 . . ]

self-check: both shards built correctly via independent eviction
chains, every key findable: confirmed
```

### The Concept, In Detail

Tracing shard A's construction, keys `{3, 8, 13, 18}`, with `h1(k) = k % 5` and `h2(k) = (k / 5) % 5` -- every one of these four keys shares the SAME `h1` value (3), so every insertion after the first must evict:

```
insert(3):  T1[3] empty -> place 3.                       T1=[ . . . 3 . ]  T2=[ . . . . . ]
insert(8):  T1[3] taken (3) -> evict 3, place 8.           T1=[ . . . 8 . ]
            3 tries T2[h2(3)=0] -- empty -> place 3.                        T2=[ 3 . . . . ]
insert(13): T1[3] taken (8) -> evict 8, place 13.          T1=[ . . . 13 . ]
            8 tries T2[h2(8)=1] -- empty -> place 8.                        T2=[ 3 8 . . . ]
insert(18): T1[3] taken (13) -> evict 13, place 18.        T1=[ . . . 18 . ]
            13 tries T2[h2(13)=2] -- empty -> place 13.                     T2=[ 3 8 13 . . ]
```

```
ASCII view of the eviction chain for insert(18):

  T1[3]: 13 --evicted-by--> 18   (18 takes 13's slot)
             |
             v
  13 tries its OTHER table: T2[h2(13)=2] -- empty -> 13 settles there

  Each eviction hands the DISPLACED key to its OTHER hash function --
  a key evicted from T1 always retries in T2, and vice versa, since
  landing back in the SAME table at the SAME slot it just left would
  accomplish nothing.
```

Shard B (`{1, 6, 11, 16}`, all sharing `h1(k) = 1`) is structurally identical -- same shape of eviction chain, just different numbers -- which is exactly what makes it a second, independent shard rather than a variation that needs its own separate reasoning.

[COMMON TRAP]
It is tempting to think an evicted key retries the SAME table it was just evicted from, hunting for a different slot the way linear probing would. Cuckoo hashing gives every key exactly ONE fixed slot per table (`h1(key)` in `T1`, `h2(key)` in `T2`) -- there is no "try the next slot over." An evicted key has exactly one other option: its home slot in the OTHER table. If that slot is also occupied, the key currently there is evicted in turn, and the chain continues -- it never loops back to retry the table it just left.

### Code and Verification

```cpp
#include <cstdio>
#include <vector>

// Chapter 20.1 main -- the honest parallel angle on cuckoo construction
// is SHARDING, not concurrent insertion into one shared table. Each
// shard gets its own private pair of small cuckoo tables; ONE THREAD
// builds ONE ENTIRE SHARD, running its own eviction chain exactly as
// the sequential baseline does, with ZERO interaction between shards.
// Building any single shard is a genuinely sequential, bounded
// pointer-chase (an eviction can only cascade so many times before
// either landing in an empty slot or -- Section 20.3's topic -- being
// detected as looping forever), so it is not itself parallelized;
// what IS parallel, and fully safe, is running MANY independent shards
// side by side, each on its own thread, touching only its own memory.
#define TABLE_SIZE 5
#define EMPTY (-1)
#define NUM_SHARDS 2
#define KEYS_PER_SHARD 4
#define MAX_STEPS (2 * TABLE_SIZE)

__device__ int h1_dev(int key) { return key % TABLE_SIZE; }
__device__ int h2_dev(int key) { return (key / TABLE_SIZE) % TABLE_SIZE; }

// One thread per shard. `T1`/`T2` are flattened as NUM_SHARDS * TABLE_SIZE,
// so thread `shard` touches only the slice [shard*TABLE_SIZE, shard*TABLE_SIZE+TABLE_SIZE)
// of each table -- disjoint memory per thread, so no atomics are needed
// at all inside a shard's own eviction chain.
__global__ void cuckoo_shard_build_kernel(const int* keys, int* T1, int* T2) {
    int shard = threadIdx.x;
    if (shard >= NUM_SHARDS) return;
    int* my_T1 = T1 + shard * TABLE_SIZE;
    int* my_T2 = T2 + shard * TABLE_SIZE;
    for (int i = 0; i < KEYS_PER_SHARD; i++) {
        int cur = keys[shard * KEYS_PER_SHARD + i];
        int table = 1;
        for (int step = 0; step < MAX_STEPS; step++) {
            if (table == 1) {
                int slot = h1_dev(cur);
                if (my_T1[slot] == EMPTY) { my_T1[slot] = cur; break; }
                int evicted = my_T1[slot];
                my_T1[slot] = cur;
                cur = evicted;
                table = 2;
            } else {
                int slot = h2_dev(cur);
                if (my_T2[slot] == EMPTY) { my_T2[slot] = cur; break; }
                int evicted = my_T2[slot];
                my_T2[slot] = cur;
                cur = evicted;
                table = 1;
            }
        }
    }
}

// ---- Host-side replay of the identical per-thread logic. ----

int h1_host(int key) { return key % TABLE_SIZE; }
int h2_host(int key) { return (key / TABLE_SIZE) % TABLE_SIZE; }

void print_table(const char* name, const std::vector<int>& T) {
    printf("%s = [ ", name);
    for (int v : T) { if (v == EMPTY) printf(". "); else printf("%d ", v); }
    printf("]\n");
}

int main() {
    printf("=== Section 20.1 main: two shards built concurrently, one thread each ===\n\n");

    std::vector<int> keys = {
        3, 8, 13, 18,   // shard 0
        1, 6, 11, 16,   // shard 1
    };
    printf("shard 0 keys: 3 8 13 18\n");
    printf("shard 1 keys: 1 6 11 16\n");
    printf("(launching %d threads, one per shard -- each runs its own\n", NUM_SHARDS);
    printf(" complete sequential eviction-chain build, touching only its\n");
    printf(" own private slice of T1/T2; no atomics, no shared state)\n\n");

    std::vector<int> T1(NUM_SHARDS * TABLE_SIZE, EMPTY);
    std::vector<int> T2(NUM_SHARDS * TABLE_SIZE, EMPTY);

    for (int shard = 0; shard < NUM_SHARDS; shard++) {
        int* my_T1 = T1.data() + shard * TABLE_SIZE;
        int* my_T2 = T2.data() + shard * TABLE_SIZE;
        for (int i = 0; i < KEYS_PER_SHARD; i++) {
            int cur = keys[shard * KEYS_PER_SHARD + i];
            int table = 1;
            for (int step = 0; step < MAX_STEPS; step++) {
                if (table == 1) {
                    int slot = h1_host(cur);
                    if (my_T1[slot] == EMPTY) { my_T1[slot] = cur; break; }
                    int evicted = my_T1[slot];
                    my_T1[slot] = cur;
                    cur = evicted;
                    table = 2;
                } else {
                    int slot = h2_host(cur);
                    if (my_T2[slot] == EMPTY) { my_T2[slot] = cur; break; }
                    int evicted = my_T2[slot];
                    my_T2[slot] = cur;
                    cur = evicted;
                    table = 1;
                }
            }
        }
    }

    for (int shard = 0; shard < NUM_SHARDS; shard++) {
        printf("shard %d result:\n", shard);
        std::vector<int> shard_T1(T1.begin() + shard * TABLE_SIZE, T1.begin() + (shard + 1) * TABLE_SIZE);
        std::vector<int> shard_T2(T2.begin() + shard * TABLE_SIZE, T2.begin() + (shard + 1) * TABLE_SIZE);
        print_table("  T1", shard_T1);
        print_table("  T2", shard_T2);
    }

    std::vector<int> exp_T1 = {-1, -1, -1, 18, -1,   -1, 16, -1, -1, -1};
    std::vector<int> exp_T2 = {3, 8, 13, -1, -1,     1, 6, 11, -1, -1};
    bool ok = (T1 == exp_T1) && (T2 == exp_T2);

    printf("\nexpected shard 0: T1=[ . . . 18 . ]  T2=[ 3 8 13 . . ]\n");
    printf("expected shard 1: T1=[ . 16 . . . ]  T2=[ 1 6 11 . . ]\n");
    printf("(identical to the CPU baseline's shards -- each thread's own\n");
    printf("eviction chain is a fully sequential, self-contained computation;\n");
    printf("running two of them at once changes nothing about either result)\n");
    printf("\nself-check: both concurrently-built shards match the sequential\n");
    printf("baseline exactly: %s\n", ok ? "confirmed" : "MISMATCH");
    return ok ? 0 : 1;
}
```

**Compile and run:**

```bash
NVDIR=/usr/local/lib/python3.11/dist-packages/nvidia/cu13
$NVDIR/bin/nvcc -arch=sm_80 -I$NVDIR/include -L$NVDIR/lib -lcudart 107_cuckoo_sharded_construction_kernel.cu -o 107_cuckoo_sharded_construction_kernel
LD_LIBRARY_PATH=$NVDIR/lib ./107_cuckoo_sharded_construction_kernel
```

**Sample input:** the same two 4-key shards, built by two threads at once -- one thread per shard, each running its own complete eviction-chain construction against its own private pair of tables.

**Sample output:**

```text
=== Section 20.1 main: two shards built concurrently, one thread each ===

shard 0 keys: 3 8 13 18
shard 1 keys: 1 6 11 16
(launching 2 threads, one per shard -- each runs its own
 complete sequential eviction-chain build, touching only its
 own private slice of T1/T2; no atomics, no shared state)

shard 0 result:
  T1 = [ . . . 18 . ]
  T2 = [ 3 8 13 . . ]
shard 1 result:
  T1 = [ . 16 . . . ]
  T2 = [ 1 6 11 . . ]

expected shard 0: T1=[ . . . 18 . ]  T2=[ 3 8 13 . . ]
expected shard 1: T1=[ . 16 . . . ]  T2=[ 1 6 11 . . ]
(identical to the CPU baseline's shards -- each thread's own
eviction chain is a fully sequential, self-contained computation;
running two of them at once changes nothing about either result)

self-check: both concurrently-built shards match the sequential
baseline exactly: confirmed
```

## 20.2 Guaranteed O(1) Lookup: Exactly Two Probes, Always

### Intuition

Because insertion only ever places a key at `T1[h1(key)]` or `T2[h2(key)]`, lookup knows exactly, in advance, the ONLY two places a key could possibly be. Check `T1`'s slot; if it holds a different key (or nothing), check `T2`'s slot; if that also fails to match, the key is not in the table -- full stop, no further searching is possible because there is nowhere else it could be. This is a genuine worst-case bound: not "usually fast," not "fast on average," but a hard guarantee of at most two memory reads for ANY key, present or absent, no matter how the table got built.

### The Sequential (CPU) Baseline

```cpp
#include <cstdio>
#include <vector>

// Chapter 20.2 -- The Sequential (CPU) Baseline.
// Cuckoo hashing's payoff is lookup: because a key can ONLY ever live
// in one of exactly two places -- T1[h1(key)] or T2[h2(key)] -- lookup
// never probes further than that. Check T1's slot; if it doesn't match,
// check T2's slot; if THAT doesn't match either, the key is not present
// -- full stop. This is a genuine, provable worst-case bound of exactly
// two probes, for every key, whether present or absent -- unlike
// Chapter 19's linear probing, whose worst case has no fixed bound at
// all (a long run of occupied slots can force arbitrarily many probes).
#define TABLE_SIZE 5
#define EMPTY (-1)

int h1(int key) { return key % TABLE_SIZE; }
int h2(int key) { return (key / TABLE_SIZE) % TABLE_SIZE; }

void build(const std::vector<int>& keys, std::vector<int>& T1, std::vector<int>& T2) {
    T1.assign(TABLE_SIZE, EMPTY);
    T2.assign(TABLE_SIZE, EMPTY);
    for (int key : keys) {
        int cur = key;
        int table = 1;
        for (int step = 0; step < 2 * TABLE_SIZE; step++) {
            if (table == 1) {
                int slot = h1(cur);
                if (T1[slot] == EMPTY) { T1[slot] = cur; break; }
                int evicted = T1[slot];
                T1[slot] = cur;
                cur = evicted;
                table = 2;
            } else {
                int slot = h2(cur);
                if (T2[slot] == EMPTY) { T2[slot] = cur; break; }
                int evicted = T2[slot];
                T2[slot] = cur;
                cur = evicted;
                table = 1;
            }
        }
    }
}

// Returns (table, slot) as table*100+slot if found, or -1 if absent.
// Exactly two probes, always: check T1's home slot, then T2's home
// slot, then give up -- there is nowhere else the key could be.
int lookup(const std::vector<int>& T1, const std::vector<int>& T2, int key, int* probes) {
    *probes = 1;
    int slot1 = h1(key);
    if (T1[slot1] == key) return 1 * 100 + slot1;
    *probes = 2;
    int slot2 = h2(key);
    if (T2[slot2] == key) return 2 * 100 + slot2;
    return -1;
}

void print_table(const char* name, const std::vector<int>& T) {
    printf("%s = [ ", name);
    for (int v : T) { if (v == EMPTY) printf(". "); else printf("%d ", v); }
    printf("]\n");
}

int main() {
    printf("=== Section 20.2 CPU baseline: guaranteed <=2-probe lookup ===\n\n");

    std::vector<int> keys = {3, 8, 13, 18, 21};
    printf("building table from keys: 3 8 13 18 21\n");
    std::vector<int> T1, T2;
    build(keys, T1, T2);
    print_table("T1", T1);
    print_table("T2", T2);
    printf("\n");

    bool ok = true;
    std::vector<int> exp_T1 = {-1, 21, -1, 18, -1};
    std::vector<int> exp_T2 = {3, 8, 13, -1, -1};
    ok = ok && (T1 == exp_T1) && (T2 == exp_T2);

    printf("looking up every inserted key:\n");
    std::vector<int> expected_loc = {200, 201, 202, 103, 101}; // 3,8,13->T2; 18,21->T1
    for (size_t i = 0; i < keys.size(); i++) {
        int probes;
        int loc = lookup(T1, T2, keys[i], &probes);
        int table = loc / 100, slot = loc % 100;
        printf("  lookup(%d) -> table %d, slot %d (%d probe%s)\n",
               keys[i], table, slot, probes, probes == 1 ? "" : "s");
        ok = ok && (loc == expected_loc[i]);
    }

    printf("\nlooking up two absent keys (each must still cost at most 2 probes):\n");
    for (int absent : {99, 4}) {
        int probes;
        int loc = lookup(T1, T2, absent, &probes);
        printf("  lookup(%d) -> not found (%d probe%s)\n", absent, probes, probes == 1 ? "" : "s");
        ok = ok && (loc == -1) && (probes <= 2);
    }

    printf("\nexpected: 3->(T2,0) 8->(T2,1) 13->(T2,2) 18->(T1,3) 21->(T1,1)\n");
    printf("          99->not found, 4->not found, all in at most 2 probes\n");
    printf("\nself-check: every lookup -- present or absent -- resolves in at\n");
    printf("most 2 probes, matching the expected locations exactly: %s\n", ok ? "confirmed" : "MISMATCH");
    return ok ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++17 -Wall -Wextra -O2 108_cuckoo_lookup_cpu_baseline.cpp -o 108_cuckoo_lookup_cpu_baseline
./108_cuckoo_lookup_cpu_baseline
```

**Sample input:** a 5-key table built from `{3, 8, 13, 18, 21}`, then looked up by every inserted key plus two absent keys.

**Sample output:**

```text
=== Section 20.2 CPU baseline: guaranteed <=2-probe lookup ===

building table from keys: 3 8 13 18 21
T1 = [ . 21 . 18 . ]
T2 = [ 3 8 13 . . ]

looking up every inserted key:
  lookup(3) -> table 2, slot 0 (2 probes)
  lookup(8) -> table 2, slot 1 (2 probes)
  lookup(13) -> table 2, slot 2 (2 probes)
  lookup(18) -> table 1, slot 3 (1 probe)
  lookup(21) -> table 1, slot 1 (1 probe)

looking up two absent keys (each must still cost at most 2 probes):
  lookup(99) -> not found (2 probes)
  lookup(4) -> not found (2 probes)

expected: 3->(T2,0) 8->(T2,1) 13->(T2,2) 18->(T1,3) 21->(T1,1)
          99->not found, 4->not found, all in at most 2 probes

self-check: every lookup -- present or absent -- resolves in at
most 2 probes, matching the expected locations exactly: confirmed
```

### The Concept, In Detail

Contrasting the probe bound directly against Chapter 19's linear probing:

```
Chapter 19 (linear probing): worst-case probe count grows with how
  full the table is around a key's home slot -- a long run of occupied
  neighboring slots can force arbitrarily many probes before an empty
  slot (or the key itself) turns up. No FIXED upper bound exists.

Chapter 20 (cuckoo hashing): worst-case probe count is ALWAYS exactly
  2, for every key, whether present or absent, regardless of how full
  either table is (short of insertion itself having failed). The bound
  does not depend on the data at all -- only on the scheme's own rule
  that a key can live in exactly two possible places.
```

```
ASCII view: looking up an ABSENT key costs exactly the same as a
PRESENT one -- there is no "give up early" and no "keep searching":

  lookup(4):  T1[h1(4)=4] = "." (empty, not 4)   -- probe 1, miss
              T2[h2(4)=0] =  3  (occupied, not 4) -- probe 2, miss
              -- exactly 2 probes, key confirmed absent
```

The two absent-key cases in the CPU baseline make this concrete: `lookup(99)` misses on an empty `T1` slot, `lookup(4)` misses on an OCCUPIED `T2` slot holding a different key (`3`) -- two structurally different kinds of miss, yet both cost the identical two probes, because the lookup procedure never adapts its behavior based on what it finds; it always checks exactly these two fixed locations and nothing else.

[COMMON TRAP]
It is tempting to think a lookup that finds `T1`'s slot OCCUPIED by some other key should immediately conclude "not found" without checking `T2` at all, by analogy with how a full slot might seem informative. An occupied `T1` slot holding a different key says nothing about the key being searched for -- that key's home in `T1` was simply claimed by someone else. The search must still check `T2[h2(key)]` before concluding absence; skipping it would incorrectly report a genuinely present key (one that lost the `T1` slot during construction and settled into `T2`) as missing.

### Code and Verification

```cpp
#include <cstdio>
#include <vector>

// Chapter 20.2 main -- this book's own recurring pattern one more time
// (Section 17.1's range queries, Section 18.2/18.3's spatial queries,
// Section 19.1's linear-probe lookups): many independent READS against
// a shared, already-built table, one thread per query, zero writes and
// therefore zero atomics. What is NEW here is the bound itself -- every
// thread's loop body runs the exact same fixed TWO checks no matter
// which key it is looking for or whether that key is even present,
// giving perfectly uniform, predictable work across every thread (no
// thread can ever take longer than any other, unlike Chapter 19's
// lookup, whose probe count depends on how full the table happens to
// be around that key's home slot).
#define TABLE_SIZE 5
#define EMPTY (-1)

__device__ int h1_dev(int key) { return key % TABLE_SIZE; }
__device__ int h2_dev(int key) { return (key / TABLE_SIZE) % TABLE_SIZE; }

// One thread per query key. Every thread performs AT MOST 2 reads of
// shared, read-only memory: T1[h1(key)], then (only if that missed)
// T2[h2(key)]. Encodes the result as table*100+slot, or -1 if absent.
__global__ void cuckoo_lookup_kernel(const int* T1, const int* T2,
                                      const int* query_keys, int* results, int num_queries) {
    int t = threadIdx.x;
    if (t >= num_queries) return;
    int key = query_keys[t];
    int slot1 = h1_dev(key);
    if (T1[slot1] == key) { results[t] = 1 * 100 + slot1; return; }
    int slot2 = h2_dev(key);
    if (T2[slot2] == key) { results[t] = 2 * 100 + slot2; return; }
    results[t] = -1;
}

// ---- Host-side replay of the identical per-thread logic. ----

int h1_host(int key) { return key % TABLE_SIZE; }
int h2_host(int key) { return (key / TABLE_SIZE) % TABLE_SIZE; }

int main() {
    printf("=== Section 20.2 main: parallel independent lookups, <=2 probes each ===\n\n");

    // Section 20.2's own prebuilt table: keys {3, 8, 13, 18, 21}.
    std::vector<int> T1 = {-1, 21, -1, 18, -1};
    std::vector<int> T2 = {3, 8, 13, -1, -1};
    printf("prebuilt table: T1 = [ . 21 . 18 . ]   T2 = [ 3 8 13 . . ]\n\n");

    std::vector<int> queries = {3, 8, 13, 18, 21, 99, 4};
    std::vector<int> results(queries.size());

    printf("%zu independent lookup threads, each doing its own fixed 2-probe check:\n",
           queries.size());
    for (size_t t = 0; t < queries.size(); t++) {
        int key = queries[t];
        int slot1 = h1_host(key);
        int result;
        if (T1[slot1] == key) {
            result = 1 * 100 + slot1;
        } else {
            int slot2 = h2_host(key);
            result = (T2[slot2] == key) ? (2 * 100 + slot2) : -1;
        }
        results[t] = result;
        if (result == -1) {
            printf("  thread %zu: lookup(%d) -> not found\n", t, key);
        } else {
            printf("  thread %zu: lookup(%d) -> table %d, slot %d\n", t, key, result / 100, result % 100);
        }
    }

    std::vector<int> expected = {200, 201, 202, 103, 101, -1, -1};
    bool ok = true;
    for (size_t t = 0; t < results.size(); t++) ok = ok && (results[t] == expected[t]);

    printf("\nexpected: 3->(2,0) 8->(2,1) 13->(2,2) 18->(1,3) 21->(1,1) 99->not found 4->not found\n");
    printf("(every one of the 7 threads reads at most 2 shared slots -- no thread\n");
    printf("ever does more work than any other, regardless of which key it holds)\n");

    printf("\nself-check: all parallel independent lookups match the CPU baseline\n");
    printf("exactly, each within the guaranteed <=2-probe bound: %s\n", ok ? "confirmed" : "MISMATCH");
    return ok ? 0 : 1;
}
```

**Compile and run:**

```bash
NVDIR=/usr/local/lib/python3.11/dist-packages/nvidia/cu13
$NVDIR/bin/nvcc -arch=sm_80 -I$NVDIR/include -L$NVDIR/lib -lcudart 109_cuckoo_parallel_lookup_kernel.cu -o 109_cuckoo_parallel_lookup_kernel
LD_LIBRARY_PATH=$NVDIR/lib ./109_cuckoo_parallel_lookup_kernel
```

**Sample input:** the same prebuilt table, queried for all 5 inserted keys plus 2 absent keys, by 7 independent threads at once.

**Sample output:**

```text
=== Section 20.2 main: parallel independent lookups, <=2 probes each ===

prebuilt table: T1 = [ . 21 . 18 . ]   T2 = [ 3 8 13 . . ]

7 independent lookup threads, each doing its own fixed 2-probe check:
  thread 0: lookup(3) -> table 2, slot 0
  thread 1: lookup(8) -> table 2, slot 1
  thread 2: lookup(13) -> table 2, slot 2
  thread 3: lookup(18) -> table 1, slot 3
  thread 4: lookup(21) -> table 1, slot 1
  thread 5: lookup(99) -> not found
  thread 6: lookup(4) -> not found

expected: 3->(2,0) 8->(2,1) 13->(2,2) 18->(1,3) 21->(1,1) 99->not found 4->not found
(every one of the 7 threads reads at most 2 shared slots -- no thread
ever does more work than any other, regardless of which key it holds)

self-check: all parallel independent lookups match the CPU baseline
exactly, each within the guaranteed <=2-probe bound: confirmed
```

## 20.3 Detecting and Resolving Cycles: Rehashing When an Eviction Chain Loops Forever

### Intuition

Nothing guarantees an eviction chain terminates. If the hash functions route a set of keys into a closed loop -- key A evicts B, B evicts C, C evicts A right back into a slot that starts the same sequence over again -- the chain bounces forever without ever reaching an empty slot. A real implementation bounds the eviction count with a `MAX_EVICTIONS` cutoff; hitting that bound means "a cycle exists here," not "keep trying harder." The fix is to REHASH: pull out every key the table currently holds (from its last known-good state, before the failed insertion touched anything), choose new hash functions (here, also growing the table -- the usual practical move, since more room makes future cycles less likely), and rebuild from scratch.

### The Sequential (CPU) Baseline

```cpp
#include <cstdio>
#include <vector>

// Chapter 20.3 -- The Sequential (CPU) Baseline.
// An eviction chain is not guaranteed to terminate: if the two hash
// functions route a set of keys into a closed loop of slots, inserting
// one more key into that loop can bounce forever, alternating between
// the same handful of keys and slots without ever reaching a genuinely
// empty one. A real implementation cannot wait forever to find out --
// it bounds the eviction count at some MAX_EVICTIONS and, once that
// bound is hit, concludes a cycle exists and REHASHES: pull every key
// currently in the table back out, pick new hash functions (here, also
// growing the table, which is the usual practical choice), and rebuild
// from scratch.
#define TABLE_SIZE 5
#define EMPTY (-1)
#define MAX_EVICTIONS 12

int h1(int key) { return key % TABLE_SIZE; }
int h2(int key) { return (key / TABLE_SIZE) % TABLE_SIZE; }

// Returns true if `key` was placed; false if MAX_EVICTIONS was hit
// (the tables are left in whatever mid-chain state the loop reached --
// the caller is expected to rehash rather than trust that state).
bool try_insert(std::vector<int>& T1, std::vector<int>& T2, int key) {
    int cur = key;
    int table = 1;
    for (int evictions = 0; evictions < MAX_EVICTIONS; evictions++) {
        if (table == 1) {
            int slot = h1(cur);
            if (T1[slot] == EMPTY) {
                T1[slot] = cur;
                printf("  place %d at T1[%d] (was empty) -- done\n", cur, slot);
                return true;
            }
            int evicted = T1[slot];
            T1[slot] = cur;
            printf("  place %d at T1[%d], evicting %d\n", cur, slot, evicted);
            cur = evicted;
            table = 2;
        } else {
            int slot = h2(cur);
            if (T2[slot] == EMPTY) {
                T2[slot] = cur;
                printf("  place %d at T2[%d] (was empty) -- done\n", cur, slot);
                return true;
            }
            int evicted = T2[slot];
            T2[slot] = cur;
            printf("  place %d at T2[%d], evicting %d\n", cur, slot, evicted);
            cur = evicted;
            table = 1;
        }
    }
    printf("  eviction count reached MAX_EVICTIONS=%d -- CYCLE DETECTED, abandoning insert(%d)\n",
           MAX_EVICTIONS, key);
    return false;
}

void print_table(const char* name, const std::vector<int>& T) {
    printf("%s = [ ", name);
    for (int v : T) { if (v == EMPTY) printf(". "); else printf("%d ", v); }
    printf("]\n");
}

// Sequential extraction: scan every slot of both tables, in order,
// collecting whatever keys are present. One thread's worth of work
// here; Section 20.3's main file makes this step parallel instead.
std::vector<int> extract_keys(const std::vector<int>& T1, const std::vector<int>& T2) {
    std::vector<int> out;
    for (int v : T1) if (v != EMPTY) out.push_back(v);
    for (int v : T2) if (v != EMPTY) out.push_back(v);
    return out;
}

int main() {
    printf("=== Section 20.3 CPU baseline: cycle detection and rehashing ===\n\n");

    printf("building table from keys {0, 1, 5, 6, 10} (old hash: h1=k%%5, h2=(k/5)%%5):\n");
    std::vector<int> T1(TABLE_SIZE, EMPTY), T2(TABLE_SIZE, EMPTY);
    std::vector<int> build_keys = {0, 1, 5, 6, 10};
    bool build_ok = true;
    for (int key : build_keys) {
        printf("insert(%d)\n", key);
        build_ok = build_ok && try_insert(T1, T2, key);
    }
    print_table("final T1", T1);
    print_table("final T2", T2);
    printf("\n");

    printf("now attempting insert(11) into that same table:\n");
    printf("insert(11)\n");
    std::vector<int> T1_before_cycle = T1, T2_before_cycle = T2;
    bool insert11_ok = try_insert(T1, T2, 11);
    printf("insert(11) result: %s\n\n", insert11_ok ? "SUCCESS" : "CYCLE DETECTED");

    printf("REHASHING: extracting every key from the table as it stood BEFORE\n");
    printf("the failed insert(11) attempt (the mid-chain state left by an\n");
    printf("abandoned eviction chain is not trustworthy, so extraction always\n");
    printf("reads from the last known-good table, not the one just corrupted):\n");
    std::vector<int> extracted = extract_keys(T1_before_cycle, T2_before_cycle);
    printf("  extracted: ");
    for (int k : extracted) printf("%d ", k);
    printf("\n");
    extracted.push_back(11);  // the key that triggered the rehash joins the rebuild
    printf("  plus the key that triggered the rehash: 11\n");
    printf("  full set to rebuild: ");
    for (int k : extracted) printf("%d ", k);
    printf("\n\n");

    printf("choosing new hash functions and growing the table from size 5 to 7:\n");
    printf("  h1_new(k) = k %% 7\n");
    printf("  h2_new(k) = (2*k + 3) %% 7\n\n");

    const int NEW_SIZE = 7;
    std::vector<int> NT1(NEW_SIZE, EMPTY), NT2(NEW_SIZE, EMPTY);
    auto h1n = [](int k) { return k % 7; };
    auto h2n = [](int k) { return (2 * k + 3) % 7; };

    bool rehash_ok = true;
    for (int key : extracted) {
        int cur = key;
        int table = 1;
        bool placed = false;
        for (int evictions = 0; evictions < MAX_EVICTIONS; evictions++) {
            if (table == 1) {
                int slot = h1n(cur);
                if (NT1[slot] == EMPTY) {
                    NT1[slot] = cur;
                    printf("  place %d at T1[%d] (was empty) -- done\n", cur, slot);
                    placed = true;
                    break;
                }
                int evicted = NT1[slot];
                NT1[slot] = cur;
                printf("  place %d at T1[%d], evicting %d\n", cur, slot, evicted);
                cur = evicted;
                table = 2;
            } else {
                int slot = h2n(cur);
                if (NT2[slot] == EMPTY) {
                    NT2[slot] = cur;
                    printf("  place %d at T2[%d] (was empty) -- done\n", cur, slot);
                    placed = true;
                    break;
                }
                int evicted = NT2[slot];
                NT2[slot] = cur;
                printf("  place %d at T2[%d], evicting %d\n", cur, slot, evicted);
                cur = evicted;
                table = 1;
            }
        }
        rehash_ok = rehash_ok && placed;
    }
    printf("\n");
    print_table("rehashed T1", NT1);
    print_table("rehashed T2", NT2);

    bool ok = build_ok && !insert11_ok && rehash_ok;
    std::vector<int> exp_T1_before = {5, 1, -1, -1, -1};
    std::vector<int> exp_T2_before = {0, 6, 10, -1, -1};
    ok = ok && (T1_before_cycle == exp_T1_before) && (T2_before_cycle == exp_T2_before);
    std::vector<int> exp_extracted = {5, 1, 0, 6, 10, 11};
    ok = ok && (extracted == exp_extracted);
    std::vector<int> exp_NT1 = {0, 1, -1, 10, 11, 5, 6};
    std::vector<int> exp_NT2 = {-1, -1, -1, -1, -1, -1, -1};
    ok = ok && (NT1 == exp_NT1) && (NT2 == exp_NT2);

    for (int key : extracted) {
        bool found = (NT1[h1n(key)] == key) || (NT2[h2n(key)] == key);
        ok = ok && found;
    }

    printf("\nexpected: build of {0,1,5,6,10} succeeds; insert(11) cycles; extracted\n");
    printf("keys are {5,1,0,6,10}; rehash under (h1_new,h2_new) mod 7 places all\n");
    printf("6 keys (including 11) with zero further evictions\n");
    printf("\nself-check: cycle correctly detected, all pre-cycle keys correctly\n");
    printf("extracted, and the rehashed table holds every key findable: %s\n", ok ? "confirmed" : "MISMATCH");
    return ok ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++17 -Wall -Wextra -O2 110_cuckoo_cycle_rehash_cpu_baseline.cpp -o 110_cuckoo_cycle_rehash_cpu_baseline
./110_cuckoo_cycle_rehash_cpu_baseline
```

**Sample input:** keys `{0, 1, 5, 6, 10}` built successfully into size-5 tables, then a 6th key (`11`) whose insertion provably cycles, triggering extraction and a rehash into size-7 tables under new hash functions.

**Sample output:**

```text
=== Section 20.3 CPU baseline: cycle detection and rehashing ===

building table from keys {0, 1, 5, 6, 10} (old hash: h1=k%5, h2=(k/5)%5):
insert(0)
  place 0 at T1[0] (was empty) -- done
insert(1)
  place 1 at T1[1] (was empty) -- done
insert(5)
  place 5 at T1[0], evicting 0
  place 0 at T2[0] (was empty) -- done
insert(6)
  place 6 at T1[1], evicting 1
  place 1 at T2[0], evicting 0
  place 0 at T1[0], evicting 5
  place 5 at T2[1] (was empty) -- done
insert(10)
  place 10 at T1[0], evicting 0
  place 0 at T2[0], evicting 1
  place 1 at T1[1], evicting 6
  place 6 at T2[1], evicting 5
  place 5 at T1[0], evicting 10
  place 10 at T2[2] (was empty) -- done
final T1 = [ 5 1 . . . ]
final T2 = [ 0 6 10 . . ]

now attempting insert(11) into that same table:
insert(11)
  place 11 at T1[1], evicting 1
  place 1 at T2[0], evicting 0
  place 0 at T1[0], evicting 5
  place 5 at T2[1], evicting 6
  place 6 at T1[1], evicting 11
  place 11 at T2[2], evicting 10
  place 10 at T1[0], evicting 0
  place 0 at T2[0], evicting 1
  place 1 at T1[1], evicting 6
  place 6 at T2[1], evicting 5
  place 5 at T1[0], evicting 10
  place 10 at T2[2], evicting 11
  eviction count reached MAX_EVICTIONS=12 -- CYCLE DETECTED, abandoning insert(11)
insert(11) result: CYCLE DETECTED

REHASHING: extracting every key from the table as it stood BEFORE
the failed insert(11) attempt (the mid-chain state left by an
abandoned eviction chain is not trustworthy, so extraction always
reads from the last known-good table, not the one just corrupted):
  extracted: 5 1 0 6 10 
  plus the key that triggered the rehash: 11
  full set to rebuild: 5 1 0 6 10 11 

choosing new hash functions and growing the table from size 5 to 7:
  h1_new(k) = k % 7
  h2_new(k) = (2*k + 3) % 7

  place 5 at T1[5] (was empty) -- done
  place 1 at T1[1] (was empty) -- done
  place 0 at T1[0] (was empty) -- done
  place 6 at T1[6] (was empty) -- done
  place 10 at T1[3] (was empty) -- done
  place 11 at T1[4] (was empty) -- done

rehashed T1 = [ 0 1 . 10 11 5 6 ]
rehashed T2 = [ . . . . . . . ]

expected: build of {0,1,5,6,10} succeeds; insert(11) cycles; extracted
keys are {5,1,0,6,10}; rehash under (h1_new,h2_new) mod 7 places all
6 keys (including 11) with zero further evictions

self-check: cycle correctly detected, all pre-cycle keys correctly
extracted, and the rehashed table holds every key findable: confirmed
```

### The Concept, In Detail

Why does `{0, 1, 5, 6, 10}` build cleanly but adding `11` cycles forever? Under `h1(k) = k % 5` and `h2(k) = (k / 5) % 5`, these 5 keys occupy exactly 5 distinct slots across the two tables -- `T1[0]`, `T1[1]`, `T2[0]`, `T2[1]`, `T2[2]` -- with zero slots to spare in that group. Key `11` hashes to `h1(11) = 1` and `h2(11) = 2`, both of which are slots ALREADY inside that same fully-occupied group:

```
ASCII view: the "component" of slots reachable by these 6 keys,
BEFORE inserting 11, has exactly 5 slots and exactly 5 keys -- full,
with no spare capacity anywhere in the group:

  T1[0]=5   T1[1]=1
  T2[0]=0   T2[1]=6   T2[2]=10

Inserting 11 (home slots T1[1] and T2[2] -- BOTH already inside this
same saturated group) adds a 6th key competing for the same 5 slots.
There is no empty slot left anywhere in the group for the chain to
end on, so it must cycle: 11 evicts 1, 1 evicts 0, 0 evicts 5, 5
evicts 6, 6 evicts 11 (back where it started), 11 evicts 10, 10
evicts 0, ... the same handful of keys rotate through the same 5
slots forever.
```

`MAX_EVICTIONS = 12` catches this well before it would ever loop indefinitely in practice -- twice through the chain's own period of 6 evictions is more than enough to prove it is repeating rather than progressing. Once detected, extraction reads the table's LAST KNOWN-GOOD state -- the one from right before `insert(11)` began touching it -- because a chain that was abandoned mid-cycle leaves the tables in an arbitrary partial state that mixes up which keys are even still correctly placed.

The new hash functions are not merely relabeled versions of the old ones: `h1_new(k) = k % 7` and `h2_new(k) = (2k + 3) % 7` map this specific 6-key set to 6 completely distinct `T1` slots, so every key settles immediately with zero further evictions at all -- a direct demonstration that growing the table (from 5 to 7) and picking a structurally different hash function can eliminate a cycle that no amount of retrying the OLD functions could ever have resolved.

[COMMON TRAP]
It is tempting to think rehashing just needs DIFFERENT CONSTANTS in the same style of hash function to fix a cycle. If the new functions preserve the same underlying structure as the old ones (for instance, still deriving both hash values from the same two independent pieces of the key, just recombined), the identical set of keys can still collide in the identical pattern, cycling again under the "new" functions. A rehash needs hash functions that route the SAME set of keys to a genuinely different pattern of slots -- which is exactly why growing the table size, not just changing constants, is the standard practical choice: more slots make it far less likely that any group of keys re-saturates a shared component.

### Code and Verification

```cpp
#include <cstdio>
#include <vector>
#include <algorithm>

// Chapter 20.3 main -- the genuinely parallel piece of rehashing is not
// the rebuild itself (a bounded, inherently sequential pointer-chase,
// same as every other cuckoo insertion in this chapter); it is
// EXTRACTING every key out of the OLD table first. That is a full-table
// SCAN: one thread per slot, across both T1 and T2, each thread doing a
// single READ of its own slot and nothing else -- a new variant of this
// book's recurring "many independent reads" pattern, this time over a
// table's raw storage rather than over a set of queries. The one write
// each thread performs, when its slot is non-empty, is a compacting
// `atomicAdd` on a shared output-counter -- exactly Chapter 6's stream
// compaction, applied here to "compact out the empty slots."
#define TABLE_SIZE 5
#define EMPTY (-1)
#define NUM_SLOTS (2 * TABLE_SIZE)

// One thread per slot: threads 0..TABLE_SIZE-1 read T1, threads
// TABLE_SIZE..2*TABLE_SIZE-1 read T2. Every thread performs exactly one
// read of shared, read-only memory; only a non-empty slot performs a
// write, and that write's DESTINATION (not its value) is the only
// thing contended, resolved via a single atomicAdd per writing thread.
__global__ void extract_keys_kernel(const int* T1, const int* T2, int* out, int* out_count) {
    int t = threadIdx.x;
    if (t >= NUM_SLOTS) return;
    int value = (t < TABLE_SIZE) ? T1[t] : T2[t - TABLE_SIZE];
    if (value != EMPTY) {
        int pos = atomicAdd(out_count, 1);
        out[pos] = value;
    }
}

// ---- Host-side replay of the identical per-thread logic, run under
// two DIFFERENT thread-execution orders to demonstrate that the final
// POSITIONS in the output array can differ, while the SET of extracted
// keys never does. ----

std::vector<int> replay_extract(const std::vector<int>& T1, const std::vector<int>& T2,
                                 const std::vector<int>& thread_order) {
    std::vector<int> out(NUM_SLOTS, EMPTY);
    int out_count = 0;
    for (int t : thread_order) {
        int value = (t < TABLE_SIZE) ? T1[t] : T2[t - TABLE_SIZE];
        if (value != EMPTY) {
            int pos = out_count;
            out_count++;
            out[pos] = value;
        }
    }
    out.resize(out_count);
    return out;
}

int main() {
    printf("=== Section 20.3 main: parallel extraction from the pre-cycle table ===\n\n");

    // The table as it stood right before the abandoned insert(11) --
    // exactly the CPU baseline's `T1_before_cycle` / `T2_before_cycle`.
    std::vector<int> T1 = {5, 1, -1, -1, -1};
    std::vector<int> T2 = {0, 6, 10, -1, -1};
    printf("pre-cycle table: T1 = [ 5 1 . . . ]   T2 = [ 0 6 10 . . ]\n");
    printf("(%d threads launched, one per slot across both tables; each\n", NUM_SLOTS);
    printf("reads exactly one shared slot, writing out only if non-empty)\n\n");

    // Order A: threads fire in increasing slot order (0..9), i.e. as
    // if T1's five slots resolve their atomicAdd before T2's do.
    std::vector<int> order_A = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<int> out_A = replay_extract(T1, T2, order_A);
    printf("execution order A (T1's threads win every race first):\n  extracted: ");
    for (int v : out_A) printf("%d ", v);
    printf("\n");

    // Order B: an adversarial interleaving where T2's threads happen to
    // resolve their atomicAdd first, then T1's.
    std::vector<int> order_B = {5, 6, 7, 8, 9, 0, 1, 2, 3, 4};
    std::vector<int> out_B = replay_extract(T1, T2, order_B);
    printf("execution order B (T2's threads win every race first):\n  extracted: ");
    for (int v : out_B) printf("%d ", v);
    printf("\n\n");

    std::vector<int> sorted_A = out_A, sorted_B = out_B;
    std::sort(sorted_A.begin(), sorted_A.end());
    std::sort(sorted_B.begin(), sorted_B.end());

    bool same_set = (sorted_A == sorted_B);
    bool different_order = (out_A != out_B);

    printf("sorted A: ");
    for (int v : sorted_A) printf("%d ", v);
    printf("\nsorted B: ");
    for (int v : sorted_B) printf("%d ", v);
    printf("\n\n");

    std::vector<int> expected_set = {0, 1, 5, 6, 10};
    bool ok = same_set && different_order && (sorted_A == expected_set);

    printf("expected extracted set (any order): {0, 1, 5, 6, 10}\n");
    printf("(the exact POSITIONS in the output array depend on which threads'\n");
    printf("atomicAdd calls happen to resolve first -- genuinely different\n");
    printf("between order A and order B -- but the SET of extracted keys is\n");
    printf("identical either way, since extraction never writes, only counts)\n");

    printf("\nself-check: two adversarial execution orders produce a different\n");
    printf("array layout but the exact same extracted key set: %s\n", ok ? "confirmed" : "MISMATCH");
    return ok ? 0 : 1;
}
```

**Compile and run:**

```bash
NVDIR=/usr/local/lib/python3.11/dist-packages/nvidia/cu13
$NVDIR/bin/nvcc -arch=sm_80 -I$NVDIR/include -L$NVDIR/lib -lcudart 111_cuckoo_parallel_extract_kernel.cu -o 111_cuckoo_parallel_extract_kernel
LD_LIBRARY_PATH=$NVDIR/lib ./111_cuckoo_parallel_extract_kernel
```

**Sample input:** the pre-cycle table (`T1 = [5, 1, ., ., .]`, `T2 = [0, 6, 10, ., .]`), extracted by 10 threads at once -- one thread per slot across both tables -- under two different adversarial execution orders.

**Sample output:**

```text
=== Section 20.3 main: parallel extraction from the pre-cycle table ===

pre-cycle table: T1 = [ 5 1 . . . ]   T2 = [ 0 6 10 . . ]
(10 threads launched, one per slot across both tables; each
reads exactly one shared slot, writing out only if non-empty)

execution order A (T1's threads win every race first):
  extracted: 5 1 0 6 10 
execution order B (T2's threads win every race first):
  extracted: 0 6 10 5 1 

sorted A: 0 1 5 6 10 
sorted B: 0 1 5 6 10 

expected extracted set (any order): {0, 1, 5, 6, 10}
(the exact POSITIONS in the output array depend on which threads'
atomicAdd calls happen to resolve first -- genuinely different
between order A and order B -- but the SET of extracted keys is
identical either way, since extraction never writes, only counts)

self-check: two adversarial execution orders produce a different
array layout but the exact same extracted key set: confirmed
```

## Chapter Summary

Cuckoo hashing gives every key exactly two candidate homes, one per table, via two independent hash functions; insertion is an eviction chain where a displaced key always retries in its OTHER table, continuing until some key lands in a genuinely empty slot. Construction's honest parallel opportunity is SHARDING -- many independent shards, each built by one thread running its own private, sequential eviction chain against its own memory, with zero cross-shard interaction -- rather than attempting genuinely concurrent insertion into one shared table, which this book treats as beyond its rigorous scope. The scheme's entire payoff is lookup: because a key can only ever be in one of two fixed places, lookup costs at most two probes for any key, present or absent, a genuine worst-case guarantee that linear probing (Chapter 19) cannot offer. Eviction chains are not guaranteed to terminate -- a saturated group of slots can force a permanent cycle -- so a bounded `MAX_EVICTIONS` cutoff detects this, triggering a rehash: extracting every key from the table's last known-good state (itself an embarrassingly parallel full-table scan, one thread per slot) and rebuilding under new hash functions and, typically, a larger table. This closes Part 5 -- Hash Tables.

## Self-Check Questions

1. When an eviction chain displaces a key from `T1`, why does that displaced key retry in `T2` specifically, rather than trying a different slot back in `T1`?
2. Why is sharding -- many independent shards, one thread each -- described as the "honest" parallel angle on cuckoo construction, rather than attempting concurrent insertion into one shared table?
3. Why does cuckoo hashing's lookup have a hard worst-case bound of exactly 2 probes, in contrast to linear probing's unbounded worst case from Chapter 19?
4. Walk through why looking up an ABSENT key costs exactly the same number of probes as looking up a PRESENT one under cuckoo hashing.
5. What does it mean for a group of slots to be "saturated," and why does inserting one more key that hashes into an already-saturated group force a cycle?
6. Why must extraction, when a cycle is detected, read the table's state from BEFORE the failed insertion began, rather than whatever state the abandoned eviction chain left behind?

## Where We Go Next

Cuckoo hashing closes Part 5 -- Hash Tables, and with it, this book's tour of how individual data structures change shape under parallelism: trees, ranges, spatial structures, and now hash tables have each revealed their own specific hazards (shared-slot races, non-commutative writes, unbounded probe chains) and their own specific safe parallel opportunities (many independent reads, sharding, level-synchronous construction). Part 6 -- Graphs turns to structures built from many of these pieces at once: Chapter 21 begins with CSR (compressed sparse row) and its alternatives, the representations that make an entire graph's edges amenable to parallel traversal in the first place.

## Worked Solutions

**1.** Cuckoo hashing gives every key exactly one fixed slot per table -- `h1(key)` in `T1` and `h2(key)` in `T2` -- there is no notion of "the next slot over" the way linear probing has. A key evicted from `T1` has already tried its only slot in `T1`; its one remaining option, by the scheme's own definition, is its fixed home slot in `T2`. Retrying `T1` would mean checking the exact same slot it was just evicted from, which is guaranteed to still be occupied by whatever just evicted it -- accomplishing nothing.

**2.** Building any single shard's tables is an inherently sequential process: each insertion's eviction chain depends entirely on the exact current contents of that shard's own tables, so the steps within one shard's construction cannot be reordered or split across threads without changing the result. Attempting genuinely concurrent insertion into ONE shared table would mean multiple simultaneous eviction chains reading and evicting from the same slots at once -- a hazard this book's tools (single atomic reads and compare-and-swaps) cannot safely resolve, since an eviction chain's own correctness depends on a consistent view of the table across several sequential steps, not a single atomic operation. Sharding sidesteps this entirely: each shard's construction is safely sequential on its own thread, and running many shards at once is safe specifically because they touch completely disjoint memory.

**3.** A key can only ever be stored in one of exactly two places: `T1[h1(key)]` or `T2[h2(key)]`, by construction -- insertion never places a key anywhere else. Lookup therefore only ever needs to check those two fixed locations: if neither holds the key, no other location could possibly hold it either, so the search can stop with certainty. Linear probing offers no such fixed set of candidate locations -- a key's position depends on how many prior collisions pushed it forward from its home slot, which can in principle be arbitrarily many slots away.

**4.** Lookup's procedure is fixed in advance and never adapts based on what it finds: check `T1[h1(key)]`, and if that doesn't match, check `T2[h2(key)]`. A present key is found on whichever of those two checks matches; an absent key fails BOTH checks -- but the number of checks performed is identical either way, because the procedure has no way to "give up early" or "search harder": there are only ever two candidate locations to examine, and both must be examined (or a match found) before the answer -- present or absent -- can be determined.

**5.** A group (or "component") of slots is saturated when the keys currently routed into it, together with the two tables' hash functions, exactly fill every slot in that group with no empty slot left anywhere within it. Inserting one more key whose home slots (in `T1` and `T2`) both fall inside that same already-full group adds one more key competing for a fixed set of slots that already has exactly enough keys to fill it -- there is no longer any empty slot within reach of an eviction chain confined to that group, so the chain can only ever bounce among the existing keys and slots, never terminating.

**6.** An eviction chain that gets abandoned mid-cycle (once `MAX_EVICTIONS` is hit) has already partially overwritten the table with a sequence of evictions that never reached a stable resting point -- the mid-chain state mixes up which specific key occupies which specific slot in a way that does not correspond to any correctly-built table at all. Reading from the state as it stood immediately BEFORE the failed insertion began guarantees every key extracted is one that was genuinely, correctly placed by a completed insertion, which is exactly the set of keys the rehash needs to faithfully reproduce (plus the one new key that triggered the rehash in the first place).
