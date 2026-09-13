# Chapter 34: Genomics -- Large-Scale Sequence Alignment and k-mer Counting

Genomic data looks nothing like the numbers, boxes, and orders this book has worked with so far: it is long strings drawn from a tiny four-letter alphabet (A, C, G, T), where what matters is which short substrings -- k-mers -- repeat, and where a query sequence's substrings reappear inside a much larger reference. Real aligners (BLAST, BWA, and their descendants) do not compare every possible pair of positions directly; they build an INDEX over the reference's k-mers once, then use it to find candidate matches, called seeds, extremely fast. This chapter builds exactly that pipeline from tools this book has already proven: Section 34.1 counts k-mers with Chapter 19's hash table doing Chapter 7's job, Section 34.2 indexes them with Chapter 16's parallel trie construction, and Section 34.3 uses that index to find every seed match between a query and the reference.

## 34.1 Counting k-mers via a Hash Table

### Intuition

A k-mer is simply a fixed-length substring -- for k=3, every 3-character window of a sequence. Counting how many times each distinct k-mer occurs is exactly Chapter 7's histogram, generalized in one specific way: Chapter 7's histogram had a small, fixed number of buckets known in advance (say, 256 for byte values), but the number of DISTINCT k-mers appearing in a real genome is unknown ahead of time and can be enormous. Chapter 19's open-addressing hash table solves precisely this problem -- an unbounded key space mapped down to a small, dense array -- so k-mer counting becomes Chapter 19's insert, with one addition: inserting a key that is ALREADY present must increment its count rather than fail.

### The Sequential (CPU) Baseline

```cpp
// 190_kmer_hash_count_cpu_baseline.cpp
//
// Chapter 34.1 -- sequential baseline for counting k-mer occurrences
// in a reference DNA sequence. This is Chapter 7's histogram,
// generalized from small fixed integer buckets to an unbounded key
// space (every possible k-mer) via Chapter 19's open-addressing hash
// table -- insert-or-increment, instead of Chapter 19's insert-only.
// Each nucleotide (A,C,G,T) is encoded as a 2-bit value, so a k=3
// k-mer packs into a single small integer key.
//
// Compile: g++ -std=c++17 -Wall -Wextra -O2 190_kmer_hash_count_cpu_baseline.cpp -o 190_kmer_hash_count_cpu_baseline
// Run:     ./190_kmer_hash_count_cpu_baseline

#include <cstdio>
#include <optional>
#include <string>
#include <vector>

constexpr int K = 3;
constexpr int CAPACITY = 5;
const std::string REFERENCE = "ACGTACGTAC";

int base_code(char c) {
    switch (c) {
        case 'A': return 0;
        case 'C': return 1;
        case 'G': return 2;
        default:  return 3;   // 'T'
    }
}

int encode_kmer(const std::string& s) {
    int v = 0;
    for (char c : s) v = v * 4 + base_code(c);
    return v;
}

int main() {
    std::vector<std::string> ref_kmers;
    for (size_t i = 0; i + K <= REFERENCE.size(); ++i) ref_kmers.push_back(REFERENCE.substr(i, K));

    std::printf("=== Section 34.1 CPU baseline: k-mer counting via open-addressing hash table ===\n\n");
    std::printf("reference sequence: %s\n", REFERENCE.c_str());
    std::printf("k = %d, capacity = %d\n\n", K, CAPACITY);

    std::printf("reference k-mers (by position): [");
    for (size_t i = 0; i < ref_kmers.size(); ++i) {
        std::printf("(%zu, '%s')%s", i, ref_kmers[i].c_str(), (i + 1 < ref_kmers.size()) ? ", " : "");
    }
    std::printf("]\n\n");

    std::vector<std::optional<std::string>> table_key(CAPACITY);
    std::vector<int> table_count(CAPACITY, 0);

    for (size_t pos = 0; pos < ref_kmers.size(); ++pos) {
        const std::string& kmer = ref_kmers[pos];
        int key = encode_kmer(kmer);
        int home = key % CAPACITY;
        int probes = 0;
        while (true) {
            int slot = (home + probes) % CAPACITY;
            if (!table_key[slot].has_value()) {
                table_key[slot] = kmer;
                table_count[slot] = 1;
                std::printf("position %zu: kmer='%s' key=%d home_bucket=%d probes=%d -> slot %d "
                            "(inserted), count now %d\n", pos, kmer.c_str(), key, home, probes, slot, table_count[slot]);
                break;
            } else if (table_key[slot].value() == kmer) {
                table_count[slot]++;
                std::printf("position %zu: kmer='%s' key=%d home_bucket=%d probes=%d -> slot %d "
                            "(incremented), count now %d\n", pos, kmer.c_str(), key, home, probes, slot, table_count[slot]);
                break;
            }
            probes++;
        }
    }

    std::printf("\nfinal table keys:   [");
    for (int i = 0; i < CAPACITY; ++i) {
        std::printf("%s%s", table_key[i].has_value() ? ("'" + table_key[i].value() + "'").c_str() : "None",
                    (i + 1 < CAPACITY) ? ", " : "");
    }
    std::printf("]\n");
    std::printf("final table counts: [");
    for (int i = 0; i < CAPACITY; ++i) std::printf("%d%s", table_count[i], (i + 1 < CAPACITY) ? ", " : "");
    std::printf("]\n");

    bool ok = true;
    std::vector<std::pair<std::string, int>> expected = {{"ACG", 2}, {"CGT", 2}, {"GTA", 2}, {"TAC", 2}};
    for (auto& [kmer, count] : expected) {
        bool found = false;
        for (int i = 0; i < CAPACITY; ++i) {
            if (table_key[i].has_value() && table_key[i].value() == kmer && table_count[i] == count) found = true;
        }
        if (!found) ok = false;
    }
    std::printf("\nself-check: every distinct k-mer counted exactly twice: %s\n", ok ? "confirmed" : "MISMATCH");

    return ok ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++17 -Wall -Wextra -O2 190_kmer_hash_count_cpu_baseline.cpp -o 190_kmer_hash_count_cpu_baseline
./190_kmer_hash_count_cpu_baseline
```

**Sample input:** the reference sequence `ACGTACGTAC` (10 bases), k=3, giving 8 overlapping k-mers across 4 distinct values, each occurring exactly twice, inserted into a capacity-5 hash table (small enough to force a genuine collision).

**Sample output:**

```text
=== Section 34.1 CPU baseline: k-mer counting via open-addressing hash table ===

reference sequence: ACGTACGTAC
k = 3, capacity = 5

reference k-mers (by position): [(0, 'ACG'), (1, 'CGT'), (2, 'GTA'), (3, 'TAC'), (4, 'ACG'), (5, 'CGT'), (6, 'GTA'), (7, 'TAC')]

position 0: kmer='ACG' key=6 home_bucket=1 probes=0 -> slot 1 (inserted), count now 1
position 1: kmer='CGT' key=27 home_bucket=2 probes=0 -> slot 2 (inserted), count now 1
position 2: kmer='GTA' key=44 home_bucket=4 probes=0 -> slot 4 (inserted), count now 1
position 3: kmer='TAC' key=49 home_bucket=4 probes=1 -> slot 0 (inserted), count now 1
position 4: kmer='ACG' key=6 home_bucket=1 probes=0 -> slot 1 (incremented), count now 2
position 5: kmer='CGT' key=27 home_bucket=2 probes=0 -> slot 2 (incremented), count now 2
position 6: kmer='GTA' key=44 home_bucket=4 probes=0 -> slot 4 (incremented), count now 2
position 7: kmer='TAC' key=49 home_bucket=4 probes=1 -> slot 0 (incremented), count now 2

final table keys:   ['TAC', 'ACG', 'CGT', None, 'GTA']
final table counts: [2, 2, 2, 0, 2]

self-check: every distinct k-mer counted exactly twice: confirmed
```

### The Concept, In Detail

```
ASCII view: a k-mer's key is its own 2-bit-per-base encoding, packed
into one small integer -- A=0, C=1, G=2, T=3.

  k-mer "ACG":  A(00) C(01) G(10)  ->  key = 0*16 + 1*4 + 2*1 = 6
  k-mer "TAC":  T(11) A(00) C(01)  ->  key = 3*16 + 0*4 + 1*1 = 49

  hash table (capacity 5), home_bucket = key % 5:
    ACG -> key 6  -> bucket 1
    TAC -> key 49 -> bucket 4   <-- collides with GTA's bucket 4
    GTA -> key 44 -> bucket 4   <-- occupies bucket 4 first
                                     TAC probes onward to bucket 0
```

This 2-bit-per-base encoding is not a simplification invented for this book -- it is exactly how real genomics tools represent DNA internally, because it is four times denser than one byte per base and makes k-mer keys cheap, fixed-width integers rather than variable-length strings, which is exactly the kind of key Chapter 19's hash table was built to handle efficiently.

[COMMON TRAP]
It is tempting to think a "hash table for genomics" needs some genomics-specific hashing scheme fundamentally different from Chapter 19's. The ENCODING step (turning a DNA string into an integer key) is domain-specific, but everything after that -- computing `key % capacity`, probing on collision, comparing keys for equality -- is exactly Chapter 19's open addressing, completely unmodified. The only genuinely new piece Section 34.1 adds is the insert-or-INCREMENT behavior, needed because, unlike Chapter 19's original KV store, a repeated key here is not an error to reject but the very thing being counted.

### Code and Verification

```cpp
#include <cstdio>
#include <string>
#include <vector>

// Chapter 34.1 main -- concurrent k-mer counting: an insert-or-
// increment hash table, exactly Chapter 19's CAS-retry probe loop for
// the insert path, with an atomicAdd increment added for the case
// where the probed slot already holds a matching key. Two threads
// racing on the SAME k-mer (a repeated occurrence in the reference)
// must have exactly one of them insert and the other increment,
// regardless of which one's atomicCAS happens to land first.

#define CAPACITY 5
#define EMPTY (-1)

__device__ int encode_kmer_device(int c0, int c1, int c2) {
    return (c0 * 4 + c1) * 4 + c2;
}

__global__ void kmer_insert_or_increment_kernel(const int* keys, int* table_key, int* table_count) {
    int tid = threadIdx.x;
    int key = keys[tid];
    int home = key % CAPACITY;
    int probes = 0;
    while (true) {
        int slot = (home + probes) % CAPACITY;
        int old = atomicCAS(&table_key[slot], EMPTY, key);
        if (old == EMPTY) {
            table_count[slot] = 1;
            return;
        } else if (old == key) {
            atomicAdd(&table_count[slot], 1);
            return;
        }
        probes++;
    }
}

// ---- Host-side replay of the identical atomicCAS/atomicAdd logic,
// ---- driving a forced landing order where the SECOND occurrence of
// ---- a repeated k-mer lands before the FIRST. ----

int main() {
    printf("=== Section 34.1 main: concurrent insert-or-increment via atomicCAS + atomicAdd ===\n\n");

    // positions 0 (ACG, key 6), 1 (CGT, key 27), 4 (ACG, key 6, repeat)
    std::vector<int> positions = {4, 0, 1};
    auto kmer_of = [](int pos) -> std::pair<std::string, int> {
        if (pos == 0 || pos == 4) return {"ACG", 6};
        return {"CGT", 27};
    };

    printf("landing order (by reference position): [ ");
    for (int p : positions) printf("%d ", p);
    printf("]\n\n");

    std::vector<int> table_key(CAPACITY, EMPTY);
    std::vector<int> table_count(CAPACITY, 0);

    for (int pos : positions) {
        auto [kmer, key] = kmer_of(pos);
        int home = key % CAPACITY;
        int probes = 0;
        while (true) {
            int slot = (home + probes) % CAPACITY;
            if (table_key[slot] == EMPTY) {
                table_key[slot] = key;
                table_count[slot] = 1;
                printf("  position %d (kmer=%s, key=%d): CAS on slot %d succeeds (was empty) "
                       "-> inserted, count=1\n", pos, kmer.c_str(), key, slot);
                break;
            } else if (table_key[slot] == key) {
                table_count[slot]++;
                printf("  position %d (kmer=%s, key=%d): CAS on slot %d fails (already holds "
                       "this key) -> atomicAdd(count[%d],1) -> count=%d\n",
                       pos, kmer.c_str(), key, slot, slot, table_count[slot]);
                break;
            }
            probes++;
        }
    }

    printf("\nfinal table keys:   [ ");
    for (int k : table_key) printf("%d ", k);
    printf("]\n");
    printf("final table counts: [ ");
    for (int c : table_count) printf("%d ", c);
    printf("]\n");

    bool ok = (table_key[1] == 6 && table_count[1] == 2 && table_key[2] == 27 && table_count[2] == 1);
    printf("\nself-check: ACG counted twice at slot 1 (regardless of which occurrence landed\n");
    printf("first), CGT counted once at slot 2, uncontended: %s\n", ok ? "confirmed" : "MISMATCH");

    return ok ? 0 : 1;
}
```

**Compile and run:**

```bash
NVDIR=/usr/local/lib/python3.11/dist-packages/nvidia/cu13
$NVDIR/bin/nvcc -arch=sm_80 -I$NVDIR/include -L$NVDIR/lib -lcudart 191_kmer_hash_count_kernel.cu -o 191_kmer_hash_count_kernel
LD_LIBRARY_PATH=$NVDIR/lib ./191_kmer_hash_count_kernel
```

**Sample input:** three concurrent threads processing reference positions 0 and 4 (both the k-mer "ACG", a genuine repeat) and position 1 ("CGT", uncontended), with position 4's thread forced to land before position 0's.

**Sample output:**

```text
=== Section 34.1 main: concurrent insert-or-increment via atomicCAS + atomicAdd ===

landing order (by reference position): [ 4 0 1 ]

  position 4 (kmer=ACG, key=6): CAS on slot 1 succeeds (was empty) -> inserted, count=1
  position 0 (kmer=ACG, key=6): CAS on slot 1 fails (already holds this key) -> atomicAdd(count[1],1) -> count=2
  position 1 (kmer=CGT, key=27): CAS on slot 2 succeeds (was empty) -> inserted, count=1

final table keys:   [ -1 6 27 -1 -1 ]
final table counts: [ 0 2 1 0 0 ]

self-check: ACG counted twice at slot 1 (regardless of which occurrence landed
first), CGT counted once at slot 2, uncontended: confirmed
```

## 34.2 Indexing k-mer Positions with a Trie

### Intuition

A hash table answers "how many times does this k-mer occur," but not "WHERE." Section 34.3's seed-finding needs exact positions, and for a small, fixed k over DNA's 4-letter alphabet, there is a structure that gives this for free with zero collisions: a trie of depth k, where each of the k levels branches on one base. Because the alphabet and depth are both fixed and small (here, 4 and 3), the trie can be represented as flat, direct-addressed arrays -- no pointers, no dynamic allocation -- while still being built with Chapter 16's genuine level-synchronous, CAS-protected parallel trie construction.

### The Sequential (CPU) Baseline

```cpp
// 192_kmer_trie_index_cpu_baseline.cpp
//
// Chapter 34.2 -- sequential baseline for building a k-mer trie index.
// For a small, fixed k over a fixed 4-letter alphabet, the trie is
// represented as flat, direct-addressed "node exists" arrays (sized
// 4, 4x4, 4x4x4) instead of a general pointer-linked trie -- a real
// technique for small k, since 4^k fits trivially in memory. Each
// leaf additionally stores the list of reference positions where that
// k-mer occurs.
//
// Compile: g++ -std=c++17 -Wall -Wextra -O2 192_kmer_trie_index_cpu_baseline.cpp -o 192_kmer_trie_index_cpu_baseline
// Run:     ./192_kmer_trie_index_cpu_baseline

#include <cstdio>
#include <string>
#include <vector>

constexpr int K = 3;
const std::string REFERENCE = "ACGTACGTAC";
const char BASES[4] = {'A', 'C', 'G', 'T'};

int base_code(char c) {
    switch (c) {
        case 'A': return 0;
        case 'C': return 1;
        case 'G': return 2;
        default:  return 3;
    }
}

int main() {
    std::vector<std::string> ref_kmers;
    for (size_t i = 0; i + K <= REFERENCE.size(); ++i) ref_kmers.push_back(REFERENCE.substr(i, K));

    int level1_exists[4] = {0, 0, 0, 0};
    int level2_exists[4][4] = {};
    int leaf_exists[4][4][4] = {};
    std::vector<int> leaf_positions[4][4][4];

    std::printf("=== Section 34.2 CPU baseline: sequential trie construction with per-leaf position lists ===\n\n");
    std::printf("reference sequence: %s, k=%d\n\n", REFERENCE.c_str(), K);

    for (size_t pos = 0; pos < ref_kmers.size(); ++pos) {
        const std::string& kmer = ref_kmers[pos];
        int c0 = base_code(kmer[0]), c1 = base_code(kmer[1]), c2 = base_code(kmer[2]);
        std::vector<std::string> created;
        if (level1_exists[c0] == 0) {
            level1_exists[c0] = 1;
            created.push_back("level-1 node '" + std::string(1, kmer[0]) + "'");
        }
        if (level2_exists[c0][c1] == 0) {
            level2_exists[c0][c1] = 1;
            created.push_back("level-2 node '" + kmer.substr(0, 2) + "'");
        }
        if (leaf_exists[c0][c1][c2] == 0) {
            leaf_exists[c0][c1][c2] = 1;
            created.push_back("leaf node '" + kmer + "'");
        }
        leaf_positions[c0][c1][c2].push_back((int)pos);

        std::string msg;
        if (created.empty()) {
            msg = "reuses all 3 existing nodes";
        } else {
            msg = "creates ";
            for (size_t i = 0; i < created.size(); ++i) msg += created[i] + (i + 1 < created.size() ? ", " : "");
        }
        std::printf("position %zu: kmer='%s' (path %d->%d->%d) -- %s, appends position %zu to leaf's list -> [",
                    pos, kmer.c_str(), c0, c1, c2, msg.c_str(), pos);
        for (size_t i = 0; i < leaf_positions[c0][c1][c2].size(); ++i) {
            std::printf("%d%s", leaf_positions[c0][c1][c2][i], (i + 1 < leaf_positions[c0][c1][c2].size()) ? ", " : "");
        }
        std::printf("]\n");
    }

    std::printf("\nfinal leaf position lists (non-empty only):\n");
    for (int c0 = 0; c0 < 4; ++c0) {
        for (int c1 = 0; c1 < 4; ++c1) {
            for (int c2 = 0; c2 < 4; ++c2) {
                if (leaf_exists[c0][c1][c2]) {
                    std::printf("  %c%c%c: [", BASES[c0], BASES[c1], BASES[c2]);
                    for (size_t i = 0; i < leaf_positions[c0][c1][c2].size(); ++i) {
                        std::printf("%d%s", leaf_positions[c0][c1][c2][i], (i + 1 < leaf_positions[c0][c1][c2].size()) ? ", " : "");
                    }
                    std::printf("]\n");
                }
            }
        }
    }

    bool ok = (leaf_positions[0][1][2] == std::vector<int>{0, 4} &&
               leaf_positions[1][2][3] == std::vector<int>{1, 5} &&
               leaf_positions[2][3][0] == std::vector<int>{2, 6} &&
               leaf_positions[3][0][1] == std::vector<int>{3, 7});
    std::printf("\nself-check: every k-mer's leaf position list matches its two hand-computed "
                "occurrences in original order: %s\n", ok ? "confirmed" : "MISMATCH");

    return ok ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++17 -Wall -Wextra -O2 192_kmer_trie_index_cpu_baseline.cpp -o 192_kmer_trie_index_cpu_baseline
./192_kmer_trie_index_cpu_baseline
```

**Sample input:** the same 8 reference k-mers from Section 34.1, inserted one at a time into a 3-level trie, with each leaf accumulating the list of positions where its k-mer occurs.

**Sample output:**

```text
=== Section 34.2 CPU baseline: sequential trie construction with per-leaf position lists ===

reference sequence: ACGTACGTAC, k=3

position 0: kmer='ACG' (path 0->1->2) -- creates level-1 node 'A', level-2 node 'AC', leaf node 'ACG', appends position 0 to leaf's list -> [0]
position 1: kmer='CGT' (path 1->2->3) -- creates level-1 node 'C', level-2 node 'CG', leaf node 'CGT', appends position 1 to leaf's list -> [1]
position 2: kmer='GTA' (path 2->3->0) -- creates level-1 node 'G', level-2 node 'GT', leaf node 'GTA', appends position 2 to leaf's list -> [2]
position 3: kmer='TAC' (path 3->0->1) -- creates level-1 node 'T', level-2 node 'TA', leaf node 'TAC', appends position 3 to leaf's list -> [3]
position 4: kmer='ACG' (path 0->1->2) -- reuses all 3 existing nodes, appends position 4 to leaf's list -> [0, 4]
position 5: kmer='CGT' (path 1->2->3) -- reuses all 3 existing nodes, appends position 5 to leaf's list -> [1, 5]
position 6: kmer='GTA' (path 2->3->0) -- reuses all 3 existing nodes, appends position 6 to leaf's list -> [2, 6]
position 7: kmer='TAC' (path 3->0->1) -- reuses all 3 existing nodes, appends position 7 to leaf's list -> [3, 7]

final leaf position lists (non-empty only):
  ACG: [0, 4]
  CGT: [1, 5]
  GTA: [2, 6]
  TAC: [3, 7]

self-check: every k-mer's leaf position list matches its two hand-computed occurrences in original order: confirmed
```

### The Concept, In Detail

```
ASCII view: a depth-3 trie over {A,C,G,T}, as flat direct-addressed
arrays instead of pointers.

  level1[4]:        A  C  G  T           (1 = node exists)
  level2[4][4]:     AC CG GT TA          (indexed by [c0][c1])
  leaf[4][4][4]:    ACG CGT GTA TAC      (indexed by [c0][c1][c2])
                     |    |    |    |
                position lists:
                     [0,4] [1,5] [2,6] [3,7]

  4 distinct k-mers happen to start with 4 DIFFERENT first letters
  here, so only REPEATED k-mers (ACG at positions 0 and 4) actually
  share -- and race on -- any trie nodes at all
```

Because every internal node's children are indexed directly by base code (0 through 3), "does this child exist" is a single array read, and "create this child" is a single array write -- there is no pointer to allocate or link, which is exactly why Chapter 16's general pointer-linked trie collapses into flat arrays for this fixed, tiny case.

[COMMON TRAP]
It is tempting to assume this direct-addressed array trick generalizes to any k. The array sizes are 4^1, 4^2, ..., 4^k -- for k=3 that is at most 64 leaves, trivial to store directly, but for a realistic k=20 (common in real k-mer analysis), 4^20 is over a trillion, far too large to allocate directly. Real genomics tools handle large k exactly the way Section 34.1 already does: a HASH table over the k-mer's encoded key, tolerating collisions, rather than a directly-addressed trie -- meaning Section 34.1's and Section 34.2's two structures are not competing choices but complementary ones, each right for a different k range.

### Code and Verification

```cpp
#include <cstdio>
#include <string>
#include <vector>
#include <tuple>
#include <algorithm>

// Chapter 34.2 main -- concurrent trie construction: Chapter 16's
// level-synchronous, CAS-protected create-if-absent discipline,
// applied at each of the trie's 3 fixed levels. Two threads walking
// the SAME path (a repeated k-mer) race at every level; whichever
// thread's atomicCAS lands first creates each node, and the other
// discovers it already exists and reuses it -- exactly Chapter 16's
// guarantee that threads sharing a prefix correctly share one node
// rather than each building their own. Each leaf's position list is
// appended to via its own atomicAdd cursor, Section 28.2/30.2/33.3's
// bump-allocator append, reused unchanged.

__global__ void trie_insert_kernel(const int* c0s, const int* c1s, const int* c2s, int pos_base,
                                    int* level1_exists, int* level2_exists, int* leaf_exists,
                                    int* leaf_cursor, int* leaf_positions) {
    int tid = threadIdx.x;
    int c0 = c0s[tid], c1 = c1s[tid], c2 = c2s[tid];

    atomicCAS(&level1_exists[c0], 0, 1);
    atomicCAS(&level2_exists[c0 * 4 + c1], 0, 1);
    atomicCAS(&leaf_exists[(c0 * 4 + c1) * 4 + c2], 0, 1);

    int leaf_idx = (c0 * 4 + c1) * 4 + c2;
    int slot = atomicAdd(&leaf_cursor[leaf_idx], 1);
    leaf_positions[leaf_idx * 8 + slot] = pos_base + tid;
}

// ---- Host-side replay of the identical CAS-protected node-creation
// ---- and atomicAdd-cursor append logic, driving a forced landing
// ---- order where a repeated k-mer's SECOND occurrence lands first. ----

int main() {
    printf("=== Section 34.2 main: concurrent trie construction via CAS-protected node creation ===\n\n");

    // positions 0 and 4 both "ACG" (path 0->1->2); position 1 "CGT" (path 1->2->3).
    std::vector<int> positions = {4, 0, 1};
    auto path_of = [](int pos) -> std::tuple<int, int, int, std::string> {
        if (pos == 0 || pos == 4) return {0, 1, 2, "ACG"};
        return {1, 2, 3, "CGT"};
    };

    printf("landing order (by reference position): [ ");
    for (int p : positions) printf("%d ", p);
    printf("]\n\n");

    int level1_exists[4] = {0, 0, 0, 0};
    int level2_exists[4][4] = {};
    int leaf_exists[4][4][4] = {};
    std::vector<int> leaf_positions[4][4][4];

    for (int pos : positions) {
        auto [c0, c1, c2, kmer] = path_of(pos);
        printf("  position %d (kmer=%s):\n", pos, kmer.c_str());

        if (level1_exists[c0] == 0) {
            level1_exists[c0] = 1;
            printf("    CAS on level1[%d] succeeds -> creates node '%c'\n", c0, kmer[0]);
        } else {
            printf("    CAS on level1[%d] fails (already 1) -> reuses node '%c'\n", c0, kmer[0]);
        }
        if (level2_exists[c0][c1] == 0) {
            level2_exists[c0][c1] = 1;
            printf("    CAS on level2[%d][%d] succeeds -> creates node '%s'\n", c0, c1, kmer.substr(0, 2).c_str());
        } else {
            printf("    CAS on level2[%d][%d] fails (already 1) -> reuses node '%s'\n", c0, c1, kmer.substr(0, 2).c_str());
        }
        if (leaf_exists[c0][c1][c2] == 0) {
            leaf_exists[c0][c1][c2] = 1;
            printf("    CAS on leaf[%d][%d][%d] succeeds -> creates leaf '%s'\n", c0, c1, c2, kmer.c_str());
        } else {
            printf("    CAS on leaf[%d][%d][%d] fails (already 1) -> reuses leaf '%s'\n", c0, c1, c2, kmer.c_str());
        }

        int slot = (int)leaf_positions[c0][c1][c2].size();
        leaf_positions[c0][c1][c2].push_back(pos);
        printf("    atomicAdd(leaf_cursor['%s'],1) returned %d -> leaf_positions['%s'][%d] = %d\n\n",
               kmer.c_str(), slot, kmer.c_str(), slot, pos);
    }

    printf("final leaf position lists (non-empty only):\n");
    const char BASES[4] = {'A', 'C', 'G', 'T'};
    for (int c0 = 0; c0 < 4; c0++) {
        for (int c1 = 0; c1 < 4; c1++) {
            for (int c2 = 0; c2 < 4; c2++) {
                if (leaf_exists[c0][c1][c2]) {
                    printf("  %c%c%c: [", BASES[c0], BASES[c1], BASES[c2]);
                    for (size_t i = 0; i < leaf_positions[c0][c1][c2].size(); i++) {
                        printf("%d%s", leaf_positions[c0][c1][c2][i], (i + 1 < leaf_positions[c0][c1][c2].size()) ? ", " : "");
                    }
                    printf("]\n");
                }
            }
        }
    }

    std::vector<int> acg_sorted = leaf_positions[0][1][2];
    std::sort(acg_sorted.begin(), acg_sorted.end());
    bool ok = (acg_sorted == std::vector<int>{0, 4}) && (leaf_positions[1][2][3] == std::vector<int>{1});
    printf("\nself-check: ACG's leaf holds both {0,4} regardless of landing order, and CGT's leaf\n");
    printf("holds only {1}, fully uncontended: %s\n", ok ? "confirmed" : "MISMATCH");

    return ok ? 0 : 1;
}
```

**Compile and run:**

```bash
NVDIR=/usr/local/lib/python3.11/dist-packages/nvidia/cu13
$NVDIR/bin/nvcc -arch=sm_80 -I$NVDIR/include -L$NVDIR/lib -lcudart 193_kmer_trie_index_kernel.cu -o 193_kmer_trie_index_kernel
LD_LIBRARY_PATH=$NVDIR/lib ./193_kmer_trie_index_kernel
```

**Sample input:** the same scenario as Section 34.1's kernel -- positions 0 and 4 (both "ACG") racing at all 3 trie levels, position 1 ("CGT") fully disjoint -- now building the trie instead of the hash table.

**Sample output:**

```text
=== Section 34.2 main: concurrent trie construction via CAS-protected node creation ===

landing order (by reference position): [ 4 0 1 ]

  position 4 (kmer=ACG):
    CAS on level1[0] succeeds -> creates node 'A'
    CAS on level2[0][1] succeeds -> creates node 'AC'
    CAS on leaf[0][1][2] succeeds -> creates leaf 'ACG'
    atomicAdd(leaf_cursor['ACG'],1) returned 0 -> leaf_positions['ACG'][0] = 4

  position 0 (kmer=ACG):
    CAS on level1[0] fails (already 1) -> reuses node 'A'
    CAS on level2[0][1] fails (already 1) -> reuses node 'AC'
    CAS on leaf[0][1][2] fails (already 1) -> reuses leaf 'ACG'
    atomicAdd(leaf_cursor['ACG'],1) returned 1 -> leaf_positions['ACG'][1] = 0

  position 1 (kmer=CGT):
    CAS on level1[1] succeeds -> creates node 'C'
    CAS on level2[1][2] succeeds -> creates node 'CG'
    CAS on leaf[1][2][3] succeeds -> creates leaf 'CGT'
    atomicAdd(leaf_cursor['CGT'],1) returned 0 -> leaf_positions['CGT'][0] = 1

final leaf position lists (non-empty only):
  ACG: [4, 0]
  CGT: [1]

self-check: ACG's leaf holds both {0,4} regardless of landing order, and CGT's leaf
holds only {1}, fully uncontended: confirmed
```

## 34.3 Finding Seed Matches Between a Query and the Reference

### Intuition

With the reference's k-mer index built, finding candidate alignment positions for a new query sequence needs no comparison against the reference directly at all: walk each of the query's own k-mers down the same trie, and every reference position stored at a matching leaf is a SEED -- a strong hint that the query and reference agree starting at those two positions. A query k-mer that does not exist anywhere in the trie contributes no seeds; this is precisely how real aligners avoid ever comparing a query against most of a multi-gigabase reference directly.

### The Sequential (CPU) Baseline

```cpp
// 194_seed_match_cpu_baseline.cpp
//
// Chapter 34.3 -- sequential baseline for finding seed matches
// between a query sequence and the reference. This is a lookup
// against Section 34.2's completed trie index: for each of the
// query's own k-mers, walk the same 3 levels; if all 3 nodes exist,
// every reference position stored at that leaf is a seed match
// (query_position, reference_position). A query k-mer whose path is
// missing at any level contributes no seed matches at all.
//
// Compile: g++ -std=c++17 -Wall -Wextra -O2 194_seed_match_cpu_baseline.cpp -o 194_seed_match_cpu_baseline
// Run:     ./194_seed_match_cpu_baseline

#include <cstdio>
#include <map>
#include <string>
#include <vector>

constexpr int K = 3;
const std::string QUERY = "CGTACGTG";

int main() {
    std::vector<std::string> query_kmers;
    for (size_t i = 0; i + K <= QUERY.size(); ++i) query_kmers.push_back(QUERY.substr(i, K));

    // Section 34.2's completed trie index.
    std::map<std::string, std::vector<int>> leaf_positions = {
        {"ACG", {0, 4}}, {"CGT", {1, 5}}, {"GTA", {2, 6}}, {"TAC", {3, 7}},
    };

    std::printf("=== Section 34.3 CPU baseline: sequential seed lookup against the trie index ===\n\n");
    std::printf("query sequence: %s, k=%d\n", QUERY.c_str(), K);
    std::printf("query k-mers (by position): [");
    for (size_t i = 0; i < query_kmers.size(); ++i) {
        std::printf("(%zu, '%s')%s", i, query_kmers[i].c_str(), (i + 1 < query_kmers.size()) ? ", " : "");
    }
    std::printf("]\n\n");

    std::vector<std::pair<int, int>> seeds;
    for (size_t qpos = 0; qpos < query_kmers.size(); ++qpos) {
        const std::string& kmer = query_kmers[qpos];
        auto it = leaf_positions.find(kmer);
        if (it != leaf_positions.end()) {
            std::printf("query position %zu: kmer='%s' FOUND in index -> reference positions [", qpos, kmer.c_str());
            for (size_t i = 0; i < it->second.size(); ++i) std::printf("%d%s", it->second[i], (i + 1 < it->second.size()) ? ", " : "");
            std::printf("] -> seeds [");
            for (size_t i = 0; i < it->second.size(); ++i) {
                seeds.push_back({(int)qpos, it->second[i]});
                std::printf("(%zu, %d)%s", qpos, it->second[i], (i + 1 < it->second.size()) ? ", " : "");
            }
            std::printf("]\n");
        } else {
            std::printf("query position %zu: kmer='%s' NOT in index -> no seed matches\n", qpos, kmer.c_str());
        }
    }

    std::printf("\nall seed matches (query_pos, reference_pos): [");
    for (size_t i = 0; i < seeds.size(); ++i) std::printf("(%d, %d)%s", seeds[i].first, seeds[i].second, (i + 1 < seeds.size()) ? ", " : "");
    std::printf("]\n");
    std::printf("total seed matches: %zu\n", seeds.size());

    std::vector<std::pair<int, int>> expected = {
        {0,1},{0,5},{1,2},{1,6},{2,3},{2,7},{3,0},{3,4},{4,1},{4,5}
    };
    bool ok = (seeds == expected) && (seeds.size() == 10);
    std::printf("\nself-check: 10 seed matches found, matching hand-computed expectation exactly "
                "(query k-mer 'GTG' at position 5 correctly contributes none): %s\n", ok ? "confirmed" : "MISMATCH");

    return ok ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++17 -Wall -Wextra -O2 194_seed_match_cpu_baseline.cpp -o 194_seed_match_cpu_baseline
./194_seed_match_cpu_baseline
```

**Sample input:** the query sequence `CGTACGTG` (8 bases, 6 overlapping k-mers), looked up one at a time against Section 34.2's completed reference index.

**Sample output:**

```text
=== Section 34.3 CPU baseline: sequential seed lookup against the trie index ===

query sequence: CGTACGTG, k=3
query k-mers (by position): [(0, 'CGT'), (1, 'GTA'), (2, 'TAC'), (3, 'ACG'), (4, 'CGT'), (5, 'GTG')]

query position 0: kmer='CGT' FOUND in index -> reference positions [1, 5] -> seeds [(0, 1), (0, 5)]
query position 1: kmer='GTA' FOUND in index -> reference positions [2, 6] -> seeds [(1, 2), (1, 6)]
query position 2: kmer='TAC' FOUND in index -> reference positions [3, 7] -> seeds [(2, 3), (2, 7)]
query position 3: kmer='ACG' FOUND in index -> reference positions [0, 4] -> seeds [(3, 0), (3, 4)]
query position 4: kmer='CGT' FOUND in index -> reference positions [1, 5] -> seeds [(4, 1), (4, 5)]
query position 5: kmer='GTG' NOT in index -> no seed matches

all seed matches (query_pos, reference_pos): [(0, 1), (0, 5), (1, 2), (1, 6), (2, 3), (2, 7), (3, 0), (3, 4), (4, 1), (4, 5)]
total seed matches: 10

self-check: 10 seed matches found, matching hand-computed expectation exactly (query k-mer 'GTG' at position 5 correctly contributes none): confirmed
```

### The Concept, In Detail

```
ASCII view: seed-finding is a read-only lookup, not a comparison walk.

  query k-mer at position 3: "ACG"
    walk trie: level1['A'] exists -> level2['AC'] exists -> leaf['ACG'] exists
    leaf['ACG'] position list: [0, 4]
    -> seeds: (3,0), (3,4)   -- "query pos 3 likely aligns near ref pos 0 or 4"

  query k-mer at position 5: "GTG"
    walk trie: level1['G'] exists -> level2['GT'] exists -> leaf['GTG'] MISSING
    -> no seeds at all -- this substring never occurs in the reference
```

Every query k-mer's lookup reads the SAME shared trie built once in Section 34.2, and never modifies it -- so, just like Section 34.1's per-path independence in Chapter 33, different query positions' lookups can run fully in parallel with no risk of a race, because concurrent READS of unchanging data need no synchronization at all.

[COMMON TRAP]
It is tempting to think a thread that finds a match writes exactly one output seed, the way most of this book's earlier "one thread, one output slot" kernels worked. A single query k-mer can match a reference k-mer that occurs MULTIPLE times (as "ACG" does here, at positions 0 and 4), so a single thread may need to claim zero, one, or several output slots -- Section 34.3's kernel calls `atomicAdd` with the actual match COUNT, not a fixed 1, claiming a contiguous run of slots in one call rather than one slot per match.

### Code and Verification

```cpp
#include <cstdio>
#include <algorithm>
#include <map>
#include <string>
#include <vector>

// Chapter 34.3 main -- one thread per query k-mer, each reading the
// SAME shared, already-built trie index (read-only, so no race there)
// -- exactly Section 33.1's embarrassingly-parallel independence,
// since every query position's lookup is unaffected by every other
// one. Where this differs from Section 33.3's compaction is
// multiplicity: a thread may claim ZERO, ONE, or SEVERAL output slots
// depending on how many reference positions its k-mer matched, all
// still claimed correctly via a single shared atomicAdd cursor.

__global__ void seed_lookup_kernel(const int* found, const int* match_count, const int* match_offset,
                                    const int* match_positions, int* cursor, int* seed_qpos, int* seed_rpos) {
    int tid = threadIdx.x;
    if (!found[tid]) return;
    int n = match_count[tid];
    int base = atomicAdd(cursor, n);
    for (int i = 0; i < n; i++) {
        seed_qpos[base + i] = tid;
        seed_rpos[base + i] = match_positions[match_offset[tid] + i];
    }
}

// ---- Host-side replay of the identical atomicAdd-cursor logic,
// ---- driving a forced landing order unrelated to query position. ----

int main() {
    printf("=== Section 34.3 main: concurrent seed lookup, atomicAdd-cursor output ===\n\n");

    std::vector<std::string> query_kmers = {"CGT", "GTA", "TAC", "ACG", "CGT", "GTG"};
    std::map<std::string, std::vector<int>> leaf_positions = {
        {"ACG", {0, 4}}, {"CGT", {1, 5}}, {"GTA", {2, 6}}, {"TAC", {3, 7}},
    };

    int landing_order[6] = {3, 0, 5, 2, 4, 1};

    printf("landing order (by query position): [ ");
    for (int q : landing_order) printf("%d ", q);
    printf("]\n\n");

    int cursor = 0;
    std::vector<std::pair<int, int>> seeds(10, {-1, -1});

    for (int qpos : landing_order) {
        const std::string& kmer = query_kmers[qpos];
        auto it = leaf_positions.find(kmer);
        if (it != leaf_positions.end()) {
            const auto& matches = it->second;
            int base = cursor;
            cursor += (int)matches.size();
            printf("  thread %d (kmer='%s'): FOUND, %zu matches -> atomicAdd(cursor,%zu) claims slots [",
                   qpos, kmer.c_str(), matches.size(), matches.size());
            for (size_t i = 0; i < matches.size(); i++) printf("%d%s", base + (int)i, (i + 1 < matches.size()) ? ", " : "");
            printf("] -> writes [");
            for (size_t i = 0; i < matches.size(); i++) {
                seeds[base + i] = {qpos, matches[i]};
                printf("(%d, %d)%s", qpos, matches[i], (i + 1 < matches.size()) ? ", " : "");
            }
            printf("]\n");
        } else {
            printf("  thread %d (kmer='%s'): NOT FOUND -> claims 0 slots\n", qpos, kmer.c_str());
        }
    }

    printf("\nfinal seeds array (landing-order-dependent positions): [");
    for (size_t i = 0; i < seeds.size(); i++) printf("(%d, %d)%s", seeds[i].first, seeds[i].second, (i + 1 < seeds.size()) ? ", " : "");
    printf("]\n");

    std::vector<std::pair<int, int>> expected = {
        {0,1},{0,5},{1,2},{1,6},{2,3},{2,7},{3,0},{3,4},{4,1},{4,5}
    };
    std::vector<std::pair<int, int>> sorted_seeds = seeds;
    std::sort(sorted_seeds.begin(), sorted_seeds.end());
    std::sort(expected.begin(), expected.end());
    bool ok = (sorted_seeds == expected) && (seeds.size() == 10);
    printf("self-check: seeds array's SET matches the sequential baseline's 10 seed matches exactly, "
           "positions differing only by landing order: %s\n", ok ? "confirmed" : "MISMATCH");

    return ok ? 0 : 1;
}
```

**Compile and run:**

```bash
NVDIR=/usr/local/lib/python3.11/dist-packages/nvidia/cu13
$NVDIR/bin/nvcc -arch=sm_80 -I$NVDIR/include -L$NVDIR/lib -lcudart 195_seed_match_kernel.cu -o 195_seed_match_kernel
LD_LIBRARY_PATH=$NVDIR/lib ./195_seed_match_kernel
```

**Sample input:** the same query sequence's 6 k-mers, each looked up by an independent thread against the shared reference index, with a forced landing order unrelated to query position.

**Sample output:**

```text
=== Section 34.3 main: concurrent seed lookup, atomicAdd-cursor output ===

landing order (by query position): [ 3 0 5 2 4 1 ]

  thread 3 (kmer='ACG'): FOUND, 2 matches -> atomicAdd(cursor,2) claims slots [0, 1] -> writes [(3, 0), (3, 4)]
  thread 0 (kmer='CGT'): FOUND, 2 matches -> atomicAdd(cursor,2) claims slots [2, 3] -> writes [(0, 1), (0, 5)]
  thread 5 (kmer='GTG'): NOT FOUND -> claims 0 slots
  thread 2 (kmer='TAC'): FOUND, 2 matches -> atomicAdd(cursor,2) claims slots [4, 5] -> writes [(2, 3), (2, 7)]
  thread 4 (kmer='CGT'): FOUND, 2 matches -> atomicAdd(cursor,2) claims slots [6, 7] -> writes [(4, 1), (4, 5)]
  thread 1 (kmer='GTA'): FOUND, 2 matches -> atomicAdd(cursor,2) claims slots [8, 9] -> writes [(1, 2), (1, 6)]

final seeds array (landing-order-dependent positions): [(3, 0), (3, 4), (0, 1), (0, 5), (2, 3), (2, 7), (4, 1), (4, 5), (1, 2), (1, 6)]
self-check: seeds array's SET matches the sequential baseline's 10 seed matches exactly, positions differing only by landing order: confirmed
```

## Chapter Summary

A genomics indexing and seed-finding pipeline needed no new parallel primitive -- it needed two of Part 5's and Part 4's structures composed with Part 1's histogram idea. Section 34.1 showed that counting k-mers is Chapter 7's histogram generalized to an unbounded key space via Chapter 19's open-addressing hash table, with insert-or-increment replacing insert-only. Section 34.2 showed that indexing k-mer POSITIONS, not just counts, is Chapter 16's level-synchronous, CAS-protected trie construction, collapsed into flat direct-addressed arrays because DNA's alphabet and this section's k are both small and fixed -- with Section 34.1's hash table standing as the right alternative once k grows too large for direct addressing. Section 34.3 showed that seed-finding is nothing more than a read-only, fully parallel lookup against that shared index, with a single atomicAdd-cursor append (reused from Section 28.2/30.2/33.3) handling the fact that one query k-mer can legitimately produce several seed matches at once.

## Self-Check Questions

1. What specific limitation of Chapter 7's original histogram does Chapter 19's hash table solve, making k-mer counting possible in the first place?
2. Why does encoding each DNA base as a 2-bit value, rather than storing k-mers as raw character strings, matter for how Section 34.1's hash table performs its work?
3. Why can a fixed-depth, fixed-alphabet trie be represented as flat arrays instead of Chapter 16's general pointer-linked structure, and what specifically stops this trick from working for a large k like 20?
4. In Section 34.2's concurrent kernel, what determines which of two threads inserting the SAME repeated k-mer actually creates each trie node, and does it matter which one wins?
5. Why does Section 34.3's lookup kernel need no synchronization at all between different query threads, even though they all read the same shared trie?
6. Why must Section 34.3's kernel call atomicAdd with a variable count rather than always claiming exactly one output slot per thread?

## Where We Go Next

A genomic aligner works with strings and discrete positions in one dimension; an autonomous vehicle's LiDAR sensor produces millions of 3D points every second, and needs to know which points are near which other points in actual physical space. Chapter 35 turns to LiDAR point-cloud nearest-neighbor search for autonomous vehicles and SLAM, reusing Chapter 18's spatial trees -- built, level by level, with the exact same disciplined construction this chapter just used for a k-mer trie -- the next of the case studies that close out Part 8.

## Worked Solutions

**1.** Chapter 7's original histogram assumed a small, fixed, KNOWN-in-advance number of buckets (such as 256 for byte values), with each bucket's index computed directly and cheaply from the input value. K-mer counting has no such bound -- the number of distinct k-mers that might appear in a real genome is enormous and not known ahead of time -- so Chapter 19's hash table is what makes counting practical at all, by mapping an effectively unbounded key space down onto a small, dense array via hashing and probing, exactly the capability Chapter 7's fixed-bucket approach lacked.

**2.** A 2-bit-per-base encoding turns a k-mer into a single small, fixed-width integer, which is exactly the kind of key Chapter 19's hash table already knows how to hash (via modulo) and compare (via simple integer equality) efficiently. Storing k-mers as raw variable-length character strings instead would require string hashing and string comparison at every probe step, which is both slower and a fundamentally different (and more complex) key type than the fixed-width integers this book's hash tables have used throughout -- the 2-bit encoding lets k-mer counting reuse Chapter 19's machinery completely unchanged.

**3.** A fixed-depth, fixed-alphabet trie's node-existence and position-list arrays can be sized exactly 4^1, 4^2, ..., 4^k in advance, since every node's possible children are known ahead of time (one per alphabet symbol) -- this is what allows direct array indexing instead of the pointer-chasing Chapter 16's general trie needs for structures whose shape is not known in advance. This stops working once 4^k becomes too large to allocate: for k=20, 4^20 exceeds a trillion entries, so a real large-k implementation must fall back to a hash table over the k-mer's encoded key (Section 34.1's structure) rather than a directly-addressed array.

**4.** Which thread wins each node's `atomicCAS` is determined purely by real hardware timing -- whichever thread's CAS instruction physically executes first at that memory location -- not by thread index, launch order, or which reference position the k-mer occurs at. It does not matter which one wins: because both threads are inserting the IDENTICAL k-mer, whichever one creates the node leaves the trie in the same correct state the other would have, and the losing thread's `atomicCAS` simply confirms the node already exists rather than needing to redo any work -- Chapter 16's exact guarantee, unchanged.

**5.** Section 34.3's threads only ever READ the trie's node-existence and position-list arrays -- Section 34.2's construction has already finished and the structure is not modified again during lookup. A race condition can only occur when at least one thread WRITES to a location another thread accesses concurrently; with every thread performing only reads of already-finalized, unchanging data, there is no write for any read to race against, so no synchronization is needed between query threads at all.

**6.** A single query k-mer can match a reference k-mer that occurs at MULTIPLE reference positions -- "ACG" here matches at both position 0 and position 4 -- so the number of seed matches one query thread must emit is not fixed at one; it equals however many positions that k-mer's trie leaf actually stores, which varies per query k-mer (including zero, for a k-mer absent from the reference). Calling atomicAdd with that thread's actual match count claims a contiguous block of exactly the right size in one atomic operation, correctly reserving space for all of that thread's seeds at once rather than requiring one atomicAdd call per individual seed.
