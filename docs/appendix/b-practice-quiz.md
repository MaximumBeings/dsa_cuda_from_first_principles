# Appendix B: Practice Quiz

Every chapter in this book already ends with six Self-Check Questions and six Worked Solutions of its own, testing that chapter's own material in isolation. This appendix is different: it tests whether the ideas actually connect across chapters -- whether you can look at a hash table's deletion bug and recognize it as the same shape of problem a lock-free stack's ABA bug is, or look at Chapter 36's arbitrage detector and name which three earlier, unrelated-looking chapters it was quietly built from. B.2 asks eighteen conceptual questions, two from each of this book's nine Parts, with answers in B.3. B.4 raises the stakes: three short, genuinely compiled programs, each built around a specific mistake this book warned about somewhere -- read the code, commit to a prediction, then compile and run it yourself before reading the revealed output.

## B.1 How to Use This Quiz

Answer every question in B.2 in your own words -- out loud, or on paper -- before reading B.3's answer. For B.4, do the same with actual code: read the program, write down what you believe it will print, and only then compile it and compare. A prediction you get wrong tells you exactly which chapter to reread; a prediction you get right by guessing, without being able to explain why, is worth treating the same way. Every one of B.4's programs compiles and runs in seconds -- there is no reason to skip the "compile and run it yourself" half of the exercise, and every reason not to: this book's entire method has been "genuinely compiled, genuinely run, never assumed," and this quiz is not an exception.

## B.2 Conceptual Review Questions

**Part 0 -- GPU Foundations (Chapters 1-3)**

1. Why does a data structure's own internal representation often need to change when moving it from a CPU to a GPU, even when the abstract data type (what operations it supports) stays exactly the same?
2. In the work-span model, what does "span" measure, and why can two algorithms with identical total work still have very different actual runtimes on a GPU?

**Part 1 -- Parallel Primitives (Chapters 4-7)**

3. Why is it incorrect for every thread in a naive parallel reduction to simply add its own value into one shared accumulator variable, and what genuinely fixes it?
4. Stream compaction and histogram-building look like different problems on the surface. What single earlier primitive does each one ultimately depend on to compute where its output values belong?

**Part 2 -- Linear Structures Under Concurrency (Chapters 8-11)**

5. Why is protecting a stack's push and pop with a single mutex not a workable general answer on a GPU running thousands of concurrent threads, even though it is a perfectly correct answer on a CPU with a handful of threads?
6. What problem does pointer jumping (list ranking) solve that a straightforward sequential traversal of a linked list cannot solve in parallel at all?

**Part 3 -- Sorting (Chapters 12-14)**

7. Bitonic sort does more total comparisons than an optimal comparison sort (O(n log^2 n) instead of O(n log n)). Why is it still a common first choice for sorting on a GPU?
8. What is the key idea that lets radix sort avoid ever comparing two elements directly, and which earlier Part's primitive does each of its digit passes depend on?

**Part 4 -- Trees (Chapters 15-18)**

9. Why must a recursive tree traversal be rewritten around an explicit stack before it can run on a GPU at all?
10. A segment tree (or Fenwick tree) and a hash table are both ways of organizing data for fast lookup. What kind of query is each one actually built to answer, and why can't the other structure answer it as well?

**Part 5 -- Hash Tables (Chapters 19-20)**

11. Why does deleting a key from an open-addressing hash table require a tombstone marker instead of simply resetting that slot back to empty?
12. What guarantee does cuckoo hashing provide about worst-case lookup time that open addressing with linear probing cannot make?

**Part 6 -- Graphs (Chapters 21-25)**

13. Why is CSR (Compressed Sparse Row) the preferred representation for a large, sparse graph on a GPU, instead of a plain adjacency matrix?
14. What is the "frontier" in a parallel breadth-first search, and why does processing it one whole level at a time parallelize well?

**Part 7 -- Priority Structures and Concurrency (Chapters 26-28)**

15. A classic binary heap represents parent-child relationships with array-index arithmetic rather than pointers. Why does that structure still become awkward once thousands of threads try to insert into it concurrently?
16. What specific problem does a memory pool (a custom allocator) solve that letting every thread call a general-purpose `malloc`/`free` directly does not?

**Part 8 -- Case Studies (Chapters 29-36)**

17. Name two Part 8 case studies that both reused Chapter 9's CAS-retry discipline, and describe what each one specifically used it for.
18. Why does Chapter 36's arbitrage detector use Bellman-Ford instead of Dijkstra, and which earlier chapter's technique does it reuse to confirm that a detected node genuinely lies on a cycle?

## B.3 Conceptual Review Answers

**1.** A GPU's performance model rewards thousands of threads doing regular, independent, coalesced memory accesses far more than it rewards any single thread's own cleverness -- a representation built around one thread doing sequential pointer-chasing (a classic CPU linked list, say) forces every other thread to sit idle waiting for that one dependency chain, while an array-based or otherwise flattened representation lets many threads touch different parts of the structure at once. The data type's contract (what `push`, `find`, or `insert` mean) is unchanged; what changes is which physical layout lets many threads exercise that contract simultaneously without serializing on each other.

**2.** Span measures the length of the longest chain of operations that must happen strictly in sequence -- the critical path -- regardless of how many processors are available. Two algorithms can perform the exact same total amount of work while having very different spans: one arranges that work into a short, wide dependency chain (many independent steps that can run simultaneously), while the other arranges the identical amount of work into a long, narrow chain where each step depends on the one before it. On a machine with enough processors, the low-span algorithm finishes much faster, because span, not total work, becomes the actual runtime bottleneck.

**3.** When many threads simultaneously read the shared accumulator, add their own value, and write the result back, two threads' read-modify-write sequences can interleave: both read the same old value, both compute their own new value from it, and whichever writes last overwrites the other thread's contribution entirely, silently losing an update. The fix is a proper reduction: either a pairwise tree reduction (each round, half the remaining values combine into the other half, with no two threads ever writing the same location in the same round) or an atomic add, which the hardware guarantees is indivisible even under contention.

**4.** Both stream compaction (deciding where each surviving element goes after filtering) and histogram-building (deciding which bucket, and which position within that bucket, each element's count belongs at) ultimately need each element to know how many OTHER qualifying elements come before it -- which is exactly what a prefix sum (scan) computes. Compaction runs an exclusive scan over a 0/1 "does this element pass the filter" array to get each surviving element's output index; a privatized histogram runs a very similar scan-based idea per-bucket to combine multiple threads' partial counts without collisions.

**5.** A single mutex serializes every thread that wants to touch the stack at all, one at a time -- on a CPU with a handful of threads the resulting contention is small, but a GPU launches thousands of threads that are DESIGNED to make progress simultaneously, and forcing all of them through one lock throws away essentially all of that parallelism, turning a supposedly parallel operation back into a sequential one with extra overhead on top. A lock-free design using compare-and-swap lets threads that are not actually conflicting on the same memory location keep making progress independently, which a single mutex cannot do no matter how it is implemented.

**6.** A sequential traversal must follow one `next` pointer at a time, so finding a node's distance from the head of an n-node list takes O(n) sequential steps no matter how many processors are available -- the dependency chain itself is the bottleneck, not a lack of parallel hardware. Pointer jumping restructures the SAME problem so that doubling every node's pointer (to its 2-hop, then 4-hop, then 8-hop ancestor) each round finds every node's rank in only O(log n) rounds, with every node's doubling step in a given round running fully in parallel with every other node's.

**7.** Bitonic sort's comparison-and-swap network has a fixed, input-independent structure -- which pair of elements gets compared at which step never depends on the DATA being sorted, only on the elements' indices. That regularity means every thread in a bitonic sort stage does the same kind of work with no data-dependent branching and a compile-time-known communication pattern, which maps extremely well onto a GPU's SIMT execution model; an asymptotically faster comparison sort whose behavior depends on the data it is sorting typically introduces exactly the kind of irregular, data-dependent control flow that causes warp divergence and hurts real GPU throughput despite doing asymptotically less work.

**8.** Radix sort processes one fixed-width digit (or bit group) of each key at a time, and for a single digit, "which output bucket does this element belong to" is decided purely by that digit's numeric value -- no element is ever compared against another element at all, only against a small, fixed set of possible digit values. Each digit pass is itself a counting sort, which depends directly on Part 1's scan (prefix sum) primitive to convert per-digit counts into each element's exact output position.

**9.** A GPU kernel has a fixed, small, per-thread call stack, and thousands of threads run the same kernel simultaneously -- a recursive traversal's call stack depth is not just a memory concern but a control-flow one, since a natural recursive implementation also branches differently per thread based on data-dependent tree shape, which a GPU's SIMT model handles poorly. Rewriting the traversal around an explicit, thread-local stack (or a level-synchronous frontier, for a full tree) makes the memory footprint bounded and known in advance and turns the traversal into an explicit loop a GPU can execute efficiently across many threads at once.

**10.** A segment tree or Fenwick tree answers RANGE queries efficiently -- "what is the sum (or min, or max) of all elements between index i and index j" -- by combining a small number of precomputed partial results, something a hash table has no way to do at all, since a hash table only ever answers "what value is stored under this exact key" and has no notion of ordering or adjacency between keys. A hash table, conversely, answers point lookups by key in expected O(1) time regardless of key ordering, which a segment tree cannot do without conceptually degenerating into a linear scan across irrelevant leaves.

**11.** Open addressing resolves collisions by having a key that could not fit at its home bucket probe forward to a nearby empty slot -- which means a LATER key's actual location can depend on an EARLIER key still occupying its own home bucket. If that earlier key is deleted by resetting its slot directly to empty, any lookup for the later key will incorrectly stop probing at that now-empty slot and report "not found," even though the later key is still genuinely stored further down the probe sequence. A tombstone marks the slot as "deleted, but keep probing past me," preserving every later key's probe chain while still allowing the slot to be reused by a future insert.

**12.** Cuckoo hashing guarantees O(1) WORST-CASE lookup time -- every key lives in one of exactly two (or a small fixed number of) possible locations, so a lookup is always at most that many probes, full stop. Open addressing with linear probing only guarantees O(1) EXPECTED lookup time under a low load factor; its worst case degrades toward O(n) if a long run of collisions happens to form, which is a real, if unlikely, possibility linear probing does not rule out the way cuckoo hashing's fixed-candidate-set guarantee does.

**13.** A plain adjacency matrix stores one entry for every possible pair of vertices, which is O(V^2) space regardless of how many edges the graph actually has -- for a large, sparse real-world graph (where the number of edges is much closer to O(V) than O(V^2)), the vast majority of that memory represents edges that do not exist at all, wasting both memory and the bandwidth spent reading it. CSR instead stores only the edges that genuinely exist, packed contiguously per vertex, which is both dramatically smaller for a sparse graph and, because each vertex's neighbor list is contiguous in memory, far friendlier to the coalesced memory access patterns a GPU depends on for good throughput.

**14.** The frontier is the set of vertices discovered at the CURRENT BFS level -- not yet visited before this level, and not yet expanded. Every vertex in the frontier can have its own neighbors examined completely independently of every other frontier vertex, since none of them can affect each other's neighbor lists within the same level; that independence is exactly what lets one thread (or one thread per edge) be assigned to each frontier vertex and have the entire level processed as a single parallel step, with the next level's frontier collected as this level's output.

**15.** A binary heap's array-index parent/child relationships assume there is exactly one value "at" each array position at any moment, and maintaining the heap property after an insert requires a chain of dependent swaps (sift-up) that can touch positions all the way from a leaf back to the root. When many threads try to insert concurrently, their sift-up chains can easily touch overlapping positions in an order-dependent way, corrupting the heap property unless carefully synchronized -- which is exactly the kind of long, data-dependent dependency chain Part 7 shows is awkward to parallelize safely, motivating GPU-friendlier alternatives such as batched insertion or coarser-grained priority structures that trade some of a classic heap's per-operation optimality for genuinely safe concurrent access.

**16.** A general-purpose `malloc`/`free` is built to handle allocations of wildly different sizes and lifetimes safely from any thread, which typically requires some form of internal locking or contention-prone bookkeeping shared across all callers -- exactly the kind of shared, serializing resource that does not scale to thousands of concurrent GPU threads all allocating small, similarly-sized objects at once. A memory pool preallocates one large block up front and hands out fixed-size chunks from it using a much simpler, often lock-free bump allocator or free-list scheme tailored to the one specific allocation pattern a given data structure actually needs, avoiding general-purpose malloc's overhead and contention entirely.

**17.** Section 32.3 (the limit order book's cancel/hazard-pointer protocol) and Section 36.2 (double-buffered Bellman-Ford's atomic minimum over doubles) both reused Chapter 9's CAS-retry discipline, for two different purposes: Section 32.3 used a CAS-retry loop to safely publish and later free an order-pool slot without a hazard pointer reader ever dereferencing memory that had already been freed out from under it, while Section 36.2 used the identical retry-on-failure shape to implement an atomic minimum over `double` values (which CUDA has no native atomic for) by reinterpreting the value's bits as a 64-bit integer and retrying `atomicCAS` until its own candidate either wins or is beaten by a smaller one.

**18.** Bellman-Ford tolerates negative edge weights, which is essential here because Section 36.1's log-transform (`weight = -log(rate)`) deliberately assigns NEGATIVE weights to exactly the profitable exchange-rate cycles this chapter needs to detect -- Dijkstra's greedy, finalize-and-never-revisit approach is only correct when every edge weight is non-negative, so it would give wrong answers, or is simply not defined, on this graph. Section 36.3 reuses Chapter 11's pointer-jumping (list-ranking) technique, doubling each node's ancestor pointer over `log2(V)` rounds, to confirm that a node the detection pass merely found "still relaxable" genuinely lies ON the negative cycle rather than just downstream of one.

## B.4 Predict-the-Output Challenges

### Challenge 1: Inclusive vs. Exclusive Scan

Chapter 5 built two different scans over the same input: an inclusive scan, where each output position includes that position's own element, and an exclusive scan, where it does not. Before compiling and running the program below, write down both scans of the array `[3, 1, 4, 1, 5]` yourself.

```cpp
// 216_quiz_scan_offset.cpp
//
// Appendix B.3, Challenge 1 -- Chapter 5 built both an inclusive scan
// (running sum INCLUDING each element itself) and an exclusive scan
// (running sum of everything BEFORE each element). Before compiling and
// running this file, predict both outputs for the array [3, 1, 4, 1, 5].
//
// Compile: g++ -std=c++17 -Wall -Wextra -O2 216_quiz_scan_offset.cpp -o 216_quiz_scan_offset
// Run:     ./216_quiz_scan_offset

#include <cstdio>
#include <vector>

int main() {
    std::vector<int> arr = {3, 1, 4, 1, 5};

    printf("array: [3, 1, 4, 1, 5]\n\n");

    std::vector<int> inclusive(arr.size());
    int running = 0;
    for (size_t i = 0; i < arr.size(); ++i) {
        running += arr[i];
        inclusive[i] = running;
    }

    std::vector<int> exclusive(arr.size());
    running = 0;
    for (size_t i = 0; i < arr.size(); ++i) {
        exclusive[i] = running;
        running += arr[i];
    }

    printf("inclusive scan: [");
    for (size_t i = 0; i < inclusive.size(); ++i) printf("%d%s", inclusive[i], i + 1 < inclusive.size() ? ", " : "");
    printf("]\n");

    printf("exclusive scan: [");
    for (size_t i = 0; i < exclusive.size(); ++i) printf("%d%s", exclusive[i], i + 1 < exclusive.size() ? ", " : "");
    printf("]\n");

    return 0;
}
```

**Compile and run:**

```bash
g++ -std=c++17 -Wall -Wextra -O2 216_quiz_scan_offset.cpp -o 216_quiz_scan_offset
./216_quiz_scan_offset
```

**Revealed output:**

```text
array: [3, 1, 4, 1, 5]

inclusive scan: [3, 4, 8, 9, 14]
exclusive scan: [0, 3, 4, 8, 9]
```

### Challenge 2: Counting CAS Retries

Chapter 9's lock-free stack push retries its compare-and-swap whenever another thread's push landed first. Two threads, A and B, both begin pushing onto an EMPTY stack at the same moment, both having read `head = nullptr` before either one attempts its own CAS. Thread A's CAS is forced to land first. Before compiling and running the program below, predict: how many total CAS attempts does thread B need before its own push finally succeeds?

```cpp
// 217_quiz_cas_retries.cpp
//
// Appendix B.3, Challenge 2 -- Chapter 9 built a lock-free stack whose
// push operation retries its CAS whenever another thread's push landed
// first. Two threads, A and B, both start pushing onto an EMPTY stack at
// the same moment, both reading head=nullptr before either one attempts
// its CAS. Thread A's CAS is forced to land first. Before compiling and
// running this file, predict: how many total CAS attempts does thread B
// need before its push succeeds?
//
// Compile: g++ -std=c++17 -Wall -Wextra -O2 217_quiz_cas_retries.cpp -o 217_quiz_cas_retries
// Run:     ./217_quiz_cas_retries

#include <cstdio>

// A simplified, single-threaded REPLAY of two threads racing to push onto
// a lock-free stack's head pointer via compare-and-swap, forced into a
// specific interleaving: thread A's CAS always lands first.

struct Node { int value; Node* next; };

int main() {
    Node* head = nullptr;

    Node nodeA{10, nullptr};
    Node nodeB{20, nullptr};

    printf("=== two threads pushing onto an empty lock-free stack ===\n\n");
    printf("thread A wants to push value 10\n");
    printf("thread B wants to push value 20\n\n");

    // Both threads read the SAME initial head (nullptr) before either CAS.
    Node* observed_by_A = head;
    Node* observed_by_B = head;
    nodeA.next = observed_by_A;
    nodeB.next = observed_by_B;

    printf("both threads observe head = nullptr before attempting their CAS\n\n");

    int b_attempts = 0;

    // Thread A's CAS lands FIRST (forced interleaving for this quiz).
    printf("thread A: CAS(head, expected=nullptr, new=&nodeA) -- ");
    if (head == observed_by_A) {
        head = &nodeA;
        printf("succeeds -- head is now node A (value=%d)\n", head->value);
    } else {
        printf("fails\n");
    }

    // Thread B's CAS attempt #1: its "expected" value (nullptr) is now stale.
    b_attempts++;
    printf("thread B: CAS(head, expected=nullptr, new=&nodeB) attempt #%d -- ", b_attempts);
    if (head == observed_by_B) {
        head = &nodeB;
        printf("succeeds\n");
    } else {
        printf("fails (head changed underneath it) -- must retry\n");
        // A correct lock-free push re-reads head and retries.
        observed_by_B = head;
        nodeB.next = observed_by_B;
    }

    // Thread B's CAS attempt #2: now using the fresh head value.
    b_attempts++;
    printf("thread B: CAS(head, expected=&nodeA, new=&nodeB) attempt #%d -- ", b_attempts);
    if (head == observed_by_B) {
        head = &nodeB;
        printf("succeeds -- head is now node B (value=%d), nodeB.next = node A (value=%d)\n",
               head->value, nodeB.next->value);
    } else {
        printf("fails\n");
    }

    printf("\nthread B needed %d total CAS attempt(s) to succeed.\n", b_attempts);

    printf("\nfinal stack, top to bottom: %d -> %d\n", head->value, head->next->value);

    return 0;
}
```

**Compile and run:**

```bash
g++ -std=c++17 -Wall -Wextra -O2 217_quiz_cas_retries.cpp -o 217_quiz_cas_retries
./217_quiz_cas_retries
```

**Revealed output:**

```text
=== two threads pushing onto an empty lock-free stack ===

thread A wants to push value 10
thread B wants to push value 20

both threads observe head = nullptr before attempting their CAS

thread A: CAS(head, expected=nullptr, new=&nodeA) -- succeeds -- head is now node A (value=10)
thread B: CAS(head, expected=nullptr, new=&nodeB) attempt #1 -- fails (head changed underneath it) -- must retry
thread B: CAS(head, expected=&nodeA, new=&nodeB) attempt #2 -- succeeds -- head is now node B (value=20), nodeB.next = node A (value=10)

thread B needed 2 total CAS attempt(s) to succeed.

final stack, top to bottom: 20 -> 10
```

### Challenge 3: The Tombstone Trap

Chapter 19 warned that deleting a key from an open-addressing table by resetting its slot directly to EMPTY (a "hard delete") is wrong, and that a tombstone marker is required instead. The program below builds a capacity-5 table where `"A"` and `"B"` both hash to the same home bucket, so `"B"` ends up one probe past `"A"`, then hard-deletes `"A"`. Before compiling and running it, predict: does `lookup("B")` still succeed afterward?

```cpp
// 218_quiz_tombstone_trap.cpp
//
// Appendix B.3, Challenge 3 -- Chapter 19 warned that deleting from an
// open-addressing hash table by resetting a slot directly to EMPTY (a
// "hard delete") is wrong, and that a tombstone marker is required
// instead. This file builds a capacity-5 table where "A" and "B" both
// hash to the same home bucket, so "B" ends up one probe past "A". It
// then hard-deletes "A". Before compiling and running this file, predict:
// does lookup("B") still succeed afterward?
//
// Compile: g++ -std=c++17 -Wall -Wextra -O2 218_quiz_tombstone_trap.cpp -o 218_quiz_tombstone_trap
// Run:     ./218_quiz_tombstone_trap

#include <cstdio>
#include <string>
#include <vector>

// A capacity-5 open-addressing table (linear probing), demonstrating why
// Chapter 19's tombstone-based deletion exists at all: this version
// deletes by resetting a slot to EMPTY directly (a "hard delete"), which
// is the WRONG way to delete from an open-addressing table.

constexpr int CAP = 5;
enum class Slot { EMPTY, OCCUPIED };

struct Table {
    Slot state[CAP];
    std::string key[CAP];
    Table() { for (int i = 0; i < CAP; ++i) state[i] = Slot::EMPTY; }
};

int hash_of(const std::string& k) {
    // Deliberately chosen so "A" and "B" collide at the same home bucket.
    if (k == "A") return 2;
    if (k == "B") return 2;
    return 0;
}

void insert(Table& t, const std::string& k) {
    int home = hash_of(k);
    for (int probe = 0; probe < CAP; ++probe) {
        int idx = (home + probe) % CAP;
        if (t.state[idx] == Slot::EMPTY) {
            t.state[idx] = Slot::OCCUPIED;
            t.key[idx] = k;
            printf("insert(\"%s\"): home bucket=%d, placed at bucket %d (probe=%d)\n", k.c_str(), home, idx, probe);
            return;
        }
    }
}

void hard_delete(Table& t, const std::string& k) {
    int home = hash_of(k);
    for (int probe = 0; probe < CAP; ++probe) {
        int idx = (home + probe) % CAP;
        if (t.state[idx] == Slot::OCCUPIED && t.key[idx] == k) {
            t.state[idx] = Slot::EMPTY;   // WRONG for open addressing -- see Chapter 19
            printf("hard_delete(\"%s\"): bucket %d reset directly to EMPTY (no tombstone)\n", k.c_str(), idx);
            return;
        }
    }
}

bool lookup(Table& t, const std::string& k) {
    int home = hash_of(k);
    for (int probe = 0; probe < CAP; ++probe) {
        int idx = (home + probe) % CAP;
        if (t.state[idx] == Slot::EMPTY) {
            printf("lookup(\"%s\"): hit EMPTY bucket %d at probe=%d -- probe sequence stops here\n", k.c_str(), idx, probe);
            return false;
        }
        if (t.key[idx] == k) {
            printf("lookup(\"%s\"): found at bucket %d (probe=%d)\n", k.c_str(), idx, probe);
            return true;
        }
    }
    return false;
}

int main() {
    Table t;
    printf("=== capacity-5 table, \"A\" and \"B\" both hash to bucket 2 ===\n\n");

    insert(t, "A");
    insert(t, "B");
    printf("\n");

    hard_delete(t, "A");
    printf("\n");

    bool found = lookup(t, "B");
    printf("\nlookup(\"B\") result: %s\n", found ? "FOUND" : "NOT FOUND");
    printf("(\"B\" is still genuinely stored in the table -- this is exactly the bug\n");
    printf("Chapter 19's tombstones exist to prevent.)\n");

    return 0;
}
```

**Compile and run:**

```bash
g++ -std=c++17 -Wall -Wextra -O2 218_quiz_tombstone_trap.cpp -o 218_quiz_tombstone_trap
./218_quiz_tombstone_trap
```

**Revealed output:**

```text
=== capacity-5 table, "A" and "B" both hash to bucket 2 ===

insert("A"): home bucket=2, placed at bucket 2 (probe=0)
insert("B"): home bucket=2, placed at bucket 3 (probe=1)

hard_delete("A"): bucket 2 reset directly to EMPTY (no tombstone)

lookup("B"): hit EMPTY bucket 2 at probe=0 -- probe sequence stops here

lookup("B") result: NOT FOUND
("B" is still genuinely stored in the table -- this is exactly the bug
Chapter 19's tombstones exist to prevent.)
```

## Appendix Summary

B.2 and B.3 asked whether this book's individual ideas actually connect across Parts -- whether a reader can recognize the same underlying shape (a dependency chain, a shared mutable resource, a probe sequence's own hidden assumption) recurring in problems that look nothing alike on the surface, from GPU foundations all the way through Part 8's case studies. B.4 made three of the book's most common real mistakes concrete and checkable: confusing an inclusive scan with an exclusive one, underestimating how many times a lock-free retry loop can genuinely spin under real contention, and deleting from an open-addressing table without a tombstone. All three compile and run in seconds -- if a prediction did not match the revealed output, that mismatch is worth chasing back to the chapter that first built the idea, which this appendix has named at every step along the way.
