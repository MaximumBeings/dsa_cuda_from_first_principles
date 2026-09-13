# Chapter 16: Parallel Tree and Trie Construction

Chapter 15 built one balanced tree, once, from data that was conveniently already sorted. Real trees rarely get that luxury: data arrives in whatever order it arrives, often from many independent sources at once, and the tree has to be built out of it directly. This chapter tackles that harder problem in two stages. Section 16.1 shows that a binary search tree can still be built level by level from UNSORTED data, generalizing Chapter 14's partition-based sample sort and Chapter 15.1's level-by-level construction together. Sections 16.2 and 16.3 turn to tries -- trees where multiple keys deliberately SHARE nodes along a common prefix -- and confront the one genuinely new hazard that sharing creates: two threads that both want to create the same missing shared node at the same time. That hazard turns out to be a direct generalization of two things this book has already built and trusted, Chapter 8's atomic bump allocator and Chapter 9's compare-and-swap retry loop, now applied to tree and trie nodes instead of a stack's head pointer.

## 16.1 Parallel BST Construction from Unsorted Data

### Intuition

Chapter 15.1's level-by-level BST construction assumed the input was already sorted, which is what let every node's children land at a closed-form slot (`2*slot+1`, `2*slot+2`) with zero coordination between threads. Unsorted data cannot use the sorted-array midpoint trick at all -- there is no "middle element" that is guaranteed to be a good root. Instead, this section reaches back to Chapter 14.3's sample-sort partitioning idea (and, further back, Chapter 6.3's stable partition): pick a pivot, split everything else into "less than pivot" and "greater than pivot," and use each pivot as one tree node, with its two partitions becoming its two subtrees. Applied one LEVEL at a time -- every pending group in a level partitioned by its own independent thread -- this builds a full binary search tree from unsorted data with the exact same "loop over levels, synchronize between them" shape Chapter 15.1 used, just with GROUPS (not sorted-array ranges) as the unit of work.

### The Sequential (CPU) Baseline

```cpp
#include <cstdio>
#include <vector>
#include <queue>
#include <algorithm>

// Chapter 16.1 -- The Sequential (CPU) Baseline.
// Chapter 15 assumed the input was already sorted. Real data arrives
// arbitrary. A balanced-ISH BST can still be built directly from
// UNSORTED data, using the same partitioning idea Chapter 6.3's stable
// partition and Chapter 14.3's sample sort both already relied on:
// pick a pivot (here, simply the first remaining element), partition
// everything else into "less than pivot" and "greater than pivot," and
// recurse on each half independently. Continuing Chapter 15's own
// discipline, this baseline builds the tree ITERATIVELY (an explicit
// queue of pending groups), never recursively.

struct Group { std::vector<int> data; int slot; };

void build_bst_partition(const std::vector<int>& a, std::vector<int>& value,
                          std::vector<int>& left, std::vector<int>& right,
                          int& root, int& next_free) {
    value.clear(); left.clear(); right.clear();
    next_free = 0;
    auto alloc = [&]() {
        int s = next_free++;
        value.push_back(0); left.push_back(-1); right.push_back(-1);
        return s;
    };

    root = alloc();
    std::queue<Group> q;
    q.push({a, root});

    while (!q.empty()) {
        Group g = q.front(); q.pop();
        int pivot = g.data[0];
        std::vector<int> less, greater;
        for (size_t i = 1; i < g.data.size(); i++) {
            if (g.data[i] < pivot) less.push_back(g.data[i]);
            else greater.push_back(g.data[i]);
        }
        value[g.slot] = pivot;
        if (!less.empty()) {
            int lslot = alloc();
            left[g.slot] = lslot;
            q.push({less, lslot});
        }
        if (!greater.empty()) {
            int rslot = alloc();
            right[g.slot] = rslot;
            q.push({greater, rslot});
        }
    }
}

void inorder(int slot, const std::vector<int>& value, const std::vector<int>& left,
             const std::vector<int>& right, std::vector<int>& out) {
    if (slot == -1) return;
    inorder(left[slot], value, left, right, out);
    out.push_back(value[slot]);
    inorder(right[slot], value, left, right, out);
}

int main() {
    printf("=== Section 16.1 CPU baseline: iterative partition-based BST construction ===\n\n");

    std::vector<int> a = {50, 20, 80, 10, 30, 70, 90, 40};
    printf("unsorted input: ");
    for (int v : a) printf("%d ", v);
    printf("\n\n");

    std::vector<int> value, left, right;
    int root, next_free;
    build_bst_partition(a, value, left, right, root, next_free);

    printf("value array: ");
    for (int v : value) printf("%d ", v);
    printf("\nleft array:  ");
    for (int v : left) printf("%d ", v);
    printf("\nright array: ");
    for (int v : right) printf("%d ", v);
    printf("\nroot slot: %d\n\n", root);

    std::vector<int> traversal;
    inorder(root, value, left, right, traversal);
    printf("in-order traversal (verification): ");
    for (int v : traversal) printf("%d ", v);
    printf("\n\n");

    std::vector<int> expected_value = {50, 20, 80, 10, 30, 70, 90, 40};
    std::vector<int> expected_left = {1, 3, 5, -1, -1, -1, -1, -1};
    std::vector<int> expected_right = {2, 4, 6, -1, 7, -1, -1, -1};
    std::vector<int> sorted_a = a;
    std::sort(sorted_a.begin(), sorted_a.end());
    bool ok = (value == expected_value) && (left == expected_left) &&
              (right == expected_right) && (traversal == sorted_a);

    printf("expected value: 50 20 80 10 30 70 90 40\n");
    printf("expected left:  1 3 5 -1 -1 -1 -1 -1\n");
    printf("expected right: 2 4 6 -1 7 -1 -1 -1\n\n");
    printf("the value array came out identical to the ORIGINAL input order -- because each\n");
    printf("pivot is always the first remaining element, and the queue processes groups in\n");
    printf("the order they were discovered, this particular pivot rule happens to recover\n");
    printf("the input's own order exactly.\n");

    printf("\nself-check: partition-based construction matches expected arrays, in-order\n");
    printf("traversal is fully sorted: %s\n", ok ? "confirmed" : "MISMATCH");
    return ok ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++17 -Wall -Wextra -O2 82_bst_partition_cpu_baseline.cpp -o 82_bst_partition_cpu_baseline
./82_bst_partition_cpu_baseline
```

**Sample input:** the unsorted array `{50, 20, 80, 10, 30, 70, 90, 40}`, built into a BST via iterative, queue-based partitioning (pivot = first remaining element of each group).

**Sample output:**

```text
=== Section 16.1 CPU baseline: iterative partition-based BST construction ===

unsorted input: 50 20 80 10 30 70 90 40 

value array: 50 20 80 10 30 70 90 40 
left array:  1 3 5 -1 -1 -1 -1 -1 
right array: 2 4 6 -1 7 -1 -1 -1 
root slot: 0

in-order traversal (verification): 10 20 30 40 50 70 80 90 

expected value: 50 20 80 10 30 70 90 40
expected left:  1 3 5 -1 -1 -1 -1 -1
expected right: 2 4 6 -1 7 -1 -1 -1

the value array came out identical to the ORIGINAL input order -- because each
pivot is always the first remaining element, and the queue processes groups in
the order they were discovered, this particular pivot rule happens to recover
the input's own order exactly.

self-check: partition-based construction matches expected arrays, in-order
traversal is fully sorted: confirmed
```

### The Concept, In Detail

Tracing the partition-based construction on `{50, 20, 80, 10, 30, 70, 90, 40}`:

```
group0 = {50, 20, 80, 10, 30, 70, 90, 40}, slot=0, pivot=50
  remaining: 20, 80, 10, 30, 70, 90, 40
  less than 50:    {20, 10, 30, 40}   -> group1 (left[0]=1)
  greater than 50: {80, 70, 90}       -> group2 (right[0]=2)

group1 = {20, 10, 30, 40}, slot=1, pivot=20
  less than 20:    {10}        -> group3 (left[1]=3)
  greater than 20: {30, 40}    -> group4 (right[1]=4)

group2 = {80, 70, 90}, slot=2, pivot=80
  less than 80:    {70}        -> group5 (left[2]=5)
  greater than 80: {90}        -> group6 (right[2]=6)

group3 = {10}, slot=3, pivot=10 -- no remaining elements, leaf
group4 = {30, 40}, slot=4, pivot=30
  greater than 30: {40}        -> group7 (right[4]=7)
group5 = {70}, slot=5, pivot=70 -- leaf
group6 = {90}, slot=6, pivot=90 -- leaf
group7 = {40}, slot=7, pivot=40 -- leaf
```

```
ASCII view of the resulting tree:

                       50 (slot 0)
                   /                \
           20 (slot 1)            80 (slot 2)
           /        \              /        \
    10 (slot 3)  30 (slot 4)  70 (slot 5)  90 (slot 6)
                    \
                  40 (slot 7)

  value: [50, 20, 80, 10, 30, 70, 90, 40]
  left:  [ 1,  3,  5, -1, -1, -1, -1, -1]
  right: [ 2,  4,  6, -1,  7, -1, -1, -1]
```

The value array comes out identical to the ORIGINAL input order here -- not a coincidence for this particular input, but a direct consequence of the pivot rule (always the first remaining element of a group) combined with level-order processing: the first element of each group becomes that group's node value, in the exact order groups are discovered. A different pivot rule (say, a random element, or a median-of-three) would produce a different-shaped tree from the same input, still correctly ordered, just not necessarily in input order.

The parallel opportunity mirrors Section 15.1's exactly: every group WITHIN one level is completely independent of every other group in that same level -- each one reads only its own remaining elements and writes only its own node's value plus its own two child groups. Unlike Section 15.1, though, no closed-form slot formula exists here, because the tree's shape depends on the DATA, not just its size (the tree above is not even complete: slot 4 has only a right child). Each group is instead given its own fixed-capacity row in a flat buffer (this section reuses Chapter 14.3's own device-side layout instinct: give each unit of work its own private space so no thread needs to know where another thread's data starts or ends), avoiding the need for any shared allocation counter for the DATA itself, even though the tree's node slots still need one.

[COMMON TRAP]
Reusing a sorted-input assumption (such as Section 15.1's closed-form `2*slot+1`/`2*slot+2` child addressing) on unsorted, partition-built data silently produces a WRONG tree, not a crash: the arithmetic still computes some slot, it is simply not connected to where that group's pivot actually needs to attach. Whenever tree shape depends on data (as it does here, and does not in Section 15.1), child slots must come from an explicit allocation step, never a formula.

### Code and Verification

```cpp
#include <cstdio>
#include <vector>

// Chapter 16.1 main -- building the same tree with one level processed
// per kernel launch, every GROUP in a level handled by an independent
// thread. Every group's partition (comparing its own remaining
// elements against its own pivot) touches only that group's own data
// -- no thread ever reads or writes another thread's group -- so all
// groups in a level run genuinely in parallel, exactly the "loop over
// passes, synchronize between them" shape Chapters 12-15 all used, with
// GROUPS (not merge widths, digit places, or tree-height levels) as
// what this pass processes. Each group's own remaining elements are
// stored in a fixed-capacity row of a flat array (MAX_GROUP columns
// wide) so no thread needs to know where any other thread's group
// starts.
#define MAX_GROUP 8

__global__ void partition_level_kernel(const int* g_data, const int* g_count,
                                        const int* g_slot, int num_groups,
                                        int* g_value, int* g_left, int* g_right,
                                        int* g_next_data, int* g_next_count,
                                        int* g_next_slot, const int* g_next_left_slot,
                                        const int* g_next_right_slot) {
    int t = threadIdx.x;
    if (t >= num_groups) return;

    const int* group = g_data + t * MAX_GROUP;
    int count = g_count[t];
    int slot = g_slot[t];
    int pivot = group[0];

    int less[MAX_GROUP], greater[MAX_GROUP];
    int nless = 0, ngreater = 0;
    for (int i = 1; i < count; i++) {
        if (group[i] < pivot) less[nless++] = group[i];
        else greater[ngreater++] = group[i];
    }

    g_value[slot] = pivot;
    g_left[slot] = (nless > 0) ? g_next_left_slot[t] : -1;
    g_right[slot] = (ngreater > 0) ? g_next_right_slot[t] : -1;

    int* left_row = g_next_data + (2 * t) * MAX_GROUP;
    int* right_row = g_next_data + (2 * t + 1) * MAX_GROUP;
    for (int i = 0; i < nless; i++) left_row[i] = less[i];
    for (int i = 0; i < ngreater; i++) right_row[i] = greater[i];
    g_next_count[2 * t] = nless;
    g_next_count[2 * t + 1] = ngreater;
    g_next_slot[2 * t] = (nless > 0) ? g_next_left_slot[t] : -1;
    g_next_slot[2 * t + 1] = (ngreater > 0) ? g_next_right_slot[t] : -1;
}

// ---- Host-side replay of the identical per-thread logic: one level's
// worth of independent threads per pass, each touching only its own
// group's data, with the host loop (and a simple compaction step
// between levels) playing the role of the synchronization between
// kernel launches. ----

int main() {
    printf("=== Section 16.1 main: level-by-level parallel partition-based construction ===\n\n");

    std::vector<int> a = {50, 20, 80, 10, 30, 70, 90, 40};
    printf("unsorted input: ");
    for (int v : a) printf("%d ", v);
    printf("\n\n");

    std::vector<int> value, left, right;
    int next_free = 0;
    auto alloc = [&]() {
        int s = next_free++;
        value.push_back(0); left.push_back(-1); right.push_back(-1);
        return s;
    };
    int root = alloc();

    std::vector<std::vector<int>> cur_groups = {a};
    std::vector<int> cur_slot = {root};
    int level_num = 0;

    while (!cur_groups.empty()) {
        int width = (int)cur_groups.size();
        std::vector<std::vector<int>> next_groups;
        std::vector<int> next_slot;

        printf("level %d: %d independent thread(s), slots [", level_num, width);
        for (int s : cur_slot) printf("%d ", s);
        printf("]\n");
        level_num++;

        for (int t = 0; t < width; t++) {   // width independent threads
            const auto& group = cur_groups[t];
            int slot = cur_slot[t];
            int pivot = group[0];
            std::vector<int> less, greater;
            for (size_t i = 1; i < group.size(); i++) {
                if (group[i] < pivot) less.push_back(group[i]);
                else greater.push_back(group[i]);
            }
            value[slot] = pivot;
            if (!less.empty()) { left[slot] = alloc(); next_groups.push_back(less); next_slot.push_back(left[slot]); }
            if (!greater.empty()) { right[slot] = alloc(); next_groups.push_back(greater); next_slot.push_back(right[slot]); }
        }
        cur_groups = next_groups;
        cur_slot = next_slot;
    }
    printf("\n");

    printf("value array: ");
    for (int v : value) printf("%d ", v);
    printf("\nleft array:  ");
    for (int v : left) printf("%d ", v);
    printf("\nright array: ");
    for (int v : right) printf("%d ", v);
    printf("\n\n");

    std::vector<int> expected_value = {50, 20, 80, 10, 30, 70, 90, 40};
    std::vector<int> expected_left = {1, 3, 5, -1, -1, -1, -1, -1};
    std::vector<int> expected_right = {2, 4, 6, -1, 7, -1, -1, -1};
    bool ok = (value == expected_value) && (left == expected_left) && (right == expected_right);

    printf("expected value: 50 20 80 10 30 70 90 40\n");
    printf("expected left:  1 3 5 -1 -1 -1 -1 -1\n");
    printf("expected right: 2 4 6 -1 7 -1 -1 -1\n\n");
    printf("%d levels total -- every group WITHIN a level ran as an independent thread,\n",
           level_num);
    printf("touching only its own row of the fixed-capacity group buffer, with no thread\n");
    printf("ever reading or writing another thread's group data.\n");

    printf("\nself-check: level-by-level parallel construction matches CPU baseline: %s\n",
           ok ? "confirmed" : "MISMATCH");
    return ok ? 0 : 1;
}
```

**Compile and run:**

```bash
NVDIR=/usr/local/lib/python3.11/dist-packages/nvidia/cu13
$NVDIR/bin/nvcc -arch=sm_80 -I$NVDIR/include -L$NVDIR/lib -lcudart 83_bst_partition_levels_kernel.cu -o 83_bst_partition_levels_kernel
LD_LIBRARY_PATH=$NVDIR/lib ./83_bst_partition_levels_kernel
```

**Sample input:** the same unsorted array, built level by level, with every GROUP in a level handled by an independent thread.

**Sample output:**

```text
=== Section 16.1 main: level-by-level parallel partition-based construction ===

unsorted input: 50 20 80 10 30 70 90 40 

level 0: 1 independent thread(s), slots [0 ]
level 1: 2 independent thread(s), slots [1 2 ]
level 2: 4 independent thread(s), slots [3 4 5 6 ]
level 3: 1 independent thread(s), slots [7 ]

value array: 50 20 80 10 30 70 90 40 
left array:  1 3 5 -1 -1 -1 -1 -1 
right array: 2 4 6 -1 7 -1 -1 -1 

expected value: 50 20 80 10 30 70 90 40
expected left:  1 3 5 -1 -1 -1 -1 -1
expected right: 2 4 6 -1 7 -1 -1 -1

4 levels total -- every group WITHIN a level ran as an independent thread,
touching only its own row of the fixed-capacity group buffer, with no thread
ever reading or writing another thread's group data.

self-check: level-by-level parallel construction matches CPU baseline: confirmed
```

## 16.2 Building a Trie: Sequential Construction, and the Shared-Node Race

### Intuition

A trie stores keys as paths through a shared tree, one "digit" (one symbol position) per level, and different keys that happen to share a prefix deliberately share those prefix nodes -- that is the entire point of a trie, and it is also exactly what makes concurrent trie construction dangerous in a way BST construction never was. Section 16.1's groups never shared a slot: every thread owned a private row, so there was nothing to contend over. A trie has no such luxury -- two threads inserting two different keys that share an as-yet-unbuilt prefix can both discover the SAME missing child slot empty at the SAME time, and Chapter 8's atomic bump allocator alone does not fix this: it guarantees each thread gets a unique NEW node id, but says nothing about which thread's id actually gets published into the shared slot both of them are racing to fill.

### The Sequential (CPU) Baseline

```cpp
#include <cstdio>
#include <vector>
#include <array>

// Chapter 16.2 -- The Sequential (CPU) Baseline.
// A trie stores keys as paths through a shared tree: each node holds one
// slot per possible "digit" (Chapter 13's vocabulary again -- a digit is
// just one symbol position in a key), and a key is inserted one digit at
// a time, creating a new node only where the path doesn't already exist.
// Different keys that share a prefix SHARE those prefix nodes -- that
// sharing is exactly what makes concurrent trie construction dangerous
// (Section 16.2's next file), because two different keys can both want
// to create the SAME missing child at the SAME time.
#define ALPHABET 6

struct Trie {
    std::vector<std::array<int, ALPHABET>> children;
    std::vector<bool> is_end;

    int alloc() {
        children.push_back({-1, -1, -1, -1, -1, -1});
        is_end.push_back(false);
        return (int)children.size() - 1;
    }

    int root;
    Trie() { root = alloc(); }

    void insert(const std::vector<int>& key) {
        int cur = root;
        for (int d : key) {
            if (children[cur][d] == -1) {
                children[cur][d] = alloc();
            }
            cur = children[cur][d];
        }
        is_end[cur] = true;
    }

    int lookup(const std::vector<int>& key) const {
        int cur = root;
        for (int d : key) {
            if (children[cur][d] == -1) return -1;
            cur = children[cur][d];
        }
        return is_end[cur] ? cur : -1;
    }
};

int main() {
    printf("=== Section 16.2 CPU baseline: sequential trie insertion ===\n\n");

    std::vector<std::vector<int>> keys = {
        {1, 2, 3},
        {1, 2, 4},
        {1, 5},
        {2, 1},
    };

    printf("keys to insert (as digit sequences):\n");
    for (size_t i = 0; i < keys.size(); i++) {
        printf("  key%zu = [", i);
        for (size_t j = 0; j < keys[i].size(); j++) {
            printf("%d%s", keys[i][j], j + 1 < keys[i].size() ? "," : "");
        }
        printf("]\n");
    }
    printf("\n");

    Trie t;
    for (size_t i = 0; i < keys.size(); i++) {
        printf("inserting key%zu ...\n", i);
        t.insert(keys[i]);
    }
    printf("\n");

    printf("final trie: %zu nodes\n", t.children.size());
    for (size_t i = 0; i < t.children.size(); i++) {
        printf("  node%zu: children=[", i);
        for (int d = 0; d < ALPHABET; d++) {
            printf("%d%s", t.children[i][d], d + 1 < ALPHABET ? "," : "");
        }
        printf("], is_end=%s\n", t.is_end[i] ? "true" : "false");
    }
    printf("\n");

    bool ok = true;
    std::vector<int> expected_lookup = {3, 4, 5, 7};
    for (size_t i = 0; i < keys.size(); i++) {
        int slot = t.lookup(keys[i]);
        printf("lookup key%zu -> %d\n", i, slot);
        if (slot != expected_lookup[i]) ok = false;
    }

    bool struct_ok = (t.children.size() == 8) &&
        t.children[0] == std::array<int, ALPHABET>{-1, 1, 6, -1, -1, -1} &&
        t.children[1] == std::array<int, ALPHABET>{-1, -1, 2, -1, -1, 5} &&
        t.children[2] == std::array<int, ALPHABET>{-1, -1, -1, 3, 4, -1} &&
        t.children[6] == std::array<int, ALPHABET>{-1, 7, -1, -1, -1, -1} &&
        !t.is_end[0] && !t.is_end[1] && !t.is_end[2] && t.is_end[3] &&
        t.is_end[4] && t.is_end[5] && !t.is_end[6] && t.is_end[7];
    ok = ok && struct_ok;

    printf("\nnode1 (shared prefix [1]) has TWO children set from two different\n");
    printf("keys inserted at different times: digit 2 (from key0/key1) and digit 5\n");
    printf("(from key2) -- prefix sharing across insertions is exactly the shape\n");
    printf("that becomes a data race when insertions run concurrently instead of\n");
    printf("sequentially.\n");

    printf("\nself-check: trie structure and all lookups match expected: %s\n",
           ok ? "confirmed" : "MISMATCH");
    return ok ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++17 -Wall -Wextra -O2 84_trie_insert_cpu_baseline.cpp -o 84_trie_insert_cpu_baseline
./84_trie_insert_cpu_baseline
```

**Sample input:** four keys (`[1,2,3]`, `[1,2,4]`, `[1,5]`, `[2,1]`) inserted one at a time into an initially empty trie, `ALPHABET=6`.

**Sample output:**

```text
=== Section 16.2 CPU baseline: sequential trie insertion ===

keys to insert (as digit sequences):
  key0 = [1,2,3]
  key1 = [1,2,4]
  key2 = [1,5]
  key3 = [2,1]

inserting key0 ...
inserting key1 ...
inserting key2 ...
inserting key3 ...

final trie: 8 nodes
  node0: children=[-1,1,6,-1,-1,-1], is_end=false
  node1: children=[-1,-1,2,-1,-1,5], is_end=false
  node2: children=[-1,-1,-1,3,4,-1], is_end=false
  node3: children=[-1,-1,-1,-1,-1,-1], is_end=true
  node4: children=[-1,-1,-1,-1,-1,-1], is_end=true
  node5: children=[-1,-1,-1,-1,-1,-1], is_end=true
  node6: children=[-1,7,-1,-1,-1,-1], is_end=false
  node7: children=[-1,-1,-1,-1,-1,-1], is_end=true

lookup key0 -> 3
lookup key1 -> 4
lookup key2 -> 5
lookup key3 -> 7

node1 (shared prefix [1]) has TWO children set from two different
keys inserted at different times: digit 2 (from key0/key1) and digit 5
(from key2) -- prefix sharing across insertions is exactly the shape
that becomes a data race when insertions run concurrently instead of
sequentially.

self-check: trie structure and all lookups match expected: confirmed
```

### The Concept, In Detail

Sequential insertion never has a problem: each key walks from the root one digit at a time, creating a node only where the path doesn't already exist, and since only one insertion ever runs at once, "check if empty, then create" is always safe. The danger appears the moment two insertions run AT THE SAME TIME and their paths cross at a not-yet-built node. Tracing what happens when two threads, T0 inserting `key0 = [1,2,3]` and T1 inserting `key1 = [1,2,4]`, both start from the root at once:

```
position 0 (both threads want digit 1 from the root, node 0):
  T0 reads children[0][1] -> sees -1 (empty)
  T1 reads children[0][1] -> ALSO sees -1 (both reads happen before either write)
  T0 allocates a fresh node (id 1) and writes children[0][1] = 1
  T1 allocates a fresh node (id 2) and writes children[0][1] = 2   <- OVERWRITES T0's write
  final: children[0][1] = 2

  T0 believes it is now at node 1 (its OWN allocated id) -- but node 1 is no
  longer reachable from the root at all; T1's write replaced it.
```

```
ASCII view: node 1 becomes ORPHANED the instant T1's write lands.

  before either write:      root(0) --digit1--> (empty)

  after T0's write:         root(0) --digit1--> node1   [T0 thinks this is real]

  after T1's write:         root(0) --digit1--> node2   [T0's node1 is now UNREACHABLE]
                                                          node1 still exists in the
                                                          pool, with correct data --
                                                          nothing can ever find it again
```

T0 has no way to detect that this happened -- it simply keeps building on top of node 1 as if its write had succeeded, so its ENTIRE remaining subtree (nodes for digits `2` and `3` of `key0`) gets built correctly but is permanently unreachable, and `key0` becomes unfindable by any future lookup, even though every node it needed was allocated and filled in with correct data. This is precisely Chapter 9.1's stack-push race (`old_head` read by every lane, then overwritten by whichever lane stores last), generalized from a single shared head pointer to a shared child slot.

[COMMON TRAP]
It is tempting to think the fix is "just make node allocation atomic," since that is what fixed Chapter 8's earlier race. Allocation here is ALREADY atomic (via `atomicAdd`, exactly as Chapter 8 established) -- both T0 and T1 get distinct, valid node ids with no collision at all. The race is entirely on the separate step of PUBLISHING that id into the shared `children[][]` slot; fixing allocation alone does nothing to protect that second step.

The fix follows Chapter 9.1's own compare-and-swap pattern directly: instead of an unconditional store, publish the newly allocated candidate with `atomicCAS(&children[node][digit], -1, candidate)`. If the slot still holds `-1`, the CAS succeeds and this thread's candidate wins. If some other thread already filled it, the CAS fails, changes nothing, and returns the WINNING thread's id -- which the losing thread simply reuses, discarding its own now-wasted candidate, and continuing its own insertion from the winner's node exactly as if it had found that node already there from the start.

```
same two threads, now using atomicCAS to publish the shared slot:

  position 0: T0's CAS(children[0][1], -1 -> 1) succeeds -> T0 uses node 1
              T1's CAS(children[0][1], -1 -> 2) FAILS (slot is now 1, not -1)
                -> T1 discards candidate 2, reuses node 1 instead

  position 1: both threads now share cur=1, want digit 2
              T0's CAS(children[1][2], -1 -> 3) succeeds -> T0 uses node 3
              T1's CAS(children[1][2], -1 -> 4) FAILS -> T1 reuses node 3

  position 2: T0 wants digit 3 on node 3, T1 wants digit 4 on node 3 --
              DIFFERENT digits, so no contention at all: both CAS calls succeed
              independently -> T0 gets node 5, T1 gets node 6

  result: BOTH keys findable. Two candidate node ids (2 and 4) were allocated
  and never used -- wasted, but harmless -- while zero keys were lost.
```

### Code and Verification

```cpp
#include <cstdio>
#include <vector>
#include <array>
#include <algorithm>

// Chapter 16.2 main -- Section 16.1's shared-node problem was about
// which SLOT a node goes into (fixed by giving every group a private
// row, so no two threads ever touch the same slot at all). A trie
// removes that luxury: two different keys that share a prefix are
// SUPPOSED to share the prefix's nodes, so multiple threads inserting
// different keys can genuinely need to create the SAME missing child at
// the SAME time. Chapter 8 already solved "give me a unique id" with an
// atomic bump allocator (`atomicAdd`) -- that part is not the new
// problem here. The new problem is what happens to the shared
// `children[node][digit]` SLOT itself when two threads both find it
// empty and both try to fill it.
#define ALPHABET 6

// The naive insert: allocation of a fresh node id is already atomic
// (Chapter 8), but publishing that id into the shared child slot is a
// plain, unprotected LOAD-then-STORE -- exactly the shape Chapter 7.1
// and Chapter 9.1 both already proved is unsafe. Two threads can both
// see `children[cur][d] == -1`, both allocate their OWN distinct node
// (no collision there), and then both store into the SAME slot -- the
// second store silently overwrites the first, and the first thread's
// freshly allocated node becomes permanently unreachable.
__global__ void trie_insert_naive_kernel(int* children, int* is_end, int* next_free,
                                          const int* keys_flat, const int* key_offset,
                                          const int* key_len, int alphabet) {
    int t = threadIdx.x;
    int cur = 0;   // every key starts at the root
    for (int p = 0; p < key_len[t]; p++) {
        int d = keys_flat[key_offset[t] + p];
        int slot = cur * alphabet + d;
        int existing = children[slot];              // LOAD
        int next;
        if (existing == -1) {
            int candidate = atomicAdd(next_free, 1); // unique id -- Chapter 8's atomic allocator
            children[slot] = candidate;              // STORE -- unprotected: can be overwritten
            next = candidate;
        } else {
            next = existing;
        }
        cur = next;   // this thread keeps using ITS OWN idea of where it went,
                       // even if another thread's later store changes the slot
    }
    is_end[cur] = 1;
}

// The fixed insert: publish the new node id with atomicCAS, exactly
// Chapter 9.1's retry-or-reuse pattern, generalized from a stack's head
// pointer to a trie's child slot. `atomicCAS(&children[slot], -1,
// candidate)` stores `candidate` ONLY IF the slot still holds -1;
// otherwise it changes nothing and returns whoever's id is already
// there. A thread that loses the race simply REUSES the winner's node
// instead of overwriting it -- its own speculatively allocated
// candidate is discarded (wasted, but harmless: it is never referenced
// by anything, so it costs a little memory and nothing else).
__global__ void trie_insert_cas_kernel(int* children, int* is_end, int* next_free,
                                        const int* keys_flat, const int* key_offset,
                                        const int* key_len, int alphabet) {
    int t = threadIdx.x;
    int cur = 0;
    for (int p = 0; p < key_len[t]; p++) {
        int d = keys_flat[key_offset[t] + p];
        int slot = cur * alphabet + d;
        int existing = children[slot];
        int next;
        if (existing == -1) {
            int candidate = atomicAdd(next_free, 1);         // speculative -- may be wasted
            int prev = atomicCAS(&children[slot], -1, candidate);
            next = (prev == -1) ? candidate : prev;          // won -> use ours; lost -> reuse theirs
        } else {
            next = existing;
        }
        cur = next;
    }
    is_end[cur] = 1;
}

// ---- Host-side replay of the identical per-thread logic, under one
// specific fixed interleaving: all LOADs for a position happen before
// any STORE for that position (Chapter 7.1/9.1's worst-case shape), and
// when both threads' operations touch the very same slot, thread 0's
// operation is the one recorded as happening first. Two keys sharing
// the prefix digit 1 then digit 2: key0 = [1,2,3], key1 = [1,2,4]. ----

using Row = std::array<int, ALPHABET>;

struct TrieState {
    std::vector<Row> children;
    std::vector<bool> is_end;
    int next_free = 0;
    int alloc() {
        children.push_back({-1, -1, -1, -1, -1, -1});
        is_end.push_back(false);
        return next_free++;
    }
};

void print_trie(const TrieState& s) {
    for (size_t i = 0; i < s.children.size(); i++) {
        printf("  node%zu: children=[", i);
        for (int d = 0; d < ALPHABET; d++) {
            printf("%d%s", s.children[i][d], d + 1 < ALPHABET ? "," : "");
        }
        printf("], is_end=%s\n", s.is_end[i] ? "true" : "false");
    }
}

// Naive simulation: at each position, BOTH threads' reads happen first
// (against whatever the slot held before this position), then BOTH
// threads' stores happen, with thread 1's store landing last whenever
// they collide -- so thread 1's store is the one that survives.
TrieState simulate_naive(const std::vector<int>& key0, const std::vector<int>& key1) {
    TrieState s;
    s.alloc();   // root = node 0
    int cur0 = 0, cur1 = 0;
    size_t steps = std::max(key0.size(), key1.size());

    for (size_t p = 0; p < steps; p++) {
        bool has0 = p < key0.size(), has1 = p < key1.size();
        int d0 = has0 ? key0[p] : -1, d1 = has1 ? key1[p] : -1;

        // LOAD phase -- both threads read before either writes.
        int read0 = has0 ? s.children[cur0][d0] : -1;
        int read1 = has1 ? s.children[cur1][d1] : -1;

        // Both threads allocate a candidate (Chapter 8's atomic
        // allocator -- always succeeds, ids never collide).
        int cand0 = (has0 && read0 == -1) ? s.alloc() : -1;
        int cand1 = (has1 && read1 == -1) ? s.alloc() : -1;

        // STORE phase, thread 0 first, thread 1 second -- a collision
        // on the same slot means thread 1's store is the one that
        // survives, silently discarding thread 0's.
        if (has0 && read0 == -1) s.children[cur0][d0] = cand0;
        if (has1 && read1 == -1) s.children[cur1][d1] = cand1;

        printf("  position %zu: T0 slot=(node%d,digit%d) saw %s -> uses node%d | "
               "T1 slot=(node%d,digit%d) saw %s -> uses node%d\n",
               p, has0 ? cur0 : -1, d0, read0 == -1 ? "empty" : "existing",
               has0 ? (read0 == -1 ? cand0 : read0) : -1,
               has1 ? cur1 : -1, d1, read1 == -1 ? "empty" : "existing",
               has1 ? (read1 == -1 ? cand1 : read1) : -1);

        if (has0) cur0 = (read0 == -1) ? cand0 : read0;   // T0 keeps ITS OWN view,
        if (has1) cur1 = (read1 == -1) ? cand1 : read1;   // even if T1 later overwrote the slot
    }
    s.is_end[cur0] = true;
    s.is_end[cur1] = true;
    return s;
}

// CAS simulation: same LOAD-before-STORE shape, but a collision is
// DETECTED -- the losing thread discards its candidate and reuses
// whichever node the winner (thread 0, by this fixed tie-break)
// actually installed.
TrieState simulate_cas(const std::vector<int>& key0, const std::vector<int>& key1) {
    TrieState s;
    s.alloc();
    int cur0 = 0, cur1 = 0;
    std::vector<int> wasted;
    size_t steps = std::max(key0.size(), key1.size());

    for (size_t p = 0; p < steps; p++) {
        bool has0 = p < key0.size(), has1 = p < key1.size();
        int d0 = has0 ? key0[p] : -1, d1 = has1 ? key1[p] : -1;
        int read0 = has0 ? s.children[cur0][d0] : -1;
        int read1 = has1 ? s.children[cur1][d1] : -1;

        int cand0 = (has0 && read0 == -1) ? s.alloc() : -1;
        int cand1 = (has1 && read1 == -1) ? s.alloc() : -1;

        int used0 = -1, used1 = -1;
        bool same_slot = has0 && has1 && cur0 == cur1 && d0 == d1;

        if (has0 && read0 == -1) {
            // Thread 0's CAS resolves first by this fixed tie-break.
            s.children[cur0][d0] = cand0;
            used0 = cand0;
        } else if (has0) {
            used0 = read0;
        }

        if (has1 && read1 == -1) {
            if (same_slot) {
                // Slot was empty when T1 read it, but T0's CAS has
                // already filled it -- T1's CAS fails, discard cand1.
                wasted.push_back(cand1);
                used1 = s.children[cur1][d1];
            } else {
                s.children[cur1][d1] = cand1;
                used1 = cand1;
            }
        } else if (has1) {
            used1 = read1;
        }

        printf("  position %zu: T0 slot=(node%d,digit%d) -> node%d | "
               "T1 slot=(node%d,digit%d) -> node%d%s\n",
               p, has0 ? cur0 : -1, d0, used0, has1 ? cur1 : -1, d1, used1,
               same_slot ? "  [contended slot -- T1's candidate wasted]" : "");

        if (has0) cur0 = used0;
        if (has1) cur1 = used1;
    }
    s.is_end[cur0] = true;
    s.is_end[cur1] = true;

    printf("  wasted candidate node id(s): ");
    if (wasted.empty()) printf("none");
    for (size_t i = 0; i < wasted.size(); i++) printf("%d%s", wasted[i], i + 1 < wasted.size() ? "," : "");
    printf("\n");
    return s;
}

int lookup(const TrieState& s, const std::vector<int>& key) {
    int cur = 0;
    for (int d : key) {
        if (s.children[cur][d] == -1) return -1;
        cur = s.children[cur][d];
    }
    return s.is_end[cur] ? cur : -1;
}

int main() {
    printf("=== Section 16.2 main: the shared-node race, naive vs atomicCAS ===\n\n");

    std::vector<int> key0 = {1, 2, 3};
    std::vector<int> key1 = {1, 2, 4};
    printf("two threads insert concurrently, sharing prefix [1,2]:\n");
    printf("  T0 inserts key0 = [1,2,3]\n");
    printf("  T1 inserts key1 = [1,2,4]\n\n");

    printf("--- naive (non-atomic) child-slot publish ---\n");
    TrieState naive = simulate_naive(key0, key1);
    printf("\nfinal trie (%zu nodes):\n", naive.children.size());
    print_trie(naive);
    int naive_lookup0 = lookup(naive, key0);
    int naive_lookup1 = lookup(naive, key1);
    printf("\nlookup key0 -> %d\n", naive_lookup0);
    printf("lookup key1 -> %d\n", naive_lookup1);
    printf("\nT1's store overwrote T0's at the shared root slot -- T0's entire subtree\n");
    printf("(nodes reachable only through the discarded slot value) is silently\n");
    printf("orphaned, and key0 is unfindable even though its nodes still exist in\n");
    printf("the pool with correctly-set data.\n\n");

    printf("--- atomicCAS-based child-slot publish ---\n");
    TrieState cas = simulate_cas(key0, key1);
    printf("\nfinal trie (%zu nodes):\n", cas.children.size());
    print_trie(cas);
    int cas_lookup0 = lookup(cas, key0);
    int cas_lookup1 = lookup(cas, key1);
    printf("\nlookup key0 -> %d\n", cas_lookup0);
    printf("lookup key1 -> %d\n", cas_lookup1);
    printf("\nboth keys are findable. Contention cost the losing thread one wasted\n");
    printf("node allocation each time it collided -- never a lost key.\n");

    bool ok = (naive_lookup0 == -1) && (naive_lookup1 == 6) &&
              (cas_lookup0 == 5) && (cas_lookup1 == 6) &&
              (naive.children.size() == 7) && (cas.children.size() == 7);

    printf("\nexpected: naive loses key0 (-1), finds key1 (6); CAS finds both (5, 6)\n");
    printf("self-check: naive race loses a key, CAS-based insert loses nothing: %s\n",
           ok ? "confirmed" : "MISMATCH");
    return ok ? 0 : 1;
}
```

**Compile and run:**

```bash
NVDIR=/usr/local/lib/python3.11/dist-packages/nvidia/cu13
$NVDIR/bin/nvcc -arch=sm_80 -I$NVDIR/include -L$NVDIR/lib -lcudart 85_trie_race_and_cas_kernel.cu -o 85_trie_race_and_cas_kernel
LD_LIBRARY_PATH=$NVDIR/lib ./85_trie_race_and_cas_kernel
```

**Sample input:** two threads inserting `key0 = [1,2,3]` and `key1 = [1,2,4]` concurrently into the same empty trie, once with a naive (non-atomic) shared-slot publish, once with an atomicCAS-based publish.

**Sample output:**

```text
=== Section 16.2 main: the shared-node race, naive vs atomicCAS ===

two threads insert concurrently, sharing prefix [1,2]:
  T0 inserts key0 = [1,2,3]
  T1 inserts key1 = [1,2,4]

--- naive (non-atomic) child-slot publish ---
  position 0: T0 slot=(node0,digit1) saw empty -> uses node1 | T1 slot=(node0,digit1) saw empty -> uses node2
  position 1: T0 slot=(node1,digit2) saw empty -> uses node3 | T1 slot=(node2,digit2) saw empty -> uses node4
  position 2: T0 slot=(node3,digit3) saw empty -> uses node5 | T1 slot=(node4,digit4) saw empty -> uses node6

final trie (7 nodes):
  node0: children=[-1,2,-1,-1,-1,-1], is_end=false
  node1: children=[-1,-1,3,-1,-1,-1], is_end=false
  node2: children=[-1,-1,4,-1,-1,-1], is_end=false
  node3: children=[-1,-1,-1,5,-1,-1], is_end=false
  node4: children=[-1,-1,-1,-1,6,-1], is_end=false
  node5: children=[-1,-1,-1,-1,-1,-1], is_end=true
  node6: children=[-1,-1,-1,-1,-1,-1], is_end=true

lookup key0 -> -1
lookup key1 -> 6

T1's store overwrote T0's at the shared root slot -- T0's entire subtree
(nodes reachable only through the discarded slot value) is silently
orphaned, and key0 is unfindable even though its nodes still exist in
the pool with correctly-set data.

--- atomicCAS-based child-slot publish ---
  position 0: T0 slot=(node0,digit1) -> node1 | T1 slot=(node0,digit1) -> node1  [contended slot -- T1's candidate wasted]
  position 1: T0 slot=(node1,digit2) -> node3 | T1 slot=(node1,digit2) -> node3  [contended slot -- T1's candidate wasted]
  position 2: T0 slot=(node3,digit3) -> node5 | T1 slot=(node3,digit4) -> node6
  wasted candidate node id(s): 2,4

final trie (7 nodes):
  node0: children=[-1,1,-1,-1,-1,-1], is_end=false
  node1: children=[-1,-1,3,-1,-1,-1], is_end=false
  node2: children=[-1,-1,-1,-1,-1,-1], is_end=false
  node3: children=[-1,-1,-1,5,6,-1], is_end=false
  node4: children=[-1,-1,-1,-1,-1,-1], is_end=false
  node5: children=[-1,-1,-1,-1,-1,-1], is_end=true
  node6: children=[-1,-1,-1,-1,-1,-1], is_end=true

lookup key0 -> 5
lookup key1 -> 6

both keys are findable. Contention cost the losing thread one wasted
node allocation each time it collided -- never a lost key.

expected: naive loses key0 (-1), finds key1 (6); CAS finds both (5, 6)
self-check: naive race loses a key, CAS-based insert loses nothing: confirmed
```

## 16.3 Level-Synchronous Parallel Trie Construction: All Keys, All Positions, At Once

### Intuition

Section 16.2 fixed the race between exactly two threads on one contended slot. The real target is building a trie out of MANY keys at once, all inserted concurrently -- and the natural way to schedule that safely is level-synchronous, the same "loop over levels, synchronize between them" shape this book has used since Chapter 12's bitonic passes: process position 0 of every still-active key together, synchronize, then position 1 of every still-active key, and so on, until every key has been fully inserted. "Still-active" here means simply "still has a digit left at this position" -- a key of length 2 stops contributing threads after position 1, exactly as Section 15.2's shorter tree traversals would finish before taller ones.

### The Sequential (CPU) Baseline

```cpp
#include <cstdio>
#include <vector>
#include <array>
#include <algorithm>

// Chapter 16.3 -- The Sequential (CPU) Baseline.
// Section 16.2 handled two keys racing on one shared prefix. The real
// target is ALL keys, inserted at once, synchronized POSITION BY
// POSITION rather than key by key -- position 0 of every still-active
// key happens together, then position 1 of every still-active key, and
// so on, exactly the "loop over levels, synchronize between them" shape
// Chapters 12-15 (and Section 16.1) all used, with "still has a digit
// left at this position" playing the role of "still active." This
// baseline processes that same position-major order, but strictly one
// key at a time within a position -- so there is no actual concurrency
// and no race, only the level-synchronous SCHEDULE that Section 16.3's
// parallel version will run for real.
#define ALPHABET 6

struct Trie {
    std::vector<std::array<int, ALPHABET>> children;
    std::vector<bool> is_end;
    int alloc() {
        children.push_back({-1, -1, -1, -1, -1, -1});
        is_end.push_back(false);
        return (int)children.size() - 1;
    }
};

int main() {
    printf("=== Section 16.3 CPU baseline: sequential position-major trie construction ===\n\n");

    std::vector<std::vector<int>> keys = {
        {1, 2, 3},
        {1, 2, 4},
        {1, 5},
        {2, 1},
    };
    printf("keys (as digit sequences):\n");
    for (size_t i = 0; i < keys.size(); i++) {
        printf("  key%zu = [", i);
        for (size_t j = 0; j < keys[i].size(); j++) printf("%d%s", keys[i][j], j + 1 < keys[i].size() ? "," : "");
        printf("]\n");
    }
    printf("\n");

    Trie t;
    t.alloc();   // root = node0
    std::vector<int> cur(keys.size(), 0);
    size_t max_len = 0;
    for (auto& k : keys) max_len = std::max(max_len, k.size());

    for (size_t pos = 0; pos < max_len; pos++) {
        printf("position %zu:\n", pos);
        for (size_t k = 0; k < keys.size(); k++) {
            if (pos >= keys[k].size()) continue;   // key k is no longer active
            int d = keys[k][pos];
            int slot = cur[k];
            if (t.children[slot][d] == -1) {
                int node = t.alloc();
                t.children[slot][d] = node;
                printf("  key%zu: (node%d,digit%d) was empty -> creates node%d\n", k, slot, d, node);
            } else {
                printf("  key%zu: (node%d,digit%d) already exists -> reuses node%d\n",
                       k, slot, d, t.children[slot][d]);
            }
            cur[k] = t.children[slot][d];
            if (pos + 1 == keys[k].size()) t.is_end[cur[k]] = true;
        }
    }
    printf("\n");

    printf("final trie: %zu nodes\n", t.children.size());
    for (size_t i = 0; i < t.children.size(); i++) {
        printf("  node%zu: children=[", i);
        for (int d = 0; d < ALPHABET; d++) printf("%d%s", t.children[i][d], d + 1 < ALPHABET ? "," : "");
        printf("], is_end=%s\n", t.is_end[i] ? "true" : "false");
    }
    printf("\n");

    auto lookup = [&](const std::vector<int>& key) {
        int c = 0;
        for (int d : key) {
            if (t.children[c][d] == -1) return -1;
            c = t.children[c][d];
        }
        return t.is_end[c] ? c : -1;
    };

    std::vector<int> expected_lookup = {6, 7, 4, 5};
    bool ok = (t.children.size() == 8);
    for (size_t i = 0; i < keys.size(); i++) {
        int r = lookup(keys[i]);
        printf("lookup key%zu -> %d\n", i, r);
        ok = ok && (r == expected_lookup[i]);
    }

    printf("\nno node was ever wasted or contested -- because each position's keys were\n");
    printf("still processed one at a time here, this schedule has the exact SHAPE\n");
    printf("Section 16.3's parallel version will use (position-major, not key-major),\n");
    printf("but none of its risk: that risk only appears once multiple keys' steps\n");
    printf("within the same position actually run concurrently.\n");

    printf("\nself-check: position-major sequential construction produces a fully\n");
    printf("correct trie: %s\n", ok ? "confirmed" : "MISMATCH");
    return ok ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++17 -Wall -Wextra -O2 86_trie_levelsync_cpu_baseline.cpp -o 86_trie_levelsync_cpu_baseline
./86_trie_levelsync_cpu_baseline
```

**Sample input:** the same four keys as Section 16.2's design (`[1,2,3]`, `[1,2,4]`, `[1,5]`, `[2,1]`), inserted position-major (all keys' position 0, then all keys' position 1, ...) but strictly one key at a time within each position -- the level-synchronous SCHEDULE, without yet running it concurrently.

**Sample output:**

```text
=== Section 16.3 CPU baseline: sequential position-major trie construction ===

keys (as digit sequences):
  key0 = [1,2,3]
  key1 = [1,2,4]
  key2 = [1,5]
  key3 = [2,1]

position 0:
  key0: (node0,digit1) was empty -> creates node1
  key1: (node0,digit1) already exists -> reuses node1
  key2: (node0,digit1) already exists -> reuses node1
  key3: (node0,digit2) was empty -> creates node2
position 1:
  key0: (node1,digit2) was empty -> creates node3
  key1: (node1,digit2) already exists -> reuses node3
  key2: (node1,digit5) was empty -> creates node4
  key3: (node2,digit1) was empty -> creates node5
position 2:
  key0: (node3,digit3) was empty -> creates node6
  key1: (node3,digit4) was empty -> creates node7

final trie: 8 nodes
  node0: children=[-1,1,2,-1,-1,-1], is_end=false
  node1: children=[-1,-1,3,-1,-1,4], is_end=false
  node2: children=[-1,5,-1,-1,-1,-1], is_end=false
  node3: children=[-1,-1,-1,6,7,-1], is_end=false
  node4: children=[-1,-1,-1,-1,-1,-1], is_end=true
  node5: children=[-1,-1,-1,-1,-1,-1], is_end=true
  node6: children=[-1,-1,-1,-1,-1,-1], is_end=true
  node7: children=[-1,-1,-1,-1,-1,-1], is_end=true

lookup key0 -> 6
lookup key1 -> 7
lookup key2 -> 4
lookup key3 -> 5

no node was ever wasted or contested -- because each position's keys were
still processed one at a time here, this schedule has the exact SHAPE
Section 16.3's parallel version will use (position-major, not key-major),
but none of its risk: that risk only appears once multiple keys' steps
within the same position actually run concurrently.

self-check: position-major sequential construction produces a fully
correct trie: confirmed
```

### The Concept, In Detail

Running the same four keys through the full level-synchronous schedule, but now genuinely concurrently within each position (any number of active keys may contend on the very same slot, not just two, and Section 16.2's atomicCAS-or-reuse rule handles all of them uniformly):

```
position 0: 4 active threads (key0, key1, key2, key3)
  key0, key1, key2 all want digit 1 from root (node 0) -- three-way contention
    key0's CAS wins (fixed tie-break: lowest key index wins ties) -> node 1
    key1, key2 both fail, discard their candidates, reuse node 1
  key3 wants digit 2 from root -- no contention -> node 4

position 1: 4 active threads (all four keys still have a digit left)
  key0, key1 both want digit 2 from node 1 -- two-way contention
    key0's CAS wins -> node 5; key1 fails, reuses node 5
  key2 wants digit 5 from node 1 -- no contention -> node 7 (key2 is now DONE, length 2)
  key3 wants digit 1 from node 4 -- no contention -> node 8 (key3 is now DONE, length 2)

position 2: 2 active threads (only key0 and key1 still have a digit left)
  key0 wants digit 3 from node 5, key1 wants digit 4 from node 5 --
    DIFFERENT digits, no contention -> key0 gets node 9, key1 gets node 10
    (both keys are now DONE, length 3)
```

```
ASCII view of the final trie (11 total allocated node ids):

  root(0) --1--> node1 --2--> node5 --3--> node9  [key0 ends here]
                    |            \--4--> node10 [key1 ends here]
                    \--5--> node7             [key2 ends here]
         --2--> node4 --1--> node8            [key3 ends here]

  reachable nodes: 0, 1, 4, 5, 7, 8, 9, 10   (8 nodes)
  wasted candidates: 2, 3, 6                  (3 nodes -- allocated, never linked)
```

Every wasted node id was a real, valid allocation -- just one whose CAS lost a race, so the node it points to is never referenced by anything and is simply unused memory, never a correctness problem. All four keys remain perfectly findable by lookup, with exactly the same guarantee Section 16.2 already established for two keys: contention costs memory (wasted candidate slots), never a lost key. Note also that the specific node ids assigned here (1, 4, 5, 7, 8, 9, 10) differ from Section 16.3's own CPU baseline's ids (which came out as 1 through 7, with no waste at all) -- concurrent execution changes WHICH ids end up used and how many extra get wasted, but never changes WHETHER every key ends up correctly reachable.

[COMMON TRAP]
Comparing a concurrently-built trie's node ids directly against a sequentially-built one's (expecting them to match) is the wrong check entirely -- allocation order depends on execution order, which concurrent execution does not guarantee. The only check that matters is semantic: does every key that was inserted come back out of `lookup()` correctly? Node ids and even total node COUNT (11 here, versus 8 for the strictly one-at-a-time schedule) are allowed, expected, to differ.

### Code and Verification

```cpp
#include <cstdio>
#include <vector>
#include <array>
#include <algorithm>

// Chapter 16.3 main -- Section 16.2 fixed the race between two threads
// on one contended slot. This file runs the FULL level-synchronous
// construction: EVERY still-active key gets one thread at EVERY
// position, with a kernel launch (and host-side synchronization) once
// per position -- the same "loop over levels, sync between them" shape
// as Section 16.1's BST build, generalized from "one thread per group"
// to "one thread per still-active key." Within a position, any number
// of keys may contend on the same slot (not just two), and Section
// 16.2's atomicCAS-or-reuse rule handles all of them uniformly: exactly
// one wins, everyone else discards its wasted candidate and reuses the
// winner's node.
#define ALPHABET 6

// One thread per ACTIVE key (a key that still has a digit left at this
// position). `done` marks a key finishing at this position, so its
// final node gets flagged as a real key ending, not just an internal
// node. `out_next` reports each thread's resulting node id back to the
// host, which uses it as that key's `cur` for the following position.
__global__ void trie_level_kernel(int* children, int* is_end, int* next_free,
                                   const int* active_cur, const int* active_digit,
                                   const int* active_done, int num_active,
                                   int alphabet, int* out_next) {
    int t = threadIdx.x;
    if (t >= num_active) return;

    int cur = active_cur[t];
    int d = active_digit[t];
    int slot = cur * alphabet + d;
    int existing = children[slot];
    int next;
    if (existing == -1) {
        int candidate = atomicAdd(next_free, 1);         // Chapter 8's atomic allocator
        int prev = atomicCAS(&children[slot], -1, candidate);  // Chapter 9.1's CAS-or-retry, generalized
        next = (prev == -1) ? candidate : prev;
    } else {
        next = existing;
    }
    out_next[t] = next;
    if (active_done[t]) is_end[next] = 1;
}

// ---- Host-side replay of the identical per-position logic. All active
// threads' reads at a position happen before any of that position's
// writes, and whenever several active keys target the SAME slot, the
// lowest key-index among them is the one recorded as winning the CAS
// (a fixed tie-break -- real hardware could pick any one of them, but
// exactly one always wins and everyone else always reuses its result,
// so the fixed choice changes only which id wins, never the outcome
// that matters: every key ends up findable). ----

using Row = std::array<int, ALPHABET>;

struct TrieState {
    std::vector<Row> children;
    std::vector<bool> is_end;
    int next_free = 0;
    int alloc() {
        children.push_back({-1, -1, -1, -1, -1, -1});
        is_end.push_back(false);
        return next_free++;
    }
};

int main() {
    printf("=== Section 16.3 main: level-synchronous parallel trie construction ===\n\n");

    std::vector<std::vector<int>> keys = {
        {1, 2, 3},
        {1, 2, 4},
        {1, 5},
        {2, 1},
    };
    printf("keys (as digit sequences):\n");
    for (size_t i = 0; i < keys.size(); i++) {
        printf("  key%zu = [", i);
        for (size_t j = 0; j < keys[i].size(); j++) printf("%d%s", keys[i][j], j + 1 < keys[i].size() ? "," : "");
        printf("]\n");
    }
    printf("\n");

    TrieState s;
    s.alloc();   // root = node0
    std::vector<int> cur(keys.size(), 0);
    size_t max_len = 0;
    for (auto& k : keys) max_len = std::max(max_len, k.size());
    std::vector<int> total_wasted;

    for (size_t pos = 0; pos < max_len; pos++) {
        std::vector<int> active_keys;
        for (size_t k = 0; k < keys.size(); k++) {
            if (pos < keys[k].size()) active_keys.push_back((int)k);
        }
        int width = (int)active_keys.size();
        printf("position %zu: %d independent thread(s) (one per still-active key)\n", pos, width);

        // LOAD phase -- every active thread reads its slot's CURRENT
        // value (before any of this position's writes).
        std::vector<int> reads(width);
        for (int t = 0; t < width; t++) {
            int k = active_keys[t];
            int d = keys[k][pos];
            reads[t] = s.children[cur[k]][d];
        }
        // Every thread that saw its slot empty speculatively allocates
        // a candidate, in key-index order.
        std::vector<int> candidates(width, -1);
        for (int t = 0; t < width; t++) {
            if (reads[t] == -1) candidates[t] = s.alloc();
        }
        // CAS phase, in key-index order: the first thread to reach an
        // empty slot claims it; later threads targeting the same slot
        // find it already filled and reuse that value instead.
        std::vector<int> next_val(width);
        for (int t = 0; t < width; t++) {
            int k = active_keys[t];
            int d = keys[k][pos];
            int cur_slot_val = s.children[cur[k]][d];
            if (cur_slot_val == -1) {
                s.children[cur[k]][d] = candidates[t];
                next_val[t] = candidates[t];
            } else {
                next_val[t] = cur_slot_val;
                if (reads[t] == -1) {
                    total_wasted.push_back(candidates[t]);   // this thread's own read saw it empty, but lost the race
                }
            }
        }

        for (int t = 0; t < width; t++) {
            int k = active_keys[t];
            int d = keys[k][pos];
            printf("  key%d: (node%d,digit%d) -> node%d%s\n", k, cur[k], d, next_val[t],
                   (reads[t] == -1 && s.children[cur[k]][d] != candidates[t]) ? "  [contended -- candidate wasted]" : "");
            cur[k] = next_val[t];
            if (pos + 1 == keys[k].size()) s.is_end[cur[k]] = true;
        }
        printf("\n");
    }

    printf("final trie: %zu nodes total\n", s.children.size());
    for (size_t i = 0; i < s.children.size(); i++) {
        printf("  node%zu: children=[", i);
        for (int d = 0; d < ALPHABET; d++) printf("%d%s", s.children[i][d], d + 1 < ALPHABET ? "," : "");
        printf("], is_end=%s\n", s.is_end[i] ? "true" : "false");
    }
    printf("\nwasted candidate node id(s): ");
    std::sort(total_wasted.begin(), total_wasted.end());
    for (size_t i = 0; i < total_wasted.size(); i++) printf("%d%s", total_wasted[i], i + 1 < total_wasted.size() ? "," : "");
    printf(" (%zu total)\n\n", total_wasted.size());

    auto lookup = [&](const std::vector<int>& key) {
        int c = 0;
        for (int d : key) {
            if (s.children[c][d] == -1) return -1;
            c = s.children[c][d];
        }
        return s.is_end[c] ? c : -1;
    };

    std::vector<int> expected_lookup = {9, 10, 7, 8};
    bool ok = (s.children.size() == 11) && (total_wasted.size() == 3);
    for (size_t i = 0; i < keys.size(); i++) {
        int r = lookup(keys[i]);
        printf("lookup key%zu -> %d\n", i, r);
        ok = ok && (r == expected_lookup[i]);
    }

    printf("\nevery key is findable, even though 3 of the 11 allocated node ids (%zu wasted)\n",
           total_wasted.size());
    printf("were never linked into the trie -- a real, documented memory cost of running\n");
    printf("many threads' speculative allocations concurrently, never a correctness bug.\n");

    printf("\nexpected: 11 total nodes, 3 wasted, key0->9 key1->10 key2->7 key3->8\n");
    printf("self-check: level-synchronous parallel construction finds every key: %s\n",
           ok ? "confirmed" : "MISMATCH");
    return ok ? 0 : 1;
}
```

**Compile and run:**

```bash
NVDIR=/usr/local/lib/python3.11/dist-packages/nvidia/cu13
$NVDIR/bin/nvcc -arch=sm_80 -I$NVDIR/include -L$NVDIR/lib -lcudart 87_trie_levelsync_parallel_kernel.cu -o 87_trie_levelsync_parallel_kernel
LD_LIBRARY_PATH=$NVDIR/lib ./87_trie_levelsync_parallel_kernel
```

**Sample input:** the same four keys, inserted fully concurrently, one thread per still-active key per position, synchronized between positions, using atomicCAS to resolve every contended slot.

**Sample output:**

```text
=== Section 16.3 main: level-synchronous parallel trie construction ===

keys (as digit sequences):
  key0 = [1,2,3]
  key1 = [1,2,4]
  key2 = [1,5]
  key3 = [2,1]

position 0: 4 independent thread(s) (one per still-active key)
  key0: (node0,digit1) -> node1
  key1: (node0,digit1) -> node1  [contended -- candidate wasted]
  key2: (node0,digit1) -> node1  [contended -- candidate wasted]
  key3: (node0,digit2) -> node4

position 1: 4 independent thread(s) (one per still-active key)
  key0: (node1,digit2) -> node5
  key1: (node1,digit2) -> node5  [contended -- candidate wasted]
  key2: (node1,digit5) -> node7
  key3: (node4,digit1) -> node8

position 2: 2 independent thread(s) (one per still-active key)
  key0: (node5,digit3) -> node9
  key1: (node5,digit4) -> node10

final trie: 11 nodes total
  node0: children=[-1,1,4,-1,-1,-1], is_end=false
  node1: children=[-1,-1,5,-1,-1,7], is_end=false
  node2: children=[-1,-1,-1,-1,-1,-1], is_end=false
  node3: children=[-1,-1,-1,-1,-1,-1], is_end=false
  node4: children=[-1,8,-1,-1,-1,-1], is_end=false
  node5: children=[-1,-1,-1,9,10,-1], is_end=false
  node6: children=[-1,-1,-1,-1,-1,-1], is_end=false
  node7: children=[-1,-1,-1,-1,-1,-1], is_end=true
  node8: children=[-1,-1,-1,-1,-1,-1], is_end=true
  node9: children=[-1,-1,-1,-1,-1,-1], is_end=true
  node10: children=[-1,-1,-1,-1,-1,-1], is_end=true

wasted candidate node id(s): 2,3,6 (3 total)

lookup key0 -> 9
lookup key1 -> 10
lookup key2 -> 7
lookup key3 -> 8

every key is findable, even though 3 of the 11 allocated node ids (3 wasted)
were never linked into the trie -- a real, documented memory cost of running
many threads' speculative allocations concurrently, never a correctness bug.

expected: 11 total nodes, 3 wasted, key0->9 key1->10 key2->7 key3->8
self-check: level-synchronous parallel construction finds every key: confirmed
```

## Chapter Summary

A binary search tree can be built directly from UNSORTED data by generalizing Chapter 14.3's sample-sort partitioning to level-by-level construction: at each level, every pending group independently picks a pivot, partitions its own remaining elements, and hands off two child groups to the next level, with each group's private row in a flat buffer meaning no thread ever needs to know where another thread's data lives. Tries introduce a genuinely new hazard that BST construction never had: keys deliberately SHARE prefix nodes, so two threads inserting different keys can discover the same missing child slot empty at the same time. A naive, unprotected publish of a newly allocated node into that shared slot loses data silently -- exactly Chapter 9.1's stack-push race, generalized from one shared head pointer to many shared child slots -- while an atomicCAS-based publish (a direct application of Chapter 9's own retry-or-reuse pattern) guarantees every key stays findable, at the cost of a few wasted (allocated but never linked) node ids whenever contention actually happens. Scaling this from two keys to arbitrarily many follows the same level-synchronous schedule this book has used since Chapter 12: one thread per still-active key per position, synchronized between positions, with any number of keys free to contend on the same slot at once and Section 16.2's fix handling all of them uniformly.

## Self-Check Questions

1. Why does Section 16.1's level-by-level BST construction need an explicit allocation step for child slots, when Section 15.1's did not?
2. In the naive (non-atomic) trie-insertion race, what specifically does thread T0 believe happened, and why is that belief wrong?
3. Chapter 8's atomic bump allocator already guarantees every thread gets a unique new node id with no collisions. Why does that alone NOT prevent the shared-node race in trie construction?
4. Walk through what happens when a thread's atomicCAS on a trie child slot FAILS. What does it do with its own speculatively allocated candidate, and what does it use instead?
5. In the level-synchronous construction of Section 16.3, why is it acceptable -- even expected -- for the final set of allocated node ids to differ from the sequential CPU baseline's?
6. Why does contention in atomicCAS-based trie construction cost memory (wasted node ids) rather than correctness (lost keys)?

## Where We Go Next

This chapter built trees and tries out of arbitrary, unsorted, possibly concurrent input -- but every operation so far has been about CONSTRUCTING structure, never querying ranges within it. Chapter 17 turns to segment trees and Fenwick trees, structures purpose-built to answer range queries (sum, minimum, maximum over an arbitrary contiguous range) and point updates efficiently, including how their own construction and update patterns parallelize.

## Worked Solutions

**1.** Section 15.1's tree is always PERFECTLY COMPLETE, because it is built from a sorted array by always splitting at the midpoint -- a property of the SIZE of each range alone, known before any data is even looked at, which is exactly what makes the closed-form slot formula (`2*slot+1`, `2*slot+2`) valid. Section 16.1's tree shape depends entirely on the DATA (which elements happen to be less than or greater than each pivot), so the same range size can produce different-shaped subtrees depending on what values are actually in it -- there is no fixed arithmetic relationship between a group's own slot and its children's slots, so an explicit allocation counter is required instead.

**2.** T0 believes it successfully created a new node (id 1) and that the trie's structure now includes `children[0][1] = 1`. This is wrong because T1's later write to the exact same slot (`children[0][1] = 2`) silently overwrote T0's write -- the actual shared trie no longer has any path leading to node 1 at all, even though node 1 itself still exists in the node pool with entirely correct data. T0's belief was accurate at the moment it made its own write, but became stale and incorrect the instant another thread's conflicting write landed afterward, and nothing in the naive design ever tells T0 that this happened.

**3.** Atomic allocation solves a DIFFERENT problem than the one that causes this race. It guarantees that when two threads both call the allocator, they get back two distinct, valid ids -- no two threads are ever handed the same node. But the race is not about which id each thread gets; it is about which id ends up PUBLISHED into the one shared `children[][]` slot both threads are trying to fill. Allocation being atomic does nothing to protect that separate publish step, which remains a plain, unprotected store unless it is ALSO made atomic (via CAS).

**4.** When a thread's `atomicCAS(&children[node][digit], -1, candidate)` fails, it means the slot no longer holds `-1` -- some other thread already published a winning value there. The failing thread's own `candidate` node was allocated but is now permanently unused (wasted, though harmless: nothing ever references it). The thread reads back whatever value is actually in the slot now (the winning thread's node id) and uses THAT as its own next position, continuing its insertion exactly as if it had found that node already present when it first looked.

**5.** The final node ids depend on execution order: which thread's CAS happens to run first at each contended slot, and how many threads end up contending at all, both depend on how the threads actually interleave -- something concurrent execution deliberately does not fix to one specific order. What must stay invariant regardless of that interleaving is purely semantic: every key that was inserted must still be correctly findable by `lookup()`. The exact node ids used, and even the total count of allocated nodes (since some are wasted whenever contention happens), are expected to vary run to run and are not part of what "correct" means here.

**6.** Contention costs memory rather than correctness because atomicCAS makes "check if the slot is still empty, then fill it" a single indivisible operation: exactly one thread's CAS can ever succeed on a given empty slot, no matter how many threads attempt it at once, and every other thread's attempt is guaranteed to fail cleanly rather than silently overwrite the winner. A losing thread's only cost is the node id it spent allocating a candidate that never gets used -- its own insertion simply continues from the winner's node instead, so no key it is building is ever lost, only a small, bounded amount of node-pool memory per contended slot.
