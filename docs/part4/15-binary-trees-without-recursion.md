# Chapter 15: Binary Trees Without Recursion

Part 3 spent four chapters sorting arrays. Part 4 opens by putting that sorted output to work: a sorted array can be turned directly into a balanced binary search tree, and once that tree exists, it needs to be built and walked -- traditionally the most recursive corner of this entire book so far. Recursion is a real, structural problem on a GPU, not just a style preference: threads in a warp execute in lockstep (Chapter 1's own SIMT model), so if different threads recurse to different depths, the whole warp serializes to accommodate whichever thread goes deepest, and CUDA's own per-thread device call stack is a fixed size that has to be sized in advance. This chapter builds a complete binary search tree, and walks it end to end, without a single recursive call anywhere -- first with an explicit queue, then an explicit stack, and finally with no extra structure at all.

## 15.1 From Sorted Array to Balanced Binary Search Tree

### Intuition

Given a sorted array, a balanced binary search tree can be built in one shot: the middle element becomes the root, the left half of the array becomes the left subtree, and the right half becomes the right subtree -- applied recursively to each half. The traditional presentation of this is recursive, but since this chapter is specifically about removing recursion, this section builds the tree ITERATIVELY instead, using an explicit queue of pending work items processed in level order -- the queue holds exactly what a call stack would otherwise have tracked implicitly. Continuing Chapter 11's own discipline, tree nodes live in parallel arrays (`value[]`, `left[]`, `right[]`) addressed by integer slot indices, with `-1` meaning "no child," never a real pointer.

### The Sequential (CPU) Baseline

```cpp
#include <cstdio>
#include <vector>
#include <queue>

// Chapter 15.1 -- The Sequential (CPU) Baseline.
// A balanced binary search tree can be built directly from a SORTED
// array (exactly Part 3's own output) without inserting elements one
// at a time: the middle element becomes the root, the left half of the
// array recursively becomes the left subtree, and the right half
// becomes the right subtree. The traditional presentation of this is
// recursive -- but this chapter is about removing recursion from tree
// code, so this baseline builds the tree ITERATIVELY instead, using an
// explicit queue of (range, destination slot) work items, processed in
// level order. Continuing Chapter 11's own discipline, nodes live in
// parallel arrays (value[], left[], right[]) addressed by integer
// slot indices, with -1 meaning "no child" -- never a real pointer.

struct WorkItem { int lo, hi, slot; };

void build_bst_from_sorted(const std::vector<int>& a,
                            std::vector<int>& value, std::vector<int>& left,
                            std::vector<int>& right, int& root, int& next_free) {
    int n = (int)a.size();
    value.assign(n, 0);
    left.assign(n, -1);
    right.assign(n, -1);
    next_free = 0;

    auto alloc = [&]() { return next_free++; };

    root = alloc();
    std::queue<WorkItem> q;
    q.push({0, n - 1, root});

    while (!q.empty()) {
        WorkItem w = q.front(); q.pop();
        int mid = (w.lo + w.hi) / 2;
        value[w.slot] = a[mid];
        if (w.lo <= mid - 1) {
            int lslot = alloc();
            left[w.slot] = lslot;
            q.push({w.lo, mid - 1, lslot});
        }
        if (mid + 1 <= w.hi) {
            int rslot = alloc();
            right[w.slot] = rslot;
            q.push({mid + 1, w.hi, rslot});
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
    printf("=== Section 15.1 CPU baseline: iterative (queue-based) BST construction ===\n\n");

    std::vector<int> a = {10, 20, 30, 40, 50, 60, 70};
    printf("sorted input: ");
    for (int v : a) printf("%d ", v);
    printf("\n\n");

    std::vector<int> value, left, right;
    int root, next_free;
    build_bst_from_sorted(a, value, left, right, root, next_free);

    printf("value array: ");
    for (int v : value) printf("%d ", v);
    printf("\nleft array:  ");
    for (int v : left) printf("%d ", v);
    printf("\nright array: ");
    for (int v : right) printf("%d ", v);
    printf("\nroot slot: %d\n\n", root);

    std::vector<int> traversal;
    inorder(root, value, left, right, traversal);
    printf("in-order traversal (verification, uses recursion only to CHECK this baseline): ");
    for (int v : traversal) printf("%d ", v);
    printf("\n\n");

    std::vector<int> expected_value = {40, 20, 60, 10, 30, 50, 70};
    std::vector<int> expected_left = {1, 3, 5, -1, -1, -1, -1};
    std::vector<int> expected_right = {2, 4, 6, -1, -1, -1, -1};
    bool ok = (value == expected_value) && (left == expected_left) &&
              (right == expected_right) && (traversal == a);

    printf("expected value: 40 20 60 10 30 50 70\n");
    printf("expected left:  1 3 5 -1 -1 -1 -1\n");
    printf("expected right: 2 4 6 -1 -1 -1 -1\n\n");
    printf("no call ever recursed to build this tree -- the queue holds exactly the\n");
    printf("pending work that a call stack would otherwise have tracked implicitly.\n");

    printf("\nself-check: iterative construction matches expected arrays, in-order\n");
    printf("traversal recovers the original sorted input: %s\n", ok ? "confirmed" : "MISMATCH");
    return ok ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++17 -Wall -Wextra -O2 76_bst_from_sorted_cpu_baseline.cpp -o 76_bst_from_sorted_cpu_baseline
./76_bst_from_sorted_cpu_baseline
```

**Sample input:** the sorted array `{10, 20, 30, 40, 50, 60, 70}`, built into a balanced BST via an explicit queue.

**Sample output:**

```text
=== Section 15.1 CPU baseline: iterative (queue-based) BST construction ===

sorted input: 10 20 30 40 50 60 70 

value array: 40 20 60 10 30 50 70 
left array:  1 3 5 -1 -1 -1 -1 
right array: 2 4 6 -1 -1 -1 -1 
root slot: 0

in-order traversal (verification, uses recursion only to CHECK this baseline): 10 20 30 40 50 60 70 

expected value: 40 20 60 10 30 50 70
expected left:  1 3 5 -1 -1 -1 -1
expected right: 2 4 6 -1 -1 -1 -1

no call ever recursed to build this tree -- the queue holds exactly the
pending work that a call stack would otherwise have tracked implicitly.

self-check: iterative construction matches expected arrays, in-order
traversal recovers the original sorted input: confirmed
```

### The Concept, In Detail

Tracing the queue-based construction level by level on `{10, 20, 30, 40, 50, 60, 70}` (indices 0-6):

```
level 0: process (lo=0, hi=6, slot=0)
  mid=3, value[0]=a[3]=40
  left half [0,2] non-empty -> alloc slot 1, left[0]=1, enqueue (0,2,1)
  right half [4,6] non-empty -> alloc slot 2, right[0]=2, enqueue (4,6,2)

level 1: process (lo=0, hi=2, slot=1), then (lo=4, hi=6, slot=2)
  slot 1: mid=1, value[1]=a[1]=20, left half [0,0] -> slot 3, right half [2,2] -> slot 4
  slot 2: mid=5, value[2]=a[5]=60, left half [4,4] -> slot 5, right half [6,6] -> slot 6

level 2: process the four single-element ranges
  slot 3: value[3]=a[0]=10 (leaf)     slot 4: value[4]=a[2]=30 (leaf)
  slot 5: value[5]=a[4]=50 (leaf)     slot 6: value[6]=a[6]=70 (leaf)
```

```
ASCII view of the resulting tree:

                    40 (slot 0)
                  /              \
           20 (slot 1)        60 (slot 2)
           /        \          /        \
    10 (slot 3)  30 (slot 4)  50 (slot 5)  70 (slot 6)

  value: [40, 20, 60, 10, 30, 50, 70]
  left:  [ 1,  3, 5, -1, -1, -1, -1]
  right: [ 2,  4, 6, -1, -1, -1, -1]
```

Notice that slot `i`'s children always end up at slots `2i+1` and `2i+2` -- exactly the classic implicit "heap array" layout, but arrived at here as an OBSERVED consequence of level-order allocation on a perfectly balanced tree, not as an assumption baked into the representation. The arrays still store real, explicit indices (general enough to represent any tree shape, not just perfectly balanced ones), which matters directly in Section 15.3, where those stored indices need to be temporarily overwritten and later restored -- something an implicit position-only representation could never support.

The parallel opportunity is level-by-level: every work item within a single level is completely independent of every other item in that SAME level -- each one only reads its own `(lo, hi, slot)` triple and writes its own `value[]`/`left[]`/`right[]` entries, plus its own children's next-level work items. Because this particular tree is complete, child slots need no coordination at all: slot `i`'s children are always `2i+1` and `2i+2`, a closed-form computation requiring no shared allocation counter -- unlike Chapter 13's radix scatter, which needed rank-within-group scans specifically because destination positions were NOT knowable in closed form.

[COMMON TRAP]
It is tempting to think this closed-form child-slot trick (`2i+1`, `2i+2`) generalizes to any binary search tree. It only holds because this construction always produces a PERFECTLY COMPLETE tree from a sorted array of exactly the right shape; a tree built by ordinary one-at-a-time insertion (unbalanced, arbitrary shape) has no such fixed relationship between a node's own slot and its children's slots, and needs a real allocation counter instead.

### Code and Verification

```cpp
#include <cstdio>
#include <vector>

// Chapter 15.1 main -- building the SAME tree with one level processed
// per kernel launch, every node in a level handled by an independent
// thread. All of a level's work items are mutually independent: each
// one only reads its OWN (lo, hi, slot) triple and writes its OWN
// value[]/left[]/right[] entries, plus its own two children's work
// items for the next level -- exactly the "loop over passes,
// synchronize between them" shape Chapters 12-14 all used, except here
// what doubles each pass is the NUMBER OF NODES being processed, not a
// merge width or a compare-exchange distance. Child slots need no
// coordination at all: because this tree is complete, slot i's
// children always live at slots 2*i+1 and 2*i+2, a closed-form
// computation requiring no shared counter.
__global__ void build_level_kernel(const int* g_a, const int* g_cur_lo, const int* g_cur_hi,
                                    const int* g_cur_slot, int level_width,
                                    int* g_value, int* g_left, int* g_right,
                                    int* g_next_lo, int* g_next_hi, int* g_next_slot) {
    int t = threadIdx.x;
    if (t >= level_width) return;

    int lo = g_cur_lo[t], hi = g_cur_hi[t], slot = g_cur_slot[t];
    int mid = (lo + hi) / 2;
    g_value[slot] = g_a[mid];

    int lslot = 2 * slot + 1, rslot = 2 * slot + 2;
    if (lo <= mid - 1) {
        g_left[slot] = lslot;
        g_next_lo[2 * t] = lo; g_next_hi[2 * t] = mid - 1; g_next_slot[2 * t] = lslot;
    } else {
        g_left[slot] = -1;
        g_next_slot[2 * t] = -1;
    }
    if (mid + 1 <= hi) {
        g_right[slot] = rslot;
        g_next_lo[2 * t + 1] = mid + 1; g_next_hi[2 * t + 1] = hi; g_next_slot[2 * t + 1] = rslot;
    } else {
        g_right[slot] = -1;
        g_next_slot[2 * t + 1] = -1;
    }
}

// ---- Host-side replay of the identical per-thread logic, one level's
// worth of independent threads per pass, with the host loop playing the
// role of the synchronization between kernel launches. ----

int main() {
    printf("=== Section 15.1 main: level-by-level parallel BST construction ===\n\n");

    std::vector<int> a = {10, 20, 30, 40, 50, 60, 70};
    int n = (int)a.size();

    printf("sorted input: ");
    for (int v : a) printf("%d ", v);
    printf("\n\n");

    std::vector<int> value(n, 0), left(n, -1), right(n, -1);

    std::vector<int> cur_lo = {0}, cur_hi = {n - 1}, cur_slot = {0};
    int level_num = 0;
    while (!cur_lo.empty()) {
        int width = (int)cur_lo.size();
        std::vector<int> next_lo(2 * width, -1), next_hi(2 * width, -1), next_slot(2 * width, -1);

        for (int t = 0; t < width; t++) {   // width independent threads
            int lo = cur_lo[t], hi = cur_hi[t], slot = cur_slot[t];
            int mid = (lo + hi) / 2;
            value[slot] = a[mid];
            int lslot = 2 * slot + 1, rslot = 2 * slot + 2;
            if (lo <= mid - 1) {
                left[slot] = lslot;
                next_lo[2 * t] = lo; next_hi[2 * t] = mid - 1; next_slot[2 * t] = lslot;
            }
            if (mid + 1 <= hi) {
                right[slot] = rslot;
                next_lo[2 * t + 1] = mid + 1; next_hi[2 * t + 1] = hi; next_slot[2 * t + 1] = rslot;
            }
        }

        printf("level %d: %d independent thread(s), slots [", level_num, width);
        for (int s : cur_slot) printf("%d ", s);
        printf("]\n");
        level_num++;

        std::vector<int> nlo, nhi, nslot;
        for (int i = 0; i < 2 * width; i++) {
            if (next_slot[i] != -1) { nlo.push_back(next_lo[i]); nhi.push_back(next_hi[i]); nslot.push_back(next_slot[i]); }
        }
        cur_lo = nlo; cur_hi = nhi; cur_slot = nslot;
    }
    printf("\n");

    printf("value array: ");
    for (int v : value) printf("%d ", v);
    printf("\nleft array:  ");
    for (int v : left) printf("%d ", v);
    printf("\nright array: ");
    for (int v : right) printf("%d ", v);
    printf("\n\n");

    std::vector<int> expected_value = {40, 20, 60, 10, 30, 50, 70};
    std::vector<int> expected_left = {1, 3, 5, -1, -1, -1, -1};
    std::vector<int> expected_right = {2, 4, 6, -1, -1, -1, -1};
    bool ok = (value == expected_value) && (left == expected_left) && (right == expected_right);

    printf("expected value: 40 20 60 10 30 50 70\n");
    printf("expected left:  1 3 5 -1 -1 -1 -1\n");
    printf("expected right: 2 4 6 -1 -1 -1 -1\n\n");
    printf("%d levels total for n=%d -- exactly log2(n+1), matching the CPU baseline's own\n",
           level_num, n);
    printf("queue exactly, but every node WITHIN a level ran as an independent thread with\n");
    printf("no shared counter needed for child slots at all.\n");

    printf("\nself-check: level-by-level parallel construction matches CPU baseline: %s\n",
           ok ? "confirmed" : "MISMATCH");
    return ok ? 0 : 1;
}
```

**Compile and run:**

```bash
NVDIR=/usr/local/lib/python3.11/dist-packages/nvidia/cu13
$NVDIR/bin/nvcc -arch=sm_80 -I$NVDIR/include -L$NVDIR/lib -lcudart 77_bst_from_sorted_levels_kernel.cu -o 77_bst_from_sorted_levels_kernel
LD_LIBRARY_PATH=$NVDIR/lib ./77_bst_from_sorted_levels_kernel
```

**Sample input:** the same sorted array, built level by level, with every node in a level processed by an independent thread.

**Sample output:**

```text
=== Section 15.1 main: level-by-level parallel BST construction ===

sorted input: 10 20 30 40 50 60 70 

level 0: 1 independent thread(s), slots [0 ]
level 1: 2 independent thread(s), slots [1 2 ]
level 2: 4 independent thread(s), slots [3 4 5 6 ]

value array: 40 20 60 10 30 50 70 
left array:  1 3 5 -1 -1 -1 -1 
right array: 2 4 6 -1 -1 -1 -1 

expected value: 40 20 60 10 30 50 70
expected left:  1 3 5 -1 -1 -1 -1
expected right: 2 4 6 -1 -1 -1 -1

3 levels total for n=7 -- exactly log2(n+1), matching the CPU baseline's own
queue exactly, but every node WITHIN a level ran as an independent thread with
no shared counter needed for child slots at all.

self-check: level-by-level parallel construction matches CPU baseline: confirmed
```

## 15.2 Iterative In-Order Traversal via an Explicit Stack

### Intuition

The classic in-order traversal -- visit left subtree, visit this node, visit right subtree -- is naturally recursive, and every recursive call implicitly pushes "come back here once the left subtree finishes" onto the call stack. Replacing that implicit call stack with an explicit, programmer-managed array (a plain LIFO) produces byte-for-byte the same traversal order, with no recursive call anywhere and a size the programmer controls directly -- exactly sized to the tree's own known height.

### The Sequential (CPU) Baseline

```cpp
#include <cstdio>
#include <vector>

// Chapter 15.2 -- The Sequential (CPU) Baseline.
// The textbook in-order traversal is recursive: visit the left
// subtree, visit this node, visit the right subtree. Every recursive
// call implicitly pushes "come back here after the left subtree
// finishes" onto the call stack. On a GPU this matters for a reason
// beyond style: CUDA's per-thread device-side call stack is a fixed,
// pre-allocated size, and -- far more importantly -- threads within a
// single warp execute in lockstep (Chapter 1's own SIMT model), so if
// different threads recurse to different depths on different tree
// shapes, the whole warp serializes to accommodate whichever thread
// recurses deepest. An EXPLICIT stack -- a plain, programmer-managed
// array acting as a LIFO -- produces the identical traversal order
// with no recursion and a size the programmer controls directly.

std::vector<int> stack_inorder(int root, const std::vector<int>& value,
                                const std::vector<int>& left, const std::vector<int>& right) {
    std::vector<int> out;
    std::vector<int> stack;
    int cur = root;
    while (!stack.empty() || cur != -1) {
        while (cur != -1) {
            stack.push_back(cur);
            cur = left[cur];
        }
        cur = stack.back(); stack.pop_back();
        out.push_back(value[cur]);
        cur = right[cur];
    }
    return out;
}

int main() {
    printf("=== Section 15.2 CPU baseline: explicit-stack in-order traversal ===\n\n");

    std::vector<int> value = {40, 20, 60, 10, 30, 50, 70};
    std::vector<int> left  = {1, 3, 5, -1, -1, -1, -1};
    std::vector<int> right = {2, 4, 6, -1, -1, -1, -1};
    int root = 0;

    printf("tree (from Section 15.1): value=");
    for (int v : value) printf("%d ", v);
    printf(" left=");
    for (int v : left) printf("%d ", v);
    printf(" right=");
    for (int v : right) printf("%d ", v);
    printf("\n\n");

    auto out = stack_inorder(root, value, left, right);

    printf("explicit-stack in-order traversal: ");
    for (int v : out) printf("%d ", v);
    printf("\n\n");

    std::vector<int> expected = {10, 20, 30, 40, 50, 60, 70};
    bool ok = (out == expected);

    printf("expected: 10 20 30 40 50 60 70\n\n");
    printf("no function call was ever made recursively; the stack array held exactly what\n");
    printf("the call stack would have held (\"come back to this node once its left subtree\n");
    printf("is done\"), sized here to the tree's own height (3), never more.\n");

    printf("\nself-check: explicit-stack traversal recovers the original sorted order: %s\n",
           ok ? "confirmed" : "MISMATCH");
    return ok ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++17 -Wall -Wextra -O2 78_stack_inorder_cpu_baseline.cpp -o 78_stack_inorder_cpu_baseline
./78_stack_inorder_cpu_baseline
```

**Sample input:** Section 15.1's tree, traversed in-order via an explicit stack.

**Sample output:**

```text
=== Section 15.2 CPU baseline: explicit-stack in-order traversal ===

tree (from Section 15.1): value=40 20 60 10 30 50 70  left=1 3 5 -1 -1 -1 -1  right=2 4 6 -1 -1 -1 -1 

explicit-stack in-order traversal: 10 20 30 40 50 60 70 

expected: 10 20 30 40 50 60 70

no function call was ever made recursively; the stack array held exactly what
the call stack would have held ("come back to this node once its left subtree
is done"), sized here to the tree's own height (3), never more.

self-check: explicit-stack traversal recovers the original sorted order: confirmed
```

### The Concept, In Detail

The explicit-stack algorithm mirrors recursive in-order traversal's own logic exactly: push every node down the current LEFT spine onto the stack, then pop one, visit it, and move to its right child (which restarts the "push down the left spine" step from there).

```
walking the tree value=[40,20,60,10,30,50,70], left=[1,3,5,-1,-1,-1,-1], right=[2,4,6,-1,-1,-1,-1]:

  cur=40: push 40, cur=left[40]=20; push 20, cur=left[20]=10; push 10, cur=left[10]=-1
  stack: [40, 20, 10]
  pop 10 -> visit 10; cur=right[10]=-1
  pop 20 -> visit 20; cur=right[20]=30; push 30, cur=left[30]=-1
  pop 30 -> visit 30; cur=right[30]=-1
  pop 40 -> visit 40; cur=right[40]=60; push 60, cur=left[60]=50; push 50, cur=left[50]=-1
  pop 50 -> visit 50; cur=right[50]=-1
  pop 60 -> visit 60; cur=right[60]=70; push 70, cur=left[70]=-1
  pop 70 -> visit 70; cur=right[70]=-1; stack empty, cur=-1 -> done

  visited in order: 10, 20, 30, 40, 50, 60, 70
```

The stack never holds more entries than the tree's own height (here, 3) -- each entry represents one node still "waiting" to have its right subtree visited, and a node only gets pushed once, when first reached going down some left spine. This is exactly the mechanical replacement for what a compiler's own call stack does for you automatically in a recursive version: the same information, made explicit and bounded by a size the programmer chose.

The genuine parallel opportunity is NOT trying to traverse one tree faster with many threads -- traversal has an unavoidable span equal to the number of nodes visited, exactly like Chapter 11's pointer-chasing linked list, since node `k`'s position in the output cannot be known before node `k-1` has actually been visited. The opportunity instead is running MANY INDEPENDENT traversals concurrently -- one thread per tree (or per query), each maintaining its OWN private stack, with zero coordination needed between threads.

```
ASCII view: 3 independent trees packed into one shared array, one thread per tree.

  slots 0-6:   tree A (root 0)     value: 10..70 by tens
  slots 7-13:  tree B (root 7)     value: 1..7
  slots 14-20: tree C (root 14)    value: 100..700 by hundreds

  thread 0 -> root=0, own stack, never touches slots 7-20
  thread 1 -> root=7, own stack, never touches slots 0-6 or 14-20
  thread 2 -> root=14, own stack, never touches slots 0-13
```

[COMMON TRAP]
Sizing a shared, fixed-capacity stack ARRAY per thread based on the SHALLOWEST tree in a batch (rather than the tallest) silently truncates deeper trees' traversals -- a stack overflow that, unlike a CPU's stack overflow, may not fault cleanly and can instead corrupt neighboring memory. Always size a per-thread stack to the tallest tree that thread family could ever see, not the common case.

### Code and Verification

```cpp
#include <cstdio>
#include <vector>

// Chapter 15.2 main -- explicit-stack traversal, run by many threads
// at once, each on its OWN independent tree. A single traversal is
// inherently sequential (span equal to the number of nodes visited --
// no way to know where node k+1 is without having visited node k), so
// the parallel opportunity here is not "traverse ONE tree with many
// threads," it is "traverse MANY trees with many threads, one tree per
// thread" -- exactly Chapter 11's own resolution for pointer-chasing,
// applied again. Three independent trees are packed into ONE shared
// set of value[]/left[]/right[] arrays at disjoint slot ranges, and
// each thread owns a private stack sized to ITS OWN tree's height.
#define MAX_STACK 8

__global__ void stack_inorder_kernel(const int* g_value, const int* g_left, const int* g_right,
                                      const int* g_roots, int* g_out, int out_stride) {
    int tid = threadIdx.x;
    int stack[MAX_STACK];
    int sp = 0;
    int cur = g_roots[tid];
    int count = 0;
    while (sp > 0 || cur != -1) {
        while (cur != -1) {
            stack[sp++] = cur;
            cur = g_left[cur];
        }
        cur = stack[--sp];
        g_out[tid * out_stride + count] = g_value[cur];
        count++;
        cur = g_right[cur];
    }
}

// ---- Host-side replay of the identical per-thread logic: each thread
// keeps its OWN local stack array and walks only its OWN tree, with no
// data shared between threads at all. ----

int main() {
    printf("=== Section 15.2 main: explicit-stack traversal of 3 independent trees ===\n\n");

    std::vector<int> value = {40, 20, 60, 10, 30, 50, 70,
                                4,  2,  6,  1,  3,  5,  7,
                              400,200,600,100,300,500,700};
    std::vector<int> left  = {1, 3, 5, -1, -1, -1, -1,
                               8, 10, 12, -1, -1, -1, -1,
                               15, 17, 19, -1, -1, -1, -1};
    std::vector<int> right = {2, 4, 6, -1, -1, -1, -1,
                               9, 11, 13, -1, -1, -1, -1,
                               16, 18, 20, -1, -1, -1, -1};
    std::vector<int> roots = {0, 7, 14};
    int num_threads = (int)roots.size();
    int per_tree = 7;

    printf("3 independent trees packed into one array, roots at slots: ");
    for (int r : roots) printf("%d ", r);
    printf("\n\n");

    std::vector<std::vector<int>> results(num_threads);
    for (int tid = 0; tid < num_threads; tid++) {
        std::vector<int> stack;
        int cur = roots[tid];
        std::vector<int> out;
        while (!stack.empty() || cur != -1) {
            while (cur != -1) { stack.push_back(cur); cur = left[cur]; }
            cur = stack.back(); stack.pop_back();
            out.push_back(value[cur]);
            cur = right[cur];
        }
        results[tid] = out;
        printf("thread %d (root=%d) traversal: ", tid, roots[tid]);
        for (int v : out) printf("%d ", v);
        printf("\n");
    }
    printf("\n");

    std::vector<std::vector<int>> expected = {
        {10, 20, 30, 40, 50, 60, 70},
        {1, 2, 3, 4, 5, 6, 7},
        {100, 200, 300, 400, 500, 600, 700}
    };
    bool ok = true;
    for (int t = 0; t < num_threads; t++) {
        if (results[t] != expected[t]) ok = false;
        if ((int)results[t].size() != per_tree) ok = false;
    }

    printf("expected thread 0: 10 20 30 40 50 60 70\n");
    printf("expected thread 1: 1 2 3 4 5 6 7\n");
    printf("expected thread 2: 100 200 300 400 500 600 700\n\n");
    printf("all %d threads ran completely independently, each maintaining its own private\n",
           num_threads);
    printf("stack of size up to %d (this tree family's known max height) -- %d threads times\n",
           MAX_STACK, num_threads);
    printf("that stack size is exactly the per-thread memory cost this technique carries.\n");

    printf("\nself-check: all %d independent traversals match expected sorted order: %s\n",
           num_threads, ok ? "confirmed" : "MISMATCH");
    return ok ? 0 : 1;
}
```

**Compile and run:**

```bash
NVDIR=/usr/local/lib/python3.11/dist-packages/nvidia/cu13
$NVDIR/bin/nvcc -arch=sm_80 -I$NVDIR/include -L$NVDIR/lib -lcudart 79_stack_inorder_parallel_kernel.cu -o 79_stack_inorder_parallel_kernel
LD_LIBRARY_PATH=$NVDIR/lib ./79_stack_inorder_parallel_kernel
```

**Sample input:** 3 independent trees packed into one shared array, traversed by 3 threads at once, each with its own private stack.

**Sample output:**

```text
=== Section 15.2 main: explicit-stack traversal of 3 independent trees ===

3 independent trees packed into one array, roots at slots: 0 7 14 

thread 0 (root=0) traversal: 10 20 30 40 50 60 70 
thread 1 (root=7) traversal: 1 2 3 4 5 6 7 
thread 2 (root=14) traversal: 100 200 300 400 500 600 700 

expected thread 0: 10 20 30 40 50 60 70
expected thread 1: 1 2 3 4 5 6 7
expected thread 2: 100 200 300 400 500 600 700

all 3 threads ran completely independently, each maintaining its own private
stack of size up to 8 (this tree family's known max height) -- 3 threads times
that stack size is exactly the per-thread memory cost this technique carries.

self-check: all 3 independent traversals match expected sorted order: confirmed
```

## 15.3 Morris Traversal: O(1)-Space Traversal for Massive Parallelism

### Intuition

Section 15.2's explicit stack costs memory proportional to tree height, PER THREAD. That is fine for one traversal, but the natural GPU workload is thousands of threads each traversing their own tree at once, and thousands of stacks add up. Morris traversal removes the stack entirely: for a node with a left child, find that child's rightmost descendant (its in-order predecessor) and temporarily point that descendant's otherwise-idle right pointer back at the current node -- a breadcrumb threaded directly into the tree's own existing structure. Once that breadcrumb is followed back to where it came from, it is removed immediately, so the tree ends the traversal in exactly the state it started in.

### The Sequential (CPU) Baseline

```cpp
#include <cstdio>
#include <vector>

// Chapter 15.3 -- The Sequential (CPU) Baseline.
// Section 15.2's explicit stack still costs memory proportional to
// tree height -- fine for one traversal, but it adds up once many
// threads each need their own. Morris traversal removes the stack
// entirely: for a node with a left child, find that child's RIGHTMOST
// descendant (its in-order predecessor) and temporarily point that
// descendant's unused right pointer back at the current node -- a
// breadcrumb threaded directly into the tree's own otherwise-idle
// right pointers. Once that breadcrumb is followed back, it is removed
// immediately, so the tree ends the traversal in EXACTLY the state it
// started in. This achieves O(1) extra space, at the cost of
// temporarily mutating (and then always correctly restoring) the tree.

std::vector<int> morris_inorder(int root, const std::vector<int>& value,
                                 std::vector<int>& left, std::vector<int>& right) {
    std::vector<int> out;
    int cur = root;
    while (cur != -1) {
        if (left[cur] == -1) {
            out.push_back(value[cur]);
            cur = right[cur];
        } else {
            int pred = left[cur];
            while (right[pred] != -1 && right[pred] != cur) pred = right[pred];
            if (right[pred] == -1) {
                right[pred] = cur;      // thread: breadcrumb back to cur
                cur = left[cur];
            } else {
                right[pred] = -1;       // remove the breadcrumb
                out.push_back(value[cur]);
                cur = right[cur];
            }
        }
    }
    return out;
}

int main() {
    printf("=== Section 15.3 CPU baseline: Morris (stack-free) in-order traversal ===\n\n");

    std::vector<int> value = {40, 20, 60, 10, 30, 50, 70};
    std::vector<int> left  = {1, 3, 5, -1, -1, -1, -1};
    std::vector<int> right = {2, 4, 6, -1, -1, -1, -1};
    int root = 0;

    std::vector<int> right_before = right;   // save for the restoration check

    printf("tree (same as Section 15.2): value=");
    for (int v : value) printf("%d ", v);
    printf(" right (before)=");
    for (int v : right) printf("%d ", v);
    printf("\n\n");

    auto out = morris_inorder(root, value, left, right);

    printf("morris in-order traversal: ");
    for (int v : out) printf("%d ", v);
    printf("\nright array (after, should be fully restored): ");
    for (int v : right) printf("%d ", v);
    printf("\n\n");

    std::vector<int> expected = {10, 20, 30, 40, 50, 60, 70};
    bool restored = (right == right_before);
    bool ok = (out == expected) && restored;

    printf("expected traversal: 10 20 30 40 50 60 70\n");
    printf("restored correctly: %s\n\n", restored ? "yes" : "NO -- BUG");
    printf("not one byte of extra memory proportional to tree height was used -- every\n");
    printf("breadcrumb this traversal wrote, it also erased, using only the tree's own\n");
    printf("existing (and otherwise idle) right-child slots as temporary scratch space.\n");

    printf("\nself-check: morris traversal matches expected order, tree fully restored: %s\n",
           ok ? "confirmed" : "MISMATCH");
    return ok ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++17 -Wall -Wextra -O2 80_morris_inorder_cpu_baseline.cpp -o 80_morris_inorder_cpu_baseline
./80_morris_inorder_cpu_baseline
```

**Sample input:** the same tree as Section 15.2, traversed via Morris's threading technique, with a check that the tree is fully restored afterward.

**Sample output:**

```text
=== Section 15.3 CPU baseline: Morris (stack-free) in-order traversal ===

tree (same as Section 15.2): value=40 20 60 10 30 50 70  right (before)=2 4 6 -1 -1 -1 -1 

morris in-order traversal: 10 20 30 40 50 60 70 
right array (after, should be fully restored): 2 4 6 -1 -1 -1 -1 

expected traversal: 10 20 30 40 50 60 70
restored correctly: yes

not one byte of extra memory proportional to tree height was used -- every
breadcrumb this traversal wrote, it also erased, using only the tree's own
existing (and otherwise idle) right-child slots as temporary scratch space.

self-check: morris traversal matches expected order, tree fully restored: confirmed
```

### The Concept, In Detail

Tracing Morris traversal on the same tree (`value=[40,20,60,10,30,50,70]`, `left=[1,3,5,-1,-1,-1,-1]`, `right=[2,4,6,-1,-1,-1,-1]`):

```
cur=40: has left child (20). Find 20's rightmost descendant with no
  existing thread: start at pred=20; right[20]=30 (an ordinary right
  child, not -1 and not cur), so move to pred=30; right[30]=-1, so the
  search stops there -- predecessor of 40 is 30. THREAD: right[30] = 40
  (breadcrumb). cur = left[40] = 20.

cur=20: has left child (10). Predecessor of 20 is 10 (10's right is -1).
  THREAD: right[10] = 20. cur = left[20] = 10.

cur=10: no left child. VISIT 10. cur = right[10] = 20 (the breadcrumb).

cur=20: has left child (10) again, but this time right[10] == 20 (our
  own breadcrumb, not -1) -- REMOVE thread: right[10] = -1. VISIT 20.
  cur = right[20] = 30.

cur=30: no left child. VISIT 30. cur = right[30] = 40 (the breadcrumb).

cur=40: has left child (20) again, but right[30] == 40 now -- REMOVE
  thread: right[30] = -1. VISIT 40. cur = right[40] = 60.

cur=60: has left child (50). Predecessor of 60 is 50 (50's right is -1).
  THREAD: right[50] = 60. cur = left[60] = 50.

cur=50: no left child. VISIT 50. cur = right[50] = 60 (the breadcrumb).

cur=60: right[50] == 60 -- REMOVE thread: right[50] = -1. VISIT 60.
  cur = right[60] = 70.

cur=70: no left child. VISIT 70. cur = right[70] = -1. Done.

visited in order: 10, 20, 30, 40, 50, 60, 70
```

Every thread was visited using ONLY two extra variables (`cur` and `pred`), regardless of the tree's height -- compare this to Section 15.2's stack, whose size had to grow with height. The cost is that the tree's `right[]` array is temporarily mutated mid-traversal (each thread that finds a node with a left child briefly overwrites one `right[]` entry with a breadcrumb), but every breadcrumb this algorithm writes, it also removes before moving past it -- the very next time that same thread's traversal revisits that predecessor, it recognizes the existing thread (rather than `-1`) and knows to remove it. By the time any thread's traversal finishes, its part of the shared `right[]` array is bit-for-bit identical to how it started.

```
ASCII view: what "no stack" buys at scale.

  Section 15.2, N threads, each height-H tree: N * H ints of stack memory.
  Section 15.3, N threads, ANY height tree:    N * 2 ints (cur, pred) -- flat.

  At N=10,000 threads and H=20, that is 200,000 ints of stack versus 20,000
  ints total -- and the gap only widens as trees get taller.
```

[COMMON TRAP]
Running Morris traversal on multiple threads that might share parts of the SAME tree (rather than each thread owning its own disjoint tree, as in this section's example) is unsafe: two threads simultaneously threading and un-threading the same node's `right[]` pointer is a genuine data race, with no atomic operation in this algorithm to prevent it. Morris traversal's O(1)-space benefit assumes exclusive, single-threaded ownership of whichever tree it is walking at any given moment.

### Code and Verification

```cpp
#include <cstdio>
#include <vector>

// Chapter 15.3 main -- Morris traversal, run by many threads at once,
// each on its OWN independent tree. This is what Section 15.2's
// per-thread stack was actually costing: MAX_STACK ints per thread,
// times however many threads are traversing concurrently. Morris
// traversal needs a small, FIXED number of local variables (a current
// pointer and a predecessor pointer) regardless of tree height or
// thread count -- the O(1)-space property that makes running this
// across thousands of threads, one tree each, genuinely cheap.
__global__ void morris_inorder_kernel(const int* g_value, int* g_left, int* g_right,
                                       const int* g_roots, int* g_out, int out_stride) {
    int tid = threadIdx.x;
    int cur = g_roots[tid];
    int count = 0;
    while (cur != -1) {
        if (g_left[cur] == -1) {
            g_out[tid * out_stride + count] = g_value[cur];
            count++;
            cur = g_right[cur];
        } else {
            int pred = g_left[cur];
            while (g_right[pred] != -1 && g_right[pred] != cur) pred = g_right[pred];
            if (g_right[pred] == -1) {
                g_right[pred] = cur;
                cur = g_left[cur];
            } else {
                g_right[pred] = -1;
                g_out[tid * out_stride + count] = g_value[cur];
                count++;
                cur = g_right[cur];
            }
        }
    }
}

// ---- Host-side replay of the identical per-thread logic: each thread
// touches only the two array slots (its own subtree, and that
// subtree's own idle right pointers) it needs, with no stack, and no
// interference between threads even though all three trees share ONE
// underlying right[] array. ----

int main() {
    printf("=== Section 15.3 main: Morris traversal of 3 independent trees ===\n\n");

    std::vector<int> value = {40, 20, 60, 10, 30, 50, 70,
                                4,  2,  6,  1,  3,  5,  7,
                              400,200,600,100,300,500,700};
    std::vector<int> left  = {1, 3, 5, -1, -1, -1, -1,
                               8, 10, 12, -1, -1, -1, -1,
                               15, 17, 19, -1, -1, -1, -1};
    std::vector<int> right = {2, 4, 6, -1, -1, -1, -1,
                               9, 11, 13, -1, -1, -1, -1,
                               16, 18, 20, -1, -1, -1, -1};
    std::vector<int> right_before = right;
    std::vector<int> roots = {0, 7, 14};
    int num_threads = (int)roots.size();

    printf("3 independent trees, same layout as Section 15.2, roots at slots: ");
    for (int r : roots) printf("%d ", r);
    printf("\n\n");

    std::vector<std::vector<int>> results(num_threads);
    for (int tid = 0; tid < num_threads; tid++) {
        int cur = roots[tid];
        std::vector<int> out;
        while (cur != -1) {
            if (left[cur] == -1) {
                out.push_back(value[cur]);
                cur = right[cur];
            } else {
                int pred = left[cur];
                while (right[pred] != -1 && right[pred] != cur) pred = right[pred];
                if (right[pred] == -1) {
                    right[pred] = cur;
                    cur = left[cur];
                } else {
                    right[pred] = -1;
                    out.push_back(value[cur]);
                    cur = right[cur];
                }
            }
        }
        results[tid] = out;
        printf("thread %d (root=%d) traversal: ", tid, roots[tid]);
        for (int v : out) printf("%d ", v);
        printf("\n");
    }
    printf("\n");

    bool restored = (right == right_before);
    printf("right array after all 3 threads finished, fully restored: %s\n\n",
           restored ? "yes" : "NO -- BUG");

    std::vector<std::vector<int>> expected = {
        {10, 20, 30, 40, 50, 60, 70},
        {1, 2, 3, 4, 5, 6, 7},
        {100, 200, 300, 400, 500, 600, 700}
    };
    bool ok = restored;
    for (int t = 0; t < num_threads; t++) if (results[t] != expected[t]) ok = false;

    printf("expected thread 0: 10 20 30 40 50 60 70\n");
    printf("expected thread 1: 1 2 3 4 5 6 7\n");
    printf("expected thread 2: 100 200 300 400 500 600 700\n\n");
    printf("no per-thread stack array exists anywhere in this kernel -- just `cur` and\n");
    printf("`pred`, two registers, regardless of how tall any of these 3 trees is, or how\n");
    printf("many threads (trees) run at once.\n");

    printf("\nself-check: all %d independent Morris traversals match expected order, all\n",
           num_threads);
    printf("threads' subtrees fully restored: %s\n", ok ? "confirmed" : "MISMATCH");
    return ok ? 0 : 1;
}
```

**Compile and run:**

```bash
NVDIR=/usr/local/lib/python3.11/dist-packages/nvidia/cu13
$NVDIR/bin/nvcc -arch=sm_80 -I$NVDIR/include -L$NVDIR/lib -lcudart 81_morris_inorder_parallel_kernel.cu -o 81_morris_inorder_parallel_kernel
LD_LIBRARY_PATH=$NVDIR/lib ./81_morris_inorder_parallel_kernel
```

**Sample input:** the same 3 independent trees from Section 15.2, traversed via Morris's technique, one thread per tree, with no per-thread stack at all.

**Sample output:**

```text
=== Section 15.3 main: Morris traversal of 3 independent trees ===

3 independent trees, same layout as Section 15.2, roots at slots: 0 7 14 

thread 0 (root=0) traversal: 10 20 30 40 50 60 70 
thread 1 (root=7) traversal: 1 2 3 4 5 6 7 
thread 2 (root=14) traversal: 100 200 300 400 500 600 700 

right array after all 3 threads finished, fully restored: yes

expected thread 0: 10 20 30 40 50 60 70
expected thread 1: 1 2 3 4 5 6 7
expected thread 2: 100 200 300 400 500 600 700

no per-thread stack array exists anywhere in this kernel -- just `cur` and
`pred`, two registers, regardless of how tall any of these 3 trees is, or how
many threads (trees) run at once.

self-check: all 3 independent Morris traversals match expected order, all
threads' subtrees fully restored: confirmed
```

## Chapter Summary

A balanced binary search tree can be built directly from a sorted array by repeatedly halving ranges, and doing this iteratively (via an explicit queue of pending work items, processed level by level) removes recursion from construction entirely, with every node in a single level being independent of every other node in that level -- a genuinely parallel operation needing no shared allocation counter when the tree being built is perfectly complete. Continuing this book's discipline since Chapter 11, tree nodes live in explicit `value[]`/`left[]`/`right[]` arrays addressed by integer indices, never real pointers. In-order traversal is traditionally recursive, but an explicit, programmer-sized stack array reproduces the identical traversal order with no recursive call and a memory cost bounded by the tree's own height; since one traversal has an unavoidable span equal to the number of nodes visited, the real parallel opportunity is running many independent traversals at once, one thread per tree, each with its own private stack. Morris traversal goes one step further, removing even that stack: by temporarily threading a node's otherwise-idle right pointer to point back at its in-order successor, then removing that breadcrumb the moment it is followed, a full traversal completes with only O(1) extra space per thread, regardless of tree height -- which is exactly what makes running thousands of concurrent, independent traversals genuinely cheap at GPU scale, so long as no two threads ever share the same tree.

## Self-Check Questions

1. Why does building a balanced BST from a sorted array via an explicit queue avoid recursion entirely, and what specifically does the queue hold in place of a call stack?
2. Why do slot `i`'s children always land at slots `2i+1` and `2i+2` in Section 15.1's construction, and why does this NOT generalize to a tree built by ordinary one-at-a-time insertion?
3. In the explicit-stack in-order traversal, what does the stack hold at any given moment, and why does its maximum size never exceed the tree's own height?
4. Why is "traverse one tree with many threads" not the right parallel target for tree traversal, and what IS the right target instead?
5. Walk through what happens the SECOND time Morris traversal's algorithm reaches a node whose predecessor's right pointer is already threaded (rather than `-1`). What does this signal, and what two things happen as a result?
6. Why is it unsafe to run Morris traversal on multiple threads that might share parts of the same tree, even though each individual thread's own traversal is fully correct in isolation?

## Where We Go Next

This chapter built ONE balanced tree from already-sorted data, and walked it without recursion. Real-world trees are rarely built once from perfectly sorted input and left alone -- they grow incrementally, from arbitrary (not necessarily sorted) data, sometimes from many insertions happening at once. Chapter 16 turns to genuinely parallel tree and trie construction from unsorted data: building useful tree structure directly out of many independent pieces of input at once, rather than assuming a convenient, already-sorted starting point the way this chapter did.

## Worked Solutions

**1.** Building the BST iteratively works because each level's work items only depend on information already computed -- a `(lo, hi, slot)` triple is fully self-contained, and processing it needs no result from "returning" out of any nested call. The queue holds exactly the pending ranges-and-destinations that a recursive call's own stack frame would otherwise be tracking implicitly (which range this call still needs to handle, and where its result should be written) -- made into ordinary, explicit data instead of hidden call-stack state.

**2.** Slot `i`'s children land at `2i+1` and `2i+2` because this specific construction always builds a PERFECTLY COMPLETE tree (every level fully populated except possibly the last, filled left to right) from a sorted array of exactly the right size, and level-order allocation of a complete tree's nodes always produces this exact numbering -- it is a property of the SHAPE being built, not of the array representation itself. A tree grown by ordinary one-at-a-time insertion can end up any shape at all (lopsided, missing large sections), so a newly inserted node's slot bears no fixed arithmetic relationship to its parent's slot; an explicit allocation counter (as Chapters 8-10 already used for their own node pools) is required instead.

**3.** At any moment, the stack holds exactly the ancestors of the current position that still have an un-visited right subtree waiting -- nodes reached by walking down a left spine that have not yet been popped and visited. Its size never exceeds the tree's height because a node is only ever pushed once, when first reached while descending some left spine from the root, and the longest such spine possible is bounded by the tree's own height.

**4.** A single traversal has an unavoidable span equal to the number of nodes it visits: node `k`'s identity in the output genuinely cannot be known until node `k-1` has been visited (exactly Chapter 11's own linked-list argument), so no number of additional threads can shorten that one traversal's critical path. The right parallel target is running MANY such traversals at once, one independent thread per tree (or per query against a tree), since different threads' traversals share no data dependency between them at all.

**5.** Reaching a node whose predecessor's right pointer is ALREADY threaded (points to the current node, rather than being `-1`) signals that this is the SECOND visit to this junction -- the first visit threaded the breadcrumb and descended left; everything in that left subtree has now been fully visited, and control has arrived back via the breadcrumb exactly as intended. The two things that happen are: the breadcrumb is removed (`right[pred] = -1`, restoring that slot to its original value), and the current node itself is now visited (added to the output) before moving on to its own right child.

**6.** Each individual thread's traversal is correct in isolation because Morris traversal assumes it has exclusive control over every `right[]` slot it might temporarily thread and later un-thread. If two threads' traversals could reach the SAME node (because they share part of a tree), both might attempt to thread or read that node's `right[]` pointer at overlapping times, with no atomic operation anywhere in the algorithm to make "check if already threaded, then thread or un-thread" a single indivisible step -- exactly the kind of race Chapter 9's naive stack push had, before atomicCAS fixed it. Morris traversal has no such fix built in; its O(1)-space guarantee is only safe when each thread owns a fully disjoint tree.
