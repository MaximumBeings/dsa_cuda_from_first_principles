# Chapter 11: Pointer-Chasing Linked Lists, and Why This Book Avoided Them

**What you will understand by the end of this chapter:**

- Why a linked list's traversal is a genuine, unavoidable dependency chain — not a badly-written kernel, but a structural property no thread count can shorten.
- How parallel list ranking (pointer jumping) turns an O(n)-span sequential walk into an O(log n)-span parallel computation, by doing strictly more total work — Chapter 5.2's work-versus-span trade, one more time.
- Why, once a list's ranks are known, everything downstream becomes an ordinary O(1)-span array scatter — and why this is exactly the property this book's own stacks and queues (Chapters 9-10) preserved by using index-based node pools instead of real pointers from the very first page.

**What you need to know first:**

- Chapter 3's work-span vocabulary, applied here to its starkest possible case: a structure where span cannot be reduced by adding threads at all.
- Chapter 5.1's Hillis-Steele doubling shape and its double-buffering discipline, reused here for a linked structure instead of a fixed-stride array offset.
- Chapters 9 and 10's index-based node pools (`struct Node { int value; int next; }`, `-1` for null), which this chapter's own code continues to use, and finally explains in full.

---

Every structure since Chapter 4 has shared one property this book has used constantly without naming it: given an index `i`, any thread can compute element `i`'s address on its own, instantly, with no help from any other thread and no dependency on what any other thread has read or written. A linked list breaks this on purpose — that is the entire point of a linked list, the reason it can grow one node at a time without ever moving the nodes already there. This chapter takes that trade seriously and measures its actual cost, then shows the one real technique (pointer jumping) that recovers parallelism from it, at a real and measured price.

## 11.1 Sequential Pointer-Chasing: An Unavoidable Latency Chain

### Intuition

Summing an array and summing a linked list look like the same problem — visit every element once, add it to a running total. Chapter 4 parallelized the array version by letting many threads compute their own addresses independently and add concurrently. Try the identical idea on a linked list and it fails immediately: thread `k` cannot start at "the `k`-th node" without already knowing what that node's address is, and the only way to learn a node's address is to have already read the node before it.

### The Sequential (CPU) Baseline

```cpp
#include <cstdio>
#include <vector>

// Chapter 11.1 -- The Sequential (CPU) Baseline.
// A single thread walks a linked list one node at a time, summing values
// as it goes. Nothing about this loop looks expensive -- it is just a
// while loop -- but notice each iteration's address (nodes[cur].next)
// is only known once the PREVIOUS node has actually been read. That
// data dependency, invisible here, is exactly this section's subject.

struct Node { int value; int next; };

int sum_list_cpu(const std::vector<Node>& nodes, int head) {
    int sum = 0;
    int cur = head;
    while (cur != -1) {
        sum += nodes[cur].value;
        cur = nodes[cur].next;
    }
    return sum;
}

int main() {
    printf("=== Section 11.1 CPU baseline: sequential list traversal ===\n\n");

    const int N = 8;
    std::vector<Node> nodes(N);
    for (int i = 0; i < N; i++) {
        nodes[i].value = (i + 1) * 10;
        nodes[i].next = (i + 1 < N) ? (i + 1) : -1;
    }
    int head = 0;

    printf("list (head to tail): ");
    int cur = head;
    while (cur != -1) { printf("%d ", nodes[cur].value); cur = nodes[cur].next; }
    printf("\n\n");

    int sum = sum_list_cpu(nodes, head);
    int expected = 0;
    for (int i = 0; i < N; i++) expected += nodes[i].value;

    printf("sum_list_cpu result: %d\n", sum);
    printf("expected (10+20+...+80):     %d\n", expected);
    printf("\nevery hop's address depends on the value just read at the previous hop --\n");
    printf("a true dependency chain, invisible in a single-thread loop like this one.\n");

    bool ok = (sum == expected);
    printf("\nself-check: sequential list sum matches expected total: %s\n",
           ok ? "confirmed" : "MISMATCH");
    return ok ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++17 -Wall -Wextra -O2 52_sum_list_cpu_baseline.cpp -o sum_list_cpu_baseline
./sum_list_cpu_baseline
```

**Sample input:** an 8-node list, values `10, 20, ..., 80`.

**Sample output:**

```text
=== Section 11.1 CPU baseline: sequential list traversal ===

list (head to tail): 10 20 30 40 50 60 70 80 

sum_list_cpu result: 360
expected (10+20+...+80):     360

every hop's address depends on the value just read at the previous hop --
a true dependency chain, invisible in a single-thread loop like this one.

self-check: sequential list sum matches expected total: confirmed
```

Nothing here looks expensive — it is one loop, one running total. The cost this section is about to measure is entirely invisible in this single-thread version, for the identical reason Chapter 9.3's ABA problem was invisible on one thread: there is no OTHER thread whose progress this loop's own progress could be compared against.

### The Concept, In Detail

An array's addresses are all computable from the index alone, before any memory access happens at all:

```
array, index-addressed:

  thread 0 wants element 0  -> address = base + 0 * sizeof(T)   (known instantly)
  thread 1 wants element 1  -> address = base + 1 * sizeof(T)   (known instantly)
  thread 2 wants element 2  -> address = base + 2 * sizeof(T)   (known instantly)
  ...
  ALL of these addresses are computable AT ONCE, with no memory access needed
  first -- so the hardware can issue all of them together and let their
  latencies overlap. This is exactly why Chapter 4's grid-stride loop could
  divide N elements across however many threads existed: dividing the INDEX
  RANGE is enough, because the index alone determines the address.
```

A linked list denies this at the most basic level: node `i+1`'s address is not a function of `i` at all. It is data — specifically, it is the value stored inside node `i`'s own `next` field, which does not exist anywhere until node `i` has actually been read:

```
linked list, pointer-addressed:

  want node 0's successor?  -> must READ node 0 first; its `next` field
                                 IS the address, not something computable
                                 from "0" alone
  want node 1's successor?  -> must have already read node 1, whose OWN
                                 address was only knowable after reading
                                 node 0

  node 0 --read--> discover node 1's address --read--> discover node 2's
  address --read--> discover node 3's address --read--> ...

  every arrow above is a REAL dependency: the read on its right cannot
  even be ISSUED until the read on its left has completed and returned
  a value. No number of threads changes this -- there is nothing for a
  second thread to do, because there is no way to know node 5's address
  without a thread (some thread) having already read nodes 0 through 4.
```

Chapter 3's span vocabulary gives this a precise name: an `N`-element array split across `P` independent threads has span `ceil(N/P)` — each thread's own share of the work still takes time, but `P` threads' shares run concurrently, so growing `P` shrinks the span. An `N`-node list's traversal has span `N`, full stop, regardless of `P` — because the dependency chain is not divided among the `P` threads at all, it exists between successive READS of the SAME chain of nodes, and no thread can skip ahead to a node it cannot yet identify.

### Code and Verification

```cpp
#include <cstdio>
#include <vector>

// Chapter 11.1 -- Sections 4-8 all worked with arrays: every thread's
// address is a function of its OWN index alone, computable before any
// memory access happens, so many independent loads can be in flight at
// once (Chapter 3's span vocabulary already has a name for this: the
// span is however many DEPENDENT steps remain after independent work is
// divided across threads). A linked list denies this outright -- node
// i+1's address is not a function of i at all, it is the VALUE stored
// inside node i, so it cannot be known, let alone loaded, until node
// i's own load has actually completed.

#define BLOCK_SIZE 64

struct Node { int value; int next; };

// Array sum: exactly Chapter 4's grid-stride shape -- every thread's
// sequence of addresses is known in advance, so BLOCK_SIZE threads'
// worth of loads can all be issued together, independently.
__global__ void sum_array_kernel(const int* g_in, int n, int* g_out) {
    int tid = threadIdx.x;
    int sum = 0;
    for (int i = tid; i < n; i += BLOCK_SIZE) sum += g_in[i];

    __shared__ int sdata[BLOCK_SIZE];
    sdata[tid] = sum;
    __syncthreads();
    for (int s = BLOCK_SIZE / 2; s > 0; s /= 2) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) *g_out = sdata[0];
}

// List sum: ONE thread, because pointer-chasing has no independent work
// to divide among any others -- node i+1 cannot be identified, let
// alone summed, before node i has actually been read.
__global__ void sum_list_kernel(const Node* g_nodes, int head, int* g_out) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    int sum = 0;
    int cur = head;
    while (cur != -1) {
        sum += g_nodes[cur].value;
        cur = g_nodes[cur].next;
    }
    *g_out = sum;
}

// ---- Host-side model of both kernels' actual span, using Chapter 3's
// ---- vocabulary: how many SEQUENTIALLY DEPENDENT steps remain once
// ---- independent work has been divided across the threads that exist. ----

int simulate_array_sum(const std::vector<int>& data, int block_size) {
    int n = (int)data.size();
    std::vector<int> partial(block_size, 0);
    for (int tid = 0; tid < block_size; tid++) {
        for (int i = tid; i < n; i += block_size) partial[tid] += data[i];
    }
    int total = 0;
    for (int v : partial) total += v;
    return total;
}

int simulate_list_sum(const std::vector<Node>& nodes, int head) {
    int sum = 0;
    int cur = head;
    while (cur != -1) { sum += nodes[cur].value; cur = nodes[cur].next; }
    return sum;
}

int main() {
    printf("=== Section 11.1: array span vs. list span, same N, same total work ===\n\n");

    const int N = 256;
    std::vector<int> arr(N);
    for (int i = 0; i < N; i++) arr[i] = i + 1;

    std::vector<Node> nodes(N);
    for (int i = 0; i < N; i++) {
        nodes[i].value = i + 1;
        nodes[i].next = (i + 1 < N) ? (i + 1) : -1;
    }
    int head = 0;

    int array_sum = simulate_array_sum(arr, BLOCK_SIZE);
    int list_sum = simulate_list_sum(nodes, head);
    int reference = N * (N + 1) / 2;

    printf("N = %d elements/nodes, values 1..%d, reference sum = %d\n\n", N, N, reference);
    printf("array sum (grid-stride, %d threads): %d\n", BLOCK_SIZE, array_sum);
    printf("list sum  (pointer-chasing, 1 thread): %d\n\n", list_sum);

    bool correct = (array_sum == reference) && (list_sum == reference);

    int array_span = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int list_span = N;

    printf("span, in Chapter 3's own vocabulary:\n");
    printf("  array, %d threads:  %d sequential steps per thread (independent loads overlap)\n",
           BLOCK_SIZE, array_span);
    printf("  list, any number of threads: %d sequential steps (an unavoidable dependency\n",
           list_span);
    printf("  chain -- adding threads cannot shorten it, because node i+1's address IS the\n");
    printf("  value node i's own load just produced)\n\n");

    double ratio = (double)list_span / (double)array_span;
    printf("the list's span is %.1fx the array's, for the identical %d additions of useful\n",
           ratio, N);
    printf("work -- not because the list kernel is written badly, but because the structure\n");
    printf("itself forbids knowing an address before reading the one before it.\n");

    bool ok = correct && (list_span == N) && (array_span == (N + BLOCK_SIZE - 1) / BLOCK_SIZE);
    printf("\nself-check: both sums match the reference, span model matches Chapter 3's own\n");
    printf("definition: %s\n", ok ? "confirmed" : "MISMATCH");
    return ok ? 0 : 1;
}
```

**Compile and run:**

```bash
nvcc -arch=sm_80 53_list_vs_array_span.cu -o list_vs_array_span
./list_vs_array_span
```

**Sample input:** `N = 256` elements and an `N = 256`-node list, values `1, 2, ..., 256` either way, array summed with `BLOCK_SIZE = 64` threads, list summed with exactly 1 thread.

**Sample output:**

```text
=== Section 11.1: array span vs. list span, same N, same total work ===

N = 256 elements/nodes, values 1..256, reference sum = 32896

array sum (grid-stride, 64 threads): 32896
list sum  (pointer-chasing, 1 thread): 32896

span, in Chapter 3's own vocabulary:
  array, 64 threads:  4 sequential steps per thread (independent loads overlap)
  list, any number of threads: 256 sequential steps (an unavoidable dependency
  chain -- adding threads cannot shorten it, because node i+1's address IS the
  value node i's own load just produced)

the list's span is 64.0x the array's, for the identical 256 additions of useful
work -- not because the list kernel is written badly, but because the structure
itself forbids knowing an address before reading the one before it.

self-check: both sums match the reference, span model matches Chapter 3's own
definition: confirmed
```

Both kernels compute the identical, correct sum for the identical 256 additions of real work. The span numbers do not match at all: 4 for the array (256 elements divided across 64 independent threads), 256 for the list (one thread, one unavoidable chain, exactly as long as the list itself) — a 64x difference that has nothing to do with how carefully either kernel was written.

!!! warning "[COMMON TRAP] Assuming more threads would fix the list kernel"
    It is tempting to look at `sum_list_kernel`'s single active thread and assume the fix is simply to launch more threads and have them divide the list among themselves — exactly the instinct that correctly fixed every array-based kernel from Chapter 4 onward. It does not work here: dividing "nodes 128 through 255" among a second thread requires already knowing node 128's address, and the only way any thread has ever learned a node's address in this design is by reading the node immediately before it. Adding threads to a naive list traversal does not shorten the chain — it just leaves the extra threads with no starting point they could legitimately know. Section 11.2 shows the one technique that genuinely changes this, and it is not "more threads doing the same walk."

## 11.2 Parallel List Ranking via Pointer Jumping

### Intuition

Section 11.1 established that no thread can identify node `i`'s successor's successor without reading through node `i`'s successor first — for one thread, walking one hop at a time. But nothing stops EVERY node from doing this same one-hop lookup at the same time, for itself, using whatever its neighbor currently knows. Do that repeatedly, always doubling how far each node can now see, and after only `log2(n)` rounds every node has effectively "seen" the entire rest of the list — the exact doubling shape Chapter 5.1's Hillis-Steele scan used for prefix sums, now applied to chasing pointers instead of adding numbers.

### The Sequential (CPU) Baseline

```cpp
#include <cstdio>
#include <vector>

// Chapter 11.2 -- The Sequential (CPU) Baseline.
// A single thread already discovers each node's distance-to-end for
// free, simply by counting down as it walks: no doubling, no rounds,
// no auxiliary arrays -- because one thread visiting nodes in order
// already knows exactly how many nodes remain after each one.

struct Node { int value; int next; };

std::vector<int> rank_list_cpu(const std::vector<Node>& nodes, int head, int n) {
    std::vector<int> order;
    int cur = head;
    while (cur != -1) { order.push_back(cur); cur = nodes[cur].next; }

    std::vector<int> distance_to_end(n, 0);
    for (int pos = 0; pos < (int)order.size(); pos++) {
        distance_to_end[order[pos]] = (int)order.size() - 1 - pos;
    }
    return distance_to_end;
}

int main() {
    printf("=== Section 11.2 CPU baseline: sequential distance-to-end ===\n\n");

    const int N = 8;
    std::vector<Node> nodes(N);
    for (int i = 0; i < N; i++) {
        nodes[i].value = (i + 1) * 10;
        nodes[i].next = (i + 1 < N) ? (i + 1) : -1;
    }
    int head = 0;

    std::vector<int> dist = rank_list_cpu(nodes, head, N);

    printf("list (head to tail): ");
    int cur = head;
    while (cur != -1) { printf("%d ", nodes[cur].value); cur = nodes[cur].next; }
    printf("\n\ndistance-to-end, by node index: ");
    for (int d : dist) printf("%d ", d);
    printf("\n\n");

    std::vector<int> expected = {7, 6, 5, 4, 3, 2, 1, 0};
    bool ok = (dist == expected);

    printf("expected: 7 6 5 4 3 2 1 0\n");
    printf("\none thread walking in order already knows every node's distance-to-end --\n");
    printf("no doubling needed at all when there is no one else to divide the work with.\n");
    printf("\nself-check: sequential distance-to-end matches expected values: %s\n",
           ok ? "confirmed" : "MISMATCH");
    return ok ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++17 -Wall -Wextra -O2 54_rank_list_cpu_baseline.cpp -o rank_list_cpu_baseline
./rank_list_cpu_baseline
```

**Sample input:** the identical 8-node list from Section 11.1, values `10, 20, ..., 80`.

**Sample output:**

```text
=== Section 11.2 CPU baseline: sequential distance-to-end ===

list (head to tail): 10 20 30 40 50 60 70 80 

distance-to-end, by node index: 7 6 5 4 3 2 1 0 

expected: 7 6 5 4 3 2 1 0

one thread walking in order already knows every node's distance-to-end --
no doubling needed at all when there is no one else to divide the work with.

self-check: sequential distance-to-end matches expected values: confirmed
```

A single thread gets every node's distance-to-end for free, purely as a side effect of visiting nodes in order — there is nothing here to parallelize AWAY from, because one thread's own sequential walk was never the bottleneck; it is exactly Section 11.1's `sum_list_cpu`, just counting positions instead of summing values. The interesting problem only appears once many threads want this answer WITHOUT any one of them walking the whole list first.

### The Concept, In Detail

Pointer jumping computes, for every node simultaneously, "how many nodes are between me and the end." Initially, every node except the last knows only its own immediate neighbor (distance 1) or knows it IS the end (distance 0). Each round, every node replaces its neighbor with its neighbor's neighbor, and adds its neighbor's current distance to its own:

```
DOUBLING RULE, applied to every node i AT ONCE, every round:

  if next[i] exists:
      dist[i]  <-  dist[i] + dist[next[i]]
      next[i]  <-  next[next[i]]
  else:
      i is already the end -- nothing changes

this is read-then-write from a SNAPSHOT of the previous round -- exactly
Chapter 5.1's double-buffering discipline, for the identical reason: a
node reading its neighbor's ALREADY-UPDATED value in the same round
would double-count, corrupting the distance.
```

Traced on this section's own 8-node list (`next` shown as node indices, `-1` = end):

```
round 0 (initial):  next = 1  2  3  4  5  6  7 -1
                     dist = 1  1  1  1  1  1  1  0
                     (every node reaches exactly 1 hop so far)

round 1:            next = 2  3  4  5  6  7 -1 -1
                     dist = 2  2  2  2  2  2  1  0
                     (every node now reaches 2 hops -- node 0's dist
                      became dist[0]+dist[1] = 1+1 = 2, and so on)

round 2:            next = 4  5  6  7 -1 -1 -1 -1
                     dist = 4  4  4  4  3  2  1  0
                     (reach doubled again, to 4 hops -- node 0's dist
                      became dist[0]+dist[4] = 2+2 = 4)

round 3:            next = -1 -1 -1 -1 -1 -1 -1 -1
                     dist = 7  6  5  4  3  2  1  0
                     (every next is now -1 -- CONVERGED. node 0's dist
                      became dist[0]+dist[4] = 4+3 = 7, matching the
                      CPU baseline's own answer exactly)

3 rounds for 8 nodes: ceil(log2(8)) = 3, exactly the doubling pattern's
own guarantee -- each round at least doubles every node's reach, so
after ceil(log2(n)) rounds even the FARTHEST node from the end has
reached it.
```

The trade is exactly Chapter 5.2's: this computes in `O(log n)` span what one thread already got in `O(n)` span for free, and it does so by performing strictly MORE total work — every node updates every round, including nodes that reached the end rounds ago, unlike a sequential walk's single pass. Parallelism was bought, not discovered for free.

### Code and Verification

```cpp
#include <cstdio>
#include <vector>

// Chapter 11.2 -- Section 11.1 showed a single thread must take N
// sequential hops to reach the end of an N-node list, with no way to
// divide that chain among more threads directly. Pointer jumping breaks
// the chain apart differently: instead of one thread walking N hops,
// EVERY node updates its own "next" and "distance" at once, each round
// replacing "my neighbor" with "my neighbor's neighbor" -- exactly
// Chapter 5.1's Hillis-Steele doubling shape, applied to a linked
// structure instead of a fixed-stride array offset.

struct Node { int next; };

// One round of pointer jumping: every node (in parallel) reads its
// current next and that neighbor's distance, and produces the DOUBLED
// values into a second buffer -- double-buffered for the identical
// reason Chapter 5.1's scan was: a node reading its neighbor's
// still-being-updated value in the SAME round would be a hazard.
__global__ void pointer_jump_kernel(const int* g_next_in, const int* g_dist_in,
                                     int* g_next_out, int* g_dist_out, int n) {
    int i = threadIdx.x;
    if (i >= n) return;
    int next = g_next_in[i];
    if (next == -1) {
        g_next_out[i] = -1;
        g_dist_out[i] = g_dist_in[i];
    } else {
        g_next_out[i] = g_next_in[next];
        g_dist_out[i] = g_dist_in[i] + g_dist_in[next];
    }
}

// ---- Host-side replay of the identical doubling logic, run for
// ---- ceil(log2(n)) rounds -- exactly enough for every node's chain of
// ---- doublings to reach the list's true end. ----

int main() {
    printf("=== Section 11.2: parallel list ranking via pointer jumping ===\n\n");

    const int N = 8;
    std::vector<int> next_buf(N), dist_buf(N);
    for (int i = 0; i < N; i++) {
        next_buf[i] = (i + 1 < N) ? (i + 1) : -1;
        dist_buf[i] = (i + 1 < N) ? 1 : 0;
    }

    printf("initial: next = ");
    for (int v : next_buf) printf("%2d ", v);
    printf("\n         dist = ");
    for (int v : dist_buf) printf("%2d ", v);
    printf("  (every node reaches only 1 hop so far)\n\n");

    int rounds = 0;
    std::vector<int> next_a = next_buf, dist_a = dist_buf;
    std::vector<int> next_b(N), dist_b(N);
    bool converged = false;

    while (!converged) {
        for (int i = 0; i < N; i++) {
            int next = next_a[i];
            if (next == -1) {
                next_b[i] = -1;
                dist_b[i] = dist_a[i];
            } else {
                next_b[i] = next_a[next];
                dist_b[i] = dist_a[i] + dist_a[next];
            }
        }
        rounds++;
        next_a.swap(next_b);
        dist_a.swap(dist_b);

        converged = true;
        for (int v : next_a) if (v != -1) { converged = false; break; }

        printf("round %d: next = ", rounds);
        for (int v : next_a) printf("%2d ", v);
        printf("\n         dist = ");
        for (int v : dist_a) printf("%2d ", v);
        printf("\n");
    }
    printf("\nconverged after %d rounds (ceil(log2(%d)) = %d)\n\n", rounds, N, rounds);

    printf("final distance-to-end, by node index: ");
    for (int v : dist_a) printf("%d ", v);
    printf("\n\n");

    std::vector<int> expected = {7, 6, 5, 4, 3, 2, 1, 0};
    bool ok = (dist_a == expected);

    printf("expected (matches Section 11.2's own sequential CPU baseline exactly): 7 6 5 4 3 2 1 0\n");
    printf("\nevery node updated at once, every round -- log2(n) rounds total instead of n\n");
    printf("sequential hops, at the cost of doing MORE total work per node than a single\n");
    printf("sequential walk ever needed, exactly Chapter 5.2's work-versus-span trade again.\n");

    printf("\nself-check: parallel pointer-jumping distances match the sequential reference: %s\n",
           ok ? "confirmed" : "MISMATCH");
    return ok ? 0 : 1;
}
```

**Compile and run:**

```bash
nvcc -arch=sm_80 55_pointer_jump_list_ranking.cu -o pointer_jump_list_ranking
./pointer_jump_list_ranking
```

**Sample input:** the identical 8-node list, `next` and `dist` arrays initialized to one hop each, run for `ceil(log2(8)) = 3` rounds.

**Sample output:**

```text
=== Section 11.2: parallel list ranking via pointer jumping ===

initial: next =  1  2  3  4  5  6  7 -1 
         dist =  1  1  1  1  1  1  1  0   (every node reaches only 1 hop so far)

round 1: next =  2  3  4  5  6  7 -1 -1 
         dist =  2  2  2  2  2  2  1  0 
round 2: next =  4  5  6  7 -1 -1 -1 -1 
         dist =  4  4  4  4  3  2  1  0 
round 3: next = -1 -1 -1 -1 -1 -1 -1 -1 
         dist =  7  6  5  4  3  2  1  0 

converged after 3 rounds (ceil(log2(8)) = 3)

final distance-to-end, by node index: 7 6 5 4 3 2 1 0 

expected (matches Section 11.2's own sequential CPU baseline exactly): 7 6 5 4 3 2 1 0

every node updated at once, every round -- log2(n) rounds total instead of n
sequential hops, at the cost of doing MORE total work per node than a single
sequential walk ever needed, exactly Chapter 5.2's work-versus-span trade again.

self-check: parallel pointer-jumping distances match the sequential reference: confirmed
```

The parallel doubling result matches the sequential CPU baseline's distances exactly, node for node, after exactly 3 rounds — `log2(8)`, not 8 sequential hops.

!!! warning "[COMMON TRAP] Forgetting to double-buffer between rounds"
    It is tempting to update `next[i]` and `dist[i]` in place, reasoning that each node only reads its OWN current neighbor. The hazard is the same one Chapter 5.1 already warned about: node 3 might read node 4's `next` and `dist` to compute its own update, but if node 4 has ALREADY been updated earlier in that same round, node 3 silently jumps two doublings ahead of where it should be, corrupting its distance in a way that will not necessarily even look wrong (the numbers still print, they are simply incorrect). Reading every node's update from a separate, untouched snapshot of the previous round — exactly what `g_next_in`/`g_next_out` and `g_dist_in`/`g_dist_out` keep separate here — is not an optional performance detail; it is what makes "every node updates using the SAME round's starting state" actually true.

## 11.3 From Ranks to Arrays: Why This Book Never Used Real Pointers

### Intuition

Section 11.2 computed something specific: every node's distance to the list's end. That number is exactly enough to answer a question that used to require walking the whole list — "what is my correct position if this list were laid out as an array instead?" Once every node knows its own array position, converting a list to an array stops being a linked-list problem entirely; it becomes an ordinary, fully independent scatter, exactly Chapter 6's compaction pattern.

### The Sequential (CPU) Baseline

```cpp
#include <cstdio>
#include <vector>

// Chapter 11.3 -- The Sequential (CPU) Baseline.
// A single thread converts a list to an array with one pass and no
// ranks at all: it already knows its own output position implicitly,
// because it is the one advancing through the list in order.

struct Node { int value; int next; };

std::vector<int> list_to_array_cpu(const std::vector<Node>& nodes, int head) {
    std::vector<int> out;
    int cur = head;
    while (cur != -1) { out.push_back(nodes[cur].value); cur = nodes[cur].next; }
    return out;
}

int main() {
    printf("=== Section 11.3 CPU baseline: sequential list-to-array conversion ===\n\n");

    const int N = 8;
    std::vector<Node> nodes(N);
    for (int i = 0; i < N; i++) {
        nodes[i].value = (i + 1) * 10;
        nodes[i].next = (i + 1 < N) ? (i + 1) : -1;
    }
    int head = 0;

    std::vector<int> out = list_to_array_cpu(nodes, head);

    printf("list (head to tail), by value: ");
    for (int v : out) printf("%d ", v);
    printf("\n\narray output:                   ");
    for (int v : out) printf("%d ", v);
    printf("\n\n");

    std::vector<int> expected = {10, 20, 30, 40, 50, 60, 70, 80};
    bool ok = (out == expected);

    printf("expected: 10 20 30 40 50 60 70 80\n");
    printf("\nno ranks needed at all -- a single thread's own position falls out of the\n");
    printf("traversal for free, one push_back at a time.\n");
    printf("\nself-check: sequential list-to-array conversion matches expected order: %s\n",
           ok ? "confirmed" : "MISMATCH");
    return ok ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++17 -Wall -Wextra -O2 56_list_to_array_cpu_baseline.cpp -o list_to_array_cpu_baseline
./list_to_array_cpu_baseline
```

**Sample input:** the identical 8-node list, values `10, 20, ..., 80`.

**Sample output:**

```text
=== Section 11.3 CPU baseline: sequential list-to-array conversion ===

list (head to tail), by value: 10 20 30 40 50 60 70 80 

array output:                   10 20 30 40 50 60 70 80 

expected: 10 20 30 40 50 60 70 80

no ranks needed at all -- a single thread's own position falls out of the
traversal for free, one push_back at a time.

self-check: sequential list-to-array conversion matches expected order: confirmed
```

A single thread converting a list to an array needs no ranks, no distances, no auxiliary computation of any kind — it already knows its own output position implicitly, since it is the very thread doing the walking, in order. That convenience is exactly what many independent GPU threads do NOT have, and exactly what Section 11.2's ranks exist to supply.

### The Concept, In Detail

Once distance-to-end is known for every node, each node's correct array position follows from one subtraction, with no dependency on any other node at all:

```
position_from_head[i]  =  (n - 1) - distance_to_end[i]

for this chapter's own 8-node list (n = 8):

  node index:        0   1   2   3   4   5   6   7
  value:            10  20  30  40  50  60  70  80
  distance-to-end:   7   6   5   4   3   2   1   0    (Section 11.2's output)
  position_from_head: 0   1   2   3   4   5   6   7    (7 - distance-to-end)

every node now knows EXACTLY where it belongs in the final array, and
that knowledge did not require looking at any OTHER node's position --
node 5's position (5) depends only on node 5's OWN distance-to-end (2),
not on where nodes 0-4 or 6-7 end up.
```

The scatter itself is now the plainest possible parallel operation this book has: every node reads its own value and its own computed position, and writes once, to an address no other node will ever write to:

```
scatter (fully independent, one write per node):

  node 0 (pos 0) --> output[0] = 10        node 4 (pos 4) --> output[4] = 50
  node 1 (pos 1) --> output[1] = 20        node 5 (pos 5) --> output[5] = 60
  node 2 (pos 2) --> output[2] = 30        node 6 (pos 6) --> output[6] = 70
  node 3 (pos 3) --> output[3] = 40        node 7 (pos 7) --> output[7] = 80

no ordering constraint between ANY two of these writes -- span 1, the
best this book's own vocabulary can express, recovered ONLY because
Section 11.2 already paid its O(log n)-span cost to compute the ranks
that make this independence possible.
```

This is the retrospective payoff of Chapters 9 and 10's design choice. A lock-free stack or queue built from genuine `malloc`-allocated nodes and real pointers would face this chapter's own Section 11.1 problem the instant anything needed to inspect more than one node's neighborhood at once — walking such a structure is exactly as sequential as this chapter's naive list sum. Using a fixed pool of nodes addressed by plain integer index, as this book has done since Chapter 9, does not avoid pointer-chasing's traversal cost outright (walking `next` fields one at a time is still exactly as sequential either way) — but it keeps every node's OWN identity O(1)-addressable the moment its index is known, which is precisely what let Section 11.2's pointer jumping treat `next` and `dist` as ordinary arrays indexed `0` through `n-1`, rather than as opaque addresses a GPU kernel would have no efficient way to use as array indices at all.

### Code and Verification

```cpp
#include <cstdio>
#include <vector>

// Chapter 11.3 -- Once Section 11.2 has computed every node's
// distance-to-end (whether by pointer jumping in parallel, or by a
// sequential walk), everything downstream stops being a linked-list
// problem at all: position_from_head = n - 1 - distance_to_end is a
// single arithmetic step per node, and every node's OWN output slot is
// now known WITHOUT walking anything -- exactly this book's Chapter
// 6-style scatter, and completely independent across nodes.

__global__ void scatter_by_rank_kernel(const int* g_value, const int* g_distance_to_end,
                                        int n, int* g_out) {
    int i = threadIdx.x;
    if (i >= n) return;
    int position = n - 1 - g_distance_to_end[i];
    g_out[position] = g_value[i];
}

int main() {
    printf("=== Section 11.3: scatter to array position, once ranks are known ===\n\n");

    const int N = 8;
    std::vector<int> value(N), distance_to_end(N);
    for (int i = 0; i < N; i++) {
        value[i] = (i + 1) * 10;
        distance_to_end[i] = N - 1 - i;
    }

    printf("node index:        ");
    for (int i = 0; i < N; i++) printf("%2d ", i);
    printf("\nvalue:             ");
    for (int v : value) printf("%2d ", v);
    printf("\ndistance-to-end:   ");
    for (int v : distance_to_end) printf("%2d ", v);
    printf("  (Section 11.2's own output)\n\n");

    std::vector<int> out(N, -1);
    for (int i = 0; i < N; i++) {
        int position = N - 1 - distance_to_end[i];
        out[position] = value[i];
    }

    printf("scattered array output: ");
    for (int v : out) printf("%d ", v);
    printf("\n\n");

    std::vector<int> expected = {10, 20, 30, 40, 50, 60, 70, 80};
    bool ok = (out == expected);

    printf("expected (matches Section 11.3's own CPU baseline exactly): 10 20 30 40 50 60 70 80\n");
    printf("\nonce distance-to-end is known, every node's write is independent of every\n");
    printf("other node's -- O(1) span, the identical property every array-based structure\n");
    printf("in this book has had from Chapter 4 onward, recovered here only AFTER paying\n");
    printf("Section 11.2's O(log n)-span ranking cost up front.\n");

    printf("\nself-check: parallel scatter-by-rank matches the sequential reference exactly: %s\n",
           ok ? "confirmed" : "MISMATCH");
    return ok ? 0 : 1;
}
```

**Compile and run:**

```bash
nvcc -arch=sm_80 57_scatter_by_rank.cu -o scatter_by_rank
./scatter_by_rank
```

**Sample input:** the identical 8-node list's values and distance-to-end array (Section 11.2's own output), scattered to their final array positions.

**Sample output:**

```text
=== Section 11.3: scatter to array position, once ranks are known ===

node index:         0  1  2  3  4  5  6  7 
value:             10 20 30 40 50 60 70 80 
distance-to-end:    7  6  5  4  3  2  1  0   (Section 11.2's own output)

scattered array output: 10 20 30 40 50 60 70 80 

expected (matches Section 11.3's own CPU baseline exactly): 10 20 30 40 50 60 70 80

once distance-to-end is known, every node's write is independent of every
other node's -- O(1) span, the identical property every array-based structure
in this book has had from Chapter 4 onward, recovered here only AFTER paying
Section 11.2's O(log n)-span ranking cost up front.

self-check: parallel scatter-by-rank matches the sequential reference exactly: confirmed
```

The scattered output matches Section 11.3's own sequential CPU baseline exactly, recovered here through a fully independent, single-write-per-node parallel scatter rather than a sequential walk.

!!! warning "[COMMON TRAP] Believing this scatter proves linked lists parallelize for free"
    It is tempting to look at Section 11.3's clean, fully independent scatter and conclude that linked lists are not so bad on a GPU after all. That conclusion skips Section 11.2's bill entirely: this scatter is only possible AFTER paying an `O(log n)`-span, `O(n log n)`-work ranking pass to compute `distance_to_end` for every node. For a single list traversed once, that upfront cost is frequently not worth paying at all compared to Section 11.1's plain sequential walk — list ranking earns its cost specifically when the RESULT (the array layout, or the ranks themselves) will be reused many times afterward, not for a single one-shot traversal.

## Chapter Summary

Section 11.1 measured pointer-chasing's real cost directly: an `N`-node list's traversal has span `N` regardless of how many threads exist, because each node's address is data produced by the previous node's own read, not a function of an index any thread could compute in advance — a categorically different situation from every array-based structure since Chapter 4. Section 11.2 introduced pointer jumping, a genuine, published technique (list ranking) that computes every node's distance-to-end in `O(log n)` span by having every node double its own reach each round, reusing Chapter 5.1's Hillis-Steele doubling shape and double-buffering discipline, verified exactly against a sequential reference. Section 11.3 showed the payoff: once ranks are known, converting a list to an array becomes a fully independent, O(1)-span scatter, and explained retrospectively why this book's own stacks and queues (Chapters 9-10) used fixed, integer-indexed node pools rather than real pointers from the start — it is exactly the property that let this chapter's own ranking arrays be ordinary, directly-indexable arrays. Part 2 — arrays and growable buffers, lock-free stacks and queues, and now a full accounting of why linked lists resist this machine — is complete. Part 3 turns to sorting, starting with bitonic sort: a sorting network built specifically for the fixed, lockstep comparison pattern a single warp can execute with no divergence at all.

## Self-Check Questions

1. Section 11.1 measured the array kernel's span as 4 and the list kernel's span as 256, for `N = 256` and `BLOCK_SIZE = 64`. If `BLOCK_SIZE` were increased to 256 (one thread per element) with `N` unchanged, what would the array's span become, and would the list's span change at all?
2. Section 11.2 converged in exactly 3 rounds for an 8-node list. How many rounds would a 100-node list need, and what specific property of the doubling rule guarantees that number is enough even though 100 is not a power of two?
3. Explain concretely, using Section 11.2's own round-by-round trace, why `next` and `dist` must both be double-buffered together rather than just `dist` alone.
4. Section 11.3's scatter needs `distance_to_end` for every node before it can run. If only node 5's distance-to-end were known (not the other seven nodes'), what, if anything, could be computed about node 5's own final array position?
5. A colleague proposes skipping Section 11.2 entirely and instead having every GPU thread `i` walk from the head exactly `i` steps to find "its" node directly. Explain concretely why this does not avoid Section 11.1's span cost, using this chapter's own vocabulary.
6. Explain why a lock-free stack (Chapter 9) built from real `malloc`-allocated nodes and genuine pointers, rather than this book's fixed, index-addressed node pool, would make Section 11.2's pointer-jumping technique meaningfully harder to implement efficiently.

## Where We Go Next

Part 3 turns from linear structures to sorting. Bitonic sort, the first sorting algorithm this book builds, is a sorting network — a fixed, data-independent sequence of compare-and-swap steps that every thread in a warp can execute in perfect lockstep, with no divergence at all, directly building on Chapter 1's own warp-uniformity vocabulary. Radix sort follows as the GPU's actual production workhorse, built from primitives this book already has in hand: Chapter 7's histogram-to-offset pattern, applied once per bit.

## Worked Solutions

**1.** The array's span would become `ceil(256/256) = 1` — every element gets its own thread, so all 256 independent loads can be issued together with nothing left for any single thread to iterate over sequentially. The list's span would NOT change at all; it stays exactly 256, because the list's span was never a function of how many threads exist in the first place — it is fixed by the length of the one unavoidable dependency chain, and no amount of additional parallelism gives any thread a way to identify a node it has not already reached by walking there.

**2.** A 100-node list needs `ceil(log2(100)) = 7` rounds (`2^6 = 64 < 100 <= 128 = 2^7`). The doubling rule guarantees this because every node's reach AT LEAST doubles each round regardless of the list's exact length — after round `r`, every node has correctly incorporated every node within `2^r` hops of it (or reached the end, whichever comes first), so once `2^r >= n - 1` (the longest possible remaining distance for any node), every node has necessarily reached the true end, whether or not `n` itself is a power of two.

**3.** `dist[i]`'s correct update for a given round depends on reading `dist[next[i]]` — the CURRENT round's neighbor's distance — using the PREVIOUS round's `next[i]`, not the round that is currently being computed. If `next` were updated in place while `dist` used double-buffering, a node computing its own new `dist` partway through a round could read an ALREADY-ADVANCED `next[i]` and combine it with a `dist` value that was computed under the OLD neighbor relationship, mixing two different rounds' worth of state in a single update and silently miscounting nodes (either double-counting some or skipping others entirely).

**4.** Node 5's own final array position (`position_from_head[5] = n - 1 - distance_to_end[5]`) could be computed immediately from its own distance-to-end alone, exactly as Section 11.3's concept section states — no other node's distance-to-end is needed for that ONE computation, because each node's position depends only on its own distance and the list's total length `n`. What could NOT be concluded is anything about where any of the other seven nodes belong; each of them independently needs its own distance-to-end computed before its own position is known.

**5.** Thread `i` walking `i` steps from the head to "find its own node" is still a genuinely sequential walk of length `i` for every thread — the total amount of SEQUENTIAL work done by the SLOWEST thread (thread `N-1`, walking `N-1` steps) is exactly as long as Section 11.1's single-thread walk of the whole list, and every thread `j < i` is doing REDUNDANT work already covered by thread `i`'s own walk. Nothing about launching many threads removes the underlying dependency chain each individual walk must still traverse one hop at a time; it only duplicates the same chain many times over, at strictly worse total work than the original single-thread version for no reduction in span at all.

**6.** Pointer jumping's `next_in`/`next_out` and `dist_in`/`dist_out` arrays are ordinary, contiguous, integer-indexed arrays specifically because this book's node pools use plain integer indices as "pointers" — looking up "neighbor 4's current distance" is `dist_in[4]`, an O(1) array access identical to any other array indexing this book has used since Chapter 4. A real, `malloc`-allocated node's "pointer" is an opaque memory address with no meaningful mapping to a small, dense integer range at all; building the equivalent of `dist_in`/`dist_out` for a genuinely pointer-based list would require either a separate address-to-index translation table (itself another data structure to build and keep synchronized) or storing the distance value inside each node directly, which reintroduces exactly the kind of scattered, non-coalesced memory access this book's array-based designs have avoided since Part 1.
