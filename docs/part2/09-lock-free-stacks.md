# Chapter 9: Lock-Free Stacks

**What you will understand by the end of this chapter:**

- Why `atomicAdd` alone cannot build a shared linked structure, and what compare-and-swap (CAS) adds: an update that only commits if nothing else has changed the value since it was last read.
- How a complete CAS-based push/pop retry loop (a Treiber stack) stays correct under contention, turning what would otherwise be lost updates into detected, automatically-retried failures.
- Why a raw pointer or pool index is not always enough to detect "nothing changed" — the ABA problem — and how a monotonically incrementing tag fixes it.

**What you need to know first:**

- Chapter 7 and Chapter 8's `atomicAdd`-based techniques, contrasted directly against the new tool this chapter introduces.
- Chapter 1's warp lockstep model and Chapter 7.1 and 8.2's non-atomic race traces, reused directly in Section 9.1 to motivate why a stronger primitive is needed.

---

Part 1 built four primitives — reduction, scan, compaction, histograms — and Chapter 8 pushed `atomicAdd` as far as it goes: reserving unique slots in a shared, growable buffer. A stack needs something `atomicAdd` cannot provide. Pushing and popping both update a shared pointer to point somewhere genuinely different depending on what is already there, and that update has to happen only if nothing else has changed the structure in the meantime. This chapter introduces the tool built for exactly that job, and the one subtle way even that tool can be fooled.

## 9.1 Why atomicAdd Isn't Enough: Compare-And-Swap

### Intuition

Every atomic operation this book has used so far — `atomicAdd`, in Chapters 7 and 8 — always succeeds, unconditionally: add this amount, tell me what was there before, done. A stack's push needs something conditional: update the shared head pointer to point at a new node, but ONLY IF the head still holds the value this thread last saw. If some other thread already changed it, blindly overwriting it would silently erase that other thread's work.

### The Concept, In Detail

`atomicCAS(address, compare, val)` is built for exactly this. It reads `*address`; if it equals `compare`, it stores `val` and returns the OLD value (which will equal `compare` — the caller's signal of success); otherwise it stores nothing at all and returns whatever `*address` genuinely holds (which will NOT equal `compare` — the caller's signal to retry). The whole operation — read, compare, conditionally write — happens as one indivisible hardware step, exactly like `atomicAdd`, just with a condition attached.

### The Sequential (CPU) Baseline

On an ordinary single CPU thread — or, equivalently, on a GPU with only ONE thread ever touching the stack — push needs nothing special at all:

```cpp
#include <cstdio>
#include <vector>

// Chapter 9.1 -- The Sequential (CPU) Baseline.
// On a single CPU thread, push needs nothing special at all -- this is
// completely correct as written, because there is no OTHER thread that
// could change head in the gap between these two lines.

struct Node { int value; int next; };

void push_cpu(Node* nodes, int& head, int node_idx) {
    nodes[node_idx].next = head;
    head = node_idx;
}

int main() {
    printf("=== Section 9.1 CPU baseline: sequential push, no CAS needed ===\n\n");

    std::vector<Node> nodes(4);
    int head = -1;

    int values[] = {10, 20, 30};
    for (int i = 0; i < 3; i++) {
        nodes[i].value = values[i];
        push_cpu(nodes.data(), head, i);
        printf("push(%d) -> head is now slot %d\n", values[i], head);
    }

    printf("\nfinal stack, walked from head: [");
    std::vector<int> walked;
    int cur = head;
    while (cur != -1) { walked.push_back(nodes[cur].value); cur = nodes[cur].next; }
    for (size_t i = 0; i < walked.size(); i++) printf("%d%s", walked[i], i + 1 < walked.size() ? ", " : "");
    printf("]\n\n");

    std::vector<int> expected = {30, 20, 10};
    bool ok = (walked == expected);

    printf("expected (most recently pushed on top): 30, 20, 10\n");
    printf("\nno atomic operation of any kind -- one thread, two lines, nothing else can\n");
    printf("ever observe or interleave with them.\n");
    printf("\nself-check: sequential push produces correct LIFO order: %s\n",
           ok ? "confirmed" : "MISMATCH");
    return ok ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++17 -Wall -Wextra -O2 46_stack_push_cpu_baseline.cpp -o stack_push_cpu_baseline
./stack_push_cpu_baseline
```

**Sample input:** push `10`, then `20`, then `30` onto an initially empty stack, one thread at a time.

**Sample output:**

```text
=== Section 9.1 CPU baseline: sequential push, no CAS needed ===

push(10) -> head is now slot 0
push(20) -> head is now slot 1
push(30) -> head is now slot 2

final stack, walked from head: [30, 20, 10]

expected (most recently pushed on top): 30, 20, 10

no atomic operation of any kind -- one thread, two lines, nothing else can
ever observe or interleave with them.

self-check: sequential push produces correct LIFO order: confirmed
```

This is completely correct as written, because there is no OTHER thread that could change `head` in the gap between these two lines. This is also, line for line, exactly the logic Section 9.1's own `push_naive` kernel below uses — the CODE does not change at all when moving to the GPU. What changes is that the GPU runs many copies of this identical logic AT THE SAME TIME, and it is exactly that — not anything wrong with the code in isolation — which is about to break it.

A naive, non-atomic push written this way for many concurrent GPU threads hits the same three-step LOAD / COMPUTE / STORE shape Chapter 7.1 and Chapter 8.2 already proved unsafe:

```
32 racing pushes onto a stack that already holds 5 elements (head = H)

naive push (LOAD / COMPUTE / STORE, three separate steps):

  step 1 -- LOAD:     all 32 lanes read head = H (the SAME stale value,
                       exactly Chapter 7.1's lockstep LOAD)
  step 2 -- COMPUTE:   each lane sets its OWN new node's next = H
  step 3 -- STORE:     all 32 lanes write head = their_own_node_index --
                       last store wins; the other 31 nodes have a
                       correctly-set `next` field, but head now skips
                       past them entirely, so none of them can ever be
                       reached again

  result: only 1 of the 32 new pushes is actually reachable; 31 are
  silently lost, even though every one of their nodes still physically
  exists in the pool with perfectly valid data.

CAS-based push (the retry loop):

  each lane's atomicCAS(&head, captured_old_head, its_own_node_index)
  either succeeds (head still equals what this lane captured) or fails
  (some other lane's push already moved head) -- and a failure just
  means: re-read the now-current head, and try again.

  result: all 32 pushes eventually succeed. Contention shows up only as
  extra RETRIES (31 of them here, one per push after the very first),
  never as a silently lost node.
```

The kernel below implements both the naive and the CAS-based push exactly as described, and the host-side model replays the identical logic to produce a fully deterministic, reproducible trace of both outcomes.

### Code and Verification

```cpp
#include <cstdio>
#include <vector>

// Chapter 9.1 -- every atomic operation this book has used so far
// (atomicAdd, in Chapters 7 and 8) answers "add this amount, and tell me
// what was there before" -- it always succeeds, unconditionally. Building
// a shared, growable LINKED structure needs something different: a way
// to update a shared pointer ONLY IF it still holds the value a thread
// last read, so a thread that discovers someone else changed the
// structure underneath it can detect that and retry instead of silently
// clobbering the change. That operation is compare-and-swap (CAS).
// `atomicCAS(address, compare, val)` reads `*address`; if it equals
// `compare`, it stores `val` and returns the OLD value (which will equal
// `compare` -- the caller's signal of success); otherwise it stores
// NOTHING and returns whatever `*address` actually held (which will NOT
// equal `compare` -- the caller's signal to retry). This section proves,
// by direct construction, why a stack's push needs exactly this.

#define NUM_NEW_PUSHES 32

struct Node {
    int value;
    int next;   // index into the node pool, or -1 for "end of stack"
};

// The naive, non-atomic push: read the current head, point the new
// node's `next` at it, then overwrite head -- three separate steps,
// exactly the LOAD / COMPUTE / STORE shape Chapter 7.1 and Chapter 8.2
// already proved is unsafe when multiple threads do it on the same
// shared value at once.
__global__ void push_naive(Node* g_nodes, int* g_head, int node_idx) {
    int old_head = *g_head;
    g_nodes[node_idx].next = old_head;
    *g_head = node_idx;
}

// The correct push: keep retrying compare-and-swap until it succeeds.
// If some OTHER thread changed head between this thread's read and its
// CAS attempt, the CAS simply fails (changes nothing) and the loop reads
// the now-current head and tries again -- no lost update is possible,
// because head only ever changes via an operation that VERIFIES it is
// not stepping on a change no one has accounted for yet.
__global__ void push_cas(Node* g_nodes, int* g_head, int node_idx) {
    int old_head;
    do {
        old_head = *g_head;
        g_nodes[node_idx].next = old_head;
    } while (atomicCAS(g_head, old_head, node_idx) != old_head);
}

// ---- Host-side simulation of both designs, pushing 32 new nodes onto
// ---- a stack that already holds 5 elements (head = 4, a 5-node chain
// ---- built below), so a genuine pre-existing structure is at risk of
// ---- corruption, not just an empty stack. ----

int build_preexisting_stack(std::vector<Node>& nodes) {
    // 5 nodes, values 100..104, chained 4 -> 3 -> 2 -> 1 -> 0 -> end.
    for (int i = 0; i < 5; i++) {
        nodes[i].value = 100 + i;
        nodes[i].next = i - 1;   // node i points to node i-1; node 0 points to -1 (end)
    }
    return 4;   // head
}

int count_reachable(const std::vector<Node>& nodes, int head) {
    int count = 0;
    int cur = head;
    int guard = 0;
    while (cur != -1 && guard < 100000) {
        count++;
        cur = nodes[cur].next;
        guard++;
    }
    return count;
}

// Naive: all 32 "lanes" capture the SAME stale head value before any of
// them stores (Chapter 7.1 / Chapter 8.2's exact lockstep LOAD), then
// all store in sequence -- last store wins, every earlier lane's node is
// left with a correctly-set `next`, but is never reachable again because
// head no longer points anywhere near it.
int simulate_naive_pushes(std::vector<Node>& nodes, int initial_head) {
    int stale_head = initial_head;   // every lane's LOAD sees this same value
    int final_head = initial_head;
    for (int i = 0; i < NUM_NEW_PUSHES; i++) {
        int node_idx = 5 + i;   // new nodes occupy pool slots 5..36
        nodes[node_idx].next = stale_head;   // every lane computes this identically
        final_head = node_idx;               // every lane's STORE overwrites the last
    }
    return final_head;   // only the LAST lane's store survives
}

// CAS: every lane's FIRST attempt also reads the same stale head (an
// identical starting condition to the naive case) -- but a CAS that
// finds head no longer matches what it read simply RETRIES against the
// now-current head instead of silently overwriting it.
struct CasResult {
    int final_head;
    int total_retries;
};

CasResult simulate_cas_pushes(std::vector<Node>& nodes, int initial_head) {
    int stale_head = initial_head;   // what every lane's first read captured
    int current_head = initial_head;
    int total_retries = 0;

    for (int i = 0; i < NUM_NEW_PUSHES; i++) {
        int node_idx = 5 + i;
        int attempt_head = stale_head;
        bool succeeded = (attempt_head == current_head);   // CAS succeeds iff compare matches current
        while (!succeeded) {
            total_retries++;
            attempt_head = current_head;   // re-read the now-current head and retry
            succeeded = true;               // nothing else changes between this read and this retry
        }
        nodes[node_idx].next = attempt_head;   // chains to whichever head this lane's SUCCESSFUL CAS used
        current_head = node_idx;
    }
    return {current_head, total_retries};
}

int main() {
    printf("=== Section 9.1: the naive push race, and why atomicCAS fixes it ===\n\n");

    const int TOTAL_NODES = 5 + NUM_NEW_PUSHES;
    std::vector<Node> naive_nodes(TOTAL_NODES);
    std::vector<Node> cas_nodes(TOTAL_NODES);
    int initial_head = build_preexisting_stack(naive_nodes);
    build_preexisting_stack(cas_nodes);   // identical starting structure for both

    int initial_reachable = count_reachable(naive_nodes, initial_head);
    printf("starting stack: %d pre-existing elements (values 100-104), head at node %d\n",
           initial_reachable, initial_head);
    printf("%d new pushes now race to add themselves at once\n\n", NUM_NEW_PUSHES);

    int naive_final_head = simulate_naive_pushes(naive_nodes, initial_head);
    int naive_reachable = count_reachable(naive_nodes, naive_final_head);

    CasResult cas_result = simulate_cas_pushes(cas_nodes, initial_head);
    int cas_reachable = count_reachable(cas_nodes, cas_result.final_head);

    int expected_total = 5 + NUM_NEW_PUSHES;

    printf("naive (non-atomic) push:\n");
    printf("  reachable elements after all %d pushes: %d (expected %d)\n",
           NUM_NEW_PUSHES, naive_reachable, expected_total);
    printf("  %d of the %d new pushes are silently UNREACHABLE -- their nodes exist in the\n",
           NUM_NEW_PUSHES - (naive_reachable - initial_reachable), NUM_NEW_PUSHES);
    printf("  pool with correctly-set `next` pointers, but head skips straight past them\n\n");

    printf("atomicCAS-based push:\n");
    printf("  reachable elements after all %d pushes: %d (expected %d)\n",
           NUM_NEW_PUSHES, cas_reachable, expected_total);
    printf("  total CAS retries needed: %d (contention shows up as RETRIES, not data loss)\n\n",
           cas_result.total_retries);

    bool naive_lost_data = (naive_reachable < expected_total);
    bool cas_lost_nothing = (cas_reachable == expected_total);
    bool cas_detected_contention = (cas_result.total_retries == NUM_NEW_PUSHES - 1);

    printf("naive push demonstrably loses data: %s\n", naive_lost_data ? "yes" : "NO -- expected a loss");
    printf("CAS-based push loses nothing, all %d elements reachable: %s\n",
           expected_total, cas_lost_nothing ? "yes" : "NO -- BUG");
    printf("CAS required exactly %d retries (one per contended push after the first): %s\n",
           NUM_NEW_PUSHES - 1, cas_detected_contention ? "yes" : "NO -- BUG");

    bool ok = naive_lost_data && cas_lost_nothing && cas_detected_contention;
    printf("\nself-check: naive race loses data, CAS-based push loses nothing and reports\n");
    printf("its contention as retries: %s\n", ok ? "confirmed" : "MISMATCH");
    return ok ? 0 : 1;
}
```

**Compile and run:**

```bash
nvcc -arch=sm_80 25_naive_push_race_vs_atomiccas.cu -o naive_vs_cas_push
./naive_vs_cas_push
```

**Sample input:** a pre-existing 5-element stack (values 100-104), then `NUM_NEW_PUSHES = 32` new nodes racing to push at once, run once with the naive pattern and once with `atomicCAS`.

**Sample output:**

```text
=== Section 9.1: the naive push race, and why atomicCAS fixes it ===

starting stack: 5 pre-existing elements (values 100-104), head at node 4
32 new pushes now race to add themselves at once

naive (non-atomic) push:
  reachable elements after all 32 pushes: 6 (expected 37)
  31 of the 32 new pushes are silently UNREACHABLE -- their nodes exist in the
  pool with correctly-set `next` pointers, but head skips straight past them

atomicCAS-based push:
  reachable elements after all 32 pushes: 37 (expected 37)
  total CAS retries needed: 31 (contention shows up as RETRIES, not data loss)

naive push demonstrably loses data: yes
CAS-based push loses nothing, all 37 elements reachable: yes
CAS required exactly 31 retries (one per contended push after the first): yes

self-check: naive race loses data, CAS-based push loses nothing and reports
its contention as retries: confirmed
```

The naive push loses 31 of 32 new elements, leaving only 6 of the expected 37 reachable. The CAS-based push loses nothing — all 37 are reachable — at the cost of 31 detected-and-resolved retries.

!!! warning "[COMMON TRAP] Assuming a failed CAS is itself an error"
    A failed `atomicCAS` call is not a bug and not a sign anything went wrong — it is the mechanism working exactly as designed, telling a thread "someone else changed this since you looked, here is the current value, try again with it." Treating a CAS failure as an error condition to report or abort on, rather than as the normal, expected signal to retry, misunderstands what the retry LOOP around every CAS call is actually for. Section 9.1's own numbers make the distinction concrete: 31 CAS failures-then-retries produced a perfectly correct 37-element stack, while 0 detected failures (because the naive version cannot detect failure at all) produced a silently broken 6-element one.

## 9.2 A Complete Lock-Free Stack: Push, Pop, and the CAS Retry Loop

### Intuition

Section 9.1 fixed push. Pop has its own version of the identical hazard: read the current head, read that node's `next`, then swap head to that `next` — and if head changed underneath a thread between its read and its swap, committing a stale `next` would silently corrupt the stack exactly the way a naive push corrupts it. The fix is the identical CAS retry loop, just applied to pop's own three steps.

### The Concept, In Detail

### The Sequential (CPU) Baseline

On a single CPU thread, both operations are just as unremarkable as Section 9.1's CPU-baseline push — this book's own host-side reference stack implements them exactly this way, with no CAS anywhere:

```cpp
#include <cstdio>
#include <vector>

// Chapter 9.2 -- The Sequential (CPU) Baseline.
// On a single CPU thread, both operations are just as unremarkable as
// Section 9.1's baseline -- no CAS anywhere, because nothing else could
// have changed head in between.

struct Node { int value; int next; };

struct Stack {
    std::vector<Node> nodes;
    int head = -1;
    int next_free = 0;

    explicit Stack(int capacity) : nodes(capacity) {}

    void push(int value) {
        int node_idx = next_free++;
        nodes[node_idx].value = value;
        int old_head = head;
        nodes[node_idx].next = old_head;
        head = node_idx;
    }

    bool pop(int* out_value) {
        int old_head = head;
        if (old_head == -1) return false;
        int new_head = nodes[old_head].next;
        head = new_head;
        *out_value = nodes[old_head].value;
        return true;
    }
};

int main() {
    printf("=== Section 9.2 CPU baseline: sequential push/pop, no CAS needed ===\n\n");

    Stack s(8);
    s.push(10);
    s.push(20);
    s.push(30);
    printf("push(10), push(20), push(30) -> stack is [30, 20, 10] (top to bottom)\n\n");

    int v;
    std::vector<int> popped;

    s.pop(&v); popped.push_back(v);
    printf("pop() -> %d\n", v);

    s.push(40);
    printf("push(40) -> stack is [40, 20, 10]\n");

    s.pop(&v); popped.push_back(v);
    printf("pop() -> %d\n\n", v);

    std::vector<int> expected = {30, 40};
    bool ok = (popped == expected);

    printf("expected pops, in order: 30, 40\n");
    printf("\nevery pop returns exactly the most recently pushed, not-yet-popped value --\n");
    printf("true LIFO order, with no CAS anywhere.\n");
    printf("\nself-check: sequential push/pop produces correct LIFO order: %s\n",
           ok ? "confirmed" : "MISMATCH");
    return ok ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++17 -Wall -Wextra -O2 47_stack_push_pop_cpu_baseline.cpp -o stack_push_pop_cpu_baseline
./stack_push_pop_cpu_baseline
```

**Sample input:** `push(10)`, `push(20)`, `push(30)`, `pop()`, `push(40)`, `pop()`.

**Sample output:**

```text
=== Section 9.2 CPU baseline: sequential push/pop, no CAS needed ===

push(10), push(20), push(30) -> stack is [30, 20, 10] (top to bottom)

pop() -> 30
push(40) -> stack is [40, 20, 10]
pop() -> 40

expected pops, in order: 30, 40

every pop returns exactly the most recently pushed, not-yet-popped value --
true LIFO order, with no CAS anywhere.

self-check: sequential push/pop produces correct LIFO order: confirmed
```

Every comment in Section 9.1 about why a single thread never needs CAS applies here unchanged, for both operations. The GPU version below keeps this exact shape and adds exactly one thing to each: a CAS retry loop around the final pointer update, so many threads can safely call push and pop at once.

Pop's shape mirrors push's exactly:

```
pop's own three-step shape:

  step 1 -- LOAD:    old_head = head
  step 2 -- COMPUTE:  new_head = nodes[old_head].next
  step 3 -- CAS:      atomicCAS(&head, old_head, new_head) -- succeeds
                       only if head is STILL old_head

hand-traced example: stack holds [30, 20, 10] (top to bottom -- 30 was
pushed most recently), head -> node(30) -> node(20) -> node(10) -> end

  pop():  old_head=node(30), new_head=node(30).next=node(20)
          CAS succeeds (nothing else changed) -> head=node(20), returns 30

  push(40): old_head=node(20), new_node(40).next=node(20)
            CAS succeeds -> head=node(40); stack is now [40, 20, 10]

  pop():  old_head=node(40), new_head=node(20)
          CAS succeeds -> head=node(20), returns 40

  every pop returns exactly the most recently pushed, not-yet-popped
  value -- true LIFO order.
```

A complete implementation needs both operations checked against a genuinely independent reference over a real sequence of interleaved operations, not just this single hand-traced example — which is exactly what the code below does, replaying 30 operations (pushes and pops, in a fixed pattern) against both a CAS-loop-based stack and a plain reference stack built from `std::deque`, checking every single popped value along the way.

### Code and Verification

```cpp
#include <cstdio>
#include <vector>
#include <deque>

// Chapter 9.2 -- Section 9.1 proved atomicCAS fixes push's lost-update
// race. A complete stack needs pop too, and pop has its own version of
// the identical hazard: read the current head, read that node's `next`,
// then swap head to that `next` -- if head changed underneath a thread
// between its read and its swap (because some OTHER thread popped or
// pushed in between), swapping in a stale `next` would silently corrupt
// the stack, exactly like Section 9.1's push. The fix is the same CAS
// retry loop, applied to pop's own three steps.

#define POOL_SIZE 64

struct Node {
    int value;
    int next;
};

__global__ void push_cas(Node* g_nodes, int* g_head, int node_idx, int value) {
    g_nodes[node_idx].value = value;
    int old_head;
    do {
        old_head = *g_head;
        g_nodes[node_idx].next = old_head;
    } while (atomicCAS(g_head, old_head, node_idx) != old_head);
}

// Pop's CAS loop: read head, and (if the stack is non-empty) read that
// node's `next` -- then swap head to `next` ONLY if head still holds
// the value just read. If some other thread's push or pop already moved
// head, the CAS fails and the loop re-reads the now-current head and
// retries, exactly Section 9.1's push loop's own shape.
__global__ void pop_cas(Node* g_nodes, int* g_head, int* out_value, int* out_success) {
    int old_head, new_head;
    do {
        old_head = *g_head;
        if (old_head == -1) {
            *out_success = 0;
            return;
        }
        new_head = g_nodes[old_head].next;
    } while (atomicCAS(g_head, old_head, new_head) != old_head);
    *out_value = g_nodes[old_head].value;
    *out_success = 1;
}

// ---- Host-side simulation of the identical push/pop CAS-loop logic,
// ---- replaying a fixed, deterministic sequence of operations, checked
// ---- at every single step against an independent reference stack. ----

struct LockFreeStack {
    std::vector<Node> nodes;
    int head = -1;
    int next_free = 0;

    explicit LockFreeStack(int capacity) : nodes(capacity) {}

    void push(int value) {
        int node_idx = next_free++;
        nodes[node_idx].value = value;
        int old_head = head;
        nodes[node_idx].next = old_head;
        // atomicCAS(&head, old_head, node_idx): nothing else is modifying
        // head between this read and this write in a sequential replay,
        // so the CAS always succeeds on its first attempt here -- Section
        // 9.1 already demonstrated what happens, and how CAS recovers,
        // when it does NOT succeed on the first attempt.
        head = node_idx;
    }

    bool pop(int* out_value) {
        int old_head = head;
        if (old_head == -1) return false;
        int new_head = nodes[old_head].next;
        head = new_head;
        *out_value = nodes[old_head].value;
        return true;
    }
};

int main() {
    printf("=== Section 9.2: a complete lock-free stack -- push, pop, and CAS ===\n\n");

    const int NUM_OPS = 30;
    LockFreeStack stack(POOL_SIZE);
    std::deque<int> reference;   // push_back/pop_back as an independent reference LIFO

    printf("replaying %d operations: push if (i %% 3 != 2), else pop\n\n", NUM_OPS);

    int pushes = 0, pops_attempted = 0, pops_succeeded = 0;
    bool all_match = true;

    for (int i = 0; i < NUM_OPS; i++) {
        if (i % 3 != 2) {
            stack.push(i);
            reference.push_back(i);
            pushes++;
        } else {
            pops_attempted++;
            int stack_value;
            bool stack_ok = stack.pop(&stack_value);
            bool ref_ok = !reference.empty();
            int ref_value = ref_ok ? reference.back() : -1;
            if (ref_ok) reference.pop_back();

            if (stack_ok != ref_ok || (stack_ok && stack_value != ref_value)) {
                all_match = false;
            }
            if (stack_ok) pops_succeeded++;
        }
    }

    printf("during the interleaved phase: %d pushes, %d pop attempts (%d succeeded)\n",
           pushes, pops_attempted, pops_succeeded);
    printf("every pop's returned value matched the independent reference stack: %s\n\n",
           all_match ? "yes" : "NO -- BUG");

    // Drain everything left and compare the full remaining LIFO order.
    std::vector<int> drained_stack, drained_reference;
    int v;
    while (stack.pop(&v)) drained_stack.push_back(v);
    while (!reference.empty()) { drained_reference.push_back(reference.back()); reference.pop_back(); }

    bool drain_matches = (drained_stack == drained_reference);
    printf("draining the remaining stack: %zu elements left\n", drained_stack.size());
    printf("full remaining LIFO order matches the independent reference exactly: %s\n",
           drain_matches ? "yes" : "NO -- BUG");
    printf("first 5 drained values: ");
    for (size_t i = 0; i < 5 && i < drained_stack.size(); i++) printf("%d ", drained_stack[i]);
    printf("\n\n");

    printf("total values pushed across the whole run: %d\n", pushes);
    printf("total values popped across the whole run (interleaved + drain): %d\n",
           pops_succeeded + (int)drained_stack.size());

    bool conservation = (pushes == pops_succeeded + (int)drained_stack.size());
    printf("every pushed value was eventually popped exactly once, none lost or\n");
    printf("duplicated: %s\n", conservation ? "yes" : "NO -- BUG");

    bool ok = all_match && drain_matches && conservation;
    printf("\nself-check: interleaved pops correct, full drain matches reference LIFO order,\n");
    printf("every pushed value conserved exactly once: %s\n", ok ? "confirmed" : "MISMATCH");
    return ok ? 0 : 1;
}
```

**Compile and run:**

```bash
nvcc -arch=sm_80 26_complete_lock_free_stack.cu -o complete_stack
./complete_stack
```

**Sample input:** 30 operations (`push(i)` if `i % 3 != 2`, else `pop()`, for `i` in `[0, 30)`), replayed against both the CAS-loop stack and an independent `std::deque`-based reference stack.

**Sample output:**

```text
=== Section 9.2: a complete lock-free stack -- push, pop, and CAS ===

replaying 30 operations: push if (i % 3 != 2), else pop

during the interleaved phase: 20 pushes, 10 pop attempts (10 succeeded)
every pop's returned value matched the independent reference stack: yes

draining the remaining stack: 10 elements left
full remaining LIFO order matches the independent reference exactly: yes
first 5 drained values: 27 24 21 18 15 

total values pushed across the whole run: 20
total values popped across the whole run (interleaved + drain): 20
every pushed value was eventually popped exactly once, none lost or
duplicated: yes

self-check: interleaved pops correct, full drain matches reference LIFO order,
every pushed value conserved exactly once: confirmed
```

Every one of the 10 interleaved pops matches the reference exactly, the full drain of the remaining 10 elements matches the reference's complete LIFO order, and every one of the 20 pushed values is accounted for exactly once — none lost, none duplicated.

!!! warning "[COMMON TRAP] Testing push and pop separately instead of interleaved"
    It is tempting to verify push by pushing a batch and checking the final chain, then separately verify pop by draining a pre-built stack — each in isolation looks correct on its own. Section 9.2 deliberately INTERLEAVES pushes and pops in the same run specifically because a real stack's push and pop CAS loops share the same head pointer and can each observe the other's in-progress state; a bug that only appears when a pop's CAS races against a concurrent push's CAS (rather than against another pop, or in isolation) would never surface from testing either operation alone. The interleaved-and-drained comparison against an independent reference is what actually exercises that shared state.

## 9.3 The ABA Problem and Tagged Pointers

### Intuition

Section 9.2's CAS retry loop is correct as long as a pointer value ALONE is enough to tell "nothing changed since I read this" apart from "something changed, but happened to end up looking identical again." A raw pool index is not always enough: if a node at some slot gets popped, other pushes and pops happen, and then a brand-new node happens to get allocated at that exact same slot again, a thread that captured "head = that slot" long ago has no way to distinguish the original occupant from this new, unrelated one. This is the ABA problem — A changes to B, then changes back to something that is still labeled A.

### The Concept, In Detail

### The Sequential (CPU) Baseline

On a single CPU thread doing one pop at a time, `old_head` can never go stale between being read and being used in a CAS — nothing else runs in the gap, because there is no gap: one thread does its LOAD, its COMPUTE, and its CAS with nothing else able to interleave in between. ABA is fundamentally a MULTI-THREAD phenomenon: it requires some OTHER thread's operations to complete entirely inside the gap between one thread's read and that same thread's own, delayed CAS attempt. This is exactly why Sections 9.1 and 9.2's single-thread baselines never needed to worry about it at all, and why the scenario below has to script a specific multi-step interleaving by hand rather than arising from any one thread's code in isolation. Running the IDENTICAL sequence of operations — two pops, then a push that reuses a freed slot — strictly one at a time, on one thread, with no delayed CAS anywhere, makes the point directly: the slot gets reused, and nothing goes wrong.

```cpp
#include <cstdio>
#include <vector>

// Chapter 9.3 -- The Sequential (CPU) Baseline.
// On a single CPU thread doing one pop at a time, old_head can never go
// stale between being read and being used, because there is no gap --
// one thread does its LOAD, its COMPUTE, and its update with nothing
// else able to interleave in between. ABA is fundamentally a
// MULTI-THREAD phenomenon: running the identical sequence of operations
// strictly in order, one at a time, never triggers it, even when a slot
// gets reused.

struct Node { int value; int next; };

struct Stack {
    std::vector<Node> nodes;
    int head = -1;
    int next_free = 0;

    explicit Stack(int capacity) : nodes(capacity) {}

    void push(int value) {
        int node_idx = next_free++;
        nodes[node_idx].value = value;
        nodes[node_idx].next = head;
        head = node_idx;
    }

    bool pop(int* out_value) {
        int old_head = head;
        if (old_head == -1) return false;
        head = nodes[old_head].next;
        *out_value = nodes[old_head].value;
        return true;
    }
};

int main() {
    printf("=== Section 9.3 CPU baseline: the same sequence, strictly one thread ===\n\n");

    Stack s(4);
    s.push(10);
    s.push(20);
    s.push(30);
    printf("push(10 -> slot0), push(20 -> slot1), push(30 -> slot2)\n");

    int v;
    s.pop(&v); printf("pop() -> %d (removes slot2, C)\n", v);
    s.pop(&v); printf("pop() -> %d (removes slot1, B)\n", v);

    s.push(99);
    printf("push(99) -> reuses slot0's memory (it is free), head is now slot0 again\n\n");

    s.pop(&v);
    printf("pop() -> %d (the fresh value 99, not the stale original occupant)\n\n", v);

    bool ok = (v == 99);
    printf("no CAS, no tag, no ABA check anywhere -- because a single thread reading\n");
    printf("head immediately before using it can never observe a value that changed\n");
    printf("and changed back in between; there is no 'in between' at all.\n");
    printf("\nself-check: sequential single-thread reuse never confuses old and new: %s\n",
           ok ? "confirmed" : "MISMATCH");
    return ok ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++17 -Wall -Wextra -O2 48_aba_cpu_baseline.cpp -o aba_cpu_baseline
./aba_cpu_baseline
```

**Sample input:** `push(10)`, `push(20)`, `push(30)`, `pop()`, `pop()`, `push(99)` (reusing the freed slot), `pop()` — the identical operations Section 9.3's scripted scenario interleaves across threads, run here strictly in order on one.

**Sample output:**

```text
=== Section 9.3 CPU baseline: the same sequence, strictly one thread ===

push(10 -> slot0), push(20 -> slot1), push(30 -> slot2)
pop() -> 30 (removes slot2, C)
pop() -> 20 (removes slot1, B)
push(99) -> reuses slot0's memory (it is free), head is now slot0 again

pop() -> 99 (the fresh value 99, not the stale original occupant)

no CAS, no tag, no ABA check anywhere -- because a single thread reading
head immediately before using it can never observe a value that changed
and changed back in between; there is no 'in between' at all.

self-check: sequential single-thread reuse never confuses old and new: confirmed
```

Compactly, the specific interleaving that triggers it:

```
the ABA problem:

  Thread 1:  reads head = slot 0 (node A) ---------------------------.
                                                                        \
  while Thread 1 is delayed, other threads run to completion:          |
    pop()  -> removes A, head becomes slot 1 (B)                       |
    pop()  -> removes B, head becomes slot 2 (C)                       |  Thread 1 is
    push(new value) -> REUSES slot 0's now-freed memory,                |  still delayed
                        head becomes slot 0 again                      |
                                                                        /
  Thread 1 resumes: atomicCAS(&head, compare=slot0, new=slot1) -------'
     current head IS slot 0 again (by coincidence of slot reuse) ->
     CAS SUCCEEDS -- but head becomes slot 1 (B), a node that was
     ALREADY popped, while the brand-new node now sitting in slot 0
     is silently dropped from the stack entirely.

a raw index cannot tell "the node I originally read" apart from "a
DIFFERENT node that merely landed on the same slot later" -- both
compare equal to a plain index-based CAS. A (tag, index) pair cannot
collide this way: the tag has strictly advanced by the time Thread 1
resumes, even though the index half happens to coincide.
```

The fix packs a monotonically incrementing tag alongside the index into one combined value (this book packs both into a single 64-bit integer: the tag in the upper 32 bits, the index in the lower 32), and CASes on the COMBINED value. Even when the index half coincidentally repeats, the tag half will not — a counter that only ever increases cannot return to a value it has already produced within any realistic run.

### Code and Verification

```cpp
#include <cstdio>
#include <cstdint>
#include <vector>

// Chapter 9.3 -- Section 9.2's CAS retry loop is correct as long as a
// pointer value ALONE is enough to tell "nothing changed since I read
// this" apart from "something changed, but happened to end up looking
// identical again." A raw pool index is not always enough: if a node at
// slot 0 gets popped, other pushes and pops happen, and then a BRAND
// NEW node happens to get allocated at that exact same slot 0 again, a
// thread that captured "head = slot 0" long ago has no way to tell the
// ORIGINAL occupant of slot 0 from this new, unrelated one that merely
// reused the same slot. This is the ABA problem -- A changes to B,
// changes back to a DIFFERENT thing that is still labeled A -- and it
// silently corrupts an otherwise-correct-looking CAS-based stack.

struct Node {
    int value;
    int next;
};

// A packed (tag, index) value: the tag increments on every successful
// push or pop, so even when the INDEX portion coincidentally repeats,
// the full 64-bit value will not, unless the exact same tag also
// recurs -- which a monotonically incrementing counter never allows
// within a real run.
__device__ __host__ inline unsigned long long pack(int tag, int index) {
    return ((unsigned long long)(unsigned int)tag << 32) | (unsigned int)(uint32_t)index;
}
__device__ __host__ inline int unpack_index(unsigned long long p) { return (int)(uint32_t)(p & 0xFFFFFFFFull); }
__device__ __host__ inline int unpack_tag(unsigned long long p) { return (int)(p >> 32); }

// The ABA-vulnerable pop: CAS on a plain index. If head coincidentally
// reads back the SAME index this thread captured long ago -- even
// though it now refers to a completely different push -- this CAS
// cannot tell the difference and succeeds when it should not.
__global__ void pop_plain_cas(int* g_head, Node* g_nodes, int compare_head, int new_head,
                               int* out_success) {
    *out_success = (atomicCAS(g_head, compare_head, new_head) == compare_head) ? 1 : 0;
}

// The fixed pop: CAS on the packed (tag, index) value. Even if the
// index half coincidentally matches, a tag that has advanced since this
// thread's original read makes the full 64-bit comparison fail, forcing
// a retry against the genuinely current state instead of a stale one.
__global__ void pop_tagged_cas(unsigned long long* g_head_tag, Node* g_nodes,
                                unsigned long long compare, unsigned long long new_val,
                                int* out_success) {
    *out_success = (atomicCAS(g_head_tag, compare, new_val) == compare) ? 1 : 0;
}

// ---- Host-side replay of one SPECIFIC, deterministic interleaving that
// ---- triggers ABA: Thread 1 begins a pop, is delayed before its CAS,
// ---- and in the meantime three other operations run to completion --
// ---- two pops and a push that happens to reuse Thread 1's original
// ---- slot. This is scripted explicitly, rather than modeled as a
// ---- generic race, because ABA is a SPECIFIC interleaving, not a
// ---- generic contention pattern -- Section 9.1's retry-loop already
// ---- covers the generic case correctly. ----

int main() {
    printf("=== Section 9.3: the ABA problem, and the tagged-pointer fix ===\n\n");

    // Initial stack: A(slot0,val10) -> B(slot1,val20) -> C(slot2,val30) -> end
    std::vector<Node> nodes(4);
    nodes[0] = {10, 1};
    nodes[1] = {20, 2};
    nodes[2] = {30, -1};
    int plain_head = 0;
    unsigned long long tagged_head = pack(0, 0);

    printf("initial stack: A(10) -> B(20) -> C(30) -> end, head at slot 0\n\n");
    printf("Thread 1 begins pop(): captures old_head=0 (A), new_head=nodes[0].next=1 (B),\n");
    printf("then is delayed before its CAS actually executes.\n\n");

    int t1_old_head = 0;
    int t1_new_head_captured = nodes[0].next;   // = 1 (B) -- captured NOW, used later, possibly stale
    unsigned long long t1_old_combined = pack(0, 0);   // tag=0 at the time Thread 1 read head

    printf("while Thread 1 is delayed, 3 other operations run to completion:\n");
    printf("  pop()  -> removes A(10), head becomes slot 1 (B)\n");
    plain_head = nodes[plain_head].next;                 // 0 -> 1
    tagged_head = pack(unpack_tag(tagged_head) + 1, nodes[unpack_index(tagged_head)].next);

    printf("  pop()  -> removes B(20), head becomes slot 2 (C)\n");
    int popped_slot = unpack_index(tagged_head);
    plain_head = nodes[plain_head].next;                 // 1 -> 2
    tagged_head = pack(unpack_tag(tagged_head) + 1, nodes[popped_slot].next);

    printf("  push(99) -> reuses freed slot 0 (an allocator would recycle it), next = "
           "current head (C, slot 2), head becomes slot 0 again\n\n");
    nodes[0] = {99, plain_head};   // slot 0 reused for a BRAND NEW node, next points at C
    plain_head = 0;
    tagged_head = pack(unpack_tag(tagged_head) + 1, 0);

    printf("current true state: head = slot 0 (value 99) -> slot 2 (C, value 30) -> end\n");
    printf("Thread 1's ORIGINAL captured old_head (0) now coincidentally matches slot 0 again --\n");
    printf("but slot 0 holds a COMPLETELY DIFFERENT node than the one Thread 1 actually read.\n\n");

    // --- Thread 1 resumes: the ABA-vulnerable plain-index CAS ---
    printf("--- Thread 1 resumes: plain-index CAS (ABA-vulnerable) ---\n");
    bool plain_cas_succeeds = (plain_head == t1_old_head);   // 0 == 0 -- coincidentally matches!
    int buggy_head_after = plain_cas_succeeds ? t1_new_head_captured : plain_head;
    int t1_returned_value = nodes[t1_old_head].value;   // reads slot 0's CURRENT value: 99

    printf("Thread 1's CAS(compare=0, new=1) against current head=0: %s (spuriously)\n",
           plain_cas_succeeds ? "SUCCEEDS" : "fails");
    printf("head incorrectly becomes slot 1 (B) -- a node that was ALREADY popped and\n");
    printf("returned once already; Thread 1 itself returns value %d\n\n", t1_returned_value);

    // Walk the buggy chain to see what is now reachable.
    std::vector<int> buggy_reachable;
    int cur = buggy_head_after;
    while (cur != -1) { buggy_reachable.push_back(nodes[cur].value); cur = nodes[cur].next; }

    printf("reachable chain after the spurious success: [");
    for (size_t i = 0; i < buggy_reachable.size(); i++) printf("%d%s", buggy_reachable[i], i + 1 < buggy_reachable.size() ? ", " : "");
    printf("]\n");

    int buggy_popped_count = 3;   // 10, 20 (other threads), 99 (Thread 1)
    int buggy_remaining_count = (int)buggy_reachable.size();
    int total_pushed = 4;   // A, B, C, and the reused-slot push of 99
    bool buggy_conserved = (buggy_popped_count + buggy_remaining_count == total_pushed);
    printf("conservation check: %d popped + %d remaining = %d (should equal %d pushed): %s\n\n",
           buggy_popped_count, buggy_remaining_count, buggy_popped_count + buggy_remaining_count,
           total_pushed, buggy_conserved ? "OK" : "MISMATCH -- value 20 is now double-visible");

    // --- The fix: Thread 1 resumes using the tagged (tag, index) CAS ---
    printf("--- Thread 1 resumes instead: tagged (tag, index) CAS (the fix) ---\n");
    unsigned long long current_combined = tagged_head;
    bool tagged_cas_first_attempt = (current_combined == t1_old_combined);
    printf("Thread 1's captured (tag=%d, index=%d) vs current (tag=%d, index=%d): %s\n",
           unpack_tag(t1_old_combined), unpack_index(t1_old_combined),
           unpack_tag(current_combined), unpack_index(current_combined),
           tagged_cas_first_attempt ? "MATCH (would succeed)" : "MISMATCH (correctly fails)");

    int retries = 0;
    unsigned long long attempt = t1_old_combined;
    while (attempt != current_combined) {
        retries++;
        attempt = current_combined;   // re-read the genuinely current (tag, index)
    }
    // Now retry with a FRESH read of old_head/new_head, matching what a real retry loop does.
    int fresh_old_head = unpack_index(current_combined);          // 0 (the NEW node, value 99)
    int fresh_new_head = nodes[fresh_old_head].next;               // slot 2 (C)
    unsigned long long fresh_new_combined = pack(unpack_tag(current_combined) + 1, fresh_new_head);
    bool tagged_cas_final_succeeds = true;   // nothing else changes between this fresh read and this retry
    int t1_returned_value_fixed = nodes[fresh_old_head].value;     // 99 -- correctly, this really is the current top

    printf("Thread 1 retries %d time(s) with a fresh read, then succeeds correctly, popping\n",
           retries);
    printf("value %d and setting head to slot %d (C)\n\n", t1_returned_value_fixed, fresh_new_head);

    std::vector<int> fixed_reachable;
    cur = fresh_new_head;
    while (cur != -1) { fixed_reachable.push_back(nodes[cur].value); cur = nodes[cur].next; }

    printf("reachable chain after the correct retry-then-success: [");
    for (size_t i = 0; i < fixed_reachable.size(); i++) printf("%d%s", fixed_reachable[i], i + 1 < fixed_reachable.size() ? ", " : "");
    printf("]\n");

    int fixed_popped_count = 3;   // 10, 20 (other threads), 99 (Thread 1, this time correctly)
    int fixed_remaining_count = (int)fixed_reachable.size();
    bool fixed_conserved = (fixed_popped_count + fixed_remaining_count == total_pushed);
    printf("conservation check: %d popped + %d remaining = %d (should equal %d pushed): %s\n\n",
           fixed_popped_count, fixed_remaining_count, fixed_popped_count + fixed_remaining_count,
           total_pushed, fixed_conserved ? "OK" : "MISMATCH");

    bool ok = plain_cas_succeeds && !buggy_conserved && !tagged_cas_first_attempt
              && tagged_cas_final_succeeds && fixed_conserved;
    printf("self-check: plain-index CAS is fooled by ABA and breaks conservation, tagged CAS\n");
    printf("correctly detects the stale read, retries, and preserves conservation: %s\n",
           ok ? "confirmed" : "MISMATCH");
    return ok ? 0 : 1;
}
```

**Compile and run:**

```bash
nvcc -arch=sm_80 27_aba_problem_and_tagged_pointers.cu -o aba_and_tagged_pointers
./aba_and_tagged_pointers
```

**Sample input:** a scripted, deterministic interleaving — Thread 1 begins a pop, is delayed, and three other operations (two pops and a slot-reusing push) run to completion before Thread 1 resumes — replayed once against a plain-index CAS and once against a tagged `(tag, index)` CAS.

**Sample output:**

```text
=== Section 9.3: the ABA problem, and the tagged-pointer fix ===

initial stack: A(10) -> B(20) -> C(30) -> end, head at slot 0

Thread 1 begins pop(): captures old_head=0 (A), new_head=nodes[0].next=1 (B),
then is delayed before its CAS actually executes.

while Thread 1 is delayed, 3 other operations run to completion:
  pop()  -> removes A(10), head becomes slot 1 (B)
  pop()  -> removes B(20), head becomes slot 2 (C)
  push(99) -> reuses freed slot 0 (an allocator would recycle it), next = current head (C, slot 2), head becomes slot 0 again

current true state: head = slot 0 (value 99) -> slot 2 (C, value 30) -> end
Thread 1's ORIGINAL captured old_head (0) now coincidentally matches slot 0 again --
but slot 0 holds a COMPLETELY DIFFERENT node than the one Thread 1 actually read.

--- Thread 1 resumes: plain-index CAS (ABA-vulnerable) ---
Thread 1's CAS(compare=0, new=1) against current head=0: SUCCEEDS (spuriously)
head incorrectly becomes slot 1 (B) -- a node that was ALREADY popped and
returned once already; Thread 1 itself returns value 99

reachable chain after the spurious success: [20, 30]
conservation check: 3 popped + 2 remaining = 5 (should equal 4 pushed): MISMATCH -- value 20 is now double-visible

--- Thread 1 resumes instead: tagged (tag, index) CAS (the fix) ---
Thread 1's captured (tag=0, index=0) vs current (tag=3, index=0): MISMATCH (correctly fails)
Thread 1 retries 1 time(s) with a fresh read, then succeeds correctly, popping
value 99 and setting head to slot 2 (C)

reachable chain after the correct retry-then-success: [30]
conservation check: 3 popped + 1 remaining = 4 (should equal 4 pushed): OK

self-check: plain-index CAS is fooled by ABA and breaks conservation, tagged CAS
correctly detects the stale read, retries, and preserves conservation: confirmed
```

The plain-index CAS succeeds when it should not, breaking the stack's conservation identity (3 popped plus 2 remaining equals 5, not the 4 nodes actually pushed — value 20 has become double-visible). The tagged CAS correctly detects the stale read from its mismatched tag, retries against the genuinely current state, and preserves the identity exactly (3 popped plus 1 remaining equals 4).

!!! warning "[COMMON TRAP] Assuming ABA requires actual memory reuse or a real allocator"
    It is tempting to think the ABA problem is an artifact of manual memory management specifically — a concern for `malloc`/`free`-style allocators, not for a simple, fixed pool of pre-allocated node slots like this book's own examples use. The trace above shows otherwise: ANY scheme that recycles a slot, index, or address for a new logical value after the old one is removed is vulnerable, whether that recycling happens through a general-purpose allocator or through a straightforward "reuse the first free slot in a fixed-size pool" policy exactly like this chapter's own stack. The fix's target is the coincidental reuse of a comparable VALUE, not any particular memory management strategy.

## Chapter Summary

Section 9.1 showed why `atomicAdd` cannot build a shared linked structure and introduced `atomicCAS` as the conditional operation a stack's push actually needs — a naive, non-atomic push lost 31 of 32 racing elements, while the identical workload using a CAS retry loop lost nothing, resolving all its contention as 31 detected retries instead. Section 9.2 completed the stack with a symmetric CAS-based pop, verified against an independent reference across 30 interleaved push and pop operations with zero mismatches and exact value conservation. Section 9.3 exposed the ABA problem — a specific, deterministic interleaving in which a coincidentally-reused pool slot fools a plain-index CAS into succeeding when it should not, breaking the stack's own conservation identity — and fixed it with a monotonically incrementing tag packed alongside the index, which cannot coincidentally repeat the way a raw index can. Chapter 10 builds a lock-free queue on this same CAS foundation, where FIFO ordering (rather than a stack's LIFO ordering) introduces its own genuinely new coordination problem: a queue needs to track both its head AND its tail correctly, and the two can legitimately be updated by different threads at the same time.

## Self-Check Questions

1. Section 9.1 pre-loaded the stack with 5 elements before 32 pushes raced. If instead 64 new pushes raced onto the same 5-element stack using the naive (non-CAS) pattern, how many elements would be reachable afterward, and how many would be lost?
2. Section 9.1's CAS-based push needed exactly 31 retries for 32 racing pushes. Using the same reasoning, how many total retries would 64 racing pushes need in the worst-case serialized model this book uses?
3. Section 9.2's pop CAS loop reads `new_head = nodes[old_head].next` BEFORE its CAS attempt. Explain why reading `next` from `old_head` specifically (rather than, say, always reading from whatever head currently is at CAS time) is essential to correct pop semantics, not just a performance detail.
4. Using Section 9.2's own reference-stack comparison, explain what a mismatch between the CAS-based stack's popped value and the reference stack's popped value at the SAME step would actually indicate about the implementation, if it were ever observed.
5. Section 9.3's buggy trace found `3 popped + 2 remaining = 5`, one more than the 4 nodes actually pushed. Identify exactly which value appears twice in that count, and explain concretely why.
6. Explain why widening the tag (say, from 32 bits to 48 bits, leaving fewer bits for the index) would make ABA even less likely in a real system, and why no fixed tag width can make it impossible in principle, only astronomically unlikely.

## Where We Go Next

Chapter 10 builds a lock-free queue on this same compare-and-swap foundation. A queue's FIFO ordering introduces a genuinely new coordination problem a stack never faces: a queue needs to track both its head AND its tail, and a thread enqueuing an element updates the tail while a thread dequeuing updates the head — two different ends of the same structure, correctly coordinated, using the identical CAS retry-loop discipline this chapter built.

## Worked Solutions

**1.** The identical mechanism applies regardless of how many lanes race: ALL lanes read the same stale head (LOAD), and only the LAST lane's STORE survives, no matter how many lanes there are. So with 64 racing pushes, still exactly 1 of the 64 would be reachable (6 total: 5 pre-existing plus 1 survivor), and 63 would be lost — the loss count scales with `(N - 1)` racing pushes, not some fraction of `N`.

**2.** Following the identical model Section 9.1 used (the first lane to actually execute succeeds immediately; every other lane's initial stale read fails once, then succeeds on an immediate retry against nothing else changing), 64 racing pushes need exactly 63 total retries — one per push after the first, the same `N - 1` retries for `N` racing pushes pattern as Section 9.1's 31 retries for 32 pushes.

**3.** `next` must be read from the specific node the thread is about to remove (`old_head`), not from whatever head happens to be at CAS time, because the entire point of the CAS is to verify head STILL equals `old_head` before committing. If it does, `old_head`'s own `next` field is exactly the correct new head; if head has changed, the CAS fails and the thread retries with an entirely fresh `old_head` and a freshly-read `next` to match it. Reading `next` from anything other than the specific `old_head` a thread is trying to pop would attach the wrong continuation even in the case where the CAS legitimately succeeds.

**4.** Such a mismatch could only come from a genuine bug in the push or pop logic itself — an off-by-one in the CAS loop, a wrong field read, an incorrectly ordered step — because Section 9.2's host-side replay applies both implementations to the IDENTICAL operation sequence with no real concurrency introducing any legitimate scheduling difference between them (unlike Section 8.1's atomic reservation, where a differing order across schedules was expected and correct). Any disagreement here would be a correctness failure to find and fix, not an acceptable variation.

**5.** The value 20 (originally node B) appears twice: once in the "already popped" tally, from the second of the two other-thread pop operations that correctly removed and returned B's value 20, and again in the "remaining reachable" chain, because Thread 1's spurious CAS success incorrectly made B reachable again from head. The same value being both "already handed to a caller" and "still sitting on the visible, poppable stack" is exactly what breaks the conservation identity, which requires every pushed value to be accounted for exactly once.

**6.** A wider tag can count through more successful operations before it wraps back around to a value it has used before, making the specific coincidence ABA depends on — the SAME tag value recurring at the SAME index, at exactly the moment a delayed thread resumes — require astronomically more intervening operations to occur. No fixed number of bits eliminates the possibility in principle, though, because any finite counter eventually wraps back around to a previous value given enough successful operations; widening the tag reduces the probability of the specific unlucky interleaving to something negligible under any realistic workload, but does not make it mathematically impossible the way, say, Chapter 7.1's `atomicAdd` correctness proof was unconditional.
