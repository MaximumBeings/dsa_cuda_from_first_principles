# Chapter 10: Lock-Free Queues

**What you will understand by the end of this chapter:**

- Why a FIFO queue needs a genuinely different design from Chapter 9's stack: enqueue's "add a node" is not one atomic step but two — link, then swing — and a thread delayed between them leaves the queue in a valid but lagging state.
- How other threads detect and complete a lagging tail themselves, instead of ever waiting for the delayed thread — the "helping" mechanism that makes the design lock-free.
- Why `head == tail` alone cannot tell a truly empty queue apart from one whose tail has simply not caught up yet, and what dequeue must check instead.

**What you need to know first:**

- Chapter 9 in full: `atomicCAS`, a complete CAS-based retry loop, and the ABA problem — all reused and extended in this chapter for a structure with two shared pointers instead of one.

---

Chapter 9's stack needed exactly one shared pointer and one CAS per operation. A queue enqueues at one end and dequeues at the other, and that second pointer is not a minor variation — it introduces a genuinely new kind of intermediate state no stack ever has: a node that is correctly, permanently part of the queue, while the shared pointer meant to point at it has simply not been updated yet. This chapter builds the standard design for handling that state correctly, the Michael-Scott lock-free queue, and the one subtle check dequeue needs that is easy to get wrong.

## 10.1 Why a Queue Needs Two Ends: Link-Then-Swing, and Helping a Lagging Tail

### Intuition

Chapter 9's push updated one pointer (`head`) with one CAS, and either it succeeded completely or it failed completely — there was no state in between. A queue's enqueue cannot work that way: attaching a new node to the end of a linked list is naturally two separate operations — point the current last node's `next` field at the new node, then move the shared `tail` pointer to point at it too — and a thread can genuinely be interrupted between them.

### The Sequential (CPU) Baseline

On a single CPU thread, link and swing might as well be one indivisible step:

```cpp
#include <cstdio>
#include <vector>

// Chapter 10.1 -- The Sequential (CPU) Baseline.
// On a single CPU thread, link and swing might as well be one
// indivisible step -- nothing else ever runs between these two lines,
// so no other code could ever observe the queue in a half-updated state.

struct Node { int value; int next; };

void enqueue_cpu(std::vector<Node>& nodes, int& tail, int new_idx) {
    nodes[tail].next = new_idx;   // link
    tail = new_idx;                // swing -- happens immediately after
}

int main() {
    printf("=== Section 10.1 CPU baseline: sequential enqueue, link+swing as one step ===\n\n");

    std::vector<Node> nodes(4);
    nodes[0] = {-1, -1};
    int head = 0, tail = 0, next_free = 1;

    int values[] = {10, 20, 30};
    for (int v : values) {
        int idx = next_free++;
        nodes[idx] = {v, -1};
        enqueue_cpu(nodes, tail, idx);
        printf("enqueue(%d) -> tail is now slot %d\n", v, tail);
    }

    printf("\nfinal queue, walked from head: [");
    std::vector<int> walked;
    int cur = nodes[head].next;
    while (cur != -1) { walked.push_back(nodes[cur].value); cur = nodes[cur].next; }
    for (size_t i = 0; i < walked.size(); i++) printf("%d%s", walked[i], i + 1 < walked.size() ? ", " : "");
    printf("]\n\n");

    std::vector<int> expected = {10, 20, 30};
    bool ok = (walked == expected);

    printf("expected FIFO order: 10, 20, 30\n");
    printf("\nno lagging tail is possible here -- nothing else ever runs between the link\n");
    printf("and the swing, with exactly one thread.\n");
    printf("\nself-check: sequential enqueue produces correct FIFO order: %s\n",
           ok ? "confirmed" : "MISMATCH");
    return ok ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++17 -Wall -Wextra -O2 49_enqueue_cpu_baseline.cpp -o enqueue_cpu_baseline
./enqueue_cpu_baseline
```

**Sample input:** `enqueue(10)`, `enqueue(20)`, `enqueue(30)` onto an initially empty queue, one thread at a time.

**Sample output:**

```text
=== Section 10.1 CPU baseline: sequential enqueue, link+swing as one step ===

enqueue(10) -> tail is now slot 1
enqueue(20) -> tail is now slot 2
enqueue(30) -> tail is now slot 3

final queue, walked from head: [10, 20, 30]

expected FIFO order: 10, 20, 30

no lagging tail is possible here -- nothing else ever runs between the link
and the swing, with exactly one thread.

self-check: sequential enqueue produces correct FIFO order: confirmed
```

Nothing else ever runs between these two lines, so no other code could ever observe the queue in a half-updated state where the link exists but the swing has not happened yet. This entire section's subject — a lagging tail, and another thread noticing and fixing it — is a problem that can only exist once multiple threads can genuinely interleave between those two lines, which is exactly what happens the moment this logic runs on a GPU.

### The Concept, In Detail

Enqueue is LINK, then SWING — two separate steps, not one:

```
before:  ... -> [tail node] -> nothing yet

LINK:    ... -> [tail node] -> [new node]     (nodes[tail].next set via CAS)

SWING:   tail pointer moves to [new node]      (tail itself set via CAS,
                                                  a SEPARATE operation)

a thread delayed BETWEEN these two steps leaves the queue in a perfectly
valid but LAGGING state: every node is correctly linked in order, tail
simply has not caught up to the true last one yet.
```

The key insight that makes this lock-free rather than merely "eventually consistent": any OTHER thread that next reads `tail` and finds `nodes[tail].next` already occupied recognizes the lag immediately, and instead of waiting for the delayed thread, it HELPS —

```
detect:  nodes[tail].next != empty   =>  tail is lagging, not "done"
help:    CAS(tail, old_tail, nodes[old_tail].next)   -- advance it
then:    retry this thread's OWN operation with the now-current tail
```

— and once a thread's own link has succeeded, its later swing attempt is allowed to fail harmlessly: if some other thread already helped, there is nothing left for it to do, because the enqueue's real work (the link) was already complete.

### Code and Verification

```cpp
#include <cstdio>
#include <vector>

// Chapter 10.1 -- Chapter 9's stack needed exactly one shared pointer
// (head) and exactly one CAS per operation. A FIFO queue enqueues at one
// end (the tail) and dequeues at the other (the head) -- two different
// pointers that most of the time can be updated independently, EXCEPT
// that adding a node to the tail is not one atomic step the way a
// stack's push is. It is two: first LINK the new node onto the current
// last node's `next` field, then SWING the shared `tail` pointer forward
// to point at it. Any thread that gets delayed between those two steps
// leaves the queue in a perfectly valid but momentarily "lagging" state
// -- and the fix is not to wait for that thread, but for ANY other
// thread that notices the lag to finish the swing on its behalf.

#define POOL_SIZE 8

struct Node {
    int value;
    int next;   // -1 means "no node linked here yet"
};

// Link step: try to attach a new node after the node `tail` currently
// points at, but ONLY if that node's `next` is still empty. If some
// other thread already linked something there, this CAS simply fails --
// exactly Chapter 9's CAS discipline, now applied to a `next` field
// instead of to `head` itself.
__global__ void enqueue_link(Node* g_nodes, int tail_idx, int new_idx, int* out_success) {
    *out_success = (atomicCAS(&g_nodes[tail_idx].next, -1, new_idx) == -1) ? 1 : 0;
}

// Swing step: try to move the shared tail pointer forward to the
// just-linked node. Deliberately allowed to FAIL harmlessly: if some
// OTHER thread already swung tail forward (because it noticed the lag
// and helped), this thread's own swing attempt has nothing left to do.
__global__ void enqueue_swing(int* g_tail, int old_tail, int new_tail, int* out_success) {
    *out_success = (atomicCAS(g_tail, old_tail, new_tail) == old_tail) ? 1 : 0;
}

// ---- Host-side replay of one specific, deterministic interleaving:
// ---- Thread 1 links its new node, then is delayed before swinging
// ---- tail. Thread 2's own enqueue attempt discovers the lag (tail's
// ---- current `next` is already occupied) and helps swing tail forward
// ---- BEFORE proceeding with its own link -- exactly the mechanism a
// ---- real concurrent queue relies on instead of ever waiting. ----

int main() {
    printf("=== Section 10.1: link-then-swing enqueue, and helping a lagging tail ===\n\n");

    std::vector<Node> nodes(POOL_SIZE);
    nodes[0] = {-1, -1};   // slot 0: the permanent dummy/sentinel node
    int head = 0;
    int tail = 0;
    int next_free = 1;

    printf("initial state: empty queue, head = tail = slot 0 (dummy)\n\n");

    printf("Thread 1 begins enqueue(10): reads tail=0, sees nodes[0].next=-1 (empty),\n");
    printf("links its new node (slot 1) there, then is delayed BEFORE swinging tail.\n");
    int node1 = next_free++;
    nodes[node1] = {10, -1};
    bool t1_link_ok = (nodes[tail].next == -1);
    nodes[tail].next = node1;   // atomicCAS(&nodes[0].next, -1, 1): succeeds
    printf("  Thread 1's link CAS: %s\n\n", t1_link_ok ? "succeeded" : "FAILED");

    printf("Thread 2 begins enqueue(20): reads tail=0 (still stale -- Thread 1 has not\n");
    printf("swung it yet), reads nodes[0].next=%d (NOT -1) -- tail is LAGGING, not empty.\n",
           nodes[0].next);
    printf("Thread 2 helps: swings tail from 0 to %d before attempting its own link.\n", node1);
    bool t2_helped_swing = (tail == 0);
    tail = node1;   // atomicCAS(&tail, 0, 1): Thread 2's helping swing succeeds
    printf("  Thread 2's helping swing CAS: %s (tail is now %d)\n\n",
           t2_helped_swing ? "succeeded" : "FAILED", tail);

    printf("Thread 2 retries its enqueue with the now-current tail=%d: reads\n", tail);
    printf("nodes[%d].next=%d (empty), links its new node (slot 2) there, then\n", tail, nodes[tail].next);
    printf("swings tail forward itself.\n");
    int node2 = next_free++;
    nodes[node2] = {20, -1};
    bool t2_link_ok = (nodes[tail].next == -1);
    nodes[tail].next = node2;
    bool t2_swing_ok = (tail == node1);
    tail = node2;
    printf("  Thread 2's link CAS: %s, Thread 2's own swing CAS: %s (tail is now %d)\n\n",
           t2_link_ok ? "succeeded" : "FAILED", t2_swing_ok ? "succeeded" : "FAILED", tail);

    printf("Thread 1 finally resumes and attempts ITS OWN delayed swing (tail: 0 -> %d):\n",
           node1);
    bool t1_swing_ok = (tail == 0);   // tail is now node2, NOT 0 -- this CAS correctly fails
    printf("  Thread 1's own swing CAS: %s (tail is already %d, not 0 -- nothing to do,\n",
           t1_swing_ok ? "succeeded" : "correctly FAILED", tail);
    printf("  and Thread 1's enqueue is already complete regardless, since its LINK\n");
    printf("  already succeeded -- the swing was only ever a courtesy for readers)\n\n");

    // Walk the final queue from head.next and verify FIFO order + full linkage.
    std::vector<int> walked_values;
    int cur = nodes[head].next;
    while (cur != -1) { walked_values.push_back(nodes[cur].value); cur = nodes[cur].next; }

    printf("final queue, walked from head: [");
    for (size_t i = 0; i < walked_values.size(); i++) printf("%d%s", walked_values[i], i + 1 < walked_values.size() ? ", " : "");
    printf("]\n");
    printf("final tail points at slot %d (value %d)\n", tail, nodes[tail].value);

    bool fifo_order_correct = (walked_values.size() == 2 && walked_values[0] == 10 && walked_values[1] == 20);
    bool tail_correctly_advanced = (tail == node2);
    bool t1_link_succeeded_despite_delayed_swing = t1_link_ok && !t1_swing_ok;

    printf("\nboth values enqueued, in the correct FIFO order despite the delayed swing: %s\n",
           fifo_order_correct ? "yes" : "NO -- BUG");
    printf("tail correctly ends up at the true last node: %s\n", tail_correctly_advanced ? "yes" : "NO -- BUG");
    printf("Thread 1's enqueue succeeded (link) even though its own swing was pre-empted\n");
    printf("by a helper: %s\n", t1_link_succeeded_despite_delayed_swing ? "yes" : "NO -- BUG");

    bool ok = fifo_order_correct && tail_correctly_advanced && t1_link_succeeded_despite_delayed_swing
              && t2_helped_swing && t1_link_ok && t2_link_ok && t2_swing_ok;
    printf("\nself-check: correct FIFO order preserved, tail correctly advanced by a helper,\n");
    printf("delayed swing correctly becomes a harmless no-op: %s\n", ok ? "confirmed" : "MISMATCH");
    return ok ? 0 : 1;
}
```

**Compile and run:**

```bash
nvcc -arch=sm_80 28_queue_link_then_swing_and_helping.cu -o link_then_swing
./link_then_swing
```

**Sample input:** an empty queue (`head = tail = dummy`), with Thread 1 enqueuing `10` and getting delayed before its swing, then Thread 2 enqueuing `20` and discovering the lag.

**Sample output:**

```text
=== Section 10.1: link-then-swing enqueue, and helping a lagging tail ===

initial state: empty queue, head = tail = slot 0 (dummy)

Thread 1 begins enqueue(10): reads tail=0, sees nodes[0].next=-1 (empty),
links its new node (slot 1) there, then is delayed BEFORE swinging tail.
  Thread 1's link CAS: succeeded

Thread 2 begins enqueue(20): reads tail=0 (still stale -- Thread 1 has not
swung it yet), reads nodes[0].next=1 (NOT -1) -- tail is LAGGING, not empty.
Thread 2 helps: swings tail from 0 to 1 before attempting its own link.
  Thread 2's helping swing CAS: succeeded (tail is now 1)

Thread 2 retries its enqueue with the now-current tail=1: reads
nodes[1].next=-1 (empty), links its new node (slot 2) there, then
swings tail forward itself.
  Thread 2's link CAS: succeeded, Thread 2's own swing CAS: succeeded (tail is now 2)

Thread 1 finally resumes and attempts ITS OWN delayed swing (tail: 0 -> 1):
  Thread 1's own swing CAS: correctly FAILED (tail is already 2, not 0 -- nothing to do,
  and Thread 1's enqueue is already complete regardless, since its LINK
  already succeeded -- the swing was only ever a courtesy for readers)

final queue, walked from head: [10, 20]
final tail points at slot 2 (value 20)

both values enqueued, in the correct FIFO order despite the delayed swing: yes
tail correctly ends up at the true last node: yes
Thread 1's enqueue succeeded (link) even though its own swing was pre-empted
by a helper: yes

self-check: correct FIFO order preserved, tail correctly advanced by a helper,
delayed swing correctly becomes a harmless no-op: confirmed
```

Both values end up correctly linked in FIFO order despite Thread 1's delay, tail correctly ends up at the true last node (advanced by Thread 2's help), and Thread 1's own eventual swing attempt correctly fails and does nothing, since its enqueue was already complete.

!!! warning "[COMMON TRAP] Assuming a lagging tail means a lost or corrupted node"
    A tail that has not yet been swung forward looks, at first glance, like something has gone wrong — the "true" last node and what `tail` points at disagree. Nothing is actually broken: the node is correctly linked into the list either way, and Section 10.1's own trace shows the queue's final state is completely correct regardless of which thread ends up performing the swing. The design deliberately tolerates this temporary disagreement, because insisting `tail` always be perfectly up to date the instant a link happens would require the linking thread to never be delayed or preempted — an assumption no real scheduler can guarantee.

## 10.2 A Complete Lock-Free Queue: Enqueue, Dequeue, and FIFO Verification

### Intuition

Section 10.1 built enqueue's link-then-swing shape. Dequeue needs a symmetric CAS loop on the OTHER end, with one extra wrinkle enqueue never faces: before removing anything, dequeue has to decide whether the queue is genuinely empty or whether `head` and `tail` merely coincide because of a lag exactly like Section 10.1's — a decision this section sets up and Section 10.3 studies in full.

### The Sequential (CPU) Baseline

Completing the trivial single-thread queue needs one more ordinary function:

```cpp
#include <cstdio>
#include <vector>

// Chapter 10.2 -- The Sequential (CPU) Baseline.
// No CAS anywhere, because there is exactly one thread and no
// possibility of another thread's enqueue or dequeue being caught
// mid-step.

struct Node { int value; int next; };

bool dequeue_cpu(std::vector<Node>& nodes, int& head, int* out_value) {
    if (nodes[head].next == -1) return false;
    int next_idx = nodes[head].next;
    *out_value = nodes[next_idx].value;
    head = next_idx;
    return true;
}

int main() {
    printf("=== Section 10.2 CPU baseline: sequential dequeue, no CAS needed ===\n\n");

    std::vector<Node> nodes(4);
    nodes[0] = {-1, 1};
    nodes[1] = {10, 2};
    nodes[2] = {20, 3};
    nodes[3] = {30, -1};
    int head = 0;

    printf("queue built directly: head -> dummy -> node(10) -> node(20) -> node(30)\n\n");

    int v;
    std::vector<int> popped;
    while (dequeue_cpu(nodes, head, &v)) {
        popped.push_back(v);
        printf("dequeue() -> %d\n", v);
    }
    printf("dequeue() -> queue empty\n\n");

    std::vector<int> expected = {10, 20, 30};
    bool ok = (popped == expected);

    printf("expected FIFO order: 10, 20, 30\n");
    printf("\nevery dequeue returns values in the exact order they were enqueued, with\n");
    printf("no coordination needed at all for exactly one thread.\n");
    printf("\nself-check: sequential dequeue produces correct FIFO order: %s\n",
           ok ? "confirmed" : "MISMATCH");
    return ok ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++17 -Wall -Wextra -O2 50_dequeue_cpu_baseline.cpp -o dequeue_cpu_baseline
./dequeue_cpu_baseline
```

**Sample input:** a queue already holding `10, 20, 30` (built directly), drained with repeated `dequeue()` calls until empty.

**Sample output:**

```text
=== Section 10.2 CPU baseline: sequential dequeue, no CAS needed ===

queue built directly: head -> dummy -> node(10) -> node(20) -> node(30)

dequeue() -> 10
dequeue() -> 20
dequeue() -> 30
dequeue() -> queue empty

expected FIFO order: 10, 20, 30

every dequeue returns values in the exact order they were enqueued, with
no coordination needed at all for exactly one thread.

self-check: sequential dequeue produces correct FIFO order: confirmed
```

No CAS anywhere, because there is exactly one thread and no possibility of another thread's enqueue or dequeue being caught mid-step. The GPU version below keeps this exact shape but adds a CAS retry loop to both operations, so many threads can safely call enqueue and dequeue at once.

### The Concept, In Detail

Dequeue's own shape mirrors enqueue's, applied to `head` instead of `tail`:

```
dequeue's own shape:

  step 1 -- read head and tail
  step 2 -- if head == tail: determine whether tail is genuinely caught
            up (queue truly empty) or lagging (Section 10.3's subject)
  step 3 -- otherwise: read the VALUE from head's next node, then CAS
            head forward to that next node
```

Hand-traced on a small example — queue holds `[10, 20]` (10 enqueued first), `head -> dummy -> node(10) -> node(20) <- tail`:

```
dequeue():   next = dummy.next = node(10); value = 10
             CAS head: dummy -> node(10); returns 10
             queue is now: head -> node(10) -> node(20) <- tail

enqueue(30): tail.next is empty; link node(30); swing tail
             queue is now: head -> node(10) -> node(20) -> node(30) <- tail

dequeue():   next = node(10).next = node(20); value = 20
             CAS head: node(10) -> node(20); returns 20
```

Every dequeue returns values in the exact order they were enqueued — true FIFO order — which the code below verifies not on this three-operation hand trace but across a full 30-operation interleaved sequence, checked against an independent reference queue at every single step.

### Code and Verification

```cpp
#include <cstdio>
#include <vector>
#include <queue>

// Chapter 10.2 -- Section 10.1 showed enqueue's link-then-swing shape
// and the helping mechanism a lagging tail relies on. A complete queue
// needs dequeue too, and dequeue faces the identical "lagging tail"
// question from the OTHER end: if head and tail happen to point at the
// same node, is the queue truly empty, or is a just-linked node simply
// waiting for its swing? Getting this right (rather than optimistically
// assuming "head == tail" always means empty) is Section 10.3's subject;
// this section builds the complete, correct enqueue/dequeue pair and
// verifies FIFO order end to end.

#define POOL_SIZE 64

struct Node {
    int value;
    int next;   // -1 means "nothing linked here yet"
};

// Enqueue: link, then swing -- exactly Section 10.1's two-step shape,
// now as one retry loop. A lagging tail (this thread's own, or some
// other thread's) is helped forward before this thread's own link is
// even attempted.
__global__ void enqueue_kernel(Node* g_nodes, int* g_tail, int new_idx) {
    while (true) {
        int tail_idx = *g_tail;
        int next_idx = g_nodes[tail_idx].next;
        if (next_idx == -1) {
            if (atomicCAS(&g_nodes[tail_idx].next, -1, new_idx) == -1) {
                atomicCAS(g_tail, tail_idx, new_idx);   // swing; a failure here is fine, a helper will finish it
                return;
            }
        } else {
            atomicCAS(g_tail, tail_idx, next_idx);   // some tail is lagging -- help it, then retry
        }
    }
}

// Dequeue: check head against tail to tell "truly empty" apart from "a
// lagging tail" (Section 10.3 studies this check in detail), then swap
// head forward past the dummy to the node actually holding the value.
__global__ void dequeue_kernel(Node* g_nodes, int* g_head, int* g_tail,
                                int* out_value, int* out_success) {
    while (true) {
        int head_idx = *g_head;
        int tail_idx = *g_tail;
        int next_idx = g_nodes[head_idx].next;
        if (head_idx == tail_idx) {
            if (next_idx == -1) { *out_success = 0; return; }   // truly empty
            atomicCAS(g_tail, tail_idx, next_idx);               // lagging tail -- help, then retry
        } else {
            int value = g_nodes[next_idx].value;
            if (atomicCAS(g_head, head_idx, next_idx) == head_idx) {
                *out_value = value;
                *out_success = 1;
                return;
            }
        }
    }
}

// ---- Host-side replay of the identical algorithm, checked step by
// ---- step against an independent std::queue reference over a fixed
// ---- sequence of interleaved operations. ----

struct LockFreeQueue {
    std::vector<Node> nodes;
    int head, tail, next_free;

    explicit LockFreeQueue(int capacity) : nodes(capacity) {
        nodes[0] = {-1, -1};   // slot 0: permanent dummy node
        head = 0;
        tail = 0;
        next_free = 1;
    }

    void enqueue(int value) {
        int new_idx = next_free++;
        nodes[new_idx] = {value, -1};
        while (true) {
            int tail_idx = tail;
            int next_idx = nodes[tail_idx].next;
            if (next_idx == -1) {
                nodes[tail_idx].next = new_idx;   // atomicCAS succeeds: nothing else races in a sequential replay
                tail = new_idx;
                return;
            } else {
                tail = next_idx;   // help a lagging tail (can still happen even without real concurrency,
            }                       // if a previous enqueue's swing was deferred -- kept for fidelity to the kernel)
        }
    }

    bool dequeue(int* out_value) {
        while (true) {
            int head_idx = head;
            int tail_idx = tail;
            int next_idx = nodes[head_idx].next;
            if (head_idx == tail_idx) {
                if (next_idx == -1) return false;   // truly empty
                tail = next_idx;                     // help a lagging tail, then retry
            } else {
                *out_value = nodes[next_idx].value;
                head = next_idx;
                return true;
            }
        }
    }
};

int main() {
    printf("=== Section 10.2: a complete lock-free queue -- enqueue, dequeue, FIFO order ===\n\n");

    const int NUM_OPS = 30;
    LockFreeQueue q(POOL_SIZE);
    std::queue<int> reference;

    printf("replaying %d operations: enqueue if (i %% 3 != 2), else dequeue\n\n", NUM_OPS);

    int enqueues = 0, dequeues_attempted = 0, dequeues_succeeded = 0;
    bool all_match = true;

    for (int i = 0; i < NUM_OPS; i++) {
        if (i % 3 != 2) {
            q.enqueue(i);
            reference.push(i);
            enqueues++;
        } else {
            dequeues_attempted++;
            int q_value;
            bool q_ok = q.dequeue(&q_value);
            bool ref_ok = !reference.empty();
            int ref_value = ref_ok ? reference.front() : -1;
            if (ref_ok) reference.pop();

            if (q_ok != ref_ok || (q_ok && q_value != ref_value)) all_match = false;
            if (q_ok) dequeues_succeeded++;
        }
    }

    printf("during the interleaved phase: %d enqueues, %d dequeue attempts (%d succeeded)\n",
           enqueues, dequeues_attempted, dequeues_succeeded);
    printf("every dequeue's returned value matched the independent reference queue: %s\n\n",
           all_match ? "yes" : "NO -- BUG");

    std::vector<int> drained_queue, drained_reference;
    int v;
    while (q.dequeue(&v)) drained_queue.push_back(v);
    while (!reference.empty()) { drained_reference.push_back(reference.front()); reference.pop(); }

    bool drain_matches = (drained_queue == drained_reference);
    printf("draining the remaining queue: %zu elements left\n", drained_queue.size());
    printf("full remaining FIFO order matches the independent reference exactly: %s\n",
           drain_matches ? "yes" : "NO -- BUG");
    printf("first 5 drained values: ");
    for (size_t i = 0; i < 5 && i < drained_queue.size(); i++) printf("%d ", drained_queue[i]);
    printf("\n\n");

    int total_dequeued = dequeues_succeeded + (int)drained_queue.size();
    bool conservation = (enqueues == total_dequeued);
    printf("total values enqueued across the whole run: %d\n", enqueues);
    printf("total values dequeued across the whole run (interleaved + drain): %d\n", total_dequeued);
    printf("every enqueued value was eventually dequeued exactly once, in FIFO order,\n");
    printf("none lost or duplicated: %s\n", conservation ? "yes" : "NO -- BUG");

    bool ok = all_match && drain_matches && conservation;
    printf("\nself-check: interleaved dequeues correct, full drain matches reference FIFO\n");
    printf("order, every enqueued value conserved exactly once: %s\n", ok ? "confirmed" : "MISMATCH");
    return ok ? 0 : 1;
}
```

**Compile and run:**

```bash
nvcc -arch=sm_80 29_complete_lock_free_queue.cu -o complete_queue
./complete_queue
```

**Sample input:** 30 operations (`enqueue(i)` if `i % 3 != 2`, else `dequeue()`, for `i` in `[0, 30)`), replayed against both the CAS-loop queue and an independent `std::queue` reference.

**Sample output:**

```text
=== Section 10.2: a complete lock-free queue -- enqueue, dequeue, FIFO order ===

replaying 30 operations: enqueue if (i % 3 != 2), else dequeue

during the interleaved phase: 20 enqueues, 10 dequeue attempts (10 succeeded)
every dequeue's returned value matched the independent reference queue: yes

draining the remaining queue: 10 elements left
full remaining FIFO order matches the independent reference exactly: yes
first 5 drained values: 15 16 18 19 21 

total values enqueued across the whole run: 20
total values dequeued across the whole run (interleaved + drain): 20
every enqueued value was eventually dequeued exactly once, in FIFO order,
none lost or duplicated: yes

self-check: interleaved dequeues correct, full drain matches reference FIFO
order, every enqueued value conserved exactly once: confirmed
```

Every one of the 10 interleaved dequeues matches the reference exactly, the full drain of the remaining 10 elements matches the reference's complete FIFO order, and every one of the 20 enqueued values is accounted for exactly once — none lost, none duplicated, none reordered.

!!! warning "[COMMON TRAP] Assuming FIFO order alone proves the implementation is race-free"
    Section 10.2's 30-operation test passing cleanly demonstrates the enqueue/dequeue LOGIC is correct — but the host-side replay is entirely sequential, with no real concurrency to exercise the lagging-tail and helping mechanisms Section 10.1 already proved matter. A logic bug (wrong field read, mishandled empty case) would absolutely be caught by this test; a bug that ONLY manifests under a specific concurrent interleaving — like the exact scenario Section 10.3 is about to construct — would not be, precisely because nothing in a purely sequential replay ever creates that interleaving. Passing this test is necessary, not sufficient, for concurrent correctness.

## 10.3 The Empty-Queue Race: Why Dequeue Must Check Before Reporting Empty

### Intuition

Section 10.2's dequeue checks `head == tail` and then, ONLY if the head node's `next` is also empty, reports the queue truly empty. It is tempting to simplify this to "head equals tail means empty" — on a single thread, that shortcut is always correct. Under concurrency it is not: head and tail can coincide for a completely different reason — a lagging tail, exactly Section 10.1's scenario, now caught from dequeue's side instead of another enqueue's.

### The Sequential (CPU) Baseline

On a single CPU thread, `head == tail` truly does always mean empty, with no exception:

```cpp
#include <cstdio>
#include <vector>

// Chapter 10.3 -- The Sequential (CPU) Baseline.
// On a single CPU thread, head == tail truly does always mean empty,
// with no exception -- there is no gap between one thread's operations
// for another thread's partially-completed enqueue to hide in.

struct Node { int value; int next; };

bool is_empty_cpu(int head, int tail) {
    return head == tail;
}

int main() {
    printf("=== Section 10.3 CPU baseline: head == tail always means empty here ===\n\n");

    std::vector<Node> nodes(4);
    nodes[0] = {-1, -1};
    int head = 0, tail = 0;

    printf("freshly created queue: head = tail = slot 0 (dummy)\n");
    bool empty1 = is_empty_cpu(head, tail);
    printf("is_empty_cpu(head, tail) -> %s\n\n", empty1 ? "true (correct -- genuinely empty)" : "false");

    nodes[1] = {99, -1};
    nodes[0].next = 1;
    tail = 1;
    printf("enqueue(99) completes fully (link AND swing, one indivisible step)\n");
    bool empty2 = is_empty_cpu(head, tail);
    printf("is_empty_cpu(head, tail) -> %s\n\n", empty2 ? "true" : "false (correct -- a value is present)");

    bool ok = (empty1 == true) && (empty2 == false);

    printf("on one thread, head == tail can ONLY mean genuinely empty -- there is no\n");
    printf("intermediate state where a link has happened but a swing has not, because\n");
    printf("nothing else ever runs in between them.\n");
    printf("\nself-check: head == tail shortcut is correct in both cases here: %s\n",
           ok ? "confirmed" : "MISMATCH");
    return ok ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++17 -Wall -Wextra -O2 51_is_empty_cpu_baseline.cpp -o is_empty_cpu_baseline
./is_empty_cpu_baseline
```

**Sample input:** a freshly created empty queue, checked, then a single `enqueue(99)` completed in full (link and swing, one indivisible step), checked again.

**Sample output:**

```text
=== Section 10.3 CPU baseline: head == tail always means empty here ===

freshly created queue: head = tail = slot 0 (dummy)
is_empty_cpu(head, tail) -> true (correct -- genuinely empty)

enqueue(99) completes fully (link AND swing, one indivisible step)
is_empty_cpu(head, tail) -> false (correct -- a value is present)

on one thread, head == tail can ONLY mean genuinely empty -- there is no
intermediate state where a link has happened but a swing has not, because
nothing else ever runs in between them.

self-check: head == tail shortcut is correct in both cases here: confirmed
```

This shortcut is completely safe here for the identical reason Chapter 9.3's ABA problem could not occur on a single CPU thread: there is no gap between one thread's operations for another thread's partially-completed enqueue to hide in. The ambiguity this section studies is, like ABA, a genuinely multi-thread phenomenon.

### The Concept, In Detail

The ambiguity dequeue must resolve, and cannot resolve from `head == tail` alone:

```
case A -- truly empty:   head == tail == dummy, dummy.next = empty
case B -- lagging tail:  head == tail == dummy, dummy.next = SOME NODE
                          (an enqueue linked it but has not swung yet)

both cases show "head == tail" -- that comparison ALONE cannot tell them
apart. Only checking the head node's own `next` field distinguishes them:

  next == empty   ->  case A: genuinely nothing to dequeue
  next != empty    ->  case B: help swing tail, then retry -- there IS
                       a value, it is simply one swing behind
```

The code below constructs case B directly and runs both a buggy dequeue (using the `head == tail` shortcut alone) and the correct dequeue (checking `next` first) against the identical starting state, to show exactly where they diverge.

### Code and Verification

```cpp
#include <cstdio>
#include <vector>

// Chapter 10.3 -- Section 10.2's dequeue checks `head_idx == tail_idx`
// and, only if the head node's `next` is ALSO empty, reports the queue
// truly empty. It is tempting to simplify this to just "head equals tail
// means empty" -- after all, in a single-threaded world, head and tail
// only ever coincide when the queue genuinely has nothing in it. Under
// concurrency this shortcut is wrong: head and tail can coincide for a
// different reason entirely -- a lagging tail, exactly Section 10.1's
// scenario, caught here from dequeue's side instead of another
// enqueue's side.

struct Node {
    int value;
    int next;
};

// The BUGGY shortcut: treat head == tail as sufficient proof of an
// empty queue, with no further check at all.
__global__ void dequeue_buggy(Node* g_nodes, int* g_head, int* g_tail, int* out_success) {
    int head_idx = *g_head;
    int tail_idx = *g_tail;
    *out_success = (head_idx == tail_idx) ? 0 : 1;   // reports empty the instant they match
}

// The correct check: head == tail is NECESSARY but not SUFFICIENT for
// "empty" -- the head node's own `next` field is what actually decides
// it, exactly Section 10.2's dequeue_kernel.
__global__ void dequeue_correct(Node* g_nodes, int* g_head, int* g_tail,
                                 int* out_value, int* out_success) {
    while (true) {
        int head_idx = *g_head;
        int tail_idx = *g_tail;
        int next_idx = g_nodes[head_idx].next;
        if (head_idx == tail_idx) {
            if (next_idx == -1) { *out_success = 0; return; }
            atomicCAS(g_tail, tail_idx, next_idx);
        } else {
            int value = g_nodes[next_idx].value;
            if (atomicCAS(g_head, head_idx, next_idx) == head_idx) {
                *out_value = value;
                *out_success = 1;
                return;
            }
        }
    }
}

// ---- Host-side replay of one specific, deterministic interleaving:
// ---- an enqueue links its node but is delayed before swinging tail,
// ---- so head and tail still coincide -- exactly the ambiguous moment
// ---- both dequeue implementations above have to interpret. ----

int main() {
    printf("=== Section 10.3: the empty-queue race -- head == tail is not enough ===\n\n");

    std::vector<Node> nodes(4);
    nodes[0] = {-1, -1};   // dummy
    int head = 0;

    printf("starting state: an empty queue, head = tail = slot 0 (dummy)\n\n");
    printf("an enqueue(99) begins: reads tail=0, sees nodes[0].next=-1, links its new\n");
    printf("node (slot 1, value 99) there, then is delayed BEFORE swinging tail.\n");
    nodes[1] = {99, -1};
    nodes[0].next = 1;   // the link succeeds
    int tail = 0;          // tail is STILL 0 -- the swing has not happened yet
    printf("current true state: head=0, tail=0 (STALE), nodes[0].next=1 -- a value IS\n");
    printf("legitimately linked into the queue already, tail just has not caught up.\n\n");

    // --- Buggy dequeue: head == tail alone -----------------------------
    printf("--- dequeue attempt using the BUGGY shortcut (head == tail means empty) ---\n");
    bool buggy_reports_empty = (head == tail);
    printf("head (%d) == tail (%d): reports queue EMPTY: %s\n\n", head, tail,
           buggy_reports_empty ? "yes -- but a value is genuinely sitting in the queue!" : "no");

    // --- Correct dequeue: check next before deciding -------------------
    printf("--- dequeue attempt using the CORRECT check (verify next before deciding) ---\n");
    int correct_head = head, correct_tail = tail;
    int retries = 0;
    int returned_value = -1;
    bool correct_reports_empty = false;
    while (true) {
        int next_idx = nodes[correct_head].next;
        if (correct_head == correct_tail) {
            if (next_idx == -1) { correct_reports_empty = true; break; }
            printf("head (%d) == tail (%d), but nodes[%d].next = %d (not -1) -- tail is\n",
                   correct_head, correct_tail, correct_head, next_idx);
            printf("LAGGING, not empty. Helping: swing tail from %d to %d, then retry.\n",
                   correct_tail, next_idx);
            correct_tail = next_idx;   // the helping swing
            retries++;
        } else {
            returned_value = nodes[next_idx].value;
            correct_head = next_idx;   // the actual dequeue
            break;
        }
    }
    printf("result: %s, value = %d, after %d helping retry(ies)\n\n",
           correct_reports_empty ? "EMPTY" : "value returned", returned_value, retries);

    printf("meanwhile, the original enqueue's own delayed swing (tail: 0 -> 1) now finds\n");
    printf("tail already at %d (the dequeue's helper got there first) -- its CAS correctly\n",
           correct_tail);
    printf("fails and does nothing further; the enqueue was already complete regardless.\n\n");

    bool buggy_is_wrong = buggy_reports_empty;   // it should NOT have reported empty
    bool correct_is_right = (!correct_reports_empty) && (returned_value == 99) && (retries == 1);

    printf("buggy shortcut incorrectly reports EMPTY on a queue that genuinely holds a\n");
    printf("value: %s\n", buggy_is_wrong ? "yes -- CONFIRMED BUG" : "no");
    printf("correct check detects the lagging tail, helps, retries once, and correctly\n");
    printf("returns the genuinely enqueued value 99: %s\n", correct_is_right ? "yes" : "NO -- BUG");

    bool ok = buggy_is_wrong && correct_is_right;
    printf("\nself-check: the buggy shortcut is demonstrably wrong on this exact scenario,\n");
    printf("the correct check handles it exactly right: %s\n", ok ? "confirmed" : "MISMATCH");
    return ok ? 0 : 1;
}
```

**Compile and run:**

```bash
nvcc -arch=sm_80 30_empty_queue_race.cu -o empty_queue_race
./empty_queue_race
```

**Sample input:** a queue where an enqueue of `99` has already linked its node but has not yet swung `tail` — so `head == tail`, yet `head`'s `next` field already points at a real, valid, unread value.

**Sample output:**

```text
=== Section 10.3: the empty-queue race -- head == tail is not enough ===

starting state: an empty queue, head = tail = slot 0 (dummy)

an enqueue(99) begins: reads tail=0, sees nodes[0].next=-1, links its new
node (slot 1, value 99) there, then is delayed BEFORE swinging tail.
current true state: head=0, tail=0 (STALE), nodes[0].next=1 -- a value IS
legitimately linked into the queue already, tail just has not caught up.

--- dequeue attempt using the BUGGY shortcut (head == tail means empty) ---
head (0) == tail (0): reports queue EMPTY: yes -- but a value is genuinely sitting in the queue!

--- dequeue attempt using the CORRECT check (verify next before deciding) ---
head (0) == tail (0), but nodes[0].next = 1 (not -1) -- tail is
LAGGING, not empty. Helping: swing tail from 0 to 1, then retry.
result: value returned, value = 99, after 1 helping retry(ies)

meanwhile, the original enqueue's own delayed swing (tail: 0 -> 1) now finds
tail already at 1 (the dequeue's helper got there first) -- its CAS correctly
fails and does nothing further; the enqueue was already complete regardless.

buggy shortcut incorrectly reports EMPTY on a queue that genuinely holds a
value: yes -- CONFIRMED BUG
correct check detects the lagging tail, helps, retries once, and correctly
returns the genuinely enqueued value 99: yes

self-check: the buggy shortcut is demonstrably wrong on this exact scenario,
the correct check handles it exactly right: confirmed
```

The buggy shortcut reports the queue empty despite a genuinely enqueued value sitting right there; the correct check recognizes the lag, helps swing tail forward, retries once, and correctly returns the value.

!!! warning "[COMMON TRAP] Believing this race is rare enough to ignore in practice"
    It is tempting to treat the exact window between a successful link and its swing as vanishingly brief and therefore not worth defending against carefully. The window's brevity affects how OFTEN a concurrent dequeuer lands inside it, not whether the `head == tail` shortcut is correct when one does — and Section 10.3's own trace shows the consequence is not a delayed or retried operation but a flatly WRONG answer handed directly to the caller. A check that is only sometimes necessary is not optional engineering; it is the difference between a queue that is correct under concurrency and one that merely appears correct in testing that never happens to land in the narrow window where it matters.

## Chapter Summary

Section 10.1 showed enqueue's link-then-swing shape and the helping mechanism that lets any thread finish a lagging tail on a delayed thread's behalf, verified on a scripted two-thread interleaving that preserved correct FIFO order despite the delay. Section 10.2 completed the queue with a symmetric dequeue CAS loop, verified against an independent reference across 30 interleaved operations with exact value conservation and zero mismatches. Section 10.3 exposed the one check dequeue cannot skip: `head == tail` alone cannot distinguish a truly empty queue from one whose tail is simply lagging, and a buggy shortcut that skips checking `next` reports a flatly wrong "empty" result on a queue that genuinely holds a value — fixed by helping the lagging tail forward before ever concluding the queue is empty. Part 2 now turns from lock-free stacks and queues to the structure they were both built to avoid: pointer-chasing linked lists, and a full accounting of why the traversal pattern so natural on a CPU is the wrong shape for a machine built around thousands of threads moving in lockstep.

## Self-Check Questions

1. Section 10.1's scripted scenario had Thread 1 link value 10 and get delayed, then Thread 2 help-swing and link value 20. If a THIRD thread had also started enqueuing value 30 before Thread 2's help-swing completed, and read the same stale `tail = 0`, what would Thread 3 discover, and what would it do?
2. Explain why enqueue's swing step (`CAS(tail, old_tail, new_tail)`) is allowed to fail silently, while the link step (`CAS(nodes[tail].next, -1, new_idx)`) failing means the enqueue must retry its ENTIRE loop from a fresh tail read, not just retry the swing.
3. Section 10.2 verified 30 interleaved operations against an independent `std::queue` reference. Explain concretely what a mismatch between the two would have indicated, given that the host-side replay is entirely sequential.
4. Section 10.3 shows a buggy dequeue reporting "empty" when a value is genuinely present. Explain why this specific bug is arguably worse than Chapter 9.1's naive push race, which at least loses data outright rather than reporting a wrong answer about data that is still recoverable.
5. Using Section 10.3's own scripted scenario, if the dequeue call had instead run BEFORE the enqueue's link step even began (so `nodes[0].next` was still `-1` at the moment of the check), what should the correct dequeue report, and would the buggy shortcut have given the same answer in that specific case?
6. Chapter 9.3 fixed the ABA problem for a stack's single `head` pointer with a tagged `(tag, index)` value. Explain concretely why a lock-free queue built on this chapter's design would need TWO separate tagged values, not one, to be fully ABA-safe.

## Where We Go Next

Part 2 now turns from lock-free stacks and queues to the structure they were both built to avoid in the first place: pointer-chasing linked lists. A CPU traverses a linked list by following one pointer at a time, paying one unpredictable memory-latency cost per hop — a pattern this book's own Chapter 1 already flagged as suspicious for a machine built around thousands of threads that need to move in lockstep. The next chapter gives that suspicion a full, measured accounting, and explains why arrays, stacks, and queues have all been built on contiguous, index-based storage throughout Part 2 rather than on real pointers, even when "pointer" is the word this book has used to describe them informally.

## Worked Solutions

**1.** Thread 3 would read `tail = 0` (the same stale value), read `nodes[0].next = 1` (Thread 1's already-linked node, not `-1`) — exactly Thread 2's own discovery — so Thread 3 would ALSO recognize a lagging tail and attempt to help swing it (`CAS(tail, 0, 1)`). Whichever of Thread 2 and Thread 3 issues that CAS first succeeds; the other's identical CAS simply fails harmlessly (tail no longer equals 0), and both then retry their own enqueue with the now-current tail — no data is lost or duplicated regardless of which of the two happened to win the helping race.

**2.** The swing step's only job is to move `tail` to a node that is ALREADY correctly linked — if this thread's swing fails, it can only be because some other thread already performed the identical (or a later) swing, so there is nothing left for this thread to do; the enqueue's real work, the link, is already complete. The link step is different: if it fails, that specific node was never actually added to the list at all — some other thread's node occupies that slot instead — so this thread's own value has not been placed anywhere yet, and it must start over with a fresh tail read to find a genuinely open slot to link into.

**3.** Because the host-side replay applies both implementations to the IDENTICAL operation sequence with no real concurrency introducing any legitimate scheduling difference between them (mirroring Section 9.2's identical reasoning for the stack), any mismatch could only come from a genuine bug in the enqueue or dequeue logic itself — a wrong field read, an incorrectly ordered step, a missed lagging-tail check — not from an acceptable variation in execution order.

**4.** Chapter 9.1's naive push race is silent but at least CONSISTENT: the lost pushes are gone and stay gone, and the final state (fewer reachable elements than expected) is a stable, checkable fact once observed. Section 10.3's buggy dequeue is worse in a specific way: it returns a definite, actionable WRONG ANSWER ("empty") about data that is NOT actually lost — the value is still sitting in the queue, genuinely retrievable a moment later — so a caller that trusts the "empty" result and gives up, rather than retrying, discards a perfectly good value based on bad information, turning a temporary, recoverable timing quirk into a real, avoidable loss purely through the caller's own reasonable trust in the queue's answer.

**5.** If the check ran before the link step began, `nodes[0].next` would genuinely still be `-1`, so the correct dequeue's own check (`next_idx == -1`) would also conclude the queue is truly empty and correctly report so. In this specific ordering, the buggy shortcut's answer (also "empty," from `head == tail` alone) would happen to AGREE with the correct check — the buggy shortcut is not wrong in every case where head equals tail, only in the specific case demonstrated in this section, where a link has already happened but the swing has not, which is exactly why the bug is so easy to miss in casual testing that does not specifically construct that interleaving.

**6.** Enqueue's `tail` pointer and dequeue's `head` pointer are two INDEPENDENT shared values, each updated by its own separate CAS loop, and each can suffer its own independent ABA collision — a reused pool slot could fool a stale `tail` comparison in one thread while having nothing to do with any `head` comparison in another, and vice versa. A single shared tag could not distinguish "this specific value of head changed" from "this specific value of tail changed," since the two pointers advance independently and at different rates; each needs its own monotonically incrementing tag, packed alongside its own index, exactly mirroring Chapter 9.3's fix applied twice, once per pointer.
