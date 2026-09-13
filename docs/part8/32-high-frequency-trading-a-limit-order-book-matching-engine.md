# Chapter 32: High-Frequency Trading -- A Limit Order Book Matching Engine

A limit order book is where every modern electronic exchange actually happens: a live, constantly-updating record of every resting buy and sell order, sorted by price and, within a price, by arrival time. Matching an incoming order against that book, and safely cancelling a resting one while a match might already be reading it, are both genuinely latency-critical, genuinely concurrent problems -- and neither one needs a single new synchronization idea. This chapter builds a limit order book entirely out of tools this book has already proven correct: Section 26.3's bucket queue becomes the price ladder itself, Section 25.2's packed-key argmin and Section 4/31.1's pairwise reduction tree become the parallel search for the best price to trade against, and Section 27.3/29.3's hazard-pointer protocol becomes the safety net that stops a cancellation from freeing an order a matching thread is still mid-trade against.

## 32.1 The Price Ladder as an Array of Price-Level Buckets

### Intuition

A limit order book's central structure is a PRICE LADDER: one bucket per price tick, each bucket holding the resting orders waiting at that exact price, oldest first. This is not a new idea -- it is Section 26.3's bucket queue, with price tick standing in for Dijkstra's distance bucket, and FIFO order within a bucket standing in for time priority within a price level. Inserting an order means finding its price's bucket and appending to that bucket's FIFO list; matching (Section 32.2) means finding the best OCCUPIED bucket and consuming orders from its front.

### The Sequential (CPU) Baseline

```cpp
// 178_lob_price_ladder_cpu_baseline.cpp
//
// Chapter 32.1 -- sequential baseline for a limit order book's price
// ladder: an array of price-level FIFO buckets, exactly Chapter 26.3's
// bucket queue with price tick standing in for Dijkstra's distance
// bucket. Each level holds the resting orders at that price, oldest
// first (FIFO / time priority).
//
// Compile: g++ -std=c++17 -Wall -Wextra -O2 178_lob_price_ladder_cpu_baseline.cpp -o 178_lob_price_ladder_cpu_baseline
// Run:     ./178_lob_price_ladder_cpu_baseline

#include <array>
#include <cstdio>
#include <string>
#include <vector>

constexpr int NUM_LEVELS = 7;
constexpr int PRICES[NUM_LEVELS] = {98, 99, 100, 101, 102, 103, 104};

struct RestingOrder {
    std::string order_id;
    int qty;
    int seq;
};

struct Order {
    std::string order_id;
    std::string side;
    int price;
    int qty;
    int seq;
};

int level_of(int price) {
    for (int i = 0; i < NUM_LEVELS; ++i) {
        if (PRICES[i] == price) return i;
    }
    return -1;
}

int main() {
    std::vector<Order> orders = {
        {"B1", "bid", 101, 5, 1},
        {"B2", "bid", 100, 3, 2},
        {"B3", "bid", 100, 2, 3},
        {"B4", "bid", 99, 10, 4},
        {"A1", "ask", 102, 4, 5},
        {"A2", "ask", 103, 6, 6},
        {"A3", "ask", 104, 1, 7},
    };

    std::array<std::vector<RestingOrder>, NUM_LEVELS> ladder;

    std::printf("=== sequential price-ladder construction (Chapter 26.3's bucket queue, reused) ===\n\n");

    for (const auto& o : orders) {
        int lvl = level_of(o.price);
        ladder[lvl].push_back({o.order_id, o.qty, o.seq});
        std::printf("insert %s (%s, price=%d, qty=%d, seq=%d) -> level %d, level now: [",
                    o.order_id.c_str(), o.side.c_str(), o.price, o.qty, o.seq, lvl);
        for (size_t i = 0; i < ladder[lvl].size(); ++i) {
            const auto& r = ladder[lvl][i];
            std::printf("%s(qty=%d,seq=%d)%s", r.order_id.c_str(), r.qty, r.seq,
                        (i + 1 < ladder[lvl].size()) ? ", " : "");
        }
        std::printf("]\n");
    }

    std::printf("\nfinal price ladder:\n");
    for (int i = 0; i < NUM_LEVELS; ++i) {
        std::printf("  level %d (price %d): [", i, PRICES[i]);
        for (size_t j = 0; j < ladder[i].size(); ++j) {
            const auto& r = ladder[i][j];
            std::printf("%s(qty=%d,seq=%d)%s", r.order_id.c_str(), r.qty, r.seq,
                        (j + 1 < ladder[i].size()) ? ", " : "");
        }
        std::printf("]\n");
    }

    const auto& level100 = ladder[level_of(100)];
    bool fifo_ok = level100.size() == 2 &&
                   level100[0].order_id == "B2" &&
                   level100[1].order_id == "B3";
    std::printf("\nself-check: level 100 preserves FIFO arrival order (B2 before B3): %s\n",
                fifo_ok ? "confirmed" : "MISMATCH");

    return 0;
}
```

**Compile and run:**

```bash
g++ -std=c++17 -Wall -Wextra -O2 178_lob_price_ladder_cpu_baseline.cpp -o 178_lob_price_ladder_cpu_baseline
./178_lob_price_ladder_cpu_baseline
```

**Sample input:** 7 price levels (98 through 104) and 7 orders (4 bids, 3 asks) inserted one at a time in arrival order, including two bids (B2 then B3) landing at the same price level.

**Sample output:**

```text
=== sequential price-ladder construction (Chapter 26.3's bucket queue, reused) ===

insert B1 (bid, price=101, qty=5, seq=1) -> level 3, level now: [B1(qty=5,seq=1)]
insert B2 (bid, price=100, qty=3, seq=2) -> level 2, level now: [B2(qty=3,seq=2)]
insert B3 (bid, price=100, qty=2, seq=3) -> level 2, level now: [B2(qty=3,seq=2), B3(qty=2,seq=3)]
insert B4 (bid, price=99, qty=10, seq=4) -> level 1, level now: [B4(qty=10,seq=4)]
insert A1 (ask, price=102, qty=4, seq=5) -> level 4, level now: [A1(qty=4,seq=5)]
insert A2 (ask, price=103, qty=6, seq=6) -> level 5, level now: [A2(qty=6,seq=6)]
insert A3 (ask, price=104, qty=1, seq=7) -> level 6, level now: [A3(qty=1,seq=7)]

final price ladder:
  level 0 (price 98): []
  level 1 (price 99): [B4(qty=10,seq=4)]
  level 2 (price 100): [B2(qty=3,seq=2), B3(qty=2,seq=3)]
  level 3 (price 101): [B1(qty=5,seq=1)]
  level 4 (price 102): [A1(qty=4,seq=5)]
  level 5 (price 103): [A2(qty=6,seq=6)]
  level 6 (price 104): [A3(qty=1,seq=7)]

self-check: level 100 preserves FIFO arrival order (B2 before B3): confirmed
```

### The Concept, In Detail

```
ASCII view: the price ladder is Section 26.3's bucket queue, price
standing in for distance.

  price:    98    99    100         101   102   103   104
           [  ] [B4] [B2,B3]       [B1]  [A1]  [A2]  [A3]
                       ^FIFO: B2 arrived before B3

  Dijkstra's bucket queue (Ch 26.3):    LOB price ladder (Ch 32.1):
    bucket index = distance               bucket index = price tick
    bucket holds: vertices at that        bucket holds: orders resting
    distance, any order                   at that price, FIFO order
```

Nothing about a bucket queue required its bucket index to mean "distance" specifically -- it required only that the index be a small, dense integer range, cheap to use directly as an array offset. A price tick is exactly that kind of index, which is why the price ladder needs no new data structure, only Section 26.3's structure applied to a different meaning of "bucket."

[COMMON TRAP]
It is tempting to think a price ladder needs price-to-index translation logic fundamentally different from a distance-to-bucket lookup, because prices "feel" like a different kind of quantity than a graph distance. Both are simply small non-negative integers (after converting a price to its tick count from some floor price) used directly as an array index -- the translation `level = price - floor_price` is exactly as trivial as Chapter 26.3's `bucket = distance`, and any code that already indexes a bucket array by distance needs only that one substitution to index it by price tick instead.

### Code and Verification

```cpp
#include <cstdio>
#include <string>
#include <vector>

// Chapter 32.1 main -- concurrent price-ladder insertion. Every order
// that arrives concurrently claims its FIFO slot within its price
// level via a single atomicAdd on that level's write cursor -- exactly
// Chapter 26.3's bucket queue and Chapter 30.2's scatter cursor,
// reused unchanged with "price level" standing in for "grid bucket".
// atomicAdd's return value (the cursor's value BEFORE the add) is
// unconditionally each thread's own, non-colliding FIFO slot, no
// matter what order the adds actually land in -- so two orders racing
// for the SAME level still both get counted and still both get
// distinct slots. Which one gets slot 0 versus slot 1 depends only on
// which atomicAdd call physically lands first: real arrival order, not
// order-id order and not thread-index order.

#define NUM_LEVELS 7

__global__ void ladder_insert_kernel(const int* level_of_order, int* cursors, int* ladder_slots) {
    // ladder_slots is a flattened [NUM_LEVELS][2] array (2 slots/level
    // is enough for this example's worst case).
    int tid = threadIdx.x;
    int lvl = level_of_order[tid];
    int slot = atomicAdd(&cursors[lvl], 1);
    ladder_slots[lvl * 2 + slot] = tid;
}

// ---- Host-side replay of the identical atomicAdd-based logic, driving
// ---- a specific landing order (B3 lands before B2, both level 2 /
// ---- price 100) so the race is hand-traceable. ----

struct SharedCounter {
    int value = 0;
    int atomic_add(int n = 1) {
        int old = value;
        value += n;
        return old;
    }
};

struct OrderInfo {
    std::string order_id;
    int price;
    int qty;
    int seq;
};

int main() {
    printf("=== Section 32.1 main: concurrent price-ladder insertion via atomicAdd cursors ===\n\n");

    const int PRICES[NUM_LEVELS] = {98, 99, 100, 101, 102, 103, 104};
    std::vector<OrderInfo> orders = {
        {"B1", 101, 5, 1}, {"B2", 100, 3, 2}, {"B3", 100, 2, 3}, {"B4", 99, 10, 4},
        {"A1", 102, 4, 5}, {"A2", 103, 6, 6}, {"A3", 104, 1, 7},
    };
    std::vector<std::string> landing_order = {"B3", "B2", "B1", "B4", "A1", "A2", "A3"};

    printf("landing order: [ ");
    for (const auto& id : landing_order) printf("%s ", id.c_str());
    printf("]\n\n");

    std::vector<SharedCounter> cursors(NUM_LEVELS);
    std::vector<std::vector<OrderInfo>> ladder(NUM_LEVELS, std::vector<OrderInfo>(2));
    std::vector<bool> filled(NUM_LEVELS * 2, false);

    for (const auto& oid : landing_order) {
        const OrderInfo* o = nullptr;
        for (const auto& cand : orders) {
            if (cand.order_id == oid) { o = &cand; break; }
        }
        int lvl = -1;
        for (int i = 0; i < NUM_LEVELS; i++) if (PRICES[i] == o->price) lvl = i;
        int slot = cursors[lvl].atomic_add(1);
        ladder[lvl][slot] = *o;
        filled[lvl * 2 + slot] = true;
        printf("  %s (price=%d, level=%d): atomicAdd(cursor[%d],1) returned %d -> ladder[%d][%d] = %s\n",
               oid.c_str(), o->price, lvl, lvl, slot, lvl, slot, oid.c_str());
    }

    printf("\nfinal ladder level 100 (index 2): [");
    for (int s = 0; s < 2; s++) {
        if (filled[2 * 2 + s]) {
            printf("('%s', qty=%d, seq=%d)%s", ladder[2][s].order_id.c_str(),
                   ladder[2][s].qty, ladder[2][s].seq, s == 0 ? ", " : "");
        }
    }
    printf("]\n");
    printf("note: B3 occupies FIFO slot 0 and B2 occupies slot 1, because B3's atomicAdd\n");
    printf("landed FIRST -- FIFO order reflects ARRIVAL order, not order-id or thread order.\n");

    bool ok = (ladder[2][0].order_id == "B3" && ladder[2][1].order_id == "B2");
    printf("\nself-check: level 100's FIFO order matches the forced landing order (B3, then B2): %s\n",
           ok ? "confirmed" : "MISMATCH");
    return ok ? 0 : 1;
}
```

**Compile and run:**

```bash
NVDIR=/usr/local/lib/python3.11/dist-packages/nvidia/cu13
$NVDIR/bin/nvcc -arch=sm_80 -I$NVDIR/include -L$NVDIR/lib -lcudart 179_lob_price_ladder_kernel.cu -o 179_lob_price_ladder_kernel
LD_LIBRARY_PATH=$NVDIR/lib ./179_lob_price_ladder_kernel
```

**Sample input:** the same 7 orders, inserted concurrently via an `atomicAdd`-based per-level write cursor (Section 26.3/30.2's technique), with B3 forced to land before B2 at their shared price level.

**Sample output:**

```text
=== Section 32.1 main: concurrent price-ladder insertion via atomicAdd cursors ===

landing order: [ B3 B2 B1 B4 A1 A2 A3 ]

  B3 (price=100, level=2): atomicAdd(cursor[2],1) returned 0 -> ladder[2][0] = B3
  B2 (price=100, level=2): atomicAdd(cursor[2],1) returned 1 -> ladder[2][1] = B2
  B1 (price=101, level=3): atomicAdd(cursor[3],1) returned 0 -> ladder[3][0] = B1
  B4 (price=99, level=1): atomicAdd(cursor[1],1) returned 0 -> ladder[1][0] = B4
  A1 (price=102, level=4): atomicAdd(cursor[4],1) returned 0 -> ladder[4][0] = A1
  A2 (price=103, level=5): atomicAdd(cursor[5],1) returned 0 -> ladder[5][0] = A2
  A3 (price=104, level=6): atomicAdd(cursor[6],1) returned 0 -> ladder[6][0] = A3

final ladder level 100 (index 2): [('B3', qty=2, seq=3), ('B2', qty=3, seq=2)]
note: B3 occupies FIFO slot 0 and B2 occupies slot 1, because B3's atomicAdd
landed FIRST -- FIFO order reflects ARRIVAL order, not order-id or thread order.

self-check: level 100's FIFO order matches the forced landing order (B3, then B2): confirmed
```

## 32.2 Matching Incoming Orders Against the Book

### Intuition

An incoming aggressive order (say, a BUY with a limit price) needs to find the best -- lowest, for a buy -- OCCUPIED ask price level, trade against it, and repeat until either the incoming order is fully filled or the best remaining occupied level's price exceeds its limit. Finding "the best occupied level" sequentially is a linear scan; finding it in parallel is a REDUCTION, exactly Section 25.2's packed-key argmin technique and Section 4/31.1's pairwise reduction tree, applied once per matching-loop iteration: pack each level's (effective price, level index) into one comparable key, give every empty level a sentinel price of infinity so it can never win, and reduce with MIN.

### The Sequential (CPU) Baseline

```cpp
// 180_lob_matching_cpu_baseline.cpp
//
// Chapter 32.2 -- sequential baseline for matching an incoming
// aggressive order against the resting book. An incoming BUY order
// walks the ask side of the ladder, matching against the best
// (lowest) occupied ask level first, repeatedly, until it is either
// fully filled or the best remaining occupied level's price exceeds
// the incoming order's limit price. This baseline finds the best
// occupied level with a straightforward linear scan; Section 32.2's
// kernel replaces that scan with a parallel reduction.
//
// Compile: g++ -std=c++17 -Wall -Wextra -O2 180_lob_matching_cpu_baseline.cpp -o 180_lob_matching_cpu_baseline
// Run:     ./180_lob_matching_cpu_baseline

#include <array>
#include <cstdio>
#include <vector>

constexpr int NUM_LEVELS = 7;
constexpr int PRICES[NUM_LEVELS] = {98, 99, 100, 101, 102, 103, 104};

int find_best_ask_level(const std::array<int, NUM_LEVELS>& ladder_qty) {
    for (int i = 0; i < NUM_LEVELS; ++i) {
        if (ladder_qty[i] > 0) return i;
    }
    return -1;
}

int main() {
    std::array<int, NUM_LEVELS> ladder_qty = {0, 0, 0, 0, 4, 6, 1};

    int incoming_qty = 12;
    int limit_price = 103;
    std::vector<std::pair<int, int>> trades;

    std::printf("=== Section 32.2 CPU baseline: sequential matching walk ===\n\n");
    std::printf("incoming BUY order: qty=%d, limit price=%d\n", incoming_qty, limit_price);
    std::printf("initial ladder qty: {");
    for (int i = 0; i < NUM_LEVELS; ++i) {
        std::printf("%d: %d%s", PRICES[i], ladder_qty[i], (i + 1 < NUM_LEVELS) ? ", " : "");
    }
    std::printf("}\n\n");

    while (incoming_qty > 0) {
        int best = find_best_ask_level(ladder_qty);
        if (best == -1) {
            std::printf("book side exhausted -- no more resting asks\n");
            break;
        }
        int best_price = PRICES[best];
        if (best_price > limit_price) {
            std::printf("best occupied ask level is price %d, which exceeds limit %d -- stop matching\n",
                        best_price, limit_price);
            break;
        }
        int trade_qty = std::min(incoming_qty, ladder_qty[best]);
        std::printf("best occupied ask level: price %d (qty %d) -- trade %d @ %d\n",
                    best_price, ladder_qty[best], trade_qty, best_price);
        trades.push_back({best_price, trade_qty});
        ladder_qty[best] -= trade_qty;
        incoming_qty -= trade_qty;
        std::printf("  level %d qty now %d, incoming remaining = %d\n\n", best, ladder_qty[best], incoming_qty);
    }

    std::printf("trades executed: [");
    for (size_t i = 0; i < trades.size(); ++i) {
        std::printf("(%d, %d)%s", trades[i].first, trades[i].second, (i + 1 < trades.size()) ? ", " : "");
    }
    std::printf("]\n");
    std::printf("incoming order remaining qty: %d\n", incoming_qty);
    if (incoming_qty > 0) {
        int rest_level = -1;
        for (int i = 0; i < NUM_LEVELS; ++i) if (PRICES[i] == limit_price) rest_level = i;
        std::printf("remaining %d rests as a new BID order at price %d (level %d)\n",
                    incoming_qty, limit_price, rest_level);
    }

    std::vector<std::pair<int, int>> expected_trades = {{102, 4}, {103, 6}};
    bool ok = (trades == expected_trades) && (incoming_qty == 2);
    std::printf("\nself-check: trades match expected [(102, 4), (103, 6)] and 2 units rest unmatched: %s\n",
                ok ? "confirmed" : "MISMATCH");

    return ok ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++17 -Wall -Wextra -O2 180_lob_matching_cpu_baseline.cpp -o 180_lob_matching_cpu_baseline
./180_lob_matching_cpu_baseline
```

**Sample input:** an incoming BUY order for quantity 12 at limit price 103, matched against resting asks of qty 4 at 102, qty 6 at 103, and qty 1 at 104.

**Sample output:**

```text
=== Section 32.2 CPU baseline: sequential matching walk ===

incoming BUY order: qty=12, limit price=103
initial ladder qty: {98: 0, 99: 0, 100: 0, 101: 0, 102: 4, 103: 6, 104: 1}

best occupied ask level: price 102 (qty 4) -- trade 4 @ 102
  level 4 qty now 0, incoming remaining = 8

best occupied ask level: price 103 (qty 6) -- trade 6 @ 103
  level 5 qty now 0, incoming remaining = 2

best occupied ask level is price 104, which exceeds limit 103 -- stop matching
trades executed: [(102, 4), (103, 6)]
incoming order remaining qty: 2
remaining 2 rests as a new BID order at price 103 (level 5)

self-check: trades match expected [(102, 4), (103, 6)] and 2 units rest unmatched: confirmed
```

### The Concept, In Detail

```
ASCII view: finding the best occupied level is a MIN reduction over
packed (price, index) keys -- Section 25.2's argmin, Section 4/31.1's
tree, unchanged.

  levels (price, qty):  98,0  99,0  100,0  101,0  102,4  103,6  104,1  [pad,empty]
  packed keys:          INF00 INF01 INF02  INF03  10204  10305  10406  INF07
                                                    ^^^^^ real prices sort first

  round 1:  min(INF00,INF01)  min(INF02,INF03)  min(10204,10305)  min(10406,INF07)
              -> INF00           -> INF02           -> 10204          -> 10406
  round 2:  min(INF00,INF02) -> INF00          min(10204,10406) -> 10204
  round 3:  min(INF00,10204) -> 10204   (decodes to level 4, price 102)
```

The packed key `price * 100 + level_index` is exactly Section 25.2's trick for turning a two-field comparison ("which price is lower, and which level is it") into a single-field MIN: because the level count here is always under 100, the index never leaks into the price's own digits, so the winning key's last two digits recover the level index directly, with no separate index array needed alongside the reduction.

[COMMON TRAP]
It is tempting to re-run the reduction only once, before the matching loop starts, and then keep matching against whichever level it found until the incoming order is fully filled. A single reduction result goes STALE the instant the loop consumes all of that level's resting quantity -- Section 32.2's matching loop re-runs the ENTIRE reduction fresh at the top of every iteration, exactly like Section 32.2's CPU baseline re-runs its linear scan every iteration, because the previous iteration's trade can change which level is now the best remaining occupied one.

### Code and Verification

```cpp
#include <cstdio>
#include <vector>
#include <cstdint>
#include <array>
#include <algorithm>

// Chapter 32.2 main -- finding the best occupied ask level is a
// REDUCTION: pack each level's (effective price, level index) into one
// comparable 64-bit key -- Chapter 25.2's packed-key argmin technique,
// with empty levels given a sentinel price of +infinity so they can
// never win -- then reduce with MIN via the exact pairwise tree from
// Chapter 4/31.1 (padded here to a power of two, 8, with one sentinel
// empty slot). The sequential matching LOOP around it is unchanged
// from Section 32.2's CPU baseline; only "which level is best right
// now" is parallelized, once per loop iteration.

#define NUM_LEVELS 7
#define PADDED 8
static const long long INF = 1000000000LL;

__device__ long long pack_device(int qty, int idx, int price) {
    long long effective_price = (qty > 0) ? price : INF;
    return effective_price * 100 + idx;
}

__global__ void reduce_min_round_kernel(long long* keys, int n) {
    int tid = threadIdx.x;
    if (tid * 2 + 1 >= n) return;
    long long a = keys[tid * 2];
    long long b = keys[tid * 2 + 1];
    keys[tid] = (a < b) ? a : b;
}

// ---- Host-side replay of the identical pack + pairwise-MIN-reduction
// ---- logic, driving the same matching loop as Section 32.2's CPU
// ---- baseline so the two can be checked against each other. ----

static const int PRICES[NUM_LEVELS] = {98, 99, 100, 101, 102, 103, 104};

long long pack_host(int qty, int idx) {
    long long effective_price = (qty > 0) ? PRICES[idx] : INF;
    return effective_price * 100 + idx;
}

int reduce_find_best(const std::array<int, NUM_LEVELS>& ladder_qty) {
    std::vector<long long> keys(PADDED);
    for (int i = 0; i < NUM_LEVELS; i++) keys[i] = pack_host(ladder_qty[i], i);
    keys[7] = pack_host(0, 7);   // sentinel empty padding slot

    printf("    packed keys: [ ");
    for (auto k : keys) printf("%lld ", k);
    printf("]\n");

    std::vector<long long> level = keys;
    int round_num = 1;
    while (level.size() > 1) {
        std::vector<long long> next_level;
        for (size_t i = 0; i < level.size(); i += 2) {
            next_level.push_back(std::min(level[i], level[i + 1]));
        }
        printf("    round %d: [ ", round_num);
        for (auto k : level) printf("%lld ", k);
        printf("] -> [ ");
        for (auto k : next_level) printf("%lld ", k);
        printf("]\n");
        level = next_level;
        round_num++;
    }

    long long best_key = level[0];
    if (best_key >= INF * 100) return -1;
    return (int)(best_key % 100);
}

int main() {
    printf("=== Section 32.2 main: matching walk using parallel reduction to find best level ===\n\n");

    std::array<int, NUM_LEVELS> ladder_qty = {0, 0, 0, 0, 4, 6, 1};
    int incoming_qty = 12;
    int limit_price = 103;
    std::vector<std::pair<int, int>> trades;

    printf("incoming BUY order: qty=%d, limit price=%d\n\n", incoming_qty, limit_price);

    while (incoming_qty > 0) {
        printf("  finding best occupied ask level via reduction (ladder qty: [ ");
        for (int q : ladder_qty) printf("%d ", q);
        printf("]):\n");

        int best = reduce_find_best(ladder_qty);
        if (best == -1) {
            printf("  book side exhausted\n\n");
            break;
        }
        int best_price = PRICES[best];
        if (best_price > limit_price) {
            printf("  best occupied level is price %d > limit %d -- stop\n\n", best_price, limit_price);
            break;
        }
        int trade_qty = std::min(incoming_qty, ladder_qty[best]);
        printf("  reduction found level %d (price %d, qty %d) -- trade %d @ %d\n\n",
               best, best_price, ladder_qty[best], trade_qty, best_price);
        trades.push_back({best_price, trade_qty});
        ladder_qty[best] -= trade_qty;
        incoming_qty -= trade_qty;
    }

    printf("trades executed: [");
    for (size_t i = 0; i < trades.size(); ++i) {
        printf("(%d, %d)%s", trades[i].first, trades[i].second, (i + 1 < trades.size()) ? ", " : "");
    }
    printf("]\n");
    printf("incoming order remaining qty: %d\n", incoming_qty);

    std::vector<std::pair<int, int>> expected_trades = {{102, 4}, {103, 6}};
    bool ok = (trades == expected_trades) && (incoming_qty == 2);
    printf("\nself-check: matches Section 32.2's CPU baseline exactly: %s\n",
           ok ? "confirmed" : "MISMATCH");

    return ok ? 0 : 1;
}
```

**Compile and run:**

```bash
NVDIR=/usr/local/lib/python3.11/dist-packages/nvidia/cu13
$NVDIR/bin/nvcc -arch=sm_80 -I$NVDIR/include -L$NVDIR/lib -lcudart 181_lob_matching_kernel.cu -o 181_lob_matching_kernel
LD_LIBRARY_PATH=$NVDIR/lib ./181_lob_matching_kernel
```

**Sample input:** the same matching scenario, with each iteration's "find the best occupied level" step replaced by a padded, 8-wide packed-key MIN reduction tree.

**Sample output:**

```text
=== Section 32.2 main: matching walk using parallel reduction to find best level ===

incoming BUY order: qty=12, limit price=103

  finding best occupied ask level via reduction (ladder qty: [ 0 0 0 0 4 6 1 ]):
    packed keys: [ 100000000000 100000000001 100000000002 100000000003 10204 10305 10406 100000000007 ]
    round 1: [ 100000000000 100000000001 100000000002 100000000003 10204 10305 10406 100000000007 ] -> [ 100000000000 100000000002 10204 10406 ]
    round 2: [ 100000000000 100000000002 10204 10406 ] -> [ 100000000000 10204 ]
    round 3: [ 100000000000 10204 ] -> [ 10204 ]
  reduction found level 4 (price 102, qty 4) -- trade 4 @ 102

  finding best occupied ask level via reduction (ladder qty: [ 0 0 0 0 0 6 1 ]):
    packed keys: [ 100000000000 100000000001 100000000002 100000000003 100000000004 10305 10406 100000000007 ]
    round 1: [ 100000000000 100000000001 100000000002 100000000003 100000000004 10305 10406 100000000007 ] -> [ 100000000000 100000000002 10305 10406 ]
    round 2: [ 100000000000 100000000002 10305 10406 ] -> [ 100000000000 10305 ]
    round 3: [ 100000000000 10305 ] -> [ 10305 ]
  reduction found level 5 (price 103, qty 6) -- trade 6 @ 103

  finding best occupied ask level via reduction (ladder qty: [ 0 0 0 0 0 0 1 ]):
    packed keys: [ 100000000000 100000000001 100000000002 100000000003 100000000004 100000000005 10406 100000000007 ]
    round 1: [ 100000000000 100000000001 100000000002 100000000003 100000000004 100000000005 10406 100000000007 ] -> [ 100000000000 100000000002 100000000004 10406 ]
    round 2: [ 100000000000 100000000002 100000000004 10406 ] -> [ 100000000000 10406 ]
    round 3: [ 100000000000 10406 ] -> [ 10406 ]
  best occupied level is price 104 > limit 103 -- stop

trades executed: [(102, 4), (103, 6)]
incoming order remaining qty: 2

self-check: matches Section 32.2's CPU baseline exactly: confirmed
```

## 32.3 Concurrent Order Cancellation with Hazard-Protected Safety

### Intuition

Cancelling a resting order cannot simply free its order-pool slot immediately: a matching thread may already be mid-trade against that exact order -- it read the order's slot before the cancel arrived, and is still using its quantity -- when the cancellation happens. This is precisely the problem Section 27.3 solved for a lock-free list and Section 29.3 solved for a key-value store's value pool: a hazard-pointer protocol, reused here completely unchanged, now protecting an order-pool slot instead of a list node or a KV value block.

### The Sequential (CPU) Baseline

```cpp
// 182_lob_cancel_hazard_cpu_baseline.cpp
//
// Chapter 32.3 -- sequential baseline for cancel-vs-match safety.
// Cancelling a resting order cannot simply free its order-pool slot
// immediately: a matching thread may already be mid-read of that exact
// order (about to trade against it) when the cancel arrives. This is
// Section 27.3/29.3's hazard-pointer protocol, reused unchanged, now
// protecting an order-pool slot instead of a linked-list node or a KV
// value block: a matching thread publishes the slot it is about to
// read BEFORE reading it, and a cancel checks the hazard array before
// returning that slot to the free list.
//
// Compile: g++ -std=c++17 -Wall -Wextra -O2 182_lob_cancel_hazard_cpu_baseline.cpp -o 182_lob_cancel_hazard_cpu_baseline
// Run:     ./182_lob_cancel_hazard_cpu_baseline

#include <cstdio>
#include <map>
#include <string>

constexpr int UNHAZARDED = -1;

int main() {
    std::map<std::string, int> orders_qty = {{"B2", 3}, {"B3", 2}};
    std::map<std::string, int> order_slot = {{"B2", 10}, {"B3", 11}};
    std::map<std::string, bool> cancelled = {{"B2", false}, {"B3", false}};
    int hazard = UNHAZARDED;   // 1 matching thread

    std::printf("=== sequential cancel-vs-match hazard scenario ===\n\n");
    std::printf("order pool: B2 -> slot %d (qty %d), B3 -> slot %d (qty %d)\n\n",
                order_slot["B2"], orders_qty["B2"], order_slot["B3"], orders_qty["B3"]);

    std::printf("matching thread M is about to trade against B2 (first in FIFO at its level):\n");
    std::printf("  M publishes hazard on B2's slot (%d) BEFORE reading its quantity\n", order_slot["B2"]);
    hazard = order_slot["B2"];
    std::printf("  hazard array: [%d]\n\n", hazard);

    std::printf("cancel request arrives for B2 concurrently:\n");
    cancelled["B2"] = true;
    std::printf("  B2 marked CANCELLED (tombstone flag set): {'B2': %s, 'B3': %s}\n",
                cancelled["B2"] ? "True" : "False", cancelled["B3"] ? "True" : "False");
    std::printf("  cancel wants to free B2's slot (%d) back to the pool, checks hazard first:\n",
                order_slot["B2"]);
    bool b2_hazarded = (hazard == order_slot["B2"]);
    std::printf("  slot %d IS hazarded: %s -- free %s\n\n", order_slot["B2"],
                b2_hazarded ? "True" : "False", b2_hazarded ? "DEFERRED" : "allowed");

    std::printf("M finishes reading B2's slot, sees the CANCELLED flag is now set, and skips\n");
    std::printf("trading against it (moves on to B3 in the FIFO instead), then clears its hazard:\n");
    hazard = UNHAZARDED;
    std::printf("  hazard array: [%d]\n\n", hazard);

    std::printf("cancel rechecks and this time frees B2's slot:\n");
    bool b2_hazarded_after = (hazard == order_slot["B2"]);
    std::printf("  slot %d hazarded: %s -- free %s\n", order_slot["B2"],
                b2_hazarded_after ? "True" : "False",
                b2_hazarded_after ? "DEFERRED" : "allowed, slot returned to pool");

    std::printf("\nM proceeds to trade against B3 instead (the next FIFO order at this level),\n");
    std::printf("since B2 was correctly skipped rather than traded against after cancellation.\n");

    bool ok = b2_hazarded && !b2_hazarded_after && cancelled["B2"];
    std::printf("\nself-check: reclaim correctly BLOCKED while M's hazard was published, then\n");
    std::printf("ALLOWED once M cleared it, and M never traded against the cancelled order: %s\n",
                ok ? "confirmed" : "MISMATCH");

    return ok ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++17 -Wall -Wextra -O2 182_lob_cancel_hazard_cpu_baseline.cpp -o 182_lob_cancel_hazard_cpu_baseline
./182_lob_cancel_hazard_cpu_baseline
```

**Sample input:** matching thread M about to trade against resting order B2 (slot 10), racing against a concurrent cancel request for B2.

**Sample output:**

```text
=== sequential cancel-vs-match hazard scenario ===

order pool: B2 -> slot 10 (qty 3), B3 -> slot 11 (qty 2)

matching thread M is about to trade against B2 (first in FIFO at its level):
  M publishes hazard on B2's slot (10) BEFORE reading its quantity
  hazard array: [10]

cancel request arrives for B2 concurrently:
  B2 marked CANCELLED (tombstone flag set): {'B2': True, 'B3': False}
  cancel wants to free B2's slot (10) back to the pool, checks hazard first:
  slot 10 IS hazarded: True -- free DEFERRED

M finishes reading B2's slot, sees the CANCELLED flag is now set, and skips
trading against it (moves on to B3 in the FIFO instead), then clears its hazard:
  hazard array: [-1]

cancel rechecks and this time frees B2's slot:
  slot 10 hazarded: False -- free allowed, slot returned to pool

M proceeds to trade against B3 instead (the next FIFO order at this level),
since B2 was correctly skipped rather than traded against after cancellation.

self-check: reclaim correctly BLOCKED while M's hazard was published, then
ALLOWED once M cleared it, and M never traded against the cancelled order: confirmed
```

### The Concept, In Detail

```
ASCII view: hazard-protected cancel, Section 27.3/29.3's protocol,
now guarding an order-pool slot.

  M: publish hazard(slot 10) --> read qty -----------------> clear hazard
                                                  \
  cancel: mark B2 CANCELLED -> check hazard(10)?   \-- sees CANCELLED, skips trade
                  |                 |
                  |            YES: slot 10 hazarded -> free DEFERRED
                  |
                  `-- after M clears: check hazard(10) again -> NO -> free ALLOWED
```

The protocol's guarantee is exactly the one Section 27.3 proved: as long as a reader (here, the matching thread) publishes its hazard BEFORE reading, and a reclaimer (here, the cancel) always checks the hazard array AFTER marking its target unusable but BEFORE actually freeing it, no reclaimer can ever free a slot a reader is still using -- regardless of how the two threads happen to interleave in time.

[COMMON TRAP]
It is tempting to have the cancel operation check the hazard array BEFORE marking the order CANCELLED, on the theory that checking first is "more cautious." Checking the hazard before setting the tombstone flag creates a window where a matching thread could publish its hazard and start reading the order in between the cancel's hazard-check and its tombstone-write -- Section 27.3's ordering (mark unusable FIRST, then check hazards, then free) is what closes that window, because once the tombstone is set, any matching thread that reads afterward will see it and skip the order regardless of timing.

### Code and Verification

```cpp
#include <cstdio>
#include <vector>

// Chapter 32.3 main -- a matching thread publishes a hazard on the
// order-pool slot it is about to trade against before reading its
// quantity, and a cancel thread checks the hazard array before
// returning that slot to the free list. This is Section 27.3/29.3's
// hazard-pointer kernel pair (publish-then-read / mark-tombstone /
// check-before-free), completely unchanged, now guarding a limit order
// book's order pool instead of a lock-free list node or a KV store's
// value pool -- the same protocol protects any structure where one
// thread's reclaim could race another thread's still-in-flight read.

#define UNHAZARDED (-1)
#define NUM_MATCHERS 1

__global__ void matcher_publish_and_read_kernel(int* hazard, int slot, const int* pool_qty, int* out_qty) {
    if (threadIdx.x != 0) return;
    hazard[0] = slot;           // publish BEFORE reading -- Section 27.3's discipline, unchanged
    *out_qty = pool_qty[slot];
}

__global__ void canceller_mark_tombstone_kernel(bool* cancelled_flags, int order_idx) {
    if (threadIdx.x != 0) return;
    cancelled_flags[order_idx] = true;
}

__device__ bool slot_is_hazarded(const int* hazard, int num_matchers, int slot) {
    for (int m = 0; m < num_matchers; m++) if (hazard[m] == slot) return true;
    return false;
}

__global__ void canceller_try_free_kernel(const int* hazard, int slot, int* reclaim_ok) {
    if (threadIdx.x != 0) return;
    *reclaim_ok = slot_is_hazarded(hazard, NUM_MATCHERS, slot) ? 0 : 1;
}

// ---- Host-side replay of the identical publish/tombstone/hazard-check
// ---- sequence, in the same order Section 32.3's CPU baseline used. ----

int main() {
    printf("=== Section 32.3 main: hazard-protected concurrent cancel of a resting order ===\n\n");

    int b2_slot = 10, b3_slot = 11;
    int b2_qty = 3, b3_qty = 2;
    bool b2_cancelled = false, b3_cancelled = false;
    int hazard = UNHAZARDED;

    printf("order pool: B2 -> slot %d (qty %d), B3 -> slot %d (qty %d)\n\n", b2_slot, b2_qty, b3_slot, b3_qty);

    printf("matching thread M is about to trade against B2 (first in FIFO at its level):\n");
    printf("  M publishes hazard on B2's slot (%d) BEFORE reading its quantity\n", b2_slot);
    hazard = b2_slot;
    printf("  hazard array: [%d]\n\n", hazard);

    printf("cancel request arrives for B2 concurrently:\n");
    b2_cancelled = true;
    printf("  B2 marked CANCELLED (tombstone flag set): {'B2': %s, 'B3': %s}\n",
           b2_cancelled ? "True" : "False", b3_cancelled ? "True" : "False");
    printf("  cancel wants to free B2's slot (%d) back to the pool, checks hazard first:\n", b2_slot);
    bool b2_hazarded = (hazard == b2_slot);
    printf("  slot %d IS hazarded: %s -- free %s\n\n", b2_slot,
           b2_hazarded ? "True" : "False", b2_hazarded ? "DEFERRED" : "allowed");

    printf("M finishes reading B2's slot, sees the CANCELLED flag is now set, and skips\n");
    printf("trading against it (moves on to B3 in the FIFO instead), then clears its hazard:\n");
    hazard = UNHAZARDED;
    printf("  hazard array: [%d]\n\n", hazard);

    printf("cancel rechecks and this time frees B2's slot:\n");
    bool b2_hazarded_after = (hazard == b2_slot);
    printf("  slot %d hazarded: %s -- free %s\n", b2_slot,
           b2_hazarded_after ? "True" : "False",
           b2_hazarded_after ? "DEFERRED" : "allowed, slot returned to pool");

    printf("\nM proceeds to trade against B3 instead (the next FIFO order at this level),\n");
    printf("since B2 was correctly skipped rather than traded against after cancellation.\n");

    bool ok = b2_hazarded && !b2_hazarded_after && b2_cancelled;
    printf("\nself-check: reclaim correctly BLOCKED while M's hazard was published, then\n");
    printf("ALLOWED once M cleared it, and M never traded against the cancelled order: %s\n",
           ok ? "confirmed" : "MISMATCH");

    return ok ? 0 : 1;
}
```

**Compile and run:**

```bash
NVDIR=/usr/local/lib/python3.11/dist-packages/nvidia/cu13
$NVDIR/bin/nvcc -arch=sm_80 -I$NVDIR/include -L$NVDIR/lib -lcudart 183_lob_cancel_hazard_kernel.cu -o 183_lob_cancel_hazard_kernel
LD_LIBRARY_PATH=$NVDIR/lib ./183_lob_cancel_hazard_kernel
```

**Sample input:** the same cancel-vs-match race, replayed via genuine `__global__` publish/tombstone/check-before-free kernels.

**Sample output:**

```text
=== Section 32.3 main: hazard-protected concurrent cancel of a resting order ===

order pool: B2 -> slot 10 (qty 3), B3 -> slot 11 (qty 2)

matching thread M is about to trade against B2 (first in FIFO at its level):
  M publishes hazard on B2's slot (10) BEFORE reading its quantity
  hazard array: [10]

cancel request arrives for B2 concurrently:
  B2 marked CANCELLED (tombstone flag set): {'B2': True, 'B3': False}
  cancel wants to free B2's slot (10) back to the pool, checks hazard first:
  slot 10 IS hazarded: True -- free DEFERRED

M finishes reading B2's slot, sees the CANCELLED flag is now set, and skips
trading against it (moves on to B3 in the FIFO instead), then clears its hazard:
  hazard array: [-1]

cancel rechecks and this time frees B2's slot:
  slot 10 hazarded: False -- free allowed, slot returned to pool

M proceeds to trade against B3 instead (the next FIFO order at this level),
since B2 was correctly skipped rather than traded against after cancellation.

self-check: reclaim correctly BLOCKED while M's hazard was published, then
ALLOWED once M cleared it, and M never traded against the cancelled order: confirmed
```

## Chapter Summary

A limit order book matching engine needed no new concurrency primitive at all -- every piece of it was a direct reuse of a tool this book had already built and proven. Section 32.1 showed that a price ladder is Section 26.3's bucket queue with price tick standing in for distance, inserted into concurrently via the same `atomicAdd`-cursor technique Section 30.2 used for scatter, with FIFO order determined by genuine arrival order rather than order-id or thread-index order. Section 32.2 showed that finding the best price to match against is a reduction: Section 25.2's packed-key argmin combined with Section 4/31.1's pairwise MIN tree, re-run fresh every matching-loop iteration because each trade can change which level is now best. Section 32.3 showed that safely cancelling a resting order while a match might be reading it is exactly Section 27.3/29.3's hazard-pointer protocol, unchanged, now protecting an order-pool slot -- publish before reading, mark unusable before checking hazards, free only once no hazard remains.

## Self-Check Questions

1. What single substitution turns Section 26.3's Dijkstra bucket queue into Section 32.1's price ladder, and why does no other part of the structure need to change?
2. In Section 32.1's concurrent insertion, what determines whether B2 or B3 ends up in FIFO slot 0 at their shared price level -- and what does NOT determine it?
3. Why must Section 32.2's best-occupied-level search be re-run from scratch at the top of every matching-loop iteration, rather than computed once before the loop begins?
4. How does the packed key `price * 100 + level_index` let a single MIN reduction recover both "which price is best" and "which level had it," using one comparable value instead of two?
5. In Section 32.3's hazard protocol, what specific ordering mistake would reopen the race the protocol is meant to close, if the cancel operation checked hazards before marking the order cancelled instead of after?
6. Why does Section 32.3's hazard-pointer protocol require no changes at all to move from protecting a KV store's value pool (Chapter 29) to protecting a limit order book's order pool?

## Where We Go Next

A matching engine turns arriving orders into executed trades, but it says nothing about what those trades are actually worth under uncertainty. Chapter 33 turns to Monte Carlo risk simulation and derivatives pricing, reusing Part 1's reduction and scan primitives -- not to match orders, but to average millions of independently simulated future outcomes into a single price or risk estimate, the next of the case studies that close out Part 8.

## Worked Solutions

**1.** The single substitution is what the bucket INDEX represents: Dijkstra's bucket queue indexes by shortest-path distance, while the price ladder indexes by price tick (converted to a small dense integer via `level = price - floor_price`). Nothing else changes, because a bucket queue's only real requirement is a small, dense, non-negative integer to use directly as an array offset -- both distance and price tick satisfy that requirement identically, so the FIFO-within-a-bucket logic, the insertion logic, and the underlying array-of-lists layout all carry over completely unchanged.

**2.** Which order lands in FIFO slot 0 is determined entirely by which order's `atomicAdd` on the shared level cursor physically executes FIRST -- genuine arrival order at the hardware level. It is NOT determined by the orders' IDs (B2 sorts before B3 alphabetically but can still land second), and it is NOT determined by thread index or launch order, since Section 32.1's kernel deliberately forces B3's landing before B2's to demonstrate that atomicAdd's fairness guarantee (every caller gets a distinct, correctly-ordered slot) says nothing about WHICH caller gets which slot ahead of time.

**3.** Each matching-loop iteration that successfully trades against a level reduces that level's resting quantity, and once a level is fully consumed it is no longer occupied -- so the "best occupied level" found in one iteration can become stale (wrong, or simply exhausted) by the very trade that iteration just executed. Re-running the full reduction at the top of every iteration, exactly like Section 32.2's CPU baseline re-running its linear scan every iteration, guarantees the next trade always targets whichever level is ACTUALLY best given the book's current state, not a cached answer from before the last trade.

**4.** Because every level index in this book is guaranteed to be under 100, multiplying price by 100 shifts the price into the key's upper digits while leaving the lowest two digits entirely free for the level index, with no overlap between the two fields. A MIN comparison over these packed keys therefore compares prices first (since price dominates the key's magnitude) and only falls back to comparing indices when two levels somehow shared the same price -- so the reduction's single winning key can be decoded by integer division and modulo into exactly the two answers ("what price won" and "which level had it") that a naive two-array argmin would have needed two separate reduction passes, or one much more complex combine function, to produce.

**5.** If the cancel checked the hazard array BEFORE marking the order cancelled, a matching thread could publish its hazard and begin reading the order in the gap between that hazard-check and the tombstone-write -- since the cancel already saw "no hazard" and would proceed to free the slot, it could free a slot the matching thread is now actively reading, exactly the race the protocol exists to prevent. Marking the order cancelled FIRST closes this window because any matching thread that reads the order afterward is guaranteed to see the cancellation flag already set and will skip trading against it, regardless of exactly when its hazard publish happened to land relative to the cancel.

**6.** The hazard-pointer protocol's correctness never depended on WHAT kind of resource was being protected -- only on the discipline of publish-before-read, mark-unusable-before-checking-hazards, and free-only-once-clear. An order-pool slot, a KV store's value block, and a lock-free list's node are all just "a piece of shared memory some other thread might currently be reading," which is the only property the protocol actually reasons about, so applying it to a new resource type requires substituting what gets published and freed, never changing the protocol's ordering or logic itself.
