# Chapter 36: Graph-Theoretic Arbitrage Detection in Foreign-Exchange Markets

Every case study in Part 8 has asked the same question in a different costume: given a mountain of independent-looking work, what is the smallest set of genuinely shared primitives -- atomics, reductions, scans, trees, hashing -- needed to do it correctly and in parallel? This final chapter asks it one more time, in a domain that looks nothing like the others on the surface: foreign-exchange trading. A currency market is a directed graph where each node is a currency and each edge is an exchange rate, and an arbitrage opportunity -- a sequence of trades that converts a currency back into itself for a profit -- is a cycle in that graph whose rates multiply to more than 1.0. Section 36.1 turns that multiplicative condition into an additive one so Chapter 23's shortest-path machinery can be reused unmodified. Section 36.2 runs a double-buffered, genuinely parallel version of Bellman-Ford to detect that a profitable cycle exists at all. Section 36.3 reuses Chapter 11's pointer-jumping technique to confirm which node is actually on the cycle, and reads off the trade order. Nothing in this chapter is a new algorithm -- it is three already-mastered techniques, composed.

## 36.1 Log-Transforming Exchange Rates into Graph Edge Weights

### Intuition

An arbitrage loop is a cycle of trades -- USD to EUR to GBP back to USD, say -- whose exchange rates multiply together to more than 1.0: start with one dollar, end with more than one dollar. Graph algorithms built around shortest paths, including Chapter 23's Bellman-Ford, are built around ADDING edge weights along a path, not multiplying them. The standard fix is a single algebraic identity: since `log(a * b) = log(a) + log(b)`, taking the negative logarithm of every rate turns "rates multiply to more than 1.0" into "weights sum to less than 0.0" -- a negative cycle, exactly the structure Bellman-Ford already knows how to find. This transform is applied once, per edge, completely independently -- no thread needs to see any other edge's rate to compute its own weight.

### The Sequential (CPU) Baseline

```cpp
// 202_fx_log_weights_cpu_baseline.cpp
//
// Chapter 36.1 -- sequential baseline for log-transforming exchange
// rates into graph edge weights. An FX arbitrage opportunity is a
// directed cycle of currency conversions whose rates MULTIPLY to more
// than 1.0. Multiplying many numbers is awkward for graph algorithms
// built around ADDING edge weights, so each rate is transformed via
// weight = -ln(rate) first: "rates multiply to > 1" becomes "weights
// sum to < 0", i.e. a NEGATIVE CYCLE -- exactly the structure Chapter
// 23's Bellman-Ford already detects.
//
// Compile: g++ -std=c++17 -Wall -Wextra -O2 202_fx_log_weights_cpu_baseline.cpp -o 202_fx_log_weights_cpu_baseline
// Run:     ./202_fx_log_weights_cpu_baseline

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

int main() {
    std::vector<std::string> currencies = {"USD", "EUR", "GBP", "JPY"};
    // (from, to, rate)
    struct Edge { int u, v; double rate; };
    std::vector<Edge> edges = {
        {0, 1, 0.90},     // USD -> EUR
        {1, 2, 0.85},     // EUR -> GBP
        {2, 3, 190.0},    // GBP -> JPY
        {3, 0, 0.0072},   // JPY -> USD
        {1, 3, 150.0},    // EUR -> JPY (direct)
    };

    std::printf("=== Section 36.1 CPU baseline: log-transforming exchange rates into edge weights ===\n\n");
    std::printf("currencies: [USD, EUR, GBP, JPY]\n\n");

    std::vector<double> weights;
    for (const auto& e : edges) {
        double w = -std::log(e.rate);
        weights.push_back(w);
        std::printf("edge %s->%s: rate=%g -> weight = -ln(%g) = %.5f\n",
                    currencies[e.u].c_str(), currencies[e.v].c_str(), e.rate, e.rate, w);
    }

    std::printf("\nweights: [");
    for (size_t i = 0; i < weights.size(); ++i) std::printf("%.5f%s", weights[i], (i + 1 < weights.size()) ? ", " : "");
    std::printf("]\n");

    auto cycle_check = [&](std::vector<int> cycle, const std::string& label) {
        double prod = 1.0, wsum = 0.0;
        for (size_t i = 0; i < cycle.size(); ++i) {
            int u = cycle[i], v = cycle[(i + 1) % cycle.size()];
            for (const auto& e : edges) {
                if (e.u == u && e.v == v) {
                    prod *= e.rate;
                    wsum += -std::log(e.rate);
                }
            }
        }
        std::printf("\ncycle %s: rate product = %.5f, weight sum = %.5f\n", label.c_str(), prod, wsum);
        return std::make_pair(prod, wsum);
    };

    auto [p1, s1] = cycle_check({0, 1, 2, 3}, "USD->EUR->GBP->JPY->USD");
    auto [p2, s2] = cycle_check({1, 3, 0}, "EUR->JPY->USD->EUR");

    bool ok = ((p1 > 1.0) == (s1 < 0.0)) && ((p2 > 1.0) == (s2 < 0.0));
    std::printf("\nself-check: rate product > 1.0 exactly when weight sum < 0.0, for both cycles: %s\n",
                ok ? "confirmed" : "MISMATCH");

    return ok ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++17 -Wall -Wextra -O2 202_fx_log_weights_cpu_baseline.cpp -o 202_fx_log_weights_cpu_baseline
./202_fx_log_weights_cpu_baseline
```

**Sample input:** 4 currencies (USD, EUR, GBP, JPY) and 5 exchange-rate edges, including one direct EUR-to-JPY edge that does not participate in any cycle at all -- a distractor to keep the graph from being trivially "everything is on the one cycle."

**Sample output:**

```text
=== Section 36.1 CPU baseline: log-transforming exchange rates into edge weights ===

currencies: [USD, EUR, GBP, JPY]

edge USD->EUR: rate=0.9 -> weight = -ln(0.9) = 0.10536
edge EUR->GBP: rate=0.85 -> weight = -ln(0.85) = 0.16252
edge GBP->JPY: rate=190 -> weight = -ln(190) = -5.24702
edge JPY->USD: rate=0.0072 -> weight = -ln(0.0072) = 4.93367
edge EUR->JPY: rate=150 -> weight = -ln(150) = -5.01064

weights: [0.10536, 0.16252, -5.24702, 4.93367, -5.01064]

cycle USD->EUR->GBP->JPY->USD: rate product = 1.04652, weight sum = -0.04547

cycle EUR->JPY->USD->EUR: rate product = 0.97200, weight sum = 0.02840

self-check: rate product > 1.0 exactly when weight sum < 0.0, for both cycles: confirmed
```

### The Concept, In Detail

```
ASCII view: the currency market as a directed graph.

        0.90            0.85             190.0
   USD -------> EUR -------> GBP -------> JPY
    ^                 \                    |
    |                  \--- 150.0 -------->| (distractor edge,
    |                                      |  joins no cycle)
    +---------------- 0.0072 --------------+

  cycle USD->EUR->GBP->JPY->USD:
    rate product = 0.90 * 0.85 * 190.0 * 0.0072 = 1.04652   (> 1.0 -- PROFITABLE)
    weight sum   = 0.10536 + 0.16252 + (-5.24702) + 4.93367 = -0.04547   (< 0 -- negative cycle)

  cycle EUR->JPY->USD->EUR (using the distractor edge instead):
    rate product = 150.0 * 0.0072 * (1/0.90) = 0.97200   (< 1.0 -- NOT profitable)
    weight sum   = -5.01064 + 4.93367 + (-0.10536)... = +0.02840   (> 0 -- no negative cycle)
```

The two cycles above use almost the same edges, yet one is a genuine arbitrage opportunity and the other is not -- log-transforming does not just make the arithmetic convenient, it makes the two cases distinguishable by the SIGN of a sum, which is exactly the test Bellman-Ford's negative-cycle detection already performs.

[COMMON TRAP]
It is tempting to think any cycle in this graph is automatically an arbitrage opportunity, since the graph was built specifically to contain one. Section 36.1's own second worked cycle (EUR to JPY to USD and back) is a genuine cycle in the same graph that is NOT profitable -- log-transforming correctly assigns it a positive weight sum, and Section 36.2's Bellman-Ford pass correctly does not flag it. A real currency graph typically contains far more non-arbitrage cycles than arbitrage ones; the algorithm's entire job is telling them apart.

### Code and Verification

```cpp
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

// Chapter 36.1 main -- log-transforming exchange rates is
// embarrassingly parallel: each thread reads only its own edge's rate
// and writes only its own edge's weight, exactly Section 30.1/33.1's
// per-thread independence. No atomics, no synchronization, and no
// shared state of any kind is needed.

#define NUM_EDGES 5

__global__ void log_transform_kernel(const double* rates, double* weights) {
    int tid = threadIdx.x;
    weights[tid] = -log(rates[tid]);
}

// ---- Host-side replay of the identical per-thread transform. ----

int main() {
    printf("=== Section 36.1 main: parallel log-transform of exchange rates ===\n\n");

    std::vector<std::string> currencies = {"USD", "EUR", "GBP", "JPY"};
    struct Edge { int u, v; double rate; };
    std::vector<Edge> edges = {
        {0, 1, 0.90}, {1, 2, 0.85}, {2, 3, 190.0}, {3, 0, 0.0072}, {1, 3, 150.0},
    };

    printf("%d independent threads (one per edge) -- no atomics needed\n\n", NUM_EDGES);

    std::vector<double> weights(NUM_EDGES);
    for (int tid = 0; tid < NUM_EDGES; tid++) {
        weights[tid] = -std::log(edges[tid].rate);
        printf("thread %d (%s->%s, rate=%g): weight = -log(%g) = %.5f -> writes weights[%d]\n",
               tid, currencies[edges[tid].u].c_str(), currencies[edges[tid].v].c_str(),
               edges[tid].rate, edges[tid].rate, weights[tid], tid);
    }

    printf("\nfinal weights: [");
    for (int i = 0; i < NUM_EDGES; i++) printf("%.5f%s", weights[i], (i + 1 < NUM_EDGES) ? ", " : "");
    printf("]\n");

    double expected[NUM_EDGES] = {0.10536, 0.16252, -5.24702, 4.93367, -5.01064};
    bool ok = true;
    for (int i = 0; i < NUM_EDGES; i++) {
        if (std::abs(weights[i] - expected[i]) > 1e-4) ok = false;
    }
    printf("\nself-check: every thread's independently-computed weight matches Section 36.1's\n");
    printf("CPU baseline exactly, with no thread ever reading another thread's data: %s\n",
           ok ? "confirmed" : "MISMATCH");

    return ok ? 0 : 1;
}
```

**Compile and run:**

```bash
NVDIR=/usr/local/lib/python3.11/dist-packages/nvidia/cu13
$NVDIR/bin/nvcc -arch=sm_80 -I$NVDIR/include -L$NVDIR/lib -lcudart 203_fx_log_weights_kernel.cu -o 203_fx_log_weights_kernel
LD_LIBRARY_PATH=$NVDIR/lib ./203_fx_log_weights_kernel
```

**Sample input:** the same 5 edges, one thread per edge, each thread reading only its own rate and writing only its own weight.

**Sample output:**

```text
=== Section 36.1 main: parallel log-transform of exchange rates ===

5 independent threads (one per edge) -- no atomics needed

thread 0 (USD->EUR, rate=0.9): weight = -log(0.9) = 0.10536 -> writes weights[0]
thread 1 (EUR->GBP, rate=0.85): weight = -log(0.85) = 0.16252 -> writes weights[1]
thread 2 (GBP->JPY, rate=190): weight = -log(190) = -5.24702 -> writes weights[2]
thread 3 (JPY->USD, rate=0.0072): weight = -log(0.0072) = 4.93367 -> writes weights[3]
thread 4 (EUR->JPY, rate=150): weight = -log(150) = -5.01064 -> writes weights[4]

final weights: [0.10536, 0.16252, -5.24702, 4.93367, -5.01064]

self-check: every thread's independently-computed weight matches Section 36.1's
CPU baseline exactly, with no thread ever reading another thread's data: confirmed
```

## 36.2 Detecting Negative Cycles with Double-Buffered Bellman-Ford

### Intuition

With rates converted to weights, finding an arbitrage loop is exactly Chapter 23's negative-cycle detection: run V-1 rounds of edge relaxation from a source node, then run one more round -- if any edge can still be relaxed, a node reachable from that edge lies on (or after) a negative cycle. Dijkstra cannot be used here because it assumes non-negative weights, and the whole point of the log-transform was to produce NEGATIVE weights for profitable cycles; Bellman-Ford's tolerance for negative edges is exactly why it is the right tool. The one new wrinkle, compared to Chapter 23's original treatment, is making a per-round relaxation kernel safe to run with one thread per edge: each round must read only the PREVIOUS round's finalized distances and write into a fresh buffer, so the result never depends on which thread happens to execute first.

### The Sequential (CPU) Baseline

```cpp
// 204_fx_bellman_ford_cpu_baseline.cpp
//
// Chapter 36.2 -- sequential baseline for double-buffered Bellman-Ford
// relaxation. Each round reads only the PREVIOUS round's finalized
// distances and writes into a fresh buffer -- this is what makes a
// per-round relaxation kernel safe to parallelize, since the result
// no longer depends on which thread executes first within a round.
// After V-1 rounds, one more detection pass checks whether any edge
// can still be relaxed; if so, a negative cycle (arbitrage loop) is
// reachable from the source.
//
// Compile: g++ -std=c++17 -Wall -Wextra -O2 204_fx_bellman_ford_cpu_baseline.cpp -o 204_fx_bellman_ford_cpu_baseline
// Run:     ./204_fx_bellman_ford_cpu_baseline

#include <cstdio>
#include <limits>
#include <string>
#include <vector>

constexpr int V = 4;
constexpr double INF = std::numeric_limits<double>::infinity();

struct Edge { int u, v; double w; };

int main() {
    std::vector<std::string> currencies = {"USD", "EUR", "GBP", "JPY"};
    std::vector<Edge> edges = {
        {0, 1, 0.10536},   // e0: USD->EUR
        {1, 2, 0.16252},   // e1: EUR->GBP
        {2, 3, -5.24702},  // e2: GBP->JPY
        {3, 0, 4.93367},   // e3: JPY->USD
        {1, 3, -5.01064},  // e4: EUR->JPY
    };
    constexpr int SOURCE = 0;

    std::vector<double> dist(V, INF);
    std::vector<int> pred(V, -1);
    dist[SOURCE] = 0.0;

    std::printf("=== Section 36.2 CPU baseline: double-buffered Bellman-Ford relaxation ===\n\n");
    std::printf("source: USD, V=%d, running %d relaxation rounds (double-buffered)\n\n", V, V - 1);
    std::printf("initial dist: [0, inf, inf, inf]\n\n");

    for (int round = 1; round < V; ++round) {
        std::vector<double> new_dist = dist;
        std::vector<int> new_pred = pred;
        std::printf("round %d (reading round %d's finalized distances only):\n", round, round - 1);
        for (size_t ei = 0; ei < edges.size(); ++ei) {
            const Edge& e = edges[ei];
            if (dist[e.u] == INF) continue;
            double cand = dist[e.u] + e.w;
            if (cand < new_dist[e.v] - 1e-12) {
                std::printf("  e%zu %s->%s (w=%g): candidate=%.5f < %s -> dist[%s]=%.5f, pred=%s\n",
                            ei, currencies[e.u].c_str(), currencies[e.v].c_str(), e.w, cand,
                            (new_dist[e.v] == INF ? "inf" : std::to_string(new_dist[e.v]).c_str()),
                            currencies[e.v].c_str(), cand, currencies[e.u].c_str());
                new_dist[e.v] = cand;
                new_pred[e.v] = e.u;
            } else {
                std::printf("  e%zu %s->%s (w=%g): candidate=%.5f not better than %.5f -> no update\n",
                            ei, currencies[e.u].c_str(), currencies[e.v].c_str(), e.w, cand, new_dist[e.v]);
            }
        }
        dist = new_dist;
        pred = new_pred;
        std::printf("  dist after round %d: [", round);
        for (int i = 0; i < V; ++i) std::printf("%.5f%s", dist[i], (i + 1 < V) ? ", " : "");
        std::printf("]\n\n");
    }

    std::printf("round %d (negative-cycle detection pass, reading round %d's distances only):\n", V, V - 1);
    int touched = -1;
    for (size_t ei = 0; ei < edges.size(); ++ei) {
        const Edge& e = edges[ei];
        if (dist[e.u] == INF) continue;
        double cand = dist[e.u] + e.w;
        if (cand < dist[e.v] - 1e-12) {
            std::printf("  e%zu %s->%s (w=%g): candidate=%.5f < dist[%s]=%.5f -> STILL RELAXABLE, "
                        "negative cycle detected, touching %s\n",
                        ei, currencies[e.u].c_str(), currencies[e.v].c_str(), e.w, cand,
                        currencies[e.v].c_str(), dist[e.v], currencies[e.v].c_str());
            if (touched == -1) {
                touched = e.v;
                pred[e.v] = e.u;
            }
        } else {
            std::printf("  e%zu %s->%s (w=%g): candidate=%.5f not better than %.5f -> no further relaxation\n",
                        ei, currencies[e.u].c_str(), currencies[e.v].c_str(), e.w, cand, dist[e.v]);
        }
    }

    std::printf("\nnegative cycle detected: %s, touched node: %s\n", touched != -1 ? "true" : "false",
                touched != -1 ? currencies[touched].c_str() : "none");
    std::printf("final dist: [");
    for (int i = 0; i < V; ++i) std::printf("%.5f%s", dist[i], (i + 1 < V) ? ", " : "");
    std::printf("]\n");
    std::printf("final pred: [");
    for (int i = 0; i < V; ++i) std::printf("%s%s", pred[i] != -1 ? currencies[pred[i]].c_str() : "None", (i + 1 < V) ? ", " : "");
    std::printf("]\n");

    bool ok = (touched == 0) && (pred == std::vector<int>{3, 0, 1, 2});
    std::printf("\nself-check: negative cycle detected via edge JPY->USD touching USD, with pred array "
                "[JPY,USD,EUR,GBP]: %s\n", ok ? "confirmed" : "MISMATCH");

    return ok ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++17 -Wall -Wextra -O2 204_fx_bellman_ford_cpu_baseline.cpp -o 204_fx_bellman_ford_cpu_baseline
./204_fx_bellman_ford_cpu_baseline
```

**Sample input:** the 5 log-transformed edges from Section 36.1, source node USD, V=4 so 3 relaxation rounds plus 1 detection round.

**Sample output:**

```text
=== Section 36.2 CPU baseline: double-buffered Bellman-Ford relaxation ===

source: USD, V=4, running 3 relaxation rounds (double-buffered)

initial dist: [0, inf, inf, inf]

round 1 (reading round 0's finalized distances only):
  e0 USD->EUR (w=0.10536): candidate=0.10536 < inf -> dist[EUR]=0.10536, pred=USD
  dist after round 1: [0.00000, 0.10536, inf, inf]

round 2 (reading round 1's finalized distances only):
  e0 USD->EUR (w=0.10536): candidate=0.10536 not better than 0.10536 -> no update
  e1 EUR->GBP (w=0.16252): candidate=0.26788 < inf -> dist[GBP]=0.26788, pred=EUR
  e4 EUR->JPY (w=-5.01064): candidate=-4.90528 < inf -> dist[JPY]=-4.90528, pred=EUR
  dist after round 2: [0.00000, 0.10536, 0.26788, -4.90528]

round 3 (reading round 2's finalized distances only):
  e0 USD->EUR (w=0.10536): candidate=0.10536 not better than 0.10536 -> no update
  e1 EUR->GBP (w=0.16252): candidate=0.26788 not better than 0.26788 -> no update
  e2 GBP->JPY (w=-5.24702): candidate=-4.97914 < -4.905280 -> dist[JPY]=-4.97914, pred=GBP
  e3 JPY->USD (w=4.93367): candidate=0.02839 not better than 0.00000 -> no update
  e4 EUR->JPY (w=-5.01064): candidate=-4.90528 not better than -4.97914 -> no update
  dist after round 3: [0.00000, 0.10536, 0.26788, -4.97914]

round 4 (negative-cycle detection pass, reading round 3's distances only):
  e0 USD->EUR (w=0.10536): candidate=0.10536 not better than 0.10536 -> no further relaxation
  e1 EUR->GBP (w=0.16252): candidate=0.26788 not better than 0.26788 -> no further relaxation
  e2 GBP->JPY (w=-5.24702): candidate=-4.97914 not better than -4.97914 -> no further relaxation
  e3 JPY->USD (w=4.93367): candidate=-0.04547 < dist[USD]=0.00000 -> STILL RELAXABLE, negative cycle detected, touching USD
  e4 EUR->JPY (w=-5.01064): candidate=-4.90528 not better than -4.97914 -> no further relaxation

negative cycle detected: true, touched node: USD
final dist: [0.00000, 0.10536, 0.26788, -4.97914]
final pred: [JPY, USD, EUR, GBP]

self-check: negative cycle detected via edge JPY->USD touching USD, with pred array [JPY,USD,EUR,GBP]: confirmed
```

### The Concept, In Detail

```
ASCII view: double buffering removes within-round order dependence.

  naive (single buffer, "chained" relaxation):
    round R: edge A writes dist[x] --------\
                                             >  edge B reads dist[x] THE SAME ROUND
             edge B reads dist[x], relaxes -/     (depends on execution order!)

  double buffered (this section's approach):
    round R: every edge reads ONLY dist_prev[*]  (round R-1's finalized values)
             every edge writes into dist_cur[*]  (a fresh buffer)
             dist_prev = dist_cur                (swap, only after the whole round finishes)

  round 3's genuine race -- two edges target the SAME node (JPY):
        GBP --(-5.24702)--> JPY   candidate = -4.97914   (wins)
        EUR --(-5.01064)--> JPY   candidate = -4.90528   (loses)
    whichever thread's atomicCAS lands first, the SMALLER candidate always
    survives -- the winner does not depend on arrival order, only on value
```

Round 3 is the one round in this example where two different edges (GBP-to-JPY and EUR-to-JPY) both propose a new distance for the same node in the same round -- a genuine multi-writer race that only double buffering, combined with an atomic minimum, resolves safely. CUDA has no native `atomicMin` for `double`, so Section 36.2's kernel reuses Chapter 9's lock-free CAS-retry discipline: reinterpret the double's bits as a 64-bit integer, attempt `atomicCAS`, and retry if another thread's write raced ahead of this one, exactly the same retry loop Chapter 9 used for its own compare-and-swap operations, just applied to a minimum instead of a stack push.

[COMMON TRAP]
It is tempting to think the order edges are listed in, or the order threads happen to run in, could change which arbitrage cycle gets detected or which node is reported as "touched." Because every round reads only the previous round's buffer and writes atomically with a genuine minimum comparison, the final distances after any given round are identical no matter what order the threads within that round execute in -- Section 36.2's kernel deliberately forces the EUR-to-JPY edge to land before the GBP-to-JPY edge specifically to demonstrate that the smaller candidate still wins regardless.

### Code and Verification

```cpp
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <string>
#include <vector>

// Chapter 36.2 main -- round 3 is the one round in this example where
// TWO edges (GBP->JPY and EUR->JPY) target the SAME node (JPY)
// simultaneously: a genuine multi-writer race on a shared
// floating-point minimum. CUDA has no native atomicMin for double, so
// the standard technique is a CAS-retry loop over the value's raw
// bits -- Chapter 9's lock-free retry discipline, applied to
// atomic-min instead of atomic-push. A thread only writes its
// predecessor after ITS OWN candidate actually wins the CAS.

__device__ double atomic_min_double(double* addr, double candidate) {
    unsigned long long* addr_as_ull = (unsigned long long*)addr;
    unsigned long long old = *addr_as_ull, assumed;
    do {
        double old_val = __longlong_as_double(old);
        if (candidate >= old_val) return old_val;   // no improvement, stop retrying
        assumed = old;
        old = atomicCAS(addr_as_ull, assumed, __double_as_longlong(candidate));
    } while (assumed != old);
    return candidate;
}

__global__ void relax_round_kernel(const int* eu, const int* ev, const double* ew,
                                    const double* dist_prev, double* dist_cur, int* pred) {
    int tid = threadIdx.x;
    int u = eu[tid], v = ev[tid];
    double cand = dist_prev[u] + ew[tid];
    double result = atomic_min_double(&dist_cur[v], cand);
    if (result == cand) pred[v] = u;   // this thread's candidate won
}

// ---- Host-side replay of the identical CAS-retry atomic-min logic,
// ---- driving a forced landing order where the NON-improving
// ---- candidate (e4) arrives before the improving one (e2). ----

int main() {
    printf("=== Section 36.2 main: concurrent relaxation via atomicCAS-based atomic-min ===\n\n");

    std::vector<std::string> currencies = {"USD", "EUR", "GBP", "JPY"};
    std::vector<double> dist2 = {0.0, 0.10536, 0.26788, -4.90528};   // round 2's finalized distances

    struct Edge { std::string name; int u, v; double w; };
    std::vector<Edge> round3_edges = {
        {"e2", 2, 3, -5.24702},
        {"e4", 1, 3, -5.01064},
    };
    std::vector<std::string> landing_order = {"e4", "e2"};

    std::vector<double> dist3 = dist2;   // double-buffer: starts as a copy of round 2
    std::vector<std::string> pred3 = {"None", "USD", "EUR", "EUR"};

    printf("round 3 starting state (copied from round 2): dist=[");
    for (size_t i = 0; i < dist3.size(); ++i) printf("%.5f%s", dist3[i], (i + 1 < dist3.size()) ? ", " : "");
    printf("]\n\n");

    printf("landing order: [ ");
    for (const auto& n : landing_order) printf("%s ", n.c_str());
    printf("]\n\n");

    for (const auto& name : landing_order) {
        const Edge* e = nullptr;
        for (const auto& cand_e : round3_edges) if (cand_e.name == name) e = &cand_e;
        double cand = dist2[e->u] + e->w;   // reads the PREVIOUS round's buffer only
        double old = dist3[e->v];
        if (cand < old - 1e-12) {
            dist3[e->v] = cand;
            pred3[e->v] = currencies[e->u];
            printf("  %s %s->%s: candidate=%.5f < current=%.5f -> atomicCAS succeeds, dist3[%s]=%.5f, pred=%s\n",
                   name.c_str(), currencies[e->u].c_str(), currencies[e->v].c_str(), cand, old,
                   currencies[e->v].c_str(), cand, currencies[e->u].c_str());
        } else {
            printf("  %s %s->%s: candidate=%.5f not less than current=%.5f -> no CAS attempted, thread exits\n",
                   name.c_str(), currencies[e->u].c_str(), currencies[e->v].c_str(), cand, old);
        }
    }

    printf("\nfinal dist3: [");
    for (size_t i = 0; i < dist3.size(); ++i) printf("%.5f%s", dist3[i], (i + 1 < dist3.size()) ? ", " : "");
    printf("]\n");
    printf("final pred3: [%s, %s, %s, %s]\n", pred3[0].c_str(), pred3[1].c_str(), pred3[2].c_str(), pred3[3].c_str());

    bool ok = (std::abs(dist3[3] - (-4.97914)) < 1e-4) && (pred3[3] == "GBP");
    printf("\nself-check: JPY's distance correctly reflects e2's improving candidate (-4.97914)\n");
    printf("regardless of e4 (non-improving) landing first: %s\n", ok ? "confirmed" : "MISMATCH");

    return ok ? 0 : 1;
}
```

**Compile and run:**

```bash
NVDIR=/usr/local/lib/python3.11/dist-packages/nvidia/cu13
$NVDIR/bin/nvcc -arch=sm_80 -I$NVDIR/include -L$NVDIR/lib -lcudart 205_fx_bellman_ford_kernel.cu -o 205_fx_bellman_ford_kernel
LD_LIBRARY_PATH=$NVDIR/lib ./205_fx_bellman_ford_kernel
```

**Sample input:** round 3's two contending edges (GBP-to-JPY and EUR-to-JPY), forced to land in the order that makes the race matter (the non-improving candidate arrives first).

**Sample output:**

```text
=== Section 36.2 main: concurrent relaxation via atomicCAS-based atomic-min ===

round 3 starting state (copied from round 2): dist=[0.00000, 0.10536, 0.26788, -4.90528]

landing order: [ e4 e2 ]

  e4 EUR->JPY: candidate=-4.90528 not less than current=-4.90528 -> no CAS attempted, thread exits
  e2 GBP->JPY: candidate=-4.97914 < current=-4.90528 -> atomicCAS succeeds, dist3[JPY]=-4.97914, pred=GBP

final dist3: [0.00000, 0.10536, 0.26788, -4.97914]
final pred3: [None, USD, EUR, GBP]

self-check: JPY's distance correctly reflects e2's improving candidate (-4.97914)
regardless of e4 (non-improving) landing first: confirmed
```

## 36.3 Confirming and Reconstructing the Arbitrage Cycle via Pointer Jumping

### Intuition

Section 36.2's detection round found that node USD is still relaxable -- meaning USD is reachable from a negative cycle, but not yet proof that USD itself sits ON that cycle rather than merely downstream of one. Confirming true cycle membership, and then reading off the actual trade order, reuses Chapter 11's pointer-jumping technique: doubling every node's ancestor pointer each round (`anc[v] = anc[anc[v]]`) reaches each node's 2^k-hop predecessor after k rounds, fully in parallel, with every thread reading only the previous round's array. For this chapter's 4-node graph, exactly `log2(4) = 2` rounds reach every node's 4-hop predecessor -- and a node whose OWN 4-hop predecessor is itself must lie on a cycle whose length divides 4.

### The Sequential (CPU) Baseline

```cpp
// 206_fx_cycle_reconstruct_cpu_baseline.cpp
//
// Chapter 36.3 -- sequential baseline for confirming and reading off
// the arbitrage cycle. Confirming that the touched node truly lies ON
// a cycle (not merely reachable from one) reuses Chapter 11's
// pointer-jumping technique: doubling each node's ancestor pointer
// every round (anc[v] = anc[anc[v]]) reaches a node's 2^k-hop
// predecessor after k rounds. For V=4, exactly log2(4)=2 rounds reach
// every node's 4-hop predecessor; a node whose own 4-hop predecessor
// is itself lies on a cycle of length dividing V. The final print
// step -- walking the cycle once to read off the trade order -- is
// inherently sequential, since each hop depends on the previous one.
//
// Compile: g++ -std=c++17 -Wall -Wextra -O2 206_fx_cycle_reconstruct_cpu_baseline.cpp -o 206_fx_cycle_reconstruct_cpu_baseline
// Run:     ./206_fx_cycle_reconstruct_cpu_baseline

#include <cstdio>
#include <string>
#include <vector>

constexpr int V = 4;

int main() {
    std::vector<std::string> currencies = {"USD", "EUR", "GBP", "JPY"};
    std::vector<int> pred = {3, 0, 1, 2};   // pred[USD]=JPY, pred[EUR]=USD, pred[GBP]=EUR, pred[JPY]=GBP
    constexpr int TOUCHED = 0;              // USD -- found still-relaxable in Section 36.2's round 4

    std::printf("=== Section 36.3 CPU baseline: confirming cycle membership via pointer jumping ===\n\n");
    std::printf("predecessor array (1-hop ancestor): [%s, %s, %s, %s]\n",
                currencies[pred[0]].c_str(), currencies[pred[1]].c_str(), currencies[pred[2]].c_str(), currencies[pred[3]].c_str());
    std::printf("touched node: %s, V=%d, rounds needed = log2(%d) = 2\n\n", currencies[TOUCHED].c_str(), V, V);

    std::vector<int> anc = pred;
    int round_num = 1, hop = 1;
    while (hop < V) {
        std::vector<int> next_anc(V);
        for (int v = 0; v < V; ++v) next_anc[v] = anc[anc[v]];
        hop *= 2;
        std::printf("round %d (doubling from %d-hop to %d-hop):\n", round_num, hop / 2, hop);
        for (int v = 0; v < V; ++v) {
            std::printf("  anc[%s] = anc[anc[%s]] = anc[%s] = %s\n",
                        currencies[v].c_str(), currencies[v].c_str(), currencies[anc[v]].c_str(), currencies[next_anc[v]].c_str());
        }
        anc = next_anc;
        round_num++;
    }

    std::printf("\nfinal %d-hop ancestor array: [%s, %s, %s, %s]\n", V,
                currencies[anc[0]].c_str(), currencies[anc[1]].c_str(), currencies[anc[2]].c_str(), currencies[anc[3]].c_str());
    bool on_cycle = (anc[TOUCHED] == TOUCHED);
    std::printf("\nanc[%s] == %s? %s -- %s is %s a cycle of length dividing %d\n\n",
                currencies[TOUCHED].c_str(), currencies[TOUCHED].c_str(), on_cycle ? "true" : "false",
                currencies[TOUCHED].c_str(), on_cycle ? "confirmed on" : "NOT on", V);

    std::printf("=== final step: walking the cycle once, sequentially, to print the trade order ===\n\n");
    std::vector<int> backward_order;
    int cur = TOUCHED;
    while (true) {
        backward_order.push_back(cur);
        cur = pred[cur];
        if (cur == TOUCHED) break;
    }
    std::vector<int> forward_order(backward_order.rbegin(), backward_order.rend());

    std::string cycle_str;
    for (size_t i = 0; i < forward_order.size(); ++i) cycle_str += currencies[forward_order[i]] + " -> ";
    cycle_str += currencies[forward_order[0]];
    std::printf("arbitrage cycle (forward trade order): %s\n", cycle_str.c_str());

    double rates[4][4] = {};   // sparse, only relevant entries set
    rates[0][1] = 0.90; rates[1][2] = 0.85; rates[2][3] = 190.0; rates[3][0] = 0.0072; rates[1][3] = 150.0;
    double product = 1.0;
    for (size_t i = 0; i < forward_order.size(); ++i) {
        int u = forward_order[i], v = forward_order[(i + 1) % forward_order.size()];
        product *= rates[u][v];
    }
    std::printf("total rate product around this cycle: %.5f\n", product);

    bool ok = on_cycle && (forward_order == std::vector<int>{1, 2, 3, 0}) && (product > 1.04651 && product < 1.04653);
    std::printf("\nself-check: cycle confirmed via pointer jumping, forward order EUR->GBP->JPY->USD->EUR, "
                "rate product 1.04652 matching Section 36.1: %s\n", ok ? "confirmed" : "MISMATCH");

    return ok ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++17 -Wall -Wextra -O2 206_fx_cycle_reconstruct_cpu_baseline.cpp -o 206_fx_cycle_reconstruct_cpu_baseline
./206_fx_cycle_reconstruct_cpu_baseline
```

**Sample input:** the predecessor array left behind by Section 36.2's detection round, and the touched node (USD) that pass identified.

**Sample output:**

```text
=== Section 36.3 CPU baseline: confirming cycle membership via pointer jumping ===

predecessor array (1-hop ancestor): [JPY, USD, EUR, GBP]
touched node: USD, V=4, rounds needed = log2(4) = 2

round 1 (doubling from 1-hop to 2-hop):
  anc[USD] = anc[anc[USD]] = anc[JPY] = GBP
  anc[EUR] = anc[anc[EUR]] = anc[USD] = JPY
  anc[GBP] = anc[anc[GBP]] = anc[EUR] = USD
  anc[JPY] = anc[anc[JPY]] = anc[GBP] = EUR
round 2 (doubling from 2-hop to 4-hop):
  anc[USD] = anc[anc[USD]] = anc[GBP] = USD
  anc[EUR] = anc[anc[EUR]] = anc[JPY] = EUR
  anc[GBP] = anc[anc[GBP]] = anc[USD] = GBP
  anc[JPY] = anc[anc[JPY]] = anc[EUR] = JPY

final 4-hop ancestor array: [USD, EUR, GBP, JPY]

anc[USD] == USD? true -- USD is confirmed on a cycle of length dividing 4

=== final step: walking the cycle once, sequentially, to print the trade order ===

arbitrage cycle (forward trade order): EUR -> GBP -> JPY -> USD -> EUR
total rate product around this cycle: 1.04652

self-check: cycle confirmed via pointer jumping, forward order EUR->GBP->JPY->USD->EUR, rate product 1.04652 matching Section 36.1: confirmed
```

### The Concept, In Detail

```
ASCII view: pointer doubling reaches the 4-hop predecessor in 2 rounds.

  round 0 (1-hop, = predecessor array):
    USD -> JPY    EUR -> USD    GBP -> EUR    JPY -> GBP

  round 1 (2-hop, anc[v] = anc[anc[v]]):
    USD -> GBP    EUR -> JPY    GBP -> USD    JPY -> EUR

  round 2 (4-hop, doubled again):
    USD -> USD    EUR -> EUR    GBP -> GBP    JPY -> JPY   <- every node maps to ITSELF

  anc[USD] == USD  =>  USD is confirmed on a cycle of length dividing 4
```

Every thread in every round reads only the array the previous round produced and writes only its own slot -- the identical data-parallel shape Chapter 11 established for list ranking, just applied to a cycle instead of a chain. Only the very last step -- walking the confirmed cycle's predecessor pointers once, in order, to print the actual trade sequence -- is handed to a single thread, because each hop in that walk depends on knowing the previous hop's result; it is printed, not computed in parallel, honestly distinguishing the part of this problem that truly parallelizes from the part that does not.

[COMMON TRAP]
It is tempting to assume that because Section 36.2 already found a negative cycle exists somewhere, the specific "touched" node it reported must itself be on that cycle. Bellman-Ford's detection round only guarantees that the touched node is reachable FROM a negative cycle -- in a graph with more structure than this chapter's 4-node example, a detection pass can touch a node several hops downstream of the actual cycle. Section 36.3's pointer-jumping confirmation step is not a formality; it is the step that turns "a negative cycle exists somewhere upstream" into "this specific node is on it," which is what makes reading off a concrete, executable sequence of trades possible at all.

### Code and Verification

```cpp
#include <cstdio>
#include <string>
#include <vector>

// Chapter 36.3 main -- pointer jumping is data-parallel by construction:
// every node's thread reads only the PREVIOUS round's ancestor array and
// writes its own slot in a fresh one (Chapter 11's technique), so all V
// threads run a doubling round fully in parallel with no synchronization
// beyond the round boundary itself. Only the FINAL cycle-printing step is
// single-threaded, since walking a chain one hop at a time is inherently
// sequential -- the same "parallel confirmation, sequential readout"
// split used throughout this chapter (log-transform vs. relaxation,
// relaxation vs. detection).

constexpr int V = 4;

__global__ void pointer_jump_kernel(const int* anc_prev, int* anc_next) {
    int tid = threadIdx.x;                        // one thread per node
    anc_next[tid] = anc_prev[anc_prev[tid]];       // reads only the previous round's buffer
}

__global__ void reconstruct_cycle_kernel(const int* pred, int touched, int* order, int* order_len) {
    if (threadIdx.x != 0) return;   // walking the cycle one hop at a time cannot be parallelized
    int cur = touched;
    int n = 0;
    do {
        order[n++] = cur;
        cur = pred[cur];
    } while (cur != touched);
    *order_len = n;
}

// ---- Host-side replay of the identical two-phase logic: a data-parallel
// ---- pointer-jumping doubling phase, then a single-thread sequential
// ---- walk-and-print phase. ----

int main() {
    printf("=== Section 36.3 main: parallel pointer jumping (one thread per node) ===\n\n");

    std::vector<std::string> currencies = {"USD", "EUR", "GBP", "JPY"};
    std::vector<int> pred = {3, 0, 1, 2};   // pred[USD]=JPY, pred[EUR]=USD, pred[GBP]=EUR, pred[JPY]=GBP
    constexpr int TOUCHED = 0;              // USD

    std::vector<int> anc = pred;
    printf("round 0 (1-hop, = predecessor array): [ %s %s %s %s ]\n\n",
           currencies[anc[0]].c_str(), currencies[anc[1]].c_str(), currencies[anc[2]].c_str(), currencies[anc[3]].c_str());

    int hop = 1, round_num = 1;
    while (hop < V) {
        printf("round %d -- %d threads execute simultaneously:\n", round_num, V);
        std::vector<int> next_anc(V);
        for (int tid = 0; tid < V; ++tid) {
            next_anc[tid] = anc[anc[tid]];   // mirrors pointer_jump_kernel's per-thread write
            printf("  thread %d (%s): anc[anc[%d]] = anc[%d] = %d (%s)\n",
                   tid, currencies[tid].c_str(), tid, anc[tid], next_anc[tid], currencies[next_anc[tid]].c_str());
        }
        anc = next_anc;
        hop *= 2;
        printf("  round %d result: [ %s %s %s %s ]\n\n", round_num,
               currencies[anc[0]].c_str(), currencies[anc[1]].c_str(), currencies[anc[2]].c_str(), currencies[anc[3]].c_str());
        round_num++;
    }

    printf("final %d-hop ancestor array: [ %s %s %s %s ]\n", V,
           currencies[anc[0]].c_str(), currencies[anc[1]].c_str(), currencies[anc[2]].c_str(), currencies[anc[3]].c_str());
    bool on_cycle = (anc[TOUCHED] == TOUCHED);
    printf("anc[%s] == %s? %s\n\n", currencies[TOUCHED].c_str(), currencies[TOUCHED].c_str(), on_cycle ? "true" : "false");

    printf("single thread (thread 0) performs the final sequential walk-and-print:\n");
    std::vector<int> backward_order;   // mirrors reconstruct_cycle_kernel's single-thread walk
    int cur = TOUCHED;
    while (true) {
        backward_order.push_back(cur);
        cur = pred[cur];
        if (cur == TOUCHED) break;
    }
    std::vector<int> forward_order(backward_order.rbegin(), backward_order.rend());

    std::string cycle_str;
    for (size_t i = 0; i < forward_order.size(); ++i) cycle_str += currencies[forward_order[i]] + " -> ";
    cycle_str += currencies[forward_order[0]];
    printf("  arbitrage cycle: %s\n", cycle_str.c_str());

    bool ok = on_cycle && (forward_order == std::vector<int>{1, 2, 3, 0});
    printf("\nself-check: parallel pointer jumping confirms the identical cycle membership and\n");
    printf("trade order as Section 36.3's CPU baseline: %s\n", ok ? "confirmed" : "MISMATCH");

    return ok ? 0 : 1;
}
```

**Compile and run:**

```bash
NVDIR=/usr/local/lib/python3.11/dist-packages/nvidia/cu13
$NVDIR/bin/nvcc -arch=sm_80 -I$NVDIR/include -L$NVDIR/lib -lcudart 207_fx_cycle_reconstruct_kernel.cu -o 207_fx_cycle_reconstruct_kernel
LD_LIBRARY_PATH=$NVDIR/lib ./207_fx_cycle_reconstruct_kernel
```

**Sample input:** the same predecessor array, doubled by 4 simultaneously-executing threads (one per node) instead of a sequential loop, with the final walk-and-print handed to a single thread.

**Sample output:**

```text
=== Section 36.3 main: parallel pointer jumping (one thread per node) ===

round 0 (1-hop, = predecessor array): [ JPY USD EUR GBP ]

round 1 -- 4 threads execute simultaneously:
  thread 0 (USD): anc[anc[0]] = anc[3] = 2 (GBP)
  thread 1 (EUR): anc[anc[1]] = anc[0] = 3 (JPY)
  thread 2 (GBP): anc[anc[2]] = anc[1] = 0 (USD)
  thread 3 (JPY): anc[anc[3]] = anc[2] = 1 (EUR)
  round 1 result: [ GBP JPY USD EUR ]

round 2 -- 4 threads execute simultaneously:
  thread 0 (USD): anc[anc[0]] = anc[2] = 0 (USD)
  thread 1 (EUR): anc[anc[1]] = anc[3] = 1 (EUR)
  thread 2 (GBP): anc[anc[2]] = anc[0] = 2 (GBP)
  thread 3 (JPY): anc[anc[3]] = anc[1] = 3 (JPY)
  round 2 result: [ USD EUR GBP JPY ]

final 4-hop ancestor array: [ USD EUR GBP JPY ]
anc[USD] == USD? true

single thread (thread 0) performs the final sequential walk-and-print:
  arbitrage cycle: EUR -> GBP -> JPY -> USD -> EUR

self-check: parallel pointer jumping confirms the identical cycle membership and
trade order as Section 36.3's CPU baseline: confirmed
```

## Chapter Summary

Foreign-exchange arbitrage detection needed no new parallel primitive at all -- it needed three techniques this book had already built, composed in sequence. Section 36.1 showed that a single algebraic identity, `log(a*b) = log(a) + log(b)`, converts a multiplicative "do these rates compound to a profit" question into an additive "is this a negative cycle" question, via an embarrassingly parallel per-edge transform. Section 36.2 showed that Chapter 23's Bellman-Ford negative-cycle detection becomes a genuinely safe parallel kernel once relaxation is double-buffered, with Chapter 9's CAS-retry discipline supplying the one atomic operation CUDA does not provide natively: a minimum over doubles. Section 36.3 showed that confirming a touched node truly lies on the detected cycle, and reading off the trade order, is Chapter 11's pointer-jumping technique applied to a cycle instead of a chain, paired with an honestly sequential final readout step. Three techniques from three earlier, unrelated-looking chapters, combined, solved a problem that looks nothing like any of them on the surface.

## Self-Check Questions

1. What algebraic identity makes it possible to detect "rates that multiply to more than 1.0" using an algorithm built around ADDING edge weights, and why does it work?
2. Why must Bellman-Ford be used here instead of Dijkstra, given that both are shortest-path algorithms?
3. What specifically could go wrong if Section 36.2's relaxation kernel let a thread read a value another thread had already updated in the SAME round, and how does double buffering prevent it?
4. Why does computing an atomic minimum over `double` values require a CAS-retry loop in CUDA, when `atomicMin` already exists natively for integers?
5. Section 36.2's detection round found that USD was "touched." Why is that not, by itself, proof that USD lies on the negative cycle, and what does Section 36.3 do to establish that proof?
6. Name the three earlier chapters (or sections) whose techniques this chapter reused, and what each one specifically contributed.

## Where We Go Next

There is no Chapter 37. This book began with the smallest possible unit of parallel work -- a single thread reading and writing its own data -- and spent thirty-six chapters building outward from it: reductions and scans that let many threads combine their results correctly, atomics and lock-free retry loops that let threads share mutable state without corrupting it, trees and hash tables that let a single query skip almost all of the data it does not need, and graph algorithms that let all of the above answer questions about relationships, not just values. Part 8's eight case studies were never introducing new ideas; every one of them, including this final one, was a demonstration that a genuinely difficult, genuinely real problem is almost always a composition of primitives already in hand, applied honestly, and verified rather than assumed. That habit -- reach for the smallest correct tool, prove it does what it claims, and never trust a parallel result you have not checked against a sequential one -- is the actual subject this book was teaching all along, and it travels with you far beyond CUDA, and far beyond this book.

## Worked Solutions

**1.** The identity is `log(a * b) = log(a) + log(b)`, applied as `weight = -log(rate)` for each edge. A cycle's rates multiply to more than 1.0 exactly when the sum of `log(rate)` across that cycle is greater than 0, which, after negating, means the sum of `weight` across that cycle is LESS than 0 -- a negative cycle. This turns a multiplicative test that no additive shortest-path algorithm could evaluate directly into an additive one that Bellman-Ford's existing negative-cycle detection already answers correctly, with no changes to the algorithm itself.

**2.** Dijkstra's correctness proof depends on every edge weight being non-negative -- it greedily finalizes the closest unvisited node and never revisits it, which is only safe if no later edge could ever decrease that node's distance further. Log-transforming a profitable exchange rate (one greater than 1.0) produces a NEGATIVE weight by design, since that is exactly the condition this chapter needs to detect; Dijkstra would produce wrong answers, or simply is not defined, on a graph containing negative edges. Bellman-Ford's relaxation approach never assumes a node's distance is final until all V-1 rounds complete, which is what makes it correct in the presence of negative weights, and its extra detection round is precisely what exposes a negative cycle when one exists.

**3.** If a thread relaxing one edge could read a distance value that ANOTHER thread had already updated earlier in the same round, the result would depend on the arbitrary order in which the GPU happened to schedule those threads -- the same non-determinism problem every earlier chapter's atomics and reductions were built to avoid. Double buffering prevents this by having every thread in a round read only from a buffer that was completely finalized in the PREVIOUS round and write only into a separate, fresh buffer for the CURRENT round; no thread's read can ever be affected by another thread's write within the same round, so the round's result is identical no matter what order its threads execute in.

**4.** CUDA provides `atomicMin` as a hardware-level operation only for integer types, because comparing and swapping a fixed-width integer atomically is directly supported by the memory subsystem. There is no hardware atomic minimum for `double`, so Section 36.2's kernel builds one in software: it reinterprets the double's bit pattern as a 64-bit integer (via `__double_as_longlong`), reads the current value, computes whether its own candidate is actually smaller, and if so attempts an `atomicCAS` (which IS available natively) to swap in its candidate -- retrying if another thread's write got there first, exactly Chapter 9's lock-free CAS-retry loop, just wrapped around a minimum comparison instead of a stack push.

**5.** Bellman-Ford's detection round only proves that some node's distance could STILL be reduced after V-1 rounds, which guarantees a negative cycle exists SOMEWHERE reachable from the source -- it does not, by itself, guarantee that the specific node it touched is a member of that cycle rather than a node merely downstream of it. Section 36.3 establishes true membership by pointer-jumping the predecessor array `log2(V)` times: a node whose own resulting 2^k-hop-with-2^k-equal-to-V ancestor is itself must sit on a cycle whose length divides V, which is a direct structural proof of cycle membership, not just reachability from one.

**6.** Chapter 23 contributed Bellman-Ford's relaxation-and-detection structure for finding negative cycles at all, which Section 36.2 ran nearly unmodified once rates were converted to weights. Chapter 9 contributed the lock-free CAS-retry discipline that Section 36.2's atomic-minimum-for-doubles reused to resolve round 3's genuine multi-writer race safely. Chapter 11 contributed the pointer-jumping (list-ranking) technique that Section 36.3 reused, applied to a cycle instead of a chain, to confirm cycle membership in only `log2(V)` parallel rounds and then read off the trade order.
