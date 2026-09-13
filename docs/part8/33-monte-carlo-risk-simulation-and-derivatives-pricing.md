# Chapter 33: Monte Carlo Risk Simulation and Derivatives Pricing

Monte Carlo methods answer a question that has no clean closed-form formula: simulate a huge number of independent, plausible futures, then let their average settle on the answer. A derivative's fair price under uncertainty, and a portfolio's exposure to a bad outcome, are both exactly this kind of question -- and a GPU's thousands of independent lanes are close to an ideal machine for it, because Monte Carlo's defining property is that every simulated path is completely independent of every other one. This chapter needs no new synchronization idea at all: Section 33.1 builds each price path with Chapter 5's scan, Section 33.2 collapses simulated payoffs into a single price estimate with Chapter 4's reduction, and Section 33.3 finds the worst-case scenarios with Chapter 6's stream compaction -- three of Part 1's four primitives, composed rather than reinvented.

## 33.1 Simulating Price Paths via Prefix Sum

### Intuition

A simulated price path is a sequence of price levels over time, each one built by adding that step's price movement to the running total so far -- which is exactly an INCLUSIVE PREFIX SUM, Chapter 5's scan, applied one path at a time. What makes Monte Carlo simulation different from every previous use of scan in this book is the SHAPE of the parallelism: instead of one long array scanned collaboratively by many threads, there are many short, independent arrays (one per path), each one small enough for a single thread to scan entirely on its own, with zero communication between threads.

### The Sequential (CPU) Baseline

```cpp
// 184_mc_path_scan_cpu_baseline.cpp
//
// Chapter 33.1 -- sequential baseline for Monte Carlo price-path
// simulation. Each simulated path is built from a sequence of
// per-step price increments via an INCLUSIVE PREFIX SUM (scan) --
// Chapter 5's scan, applied here to produce the running price level
// at every time step along one path. Paths are completely independent
// of each other (Monte Carlo's defining property), so this baseline
// simply runs the scan once per path, in sequence.
//
// Compile: g++ -std=c++17 -Wall -Wextra -O2 184_mc_path_scan_cpu_baseline.cpp -o 184_mc_path_scan_cpu_baseline
// Run:     ./184_mc_path_scan_cpu_baseline

#include <array>
#include <cstdio>

constexpr int NUM_PATHS = 8;
constexpr int NUM_STEPS = 4;
constexpr double S0 = 100.0;

int main() {
    // Per-step price increments for each of 8 simulated paths -- a
    // stand-in for random draws, fixed here so the whole computation
    // is exactly reproducible without needing real hardware RNG.
    double increments[NUM_PATHS][NUM_STEPS] = {
        {3, 2, -1, 4},
        {-2, -3, 1, -1},
        {5, -2, 3, 2},
        {-5, -4, -3, -2},
        {1, 1, 1, 1},
        {-6, 2, -1, -8},
        {4, 4, -2, -1},
        {-3, -2, -4, -5},
    };

    std::printf("=== Section 33.1 CPU baseline: path simulation via inclusive prefix sum (scan) ===\n\n");
    std::printf("S0 = %.1f, %d paths, %d steps each\n\n", S0, NUM_PATHS, NUM_STEPS);

    std::array<double, NUM_PATHS> terminal_prices{};

    for (int p = 0; p < NUM_PATHS; ++p) {
        double running = 0.0;
        double path_prices[NUM_STEPS];
        double scanned[NUM_STEPS];
        for (int s = 0; s < NUM_STEPS; ++s) {
            running += increments[p][s];
            scanned[s] = running;
            path_prices[s] = S0 + running;
        }
        terminal_prices[p] = path_prices[NUM_STEPS - 1];

        std::printf("path %d: increments=[%.0f, %.0f, %.0f, %.0f] -> cumulative sum (scan) = "
                    "[%.1f, %.1f, %.1f, %.1f] -> prices = [%.1f, %.1f, %.1f, %.1f]\n",
                    p, increments[p][0], increments[p][1], increments[p][2], increments[p][3],
                    scanned[0], scanned[1], scanned[2], scanned[3],
                    path_prices[0], path_prices[1], path_prices[2], path_prices[3]);
    }

    std::printf("\nterminal prices S_T: [");
    for (int p = 0; p < NUM_PATHS; ++p) {
        std::printf("%.1f%s", terminal_prices[p], (p + 1 < NUM_PATHS) ? ", " : "");
    }
    std::printf("]\n");

    double expected[NUM_PATHS] = {108.0, 95.0, 108.0, 86.0, 104.0, 87.0, 105.0, 86.0};
    bool ok = true;
    for (int p = 0; p < NUM_PATHS; ++p) {
        if (terminal_prices[p] != expected[p]) ok = false;
    }
    std::printf("\nself-check: terminal prices match hand-computed expectation: %s\n",
                ok ? "confirmed" : "MISMATCH");

    return ok ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++17 -Wall -Wextra -O2 184_mc_path_scan_cpu_baseline.cpp -o 184_mc_path_scan_cpu_baseline
./184_mc_path_scan_cpu_baseline
```

**Sample input:** 8 simulated price paths, starting at S0=100, each with 4 per-step price increments (a stand-in for random draws), each path's running price computed via inclusive prefix sum.

**Sample output:**

```text
=== Section 33.1 CPU baseline: path simulation via inclusive prefix sum (scan) ===

S0 = 100.0, 8 paths, 4 steps each

path 0: increments=[3, 2, -1, 4] -> cumulative sum (scan) = [3.0, 5.0, 4.0, 8.0] -> prices = [103.0, 105.0, 104.0, 108.0]
path 1: increments=[-2, -3, 1, -1] -> cumulative sum (scan) = [-2.0, -5.0, -4.0, -5.0] -> prices = [98.0, 95.0, 96.0, 95.0]
path 2: increments=[5, -2, 3, 2] -> cumulative sum (scan) = [5.0, 3.0, 6.0, 8.0] -> prices = [105.0, 103.0, 106.0, 108.0]
path 3: increments=[-5, -4, -3, -2] -> cumulative sum (scan) = [-5.0, -9.0, -12.0, -14.0] -> prices = [95.0, 91.0, 88.0, 86.0]
path 4: increments=[1, 1, 1, 1] -> cumulative sum (scan) = [1.0, 2.0, 3.0, 4.0] -> prices = [101.0, 102.0, 103.0, 104.0]
path 5: increments=[-6, 2, -1, -8] -> cumulative sum (scan) = [-6.0, -4.0, -5.0, -13.0] -> prices = [94.0, 96.0, 95.0, 87.0]
path 6: increments=[4, 4, -2, -1] -> cumulative sum (scan) = [4.0, 8.0, 6.0, 5.0] -> prices = [104.0, 108.0, 106.0, 105.0]
path 7: increments=[-3, -2, -4, -5] -> cumulative sum (scan) = [-3.0, -5.0, -9.0, -14.0] -> prices = [97.0, 95.0, 91.0, 86.0]

terminal prices S_T: [108.0, 95.0, 108.0, 86.0, 104.0, 87.0, 105.0, 86.0]

self-check: terminal prices match hand-computed expectation: confirmed
```

### The Concept, In Detail

```
ASCII view: many independent short scans, not one long shared scan.

  Chapter 5's original scan:      ONE array, MANY threads collaborating
    [a0 a1 a2 a3 a4 a5 a6 a7] -> one inclusive prefix sum, computed
                                   cooperatively across the whole block

  Section 33.1's per-path scan:   MANY arrays, ONE thread each
    path 0: [i0 i1 i2 i3] -> thread 0's own private scan
    path 1: [i0 i1 i2 i3] -> thread 1's own private scan
    path 2: [i0 i1 i2 i3] -> thread 2's own private scan
    ...                        (zero communication between threads)
```

Chapter 5's scan was built to extract parallelism WITHIN one long sequential dependency chain, using techniques like the work-efficient up-sweep/down-sweep pattern precisely because a single thread computing that whole chain alone would be too slow. Here, the opposite problem holds: there are plenty of independent chains (8 of them), each too short to be worth parallelizing internally, so the parallelism this section extracts is ACROSS paths rather than within one.

[COMMON TRAP]
It is tempting to think that because Chapter 5 built an elaborate parallel scan algorithm, every scan in this book must use that same collaborative, cross-thread machinery. The actual requirement Chapter 5 established was narrower: a scan is a specific computation (a running combine of a sequence), not a specific parallelization strategy for computing it. When the sequence is short and there are many independent copies of the problem, as here, the right parallelization is simply to give each copy its own thread and let it compute the tiny scan sequentially -- Chapter 5's parallel algorithm exists for the opposite case, one long sequence with no other parallelism available.

### Code and Verification

```cpp
#include <cstdio>
#include <array>

// Chapter 33.1 main -- each thread simulates exactly one price path,
// independently of every other thread: Monte Carlo's defining
// property is that paths share no state, so there is nothing to
// synchronize and no atomics are needed at all, exactly like Section
// 30.1's per-particle hashing. Within its own path, a thread computes
// the SAME inclusive prefix sum (scan) Chapter 5 already built --
// just run once, sequentially, by a single thread, because four tiny
// steps offer no further parallelism worth extracting. This is the
// same "the algorithm doesn't change, only the scale does" point
// Section 31.1 made about reduction, now made about scan.

#define NUM_PATHS 8
#define NUM_STEPS 4
#define S0 100.0

__global__ void simulate_path_kernel(const double* increments, double* terminal_prices) {
    int tid = threadIdx.x;
    double running = 0.0;
    for (int s = 0; s < NUM_STEPS; s++) {
        running += increments[tid * NUM_STEPS + s];
    }
    terminal_prices[tid] = S0 + running;
}

// ---- Host-side replay of the identical per-thread scan logic. Every
// ---- path is independent, so there is no landing order to force --
// ---- the replay simply runs each "thread" in turn. ----

int main() {
    printf("=== Section 33.1 main: per-thread path simulation via inclusive prefix sum (scan) ===\n\n");

    double increments[NUM_PATHS][NUM_STEPS] = {
        {3, 2, -1, 4},
        {-2, -3, 1, -1},
        {5, -2, 3, 2},
        {-5, -4, -3, -2},
        {1, 1, 1, 1},
        {-6, 2, -1, -8},
        {4, 4, -2, -1},
        {-3, -2, -4, -5},
    };

    printf("S0 = %.1f, %d independent threads (one per path), %d steps each -- no atomics needed\n\n",
           S0, NUM_PATHS, NUM_STEPS);

    std::array<double, NUM_PATHS> terminal_prices{};

    for (int tid = 0; tid < NUM_PATHS; tid++) {
        double running = 0.0;
        double scanned[NUM_STEPS];
        for (int s = 0; s < NUM_STEPS; s++) {
            running += increments[tid][s];
            scanned[s] = running;
        }
        terminal_prices[tid] = S0 + running;
        printf("thread %d: increments=[%.0f, %.0f, %.0f, %.0f] -> own scan = [%.1f, %.1f, %.1f, %.1f] "
               "-> writes terminal_prices[%d] = %.1f\n",
               tid, increments[tid][0], increments[tid][1], increments[tid][2], increments[tid][3],
               scanned[0], scanned[1], scanned[2], scanned[3], tid, terminal_prices[tid]);
    }

    printf("\nfinal terminal_prices: [");
    for (int p = 0; p < NUM_PATHS; p++) {
        printf("%.1f%s", terminal_prices[p], (p + 1 < NUM_PATHS) ? ", " : "");
    }
    printf("]\n");

    double expected[NUM_PATHS] = {108.0, 95.0, 108.0, 86.0, 104.0, 87.0, 105.0, 86.0};
    bool ok = true;
    for (int p = 0; p < NUM_PATHS; p++) {
        if (terminal_prices[p] != expected[p]) ok = false;
    }
    printf("\nself-check: every thread's independently-scanned terminal price matches Section\n");
    printf("33.1's CPU baseline exactly, with no thread ever reading another thread's data: %s\n",
           ok ? "confirmed" : "MISMATCH");

    return ok ? 0 : 1;
}
```

**Compile and run:**

```bash
NVDIR=/usr/local/lib/python3.11/dist-packages/nvidia/cu13
$NVDIR/bin/nvcc -arch=sm_80 -I$NVDIR/include -L$NVDIR/lib -lcudart 185_mc_path_scan_kernel.cu -o 185_mc_path_scan_kernel
LD_LIBRARY_PATH=$NVDIR/lib ./185_mc_path_scan_kernel
```

**Sample input:** the same 8 paths, each simulated by its own independent thread, with no atomics or synchronization of any kind.

**Sample output:**

```text
=== Section 33.1 main: per-thread path simulation via inclusive prefix sum (scan) ===

S0 = 100.0, 8 independent threads (one per path), 4 steps each -- no atomics needed

thread 0: increments=[3, 2, -1, 4] -> own scan = [3.0, 5.0, 4.0, 8.0] -> writes terminal_prices[0] = 108.0
thread 1: increments=[-2, -3, 1, -1] -> own scan = [-2.0, -5.0, -4.0, -5.0] -> writes terminal_prices[1] = 95.0
thread 2: increments=[5, -2, 3, 2] -> own scan = [5.0, 3.0, 6.0, 8.0] -> writes terminal_prices[2] = 108.0
thread 3: increments=[-5, -4, -3, -2] -> own scan = [-5.0, -9.0, -12.0, -14.0] -> writes terminal_prices[3] = 86.0
thread 4: increments=[1, 1, 1, 1] -> own scan = [1.0, 2.0, 3.0, 4.0] -> writes terminal_prices[4] = 104.0
thread 5: increments=[-6, 2, -1, -8] -> own scan = [-6.0, -4.0, -5.0, -13.0] -> writes terminal_prices[5] = 87.0
thread 6: increments=[4, 4, -2, -1] -> own scan = [4.0, 8.0, 6.0, 5.0] -> writes terminal_prices[6] = 105.0
thread 7: increments=[-3, -2, -4, -5] -> own scan = [-3.0, -5.0, -9.0, -14.0] -> writes terminal_prices[7] = 86.0

final terminal_prices: [108.0, 95.0, 108.0, 86.0, 104.0, 87.0, 105.0, 86.0]

self-check: every thread's independently-scanned terminal price matches Section
33.1's CPU baseline exactly, with no thread ever reading another thread's data: confirmed
```

## 33.2 Reducing Simulated Payoffs into a Single Price Estimate

### Intuition

Once every path's terminal price is known, a European call option's simulated payoff on that path is `max(S_T - K, 0)` -- zero if the path finished below the strike price K, the gain above K otherwise. The Monte Carlo price ESTIMATE is the average of these payoffs across every simulated path, discounted back to today -- and averaging across independent values is exactly Chapter 4's reduction, with the combine operator being plain addition, needing no generalization at all.

### The Sequential (CPU) Baseline

```cpp
// 186_mc_reduce_payoff_cpu_baseline.cpp
//
// Chapter 33.2 -- sequential baseline for averaging simulated option
// payoffs into a single price estimate. Once every path's terminal
// price is known (Section 33.1), a European call option's simulated
// payoff at each path is max(S_T - K, 0). Averaging these payoffs
// across all paths, then discounting, is a SUM REDUCTION -- Chapter
// 4's reduction, unchanged, with the combine operator being plain
// addition, exactly like Chapter 4's own original numeric case.
//
// Compile: g++ -std=c++17 -Wall -Wextra -O2 186_mc_reduce_payoff_cpu_baseline.cpp -o 186_mc_reduce_payoff_cpu_baseline
// Run:     ./186_mc_reduce_payoff_cpu_baseline

#include <algorithm>
#include <array>
#include <cstdio>

constexpr int NUM_PATHS = 8;
constexpr double K = 100.0;
constexpr double DISCOUNT = 0.95;

int main() {
    std::array<double, NUM_PATHS> terminal_prices = {108.0, 95.0, 108.0, 86.0, 104.0, 87.0, 105.0, 86.0};

    std::array<double, NUM_PATHS> payoffs{};
    for (int i = 0; i < NUM_PATHS; ++i) {
        payoffs[i] = std::max(terminal_prices[i] - K, 0.0);
    }

    std::printf("=== Section 33.2 CPU baseline: sequential sum + average of simulated payoffs ===\n\n");
    std::printf("terminal prices: [");
    for (int i = 0; i < NUM_PATHS; ++i) std::printf("%.1f%s", terminal_prices[i], (i + 1 < NUM_PATHS) ? ", " : "");
    std::printf("]\n");
    std::printf("strike K = %.1f\n", K);
    std::printf("payoffs (max(S_T - K, 0)): [");
    for (int i = 0; i < NUM_PATHS; ++i) std::printf("%.1f%s", payoffs[i], (i + 1 < NUM_PATHS) ? ", " : "");
    std::printf("]\n\n");

    double total = 0.0;
    for (int i = 0; i < NUM_PATHS; ++i) {
        total += payoffs[i];
        std::printf("  running sum after path %d: %.1f\n", i, total);
    }

    double mean_payoff = total / NUM_PATHS;
    double price_estimate = mean_payoff * DISCOUNT;
    std::printf("\nsum of payoffs = %.1f\n", total);
    std::printf("mean payoff = %.1f / %d = %.3f\n", total, NUM_PATHS, mean_payoff);
    std::printf("discounted price estimate = %.3f * %.2f = %.5f\n", mean_payoff, DISCOUNT, price_estimate);

    bool ok = (total == 25.0) && (mean_payoff == 3.125) &&
              (price_estimate > 2.96874 && price_estimate < 2.96876);
    std::printf("\nself-check: sum=25, mean=3.125, discounted price=2.96875: %s\n",
                ok ? "confirmed" : "MISMATCH");

    return ok ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++17 -Wall -Wextra -O2 186_mc_reduce_payoff_cpu_baseline.cpp -o 186_mc_reduce_payoff_cpu_baseline
./186_mc_reduce_payoff_cpu_baseline
```

**Sample input:** the 8 terminal prices from Section 33.1, a strike price of 100, and a flat discount factor of 0.95.

**Sample output:**

```text
=== Section 33.2 CPU baseline: sequential sum + average of simulated payoffs ===

terminal prices: [108.0, 95.0, 108.0, 86.0, 104.0, 87.0, 105.0, 86.0]
strike K = 100.0
payoffs (max(S_T - K, 0)): [8.0, 0.0, 8.0, 0.0, 4.0, 0.0, 5.0, 0.0]

  running sum after path 0: 8.0
  running sum after path 1: 8.0
  running sum after path 2: 16.0
  running sum after path 3: 16.0
  running sum after path 4: 20.0
  running sum after path 5: 20.0
  running sum after path 6: 25.0
  running sum after path 7: 25.0

sum of payoffs = 25.0
mean payoff = 25.0 / 8 = 3.125
discounted price estimate = 3.125 * 0.95 = 2.96875

self-check: sum=25, mean=3.125, discounted price=2.96875: confirmed
```

### The Concept, In Detail

```
ASCII view: Monte Carlo averaging IS Chapter 4's reduction, unchanged.

  8 payoffs:        [8.0  0.0  8.0  0.0  4.0  0.0  5.0  0.0]
  round 1 (pairs):  [8.0       8.0       4.0       5.0     ]
  round 2 (pairs):  [       16.0                9.0        ]
  round 3 (pairs):  [                  25.0                 ]

  price estimate = (sum / num_paths) * discount_factor
                 = (25.0 / 8) * 0.95 = 2.96875
```

Section 31.1 needed a new combine operator (box union) to reuse Chapter 4's reduction for a different payload; this section needs nothing new whatsoever, because averaging numbers is precisely the case Chapter 4 was originally built for. The only genuinely new idea here is the FRAMING: a Monte Carlo price estimate is, mechanically, nothing more than a mean, and a mean is a sum (reduction) followed by one division.

[COMMON TRAP]
It is tempting to think a larger, more realistic Monte Carlo simulation (millions of paths instead of 8) would need a fundamentally different reduction strategy. The reduction tree's SHAPE scales the same way regardless of path count -- log2(N) rounds either way -- and the only real-world complication at scale is that N is rarely a power of two, which Chapter 31.3's `[COMMON TRAP]` already covered: pad to the next power of two with identity-value (zero-payoff) entries, exactly as Section 33.2's own 8-path example would if its path count were not already a power of two.

### Code and Verification

```cpp
#include <cstdio>
#include <algorithm>
#include <array>
#include <vector>

// Chapter 33.2 main -- Chapter 4's PAIRWISE TREE reduction, unchanged,
// combining 8 simulated payoffs with plain addition: log2(8)=3 rounds,
// no padding needed since the path count is already a power of two.
// Unlike Section 31.1's box union, this combine operator needs no
// generalization at all -- it is exactly Chapter 4's original numeric
// sum, which is the point: Monte Carlo averaging is not a new use of
// reduction, it is the textbook one.

#define NUM_PATHS 8

__device__ double add_device(double a, double b) { return a + b; }

__global__ void reduce_sum_round_kernel(double* payoffs, int n) {
    int tid = threadIdx.x;
    if (tid * 2 + 1 >= n) return;
    payoffs[tid] = add_device(payoffs[tid * 2], payoffs[tid * 2 + 1]);
}

// ---- Host-side replay of the identical pairwise tree-reduction
// ---- logic, run round by round exactly as log2(8)=3 kernel launches
// ---- would. ----

int main() {
    printf("=== Section 33.2 main: parallel tree reduction (sum), log2(8)=3 rounds ===\n\n");

    std::array<double, NUM_PATHS> terminal_prices = {108.0, 95.0, 108.0, 86.0, 104.0, 87.0, 105.0, 86.0};
    constexpr double K = 100.0;
    constexpr double DISCOUNT = 0.95;

    std::array<double, NUM_PATHS> level{};
    for (int i = 0; i < NUM_PATHS; i++) level[i] = std::max(terminal_prices[i] - K, 0.0);

    printf("round 0 (leaves, per-path payoffs): [ ");
    for (double v : level) printf("%.1f ", v);
    printf("]\n\n");

    std::vector<double> cur(level.begin(), level.end());
    int round_num = 1;
    while (cur.size() > 1) {
        std::vector<double> next_level;
        printf("round %d:\n", round_num);
        for (size_t i = 0; i < cur.size(); i += 2) {
            double merged = cur[i] + cur[i + 1];
            printf("  thread %zu: add(slot %zu=%.1f, slot %zu=%.1f) -> %.1f\n",
                   i / 2, i, cur[i], i + 1, cur[i + 1], merged);
            next_level.push_back(merged);
        }
        cur = next_level;
        printf("round %d result: [ ", round_num);
        for (double v : cur) printf("%.1f ", v);
        printf("]\n\n");
        round_num++;
    }

    double total = cur[0];
    double mean_payoff = total / NUM_PATHS;
    double price_estimate = mean_payoff * DISCOUNT;

    printf("final reduced sum: %.1f\n", total);
    printf("mean payoff = %.1f / %d = %.3f\n", total, NUM_PATHS, mean_payoff);
    printf("discounted price estimate = %.3f * %.2f = %.5f\n", mean_payoff, DISCOUNT, price_estimate);

    bool ok = (total == 25.0) && (mean_payoff == 3.125) &&
              (price_estimate > 2.96874 && price_estimate < 2.96876);
    printf("\nself-check: the parallel tree reduction's final sum matches Section 33.2's\n");
    printf("sequential CPU baseline exactly (25.0), confirming plain addition tolerates\n");
    printf("reassociation the same way Chapter 4's original reduction always did: %s\n",
           ok ? "confirmed" : "MISMATCH");
    return ok ? 0 : 1;
}
```

**Compile and run:**

```bash
NVDIR=/usr/local/lib/python3.11/dist-packages/nvidia/cu13
$NVDIR/bin/nvcc -arch=sm_80 -I$NVDIR/include -L$NVDIR/lib -lcudart 187_mc_reduce_payoff_kernel.cu -o 187_mc_reduce_payoff_kernel
LD_LIBRARY_PATH=$NVDIR/lib ./187_mc_reduce_payoff_kernel
```

**Sample input:** the same 8 payoffs, reduced via a pairwise tree (log2(8)=3 rounds) instead of a sequential running sum.

**Sample output:**

```text
=== Section 33.2 main: parallel tree reduction (sum), log2(8)=3 rounds ===

round 0 (leaves, per-path payoffs): [ 8.0 0.0 8.0 0.0 4.0 0.0 5.0 0.0 ]

round 1:
  thread 0: add(slot 0=8.0, slot 1=0.0) -> 8.0
  thread 1: add(slot 2=8.0, slot 3=0.0) -> 8.0
  thread 2: add(slot 4=4.0, slot 5=0.0) -> 4.0
  thread 3: add(slot 6=5.0, slot 7=0.0) -> 5.0
round 1 result: [ 8.0 8.0 4.0 5.0 ]

round 2:
  thread 0: add(slot 0=8.0, slot 1=8.0) -> 16.0
  thread 1: add(slot 2=4.0, slot 3=5.0) -> 9.0
round 2 result: [ 16.0 9.0 ]

round 3:
  thread 0: add(slot 0=16.0, slot 1=9.0) -> 25.0
round 3 result: [ 25.0 ]

final reduced sum: 25.0
mean payoff = 25.0 / 8 = 3.125
discounted price estimate = 3.125 * 0.95 = 2.96875

self-check: the parallel tree reduction's final sum matches Section 33.2's
sequential CPU baseline exactly (25.0), confirming plain addition tolerates
reassociation the same way Chapter 4's original reduction always did: confirmed
```

## 33.3 Identifying Tail-Risk Scenarios via Stream Compaction

### Intuition

A single average price estimate says nothing about RISK -- a risk manager also needs to know which specific simulated scenarios represent a dangerous loss, so they can be examined individually or fed into a Value-at-Risk calculation. Extracting only the paths whose terminal loss exceeds some threshold, packed densely into their own array, is exactly Chapter 6's stream compaction: a predicate per path, then only the paths where it is true survive into the output.

### The Sequential (CPU) Baseline

```cpp
// 188_mc_compaction_tail_risk_cpu_baseline.cpp
//
// Chapter 33.3 -- sequential baseline for identifying tail-risk
// scenarios. A path represents a tail-risk scenario when its terminal
// loss (S0 - S_T) exceeds a threshold. Extracting only the paths
// whose predicate is true, packed densely, is Chapter 6's stream
// compaction: predicate array, exclusive scan for output offsets,
// then scatter.
//
// Compile: g++ -std=c++17 -Wall -Wextra -O2 188_mc_compaction_tail_risk_cpu_baseline.cpp -o 188_mc_compaction_tail_risk_cpu_baseline
// Run:     ./188_mc_compaction_tail_risk_cpu_baseline

#include <array>
#include <cstdio>
#include <vector>

constexpr int NUM_PATHS = 8;
constexpr double S0 = 100.0;
constexpr double LOSS_THRESHOLD = 10.0;

int main() {
    std::array<double, NUM_PATHS> terminal_prices = {108.0, 95.0, 108.0, 86.0, 104.0, 87.0, 105.0, 86.0};
    std::array<double, NUM_PATHS> losses{};
    for (int i = 0; i < NUM_PATHS; ++i) losses[i] = S0 - terminal_prices[i];

    std::printf("=== Section 33.3 CPU baseline: sequential predicate + scan + scatter compaction ===\n\n");
    std::printf("terminal prices: [");
    for (int i = 0; i < NUM_PATHS; ++i) std::printf("%.1f%s", terminal_prices[i], (i + 1 < NUM_PATHS) ? ", " : "");
    std::printf("]\n");
    std::printf("losses (S0 - S_T): [");
    for (int i = 0; i < NUM_PATHS; ++i) std::printf("%.1f%s", losses[i], (i + 1 < NUM_PATHS) ? ", " : "");
    std::printf("]\n");
    std::printf("loss threshold: %.1f\n\n", LOSS_THRESHOLD);

    std::array<int, NUM_PATHS> predicate{};
    for (int i = 0; i < NUM_PATHS; ++i) predicate[i] = (losses[i] > LOSS_THRESHOLD) ? 1 : 0;

    std::printf("predicate (loss > %.1f): [", LOSS_THRESHOLD);
    for (int i = 0; i < NUM_PATHS; ++i) std::printf("%d%s", predicate[i], (i + 1 < NUM_PATHS) ? ", " : "");
    std::printf("]\n");

    std::array<int, NUM_PATHS> offsets{};
    int running = 0;
    for (int i = 0; i < NUM_PATHS; ++i) {
        offsets[i] = running;
        running += predicate[i];
    }
    int total_matches = running;

    std::printf("exclusive scan of predicate (output offsets): [");
    for (int i = 0; i < NUM_PATHS; ++i) std::printf("%d%s", offsets[i], (i + 1 < NUM_PATHS) ? ", " : "");
    std::printf("]\n");
    std::printf("total matching paths: %d\n\n", total_matches);

    std::vector<int> compacted(total_matches, -1);
    for (int i = 0; i < NUM_PATHS; ++i) {
        if (predicate[i]) {
            compacted[offsets[i]] = i;
            std::printf("  path %d (loss=%.1f): predicate true -> writes compacted[%d] = %d\n",
                        i, losses[i], offsets[i], i);
        }
    }

    std::printf("\ncompacted tail-risk path indices (original order preserved): [");
    for (size_t i = 0; i < compacted.size(); ++i) std::printf("%d%s", compacted[i], (i + 1 < compacted.size()) ? ", " : "");
    std::printf("]\n");

    std::vector<int> expected = {3, 5, 7};
    bool ok = (compacted == expected);
    std::printf("self-check: compacted set matches expected [3, 5, 7] in original order: %s\n",
                ok ? "confirmed" : "MISMATCH");

    return ok ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++17 -Wall -Wextra -O2 188_mc_compaction_tail_risk_cpu_baseline.cpp -o 188_mc_compaction_tail_risk_cpu_baseline
./188_mc_compaction_tail_risk_cpu_baseline
```

**Sample input:** the same 8 terminal prices, a loss threshold of 10, and Chapter 6's predicate + exclusive scan + scatter compaction pipeline.

**Sample output:**

```text
=== Section 33.3 CPU baseline: sequential predicate + scan + scatter compaction ===

terminal prices: [108.0, 95.0, 108.0, 86.0, 104.0, 87.0, 105.0, 86.0]
losses (S0 - S_T): [-8.0, 5.0, -8.0, 14.0, -4.0, 13.0, -5.0, 14.0]
loss threshold: 10.0

predicate (loss > 10.0): [0, 0, 0, 1, 0, 1, 0, 1]
exclusive scan of predicate (output offsets): [0, 0, 0, 0, 1, 1, 2, 2]
total matching paths: 3

  path 3 (loss=14.0): predicate true -> writes compacted[0] = 3
  path 5 (loss=13.0): predicate true -> writes compacted[1] = 5
  path 7 (loss=14.0): predicate true -> writes compacted[2] = 7

compacted tail-risk path indices (original order preserved): [3, 5, 7]
self-check: compacted set matches expected [3, 5, 7] in original order: confirmed
```

### The Concept, In Detail

```
ASCII view: compaction keeps only the paths that cross the loss line.

  path:       0     1     2     3     4     5     6     7
  loss:      -8.0   5.0  -8.0  14.0  -4.0  13.0  -5.0  14.0
  predicate:  0     0     0     1     0     1     0     1
                                ^           ^           ^
  compacted (dense, tail-risk only):  [ 3 ,  5 ,  7 ]
```

Section 33.3's compaction differs from Chapter 6's original in one respect worth naming: Chapter 6 built the full predicate+scan+scatter pipeline to guarantee a STABLE output order (matching input order); Section 33.3's kernel instead uses a single shared `atomicAdd` cursor -- Section 28.2/30.2's bump-allocator-style append -- which claims output slots correctly but in whatever order threads happen to land, not necessarily input order. This is a legitimate, simpler alternative whenever only the compacted SET matters, not its exact ordering.

[COMMON TRAP]
It is tempting to assume the concurrent, atomicAdd-based compaction kernel is simply "wrong" when its output order does not match the sequential CPU baseline's output order. The self-check that actually matters compares the compacted SET (which paths ended up in the output at all), not their positions -- exactly as Section 30.2's particle-scatter self-check did -- because a single shared cursor guarantees every matching thread gets a distinct, correct slot, but says nothing about which matching thread's `atomicAdd` happens to execute first.

### Code and Verification

```cpp
#include <cstdio>
#include <array>
#include <algorithm>
#include <vector>

// Chapter 33.3 main -- concurrent compaction via a single shared
// atomicAdd cursor: Chapter 28.2/30.2's bump-allocator-style append,
// reused here as a simpler, well-known alternative to the full
// predicate+scan+scatter pipeline when only the compacted SET, not a
// specific stable order, is required. Each thread checks its own
// path's predicate; if true, it claims its output slot with a single
// atomicAdd on a shared cursor -- exactly like Section 30.2's scatter
// cursor, just without a per-bucket offset table, since every match
// goes into the same single output array.

#define NUM_PATHS 8

__global__ void compact_tail_risk_kernel(const int* predicate, int* cursor, int* compacted) {
    int tid = threadIdx.x;
    if (predicate[tid]) {
        int slot = atomicAdd(cursor, 1);
        compacted[slot] = tid;
    }
}

// ---- Host-side replay of the identical atomicAdd-based logic, driving
// ---- a forced landing order unrelated to path index, so the SET,
// ---- not the position, is what the self-check verifies. ----

struct SharedCounter {
    int value = 0;
    int atomic_add(int n = 1) {
        int old = value;
        value += n;
        return old;
    }
};

int main() {
    printf("=== Section 33.3 main: concurrent compaction via a single shared atomicAdd cursor ===\n\n");

    std::array<double, NUM_PATHS> terminal_prices = {108.0, 95.0, 108.0, 86.0, 104.0, 87.0, 105.0, 86.0};
    constexpr double S0 = 100.0;
    constexpr double LOSS_THRESHOLD = 10.0;

    std::array<double, NUM_PATHS> losses{};
    for (int i = 0; i < NUM_PATHS; i++) losses[i] = S0 - terminal_prices[i];

    std::array<int, NUM_PATHS> predicate{};
    for (int i = 0; i < NUM_PATHS; i++) predicate[i] = (losses[i] > LOSS_THRESHOLD) ? 1 : 0;

    int landing_order[NUM_PATHS] = {6, 3, 0, 5, 1, 7, 2, 4};

    printf("landing order: [ ");
    for (int p : landing_order) printf("%d ", p);
    printf("]\n\n");

    SharedCounter cursor;
    std::vector<int> compacted;

    for (int pid : landing_order) {
        if (predicate[pid]) {
            int slot = cursor.atomic_add(1);
            compacted.push_back(pid);
            printf("  path %d (loss=%.1f): predicate true -> atomicAdd(cursor,1) returned %d "
                   "-> compacted[%d] = %d\n", pid, losses[pid], slot, slot, pid);
        } else {
            printf("  path %d (loss=%.1f): predicate false -> no slot claimed\n", pid, losses[pid]);
        }
    }

    printf("\nfinal compacted set (concurrent, landing-order-dependent positions): [");
    for (size_t i = 0; i < compacted.size(); i++) printf("%d%s", compacted[i], (i + 1 < compacted.size()) ? ", " : "");
    printf("]\n");

    std::vector<int> expected = {3, 5, 7};
    std::vector<int> sorted_compacted = compacted;
    std::sort(sorted_compacted.begin(), sorted_compacted.end());
    bool ok = (sorted_compacted == expected);
    printf("self-check: compacted SET matches [3, 5, 7] regardless of landing order (positions\n");
    printf("may differ from the sequential run): %s\n", ok ? "confirmed" : "MISMATCH");

    return ok ? 0 : 1;
}
```

**Compile and run:**

```bash
NVDIR=/usr/local/lib/python3.11/dist-packages/nvidia/cu13
$NVDIR/bin/nvcc -arch=sm_80 -I$NVDIR/include -L$NVDIR/lib -lcudart 189_mc_compaction_tail_risk_kernel.cu -o 189_mc_compaction_tail_risk_kernel
LD_LIBRARY_PATH=$NVDIR/lib ./189_mc_compaction_tail_risk_kernel
```

**Sample input:** the same 8 paths and loss threshold, compacted concurrently via a single shared `atomicAdd` cursor, with a forced landing order unrelated to path index.

**Sample output:**

```text
=== Section 33.3 main: concurrent compaction via a single shared atomicAdd cursor ===

landing order: [ 6 3 0 5 1 7 2 4 ]

  path 6 (loss=-5.0): predicate false -> no slot claimed
  path 3 (loss=14.0): predicate true -> atomicAdd(cursor,1) returned 0 -> compacted[0] = 3
  path 0 (loss=-8.0): predicate false -> no slot claimed
  path 5 (loss=13.0): predicate true -> atomicAdd(cursor,1) returned 1 -> compacted[1] = 5
  path 1 (loss=5.0): predicate false -> no slot claimed
  path 7 (loss=14.0): predicate true -> atomicAdd(cursor,1) returned 2 -> compacted[2] = 7
  path 2 (loss=-8.0): predicate false -> no slot claimed
  path 4 (loss=-4.0): predicate false -> no slot claimed

final compacted set (concurrent, landing-order-dependent positions): [3, 5, 7]
self-check: compacted SET matches [3, 5, 7] regardless of landing order (positions
may differ from the sequential run): confirmed
```

## Chapter Summary

A Monte Carlo risk and pricing engine needed no new parallel primitive -- it needed three of Part 1's tools applied at the right scale and in the right composition. Section 33.1 showed that simulating many independent price paths is Chapter 5's scan, but reshaped: many small, independent scans (one per path, one thread each) rather than one large collaborative one, because Monte Carlo's defining property is that paths share no state. Section 33.2 showed that averaging simulated payoffs into a single price estimate is Chapter 4's reduction in its most literal form -- plain numeric addition, no combine-operator generalization needed, just a mean computed as a sum followed by one division. Section 33.3 showed that finding the worst-case scenarios for risk analysis is Chapter 6's stream compaction, with a simpler atomicAdd-cursor variant standing in for the full predicate+scan+scatter pipeline whenever only the compacted set, not its exact order, is required.

## Self-Check Questions

1. Why does Section 33.1's use of scan look different in SHAPE from Chapter 5's original scan, even though both are computing the same underlying operation?
2. What specific property of Monte Carlo simulation makes it possible for Section 33.1's kernel to launch with zero atomics and zero synchronization?
3. Why does Section 33.2's reduction need no new combine operator, unlike Section 31.1's box-union reduction?
4. What would have to change about Section 33.2's reduction tree if the number of simulated paths were not already a power of two, and where in the book was that exact adjustment already established?
5. What does Section 33.3's atomicAdd-cursor compaction kernel guarantee about its output, and what does it explicitly NOT guarantee, compared to Chapter 6's original scan-based compaction?
6. A risk manager complains that Section 33.3's kernel produced tail-risk paths in a different order than the CPU baseline did. Is this a bug? Why or why not?

## Where We Go Next

A Monte Carlo engine treats every simulated scenario as an independent, self-contained sequence of numbers -- but genomic data is neither independent nor purely numeric: it is long strings drawn from a tiny alphabet, where what matters is how they align and overlap with each other. Chapter 34 turns to large-scale genomic sequence alignment and k-mer counting, reusing Part 5's GPU hash tables to count and match substrings at a scale no sequential scan of a genome could keep up with -- the next of the case studies that close out Part 8.

## Worked Solutions

**1.** Chapter 5's original scan solves one long sequential dependency chain by extracting parallelism WITHIN it, using many threads to cooperatively compute a single running combine. Section 33.1 instead has many short, completely independent chains (one per simulated path), so the useful parallelism is ACROSS paths rather than within any single one -- each thread computes its own short scan alone, with the underlying operation (a running prefix sum) identical in both cases, only the parallelization strategy differing because the shape of the workload differs.

**2.** Monte Carlo paths are, by construction, statistically and computationally independent of one another -- no path's simulated trajectory depends on reading or writing any other path's data. Since a race condition can only occur when multiple threads access the same shared memory location, and no thread here ever touches another thread's path data, there is no possible race to protect against, which is exactly why the kernel needs no atomics and no synchronization at all.

**3.** Section 31.1's box union needed a new combine operator because its payload (a bounding box, four numbers) was not the plain scalar Chapter 4's reduction was originally built for. Section 33.2's payload is already a single number (a payoff value), and averaging numbers via addition is precisely the operation Chapter 4's reduction was designed around from the start, so no generalization or adaptation of the combine function is needed -- only the specific numbers being reduced change.

**4.** With a non-power-of-two path count, the reduction tree would need padding with identity-value entries (payoffs of zero, which do not affect a sum) up to the next power of two, so that every round can still pair up neighbors cleanly with no leftover unpaired element. This exact adjustment -- padding to a power of two with a sentinel/identity value -- was already established in Section 32.2's packed-key MIN reduction (using a sentinel infinite price for empty, non-participating levels) and traces back to the same general principle Section 31.3's `[COMMON TRAP]` raised about non-power-of-two leaf counts.

**5.** The atomicAdd-cursor kernel guarantees that every path whose predicate is true claims exactly one, distinct output slot, and that the resulting compacted array contains exactly the correct SET of matching path indices with no duplicates and no omissions. It explicitly does NOT guarantee that the compacted array's order matches the original input order, because which matching thread's atomicAdd executes first depends on real hardware timing, not on path index -- unlike Chapter 6's original predicate+scan+scatter pipeline, which computes each match's output position from a deterministic exclusive scan and therefore always preserves input order.

**6.** This is not a bug. Section 33.3's `[COMMON TRAP]` explicitly establishes that the atomicAdd-cursor kernel's correctness guarantee covers the compacted SET (which paths are present in the output), not their positions -- the self-check accordingly compares sorted output against the expected set, exactly as Section 30.2's particle-scatter self-check did for the same reason. If a risk workflow genuinely needs a stable, input-order-preserving compaction, Chapter 6's full predicate+exclusive-scan+scatter pipeline -- used by this section's own CPU baseline -- is the tool that provides that stronger guarantee; the atomicAdd-cursor variant is a deliberate, simpler trade-off for when order does not matter.
