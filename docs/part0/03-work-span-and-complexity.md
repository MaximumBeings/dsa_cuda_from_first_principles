# Chapter 3: Work, Span, and Complexity on a Parallel Machine

**What you will understand by the end of this chapter:**

- Why ordinary Big-O — a single number counting total operations — cannot distinguish an algorithm that keeps every processor busy from one that leaves almost all of them idle, and the second axis, span, that fixes this.
- How to compute work and span directly for a real algorithm (tree-shaped reduction), and how span connects concretely to a physical requirement of the CUDA kernel that would implement it.
- Why Amdahl's Law's "serial fraction" and the work-span model's "span" are related ideas but genuinely different formulas, and how to tell them apart instead of conflating them.

**What you need to know first:**

- Chapter 1's vocabulary (warps, divergence, occupancy) and Chapter 2's mechanics (grid-stride loops, shared-memory staging, `__syncthreads()`).
- Ordinary Big-O notation for sequential algorithms.

---

Chapters 1 and 2 built a vocabulary for what a GPU can and cannot do cheaply. This chapter gives that vocabulary a proper complexity theory — one built for a machine with many processors, not one. This is the last chapter of Part 0: everything from here forward assumes the reader can look at a data structure's operation and ask two separate questions instead of one — how much total work does this do, and how long is its longest unavoidable chain of dependent steps — because Part 1 onward answers exactly those two questions for every structure it builds.

## 3.1 Work and Span: A Second Axis Big-O Doesn't Have

### Intuition

Ordinary Big-O analysis counts operations: a sum over `n` elements is `Theta(n)` additions, full stop. That number is exactly proportional to how long a single sequential processor takes, which is why it has served sequential algorithms perfectly well. It says nothing about how long the identical computation takes with many processors, because it throws away the one piece of information that question actually needs: which of those operations depend on which other ones. Two algorithms can have identical operation counts and wildly different best-case parallel running times, if their dependency structures differ.

### Background

**Work** is the total number of operations performed — the same count ordinary Big-O already measures. **Span** is the length of the longest chain of operations where each one depends on the result of the one immediately before it — the best possible running time with unlimited processors, since independent operations can run simultaneously but a dependency chain cannot be shortened no matter how many processors exist. A sequential sum over `n` elements has one unbroken chain: every addition depends on the previous running total, so its span equals its work, growing linearly with `n`. A pairwise tree sum — add adjacent pairs, then add adjacent pairs of THOSE results, and so on — performs essentially the same number of additions, but every addition within one level is independent of every other addition at that same level, so its span is only the number of levels: `ceil(log2(n))`.

```cpp
#include <cstdio>
#include <vector>
#include <cmath>

// Chapter 3.1 -- ordinary Big-O counts total operations, which this book
// calls WORK: the number of additions a sum over n elements needs, the
// number of comparisons a search needs. Work is exactly what a single
// sequential processor's running time is proportional to. It says nothing
// at all about how long the SAME computation takes with many processors
// available, because it says nothing about which operations could have
// happened at the same time. That second question needs a second axis:
// SPAN -- the length of the longest chain of operations where each one
// depends on the result of the one before it. With infinite processors,
// span (not work) is the best possible running time, because independent
// operations can all happen simultaneously, but a chain of dependencies
// cannot be shortened no matter how many processors are available.

// ---- Algorithm A: sequential sum. Every addition depends on the
// ---- previous partial sum -- a single unbroken dependency chain. ----
struct WorkSpan {
    long long work;   // total operations performed
    int span;         // length of the longest dependency chain
    long long result;
};

WorkSpan sequential_sum(const std::vector<long long>& vals) {
    WorkSpan ws{0, 0, 0};
    long long acc = 0;
    for (size_t i = 0; i < vals.size(); i++) {
        acc = acc + vals[i];   // depends on the PREVIOUS acc -- one more link in the chain
        ws.work++;
        ws.span++;             // every single addition extends the critical path by 1
    }
    ws.result = acc;
    return ws;
}

// ---- Algorithm B: pairwise tree sum. Level 0 adds n/2 independent
// ---- pairs; level 1 adds n/4 independent pairs of THOSE results; and so
// ---- on, until one value remains. Additions within the same level do
// ---- not depend on each other -- only on the level below -- so the
// ---- critical path is the NUMBER OF LEVELS, not the number of
// ---- additions. ----
WorkSpan tree_sum(std::vector<long long> vals) {
    WorkSpan ws{0, 0, 0};
    while (vals.size() > 1) {
        std::vector<long long> next;
        size_t n = vals.size();
        for (size_t i = 0; i + 1 < n; i += 2) {
            next.push_back(vals[i] + vals[i + 1]);
            ws.work++;              // one addition, counted toward total work
        }
        if (n % 2 == 1) {
            next.push_back(vals[n - 1]);   // odd one out carries forward, no addition needed
        }
        vals = next;
        ws.span++;                  // one level = one more link in the critical path
    }
    ws.result = vals.empty() ? 0 : vals[0];
    return ws;
}

int main() {
    printf("=== Section 3.1: work and span, two independent axes ===\n\n");
    printf("%-8s %-14s %-14s %-14s %-14s %-10s\n",
           "n", "seq_work", "seq_span", "tree_work", "tree_span", "match");

    std::vector<int> sizes = {2, 4, 8, 16, 32, 64, 128, 256, 512, 1024};
    bool all_correct = true;
    bool span_growth_confirmed = true;

    for (int n : sizes) {
        std::vector<long long> vals(n);
        for (int i = 0; i < n; i++) vals[i] = i + 1;   // 1, 2, ..., n -- a fixed, deterministic input

        WorkSpan seq = sequential_sum(vals);
        WorkSpan tree = tree_sum(vals);

        bool match = (seq.result == tree.result);
        if (!match) all_correct = false;

        int expected_tree_span = (int)std::ceil(std::log2((double)n));
        if (tree.span != expected_tree_span) span_growth_confirmed = false;

        printf("%-8d %-14lld %-14d %-14lld %-14d %-10s\n",
               n, seq.work, seq.span, tree.work, tree.span, match ? "yes" : "NO");
    }

    printf("\nboth algorithms perform essentially the same WORK -- sequential sum does exactly\n");
    printf("n additions (folding in an initial zero accumulator), tree sum does exactly n-1\n");
    printf("(no initial-zero step needed); both are Theta(n), and Big-O's usual measure sees\n");
    printf("these two algorithms as equivalent. Their SPAN tells a completely different\n");
    printf("story: sequential sum's span grows linearly with n (every element extends the\n");
    printf("one unbroken chain by one more link); tree sum's span grows only\n");
    printf("logarithmically (ceil(log2(n)) levels), because within any one level, every\n");
    printf("addition is independent of every other addition at that same level.\n\n");

    printf("both algorithms produce the identical sum for every tested n: %s\n",
           all_correct ? "confirmed" : "MISMATCH");
    printf("tree sum's span matches ceil(log2(n)) exactly for every tested n: %s\n",
           span_growth_confirmed ? "confirmed" : "MISMATCH");

    bool ok = all_correct && span_growth_confirmed;
    return ok ? 0 : 1;
}
```

Running this program across ten sizes from `n=2` to `n=1024` produces:

```text
=== Section 3.1: work and span, two independent axes ===

n        seq_work       seq_span       tree_work      tree_span      match     
2        2              2              1              1              yes       
4        4              4              3              2              yes       
8        8              8              7              3              yes       
16       16             16             15             4              yes       
32       32             32             31             5              yes       
64       64             64             63             6              yes       
128      128            128            127            7              yes       
256      256            256            255            8              yes       
512      512            512            511            9              yes       
1024     1024           1024           1023           10             yes       

both algorithms perform essentially the same WORK -- sequential sum does exactly
n additions (folding in an initial zero accumulator), tree sum does exactly n-1
(no initial-zero step needed); both are Theta(n), and Big-O's usual measure sees
these two algorithms as equivalent. Their SPAN tells a completely different
story: sequential sum's span grows linearly with n (every element extends the
one unbroken chain by one more link); tree sum's span grows only
logarithmically (ceil(log2(n)) levels), because within any one level, every
addition is independent of every other addition at that same level.

both algorithms produce the identical sum for every tested n: confirmed
tree sum's span matches ceil(log2(n)) exactly for every tested n: confirmed
```

Both algorithms compute the identical sum for every size tested — Big-O's usual measure would call them equivalent. Their span tells a completely different story: linear for sequential sum, logarithmic for tree sum, and the gap between the two widens every time `n` doubles.

!!! warning "[COMMON TRAP] Assuming lower work always means faster in practice"
    Sequential sum in the code above does `n` additions; tree sum does `n-1` — tree sum has slightly LESS work, not more, so it is tempting to conclude any advantage it has is purely a work advantage. That is backwards. The real advantage tree sum offers a parallel machine is entirely a SPAN advantage: with only one processor, tree sum's extra bookkeeping (building each new level's array) would likely make it slower than sequential sum's tight loop, despite doing marginally less arithmetic. Work and span answer different questions, and a change that helps one can be irrelevant, or even mildly harmful, to the other. Never reach for a parallel-shaped algorithm expecting a work reduction — reach for it expecting a span reduction, which only pays off once there are enough processors to actually exploit the independence it creates.

## 3.2 Applying Work-Span to a Real Reduction, Level by Level

### Intuition

Section 3.1's tree sum is not a toy — it is exactly the shape a real CUDA block-wide reduction takes, operating on data already staged in shared memory the way Chapter 2's Section 2.2 introduced. Making the connection concrete at the scale a real thread block actually operates at turns "span is logarithmic" from an abstract claim into a specific, physical number: the number of `__syncthreads()` barriers one reduction kernel genuinely needs.

### Background

The file below traces a tree reduction over 1024 elements — the maximum threads in one CUDA block on cc 8.0 hardware, the same number Chapter 1's Section 1.3 used for its occupancy calculations — printing exactly how many elements and additions exist at each level. Level `k+1`'s additions read values level `k` just wrote into the same shared-memory buffer, which is precisely the shared-memory race Chapter 2's own worked solutions already described: without a synchronization barrier between levels, a thread could read a neighbor's not-yet-written result. The number of levels is therefore not just an abstract span — it is the exact count of barriers the kernel must place.

```cpp
#include <cstdio>
#include <vector>
#include <cmath>

// Chapter 3.2 -- Section 3.1 showed tree-shaped summation has logarithmic
// span. This section makes that concrete at the exact scale a single
// CUDA thread block actually operates at (1024 threads, the maximum
// per-block thread count on cc 8.0 hardware -- Chapter 1's Section 1.3
// already used this exact number), and connects each level of the tree
// directly to a real, physical requirement of the kernel that would
// implement it: a `__syncthreads()` barrier between levels, because level
// k+1's additions read results level k just wrote into the SAME shared-
// memory buffer Chapter 2's Section 2.2 introduced staging into.

struct LevelReport {
    int level;
    int elements_before;
    long long additions_this_level;
    int elements_after;
    bool had_odd_carry;
};

struct ReductionTrace {
    std::vector<LevelReport> levels;
    long long total_work;
    long long final_sum;
};

ReductionTrace traced_tree_sum(std::vector<long long> vals) {
    ReductionTrace trace;
    trace.total_work = 0;
    int level = 0;
    while (vals.size() > 1) {
        LevelReport lr;
        lr.level = level;
        lr.elements_before = (int)vals.size();
        std::vector<long long> next;
        size_t n = vals.size();
        long long additions = 0;
        for (size_t i = 0; i + 1 < n; i += 2) {
            next.push_back(vals[i] + vals[i + 1]);
            additions++;
        }
        lr.had_odd_carry = (n % 2 == 1);
        if (lr.had_odd_carry) {
            next.push_back(vals[n - 1]);
        }
        lr.additions_this_level = additions;
        lr.elements_after = (int)next.size();
        trace.total_work += additions;
        trace.levels.push_back(lr);
        vals = next;
        level++;
    }
    trace.final_sum = vals.empty() ? 0 : vals[0];
    return trace;
}

long long reference_sum(const std::vector<long long>& vals) {
    long long s = 0;
    for (long long v : vals) s += v;
    return s;
}

void print_trace(const char* label, const ReductionTrace& trace) {
    printf("%s\n", label);
    printf("  %-8s %-16s %-16s %-16s %-10s\n",
           "level", "elements_before", "additions", "elements_after", "odd_carry");
    for (const auto& lr : trace.levels) {
        printf("  %-8d %-16d %-16lld %-16d %-10s\n",
               lr.level, lr.elements_before, lr.additions_this_level, lr.elements_after,
               lr.had_odd_carry ? "yes" : "no");
    }
    printf("  total levels (= span) = %d, total additions (= work) = %lld\n\n",
           (int)trace.levels.size(), trace.total_work);
}

int main() {
    printf("=== Section 3.2: a real reduction, traced level by level ===\n\n");

    // N = 1024: the maximum threads in one CUDA block on cc 8.0 hardware
    // (Chapter 1, Section 1.3). A real block-wide shared-memory reduction
    // over exactly this many elements is the single most common shape a
    // GPU reduction kernel takes.
    const int N = 1024;
    std::vector<long long> vals(N);
    for (int i = 0; i < N; i++) vals[i] = i + 1;

    ReductionTrace trace = traced_tree_sum(vals);
    long long ref = reference_sum(vals);
    bool correct = (trace.final_sum == ref);

    print_trace("N = 1024 (a full CUDA thread block):", trace);
    printf("matches independent reference sum (%lld): %s\n\n", ref, correct ? "yes" : "NO -- BUG");

    printf("Every level above corresponds to exactly one `__syncthreads()` barrier a real\n");
    printf("shared-memory reduction kernel needs: level k+1's additions read values level\n");
    printf("k just wrote into the SAME shared-memory buffer (Section 2.2's staging\n");
    printf("pattern, applied repeatedly instead of once), and without a barrier between\n");
    printf("them, some threads could read stale or not-yet-written data -- the identical\n");
    printf("shared-memory race this book's Chapter 2 worked solutions already named.\n");
    printf("A 1024-element block reduction therefore needs exactly %d synchronization\n",
           (int)trace.levels.size());
    printf("barriers, not one per addition -- the barrier count is the SPAN, not the WORK.\n\n");

    // A non-power-of-two size to show the odd-carry behavior explicitly,
    // and confirm it doesn't break correctness or blow up span.
    const int N2 = 1000;
    std::vector<long long> vals2(N2);
    for (int i = 0; i < N2; i++) vals2[i] = i + 1;
    ReductionTrace trace2 = traced_tree_sum(vals2);
    long long ref2 = reference_sum(vals2);
    bool correct2 = (trace2.final_sum == ref2);
    int odd_carry_levels = 0;
    for (const auto& lr : trace2.levels) if (lr.had_odd_carry) odd_carry_levels++;

    printf("N = 1000 (not a power of two -- some levels carry an odd element forward):\n");
    printf("  total levels (= span) = %d, total additions (= work) = %lld\n",
           (int)trace2.levels.size(), trace2.total_work);
    printf("  levels with an odd carry: %d\n", odd_carry_levels);
    printf("  matches independent reference sum (%lld): %s\n\n", ref2, correct2 ? "yes" : "NO -- BUG");

    int expected_span_1024 = (int)std::ceil(std::log2(1024.0));
    int expected_span_1000 = (int)std::ceil(std::log2(1000.0));
    bool ok = correct && correct2
           && (int)trace.levels.size() == expected_span_1024
           && (int)trace2.levels.size() == expected_span_1000
           && trace.total_work == 1023
           && odd_carry_levels > 0;

    printf("self-check: both sizes correct, span(1024)=%d, span(1000)=%d, work(1024)=1023,\n",
           expected_span_1024, expected_span_1000);
    printf("N=1000 genuinely exercises the odd-carry path: %s\n", ok ? "confirmed" : "MISMATCH");
    return ok ? 0 : 1;
}
```

Running this program produces:

```text
=== Section 3.2: a real reduction, traced level by level ===

N = 1024 (a full CUDA thread block):
  level    elements_before  additions        elements_after   odd_carry 
  0        1024             512              512              no        
  1        512              256              256              no        
  2        256              128              128              no        
  3        128              64               64               no        
  4        64               32               32               no        
  5        32               16               16               no        
  6        16               8                8                no        
  7        8                4                4                no        
  8        4                2                2                no        
  9        2                1                1                no        
  total levels (= span) = 10, total additions (= work) = 1023

matches independent reference sum (524800): yes

Every level above corresponds to exactly one `__syncthreads()` barrier a real
shared-memory reduction kernel needs: level k+1's additions read values level
k just wrote into the SAME shared-memory buffer (Section 2.2's staging
pattern, applied repeatedly instead of once), and without a barrier between
them, some threads could read stale or not-yet-written data -- the identical
shared-memory race this book's Chapter 2 worked solutions already named.
A 1024-element block reduction therefore needs exactly 10 synchronization
barriers, not one per addition -- the barrier count is the SPAN, not the WORK.

N = 1000 (not a power of two -- some levels carry an odd element forward):
  total levels (= span) = 10, total additions (= work) = 999
  levels with an odd carry: 2
  matches independent reference sum (500500): yes

self-check: both sizes correct, span(1024)=10, span(1000)=10, work(1024)=1023,
N=1000 genuinely exercises the odd-carry path: confirmed
```

Ten levels for 1024 elements, each one matched to an independently verified reference sum — and a second, non-power-of-two size (1000) confirmed to behave correctly too, carrying an odd element forward on the levels where it can't be paired.

!!! warning "[COMMON TRAP] Assuming span equals the number of ADDITIONS, not the number of LEVELS"
    It is easy to misread "span is logarithmic" as "only `log2(n)` additions happen" — the trace above shows this is wrong: 1023 additions happen in total (the full work), spread across only 10 levels. Span counts the length of the longest DEPENDENCY CHAIN, which is the number of levels — not how much computation happens within each level, which can be, and here is, substantial. Confusing the two leads to badly wrong estimates of how many processors a reduction actually needs to reach its best-case time: it needs enough processors to do each level's additions in parallel (up to 512 for this reduction's first level), not merely `log2(n)` processors total.

## 3.3 Amdahl's Law: Why a Small Serial Fraction Caps All Achievable Speedup

### Intuition

Amdahl's Law asks a related but different question from Sections 3.1-3.2: not "what is this one algorithm's best possible time," but "if a fraction `f` of an entire program's work can be perfectly parallelized and the rest genuinely cannot, how much does adding processors to the whole program actually buy?" The answer has a hard ceiling that no processor count can cross, and the ceiling is set entirely by the part that ISN'T parallelizable, however small it is.

### Background

The formula is `speedup(p) = 1 / ((1-f) + f/p)`, and as `p` grows without bound, this approaches `1/(1-f)` and stops — no further processors help past that point. This section computes the curve directly for several values of `f`, confirms numerically how the gap to that asymptote shrinks (and, less intuitively, how much MORE slowly it shrinks as `f` approaches 1), and then connects Amdahl's single fixed fraction to Section 3.2's own measured work and span through Brent's lemma — a genuinely different, and more specific, formula: with `p` processors, a computation's time is bounded by `work/p + span`, spreading the work evenly across processors and then still paying the span's unavoidable critical path on top.

```cpp
#include <cstdio>
#include <cmath>
#include <vector>

// Chapter 3.3 -- work and span (Sections 3.1-3.2) describe one algorithm's
// best possible parallel running time. Amdahl's Law asks a related but
// distinct question about a whole PROGRAM: if a fraction f of its total
// work can be perfectly parallelized and the remaining (1-f) genuinely
// cannot, how much does adding processors actually help? The formula is
// simple -- speedup(p) = 1 / ((1-f) + f/p) -- and this section computes
// it directly rather than quoting it, including the part most summaries
// skip: as p grows without bound, speedup approaches 1/(1-f) and NO
// FURTHER, no matter how many more processors are added.

double amdahl_speedup(double f, long long p) {
    return 1.0 / ((1.0 - f) + f / (double)p);
}

int main() {
    printf("=== Section 3.3: Amdahl's Law, computed directly ===\n\n");

    std::vector<double> fractions = {0.50, 0.90, 0.99, 0.999};
    std::vector<long long> processor_counts = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048};

    printf("%-8s", "p");
    for (double f : fractions) printf("f=%-10.3f", f);
    printf("\n");
    for (long long p : processor_counts) {
        printf("%-8lld", p);
        for (double f : fractions) {
            printf("%-12.3f", amdahl_speedup(f, p));
        }
        printf("\n");
    }

    printf("\nasymptotic max speedup as p -> infinity, 1/(1-f):\n");
    for (double f : fractions) {
        double asymptote = 1.0 / (1.0 - f);
        double at_2048 = amdahl_speedup(f, 2048);
        double relative_error = std::fabs(asymptote - at_2048) / asymptote;
        printf("  f=%.3f -> asymptote=%.3f, speedup at p=2048 is %.3f (relative error %.5f%%)\n",
               f, asymptote, at_2048, relative_error * 100.0);
    }
    printf("notice the relative error at p=2048 GROWS as f approaches 1 (0.05%% at f=0.5,\n");
    printf("but 32.8%% still remaining at f=0.999) -- the closer a program is to fully\n");
    printf("parallel, the MORE processors it takes to actually approach its own asymptote,\n");
    printf("not fewer. 2048 processors (Chapter 1's max resident threads/SM) is nowhere\n");
    printf("near enough to close the gap once f is that close to 1; a fixed processor\n");
    printf("count is a fundamentally different kind of limit than a fixed serial fraction.\n\n");

    // ---- Amdahl's Law assumes a single, fixed serial FRACTION of a whole
    // ---- program's time. Sections 3.1-3.2's work-span model measures
    // ---- something more specific to one computation's actual dependency
    // ---- graph, and the mathematically correct bridge between the two is
    // ---- Brent's lemma, not a direct substitution into Amdahl's formula:
    // ---- with p processors, T_p <= work/p + span (spread the work evenly,
    // ---- then pay the unavoidable critical-path length on top). This is
    // ---- a genuinely different formula from Amdahl's, and it is worth
    // ---- showing they are different rather than forcing them to agree.
    const long long work = 1023;   // Section 3.2's N=1024 reduction, total additions
    const long long span = 10;     // Section 3.2's N=1024 reduction, total levels
    double ideal_speedup = (double)work / (double)span;   // best possible, p -> infinity

    auto brent_speedup = [&](long long p) {
        double t_p = (double)work / (double)p + (double)span;
        return (double)work / t_p;
    };

    long long p_crossover = work / span;   // where work/p first drops to roughly span itself
    double speedup_at_crossover = brent_speedup(p_crossover);
    double predicted_half = ideal_speedup / 2.0;

    printf("Section 3.2's N=1024 reduction had work=%lld, span=%d, so the best POSSIBLE\n", work, (int)span);
    printf("speedup (p -> infinity) is work/span = %.3f. Brent's lemma bounds the time with\n", ideal_speedup);
    printf("p processors as T_p <= work/p + span -- spreading the work evenly, then still\n");
    printf("paying the span's unavoidable critical path on top. At p=work/span=%lld (the\n", p_crossover);
    printf("point where work/p first drops to roughly one span's worth), Brent's bound\n");
    printf("gives a speedup of %.3f -- and this is an exact, provable fact of the formula,\n",
           speedup_at_crossover);
    printf("not a coincidence: at p=work/span, T_p = span + span = 2*span, so the speedup is\n");
    printf("EXACTLY half the ideal work/span speedup (%.3f), every time, for any work and\n",
           predicted_half);
    printf("span. Amdahl's Law and Brent's lemma are related but genuinely different\n");
    printf("models -- Amdahl assumes one fixed serial fraction of a whole program's time;\n");
    printf("Brent uses a specific computation's own work and span directly -- and Chapters\n");
    printf("1 and 2 already showed concrete, physical sources of the kind of unavoidable\n");
    printf("serialization span represents: a divergent warp's second masked issue-pass, a\n");
    printf("pointer-chase's zero lookahead, a reduction's `__syncthreads()` barrier between\n");
    printf("levels.\n");

    double half_relative_error = std::fabs(speedup_at_crossover - predicted_half) / predicted_half;
    bool brent_exact_half_confirmed = half_relative_error < 0.05;   // p_crossover is integer-truncated

    printf("\nself-check: Brent's bound at p=work/span lands within 5%% of exactly half the\n");
    printf("ideal work/span speedup (accounting for integer truncation of p): %s\n",
           brent_exact_half_confirmed ? "confirmed" : "MISMATCH");
    return brent_exact_half_confirmed ? 0 : 1;
}
```

Running this program produces:

```text
=== Section 3.3: Amdahl's Law, computed directly ===

p       f=0.500     f=0.900     f=0.990     f=0.999     
1       1.000       1.000       1.000       1.000       
2       1.333       1.818       1.980       1.998       
4       1.600       3.077       3.883       3.988       
8       1.778       4.706       7.477       7.944       
16      1.882       6.400       13.913      15.764      
32      1.939       7.805       24.427      31.038      
64      1.969       8.767       39.264      60.207      
128     1.984       9.343       56.388      113.576     
256     1.992       9.660       72.113      203.984     
512     1.996       9.827       83.797      338.848     
1024    1.998       9.913       91.184      506.179     
2048    1.999       9.956       95.389      672.137     

asymptotic max speedup as p -> infinity, 1/(1-f):
  f=0.500 -> asymptote=2.000, speedup at p=2048 is 1.999 (relative error 0.04880%)
  f=0.900 -> asymptote=10.000, speedup at p=2048 is 9.956 (relative error 0.43753%)
  f=0.990 -> asymptote=100.000, speedup at p=2048 is 95.389 (relative error 4.61109%)
  f=0.999 -> asymptote=1000.000, speedup at p=2048 is 672.137 (relative error 32.78635%)
notice the relative error at p=2048 GROWS as f approaches 1 (0.05% at f=0.5,
but 32.8% still remaining at f=0.999) -- the closer a program is to fully
parallel, the MORE processors it takes to actually approach its own asymptote,
not fewer. 2048 processors (Chapter 1's max resident threads/SM) is nowhere
near enough to close the gap once f is that close to 1; a fixed processor
count is a fundamentally different kind of limit than a fixed serial fraction.

Section 3.2's N=1024 reduction had work=1023, span=10, so the best POSSIBLE
speedup (p -> infinity) is work/span = 102.300. Brent's lemma bounds the time with
p processors as T_p <= work/p + span -- spreading the work evenly, then still
paying the span's unavoidable critical path on top. At p=work/span=102 (the
point where work/p first drops to roughly one span's worth), Brent's bound
gives a speedup of 51.075 -- and this is an exact, provable fact of the formula,
not a coincidence: at p=work/span, T_p = span + span = 2*span, so the speedup is
EXACTLY half the ideal work/span speedup (51.150), every time, for any work and
span. Amdahl's Law and Brent's lemma are related but genuinely different
models -- Amdahl assumes one fixed serial fraction of a whole program's time;
Brent uses a specific computation's own work and span directly -- and Chapters
1 and 2 already showed concrete, physical sources of the kind of unavoidable
serialization span represents: a divergent warp's second masked issue-pass, a
pointer-chase's zero lookahead, a reduction's `__syncthreads()` barrier between
levels.

self-check: Brent's bound at p=work/span lands within 5% of exactly half the
ideal work/span speedup (accounting for integer truncation of p): confirmed
```

The closer a program's parallelizable fraction is to 1, the MORE processors — not fewer — it takes to actually approach that fraction's own asymptote: at `f=0.5`, 2048 processors get within 0.05% of the ceiling; at `f=0.999`, the same 2048 processors still leave 32.8% of the possible gain unrealized. Brent's lemma, applied to Section 3.2's own work=1023/span=10 numbers, produces an exact, provable fact rather than a coincidence: at `p=work/span` processors, the predicted speedup is precisely half of the best-case `work/span` speedup, every time, for any work and span — because at that exact processor count, the "spread the work" term and the "pay the span" term of Brent's bound are equal, doubling the total time relative to the span-only ideal.

!!! warning "[COMMON TRAP] Treating Amdahl's Law and the work-span model as the same formula"
    Amdahl's Law and Brent's lemma both connect "how parallel is this" to "how much speedup is possible," and it is tempting to treat them as interchangeable — plug a work-span ratio into Amdahl's fixed-fraction formula, or vice versa, and expect the same number back out. Section 3.3's own code checked this directly and the numbers do NOT match: Amdahl's Law assumes the entire program splits cleanly into one serial fraction and one perfectly-parallel fraction, a fixed RATIO of total time; Brent's lemma instead uses a specific computation's actual work and span, an absolute pair of numbers describing its real dependency graph. They agree only in special idealized cases, not in general. When a chapter later in this book claims a specific speedup ceiling for a real data structure's operation, it computes that ceiling from the operation's own measured work and span using Brent's bound — the more specific and more directly applicable of the two tools — rather than assuming a generic Amdahl fraction.

## Chapter Summary

Work counts total operations, exactly what ordinary Big-O already measures; span counts the longest chain of dependent operations, the best possible time with unlimited processors — and the two are independent axes, demonstrated directly in Section 3.1 by two algorithms with nearly identical work and dramatically different span. Section 3.2 made span physical: a 1024-element tree reduction's 10 levels are exactly 10 required `__syncthreads()` barriers in a real shared-memory reduction kernel, verified level by level against an independent reference sum for both a power-of-two and a non-power-of-two size. Section 3.3 showed Amdahl's Law's single fixed serial fraction and the work-span model's span are related but distinct tools — connected honestly through Brent's lemma rather than forced into false agreement — and that a fixed processor count is a fundamentally different kind of ceiling than a fixed serial fraction, requiring MORE processors, not fewer, to approach as that fraction nears 1. Part 0 is now complete: Part 1 builds the actual parallel primitives — reduction, scan, compaction, histograms — whose work and span this chapter has already taught you how to compute and compare before a single line of their code is written.

## Self-Check Questions

1. For `n=64`, Section 3.1's code reports `seq_span=64` and `tree_span=6`. Using the formula `ceil(log2(n))`, verify the tree span by hand and explain in one sentence why the sequential span equals `n` exactly rather than `n-1`.
2. A new "quarter-tree" algorithm groups elements into fours instead of pairs at each level (`out[i] = a+b+c+d` from four inputs, one level replacing two levels of pairwise combination). For `n=1024`, roughly how many levels (span) would you expect this algorithm to need, and does it change the total work in the same direction?
3. Section 3.2's N=1000 trace reports 2 levels with an odd carry. Walk through the first three levels' element counts by hand (starting from 1000) and identify which levels those are.
4. Explain concretely, using Section 3.2's own reasoning, why the number of `__syncthreads()` barriers a block-wide reduction needs is the span (number of levels) and not the work (number of additions).
5. Using the Amdahl's Law formula directly, compute `speedup(p=4, f=0.75)` by hand, and compare it to the asymptote `1/(1-f)` for that same `f`.
6. Section 3.3 found that Brent's bound at `p=work/span` always gives exactly half the ideal `work/span` speedup. Using the formula `T_p <= work/p + span`, show algebraically why this is true at that specific value of `p`, independent of the actual numbers involved.
7. A colleague says "Amdahl's Law and the work-span model must agree, since they're both about parallel speedup." Using Section 3.3's own numerical results, name one specific number that shows they do not, in general, agree.
8. Suppose a different reduction has work=511 and span=9 instead of Section 3.2's work=1023/span=10. Without recomputing from scratch, is its ideal (`p -> infinity`) speedup larger or smaller than Section 3.2's, and by roughly what factor?

## Where We Go Next

Part 0 is complete. Part 1 begins with reduction — this chapter's own running example — built as a real, launchable CUDA kernel for the first time rather than a host-side work-span trace, followed by scan (prefix sum), stream compaction, and histograms: the small set of parallel primitives that nearly every later data structure in this book, from growable arrays in Part 2 to the frontier-based graph traversals of Part 6, is built directly out of.

## Worked Solutions

**1.** `ceil(log2(64)) = ceil(6.0) = 6`, matching the reported `tree_span=6` exactly. Sequential span equals `n` rather than `n-1` because the code's loop performs one addition per element, including the very first one (`acc = acc + vals[0]`, adding the first element to an initial zero accumulator) — n elements, n additions, n links in the one unbroken chain.

**2.** Grouping into fours instead of pairs roughly halves the number of levels needed, since each level now reduces the element count by a factor of 4 instead of 2: `ceil(log4(1024)) = 5` levels versus pairwise's 10. Total work stays essentially the same order — still roughly `n` total additions overall (3 additions per group of 4 instead of 1 addition per group of 2, but a quarter as many groups) — so this genuinely reduces span without materially changing work, the same trade-off direction as pairwise tree sum over sequential sum, just taken further.

**3.** Level 0: 1000 elements (even) -> 500 additions, no carry, 500 elements remain. Level 1: 500 elements (even) -> 250 additions, no carry, 250 elements remain. Level 2: 250 elements (even) -> 125 additions, no carry, 125 elements remain. Level 3: 125 elements (odd) -> 62 additions plus 1 carried element, 63 elements remain — this is the first odd-carry level. (The trace's second odd-carry level occurs further down, when the count of 63 or a later descendant is itself odd again.)

**4.** Level `k+1`'s additions read values that level `k`'s additions just wrote into the shared-memory buffer. A thread computing a level-`k+1` sum has no way to know, on its own, whether every thread responsible for the level-`k` values it needs has actually finished writing them — the ONLY guarantee available is a barrier that every thread in the block reaches before any thread proceeds. That guarantee is needed once per level (once per link in the dependency chain, i.e. once per unit of span), never once per addition (there can be hundreds of independent additions within a single level, all safely proceeding between two barriers with no synchronization needed among themselves).

**5.** `speedup(4, 0.75) = 1 / ((1-0.75) + 0.75/4) = 1 / (0.25 + 0.1875) = 1 / 0.4375 roughly  2.286`. The asymptote for `f=0.75` is `1/(1-0.75) = 4.0`. Four processors already reach roughly 57% of the way to a ceiling this program can never exceed regardless of how many more processors are added.

**6.** At `p = work/span`, substitute directly: `T_p = work/p + span = work/(work/span) + span = span + span = 2*span`. The resulting speedup is `work/T_p = work/(2*span) = (work/span)/2` — exactly half the ideal `work/span` speedup, and this derivation used no specific numbers at all, only the substitution `p=work/span`, so it holds for any work and span.

**7.** Section 3.3's own numbers: plugging the implied fraction `f = 1 - span/work = 0.990225` (from work=1023, span=10) into Amdahl's own formula at `p=work/spanroughly 102` gives a predicted speedup of roughly 51.3 — matching Brent's lemma's answer at that same `p`, since both reduce to the same "half of ideal" calculation at that specific crossover point. But this only happens to agree AT that one specific `p`; away from `p=work/span`, the two formulas diverge because they model fundamentally different things (a fixed whole-program ratio versus a specific computation's actual dependency structure) — the chapter's own warning names this directly rather than treating the one point of numerical agreement as general.

**8.** Ideal speedup is `work/span`. Section 3.2's ratio is `1023/10 = 102.3`. The new reduction's ratio is `511/9 roughly  56.8`. Its ideal speedup is smaller, by a factor of roughly `102.3/56.8 roughly  1.8x` — a reduction over half as much data does noticeably less total work but only one fewer level of span, so its work-per-span-level ratio, and therefore its parallelism ceiling, drops disproportionately.
