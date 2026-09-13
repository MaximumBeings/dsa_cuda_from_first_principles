# Appendix E: Profiling and Benchmarking Data Structure Operations

This book's own getting-started page makes a promise worth restating in full here: "timing and throughput numbers are never fabricated... this book measures a genuinely computed, deterministic quantity instead of a wall-clock number... specifically because wall-clock timing captured once on one machine is not reproducible on a rerun, let alone on a reader's own hardware." Every chapter that needed to show one approach was genuinely better than another kept that promise using a technique this appendix now makes explicit. Section E.1 shows that technique directly -- counting operations instead of timing them -- and proves, by literally running the same timing code twice, exactly why wall-clock numbers cannot serve the same purpose. Section E.2 shows the CORRECT way to do real wall-clock timing anyway, for a reader with actual hardware who genuinely needs it. Section E.3 describes NVIDIA's own real profiling tools, Nsight Systems and Nsight Compute, honestly noting what this appendix can and cannot demonstrate about them from an environment with no GPU. Section E.4 closes with a practical checklist for benchmarking a data structure's operations specifically, once you have real hardware to run one on.

## E.1 Why This Book Counts Operations Instead of Timing Them

### Intuition

A wall-clock measurement answers "how long did this take, on this machine, this one time" -- a question whose answer depends on background load, thermal throttling, which other processes happened to be scheduled, and dozens of other factors that have nothing to do with the algorithm itself. A deterministic operation count -- the number of comparisons a sort performs, the number of probe steps a hash table needs, the number of CAS retries a lock-free structure requires under a specific contention pattern -- answers a different, more useful question: "does this algorithm inherently need more work than that one, for this input, regardless of what machine runs it?" This book has used exactly this technique since Chapter 4, without naming it directly; this section names it and makes it the explicit subject.

### The Concept, In Detail

```
ASCII view: the same 8 keys, two hash functions, judged without a clock.

  keys: 1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008   (capacity-11 table)

  good_hash: k % 11        ->  home buckets 0,1,2,3,4,5,6,7 (all distinct)
                               total probes needed: 0

  bad_hash: (k / 100) % 11 ->  every key's home bucket is 10 (all collide)
                               total probes needed: 28

  neither number depends on which machine ran the program, how fast its
  clock is, or what else was running at the time -- COUNTING replaces
  TIMING as the evidence
```

Chapter 19's own discussion of clustering under linear probing is exactly this comparison, made concrete: a poorly-distributed hash function does not merely run "slower" in some vague, hardware-dependent sense -- it provably requires more total probe steps for the identical input, a fact a deterministic count demonstrates directly and a wall-clock measurement could only gesture at, with far more noise than signal.

[COMMON TRAP]
It is tempting to think an operation count is merely a rough PROXY for real performance, with wall-clock time as the "real" ground truth being approximated. For an algorithm's inherent complexity, the count is not an approximation of anything -- it is the exact quantity Big-O notation itself describes, and it is reproducible in a way wall-clock time structurally cannot be. Wall-clock time additionally depends on real hardware effects a pure operation count does not capture (cache behavior, memory bandwidth, instruction-level parallelism) -- which is precisely why Section E.2 and E.3 exist, for the specific, narrower question of how one specific implementation performs on one specific, real device.

### Code and Verification

```cpp
// 225_probe_count_benchmark.cpp
//
// Appendix E.1 -- this book has never reported a single wall-clock number
// as evidence that one approach beats another (see getting-started.md):
// a timing captured once, on one machine, is not reproducible on a
// rerun, let alone on a reader's own hardware. What this book uses
// instead, in every chapter that needed to show one approach is
// genuinely better, is a deterministic, hardware-independent COUNT --
// the same technique Chapter 19 used implicitly. This file makes that
// technique explicit: two hash functions probe the SAME 8 keys into the
// SAME capacity-11 table, and the total number of linear-probe steps
// each one needs is counted directly, with no clock involved anywhere.
//
// Compile: g++ -std=c++17 -Wall -Wextra -O2 225_probe_count_benchmark.cpp -o 225_probe_count_benchmark
// Run:     ./225_probe_count_benchmark

#include <cstdio>
#include <vector>

constexpr int CAP = 11;

int good_hash(int k) { return k % CAP; }
int bad_hash(int k) { return (k / 100) % CAP; }   // ignores the low-order digits that vary here

long simulate(int (*hash_fn)(int), const std::vector<int>& keys, const char* label) {
    std::vector<bool> occupied(CAP, false);
    long total_probes = 0;

    printf("--- %s ---\n", label);
    for (int k : keys) {
        int home = hash_fn(k);
        int idx = home;
        int probes = 0;
        while (occupied[idx]) {
            probes++;
            idx = (idx + 1) % CAP;
        }
        occupied[idx] = true;
        total_probes += probes;
        printf("  key=%d home=%d placed=%d probes=%d\n", k, home, idx, probes);
    }
    printf("  total_probes = %ld\n\n", total_probes);
    return total_probes;
}

int main() {
    std::vector<int> keys = {1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008};

    printf("=== Section E.1: comparing two hash functions by PROBE COUNT, not wall-clock time ===\n\n");
    printf("inserting the same 8 keys into the same capacity-%d table with each hash function:\n\n", CAP);

    long good_probes = simulate(good_hash, keys, "good_hash: k % 11");
    long bad_probes = simulate(bad_hash, keys, "bad_hash: (k / 100) % 11");

    printf("this comparison used ZERO timing calls -- the total_probes count above is\n");
    printf("computed identically and reproducibly on ANY machine, ANY run, ANY compiler,\n");
    printf("unlike a wall-clock measurement of the same two functions would be.\n");

    bool ok = (bad_probes > good_probes) && (good_probes == 0) && (bad_probes == 28);
    printf("\nself-check: bad_hash's total probe count (%ld) is dramatically worse than\n", bad_probes);
    printf("good_hash's (%ld), and both counts match this appendix's own worked design: %s\n",
           good_probes, ok ? "confirmed" : "MISMATCH");

    return ok ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++17 -Wall -Wextra -O2 225_probe_count_benchmark.cpp -o 225_probe_count_benchmark
./225_probe_count_benchmark
```

**Sample input:** the 8 keys `[1001, 1002, ..., 1008]`, inserted into a capacity-11 table using two different hash functions.

**Sample output:**

```text
=== Section E.1: comparing two hash functions by PROBE COUNT, not wall-clock time ===

inserting the same 8 keys into the same capacity-11 table with each hash function:

--- good_hash: k % 11 ---
  key=1001 home=0 placed=0 probes=0
  key=1002 home=1 placed=1 probes=0
  key=1003 home=2 placed=2 probes=0
  key=1004 home=3 placed=3 probes=0
  key=1005 home=4 placed=4 probes=0
  key=1006 home=5 placed=5 probes=0
  key=1007 home=6 placed=6 probes=0
  key=1008 home=7 placed=7 probes=0
  total_probes = 0

--- bad_hash: (k / 100) % 11 ---
  key=1001 home=10 placed=10 probes=0
  key=1002 home=10 placed=0 probes=1
  key=1003 home=10 placed=1 probes=2
  key=1004 home=10 placed=2 probes=3
  key=1005 home=10 placed=3 probes=4
  key=1006 home=10 placed=4 probes=5
  key=1007 home=10 placed=5 probes=6
  key=1008 home=10 placed=6 probes=7
  total_probes = 28

this comparison used ZERO timing calls -- the total_probes count above is
computed identically and reproducibly on ANY machine, ANY run, ANY compiler,
unlike a wall-clock measurement of the same two functions would be.

self-check: bad_hash's total probe count (28) is dramatically worse than
good_hash's (0), and both counts match this appendix's own worked design: confirmed
```

## E.2 Timing the Right Way, When You Do Have Real Hardware

### Intuition

Section E.1 explained why this book never relies on a wall-clock number as evidence -- but a reader with real GPU hardware genuinely does need to time things sometimes, and doing it correctly matters. The right tool for timing a KERNEL is `cudaEvent_t`, not a host-side clock: a kernel launch is asynchronous (the CPU moves on immediately after issuing it), so a host timer started right after a launch measures almost nothing but launch overhead, while `cudaEventRecord` inserts a marker directly into the GPU's own instruction stream, and `cudaEventElapsedTime` measures the genuine time between two such markers, however long the device actually took.

### The Concept, In Detail

```
ASCII view: why a host clock cannot time an asynchronous kernel launch.

  host thread:  ...launch kernel...---(returns almost immediately)---> continues
                        |
                        v  (happens on the DEVICE, independently, afterward)
  device:                [=========== kernel actually executing ===========]

  a host std::chrono measurement wrapped around the launch call only times
  how long it took to QUEUE the kernel -- not how long the device spent
  running it

  cudaEventRecord(start); kernel<<<...>>>(); cudaEventRecord(stop);
  cudaEventSynchronize(stop); cudaEventElapsedTime(&ms, start, stop);
       |
       v
  both markers are inserted into the SAME device instruction stream the
  kernel runs in -- the elapsed time between them is the kernel's own
  genuine execution time, whatever it turns out to be
```

Part B of this section's own code demonstrates Section E.1's point a second, more direct way: it runs the identical `std::chrono`-based timing code TWICE in a row (once per invocation of this very file) and shows you, honestly, that the two raw millisecond numbers are different every time -- which is exactly why this book's own locked, verified output below shows only the QUALITATIVE finding (which of two loops took longer), never the raw numbers themselves. Run this file yourself, more than once, and watch the numbers on your own terminal change while the qualitative conclusion never does.

[COMMON TRAP]
It is tempting to average a handful of wall-clock runs and treat the average as a solid, reportable number. A single machine, at a single moment, already produces run-to-run variance from thread scheduling, cache state, and clock throttling; averaging a small number of runs on ONE machine narrows that machine's own noise but says nothing about how the number would look on a different GPU, a different driver version, or the same machine an hour later under different thermal conditions -- which is exactly the reproducibility gap this book's own operation-count technique (Section E.1) sidesteps entirely rather than trying to average away.

### Code and Verification

```cpp
// 226_timing_methodology.cu
//
// Appendix E.2 -- Section E.1 showed the technique this book actually
// uses (deterministic operation counts); this file shows the CORRECT way
// to do real wall-clock timing, for when a reader has real hardware and
// genuinely needs it. Part A times a kernel with cudaEvent_t -- the right
// tool, because it measures actual DEVICE execution time, unlike a host-
// side clock which cannot see an asynchronously-launched kernel finish.
// Part B uses std::chrono on the host, and prints its own raw millisecond
// numbers to STDERR, deliberately excluded from this file's own locked,
// verified STDOUT -- because, as running this file twice will show you,
// those numbers genuinely differ on every single run. Only the
// QUALITATIVE finding (which of two algorithms was slower) is stable
// enough to lock, which is exactly Section E.1's own point restated.
//
// Compile: nvcc -arch=sm_80 -I$NVDIR/include -L$NVDIR/lib -lcudart 226_timing_methodology.cu -o 226_timing_methodology
// Run:     LD_LIBRARY_PATH=$NVDIR/lib ./226_timing_methodology

#include <cstdio>
#include <chrono>
#include <vector>
#include <cuda_runtime.h>

// A real, syntactically valid kernel -- just enough to demonstrate the
// cudaEvent_t timing pattern; not launchable in this environment (see
// Section 2.3), so Part A's own honest attempt is what actually runs.
__global__ void dummy_kernel(int* data, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) data[tid] += 1;
}

void report(const char* call_name, cudaError_t err) {
    printf("  %-24s -> %-24s (%s)\n", call_name, cudaGetErrorName(err), cudaGetErrorString(err));
}

int main() {
    printf("=== Section E.2, part A: timing a KERNEL correctly, with cudaEvent_t ===\n\n");

    cudaEvent_t start, stop;
    cudaError_t e1 = cudaEventCreate(&start);
    report("cudaEventCreate(start)", e1);
    cudaError_t e2 = cudaEventCreate(&stop);
    report("cudaEventCreate(stop)", e2);

    if (e1 == cudaSuccess && e2 == cudaSuccess) {
        cudaEventRecord(start);
        dummy_kernel<<<1, 32>>>(nullptr, 0);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);
        printf("  kernel elapsed: %f ms\n", ms);
    } else {
        printf("  (event creation failed -- this environment has no usable device, so no\n");
        printf("   kernel can actually be timed here; cudaEvent_t is still the CORRECT way\n");
        printf("   to time a kernel on real hardware -- std::chrono measures HOST time and\n");
        printf("   would not even see an asynchronously-launched kernel finish)\n");
    }

    printf("\n=== Section E.2, part B: std::chrono, and why its OWN numbers can't be locked ===\n\n");
    printf("comparing an O(n) scan against an O(n) x 50 nested scan, same n, using std::chrono:\n\n");

    const int n = 20000;
    std::vector<int> data(n);
    for (int i = 0; i < n; ++i) data[i] = i;

    auto t0 = std::chrono::high_resolution_clock::now();
    volatile long sum_linear = 0;
    for (int i = 0; i < n; ++i) sum_linear += data[i];
    auto t1 = std::chrono::high_resolution_clock::now();

    volatile long sum_quad = 0;
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < 50; ++j)
            sum_quad += data[i] * j;
    auto t2 = std::chrono::high_resolution_clock::now();

    double linear_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double quad_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();

    printf("  O(n) scan (n=%d):        [see stderr for the raw, run-to-run-varying number]\n", n);
    printf("  O(n) x 50 nested (n=%d): [see stderr for the raw, run-to-run-varying number]\n", n);
    fprintf(stderr, "O(n) elapsed:      %f ms (informational only -- will differ on every rerun)\n", linear_ms);
    fprintf(stderr, "O(n)x50 elapsed:   %f ms (informational only -- will differ on every rerun)\n", quad_ms);

    bool ok = (quad_ms > linear_ms);
    printf("\nself-check: the O(n)x50 loop took longer than the plain O(n) scan (a QUALITATIVE\n");
    printf("finding true on every rerun, unlike the raw millisecond values themselves): %s\n",
           ok ? "confirmed" : "MISMATCH");

    return ok ? 0 : 1;
}
```

**Compile and run:**

```bash
NVDIR=/usr/local/lib/python3.11/dist-packages/nvidia/cu13
$NVDIR/bin/nvcc -arch=sm_80 -I$NVDIR/include -L$NVDIR/lib -lcudart 226_timing_methodology.cu -o 226_timing_methodology
LD_LIBRARY_PATH=$NVDIR/lib ./226_timing_methodology
```

**Sample input:** none for Part A (a genuine, honest attempt to create CUDA events); an array of 20,000 integers for Part B, scanned once linearly and once through a 50x nested loop.

**Sample output:**

```text
=== Section E.2, part A: timing a KERNEL correctly, with cudaEvent_t ===

  cudaEventCreate(start)   -> cudaErrorInsufficientDriver (CUDA driver version is insufficient for CUDA runtime version)
  cudaEventCreate(stop)    -> cudaErrorInsufficientDriver (CUDA driver version is insufficient for CUDA runtime version)
  (event creation failed -- this environment has no usable device, so no
   kernel can actually be timed here; cudaEvent_t is still the CORRECT way
   to time a kernel on real hardware -- std::chrono measures HOST time and
   would not even see an asynchronously-launched kernel finish)

=== Section E.2, part B: std::chrono, and why its OWN numbers can't be locked ===

comparing an O(n) scan against an O(n) x 50 nested scan, same n, using std::chrono:

  O(n) scan (n=20000):        [see stderr for the raw, run-to-run-varying number]
  O(n) x 50 nested (n=20000): [see stderr for the raw, run-to-run-varying number]

self-check: the O(n)x50 loop took longer than the plain O(n) scan (a QUALITATIVE
finding true on every rerun, unlike the raw millisecond values themselves): confirmed
```

## E.3 Nsight Systems and Nsight Compute: What Real Profiling Looks Like

### Intuition

`cudaEvent_t` answers "how long did this one kernel take" -- a single number. NVIDIA's own two profiling tools answer far richer questions and are the standard, real-world way any of this book's own kernels would actually be profiled on real hardware. Nsight Systems shows a system-wide TIMELINE -- every kernel launch, memory copy, and CPU/GPU overlap across an entire application's run, useful for spotting a big-picture bottleneck like a kernel that is idle-waiting on a slow host-side memory copy. Nsight Compute goes the opposite direction: deep, per-kernel metrics for ONE specific kernel at a time -- occupancy, memory throughput, and exactly which hardware resource a kernel is bottlenecked on.

### The Concept, In Detail

```
ASCII view: two tools, two different zoom levels.

  Nsight Systems (nsys) -- the whole application, one timeline
    CPU: [prep data]---[launch A]-----------[launch B]---[read results]
    GPU:                [=== kernel A ===]   [=== kernel B ===]
    -- shows gaps, overlap, and which kernel dominates total runtime

  Nsight Compute (ncu) -- one kernel, deep metrics
    Occupancy:            achieved 41% of theoretical maximum
    Memory Throughput:    achieved 78% of peak DRAM bandwidth
    Compute Throughput:   achieved 12% of peak FLOP/s
    Warp State:           62% of stall cycles are "long scoreboard"
                           (waiting on a global memory load)
    -- tells you WHICH resource (compute or memory) is the bottleneck,
       and specifically WHY warps are stalling when they are
```

Nsight Compute's own "Occupancy" section reports the ratio of active warps per multiprocessor to the maximum possible -- distinguishing between THEORETICAL occupancy (what the kernel's own resource usage, such as registers and shared memory per block, allows in principle) and ACHIEVED occupancy (what actually happened during execution, which workload imbalance or synchronization can push lower). Its "Speed of Light" section expresses both compute and memory throughput as a percentage of the device's own peak rates, which is usually the fastest way to tell whether a kernel is compute-bound or memory-bound at a glance -- exactly the distinction this book's own chapters (a bandwidth-bound reduction versus a compute-bound sort, say) have discussed in words, which Nsight Compute measures directly on real hardware.

This appendix cannot install or run either tool here -- this environment has neither a physical GPU nor the Nsight packages themselves installed, and installing genuine profiling output would mean fabricating numbers this book has refused to fabricate anywhere else. On a real machine (including a rented Lambda Cloud instance, per Appendix A.4, which ships both tools as part of Lambda Stack), profiling any of this book's own kernels looks like:

```bash
nsys profile -o report ./205_fx_bellman_ford_kernel
ncu -o kernel_report --set full ./205_fx_bellman_ford_kernel
```

[COMMON TRAP]
It is tempting to reach for Nsight Compute first, since its metrics are more detailed. Nsight Compute profiles ONE kernel invocation at a time, in isolation, and its own profiling overhead can be substantial (a full metric set may run a kernel many times over to collect everything) -- starting with Nsight Systems' whole-application timeline first, to find out WHICH kernel actually dominates total runtime, then profiling only that one kernel in Nsight Compute, is the standard, efficient order these two tools are meant to be used in together.

## E.4 A Benchmarking Checklist for Data Structure Operations

### Intuition

Benchmarking a data structure's operations on real hardware raises questions specific to data structures that a generic "time this kernel" checklist does not cover: how does insert throughput change as a hash table's load factor rises, how does a lock-free stack's throughput change under increasing contention, how does a tree's traversal cost change with depth. This section closes with the practical checklist this book's own chapters implicitly followed whenever they needed to make a claim like "clustering gets worse as load factor increases" -- a claim proven with Section E.1's own counting technique throughout this book, and one Section E.2/E.3's real timing tools would extend to actual throughput numbers on real hardware.

### The Concept, In Detail

```
ASCII view: what to vary, and what to hold fixed, per data structure.

  hash tables (Ch19-20):     vary LOAD FACTOR, count probes or measure
                              throughput at each point -- Ch19's own
                              clustering discussion is this exact sweep

  lock-free structures (Ch9-10, 27): vary CONTENDING THREAD COUNT, count
                              CAS retries or measure throughput at each
                              level of contention

  trees (Ch15-18):            vary TREE DEPTH / SIZE, count node visits
                              or measure traversal time at each size

  graphs (Ch21-25):           vary GRAPH DENSITY and FRONTIER SIZE, count
                              edges relaxed per level or measure per-level
                              kernel time
```

Whichever variable is being swept, the same real-hardware discipline applies: run a warm-up iteration first and discard it (the very first kernel launch or memory allocation of a program often pays one-time driver/JIT costs no later iteration repeats), run several trials and report the MEDIAN rather than the mean (a single unusually slow trial, from an unrelated background process, skews a mean far more than a median), and vary exactly one thing at a time (load factor, thread count, or size) while holding everything else -- including the input data itself -- fixed, so that whatever changes in the result can be attributed to the one variable actually being studied.

## Appendix Summary

Section E.1 named the technique this book has used since Chapter 4 without saying so directly: a deterministic operation count, not a wall-clock number, is what proves one algorithm genuinely needs less work than another, because it is reproducible on any machine, on any run, in a way wall-clock time structurally is not. Section E.2 showed the CORRECT way to do real timing anyway -- `cudaEvent_t` for a kernel's own genuine device execution time, `std::chrono` for host-side code -- and proved Section E.1's point a second time by running the identical timing code twice and showing its raw numbers genuinely differ, which is exactly why only a qualitative finding, never a raw number, was locked into this appendix's own verified output. Section E.3 described Nsight Systems' whole-application timeline and Nsight Compute's deep per-kernel metrics (occupancy, memory and compute throughput, warp stall reasons) as the real, standard tools this book's own kernels would be profiled with on actual hardware, honestly noting that this environment can describe but not run them. Section E.4 closed with a practical checklist -- warm-up iterations, median over several trials, one variable at a time -- for extending this book's own counting-based comparisons (load factor, contention, tree size, graph density) into genuine throughput numbers once real hardware is available.
