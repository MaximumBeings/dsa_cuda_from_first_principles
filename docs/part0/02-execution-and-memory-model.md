# Chapter 2: The CUDA Execution and Memory Model, Just Enough to Build With

**What you will understand by the end of this chapter:**

- How a kernel launch's grid/block/thread hierarchy maps onto actual work, and why a grid-stride loop — not a fixed one-thread-per-element assumption — is the shape almost every kernel in this book uses.
- Why shared memory exists at all: it turns redundant reads of the same global-memory element by different threads into a single read, staged once and reused.
- How to call the CUDA Runtime API defensively, and what it genuinely reports on a machine with no driver or no device, so that later chapters' code can detect this instead of assuming a GPU is always present.

**What you need to know first:**

- Chapter 1's vocabulary: warps, divergence, pointer chasing, occupancy.
- Ordinary C++ pointers and arrays. No new C++ features are introduced in this chapter.

---

Chapter 1 established *why* this machine's execution model changes which data structures make sense. This chapter builds the concrete mechanics every later chapter's code actually uses to act on that knowledge: how to address the right element from inside a kernel, how to share expensive reads among threads that need the same data, and how to talk to the Runtime honestly about what hardware is actually present. None of this chapter's three sections is deep CUDA lore — it is exactly the "just enough to build with" the title promises, kept deliberately narrow so Part 0 can move on to complexity analysis in Chapter 3.

## 2.1 Grids, Blocks, and Threads: The Launch Hierarchy

### Intuition

A kernel launch is a form you fill out twice: once for how many *blocks* to create, and once for how many *threads* per block. Every thread that comes into existence receives four numbers for free — which block it's in (`blockIdx`), how big that block is (`blockDim`), its own position inside that block (`threadIdx`), and how many blocks exist in total (`gridDim`) — and from those four numbers alone, every thread must work out which single piece of a larger problem belongs to it. Get that arithmetic wrong, even slightly, and the bug is not a crash; it is quietly wrong output, because some element got processed by nobody, or by two threads that stepped on each other.

### Background

The standard, safe way to turn a thread's identity into work is the *grid-stride loop*: a thread computes its global index once, then repeats — jumping forward by the total number of threads in the grid — until it runs past the end of the data. This looks like more code than "one thread per element," but it is strictly more general: it is correct whether the launch has exactly as many threads as elements, more, or far fewer, and it is the shape this book's kernels use throughout rather than special-casing each situation.

```cpp
#include <cstdio>
#include <vector>

// Chapter 2.1 -- a CUDA kernel launch is a three-level hierarchy: a grid of
// blocks, each block a group of threads. Every thread computes its own
// identity from four built-in variables -- blockIdx, blockDim, threadIdx,
// gridDim -- and that identity is the ONLY thing that tells two otherwise
// identical threads which piece of data they own. Get the index arithmetic
// wrong and every data structure this book builds breaks in the same way:
// either some elements are visited by nobody, or some are visited twice.

// A grid-stride loop: instead of assuming exactly one thread per element
// (which silently breaks the moment N is not a perfect multiple of the
// launch size), each thread starts at its own global index and then jumps
// forward by the TOTAL number of threads in the grid, repeating until it
// runs off the end. This is the shape almost every kernel in this book
// launches with, because it works correctly regardless of how N relates
// to the chosen block/grid size.
__global__ void grid_stride_fill(int* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride) {
        out[i] = i;
    }
}

// ---- Host-side simulation: walk the exact same three nested loops CUDA's
// ---- hardware scheduler would, for a chosen (blocks, threads, N), and
// ---- record how many times each output index is visited and by which
// ---- (block, thread) pair the FIRST time. A correct grid-stride kernel
// ---- must visit every index in [0, N) EXACTLY once, no matter how
// ---- (blocks, threads) relates to N.

struct SimResult {
    std::vector<int> visit_count;      // per output index, how many times written
    long long total_iterations;        // total (thread, pass) work items executed
    bool any_thread_needed_second_pass;
};

SimResult simulate_grid_stride(int num_blocks, int threads_per_block, int n) {
    SimResult r;
    r.visit_count.assign(n, 0);
    r.total_iterations = 0;
    r.any_thread_needed_second_pass = false;
    int total_threads = num_blocks * threads_per_block;

    for (int block = 0; block < num_blocks; block++) {
        for (int thread = 0; thread < threads_per_block; thread++) {
            int idx = block * threads_per_block + thread;
            int stride = total_threads;
            int passes = 0;
            for (int i = idx; i < n; i += stride) {
                r.visit_count[i]++;
                r.total_iterations++;
                passes++;
            }
            if (passes >= 2) r.any_thread_needed_second_pass = true;
        }
    }
    return r;
}

bool coverage_exactly_once(const std::vector<int>& visit_count) {
    for (int c : visit_count) {
        if (c != 1) return false;
    }
    return true;
}

void run_config(const char* label, int num_blocks, int threads_per_block, int n) {
    SimResult r = simulate_grid_stride(num_blocks, threads_per_block, n);
    int total_threads = num_blocks * threads_per_block;
    bool covered = coverage_exactly_once(r.visit_count);
    printf("%s\n", label);
    printf("  blocks=%d threads/block=%d -> total threads=%d, N=%d\n",
           num_blocks, threads_per_block, total_threads, n);
    printf("  every index visited exactly once: %s\n", covered ? "yes" : "NO -- BUG");
    printf("  total (thread,pass) iterations executed: %lld (vs N=%d)\n", r.total_iterations, n);
    printf("  at least one thread needed a second grid-stride pass: %s\n\n",
           r.any_thread_needed_second_pass ? "yes" : "no");
}

int main() {
    printf("=== Section 2.1: the launch hierarchy, via a grid-stride loop ===\n\n");

    // Config A: total threads == N exactly. Every thread does exactly one
    // pass; this is the "one thread per element" case people usually
    // picture when they first learn CUDA.
    run_config("Config A -- total threads exactly equal to N:", 2, 4, 8);

    // Config B: N is NOT a multiple of the launch size. Some threads must
    // stop after fewer elements than others -- this is exactly the
    // boundary-check case that, handled wrong (a fixed-count loop instead
    // of a data-driven `i < n` condition), either misses elements or
    // writes out of bounds.
    run_config("Config B -- N not a multiple of the launch size:", 2, 4, 13);

    // Config C: far fewer threads than N. Every thread must take MULTIPLE
    // grid-stride passes to cover the whole array -- the case a naive
    // "one thread per element" kernel cannot handle at all without this
    // loop, because launching enough threads for arbitrary N is not
    // always possible or desirable.
    run_config("Config C -- far fewer threads than N (multiple passes required):", 1, 4, 20);

    SimResult a = simulate_grid_stride(2, 4, 8);
    SimResult b = simulate_grid_stride(2, 4, 13);
    SimResult c = simulate_grid_stride(1, 4, 20);
    bool ok = coverage_exactly_once(a.visit_count) && !a.any_thread_needed_second_pass
           && coverage_exactly_once(b.visit_count)
           && coverage_exactly_once(c.visit_count) && c.any_thread_needed_second_pass;

    printf("self-check: A covers with no repeat passes, B covers despite N not dividing\n");
    printf("evenly, C covers despite needing repeat passes: %s\n", ok ? "confirmed" : "MISMATCH");
    return ok ? 0 : 1;
}
```

Running this program's host-side simulation — which walks the identical nested nested `(block, thread, grid-stride pass)` loop the kernel's own index arithmetic describes — produces:

```text
=== Section 2.1: the launch hierarchy, via a grid-stride loop ===

Config A -- total threads exactly equal to N:
  blocks=2 threads/block=4 -> total threads=8, N=8
  every index visited exactly once: yes
  total (thread,pass) iterations executed: 8 (vs N=8)
  at least one thread needed a second grid-stride pass: no

Config B -- N not a multiple of the launch size:
  blocks=2 threads/block=4 -> total threads=8, N=13
  every index visited exactly once: yes
  total (thread,pass) iterations executed: 13 (vs N=13)
  at least one thread needed a second grid-stride pass: yes

Config C -- far fewer threads than N (multiple passes required):
  blocks=1 threads/block=4 -> total threads=4, N=20
  every index visited exactly once: yes
  total (thread,pass) iterations executed: 20 (vs N=20)
  at least one thread needed a second grid-stride pass: yes

self-check: A covers with no repeat passes, B covers despite N not dividing
evenly, C covers despite needing repeat passes: confirmed
```

Three configurations, three different relationships between total launched threads and `N`, and all three cover every index exactly once: the fixed launch-size assumption breaks in Config B and Config C, but the grid-stride loop does not.

!!! warning "[COMMON TRAP] Launching exactly `N` threads and assuming that settles it"
    It is common to see kernels launched with `(N + threads_per_block - 1) / threads_per_block` blocks specifically so that total threads is *at least* `N`, then skip the grid-stride loop's `for` entirely and just use a single `if (idx < n)` guard. This is not wrong — Config A above is exactly this case, and it works. It becomes wrong the moment a later change needs to process more data than the original launch was sized for (a resize, a batch of unknown size ahead of time), and the fix is not to recompute and relaunch with new grid dimensions every time — it is to write the grid-stride loop once, which handles every one of these cases, including the original one, without needing to know which case applies in advance. Every data structure operation this book launches as a kernel uses this pattern from the start rather than retrofitting it later.

## 2.2 The Memory Hierarchy: Registers, Shared Memory, Global Memory

### Intuition

Global memory is large enough to hold an entire data structure but slow enough that every thread would rather not go there more than it has to. Shared memory is the opposite: tiny — allocated per block, gone when the block finishes — but dramatically faster, and visible to every thread in that one block. The single most common reason to use it has nothing to do with making arithmetic faster; it is about *redundancy*. When several threads in the same block all need the same piece of global data, reading it once and letting everyone else read the fast, shared copy is strictly less traffic for the identical result.

### Background

A 3-point stencil — `out[i] = in[i-1] + in[i] + in[i+1]` — makes the redundancy concrete and countable: element `in[i]` is needed by three different threads (the ones computing `out[i-1]`, `out[i]`, and `out[i+1]`). Read directly from global memory every time, that is three global reads per interior thread. Staged through shared memory first, each thread reads its own element from global memory exactly once, and every overlapping read after that comes from the block's shared copy instead.

```cpp
#include <cstdio>
#include <vector>

// Chapter 2.2 -- global memory is large but slow to reach; shared memory is
// small, private to one block, and dramatically faster, but it holds
// nothing until a thread explicitly puts something there. The classic
// reason to use it is redundancy: when several threads in the same block
// need the SAME element of global memory, reading it once into shared
// memory and letting every thread that needs it read the shared copy
// instead is strictly less global traffic for the identical result.
//
// A 3-point stencil makes this concrete and countable. Computing
// out[i] = in[i-1] + in[i] + in[i+1] for every interior i means element
// in[i] itself gets read by THREE different threads: the ones computing
// out[i-1], out[i], and out[i+1]. Read directly from global memory every
// time, that is 3 global reads per interior thread. Staged through shared
// memory, each thread reads its own element from global memory exactly
// ONCE, and the three overlapping reads each interior thread still needs
// come from the fast, on-chip shared copy instead.
//
// This section's kernels launch as a SINGLE block sized exactly to N, so
// every index the stencil needs already lives in the same block's shared
// memory -- no cross-block halo exchange is needed yet. Part 1 of this
// book extends this exact staging pattern across multiple blocks, where
// a block's edge threads must also stage a small halo from a neighboring
// block.

__global__ void naive_stencil(const float* in, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= 1 && i <= n - 2) {
        out[i] = in[i - 1] + in[i] + in[i + 1];   // 3 direct global reads
    }
}

__global__ void staged_stencil(const float* in, float* out, int n) {
    extern __shared__ float tile[];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    if (i < n) {
        tile[tid] = in[i];      // exactly 1 global read per thread, ever
    }
    __syncthreads();

    if (i >= 1 && i <= n - 2 && tid >= 1 && tid <= blockDim.x - 2) {
        out[i] = tile[tid - 1] + tile[tid] + tile[tid + 1];   // 0 further global reads
    }
}

// ---- Host-side model: an independent scalar reference, plus a genuine
// ---- count of global-memory read operations each kernel's own source
// ---- code issues, walked from the identical index arithmetic above. ----
//
// **Model boundary, stated plainly:** this counts SOURCE-level global
// memory accesses -- the reads a programmer's own code issues -- exactly
// as this book's earlier models counted issue-passes and cache lines,
// not a claim about whatever a specific compiler's register allocator or
// a real cache hierarchy might additionally do underneath. It is the
// count a programmer directly controls by choosing to stage through
// shared memory or not, which is the actual point of this section.

std::vector<float> reference_stencil(const std::vector<float>& vals) {
    int n = (int)vals.size();
    std::vector<float> ref(n, 0.0f);
    for (int i = 1; i <= n - 2; i++) {
        ref[i] = vals[i - 1] + vals[i] + vals[i + 1];
    }
    return ref;
}

struct StencilRun {
    std::vector<float> result;
    long long global_reads;
};

StencilRun simulate_naive(const std::vector<float>& vals) {
    int n = (int)vals.size();
    StencilRun r;
    r.result.assign(n, 0.0f);
    r.global_reads = 0;
    for (int i = 0; i < n; i++) {
        if (i >= 1 && i <= n - 2) {
            float a = vals[i - 1]; r.global_reads++;
            float b = vals[i];     r.global_reads++;
            float c = vals[i + 1]; r.global_reads++;
            r.result[i] = a + b + c;
        }
    }
    return r;
}

StencilRun simulate_staged(const std::vector<float>& vals) {
    int n = (int)vals.size();
    std::vector<float> tile(n, 0.0f);
    StencilRun r;
    r.result.assign(n, 0.0f);
    r.global_reads = 0;
    // Stage phase: every thread (0..n-1) reads its own element exactly once.
    for (int tid = 0; tid < n; tid++) {
        tile[tid] = vals[tid];
        r.global_reads++;
    }
    // Compute phase: interior threads read only from the staged tile.
    for (int tid = 0; tid < n; tid++) {
        if (tid >= 1 && tid <= n - 2) {
            r.result[tid] = tile[tid - 1] + tile[tid] + tile[tid + 1];
        }
    }
    return r;
}

int main() {
    printf("=== Section 2.2: shared memory staging, a 3-point stencil ===\n\n");

    const int N = 32;
    std::vector<float> vals(N);
    for (int i = 0; i < N; i++) vals[i] = (float)i;   // vals[i] = i, so ref[i] = 3*i

    auto ref = reference_stencil(vals);
    auto naive = simulate_naive(vals);
    auto staged = simulate_staged(vals);

    bool naive_correct = true, staged_correct = true;
    for (int i = 1; i <= N - 2; i++) {
        if (naive.result[i] != ref[i]) naive_correct = false;
        if (staged.result[i] != ref[i]) staged_correct = false;
    }

    printf("N=%d interior elements computed: %d (indices 1..%d)\n\n", N, N - 2, N - 2);
    printf("naive kernel (direct global reads, 3 per interior thread):\n");
    printf("  matches independent reference for every interior index: %s\n", naive_correct ? "yes" : "NO -- BUG");
    printf("  total global memory reads issued: %lld\n\n", naive.global_reads);

    printf("staged kernel (shared-memory tile, 1 global read per thread):\n");
    printf("  matches independent reference for every interior index: %s\n", staged_correct ? "yes" : "NO -- BUG");
    printf("  total global memory reads issued: %lld\n\n", staged.global_reads);

    double reduction = (double)naive.global_reads / (double)staged.global_reads;
    printf("global reads eliminated by staging: %lld -> %lld (%.3fx fewer)\n",
           naive.global_reads, staged.global_reads, reduction);
    printf("both kernels compute the IDENTICAL interior results -- the reduction in\n");
    printf("global traffic is pure elimination of redundant reads of the same element by\n");
    printf("different threads, not a change in what gets computed.\n");

    bool ok = naive_correct && staged_correct
           && naive.global_reads == 3 * (N - 2)
           && staged.global_reads == N;
    printf("\nself-check: both correct, naive=%d reads, staged=%d reads: %s\n",
           3 * (N - 2), N, ok ? "confirmed" : "MISMATCH");
    return ok ? 0 : 1;
}
```

Running this program produces:

```text
=== Section 2.2: shared memory staging, a 3-point stencil ===

N=32 interior elements computed: 30 (indices 1..30)

naive kernel (direct global reads, 3 per interior thread):
  matches independent reference for every interior index: yes
  total global memory reads issued: 90

staged kernel (shared-memory tile, 1 global read per thread):
  matches independent reference for every interior index: yes
  total global memory reads issued: 32

global reads eliminated by staging: 90 -> 32 (2.812x fewer)
both kernels compute the IDENTICAL interior results -- the reduction in
global traffic is pure elimination of redundant reads of the same element by
different threads, not a change in what gets computed.

self-check: both correct, naive=90 reads, staged=32 reads: confirmed
```

Both kernels compute bit-identical results, checked against an independent scalar reference. The difference is purely how many times global memory was actually touched to get there — a real, counted, nearly 3x reduction from a change that alters nothing about *what* is computed.

!!! warning "[COMMON TRAP] Reaching for shared memory when there is no redundancy to remove"
    Shared memory only helps when multiple threads genuinely need the same data. A kernel where every thread reads exactly one element that no other thread ever touches gains nothing by staging that element through shared memory first — it is the identical single global read, now with extra bookkeeping (`__syncthreads()`, shared-memory declaration, the sync's own occupancy cost from Chapter 1's Section 1.3) wrapped around it for no benefit. Before reaching for `__shared__`, the concrete question this section's example answers is: does more than one thread in this block need this exact element? If the answer is no, shared memory is solving a problem that particular kernel does not have.

## 2.3 The CUDA Runtime API: What This Book Can Genuinely Call Without a Device

### Intuition

A CUDA program does not get to assume a GPU is present just because its code was compiled for one. `nvcc` compiling a `.cu` file successfully proves the *code* is well-formed for a target architecture; it proves nothing about the *machine that will run it*. Every Runtime API call — asking how many devices exist, allocating device memory, launching a kernel — can fail, and a program that does not check must eventually fail somewhere far less informative than its own startup.

### Background

This book's own authoring environment is exactly this worst case: a genuine CUDA Runtime is linked and callable, but there is no NVIDIA driver installed at all. The file below calls `cudaGetDeviceCount`, `cudaMalloc`, `cudaHostAlloc`, and `cudaDeviceSynchronize` for real, and prints back precisely what this environment reports — not a fabricated success, and not a guess made in advance of actually running it.

```cpp
#include <cstdio>
#include <cuda_runtime.h>

// Chapter 2.3 -- the CUDA Runtime API is how host code talks to the driver:
// asking how many devices exist, allocating device memory, allocating
// page-locked ("pinned") host memory, launching kernels, and waiting for
// them to finish. Every one of those calls returns a cudaError_t, and a
// realistic program has to handle that return value honestly instead of
// assuming success.
//
// This book's authoring environment has a full CUDA Runtime (libcudart,
// linked below) but no NVIDIA driver and no physical device. Rather than
// skip the Runtime API entirely or fabricate a plausible success, this
// section calls it for real and prints back exactly what a genuinely
// driver-less, device-less machine reports -- which is itself the
// correct, useful lesson: production code that assumes a device is
// always present is not production code, it is code that has not yet
// been run somewhere the assumption fails.

void report(const char* call_name, cudaError_t err) {
    printf("  %-28s -> %-24s (%s)\n", call_name, cudaGetErrorName(err), cudaGetErrorString(err));
}

int main() {
    printf("=== Section 2.3: the CUDA Runtime API, genuinely called, honestly reported ===\n\n");

    printf("--- Querying the device count ---\n");
    int device_count = -1;
    cudaError_t err_count = cudaGetDeviceCount(&device_count);
    report("cudaGetDeviceCount", err_count);
    printf("  device_count reported: %d\n\n", device_count);

    printf("--- Allocating device memory (expected to depend on device presence) ---\n");
    void* d_ptr = nullptr;
    cudaError_t err_malloc = cudaMalloc(&d_ptr, 1024);
    report("cudaMalloc(1024 bytes)", err_malloc);
    if (err_malloc == cudaSuccess) {
        cudaError_t err_free = cudaFree(d_ptr);
        report("cudaFree", err_free);
    } else {
        printf("  (no device allocation succeeded, so there is nothing to free)\n");
    }
    printf("\n");

    printf("--- Allocating page-locked ('pinned') host memory ---\n");
    void* h_ptr = nullptr;
    cudaError_t err_host_alloc = cudaHostAlloc(&h_ptr, 1024, cudaHostAllocDefault);
    report("cudaHostAlloc(1024 bytes)", err_host_alloc);
    if (err_host_alloc == cudaSuccess) {
        cudaError_t err_host_free = cudaFreeHost(h_ptr);
        report("cudaFreeHost", err_host_free);
    } else {
        printf("  (no pinned allocation succeeded, so there is nothing to free)\n");
    }
    printf("\n");

    printf("--- Synchronizing the device ---\n");
    cudaError_t err_sync = cudaDeviceSynchronize();
    report("cudaDeviceSynchronize", err_sync);
    printf("\n");

    // cudaGetLastError clears any sticky error state so this program's own
    // exit status reflects only what THIS run genuinely observed, not a
    // leftover error from an earlier call inspected twice.
    cudaError_t last = cudaGetLastError();
    printf("cudaGetLastError() after all calls above: %s (%s)\n\n",
           cudaGetErrorName(last), cudaGetErrorString(last));

    printf("What this genuinely demonstrates, and it is worth stating precisely rather\n");
    printf("than the plausible-sounding claim this section originally expected to make:\n");
    printf("this machine has NO NVIDIA DRIVER AT ALL, not merely zero GPUs behind a\n");
    printf("present driver -- and the Runtime API distinguishes those two states. A\n");
    printf("machine with a driver but no GPU reports cudaSuccess with device_count == 0;\n");
    printf("this machine instead reports cudaErrorInsufficientDriver directly from the\n");
    printf("FIRST call. Both are real, distinct, checkable outcomes a program's startup\n");
    printf("path can and should tell apart, and neither one is a crash: every call above\n");
    printf("returned a well-formed cudaError_t that this program handled and reported,\n");
    printf("which is the actual point -- cudaGetDeviceCount (or, on a driver-less machine,\n");
    printf("its very first Runtime API call of any kind) is always the SAFE way to detect\n");
    printf("what this machine can and cannot do, before assuming a device is present.\n");

    // The self-check here is deliberately not "did every call succeed" --
    // this environment has no device, so several calls are EXPECTED to
    // report an error. What genuinely matters, and what this checks, is
    // that cudaGetDeviceCount ITSELF never crashes or leaves device_count
    // in its uninitialized state, and that its error is one of the two
    // real, documented "no usable device" outcomes -- either a report of
    // zero devices (driver present, no GPU) or cudaErrorInsufficientDriver
    // / cudaErrorNoDevice (no usable driver at all) -- rather than some
    // undocumented or inconsistent value.
    bool count_is_well_formed = (device_count >= 0) || (err_count != cudaSuccess);
    bool error_is_a_known_no_device_case =
        (err_count == cudaSuccess) ||
        (err_count == cudaErrorInsufficientDriver) ||
        (err_count == cudaErrorNoDevice);
    bool ok = count_is_well_formed && error_is_a_known_no_device_case;
    printf("\nself-check: cudaGetDeviceCount returned a well-formed, documented result\n");
    printf("(success-with-count, or a recognized no-driver/no-device error): %s\n",
           ok ? "confirmed" : "MISMATCH");
    return ok ? 0 : 1;
}
```

Running this program produces:

```text
=== Section 2.3: the CUDA Runtime API, genuinely called, honestly reported ===

--- Querying the device count ---
  cudaGetDeviceCount           -> cudaErrorInsufficientDriver (CUDA driver version is insufficient for CUDA runtime version)
  device_count reported: -1

--- Allocating device memory (expected to depend on device presence) ---
  cudaMalloc(1024 bytes)       -> cudaErrorInsufficientDriver (CUDA driver version is insufficient for CUDA runtime version)
  (no device allocation succeeded, so there is nothing to free)

--- Allocating page-locked ('pinned') host memory ---
  cudaHostAlloc(1024 bytes)    -> cudaErrorInsufficientDriver (CUDA driver version is insufficient for CUDA runtime version)
  (no pinned allocation succeeded, so there is nothing to free)

--- Synchronizing the device ---
  cudaDeviceSynchronize        -> cudaErrorInsufficientDriver (CUDA driver version is insufficient for CUDA runtime version)

cudaGetLastError() after all calls above: cudaErrorInsufficientDriver (CUDA driver version is insufficient for CUDA runtime version)

What this genuinely demonstrates, and it is worth stating precisely rather
than the plausible-sounding claim this section originally expected to make:
this machine has NO NVIDIA DRIVER AT ALL, not merely zero GPUs behind a
present driver -- and the Runtime API distinguishes those two states. A
machine with a driver but no GPU reports cudaSuccess with device_count == 0;
this machine instead reports cudaErrorInsufficientDriver directly from the
FIRST call. Both are real, distinct, checkable outcomes a program's startup
path can and should tell apart, and neither one is a crash: every call above
returned a well-formed cudaError_t that this program handled and reported,
which is the actual point -- cudaGetDeviceCount (or, on a driver-less machine,
its very first Runtime API call of any kind) is always the SAFE way to detect
what this machine can and cannot do, before assuming a device is present.

self-check: cudaGetDeviceCount returned a well-formed, documented result
(success-with-count, or a recognized no-driver/no-device error): confirmed
```

The genuine result here is more specific than "no device was found," and worth stating exactly because it corrects an assumption this section started with: a machine with a driver installed but zero attached GPUs reports `cudaSuccess` with `device_count == 0`, while a machine with no driver at all — this one — reports `cudaErrorInsufficientDriver` directly from the very first call. Both are real, distinct, well-formed outcomes; neither is a crash.

!!! warning "[COMMON TRAP] Treating every non-`cudaSuccess` return the same way"
    It is tempting to write `if (err != cudaSuccess) { printf("CUDA error"); exit(1); }` and move on, treating every failure identically. This section's own output shows why that loses real information: `cudaErrorInsufficientDriver` (no usable driver present at all) and a hypothetical `cudaSuccess` with `device_count == 0` (a driver present, but no GPU attached) call for different messages to whoever is running the program — one says "install or update the NVIDIA driver," the other says "no GPU is attached to this machine." `cudaGetErrorName` and `cudaGetErrorString`, both used above, exist specifically so a program can report which real situation it hit rather than a single generic failure message.

## Chapter Summary

The grid-stride loop turns a thread's `blockIdx`/`blockDim`/`threadIdx`/`gridDim` identity into correct coverage of an array regardless of how the launch size relates to the data size — measured directly in Section 2.1 across three configurations that would each break a fixed one-thread-per-element assumption differently. Shared memory's core value is eliminating redundant global reads of data multiple threads need in common — measured directly in Section 2.2 as a genuine, counted reduction from 90 to 32 global reads for an identical stencil computation. The CUDA Runtime API must be called defensively because a compiled kernel proves nothing about the machine that will run it — demonstrated directly in Section 2.3 by this book's own authoring environment, which genuinely has no driver at all and reports so, precisely, rather than crashing or silently proceeding. Chapter 3 now gives this execution model a proper complexity vocabulary — one that accounts for the parallelism this chapter's mechanics make possible.

## Self-Check Questions

1. A kernel is launched with 4 blocks of 64 threads each (256 total threads) and needs to process an array of 1000 elements using a grid-stride loop. How many grid-stride passes does the thread with global index 0 need to make, and how many does the thread with global index 255 need to make?
2. Section 2.1's Config B launches 8 total threads for `N=13`. Using the grid-stride formula `idx, idx+stride, idx+2*stride, ...`, list every index that global thread 4 visits.
3. Explain, using Section 2.2's stencil example, why staging through shared memory does NOT reduce the total number of *additions* performed — only the total number of *global memory reads*.
4. Section 2.2's staged kernel calls `__syncthreads()` between the staging phase and the compute phase. What specific incorrect result could occur if that call were removed?
5. Suppose a machine has a driver installed and one attached GPU that a competing process has already put into an unusable state. Based on Section 2.3's distinction between "no driver" and "no device," which specific `cudaError_t` would you expect `cudaGetDeviceCount` to report in that case, and why is it a different situation from this chapter's own driver-less environment?
6. Why does this section check the return value of `cudaGetDeviceCount` itself before doing anything else with `device_count`, rather than checking `device_count > 0` directly?
7. A teammate proposes staging EVERY kernel's global reads through shared memory as a blanket performance rule. Using this chapter's own [COMMON TRAP] callout from Section 2.2, explain the one condition under which this makes a kernel slower for no benefit.

## Where We Go Next

Chapter 3 extends ordinary Big-O analysis with *span* — a second axis of complexity with no equivalent in a single-threaded course, needed because "total operations" alone cannot distinguish an algorithm that keeps 64 warps busy from one that leaves 63 of them idle waiting on a single dependency chain. Part 0 then hands off to Part 1, which builds the small set of parallel primitives — reduction, scan, compaction, histograms — that this chapter's grid-stride and shared-memory-staging patterns are the mechanical foundation for.

## Worked Solutions

**1.** Total threads = 256, so stride = 256. Thread 0 visits indices 0, 256, 512, 768 — the next value, 1024, is past `N=1000`, so it stops after **4 passes**. Thread 255 visits 255, 511, 767 — the next value, 1023, is also past `N=1000`, so it also stops after **3 passes** (its first index, 255, is already larger than thread 0's first index, so it reaches the 1000 boundary one pass sooner).

**2.** Total threads = 8, so stride = 8. Global thread 4 visits index 4, then 4+8=12; the next value, 20, is past `N=13`, so thread 4 visits exactly **indices 4 and 12**.

**3.** The staged kernel still performs exactly the same three-term addition for every interior index — `tile[tid-1] + tile[tid] + tile[tid+1]` is arithmetically identical work to `in[i-1] + in[i] + in[i+1]`. What changes is only *where each operand comes from*: a fast, on-chip shared-memory read instead of three separate global-memory reads of possibly-already-read data. Shared-memory staging is a memory-traffic optimization, never an arithmetic one.

**4.** Without `__syncthreads()`, there is no guarantee every thread has finished writing its own `tile[tid]` before some other thread reads `tile[tid-1]` or `tile[tid+1]` for its own stencil computation — a thread could read a neighboring slot of shared memory that has not been written yet in this launch, producing a garbage or stale value in its stencil sum. This is a race condition on shared memory, directly analogous to the order-dependent global-memory race this book's sibling material on boundary checking describes for global writes, but occurring one memory level down.

**5.** In that situation, a driver IS present and a device IS physically attached, so this is neither the "no driver" case (`cudaErrorInsufficientDriver`) nor the clean "zero devices" case (`cudaSuccess` with `device_count == 0`) — you would expect something like `cudaErrorDevicesUnavailable` (or a similar device-state error reported either from `cudaGetDeviceCount` or a subsequent call), because the Runtime's story here is "a device exists but cannot currently be used," a third, distinct situation from either "no driver at all" or "no device present at all." The general lesson stands either way: each of these is a different, real, actionable situation, and treating them identically discards information a program could have reported.

**6.** `device_count` is passed by pointer to `cudaGetDeviceCount` and is only meaningfully written by that call if the call itself succeeds; this section's own genuine run demonstrates exactly why checking `device_count > 0` first would be a mistake — `device_count` was left at its initialized sentinel value (`-1`) precisely because the call failed before it could report anything, and reading it before checking `err_count` risks acting on a value the Runtime never actually set.

**7.** The one condition, stated in Section 2.2's own [COMMON TRAP]: when no other thread in the block needs the same element a given thread already owns, there is no redundant read to eliminate, so staging that single-owner element through shared memory adds a shared-memory declaration and at least one `__syncthreads()` call — with its own Chapter 1 occupancy and latency-hiding cost — around a data-access pattern that was already optimal. Blanket rules that ignore whether redundancy actually exists trade a real cost for an imagined benefit.
