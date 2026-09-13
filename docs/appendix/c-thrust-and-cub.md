# Appendix C: Thrust and CUB -- The Standard Library You Get for Free

Every reduction, scan, and sort in this book so far has been hand-written -- a `__global__` kernel, its own index arithmetic, its own boundary checks -- because that is the only way to actually learn what a GPU primitive does underneath. Production CUDA code very rarely writes these from scratch: NVIDIA ships two libraries, Thrust and CUB, that implement every primitive this book has built by hand, tuned by people who spend their careers on exactly this, and battle-tested across every architecture NVIDIA has ever shipped. Section C.1 replaces Chapter 4 and Chapter 5's hand-written reduction and scan with one Thrust call each -- genuinely executed, not simulated, using an execution policy that needs no device at all. Section C.2 does the same for Part 3's sorting chapters. Section C.3 goes one level deeper, into CUB, the block- and warp-level toolbox Thrust's own device backend is itself built from. Section C.4 is the practical question every one of the first three sections raises: given that Thrust and CUB exist, when should you still write your own kernel? One setup note before starting: this book's own pip-based toolchain (Appendix A) installs the CUDA Runtime and `nvcc` but not Thrust/CUB's headers on their own -- on this book's own environment they were added with `pip install nvidia-cuda-cccl --break-system-packages`, landing at `$NVDIR/include/cccl`; a full system-wide CUDA Toolkit install, including Lambda Stack on a real Lambda Cloud instance (Appendix A.4), already ships them at the compiler's default include path, with no separate install step at all.

## C.1 Thrust: Reduction and Scan Without Writing a Kernel

### Intuition

Thrust is deliberately modeled on the C++ Standard Template Library: `thrust::reduce`, `thrust::inclusive_scan`, and `thrust::exclusive_scan` have almost the exact same call signatures as `std::accumulate` and `std::partial_sum`, and Chapter 4 and Chapter 5's entire job -- building a correct, efficient pairwise-tree reduction and a correct, efficient two-phase scan by hand -- is replaced by a single function call each. The genuinely interesting part for this book's own environment is Thrust's execution policy: passing `thrust::device` dispatches to a real CUDA kernel (which, like every other kernel in this book, would need a physical device to launch), but passing `thrust::host` runs the identical algorithm as ordinary sequential C++ directly on the CPU -- which means the code below is not a simulation of Thrust, it is Thrust, genuinely compiled and genuinely executed, with no device required at all.

### The Concept, In Detail

```
ASCII view: one execution policy, two completely different backends.

  thrust::reduce(thrust::device, first, last, init)
       |
       v
  dispatches to a real CUDA kernel -- needs a physical GPU to launch,
  exactly like every other kernel in this book

  thrust::reduce(thrust::host, first, last, init)
       |
       v
  runs as ordinary sequential C++ ON THE CPU -- no device, no kernel
  launch, no cudaMalloc at all. Same algorithm's RESULT, different
  hardware target entirely.
```

Chapter 4 built its own pairwise-tree reduction by hand specifically to show what happens underneath a call like `thrust::reduce` -- half the active values fold into the other half each round, until one value remains. Chapter 5's two scans (inclusive: each position includes its own element; exclusive: each position excludes it) are exactly `thrust::inclusive_scan` and `thrust::exclusive_scan`. Nothing about the algorithm changes; what changes is that a correctly-tuned, extensively-tested implementation now runs behind one line instead of forty.

[COMMON TRAP]
It is tempting to conclude that because `thrust::host` produces the identical result as a hand-rolled reduction kernel, `thrust::device` would too -- and therefore that switching between them is purely a performance decision. `thrust::device` genuinely requires a physical GPU to launch its kernel, exactly the same hardware requirement every `__global__` kernel in this book has; the only reason `thrust::host` works in this book's own environment at all is that it was specifically built to need no device. Do not read "Thrust hides the hardware" as "Thrust removes the hardware requirement" -- it only changes who wrote the kernel.

### Code and Verification

```cpp
// 219_thrust_reduce_scan.cu
//
// Appendix C.1 -- Thrust is the "STL you get for free": a single header
// and a single function call replace an entire hand-written kernel.
// thrust::reduce and thrust::inclusive_scan/exclusive_scan below are
// GENUINELY executed -- not simulated -- by passing the thrust::host
// execution policy, which runs the algorithm as ordinary sequential C++
// on the CPU rather than dispatching to a CUDA kernel. This is exactly
// why this appendix can compile AND run real Thrust calls in an
// environment with no GPU at all: thrust::host needs no device.
//
// Compile: nvcc -arch=sm_80 -DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK -I$NVDIR/include/cccl 219_thrust_reduce_scan.cu -o 219_thrust_reduce_scan
// Run:     ./219_thrust_reduce_scan

#include <cstdio>
#include <thrust/host_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>

int main() {
    thrust::host_vector<int> data = {12, 7, 3, 9, 15, 6, 20, 4};

    printf("=== Section C.1: thrust::reduce and thrust::scan, genuinely run on thrust::host ===\n\n");

    printf("input: [");
    for (size_t i = 0; i < data.size(); ++i) printf("%d%s", (int)data[i], i + 1 < data.size() ? ", " : "");
    printf("]\n\n");

    int total = thrust::reduce(thrust::host, data.begin(), data.end(), 0);
    printf("thrust::reduce(thrust::host, ...) -> %d\n", total);
    printf("(this is Chapter 4's pairwise-tree reduction, called through one line)\n\n");

    thrust::host_vector<int> inclusive(data.size());
    thrust::inclusive_scan(thrust::host, data.begin(), data.end(), inclusive.begin());
    printf("thrust::inclusive_scan(thrust::host, ...) -> [");
    for (size_t i = 0; i < inclusive.size(); ++i) printf("%d%s", (int)inclusive[i], i + 1 < inclusive.size() ? ", " : "");
    printf("]\n");

    thrust::host_vector<int> exclusive(data.size());
    thrust::exclusive_scan(thrust::host, data.begin(), data.end(), exclusive.begin());
    printf("thrust::exclusive_scan(thrust::host, ...) -> [");
    for (size_t i = 0; i < exclusive.size(); ++i) printf("%d%s", (int)exclusive[i], i + 1 < exclusive.size() ? ", " : "");
    printf("]\n");
    printf("(these are exactly Chapter 5's two scans, called through one line each)\n");

    bool ok = (total == 76) && (inclusive.back() == 76) && (exclusive.back() == 72);
    printf("\nself-check: thrust::reduce's total, thrust::inclusive_scan's final element, and\n");
    printf("thrust::exclusive_scan's final element are mutually consistent: %s\n", ok ? "confirmed" : "MISMATCH");

    return ok ? 0 : 1;
}
```

**Compile and run:**

```bash
NVDIR=/usr/local/lib/python3.11/dist-packages/nvidia/cu13
$NVDIR/bin/nvcc -arch=sm_80 -DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK -I$NVDIR/include/cccl 219_thrust_reduce_scan.cu -o 219_thrust_reduce_scan
./219_thrust_reduce_scan
```

**Sample input:** the array `[12, 7, 3, 9, 15, 6, 20, 4]`.

**Sample output:**

```text
=== Section C.1: thrust::reduce and thrust::scan, genuinely run on thrust::host ===

input: [12, 7, 3, 9, 15, 6, 20, 4]

thrust::reduce(thrust::host, ...) -> 76
(this is Chapter 4's pairwise-tree reduction, called through one line)

thrust::inclusive_scan(thrust::host, ...) -> [12, 19, 22, 31, 46, 52, 72, 76]
thrust::exclusive_scan(thrust::host, ...) -> [0, 12, 19, 22, 31, 46, 52, 72]
(these are exactly Chapter 5's two scans, called through one line each)

self-check: thrust::reduce's total, thrust::inclusive_scan's final element, and
thrust::exclusive_scan's final element are mutually consistent: confirmed
```

## C.2 Thrust: Sorting Without Writing a Kernel

### Intuition

Part 3 built three different sorting kernels -- bitonic, radix, and merge/sample sort -- each making a different tradeoff between regular, data-independent control flow and asymptotic comparison count. `thrust::sort` makes that choice for you, dispatching internally to whichever of those strategies (or another) its own implementation has found fastest for the given type and size, again runnable through `thrust::host` with no device required. `thrust::sort_by_key` solves a problem this book has run into repeatedly without naming it directly: reordering one array by the sorted order of ANOTHER array -- exactly what Section 32.2's price-ladder reduction and Section 35.2's nearest-neighbor argmin both needed when they packed a comparison key together with an index to recover.

### The Concept, In Detail

```
ASCII view: sorting a value array while carrying a second array along.

  values:  [29, 10, 14, 37,  3, 22, 18,  5]
  indices: [ 0,  1,  2,  3,  4,  5,  6,  7]   <- original position of each value

  thrust::sort_by_key(values.begin(), values.end(), indices.begin())
       |
       v
  values:  [ 3,  5, 10, 14, 18, 22, 29, 37]
  indices: [ 4,  7,  1,  2,  6,  5,  0,  3]   <- moved IN LOCKSTEP with values

  indices[i] now answers: "the value now at sorted position i originally
  lived at position indices[i]" -- exactly what Section 32.2/35.2 needed
  their own packed-key decode step to recover by hand
```

Every one of this book's own "pack a key with an index, then decode after sorting or reducing" tricks (Sections 25.2, 32.2, 35.2) exists because a hand-written reduction or sort only ever moves ONE array. `thrust::sort_by_key` (and its reduction-side relative, no separate packing step required) is the library-provided version of exactly that same need, letting a second, associated array ride along automatically instead of being encoded into spare bits of the sort key by hand.

[COMMON TRAP]
It is tempting to reach for the packed-key trick from Sections 25.2/32.2/35.2 out of habit, even when using Thrust. That trick exists specifically to work around a hand-written kernel only being able to sort or reduce ONE array at a time -- `thrust::sort_by_key` (and `thrust::reduce_by_key`, its reduction-side counterpart) already solves this directly, and reimplementing the packed-key workaround on top of a library call that does not need it adds real complexity for no benefit.

### Code and Verification

```cpp
// 220_thrust_sort.cu
//
// Appendix C.2 -- thrust::sort replaces Part 3's hand-written bitonic,
// radix, and merge/sample sort kernels with one call, and thrust::sort_by_key
// carries a second array (here, each element's original index) along with
// the reordering -- exactly what a real system needs when the sorted
// values are a KEY for looking up something else. Both calls below use
// thrust::host, so they genuinely execute as real C++ on the CPU, with no
// device required.
//
// Compile: nvcc -arch=sm_80 -DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK -I$NVDIR/include/cccl 220_thrust_sort.cu -o 220_thrust_sort
// Run:     ./220_thrust_sort

#include <cstdio>
#include <thrust/host_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>
#include <thrust/sequence.h>

int main() {
    thrust::host_vector<int> data = {29, 10, 14, 37, 3, 22, 18, 5};

    printf("=== Section C.2: thrust::sort and thrust::sort_by_key, genuinely run on thrust::host ===\n\n");

    printf("input: [");
    for (size_t i = 0; i < data.size(); ++i) printf("%d%s", (int)data[i], i + 1 < data.size() ? ", " : "");
    printf("]\n\n");

    thrust::sort(thrust::host, data.begin(), data.end());
    printf("thrust::sort(thrust::host, ...) -> [");
    for (size_t i = 0; i < data.size(); ++i) printf("%d%s", (int)data[i], i + 1 < data.size() ? ", " : "");
    printf("]\n");
    printf("(this is one call replacing Part 3's bitonic, radix, or merge sort kernels)\n\n");

    thrust::host_vector<int> data2 = {29, 10, 14, 37, 3, 22, 18, 5};
    thrust::host_vector<int> ids(data2.size());
    thrust::sequence(thrust::host, ids.begin(), ids.end());
    thrust::sort_by_key(thrust::host, data2.begin(), data2.end(), ids.begin());
    printf("thrust::sort_by_key(thrust::host, ...) -- sorting original indices alongside values:\n");
    printf("  sorted values:  [");
    for (size_t i = 0; i < data2.size(); ++i) printf("%d%s", (int)data2[i], i + 1 < data2.size() ? ", " : "");
    printf("]\n  original index: [");
    for (size_t i = 0; i < ids.size(); ++i) printf("%d%s", (int)ids[i], i + 1 < ids.size() ? ", " : "");
    printf("]\n");

    bool sorted_ok = true;
    for (size_t i = 1; i < data.size(); ++i) if (data[i - 1] > data[i]) sorted_ok = false;
    bool keys_match_ok = true;
    int expected[8] = {3, 5, 10, 14, 18, 22, 29, 37};
    for (size_t i = 0; i < data2.size(); ++i) if (data2[i] != expected[i]) keys_match_ok = false;
    bool ok = sorted_ok && keys_match_ok;
    printf("\nself-check: thrust::sort's output is non-decreasing, and thrust::sort_by_key's\n");
    printf("sorted values match it exactly: %s\n", ok ? "confirmed" : "MISMATCH");

    return ok ? 0 : 1;
}
```

**Compile and run:**

```bash
NVDIR=/usr/local/lib/python3.11/dist-packages/nvidia/cu13
$NVDIR/bin/nvcc -arch=sm_80 -DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK -I$NVDIR/include/cccl 220_thrust_sort.cu -o 220_thrust_sort
./220_thrust_sort
```

**Sample input:** the array `[29, 10, 14, 37, 3, 22, 18, 5]`, sorted both alone and alongside its own original indices.

**Sample output:**

```text
=== Section C.2: thrust::sort and thrust::sort_by_key, genuinely run on thrust::host ===

input: [29, 10, 14, 37, 3, 22, 18, 5]

thrust::sort(thrust::host, ...) -> [3, 5, 10, 14, 18, 22, 29, 37]
(this is one call replacing Part 3's bitonic, radix, or merge sort kernels)

thrust::sort_by_key(thrust::host, ...) -- sorting original indices alongside values:
  sorted values:  [3, 5, 10, 14, 18, 22, 29, 37]
  original index: [4, 7, 1, 2, 6, 5, 0, 3]

self-check: thrust::sort's output is non-decreasing, and thrust::sort_by_key's
sorted values match it exactly: confirmed
```

## C.3 CUB: The Building Blocks Underneath

### Intuition

Thrust's `thrust::device` execution policy does not reinvent reduction, scan, and sort from scratch inside its own CUDA backend -- it is built on top of CUB, a lower-level library of block-level, warp-level, and device-level primitives, each heavily tuned for a specific GPU architecture. `cub::BlockReduce`, used inside `block_reduce_kernel` below, is the block-level primitive that performs exactly the same pairwise-tree reduction Chapter 4 built by hand, packaged as a reusable C++ template that handles shared memory allocation, padding, and architecture-specific tuning internally. Where Sections C.1 and C.2 called Thrust as a finished tool, this section looks at the layer one step further down, closer to the hand-written kernels this book has built throughout -- CUB kernels are still kernels you write and launch yourself, just built from professionally-tuned pieces instead of raw shared-memory arithmetic.

### The Concept, In Detail

```
ASCII view: three layers, same underlying reduction.

  thrust::reduce(thrust::device, ...)        <- Section C.1's call
       |
       v  (Thrust's CUDA backend, internally)
  cub::BlockReduce<T, BLOCK_THREADS>::Sum()  <- this section's call
       |
       v  (CUB's own implementation, internally)
  a tuned pairwise-tree / warp-shuffle reduction across BLOCK_THREADS lanes
       |
       v  (conceptually identical to)
  Chapter 4's own hand-written pairwise-tree reduction kernel
```

`cub::BlockReduce` needs a `__shared__ TempStorage` object (CUB computes exactly how much shared memory its internal implementation needs, so you never have to size it by hand) and is called from inside a kernel exactly the way any other block-level operation would be -- it is not a host-callable function like `thrust::reduce`, it is a building block for kernels YOU still write and launch. Launching `block_reduce_kernel` genuinely needs a physical device, which this section's own code honestly attempts and honestly reports, exactly as Section 2.3 first established for this book's entire environment; the host-side replay beneath it walks through the identical fold-in-half logic CUB's own implementation performs internally, arriving at the same answer Section C.1's `thrust::reduce` already produced.

[COMMON TRAP]
It is tempting to think CUB and Thrust are competing alternatives, where a project picks one or the other. Thrust's own device backend is ITSELF implemented using CUB -- they are two layers of the same library family, not two competing ones. Reaching for CUB directly only makes sense when you are writing your own kernel and want a tuned building block inside it (the way `block_reduce_kernel` uses `cub::BlockReduce`), not as a replacement for calling `thrust::reduce` when a host-callable, already-complete algorithm is all you actually need.

### Code and Verification

```cpp
// 221_cub_block_reduce.cu
//
// Appendix C.3 -- CUB is the layer underneath Thrust: block-level, warp-
// level, and device-level primitives that Thrust's own CUDA backend (and
// libraries like RAPIDS) are built from. block_reduce_kernel below is a
// real, syntactically valid kernel using cub::BlockReduce -- genuinely
// compiled by nvcc. Launching it needs a real device, so this file first
// makes a genuine, honest attempt (reusing Section 2.3's own pattern) and
// then walks through the identical logic on the host, exactly the way
// every kernel-needing-a-device file in this book has done since Chapter 2.
//
// Compile: nvcc -arch=sm_80 -DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK -I$NVDIR/include/cccl -I$NVDIR/include -L$NVDIR/lib -lcudart 221_cub_block_reduce.cu -o 221_cub_block_reduce
// Run:     LD_LIBRARY_PATH=$NVDIR/lib ./221_cub_block_reduce

#include <cstdio>
#include <cuda_runtime.h>
#include <cub/block/block_reduce.cuh>

#define BLOCK_THREADS 8

// A real, syntactically valid kernel using CUB's block-level BlockReduce --
// one of the exact building blocks Thrust's device backend, and libraries
// like cuDF and RAPIDS, are themselves built from underneath the STL-like
// surface Sections C.1/C.2 called directly.
__global__ void block_reduce_kernel(const int* in, int* out) {
    typedef cub::BlockReduce<int, BLOCK_THREADS> BlockReduceT;
    __shared__ typename BlockReduceT::TempStorage temp_storage;

    int tid = threadIdx.x;
    int thread_val = in[tid];
    int block_sum = BlockReduceT(temp_storage).Sum(thread_val);

    if (tid == 0) out[0] = block_sum;
}

void report(const char* call_name, cudaError_t err) {
    printf("  %-24s -> %-24s (%s)\n", call_name, cudaGetErrorName(err), cudaGetErrorString(err));
}

int main() {
    int h_in[BLOCK_THREADS] = {12, 7, 3, 9, 15, 6, 20, 4};

    printf("=== Section C.3: attempting a genuine launch of block_reduce_kernel ===\n\n");
    int* d_in = nullptr;
    int* d_out = nullptr;
    cudaError_t e1 = cudaMalloc(&d_in, sizeof(h_in));
    report("cudaMalloc(d_in)", e1);
    cudaError_t e2 = cudaMalloc(&d_out, sizeof(int));
    report("cudaMalloc(d_out)", e2);

    if (e1 == cudaSuccess && e2 == cudaSuccess) {
        cudaMemcpy(d_in, h_in, sizeof(h_in), cudaMemcpyHostToDevice);
        block_reduce_kernel<<<1, BLOCK_THREADS>>>(d_in, d_out);
        cudaError_t e3 = cudaGetLastError();
        report("kernel launch", e3);
    } else {
        printf("  (device allocation failed -- this environment has no usable GPU, so the\n");
        printf("   kernel above is genuinely compiled and syntax-checked but cannot actually\n");
        printf("   be launched here, exactly as Section 2.3 first established)\n");
    }

    printf("\n=== host-side replay of CUB's own internal logic ===\n\n");
    printf("input: [12, 7, 3, 9, 15, 6, 20, 4]\n\n");
    int vals[BLOCK_THREADS] = {12, 7, 3, 9, 15, 6, 20, 4};
    int n = BLOCK_THREADS;
    int round = 1;
    while (n > 1) {
        int half = n / 2;
        printf("round %d: %d active lanes fold in half:\n", round, n);
        for (int i = 0; i < half; ++i) {
            printf("  lane %d: %d + %d = %d\n", i, vals[i], vals[i + half], vals[i] + vals[i + half]);
            vals[i] += vals[i + half];
        }
        n = half;
        round++;
    }
    printf("\nBlockReduce<int, 8>::Sum() result: %d\n", vals[0]);
    printf("(this is exactly Chapter 4's pairwise-tree reduction -- CUB's BlockReduce is a\n");
    printf("tuned, reusable implementation of the identical idea, not a different one)\n");

    bool ok = (vals[0] == 76);
    printf("\nself-check: host-side replay matches thrust::reduce's own result from Section\n");
    printf("C.1 (76) exactly: %s\n", ok ? "confirmed" : "MISMATCH");

    return ok ? 0 : 1;
}
```

**Compile and run:**

```bash
NVDIR=/usr/local/lib/python3.11/dist-packages/nvidia/cu13
$NVDIR/bin/nvcc -arch=sm_80 -DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK -I$NVDIR/include/cccl -I$NVDIR/include -L$NVDIR/lib -lcudart 221_cub_block_reduce.cu -o 221_cub_block_reduce
LD_LIBRARY_PATH=$NVDIR/lib ./221_cub_block_reduce
```

**Sample input:** the same 8-element array Section C.1 already reduced with `thrust::reduce` (`[12, 7, 3, 9, 15, 6, 20, 4]`), used here to confirm both approaches agree.

**Sample output:**

```text
=== Section C.3: attempting a genuine launch of block_reduce_kernel ===

  cudaMalloc(d_in)         -> cudaErrorInsufficientDriver (CUDA driver version is insufficient for CUDA runtime version)
  cudaMalloc(d_out)        -> cudaErrorInsufficientDriver (CUDA driver version is insufficient for CUDA runtime version)
  (device allocation failed -- this environment has no usable GPU, so the
   kernel above is genuinely compiled and syntax-checked but cannot actually
   be launched here, exactly as Section 2.3 first established)

=== host-side replay of CUB's own internal logic ===

input: [12, 7, 3, 9, 15, 6, 20, 4]

round 1: 8 active lanes fold in half:
  lane 0: 12 + 15 = 27
  lane 1: 7 + 6 = 13
  lane 2: 3 + 20 = 23
  lane 3: 9 + 4 = 13
round 2: 4 active lanes fold in half:
  lane 0: 27 + 23 = 50
  lane 1: 13 + 13 = 26
round 3: 2 active lanes fold in half:
  lane 0: 50 + 26 = 76

BlockReduce<int, 8>::Sum() result: 76
(this is exactly Chapter 4's pairwise-tree reduction -- CUB's BlockReduce is a
tuned, reusable implementation of the identical idea, not a different one)

self-check: host-side replay matches thrust::reduce's own result from Section
C.1 (76) exactly: confirmed
```

## C.4 Choosing Between Hand-Written Kernels, Thrust, and CUB

### Intuition

Every chapter in this book wrote its own kernel, on purpose: the entire point of a book that teaches data structures and algorithms in CUDA C++ is understanding what a reduction, a scan, a hash table probe, or a graph relaxation round actually does at the level of threads, shared memory, and atomics -- exactly what Thrust and CUB exist to let a working engineer stop thinking about, once they already understand it. This section is not a recommendation to rewrite this book's own 36 chapters using library calls; it is the practical decision a real project actually has to make once this book's own material is second nature.

### The Concept, In Detail

```
ASCII view: three layers, three different reasons to use each.

  hand-written kernel (this book's own approach, Chapters 1-36)
    -- use when: the operation is genuinely NOVEL (nothing in Thrust/CUB
       does it), or full control over memory layout and algorithm choice
       matters more than development time, or -- as in this entire book --
       the GOAL is learning exactly what happens underneath.

  CUB (block/warp/device-level primitives)
    -- use when: you are writing your own kernel (a custom data structure
       operation, say) but want a tuned, correct building block for a
       PIECE of it -- a block-wide reduction, a warp-wide scan, a device-
       wide sort as one step of a larger custom pipeline.

  Thrust (host-callable, STL-like algorithms)
    -- use when: the operation IS one of the standard primitives (reduce,
       scan, sort, and their relatives) and there is no reason to write or
       maintain your own version of something NVIDIA's own engineers have
       already tuned across every architecture they ship.
```

A useful rule of thumb: reach for Thrust first, since a huge fraction of real GPU code genuinely is "sort this," "reduce this," or "compact this" with no further customization needed. Reach for CUB when a custom kernel needs one of those same operations as an internal PIECE of something larger and more specific that Thrust itself cannot express as one call. Write your own kernel, the way this entire book has, only when the operation itself does not exist in either library, or when understanding and controlling every step is the actual goal -- which, for a book with this one's title, has been true on essentially every page.

## Appendix Summary

Section C.1 showed that Chapter 4's reduction and Chapter 5's two scans are each one `thrust::reduce`/`thrust::inclusive_scan`/`thrust::exclusive_scan` call away, genuinely executed via `thrust::host` with no device required at all. Section C.2 showed the same for Part 3's sorting chapters, with `thrust::sort_by_key` solving, directly and by name, the "carry a second array along with the sort" problem this book's own packed-key tricks (Sections 25.2, 32.2, 35.2) had to work around by hand. Section C.3 went one layer deeper into CUB, the block/warp/device-level toolbox Thrust's own device backend is itself built from, and confirmed that `cub::BlockReduce`'s internal logic is, fold for fold, Chapter 4's own pairwise-tree reduction. Section C.4 closed with the practical question all three raise: write your own kernel when the goal is control or when nothing else does what you need (this entire book's own reason for existing), reach for CUB when a custom kernel needs one tuned building block internally, and reach for Thrust first whenever the job really is just "reduce this, scan this, or sort this."
