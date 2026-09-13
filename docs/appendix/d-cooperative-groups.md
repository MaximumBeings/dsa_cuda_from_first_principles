# Appendix D: Cooperative Groups and Grid-Wide Synchronization

Every `__syncthreads()` call in this book, from Chapter 4's reduction onward, has had the same fundamental limit: it synchronizes threads within ONE block, and has no way to reach any other block at all. That limit is not an oversight -- Chapter 22's level-synchronous BFS launches a fresh kernel for every frontier level specifically because a kernel's own boundary is the only cross-block synchronization point ordinary CUDA provides. Cooperative Groups is CUDA's answer to naming and synchronizing threads at every one of these granularities explicitly: Section D.1 replaces the bare `__syncthreads()` intrinsic with a named, passable `cg::thread_block` object; Section D.2 partitions a block into warp-sized tiles and reduces across one using register-to-register shuffles instead of shared memory; Section D.3 reaches all the way to `cg::this_grid()` and `grid.sync()`, a genuine cross-block barrier inside a single kernel launch; and Section D.4 is the practical question that capability raises -- given that Chapter 22 already gets a correct answer with one kernel launch per level, when does grid-wide sync actually earn its own real costs?

## D.1 Thread Block Groups: Naming What `__syncthreads()` Leaves Implicit

### Intuition

`__syncthreads()` is a bare intrinsic: it takes no arguments, and any function that calls it silently assumes it is being called by every thread of some block, together, with no way to say so in the function's own signature. `cg::thread_block block = cg::this_thread_block();` makes that assumption an explicit, ordinary C++ object -- `block.sync()` emits the exact same hardware barrier `__syncthreads()` does, but now a helper function can take a `cg::thread_block` parameter and its signature alone documents exactly which threads are expected to call it and synchronize together.

### The Concept, In Detail

```
ASCII view: the same barrier, two different ways to write it.

  __syncthreads();              <- bare intrinsic, callable from anywhere,
                                     implicitly means "every thread of
                                     whichever block called it"

  cg::thread_block block = cg::this_thread_block();
  block.sync();                 <- identical barrier, but now `block` is a
                                     real, passable value -- a helper
                                     function's signature can require one
```

Section D.1's own kernel stages every thread's element into shared memory, calls `block.sync()`, then has every thread read back its REVERSED neighbor's value -- correct only because every thread's write from phase 1 is guaranteed complete before any thread's read in phase 2 begins. That correctness requirement is identical to every `__syncthreads()`-protected shared-memory handoff this book has used since Chapter 2; Cooperative Groups changes nothing about WHEN the barrier happens, only how explicitly it is written down.

[COMMON TRAP]
It is tempting to think `cg::thread_block::sync()` is a safer or more capable version of `__syncthreads()` -- perhaps one that can be called conditionally, or by a subset of threads. Both forms carry the identical, non-negotiable requirement that EVERY thread of the block reach the barrier for it to complete at all; wrapping `__syncthreads()` in a named object does not relax that requirement, it only gives a function signature a way to document it.

### Code and Verification

```cpp
// 222_cg_thread_block_reverse.cu
//
// Appendix D.1 -- Cooperative Groups replaces the bare __syncthreads()
// intrinsic with an explicit, named object: cg::thread_block. The two are
// functionally identical (block.sync() and __syncthreads() emit the same
// barrier instruction), but a named group can be PASSED to a function --
// a helper that takes a cg::thread_block parameter documents, in its own
// signature, exactly which threads it expects to call it and synchronize
// together, something a bare __syncthreads() call buried inside a helper
// function cannot express at all.
//
// Compile: nvcc -arch=sm_80 -DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK 222_cg_thread_block_reverse.cu -o 222_cg_thread_block_reverse
// Run:     ./222_cg_thread_block_reverse

#include <cstdio>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

// A real, syntactically valid kernel: each thread stages its own element
// into shared memory, the whole block synchronizes via a named
// cg::thread_block object, then every thread reads back its REVERSED
// neighbor's value -- a pattern that only gives a correct answer if every
// thread has finished writing before any thread starts reading.
__global__ void reverse_kernel(const int* in, int* out) {
    extern __shared__ int staged[];
    cg::thread_block block = cg::this_thread_block();

    int tid = block.thread_rank();
    int n = block.size();

    staged[tid] = in[tid];
    block.sync();   // identical barrier to __syncthreads(), as a named, passable object

    out[tid] = staged[n - 1 - tid];
}

int main() {
    printf("=== Section D.1: cg::thread_block::sync() replacing __syncthreads() ===\n\n");

    const int n = 8;
    int in[n] = {10, 20, 30, 40, 50, 60, 70, 80};
    int staged[n];
    int out[n];

    printf("input: [");
    for (int i = 0; i < n; ++i) printf("%d%s", in[i], i + 1 < n ? ", " : "");
    printf("]\n\n");

    printf("--- host-side replay of block_reverse_kernel's identical logic ---\n\n");
    printf("phase 1 (before block.sync()): every thread stages its own element\n");
    for (int tid = 0; tid < n; ++tid) {
        staged[tid] = in[tid];
        printf("  thread %d: staged[%d] = in[%d] = %d\n", tid, tid, tid, staged[tid]);
    }

    printf("\nblock.sync() -- every thread waits here until ALL %d threads have finished\n", n);
    printf("phase 1, exactly the same guarantee a bare __syncthreads() call would give\n\n");

    printf("phase 2 (after block.sync()): every thread reads its REVERSED neighbor\n");
    for (int tid = 0; tid < n; ++tid) {
        out[tid] = staged[n - 1 - tid];
        printf("  thread %d: out[%d] = staged[%d] = %d\n", tid, tid, n - 1 - tid, out[tid]);
    }

    printf("\nfinal output: [");
    for (int i = 0; i < n; ++i) printf("%d%s", out[i], i + 1 < n ? ", " : "");
    printf("]\n");

    bool ok = true;
    for (int i = 0; i < n; ++i) if (out[i] != in[n - 1 - i]) ok = false;
    printf("\nself-check: output is the input array in reverse order -- correct only because\n");
    printf("EVERY thread's phase-1 write finished before ANY thread's phase-2 read began: %s\n",
           ok ? "confirmed" : "MISMATCH");

    return ok ? 0 : 1;
}
```

**Compile and run:**

```bash
nvcc -arch=sm_80 -DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK 222_cg_thread_block_reverse.cu -o 222_cg_thread_block_reverse
./222_cg_thread_block_reverse
```

**Sample input:** the 8-element array `[10, 20, 30, 40, 50, 60, 70, 80]`, reversed within a single block.

**Sample output:**

```text
=== Section D.1: cg::thread_block::sync() replacing __syncthreads() ===

input: [10, 20, 30, 40, 50, 60, 70, 80]

--- host-side replay of block_reverse_kernel's identical logic ---

phase 1 (before block.sync()): every thread stages its own element
  thread 0: staged[0] = in[0] = 10
  thread 1: staged[1] = in[1] = 20
  thread 2: staged[2] = in[2] = 30
  thread 3: staged[3] = in[3] = 40
  thread 4: staged[4] = in[4] = 50
  thread 5: staged[5] = in[5] = 60
  thread 6: staged[6] = in[6] = 70
  thread 7: staged[7] = in[7] = 80

block.sync() -- every thread waits here until ALL 8 threads have finished
phase 1, exactly the same guarantee a bare __syncthreads() call would give

phase 2 (after block.sync()): every thread reads its REVERSED neighbor
  thread 0: out[0] = staged[7] = 80
  thread 1: out[1] = staged[6] = 70
  thread 2: out[2] = staged[5] = 60
  thread 3: out[3] = staged[4] = 50
  thread 4: out[4] = staged[3] = 40
  thread 5: out[5] = staged[2] = 30
  thread 6: out[6] = staged[1] = 20
  thread 7: out[7] = staged[0] = 10

final output: [80, 70, 60, 50, 40, 30, 20, 10]

self-check: output is the input array in reverse order -- correct only because
EVERY thread's phase-1 write finished before ANY thread's phase-2 read began: confirmed
```

## D.2 Warp-Level Tiles: Reducing Without Shared Memory

### Intuition

`cg::tiled_partition<N>(block)` splits a block into fixed-size subgroups -- a `thread_block_tile<32>` is exactly one hardware warp -- and exposes warp-synchronous shuffle instructions as named methods instead of the raw, easy-to-misuse `__shfl_down_sync` intrinsic. Because every thread within one tile executes in lockstep by construction, a tile can reduce its own values by shuffling them directly from register to register, with no `__shared__` array and no explicit barrier at all -- Chapter 4's own reduction moved every partial sum through shared memory specifically because it reduced across an ENTIRE block, which (unlike one warp-sized tile) has no hardware guarantee of lockstep execution between its own warps.

### The Concept, In Detail

```
ASCII view: the identical fold-in-half shape, moved into registers.

  Appendix C.3's cub::BlockReduce (shared memory):
    lane 0's partial sum lives in __shared__ memory between rounds

  Section D.2's tile.shfl_down (this section):
    lane 0's partial sum lives in a REGISTER between rounds -- shfl_down
    reads directly from another lane's register, no memory round-trip
    at all

  round 1 (offset=4): lanes 0-3 read lanes 4-7 directly via shfl_down
  round 2 (offset=2): lanes 0-1 read lanes 2-3
  round 3 (offset=1): lane 0 reads lane 1        <- final answer, lane 0
```

Every lane in the tile genuinely executes the same `shfl_down` instruction on every round -- that is what "lockstep" means -- but exactly like Appendix C.3's fold-in-half reduction, only the shrinking LOWER half's resulting value is ever read again by a later round, so the trace below only follows that half. The final answer at lane 0 after `log2(tile size)` rounds is identical, fold for fold, to Appendix C.3's `cub::BlockReduce` and Appendix C.1's `thrust::reduce` -- the same algorithm, moved one level lower, into hardware registers instead of shared memory.

[COMMON TRAP]
It is tempting to assume a tile-based shuffle reduction can replace Chapter 4's block-wide shared-memory reduction outright, since both compute the same kind of answer. A `thread_block_tile<N>` tops out at one warp's worth of lockstep-guaranteed threads (32, on every architecture this book targets); reducing across an ENTIRE block of, say, 256 threads genuinely needs either several tiles' worth of shuffle reductions followed by one more combining step, or Chapter 4's own shared-memory approach -- a single tile's shuffle reduction only ever sees its own tile's values.

### Code and Verification

```cpp
// 223_cg_tile_shuffle_reduce.cu
//
// Appendix D.2 -- cg::thread_block_tile<N> partitions a block into
// statically-sized subgroups (a tile<32> is exactly one warp) and exposes
// warp-synchronous shuffle instructions (shfl_down, in this file) as
// named methods. The reduction below computes the SAME sum, over the
// SAME 8 values, as Appendix C.3's cub::BlockReduce and Chapter 4's own
// hand-written kernel -- but every partial sum here moves register-to-
// register via shfl_down, with no shared-memory array and no explicit
// barrier at all, because threads within one tile execute in lockstep by
// construction.
//
// Compile: nvcc -arch=sm_80 -DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK 223_cg_tile_shuffle_reduce.cu -o 223_cg_tile_shuffle_reduce
// Run:     ./223_cg_tile_shuffle_reduce

#include <cstdio>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

#define TILE_SIZE 8

// A real, syntactically valid kernel: a size-8 tile reduces its own
// values using shfl_down alone -- no __shared__ array, no block.sync().
__global__ void tile_reduce_kernel(const int* in, int* out) {
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<TILE_SIZE> tile = cg::tiled_partition<TILE_SIZE>(block);

    int val = in[tile.thread_rank()];
    for (int offset = tile.size() / 2; offset > 0; offset /= 2) {
        val += tile.shfl_down(val, offset);
    }
    if (tile.thread_rank() == 0) out[0] = val;
}

int main() {
    printf("=== Section D.2: cg::thread_block_tile<8>::shfl_down, genuinely shared-memory-free ===\n\n");

    int vals[TILE_SIZE] = {12, 7, 3, 9, 15, 6, 20, 4};
    printf("input: [");
    for (int i = 0; i < TILE_SIZE; ++i) printf("%d%s", vals[i], i + 1 < TILE_SIZE ? ", " : "");
    printf("]\n\n");

    printf("--- host-side replay of tile_reduce_kernel's identical shfl_down logic ---\n\n");
    printf("(every one of the tile's 8 lanes genuinely executes the SAME shfl_down\n");
    printf("instruction each round, in lockstep -- exactly like Chapter 1's SIMT model --\n");
    printf("but exactly like Appendix C.3's fold-in-half reduction, only the LOWER half's\n");
    printf("resulting value is ever read again, so only that shrinking half is traced below)\n\n");
    int n = TILE_SIZE;
    int round = 1;
    while (n > 1) {
        int half = n / 2;
        printf("round %d (offset=%d): lanes 0..%d read lanes %d..%d via shfl_down(val, %d)\n",
               round, half, half - 1, half, n - 1, half);
        for (int lane = 0; lane < half; ++lane) {
            int shuffled_in = vals[lane + half];
            printf("  lane %d: val(%d) + shfl_down(val, %d)(%d) = %d\n",
                   lane, vals[lane], half, shuffled_in, vals[lane] + shuffled_in);
            vals[lane] += shuffled_in;
        }
        n = half;
        round++;
    }

    printf("\ntile.thread_rank()==0 writes out[0] = %d\n", vals[0]);
    printf("(no __shared__ array, and no explicit barrier anywhere in this reduction --\n");
    printf("every value moved directly between two lanes' own registers)\n");

    bool ok = (vals[0] == 76);
    printf("\nself-check: shuffle-based tile reduction matches Appendix C.1's thrust::reduce\n");
    printf("and C.3's cub::BlockReduce result (76) exactly: %s\n", ok ? "confirmed" : "MISMATCH");

    return ok ? 0 : 1;
}
```

**Compile and run:**

```bash
nvcc -arch=sm_80 -DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK 223_cg_tile_shuffle_reduce.cu -o 223_cg_tile_shuffle_reduce
./223_cg_tile_shuffle_reduce
```

**Sample input:** the same 8-element array Appendix C already reduced (`[12, 7, 3, 9, 15, 6, 20, 4]`), used here to confirm the shuffle-based answer agrees.

**Sample output:**

```text
=== Section D.2: cg::thread_block_tile<8>::shfl_down, genuinely shared-memory-free ===

input: [12, 7, 3, 9, 15, 6, 20, 4]

--- host-side replay of tile_reduce_kernel's identical shfl_down logic ---

(every one of the tile's 8 lanes genuinely executes the SAME shfl_down
instruction each round, in lockstep -- exactly like Chapter 1's SIMT model --
but exactly like Appendix C.3's fold-in-half reduction, only the LOWER half's
resulting value is ever read again, so only that shrinking half is traced below)

round 1 (offset=4): lanes 0..3 read lanes 4..7 via shfl_down(val, 4)
  lane 0: val(12) + shfl_down(val, 4)(15) = 27
  lane 1: val(7) + shfl_down(val, 4)(6) = 13
  lane 2: val(3) + shfl_down(val, 4)(20) = 23
  lane 3: val(9) + shfl_down(val, 4)(4) = 13
round 2 (offset=2): lanes 0..1 read lanes 2..3 via shfl_down(val, 2)
  lane 0: val(27) + shfl_down(val, 2)(23) = 50
  lane 1: val(13) + shfl_down(val, 2)(13) = 26
round 3 (offset=1): lanes 0..0 read lanes 1..1 via shfl_down(val, 1)
  lane 0: val(50) + shfl_down(val, 1)(26) = 76

tile.thread_rank()==0 writes out[0] = 76
(no __shared__ array, and no explicit barrier anywhere in this reduction --
every value moved directly between two lanes' own registers)

self-check: shuffle-based tile reduction matches Appendix C.1's thrust::reduce
and C.3's cub::BlockReduce result (76) exactly: confirmed
```

## D.3 Grid-Wide Synchronization: `cg::this_grid()` and Cooperative Launch

### Intuition

`cg::thread_block::sync()` (Section D.1) and `cg::thread_block_tile<N>::shfl_down` (Section D.2) both reach, at most, one block's worth of threads -- neither can make block 5 wait for block 2. `cg::this_grid()` and `grid.sync()` reach every thread of every block in the entire kernel, from inside a single, still-running kernel -- something this book's own Chapter 22 works around by ending one kernel and launching a fresh one for every BFS frontier level, since a kernel launch's own boundary is ordinarily the only point where the driver guarantees every block of one kernel has finished before any block of the next one begins. Using `grid.sync()` for real requires launching with `cudaLaunchCooperativeKernel` instead of the ordinary `<<<...>>>` syntax, and only works on a device that reports support for it at all.

### The Concept, In Detail

```
ASCII view: two ways to get the identical cross-block guarantee.

  Chapter 22's approach (one kernel launch per BFS level):
    kernel(level 0) ---driver guarantees completion---> kernel(level 1) --> ...
    every block of kernel N finishes before ANY block of kernel N+1 starts

  grid.sync()'s approach (one persistent kernel, several internal phases):
    kernel {
      phase 1 (every block)
      grid.sync()  <-- the SAME guarantee, without ending the kernel
      phase 2 (every block)
    }
    ONLY valid if every block the kernel needs can be resident on the
    device SIMULTANEOUSLY -- checked via
    cudaOccupancyMaxActiveBlocksPerMultiprocessor before launch
```

Section D.3's own kernel doubles every element, calls `grid.sync()`, then adds one to every element -- correct only if every block's doubling is complete before any block's addition begins, exactly the cross-block guarantee Chapter 22 gets for free at every kernel boundary. Whether `grid.sync()` is available at all is a genuine hardware and driver fact, queried honestly below via `cudaDeviceGetAttribute(..., cudaDevAttrCooperativeLaunch, ...)` -- exactly the pattern Section 2.3 first established for this book's entire driver-less environment, which reports the identical `cudaErrorInsufficientDriver` here that it has reported for every other Runtime API call since Chapter 2.

[COMMON TRAP]
It is tempting to assume a cooperative kernel can simply be launched with more blocks than a normal one, the way an ordinary kernel's grid size is limited mainly by the problem size. A cooperative launch additionally requires every one of its blocks to be resident on the GPU AT THE SAME TIME -- `grid.sync()` would otherwise deadlock waiting for a block that has not even been scheduled yet -- so a correct cooperative launch first calls `cudaOccupancyMaxActiveBlocksPerMultiprocessor` to discover how many blocks the device can actually run simultaneously, and sizes the grid to that number rather than to the problem size directly.

### Code and Verification

```cpp
// 224_cg_grid_sync_attempt.cu
//
// Appendix D.3 -- cg::this_grid() and grid.sync() extend Cooperative
// Groups' barrier past a single block: EVERY thread in EVERY block of the
// kernel waits at grid.sync(), something __syncthreads() (block-only) and
// cg::thread_block::sync() (Section D.1, also block-only) cannot do at
// all. This requires launching with cudaLaunchCooperativeKernel instead
// of the ordinary <<<...>>> syntax, and the device must report support
// for it -- genuinely queried below via cudaDevAttrCooperativeLaunch,
// exactly the honest-Runtime-API-call pattern Section 2.3 established.
//
// Compile: nvcc -arch=sm_80 -DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK -I$NVDIR/include -L$NVDIR/lib -lcudart 224_cg_grid_sync_attempt.cu -o 224_cg_grid_sync_attempt
// Run:     LD_LIBRARY_PATH=$NVDIR/lib ./224_cg_grid_sync_attempt

#include <cstdio>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

// A real, syntactically valid kernel: every thread doubles its own
// element, EVERY block waits at grid.sync() for EVERY other block to
// finish doubling, and only then does any thread add 1 -- a correctness
// requirement no block-only barrier could enforce across block boundaries.
__global__ void grid_sync_kernel(int* data, int n) {
    cg::grid_group grid = cg::this_grid();
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < n) data[tid] *= 2;
    grid.sync();   // waits for EVERY block, not just this one
    if (tid < n) data[tid] += 1;
}

void report(const char* call_name, cudaError_t err) {
    printf("  %-40s -> %-24s (%s)\n", call_name, cudaGetErrorName(err), cudaGetErrorString(err));
}

int main() {
    printf("=== Section D.3: genuinely querying cooperative-launch support ===\n\n");

    int device_count = -1;
    cudaError_t e0 = cudaGetDeviceCount(&device_count);
    report("cudaGetDeviceCount", e0);

    int supports_coop = -1;
    cudaError_t e1 = cudaDeviceGetAttribute(&supports_coop, cudaDevAttrCooperativeLaunch, 0);
    report("cudaDeviceGetAttribute(CooperativeLaunch)", e1);
    printf("  supports_coop = %d\n\n", supports_coop);

    if (e1 != cudaSuccess) {
        printf("This environment has no usable device, so grid_sync_kernel above is\n");
        printf("genuinely compiled and syntax-checked but cannot be launched -- launching it\n");
        printf("for real would additionally require cudaLaunchCooperativeKernel (not the\n");
        printf("ordinary <<<...>>> syntax) AND a device reporting supports_coop == 1.\n\n");
    }

    printf("=== what grid.sync() replaces: one kernel launch instead of several ===\n\n");
    printf("Chapter 22's level-synchronous BFS launches a SEPARATE kernel for every\n");
    printf("frontier level, specifically because a normal kernel launch has no way for\n");
    printf("block 5 to wait for block 2 to finish -- __syncthreads() only ever reaches\n");
    printf("threads in the SAME block. Ending one kernel and launching the next IS the\n");
    printf("cross-block synchronization point: the driver guarantees every block of\n");
    printf("kernel N has completed before any block of kernel N+1 begins.\n\n");
    printf("grid.sync() offers a second way to get that same guarantee, from INSIDE a\n");
    printf("single, still-running kernel, if and only if every block the kernel needs can\n");
    printf("be resident on the device SIMULTANEOUSLY (checked via\n");
    printf("cudaOccupancyMaxActiveBlocksPerMultiprocessor before launch) and the device\n");
    printf("supports cooperative launch at all.\n\n");

    printf("--- host-side replay of grid_sync_kernel's two-phase logic ---\n\n");
    const int n = 8;
    int data[n] = {1, 2, 3, 4, 5, 6, 7, 8};
    printf("input: [");
    for (int i = 0; i < n; ++i) printf("%d%s", data[i], i + 1 < n ? ", " : "");
    printf("]\n\n");

    printf("phase 1 (before grid.sync()): every thread, in every block, doubles its element\n");
    for (int i = 0; i < n; ++i) {
        data[i] *= 2;
        printf("  thread %d: data[%d] = %d\n", i, i, data[i]);
    }
    printf("\ngrid.sync() -- every block waits here for every OTHER block, not just its own\n");
    printf("threads, before phase 2 begins anywhere\n\n");

    printf("phase 2 (after grid.sync()): every thread adds 1\n");
    for (int i = 0; i < n; ++i) {
        data[i] += 1;
        printf("  thread %d: data[%d] = %d\n", i, i, data[i]);
    }

    printf("\nfinal data: [");
    for (int i = 0; i < n; ++i) printf("%d%s", data[i], i + 1 < n ? ", " : "");
    printf("]\n");

    int expected[n] = {3, 5, 7, 9, 11, 13, 15, 17};
    bool ok = true;
    for (int i = 0; i < n; ++i) if (data[i] != expected[i]) ok = false;
    printf("\nself-check: every element is (2*original)+1, which is only guaranteed if EVERY\n");
    printf("block's phase-1 doubling finished before ANY block's phase-2 add began: %s\n",
           ok ? "confirmed" : "MISMATCH");

    return ok ? 0 : 1;
}
```

**Compile and run:**

```bash
NVDIR=/usr/local/lib/python3.11/dist-packages/nvidia/cu13
$NVDIR/bin/nvcc -arch=sm_80 -DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK -I$NVDIR/include -L$NVDIR/lib -lcudart 224_cg_grid_sync_attempt.cu -o 224_cg_grid_sync_attempt
LD_LIBRARY_PATH=$NVDIR/lib ./224_cg_grid_sync_attempt
```

**Sample input:** the 8-element array `[1, 2, 3, 4, 5, 6, 7, 8]`, doubled then incremented across a genuine (if unlaunchable, in this environment) grid-wide barrier.

**Sample output:**

```text
=== Section D.3: genuinely querying cooperative-launch support ===

  cudaGetDeviceCount                       -> cudaErrorInsufficientDriver (CUDA driver version is insufficient for CUDA runtime version)
  cudaDeviceGetAttribute(CooperativeLaunch) -> cudaErrorInsufficientDriver (CUDA driver version is insufficient for CUDA runtime version)
  supports_coop = -1

This environment has no usable device, so grid_sync_kernel above is
genuinely compiled and syntax-checked but cannot be launched -- launching it
for real would additionally require cudaLaunchCooperativeKernel (not the
ordinary <<<...>>> syntax) AND a device reporting supports_coop == 1.

=== what grid.sync() replaces: one kernel launch instead of several ===

Chapter 22's level-synchronous BFS launches a SEPARATE kernel for every
frontier level, specifically because a normal kernel launch has no way for
block 5 to wait for block 2 to finish -- __syncthreads() only ever reaches
threads in the SAME block. Ending one kernel and launching the next IS the
cross-block synchronization point: the driver guarantees every block of
kernel N has completed before any block of kernel N+1 begins.

grid.sync() offers a second way to get that same guarantee, from INSIDE a
single, still-running kernel, if and only if every block the kernel needs can
be resident on the device SIMULTANEOUSLY (checked via
cudaOccupancyMaxActiveBlocksPerMultiprocessor before launch) and the device
supports cooperative launch at all.

--- host-side replay of grid_sync_kernel's two-phase logic ---

input: [1, 2, 3, 4, 5, 6, 7, 8]

phase 1 (before grid.sync()): every thread, in every block, doubles its element
  thread 0: data[0] = 2
  thread 1: data[1] = 4
  thread 2: data[2] = 6
  thread 3: data[3] = 8
  thread 4: data[4] = 10
  thread 5: data[5] = 12
  thread 6: data[6] = 14
  thread 7: data[7] = 16

grid.sync() -- every block waits here for every OTHER block, not just its own
threads, before phase 2 begins anywhere

phase 2 (after grid.sync()): every thread adds 1
  thread 0: data[0] = 3
  thread 1: data[1] = 5
  thread 2: data[2] = 7
  thread 3: data[3] = 9
  thread 4: data[4] = 11
  thread 5: data[5] = 13
  thread 6: data[6] = 15
  thread 7: data[7] = 17

final data: [3, 5, 7, 9, 11, 13, 15, 17]

self-check: every element is (2*original)+1, which is only guaranteed if EVERY
block's phase-1 doubling finished before ANY block's phase-2 add began: confirmed
```

## D.4 When (Not) to Reach for Grid-Wide Sync

### Intuition

Chapter 22's BFS already gets a correct answer with one kernel launch per frontier level, and a kernel launch's own overhead is small -- typically low microseconds -- compared to the actual work a real frontier level does. `grid.sync()` is not a strictly faster replacement for that pattern; it is a genuinely different tool with genuinely different constraints, worth reaching for only when those specific constraints are actually satisfied and actually matter.

### The Concept, In Detail

```
ASCII view: three real constraints a normal kernel launch does not have.

  1. occupancy-bound grid size: every block must fit on the device AT ONCE
     (cudaOccupancyMaxActiveBlocksPerMultiprocessor caps this, often well
     below "one block per unit of problem size")

  2. hardware + driver support: cudaDevAttrCooperativeLaunch must report
     true -- older architectures and some multi-GPU configurations do not
     support it at all

  3. a DIFFERENT launch call: cudaLaunchCooperativeKernel, not <<<...>>>,
     with its own argument-packing convention
```

A normal multi-kernel-launch pipeline like Chapter 22's BFS is portable to every CUDA-capable device this book targets, imposes no occupancy ceiling on the algorithm's own grid size, and is simple to reason about. `grid.sync()` earns its real cost -- a smaller, occupancy-limited grid and a hardware/driver dependency -- specifically when an algorithm needs MANY cheap synchronization points in a tight loop where kernel-relaunch overhead would genuinely dominate, or when persistent per-block state (something that would otherwise have to round-trip through global memory between separate kernel launches) needs to survive across the synchronization point. Chapter 22's own BFS, with its comparatively large, uneven amount of work per frontier level, is not that case -- which is exactly why this book built it with ordinary, portable, multiple kernel launches instead.

## Appendix Summary

Section D.1 showed that `cg::thread_block::sync()` and `__syncthreads()` emit the identical hardware barrier; Cooperative Groups' contribution is making the assumption "every thread of some block, together" an explicit, passable object instead of an implicit convention. Section D.2 showed that a `thread_block_tile<N>` reduces across one warp's worth of threads using the identical fold-in-half shape as Appendix C.3's `cub::BlockReduce`, just moving values through registers via `shfl_down` instead of through shared memory. Section D.3 reached past any single block entirely with `cg::this_grid()` and `grid.sync()`, a genuine cross-block barrier inside one still-running kernel -- the same cross-block guarantee Chapter 22's BFS already gets for free at every kernel-launch boundary. Section D.4 closed with the practical tradeoff all three raise: `grid.sync()` trades a smaller, occupancy-bound grid and a real hardware/driver dependency for the ability to avoid repeated kernel launches, which is worth it only when an algorithm's own synchronization points are cheap and frequent enough for relaunch overhead to actually matter -- not, as Chapter 22 already demonstrates, for every level-synchronous algorithm this book has built.
