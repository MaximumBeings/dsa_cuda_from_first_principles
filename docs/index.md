# Data Structures and Algorithms in CUDA C++

*Parallel-First Data Structures, From Arrays to Concurrent Hash Tables*

A data structures and algorithms course usually starts from a sequential machine and asks "how fast is this operation." This book starts from a different machine — thousands of threads executing in lockstep groups of 32, with no cache coherence guarantee across most of them and no free way for one thread to know what another just wrote — and asks the same question again, honestly. Some familiar answers survive unchanged. Many do not. A linked list, the first data structure most people learn, turns out to be close to the worst possible structure to hand a GPU, for reasons this book proves rather than states. A sort that is optimal on one core is not the sort you want on ten thousand.

This book is independent: it does not assume or reference any other book in any series, and it builds its own CUDA foundations from Chapter 1 rather than pointing elsewhere for them. What it keeps is a discipline: every piece of code in this book is genuinely compiled, and wherever a real answer does not require a physical GPU to give one honestly, it is genuinely run. Where a claim depends on hardware this book's own authoring environment does not have — a real kernel launch, an actual timing number — that limitation is stated plainly rather than papered over with a plausible-looking, unverified number.

## What this book covers

- **Part 0 — GPU Foundations for Data Structures and Algorithms.** Why the SIMT execution model changes which data structures are viable at all, the CUDA memory and execution model built from scratch, and a genuine second axis of complexity — span, not just work — that a sequential Big-O analysis has no room for.
- **Part 1 — Parallel Primitives.** Reduction, scan (prefix sum), stream compaction, and histograms: the small set of parallel building blocks that nearly every data structure in the rest of this book is built out of.
- **Part 2 — Linear Structures.** Arrays and growable buffers under concurrent access, stacks and queues without a lock, and a full accounting of exactly why a pointer-chasing linked list is the wrong shape for this machine.
- **Part 3 — Sorting.** Bitonic sort as a sorting network built for warps, radix sort as the GPU's actual workhorse, and merge- and sample-based sorting for data too large to fit one block.
- **Part 4 — Trees.** Binary trees without recursion, parallel tree and trie construction, segment and Fenwick trees for range queries, and the spatial trees (k-d trees, quadtrees, octrees, bounding volume hierarchies) that make nearest-neighbor and collision queries tractable.
- **Part 5 — Hash Tables.** Open addressing under concurrent insertion, and cuckoo hashing's genuine, guaranteed O(1) worst-case lookup.
- **Part 6 — Graphs.** CSR and its alternatives, the frontier model that makes breadth-first search parallel, single-source shortest paths, connected components, and minimum spanning trees.
- **Part 7 — Priority Structures and Concurrency.** Heaps and priority queues on a GPU, lock-free and atomic-based concurrent structures, and the memory pools that back them.
- **Part 8 — Case Studies.** A GPU key-value store, spatial hashing for particle simulation, and parallel BVH construction for ray tracing — putting the whole book's toolbox to work on problems that need more than one structure at once.

## How to read this book

Every chapter follows the same shape: an intuition built from a concrete picture before any code, a background section with the real, complete implementation, one or more worked examples with genuinely computed output, a `[COMMON TRAP]` callout naming a specific, real mistake rather than a vague warning, a chapter summary, self-check questions, and worked solutions. Code that can genuinely run in this book's own authoring environment (a real `nvcc` and `g++` toolchain, no NVIDIA driver or physical device) is genuinely run and its exact output locked into the page. Code that needs an actual device to execute is genuinely compiled for a real architecture and checked instead by a host-side simulation of its own exact grid, block, and thread structure — Getting Started explains this discipline in full.
